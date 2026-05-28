"""The real-time closed-loop orchestrator.

Hot path (one thread, hardware-paced by the camera trigger):
    grab -> preprocess -> classify (stateful GRU) -> stim decision ->
    hand the original frame + log row to the writer thread.

The writer (separate thread) does all ffmpeg encoding and CSV writing, so the
hot path never blocks on disk/encode. A latency monitor watches every frame.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time

from .camera import make_source
from .camera_trigger import CameraTrigger
from .classifier import Classifier
from .config import ControllerConfig
from .latency import LatencyMonitor
from .preview import Preview
from .stim import StimArduino, StimController
from .writer import VideoLogWriter


def _pix_fmt_in(source) -> str:
    if not source.is_color:
        return "gray"
    return "bgr24" if source.color_is_bgr else "rgb24"


class RealTimeController:
    def __init__(self, config: ControllerConfig):
        self.config = config
        self.classifier = Classifier(
            config.model_checkpoint, device=config.device,
            spatial_scale=config.spatial_scale)
        self.source = make_source(config)
        self.stim_arduino = StimArduino(config) if config.stim_enabled else None
        self.stim = StimController(config, self.stim_arduino)
        self.monitor = LatencyMonitor(config)
        self.preview = Preview(config)

        self.use_cam_trigger = (config.cam_trigger_enabled
                                and config.source == "flir")
        self.cam_trigger = CameraTrigger(config) if self.use_cam_trigger else None
        self.writer: VideoLogWriter | None = None

        self._running = False
        self.frame_number = 0
        self.t_start = 0.0

    # -- setup --------------------------------------------------------------
    def setup(self) -> None:
        self.config.validate()

        # Open the source first so we know the real frame size + colour order.
        self.source.open()
        h, w = self.source.frame_height, self.source.frame_width
        self.classifier.set_color_is_bgr(self.source.color_is_bgr)

        mh, mw = self.classifier.model_input_size(h, w)
        print(f"[setup] device={self.classifier.device} "
              f"num_classes={self.classifier.num_classes}")
        print(f"[setup] camera frame {h}H x {w}W  ->  x{self.classifier.spatial_scale}  "
              f"->  model input {mh}H x {mw}W")
        print("[setup] NOTE: the model input size must match the resolution the "
              "model was TRAINED on. Confirm this matches your training videos.")
        print(f"[setup] trigger_classes={sorted(self.stim.trigger_set)} "
              f"stim_enabled={self.config.stim_enabled} "
              f"dry_run={self.config.serial_dry_run}")

        # Prime CUDA kernels so the first real frame isn't an outlier.
        self.classifier.warmup(h, w)

        # Writer (knows frame size + pixel format now).
        self.writer = VideoLogWriter(self.config, w, h, _pix_fmt_in(self.source))
        self.writer.open()
        self.writer.start()

        # Stim board: configure before any frames arrive.
        if self.stim_arduino is not None:
            self.stim_arduino.open()

        # Live preview (daemon thread; decoupled from the hot path).
        self.preview.open(self.source.is_color, self.source.color_is_bgr)

        # Start the camera-trigger Teensy LAST: the camera is already grabbing,
        # so it's armed and waiting for the hardware pulses that pace the loop.
        if self.cam_trigger is not None:
            self.cam_trigger.start()

    # -- main loop ----------------------------------------------------------
    def run(self) -> None:
        self._running = True
        self.t_start = time.perf_counter()
        last_arrival = None
        class_names = self.config.class_names
        finite_source = self.config.source in ("video", "synthetic")

        try:
            while self._running:
                if (self.config.max_frames > 0
                        and self.frame_number >= self.config.max_frames):
                    break
                if (self.config.rec_time_sec > 0
                        and time.perf_counter() - self.t_start >= self.config.rec_time_sec):
                    break

                grabbed = self.source.grab()
                if grabbed is None:
                    if finite_source:
                        break               # end of video / generator
                    continue                # FLIR: waiting on next trigger
                frame, cam_ts = grabbed

                arrival = time.perf_counter()
                interval_ms = ((arrival - last_arrival) * 1000.0
                               if last_arrival is not None else None)
                last_arrival = arrival

                gray = self.classifier.preprocess(frame)
                t_pre = time.perf_counter()
                pred = self.classifier.step(gray)
                t_inf = time.perf_counter()

                stim_on = self.stim.update(pred.pred_class, now=t_inf)

                preprocess_ms = (t_pre - arrival) * 1000.0
                inference_ms = (t_inf - t_pre) * 1000.0
                hotpath_ms = (time.perf_counter() - arrival) * 1000.0

                cname = (class_names[pred.pred_class]
                         if 0 <= pred.pred_class < len(class_names) else "")
                row = {
                    "frame": self.frame_number,
                    "t_capture_s": round(cam_ts, 6),
                    "t_arrival_s": round(arrival - self.t_start, 6),
                    "pred_class": pred.pred_class,
                    "class_name": cname,
                    "confidence": round(pred.confidence, 4),
                    "stim_on": int(stim_on),
                    "frame_written": True,   # writer.submit overwrites if dropped
                    "preprocess_ms": round(preprocess_ms, 3),
                    "inference_ms": round(inference_ms, 3),
                    "hotpath_ms": round(hotpath_ms, 3),
                    "interval_ms": (round(interval_ms, 3)
                                    if interval_ms is not None else ""),
                }
                self.writer.submit(frame, row)
                self.preview.submit(frame, pred.pred_class, cname,
                                    pred.confidence, stim_on, self.frame_number)
                self.monitor.record(
                    interval_ms=interval_ms, preprocess_ms=preprocess_ms,
                    inference_ms=inference_ms, hotpath_ms=hotpath_ms,
                    queue_depth=self.writer.queue_depth())

                self.frame_number += 1
                if self.preview.quit_requested:
                    print("\n[run] preview window closed (q/Esc) -- stopping.")
                    break

        except KeyboardInterrupt:
            print("\n[run] interrupted by user -- shutting down cleanly.")
        finally:
            self.shutdown()

    # -- shutdown -----------------------------------------------------------
    def shutdown(self) -> None:
        self._running = False
        # Laser off first, always.
        try:
            self.stim.shutdown()
        except Exception as e:
            print(f"[shutdown] stim off failed: {e}")
        if self.cam_trigger is not None:
            try:
                self.cam_trigger.stop()
            except Exception as e:
                print(f"[shutdown] cam-trigger stop failed: {e}")
        try:
            self.source.close()
        except Exception as e:
            print(f"[shutdown] source close failed: {e}")
        if self.stim_arduino is not None:
            try:
                self.stim_arduino.close()
            except Exception as e:
                print(f"[shutdown] stim close failed: {e}")
        if self.writer is not None:
            self.writer.stop()
        try:
            self.preview.close()
        except Exception as e:
            print(f"[shutdown] preview close failed: {e}")

        self.monitor.print_summary()
        self._write_metadata()

    def _write_metadata(self) -> None:
        if self.writer is None or not self.writer.session_dir:
            return
        meta = {
            "config": dataclasses.asdict(self.config),
            "num_classes": self.classifier.num_classes,
            "spatial_scale": self.classifier.spatial_scale,
            "device": str(self.classifier.device),
            "frames_processed": self.frame_number,
            "frames_written": self.writer.n_written,
            "frames_dropped": self.writer.n_dropped,
            "peak_writer_queue": self.writer.max_queue_depth,
            "stim_activations": self.stim.n_activations,
            "timing": self.monitor.summary(),
        }
        path = os.path.join(self.writer.session_dir, "metadata.json")
        with open(path, "w") as f:
            json.dump(meta, f, indent=2, default=str)
        print(f"[shutdown] metadata -> {path}")
