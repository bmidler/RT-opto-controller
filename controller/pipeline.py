"""The real-time closed-loop orchestrator.

Hot path (one thread, hardware-paced by the camera trigger):
    grab -> preprocess -> classify (stateful GRU) -> stim decision ->
    hand the original frame + log row to the writer thread.

The writer (separate thread) does all ffmpeg encoding and CSV writing, so the
hot path never blocks on disk/encode. A latency monitor watches every frame.
"""

from __future__ import annotations

import bisect
import csv
import dataclasses
import json
import os
import time
from datetime import datetime

from .camera import make_sources
from .camera_trigger import CameraTrigger
from .classifier import Classifier
from .config import ControllerConfig
from .latency import LatencyMonitor
from .preview import Preview
from .stim import (StimArduino, StimController, build_activation_windows,
                   expand_pulses)
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

        # One frame source per physical camera; exactly one is the inference cam.
        self.cam_configs = config.per_camera_configs()
        self.sources, self.inf_idx, self.flir_system = make_sources(config)

        self.stim_arduino = StimArduino(config) if config.stim_enabled else None
        self.stim = StimController(config, self.stim_arduino)
        self.monitor = LatencyMonitor(config)
        self.preview = Preview(config)

        self.use_cam_trigger = (config.cam_trigger_enabled
                                and config.source == "flir")
        self.cam_trigger = CameraTrigger(config) if self.use_cam_trigger else None
        # One writer per camera; writers[inf_idx] records the inference camera.
        self.writers: list[VideoLogWriter] = []
        self.session_root = ""

        self._running = False
        self.frame_number = 0                       # inference-camera frames
        self.sec_counts = [0] * len(self.sources)   # per-camera written frames
        self.t_start = 0.0                          # first-frame arrival time
        self.t_last = 0.0                           # last-frame arrival time
        self.target_frames = 0                      # 0 => unbounded
        # (t_arrival_s, frame) for the inference camera, used to map each
        # reconstructed stim pulse onto the video frame showing at that instant.
        self.frame_times: list[tuple[float, int]] = []

    # -- setup --------------------------------------------------------------
    def setup(self) -> None:
        self.config.validate()

        # Open every source first so we know real frame sizes + colour order.
        for src in self.sources:
            src.open()
        inf = self.sources[self.inf_idx]
        h, w = inf.frame_height, inf.frame_width
        self.classifier.set_color_is_bgr(inf.color_is_bgr)

        mh, mw = self.classifier.model_input_size(h, w)
        print(f"[setup] device={self.classifier.device} "
              f"num_classes={self.classifier.num_classes}")
        cam_list = ", ".join(
            f"{c.camera_name}{' (inference)' if i == self.inf_idx else ''}"
            for i, c in enumerate(self.cam_configs))
        print(f"[setup] cameras: {cam_list}")
        print(f"[setup] inference camera frame {h}H x {w}W  ->  "
              f"x{self.classifier.spatial_scale}  ->  model input {mh}H x {mw}W")
        print("[setup] NOTE: the model input size must match the resolution the "
              "model was TRAINED on. Confirm this matches your training videos.")
        print(f"[setup] trigger_classes={sorted(self.stim.trigger_set)} "
              f"stim_enabled={self.config.stim_enabled} "
              f"dry_run={self.config.serial_dry_run}")

        # Prime CUDA kernels so the first real frame isn't an outlier.
        self.classifier.warmup(h, w)

        # One session folder shared by all cameras' subfolders. Its name comes
        # from config.session_name; if unset, fall back to a timestamped name.
        stem, _ = os.path.splitext(self.config.video_filename)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        session_dirname = self.config.session_name.strip() or f"{ts}_{stem}"
        self.session_root = os.path.join(self.config.output_folder, session_dirname)

        # One writer per camera (each knows its own frame size + pixel format).
        for src, cfg in zip(self.sources, self.cam_configs):
            writer = VideoLogWriter(cfg, src.frame_width, src.frame_height,
                                    _pix_fmt_in(src))
            writer.open(session_root=self.session_root, ts=ts)
            writer.start()
            self.writers.append(writer)

        # Stim board: configure before any frames arrive.
        if self.stim_arduino is not None:
            self.stim_arduino.open()

        # Live preview (daemon thread; decoupled from the hot path). Colour info
        # is the inference camera's; secondary feeds carry their own per-frame.
        self.preview.open(inf.is_color, inf.color_is_bgr)

        # Early metadata.json snapshot: records the recording/stim/camera info
        # that is already known (frame sizes, serials, model, stim protocol) so a
        # mid-run hard crash still leaves a metadata file on disk. Overwritten
        # with the final counts + timing at shutdown.
        self._write_metadata_json(tag="setup")

        # Start the camera-trigger Teensy LAST: the cameras are already grabbing,
        # so they're armed and waiting for the hardware pulses that pace the loop.
        if self.cam_trigger is not None:
            self.cam_trigger.start()

    # -- main loop ----------------------------------------------------------
    def run(self) -> None:
        self._running = True
        self.t_start = 0.0          # set when the FIRST frame actually arrives
        last_arrival = None
        class_names = self.config.class_names
        finite_source = self.config.source in ("video", "synthetic")

        inf_idx = self.inf_idx
        inf_source = self.sources[inf_idx]
        inf_writer = self.writers[inf_idx]
        sec_indices = [i for i in range(len(self.sources)) if i != inf_idx]

        # How many frames to record. rec_time_sec is converted to a frame count
        # so BOTH modes stop on the same deterministic counter (the hardware
        # trigger fixes the frame rate). 0 => run until the source ends / Ctrl-C.
        # Counting only frames that actually arrive means startup lag before the
        # first trigger pulse never eats into the requested total.
        if self.config.max_frames > 0:
            target_frames = int(self.config.max_frames)
        elif self.config.rec_time_sec > 0:
            target_frames = int(round(self.config.frame_rate
                                      * self.config.rec_time_sec))
        else:
            target_frames = 0
        self.target_frames = target_frames
        # Backstop: once bounded, don't spin forever if the trigger stops early.
        max_empty_grabs = 10
        empty_grabs = 0

        try:
            while self._running:
                if target_frames > 0 and self.frame_number >= target_frames:
                    break

                # --- Inference camera: the closed-loop hot path ---------------
                grabbed = inf_source.grab()
                if grabbed is None:
                    if finite_source:
                        break               # end of video / generator
                    empty_grabs += 1
                    if target_frames > 0 and empty_grabs >= max_empty_grabs:
                        print(f"\n[run] no frame for {empty_grabs} grab timeouts "
                              f"-- trigger stopped? stopping at "
                              f"{self.frame_number}/{target_frames} frames.")
                        break
                    continue                # FLIR: waiting on next trigger
                empty_grabs = 0
                frame, cam_ts = grabbed

                arrival = time.perf_counter()
                if self.t_start == 0.0:     # first real frame defines t = 0
                    self.t_start = arrival
                self.t_last = arrival
                interval_ms = ((arrival - last_arrival) * 1000.0
                               if last_arrival is not None else None)
                last_arrival = arrival

                gray = self.classifier.preprocess(frame)
                t_pre = time.perf_counter()
                pred = self.classifier.step(gray)
                t_inf = time.perf_counter()

                stim_on = self.stim.update(pred.pred_class, now=t_inf,
                                           frame=self.frame_number)

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
                inf_writer.submit(frame, row)
                self.frame_times.append((row["t_arrival_s"], self.frame_number))
                self.sec_counts[inf_idx] += 1

                # --- Secondary cameras: record only, no inference -------------
                # Done after the stim decision so the closed-loop latency above
                # is unchanged. They share the trigger pulse, so their frames are
                # already buffered and grab() returns promptly.
                sec_frames = []
                for i in sec_indices:
                    g = self.sources[i].grab()
                    if g is None:
                        continue            # missed/incomplete frame this tick
                    sframe, sts = g
                    self.writers[i].submit(sframe, {
                        "frame": self.sec_counts[i],
                        "t_capture_s": round(sts, 6),
                        "t_arrival_s": round(time.perf_counter() - self.t_start, 6),
                        "frame_written": True,
                    })
                    self.sec_counts[i] += 1
                    sec_frames.append((self.cam_configs[i].camera_name, sframe,
                                       self.sources[i].color_is_bgr))

                self.preview.submit(frame, pred.pred_class, cname,
                                    pred.confidence, stim_on, self.frame_number,
                                    secondary=sec_frames)
                self.monitor.record(
                    interval_ms=interval_ms, preprocess_ms=preprocess_ms,
                    inference_ms=inference_ms, hotpath_ms=hotpath_ms,
                    queue_depth=inf_writer.queue_depth())

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
        """Tear down in an order that persists ALL data BEFORE touching PySpin.

        The FLIR/PySpin camera + system teardown (src.close / flir_system.close)
        can abort the whole PROCESS at the C++ level -- e.g. System.ReleaseInstance()
        with camera references still active -- which Python try/except CANNOT
        catch and which leaves no traceback. If that teardown ran before the
        metadata/stim CSVs were written, those files would silently never appear
        (the exact symptom we saw on real cameras: classifications.csv present,
        every shutdown-written file missing, zero "[shutdown]" prints).

        So the order is: laser OFF (safety, always first) -> stop new frames ->
        finalize the stim pulse log -> flush writers -> WRITE ALL METADATA + STIM
        CSVS -> only THEN close the cameras and release the PySpin system.
        """
        self._running = False
        # 1) Laser off first, always. Pass the current time/frame so a still-open
        #    activation window is closed in the stim event log.
        try:
            self.stim.shutdown(now=time.perf_counter(), frame=self.frame_number)
        except Exception as e:
            print(f"[shutdown] stim off failed: {e}")
        # 2) Stop the hardware trigger so no new frames arrive.
        if self.cam_trigger is not None:
            try:
                self.cam_trigger.stop()
            except Exception as e:
                print(f"[shutdown] cam-trigger stop failed: {e}")
        # 3) Close the stim board (stops its reader thread, finalizing the
        #    measured pulse_marks the stim log needs).
        if self.stim_arduino is not None:
            try:
                self.stim_arduino.close()
            except Exception as e:
                print(f"[shutdown] stim close failed: {e}")
        # 4) Flush every writer -> classifications.csv complete + final counts.
        for writer in self.writers:
            try:
                writer.stop()
            except Exception as e:
                print(f"[shutdown] writer stop failed: {e}")
        # 5) Timing summary (cheap, pure-Python).
        try:
            self.monitor.print_summary()
        except Exception as e:
            print(f"[shutdown] timing summary failed: {e}")
        # 6) PERSIST EVERYTHING. This MUST happen before the PySpin teardown
        #    below, which may hard-crash the process. Reads frame size / serial /
        #    model off the still-open source objects (plain attributes set during
        #    open(), valid until close()).
        try:
            self._write_metadata()
        except Exception as e:
            print(f"[shutdown] metadata write failed: {e}")
        # 7) Preview window down (daemon thread; bounded join).
        try:
            self.preview.close()
        except Exception as e:
            print(f"[shutdown] preview close failed: {e}")
        # 8) ONLY NOW tear down the cameras. A C-level abort here can no longer
        #    cost us any data -- it is already on disk.
        for src in self.sources:
            try:
                src.close()
            except Exception as e:
                print(f"[shutdown] source close failed: {e}")
        # 9) Shared PySpin system (multi-camera FLIR) released once, last.
        if self.flir_system is not None:
            try:
                self.flir_system.close()
            except Exception as e:
                print(f"[shutdown] flir system close failed: {e}")

    def _metadata_dict(self) -> dict:
        """Build the metadata.json payload from current state.

        Safe to call at setup (counts/timing are zero then) or at shutdown.
        Each camera entry is built defensively so one bad source can't sink the
        rest.
        """
        cameras = []
        for i, (src, cfg, w) in enumerate(
                zip(self.sources, self.cam_configs, self.writers)):
            try:
                cameras.append({
                    "camera_name": cfg.camera_name,
                    "is_inference": i == self.inf_idx,
                    "camera_serial": src.camera_serial or cfg.camera_serial,
                    "camera_model": getattr(src, "camera_model", ""),
                    "frame_width": src.frame_width,
                    "frame_height": src.frame_height,
                    "frames_written": w.n_written,
                    "frames_dropped": w.n_dropped,
                    "peak_writer_queue": w.max_queue_depth,
                    "session_dir": w.session_dir,
                })
            except Exception as e:
                print(f"[meta] camera entry for {cfg.camera_name} failed: {e}")
        return {
            "config": dataclasses.asdict(self.config),
            "num_classes": self.classifier.num_classes,
            "spatial_scale": self.classifier.spatial_scale,
            "device": str(self.classifier.device),
            "inference_camera": self.cam_configs[self.inf_idx].camera_name,
            "frames_processed": self.frame_number,
            "stim_activations": self.stim.n_activations,
            "stim_events_logged": len(self.stim.events),
            "stim_pulses_measured": (len(self.stim_arduino.pulse_marks)
                                     if self.stim_arduino is not None else 0),
            "stim_board_resets": (self.stim_arduino.n_board_resets
                                  if self.stim_arduino is not None else 0),
            "stim_config_errors": (self.stim_arduino.n_config_errors
                                   if self.stim_arduino is not None else 0),
            "stim_onsets_suppressed": self.stim.n_onsets_suppressed,
            "cameras": cameras,
            "timing": self.monitor.summary(),
        }

    def _write_metadata_json(self, tag: str = "shutdown") -> None:
        """Write (or overwrite) metadata.json. Called once early at setup as a
        crash-resilient snapshot, then again at shutdown with the final counts."""
        if not self.session_root or not self.writers:
            return
        try:
            path = os.path.join(self.session_root, "metadata.json")
            with open(path, "w") as f:
                json.dump(self._metadata_dict(), f, indent=2, default=str)
            print(f"[{tag}] metadata -> {path}")
        except Exception as e:
            print(f"[{tag}] metadata.json write failed: {e}")

    def _write_metadata(self) -> None:
        if not self.session_root or not self.writers:
            return
        self._write_metadata_json(tag="shutdown")

        # Per-camera metadata CSV inside each camera's video folder. Each camera
        # is independent so one failure can't skip the others -- or the stim
        # logs that follow.
        for i, (src, cfg, w) in enumerate(
                zip(self.sources, self.cam_configs, self.writers)):
            try:
                self._write_camera_metadata_csv(i, src, cfg, w)
            except Exception as e:
                print(f"[shutdown] camera metadata failed ({cfg.camera_name}): {e}")

        # Stimulation logs (event windows + measured/reconstructed pulses) next
        # to the inference camera's classifications.csv.
        try:
            self._write_stim_logs()
        except Exception as e:
            print(f"[shutdown] stim logs failed: {e}")

        # Everything the stim board said that wasn't a pulse marker: boot
        # banners, CONFIG OK/ERR, watchdog trips. These make a mid-session board
        # reset visible in the session record instead of silently discarded.
        try:
            self._write_stim_board_log()
        except Exception as e:
            print(f"[shutdown] stim board log failed: {e}")

    def _write_camera_metadata_csv(self, i, src, cfg, writer) -> None:
        """Write a key,value metadata.csv into one camera's video folder.

        Building `rows` dereferences many camera/config attributes; if any of
        those raised it would previously abort the whole _write_metadata pass
        (taking the stim logs down with it), so the build is inside the try.
        """
        if not writer.session_dir:
            return
        path = os.path.join(writer.session_dir, "metadata.csv")
        try:
            actual_duration_s = (round(self.t_last - self.t_start, 6)
                                 if self.t_last > self.t_start else 0.0)
            rows = [
                # --- session / save locations -------------------------------------
                ("session_name", os.path.basename(self.session_root)),
                ("output_folder", self.config.output_folder),
                ("session_root", self.session_root),
                ("camera_dir", writer.session_dir),
                ("video_path", writer.video_path),
                ("log_path", writer.log_path),
                # --- recording arguments ------------------------------------------
                ("source", self.config.source),
                ("frame_rate_hz", self.config.frame_rate),
                ("rec_time_sec_requested", self.config.rec_time_sec),
                ("max_frames_requested", self.config.max_frames),
                ("target_frames", self.target_frames),
                ("frames_processed_inference", self.frame_number),
                ("frames_written", writer.n_written),
                ("frames_dropped", writer.n_dropped),
                ("peak_writer_queue", writer.max_queue_depth),
                ("actual_duration_sec", actual_duration_s),
                # --- this camera --------------------------------------------------
                ("camera_name", cfg.camera_name),
                ("is_inference_camera", i == self.inf_idx),
                ("camera_serial", src.camera_serial or cfg.camera_serial),
                ("camera_model", getattr(src, "camera_model", "")),
                ("camera_selection", cfg.camera_selection),
                ("frame_width", src.frame_width),
                ("frame_height", src.frame_height),
                ("pixel_format_input", cfg.pixel_format_input),
                ("camera_trigger", cfg.camera_trigger),
                ("camera_exposure_us", cfg.camera_exposure_us),
                ("camera_gain", cfg.camera_gain),
                ("codec", self.config.codec),
                ("quality", self.config.quality),
                # --- model --------------------------------------------------------
                ("model_checkpoint", self.config.model_checkpoint),
                ("num_classes", self.classifier.num_classes),
                ("spatial_scale", self.classifier.spatial_scale),
                ("device", str(self.classifier.device)),
                # --- stim parameters ----------------------------------------------
                ("stim_enabled", self.config.stim_enabled),
                ("stim_serial_port", self.config.stim_serial_port),
                ("stim_pin", self.config.stim_pin),
                ("pulse_width_us", self.config.pulse_width_us),
                ("frequency_hz", self.config.frequency_hz),
                ("max_pulses", self.config.max_pulses),
                ("watchdog_ms", self.config.watchdog_ms),
                ("keepalive_ms", self.config.keepalive_ms),
                ("trigger_classes", ";".join(str(x) for x in self.config.trigger_classes)),
                ("onset_frames", self.config.onset_frames),
                ("offset_frames", self.config.offset_frames),
                ("stim_duration_sec", self.config.stim_duration_sec),
                ("stim_refractory_sec", self.config.stim_refractory_sec),
                ("stim_activations", self.stim.n_activations),
                ("stim_onsets_suppressed", self.stim.n_onsets_suppressed),
                ("stim_board_resets", (self.stim_arduino.n_board_resets
                                       if self.stim_arduino is not None else 0)),
                ("stim_config_errors", (self.stim_arduino.n_config_errors
                                        if self.stim_arduino is not None else 0)),
                ("serial_dry_run", self.config.serial_dry_run),
            ]
            with open(path, "w", newline="") as f:
                wr = csv.writer(f)
                wr.writerow(["parameter", "value"])
                wr.writerows(rows)
            print(f"[shutdown] camera metadata -> {path}")
        except Exception as e:
            print(f"[shutdown] camera metadata write failed ({cfg.camera_name}): {e}")

    def _frame_for_time(self, t_s: float):
        """Video frame showing at session-time `t_s` (the last frame whose
        arrival was <= t_s). Returns "" if no frames were recorded."""
        if not self.frame_times:
            return ""
        times = [t for t, _ in self.frame_times]
        idx = bisect.bisect_right(times, t_s) - 1
        if idx < 0:
            idx = 0
        return self.frame_times[idx][1]

    @staticmethod
    def _activation_index_for_time(t_s: float, windows):
        """Index of the activation window a pulse at `t_s` belongs to.

        Prefers the window that contains the time; otherwise the most recent
        window that had already opened (covers small edge skew between the PC
        marker arrival and the commanded STOP). Returns "" if none qualifies.
        """
        last = ""
        for wi, w in enumerate(windows):
            if w["onset_t_s"] <= t_s:
                last = wi
            if w["onset_t_s"] <= t_s <= w["offset_t_s"]:
                return wi
        return last

    def _write_stim_logs(self) -> None:
        """Write stim_events.csv (activation windows) and stim_pulses.csv
        (individual laser pulses) beside classifications.csv.

        Pulse times are the ACTUAL rising edges the stim Arduino reports back
        over USB ("P,<index>,<micros>"), timestamped on arrival against the PC
        clock -- the same clock as classifications.csv's t_arrival_s -- so each
        pulse maps directly to the video frame on screen at its onset.

        Only when no markers were received (e.g. --dry-run, or stim hardware
        absent) does it fall back to reconstructing pulses from the commanded
        START/STOP windows + configured frequency; such rows are flagged
        reconstructed=True so they're never mistaken for measured times.
        """
        if not self.writers:
            return
        # Stim is a single session-level signal; write it alongside the
        # inference camera's per-frame log.
        out_dir = self.writers[self.inf_idx].session_dir
        if not out_dir:
            return

        t_end = (self.t_last - self.t_start) if self.t_last > self.t_start else 0.0
        windows = build_activation_windows(self.stim.events, self.t_start, t_end)
        pulse_width_ms = round(self.config.pulse_width_us / 1000.0, 4)
        pulse_width_s = self.config.pulse_width_us / 1e6

        # Prefer measured pulses from the board; reconstruct only as a fallback.
        marks = (self.stim_arduino.pulse_marks
                 if self.stim_arduino is not None else [])
        if marks:
            reconstructed = False
            pulses = []
            for m in marks:
                onset = m["t"] - self.t_start
                pulses.append({
                    "activation_index": self._activation_index_for_time(
                        onset, windows),
                    "pulse_in_activation": (m["board_index"]
                                            if m["board_index"] >= 0 else ""),
                    "t_onset_s": onset,
                    "t_offset_s": onset + pulse_width_s,
                    "board_us": m["board_us"] if m["board_us"] >= 0 else "",
                })
        else:
            reconstructed = True
            pulses = expand_pulses(windows, self.config.frequency_hz,
                                   self.config.pulse_width_us,
                                   self.config.max_pulses)
            for p in pulses:
                p["board_us"] = ""

        # Pulses per activation window (skip any pulse we couldn't place).
        pulses_per_window = [0] * len(windows)
        for p in pulses:
            ai = p["activation_index"]
            if isinstance(ai, int) and 0 <= ai < len(windows):
                pulses_per_window[ai] += 1

        events_path = os.path.join(out_dir, "stim_events.csv")
        pulses_path = os.path.join(out_dir, "stim_pulses.csv")
        try:
            with open(events_path, "w", newline="") as f:
                wr = csv.writer(f)
                wr.writerow(["activation_index", "onset_frame", "onset_t_s",
                             "offset_frame", "offset_t_s", "duration_s",
                             "n_pulses"])
                for wi, w in enumerate(windows):
                    dur = w["offset_t_s"] - w["onset_t_s"]
                    wr.writerow([
                        wi,
                        "" if w["onset_frame"] is None else w["onset_frame"],
                        round(w["onset_t_s"], 6),
                        "" if w["offset_frame"] is None else w["offset_frame"],
                        round(w["offset_t_s"], 6),
                        round(dur, 6),
                        pulses_per_window[wi],
                    ])
            print(f"[shutdown] stim events -> {events_path} "
                  f"({len(windows)} activations)")
        except Exception as e:
            print(f"[shutdown] stim_events.csv write failed: {e}")

        try:
            with open(pulses_path, "w", newline="") as f:
                wr = csv.writer(f)
                wr.writerow(["pulse_index", "activation_index",
                             "pulse_in_activation", "t_onset_s", "t_offset_s",
                             "pulse_width_ms", "frequency_hz", "video_frame",
                             "reconstructed", "board_us"])
                for pi, p in enumerate(pulses):
                    wr.writerow([
                        pi,
                        p["activation_index"],
                        p["pulse_in_activation"],
                        round(p["t_onset_s"], 6),
                        round(p["t_offset_s"], 6),
                        pulse_width_ms,
                        self.config.frequency_hz,
                        self._frame_for_time(p["t_onset_s"]),
                        reconstructed,
                        p["board_us"],
                    ])
            kind = "reconstructed" if reconstructed else "measured"
            print(f"[shutdown] stim pulses -> {pulses_path} "
                  f"({len(pulses)} {kind} pulses)")
        except Exception as e:
            print(f"[shutdown] stim_pulses.csv write failed: {e}")

    def _write_stim_board_log(self) -> None:
        """Write stim_board_log.csv: the board's own non-pulse output.

        A board reset, a rejected config or a watchdog trip is the difference
        between "the laser did what we asked" and "the laser did something
        else", so it belongs in the session record next to the pulse log.
        """
        if self.stim_arduino is None or not self.writers:
            return
        messages = self.stim_arduino.board_messages
        out_dir = self.writers[self.inf_idx].session_dir
        if not out_dir:
            return
        path = os.path.join(out_dir, "stim_board_log.csv")
        with open(path, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["t_s", "message"])
            for m in messages:
                ts = (m["t"] - self.t_start) if self.t_start else 0.0
                wr.writerow([round(ts, 6), m["text"]])
        note = ""
        if self.stim_arduino.n_board_resets:
            note = (f"  *** {self.stim_arduino.n_board_resets} BOARD RESET(S) "
                    f"during this session ***")
        print(f"[shutdown] stim board log -> {path} "
              f"({len(messages)} messages){note}")
