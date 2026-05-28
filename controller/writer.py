"""Decoupled video writer + per-frame log.

Runs ffmpeg encoding in its own thread so a slow encoder/disk can never stall
the real-time classify-and-stim hot path. Two streams:

  * VIDEO  -- the original full-resolution frames, encoded via imageio-ffmpeg
              (same codec machinery as campy). Bounded queue; under sustained
              overload the *current* frame may be dropped (and recorded as
              frame_written=False) so memory stays bounded and the loop keeps
              its real-time cadence. The log remains the source of truth.
  * LOG    -- one CSV row per frame (classification + stim state + timing).
              Never dropped (rows are tiny and queued unbounded).

Because every frame's log row records its frame number and whether it made it
into the video, video<->log alignment is always reconstructable even if some
video frames were dropped.
"""

from __future__ import annotations

import csv
import os
import threading
import time
from collections import deque
from datetime import datetime

from imageio_ffmpeg import write_frames

from .config import ControllerConfig

LOG_FIELDS = [
    "frame", "t_capture_s", "t_arrival_s",
    "pred_class", "class_name", "confidence",
    "stim_on", "frame_written",
    "preprocess_ms", "inference_ms", "hotpath_ms", "interval_ms",
]


def build_ffmpeg_params(config: ControllerConfig, pix_fmt_in: str):
    """Return (codec, pix_fmt_out, output_params). Mirrors campy/writer.py."""
    pix_fmt_out = config.pixel_format_output
    codec = config.codec
    quality = str(config.quality)
    preset = config.preset if config.preset != "None" else "fast"

    if config.gpu_id == -1:  # CPU
        output_params = ["-preset", preset, "-tune", "fastdecode",
                         "-crf", quality, "-bufsize", "20M", "-maxrate", "10M",
                         "-bf:v", "4"]
        if pix_fmt_out in ("rgb0", "bgr0"):
            pix_fmt_out = "yuv420p"
        if config.codec == "h264":
            codec = "libx264"
            output_params += ["-x264-params", "nal-hrd=cbr"]
        elif config.codec == "h265":
            codec = "libx265"
    else:  # GPU
        if config.gpu_make == "nvidia":
            output_params = ["-preset", preset, "-qp", quality, "-bf:v", "0",
                             "-gpu", str(config.gpu_id), "-vsync", "0"]
            codec = "h264_nvenc" if config.codec == "h264" else "hevc_nvenc"
        elif config.gpu_make == "amd":
            output_params = ["-usage", "lowlatency", "-rc", "cqp",
                             "-qp_i", quality, "-qp_p", quality, "-qp_b", quality,
                             "-bf:v", "0", "-hwaccel_device", str(config.gpu_id)]
            if pix_fmt_out in ("rgb0", "bgr0"):
                pix_fmt_out = "yuv420p"
            codec = "h264_amf" if config.codec == "h264" else "hevc_amf"
        elif config.gpu_make == "intel":
            output_params = ["-bf:v", "0", "-preset", preset,
                             "-q", str(int(config.quality) + 1)]
            if pix_fmt_out in ("rgb0", "bgr0"):
                pix_fmt_out = "nv12"
            codec = "h264_qsv" if config.codec == "h264" else "hevc_qsv"
        else:
            raise ValueError(f"Unknown gpu_make {config.gpu_make!r}")
    return codec, pix_fmt_out, output_params


class VideoLogWriter:
    def __init__(self, config: ControllerConfig, frame_w: int, frame_h: int,
                 pix_fmt_in: str):
        self.config = config
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.pix_fmt_in = pix_fmt_in

        self._video_q: deque = deque()
        self._log_q: deque = deque()
        self._maxsize = max(8, int(config.write_queue_maxsize))
        self._drop = config.drop_video_when_full

        self._writer = None
        self._csv_file = None
        self._csv_writer = None
        self._thread = None
        self._stop = threading.Event()

        self.session_dir = ""
        self.video_path = ""
        self.log_path = ""

        # stats
        self.n_written = 0
        self.n_dropped = 0
        self.max_queue_depth = 0

    # -- lifecycle ----------------------------------------------------------
    def open(self) -> None:
        stem, ext = os.path.splitext(self.config.video_filename)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.session_dir = os.path.join(self.config.output_folder,
                                        f"{ts}_{stem}", self.config.camera_name)
        os.makedirs(self.session_dir, exist_ok=True)

        self.video_path = os.path.join(self.session_dir, f"{ts}-{stem}{ext}")
        self.log_path = os.path.join(self.session_dir, "classifications.csv")

        codec, pix_fmt_out, output_params = build_ffmpeg_params(
            self.config, self.pix_fmt_in)
        self._writer = write_frames(
            self.video_path,
            [self.frame_w, self.frame_h],
            fps=self.config.frame_rate,
            quality=None,
            bitrate=None,
            codec=codec,
            pix_fmt_in=self.pix_fmt_in,
            pix_fmt_out=pix_fmt_out,
            ffmpeg_log_level=self.config.ffmpeg_log_level,
            input_params=["-an"],
            output_params=output_params,
        )
        self._writer.send(None)  # prime the generator

        self._csv_file = open(self.log_path, "w", newline="")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=LOG_FIELDS)
        self._csv_writer.writeheader()
        print(f"[writer] video -> {self.video_path}")
        print(f"[writer] log   -> {self.log_path}")

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="writer",
                                        daemon=True)
        self._thread.start()

    # -- producer-facing ----------------------------------------------------
    def submit(self, frame, log_row: dict) -> bool:
        """Queue one frame + its log row. Returns whether the video frame was
        accepted (False => dropped under backpressure)."""
        accepted = True
        depth = len(self._video_q)
        if depth >= self._maxsize:
            if self._drop:
                accepted = False
                self.n_dropped += 1
            else:
                # Backpressure: wait for the encoder to catch up. Preserves the
                # video timeline at the cost of pausing the producer.
                while len(self._video_q) >= self._maxsize and not self._stop.is_set():
                    time.sleep(0.001)
        log_row["frame_written"] = accepted
        if accepted:
            self._video_q.append(frame)
        self._log_q.append(dict(log_row))
        if depth > self.max_queue_depth:
            self.max_queue_depth = depth
        return accepted

    def queue_depth(self) -> int:
        return len(self._video_q)

    # -- worker thread ------------------------------------------------------
    def _run(self) -> None:
        while not self._stop.is_set() or self._video_q or self._log_q:
            did_work = False
            if self._video_q:
                frame = self._video_q.popleft()
                try:
                    self._writer.send(frame)
                    self.n_written += 1
                except Exception as e:
                    print(f"[writer] ffmpeg send failed: {e}")
                did_work = True
            if self._log_q:
                row = self._log_q.popleft()
                self._csv_writer.writerow(row)
                did_work = True
            if not did_work:
                time.sleep(0.002)
        self._csv_file.flush()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=30)
        try:
            if self._writer is not None:
                self._writer.close()
        finally:
            if self._csv_file is not None:
                self._csv_file.flush()
                self._csv_file.close()
        print(f"[writer] wrote {self.n_written} frames, "
              f"dropped {self.n_dropped}, peak queue {self.max_queue_depth}")
