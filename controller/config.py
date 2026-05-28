"""Controller configuration: a single dataclass loaded from YAML.

Only the fields the controller actually needs are defined here. Model
architecture parameters (cnn_channels, gru_hidden, spatial_scale, ...) are
NOT stored here on purpose -- they are read back from the training checkpoint
so live inference can never silently diverge from how the model was trained.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ControllerConfig:
    # --- Frame source ------------------------------------------------------
    # "flir"      -> real FLIR camera via campy/PySpin (hardware-triggered)
    # "video"     -> read frames from a video file (offline testing)
    # "synthetic" -> generated frames (smoke testing, no hardware)
    source: str = "flir"
    source_video: str = ""          # path, required when source == "video"

    frame_rate: float = 30.0        # target fps (must match model training fps)
    rec_time_sec: float = 0.0       # 0 => run until Ctrl-C / max frames
    max_frames: int = 0             # 0 => unlimited

    # --- Camera (FLIR / campy cam_params) ---------------------------------
    camera_make: str = "flir"
    camera_selection: int = 0       # device index
    camera_name: str = "AssayCamera"
    camera_serial: str = ""         # filled in from device if blank
    frame_width: int = 2048
    frame_height: int = 1536
    pixel_format_input: str = "gray"   # "gray" (mono8), "rgb24", "bayer_rggb8"
    camera_trigger: str = "Line3"      # hardware trigger line, or "None"
    camera_exposure_us: int = 10000
    camera_gain: float = 15.0
    disable_gamma: bool = False
    buffer_mode: str = "OldestFirst"
    buffer_size: int = 100
    camera_debug: bool = False

    # --- Model / inference -------------------------------------------------
    model_checkpoint: str = "RT-opto-main/output/best_model.pt"
    device: str = "auto"            # "auto" | "cuda" | "mps" | "cpu"
    spatial_scale: Optional[float] = None   # None => use value stored in ckpt

    # --- Classification -> stim decision ----------------------------------
    # Class indices whose prediction should drive the laser.
    trigger_classes: list = field(default_factory=list)
    class_names: list = field(default_factory=list)   # optional, for logging
    # Debounce: frames of agreement required to flip the stim state.
    onset_frames: int = 1
    offset_frames: int = 1

    # --- Stim Arduino ------------------------------------------------------
    stim_enabled: bool = True
    stim_serial_port: str = ""      # e.g. "COM5" / "/dev/ttyACM1"
    stim_baud: int = 115200
    stim_pin: int = 9               # TTL gate pin into the laser driver
    pulse_width_us: int = 5000
    frequency_hz: float = 20.0
    max_pulses: int = 0             # 0 => continuous while detected
    watchdog_ms: int = 500          # auto-off if no keepalive within this window
    keepalive_ms: int = 150         # how often to refresh the watchdog while ON

    # --- Camera-trigger Arduino (existing campy Teensy) -------------------
    cam_trigger_enabled: bool = True
    cam_trigger_serial_port: str = ""   # e.g. "COM4"
    cam_trigger_baud: int = 115200
    cam_trigger_pins: list = field(default_factory=lambda: [6])

    # When True, no real serial ports are opened; commands are logged instead.
    # Lets the whole pipeline run without Arduinos attached.
    serial_dry_run: bool = False

    # --- Output (video + per-frame log) -----------------------------------
    output_folder: str = "./rt_opto_recordings"
    video_filename: str = "session.mp4"
    codec: str = "h264"             # "h264" | "h265"
    quality: int = 23
    preset: str = "fast"
    gpu_id: int = -1                # -1 => CPU encode; 0+ => GPU index
    gpu_make: str = "nvidia"        # "nvidia" | "amd" | "intel"
    pixel_format_output: str = "rgb0"
    ffmpeg_log_level: str = "warning"

    # --- Live preview ------------------------------------------------------
    preview_enabled: bool = True
    preview_fps: float = 15.0       # display refresh rate (decoupled from acq)
    preview_downsample: int = 2     # show every Nth pixel (faster, smaller window)

    # --- Latency / backpressure -------------------------------------------
    # Warn when a hot-path frame takes longer than warn_factor * frame_budget.
    warn_factor: float = 0.9
    warn_min_interval_sec: float = 2.0   # rate-limit repeated warnings
    write_queue_maxsize: int = 256       # bounded; protects against memory blowup
    drop_video_when_full: bool = True    # drop video frames (not log/inference)

    # ----------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str) -> "ControllerConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(
                f"Unrecognized keys in {path}: {sorted(unknown)}.\n"
                f"Valid keys: {sorted(known)}"
            )
        return cls(**raw)

    @property
    def frame_budget_ms(self) -> float:
        return 1000.0 / self.frame_rate if self.frame_rate > 0 else float("inf")

    def to_campy_cam_params(self) -> dict:
        """Build the dict that campy's FLIR functions expect.

        Mirrors the keys campy's flir.py / writer.py read from cam_params.
        """
        return {
            "cameraMake": self.camera_make,
            "cameraName": self.camera_name,
            "cameraSelection": self.camera_selection,
            "cameraSerialNo": self.camera_serial,
            "frameWidth": self.frame_width,
            "frameHeight": self.frame_height,
            "frameRate": self.frame_rate,
            "pixelFormatInput": self.pixel_format_input,
            "pixelFormatOutput": self.pixel_format_output,
            "cameraTrigger": self.camera_trigger,
            "cameraExposureTimeInUs": self.camera_exposure_us,
            "cameraGain": self.camera_gain,
            "disableGamma": self.disable_gamma,
            "bufferMode": self.buffer_mode,
            "bufferSize": self.buffer_size,
            "cameraDebug": self.camera_debug,
            # writer / encode params
            "videoFolder": self.output_folder,
            "videoFilename": self.video_filename,
            "codec": self.codec,
            "quality": self.quality,
            "preset": self.preset,
            "gpuID": self.gpu_id,
            "gpuMake": self.gpu_make,
            "ffmpegLogLevel": self.ffmpeg_log_level,
        }

    def validate(self) -> None:
        if self.frame_rate <= 0:
            raise ValueError("frame_rate must be > 0")
        if self.source == "video" and not self.source_video:
            raise ValueError("source == 'video' requires source_video path")
        if not self.trigger_classes:
            print("[config] WARNING: trigger_classes is empty -> the laser "
                  "will never fire. Set the class indices that should stimulate.")
        if self.stim_enabled and not self.stim_serial_port and not self.serial_dry_run:
            raise ValueError(
                "stim_enabled but stim_serial_port is empty "
                "(set serial_dry_run: true to test without hardware)")
        if not Path(self.model_checkpoint).exists():
            print(f"[config] WARNING: model_checkpoint not found: "
                  f"{self.model_checkpoint}")
