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


# Fields that an entry in the `cameras` list is allowed to override per-camera.
# Everything else (frame_rate, codec, model, stim, ...) is shared across cameras.
PER_CAMERA_FIELDS = {
    "camera_make", "camera_name", "camera_selection", "camera_serial",
    "frame_width", "frame_height", "pixel_format_input", "camera_trigger",
    "camera_exposure_us", "camera_gain", "disable_gamma",
    "buffer_mode", "buffer_size", "camera_debug",
}


@dataclass
class ControllerConfig:
    # --- Frame source ------------------------------------------------------
    # "flir"      -> real FLIR camera via utils/flir.py + PySpin (hw-triggered)
    # "video"     -> read frames from a video file (offline testing)
    # "synthetic" -> generated frames (smoke testing, no hardware)
    source: str = "flir"
    source_video: str = ""          # path, required when source == "video"

    frame_rate: float = 30.0        # target fps (must match model training fps)
    rec_time_sec: float = 0.0       # 0 => run until Ctrl-C / max frames
    max_frames: int = 0             # 0 => unlimited

    # --- Camera (FLIR cam_params) -----------------------------------------
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

    # --- Multi-camera ------------------------------------------------------
    # Leave `cameras` empty to run the single camera described by the camera_*
    # fields above (legacy behaviour). Otherwise each entry is a mapping that
    # overrides only the PER_CAMERA_FIELDS you name; anything omitted is
    # inherited from the camera_* values above. EVERY listed camera is recorded
    # to its own video subfolder; exactly ONE -- the one whose camera_name
    # equals `inference_camera` -- is fed to the model.
    cameras: list = field(default_factory=list)
    inference_camera: str = ""      # camera_name used for inference

    # --- Model / inference -------------------------------------------------
    model_checkpoint: str = "utils/checkpoints/best_model.pt"
    device: str = "auto"            # "auto" | "cuda" | "mps" | "cpu"
    spatial_scale: Optional[float] = None   # None => use value stored in ckpt

    # --- Classification -> stim decision ----------------------------------
    # Class indices whose prediction should drive the laser.
    trigger_classes: list = field(default_factory=list)
    class_names: list = field(default_factory=list)   # optional, for logging
    # Debounce: consecutive trigger frames required to register an event onset.
    onset_frames: int = 1
    # offset_frames is retained for logging/back-compat only. In fixed-duration
    # mode the train length is set by stim_duration_sec, NOT by event offset.
    offset_frames: int = 1
    # Fixed-duration mode: on each detected onset the laser pulses for this many
    # seconds, independent of how long the event lasts. A new onset within an
    # active train resets the timer to the new onset. With max_pulses left at 0
    # the PC controls the duration; otherwise the Arduino also caps at
    # max_pulses (set it to round(stim_duration_sec * frequency_hz) to match).
    stim_duration_sec: float = 25.0
    # Ignore a new onset within this many seconds of the last one. Guards
    # against a flickering classifier re-triggering the board many times a
    # second (which re-zeroes its pulse counter each time). 0.0 => disabled.
    stim_refractory_sec: float = 0.0

    # --- Stim Arduino ------------------------------------------------------
    stim_enabled: bool = True
    stim_serial_port: str = ""      # e.g. "COM5" / "/dev/ttyACM1"
    stim_baud: int = 115200
    stim_pin: int = 9               # TTL gate pin into the laser driver
    pulse_width_us: int = 5000
    frequency_hz: float = 20.0
    max_pulses: int = 0             # hardware safety cap on pulses/train; 0 => no cap
    watchdog_ms: int = 500          # auto-off if no keepalive within this window
    keepalive_ms: int = 150         # how often to refresh the watchdog while ON

    # --- Camera-trigger Arduino (arduino/camera_controller/trigger.ino) ---
    cam_trigger_enabled: bool = True
    cam_trigger_serial_port: str = ""   # e.g. "COM4"
    cam_trigger_baud: int = 115200
    cam_trigger_pins: list = field(default_factory=lambda: [6])

    # When True, no real serial ports are opened; commands are logged instead.
    # Lets the whole pipeline run without Arduinos attached.
    serial_dry_run: bool = False

    # --- Output (video + per-frame log) -----------------------------------
    output_folder: str = "./rt_opto_recordings"
    # Name of the session subfolder inside output_folder (each camera gets its
    # own folder beneath it). Empty => a timestamped name is generated. NOTE: a
    # fixed name is reused across runs, so set a unique name per session to
    # avoid overwriting an earlier session's per-frame log.
    session_name: str = ""
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

    # --- Multi-camera helpers ---------------------------------------------
    def per_camera_configs(self) -> list["ControllerConfig"]:
        """One ControllerConfig per physical camera.

        With an empty `cameras` list this returns [self] (legacy single-cam).
        Otherwise each entry overrides its PER_CAMERA_FIELDS on a copy of self;
        omitted fields inherit this config's values. The `cameras` list is
        cleared on the copies so they are unambiguously single-camera configs.
        """
        if not self.cameras:
            return [self]
        out = []
        for i, entry in enumerate(self.cameras):
            if not isinstance(entry, dict):
                raise ValueError(
                    f"cameras[{i}] must be a mapping of field->value, "
                    f"got {type(entry).__name__}")
            unknown = set(entry) - PER_CAMERA_FIELDS
            if unknown:
                raise ValueError(
                    f"cameras[{i}] has unsupported keys {sorted(unknown)}; "
                    f"allowed per-camera keys: {sorted(PER_CAMERA_FIELDS)}")
            out.append(dataclasses.replace(self, cameras=[], **entry))
        return out

    def inference_camera_index(self) -> int:
        """Index into per_camera_configs() of the camera used for inference."""
        cams = self.per_camera_configs()
        if len(cams) == 1:
            return 0
        if not self.inference_camera:
            raise ValueError(
                "multiple cameras are configured but inference_camera is unset; "
                "set inference_camera to one of "
                f"{[c.camera_name for c in cams]}")
        names = [c.camera_name for c in cams]
        if self.inference_camera not in names:
            raise ValueError(
                f"inference_camera={self.inference_camera!r} is not one of the "
                f"configured camera_name values {names}")
        return names.index(self.inference_camera)

    @property
    def frame_budget_ms(self) -> float:
        return 1000.0 / self.frame_rate if self.frame_rate > 0 else float("inf")

    def to_flir_cam_params(self) -> dict:
        """Build the cam_params dict that utils/flir.py expects.

        Mirrors the keys utils/flir.py reads off cam_params. The encode keys at
        the bottom are unused by flir.py; controller/writer.py reads its encode
        settings straight off this config instead.
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
        # Multi-camera sanity: unique names + a resolvable inference camera.
        cam_cfgs = self.per_camera_configs()
        names = [c.camera_name for c in cam_cfgs]
        if len(set(names)) != len(names):
            raise ValueError(f"camera_name values must be unique, got {names}")
        self.inference_camera_index()   # raises if inference_camera unresolved
        if not self.trigger_classes:
            print("[config] WARNING: trigger_classes is empty -> the laser "
                  "will never fire. Set the class indices that should stimulate.")
        if self.stim_enabled and not self.stim_serial_port and not self.serial_dry_run:
            raise ValueError(
                "stim_enabled but stim_serial_port is empty "
                "(set serial_dry_run: true to test without hardware)")
        self._validate_stim()
        if not Path(self.model_checkpoint).exists():
            print(f"[config] WARNING: model_checkpoint not found: "
                  f"{self.model_checkpoint}")

    def _validate_stim(self) -> None:
        """Reject stim parameter sets the firmware would refuse or that would
        defeat its safety net. Checked here so a bad config fails before any
        hardware is touched, and mirrors the validation in
        arduino/stim_controller/stim_controller.ino."""
        if not self.stim_enabled:
            return
        if self.frequency_hz <= 0:
            raise ValueError("frequency_hz must be > 0 when stim_enabled")
        if self.pulse_width_us <= 0:
            raise ValueError("pulse_width_us must be > 0 when stim_enabled")

        # Duty cycle >= 100% leaves the gate permanently HIGH: the firmware's
        # "end of pulse" condition can never be reached.
        duty = self.pulse_width_us * 1e-6 * self.frequency_hz
        if duty >= 1.0:
            raise ValueError(
                f"pulse_width_us x frequency_hz = {duty:.2f} (>= 100% duty): "
                f"the laser would never switch off. Reduce pulse_width_us "
                f"({self.pulse_width_us}) or frequency_hz ({self.frequency_hz}).")
        if duty > 0.5:
            print(f"[config] WARNING: stim duty cycle is {duty:.0%}")

        # The watchdog is the last line of defence against a stuck-on laser;
        # the firmware refuses to disable it, so refuse here too.
        if self.watchdog_ms <= 0:
            raise ValueError("watchdog_ms must be > 0 (the stim board's "
                             "auto-off cannot be disabled)")
        if self.keepalive_ms >= self.watchdog_ms:
            raise ValueError(
                f"keepalive_ms ({self.keepalive_ms}) must be < watchdog_ms "
                f"({self.watchdog_ms}) or the watchdog will fire mid-train")
        if self.keepalive_ms > self.watchdog_ms / 2:
            print(f"[config] WARNING: keepalive_ms ({self.keepalive_ms}) leaves "
                  f"little margin under watchdog_ms ({self.watchdog_ms}); one "
                  f"dropped keepalive will cut the train short")

        # Keepalives are only sent when a frame arrives, so the frame period
        # bounds how often the watchdog can actually be fed.
        if self.frame_rate > 0:
            frame_ms = 1000.0 / self.frame_rate
            if frame_ms > self.watchdog_ms / 2:
                print(f"[config] WARNING: frame period {frame_ms:.0f} ms is "
                      f"large relative to watchdog_ms ({self.watchdog_ms}); "
                      f"keepalives ride on frame arrivals")

        # max_pulses is the board-side cap; it should match the PC-side window.
        if self.max_pulses > 0:
            expected = round(self.stim_duration_sec * self.frequency_hz)
            if self.max_pulses != expected:
                print(f"[config] WARNING: max_pulses={self.max_pulses} but "
                      f"stim_duration_sec x frequency_hz = {expected}; the "
                      f"smaller of the two will end the train")

        if self.stim_pin in (0, 1):
            print(f"[config] WARNING: stim_pin={self.stim_pin} is a UART pin on "
                  f"many boards; the firmware pins STIM_PIN at compile time and "
                  f"will reject a mismatching config line")
