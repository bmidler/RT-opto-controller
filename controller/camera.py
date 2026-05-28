"""Frame sources.

`FlirSource` wraps campy's FLIR/PySpin functions for a single
hardware-triggered camera. `VideoFileSource` and `SyntheticSource` let the
whole pipeline run with no hardware attached (offline testing / smoke tests).

All sources implement the same tiny interface:
    open()  -> configure, sets .frame_height / .frame_width / .is_color
    grab()  -> (frame_uint8, camera_timestamp_sec) or None if no frame ready
    close()
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np

from .config import ControllerConfig
from .paths import setup_paths

setup_paths()

Frame = Tuple[np.ndarray, float]


class FrameSource:
    frame_height: int = 0
    frame_width: int = 0
    is_color: bool = False
    color_is_bgr: bool = True

    def open(self) -> None: ...
    def grab(self) -> Optional[Frame]: ...
    def close(self) -> None: ...


class FlirSource(FrameSource):
    """Single FLIR camera via campy.cameras.flir (PySpin).

    The camera is expected to be in hardware-trigger mode (cameraTrigger e.g.
    'Line3'); GetNextImage blocks until the camera-trigger Arduino fires, which
    paces this source at exactly the configured frame rate.
    """

    def __init__(self, config: ControllerConfig):
        self.config = config
        self.cam_params = config.to_campy_cam_params()
        self._flir = None
        self._system = None
        self._device_list = None
        self._camera = None
        # Give up on a single GetNextImage after this long so the hot loop can
        # notice a stop request (triggers stopped) instead of blocking forever.
        self._grab_timeout_ms = int(max(1000, 5 * config.frame_budget_ms))

    def open(self) -> None:
        from campy.cameras import flir  # lazy: imports PySpin
        self._flir = flir

        self._system = flir.LoadSystem(self.cam_params)
        self._device_list = flir.GetDeviceList(self._system)
        n = len(self._device_list)
        sel = self.config.camera_selection
        if sel >= n:
            raise RuntimeError(f"camera_selection={sel} but only {n} camera(s) found")

        device = self._device_list[sel]
        self.cam_params["device"] = device
        self.cam_params["camera"] = device
        self.cam_params["cameraSerialNo"] = flir.GetSerialNumber(device)

        self._camera, self.cam_params = flir.OpenCamera(self.cam_params)
        if not flir.StartGrabbing(self._camera):
            raise RuntimeError("FLIR StartGrabbing failed")

        self.frame_height = int(self.cam_params["frameHeight"])
        self.frame_width = int(self.cam_params["frameWidth"])
        self.is_color = self.config.pixel_format_input not in ("gray", "mono8")
        # campy configures PixelFormat_RGB8Packed for rgb24 -> channel order RGB.
        self.color_is_bgr = False

    def grab(self) -> Optional[Frame]:
        flir = self._flir
        try:
            image_result = self._camera.GetNextImage(self._grab_timeout_ms)
        except Exception:
            return None  # timeout while waiting for the next hardware trigger
        try:
            if image_result.IsIncomplete():
                return None
            img = np.array(image_result.GetNDArray(), copy=True)
            try:
                ts = image_result.GetChunkData().GetTimestamp() * 1e-9
            except Exception:
                ts = time.perf_counter()
            return img, ts
        finally:
            image_result.Release()

    def close(self) -> None:
        flir = self._flir
        try:
            if self._camera is not None:
                flir.CloseCamera(self.cam_params, self._camera)
        finally:
            if self._system is not None and self._device_list is not None:
                try:
                    flir.CloseSystem(self._system, self._device_list)
                except Exception:
                    pass


class VideoFileSource(FrameSource):
    """Read frames from a video file, paced to the target frame rate.

    Useful for replaying a recorded session through the full pipeline offline.
    cv2 returns BGR frames (same as the RT-opto training pipeline).
    """

    def __init__(self, config: ControllerConfig, realtime: bool = True):
        self.config = config
        self.realtime = realtime
        self._cap = None
        self._period = 1.0 / config.frame_rate if config.frame_rate > 0 else 0.0
        self._next_t = None
        self._idx = 0

    def open(self) -> None:
        import cv2
        self._cap = cv2.VideoCapture(self.config.source_video)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video {self.config.source_video}")
        self.frame_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.is_color = True
        self.color_is_bgr = True

    def grab(self) -> Optional[Frame]:
        if self.realtime and self._period > 0:
            now = time.perf_counter()
            if self._next_t is None:
                self._next_t = now
            if now < self._next_t:
                time.sleep(self._next_t - now)
            self._next_t += self._period

        ret, frame = self._cap.read()
        if not ret:
            return None
        ts = self._idx / self.config.frame_rate
        self._idx += 1
        return frame, ts

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()


class SyntheticSource(FrameSource):
    """Generated grayscale frames (a drifting bright bar), paced to fps.

    Lets the pipeline run end-to-end with no camera and no video file.
    """

    def __init__(self, config: ControllerConfig):
        self.config = config
        self.frame_height = config.frame_height
        self.frame_width = config.frame_width
        self.is_color = False
        self.color_is_bgr = False
        self._period = 1.0 / config.frame_rate if config.frame_rate > 0 else 0.0
        self._next_t = None
        self._idx = 0

    def open(self) -> None:
        pass

    def grab(self) -> Optional[Frame]:
        if self._period > 0:
            now = time.perf_counter()
            if self._next_t is None:
                self._next_t = now
            if now < self._next_t:
                time.sleep(self._next_t - now)
            self._next_t += self._period

        h, w = self.frame_height, self.frame_width
        frame = np.zeros((h, w), dtype=np.uint8)
        bar = (self._idx * 7) % w
        frame[:, bar:min(bar + max(1, w // 20), w)] = 255
        ts = self._idx / self.config.frame_rate
        self._idx += 1
        return frame, ts

    def close(self) -> None:
        pass


def make_source(config: ControllerConfig) -> FrameSource:
    if config.source == "flir":
        return FlirSource(config)
    if config.source == "video":
        return VideoFileSource(config)
    if config.source == "synthetic":
        return SyntheticSource(config)
    raise ValueError(f"Unknown source: {config.source!r}")
