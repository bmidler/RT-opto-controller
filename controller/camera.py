"""Frame sources.

`FlirSource` wraps the vendored FLIR/PySpin helpers (utils/flir.py) for a
single hardware-triggered camera. `VideoFileSource` and `SyntheticSource` let
the whole pipeline run with no hardware attached (offline / smoke tests).

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

Frame = Tuple[np.ndarray, float]


def _read_camera_int(camera, node_name: str, default) -> int:
    """Read an integer GenICam node (e.g. "Width"/"Height") off a PySpin camera.

    PySpin exposes configured ROI nodes as convenience attributes on the camera
    object once it is Init()'d, so `camera.Width.GetValue()` returns the value
    the sensor actually applied. Returns `default` (the requested value) if the
    node can't be read for any reason.
    """
    try:
        return int(getattr(camera, node_name).GetValue())
    except Exception:
        return int(default) if default is not None else 0


class FrameSource:
    frame_height: int = 0
    frame_width: int = 0
    is_color: bool = False
    color_is_bgr: bool = True
    camera_serial: str = ""        # filled in by hardware sources on open()
    camera_model: str = ""         # filled in by hardware sources on open()

    def open(self) -> None: ...
    def grab(self) -> Optional[Frame]: ...
    def close(self) -> None: ...


class FlirSystem:
    """Owns the PySpin System + device list, shared across FlirSource cameras.

    PySpin's System is a process-wide singleton and its device list must be
    cleared / the system released exactly once. When several cameras run at
    the same time they share ONE of these: each FlirSource opens its own device
    from the shared list, and the controller closes the system once at the end.
    """

    def __init__(self, config: ControllerConfig):
        self.config = config
        self._flir = None
        self.system = None
        self.device_list = None

    def open(self) -> "FlirSystem":
        from utils import flir  # lazy: imports PySpin
        self._flir = flir
        self.system = flir.LoadSystem(self.config.to_flir_cam_params())
        self.device_list = flir.GetDeviceList(self.system)
        return self

    def __len__(self) -> int:
        return len(self.device_list) if self.device_list is not None else 0

    def device(self, selection: int):
        n = len(self)
        if selection >= n:
            raise RuntimeError(
                f"camera_selection={selection} but only {n} camera(s) found")
        return self.device_list[selection]

    def close(self) -> None:
        if self.system is not None and self.device_list is not None:
            try:
                self._flir.CloseSystem(self.system, self.device_list)
            except Exception:
                pass
            self.system = None
            self.device_list = None


class FlirSource(FrameSource):
    """A single FLIR camera via utils.flir (PySpin).

    The camera is expected to be in hardware-trigger mode (cameraTrigger e.g.
    'Line3'); GetNextImage blocks until the camera-trigger Arduino fires, which
    paces this source at exactly the configured frame rate.

    `system` is a shared FlirSystem when more than one camera runs at once. If
    None, this source owns its own system (legacy single-camera behaviour).
    """

    def __init__(self, config: ControllerConfig, system: Optional["FlirSystem"] = None):
        self.config = config
        self.cam_params = config.to_flir_cam_params()
        self._flir = None
        self._system = system
        self._owns_system = system is None
        self._camera = None
        # Give up on a single GetNextImage after this long so the hot loop can
        # notice a stop request (triggers stopped) instead of blocking forever.
        self._grab_timeout_ms = int(max(1000, 5 * config.frame_budget_ms))

    def open(self) -> None:
        from utils import flir  # lazy: imports PySpin
        self._flir = flir

        if self._system is None:
            self._system = FlirSystem(self.config).open()

        device = self._system.device(self.config.camera_selection)
        self.cam_params["device"] = device
        self.cam_params["camera"] = device
        # GetSerialNumber reads the transport-layer nodemap (available before
        # Init); flir returns [] when it's unreadable, so coerce that to "".
        serial = flir.GetSerialNumber(device)
        self.camera_serial = str(serial) if serial else ""
        self.cam_params["cameraSerialNo"] = self.camera_serial

        self._camera, self.cam_params = flir.OpenCamera(self.cam_params)
        if not flir.StartGrabbing(self._camera):
            raise RuntimeError("FLIR StartGrabbing failed")

        # Read the ACTUAL geometry back off the camera after configuration:
        # utils/flir.py rounds frameWidth/Height to a multiple of 16 and the
        # sensor may clamp the ROI further, so requested values can be wrong.
        # Fall back to the requested values only if the node read fails.
        self.frame_width = _read_camera_int(
            self._camera, "Width", self.cam_params.get("frameWidth"))
        self.frame_height = _read_camera_int(
            self._camera, "Height", self.cam_params.get("frameHeight"))
        self.camera_model = str(self.cam_params.get("cameraModel", "") or "")
        # Keep cam_params consistent with what the writer will encode.
        self.cam_params["frameWidth"] = self.frame_width
        self.cam_params["frameHeight"] = self.frame_height
        self.is_color = self.config.pixel_format_input not in ("gray", "mono8")
        # utils/flir.py sets PixelFormat_RGB8Packed for rgb24 -> channel order RGB.
        self.color_is_bgr = False

    def grab(self) -> Optional[Frame]:
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
            # Drop EVERY reference to the PySpin camera. flir.CloseCamera only
            # del's its local handle, so self._camera and the cam_params entries
            # still point at the device. If any camera reference is alive when
            # the system is released (FlirSystem.close -> System.ReleaseInstance),
            # Spinnaker aborts the whole process at C++ level -- uncatchable, no
            # traceback. Releasing them here makes that teardown safe.
            self._camera = None
            self.cam_params["camera"] = None
            self.cam_params["device"] = None
            # Only tear the system down if we created it; a shared system is
            # closed once by the controller after every camera is closed.
            if self._owns_system and self._system is not None:
                self._system.close()


class VideoFileSource(FrameSource):
    """Read frames from a video file, paced to the target frame rate.

    Useful for replaying a recorded session through the full pipeline offline.
    cv2 returns BGR frames (same as the model's training pipeline).
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

    def __init__(self, config: ControllerConfig, index: int = 0):
        self.config = config
        self.index = index            # offsets the bar so each camera differs
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
        # Offset both speed and start so each synthetic camera looks distinct.
        bar = (self._idx * 7 + self.index * w // 5) % w
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


def make_sources(config: ControllerConfig):
    """Build one FrameSource per configured camera.

    Returns (sources, inference_index, flir_system) where:
      * sources[i]        corresponds to config.per_camera_configs()[i]
      * inference_index   is the source fed to the model
      * flir_system       is the shared FlirSystem to close at shutdown
                          (None for non-FLIR sources, or single-camera FLIR
                          where the source owns its own system).
    """
    cam_configs = config.per_camera_configs()
    inf_idx = config.inference_camera_index()

    if config.source == "flir":
        # Several cameras must share one PySpin system; a lone camera keeps the
        # legacy path where the source owns its system.
        system = FlirSystem(config).open() if len(cam_configs) > 1 else None
        sources = [FlirSource(c, system=system) for c in cam_configs]
        return sources, inf_idx, system

    if config.source == "video":
        # Each camera reads its own capture of source_video (offline testing).
        return [VideoFileSource(c) for c in cam_configs], inf_idx, None

    if config.source == "synthetic":
        return ([SyntheticSource(c, index=i) for i, c in enumerate(cam_configs)],
                inf_idx, None)

    raise ValueError(f"Unknown source: {config.source!r}")
