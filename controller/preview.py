"""Live preview window (optional).

Runs entirely in a daemon thread fed by a single-slot buffer, so the
classify->stim hot path is never slowed by drawing or GUI calls -- it just
overwrites "the latest frame to show" and moves on. The preview thread paces
itself to preview_fps and always renders the most recent frame, dropping any
it couldn't keep up with.

An overlay shows the frame number, predicted class + confidence, and a big
STIM ON / stim off indicator (red border while the laser is firing) so you can
see exactly what is driving stimulation in real time.

If no display is available (headless / SSH), GUI failures are caught and the
preview quietly disables itself; the pipeline keeps running.
"""

from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np

from .config import ControllerConfig

WINDOW = "RT-opto live"


class Preview:
    def __init__(self, config: ControllerConfig):
        self.enabled = config.preview_enabled
        self.fps = max(1.0, config.preview_fps)
        self.downsample = max(1, int(config.preview_downsample))
        self.color_is_bgr = True
        self.is_color = False

        self._q: deque = deque(maxlen=1)
        self._thread = None
        self._stop = threading.Event()
        self.quit_requested = False

    # -- lifecycle ----------------------------------------------------------
    def open(self, is_color: bool, color_is_bgr: bool) -> None:
        if not self.enabled:
            return
        self.is_color = is_color
        self.color_is_bgr = color_is_bgr
        self._thread = threading.Thread(target=self._run, name="preview",
                                        daemon=True)
        self._thread.start()

    def submit(self, frame, pred_class: int, class_name: str,
               confidence: float, stim_on: bool, frame_number: int) -> None:
        if not self.enabled:
            return
        # Overwrites the single slot -- O(1), never blocks the hot path.
        self._q.append((frame, pred_class, class_name, confidence,
                        stim_on, frame_number))

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    # -- rendering (pure; unit-testable without a GUI) ---------------------
    def render(self, frame, pred_class, class_name, confidence, stim_on,
               frame_number):
        import cv2
        if frame.ndim == 2:
            bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif self.color_is_bgr:
            bgr = frame
        else:
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if self.downsample > 1:
            bgr = bgr[::self.downsample, ::self.downsample]
        bgr = np.ascontiguousarray(bgr).copy()   # never draw on the queued frame

        label = class_name if class_name else f"class {pred_class}"
        line1 = f"frame {frame_number}   {label}   {confidence * 100:.0f}%"
        _text(cv2, bgr, line1, (10, 28), 0.6, (255, 255, 255))

        if stim_on:
            _text(cv2, bgr, "STIM ON", (10, 62), 0.9, (0, 0, 255))
            h, w = bgr.shape[:2]
            cv2.rectangle(bgr, (1, 1), (w - 2, h - 2), (0, 0, 255), 4)
        else:
            _text(cv2, bgr, "stim off", (10, 62), 0.9, (0, 180, 0))
        return bgr

    # -- worker thread ------------------------------------------------------
    def _run(self) -> None:
        try:
            import cv2
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        except Exception as e:
            print(f"[preview] disabled (no display available): {e}")
            self.enabled = False
            return

        period = 1.0 / self.fps
        while not self._stop.is_set():
            t0 = time.perf_counter()
            if self._q:
                item = self._q[-1]
                try:
                    img = self.render(*item)
                    cv2.imshow(WINDOW, img)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):   # 'q' or Esc
                        self.quit_requested = True
                except Exception as e:
                    print(f"[preview] render/show failed, disabling: {e}")
                    break
            else:
                time.sleep(0.005)
            dt = time.perf_counter() - t0
            if dt < period:
                time.sleep(period - dt)

        try:
            import cv2
            cv2.destroyWindow(WINDOW)
        except Exception:
            pass


def _text(cv2, img, text, org, scale, color):
    # White/colored text with a dark outline so it stays readable on any frame.
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, 2, cv2.LINE_AA)
