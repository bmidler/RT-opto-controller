"""Real-time timing watchdog.

Tracks, per frame: the inter-frame interval, the per-stage hot-path timings,
and the writer queue depth. Emits rate-limited warnings when the loop is at
risk of falling behind the fixed frame rate, and prints a summary at the end.

The point is to make timing problems *visible* (and bounded) rather than to
let them silently cascade: a fixed-rate acquisition that can't keep up will
either back up the camera buffer or fill the writer queue, both of which show
up here first.
"""

from __future__ import annotations

import time

import numpy as np

from .config import ControllerConfig


class LatencyMonitor:
    def __init__(self, config: ControllerConfig):
        self.budget_ms = config.frame_budget_ms
        self.warn_ms = config.warn_factor * self.budget_ms
        self.warn_min_interval = config.warn_min_interval_sec
        self.queue_warn_depth = max(8, config.write_queue_maxsize // 2)

        self._last_warn = {}     # category -> last warn time
        self.hotpath_ms = []
        self.interval_ms = []
        self.n = 0
        self.n_over_budget = 0
        self.n_interval_over = 0

    def _warn(self, category: str, msg: str) -> None:
        now = time.perf_counter()
        last = self._last_warn.get(category, 0.0)
        if now - last >= self.warn_min_interval:
            self._last_warn[category] = now
            print(f"[latency] WARNING: {msg}", flush=True)

    def record(self, *, interval_ms: float | None, preprocess_ms: float,
               inference_ms: float, hotpath_ms: float, queue_depth: int) -> None:
        self.n += 1
        self.hotpath_ms.append(hotpath_ms)
        if interval_ms is not None:
            self.interval_ms.append(interval_ms)

        if hotpath_ms > self.budget_ms:
            self.n_over_budget += 1
            self._warn("hotpath",
                       f"frame processing {hotpath_ms:.1f} ms exceeded the "
                       f"{self.budget_ms:.1f} ms budget "
                       f"(preprocess {preprocess_ms:.1f}, infer {inference_ms:.1f}). "
                       f"Inference/stim cannot keep up at this frame rate.")
        elif hotpath_ms > self.warn_ms:
            self._warn("hotpath_near",
                       f"frame processing {hotpath_ms:.1f} ms is close to the "
                       f"{self.budget_ms:.1f} ms budget.")

        if interval_ms is not None and interval_ms > 1.5 * self.budget_ms:
            self.n_interval_over += 1
            self._warn("interval",
                       f"inter-frame interval {interval_ms:.1f} ms >> expected "
                       f"{self.budget_ms:.1f} ms: dropped/late camera triggers?")

        if queue_depth >= self.queue_warn_depth:
            self._warn("queue",
                       f"writer queue depth {queue_depth} is growing -- the "
                       f"video encoder/disk cannot sustain the frame rate. "
                       f"Use GPU encoding, lower resolution, or a faster disk.")

    def summary(self) -> dict:
        hp = np.array(self.hotpath_ms) if self.hotpath_ms else np.array([0.0])
        iv = np.array(self.interval_ms) if self.interval_ms else np.array([0.0])
        s = {
            "frames": self.n,
            "hotpath_mean_ms": float(hp.mean()),
            "hotpath_p95_ms": float(np.percentile(hp, 95)),
            "hotpath_max_ms": float(hp.max()),
            "interval_mean_ms": float(iv.mean()),
            "interval_p95_ms": float(np.percentile(iv, 95)),
            "budget_ms": self.budget_ms,
            "frames_over_budget": self.n_over_budget,
            "frames_interval_over": self.n_interval_over,
            "effective_fps": (1000.0 / iv.mean()) if iv.mean() > 0 else 0.0,
        }
        return s

    def print_summary(self) -> None:
        s = self.summary()
        print("\n=== Timing summary ===")
        print(f"  frames processed   : {s['frames']}")
        print(f"  budget             : {s['budget_ms']:.2f} ms/frame")
        print(f"  hot-path mean/p95/max: {s['hotpath_mean_ms']:.2f} / "
              f"{s['hotpath_p95_ms']:.2f} / {s['hotpath_max_ms']:.2f} ms")
        print(f"  interval mean/p95  : {s['interval_mean_ms']:.2f} / "
              f"{s['interval_p95_ms']:.2f} ms  (~{s['effective_fps']:.1f} fps)")
        print(f"  frames over budget : {s['frames_over_budget']} / {s['frames']}")
        if s["frames_over_budget"] == 0 and s["frames"] > 0:
            print("  -> kept up with the frame rate.")
