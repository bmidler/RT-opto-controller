"""Streaming wrapper around the RT-opto CNN-GRU classifier.

Loads the trained checkpoint, replicates the exact training preprocessing,
and exposes a stateful single-frame `classify()` that carries the GRU hidden
state across frames (the model is recurrent -- every frame must be fed in
order, none skipped, or temporal context is corrupted).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch

from .paths import setup_paths

setup_paths()
from model import VideoClassifier  # noqa: E402  (provided by RT-opto-main/)


# RT-opto Config defaults, used only if a key is missing from the checkpoint.
_ARCH_DEFAULTS = dict(
    cnn_channels=[16, 32, 64, 128],
    gru_hidden=128,
    gru_layers=1,
    dropout=0.5,
    spatial_scale=0.35,
    target_fps=30,
)


@dataclass
class Prediction:
    pred_class: int
    confidence: float          # softmax prob of pred_class
    logits: np.ndarray         # (num_classes,)


def _select_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


class Classifier:
    def __init__(self, checkpoint_path: str, device: str = "auto",
                 spatial_scale: Optional[float] = None,
                 color_input_is_bgr: bool = True):
        self.device = _select_device(device)
        ckpt = torch.load(checkpoint_path, map_location=self.device,
                           weights_only=False)

        cfg = ckpt.get("config", {}) or {}

        def arch(key):
            return cfg.get(key, _ARCH_DEFAULTS[key])

        self.num_classes = int(ckpt["num_classes"])
        self.spatial_scale = (spatial_scale if spatial_scale is not None
                              else float(arch("spatial_scale")))
        self.target_fps = float(arch("target_fps"))
        self._color_is_bgr = color_input_is_bgr

        self.model = VideoClassifier(
            num_classes=self.num_classes,
            cnn_channels=list(arch("cnn_channels")),
            gru_hidden=int(arch("gru_hidden")),
            gru_layers=int(arch("gru_layers")),
            dropout=float(arch("dropout")),
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        self._h = None
        self.reset()

    # -- state --------------------------------------------------------------
    def reset(self) -> None:
        """Clear GRU temporal context (call at the start of a recording)."""
        self._h = self.model.init_hidden(1, self.device)

    # -- preprocessing ------------------------------------------------------
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Raw camera frame -> normalised float32 grayscale, downscaled.

        Matches RT-opto training: grayscale, resize with INTER_AREA by
        spatial_scale, then /255. A mono8 (H,W) frame is already grayscale.
        """
        if frame.ndim == 3 and frame.shape[2] == 3:
            code = cv2.COLOR_BGR2GRAY if self._color_is_bgr else cv2.COLOR_RGB2GRAY
            gray = cv2.cvtColor(frame, code)
        elif frame.ndim == 2:
            gray = frame
        else:
            raise ValueError(f"Unexpected frame shape {frame.shape}")

        if self.spatial_scale != 1.0:
            h, w = gray.shape[:2]
            new_w = int(w * self.spatial_scale)
            new_h = int(h * self.spatial_scale)
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return gray.astype(np.float32) / 255.0

    def set_color_is_bgr(self, is_bgr: bool) -> None:
        self._color_is_bgr = is_bgr

    def model_input_size(self, frame_h: int, frame_w: int) -> tuple[int, int]:
        return (int(frame_h * self.spatial_scale),
                int(frame_w * self.spatial_scale))

    # -- inference ----------------------------------------------------------
    @torch.inference_mode()
    def step(self, gray: np.ndarray) -> Prediction:
        """Run one streaming step on an already-preprocessed frame.

        Updates the internal GRU state. Split from preprocess() so the caller
        can time the two stages independently.
        """
        # (H,W) -> (B=1, T=1, C=1, H, W)
        tensor = (torch.from_numpy(gray)
                  .unsqueeze(0).unsqueeze(0).unsqueeze(0)
                  .to(self.device, non_blocking=True))
        logits, self._h = self.model(tensor, self._h)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        vec = logits.reshape(self.num_classes)
        probs = torch.softmax(vec, dim=0)
        pred = int(torch.argmax(vec).item())
        return Prediction(
            pred_class=pred,
            confidence=float(probs[pred].item()),
            logits=vec.detach().float().cpu().numpy(),
        )

    def classify(self, frame: np.ndarray) -> Prediction:
        """Convenience: preprocess + step in one call."""
        return self.step(self.preprocess(frame))

    @torch.inference_mode()
    def warmup(self, frame_h: int, frame_w: int, n: int = 10) -> None:
        """Prime CUDA kernels / allocator so the first real frame isn't slow."""
        dummy = np.zeros((frame_h, frame_w), dtype=np.uint8)
        for _ in range(n):
            self.classify(dummy)
        self.reset()
