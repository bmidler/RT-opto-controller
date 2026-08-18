"""Make the sibling campy and RT-opto codebases importable.

The controller deliberately lives outside both repos so they stay pristine
and can be updated independently. We add them to sys.path at import time.

    Campy-main/  -> provides the `campy` package (campy.cameras.flir, ...)
    RT-opto-main/ -> provides top-level `model.py` (VideoClassifier)
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CAMPY_ROOT = REPO_ROOT / "Campy-main"
RTOPTO_ROOT = REPO_ROOT / "RT-opto-main"


def setup_paths() -> None:
    for p in (CAMPY_ROOT, RTOPTO_ROOT):
        sp = str(p)
        if p.is_dir() and sp not in sys.path:
            sys.path.insert(0, sp)
