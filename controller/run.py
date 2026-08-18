"""CLI entry point for the real-time closed-loop optogenetics controller.

    python -m controller.run configs/controller_config.yaml
    python -m controller.run configs/controller_config.yaml --source synthetic \
        --dry-run --max-frames 300

Run from the repository root (the directory containing Campy-main/ and
RT-opto-main/).
"""

from __future__ import annotations

import argparse

from .config import ControllerConfig
from .pipeline import RealTimeController


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RT-opto closed-loop controller")
    p.add_argument("config", help="Path to controller YAML config")
    p.add_argument("--source", choices=["flir", "video", "synthetic"],
                   help="Override frame source")
    p.add_argument("--source-video", help="Video path (for --source video)")
    p.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"],
                   help="Override inference device")
    p.add_argument("--dry-run", action="store_true",
                   help="Do not open serial ports; log stim/trigger commands")
    p.add_argument("--rec-time", type=float, help="Recording time in seconds")
    p.add_argument("--max-frames", type=int, help="Stop after N frames")
    p.add_argument("--output", help="Override output folder")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = ControllerConfig.from_yaml(args.config)

    if args.source is not None:
        config.source = args.source
    if args.source_video is not None:
        config.source_video = args.source_video
    if args.device is not None:
        config.device = args.device
    if args.dry_run:
        config.serial_dry_run = True
    if args.rec_time is not None:
        config.rec_time_sec = args.rec_time
    if args.max_frames is not None:
        config.max_frames = args.max_frames
    if args.output is not None:
        config.output_folder = args.output

    controller = RealTimeController(config)
    controller.setup()
    controller.run()


if __name__ == "__main__":
    main()
