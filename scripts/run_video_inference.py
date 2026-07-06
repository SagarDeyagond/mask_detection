#!/usr/bin/env python3
"""CLI entry point for real-time face mask detection.

Usage:
    python scripts/run_video_inference.py                # default webcam
    python scripts/run_video_inference.py --source video.avi
    python scripts/run_video_inference.py --confidence 0.5
"""

import argparse
import logging
import sys
from pathlib import Path

# Allow running the script directly from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mask_detection.config import DetectorConfig  # noqa: E402
from mask_detection.pipeline import MaskDetectionPipeline  # noqa: E402
from mask_detection.video import VideoMaskDetector  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Real-time face mask detection on a video stream."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index or path to a video file (default: 0, the webcam).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.8,
        help="Minimum face detection confidence (default: 0.8).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=400,
        help="Display width in pixels (default: 400).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the video mask detector and return an exit code."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse_args(argv)

    # A purely numeric source is a camera index; anything else is a path.
    source: int | str = int(args.source) if args.source.isdigit() else args.source

    config = DetectorConfig(face_confidence=args.confidence)
    pipeline = MaskDetectionPipeline(config)
    detector = VideoMaskDetector(
        pipeline=pipeline, source=source, display_width=args.width
    )

    try:
        detector.run()
    except (FileNotFoundError, RuntimeError) as exc:
        logging.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
