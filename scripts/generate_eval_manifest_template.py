from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from manifest_template import generate_eval_manifest_template


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a MaskReview eval manifest template.")
    parser.add_argument("--video-dir", type=Path, default=Path("data/eval_videos"))
    parser.add_argument("--output", type=Path, default=Path("data/eval_manifest.template.json"))
    parser.add_argument("--first-frame-dir", type=Path, default=Path("data/eval_first_frames"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = generate_eval_manifest_template(
        video_dir=args.video_dir,
        output_path=args.output,
        first_frame_dir=args.first_frame_dir,
        overwrite=args.overwrite,
    )
    print(f"template: {output_path}")
    print("fill init_box_xyxy before running scripts/run_evaluation.py")


if __name__ == "__main__":
    main()
