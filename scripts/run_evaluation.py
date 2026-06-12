from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluation import EvalConfig, run_evaluation, run_threshold_calibration
from pipeline import DEFAULT_SAM2_CHECKPOINT, DEFAULT_SAM2_MODEL_CFG, ReviewPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MaskReview evaluation harness.")
    parser.add_argument("manifest", type=Path, help="Path to eval manifest JSON.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--fixed-interval", type=int, default=10)
    parser.add_argument("--sam2-model-cfg", default=DEFAULT_SAM2_MODEL_CFG)
    parser.add_argument("--sam2-checkpoint", type=Path, default=DEFAULT_SAM2_CHECKPOINT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vos-optimized", action="store_true")
    parser.add_argument("--area-jump-ratio", type=float, default=0.6)
    parser.add_argument("--center-jump-ratio", type=float, default=0.25)
    parser.add_argument("--iou-drop-threshold", type=float, default=0.2)
    parser.add_argument("--edge-margin", type=int, default=0)
    parser.add_argument("--area-decline-ratio", type=float, default=0.35)
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Run sensitive/default/conservative threshold calibration instead of one policy.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = EvalConfig(
        output_dir=args.output_dir,
        fixed_interval=args.fixed_interval,
        sam2_model_cfg=args.sam2_model_cfg,
        sam2_checkpoint=args.sam2_checkpoint,
        device=args.device,
        vos_optimized=args.vos_optimized,
        review_policy=ReviewPolicy(
            area_jump_ratio=args.area_jump_ratio,
            center_jump_ratio=args.center_jump_ratio,
            iou_drop_threshold=args.iou_drop_threshold,
            edge_margin=args.edge_margin,
            area_decline_ratio=args.area_decline_ratio,
        ),
    )
    if args.calibrate:
        calibration = run_threshold_calibration(manifest_path=args.manifest, config=config)
        print(f"policies: {len(calibration.rows)}")
        print(f"csv: {calibration.csv_path}")
        print(f"report: {calibration.report_path}")
        return

    summary = run_evaluation(manifest_path=args.manifest, config=config)
    print(f"cases: {len(summary.rows)}")
    print(f"csv: {summary.csv_path}")
    print(f"report: {summary.report_path}")


if __name__ == "__main__":
    main()
