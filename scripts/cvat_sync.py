"""CLI: sync MaskReview's review loop with a self-hosted CVAT task (P8 step 2).

Credentials come from the environment (never hard-coded):
    CVAT_HOST (default http://localhost:8080), CVAT_USER, CVAT_PASSWORD

Typical flow against task <id> whose frame 0 has a seed rectangle:
    python scripts/cvat_sync.py push   --task-id <id>
    # ... annotator fixes the flagged frames in CVAT, resolving the issues ...
    python scripts/cvat_sync.py export --task-id <id> --out corrected.zip

`push` = pull frames + seed box from CVAT -> run SAM2 -> import masks as COCO
annotations -> raise one CVAT issue per low-confidence/anomalous frame.
See docs/cvat_plugin.md.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cvat.client import CvatBridge  # noqa: E402
from cvat.export import interaction_report, queue_to_issues  # noqa: E402
from pipeline import (  # noqa: E402
    DEFAULT_SAM2_CHECKPOINT,
    DEFAULT_SAM2_MODEL_CFG,
    ReviewPolicy,
    build_review_queue,
)
from sam2_runner import Sam2VideoSegmenter  # noqa: E402


def _review_sidecar(task_id: int) -> Path:
    """Local file where `push` records the review estimate so `stats` can compare later."""
    return Path("outputs") / "cvat" / f"task_{task_id}_review.json"


def _bridge() -> CvatBridge:
    host = os.environ.get("CVAT_HOST", "http://localhost:8080")
    try:
        user, password = os.environ["CVAT_USER"], os.environ["CVAT_PASSWORD"]
    except KeyError as exc:
        raise SystemExit(f"Set the {exc} environment variable (CVAT_USER / CVAT_PASSWORD).") from exc
    return CvatBridge(host, (user, password))


def _build_queue(masks: dict[int, np.ndarray], frame_count: int, confidences) -> list[dict]:
    areas = [int((np.asarray(masks[i]) > 0).sum()) for i in range(frame_count)]
    p = ReviewPolicy()
    return build_review_queue(
        areas,
        masks=masks,
        confidences=confidences or None,
        jump_ratio=p.area_jump_ratio,
        center_jump_ratio=p.center_jump_ratio,
        iou_drop_threshold=p.iou_drop_threshold,
        edge_margin=p.edge_margin,
        area_decline_ratio=p.area_decline_ratio,
        low_confidence_threshold=p.low_confidence_threshold,
        collapse_low_confidence_runs=p.collapse_low_confidence_runs,
        oversized_area_ratio=p.oversized_area_ratio,
        oversized_baseline_multiple=p.oversized_baseline_multiple,
        oversized_max_fill_ratio=p.oversized_max_fill_ratio,
    )


def _mask_bboxes(masks: dict[int, np.ndarray]) -> dict[int, tuple[int, int, int, int]]:
    bboxes: dict[int, tuple[int, int, int, int]] = {}
    for idx, mask in masks.items():
        ys, xs = np.nonzero(np.asarray(mask) > 0)
        if xs.size:
            bboxes[int(idx)] = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    return bboxes


def cmd_push(args: argparse.Namespace) -> None:
    bridge = _bridge()
    frame_dir = Path(args.frame_dir or Path(tempfile.gettempdir()) / f"cvat_task_{args.task_id}_frames")
    pulled = bridge.pull_task(args.task_id, frame_dir, seed_frame=args.seed_frame)
    print(f"pulled task {pulled.task_id}: {pulled.frame_count} frames "
          f"{pulled.width}x{pulled.height}, job {pulled.job_id}, seed box {pulled.init_box_xyxy}")
    if pulled.init_box_xyxy is None:
        raise SystemExit(
            f"No rectangle on frame {args.seed_frame} of task {args.task_id}. "
            "Draw a seed box in CVAT and save first."
        )

    segmenter = Sam2VideoSegmenter(
        model_cfg=args.sam2_model_cfg,
        checkpoint_path=Path(args.sam2_checkpoint),
        device=args.device,
    )
    masks = segmenter.segment(
        pulled.frame_dir, pulled.init_box_xyxy, pulled.frame_count, init_frame_index=pulled.init_frame_index
    )
    queue = _build_queue(masks, pulled.frame_count, getattr(segmenter, "last_confidences", None))
    print(f"SAM2 propagated {pulled.frame_count} frames; {len(queue)} flagged for review")

    if args.via == "shapes":
        count = bridge.import_masks_native(args.task_id, masks, label=args.label)
        print(f"imported {count} masks as native CVAT mask shapes (editable)")
    else:
        bridge.import_masks_coco(args.task_id, masks, label=args.label, frame_names=pulled.frame_names)
        print(f"imported masks as COCO annotations ({sum((np.asarray(m) > 0).any() for m in masks.values())} non-empty)")

    issues = queue_to_issues(queue, min_score=args.min_score, bboxes=_mask_bboxes(masks))
    created = bridge.create_issues(args.task_id, issues)
    report = interaction_report(queue, created)
    print(f"created {created} CVAT issues (min_score={args.min_score}); "
          f"estimated_interactions={report['estimated_interactions']}")

    sidecar = _review_sidecar(args.task_id)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps({
        "task_id": args.task_id,
        "queued_frames": len(queue),
        "estimated_interactions": report["estimated_interactions"],
        "issues_created": created,
        "review_queue": queue,
    }, default=str), encoding="utf-8")
    print(f"wrote review estimate -> {sidecar}")


def cmd_stats(args: argparse.Namespace) -> None:
    sidecar = _review_sidecar(args.task_id)
    if not sidecar.exists():
        raise SystemExit(f"No review estimate at {sidecar}. Run `push --task-id {args.task_id}` first.")
    saved = json.loads(sidecar.read_text(encoding="utf-8"))
    counts = _bridge().count_issues(args.task_id)
    report = interaction_report(saved["review_queue"], counts["resolved"], issues_created=counts["total"])
    print(f"task {args.task_id} interaction report:")
    print(f"  queued frames        : {report['queued_frames']}")
    print(f"  estimated interactions: {report['estimated_interactions']}")
    print(f"  issues on task        : {report['issues_created']}")
    print(f"  issues resolved       : {report['resolved_issues']}")
    print(f"  review coverage       : {report['review_coverage']}")


def cmd_export(args: argparse.Namespace) -> None:
    out = _bridge().export_annotations(args.task_id, Path(args.out))
    print(f"exported corrected annotations -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync MaskReview with a CVAT task.")
    sub = parser.add_subparsers(dest="command", required=True)

    push = sub.add_parser("push", help="pull task -> run SAM2 -> push masks + review issues")
    push.add_argument("--task-id", type=int, required=True)
    push.add_argument("--seed-frame", type=int, default=0)
    push.add_argument("--label", default="object")
    push.add_argument("--min-score", type=float, default=0.4,
                      help="drop queue frames below this review_score from issues (damped/collapsed).")
    push.add_argument("--via", choices=["coco", "shapes"], default="coco",
                      help="how to import masks: coco (dataset import) or shapes (native editable mask shapes).")
    push.add_argument("--frame-dir", default=None, help="where to cache pulled frames.")
    push.add_argument("--device", default="cuda")
    push.add_argument("--sam2-model-cfg", default=DEFAULT_SAM2_MODEL_CFG)
    push.add_argument("--sam2-checkpoint", default=str(DEFAULT_SAM2_CHECKPOINT))
    push.set_defaults(func=cmd_push)

    stats = sub.add_parser("stats", help="compare the review estimate vs issues resolved in CVAT")
    stats.add_argument("--task-id", type=int, required=True)
    stats.set_defaults(func=cmd_stats)

    export = sub.add_parser("export", help="download the human-corrected annotations")
    export.add_argument("--task-id", type=int, required=True)
    export.add_argument("--out", default="cvat_corrected.zip")
    export.set_defaults(func=cmd_export)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
