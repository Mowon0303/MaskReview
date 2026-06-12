from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Optional

from corrections import save_correction
from pipeline import (
    CorrectionPropagationRequest,
    PromptableVideoSegmentationPipeline,
    ReviewPolicy,
    VideoSegmentationRequest,
)


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    video_path: Path
    object_name: str
    init_box_xyxy: tuple[int, int, int, int]
    expected_challenge: str
    notes: str = ""
    expected_review_frames: Optional[tuple[int, ...]] = None
    corrections: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class EvalConfig:
    output_dir: Path = Path("outputs/evaluation")
    fixed_interval: int = 10
    sam2_model_cfg: str = "configs/sam2.1/sam2.1_hiera_t.yaml"
    sam2_checkpoint: Path = Path("checkpoints/sam2.1_hiera_tiny.pt")
    device: str = "cuda"
    vos_optimized: bool = False
    review_policy: ReviewPolicy = field(default_factory=ReviewPolicy)


@dataclass(frozen=True)
class EvalSummary:
    rows: list[dict[str, Any]]
    csv_path: Path
    report_path: Path


@dataclass(frozen=True)
class CalibrationSummary:
    rows: list[dict[str, Any]]
    csv_path: Path
    report_path: Path


def load_eval_manifest(manifest_path: Path) -> list[EvalCase]:
    manifest_path = Path(manifest_path)
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_cases = raw.get("cases", raw) if isinstance(raw, dict) else raw
    if not isinstance(raw_cases, list):
        raise ValueError("Eval manifest must be a list or an object with a 'cases' list.")

    cases = [parse_eval_case(item, manifest_path.parent) for item in raw_cases]
    case_ids = [case.case_id for case in cases]
    duplicate_ids = sorted({case_id for case_id in case_ids if case_ids.count(case_id) > 1})
    if duplicate_ids:
        raise ValueError(f"Duplicate eval case ids: {', '.join(duplicate_ids)}")
    return cases


def parse_eval_case(raw_case: dict[str, Any], base_dir: Path) -> EvalCase:
    case_id = str(raw_case.get("id") or raw_case.get("case_id") or "").strip()
    if not case_id:
        raise ValueError("Eval case is missing id.")

    raw_video_path = raw_case.get("video_path")
    if not raw_video_path:
        raise ValueError(f"Eval case {case_id} is missing video_path.")
    video_path = Path(str(raw_video_path))
    if not video_path.is_absolute():
        video_path = base_dir / video_path

    init_box = parse_manifest_box(raw_case.get("init_box_xyxy"), field_name="init_box_xyxy")
    expected_review_frames = parse_optional_frame_indices(
        raw_case,
        field_names=(
            "expected_review_frames",
            "expected_review_frame_indices",
            "expected_failure_frames",
            "expected_drift_frames",
        ),
    )
    corrections = tuple(
        normalize_manifest_correction(correction)
        for correction in raw_case.get("corrections", [])
    )
    return EvalCase(
        case_id=sanitize_case_id(case_id),
        video_path=video_path,
        object_name=str(raw_case.get("object_name", "")),
        init_box_xyxy=init_box,
        expected_challenge=str(raw_case.get("expected_challenge", "unspecified")),
        notes=str(raw_case.get("notes", "")),
        expected_review_frames=expected_review_frames,
        corrections=corrections,
    )


def run_evaluation(
    manifest_path: Path,
    config: Optional[EvalConfig] = None,
    pipeline: Optional[PromptableVideoSegmentationPipeline] = None,
) -> EvalSummary:
    config = config or EvalConfig()
    pipeline = pipeline or PromptableVideoSegmentationPipeline()
    cases = load_eval_manifest(manifest_path)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        evaluate_case(case=case, config=config, pipeline=pipeline)
        for case in cases
    ]
    csv_path = config.output_dir / "evaluation_results.csv"
    report_path = config.output_dir / "evaluation_report.md"
    write_results_csv(rows, csv_path)
    write_report(rows, report_path, manifest_path, config)
    return EvalSummary(rows=rows, csv_path=csv_path, report_path=report_path)


def evaluate_case(
    case: EvalCase,
    config: EvalConfig,
    pipeline: PromptableVideoSegmentationPipeline,
) -> dict[str, Any]:
    if not case.video_path.exists():
        raise FileNotFoundError(f"Eval video does not exist: {case.video_path}")

    result = pipeline.run(
        VideoSegmentationRequest(
            video_path=case.video_path,
            init_box_xyxy=case.init_box_xyxy,
            output_dir=config.output_dir,
            run_id=case.case_id,
            sam2_model_cfg=config.sam2_model_cfg,
            sam2_checkpoint=config.sam2_checkpoint,
            device=config.device,
            vos_optimized=config.vos_optimized,
            review_policy=config.review_policy,
        )
    )
    metrics_before = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    fixed_interval_frames = fixed_interval_review_indices(result.frame_count, config.fixed_interval)
    queue_label_metrics = calculate_label_metrics(
        predicted_frames=result.review_frame_indices,
        expected_frames=case.expected_review_frames,
        prefix="queue",
    )
    fixed_label_metrics = calculate_label_metrics(
        predicted_frames=fixed_interval_frames,
        expected_frames=case.expected_review_frames,
        prefix="fixed_interval",
    )
    row: dict[str, Any] = {
        "case_id": case.case_id,
        "video_path": str(case.video_path),
        "object_name": case.object_name,
        "expected_challenge": case.expected_challenge,
        "notes": case.notes,
        "review_policy_config": json.dumps(config.review_policy.to_dict(), sort_keys=True),
        "frame_count": result.frame_count,
        "fps": result.fps,
        "duration_seconds": metrics_before.get("video_duration_seconds", 0.0),
        "expected_review_frames_labeled": case.expected_review_frames is not None,
        "expected_review_frames": (
            len(case.expected_review_frames) if case.expected_review_frames is not None else ""
        ),
        "expected_review_frame_indices": (
            json.dumps(list(case.expected_review_frames))
            if case.expected_review_frames is not None
            else ""
        ),
        "queue_review_frames": len(result.review_frame_indices),
        "queue_review_frame_indices": json.dumps(result.review_frame_indices),
        "queue_reason_counts": json.dumps(metrics_before.get("review_reason_counts", {}), sort_keys=True),
        "queue_estimated_interactions": result.estimated_min_interactions,
        "queue_interactions_per_video_minute": result.interactions_per_video_minute,
        "fixed_interval": config.fixed_interval,
        "fixed_interval_review_frames": len(fixed_interval_frames),
        "fixed_interval_frame_indices": json.dumps(fixed_interval_frames),
        "saved_interactions_vs_fixed_interval": (
            len(fixed_interval_frames) - result.estimated_min_interactions
        ),
        "baseline_overlay_path": str(result.overlay_video_path),
        "baseline_metrics_path": str(result.metrics_path),
        "run_dir": str(result.run_dir),
        "corrections_applied": 0,
        "after_review_frames": "",
        "after_reason_counts": "",
        "actual_interactions": "",
        "actual_interactions_per_video_minute": "",
        "after_overlay_path": "",
        "after_metrics_path": "",
        "comparison_path": "",
    }
    row.update(queue_label_metrics)
    row.update(fixed_label_metrics)

    if case.corrections:
        for correction in case.corrections:
            save_correction(result.run_dir, correction)
        corrected = pipeline.run_with_corrections(
            CorrectionPropagationRequest(
                run_dir=result.run_dir,
                video_path=case.video_path,
                init_box_xyxy=case.init_box_xyxy,
                sam2_model_cfg=config.sam2_model_cfg,
                sam2_checkpoint=config.sam2_checkpoint,
                device=config.device,
                vos_optimized=config.vos_optimized,
                review_policy=config.review_policy,
            )
        )
        row.update(
            {
                "corrections_applied": len(case.corrections),
                "after_review_frames": len(corrected.review_queue),
                "after_reason_counts": json.dumps(
                    json.loads(corrected.metrics_path.read_text(encoding="utf-8")).get(
                        "review_reason_counts",
                        {},
                    ),
                    sort_keys=True,
                ),
                "actual_interactions": corrected.actual_interactions,
                "actual_interactions_per_video_minute": corrected.interactions_per_video_minute,
                "after_overlay_path": str(corrected.overlay_video_path),
                "after_metrics_path": str(corrected.metrics_path),
                "comparison_path": str(corrected.comparison_path),
            }
        )

    return row


def fixed_interval_review_indices(frame_count: int, interval: int) -> list[int]:
    if interval <= 0:
        raise ValueError("fixed_interval must be greater than zero.")
    return list(range(interval, frame_count, interval))


def write_results_csv(rows: list[dict[str, Any]], csv_path: Path) -> None:
    if not rows:
        raise ValueError("Cannot write evaluation CSV without rows.")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    rows: list[dict[str, Any]],
    report_path: Path,
    manifest_path: Path,
    config: EvalConfig,
) -> None:
    total_cases = len(rows)
    cases_with_corrections = sum(1 for row in rows if int(row["corrections_applied"]) > 0)
    total_queue_interactions = sum(int(row["queue_estimated_interactions"]) for row in rows)
    total_fixed_interactions = sum(int(row["fixed_interval_review_frames"]) for row in rows)
    total_saved = total_fixed_interactions - total_queue_interactions
    label_summary = summarize_label_metrics(rows)

    lines = [
        "# MaskReview Evaluation Report",
        "",
        f"- Manifest: `{manifest_path}`",
        f"- Cases: {total_cases}",
        f"- Cases with saved corrections: {cases_with_corrections}",
        f"- Fixed interval baseline: every {config.fixed_interval} frames",
        f"- Queue estimated interactions: {total_queue_interactions}",
        f"- Fixed interval interactions: {total_fixed_interactions}",
        f"- Saved interactions vs fixed interval: {total_saved}",
        f"- Labeled cases: {label_summary['labeled_cases']}",
    ]
    if label_summary["labeled_cases"]:
        lines.extend(
            [
                f"- Expected review frames: {label_summary['total_expected_review_frames']}",
                f"- Queue label precision/recall/F1: {label_summary['queue_precision']} / {label_summary['queue_recall']} / {label_summary['queue_f1']}",
                f"- Fixed interval label precision/recall/F1: {label_summary['fixed_interval_precision']} / {label_summary['fixed_interval_recall']} / {label_summary['fixed_interval_f1']}",
            ]
        )
    lines.extend(
        [
            "",
            "## Case Results",
            "",
            "| Case | Challenge | Expected | Queue frames | Queue hit/miss | Queue F1 | Fixed frames | Saved | Corrections | After queue |",
            "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        after_review_frames = row["after_review_frames"] if row["after_review_frames"] != "" else "n/a"
        expected = row["expected_review_frames"] if row["expected_review_frames"] != "" else "n/a"
        queue_hit_miss = format_hit_miss(row, "queue")
        queue_f1 = row["queue_f1"] if row["queue_f1"] != "" else "n/a"
        lines.append(
            "| {case_id} | {challenge} | {expected} | {queue} | {hit_miss} | {queue_f1} | {fixed} | {saved} | {corrections} | {after} |".format(
                case_id=row["case_id"],
                challenge=row["expected_challenge"],
                expected=expected,
                queue=row["queue_review_frames"],
                hit_miss=queue_hit_miss,
                queue_f1=queue_f1,
                fixed=row["fixed_interval_review_frames"],
                saved=row["saved_interactions_vs_fixed_interval"],
                corrections=row["corrections_applied"],
                after=after_review_frames,
            )
        )

    lines.extend(
        [
            "",
            "## Queue Reason Counts",
            "",
            "| Case | Before reasons | After reasons |",
            "| --- | --- | --- |",
        ]
    )
    for row in rows:
        after_reasons = row.get("after_reason_counts") or "n/a"
        lines.append(f"| {row['case_id']} | `{row['queue_reason_counts']}` | `{after_reasons}` |")

    lines.extend(
        [
            "",
            "## Interpretation Checklist",
            "",
            "- Queue frames should be materially fewer than fixed-interval review frames on stable videos.",
            "- Hard videos should enter the queue near real drift, occlusion, or target-loss moments.",
            "- When `expected_review_frames` is labeled, prioritize recall before reducing false positives.",
            "- After-correction runs should reduce the queue or restore mask stability when the target remains visible.",
            "- Cases without corrections only validate detection and cost estimates, not recovery.",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def default_calibration_policies() -> list[tuple[str, ReviewPolicy]]:
    return [
        (
            "sensitive",
            ReviewPolicy(
                area_jump_ratio=0.45,
                center_jump_ratio=0.15,
                iou_drop_threshold=0.35,
                edge_margin=0,
                area_decline_ratio=0.25,
            ),
        ),
        ("default", ReviewPolicy()),
        (
            "conservative",
            ReviewPolicy(
                area_jump_ratio=0.75,
                center_jump_ratio=0.35,
                iou_drop_threshold=0.1,
                edge_margin=0,
                area_decline_ratio=0.5,
            ),
        ),
    ]


def run_threshold_calibration(
    manifest_path: Path,
    config: Optional[EvalConfig] = None,
    policies: Optional[list[tuple[str, ReviewPolicy]]] = None,
    pipeline_factory: Optional[Callable[[], PromptableVideoSegmentationPipeline]] = None,
) -> CalibrationSummary:
    config = config or EvalConfig()
    policies = policies or default_calibration_policies()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for policy_name, review_policy in policies:
        policy_config = replace(
            config,
            output_dir=config.output_dir / policy_name,
            review_policy=review_policy,
        )
        pipeline = pipeline_factory() if pipeline_factory else PromptableVideoSegmentationPipeline()
        summary = run_evaluation(
            manifest_path=manifest_path,
            config=policy_config,
            pipeline=pipeline,
        )
        rows.append(summarize_calibration_policy(policy_name, review_policy, summary.rows, summary.report_path))

    csv_path = config.output_dir / "threshold_calibration.csv"
    report_path = config.output_dir / "threshold_calibration_report.md"
    write_results_csv(rows, csv_path)
    write_threshold_calibration_report(rows, report_path, manifest_path)
    return CalibrationSummary(rows=rows, csv_path=csv_path, report_path=report_path)


def summarize_calibration_policy(
    policy_name: str,
    review_policy: ReviewPolicy,
    eval_rows: list[dict[str, Any]],
    eval_report_path: Path,
) -> dict[str, Any]:
    total_cases = len(eval_rows)
    total_queue_frames = sum(int(row["queue_review_frames"]) for row in eval_rows)
    total_queue_interactions = sum(int(row["queue_estimated_interactions"]) for row in eval_rows)
    total_fixed_frames = sum(int(row["fixed_interval_review_frames"]) for row in eval_rows)
    total_saved = total_fixed_frames - total_queue_interactions
    corrected_rows = [row for row in eval_rows if str(row["after_review_frames"]) != ""]
    total_after_frames = sum(int(row["after_review_frames"]) for row in corrected_rows)
    total_actual_interactions = sum(
        int(row["actual_interactions"])
        for row in corrected_rows
        if str(row["actual_interactions"]) != ""
    )
    label_summary = summarize_label_metrics(eval_rows)

    return {
        "policy_name": policy_name,
        "review_policy_config": json.dumps(review_policy.to_dict(), sort_keys=True),
        "cases": total_cases,
        "total_queue_frames": total_queue_frames,
        "avg_queue_frames_per_case": round(total_queue_frames / total_cases, 3) if total_cases else 0.0,
        "total_queue_estimated_interactions": total_queue_interactions,
        "total_fixed_interval_frames": total_fixed_frames,
        "saved_interactions_vs_fixed_interval": total_saved,
        "cases_with_corrections": len(corrected_rows),
        "total_after_queue_frames": total_after_frames,
        "total_actual_interactions": total_actual_interactions,
        "labeled_cases": label_summary["labeled_cases"],
        "total_expected_review_frames": label_summary["total_expected_review_frames"],
        "queue_label_hits": label_summary["queue_label_hits"],
        "queue_false_positives": label_summary["queue_false_positives"],
        "queue_missed_expected_frames": label_summary["queue_missed_expected_frames"],
        "queue_precision": label_summary["queue_precision"],
        "queue_recall": label_summary["queue_recall"],
        "queue_f1": label_summary["queue_f1"],
        "fixed_interval_precision": label_summary["fixed_interval_precision"],
        "fixed_interval_recall": label_summary["fixed_interval_recall"],
        "fixed_interval_f1": label_summary["fixed_interval_f1"],
        "eval_report_path": str(eval_report_path),
    }


def write_threshold_calibration_report(
    rows: list[dict[str, Any]],
    report_path: Path,
    manifest_path: Path,
) -> None:
    lines = [
        "# MaskReview Threshold Calibration",
        "",
        f"- Manifest: `{manifest_path}`",
        f"- Policies evaluated: {len(rows)}",
        "",
        "| Policy | Queue frames | Fixed frames | Saved | Labeled cases | Queue P/R/F1 | Fixed P/R/F1 | Config |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {policy} | {queue} | {fixed} | {saved} | {labeled} | {queue_prf} | {fixed_prf} | `{config}` |".format(
                policy=row["policy_name"],
                queue=row["total_queue_frames"],
                fixed=row["total_fixed_interval_frames"],
                saved=row["saved_interactions_vs_fixed_interval"],
                labeled=row["labeled_cases"],
                queue_prf=format_prf(row, "queue"),
                fixed_prf=format_prf(row, "fixed_interval"),
                config=row["review_policy_config"],
            )
        )

    lines.extend(
        [
            "",
            "## How To Choose A Policy",
            "",
            "- Start with `default` unless real videos show too many missed drifts.",
            "- Prefer `sensitive` when missing drift is more costly than reviewing extra frames.",
            "- Prefer `conservative` only when review volume is too high and drift misses are acceptable.",
            "- With `expected_review_frames` labels, choose the least expensive policy that keeps recall high.",
            "- Full mask IoU/J&F is still pending for final quality scoring.",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def normalize_manifest_correction(raw_correction: dict[str, Any]) -> dict[str, Any]:
    correction = dict(raw_correction)
    correction_type = str(correction.get("type", ""))
    correction["frame_index"] = int(correction["frame_index"])
    correction.setdefault("estimated_interactions", 1)

    if correction_type in {"positive_point", "negative_point"}:
        if "points" not in correction:
            point = correction.get("point_xy")
            if point is None:
                raise ValueError(f"Point correction missing points or point_xy: {raw_correction}")
            x, y = parse_manifest_point(point)
            label = "positive" if correction_type == "positive_point" else "negative"
            correction["points"] = [{"x": x, "y": y, "label": label}]
        return correction

    if correction_type == "tight_box":
        correction["box_xyxy"] = list(parse_manifest_box(correction.get("box_xyxy"), "box_xyxy"))
        return correction

    raise ValueError(f"Unsupported correction type in eval manifest: {correction_type}")


def parse_manifest_point(raw_point: Any) -> tuple[int, int]:
    if isinstance(raw_point, str):
        values = [int(value.strip()) for value in raw_point.split(",") if value.strip()]
    else:
        values = [int(value) for value in raw_point]
    if len(values) != 2:
        raise ValueError("point_xy must contain exactly two values.")
    return values[0], values[1]


def parse_optional_frame_indices(
    raw_case: dict[str, Any],
    field_names: tuple[str, ...],
) -> Optional[tuple[int, ...]]:
    for field_name in field_names:
        if field_name in raw_case:
            raw_value = raw_case[field_name]
            if raw_value is None:
                return None
            return tuple(parse_manifest_frame_indices(raw_value, field_name))
    return None


def parse_manifest_frame_indices(raw_value: Any, field_name: str) -> list[int]:
    if isinstance(raw_value, str):
        values: list[int] = []
        for chunk in raw_value.split(","):
            token = chunk.strip()
            if not token:
                continue
            if "-" in token:
                start_raw, end_raw = [part.strip() for part in token.split("-", maxsplit=1)]
                start = int(start_raw)
                end = int(end_raw)
                if start > end:
                    raise ValueError(f"{field_name} range must satisfy start <= end.")
                values.extend(range(start, end + 1))
            else:
                values.append(int(token))
    else:
        values = [int(value) for value in raw_value]

    if any(value < 0 for value in values):
        raise ValueError(f"{field_name} cannot contain negative frame indices.")
    return sorted(set(values))


def parse_manifest_box(raw_box: Any, field_name: str) -> tuple[int, int, int, int]:
    if isinstance(raw_box, str):
        values = [int(value.strip()) for value in raw_box.split(",") if value.strip()]
    else:
        values = [int(value) for value in raw_box]
    if len(values) != 4:
        raise ValueError(f"{field_name} must contain exactly four values.")
    x1, y1, x2, y2 = values
    if x1 >= x2 or y1 >= y2:
        raise ValueError(f"{field_name} must satisfy x1 < x2 and y1 < y2.")
    return x1, y1, x2, y2


def calculate_label_metrics(
    predicted_frames: list[int],
    expected_frames: Optional[tuple[int, ...]],
    prefix: str,
) -> dict[str, Any]:
    if expected_frames is None:
        return {
            f"{prefix}_label_hits": "",
            f"{prefix}_false_positives": "",
            f"{prefix}_missed_expected_frames": "",
            f"{prefix}_precision": "",
            f"{prefix}_recall": "",
            f"{prefix}_f1": "",
        }

    predicted = set(predicted_frames)
    expected = set(expected_frames)
    hits = predicted & expected
    false_positives = predicted - expected
    missed = expected - predicted
    precision = divide_or_perfect(len(hits), len(predicted))
    recall = divide_or_perfect(len(hits), len(expected))
    return {
        f"{prefix}_label_hits": len(hits),
        f"{prefix}_false_positives": len(false_positives),
        f"{prefix}_missed_expected_frames": len(missed),
        f"{prefix}_precision": round(precision, 3),
        f"{prefix}_recall": round(recall, 3),
        f"{prefix}_f1": round(calculate_f1(precision, recall), 3),
    }


def summarize_label_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labeled_rows = [row for row in rows if row.get("expected_review_frames_labeled") is True]
    summary: dict[str, Any] = {
        "labeled_cases": len(labeled_rows),
        "total_expected_review_frames": sum(
            int(row["expected_review_frames"])
            for row in labeled_rows
        ),
    }
    for prefix in ("queue", "fixed_interval"):
        if not labeled_rows:
            summary.update(
                {
                    f"{prefix}_label_hits": 0,
                    f"{prefix}_false_positives": 0,
                    f"{prefix}_missed_expected_frames": 0,
                    f"{prefix}_precision": "",
                    f"{prefix}_recall": "",
                    f"{prefix}_f1": "",
                }
            )
            continue
        hits = sum(int(row[f"{prefix}_label_hits"]) for row in labeled_rows)
        false_positives = sum(int(row[f"{prefix}_false_positives"]) for row in labeled_rows)
        missed = sum(int(row[f"{prefix}_missed_expected_frames"]) for row in labeled_rows)
        predicted = hits + false_positives
        expected = hits + missed
        precision = divide_or_perfect(hits, predicted)
        recall = divide_or_perfect(hits, expected)
        summary.update(
            {
                f"{prefix}_label_hits": hits,
                f"{prefix}_false_positives": false_positives,
                f"{prefix}_missed_expected_frames": missed,
                f"{prefix}_precision": round(precision, 3),
                f"{prefix}_recall": round(recall, 3),
                f"{prefix}_f1": round(calculate_f1(precision, recall), 3),
            }
        )
    return summary


def divide_or_perfect(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 1.0
    return numerator / denominator


def calculate_f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def format_hit_miss(row: dict[str, Any], prefix: str) -> str:
    if row[f"{prefix}_label_hits"] == "":
        return "n/a"
    return "{hits}/{missed}".format(
        hits=row[f"{prefix}_label_hits"],
        missed=row[f"{prefix}_missed_expected_frames"],
    )


def format_prf(row: dict[str, Any], prefix: str) -> str:
    if row.get("labeled_cases") == 0:
        return "n/a"
    if row[f"{prefix}_precision"] == "":
        return "n/a"
    return "{precision}/{recall}/{f1}".format(
        precision=row[f"{prefix}_precision"],
        recall=row[f"{prefix}_recall"],
        f1=row[f"{prefix}_f1"],
    )


def sanitize_case_id(raw_case_id: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_case_id.strip())
    return value.strip("._-") or "case"
