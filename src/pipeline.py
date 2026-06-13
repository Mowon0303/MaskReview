from __future__ import annotations

import json
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Protocol

import numpy as np

from corrections import corrections_path, load_corrections
from sam2_runner import Sam2VideoSegmenter
from video_io import (
    VideoMetadata,
    extract_video_frames,
    render_overlay_video,
    save_mask,
)


DEFAULT_SAM2_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"
DEFAULT_SAM2_CHECKPOINT = Path("checkpoints/sam2.1_hiera_tiny.pt")


@dataclass(frozen=True)
class ReviewPolicy:
    area_jump_ratio: float = 0.6
    center_jump_ratio: float = 0.25
    iou_drop_threshold: float = 0.2
    edge_margin: int = 0
    area_decline_ratio: float = 0.35
    # Flag frames whose SAM2 confidence (sigmoid of peak object logit) is below this.
    # Only applied when the segmenter exposes per-frame confidence.
    low_confidence_threshold: float = 0.5
    # Collapse a run of consecutive empty + low-confidence frames (a likely object
    # absence/loss that SAM2 itself is unsure about) into a single onset interaction,
    # instead of charging one correction per frame. Keeps the frames visible in the
    # queue but stops disappearance from inflating the interaction estimate.
    collapse_low_confidence_runs: bool = True

    def to_dict(self) -> dict[str, float | int]:
        return {
            "area_jump_ratio": self.area_jump_ratio,
            "center_jump_ratio": self.center_jump_ratio,
            "iou_drop_threshold": self.iou_drop_threshold,
            "edge_margin": self.edge_margin,
            "area_decline_ratio": self.area_decline_ratio,
            "low_confidence_threshold": self.low_confidence_threshold,
            "collapse_low_confidence_runs": self.collapse_low_confidence_runs,
        }


@dataclass
class VideoSegmentationRequest:
    video_path: Path
    text_prompt: Optional[str] = None
    init_box_xyxy: Optional[tuple[int, int, int, int]] = None
    output_dir: Path = Path("outputs")
    run_id: Optional[str] = None
    sam2_model_cfg: str = DEFAULT_SAM2_MODEL_CFG
    sam2_checkpoint: Path = DEFAULT_SAM2_CHECKPOINT
    device: str = "cuda"
    vos_optimized: bool = False
    review_policy: ReviewPolicy = field(default_factory=ReviewPolicy)


@dataclass
class VideoSegmentationResult:
    overlay_video_path: Path
    mask_dir: Path
    review_frame_indices: list[int]
    review_queue_path: Path
    review_queue: list[dict[str, object]]
    run_dir: Path
    metrics_path: Path
    frame_count: int
    fps: float
    estimated_min_interactions: int
    interactions_per_video_minute: float


@dataclass
class CorrectionPropagationRequest:
    run_dir: Path
    video_path: Path
    init_box_xyxy: tuple[int, int, int, int]
    sam2_model_cfg: str = DEFAULT_SAM2_MODEL_CFG
    sam2_checkpoint: Path = DEFAULT_SAM2_CHECKPOINT
    device: str = "cuda"
    vos_optimized: bool = False
    review_policy: ReviewPolicy = field(default_factory=ReviewPolicy)


@dataclass
class CorrectionPropagationResult:
    overlay_video_path: Path
    mask_dir: Path
    review_queue_path: Path
    review_queue: list[dict[str, object]]
    metrics_path: Path
    comparison_path: Path
    corrections: list[dict[str, Any]]
    frame_count: int
    fps: float
    actual_interactions: int
    interactions_per_video_minute: float


class VideoSegmenter(Protocol):
    def segment(
        self,
        frame_dir: Path,
        init_box_xyxy: tuple[int, int, int, int],
        frame_count: int,
    ) -> dict[int, np.ndarray]:
        ...


class CorrectionAwareVideoSegmenter(VideoSegmenter, Protocol):
    def segment_with_corrections(
        self,
        frame_dir: Path,
        init_box_xyxy: tuple[int, int, int, int],
        frame_count: int,
        corrections: list[dict[str, Any]],
    ) -> dict[int, np.ndarray]:
        ...


class PromptableVideoSegmentationPipeline:
    """Run SAM2 propagation and build a low-confidence review queue."""

    def __init__(self, segmenter: Optional[VideoSegmenter] = None) -> None:
        self.segmenter = segmenter

    def load_models(self) -> None:
        """Kept for backwards compatibility; models are loaded lazily in run()."""

    def run(self, request: VideoSegmentationRequest) -> VideoSegmentationResult:
        if request.init_box_xyxy is None:
            raise ValueError("First version only supports an initial box in x1,y1,x2,y2 format.")
        if request.text_prompt:
            raise ValueError("Text prompts are planned for phase 2. Use box_xyxy for this demo.")

        video_path = Path(request.video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video does not exist: {video_path}")

        run_id = request.run_id or make_run_id(video_path)
        run_dir = Path(request.output_dir) / run_id
        frame_dir = run_dir / "frames"
        mask_dir = run_dir / "masks"
        overlay_path = run_dir / "overlay.mp4"
        metrics_path = run_dir / "metrics.json"
        review_queue_path = run_dir / "review_queue.json"
        mask_dir.mkdir(parents=True, exist_ok=True)

        metadata = extract_video_frames(video_path, frame_dir)
        validate_box_xyxy(request.init_box_xyxy, metadata)

        segmenter = self.segmenter or Sam2VideoSegmenter(
            model_cfg=request.sam2_model_cfg,
            checkpoint_path=request.sam2_checkpoint,
            device=request.device,
            vos_optimized=request.vos_optimized,
        )
        mask_by_frame = segmenter.segment(
            frame_dir=frame_dir,
            init_box_xyxy=request.init_box_xyxy,
            frame_count=metadata.frame_count,
        )
        confidences = getattr(segmenter, "last_confidences", None) or None

        masks: dict[int, np.ndarray] = {}
        mask_areas: list[int] = []
        for frame_idx in range(metadata.frame_count):
            mask = normalize_mask(mask_by_frame.get(frame_idx), metadata)
            masks[frame_idx] = mask
            mask_areas.append(int(mask.sum()))
            save_mask(mask, mask_dir / f"{frame_idx:06d}.png")

        review_queue = build_review_queue(
            mask_areas,
            frame_dir=frame_dir,
            mask_dir=mask_dir,
            masks=masks,
            jump_ratio=request.review_policy.area_jump_ratio,
            center_jump_ratio=request.review_policy.center_jump_ratio,
            iou_drop_threshold=request.review_policy.iou_drop_threshold,
            edge_margin=request.review_policy.edge_margin,
            area_decline_ratio=request.review_policy.area_decline_ratio,
            confidences=confidences,
            low_confidence_threshold=request.review_policy.low_confidence_threshold,
            collapse_low_confidence_runs=request.review_policy.collapse_low_confidence_runs,
        )
        review_frame_indices = [int(item["frame_index"]) for item in review_queue]
        estimated_min_interactions = sum(int(item["estimated_interactions"]) for item in review_queue)
        video_duration_seconds = compute_video_duration_seconds(metadata)
        interactions_per_video_minute = compute_interactions_per_video_minute(
            estimated_min_interactions,
            metadata,
        )
        review_queue_path.write_text(json.dumps(review_queue, indent=2), encoding="utf-8")

        render_overlay_video(
            frame_dir=frame_dir,
            masks=masks,
            output_path=overlay_path,
            fps=metadata.fps,
            init_box_xyxy=request.init_box_xyxy,
        )

        metrics = {
            "run_id": run_id,
            "video_path": str(video_path),
            "frame_count": metadata.frame_count,
            "fps": metadata.fps,
            "width": metadata.width,
            "height": metadata.height,
            "box_xyxy": list(request.init_box_xyxy),
            "mask_areas": mask_areas,
            "mask_confidences": (
                [round(float(confidences.get(i, 0.0)), 4) for i in range(metadata.frame_count)]
                if confidences else []
            ),
            "review_frame_indices": review_frame_indices,
            "review_queue_path": str(review_queue_path),
            "review_queue": review_queue,
            "review_reason_counts": count_review_reasons(review_queue),
            "review_policy": "Queue low-confidence propagation frames (geometric drift signals + SAM2 mask confidence); ask for one point/box correction per queued frame.",
            "review_policy_config": request.review_policy.to_dict(),
            "estimated_min_interactions": estimated_min_interactions,
            "video_duration_seconds": round(video_duration_seconds, 3),
            "estimated_interactions_per_video_minute": interactions_per_video_minute,
            "overlay_video_path": str(overlay_path),
            "mask_dir": str(mask_dir),
            "sam2_model_cfg": request.sam2_model_cfg,
            "sam2_checkpoint": str(request.sam2_checkpoint),
            "device": request.device,
        }
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        return VideoSegmentationResult(
            overlay_video_path=overlay_path,
            mask_dir=mask_dir,
            review_frame_indices=review_frame_indices,
            review_queue_path=review_queue_path,
            review_queue=review_queue,
            run_dir=run_dir,
            metrics_path=metrics_path,
            frame_count=metadata.frame_count,
            fps=metadata.fps,
            estimated_min_interactions=estimated_min_interactions,
            interactions_per_video_minute=interactions_per_video_minute,
        )

    def run_with_corrections(
        self,
        request: CorrectionPropagationRequest,
    ) -> CorrectionPropagationResult:
        run_dir = Path(request.run_dir)
        frame_dir = run_dir / "frames"
        metrics_before_path = run_dir / "metrics.json"
        if not frame_dir.exists():
            raise FileNotFoundError(f"Run frames directory does not exist: {frame_dir}")
        if not metrics_before_path.exists():
            raise FileNotFoundError(f"Baseline metrics not found: {metrics_before_path}")

        metrics_before = json.loads(metrics_before_path.read_text(encoding="utf-8"))
        metadata = VideoMetadata(
            fps=float(metrics_before["fps"]),
            width=int(metrics_before["width"]),
            height=int(metrics_before["height"]),
            frame_count=int(metrics_before["frame_count"]),
        )
        validate_box_xyxy(request.init_box_xyxy, metadata)

        corrections = load_corrections(run_dir)
        if not corrections:
            raise ValueError(f"No corrections found in {corrections_path(run_dir)}.")
        validate_corrections(corrections, metadata)

        segmenter = self.segmenter or Sam2VideoSegmenter(
            model_cfg=request.sam2_model_cfg,
            checkpoint_path=request.sam2_checkpoint,
            device=request.device,
            vos_optimized=request.vos_optimized,
        )
        if not hasattr(segmenter, "segment_with_corrections"):
            raise RuntimeError("The configured segmenter does not support correction propagation.")

        mask_by_frame = segmenter.segment_with_corrections(
            frame_dir=frame_dir,
            init_box_xyxy=request.init_box_xyxy,
            frame_count=metadata.frame_count,
            corrections=corrections,
        )
        confidences = getattr(segmenter, "last_confidences", None) or None

        mask_dir = run_dir / "masks_after"
        overlay_path = run_dir / "overlay_after.mp4"
        metrics_after_path = run_dir / "metrics_after.json"
        review_queue_path = run_dir / "review_queue_after.json"
        comparison_path = run_dir / "comparison.json"
        mask_dir.mkdir(parents=True, exist_ok=True)

        masks: dict[int, np.ndarray] = {}
        mask_areas: list[int] = []
        for frame_idx in range(metadata.frame_count):
            mask = normalize_mask(mask_by_frame.get(frame_idx), metadata)
            masks[frame_idx] = mask
            mask_areas.append(int(mask.sum()))
            save_mask(mask, mask_dir / f"{frame_idx:06d}.png")

        review_queue = build_review_queue(
            mask_areas,
            frame_dir=frame_dir,
            mask_dir=mask_dir,
            masks=masks,
            jump_ratio=request.review_policy.area_jump_ratio,
            center_jump_ratio=request.review_policy.center_jump_ratio,
            iou_drop_threshold=request.review_policy.iou_drop_threshold,
            edge_margin=request.review_policy.edge_margin,
            area_decline_ratio=request.review_policy.area_decline_ratio,
            confidences=confidences,
            low_confidence_threshold=request.review_policy.low_confidence_threshold,
            collapse_low_confidence_runs=request.review_policy.collapse_low_confidence_runs,
        )
        review_frame_indices = [int(item["frame_index"]) for item in review_queue]
        actual_interactions = count_correction_interactions(corrections)
        interactions_per_video_minute = compute_interactions_per_video_minute(
            actual_interactions,
            metadata,
        )
        review_queue_path.write_text(json.dumps(review_queue, indent=2), encoding="utf-8")

        render_overlay_video(
            frame_dir=frame_dir,
            masks=masks,
            output_path=overlay_path,
            fps=metadata.fps,
            init_box_xyxy=request.init_box_xyxy,
        )

        metrics_after = {
            "run_id": metrics_before.get("run_id"),
            "phase": "after_corrections",
            "video_path": str(request.video_path),
            "frame_count": metadata.frame_count,
            "fps": metadata.fps,
            "width": metadata.width,
            "height": metadata.height,
            "box_xyxy": list(request.init_box_xyxy),
            "mask_areas": mask_areas,
            "mask_confidences": (
                [round(float(confidences.get(i, 0.0)), 4) for i in range(metadata.frame_count)]
                if confidences else []
            ),
            "review_frame_indices": review_frame_indices,
            "review_queue_path": str(review_queue_path),
            "review_queue": review_queue,
            "review_reason_counts": count_review_reasons(review_queue),
            "review_policy_config": request.review_policy.to_dict(),
            "corrections_path": str(corrections_path(run_dir)),
            "corrections_applied": corrections,
            "actual_interactions": actual_interactions,
            "video_duration_seconds": round(compute_video_duration_seconds(metadata), 3),
            "actual_interactions_per_video_minute": interactions_per_video_minute,
            "overlay_video_path": str(overlay_path),
            "mask_dir": str(mask_dir),
            "sam2_model_cfg": request.sam2_model_cfg,
            "sam2_checkpoint": str(request.sam2_checkpoint),
            "device": request.device,
        }
        metrics_after_path.write_text(json.dumps(metrics_after, indent=2), encoding="utf-8")

        comparison = build_correction_comparison(metrics_before, metrics_after)
        comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

        return CorrectionPropagationResult(
            overlay_video_path=overlay_path,
            mask_dir=mask_dir,
            review_queue_path=review_queue_path,
            review_queue=review_queue,
            metrics_path=metrics_after_path,
            comparison_path=comparison_path,
            corrections=corrections,
            frame_count=metadata.frame_count,
            fps=metadata.fps,
            actual_interactions=actual_interactions,
            interactions_per_video_minute=interactions_per_video_minute,
        )


def make_run_id(video_path: Path) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    safe_stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in video_path.stem)
    return f"{stamp}-{safe_stem}-{uuid.uuid4().hex[:8]}"


def parse_box_xyxy(raw_box: str) -> tuple[int, int, int, int]:
    values = [int(v.strip()) for v in raw_box.split(",") if v.strip()]
    if len(values) != 4:
        raise ValueError("Box must be formatted as x1,y1,x2,y2.")
    box = tuple(values)
    if box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError("Box must satisfy x1 < x2 and y1 < y2.")
    return box


def validate_box_xyxy(box: tuple[int, int, int, int], metadata: VideoMetadata) -> None:
    x1, y1, x2, y2 = box
    if x1 < 0 or y1 < 0 or x2 > metadata.width or y2 > metadata.height:
        raise ValueError(
            f"Box {box} is outside the first frame size {metadata.width}x{metadata.height}."
        )


def normalize_mask(mask: Optional[np.ndarray], metadata: VideoMetadata) -> np.ndarray:
    if mask is None:
        return np.zeros((metadata.height, metadata.width), dtype=np.uint8)

    normalized = np.asarray(mask)
    normalized = np.squeeze(normalized)
    if normalized.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {normalized.shape}.")
    if normalized.shape != (metadata.height, metadata.width):
        raise ValueError(
            "Mask shape does not match video frames: "
            f"{normalized.shape} vs {(metadata.height, metadata.width)}."
        )
    return (normalized > 0).astype(np.uint8)


def validate_corrections(corrections: list[dict[str, Any]], metadata: VideoMetadata) -> None:
    for correction in corrections:
        frame_index = int(correction["frame_index"])
        if frame_index < 0 or frame_index >= metadata.frame_count:
            raise ValueError(
                f"Correction frame_index {frame_index} is outside 0..{metadata.frame_count - 1}."
            )
        correction_type = str(correction["type"])
        if correction_type in {"positive_point", "negative_point"}:
            points = correction.get("points") or []
            if not points:
                raise ValueError(f"Point correction has no points: {correction}")
            for point in points:
                x = float(point["x"])
                y = float(point["y"])
                if x < 0 or y < 0 or x >= metadata.width or y >= metadata.height:
                    raise ValueError(
                        f"Point correction {(x, y)} is outside frame size "
                        f"{metadata.width}x{metadata.height}."
                    )
        elif correction_type == "tight_box":
            x1, y1, x2, y2 = [int(value) for value in correction["box_xyxy"]]
            validate_box_xyxy((x1, y1, x2, y2), metadata)
        else:
            raise ValueError(f"Unsupported correction type: {correction_type}")


def count_correction_interactions(corrections: list[dict[str, Any]]) -> int:
    return sum(int(correction.get("estimated_interactions", 1)) for correction in corrections)


def build_correction_comparison(
    metrics_before: dict[str, Any],
    metrics_after: dict[str, Any],
) -> dict[str, Any]:
    before_review_count = len(metrics_before.get("review_frame_indices", []))
    after_review_count = len(metrics_after.get("review_frame_indices", []))
    before_areas = [int(value) for value in metrics_before.get("mask_areas", [])]
    after_areas = [int(value) for value in metrics_after.get("mask_areas", [])]
    area_delta = [
        after - before
        for before, after in zip(before_areas, after_areas)
    ]
    return {
        "run_id": metrics_before.get("run_id"),
        "before_overlay_video_path": metrics_before.get("overlay_video_path"),
        "after_overlay_video_path": metrics_after.get("overlay_video_path"),
        "before_review_frame_count": before_review_count,
        "after_review_frame_count": after_review_count,
        "review_frame_count_delta": after_review_count - before_review_count,
        "corrections_applied_count": len(metrics_after.get("corrections_applied", [])),
        "actual_interactions": metrics_after.get("actual_interactions", 0),
        "actual_interactions_per_video_minute": metrics_after.get(
            "actual_interactions_per_video_minute",
            0.0,
        ),
        "mask_area_delta": area_delta,
    }


def count_review_reasons(review_queue: list[dict[str, object]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for item in review_queue:
        reasons = item.get("reasons") or [item.get("reason")]
        for reason in reasons:
            if reason:
                counter[str(reason)] += 1
    return dict(sorted(counter.items()))


def detect_review_frames(mask_areas: Iterable[float], jump_ratio: float = 0.6) -> list[int]:
    """Return frame indices that should enter the human review queue."""
    return [
        int(item["frame_index"])
        for item in build_review_queue(mask_areas, jump_ratio=jump_ratio)
    ]


def build_review_queue(
    mask_areas: Iterable[float],
    jump_ratio: float = 0.6,
    frame_dir: Optional[Path] = None,
    mask_dir: Optional[Path] = None,
    masks: Optional[dict[int, np.ndarray]] = None,
    center_jump_ratio: float = 0.25,
    iou_drop_threshold: float = 0.2,
    edge_margin: int = 0,
    area_decline_ratio: float = 0.35,
    confidences: Optional[dict[int, float]] = None,
    low_confidence_threshold: float = 0.5,
    collapse_low_confidence_runs: bool = True,
) -> list[dict[str, object]]:
    """Create the smallest set of frames that need human correction.

    Geometric signals (area/center/IoU/edge) act as drift proxies. When the segmenter
    exposes per-frame SAM2 confidence (``confidences``), frames below
    ``low_confidence_threshold`` are additionally flagged, and every queued frame is
    annotated with its ``confidence`` for triage (a flagged frame with high confidence
    is likely a geometric false positive; low confidence suggests real loss/occlusion).
    """
    areas = [float(area) for area in mask_areas]
    review_by_frame: dict[int, dict[str, object]] = {}
    mask_stats = build_mask_stats(masks) if masks else {}

    if not areas:
        return []

    if areas[0] <= 0:
        add_review_queue_signal(
            review_by_frame,
            frame_index=0,
            reason="empty_initial_mask",
            current_area=areas[0],
            previous_area=None,
            area_change_ratio=1.0,
            frame_dir=frame_dir,
            mask_dir=mask_dir,
        )

    for idx in range(1, len(areas)):
        prev = max(areas[idx - 1], 1.0)
        change = abs(areas[idx] - areas[idx - 1]) / prev
        if change >= jump_ratio:
            reason = "mask_area_dropped" if areas[idx] < areas[idx - 1] else "mask_area_spiked"
            add_review_queue_signal(
                review_by_frame,
                frame_index=idx,
                reason=reason,
                current_area=areas[idx],
                previous_area=areas[idx - 1],
                area_change_ratio=change,
                frame_dir=frame_dir,
                mask_dir=mask_dir,
            )

        if idx >= 2 and areas[idx] < areas[idx - 1] < areas[idx - 2]:
            decline = (areas[idx - 2] - areas[idx]) / max(areas[idx - 2], 1.0)
            if decline >= area_decline_ratio:
                add_review_queue_signal(
                    review_by_frame,
                    frame_index=idx,
                    reason="mask_area_trending_down",
                    current_area=areas[idx],
                    previous_area=areas[idx - 1],
                    area_change_ratio=decline,
                    frame_dir=frame_dir,
                    mask_dir=mask_dir,
                    diagnostics={"area_decline_ratio": round(float(decline), 4)},
                )

        if mask_stats and idx in mask_stats and idx - 1 in mask_stats:
            current_stats = mask_stats[idx]
            previous_stats = mask_stats[idx - 1]
            if current_stats["area"] <= 0:
                add_review_queue_signal(
                    review_by_frame,
                    frame_index=idx,
                    reason="empty_mask",
                    current_area=areas[idx],
                    previous_area=areas[idx - 1],
                    area_change_ratio=1.0,
                    frame_dir=frame_dir,
                    mask_dir=mask_dir,
                )
            if current_stats["center"] is not None and previous_stats["center"] is not None:
                center_shift = compute_center_shift_ratio(previous_stats, current_stats)
                if center_shift >= center_jump_ratio:
                    add_review_queue_signal(
                        review_by_frame,
                        frame_index=idx,
                        reason="mask_center_jumped",
                        current_area=areas[idx],
                        previous_area=areas[idx - 1],
                        area_change_ratio=change,
                        frame_dir=frame_dir,
                        mask_dir=mask_dir,
                        diagnostics={"center_shift_ratio": round(float(center_shift), 4)},
                    )

            if idx in masks and idx - 1 in masks:
                iou = compute_mask_iou(masks[idx - 1], masks[idx])
                if current_stats["area"] > 0 and previous_stats["area"] > 0 and iou <= iou_drop_threshold:
                    add_review_queue_signal(
                        review_by_frame,
                        frame_index=idx,
                        reason="mask_iou_dropped",
                        current_area=areas[idx],
                        previous_area=areas[idx - 1],
                        area_change_ratio=change,
                        frame_dir=frame_dir,
                        mask_dir=mask_dir,
                        diagnostics={"previous_mask_iou": round(float(iou), 4)},
                    )

            current_touches_edge = mask_touches_edge(current_stats, edge_margin=edge_margin)
            previous_touches_edge = mask_touches_edge(previous_stats, edge_margin=edge_margin)
            if current_touches_edge and not previous_touches_edge:
                add_review_queue_signal(
                    review_by_frame,
                    frame_index=idx,
                    reason="mask_touches_frame_edge",
                    current_area=areas[idx],
                    previous_area=areas[idx - 1],
                    area_change_ratio=change,
                    frame_dir=frame_dir,
                    mask_dir=mask_dir,
                    diagnostics={"edge_margin": edge_margin},
                )

    if confidences is not None:
        for idx in range(len(areas)):
            confidence = float(confidences.get(idx, 0.0))
            if confidence < low_confidence_threshold:
                add_review_queue_signal(
                    review_by_frame,
                    frame_index=idx,
                    reason="low_mask_confidence",
                    current_area=areas[idx],
                    previous_area=areas[idx - 1] if idx > 0 else None,
                    area_change_ratio=0.0,
                    frame_dir=frame_dir,
                    mask_dir=mask_dir,
                    diagnostics={"confidence": round(confidence, 4)},
                )
        # annotate every queued frame with SAM2 confidence so callers can triage
        for idx, item in review_by_frame.items():
            diagnostics = dict(item.get("diagnostics", {}))
            diagnostics.setdefault("confidence", round(float(confidences.get(idx, 0.0)), 4))
            item["diagnostics"] = diagnostics

        # Collapse a run of consecutive empty + low-confidence frames (a likely object
        # absence/loss SAM2 is itself unsure about) into a single onset interaction:
        # the human makes one decision ("object left / re-point here"), not one per frame.
        if collapse_low_confidence_runs:
            absent = [
                idx
                for idx in sorted(review_by_frame)
                if areas[idx] == 0 and float(confidences.get(idx, 1.0)) < low_confidence_threshold
            ]
            for run in _consecutive_runs(absent):
                for position, idx in enumerate(run):
                    item = review_by_frame[idx]
                    item["status"] = "low_confidence_empty"
                    item["estimated_interactions"] = 1 if position == 0 else 0

    # Unified, rankable review priority per frame (geometric reasons + SAM2 confidence,
    # with collapsed/suppressed frames ranked low). Callers can sort the queue by this.
    for item in review_by_frame.values():
        item["review_score"] = compute_review_score(item)

    return [review_by_frame[idx] for idx in sorted(review_by_frame)]


def _consecutive_runs(indices: list[int]) -> list[list[int]]:
    """Group sorted frame indices into runs of consecutive integers."""
    runs: list[list[int]] = []
    for idx in indices:
        if runs and idx == runs[-1][-1] + 1:
            runs[-1].append(idx)
        else:
            runs.append([idx])
    return runs


# Per-reason base risk weights — the single source of truth for review priority.
# (The Gradio UI reads each item's `review_score` instead of re-deriving its own.)
REVIEW_REASON_WEIGHTS: dict[str, float] = {
    "empty_initial_mask": 0.95,
    "empty_mask": 0.92,
    "low_mask_confidence": 0.88,
    "mask_center_jumped": 0.84,
    "mask_iou_dropped": 0.78,
    "mask_area_spiked": 0.72,
    "mask_touches_frame_edge": 0.68,
    "mask_area_dropped": 0.66,
    "mask_area_trending_down": 0.62,
}


def compute_review_score(item: dict[str, object]) -> float:
    """Combine an item's reasons (and collapse state) into a 0–1 review priority.

    Frames suppressed by absence-run collapse (estimated_interactions == 0) rank low;
    otherwise the score is the strongest reason weight, nudged up when several signals
    corroborate. SAM2 confidence enters via the `low_mask_confidence` reason weight and
    the collapse, so a high-confidence geometric false positive ranks below a real loss.
    """
    if int(item.get("estimated_interactions", 1)) == 0:
        return 0.15
    reasons = list(item.get("reasons", []) or [item.get("reason")])
    base = max((REVIEW_REASON_WEIGHTS.get(reason, 0.55) for reason in reasons), default=0.55)
    return round(min(0.99, base + 0.04 * (len(reasons) - 1)), 4)


def add_review_queue_signal(
    review_by_frame: dict[int, dict[str, object]],
    frame_index: int,
    reason: str,
    current_area: float,
    previous_area: Optional[float],
    area_change_ratio: float,
    frame_dir: Optional[Path],
    mask_dir: Optional[Path],
    diagnostics: Optional[dict[str, object]] = None,
) -> None:
    if frame_index not in review_by_frame:
        review_by_frame[frame_index] = make_review_queue_item(
            frame_index=frame_index,
            reason=reason,
            current_area=current_area,
            previous_area=previous_area,
            area_change_ratio=area_change_ratio,
            frame_dir=frame_dir,
            mask_dir=mask_dir,
            diagnostics=diagnostics,
        )
        return

    item = review_by_frame[frame_index]
    reasons = list(item.get("reasons", [item["reason"]]))
    if reason not in reasons:
        reasons.append(reason)
    item["reasons"] = reasons
    item["reason"] = reasons[0]
    item["recommended_correction"] = (
        "Add one positive/negative point or adjust one tight box on this frame, "
        "then re-propagate from the correction frame."
    )
    if diagnostics:
        item_diagnostics = dict(item.get("diagnostics", {}))
        item_diagnostics.update(diagnostics)
        item["diagnostics"] = item_diagnostics


def build_mask_stats(masks: dict[int, np.ndarray]) -> dict[int, dict[str, object]]:
    return {
        int(frame_idx): compute_mask_stats(mask)
        for frame_idx, mask in masks.items()
    }


def compute_mask_stats(mask: np.ndarray) -> dict[str, object]:
    normalized = np.squeeze(np.asarray(mask)) > 0
    if normalized.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {normalized.shape}.")
    height, width = normalized.shape
    ys, xs = np.nonzero(normalized)
    area = int(len(xs))
    if area == 0:
        return {
            "area": 0,
            "center": None,
            "bbox_xyxy": None,
            "width": width,
            "height": height,
        }

    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max())
    y2 = int(ys.max())
    return {
        "area": area,
        "center": (float(xs.mean()), float(ys.mean())),
        "bbox_xyxy": (x1, y1, x2, y2),
        "width": width,
        "height": height,
    }


def compute_center_shift_ratio(
    previous_stats: dict[str, object],
    current_stats: dict[str, object],
) -> float:
    previous_center = previous_stats["center"]
    current_center = current_stats["center"]
    if previous_center is None or current_center is None:
        return 0.0

    px, py = previous_center
    cx, cy = current_center
    width = float(current_stats["width"])
    height = float(current_stats["height"])
    diagonal = max((width**2 + height**2) ** 0.5, 1.0)
    return (((float(cx) - float(px)) ** 2 + (float(cy) - float(py)) ** 2) ** 0.5) / diagonal


def compute_mask_iou(previous_mask: np.ndarray, current_mask: np.ndarray) -> float:
    previous = np.squeeze(np.asarray(previous_mask)) > 0
    current = np.squeeze(np.asarray(current_mask)) > 0
    if previous.shape != current.shape:
        raise ValueError(f"Mask shapes differ: {previous.shape} vs {current.shape}.")

    union = np.logical_or(previous, current).sum()
    if union == 0:
        return 1.0
    intersection = np.logical_and(previous, current).sum()
    return float(intersection) / float(union)


def mask_touches_edge(mask_stats: dict[str, object], edge_margin: int = 0) -> bool:
    bbox = mask_stats["bbox_xyxy"]
    if bbox is None:
        return False

    x1, y1, x2, y2 = [int(value) for value in bbox]
    width = int(mask_stats["width"])
    height = int(mask_stats["height"])
    return (
        x1 <= edge_margin
        or y1 <= edge_margin
        or x2 >= width - 1 - edge_margin
        or y2 >= height - 1 - edge_margin
    )


def make_review_queue_item(
    frame_index: int,
    reason: str,
    current_area: float,
    previous_area: Optional[float],
    area_change_ratio: float,
    frame_dir: Optional[Path],
    mask_dir: Optional[Path],
    diagnostics: Optional[dict[str, object]] = None,
) -> dict[str, object]:
    item: dict[str, object] = {
        "frame_index": frame_index,
        "reason": reason,
        "reasons": [reason],
        "status": "needs_review",
        "current_mask_area": int(current_area),
        "previous_mask_area": int(previous_area) if previous_area is not None else None,
        "area_change_ratio": round(float(area_change_ratio), 4),
        "recommended_correction": (
            "Add one positive/negative point or adjust one tight box on this frame, "
            "then re-propagate from the correction frame."
        ),
        "estimated_interactions": 1,
    }
    if frame_dir is not None:
        item["frame_path"] = str(frame_dir / f"{frame_index:06d}.jpg")
    if mask_dir is not None:
        item["mask_path"] = str(mask_dir / f"{frame_index:06d}.png")
    if diagnostics:
        item["diagnostics"] = diagnostics
    return item


def compute_video_duration_seconds(metadata: VideoMetadata) -> float:
    if metadata.fps <= 0:
        return 0.0
    return float(metadata.frame_count) / float(metadata.fps)


def compute_interactions_per_video_minute(interactions: int, metadata: VideoMetadata) -> float:
    duration_seconds = compute_video_duration_seconds(metadata)
    if duration_seconds <= 0:
        return 0.0
    return round(float(interactions) / (duration_seconds / 60.0), 3)
