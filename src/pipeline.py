from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Protocol

import numpy as np

from sam2_runner import Sam2VideoSegmenter
from video_io import (
    VideoMetadata,
    extract_video_frames,
    render_overlay_video,
    save_mask,
)


DEFAULT_SAM2_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"
DEFAULT_SAM2_CHECKPOINT = Path("checkpoints/sam2.1_hiera_tiny.pt")


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


class VideoSegmenter(Protocol):
    def segment(
        self,
        frame_dir: Path,
        init_box_xyxy: tuple[int, int, int, int],
        frame_count: int,
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

        masks: dict[int, np.ndarray] = {}
        mask_areas: list[int] = []
        for frame_idx in range(metadata.frame_count):
            mask = normalize_mask(mask_by_frame.get(frame_idx), metadata)
            masks[frame_idx] = mask
            mask_areas.append(int(mask.sum()))
            save_mask(mask, mask_dir / f"{frame_idx:06d}.png")

        review_queue = build_review_queue(mask_areas, frame_dir=frame_dir, mask_dir=mask_dir)
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
            "review_frame_indices": review_frame_indices,
            "review_queue_path": str(review_queue_path),
            "review_queue": review_queue,
            "review_policy": "Queue only low-confidence propagation frames; ask for one point/box correction per queued frame.",
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
) -> list[dict[str, object]]:
    """Create the smallest set of frames that need human correction.

    Until SAM2 confidence logits are exposed by the runner, abrupt mask-area
    changes act as the low-confidence proxy for propagation drift.
    """
    areas = [float(area) for area in mask_areas]
    review: list[dict[str, object]] = []

    if not areas:
        return review

    if areas[0] <= 0:
        review.append(
            make_review_queue_item(
                frame_index=0,
                reason="empty_initial_mask",
                current_area=areas[0],
                previous_area=None,
                area_change_ratio=1.0,
                frame_dir=frame_dir,
                mask_dir=mask_dir,
            )
        )

    for idx in range(1, len(areas)):
        prev = max(areas[idx - 1], 1.0)
        change = abs(areas[idx] - areas[idx - 1]) / prev
        if change >= jump_ratio:
            reason = "mask_area_dropped" if areas[idx] < areas[idx - 1] else "mask_area_spiked"
            review.append(
                make_review_queue_item(
                    frame_index=idx,
                    reason=reason,
                    current_area=areas[idx],
                    previous_area=areas[idx - 1],
                    area_change_ratio=change,
                    frame_dir=frame_dir,
                    mask_dir=mask_dir,
                )
            )

    return review


def make_review_queue_item(
    frame_index: int,
    reason: str,
    current_area: float,
    previous_area: Optional[float],
    area_change_ratio: float,
    frame_dir: Optional[Path],
    mask_dir: Optional[Path],
) -> dict[str, object]:
    item: dict[str, object] = {
        "frame_index": frame_index,
        "reason": reason,
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
