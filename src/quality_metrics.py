"""Ground-truth mask quality metrics: region IoU (J), boundary F, and DAVIS-style J&F.

These let the evaluation harness score *mask quality* against labeled ground-truth
masks, instead of only counting review-queue frames and interactions. Region IoU is
the DAVIS "J" measure; boundary F is an approximation of the DAVIS "F" contour
measure; J&F is their mean. OpenCV is imported lazily so that ``region_iou`` and the
pure-Python aggregation stay usable even where cv2 is unavailable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np


def _cv2():
    try:
        import cv2  # noqa: WPS433 (lazy import by design)
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("OpenCV (opencv-python) is required for boundary/mask IO metrics.") from exc
    return cv2


def region_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Intersection-over-union of two binary masks (the DAVIS J measure).

    Two empty masks count as perfect agreement (1.0); one empty and one non-empty
    is total disagreement (0.0).
    """
    pred_bool = np.asarray(pred).astype(bool)
    gt_bool = np.asarray(gt).astype(bool)
    union = int(np.logical_or(pred_bool, gt_bool).sum())
    if union == 0:
        return 1.0
    intersection = int(np.logical_and(pred_bool, gt_bool).sum())
    return intersection / union


def boundary_f_measure(pred: np.ndarray, gt: np.ndarray, bound_th: float = 0.008) -> float:
    """Boundary F-measure between two binary masks (approx. the DAVIS F measure).

    Boundaries are matched within a tolerance of ``bound_th`` of the image diagonal
    (at least 1 pixel). Two empty masks score 1.0; exactly one empty scores 0.0.
    """
    pred_b = np.asarray(pred).astype(np.uint8)
    gt_b = np.asarray(gt).astype(np.uint8)
    pred_area = int(pred_b.sum())
    gt_area = int(gt_b.sum())
    if pred_area == 0 and gt_area == 0:
        return 1.0
    if pred_area == 0 or gt_area == 0:
        return 0.0

    cv2 = _cv2()
    height, width = pred_b.shape[:2]
    diagonal = (height * height + width * width) ** 0.5
    radius = max(1, int(round(bound_th * diagonal)))

    # borderValue=0 so that a mask touching the image edge still yields a boundary
    # ring there (otherwise a full-frame mask would erode to itself and have no
    # boundary, making J&F of two identical full-frame masks collapse to ~0.5).
    cross = np.ones((3, 3), dtype=np.uint8)
    pred_boundary = pred_b - cv2.erode(pred_b, cross, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    gt_boundary = gt_b - cv2.erode(gt_b, cross, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    n_pred = int(pred_boundary.sum())
    n_gt = int(gt_boundary.sum())
    if n_pred == 0 or n_gt == 0:
        return 0.0

    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    pred_dil = cv2.dilate(pred_boundary, disk)
    gt_dil = cv2.dilate(gt_boundary, disk)

    precision = int((pred_boundary & gt_dil).sum()) / n_pred
    recall = int((gt_boundary & pred_dil).sum()) / n_gt
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def sequence_quality(
    pred_masks: dict[int, np.ndarray],
    gt_masks: dict[int, np.ndarray],
) -> dict[str, Any]:
    """Aggregate J, F and J&F over the frames present in both mask dicts."""
    frames = sorted(set(pred_masks) & set(gt_masks))
    if not frames:
        return {
            "frames_scored": 0,
            "mean_iou": None,
            "mean_boundary_f": None,
            "jf": None,
            "per_frame_iou": {},
        }

    ious: list[float] = []
    boundary_fs: list[float] = []
    per_frame_iou: dict[int, float] = {}
    for idx in frames:
        iou = region_iou(pred_masks[idx], gt_masks[idx])
        ious.append(iou)
        per_frame_iou[idx] = round(iou, 4)
        boundary_fs.append(boundary_f_measure(pred_masks[idx], gt_masks[idx]))

    mean_iou = sum(ious) / len(ious)
    mean_boundary_f = sum(boundary_fs) / len(boundary_fs)
    return {
        "frames_scored": len(frames),
        "mean_iou": round(mean_iou, 4),
        "mean_boundary_f": round(mean_boundary_f, 4),
        "jf": round((mean_iou + mean_boundary_f) / 2, 4),
        "per_frame_iou": per_frame_iou,
    }


def load_indexed_masks(
    mask_dir: Path,
    frame_count: int,
    height: int,
    width: int,
) -> dict[int, np.ndarray]:
    """Load ``000000.png`` .. masks from a directory as binary uint8 arrays.

    Frames whose file is missing or unreadable are skipped. Masks not matching the
    target size are nearest-neighbour resized so predictions and ground truth align.
    """
    cv2 = _cv2()
    mask_dir = Path(mask_dir)
    masks: dict[int, np.ndarray] = {}
    for idx in range(frame_count):
        path = mask_dir / f"{idx:06d}.png"
        if not path.exists():
            continue
        raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            continue
        if raw.shape[:2] != (height, width):
            raw = cv2.resize(raw, (width, height), interpolation=cv2.INTER_NEAREST)
        masks[idx] = (raw > 0).astype(np.uint8)
    return masks


def score_mask_dirs(
    pred_mask_dir: Path,
    gt_mask_dir: Path,
    frame_count: int,
    height: int,
    width: int,
) -> dict[str, Any]:
    """Convenience: load predicted + ground-truth mask dirs and score the sequence."""
    pred_masks = load_indexed_masks(pred_mask_dir, frame_count, height, width)
    gt_masks = load_indexed_masks(gt_mask_dir, frame_count, height, width)
    return sequence_quality(pred_masks, gt_masks)


def resolve_ground_truth_dir(raw_value: Optional[Any], base_dir: Path) -> Optional[Path]:
    """Resolve a manifest ground-truth dir (relative to the manifest) to a Path."""
    if raw_value in (None, ""):
        return None
    path = Path(str(raw_value))
    if not path.is_absolute():
        path = Path(base_dir) / path
    return path
