"""Offline export layer for the CVAT plugin (P8).

Pure functions, no cvat-sdk / network dependency, so the whole thing is unit-testable
without a running CVAT:

- masks -> COCO instance-segmentation JSON (RLE)         [push path A: dataset import]
- mask  -> CVAT native mask-shape RLE points             [push path B: editable shape]
- review_queue -> CVAT issue payloads                    [the "review queue" in CVAT]
- interaction report (estimated vs resolved)

RLE conventions (see docs/cvat_plugin.md §6.3/§6.4; confirm against your CVAT version):
- COCO uncompressed RLE: counts are column-major (Fortran) run lengths, starting with a
  background (0) run.
- CVAT mask shape: row-major run lengths over the cropped bbox, starting with a 0 run,
  followed by [left, top, right, bottom].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np


# --- run-length helpers -------------------------------------------------------

def _rle_counts(flat: np.ndarray) -> list[int]:
    """Run lengths of a 1-D 0/1 array, starting with a 0 (background) run."""
    flat = (np.asarray(flat) > 0).astype(np.int8)
    n = int(flat.size)
    if n == 0:
        return []
    change = np.nonzero(np.diff(flat))[0] + 1
    bounds = np.concatenate(([0], change, [n]))
    runs = [int(r) for r in np.diff(bounds)]
    # runs alternate starting with flat[0]; COCO/CVAT count zeros first
    return ([0] + runs) if flat[0] == 1 else runs


def _rle_decode(counts: list[int], n: int) -> np.ndarray:
    """Inverse of :func:`_rle_counts` — rebuild the flat 0/1 array (for tests/round-trip)."""
    flat = np.zeros(int(n), dtype=np.uint8)
    pos, val = 0, 0
    for count in counts:
        if val:
            flat[pos:pos + count] = 1
        pos += int(count)
        val ^= 1
    return flat


def _binary(mask: np.ndarray) -> np.ndarray:
    arr = np.squeeze(np.asarray(mask)) > 0
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {arr.shape}.")
    return arr


def _bbox_xyxy(mask: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    ys, xs = np.nonzero(_binary(mask))
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


# --- path A: COCO instance segmentation --------------------------------------

def coco_rle(mask: np.ndarray) -> dict[str, Any]:
    """Encode a binary mask as COCO uncompressed RLE: {"counts": [...], "size": [h, w]}."""
    arr = _binary(mask)
    h, w = arr.shape
    return {"counts": _rle_counts(arr.flatten(order="F")), "size": [int(h), int(w)]}


def decode_coco_rle(rle: dict[str, Any]) -> np.ndarray:
    """Inverse of :func:`coco_rle` (used in tests / verification)."""
    h, w = int(rle["size"][0]), int(rle["size"][1])
    return _rle_decode(rle["counts"], h * w).reshape((h, w), order="F")


def masks_to_coco(
    masks: dict[int, np.ndarray],
    label: str = "object",
    *,
    category_id: int = 1,
    frame_name: Callable[[int], str] = lambda i: f"frame_{i:06d}",
) -> dict[str, Any]:
    """Convert per-frame masks to a COCO instance-segmentation document.

    One image per frame (even when the mask is empty), one RLE annotation per non-empty
    mask. ``frame_name`` maps a frame index to the COCO ``file_name`` — the live client
    must align this with how CVAT names the task's frames (docs/cvat_plugin.md §6.1).
    """
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    next_ann_id = 1
    for frame_index in sorted(masks):
        arr = _binary(masks[frame_index])
        h, w = arr.shape
        image_id = int(frame_index) + 1
        images.append(
            {
                "id": image_id,
                "file_name": frame_name(int(frame_index)),
                "width": int(w),
                "height": int(h),
                "frame": int(frame_index),
            }
        )
        bbox = _bbox_xyxy(arr)
        if bbox is None:
            continue  # empty mask -> image with no annotation
        x1, y1, x2, y2 = bbox
        annotations.append(
            {
                "id": next_ann_id,
                "image_id": image_id,
                "category_id": int(category_id),
                "segmentation": coco_rle(arr),
                "area": int(arr.sum()),
                "bbox": [x1, y1, x2 - x1 + 1, y2 - y1 + 1],  # COCO xywh
                "iscrowd": 0,
            }
        )
        next_ann_id += 1
    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": int(category_id), "name": label}],
    }


# --- path B: CVAT native mask shape ------------------------------------------

def mask_to_cvat_points(mask: np.ndarray) -> list[int]:
    """Encode a binary mask as a CVAT mask-shape ``points`` list.

    Row-major RLE over the cropped bbox (starting with a 0 run) followed by
    [left, top, right, bottom]. Empty mask -> empty list.
    """
    arr = _binary(mask)
    bbox = _bbox_xyxy(arr)
    if bbox is None:
        return []
    left, top, right, bottom = bbox
    crop = arr[top:bottom + 1, left:right + 1]
    counts = _rle_counts(crop.flatten(order="C"))
    return [*counts, int(left), int(top), int(right), int(bottom)]


def decode_cvat_points(points: list[int], height: int, width: int) -> np.ndarray:
    """Inverse of :func:`mask_to_cvat_points` into a full ``height`` x ``width`` mask."""
    full = np.zeros((int(height), int(width)), dtype=np.uint8)
    if not points:
        return full
    counts = [int(v) for v in points[:-4]]
    left, top, right, bottom = (int(v) for v in points[-4:])
    crop_h, crop_w = bottom - top + 1, right - left + 1
    crop = _rle_decode(counts, crop_h * crop_w).reshape((crop_h, crop_w), order="C")
    full[top:bottom + 1, left:right + 1] = crop
    return full


# --- review queue -> CVAT issues ---------------------------------------------

@dataclass
class IssuePayload:
    """A CVAT issue to raise on one flagged frame (the live client attaches it to a job)."""

    frame: int
    message: str
    review_score: float
    reasons: list[str] = field(default_factory=list)
    status: Optional[str] = None
    # [x1, y1, x2, y2] rectangle for the issue marker; None -> client fills from the mask bbox
    position: Optional[list[float]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "message": self.message,
            "review_score": self.review_score,
            "reasons": self.reasons,
            "status": self.status,
            "position": self.position,
        }


def _issue_message(item: dict[str, Any]) -> str:
    reasons = item.get("reasons") or [item.get("reason", "needs_review")]
    reasons = [str(r) for r in reasons if r]
    score = item.get("review_score")
    prefix = f"[{float(score):.2f}] " if isinstance(score, (int, float)) else ""
    body = ", ".join(reasons) if reasons else "needs_review"
    fix = item.get("recommended_correction")
    suffix = f" | fix: {fix}" if fix else ""  # ASCII separator: issue text crosses systems
    return f"{prefix}{body}{suffix}"


def queue_to_issues(
    review_queue: list[dict[str, Any]],
    *,
    min_score: float = 0.0,
    bboxes: Optional[dict[int, tuple[int, int, int, int]]] = None,
) -> list[IssuePayload]:
    """Map the review queue to CVAT issue payloads, highest review_score first.

    ``min_score`` drops low-priority frames (e.g. confidence-damped ``likely_occlusion``
    shrinks at 0.3 and collapsed absence frames at 0.15), so the human sees real problems
    as issues. ``bboxes`` (frame -> xyxy, e.g. from the masks) fills the issue marker
    position when available.
    """
    issues: list[IssuePayload] = []
    for item in review_queue:
        score = item.get("review_score")
        score_val = float(score) if isinstance(score, (int, float)) else 1.0
        if score_val < min_score:
            continue
        frame = int(item["frame_index"])
        position = None
        if bboxes and frame in bboxes:
            x1, y1, x2, y2 = bboxes[frame]
            position = [float(x1), float(y1), float(x2), float(y2)]
        reasons = [str(r) for r in (item.get("reasons") or [item.get("reason")]) if r]
        issues.append(
            IssuePayload(
                frame=frame,
                message=_issue_message(item),
                review_score=score_val,
                reasons=reasons,
                status=item.get("status"),
                position=position,
            )
        )
    issues.sort(key=lambda issue: issue.review_score, reverse=True)
    return issues


def interaction_report(
    review_queue: list[dict[str, Any]],
    resolved_issue_count: int,
    issues_created: Optional[int] = None,
) -> dict[str, Any]:
    """Compare MaskReview's estimate against what the annotator actually resolved in CVAT.

    When ``issues_created`` is given (issues currently on the task), also reports review
    coverage = resolved / created — how much of the flagged work the human has addressed.
    """
    estimated = sum(int(item.get("estimated_interactions", 1)) for item in review_queue)
    report = {
        "queued_frames": len(review_queue),
        "estimated_interactions": estimated,
        "resolved_issues": int(resolved_issue_count),
        "delta_vs_estimate": int(resolved_issue_count) - estimated,
    }
    if issues_created is not None:
        report["issues_created"] = int(issues_created)
        report["review_coverage"] = (
            round(int(resolved_issue_count) / issues_created, 3) if issues_created else None
        )
    return report
