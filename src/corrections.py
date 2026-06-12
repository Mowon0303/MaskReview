from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


CORRECTIONS_FILENAME = "corrections.json"
POINT_CORRECTION_TYPES = {"positive_point", "negative_point"}
VALID_CORRECTION_TYPES = POINT_CORRECTION_TYPES | {"tight_box"}


def corrections_path(run_dir: Path) -> Path:
    return Path(run_dir) / CORRECTIONS_FILENAME


def load_corrections(run_dir: Path) -> list[dict[str, Any]]:
    path = corrections_path(run_dir)
    if not path.exists():
        return []

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected {path} to contain a JSON list.")
    return raw


def save_correction(run_dir: Path, correction: dict[str, Any]) -> Path:
    path = corrections_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    corrections = load_corrections(run_dir)

    frame_index = int(correction["frame_index"])
    corrections = [item for item in corrections if int(item["frame_index"]) != frame_index]
    corrections.append(correction)
    corrections.sort(key=lambda item: int(item["frame_index"]))

    path.write_text(json.dumps(corrections, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def build_correction(
    frame_index: int,
    correction_type: str,
    point_xy: str = "",
    box_xyxy: str = "",
    note: str = "",
    review_item: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    if frame_index < 0:
        raise ValueError("frame_index must be non-negative.")
    if correction_type not in VALID_CORRECTION_TYPES:
        allowed = ", ".join(sorted(VALID_CORRECTION_TYPES))
        raise ValueError(f"correction_type must be one of: {allowed}.")

    correction: dict[str, Any] = {
        "frame_index": int(frame_index),
        "type": correction_type,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "estimated_interactions": 1,
    }
    if review_item:
        correction["review_reason"] = review_item.get("reason")
        if "frame_path" in review_item:
            correction["frame_path"] = review_item["frame_path"]
        if "mask_path" in review_item:
            correction["mask_path"] = review_item["mask_path"]

    if correction_type in POINT_CORRECTION_TYPES:
        x, y = parse_point_xy(point_xy)
        label = "positive" if correction_type == "positive_point" else "negative"
        correction["points"] = [{"x": x, "y": y, "label": label}]
    else:
        correction["box_xyxy"] = list(parse_box_xyxy(box_xyxy))

    note = note.strip()
    if note:
        correction["note"] = note

    return correction


def parse_point_xy(raw_point: str) -> tuple[int, int]:
    values = _parse_int_csv(raw_point)
    if len(values) != 2:
        raise ValueError("Point must be formatted as x,y.")
    x, y = values
    if x < 0 or y < 0:
        raise ValueError("Point coordinates must be non-negative.")
    return x, y


def parse_box_xyxy(raw_box: str) -> tuple[int, int, int, int]:
    values = _parse_int_csv(raw_box)
    if len(values) != 4:
        raise ValueError("Box must be formatted as x1,y1,x2,y2.")
    x1, y1, x2, y2 = values
    if min(values) < 0:
        raise ValueError("Box coordinates must be non-negative.")
    if x1 >= x2 or y1 >= y2:
        raise ValueError("Box must satisfy x1 < x2 and y1 < y2.")
    return x1, y1, x2, y2


def _parse_int_csv(raw_value: str) -> list[int]:
    try:
        return [int(value.strip()) for value in raw_value.split(",") if value.strip()]
    except ValueError as exc:
        raise ValueError("Coordinates must be integers.") from exc
