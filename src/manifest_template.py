from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


@dataclass(frozen=True)
class VideoTemplateMetadata:
    width: int
    height: int
    frame_count: int
    fps: float
    duration_seconds: float

    def to_dict(self) -> dict[str, int | float]:
        return {
            "width": self.width,
            "height": self.height,
            "frame_count": self.frame_count,
            "fps": self.fps,
            "duration_seconds": self.duration_seconds,
        }


def generate_eval_manifest_template(
    video_dir: Path,
    output_path: Path,
    first_frame_dir: Optional[Path] = None,
    overwrite: bool = False,
) -> Path:
    video_dir = Path(video_dir)
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Manifest template already exists: {output_path}")

    first_frame_dir = Path(first_frame_dir) if first_frame_dir else output_path.parent / "eval_first_frames"
    first_frame_dir.mkdir(parents=True, exist_ok=True)

    video_paths = find_video_files(video_dir)
    cases = [
        build_manifest_case(
            video_path=video_path,
            manifest_dir=output_path.parent,
            first_frame_dir=first_frame_dir,
        )
        for video_path in video_paths
    ]

    manifest = {
        "instructions": [
            "Fill init_box_xyxy before running evaluation.",
            "Use first_frame_path to inspect the first frame for each video.",
            "Set expected_challenge to clean_tracking, occlusion, target_loss, similar_object, scale_change, or camera_motion.",
            "Optionally fill expected_review_frames after manual inspection to score queue precision/recall.",
            "Optional corrections can be added after reviewing baseline output.",
        ],
        "cases": cases,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def find_video_files(video_dir: Path) -> list[Path]:
    video_dir = Path(video_dir)
    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory does not exist: {video_dir}")
    return sorted(
        path
        for path in video_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def build_manifest_case(
    video_path: Path,
    manifest_dir: Path,
    first_frame_dir: Path,
) -> dict[str, Any]:
    case_id = sanitize_case_id(video_path.stem)
    first_frame_path = first_frame_dir / f"{case_id}.jpg"
    metadata = inspect_video(video_path, first_frame_path=first_frame_path)
    return {
        "id": case_id,
        "video_path": make_relative_path(video_path, manifest_dir),
        "object_name": "",
        "init_box_xyxy": None,
        "expected_challenge": "unspecified",
        "expected_review_frames": None,
        "notes": "",
        "first_frame_path": make_relative_path(first_frame_path, manifest_dir),
        "video_metadata": metadata.to_dict(),
        "corrections": [],
    }


def inspect_video(video_path: Path, first_frame_path: Optional[Path] = None) -> VideoTemplateMetadata:
    cv2 = import_cv2()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    ok, frame = capture.read()
    if not ok:
        capture.release()
        raise ValueError(f"Could not read first frame from video: {video_path}")
    if width <= 0 or height <= 0:
        height, width = frame.shape[:2]
    if frame_count <= 0:
        frame_count = 1

    if first_frame_path is not None:
        first_frame_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(first_frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

    capture.release()
    duration_seconds = round(float(frame_count) / fps, 3) if fps > 0 else 0.0
    return VideoTemplateMetadata(
        width=width,
        height=height,
        frame_count=frame_count,
        fps=round(fps, 3),
        duration_seconds=duration_seconds,
    )


def make_relative_path(path: Path, base_dir: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(base_dir).resolve()))
    except ValueError:
        return str(path)


def sanitize_case_id(raw_case_id: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_case_id.strip())
    return value.strip("._-") or "case"


def import_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("OpenCV is required. Install it with: pip install opencv-python") from exc
    return cv2
