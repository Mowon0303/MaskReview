from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class VideoMetadata:
    fps: float
    width: int
    height: int
    frame_count: int


def extract_video_frames(video_path: Path, frame_dir: Path) -> VideoMetadata:
    cv2 = _import_cv2()
    frame_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    frame_count = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if width <= 0 or height <= 0:
            height, width = frame.shape[:2]
        frame_path = frame_dir / f"{frame_count:06d}.jpg"
        cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        frame_count += 1

    capture.release()
    if frame_count == 0:
        raise ValueError(f"No frames were decoded from video: {video_path}")
    return VideoMetadata(fps=fps, width=width, height=height, frame_count=frame_count)


def save_mask(mask: np.ndarray, output_path: Path) -> None:
    cv2 = _import_cv2()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_u8 = (np.asarray(mask) > 0).astype(np.uint8) * 255
    cv2.imwrite(str(output_path), mask_u8)


def render_overlay_video(
    frame_dir: Path,
    masks: dict[int, np.ndarray],
    output_path: Path,
    fps: float,
    init_box_xyxy: tuple[int, int, int, int] | None = None,
    color_rgb: tuple[int, int, int] = (0, 255, 180),
    alpha: float = 0.55,
) -> Path:
    cv2 = _import_cv2()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(frame_dir.glob("*.jpg"))
    if not frame_paths:
        raise ValueError(f"No extracted frames found in: {frame_dir}")

    first_frame = cv2.imread(str(frame_paths[0]))
    if first_frame is None:
        raise ValueError(f"Could not read extracted frame: {frame_paths[0]}")
    height, width = first_frame.shape[:2]

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps if fps > 0 else 30.0,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create overlay video writer: {output_path}")

    color_bgr = np.array(color_rgb[::-1], dtype=np.float32)
    for frame_idx, frame_path in enumerate(frame_paths):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            writer.release()
            raise ValueError(f"Could not read extracted frame: {frame_path}")

        mask = np.asarray(masks.get(frame_idx, np.zeros((height, width), dtype=np.uint8)))
        mask = np.squeeze(mask).astype(bool)
        if mask.shape != (height, width):
            mask = cv2.resize(
                mask.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        blended = frame.astype(np.float32)
        blended[mask] = blended[mask] * (1.0 - alpha) + color_bgr * alpha
        frame = np.clip(blended, 0, 255).astype(np.uint8)

        if init_box_xyxy is not None:
            x1, y1, x2, y2 = init_box_xyxy
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 180), 2)

        writer.write(frame)

    writer.release()
    return output_path


def _import_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("OpenCV is required. Install it with: pip install opencv-python") from exc
    return cv2
