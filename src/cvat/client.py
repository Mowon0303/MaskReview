"""Live cvat-sdk client for the CVAT plugin (P8 step 2).

The only module that touches cvat-sdk / the network. Thin wrappers around the high-level
SDK (tasks: frames, annotations, dataset) plus the low-level issues API. The offline
`export` layer does all the format work; this layer just moves bytes to/from CVAT.

Verified against cvat-sdk 2.68.0; method names drift between versions (see
docs/cvat_plugin.md §10) — re-check if you upgrade. Needs a reachable CVAT; not unit-tested
here (integration-tested against a real task on the user's machine).
"""

from __future__ import annotations

import json
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

try:  # pragma: no cover - import guarded so the rest of the repo imports without cvat-sdk
    from cvat_sdk import make_client, models
except ImportError:  # pragma: no cover
    make_client = None
    models = None

from cvat.export import IssuePayload, masks_to_coco


@dataclass
class PulledTask:
    """What the plugin needs from a CVAT task to run a review pass."""

    task_id: int
    job_id: int
    frame_count: int
    width: int
    height: int
    frame_dir: Path
    frame_names: list[str]                      # CVAT's per-frame names (for COCO alignment)
    init_box_xyxy: Optional[tuple[int, int, int, int]]
    init_frame_index: int
    labels: dict[str, int] = field(default_factory=dict)  # name -> label_id


class CvatBridge:
    """Thin cvat-sdk bridge. Construct with host + (user, password); each method opens a
    short-lived client. Credentials should come from env vars, never hard-coded."""

    def __init__(self, host: str, credentials: tuple[str, str]):
        if make_client is None:
            raise ImportError("cvat-sdk is not installed. Run: pip install cvat-sdk")
        self._host = host
        self._credentials = credentials

    # --- pull -----------------------------------------------------------------

    def pull_task(self, task_id: int, frame_dir: Path, *, seed_frame: int = 0) -> PulledTask:
        """Download the task's frames + read the seed-frame rectangle.

        Frames are written as ``frame_dir/<cvat_name>`` so MaskReview runs on the *exact*
        frames CVAT indexes (docs/cvat_plugin.md §6.1) and the COCO file_names line up.
        """
        frame_dir = Path(frame_dir)
        frame_dir.mkdir(parents=True, exist_ok=True)
        with make_client(self._host, credentials=self._credentials) as client:
            task = client.tasks.retrieve(task_id)
            meta = task.get_meta()
            frame_count = int(meta.size)
            frames_info = list(meta.frames)
            width = int(frames_info[0].width)
            height = int(frames_info[0].height)
            # Image tasks expose one meta.frames entry per frame; a VIDEO task exposes a single
            # entry (the video file). CVAT's COCO export names video frames frame_%06d.png
            # (id = frame+1), which masks_to_coco already mirrors — so match that here.
            if len(frames_info) == frame_count:
                frame_names = [Path(str(fi.name)).name for fi in frames_info]
            else:
                frame_names = [f"frame_{i:06d}.png" for i in range(frame_count)]

            for i in range(frame_count):
                raw = task.get_frame(i, quality="original")
                (frame_dir / f"{i:06d}.jpg").write_bytes(raw.read())

            labels = {str(l.name): int(l.id) for l in task.get_labels()}
            job_id = int(task.get_jobs()[0].id)
            init_box = self._read_seed_box(task, seed_frame)

        return PulledTask(
            task_id=task_id,
            job_id=job_id,
            frame_count=frame_count,
            width=width,
            height=height,
            frame_dir=frame_dir,
            frame_names=frame_names,
            init_box_xyxy=init_box,
            init_frame_index=seed_frame,
            labels=labels,
        )

    @staticmethod
    def _read_seed_box(task, seed_frame: int) -> Optional[tuple[int, int, int, int]]:
        annotations = task.get_annotations()
        for shape in getattr(annotations, "shapes", []) or []:
            if str(shape.type) == "rectangle" and int(shape.frame) == int(seed_frame):
                x1, y1, x2, y2 = (float(v) for v in shape.points[:4])
                return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
        return None

    # --- push -----------------------------------------------------------------

    def import_masks_coco(
        self,
        task_id: int,
        masks: dict[int, Any],
        *,
        label: str = "object",
        frame_names: Optional[list[str]] = None,
        conv_mask_to_poly: bool = False,
    ) -> None:
        """Convert masks -> COCO and import as the task's annotations (async; SDK polls)."""
        name_fn = (lambda i: frame_names[i]) if frame_names else (lambda i: f"frame_{i:06d}")
        coco = masks_to_coco(masks, label=label, frame_name=name_fn)
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "coco.zip"
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("annotations/instances_default.json", json.dumps(coco))
            with make_client(self._host, credentials=self._credentials) as client:
                task = client.tasks.retrieve(task_id)
                task.import_annotations("COCO 1.0", str(zip_path), conv_mask_to_poly=conv_mask_to_poly)

    def create_issues(self, task_id: int, issues: list[IssuePayload]) -> int:
        """Raise one CVAT issue per payload on the task's job. Returns the count created."""
        created = 0
        with make_client(self._host, credentials=self._credentials) as client:
            task = client.tasks.retrieve(task_id)
            job_id = int(task.get_jobs()[0].id)
            for issue in issues:
                position = issue.position or [10.0, 10.0, 60.0, 60.0]  # fallback marker
                client.api_client.issues_api.create(
                    models.IssueWriteRequest(
                        frame=int(issue.frame),
                        position=[float(v) for v in position],
                        job=job_id,
                        message=str(issue.message),
                    )
                )
                created += 1
        return created

    # --- export ---------------------------------------------------------------

    def export_annotations(
        self,
        task_id: int,
        out_path: Path,
        *,
        fmt: str = "COCO 1.0",
        include_images: bool = False,
    ) -> Path:
        """Download the (human-corrected) annotations after review."""
        out_path = Path(out_path)
        with make_client(self._host, credentials=self._credentials) as client:
            task = client.tasks.retrieve(task_id)
            task.export_dataset(fmt, str(out_path), include_images=include_images)
        return out_path
