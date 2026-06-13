from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Sam2VideoSegmenter:
    model_cfg: str
    checkpoint_path: Path
    device: str = "cuda"
    obj_id: int = 1
    vos_optimized: bool = False
    offload_video_to_cpu: bool = True
    async_loading_frames: bool = False

    def __post_init__(self) -> None:
        self.checkpoint_path = Path(self.checkpoint_path)
        self._predictor = None
        # Per-frame SAM2 confidence (sigmoid of the peak object logit), filled after segment().
        self.last_confidences: dict[int, float] = {}

    def segment(
        self,
        frame_dir: Path,
        init_box_xyxy: tuple[int, int, int, int],
        frame_count: int,
        init_frame_index: int = 0,
    ) -> dict[int, np.ndarray]:
        return self.segment_with_corrections(
            frame_dir=frame_dir,
            init_box_xyxy=init_box_xyxy,
            frame_count=frame_count,
            corrections=[],
            init_frame_index=init_frame_index,
        )

    def segment_with_corrections(
        self,
        frame_dir: Path,
        init_box_xyxy: tuple[int, int, int, int],
        frame_count: int,
        corrections: list[dict[str, Any]],
        init_frame_index: int = 0,
    ) -> dict[int, np.ndarray]:
        torch = self._import_torch()
        predictor = self._load_predictor()
        box = np.array(init_box_xyxy, dtype=np.float32)
        seed = int(init_frame_index)
        if seed < 0 or seed >= max(int(frame_count), 1):
            raise ValueError(
                f"init_frame_index {seed} is out of range for {frame_count} frames."
            )

        masks: dict[int, np.ndarray] = {}
        confidences: dict[int, float] = {}
        with torch.inference_mode(), self._autocast_context(torch):
            state = self._init_state(predictor, frame_dir)
            self._add_box_prompt(predictor, state, frame_idx=seed, box=box)
            for correction in sorted(corrections, key=lambda item: int(item["frame_index"])):
                self._add_correction_prompt(predictor, state, correction)

            # Propagate forward from the seed frame to the end, then (when the object is
            # seeded mid-video) backward from the seed to frame 0, so a target first prompted
            # partway through the clip is tracked in both directions.
            self._collect_propagation(
                predictor, state, masks, confidences,
                start_frame_idx=seed, max_frames=frame_count, reverse=False,
            )
            if seed > 0:
                self._collect_propagation(
                    predictor, state, masks, confidences,
                    start_frame_idx=seed, max_frames=seed + 1, reverse=True,
                )

        self.last_confidences = confidences
        if not masks:
            raise RuntimeError("SAM2 did not return any propagated masks.")
        missing = sorted(set(range(frame_count)) - set(masks))
        if missing:
            missing_preview = ", ".join(str(idx) for idx in missing[:10])
            raise RuntimeError(f"SAM2 did not return masks for frames: {missing_preview}")
        return masks

    def _collect_propagation(
        self,
        predictor,
        state,
        masks: dict[int, np.ndarray],
        confidences: dict[int, float],
        start_frame_idx: int,
        max_frames: int,
        reverse: bool,
    ) -> None:
        for output in predictor.propagate_in_video(
            state,
            start_frame_idx=start_frame_idx,
            max_frame_num_to_track=max_frames,
            reverse=reverse,
        ):
            frame_idx, obj_ids, mask_logits = output[:3]
            mask, confidence = self._select_object_mask(obj_ids, mask_logits)
            if mask is not None:
                masks[int(frame_idx)] = mask
                confidences[int(frame_idx)] = confidence

    def _load_predictor(self):
        if self._predictor is not None:
            return self._predictor
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                "SAM2 checkpoint not found: "
                f"{self.checkpoint_path}. Download one into checkpoints/ or pass "
                "--sam2-checkpoint on the app command line."
            )

        try:
            from sam2.build_sam import build_sam2_video_predictor
        except ImportError as exc:
            raise ImportError(
                "SAM2 is not installed. On the cloud GPU machine, install the official "
                "repo with: pip install git+https://github.com/facebookresearch/sam2.git"
            ) from exc

        build_kwargs = {"device": self.device}
        if self.vos_optimized:
            build_kwargs["vos_optimized"] = True
        try:
            self._predictor = build_sam2_video_predictor(
                self.model_cfg,
                str(self.checkpoint_path),
                **build_kwargs,
            )
        except TypeError:
            self._predictor = build_sam2_video_predictor(
                self.model_cfg,
                str(self.checkpoint_path),
            )
        return self._predictor

    def _init_state(self, predictor, frame_dir: Path):
        kwargs = {
            "video_path": str(frame_dir),
            "offload_video_to_cpu": self.offload_video_to_cpu,
            "async_loading_frames": self.async_loading_frames,
        }
        try:
            return predictor.init_state(**kwargs)
        except TypeError:
            return predictor.init_state(video_path=str(frame_dir))

    def _add_box_prompt(self, predictor, state, frame_idx: int, box: np.ndarray) -> None:
        try:
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=frame_idx,
                obj_id=self.obj_id,
                box=box,
            )
        except TypeError:
            predictor.add_new_points_or_box(state, frame_idx, self.obj_id, box=box)

    def _add_points_prompt(
        self,
        predictor,
        state,
        frame_idx: int,
        points: list[dict[str, Any]],
        default_label: int = 1,
        clear_old_points: bool = True,
    ) -> None:
        coords = np.array([[float(p["x"]), float(p["y"])] for p in points], dtype=np.float32)
        labels = np.array(
            [
                1 if str(p.get("label", "")).lower().startswith("pos")
                else (0 if "label" in p else default_label)
                for p in points
            ],
            dtype=np.int32,
        )
        try:
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=frame_idx,
                obj_id=self.obj_id,
                points=coords,
                labels=labels,
                clear_old_points=clear_old_points,
            )
        except TypeError:
            predictor.add_new_points_or_box(
                state,
                frame_idx,
                self.obj_id,
                points=coords,
                labels=labels,
                clear_old_points=clear_old_points,
            )

    def _add_correction_prompt(self, predictor, state, correction: dict[str, Any]) -> None:
        frame_idx = int(correction["frame_index"])
        correction_type = str(correction["type"])
        if correction_type == "tight_box":
            self._add_box_prompt(
                predictor,
                state,
                frame_idx=frame_idx,
                box=np.array(correction["box_xyxy"], dtype=np.float32),
            )
            return

        if correction_type in {"positive_point", "negative_point"}:
            points = correction.get("points") or []
            if not points:
                raise ValueError(f"Point correction has no points: {correction}")
            default_label = 1 if correction_type == "positive_point" else 0
            self._add_points_prompt(
                predictor,
                state,
                frame_idx=frame_idx,
                points=points,
                default_label=default_label,
                clear_old_points=frame_idx != 0,
            )
            return

        raise ValueError(f"Unsupported correction type: {correction_type}")

    def _select_object_mask(self, obj_ids, mask_logits) -> tuple[np.ndarray | None, float]:
        """Return (binary mask, confidence) for this object.

        Confidence is the sigmoid of the peak object logit — a SAM2-grounded proxy for
        "how strongly the model believes the object is present this frame" (~1.0 when the
        object is confidently tracked, <0.5 when the predicted foreground is all negative,
        i.e. the object is likely lost/occluded). Returns (None, 0.0) if the object is
        absent from this frame's outputs.
        """
        obj_id_list = [int(obj_id) for obj_id in obj_ids]
        if self.obj_id not in obj_id_list:
            return None, 0.0

        obj_index = obj_id_list.index(self.obj_id)
        selected = mask_logits[obj_index]
        if hasattr(selected, "detach"):
            selected = selected.detach().cpu().numpy()
        selected = np.squeeze(np.asarray(selected))
        peak = float(selected.max()) if selected.size else float("-inf")
        confidence = 1.0 / (1.0 + np.exp(-float(np.clip(peak, -30.0, 30.0))))
        return (selected > 0.0).astype(np.uint8), float(confidence)

    def _import_torch(self):
        try:
            import torch
        except ImportError as exc:
            raise ImportError("PyTorch is required to run SAM2 video segmentation.") from exc

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("Device is set to cuda, but torch.cuda.is_available() is false.")
        return torch

    def _autocast_context(self, torch):
        if self.device.startswith("cuda"):
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

