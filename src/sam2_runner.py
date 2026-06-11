from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

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

    def segment(
        self,
        frame_dir: Path,
        init_box_xyxy: tuple[int, int, int, int],
        frame_count: int,
    ) -> dict[int, np.ndarray]:
        torch = self._import_torch()
        predictor = self._load_predictor()
        box = np.array(init_box_xyxy, dtype=np.float32)

        with torch.inference_mode(), self._autocast_context(torch):
            state = self._init_state(predictor, frame_dir)
            self._add_box_prompt(predictor, state, box)

            masks: dict[int, np.ndarray] = {}
            for output in predictor.propagate_in_video(state):
                frame_idx, obj_ids, mask_logits = output[:3]
                mask = self._select_object_mask(obj_ids, mask_logits)
                if mask is not None:
                    masks[int(frame_idx)] = mask

        if not masks:
            raise RuntimeError("SAM2 did not return any propagated masks.")
        missing = sorted(set(range(frame_count)) - set(masks))
        if missing:
            missing_preview = ", ".join(str(idx) for idx in missing[:10])
            raise RuntimeError(f"SAM2 did not return masks for frames: {missing_preview}")
        return masks

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

    def _add_box_prompt(self, predictor, state, box: np.ndarray) -> None:
        try:
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=0,
                obj_id=self.obj_id,
                box=box,
            )
        except TypeError:
            predictor.add_new_points_or_box(state, 0, self.obj_id, box=box)

    def _select_object_mask(self, obj_ids, mask_logits) -> np.ndarray | None:
        obj_id_list = [int(obj_id) for obj_id in obj_ids]
        if self.obj_id not in obj_id_list:
            return None

        obj_index = obj_id_list.index(self.obj_id)
        selected = mask_logits[obj_index]
        if hasattr(selected, "detach"):
            selected = selected.detach().cpu().numpy()
        selected = np.squeeze(np.asarray(selected))
        return (selected > 0.0).astype(np.uint8)

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

