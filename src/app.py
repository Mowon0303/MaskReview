from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    import gradio as gr
except ImportError:  # pragma: no cover
    gr = None

from pipeline import (
    DEFAULT_SAM2_CHECKPOINT,
    DEFAULT_SAM2_MODEL_CFG,
    PromptableVideoSegmentationPipeline,
    VideoSegmentationRequest,
    parse_box_xyxy,
)


def normalize_gradio_video_path(video: Any) -> str:
    if isinstance(video, str):
        return video
    if isinstance(video, dict) and "path" in video:
        return str(video["path"])
    if hasattr(video, "name"):
        return str(video.name)
    raise ValueError("Upload a video file first.")


def build_runner(
    pipeline: PromptableVideoSegmentationPipeline,
    output_dir: Path,
    sam2_model_cfg: str,
    sam2_checkpoint: Path,
    device: str,
    vos_optimized: bool,
):
    def run_demo(video: Any, box_xyxy: str):
        video_path = normalize_gradio_video_path(video)
        init_box = parse_box_xyxy(box_xyxy)

        request = VideoSegmentationRequest(
            video_path=Path(video_path),
            init_box_xyxy=init_box,
            output_dir=output_dir,
            sam2_model_cfg=sam2_model_cfg,
            sam2_checkpoint=sam2_checkpoint,
            device=device,
            vos_optimized=vos_optimized,
        )
        result = pipeline.run(request)
        metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
        review_summary = (
            f"queued_frames: {len(result.review_queue)}\n"
            f"estimated_min_interactions: {result.estimated_min_interactions}\n"
            f"estimated_interactions_per_video_minute: "
            f"{result.interactions_per_video_minute}"
        )
        artifact_summary = (
            f"run_dir: {result.run_dir}\n"
            f"overlay: {result.overlay_video_path}\n"
            f"masks: {result.mask_dir}\n"
            f"review_queue: {result.review_queue_path}\n"
            f"metrics: {result.metrics_path}"
        )
        return str(result.overlay_video_path), review_summary, result.review_queue, artifact_summary, metrics

    return run_demo


def build_demo(args: argparse.Namespace):
    pipeline = PromptableVideoSegmentationPipeline()
    run_demo = build_runner(
        pipeline=pipeline,
        output_dir=args.output_dir,
        sam2_model_cfg=args.sam2_model_cfg,
        sam2_checkpoint=args.sam2_checkpoint,
        device=args.device,
        vos_optimized=args.vos_optimized,
    )

    with gr.Blocks(title="MaskReview") as demo:
        gr.Markdown("# MaskReview")
        with gr.Row():
            video = gr.Video(label="Input video")
            output = gr.Video(label="SAM2 propagation overlay")
        box_xyxy = gr.Textbox(label="Initial object box x1,y1,x2,y2", placeholder="80,120,260,420")
        run_button = gr.Button("Run review pass", variant="primary")
        with gr.Row():
            review = gr.Textbox(label="Review KPI")
            artifact_summary = gr.Textbox(label="Saved artifacts", lines=4)
        review_queue = gr.JSON(label="Low-confidence frame queue")
        metrics = gr.JSON(label="metrics.json")
        run_button.click(
            run_demo,
            inputs=[video, box_xyxy],
            outputs=[output, review, review_queue, artifact_summary, metrics],
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM2 propagation review-loop demo.")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link.")
    parser.add_argument("--server-name", default="0.0.0.0", help="Gradio server host.")
    parser.add_argument("--server-port", type=int, default=7860, help="Gradio server port.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for run artifacts.")
    parser.add_argument("--sam2-model-cfg", default=DEFAULT_SAM2_MODEL_CFG, help="SAM2 model config name/path.")
    parser.add_argument(
        "--sam2-checkpoint",
        type=Path,
        default=DEFAULT_SAM2_CHECKPOINT,
        help="Path to a SAM2 checkpoint .pt file.",
    )
    parser.add_argument("--device", default="cuda", help="Torch device, usually cuda on cloud GPU.")
    parser.add_argument(
        "--vos-optimized",
        action="store_true",
        help="Enable SAM2's compiled VOS optimization when supported by the installed SAM2 version.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if gr is None:
        raise RuntimeError("Install gradio first: pip install gradio")

    demo = build_demo(args)
    demo.launch(
        share=args.share,
        server_name=args.server_name,
        server_port=args.server_port,
    )


if __name__ == "__main__":
    main()
