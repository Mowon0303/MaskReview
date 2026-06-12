from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pipeline
from corrections import build_correction, save_correction
from pipeline import (
    CorrectionPropagationRequest,
    PromptableVideoSegmentationPipeline,
    VideoSegmentationRequest,
    build_review_queue,
    compute_mask_iou,
    count_review_reasons,
    detect_review_frames,
    parse_box_xyxy,
)
from video_io import VideoMetadata


class FakeSegmenter:
    def segment(self, frame_dir: Path, init_box_xyxy, frame_count: int):
        return {
            0: np.ones((4, 6), dtype=np.uint8),
            1: np.ones((4, 6), dtype=np.uint8),
            2: np.zeros((4, 6), dtype=np.uint8),
        }


class FakeCorrectionSegmenter(FakeSegmenter):
    def __init__(self) -> None:
        self.last_corrections = []

    def segment_with_corrections(self, frame_dir: Path, init_box_xyxy, frame_count: int, corrections):
        self.last_corrections = corrections
        return {
            0: np.ones((4, 6), dtype=np.uint8),
            1: np.ones((4, 6), dtype=np.uint8),
            2: np.ones((4, 6), dtype=np.uint8),
        }


class PipelineTest(unittest.TestCase):
    def test_detect_review_frames_flags_area_jumps(self) -> None:
        self.assertEqual(detect_review_frames([100, 105, 20], jump_ratio=0.6), [2])
        self.assertEqual(detect_review_frames([100, 110, 115], jump_ratio=0.6), [])

    def test_build_review_queue_describes_minimal_corrections(self) -> None:
        queue = build_review_queue([100, 105, 20], jump_ratio=0.6)
        self.assertEqual(len(queue), 1)
        self.assertEqual(queue[0]["frame_index"], 2)
        self.assertEqual(queue[0]["reason"], "mask_area_dropped")
        self.assertEqual(queue[0]["reasons"], ["mask_area_dropped"])
        self.assertEqual(queue[0]["estimated_interactions"], 1)
        self.assertIn("point", queue[0]["recommended_correction"])

    def test_build_review_queue_flags_center_jump_and_iou_drop(self) -> None:
        first = np.zeros((10, 10), dtype=np.uint8)
        second = np.zeros((10, 10), dtype=np.uint8)
        first[2:4, 1:3] = 1
        second[2:4, 7:9] = 1

        queue = build_review_queue(
            [int(first.sum()), int(second.sum())],
            masks={0: first, 1: second},
        )

        self.assertEqual(len(queue), 1)
        self.assertEqual(queue[0]["frame_index"], 1)
        self.assertEqual(queue[0]["reason"], "mask_center_jumped")
        self.assertIn("mask_iou_dropped", queue[0]["reasons"])
        self.assertEqual(count_review_reasons(queue)["mask_center_jumped"], 1)
        self.assertEqual(count_review_reasons(queue)["mask_iou_dropped"], 1)

    def test_build_review_queue_flags_edge_contact_and_area_decline(self) -> None:
        first = np.zeros((10, 10), dtype=np.uint8)
        second = np.zeros((10, 10), dtype=np.uint8)
        first[3:7, 3:7] = 1
        second[3:7, 7:10] = 1

        edge_queue = build_review_queue(
            [int(first.sum()), int(second.sum())],
            masks={0: first, 1: second},
            iou_drop_threshold=-1.0,
            center_jump_ratio=1.0,
        )
        self.assertEqual(edge_queue[0]["reason"], "mask_touches_frame_edge")

        decline_queue = build_review_queue([100, 80, 60], area_decline_ratio=0.35)
        self.assertEqual(decline_queue[0]["frame_index"], 2)
        self.assertEqual(decline_queue[0]["reason"], "mask_area_trending_down")

    def test_compute_mask_iou(self) -> None:
        first = np.zeros((4, 4), dtype=np.uint8)
        second = np.zeros((4, 4), dtype=np.uint8)
        first[0:2, 0:2] = 1
        second[1:3, 1:3] = 1

        self.assertAlmostEqual(compute_mask_iou(first, second), 1 / 7)

    def test_parse_box_xyxy(self) -> None:
        self.assertEqual(parse_box_xyxy("1, 2, 30, 40"), (1, 2, 30, 40))
        with self.assertRaises(ValueError):
            parse_box_xyxy("1,2,3")
        with self.assertRaises(ValueError):
            parse_box_xyxy("10,2,3,40")

    def test_pipeline_writes_expected_artifacts_with_fake_segmenter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video_path = root / "sample.mp4"
            video_path.write_bytes(b"fake")

            def fake_extract(video_path_arg: Path, frame_dir: Path):
                frame_dir.mkdir(parents=True, exist_ok=True)
                return VideoMetadata(fps=5.0, width=6, height=4, frame_count=3)

            def fake_save_mask(mask: np.ndarray, output_path: Path) -> None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes((mask * 255).astype(np.uint8).tobytes())

            def fake_render_overlay(**kwargs):
                output_path = kwargs["output_path"]
                output_path.write_bytes(b"mp4")
                return output_path

            with patch.object(pipeline, "extract_video_frames", side_effect=fake_extract), patch.object(
                pipeline, "save_mask", side_effect=fake_save_mask
            ), patch.object(pipeline, "render_overlay_video", side_effect=fake_render_overlay):
                result = PromptableVideoSegmentationPipeline(FakeSegmenter()).run(
                    VideoSegmentationRequest(
                        video_path=video_path,
                        init_box_xyxy=(0, 0, 5, 3),
                        output_dir=root / "outputs",
                        run_id="unit-test",
                    )
                )

            self.assertTrue(result.overlay_video_path.exists())
            self.assertTrue(result.metrics_path.exists())
            self.assertTrue(result.review_queue_path.exists())
            self.assertEqual(result.frame_count, 3)
            self.assertEqual(result.review_frame_indices, [2])
            self.assertEqual(result.estimated_min_interactions, 1)
            self.assertEqual(result.interactions_per_video_minute, 100.0)
            self.assertEqual(len(list(result.mask_dir.glob("*.png"))), 3)

            metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(metrics["frame_count"], 3)
            self.assertEqual(metrics["mask_areas"], [24, 24, 0])
            self.assertEqual(metrics["estimated_min_interactions"], 1)
            self.assertEqual(metrics["estimated_interactions_per_video_minute"], 100.0)
            self.assertEqual(metrics["review_queue"][0]["frame_index"], 2)

            review_queue = json.loads(result.review_queue_path.read_text(encoding="utf-8"))
            self.assertEqual(review_queue[0]["mask_path"], str(result.mask_dir / "000002.png"))

    def test_pipeline_writes_after_artifacts_with_corrections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video_path = root / "sample.mp4"
            video_path.write_bytes(b"fake")
            segmenter = FakeCorrectionSegmenter()

            def fake_extract(video_path_arg: Path, frame_dir: Path):
                frame_dir.mkdir(parents=True, exist_ok=True)
                return VideoMetadata(fps=5.0, width=6, height=4, frame_count=3)

            def fake_save_mask(mask: np.ndarray, output_path: Path) -> None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes((mask * 255).astype(np.uint8).tobytes())

            def fake_render_overlay(**kwargs):
                output_path = kwargs["output_path"]
                output_path.write_bytes(b"mp4")
                return output_path

            with patch.object(pipeline, "extract_video_frames", side_effect=fake_extract), patch.object(
                pipeline, "save_mask", side_effect=fake_save_mask
            ), patch.object(pipeline, "render_overlay_video", side_effect=fake_render_overlay):
                baseline = PromptableVideoSegmentationPipeline(segmenter).run(
                    VideoSegmentationRequest(
                        video_path=video_path,
                        init_box_xyxy=(0, 0, 5, 3),
                        output_dir=root / "outputs",
                        run_id="unit-test",
                    )
                )
                save_correction(
                    baseline.run_dir,
                    build_correction(2, "positive_point", point_xy="1,2"),
                )
                corrected = PromptableVideoSegmentationPipeline(segmenter).run_with_corrections(
                    CorrectionPropagationRequest(
                        run_dir=baseline.run_dir,
                        video_path=video_path,
                        init_box_xyxy=(0, 0, 5, 3),
                    )
                )

            self.assertEqual(segmenter.last_corrections[0]["frame_index"], 2)
            self.assertTrue(corrected.overlay_video_path.exists())
            self.assertTrue(corrected.metrics_path.exists())
            self.assertTrue(corrected.comparison_path.exists())
            self.assertEqual(len(list(corrected.mask_dir.glob("*.png"))), 3)

            metrics_after = json.loads(corrected.metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(metrics_after["phase"], "after_corrections")
            self.assertEqual(metrics_after["mask_areas"], [24, 24, 24])
            self.assertEqual(metrics_after["actual_interactions"], 1)
            self.assertEqual(metrics_after["review_frame_indices"], [])

            comparison = json.loads(corrected.comparison_path.read_text(encoding="utf-8"))
            self.assertEqual(comparison["before_review_frame_count"], 1)
            self.assertEqual(comparison["after_review_frame_count"], 0)
            self.assertEqual(comparison["review_frame_count_delta"], -1)


if __name__ == "__main__":
    unittest.main()
