from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pipeline as pipeline_module
from evaluation import (
    EvalConfig,
    fixed_interval_review_indices,
    load_eval_manifest,
    run_evaluation,
    run_threshold_calibration,
)
from pipeline import PromptableVideoSegmentationPipeline, ReviewPolicy
from video_io import VideoMetadata


class FakeEvalSegmenter:
    def segment(self, frame_dir: Path, init_box_xyxy, frame_count: int):
        return {
            0: np.ones((4, 6), dtype=np.uint8),
            1: np.ones((4, 6), dtype=np.uint8),
            2: np.zeros((4, 6), dtype=np.uint8),
        }

    def segment_with_corrections(self, frame_dir: Path, init_box_xyxy, frame_count: int, corrections):
        return {
            0: np.ones((4, 6), dtype=np.uint8),
            1: np.ones((4, 6), dtype=np.uint8),
            2: np.ones((4, 6), dtype=np.uint8),
        }


class EvaluationTest(unittest.TestCase):
    def test_fixed_interval_review_indices_skips_initial_prompt_frame(self) -> None:
        self.assertEqual(fixed_interval_review_indices(frame_count=11, interval=5), [5, 10])
        with self.assertRaises(ValueError):
            fixed_interval_review_indices(frame_count=11, interval=0)

    def test_load_eval_manifest_normalizes_case_and_correction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "eval_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "cases": [
                            {
                                "id": "case one",
                                "video_path": "videos/a.mp4",
                                "object_name": "cup",
                                "init_box_xyxy": [0, 0, 5, 3],
                                "expected_challenge": "occlusion",
                                "expected_review_frames": "2,4-5,4",
                                "corrections": [
                                    {
                                        "frame_index": 2,
                                        "type": "negative_point",
                                        "point_xy": "1,2",
                                    }
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            cases = load_eval_manifest(manifest_path)

            self.assertEqual(cases[0].case_id, "case_one")
            self.assertEqual(cases[0].video_path, root / "videos" / "a.mp4")
            self.assertEqual(cases[0].expected_review_frames, (2, 4, 5))
            self.assertEqual(cases[0].corrections[0]["points"][0]["label"], "negative")

    def test_run_evaluation_writes_csv_report_and_after_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video_path = root / "sample.mp4"
            video_path.write_bytes(b"fake")
            manifest_path = root / "eval_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "cases": [
                            {
                                "id": "unit-eval",
                                "video_path": "sample.mp4",
                                "object_name": "toy",
                                "init_box_xyxy": [0, 0, 5, 3],
                                "expected_challenge": "mask_drop",
                                "expected_review_frames": [2],
                                "corrections": [
                                    {
                                        "frame_index": 2,
                                        "type": "positive_point",
                                        "point_xy": [1, 2],
                                    }
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

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

            with patch.object(pipeline_module, "extract_video_frames", side_effect=fake_extract), patch.object(
                pipeline_module, "save_mask", side_effect=fake_save_mask
            ), patch.object(pipeline_module, "render_overlay_video", side_effect=fake_render_overlay):
                summary = run_evaluation(
                    manifest_path=manifest_path,
                    config=EvalConfig(
                        output_dir=root / "outputs",
                        fixed_interval=1,
                        review_policy=ReviewPolicy(area_jump_ratio=0.5),
                    ),
                    pipeline=PromptableVideoSegmentationPipeline(FakeEvalSegmenter()),
                )

            self.assertTrue(summary.csv_path.exists())
            self.assertTrue(summary.report_path.exists())
            self.assertIn("area_jump_ratio", summary.rows[0]["review_policy_config"])
            self.assertEqual(summary.rows[0]["queue_review_frames"], 1)
            self.assertEqual(summary.rows[0]["expected_review_frames_labeled"], True)
            self.assertEqual(summary.rows[0]["expected_review_frames"], 1)
            self.assertEqual(summary.rows[0]["queue_label_hits"], 1)
            self.assertEqual(summary.rows[0]["queue_missed_expected_frames"], 0)
            self.assertEqual(summary.rows[0]["queue_precision"], 1.0)
            self.assertEqual(summary.rows[0]["queue_recall"], 1.0)
            self.assertEqual(summary.rows[0]["queue_f1"], 1.0)
            self.assertEqual(summary.rows[0]["fixed_interval_label_hits"], 1)
            self.assertEqual(summary.rows[0]["fixed_interval_false_positives"], 1)
            self.assertEqual(summary.rows[0]["fixed_interval_f1"], 0.667)
            self.assertIn("mask_area_dropped", summary.rows[0]["queue_reason_counts"])
            self.assertEqual(summary.rows[0]["fixed_interval_review_frames"], 2)
            self.assertEqual(summary.rows[0]["after_review_frames"], 0)
            self.assertEqual(summary.rows[0]["after_reason_counts"], "{}")
            self.assertTrue((root / "outputs" / "unit-eval" / "overlay_after.mp4").exists())
            self.assertTrue((root / "outputs" / "unit-eval" / "comparison.json").exists())

            with summary.csv_path.open(encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(rows[0]["case_id"], "unit-eval")
            report = summary.report_path.read_text(encoding="utf-8")
            self.assertIn("MaskReview Evaluation Report", report)
            self.assertIn("Queue label precision/recall/F1: 1.0 / 1.0 / 1.0", report)
            self.assertIn("Queue Reason Counts", report)

    def test_run_threshold_calibration_writes_policy_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video_path = root / "sample.mp4"
            video_path.write_bytes(b"fake")
            manifest_path = root / "eval_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "cases": [
                            {
                                "id": "unit-calibration",
                                "video_path": "sample.mp4",
                                "object_name": "toy",
                                "init_box_xyxy": [0, 0, 5, 3],
                                "expected_challenge": "mask_drop",
                                "expected_review_frames": [2],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

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

            with patch.object(pipeline_module, "extract_video_frames", side_effect=fake_extract), patch.object(
                pipeline_module, "save_mask", side_effect=fake_save_mask
            ), patch.object(pipeline_module, "render_overlay_video", side_effect=fake_render_overlay):
                summary = run_threshold_calibration(
                    manifest_path=manifest_path,
                    config=EvalConfig(output_dir=root / "calibration", fixed_interval=1),
                    policies=[
                        ("sensitive-test", ReviewPolicy(area_jump_ratio=0.4)),
                        ("conservative-test", ReviewPolicy(area_jump_ratio=0.9)),
                    ],
                    pipeline_factory=lambda: PromptableVideoSegmentationPipeline(FakeEvalSegmenter()),
                )

            self.assertTrue(summary.csv_path.exists())
            self.assertTrue(summary.report_path.exists())
            self.assertEqual([row["policy_name"] for row in summary.rows], ["sensitive-test", "conservative-test"])
            self.assertEqual(summary.rows[0]["labeled_cases"], 1)
            self.assertEqual(summary.rows[0]["queue_label_hits"], 1)
            self.assertEqual(summary.rows[0]["queue_recall"], 1.0)
            report = summary.report_path.read_text(encoding="utf-8")
            self.assertIn("Threshold Calibration", report)
            self.assertIn("Queue P/R/F1", report)
            self.assertIn("sensitive-test", report)


if __name__ == "__main__":
    unittest.main()
