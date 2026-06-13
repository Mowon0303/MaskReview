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
    def segment(self, frame_dir: Path, init_box_xyxy, frame_count: int, init_frame_index: int = 0):
        self.last_init_frame_index = init_frame_index
        return {
            0: np.ones((4, 6), dtype=np.uint8),
            1: np.ones((4, 6), dtype=np.uint8),
            2: np.zeros((4, 6), dtype=np.uint8),
        }


class FakeCorrectionSegmenter(FakeSegmenter):
    def __init__(self) -> None:
        self.last_corrections = []

    def segment_with_corrections(
        self, frame_dir: Path, init_box_xyxy, frame_count: int, corrections, init_frame_index: int = 0
    ):
        self.last_corrections = corrections
        self.last_init_frame_index = init_frame_index
        return {
            0: np.ones((4, 6), dtype=np.uint8),
            1: np.ones((4, 6), dtype=np.uint8),
            2: np.ones((4, 6), dtype=np.uint8),
        }


class FakeConfidenceSegmenter(FakeSegmenter):
    """Like FakeSegmenter but also exposes per-frame SAM2 confidence (frame 2 is low)."""

    def __init__(self) -> None:
        self.last_confidences = {0: 0.95, 1: 0.9, 2: 0.1}


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

    def test_build_review_queue_flags_low_sam2_confidence(self) -> None:
        # geometrically stable areas, but frame 2 has low SAM2 confidence
        queue = build_review_queue(
            [100, 100, 100, 100],
            confidences={0: 0.99, 1: 0.98, 2: 0.2, 3: 0.97},
            low_confidence_threshold=0.5,
        )
        self.assertEqual([item["frame_index"] for item in queue], [2])
        self.assertEqual(queue[0]["reason"], "low_mask_confidence")
        self.assertEqual(queue[0]["diagnostics"]["confidence"], 0.2)
        # without confidences the behaviour is unchanged (geometric only)
        self.assertEqual(build_review_queue([100, 100, 100, 100]), [])

    def test_build_review_queue_flags_oversized_sprawled_blob(self) -> None:
        # frames 0-3: a small compact object; frame 4: a sprawled blob (two full-width
        # bands) that is large, far above the typical size, and only ~40% bbox fill.
        masks = {}
        for i in range(4):
            m = np.zeros((100, 100), dtype=np.uint8)
            m[45:55, 45:55] = 1  # 100 px, centered, off the edges
            masks[i] = m
        blob = np.zeros((100, 100), dtype=np.uint8)
        blob[0:20, :] = 1
        blob[80:100, :] = 1  # 4000 px, bbox spans the whole frame -> fill 0.40
        masks[4] = blob
        areas = [int(masks[i].sum()) for i in range(5)]

        queue = build_review_queue(areas, masks=masks)
        by_frame = {item["frame_index"]: item for item in queue}

        self.assertIn(4, by_frame)
        self.assertIn("mask_oversized", by_frame[4]["reasons"])
        diag = by_frame[4]["diagnostics"]
        self.assertAlmostEqual(diag["fill_ratio"], 0.40, places=2)
        self.assertGreaterEqual(diag["frame_fraction"], 0.15)
        # the small-object frames are never flagged oversized
        self.assertNotIn(0, by_frame)

    def test_build_review_queue_spares_solid_large_object(self) -> None:
        # frame 4 is large (64% of frame) and far above the typical size, but SOLID
        # (fill 1.0) -- a legitimately large object (e.g. a close-up car), not over-seg.
        masks = {}
        for i in range(4):
            m = np.zeros((100, 100), dtype=np.uint8)
            m[45:55, 45:55] = 1
            masks[i] = m
        solid = np.zeros((100, 100), dtype=np.uint8)
        solid[10:90, 10:90] = 1  # 6400 px, fully fills its bbox
        masks[4] = solid
        areas = [int(masks[i].sum()) for i in range(5)]

        by_frame = {item["frame_index"]: item for item in build_review_queue(areas, masks=masks)}
        # it may trip area_spiked, but the fill gate keeps it out of mask_oversized
        if 4 in by_frame:
            self.assertNotIn("mask_oversized", by_frame[4]["reasons"])

    def test_review_queue_annotates_confidence_for_triage(self) -> None:
        # geometric drop flags frame 2; its (high) confidence is attached for triage
        queue = build_review_queue(
            [100, 100, 20],
            jump_ratio=0.6,
            confidences={0: 0.99, 1: 0.98, 2: 0.95},
            low_confidence_threshold=0.5,
        )
        item = next(it for it in queue if it["frame_index"] == 2)
        self.assertIn("mask_area_dropped", item["reasons"])
        self.assertNotIn("low_mask_confidence", item["reasons"])  # high conf -> not a confidence flag
        self.assertEqual(item["diagnostics"]["confidence"], 0.95)

    def test_collapse_low_confidence_empty_runs(self) -> None:
        # frames 2,3,4 are empty + low SAM2 confidence -> a likely object-absence run.
        areas = [100, 100, 0, 0, 0]
        conf = {0: 0.99, 1: 0.98, 2: 0.05, 3: 0.04, 4: 0.06}
        queue = build_review_queue(areas, confidences=conf, low_confidence_threshold=0.5)
        by_frame = {it["frame_index"]: it for it in queue}
        self.assertEqual(sorted(by_frame), [2, 3, 4])
        # collapsed to one onset interaction instead of one per frame
        self.assertEqual(by_frame[2]["estimated_interactions"], 1)
        self.assertEqual(by_frame[3]["estimated_interactions"], 0)
        self.assertEqual(by_frame[4]["estimated_interactions"], 0)
        self.assertEqual(by_frame[3]["status"], "low_confidence_empty")
        self.assertEqual(sum(int(it["estimated_interactions"]) for it in queue), 1)
        # disabling collapse restores one interaction per flagged frame
        queue_raw = build_review_queue(
            areas, confidences=conf, low_confidence_threshold=0.5, collapse_low_confidence_runs=False
        )
        self.assertEqual(sum(int(it["estimated_interactions"]) for it in queue_raw), 3)

    def test_review_score_ranks_and_suppresses(self) -> None:
        # a benign scale-trend flag ranks lower than an identity-change flag
        trend = build_review_queue([100, 80, 60], area_decline_ratio=0.35)[0]
        self.assertLess(trend["review_score"], 0.7)
        first = np.zeros((10, 10), dtype=np.uint8)
        second = np.zeros((10, 10), dtype=np.uint8)
        first[2:4, 1:3] = 1
        second[2:4, 7:9] = 1
        jump = build_review_queue([int(first.sum()), int(second.sum())], masks={0: first, 1: second})[0]
        self.assertGreater(jump["review_score"], trend["review_score"])  # center/IoU jump > scale trend
        # collapsed absence frames are ranked low; the onset keeps real priority
        areas = [100, 100, 0, 0, 0]
        conf = {0: 0.99, 1: 0.98, 2: 0.05, 3: 0.04, 4: 0.06}
        by_frame = {it["frame_index"]: it for it in build_review_queue(areas, confidences=conf)}
        self.assertEqual(by_frame[3]["review_score"], 0.15)
        self.assertGreater(by_frame[2]["review_score"], 0.5)

    def test_confident_smooth_shrink_is_damped_not_hidden(self) -> None:
        # a smooth decline while SAM2 stays highly confident = likely partial occlusion /
        # receding object: ranked low, but still present in the queue (never hidden).
        high = build_review_queue(
            [100, 80, 60], area_decline_ratio=0.35, confidences={0: 0.99, 1: 0.95, 2: 0.92}
        )
        by = {it["frame_index"]: it for it in high}
        self.assertIn(2, by)  # still queued — not suppressed
        self.assertEqual(by[2]["reasons"], ["mask_area_trending_down"])
        self.assertEqual(by[2]["review_score"], 0.3)
        self.assertEqual(by[2]["status"], "likely_occlusion")
        self.assertEqual(by[2]["estimated_interactions"], 1)  # cost still counted, not zeroed
        # the same decline at medium confidence (below the damp floor) keeps its normal rank
        med = build_review_queue(
            [100, 80, 60], area_decline_ratio=0.35, confidences={0: 0.7, 1: 0.7, 2: 0.7}
        )
        by_med = {it["frame_index"]: it for it in med}
        self.assertGreater(by_med[2]["review_score"], 0.5)
        self.assertNotEqual(by_med[2].get("status"), "likely_occlusion")

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

    def test_pipeline_threads_init_frame_index_to_segmenter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video_path = root / "sample.mp4"
            video_path.write_bytes(b"fake")

            def fake_extract(video_path_arg: Path, frame_dir: Path):
                frame_dir.mkdir(parents=True, exist_ok=True)
                return VideoMetadata(fps=5.0, width=6, height=4, frame_count=3)

            def fake_save_mask(mask: np.ndarray, output_path: Path) -> None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"x")

            def fake_render_overlay(**kwargs):
                kwargs["output_path"].write_bytes(b"mp4")
                return kwargs["output_path"]

            segmenter = FakeSegmenter()
            with patch.object(pipeline, "extract_video_frames", side_effect=fake_extract), patch.object(
                pipeline, "save_mask", side_effect=fake_save_mask
            ), patch.object(pipeline, "render_overlay_video", side_effect=fake_render_overlay):
                result = PromptableVideoSegmentationPipeline(segmenter).run(
                    VideoSegmentationRequest(
                        video_path=video_path,
                        init_box_xyxy=(0, 0, 5, 3),
                        init_frame_index=1,  # seed mid-clip
                        output_dir=root / "outputs",
                        run_id="seed-test",
                    )
                )

            self.assertEqual(segmenter.last_init_frame_index, 1)
            metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(metrics["init_frame_index"], 1)

    def test_pipeline_records_sam2_confidence(self) -> None:
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
                kwargs["output_path"].write_bytes(b"mp4")
                return kwargs["output_path"]

            with patch.object(pipeline, "extract_video_frames", side_effect=fake_extract), patch.object(
                pipeline, "save_mask", side_effect=fake_save_mask
            ), patch.object(pipeline, "render_overlay_video", side_effect=fake_render_overlay):
                result = PromptableVideoSegmentationPipeline(FakeConfidenceSegmenter()).run(
                    VideoSegmentationRequest(
                        video_path=video_path,
                        init_box_xyxy=(0, 0, 5, 3),
                        output_dir=root / "outputs",
                        run_id="conf-test",
                    )
                )

            metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(metrics["mask_confidences"], [0.95, 0.9, 0.1])
            item2 = next(it for it in metrics["review_queue"] if it["frame_index"] == 2)
            self.assertIn("low_mask_confidence", item2["reasons"])
            self.assertEqual(item2["diagnostics"]["confidence"], 0.1)

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
