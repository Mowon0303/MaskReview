from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from corrections import build_correction, load_corrections, save_correction


class CorrectionsTest(unittest.TestCase):
    def test_build_positive_point_correction(self) -> None:
        correction = build_correction(
            frame_index=7,
            correction_type="positive_point",
            point_xy="12, 34",
            review_item={
                "reason": "mask_area_dropped",
                "frame_path": "frames/000007.jpg",
                "mask_path": "masks/000007.png",
            },
        )

        self.assertEqual(correction["frame_index"], 7)
        self.assertEqual(correction["type"], "positive_point")
        self.assertEqual(correction["points"], [{"x": 12, "y": 34, "label": "positive"}])
        self.assertEqual(correction["estimated_interactions"], 1)
        self.assertEqual(correction["review_reason"], "mask_area_dropped")
        self.assertTrue(str(correction["created_at"]).endswith("Z"))

    def test_build_tight_box_correction(self) -> None:
        correction = build_correction(
            frame_index=3,
            correction_type="tight_box",
            box_xyxy="1,2,30,40",
            note="edge drift",
        )

        self.assertEqual(correction["box_xyxy"], [1, 2, 30, 40])
        self.assertEqual(correction["note"], "edge drift")

    def test_build_multi_point_correction_counts_each_click(self) -> None:
        correction = build_correction(
            frame_index=5,
            correction_type="positive_point",
            point_xy="10,20;30,40;50,60",
        )
        self.assertEqual(
            correction["points"],
            [
                {"x": 10, "y": 20, "label": "positive"},
                {"x": 30, "y": 40, "label": "positive"},
                {"x": 50, "y": 60, "label": "positive"},
            ],
        )
        # one human interaction per click — a hard frame may need several points
        self.assertEqual(correction["estimated_interactions"], 3)

    def test_build_mixed_point_labels_per_click(self) -> None:
        # a single correction can carry both positive and negative points (the real fix:
        # one + on the object, several - on background/occluder to carve the mask down)
        correction = build_correction(
            frame_index=5,
            correction_type="positive_point",
            point_xy="10,20,p;30,40,n;50,60,n",
        )
        self.assertEqual(
            [p["label"] for p in correction["points"]],
            ["positive", "negative", "negative"],
        )
        self.assertEqual(correction["estimated_interactions"], 3)

    def test_unlabeled_points_fall_back_to_correction_type(self) -> None:
        correction = build_correction(
            frame_index=5,
            correction_type="negative_point",
            point_xy="10,20;30,40",
        )
        self.assertEqual([p["label"] for p in correction["points"]], ["negative", "negative"])

    def test_rejects_invalid_correction_inputs(self) -> None:
        with self.assertRaises(ValueError):
            build_correction(frame_index=0, correction_type="positive_point", point_xy="1")
        with self.assertRaises(ValueError):
            build_correction(frame_index=0, correction_type="tight_box", box_xyxy="10,2,3,4")
        with self.assertRaises(ValueError):
            build_correction(frame_index=0, correction_type="scribble")

    def test_save_correction_upserts_one_entry_per_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            first = build_correction(2, "positive_point", point_xy="1,2")
            second = build_correction(2, "negative_point", point_xy="3,4")

            path = save_correction(run_dir, first)
            save_correction(run_dir, second)

            self.assertTrue(path.exists())
            corrections = load_corrections(run_dir)
            self.assertEqual(len(corrections), 1)
            self.assertEqual(corrections[0]["type"], "negative_point")
            self.assertEqual(corrections[0]["points"][0]["label"], "negative")

            raw = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(raw, corrections)


if __name__ == "__main__":
    unittest.main()
