from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quality_metrics import (
    boundary_f_measure,
    load_indexed_masks,
    region_iou,
    score_mask_dirs,
    sequence_quality,
)


def _square(size: int, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 1
    return mask


class RegionIouTest(unittest.TestCase):
    def test_identical_masks_score_one(self) -> None:
        mask = _square(8, 1, 5, 1, 5)
        self.assertEqual(region_iou(mask, mask), 1.0)

    def test_two_empty_masks_are_perfect_agreement(self) -> None:
        empty = np.zeros((8, 8), dtype=np.uint8)
        self.assertEqual(region_iou(empty, empty), 1.0)

    def test_one_empty_one_full_is_zero(self) -> None:
        empty = np.zeros((8, 8), dtype=np.uint8)
        full = np.ones((8, 8), dtype=np.uint8)
        self.assertEqual(region_iou(full, empty), 0.0)
        self.assertEqual(region_iou(empty, full), 0.0)

    def test_half_overlap_iou(self) -> None:
        pred = _square(4, 0, 2, 0, 4)  # rows 0-1
        gt = _square(4, 1, 3, 0, 4)  # rows 1-2
        # intersection = row 1 (4 px), union = rows 0-2 (12 px)
        self.assertAlmostEqual(region_iou(pred, gt), 4 / 12)


class BoundaryFTest(unittest.TestCase):
    def test_identical_masks_score_one(self) -> None:
        mask = _square(20, 4, 16, 4, 16)
        self.assertEqual(boundary_f_measure(mask, mask), 1.0)

    def test_two_empty_masks_score_one(self) -> None:
        empty = np.zeros((20, 20), dtype=np.uint8)
        self.assertEqual(boundary_f_measure(empty, empty), 1.0)

    def test_one_empty_is_zero(self) -> None:
        empty = np.zeros((20, 20), dtype=np.uint8)
        full = _square(20, 4, 16, 4, 16)
        self.assertEqual(boundary_f_measure(full, empty), 0.0)

    def test_disjoint_masks_score_zero(self) -> None:
        a = _square(40, 2, 8, 2, 8)
        b = _square(40, 30, 36, 30, 36)
        self.assertEqual(boundary_f_measure(a, b), 0.0)

    def test_partial_overlap_between_zero_and_one(self) -> None:
        pred = _square(20, 4, 14, 4, 14)
        gt = _square(20, 4, 14, 8, 18)  # shifted right
        score = boundary_f_measure(pred, gt)
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)


class SequenceQualityTest(unittest.TestCase):
    def test_perfect_sequence(self) -> None:
        ones = np.ones((4, 4), dtype=np.uint8)
        zeros = np.zeros((4, 4), dtype=np.uint8)
        masks = {0: ones, 1: ones, 2: zeros}
        result = sequence_quality(masks, masks)
        self.assertEqual(result["frames_scored"], 3)
        self.assertEqual(result["mean_iou"], 1.0)
        self.assertEqual(result["mean_boundary_f"], 1.0)
        self.assertEqual(result["jf"], 1.0)

    def test_one_wrong_frame_lowers_mean(self) -> None:
        ones = np.ones((4, 4), dtype=np.uint8)
        zeros = np.zeros((4, 4), dtype=np.uint8)
        pred = {0: ones, 1: ones, 2: zeros}
        gt = {0: ones, 1: zeros, 2: zeros}  # frame 1 disagrees (one empty)
        result = sequence_quality(pred, gt)
        self.assertEqual(result["frames_scored"], 3)
        self.assertAlmostEqual(result["mean_iou"], round((1.0 + 0.0 + 1.0) / 3, 4))
        self.assertAlmostEqual(result["jf"], round((1.0 + 0.0 + 1.0) / 3, 4))

    def test_no_common_frames_returns_empty(self) -> None:
        result = sequence_quality({0: np.ones((4, 4), np.uint8)}, {1: np.ones((4, 4), np.uint8)})
        self.assertEqual(result["frames_scored"], 0)
        self.assertIsNone(result["jf"])


class MaskIoTest(unittest.TestCase):
    def _write_mask(self, path: Path, mask: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), (mask * 255).astype(np.uint8))

    def test_load_and_score_perfect_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred_dir = root / "masks"
            gt_dir = root / "gt"
            mask = _square(8, 1, 6, 1, 6)
            for idx in range(3):
                self._write_mask(pred_dir / f"{idx:06d}.png", mask)
                self._write_mask(gt_dir / f"{idx:06d}.png", mask)
            result = score_mask_dirs(pred_dir, gt_dir, frame_count=3, height=8, width=8)
            self.assertEqual(result["frames_scored"], 3)
            self.assertEqual(result["jf"], 1.0)

    def test_missing_files_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mask = _square(8, 1, 6, 1, 6)
            self._write_mask(root / "000000.png", mask)
            # frames 1,2 absent
            masks = load_indexed_masks(root, frame_count=3, height=8, width=8)
            self.assertEqual(sorted(masks), [0])

    def test_mismatched_size_is_resized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_mask(root / "000000.png", _square(16, 2, 12, 2, 12))
            masks = load_indexed_masks(root, frame_count=1, height=8, width=8)
            self.assertEqual(masks[0].shape, (8, 8))


if __name__ == "__main__":
    unittest.main()
