from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cvat.export import (  # noqa: E402
    IssuePayload,
    coco_rle,
    decode_coco_rle,
    decode_cvat_points,
    interaction_report,
    mask_to_cvat_points,
    masks_to_coco,
    queue_to_issues,
)


def _mask(h, w, y0, y1, x0, x1):
    m = np.zeros((h, w), dtype=np.uint8)
    m[y0:y1, x0:x1] = 1
    return m


class CocoRleTest(unittest.TestCase):
    def test_round_trips_various_masks(self) -> None:
        for m in [
            _mask(8, 6, 1, 4, 2, 5),
            np.zeros((4, 4), dtype=np.uint8),
            np.ones((5, 7), dtype=np.uint8),
        ]:
            rle = coco_rle(m)
            self.assertEqual(rle["size"], [m.shape[0], m.shape[1]])
            np.testing.assert_array_equal(decode_coco_rle(rle), m)

    def test_first_pixel_set_starts_with_zero_run(self) -> None:
        m = np.zeros((3, 3), dtype=np.uint8)
        m[0, 0] = 1  # first pixel (column-major) is foreground
        rle = coco_rle(m)
        self.assertEqual(rle["counts"][0], 0)  # COCO counts background first
        np.testing.assert_array_equal(decode_coco_rle(rle), m)


class MasksToCocoTest(unittest.TestCase):
    def test_images_annotations_and_rle(self) -> None:
        masks = {
            0: _mask(8, 6, 1, 4, 2, 5),
            1: np.zeros((8, 6), dtype=np.uint8),  # empty frame
            2: _mask(8, 6, 0, 2, 0, 2),
        }
        doc = masks_to_coco(masks, label="bottle")

        self.assertEqual(len(doc["images"]), 3)  # one image per frame, incl. empty
        self.assertEqual(len(doc["annotations"]), 2)  # empty frame contributes no annotation
        self.assertEqual(doc["categories"], [{"id": 1, "name": "bottle"}])

        ann0 = next(a for a in doc["annotations"] if a["image_id"] == 1)  # frame 0 -> image_id 1
        self.assertEqual(ann0["bbox"], [2, 1, 3, 3])  # xywh
        self.assertEqual(ann0["area"], 9)
        np.testing.assert_array_equal(decode_coco_rle(ann0["segmentation"]), masks[0])

    def test_frame_name_callable_used(self) -> None:
        doc = masks_to_coco({3: _mask(4, 4, 0, 2, 0, 2)}, frame_name=lambda i: f"f{i}.png")
        self.assertEqual(doc["images"][0]["file_name"], "f3.png")
        self.assertEqual(doc["images"][0]["frame"], 3)


class CvatPointsTest(unittest.TestCase):
    def test_round_trip_within_bbox(self) -> None:
        m = _mask(10, 12, 3, 7, 4, 9)
        points = mask_to_cvat_points(m)
        self.assertEqual(points[-4:], [4, 3, 8, 6])  # left, top, right, bottom
        np.testing.assert_array_equal(decode_cvat_points(points, 10, 12), m)

    def test_empty_mask_is_empty_points(self) -> None:
        self.assertEqual(mask_to_cvat_points(np.zeros((5, 5), dtype=np.uint8)), [])


class QueueToIssuesTest(unittest.TestCase):
    def _queue(self):
        return [
            {"frame_index": 2, "reasons": ["mask_oversized"], "review_score": 0.8,
             "recommended_correction": "tight box"},
            {"frame_index": 5, "reasons": ["mask_area_trending_down"], "review_score": 0.3,
             "status": "likely_occlusion"},
            {"frame_index": 7, "reason": "empty_mask", "review_score": 0.92},
        ]

    def test_sorted_by_score_and_min_score_filter(self) -> None:
        issues = queue_to_issues(self._queue(), min_score=0.4)
        self.assertEqual([i.frame for i in issues], [7, 2])  # 0.3 frame dropped, sorted desc
        self.assertIn("[0.92]", issues[0].message)
        self.assertIn("empty_mask", issues[0].message)
        self.assertIn("fix: tight box", issues[1].message)

    def test_keeps_low_priority_when_unfiltered_and_fills_position(self) -> None:
        issues = queue_to_issues(self._queue(), min_score=0.0, bboxes={2: (10, 20, 30, 40)})
        by_frame = {i.frame: i for i in issues}
        self.assertIn(5, by_frame)  # likely_occlusion kept when not filtered
        self.assertEqual(by_frame[5].status, "likely_occlusion")
        self.assertEqual(by_frame[2].position, [10.0, 20.0, 30.0, 40.0])
        self.assertIsNone(by_frame[7].position)  # no bbox supplied for frame 7

    def test_issue_payload_to_dict(self) -> None:
        issue = queue_to_issues([{"frame_index": 1, "reason": "empty_mask", "review_score": 0.9}])[0]
        self.assertEqual(issue.to_dict()["frame"], 1)
        self.assertEqual(issue.to_dict()["reasons"], ["empty_mask"])


class InteractionReportTest(unittest.TestCase):
    def test_counts_and_delta(self) -> None:
        queue = [
            {"frame_index": 1, "estimated_interactions": 1},
            {"frame_index": 2, "estimated_interactions": 0},  # collapsed
            {"frame_index": 3, "estimated_interactions": 1},
        ]
        report = interaction_report(queue, resolved_issue_count=3)
        self.assertEqual(report["queued_frames"], 3)
        self.assertEqual(report["estimated_interactions"], 2)
        self.assertEqual(report["resolved_issues"], 3)
        self.assertEqual(report["delta_vs_estimate"], 1)


if __name__ == "__main__":
    unittest.main()
