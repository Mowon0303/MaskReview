from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app import (
    apply_box_click,
    apply_point_click,
    load_review_frame,
    make_review_frame_choices,
    on_frame_click,
    on_queue_click,
    render_staged_frame,
    save_selected_correction,
    select_review_frame,
)


class AppHelperTest(unittest.TestCase):
    def test_make_review_frame_choices_uses_frame_index_as_value(self) -> None:
        choices = make_review_frame_choices(
            [
                {"frame_index": 4, "reason": "mask_area_dropped"},
                {"frame_index": 9, "reason": "mask_area_spiked"},
            ]
        )

        self.assertEqual(
            choices,
            [
                ("000004 | mask_area_dropped", "4"),
                ("000009 | mask_area_spiked", "9"),
            ],
        )

    def test_load_review_frame_returns_existing_artifact_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame_path = root / "frames" / "000002.jpg"
            mask_path = root / "masks" / "000002.png"
            frame_path.parent.mkdir(parents=True)
            mask_path.parent.mkdir(parents=True)
            frame_path.write_bytes(b"jpg")
            mask_path.write_bytes(b"png")
            item = {
                "frame_index": 2,
                "reason": "mask_area_dropped",
                "frame_path": str(frame_path),
                "mask_path": str(mask_path),
            }

            frame, mask, selected = load_review_frame("2", {"review_queue": [item]})

            self.assertEqual(frame, str(frame_path))
            self.assertEqual(mask, str(mask_path))
            self.assertEqual(selected, item)

    def test_save_selected_correction_writes_to_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_state = {
                "run_dir": tmp,
                "review_queue": [
                    {
                        "frame_index": 2,
                        "reason": "mask_area_dropped",
                        "frame_path": "frames/000002.jpg",
                        "mask_path": "masks/000002.png",
                    }
                ],
            }

            corrections, status = save_selected_correction(
                run_state,
                "2",
                "tight_box",
                "",
                "1,2,30,40",
                "",
            )

            self.assertEqual(corrections[0]["frame_index"], 2)
            self.assertEqual(corrections[0]["box_xyxy"], [1, 2, 30, 40])
            self.assertIn("corrections.json", status)
            self.assertTrue((Path(tmp) / "corrections.json").exists())

    def test_apply_point_click_appends_labeled_multipoint(self) -> None:
        self.assertEqual(apply_point_click("", 120, 180, "p"), "120,180,p")
        self.assertEqual(apply_point_click("120,180,p", 200, 90, "n"), "120,180,p;200,90,n")
        self.assertEqual(apply_point_click("120,180,p;", 50, 60, "p"), "120,180,p;50,60,p")

    def test_apply_box_click_two_corners(self) -> None:
        self.assertEqual(apply_box_click("", 200, 300), "200,300")  # first corner
        self.assertEqual(apply_box_click("200,300", 100, 350), "100,300,200,350")  # normalized box
        self.assertEqual(apply_box_click("10,20,30,40", 5, 5), "5,5")  # complete -> restart

    def test_on_frame_click_labels_clicks_by_active_tool(self) -> None:
        class _Evt:  # minimal stand-in for gr.SelectData
            index = (300, 150)

        # no run_state -> no frame to draw on, image output is None; assert the staged text
        pts, box, img = on_frame_click("positive_point", "10,20,p", "", "", {}, _Evt())
        self.assertEqual((pts, box), ("10,20,p;300,150,p", ""))
        self.assertIsNone(img)
        # negative tool tags the click 'n' — this is how a single correction mixes + and -
        pts, box, _ = on_frame_click("negative_point", "10,20,p", "", "", {}, _Evt())
        self.assertEqual(pts, "10,20,p;300,150,n")
        # tight_box mode stages a corner into the box field, leaves points untouched
        pts, box, _ = on_frame_click("tight_box", "10,20,p", "", "", {}, _Evt())
        self.assertEqual((pts, box), ("10,20,p", "300,150"))

    def test_select_review_frame_clears_staged_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame_path = root / "frames" / "000002.jpg"
            mask_path = root / "masks" / "000002.png"
            frame_path.parent.mkdir(parents=True)
            mask_path.parent.mkdir(parents=True)
            frame_path.write_bytes(b"jpg")
            mask_path.write_bytes(b"png")
            item = {
                "frame_index": 2,
                "reason": "mask_area_dropped",
                "frame_path": str(frame_path),
                "mask_path": str(mask_path),
            }

            frame, mask, selected, points, box = select_review_frame("2", {"review_queue": [item]})

            self.assertEqual(frame, str(frame_path))
            self.assertEqual(mask, str(mask_path))
            self.assertEqual(selected, item)
            # switching frames must reset staged clicks so they don't leak into the next correction
            self.assertEqual(points, "")
            self.assertEqual(box, "")

    def test_render_staged_frame_returns_path_when_nothing_staged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            import cv2
            import numpy as np

            frame_path = Path(tmp) / "f.png"
            cv2.imwrite(str(frame_path), np.zeros((40, 60, 3), dtype=np.uint8))
            # no points/box -> hand back the original path unchanged (no re-render)
            self.assertEqual(render_staged_frame(str(frame_path), "", ""), str(frame_path))
            # a staged point -> an annotated RGB image, same resolution as the frame
            out = render_staged_frame(str(frame_path), "30,20,p", "")
            self.assertEqual(out.shape, (40, 60, 3))
            self.assertTrue((out != 0).any())  # something was drawn

    def test_render_staged_frame_tolerates_missing_frame(self) -> None:
        self.assertIsNone(render_staged_frame(None, "10,20,p", ""))

    def test_on_queue_click_syncs_dropdown_and_clears(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame_path = root / "frames" / "000005.jpg"
            frame_path.parent.mkdir(parents=True)
            frame_path.write_bytes(b"jpg")
            item = {"frame_index": 5, "reason": "empty_mask", "frame_path": str(frame_path)}

            selected, frame, mask, chosen, points, box = on_queue_click("5", {"review_queue": [item]})

            self.assertEqual(selected, "5")  # dropdown value synced to the clicked card
            self.assertEqual(frame, str(frame_path))
            self.assertIsNone(mask)  # item has no mask_path
            self.assertEqual(chosen, item)
            self.assertEqual(points, "")
            self.assertEqual(box, "")


if __name__ == "__main__":
    unittest.main()
