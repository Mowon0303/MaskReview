from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app import load_review_frame, make_review_frame_choices, save_selected_correction


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


if __name__ == "__main__":
    unittest.main()
