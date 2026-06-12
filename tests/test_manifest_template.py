from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import manifest_template
from manifest_template import (
    VideoTemplateMetadata,
    find_video_files,
    generate_eval_manifest_template,
    sanitize_case_id,
)


class ManifestTemplateTest(unittest.TestCase):
    def test_find_video_files_filters_supported_extensions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "b.mov").write_bytes(b"mov")
            (root / "a.mp4").write_bytes(b"mp4")
            (root / "notes.txt").write_text("skip", encoding="utf-8")

            self.assertEqual([path.name for path in find_video_files(root)], ["a.mp4", "b.mov"])

    def test_generate_eval_manifest_template_writes_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video_dir = root / "eval_videos"
            video_dir.mkdir()
            video_path = video_dir / "cup sample.mp4"
            video_path.write_bytes(b"fake")
            output_path = root / "eval_manifest.template.json"
            first_frame_dir = root / "eval_first_frames"

            with patch.object(
                manifest_template,
                "inspect_video",
                return_value=VideoTemplateMetadata(
                    width=640,
                    height=360,
                    frame_count=120,
                    fps=30.0,
                    duration_seconds=4.0,
                ),
            ) as inspect_video:
                generate_eval_manifest_template(
                    video_dir=video_dir,
                    output_path=output_path,
                    first_frame_dir=first_frame_dir,
                )

            inspect_video.assert_called_once_with(video_path, first_frame_path=first_frame_dir / "cup_sample.jpg")
            manifest = json.loads(output_path.read_text(encoding="utf-8"))
            case = manifest["cases"][0]
            self.assertEqual(case["id"], "cup_sample")
            self.assertEqual(case["video_path"], "eval_videos/cup sample.mp4")
            self.assertIsNone(case["init_box_xyxy"])
            self.assertEqual(case["first_frame_path"], "eval_first_frames/cup_sample.jpg")
            self.assertEqual(case["video_metadata"]["width"], 640)

            with self.assertRaises(FileExistsError):
                generate_eval_manifest_template(
                    video_dir=video_dir,
                    output_path=output_path,
                    first_frame_dir=first_frame_dir,
                )

    def test_sanitize_case_id(self) -> None:
        self.assertEqual(sanitize_case_id(" cup sample 01 "), "cup_sample_01")
        self.assertEqual(sanitize_case_id("///"), "case")


if __name__ == "__main__":
    unittest.main()
