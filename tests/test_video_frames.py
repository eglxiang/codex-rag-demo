import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from video_frames import extract_frames


class VideoFrameTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.video = self.root / "sample.mp4"
        self.video.touch()
        self.output = self.root / "outputs"

    def mock_capture(self, fps=2, frame_count=22):
        patcher = patch("video_frames.cv2.VideoCapture")
        capture = patcher.start().return_value
        self.addCleanup(patcher.stop)
        capture.isOpened.return_value = True
        capture.get.return_value = fps
        capture.read.side_effect = [(True, np.full((16, 24, 3), i, dtype=np.uint8))
                                    for i in range(frame_count)] + [(False, None)]
        return capture

    def test_sampling_metadata_and_saved_images(self):
        capture = self.mock_capture()
        metadata = extract_frames(self.video, self.output, interval=5)
        self.assertEqual([item["timestamp"] for item in metadata], [0, 5, 10])
        self.assertEqual([item["frame_index"] for item in metadata], [0, 10, 20])
        self.assertEqual(json.loads((self.output / "frames.json").read_text()), metadata)
        for item in metadata:
            self.assertEqual(item["video_filename"], "sample.mp4")
            image = cv2.imread(item["frame_path"])
            self.assertEqual(image.shape[:2], (item["height"], item["width"]))
            self.assertAlmostEqual(float(image.mean()), item["frame_index"], delta=1)
        capture.release.assert_called_once()

    def test_short_video(self):
        self.mock_capture(frame_count=2)
        self.assertEqual(len(extract_frames(self.video, self.output)), 1)

    def test_fractional_fps_samples_first_frame_after_target(self):
        self.mock_capture(fps=2.5, frame_count=27)
        metadata = extract_frames(self.video, self.output, interval=5)
        self.assertEqual([item["frame_index"] for item in metadata], [0, 13, 25])
        self.assertEqual([item["timestamp"] for item in metadata], [0, 5.2, 10])

    def test_invalid_fps(self):
        for fps in (0, -1, float("nan"), float("inf")):
            with self.subTest(fps=fps):
                capture = self.mock_capture(fps=fps)
                with self.assertRaisesRegex(ValueError, "frame rate"):
                    extract_frames(self.video, self.output)
                capture.release.assert_called_once()

    def test_unopenable_and_empty_video(self):
        capture = self.mock_capture(frame_count=0)
        with self.assertRaisesRegex(ValueError, "no decodable"):
            extract_frames(self.video, self.output)
        capture.isOpened.return_value = False
        with self.assertRaisesRegex(ValueError, "Cannot open"):
            extract_frames(self.video, self.output)
        self.assertFalse(self.output.exists())

    def test_missing_wrong_extension_and_existing_output(self):
        with self.assertRaisesRegex(ValueError, "does not exist"):
            extract_frames(self.root / "missing.mp4", self.output)
        other = self.root / "sample.txt"
        other.touch()
        with self.assertRaisesRegex(ValueError, "MP4"):
            extract_frames(other, self.output)
        self.output.mkdir()
        sentinel = self.output / "keep.txt"
        sentinel.write_text("keep")
        with self.assertRaisesRegex(ValueError, "already exists"):
            extract_frames(self.video, self.output)
        self.assertEqual(sentinel.read_text(), "keep")

    def test_write_failure_releases_capture(self):
        capture = self.mock_capture()
        with patch("video_frames.cv2.imwrite", return_value=False):
            with self.assertRaisesRegex(OSError, "Cannot write"):
                extract_frames(self.video, self.output)
        capture.release.assert_called_once()
        self.assertFalse((self.output / "frames.json").exists())

    def test_cli_with_synthetic_mp4(self):
        writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*"mp4v"),
                                 2, (24, 16))
        self.assertTrue(writer.isOpened(), "MP4 encoding support is required")
        try:
            for i in range(22):
                writer.write(np.full((16, 24, 3), i * 10, dtype=np.uint8))
        finally:
            writer.release()
        script = Path(__file__).resolve().parents[1] / "video_frames.py"
        result = subprocess.run([sys.executable, str(script), str(self.video)],
                                cwd=self.root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Saved 11 frames", result.stdout)
        metadata = json.loads((self.output / "frames.json").read_text())
        self.assertEqual([item["timestamp"] for item in metadata], list(range(11)))
        for item in metadata:
            image = cv2.imread(str(self.root / item["frame_path"]))
            self.assertIsNotNone(image)
            self.assertAlmostEqual(float(image.mean()), item["frame_index"] * 10, delta=8)

        result = subprocess.run(
            [sys.executable, str(script), str(self.video), "--interval", "5",
             "--output-dir", "five_seconds"],
            cwd=self.root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        metadata = json.loads((self.root / "five_seconds" / "frames.json").read_text())
        self.assertEqual([item["timestamp"] for item in metadata], [0, 5, 10])

    def test_invalid_intervals(self):
        for interval in (0, -1, float("nan"), float("inf")):
            with self.subTest(interval=interval):
                with self.assertRaisesRegex(ValueError, "interval"):
                    extract_frames(self.video, self.output, interval=interval)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
