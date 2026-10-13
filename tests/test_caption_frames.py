import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from caption_frames import caption_frames, DEFAULT_MODEL


class CaptionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.metadata = root / "frames.json"
        self.output = root / "captions.json"
        self.records = []
        for i in range(4):
            path = root / f"frame_{i}.jpg"
            Image.new("RGB", (24, 16), (i * 50, 0, 0)).save(path)
            self.records.append({"timestamp": float(i), "frame_path": str(path),
                                 "video_filename": "sample.mp4"})
        self.metadata.write_text(json.dumps(self.records))

    @patch("caption_frames.load_captioner")
    def test_limit_and_metadata(self, loader):
        loader.return_value.side_effect = ["a red image", "a dark image", "a bright image"]
        results = caption_frames(self.metadata, self.output, limit=3)
        loader.assert_called_once_with(DEFAULT_MODEL)
        self.assertEqual(loader.return_value.call_count, 3)
        self.assertEqual(len(results), 3)
        for source, result in zip(self.records, results):
            for key, value in source.items():
                self.assertEqual(result[key], value)
            self.assertEqual(result["model"], DEFAULT_MODEL)
        self.assertEqual(results[0]["caption"], "a red image")
        self.assertEqual(json.loads(self.output.read_text()), results)

    @patch("caption_frames.load_captioner")
    def test_all_frames(self, loader):
        loader.return_value.return_value = "a picture"
        self.assertEqual(len(caption_frames(self.metadata, self.output)), 4)

    @patch("caption_frames.load_captioner")
    def test_empty_input_does_not_load_model(self, loader):
        self.metadata.write_text("[]")
        self.assertEqual(caption_frames(self.metadata, self.output), [])
        loader.assert_not_called()

    @patch("caption_frames.load_captioner")
    def test_invalid_input(self, loader):
        for limit in (0, -1):
            with self.assertRaisesRegex(ValueError, "limit"):
                caption_frames(self.metadata, self.output, limit=limit)
        for value in ({}, [{"timestamp": 0}],
                      [{"timestamp": 0, "frame_path": "missing.jpg"}]):
            self.metadata.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                caption_frames(self.metadata, self.output)
        loader.assert_not_called()

    @patch("caption_frames.load_captioner")
    def test_model_failure_preserves_output(self, loader):
        self.output.write_text("previous output")
        loader.return_value.side_effect = RuntimeError("inference failed")
        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            caption_frames(self.metadata, self.output)
        self.assertEqual(self.output.read_text(), "previous output")

    @patch("caption_frames.load_captioner")
    def test_empty_caption_is_rejected(self, loader):
        loader.return_value.return_value = " "
        with self.assertRaisesRegex(ValueError, "empty caption"):
            caption_frames(self.metadata, self.output)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
