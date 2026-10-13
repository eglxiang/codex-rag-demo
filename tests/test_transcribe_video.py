import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from transcribe_video import extract_audio, transcribe_video


class TranscriptionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.video = root / "video.mp4"
        self.video.touch()
        self.output = root / "outputs" / "transcript.json"

    @patch("transcribe_video.load_model")
    @patch("transcribe_video.extract_audio")
    def test_transcription_preserves_segment_and_word_timestamps(self, extract, loader):
        word = SimpleNamespace(start=7.25, end=7.75, word=" Hello", probability=.9)
        segment = SimpleNamespace(start=7.25, end=8.0, text=" Hello.", words=[word])
        info = SimpleNamespace(language="en", language_probability=.95, duration=12.0)
        loader.return_value.transcribe.return_value = (iter([segment]), info)
        result = transcribe_video(self.video, self.output)
        self.assertEqual(result["language"], "en")
        self.assertEqual(result["video_filename"], "video.mp4")
        self.assertEqual(result["segments"][0]["start"], 7.25)
        self.assertEqual(result["segments"][0]["end"], 8.0)
        self.assertEqual(result["segments"][0]["words"][0]["start"], 7.25)
        self.assertEqual(result["segments"][0]["text"], "Hello.")
        self.assertEqual(json.loads(self.output.read_text()), result)
        self.assertGreaterEqual(result["total_seconds"], result["inference_seconds"])
        kwargs = loader.return_value.transcribe.call_args.kwargs
        self.assertTrue(kwargs["word_timestamps"])
        self.assertFalse(kwargs["vad_filter"])
        self.assertFalse(Path(extract.call_args.args[1]).exists())

    @patch("transcribe_video.load_model")
    @patch("transcribe_video.extract_audio")
    def test_no_speech(self, extract, loader):
        loader.return_value.transcribe.return_value = (
            iter([]), SimpleNamespace(language="en", language_probability=.1, duration=2.0))
        self.assertEqual(transcribe_video(self.video, self.output)["segments"], [])

    @patch("transcribe_video.load_model")
    @patch("transcribe_video.extract_audio")
    def test_model_and_explicit_language(self, extract, loader):
        loader.return_value.transcribe.return_value = (
            iter([]), SimpleNamespace(language="fa", language_probability=1.0, duration=2.0))
        result = transcribe_video(self.video, self.output, model_name="small", language="fa")
        loader.assert_called_once_with("small")
        self.assertEqual(loader.return_value.transcribe.call_args.kwargs["language"], "fa")
        self.assertEqual(result["model"], "small")
        self.assertEqual(result["requested_language"], "fa")
        self.assertEqual(result["compute_type"], "int8")

    @patch("transcribe_video.extract_audio", side_effect=ValueError("no audio stream"))
    def test_failure_preserves_existing_output(self, extract):
        self.output.parent.mkdir()
        self.output.write_text("previous")
        with self.assertRaisesRegex(ValueError, "no audio"):
            transcribe_video(self.video, self.output)
        self.assertEqual(self.output.read_text(), "previous")

    def test_invalid_video(self):
        with self.assertRaisesRegex(ValueError, "does not exist"):
            transcribe_video(self.video.with_name("missing.mp4"), self.output)
        other = self.video.with_suffix(".txt")
        other.touch()
        with self.assertRaisesRegex(ValueError, "MP4"):
            transcribe_video(other, self.output)

    @patch("transcribe_video.subprocess.run")
    def test_audio_extraction_retains_timeline(self, run):
        run.return_value = SimpleNamespace(returncode=0)
        extract_audio(self.video, "audio.wav")
        command = run.call_args.args[0]
        self.assertIn("-copyts", command)
        self.assertIn("aresample=16000:async=1:first_pts=0", command)
        self.assertNotIn("-ss", command)

    @patch("transcribe_video.subprocess.run")
    def test_audio_extraction_failure(self, run):
        run.return_value = SimpleNamespace(returncode=1, stderr="Stream map matches no streams")
        with self.assertRaisesRegex(ValueError, "Stream map matches no streams"):
            extract_audio(self.video, "audio.wav")


if __name__ == "__main__":
    unittest.main()
