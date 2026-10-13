import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from video_retrieval import build_chunks, persian_tokens, search_transcript


class VideoRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.transcript = {"video_filename": "sample.mp4", "segments": [
            {"start": 0.5, "end": 5.5, "text": "انتخابات پارلمانی مراکش"},
            {"start": 5.5, "end": 10.5, "text": "مردم دوباره اعتماد کردند"},
            {"start": 10.5, "end": 15.5, "text": "قیمت کالاها افزایش یافت"},
            {"start": 15.5, "end": 20.5, "text": "بیکاری جوانان مشکل اقتصادی است"},
        ]}

    def test_chunks_overlap_and_preserve_timestamps_and_text(self):
        chunks = build_chunks(self.transcript, 10, 5)
        self.assertEqual((chunks[0]["start"], chunks[0]["end"]), (0.5, 10.5))
        self.assertEqual((chunks[1]["start"], chunks[1]["end"]), (5.5, 15.5))
        self.assertEqual(chunks[0]["segments"][1], chunks[1]["segments"][0])
        for chunk in chunks:
            for segment in chunk["segments"]:
                original = self.transcript["segments"][segment["segment_index"]]
                self.assertEqual({k: segment[k] for k in ("start", "end", "text")}, original)

    def test_bounds_include_full_segments(self):
        chunks = build_chunks(self.transcript, 7, 3)
        self.assertEqual(chunks[0]["end"], 10.5)
        self.assertIn("مردم دوباره اعتماد کردند", chunks[0]["text"])

    def test_persian_ranking(self):
        results = search_transcript(self.transcript, "بیکاری جوانان", 2, 10, 5)
        self.assertEqual(len(results), 2)
        self.assertIn("بیکاری جوانان", results[0]["text"])
        self.assertGreaterEqual(results[0]["score"], results[1]["score"])
        self.assertEqual(results[0]["end"], 20.5)

    def test_normalization_preserves_evidence(self):
        self.assertEqual(persian_tokens("كِتاب يک می\u200cرود"), persian_tokens("کتاب یک می رود"))
        transcript = {"segments": [{"start": 1, "end": 2, "text": "كِتاب يک"}]}
        result = search_transcript(transcript, "کتاب یک")[0]
        self.assertEqual(result["text"], "كِتاب يک")
        self.assertAlmostEqual(result["score"], 1)

    def test_empty_and_unknown_queries(self):
        self.assertEqual(search_transcript(self.transcript, "unknownxyz"), [])
        self.assertEqual(search_transcript(self.transcript, ""), [])
        self.assertEqual(build_chunks({"segments": []}), [])

    def test_invalid_configuration_and_segments(self):
        for window, overlap in [(0, 0), (5, 5), (5, -1), (float("inf"), 1)]:
            with self.assertRaises(ValueError):
                build_chunks(self.transcript, window, overlap)
        for segment in [{"start": -1, "end": 1, "text": "x"},
                        {"start": 2, "end": 1, "text": "x"},
                        {"start": float("nan"), "end": 2, "text": "x"}]:
            with self.assertRaises(ValueError):
                build_chunks({"segments": [segment]})
        with self.assertRaises(ValueError):
            search_transcript(self.transcript, "مردم", top_k=0)

    def test_silence_and_duplicate_windows(self):
        transcript = {"segments": [{"start": 1, "end": 25, "text": "long segment"},
                                   {"start": 10000, "end": 10002, "text": "late segment"}]}
        chunks = build_chunks(transcript, 10, 5)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[1]["start"], 10000)

    def test_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transcript.json"
            path.write_text(json.dumps(self.transcript), encoding="utf-8")
            script = Path(__file__).resolve().parents[1] / "video_retrieval.py"
            result = subprocess.run([sys.executable, str(script), "--transcript", str(path),
                                     "--query", "اعتماد", "--top-k", "1"],
                                    capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("اعتماد کردند", result.stdout)
        self.assertIn("score=", result.stdout)
        self.assertIn("0.50–20.50s", result.stdout)


if __name__ == "__main__":
    unittest.main()
