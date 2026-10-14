import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from retrieval import Retriever
from visual_retrieval import caption_evidence, search_captions


class VisualRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.captions = [
            {"caption": "a classroom with students", "timestamp": 1.25,
             "frame_path": "outputs/frames/first.jpg", "video_filename": "sample.mp4"},
            {"caption": "a car on a road", "timestamp": 5.5,
             "frame_path": "outputs/frames/second.jpg", "video_filename": "sample.mp4"},
            {"caption": "students standing in a classroom", "timestamp": 8,
             "frame_path": "outputs/frames/third.jpg", "video_filename": "sample.mp4"},
        ]

    def test_evidence_schema_and_timestamp_preservation(self):
        before = copy.deepcopy(self.captions)
        evidence = caption_evidence(self.captions)
        self.assertEqual(evidence[0], {
            "id": "visual_0000", "type": "visual", "text": self.captions[0]["caption"],
            "start": 1.25, "end": 1.25, "frame_path": self.captions[0]["frame_path"],
            "video_filename": "sample.mp4"})
        self.assertEqual(len({item["id"] for item in evidence}), 3)
        self.assertEqual(self.captions, before)

    def test_ranking_and_default_tokenizer_scores(self):
        results = search_captions(self.captions, "CAR road!", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "visual_0001")
        expected = Retriever(caption_evidence(self.captions)).search("CAR road!", 1)
        self.assertAlmostEqual(results[0]["score"], expected[0][1])
        self.assertEqual(results[0]["start"], 5.5)

    def test_top_k_and_descending_scores(self):
        results = search_captions(self.captions, "classroom students", top_k=2)
        self.assertEqual(len(results), 2)
        self.assertGreaterEqual(results[0]["score"], results[1]["score"])
        self.assertEqual({item["id"] for item in results}, {"visual_0000", "visual_0002"})

    def test_ties_preserve_collection_order(self):
        self.captions[2]["caption"] = self.captions[0]["caption"]
        results = search_captions(self.captions, "classroom")
        self.assertEqual([item["id"] for item in results], ["visual_0000", "visual_0002"])

    def test_empty_and_unknown_matches(self):
        self.assertEqual(search_captions([], "classroom"), [])
        for query in ("", "!!!", "unknownxyz"):
            self.assertEqual(search_captions(self.captions, query), [])

    def test_invalid_input_and_limits(self):
        for value in ({}, [None], [{"caption": "x"}]):
            with self.assertRaises(ValueError):
                caption_evidence(value)
        for timestamp in (None, True, -1, float("nan"), float("inf"), "1"):
            bad = {**self.captions[0], "timestamp": timestamp}
            with self.assertRaises(ValueError):
                caption_evidence([bad])
        for top_k in (0, -1):
            with self.assertRaises(ValueError):
                search_captions(self.captions, "car", top_k)

    def test_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "captions.json"
            path.write_text(json.dumps(self.captions), encoding="utf-8")
            script = Path(__file__).resolve().parents[1] / "visual_retrieval.py"
            command = [sys.executable, str(script), "--captions", str(path),
                       "--query", "car", "--top-k", "1"]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)[0]["id"], "visual_0001")
            result = subprocess.run(command[:-1] + ["0"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("top_k", result.stderr)


if __name__ == "__main__":
    unittest.main()
