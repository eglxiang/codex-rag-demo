import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from retrieval import DEFAULT_DOCUMENTS, Retriever, cosine_similarity, load_documents, tokenize


ROOT = Path(__file__).resolve().parents[1]


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.retriever = Retriever(load_documents(DEFAULT_DOCUMENTS))

    def test_tokenization(self):
        self.assertEqual(tokenize("Hello, HELLO! soil."), ["hello", "hello", "soil"])

    def test_cosine(self):
        self.assertAlmostEqual(cosine_similarity({"a": 1}, {"a": 1, "b": 1}),
                               1 / math.sqrt(2))
        self.assertEqual(cosine_similarity({}, {"a": 1}), 0)
        self.assertEqual(cosine_similarity({"a": 1}, {"b": 1}), 0)

    def test_ranks_relevant_document_first(self):
        matches = self.retriever.search("tomatoes soil sunlight")
        self.assertEqual(matches[0][0]["id"], "gardening")
        self.assertGreater(matches[0][1], 0)

    def test_tfidf_weights_and_exact_match(self):
        docs = [{"id": "a", "title": "A", "text": "common rare rare"},
                {"id": "b", "title": "B", "text": "common other"}]
        retriever = Retriever(docs)
        self.assertAlmostEqual(retriever.idf["rare"], math.log(3 / 2) + 1)
        self.assertEqual(retriever.idf["common"], 1)
        self.assertAlmostEqual(retriever.vectors[0]["rare"], 2 * retriever.idf["rare"])
        self.assertAlmostEqual(retriever.search("common rare rare")[0][1], 1)

    def test_top_k_and_descending_scores(self):
        matches = self.retriever.search("documents", top_k=2)
        self.assertEqual(len(matches), 2)
        self.assertGreaterEqual(matches[0][1], matches[1][1])

    def test_empty_and_unknown_queries(self):
        for query in ("", "!!!", "unknownxyz"):
            self.assertEqual(self.retriever.search(query), [])
        self.assertEqual(Retriever([]).search("hello"), [])

    def test_empty_document_and_stable_ties(self):
        docs = [{"id": str(i), "title": "Title", "text": text}
                for i, text in enumerate(["", "same", "same"])]
        self.assertEqual([doc["id"] for doc, _ in Retriever(docs).search("same")],
                         ["1", "2"])

    def test_invalid_top_k(self):
        for top_k in (0, -1):
            with self.assertRaises(ValueError):
                self.retriever.search("soil", top_k)

    def test_invalid_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "docs.json"
            path.write_text(json.dumps([{"text": 42}]), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_documents(path)

    def test_cli_from_another_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, str(ROOT / "retrieval.py"), "tomatoes soil", "--top-k", "1"],
                cwd=directory, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Growing tomatoes [gardening]", result.stdout)
        self.assertNotIn("2.", result.stdout)

    def test_cli_no_matches_and_invalid_argument(self):
        result = subprocess.run([sys.executable, str(ROOT / "retrieval.py"), "unknownxyz"],
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout.strip(), "No matching documents.")
        result = subprocess.run([sys.executable, str(ROOT / "retrieval.py"), "soil", "--top-k", "0"],
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("top_k must be at least 1", result.stderr)


if __name__ == "__main__":
    unittest.main()
