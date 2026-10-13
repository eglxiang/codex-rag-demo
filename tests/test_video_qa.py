import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError

from video_qa import ask_video, build_messages, OpenAICompatibleBackend, validate_answer


class VideoQATests(unittest.TestCase):
    def setUp(self):
        self.transcript = {"video_filename": "test.mp4", "segments": [
            {"start": 32.44, "end": 38.44, "text": "مردم به آنها اعتماد کردند"}]}
        self.question = "چرا مردم اعتماد کردند؟"

    def test_mocked_answer_has_verified_citations_and_timestamps(self):
        backend = Mock()
        backend.complete.return_value = json.dumps({"answer": "مردم به آنها اعتماد کردند.",
            "abstained": False, "citations": [{"evidence_id": "chunk_0000",
                                                "quote": "مردم به آنها اعتماد کردند"}]})
        result = ask_video(self.transcript, self.question, backend)
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["citations"][0]["start"], 32.44)
        self.assertEqual(result["citations"][0]["end"], 38.44)
        self.assertEqual(result["evidence"][0]["text"], self.transcript["segments"][0]["text"])
        self.assertIn("Persian", backend.complete.call_args.args[0][0]["content"])

    def test_mocked_insufficient_evidence_abstention(self):
        backend = Mock()
        backend.complete.return_value = json.dumps({"answer": "دلیل اعتماد مشخص نیست.",
                                                    "abstained": True, "citations": []})
        result = ask_video(self.transcript, self.question, backend)
        self.assertTrue(result["abstained"])
        self.assertEqual(result["status"], "abstained")

    def test_no_evidence_abstains_without_api_call(self):
        backend = Mock()
        result = ask_video(self.transcript, "unknownxyz", backend)
        self.assertEqual(result["status"], "no_evidence")
        backend.complete.assert_not_called()

    def test_dry_run_constructs_prompt_without_generation(self):
        backend = Mock()
        result = ask_video(self.transcript, self.question, backend, dry_run=True)
        self.assertIsNone(result["answer"])
        self.assertEqual(result["status"], "prompt_only")
        content = json.loads(result["messages"][1]["content"])
        self.assertEqual(content["question"], self.question)
        self.assertEqual(content["evidence"][0]["start"], 32.44)
        self.assertIn("explicitly abstain", result["messages"][0]["content"])
        backend.complete.assert_not_called()

    def test_rejects_malformed_and_fabricated_citations(self):
        evidence = ask_video(self.transcript, self.question, dry_run=True)["evidence"]
        for content in ['not JSON', '{}', json.dumps({"answer": "x", "abstained": False,
                         "citations": []}), json.dumps({"answer": "x", "abstained": False,
                         "citations": [{"evidence_id": "fake", "quote": "مردم"}]}),
                         json.dumps({"answer": "x", "abstained": False,
                         "citations": [{"evidence_id": "chunk_0000", "quote": "invented"}]})]:
            with self.assertRaises(ValueError):
                validate_answer(content, evidence)

    @patch("video_qa.urlopen")
    def test_openai_request_and_response(self, open_url):
        open_url.return_value.__enter__.return_value = io.StringIO(json.dumps(
            {"choices": [{"message": {"content": "response"}}]}))
        backend = OpenAICompatibleBackend("https://example.invalid/v1", "configured-model", "test-key")
        messages = build_messages(self.question, [])
        self.assertEqual(backend.complete(messages), "response")
        request = open_url.call_args.args[0]
        self.assertEqual(request.full_url, "https://example.invalid/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        body = json.loads(request.data)
        self.assertEqual(body["model"], "configured-model")
        self.assertEqual(body["messages"], messages)

    @patch("video_qa.urlopen")
    def test_api_failure_does_not_expose_credentials(self, open_url):
        open_url.side_effect = HTTPError("url", 401, "secret-provider-error", {}, None)
        backend = OpenAICompatibleBackend("https://example.invalid/v1", "model", "secret-key")
        with self.assertRaisesRegex(ValueError, "HTTP 401") as error:
            backend.complete([])
        self.assertNotIn("secret", str(error.exception))

    def test_configuration_is_required(self):
        for base, model, key in [(None, "model", "key"), ("https://example.invalid", None, "key"),
                                 ("https://example.invalid", "model", None)]:
            with self.assertRaises(ValueError):
                OpenAICompatibleBackend(base, model, key)
        OpenAICompatibleBackend("http://localhost:8080/v1", "model", no_auth=True)
        with self.assertRaisesRegex(ValueError, "backend"):
            ask_video(self.transcript, self.question)

    def test_cli_dry_run(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transcript.json"
            path.write_text(json.dumps(self.transcript), encoding="utf-8")
            script = Path(__file__).resolve().parents[1] / "video_qa.py"
            result = subprocess.run([sys.executable, str(script), "--transcript", str(path),
                                     "--question", self.question, "--dry-run"],
                                    capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "prompt_only")


if __name__ == "__main__":
    unittest.main()
