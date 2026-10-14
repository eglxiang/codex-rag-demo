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


class FusionTests(unittest.TestCase):
    def setUp(self):
        self.transcript = {"video_filename": "test.mp4", "segments": [
            {"start": 0, "end": 5, "text": "مردم اعتماد کردند"},
            {"start": 20, "end": 25, "text": "مردم دوباره اعتماد کردند"}]}
        self.captions = [{"timestamp": i, "caption": "people in a classroom",
                          "frame_path": f"frames/{i}.jpg", "video_filename": "test.mp4"}
                         for i in (1, 2)]

    def test_asr_default_preserves_ids_and_ignores_visual_input(self):
        default = ask_video(self.transcript, "مردم", dry_run=True, captions=self.captions)
        explicit = ask_video(self.transcript, "مردم", dry_run=True, mode="asr")
        self.assertEqual(default, explicit)
        self.assertTrue(all(item["type"] == "asr" for item in default["evidence"]))
        self.assertTrue(default["evidence"][0]["id"].startswith("chunk_"))

    def test_visual_only_and_point_citation(self):
        backend = Mock()
        backend.complete.return_value = json.dumps({"answer": "افرادی در کلاس دیده می‌شوند.",
            "abstained": False, "citations": [{"evidence_id": "visual_0000",
                                               "quote": "people in a classroom"}]})
        result = ask_video(None, "classroom", backend, mode="visual", captions=self.captions)
        self.assertEqual(result["status"], "answered")
        self.assertTrue(all(item["type"] == "visual" for item in result["evidence"]))
        citation = result["citations"][0]
        self.assertEqual((citation["start"], citation["end"]), (1, 1))
        self.assertEqual(citation["type"], "visual")
        self.assertEqual(citation["frame_path"], "frames/1.jpg")

    def test_multimodal_separate_budgets_and_prompt_provenance(self):
        result = ask_video(self.transcript, "مردم", dry_run=True, mode="multimodal",
                           captions=self.captions, asr_top_k=1, visual_top_k=2,
                           visual_query="classroom")
        self.assertEqual([item["type"] for item in result["evidence"]], ["asr", "visual", "visual"])
        self.assertEqual(len({item["id"] for item in result["evidence"]}), 3)
        prompt = json.loads(result["messages"][1]["content"])["evidence"]
        self.assertEqual(prompt[0]["type"], "asr")
        self.assertEqual(prompt[1]["frame_path"], "frames/1.jpg")
        self.assertIn("not speech", result["messages"][0]["content"])

    @patch("video_qa.search_transcript")
    @patch("video_qa.search_captions")
    def test_scores_are_not_sorted_across_modalities(self, visual, asr):
        asr.return_value = [{"id": "chunk_0000", "start": 0, "end": 5, "text": "مردم", "score": .01}]
        visual.return_value = [{"id": "visual_0000", "type": "visual", "start": 1,
                                "end": 1, "text": "people", "frame_path": "x.jpg", "score": .99}]
        result = ask_video(self.transcript, "مردم", dry_run=True, mode="multimodal",
                           captions=self.captions, asr_top_k=2, visual_top_k=1)
        self.assertEqual([item["id"] for item in result["evidence"]], ["chunk_0000", "visual_0000"])
        self.assertEqual(asr.call_args.args[2], 2)
        self.assertEqual(visual.call_args.args[2], 1)

    def test_multimodal_citations_validate_both_sources(self):
        evidence = ask_video(self.transcript, "مردم", dry_run=True, mode="multimodal",
                             captions=self.captions, visual_query="classroom")["evidence"]
        content = json.dumps({"answer": "test", "abstained": False, "citations": [
            {"evidence_id": evidence[0]["id"], "quote": "مردم"},
            {"evidence_id": "visual_0000", "quote": "classroom"}]})
        citations = validate_answer(content, evidence)["citations"]
        self.assertEqual([c["type"] for c in citations], ["asr", "visual"])
        with self.assertRaises(ValueError):
            validate_answer(content.replace("classroom", "invented"), evidence)
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_answer(content, evidence + [evidence[0]])

    def test_abstention_in_each_mode(self):
        for mode in ("asr", "visual", "multimodal"):
            backend = Mock()
            no_matches = ask_video(self.transcript, "unknownxyz", backend,
                                   mode=mode, captions=self.captions)
            self.assertEqual(no_matches["status"], "no_evidence")
            backend.complete.assert_not_called()
            backend.complete.return_value = json.dumps({"answer": "شواهد کافی نیست.",
                                                        "abstained": True, "citations": []})
            result = ask_video(self.transcript, "مردم classroom", backend,
                               mode=mode, captions=self.captions)
            self.assertEqual(result["status"], "abstained")

    def test_visual_does_not_translate_persian_queries(self):
        result = ask_video(None, "کلاس", mode="visual", captions=self.captions)
        self.assertEqual(result["status"], "no_evidence")

    def test_required_captions_and_mode_validation(self):
        for mode in ("visual", "multimodal", "invalid"):
            with self.assertRaises(ValueError):
                ask_video(self.transcript, "مردم", mode=mode, dry_run=True)

    def test_visual_cli_does_not_require_transcript(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "captions.json"
            path.write_text(json.dumps(self.captions))
            script = Path(__file__).resolve().parents[1] / "video_qa.py"
            result = subprocess.run([sys.executable, str(script), "--mode", "visual",
                "--captions", str(path), "--transcript", "missing.json",
                "--question", "classroom", "--dry-run"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["evidence"][0]["type"], "visual")


if __name__ == "__main__":
    unittest.main()
