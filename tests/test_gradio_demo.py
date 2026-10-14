import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from demo_services import DemoConfig, DemoServices, evidence_rows
from experiments.visual_backend import LocalVisualBackend, temporal_prompt, visual_prompt
from gradio_demo import build_demo, main


class DemoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.config = DemoConfig(video=root / "video.mp4", frames=root / "frames.json",
                                 transcript=root / "transcript.json", captions=root / "captions.json")
        frames = []
        for i in (1, 2):
            path = root / f"{i}.jpg"
            Image.new("RGB", (16, 16)).save(path)
            frames.append({"timestamp": i, "frame_path": str(path)})
        self.config.frames.write_text(json.dumps(frames))
        self.config.transcript.write_text(json.dumps({"video_filename": "video.mp4", "segments": [
            {"start": 0, "end": 5, "text": "مردم اعتماد کردند"}]}))
        self.config.captions.write_text(json.dumps([{**item, "caption": "people in a classroom",
                                                    "video_filename": "video.mp4"} for item in frames]))
        self.visual = Mock()
        self.services = DemoServices(self.config, self.visual)

    def test_builds_three_tabs_without_loading_models_or_connecting(self):
        with patch.object(self.services, "backend", side_effect=AssertionError("network call")):
            demo = build_demo(self.services)
        tabs = [c["props"]["label"] for c in demo.config["components"] if c["type"] == "tabitem"]
        self.assertEqual(tabs, ["Video-RAG QA", "Visual Caption Comparison", "Temporal Understanding"])
        self.visual.generate.assert_not_called()
        self.addCleanup(demo.close)

    def test_qa_modes_and_evidence_rendering(self):
        for mode in ("asr", "visual", "multimodal"):
            result = self.services.qa("مردم classroom", mode, "classroom", 1, 1, True)
            self.assertIsNone(result["answer"])
            expected = {"asr"} if mode == "asr" else {"visual"} if mode == "visual" else {"asr", "visual"}
            self.assertEqual({row[1] for row in evidence_rows(result)}, expected)

    def test_mocked_qa_answer_and_citations(self):
        backend = Mock()
        backend.complete.return_value = json.dumps({"answer": "مردم اعتماد کردند.", "abstained": False,
            "citations": [{"evidence_id": "chunk_0000", "quote": "مردم اعتماد کردند"}]})
        with patch.object(self.services, "backend", return_value=backend):
            result = self.services.qa("مردم", "asr", "", 1, 1, False)
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["citations"][0]["start"], 0)

    def test_comparison_prompts_and_independent_outputs(self):
        self.visual.generate.side_effect = ["direct description", "context analysis"]
        self.assertEqual(self.services.compare("0", "election report"),
                         ("direct description", "context analysis"))
        calls = self.visual.generate.call_args_list
        self.assertNotIn("election report", calls[0].args[1])
        self.assertIn("election report", calls[1].args[1])
        self.assertIn("never as visual proof", calls[1].args[1])

    def test_temporal_uses_actual_timestamps_and_hypothesis_warning(self):
        self.visual.generate.return_value = "uncertain correspondence"
        self.assertEqual(self.services.temporal("0", "1"), "uncertain correspondence")
        prompt = self.visual.generate.call_args.args[1]
        self.assertIn("1s", prompt)
        self.assertIn("2s", prompt)
        self.assertIn("not verified", prompt)
        self.assertEqual(len(self.visual.generate.call_args.args[0]), 2)
        with self.assertRaises(ValueError):
            self.services.temporal("1", "0")

    def test_invalid_and_missing_frame(self):
        for selection in (None, "-1", "999", "invalid"):
            with self.assertRaises(ValueError):
                self.services.frame(selection)
        path, _ = self.services.frame("0")
        path.unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            self.services.frame("0")

    def test_backend_discovery_uses_existing_server(self):
        response = io.StringIO(json.dumps({"data": [{"id": "served-model"}]}))
        with patch("demo_services.urlopen") as open_url:
            open_url.return_value.__enter__.return_value = response
            backend = self.services.backend()
        self.assertEqual(backend.model, "served-model")
        self.assertEqual(open_url.call_args.args[0].full_url, "http://localhost:8000/v1/models")

    def test_lazy_local_only_configuration(self):
        backend = LocalVisualBackend("/nonexistent/checkpoint", "cuda:0")
        self.assertIsNone(backend.model)
        with self.assertRaisesRegex(ValueError, "existing local"):
            backend._load()
        self.assertIn("UNCERTAINTY", visual_prompt())
        self.assertIn("not verified tracking", temporal_prompt(30, 31))

    @patch("gradio_demo.DemoServices")
    @patch("gradio_demo.build_demo")
    def test_launch_is_loopback_without_public_sharing(self, build, services):
        services.return_value.config = self.config
        services.return_value.frames = []
        with patch("sys.argv", ["gradio_demo.py"]):
            main()
        options = build.return_value.queue.return_value.launch.call_args.kwargs
        self.assertEqual(options["server_name"], "127.0.0.1")
        self.assertEqual(options["server_port"], 7860)
        self.assertFalse(options["share"])

    def test_mocked_local_model_load_is_offline(self):
        import sys
        fake_torch = Mock()
        fake_torch.device.return_value.type = "cpu"
        fake_transformers = Mock()
        processor = fake_transformers.AutoProcessor.from_pretrained
        model = fake_transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained
        with patch.dict(sys.modules, {"torch": fake_torch, "transformers": fake_transformers}):
            backend = LocalVisualBackend(self.temp.name, "cpu")
            backend._load()
            backend._load()
        processor.assert_called_once_with(self.temp.name, local_files_only=True)
        self.assertTrue(model.call_args.kwargs["local_files_only"])
        self.assertEqual(model.call_count, 1)


if __name__ == "__main__":
    unittest.main()
