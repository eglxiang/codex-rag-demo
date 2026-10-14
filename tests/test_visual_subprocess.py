import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from demo_services import DemoConfig, DemoServices
from experiments.subprocess_backend import SubprocessVisualBackend
from experiments.visual_worker import main, run_request


class SubprocessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.backend = SubprocessVisualBackend(self.temp.name, "cuda:0", sys.executable)

    @patch("experiments.subprocess_backend.subprocess.run")
    def test_json_transport_and_gpu_environment(self, run):
        run.return_value = Mock(returncode=0, stdout=json.dumps({"text": "مشاهده واقعی"}), stderr="")
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2"}):
            result = self.backend.generate([Path(self.temp.name) / "frame.jpg"], "describe تصویر")
        self.assertEqual(result, "مشاهده واقعی")
        command = run.call_args.args[0]
        self.assertEqual(command[0], str(Path(sys.executable).absolute()))
        self.assertTrue(command[1].endswith("visual_worker.py"))
        self.assertNotIn("shell", run.call_args.kwargs)
        request = json.loads(run.call_args.kwargs["input"])
        self.assertEqual(request["prompt"], "describe تصویر")
        self.assertEqual(request["device"], "cuda:0")
        self.assertTrue(Path(request["frame_paths"][0]).is_absolute())
        env = run.call_args.kwargs["env"]
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "2")
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        self.assertEqual(env["TRANSFORMERS_OFFLINE"], "1")
        self.assertEqual(run.call_args.kwargs["timeout"], 300)

    @patch("experiments.subprocess_backend.subprocess.run")
    def test_two_frames_passed_in_order(self, run):
        run.return_value = Mock(returncode=0, stdout='{"text":"hypothesis"}', stderr="")
        self.backend.generate(["first.jpg", "second.jpg"], "compare")
        request = json.loads(run.call_args.kwargs["input"])
        self.assertEqual([Path(path).name for path in request["frame_paths"]], ["first.jpg", "second.jpg"])

    @patch("experiments.subprocess_backend.subprocess.run")
    def test_errors_are_actionable(self, run):
        run.return_value = Mock(returncode=1, stdout='{"error":"RuntimeError: CUDA out of memory"}', stderr="")
        with self.assertRaisesRegex(ValueError, "CUDA out of memory"):
            self.backend.generate(["frame.jpg"], "describe")
        run.return_value = Mock(returncode=9, stdout="", stderr="GPU environment lacks Transformers")
        with self.assertRaisesRegex(ValueError, "lacks Transformers"):
            self.backend.generate(["frame.jpg"], "describe")
        run.return_value = Mock(returncode=0, stdout="not JSON", stderr="")
        with self.assertRaisesRegex(ValueError, "invalid JSON"):
            self.backend.generate(["frame.jpg"], "describe")
        run.side_effect = subprocess.TimeoutExpired("worker", 300)
        with self.assertRaisesRegex(ValueError, "timed out"):
            self.backend.generate(["frame.jpg"], "describe")

    def test_missing_configuration_never_starts_process(self):
        with patch.dict(os.environ, {}, clear=True), patch("experiments.subprocess_backend.subprocess.run") as run:
            backend = SubprocessVisualBackend()
            with self.assertRaisesRegex(ValueError, "QWEN_VL_PYTHON"):
                backend.generate(["frame.jpg"], "describe")
            run.assert_not_called()

    def test_demo_uses_subprocess_by_default(self):
        config = DemoConfig(frames=Path(self.temp.name) / "missing.json")
        services = DemoServices(config)
        self.assertIsInstance(services.visual, SubprocessVisualBackend)

    @patch("experiments.visual_worker.LocalVisualBackend")
    def test_worker_single_context_and_two_frame_requests(self, loader):
        loader.return_value.generate.return_value = "analysis"
        for paths, prompt in [(["a.jpg"], "independent"), (["a.jpg"], "context"),
                              (["a.jpg", "b.jpg"], "temporal")]:
            response = run_request({"frame_paths": paths, "prompt": prompt,
                                    "model_path": self.temp.name, "device": "cuda:0"})
            self.assertEqual(response, {"text": "analysis"})
            loader.return_value.generate.assert_called_with(paths, prompt)

    @patch("experiments.visual_worker.run_request")
    def test_worker_stdout_is_json_and_logs_go_to_stderr(self, run):
        def generate(request):
            print("model progress")
            return {"text": "analysis"}
        run.side_effect = generate
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch("sys.stdin", io.StringIO("{}")), patch("sys.stdout", stdout), patch("sys.stderr", stderr):
            self.assertEqual(main(), 0)
        self.assertEqual(json.loads(stdout.getvalue()), {"text": "analysis"})
        self.assertIn("model progress", stderr.getvalue())

    def test_worker_bad_input_returns_json_error(self):
        stdout = io.StringIO()
        with patch("sys.stdin", io.StringIO("[]")), patch("sys.stdout", stdout):
            self.assertEqual(main(), 1)
        self.assertIn("JSON object", json.loads(stdout.getvalue())["error"])
