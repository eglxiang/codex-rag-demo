import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


class DeploymentScriptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project = self.root / "project with spaces"
        self.project.mkdir()
        (self.project / "gradio_demo.py").touch()
        self.environment = {**os.environ, "BRTX_PROJECT_DIR": str(self.project),
                            "BRTX_VENV_DIR": str(self.project / ".venv")}

    def executable(self, path, contents):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/bash\n" + contents)
        path.chmod(0o755)

    def run_script(self, name, *args):
        return subprocess.run(["bash", str(SCRIPTS / name), *args],
                              env=self.environment, text=True, capture_output=True)

    def test_demo_uses_existing_venv_and_preserves_gpu_configuration(self):
        self.environment.update(QWEN_VL_PYTHON="/gpu/env/bin/python",
                                QWEN_VL_MODEL_PATH="/models/local checkpoint",
                                CUDA_VISIBLE_DEVICES="2", QWEN_VL_DEVICE="cuda:0")
        python = self.project / ".venv/bin/python"
        self.executable(python, 'printf "%s\\n" "$PWD" "$@" "$QWEN_VL_PYTHON" "$QWEN_VL_MODEL_PATH" "$CUDA_VISIBLE_DEVICES" "$QWEN_VL_DEVICE"\n')
        result = self.run_script("start_demo.sh", "--model", "served model")
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        self.assertEqual(lines[:4], [str(self.project), str(self.project / "gradio_demo.py"),
                                    "--model", "served model"])
        self.assertEqual(lines[4:], ["/gpu/env/bin/python", "/models/local checkpoint", "2", "cuda:0"])

    def test_missing_environment_fails_without_installing(self):
        result = self.run_script("start_demo.sh")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No dependencies will be installed", result.stderr)

    def test_demo_no_arguments_uses_brtx_defaults(self):
        self.executable(self.project / ".venv/bin/python", 'printf "%s\\n" "$@"\n')
        result = self.run_script("start_demo.sh")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), [str(self.project / "gradio_demo.py"),
            "--frames", "outputs/frames/frames.json", "--model", "Qwen/Qwen2.5-7B-Instruct"])

    def test_demo_explicit_arguments_replace_all_defaults(self):
        self.executable(self.project / ".venv/bin/python", 'printf "%s\\n" "$@"\n')
        result = self.run_script("start_demo.sh", "--frames", "custom frames.json", "--model", "custom-model")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), [str(self.project / "gradio_demo.py"),
            "--frames", "custom frames.json", "--model", "custom-model"])

    def test_optional_tunnel_has_fixed_local_target_and_no_update(self):
        binary = self.root / "cloudflared test"
        self.executable(binary, 'printf "%s\\n" "$@"\n')
        self.environment["BRTX_CLOUDFLARED"] = str(binary)
        result = self.run_script("start_quick_tunnel.sh")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["--no-autoupdate", "tunnel", "--url", "http://127.0.0.1:7860"])
        self.assertIn("NO authentication", result.stderr)

    def test_missing_tunnel_binary_does_not_download(self):
        self.environment["BRTX_CLOUDFLARED"] = str(self.root / "missing")
        result = self.run_script("start_quick_tunnel.sh")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Nothing will be installed", result.stderr)

    def test_status_with_mocked_tmux_and_http(self):
        binaries = self.root / "bin"
        self.executable(binaries / "tmux", '[[ "$3" == "=videorag-demo" ]]\n')
        self.executable(binaries / "curl", 'exit "${TEST_HTTP_STATUS:-0}"\n')
        self.environment["PATH"] = str(binaries) + os.pathsep + os.environ["PATH"]
        self.environment["BRTX_DEMO_SESSION"] = "videorag-demo"
        self.environment["BRTX_TUNNEL_SESSION"] = "videorag-tunnel"
        result = self.run_script("demo_status.sh")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("HTTP readiness: ready", result.stdout)
        self.assertIn("absent (optional)", result.stdout)
        self.environment["TEST_HTTP_STATUS"] = "7"
        result = self.run_script("demo_status.sh")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("HTTP readiness: unavailable", result.stdout)


if __name__ == "__main__":
    unittest.main()
