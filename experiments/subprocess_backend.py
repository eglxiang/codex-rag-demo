"""Run visual inference in a separately configured Python environment."""

import json
import math
import os
from pathlib import Path
import subprocess
from threading import Lock


class SubprocessVisualBackend:
    def __init__(self, model_path=None, device=None, python=None, timeout=300):
        self.model_path = model_path or os.environ.get("QWEN_VL_MODEL_PATH")
        self.device = device or os.environ.get("QWEN_VL_DEVICE", "cuda:0")
        self.python = python or os.environ.get("QWEN_VL_PYTHON")
        self.timeout = timeout
        self.lock = Lock()

    def generate(self, frame_paths, prompt):
        if not self.python:
            raise ValueError("Set QWEN_VL_PYTHON to the GPU environment's Python executable.")
        executable = Path(self.python).expanduser().absolute()
        # Do not resolve a venv Python symlink: its path selects the environment.
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise ValueError(f"QWEN_VL_PYTHON is not an executable file: {executable}")
        if not self.model_path or not Path(self.model_path).expanduser().is_dir():
            raise ValueError("Set QWEN_VL_MODEL_PATH to an existing local checkpoint directory.")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("Visual subprocess timeout must be a finite positive number")
        payload = {"frame_paths": [str(Path(path).resolve()) for path in frame_paths],
                   "prompt": prompt, "model_path": str(Path(self.model_path).expanduser().resolve()),
                   "device": self.device}
        environment = os.environ.copy()
        # CUDA_VISIBLE_DEVICES is inherited unchanged; device indices are relative
        # to that mask. Offline flags apply only to the child, not the vLLM service.
        environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        worker = Path(__file__).with_name("visual_worker.py").resolve()
        try:
            with self.lock:
                result = subprocess.run([str(executable), str(worker)],
                    input=json.dumps(payload, ensure_ascii=False), capture_output=True,
                    text=True, encoding="utf-8", env=environment, timeout=self.timeout)
        except subprocess.TimeoutExpired:
            raise ValueError(f"Visual inference timed out after {self.timeout:g}s; the worker was stopped.") from None
        except OSError as error:
            raise ValueError(f"Cannot start GPU Python: {error}") from None
        try:
            response = json.loads(result.stdout)
        except (ValueError, TypeError):
            response = None
        if result.returncode:
            detail = (response.get("error") if isinstance(response, dict) else None)
            detail = detail or result.stderr.strip()[-4000:] or "No error details returned"
            raise ValueError(f"Visual worker failed (exit {result.returncode}): {detail}")
        if (not isinstance(response, dict) or not isinstance(response.get("text"), str)
                or not response["text"].strip()):
            detail = result.stderr.strip()[-1000:]
            raise ValueError("Visual worker returned invalid JSON or empty text" + (f": {detail}" if detail else ""))
        return response["text"]
