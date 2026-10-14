"""UI-independent services for Video-RAG Explorer."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

from experiments.visual_backend import LocalVisualBackend, temporal_prompt, visual_prompt
from video_qa import OpenAICompatibleBackend, ask_video


ROOT = Path(__file__).resolve().parent


@dataclass
class DemoConfig:
    video: Path = ROOT / "6nqGlKyiWskr-qPC_0000.mp4"
    frames: Path = ROOT / "outputs/frames.json"
    transcript: Path = ROOT / "outputs/transcript_small.json"
    captions: Path = ROOT / "outputs/captions.json"
    base_url: str = "http://localhost:8000/v1"
    model: str | None = None
    device: str = "cuda:0"
    model_path: str | None = None


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class DemoServices:
    def __init__(self, config, visual_backend=None):
        self.config = config
        self.visual = visual_backend or LocalVisualBackend(config.model_path, config.device)
        self.frames = read_json(config.frames) if config.frames.is_file() else []
        if not isinstance(self.frames, list):
            raise ValueError("Frame metadata must be a list")

    def frame(self, selection):
        try:
            index = int(selection)
            if index < 0:
                raise ValueError
            record = self.frames[index]
            path = Path(record["frame_path"])
            if not path.is_absolute():
                path = ROOT / path
            path = path.resolve()
            if not path.is_file():
                raise ValueError("Selected frame file is missing; check frames.json paths.")
            return path, float(record["timestamp"])
        except (TypeError, IndexError, KeyError):
            raise ValueError("Select an available frame") from None

    def frame_choices(self):
        return [(f"{item['timestamp']:g}s — {Path(item['frame_path']).name}", str(i))
                for i, item in enumerate(self.frames)]

    def backend(self):
        model = self.config.model
        key = os.environ.get("VIDEO_RAG_LLM_API_KEY")
        if not model:
            request = Request(self.config.base_url.rstrip("/") + "/models")
            if key:
                request.add_header("Authorization", "Bearer " + key)
            try:
                with urlopen(request, timeout=5) as response:
                    models = json.load(response)["data"]
                if len(models) != 1:
                    raise ValueError
                model = models[0]["id"]
            except (OSError, ValueError, KeyError, TypeError):
                raise ValueError("Cannot discover one served vLLM model. Set VIDEO_RAG_LLM_MODEL and verify the existing server.") from None
        return OpenAICompatibleBackend(self.config.base_url, model, key, no_auth=not key)

    def qa(self, question, mode, visual_query, asr_budget, visual_budget, dry_run):
        transcript = read_json(self.config.transcript) if mode != "visual" else None
        captions = read_json(self.config.captions) if mode != "asr" else None
        return ask_video(transcript, question, None if dry_run else self.backend(),
                         dry_run=dry_run, mode=mode, captions=captions,
                         asr_top_k=int(asr_budget), visual_top_k=int(visual_budget),
                         visual_query=visual_query.strip() or None)

    def compare(self, selection, context):
        path, _ = self.frame(selection)
        independent = self.visual.generate([path], visual_prompt())
        conditioned = self.visual.generate([path], visual_prompt(context))
        return independent, conditioned

    def temporal(self, first, second):
        path1, time1 = self.frame(first)
        path2, time2 = self.frame(second)
        if time1 >= time2:
            raise ValueError("Choose a second frame later than the first frame.")
        return self.visual.generate([path1, path2], temporal_prompt(time1, time2))


def evidence_rows(result):
    return [[e["id"], e["type"], e["start"], e["end"], e["score"], e["text"],
             e.get("frame_path", "")] for e in result["evidence"]]
