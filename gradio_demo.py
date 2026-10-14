"""Local research UI; run python gradio_demo.py and connect via SSH forwarding."""

import argparse
import os
from pathlib import Path

from demo_services import DemoConfig, DemoServices, evidence_rows


def build_demo(services):
    import gradio as gr

    def qa(*args):
        try:
            result = services.qa(*args)
            answer = result["answer"] or "Retrieval and prompt only; no answer generated."
            return answer, result["status"], evidence_rows(result), result["citations"]
        except Exception as error:
            raise gr.Error(str(error)) from None

    def preview(selection):
        try:
            path, timestamp = services.frame(selection)
            return str(path), f"Timestamp: {timestamp:g}s"
        except Exception as error:
            raise gr.Error(str(error)) from None

    def compare(selection, context):
        try:
            return services.compare(selection, context)
        except Exception as error:
            raise gr.Error(str(error)) from None

    def temporal(first, second):
        try:
            return services.temporal(first, second)
        except Exception as error:
            raise gr.Error(str(error)) from None

    choices = services.frame_choices()
    with gr.Blocks(title="Video-RAG Explorer", analytics_enabled=False) as demo:
        gr.Markdown("# Video-RAG Explorer\nResearch demonstration: answers and visual analyses are model outputs, not verified facts.")
        with gr.Tab("Video-RAG QA"):
            gr.Video(value=str(services.config.video) if services.config.video.is_file() else None,
                     label="Local research video", interactive=False)
            question = gr.Textbox(label="Question / پرسش", lines=2)
            mode = gr.Radio(["asr", "visual", "multimodal"], value="asr", label="Evidence mode")
            visual_query = gr.Textbox(label="Optional English visual retrieval query")
            gr.Markdown("Persian ASR and English captions use lexical retrieval without translation. Scores are ranked separately by modality.")
            with gr.Row():
                asr_budget = gr.Slider(1, 10, value=3, step=1, label="ASR evidence budget")
                visual_budget = gr.Slider(1, 10, value=3, step=1, label="Visual evidence budget")
            dry_run = gr.Checkbox(label="Retrieve evidence without generating an answer", value=False)
            submit = gr.Button("Ask about the video", variant="primary")
            answer = gr.Textbox(label="Answer", lines=5, interactive=False)
            status = gr.Textbox(label="Answer status", interactive=False)
            evidence = gr.Dataframe(headers=["ID", "Type", "Start (s)", "End (s)", "Score", "Evidence", "Frame path"],
                                    datatype=["str", "str", "number", "number", "number", "str", "str"],
                                    interactive=False, label="Retrieved supporting evidence")
            citations = gr.JSON(label="Validated citations")
            submit.click(qa, [question, mode, visual_query, asr_budget, visual_budget, dry_run],
                         [answer, status, evidence, citations])
        with gr.Tab("Visual Caption Comparison"):
            selection = gr.Dropdown(choices, label="Extracted frame")
            image = gr.Image(label="Selected frame", interactive=False)
            timestamp = gr.Textbox(label="Frame time", interactive=False)
            selection.change(preview, selection, [image, timestamp])
            context = gr.Textbox(label="Optional background context (not visual proof)", lines=3)
            gr.Markdown("Both outputs separate **direct observations**, **contextual inferences**, and **uncertainty**. Context never verifies a visual claim.")
            compare_button = gr.Button("Compare visual descriptions")
            with gr.Row():
                independent = gr.Textbox(label="Independent description", lines=15, interactive=False)
                contextual = gr.Textbox(label="Context-conditioned analysis", lines=15, interactive=False)
            compare_button.click(compare, [selection, context], [independent, contextual])
        with gr.Tab("Temporal Understanding"):
            gr.Markdown("Object correspondence is a **model-generated hypothesis, not verified tracking**. Two frames cannot establish events between them.")
            with gr.Row():
                with gr.Column():
                    first = gr.Dropdown(choices, label="Earlier frame")
                    image1 = gr.Image(label="Frame 1", interactive=False)
                    time1 = gr.Textbox(label="Timestamp 1", interactive=False)
                with gr.Column():
                    second = gr.Dropdown(choices, label="Later frame")
                    image2 = gr.Image(label="Frame 2", interactive=False)
                    time2 = gr.Textbox(label="Timestamp 2", interactive=False)
            first.change(preview, first, [image1, time1])
            second.change(preview, second, [image2, time2])
            analyze = gr.Button("Analyze temporal evidence")
            analysis = gr.Textbox(label="Continuity, correspondence hypotheses, state changes, uncertainty", lines=18, interactive=False)
            analyze.click(temporal, [first, second], analysis)
    return demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = DemoConfig()
    for name in ("video", "frames", "transcript", "captions"):
        parser.add_argument("--" + name, type=Path, default=getattr(defaults, name))
    parser.add_argument("--base-url", default=os.environ.get("VIDEO_RAG_LLM_BASE_URL", defaults.base_url))
    parser.add_argument("--model", default=os.environ.get("VIDEO_RAG_LLM_MODEL"))
    parser.add_argument("--device", default=os.environ.get("QWEN_VL_DEVICE", "cuda:0"))
    args = parser.parse_args()
    services = DemoServices(DemoConfig(**vars(args), model_path=os.environ.get("QWEN_VL_MODEL_PATH")))
    allowed = [str(services.config.video.resolve())] if services.config.video.is_file() else []
    for i in range(len(services.frames)):
        try:
            allowed.append(str(services.frame(str(i))[0]))
        except ValueError:
            pass
    build_demo(services).queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1", server_port=7860, share=False,
        allowed_paths=allowed, show_error=False)


if __name__ == "__main__":
    main()
