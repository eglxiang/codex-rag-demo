"""Caption extracted video frames with BLIP on the CPU."""

import argparse
import json
from pathlib import Path
import time

from PIL import Image


DEFAULT_MODEL = "Salesforce/blip-image-captioning-base"


def load_captioner(model_name):
    """Load the pretrained model once, keeping weights and inference on CPU."""
    import torch
    from transformers import BlipForConditionalGeneration, BlipProcessor

    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name).to("cpu")
    model.eval()

    def caption(image):
        inputs = processor(images=image, return_tensors="pt")
        with torch.inference_mode():
            tokens = model.generate(**inputs, max_new_tokens=30, num_beams=3,
                                    do_sample=False)
        return processor.decode(tokens[0], skip_special_tokens=True).strip()

    return caption


def caption_frames(metadata_path="outputs/frames.json", output_path="outputs/captions.json",
                   limit=None, model_name=DEFAULT_MODEL):
    """Caption the first N entries in metadata order, preserving source fields.

    Relative frame paths are resolved from the current working directory, as
    written by video_frames.py. A failed run does not replace previous output.
    """
    if limit is not None and limit < 1:
        raise ValueError("limit must be a positive integer")
    records = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Frame metadata must be a JSON list")
    selected = records if limit is None else records[:limit]
    for record in selected:
        if (not isinstance(record, dict)
                or not isinstance(record.get("frame_path"), str)
                or not isinstance(record.get("timestamp"), (int, float))):
            raise ValueError("Each frame requires a frame_path and numeric timestamp")
        if not Path(record["frame_path"]).is_file():
            raise ValueError(f"Frame not found: {record['frame_path']}")
    results = []
    if selected:
        captioner = load_captioner(model_name)
        for record in selected:
            with Image.open(record["frame_path"]) as source:
                image = source.convert("RGB")
                try:
                    description = captioner(image)
                finally:
                    image.close()
            if not isinstance(description, str) or not description.strip():
                raise ValueError(f"Model returned an empty caption for {record['frame_path']}")
            results.append({**record, "caption": description.strip(), "model": model_name})
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, default=Path("outputs/frames.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/captions.json"))
    parser.add_argument("--limit", type=int, help="process only the first N frames")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    started = time.perf_counter()
    try:
        results = caption_frames(args.frames, args.output, args.limit, args.model)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    for record in results:
        print(f"{record['timestamp']:g}s: {record['caption']}")
    print(f"Model: {args.model}")
    print(f"Saved {len(results)} captions to {args.output}; elapsed: {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
