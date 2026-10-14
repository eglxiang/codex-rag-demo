"""Retrieve timestamped visual captions with the default TF-IDF tokenizer."""

import argparse
import json
import math
from pathlib import Path

from retrieval import Retriever


def caption_evidence(captions):
    """Convert caption records to point-in-time visual evidence.

    A sampled image supports only its timestamp, so start and end are equal.
    IDs use collection indices and remain stable across queries.
    """
    if not isinstance(captions, list):
        raise ValueError("Captions must be a JSON list")
    evidence = []
    for index, caption in enumerate(captions):
        if not isinstance(caption, dict) or any(
            not isinstance(caption.get(key), str)
            for key in ("caption", "frame_path", "video_filename")
        ):
            raise ValueError("Each caption requires string caption, frame_path, and video_filename fields")
        timestamp = caption.get("timestamp")
        if (isinstance(timestamp, bool) or not isinstance(timestamp, (int, float))
                or not math.isfinite(timestamp) or timestamp < 0):
            raise ValueError("Caption timestamps must be finite nonnegative numbers")
        evidence.append({
            "id": f"visual_{index:04d}",
            "type": "visual",
            "text": caption["caption"],
            "start": timestamp,
            "end": timestamp,
            "frame_path": caption["frame_path"],
            "video_filename": caption["video_filename"],
        })
    return evidence


def search_captions(captions, query, top_k=3):
    evidence = caption_evidence(captions)
    return [{**item, "score": score}
            for item, score in Retriever(evidence).search(query, top_k)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captions", type=Path, default=Path("outputs/captions.json"))
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    try:
        captions = json.loads(args.captions.read_text(encoding="utf-8"))
        results = search_captions(captions, args.query, args.top_k)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
