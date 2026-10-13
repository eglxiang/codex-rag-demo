"""Retrieve timestamped ASR evidence with TF-IDF, without generating answers."""

import argparse
import json
import math
from pathlib import Path
import re
import unicodedata

from retrieval import Retriever, tokenize


def persian_tokens(text):
    """Normalize Persian spelling variants for indexing, preserving source text."""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک",
                                       "\u200c": " ", "ـ": ""}))
    text = re.sub(r"[\u064b-\u065f\u0670]", "", text)
    return tokenize(text)


def build_chunks(transcript, window_seconds=20, overlap_seconds=10):
    """Slide windows over time and include complete intersecting ASR segments.

    Report actual source segment bounds, which can extend beyond a nominal
    window. Identical segment selections are deduplicated. Silent windows are
    skipped, and overlap repeats evidence across neighboring chunks.
    """
    if (not math.isfinite(window_seconds) or window_seconds <= 0
            or not math.isfinite(overlap_seconds)
            or not 0 <= overlap_seconds < window_seconds):
        raise ValueError("Require window > 0 and 0 <= overlap < window (finite seconds)")
    if not isinstance(transcript, dict) or not isinstance(transcript.get("segments"), list):
        raise ValueError("Transcript must contain a segments list")
    segments = transcript["segments"]
    previous_start = -1
    for segment in segments:
        if not isinstance(segment, dict) or not isinstance(segment.get("text"), str):
            raise ValueError("Each segment requires text and valid start/end timestamps")
        start, end = segment.get("start"), segment.get("end")
        if (any(isinstance(t, bool) or not isinstance(t, (int, float))
                or not math.isfinite(t) for t in (start, end))
                or not 0 <= start < end or start < previous_start):
            raise ValueError("Segments require finite 0 <= start < end, ordered by start")
        previous_start = start
    if not segments:
        return []
    step = window_seconds - overlap_seconds
    origin = segments[0]["start"]
    # Only visit windows intersecting speech, including across long silent gaps.
    candidates = set()
    for segment in segments:
        first = max(0, math.floor((segment["start"] - origin - window_seconds) / step) + 1)
        last = math.ceil((segment["end"] - origin) / step) - 1
        candidates.update(range(first, last + 1))
    chunks, seen = [], set()
    for index in sorted(candidates):
        window_start = origin + index * step
        selected = tuple(i for i, segment in enumerate(segments)
                         if segment["start"] < window_start + window_seconds
                         and segment["end"] > window_start and segment["text"].strip())
        if not selected or selected in seen:
            continue
        seen.add(selected)
        source = [{"segment_index": i, "start": segments[i]["start"],
                   "end": segments[i]["end"], "text": segments[i]["text"]}
                  for i in selected]
        chunks.append({
            "id": f"chunk_{len(chunks):04d}",
            "video_filename": transcript.get("video_filename"),
            "start": min(s["start"] for s in source),
            "end": max(s["end"] for s in source),
            "text": " ".join(s["text"] for s in source),
            "segments": source,
        })
    return chunks


def search_transcript(transcript, query, top_k=3, window_seconds=20, overlap_seconds=10):
    chunks = build_chunks(transcript, window_seconds, overlap_seconds)
    return [{**chunk, "score": score}
            for chunk, score in Retriever(chunks, tokenizer=persian_tokens).search(query, top_k)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--transcript", type=Path, default=Path("outputs/transcript_small.json"))
    parser.add_argument("--window", type=float, default=20, help="window duration in seconds")
    parser.add_argument("--overlap", type=float, default=10, help="overlap in seconds")
    args = parser.parse_args()
    try:
        transcript = json.loads(args.transcript.read_text(encoding="utf-8"))
        results = search_transcript(transcript, args.query, args.top_k, args.window, args.overlap)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if not results:
        print("No matching chunks.")
    for rank, result in enumerate(results, 1):
        print(f"{rank}. [{result['start']:.2f}–{result['end']:.2f}s] "
              f"score={result['score']:.4f} ({result['video_filename']})")
        print(result["text"])
        print()


if __name__ == "__main__":
    main()
