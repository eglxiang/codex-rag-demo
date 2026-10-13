"""Transcribe an MP4 with faster-whisper on CPU (tiny by default)."""

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import time


def extract_audio(video, destination):
    """Decode the first audio stream to 16 kHz mono PCM on the video timeline.

    copyts preserves presentation times. Resampling inserts silence for initial
    offsets and timestamp gaps, rather than moving later speech to time zero.
    """
    result = subprocess.run([
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-copyts", "-i", str(video), "-map", "0:a:0", "-vn",
        "-af", "aresample=16000:async=1:first_pts=0", "-ac", "1",
        "-c:a", "pcm_s16le", str(destination),
    ], capture_output=True, text=True)
    if result.returncode:
        raise ValueError(f"Audio extraction failed: {result.stderr.strip()}")


def load_model(model_name="tiny"):
    from faster_whisper import WhisperModel
    return WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=4)


def transcribe_video(video, output="outputs/transcript.json", model_name="tiny", language=None):
    video = Path(video)
    if not video.is_file():
        raise ValueError(f"Video file does not exist: {video}")
    if video.suffix.lower() != ".mp4":
        raise ValueError("Input must be an MP4 file")
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="video-rag-audio-") as directory:
        audio = Path(directory) / "audio.wav"
        extract_audio(video, audio)
        model = load_model(model_name)
        inference_start = time.perf_counter()
        segments, info = model.transcribe(str(audio), word_timestamps=True,
                                          vad_filter=False, beam_size=5, language=language)
        records = []
        # faster-whisper is lazy: consuming segments executes transcription.
        for segment in segments:
            records.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
                "words": [{"start": word.start, "end": word.end,
                           "word": word.word, "probability": word.probability}
                          for word in (segment.words or [])],
            })
        inference_seconds = time.perf_counter() - inference_start
    result = {
        "video_filename": video.name,
        "model": model_name,
        "device": "cpu",
        "compute_type": "int8",
        "requested_language": language,
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
        "inference_seconds": inference_seconds,
        "total_seconds": time.perf_counter() - started,
        "segments": records,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/transcript.json"))
    parser.add_argument("--model", default="tiny", help="Whisper model size (default: tiny)")
    parser.add_argument("--language", help="language code, e.g. fa (default: automatic detection)")
    args = parser.parse_args()
    try:
        result = transcribe_video(args.video, args.output, args.model, args.language)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(f"Language: {result['language']} ({result['language_probability']:.3f})")
    for segment in result["segments"]:
        print(f"[{segment['start']:.2f}–{segment['end']:.2f}s] {segment['text']}")
    print(f"Inference: {result['inference_seconds']:.2f}s; total: {result['total_seconds']:.2f}s")
    print(f"Saved transcript to {args.output}")


if __name__ == "__main__":
    main()
