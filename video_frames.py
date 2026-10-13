"""Extract representative video frames at a configurable interval."""

import argparse
import json
import math
from pathlib import Path

import cv2


INTERVAL_SECONDS = 1


def extract_frames(video_path, output_dir="outputs", interval=INTERVAL_SECONDS):
    """Decode an MP4 and sample at zero and every interval seconds.

    Uses the first frame at or after each sampling point. Timestamps are
    frame-index/FPS seconds, suitable for constant-frame-rate MP4 inputs.
    Existing output directories are refused to prevent accidental overwrites.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("Sampling interval must be a finite positive number")
    if not video_path.is_file():
        raise ValueError(f"Video file does not exist: {video_path}")
    if video_path.suffix.lower() != ".mp4":
        raise ValueError("Input must be an MP4 file")
    if output_dir.exists():
        raise ValueError(f"Output directory already exists: {output_dir}; choose a new --output-dir")

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video has an invalid frame rate")
        ok, frame = capture.read()
        if not ok or frame is None:
            raise ValueError("Video contains no decodable frames")

        frames_dir = output_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=False)
        metadata = []
        index = 0
        next_sample = 0
        while ok and frame is not None:
            timestamp = index / fps
            if timestamp + 1e-9 >= next_sample * interval:
                frame_path = frames_dir / f"frame_{index:08d}.jpg"
                if not cv2.imwrite(str(frame_path), frame):
                    raise OSError(f"Cannot write frame: {frame_path}")
                height, width = frame.shape[:2]
                metadata.append({
                    "video_filename": video_path.name,
                    "timestamp": timestamp,
                    "frame_path": frame_path.as_posix(),
                    "frame_index": index,
                    "width": width,
                    "height": height,
                })
                # Low-FPS videos cannot supply distinct frames for every interval.
                next_sample = math.floor((timestamp + 1e-9) / interval) + 1
            index += 1
            ok, frame = capture.read()
        (output_dir / "frames.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        return metadata
    finally:
        capture.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="local MP4 file")
    parser.add_argument("--interval", type=float, default=INTERVAL_SECONDS,
                        help="sampling interval in seconds (default: 1)")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"),
                        help="new output directory (default: outputs)")
    args = parser.parse_args()
    try:
        metadata = extract_frames(args.video, args.output_dir, args.interval)
    except (ValueError, OSError, cv2.error) as error:
        parser.error(str(error))
    print(f"Saved {len(metadata)} frames; metadata: {args.output_dir / 'frames.json'}")


if __name__ == "__main__":
    main()
