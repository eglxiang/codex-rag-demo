# Python retrieval and Video-RAG MVP

A small demo of the retrieval stage of retrieval augmented generation (RAG).
It ranks a sample document collection using TF-IDF and cosine similarity and
prints the retrieved text, which could be passed to a language model as context.
This demo does not call a model or generate answers. The first Video-RAG stage
extracts representative frames from a local MP4 for later indexing.

## Requirements

Python 3.9 or newer. Text retrieval uses only the standard library. Video extraction
uses CPU-only `opencv-python-headless` and its NumPy dependency. No pretrained
models, GPU dependencies, API keys, or external services are needed.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Extract video frames

From the repository directory, with the virtual environment activated:

```bash
python video_frames.py /path/to/video.mp4
```

The CLI samples at 0, 5, 10, ... seconds, saves JPEG images under
`outputs/frames/`, and writes the metadata list to `outputs/frames.json`.
These are representative sampled frames, rather than codec keyframes or
scene-change detections. Each record includes:

```json
{
  "video_filename": "video.mp4",
  "timestamp": 5.0,
  "frame_path": "outputs/frames/frame_00000150.jpg",
  "frame_index": 150,
  "width": 1920,
  "height": 1080
}
```

Timestamps are seconds, calculated from decoded frame index and reported FPS.
Use constant-frame-rate MP4 inputs; variable-frame-rate timing is not supported
yet. The first frame at or after each target is selected, so a target may round
up to the next available frame. Short videos still produce a frame at zero.
Frames are decoded sequentially, without loading the entire video into memory.

Output paths are relative to the invocation directory when a relative output
directory is supplied. The CLI refuses an existing output directory to protect
previous results. For another run, choose a new directory:

```bash
python video_frames.py /path/to/another.mp4 --output-dir outputs/another
```

Invalid inputs and image-writing errors return a nonzero exit status. A failed
run may leave partial images; inspect them and choose a new output directory
before retrying. Metadata is written after extraction completes. OpenCV decoder
failures during reading are treated as end of stream, so damaged videos may
produce partial results. This stage does not yet connect frames to text retrieval.

## Query the collection

From the repository directory:

```bash
python3 retrieval.py "tomatoes soil sunlight" --top-k 2
python3 retrieval.py "cosine similarity"
python3 retrieval.py --help
```

Results include the document title, ID, similarity score, and full text. Only
positive matches are returned; an empty or unknown query reports no matches.
The default maximum is three results. `--top-k` must be a positive integer.

Use your own collection with:

```bash
python3 retrieval.py "your query" --documents /path/to/documents.json
```

The JSON file must contain a list of objects with string `id`, `title`, and `text`
fields. See `documents.json` for five small examples. An empty collection is valid.
The default sample path is resolved relative to the script, so the CLI can also
run from another directory.

## How retrieval works

Text is lowercased and split into word tokens. Each document is represented by
raw term counts multiplied by smoothed inverse document frequency:
`idf(term) = log((1 + document_count) / (1 + document_frequency)) + 1`.
The query uses the same IDF weights and vocabulary. Cosine similarity compares
the query and document vectors; results are sorted from highest score to lowest,
with collection order breaking ties.

This is lexical retrieval: it has no stemming, stop-word removal, or semantic
embeddings. Unknown query terms are ignored. Scores measure vector similarity,
not probabilities or confidence in an answer.

## Run the tests

From the repository directory:

```bash
python3 -m unittest discover -s tests -v
```

Tests cover tokenization, TF-IDF weights, cosine similarity, ranking, result
limits, empty inputs, validation, and command-line behavior. Video tests use
mocked captures and a tiny synthetic MP4 to check sampling, image contents,
metadata, invalid inputs, resource cleanup, and the CLI end to end.
