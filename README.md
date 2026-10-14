# Python retrieval and Video-RAG MVP

## Video-RAG Explorer web demo

The modular Gradio demo has three research tabs:

- **Video-RAG QA** shows the local MP4, accepts English/Persian questions, and
  reuses ASR/visual/multimodal evidence retrieval and validated citations. Answers,
  source types, timestamps, scores, and exact evidence are displayed. A retrieval
  only checkbox lets you inspect real evidence without an LLM call.
- **Visual Caption Comparison** displays an extracted frame and compares an
  independent description against context-conditioned Qwen2.5-VL-3B analysis.
  Both prompts separate observations, contextual inferences, and uncertainty.
- **Temporal Understanding** displays two frames with their actual timestamps
  and analyzes continuity, correspondence hypotheses, state changes, and uncertainty.
  Object correspondence is a model hypothesis, not verified tracking.

Use Python 3.10+ and an existing virtual environment with platform-appropriate
PyTorch installed. Install the demo profile **after** any older captioning profile
(the demo requires a newer Transformers version for Qwen2.5-VL):

```bash
source .venv/bin/activate
python -m pip install -r requirements-demo.txt
```

This installs software dependencies only. The app never downloads model weights,
videos, or datasets. Prepare the local MP4, `outputs/frames.json`, referenced
JPEGs, `outputs/transcript_small.json`, and `outputs/captions.json` separately using
the existing CLIs. Missing files give an error on the relevant operation; an absent
frame metadata file leaves the frame selectors empty. Relative frame paths resolve
from the repository root; paths must match real files (duplicated `frames/frames/`
paths are not silently repaired). Use `--video`, `--frames`, `--transcript`, and
`--captions` to select other local inputs.

QA connects to your **already running** vLLM server at `http://localhost:8000/v1`.
The app neither starts nor modifies vLLM. Set `VIDEO_RAG_LLM_MODEL` to its served
model name, or leave it unset to discover a single model via `/models` on demand.
`VIDEO_RAG_LLM_BASE_URL` / `--base-url` overrides the API base. If the server requires
a key, supply `VIDEO_RAG_LLM_API_KEY` securely. No key is sent when it is unset.
English captions and Persian ASR still have the documented lexical cross-language
retrieval limitation; supply an English visual query as needed.

For the two visual tabs, set `QWEN_VL_PYTHON` to a separate GPU environment's
Python executable and `QWEN_VL_MODEL_PATH` to your **existing local** Qwen2.5-VL-3B
checkpoint directory. `QWEN_VL_DEVICE` / `--device` selects the device (default
`cuda:0`). `CUDA_VISIBLE_DEVICES` controls which physical GPUs the worker sees;
device indices refer to that visible list, not the original physical GPU numbering.
No cache path or model-hub fallback is built in.

Gradio never loads Qwen2.5-VL in its own process. On each visual inference request,
it starts `experiments/visual_worker.py` with the configured Python, passes absolute
frame paths, prompt, checkpoint, and device via UTF-8 JSON on stdin, and receives
JSON on stdout. Model logs go to stderr. The worker uses the shared local-only
inference logic and exits after generation, releasing its model allocations.
Comparison runs two independent worker requests; model reload overhead is expected.
Workers are serialized, use a 300-second timeout, and report missing configuration,
missing packages, CUDA errors, invalid responses, and timeouts to the Gradio UI.
The child inherits `CUDA_VISIBLE_DEVICES`; offline Hugging Face flags and
`local_files_only=True` prevent weight downloads. Standalone experiment and other
CLI entrypoints keep their existing behavior. Research outputs may hallucinate.

### BRTX deployment with separate CPU and GPU environments

Use the existing environments without installing or modifying packages in either:

```bash
cd ~/codex-rag-demo
export QWEN_VL_PYTHON="$HOME/venvs/vllm/bin/python"
export QWEN_VL_MODEL_PATH="/absolute/path/to/your/existing/Qwen2.5-VL-3B-checkpoint"
export CUDA_VISIBLE_DEVICES=1
export QWEN_VL_DEVICE=cuda:0
export GRADIO_ANALYTICS_ENABLED=False
.venv/bin/python gradio_demo.py
```

Replace the checkpoint path with the tested local directory, and select a physical
GPU with free memory. With `CUDA_VISIBLE_DEVICES=1`, that GPU is exposed to the
worker as `cuda:0`. Gradio remains in `.venv` with CPU-only PyTorch; inference uses
the already tested `~/venvs/vllm/bin/python` (PyTorch 2.8.0+cu128). The executable's
venv path is preserved even if it is a symlink, so its environment stays selected.
No packages, weights, videos, or datasets are installed or downloaded by the app.

The existing vLLM server remains separate and unchanged at port 8000. Choose a GPU
with enough free memory for worker loading alongside vLLM, or use a different GPU;
the demo does not alter vLLM GPU visibility, configuration, processes, or allocations.
The QA tab continues using the same OpenAI-compatible API backend. Missing visual
configuration does not prevent QA, frame previews, or Gradio startup.

Start on the remote machine from the repository directory:

```bash
python gradio_demo.py
```

Run SSH forwarding on your workstation, replacing `user@host` with your remote
login:

```bash
ssh -N -L 7860:127.0.0.1:7860 user@host
```

Then open your workstation browser on loopback port 7860. The server binds only
`127.0.0.1:7860`, with `share=False`; public Gradio sharing is disabled. Only the
configured video and existing indexed frame files are explicitly allowed for
serving. Access is intended for the SSH user; do not expose this research UI to
an untrusted network. The app writes no research outputs or model weights.

Run the full test suite with the demo dependencies installed:

```bash
python -m unittest discover -s tests
```

Demo tests cover all QA modes, mocked answers/citations, independent/context
prompts, temporal timestamps, frame validation, served-model discovery, lazy offline
model loading, actual Gradio tab construction, and loopback launch configuration.
Codex validation covered local UI startup, real-data retrieval and frame previews,
and mocked model integration. CUDA and the existing vLLM service were unavailable
in that environment; successful GPU or live vLLM inference is not claimed.

A small demo of the retrieval stage of retrieval augmented generation (RAG).
It ranks a sample document collection using TF-IDF and cosine similarity and
prints the retrieved text, which could be passed to a language model as context.
This demo does not generate answers. The Video-RAG stages extract representative
frames from a local MP4 and generate visual captions for later indexing.

## Requirements

Python 3.9 or newer. Text retrieval uses only the standard library. Video extraction
uses CPU-only `opencv-python-headless` and its NumPy dependency. No pretrained
models, GPU dependencies, API keys, or external services are needed for extraction.
Optional captioning uses a pretrained Hugging Face model on the CPU.

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

The CLI samples at 0, 1, 2, ... seconds by default, saves JPEG images under
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

Set a different positive sampling interval (fractional seconds are supported):

```bash
python video_frames.py /path/to/video.mp4 --interval 5
```

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

## Caption extracted frames

Install the captioning dependencies in the same virtual environment. CPU PyTorch
is installed separately to avoid downloading CUDA dependencies:

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-captioning.txt
```

From the repository directory, first verify the pipeline on three frames:

```bash
HF_HOME=/workspace/shared/huggingface OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python caption_frames.py --limit 3
```

`HF_HOME` is a model cache outside the checkout; on another machine choose a
writable cache directory outside your repository. The first invocation downloads
`Salesforce/blip-image-captioning-base` (roughly 1 GB of model weights) from
Hugging Face and requires Internet access. Later runs reuse the cached weights.
Inference always uses CPU, evaluation mode, and deterministic beam search with
up to 30 new tokens per caption. Thread limits help avoid excessive CPU contention.

The CLI reads `outputs/frames.json` in collection order and writes
`outputs/captions.json`. Each entry preserves the original fields, including
`timestamp` and `frame_path`, and adds `caption` and `model`. Relative frame paths
are resolved from the invocation directory, just as in frame extraction.
`--limit N` processes only the first N frames; omit it to caption all frames.
`--frames` and `--output` override the input/output JSON paths; `--model` can select
another compatible BLIP checkpoint. Caption output is replaced only after a
successful run, while the frame images and source metadata are left intact.

Generated captions describe appearance and may be inaccurate; this stage does
not transcribe audio, recognize actions across time, or index captions for retrieval.
Generated images and output JSON are ignored by Git. Keep model caches outside
the checkout. No captioning test downloads or loads pretrained weights.

## Transcribe video audio

Install FFmpeg (including the `ffmpeg` command) using your operating system's
package manager, then install the optional ASR dependencies:

```bash
python -m pip install -r requirements-asr.txt
HF_HOME=/workspace/shared/huggingface HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=4 \
  python transcribe_video.py 6nqGlKyiWskr-qPC_0000.mp4
```

Choose a writable `HF_HOME` outside the checkout on other machines. The first run
downloads the faster-whisper `tiny` checkpoint (`Systran/faster-whisper-tiny`)
from Hugging Face. Inference uses CPU INT8 with four threads; no GPU is needed.
The CLI detects the language automatically and prints segments with timestamps,
inference time, and total runtime including extraction and model loading/download.

Use `--model` to select another model and `--language` to force a language instead
of detecting it. For example, run small on CPU INT8 with Persian forced, keeping
the tiny output intact:

```bash
HF_HOME=/workspace/shared/huggingface HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=4 \
  python transcribe_video.py 6nqGlKyiWskr-qPC_0000.mp4 \
  --model small --language fa --output outputs/transcript_small.json
```

When language is supplied, its reported probability reflects the forced choice,
not independent language detection. Model, compute type, and requested language
are recorded in the output JSON.

The first audio stream is extracted to a temporary 16 kHz mono WAV using FFmpeg.
Original presentation timestamps are retained: initial audio delays and gaps
are filled with silence so speech timestamps remain on the video's timeline,
whose origin is presentation time zero. No trimming or VAD filtering is applied.
The temporary audio file is removed after processing. Videos without decodable
audio and failed model downloads produce an error, rather than a fake transcript.

`outputs/transcript.json` contains `video_filename`, model/device, detected
`language` and its probability, audio duration, runtime measurements, and a
`segments` list. Each segment has `start`, `end`, `text`, and a `words` list with
word timestamps and probabilities. All timestamps are seconds on the original
video timeline. `--output` selects another destination. Existing output is replaced
only after transcription succeeds. Silent videos may produce an empty segment
list, and language detection for silence is unreliable. Tiny-model transcripts
and word timing are estimates and may contain recognition errors.

Generated transcripts remain under Git-ignored `outputs/`; model weights stay
in the external cache. ASR unit tests mock extraction and transcription results,
so they require no model download.

## Retrieve timestamped transcript evidence

The transcript retrieval CLI uses the existing TF-IDF and cosine-similarity
implementation, with no new dependencies, LLM, GPU, or network calls:

```bash
python video_retrieval.py --query "چرا مردم دوباره اعتماد کردند؟" --top-k 3
python video_retrieval.py --query "مراکش با چه مشکلات اقتصادی روبرو بود؟" \
  --transcript outputs/transcript_small.json --window 20 --overlap 10
```

By default it reads `outputs/transcript_small.json`, slides 20-second windows
with 10 seconds of overlap, and indexes the complete ASR segments intersecting
each window. Chunk start/end timestamps are the actual included segment bounds,
so chunks can extend beyond a nominal window. Source text is preserved verbatim;
neighboring chunks repeat overlapping evidence. Empty windows and identical
segment selections are skipped. `--overlap` must be nonnegative and smaller than
`--window`. Transcript segments must have finite, ordered start timestamps and
positive durations.

Persian indexing normalizes Arabic/Persian variants of kaf and yeh, removes
vowel marks and tatweel, and treats zero-width non-joiners as word boundaries.
Normalization applies to both queries and indexed text, while printed evidence
retains the original ASR spelling. There is no stemming, translation, semantic
search, or correction of transcription errors. Unknown query terms are ignored.

Each result prints its cosine score, original video filename, timestamp range,
and evidence text. Only positive matches are returned, up to `--top-k`; ties
retain chronological chunk order. Overlapping top results are expected. Scores
are similarities rather than probabilities, and results are retrieved passages,
not generated answers. The CLI builds chunks in memory without saving data.

Manual evaluation on the real small-model transcript used these Persian queries:

| Query | Top chunk | Score |
| --- | --- | --- |
| چرا مردم دوباره اعتماد کردند؟ | 29.96–51.58s | 0.2827 |
| چرا مشارکت در انتخابات پارلمانی پایین بود؟ | 7.14–32.44s | 0.4269 |
| مراکش با چه مشکلات اقتصادی روبرو بود؟ | 79.60–90.62s | 0.2100 |

These are observed retrieval results rather than an accuracy benchmark.
The election passage mentions participation but does not establish its cause.
The local `outputs/retrieval_evaluation.json` records all three top-k result sets,
including exact evidence and source segment timestamps; it is Git-ignored.

## Ask evidence-grounded questions

`video_qa.py` retrieves timestamped ASR chunks using `video_retrieval.py`, then
optionally sends those chunks and the question to a configurable OpenAI-compatible
Chat Completions API. It uses Python's standard library for HTTP and adds no
dependencies. ASR-only remains the default. v0.2 optionally fuses ASR text and
visual caption evidence; it does not send images or video directly to the LLM.

Validate retrieval and inspect the exact prompt without an endpoint or API key:

```bash
python video_qa.py --question "چرا مردم دوباره اعتماد کردند؟" --dry-run
```

Dry-run JSON has `status: "prompt_only"`, `answer: null`, retrieved `evidence`
with source text and timestamps, and the system/user `messages`. It makes no LLM
request and does not invent an answer.

For real generation, configure these environment variables through your runtime
or environment settings:

| Variable | Purpose |
| --- | --- |
| `VIDEO_RAG_LLM_BASE_URL` | Your API base URL, including its API prefix such as `/v1` |
| `VIDEO_RAG_LLM_MODEL` | Model identifier supported by your provider |
| `VIDEO_RAG_LLM_API_KEY` | Credential supplied securely; never save it in the repository |

There are no built-in endpoints, model names, or credentials. `--base-url` and
`--model` override the corresponding settings; keys are read only from the
environment. The request goes to `<base-url>/chat/completions`. Use `--no-auth`
only for an endpoint that explicitly requires no credential. If cloud networking
is restricted, allow your selected provider hostname in environment settings.
Selected questions and ASR excerpts are sent to that provider for generation.

```bash
python video_qa.py --question "چرا مردم دوباره اعتماد کردند؟" --top-k 3
```

`--transcript`, `--window`, and `--overlap` configure retrieval, and `--timeout`
sets the API timeout in seconds. Output is JSON on stdout; to retain a run,
redirect it to a file under Git-ignored `outputs/`.

The system prompt requires Persian answers to Persian questions, forbids outside
knowledge and following instructions in source text, and explicitly requires
abstention when retrieved evidence cannot support an answer. API responses must
be JSON with `answer`, `abstained`, and `citations`; each citation has an
`evidence_id` and an exact source `quote`. Validated output adds citation start/end
timestamps and includes all retrieved chunks with their source segments and scores.
No retrieved evidence produces a local Persian abstention without an API call
through the Python `ask_video` API. LLM abstentions retain retrieved evidence for
inspection. Malformed responses, unknown citation IDs, invented quotations, and
answers without citations are rejected; provider failures return an error.

Citation validation verifies provenance and exact quotes, not whether every
generated claim logically follows from them. An LLM can still misinterpret ASR
errors or insufficient evidence; inspect the cited passages. No real generated
answer has been validated in the current environment because no LLM endpoint,
model configuration, or credential is available. Mocked tests exercise supported
answers, abstention, citations, API requests, failures, and the dry-run CLI.

### v0.2 evidence fusion modes

`--mode asr` (default) preserves the previous CLI and retrieval behavior. It does
not load captions, even if `--captions` is supplied. ASR evidence retains its
`chunk_...` IDs and now explicitly carries `type: "asr"`.

`--mode visual` requires `--captions` and does not read a transcript. Visual
evidence retains `visual_...` IDs, `type: "visual"`, frame paths, and point
timestamps (`start == end`). `--mode multimodal` requires both the transcript
and `--captions`; it appends independently ranked visual results after ASR results.

```bash
python video_qa.py --mode visual --captions outputs/captions.json \
  --question "What is visible in the classroom?" --visual-top-k 2 --dry-run

python video_qa.py --mode multimodal --captions outputs/captions.json \
  --question "چرا مردم دوباره اعتماد کردند؟" --visual-query "people classroom" \
  --asr-top-k 3 --visual-top-k 2 --dry-run
```

`--asr-top-k` defaults to the existing `--top-k` value; `--visual-top-k` defaults
to 3. Each modality has its own budget and within-modality ranking. Their TF-IDF
scores use different corpora and are never compared or globally sorted. Evidence
IDs remain distinct across modalities, and prompts include source type, video
filename, timestamp bounds, and visual frame path. Citation validation accepts
ASR intervals and visual points and adds source type/frame path to citations.

The prompt tells Qwen (or another configured compatible LLM) to distinguish
spoken content from model-generated visual descriptions and to abstain for
unsupported claims, conflicting or insufficient evidence. Existing API/model
configuration is unchanged; remove `--dry-run` to use your configured backend.

**Cross-language limitation:** ASR is Persian, but current captions are English.
TF-IDF does not translate questions. A Persian question may retrieve no visual
evidence, even when a relevant frame exists. `--visual-query` supplies a separate
English retrieval query without changing the original question sent to the LLM.
No translation, embedding model, or new dependency is introduced. A returned
caption describes a single frame and does not establish identities, motivations,
speech, or events between frames. This is evidence fusion rather than a fully
multimodal perception system; ASR and caption errors remain possible.

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
Captioning tests use a mocked model to check limits, metadata preservation,
empty inputs, validation, and output preservation when inference fails. Install
`requirements-captioning.txt` to run the complete test suite.
Timestamped retrieval tests cover Persian normalization and ranking, overlapping
chunks, source timestamp/text preservation, silence, invalid inputs, and the CLI.
