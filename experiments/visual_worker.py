"""One JSON request on stdin, one JSON response on stdout; GPU work stays here."""

from contextlib import redirect_stdout
import json
import sys

if __package__:
    from .visual_backend import LocalVisualBackend
else:
    from visual_backend import LocalVisualBackend


def run_request(request):
    if not isinstance(request, dict):
        raise ValueError("Request must be a JSON object")
    paths = request.get("frame_paths")
    if (not isinstance(paths, list) or len(paths) not in (1, 2)
            or any(not isinstance(path, str) or not path for path in paths)):
        raise ValueError("Supply one or two frame paths")
    for key in ("prompt", "model_path", "device"):
        if not isinstance(request.get(key), str) or not request[key].strip():
            raise ValueError(f"Request requires {key}")
    backend = LocalVisualBackend(request["model_path"], request["device"])
    return {"text": backend.generate(paths, request["prompt"])}


def main():
    try:
        request = json.load(sys.stdin)
        # Keep model progress/logs out of the machine-readable response stream.
        with redirect_stdout(sys.stderr):
            response = run_request(request)
    except Exception as error:
        print(json.dumps({"error": f"{type(error).__name__}: {error}"}, ensure_ascii=False))
        return 1
    print(json.dumps(response, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
