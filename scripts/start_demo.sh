#!/usr/bin/env bash
# Start only Gradio. Existing environments and vLLM are left untouched.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_dir=${BRTX_PROJECT_DIR:-$(cd -- "$script_dir/.." && pwd)}
venv_dir=${BRTX_VENV_DIR:-$project_dir/.venv}

if [[ ! -f "$project_dir/gradio_demo.py" ]]; then
    printf 'Missing gradio_demo.py in BRTX_PROJECT_DIR: %s\n' "$project_dir" >&2
    exit 1
fi
if [[ ! -x "$venv_dir/bin/python" ]]; then
    printf 'Missing existing virtualenv Python: %s/bin/python\nNo dependencies will be installed.\n' "$venv_dir" >&2
    exit 1
fi
export GRADIO_ANALYTICS_ENABLED=${GRADIO_ANALYTICS_ENABLED:-False}
cd -- "$project_dir"
# gradio_demo.py fixes server_name=127.0.0.1, server_port=7860, share=False.
exec "$venv_dir/bin/python" "$project_dir/gradio_demo.py" "$@"
