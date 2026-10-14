#!/usr/bin/env bash
set -euo pipefail

demo_session=${BRTX_DEMO_SESSION:-videorag-demo}
tunnel_session=${BRTX_TUNNEL_SESSION:-videorag-tunnel}
status=0

if command -v tmux >/dev/null 2>&1; then
    if tmux has-session -t "=$demo_session" 2>/dev/null; then
        printf 'Gradio tmux session: running (%s)\n' "$demo_session"
    else
        printf 'Gradio tmux session: absent (%s)\n' "$demo_session"
        status=1
    fi
    if tmux has-session -t "=$tunnel_session" 2>/dev/null; then
        printf 'Quick Tunnel tmux session: running (%s)\n' "$tunnel_session"
    else
        printf 'Quick Tunnel tmux session: absent (optional)\n'
    fi
else
    printf 'tmux is unavailable; session status cannot be checked.\n'
    status=1
fi

if command -v curl >/dev/null 2>&1; then
    if curl --noproxy '*' --fail --silent --max-time 5 http://127.0.0.1:7860/config >/dev/null; then
        printf 'Gradio HTTP readiness: ready on 127.0.0.1:7860\n'
    else
        printf 'Gradio HTTP readiness: unavailable\n'
        status=1
    fi
else
    printf 'curl is unavailable; HTTP readiness cannot be checked.\n'
    status=1
fi
exit "$status"
