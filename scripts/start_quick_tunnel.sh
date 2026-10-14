#!/usr/bin/env bash
# Explicitly optional: this exposes the local research UI without authentication.
set -euo pipefail

tunnel_binary=${BRTX_CLOUDFLARED:-cloudflared}
if ! command -v "$tunnel_binary" >/dev/null 2>&1; then
    printf 'cloudflared is not available. Set BRTX_CLOUDFLARED to an existing binary.\nNothing will be installed.\n' >&2
    exit 1
fi
printf 'WARNING: Quick Tunnel exposes the demo publicly with NO authentication.\nTemporary testing only; stop the tunnel when finished.\n' >&2
# The temporary public URL is assigned by Cloudflare and printed by cloudflared.
# No login, credentials, persistent tunnel, or dependency installation is needed.
exec "$tunnel_binary" --no-autoupdate tunnel --url http://127.0.0.1:7860
