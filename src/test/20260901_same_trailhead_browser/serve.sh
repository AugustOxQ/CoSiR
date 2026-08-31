#!/usr/bin/env bash
# Serve this directory over HTTP so the browser page can be viewed from outside
# this (headless) container, e.g. via an SSH port-forward:
#   ssh -L 8000:localhost:8000 <container-host>
# then open http://localhost:8000/same_trailhead_browser.html on your machine.
set -euo pipefail

PORT="${1:-8000}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Serving $DIR on http://localhost:$PORT/same_trailhead_browser.html"
echo "If viewing from outside this container, forward the port first:"
echo "  ssh -L $PORT:localhost:$PORT <container-host>"
echo "Press Ctrl+C to stop."

cd "$DIR"
exec python3 -m http.server "$PORT"
