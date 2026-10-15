#!/usr/bin/env bash
# Opens a new tab in the caller's Herdr workspace (run it from a pane in the CoSiR workspace), starts a fresh Claude
# chat there as agent "next" and sends it the start prompt of the whats-next handoff.
#   bash docs/superpowers/handoffs/2026-10-08-whats-next-launch.sh
set -euo pipefail

test "${HERDR_ENV:-}" = 1 || { echo "not inside a Herdr pane: run this from a pane of the CoSiR workspace" >&2; exit 1; }

REPO=/project/CoSiR
PROMPT_FILE="$REPO/docs/superpowers/handoffs/2026-10-08-whats-next-start-prompt.md"
LABEL=whats-next
AGENT=next
WS="${HERDR_WORKSPACE_ID:?HERDR_WORKSPACE_ID is not set}"

json() { python3 -c "import json,sys; d=json.load(sys.stdin); print(eval(sys.argv[1], {'d': d}))" "$1"; }

WS_LABEL=$(herdr workspace list | json "[w['label'] for w in d['result']['workspaces'] if w['workspace_id']=='$WS'][0]")
echo "workspace $WS ($WS_LABEL)"
[ "$WS_LABEL" = "CoSiR" ] || { echo "this pane's workspace is '$WS_LABEL', not CoSiR; run the script from a CoSiR pane" >&2; exit 1; }

TAB_JSON=$(herdr tab create --workspace "$WS" --cwd "$REPO" --label "$LABEL" --no-focus)
PANE=$(printf '%s' "$TAB_JSON" | json "d['result']['root_pane']['pane_id']")
TAB=$(printf '%s' "$TAB_JSON" | json "d['result']['tab']['tab_id']")
echo "tab $TAB, pane $PANE"

# `herdr agent start` refuses a fresh fish pane ("not an available shell", 2026-10-08), so start claude as a
# command, wait for its banner, then name the detected agent.
herdr pane wait-output "$PANE" --regex '[#$>] *$|╰─' --timeout 30000 >/dev/null
herdr pane run "$PANE" "claude" >/dev/null
herdr pane wait-output "$PANE" --match "Claude Code" --timeout 90000 >/dev/null
sleep 3
herdr agent rename "$PANE" "$AGENT" >/dev/null
herdr agent prompt "$AGENT" "$(cat "$PROMPT_FILE")"
echo "agent '$AGENT' started in tab '$LABEL' ($TAB) with the handoff prompt; focus it with: herdr tab focus $TAB"
