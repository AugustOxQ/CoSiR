#!/usr/bin/env bash
# Opens a new tab in the caller's Herdr workspace (run it from a pane in the CoSiR workspace), starts a fresh Claude
# chat there as agent "improve" and sends it the start prompt of the method-improvements handoff.
#   bash docs/superpowers/handoffs/2026-10-07-method-improvements-launch.sh
set -euo pipefail

test "${HERDR_ENV:-}" = 1 || { echo "not inside a Herdr pane: run this from a pane of the CoSiR workspace" >&2; exit 1; }

REPO=/project/CoSiR
PROMPT_FILE="$REPO/docs/superpowers/handoffs/2026-10-07-method-improvements-start-prompt.md"
LABEL=method-improvements
AGENT=improve
WS="${HERDR_WORKSPACE_ID:?HERDR_WORKSPACE_ID is not set}"

json() { python3 -c "import json,sys; d=json.load(sys.stdin); print(eval(sys.argv[1], {'d': d}))" "$1"; }

WS_LABEL=$(herdr workspace list | json "[w['label'] for w in d['result']['workspaces'] if w['workspace_id']=='$WS'][0]")
echo "workspace $WS ($WS_LABEL)"
[ "$WS_LABEL" = "CoSiR" ] || { echo "this pane's workspace is '$WS_LABEL', not CoSiR; run the script from a CoSiR pane" >&2; exit 1; }

TAB_JSON=$(herdr tab create --workspace "$WS" --cwd "$REPO" --label "$LABEL" --no-focus)
PANE=$(printf '%s' "$TAB_JSON" | json "d['result']['root_pane']['pane_id']")
TAB=$(printf '%s' "$TAB_JSON" | json "d['result']['tab']['tab_id']")
echo "tab $TAB, pane $PANE"

herdr agent start "$AGENT" --kind claude --pane "$PANE" --timeout 90000
herdr agent prompt "$AGENT" "$(cat "$PROMPT_FILE")"
echo "agent '$AGENT' started in tab '$LABEL' ($TAB) with the handoff prompt; focus it with: herdr tab focus $TAB"
