#!/usr/bin/env bash
# Pull finished experiment results from the cluster node back to this container.
# Usage: ./cluster_sync_down.sh [conf path, default ./cluster_sync.conf] [--node NODE]
set -euo pipefail

ARGS=()
NODE_OVERRIDE=""
_i=0
_argv=("$@")
while [[ "$_i" -lt "${#_argv[@]}" ]]; do
  _arg="${_argv[$_i]}"
  case "$_arg" in
    --node)
      _i=$((_i+1))
      NODE_OVERRIDE="${_argv[$_i]:-}"
      [[ -z "$NODE_OVERRIDE" ]] && { echo "--node requires a value" >&2; exit 1; }
      ;;
    *) ARGS+=("$_arg") ;;
  esac
  _i=$((_i+1))
done
if [[ "${#ARGS[@]}" -gt 1 ]]; then
  echo "Usage: $0 [conf path] [--node NODE]" >&2
  exit 1
fi
CONF="${ARGS[0]:-./cluster_sync.conf}"
if [[ -n "$NODE_OVERRIDE" && ! "$NODE_OVERRIDE" =~ ^node4[0-9]{2}$ ]]; then
  echo "--node value doesn't look like a valid node name (expected e.g. node411): $NODE_OVERRIDE" >&2
  exit 1
fi
if [[ ! -f "$CONF" ]]; then
  echo "Conf file not found: $CONF" >&2
  exit 1
fi

# ---- Safe config loading: parsed as plain KEY=VALUE, never sourced as executable shell code ----
CONF_ALLOWED_VARS=" NODE CODE_LOCAL CODE_REMOTE DATA_LOCAL_BASE DATA_REMOTE_BASE RESULTS_REMOTE RESULTS_LOCAL LOG_FILE "
load_conf() {
  local conf_file="$1" key value
  while IFS='=' read -r key value || [[ -n "$key" ]]; do
    [[ -z "$key" || "$key" == \#* ]] && continue
    if [[ "$CONF_ALLOWED_VARS" != *" $key "* ]]; then
      echo "Conf file contains an unknown variable, refusing to load: $key" >&2
      exit 1
    fi
    printf -v "$key" '%s' "$value"
  done < "$conf_file"
}
load_conf "$CONF"
if [[ -n "$NODE_OVERRIDE" ]]; then
  NODE="$NODE_OVERRIDE"
fi

: "${NODE:?NODE not set in conf file}"
: "${RESULTS_REMOTE:?RESULTS_REMOTE not set}"
: "${RESULTS_LOCAL:?RESULTS_LOCAL not set}"

# Whether NODE came from the conf file or --node, it must look like a real node
# name before use — the conf file is "external input" too, and a NODE value
# starting with '-' would be parsed by rsync as a command-line option (option
# injection) instead of a real hostname.
if [[ ! "$NODE" =~ ^node4[0-9]{2}$ ]]; then
  echo "NODE doesn't look like a valid node name (expected e.g. node411, got: '$NODE')" >&2
  exit 1
fi

# Log this invocation locally (not committed to git; records time/node for later reference)
LOG_FILE="${LOG_FILE:-cluster_sync.log}"
log_event() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$LOG_FILE"
}
trap 'ec=$?; log_event "DOWN end node=$NODE status=$ec"; exit $ec' EXIT
log_event "DOWN start node=$NODE"

mkdir -p "$RESULTS_LOCAL"

echo "==> Pulling results: $NODE:$RESULTS_REMOTE/ -> $RESULTS_LOCAL/"
rsync -avz --protect-args --progress -- "${NODE}:${RESULTS_REMOTE%/}/" "${RESULTS_LOCAL%/}/"

echo "==> Done, results are in: $RESULTS_LOCAL"
