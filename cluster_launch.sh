#!/usr/bin/env bash
# Automatically locate the DAS6 tmux window that's "already connected to the
# reserved compute node" and send the training command into it. No need to
# attach, and no need to know in advance which node it is — detected fresh
# every time.
#
# Prerequisite: your DAS6 tmux always keeps this layout — one window running a
#       monitor like mywatch, another window being the interactive shell that
#       auto-ssh'd into whatever node got reserved. The convention is to name
#       that window "srun-node4XX-gpu-XX" (e.g. srun-node411-gpu-02) — this is
#       more robust than "scan the current screen content for a prompt
#       pattern", since it doesn't break when the login shell/prompt format
#       changes. This script checks the window name first, falling back to
#       scanning screen content only if that doesn't match (see the order in
#       DETECT_SCRIPT).
#
# Usage: ./cluster_launch.sh [conf path] "<full command to run on the node>" [--node NODE]
#        ./cluster_launch.sh [conf path] --detect-only [--node NODE]
#   With exactly one non-flag argument, it's treated as the command, and the
#   conf path defaults to ./cluster_sync.conf.
#   --detect-only: only detect the node and update cluster_sync.conf's NODE,
#       then exit — no command is sent.
#   --node NODE: only needed when multiple nodes are reserved at once — picks
#       which one to control (e.g. node411 and node412 both reserved, and you
#       want to send a command to node412 specifically). Without this flag,
#       the old behavior applies: exactly one node must currently be in use,
#       or it errors out asking you to confirm.
#   It's a good idea to run --detect-only once before cluster_sync_up.sh pushes
#   code, otherwise the code might get pushed to a stale, no-longer-reserved node.
set -euo pipefail

DETECT_ONLY=0
NODE_ARG=""
ARGS=()
_i=0
_argv=("$@")
while [[ "$_i" -lt "${#_argv[@]}" ]]; do
  _arg="${_argv[$_i]}"
  case "$_arg" in
    --detect-only) DETECT_ONLY=1 ;;
    --node)
      _i=$((_i+1))
      NODE_ARG="${_argv[$_i]:-}"
      [[ -z "$NODE_ARG" ]] && { echo "--node requires a value" >&2; exit 1; }
      ;;
    *) ARGS+=("$_arg") ;;
  esac
  _i=$((_i+1))
done

if [[ -n "$NODE_ARG" && ! "$NODE_ARG" =~ ^node4[0-9]{2}$ ]]; then
  echo "--node value doesn't look like a valid node name (expected e.g. node411): $NODE_ARG" >&2
  exit 1
fi

if [[ "$DETECT_ONLY" -eq 1 ]]; then
  if [[ "${#ARGS[@]}" -gt 1 ]]; then
    echo "Usage: $0 [conf path] --detect-only [--node NODE]" >&2
    exit 1
  fi
  CONF="${ARGS[0]:-./cluster_sync.conf}"
  CMD=""
else
  if [[ "${#ARGS[@]}" -lt 1 || "${#ARGS[@]}" -gt 2 ]]; then
    echo "Usage: $0 [conf path] \"<command>\" [--node NODE]" >&2
    exit 1
  fi
  if [[ "${#ARGS[@]}" -eq 1 ]]; then
    CONF="./cluster_sync.conf"
    CMD="${ARGS[0]}"
  else
    CONF="${ARGS[0]}"
    CMD="${ARGS[1]}"
  fi
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

LOG_FILE="${LOG_FILE:-cluster_sync.log}"
log_event() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$LOG_FILE"
}
trap 'ec=$?; log_event "LAUNCH end status=$ec"; exit $ec' EXIT

# Safely run a script on the remote host, optionally carrying any number of
# KEY=VALUE variables (the last argument is always the script body). Same
# reasoning as remote_bash_kv in cluster_sync_up.sh — a variable must never
# appear on the ssh command line, including the seemingly-safe idea of
# inlining `printf '%q'` output into a quoted string passed whole to ssh: %q is
# bash's own escaping format, and the remote login shell is fish, which
# doesn't understand bash-specific escapes — it has to go entirely through
# stdin to a fixed `bash -s`.
remote_bash_kv() {
  local node="$1"; shift
  if [[ "$#" -lt 1 ]]; then
    echo "remote_bash_kv: missing script argument" >&2
    return 1
  fi
  local script="${*: -1}"
  local n=$#
  local kvs=("${@:1:$((n-1))}")
  if (( ${#kvs[@]} % 2 != 0 )); then
    echo "remote_bash_kv: KEY/VALUE arguments must come in pairs" >&2
    return 1
  fi
  local i key
  {
    for ((i = 0; i < ${#kvs[@]}; i += 2)); do
      key="${kvs[i]}"
      if [[ ! "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
        echo "remote_bash_kv: invalid variable name: $key" >&2
        return 1
      fi
      printf '%s=%q\n' "$key" "${kvs[i+1]}"
    done
    printf '%s\n' "$script"
  } | ssh -- "$node" bash -s
}

echo "==> Scanning tmux windows on DAS6 for one connected to a compute node..."

# Remote detection script: for each window, first check whether the window
# *name* matches the "srun-node4XX-..." convention (most robust, doesn't
# depend on shell/prompt format), only falling back to capturing the current
# screen content and looking for a "wding at node4XX" prompt pattern (fish
# prompt convention; on 2026-09-02 the prompt format changed once and broke
# this match, which is why it's now only a fallback, not the primary signal).
# Only look at the currently visible screen (capture-pane's default), not
# scrollback history, to avoid matching stale content in a monitoring window's
# scrollback that happens to mention some node4XX name (e.g. a squeue table).
# Uses `while read` rather than `for x in $(...)`: the latter word-splits/globs
# the output, which would mis-split a window name containing spaces or glob
# characters. The regex ends with (?!\d) to avoid misreading the first three
# digits of a longer number (in case the cluster ever grows to node4XXX) as a
# valid match.
DETECT_SCRIPT='
tmux list-windows -a -F "#{session_name}:#{window_index}|#{window_name}" 2>/dev/null | while IFS="|" read -r target wname; do
  node=$(printf "%s\n" "$wname" | grep -oP "(?<=^srun-)node4[0-9]{2}(?=-)" | tail -1 || true)
  if [ -z "$node" ]; then
    node=$(tmux capture-pane -t "$target" -p 2>/dev/null | grep -oP "wding at node4[0-9]{2}(?!\d)" | tail -1 | grep -oE "node4[0-9]{2}" || true)
  fi
  if [ -n "$node" ]; then
    printf "%s %s\n" "$target" "$node"
  fi
done
'
MATCHES="$(ssh DAS6 bash -s <<EOF
$DETECT_SCRIPT
EOF
)"

if [[ -z "$MATCHES" ]]; then
  echo "No tmux window found connected to a compute node (neither window name nor prompt matched)." >&2
  echo "The node reservation may have expired, or the auto-connecting window hasn't connected yet — please check yourself." >&2
  exit 1
fi

if [[ -n "$NODE_ARG" ]]; then
  # A specific node was requested: filter directly by node. Multiple nodes
  # being reserved at once isn't "ambiguous" in this case, since the caller
  # has already said which one they mean.
  FILTERED="$(printf '%s\n' "$MATCHES" | awk -v n="$NODE_ARG" '$2==n')"
  if [[ -z "$FILTERED" ]]; then
    echo "No tmux window found connected to the requested node $NODE_ARG. Nodes currently visible:" >&2
    printf '%s\n' "$MATCHES" | awk '{print $2}' | sort -u >&2
    exit 1
  fi
  MATCHES="$FILTERED"
else
  DISTINCT_NODE_COUNT="$(printf '%s\n' "$MATCHES" | awk '{print $2}' | sort -u | wc -l)"
  if [[ "$DISTINCT_NODE_COUNT" -gt 1 ]]; then
    echo "Found windows connected to more than one distinct node — can't pick automatically. Use --node <name> to specify one, or confirm yourself:" >&2
    printf '%s\n' "$MATCHES" >&2
    exit 1
  fi
fi

TARGET="$(printf '%s\n' "$MATCHES" | head -1 | awk '{print $1}')"
NODE="$(printf '%s\n' "$MATCHES" | head -1 | awk '{print $2}')"
echo "==> Found: tmux window $TARGET, currently connected to node $NODE"

# Write the detected real node back into the conf file, so up/down scripts
# stay consistent with what was found here (only when --node wasn't given —
# passing --node means a one-off override for this call, and shouldn't clobber
# the default other, --node-less callers rely on).
if [[ -z "$NODE_ARG" ]]; then
  if grep -q '^NODE=' "$CONF"; then
    sed -i "s/^NODE=.*/NODE=$NODE/" "$CONF"
  else
    printf 'NODE=%s\n' "$NODE" >> "$CONF"
  fi
  echo "==> Updated NODE in $CONF to $NODE"
fi

if [[ "$DETECT_ONLY" -eq 1 ]]; then
  log_event "LAUNCH detect-only node=$NODE tmux_target=$TARGET"
  echo "==> --detect-only: no command sent, stopping here."
  exit 0
fi

log_event "LAUNCH start node=$NODE tmux_target=$TARGET"

echo "==> About to send this command to $TARGET:"
echo "    $CMD"

remote_bash_kv DAS6 TARGET "$TARGET" CMD "$CMD" 'tmux send-keys -t "$TARGET" "$CMD" Enter'

log_event "LAUNCH sent node=$NODE tmux_target=$TARGET cmd=$CMD"

sleep 3
echo "==> Window content right after sending:"
remote_bash_kv DAS6 TARGET "$TARGET" 'tmux capture-pane -t "$TARGET" -p | grep -v "^[[:space:]]*$" | tail -15'

echo "==> Done. To keep watching it run, attach to that tmux session on DAS6 yourself — this won't kick you off."
