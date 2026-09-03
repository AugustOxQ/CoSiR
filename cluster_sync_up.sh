#!/usr/bin/env bash
# Push project code to a DAS6 compute node, and optionally sync one dataset's
# pre-extracted feature cache.
#
# Code sync goes through git (this repo is public, the node does an anonymous
# https clone/pull, no credentials needed on the node) instead of a full-tree
# rsync mirror: locally commit+push a branch (or commit message) that mentions
# "cluster run", and the node clones/fetches that same branch. Matching git
# HEAD SHAs on both sides is the most direct proof that "the code is current" —
# simpler and more trustworthy than a full-tree checksum diff.
#
# Usage: ./cluster_sync_up.sh [conf path] [dataset name] [--force-data] [--clean] [--node NODE] [--allow-any-branch]
#   conf path: defaults to ./cluster_sync.conf
#   dataset name: a subdirectory name under DATA_LOCAL_BASE (e.g. redcaps_300k_diverse);
#             omit to sync code only, leaving data untouched
#   --force-data: re-sync the dataset even if it already exists on the node (default: skip)
#   --clean: force a fresh `git clone` on the node even if one already exists there
#            (normally it does an incremental fetch+reset --hard instead, which
#            ends up equally clean — --clean is only for "I just want a guaranteed
#            fresh start", not needed in normal use)
#   --node NODE: override the NODE from the conf file (e.g. when node411 and
#            node412 are both reserved and you want to push to just one of them
#            without touching the cached default in cluster_sync.conf)
#   --allow-any-branch: skip the "branch name or commit message must mention
#            cluster run" check (see the MARKER section below; normally not
#            needed — the check exists to prevent accidentally syncing a branch
#            that was never meant to run)
set -euo pipefail

# ---- Argument parsing: strip all flags first, then parse what's left positionally; extra args are an error ----
ARGS=()
FORCE_DATA=0
CLEAN=0
ALLOW_ANY_BRANCH=0
NODE_OVERRIDE=""
_i=0
_argv=("$@")
while [[ "$_i" -lt "${#_argv[@]}" ]]; do
  _arg="${_argv[$_i]}"
  case "$_arg" in
    --force-data) FORCE_DATA=1 ;;
    --clean) CLEAN=1 ;;
    --allow-any-branch) ALLOW_ANY_BRANCH=1 ;;
    --node)
      _i=$((_i+1))
      NODE_OVERRIDE="${_argv[$_i]:-}"
      [[ -z "$NODE_OVERRIDE" ]] && { echo "--node requires a value" >&2; exit 1; }
      ;;
    *) ARGS+=("$_arg") ;;
  esac
  _i=$((_i+1))
done
if [[ "${#ARGS[@]}" -gt 2 ]]; then
  echo "Usage: $0 [conf path] [dataset name] [--force-data] [--clean] [--node NODE] [--allow-any-branch]" >&2
  exit 1
fi
CONF="${ARGS[0]:-./cluster_sync.conf}"
DATASET="${ARGS[1]:-}"

if [[ -n "$NODE_OVERRIDE" && ! "$NODE_OVERRIDE" =~ ^node4[0-9]{2}$ ]]; then
  echo "--node value doesn't look like a valid node name (expected e.g. node411): $NODE_OVERRIDE" >&2
  exit 1
fi

if [[ ! -f "$CONF" ]]; then
  echo "Conf file not found: $CONF" >&2
  exit 1
fi

# ---- Safe config loading: parsed as plain KEY=VALUE, never sourced as executable shell code ----
# (the conf path comes from a command-line argument; sourcing it would let the
# caller run arbitrary local code)
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
: "${CODE_LOCAL:?CODE_LOCAL not set}"
: "${CODE_REMOTE:?CODE_REMOTE not set}"

# Whether NODE came from the conf file or --node, it must look like a real node
# name before use — the conf file is "external input" too (the caller can pass
# any path), and a NODE value starting with '-' would be parsed by ssh/rsync as
# a command-line option (option injection) instead of a real hostname.
if [[ ! "$NODE" =~ ^node4[0-9]{2}$ ]]; then
  echo "NODE doesn't look like a valid node name (expected e.g. node411, got: '$NODE')" >&2
  exit 1
fi

# Safely run a script on the remote host, optionally carrying any number of
# KEY=VALUE variables (the last argument is always the script body).
# Important: verified by direct testing — passing a variable as an ssh
# command-line argument (`ssh host cmd "$var"`) is unsafe: ssh just joins
# multiple command-line arguments with spaces and hands that to the remote
# login shell, without re-quoting, so any value containing a space gets split
# into several words (`env VAR=val bash -s` has the same problem). So a
# variable must never appear on the ssh command line — it can only go through
# stdin, to a fixed `bash -s` with no variables on its command line.
# `ssh --` is a second line of defense in case NODE somehow slips past
# validation while starting with '-' and gets parsed as an option; the primary
# defense is the NODE format check above.
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

# Log this invocation locally (not committed to git; records time/node/dataset for later reference)
LOG_FILE="${LOG_FILE:-cluster_sync.log}"
log_event() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$LOG_FILE"
}
trap 'ec=$?; log_event "UP end node=$NODE dataset=${DATASET:-none} status=$ec"; exit $ec' EXIT
log_event "UP start node=$NODE dataset=${DATASET:-none} force_data=$FORCE_DATA clean=$CLEAN"

if [[ ! -d "$CODE_LOCAL" ]]; then
  echo "Local code directory doesn't exist: $CODE_LOCAL" >&2
  exit 1
fi
if ! git -C "$CODE_LOCAL" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Local code directory isn't a git working tree: $CODE_LOCAL" >&2
  exit 1
fi

# ---- 1. Local tree must be clean: uncommitted/untracked changes won't show up via git pull, treating that as fine is dangerous ----
DIRTY="$(git -C "$CODE_LOCAL" status --porcelain)"
if [[ -n "$DIRTY" ]]; then
  echo "Local tree has uncommitted/untracked changes, which git sync won't carry over — deal with them first:" >&2
  printf '%s\n' "$DIRTY" >&2
  exit 1
fi

BRANCH="$(git -C "$CODE_LOCAL" rev-parse --abbrev-ref HEAD)"
if [[ "$BRANCH" == "HEAD" ]]; then
  echo "Local repo is in detached HEAD state — don't know which branch to push, checkout a named branch first" >&2
  exit 1
fi
LOCAL_SHA="$(git -C "$CODE_LOCAL" rev-parse HEAD)"
SUBJECT="$(git -C "$CODE_LOCAL" log -1 --format=%s)"

REPO_URL="$(git -C "$CODE_LOCAL" remote get-url origin)"
case "$REPO_URL" in
  https://*|http://*) ;;
  *)
    echo "origin isn't an http(s) URL ($REPO_URL) — the node does an anonymous https clone/pull with no ssh key, so please point origin at the https form" >&2
    exit 1
    ;;
esac

# ---- 2. "cluster run" marker check: guards against accidentally syncing a branch that was never meant to run ----
# If either the branch name or the latest commit message contains "cluster run" /
# "cluster-run" / "clusterrun" (case-insensitive) anywhere, this push is treated
# as intentional. This is purely a self-imposed safety valve, not a git
# mechanism — --allow-any-branch skips it.
MARKER_RE='cluster[-_ ]?run'
if [[ "$ALLOW_ANY_BRANCH" -ne 1 ]] && ! printf '%s\n%s\n' "$BRANCH" "$SUBJECT" | grep -qiE "$MARKER_RE"; then
  echo "Neither the branch name ('$BRANCH') nor the latest commit message ('$SUBJECT') mentions 'cluster run' —" >&2
  echo "this is a deliberate check to prevent accidentally syncing a branch/commit that wasn't meant for a cluster run." >&2
  echo "Either rename the branch / update the commit message to include that marker, or pass --allow-any-branch to skip it." >&2
  exit 1
fi

# ---- 3. Make sure origin is current: push if local is ahead, abort if local is behind/diverged (never auto-rebase/overwrite) ----
# It's normal, not an error, for the branch to not exist on the remote yet
# (e.g. the first push of a new cluster-run branch) — probe with ls-remote
# first and only fetch if it exists; otherwise a bare `git fetch origin
# "$BRANCH"` would itself fail because the remote ref doesn't exist, aborting
# the whole script right here under set -e, before ever reaching the push below.
echo "==> Checking whether origin already has branch $BRANCH ..."
if git -C "$CODE_LOCAL" ls-remote --exit-code --heads origin "$BRANCH" >/dev/null 2>&1; then
  git -C "$CODE_LOCAL" fetch origin "$BRANCH"
  REMOTE_TRACK_SHA="$(git -C "$CODE_LOCAL" rev-parse "origin/$BRANCH" 2>/dev/null || echo "")"
else
  echo "==> origin doesn't have branch $BRANCH yet (treating this as a first-time push)"
  REMOTE_TRACK_SHA=""
fi
if [[ "$REMOTE_TRACK_SHA" != "$LOCAL_SHA" ]]; then
  if [[ -n "$REMOTE_TRACK_SHA" ]] && ! git -C "$CODE_LOCAL" merge-base --is-ancestor "$REMOTE_TRACK_SHA" HEAD; then
    echo "Local branch is behind or has diverged from origin/$BRANCH — not handling this automatically (to avoid overwriting remote history), pull/rebase manually first" >&2
    exit 1
  fi
  echo "==> Local is ahead of origin/$BRANCH, pushing: git push origin $BRANCH"
  git -C "$CODE_LOCAL" push origin "$BRANCH"
else
  echo "==> Local already matches origin/$BRANCH, skipping push"
fi

# ---- 4. On the node: clone, or fetch+reset --hard, to the same commit ----
MANIFEST_PATH="${CODE_LOCAL%/}/.cluster-extra-sync"
EXTRA_PATHS=""
if [[ -f "$MANIFEST_PATH" ]]; then
  EXTRA_PATHS="$(grep -vE '^[[:space:]]*(#|$)' "$MANIFEST_PATH" || true)"
fi

echo "==> Syncing code on $NODE (git): $CODE_REMOTE @ $BRANCH"
GIT_SYNC_SCRIPT='
set -euo pipefail
mkdir -p "$(dirname "$CODE_REMOTE")"
{
  if [ "$CLEAN" = "1" ] || [ ! -d "$CODE_REMOTE/.git" ]; then
    rm -rf "$CODE_REMOTE"
    git clone --branch "$BRANCH" "$REPO_URL" "$CODE_REMOTE"
  else
    git -C "$CODE_REMOTE" remote set-url origin "$REPO_URL"
    git -C "$CODE_REMOTE" fetch origin "$BRANCH"
    git -C "$CODE_REMOTE" checkout -B "$BRANCH" "origin/$BRANCH"
    git -C "$CODE_REMOTE" reset --hard "origin/$BRANCH"
  fi
  CLEAN_EXCLUDES=()
  if [ -n "$EXTRA_PATHS" ]; then
    while IFS= read -r p; do
      [ -z "$p" ] && continue
      CLEAN_EXCLUDES+=(-e "$p")
    done <<< "$EXTRA_PATHS"
  fi
  git -C "$CODE_REMOTE" clean -fdx "${CLEAN_EXCLUDES[@]}"
} 1>&2
git -C "$CODE_REMOTE" rev-parse HEAD
'
REMOTE_SHA="$(remote_bash_kv "$NODE" REPO_URL "$REPO_URL" BRANCH "$BRANCH" CODE_REMOTE "$CODE_REMOTE" CLEAN "$CLEAN" EXTRA_PATHS "$EXTRA_PATHS" "$GIT_SYNC_SCRIPT")"

echo "==> Verifying: do both sides' git HEADs match exactly (this is the proof that \"the code is current\", not a guess)"
if [[ "$REMOTE_SHA" != "$LOCAL_SHA" ]]; then
  echo "Warning: the node's git HEAD ($REMOTE_SHA) doesn't match local ($LOCAL_SHA)" >&2
  exit 1
fi
echo "==> Verified: $CODE_REMOTE on the node matches local's git HEAD exactly ($LOCAL_SHA), branch $BRANCH"

# ---- 5. Manually sync anything listed in .cluster-extra-sync — gitignored, but still needed on the node ----
# These are typically local-only overrides, local-only credentials, etc. that
# genuinely shouldn't be in git but are still needed when the node runs the
# code — few enough in number that a targeted rsync per path is much simpler
# than a full-tree sync.
if [[ -n "$EXTRA_PATHS" ]]; then
  echo "==> Syncing the extra paths listed in .cluster-extra-sync (outside of git)"
  while IFS= read -r rel; do
    [[ -z "$rel" ]] && continue
    case "$rel" in
      /*|*..*)
        echo "Illegal path in .cluster-extra-sync (must not be absolute or contain ..): $rel" >&2
        exit 1
        ;;
    esac
    LOCAL_EXTRA="${CODE_LOCAL%/}/${rel}"
    if [[ ! -e "$LOCAL_EXTRA" ]]; then
      echo "Path listed in .cluster-extra-sync doesn't exist locally (this manifest lists things the node needs, so a missing entry aborts rather than being silently skipped): $rel" >&2
      exit 1
    fi
    REMOTE_EXTRA="${CODE_REMOTE%/}/${rel}"
    echo "    - $rel"
    remote_bash_kv "$NODE" REMOTE_PARENT "$(dirname "$REMOTE_EXTRA")" 'mkdir -p "$REMOTE_PARENT"'
    if [[ -d "$LOCAL_EXTRA" ]]; then
      rsync -avz --protect-args -- "${LOCAL_EXTRA%/}/" "${NODE}:${REMOTE_EXTRA%/}/"
    else
      rsync -avz --protect-args -- "$LOCAL_EXTRA" "${NODE}:${REMOTE_EXTRA}"
    fi
  done <<< "$EXTRA_PATHS"
fi

# ---- 6. Dataset sync (unrelated to code; binary data doesn't suit git, keeps using rsync) ----
if [[ -n "$DATASET" ]]; then
  case "$DATASET" in
    */*|.|..|"")
      echo "Invalid dataset name: '$DATASET' (must not contain /, and must not be . or ..)" >&2
      exit 1
      ;;
  esac

  : "${DATA_LOCAL_BASE:?DATA_LOCAL_BASE not set}"
  : "${DATA_REMOTE_BASE:?DATA_REMOTE_BASE not set}"
  DATA_LOCAL="${DATA_LOCAL_BASE%/}/${DATASET}"
  DATA_REMOTE="${DATA_REMOTE_BASE%/}/${DATASET}"

  if [[ ! -d "$DATA_LOCAL" ]]; then
    echo "Local dataset directory doesn't exist: $DATA_LOCAL" >&2
    exit 1
  fi

  DATA_EXISTS=1
  if [[ "$FORCE_DATA" -eq 0 ]]; then
    if remote_bash_kv "$NODE" REMOTE_DIR "$DATA_REMOTE" '[ -d "$REMOTE_DIR" ] && [ -n "$(ls -A "$REMOTE_DIR" 2>/dev/null)" ]'; then
      DATA_EXISTS=0
    else
      rc=$?
      if [[ "$rc" -gt 1 ]]; then
        echo "SSH failed while checking the dataset directory on the node (exit code $rc), aborting" >&2
        exit "$rc"
      fi
    fi
  fi

  if [[ "$DATA_EXISTS" -eq 0 ]]; then
    echo "==> Dataset '$DATASET' already exists on the node ($NODE:$DATA_REMOTE), skipping sync (use --force-data to force a re-transfer)"
  else
    echo "==> Creating data directory on $NODE: $DATA_REMOTE"
    remote_bash_kv "$NODE" TARGET_DIR "$DATA_REMOTE" 'mkdir -p "$TARGET_DIR"'
    echo "==> Syncing dataset '$DATASET': $DATA_LOCAL/ -> $NODE:$DATA_REMOTE/"
    rsync -avz --protect-args --progress -- "${DATA_LOCAL%/}/" "${NODE}:${DATA_REMOTE%/}/"
  fi

  # Prove the dataset is "actually completely there" rather than trusting a
  # rough "directory is non-empty" check — datasets are usually several GB of
  # binary files, too slow to checksum file-by-file, so comparing file count
  # and total byte count is enough to catch an obviously-incomplete transfer
  # ("it died halfway through"). Using `find -printf '%s'` (exact per-file byte
  # sum), not `du -sb` — du reports disk-block-rounded usage, which can differ
  # between local and remote filesystems even for byte-identical content,
  # producing a false "incomplete" verdict for something that's actually fine.
  LOCAL_COUNT="$(find "$DATA_LOCAL" -type f | wc -l | tr -d ' ')"
  LOCAL_BYTES="$(find "$DATA_LOCAL" -type f -printf '%s\n' | awk '{sum+=$1} END{print sum+0}')"
  REMOTE_STATS="$(remote_bash_kv "$NODE" REMOTE_DIR "$DATA_REMOTE" 'find "$REMOTE_DIR" -type f | wc -l
find "$REMOTE_DIR" -type f -printf "%s\n" | awk "{sum+=\$1} END{print sum+0}"')"
  REMOTE_COUNT="$(printf '%s\n' "$REMOTE_STATS" | sed -n 1p | tr -d ' ')"
  REMOTE_BYTES="$(printf '%s\n' "$REMOTE_STATS" | sed -n 2p)"
  echo "==> Dataset '$DATASET' comparison: local ${LOCAL_COUNT} files / ${LOCAL_BYTES} bytes, node ${REMOTE_COUNT} files / ${REMOTE_BYTES} bytes"
  if [[ "$LOCAL_COUNT" != "$REMOTE_COUNT" || "$LOCAL_BYTES" != "$REMOTE_BYTES" ]]; then
    echo "Warning: file count or total byte count doesn't match — the dataset may be incomplete on the node, consider --force-data to re-transfer" >&2
    exit 1
  else
    echo "==> Verified: file count and total byte count match (note: this is a coarse check, not a byte-for-byte checksum —" \
         "the chance of two different datasets coincidentally sharing both file count and total size is vanishingly small but not zero)"
  fi
else
  echo "==> No dataset name given, skipping data sync"
fi

echo "==> All done"
