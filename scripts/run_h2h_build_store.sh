#! /bin/bash
set -euo pipefail

# Build (or load, if another process already built it) the matched
# head-to-head fixed-input store on this DAS6 node, then print what it holds.
# Usage (from a DAS6 node's reserved srun shell, via `cluster launch`):
#   bash scripts/run_h2h_build_store.sh

# cluster-run's job.sh already sources conda and activates the CoSiR env
# before invoking this script -- do not re-source it here.

# This script only ever runs on a DAS6 node, where the ArtELingo data lives
# under /local/wding/..., not this container's /data/PDD, /data/SSD2 paths;
# the store builder reads these two env vars (defaulting to the local-only paths).
export PERCEPT_FEATURE_ROOT=/local/wding/pre_extract
export PERCEPT_RAW_JSON_ROOT=/local/wding/Dataset/artelingo
export CUBLAS_WORKSPACE_CONFIG=:4096:8

exec python - <<'EOF'
import sys
sys.path.insert(0, ".")
from scripts.buddy_percept_sweep.h2h_split import make_split
from scripts.buddy_percept_sweep.h2h_store import default_cache_dir, load_or_build_store

store = load_or_build_store()
print(f"STORE_DIR {default_cache_dir()}", flush=True)
for name, value in vars(store).items():
    print(f"STORE_FIELD {name} shape={tuple(value.shape)} dtype={value.dtype}", flush=True)
split = make_split(store.heldout_emotion, store.heldout_genre)
print(f"STORE_SPLIT val={len(split.val_idx)} test={len(split.test_idx)} digest={split.digest}", flush=True)
EOF
