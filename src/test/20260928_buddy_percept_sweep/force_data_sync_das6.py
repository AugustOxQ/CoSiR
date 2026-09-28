"""Force-sync the buddy-percept sweep's ArtELingo data to a DAS6 node,
bypassing cluster.py's `check --sync-data` existence pre-check, which was
found (2026-09-28) to disagree with what an actual srun job sees for
brand-new /local/wding/... node-side paths -- a bare-SSH check (what
`check` uses) reported these paths as already present while a real job
(what `launch` runs inside) found them genuinely missing. Root cause not
diagnosed; the user independently confirmed the data exists somewhere on
the cluster but couldn't explain the discrepancy either. Do not assume
`cluster check --sync-data` is reliable for a NEW node-side path prefix
until this is understood -- always verify with a real in-job check
(pattern: launch a `bash scripts/run_*.sh` job that does
`python3 -c "import os; print(os.path.exists(...))"` and inspect the log,
same as this file's sibling investigation did).

Reuses cluster.py's own plan_data_sync/run_data_sync/parse_data_map
unmodified, as a library -- no modification to that tool's code.

IMPORTANT: run with system python3 (`/usr/bin/python3`), NOT inside the
CoSiR conda env -- conda's bundled OpenSSL conflicts with the system ssh
binary cluster.py's remote() shells out to (confirmed: "OpenSSL version
mismatch" when run under `conda activate CoSiR`).

Usage: /usr/bin/python3 force_data_sync_das6.py <node403|node404|node405>
"""
import sys

sys.path.insert(0, "/root/.claude/skills/cluster-run")
import cluster  # type: ignore

NODE = sys.argv[1] if len(sys.argv) > 1 else "node403"

cfg = cluster.load_config("/root/.claude/skills/cluster-run/cluster.conf")

# Resolved node-side paths from configs/dataset/artelingo_cluster.yaml.
# If a future trial fails with a new "does not exist" path, add it here,
# re-run for the affected node(s), and re-launch -- this is how the
# original 3 missing files (heldout split, genre map) were each found.
missing = {
    "train_annotation_path": "/local/wding/Dataset/artelingo/artelingo_train.json",
    "test_annotation_path": "/local/wding/Dataset/artelingo/artelingo_val.json",
    "storage_dir": "/local/wding/pre_extract/artelingo/features",
    "patch_feature_dir": "/local/wding/pre_extract/artelingo_percept_patch_features",
    "heldout_json": "/local/wding/Dataset/artelingo/artelingo_val_test.json",
    "heldout_storage_dir": "/local/wding/pre_extract/artelingo_heldout/features",
    "genre_json": "/local/wding/Dataset/artelingo/artelingo_genre_emotion_eng.json",
}
# Not included: train_image_path/test_image_path (DATA_OPTIONAL_KEYS, not
# needed by the sweep -- it reads cached features/patches, never raw images).

actions, unresolved = cluster.plan_data_sync(cfg, missing, missing)
if unresolved:
    print("UNRESOLVED:", unresolved)
    sys.exit(1)

print(f"Planned {len(actions)} action(s) for {NODE}:")
for action in actions:
    print(f"  {action['key']}: {action['kind']}, {action['bytes'] / 1024**3:.2f} GB, "
          f"{action['local']} -> {action['remote']}")

total_gb = sum(a["bytes"] for a in actions) / 1024**3
max_gb = float(cfg["DATA_MAX_GB"])
if total_gb > max_gb:
    print(f"REFUSING: total {total_gb:.1f} GB exceeds DATA_MAX_GB={max_gb:g}")
    sys.exit(1)

print(f"Total: {total_gb:.2f} GB (cap {max_gb:g} GB). Starting rsync...")
done = cluster.run_data_sync(cfg, NODE, actions)
print("Done:", done)
