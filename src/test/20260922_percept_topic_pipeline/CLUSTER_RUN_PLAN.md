# PercepT branch — cluster-run plan (drafted overnight, needs your review + confirmation)

Written while working locally overnight, per your instruction to have a
clear plan ready for when you wake up. **Nothing in this file has been
executed** — `cluster_launch.sh`/`cluster_sync_up.sh` are blocked from this
session by a hard runtime floor ("remote copy/sync requires human
approval") that fired even on a read-only `ls` mentioning those filenames,
so I could not even run the safe `--detect-only` node check. Everything
below is preparation only.

## 1. This branch doesn't fit the existing cluster-run plumbing cleanly — three real gaps

The `cluster-run` skill (`.claude/skills/cluster-run/SKILL.md`) is built
around `main_cosir.py` + Hydra dataset configs with a `_cluster.yaml`
variant per dataset, syncing exactly one feature-cache directory per
dataset via `DATA_LOCAL_BASE`/`DATA_REMOTE_BASE`. This whole PercepT branch
is a different shape of work: standalone scripts under
`src/test/20260922_percept_topic_pipeline/`, not `main_cosir.py`, with data
dependencies the skill was never wired up for:

- **No `configs/dataset/artelingo_cluster.yaml` exists.** Only
  `configs/dataset/artelingo.yaml` (local paths) is present. Every other
  dataset (coco, cc3m, redcaps, redcaps2, redcaps_150k) has a `_cluster.yaml`
  sibling; ArtELingo never got one, presumably because until this branch
  nothing needed ArtELingo on the cluster.
- **Two feature caches, both outside git, need to land on the node:**
  - `/data/SSD2/pre_extract/artelingo/features` (train CLIP cache) — **1.2 GB**
  - `/data/SSD2/pre_extract/artelingo_heldout/features` (held-out CLIP cache) — **184 MB**
  - `/data/SSD2/pre_extract/artelingo_percept_patch_features/` (new patch-token
    cache this session built) — **6.8 GB** (`train_patch_features.pt` +
    `heldout_patch_features.pt`)
  - Total: **~8.2 GB**. None of this is `.cluster-extra-sync`-eligible —
    that mechanism only rsyncs paths *relative to the repo root* (for
    small, gitignored, in-repo files like local credentials); these caches
    live entirely outside the repo tree, which is exactly what
    `DATA_LOCAL_BASE`/`DATA_REMOTE_BASE` (the dataset-config-driven
    mechanism) is for instead.
  - **Raw WikiArt images (`/data/PDD/wikiart_proj/wikiart/`, tens of GB) are
    NOT needed on the node** — they were only needed for the one-time patch
    extraction step, which is already done and cached locally. As long as
    the patch cache above is synced, the node never needs to touch raw
    images.
- **This session's PercepT work is entirely uncommitted.** `git status`
  shows `src/test/20260922_percept_topic_pipeline/` (all briefs, pilot
  scripts, and reports from tonight) plus
  `docs/reports/2026-09-23_artelingo_percept_stage1_report.md` as untracked.
  `cluster_sync_up.sh` refuses to run on a dirty tree — this needs a commit
  before any push, and per the skill's deliberate safety marker, either the
  branch name or the commit message must contain "cluster run"
  (case-insensitive) or `cluster_sync_up.sh` refuses to proceed at all.
  **I have not committed anything — only commit when you explicitly ask.**
  Two older untracked files from an unrelated earlier session are also
  sitting in the tree (`affect_pilot_communities.npz`, `buddy_graph_E.npz`
  under `20260923_artelingo_buddy_analysis/`) — not part of tonight's work,
  flagging in case they matter to you.

## 2. Recommended one-time setup (your call, not yet done)

1. Commit tonight's PercepT work with "cluster run" in the message (or rename
   to a branch containing it) — this repo's own convention.
2. Add `configs/dataset/artelingo_cluster.yaml`, following the exact
   `redcaps_cluster.yaml` template (`@package _global_`, node paths under
   `/var/scratch/wding/Dataset/...` for raw annotations/images and
   `/local/wding/Dataset/pre_extract/artelingo/features` for the
   FeatureManager cache) — draft below, not yet added to `configs/`:

   ```yaml
   # @package _global_
   # ArtELingo Dataset Configuration (cluster)
   data:
     dataset_type: "artelingo"
     train_annotation_path: "/var/scratch/wding/Dataset/artelingo/artelingo_train.json"
     test_annotation_path: "/var/scratch/wding/Dataset/artelingo/artelingo_val.json"
     train_image_path: "/var/scratch/wding/Dataset/wikiart_proj/wikiart"
     test_image_path: "/var/scratch/wding/Dataset/wikiart_proj/wikiart"
   featuremanager:
     storage_dir: "/local/wding/Dataset/pre_extract/artelingo/features"
   experiment:
     name: "CoSiR_Experiment"
     tags: ["CoSiR", "ArtELingo", "Cluster"]
     results_dir: "/local/wding/res/CoSiR_Experiment/artelingo"
   ```

   This only covers the standard CLIP feature cache, matching the existing
   per-dataset convention. The new patch-feature cache
   (`artelingo_percept_patch_features`) isn't a FeatureManager-managed
   dataset at all — it's this branch's own ad hoc cache, so it would need
   either a second dataset-style entry or a one-off rsync call added
   specifically for this branch's pilots. Since none of the pilot scripts
   read config through Hydra (they hardcode
   `/data/SSD2/pre_extract/...` paths directly), the actual node path this
   cache lands at needs to match what `run_percept_stage2_pilot.py` and
   friends expect — either update those scripts' hardcoded paths to the
   node convention, or place the synced cache at an identical absolute path
   on the node (simplest, least error-prone: keep the constant paths
   unchanged and land the cache at the same absolute path on the node,
   since the node's `/data/SSD2/` may or may not exist/be shared storage —
   worth confirming with you rather than assuming).
3. Confirm whether the node has `/data/SSD2/` available at all (this
   session doesn't know DAS6's filesystem layout for that mount) — if not,
   every hardcoded `/data/SSD2/pre_extract/...` path in these pilot scripts
   needs a node-side override, which is more surgery than a config addition.

## 3. What's actually worth running on two extra GPUs, once unblocked

Both are genuine, well-motivated follow-ups to tonight's results, not
padding — pick either, both, or neither:

**Node A — Stage 1 statistical strength.** K=60/40 cleared the held-out
Pareto bar in 4/4 seeds tonight, but 4 seeds is a small sample and the
emotion margin is thin (worst seed cleared by only 0.0002). Extending to
10-15 seeds would turn "4/4" into a real confidence interval instead of a
small-sample anecdote, and a finer K sweep around 55-65 (only 40/27, 60/40,
80/53 were tested intermediately) could sharpen exactly where the
crossover sits.

**Node B — Stage 2 statistical strength + architecture ablation.** The
Stage 2 sweep launched tonight (still running as this plan is being
written) tests 4 mapper-init seeds and a small learning-rate/threshold
grid. Worth extending to more seeds for the same reason as Node A, and/or
testing a genuinely richer mapper (multi-head attention, or a small 2-layer
MLP head) now that the basic single-query-attention design is validated —
the paper's own finding that simpler beats fancier was for THEIR dataset,
not verified yet for this one.

## 4. Exact commands ready to fire once you confirm (not run yet)

```bash
# Step 2 — detect both reserved nodes (read-only, safe)
./cluster_launch.sh --detect-only

# Step 3 — push code (after committing with a "cluster run" message)
./cluster_sync_up.sh ./cluster_sync.conf --node node4XX
./cluster_sync_up.sh ./cluster_sync.conf --node node4YY

# Step 3b — push data, once artelingo_cluster.yaml exists and the patch-cache
# sync question above is resolved
./cluster_sync_up.sh ./cluster_sync.conf artelingo --node node4XX
./cluster_sync_up.sh ./cluster_sync.conf artelingo --node node4YY

# Step 4 — launch (needs your live confirmation each time, by design)
./cluster_launch.sh "python src/test/20260922_percept_topic_pipeline/<script>.py" --node node4XX
./cluster_launch.sh "python src/test/20260922_percept_topic_pipeline/<script>.py" --node node4YY

# Step 5 — pull back once you say each run is done
./cluster_sync_down.sh --node node4XX
./cluster_sync_down.sh --node node4YY
```

Note step 5 pulls `RESULTS_REMOTE` (`/local/wding/res/`) — these pilots
write their `.md` reports inside the code tree, not `res/`, so a plain
`cluster_sync_down.sh` won't retrieve them. Simplest fix: commit + push the
report from the node (git, same as local work), or `scp`/rsync the specific
report file back explicitly.
