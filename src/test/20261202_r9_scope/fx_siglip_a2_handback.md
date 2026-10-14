# Side branch r9-siglip-a2: k-means restart rule (NOT merged, no push)

Worktree `/project/CoSiR-wt/r9-siglip-a2`, branch `r9-siglip-a2`. This branch edits the closure module
`r9_recipe_groups.py` on purpose (the one exception to fx_common). It is for the user to accept or drop.

## The rule (`F/r9_recipe_groups.py`)
- `MIN_GROUP_ROWS = 100`, `MAX_RESTARTS = 10`. `fit_partitions(..., min_restart_rows=MIN_GROUP_ROWS, restarts=None)`.
- After the usual checks on each k-means (image, caption), if the smallest group has fewer than 100 rows: refit with
  seed+1, seed+2, ... (each refit re-checked), stop at the first fit with smallest >= 100. After 10 refits: AssertionError
  naming the view and all sizes tried. Caption refit = `kmeans_partition`; image refit = `kmeans_partition` on the image
  features (test shows it is the same fit as ClipClusterSource's image view, same seed).
- The existing label asserts (no -1, exactly 64 groups) run first, so they keep their meaning.
- When the rule does not fire: the same calls run in the same order; output arrays are byte-identical (tested; the proof
  below). `partitions.npz.record.json` gets a `kmeans_restarts` field only if it fired:
  `{view: {min_group_rows, tries: [{seed, smallest}, ...], seed_used}}`. Records of CLIP and Qwen differ from today's only
  by `script_sha256` and `code` (the file bytes changed).

## Tests (`F/test_r9_recipe_groups.py`)
Command: `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/CoSiR/bin/python -m pytest -q -p no:cacheprovider src/test/20261202_r9_scope/test_r9_recipe_groups.py`
Result: 48 passed (39 old, 9 new test functions, 10 new cases). New: `test_restart_constants`,
`test_rule_does_not_trigger_on_a_normal_fit_and_changes_nothing` (bit-equal to the rule switched off, no log),
`test_rule_triggers_on_a_dead_centre_and_refits_with_the_next_seed` (caption; seeds [42, 43]),
`test_rule_triggers_on_image_too`, `test_source_image_view_equals_kmeans_partition_of_the_same_seed`,
`test_rule_on_a_real_small_fit_logs_the_sizes` (real k-means, blurred data, both views fire), `test_rule_gives_up_after_ten_restarts`
(seeds 42..52), `test_rule_boundary_is_strictly_fewer_than_min_rows` (49 fires, 50 does not),
`test_record_field_only_when_the_rule_fires`, `test_record_field_when_the_rule_fires`.
Mutations, each caught and the file restored (git diff checked): rule never fires; `>` for `>=` (first missed it, boundary
test added, then caught); seed skips one; 11 restarts; threshold 50; record field always written; first try missing from the log.

## CLIP and Qwen stay bit-equal (item 2)
Both stored partitions have no group under 100 (CLIP: image 1605, caption 635, affect 244; Qwen: image 1549, caption
1510, affect 244), so the rule cannot fire. I also refitted both on CPU with the new code
(`F/fx_siglip_a2_prove.py`, output `F/fx_siglip_a2_prove_output.txt`; 12 s and 30 s): all four arrays of each are
byte- and dtype-equal to the stored `partitions.npz` (CLIP: `src/test/20261031_pseudo_partitions/results/partitions.npz`,
Qwen: `F/results/recipe/qwen3vl_emb_2b/partitions.npz`), and `restarts` is empty.

## SigLIP with the rule (item 3, CPU, 24 s)
Caption: seed 42 gives smallest 1; the rule lands on **seed 43**, smallest **939**. Caption sizes, five smallest: 939, 994,
1139, 1313, 1314; max 6249. Image: no restart, seed 42, byte-equal to the stored image labels (smallest 773, 1362, 1426,
1609, 1679; max 5400). affect and local_groups unchanged. Nothing under `F/results/` was touched.

## For the controller (item 4; times from the earlier logs, not re-measured)
Merging changes the SHA of `r9_recipe_groups.py` (stage 0b has `abc5c550...`). Then:
1. **Stage 0b re-compare.** `run_r9_stage0b.py` item 4 requires every replay record's `script_sha256` to equal the
   current script, and `run_r9_stage2.assert_modules` requires `stage0b.json module_sha256` to match. The five replay
   records made by `r9_recipe_groups.py` (partitions, graph, bank_AIC, n6_posteriors, step1_heads_style) carry the old SHA,
   so the CLIP replay step 07 must be redone (GPU, about 12 min, `run_r9_stage0b_replay.sh` step 1; the other steps' outputs
   do not change because CLIP's partitions are bit-equal, but whether to reuse them or redo the whole queue (about 53 min)
   is the controller's choice; a fresh replay folder is needed, the queue refuses an existing one). Then the compare
   (`run_r9_stage0b.py`, 308 s last time). Result: new `results/stage0b.json`.
2. **Qwen stage 2 rerun (N5).** Everything in `F/results/recipe/qwen3vl_emb_2b/` and `results/stage2_*qwen*.json`,
   `pins_qwen3vl_emb_2b.json` carries the old script/module SHA. The last full run took 00:54 to 02:37 (about 1 h 45, groups
   step 9 min). The outputs should come out the same (partitions bit-equal), which is a check to make. Needs the old files
   moved aside (the driver refuses to overwrite); I moved nothing.
3. **SigLIP stage 2.** First move aside (I did not) the 8 files in `F/results/recipe/siglip2_so400m/`: `partitions.npz`,
   `graph.npz`, `bank_AIC.npz`, `build_record.json` and the three `.record.json`... precisely: `partitions.npz`, `graph.npz`,
   `bank_AIC.npz`, `build_record.json`, and `partitions.npz.record.json`, `graph.npz.record.json`, `bank_AIC.npz.record.json`
   (7 files; no heads files were written). The graph does not depend on the partitions but the driver refuses a part-filled
   folder, so all are remade. Estimate about 1 h 45 as for Qwen: groups about 10 min (the failed run reached the heads at
   282 s), then a3, reader, baselines, cells, record, pins. Heads should now pass: smallest caption group 939.
4. **Stale if merged:** `results/stage0b.json` (and `stage0b_ee7a76d.json`), Qwen's recipe records and stage 2 records and
   pins, SigLIP's partial records, `results/t13_clip_dryrun/*` (hold the old SHA), `t22_mutation_results.json` (names the
   file). Not stale: `pins_clip.json`, `pins_a28.json`, CLIP's stored partitions. Other 16 closure modules unchanged, but
   `r9_recipe_a3`, `r9_recipe_baselines`, `r9_recipe_reader` import `r9_recipe_groups`, so their `code` lists change too.

## Choices the brief did not fix
- Threshold applies to the number of rows per group (strictly fewer than 100 fires).
- Restart log is an optional `restarts` dict argument, not a return value, so `fit_partitions` still returns the same 4 keys.
- Refit of the image view uses `kmeans_partition` (equal to the source's image view; tested), so ClipClusterSource is not rebuilt.
- Not tested: how often a restart would be needed on CLIP/Qwen other seeds (not relevant, seed 42 only).
