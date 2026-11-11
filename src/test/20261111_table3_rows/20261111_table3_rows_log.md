# 20261111 table3_rows (r27): scorer for two rows of the paper's retrieval table

Written 2026-11-11 (Amsterdam). Status: code and unit tests done (11 pass, 59 s). No gate and no full scoring run yet:
the controller runs them through the CPU queue. Nothing outside this folder was edited; nothing committed.

## What the code does
Image-to-image label mAP on the ArtELingo test split (the code's `held`; 12,281 paintings), each method at the read-out it
picks on the validation split (6,152 paintings). One read-out for the three aspects, by r11's cross-fit rule.
- `t3_core.py`: generic `pick_readout(get_val, labels, unit, seeds)` and `score_test(get_val, get_test, pick, labels, seeds)`;
  units, labels, keys, hashing, JSON writer, `check_validation_keys` (sorted names, count, no key shared with test).
- `t3_inputs.py`: `Vlm2VecStore` (reads `<root>/<aspect>/{emb.npy,keys.json,meta.json}`, checks meta `n` and `emb_sha256`
  against the file, unit rows, reorders by name to the project's keys after asserting the same key set) and `PerceptFiles`
  (npz `prob`, `names`; rows = `ns_ret._unit(prob)`; names checked/reordered to the project's keys).
- `t3_run.py`: subcommands `gate-tpips`, `gate-percept`, `vlm2vec`, `percept-clip`; each takes `--workers N` (default 8, max 8).
- `test_t3_core.py`: 11 synthetic tests (seeded pick equals `arm_table` on 5 seeds; unseeded equals seeded with 1 seed; subset
  pick is the mean over kept seeds; per-seed scoring equals `arm_ap`; alignment; store checks; JSON NaN refusal).

## Commands (cwd /project/CoSiR/src/test/20261111_table3_rows; prefix `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /root/miniconda3/envs/CoSiR/bin/python`)
    t3_run.py gate-tpips                      -> out/gate_tpips.json
    t3_run.py gate-percept                    -> out/gate_percept.json
    t3_run.py vlm2vec [--test-root R --val-root R]          (defaults: pre_extract/artelingo/vlm2vec_v2 and vlm2vec_v2_val) -> out/vlm2vec.json
    t3_run.py percept-clip --seeds 0,1,3 --test-dir <r26 test dir> [--val-dir <r26 val/k40>] [--file-pattern 'L1__s{seed}.npz'] [--test-file-pattern ...] -> out/percept_clip.json
Unit tests: `python -m pytest -q -p no:cacheprovider src/test/20261111_table3_rows` (from /project/CoSiR; about 1 min).

Gates pass at max abs difference <= 1e-9 and the pick equal to the record. Rough durations (no timing record for the exact path):
gate-tpips 4 to 8 min (r18's TPIPS pick alone took 120 s; plus 6 TPIPS arrays of 12k/6k x 4096 and 3 test scorings);
gate-percept 3 to 8 min (5 seeds x 15 read-outs on 40-wide rows, then 5 seeds x 3 aspects on 12,281 queries);
vlm2vec and percept-clip about the same as the matching gate.

## Gate records compared
- gate-tpips: pick == `none+k10` == `records/stage1_om_picks.json['tpips/artelingo']['pick']`; test mAP == `records/labels.json`
  cells `SC2/artelingo/qwen/<aspect>/map/-/tpips`, field `comp` (0.25649626..., 0.23649215..., 0.56266194...); validation mAP at
  the pick == that record's `scores.map`.
- gate-percept: pick == r17 `records/stage1_picks.json['matched/artelingo/clip_l14']['pick']` (center+k10); test mAP == r17
  `records/verdicts.json` cells `SC8/artelingo/clip_l14/<aspect>/map/-/matched`, field `comp` (0.24393653..., 0.16582680..., 0.53157628...);
  validation mAP at the pick == the record's `scores.map`. Rows: `ns_ret._percept_rows` (r17's own probabilities), seeds 0..4.

## Reuse map
1. Pick, unseeded: `om_stage1.readout_doc` -> `ac_select.arm_table` (unit `om_stage1.units('tpips', 'artelingo')` = Qwen c* ranges via `ns_picks.ranges`).
2. Pick, seeded: `rt_readout.crossfit_all` per seed (it takes seeds None or 0..4 only), mean over kept seeds, then `ac_select.readout_scores / readout_guard / pick_readout` (arm_table's own last step); unit `ns_picks.ranges('artelingo', 'clip_l14')`, the path of r17's `matched/artelingo/clip_l14` pick (`ns_picks.pick` -> `ranges(set_, enc)`).
3. Test scoring: `ns_ret.arm_ap(get_val, get_test, pick, y, seeds=None)` as `om_cells.Loaders.ap` does for TPIPS; the seed subset is scored per seed and averaged (arm_ap takes only None or 0..4).
4. Rows and labels: `om_data.keys/labels/tpips`, `ns_ret._percept_rows`/`_unit`, `ns_picks.check_order`; `om_paths` imported first.
5. Not reused: r25's `pb_*` has no store loader (only the writer), so `t3_inputs.Vlm2VecStore` is new.

## Output JSON (vlm2vec.json, percept_clip.json)
pick, readout (w, size, k), validation_table (all 15 read-outs: map per aspect, score, kept), per_seed_validation_map (seeded
only), test_map, test_map_per_seed, n_queries (per aspect), dropped counts, seeds, unit and unit_source, inputs (path and sha256
of emb.npy, keys.json, meta.json per aspect; or of each npz per seed, validation and test), git_head, arm_note.
The unit for VLM2Vec is stated as Qwen c*'s validation mAP ranges, as TPIPS (field `unit_source`).

## Open questions
- Unit for the PercepT row: CLIP L/14 c*'s ranges, copied from r17's matched pick. If the paper row should instead use Qwen
  c*'s ranges (one unit for all methods in the table), the unit is one line in `t3_run.run_percept_clip`.
- The seeded pick with a seed subset is new code (no earlier use of a subset); with all five seeds the unit test shows it equals `arm_table`.
- r26's collapse record and test file path are unknown: `--test-dir` and `--seeds` are required; the pattern is assumed to be the same as validation.
- Not run: both gates, both scoring runs. gate-percept needs r17's matched probability files (held, seeds 0..4) on disk (`ns_data.percept_probs` checks them).
- Nothing to remove. Empty folders `out/` and `logs/` hold only `logs/pytest.log`.

## Gates, run by the controller (2026-11-11 06:41, `res-queue cpu --short`)

- `gate-tpips`: pass. Pick `none+k10` as recorded; test and validation mAP equal to the records (max abs difference
  0, 6 values); 3.9 min. Log `logs/gate_tpips.log`, record `out/gate_tpips.json`.
- `gate-percept`: pass. Pick `center+k10` as recorded; max abs difference 2.8e-17 (6 values); 5.0 min. Log
  `logs/gate_percept.log`, record `out/gate_percept.json`.

## VLM2Vec row, scored (2026-11-11 09:16 to 09:18, "paper sec4 tables 5" tab)

- `t3_wait_vlm2vec.sh` saw the six store meta files at 09:16 (r25's `r25:extract_val` ended at 09:16, exit 0) and ran
  `t3_run.py vlm2vec` through `res-queue cpu --short` (1.9 min, exit 0). Log `logs/vlm2vec.log`.
- Pick on validation: `center+k10` (validation mAP 0.2523 / 0.2840 / 0.5752). Test mAP: emotion 0.24795, style
  0.28019, genre 0.55895 (queries 12,281 / 12,281 / 9,949; none dropped). Record `records/vlm2vec.json` (a copy of
  `out/vlm2vec.json`, added with `git add -f`).
- Table 3 (REPO `33a9764`): 0.248 / 0.280 / 0.559. Below CoSiR in all three aspects, so the bold is unchanged.
