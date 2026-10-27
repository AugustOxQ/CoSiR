# Ticket 12 build notes (rs_vp.py)

Status: code and tests done (42 tests, green) on branch r10-rs-12, merged with r10-recipe-sweep (tip b6ad07b). The verdict
layouts are now adapted to the example files the scorers wrote. Not done: the sections under OPEN.

Independence (C13): rs_sweep.py, rs_stage3.py, rs_ablation.py and their tests were never opened, imported or grepped, by
either worker, before or after the merge. The layouts below come from example JSON files only
(/tmp/claude-0/r10-fixtures/examples/, /tmp/claude-0/r10-rs-10/outputs/ablation/dev.json).

## Built
`rs_vp.py` modes: `partitions --machine M [--runs ...]`, `sweep --machine M`, `stage3 --sweep-machine M [--draws N]`,
`ablation --final NAME [--draws N]` (`--draws -1` = the dev.json's own count), `self-check [--draws N] [--all-arms]`.
`--outputs` everywhere. Each writes `rederive_<check>.json` (pass, items with value, expected, diff, SHA-256 of every
input read) beside the file it checks: scores/M/rederive_partitions.json, sweep/M/rederive_stage{1,2}.json,
stage3/rederive_stage3.json, ablation/rederive_dev.json, <outputs>/rederive_selfcheck.json. Exit 1 on any disagreement.

## Definitions settled (one line of why each)
- r_a, tolerances, guard, grid and tie rules are constants in rs_vp.py with the rule section in a comment (C13).
- Positions for the TOP tie-break are read off rs_sheet.stage1() order (the sheet's listed order), not rs_sheet.positions.
- Part (b) derives twice (scorer's A, my A from rederive_partitions.json) and requires the same decisions; the stage
  file is compared with the scorer-A derivation. Lists the rule fixes no order for (grid knobs, levels, grid points,
  pool, failed, qualifiers, not_qualifying) compare as sets; TOP is ordered.
- Stage 1 file (example stage1.json, stage1_sweep_none.json): `runs` holds the 39 stage 1 rows and not G (G is
  `reference`); `failed` = [{name, cause}]; under SWEEP-NONE `grid.points` is empty (no G0 point). Mine now derives
  no grid points without a grid knob, compares rows of the 39 names only, and compares the failed names as a set.
  Stage 2 file (example stage2.json): `new_runs`, `failed`, `pool`, `top` (rows with `name`), outcome 'TOP'. All fit.
  A run named G-repeat (the node's G0') is ignored in the comparison of rows.
- Stage 3 takes TOP from my own stage 2 derivation (my A values), so `stage3` needs `--sweep-machine`; the verdict's
  `top` must equal it in order, equal the stage2.json's `top`, `stage2_machine` equal the flag, and
  `inputs.stage2_sha256` equal the SHA-256 of the stage2.json I read.
- Stage 3 verdict (examples stage3_keepg*.json): compared are ranges, seeds, draw_mode ('joint'), n_draws (must equal
  mine, so run `stage3` with the verdict's draws, default 2000), reference.name 'G', top, outcome, pick, pick_config
  (= rs_sheet.parse(pick), JSON round trip), qualifiers / not_qualifying / failed as sets, the names under `results`,
  and per configuration: status, cause (both prose or both none), qualifies and, for an ok one, Abar, e (e, lo, hi),
  M (M, lo, hi). Not compared: `written`, `lead`, `conditions`, `smoke`, `host`, code fields, `p_sha256`.
- OK-configuration layout is a GUESS: no example holds one (both KEEP-G examples are failed configurations).
  `norm_config` reads `effects[a] = {e, lo, hi}` with `Abar` (a number or {config, G}) and `A` ({config: {seed}}) beside
  (12-27's stage B layout), and a flat `Abar` {aspect: value}; `M` as {M, lo, hi} or a number. A mismatch of layout shows
  as `<missing>` items, so it fails loudly; `norm_config` is the one place to adapt.
- Stage 3's partitions are read from `<outputs>/stage3/cache/<name>/s<seed>/{aspect}__partitions.npz` (key `labels`,
  int64 [3, n]); required for every ok configuration. G's: 12-27's RG_STAGE_B partitions, and the cache's when present.
  The scores/ score files are no longer used for stages 3 and the ablation (the scorer writes none for them).
- Joint bootstrap: draws are identical for every configuration (own default_rng(0)), so all arms are scored in one pass.
- Ablation (example dev.json): results are keyed by removal x, arm = the folder name `rs_sheet.name(config(F, remove=(x,)))`.
  Per ok arm: effect = Abar(arm) - Abar(F) (checked: equals the file's), lo, hi (percentiles of the joint draws, with
  --draws), arm_ami, final_ami, arm_seed_ami, final_seed_ami (per-seed A, string keys), M {M, lo, hi}. A failed arm: status,
  cause (prose), aspects and M null. Also compared: final.name, final_key, ranges, seeds, draw_mode, n_draws (when I draw),
  removals and results names (sets), failed (set of removal:arm), inputs.final_runs / arm_runs (run.json SHA-256 and the
  export SHA-256 per aspect and seed), p_sha256, and final.outcome / pick against stage3/verdict.json. The removals
  compared are the file's `removals`; a non-smoke file must list the rule's eight (refused otherwise). Partitions under
  `<outputs>/ablation/cache/<arm>/s<seed>/` are compared for every ok arm (required) and for F's if present.
- Ablation with PICK and a smoke file works on the fixture; with G as F the F runs are 12-27's frozen GS-0.02 runs.
- code_state returns ('no-git', None) when git is missing (the node copy may have no .git).

## Self-check (points only, run 2026-10-27, before the merge; unchanged code path)
`python rs_vp.py self-check --outputs /tmp/claude-0/r10-rs-12`: 57 items, 0 differ (stage A GS-0.02 and R6-5 seed 0 partitions and
A; stage B GS-0.02 vs R6-5 seeds 1-3 partitions, A-bar, per-seed A and e-bar), see rederive_selfcheck.json there.
Not rerun after the merge (the self-check code is untouched; the 42 tests ran).

## Checks against the example files (2026-10-27)
- stage1.json (GRID-2), stage1_sweep_none.json (SWEEP-NONE), stage2.json (TOP): `derive_stage1` / `derive_stage2` on the
  files' own A values, then `compare_stage1` / `compare_stage2`: 424, 412 and 20 items, 0 differ (after the fixes above).
  Before them: G was expected as a row, and SWEEP-NONE gave a G0 grid point.
- ablation/dev.json of ticket 10's smoke (fixture /tmp/claude-0/r10-rs-10/outputs, F = k6=0.0005 under PICK, 20 draws,
  arms 'direct' ok and 'buddy' failed): `rs_vp.py ablation --final k6=0.0005 --outputs /tmp/claude-0/r10-rs-10/outputs
  --draws -1` (3 min 14 s, 8 threads) gave pass=True, 99 items, 0 differ, largest numeric diff 0.0: effects, intervals
  (20 draws), M, per-seed AMI, the failed arm, run.json and export SHA-256s, and the scorer's partitions all equal.
  Written to /tmp/claude-0/r10-rs-10/outputs/ablation/rederive_dev.json (beside dev.json). A first run found one real
  bug of mine (`sets_equal` on a dict), fixed and covered by the tests.

## OPEN
- `self-check --draws 2000` (about 10 min on 8 workers; does not fit 4 minutes): expect per-aspect lo/hi for all three
  aspects and joint lo/hi for emotion and style equal to stage B's, within 1e-6.
- `stage3` against a verdict with an ok configuration (layout guessed, see above); `partitions` against a real outputs
  tree (stage 1 first); `ablation` with real draws (2000) is long: nine arms, run in the background.
- `lead` and `conditions` of the stage 3 verdict are not compared (free form so far).
