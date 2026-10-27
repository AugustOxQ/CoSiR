# Ticket 12 build notes (rs_vp.py)

Status: build-10 review fixes implemented on r10-rs-12; 81 tests green. Merged r10-recipe-sweep at 26f556d
(merge f5583a7). Current validation and remaining controller commands are recorded in the build-10 section below.
The older sections retain the build-9 validation history.

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
- Build-9 OK-configuration layout was a guess; the build-10 PICK example confirms it.
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

## Build-9 open items (superseded by build-10 validation below)
- `self-check --draws 2000` (about 10 min on 8 workers; does not fit 4 minutes): expect per-aspect lo/hi for all three
  aspects and joint lo/hi for emotion and style equal to stage B's, within 1e-6.
- `stage3` against a verdict with an ok configuration (layout guessed, see above); `partitions` against a real outputs
  tree (stage 1 first); `ablation` with real draws (2000) is long: nine arms, run in the background.
- `lead` and `conditions` of the stage 3 verdict are not compared (free form so far).


## Build 10: review fixes (2026-10-27)

Scope and independence: only rs_vp.py, test_rs_vp.py, this note and the requested .claude change log changed.
No scorer or scorer test was opened, imported or searched; rs_stage0.py and rs_stage4.py were also kept closed.
No other worktree was read. The supplied brief/review, review scratch case, example JSONs and partition memo were
read as authorized. Single agent, CPU only. Explicit scratch artifacts are under /tmp/claude-0/r10-fix-12/; see the pytest temp-directory correction below.

Definitions and fixes, with their reason:
- F = G uses the scorer's run key GS-0.02 (rs_paths.G_ARM), while the recipe name remains G; real dev.json facts
  replace the old helper's guessed final_key. KEEP-G now agrees.
- Without stage3/verdict.json, independently enumerate every sweep/*/stage1.json and read each available stage2.json.
  Require every participating machine to end in SWEEP-NONE or TOP-NONE, and at least one participating machine;
  otherwise refuse with the machine and missing-stage-3 reason. Both outcomes keep G (§4.2, §2.5).
  A mixed terminal set licenses either observed NONE label; the rule gives neither machine or label priority.
  A stage3-only label (KEEP-G) cannot pass without a stage3 verdict.
- A failed gates/k11_sample.json supplies failed, non-counting rows for k11's three values in partitions, sweep and
  stage3; it does not demand nonexistent runs or stop unrelated configurations (§2.2). Part (a) also tolerates
  absent score files for those unrun values, while checking any failure score files that exist.
- A failed gates/ablation_graphs.json variant supplies a failed removal record: uncorrected => content_correction,
  default_view => instructed_graphs. Other arms continue; an ungated missing arm remains a refusal (§2.5).
- The scorer's G cache is read under GS-0.02 with items under partitions/G_cache; frozen G remains partitions/G.
  Both differences survive. The same applies to the ablation when F = G; a picked F's stage3 cache is also compared
  when present, with a separate item prefix to preserve any ablation-cache difference.
- Missing failed lists, runs/new_runs lists and other compared keys produce disagreements; no KeyError hides the
  rederive report. Failed stage3 configurations require null effects and M.
- V1c(b) compares decisions between the two A sets, rather than requiring M and knob gains to agree to 1e-6 in seed
  range units: permitted A differences can amplify beyond that tolerance (§4.2). The stage files' numbers still
  compare against the scorer-A derivation.
- Stage3 also compares G's per-seed A and inputs.runs run/export hashes, binding its verdict to the raw inputs read.
  lead and conditions remain reported fields, not independent comparison items.
- Unparseable train folder names are skipped and listed in skipped_folders, so smoke leftovers cannot stop V1c(a).
- Default CPU threads are always 8 (RB §3.1), regardless of OMP_NUM_THREADS; partitions retains explicit --threads.
- Regression tests hold the guard item, nonempty-report requirement, CLI exit 1, F against stage3 and TOP against
  stage2. Grid ties now include the fourth-slot cutoff; positive-M values failing the guard cannot become levels.
- Tests rename fixtures instead of unlink/rmtree. Nothing was deleted.

Test-first evidence and commands:
- Post-merge baseline: the prescribed CPU-only pytest command, 42 passed in 32.16 s.
- Correcting dev_like's final_key first failed the existing comparison test. The new no-stage3 case failed with
  FileNotFoundError; the k11 and ablation graph cases failed on the old missing-run assertions. G cache, missing
  failed/row lists, fixed threads, failed-null and stray-folder cases all failed before their fixes.
- M07, M10, M33, M34, M36, M44, M45, M47 and M48 each failed its new targeted regression test when applied in a shared
  clone. The cutoff test's initial typo (k1=4, absent from the sheet) was corrected to k1=2, verified green, and M07
  rerun to establish a real kill. Logs: red_*.log, red-M*.log in scratch.
- Full implementation check: 79 passed in 90.46 s. Exact command (from WT):
  timeout 235 env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES='' \
    /root/miniconda3/envs/CoSiR/bin/python -m pytest -q -p no:cacheprovider \
    --basetemp /tmp/claude-0/r10-fix-12/pytest-full3 src/test/20261027_recipe_sweep/test_rs_vp.py
- Final shared-clone rerun caught all nine surviving mutants: M07, M10, M33 (full test file with -x);
  M34, M36, M44, M45, M47, M48 (their focused regressions). Every run restored rs_vp.py by copying its original
  back; cmp confirmed restoration. After the mixed-terminal fix, all nine were rerun again against the final source
  and all were caught by their focused tests (final_mutants_all.log). No equivalents or unexplained survivors.

Controller command for the full frozen self-check (about 10 min on 8 workers, plus partition time; exceeds this
fixer's 235-second command cap):

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES='' \
  /root/miniconda3/envs/CoSiR/bin/python src/test/20261027_recipe_sweep/rs_vp.py self-check \
  --draws 2000 --workers 8 --outputs /tmp/claude-0/r10-fix-12/selfcheck-full
```

Expected: pass, 0 differing items within 1e-6, including all three per-aspect intervals and emotion/style joint
intervals. The prior points-only self-check remains 57 items, 0 differ, largest difference 0; its numeric code is
unchanged. Full 2,000-draw stage3/ablation and V1c(a) on the real node outputs remain controller work, as in build 9.
The node command (after copying the code and frozen inputs, from the node checkout) is:

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES='' \
  /root/miniconda3/envs/CoSiR/bin/python src/test/20261027_recipe_sweep/rs_vp.py partitions \
  --machine node402 --outputs /data/SSD2/cosir_r10/20261027_recipe_sweep/outputs
```

Expected: equal partitions, A within 1e-6, 0 differ; duration depends on the completed sweep. Adjust the outputs
path to the node's actual mounted tree. Files proposed for eventual removal (nothing removed): the complete
/tmp/claude-0/r10-fix-12/ scratch tree after the controller has retained the reports and logs.


End-to-end commands and bounded-run method:
- Outputs were isolated as requested: symlinks to source entries, with stage3/ (PICK) and ablation/ (ticket 10)
  copied into owned scratch. Neither source example tree was written.
- Both direct commands reached timeout 235 (exit 124) here, so they remain unrestricted controller retries below.
  The PICK recheck with the review's partition memo passed: 165 items, 0 differ, largest numeric difference 0.
- To fit bounded commands, copied the review's vp_memo into owned scratch. Its key is SHA-256 of the full float32
  export array plus thread count; AST comparison confirmed rs_vp.partitions is unchanged from the pre-fix source.
  /tmp/claude-0/r10-fix-12/vp_memo.py calls rs_vp.main and memoizes only partitions. It still reads and validates raw
  exports, recomputes AMIs/bootstrap statistics, checks hashes, frozen/scorer partitions and the verdict. Any missing
  memo entries are computed by rs_vp.partitions and written only to owned scratch. This is scratch tooling, not a
  change to rs_vp's CLI or a scorer dependency.
- Full final tests after the mixed-terminal change: 80 passed in 67.90 s (test-final.log).

Unrestricted direct controller retries (both exceeded 235 seconds here; allow several minutes each):

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES='' \
  /root/miniconda3/envs/CoSiR/bin/python src/test/20261027_recipe_sweep/rs_vp.py stage3 \
  --sweep-machine local --draws 20 --outputs /tmp/claude-0/r10-fix-12/pick/outputs
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES='' \
  /root/miniconda3/envs/CoSiR/bin/python src/test/20261027_recipe_sweep/rs_vp.py ablation \
  --final k6=0.0005 --draws -1 --outputs /tmp/claude-0/r10-fix-12/abl/outputs
```

Expected: pass, 0 differ; memoized results are recorded above/below. The full self-check command above is still
required; neither the 20-draw examples nor the 57-item points check substitutes for its 2,000 draws.

Scope correction: the baseline and early red pytest invocations used pytest's default fixture root rather than
an explicit --basetemp. Known directories from the saved failure logs: /tmp/pytest-of-root/pytest-984 through
pytest-989. Subsequent runs set --basetemp beneath owned scratch, including all clone mutation runs. No cleanup was
performed. Those known test fixture directories are also candidates for eventual removal after ownership checks;
no removal of the shared /tmp/pytest-of-root directory is proposed.


Final gate edge: the no-score-file variant of the k11 part-(a) case first failed with FileNotFoundError
(red_gate_no_score.log). The gate-derived failed rows are now retained without requiring score files for runs
that never started. Score files, when present, still compare against failed / A=null. This adds one parameterized
case to the suite. The final-source mutation rerun is mutations-final81.log (all nine caught; restored by copy).

Final isolated example results (both use the source files' 20 draws, original examples untouched):

| Check | Items | Differ | Largest numeric difference | Result |
|---|---:|---:|---:|---|
| PICK stage3, F=k3=12 | 165 | 0 | 0 | pass |
| Development ablation, F=k6=0.0005 | 99 | 0 | 0 | pass |

Reports: /tmp/claude-0/r10-fix-12/pick/outputs/stage3/rederive_stage3.json and
/tmp/claude-0/r10-fix-12/abl/outputs/ablation/rederive_dev.json. Both reports' code_sha256 equals the final rs_vp.py.
The ablation's first memo run reached 235 seconds while filling entries; a second bounded call reused those entries
and passed. Final logs: pick-final2.log, abl-final.log. Each final call ran the same CLI arguments as the direct
commands above through vp_memo.py. The final integration merge check said "Already up to date" (26f556d).


Post-example final check: 81 passed in 54.40 s, after the final integration merge check (already up to date).
Command: the full CPU-only pytest command above with --basetemp
/tmp/claude-0/r10-fix-12/pytest-post-examples. Log: tests-post-examples.log. git diff --check passed.
Open controller work: full self-check --draws 2000; unrestricted direct example retries; full-draw production
stage3/ablation and V1c(a) on the real node. No unresolved unit-test or named-mutation failures.
