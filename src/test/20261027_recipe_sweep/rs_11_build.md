# Ticket 11 build note (rs_vh.py)

State after the review fix wave (build 11): rs_vh.py (stage0, stage4, ablation, selfcheck modes) and test_rs_vh.py
(113 tests, about 55 s) are done. Stage 0, stage 4 and the ablation were run end to end against example verdict files
(below, "Build 11"). Earlier history of build 10 follows where it still holds.

## Review fixes (build 11; review_11.md)
- Must-fix 1 (F = G): the held ablation's `final_key` is `arm_name(final)`: G's folder is 12-27's arm GS-0.02, every other
  final its own name. Removal names and the held folder still use 'G'. Tests: `final_key` GS-0.02 with F = G passes,
  'G' disagrees.
- Must-fix 2 (equal partitions): `partitions.mismatches == 0` and `partitions.unchecked == 0` are items per scale at
  stage 0, per aspect and for the margin at stage 4 (`partition_items`). B1 to B6 are compared with 12-24's cache
  (`RB_PARTITIONS/<scale>/<arm>/percept__s<seed>__<reading>.npz`), not the CT's folder (it has no B6). At stage 4 each
  aspect's candidate partitions are compared with `stage4/partitions/full/<name>/{aspect}__s{seed}__r20.npz`; the margin
  uses the emotion files. A missing file counts as unchecked and fails.
- Should-fix 1 (no outcome): a `failed__{aspect}__s{seed}.json` record or a missing output is a problem of that set
  (`student_tasks`), not a crash. Stage 0 checks V0c on all three aspects of each scale (`student_problems`, no
  clustering) and a failure leaves the scale with no outcome. Stage 4 (`stage4_failure`): a training failure record,
  a `v4c.json` with pass not true, or any held set with a problem leaves every aspect and the margin without outcome; the
  verdict's top-level `v_failure` is read. On a no-outcome verdict only the outcomes are compared (mine None and theirs
  None, or both outcomes absent: a disagreement), the causes are printed (NOTE) and the rederive file is written.
- Should-fix 2: `smoke` is written into the rederive file; a smoke verdict whose outputs folder is not under
  `rs_paths.SMOKE_ROOT` (/tmp/claude-0) is refused (assert, no file written).
- Nits: a missing verdict key is an item (stage 0: block, outcomes, bootstrap_d, per_seed); per-seed values are compared
  over every arm I computed (the key sets must be equal); the sidecar's `arm` is checked (`arm_name`; G's sample outputs
  expect `STAGE0_SAMPLE_ARM` = 'G'); the partition constants cite CT rule 3; partition files read are in
  `inputs_sha256`.
- Computing code changed only in `student_parts` (new `arm`, failure records), `stage0_scale`, `stage4`, `entry_parts`
  (comparison with 12-24's cache); the point values, draws and bounds are untouched.

## Definitions settled within the ticket
- T_a partitions are recomputed from the instructed-image store as float32 rows divided by their float32 norm
  (`ct_diag.target_images` does this; the style store is not exactly unit norm, and without the division the style
  partitions differ from 12-24's frozen file; with it they are equal). Why: only that reading reproduces the frozen file.
- Student outputs are read as float16, unit norm checked, not repaired (RB rule 3.1); comparator probabilities are
  divided by their float32 row norm (CT rule 3).
- G's full-scale held outputs and the stage 4 candidate "G" are read from 12-27's folder (`held_dir`); every other run
  from `full_root(outputs)/held/<name>`; removal arms are named by `rs_sheet.name(config with remove)`.
- A scale whose outputs fail V0c in any aspect has no outcome (V0d); stage 4 with one failing set has none for every
  aspect and the margin (V4d); the comparison then needs the verdict to say the same (outcome key present and None, with
  a `v_failure` cause). A missing outcome key is a disagreement, not a stated failure.
- The self-check leaves out G's emotion margin over PercepT (never computed before; it is stage 0's held result).
- Verdict keys are pinned to the example files of build 10, no alternatives tried: a missing key is a disagreement,
  extra keys are ignored (a fixer may add some). To follow a rename edit `compare_t_block`, `compare_block`,
  `compare_stage4`, `_ablation_numbers` and `compare_ablation` (one place each).
  - Ticket 09, `stage4/verdict.json`: `aspects[a]{d, lo, hi, outcome, A_cand, A_T, d_in_ranges, labelled_rows,
    labels_masked}`, `emotion_margin{d, lo, hi, outcome, A_cand, A_e, max_entry, arms, eligible_seeds}`, `ranges`,
    `candidate`, `n_resamples`, `held_sha256`. Not compared: `kind` (stage 3's), `sources`, `gates`, `code_sha256`,
    `v4b`, and `reported.json` (the rule's V4b names the verdict's numbers; the reported margins come after it).
  - Ticket 10, `ablation/held.json`: `final_key`, `n_draws`, `seeds`, `labelled_rows`, `removals` (list), `results[x]{arm,
    status, cause, aspects[a]{effect, lo, hi, arm_ami, final_ami, arm_seed_ami{seed}, final_seed_ami{seed}}}`.
    A removal with status `failed`, or one with no outputs, has no numbers: only the status is compared. A removal that
    the verdict does not list must have no outputs either. A verdict that is not a smoke must list all eight.
- A verdict with `smoke: true` sets its own draw count (`draws_for`); any other verdict is re-derived with 2,000 draws
  and its own count is compared (an `n_resamples` or `n_draws` item).
- Stage 4 verdict with no outcome: top-level `v_failure`, `outcome` None and `v_failure` in each aspect and in
  `emotion_margin` (ticket 09 after its fixes): only those are compared with mine.

## Changes to computing code in build 10 (small; the self-check was rerun)
- `paired_statistic` also returns the per-seed point AMIs of the arm and F (`arm_seed_ami`, `final_seed_ami`), and
  `ablation` returns `labelled_rows`; `v0c_problems` reads N_HELD and WIDTH at call time (so tests can shrink H).

## Commands
- Tests (from the worktree): `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' /root/miniconda3/envs/CoSiR/bin/python -m pytest -q -p no:cacheprovider src/test/20261027_recipe_sweep/test_rs_vh.py`
- Full self-check (2,000 draws, about 8 minutes: run it in the background): `python rs_vh.py selfcheck --outputs /tmp/claude-0/r10-rs-11/full`
  Expected: 0 disagree, largest numeric difference 0; the item count is 75 plus the two `partitions_unchecked` items (build 9, 12:33: 75).
- Short self-check (200 draws, about 5 minutes under load): `python rs_vh.py selfcheck --draws 200 --outputs /tmp/claude-0/r10-rs-11b/self200c`
- Stage 4 on 09's example: `python rs_vh.py stage4 --candidate G --outputs /tmp/claude-0/r10-fix-11/e2e/s4/outputs`
- Ablation on 10's example: `python rs_vh.py ablation --final 'k6=0.0005' --outputs /tmp/claude-0/r10-fix-11/e2e/abl/outputs`
  (`/tmp/claude-0/r10-fix-11/e2e/*/outputs` are folders of symlinks to the example, except the folder with the verdict file,
  which is a copy; the rederive file is written beside the copy.)

## Results
Build 11, 27 October 2026 (fix commit on `worktree-agent-a9b6b08a3a20079fc`; end-to-end runs under
/tmp/claude-0/r10-fix-11/e2e/, each with the checked file's folder copied and the rest symlinked):
- Stage 0 on a copy of the review's tree (`s0`: the scorer's own code, 50 draws, G's sample set = K4's files with the
  sidecars' arm rewritten to 'G', because the fixture's sidecars say K4): 53 items, 0 disagree, largest difference 0.
  The same tree with one partition of `sample/G/emotion__s0__r20.npz` permuted (clusters 0 and 1 swapped, 3,171 rows):
  exit 1, 1 disagree (`sample.partitions.mismatches`).
- Stage 4 on 09's example (`--candidate G`, 2,000 draws, about 3.5 minutes): 66 items, 0 disagree, largest difference 0.
- Ablation on 10's example (`--final k6=0.0005`, smoke, 20 draws, 107 s): 63 items, 0 disagree, largest difference 0.
- Self-check with 50 draws: 62 items, 0 disagree, largest difference 0 (167 s). Point values (`--draws 0`): 27 items, 0 disagree, largest difference 0 (the K4 partitions of B1 to B6, B6
  included, are all checked: `partitions_unchecked` 0).
- Tests: 113 passed (about 55 s with the K4 integration test). On the old code 32 of the new or changed tests failed
  (red), all pass after the fix.
- Mutation check (shared clone, 36 mutants, run without the K4 test): all caught at the end. Two survived at first and
  led to test changes: dropping the own reading in the maximum (the case's own and r20 readings were equal; `make_case`
  now differs them) and `fixed_T_draws` with a one-seed pool (equivalent: a pool of one element draws no random bits;
  rerun with a two-seed pool: caught).
  Mutants: final_key rule; partition items (mismatches, unchecked, stage 0 items dropped); B6 not compared; candidate
  partitions skipped (stage 0, stage 4); missing output not a problem; `v4c.json` ignored; training failure ignored;
  margin kept on failure; one aspect dropped from the no-outcome items; V0c on emotion only; smoke refusal removed;
  smoke not written; sidecar arm unchecked; emotion tie to the largest code; share >= 0.5; kept count edge; non-finite
  seed kept; pool order reversed; own reading dropped; upper percentile; GO edge; T-LEVEL edge; T drawing; shared S_b
  split; unlabelled rows kept; E0 sample outcome; tolerance 1e-5; style label constant unchecked; genre conflict not
  masked; no outcome without cause; A_cand minus the minimum; per-seed arms skipped.
- Build 10 (archived, before the fixes): stage 4 on 09's example 58 items; ablation 63 items; short self-check 60 items;
  build 9's full self-check 12:33: 75 items, 0 disagree, largest difference 0.

## Left open
- Stage 0 against the scorer's real output (the example is the review's fixture: K4's files as G's sample set). The
  sidecar arm of G's sample-scale held outputs, 'G' (`STAGE0_SAMPLE_ARM`), is confirmed by the re-review (rereview_11.md, c).
- The scorer's `gates/v0c.json` is not compared (not asked): my own V0c verdict per scale could be checked against its
  `scale_pass`.
- A seed with both a `failed__` record and an output is treated as failed (tested since build 12: `test_a_seed_with_a_failure_record_and_an_output_is_failed`).
- The real run of stage 4 for a PICK candidate other than G (needs ticket 09 on a real c*): the code path is the same
  with `held_dir(outputs, name)` under `full_root(outputs)`; only G was run.
- The real (non-smoke) ablation: all eight removals and 2,000 draws; the fixture run had two removals, 20 draws.
- A missing partition file is now an item that fails (unchecked > 0); the scorer's partition cache must be complete.
