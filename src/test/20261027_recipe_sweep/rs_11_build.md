# Ticket 11 build note (rs_vh.py)

State at build 10: rs_vh.py (stage0, stage4, ablation, selfcheck modes) and test_rs_vh.py (82 tests, about 45 s) are
done. Stage 4 and the ablation were run end to end against the example verdict files of tickets 09 and 10 (below).
Stage 0 was not run against a real verdict file: no example was given (the comparison is unit-tested and the same
block comparison is used for K4 in the self-check).

## Definitions settled within the ticket
- T_a partitions are recomputed from the instructed-image store as float32 rows divided by their float32 norm
  (`ct_diag.target_images` does this; the style store is not exactly unit norm, and without the division the style
  partitions differ from 12-24's frozen file; with it they are equal). Why: only that reading reproduces the frozen file.
- Student outputs are read as float16, unit norm checked, not repaired (RB rule 3.1); comparator probabilities are
  divided by their float32 row norm (CT rule 3).
- G's full-scale held outputs and the stage 4 candidate "G" are read from 12-27's folder (`held_dir`); every other run
  from `full_root(outputs)/held/<name>`; removal arms are named by `rs_sheet.name(config with remove)`.
- A scale or aspect whose outputs fail V0c has no outcome (V0d, V4d); the comparison then needs the verdict to say the
  same (outcome key present and None, with a cause). A missing outcome key is a disagreement, not a stated failure.
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
- A verdict with no outcome and a top-level `cause` (and no `aspects`): only the cause is compared; the run prints a
  NOTE and the item "a V0c failure in my outputs" must hold. Where ticket 09 will put this is a guess (top-level
  `cause`, or `outcome` None with `cause` inside an aspect: both are handled).

## Changes to computing code in build 10 (small; the self-check was rerun)
- `paired_statistic` also returns the per-seed point AMIs of the arm and F (`arm_seed_ami`, `final_seed_ami`), and
  `ablation` returns `labelled_rows`; `v0c_problems` reads N_HELD and WIDTH at call time (so tests can shrink H).

## Commands
- Tests (from the worktree): `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' /root/miniconda3/envs/CoSiR/bin/python -m pytest -q -p no:cacheprovider src/test/20261027_recipe_sweep/test_rs_vh.py`
- Full self-check (2,000 draws, about 8 minutes: run it in the background): `python rs_vh.py selfcheck --outputs /tmp/claude-0/r10-rs-11/full`
  Expected: 75 items, 0 disagree, largest numeric difference 0 (build 9, 12:33).
- Short self-check (200 draws, about 5 minutes under load): `python rs_vh.py selfcheck --draws 200 --outputs /tmp/claude-0/r10-rs-11b/self200c`
- Stage 4 on 09's example: `python rs_vh.py stage4 --candidate G --outputs /tmp/claude-0/r10-rs-11b/stage4/outputs`
- Ablation on 10's example: `python rs_vh.py ablation --final 'k6=0.0005' --outputs /tmp/claude-0/r10-rs-11b/ablation/outputs`
  (`/tmp/claude-0/r10-rs-11b/*/outputs` are folders of symlinks to the example, except the folder with the verdict file,
  which is a copy; the rederive file is written beside the copy.)

## Results
Build 10, 27 October 2026 (the machine load was 20 to 30, so times are long):
- Stage 4 on ticket 09's example (G under a fake KEEP-G, 2,000 draws): 58 items, 0 disagree, largest numeric difference
  0. It ran 518 s (more than the 235 s limit; run it in the background). The T partitions of all three aspects equal
  12-24's frozen files; labels_masked, d_in_ranges, max_entry and arms agree.
- Ablation on ticket 10's example (smoke, `final` k6=0.0005, removals direct and buddy, 20 draws taken from the verdict's
  `smoke: true`): 63 items, 0 disagree, largest numeric difference 0 (187 s). `direct` ok with every number and per-seed
  AMI; `buddy` failed (no outputs), only its status compared; the other six removals are not listed and have no outputs.
- Short self-check (200 draws, 286 s under load): 60 items, 0 disagree, largest numeric difference 0. With fewer draws
  than the frozen files the bounds and outcomes are left out, and the bootstrap draws are compared on the first 200 of
  the frozen list (the same stream; only RG's genre file keeps the list: 1 item) besides all point values, partitions
  and eligible seeds. Full self-check (2,000 draws) was not rerun in build 10: the controller runs it (command above).
  Build 9's full self-check, 12:33 (before the build 10 changes): 75 items, 0 disagree, largest difference 0.
- Both example-run reports: `/tmp/claude-0/r10-rs-11b/stage4/outputs/stage4/rederive_stage4.json` and
  `/tmp/claude-0/r10-rs-11b/ablation/outputs/ablation/rederive_ablation.json`.

## Left open
- Stage 0 against a real verdict file (ticket 05's real output, the sample scale needs the G sample held outputs).
- The real run of stage 4 for a PICK candidate other than G (needs ticket 09 on a real c*): the code path is the same
  with `held_dir(outputs, name)` under `full_root(outputs)`; only G was run.
- The real (non-smoke) ablation: all eight removals and 2,000 draws; the fixture run had two removals, 20 draws.
- Partitions of G's sample held outputs are compared with `<outputs>/stage0/partitions/<scale>/G/emotion__s{s}__r20.npz`
  only when the file exists; the count of unchecked partitions is in the report.
