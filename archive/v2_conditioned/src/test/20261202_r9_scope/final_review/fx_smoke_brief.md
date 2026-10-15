# Fix ticket fx-smoke: the CLIP pre-read smoke refuses its own H6-path pass (r9 run, 2026-10-14)

Work ONLY inside `/project/CoSiR-wt/r9-fx-smoke` (git worktree, branch `r9-fx-smoke`). Every path below is relative to
that folder unless absolute. F = `src/test/20261202_r9_scope`. Do not create `.ccg` task records. Do not delete any
file outside a test's own tmp_path. Commit your work on the branch (one or two commits); do not merge, do not push.

## Rules for running commands
- Python: `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python`.
- Never wait more than about 4 minutes on one command. Run only the test files and test functions your change touches
  (`pytest -q -p no:cacheprovider <file>::<test>`); list any longer run in your hand-back instead of running it.
- Keep command output short (`-q`, tails, greps).
- Nothing is deleted: no `rm` of files you did not create in this task, no `git worktree remove`, no branch deletion.

## The failure
The real pre-read `run_r9_preread.py --read H8` (backbone clip_b32), run in the main checkout, passed inputs, heads,
regression, sensitivity and phase 1, then stopped at the smoke:

    smoke: failed, Stop: "held_pass_H6.json: its read and backbone are not H6 and qwen3vl_emb_2b"

Cause, as far as the controller traced it:
- `F/run_r9_preread.py` about line 315: `held_read = next((r for r in ('H6', 'H7') if RH.READ_BACKBONES[r] ==
  state.backbone), 'H6')`. CLIP has no ArtELingo held read in round 9, so the CLIP smoke runs the ArtELingo path as
  read H6 on backbone clip_b32 (by design of ticket 22: the smoke exercises both the ArtELingo path and the A28 path).
  The held runner's smoke writes `held_pass_H6.json` with `read: H6, backbone: clip_b32`.
- `F/run_r9_apply_rule.py` `load_pass` (about line 270, `# GUARD: pass-read`, added by fix ticket fx-held, A1 L1/L2)
  requires `rec["backbone"] == READ_BACKBONE_IDS[read]` in smoke mode too, so `AR.apply("H6", smoke=True, ...)`
  refuses. Qwen's pre-reads smoke H6 on Qwen and SigLIP's smoke H7 on SigLIP, so only CLIP's pre-read hits it.

## What to do
1. Confirm or correct the cause by reading the code (`run_r9_preread.py` smoke function, `run_r9_held.py` smoke
   path, `run_r9_apply_rule.py`, `run_r9_descriptive.py`). Then trace the WHOLE CLIP smoke chain for the H6 path past
   this guard: held runner smoke, `smoke_agreement`, `AR.apply(read, smoke=True, results=out)` (including how the
   verdict's claim names the backbone in smoke mode), and `RD.run(read, smoke=True, smoke_dir=out, smoke_tag=tag)`
   (fx-desc's `bind_read` compares a ledger name; any pass, verdict or backbone check there). List every check that
   would refuse a CLIP-on-H6 smoke, not only the first.
2. Fix them so a smoke pass of read H6 (or H7) made on another backbone is accepted IN SMOKE MODE ONLY, with the
   backbone still checked to be one of the three known backbone ids and consistent across the smoke's pass, verdict
   and descriptive files. Real (held) mode must keep every guard exactly as strict as now: a real pass whose backbone
   is not `READ_BACKBONE_IDS[read]` is still refused. The smallest change that does this is preferred. If you judge a
   different fix better (for example in the pre-read), explain why in the hand-back; do not drop the H6-path smoke
   for CLIP.
3. Do NOT edit any of these 17 stage 2 closure modules, and do not make any of them import a new module:
   r9_backbone, r9_bundle, r9_common, r9_context, r9_data, r9_episodes, r9_heads, r9_picks, r9_pins,
   r9_recipe_a3, r9_recipe_baselines, r9_recipe_cells, r9_recipe_groups, r9_recipe_reader, r9_score, r9_stats,
   run_r9_stage0a. Also do not edit run_r9_stage2.py. If the fix truly needs one of them, stop and say so in the
   hand-back. (An edit there would make the running stage 2 and the passed stage 0b stale.)
4. Tests (this is verdict code): add tests in the existing test files (`F/test_r9_apply_rule.py`,
   `F/test_r9_preread.py`, `F/test_r9_descriptive.py` or whichever holds the guard's tests) for: (a) a smoke pass of
   H6 with backbone clip_b32 goes through apply (and through the descriptive smoke binding if that had a check);
   (b) a real (held-mode) pass with the wrong backbone is still refused; (c) a smoke pass with an unknown backbone id
   is refused. Mutation check, each restored afterwards (`git diff` clean of the mutant): revert your fix, so (a)
   fails; make the real-mode guard smoke-only, so (b) fails; drop the known-id check, so (c) fails. Record the result.
5. Run the touched test files (each under 4 minutes; split by `::test` if needed). Report counts.
6. Write the hand-back `F/fx_smoke_handback.md` (short): cause confirmed, every check found in step 1, the diff in
   one paragraph per file, tests and mutants with results, and the exact command for an end-to-end scratch pre-read
   of H8 in the MAIN checkout (`/project/CoSiR`) with `--results` pointing at a fresh scratch folder under
   `F/results/` there. For that command, check in the code and state: (i) what the scratch run reads from the default
   results root versus the scratch root (stage 0a record, agreements, inputs), so it can pass; (ii) every path it
   writes outside the scratch root, if any (it must not occupy the names the real `run_r9_preread.py --read H8
   --fix 1` will later write in `F/results/` or `F/rederive/rd9_out/`).
7. Append a change-log entry to `.claude/20261014_log.md` (a `# <path>` header per edited file, before/after
   snippet, why). Commit everything (the log file may need `git add -f`).

Print at the end: the commit SHA(s), the test counts, the mutant results, and the scratch command.
