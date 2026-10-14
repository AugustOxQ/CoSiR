# Fix ticket fx-inf: Qwen's infinite pick breaks the descriptive smoke, and the pre-read names cannot absorb a second code change (r9 run, 2026-10-14)

Work ONLY inside `/project/CoSiR-wt/r9-fx-inf` (git worktree, branch `r9-fx-inf`). Paths are relative to that folder
unless absolute. F = `src/test/20261202_r9_scope`. Do not create `.ccg` task records. Do not delete any file outside a
test's own tmp_path. Commit on the branch; do not merge, do not push. Read-only access to the main checkout
`/project/CoSiR` is fine (its `F/results/` holds the real outputs named below); never write there.

## Rules for running commands
- Python: `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python`.
- Never wait more than about 4 minutes on one command. Run only the test files or `::test` functions your change
  touches; list any longer run in the hand-back instead.
- Keep command output short (`-q`, tails, greps).
- Nothing is deleted: no `rm` of files you did not create in this task, no `git worktree remove`, no branch deletion.

## Do NOT edit (stage 2 closure; an edit makes the running SigLIP stage 2 and the passed stage 0b stale)
r9_backbone, r9_bundle, r9_common, r9_context, r9_data, r9_episodes, r9_heads, r9_picks, r9_pins, r9_recipe_a3,
r9_recipe_baselines, r9_recipe_cells, r9_recipe_groups, r9_recipe_reader, r9_score, r9_stats, run_r9_stage0a, and
run_r9_stage2.py. Do not make any of them import a new module. If the fix truly needs one, stop and say so. Do not edit
any file outside F (earlier rounds' code and every pinned input are frozen).

## Part 1: the inf failure
The real pre-read `run_r9_preread.py --read H6` (backbone qwen3vl_emb_2b, main checkout) passed inputs, heads,
regression, sensitivity and phase 1, then its smoke failed with `ValueError: Out of range float values are not JSON
compliant: inf`. Its smoke folder `/project/CoSiR/src/test/20261202_r9_scope/results/smoke/H6_smoke_H6/` holds the held runner's smoke outputs
and a smoke verdict, so the held runner and the apply step passed; the failure is in the descriptive pass
(`RD.run(read, smoke=True, ...)`; `run_r9_descriptive.py` about line 519 dumps with `allow_nan=False`). The CLIP
smoke passed the same chain (H6 path and H8 path).
Likely source: Qwen's pinned seed-42 picks hold an infinite lambda: `"wang": {"0": 16.0, "1": "inf"}` in
`/project/CoSiR/src/test/20261202_r9_scope/results/recipe/qwen3vl_emb_2b/picks_seed42.json` (also baselines_seed42.json and
pins_qwen3vl_emb_2b.json store it as the string "inf"); CLIP's picks have none.
1. Find exactly where a non-finite float enters the descriptive report (file:line), by reading the code; a small
   repro that loads the real pinned Qwen picks JSON is welcome if it runs in under 4 minutes.
2. Fix it in the descriptive output only: encode non-finite floats as the strings "inf", "-inf", "nan", the way the
   recipe's JSON files already store them, at the point where the report is built or serialised. No number changes;
   the pass, verdict and every pinned file are untouched. Do not change `run_r9_held._plain` globally if the held
   runner uses it for pass files (keep the change local to the descriptive pass), unless you show it cannot change any
   existing output byte.
3. Check the real (held-mode) descriptive path for H6, H9 (Qwen) and H7, H10 (SigLIP 2) would not hit any other
   non-finite value from the picks, cells, taus or sensitivity (SigLIP's stage 2 is still running; its files will
   appear under `/project/CoSiR/src/test/20261202_r9_scope/results/recipe/siglip2_so400m/`; reason from the code paths).
4. Tests: a descriptive test whose picks hold an infinite lambda: before the fix it fails with this ValueError,
   after it the report is written with "inf"; mutant (fix reverted) caught.

## Part 2: a second code change after a pre-read
The pre-read kinds are fixed (`run_r9_preread.py` VARIANTS: smoke, fix1, crash1, reserve, heldfix1). The canonical
smoke records are shared by all reads, one per kind (`run_r9_apply_rule.SMOKE_RECORDS`), exclusive writes, and both
smoke-record selections (`run_r9_held.latest_smoke_record` about line 495 and `run_r9_apply_rule.latest_smoke_record`
about line 557) take the first existing of heldfix1, fix1, crash1, then smoke. The held launch gate
(`run_r9_preread.preread_guard` about line 418) takes fix1, then crash1, then the base pre-read.
State in the main checkout now: H8's base pre-read failed (smoke; fixed by fx-smoke), H8 `--fix 1` PASSED and wrote
`smoke_record_fix1.json` (+ the committed F copy) and `preread_H8_fix1.json`; H6's base pre-read failed (this inf).
Part 1 changes `run_r9_descriptive.py` (and maybe more), so every existing smoke record and pre-read becomes stale.
Then: H8 has no free pre-read name (base and fix1 used); H6 `--fix 1` would fail in `publish_smoke` because
`smoke_record_fix1.json` exists and is stale; and every gate prefers the stale fix1 record over any newer base or
crash1 record. The rule (`F/DECISION_RULE.md` section 6) says a failed smoke "is fixed and rerun", with no limit on
the number of fixes; the one-fix limit is the implementation's.

Implement numbered smoke generations, the smallest change that works:
1. `run_r9_preread.py --fix N`, N in {1, 2, 3}. Kind `fixN`; names `preread_<Hn>_fixN.json`, its per-read proof (as
   fix1's today), canonical `smoke_record_fixN.json` and its committed F copy (as fix1's), smoke folders
   `<Hn>_fixN_<read>` (as today's tag scheme). `--fix N` must not require an earlier pre-read of the same read (a read
   may go straight to the generation that matches the current bytes). `--fix 1` behaves exactly as today.
2. Every selection takes the newest generation first: smoke records heldfix1, fix3, fix2, fix1, crash1, smoke (both
   `latest_smoke_record` functions; the reserve and held `--fix 1` branches unchanged); the first-launch pre-read
   gate fix3, fix2, fix1, crash1, base. Keep every existing protection: exclusive writes, never a replaced or deleted
   record, a failed newer proof blocks an older pass, a stale proof never authorises, `publish_smoke` keeps a
   current canonical record and refuses a stale one.
3. Update `SMOKE_RECORDS`, VARIANTS, the CLI choices, docstrings and any other place that lists the kinds (grep the
   round folder, `rederive/` included: e.g. `run_r9_descriptive.py`, `run_r9_held.py` smoke names, mutation scripts,
   tests). Held `--fix 1` (heldfix1) semantics are unchanged.
4. Tests (this gates held reads, so it is verdict code): with a stale passed `smoke_record_fix1.json` present, a
   `fix2` pre-read publishes `smoke_record_fix2.json` and both smoke-record selections and the launch gate take it;
   a failed fix2 record blocks a passed fix1 record of the same bytes; a stale fix2 is refused; `--fix 1` names are
   unchanged; `--fix 4` is refused by the CLI. Mutants, each restored (`git diff` clean of the mutant): selection
   order without fix2 (caught), fix2 accepted when stale (caught), publish of a stale fix2 accepted (caught).

## After both parts
5. Run the touched test files (each under 4 minutes; split by `::test` if needed); report counts.
6. Write `F/fx_inf_handback.md` (short): the inf source (file:line), the fix, every place the kinds are listed and
   what changed, tests and mutants with results, and the controller's commands for the main checkout, in order:
   stage 0a; a scratch pre-read of H6 (`--results` a fresh folder under `F/results/`), saying what it writes outside
   the scratch root (it must not occupy any real name); then the real pre-reads with `--fix 2` for H8, H6, H9, H7,
   H10, and which of them may run at the same time (two runs that create the same canonical record at once must not
   race: say which runs first).
7. Append a change-log entry to `.claude/20261014_log.md` (a `# <path>` header per edited file, before/after snippet,
   why). Commit everything (the log file may need `git add -f`).

Print at the end: the commit SHA(s), the test counts, the mutant results, and the command list.
