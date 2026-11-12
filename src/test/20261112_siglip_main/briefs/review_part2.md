# r28 final review, part 2: the settings check and the Table 5 ablation on SigLIP 1 (verdict code and results)

You review; you do not fix. Write only your report, `src/test/20261112_siglip_main/logs/review_part2.md`, and scratch
files under `/tmp/claude-0/-project-CoSiR/dae700bf-e7a2-4618-ae57-fa086b1e35b9/scratchpad/review2/`. Delete nothing;
list what you would remove. Keep command output short (`pytest -q`; long output to a file, then its tail or a grep).
No command waits longer than about 4 minutes. No GPU. Python: `env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
MKL_NUM_THREADS=8 /root/miniconda3/envs/CoSiR/bin/python` (the shell is zsh: write the command out in full, a `$PY`
variable does not word-split). The repo `/project/CoSiR` is public; `docs/paper/` is read only.

## Context
r28 puts CoSiR on the SigLIP 1 student in the paper's tables. Part 1 of this review (Opus, 20:33) passed every
scored-from-saved-models cell. Part 2 covers what ran after it. Handoff:
`docs/superpowers/handoffs/2026-11-12-r28-run-handoff.md` section 4 items 9 and 10. Round folder
`src/test/20261112_siglip_main/` (run log `20261112_siglip_main_log.md`, `decisions.md`). Briefs of the builders:
`briefs/t1_trainer.md` (trainer, built by Codex, reviewed by Opus high: PASS, with one gap: no test checks that `k11`
reaches the graph build) and `briefs/f_settings_ablation.md` (both scorers).

## What to check, in order of weight
1. **The removal rows train what the Qwen3 ablation removed.** `r28_trainer.py`, `r28_trainer_loss.py` against r10's
   removal keys (`src/test/20261027_recipe_sweep/rs_sheet.py`, `remove`, and how r10's trainer applied each) and r14's
   trainer (`src/test/20261102_backbones_sweep/bs_cstar.py`, `bs_cstar_loss.py`). From the run records and logs under
   `/data/SSD2/cosir_r28/20261112_siglip_main/cstar/siglip1/` (`train/<arm>/`, `weights/<arm>/`, `held/<arm>/`), show
   that each of the four rows ran with its removal applied and nothing else changed (config in the run record, loss
   terms logged), five seeds each, and that the held exports join by the extraction's sample IDs (constitution C6).
2. **The settings rows changed the setting.** Close the part-1 gap: show from the run records or the saved graphs
   that `k11=10`, `k11=20`, `k11=80` built their graphs with that k11, and `k6=0.0005` used that weight (seed 0).
   `k11=20` failed once with a CUDA error and was rerun (`run_k11_20_rerun.log`); check the rerun is complete.
3. **The settings check** (`r28_settings.py`, `test_r28_settings.py`, `records/settings_check.json`). The fallback
   criterion (mean of six validation cells, c* wins ties) was used because r10's stage 3 inputs do not exist for
   these runs (decision line in `decisions.md`). Check that reason against `src/test/20261027_recipe_sweep/DECISION_RULE.md`
   §3.3 and §4.3; check no test file or test split is read; check c* (seed 0) is the r14 c* validation export.
   The outcome says k6=0.0005 wins by +0.0007; that is flagged for the user, not acted on. Re-derive the six-cell means
   of c* and of k6=0.0005 yourself from the validation exports, with your own code calling only the metric functions.
4. **The ablation scorer and its numbers** (`r28_ablation.py`, `test_r28_ablation.py`, `records/ablation_siglip1.json`,
   log `/data/SSD2/cosir_r28/20261112_siglip_main/ablation/score.log`). The c* gate must be exact against r17; the
   removal deltas are removal minus c* of the five-seed means. Re-derive, with your own code, at least two removal
   cells end to end from the held exports (one AMI at the class count, one plain-cosine mAP; for example
   `instructed_graphs` style AMI at k=27 and `relational` genre mAP), and say whether they match to 1e-6. Check the
   scorer reads the test split only to score, never to choose.
5. **The numbers sheet** `paper_cells.md` Table 5 rows and the change to `r28_cells_check.py` (`norm`, `--values-only`)
   in commit e1ab6fdb: is the checker still strict (a wrong digit or a wrong sign is caught)? Run
   `r28_cells_check.py --values-only`.
6. **Small:** commit 18229a70 narrowed a guard in `r28_clay_crl.py` `fix_records` (the pick rerun had crashed on the
   clustering rule). Is the new guard right, and does it miss any record whose rule names the CLIP L/14 unit?

Run `pytest -q src/test/20261112_siglip_main/test_r28_trainer.py test_r28_settings.py test_r28_ablation.py
test_r28_clay_crl.py` once (from the round folder; skip a test that needs over 4 minutes and say so).

## Report
`logs/review_part2.md`: verdict PASS or FAIL first; then findings ranked by severity, each with file:line, what is
wrong, the evidence, and whether it changes a number in Table 5 or the settings outcome. Mark each finding blocking
or not. List what you re-derived, with your value beside the record's. Hand back the verdict, the blocking findings
and the re-derived values in under 300 words.
