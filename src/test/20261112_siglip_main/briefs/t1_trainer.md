# r28 job T1: open r14's trainer to knob rows and removal rows on SigLIP 1 (verdict code)

Work only in the worktree `/project/CoSiR-r28-trainer` (branch `r28-trainer`). Do not create `.ccg` task records. Do
not touch any other folder except writing outputs under `/data/SSD2/cosir_r28/20261112_siglip_main/` and scratch under
`/data/SSD2/cosir_r28/20261112_siglip_main/smoke/`. Delete nothing; list what you would remove. Keep command output
short (`pytest -q`; long output to a file, then its tail). No command waits longer than about 4 minutes.

Read first: `src/test/20261112_siglip_main/briefs/common.md` (fixed facts), the handoff
`docs/superpowers/handoffs/2026-11-12-r28-run-handoff.md` sections 4 (items 9 and 10) and 5, and
`src/test/20261112_siglip_main/siglip_inventory.md` Q4.

## Goal
r14's trainer (`src/test/20261102_backbones_sweep/bs_cstar.py`, loss `bs_cstar_loss.py`) trains c* (arm `k11=40`) on
any image encoder but, for a non-Qwen encoder, takes only the row `ref`: knob rows are Qwen only ("the knob patches
build the 2048-wide trunk", `bs_cstar.py` docstring) and the non-Qwen loss asserts c*'s original switches
(`bs_cstar_loss.py` around line 40). r28 needs, on encoder `siglip1`:

1. **Knob rows** `k11=10`, `k11=20`, `k11=80`, `k6=0.0005` (r14's `row_config` / r10's `rs_sheet` knobs), each with
   the trunk's first layer at the encoder's width (1152) and everything else as the Qwen knob row.
2. **Removal rows** `instructed_graphs`, `content_correction`, `buddy`, `relational`: r10's removal keys
   (`src/test/20261027_recipe_sweep/rs_sheet.py`, `remove`; read how r10's trainer applied each). Each is c* with that
   one part removed, exactly as r10 removed it on Qwen, with only the image input and trunk width changed. The
   graphs, content correction, captions and teachers stay on Qwen3 as in r14 (so `instructed_graphs` keeps r10's
   single graph on the Qwen3 plain embedding).
3. `--keep all` outputs in r14's layout (weights, validation export `train/<arm>/R6-5/`, held export, run record), so
   the existing readers (`src/test/20261111_ablation_classcount/t5_ablation_ami.py`, r14's scorers) read them with only
   the root changed. Arm names: the row name.

## Rules
- New code goes in the round folder of the worktree: `src/test/20261112_siglip_main/r28_trainer.py` (and
  `r28_trainer_loss.py` if needed), importing r14's and r10's modules. Prefer patching/extension over copies. If an
  r14 file must change, the change must keep every existing call's behaviour identical; log it in
  `.claude/20261112_log.md`.
- The non-Qwen loss for each switch set must compute what r10's loss computes for that switch set (same terms,
  same order, same weights), with only the teacher-shape assertion adapted as r14 did. If a switch set needs a code
  path r14 copied away, take r10's path, not a re-derivation.
- `ref` on siglip1 must stay exactly r14's c* (same config, same model, same loss): the existing r14 SigLIP 1 c*
  weights are the ablation's reference row and are not retrained.
- Qwen behaviour for every row is unchanged.
- CLI: `r28_trainer train --encoder siglip1 --row ROW [--seeds 0 1 2 3 4] --keep all --root DIR [--smoke]`, plus
  `export` if r14 needs it separately. One run of one row and one seed must take under 15 minutes (r14's SigLIP 1
  ref seed 0 took 341 s).

## Tests (`src/test/20261112_siglip_main/test_r28_trainer.py`, fast, CPU)
- For every knob and removal row on siglip1: the config reaching the trainer equals r10's config for that row
  (Qwen) except the encoder fields; the switches reaching the loss equal r10's.
- The loss on siglip1 inputs for each removal row equals r10's loss function on the same tensors (tiny random batch,
  same seed), term by term where possible.
- `ref` on siglip1 gives the same config and loss as r14's path.
- A test that fails if the removal is silently ignored (for example: the `buddy` removal's loss differs from `ref`'s
  on the same batch).
Run them with `pytest -q`.

## Smoke
One `--smoke` run of `relational` and one of `k11=80` on siglip1, through
`res-queue gpu --no-wait --short --label r28:t1-smoke -- <cmd>` (exit 75: busy, say so), writing under
`/data/SSD2/cosir_r28/20261112_siglip_main/smoke/`. Never write into r10's or r14's output folders.

## Finish
Commit on branch `r28-trainer` (one or a few scoped commits; message ends with
`Co-Authored-By: Codex <noreply@openai.com>`). Do not push. Final message (at most 350 words): files, the full-run
command for the controller, test result line, smoke result, every result-shaping choice you made (choice · main
alternative · where · could it change a verdict), and what you left out.
