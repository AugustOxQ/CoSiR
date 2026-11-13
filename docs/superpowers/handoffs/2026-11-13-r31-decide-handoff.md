# Handoff: read r29's results with the user (step 1, decision C), wrap up r29, then decide the next direction

## Brief
- Done: r29 (SigLIP-only adapter: no Qwen3 pass for texts at test time) is reported: auto report 2027-01-18 and user-read report 2026-11-13, both checked.
- State: the validation pass rule fails on text (0.2549 against the bound 0.2573), so the model is an extra row; on test it beats plain SigLIP 1 in 13 of 15 cells (level 1).
- Next: show the user the results brief without our reading, get their reading, then wrap up and decide (step 2).
- Blocked: nothing.

> Written 2026-11-13 06:40 by "r29 run 3". Loop step reached: r29 step 9 done (both reports committed); the next loop
> starts at step 1 (decision C on r29). Direction log entry: r29's design is in
> [2026-11-12 · r29 decide](../direction_log.md); this chat writes r29's result entry at its wrap-up.

## 1. Read in this order

1. The results brief at the top of the auto report, `docs/reports/auto/percept/2027-01-18_siglip_only.md`: show it to
   the user **without** the folded "Our reading and recommendation" block (`research-loop.md`, "Decision C and the
   results brief").
2. The user-read report, `docs/user_read/2026-11-13_siglip_only.md`: the user may read it first; its advice block is
   folded the same way.
3. Only when a question needs the detail: the run log `src/test/20261112_siglip_only/20261112_siglip_only_log.md`, the
   numbers sheet `src/test/20261112_siglip_only/paper_cells.md`, the spec
   `docs/superpowers/specs/2026-11-12-siglip-only-adapter-design.md` (§6 is the rule).
4. For the wrap-up: `~/.claude/templates/direction_log_entry.md`, `~/.claude/references/storage-summary.md`.

Skills: none for step 1. Step 5 (the grill), if a new idea comes: `grilling` and `domain-modeling` (`grill.md`).

## 2. Decided by the user, not to be reopened

- r29's design: the spec's D1 to D25 (approved 2026-11-12 22:42). D3: a fail makes the new model an extra row or
  ablation, and stage 2 runs only on a pass. D9: the test cells go to the numbers sheet, and the paper chat places them.
- The user writes the paper; agents give numbers and drafts for reference only (project memory).

## 3. Where things stand

- Caption form (validation, seed 0): prefixed 0.2552827 against plain 0.2552660, a close call.
- Pass rule (validation, five seeds): level 0. Image group 0.3882 (bound 0.3833, current model 0.3863); text group
  0.2549 (bound 0.2573, current model 0.2594), all five seeds below. The loss is mostly emotion text to text (-0.0245).
  Stage 2 (other encoders, MMArt, photo model, ablation) did not run.
- ArtELingo test, level 1: 13 of 15 WIN over plain SigLIP 1. Style image mAP is BELOW under the joined three-head
  vector (0.287 against 0.296; the current model's cell is BELOW too; with the style head alone it is a WIN, +0.0588).
  Genre text to image is inconclusive (one number 1.02e-6 over the 1e-6 re-derivation tolerance). Against the
  instructed Qwen3: 12 WIN, the three emotion text cells BELOW.
- Text pass 1.138 ms per caption against Qwen3's 3.616 ms.
- Checks: Codex re-derived SC1, SC2 and all 30 test cells; Claude's claim review; Codex's claim check of the auto
  report, the user-read fidelity check and a cold read, one fix wave (report section "Claim review").
- r30 decide (r21's results) is open in its own tab. Both decide chats need the user; mention the other in one line.

## 4. Open points for the next step

1. **Key, decision C:** the user's reading of r29, given before ours (a sentence or two). Then open our reading and say
   where the two differ. Record it in r29's direction log entry.
2. **Key, decision A:** the next step among the brief's options: a. the current model stays main, the SigLIP-only
   model is an extra row (Tables 2, 3, 8; no runs); b. part of stage 2 for the extra row (about 6 GPU-hours; the stage
   2 script refuses without a pass, so this needs the user's amendment of D3); c. a new loop to close the emotion text
   gap without a Qwen3 pass. Ask the user first; our recommendation (a, park c until after 11-16) comes after.
3. **Key, for the paper:** how Table 2 shows image mAP against plain SigLIP 1: the joined three-head vector (spec D18,
   the agent default; the like-for-like form) or the aspect head (all three WIN; with the open genre cell accepted,
   level 3). The user decides; the paper chat places it.
4. Minor: the direction log's "Now" still has "the user's reading waits" for r25, r26 and r28; mention it in one line.

## 5. Code and pitfalls

- Records are gitignored (`*.json`): commit with `git add -f`.
- The run log's 06:23 line said best epochs are "as K0's"; it is wrong for emotion (corrected in its "Run 3" section).
- Option b: `r29_run_stage2.sh` refuses unless `records/pass_rule.json` says pass (D3). Do not edit the record; a run
  needs a written amendment and a flag.

## 6. State at handoff

- Running: nothing of r29 (queue idle).
- Uncommitted: Codex's working notes in `src/test/20261112_siglip_only/rederive/` (`findings.md`, `progress.md`,
  `task_plan.md`, `r29rd_c1_replay_text_impl.txt`), untracked; they can stay out of git. Three Codex ledger rows in
  `/project/claude-config/codex/jobs.md`: the wrap-up's `sync.sh` and commit cover them.
- On disk over 1 GB, for the wrap-up's storage summary:
  - `/data/SSD2/cosir_r29/20261112_siglip_only/` 52 GB: Codex's cache `rd_c1_cache` 38 GB (link
    `rederive/r29rd_c1_cache`); `gate_vi/try1/` 1.5 GB; the rest models, exports, scores, test and smoke folders.
    The paper cells cite the scores; the exports are needed only for a re-score or option b.
  - r29's items on `.scratch/pending_deletions.md` (rows dated 2026-11-13 00:20 to 06:28): build worktrees and
    branches, test scratch (4.6 GB on the system disk), smoke outputs, scratchpads.
- `wrapup-check` printed nothing at 06:38.
