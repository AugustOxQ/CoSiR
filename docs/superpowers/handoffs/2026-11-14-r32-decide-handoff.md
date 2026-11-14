# Handoff: read r31's results with the user (step 1, decision C), wrap up r29 and r31, then decide the next direction

## Brief
- Done: r31 (stage 2 of the SigLIP-only model) is reported: auto report 2027-01-19 and user-read report 2026-11-14, both checked.
- State: every stage 2 claim reaches level 3 (SigLIP 2, CLIP L/14, MMArt, photos); SigLIP 1 wins the encoder pick, a close call (0.3882 against 0.3838).
- Next: show the user the results brief without our reading, get their reading, then wrap up and decide (step 2).
- Blocked: nothing. The GPU and CPU are free, so r30 (the small check from saved outputs) can run.

> Written 2026-11-14 02:19 by "r31 run 5". Loop step reached: r31 step 9 done (both reports committed); the next loop
> starts at step 1 (decision C on r31). Direction log entry: r31's design is in
> [2026-11-13 · r29 read](../direction_log.md); this chat writes r31's result entry at its wrap-up.

## 1. Read in this order

1. The results brief at the top of the auto report, `docs/reports/auto/percept/2027-01-19_siglip_only_stage2.md`:
   show it to the user **without** the folded "Our reading and recommendation" block (`research-loop.md`, "Decision C
   and the results brief").
2. The user-read report, `docs/user_read/2026-11-14_siglip_only_stage2.md`: the user may read it first; its advice
   block is folded the same way.
3. Only when a question needs the detail: the run log `src/test/20261112_siglip_only/20261112_siglip_only_log.md`
   (section "Stage 2 run"), the numbers sheet `src/test/20261112_siglip_only/paper_cells.md`, the spec
   `docs/superpowers/specs/2026-11-12-siglip-only-adapter-design.md` (§6 is the rule; amendments A1 and A2).
4. For the wrap-up: `~/.claude/templates/direction_log_entry.md`, `~/.claude/references/storage-summary.md`.

Skills: none for step 1. Step 5 (the grill), if a new idea comes: `grilling` and `domain-modeling` (`grill.md`).

## 2. Decided by the user, not to be reopened

- r29's design: the spec's D1 to D25 (approved 2026-11-12 22:42).
- A1 (2026-11-13, after seeing r29's results): stage 2 runs on the fail; the SigLIP-only model stays an ablation or
  supplementary row. A2: an encoder pick among the SigLIP-only models on validation (SC9).
- The user with the co-authors, 2026-11-13 15:25: the SigLIP-only model is a nice to have; change few numbers now and
  focus on the writing.
- The user writes the paper; agents give numbers and drafts for reference only (project memory).

## 3. Where things stand

- **Other encoders (SC4), level 3 each.** SigLIP 2 and CLIP L/14: 6 of 6 WIN over the plain encoder; against the
  instructed Qwen3, 5 WIN and emotion retrieval LEVEL (SigLIP 2 +0.0011 [-0.0001, +0.0021]; CLIP L/14 -0.0016
  [-0.0027, -0.0004], wholly below zero, inside the LEVEL unit 0.0041).
- **MMArt (SC5), level 3:** 6 of 6 WIN over plain SigLIP 1; against the instructed Qwen3 5 WIN, emotion retrieval LEVEL
  (+0.0001 [-0.0004, +0.0005]).
- **Photos (SC6), level 3:** 4 of 4 WIN over both comparators (Affection mAP 0.325 against 0.295; EmoSet 0.487 against
  0.452).
- **Ablation (SC7), no level:** style falls most without the instructed graphs (AMI -0.215) or the graph-positive loss
  (-0.169); only the relational term's removal lowers genre (AMI -0.060, mAP -0.114).
- **Encoder pick (SC9):** SigLIP 1 0.3882, SigLIP 2 0.3838, CLIP L/14 0.3762 (validation); margin 0.0045, ahead in
  5 of 5 seeds; a close call (below SigLIP 1's seed range 0.0055).
- Stage 1's text fail (SC2) stands: the model stays an extra row.
- Checks: Codex re-derived every stage 2 number (C13, all pieces pass at 1e-6); Claude's claim review; the user-read
  fidelity check and cold read, one fix wave (report section "Claim review").
- Three agent choices could change a claim (folded block of the brief): the gate (vii) recheck (without it, no
  ablation), the void start-up crash of SigLIP 2 (counted as a crash: no encoder pick and no SC4), and the shared seed
  draws of the ablation intervals (one small genre change crosses zero without them).

## 4. Open points for the next step

1. **Key, decision C:** the user's reading of r31, given before ours (a sentence or two). Then open our reading and say
   where the two differ. Record it in a new direction log entry for r31 (r29's entry holds its stage 1 reading).
2. **Key, decision A:** the next step among the brief's options: a. write the supplementary rows (paper work, no runs);
   b. r30, the planned check from saved outputs (one shared UMAP map; do long DEC's two emotion groups split positive
   from negative emotions?), bounded CPU work, free to run now; c. caption form and prefix searches per encoder after
   the deadline. Ask the user first; our recommendation (a, then b; c after 11-16) comes after.
3. **Key, for the paper:** how image retrieval against a plain embedding is shown (the joined vector, as the spec, or
   the aspect head). Still open from r29; the paper chat places it.
4. Minor: the direction log's "Now" still says "the user's reading waits" for r25, r26 and r28; mention it in one line.

## 5. Code and pitfalls

- Records are gitignored (`*.json`): commit with `git add -f`. Never edit a record or a gate record.
- Report names keep CoSiR's sequence dates (the next auto report is 2027-01-20); user-read reports use real dates.
- The paper is due 2026-11-16 (supplementary 2026-11-23).

## 6. State at handoff

- Running: nothing of r31 (queue idle, GPU free).
- Uncommitted: Codex's working files in `src/test/20261112_siglip_only/rederive/` (about 114 MB: caches, probe
  scripts, notes), untracked; they stay out of git. The Codex ledger rows in `/project/claude-config/codex/jobs.md` are
  synced by this chat's `sync.sh`.
- On disk over 1 GB, for the wrap-up's storage summary:
  - `/data/SSD2/cosir_r29/20261112_siglip_only/` 23 GB (5.8 GB before stage 2; ablation 10 GB, cstar 7.7 GB, mmart
    1.5 GB). The numbers sheet cites the records; the exports are needed only for a re-score.
  - New feature stores on `/data/SSD2/pre_extract/`: `affection_local/siglip_so400m_txt` 1.2 GB,
    `artelingo_full/siglip2_so400m_capprefix` 0.7 GB, `artelingo_full/clip_l14_capprefix` 0.5 GB.
  - r31's items on `.scratch/pending_deletions.md` (rows dated 2026-11-13): build worktrees (about 925 MB), branches,
    the build tracker `.scratch/r31-stage2/`.
- `wrapup-check` printed nothing at 02:18. Queue idle (GPU and CPU free).
