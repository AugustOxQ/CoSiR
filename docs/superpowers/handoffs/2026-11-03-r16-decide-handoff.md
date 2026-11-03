# Handoff: r16 decide: the user reads the r15 results (decision C), then the wrap-up, then the next direction (step 2)

## Brief
- Done: r15 ran overnight: the MMArt test on four encoders, the C13 re-derivation, ten diagnostics, both reports.
- State: holds, GO on all four encoders; C13 matched all 6,697 items (max difference 0). Teacher: 18 WIN, 4 LEVEL,
  2 BELOW (emotion retrieval, Qwen and CLIP L/14).
- Next: show the user the results brief without our block and ask for their reading (decision C).
- Blocked: nothing; storage decisions wait for the user.

> Written 2026-11-03 05:05 by tab "r15 run". Loop step reached: r15 step 9 done; r16 starts at step 1 (decision C).
> Direction log entry: none yet for r15's result (the wrap-up writes it); the r15 decide entry is
> [2026-11-02 · Backbones sweep read](../direction_log.md).

## 1. Read in this order

1. The auto report's results brief, [`docs/reports/auto/percept/2027-01-03_mmart_test.md`](../../reports/auto/percept/2027-01-03_mmart_test.md):
   step 1 shows it without the folded "Our reading and recommendation", asks the user's reading in a sentence or two,
   then opens ours and says where the two differ.
2. The user-read report [`docs/user_read/2026-11-03_mmart_test.md`](../../user_read/2026-11-03_mmart_test.md), for
   the user (its advice is folded the same way).
3. The round log [`src/test/20261102_mmart_test/20261102_mmart_test_log.md`](../../../src/test/20261102_mmart_test/20261102_mmart_test_log.md)
   ("Run" section) only if a question on the run comes up.

Skills: none for step 1; `grilling` and `domain-modeling` (grill-with-docs) later, at step 5.

## 2. Decided by the user, not to be reopened

- The r15 design and its decision rule (spec approved 2026-11-02 18:37, e96a7cd) and the test cut at cosine 0.95
  (D19, user, 471859e).
- Deadlines: abstract 2026-11-10, paper 2026-11-16, supplementary 2026-11-23 (CLAUDE.md, direction log "Now").

## 3. Where things stand

- Verdict per encoder (Qwen3-VL-Embedding-2B, SigLIP 2, SigLIP 1, CLIP L/14): holds, GO; all nine plain-image and
  PercepT cells WIN for each. Example: Qwen emotion AMI 0.0609 against 0.0407 for the plain image.
- Against the teacher: all 15 cells set for "better than" WIN (narrowest lower bound +0.0004, CLIP L/14 emotion AMI);
  r14 on ArtELingo had 18 WIN, 5 LEVEL, 1 BELOW.
- Diagnostics (decide nothing): on the 11,528 shared paintings, Qwen's emotion clustering gain with MMArt labels is
  about half the gain with ArtELingo labels (+0.0192 against +0.0364); the unseen-artist subset has 15 paintings, so
  no cells.
- Options in the brief: freeze and write; a second aspect on photos; a painting test with new artists. Our
  recommendation (folded): freeze and write.

## 4. Open points for the next step

- Key: the user's reading of the results (decision C), asked before ours is shown.
- Key, at step 2: the direction (decision A), with the abstract a week away.
- Minor, at the wrap-up: the type note. The direction log called r15 a confirmation; the spec ran stage 2 as a claim
  test (agent default D6). The direction log entry records what ran.
- Minor, at the wrap-up: the storage summary. `.scratch/pending_deletions.md` has the r15 build rows (smoke
  `smoke_build3/` 12 GB; system-disk smoke 3.3 GB and a build scratchpad 6.4 GB; 11 worktrees and the ticket
  branches) and the r15 run rows (C13 smoke and cache, 16 MB each; `OUT/exports/` 9.0 GB, recommended keep;
  `OUT/stage0/` 2.9 GB). Items of 1 GB or more go in the disk ledger.
- Minor, at the wrap-up: `routing-check` prints the three items the user deferred to a tooling chat after
  2026-11-09; nothing new. `/project/claude-config/sync.sh` was not run by the run chat.

## 5. Code and pitfalls

- Agent defaults of the run (round log): C13 fits k-means at 8 threads, the round's setting (one thread moves one AMI
  by 1.3e-5); no cut to a second run chat after the compaction.
- Fixed in the run: the C13 compare's source check (it must rebuild `bs_c13.frozen_identity()` with the committed
  bytes, since that identity hashes the loaded `mt_c13.py`; f13117d), and a display bug in `mt_diag.py` (cell lookups
  lacked the task; a3b7e29; the two affected tables were re-rendered from their saved JSON, cells unchanged).
- zsh in this container: arrays start at 1, `$c:s…` is a history modifier, and `mt_run_cpu.sh`'s printed commit
  command fails (a literal `\n`, an unsplit `$P`); use an array and `git commit -F - <<'EOF'`.
- `*.json` is gitignored: records need `git add -f`. Other chats (the paper chat) commit on this repo too: commit
  only your own paths.

## 6. State at handoff

- Running: nothing (no background job, subagent, GPU lock or Codex job of the run chat).
- Uncommitted: nothing of the run chat (pre-existing untracked files in `archive/` and `src/test/2026101*` are not
  this loop's).
- On disk over 1 GB: `/data/SSD2/cosir_r15/20261102_mmart_test/outputs/` 12 GB (exports 9.0 GB, stage 0 2.9 GB),
  `/data/SSD2/cosir_r15/20261102_mmart_test/smoke_build3/` 12 GB, and the build's system-disk items above.
