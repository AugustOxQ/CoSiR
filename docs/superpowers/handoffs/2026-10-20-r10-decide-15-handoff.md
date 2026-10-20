# Handoff: decision C on claim test stage 1, the morning check of the spec's brief, and decision B for track 2; then the wrap-up

## Brief
- Done: claim test stage 1 is GO; the auto report and the user-read report are committed.
- State: held emotion AMI, D2 0.1001 against PercepT core 0.0855 (d +0.0146 [+0.0093, +0.0157]).
- Next: your reading of stage 1 (decision C), then the check of the spec's brief, then track 2's idea (decision B).
- Blocked: nothing; the stage 2 build runs unattended in "r10 build 13".

> Written 2026-10-20 06:56 by "r10 run 7". Loop step reached: track 1 (claim test), step 9 done, step 1 (decision C)
> next; track 2 (cluster-level losses) at step 4 (decision B). Direction log entry:
> [2026-10-20 · Stage 2 model sheet read](../direction_log.md), bullet "Grill, claim test".

## 1. Read in this order

1. This file.
2. The auto report, [2026-12-18_claim_test_stage1.md](../../reports/auto/percept/2026-12-18_claim_test_stage1.md):
   its results brief only. Show the user the brief without the collapsed block "Our reading and recommendation".
3. The user-read report, [2026-10-20_claim_test_stage1.md](../../user_read/2026-10-20_claim_test_stage1.md): the user
   may read it instead of the brief; its advice is folded away too.
4. The spec's brief, [2026-10-20-claim-test-design.md](../specs/2026-10-20-claim-test-design.md) ("Brief", "Check
   these").
5. For decision B: [the cluster-level losses note](../../reports/literature/2026-12-17_cluster_level_losses.md),
   "Summary" and "Candidate ideas for a one-day exploration loop (ranked)".
6. `~/.claude/rules/research-loop.md` ("Decision C and the results brief", "Wrap-up and handoff") and, if decision B
   leads to a grill, `~/.claude/references/grill.md`.

Skills: none for decision C; `grilling` and `domain-modeling` (grill-with-docs) if decision B goes on to a grill.

## 2. Decided by the user, not to be reopened

- Every grill answer of the claim test (spec §4, D1 to D15); night mode and the overnight go-ahead (2026-10-20 01:27,
  spec D29); node408 allowed (spec D23).
- The rule's consequence of this outcome (rule §4, user 8b and S8): after S1-GO, stage 2 runs. So "r10 build 13" was
  opened at once. The 10-29 gate passes on S1-GO (agent default in the rule, D34; part of the brief check below).

## 3. Where things stand

- **Outcome S1-GO** (`src/test/20261020_claim_test/verdict_stage1.json`, commit b77d3ec; code 81113a1, not dirty).
  12,281 held paintings, 5 model seeds, k-means at 20 clusters. D2 0.1001 against the strongest entry, PercepT core
  read by its own top topic, 0.0855. The other entries: 0.0652 to 0.0779. Bar: d at least 0.01, lower bound above 0.
- **Paper settings PercepT:** all 10 strict runs collapsed (kept 1, 2, 2, 98, 98 and 98, 99, 99, 99, 99 of 100
  topics); the K = 40 versions entered with all five seeds and scored below core (0.0707 and 0.0687, own reading).
- **Checks:** the stage 0 regression check passed twice; the independent re-derivation agrees exactly (partitions,
  eligibility, every score, d, both bounds, all 2,000 resampled d); V3 and V4 pass.
- **Beside the verdict, deciding nothing** (`diag_stage1.md`):

  | Reading | D2 minus comparator |
  |---|---:|
  | Emotion, Leiden, against core | +0.0019 [−0.0030, +0.0053] |
  | Style, against core | +0.1436 |
  | Genre, against core | +0.0079 [+0.0002, +0.0153] |
  | Instructed Qwen image (target), emotion / style / genre | 0.0921 / 0.4026 / 0.5062 against D2 0.1001 / 0.3719 / 0.4490 |

- **Reports:** auto report (bc49a6e) and user-read report (this chat) committed; held ledger row H11 committed.
- **Stage 2 build** ("r10 build 13", unattended) started at the same time as this chat. Its handoff:
  [2026-10-20-r10-build-13-handoff.md](2026-10-20-r10-build-13-handoff.md). It builds and smoke-tests only; the full
  run waits for the evening or goes to node408. Its first estimate (not checked): extraction of the missing instructed
  images about 2.3 to 2.7 GPU h, training about 2.6 GPU h if time scales with the data, so about 5 GPU h.

## 4. Open points for the next step

All the user's. Ask each key question answer-first (the user's reading or pick, one line of why, before ours).

a. **Decision C, key.** The user's reading of stage 1 from the auto report's results brief, given before ours. Show
   the brief without the collapsed block; then open ours and say where they differ. Record the reading.
b. **The morning check of the spec's brief** (spec D29: the user approved by "automate and hand me the results",
   01:27). Go through its "Check these" items, above all "Please confirm: after stage 2, its outcome alone sets the
   next step" (S1, S6; rule §4, D35). A change that touches the verdict is an amendment after results: record it in
   the spec with the date and "after results", commit it, and tell the build chat through the commit (it checks the
   spec's git log before tickets and before the merge).
c. **Decision B for track 2, key, answer first.** From the cluster-level losses note, three ranked ideas for a
   one-day exploration loop: 1 a teacher-partition anchor per head (soft, on core samples); 2 a TEMI-style neighbour
   self-distillation head on the buddy pairs; 3 a comparator and ceiling probe with no new training of our model. The
   note finds no support for DEC started from a good partition. The Leiden result above (+0.0019) bears on the user's
   challenge of the clustering step. Track 2 runs on development data only; the held paintings stay for the claim.
d. **Where stage 2 runs**, if the user wants to say before build 13's estimate: local overnight (GPU free in the
   evening, ends well before 11:00 at about 5 GPU h) or node408 (3 GPUs, can start in the daytime). If the user
   decides, commit the choice in the direction log; build 13 reads it.
e. **The wrap-up** (`research-loop.md`): decision C and the direction log entry, its "Now" paragraph; the storage
   summary with every row of `.scratch/pending_deletions.md` (`storage.md`); `wrapup-check`, `disk-check`,
   `routing-check`; `/project/claude-config/sync.sh`, commit and push that repo.

## 5. Code and pitfalls

- Stage 1's round folder is closed: no edit to any `ct_*.py`, queue script or `DECISION_RULE.md` (it breaks the stage
  0 pass that V1 checks; held NPZs exist, so stage 0 cannot retrain).
- Build 13 works in the same repo at the same time: commit only your own files (docs, the direction log), and pull
  before you push.
- JSON outputs are gitignored: `git add -f` for any new verdict or diagnostics JSON.
- The auto-mode check refused a few plain Bash calls for some minutes after a Codex launch (seen twice this morning);
  the Read tool works meanwhile.

## 6. State at handoff

- Running: "r10 build 13" (unattended, opened with this chat). No background job, GPU lock or Codex job of "r10 run 7".
- Uncommitted: nothing in git after this cut's commit. Outside git: `.scratch/r10-claim-test/report/` (briefs, Codex
  logs, drafts), `.scratch/r10-claim-test/v2/` (V2 files, 7.4 MB).
- On disk over 1 GB: `src/test/20261020_claim_test/outputs/`, 3.1 GB (keep: stage 1 artifacts the report rests on;
  disk ledger row added); `src/test/20261019_model_sheet/outputs/`, 3.4 GB, kept (disk ledger);
  `/project/CoSiR-wt/ct-09/.scratch/`, 1.6 GB, on `.scratch/pending_deletions.md`.
