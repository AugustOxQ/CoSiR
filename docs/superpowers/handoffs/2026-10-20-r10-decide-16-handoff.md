# Handoff: the user's reading of the claim test's stage 2 (decision C), then the rule's next step and the wrap-up

## Brief
- Done: claim test stage 2 ran locally; outcome GO (S2-GO); the independent check agrees exactly; both reports committed.
- State: D2 emotion AMI 0.1008 against PercepT core's own reading 0.0874; d +0.0133 [+0.0081, +0.0146].
- Next: the user's reading of stage 2 (decision C), then the paper sentence and ArtELingo-28's facts brief and spec.
- Blocked: nothing; "r10 decide 15" may still be open with stage 1's reading and track 2's decision B.

> Drafted 2026-10-20 23:20 by "r10 run 10"; finished by "r10 run 11" after the user-read report. Loop step reached: track 1 (claim test), stage 2, step 9 done (verify and
> report); step 1 (read results, decision C) next. Direction log entry: [2026-10-20 · Stage 2 model sheet
> read](../direction_log.md), bullet "Grill, claim test".

## 1. Read in this order

1. This file.
2. The auto report, [2026-12-19_claim_test_stage2.md](../../reports/auto/percept/2026-12-19_claim_test_stage2.md):
   the results brief and the section heads. Its numbers were checked against the sources (claim review, Sonnet
   `high`); do not re-derive them. The user-read report beside it:
   [2026-10-20_claim_test_stage2.md](../../user_read/2026-10-20_claim_test_stage2.md).
3. The rule `src/test/20261020_claim_test/DECISION_RULE.md` §4 ("What each outcome leads to") and the spec
   `docs/superpowers/specs/2026-10-20-claim-test-design.md` §6 (the licensed claim, the paper's sentence).
4. [The r10 decide 15 handoff](2026-10-20-r10-decide-15-handoff.md) §4: its open points, if decide 15 has not closed
   them (see §4 b below).
5. `~/.claude/rules/research-loop.md` (wrap-up), `~/.claude/references/grill.md` only if a new idea is grilled.

Skills: none for decision C; `grilling` and `domain-modeling` only if the user grills a next idea here.

## 2. Decided by the user, not to be reopened

- Every grill answer of the claim test (spec §4, D1 to D15); night mode and the overnight go-ahead (spec D29).
- After stage 2, its outcome alone sets the next step (rule §4, user S1). For S2-GO: the paper's sentence (spec §6,
  stage 2), then ArtELingo-28's facts brief and spec.
- Stage 2 device: node408 (user, 2026-10-20 about 14:55), local GPU from 19:00 as the fallback. The fallback ran
  (§3).
- Global rule (user, 2026-10-20 15:00, `shared-resources.md`): no long heavy local GPU run on weekdays 10:00 to
  19:00.

## 3. Where things stand

- **Outcome S2-GO** (`src/test/20261020_claim_test/verdict_stage2.json`, commit 34cbf2d; code c2eabfb; `dirty` true
  only for the run log, edited during the run). D2 0.1008 against the strongest entry, PercepT core read by its own
  top topic, 0.0874; d +0.0133 [+0.0081, +0.0146] against the bar d ≥ 0.01 and lower bound > 0. Stage 1 was d
  +0.0146 [+0.0093, +0.0157]. All 10 strict paper settings runs collapsed again; the K = 40 versions entered with
  seeds 0 to 4 and scored 0.064 to 0.071. No paper settings entry reaches D2, so none becomes the comparator to
  improve against.
- **V checks:** V1 local stage 2a PASS (28 items, every regression bit for bit on the 3090); V2 on Codex: exact
  agreement (partitions, eligibility, every score, d, both bounds, all 2,000 resampled d;
  `.scratch/r10-claim-test-s2/v2_compare/compare_v2.log`); V3 and V4 hold. Held ledger row H12.
- **Diagnostics** (`diag_stage2.md`, decide nothing): Leiden emotion margin −0.0005 [−0.0048, +0.0023] (stage 1
  +0.0019 [−0.0030, +0.0053]): under Leiden the emotion lead vanishes again. Style +0.1388 [+0.1283, +0.1427], genre
  +0.0208 [+0.0103, +0.0310] over core. Instructed Qwen image (the 11-10 target) 0.0921 / 0.4026 / 0.5062 against D2
  0.1008 / 0.3685 / 0.4438: D2 above on emotion only.
- **The device story** (report, "Deviations"): node408 stopped at a guard bug (fixed, de45b87), then its A6000 did
  not reproduce stage 1 (AMI differences up to 0.0168, limit 1e-4); the node route closed; the local 3090 reproduced
  stage 1 bit for bit and ran stage 2 from 19:00 to 23:02. Spec D23 (PercepT on node408) did not hold; the report
  says why.
- **Run log:** `src/test/20261020_claim_test/20261020_claim_test_run_log.md`, section "Local stage 2 run (chat r10
  run 10)".

## 4. Open points for the next step

All the user's. Ask each key question answer-first (the user's reading or pick, one line of why, before ours).

a. **Decision C, key.** The user's reading of stage 2 from the auto report's results brief, given before ours. Show
   the brief without the collapsed block; then open ours and say where they differ. Record the reading. The user
   already saw the headline numbers in the run chat (2026-10-20 about 22:40), without our reading.
b. **"r10 decide 15"** was opened at 06:54 and had made no commit by 23:08 (its session was still active). Ask the
   user whether it is still open. If so, its points (stage 1's reading, the spec brief check, track 2's decision B,
   its wrap-up) stay there, and this chat does only stage 2's points; one wrap-up covers both. If not, take its
   open points here.
c. **The Leiden result, minor.** Under Leiden the emotion lead vanishes in both stages. The licensed claim names
   k-means with 20 clusters, so the verdict stands; the question is how the paper presents it and whether it shapes
   track 2's decision B or the next spec.
d. **The rule's next step (S2-GO):** the paper's sentence (spec §6) and ArtELingo-28's facts brief and spec. Decision
   A for that loop (type and size) is the user's.
e. **The wrap-up** (`research-loop.md`): decision C and the direction log entry, its "Now" paragraph (still says
   "stage 1 building and running overnight"); the storage summary with every row of `.scratch/pending_deletions.md`
   (`storage.md`), including `cluster du --node node408` for the node items (about 33 GB); a disk ledger row for
   `src/test/20261020_claim_test/outputs/stage2/` (4.6 GB, recommendation keep: the stage 2 report rests on it);
   `wrapup-check`, `disk-check`, `routing-check`; `/project/claude-config/sync.sh`, commit and push that repo.

## 5. Code and pitfalls

- The round folder is closed for stage 2: held outputs exist, so V4 refuses any retraining; do not edit `ct_*.py`,
  `ct2_*.py`, the queue scripts or `DECISION_RULE.md` (any edit breaks the stage 2a binding).
- JSON outputs are gitignored: `git add -f` for any new verdict or diagnostics JSON.
- `.scratch/r10-claim-test-s2/v2/` (194 MB) is the V2 evidence; on pending deletions with "keep until the decide chat
  has read the report".
- Cluster CLI: `du`, `watch` and `logs` need `--node node408` (the default node is node401).

## 6. State at handoff

- Running: nothing (no GPU lock, background job, Codex job, node job or subagent).
- Uncommitted: nothing in git. Outside git: `/project/claude-config/codex/jobs.md` (three new rows: V2, report draft,
  user-read draft), synced at the wrap-up.
- On disk over 1 GB: `src/test/20261020_claim_test/outputs/stage2/` (4.6 GB, keep; no disk ledger row yet); the
  items on `.scratch/pending_deletions.md` (node408 inputs about 33 GB, node smoke scratch 1.6 GB, old pulled results
  2.0 GB, ...).
