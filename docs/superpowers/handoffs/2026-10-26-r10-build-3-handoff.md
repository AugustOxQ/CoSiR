# Handoff: finish the relational genre build (review 10, re-review 06, fix 04, then 09, 11, gate checks, final review, merge, run handoff)

## Brief
- Done: tickets 01, 02, 03, 07, 08 reviewed and merged (141 tests); 06 fixed after its review; 04 and 10 built; 05 built on 04 (its smoke matches 12-24's R6-5 readings).
- State: the real checks pass: VA1 scorer 16 of 16 and VB1 15 of 15 against 12-25, VC1 scorer 25 of 25 against 12-24 (all at 0.0); VC2's K4 check agrees.
- Next: Opus review of 10, re-review of 06, fixer for 04; then 09 (Codex), 11, the gate checks, the final review.
- Blocked: nothing. No container restart (user, 20:01, for this build).

> Written 2026-10-26 20:54 by "r10 build 2" (cut at 200k context, all subagents returned). Loop step reached: step 7
> (build), 6 of 11 tickets merged. Direction log entry: [2026-10-26 · Encoders and genre round read](../direction_log.md).

## 1. Read in this order

1. This handoff.
2. [2026-10-26-r10-build-2-handoff.md](2026-10-26-r10-build-2-handoff.md) §3 to §5 (routing, Codex room rule, frozen
   inputs, pitfalls) and the first build handoff [2026-10-26-r10-build-handoff.md](2026-10-26-r10-build-handoff.md)
   §2 to §5. They still hold; this file adds only the state.
3. `/project/CoSiR/.scratch/r10-relational-genre/notes/build_progress.md`: every hand-back and review finding in
   order, with the findings still to fix (04, the minors of 02).
4. The ticket index `.scratch/r10-relational-genre/issues/00-index.md` and the tickets still open (04, 05, 06, 09,
   10, 11); `notes/implementer_brief.md` and `reviewer_brief.md` (give their paths to every implementer and reviewer).
5. `~/.claude/references/matt-chain.md`, `codex-jobs.md` and `~/.claude/rules/agent-routing.md` before the first
   dispatch.

Skills: none beyond those `matt-chain.md` names.

## 2. Decided by the user, not to be reopened

- As the build handoffs: the spec's D1 to D20, M1 to M10, the brief approved at 19:57; no container restart for this
  build (user, 20:01).

## 3. Where things stand

Integration branch `r10-relational-genre` at 5b9310f (worktree `/project/CoSiR-r10-rg`, pushed): rg_loss (01),
rg_train (02), rg_score stage A (03), rg_held with `report` (07, 08); 141 tests pass on the round folder.

| # | State | Branch, worktree | Next step |
|---|---|---|---|
| 01, 02, 03, 07, 08 | done, merged | — | — |
| 04 | built 9484d19; review FIX (findings in the notes) | r10-rg-t04, `.claude/worktrees/agent-a6636457bc89d2f28` | Sonnet high fixer, then merge |
| 05 | built 32af0be, tip dac9391 on r10-rg-t04 (smoke: stage A and loss scale; R6-5 readings equal 12-24's; report code in rg_report.py so verdict hashes stay put) | r10-rg-t05, `.claude/worktrees/agent-aeb11f056ffa03bb7` | controller diff read; merge after 04's fix (it carries 04) |
| 06 | fixed after review FIX: d3ce1be, tip 63d0496 (47 tests) | r10-rg-t06, `.claude/worktrees/agent-a1728c488afe0c9cb` | scoped re-review (Sonnet high) of d3ce1be, then merge |
| 09 | not started (after 04) | — | Codex high (re-derivation; 74% of the 5-h window left at 20:47, reset 01:12) |
| 10 | built 234b119, tip 4fd7937 (14 tests; `--check-k4` agrees, max diff 6.9e-18) | r10-rg-t10, `/project/CoSiR-r10-t10` | Opus high review, then merge |
| 11 | not started (after all) | — | Sonnet medium |

Checks already run (scratch outputs unless named): `check va1` in the real outputs (PASS 16/16, now stale: code
4a536ba, before 03's fix); `check vb1` (PASS 15/15, `/tmp/claude-0/rg_t04_out`); `check vc1` (pass, 25 items,
`/tmp/claude-0/rg_t07_out`); `rg_vc2 --check-k4` (agree); 06's `train --arm GQ-8 --smoke` on the GPU (exit 0, anchors
4,096 of 42,969 fit paintings, the anchor and dev-count guards held); 08's `report --smoke` with K4 in G's place
(every point estimate equals 12-24's K4 block).

## 4. Open points for the build (agent work; nothing for the user)

a. The 04 fix (the notes list five findings; the first is should-fix: a stale `vb1.json` cannot be refreshed). Apply
   the same check to 03's `check va1`: the real `regression/va1_scorer.json` is stale; move it aside, never delete.
b. The final review's fix wave also takes 02's five minors and aligns every module's dirty flag with rg_held's
   (`git status --porcelain -- '*.py'` in the round folder).
c. The gate checks in the real outputs, **after the final review's fix wave** (the records bind to the code hash):
   `regress va1` (GPU), `eq`, `check va1`, `check vb1`, `regress vc1` (GPU, full-scale R6-5 seed 0), `check vc1`; the
   smokes 02 (`train --smoke --arms GQ-8 GN-8`, check `term_means`: 2 entries, pairs 5,000, calls 10; `lossscale
   --smoke`), 04 (`b --smoke`), 06 (`held --smoke`). GPU work under the lock, one at a time.
d. Then the final whole-branch review (Opus `xhigh`), one fix wave, the scoped re-review (Sonnet `high`), merge into
   main, push, worktrees and ticket branches on `.scratch/pending_deletions.md`, the run chat's handoff (build
   handoff §4 e) and `loop-next <handoff> --label "r10 run" --unattended` after the GPU check.
e. For the user, in the run handoff and the report: the rule's line 48 says a batch is 1,024 pairs (2,048 rows); the
   frozen trainer uses 512 pairs (1,024 rows), which VA1's bit-for-bit match needs. The code is right; the report
   must not repeat the rule's figure.

## 5. Code and pitfalls

- **Code-hash gates:** rg_score's `check va1` and `check vb1` records, rg_held's `vc1_scorer.json` and rg_full's VC1
  code identity (G's weights must carry the code hashes of the VC1 run) all bind to the code. Any edit to a round
  module after a check means rerunning it; any edit to rg_loss, rg_variants or rg_full after VC1 means rerunning
  `regress vc1` and retraining G.
- **Reviewers mutate in scratch copies** under `/tmp/claude-0/` (the auto-mode check refused in-place edits with a
  `git checkout` restore).
- **Ticket 06 run notes:** `held` needs all five seeds trained and refuses a partial seed list; `vc1_train.json` must
  be a non-smoke run; failure files (`failed__s{seed}.json`, `failed__<aspect>__s{seed}.json`) are the record and
  are never cleared. A held-output VC3 failure gets one in-memory inference rerun from the same weights (agent
  default, confirmed by 06's review: training is deterministic).
- **Ticket 04:** `--failed` takes only new arms; `b` needs a non-smoke, same-code `vb1.json`.
- Codex: one implementer at a time; a re-derivation needs 20% of the five-hour window.

## 6. State at handoff

- Running: nothing (no subagent, Codex job, GPU lock or background job of this chat).
- Uncommitted: nothing of this build (this handoff is committed with this cut; `.scratch/` is outside git).
- On disk over 1 GB: unchanged. Real outputs 4.3 MB. Scratch under `/tmp/claude-0/` (smoke and check trees, 241 MB
  for `rg06_smoke`, the rest under 20 MB each): delete at the build's end (smoke rule).
