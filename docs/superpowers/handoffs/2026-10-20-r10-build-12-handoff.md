# Handoff: finish the build (final review items 7 and 8, one fix wave, re-review, merge to main), unattended; then open the run chat

## Brief
- Done: tickets 06 to 09 built, reviewed, fixed and merged; final review half done, provisional PASS (every re-derived number matches).
- State: 300 tests pass; the GPU smoke passes all nine steps; nothing trained or read on held data yet.
- Next: final review items 7 and 8, one fix wave (three low findings), re-review, merge to main, open "r10 run 5".
- Blocked: nothing (Codex has 19% until 07:01, so fixes go to Claude).

> Written 2026-10-20 03:27 by "r10 build 11" (cut at the context meter's 200k note). Loop step reached: track 1
> (claim test), step 7 (build), all nine tickets merged on `r10-claim-test`, final review half done. Direction log
> entry: [2026-10-20 · Stage 2 model sheet read](../direction_log.md), bullet "Grill, claim test".

## 1. Read in this order

1. [The r10 build 10 handoff](2026-10-20-r10-build-10-handoff.md) §1 to §7 (spec, rule, user decisions, pitfalls,
   and §7, what to pass on to the run chat) and [the r10 build 11 handoff](2026-10-20-r10-build-11-handoff.md) §5.
2. This file.
3. The final review so far: `.scratch/r10-claim-test/notes/final_review.md`, and its brief
   `.scratch/r10-claim-test/jobs/final_review.txt` (items 7 and 8 are left).
4. `src/test/20261020_claim_test/ct_queue.md` in the integration worktree `/project/CoSiR-wt/integration`: how the
   run chat runs stage 0, stage 1 and the CPU queue.
5. `~/.claude/references/matt-chain.md` ("The final whole-branch review") and `~/.claude/references/loop-chats.md`
   ("How to cut").

Skills: none named (the chain's final review, fix wave and merge).

## 2. Decided by the user, not to be reopened

- As in the r10 build 10 handoff §2 (every grill answer; night mode and the overnight go-ahead; node408 allowed).
- Agent defaults made in this chat (not the user's): a missing stage 0 training-code record with seed 0 files present
  counts as stale (stricter than the brief); a non-finite paper Stage 1 (DEC) stays a loud crash, since R-COL needs
  finite centers; the held instructed images go to the shared cache, as the ticket index says.

## 3. Where things stand

| Ticket | State | Review |
|---|---|---|
| 01 to 05 | merged in build 10 | see the build 11 handoff §3 |
| 06 held instructed images (`ct_extract.py`) | merged | diff read; GPU parity check: minimum cosine 1.0 on 64 paintings × 3 instructions |
| 07 paper settings PercepT (`ct_paper_sept.py`, `ct_percept.py`) | merged | Opus `high` PASS; 2 fixes (counts and topics from one fit Q, Stage 2 width against Stage 1) |
| 08 diagnostics, Leiden (`ct_diag.py`) | merged | diff read; adds an optional r20-only reading set to `ct_score.paired_bootstrap` (verdict default unchanged) |
| 09 stage 0 gate, queues, smoke (`ct_stage0.py`, `ct_run.py`, `ct_queue_*.sh`, `ct_smoke.py`) | merged | Opus `high` PASS; 7 fixes (training-code identity, CUBLAS env, gate tests, score tolerance 5e-7, shared image cache, wiring tests, budget note) |
| V4 fix (paper non-finite Stage 2 and held outputs saved, flagged, counted collapsed) | merged | Sonnet `high`, five new tests red then green |

- Integration branch `r10-claim-test` at d1e4c84, pushed. Full round suite: 300 passed, 2 skipped (4.5 min). GPU
  smoke on 79cde2c: 9 of 9 steps PASS (only a test fixture changed after it).
- Final review (Opus `xhigh`, items 1 to 6 done) re-derived with its own code: the held list (12,281 paintings,
  SHA-256 a0b1139c…a876), the held labels, the development AMIs of D2 and PercepT core seeds 0 to 2 (difference 0),
  the bootstrap (all 2,000 draws equal), the four outcomes S1-GO, S1-MM, S1-IN, S1-NO on synthetic cases, and R-COL
  on full-size synthetic artifacts. All match.
- Its findings, all low, none changes a verdict:
  1. V1 is enforced only through `ct_run.py`; `ct_score.py`, `ct_diag.py`, `ct_d2.py --held` and
     `ct_percept.py --stage held` run without a stage 0 pass.
  2. `require_pass` ties the pass to the host name (the container id); a recreated container fails closed.
  3. `ct_run.py verdict` has no `--replace` pass-through for a rerun forced by a failed V check (rule V4).
  4. and 5. Notes only (the R-COL share near 0.5; npz hashes are deterministic).
- Ticket list for the run chat's handoff (the record of matt-chain "to-tickets" step 4): the table in
  `.scratch/r10-claim-test/issues/00-index.md` plus its coverage line; copy both into the run chat's handoff.

## 4. Open points for the next step

None for the user. Order:
1. A fresh Opus `xhigh` worker finishes the final review: item 7 (mutation checks on the verdict modules), item 8
   (each queue step at the real sizes; `ct_diag` against its 30-minute budget, timed on CPU where it can be), the
   per-file tests; it appends to `final_review.md` and gives the final verdict. Brief: `final_review.txt` with a
   note that items 1 to 6 are done.
2. One fix wave (verdict code: Sonnet `high`; Codex has no room until 07:01): findings 1 to 3 and any from step 1.
   Suggested: `ct_stage0.require_pass()` at the top of the `ct_score.py` and `ct_diag.py` mains and of the held modes
   of `ct_d2.py` and `ct_percept.py`; a `--replace "<reason>"` pass-through in `ct_run.py verdict`; one line in
   `ct_queue.md` on the host-name tie.
3. A scoped re-review of the fixes (Sonnet `high`), the full suite and the GPU smoke under the lock.
4. Merge `r10-claim-test` into main locally, push main; list the worktrees `/project/CoSiR-wt/integration` and
   `ct-01` to `ct-10` and the branches `r10-ct-01` to `r10-ct-10` on `.scratch/pending_deletions.md`.
5. Write the run chat's handoff (pass on build 10 §7, the ticket list, `ct_queue.md`'s order, the timings below),
   run the GPU check, open "r10 run 5" with `loop-next ... --label "r10 run 5" --unattended`.

## 5. Code and pitfalls

- **Every fix lands before stage 0.** `ct_stage0.code_identity()` hashes the round's code; any later edit to a
  covered file fails `require_pass` and forces a stage 0 rerun, which retrains seed 0 (about 15 min).
- The run chat runs from `/project/CoSiR` after the merge: outputs land in the main checkout's round folder, and the
  modules' default paths point there.
- Timings (ticket 09 review, from facts §9): stage 1 GPU about 1.1 to 2.5 h (D2 5 × 0.6 min, core 5 × 3 min, paper
  0.5 to 1.7 h, held inference minutes, instructed images about 0.5 h); CPU verdict and diagnostics not measured.
  The queue's budget step prints a fixed upper bound (3 GPU h, 1.5 CPU h).
- `subagent-watch` printed "No running subagents" while worker subagents ran; rely on their completion notices.
- Codex: 5 h window 19% left, reset 07:01; weekly 46% left. Ledger row for 06 to 09 added in `claude-config`.

## 6. State at handoff

- Running: nothing (no Codex job, subagent, GPU lock or background job). GPU check at 03:26 passed (driver
  595.99.02 in all three places); night mode.
- Uncommitted: nothing in git; review notes and job logs in `.scratch/r10-claim-test/` (outside git).
- Worktrees: `/project/CoSiR-wt/integration` (r10-claim-test), `ct-01` to `ct-10` (ticket branches).
- On disk over 1 GB: `src/test/20261019_model_sheet/outputs/`, 3.4 GB, kept (disk ledger);
  `/project/CoSiR-wt/ct-09/.scratch/`, 1.6 GB of Codex smoke output, on `.scratch/pending_deletions.md`. The two GPU
  smoke folders (1.1 GB) in this session's scratchpad were deleted (smoke rule).
