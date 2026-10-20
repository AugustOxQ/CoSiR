# Handoff: finish the stage 0 and stage 1 build (tickets 06 to 09, final review, merge), unattended; then open the run chat

## Brief
- Done: tickets 01 to 05 of the claim test build, each reviewed and merged: held list and labels, collapse rule, held scorer and bootstrap, D2 and PercepT core with saved weights.
- State: nothing trained or read on held data yet; integration branch `r10-claim-test` pushed.
- Next: tickets 06 to 09 (held instructed images, paper settings PercepT, diagnostics, stage 0 and queues), final review, merge.
- Blocked: nothing.

> Written 2026-10-20 02:27 by "r10 build 10" (cut at the context meter's 200k note). Loop step reached: track 1
> (claim test), step 7 (build), tickets 01 to 05 merged. Direction log entry:
> [2026-10-20 · Stage 2 model sheet read](../direction_log.md), bullet "Grill, claim test".

## 1. Read in this order

1. [The r10 build 10 handoff](2026-10-20-r10-build-10-handoff.md): sections 1 to 7 still hold (what to read, the
   user's decisions, pitfalls, and what to pass on to the run chat). Read it first, then this file.
2. The ticket index `.scratch/r10-claim-test/issues/00-index.md` (shared interfaces, rules, coverage line) and tickets
   06 to 09 in the same folder. Status lines: 01 to 05 `done`.
3. The merged modules in the integration worktree `/project/CoSiR-wt/integration/src/test/20261020_claim_test/`:
   `ct_held.py`, `ct_rcol.py` (its docstring fixes the paper Stage 1 NPZ keys for ticket 07), `ct_score.py`,
   `ct_d2.py`, `ct_percept.py`.
4. `~/.claude/references/matt-chain.md` and `~/.claude/references/codex-jobs.md`.

Skills: none named (the chain is `to-tickets` done, `implement-spec` in progress, then the final whole-branch review).

## 2. Decided by the user, not to be reopened

- As in the r10 build 10 handoff §2 (every grill answer; night mode and the overnight go-ahead; node408 allowed).

## 3. Where things stand

| Ticket | State | Branch, merged commit |
|---|---|---|
| 01 held list, labels, loaders | done; Opus high review PASS; one assert added (every caption has an emotion) | r10-ct-01 |
| 02 cuts, R-COL | done; review PASS, five fixes (topic rule stated, `centers` and `fit_latent` keys, count cross-check, refuse overwrite, one output path); controller gave the count cross-check a 0.01% near-tie tolerance | r10-ct-02 |
| 03 held scorer, bootstrap, verdict | done; review FAIL, then fixed (R-COL keys, repeated-seed test, partition identity, verdict refuses replacement without `--replace`) | r10-ct-03 |
| 04 D2 weights, held inference | done; review FAIL (CPU defaults), then fixed (device from the weights, CUDA default, strict resume, batch-size test) | r10-ct-04 |
| 05 PercepT core weights, held inference | done; review FAIL (CPU default; core NPZ lacked `centers`, `fit_latent`, `kept_k40`, `fit_topics_k40`), then fixed; all 188 round tests pass on it | r10-ct-05 |
| 06 to 09 | not started; worktree `/project/CoSiR-wt/ct-06` exists (from the integration branch before 02 to 04 merged: run `git merge r10-claim-test` in it first) | |

- Codex task files for 06 to 09 are ready: `.scratch/r10-claim-test/jobs/task_NN.txt` (common rules in
  `notes/brief_common.md`); review brief for verdict code: `notes/review_brief.md`. Ticket 07's first criterion was
  extended: core and paper Stage 1 save the NPZ keys of `ct_rcol.py`'s docstring.
- Codex 5 h window: 63% left at 02:26, reset 07:01; weekly 53% left. Each implementer job used about 5 to 8%.
- Measured: one AMI on 12,281 rows takes about 6 ms; the review estimates the bootstrap at 2 to 3 min on 8 workers.

## 4. Open points for the next step

None for the user. Order: 06 and 07 now (07 is blocked by 02 and 05, both merged); 08 after 03 (merged); 09 last.
Verdict-code tickets (07, 09) get an Opus `high` review before merge; 06 and 08 a diff read and the smoke run.

## 5. Code and pitfalls

- Every review so far found CPU defaults: check each new CLI's `--device` (training and held inference must use the
  device recorded with the weights; full runs refuse CPU).
- Saved sheet and core outputs are gitignored: modules read them from the main checkout `/project/CoSiR/...`; the
  round's outputs also live in the main checkout (`ct_rcol.STAGE1`, `ct_score` derives from it). The run chat runs
  from `/project/CoSiR` after the merge to main.
- `ct_score` keys saved partitions on `pav_score.py`, `apc_data.py` and the partition function's source; a verdict is
  never replaced without `--replace "<reason>"` (rule V4).
- Codex fix rounds by resume: `codex e --dangerously-bypass-approvals-and-sandbox ... -C <worktree> resume <thread_id> - < fix.txt`
  (the command must start with `codex e`).

## 6. State at handoff

- Running: nothing (no Codex job, subagent, GPU lock or background job). GPU check at 01:52 passed (driver 595.99.02, CUDA available); night mode, so no restart question.
- Uncommitted: nothing in git; the tickets and Codex logs are in `.scratch/r10-claim-test/` (outside git).
- Worktrees: `/project/CoSiR-wt/integration` (r10-claim-test), `ct-01` to `ct-06` (ticket branches); listed on
  `.scratch/pending_deletions.md` at the end of the build, not before.
- On disk over 1 GB: `src/test/20261019_model_sheet/outputs/`, 3.4 GB, kept (disk ledger).
