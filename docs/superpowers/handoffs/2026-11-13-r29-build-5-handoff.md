# Handoff: r29 build 5: fix 12 and 14, finish 11, review 10 and 11, build 13 and 15, final review, merge, open the run chat

## Brief
- Done: 05, 07, 08 fixed, re-reviewed and merged; integration fix (pass rule before test numbers) merged; 12 and 14 reviewed; 10 built; stage 0 gates (i) to (iii), (v) image and text pass.
- State: integration branch 591b459d, 360 tests pass; gate (iv) waits for its hash part (stage 1).
- Next: fixer for 12, smoke fix and re-review of 14, finish 11, reviews of 10 and 11, then 13 and 15.
- Blocked: nothing.

> Written 2026-11-13 01:45 by the "r29 build 4" tab (cut at the context meter's 200k note). Loop step reached: step 7
> (build) in progress. Direction log entry: [2026-11-12 · r29 decide](../direction_log.md).

**Flag:** ticket 05's fixer could not `git rm` the scratch-root record `records/gates/i_c_valpick.json` (the permission
check refused it), and the controller did not run it either. The record was merged with tries 2 and re-run on its real
root `gate_i_c/try1`: pass, max abs diff 0, now try 3 of 3. Decision line written (`decisions.md`, 00:36).

## 1. Read in this order

1. **The controller's open items**: `/project/CoSiR/.scratch/r29-siglip-only/notes/controller_todo.md`, section "r29
   build 4 progress" at its end: every ticket's state, the notes each step needs.
2. The first build handoff, sections 2 to 5: [2026-11-12-r29-build-handoff.md](2026-11-12-r29-build-handoff.md), and
   section 5 below.
3. The tickets and index: `/project/CoSiR/.scratch/r29-siglip-only/issues/00-index.md`. "Shared interfaces" gained
   (build 4) the crash records `records/crash_<arm>.json` and ticket 10's store gate ids `viii_<cap_source>`.
4. Notes (`/project/CoSiR/.scratch/r29-siglip-only/notes/`): `t12_review.md` with the ready fix brief
   `t12_fix_brief.md`; `t14_review.md`, `t14_fix_handback.md`, `t14_gpu_smoke.log`; `t10_handback.md`;
   `t11_handback.md` (progress note); `t13_brief.md`; the review template `review_common.md` and the scoped re-review
   briefs of build 4 (`t05_rereview_brief.md`, `t08_rereview_brief.md`) as patterns.
5. `~/.claude/references/matt-chain.md`, `~/.claude/references/codex-jobs.md`, `~/.claude/references/loop-chats.md`.

Skills: none by name; the chain is in `matt-chain.md`.

## 2. Decided by the user, not to be reopened

As in the first build handoff, section 2 (spec §4, D1 to D9 and D25). Run mode is night.

## 3. Where things stand

Integration branch `r29-siglip-only`, worktree `/project/CoSiR-r29`, pushed (591b459d): tickets 01 to 09, the import
fix, the integration fix (`r29_paths.require_committed`: tickets 04 and 06 refuse candidate test numbers without a
committed `pass_rule.json`), stage 0 records. Round suite: 360 passed (before 05's merge; run it again after the next
merges).

| Ticket | Branch, worktree | State |
|---|---|---|
| 01 to 09 | merged | 05 (f18e60f9), 07 (a03ab78c), 08 (8caadaa2): scoped re-reviews pass |
| 10 | r29/t10, /project/CoSiR-r29-t10, be235788 | built (59 tests; smoke joins pass on r14's SigLIP 2 and CLIP L/14 exports). Next: Opus `high` review. Its 9 decision lines are not in `decisions.md` yet (hand-back list). Its own three-try count in `gate_viii` predates 05's limit in `write_gate_record`: check for a double count |
| 11 | r29/t11, /project/CoSiR-r29-t11, 61b6e667 (WIP) | module written, 34 tests, 2 failing on a too-strong synthetic fixture. The worktree is clean at 61b6e667 (the cut-off implementer's fixture edit did not stay). Its last full run of `test_r29_mmart.py` hit its 230 s timeout after 26 of 34 tests (one F): the file is slow, so a subagent runs single tests with `-k` and the controller runs the whole file. Next: fresh Sonnet `high` implementer from `t11_handback.md`, then gates (v) MMArt and (vi) (CPU queue), Opus `high` review |
| 12 | r29/t12, /project/CoSiR-r29-t12, 577e25d5 | GPU smoke pass; Opus review: pass after fixes. Next: Sonnet `high` fixer with `t12_fix_brief.md` (ready), scoped re-review, merge |
| 13 | none | brief ready; needs 12 merged (and 08: merged) |
| 14 | r29/t14, /project/CoSiR-r29-t14, d082cf94 | Opus review pass after fixes; fixed (30 tests, gate (vii) ablation pass at try 2). GPU smoke FAILED in its K0 stand-in step (below). Next: fix, smoke again, scoped re-review, merge |
| 15 | none | after 10, 11, 13, 14 |

Stage 0 gates in `records/gates/` (merged `records/gates.json`): `i_a` pass (bit-equal to r14, no CUDA fallback),
`i_c` pass, `ii`, `iii`, `v_image`, `v_text` pass; `iv` lacks its `iv_hash` part (stage 1, after K1 and K2 train).
`vii_trainer` (ticket 12) and `vii_ablation` (ticket 14) are on their branches.

Next steps, in order:
1. Ticket 12 fixer (`t12_fix_brief.md`); ticket 14 smoke fix (below); ticket 11 finisher. Codex had no room all night
   (77 % left, an implementer job needs 80 %; window resets 04:44): check `codex-usage --room implementer` first.
2. Reviews: 10 and 11 (Opus `high`, verdict code), scoped re-reviews of 12 and 14 (Sonnet `high`). Merge each.
3. Ticket 13 after 12 merges (brief ready). Then 15 (plumbing: the controller's diff read and a dry run; the scripts
   never pass `--allow-uncommitted` or set `R29_TEXT_SMOKE`).
4. Final whole-branch review (Opus `xhigh`), one fix wave, scoped re-review; merge into main, push; run handoff; GPU
   check; `loop-next` (first build handoff, section 5, with r29's main chat rule).

## 4. Open points for the next step

None for the user (night mode). Agent defaults to carry into the run report: ticket 12's Affection SigLIP 1 store cuts
0.62 % (plain) or 0.77 % (prefixed) of explanations at 64 tokens (D14: cut and count; flagged in
`viii_affection.json`); SC7 is parked only by gate (vii)'s ablation part, SC6 by its photo parts (per-part parking;
alternative: any part parks both, the literal §3.5 reading; decision line on r29/t14).

## 5. Code and pitfalls

- **Ticket 14's smoke failure:** the removal training ran (2 epochs, finite losses), then `smoke_checkpoint`
  (`r29_ablation.py:374`) calls `T.held_inputs(B.backbone('siglip1'))`, and r14's `rb_encoders.full_images` asserts
  `siglip_so400m in WIDTH` (`rb_encoders.py:58`): the step misses r14's backbone registration or patch context. Fix,
  then run `res-queue gpu --short --label r29:t14-smoke -- ... r29_ablation.py train --row buddy --seeds 0 --smoke --k0-standin --device cuda` from the t14 worktree.
- **Gate status rule (05, 08, 14):** a `fail` before try 3 raises "still open"; `fail` at try 3 or `parked` parks; an
  absent record raises "not run". `write_gate_record` raises on a fourth recorded try. New tickets (11, 13) follow it.
- **Forced rewrites (05):** a `--force` keeps a record at its interface keys; the reason goes to `<name>_forced.json`.
- `decisions.md` conflicts at every merge: keep both sides (delete the three marker lines).
- Earlier pitfalls (older rounds import from the main checkout; the "BS_OUTPUTS is set" import clash for ticket 15's
  runners; ticket 12's edit of 02's `r29_extract.py`) hold as in the build 4 handoff, section 5.
- Run-chat note: R3 (remove buddy) should equal r28's buddy exports bit for bit (no caption gradient): a free check.
- Subagents' commits carry a Sonnet trailer; fine.

## 6. State at handoff

- Running: nothing of this chat.
- Uncommitted: nothing in the ticket worktrees. Ticket branches r29/t10, t11, t12, t14 are committed locally, not pushed.
- On disk over 1 GB: as in the build 4 handoff (`.scratch/pending_deletions.md`, r29 build 3 rows); build 4 added the
  worktrees `/project/CoSiR-r29-intfix`, `-t10`, `-t11`, the t12 GPU smoke (173 MB) and build 4's session scratchpad
  (2.1 GB on the system disk: reviewers' mutation copies), listed on that file.
