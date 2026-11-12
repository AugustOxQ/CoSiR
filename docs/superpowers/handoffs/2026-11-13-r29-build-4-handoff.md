# Handoff: r29 build 4: fix and merge 05, 07, 08; review 12, 14; build 10, 11, 13, then 15, final review, merge, open the run chat

## Brief
- Done: import clash fixed; 02 merged; 05, 07, 08, 12, 14 built; 05, 07, 08 reviewed (pass after fixes); 07 fixed.
- State: stage 0 gates (i)(c), (ii), (iv) store, (v) image and (v) text pass on the local GPU and CPU; gate (i)(a) weights bit-equal to r14.
- Next: Codex fix of 05, fixer for 08, re-review of 07; merge; reviews of 12 and 14; then 10, 11, 13, then 15.
- Blocked: nothing.

> Written 2026-11-13 00:21 by the "r29 build 3" tab (cut at the context meter's 200k note). Loop step reached: step 7
> (build) in progress. Direction log entry: [2026-11-12 · r29 decide](../direction_log.md).

## 1. Read in this order

1. **The controller's open items**: `/project/CoSiR/.scratch/r29-siglip-only/notes/controller_todo.md`. It lists every
   fix round, review and run left for the tickets built so far, with the notes each needs. Work from it.
2. The first build handoff, sections 2 to 5: [2026-11-12-r29-build-handoff.md](2026-11-12-r29-build-handoff.md), and
   build 3's section 5 (below). Build 2's section 5 still holds for 05 and 10.
3. The tickets and index: `/project/CoSiR/.scratch/r29-siglip-only/issues/00-index.md`. "Shared interfaces" now fixes
   the `prefix_pick.json` keys (`kstar`, `kstar_form`, `p1_form` = `plain` | `prefixed`, `flag`) and the cell record
   names, comparator names, gate ids and ladder record names (build 3).
4. Notes (`/project/CoSiR/.scratch/r29-siglip-only/notes/`): reviews `t05_review.md`, `t07_review.md`,
   `t08_review.md`; hand-backs `t05_handback.md`, `t07_handback.md`, `t07_fix_handback.md`, `t08_handback.md`,
   `t12_handback.md`, `t14_handback.md`; briefs ready to launch: `t10_brief.md`, `t11_brief.md`, `t13_brief.md`
   (their "holds tickets" line names what the integration branch must hold first).
5. `~/.claude/references/matt-chain.md`, `~/.claude/references/codex-jobs.md`, `~/.claude/references/loop-chats.md`.

Skills: none by name; the chain is in `matt-chain.md`.

## 2. Decided by the user, not to be reopened

As in the first build handoff, section 2 (spec §4, D1 to D9 and D25). Run mode is night.

## 3. Where things stand

Integration branch `r29-siglip-only`, worktree `/project/CoSiR-r29`, pushed (94bcf178): tickets 01, 02 (with
review fixes), 03, 04, 06, 09, the import fix (`r29_paths.use_older_rounds()` and `check_older_imports()`: older rounds
import from the main checkout), main merged in (r28's last commits). Round suite: 193 tests pass in one process.

| Ticket | Branch, worktree | State |
|---|---|---|
| 01, 03, 04, 06, 09 | merged | 03's Codex fix re-reviewed: pass (`t03_rereview.md`) |
| 02 | merged (66e7a79f, fc5cda07) | fix re-reviewed: pass; the CLI `check` now needs `--plain` or `--expected-prefix` |
| 05 | r29/t05, /project/CoSiR-r29-t05, 6735ba47 | Codex built (64 tests); Opus review: pass after fixes. Fix by Codex resume (todo 3) |
| 07 | r29/t07, /project/CoSiR-r29-t07, 2145391d | reviewed (pass after fixes), fixed (56 tests, gate (v) text pass, `v_text.json` tries 2); scoped re-review next (todo 5) |
| 08 | r29/t08, /project/CoSiR-r29-t08, 549748a1 | built (56 tests); Opus review: pass after fixes (gate parking per §3.5); fresh fixer next (todo 4) |
| 12 | r29/t12, /project/CoSiR-r29-t12, 577e25d5 | built (31 tests + 02's 35); gate (vii) trainer part pass; GPU smoke and review next (todo 7) |
| 14 | r29/t14, /project/CoSiR-r29-t14, 83c1d35a | Codex built (22 tests); gate (vii) scorer replay, GPU smoke and review next (todo 6) |
| 10, 11, 13 | none | briefs ready; 10 and 11 need 08 merged, 13 needs 12 and 08 |
| 15 | none | after 05, 08, 09 (stage 0, 1) and 10, 11, 13, 14 (stage 2) |

Stage 0 gate records committed in the integration worktree (`records/gates/`): (i)(c) pass (local GPU, `gate_i_c/try1`),
(ii) pass (max abs diff 0.0), (iii) pass, (iv) store part pass (SigLIP 1 prefixed store
`/data/SSD2/pre_extract/artelingo_full/siglip_so400m_capprefix`, 308,723 captions, 0 cut, 714 MB), (v) image pass
(full replay, max abs diff 0, 36 min CPU). (v) text pass is on branch r29/t07. Gate (i)(a) weights part pass: the new trainer (caption width 2048, Qwen3) and r14 rerun (r14, r14_b) are bit-equal on the local GPU, 3 heads (`records/gates/i_a_weights.json`); the verdict `i_a` is ticket 05's `gate-i-a` after its merge. Still to
run: gate (iv) hash part (`r29_text.py hash-check`, after K1 and K2 train: stage 1), ticket 05's `gate-i-c-valpick`
on `gate_i_c/try1` (after 05's merge), and the `gates` merge (ticket 05).

Next steps, in order (detail in `controller_todo.md`):
1. Codex resume for 05's fix (session id in the first line of `notes/t05_codex.log`; `codex-usage --room implementer`
   first); a fresh Sonnet `high` fixer for 08; a Sonnet `high` scoped re-review of 07's fix. Then the re-reviews of
   05 and 08, and merge 05, 07, 08 (decisions.md conflicts: keep both sides).
2. Todo 1: one integration fixer so ticket 04's held scoring and ticket 06's `cells` refuse without a committed
   `pass_rule.json`.
3. Ticket 12: GPU smoke, Opus `high` review, fix, merge. Ticket 14: gate (vii) scorer replay (CPU queue), GPU smoke,
   Opus `high` review, fix, merge.
4. Tickets 10, 11 (after 08 merges) and 13 (after 12 merges): briefs ready; Codex for one when it has room; reviews
   Opus `high`. Then 15 (plumbing: controller's diff read and a dry run).
5. Final whole-branch review (Opus `xhigh`), one fix wave, scoped re-review; merge into main, push; run handoff; GPU
   check; `loop-next` (first build handoff, section 5, with r29's main chat rule).

## 4. Open points for the next step

None for the user (night mode). One agent default to carry into the run report: ticket 12's Affection SigLIP 1 store
cuts 0.62 % (plain) or 0.77 % (prefixed) of explanations at 64 tokens; D14 says cut and count; it is flagged in
`viii_affection.json`.

## 5. Code and pitfalls

- **Older rounds import from the main checkout** in every r29 module (`r29_paths.use_older_rounds()`), so a worktree
  run uses main's r28 and r14 code. A new module calls it and `check_older_imports()`; never its own
  `sys.path.insert` of an older round. Ticket 12 and 14 were built after the fix; 05, 07 and 08 too.
- A plain `python -c "import r29_time, r29_text, r29_image_cells, r29_trainer"` raises "BS_OUTPUTS is set"
  (`ns_photo.py:35`), from before the fix; pytest is unaffected. Ticket 15's runners must not import both in one
  process (todo 2).
- Ticket 12 edited ticket 02's `r29_extract.py` (`scheme=` for Affection keys): check the merge against 02's tests.
- Ticket 07's `cells` refuses unless `pass_rule.json` and `prefix_pick.json` are committed and `kstar` names its arm.
- Ticket 08's ladder records are written once; a ladder run before its cell files exist must be refused (its review).
- Codex ledger rows written: t02 fix (23:40), t05 (23:54), t14 (00:11).
- GPU smokes of 12 and 14 hit exit 75 while this chat's stage 0 jobs ran; the queue is free after 00:20.

## 6. State at handoff

- Running: nothing of this chat.
- Uncommitted: nothing of r29 after this handoff's commit. Ticket branches r29/t05, t07, t08, t12, t14 are committed
  locally, not pushed.
- On disk over 1 GB: the test and scratch folders listed on `.scratch/pending_deletions.md` (r29 build 3 rows, about
  8.5 GB, 1.4 GB of it on the system disk); the new prefixed store (714 MB, kept: K2's input).
