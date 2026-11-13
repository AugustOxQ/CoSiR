# Handoff: r29 build 6: ticket 11 fixes and gate (vi), ticket 13 gate and review, final review, merge, open the run chat

## Brief
- Done: 10, 12, 14, 15 merged (12 and 14 GPU smokes pass); 11 and 13 built; gates v_mmart and v_photo pass.
- State: integration branch 0a6678dd (pushed); 11 and 13 on their branches; gate (vi) score runs on CPU until about 04:30.
- Next: 11 review fixes and merge; 13 gate (vii) photo, review, merge; final review, fix wave, gate (vii) ablation try 3, merge into main, run chat.
- Blocked: nothing.

> Written 2026-11-13 02:40 by the "r29 build 5" tab (cut at the context meter's 200k note). Loop step reached: step 7
> (build) in progress. Direction log entry: [2026-11-12 · r29 decide](../direction_log.md).

**Two user decisions in this chat** (decision lines on the integration branch, 0a6678dd), for the run chat:
1. GPU runs may run in weekday daytime this round: stage 2 starts right after a stage 1 pass, with
   `R29_ALLOW_DAYTIME=1` for `r29_run_stage2.sh` (amends D7 for r29 only). Expected: stage 1 about 06:00 to 10:00,
   stage 2 about 10:00 to 16:00, reports in the evening.
2. r29's run goes before r21's next GPU job: before stage 1, the run chat messages r21's run chat ("CoSiR r21 run 3",
   or its successor; `ListAgents`) to hold its next run, queues r29's steps, then tells r21 it may go on. Quote the
   user: "if you want GPU, [tell the] r21 chat to cease the next run and queue yours first".

## 1. Read in this order

1. The controller's open items: `/project/CoSiR/.scratch/r29-siglip-only/notes/controller_todo.md`, sections "r29
   build 4 progress" and "r29 build 5 progress" at its end.
2. The first build handoff, sections 2 to 5: [2026-11-12-r29-build-handoff.md](2026-11-12-r29-build-handoff.md), and
   section 5 below.
3. The tickets and index: `/project/CoSiR/.scratch/r29-siglip-only/issues/00-index.md`.
4. Notes (`/project/CoSiR/.scratch/r29-siglip-only/notes/`): `t11_review.md` (pass, no must-fix),
   `t11_handback.md`, `t13_handback.md`,
   `final_review_brief.md` (ready), `review_common.md`, and the scoped re-review briefs of build 5
   (`t12_rereview_brief.md`, `t14_rereview_brief.md`) as patterns.
5. `~/.claude/references/matt-chain.md`, `~/.claude/references/codex-jobs.md`, `~/.claude/references/loop-chats.md`.

## 2. Decided by the user, not to be reopened

As in the first build handoff, section 2 (spec §4, D1 to D9 and D25), plus the two decisions above. Run mode is night.

## 3. Where things stand

Integration branch `r29-siglip-only`, worktree `/project/CoSiR-r29`, pushed (0a6678dd): tickets 01 to 10, 12, 14, 15,
stage 0 records. Round suite: not run since build 4 (360 passed then); run it once after 11 and 13 merge
(`res-queue cpu`; `test_r29_mmart.py` alone takes 2.7 min).

| Ticket | Branch, worktree | State |
|---|---|---|
| 01 to 10, 12, 14, 15 | merged | 10: Opus review pass after fixes, fix diff read. 12: re-review pass, GPU smoke 2 pass. 14: re-review pass, GPU smoke 2 pass. 15: diff read, dry runs |
| 11 | r29/t11, /project/CoSiR-r29-t11, 2362802e | done; 38 tests pass (whole file); gate `v_mmart` pass (try 1); gate (vi) export part bit-equal (30 exports in `OUT/gate_vi/try1`, 1.5 GB, keep). Gate (vi) score `--try 1` RUNS (res-queue cpu, label `r29:t11-gate-vi`, est. end 04:32, log `notes/t11_gate_vi_score.log`; writes `records/gates/vi.json` in the t11 worktree: commit it with `git add -f`). Opus high review: pass, no must-fix (`t11_review.md`). Next: should-fixes (step 2), merge |
| 13 | r29/t13, /project/CoSiR-r29-t13, 52729f57 | done; 35 tests; 3 of 3 mutations; gate `v_photo` pass (try 1). Next: gate (vii) photo (below), Opus high review (verdict code; brief in the pattern of `t11_review_brief.md`), merge |

Next steps, in order:
1. Check the gate (vi) job: `res-queue status`; `tail notes/t11_gate_vi_score.log`. If it died with the build 5 tab
   (no `vi.json` written, no job in the queue), re-run it with the same command (`t11_handback.md`); no try was used.
2. Ticket 11: the Opus review passed with no must-fix (`t11_review.md`: ranges re-derived exactly; 6 of 10 mutations
   caught by tests, 2 only by gate (vi), 2 by nothing). A Sonnet high fixer (pattern `t10_fix_brief.md`) for its three
   should-fixes: a test that `run_cells` refuses to overwrite; the untold pick's unit stored in the picks record
   (`r29_mmart.py:324`) with a test that it equals the `map_ranges` r; `--allow-uncommitted` refused outside smoke
   (`:666`, `:432`). Plus the decision lines the review lists. Then merge. The fix must wait until the gate (vi) job
   has written `vi.json` (it reads `r29_mmart.py`); then check whether the fix changes anything gate (vi) checks
   (exports carry `r29_code_sha256` of the module, `:211`). If it does, gate (vi) score is re-run as try 2. Also set
   ticket 15's stage 0 `gate-vi score` estimate from `--est 45m` to the measured time.
3. Ticket 13: gate (vii) photo, first `--no-record`, then recorded (command in `t13_handback.md`, about 25 min CPU,
   `res-queue cpu`); an Opus high review; fixes; merge.
4. Check ticket 15's ticket 13 block and MMArt and photo steps against the merged CLIs (the implementer says the
   names match).
5. Final whole-branch review (Opus `xhigh`, `notes/final_review_brief.md`), one fix wave (Sonnet high), scoped
   re-review (Sonnet high); then gate (vii) ablation try 3 (below); merge into main, push; run handoff; GPU check;
   `loop-next` (first build handoff, section 5, with r29's main chat rule).

## 4. Open points for the next step

None for the user (night mode). Agent defaults to carry into the run report: as in the build 5 handoff (ticket 12's
D14 truncation; per-part parking of SC6 and SC7), plus ticket 13's 8 choices (on its branch's `decisions.md`) and
ticket 11's (its hand-back).

## 5. Code and pitfalls

- **Gate (vii) ablation is stale on purpose.** Its record (try 2, pass) predates the t14 smoke fix and ticket 05's
  merge; its code hash covers `r29_ablation.py`, `r29_trainer.py`, `r29_model.py`, `r29_captions.py`. Re-run it once
  after the final fix wave, from `/project/CoSiR-r29`:
  `res-queue cpu --label r29:gate-vii -- env PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/CoSiR/bin/python r29_ablation.py gate-vii`
  (about 2.3 min; try 3 of 3; commit with `git add -f`). After it those files are frozen: any edit leaves SC7 stale.
  Fixers' briefs must say so (build 5 briefs did).
- **Gate status rule** (all tickets now): `pass`; `fail` before try 3; `parked` at try 3; only
  `r29_captions.write_gate_record` counts a try; an absent record raises "not run".
- **Smoke roots:** a smoke's `run__s<seed>.json` refuses a rerun after a code change; give a fresh `--root` under
  `SMOKE_OUT` instead of deleting (night mode).
- `decisions.md` conflicts at every merge: keep both sides (delete the three marker lines).
- Ticket 15: stage 2a refuses weekdays 10:00 to 19:00 unless `R29_ALLOW_DAYTIME=1` (the user allowed it, above).
  Steps whose record exists are skipped (`R29_FORCE=1` overrides). Its estimates marked "guess" have no timing.
- Codex had no room all night (72 % left at 02:30; an implementer job needs 80 %; the window resets 04:44).
- Earlier pitfalls (older rounds import from the main checkout; "BS_OUTPUTS is set"; ticket 12's edit of 02's
  `r29_extract.py`) hold as in the build 4 handoff, section 5. Run-chat note: R3 (remove buddy) should equal r28's
  buddy exports bit for bit (a free check).

## 6. State at handoff

- Running: gate (vi) score of ticket 11 (res-queue cpu, started 02:32 from this tab's background shell; see step 1).
  Nothing else of this chat.
- Uncommitted: nothing in the ticket worktrees. Ticket branches r29/t11, t13 are committed locally, not pushed.
- On disk over 1 GB: `.scratch/pending_deletions.md` (build 5 rows): ticket 12 fixer's test scratch
  `/project/CoSiR-r29-t12/.scratch/r29-t12-tests/` (4.6 GB, system disk), the smokes of 12 and 14 (each under 0.4 GB).
  `OUT/gate_vi/try1` (1.5 GB) is a gate input: keep.
