# Handoff: finish the r31 build: review and merge ticket 02 (SC9), run its replays, build ticket 03, final review, merge; then open "r31 run"

## Brief
- Done: tickets written (3); ticket 01 (the A1 flag) merged into r31-stage2; ticket 02 (SC9 encoder pick) built by Codex, 174 tests pass, not yet reviewed.
- State: integration branch r31-stage2 pushed; nothing running; GPU check passed 16:16.
- Next: Opus review of ticket 02, merge, SC9 replays on the CPU, ticket 03, final review, fixes, merge to main; open "r31 run" (stage 2 from 19:00).
- Blocked: nothing.

> Written 2026-11-13 16:17 by "r31 build" (cut at the 200k context meter). Loop step reached: step 7, after ticket 01's
> merge and ticket 02's build. Direction log entry: [2026-11-13 · r29 read](../direction_log.md).

## 1. Read in this order

1. The first build handoff, [2026-11-13-r31-build-handoff.md](2026-11-13-r31-build-handoff.md): §2 (decided), §4 (what
   to build), §5 (pitfalls), §6 (what the run chat's handoff must say). It still holds.
2. The tickets: `/project/CoSiR/.scratch/r31-stage2/issues/` (00-index with the shared interfaces, 01 done, 02, 03).
3. Ticket 02's hand-back: `/project/CoSiR/.scratch/r31-stage2/notes/t02_handback.md` (record keys, choices, tests).
4. `~/.claude/references/matt-chain.md` and `~/.claude/references/codex-jobs.md`.

Skills: `implement-spec`, `tdd` (for any verdict-code fix), `writing-for-agents` (briefs).

## 2. Decided by the user, not to be reopened

As in the first build handoff §2: A1 (stage 2 on the fail; `pass_rule.json` stays "fail"; K* an ablation or
supplementary row), A2 (SC9, r28's rule as agent default), stage 2 on the local GPU from 19:00 (D7), order D24.

## 3. Where things stand

- **Integration branch `r31-stage2`** (worktree `/project/CoSiR-r31`, pushed, head 3ea7edbc):
  - fa032409: `records/amendment_A1.json`, the committed A1 record (written by this chat from spec §4 row A1).
  - 39f43480: ticket 01 merged. `r29_paths.stage2_gate(records)` answers "pass" or "A1" or raises; every stage 2 check
    (encoders, trainer, ablation, MMArt, photo, photo cells, ladder SC4 to SC6) uses it. The runner takes
    `--amendment A1` and then puts `R29_AMENDMENT=A1` on every Python step. Ladders, the sheet and the SC7 record carry
    "stage 2 ran on the fail under A1". Dry runs checked by the controller: without the flag a real run refuses; with it
    all 25 part a and 13 part b commands carry the variable. Tests: paths 23, ladder 110, ablation 36, encoders 64,
    MMArt 44, photo 56, photo cells 48, trainer 40, all pass (worker's run).
  - 3ea7edbc: decision lines of tickets 01 and 02 in the round's `decisions.md`.
- **Ticket 02** (verdict code): branch `r31-t02` (worktree `/project/CoSiR-r31-t02`, pushed), commit 9fba1550 by Codex
  (`gpt-6.1-sol` high, session thread 01a1d570-aea7-7aa2-be74-f05debb325b8, ledger row added). Changes
  `r29_decide.py` (encoder-aware `image_groups`; `sc9-replay`, `image-groups`, `encoder-pick`), `r29_encoders.py`
  (`kstar_source` takes a records folder), `test_r29_decide.py`. Not reviewed, not merged. Its branch base is fa032409,
  before ticket 01; `r29_encoders.py` is touched by both (expect a clean merge, check it).
- **Ticket 03** (runner wiring, sheet SC9 section): not started; blocked by 02.

## 4. Steps left (the run chat needs all of them before 19:00 if it can)

1. **Review ticket 02** with an Opus `high` worker (verdict-code ticket review): the diff against ticket 02 and spec §6
   SC9; it re-derives the pick rule on a small example and checks that `image_groups` with SigLIP 1 is unchanged. Fix
   round by resume of the same Codex thread (`codex-jobs.md`, "One fix round by resume"); a second failure goes to Claude.
2. **Merge** r31-t02 into r31-stage2; run `test_r29_decide.py` and `test_r29_encoders.py` on the merged branch.
3. **Run the two SC9 replays** from `/project/CoSiR-r31/src/test/20261112_siglip_only`, in parallel, in the background:
   `res-queue cpu --est 45m --label r31:sc9-replay-<E> -- env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
   MKL_NUM_THREADS=8 /root/miniconda3/envs/CoSiR/bin/python r29_decide.py sc9-replay --encoder <E>` for `siglip2` and
   `clip_l14` (validation only; no stage 2 gate needed). Both must pass (within 1e-6 of r28's `valpick.json`: SigLIP 2
   0.3855510963, CLIP L/14 0.3756306625). Commit the two gate records with `git add -f`. A fail: diagnose (code bugs
   only, three tries, `diagnosing-bugs`); a gate that stays failed parks SC9 (inconclusive) and the run goes on.
4. **Ticket 03** (plumbing): Codex if `codex-usage --room implementer` says room (`gpt-6.1-terra` medium), else a
   Sonnet `medium` worker; worktree from r31-stage2 after the merge. Controller reads the diff and checks both dry runs.
5. **Final whole-branch review** (Opus `xhigh` worker; `matt-chain.md`): spec §4 A1, A2, §6 SC9, hard rule 6, the
   constitution. Then one fix wave and a Sonnet `high` re-review. Merge r31-stage2 into main locally, push main.
6. List the worktrees `/project/CoSiR-r31`, `-r31-t01`, `-r31-t02` (and t03's) and their branches on
   `.scratch/pending_deletions.md`; the tracker `.scratch/r31-stage2/` goes at the wrap-up.
7. **Write the run chat's handoff** (first build handoff §6 lists what it says), with the ticket list (title, blocked
   by, spec IDs, verdict code) from `00-index.md`. Add for the run chat: start part a as
   `r29_run_stage2.sh a --amendment A1` (from the main checkout after the merge) at 19:00 or later; run the ladders and
   `r29_ladder.py sheet` with `R29_AMENDMENT=A1` in the environment; commit `encoder_pick.json` with the other picks and
   have Codex re-derive it (C13) before part b. GPU check, then `loop-next <handoff> --label "r31 run" --unattended`;
   commit `chats.tsv` and `briefs.md`.

## 5. Code and pitfalls

- First build handoff §5 holds (records gitignored: `git add -f`; never edit `pass_rule.json`; weekday 10:00 to 19:00
  refusal of long GPU work; new code in the r29 round folder with the `r29_` prefix).
- Run mode is night: no questions to the user; take a named default and write a decision line.
- Ticket 01's worker log `/project/CoSiR-r31-t01/.claude/20261113_r31_t01_log.md` is uncommitted (`.claude` is
  gitignored); the change log rule wants it: add it with `git add -f` under the main checkout's `.claude/` at the merge,
  or copy its content into `.claude/20261113_log.md`.
- Ticket 02 notes: r28's tie band (1e-12, no pick) was replaced by the spec's exact tie to SigLIP 1; runner-up ties
  use the fixed encoder order. `r29_run_stage0.sh` and `stage1.sh` ignore `--amendment` silently (harmless).
- The `subagent-watch --watch` background check exits at once when only Codex runs (it watches Claude subagents).

## 6. State at handoff

- Running: nothing (queue idle; Codex done).
- Uncommitted: nothing of the build in git; the tracker `.scratch/r31-stage2/` is outside git by design.
- On disk over 1 GB: nothing new (r29 outputs 5.8 GB as before).
- Remote branches pushed by this build: `r31-stage2`, `r31-t02` (list both on `pending_deletions.md` after the merge).
