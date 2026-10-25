# Handoff: finish the encoders-and-genre build (review 05, fix 11, check 07; tickets 06, 09, 10, 13 to 16; gates; final review; merge into main), then open "r10 run 21"

## Brief
- Done: tickets 01, 02, 03, 04, 08 and 12 reviewed and merged; X0 passes, and the student and matched halves of X1 and X2 pass on the GPU.
- State: 05 built (scorer checks on 12-24 pass, diff 0), its review not yet run; 11 review FIX (one missing refusal); 07 built, not checked. Six tickets not started.
- Next: Opus review of 05, fix round of 11, check of 07; then 13, 06, 09 (Codex after 21:26).
- Blocked: nothing.

> Written 2026-10-25 19:41 by "r10 build 21b" (cut at 200k tokens of context, at a clean point: no subagent, Codex
> job or GPU job running). Loop step reached: step 7 (build), 6 of 16 tickets merged. Direction log entry:
> [2026-10-25 · Recipe and backbones round read](../direction_log.md).

## 1. Read in this order

1. This handoff, then the progress notes `.scratch/r10-build-21/notes/build_progress.md`: per ticket, the result,
   the gate results, the review findings still open, and interface notes. They are the record for the run handoff.
2. The handoff before this one, [2026-10-25-r10-build-21b-handoff.md](2026-10-25-r10-build-21b-handoff.md) §2 to §5
   (decided items; waves; GPU work during the build; final review; the run chat's handoff); they still hold. Its
   §5 pitfalls hold too.
3. The ticket index `.scratch/r10-build-21/issues/00-index.md` (`run.json` now also holds `p_sha256`), and the two
   briefs this chat used: `notes/implementer_brief.md` (every implementer prompt points to it) and
   `notes/reviewer_brief.md` (every verdict-code review). Tickets 13 and 15 carry controller notes at their end.
4. `~/.claude/references/matt-chain.md`, `~/.claude/rules/agent-routing.md`, `~/.claude/references/codex-jobs.md`.

Skills: `tdd` (named in verdict-code briefs, which point to its SKILL.md), `writing-for-agents` (briefs).

## 2. Decided by the user, not to be reopened

As the 21b handoff §2 says: the spec's §4 rows D1 to D17 marked "user" or "user, after seeing the recommendation",
the brief's approval at 18:35 and the container restart at 18:44. This cut stays inside the build phase, as the cut
before it did, so no restart question was asked (GPU check below); the cut to the run chat asks again.

## 3. Where things stand

- **Integration branch** `r10-encoders-genre` at 69b02cc, pushed (worktree `/project/CoSiR-r10-int`). Merged:
  01 (6425e0b), 12 (6c0392e), 03 (867013a), 04 (f78ff0e), 02 (a07ce06), 08 (69b02cc). Status `done` in each ticket.
- **Gate files written in the build** (under `/data/SSD2/cosir_r10/20261025_encoders_genre/outputs`):
  `stores/x0.json` pass; `part2/regression/x1_student.json`, `x2_student.json`, `x1_matched.json`,
  `x2_matched.json` pass. Not yet written: `x1_scorer.json`, `x2_scorer.json` (ticket 05), `stores/v5.json` (the
  run chat, after extraction), `stage_ii/vl1.json` and `vl1_scorer.json`, `stage_iii/vg1.json`,
  `vg1_scorer.json`, `fresh_parity.json`.
- **Built, not merged** (each in its agent worktree under `/project/CoSiR/.claude/worktrees/agent-<id>/`, branch
  `worktree-agent-<id>`):

  | Ticket | Branch id, last commit | What is left |
  |---|---|---|
  | 05 stage i scorer | af50f926112c2d9a7, 327d8cc | Opus `high` review (reviewer brief), then merge; then the real `--check x1` and `--check x2` into the round's outputs (CPU, cold cache about 17 and 8 min) |
  | 11 stage iii trainer | a3b603828f1d44efd, d039da9 | fix round (finding 1 and minors 1, 2 in the notes), then merge; then `regress vg1` (GPU, about 6 min) and `parity --seeds 1 2 3` (GPU, about 18 min) |
  | 07 stage ii P inference | a4a2e9e6100019778, 6e5a9f6 | diff read; add the sidecar keys ticket 08 now requires; VL1 (b), (c) on the GPU into a scratch outputs folder; then merge |

- **Not started:** 06 (ready once 05 merges), 13 (ready: 12 is merged), 09 (ready: 08 is merged), 10 (after 06),
  14 (after 13), 15 (after 05, 07, 11), 16 last.
- **Codex:** five-hour window 76% at 19:10 (reset 21:26); an implementer job needs 80%. After the reset, Codex can
  take 09 (kind `rederivation`, 10%) and one plumbing ticket (10, 15 or 16) if `codex-usage --room` says room.

## 4. Open points for the build (agent work; nothing for the user)

a. **Checks still to run in the build** (short, one GPU job at a time under the lock): 05's two scorer checks into
   the real outputs, then B1 and B6 on ArtELingo-28 at the sample scale (`eg_matched.py --scale sample --arm B1`,
   then `B6`; gated on all three `x1_*`); 08's `eg_lift.py check` (VL1 a, about 90 s CPU); 12's `eg_score3.py check
   vg1` (about 40 s CPU); 11's VG1 and parity; 07's VL1 (b), (c).
b. **Ticket 07 and the VL4 plan lock:** `eg_pinfer.py infer|all` on the real outputs writes the arm plan
   (`stage_ii/p/index.json`) and locks it. Before the K5 to K8 and M5 to M8 weights exist, those arms would stay
   "missing" for good. In the build, run VL1 (b), (c) only into a scratch `--outputs`, or add a VL1-only mode.
c. **07 to 08 interface:** the scorer requires sidecar keys `p_sha256`, `arm`, `seed`, `aspect`, `rows` 29232,
   `width` 2048 (students) and `p_sha256`, `arm`, `seed`, `rows` 29232, `width` 40 (own arms). 07 writes
   `p_sha256`, `weights_sha256`, `store`, `batch` and more: add what is absent, and smoke the pair end to end (08's
   smoke reads full-P exports; 07's smoke cuts P to 1,024 rows).
d. **Notes for later tickets** (also in the progress notes): 10 and 15 state no stage ii outcome before
   `rederive_vl2.json` passes; 09 uses `prob / float32 norm` for the own-arm r20 vectors; 15 runs `eg_score3
   first|all` only after VG1 passes (it does not refuse by itself); 06 or 15 decides how a kept block without an
   outcome is replaced after a fix (05's route is `--replace REASON`); 16's queue must not pass `--rerun-failed`
   twice (until 11's fix lands).
e. **Then** the final whole-branch review, one fix wave, a Sonnet `high` re-review, the merge into main, the
   `EG/outputs` link in the main checkout, the worktrees and ticket branches on `.scratch/pending_deletions.md`,
   and the run chat's handoff (21b handoff §4 d and e).

## 5. Code and pitfalls

- Implementer prompts stay short: ticket path, "follow `notes/implementer_brief.md`", the code map sections, verdict
  code or not, and the interface facts of sibling tickets. A fix after a pause of over 5 minutes goes to a fresh
  `worker` pointed at the existing worktree.
- Subagents run as an unattended chat: a hook refuses `rm`; they leave scratch under `/tmp/claude-0/`, listed on
  `.scratch/pending_deletions.md`.
- Each regression run left its training folder under `outputs/part2/regression/` (`x1_run`, `x2_run`, 356 and
  383 MB; `x1_matched_run`, `x2_matched_run`, 7 MB). Keep them until the run report; they back the gate files.

## 6. State at handoff

- Running: nothing (no subagent, Codex job, GPU lock or background job of this chat). GPU check at 19:41: driver
  595.99.02 answers, GPU idle.
- Uncommitted: nothing in git; tickets and notes are in `.scratch/` (outside git).
- On disk over 1 GB: `/tmp/claude-0/r12_smoke` (2.1 GB, ticket 12's fake runs, on the system disk; listed on
  `.scratch/pending_deletions.md`). The round's outputs folder is 774 MB on `/data/SSD2`.
