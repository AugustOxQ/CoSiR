# Handoff: finish the encoders-and-genre build (fix rounds of 06 and 14, ticket 16, gates; final review; merge into main), then open "r10 run 21"

## Brief
- Done: tickets 09, 10 and 13 merged (13 of 16). X1 and X2 scorer checks pass on the real outputs; B1 and B6 sample-scale ArtELingo-28 inference done.
- State: 06 and 14 reviewed FIX (numbers right; 06 keeps stale entries and lacks the V5 and X0 section, 14 fails on the real file formats). 16 not started.
- Next: Codex fix of 14 by resume and a Sonnet fixer for 06, in parallel; then 16.
- Blocked: nothing.

> Written 2026-10-25 20:45 by "r10 build 21d" (cut at 200k tokens of context, at a clean point: no subagent, Codex
> job, GPU job or background job running). Loop step reached: step 7 (build), 13 of 16 tickets merged. Direction log
> entry: [2026-10-25 · Recipe and backbones round read](../direction_log.md).

## 1. Read in this order

1. This handoff, then the progress notes `.scratch/r10-build-21/notes/build_progress.md`, section "r10 build 21d
   progress": every result of this chat, and the two review finding lists with their fix plans (06, 14).
2. [2026-10-25-r10-build-21b-handoff.md](2026-10-25-r10-build-21b-handoff.md) §2 to §5 (decided items, waves, GPU work
   in the build, final review, the run chat's handoff); they still hold. The 21c handoff's §5 pitfalls hold too.
3. The ticket index `.scratch/r10-build-21/issues/00-index.md`, ticket 16 with its controller notes, and the briefs
   `notes/implementer_brief.md`, `notes/reviewer_brief.md`.
4. `~/.claude/references/matt-chain.md`, `~/.claude/rules/agent-routing.md`, `~/.claude/references/codex-jobs.md`.

Skills: `tdd` (named in verdict-code briefs), `writing-for-agents` (briefs).

## 2. Decided by the user, not to be reopened

As the 21b handoff §2: the spec's §4 rows D1 to D17 marked "user" or "user, after seeing the recommendation", the
brief's approval at 18:35, the container restart at 18:44. This cut stays inside the build phase, so no restart
question was asked (GPU check below); the cut to the run chat asks again.

## 3. Where things stand

- **Integration branch** `r10-encoders-genre` at 2a6d765, pushed (worktree `/project/CoSiR-r10-int`). Merged in this
  chat: 13 (447a32f, after a fix round), 09 (ca537d0, plus test fix 0b12961), 10 (2a6d765). Controller fix b8b8b5a in
  ticket 04's `eg_matched.require_gate`: V5 is asked only for a new encoder's own arm (B1 is Qwen's).
  Done: 01, 02, 03, 04, 05, 07, 08, 09, 10, 11, 12, 13, 15.
- **Gate files on the real outputs** (`/data/SSD2/cosir_r10/20261025_encoders_genre/outputs`): new in this chat
  `part2/regression/x1_scorer.json` and `x2_scorer.json`, both pass (every stage 3, 4, 5 outcome equals 12-24's), so X1
  and X2 hold on all three halves. Also new: `part2/sample/a28/B1/P3__s{0..4}` and `B6/P2__s{0..4}` (GPU, 20:31 and
  20:33). Earlier ones as the 21c handoff §3. Still the run chat's: `stores/v5.json`, `stage_ii/vl1.json`.
- **Built, not merged** (finding lists and fix plans in the progress notes):

  | Ticket | Branch, last commit | What is left |
  |---|---|---|
  | 06 V2 wrapper, reporter | `worktree-agent-abf77ee30f3056eef` 431a583, worktree `/project/CoSiR/.claude/worktrees/agent-abf77ee30f3056eef` | Opus review FIX, findings 1 to 5: a fresh Sonnet `high` fixer in that worktree; then rerun the V2 stage 5 parity (about 30 s, commands in the notes) and read the new stores section on the real `x0.json`; merge |
  | 14 VG2 (Codex) | `r10-t14-vg2` 9c08391, worktree `/project/CoSiR/.claude/worktrees/codex-t14` | Opus review FIX, findings 1 to 6: Codex fix round by resume, thread `01a17476-8a70-75e0-976c-a5b5957103b1`; then rerun `/tmp/claude-0/eg_checks/vg2_smoke.sh` (phases first and all on `/tmp/claude-0/r12_smoke`); merge |

- **Parity of 06 before its fixes:** all pass against 12-24's frozen files on K2 and K3 (V2 stages 3 and 5, reporter
  stages 3 and 5: 0 differences). 14's smoke before its fixes: 0 differences in numbers, partitions and outcomes;
  the only differences were keys that r12_smoke's older run records lack.
- **Not started:** 16 (after 06 and 14 merge; Sonnet `medium`).
- **Codex:** five-hour window 43% left at 20:44 (reset 21:26); a rederivation fix round needs 40%.

## 4. Open points for the build (agent work; nothing for the user)

a. **Next, in parallel:** the Codex resume fix of 14 (brief from finding list "14 review" in the notes; tell it to
   read the real `run.json`, `fresh_parity.json` and sidecars under the real outputs' `stage_iii/` before writing
   fixtures); a fresh Sonnet `high` fixer for 06 (finding list "06 review").
b. **Ticket 16**, adding to its controller notes: B1 and B6 sample ArtELingo-28 outputs already exist (the runner
   skips them); call `eg_v2.py --stage N` without `--candidates`; run `eg_train3.py train --rerun-failed` before
   `score3 first`, because `eg_score3.read_run` refuses a VG3 failure that was not rerun; its smoke runs the stage iii
   chain at smoke scale through `fresh`, VG2 fresh and `report` (this is where 14's fresh phase is first proven); run
   `replace` once (move stale partition caches aside first).
c. **Then** the final whole-branch review, one fix wave, a Sonnet `high` re-review, the merge into main, the
   `EG/outputs` link in the main checkout, the worktrees and branches on `.scratch/pending_deletions.md`, and the run
   chat's handoff (21b handoff §4 d and e).
d. **Into the final fix wave:** `x{1,2}_scorer.json` record no `code_commit` (ticket 05's writer; every other gate file
   records one): add it, then rerun `eg_score.py --check x1` and `--check x2` (about 12 and 13 min CPU, script
   `/tmp/claude-0/eg_checks/cpu_x12.sh`). `stage_iii/vg1_scorer.json` ran on 39f20da; `eg_score3.py` changed since
   (13's `read_run`): rerun it with the other CPU gates after the fix wave (21c handoff §4 e).

## 5. Code and pitfalls

- The 21b and 21c pitfalls hold (the Bash tool runs zsh: put multi-step jobs in a bash script; `eg_pinfer.py` without
  `--smoke` locks the stage ii plan; `--replace` needs `--candidates` and stale partition caches moved aside).
- New: pytest's temporary folder is `/tmp/pytest-of-root`, outside `/tmp/claude-0`; a test that calls a runner with
  `smoke=True` must point `eg_paths.SMOKE_ROOT` at its `tmp_path` (09's test failed this way after the merge). Codex
  set `PYTEST_DEBUG_TEMPROOT` and passed; always rerun a Codex ticket's tests in the integration worktree after the
  merge.
- 06's parity trees `/tmp/claude-0/eg06_parity_*` link into 12-24's frozen outputs: never `rm -r` through them.
- The machine was loaded (load about 20) while three CPU checks ran beside the subagents; 12-24's stage 3 reporter
  for one candidate took 22.5 min under that load.

## 6. State at handoff

- Running: nothing (no subagent, Codex job, GPU lock or background job of this chat). GPU check at 20:44: driver
  595.99.02 answers, kernel module and library 595.99.02, GPU idle; container started 18:44.
- Uncommitted: nothing in git; tickets and notes are in `.scratch/` (outside git).
- On disk over 1 GB: unchanged from the 21c handoff (`/tmp/claude-0/eg07e2e` 5.2 GB, 13's fixture 2.9 GB,
  `/tmp/claude-0/r12_smoke` 2.1 GB, all on `.scratch/pending_deletions.md`); this chat added about 80 MB of scratch,
  also listed. The round's outputs are about 2.8 GB on `/data/SSD2`.
