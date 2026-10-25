# Handoff: run the final whole-branch review of the encoders-and-genre build (three Opus `xhigh` reviewers, one per stage), one fix wave, the re-review, the merge into main; then open "r10 run 21"

## Brief
- Done: all 16 tickets merged (06 and 14 after fix rounds; 16 the queues). 275 tests pass; the stores and stage iii queue smokes pass, and VG2 agrees on its first and all phases.
- State: integration branch r10-encoders-genre at b4e11f0, pushed. The fresh-seed phase of stage iii has not run end to end.
- Next: the final review, three Opus xhigh reviewers by stage (brief written).
- Blocked: nothing.

> Written 2026-10-25 21:30 by "r10 build 21e" (cut at 200k tokens of context, at a clean point: no subagent, Codex
> job, GPU job or background job running). Loop step reached: step 7 (build), 16 of 16 tickets merged, final review
> next. Direction log entry: [2026-10-25 · Recipe and backbones round read](../direction_log.md).

## 1. Read in this order

1. This handoff, then the progress notes `.scratch/r10-build-21/notes/build_progress.md`, section "r10 build 21e
   progress": every result of this chat.
2. The final review brief `.scratch/r10-build-21/notes/final_review_brief.md` (written in this chat, for the three
   reviewers), and §4 a below for the split.
3. [2026-10-25-r10-build-21b-handoff.md](2026-10-25-r10-build-21b-handoff.md) §4 d and e (final review, merge, the run
   chat's handoff) and §5; the 21c handoff §5 and the 21d handoff §5 pitfalls; they still hold.
4. `~/.claude/references/matt-chain.md` ("The final whole-branch review"), `~/.claude/rules/agent-routing.md`,
   `~/.claude/references/codex-jobs.md`, and before the cut `~/.claude/references/loop-chats.md` ("How to cut").

Skills: `writing-for-agents` (briefs).

## 2. Decided by the user, not to be reopened

As the 21b handoff §2: the spec's §4 rows D1 to D17 marked "user" or "user, after seeing the recommendation", the
brief's approval at 18:35, the container restart at 18:44. This cut stays inside the build phase, so no restart
question was asked (as at the 21b, 21c and 21d cuts); the cut to the run chat asks again.

## 3. Where things stand

- **Integration branch** `r10-encoders-genre` at b4e11f0, pushed (worktree `/project/CoSiR-r10-int`). Merged in this
  chat: 06 (83ca842, after its fix round), 16 (a9cbdf0), 14 (b4e11f0, after its Codex fix round). All of 01 to 16 are
  done. The round's tests: 275 pass on b4e11f0 (4 min 50 s; log `/tmp/claude-0/eg_checks/tests_b4e11f0.log`).
- **06** (V2 wrapper, reporter): V2 re-derives every candidate on each run; `--candidates` only for scratch outputs;
  the reporter's stage 3 file gets a `stores` block (V5 and X0). V2 stage 5 parity on 12-24's K2 and K3 after the
  fix: 0 differences. The reporter has not run end to end on this round's files yet (it needs stage 3 outputs).
- **14** (VG2): reads the trainer's real formats; on the real R6-5 seeds 1 to 3 (run.json, sidecars, parity) Codex's
  read-only probe found 0 differences. Each of five operator mutations is killed by a test.
- **16** (queues): `eg_queue_{common,stores,stage_i,cpu_i,stage_ii,stage_iii,gpu}.sh` and `eg_queue.md`. Smokes:
  - stores queue: 8 extractions at `--limit 16` (widths 512, 768, 1152, 1024) and the V5 calibration (0.159, 0.296,
    as before);
  - stage iii queue at smoke scale, 19 min: five variants, `first`, VG2 first agrees, extras, `all`, VG2 all agrees,
    `report`. No variant was guarded, so the fresh seeds did not trigger;
  - not run: stage i, cpu_i and stage ii queues (dry runs only). The stage i smoke needs the real new stores, because
    V5 and the runners read `/data/SSD2/pre_extract`.
- **Gate files on the real outputs** (`/data/SSD2/cosir_r10/20261025_encoders_genre/outputs`): unchanged from the 21d
  handoff §3. Correction to the 21d handoff §4 d: `x{1,2}_scorer.json` do record `code_commit` (443c735, under
  `items.code`), and `eg_score.py`, `rb_score.py`, `rb_a28.py` have not changed since, so they need no rerun unless the
  fix wave touches those files. `stage_iii/vg1_scorer.json` (39f20da) still needs its rerun after the fix wave.
- **Codex:** five-hour window 100% left at 21:27 (reset 02:27), weekly 83%.

## 4. Open points for the build (agent work; nothing for the user)

a. **The final review**, three `worker`s at Opus `xhigh`, in parallel, each with the brief
   `.scratch/r10-build-21/notes/final_review_brief.md` and its stage (agent default: one reviewer for 10k lines
   would pass 200k tokens):
   - stage i and the cross-stage driver: `eg_paths`, `eg_encoders`, `eg_x0`, `eg_v5`, `src/data/feature_extract.py`,
     `scripts/extract_features.py`, `eg_student`, `eg_matched`, `eg_score`, `eg_v2`, `eg_report_i`, `eg_run.py` (all
     but the stage ii and iii gates), `eg_queue_{common,stores,stage_i,cpu_i,gpu}.sh`, `eg_queue.md`; rule §2.1, §3.1,
     §4.1, §5; spec §3.5, §3.6 stage i; tickets 01 to 06, 15, 16;
   - stage ii: `eg_pinfer`, `eg_lift`, `eg_vl2`, `eg_stats`, `eg_run.py` (vl1, vl2, lift-score),
     `eg_queue_stage_ii.sh`; rule §2.2, §3.2, §4.2; tickets 07 to 10;
   - stage iii: `eg_variants`, `eg_train3`, `eg_score3`, `eg_vg2`, `eg_run.py` (vg1, vg2, parity, score3),
     `eg_queue_stage_iii.sh`; rule §2.3, §3.3, §4.3; tickets 11 to 14. Ask it to trace the fresh-seed phase by hand
     (winner and R6-5 on seeds 1 to 3, parity, `eg_score3 fresh`, VG2 fresh): it has not run end to end (§3).
b. **One fix wave** (Codex has room; verdict code test-first), then a Sonnet `high` re-review of the fixes. Then rerun
   `eg_score3.py check vg1` (CPU, a few minutes; `/tmp/claude-0/eg_checks/cpu_checks.sh` line 2), and X1 and X2
   scorer checks only if the fix wave changes `eg_score.py`, `rb_score.py` or `rb_a28.py`
   (`/tmp/claude-0/eg_checks/cpu_x12.sh`, about 25 min; move the old files aside first: a regression file that exists
   is never rerun).
c. **Then** as the 21b handoff §4 d and e: merge into main, push, `ln -sfn` the `EG/outputs` link in the main
   checkout, the worktrees and branches on `.scratch/pending_deletions.md`, and the run chat's handoff with the
   ticket list, the queue order, the gates already passed, the end estimate against 11:00 Monday 2026-10-26 (spec
   D15: else start Monday 19:00) and the deferral order K5, K8, K7, K6.
d. **Into the run chat's handoff:** (1) after the real stores queue, a stage i smoke (`EG_SMOKE=1`, `eg_queue.md`
   "Smoke") before the real stage i, since only real stores make it possible; (2) the fresh-seed phase is proven only
   by tests and the probe: if VG2 fresh disagrees in the real run, stage iii stops before any C outcome is stated,
   and the run chat traces it; (3) the stage 3 reporter's first real run is also its first end-to-end run.

## 5. Code and pitfalls

- `/tmp/claude-0/r12_smoke` no longer serves VG2: its run.json files lack `p_sha256`, `code_commit` and, on the failed
  GA-1 run, `failure_kind`, which VG2 now requires. Use `/tmp/claude-0/eg16smoke/eg` (current formats; stage iii at
  smoke scale; its `stores/v5.json` holds only the calibration).
- A V2 parity tree on 12-24's outputs needs a copy of `part2/partitions/` (not only the `full` and `sample` links);
  without it every partition is "unchecked" and `agree` is False.
- `eg_queue_gpu.sh` goes on after a queue stops and exits 1 at the end; the cpu_i report queues run in the background
  after stage i, one at a time (`/tmp/eg_cpu.lock`). The stage iii `report` step took 514 s at smoke scale.
- The 21b, 21c and 21d pitfalls hold (zsh in the Bash tool: multi-step jobs in a bash script; pytest's temp folder
  `/tmp/pytest-of-root`; never `rm -r` through parity symlinks).

## 6. State at handoff

- Running: nothing (no subagent, Codex job, GPU lock or background job of this chat). GPU check at 21:27: driver
  595.99.02 answers, kernel module and library 595.99.02, GPU idle; container started 18:44.
- Uncommitted: nothing in git; tickets and notes are in `.scratch/` (outside git).
- On disk over 1 GB: new `/tmp/claude-0/eg16smoke` 4.4 GB (16's smokes; on `.scratch/pending_deletions.md`); the
  21d rows unchanged (`/tmp/claude-0/eg07e2e` 5.2 GB, 13's fixture 2.9 GB, `/tmp/claude-0/r12_smoke` 2.1 GB). The
  round's outputs are about 2.8 GB on `/data/SSD2`.
