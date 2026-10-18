# Handoff: run the r10 PercepT core queues (stage 0 and the head rows), verify, report, then open the decide chat

## Brief
- Done: the build. Tickets 01 to 09 are merged into main. The final review (Opus `xhigh`) re-derived every load-bearing number; its fix wave passed the re-review.
- State: stage 0 fails on this machine (emotion 0.1238 vs recorded 0.1225, genre 0.2617 vs 0.2274). Cause: the recorded values come from the DAS6 node node404. This machine reproduces its own September record bit for bit.
- Next: run both queues (about 1 hour): stage 0, then the 24 head rows. Verify, report, open the decide chat.
- Blocked: the five PercepT arms, until the user decides on stage 0 (spec §3.5).

> Written 2026-10-19 01:32 by "r10 build 3". Loop step reached: step 8 (run) next; the build (step 7) is done.
> Direction log entry: [2026-10-18, aspect graph follow-up](../direction_log.md).

## 1. Read in this order

1. This handoff.
2. `src/test/20261018_percept_core/pc_queue.md` (the queues) and the round log
   `src/test/20261018_percept_core/20261018_percept_core_log.md`.
3. The spec [2026-10-18-percept-core-design.md](../specs/2026-10-18-percept-core-design.md): §3.4 (measures), §3.5
   (gates), §6 (established, licensed claims), §9 (deliverables).
4. The stage 0 trace: `src/test/20261019_stage0_trace/20261019_stage0_trace_log.md`.
5. `~/.claude/references/loop-chats.md` ("Unattended chats", "How to cut"); `report-kinds.md`,
   `user-read-reports.md` and `~/.claude/templates/results_brief.md` before the reports.

Skills: none.

## 2. Decided by the user, not to be reopened

- The spec's §4 (D1 to D15) and its approval, 2026-10-18 23:34, including the stage 0 gate and its failure action
  (§3.5: "stop stage 1, trace the cause, note it for the user; stage 2 still runs"). No container restart (23:34).
- Night mode since 2026-10-18 23:40 ("switch to overnight-run mode. I am going to sleep."): no questions.
- Exploration framing: "as many trials as possible to enlarge the ablation table and then pick".

## 3. Where things stand

### Tickets (all merged into main)

| # | Ticket | Blocked by | Spec IDs | Verdict code | Checks |
|---|---|---|---|---|---|
| 01 | Frozen-sample pair store and feature readers (`pc_data`) | none | D1, D2, D8, §3.3, C6 | yes | Opus review; fix: RoBERTa reader returns an array |
| 02 | RoBERTa-GoEmotions per-caption extraction (`pc_roberta`) | 01 | D1, D8, §3.7 | no | GPU smoke max abs diff 0 vs September; store written |
| 03 | The four fixes (`pc_fixes`) | none | §3.2 fixes 1-4, D5, D9-D11, §6 tests | yes | Opus review; fix: sign rule for the PCA and QR maps |
| 04 | Stage 1 runner (`pc_stage1`) | 01, 03 | §3.2 Stage 1, D2, D5, D7, D9 | yes | Opus review; three test gaps fixed |
| 05 | Stage 2 runner (`pc_stage2`) | 01, 03, 04 | §3.2 Stage 2, D3, D4, D7, D12, §6 | yes | Opus review; CPU retrain matches GPU dev AUC to 1e-7 |
| 06, 06b | Scoring, row 1 and the sheet (`pc_score`) | 01 | §3.4, D6, D13, D15, §3.5 row 2 | yes | Opus reviews; unit-test line gates "established" |
| 07 | Rows 6, 10, 13 and the head gate (`pc_rows`) | none | §3.2 rows, D13, §3.5 gate | yes | Opus review; fix: sd only with 3 seeds |
| 08 | Stage 0 and the queues | 02, 04-07 | §3.5 stage 0, D7, D14, §9 | no | end-to-end smoke, every step exit 0 |
| 09 | Final-review fix wave | 08 | F1, F2, review items 3-5 | part | Sonnet `high` re-review: merge, 6 of 6 mutants caught; 128 round tests pass |

Agent defaults set at ticketing (index items 1 to 6, `.scratch/r10-percept-core/issues/00-index.md`): P2's CLIP
vector is the feature store's image embedding; global Stage 2 inputs l2-normalized, patch tokens as stored; Stage 1
trains on fit pairs, Stage 2 on fit paintings with the dev AUC on dev paintings; E for P1/P2 is RoBERTa PCA to 512 on
fit captions, for P5 a fixed orthonormal 2048 x 768 map; row 1 caption side is the l2-normalized painting mean; the
head rows are 24 runs plus the gate rerun (the spec's budget says 21).

### The final review (2026-10-19, Opus `xhigh`)

Fix first, two queue fixes; the numbers are right. Its own CPU script matches `pc_score.py` to 1e-10 on P1 and P2
seed 0 (image side, both K-native AMIs, the K table, the kept mask). P2 seed 0 image side: emotion 0.06339, style
0.25000, genre 0.41167 (timing run, not a result). No positional joins; no evaluation painting in training, selection
or the dev AUC. Ticket 09 fixed: the `gate.json` race (F1), the sheet waits for the GPU queue (F2), the queue doc (F3),
the scoring test fixture, and the K < 2 cause on the sheet. Review files: `.scratch/r10-percept-core/review_tmp/`.

- Merged: `r10-percept-core` into `main` at `778db99`, pushed; `/project/CoSiR` is on `main`. 128 round tests pass
  (71 s).

### Stage 0

| Check | Recorded | This machine | Status |
|---|---|---|---|
| September held-out emotion AMI (both scripts) | 0.1225 | 0.1238 | FAIL |
| September held-out genre AMI (both scripts) | 0.2274 | 0.2617 | FAIL |
| Seed 42 Stage 2 macro AUC | 0.8296 | none (the script stops at its own re-fit check) | FAIL |
| Row 2, our scoring | 0.088556 / 0.170954 / 0.393076 | equal | PASS |
| Four-fix unit tests | pass | pass | PASS |

The trace (Codex `gpt-6.1-sol` high, read-only, plus a controller GPU test and checks):

- The recorded values come from runs on node404 (RTX A6000 in October), 2026-09-23 and 24. Commit `3b3ce61` rebased
  the scripts' expected values to that fit. Five node404 fits in September were identical (deterministic per machine).
- This machine gave 0.1238 / 0.2617 in September too: the local sweep report of 2026-09-23 02:19 (before any node
  launch), with pretraining losses equal to today's at all ten printed epochs.
- The scripts and every module they import are byte-identical. The saved node404 snapshot re-scores to 0.122469 /
  0.227419 today. The thread cap is ruled out (default threads give the same numbers).
- The genre AMI rests on 159 held-out paintings; one seed moves it by 0.034 between the two GPUs.
- node404 could not be used: no reservation exists on it (the user holds node406 and node408), and agents do not pick
  nodes. So the spec's gate holds: the GPU queue blocks all Stage 1 and Stage 2 runs of PercepT core.

### Topic count (spec §8 case, not a bug)

Full-size timing runs, seed 0: clip_roberta K = 99 (largest gap at the top, 7.75 x the mean gap), qwen_qwen K = 1
(largest gap at the bottom, 7.06 x), the same at every γ. Spec: report it, no rerun with another γ. The sheet now
gives the gap and its position for each K < 2 seed.

## 4. Open points for the next step

None for the user tonight. The run chat's steps:

1. **GPU check** (night mode: the check only): `nvidia-smi` answers; module and library name the same driver; no
   process and no lock on the GPU.
2. **Start both queues from `/project/CoSiR`, as two Bash calls with `run_in_background: true`, the GPU one first:**
   - `bash src/test/20261018_percept_core/pc_queue_gpu.sh --foreground --lock-wait 14400`
   - `bash src/test/20261018_percept_core/pc_queue_cpu.sh`

   Not the `&` form of `pc_queue.md`: the chat must get a notice when each ends. The CPU queue's sheet step waits for
   the GPU queue's `PC_gpu_DONE` (`gpu_wait=ok` in the status file).
3. **What to expect.** GPU queue about 9 minutes: stage 0 FAIL again (8 min), RoBERTa skipped, every Stage 1 and
   Stage 2 run `blocked`. CPU queue about 50 minutes: graphs, the gate (verified and skipped, PASS, difference 0),
   three aspect workers (rows 6, 10, 13; 24 head runs, 260 s each on 8 threads), the rows score, the sheet. Started
   near 02:00, it ends near 03:00, long before 11:00.
4. **Results:** `outputs/pc_rows_tables.md` and `pc_rows_results.json` (rows 6, 10, 13), `outputs/pc_tables.md` and
   `pc_results.json` (stage 0, row 1, the arms as "not established: stage 0"), status lines in
   `pc_queue_status.txt`, step logs in `logs/`.
5. **Verify (C13):** re-derive the head rows' image-side and caption-side AMIs and their 3-seed means from
   `outputs/heads/*.npz` with separate code (a Codex re-derivation gets the rule and the data, never our numbers; the
   reviewer's `.scratch/r10-percept-core/review_tmp/scripts/rederive.py` shows the scoring rule). The PercepT
   comparator has no numbers tonight.
6. **Reports:** the auto report with its results brief, the user-read report, and the ablation table of the
   follow-up's user-read report updated with rows 1, 6, 10, 13 (§9). Fill the round log's "Results". The PercepT
   arms appear as blocked by stage 0, with the cause.
7. **Stage 0 for the user** (in the reports and the next decide chat; the user decides, key):
   a. Accept this machine's own September record as stage 0's reference for local runs (0.1238 / 0.2617; identical
      pretraining). The seed 42 AUC check cannot run here (the unchanged script stops at its own re-fit check), so
      it is dropped or replaced. Then the GPU queue's PercepT part runs (about 25 minutes) and the sheet is rescored.
   b. Reserve node404 and run stage 0 there; if it gives 0.1225 / 0.2274 and 0.8296, the gate passes as written.
   Our recommendation: a. The check's purpose (C12: catch a silent change in reused code) is met: the code is
   byte-identical, pretraining is bit-identical on the same machine, and the saved snapshot re-scores exactly. Option b
   costs a reservation and tells nothing new about our code. Note for the reading: one seed moves September's genre
   AMI by 0.034 between two GPUs.
8. **Next chat:** a decide chat with `--notify` (the user's stage 0 decision and decision C). Whether it is a new
   loop or r10's next decide chat: r10 is not finished while the arms are blocked, so r10's decide chat.

## 5. Code and pitfalls

- The queues and the sheet read and write the main checkout's round folder (`/project/CoSiR/src/test/
  20261018_percept_core/`); `outputs/` and `sept_stage0/` are gitignored.
- The gate output `outputs/heads/genre__R0__s0.npz` and `outputs/gate.json` are already there (PASS, difference 0); the
  CPU queue verifies and skips the rerun.
- `pc_queue_status.txt` is appended across runs; the GPU wait reads the last `PC_gpu_SUBMIT`. Start one GPU queue
  only: with two queued launches, the first one's `PC_gpu_DONE` ends the CPU queue's wait (re-review edge case).
- The September modules collide with main's `src` package: only fresh processes (stage 0 does this).
- A subagent once handed back only "placeholder" (build 2); read the diff or the output files before trusting a
  hand-back.

## 6. State at handoff

- Running: nothing (no subagents, no background jobs, no GPU lock). GPU check at 01:31: driver, module and library
  595.99.02; no process, no lock; load 1.2; system disk 257 GB free.
- Uncommitted: nothing.
- On disk over 1 GB: unchanged (stage 1 `outputs/` 1.1 GB and `outputs_fu/` 1.2 GB of the aspect graph sheet,
  deferred). Build 3 added rows to `.scratch/pending_deletions.md` (worktree 09, the integration branch, the review
  temp folder) and marked build 2's worktree rows 04, 05, 08 as free to go.
