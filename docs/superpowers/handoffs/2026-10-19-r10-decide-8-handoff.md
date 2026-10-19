# Handoff: read the r10 run's results with the user (decision C), then settle stage 0, so PercepT core can run

## Brief
- Done: the r10 run. Stage 0 failed again, so the five PercepT core settings stayed blocked. Rows 1, 6, 10, 13 of the ablation table are complete; Codex re-derived them equal.
- State: the September fit does not repeat itself: its k-means start sums centers in a changing thread order. Eight local runs gave three results; one thread repeats exactly. Row 6 (default graph) is below row 2: the aspect graph carries the gain.
- Next: decision C on the results brief, then the user's stage 0 decision.
- Blocked: the PercepT core runs, until the user decides on stage 0.

> Written 2026-10-19 03:19 by "r10 run". Loop step reached: steps 8 and 9 done (run, verify, report); next step 1
> (decision C) for this run, then the stage 0 decision; r10 is not finished while the PercepT arms are blocked.
> Direction log entry: [2026-10-18, aspect graph follow-up](../direction_log.md) (no new entry yet; the wrap-up adds
> it).

## 1. Read in this order

1. The auto report [2026-12-13_percept_core.md](../../reports/auto/percept/2026-12-13_percept_core.md): show the
   user its results brief without the collapsed block, get the user's reading, then open ours (`research-loop.md`).
2. The user-read report [2026-10-19_percept_core.md](../../user_read/2026-10-19_percept_core.md) (its advice is
   folded away the same way) and the updated table in
   [2026-10-18_aspect_graph_followup.md](../../user_read/2026-10-18_aspect_graph_followup.md).
3. The load check log `src/test/20261019_stage0_load_check/20261019_stage0_load_check_log.md` (the stage 0 cause).
4. The spec [2026-10-18-percept-core-design.md](../specs/2026-10-18-percept-core-design.md): §3.5 (gates), §6
   ("established"), §9.
5. The round log `src/test/20261018_percept_core/20261018_percept_core_log.md`, "Results".

Skills: none.

## 2. Decided by the user, not to be reopened

- The spec's §4 (D1 to D15) and its approval, 2026-10-18 23:34, including the stage 0 gate and its failure action
  (§3.5).
- Exploration framing: "as many trials as possible to enlarge the ablation table and then pick" (2026-10-18).

## 3. Where things stand

- **Queues** (01:32 to 02:54): GPU queue exit 1 (stage 0 FAIL; RoBERTa skip; 24 PercepT runs blocked). CPU queue
  exit 0: gate PASS (difference 0), 24 head runs, scores, sheet.
- **Stage 0** tonight: script 1 0.1244 / 0.2617, script 2 0.1287 / 0.2804 (recorded 0.1225 / 0.2274); row 2 and the 21
  fix tests pass. The build had both at 0.1238 / 0.2617.
- **Cause** (load check): the September `KMeans(60, n_init=10)` with 8 threads gives new last-bit centers every run,
  idle or loaded (labels equal); with one thread the centers repeat, and two full one-thread runs of script 1 gave
  identical logs (0.1287 / 0.2804). Eight 8-thread runs: 0.1238 / 0.2617 five times, 0.1244 / 0.2617 once, 0.1287 /
  0.2804 twice. PercepT core's Stage 1 (`pc_sept.py`, 8 threads) has the same k-means start.
- **Rows** (target AMI, 3 seeds): row 6 0.082 / 0.163 / 0.388 (row 2: 0.089 / 0.171 / 0.393); row 10 emotion 0.086,
  style 0.154 (row 7: 0.118, 0.213); row 13 0.089 / 0.156 / 0.434 (row 3 genre 0.424). Row 1: CLIP + RoBERTa 0.127 /
  0.174 / 0.240; CLIP + CLIP text 0.066 / 0.205 / 0.371. Sides: RoBERTa captions emotion 0.189 (Qwen default 0.107);
  CLIP images style 0.300 (Qwen default 0.222).
- **C13:** Codex (`gpt-6.1-sol` high) re-derived all head-row scores and row 1 from the stored outputs: equal (max abs
  difference 0). The comparator has no numbers yet.

## 4. Open points for the next step

1. **Decision C (key):** the user's reading of the results brief, before ours.
2. **Stage 0 (key; the user decides; options as in the brief):**
   a. Check only what repeats exactly: the saved node404 snapshot re-scores to 0.122469 / 0.227419, the reused code is
      byte-identical, k-means on the saved latents gives the saved labels. A spec amendment (§3.5) and a small code
      change to `pc_stage0.py` or the queue's gate (`pc_queue_gpu.sh` blocks Stage 1 on stage 0's exit code).
   b. Keep a September re-fit with a tolerance from the local spread (emotion about 0.005, genre about 0.02).
   c. Run stage 0 on node404 (needs a reservation; may not repeat either).
   Our recommendation: a.
3. **One-thread k-means in PercepT core (key, ours to propose):** makes its runs repeat (the September start: 25 s
   instead of 10 s). A change to reviewed verdict code (`pc_sept.py` or `pc_stage1.py`): one ticket with a review
   (`matt-chain.md`). Without it, PercepT core's seeds carry this noise too.
4. **When the PercepT part runs (minor):** about 25 GPU minutes per the build's timing (the spec's estimate was 2.5 to
   3 GPU hours; not re-measured). Then the sheet is rescored (`pc_score.py`).
5. **Wrap-up after decision C:** the direction log entry; the storage summary, including the r10 run's rows on
   `.scratch/pending_deletions.md` (head outputs 977 MB, `sept_stage0/` 81 MB, the run's session scratchpad 82 MB) and
   the earlier deferred items; `wrapup-check`.

## 5. Code and pitfalls

- A changed stage 0 needs a code change: the GPU queue runs `pc_stage0.py` first and blocks every Stage 1 and Stage 2
  run on its exit code.
- `pc_queue_status.txt` is appended across runs; the CPU queue's sheet waits for the `PC_gpu_DONE` after the last
  `PC_gpu_SUBMIT`. Start one GPU queue only.
- With three CPU head workers beside the GPU queue, head runs took 400 to 680 s each (the build measured 260 s alone).
- Row 6's emotion and genre heads share the default graph and input: their outputs are byte-identical.
- The September modules collide with main's `src` package: only fresh processes.
- `pc_ablation.py` rebuilds the ablation table from the stored results (`outputs/pc_ablation_table.md`).

## 6. State at handoff

- Running: nothing (checked at the cut).
- Uncommitted: nothing.
- On disk over 1 GB: nothing new (head outputs 977 MB); older deferred items unchanged.
