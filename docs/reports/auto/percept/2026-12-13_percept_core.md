# PercepT core run: stage 0 fails because the September fit does not repeat itself, the PercepT arms stay blocked, and rows 1, 6, 10 and 13 fill the ablation table

> 2026-10-19 (Amsterdam), local RTX 3090 and CPU; queues 01:32 to 02:54, stage 0 checks to about 03:20. The file name
> uses CoSiR's sequence date. Spec: [PercepT core](../../../superpowers/specs/2026-10-18-percept-core-design.md)
> (approved 2026-10-18 23:34). Decision rule: none (descriptive). Run log:
> [round log](../../../../src/test/20261018_percept_core/20261018_percept_core_log.md), "Results"; stage 0 check:
> [load check log](../../../../src/test/20261019_stage0_load_check/20261019_stage0_load_check_log.md). Train side
> only; labels used only to score.

## Results brief

**What we tried.** The run of the PercepT core spec: the regression check of the unchanged September PercepT (stage
0), then five PercepT core settings on the GPU, then four rows of the ablation table on the CPU: row 1 (k-means on the
September encoders) and rows 6, 10 and 13 (our network, 24 runs).

**Outcome against the bar.** No bar (descriptive). Target AMI: mean of image-side and caption-side AMI for the
aspect's own label, k = 20, 5,000 evaluation paintings; network rows give mean ± sd over 3 seeds.
- Stage 0 failed, so the five PercepT settings did not run (spec §3.5). The September Stage 1 gave held-out emotion
  AMI 0.1244 and 0.1287 (recorded 0.1225) and genre 0.2617 and 0.2804 (recorded 0.2274). Our scoring (row 2) and the
  fix tests passed.
- The cause: the September fit does not repeat itself, also on this machine. Eight local runs of the same code gave
  three outcomes. Its k-means start adds up the centers in a thread order that changes per run. With one CPU thread, two
  full runs gave identical results (0.1287 / 0.2804), one of them under load.
- Row 6, network on the default graph: emotion 0.082, style 0.163, genre 0.388. Each is below the default embedding
  clustered directly (row 2: 0.089, 0.171, 0.393). The instruction in the graph adds 0.036, 0.050 and 0.014 (row 7).
- Rows 10 and 13, the other side's aspect graph: emotion 0.086 and 0.089, style 0.154 and 0.156. The plain aspect
  graph gives 0.118 and 0.213. For genre, row 13 (prompted input) gives 0.434, the highest genre value in the table
  (prompted embedding 0.424).
- Row 1, no training: CLIP image with RoBERTa captions gives emotion 0.127, style 0.174, genre 0.240 (row 2: 0.089,
  0.171, 0.393). On the caption side, RoBERTa gives emotion 0.189 (Qwen default 0.107). On the image side, CLIP gives
  style 0.300 (Qwen default 0.222).

**What it does not show.** No PercepT core number exists, so the comparator for 2026-10-29 is still open.
Development data only: one frozen sample, whose 5,000 evaluation paintings also chose our rows. PercepT core's Stage
1 uses the same k-means start, so its runs carry the same noise. Row 1's spread covers k-means seeds only.

**Options for next.**
- a. Change the stage 0 check to the parts that repeat exactly: the saved September snapshot re-scores to 0.122469 /
  0.227419, the reused code is byte-identical, and k-means on the saved latents gives the saved labels. Then run the
  PercepT part (about 25 GPU minutes) and rescore. Costs a spec amendment. Tells where PercepT core stands.
- b. Keep a September re-fit, with a tolerance from the local spread (emotion about 0.005, genre about 0.02).
  Same run cost; a weak check, since the spread is from eight runs.
- c. Run stage 0 on node404, as the spec says. Costs a reservation; node404 may not repeat itself either.
- Beside any option: one-thread k-means in PercepT core's Stage 1, so its runs repeat (a small change to reviewed
  code; the September start takes 25 s instead of 10 s).

**Your reading.** (left for the user)

<details>
<summary>Our reading and recommendation (open after writing yours)</summary>

**Our reading.** Stage 0 failed on noise in the September fit, not on our code. The check's purpose (constitution
C12: catch a silent change in reused code) is met by the parts that repeat exactly. Sure about the cause; the
mechanism from the first k-means bit to the AMI is an inference. On the rows: our network gains nothing without the
instruction in the graph (row 6 is below row 2). The side of the graph matters most: captions for emotion, images for
style and genre. The September encoders are strong per side: RoBERTa captions for emotion and CLIP images for style.
So a PercepT core comparator built on them could be strong on emotion. Medium sure; development data only.

**Our recommendation.** Option a, with one-thread k-means in PercepT core, then the PercepT part in one evening run.
Our view.

</details>

## What ran

- **Stages and types:** stage 0 regression check; stage 1 baseline (arms P1 to P5, blocked); stage 2 ablation (rows
  1, 6, 10, 13). One unplanned diagnostic: the stage 0 load check (exploration; the spec's failure action allows a
  trace).
- **Data and seeds:** the frozen sample (fit 13,400 paintings, dev 1,489, evaluation 5,000); head seeds 0 to 2; 3
  k-means seeds per view. Stage 0 reads the September split; nothing tuned. No held data.
- **Code:** `main` at `0f47271`; round folder `src/test/20261018_percept_core/`; GPU queue 01:32 to 01:44, CPU queue
  01:32 to 02:54 (three workers, 8 threads each). The table script `pc_ablation.py` reads the stored results only.

## Results

| ID | Measured | Number | Compared against | Bar | Outcome |
|---|---|---|---|---|---|
| Stage 0 | September held-out emotion / genre AMI, script 1 | 0.1244 / 0.2617 | recorded 0.1225 / 0.2274 | equal at 4 decimals | fail |
| Stage 0 | same, script 2 | 0.1287 / 0.2804 | recorded 0.1225 / 0.2274 | equal at 4 decimals | fail |
| Stage 0 | seed 42 Stage 2 macro AUC | none (script stops at its re-fit check) | 0.8296 | equal at 4 decimals | fail |
| Stage 0 | row 2, our scoring | 0.088556 / 0.170954 / 0.393076 | recorded | 1e-6 | pass |
| Stage 0 | four-fix unit tests | 21 passed | — | pass | pass |
| Gate | genre R0 seed 0 rerun (build), target AMI | 0.406099 | stage 1: 0.406099 | 1e-6 | pass (difference 0) |
| Arms | P1 to P5 | not run | — | §6 "established" | not established: stage 0 |

**Verdict under the rule:** none (descriptive). The PercepT arms are blocked by stage 0.

The ablation table (target AMI; network rows mean ± sd over 3 head seeds; rows 2, 3, 7 to 12 from the follow-up):

| # | Input | Method | Graph | Emotion | Style | Genre |
|---|---|---|---|---:|---:|---:|
| 1a | CLIP image + CLIP text | k-means | none | 0.066 | 0.205 | 0.371 |
| 1b | CLIP image + RoBERTa | k-means | none | 0.127 | 0.174 | 0.240 |
| 2 | Qwen default | k-means | none | 0.089 | 0.171 | 0.393 |
| 3 | Qwen prompted | k-means | none | 0.155 | 0.265 | 0.424 |
| 4, 5 | | PercepT core | none | blocked | blocked | blocked |
| 6 | Qwen default | network | default graph | 0.082 ± 0.003 | 0.163 ± 0.002 | 0.388 ± 0.002 |
| 7 | Qwen default | network | plain aspect graph | 0.118 ± 0.002 | 0.213 ± 0.005 | 0.402 ± 0.005 |
| 9 | Qwen default | network | stage 1 pick | 0.137 ± 0.002 | 0.260 ± 0.003 | 0.420 ± 0.002 |
| 10 | Qwen default | network | other side alone | 0.086 ± 0.000 | 0.154 ± 0.001 | 0.428 ± 0.003 |
| 12 | Qwen prompted | network | stage 1 pick | 0.151 ± 0.003 | 0.284 ± 0.002 | 0.424 ± 0.004 |
| 13 | Qwen prompted | network | other side alone | 0.089 ± 0.001 | 0.156 ± 0.000 | 0.434 ± 0.003 |

Image and caption sides of the new rows (target label, mean over seeds):

| # | Emotion img | Emotion cap | Style img | Style cap | Genre img | Genre cap |
|---|---:|---:|---:|---:|---:|---:|
| 1a | 0.065 | 0.067 | 0.300 | 0.110 | 0.412 | 0.330 |
| 1b | 0.065 | 0.189 | 0.300 | 0.049 | 0.412 | 0.068 |
| 6 | 0.071 | 0.094 | 0.186 | 0.140 | 0.417 | 0.359 |
| 10 | 0.083 | 0.088 | 0.172 | 0.137 | 0.458 | 0.397 |
| 13 | 0.084 | 0.094 | 0.175 | 0.138 | 0.468 | 0.400 |

Rows 8 and 11 and every side value of rows 2, 3, 7 to 12 are unchanged; they are in the
[follow-up's user-read report](../../../user_read/2026-10-18_aspect_graph_followup.md), now updated with these rows.

Re-derivation (C13): Codex re-derived every head-row score and row 1 with its own code from the stored outputs,
given the rule and the data only; all values equal ours (max abs difference 0 over 150 per-run side scores, the
3-seed means and row 1).

## Analysis

- **Stage 0 noise.** The September k-means start (K = 60, 10 restarts, 8 threads) gives different centers at the last
  float32 bit in every run, idle or loaded; the labels stay identical. With one thread, the
  centers are identical in three of three runs, and two full runs of script 1 gave identical logs (one at load 8.5). DEC turns these
  differences into one of a few outcomes: 0.1238 / 0.2617 (five runs), 0.1244 / 0.2617 (one), 0.1287 / 0.2804 (two).
  The genre AMI rests on 159 held-out paintings. node404's five September fits were identical, so that machine or its
  stack may have been steadier; not checked.
- **Instruction in the graph.** The network on the default graph is 0.005 to 0.008 below the default embedding on
  every aspect. On the plain aspect graph it is 0.009 to 0.042 above. So the network's gain comes from the aspect
  pairs, and the network alone adds nothing. Our reading.
- **Side of the graph.** For emotion and style the other side's graph loses most of the gain (emotion 0.086 against
  0.118; style 0.154 against 0.213), while it wins for genre (0.428 against 0.402). The prompted input adds at most
  0.006 on these graphs (row 13 vs 10).
- **September encoders.** Each is strong on one side only: RoBERTa captions for emotion (0.189), CLIP images for
  style (0.300). RoBERTa captions are weak for style and genre (0.049, 0.068), which keeps row 1b's mean low.
- **Timing.** The head runs took 400 to 680 s each, not 260 s: three workers ran beside stage 0 and the load check.

## Deviations

- The load check (`src/test/20261019_stage0_load_check/`) is an unplanned diagnostic, inside the spec's failure
  action ("trace the cause"). Its stage 0 reruns wrote only to the session scratchpad; `outputs/stage0.json` is the
  queue's own run.
- The handoff said both scripts reproduce 0.1238 / 0.2617 bit for bit on this machine. Tonight's runs contradict it;
  this report replaces that reading.
- The gate was verified, not retrained, tonight: the build's rerun output re-scored with difference 0.
- Row 6's emotion and genre heads use the same default graph and input, so their outputs are byte-identical (one
  network, scored on two labels; by design, D13).
- C13: the comparator has no numbers. The head rows and row 1 were re-derived instead (see Results).

## Files and storage

- **Outputs:** `src/test/20261018_percept_core/outputs/` (gitignored): `pc_rows_results.json`, `pc_rows_tables.md`,
  `pc_results.json`, `pc_tables.md`, `pc_ablation_table.md`, `stage0.json`, `heads/` (25 runs); status lines in
  `pc_queue_status.txt`; logs in `logs/`, `pc_queue_gpu.log`, `pc_queue_cpu.log`.
- **Storage:** nothing deleted (unattended chat). Left on disk: `outputs/heads/` 977 MB (the 24 head runs and the
  gate, which the scores rest on), `sept_stage0/` 81 MB (the exported September tree), the session scratchpad 82 MB
  (stage 0 reruns; the key logs are copied to `src/test/20261019_stage0_load_check/runs/`). Listed on
  `.scratch/pending_deletions.md` for the next storage summary; nothing of 1 GB or more, so no new ledger row.
