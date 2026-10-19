# PercepT core with K = 40: all five arms are established with a small seed spread; our network is above the comparator on style and genre, level on emotion

> 2026-10-19 (Amsterdam), local RTX 3090; GPU queue 16:34 to 16:51, score 16:52. The file name uses CoSiR's sequence
> date. Spec: [PercepT core](../../../superpowers/specs/2026-10-18-percept-core-design.md) (approved 2026-10-18 23:34;
> amended A1 and A2 2026-10-19 08:25, A3 10:45). Decision rule: none (descriptive). Run log:
> [round log](../../../../src/test/20261018_percept_core/20261018_percept_core_log.md), "Results, run 3". Extends
> [run 2's report](2026-12-14_percept_core_arms.md) (Algorithm 1, 1 to 99 topics) and the
> [first run's report](2026-12-13_percept_core.md) (rows 1, 6, 10, 13). Train side only; labels used only to score.

## Results brief

**What we tried.** The rerun of PercepT core's five settings (arms P1 to P5, 3 seeds each) after amendment A3. Each
Stage 1 run now keeps a fixed K = 40 topics: the 40 of 100 trained topic centers with the most training pairs.
Algorithm 1's count is only reported. Ranking by pairs, not norms: in run 2's 99-topic runs norm and pair share
correlate positively (Spearman 0.40 to 0.44), so a norm cut would keep little-used topics (spec A3). Score: each arm's Stage 2 image output on the 5,000 evaluation paintings,
clustered at k = 20, AMI against each aspect's label (image-side AMI).

**Outcome against the bar.** No bar (descriptive).
- All five arms are established: stage 0 passed (30 fix tests), and dev AUC is 0.85 to 0.96 on every seed (bar:
  0.75). The population sd over the 3 seeds is at most 0.007 per arm and aspect (run 2: up to 0.143).
- Comparator (best established arm per aspect), emotion / style / genre: P3 0.077, P2 0.229, P3 0.436. Our network
  (stage 1 pick, default input; rows 9 and, for genre, 10): 0.081 / 0.348 / 0.458. Differences (unrounded means): +0.003 (level), +0.120, +0.022.
- PercepT core is level with or below its image encoder clustered directly. P3 against Qwen default images: 0.077 /
  0.189 / 0.436 against 0.070 / 0.222 / 0.449. P2 against CLIP images: 0.066 / 0.229 / 0.400 against 0.065 /
  0.300 / 0.412.
- The Stage 1 center norms and Algorithm 1 counts equal run 2's bitwise (9 of 9 runs), so the training before the
  cut is unchanged. On the six seeds that kept 99 topics in run 2, K = 40 moves image-side AMI by at most 0.02. The
  cut is close: the 40th and 41st topics differ by 1 to 14 fit pairs.

**What it does not show.** Development data only: the same 5,000 paintings chose our rows and the comparator arm.
K = 40 is our choice, not PercepT's rule; no other K below 99 was run, and another K could move the comparator.
Our network's numbers are the stage 1 picks; other default-input heads reach 0.085 and 0.089 on emotion. The claim is PercepT core with
K = 40 (four fixes, our features), not PercepT as published. Three seeds give a spread, not a test.

**Options for next.**
- a. Take PercepT core with K = 40 as the comparator (P3, P2, P3 per aspect) and go to the stage 2 decision. No new
  run. Emotion is level with our network, so stage 2 must raise image-side emotion.
- b. As a, and add the image encoders clustered directly as a second bar per aspect (Qwen default images for emotion
  and genre, CLIP images for style; 0.070 / 0.300 / 0.449). No new run. A stricter bar on style and genre.
- c. A K check before stage 2: rerun the five arms at two more counts (for example 20 and 60). A short build and
  about 35 GPU minutes. Shows how much the comparator depends on K, mainly for emotion.

**Your reading.** Left for the user.

<details>
<summary>Our reading and recommendation (open after writing yours)</summary>

**Our reading.** The fixed count removes the seed lottery on these three seeds: every arm trains well, and most arm
differences exceed their spread (P3 and P5 tie on genre). PercepT core's topics add little on the image side. The best arm per aspect is level with or below the
image encoder it reads, clearly lower on style (P2 0.229 against CLIP images 0.300). Our network is above PercepT core on style and genre. Emotion is the
open aspect: our network, PercepT core and the plain Qwen image are all within 0.011. Sure about the order (three
seeds, small spread); development data only.

**Our recommendation.** Option b, then the stage 2 decision in the same chat. The image encoders clustered directly
are real comparators: CLIP images beat P2 on style (0.300 against 0.229), Qwen default images beat P3 on genre
(0.449 against 0.436) (constitution C1). Option c can wait: on the same
training, 40 against 99 topics moved scores by at most 0.02, and emotion moved by at most 0.005. Our view.

</details>

## What ran

- **Stages and types:** stage 0 regression check (amended, A1); stage 1 baseline, arms P1 to P5 (spec §3.2), with
  the topic count of A3. Rows 1, 6, 10, 13 and the head gate are the first run's and were not rerun.
- **Data and seeds:** the frozen sample (fit 13,400 paintings, 67,327 caption pairs; dev 1,489; evaluation 5,000);
  PercepT seeds 0 to 2; 3 k-means seeds per scored matrix. Stage 0 reads the September split; nothing tuned. No held
  data.
- **Code:** `main` at `dfa3a85` (build 6 merged at `bfff3ec`); round folder `src/test/20261018_percept_core/`. GPU
  queue `pc_queue_gpu.sh --foreground --lock-wait 14400`, 16:34:09 to 16:51:49, exit 0, every step 0: stage 0 51 s,
  RoBERTa `skip`, 9 Stage 1 runs (77 to 99 s), 15 Stage 2 steps (5 to 8 s), the Stage 1 repeat (77 s) and its check.
  Then `pc_score.py && pc_ablation.py` in one chain (exit 0). Run 2's files are kept as `*_run2*`.

## Results

| ID | Measured | Number | Compared against | Bar | Outcome |
|---|---|---|---|---|---|
| Stage 0 (i) | September files, record commits vs `1b70d7f` | 0 of 74 and 0 of 77 differ | identical | identical | pass |
| Stage 0 (ii) | parts copied by `pc_sept.py` | 0 differ | identical | identical | pass |
| Stage 0 (iii) | saved snapshot, held-out emotion / genre AMI | 0.12246903 / 0.22741920 (9,365 / 159 rows) | recorded | 1e-6 | pass |
| Stage 0 (iv) | k-means labels on the snapshot latents, one thread | 0 differ (25 s) | saved labels | exact | pass |
| Stage 0 | row 2, our scoring | 0.088556 / 0.170954 / 0.393076 | recorded | 1e-6 | pass |
| Stage 0 | fix unit tests (after A3) | 30 passed | — | 30, none skipped | pass |
| A2 repeat | Stage 1 CLIP + RoBERTa seed 0, run twice | bitwise equal (sha256 `255a5b03...`) | — | reported | equal |
| Training check | Stage 1 `center_norms` and Algorithm 1 K per γ, run 3 vs run 2 | 9 of 9 bitwise equal | run 2 | reported | equal |

"Stage 0 passed" means these checks (spec §7, the C12 exception under A1). The training check compares norms and
counts, not the full center vectors; their bitwise equality points to the same training before the cut.
Established (spec §6) also needs dev AUC at least 0.75 on every seed.

Arms, image-side AMI at k = 20 (mean ± population sd over 3 PercepT seeds; all established per spec §6):

| Arm (row) | Stage 1 → Stage 2 | Emotion | Style | Genre | Dev AUC |
|---|---|---:|---:|---:|---:|
| P1 (4a) | CLIP + RoBERTa → CLIP patches | 0.053 ± 0.001 | 0.167 ± 0.003 | 0.356 ± 0.001 | 0.852 ± 0.002 |
| P2 (4b) | CLIP + RoBERTa → CLIP global vector | 0.066 ± 0.002 | 0.229 ± 0.005 | 0.400 ± 0.005 | 0.907 ± 0.003 |
| P3 (5a) | Qwen + Qwen emotion → Qwen vector | 0.077 ± 0.001 | 0.189 ± 0.003 | 0.436 ± 0.004 | 0.959 ± 0.001 |
| P4 (5b) | Qwen + Qwen emotion → CLIP patches | 0.054 ± 0.000 | 0.162 ± 0.003 | 0.361 ± 0.001 | 0.862 ± 0.003 |
| P5 (5c) | Qwen + RoBERTa → Qwen vector | 0.074 ± 0.002 | 0.181 ± 0.005 | 0.435 ± 0.007 | 0.961 ± 0.002 |

Comparator (D15: per aspect, the established arm with the highest mean; ties within 0.003 go to the lower arm):
emotion P3 (0.077), style P2 (0.229), genre P3 (0.436; P5 0.435 is within 0.003). The pick over all arms is the same.

Kept topics (A3; per Stage 1 version, 3 seeds): every kept topic holds at least 1% of the fit pairs after the cut
(40 of 40 on all 9 runs). The cut is close: the 40th and 41st centers differ by 1 to 2 fit pairs in the CLIP +
RoBERTa runs and by 2 to 14 in the others (about 670 to 740 pairs each, near 1%). Before the cut, the kept 40 hold
this share of the fit pairs:

| Stage 1 version (arms) | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| CLIP + RoBERTa (P1, P2) | 0.568 | 0.567 | 0.581 |
| Qwen + Qwen emotion (P3, P4) | 0.611 | 0.626 | 0.627 |
| Qwen + RoBERTa (P5) | 0.594 | 0.593 | 0.586 |

Algorithm 1's count (reported only, the same converged centers as run 2): CLIP + RoBERTa 99, 2 (100 at γ = 5), 3;
Qwen + Qwen 1, 99, 99; Qwen + RoBERTa 1, 1, 1.

K-native AMI (argmax topic at K = 40, emotion / style / genre, 3-seed mean): Stage 1 painting topic CLIP + RoBERTa
0.143 / 0.160 / 0.306, Qwen + Qwen 0.132 / 0.172 / 0.375, Qwen + RoBERTa 0.122 / 0.166 / 0.367. Stage 2 image: P1
0.058 / 0.153 / 0.351, P2 0.066 / 0.220 / 0.383, P3 0.082 / 0.191 / 0.400, P4 0.055 / 0.152 / 0.359, P5 0.078 /
0.188 / 0.401. Full values: `outputs/pc_tables.md`.

Run 2 against run 3, per seed (same Stage 1 training; image-side AMI change from run 2's count to K = 40):

| Run 2 count | Seeds | Emotion | Style | Genre | Dev AUC, run 2 → run 3 |
|---|---|---|---|---|---|
| 99 | P1 s0, P2 s0, P3 s1 and s2, P4 s1 and s2 | +0.000 to +0.005 | −0.020 to +0.005 | −0.019 to +0.015 | 0.78–0.95 → 0.85–0.96 |
| 2 or 3 | P1 s1 and s2, P2 s1 and s2 | +0.034 to +0.046 | +0.023 to +0.162 | +0.155 to +0.320 | 0.62–0.90 → 0.85–0.91 |
| 1 (not trained) | P3 s0, P4 s0, P5 s0 to s2 | — | — | — | — → 0.86–0.96 |

The ablation table, image side (rows 4a to 5c new; other rows unchanged against run 2's table, diff checked):

| # | Method | Emotion | Style | Genre |
|---|---|---:|---:|---:|
| 1 | CLIP image, k-means | 0.065 | 0.300 | 0.412 |
| 2 | Qwen default image, k-means | 0.070 | 0.222 | 0.449 |
| 3 | Qwen prompted image, k-means | 0.093 | 0.395 | 0.496 |
| 4a | PercepT core P1 | 0.053 | 0.167 | 0.356 |
| 4b | PercepT core P2 | 0.066 | 0.229 | 0.400 |
| 5a | PercepT core P3 | 0.077 | 0.189 | 0.436 |
| 5b | PercepT core P4 | 0.054 | 0.162 | 0.361 |
| 5c | PercepT core P5 | 0.074 | 0.181 | 0.435 |
| 9 | network, stage 1 pick, default input | 0.081 | 0.348 | 0.454 |
| 10 | network, other side alone, default input | 0.083 | 0.172 | 0.458 |
| 12 | network, stage 1 pick, prompted input | 0.087 | 0.393 | 0.458 |

Arm differences (image side, emotion / style / genre): CLIP global vector against CLIP patches on the same topics
(4b − 4a) +0.013 / +0.061 / +0.044; CLIP patches against the Qwen vector on Qwen topics (5b − 5a) −0.024 / −0.026 /
−0.074; RoBERTa against Qwen emotion as the emotion input (5c − 5a) −0.003 / −0.008 / −0.001; Qwen against CLIP
inputs, both with RoBERTa (5c − 4b) +0.008 / −0.048 / +0.035. The both-side table keeps rows 4 and 5 empty (D6). Full
table: `outputs/pc_ablation_table.md`.

Re-derivation (C13): Codex (`gpt-6.1-sol`, high) re-derived every arm file's image-side and argmax AMI, the arm
means and sds, the comparator over established arms and over all arms, and the Stage 1 kept-topic checks, given the
scoring rule, the data and the list of established arms only. All 90 per-file values, 60 arm aggregates, 9 kept
shares and 27 Stage 1 argmax values equal ours (max abs difference 0); its top-40 pick from `fit_counts` equals the
stored kept mask in 9 of 9 files; its 286 input checks pass. Both comparator picks equal ours. Files:
`src/test/20261018_percept_core/rederive_run3/`.

Back-fill, added 2026-10-19 22:00 by the stage 2 run chat (stage 2 spec D7, §3.4): V-measure, ARI and FMI on the
same k = 20 partitions, image side, 3-seed means. Our network's emotion and style cells use stage 0's regenerated
heads of the stage 2 run. The code asserts each cell's AMI equals the table above at 3 decimals (24/24 cells). Full
values with spreads: [ms_backfill_k40.md](../../../../src/test/20261019_model_sheet/ms_backfill_k40.md).

| Row | Emo V | Emo ARI | Emo FMI | Style V | Style ARI | Style FMI | Genre V | Genre ARI | Genre FMI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CLIP image | 0.071 | **0.030** | 0.139 | 0.313 | 0.120 | 0.180 | 0.417 | 0.225 | 0.321 |
| Qwen default image | 0.076 | 0.027 | 0.135 | 0.236 | 0.074 | 0.136 | 0.453 | 0.265 | 0.362 |
| P1 | 0.059 | 0.022 | 0.130 | 0.182 | 0.055 | 0.118 | 0.361 | 0.207 | 0.302 |
| P2 | 0.072 | 0.026 | **0.142** | 0.243 | 0.084 | 0.150 | 0.405 | 0.251 | 0.344 |
| P3 | 0.083 | 0.024 | 0.134 | 0.203 | 0.059 | 0.123 | 0.441 | 0.250 | 0.346 |
| P4 | 0.060 | 0.020 | 0.128 | 0.178 | 0.050 | 0.113 | 0.367 | 0.199 | 0.294 |
| P5 | 0.080 | 0.021 | 0.134 | 0.195 | 0.057 | 0.123 | 0.440 | 0.263 | 0.357 |
| Our network | **0.087** | 0.028 | 0.135 | **0.360** | **0.156** | **0.214** | **0.463** | **0.266** | **0.364** |

V ranks the rows as AMI does. ARI and FMI agree on style and genre (genre ARI and FMI level with Qwen, within 0.002).
On emotion they do not: CLIP image has the highest ARI (0.030 ± 0.005, our network 0.028) and P2 the highest FMI
(0.142 ± 0.004, ours 0.135). All emotion ARI values lie between 0.020 and 0.030.

## Analysis

- **A fixed count makes PercepT core stable across the three seeds.** The per-arm population sd is at most 0.007 on
  every aspect, against up to 0.143 in run 2. All 15 Stage 2 models pass the dev AUC bar (0.850 to 0.963). The kept 40 hold 57 to 63% of the
  fit pairs before the cut, and every kept topic still holds at least 1% after it.
- **The count, not the training, caused run 2's weak comparator** (our reading). The Stage 1 center norms are
  bitwise the same in both runs. Where run 2 kept 99 topics, K = 40 changes scores by at most 0.02 and raises dev AUC. Where run 2 kept 2 or
  3 topics, K = 40 adds up to 0.32 genre AMI.
- **PercepT core's topics add little on the image side.** P3 against Qwen default images: +0.008 emotion, −0.033
  style, −0.013 genre. P2 against CLIP images: +0.001, −0.071, −0.012. Style loses most: the topics come from
  captions and the emotion input, and style is mostly visual. Our reading.
- **The encoder sets the profile.** Qwen arms lead on emotion and genre, CLIP arms on style, as when the encoders are
  clustered directly (row 2 against row 1: +0.005, −0.078, +0.037). The emotion input (RoBERTa or Qwen emotion,
  P5 against P3) changes scores by at most 0.008, within the spread. Our reading.
- **The CLIP patch reader is the weak part** (our reading). On the same topics, patches lose to a global vector: P1 against P2,
  and P4 against P3 (−0.024 / −0.026 / −0.074). P4 is below CLIP images clustered directly on every aspect, so Qwen
  topics do not carry over to the patch reader.
- **Pairs weigh paintings unequally.** Stage 1 trains on caption pairs, so paintings with more captions weigh more,
  and "most-used" counts pairs, not paintings (spec §8).
- **Against our network.** Style +0.120 (0.348 against P2's 0.229) and genre +0.022 (0.458 against P3's 0.436) are
  far above the spreads. Emotion is +0.003 (0.081 against P3's 0.077), level. Rows 7 and 8 (other default-input heads) reach
  0.085 and 0.089 on image-side emotion; the stage 1 rule did not pick them. Selection: our rows and the
  comparator were chosen on these 5,000 paintings (spec §8).

## Deviations

- None from the amended spec. The GPU queue took 18 minutes (estimate about 20).
- The training check (Stage 1 norms against run 2) and the run 2 against run 3 table are additions from the
  handoff, reported, not gates.
- The re-derivation brief added a Stage 1 kept-topic check; the fit rows were given to it as a data file
  (sha256 equal to the runs' `fit_rows_sha256`).

## Files and storage

- **Outputs:** `src/test/20261018_percept_core/outputs/` (gitignored): `stage1/` 6.8 MB, `stage2/` 11 MB,
  `repeat/` 0.8 MB, `stage1_repeat.json`, `stage0.json`, `pc_results.json` (schema 2), `pc_tables.md`,
  `pc_ablation_table.md`; run 2's copies `*_run2*`, `logs/run2/`. Logs: `logs/`, `pc_queue_gpu.log`,
  `pc_queue_status.txt` (the block after `PC_gpu_SUBMIT 2026-10-19T16:34:09`).
- **Re-derivation:** `rederive_run3/` (brief, scripts, summary, comparison; the results JSON is gitignored).
- **Storage:** nothing deleted (unattended chat). New files total about 19 MB; nothing of 1 GB or more, so no new
  ledger row.
