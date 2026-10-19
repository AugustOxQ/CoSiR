# PercepT core arms: stage 0 passes, but the topic count falls to 1 to 3 on most seeds, so the one established comparator is weak

> 2026-10-19 (Amsterdam), local RTX 3090; GPU queue 09:50 to 10:08, score 10:09. The file name uses CoSiR's sequence
> date. Spec: [PercepT core](../../../superpowers/specs/2026-10-18-percept-core-design.md) (approved 2026-10-18 23:34;
> amended 2026-10-19 08:25, A1 and A2). Decision rule: none (descriptive). Run log:
> [round log](../../../../src/test/20261018_percept_core/20261018_percept_core_log.md), "Results, run 2". Extends the
> [first run's report](2026-12-13_percept_core.md) (stage 0 failure; rows 1, 6, 10, 13). Train side only; labels used
> only to score.

## Results brief

**What we tried.** The second run of the PercepT core spec: the amended stage 0 check, then PercepT core in five
settings (arms P1 to P5), 3 seeds each, on the local GPU. Each arm's Stage 2 image output on the 5,000 evaluation
paintings is clustered at k = 20 and scored by AMI against each aspect's label (image-side AMI).

**Outcome against the bar.** No bar (descriptive).
- Stage 0 passed every amended check. A repeat of one Stage 1 run was bitwise equal, so one-thread k-means works.
- The topic count rule (Algorithm 1, γ = 2) kept 1, 2, 3 or 99 topics, by seed. In every run the largest gap between
  sorted center norms sat at one end of 100 evenly spread norms.
- Only P2 is established (CLIP + RoBERTa topics, Stage 2 on the CLIP global vector). It is the comparator for all
  three aspects. Image-side AMI, emotion / style / genre: P2 0.036 / 0.162 / 0.217; our best network with default
  input 0.081 / 0.348 / 0.458.
- The other arms: P1 fails the dev AUC bar on one seed (0.620, 2 topics). P3 and P4 lose seed 0 (1 topic). P5 keeps
  1 topic on every seed and never trains.
- With 99 topics, P3 and P2 come near their image encoder clustered directly on emotion and genre and stay lower on
  style. P3 (2 seeds): 0.072 / 0.194 / 0.446; Qwen default images 0.070 / 0.222 / 0.449. P2 seed 0: 0.063 / 0.250
  / 0.412; CLIP images 0.065 / 0.300 / 0.412. P1 seed 0 and P4 are below CLIP images on every aspect.

**What it does not show.** Development data only: the same 5,000 paintings chose our rows and the comparator arm.
Three seeds per arm; the comparator's spread appears to come mainly from the topic count (no fixed-count repeats
separate it from training noise). The claim is
PercepT core (four fixes, our features), not PercepT as published. Which emotion source helps (P5 against P3) stays
open.

**Options for next.**
- a. Keep P2 as the comparator. Add the image encoders clustered directly (rows 1 and 2, image side) as a second bar
  for 2026-10-29. Go to the stage 2 decision. No new run; a reviewer may call the comparator weak.
- b. Amend PercepT core's topic count so that a gap at the ends cannot set it (for example a fixed count, or a cut
  only inside the norm range). Rerun the five arms: a short build and about 20 GPU minutes. Aims at a stable comparator
  and a P5 result; not tested yet.
- c. Keep the rule and run seeds 3 to 9: about 15 GPU minutes. Measures how often the count falls; gives no stable
  comparator.

**Your reading.** Left for the user.

<details>
<summary>Our reading and recommendation (open after writing yours)</summary>

**Our reading.** The code is the September code, and the one repeated Stage 1 run was exact. The weak comparator comes from the
topic count rule. With 100 evenly spread norms, the largest single gap falls at one end, so the cut keeps almost all
centers or almost none. Training is not the problem: with 99 topics the arms reach dev AUC 0.78 to 0.95. With 99
topics the best settings (P3, P2) add nothing over their image encoder clustered directly. Against the best 99-topic
seed per aspect (picked after the fact), our network's image side is 0.009 higher on emotion, 0.012 on genre and
0.098 on style (0.348 against 0.250, P2 seed 0). Sure about the
count mechanism (norms read directly); medium sure about the 99-topic reading (one or two seeds per arm).

**Our recommendation.** Option b, with the new count rule chosen in the decide chat, then the stage 2 decision in the
same chat. A comparator that falls to 2 or 3 topics on two of three seeds is not the strongest real comparator
(constitution C1). Our view.

</details>

## What ran

- **Stages and types:** stage 0 regression check (amended, A1); stage 1 baseline, arms P1 to P5 (spec §3.2). The CPU
  rows (1, 6, 10, 13) and the head gate are from the first run and were not rerun.
- **Data and seeds:** the frozen sample (fit 13,400 paintings, dev 1,489, evaluation 5,000); PercepT seeds 0 to 2;
  3 k-means seeds per scored matrix. Stage 0 reads the September split; nothing tuned. No held data.
- **Code:** `main` at `6cd4526`; round folder `src/test/20261018_percept_core/`. GPU queue
  `pc_queue_gpu.sh --foreground --lock-wait 14400`, 09:50:39 to 10:08:02, exit 0: stage 0 51 s, RoBERTa `skip`, 9
  Stage 1 runs (76 to 99 s), 15 Stage 2 steps (3 to 8 s), the Stage 1 repeat (77 s) and its check. Then
  `pc_score.py` and `pc_ablation.py`. The first run's files are kept as `*_run1.*`.

## Results

| ID | Measured | Number | Compared against | Bar | Outcome |
|---|---|---|---|---|---|
| Stage 0 (i) | September files, record commits vs `1b70d7f` | 0 of 74 and 0 of 77 differ | identical | identical | pass |
| Stage 0 (ii) | parts copied by `pc_sept.py` | 0 differ | identical | identical | pass |
| Stage 0 (iii) | saved snapshot, held-out emotion / genre AMI | 0.12246903 / 0.22741920 (9,365 / 159 rows) | recorded | 1e-6 | pass |
| Stage 0 (iv) | k-means labels on the snapshot latents, one thread | 0 differ (25 s) | saved labels | exact | pass |
| Stage 0 | row 2, our scoring | 0.088556 / 0.170954 / 0.393076 | recorded | 1e-6 | pass |
| Stage 0 | four-fix unit tests | 21 passed | — | 21, none skipped | pass |
| A2 repeat | Stage 1 CLIP + RoBERTa seed 0, run twice | bitwise equal (same sha256) | — | reported | equal |

"Stage 0 passed" means these checks (spec §7, the C12 exception under A1): the saved September model re-scores to
its numbers, the reused code is unchanged, and k-means repeats its labels. The September re-fit is not part of it.

Arms, image-side AMI at k = 20 (mean ± population sd over 3 PercepT seeds; "established" per spec §6):

| Arm (row) | Stage 1 → Stage 2 | Emotion | Style | Genre | Established |
|---|---|---:|---:|---:|---|
| P1 (4a) | CLIP + RoBERTa → CLIP patches | 0.029 ± 0.016 | 0.137 ± 0.028 | 0.234 ± 0.084 | no: seed 1 dev AUC 0.620 |
| P2 (4b) | CLIP + RoBERTa → CLIP global vector | 0.036 ± 0.020 | 0.162 ± 0.078 | 0.217 ± 0.143 | yes |
| P3 (5a) | Qwen + Qwen emotion → Qwen vector | — | — | — | no: seed 0 K = 1 |
| P4 (5b) | Qwen + Qwen emotion → CLIP patches | — | — | — | no: seed 0 K = 1 |
| P5 (5c) | Qwen + RoBERTa → Qwen vector | — | — | — | no: K = 1 on all seeds |

Per seed (image-side AMI; K at γ = 2; Stage 2 dev macro AUC):

| Arm | Seed | K | Emotion | Style | Genre | Dev AUC |
|---|---:|---:|---:|---:|---:|---:|
| P1 | 0 / 1 / 2 | 99 / 2 / 3 | 0.051 / 0.016 / 0.020 | 0.166 / 0.099 / 0.145 | 0.349 / 0.152 / 0.201 | 0.799 / 0.620 / 0.830 |
| P2 | 0 / 1 / 2 | 99 / 2 / 3 | 0.063 / 0.023 / 0.020 | 0.250 / 0.060 / 0.175 | 0.412 / 0.074 / 0.165 | 0.896 / 0.752 / 0.901 |
| P3 | 1 / 2 | 99 / 99 | 0.072 / 0.072 | 0.192 / 0.197 | 0.441 / 0.452 | 0.946 / 0.943 |
| P4 | 1 / 2 | 99 / 99 | 0.051 / 0.053 | 0.156 / 0.156 | 0.349 / 0.347 | 0.784 / 0.791 |

Topic count per γ (Stage 1 runs; the same at every γ unless stated): CLIP + RoBERTa 99, 2 (100 at γ = 5), 3; Qwen +
Qwen 1, 99, 99; Qwen + RoBERTa 1, 1, 1.

K-native AMI (argmax topic, emotion / style / genre, mean over the seeds that trained): P1 Stage 2 image 0.026 /
0.100 / 0.192; P2 Stage 2 image 0.033 / 0.151 / 0.203; Stage 1 painting topic (P1 and P2 share it) 0.064 / 0.111 /
0.155. At 99 topics: Stage 1 painting topic 0.143 / 0.168 / 0.313 (CLIP + RoBERTa seed 0) and 0.130 / 0.167 / 0.361
(Qwen + Qwen, 2 seeds). Full values: `outputs/pc_tables.md`.

Comparator (D15: per aspect, the established arm with the highest mean; ties within 0.003 go to the lower arm): P2 for
emotion (0.036), style (0.162) and genre (0.217). Over all arms with three trained seeds, genre would go to P1 (0.234,
not established).

The ablation table, image side (rows 4a and 4b new; our rows from the follow-up and the first run, mean over seeds):

| # | Method | Emotion | Style | Genre |
|---|---|---:|---:|---:|
| 1 | CLIP image, k-means | 0.065 | 0.300 | 0.412 |
| 2 | Qwen default image, k-means | 0.070 | 0.222 | 0.449 |
| 3 | Qwen prompted image, k-means | 0.093 | 0.395 | 0.496 |
| 4a | PercepT core P1 (not established) | 0.029 | 0.137 | 0.234 |
| 4b | PercepT core P2 | 0.036 | 0.162 | 0.217 |
| 5a to 5c | PercepT core P3 to P5 | not established | not established | not established |
| 9 | network, stage 1 pick, default input | 0.081 | 0.348 | 0.454 |
| 10 | network, other side alone, default input | 0.083 | 0.172 | 0.458 |
| 12 | network, stage 1 pick, prompted input | 0.087 | 0.393 | 0.458 |
| 13 | network, other side alone, prompted input | 0.084 | 0.175 | 0.468 |

The both-side table keeps rows 4 and 5 empty (PercepT has no caption-only path, D6). The full table:
`outputs/pc_ablation_table.md`; rows other than 4a to 5c are unchanged from the first run (diff checked).

Re-derivation (C13): Codex (`gpt-6.1-sol`, high) re-derived every arm file's image-side and argmax AMI, the arm means
and sds, and the comparator, given the scoring rule and the data only. All 90 per-file values and 24 arm aggregates
equal ours (max abs difference 0); K per file agrees. Its brief had no "established" filter, so its genre pick is P1,
equal to our "all arms" pick. Files: `src/test/20261018_percept_core/rederive_run2/`.

## Analysis

- **The topic count reads a tail gap.** In all 9 Stage 1 runs the center norms spread evenly from about 0.33 to
  0.67. The largest adjacent gap is 4.3 to 11.8 times the mean gap and always sits at an end: between positions 1
  and 2, 2 and 3, 3 and 4 (low end; K = 1, 2, 3) or 99 and 100 (high end; K = 99). Any γ from 1.5 to 5 passes such a
  gap, so the K table barely moves with γ. Which end wins changes by seed. The spec's risk table names this case and
  rules out a rerun with another γ.
- **K = 99 runs train well.** About 40 of the 99 topics hold at least 1% of the pairs (40, 42, 40), close to
  September's 40 topics. Dev AUC is 0.78 to 0.95 there. With K = 2 or 3 the Stage 2 output has 2 or 3 numbers per
  painting, so k = 20 clustering on it carries little aspect information. P2 seed 1 passes the AUC bar by 0.002 on 2
  topics.
- **At 99 topics the best settings are close to their image encoder on emotion and genre.** P3 against Qwen default
  images: emotion 0.072 vs 0.070, style 0.194 vs 0.222, genre 0.446 vs 0.449. P2 seed 0 against CLIP images: 0.063
  vs 0.065, 0.250 vs 0.300, 0.412 vs 0.412. Style stays lower by 0.028 and 0.050. P1 seed 0 (0.051 / 0.166 / 0.349)
  and P4 are below CLIP images on every aspect. The spec expected PercepT core near the Qwen default embedding; P3's
  two seeds agree. Our reading, on one or two seeds.
- **Topics transfer to a CLIP reader only in part.** On the same Qwen topics (seeds 1 and 2), the CLIP patch reader
  (P4) gives 0.052 / 0.156 / 0.348 against the Qwen reader's (P3) 0.072 / 0.194 / 0.446, with dev AUC 0.79 against
  0.94. P4 is also below CLIP images clustered directly.
- **Stage 2 reader on CLIP topics.** Global vector (P2) against patch tokens (P1): +0.007, +0.025, −0.017 over 3
  seeds; on the one 99-topic seed, P2 is higher on every aspect (0.063 / 0.250 / 0.412 against 0.051 / 0.166 /
  0.349). Within the topic-count spread.
- **Against our network.** The comparator P2 is 0.045 to 0.241 below our best default-input network on the image
  side. Against the best 99-topic seed per aspect the gap is 0.009 (emotion, P3), 0.098 (style, P2 seed 0) and 0.012 (genre, P3). Selection: our rows
  and the comparator were both chosen on these 5,000 paintings (spec §8).

## Deviations

- None from the amended spec. The topic count case is a listed risk (spec §8: report, no rerun with another γ).
- The GPU queue took 17 minutes, not the estimated 35: the Stage 2 steps took 3 to 8 s each.
- The CPU queue was not rerun, as the handoff says; rows 1, 6, 10, 13 and the gate are the first run's.
- The Codex brief's comparator had no "established" filter; the established pick comes from our sheet, and both
  agree on every number.

## Files and storage

- **Outputs:** `src/test/20261018_percept_core/outputs/` (gitignored): `stage1/` 5.4 MB, `stage2/` 11 MB, `repeat/`
  1.7 MB, `stage1_repeat.json`, `stage0.json`, `pc_results.json`, `pc_tables.md`, `pc_ablation_table.md`; the first
  run's copies `*_run1.*`. Logs: `logs/`, `pc_queue_gpu.log`, `pc_queue_status.txt` (the block after
  `PC_gpu_SUBMIT 2026-10-19T09:50:39`).
- **Re-derivation:** `rederive_run2/` (brief, scripts, summary, comparison; the results JSON is gitignored).
- **Storage:** nothing deleted (unattended chat). New files total about 18 MB; nothing of 1 GB or more, so no new
  ledger row.
