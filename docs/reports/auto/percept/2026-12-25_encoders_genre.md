# CLIP L/14 and SigLIP 1 each pass the held emotion test; a relational term reaches the genre target on development data

> 2026-10-26 (sequence name 12-25) · Spec: [design](../../../superpowers/specs/2026-10-25-encoders-and-genre-design.md) · Decision rule: [binding](../../../../src/test/20261025_encoders_genre/DECISION_RULE.md) @ f0f7c91 · Run log: [round log](../../../../src/test/20261025_encoders_genre/20261025_encoders_genre_run_log.md) · Previous round: [12-24](2026-12-24_recipe_backbones.md)

## Results brief

**What we tried.** We tested four more encoders on held paintings with the current recipe and Qwen’s graphs and teacher. We also measured development gains and tested genre loss changes.

**Outcome against the bar.** A pass needs emotion AMI at least 0.0100 above the strongest eligible PercepT entry, with the interval above zero. AMI measures cluster agreement with labels.

| Encoder | Emotion AMI above PercepT | Held outcome |
|---|---|---|
| CLIP B/16 | +0.0070 [+0.0016, +0.0085] | Inconclusive, full scale |
| CLIP L/14 | +0.0122 [+0.0068, +0.0136] | Pass, full scale |
| SigLIP 1 | +0.0162 [+0.0106, +0.0179] | Pass, full scale |
| DINOv2-large | -0.0041 [-0.0094, -0.0030] | Fail, sample scale |

All seven encoders gain emotion AMI over their own PercepT arm on development paintings. The relational term at weight 8 reaches genre AMI 0.5018 against the instructed Qwen genre image's 0.4989 at seed 0.

Fresh seeds give genre AMI 0.5022 against the current recipe's 0.4694. The effect is +0.0328 [+0.0235, +0.0442]; emotion and style stay within the recipe's seed ranges.

**What it does not show.** Every held claim needs Qwen3-VL-Embedding at training and covers k-means only: the Leiden emotion intervals of CLIP L/14 and SigLIP 1 include zero. CLIP B/16 fails on ArtELingo-28; the other three results are qualified: positive estimate, interval includes zero. Development gains support no held claim. Prior recipe selection inflates Qwen’s gain. The genre result is Qwen only, development only, selected among seven variants on the same paintings. Only weight 8 ran on fresh seeds; none ran on held paintings.

**Options for next.**
1. Test genre on held paintings with a new rule. Seed-0 training took 39 s; a held test also needs inference and scoring.
2. Compare relational weights on fresh seeds. Costs minutes of GPU training per run; could resolve the weight choice.
3. Pause genre and write the separate encoder claims. Costs writing time.

**Your reading.**

<details>
<summary>Our reading and recommendation (open after writing yours)</summary>

**Our reading.** The encoder results fit our priors (rule §1): CLIP L/14 passes, SigLIP 1 passes, CLIP B/16 is inconclusive, DINOv2 fails. The genre result is the surprise. Every relational weight gives at least +0.0261 on genre against a prior of at most +0.0120. We are fairly sure of the genre gain on development data, because fresh seeds agree with seed 0. We are less sure which weight is best, and selection on the same development paintings remains a limitation.

**Our recommendation.** Compare weights 2, 4, 8 and above on fresh seeds, then fix one variant for a held test.

</details>

## What ran

All planned encoder tests and genre variants completed.

- **Stages:** Held emotion, current recipe, seeds 0 to 4; development lift, seven encoders; genre development, five variants and two extras at seed 0, winner and recipe at seeds 1 to 3.
- **Data:** Held: 12,281 paintings; ArtELingo-28: 746. Development: 29,232 paintings, genre labels on 23,682. Held comparisons reuse prior outputs.
- **Code:** “r10 run 21”, main checkout, RTX 3090 and CPU, commit 75d2225; 2026-10-25 22:58 to 2026-10-26 03:34. No candidate deferred.

The build ran gates X0, X1, X2, VG1 and VL1 (a) on this round's code on 2026-10-25 (19:17 to 22:40); all pass, and the run's queues reused those result files by design. Times are file mtimes.

Store and genre queues ran together under the shared GPU lock.

## Results

### Stage i: held emotion, per candidate

CLIP L/14 passes its held emotion test. SigLIP 1 passes its held emotion test. CLIP B/16 is inconclusive; DINOv2-large fails at sample scale and therefore has no full-scale test.

PercepT core’s own reading wins every comparator row, including the matched arm in the comparison.

Table codes: `A_cand` is AMI, `d` its effect, `B1-own` core’s own reading. Steps 3/4/5 and S1/S2/A mean sample/full/ArtELingo-28; GO passes, IN inconclusive, NO/FAILS fails, QUAL qualified.

| ID | Step | A_cand | Bar entry (value) | d [95% interval] | Outcome |
|---|---|---:|---|---|---|
| K6 CLIP B/16 | 3 | 0.0944 | B1-own (0.0855) | +0.0089 [+0.0036, +0.0101] | S1-IN |
| K6 | 4 | 0.0945 | B1-own (0.0874) | +0.0070 [+0.0016, +0.0085] | S2-IN |
| K6 | 5 | 0.1276 | B1-own (0.1277) | -0.0001 [-0.0259, +0.0077] | A-FAILS |
| K7 CLIP L/14 | 3 | 0.0992 | B1-own (0.0855) | +0.0138 [+0.0083, +0.0150] | S1-GO |
| K7 | 4 | 0.0996 | B1-own (0.0874) | +0.0122 [+0.0068, +0.0136] | S2-GO |
| K7 | 5 | 0.1402 | B1-own (0.1277) | +0.0125 [-0.0140, +0.0191] | A-QUAL |
| K8 SigLIP 1 | 3 | 0.1046 | B1-own (0.0855) | +0.0191 [+0.0137, +0.0202] | S1-GO |
| K8 | 4 | 0.1036 | B1-own (0.0874) | +0.0162 [+0.0106, +0.0179] | S2-GO |
| K8 | 5 | 0.1477 | B1-own (0.1277) | +0.0200 [-0.0061, +0.0250] | A-QUAL |
| K5 DINOv2 | 3 | 0.0814 | B1-own (0.0855) | -0.0041 [-0.0094, -0.0030] | S1-NO |
| K5 | 5 (sample) | 0.1253 | B1-own (0.1229) | +0.0024 [-0.0227, +0.0069] | A-QUAL |

Licensed claim at full scale, under [12-24's spec §6](../../../superpowers/specs/2026-10-25-recipe-and-backbones-design.md#6-success-criteria), separately for CLIP L/14 and for SigLIP 1: on 12,281 held paintings, the student's emotion AMI is at least 0.0100 above the strongest PercepT core entry, the matched arm included, with a positive paired interval over paintings and seeds. Training: 42,969 paintings with Qwen3-VL-Embedding's instructed graphs, teacher and default captions; evaluation: k-means with 20 clusters, five seeds. The sample-scale pass licenses the same sentence with 13,400 training paintings. On ArtELingo-28 both are qualified; CLIP B/16 fails there and DINOv2-large is qualified at sample scale.

### Stage ii: development lift, per encoder

Each of the seven development lift intervals is above zero. Each student trains on 13,400 paintings; evaluation uses 29,232 development paintings, five seeds and 2,000 bootstrap draws.

Lift is student emotion AMI minus its own PercepT arm’s best reading, here always the own reading. L-HOLDS means a positive interval; codes identify runs.

| Encoder (student / own arm) | Student AMI | Own best | Frozen image | Lift [95% interval] | Outcome |
|---|---:|---:|---:|---|---|
| Qwen (K4 / B1) | 0.1050 | 0.0844 | 0.0690 | +0.0206 [+0.0171, +0.0223] | L-HOLDS |
| SigLIP 2 (K2 / M2) | 0.1042 | 0.0756 | 0.0571 | +0.0286 [+0.0249, +0.0302] | L-HOLDS |
| CLIP B/32 (K3 / M3) | 0.0905 | 0.0708 | 0.0624 | +0.0196 [+0.0163, +0.0209] | L-HOLDS |
| CLIP B/16 (K6 / M6) | 0.0952 | 0.0728 | 0.0643 | +0.0225 [+0.0195, +0.0235] | L-HOLDS |
| CLIP L/14 (K7 / M7) | 0.1031 | 0.0772 | 0.0648 | +0.0259 [+0.0229, +0.0271] | L-HOLDS |
| SigLIP 1 (K8 / M8) | 0.1073 | 0.0784 | 0.0631 | +0.0288 [+0.0250, +0.0307] | L-HOLDS |
| DINOv2 (K5 / M5) | 0.0839 | 0.0722 | 0.0587 | +0.0117 [+0.0090, +0.0127] | L-HOLDS |

Each row licenses only that encoder’s development gain.

### Stage iii: genre variants on Qwen, seed 0

The relational term at weight 8 wins the guarded comparison on development data.

| Current recipe reference | Emotion | Style | Genre |
|---|---:|---:|---:|
| Seed-0 AMI | 0.1071 | 0.3987 | 0.4674 |
| Seed range | 0.0046 | 0.0082 | 0.0128 |

Guards: emotion effect ≥ -0.0046; style effect ≥ -0.0082. Table codes: GT raises teacher weight, GR adds the relational term, GA adds regression; suffixes give weights. `A_genre` is genre AMI; `e` its recipe effect.

| Variant | A_genre | e emotion [95%] | e style [95%] | e genre [95%] | Guarded |
|---|---:|---|---|---|---|
| GT-2 (teacher weight x2) | 0.4594 | -0.0020 [-0.0034, -0.0006] | -0.0038 [-0.0057, -0.0018] | -0.0079 [-0.0107, -0.0051] | yes |
| GT-4 (teacher weight x4) | 0.4478 | -0.0013 [-0.0026, +0.0001] | -0.0003 [-0.0020, +0.0014] | -0.0195 [-0.0228, -0.0165] | yes |
| GR-0.5 (relational, weight 0.5) | 0.4935 | -0.0064 [-0.0078, -0.0049] | -0.0034 [-0.0053, -0.0013] | +0.0261 [+0.0234, +0.0288] | no (emotion) |
| GR-2 | 0.4948 | +0.0001 [-0.0012, +0.0015] | -0.0016 [-0.0035, +0.0002] | +0.0274 [+0.0243, +0.0306] | yes |
| GA-1 (regression term, weight 1) | 0.4675 | -0.0045 [-0.0058, -0.0031] | -0.0093 [-0.0111, -0.0073] | +0.0002 [-0.0026, +0.0030] | no (style) |
| GR-4 (extra) | 0.4938 | -0.0004 [-0.0019, +0.0010] | -0.0002 [-0.0019, +0.0016] | +0.0264 [+0.0232, +0.0295] | yes |
| GR-8 (extra) | 0.5018 | -0.0032 [-0.0045, -0.0019] | -0.0052 [-0.0070, -0.0034] | +0.0345 [+0.0315, +0.0377] | yes |

Weight 2 led the first five variants under the guards, triggering extras at weights 4 and 8. Weight 8 reaches the instructed Qwen genre image’s target (0.4989); the promising threshold is 0.4861. Its gain exceeds one seed range, triggering fresh seeds.

Fresh seeds 1 to 3: weight-8 relational term (GR-8) against current recipe (R6-5).

| Aspect | Mean, GR-8 | Mean, R6-5 | e (3-seed means) [95%] |
|---|---:|---:|---|
| Emotion | 0.1049 | 0.1045 | +0.0004 [-0.0012, +0.0022] |
| Style | 0.3931 | 0.3941 | -0.0010 [-0.0056, +0.0028] |
| Genre | 0.5022 | 0.4694 | +0.0328 [+0.0235, +0.0442] |

**Verdict under the rule:** the genre target is reached (III-GO), and the fresh-seed gain repeats with both guards met (C-HOLDS).

The ceiling is “promising on development data”.

### Validity checks

Recorded gates pass and independent re-derivations agree; separate output-validity records are absent.

| Check and meaning | Result |
|---|---|
| X0: extractor regression | Pass; new and old extractors equal bit for bit. Stored-row difference: CLIP B/32 0; SigLIP 2 maximum 0.0112. |
| X1, X2: student, matched-arm and scorer regressions | Pass. |
| V5: store content, counts and order | Pass for all eight stores; the 1% retrieval bar is not void. Rates and cosine below. |
| V2: held re-derivation | Agree at steps 3, 4, 5. |
| VL1: development scorer and inference regression | Pass. |
| VL2, VL3: lift re-derivation and exports | Agree; pass. |
| VG1, VG2: genre regression and re-derivation | Pass; agree in all three phases. |
| Fresh-seed recipe parity | Pass: recipe weights equal 12-24's Qwen results on all three aspects. |

Top-1 image-to-caption retrieval rates for the store check:

| Encoder | First 1,000 development paintings | ArtELingo-28 |
|---|---:|---:|
| CLIP B/32 (calibration) | 0.159 | 0.555 |
| SigLIP 2 (calibration) | 0.296 | 0.855 |
| CLIP B/16 | 0.178 | 0.609 |
| CLIP L/14 | 0.227 | 0.684 |
| SigLIP 1 | 0.238 | 0.838 |

DINOv2 matches 49,121 archived paintings: minimum cosine 0.9926, mean 0.9999891.

Steps 3 and 4 record no non-finite held output; genre scores record no failed run. Real-run logs record no rerun or run beyond prescribed variants, extras and fresh seeds. These records do not establish every output-validity condition.

## Analysis

**Our reading:** three sample-scale effects fall within their prior ranges. SigLIP 1 is slightly above its range. 

### Stage i diagnostics (decide nothing)

The four students remain below the instructed Qwen image on genre. Margins at full scale (DINOv2 at sample scale): student minus the comparator's best entry; T is the instructed Qwen image.

| Candidate | Emotion vs T | Emotion Leiden | Style vs PercepT | Style vs T | Genre vs PercepT | Genre vs T |
|---|---:|---:|---:|---:|---:|---:|
| CLIP B/16 | +0.0024 | -0.0030 | +0.1597 | -0.0132 | +0.0266 | -0.0487 |
| CLIP L/14 | +0.0075 | +0.0014 | +0.1827 | +0.0108 | +0.0221 | -0.0408 |
| SigLIP 1 | +0.0116 | +0.0036 | +0.2054 | +0.0325 | +0.0163 | -0.0360 |
| DINOv2 | -0.0107 | -0.0056 | +0.1060 | -0.0683 | +0.0133 | -0.0517 |

**Our reading:** every student is above its frozen image on emotion (+0.0211 to +0.0364) and above PercepT on style and genre. Only SigLIP 1 and CLIP L/14 reach the instructed image on style; none reaches it on genre, as in 12-24. CLIP B/16's Leiden emotion margin is negative, -0.0030 [-0.0076, -0.0003], like CLIP B/32's in 12-24 (-0.0043); the Leiden intervals of CLIP L/14 and SigLIP 1 include zero, so the claims are k-means only. Seed spreads, best epochs and matched arms: `outputs/part2/reported_stage{3,4}.md`.

### Stage ii statistics (descriptive, seven points)

**Our reading:** frozen-image AMI does not track lift (Spearman +0.036); own-arm AMI tracks it weakly (+0.571). Seven points support no causal claim. Qwen’s prior selection affects correlations. Full statistics: `outputs/stage_ii/stats.md`.

Spearman rho of each encoder statistic (on its default image) with the emotion lift:

| Statistic | Emotion | Style | Genre |
|---|---:|---:|---:|
| 20-NN label purity | +0.464 | +0.929 | +0.893 |
| Cosine silhouette | +0.893 | +0.964 | +0.286 |

Other statistics: top component's variance share -0.786, mean pairwise cosine +0.179, hubness skew -0.214, maximum in-degree -0.071, share with zero in-degree +0.500. DINOv2 has the lowest mean pairwise cosine (0.1785) and the lowest purity on all three aspects.

**Our reading:** the style statistics track the lift best, but SigLIP 1, CLIP L/14 and SigLIP 2 have the highest style purity, so this may only separate larger encoders from smaller ones.

Held and development ranks correlate strongly; lifts use each encoder’s own PercepT arm.

| Encoder | Held lift (sample) | Rank held | P lift | Rank P |
|---|---:|---:|---:|---:|
| Qwen | 0.0190 | 6 | 0.0206 | 5 |
| SigLIP 2 | 0.0243 | 1 | 0.0286 | 2 |
| CLIP B/32 | 0.0202 | 5 | 0.0196 | 6 |
| CLIP B/16 | 0.0218 | 3 | 0.0225 | 4 |
| CLIP L/14 | 0.0215 | 4 | 0.0259 | 3 |
| SigLIP 1 | 0.0241 | 2 | 0.0288 | 1 |
| DINOv2 | 0.0088 | 7 | 0.0117 | 7 |

**Our reading:** rankings broadly agree (Spearman +0.893), but six encoders change rank. The two SigLIPs lead at sample scale; DINOv2 is last. Held lifts are smaller than development lifts, except CLIP B/32 at sample scale.

Full-scale Spearman is +0.886 over six encoders. The complete rebuilt table: `outputs/stage_ii/held_table.md`.

### Stage iii diagnostics (decide nothing)

The winner exceeds the recipe on both descriptive genre readings: old 5,000 paintings, 0.5119 against 0.4794; Leiden, 0.4968 against 0.4790. All variants and aspects: `outputs/stage_iii/reported.md`.

**Our reading:** the lowest relational weight already gives a large genre gain, but fails the emotion guard. Regression gives little gain.

| Regression diagnostic | Genre AMI |
|---|---:|
| Adaptor | 0.4985 |
| Head | 0.4675 |
| Adaptor minus head | +0.0310 |
| Ridge probe reference | 0.5040 |
| Instructed Qwen image reference | 0.4989 |

**Our reading:** the adaptor may absorb the gain.

**Our reading:** the genre effect repeats on fresh seeds. Selection among seven variants on the same development paintings remains a limitation. Weights 2 and 4 have no fresh-seed test.

## Deviations

The smoke gap and held-table race were the main issues.

- **Smoke gap:** A missing seed-1 export stopped smoke: the scorer needs five seeds, smoke training creates one. Full-scale training, ArtELingo-28 with models and outcome reporting lacked smoke tests. The agent started without longer smoke; no real-run failure was recorded.
- **Late queues:** Development scoring and sample-scale reporting started at 00:42 during full-scale training, after all sample-scale outcomes.
- **Held-table race:** At 02:45, only CLIP B/16 among new encoders had a full-scale row. The queue checked file existence while reporting continued. The first table was archived in `outputs/aside/20261026_held_table_race/`. This report uses the 03:25 rebuild after reporting finished; code was unchanged.

## Files and storage

Outputs record results and the keep/pending policy.

- **Outputs:** `/data/SSD2/cosir_r10/20261025_encoders_genre/outputs/` (gitignored): held `part2/verdict_stage{3,4,5}.json`, development `stage_ii/{lift,stats,held_table}.json`, genre `stage_iii/score_{first,all,fresh}.json`; diagnostics and queue logs alongside.
- **Storage:** The queue lists teacher-weight and regression weights for pending deletion, and keeps four relational variants and fresh-seed weights. The next storage summary (the decide chat) decides what stays; items of 1 GB or more then get their disk ledger rows.
