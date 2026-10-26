# Encoders and genre: CLIP L/14 and SigLIP 1 pass the held emotion test; a relational term for genre is promising on development data

> 2026-10-26 (sequence name 12-25) · Spec: [design](../../../superpowers/specs/2026-10-25-encoders-and-genre-design.md) · Decision rule: [binding](../../../../src/test/20261025_encoders_genre/DECISION_RULE.md) @ f0f7c91 · Run log: [round log](../../../../src/test/20261025_encoders_genre/20261025_encoders_genre_run_log.md) · Previous round: [12-24](2026-12-24_recipe_backbones.md)

## Results brief

**What we tried.** Report 12-24 picked a recipe (R6-5) and tested three image encoders. This round has three stages. Stage i gives four more encoders the same held test on 12,281 ArtELingo paintings, each against the strongest PercepT entry. Stage ii measures, for seven encoders, the student's emotion lift over its own PercepT arm on 29,232 development paintings. Stage iii tries five loss changes for the genre head, aimed at the instructed Qwen image, on Qwen at seed 0 and then on fresh seeds.

AMI is cluster agreement with labels. A student is our model on one encoder's image. Lift is the student's AMI minus its own PercepT arm's best reading.

**Outcome against the bar.** Stage i needs d of at least 0.0100 with a positive lower bound over PercepT core (0.0874 at full scale).

| Encoder (code) | Held d, full scale | ArtELingo-28 d | Development lift, emotion |
|---|---|---|---|
| CLIP B/16 (K6) | +0.0070 [+0.0016, +0.0085] IN | -0.0001 [-0.0259, +0.0077] FAILS | +0.0225 [+0.0195, +0.0235] HOLDS |
| CLIP L/14 (K7) | +0.0122 [+0.0068, +0.0136] GO | +0.0125 [-0.0140, +0.0191] QUAL | +0.0259 [+0.0229, +0.0271] HOLDS |
| SigLIP 1 (K8) | +0.0162 [+0.0106, +0.0179] GO | +0.0200 [-0.0061, +0.0250] QUAL | +0.0288 [+0.0250, +0.0307] HOLDS |
| DINOv2-large (K5) | sample scale: -0.0041 [-0.0094, -0.0030] NO-GO | sample scale: +0.0024 [-0.0227, +0.0069] QUAL | +0.0117 [+0.0090, +0.0127] HOLDS |

IN is inconclusive. QUAL is qualified: d above 0, lower bound not. Sample-scale results and the lift for three more encoders are in Results.

Stage iii picked the relational term at weight 8. Its seed-0 genre AMI is 0.5018. The comparators are R6-5 at 0.4674 and the instructed Qwen image at 0.4989 (III-GO).
On fresh seeds 1 to 3 its genre AMI is 0.5022 against R6-5 at 0.4694, effect +0.0328 [+0.0235, +0.0442]. Emotion changed by +0.0004 and style by -0.0010 (C-HOLDS).
Raising the teacher weight lowered genre AMI (all variants and codes are in Results). The regression term gave +0.0002 [-0.0026, +0.0030].

**What it does not show.** Stage i covers held emotion, one encoder at a time, one recipe; ArtELingo-28 stays qualified. Stage ii is a development reading with no held claim, and the Qwen lift is inflated by 12-24's pick. Stage iii is development data, Qwen only, picked on one seed among seven variants. Only the weight-8 variant ran on fresh seeds. No genre variant was tested on held paintings.

**Options for next.**
1. Held claim test of the weight-8 variant on genre. It needs a new rule and a held read; a run is short (39 s for the seed-0 run). It would show whether the gain holds on held paintings.
2. More development work on the relational term: weights above 8, and other weights on fresh seeds. It costs minutes of GPU per run. It would show how much of the gain comes from the term and how much from its weight.
3. Pause genre and write up the CLIP L/14 and SigLIP 1 results. It costs writing time and fixes the per-encoder claims.

**Your reading.**

<details>
<summary>Our reading and recommendation (open after writing yours)</summary>

**Our reading.** Our view: the encoder results fit the prior. SigLIP 1 and CLIP L/14 pass the held test, CLIP B/16 is inconclusive, DINOv2 fails, and every student lifts over its own PercepT arm on development data. The genre result is the surprise. Weights 0.5 to 4 of the relational term gave +0.0261 to +0.0274, far above our prior (-0.005 to +0.012), and weight 8 gave +0.0345. We are fairly sure of the genre gain on development data, because fresh seeds agree with seed 0. We are less sure which weight is best.

**Our recommendation.** Our view: option 2, then option 1. Compare weights 2, 4, 8 and above on fresh seeds first, so the held test gets one fixed variant. Keep the encoder results as per-encoder claims.

</details>

## What ran

- **Stages and types:** Stage i: claim test on held ArtELingo (12,281 paintings; ArtELingo-28, 746), recipe R6-5, seeds 0 to 4; steps 3, 4, 5 are sample scale, full scale, ArtELingo-28, as in 12-24. Stage ii: confirmation on the 29,232 development paintings (P), seven encoders. Stage iii: development experiment on Qwen, five variants and two extras at seed 0, then the winner and R6-5 on seeds 1 to 3.
- **Data and seeds:** P (genre labels on 23,682) is development data. Stage i reads held data. Stage ii reads P; its held table reads 12-24 and stage i outputs.
- **Code:** run chat "r10 run 21", main checkout, RTX 3090 and CPU, commit 75d2225. Run 2026-10-25 22:58 to 2026-10-26 03:34; no candidate deferred.

## Results

### Stage i: held emotion, per candidate

12,281 held paintings (ArtELingo-28: 746), five seeds. Step 3 is sample scale, 4 full scale, 5 ArtELingo-28. The bar entry is B1-own (PercepT core's own reading) in every row.

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

DINOv2 has no step 4 (its sample-scale outcome was NO-GO). Verdicts, one per encoder:
- **CLIP L/14:** full-scale GO, so 12-24's claim sentence holds for this encoder (the sample-scale GO licenses it at 13,400 training paintings). ArtELingo-28: qualified only.
- **SigLIP 1:** full-scale GO, the same sentence for this encoder. ArtELingo-28: qualified only.
- **CLIP B/16:** inconclusive at both scales; it fails on ArtELingo-28.
- **DINOv2-large:** NO-GO at sample scale.

### Stage ii: development lift, per encoder

29,232 paintings, five seeds, 2,000 bootstrap draws. Own best is each own arm's best emotion reading (all "own").

| Encoder (student / own arm) | Student AMI | Own best | Frozen image | Lift [95% interval] | Outcome |
|---|---:|---:|---:|---|---|
| Qwen (K4 / B1) | 0.1050 | 0.0844 | 0.0690 | +0.0206 [+0.0171, +0.0223] | L-HOLDS |
| SigLIP 2 (K2 / M2) | 0.1042 | 0.0756 | 0.0571 | +0.0286 [+0.0249, +0.0302] | L-HOLDS |
| CLIP B/32 (K3 / M3) | 0.0905 | 0.0708 | 0.0624 | +0.0196 [+0.0163, +0.0209] | L-HOLDS |
| CLIP B/16 (K6 / M6) | 0.0952 | 0.0728 | 0.0643 | +0.0225 [+0.0195, +0.0235] | L-HOLDS |
| CLIP L/14 (K7 / M7) | 0.1031 | 0.0772 | 0.0648 | +0.0259 [+0.0229, +0.0271] | L-HOLDS |
| SigLIP 1 (K8 / M8) | 0.1073 | 0.0784 | 0.0631 | +0.0288 [+0.0250, +0.0307] | L-HOLDS |
| DINOv2 (K5 / M5) | 0.0839 | 0.0722 | 0.0587 | +0.0117 [+0.0090, +0.0127] | L-HOLDS |

Licensed claim, per encoder: "On 29,232 development ArtELingo paintings, the student on X (13,400 training paintings, 5 seeds) gives emotion AMI <lift> [interval] above X's own PercepT arm." It is a development reading with no held claim.

### Stage iii: genre variants on Qwen, seed 0

29,232 paintings (genre labels on 23,682). R6-5: emotion 0.1071, style 0.3987, genre 0.4674. Seed ranges 0.0046, 0.0082, 0.0128. The guard needs emotion effect at least -0.0046 and style effect at least -0.0082.

| Variant | A_genre | e emotion [95%] | e style [95%] | e genre [95%] | Guarded |
|---|---:|---|---|---|---|
| GT-2 (teacher weight x2) | 0.4594 | -0.0020 [-0.0034, -0.0006] | -0.0038 [-0.0057, -0.0018] | -0.0079 [-0.0107, -0.0051] | yes |
| GT-4 (teacher weight x4) | 0.4478 | -0.0013 [-0.0026, +0.0001] | -0.0003 [-0.0020, +0.0014] | -0.0195 [-0.0228, -0.0165] | yes |
| GR-0.5 (relational, weight 0.5) | 0.4935 | -0.0064 [-0.0078, -0.0049] | -0.0034 [-0.0053, -0.0013] | +0.0261 [+0.0234, +0.0288] | no (emotion) |
| GR-2 | 0.4948 | +0.0001 [-0.0012, +0.0015] | -0.0016 [-0.0035, +0.0002] | +0.0274 [+0.0243, +0.0306] | yes |
| GA-1 (regression term, weight 1) | 0.4675 | -0.0045 [-0.0058, -0.0031] | -0.0093 [-0.0111, -0.0073] | +0.0002 [-0.0026, +0.0030] | no (style) |
| GR-4 (extra) | 0.4938 | -0.0004 [-0.0019, +0.0010] | -0.0002 [-0.0019, +0.0016] | +0.0264 [+0.0232, +0.0295] | yes |
| GR-8 (extra) | 0.5018 | -0.0032 [-0.0045, -0.0019] | -0.0052 [-0.0070, -0.0034] | +0.0345 [+0.0315, +0.0377] | yes |

The closest method is the relational term: its weight-2 variant led the five first variants and was guarded, and the largest weight run was 2, so the extras were weights 4 and 8. **Winner:** GR-8, the relational term at weight 8. **III outcome:** III-GO (A_genre 0.5018 against T_g 0.4989, the instructed Qwen genre image; promising level 0.4861).

Fresh seeds 1 to 3, GR-8 against R6-5 (parity holds, see below):

| Aspect | Mean, GR-8 | Mean, R6-5 | e (3-seed means) [95%] |
|---|---:|---:|---|
| Emotion | 0.1049 | 0.1045 | +0.0004 [-0.0012, +0.0022] |
| Style | 0.3931 | 0.3941 | -0.0010 [-0.0056, +0.0028] |
| Genre | 0.5022 | 0.4694 | +0.0328 [+0.0235, +0.0442] |

**Verdict under the rule:** III-GO with C-HOLDS. The fresh-seed genre mean 0.5022 is above T_g by +0.0033 and above the promising level by +0.0161 (file values). Licensed sentence: "On 29,232 development ArtELingo paintings (Qwen; picked on one seed, tested on three fresh seeds), the relational term at weight 8 gives genre AMI 0.5022 against the recipe's 0.4694 on the fresh seeds (instructed Qwen image: 0.4989), with emotion and style within one seed range of the recipe." Genre labels cover 23,682 of the 29,232 paintings. The ceiling is "promising on development data".

### Validity checks

| Check | Result |
|---|---|
| X0 | Pass. New and old extractor rows are equal bit for bit. Against stored rows: CLIP B/32 difference 0 (0 of 512 rows), SigLIP 2 maximum 0.0112 (512 of 512); reported only. |
| X1, X2 | Pass (K2, M2, seed 0): outputs equal 12-24's; scorer items 112 (sample), 56 (full), maximum difference 0. |
| V5 | Pass for all eight new stores; the 1% bar is not void. Top-1 retrieval, first 1,000 of P: calibration CLIP B/32 0.159, SigLIP 2 0.296; CLIP B/16 0.178, CLIP L/14 0.227, SigLIP 1 0.238. ArtELingo-28: calibration 0.555, 0.855; new 0.609, 0.684, 0.838. DINOv2 against the archive (49,121 paintings): minimum cosine 0.9926, mean 0.9999891. |
| V1 to V4 | V2 agrees at steps 3, 4, 5 (1e-6). No V3 or V4 failure recorded (`nonfinite_held` empty). |
| VL1 to VL4 | VL1 (a) maximum difference 0.0 on 18 values; (b) K4 equal on 3 aspects; (c) M2 held equal, 12,281 rows. VL2 agrees. VL3: 140 of 140 files pass, maximum norm error 4.1e-05. VL4: no rerun. |
| VG1 to VG4 | VG1: R6-5 exports and states equal 12-24's; scorer and T_g differ by 0.0. VG2 agrees in all three phases. VG3, VG4: no failed run, no variant added or dropped. |
| Parity | Pass. R6-5 weights on seeds 1, 2, 3 equal 12-24's K4 on all three aspects. |

## Analysis

**Our reading:** the encoder results follow the rule's prior: every sample-scale d sits in or at the edge of its prior range (SigLIP 1 +0.0191 against 0.008 to 0.018). CLIP L/14 passed with a GO prior of 0.30.

### Stage i diagnostics (decide nothing)

Margins at full scale (DINOv2 at sample scale): student minus the comparator's best entry. T is the instructed Qwen image.

| Candidate | Emotion vs T | Emotion Leiden | Style vs PercepT | Style vs T | Genre vs PercepT | Genre vs T |
|---|---:|---:|---:|---:|---:|---:|
| CLIP B/16 | +0.0024 | -0.0030 | +0.1597 | -0.0132 | +0.0266 | -0.0487 |
| CLIP L/14 | +0.0075 | +0.0014 | +0.1827 | +0.0108 | +0.0221 | -0.0408 |
| SigLIP 1 | +0.0116 | +0.0036 | +0.2054 | +0.0325 | +0.0163 | -0.0360 |
| DINOv2 | -0.0107 | -0.0056 | +0.1060 | -0.0683 | +0.0133 | -0.0517 |

**Our reading:** every student is above its frozen image on emotion (+0.0211 to +0.0364) and above PercepT on style and genre. Only SigLIP 1 and CLIP L/14 reach the instructed image on style. No student reaches it on genre, as in 12-24 (SigLIP 2 -0.0375, CLIP B/32 -0.0577, Qwen -0.0527). CLIP B/16's Leiden emotion margin is negative, -0.0030 [-0.0076, -0.0003], like CLIP B/32's in 12-24 (-0.0043). The Leiden intervals of CLIP L/14 and SigLIP 1 include zero.
Emotion AMI over seeds at full scale (min to max): CLIP B/16 0.0928 to 0.0966, CLIP L/14 0.0987 to 0.1010, SigLIP 1 0.1016 to 0.1063. Best epochs: emotion 4 to 11, style 1 to 4, genre 1 to 2. The matched arms (0.0749, 0.0793, 0.0821) stay below B1 own (0.0874).

### Stage ii statistics (descriptive, seven points)

**Our reading:** the lift does not follow the frozen image's AMI (Spearman +0.036) and follows the own arm's AMI weakly (+0.571). The lowest lift belongs to DINOv2, which has the lowest mean pairwise cosine (0.1785) and the lowest 20-NN purity on all three aspects. With seven points we claim no cause. Qwen's lift is inflated by 12-24's pick. Norms are unit for all seven.

Spearman rho with the emotion lift: 20-NN purity emotion +0.464, style +0.929, genre +0.893; cosine silhouette emotion +0.893, style +0.964, genre +0.286; top component's variance share -0.786; mean pairwise cosine +0.179; hubness skew -0.214, maximum in-degree -0.071, share with zero in-degree +0.500.

**Our reading:** the style statistics track the lift best, but SigLIP 1, CLIP L/14 and SigLIP 2 have the highest style purity, so this may only separate larger encoders from smaller ones.

Held table (descriptive). Emotion lift on held paintings against P; Spearman +0.893 at sample scale (seven encoders), +0.886 at full scale (six; DINOv2 has no step 4).

| Encoder | Held lift (sample) | Rank held | P lift | Rank P | Held lift (full) |
|---|---:|---:|---:|---:|---:|
| Qwen | 0.0190 | 6 | 0.0206 | 5 | 0.0154 |
| SigLIP 2 | 0.0243 | 1 | 0.0286 | 2 | 0.0231 |
| CLIP B/32 | 0.0202 | 5 | 0.0196 | 6 | 0.0169 |
| CLIP B/16 | 0.0218 | 3 | 0.0225 | 4 | 0.0196 |
| CLIP L/14 | 0.0215 | 4 | 0.0259 | 3 | 0.0203 |
| SigLIP 1 | 0.0241 | 2 | 0.0288 | 1 | 0.0216 |
| DINOv2 | 0.0088 | 7 | 0.0117 | 7 | not run |

**Our reading:** the order on held and on P agrees; the two SigLIPs lead and DINOv2 is last on both. Held lifts are smaller than P lifts, except CLIP B/32 at sample scale.

### Stage iii diagnostics (decide nothing)

Genre AMI on the old 5,000 paintings and Leiden genre AMI on P (union 20-NN graph, mean of seeds 0 to 2). Emotion and style on both are in `stage_iii/reported.md`.

| Variant | R6-5 | GT-2 | GT-4 | GR-0.5 | GR-2 | GA-1 | GR-4 | GR-8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Old 5,000, genre | 0.4794 | 0.4704 | 0.4261 | 0.4939 | 0.4954 | 0.4597 | 0.5002 | 0.5119 |
| Leiden on P, genre | 0.4790 | 0.4620 | 0.4577 | 0.4872 | 0.4957 | 0.4795 | 0.4948 | 0.4968 |

**Our reading:** the three views agree on genre. Every relational variant is above R6-5 on the old 5,000 and on Leiden, and both teacher-weight variants are below. The effect comes mostly from adding the term at all: weights 0.5, 2 and 4 give +0.0261, +0.0274 and +0.0264, weight 8 gives +0.0345. Our priors missed: positive for the teacher weight, -0.005 to +0.012 for relational weights 0.5 and 2. On Leiden, relational variants lose some style (0.3727 to 0.3783 against 0.3809).

**Our reading:** the regression term did not move the head. The adaptor's own genre AMI is 0.4985 against the head's 0.4675 (+0.0310 apart), the risk the spec named: the adaptor absorbs the gain. The ridge probe (0.5040) and T_g (0.4989) are references.

Best epochs at seed 0: emotion 3 and style 2 for every variant; genre 1 (R6-5, GT-2, GR-0.5), 2 (GA-1) or 3 (GT-4, GR-2, GR-4, GR-8). On fresh seeds the weight-8 variant has genre epochs 4, 2, 3 and R6-5 has 1, 2, 1. **Our reading:** the stronger relational variants train the genre head longer. The weight-8 variant's emotion cost at seed 0 (-0.0032 [-0.0045, -0.0019]) is not seen on fresh seeds (+0.0004). Seed-0 and fresh genre effects are close (+0.0345, +0.0328), so the pick among seven variants inflated little. The choice among weights 2, 4 and 8 is open, since only weight 8 ran on fresh seeds.

## Deviations

- **Stores and stage iii together.** The two queues started together at 22:59. Both take the GPU lock per step, so stores extracted while stage iii scored on the CPU.
- **Stage i smoke gap.** The scorer's smoke mode reads seeds 0 to 4 at steps 3 and 4. The runners' smoke trains seed 0 only (`FileNotFoundError ... emotion__s1.npz`). The smoke reached no outcome, so step 4, step 5 with models and a report with outcomes were not smoke-tested. The real run started without a longer smoke (agent default); nothing failed.
- **Late queues.** Stage ii and the step 3 report queue started during step 4 (00:42), after every candidate had a step 3 outcome.
- **Held table race.** The first held table (02:45) held step 4 for CLIP B/16 only, because the step 4 report was still being written. It was moved to `outputs/aside/20261026_held_table_race/` and rebuilt at 03:25, after that report. The queue's guard checks only that the report file exists, while the reporter writes one candidate at a time. The code was not changed. This report uses the rebuilt table.
- **Gate files from the build (found in the files).** The queues skipped the X0, X1, X2, VG1 and VL1 (a) steps because their result files existed. The files date from 2026-10-25, 19:17 to 22:40, before the run, and carry older code commits than the run's 75d2225. The rule asks for this round's code. All pass; the regression runs were not repeated.
- **No V3, V4 or VG3 record file** was found. We rely on `nonfinite_held`, the empty `failed` entries and the queue logs, which show no rerun.

## Files and storage

- **Outputs:** `/data/SSD2/cosir_r10/20261025_encoders_genre/outputs/` (gitignored): `stores/{x0,v5}.json`; `part2/{verdict,rederive_v2,reported}_stage{3,4,5}.json` and `reported_stage*.md`; `stage_ii/{vl1,lift,stats,held_table,rederive_vl2}.json`; `stage_iii/{score_first,score_all,score_fresh,rederive_vg2_*,fresh_parity,vg1,reported}.json` and `reported.md`; `queue_*_status.txt`; `aside/20261026_held_table_race/`.
- **Storage:** see the run chat's storage summary. The queue put the weights of the two teacher-weight variants and the regression variant on the pending deletions list. The weights of the four relational variants and the fresh-seed runs are kept.
