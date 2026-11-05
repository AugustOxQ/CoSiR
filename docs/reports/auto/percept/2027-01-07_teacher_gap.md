# Teacher gap in ARI and ACC: on ArtELingo emotion it comes from one class, contentment

> 2026-11-05 (sequence name 2027-01-07) · Spec: none (exploration in the r20 decide chat, asked for at decision C of r17) · Decision rule: none (descriptive) · [Run log](../../../../src/test/20261104_teacher_gap_why/20261104_teacher_gap_why_log.md)

## Results brief

**What we tried.** r17 gave 26 of 48 ARI and ACC cells against the teacher without a WIN, on paintings at the class
count ([report](2027-01-04_new_scores.md)). We looked for the reason. We split each gap into one part per class, masked
one class after clustering, weighted all classes equally, and compared the clusters with the genre labels. All of this
reads saved outputs; there was no training.

**Outcome against the bar.** There is no bar: this is an exploration, and r17's 412 cells stay as they are. On ArtELingo
emotion, contentment carries the whole gap. Its part of the ARI difference is -0.0175 to -0.0238, against a full
difference of -0.0101 to -0.0144. With contentment masked, c* is above the teacher in ARI on all four encoders
(+0.0053 to +0.0104, intervals above 0). The teacher's contentment cluster holds 20.0% of the rows at precision 0.651;
c*'s holds 15.1% to 15.8% at 0.669 to 0.685.

**What it does not show.** It shows where the gap sits, not what in training causes it. We did not check where c*'s
other landscape rows go, or whether they split by emotion. We chose the mask and the equal class weights after seeing
r17's results, so they give no claim. Equal-weight and within-genre values have no interval. The control classes are
smaller than contentment. On ArtELingo validation rows the gap is smaller and mostly absent. Only 11 of the 26 non-WIN
cells have an interval wholly below 0. On MMArt the reason is partial: fear also carries part of the ACC gap, and we did
not mask fear.

**Options for next.**

1. Use this as analysis text beside the teacher cells, in the paper. No compute.
2. Find where c*'s other landscape rows go. CPU minutes. It could tell if c* splits landscapes by emotion.
3. Add genre labels as a stated baseline row, with a rule set first and intervals. A short spec, CPU under 1 h. It could
   answer a reviewer who asks if the emotion clusters are genre clusters.

**Your reading.**

<details>
<summary>Our reading and recommendation (open after yours)</summary>

**Our reading.** c* trades recall in the largest class for gains in the other classes. AMI and NMI count the gains; ARI
and ACC weigh the largest class most. So the gap does not show a weak emotion head, and a new head may not close it (not
tested). The larger risk is the genre link. On ArtELingo, genre labels used as the partition reach ACC 0.258, level with
c* (0.256) and the teacher (0.263). They reach ARI 0.055, close to c* (0.062) and below the teacher (0.077).

**Our recommendation.** Our view: options 1 and 3. Keep AMI beside ARI and ACC, and say that emotion clusters on paintings
track genre in both c* and the teacher. Option 2 only if the paper needs the mechanism.

</details>

## What ran

- **Stages and types:** one exploration, three scripts, no bar.
- **Data and seeds:** ArtELingo held and MMArt test, five c* seeds, three k-means states; validation rows for emotion.
  A replay matched r17's `verdicts.json` with a largest difference of 2.2e-16.
- **Code:** `src/test/20261104_teacher_gap_why/`, importing r17's scorer with no change; local CPU queue, about 18 min.

## Results

The 48 teacher cells in ARI and ACC, by aspect:

| Aspect | Cells | WIN | Interval reaches 0 | Whole interval below 0 |
|---|---:|---:|---:|---:|
| Emotion | 16 | 0 | 7 | 9 |
| Genre | 16 | 8 | 6 | 2 |
| Style | 16 | 14 | 2 | 0 |

ArtELingo emotion, held rows, range over four encoders. Each class part sums to the metric's difference:

| Quantity | ARI | ACC | Mutual information (nats) |
|---|---:|---:|---:|
| c* minus teacher | -0.0101 to -0.0144 | -0.0029 to -0.0080 | +0.0192 to +0.0275 |
| Contentment's part | -0.0175 to -0.0238 | -0.0169 to -0.0257 | -0.0011 to -0.0038 |
| Other eight classes | +0.0074 to +0.0107 | +0.0095 to +0.0228 | +0.0211 to +0.0287 |

Masking one class after clustering (2,000 draws), c* minus teacher:

| Set, rows | Masked class | ARI | ACC |
|---|---|---:|---:|
| ArtELingo held | contentment | +0.0053 to +0.0104; 4 of 4 above 0 | +0.0033 to +0.0211; 2 of 4 above 0 |
| ArtELingo held | sadness, amusement or awe | -0.0107 to -0.0207; 12 of 12 below 0 | -0.0049 to -0.0120; 7 of 12 below 0 |
| ArtELingo validation | contentment | +0.0111 to +0.0151; 4 of 4 above 0 | +0.0116 to +0.0267; 4 of 4 above 0 |
| MMArt test | contentment | +0.0009 to +0.0028; 1 of 4 above 0 | -0.0023 to +0.0058; 4 of 4 reach 0 |

The control classes are 8% to 20% of the rows and contentment is 35%, so the controls are not matched in size.

Genre labels used directly as the partition, on rows with both labels (9,949 and 13,271), beside Qwen c*:

| Metric | ArtELingo genre labels | Teacher | c* | MMArt genre labels | Teacher | c* |
|---|---:|---:|---:|---:|---:|---:|
| AMI | 0.062 | 0.086 | 0.099 | 0.034 | 0.046 | 0.049 |
| ARI | 0.055 | 0.077 | 0.062 | 0.022 | 0.029 | 0.030 |
| ACC | 0.258 | 0.263 | 0.256 | 0.200 | 0.204 | 0.206 |

## Analysis

Our reading: on ArtELingo the gap is a difference in one cluster. These values use rows with both labels:

| Contentment's matched cluster | Teacher | c*, four encoders |
|---|---:|---:|
| Share of rows | 20.0% | 15.1% to 15.8% |
| Precision | 0.651 | 0.669 to 0.685 |
| Share of landscapes in it | 73% | 64% to 72% |
| Contentment landscapes caught | 80% | 58% to 65% |

**Why ARI and ACC, not AMI.** Contentment is 35% of the rows and about 60% of the same-class pairs. On MMArt it is about
25% of the pairs, and 30% of landscapes are contentment against 60% on ArtELingo. This fits MMArt's missing ARI gap.

**All classes weighted equally.** c* is above the teacher in ARI and ACC on every encoder and both sets (point values).
ArtELingo: ARI 0.0543 to 0.0570 against 0.0483; ACC 0.2373 to 0.2455 against 0.2286. MMArt: ARI 0.0302 to 0.0323
against 0.0288; ACC 0.2049 to 0.2070 against 0.2013. The spread over single partitions includes the teacher's value in
ACC on all eight encoder and set pairs, and in ARI on six of eight.

**Genre.** Our first guess, that c* carries less genre, was wrong. AMI of the emotion partition against genre is 0.391
for the teacher and 0.385 to 0.400 for c* on ArtELingo. Within one genre the lead mostly goes: emotion ARI 0.0618 for the
teacher, 0.0597 to 0.0664 for c*, against a gap of -0.012 to -0.017 on the same rows.

**MMArt ACC.** The difference is -0.0026 to -0.0092. Contentment's part is -0.0029 to -0.0092 and fear's is -0.0060 to
-0.0072; fear's is the larger on Qwen and SigLIP 1. Amusement and "something else" offset part of them. With contentment
masked all four intervals reach 0, and the point value stays below 0 on the two SigLIP encoders.

**Genre cells.** Six of the eight non-WIN cells reach 0. The two clear ones are ArtELingo on CLIP L/14 (ARI -0.0171, ACC
-0.0091), mostly portrait and landscape. We see no pattern across encoders.

## Deviations

An exploration the spec of r17 did not ask for. r17's stage 3 per-class AP named contentment; this round tests it.

On ArtELingo validation rows the gap is smaller and mostly absent. The ARI difference is -0.0076 to +0.0028 and the ACC
difference -0.0012 to +0.0088, with c* above the teacher in both on CLIP L/14. Contentment's part stays negative on all
four encoders (ARI -0.0027 to -0.0150).

## Files and storage

- **Outputs:** `src/test/20261104_teacher_gap_why/outputs/` (gitignored), with the scripts and the log, 244 KB.
- **Storage:** nothing of 1 GB or more left; r17's rows of the disk ledger are dated 2026-11-05.
