# Why ARI and ACC miss the teacher on emotion (r20 decide, 2026-11-04)

A diagnostic asked for by the user at decision C of r17: "ARI and ACC on teacher seems worth digging more cause half
not win. So do we have a reason yet?" Nothing here changes r17's cells, bars or picks; no outcome word applies.

## Setup

Same partitions as r17 (imports `ns_clu_cells`, `ns_metrics`, `ns_clu.paired_cell`; no r17 file changed), at the class
count, c* (5 seeds, 3 states) against the Qwen teacher (3 states). Replay against `verdicts.json`: largest difference
2.2e-16 over the 16 held emotion and genre runs. Local CPU queue: `tg_why.py` 12.7 min, `tg_genre_link.py` 5.2 min,
`tg_genre_ami.py` under 3 min, unqueued.

- `tg_why.py`: per-class parts of ARI, ACC and MI (they sum to the metric), cluster balance, class-balanced ARI, ACC and
  NMI, and emotion ARI and ACC with one class masked after clustering (2,000 draws, r17's bootstrap). Held rows for
  emotion and genre, validation rows for emotion.
- `tg_genre_link.py`: rows with both labels; the contentment cluster's precision, size, genre mix and recall per genre,
  AMI against genre, emotion ARI within each genre.
- `tg_genre_ami.py`: genre labels used directly as an emotion partition, beside Qwen c* and the teacher on the same rows.

JSON outputs: `outputs/` (gitignored, 244 KB with the scripts).

## Split of the 48 teacher ARI and ACC cells

| Aspect | Cells | WIN | Interval reaches 0 | Whole interval below 0 |
|---|---:|---:|---:|---:|
| Emotion | 16 | 0 | 7 | 9 |
| Genre | 16 | 8 | 6 | 2 |
| Style | 16 | 14 | 2 | 0 |

## ArtELingo emotion, held rows, four encoders

| Quantity | Result |
|---|---|
| ARI difference | -0.0101 to -0.0144 |
| Contentment's part of it | -0.0175 to -0.0238 |
| Other eight classes together | +0.0074 to +0.0107 |
| ACC difference | -0.0029 to -0.0080 |
| Contentment's part of it | -0.0169 to -0.0257 |
| Other eight classes together | +0.0094 to +0.0228 |
| MI difference (nats) | +0.0192 to +0.0275 |
| Contentment's part of it | -0.0011 to -0.0038 |
| Contentment recall in its matched cluster | c* 0.256 to 0.282, teacher 0.331 |

Contentment masked after clustering: ARI d +0.0053 to +0.0104, all four intervals above 0; ACC d +0.0033 to +0.0211,
two above 0 (Qwen, CLIP L/14), two reach 0 (SigLIP 2, SigLIP 1). Controls: with sadness, amusement or awe masked, ARI d
stays negative with all 12 intervals below 0. Contentment is 35% of the rows and about 60% of the same-class pairs.

Validation rows show the same direction, with a smaller gap: ARI d -0.0076 to +0.0028, contentment's part -0.0027 to
-0.0150; contentment masked: ARI d +0.0111 to +0.0151, ACC d +0.0116 to +0.0267, all eight intervals above 0.

Class-balanced point values (no interval): ARI 0.0543 to 0.0570 against 0.0483; ACC 0.2373 to 0.2455 against 0.2286;
NMI 0.0939 to 0.0996 against 0.0843.

## How: the contentment cluster (rows with both labels, 9,949)

| | Teacher | c* (four encoders) |
|---|---:|---:|
| Share of rows | 20.0% | 15.1% to 15.8% |
| Precision for contentment | 0.651 | 0.669 to 0.685 |
| Landscapes in the cluster | 73% | 64% to 72% |
| Contentment landscapes caught | 80% | 58% to 65% |

60% of ArtELingo landscapes have contentment as their label (30% on MMArt). So c*'s cluster is as pure and smaller.

| Rows with both labels | Genre labels as partition | Teacher | c* on Qwen |
|---|---:|---:|---:|
| ArtELingo AMI | 0.062 | 0.086 | 0.099 |
| ArtELingo ARI | 0.055 | 0.077 | 0.062 |
| ArtELingo ACC | 0.258 | 0.263 | 0.256 |
| MMArt AMI | 0.034 | 0.046 | 0.049 |
| MMArt ARI | 0.022 | 0.029 | 0.030 |
| MMArt ACC | 0.200 | 0.204 | 0.206 |

Within-genre emotion ARI on ArtELingo (row-weighted mean, point values): teacher 0.0618; c* 0.0597, 0.0664, 0.0653,
0.0606. The gap on the same rows is -0.012 to -0.017, so most of the teacher's lead lies between genres.

The first guess, that c* carries less genre, is wrong: AMI of the emotion partition against genre is 0.391 for the
teacher and 0.385 to 0.400 for c* on ArtELingo, 0.399 and 0.383 to 0.411 on MMArt.

## MMArt emotion, test rows

ARI has no gap (d -0.0012 to +0.0002). ACC d -0.0026 to -0.0092. Contentment's part is -0.0029 to -0.0092, fear's
-0.0060 to -0.0072, and amusement and "something else" offset part of them. Contentment masked: ACC d -0.0023 to
+0.0058, all four intervals reach 0. Controls (sadness, amusement, awe masked): 9 of 12 intervals below 0. Fear was not
masked. Class-balanced point values: ARI 0.0302 to 0.0323 against 0.0288; ACC 0.2049 to 0.2070 against 0.2013.

## Genre

Six of the eight non-WIN cells reach 0. The two clear ones are ArtELingo on CLIP L/14: ARI -0.0171, ACC -0.0091. Most of
that is the portrait class (ARI part -0.0264, ACC part -0.0244) and landscape (-0.0125 in both). One encoder, no pattern.

## Root cause and limits

Finding: on ArtELingo the gap comes from one class. The teacher forms a larger landscape cluster that holds more of
contentment, the largest class, at about the same precision. c* gains in the other classes. AMI and NMI count those
gains, because contentment's part of MI is small; ARI and ACC weigh the largest class most.

Limits: a decomposition and a masking test show where the gap sits, not what in training causes it. Not checked: where
c*'s other landscape rows go, and whether they split by emotion. The balanced and within-genre values have no interval.
The mask and the balanced weighting were chosen after the results, so they stay descriptive.
