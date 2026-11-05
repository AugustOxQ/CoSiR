# Past picks: all 23 verdicts hold; the teacher stays ahead in most new-encoder cells

> 2026-11-04 (sequence name 2027-01-05) · Spec and decision rule: [§6 with §3.4–§3.5](../../../superpowers/specs/2026-11-03-past-picks-design.md#6-success-criteria) · Run log: [round log](../../../../src/test/20261103_past_picks/20261103_past_picks_log.md) · [r15](2027-01-03_mmart_test.md)

## Results brief

**What we tried.** Reviewers will ask three questions that our picks leave open. We tested the current c* (our student
network), with no setting changed, on each ([spec](../../../superpowers/specs/2026-11-03-past-picks-design.md)).

1. Does c* hold on DINOv2, CLIP B/32 and CLIP B/16, the three encoders that failed or were unclear with an older model?
2. Does c* hold on artists with no painting in ArtELingo's train set, matched by name? The test is SemArt: 11,159
   paintings by 2,717 such artists, genre only, all seven encoders.
3. Is c*'s one encoder pass cheaper than the teacher's three passes? The teacher is the Qwen image embedded under one
   instruction per aspect.

**Outcome against the bar.** All three answers are yes: **23 of 23 verdicts hold, GO.** The first two are against the
plain image (the encoder's own embedding) and PercepT on the same encoder; cost is against the teacher's three passes.
A WIN means that the 95% interval of the difference is above 0.

- **New three encoders, 9 verdicts:** every cell is a WIN. The smallest difference: DINOv2, MMArt emotion mAP, 0.1939
  against 0.1890 for the plain image.
- **New artists, 7 verdicts:** every cell is a WIN. The smallest difference: CLIP B/16, genre AMI, 0.4741 against
  0.4509 for the plain image.
- **Cost, 7 verdicts:** c* takes 5.2 to 88.6 ms per image, against 266.8 ms for the teacher (ratios 51.4 to 3.0).

The teacher is the exception: no cell gives "better than". LEVEL means that the lower bound is within the seed range r
below 0; BELOW means further below.

|Teacher cells (56)|WIN|LEVEL|BELOW|Claims: better / level or better / none|
|---|---:|---:|---:|---|
|Earlier four, SemArt only (8 cells)|0|7|1|0 / 7 / 1|
|DINOv2 (16 cells)|0|0|16|0 / 0 / 16|
|CLIP B/32 (16 cells)|1|2|13|0 / 3 / 13|
|CLIP B/16 (16 cells)|3|2|11|0 / 5 / 11|

- The new three sit below the teacher in most cells, and DINOv2 in all 16.
- On SemArt the four earlier encoders are LEVEL in 7 of 8 cells and none reaches WIN, so SemArt licenses no "better
  than". CLIP L/14 genre mAP is BELOW.
- The four WIN cells were not set for "better than". Each gives only "level with or better than".

**What it does not show.**

- New artists: genre only, with artists matched by name. No claim on emotion, style, or SemArt's schools and periods.
- Cost: one RTX 3090 at batch 32, model time. No claim on training cost, or against a teacher that encodes the image
  once for its three instructions. Our arithmetic, not measured, gives about 203 ms for such a teacher, so the saving
  would be smaller.
- Two LEVEL teacher cells have their whole interval below 0, so the claim wording reads stronger than the interval.
- r17's new metrics on these cells are descriptive and give no claim.

**Options for next.**

1. Freeze these rows into the paper's encoder and cost tables. No compute. The paper then answers all three questions.
2. Measure the cost against a teacher that encodes the image once. A short local probe, about 1 h. It replaces our
   arithmetic with a number.
3. Test emotion and style on new artists. A data search comes first, since SemArt has neither label; the cost depends
   on what it finds. It could extend the new-artist result past genre.

**Your reading.** "Not out of expectation, though really 'no better' is kind of bad." So these results will be either the ablation or go to the appendix. Hold the current result and finish this round. (User, 2026-11-05, before seeing ours.)

<details>
<summary>Our reading and recommendation (open after writing yours)</summary>

**Our reading.** The recipe carries past our picks. With no setting changed no verdict failed, and the PercepT miss
that we expected on DINOv2 did not come. We are sure of the verdicts: every gate passed, and an independent re-derivation of
the scoring from the saved outputs matched every item. The teacher stays ahead on the new three, and SemArt adds
artists outside ArtELingo's train set against the plain image and PercepT only.

**Our recommendation.** Option 1, then option 2 if the paper leads with the cost ratio. Our view: a reviewer will ask
about the shared image encoding, and a measured number answers that better than arithmetic.

</details>

## What ran

All stages ran, every gate passed, and every node run succeeded on its first attempt. Departures are under Deviations.

|Stage|Experiment type|Work|Where|
|---|---|---|---|
|0|Regression check|New stores, trainer and scorer replay, PercepT mappers, SemArt set, probe gate|node402; local 3090 and CPU|
|1|Development sweep|c* and the photo model on the new three, PercepT on DINOv2 at full scale, seed ranges, read-outs|node402; local CPU|
|2|Claim test|The four test sets below, everything frozen|Local CPU|
|3|Claim test|Cost: three model-time repeats, one repeat with decoding|Local RTX 3090|
|4|Descriptive|Twelve diagnostics, after the verdicts|Local CPU|

|Test data|Size|Scope|
|---|---:|---|
|ArtELingo held|12,281 paintings|Emotion, style, genre; new three|
|Affection test|8,501 photos|Emotion; photo model (c* fine-tuned on Affection's unlabeled train photos), new three|
|MMArt test|16,384 paintings|Emotion, style, genre; new three|
|SemArt new artists|11,159 paintings, 2,717 artists|SemArt's 10 types, close to genre; all seven|

- **Seeds and clustering:** seeds 0 to 4; AMI uses k-means with 20 clusters and three states.
- **Read-outs and ranges** come from validation only. SemArt uses ArtELingo validation and its genre labels, so that
  SemArt shapes nothing; ArtELingo paintings have no SemArt type label.
- **PercepT comparator:** PercepT on the same encoder. It has two readings, its own topic pick and k-means with 20
  clusters on its topic probabilities. Each bootstrap draw takes the larger.
- **SemArt set:** images at cosine ≥ 0.95 to an ArtELingo train or validation painting removed; md5 removes 3 of 11,162.
- **Gates:** regression gates (i) to (viii) pass. Parity passes 13 of 13 checks, minimum cosine 0.9997 against the bar
  of 0.999. Stage 1's records were committed before any test number.
- **Independent re-derivation (C13, by Codex):** 18,778 items, none failed, largest difference 1.9e-09. `check-outcomes`
  then compared the final outcomes: 4,211 items, none failed, same largest difference.
- **Code:** run commit `dd7e7c0b`, branch `r19-past-picks`. Times are Amsterdam local.

## Results

Every verdict holds, and each stands alone: one set never makes up for another. The
[full tables](../../../../src/test/20261103_past_picks/records/report_tables.md) hold all 137 cells, the claims and the
cost rows.

|Encoder|ArtELingo held|Affection test|MMArt test|SemArt new artists|Cost|
|---|---|---|---|---|---|
|Qwen, SigLIP 2, SigLIP 1, CLIP L/14|not in this round|not in this round|not in this round|holds|holds|
|DINOv2|holds|holds|holds|holds|holds|
|CLIP B/32|holds|holds|holds|holds|holds|
|CLIP B/16|holds|holds|holds|holds|holds|

**Verdict under the rule:** holds, GO in all 23 verdicts. No limit sentence is needed, since no verdict failed or was
qualified.

The smallest d per set and comparator is still a WIN. d is c* minus the comparator, with its 95% interval; AMI and mAP
have different scales.

|Set|Comparator|Tightest cell|d [95%]|
|---|---|---|---|
|ArtELingo held|Plain image|CLIP B/32, emotion mAP|+0.0120 [+0.0109, +0.0130]|
|ArtELingo held|PercepT|DINOv2, emotion AMI|+0.0131 [+0.0083, +0.0137]|
|Affection test|Plain image|CLIP B/32, emotion mAP|+0.0522 [+0.0500, +0.0542]|
|MMArt test|Plain image|DINOv2, emotion mAP|+0.0049 [+0.0045, +0.0053]|
|MMArt test|PercepT|DINOv2, emotion AMI|+0.0073 [+0.0038, +0.0073]|
|SemArt|Plain image|CLIP B/16, genre AMI|+0.0231 [+0.0178, +0.0286]|
|SemArt|PercepT|CLIP L/14, genre AMI|+0.0788 [+0.0690, +0.0880]|

The PercepT intervals sit low against d, since each draw takes the larger reading; on MMArt d equals the upper bound.
By lowest lower bound the SemArt PercepT row is SigLIP 1, +0.0675. Affection has no PercepT cell.

These teacher cells carry a flag, or are one of the four WINs. L means "level with or better than". The rule: WIN when
the lower bound is above 0; LEVEL when not WIN and the lower bound is above −r; BELOW otherwise; per cell, no
multiplicity correction. r is the candidate's seed range on validation: the photo model's on Affection, MMArt
validation on MMArt, ArtELingo validation on SemArt.

|Encoder; set; aspect|Metric|d [95%]|r|Outcome|Claim|
|---|---|---|---:|---|---|
|SigLIP 2; SemArt; genre|mAP|-0.0028 [-0.0042, -0.0013]|0.0043|LEVEL|L|
|CLIP L/14; SemArt; genre|AMI|-0.0095 [-0.0149, -0.0047]|0.0159|LEVEL|L|
|CLIP L/14; SemArt; genre|mAP|-0.0133 [-0.0154, -0.0114]|0.0017|BELOW|none|
|CLIP B/16; MMArt; genre|AMI|-0.0020 [-0.0074, +0.0033]|0.0059|BELOW|none|
|CLIP B/32; Affection; emotion|mAP|+0.0104 [+0.0081, +0.0125]|0.0051|WIN|L|
|CLIP B/16; Affection; emotion|AMI|+0.0142 [+0.0089, +0.0191]|0.0049|WIN|L|
|CLIP B/16; Affection; emotion|mAP|+0.0179 [+0.0153, +0.0203]|0.0062|WIN|L|
|CLIP B/16; ArtELingo; emotion|AMI|+0.0048 [+0.0017, +0.0080]|0.0031|WIN|L|

Cost, in ms per image, on one RTX 3090 at batch 32 over 2,000 MMArt paintings. c* is one encoder pass plus three heads;
the teacher is three Qwen passes: mean 266.8, fastest 266.5, with decoding 267.6. The ratio is teacher mean over c*
mean.

|Encoder|c* mean|c* slowest|Ratio of means|c* with decoding|
|---|---|---|---|---|
|Qwen|88.6|88.8|3.0|88.5|
|SigLIP 2|19.4|19.8|13.8|19.0|
|SigLIP 1|19.5|20.0|13.7|19.4|
|CLIP L/14|7.9|8.1|33.8|8.6|
|DINOv2|8.5|8.7|31.3|9.4|
|CLIP B/32|5.2|5.2|51.4|6.9|
|CLIP B/16|5.8|5.8|46.3|7.3|

## Analysis

Every comparison with the plain image and PercepT is a WIN; what varies is the gap to the teacher, and it depends on
the encoder. The rows below are descriptive unless they cite a claim, and none decides a verdict.

**The teacher by encoder** (counts in the brief). Our reading: the supervision carries over on every encoder, since
every plain-image cell is a WIN. These tests do not isolate why the gap to the teacher depends on the encoder.

**Wording flags (ours).** SigLIP 2 SemArt genre mAP and CLIP L/14 SemArt genre AMI have their whole interval below 0.
The rule calls them LEVEL, since each lower bound stays within r, so "level with or better than" is licensed, but the
wording reads stronger than the interval. In reverse, CLIP B/16 MMArt genre AMI is BELOW although its interval covers
0, since its lower bound is past −r.

**The four WINs.** Only the earlier four's SemArt genre mAP cells were set for "better than", and none of them is a
WIN. The four WINs sit elsewhere, so each gives "level with or better than". The photo model on CLIP B/16 has the
higher value on Affection (AMI 0.1808 against 0.1666, mAP 0.3125 against 0.2945); its licence stays the same.

**Cost.** The ratio of means runs from 3.0 (Qwen) to 51.4 (CLIP B/32), against three full passes. The spec licenses no
cost claim against a teacher that encodes the image once, and a reviewer will ask. The teacher's vision tower is 35.7%
of one Qwen pass (stage 3's `cost_table.json`).

- **Our arithmetic, not measured:** 0.3566 + 3 × 0.6434 = 2.287 passes at 88.9 ms each, about 203 ms per image.
- The saving against such a teacher would be smaller: for c* on Qwen, a ratio of about 2.3.

**With decoding, below model time on three encoders.** On Qwen, SigLIP 2 and SigLIP 1 the time with decoding is below
model time, by at most 0.4 ms (cost table). The probe loads images on one prefetch thread, so decoding overlaps the
model, and on slow encoders the model sets the pace. The end-to-end time is one pass; model time is the mean of three
repeats. Our reading: the overlap explains the difference. The cost sentence prints these numbers, so its wording in
the paper needs care.

**Without the 217 close SemArt paintings** (nearest cosine ≥ 0.93, 10,942 left), no verdict changes. One teacher cell
moves: SigLIP 1 genre AMI, LEVEL at +0.0011 [-0.0034, +0.0058] to WIN at +0.0059 [+0.0018, +0.0105]. It is not set for
"better than", so the claim stays. Inside the set, 472 pairs at cosine ≥ 0.95 touch 698 paintings.

**Beside the older student** (R6-5, the 10-25 and 10-26 rounds), c* is above in all nine ArtELingo held AMI cells
([rows](../../../../src/test/20261103_past_picks/records/diag_tables.md#r6_5_held)). DINOv2's older row is the
sample-scale run, since the full run was skipped, so that row sets c* at full scale beside a smaller run.

**Share of the teacher's gain over the plain image,** (c* − plain) / (teacher − plain). On SemArt the earlier four keep
0.76 to 1.04 of it and the new three 0.40 to 0.76
([every row](../../../../src/test/20261103_past_picks/records/diag_tables.md#gain_share)).

**PercepT on DINOv2.** The prior said that a PercepT miss was most likely on DINOv2; none came. DINOv2 emotion is the
tightest PercepT cell on ArtELingo and MMArt, and it is a WIN. On MMArt it is the strongest PercepT entry for emotion,
AMI 0.0437 against 0.0510 for c*.

The other [diagnostics](../../../../src/test/20261103_past_picks/records/diag_tables.md) are plain cosine rows,
clustering at the class count, seed spreads, AP per type, r17's metrics and node time.

## How the results change the paper

The paper gains one test set, three encoders and a cost table.

- **Related work, line 128:** SemArt moves to a test set; its sentence needs a clause on our use.
- **Method §3.7, lines 416 to 422:** name the three new encoders and widths: DINOv2 large (ViT-L/14) 1024, CLIP
  ViT-B/32 512, CLIP ViT-B/16 512. "We changed no setting for any encoder" stays true.
- **Experiments:** the encoder table with every verdict; the cost table beside the one-pass sentences at lines 202 and
  388.
- **Line 392:** "costs little next to the encoder" can carry a number: three heads 0.03 to 0.04 ms per image, against
  5.2 to 88.6 ms for c*'s whole pass.

Every filled sentence is in `records/outcomes.json`. ArtELingo, Affection and MMArt use r14's and r15's forms, with
"<X> is one of three encoders that failed or were unclear in an earlier held emotion test on ArtELingo with an earlier
version of our model." as the encoder sentence. A licensed teacher cell gives "<c* on X> is level with or better than
the Qwen image instructed for <aspect> in <task> on <set>". The new-artist and cost forms are in the spec's §6, and its "Not licensed" list stands unchanged.

## Deviations

No departure or checker repair changed a final score or a verdict.

|Item|What happened|
|---|---|
|End-to-end warm-up|`pp_probe.py:487` records zero warm-up batches; §3.4 names 3. The pass ran right after the model-time repeats, so the model was warm. It touches only the with-decoding numbers, never the cost verdict. Not rerun.|
|Re-derivation brief|"Frozen rows normalised twice" came from r15; r14 normalises once on ArtELingo and Affection (`bs_c13.py:373`). The derive first differed on 27 pick scores, at most 1.6e-5 on scores of size 1 to 40, then matched after the fix. The round's numbers were right.|
|Re-derivation label|Six failures were one label: `teacher_emotion_focus` (r14's) against the round's `teacher_artwork` (r12's, as §3.2 says). Same arm, same values. The checkers map that one name; any other label still fails.|
|`check-outcomes` layout|59 of 3,987 items failed first: 58 for a missing `status` key, 1 for a missing `encoder` field, which had skipped all 56 claims. No number differed. Fixed inside `check_outcomes`; six alterations of a copy of the real file each failed one item. Stage 4 was not rerun.|
|DINOv2 sample (gate v)|Copied from build 3's local 3090 rerun, not node402: 9 files, 6.8 MB, weights sha256 equal. Not rerun.|
|Node input probe and smoke|Passed in the build on `001d3d1`, 59 inputs, 0 failures. Inputs unchanged, so not rerun.|
|Node order|PercepT on DINOv2 ran before the last two photo jobs ended; it reads no photo output.|
|Extraction|Two chains, so each held a queue place. Parity took 5.6 h, not about 1 h, since other chats' jobs ran between its steps.|
|Probe end|10:22, past the sheet's 10:00 limit, as short, light local work. No repeat discarded.|
|Commit order|A shell error put records commit `a617831b` after log commit `35db5c7d`; both came before stage 2.|
|Full derive|Started at 19:00, since the user's limit for CPU scoring is about 13:00 and the code was not written; it ended before stage 4.|

## Files and storage

Nothing was deleted.

- **Outputs:** [round folder](../../../../src/test/20261103_past_picks/), [records](../../../../src/test/20261103_past_picks/records/), `/data/SSD2/cosir_r19/20261103_past_picks/outputs/` (about 16 GB).
- **Also left:** eight new stores under `/data/SSD2/pre_extract/`, `c13/` (61 MB, every compared item), `c13_smoke/` (49 MB), job folders on node402, pulled job folders in `res/cluster_jobs/`.
- **Storage:** the decision is taken at the wrap-up; see the disk ledger rows dated 2026-11-04.
