# Retrieval passes after validation-picked read-outs; emotion fails against the teacher with plain cosine

> 2026-10-29 (sequence name 12-29) · Spec: [design](../../../superpowers/specs/2026-10-29-retrieval-design.md) · Decision rule: inline in the spec (§6 with §3.4 and §3.5) @ b00e0d0, amendments A1 e8ecc4e and A2 1a00ff8 · Run log: [round log](../../../../src/test/20261029_retrieval/20261029_retrieval_log.md) · Previous round: [12-28](2026-12-28_recipe_sweep.md)

## Results brief

**What we tried.** We tested whether frozen c* retrieves paintings by a named aspect ([spec](../../../superpowers/specs/2026-10-29-retrieval-design.md)).
A read-out compares saved embeddings at retrieval time, using centering, PCA whitening or average query expansion; each model picks one on validation.

**Outcome against the bar.** Stage 1, plain cosine, is NO-GO; stage 5, with picked read-outs, is GO.
The table shows c* minus the instructed Qwen image in mAP, with 95% joint bootstrap intervals.

| Aspect | Stage 1: plain cosine | Stage 5: picked read-out |
|---|---|---|
| emotion | -0.0023 [-0.0032, -0.0014]; BELOW | +0.0007 [-0.0002, +0.0016]; LEVEL |
| style | +0.0019 [+0.0008, +0.0030]; WIN | +0.0017 [-0.0001, +0.0036]; LEVEL |
| genre | +0.0037 [+0.0024, +0.0051]; WIN | +0.0027 [+0.0013, +0.0042]; WIN |

- c* beats the plain Qwen image on every aspect at both stages.
- The switch test is GO; graded emotion confirmation is qualified.

**What it does not show.** These results cover ArtELingo and Qwen3-VL-Embedding-2B features only, with no text queries.
Caption majority labels are noisy; emotion mAP sits near chance, 0.2007.
LEVEL does not mean better: c* does not beat its teacher on emotion or style after read-out selection.
Validation also chose c*'s epochs within development data that selected c* among about 60 configurations; only held numbers are results.
Earlier rounds read this held split for clustering. We make no claim against CLIP, SigLIP 2, PercepT core or CRL.

**Options for next.**

1. Widen the read-out menu above k = 10 and add standard re-ranking, picked on validation, then score held again.
   A short spec and about one CPU hour could test an emotion win; the instructed image may gain as much.
2. Retrain c* for emotion retrieval: days of cluster GPU and new clustering claim tests, colliding with the 2026-11-10 recipe freeze.
3. Continue r12 as planned: frozen c* on Affection, Impressions and PolyArt, testing clustering and retrieval beyond ArtELingo.
   This costs three dataset evaluations; no time estimate is available.

Open decisions: claim the switch test (D7), and give CRL a bar now that its number exists (D6)?

**Your reading.**

<details>
<summary>Our reading and recommendation (open after writing yours)</summary>

**Our reading.** The planned claim holds, but c* matches its teacher overall: emotion and style are LEVEL, genre a narrow WIN.
The large margins are over the plain image: emotion +0.0208, style +0.1311, genre +0.0790 mAP.
A retrieval seed range (r) is c*'s per-aspect validation mAP spread, maximum minus minimum, over seeds 0 to 4 with plain cosine.
Emotion sits about 1.6 r below the teacher at plain cosine; its claim depends on validation-picked read-outs.
We have high confidence in the plain-image wins and switch test, and moderate confidence in matching the teacher.
Emotion and style lower bounds sit just below zero.

**Our recommendation.** Our view: take option 3 as planned; leave retrieval retraining for r13 after 2026-11-10.
Show stage 5 beside the untuned stage 1 row. Claim the switch test in one supporting sentence; the teacher shows it too.
Give our re-implementation of CRL a bar in r12's spec: one held bootstrap comparison at stage 5, costing minutes of CPU.

</details>

## What ran

The run scored frozen embeddings without touching training data. c* is r10's frozen recipe, model G with 40 graph neighbours ([12-28](2026-12-28_recipe_sweep.md)).
Held comparisons use 12,281 paintings, with genre labels on 9,949; c* and PercepT core use 5 seeds, averaged per query.
Validation uses 6,152 paintings for ranges, fitting without labels and selection with labels; training is untouched.

| Stage | Type | Role |
|---|---|---|
| 0a | measurement | Commit validation ranges before held retrieval |
| 0b | regression check | Clustering, joins, hashes, head inference |
| 1 | claim test | Held label retrieval with plain cosine |
| 2 | claim test | Held aspect-switch comparisons |
| 3 | confirmation | Held graded emotion nDCG |
| 4 | sweep | Pick one read-out per model on validation |
| 5 | claim test | Held label retrieval with frozen picks |

Plain Qwen and aspect-instructed Qwen, c*'s teacher, carry the bar.
CLIP L/14, SigLIP 2, PercepT core and our re-implementation of CRL (NeurIPS 2025) are descriptive arms.
Code: [round folder](../../../../src/test/20261029_retrieval/) at main 12e5119.
Second start: local CPU, 8 threads, 22:10 to 23:02; PercepT export under the GPU lock, about one minute.

Unlike GLOSSARY.md's seed range (r10 clustering AMI), these retrieval ranges were committed before held retrieval:

| Range | Emotion | Style | Genre |
|---|---|---|---|
| mAP r | 0.0014 | 0.0022 | 0.0038 |
| emotion nDCG r | 0.00031 | - | - |

Each labelled query ranks the other labelled paintings; matching labels define relevance.
Intervals: 95% joint bootstrap over queries and seeds, 10,000 draws, fixed gallery.
WIN means lower bound above zero; otherwise LEVEL means lower bound above minus r; otherwise BELOW.
Differences mean c* minus the comparator, before rounding.

## Results

Stage 5 passes the paper's bar; the untuned stage 1 does not.

| ID | Stage | Aspect | Comparator | c* mAP | Comparator mAP | Difference [95%] | r | Outcome | Bar | Pass |
|---|---|---|---|---|---|---|---|---|---|---|
| SC1 | 1 | emotion | plain | 0.2642 | 0.2424 | +0.0217 [+0.0209, +0.0226] | 0.0014 | WIN | WIN | yes |
| SC2 | 1 | style | plain | 0.3110 | 0.1801 | +0.1309 [+0.1285, +0.1332] | 0.0022 | WIN | WIN | yes |
| SC3 | 1 | genre | plain | 0.5896 | 0.4760 | +0.1136 [+0.1113, +0.1161] | 0.0038 | WIN | WIN | yes |
| SC4 | 1 | emotion | instructed | 0.2642 | 0.2665 | -0.0023 [-0.0032, -0.0014] | 0.0014 | BELOW | LEVEL or WIN | no |
| SC5 | 1 | style | instructed | 0.3110 | 0.3091 | +0.0019 [+0.0008, +0.0030] | 0.0022 | WIN | LEVEL or WIN | yes |
| SC6 | 1 | genre | instructed | 0.5896 | 0.5860 | +0.0037 [+0.0024, +0.0051] | 0.0038 | WIN | LEVEL or WIN | yes |
| SC7 | 2 | emotion | style head | 0.2642 | 0.2487 | +0.0154 [+0.0141, +0.0167] | 0.0014 | WIN | WIN | yes |
| SC7 | 2 | emotion | genre head | 0.2642 | 0.2539 | +0.0103 [+0.0094, +0.0112] | 0.0014 | WIN | WIN | yes |
| SC8 | 2 | style | emotion head | 0.3110 | 0.1529 | +0.1581 [+0.1553, +0.1609] | 0.0022 | WIN | WIN | yes |
| SC8 | 2 | style | genre head | 0.3110 | 0.2167 | +0.0943 [+0.0923, +0.0963] | 0.0022 | WIN | WIN | yes |
| SC9 | 2 | genre | emotion head | 0.5896 | 0.4520 | +0.1376 [+0.1335, +0.1420] | 0.0038 | WIN | WIN | yes |
| SC9 | 2 | genre | style head | 0.5896 | 0.3593 | +0.2304 [+0.2256, +0.2351] | 0.0038 | WIN | WIN | yes |
| SC10 | 5 | emotion | plain | 0.2691 | 0.2483 | +0.0208 [+0.0198, +0.0217] | 0.0014 | WIN | WIN | yes |
| SC11 | 5 | style | plain | 0.3235 | 0.1924 | +0.1311 [+0.1282, +0.1340] | 0.0022 | WIN | WIN | yes |
| SC12 | 5 | genre | plain | 0.6151 | 0.5361 | +0.0790 [+0.0764, +0.0817] | 0.0038 | WIN | WIN | yes |
| SC13 | 5 | emotion | instructed | 0.2691 | 0.2684 | +0.0007 [-0.0002, +0.0016] | 0.0014 | LEVEL | LEVEL or WIN | yes |
| SC14 | 5 | style | instructed | 0.3235 | 0.3217 | +0.0017 [-0.0001, +0.0036] | 0.0022 | LEVEL | LEVEL or WIN | yes |
| SC15 | 5 | genre | instructed | 0.6151 | 0.6124 | +0.0027 [+0.0013, +0.0042] | 0.0038 | WIN | LEVEL or WIN | yes |

**Verdict under the rule:** stage 1 NO-GO because emotion is BELOW the instructed image; stage 5 GO; switch GO; stage 3 qualified.

The spec licenses this stage 5 sentence:

> On 12,281 held ArtELingo paintings with Qwen3-VL-Embedding-2B features, c*, given a painting and an aspect name, retrieves paintings with the same emotion, style or genre label better than the plain Qwen image, and level with or better than the Qwen image instructed for that aspect (mAP, 5 seeds, each model with a read-out picked on validation paintings).

Stage 1 is NO-GO, so the plain-cosine form of this sentence is not licensed.
Also not licensed: other datasets or encoders, text queries, per-class margins, or claims against descriptive arms.

Graded emotion prefers c* over both images: qualified.
Its gain measures caption-emotion distribution overlap over the full ranking.

| Stage 3 | c* nDCG | Comparator nDCG | Difference [95%] | Outcome |
|---|---|---|---|---|
| plain | 0.9343 | 0.9292 | +0.0052 [+0.0050, +0.0053] | WIN |
| instructed | 0.9343 | 0.9341 | +0.0002 [+0.0001, +0.0004] | WIN |

Stage 4 picked centering plus query expansion except for our CRL; k10 means k = 10.
Scores average validation mAP gains over plain cosine in units of c*'s r; validation halves fit and score each other.
One pick serves all aspects, guarded at minus r; stage 5 fits on validation.

| Arm | Pick | Score gain (retrieval seed ranges) | Validation mAP: emotion / style / genre |
|---|---|---|---|
| c* | center + k10 | +5.43 | 0.2730 / 0.3253 / 0.6257 |
| plain | center + k10 | +9.25 | 0.2529 / 0.1954 / 0.5457 |
| instructed | center + k10 | +4.72 | 0.2731 / 0.3253 / 0.6206 |
| CLIP L/14 | center + k10 | +12.40 | 0.2533 / 0.2782 / 0.5109 |
| SigLIP 2 | center + k10 | +11.97 | 0.2478 / 0.2623 / 0.5109 |
| PercepT core | center + k10 | +2.55 | 0.2514 / 0.1537 / 0.5017 |
| our CRL | no center + k10 | +3.99 | 0.2600 / 0.2873 / 0.5089 |

c* exceeds all descriptive arms on every aspect at both stages: point values only, no computed intervals or licensed claims.

| Arm | Stage | Emotion mAP | Style mAP | Genre mAP |
|---|---|---|---|---|
| CLIP L/14 | 1 | 0.2424 | 0.2421 | 0.4272 |
| SigLIP 2 | 1 | 0.2364 | 0.2298 | 0.4327 |
| PercepT core | 1 | 0.2415 | 0.1486 | 0.4982 |
| our CRL | 1 | 0.2538 | 0.2722 | 0.4744 |
| CLIP L/14 | 5 | 0.2479 | 0.2732 | 0.4981 |
| SigLIP 2 | 5 | 0.2443 | 0.2576 | 0.4968 |
| PercepT core | 5 | 0.2483 | 0.1523 | 0.5049 |
| our CRL | 5 | 0.2554 | 0.2816 | 0.5033 |

Chance from squared class shares: emotion 0.2007, style 0.0792, genre 0.1466.
Independent re-derivation: every value agreed (tolerance 1e-6):

| Stage | 0a | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| Values checked | 24 | 68 | 47 | 19 | 790 | 47 |
| Disagreements | 0 | 0 | 0 | 0 | 0 | 0 |

Stage 0b passed 54 of 54 items after the mapper correction.

## Analysis

Our reading: teacher comparisons reverse the prior's aspect order.
The prior expected emotion safest and genre riskiest. Genre wins at both read-outs; emotion moves from BELOW to LEVEL.
Style moves from WIN to LEVEL; its lower bound sits just under zero.
The prior held for the wide wins over plain Qwen and for the switch test, genre column included.
It expected read-outs to help the plain and instructed images more than c*: true for plain, not for instructed (validation scores below).
Centering and query expansion carried the gains, not whitening as the prior guessed.

Contentment carries the plain-cosine emotion loss: -0.0324 AP, 4,244 of 12,281 queries.
The other eight classes gain between +0.0027 and +0.0340 AP.
mAP weighs queries equally, so this largest class weighs most. Per-class differences are diagnostic, not licensed claims.
Style's class differences span -0.0434 to +0.0812; genre's span -0.0071 to +0.0187, also diagnostic only.
Our reading: c* may follow the caption-emotion mix while losing on the exact majority label; this hypothesis remains unchecked.

Our reading: the fixed emotion margin was slightly strict, but did not alter any outcome.

| Aspect | Held per-seed mAP spread | Committed validation r |
|---|---|---|
| emotion | 0.00171 | 0.00141 |
| style | 0.00196 | 0.00223 |
| genre | 0.00356 | 0.00382 |

Substituting held spreads changes no label-retrieval or switch outcome, checked using unrounded bounds.
The stage 1 emotion gap is -1.62 r; its lower bound stays below minus the held spread. r remains fixed.

Our reading: the sweep has not exhausted read-out gains.
Every pick uses the largest k, 10; larger k may help, but must again be picked on validation.
No whitening read-out passed the guard for any arm (`stage4/scores.json`); for our CRL, centering failed it too.
Plain Qwen gains +9.25 retrieval seed ranges against c*'s +5.43 and the instructed image's +4.72; CLIP and SigLIP 2 gain most, +12.40 and +11.97.

Our reading: aspect switching is shared by all aspect-specific arms.
c*'s own head beats the other two by 0.0103 to 0.2304 mAP in each label column, with joint intervals above zero.
The teacher and our CRL share the diagonal pattern (point values; c* is in the claim table):

| Arm | Head | Emotion mAP | Style mAP | Genre mAP |
|---|---|---|---|---|
| instructed | emotion | 0.2665 | 0.1518 | 0.4595 |
| instructed | style | 0.2549 | 0.3091 | 0.4021 |
| instructed | genre | 0.2534 | 0.2126 | 0.5860 |
| our CRL | emotion | 0.2538 | 0.1497 | 0.3550 |
| our CRL | style | 0.2412 | 0.2722 | 0.2842 |
| our CRL | genre | 0.2423 | 0.1584 | 0.4744 |

c* provides this pattern in one model that reads the default image once.
Label names hint at the target for our CRL: emotion 4, style 3 and genre 4 of 50 values each.

Descriptive P@10 in `<outputs>/summary.md`: c* leads the teacher on emotion and genre, trails on style, at both stages; no verdict changes.

## Deviations

The mapper correction restored the frozen comparator before held retrieval; the rule stayed fixed.

- A1, build before results: validation PercepT outputs were re-inferred from the held mapper for consistent fit and score.
- A2, run at 22:10 after stage 0a: stage 0b failed one of 54 items. PercepT core held emotion AMI: 0.08549055120302276 against 0.08742922158584242. A1 used the sample-scale mapper; r10 used full scale. Branch r11-fix3 corrected it, merged 625eb64. The queue reran from stage 0a, reproducing committed ranges byte for byte. PercepT core now uses full scale. Failed evidence: `<outputs>/stage0b/check_failed_A1_sample_scale.json` and `logs/stage0b_failed_A1_sample_scale.log`.
- Build fixes changed no rule: C13 disagreed on 174 of 790 stage 4 smoke values: float32 rounding and truncated whitening null space at 1,024 dimensions. A written numeric convention fixed agreement; smoke size rose to 2,100 rows. A `.gitignore` exception permits the committed ranges file.
- Stage 4 dropped one style query in one validation half because that half contained no relevant painting, counted as specified.
- Selection disclosure: validation chose c*'s epochs within development data that selected c* among about 60 configurations. Stage 4 adds up to 25 read-outs per arm. Only held numbers are results; earlier rounds read held for clustering.

## Files and storage

Storage decisions wait for the next decide chat; nothing was deleted.

- Outputs: `/data/SSD2/cosir_r11/20261029_retrieval/outputs` (59 MB); `summary.md`, stage verdicts, picks, per-query arrays and diagnostics. Nothing from this round exceeds 1 GB.
- Pending only: `.scratch/pending_deletions.md` lists the old `percept_val/` exports (4.5 MB), plus the r11-fix3 worktree and branch. 
