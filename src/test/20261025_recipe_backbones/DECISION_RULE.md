# Decision rule: the recipe pick on fresh development paintings (part 1), the backbone claim tests on held paintings (part 2) and the ArtELingo-28 reading

> Spec: [2026-10-25-recipe-and-backbones-design.md](../../../docs/superpowers/specs/2026-10-25-recipe-and-backbones-design.md)
> @ (first commit) · Written 2026-10-25 03:16 (Amsterdam) before any implementation script exists; revised 03:30
> after the review · Reviewer: a fresh Opus `xhigh` worker, 2026-10-25 03:27 (pass after fixes; all fourteen findings
> applied)

The trimmed constitution sets no statistics, development bar or failure reading. Part 1 states its own (§3.1). Part 2
and the ArtELingo-28 reading inherit from the claim test's rule, `src/test/20261020_claim_test/DECISION_RULE.md`
(below "the CT rule"), its §3 and, from its §4, V1 to V4, R-COL and the outcome thresholds, with only the changes
written here. The CT rule's 10-29 gate, its "what each outcome leads to" and its "comparator to improve against" do
not apply; §4.2 replaces them. C7, C8, C12 and C13 apply as written. Where this rule and the spec differ, this rule
governs. Paths below are relative to `src/test/20261020_claim_test/` ("CT/") unless they start with `/` or `src/`.

## 1. Prior (C8)

Written before any AMI on the 29,232 paintings, any student or matched-arm number on the new encoders, and any model
score on ArtELingo-28. Development numbers for CLIP's frozen image (12-15) and the CT's stage 2 stopping losses on the
6,152 validation paintings already exist.

**Part 1.** Part 1 reuses 12-22's seed-0 models. On the old 5,000 (which chose D2), their seed-0 effects in seed
ranges were: R6 +0.56 / +2.40 / +0.62 (g 1.19; emotion is not a gain), R7 +1.43 / +2.72 / −0.86 (g 1.10). Both pass
there, and R6-3 would win. R0 seed 0 was 12-22's best genre seed (0.4599), so genre effects on P may start lower.
The gains should shrink on paintings that did not choose D2; style's (over two ranges) is the most likely to survive.
The teacher share alone (R0-3) is untested; the no-teacher row cost style 0.0106 in 12-22.

| Outcome | Prior |
|---|---|
| P-KEEP (D2 stays) | 0.25 |
| P-R0-3 | 0.10 |
| P-R6-5 or P-R6-3 | 0.40 |
| P-R7-5 or P-R7-3 (DINOv2 runs) | 0.25 |

**Part 2**, per candidate, stage 3 (sample scale). Point guesses for d: K2 +0.003 to +0.012; K3 −0.005 to +0.008
(CLIP's frozen image emotion 0.065 against Qwen's 0.070 on development); K4 +0.010 to +0.020 (K1's +0.0146 plus part
of 12-22's emotion gain); K5 −0.015 to +0.005 (no language alignment in DINOv2's pretraining). Interval half-width:
about 0.003 to 0.006 (the CT's stage 1 interval was [+0.0093, +0.0157]).

| Outcome | K2 SigLIP 2 | K3 CLIP B/32 | K4 Qwen, new recipe | K5 DINOv2 |
|---|---|---|---|---|
| GO | 0.25 | 0.10 | 0.55 | 0.05 |
| Inconclusive, margin met | 0.05 | 0.05 | 0.05 | 0.05 |
| Inconclusive | 0.50 | 0.40 | 0.30 | 0.30 |
| NO-GO | 0.20 | 0.45 | 0.10 | 0.60 |

Stage 4 (full scale): as stage 3; the CT's d moved from +0.0146 to +0.0133 between the scales.

**ArtELingo-28.** A-HOLDS for K1 about 0.5 (746 paintings, labels from about 100 votes, other image files); for each
other candidate, half its stage 4 GO prior plus 0.1.

Disclosure: spec §8.

## 2. Candidates and comparators

**Part 1.** Loss = buddy terms (mean of four) + own-caption terms (mean of two) + w × teacher term; the same loss on
the 1,489 stopping paintings sets each aspect's best epoch. Every recipe trains on the 13,400 fit paintings of the
frozen sample with its graphs, model seed 0. Its exports are the image head outputs from each aspect's best state
(trunk and head), as 12-22.

| ID | What | Role |
|---|---|---|
| R0-5 | D2, w = 0.5 (teacher share 1/5); 12-22's R0 | reference; the default |
| R0-3 | D2, w = 1.0 (1/3) | candidate |
| R6-5 | no own-caption terms, w = 0.25 (1/5) | candidate |
| R6-3 | no own-caption terms, w = 0.5 (1/3); 12-22's R6 | candidate |
| R7-5 | image endpoints only (image–image buddy term and teacher), w = 0.25 (1/5) | candidate |
| R7-3 | image endpoints only, w = 0.5 (1/3); 12-22's R7 | candidate |

**Part 2.** The picked recipe c* (§4.1). A student is c* with the image network's first layer taking the encoder's
width; its caption network (when c* has one) reads the default Qwen caption mean; graphs, teacher (instructed Qwen
images), settings and 2048-wide outputs are unchanged. Every encoder image (student input, matched-arm Stage 2 input,
Fk) is the float32 store row divided by its float32 norm. Stage 3 trains on the frozen sample as the CT's stage 1;
stage 4 on the 42,969 train paintings with the CT's 42,969-node graphs (`CT/outputs/stage2/full/d2/graphs`) and
early stopping on the 6,152 validation paintings, as the CT's stage 2. Its outputs: each aspect's head output
(emotion, style, genre) on the encoder's default held image and on its ArtELingo-28 image, each from that head's best
dev-loss state; float16, 2048 wide. Emotion decides; style and genre are reported.

| ID | What | Role |
|---|---|---|
| K2 | student on SigLIP 2 so400m (1152) | candidate |
| K3 | student on CLIP B/32 (512), new canonical store | candidate |
| K4 | student on the default Qwen image; exists only when the pick is not P-KEEP | candidate |
| K5 | student on DINOv2-large (1024); exists only under P-R7-5 or P-R7-3 | candidate |
| B1 to B5 | the CT rule's B1 to B5: saved held outputs of that scale | comparators |
| B6 | PercepT core P2, K = 40: saved held outputs of that scale | comparator |
| M2, M3, M5 | matched arm of K2, K3, K5: PercepT core P3, K = 40, with the CT's saved Qwen Stage 1 artifacts of seed s at that scale (`CT/outputs/percept/core/stage1/qwen_qwen__s{s}`, `CT/outputs/stage2/full/percept/core/stage1/qwen_qwen__s{s}`); Stage 2 trained on the candidate's encoder image with P3's Stage 2 settings; PercepT seed s in Stage 2 | comparators |
| F1, F2, F3, F5 | the frozen default image of Qwen (F1, ArtELingo-28 only) and of each candidate's encoder | reported only |
| T | the instructed Qwen image of each aspect (the CT's held stores; on ArtELingo-28, the new 746-painting extraction) | reported only |

K4's matched arm is B1. The **entries** of candidate Kk are the two readings (r20, own) of B1 to B6 and of its
matched arm: up to 14. **Eligibility:** B2 to B5 per seed from the CT's R-COL record of that scale
(`CT/outputs/percept/rcol_stage1.json`, `CT/outputs/stage2/full/percept/rcol.json`; at the time of writing B2 and B3
are not eligible at either scale, B4 and B5 are eligible on seeds 0 to 4), frozen. B1, B6 and the matched arms have
no collapse rule and are always eligible.

**Stage 5 (ArtELingo-28).** Models: K1 (the CT's saved weights) and each K2 to K5 that exists, at full scale when its
stage 4 ran, else at the sample scale; their entries there: B1, B6 and their matched arm at the same scale (P1 and P4
need patch features, absent on ArtELingo-28). Images: each model reads ArtELingo-28's own store of its encoder.

## 3. The bar and the statistics

### 3.1 Part 1

**Data.** P = the 29,232 train-side paintings outside the 19,889-painting frozen sample (23,080 train, 6,152
validation), in Python `sorted()` order of their names. Its SHA-256 is the hash of `'\n'.join(names).encode('utf-8')`,
saved before any export on P and recorded in the pick file.

**Labels**, as `src/test/20261021_d2_ablation/ab_score.py`: emotion = the plurality of the painting's caption emotion
codes, "something else" included, ties to the smallest code (codes sorted by label name); style and genre from the
first caption. Labels < 0 are masked (genre: no WikiArt label). L_a = the positions in P with a label ≥ 0 for
aspect a.

**Partitions, fixed once.** Per recipe and aspect: MiniBatchKMeans, k = 20, random states 0, 1, 2, n_init 3, batch
size 4,096, on all of P, on the float32 cast of the saved float16 export (unit norm checked, |norm − 1| < 1e-3, not
re-normalized), under `threadpoolctl` at 8 threads; the scikit-learn and NumPy versions are recorded.

**Scores.** For an index vector I over L_a (with repeats): A_a(c, I) = the mean over the three partitions of
`sklearn.metrics.adjusted_mutual_info_score(labels_a[I], π[I])`, default arguments. e_a(c, I) = A_a(c, I) −
A_a(R0-5, I).

**Point estimate.** e_a(c) = e_a(c, L_a).

**Interval** (12-22's draws). Per aspect, its own `rng = numpy.random.default_rng(0)`; for b = 1 to 2,000: I_b =
L_a[`rng.integers(0, len(L_a), size=len(L_a))`]. The same I_b serves every recipe and partition of that aspect;
model seeds are not resampled (one seed). The 95% interval is `numpy.percentile(e_b, [2.5, 97.5])` (default linear
method).

**Seed ranges:** r_emotion = 0.0046, r_style = 0.0082, r_genre = 0.0128 exactly (the user's values, not 12-22's
unrounded ranges). Decisions use unrounded e, g and bounds, except the pick's `round(g, 2)`; reports give 4 decimals.

**Conditions** for a candidate c:
- **Gain on aspect a** ("at least small"): the interval's lower bound > 0 and e_a(c) ≥ r_a.
- **Guard:** e_a(c) ≥ −r_a on all three aspects (point estimates).
- **Pass:** a gain on at least one aspect, and the guard. A passing candidate with g(c) ≤ 0 still passes (the
  recorded rule has no condition on g); the report flags such a pick.
- **Mean gain:** g(c) = (e_emotion / r_emotion + e_style / r_style + e_genre / r_genre) / 3, losses included.

### 3.2 Part 2

The CT rule §3 applies (H, its hash, which must equal the CT's
`a0b1139c6056fd858ddbd2ee1dd4bafd1453e590818c6aa59883364c0288a876`; the emotion labels, as `CT/ct_held.py`'s
`painting_labels` in annotation order; partitions r20 and own; AMI; the scores A; the point estimate; the bar), with
these changes:
- **Per candidate.** Kk takes K1's place; each candidate has its own d, entries and bootstrap.
- **Bootstrap.** Each candidate has its own `rng = numpy.random.default_rng(0)`. Each draw b, in this order: I_b =
  `rng.integers(0, 12281, size=12281)`; then, for each arm with eligible seeds in the order Kk, B1, B2, B3, B4, B5,
  B6, then its matched arm (none for K4), S_b = `rng.choice(seeds, size=len(seeds), replace=True)` from its eligible
  seeds in ascending order. Arms without seeds (Fk, T) draw nothing. 2,000 draws; the maximum over entries is taken
  again in each draw; the interval as the CT rule.
- **Kk vectors** as K1's in the CT rule (float32 cast, unit norm checked, not re-normalized). **Matched arm and B6
  vectors** as the CT's PercepT vectors (K sigmoid probabilities, divided by their row norm).
- **The bar:** d ≥ 0.01 and the interval's lower bound > 0, as the CT rule.
- **The scorer checks** of stages 2 and 3a run on the CT's saved outputs with the CT's arm list (K1, B1 to B5; no B6,
  no matched arm) and must give `CT/verdict_stage1.json` and `CT/verdict_stage2.json` values (each A_e, d, both
  bounds) within 1e-6.

**Reported, deciding nothing.** Style and genre: the same statistic per aspect, labels as the CT (first caption;
labels < 0 masked), against (a) the maximum over the eligible readings of B1 to B6, (b) the matched arm's readings,
(c) the frozen encoder image Fk (its three r20 partitions), and the maximum of (a) and (b). Fk and T are also reported
on emotion. None enters a maximum. Each reported margin runs its own `default_rng(0)` bootstrap over the arms it
compares, in the order above. K1's d and interval are recomputed with K4's entry set (B1 to B6) from saved outputs and
reported beside K4's.

### 3.3 Stage 5 (ArtELingo-28)

The statistics of §3.2 on the 746 read-set paintings in sorted name order (SHA-256 saved as for H), with the
bootstrap drawing `rng.integers(0, 746, size=746)`.

**Labels** from the 74,569 read-set rows (`/data/SSD2/pre_extract/artelingo28/read_set_a28.parquet`), joined by
`row_id` (the public CSV's data-row position) to the public CSV's `emotion`, `art_style` and `genre` strings
(`/data/PDD/artelingo/ArtELingo/semeval-artelingo28/artelingo28_public_noworkerid_split.csv`), not the parquet's
codes. Emotion: "other" becomes "something else"; plurality over our nine names; codes as ours (names sorted,
"something else" = 8); ties to the smallest code. Style and genre: our codes by name, constant per painting (asserted).
No label is masked.

d_A28 = A_model − the maximum over its entries' readings (§2, stage 5). F1, Fk and T are reported.

## 4. Rules

### 4.1 Part 1

Validity first; no pick is stated until V0 and VP1 to VP3 hold.
- **V0** (C12) Stage 0 passed: through part 1's code, R0-5, R6-3 and R7-3 at seed 0 export, on the old 5,000
  evaluation paintings, vectors bit for bit equal to 12-22's R0, R6 and R7 seed 0 exports
  (`src/test/20261021_d2_ablation/outputs/train/{R0,R6,R7}/*__s0.npz`). Otherwise no export on P is computed, part 1
  stops and part 2 does not start; the cause goes in the report.
- **VP1** (C13) Separate code, given this rule, P's names, the captions and the saved exports (never our numbers),
  recomputes the labels, the partitions, each e_a(c), each interval and the pick. It must agree: equal partitions,
  each e and both bounds within 1e-6, the same pick. A disagreement is traced and fixed before the pick is stated,
  settled by this rule's text; both values and the fix go in the report.
- **VP2** Every export on P has 29,232 finite rows in P's order with unit norm; P's SHA-256 matches the saved one.
- **VP3** (seed policy) After any export on P exists, no recipe is added, dropped or retrained, except a rerun forced
  by a failed V check (same seed and code), which the report lists. A candidate still failing VP2 does not pass; if
  R0-5 fails, part 1 stops as under V0.

Outcomes:
- **P-KEEP:** no candidate passes. Part 2 uses R0-5.
- **P-c:** at least one candidate passes. c = the passing candidate with the largest `round(g, 2)` (Python's
  `round`); on equal values, the first in the order R0-3, R6-5, R6-3, R7-5, R7-3. Part 2 uses c.
- **Triggers:** K4 exists unless P-KEEP. K5 (and M5) exist only under P-R7-5 or P-R7-3.
- The old 5,000, the caption side, Leiden and the epoch-4 export are reported and decide nothing.

### 4.2 Part 2

Validity first; no outcome is stated for a candidate until V1 to V4 hold for it.
- **V1** Stage 3 needs a passed stage 2, stage 4 a passed stage 3a (spec §3.5), each bit for bit (C12). Otherwise no
  held output of a new arm is computed or read; the cause goes in the report.
- **V2** (C13) As the CT rule's V2, per candidate: separate code, given this rule, the CT rule, the held list, the
  captions, the CT's saved artifacts and records, and the saved outputs (never our scores), recomputes the labels,
  the partitions, the eligibility, each entry's A_e(I₀), d and the interval. Equal partitions and eligibility; d, each
  A_e and both bounds within 1e-6.
- **V3** As the CT rule's V3, for every Kk output set (all three aspects) and every matched-arm output set: 12,281
  finite rows in the saved order, Kk rows of unit norm; the held list's SHA-256 equals the CT's.
- **V4** The CT rule's seed policy, for every new arm. A Kk or matched-arm seed that still fails V3 after the forced
  rerun leaves the candidate with no outcome (reported, with the cause). A triggered candidate may be deferred for
  time (spec D32), never dropped.

Outcomes per candidate (the CT rule's names, with the candidate's ID):
- **S1-GO** d ≥ 0.01 and lower bound > 0. **S1-MM** (inconclusive, margin met) d ≥ 0.01 and lower bound ≤ 0.
  **S1-IN** (inconclusive) 0 < d < 0.01. **S1-NO** (NO-GO) d ≤ 0. All at the sample scale (stage 3).
- **S2-GO**, **S2-MM**, **S2-IN**, **S2-NO**: the same at full scale (stage 4), every arm at full scale.

What each outcome leads to:
- S1-NO: the candidate's stage 4 (and its matched arm's) does not run; its stage 3 result is final (user). Any other
  S1 outcome: its stage 4 runs (user).
- S2-GO, or S1-GO with no stage 4: the licensed sentence for that encoder (spec §6; "and default captions" is
  dropped under P-R7-5 or P-R7-3).
- Every outcome is reported, pass or not, and goes to the next decide chat's decision C; nothing follows from it
  automatically.
- The paper's recipe is part 1's pick (development data). The held outcomes of K1 and K4 test it and do not choose
  between them. If a later chat changes the recipe after reading them, the report says the recipe was chosen after
  reading held data.
- Each candidate's verdict is its own sentence; there is no correction for several candidates (agent default).
- The diagnostics (spec §3.6) and the reported margins are computed only after the outcome is written to the verdict
  file.

### 4.3 Stage 5 (ArtELingo-28)

Validity: finite rows in the 746's order and unit norm for students; the list's SHA-256 matches the saved one; V2's
re-derivation of d_A28 and its interval.
- **A-HOLDS** d_A28 > 0 and lower bound > 0. **A-QUAL** (qualified) d_A28 > 0 and lower bound ≤ 0. **A-FAILS**
  d_A28 ≤ 0.
- Per model; a confirmation: it never rescues a failed part 2 test. The 248 and 498 subsets, style, genre, F1, Fk, T
  and the culture split are reported and decide nothing.

## 5. Frozen and rerun

- **Frozen:** the frozen sample and its graphs (manifest hash); 12-22's seed ranges (as §3.1) and its seed-0 exports
  of R0, R6 and R7; the CT's 42,969-node graphs, Stage 1 artifacts, R-COL records, held outputs of K1 and B1 to B6 at
  both scales, K1 weights, held list and hash; the CT rule's statistics; this rule.
- **Rerun:** part 1's six recipes; K2 to K5 and their matched arms at both scales, seeds 0 to 4; held and ArtELingo-28
  inference of every new arm; all scoring and bootstraps.

## 6. Exceptions to the constitution

| Principle | Exception | Why | Simpler option rejected |
|---|---|---|---|
| none | — | — | — |

## 7. ID glossary

| ID | Meaning |
|---|---|
| R0-5, R0-3 | D2 with the teacher share at 1/5 (D2 itself) and 1/3 |
| R6-5, R6-3 | no own-caption terms, teacher share 1/5 and 1/3 |
| R7-5, R7-3 | image endpoints only, teacher share 1/5 and 1/3 |
| P, L_a | the 29,232 fresh development paintings; those with a label for aspect a |
| e_a, r_a, g | a recipe's effect on aspect a against R0-5; the fixed seed range of a; the mean gain in seed ranges |
| P-KEEP, P-c | part 1's outcomes: D2 stays; recipe c replaces it |
| V0, VP1 to VP3 | part 1's validity checks: regression, re-derivation, output checks, seed policy |
| K1 | D2 on Qwen, the CT's candidate |
| K2, K3, K4, K5 | students on SigLIP 2, CLIP B/32, Qwen with the new recipe, DINOv2-large |
| B1 to B5 | the CT's PercepT entries: core P3; paper strict P1, P4; paper K = 40 P1, P4 |
| B6 | PercepT core P2 |
| M2, M3, M5 | matched arms: PercepT core P3 with Qwen's Stage 1 topics and the candidate's encoder image in Stage 2 |
| F1; F2, F3, F5; T | the frozen default Qwen image (ArtELingo-28 only); the frozen encoder images; the instructed Qwen image (all reported only) |
| S1-*, S2-* | outcomes at the sample and full scale: GO, MM (inconclusive, margin met), IN (inconclusive), NO (NO-GO) |
| V1 to V4 | part 2's validity checks, as the CT rule's |
| A-HOLDS, A-QUAL, A-FAILS | the ArtELingo-28 reading per model |
