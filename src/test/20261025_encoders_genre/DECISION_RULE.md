# Decision rule: four more encoders on held paintings (stage i), each student's lift over its encoder's own PercepT arm on development paintings (stage ii), and genre variants toward the instructed Qwen image (stage iii)

> Spec: [2026-10-25-encoders-and-genre-design.md](../../../docs/superpowers/specs/2026-10-25-encoders-and-genre-design.md)
> @ (first commit) · Written 2026-10-25 17:55 (Amsterdam) before any implementation script exists; revised 18:40
> after the review · Reviewer: a fresh Opus `xhigh` worker, 2026-10-25 18:30 (pass after fixes; all thirteen findings
> applied), and a cold Sonnet `high` check against the grill record (five must-fix points applied)

The trimmed constitution sets no statistics, development bar or failure reading, so this rule states or inherits them.
**Stage i** inherits 12-24's rule, `src/test/20261025_recipe_backbones/DECISION_RULE.md` (below "the RB rule"): its
part 2 and stage 5 text (§2 from "Part 2" on, §3.2, §3.3, §4.2, §4.3), which in turn inherits the claim test's rule
(`src/test/20261020_claim_test/DECISION_RULE.md`, "the CT rule"). The changes (§2.1, §3.1, §4.1): the encoder list,
the recipe R6-5 for every candidate, K5 without a trigger, the stage 5 model list, the order and deferral, the store
check V5, and the regression gates X0, X1, X2. **Stages ii and iii** take the RB rule's part 1 statistics (§3.1: P,
its hash, the labels, the partitions, the scores, the interval, the seed ranges) with the changes written here. C7,
C8, C12 and C13 apply as written. Where this rule and the spec differ, this rule governs. Paths below are relative to
`src/test/20261025_recipe_backbones/` ("RB/") unless they start with `/` or `src/`.

## 1. Prior (C8)

Written before any number of the four new encoders exists, before any stage iii variant is trained, and before any arm
is scored on P except those 12-24 scored there (part 1's six recipes, the default and instructed Qwen images, the
ceiling probe). Known: 12-24's held verdicts and tables for K1 to K4, M2, M3, F2 to F4; its P values.

**Stage i**, per candidate, stage 3 (sample scale). 12-24's stage 3 d: K2 +0.0165, K3 +0.0046, K4 +0.0190. Point
guesses for d: K6 −0.002 to +0.010 (CLIP B/32's result with a finer patch grid); K7 +0.002 to +0.014 (a larger CLIP);
K8 +0.008 to +0.018 (SigLIP 2's twin); K5 −0.012 to +0.006 (no language alignment in DINOv2's pretraining).

| Outcome | K6 CLIP B/16 | K7 CLIP L/14 | K8 SigLIP 1 | K5 DINOv2 |
|---|---|---|---|---|
| GO | 0.15 | 0.30 | 0.45 | 0.10 |
| Inconclusive, margin met | 0.05 | 0.05 | 0.05 | 0.05 |
| Inconclusive | 0.45 | 0.40 | 0.35 | 0.30 |
| NO-GO | 0.35 | 0.25 | 0.15 | 0.55 |

Stage 4 (full scale): as stage 3 (12-24's d moved from +0.0165 to +0.0150 for K2 and from +0.0046 to +0.0011 for K3).
ArtELingo-28: A-HOLDS for each candidate about half its stage 4 GO prior plus 0.1, as the RB rule.

**Stage ii.** On held paintings at the sample scale, 12-24's students sit above their own PercepT arm's best reading
on emotion by +0.0243 (K2), +0.0202 (K3) and +0.0190 (K4) (arithmetic from 12-24's tables). On P we expect the same
sign and a similar size. L-HOLDS: 0.9 for K2, K3 and K4; 0.8 for K6, K7 and K8; 0.7 for K5. The lift's relation to the
frozen image's AMI: no prior direction (on held paintings CLIP B/32's frozen emotion AMI, 0.0642, is above SigLIP 2's,
0.0569, yet its student lifts less). Disclosure: K4 at seed 0 is the model that won 12-24's recipe pick on P, so
K4's lift on P is inflated by that pick against the other six encoders', and so are the correlations.

**Stage iii.** On P, seed 0, R6-5's genre AMI is 0.4674 and the target T_g is 0.4989 (2.5 genre seed ranges above).
Point guesses for the genre effect: GT-2 +0.003 to +0.012; GT-4 +0.005 to +0.020 (emotion −0.002 to −0.006 through
the shared trunk); GR-0.5 −0.005 to +0.008; GR-2 −0.005 to +0.012; GA-1 +0.005 to +0.025 (the ceiling probe reached
0.5040 from the default image). III-GO 0.15, III-PROM 0.25, III-NONE 0.60. The fresh-seed check runs: 0.5; C-HOLDS
when it runs: 0.5.

Disclosure: spec §8.

## 2. Candidates, arms and comparators

### 2.1 Stage i

The RB rule §2 (part 2 and stage 5), with these changes:
- **Recipe:** c* = R6-5 (the RB rule's outcome P-R6-5, frozen) for every candidate.
- **Candidates** (each image is the encoder's projected image embedding from the checkpoint's own processor defaults;
  as every encoder image, the float32 row divided by its float32 norm):

| ID | Encoder (checkpoint, input size, width) | Matched arm | Frozen image |
|---|---|---|---|
| K6 | CLIP ViT-B/16 (`openai/clip-vit-base-patch16`, 224 px, 512) | M6 | F6 |
| K7 | CLIP ViT-L/14 (`openai/clip-vit-large-patch14`, 224 px, 768) | M7 | F7 |
| K8 | SigLIP 1 So400m/14 (`google/siglip-so400m-patch14-384`, 384 px, 1152) | M8 | F8 |
| K5 | DINOv2-large (`facebook/dinov2-large`, the RB spec's D27: pooled CLS after the final layer norm, 1024) | M5 | F5 |

- **K5 exists unconditionally**; the RB rule's K5 trigger (§4.1, P-R7-5 or P-R7-3) does not apply.
- K1 to K4 and their outcomes stand from 12-24 and are not rerun. M5 to M8 are built as the RB rule's M2 and M3.
- **Entries** of each candidate: the two readings of B1 to B6 and of its matched arm, with the RB rule's eligibility.
- **Stage 5:** K5 to K8 with B1, B6 and their matched arm, at full scale when the candidate's stage 4 ran to an
  outcome, else at the sample scale; each reads ArtELingo-28's own store of its encoder. F1, Fk and T are reported.
- **Order:** K6, K7, K8, K5. A candidate may be deferred for time only, never dropped; deferral takes K5 first, then
  K8, K7, K6. A deferral is decided from the time estimate before that candidate's stage 3 starts, and recorded with
  the estimate.

### 2.2 Stage ii

Seven encoders. Every arm was trained on the frozen sample (13,400 fit and 1,489 stopping paintings), outside P.

| Encoder | Student (R6-5, sample scale, seeds 0 to 4) | Own PercepT arm (seeds 0 to 4) | Frozen image |
|---|---|---|---|
| Qwen default | K4 (12-24's saved weights) | B1 (the CT's sample-scale core P3 weights) | F4 |
| SigLIP 2 | K2 (12-24) | M2 (12-24) | F2 |
| CLIP B/32 | K3 (12-24) | M3 (12-24) | F3 |
| CLIP B/16, CLIP L/14, SigLIP 1, DINOv2 | K6, K7, K8, K5 (this round's stage 3) | M6, M7, M8, M5 | F6, F7, F8, F5 |

F4 is the frozen default Qwen image (12-24's held tables name it F4; the RB rule's F1 on ArtELingo-28 is the same
image). Outputs on P: each student's three aspect heads, each from its aspect's saved best state; each own arm's
K = 40 sigmoid vectors divided by their row norm; each frozen image as its normalized row. Stage ii's P inference
starts when every candidate not deferred has its stage 3 weights; a candidate deferred then, or without a stage 3
outcome, is missing from stage ii (reported as such, never added later).

### 2.3 Stage iii

**Base:** R6-5 on the default Qwen image, trained on the frozen sample, model seed 0; its P exports from 12-24's part 1
(`RB/outputs/part1/train/R6-5`), frozen. **Variants:** R6-5 with one change to the genre head's loss; the emotion and
style losses are unchanged (the shared trunk sees every change).

| ID | Change to the genre loss (R6-5's genre teacher weight is 0.25) |
|---|---|
| GT-2, GT-4 | teacher term weight 0.5 and 1.0 (×2 and ×4) |
| GR-0.5, GR-2 | plus λ × the relational term, λ = 0.5 and 2: KL from the teacher's to the student's in-batch similarity distribution, temperature 0.05, same-painting pairs masked (`src/test/20261019_model_sheet/ms_train.py`, `distill_loss`, kind "relational"), toward the instructed genre image; the direct term stays |
| GA-1 | plus w × the regression term, w = 1, through an adaptor (below) |

**The GA adaptor:** linear 2048 to 2048, GELU, linear 2048 to 2048. Its input is the genre head's unit-norm output.
It predicts the instructed genre image standardized per dimension as `RB/rb_probe.py`'s `Standardizer` (float64 mean
and population standard deviation plus 1e-6) on the 13,400 fit paintings. The term is the ceiling probe's loss (mean
of 1 − cosine and smooth-L1 with β = 1; `RB/rb_probe.py`, `loss_fn`). The adaptor is built after the model from its
own `torch.Generator` seeded with the model seed, so the model's random stream is unchanged; it trains with the model.
Its output on P is exported for the diagnostics, and then it is discarded; the variant's export is the head output
below it.

**Extras:** two more runs of the closest method (§4.3), at rungs of its ladder (rungs run first marked *):

| Method | Ladder |
|---|---|
| GT (multiplier of 0.25) | 1.5, 2*, 3, 4*, 6, 8 |
| GR (λ) | 0.25, 0.5*, 1, 2*, 4, 8 |
| GA (w) | 0.25, 0.5, 1*, 2, 4 |

An extra is named by its method and rung (GT-6, GR-1, GA-0.5). **Every run** keeps R6-5's settings (12-24's part 1):
the same per-aspect early stopping on that aspect's loss (new terms included) over the 1,489, and exports of the image
head outputs from each aspect's best state on P and on the old 5,000 (float16, 2048 wide). **References**, without
training: the instructed Qwen genre image on P (T_g, below) and the ceiling probe's ridge prediction (0.504009),
reported only.

## 3. The statistics

### 3.1 Stage i

The RB rule §3.2 and §3.3, per candidate K5 to K8, unchanged, except two of §3.2's bullets: its scorer checks against
the CT's verdicts are replaced by X1 and X2 (§4.1), and K1's recompute with K4's entry set is not repeated.

### 3.2 Stage ii

**Data and labels:** P, its SHA-256 (must equal 12-24's,
`169f137e9088577a3c2f5ad352e430c5ef4d1ecb58e74dffacd4d0ea71181ef7`), the labels and L_a as the RB rule §3.1. Each
aspect's label hash, the SHA-256 of its int64 labels over all of P with −1 where masked (`RB/rb_pick.py`,
`array_sha256`), must equal the one in `RB/outputs/part1/pick.json`.

**Partitions, fixed once,** per arm, seed and aspect, with the RB rule §3.1's k-means settings (k = 20, random states
0 to 2, n_init 3, batch 4,096, 8 threads): students and frozen images, r20 (three partitions; frozen images have no
seed); own arms, r20 on their normalized vectors and "own" (the arg-max topic, one partition).

**Scores.** A_a(arm, I, S) = the mean over the seeds in S of the mean over that seed's partitions of
`adjusted_mutual_info_score(labels_a[I], π[I])`. For a seedless arm, S is ignored.

**Lift** of encoder e on aspect a: lift_a(e) = A_a(Kk, L_a, {0..4}) − max over the two readings of A_a(Ok, L_a,
{0..4}), where Kk and Ok are e's student and own arm.

**Interval,** per encoder and aspect, its own `rng = numpy.random.default_rng(0)`; for b = 1 to 2,000, in this order:
I_b = L_a[`rng.integers(0, len(L_a), size=len(L_a))`]; S_b for Kk, then S_b for Ok, each
`rng.choice([0, 1, 2, 3, 4], size=5, replace=True)` (paintings shared, seeds drawn per arm, as the CT rule); lift_b
with the maximum over Ok's readings taken again. The 95% interval is `numpy.percentile(lift_b, [2.5, 97.5])`. Emotion
decides; style and genre are reported.

**Reported, deciding nothing:** A_a(Fk); A_a(Kk) − A_a(Fk); the encoder statistics (spec §3.6); Spearman correlations
over the encoders between the emotion lift and each of A_emotion(Fk), the own arm's best emotion reading and each
statistic; the held table (spec §3.6).

### 3.3 Stage iii

**Data, labels, partitions, scores, point estimate and interval:** the RB rule §3.1 with R6-5 in R0-5's place:
e_a(v, I) = A_a(v, I) − A_a(R6-5, I), e_a(v) = e_a(v, L_a), the interval from the RB rule's per-aspect draws (the same
I_b for every variant of that aspect). Seed ranges r = 0.0046 / 0.0082 / 0.0128. A_genre(v) = A_genre(v, L_genre),
unrounded.

**Target:** T_g = the unrounded `references_p.genre_word.genre` of `RB/outputs/part1/reported.json` (0.4989162, the
instructed Qwen genre image on P; frozen). The promising level is T_g − r_genre (0.4861162). The user's 0.4989 and
0.4861 are these values rounded; the rule compares unrounded values.

**Fresh seeds:** S = {1, 2, 3}, for the winner v* and R6-5, each trained as its seed-0 run. Ā_a(v) = the mean of
A_a(v, L_a, s) over s in S; ē_a = Ā_a(v*) − Ā_a(R6-5). Interval per aspect, its own `rng = default_rng(0)`; for b = 1
to 2,000, in this order: I_b = L_a[`rng.integers(0, len(L_a), size=len(L_a))`]; S_b =
`rng.choice([1, 2, 3], size=3, replace=True)`, shared by both arms (paired by seed); ē_b = the mean over S_b of
A_a(v*, I_b, s) − A_a(R6-5, I_b, s). The 95% interval is `numpy.percentile(ē_b, [2.5, 97.5])`.

## 4. Rules

### 4.1 Stage i

The RB rule §4.2 and §4.3 (V1 to V4; S1-GO, S1-MM, S1-IN, S1-NO; S2-GO to S2-NO; A-HOLDS, A-QUAL, A-FAILS) per
candidate K5 to K8, with these changes:
- **X0** (C12, the extractor) Before any new store row is written, the changed extractor and the extractor at the
  commit before this round's change re-extract the first 512 images (store order) of the CLIP B/32 and SigLIP 2
  ArtELingo stores with the extractor's default batch settings; their rows equal each other bit for bit. Each one's
  difference from the stored rows (maximum absolute difference, rows affected) is reported and decides nothing (the
  stores were written by older extractor versions). If X0 fails, no new store is written and stage i stops; stages ii
  (with K2 to K4) and iii go on; the cause goes in the report.
- **V5** (stores) Each new store has 61,402 (ArtELingo) or 746 (ArtELingo-28) image rows, its names unique and in the
  order of the existing stores, finite rows of the stated width. Content check on the first 1,000 paintings of P in
  sorted order: for K6 to K8, top-1 retrieval from the image to the painting's own caption mean (cosine, the encoder's
  own text embeddings) is at least 1% (10 times chance); CLIP B/32 and SigLIP 2 run the same check first, and if either
  falls below 1% the 1% bar is void (reported) and only the other checks stand. For K5: on the paintings it shares with
  the archived DINOv2-large store (`/data/SSD2/pre_extract/artelingo/r8_enc_dinov2l`, 2026-10-13, same checkpoint and
  pooling), the per-painting cosine is at least 0.99. A store that fails is traced; it may be fixed and checked again
  before any training of that candidate, and the report lists it. If the cause lies only in the text path, the image
  store stands. A candidate whose image store still fails has no outcome (reported with the cause); the others go on.
  A store that fails on ArtELingo-28 alone removes only that candidate's A outcome.
- **X1** (C12) replaces the RB rule's stage 2 gate. Before stage 3: this round's code trains K2 and M2 at seed 0 on the
  sample, and their held and ArtELingo-28 outputs equal 12-24's (`RB/outputs/part2/sample/`) bit for bit; its scorer,
  on 12-24's saved outputs with 12-24's arm lists, gives the values of `RB/outputs/part2/verdict_stage3.json` and
  `verdict_stage5.json` (each A_e, d, both bounds) within 1e-6. If X1 fails, no stage 3 arm trains; stages ii (with K2
  to K4) and iii go on; the cause goes in the report.
- **X2** (C12) replaces the RB rule's stage 3a gate. Before stage 4: the same for K2 and M2 at seed 0 at full scale
  (`RB/outputs/part2/full/`) and `verdict_stage4.json`. If X2 fails, no stage 4 arm trains, and stage 5 runs at the
  sample scale.
- V1 reads: stage 3 needs X0, V5 (for that candidate) and X1; stage 4 needs X2.
- A candidate's stage 4 starts only after its S1 outcome is written.
- What each outcome leads to: the RB rule §4.2. Each candidate's verdict is its own sentence, with no sentence pooled
  across encoders (user) and no correction for several candidates (agent default, as the RB rule). Every outcome goes
  to the next decide chat.

### 4.2 Stage ii

Validity first; no outcome is stated until VL1 to VL4 hold.
- **VL1** (C12) (a) This round's P scorer, on 12-24's saved part 1 exports, gives `RB/outputs/part1/pick.json`'s A
  values for R0-5 and R6-5 and `RB/outputs/part1/reported.json`'s P values for the default and instructed Qwen images,
  within 1e-6. (b) Its P inference of K4 at seed 0 (12-24's saved weights) gives
  `RB/outputs/part1/train/R6-5/{aspect}__s0__P.npz` bit for bit (the same weights; `RB/rb_part1.py`'s inference batch
  size). (c) Its PercepT inference of M2 at seed 0 on the held images gives `RB/outputs/part2/sample/held/M2/P3__s0.npz`
  bit for bit. Otherwise stage ii stops; the cause goes in the report.
- **VL2** (C13) Separate code, given this rule, P's names, the captions and the saved P outputs (never our scores),
  recomputes the labels, the partitions, each A, each lift and each interval. Equal partitions; each value within 1e-6.
  A disagreement is traced and fixed before an outcome is stated, settled by this rule's text; both values and the fix
  go in the report.
- **VL3** Every P output set has 29,232 finite rows in P's order; student rows have unit norm (|norm − 1| < 1e-3);
  P's SHA-256 equals 12-24's.
- **VL4** After any P output exists, no arm or seed is added, dropped or rerun, except a rerun forced by a failed VL3
  (same weights and code), which the report lists.

Outcomes per encoder, on emotion:
- **L-HOLDS** lift > 0 and lower bound > 0. **L-QUAL** (qualified) lift > 0 and lower bound ≤ 0. **L-FAILS** lift ≤ 0.

A confirmation, on development paintings, of 12-24's reading that each student lifts its encoder above its own PercepT
arm. It changes no stage i verdict and chooses nothing in this loop. A method idea drawn from it is tested on
development paintings in a later loop, and on held paintings only as a new claim test (user). The P table (outcomes,
statistics, correlations) is written before the held table is built.

### 4.3 Stage iii

Validity first; no outcome is stated until VG1 to VG4 hold.
- **VG1** (C12) This round's trainer with every variant switch off (multiplier 1, λ = 0, no adaptor) trains R6-5 at
  seed 0, and its P exports equal `RB/outputs/part1/train/R6-5`'s seed-0 exports bit for bit; its scorer gives
  `pick.json`'s R6-5 values and T_g within 1e-6. Otherwise no variant is trained; the cause goes in the report.
- **VG2** (C13) Separate code, given this rule, P's names, the captions and the saved exports (never our numbers),
  recomputes the labels, the partitions, each A, e and interval, the guard, the extras, the winner, the fresh-seed
  effects and intervals, and the outcomes. Equal partitions; values within 1e-6; the same extras, winner and outcomes.
  A disagreement is traced and fixed before an outcome is stated, settled by this rule's text; both values and the fix
  go in the report.
- **VG3** Every export on P has 29,232 finite rows in P's order with unit norm; P's SHA-256 equals 12-24's.
- **VG4** (seed policy) After any variant export on P exists, no variant is added, dropped or retrained beyond the
  extras and fresh-seed runs this rule defines, except a rerun forced by a failed VG3 (same seed and code), which the
  report lists.
- **Failed runs.** A variant whose run stops on a non-finite loss, or still fails VG3 after its rerun, has no
  A_genre: it is not guarded, cannot be the closest method's variant or the winner, and counts as run for its ladder.
  The report gives the cause.

Conditions:
- **Guard:** e_emotion(v) ≥ −r_emotion and e_style(v) ≥ −r_style (point estimates).
- **Closest method:** the method of the guarded variant with the largest A_genre among the five first variants; if
  none is guarded, of the variant with the largest A_genre among the five; on equal values, the first in the order
  GT-2, GT-4, GR-0.5, GR-2, GA-1. w* is that variant's rung. If w* is the largest rung the method has run and that
  variant is guarded, the extras are the next two rungs above w*. Otherwise they are the nearest rung not yet run below
  w* and the nearest rung not yet run above w*.
- **Winner v*:** among all seven variants, the guarded one with the largest A_genre (unrounded); on equal values, the
  first in the order GT-2, GT-4, GR-0.5, GR-2, GA-1, then the extras in ladder order. If none is guarded, there is no
  winner.

Outcomes, seed 0 (the user's levels):
- **III-GO** a winner exists and A_genre(v*) ≥ T_g.
- **III-PROM** (promising) a winner exists and T_g − r_genre ≤ A_genre(v*) < T_g.
- **III-NONE** otherwise. R6-5 stays the recipe.

Fresh seeds, when a winner exists and e_genre(v*) ≥ r_genre (point estimate): v* and R6-5 train on seeds 1, 2 and 3.
R6-5's fresh-seed weights must equal 12-24's K4 weights of the same seed (`RB/outputs/part2/sample/weights/K4/`) bit
for bit (C12); otherwise no C outcome is stated and the cause is traced. A fresh-seed run that fails as under "Failed
runs" leaves no C outcome.
- **C-HOLDS** ē_genre ≥ r_genre with its lower bound > 0, and ē_emotion ≥ −r_emotion and ē_style ≥ −r_style.
- **C-QUAL** (qualified) ē_genre > 0 and both guards on ē hold, but not C-HOLDS. **C-FAILS** otherwise.
- Ā_genre(v*) is reported against T_g and T_g − r_genre.
- **Reading both** (agent default): the seed-0 outcome stands as defined; the C outcome says whether it repeats. A
  III-GO or III-PROM winner with C-FAILS is reported as "not repeated on fresh seeds". A III-NONE winner's C outcome
  is reported and makes it neither GO nor promising.

Weights: every run saves its weights. Those of each variant with A_genre ≥ T_g − r_genre are kept (user); those of
the fresh-seed runs are kept too (agent default); the others go on the pending deletions list for the wrap-up.

What the outcomes lead to: nothing automatically. Stage i's recipe stays R6-5 whatever stage iii finds. Every outcome
goes to the next decide chat; a held test of a winner is a later loop's claim test (user). The claim ceiling is
"promising on development data". Reported, deciding nothing: each variant's three aspects with intervals, the old
5,000, a Leiden reading (as the RB rule's part 1), best epochs.

## 5. Frozen and rerun

- **Frozen:** the RB rule's frozen list (§5, "Frozen" only); 12-24's outputs and weights of K2 to K4, M2 and M3 at
  both scales; the CT's sample-scale core P3 weights (B1) and Stage 1 artifacts; 12-24's part 1 R6-5 exports on P,
  T_g, P and its hash, the labels and their hashes, the seed ranges; the existing encoder stores; this rule.
- **Rerun:** K5 to K8 and M5 to M8 at both scales, seeds 0 to 4, with their held and ArtELingo-28 inference; B1 and
  B6 ArtELingo-28 inference at the sample scale from the CT's saved weights, when a stage 5 runs at that scale; the P
  inference of every stage ii arm; every stage iii run; all scoring and bootstraps.
- **New:** stores of CLIP B/16, CLIP L/14 and SigLIP 1 (images and texts) and DINOv2 (images) for ArtELingo and
  ArtELingo-28.

## 6. Exceptions to the constitution

| Principle | Exception | Why | Simpler option rejected |
|---|---|---|---|
| none | — | — | — |

## 7. ID glossary

| ID | Meaning |
|---|---|
| RB rule, CT rule | 12-24's rule; the claim test's rule (2026-10-20) |
| R6-5 | the recipe without own-caption terms at teacher share 1/5 (12-24's pick) |
| K1 to K4 | 12-24's students (K1: D2 on Qwen; K2 SigLIP 2; K3 CLIP B/32; K4 Qwen with R6-5) |
| K5, K6, K7, K8 | students on DINOv2-large, CLIP B/16, CLIP L/14, SigLIP 1 |
| B1 to B6 | the CT's and 12-24's PercepT entries (B1: core P3, the Qwen student's own arm) |
| M2, M3, M5 to M8 | matched arms (each encoder's own PercepT arm): core P3 with Qwen's Stage 1 topics, Stage 2 on that encoder's image |
| F1 to F8, T | frozen default images (F1 and F4: Qwen; F2, F3, F5 to F8: as the K of the same number); the instructed Qwen image |
| X0, X1, X2; V1 to V5 | stage i's regression gates (extractor, sample scale, full scale); its validity checks |
| S1-*, S2-*, A-* | stage i outcomes at sample and full scale; on ArtELingo-28 |
| P, L_a, r_a | the 29,232 development paintings; those labelled for aspect a; the fixed seed ranges |
| lift, L-HOLDS, L-QUAL, L-FAILS; VL1 to VL4 | stage ii: student minus its own arm's best reading; its outcomes; its validity checks |
| GT-m, GR-λ, GA-w | stage iii variants: genre teacher weight ×m; plus relational term λ; plus regression adaptor term w |
| A_genre(v), T_g, v* | a variant's genre AMI on P; the instructed Qwen genre image's (0.4989162); the winner |
| III-GO, III-PROM, III-NONE; C-HOLDS, C-QUAL, C-FAILS; VG1 to VG4 | stage iii outcomes at seed 0; on the fresh seeds; its validity checks |
