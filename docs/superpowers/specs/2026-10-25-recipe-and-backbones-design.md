# The recipe pick on fresh development paintings, then D2 students on SigLIP 2 and CLIP B/32 against PercepT on held paintings: design

> Date: 2026-10-25 03:15, revised 03:28 after the rule review and the cold check · Status: approved 2026-10-25 03:02 through the closing summary (the user confirmed it in
> r10 decide 18, option a; a cold check compares this spec and its rule with the grill record) · Kind: experiment
> Builds on: the [D2 ablation report 12-22](../../reports/auto/percept/2026-12-22_d2_ablation.md) and its
> [spec](2026-10-21-d2-ablation-design.md); the [claim test spec](2026-10-20-claim-test-design.md) and its
> [rule](../../../src/test/20261020_claim_test/DECISION_RULE.md); the grill record in the
> [decide 19 handoff](../handoffs/2026-10-25-r10-decide-19-handoff.md) §2 · Constitution: [constitution.md](../constitution.md)
> Decision rule: `src/test/20261025_recipe_backbones/DECISION_RULE.md` @ a32b63d (first commit, after the review)

## Brief (for the user; agents read and decide from §1 onward)

**What and why.** Part 1 checks whether a recipe without the own-caption terms beats our model (D2) on 29,232
development paintings that never chose D2; a fixed rule picks part 2's recipe. Part 2 trains that model on another
image encoder (SigLIP 2, then CLIP B/32) and tests it against PercepT on the 12,281 held paintings, as the claim test
did for Qwen. ArtELingo-28 comes last. You approved this through the closing summary (2026-10-25 03:02).

**Plan.**
1. Part 1 (confirmation, development data): a regression check, six recipes on one seed, the pick. Beside it, two
   explorations: the information ceiling probe and the four-view statistics sheet.
2. Part 2 code: the student's image input takes any width; a CLIP B/32 store; DINOv2 code; a regression check.
3. Part 2 held tests (claim tests), per encoder, 5 seeds: trained on the 13,400 sample paintings, then on all 42,969
   train paintings unless NO-GO. Qwen reruns if the recipe changes; DINOv2 runs only if "image endpoints only" wins.
4. ArtELingo-28 (confirmation, test data): every model on its 746 paintings, and a descriptive split by language.

**Compared against.** Part 1: D2. Part 2: the strongest PercepT core entry, a matched arm included (PercepT core
with the new encoder's image in its Stage 2). The frozen encoder image and the instructed Qwen image are reported.

**Outcomes.** AMI: agreement of k-means clusters with the labels. Seed range: D2's spread over 3 seeds in 12-22.
- Part 1: a recipe replaces D2 when it gains at least one seed range on an aspect (interval above 0) and loses at
  most one seed range on every aspect. The largest mean gain wins; if none passes, D2 stays.
- Part 2, per encoder: d = the student's held emotion AMI minus the strongest entry's. GO: d ≥ 0.01, interval above
  0. Inconclusive: d ≥ 0.01 with the interval reaching 0, or 0 < d < 0.01. NO-GO: d ≤ 0 (no full stage).
- ArtELingo-28: a model "holds" when it beats PercepT there with the interval above 0.
- We expect: D2 replaced (about 0.75); SigLIP 2 d +0.003 to +0.012; CLIP −0.005 to +0.008.

**Cost.** Local 3090: about 2 to 2.5 GPU h and 4 to 6 CPU h (a guess). Planned to end before 11:00 (Sunday 10-25);
if part 2 would end later, the run chat moves it to Sunday evening by itself. Reports by Monday 10:00.

**Check these.**
- Taken after our recommendation: the encoders, Qwen captions on every student, the bar (the stronger of PercepT
  core and the matched arm), and the full stage unless NO-GO.
- Agent defaults: part 2 as claim tests (stricter than "incremental"); PercepT's style arm in every aspect's bar,
  emotion included; part 1's guard on point estimates, its mean gain over all three aspects, ties at two decimals; a
  recipe passes even with a mean gain at or below 0 (your rule read literally; the report would flag it).
- Agent default: on ArtELingo-28 only PercepT arms on global images can run (it has no patch features).
- Most likely wrong: the end before 11:00, since the build takes the night.

## 1. Context and goal

Asked (decision A, r10 decide 17, 2026-10-25 02:20, the user's pick of the recommended two-part loop): "a as
recommended". The rule's form (r10 decide 18, 02:38): "I will go with 1, this is a more comprehensive one and at most
level keep the best one".

D2 passed both claim tests on held emotion (S1-GO, S2-GO; direction log, 2026-10-21). The ablation (12-22) found
that without the own-caption terms, emotion and style rise on the 5,000 development paintings that chose D2, and the
teacher's loss share rises from 1/5 to 1/3 with them. The framing note 12-21 allows "any backbone" only if a
SigLIP 2 or CLIP B/32 student stays above PercepT core.

Goal: fix the training recipe on fresh development paintings; then test whether the emotion lead over PercepT holds
when the student reads another encoder's image.
Questions this step answers: (part 1) does a recipe without own-caption terms, or with a larger teacher share, beat D2
on paintings that never chose it? (part 2) does a student on SigLIP 2 or CLIP B/32, trained with Qwen's graphs,
teacher and captions, beat the strongest PercepT core arm on held emotion by 0.01?

## 2. Scope

In: part 1 (stages 0, 1, 1x); part 2 (stages 2 to 5): the code, the CLIP B/32 store, the DINOv2 code and (if
triggered) its stores, the held tests, ArtELingo-28.
Out: retrieval and the instructed Qwen bar as a gate (11-10; the target row is only reported); PE-Core; tuning per
encoder; new losses (literature note 12-20, ideas 2 to 12); the parked ideas, culture as a condition included (stage
5 only describes); paper text. They go to the next decide chat.

## 3. Design

Part 1 inherits the ablation spec §3 (heads, losses, settings, label-free early stopping, scorer); part 2 inherits the
claim test spec §3. Changes below. IDs follow the rule (§7 there).

### 3.1 Hypothesis and prior

- **Part 1 (confirmation).** The claim to check (12-22): without the own-caption terms, or with image endpoints only,
  image AMI rises on emotion and style (+0.0055 / +0.0180 and +0.0057 / +0.0248). The six recipes separate the
  removal from the teacher share. Prior: the gains shrink on fresh paintings (they were found on the 5,000 that chose
  D2); the style gains are two to three seed ranges, so they likely survive; image endpoints only sits near the genre
  guard (−0.0086 against −0.0128). On 12-22's numbers, "no own-caption terms" has the larger mean gain (1.29 against
  1.20 seed ranges). Rule §1 gives the prior per outcome.
- **Part 2 (claim tests).** The instructed Qwen graphs and teacher carry D2's gain (12-22). So a student that reads
  another encoder's image but trains on Qwen's graphs, teacher and captions should keep most of it. Prior d at the
  sample scale: SigLIP 2 +0.003 to +0.012; CLIP B/32 −0.005 to +0.008 (CLIP's frozen image emotion 0.065 on
  development, below Qwen's 0.070). A SigLIP 2 d above 0.02, or a CLIP d below −0.015, would surprise us.

### 3.2 Arms and controls

Part 1. Loss = buddy terms + own-caption terms + w × teacher term; the same loss on the 1,489 stopping paintings
stops training per aspect.

| Arm | Recipe (teacher share) | w | Role |
|-----|------------------------|---|------|
| R0-5 | D2 (1/5); 12-22's R0 | 0.5 | reference; the default |
| R0-3 | D2 (1/3) | 1.0 | candidate: the teacher share alone |
| R6-5 | no own-caption terms (1/5) | 0.25 | candidate: the removal alone |
| R6-3 | no own-caption terms (1/3); 12-22's R6 | 0.5 | candidate |
| R7-5 | image endpoints only: image–image buddy term and teacher (1/5) | 0.25 | candidate |
| R7-3 | image endpoints only (1/3); 12-22's R7 | 0.5 | candidate |

References without training, on the same paintings: the default and instructed Qwen images.

Part 2. A student is the picked recipe with one change: the image network's first layer takes the encoder's width.
Its caption network reads the default Qwen caption mean (when the recipe has one); graphs, teacher, settings and
2048-wide outputs stay. Nothing is tuned per encoder.

| Arm | What | Role |
|-----|------|------|
| K2, K3 | student on the SigLIP 2 so400m image (1152) / the CLIP B/32 image (512) | candidates |
| K4 | student on the default Qwen image; only if part 1 changes the recipe | candidate; D2's claim result (K1) beside it |
| K5 | student on the DINOv2-large image (1024); only if part 1 picks image endpoints only | candidate |
| B1 to B5 | the claim test's PercepT entries (core P3; paper strict and K = 40, P1 and P4), saved held outputs | comparators |
| B6 | PercepT core P2 (CLIP + RoBERTa topics, CLIP global image), saved held outputs | comparator |
| M2, M3, M5 | matched arm: PercepT core P3 with the claim test's saved Qwen Stage 1 topics; Stage 2 trained on the candidate's encoder image (K4's matched arm is B1) | comparator: drops only the student |
| F2, F3, F5 | the candidate's frozen encoder image, clustered directly | reported |
| T | the instructed Qwen image of the aspect | reported |

Stage 5 runs K1 (the claim test's saved weights) and K2 to K5 where they exist, with comparators that need only
global images: B1, B6 and the matched arms (P1 and P4 need patch features, absent on ArtELingo-28). F1, the frozen
default Qwen image, is reported there beside F2, F3 and F5.

### 3.3 Data roles

| Data | Role | Stage |
|------|------|-------|
| 13,400 fit and 1,489 stopping paintings (the frozen sample) | training; label-free early stopping | 0, 1, 1x, 2, 3 |
| 5,000 old evaluation paintings | development: regression; reported beside | 0, 1, 1x |
| 29,232 train-side paintings outside the 19,889-painting sample (23,080 train, 6,152 validation) | development: the pick, the probe and the sheet; in stage 4, training (23,080) and early stopping (6,152) | 1, 1x, 4 |
| 42,969 train paintings | training | 3a, 4 |
| 12,281 held paintings | held: images for inference, captions for labels only | 2 (regression outputs), 3, 3a, 4 |
| ArtELingo-28 read set, 746 paintings (248 in our held set, 498 outside our store): its own images and translated captions | test: images for inference, captions for labels only | 5 |

The 29,232 never chose anything: claim test stage 2 only trained and stopped on them. No held or ArtELingo-28 number
enters a choice (`held-data.md`).

### 3.4 Measurement

- **Part 1.** Image-side AMI per aspect against its own label: k-means k = 20, random states 0 to 2, on the 29,232
  (genre: its labelled paintings); model seed 0; early-stopped exports; 12-22's scorer and labels. Effect = recipe
  minus R0-5, with a 95% paired percentile bootstrap over each aspect's labelled paintings, 2,000 draws, as 12-22
  (rule §3.1). Seed ranges fixed from
  12-22: 0.0046 / 0.0082 / 0.0128. Beside, deciding nothing: the old 5,000; caption-side AMI (none for R7); a Leiden
  reading (union 20-NN, resolution 1.0, seeds 0 to 2); an export at epoch 4; best epochs.
- **Part 2.** The claim test's protocol (its rule §3): readings r20 and own; mean over k-means states, then over model
  seeds 0 to 4; d = candidate minus the maximum over eligible entries; paired bootstrap over held paintings and seeds,
  maximum taken again per draw; Leiden beside. Style and genre: the same statistic against (a) the maximum over B1 to
  B6, (b) the matched arm, (c) the frozen encoder image, and the maximum of (a) and (b).
- **Stage 5.** The same scorer on the 746. Labels: emotion with "other" merged into "something else", plurality over
  nine classes, ties to the smallest code (729 unique, 17 ties); style, its 27; genre, its nine. Also on the 248 and
  the 498 subsets. Culture split: per language, a painting's label is the plurality of that language's captions (same
  rule), on paintings with at least one caption in it; AMI of each model's partition (fit on all 746) against it, and
  its agreement with the pooled label.
- **Seeds.** Part 1: one seed (`held-data.md`, "Seeds": pick on one, test on several). Part 2: five, as the claim test.

### 3.5 Stages, gates and the regression check

| Stage | Type | Input | Gate to pass | If it fails |
|-------|------|-------|--------------|-------------|
| 0 | regression (C12) | part 1 code; R0-5, R6-3, R7-3, seed 0 | old-5,000 exports equal 12-22's R0, R6, R7 seed 0 bit for bit; genre labels on the 29,232 counted; one export's scoring time measured | stop; trace; report; part 2 does not start |
| 1 | confirmation, selection rule | six recipes, seed 0 | rule §4.1, after the re-derivation agrees | — |
| 1x | exploration | the probe; the four-view sheet | none | reported as missing |
| 2 | engineering, regression | width-opened code with the Qwen input; CLIP B/32 store; DINOv2 code | claim test K1 and B1 seed 0, sample scale, held outputs bit for bit; the part 2 scorer on the claim test's saved outputs gives its stage 1 values (each entry, d, bounds) within 1e-6 | stop part 2; trace; report |
| 3 | claim test, sample scale | K2, K3 (K4, K5 if triggered), matched arms, seeds 0 to 4 | rule S1 outcomes | — |
| 3a | regression | the same code at full scale, Qwen input, seed 0 | claim test K1 and B1 full-scale held outputs bit for bit; the scorer gives its stage 2 values within 1e-6 | stop stage 4; trace; report |
| 4 | claim test, full scale | candidates not S1-NO, with their matched arms | rule S2 outcomes | — |
| 5 | confirmation (test data); exploration (culture split) | ArtELingo-28 | holds / qualified / fails (rule §4.3) | — |

Order: part 1 first (part 2 needs the pick); candidates in the order K2, K3, K4, K5, so DINOv2 is the first deferred if time runs
short (never dropped); a candidate's stage 4 starts only after its S1 outcome is written; stage 5 last.

### 3.6 Diagnostics (decide nothing; part 2's after the verdict file)

- **Ceiling probe (1x).** Per aspect, ridge and a 4-layer MLP predict the instructed image from the default one
  (cosine plus smooth-L1 on standardized targets); fit on the 13,400, stopped on the 1,489's regression loss (no
  labels); scored on the 29,232 and the old 5,000: AMI of the predictions (as part 1), beside the instructed and
  default images and R0-5. Reading: genre near the instructed image (0.4956 on the old 5,000) means the default image
  holds the information and the losses are the limit (ideas 2 and 3 next); near the default image (0.4486) means it
  is missing (ideas 5 or 8 next).
- **Four-view sheet (1x).** Per view (default; emotion, style, genre instructed): norms as stored, anisotropy (mean
  pairwise cosine, top component's variance share), each instructed view's cosine to the default (mean, quartiles),
  20-NN label purity per aspect, cosine silhouette per label, hubness (skew of 20-NN in-degree). Image side on both
  sets; caption side on the old 5,000 only (the 29,232 lack style and genre instructed captions). Reading: high purity
  with low AMI points at the clustering, not the information.
- **Part 2.** Style and genre margins against the three bars; T; Leiden; each arm's seed spread; best epochs per
  student; the CLIP B/32 store's cosine to the legacy store, per painting.

### 3.7 Frozen vs rerun

Frozen: the sample and its graphs (manifest hash); the claim test's 42,969-node graphs, Stage 1 artifacts, held
outputs (B1 to B6, K1), K1 weights and collapse records; the held list and its hash; the scorers' protocols; 12-22's
seed ranges; this rule. Rerun: part 1's six recipes; part 2's students and matched arms at both scales; inference of
every new arm; scoring and bootstraps. New: the CLIP B/32 store (61,402 paintings), the DINOv2 stores (61,402 and
746, if triggered), three instructed Qwen images of the 746.

## 4. Decisions

Times are Amsterdam time on 2026-10-25, from the user's messages in the decide 17 and 18 chat logs (decide 17: 02:20
to 02:31; decide 18's round 2: 02:35 to 03:02, messages 02:38 to 03:00).

| ID | Question | Decision | Why (alternative rejected) | Decided by | When |
|----|----------|----------|----------------------------|------------|------|
| D1 | Direction (A) | two parts in one medium loop | part 1 fixes the recipe part 2 runs; saves a loop before 11-10 | user, after seeing the recommendation | 02:20 |
| D2 | Part 1 types | 1: confirmation with a selection rule; 1x: exploration | decision A | user, after seeing the recommendation | 02:20 |
| D3 | Part 2 types | 2, 3a: engineering and regression; 3, 4: claim tests; 5: confirmation and exploration | held paintings need a committed bar; stricter than A's "incremental" | agent default | 03:15 |
| D4 | Fresh paintings | all 29,232 outside the sample; training stays on the 13,400 and 1,489; the old 5,000 beside | "b will be a more comprehensive choice" (a 5,000 draw) | user | 02:31 |
| D5 | How part 1 picks | a fixed rule | descriptive (changed after the what-if) | user, after seeing the recommendation | 02:31 |
| D6 | The rule's form | guarded replacement (rule §3.1, §4.1) | ranked rule; a recipe per head | user | 02:38 |
| D7 | Its what-if (the genre guard blocks a larger style gain; none passes: D2 at 1/5) | accepted | — | user | 02:44 |
| D8 | Encoders | SigLIP 2, then CLIP B/32; DINOv2 only if image endpoints only wins; its code built anyway | all three in order (the user's first answer) | user, after seeing the recommendation (flag) | 02:48 |
| D9 | The student's caption side | Qwen default captions for every student | the encoder's own captions (user unsure) | user, after seeing the recommendation (flag) | 02:52 |
| D10 | Bars | (a) PercepT core as now, (b) the matched arm, (c) the frozen encoder image: all reported | one bar | user | 02:48 |
| D11 | The claim's bar | the strongest entry per aspect, the matched arm included | (b) alone (the user's first answer) | user, after seeing the recommendation (flag) | 02:54 |
| D12 | Paintings | held; trained on the 13,400 first, then the full split | development only | user | 02:52 |
| D13 | When the full stage runs | unless S1-NO | only after S1-GO | user, after seeing the recommendation (flag) | 02:54 |
| D14 | Part 1 minors 1 to 6 | six recipes and weights (§3.2); seed 0 and 12-22's ranges; early-stopped exports, 2,000 draws, ties to fewer changes; epoch-4 export reported only; the probe (§3.6); the sheet with the user's statistics (direction log, 2026-10-21 entry), the genre count and the scoring time (§3.5, §3.6) | — | user, after seeing the recommendation | 02:57 |
| D15 | Part 2 minors a to f | order (§3.5); students (§3.2); K4 if the recipe changes; bars; pass on emotion, 5 seeds, stage 4 unless S1-NO; local 3090, end before 11:00, no heavy run over 1 h on weekdays 10:00 to 19:00 | — | user, after seeing the recommendation | 02:57 |
| D16 | ArtELingo-28 | (a) test set on the 746, (c) descriptive culture split | (a) alone | user | 02:31 |
| D17 | ArtELingo-28 minors i to v | own images for all 746 (its Qwen image differs from ours on the 248 shared, mean cosine 0.934, cause not checked); labels (§3.4); three instructed Qwen images of the 746; every student and Qwen scored | — | user, after seeing the recommendation | 02:57 |
| D18 | Retrieval for 11-10 | image queries with an aspect name only; not in this spec | — | user, after seeing the recommendation | 02:20 |
| D19 | Plan page; approval; restart | none ("plain"); approval through the closing summary, a cold Sonnet `high` check against the grill record; no restart | — | user | 02:57, 03:00 |
| D20 | B6 | core P2 enters every aspect's entries at both scales, from saved held outputs | P2 for style only (the max can only rise; C1) | agent default | 03:15 |
| D21 | B1 to B5 | the claim test's saved held outputs and collapse records, frozen | retrain (same code and seeds; costs hours) | agent default | 03:15 |
| D22 | Matched arm | the claim test's saved Qwen Stage 1 artifacts of seed s (that scale), K = 40, P3's Stage 2 settings, input = the encoder's default image | retrain Stage 1 (it does not see the image encoder) | agent default | 03:15 |
| D23 | Regression form | bit for bit, as C12 says, in stages 0, 2 and 3a (3a at full scale); the part 2 scorer reproduces the claim test's verdict values within 1e-6; no tolerance escape | the claim test's AMI escape of 1e-4 (the local 3090 repeated bit for bit on 2026-10-20) | agent default | 03:15 |
| D24 | Re-derivation (C13) | separate code written from the rule alone recomputes part 1's effects, intervals and pick, and part 2's d, intervals and eligibility | — | agent default | 03:15 |
| D25 | Rule details the grill left open | the guard reads point estimates; the mean gain averages all three aspects (losses included); ties on the gain rounded to two decimals; a pass with a mean gain ≤ 0 still passes and is flagged; the seed ranges are the user's rounded values; bootstrap draws per aspect over its labelled paintings, as 12-22 | the interval for the guard (stricter than the user's words); a condition g > 0 (not in the user's rule) | agent default | 03:15 |
| D26 | CLIP B/32 store | new canonical store of 61,402 paintings from the shared extractor; K3, M3 and F3 read it; its cosine to the legacy store is reported | the legacy caption-indexed store (D15, minor a) | agent default | 03:15 |
| D27 | DINOv2 encoder | the pooled output (CLS after the final layer norm), L2-normalized, the processor's default input size; stores for 61,402 and 746 paintings | mean of patch tokens | agent default | 03:15 |
| D28 | ArtELingo-28 models and reading | "Qwen" in minor v = K1 and K4, with the frozen default Qwen image (F1) reported; full-scale weights when stage 4 ran, else stage 3's; comparators B1, B6, matched arms; "holds" per rule §4.3; the 248 and 498 subsets | PercepT absent there (fails C1) | agent default | 03:15 |
| D29 | Culture split | §3.4 | per-language counts only | agent default | 03:15 |
| D30 | Leiden | beside, in parts 1 and 2 | — | user ("a Leiden reading in all major tests", r10 decide 12; claim test spec D32) | 2026-10-20 |
| D31 | What part 2 outcomes lead to | S2-GO gives its sentence (§6); every outcome goes to the next decide chat; the paper's recipe is part 1's pick, and K1's and K4's held outcomes test it without choosing between them (a later change says it was made after reading held data) | choosing between K1 and K4 on held numbers (selection on held data) | agent default | 03:15 |
| D32 | Timing | planned to end before 11:00 (Sunday 10-25); the run chat estimates the end at start; if part 2 ends after 11:00, it chooses a split by itself and records why; reports by 2026-10-26 10:00 | — | user (end before 11:00, minor f); the rest agent default (`shared-resources.md`) | 03:15 |
| D33 | Names | round folder `src/test/20261025_recipe_backbones/`, module prefix `rb_`; report sequence name 12-24 | — | agent default | 03:15 |
| D34 | Details beyond the record | the probe's loss (literature note 12-20, idea 1) and its reading; the sheet's caption side on the old 5,000 only (the 29,232 lack style and genre instructed captions); part 1's extras (caption AMI, Leiden, best epochs); the priors for part 1, K4, K5 and ArtELingo-28 (C8); a candidate's stage 4 waits for its S1 outcome | each the plain form of the record's item | agent default | 03:15 |

**Amendments:** none.

## 5. Assumptions and open questions

- Assumption: the 29,232 have genre labels on most paintings. Checked: not checked (the old 5,000 had 4,053); stage 0
  counts them.
- Assumption: the claim test's saved outputs, weights and Stage 1 artifacts are complete and match its verdicts.
  Checked: listed on 2026-10-25 (P2 and P3 held outputs and weights at both scales); stages 2 and 3a reproduce them.
- Assumption: the local 3090 repeats training bit for bit. Checked: the claim test's stage 2a on 2026-10-20; again in
  stages 0, 2 and 3a.
- Assumption: the SigLIP 2 store's values are correct. Checked: IDs and its finite and norm gate; values against
  another store not checked.
- Assumption: the CLIP raw images and the DINOv2 weights load. Checked: not checked (cached, listed).
- Open questions: none.

## 6. Success criteria

**Decision mode.** Binding rule: `src/test/20261025_recipe_backbones/DECISION_RULE.md` at the commit in the header,
committed before any code; it governs where it differs from this spec. Its criteria:
- P-KEEP and P-<recipe>: part 1's pick (rule §4.1); V0, VP1, VP2: its validity checks.
- S1-GO, S1-MM, S1-IN, S1-NO and S2-GO to S2-NO per candidate (K2 to K5): the claim test's outcomes; V1 to V4.
- A-HOLDS, A-QUAL, A-FAILS per model on ArtELingo-28 (rule §4.3).

**Selection rule.** Part 1's pick (rule §3.1, §4.1): image AMI effect per aspect; "at least small" and the guard in
12-22 seed ranges; the largest mean gain; tie order R0-3, R6-5, R6-3, R7-5, R7-3.

**Licensed claims.**
- Part 1: "On 29,232 development ArtELingo paintings that did not choose D2 (one seed), <recipe> changes image AMI
  by <e> [interval] on <aspect> against D2." Development only; it says nothing about held data.
- Part 2, S2-GO for encoder X: "On 12,281 held ArtELingo paintings, a model whose image side reads the default X
  image, trained on the 42,969 train paintings with Qwen3-VL-Embedding's instructed graphs, instructed teacher and
  default captions, gives emotion clusters (k-means, 20 clusters) with AMI at least 0.01 above the strongest PercepT
  core reimplementation trained on the same paintings, one whose Stage 2 reads the same X image included (5 seeds;
  paired 95% interval over paintings and seeds above 0)." S1-GO: the same with "trained on 13,400 paintings". Not
  covered: training without Qwen, style and genre, other datasets, PercepT with its own encoders, retrieval. Under an
  image-endpoints-only pick the student has no caption network, so "and default captions" is dropped.
- Stage 5, A-HOLDS: "The emotion lead over PercepT core holds on ArtELingo-28's 746 paintings (their own images,
  English translations of their captions)." A confirmation; never a rescue of a failed part 2 test.

## 7. Constitution check

| Principle | Pass or exception | Note |
|-----------|-------------------|------|
| C1 strongest comparator | pass | the maximum over the claim test's entries, P2 and the matched arm; frozen encoder and instructed Qwen images reported |
| C6 sample IDs | pass | the held list and its hash; joins by painting names and extraction IDs; ArtELingo-28's `row_id` against our `sample_id` |
| C7 rule before code | pass | reviewed by a fresh Opus `xhigh` worker; committed before any code |
| C8 prior and disclosure | pass | rule §1; §8 below |
| C12 regression first | pass | stages 0, 2, 3a |
| C13 re-derivation | pass | D24 |
| `held-data.md` | pass | part 1 reads development only; nothing reads held data to choose |

Exceptions: none.

## 8. Risks

| What could make the result wrong or misleading | How we would notice | What we do then |
|---|---|---|
| One seed: a single seed's noise is near one seed range, the "small" threshold | effects near one range | the rule stands; part 2's five seeds on held data, and K4 against K1, test the pick |
| The genre guard rests on exports at epoch 1 or 2 (12-22) | best epochs | the epoch-4 export is reported; the rule stands |
| The matched arm is weaker than PercepT core on Qwen (CLIP estimate 0.067 against 0.0855 on held) | entry table | the bar is then the strongest entry, B1 among them, as the user chose (D11) |
| Another encoder's store differs in preprocessing | the frozen encoder row; CLIP parity | reported; the regression covers only the Qwen path |
| ArtELingo-28's images differ from ours (248 shared, mean cosine 0.934) | the 248 subset | stated in the report |
| Part 2 ends after 11:00 | the end estimate at start | the run chat splits it (D32) |

Selection disclosure. Part 1: the 29,232 are new to any choice; six recipes on one seed, so the winner's gain is
inflated by the pick among five candidates. The old 5,000 chose D2 among 13 sheet rows, and the caption-free rows came
from the ablation on them. Part 2: the held paintings were read by the claim test (K1 and B1 to B5 at both scales) and
by older lines (held ledger); no choice was made on them, and none is made here. The encoders were chosen without held
numbers. Protection: held data, five seeds, the comparator's maximum taken on held data.

## 9. Deliverables and review

- At the end: round folder `src/test/20261025_recipe_backbones/` with the rule, the pick file, the verdict files and
  the re-derivations (outputs gitignored); the new stores under `/data/SSD2/pre_extract/`; one auto report (12-24)
  in `docs/reports/auto/percept/` with a results brief, and a user-read report; one row per held read in
  `docs/superpowers/held_ledger.md`.
- Reviews: the rule by a fresh Opus `xhigh` worker before its commit, and a cold Sonnet `high` check of spec and rule
  against the grill record; the build per `matt-chain.md`, with ticket reviews and unit tests for verdict code (the
  pick, part 2's scorer and bootstrap, the ArtELingo-28 labels, the re-derivations); the final whole-branch review;
  Claude's claim review of the auto report.
- Stop points: none, except a failed stage 0, 2 or 3a.
- Budget: GPU about 1.6 h on the local 3090 under the lock, 2.4 h with K4 and DINOv2 (part 1 about 25 min; CLIP
  store about 10 min; students and matched arms at both scales about 45 min; K4 about 15 min; DINOv2 about 35 min;
  ArtELingo-28 about 5 min; regressions about 10 min). CPU 4 to 6 h at 8 threads (scoring on 29,232 paintings, bootstraps, Leiden, the sheet,
  re-derivations). Time-box: reports by 2026-10-26 10:00.
- Length: the body is about 230 lines against about 180 for a confirmation plus two claim stages, because of the 34
  carried and new decisions and the two arm tables.
