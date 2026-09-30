# CoSiR v2 Candidate A: condition-aware factor learning, 2×2 selection

## Verdict

**Stop: no cell qualifies under the pre-registered rule, so no cell goes forward and the held test is not
run.** We trained four factor models on scorer-train rows (seed 42, 2,000 steps each) and compared each change
with the matched control C0 on the stage-(d) selection label episodes. The criterion is `D`, the change over
C0 in R@1 (the share of episodes where the one correct candidate out of 13 ranks first) when the factor
weights come from the parameter-free naive rule at β 0.3, pooled over emotion and art style and averaged over
the two retrieval directions. A cell qualifies only if it passes all nine collapse gates (checks that the
factor space has not collapsed, listed below), the 95% CI of `D` lies above 0, and the lower bound of the
emotion-only difference `D_emotion` lies above −1.0. All terms are defined in "What we tested".

| Cell | What changed vs C0 | Gates | `D` (points) | `D_emotion` (points) | Qualifies |
|---|---|---:|---:|---:|---|
| C0 | nothing (matched control) | **9/9** | (baseline) | (baseline) | (control) |
| A | painting-level agreement | 8/9 (readout) | −3.03 [−3.85, −2.20] | −1.90 [−3.03, −0.81] | no: gate, `D`, guard |
| S | CLIP-image-cluster condition episodes | 8/9 (sparsity) | **+1.44 [+0.61, +2.26]** | −0.73 [−1.83, +0.39] | no: gate, guard |
| AS | both | 8/9 (readout) | −0.77 [−1.62, +0.07] | −1.59 [−2.69, −0.51] | no: gate, `D`, guard |

In plain terms:

- **The setup is sound.** C0 passes all nine gates and scores 19.71% naive R@1, level with the current system
  (original R3, 19.36%; difference +0.35 [−0.34, +1.04]).
- **The style signal works, but at a price.** S is the only cell that beats C0: +1.44 points pooled, all of it
  from art style (+3.61 [+2.37, +4.86]), mostly in text→image retrieval. It pays with emotion (−0.73, lower
  bound −1.83, below the −1.0 guard) and with denser caption codes (active fraction 0.559 against the 0.50 cap).
  Either failure alone stops it.
- **Painting-level agreement does not free emotion.** A is worse than C0 on both labels (emotion −1.90, style
  −4.15), and its label oracle falls as well, so its code holds less label information, not just differently
  weighted information. The agreement hypothesis, in the form tested here, is not supported.
- **Combining the two cancels the style gain.** AS loses the style gain (+0.05) and keeps an emotion loss (−1.59).

Per spec §6 this ends the experiment at the selection stage. Replication (seeds 43 and 44) and the held test
are not run, and the held rows were not read. What to try next is the user's decision.

![D and D_emotion per cell against the matched control](../../assets/2026-10-16_factor_learning/criterion_d.png)

*Figure 1. Paired differences to C0 in naive R@1 at β 0.3 (R@1 points, mean of image→text and text→image), with
95% bootstrap CIs over episodes. Blue: `D`, the criterion (pooled over 4,096 episodes; its lower bound must
exceed 0). Orange: `D_emotion`, the guard (2,048 emotion episodes; its lower bound must exceed −1.0). Green: the
art-style difference, shown for context and not part of the rule. The row labels give each cell's gate count.*

## What we tested

**The task.** Candidate A scores an image and a caption under a condition given by examples: 4 **supports**
that have the condition and 4 **contrasts** that do not. Each image and caption has a 32-number non-negative
**code** from a small encoder on frozen CLIP ViT-B/32 features (the **factors**). The **naive rule** turns the
examples into factor weights with no parameters: `w = ReLU(mean support pair code − mean contrast pair code)`,
L1-normalized, where a pair code is the mean of a row's image and caption codes. A query `q` and a candidate
`c` then score `β · cos(CLIP_q, CLIP_c) + Σ_l w_l q_l c_l`. **β** weighs plain CLIP similarity against the
condition-weighted factor term and was fixed at 0.3 before the run.

**Episodes.** For a target label (one emotion or one art style), an episode has an anchor with the label, 4
supports and 4 contrasts, and 13 candidates: 1 positive with the label and 12 negatives from paintings never
given it. **R@1** is the share of episodes where the positive ranks first (chance 7.69%). **i2t** uses the
anchor's image as the query and ranks captions; **t2i** uses its caption and ranks images. We used stage (d)'s
2,048 emotion episodes (8 target emotions) and 2,048 art-style episodes (23 target styles) on the selection
rows. Their SHA-256s were asserted equal to stage (d)'s, and every episode row was asserted to be a selection
row.

**Rows.** The 216,107 train rows were split by painting into **scorer-train** (183,694 rows) and **selection**
(32,413 rows, 6,451 paintings). All four models trained on scorer-train rows only; the content graph, the
condition partition and every gate fit used scorer-train rows; all metrics used selection rows. Val and held
rows were never read.

**The four cells** (spec §4). All start from R3's recipe (`R3_CONFIG`: 32 factors, InfoNCE agreement,
decorrelation, 2,000 Adam steps, seed 42) with one shared batch sampler that adds every scorer-train row of each
sampled painting (about 9,300 rows per step).

| Cell | Agreement term | Condition loss | Role |
|---|---|---|---|
| C0 | per pair: InfoNCE between each image and its own caption, same-painting negatives masked | none | matched control |
| A | per painting: InfoNCE between a painting's image code and the mean code of its captions | none | agreement hypothesis |
| S | per pair | 64 episodes per step from CLIP image k-means (64 clusters), naive rule at β 0.3, learnable τ | style signal |
| AS | per painting | as S | both |

The **agreement hypothesis** was that per-pair agreement pulls every caption code toward its painting's single
image code and so keeps caption-only information such as emotion out of the factors. The **style signal** was
that CLIP image clusters, the one label-free partition known to line up with art style (adjusted mutual
information 0.32 in the [headroom probe](2026-10-15_candidate_a_factor_headroom_probe.md)), can train the
factors toward conditions people ask for.

**Collapse gates.** Nine pass/fail checks on a factor space from the R3 repair, with the amended thresholds of
2026-09-29: participation ratio ≥ 8, maximum factor correlation ≤ 0.90, linear readout of CLIP features no worse
than R0's on the same rows, active fraction ≤ 0.50, no dead factors, at most one modality-private factor, top-2
usage share ≤ 0.20, community spanning ≥ 0.75, and code pair retrieval at least half of CLIP's.

**The label oracle** is a diagnostic, not a method: it fits one weight vector per label on half of that label's
episodes and ranks the other half (2 folds, 200 steps), which measures how much label information a code
carries whatever the weighting rule. Its **null** declares a random negative the positive and should sit at
chance.

**Where and how long.** Everything ran on the local RTX 3090. The Task 3 timing smoke projected 8.2 to 9.2
minutes per sequential run and a 3.9 GiB peak. C0, A and S then trained as three parallel processes in 11.1 to
11.3 minutes each. AS was launched in parallel too but ran out of GPU memory at start-up (four processes
reserved more memory than their 3.9 GiB allocated peak suggested), so we reran the same command alone once the
others finished; it took 8.8 minutes. The evaluation took 104 seconds.

## Result 1: the rule, recomputed by hand

The rule (spec §6) has three steps. **Eligible:** all nine gates pass. **Qualifies:** eligible, the lower
bound of `D` > 0, and the lower bound of `D_emotion` > −1.0. **Pick:** the highest `D`, with a 0.5-point tie
band resolved toward fewer changes.

| Cell | All gates pass | `D` lower bound > 0 | `D_emotion` lower bound > −1.0 | Qualifies |
|---|---|---|---|---|
| A | no (readout) | no (−3.85) | no (−3.03) | no |
| S | no (sparsity) | **yes (+0.61)** | no (−1.83) | no |
| AS | no (readout) | no (−1.62) | no (−2.69) | no |

C0 passed every gate, so the first stop point (a broken setup) did not fire. No cell qualifies, so the second
stop point fired. The outcome does not hinge on the gates: with the gates ignored, S would still fail the
emotion guard, and A and AS would still fail on `D`. The rule's own code (`apply_rule`) returned the same
outcome, after its five hand-checked cases passed.

## Result 2: naive R@1 per model, label and direction

![Naive R@1 per model and label type](../../assets/2026-10-16_factor_learning/naive_r1.png)

*Figure 2. Naive-rule R@1 at β 0.3 (mean of the two directions) for the four cells, with 95% bootstrap CIs. Gray
bar and solid line: C0. Dashed line: original R3, the current system. Dash-dot line: CLIP only. Dotted line:
chance.*

Naive R@1 (%) at β 0.3 with 95% CIs for the mean of directions. C0 is the criterion's baseline; original R3
(stage (d)'s checkpoint, trained on all train rows with the original sampler) is the current system.

| Model | Pooled i2t | Pooled t2i | **Pooled mean** | Emotion i2t | Emotion t2i | **Emotion mean** | Style i2t | Style t2i | **Style mean** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CLIP only | 12.18 | 14.67 | 13.43 [12.65, 14.23] | 9.57 | 11.77 | 10.67 | 14.79 | 17.58 | 16.19 |
| original R3 | 18.36 | 20.36 | 19.36 [18.35, 20.35] | 16.50 | 15.14 | 15.82 | 20.21 | 25.59 | 22.90 |
| **C0** | 18.04 | 21.39 | **19.71** [18.74, 20.68] | 14.40 | 15.67 | **15.04** | 21.68 | 27.10 | **24.39** |
| A | 15.99 | 17.38 | 16.69 [15.80, 17.57] | 12.65 | 13.62 | 13.13 | 19.34 | 21.14 | 20.24 |
| S | 18.70 | 23.61 | **21.15** [20.15, 22.13] | 14.36 | 14.26 | 14.31 | 23.05 | 32.96 | **28.00** |
| AS | 17.33 | 20.56 | 18.95 [17.98, 19.87] | 12.99 | 13.92 | 13.45 | 21.68 | 27.20 | 24.44 |

Paired differences to C0 per label and direction (R@1 points, 95% CI):

| Cell | Emotion i2t | Emotion t2i | Style i2t | Style t2i |
|---|---:|---:|---:|---:|
| A | −1.76 [−3.22, −0.24] | −2.05 [−3.56, −0.54] | −2.34 [−4.00, −0.68] | −5.96 [−7.67, −4.25] |
| S | −0.05 [−1.56, +1.46] | −1.42 [−2.93, +0.10] | +1.37 [−0.24, +2.98] | **+5.86 [+4.05, +7.67]** |
| AS | −1.42 [−2.93, +0.10] | −1.76 [−3.22, −0.39] | +0.00 [−1.76, +1.71] | +0.10 [−1.66, +1.81] |

**Reading.** S's whole gain sits in one cell of this table: art style, text→image (+5.86). There the candidates
are images, and the headroom probe found that art style is read from the image (linear probe 59.8%) far better
than from the caption (26.0%). A condition loss built on CLIP image clusters therefore sharpened exactly the
side that carries style. Its emotion loss is also on the text→image side (−1.42), where the image candidates
carry little emotion to begin with.

For A, the largest loss is the mirror image of S's gain: style text→image (−5.96). A's image codes lost style
information. Both of A's emotion directions also fell, which is the opposite of what the agreement hypothesis
predicted for caption candidates (image→text).

**Against the current system.** C0 itself is level with original R3 overall (+0.35 [−0.34, +1.04]), slightly
better on style (+1.49 [+0.46, +2.49]) and slightly worse on emotion (−0.78 [−1.73, +0.20]). So the new sampler
and the smaller training set did not move the baseline much, and the control is a fair stand-in for R3. S
against R3 is +1.79 [+0.93, +2.65] pooled: +5.10 [+3.76, +6.35] on style and −1.51 [−2.64, −0.39] on emotion.

## Result 3: the gates

Values on selection rows (fit on scorer-train rows); image / caption where the gate has two sides.

| Gate (threshold) | original R3 | C0 | A | S | AS |
|---|---:|---:|---:|---:|---:|
| participation ratio (≥ 8) | 21.8 / 20.8 | 22.1 / 21.1 | 25.3 / 27.7 | 20.0 / 18.3 | 22.5 / 25.1 |
| max factor correlation (≤ 0.90) | 0.446 | 0.547 | 0.254 | 0.560 | 0.462 |
| readout rel. L2 (≤ R0: 0.4930 / 0.4691) | 0.4789 / 0.4611 | 0.4804 / 0.4625 | **0.5000 / 0.4713 fail** | 0.4732 / 0.4612 | 0.4899 / **0.4699 fail** |
| active fraction (≤ 0.50) | 0.453 / 0.485 | 0.419 / 0.479 | 0.271 / 0.178 | 0.424 / **0.559 fail** | 0.297 / 0.247 |
| dead factors (0) | 0 | 0 | 0 | 0 | 0 |
| modality-private factors (≤ 1) | 0 | 0 | 0 | 0 | 0 |
| top-2 usage share (≤ 0.20) | 0.079 | 0.078 | 0.083 | 0.081 | 0.078 |
| community spanning (≥ 0.75) | 1.000 | 1.000 | 0.875 | 1.000 | 0.906 |
| pair retrieval / CLIP (≥ 0.5) | 1.194 | 1.116 | 0.875 | 1.071 | 0.934 |
| **gates passed** | **9/9** | **9/9** | 8/9 | 8/9 | 8/9 |

**Why each cell failed.**

- **S, sparsity.** Only the caption side became denser: 55.9% of caption factors are active per row, against
  47.9% for C0, while the image side is unchanged (42.4% against 41.9%). A plausible reading, which we did not
  test: the style conditions are defined by image clusters, which captions express only weakly, and a caption
  code that spreads over more factors scores higher under more weight vectors.
- **A, readout.** A's codes reconstruct CLIP features worse than R0's collapsed codes (image 0.5000 against
  0.4930). They are also much sparser (27% / 18% active), more spread out (participation ratio 25 / 28) and less
  correlated (0.254), and they match an image to its own caption less well than C0's (pair retrieval 0.875 of
  CLIP against 1.116). The code is spread thin: many weakly used factors carrying less of CLIP.
- **AS, readout, by a small margin.** The caption readout misses R0's level by 0.0008 (0.4699 against 0.4691).
  This fail is marginal, but it does not change the outcome, because AS also fails on `D` and on the guard.

## Result 4: the label oracle (does any cell carry more label information?)

![Naive rule vs label oracle per model](../../assets/2026-10-16_factor_learning/label_oracle.png)

*Figure 3. Naive rule at β 0.3 (blue) and the cross-validated label oracle at β 0 (orange), mean of the two
directions, with 95% CIs. Dashed line: original R3's oracle. Dotted line: the oracle null, averaged over
models.*

R@1 (%), mean of the two directions, 95% CIs. R3's oracle at β 0 (20.47%) is the value the headroom probe
reported as 20.5%.

| Model | Naive, β 0.3 | Oracle, β 0.3 | **Oracle, β 0** | Oracle β 0: emotion | Oracle β 0: style | Null, β 0 | Oracle β 0 − own naive | Oracle β 0 − R3 oracle β 0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| original R3 | 19.36 | 19.73 | **20.47** [19.49, 21.47] | 16.77 | 24.17 | 7.54 | +1.11 [+0.23, +2.04] | (reference) |
| C0 | 19.71 | 20.25 | 21.12 [20.13, 22.09] | 17.26 | 24.98 | 7.75 | +1.40 [+0.54, +2.30] | +0.65 [−0.12, +1.44] |
| A | 16.69 | 17.66 | 16.83 [15.93, 17.74] | 14.26 | 19.41 | 7.62 | +0.15 [−0.67, +0.93] | −3.64 [−4.55, −2.73] |
| S | 21.15 | 21.29 | **21.86** [20.86, 22.86] | 15.16 | 28.56 | 7.52 | +0.71 [−0.16, +1.61] | **+1.39 [+0.51, +2.28]** |
| AS | 18.95 | 18.93 | 19.34 [18.37, 20.29] | 13.55 | 25.12 | 7.53 | +0.39 [−0.40, +1.23] | −1.14 [−2.05, −0.23] |

**Reading.** Only S raises the oracle above R3's, by +1.39 points, and the split shows where it came from:
style +4.39 [+2.98, +5.74] and emotion −1.61 [−2.76, −0.44] against R3's oracle. Against C0's oracle, S is
+0.74 [−0.17, +1.64] pooled, with style +3.59 [+2.20, +4.93] and emotion −2.10 [−3.30, −0.83]. So S's code did
not gain label information overall; it **traded emotion information for style information**. The
naive-rule results in Result 2 are therefore a property of the code, not of the weighting. A's code lost both
(oracle −4.28 [−5.19, −3.39] against C0's). Every model stays far below the 49.8% ceiling of the probe's
label-aligned code.

As in stage (d), the oracle beats each model's own naive rule by at most 1.4 points. The naive rule already
extracts nearly all the label information these codes hold. Every null sits at chance (7.52% to 7.75%
against 7.69%), so the oracle does not fit noise.

## Result 5: the β grid (is a gain a code-scale effect?)

Because the factor codes' scale is learned, a larger code makes the factor term outweigh `0.3 · cos`, which acts
like a smaller β (spec §5). We measured this directly as the ratio of the two terms' spread across the 13
candidates at β 0.3, averaged over episodes.

| Model | Code RMS (scorer-train) | Factor term / (0.3 · cos term), spread ratio | Mean active weights per episode |
|---|---:|---:|---:|
| original R3 | 0.315 | 4.43 | 15.7 of 32 |
| C0 | 0.268 | 3.39 | 15.6 |
| A | 0.205 | 1.79 | 14.6 |
| S | 0.325 | **5.16** | 15.6 |
| AS | 0.224 | 2.44 | 15.0 |

S's codes are 21% larger than C0's, so at β 0.3 S leans on CLIP about a third less than C0 does. Naive R@1 (%)
on the β grid, mean of directions, pooled / emotion / style:

| Model | β 0 | β 0.03 | β 0.1 | **β 0.3** | β 1 |
|---|---|---|---|---|---|
| original R3 | 19.58 / 16.19 / 22.97 | 19.58 / 16.14 / 23.02 | 19.71 / 16.43 / 23.00 | 19.36 / 15.82 / 22.90 | 18.40 / 14.16 / 22.63 |
| C0 | 20.25 / 15.89 / 24.61 | 20.09 / 15.58 / 24.61 | 20.13 / 15.58 / 24.68 | 19.71 / 15.04 / 24.39 | 18.36 / 13.40 / 23.32 |
| A | 16.38 / 13.50 / 19.26 | 16.87 / 13.77 / 19.97 | 16.70 / 13.31 / 20.09 | 16.69 / 13.13 / 20.24 | 16.08 / 12.08 / 20.07 |
| S | 21.14 / 13.79 / 28.49 | 21.06 / 13.84 / 28.27 | 21.24 / 14.18 / 28.30 | 21.15 / 14.31 / 28.00 | 20.36 / 13.26 / 27.47 |
| AS | 19.04 / 13.67 / 24.41 | 19.14 / 13.82 / 24.46 | 19.09 / 13.77 / 24.41 | 18.95 / 13.45 / 24.44 | 18.02 / 12.45 / 23.58 |

S − C0 at the same β (paired, R@1 points, 95% CI; context, not part of the rule):

| β | Pooled | Emotion | Style |
|---:|---:|---:|---:|
| 0 | +0.89 [+0.00, +1.76] | −2.10 [−3.27, −0.90] | +3.88 [+2.56, +5.18] |
| 0.03 | +0.96 [+0.09, +1.84] | −1.73 [−2.88, −0.56] | +3.66 [+2.37, +4.93] |
| 0.1 | +1.11 [+0.26, +1.97] | −1.39 [−2.54, −0.22] | +3.61 [+2.32, +4.91] |
| **0.3** | **+1.44 [+0.61, +2.26]** | **−0.73 [−1.83, +0.39]** | +3.61 [+2.37, +4.86] |
| 1 | +2.00 [+1.28, +2.72] | −0.15 [−1.05, +0.76] | +4.15 [+3.03, +5.27] |

**Reading.** At β 0 the ranking uses the factor term alone and does not depend on the codes' overall scale, so
a difference there is a property of the code. S's style gain is +3.88 at β 0, about the same as at β 0.3: it is
**in the code, not a β effect**. S's emotion loss is largest at β 0 (−2.10) and shrinks as β grows, because the
CLIP term masks part of what the code lost. The pooled gain therefore grows with β. Even at β 1 the emotion
difference (−0.15 [−1.05, +0.76]) would sit just below the guard's bound, and β 1 lowers every model's R@1; in
any case, choosing β after seeing the results is what the pre-registration rules out. A is worse than C0 at every β (−3.87 at β 0 to −2.28 at β 1), so its loss is not a
scale artefact either, even though its codes are the smallest.

## Result 6: the condition loss and τ over training

![Condition loss and learned temperature over training](../../assets/2026-10-16_factor_learning/condition_training.png)

*Figure 4. The condition episode loss on the current batch (logged every 50 steps; one batch of 64 episodes, so
noisy) and the learned softmax temperature τ, for the two cells with the condition loss. With 4 positives among
16 candidates, uniform scores give log 4 = 1.386.*

| Cell | Loss at step 1 | Loss at step 200 | Final loss (step 2,000) | τ at step 1 | Final τ |
|---|---:|---:|---:|---:|---:|
| S | 1.100 | 0.405 | 0.466 | 0.0193 | 0.0403 |
| AS | 1.100 | 0.489 | 0.522 | 0.0193 | 0.0345 |

The condition loss fell from 1.10 to between 0.4 and 0.5 within the first 200 steps and then stayed flat and
noisy. AS stayed above S at every logged step after the first: with painting-level agreement, the codes fit the style episodes less well. τ rose
steadily and had not levelled off at step 2,000. τ does not change rankings; a rising τ with a flat loss means
the score gaps grew at the same rate, which matches S's larger code scale (Result 5). The loss curve suggests
most of the condition loss's effect on the fit was reached early, but it does not show whether the ranking
effects (the style gain, the emotion loss) were still growing.

## What this means

1. **The style signal does what it was built to do, and nothing more.** Training on CLIP image clusters moved
   the factors toward art style (+3.61 naive, +3.59 oracle against C0), which confirms that the probe's
   alignment measure (AMI 0.32) points at usable signal. The factors have a fixed budget of 32, and the style
   gain came with an emotion loss (−2.10 in the oracle against C0) and with denser caption codes. As the
   design expected, a style-aligned signal alone does not improve emotion, and here it made emotion worse.
2. **Relaxing agreement is not the same as adding a signal.** Painting-level agreement removes a constraint on
   individual captions, but nothing in the remaining losses rewards emotion, so the freed capacity did not go to
   emotion. Instead the codes became sparser and carried less of CLIP and of both labels. The agreement
   hypothesis may still be right about why R3 lacks emotion; this experiment shows that removing the constraint
   is not enough to put emotion in.
3. **The limit identified by stage (d) and the probe still holds.** Every oracle stays within 1.4 points of its
   own naive rule and far below the 49.8% ceiling. What the factors carry, not how they are weighted, sets the
   result.

## Caveats (spec §11)

- **The control is not R3.** C0 uses the expanded sampler and scorer-train rows only. It is level with original
  R3 pooled (+0.35 [−0.34, +1.04]) and slightly stronger on style (+1.49 [+0.46, +2.49]), so S's gain against R3
  (+1.79) is a little larger than against C0 (+1.44).
- **The selection rows had been read before**: by stage (d)'s selection and post-hoc analyses and by the headroom
  probe, whose readings shaped this design. This step was for choosing, and the held test on new episodes was
  meant to confirm a choice. With a stop, no result here is confirmed on held data, in either direction.
- **Style, not emotion.** The only condition signal was style-aligned, and emotion gains were not expected. The
  guard fired because emotion got worse, not because it failed to improve.
- **Content vs style.** CLIP image clusters also follow content (AMI with style 0.32 is far from 1). S's style
  gain (+3.61) is real but modest, consistent with the factors learning a mix of content and style clusters.
- **Code scale acts like β.** S's codes are 21% larger than C0's. The β grid shows its style gain is in the code
  and its emotion loss is partly masked by CLIP at higher β.
- **Emotion labels are per annotation**, and the image carries little of them (image→emotion probe 35.2% against
  a 28.4% majority), so emotion differences in text→image retrieval rest on weak image evidence.
- **One seed, one split.** The verdict rests on seed 42 alone. The bootstrap CIs resample episodes under fixed
  models and do not cover training randomness. S failed the sparsity cap by 0.059 and the guard by 0.83 points
  on the lower bound; we cannot tell from one seed how stable either margin is. Replication was not run because
  the rule stopped.
- **Training budget.** In Task 2's synthetic check, where the condition signal was weak, the condition loss
  gave no gain at 300 steps but +0.23 R@1 at the real 2,000-step budget. The effect of this loss depends on
  training length, and 2,000 steps was fixed before the run. We did not test whether S's style gain or its
  emotion loss keeps growing with longer training.
- **AS was rerun alone.** The parallel launch of AS ran out of GPU memory at start-up; the rerun used the same
  command, settings and seed, alone on the GPU.

## Sanity checks

- The selection episodes' SHA-256s equal stage (d)'s (emotion `84056321…`, art style `e1cfe1ba…`), and every
  episode row is a selection row (both asserted).
- Every model's codes are finite on scorer-train and selection rows and NaN elsewhere; evaluation codes and CLIP
  features are NaN outside selection rows (asserted).
- Each checkpoint's stored config equals the pre-registered cell config (2,000 steps, seed 42; asserted), and
  every training history covers 2,000 steps. Smoke checkpoints were not evaluated.
- Original R3 reproduces the headroom probe's stored ranks exactly (identical-rank share 1.000): the naive rule
  at all five β values and CLIP only (asserted), the label oracle at β 0.3 and 0 and its null (reported).
- Rerunning the whole evaluation gave identical results.

## Files

- Spec: [factor-learning design](../../../superpowers/specs/2026-09-30-cosir-v2-candidate-a-factor-learning-design.md)
  (§4 cells, §5 losses, §6 rule and stop points, §11 caveats).
- Script: `src/test/20261016_factor_learning_grid/run_grid.py` (`--run CELL`, `--evaluate`, `--tables`;
  `apply_rule` holds the pre-registered rule). It reuses the headroom probe's helpers
  (`src/test/20261015_factor_headroom_probe/run_probe.py`) and stage (d)'s cache.
- Log: `src/test/20261016_factor_learning_grid/20261016_factor_learning_grid_log.md`.
- Figures: `docs/reports/assets/2026-10-16_factor_learning/`, built by
  `docs/reports/assets/build_2026-10-16_factor_learning_figures.py` from the stored results.
- Gitignored, local only: `results/selection_results.json` (every number in this report),
  `results/selection_ranks.npz`, `results/history_*_seed42.json`, the four checkpoints and the run logs.
