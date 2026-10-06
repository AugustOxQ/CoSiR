# Decision rule: the reader fix with the CSD style grouping (plan (a)), committed before any code or number

**Written** 2026-10-06 02:37 (Amsterdam), after the ARS methodology-focus review of the plan and the user's decisions
on it (2026-10-06 02:25), before any script of this folder exists and before any number it governs has been computed.

**Precedence.** Where this file differs from the plan handoff (`docs/superpowers/handoffs/2026-10-06-reader-fix-with-csd-handoff.md`,
§5.1 to §5.6), the run handoff (`docs/superpowers/handoffs/2026-10-06-reader-fix-run-handoff.md`), the review memo
(`ars_review/memo.md`), its Appendices A to F, the editorial decision or the review report
(`docs/reports/auto/v2/2026-11-17_ars_reader_fix_plan_review.md`), **this file governs**. Everything needed to apply the
rule is written here. Letter item S8's memo edits are carried here instead: the memo is the frozen review packet and is
not edited, and its bar line and its label "RCA (13.38, the GO bar)" are superseded by D12 and §6.5.

**Checked** before its commit by a fresh reviewer against the letter's acceptance criteria (2026-10-06 02:38 to 02:50);
its 16 findings were applied.

**Status.** Items 1 to 5 (§5) are development selection on episode seed 42, which has been read many times; their
numbers are exploratory. Item 6 (§6), the fresh-seed test on seeds 49, 50 and 51, is the only confirmatory step.

**Authorisation.** The user adopted every review fix (letter items R1 to R5 and S1 to S17, and the controller's AR
check), set the cutoff (Thursday 8 October 12:00), ruled that a GO on an A0 configuration counts as a GO for plan (a),
and allowed this file to be committed to `main` without a further request (2026-10-06 02:25). After its commit this
file changes only with the user's approval.

**Dates.** Folder and report dates (`20261117`, `2026-11-17`) are sequence numbers, not calendar dates. Calendar times
in this file are Amsterdam local time.

**Prior.** Our own estimate of the chance of a GO is moderate at best: every fresh-seed test so far roughly halved the
development effect (§3, D12).

## 1. Glossary

| Term | Meaning in this file |
|---|---|
| episode | a query (the anchor row: its image for image-to-caption ranking, its caption for caption-to-image ranking), 4 support pairs, 4 contrast pairs and 13 candidates in the other modality; p_A shares the query's value of aspect A, p_B its value of aspect B, 11 negatives share neither |
| per-anchor | per episode (one value per episode, averaged over its 4 rankings) |
| configuration | a set of groupings the reader chooses among (A0, A1, AR) |
| candidate | a reader on a configuration (§5, item 1); the **carried candidate** is the one item 4 sends to the test |
| condition a / b | a: the supports show aspect A and the target is p_A; b: supports and contrasts swapped, target p_B. Each episode gives 4 rankings (2 conditions × 2 directions) |
| grouping | a label-free partition of the scorer-train rows (§3, D1). The plan's groupings are fixed |
| head | a logistic regression on frozen CLIP ViT-B/32 features predicting a row's group of one grouping, from the image alone (image head) or the caption alone (caption head) |
| p_h(x) | the head posterior of item x on grouping h, from its own modality's head |
| Δ_h | support-pair agreement minus contrast-pair agreement on grouping h (D3) |
| s_h | the grouping score of a candidate: query posterior · candidate posterior on grouping h (D4) |
| reader | the rule that decides, per episode and condition, which grouping (or which mixture) scores the candidates |
| T | the reader term: the grouping score the reader selects (D5) |
| T_cf | the two-condition mean of T, the matched counterpart's term (D7) |
| B | the best condition-free score of the project (D8): cosine, the centred factor term T_N1u and the averaged-heads term T_6u, cross-fitted |
| T_N1u | the centred uniform factor term of the method-A checkpoint (`src/test/20261101_aspect_factor_gonogo/checkpoints/A3_seed42.pt`, SHA-256 dadfef1bed95bbd647c39fa0ea483edea78627d440a23b5a032083e76dd469e2), a condition-free term |
| T_6u | the averaged-heads term: the mean over a set of groupings of s_h, condition-free |
| B′ | B rebuilt with T_6u averaged over a configuration's own groupings (D9) |
| A0, A1, AR | configurations: the groupings a reader chooses among (D1). AR is the random-grouping control, never a candidate |
| R-a, R-b, R-c | the reader candidates: scaled Δ (§4.1), learned reader (§4.2), confidence gate (§4.3) |
| told mapping | the evaluation-label map aspect → grouping (D13), used only for the diagnostic pick accuracy |
| bank | pseudo-aspect episodes built from the groupings on scorer-train rows (§4.2) |
| parity halves | the two cross-fit halves of a seed's episodes: episode index parity |
| letter R1 to R5, S1 to S17 | the items of `ars_review/editorial_decision.md`; they are not rule items |

## 2. Data, metrics and intervals

- **Development episodes:** `src/test/20261030_aspect_baselines/results/episodes_seed42.npz` (SHA-256
  12af979432ff1a20c88b614ab9e672f203eeeed9c2465bff305e72d01e6c0986): 12,288 episodes (4,096 per aspect pair:
  emotion × style, emotion × genre, style × genre) on 4,602 anchor paintings, selection rows, loaded through the
  step-1 context (`src/test/20261112_community_sweep/run_sweep.py::setup`).
- **Metrics,** per episode averaged over both conditions and both directions, pooled over the three aspect pairs, in
  percentage points: R@1 (the target ranks strictly first; ties miss), other-aspect rate, condition gain = R@1 − other,
  either rate = R@1 + other. A condition-free scorer has gain exactly 0 on every episode.
- **Intervals:** 95% percentile intervals of the bootstrap that resamples anchor paintings (5,000 resamples, seed 42;
  `src.eval.aspect_metrics.cluster_bootstrap`, as `run_checks.point_ci` calls it in step 1). Cross-fit picks are fixed
  before resampling. "A lower bound above 0" means strictly greater than 0.
- **Precision:** every threshold in this file applies to the full-precision point or bound, never to a rounded value.
- Held rows are not read. Nothing in this file uses the GPU.

## 3. Definitions (the only definitions; every item below refers to them)

**D1. Groupings and configurations.** Five groupings, none re-chosen:
- *affect*: Leiden communities on GoEmotions caption probabilities, 41 groups (`partition_L` in
  `src/test/20261111_community_told_oracle/results/per_anchor_told_oracle.npz`, SHA-256
  27e8101161607da2d6936aa5a8c84f8c296629fe08e37a68c8418c9356af1366);
- *image* and *caption*: k-means with 64 clusters on CLIP image or caption features, built in folder
  `20261031_pseudo_partitions` (E2; `src/test/20261031_pseudo_partitions/results/partitions.npz`, SHA-256
  cfd57dbb21216cdc9d1c2e42d1b408648bc18147f62354b52202ade4ab8a9caa);
- *csd*: Leiden communities on CSD style embeddings of the painting images, 17 groups (`style_csd` in
  `src/test/20261116_grouping_step1_style/results/step1_group_style.npz`, SHA-256
  b04d96b4798acdfdbc9ca87436012755a681c931ac4f7612fa5683350b7a20e2);
- *rand*: `style_rand` in the same file, csd's painting labels permuted across paintings (seed 0); it carries nothing.

E2's own affect grouping (k-means 64 on GoEmotions probabilities, "affect-km") is used only inside B (D8) and is not
one of the five.

Configurations, in this order (the order breaks every arg-max tie: the first grouping wins):
A0 = (affect, image, caption); A1 = (affect, image, caption, csd); AR = (affect, image, caption, rand).

**D2. Standard heads** (used for every evaluation, on seed 42 and on the test seeds), exactly as in step 1: the affect
heads refit with `run_told_oracle.fit_one_head` (60,000-row draw, `LogisticRegression(C=1, max_iter=300)` on
unit-normalised CLIP features; identity of its provenance with told_oracle.json arm L's head asserted, and A0's
arrays reproduced, as in step 1); the image and caption posteriors stored in
`src/test/20261108_new_method_quick_checks/results/n6_posteriors.npz` (SHA-256
2ad75026e5869c461fed156ffcdcf207c66f04a59dae514358522fbbaa9df2e0; keys `image__*`, `caption__*`); the csd and rand
CLIP heads stored in `src/test/20261116_grouping_step1_style/results/step1_heads_style.npz` (SHA-256
898a37017d82d3e130b20155e90f69e51f82f30892e04316dcb56d7aaaf8df8b; keys `style_csd__img/txt`, `style_rand__img/txt`;
the source-feature heads `__img_src` are not used).

**D3. Agreement and Δ.** For an image i and a caption t, the agreement on grouping h is a_h(i, t) = p_h(i) · p_h(t).
Under condition c, S_h^c is the mean agreement over the 4 support pairs, C_h^c the mean over the 4 contrast pairs,
and Δ_h^c = S_h^c − C_h^c (`src.eval.aspect_quick_checks.aspect_deltas`). Δ_h^b = −Δ_h^a exactly.

**D4. Grouping score.** s_h(q, k) = p_h(q) · p_h(k), the query's posterior from its own modality's head and the
candidate's from its own (`probe_dots`). It does not depend on the condition.

**D5. Reader term.** A hard reader picks one grouping π^c per episode and condition, and T^c = s_{π^c} (as
`inferred_scores(..., "hard", ...)` does for the current reader). R-b's expected scoring uses T^c = Σ_h P^c(h)·s_h.

**D6. Fused reader** (the fusion base is B, as in step 1): `crossfit_nested(B, B, T, parity)`. Every term is z-scored
per ranking row (`zscore_rows`); the score is (1 + λ_u)·z(B) + λ_a·z(T^c) with λ_u ∈ {0, 0.5, 1, 2, 4, 8, 16} and
λ_a ∈ {0, 0.25, 0.5, 1, 2, 4, 8, 16} (56 cells, ordered λ_u outer, λ_a inner, ascending). On each parity half the cell
that maximises min(R@1 − R@1 of B, condition gain) on that half is chosen (ties to the first cell) and scores the other
half. "R@1 of B" is the R@1 of the nested control (1 + σ)·z(B), which ranks exactly as B. R-c's fusion is in §4.3.

**D7. Matched counterpart.** T_cf = (T^a + T^b) / 2, identical under both conditions, computed for every reader as
`diagnose_counterparts.cf_version(T)` and fused by `crossfit_condition_free(B, B, T_cf, parity)`: the same 56 cells,
each half picks the cell with the highest R@1 (ties to the first) and scores the other half. For R-b expected this is
Σ_h P̄(h)·s_h with P̄ = (P^a + P^b) / 2 (equal up to floating-point rounding; `cf_version` is the one computed). R-c's counterpart is G_cf (§4.3). The counterpart keeps every ingredient and removes only the
condition.

**D8. B.** The stored condition-free score of step 1: `crossfit_condition_free(cos, T_N1u, T_6u, parity)` with T_6u
averaged over E2's three k-means-64 groupings (affect-km, image, caption), from the posteriors `affect__*`, `image__*`
and `caption__*` of `n6_posteriors.npz` (as `run_sweep.setup` builds it through `n6.n6_terms`). Seed 42: R@1 18.34
[17.97, 18.70], gain 0. On a test seed it is rebuilt with the same call on that seed's episodes and halves.

**D9. B′(configuration).** `crossfit_condition_free(cos, T_N1u, T_6u(config), parity)`, with T_6u(config) =
`uniform_probe_scores` over the configuration's own groupings. B′ is B rebuilt, not B plus a term; it depends on the
configuration only, never on the reader. Seed 42 (step 1): A0 18.44, A1 18.80, AR 18.45.

**D10. Comparators and the bar comparator.** Each candidate has three condition-free comparators: B, B′ of its
configuration, and its matched counterpart. Its **bar comparator** is whichever of the three has the largest mean R@1
over all episodes of the seed (full precision); ties go to the earliest in the order B′, counterpart, B. The same
comparator is used for every aspect pair.

**D11. Margins and the gain statistic.**
- *Margin* = fused reader minus matched counterpart, paired per anchor, R@1.
- *Bar margin* = fused reader minus bar comparator, paired per anchor, R@1, with its interval.
- *Gain statistic* = `fusedT_vs_fusedTcf.gain`: per-anchor condition gain of the fused reader minus that of its fused
  counterpart (the latter is 0 by construction), with its interval. Because B, B′ and the counterpart all have gain 0,
  this is the reader's gain over each of them; it is one number.

**D12. Development bar.** A candidate **clears the bar** if and only if all three hold on seed 42:
1. its bar margin's point estimate is at least +0.5 (full precision);
2. its bar margin's 95% lower bound is above 0;
3. its gain statistic's 95% lower bound is above 0.

Clauses 2 and 3 are kept for continuity with step 1 (no step-1 arm failed clause 3, and clause 2 is implied by clause 1
at seed-42 half-widths of 0.21 to 0.25). Basis of the +0.5: the two earlier fresh-seed tests roughly halved the
development effect, but both measured **condition gains**, not R@1 margins against a matched counterpart (E3, the
nested factor method: gain 0.52 on seed 42, 0.26 on its fresh seed; N1, the centred factor rule: 0.26 and 0.15).
N1's R@1 margin against its declared, non-matched control fell from +0.24 to +0.07 on fresh seeds (stage report
`docs/reports/stage/2026-10-04_aspect_conditioned_similarity_methods.md`). No reader margin against a matched
counterpart has yet been tested on fresh seeds, so the calibration is a heuristic.
Carrying the best of seven configurations inflates the development margin by roughly 0.1 to 0.15 R@1, inside the
halving the bar allows.

**D13. Told mapping and pick accuracy (diagnostic only; enters no rule).**
A0: emotion → affect, style → image, genre → image. A1: emotion → affect, style → csd, genre → image. AR: emotion →
affect, style → rand, genre → image. Pick accuracy = per episode, the mean over the two conditions of
1[picked grouping = told grouping], pooled over episodes, with its interval. Chance: 1/3 on A0, 1/4 on A1 and AR.
Reference values (the current arg-max reader, seed 42, fused on B, step 1): A0 54.7%, A1 43.2%, AR 41.9%.

**Reference rows on seed 42** (step 1, current arg-max reader; for orientation and for the regression check of §5,
item 2):

| Configuration | B′ | Counterpart R@1 | Fused reader R@1 | Margin | Bar margin (comparator) | Gain statistic |
|---|---|---|---|---|---|---|
| A0 | 18.437 | 18.396 | 18.750 | +0.354 [0.146, 0.566] | +0.313 [0.102, 0.528] (B′) | +1.337 [1.029, 1.649] |
| A1 | 18.805 | 18.750 | 18.813 | +0.063 [−0.154, 0.280] | +0.008 [−0.225, 0.243] (B′) | +1.296 [0.964, 1.630] |
| AR | 18.451 | 18.357 | 18.612 | +0.254 [0.095, 0.423] | +0.161 [−0.020, 0.348] (B′) | +0.712 [0.464, 0.967] |

External baselines on seed 42: cosine 12.96, RCA 13.38 (the strongest raw metric learned from the example pairs).

## 4. The reader candidates

### 4.1 R-a: scaled Δ

- **Noise scale.** For each grouping h ∈ {affect, image, caption, csd, rand}:
  σ_h = sqrt( mean over the 12,288 seed-42 episodes of (s²_S,h + s²_C,h) / 4 ),
  where s²_S,h and s²_C,h are the sample variances (ddof 1) of the four support-pair and the four contrast-pair
  agreements a_h on grouping h, from the standard heads. It is the standard error of Δ_h implied by pair-to-pair
  scatter; it is label-free and identical under both conditions (the two sets swap). The five σ_h are computed once,
  written to the results at full precision before any R-a score is computed, and reused unchanged on the test seeds.
  (The root mean square of Δ is not used: it adds mean squared signal to the scale.)
- **Pick.** π^c = arg max over the configuration's groupings of Δ_h^c / σ_h (ties to the first). T^c = s_{π^c}.
- **Top-two margin** (for R-c): m^c = the largest minus the second-largest Δ_h^c / σ_h.
- **Counterpart:** D7. **Runs on:** A1 and A0 (candidates) and AR (AR check, decides nothing).

### 4.2 R-b: the learned reader (frozen specification)

No R-b setting below changes after any seed-42 number of this folder has been read. A change would be a new candidate,
added only with the user's approval and reported as such beside the original. Correcting code so that it matches this
specification is not a change; such a correction is logged with the numbers before and after.

1. **Painting halves.** The 36,518 scorer-train paintings (sorted painting ids) are permuted with
   `numpy.random.default_rng(0).permutation`; half 0 is the first 18,259, half 1 the remaining 18,259. Each row follows
   its painting (about 92,000 rows per half). "Half k's rows" always means its scorer-train rows in ascending global
   row order; that order feeds both the head draw (item 2) and the bank (item 3).
2. **Cross-fitted heads.** For each half k ∈ {0, 1} and each grouping h of the configuration, an image head and a
   caption head with `fit_one_head`'s recipe (`LogisticRegression(C=1, max_iter=300)` on unit-normalised CLIP ViT-B/32
   features) on a 60,000-row draw from half k's rows (`default_rng(0).choice(rows, 60000, replace=False)`, as
   `fit_one_head`). Every group of h must appear in the draw, otherwise the step stops and goes to the user.
   Convergence warnings are counted and reported and change no setting. Reported for each head before any reader is
   trained, beside the standard heads' held-out accuracies (image / caption head: affect 9.81 / 35.72, image 92.7 /
   22.9, caption 21.6 / 89.7, csd 85.10 / 40.41, rand 15.64 / 13.20): the accuracy on 10,000 rows of half k outside
   the draw (`default_rng(1)`, as `fit_one_head`) and on 10,000 rows of half 1 − k (`default_rng(2)`). The heads of
   half k produce posteriors only on half 1 − k's rows. The affect, image and caption heads are shared by A1, A0 and AR.
3. **Banks.** For each half j, `src.train.pseudo_partitions.build_episode_bank(partitions, groups, rows,
   n_per_pair=16384, seed, min_paintings=30)` with: `partitions` a dict with exactly the keys `affect`, `image`,
   `caption`, `csd`, `rand` (those of the configuration), each a local scorer-train label array (index = position in
   the ascending scorer-train row list, as `src/test/20261105_method_repair_diagnostics/common.py` maps bank rows);
   `groups` = the painting id of each local row; `rows` = half j's local rows in ascending order; `seed` = 11700 for
   half 0 and 11800 for half 1. The builder sorts the keys and gives block i the seed `seed + i`; block i is bank
   episodes i·16,384 to (i + 1)·16,384 − 1. The blocks are:
   - A1: (affect, caption), (affect, csd), (affect, image), (caption, csd), (caption, image), (csd, image); no third
     grouping is controlled;
   - AR: (affect, caption), (affect, image), (affect, rand), (caption, image), (caption, rand), (image, rand); no third
     grouping is controlled;
   - A0: (affect, caption; third image), (affect, image; third caption), (caption, image; third affect), the third
     grouping controlled on the candidates.

   Sizes per half: A1 and AR 98,304 episodes, A0 49,152. Bank episodes use scorer-train rows only. If a block cannot be
   filled, the step stops and goes to the user.
4. **Features,** per episode, condition c and grouping h of the configuration, in configuration order (6 per grouping:
   24 on A1 and AR, 18 on A0): S_h^c, C_h^c, Δ_h^c, the sample standard deviation (ddof 1) of the four support-pair
   agreements, the same for the four contrast-pair agreements, and the share of the 4 support pairs whose image and
   caption arg-max groups coincide. Condition b's features are computed from the swapped supports and contrasts. Bank
   features use the posteriors of the heads of the other half; evaluation features use the standard heads (D2).
5. **Labels.** Each bank episode gives two training examples: condition a, labelled with the grouping its supports
   share (the block's first grouping, from the block's position), and condition b, labelled with the block's second
   grouping. The classes are balanced by construction. Class index = position of the grouping in configuration order.
6. **Model.** Per half j: one `StandardScaler` fitted once on all of half j's bank features (both conditions); a
   multinomial `LogisticRegression` (lbfgs, `max_iter=2000`, no class weights) with C ∈ {0.01, 0.1, 1, 10, 100}
   chosen by five-fold cross-validation on half j's bank episodes only (folds by episode, so both conditions of an
   episode share a fold; `KFold(5, shuffle=True, random_state=0)` over episode indices), criterion the mean over the
   five folds of `sklearn.metrics.log_loss` on the held-out fold, ties to the smaller C; then refit on all of half j's
   bank at the chosen C. Convergence warnings are counted and reported. The two half-readers' probabilities are
   averaged: P^c(h) = mean over j of half-reader j's probability on the evaluation features.
7. **Scorings.** *R-b arg-max:* T^c = s_{π^c}. *R-b expected:* T^c = Σ_h P^c(h)·s_h. Both scorings have the pick
   π^c = arg max_h P^c(h) (ties to the first grouping in configuration order); it is used for pick accuracy and for the
   AR check's share of picks that go to rand. Counterparts: D7. Top-two margin (for R-c): the largest minus the
   second-largest P^c(h).
8. **Runs on:** A1 and A0 (candidates, each with its own bank and readers) and AR (AR check, decides nothing).
9. **Not used:** an EM correction of R-b's class prior.
10. **Disclosed limits.** The bank's classes are groupings with a uniform prior and supports that share a group
    exactly; genre is not a class, and the caption grouping is a class although no aspect maps to it under the told
    mapping. R-b can learn that csd and image agreeing together signals genre only as an agreement pattern the bank
    happens to contain.
11. **R-b diagnostics (letter R4; reported for every R-b configuration, enter no rule):** (a) bank accuracy: the
    out-of-fold accuracy of each half-reader at its chosen C; (b) seed-42 pick accuracy (D13); (c) the label-free shift
    report: for each input feature, the standardised mean difference (mean on seed 42 − mean on the bank) /
    sqrt((variance on seed 42 + variance on the bank) / 2), seed-42 features from the standard heads (both conditions)
    against bank features from both halves (both conditions); and the distribution (mean and deciles) of R-b's top
    probability max_h P(h) on the bank (each half-reader's out-of-fold probabilities at its chosen C, pooled over both
    halves) and on seed 42. **There is no kill for R-b:** it survives or falls
    on the development bar like every other candidate.

### 4.3 R-c: the confidence gate

- **Parent.** The candidate among R-a/A1, R-b arg-max/A1, R-b expected/A1, R-a/A0, R-b arg-max/A0 and R-b expected/A0
  with the largest bar margin (full precision), **whether or not it clears the bar**; exact ties go to the earliest in
  that list. R-c is built once, when all six have development numbers or at Thursday 8 October 09:00, whichever comes
  first, on the completed ones. If none has numbers by then, R-c is not built. R-c belongs to its parent's
  configuration.
- **Margin.** m^c, the parent's top-two margin under condition c: scaled Δ for R-a (§4.1), probabilities for R-b
  (§4.2, item 7).
- **Threshold grid.** τ_0, τ_1, τ_2, τ_3 = the 0th, 25th, 50th and 75th percentiles (`numpy.percentile`, linear) of
  the parent's m^c over the 24,576 seed-42 (episode, condition) values. τ_0 is the seed-42 minimum, so on seed 42 its
  gate is always open and gives the parent itself (on a test seed a margin below τ_0 closes the gate). The four values
  are written at full precision before any R-c score is computed and reused unchanged on the test seeds.
- **Gate.** g^c = 1 if m^c ≥ τ, else 0, per episode and condition, the same for both directions.
- **Fused score.** (1 + λ_u)·z(B) + λ_a·g^c·z(T^c), computed in `aspect_nested._combine`'s order
  z(B) + λ_u·z(B) + λ_a·(g^c·z(T^c)) (terms with weight 0 left out): z-scoring per ranking row first, the gate
  multiplies after. 224
  cells = 4 thresholds × the 56 (λ_u, λ_a) cells of D6, ordered τ outer, then λ_u, then λ_a. Cross-fit: on each parity
  half the cell that maximises min(R@1 − R@1 of B, condition gain) on that half (ties to the first) scores the other
  half.
- **Matched counterpart.** G_cf = (g^a·z(T^a) + g^b·z(T^b)) / 2, the two-condition mean of the whole gated term,
  identical under both conditions (asserted in code, as `crossfit_condition_free` asserts). Fused as
  (1 + λ_u)·z(B) + λ_a·G_cf (G_cf is not z-scored again) on the same 224 cells; each half picks the cell with the
  highest R@1 (ties to the first) and scores the other half. The product ḡ·z(T_cf) is not used: it drops the
  covariance of gate and term and removes more than the condition.
- **Sanity report** (decides nothing): restricted to the τ_0 cells, R-c's fused score equals the parent's fused score
  cell by cell, and its counterpart differs from the parent's only by z-scoring before rather than after the
  two-condition average. Both reported.
- Also reported: the share of open gates at each τ (label-free) and R-c's pick accuracy, which is its parent's (S1).

## 5. Development selection on seed 42 (items 1 to 5)

**Item 1. Candidates.** Seven candidates: R-a/A1, R-b arg-max/A1, R-b expected/A1, R-a/A0, R-b arg-max/A0,
R-b expected/A0, and R-c on its parent's configuration. A0 runs are candidates under the A1 priority of item 4. If an
A0 candidate is carried, a GO supports the reader fix without the CSD grouping and the CSD question stays open; by
the user's ruling it counts as a GO for plan (a), reported as a reader-fix GO that leaves the CSD question open. AR runs
are never candidates.

**Item 2. Measures,** reported for every candidate; only item 3 decides.
- Fused reader against its matched counterpart (R@1, gain, either), against B and against B′; the bar margin and its
  comparator; the gain statistic; all per aspect pair as well (same comparator).
- Paired differences against the same configuration's current arg-max reader (step 1): R@1 margin and bar margin.
- **A1 − A0 under each reader** (R-a, R-b arg-max, R-b expected): the paired per-anchor difference in fused R@1 and in
  bar margin, with intervals; descriptive.
- Diagnostics (D13; enter no rule): pick accuracy overall and per aspect pair and condition, beside the reference
  values; R-b's diagnostics (§4.2, item 11); R-c's sanity report and gate shares (§4.3).
- **AR check** (the controller's addition; decides nothing): R-a, R-b arg-max and R-b expected on AR, with the current
  arg-max reader as the reference: the share of (episode, condition) picks that go to rand, overall and per aspect pair
  and condition, and each reader's margin and bar margin on AR. A reader whose scaling works should rarely pick rand.
- **Regression check** before any candidate number: the evaluation code reproduces step 1's stored arg-max reader
  arrays for A0, A1 and AR (`step1_eval_style.npz`, SHA-256
  8d10a0fbbd34212c73849239faed9d0a07372e68732dc452bf2fd57f7409c68c) exactly, together with B and the three B′ values.
  If it does not, the work stops and goes to the user.

**Item 3. Development bar:** D12.

**Item 4. Carry.** Let E be the set of candidates that clear the bar.
- If E contains an A1 candidate, the pool is the A1 members of E; otherwise the pool is E.
- Let M be the largest bar margin in the pool. Every pool member whose bar margin is at least M − 0.05 is tied.
- The carried candidate is the tied member that comes first in the order R-a, R-b arg-max, R-b expected, R-c.

**Item 5. Kill.** If no candidate clears the development bar, no test is built and the result goes to the user for the
Friday choice. Pick accuracy under the told mapping, R-b's bank accuracy and R-b's shift report are seed-42 diagnostics,
reported for every candidate, and enter no rule. Pick accuracy is not computed on test seeds until the GO verdict is
recorded.

## 6. The test (item 6), the only confirmatory step

1. **Sensitivity, before the seeds are built.** For each GO check of item 6.5, take the carried candidate's
   seed-42 per-episode difference (the fused reader's R@1 minus the comparator's, or the gain difference) and split its
   variance by anchor painting with the one-way decomposition: σ_ε² = the within-painting mean square, σ_a² =
   max(0, (between-painting mean square − σ_ε²) / n₀) with n₀ = (n − Σ_p m_p² / n) / (P − 1), where n = 12,288
   episodes, P the number of anchor paintings and m_p the episodes per painting on seed 42. The projected pooled
   standard error over seeds 49 to 51 is SE² = (σ_a²·(9·Σ_p m_p² − 6n) + σ_ε²·3n) / (3n)² (three independent draws of
   the same anchor distribution, counts approximated as Poisson). Written to the log: SE, the projected half-width
   1.96·SE, the **detectable margin x = 2.80·SE** (the true margin at which one check's lower bound clears 0 with
   probability 0.8), and the seed-42 bootstrap half-width beside them.
2. **Build.** Seeds 49, 50 and 51, once each, with `src/test/20261030_aspect_baselines/run_baselines.py
   --episodes-seed <s>` (4,096 episodes per aspect pair on selection rows; it also scores cosine and RCA). The 9 new
   per-pair episode SHA-256s must differ from each other and from every per-pair SHA-256 recorded in
   `src/test/20261030_aspect_baselines/results/baselines_seed{42,43,45,47,48}.json`; on a match the test stops and goes
   to the user. `docs/superpowers/episode_seed_ledger.md` gets a row.
3. **What is frozen from seed 42:** the carried candidate (its reader and configuration), the standard heads, σ_h (R-a), the two
   half-readers with their scalers (R-b), τ_0 to τ_3 (R-c), the recipes of B and B′, and the method-A checkpoint.
   **What is rerun on each seed's own parity halves:** B, B′, the fused reader and its matched counterpart. Per-seed
   cross-fitting is part of the method's definition; the intervals hold the cross-fit picks fixed.
4. **Order of computation.** On the test seeds only the GO quantities are computed until the verdict is written to
   `results/test_verdict.json` (with this file's SHA-256 and the Amsterdam time): the per-anchor arrays of the fused
   reader, its counterpart, B, B′, cosine and RCA on each seed, and the pooled differences of item 6.5. The per-seed
   cosine and RCA summaries that `run_baselines.py` writes when it builds a seed are allowed and are not read before
   the verdict. Pick accuracy, per-seed and per-pair numbers of the candidate, the R-b diagnostics and the frozen-cell
   line come only after the verdict.
5. **GO** if, pooled over the three seeds (36,864 episodes; one cluster per anchor painting across seeds; 5,000
   resamples, seed 42), every one of these seven checks has a 95% lower bound above 0:
   - R@1, fused reader minus each of cosine, RCA, B, B′ and the matched counterpart (five checks);
   - the gain statistic (D11), which is also the gain difference against cosine, B and B′, counted once;
   - condition gain, fused reader minus RCA.
6. **NO-GO** if any check fails, including a partial pass. Its reading: if every failed check has a pooled point
   estimate above 0, the result is reported as **inconclusive at a detectable margin of x** (with x of each failed
   check, from item 6.1), not as evidence that the reader fails; if any failed check has a point estimate at or below
   0, it is reported as "the carried candidate did not beat <comparator> on fresh episodes", naming each such check.
7. **Per-seed and per-pair results** are reported and never change the pooled verdict; per-pair results are not tested.
8. **Frozen-cell line** (descriptive, after the verdict, not a second verdict): the same comparisons with the seed-42
   cross-fit cells applied unchanged on each test seed.
9. **Claim licensed.** A GO shows that the carried candidate beats each comparator, pooled over the three aspect
   pairs, on new episodes drawn from the same 6,451 selection paintings. It does not show transfer to new paintings
   (the held split, 12,281 paintings, stays reserved for the paper test) or a margin on each aspect pair. For the
   paper: the groupings were built without evaluation labels, and the label-free diagnostics also rank CSD above the
   VGG-Gram alternative (placeability P_ami 0.208 against 0.150, Leiden-seed stability 0.807 against 0.673); the
   decision to continue with CSD was taken after its told margins had been read, and this is disclosed.

## 7. Order of work, cutoff and timeline

- **Order of computation:** (1) the regression check, the five σ_h and R-a on A1, A0 and AR; (2) R-b on A1: halves,
  cross-fitted heads (accuracies reported), banks, readers, both scorings; (3) R-b on A0; (4) R-c at its trigger
  (§4.3); (5) R-b on AR. The 09:00 trigger of §4.3 carries out the user's fallback order (R-a and R-c on R-a first,
  then R-b arg-max on A1, the rest last): if R-b is late, R-c is built on the best completed candidate, which is at
  least an R-a.
- **Cutoff: Thursday 8 October 12:00.** A candidate "has development numbers" when its bar margin and its gain
  statistic, each with its interval, are written to its results file. Candidates without development numbers by the
  cutoff are dropped from this round. The rule is applied at the cutoff, or earlier as soon as all seven candidates
  have numbers. Before it is applied, the controller re-derives, with independent code, the bar margin and gain
  statistic of every candidate finished after Wednesday's re-derivation. An AR run without numbers by the cutoff is
  reported as not run.
- If the test cannot be completed before the Friday decision, no partial verdict is issued; the user is told and
  decides.

| Day (Amsterdam) | Work |
|---|---|
| Tue 6 Oct | This rule written, checked by a fresh reviewer against the letter (findings applied), committed (authorised by the user at 02:25) and sent to the user; regression check, σ_h, R-a and its AR check; R-b's halves, cross-fitted heads and banks start |
| Wed 7 Oct | R-b readers and evaluation on A1, then A0; the controller re-derives every load-bearing development number with independent code; seed-42 results sent to the user |
| Thu 8 Oct | R-c by 09:00 at the latest; cutoff 12:00; the rule is applied; if a candidate is carried: the sensitivity projection (§6.1), then seeds 49 to 51 are built and the test is run to its verdict. Evening: the controller re-derives the test's GO quantities with independent code |
| Fri 9 Oct | Morning: whole-branch final review on the most capable model (re-derives every load-bearing number; one fix wave and a scoped re-review), then the report; the user decides GO or NO-GO and what follows |

## 8. Outcome-to-action table

| Outcome | Action |
|---|---|
| A script check fails (a SHA-256, the regression check of item 2, a condition-free assertion, a bank block that cannot be filled) | Stop that step and report to the user; nothing is improvised |
| A candidate has no development numbers at the cutoff | Dropped from this round; reported as such |
| R-b's bank accuracy, pick accuracy or shift report look good or bad | Reported only; no action. R-b stands or falls on the bar (there is no R-b kill, so "R-b killed after clearing the bar" cannot occur) |
| No candidate clears the bar | No test is built; the seed-42 results go to the user, who chooses on Friday among: design L (late-fusion refinement of the groupings), a change of course (benchmark or analysis paper), or another reader round under a new pre-registered rule |
| Exactly one candidate clears the bar | It is carried to the test (§6) |
| Several candidates clear the bar | Item 4: A1 priority, then the largest bar margin, ties within 0.05 to the earliest of R-a, R-b arg-max, R-b expected, R-c |
| On the test seeds, before the verdict | Only the GO quantities (§6.4) |
| Test: all seven checks pass, carried candidate on A1 | **GO**: a reader-fix GO with the CSD grouping, within the claim of §6.9 |
| Test: all seven checks pass, carried candidate on A0 | **GO** for plan (a): a reader-fix GO that leaves the CSD question open, within §6.9 |
| Test: any check fails, including a partial pass | **NO-GO**, read by §6.6 (inconclusive at x, or not beaten); the user chooses among the options of the "no candidate" row |
| The controller's re-derivation or the final review disagrees with a reported number or verdict | The number re-derived from the data settles it; the corrected verdict goes to the user with the cause; the rule is not changed |
| Any situation this file does not cover | The user decides; this file is not changed after its commit without the user |

## 9. Process

- Scripts of this folder assert this file's SHA-256 (as step 1 asserted its PLAN.md), refuse to overwrite non-smoke
  results, and write to `results/` (gitignored). CPU only: `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8`,
  at most 3 processes, `uptime` and `free -g` checked first.
- Records: the log `20261117_reader_fix_csd_log.md` in this folder; a report under `docs/reports/auto/v2/` with a row in
  `docs/reports/reports_sum.md`; `.claude/<yyyymmdd>_log.md` for any edit to an existing source file; the seed ledger.
