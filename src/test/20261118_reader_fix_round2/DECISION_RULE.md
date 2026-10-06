# Decision rule: reader fix, round 2 (follow-ups of the confidence-gated reader), committed before any code or number

**Written** 2026-10-06 15:35 (Amsterdam), from the approved spec
`docs/superpowers/specs/2026-10-06-reader-fix-round2-design.md` (commit 7659034, SHA-256
ceecc515beac5402156b1c4ee613e55308c4862fb1d8da6963f07c64301e11c8), before any script of this folder exists and before
any number it governs has been computed. The only numbers it states are round 1's, which it reuses.

**Precedence.** Where this file differs from the spec or from round 1's rule
(`src/test/20261117_reader_fix_csd/DECISION_RULE.md`, SHA-256
613d8c9d13fb7d4787ec0fb91dc65cfd4a192d07c88ab8f961da3adc1c35d23c), **this file governs**. Everything needed to apply
the rule is written here; round 1's rule and the spec are cited for provenance only.

**Checked** before its commit by a fresh Opus reviewer, which the user chose in place of an ARS round (spec §1); its
findings are applied before the commit. The check (report `rule_check/opus_rule_check.md`) found 2 blocking, 7 should-fix
and 9 nit findings, all applied before the commit.

**Status.** Items 1 to 5 (§5) are development selection on episode seed 42, which has been read many times, by round 1's
seven candidates among others; their numbers are exploratory. Item 6 (§6), the fresh-seed test on seeds 49, 50 and 51,
is the only confirmatory step.

**Authorisation.** The user's decisions of 2026-10-06 (spec §1): test both improvement families (better reader
probabilities and a top-k restriction) with the confidence gate built into every candidate; develop without the style
grouping (configuration A0) and keep it (A1) as a descriptive ablation; a fresh Opus check of this rule instead of an
ARS round; design L later; results as early as possible (today if the runs allow). The user approved the design section
by section in chat and allowed this file, and the work under it, to be committed to `main` without asking. After its
commit this file changes only with the user's approval.

**Dates.** Folder and report dates (`20261118`, round 1's `20261117` and `2026-11-18`) are sequence numbers, not calendar
dates. Calendar times in this file are Amsterdam local time.

**Prior.** Our own estimate of the chance of a GO is moderate at best, as in round 1: round 1's best candidate fell
0.056 R@1 short of the development bar on seed 42, every fresh-seed test so far roughly halved the development effect,
and this round's selection inflation is larger than round 1's (D12).

## 1. Glossary

| Term | Meaning in this file |
|---|---|
| episode | a query (the anchor row: its image for image-to-caption ranking, its caption for caption-to-image ranking), 4 support pairs, 4 contrast pairs and 13 candidates in the other modality; p_A (candidate column 0) shares the query's value of aspect A, p_B (column 1) its value of aspect B, 11 negatives (columns 2 to 12) share neither |
| per-anchor | per episode (one value per episode, averaged over its 4 rankings) |
| ranking row | one (episode, condition, direction): the 13 candidate scores that one ranking orders |
| configuration | the groupings a reader chooses among: **A0** = (affect, image, caption), the only candidate configuration; **A1** = A0 plus csd, used only in the ablation of §4.8 |
| candidate | one of the three readers R1, R2, R3 on A0, each with the fusion family of §4.5; the **carried candidate** is the one item 4 sends to the test |
| condition a / b | a: the supports show aspect A and the target is p_A; b: supports and contrasts swapped, target p_B. Each episode gives 4 rankings (2 conditions × 2 directions) |
| grouping | a label-free partition of the scorer-train rows (D1); fixed, none re-chosen |
| head | a logistic regression on frozen CLIP ViT-B/32 features predicting a row's group of one grouping, from the image alone (image head) or the caption alone (caption head) |
| p_h(x), Δ_h, s_h | head posterior of item x on grouping h; support minus contrast agreement (D3); grouping score (D4) |
| half-reader | one of the two multinomial logistic regressions of a learned reader, each trained on the bank of one painting half of the scorer-train rows (round 1 §4.2) |
| P^c(h) | the reader's probability, under condition c, that grouping h is the one the supports share; the mean over the two half-readers (for R2 after its adaptation, P′) |
| T^c | the weighted reader term Σ_h P^c(h)·s_h (D5) |
| m^c | the top-two margin: the largest minus the second-largest P^c(h) |
| τ, gate g^c | a threshold on m^c (§4.5 item 2) and g^c = 1[m^c ≥ τ] |
| k_top, K | the top-k restriction's size, k_top ∈ {13, 5, 3, 2}, and the top-k set: B's k_top highest-scoring candidates of a ranking row (§4.5 item 5) |
| cell | one (k_top, τ index, λ_u, λ_a) of the 896 of §4.5 item 7 |
| G_cf | the two-condition mean of the gated term, the matched counterpart's term (§4.6) |
| B, T_N1u, T_6u, B′ | the best condition-free score of the project and its parts (D8), and B rebuilt on a configuration's own groupings (D9) |
| round-1 R-c | round 1's confidence-gated reader on the expected term of its learned reader on A0 (`cand_Rc_Rb_expected_A0`): bar margin +0.4435 against its counterpart, the best of round 1 |
| bank | round 1's pseudo-aspect episodes built from the groupings on one painting half of the scorer-train rows, whose supports share a group of one grouping and whose contrasts share a group of another |
| purity k, impure bank | R3's number of support pairs and of contrast pairs per bank episode that keep their shared group, k ∈ {1, 2, 3, 4}; an impure bank is a bank with k < 4 (§4.4) |
| π_train, π̂ | R2's training class prior (uniform) and the class prior its EM correction estimates on seed 42 (§4.3) |
| μ42, σ42 | R2's per-feature mean and standard deviation of the seed-42 reader inputs (§4.3) |
| told mapping | the evaluation-label map aspect → grouping (D13), used only for the diagnostic pick accuracy |
| parity halves | the two cross-fit halves of a seed's episodes: episode index parity |

## 2. Data, metrics and intervals

- **Development episodes:** `src/test/20261030_aspect_baselines/results/episodes_seed42.npz` (SHA-256
  12af979432ff1a20c88b614ab9e672f203eeeed9c2465bff305e72d01e6c0986): 12,288 episodes (4,096 per aspect pair:
  emotion × style, emotion × genre, style × genre) on 4,602 anchor paintings, selection rows, loaded through the
  step-1 context (`src/test/20261112_community_sweep/run_sweep.py::setup`, as round 1's `common.load_bundle` calls it).
- **Metrics,** per episode averaged over both conditions and both directions, pooled over the three aspect pairs, in
  percentage points: R@1 (the target ranks strictly first; ties miss), other-aspect rate, condition gain = R@1 − other,
  either rate = R@1 + other. A condition-free scorer has gain exactly 0 on every episode.
- **Intervals:** 95% percentile intervals of the bootstrap that resamples anchor paintings (5,000 resamples, seed 42;
  `src.eval.aspect_metrics.cluster_bootstrap`, as round 1's `common.point_ci` calls it). Cross-fit picks are fixed
  before resampling. "A lower bound above 0" means strictly greater than 0.
- **Precision:** every threshold in this file applies to the full-precision point or bound, never to a rounded value.
- Held rows are not read. Nothing in this file uses the GPU.

## 3. Definitions (the only definitions; every item below refers to them)

**D1. Groupings and configurations.** Four groupings, none re-chosen:
- *affect*: Leiden communities on GoEmotions caption probabilities, 41 groups (`partition_L` in
  `src/test/20261111_community_told_oracle/results/per_anchor_told_oracle.npz`, SHA-256
  27e8101161607da2d6936aa5a8c84f8c296629fe08e37a68c8418c9356af1366);
- *image* and *caption*: k-means with 64 clusters on CLIP image or caption features (E2;
  `src/test/20261031_pseudo_partitions/results/partitions.npz`, SHA-256
  cfd57dbb21216cdc9d1c2e42d1b408648bc18147f62354b52202ade4ab8a9caa);
- *csd* (A1 ablation only): Leiden communities on CSD style embeddings of the painting images, 17 groups (`style_csd` in
  `src/test/20261116_grouping_step1_style/results/step1_group_style.npz`, SHA-256
  b04d96b4798acdfdbc9ca87436012755a681c931ac4f7612fa5683350b7a20e2).

E2's own affect grouping (k-means 64 on GoEmotions probabilities, "affect-km") is used only inside B (D8). Round 1's
random grouping (rand) and its configuration AR are not used in this round.

Configurations, in this order (the order breaks every arg-max tie: the first grouping wins):
A0 = (affect, image, caption); A1 = (affect, image, caption, csd).

**D2. Standard heads** (used for every evaluation, on seed 42 and on the test seeds), exactly as in round 1 and step 1:
the affect heads refit with `run_told_oracle.fit_one_head` (60,000-row draw, `LogisticRegression(C=1, max_iter=300)` on
unit-normalised CLIP features; identity with told_oracle.json arm L's head asserted by `common.load_bundle`); the image
and caption posteriors stored in `src/test/20261108_new_method_quick_checks/results/n6_posteriors.npz` (SHA-256
2ad75026e5869c461fed156ffcdcf207c66f04a59dae514358522fbbaa9df2e0; keys `image__*`, `caption__*`); for the A1 ablation,
the csd CLIP heads stored in `src/test/20261116_grouping_step1_style/results/step1_heads_style.npz` (SHA-256
898a37017d82d3e130b20155e90f69e51f82f30892e04316dcb56d7aaaf8df8b; keys `style_csd__img/txt`).

**D3. Agreement and Δ.** For an image i and a caption t, the agreement on grouping h is a_h(i, t) = p_h(i) · p_h(t).
Under condition c, S_h^c is the mean agreement over the 4 support pairs, C_h^c the mean over the 4 contrast pairs,
and Δ_h^c = S_h^c − C_h^c (`src.eval.aspect_quick_checks.aspect_deltas`). Δ_h^b = −Δ_h^a exactly.

**D4. Grouping score.** s_h(q, k) = p_h(q) · p_h(k), the query's posterior from its own modality's head and the
candidate's from its own (`common.grouping_stack`). It does not depend on the condition.

**D5. Reader term.** Every candidate uses the weighted scoring T^c = Σ_h P^c(h)·s_h over its configuration's groupings
(round 1's "R-b expected", `common.expected_term`). Scoring by the top pick alone is dropped: on A0 in round 1 it gave a
bar margin of +0.144 against the weighted scoring's +0.313. The pick π^c = arg max_h P^c(h) (ties to the first grouping
in configuration order) is used only for pick accuracy (D13). The top-two margin m^c is the largest minus the
second-largest P^c(h).

**D6. Fusion base.** Every term is z-scored per ranking row over its 13 candidates (`zscore_rows`: population standard
deviation; a constant row becomes zeros), through `aspect_nested._zdict`. The 56 weight cells are (λ_u, λ_a) with
λ_u ∈ NESTED_U = (0, 0.5, 1, 2, 4, 8, 16) and λ_a ∈ NESTED_A = (0, 0.25, 0.5, 1, 2, 4, 8, 16), λ_u outer, λ_a inner,
ascending (`aspect_nested.nested_cells`). The nested control is (1 + σ)·z(B) with σ among the 30 distinct sums λ_u + λ_a
(`aspect_nested.control_sums`, ascending: 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.25, 2.5, 3, 4, 4.25, 4.5, 5, 6, 8, 8.25,
8.5, 9, 10, 12, 16, 16.25, 16.5, 17, 18, 20, 24, 32); it ranks exactly as B. The round-2 fusion family (§4.5) adds the gate and the
top-k restriction to this base.

**D7. Matched counterpart.** The counterpart keeps every ingredient of the fused reader and removes only the condition:
the gated term is replaced by its two-condition mean G_cf (§4.6), fused on the same cells, and each cross-fit half picks
the cell with the highest R@1 (the most favourable rule for a control).

**D8. B.** The stored condition-free score of step 1: `crossfit_condition_free(cos, T_N1u, T_6u, parity)` with T_N1u the
centred uniform factor term of the method-A checkpoint
(`src/test/20261101_aspect_factor_gonogo/checkpoints/A3_seed42.pt`, SHA-256
dadfef1bed95bbd647c39fa0ea483edea78627d440a23b5a032083e76dd469e2) and T_6u averaged over E2's three k-means-64
groupings (affect-km, image, caption), from the posteriors `affect__*`, `image__*` and `caption__*` of
`n6_posteriors.npz` (as `run_sweep.setup` builds it). Seed 42: R@1 18.341 [17.975, 18.697] (full precision
18.341064453125). On a test seed it is rebuilt with the same call on that seed's episodes and halves.

**D9. B′(configuration).** `crossfit_condition_free(cos, T_N1u, T_6u(config), parity)`, with T_6u(config) =
`uniform_probe_scores` over the configuration's own groupings. B′ is B rebuilt, not B plus a term; it depends on the
configuration only, never on the reader. Seed 42 (step 1): A0 18.437 (18.436686197916664), A1 18.805
(18.804931640625).

**D10. Comparators and the bar comparator.** Each candidate has three condition-free comparators: B, B′(A0) and its
matched counterpart. Its **bar comparator** is whichever of the three has the largest mean R@1 over all episodes of the
seed (full precision); ties go to the earliest in the order B′, counterpart, B (`common.bar_comparator`). The same
comparator is used for every aspect pair.

**D11. Margins and the gain statistic.**
- *Margin* = fused reader minus matched counterpart, paired per anchor, R@1.
- *Bar margin* = fused reader minus bar comparator, paired per anchor, R@1, with its interval.
- *Gain statistic* = per-anchor condition gain of the fused reader minus that of its fused counterpart (the latter is 0
  by construction), with its interval (`common.evaluate_fused`, key `gain_statistic`). Because B, B′ and the
  counterpart all have gain 0, this is the reader's gain over each of them; it is one number.

**D12. Development bar.** A candidate **clears the bar** if and only if all three hold on seed 42:
1. its bar margin's point estimate is at least +0.5 (full precision);
2. its bar margin's 95% lower bound is above 0;
3. its gain statistic's 95% lower bound is above 0.

The bar is round 1's, kept unchanged by the user's decision although it is heuristic (spec §3). Its basis: the two
earlier fresh-seed tests roughly halved the development effect, but both measured condition gains, not R@1 margins
against a matched counterpart (E3: gain 0.52 on seed 42, 0.26 on its fresh seed; N1: 0.26 and 0.15); N1's R@1 margin
against its declared, non-matched control fell from +0.24 to +0.07. Round 1 built no test, so no reader margin against a
matched counterpart has yet been measured on fresh seeds. Round 1 estimated the inflation from carrying the best of seven
configurations at roughly 0.1 to 0.15 R@1. This round's inflation is larger: its three candidates were designed after
seeing round 1's seed-42 results for seven candidates (round-1 R-c's +0.4435 among them), and each candidate's cross-fit
chooses among 896 cells instead of 224. We do not estimate the amount; the fresh-seed test is the protection (§6.10).

**D13. Told mapping and pick accuracy (diagnostic only; enters no rule).**
A0: emotion → affect, style → image, genre → image. A1: emotion → affect, style → csd, genre → image. Pick accuracy =
per episode, the mean over the two conditions of 1[π^c = told grouping], pooled over episodes, with its interval
(`common.pick_statistics`). Chance: 1/3 on A0, 1/4 on A1. Round 1 values on seed 42: the step-1 arg-max reader 54.7%
(A0) and 43.2% (A1); round 1's learned reader 51.3% (A0, 51.261393229166664) and 48.7% (A1). R1's pick accuracy is
round 1's learned reader's.

**Reference rows on seed 42** (round 1; for orientation and for the regression check of §4.7):

| Reader (round 1) | Config | B′ | Counterpart R@1 | Fused R@1 | Margin | Bar margin (comparator) | Gain statistic |
|---|---|---|---|---|---|---|---|
| step-1 arg-max reader | A0 | 18.437 | 18.396 | 18.750 | +0.354 [0.146, 0.566] | +0.313 [0.102, 0.528] (B′) | +1.337 [1.029, 1.649] |
| learned reader, weighted term | A0 | 18.437 | 18.365 | 18.750 | +0.385 [0.152, 0.624] | +0.313 [0.076, 0.550] (B′) | +2.112 [1.815, 2.422] |
| round-1 R-c | A0 | 18.437 | 18.475 | 18.919 | +0.444 [0.216, 0.674] | +0.4435 [0.216, 0.674] (counterpart) | +2.667 [2.325, 3.012] |
| learned reader, weighted term | A1 | 18.805 | 18.872 | 18.970 | +0.098 [−0.059, 0.253] | +0.098 [−0.059, 0.253] (counterpart) | +0.535 [0.336, 0.729] |

Round-1 R-c at full precision: fused 18.918863932291664, counterpart 18.475341796875, bar margin 0.4435221354166667
[0.21646171563312194, 0.6735669710776852] against the counterpart, gain statistic 2.667236328125 [2.325087836946873,
3.012361650695922]. The told ceiling on A0 is +1.64 [1.37, 1.92] over its matched counterpart. External baselines on
seed 42: cosine 12.96, RCA 13.38 (the strongest raw metric learned from the example pairs).

**D14. Round-1 inputs.** Read only; never written or modified. Paths are under `src/test/20261117_reader_fix_csd/`.
Scripts assert every SHA-256 below before using the file; a mismatch stops the run.

| File | SHA-256 | Used for |
|---|---|---|
| `DECISION_RULE.md` | 613d8c9d13fb7d4787ec0fb91dc65cfd4a192d07c88ab8f961da3adc1c35d23c | asserted by `common.load_bundle` |
| `common.py` | 99496dcf859ff0b0f746266125c975c5e2f9632e0c2c91d74df17102c65c10a7 | bundle, B, B′, terms, evaluation (imported) |
| `rc_core.py` | e649fff5d8253c8ba7aae0dbfff7a68321cef5fe3caa06d7b325a27fe904185c | gate, G_cf, cross-fit pieces (imported) |
| `rb_build.py` | 63c7310c890675cc63e49838feb364cdd1c5c68aafdfb0f0f7972128476cba7b | reader loading and training recipe (imported) |
| `rb_eval.py` | 1826e65fd73b8c6cecae33b5df48503bf2cf33e067fc291d5c078c51184e0b14 | seed-42 features and half-reader probabilities (imported) |
| `rb_features.py` | 1e758f6d5de1b1dd246592f4045e0ae1dcc520c1d6012540a9e69b3314fb6caf | features, labels, folds, SMD (imported) |
| `results/rb_reader_A0.pkl` | 6387b469662446734e650571c88a152651da3d66bcade28fb6cdd7828f58d34c | the two A0 half-readers with their scalers (R1, R2) |
| `results/rb_reader_A0.json` | cd90da922967cf1b62e437d3927c97cd9cfdae2c139ee392744538d3e3e5c6a8 | the pickle's record |
| `results/rb_reader_A0.npz` | 41ef7f5d9c8c0f0c91d2bf36f2f5d79ae4ea1b7b2fbb3ba89e7febb9404cc4d0 | round 1's bank features `half{j}__X` (R3's purity-4 check) |
| `results/rb_bank_A0_half0.npz` | f0711629d18d795f357368ac080c9f816e480cec44c3ec0d057e13eac9e8a1e2 | bank of half 0 (R3) |
| `results/rb_bank_A0_half0.json` | 7ba7b49a0f7c6344949a92b3aa3cdcf2fe6df796c4d775beb9fff75315331347 | its record |
| `results/rb_bank_A0_half1.npz` | edd48914d8db50e492e21ad725619464a655098f77651e7226c200c86db8ca80 | bank of half 1 (R3) |
| `results/rb_bank_A0_half1.json` | 7d12804ae7a9835648e6a0c523fcb5495b2c83ceaedc34bd11aed9869641c50d | its record |
| `results/rb_halves.npz` | ed2d359e6517b4237f8aa4de5e78f787699b366a975ac993eaef0e7c4491c7da | painting halves, half rows, painting of each local row (R3) |
| `results/rb_halves.json` | cbdecbd9e33807701791ec4df20da8967edcfa6090133e056ab3392fb10676b2 | its record |
| `results/rb_heads_affect.npz` | e21e846be8b9e565a7443c5e7ecd4135a04def709ce64bef6dceae6545823324 | cross-fitted affect posteriors (R3) |
| `results/rb_heads_affect.json` | 9492b4cbb3e07961c8d4dbc5c0c1e143ba479be261602b1436b6d844af53dda1 | its record |
| `results/rb_heads_image.npz` | 3743b3fe9c8942cddf399b8ffc7ef47806d8d2a8d4a6ec8ea474473010f3c46f | cross-fitted image posteriors (R3) |
| `results/rb_heads_image.json` | 8888c2904b48e2cd14f6efd14e9b172d49c1267243c537be8e9805ec28bff121 | its record |
| `results/rb_heads_caption.npz` | 00782ea90b9dcccea8ad164ce21f48792d7422adba89bebf9acaf7705168cea8 | cross-fitted caption posteriors (R3) |
| `results/rb_heads_caption.json` | 02e9745f0946083c638b954048e1c5b3de18ab762a67012d45bab482ba765c07 | its record |
| `results/rb_diag_A0.json` | 7364a27804db4f902f1177bdc449dd2f2b79dde1e8584e2ac26e691d53f7a03f | round 1's shift report (R3's purity-4 check) |
| `results/cand_Rc_Rb_expected_A0.npz` | 628b21aeaf6d306f82bd9abb68e6c257348535c3aff0fad0a2d065fc48302981 | round-1 R-c arrays (regression check, R1 minus round-1 R-c) |
| `results/cand_Rc_Rb_expected_A0.json` | c6e2f83b73c47b9c054e6381a5d020a05618728cb820861b89f0b80de5a8a16e | round-1 R-c summary (regression check) |
| `results/rc_tau.json` | e10cf52b2e81ea5b5f7243363d7ddf226440c3a95512cbc7e0b9ed18c5e702bf | round-1 R-c thresholds (regression check) |
| `results/rule_application.txt` | 9a074850e70db8075e85bc5475e9361905182d6f0815f0023eaf380ccdee9785 | round 1's verdict (source of +0.4435) |
| `results/rb_reader_A1.pkl` | 4e1e4e23c20333b839aa1f92942891bdef51955317640d1014db5d097b7060f6 | A1 half-readers (ablation) |
| `results/rb_reader_A1.json` | 41ec3bd1ef99523430ce6aacb2576c7928446a93b65fb9ded2b3ae6a6089d649 | its record |
| `results/rb_reader_A1.npz` | f0297b92276cb0eda69f3717545c74a766203517950622f2966978b209c488cd | A1 bank features (ablation, purity-4 check) |
| `results/rb_bank_A1_half0.npz` | a499041d94b2396d28c586d1027bfe9fbd39d733334319ca8e2fe1ebec83bac7 | A1 bank of half 0 (ablation) |
| `results/rb_bank_A1_half0.json` | 7a60c1a35d920af3f0c7cdb7c63599a6be8c6e76a7eb3e721911847c1313bf3d | its record |
| `results/rb_bank_A1_half1.npz` | 887c80658ea295c72e21f2724ac8b289cbf9aee6789ffa5948f892f94ade87fd | A1 bank of half 1 (ablation) |
| `results/rb_bank_A1_half1.json` | 023f6b993258a7dad0c3cbc8fc6a670772afa0522c75d4281ddf7c1dce819692 | its record |
| `results/rb_heads_csd.npz` | 97906e87bade9756272b2120cfd3dcabaf2dae52305833554886868bcb96d028 | cross-fitted csd posteriors (ablation) |
| `results/rb_heads_csd.json` | 9c75532eb14321e1bb7ebe28171c09487665f2462d968943684ef308c785eda4 | its record |
| `results/rb_diag_A1.json` | 658503b6a50134f375d6cbd44626e28b827e6ec764b74fb43022189fadeefef0 | A1 shift report (ablation, purity-4 check) |

The pickles were written with scikit-learn 1.6.1 and numpy 2.2.6; the run uses the same versions (round 1's
`rb_build.load_readers` refuses another scikit-learn version). The step-1 inputs of D1, D2, D8 and §2 are asserted by
round 1's `common.verify_inputs`.

## 4. The candidates and the fusion family

### 4.1 What the three candidates share

- Configuration A0; the standard heads (D2); B and B′(A0) (D8, D9), built by round 1's `common.load_bundle` together
  with its regression check (step-1 arrays, B and the three B′ reproduced exactly; a failure stops the run).
- **Seed-42 reader inputs:** the 18 features of round 1 (per grouping, in A0 order: S_h^c, C_h^c, Δ_h^c, the sample
  standard deviation (ddof 1) of the four support-pair agreements, the same for the four contrast-pair agreements, and
  the share of the 4 support pairs whose image and caption arg-max groups coincide), from the standard heads, computed
  and checked by round 1's `rb_eval.seed42_features`; condition b from the swapped supports and contrasts. "The 24,576
  seed-42 feature rows" are condition a's 12,288 rows followed by condition b's.
- The weighted term T^c (D5), the fusion family (§4.5) and the matched counterpart (§4.6).
- The three candidates differ only in P^c(h).

### 4.2 R1: the current learned reader

Round 1's two A0 half-readers (`results/rb_reader_A0.pkl`, D14; each a `StandardScaler` fitted on its half's bank and a
multinomial logistic regression, chosen C 1.0 on half 0 and 100.0 on half 1), frozen. P^c(h) = the mean over the two
half-readers of `model_j.predict_proba(scaler_j.transform(x))` (round 1's `rb_eval.half_reader_probs` and
`rb_features.average_probs`). No setting is chosen in this round. Restricted to the k_top = 13 cells, R1 is round-1 R-c
exactly (§4.7).

### 4.3 R2: the adapted learned reader

The same two half-readers as R1, adapted to real episodes without labels. Their fitted models (coefficients,
intercepts, chosen C) are not changed.

a. **Re-standardisation.** μ42 = `X.mean(axis=0)` and σ42 = `X.std(axis=0)` (numpy, float64, ddof 0), with X =
   `numpy.vstack([F_a, F_b])` and F_c the float64 arrays of `rb_eval.seed42_features`, the 24,576 × 18 seed-42 feature
   matrix of §4.1. An entry of σ42 that is exactly 0 is replaced by 1 and reported; none is expected. Half-reader j's probability is
   `model_j.predict_proba((x − μ42) / σ42)` in place of `model_j.predict_proba(scaler_j.transform(x))`. Both
   half-readers use the same μ42 and σ42. P_n(h) = the mean over the two half-readers, for each of the 24,576 seed-42
   (episode, condition) values n.
b. **EM class-prior correction** (Saerens, Latinne and Decaestecker, 2002, "Adjusting the outputs of a classifier to new
   a priori probabilities: a simple procedure", Neural Computation 14(1):21 to 41), applied to the averaged
   probabilities of (a). The training prior is π_train(h) = 1/3 for each A0 grouping (the bank's classes are balanced).
   Start with π^(0)(h) = 1/3. For s = 0, 1, 2, ...:
   P′_n^(s)(h) = P_n(h)·π^(s)(h)/π_train(h) / Σ_h′ P_n(h′)·π^(s)(h′)/π_train(h′), and
   π^(s+1)(h) = the mean of P′_n^(s)(h) over the 24,576 values.
   Stop at the first s with max_h |π^(s+1)(h) − π^(s)(h)| < 1e-10, or when s + 1 = 10,000; then π̂ = π^(s+1). Everything
   in float64. The adapted probabilities are P′^c(h) = P^c(h)·π̂(h)/π_train(h) / Σ_h′ P^c(h′)·π̂(h′)/π_train(h′). The
   iteration count and whether the cap was reached are reported.
c. **Reader.** T^c = Σ_h P′^c(h)·s_h; the pick is arg max_h P′^c(h) (ties to the first grouping); the top-two margin is
   the largest minus the second-largest P′^c(h).
d. **Frozen.** μ42 and σ42 (18 values each) and π̂ (3 values) are written to the results at full precision before any R2
   score is computed and are reused unchanged on the test seeds, where P′ is computed from that seed's features with the
   frozen μ42, σ42 and π̂ (no EM is run on a test seed).
e. **Code check** (before any R2 score): R2's code with μ42 and σ42 replaced by each half's own scaler statistics
   (`scaler_j.mean_`, `scaler_j.scale_`) and without the EM step reproduces R1's P^c on seed 42 to within 1e-12
   (absolute) with identical picks. A failure stops the step.
f. **Shift report** (diagnostic; enters no rule): (i) per feature and half-reader j, (μ42 − `scaler_j.mean_`) /
   `scaler_j.scale_` and σ42 / `scaler_j.scale_`; (ii) π̂ and the iteration count; (iii) the distribution (mean and
   deciles) of the top probability max_h P′(h) over the 24,576 seed-42 values, beside R1's on seed 42 (round 1: mean
   0.694) and the bank's out-of-fold value (round 1: mean 0.781); (iv) the share of (episode, condition) values whose
   pick differs from R1's.

### 4.4 R3: the realistic-practice learned reader

Retrained with round 1's recipe on round 1's A0 banks made impure.

a. **Base banks.** Round 1's A0 banks of half 0 and half 1 (D14): 49,152 episodes each, in three blocks of 16,384
   episodes, in this order: (affect, caption; third grouping image), (affect, image; third caption), (caption, image;
   third affect). Bank rows are local scorer-train rows (positions in the ascending list of the 183,694 scorer-train
   rows); the painting of a local row is `painting_of_local_row` of `rb_halves.npz`; half j's rows are
   `local_rows_half{j}` (91,949 rows for half 0, 91,745 for half 1, ascending).
b. **Replacement draws.** For each half j, one generator g = `numpy.random.default_rng(s_j)` with s_0 = 21700 and
   s_1 = 21800, used in this order (N = 49,152; A = the bank's `anchor` array, (N,) int64 local rows; R = `local_rows_half{j}`
   of `rb_halves.npz`; paint = `painting_of_local_row`; side 0 = the arrays `pairs_a_img` and `pairs_a_txt`, side 1 = the
   arrays `pairs_b_img` and `pairs_b_txt`, each (N, 4) int64 local rows; position p = column p of both arrays of a side):
   1. *Positions:* U = g.random((N, 2, 4)); order = `numpy.argsort(U, axis=2, kind="stable")`.
   2. *Image rows:* img = R[g.integers(0, len(R), size=(N, 2, 4))]. Repeat: bad = (paint[img] == paint[A][:, None,
      None]); if no entry is bad, stop; otherwise img[bad] = R[g.integers(0, len(R), size=bad.sum())] (numpy fills the
      masked entries in row-major order).
   3. *Caption rows:* cap = R[g.integers(0, len(R), size=(N, 2, 4))]. Repeat: bad = (paint[cap] == paint[A][:, None,
      None]) | (paint[cap] == paint[img]); if no entry is bad, stop; otherwise cap[bad] = R[g.integers(0, len(R),
      size=bad.sum())].

   A replacement pair is thus the image of one row and the caption of another, both drawn uniformly from half j's
   scorer-train rows, from two different paintings, neither of them the anchor's painting. It may share a group of some
   grouping by chance. A replacement pair is cross-painting because every pair in the banks and in the real episodes is
   the image of one row and the caption of another row from a different painting. Replacement rows are drawn without
   regard to the paintings already in the episode other than the anchor's; in a dry run 0.31% (half 0) and 0.34% (half
   1) of replacement slots reused a painting of the episode, while round 1's banks and the seed-42 episodes have
   all-distinct paintings. This is negligible and fully specified.
c. **Impure bank of purity k** (k ∈ {1, 2, 3}): a copy of half j's round-1 bank in which, for every episode n and side
   s, the pairs at the positions order[n, s, 0] to order[n, s, 3 − k] (the first 4 − k entries of order[n, s]) are
   replaced: `pairs_{a|b}_img[n, p]` becomes img[n, s, p] and `pairs_{a|b}_txt[n, p]` becomes cap[n, s, p] at position p
   (a for side 0, b for side 1). These four arrays are the bank's only per-pair arrays; `anchor`, `candidates`,
   `block_pairs`, `block_size`, `seed` and `half` are unchanged. So only k of the 4 support pairs and k of the 4
   contrast pairs keep their shared group, under both conditions. Labels are unchanged. The banks are nested: a pair replaced at purity k is also replaced, by the same
   replacement pair, at every smaller k. Purity 4 is round 1's bank unchanged.
d. **Features** of each bank (both conditions, condition b from the swapped sets) with round 1's
   `rb_features.both_conditions`, from the cross-fitted posteriors of `rb_heads_{affect,image,caption}.npz` (a row of
   half j carries the posteriors of the heads trained on half 1 − j; replacement rows are rows of half j, so they are
   cross-fitted too), as round 1's `rb_build.bank_features` computes them. Check: the purity-4 features equal round 1's
   `half{j}__X` in `rb_reader_A0.npz` exactly; a failure stops the step. The SHA-256 of `order`, `img` and `cap` (int64,
   C order) of each half and of each purity's feature matrix of each half are written to the results with the D(k)
   table.
e. **Labels** unchanged (round 1's `rb_features.bank_labels`): condition a is labelled with the block's first grouping,
   condition b with its second.
f. **Choice of k, label-free.** For each k ∈ {1, 2, 3, 4}: SMD per feature between the 24,576 seed-42 feature rows
   (§4.1, standard heads) and the 196,608 bank rows of purity k (both halves, both conditions, half 0 first, condition a
   before b within a half), with round 1's formula (`rb_features.smd`): (mean on seed 42 − mean on the bank) /
   sqrt((variance on seed 42 + variance on the bank) / 2), variances ddof 1. D(k) = the mean over the 18 features of
   |SMD|. The chosen purity k* is the k with the smallest D(k) at full precision; exact ties go to the larger k. An SMD
   that is not finite stops the step. Check: the 18 purity-4 SMDs equal round 1's shift report
   (`rb_diag_A0.json`, `c_shift_report.smd`) exactly. The D(k) table and k* are written to the results before any R3
   half-reader is trained.
g. **Training at k*,** round 1's recipe unchanged. Per half j, X, y and the episode index are
   `rf.stack_conditions(F_a, F_b, y_a, y_b)` of the impure bank (rows 0 to N − 1 condition a, N to 2N − 1 condition b,
   episodes in the bank's order), the folds are `rf.episode_folds(N)`, and the fit is
   `rb_build.fit_half_reader(X, y, episode, N, 3)`, as `rb_build.stage_train` calls it (`rf` = `rb_features`). In it: one
   `StandardScaler` fitted once on all of half j's impure-bank features (both conditions); a multinomial
   `LogisticRegression` (lbfgs, `max_iter=2000`, no class weights) with C ∈ {0.01, 0.1, 1, 10, 100} chosen by five-fold
   cross-validation over half j's bank episodes (`KFold(5, shuffle=True, random_state=0)` over episode indices, both
   conditions of an episode in one fold), criterion the mean over the folds of `sklearn.metrics.log_loss` on the
   held-out fold, ties to the smaller C; then refit on all of half j's impure bank at the chosen C. Convergence warnings
   are counted and reported. P^c(h) = the mean over the two half-readers of `model_j.predict_proba(scaler_j.transform(x))`
   on the seed-42 features. Only k* is trained.
h. **If k* = 4,** R3 is R1 (round 1's half-readers, no retraining): it is reported as identical to R1, is not evaluated a
   second time, has R1's development numbers, and by the carry order (item 4) it cannot be carried.
i. **Reader and diagnostics.** T^c = Σ_h P^c(h)·s_h; pick and top-two margin as D5. Reported (enters no rule): the D(k)
   table with all 18 SMDs for each k, k*, each half-reader's chosen C and out-of-fold bank accuracy at it, and the
   distribution (mean and deciles) of the top probability on R3's impure bank (out-of-fold at the chosen C, pooled over
   halves) and on seed 42. D(k) matches levels: the three Δ features have SMD 0 for every k by construction (Δ^b = −Δ^a),
   and S/C and the two standard deviations mirror each other, so D(k) measures shifts in the levels of S, C, their
   spread and the match share, not the strength of the support-contrast difference. Reported beside it, entering no
   rule: the mean of |Δ_h| per grouping on seed 42 and on the bank of each purity.

### 4.5 The fusion family: 896 cells, the same for R1, R2 and R3

1. **z-scores.** z(B) and z(T^c) per ranking row over all 13 candidates (D6), before any gate and before the
   restriction.
2. **Thresholds.** τ_0, τ_1, τ_2, τ_3 = `numpy.percentile(m, [0, 25, 50, 75])` (method linear, numpy's default) of the
   candidate's own top-two margins m over the 24,576 seed-42 (episode, condition) values (condition a's 12,288 followed
   by condition b's), as round 1's `rc_core.thresholds`; for R2 the margins of its adapted probabilities P′. τ_0 is the
   seed-42 minimum, so on seed 42 its gate is always open (no gate); on a test seed a margin below τ_0 closes it. Each
   candidate's four values are written to the results at full precision before any score of that candidate is computed
   and are reused unchanged on the test seeds.
3. **Gate.** g^c = 1 if m^c ≥ τ, else 0, per episode and condition, the same for both directions.
4. **Unrestricted fused score** of a cell (k_top, τ_t, λ_u, λ_a): S = z(B) + λ_u·z(B) + λ_a·(g^c·z(T^c)), in
   `aspect_nested._combine`'s order with terms of weight 0 left out (float32), the gate multiplying after z-scoring
   (round 1's `rc_core.gated_terms`). S is computed in float32 exactly so; casting it to float64 changes no value. The
   restricted score S′ of item 6 may be held in float32 or float64 with identical ranks: a z-score over 13 candidates
   is at most √12 ≈ 3.464, so |S| ≤ 33·√12 ≈ 114.3 < 128, where float32 spacing is at most 7.6e-6, and `per_anchor`
   casts to float64.
5. **Top-k set.** For each ranking row, o = `numpy.argsort(−b, kind="stable")` with b the row's 13 stored scores of B
   (`bundle.B[c][d]`, float32; `B[a] == B[b]` is asserted before K is computed), so ties go to the lower candidate index; K = {o[0], ..., o[k_top − 1]}. B is condition-free, so K is the same under both
   conditions of an episode in each direction. On a test seed K comes from that seed's B.
6. **Restriction.** S′(j) = S(j) for every candidate j in K. The candidate at position p of o with p ≥ k_top gets
   S′(o[p]) = min over i in K of S(i) − 1 − (p − k_top), computed in float64. Every candidate outside K then scores at
   least 1 below the lowest score in K, and the candidates outside K keep B's order (ties broken as in o). With
   k_top = 13, K holds all 13 candidates and S′ = S unchanged. Since every candidate outside K scores below every
   candidate in K, first place always comes from K, so R@1, the other-aspect rate and the gain depend only on S within
   K. Members of K keep their float32 values exactly, and gates compare float64 margins with float64 τ. This reads the
   spec's "below it, B's order" as "outside K, B's order". Both readings put the same candidate first and count a tie for
   first as a miss, so R@1, the other-aspect rate, the gain, `strict`, either, every cross-fit pick, bar margin, gain
   statistic and GO check are the same under both. Only `swap` can differ, for k_top < 13, and `swap` enters no rule. At
   k_top = 13 both readings leave S unchanged.
7. **Cells.** 896 = 4 k_top values × 4 thresholds × the 56 (λ_u, λ_a) cells of D6, ordered k_top outer in the order
   13, 5, 3, 2, then τ index 0 to 3, then λ_u over NESTED_U, then λ_a over NESTED_A (inner). With zero-based indices
   κ (k_top), t, u and a, the cell number is ((κ·4 + t)·7 + u)·8 + a. Every tie goes to the lowest cell number, so no
   restriction and no gate win ties. Cells 0 to 223 (k_top = 13) are round-1 R-c's 224 cells in round 1's order.
8. **Cross-fit, min-margin, compared exactly** (round 1's `rc_core.control_choice` and `select_fused`, extended to 896
   cells). Every per-episode R@1 and gain of `per_anchor` is a mean over 4 rankings, hence a multiple of 0.25. For a
   score x and a tune half (the episodes with parity h), ρ(x) = Σ 4·R@1 and γ(x) = Σ 4·gain over the tune half's
   episodes, both integers. Dividing by the tune half's size gives round 1's mean criterion, so the choice is the same
   quantity computed exactly. The control takes σ* = the smallest of the 30 control sums (D6) with the largest
   ρ((1 + σ)·z(B)), unrestricted; ρ_ctrl = ρ((1 + σ*)·z(B)). The fused reader takes the cell with the largest
   min(ρ(S′) − ρ_ctrl, γ(S′)), compared as integers, ties to the lowest cell number; that cell's S′ scores the episodes
   of the other half (parity ≠ h). Means and their differences are never compared as floating-point numbers. On round
   1's 224 cells this gives `rc_core.select_fused`'s picks (cells 116 and 119, σ* = 0 on both halves). Per-anchor
   metrics come from the assembled scores (`per_anchor`).

### 4.6 The matched counterpart

G_cf = (g^a·z(T^a) + g^b·z(T^b)) / 2, averaged in float64 then cast to float32 (round 1's `rc_core.g_cf`), identical
under both conditions (asserted). The unrestricted counterpart score of a cell is z(B) + λ_u·z(B) + λ_a·G_cf (G_cf is not
z-scored again), and the restriction of §4.5 item 6 is applied with the same K. Because K depends on B only, the
restricted counterpart is condition-free; this is asserted per cell, as `_require_condition_free` asserts it. Cross-fit,
max-R@1: on each tune half the cell with the largest ρ (§4.5 item 8) among the same 896, ties to the lowest cell number,
scores the other half (on round 1's 224 cells: 58 and 123, as `rc_core.select_cf`). The product ḡ·z(T_cf) is not used: it drops the covariance of gate and term and removes more
than the condition.

### 4.7 Regression check (before any round-2 candidate number)

R1 on the k_top = 13 cells (cells 0 to 223), with both cross-fits restricted to those cells, must reproduce round-1 R-c
(`results/cand_Rc_Rb_expected_A0.{npz,json}` and `results/rc_tau.json`, D14):
- R1's T^c arrays equal the stored `T__{a,b}__{i2t,t2i}`, its top-two margins `margin__{a,b}` and its picks
  `pick__{a,b}`, exactly in value (the stored picks are int8 and R1's int64); its τ_0 to τ_3 equal rc_tau.json's 3.8684538364530674e-05, 0.21702129490553143,
  0.47973989883399526 and 0.7502585816077211 exactly;
- the chosen cells, by the integer criteria of §4.5 item 8 and §4.6 (which reproduce round 1's picks), are round 1's:
  fused cells 116 and 119, that is τ_2 with (λ_u, λ_a) = (0, 2) on half 0 and τ_2 with (0, 16) on half 1; counterpart
  cells 58 and 123, that is τ_1 with (0, 0.5) on half 0 and τ_2 with (0.5, 1) on half 1; control σ* = 0 on both halves;
- the per-anchor arrays `fused__{r1,gain,other,swap,strict}`, `cf__{r1,gain,other,swap,strict}` and `bar_v` are equal
  exactly;
- the bar margin is 0.4435221354166667 [0.21646171563312194, 0.6735669710776852] against the counterpart and the gain
  statistic 2.667236328125 [2.325087836946873, 3.012361650695922], at full precision.

R1's 896-cell statistics may be computed in the same pass as this check, but no number from cells 224 to 895 is written
or printed before the check passes. If any part fails, no candidate number is written, and the work stops and goes to
the user. R1's full result (896 cells)
minus round-1 R-c is then the effect of the top-k restriction alone; R2 and R3 against R1 are the effects of the reader
fixes.

### 4.8 A1 ablation (descriptive; seed 42 only)

- **Which reader:** the carried candidate. If no candidate is carried, the candidate with the largest bar margin (exact
  ties to the earliest of R1, R2, R3), labelled "best development candidate, not carried".
- **Built the same way on A1** = (affect, image, caption, csd), from round 1's A1 readers and banks (D14): R1/A1 uses
  round 1's A1 half-readers; R2/A1 uses μ42 and σ42 of the 24 seed-42 A1 features and the EM correction with
  π_train(h) = 1/4; R3/A1 uses round 1's A1 banks (98,304 episodes per half in six blocks of 16,384, in the order
  (affect, caption), (affect, csd), (affect, image), (caption, csd), (caption, image), (csd, image), no third grouping)
  with the draw procedure and
  seeds of §4.4 (N = 98,304), the cross-fitted csd posteriors of `rb_heads_csd.npz`, its own k* chosen by §4.4 item f
  over the 24 features (purity-4 checks against `rb_reader_A1.npz` and `rb_diag_A1.json`), and the training recipe of
  §4.4 item g. Seed-42 features use the standard heads of D2, csd included; its SMDs compare the 24,576 seed-42 rows
  with the 393,216 bank rows of each purity. If k*_A1 = 4, R3/A1 is R1/A1; R2/A1 runs the code check of §4.3 item e. The
  fusion family of §4.5 with its own thresholds, written before any A1 score; comparators B, B′(A1) = 18.805 and its own
  matched counterpart.
- **Reported:** the measures of item 2 and the A1 − A0 difference against the same reader on A0, paired per anchor
  (fused R@1 and bar margin), with intervals.
- It is built after the rule has been applied, does not delay the test, is never computed on a test seed, and decides
  nothing.

## 5. Development selection on seed 42 (items 1 to 5)

**Item 1. Candidates.** Three candidates on A0, each with the 896-cell fusion family: R1 (§4.2), R2 (§4.3) and R3
(§4.4). The A1 versions (§4.8) are never candidates.

**Item 2. Measures,** reported for every candidate; only item 3 decides.
- Fused reader R@1, counterpart R@1, B and B′(A0); the margin against the counterpart (R@1, gain, either); the fused
  reader against B and against B′ (R@1, gain, either); the counterpart against B; the bar margin and its comparator
  (D10, D11); the gain statistic; all per aspect pair as well (same comparator).
- **R1 minus round-1 R-c,** paired per anchor (fused R@1, margin and bar margin), with intervals: the effect of the
  top-k restriction alone. Round-1 R-c's arrays are `fused__r1`, `cf__r1` and `bar_v` of
  `cand_Rc_Rb_expected_A0.npz`.
- **R2 minus R1 and R3 minus R1,** paired per anchor (fused R@1, margin and bar margin), with intervals: the effects of
  the reader fixes under the same fusion family.
- Diagnostics (enter no rule): pick accuracy (D13) overall and per aspect pair and condition, beside round 1's values;
  the chosen cells of both cross-fit halves (k_top, τ index and value, λ_u, λ_a) for the fused reader and for the
  counterpart, and the control's σ; the share of open gates at each τ, overall, per condition and per aspect pair
  (label-free); R2's shift report (§4.3 item f); R3's D(k) table, k*, bank accuracy and top probabilities (§4.4 item
  i).
- The regression check (§4.7), before any candidate number.
- The A1 ablation (§4.8), after the rule has been applied.

**Item 3. Development bar:** D12.

**Item 4. Carry.** Let E be the set of candidates that clear the bar. Let M be the largest bar margin in E. Every member
of E whose bar margin is at least M − 0.05 is tied. The carried candidate is the tied member that comes first in the
order R1, R2, R3.

**Item 5. Kill.** If no candidate clears the development bar, no test is built and the seed-42 results go to the user,
who decides what follows. Pick accuracy, R2's shift report, R3's D(k) table and the bank accuracies are seed-42
diagnostics and enter no rule. Pick accuracy is not computed on test seeds until the GO verdict is recorded.

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
2. **Build.** Seeds 49, 50 and 51 (free in `docs/superpowers/episode_seed_ledger.md`; round 1 built none), once each,
   with `src/test/20261030_aspect_baselines/run_baselines.py --episodes-seed <s>` (4,096 episodes per aspect pair on
   selection rows; it also scores cosine and RCA). The 9 new per-pair episode SHA-256s must differ from each other and
   from every per-pair SHA-256 recorded in `src/test/20261030_aspect_baselines/results/baselines_seed{42,43,45,47,48}.json`;
   on a match the test stops and goes to the user. The seed ledger gets a row.
3. **What is frozen from seed 42:** the carried candidate's reader on A0 (R1: round 1's two half-readers with their
   scalers; R2: the same half-readers with μ42, σ42 and π̂; R3: k* and its two half-readers with their scalers), its
   τ_0 to τ_3, the 896-cell family (k_top values, threshold indices, λ grid, cell order and tie rules), the standard
   heads, the recipes of B and B′(A0), and the method-A checkpoint. **What is rerun on each seed's own parity halves:**
   B, B′(A0), the top-k sets K (from that seed's B), the gates (that seed's margins against the frozen τ), the fused
   reader (min-margin cross-fit over the 896 cells, by the integer criteria ρ and γ of §4.5 item 8, with the seed's own
   control σ*) and its matched counterpart (max-ρ cross-fit over the same 896 cells, §4.6). Per-seed cross-fitting is part of the method's definition; the counterpart has the same freedom; the
   intervals hold the cross-fit picks fixed.
4. **Order of computation.** On the test seeds only the GO quantities are computed until the verdict is written to
   `results/test_verdict.json` of this folder (with this file's SHA-256 and the Amsterdam time): the per-anchor arrays
   of the fused reader, its counterpart, B, B′, cosine and RCA on each seed, and the pooled differences of item 6.5. The
   per-seed cosine and RCA summaries that `run_baselines.py` writes when it builds a seed are allowed and are not read
   before the verdict. Pick accuracy, per-seed and per-pair numbers of the candidate, the diagnostics and the
   frozen-cell line come only after the verdict.
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
8. **Frozen-cell line** (descriptive, after the verdict, not a second verdict): on each test seed the fused reader and
   its counterpart scored with the cells that seed 42's cross-fits chose (the cell chosen on seed-42 half h scores the
   test seed's episodes of parity 1 − h; the chosen cells are those of §4.5 item 8 and §4.6), compared with the same
   comparators as item 6.5.
9. **Claim licensed.** A GO shows that the carried candidate, a reader on A0 without the CSD grouping, beats each
   comparator, pooled over the three aspect pairs, on new episodes drawn from the same 6,451 selection paintings. It is a
   reader-fix GO that leaves the CSD question open (round 1's user ruling on A0 GOs; the A1 ablation is descriptive). It
   does not show transfer to new paintings (the held split, 12,281 paintings, stays reserved for the paper test) or a
   margin on each aspect pair. For the paper: the groupings were built without evaluation labels.
10. **Multiplicity disclosure** (reported with any result of this round): these candidates were designed after seeing
    the seed-42 results of round 1's seven candidates, so the selection inflation of the development numbers is larger
    than in round 1 (D12), and each candidate's cross-fit chooses among 896 cells, as does its counterpart's. The
    fresh-seed test is the protection; the frozen-cell line of item 6.8 accompanies it.

## 7. Order of work, cutoff and timeline

- **Order of computation:** (1) round 1's bundle check and this rule's regression check (§4.7); (2) R1 on the 896
  cells; (3) R2: μ42, σ42, its code check and π̂ written, then its thresholds, then its scores; (4) R3: the replacement
  draws, the impure banks, their features, the purity-4 checks, the D(k) table and k* written, then training at k*, its
  thresholds and its scores. Steps 3 and 4 may run in parallel with steps 1 and 2 up to the point where a candidate
  number would be computed; no candidate number is computed before the regression check has passed. (5) The independent
  re-derivation; (6) the rule applied; (7) if a candidate is carried, the test; (8) the A1 ablation.
- **Cutoff: Thursday 8 October 12:00.** A candidate "has development numbers" when its bar margin and its gain
  statistic, each with its interval, are written to its results file. Candidates without development numbers by the
  cutoff are dropped from this round. **Target:** all three on Tuesday 6 October, the day this rule is written.
- The rule is applied as soon as all three candidates have development numbers, or at the cutoff with those that have
  them. Before it is applied, an independent agent re-derives, with its own code, every decision number: each
  candidate's bar margin, its interval and its comparator, its gain statistic and its interval, the regression check,
  each candidate's τ, μ42, σ42, π̂, the D(k) table and k*, R3's chosen C per half, and the chosen cells of both
  cross-fits. The re-derivation writes its own code for everything except the bootstrap, which is
  `src.eval.aspect_metrics.cluster_bootstrap` itself (5,000 resamples, seed 42, chunk 250, clusters the anchor
  paintings). **Agreement** means: every discrete quantity identical (picks, gates, top-k sets, chosen cells, σ*, k*,
  chosen C, the bar comparator, each clause of D12); μ42, σ42, π̂, τ and D(k) equal to within 1e-12 (relative); every bar
  margin, gain statistic and bound equal to within 1e-9 percentage points. A difference beyond these is traced to its
  cause before the rule is applied; the computation that follows this file's text settles it, and the user is told.
- **Time-box:** ends Tuesday 13 October. If the test cannot be completed by then, no partial verdict is issued; the user
  is told and decides.

| Day (Amsterdam) | Work |
|---|---|
| Tue 6 Oct | This rule written, checked by a fresh Opus reviewer (findings applied), committed and sent to the user; implementation by two subagents in parallel (fusion stream: top-k and the 896-cell cross-fit with its counterpart; reader stream: R2 and R3), reusing round 1's code by import; regression check; R1, R2 and R3 on seed 42; independent re-derivation; the rule applied (target) |
| Wed 7 Oct | Anything left from Tuesday. If a candidate is carried: the sensitivity projection (§6.1), seeds 49 to 51 built, the test run to its verdict, then an independent re-derivation of the GO quantities; the A1 ablation |
| Thu 8 Oct | 12:00: cutoff for development numbers; the rule is applied at the latest then |
| by Tue 13 Oct | Whole-branch final review on the most capable model (re-derives every load-bearing number; one fix wave and a scoped re-review), then the report; the time-box ends |

## 8. Outcome-to-action table

| Outcome | Action |
|---|---|
| A script check fails (a SHA-256 of this rule or of a D14 input, round 1's bundle check, the regression check of §4.7, R2's code check, R3's purity-4 checks, a non-finite SMD, a condition-free assertion) | Stop that step and report to the user; nothing is improvised |
| A candidate has no development numbers at the cutoff | Dropped from this round; reported as such |
| R3's k* is 4 | R3 is R1; reported as identical; it cannot be carried (§4.4 item h) |
| Pick accuracy, R2's shift report, π̂, R3's D(k) table or a bank accuracy look good or bad | Reported only; no action |
| No candidate clears the bar | No test is built; the seed-42 results go to the user, who decides what follows (design L is scheduled for later); the A1 ablation runs on the best candidate (§4.8) |
| Exactly one candidate clears the bar | It is carried to the test (§6) |
| Several candidates clear the bar | Item 4: the largest bar margin, ties within 0.05 to the earliest of R1, R2, R3 |
| On the test seeds, before the verdict | Only the GO quantities (§6.4) |
| Test: all seven checks pass | **GO**: a reader-fix GO on A0 that leaves the CSD question open, within the claim of §6.9, with the disclosure of §6.10 |
| Test: any check fails, including a partial pass | **NO-GO**, read by §6.6 (inconclusive at x, or not beaten); the user decides what follows |
| The independent re-derivation or the final review disagrees with a reported number or verdict beyond the agreement of §7 | The difference is traced to its cause; the computation that follows this file's text settles it; the corrected verdict goes to the user with the cause; the rule is not changed |
| A run crashes, or a bug is found before the rule is applied (by the implementer, a test or the re-derivation) | Correcting code so that it matches this file is not a change of the rule. A run that crashed before its results file was complete is repeated after its partial outputs are deleted, and the crash is logged. A results file already written is not overwritten: the corrected run writes beside it with the suffix `_fix<n>`, the old file is kept, the log records the cause and the numbers before and after, and the rule uses the corrected file |
| A step stops on a failed check | The other candidates continue. The stopped candidate has no development numbers until the user rules; the user may drop it at once, after which the rule is applied to the rest as soon as they have numbers; otherwise it is applied at the cutoff |
| Any situation this file does not cover | The user decides; this file is not changed after its commit without the user |

## 9. Process

- Scripts of this folder assert this file's SHA-256 (as round 1's scripts asserted round 1's rule) and the SHA-256 of
  every D14 input they read, refuse to overwrite non-smoke results, and write only to this folder's `results/`
  (gitignored by this folder's `.gitignore`, a copy of round 1's), except that `run_baselines.py` (§6.2), run without
  `--overwrite`, writes its own outputs for seeds 49, 50 and 51 in `src/test/20261030_aspect_baselines/results/`, and
  `docs/superpowers/episode_seed_ledger.md` gets its row. Smoke runs write to `results/smoke/`, may be
  overwritten and are not results.
- Round 1's folder is read only: its modules are imported by path and not modified, and its functions that write
  files (`common.save_candidate`, `common.write_json_once`, the `rb_build.py` stages, `run_rc.py`) are not called.
- CPU only: `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8`, at most 3 processes, `uptime` and `free -g`
  checked first. The main session launches the runs; subagents implement.
- Records: the log `20261118_reader_fix_round2_log.md` in this folder; a report under `docs/reports/auto/v2/` with a row
  in `docs/reports/reports_sum.md` and `scripts/check_reports_sum.py` run; `.claude/<yyyymmdd>_log.md` for any edit to
  an existing source file; the seed ledger if seeds are built. Commits go to `main` without a further request
  (authorisation above).
