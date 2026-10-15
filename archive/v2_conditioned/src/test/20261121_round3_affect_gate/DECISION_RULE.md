# Decision rule: reader fix, round 3 (one-sided affect steering on R1, fresh-seed test), committed before any code

**Written** 2026-10-06 from 19:28 (Amsterdam), from the approved spec
`docs/superpowers/specs/2026-10-06-round3-affect-gate-design.md` (commit 728f5d7, SHA-256
929ecec782467e7cae70ff82c0cf696220e783af4961d4e240654c3ca2de887f), before any implementation script of this folder
exists. The only
numbers it states are earlier rounds' and the brainstorm's, which it reuses.

**Precedence.** Where this file differs from the spec, from round 2's rule
(`src/test/20261118_reader_fix_round2/DECISION_RULE.md`, SHA-256
368bec11363b222d348622a37a6e3aedcfe795d772781d7a255d79b60fab265c) or from round 1's rule
(`src/test/20261117_reader_fix_csd/DECISION_RULE.md`, SHA-256
613d8c9d13fb7d4787ec0fb91dc65cfd4a192d07c88ab8f961da3adc1c35d23c), **this file governs**. Everything needed to apply
the rule is written here; the earlier rules and the spec are cited for provenance only.

**Checked** before its commit by a fresh Opus reviewer (spec §7; not an ARS round; report
`rule_check/opus_rule_check.md`). It found 2 blocking, 7 should-fix and 16 nit findings, all applied before the commit.
It verified all 36 SHA-256s and reproduced every seed-42 number of §5 items 2 and 3 through the float32 path at full
precision.

**Status.** One candidate, AFF, tested on the fresh episode seeds 49, 50 and 51 (§6), the only confirmatory step. Seed
42 serves only as a regression check of the code (§5): AFF was found on it by a search of about 50 label-free variants
(brainstorm `docs/reports/auto/v2/2026-11-20_r1_levers_brainstorm.md`), so seed 42 cannot be its development data. R1
runs beside AFF and is descriptive except as the input of one pre-registered secondary check (§6.6).

**Authorisation.** The user's decisions of 2026-10-06 (handoff `docs/superpowers/handoffs/2026-10-06-round3-affect-gate-handoff.md`
§2, and the spec's open points settled one at a time in chat between 19:11 and 19:21): keep improving R1, no design L
now; AFF as a pre-registered round with R1 beside it, straight on seeds 49 to 51; affect frozen by name; seed 42 as two
regression checks (§5 items 2 and 3; items 1 and 4 add the bundle check and the recorded bar); GO = round 2's seven checks; AFF minus R1 as a pre-registered secondary check that never changes GO;
R1 run in full on the test seeds, descriptive; the sensitivity projection logged only; the random-share control as a
descriptive ride-along; the disclosure with a stated prior. The user approved the spec and allowed this file, and the
work under it, to be committed to `main` without asking. After its commit this file changes only with the user's
approval.

**Dates.** Folder and report dates (`20261121`, `2026-11-21`; earlier rounds' `20261117`, `20261118`, `20261120`) are
sequence numbers, not calendar dates. Calendar times in this file are Amsterdam local time.

**Prior** (written before any test number; spec §6). We expect AFF's pooled fresh-seed bar margin (§7 item 2) to be
about +0.3 R@1: about half the median of AFF's one-sided cluster on seed 42 (+0.63), since earlier fresh-seed tests
roughly halved development effects. That is above the detectable margin of about 0.2 that the round-2 tab projected,
exploratorily, for R1 against its counterpart. Our chance of a GO is moderate: a NO-GO would not be a surprise, and a
large fresh-seed margin would be.

## 1. Glossary

| Term | Meaning in this file |
|---|---|
| episode | a query (the anchor row: its image for image-to-caption ranking, its caption for caption-to-image ranking), 4 support pairs, 4 contrast pairs and 13 candidates in the other modality; p_A (candidate column 0) shares the query's value of aspect A, p_B (column 1) its value of aspect B, 11 negatives (columns 2 to 12) share neither |
| per-anchor | per episode (one value per episode, averaged over its 4 rankings) |
| ranking row | one (episode, condition, direction): the 13 candidate scores that one ranking orders |
| condition a / b | a: the supports show aspect A and the target is p_A; b: supports and contrasts swapped, target p_B. Each episode gives 4 rankings (2 conditions × 2 directions) |
| aspect pairs | emotion × style, emotion × genre, style × genre, in this order; the first aspect is A |
| A0 | the configuration (affect, image, caption), in this order; the order breaks every arg-max tie (the first grouping wins) |
| grouping, head | a label-free partition of the scorer-train rows (D1); a logistic regression on frozen CLIP ViT-B/32 features predicting a row's group from the image alone (image head) or the caption alone (caption head) |
| p_h(x), Δ_h, s_h | head posterior of item x on grouping h; support minus contrast agreement (D3); grouping score (D4) |
| half-reader | one of round 1's two A0 multinomial logistic regressions, each trained on the practice bank of one painting half of the scorer-train rows |
| P^c(h), T^c, m^c, π^c | the reader's probability that grouping h is the one the supports share under condition c (mean over the two half-readers); the weighted term Σ_h P^c(h)·s_h; the top-two margin; the pick arg max_h P^c(h) |
| R1 | round 1's learned reader with the confidence gate g^c = 1[m^c ≥ τ] (D6) |
| AFF | R1 with the gate opened only on affect picks: g^c = 1[m^c ≥ τ]·1[π^c = affect] (D6) |
| τ_0..τ_3 | R1's frozen thresholds (D6) |
| cell | one (τ index, λ_u, λ_a) of the 224 of D8 |
| G_cf | the two-condition mean of the gated term, the matched counterpart's term (D9) |
| B, B′(A0) | the best condition-free score of the project (D10) and B rebuilt on A0's own groupings (D11) |
| round-1 R-c | round 1's confidence-gated reader on the expected term of its learned reader on A0 (`cand_Rc_Rb_expected_A0`); R1 restricted to the 224 cells is round-1 R-c exactly |
| parity halves, tune half h | the two cross-fit halves of a seed's episodes by episode index parity; the cell chosen on tune half h (episodes of parity h) scores the episodes of parity 1 − h |
| bundle | everything a seed's evaluation needs: episodes, cosine, B, B′(A0), the A0 posteriors, the grouping scores and the reader features (§4.1) |
| test seeds | episode seeds 49, 50 and 51 (§6) |
| smoke seeds | episode seeds 9001, 9002 and 9003, built only with `--smoke` (64 episodes per pair) for the end-to-end wiring smoke test (§10); never results |
| told mapping | the evaluation-label map aspect → grouping (D14), used only for the diagnostic pick accuracy |

## 2. Data, metrics and intervals

- **Seed-42 episodes:** `src/test/20261030_aspect_baselines/results/episodes_seed42.npz` (SHA-256
  12af979432ff1a20c88b614ab9e672f203eeeed9c2465bff305e72d01e6c0986): 12,288 episodes (4,096 per aspect pair) on 4,602
  anchor paintings, selection rows.
- **Test-seed episodes:** `src/test/20261030_aspect_baselines/results/episodes_seed{49,50,51}.npz`, built once each by
  §6.2, 12,288 episodes per seed on selection rows, loaded through `src/test/20261101_aspect_factor_gonogo/run_gonogo.py::EvalContext(seed, False)`,
  which checks each pair's episode SHA-256 against `baselines_seed{s}.json`.
- **Metrics,** per episode averaged over both conditions and both directions, pooled over the three aspect pairs, in
  percentage points: R@1 (the target ranks strictly first; ties miss), other-aspect rate, condition gain = R@1 − other,
  either rate = R@1 + other (`src.eval.aspect_metrics.per_anchor` returns r1, gain, other, swap and strict; the either
  rate is round 1's `common.either`). A condition-free scorer has gain exactly 0 on every
  episode.
- **Intervals:** 95% percentile intervals of the bootstrap that resamples anchor paintings
  (`src.eval.aspect_metrics.cluster_bootstrap`: 5,000 resamples, seed 42, chunk 250; scaled to percentage points as
  round 1's `common.point_ci` does). Cross-fit picks are fixed before resampling. **Pooled over the test seeds:** the
  per-anchor values of seeds 49, 50 and 51 are concatenated in that order (36,864 episodes) and the clusters are the
  anchor paintings (`groups[anchor]`), so a painting that anchors episodes on several seeds is one cluster. "A lower
  bound above 0" means strictly greater than 0.
- **Precision:** every threshold in this file applies to the full-precision point or bound, never to a rounded value.
- Held rows are not read. Nothing in this file uses the GPU.

## 3. Definitions (the only definitions; every item below refers to them)

**D1. Groupings.** Three groupings, none re-chosen:
- *affect*: Leiden communities on GoEmotions caption probabilities, 41 groups (`partition_L` in
  `src/test/20261111_community_told_oracle/results/per_anchor_told_oracle.npz`, SHA-256
  27e8101161607da2d6936aa5a8c84f8c296629fe08e37a68c8418c9356af1366);
- *image* and *caption*: k-means with 64 clusters on CLIP image or caption features (E2;
  `src/test/20261031_pseudo_partitions/results/partitions.npz`, SHA-256
  cfd57dbb21216cdc9d1c2e42d1b408648bc18147f62354b52202ade4ab8a9caa).

E2's own affect grouping (k-means 64 on GoEmotions probabilities, "affect-km") is used only inside B (D10). The csd
grouping and configuration A1 are not used in this round.

**D2. Standard heads,** exactly as in rounds 1 and 2: the affect heads refit with `run_told_oracle.fit_one_head`
(60,000-row draw, `LogisticRegression(C=1, max_iter=300)` on unit-normalised CLIP features; identity with
told_oracle.json arm L's head asserted); the image and caption posteriors stored in
`src/test/20261108_new_method_quick_checks/results/n6_posteriors.npz` (SHA-256
2ad75026e5869c461fed156ffcdcf207c66f04a59dae514358522fbbaa9df2e0; keys `image__*`, `caption__*`). Heads and
posteriors are fitted on scorer-train rows and do not depend on the episode seed; the same posteriors serve every seed.

**D3. Agreement and Δ.** For an image i and a caption t, a_h(i, t) = p_h(i)·p_h(t). Under condition c, S_h^c is the mean
agreement over the 4 support pairs, C_h^c the mean over the 4 contrast pairs, and Δ_h^c = S_h^c − C_h^c
(`src.eval.aspect_quick_checks.aspect_deltas`). Δ_h^b = −Δ_h^a exactly.

**D4. Grouping score.** s_h(q, k) = p_h(q)·p_h(k), the query's posterior from its own modality's head and the
candidate's from its own (round 1's `common.grouping_stack`). It does not depend on the condition.

**D5. Reader, term and pick.** Round 1's two A0 half-readers (`results/rb_reader_A0.pkl` of round 1, D15; each a
`StandardScaler` fitted on its half's bank and a multinomial logistic regression, chosen C 1.0 on half 0 and 100.0 on
half 1), frozen. Their input is the 18 features of round 1: per grouping, in A0 order, S_h^c, C_h^c, Δ_h^c, the sample
standard deviation (ddof 1) of the four support-pair agreements, the same for the four contrast-pair agreements, and
the share of the 4 support pairs whose image and caption arg-max groups coincide, computed from the standard heads by
round 1's `rb_eval.seed42_features` (it takes a bundle and works on any seed's bundle; condition b from the swapped
supports and contrasts). P^c(h) = the mean over the two half-readers of `model_j.predict_proba(scaler_j.transform(x))`
(round 1's `rb_eval.half_reader_probs` and `rb_features.average_probs`). T^c = Σ_h P^c(h)·s_h (round 1's
`common.expected_term`). π^c = arg max_h P^c(h), ties to the first grouping in A0 order (`numpy.argmax`); index 0 is
affect. m^c = the largest minus the second-largest P^c(h) (round 1's `common.top_two_margin`). The same for R1 and AFF.

**D6. Thresholds and gates.** τ_0, τ_1, τ_2, τ_3 = 3.8684538364530674e-05, 0.21702129490553143, 0.47973989883399526,
0.7502585816077211 (round 1's `results/rc_tau.json`: `numpy.percentile` at 0, 25, 50, 75 of R1's 24,576 seed-42 top-two
margins), read from that file (SHA-256 asserted) and used unchanged on every seed. For τ index t, per episode and
condition, the same for both directions:
- **R1's gate:** g_t^c = 1 if m^c ≥ τ_t, else 0;
- **AFF's gate:** g_t^c = 1 if m^c ≥ τ_t and π^c = affect, else 0.

Margins are float64 and compared with float64 τ. On seed 42 τ_0 is the minimum margin, so R1's gate at τ_0 is always
open there; on a test seed a margin below τ_0 closes it.

**D7. Affect, frozen by name, and its label-free reason.** Only affect picks may open AFF's gate. The reason recorded
for the paper, computed without labels: affect is the A0 grouping least redundant with B. Redundancy of grouping h in
direction d = the mean, over the seed's ranking rows of direction d (one row per episode; B and s_h are condition-free),
of the Pearson correlation over the 13 candidates between z(s_h) and z(B), where z is D8's per-row z-score (float32,
then cast to float64) and rows whose correlation denominator is 0 are left out (the brainstorm's `bs_05_aff.row_corr`).
Seed-42 values (brainstorm, exploratory): affect 0.35348060377541385 (i2t) and 0.3828024789253903 (t2i); image
0.7145397990123284 and 0.7090878258485419; caption 0.6182295729609555 and 0.665152773464146. **Check:** round-3 code
recomputes the six values on seed 42; they must equal the values above exactly, and affect must have the smallest value
in both directions. The criterion is not
re-chosen on the test seeds; its test-seed values are reported after the verdict (§7). It was stated after affect was
seen to pay, so it explains the choice and does not protect it.

**D8. Fusion family: 224 cells.**
1. *z-scores.* z(B) and z(T^c) per ranking row over the 13 candidates (`zscore_rows`: population standard deviation; a
   constant row becomes zeros), before any gate (round 1's `aspect_nested._zdict`).
2. *Gated term.* g_t^c·z(T^c), the gate multiplying after z-scoring (round 1's `rc_core.gated_terms`).
3. *Fused score* of a cell (t, λ_u, λ_a): S = z(B) + λ_u·z(B) + λ_a·(g_t^c·z(T^c)), in `aspect_nested._combine`'s order
   with terms of weight 0 left out, float32.
4. *Cells.* λ_u ∈ NESTED_U = (0, 0.5, 1, 2, 4, 8, 16), λ_a ∈ NESTED_A = (0, 0.25, 0.5, 1, 2, 4, 8, 16); 224 = 4 τ indices
   × 7 × 8, ordered τ index outer (0 to 3), then λ_u, then λ_a (inner). With zero-based indices t, u, a the cell number
   is (t·7 + u)·8 + a. These are round 2's cells 0 to 223 (k_top = 13, where its top-k restriction is the identity) and
   round 1's 224 cells in round 1's order (round 2's `r2_fusion.cell_statistics(..., n_kappa=1)`). No top-k
   restriction is used.
5. *Cross-fit, min-margin, compared exactly* (round 2's rule §4.5 item 8; `r2_fusion.control_choice` and
   `select_fused`). Every per-episode R@1 and gain is a multiple of 0.25. For a score x and tune half h, ρ(x) = Σ 4·R@1
   and γ(x) = Σ 4·gain over the tune half's episodes, both integers. The nested control takes σ* = the smallest of the
   30 control sums (`aspect_nested.control_sums`: 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.25, 2.5, 3, 4, 4.25, 4.5, 5, 6,
   8, 8.25, 8.5, 9, 10, 12, 16, 16.25, 16.5, 17, 18, 20, 24, 32) with the largest ρ((1 + σ)·z(B)); ρ_ctrl = ρ((1 +
   σ*)·z(B)). The fused reader takes the cell with the largest min(ρ(S) − ρ_ctrl, γ(S)), compared as integers, ties to
   the lowest cell number; that cell scores the episodes of parity 1 − h. Means and their differences are never
   compared as floating-point numbers. Per-anchor metrics come from the assembled scores (`per_anchor`).

**D9. Matched counterpart.** G_cf,t = (g_t^a·z(T^a) + g_t^b·z(T^b)) / 2, averaged in float64 then cast to float32
(round 1's `rc_core.g_cf`), identical under both conditions (asserted). The counterpart score of a cell is z(B) +
λ_u·z(B) + λ_a·G_cf,t (G_cf is not z-scored again), condition-free (asserted per cell, as `_require_condition_free`
does). It keeps every ingredient of the fused reader, its own gates included, and removes only the condition. Cross-fit,
max-R@1: on each tune half the cell with the largest ρ among the same 224, ties to the lowest cell number, scores the
other half (`r2_fusion.select_cf`). The product ḡ·z(T_cf) is never used.

**D10. B.** `crossfit_condition_free(cos, T_N1u, T_6u, parity)` on the seed's episodes and parity halves, with T_N1u the
centred uniform factor term of the method-A checkpoint (`src/test/20261101_aspect_factor_gonogo/checkpoints/A3_seed42.pt`,
SHA-256 dadfef1bed95bbd647c39fa0ea483edea78627d440a23b5a032083e76dd469e2; the call sequence is §4 item 1's) and T_6u averaged over E2's three
k-means-64 groupings (affect-km, image, caption) from the posteriors `affect__*`, `image__*` and `caption__*` of
`n6_posteriors.npz` (`n6.n6_terms`). Seed 42: R@1 18.341064453125.

**D11. B′(A0).** `crossfit_condition_free(cos, T_N1u, T_6u(A0), parity)`, with T_6u(A0) = `uniform_probe_scores` over
A0's groupings with the standard heads (as step 1's `run_step1.evaluate_general` computes it). It depends on the seed
only, never on the reader. Seed 42: 18.436686197916664.

**D12. Comparators and margins.**
- AFF's (and, descriptively, R1's) condition-free comparators: B, B′(A0) and its own matched counterpart (D9). External
  baselines: cosine and RCA (the strongest raw metric learned from the example pairs), as stored per anchor by
  `run_baselines.py` in `per_anchor_seed{s}.npz` (keys `cosine__*`, `rca__*`).
- *Margin* = fused reader minus its matched counterpart, paired per anchor, R@1.
- *Bar comparator* = whichever of B′(A0), the counterpart and B has the largest mean R@1 over the episodes considered
  (full precision); ties to the earliest in the order B′, counterpart, B (round 1's `common.bar_comparator`). It is
  chosen once per scope: over the seed's episodes for that seed's numbers, over the 36,864 pooled episodes for pooled
  numbers. A per-pair bar margin uses the comparator of the scope it breaks down (round 1's `common.bar_info`). *Bar
  margin* = fused reader minus the bar comparator, paired per anchor, R@1, with its interval.
- *Gain statistic* = per-anchor condition gain of the fused reader minus that of its counterpart (the latter is 0 by
  construction, asserted), with its interval (round 1's `common.evaluate_fused`, key `gain_statistic`).

**D13. Development bar** (round 2's D12, evaluated for AFF on seed 42 and recorded; §5 item 4): (1) bar margin point at
least +0.5; (2) bar margin lower bound above 0; (3) gain statistic lower bound above 0.

**D14. Told mapping and pick accuracy (diagnostic only; enters no rule).** Emotion → affect, style → image, genre →
image. Pick accuracy = per episode, the mean over the two conditions of 1[π^c = told grouping], pooled, with its interval
(round 1's `common.pick_statistics`). Chance 1/3. R1 on seed 42: 51.3% (51.261393229166664).

**D15. Inputs.** Read only; never written or modified. Scripts assert every SHA-256 below before using the file; a
mismatch stops the run. The pipeline calls round 1's `common.verify_inputs()` on every seed (the D1, D2 and D10 inputs and
§2's seed-42 episodes) and asserts `src/test/20261111_community_told_oracle/results/told_oracle.json` (SHA-256
76d9ec896b941a518e7db6684805fe0bc329f6afe4a48e1993992fbb221d70d2; D2's head identity). After each build (§6.2) the
SHA-256s of `episodes_seed{s}.npz`, `per_anchor_seed{s}.npz` and `baselines_seed{s}.json` are written to
`results/build_seed{s}.json` and asserted by every later step.

| File (under `src/test/`) | SHA-256 | Used for |
|---|---|---|
| `20261117_reader_fix_csd/DECISION_RULE.md` | 613d8c9d13fb7d4787ec0fb91dc65cfd4a192d07c88ab8f961da3adc1c35d23c | asserted by round 1's `common.load_bundle` |
| `20261117_reader_fix_csd/common.py` | 99496dcf859ff0b0f746266125c975c5e2f9632e0c2c91d74df17102c65c10a7 | bundle (seed-42 check), terms, margins, evaluation (imported) |
| `20261117_reader_fix_csd/rc_core.py` | e649fff5d8253c8ba7aae0dbfff7a68321cef5fe3caa06d7b325a27fe904185c | gated terms, G_cf, gates (imported) |
| `20261117_reader_fix_csd/rb_build.py` | 63c7310c890675cc63e49838feb364cdd1c5c68aafdfb0f0f7972128476cba7b | reader loading (imported) |
| `20261117_reader_fix_csd/rb_eval.py` | 1826e65fd73b8c6cecae33b5df48503bf2cf33e067fc291d5c078c51184e0b14 | reader features and half-reader probabilities (imported) |
| `20261117_reader_fix_csd/rb_features.py` | 1e758f6d5de1b1dd246592f4045e0ae1dcc520c1d6012540a9e69b3314fb6caf | features, probability averaging (imported) |
| `20261117_reader_fix_csd/results/rb_reader_A0.pkl` | 6387b469662446734e650571c88a152651da3d66bcade28fb6cdd7828f58d34c | the two A0 half-readers with their scalers |
| `20261117_reader_fix_csd/results/rb_reader_A0.json` | cd90da922967cf1b62e437d3927c97cd9cfdae2c139ee392744538d3e3e5c6a8 | the pickle's record |
| `20261117_reader_fix_csd/results/cand_Rc_Rb_expected_A0.npz` | 628b21aeaf6d306f82bd9abb68e6c257348535c3aff0fad0a2d065fc48302981 | round-1 R-c arrays (regression check 2) |
| `20261117_reader_fix_csd/results/cand_Rc_Rb_expected_A0.json` | c6e2f83b73c47b9c054e6381a5d020a05618728cb820861b89f0b80de5a8a16e | round-1 R-c summary (regression check 2) |
| `20261117_reader_fix_csd/results/rc_tau.json` | e10cf52b2e81ea5b5f7243363d7ddf226440c3a95512cbc7e0b9ed18c5e702bf | τ_0..τ_3 (D6) |
| `20261117_reader_fix_csd/results/rule_application.txt` | 9a074850e70db8075e85bc5475e9361905182d6f0815f0023eaf380ccdee9785 | round 1's verdict (source of +0.4435) |
| `20261118_reader_fix_round2/DECISION_RULE.md` | 368bec11363b222d348622a37a6e3aedcfe795d772781d7a255d79b60fab265c | provenance (round 2's rule) |
| `20261118_reader_fix_round2/r2_fusion.py` | 1406e8e71629e74c468f8b009a110e49aa05c481ed52cc1b8d99b59f87016027 | cell statistics, integer cross-fits, assembly (imported) |
| `20261118_reader_fix_round2/results/rule_application.txt` | cf1ec3852e334b1d56920bd8ab25c2db858bf97ab46ea36a7f65ebdd0916741f | round 2's verdict (provenance) |
| `20261120_r1_levers_brainstorm/bs_lib.py` | 45cc5abd941ca1cd7e47f415cdf0e11d440fd9c703b99516e23eb6e41d49bf88 | provenance of AFF's seed-42 numbers (not imported) |
| `20261120_r1_levers_brainstorm/bs_04_readers.py` | 0260fe5ad61e3d4a664cbb921f7d84efed8c9329716229310651712697a372b8 | provenance (AFF; not imported) |
| `20261120_r1_levers_brainstorm/bs_05_aff.py` | 4c947b1d42df1caa7fb7f769f245e7df9a59bce22ca2611413c6a5a7fd45d734 | provenance (redundancy, AFF minus R1; not imported) |
| `20261120_r1_levers_brainstorm/bs_10_subsets.py` | b00d9ed0b18e65e1e2be4dd8cdc5e05e277f673dbec5c826ddce4e18669cd7b4 | provenance (random-share control; not imported) |
| `20261120_r1_levers_brainstorm/results/bs_04_readers.json` | 42bf6f204598c011bfada46e5eabf42dc7759fc6f6750721527d16239141691e | AFF's recorded seed-42 numbers (regression check 3; gitignored) |
| `20261120_r1_levers_brainstorm/results/bs_05_aff.json` | a96719ba505b72883bfa2eeaca66ab872da128314654ba8f74580998b2a72b10 | redundancy values, AFF minus R1 (gitignored) |
| `20261120_r1_levers_brainstorm/results/bs_10_subsets.json` | c9d64815c1c6dba9e84177f7a33fb4e917a326f801ed9b115ae770ce1a1988f1 | random-share control on seed 42 (orientation; gitignored) |
| `20261030_aspect_baselines/run_baselines.py` | 26508dde35f77850c4a80e98d08b0ea6576638577c84d3e96c39660ef02e63c4 | builds the test seeds (§6.2) |
| `20261030_aspect_baselines/results/per_anchor_seed42.npz` | a4818ba0fa5f7249355afe2d2483404dcd34d22cae26984b76be787bb6e9e59d | cosine and RCA per anchor on seed 42 (sensitivity, §6.1) |
| `20261030_aspect_baselines/results/codes_provenance.json` | 8e6a517b73610d4d21b42e2eb39f6fd4118dc4132049364c8e89eecb7716bfaf | rewritten by each `run_baselines.py` build; must stay identical (§6.2) |
| `20261030_aspect_baselines/results/baselines_seed42.json` | ce42c81e8eec256496454e88fc07dc4fcfb02e2d5f1b043f2274ba6008564ce6 | earlier episode SHA-256s (§6.2) |
| `20261030_aspect_baselines/results/baselines_seed43.json` | b250e89caadb1f5ad5937fb36b92ccf1f0f3b63a82b71056dc1ce2b9cc7e2859 | earlier episode SHA-256s (§6.2) |
| `20261030_aspect_baselines/results/baselines_seed45.json` | ecbcf5e900a5f844af9a3e986d1154d2764bb10ba2e2123aba2aaf77917c9890 | earlier episode SHA-256s (§6.2) |
| `20261030_aspect_baselines/results/baselines_seed47.json` | 56e5c0661ad20372cd1c9976d2e48f07c995e44ce33f0649da5cc351f8652386 | earlier episode SHA-256s (§6.2) |
| `20261030_aspect_baselines/results/baselines_seed48.json` | feed925f3eddbf2957b4bce284e1bbf5e9d85132b760f94f6d2c76cb08654286 | earlier episode SHA-256s (§6.2) |

The pickle was written with scikit-learn 1.6.1 and numpy 2.2.6; the run uses the same versions (round 1's
`rb_build.load_readers` refuses another scikit-learn version).

## 4. The seed-parameterised pipeline

One code path, a function of the episode seed s (and of `smoke` for the wiring smoke test, §10), produces everything
below. It is used unchanged on seed 42 (§5) and on the test seeds (§6).

1. **Bundle,** in this call sequence (module paths under `src/test/`; `run_checks` is
   `20261108_new_method_quick_checks/run_checks.py`, `run_n6` the same folder's `run_n6.py`, `run_told_oracle`
   `20261111_community_told_oracle/run_told_oracle.py`):
   - ctx = `run_gonogo.EvalContext(s, smoke)` (episodes, parity halves, anchor paintings, pair index, cosine);
     scorer_train = `artelingo_splits(ctx.data).scorer_train`;
   - T_N1u = `centered_term(run_checks.model_inputs(ctx, "A3", scorer_train, False)[0], ctx.pooled, uniform=True)`;
   - post_E2 = `run_n6.load_posteriors(<n6_posteriors.npz>, ctx)`; B = `crossfit_condition_free(ctx.cos, T_N1u,
     run_n6.n6_terms(post_E2, ctx.pooled)[2], ctx.parity)[0]` (D10);
   - the affect heads = `run_told_oracle.fit_one_head(ctx, run_told_oracle.global_labels(partition_L, scorer_train,
     len(ctx.groups)), scorer_train, run_n6.HEAD_ROWS)`, with the identity with told_oracle.json arm L's head asserted;
     post = {affect: those heads, image: post_E2["image"], caption: post_E2["caption"]} (D2);
   - B′(A0) = `crossfit_condition_free(ctx.cos, T_N1u, uniform_probe_scores(post, ctx.pooled, A0), ctx.parity)[0]`
     (D11);
   - the grouping scores s_h (D4) in both directions (round 1's `common.grouping_stack`); the 18 reader features of both
     conditions (D5, round 1's `rb_eval.seed42_features`);
   - the readers = `rb_build.load_readers("A0", False)`.

   The A3 checkpoint, the head draw (60,000 rows), the posteriors and the readers are the same in every mode: in smoke
   mode only the episode source changes (`EvalContext(s, True)` reads `src/test/20261030_aspect_baselines/results/smoke/`),
   and the smoke flag is never passed to `model_inputs` or `load_readers`. No step of the bundle reads seed-42 arrays;
   `run_sweep.setup` and `common.load_bundle` are called only by the seed-42 comparison of §5 item 1, a separate layer on
   top.
2. **External baselines.** The per-anchor cosine and RCA arrays from `per_anchor_seed{s}.npz` (`run_baselines.py`'s
   output for that seed); asserted: its `anchor_group` and `pair_index` equal the bundle's, and its `cosine__*` arrays
   equal `per_anchor` of the bundle's cosine scores exactly.
3. **Reader.** P^c, T^c, m^c and π^c from the frozen half-readers on the seed's features (D5).
4. **Gates.** R1's and AFF's gates at τ_0..τ_3 (D6).
5. **Families.** For each reader (R1, AFF): the 224 cells (D8), z(T^c) z-scored before the gate, the gated terms, G_cf
   (D9), the integer statistics ρ and γ per cell and episode for the fused reader and ρ for the counterpart, the nested
   control's σ* per tune half (it depends on B only, so it is the same for both readers), the min-margin cross-fit of
   the fused reader and the max-R@1 cross-fit of the counterpart, and the assembled per-anchor arrays.
6. **Records.** Per seed and reader: the chosen cells of both cross-fits on both tune halves (cell number, τ index and
   value, λ_u, λ_a) and σ*; the per-anchor arrays (`r1`, `gain`, `other`, `swap`, `strict`) of the fused reader and the
   counterpart; B, B′(A0), cosine and RCA per anchor. What may be computed before the test verdict is limited by §6.4.

## 5. Seed 42: the regression checks (before any seed is built)

Run on seed 42, the pipeline must pass four checks, in this order. No AFF number is written or printed before checks 1
and 2 pass. If any check fails, no further number is written, and the work stops and goes to the user with the cause
traced.

**Item 1. Bundle.** The pipeline's seed-42 bundle equals round 1's `common.load_bundle()` (which asserts the step-1
arrays itself), exactly: the episodes' anchors, parity, anchor paintings and pair index; cosine scores; B (scores and
per-anchor arrays); B′(A0) (scores and per-anchor arrays); the A0 posteriors (image and caption sides); the grouping
scores s_h in both directions (round 1's `common.grouping_stack` on the bundle); the 18 features of both conditions
(round 1's `rb_eval.seed42_features` on round 1's bundle). The cosine and RCA arrays of `per_anchor_seed42.npz` pass
§4 item 2's assertions. The redundancy check of D7 passes (affect smallest in both directions); the six values are
written.

**Item 2. R1 = round-1 R-c,** as round 2's rule §4.7 states it for its k_top 13 cells:
- R1's T^c equal the stored `T__{a,b}__{i2t,t2i}`, its margins `margin__{a,b}` and its picks `pick__{a,b}`, exactly in
  value (stored picks are int8); τ_0..τ_3 recomputed as `numpy.percentile(m, [0, 25, 50, 75])` of the 24,576 seed-42
  margins (condition a's 12,288 first) equal rc_tau.json's exactly;
- the chosen cells are round 1's: fused cell 116 on tune half 0 (τ_2, λ_u 0, λ_a 2) and 119 on tune half 1 (τ_2, 0, 16);
  counterpart 58 (τ_1, 0, 0.5) and 123 (τ_2, 0.5, 1); control σ* = 0 on both halves;
- the per-anchor arrays `fused__{r1,gain,other,swap,strict}`, `cf__{r1,gain,other,swap,strict}` and `bar_v` equal
  exactly;
- bar margin 0.4435221354166667 [0.21646171563312194, 0.6735669710776852] against the counterpart, gain statistic
  2.667236328125 [2.325087836946873, 3.012361650695922], at full precision.

**Item 3. AFF = the brainstorm's recorded numbers** (`bs_04_readers.json`, key `results.AFF`, and `bs_05_aff.json`,
key `A0.AFF_minus_R1`; D15), at full precision:
- fused R@1 19.136555989583336; counterpart R@1 18.39599609375; bar comparator B′(A0);
- bar margin 0.6998697916666667 [0.4598852740816973, 0.9371680126852968];
- margin against the counterpart 0.7405598958333333 [0.5196896694963071, 0.9598857494832738];
- gain statistic 3.110758463541667 [2.780005709854805, 3.4559584315470384]; either change against the counterpart
  −1.629638671875;
- per-pair bar margins 0.9765625 (emotion × style), 1.45263671875 (emotion × genre), −0.32958984375 (style × genre);
- chosen cells: fused cell 39 on tune half 0 (τ_0, λ_u 4, λ_a 16) and 119 on tune half 1 (τ_2, 0, 16); counterpart
  149 (τ_2, 4, 4) and 10 (τ_0, 0.5, 0.5); σ* = 0 on both halves;
- paired per anchor, AFF minus R1: fused R@1 0.21769205729166666 [0.06425880757348419, 0.3709597330984391], bar margin
  0.25634765625 [0.04280778303598444, 0.46195041633015954];
- AFF's τ_0 gate open on 9,941 of the 12,288 condition-a values and 3,627 of the condition-b values (80.90006510416667%
  and 29.5166015625%), compared as integer counts. `bs_10_subsets.json` (`"H=affect".open_share_tau0`) prints
  80.90006709098816% for condition a, the float32 mean of the same 9,941 values.

The brainstorm scored in float64 and this pipeline scores in float32 (D8). A difference is traced to its cause before
anything else runs (a near-tie whose order differs between float32 and float64 would be such a cause) and goes to the
user, who decides before anything else runs. (The rule check reproduced every number of this item through the float32
path on seed 42, and the float32 and float64 integer statistics agree in all 224 × 12,288 entries for both readers.)

**Item 4. Development bar** (D13) for AFF on seed 42, recorded with the disclosure of §6.11. It selects nothing. If it
fails, which only a code difference could cause, the work stops.

No other variant is computed on seed 42.

## 6. The test (the only confirmatory step)

1. **Sensitivity, after §5 and before the seeds are built** (round 2's rule §6.1). For each GO check of item 5 and for
   the secondary check of item 6, take AFF's seed-42 per-episode difference (the fused reader's R@1 minus the
   comparator's, the fused gain minus 0 for the gain statistic, the fused gain minus RCA's gain, or AFF's fused R@1
   minus R1's) and split its variance by anchor painting with the one-way decomposition: σ_ε² = the within-painting
   mean square, σ_a² = max(0, (between-painting mean square − σ_ε²) / n₀) with n₀ = (n − Σ_p m_p² / n) / (P − 1), where
   n = 12,288 episodes, P the number of anchor paintings and m_p the episodes per painting on seed 42. The projected
   pooled standard error over three seeds is SE² = (σ_a²·(9·Σ_p m_p² − 6n) + σ_ε²·3n) / (3n)² (three independent draws
   of the same anchor distribution, counts approximated as Poisson). Written to the log and to
   `results/sensitivity.json`: SE, the projected half-width 1.96·SE, the **detectable margin x = 2.80·SE** (the true
   margin at which the check's lower bound clears 0 with probability 0.8), and the seed-42 bootstrap half-width (half the width of the
   seed-42 95% interval of the same paired difference) beside them. It is used only to read a failed check (item 7); it never stops the round.
2. **Build.** Seeds 49, 50 and 51 (free in `docs/superpowers/episode_seed_ledger.md`), once each, with
   `src/test/20261030_aspect_baselines/run_baselines.py --episodes-seed <s>`, without `--overwrite` (4,096 episodes per
   aspect pair on selection rows; it also scores cosine and RCA and writes `episodes_seed{s}.npz`,
   `per_anchor_seed{s}.npz` and `baselines_seed{s}.json`). The 9 new per-pair episode SHA-256s must differ from each
   other and from every per-pair SHA-256 in `baselines_seed{42,43,45,47,48}.json`; on a match the test stops and goes
   to the user. `run_baselines.py` also rewrites `results/codes_provenance.json`; its SHA-256 is recorded before and
   after each build and must stay as in D15, otherwise the test stops and goes to the user. Each build's console output
   (it prints every scorer's R@1 table) goes to `results/build_seed{s}.log`, which is not opened before the verdict; the
   build is checked by its exit status and the files it wrote. A build that crashed before `baselines_seed{s}.json` was
   written is repeated once for the same seed after its partial outputs are deleted (the episodes are a deterministic
   function of the seed and nothing has been computed from them); the crash and both runs' episode SHA-256s, which must
   be equal, are logged. "Never rebuilt" (§9) means that a completed seed is never rebuilt or replaced by another seed.
   After the hash check passes, the seed ledger gets its rows: seeds 49, 50, 51 (test, round 3), 52 and later free, and
   9001 to 9003 (smoke, wiring only, 64 episodes per pair, never results).
3. **Frozen from seed 42:** the half-readers with their scalers, τ_0..τ_3, the affect restriction, the 224-cell family
   (λ grid, cell order, tie rules), the standard heads, the recipes of B and B′(A0), the method-A checkpoint. **Rerun on
   each test seed's own parity halves:** B, B′(A0), the reader's probabilities on that seed's features, the gates (that
   seed's margins and picks against the frozen τ), the nested control's σ*, the fused reader's min-margin cross-fit and
   the counterpart's max-R@1 cross-fit over the 224 cells. Per-seed cross-fitting is part of the method's definition;
   the counterpart has the same freedom; the intervals hold the picks fixed.
4. **Order of computation.** On the test seeds, until the verdict is written to `results/test_verdict.json` (with this
   file's SHA-256 and the Amsterdam time), only these are computed and written: the per-seed bundles and reader
   arrays of §4 items 1 to 4 (cached for later use); the per-anchor arrays of AFF's fused reader and counterpart, of B,
   B′(A0), cosine and RCA, and of R1's fused reader (R1's counterpart is not cross-fitted, assembled or written;
   its per-cell statistics may be computed in the same pass and are not read); the chosen cells and σ*
   needed to assemble them; and the pooled quantities of items 5 and 6. The per-seed cosine and RCA summaries that
   `run_baselines.py` prints and writes when it builds a seed are allowed and are not read before the verdict. Per-seed
   and per-pair numbers, the bar margin, pick accuracy, gate shares, R1's counterpart and checks, the frozen-cell line
   and the random-share control come only after the verdict (§7).
5. **GO** if, pooled over the three seeds (§2), every one of these seven checks has a 95% lower bound above 0:
   - R@1, AFF's fused reader minus each of cosine, RCA, B, B′(A0) and AFF's matched counterpart (five checks);
   - the gain statistic (D12), which is also the gain difference against cosine, B and B′, counted once;
   - condition gain, AFF's fused reader minus RCA.
6. **Secondary check (pre-registered; never changes GO):** R@1, AFF's fused reader minus R1's fused reader, paired per
   anchor, pooled. It **passes** if its 95% lower bound is above 0. Its point, interval and pass or fail are written to
   `results/test_verdict.json` with the verdict.
7. **NO-GO** if any of the seven checks fails, including a partial pass. Each failed check is read on its own: if its
   pooled point is above 0, it is **inconclusive at a detectable margin of x** (that check's x from item 1, with the
   realised pooled half-width beside it), not evidence that AFF fails; if its point is at or below 0, it is "AFF did not
   beat <name> on fresh episodes", where <name> is cosine, RCA, B, B′(A0) or the matched counterpart for the R@1 checks,
   "the condition-free comparators on condition gain" for the gain statistic, and "RCA on condition gain" for the last
   check. The NO-GO is reported with every failed check's reading. The secondary check is read the same way, with
   <name> = R1.
8. **Per-seed and per-pair results** are reported after the verdict and never change it; per-pair results are not
   tested.
9. **Frozen-cell line** (descriptive, after the verdict): on each test seed AFF's fused reader and its counterpart
   scored with the cells that seed 42's cross-fits chose (the cell chosen on seed-42 tune half h scores the test seed's
   episodes of parity 1 − h: fused 39 and 119, counterpart 149 and 10), with the same comparators as item 5; the same
   for R1 (fused 116 and 119, counterpart 58 and 123).
10. **Claim licensed.** A GO shows that AFF, a reader on A0 that steers only when it picks the affect grouping, beats
    each comparator, pooled over the three aspect pairs, on new episodes drawn from the same 6,451 selection paintings.
    It does not show transfer to new paintings (the held split stays reserved for the paper test) or a margin on each
    aspect pair. A passed secondary check adds that AFF beats R1 on the same episodes. For the paper: the groupings
    were built without evaluation labels; AFF was selected on seed 42 (item 11).
11. **Multiplicity disclosure** (reported with every AFF number of this round): AFF was found among about 50 label-free
    variants read on seed 42, beside 15 declared oracles (with labels) and 2 controls (brainstorm §6); the median of its one-sided cluster (+0.63, range +0.42 to +0.72) is a
    better guide than its +0.700; the fresh seeds 49 to 51 are the protection; the frozen-cell line accompanies the
    test.

## 7. After the verdict (descriptive; decides nothing)

Computed only after `results/test_verdict.json` is written, from the cached per-seed arrays where possible:

1. Per-seed and per-pair results for AFF (the measures of §6.5 and §6.6 per seed, and per pair pooled over seeds).
2. **AFF's pooled bar margin** (D12, with the bar comparator chosen by the pooled mean R@1 over the 36,864 episodes) and
   per seed; the margin, gain and either rate against the counterpart; AFF minus R1 in bar margin (paired); the chosen
   cells per seed; the frozen-cell line (§6.9).
3. **R1's own seven checks** on the test seeds (its counterpart cross-fit run now), with its bar margin, labelled
   descriptive. R1 never gives a second verdict, even if it passes and AFF fails; the next step is then the user's.
4. Gate-open shares at each τ, overall, per condition and per pair (label-free), for AFF and R1, per seed and pooled.
5. Pick accuracy (D14), overall and per pair and condition, pooled.
6. The redundancy values of D7 on each test seed.
7. **Random-share control** (a mechanism control that reads which condition is a, so never a method). On each test seed
   s and for each draw r ∈ {0, 1}: share_c = (the number of episodes whose AFF τ_0 gate is open in condition c) / E,
   float64, with E the seed's episode count; one generator `numpy.random.default_rng(100·s + r)`; keep^a = 1[u <
   share_a] with u = generator.random(E), then keep^b the same with a second call (condition a first, as the
   brainstorm's `bs_10_subsets.py` drew it). Its
   gates are R1's gates times keep^c at every τ index; the term, the 224 cells, the cross-fit, its own counterpart (D9
   with these gates) and the comparators are as for AFF. Reported per draw, pooled over the seeds: its bar margin, and
   AFF minus the control (fused R@1 and bar margin, paired per anchor). Orientation, seed 42 (brainstorm, generators 0
   and 1 and the float32 share, which this definition does not reproduce): +0.665 [+0.441, +0.894] and +0.564 [+0.332,
   +0.798] against AFF's +0.700.
8. The brainstorm's ideas 2 to 4 are not part of this round; nothing else is computed on the test seeds.

## 8. Order of work, re-derivation and timeline

- **Order:** (1) implementation with unit tests; (2) the seed-42 regression checks (§5); (3) the sensitivity
  projection (§6.1); (4) the end-to-end wiring smoke test on seeds 9001 to 9003 (§10); (5) the three seeds built
  (§6.2); (6) the GO pass on the test seeds (§6.4); (7) the independent re-derivation of the GO quantities; (8) the rule
  applied, the verdict written; (9) the descriptive pass (§7); (10) the whole-branch final review, one fix wave and a
  scoped re-review; (11) the report.
- **Independent re-derivation.** An agent that has not written or read the implementation re-derives, with its own code,
  in two phases. Phase 1 (may start once this file is committed): §5 items 1 to 4 on seed 42 (bundle, R1 = round-1 R-c,
  AFF's numbers, the redundancy values, D13's clauses). Phase 2 (after the GO pass, before the rule is applied): the
  hash check of §6.2; per test seed B, B′(A0), the gates, σ*, the chosen cells of AFF's two cross-fits and of R1's
  fused cross-fit; the seven GO checks and the secondary check (points, bounds, pass or fail). The detectable margins x
  of §6.1 are re-derived in phase 1. The re-derivation may import the data loaders and frozen components
  (`EvalContext`, `run_checks.model_inputs` with `centered_term`, `run_n6.load_posteriors`, `run_told_oracle.fit_one_head`,
  `rb_build.load_readers`, `zscore_rows`, `crossfit_condition_free`, `uniform_probe_scores`, `cluster_bootstrap`). It
  writes its own code for the features, P, T, margins, picks, gates, G_cf, the 224 cells' integer statistics, σ*, both
  cross-fits, the assembly, the per-anchor metrics, the comparisons, the checks and x. It writes only to `rederive/` of
  this folder.
- **Agreement** means: every discrete quantity identical (picks, gates, chosen cells, σ*, the bar comparator, each
  check's pass or fail, the redundancy order); τ equal to rc_tau.json's within 1e-15 absolute or 1e-9 relative;
  redundancy values within 1e-9 absolute; every margin, gain statistic, check point and bound within 1e-9 percentage
  points; every per-anchor array exactly. A difference beyond these is traced to its cause before the rule is applied;
  the computation that follows this file's text settles it, and the user is told.
- **Boundaries.** A check whose lower bound lies within 1e-12 of 0 is reported to the user with both values before the
  verdict is recorded (the strict inequality of §2 still decides).
- **Time-box:** ends Tuesday 13 October. If the test cannot be completed by then, no partial verdict is issued; the user
  is told and decides.

| Day (Amsterdam) | Work |
|---|---|
| Tue 6 Oct | This rule written, checked by a fresh Opus reviewer (findings applied), committed and sent to the user; the plan; implementation by subagents; the seed-42 regression checks; the sensitivity projection; seeds 49 to 51 built; the GO pass; re-derivation; the verdict (target) |
| Wed 7 Oct | Anything left from Tuesday; the descriptive pass (§7); the whole-branch final review and its fix wave; the report |
| by Tue 13 Oct | Time-box ends |

## 9. Outcome-to-action table

| Outcome | Action |
|---|---|
| A script check fails (a SHA-256 of this rule or of a D15 input, round 1's bundle check, §5 items 1 to 4, the redundancy check, an episode-hash match, a change of `codes_provenance.json`, a condition-free or alignment assertion) | Stop that step and report to the user; nothing is improvised |
| §5 item 3 differs from the brainstorm's numbers | Traced to its cause; the user decides before anything else runs |
| The wiring mutation of §10 does not fire an assertion | Stop; no test seed is built; the user is told |
| On the test seeds, before the verdict | Only the quantities of §6.4 |
| All seven checks pass | **GO** within the claim of §6.10, with the disclosure of §6.11; the secondary check is reported with its pass or fail |
| Any of the seven checks fails, including a partial pass | **NO-GO**, read by §6.7 (inconclusive at x, or not beaten); the user decides what follows |
| The secondary check passes or fails | Recorded with the verdict and read by §6.7; it never changes GO |
| R1 passes its own seven checks (§7 item 3) while AFF gives NO-GO, or the random-share control matches or beats AFF | Reported only; no second verdict; the user decides what follows |
| The re-derivation or the final review disagrees with a reported number or verdict beyond the agreement of §8 | Traced to its cause; the computation that follows this file's text settles it; the corrected verdict goes to the user with the cause; the rule is not changed |
| A run crashes, or a bug is found before the rule is applied | Correcting code so that it matches this file is not a change of the rule. A run that crashed before its results file was complete is repeated after its partial outputs are deleted, and the crash is logged. A results file already written is not overwritten: the corrected run writes beside it with the suffix `_fix<n>`, the old file is kept, the log records the cause and the numbers before and after, and the rule uses the corrected file. A completed seed is never rebuilt (§6.2) |
| Any situation this file does not cover | The user decides; this file is not changed after its commit without the user |

## 10. Process

- Scripts of this folder assert this file's SHA-256 and the SHA-256 of every D15 input they read, refuse to overwrite
  non-smoke results, and write only to this folder's `results/` (gitignored by this folder's `.gitignore`, a copy of
  round 2's), except that `run_baselines.py` (§6.2) writes its own outputs for seeds 49, 50 and 51 in
  `src/test/20261030_aspect_baselines/results/` and rewrites `codes_provenance.json` there, the smoke builds of seeds
  9001 to 9003 write to that folder's `results/smoke/`, and the seed ledger gets its rows.
- Module names of this folder start with `r3_` (or `run_r3_`, `test_r3_`), so that none shares a name with a module of
  round 1 or round 2, whose folders are on `sys.path`. Round 1's and round 2's folders are read only: their modules are
  imported by path and not modified, and their functions that write files (round 1's `common.save_candidate`,
  `common.write_json_once`, the `rb_build.py` stages, `run_rc.py`; round 2's runners) are not called.
- **Smoke runs** write to `results/smoke/`, may be overwritten, and are not results. Smoke runs never print or log a
  metric value of any scorer on any seed: the console and the log show only shapes, counts, file names and the pass or
  fail of each assertion. Their files in `results/smoke/` are opened only by the assertions and are deleted when the
  smoke test has passed. They never use the test seeds. The end-to-end wiring smoke test (round 2's final review, N8)
  runs after §6.1 (§8): the GO pass, the rule application (which in smoke mode reads `results/sensitivity.json`) and
  the descriptive pass on the smoke seeds 9001, 9002 and 9003, built with `run_baselines.py --smoke --episodes-seed <s>`
  (64 episodes per pair, written to `src/test/20261030_aspect_baselines/results/smoke/`). It must complete with every
  assertion passing before any test seed is built, and it also runs at least one mutation of the wiring (for example,
  AFF's gated term passed where the counterpart expects G_cf) and shows that an assertion fires. This test is a
  deliberate addition to the spec's seed-42 wiring check (spec §3): the seed-42 run does not exercise the seed build,
  the pooling, the rule application or the descriptive pass.
- CPU only: `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1` (so that importing the
  read-only folders writes no `__pycache__` into them), at most 3 processes, `uptime` and `free -g`
  checked first. The main session launches every real run (`run_in_background`); subagents implement.
- Records: the log `20261121_round3_affect_gate_log.md` in this folder (times from `TZ=Europe/Amsterdam date`); the
  report `docs/reports/auto/v2/2026-11-21_round3_affect_gate.md` with a row in `docs/reports/reports_sum.md` and
  `scripts/check_reports_sum.py` run; `.claude/<yyyymmdd>_log.md` for any edit to an existing source file; the seed
  ledger row. Commits go to `main` without a further request (authorisation above), scoped by explicit path, never
  pushed.
