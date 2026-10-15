# CoSiR v2: next step after the A′ repair and the 8B probe: three quick checks (draft for review)

**Date:** 2026-10-04. **Status:** draft for the user to check; nothing in it has been run or committed as a rule.
**Scope:** what to run next to decide whether a new method is worth building before the CVPR abstract (Nov 10), or
whether the project moves to the analysis paper (branch 3).

## 1. The problem in brief

**Task (spec §3).** A *query* (an image or a caption of an artwork) comes with a *condition* made of example pairs:

- 4 **support pairs**, each an image of one painting with a caption of another painting (*cross-item*) that share a value
  of the wanted *aspect* A (for example both "sad", or both "Impressionism"). The four pairs show four different values
  of A, and **never the query's own value** (*value-disjoint*).
- 4 **contrast pairs** built the same way for another aspect B.

The system ranks 13 *candidates* in the other modality: p_A shares the query's value of A, p_B shares its value of B, and
11 negatives share neither. Swapping supports and contrasts must move p_B to the top. On ArtELingo the aspects are
emotion (8 values), style (23) and genre (10); the three aspect pairs are pooled.

**Metrics** (per ranking, averaged over both conditions and both directions, image→caption and caption→image):

- **R@1:** the target is ranked strictly first (a tie is a miss).
- **Other-aspect rate:** the other aspect's candidate is ranked first.
- **Condition gain** = R@1 − other-aspect rate. It is exactly 0 for any scorer that ignores the condition.
- **Either rate** = R@1 + other-aspect rate: how often *some* aspect-sharing candidate is first. Hence
  **R@1 = (either rate + gain) / 2**: a scorer can raise R@1 by finding aspect-sharing candidates more often, or by
  choosing the conditioned one more often.
- Uncertainty: 95% intervals from a bootstrap that resamples whole anchor paintings (5,000 resamples).

**Data rows.** Training uses the 183,694 *scorer-train* rows; development episodes use the 32,413 *selection* rows
(6,451 paintings; the two sets share no painting). Val and held rows are not used.

## 2. What has been tried, and what it showed

| Approach | Result (development or test episodes, 12,288 per draw) | Report |
|---|---|---|
| CLIP ViT-B/32 cosine (*backbone only*) | R@1 12.96, gain 0 (seed 42); 13.53 (seed 43) | E1 |
| Nine raw-feature "metric from pairs" baselines (KISSME, RCA, Xing, CVS-style per-query weights, a pair probe, …); best one is **RCA**, the *GO bar* | RCA R@1 13.38, gain 0.10 (seed 42); none separated the aspects | E1 |
| **Method A**: a shared sparse image-text *factor basis* (32 non-negative factors per item) trained on pseudo-aspect episodes; the *agreement rule* reads the condition; picked run **A3** | Test: R@1 13.76, gain 0.26 [−0.04, 0.56]; its *uniform-weight control* (same factors, condition removed) R@1 **16.72**. NO-GO | E3 |
| Why A failed (post-hoc) | The uniform factor term finds aspect-sharing candidates often (either 33.4 vs cosine 27.1); the condition-weighted term selects the aspect a little (gain ≈ 1.0 alone) but finds aspect candidates *less* often than cosine (either 21.4), so any weight on it costs more R@1 than it adds | E3 §6.1 |
| **A′ nested score** z(cos) + λ_u·z(uniform term) + λ_a·z(weighted term) on A3 | R@1 16.52 vs its condition-free control 16.55; gain −0.01: the two terms do not add | A′ diagnostics |
| Factors trained on the **true labels** (diagnostic) | Weighted term's gain about doubles (1.85 vs 0.99) but under the nested score it vanishes again (≤ 0.05) | A′ diagnostics |
| **Qwen3-VL-8B** given the examples in context | R@1 +1.07 [0.17, 1.93] over cosine but gain +0.21 [−0.51, 0.94]: finds aspect candidates more often, does not select the demonstrated aspect | 8B probe log |
| Reference: label probes **told the aspect** (aspect-episode spike) | pooled R@1 23.09 vs CLIP 11.13 (emotion and style episodes) | spike |

**The common failure.** Every scorer that reads the condition either finds aspect-sharing candidates *less* often than a
condition-free score (factor methods) or does not select the conditioned aspect at all (8B MLLM). To pass a GO, a method
must beat its own condition-free control, which in the factor setting has R@1 about 16.5 to 16.7 because it keeps the
high either rate.

## 3. Candidates and the literature check

Six candidate methods (N1 to N6) were drafted from this evidence and each was checked against the literature with ARS
deep-research (one verified scan per candidate, then a synthesis): `src/test/20261107_new_method_candidates/`
(`candidates_draft.md`, `scan_N1.md` … `scan_N6.md`, `synthesis.md`). Every candidate "partially exists" (each ingredient
is published, the combination for this task was not found within the scans' search bounds). The synthesis dropped the
learned meta-conditioner (N4) for this window, demoted the concept-basis (N3) and MLLM-naming (N5) ideas behind cheap
privileged references, and proposed the order used here: a diagnostic **D0**, then **N1** and **N2**, then **N6** only if
D0 passes.

## 4. The three checks

All three run on CPU in minutes, on the **existing development episodes** (E1's seed-42 episodes,
`src/test/20261030_aspect_baselines/results/episodes_seed42.npz`) with existing checkpoints:

| Name | What it is | Where |
|---|---|---|
| **A3** | E3's picked factor model (pseudo-aspect episode loss, λ_aspect 3, trained on affect, image and caption k-means partitions of scorer-train rows) | `src/test/20261101_aspect_factor_gonogo/checkpoints/A3_seed42.pt` |
| **C0** | the same factor recipe with no condition training (reference) | codes `src/test/20261030_aspect_baselines/results/codes_C0.npz` |
| **SE** | C0 plus value-condition episodes (GoEmotions affect clusters, image clusters) | codes `src/test/20261030_aspect_baselines/results/codes_SE.npz` |
| **L3, LT** | factors trained on the **true labels** (A′ diagnostic; never a candidate) | `src/test/20261105_method_repair_diagnostics/checkpoints/` |

### 4.1 D0: can the aspect be read from 4 example pairs at all? (diagnostic, not a method)

- **Why.** The label-probe reference (23.09) was *told* which aspect the condition is about. Nothing so far has tested
  whether that aspect can be *inferred from the pairs* when items are well represented. This one test separates
  "the representation is the bottleneck" from "reading the condition from 4 pairs is the bottleneck".
- **Representation.** The spike's label probes (`src/test/20261023_aspect_episode_spike/aspect_ceiling.py`): one logistic
  regression per aspect and per modality (image, caption), on normalised CLIP features of 60,000 scorer-train rows with
  their labels. The spike fitted emotion and style; genre is added the same way. Every image or caption gets three posterior vectors, p_h(·), one per aspect h.
- **Told scorer.** score(query, candidate) = p_A(query) · p_A(candidate) on the conditioned aspect A (each item scored
  with its own modality's probe).
- **Inferred scorer.** For each aspect h, the support pairs' within-pair agreement S_h = mean over the 4 support pairs of
  p_h(image_i) · p_h(caption_i), and the same over the contrast pairs, C_h. Support pairs agree on A and differ on B by
  construction, contrast pairs the reverse, so Δ_h = S_h − C_h should be largest for A. *Hard:* pick the aspect with the
  largest Δ_h and score as in Told. *Soft:* weight the three aspects by max(Δ_h, 0) (normalised) and sum the Told scores.
- **Report.** R@1, gain and either rate for Told, Inferred (hard and soft) and cosine; how often the inferred aspect is
  the true one.
- **Reading.** Inferred close to Told: the condition *can* be read from 4 pairs once values are well represented, so the
  bottleneck is the representation (N6 becomes worth building). Inferred far below Told: reading the condition is itself
  the limit, which sinks N1, N3 and N6 together.
- **Caveat.** It reads evaluation labels on training rows, so it is a diagnostic only and never a method.

### 4.2 N1: a centered agreement rule on the existing factors

- **Why.** The current agreement rule weights factor l by mean over supports of a_I,l(image)·a_T,l(caption) minus the same
  over contrasts: an *uncentered* moment that mostly rewards factors that are active in general. Aspect A should instead
  appear as image and caption codes **varying together across the four support pairs** (each pair a different value of A).
- **Rule.** w_l = max(cov_S(l) − cov_C(l), 0), normalised to sum 1, where cov_S(l) is the covariance across the 4 support
  pairs between the image code and the caption code of factor l (cov_C the same over contrasts). Score with centered
  codes: Σ_l w_l (q_l − μ^q_l)(c_l − μ^c_l), with μ^c the mean over the 13 candidates and μ^q the mean over the 8 example
  items of the query's modality.
- **Run on:** A3, C0, SE (L3, LT as diagnostics only).
- **Compared with:** the current agreement rule; diagonal KISSME on the same codes (per factor, 1/var over supports minus
  1/var over contrasts of the image-minus-caption code difference; the closest published rule, Köstinger et al. 2012);
  cosine.
- **Report:** the term alone (R@1, gain, either rate) and inside the nested score, with the same cross-fitting as A′
  against its condition-free control.
- **Stop if:** the term alone gains no more than the current rule, or its either rate stays below cosine's.

### 4.3 N2: find first, then select

- **Why.** In the A′ profile the condition term hurt because it promoted negatives over the aspect-sharing candidates.
  Inside a short list that already contains p_A and p_B, it would only have to choose between them.
- **Rule.** Rank by A3's condition-free score (its nested uniform control); keep the top k (k = 2, 3, 5); reorder only
  those k by A3's agreement term (and by N1's term if N1 survives).
- **Report:** first, how often both p_A and p_B are in the top k (this caps any gain); then R@1, gain and either rate per
  k against the unreordered ranking.
- **Stop if:** both aspect candidates rarely reach the top k, or the reordering adds no gain.

## 5. One decision after the checks

| Result on the development episodes | Next step |
|---|---|
| N1 or N2 beats its **own condition-free control** on both R@1 and gain (95% lower bounds above 0) | Fix that configuration and test it on **3 fresh episode seeds** (45, 47, 48; selection rows, 4,096 episodes per pair each), reporting each seed and the pooled result, against cosine, RCA and its condition-free control. GO = both R@1 and gain above all three (pooled 95% lower bounds above 0), written down before the test episodes are built |
| N1 and N2 fail; D0 Inferred close to Told | Build N6 (cross-modal classifier heads on the label-free k-means partitions, the aspect picked by D0's rule), run the same checks, then the same test |
| N1 and N2 fail; D0 Inferred far below Told | Stop method work; move to branch 3, with D0, N1 and N2 reported as analysis results |

("Close to" and "far below" for D0 get numbers before D0 is run; proposal: Inferred keeps at least half of Told's gain
over cosine.)

## 6. Cost and order

1. Commit this decision rule (with D0's numeric threshold).
2. One implementation task: the three scorers (D0 told/inferred, N1, diagonal KISSME on codes, N2 cascade) with unit
   tests, reusing `src/eval/aspect_scorers.py`, `src/eval/aspect_nested.py`, `src/eval/aspect_metrics.py` and the E3
   runner's episode loader (`src/test/20261101_aspect_factor_gonogo/run_gonogo.py`, `EvalContext`). About half a day
   with review.
3. Three CPU runs of a few minutes each, then the decision table.
4. If a configuration passes: write its GO rule, build the three test seeds, test once per seed, report.

## 7. Sources

- Spec: `docs/superpowers/specs/2026-10-02-cosir-v2-cvpr-publication-plan-design.md` (§3 task, §4 branches, §15 A′).
- E1 baselines: `docs/reports/auto/v2/2026-10-30_aspect_baselines.md`.
- E3 go/no-go: `docs/reports/auto/v2/2026-11-01_aspect_factor_gonogo.md`.
- A′ diagnostics: `docs/reports/auto/v2/2026-11-05_method_repair_diagnostics.md`.
- 8B probe: `src/test/20261106_mllm_probe_8b/20261106_mllm_probe_8b_log.md`.
- Aspect-episode spike (label-probe reference): `docs/reports/auto/v2/2026-10-23_aspect_episode_spike.md`.
- Candidates and literature check: `src/test/20261107_new_method_candidates/`.
