# Weekly Report — Conditional Buddies Publication Track

**Week of:** 2026-08-19 to 2026-08-26  
**Project:** CoSiR — conditional-buddies initialization / publication track  
**Branch:** `experiment/condition_drift_retrieval_correlation`

---

## 1. Executive summary

This week moved the track from establishing that buddy-init helps relative to the generic initializer to identifying which parts of that construction and its post-init optimization actually matter. Experiment 8 found that the graph's source encoder pair is consequential for i2t but essentially not for t2i; Experiment 9 made RedCaps' aggregate signal more interpretable and showed that its apparent size effect is a lift-normalization artifact rather than content dilution; and Experiment 10 showed that a real typed-edge construction correction does not move downstream retrieval at RedCaps-150k. Most importantly, Experiment 11.1 found that freezing the buddy-initialized table decisively improves held-out oracle i2t, while 11.2 established that the trained table nevertheless has a real, large, drift-graded effect on in-sample rank when isolated from whole-model divergence. Experiment 11.3's coupling toggle, sweep runner, and paired analysis are implemented, but its three-run results are pending in a separate session and are not reported here.

## 2. Objective and scope

The current publication claim remains deliberately narrower than a claim of beating raw CLIP: buddy-graph structure is a robust, content-grounded initializer and a better in-model starting point than the generic alternative. This report covers only developments after the 2026-08-19 progress deck; its earlier signal-validation, C5/C6 initialization comparison, scale result, and raw-CLIP gap are context, not new findings here.

All retrieval ablations below use the project convention of paired-within-seed deltas, the `mean/SEM` read, and the measured ~0.1–0.7 R1 noise floor. Unless specified otherwise, the training operating point is RedCaps-150k, frozen CLIP ViT-B/32, `lr=1e-3`, `lr_label=1e-4`, 16-D conditions, `alpha=0.5`, buddy initialization, 100 epochs, and training-time buddy terms off.

## 3. Experiment 8 — buddy-init encoder-pair ablation

### What was tested

The buddy graph/init source was varied over all 16 cached vision × text encoder pairs (4 × 4), while the frozen CLIP training backbone and all other training settings stayed fixed. Each pair was trained for three seeds (48 runs). Deltas are each non-CLIP pair minus the existing `clip_img:clip_txt` source baseline. The analysis also joined each pair to its already-measured cross-VLM survival rate.

### Key results

`test_oracle/t2i_R1` was effectively invariant to the source pair:

| summary | result |
|---|---:|
| non-baseline pairs | 15 |
| only pair clearing `|mean/SEM| ≥ 2` | `dinov2:minilm`: −0.07, mean/SEM −2.0 |
| interpretation of that exception | below the ~0.1–0.7 R1 noise floor |

`test_oracle/i2t_R1` was materially different:

| encoder pair | mean Δ vs. CLIP source | mean/SEM | wins/3 |
|---|---:|---:|---:|
| `dinov2:minilm` | +2.30 | +6.5 | 3 |
| `dinov2:bge` | +1.97 | +4.5 | 3 |
| `vit_sup:clip_txt` | +1.77 | +4.0 | 3 |
| `clip_img:bge` | +1.43 | +6.6 | 3 |
| `siglip_v:e5` | −0.37 | −0.8 | 1 |
| `vit_sup:e5` | −0.80 | −6.9 | 0 |

Twelve of 15 non-baseline pairs were positive and statistically significant on i2t, all with 3/3 seed wins; 10 of those 12 also exceeded the noise floor in magnitude. The survival-rate join did not validate survival as a usefulness proxy: Pearson `r = +0.010` for t2i and `r = −0.410` for i2t (`n = 15`).

### Verdict

The source encoder pair matters substantially for i2t and the CLIP-sourced default is not the strongest choice in this 150k grid; t2i instead supports the complementary conclusion that downstream usefulness is insensitive to the particular source pair. The survival-rate result closes the deferred Experiment 6 question: it is near-null for t2i and weakly negative, not positive, for i2t. The next needed check is whether the promising i2t pairs also improve `test_pre_diff` and survive a 300k extension.

## 4. Experiment 9 — RedCaps subreddit signal-strength correlates

### What was tested

The aggregate same-subreddit buddy-edge lift was decomposed per subreddit and related to subreddit size, caption diversity, and visual homogeneity. A confidence-oriented enrichment z-score and independently drawn all-350-subreddit samples at 150k, 300k, and 500k were added to distinguish signal effect size from sample-size confidence.

### Key results

At 150k, aggregate lift was **22.80×** across **159 of 350** qualifying subreddits. Individual lift ranged from **4.41×** (`pics`) to **670.79×** (`f1porn`), with median **83.98×** and mean **114.10×**.

| property vs. lift, 150k | Pearson r | Spearman ρ | interpretation |
|---|---:|---:|---|
| size | −0.328 | −0.523 | real curved/rank relationship |
| caption diversity | −0.219 | −0.224 | null |
| visual homogeneity | +0.204 | +0.230 | null |

The size association is not evidence that large subreddits are topically diluted. `Spearman(deg_s, size) = +0.973`, while `Spearman(size, purity) = +0.214` and `Pearson(log(size), purity) = +0.187`: purity rises slightly with size. Lift is mechanically pulled down by the degree-dependent denominator; `Pearson(log(size), lift) = −0.509` and `Pearson(log(size), log(lift)) = −0.583` expose that structural relationship.

| metric | 150k | 300k | 500k |
|---|---:|---:|---:|
| overall lift | 22.80× | 22.74× | 22.79× |
| lift-qualifying subreddits | 159 (45%) | 197 (56%) | 214 (61%) |
| z-qualifying subreddits | 124 (35%) | 157 (45%) | 185 (53%) |
| lift-size Spearman ρ | −0.523 | −0.532 | −0.590 |
| z-size Spearman ρ | +0.221 | +0.163 | +0.226 |

### Verdict

C1's aggregate signal is stable over three independent samples, while caption diversity and visual homogeneity do not explain where it is strongest. The negative lift–size pattern is a property of lift's degree-normalized construction, not a content-level claim that small niche subreddits inherently have stronger buddy signal. The positive z–size relation is the expected confidence effect and independently confirms this interpretation. A 1M/3.1M coverage check and a joint model of the three properties remain open.

## 5. Experiment 10 — buddy distance-mode (typed-edge) ablation

### What was tested

The prior fixed image/text distance blend uses both modalities for every union-graph edge, even though ~98% of edges are supported by only one modality. `distance_mode="typed"` instead uses the supporting modality's own rank for image-only/text-only edges; `blend` remains the default. The ablation was 2 modes × 3 seeds.

### Key results

| paired metric, typed − blend | mean Δ | ± std | mean/SEM | typed wins |
|---|---:|---:|---:|---:|
| `test_oracle/t2i_R1` | +0.00 | 0.26 | +0.0 | 2/3 |
| `test_oracle/i2t_R1` | −0.37 | 0.59 | −1.1 | 1/3 |
| `test_pre_diff/t2i_R1` | +0.00 | 0.10 | +0.0 | 1/3 |
| `test_pre_diff/i2t_R1` | +0.17 | 0.55 | +0.5 | 2/3 |

`test_raw` was identical across all six runs: `t2i_R1 = 28.1`, `i2t_R1 = 29.7`. This is not a barely-changed-init null: mean `|typed − blend| = 0.5117`, **1.02×** the mean absolute embedding value (0.5000); per-dimension correlation ranged from **+0.985** to **−0.996**, with mean **+0.240**.

### Verdict

The graph-level flaw is real, its correction substantially changes the saved initializer, and yet retrieval and the gap to CLIP are unchanged within the project's significance/noise criteria at this scale. The informative conclusion is that buddy-derived structure itself appears more load-bearing than this fine edge-weighting choice; this is a citable null, not evidence that the diagnostic or fix was invalid.

## 6. Experiment 11.1 — condition freeze ablation

### What was tested

Post-init condition-table training (`em_interval=-1`) was compared with a fully frozen table (`em_interval=101`, beyond the 100-epoch budget), holding buddy init, backbone, loss stack, and all other settings fixed. There were two arms × three seeds. The geometry diagnostic was recomputed using the raw, un-normalized features that the scale-sensitive combiner actually receives in training/evaluation.

### Key results

| metric, frozen − trained | mean Δ | mean/SEM | per-seed Δ | interpretation |
|---|---:|---:|---|---|
| `test_oracle/t2i_R1` | −0.27 | −2.0 | −0.40 / −0.40 / +0.00 | flagged but inside noise floor |
| `test_oracle/i2t_R1` | +4.67 | +32.1 | +4.90 / +4.40 / +4.70 | decisive frozen win |

The freeze sanity check passed: frozen drift from init was exactly 0 for all three seeds. Trained drift had mean **0.0845**, range **[0.0834, 0.0855]**.

| final-epoch geometry, trained − frozen | mean Δ | mean/SEM | reading |
|---|---:|---:|---|
| `shift_mean` | +0.0081 | +13.9 | trained shifts the combine-side embedding further |
| `shift_std` | +0.0079 | +31.9 | trained broadens the shift distribution |
| row diversity | +0.0141 | +15.9 | conditions matter more across a fixed image |
| column diversity | −0.0127 | −10.2 | less across-image diversity at a fixed condition |
| conditioned effective dimensions | 0 | — | exactly 301 in all six runs |

The most-changed-set cross-arm Jaccard values at epoch 99 (0.176, 0.176, 0.143) fall inside the within-arm cross-seed range (0.111–0.250), so that ranking does not distinguish the arms.

### Verdict

Post-init table training is not inert, but the operational result is direction-asymmetric: freezing decisively beats continued training for held-out oracle i2t while t2i is a noise-floor null. Training makes the condition effect more differentiated and shifts embeddings slightly further, yet that does not translate to better held-out oracle i2t. The result cleanly opened 11.2's diagnostic gate. The raw-feature correction invalidated earlier effective-dimension and Jaccard claims, but it did not affect retrieval, the shift distribution, or the grid-diversity split.

## 7. Experiment 11.2 — drift/shift versus retrieval-rank correlation

### What was tested

This no-new-training diagnostic reused the 11.1 checkpoints. For each seed pair, 3,000 in-sample queries used their own assigned condition and a full 150,000-item training gallery. Crucially, it separates a naive cross-arm rank delta from a condition-only counterfactual that holds the trained combiner, `other_proj`, and gallery fixed while replacing only the trained condition table with buddy init.

### Key results

| statistic (Spearman ρ; n = 3000 per seed) | seed 1 | seed 2 | seed 3 |
|---|---:|---:|---:|
| `rho(delta_rank, drift)` | +0.026 | +0.017 | +0.009 |
| `rho(|delta_rank|, drift)` | +0.160 | +0.141 | +0.156 |
| `rho(|delta_rank|, shift)` | −0.288 | −0.296 | −0.302 |
| `rho(delta_rank_swap, drift)` | +0.466 | +0.466 | +0.477 |

For the counterfactual, all three `rho(delta_rank_swap, drift)` values have **p < 1e−160**. The corresponding mean `delta_rank_swap` values are **+343.5 / +352.0 / +357.1**, with medians **+6 / +7 / +7**: reverting high-drift samples to init conditions costs rank inside the fixed trained model.

On this diagnostic's own in-sample, own-condition metric, the trained arm wins:

| seed | mean rank frozen / trained | R1 frozen / trained | fraction improved |
|---:|---|---|---:|
| 1 | 1118.1 / 975.0 | 0.0923 / 0.1007 | 0.636 |
| 2 | 1122.8 / 980.4 | 0.0880 / 0.0997 | 0.640 |
| 3 | 1112.5 / 996.2 | 0.0893 / 0.1033 | 0.639 |

### Verdict and scope boundary

The condition table has a real, large, drift-graded per-sample effect on rank; the near-zero signed cross-arm correlation is an estimator problem, because its standard deviation (**1055–1343** ranks) is dominated by independently trained combiner/`other_proj` divergence. Drift predicts how much a rank moves, rather than its direction, and high-shift samples are comparatively stable across arms.

This does **not** contradict or explain 11.1's frozen-wins result. Experiment 11.1 measures **held-out**, **oracle-max-over-conditions** i2t retrieval; 11.2 measures **in-sample**, **own-condition** rank for rows with learned per-sample conditions. Those are two distinct construct mismatches. The in-sample trained advantage therefore constrains the interpretation—training can be useful for assigned training rows—without resolving why the trained condition-table codebook generalizes worse under held-out oracle retrieval.

## 8. Experiment 11.3 — bidirectional table↔predictor coupling

### What is implemented

Experiment 11.3 tests whether the predictor-distillation loss should remain one-way. The new `loss.pred_stopgrad` flag defaults to `True`, preserving current behavior: the condition table is detached and gradients flow table → predictor. With `loss.pred_stopgrad=false`, the same loss also pulls the trainable table toward what `condition_predictor` can represent. The `pred_coupled` arm uses the trained-table regime and will run three RedCaps-150k seeds against 11.1's existing trained/frozen arms, sharing the same buddy-init template and wandb group.

The repository confirms the implementation boundary: the toggle was added in `038f956`, the sweep runner in `be0efb7` (one file, 85 insertions), and paired analysis in `c821513`. The requested `git log --oneline -10` has **no results commit**.

### Status and planned readout

**Results are pending and must not be inferred from 11.1/11.2.** The full sweep is being run in a separate parallel session. When complete, the analysis will report paired `pred_coupled − trained` and `pred_coupled − frozen` deltas for `test_oracle` and `test_pre_diff` in both directions, with drift-from-init and predictor loss as diagnostic context. The pre-registered interpretations are: a seed-replicated above-noise recovery toward/past frozen is a real signal and gates a `lambda_pred` strength sweep; flat results versus trained are a clean null; a worse primary metric versus trained is reportable instability.

## 9. Synthesis — what 11.1 and 11.2 jointly say

The results make sense together once the two evaluation constructs are kept separate. On the metric that matters for the current held-out codebook claim—11.1's held-out, oracle-max i2t—**frozen beats trained by +4.67 R1**. That says continued table optimization produces a condition-table collection that generalizes less well when held-out queries may select their best table entry.

At the same time, 11.2 shows that drift is not arbitrary noise. Inside the trained model, with every component except the table held fixed, reverting a high-drift condition toward buddy init causes a correspondingly large rank cost (`rho = +0.466 / +0.466 / +0.477`). The trained arm also wins its own in-sample, own-condition task. Thus, post-init training learns useful sample-specific co-adaptation for assigned training rows, but that is not the same thing as learning a table/codebook that transfers to unseen queries under an oracle lookup. The held-out-vs-in-sample and oracle-vs-own-condition mismatches are why the two aggregate directions cannot be treated as competing estimates of one quantity.

This is precisely the motivation for 11.3: it asks whether making the table more feature-predictable can preserve useful trained-table structure while reducing the apparent generalization failure. It is a targeted next test, not a conclusion about its efficacy before the pending runs finish.

## 10. Next steps

1. Complete and analyze Experiment 11.3's three `pred_coupled` runs; report `test_oracle`, `test_pre_diff`, drift, and predictor-loss results without extrapolating from the current implementation commits.
2. For Experiment 8, test the strongest i2t source pairs on `test_pre_diff`; prepare the optional 300k extension only if the held-out feature re-extraction is justified.
3. For Experiment 9, extend the coverage analysis to 1M/3.1M and, if explanation rather than coverage is the goal, fit a joint rather than property-by-property model.
4. Treat Experiment 10 as a completed null at 150k; only revisit typed edges if a scale or operating-point rationale emerges.
5. For Experiment 11, separate future claims by construct: retain frozen condition tables as the held-out-oracle i2t winner at this operating point, and avoid using the in-sample condition-only diagnostic as an explanation of that held-out result. A held-out-reachable predictor-based diagnostic would be required to bridge them.

## Appendix — source artifacts

- Prior weekly structure: [2026-06-09 weekly report](2026-06-09_weekly_conditional_buddies.md)
- Prior weekly framing/deck: [2026-08-19 progress deck](2026-08-19_weekly_conditional_buddies_slides.md)
- Experiment 8: [buddy-init encoder ablation](2026-08-24_buddy_init_encoder_ablation.md)
- Experiment 9: [RedCaps subreddit signal correlates](2026-08-24_redcaps_subreddit_signal_correlates.md)
- Experiment 10: [buddy distance-mode ablation](2026-08-24_buddy_distance_mode_ablation.md)
- Experiments 11.1 and 11.2: [condition freeze ablation and drift/rank analysis](2026-08-25_condition_freeze_ablation.md)
- Master status and gates: [publication-plan design](../superpowers/specs/2026-08-04-buddy-publication-plan-design.md)
- Experiment 11.3 implementation plan: [pred-stopgrad ablation plan](../superpowers/plans/2026-08-26-pred-stopgrad-ablation.md)
