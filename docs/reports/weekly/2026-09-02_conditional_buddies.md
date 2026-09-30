# Weekly Report — Conditional Buddies Publication Track

**Week of:** 2026-08-26 to 2026-09-02
**Project:** CoSiR — conditional-buddies initialization / publication track
**Branch:** `experiment/condition_drift_retrieval_correlation`

---

## 1. Executive summary

This was the busiest week of the track so far, and it resolved the two biggest open questions left by last week's report while opening two new, promising axes. First, the mystery behind Experiment 11.1's headline i2t-vs-t2i freeze-ablation asymmetry was chased down and substantially explained: Experiment 12.3 traced part of a related subgroup effect to which side the combiner conditions, the `combine_side="txt"` replication confirmed that mechanism but showed the main asymmetry only shrinks (~11×) rather than flips, and Experiment 13 then showed that removing the combiner's one-sidedness altogether collapses the entire headline effect to a noise-floor null — C9's "post-init training regresses i2t" finding is now best read as substantially an architectural artifact, not a generic property of buddy-graph training. Second, Experiment 12's "false transitivity" concern (bridge nodes get pulled together even without a real edge) was given a genuine positive control in Experiment 14: the buddy embedding does discriminate a real edge from a genuine non-edge, by a stable ~31%, and a same-day addendum showed a clean dose-response ordering across all edge types. Experiment 15 then attributed that erosion mechanistically — ordinary retrieval training erodes discrimination on its own even with all buddy-specific losses off, and only the static (non-refreshed) contrastive term recovers a small, real fraction of it, a result that reprioritized the whole 15.x sub-plan and motivated the now-approved 15.4 hub-repulsion mechanism. Separately, two new investigative tracks opened and delivered results: Experiment 16 found that mutual-KNN K should scale sublinearly with dataset size, and Stage B confirmed this structural prediction against real retrieval numbers — 500k/K=39 (the invariant-matched prediction) beats the project's universal K=30 default by +0.97 i2t R1 (mean/SEM +14.5), the cleanest single result of the week. And a literature-grounded combiner-architecture search found that fusion *family*, not buddy dimensionality, is what actually matters: a rank-16 low-rank residual adapter beats the production combiner on oracle i2t (+3.1 R1) while staying comparatively robust on the deployment-tier metric, where two other literature-motivated alternatives collapse.

## 2. Objective and scope

The publication claim remains unchanged from last week: buddy-graph structure is a robust, content-grounded initializer and a better in-model starting point than the generic alternative — not a claim of beating raw CLIP. This report covers only work after the 2026-08-26 report; that report's own content (Experiments 8–11.3) is context, not repeated here.

Unless noted otherwise, all retrieval ablations below use the project's paired-within-seed deltas, the `mean/SEM` significance read, and the measured ~0.1–0.7 R1 noise floor, at the RedCaps-150k operating point (frozen CLIP ViT-B/32, `lr=1e-3`, `lr_label=1e-4`, 16-D conditions, `alpha=0.5`, buddy initialization, `combine_side="img"` unless stated, 100 epochs).

## 3. Experiment 12 — cross-modal bridge / false-transitivity diagnostic (+ 12.2, 12.3)

### What was tested

![The bridge pair (A, B, C): A is the bridge/hub node, B is its real image-only neighbor, C is its real text-only neighbor, and the dashed red line marks the never-real B–C connection that the trained embedding nonetheless pulls together ("false transitivity").](../assets/diagrams/exp12_bridge_abc.png)

Every RedCaps-150k node was labeled by cross-modal mutual-kNN disagreement (`img_only_only` / `txt_only_only` / `bridge` / `neither`), and bridge-node (B, C) pairs — never directly connected in either modality — were checked for whether the trained embedding pulls them together, and whether that pull is explained by real shared-neighbor overlap ("false transitivity"). A first cross-reference against retrieval-rank change used one seed's unsigned `|delta_rank|`; Experiment 12.2 then audited whether the checkpoint used for that (epoch 99) was even representative, and Experiment 12.3 re-ran the cross-reference with a signed measure pooled across 6 independently-trained runs.

### Key results

Bridge structure is pervasive: **80.2%** of RedCaps-150k nodes are bridge nodes. Pull is large and reliable — mean **+1.98** (mean/SEM +102.1, 91.4% of 5,000 sampled pairs pulled closer than baseline) — but barely explained by real shared-neighbor Jaccard overlap (`rho = +0.076`, p = 8.6e-8, rho² ≈ 0.6% of variance): the "false transitivity, not real+graded" branch.

Experiment 12.2's trajectory audit found the i2t deficit from 11.1 is **late-onset but saturating**, not an ongoing collapse: near-zero at epoch 0/10, declining steadily to roughly −4.2 to −5.3 R1 by epoch ~40–50, then flat through epoch 99 across all 6 already-completed `trained`/`pred_coupled` runs. Epoch 99 is therefore a fair representative of the settled effect, not a mid-collapse snapshot.

Experiment 12.3's signed, 6-run-pooled replication overturned the original single-run null:

| statistic | value |
|---|---:|
| `corr(is_polysemic, delta_rank)`, pooled | mean rho=−0.025, mean/SEM=**−9.9** |
| `corr(is_polysemic, |delta_rank|)`, pooled | mean rho=+0.017, mean/SEM=**+8.3** |
| sign agreement across all 6 runs | 6/6 |

The effect concentrates almost entirely in `img_only_only` nodes (14.5% of the graph): median signed `delta_rank` **−17 to −28** across all 6 runs, versus `bridge` nodes' tight **−3 to −4**. A second candidate mechanism was flagged at the same time: every run uses `combine_side="img"`, so "trained vs. frozen" is by construction only ever a difference in the image-side representation — which would predict exactly this pattern (i2t moves, `img_only_only` moves most) for architectural rather than graph-topological reasons.

### Verdict

Experiment 12's original "behaviorally inert" framing does not survive at full strength: the bridge-pair pull is real, large, and poorly explained by shared-neighbor structure, and it has a small but real, replicated behavioral signature concentrated in one 14.5%-of-graph subgroup. The two candidate mechanisms (graph-topology vs. `combine_side`-architecture) were left unresolved here and are exactly what Experiment 13 and the `combine_side="txt"` replication in Section 4 disambiguate. Nothing here shows this mechanism explains 11.1's much larger i2t gap.

## 4. `combine_side="txt"` replication — disambiguating the two candidate mechanisms

### What was tested

Experiment 11.1's exact design (`trained` vs. `frozen`, 3 seeds) was re-run with `model.combine_side=txt` (condition fused into the text embedding instead of image), reusing the same buddy-init template. Three questions were asked in one pass: does the i2t-vs-t2i asymmetry flip to t2i (as the architectural hypothesis predicts), does the `img_only_only` bridge-subgroup effect flip to `txt_only_only`, and does the trajectory shape change.

### Key results

| | `combine_side="img"` (11.1) | `combine_side="txt"` (this test) |
|---|---:|---:|
| t2i mean Δ (frozen−trained) | −0.27, mean/SEM=−2.0 (noise floor) | +0.13, mean/SEM=+0.3 (null) |
| i2t mean Δ (frozen−trained) | **+4.67, mean/SEM=+32.1** | **+0.40, mean/SEM=+4.0** |

The asymmetry did **not** flip to t2i — i2t stays the significant direction — but shrank to roughly **1/11th** of the img-combine magnitude, right at the noise floor's edge, with a noisy, non-saturating trajectory rather than 12.2's clean decline-then-plateau.

The bridge-subgroup effect, by contrast, **flipped cleanly in all 3 seeds**:

| label | img-combine median Δrank (12.3) | txt-combine median Δrank (this test) |
|---|---:|---:|
| `img_only_only` | −17.0 to −28.0 | −1.0 to −2.0 (matches `bridge`) |
| `txt_only_only` | −3.0 to −5.0 | **−3.0 to −10.0** (outsized) |
| `bridge` | −3.0 to −4.0 | −2.0 to −2.0 |

### Verdict

The two candidate explanations from Experiment 12.3 split apart: the bridge-subgroup concentration effect **is** `combine_side`-driven and mirrors exactly as predicted, while the headline i2t-vs-t2i asymmetry is **not** simply a `combine_side` artifact — `combine_side="img"` clearly *amplifies* it (~11×) but does not appear to *create* it from nothing. This is the direct motivation for Experiment 13.

## 5. Experiment 13 — symmetric conditioning (resolves C9's asymmetry)

### What was tested

Rather than flipping which side the combiner favors, this experiment removed the architectural asymmetry itself: **Option A**, a single shared condition table and a single tied-weight combiner applied to *both* image and text embeddings, with every auxiliary loss term symmetrized. Implemented on `experiment/two_side_conditioning` (promoted docs-only to this branch); 3 seeds × {trained, frozen}, matched to every other 11.x/12.x operating point.

### Key results

| Metric tier | Direction | Option A (symmetric) | txt-combine (§4) | img-combine (C9, default) |
|---|---|---:|---:|---:|
| Primary (`test_coupled_oracle`) | t2i | +0.10, mean/SEM=+0.5 (n.s.) | +0.13, mean/SEM=+0.3 (n.s.) | −0.27, mean/SEM=−2.0 (noise floor) |
| Primary (`test_coupled_oracle`) | i2t | **+0.53, mean/SEM=+0.8 (n.s., 2/3 wins)** | +0.40, mean/SEM=+4.0 (sig, 3/3) | **+4.67, mean/SEM=+32.1 (sig, 3/3)** |
| Deployable (`test_two_sided_predictor`) | i2t | −0.40, mean/SEM=−1.6 (n.s., 0/3 wins) | — | — |

The pre-registered asymmetry contrast `A = Δ_i2t − Δ_t2i`: **img-combine +4.94 → txt-combine +0.27 → symmetric +0.43**. Both primary and deployable metrics land inside the noise floor, and the deployable-tier i2t delta is not even sign-consistent with the primary tier's — the signature of noise, not a real directional effect.

### Verdict

C9's headline i2t freeze-vs-trained effect collapses to a noise-floor null under symmetric conditioning — a stronger disconfirmation than the txt-combine replication's ~11× shrinkage. **C9 ("continued post-init condition training regresses i2t retrieval") should no longer be read as a generic property of buddy-graph training dynamics**; the evidence chain C9 → §4 → Experiment 13 now points to this being substantially an architectural artifact of the current one-sided combiner design. The mechanistic *why* (gradient asymmetry into the shared table, or `other_proj`'s identity-init pathway never adapting) remains open, and only Option A (the smallest matched-capacity symmetric variant) was tested.

## 6. Experiment 14 — closed-triangle positive control for Experiment 12's false-transitivity finding

### What was tested

![Side-by-side positive control: the genuinely-unconnected control (left, C and D share hub A but have no direct edge of any kind) versus the closed-triangle positive control (right, C and D share hub A and also have a real image-only edge) — the positive control pulls 1.31× harder.](../assets/diagrams/exp14_positive_control_abcd.png)

Experiment 12 never had a positive control: every bridge pair it measured was, by construction, never directly connected. Experiment 14 sampled **closed-triangle** hub pairs (two of a hub's text-only neighbors that are *also* directly connected by a real image-only edge) against **genuinely-unconnected** pairs (no edge of any kind), pooled across 3 sampling seeds on RedCaps-150k (110,405 hub nodes, 70,544 closed-triangle candidates — no escalation to 300k needed).

### Key results

| Statistic | Closed triangle | Genuinely unconnected | Contrast |
|---|---:|---:|---:|
| Pooled mean pull (± SEM) | **+3.1772 ± 0.0034** | **+2.4248 ± 0.0096** | **+0.7524 ± 0.0063** |
| mean/SEM | +927.0 | +253.3 | **+119.7** |
| Relative ratio | | | **1.31×** |

A same-day whole-branch review caught that the first pass's "open" control group (defined only as "not an `img_only` edge") was **51.7% contaminated** by pairs connected via some other edge type (mostly `txt_only`, since both endpoints already share a text-only hub); correcting to genuinely-unconnected **widened** the gap from ~1.2× to ~1.31×, strengthening rather than weakening the "discriminates" verdict.

The 2026-09-01 addendum broke C–D's relationship into all 5 edge types and found a clean monotone dose-response ordering: `both` (+3.26, 1.35×) > `img_only` (+3.17, 1.32×) > `txt_only` (+2.73, 1.13×, the modal 49.5%-of-population outcome and the source of the original contamination) > unconnected (+2.41, baseline). `img_only` pulls ~16% harder than `txt_only` despite both being "one edge" — a modality asymmetry echoing the img/txt theme from Sections 4–5, not mechanistically explained here.

### Verdict

**The buddy-init embedding does discriminate a genuine edge from a false-transitivity artifact, by a real and seed-stable ~31%.** This qualifies rather than reverses Experiment 12's caution — a genuinely-unconnected artifact pair still pulls together at roughly three-quarters the strength of a real edge — but it discriminates more clearly than the first, contaminated pass suggested. The control-group contamination itself is a methodological lesson: defining "open" as "not the specific edge type under test" rather than "no edge at all" is an easy mistake with multiple edge types in play, and it survived five individual task reviews before the final whole-branch review caught it.

## 7. Experiment 15 — attributing and refining buddy supervision (15.1–15.4)

### What was tested

Motivated by Experiment 14's discovery that even fully-unconnected hub pairs get pulled together, a joint brainstorm traced a plausible mechanism: the union graph `E` feeds training through three channels (buddy-init, the Laplacian smoothness term "Family #1," and the multi-positive contrastive term "Family #2"), all sampling uniformly and positive-only from an untyped edge list, with a self-refreshing variant ("Family #3") that can re-discover training-induced proximity as a spurious "new" edge. Experiment 15 staged four items to persist edge-type provenance, attribute the erosion to a specific stage, and refine or replace supervision accordingly.

### Key results

**15.1 (provenance + bug fix, `2684c50`):** edge-type provenance (`img_only`/`txt_only`/`both`/`repair`) is now persisted alongside every buddy-init template; `init_conditions.py`'s standalone runner now correctly forwards `distance_mode`; Family #1's CSR is now decoupled from Family #3's refreshable one, closing a real bug the code's own comment had flagged but not fixed.

**15.2 (stage attribution):** measuring in retrieval (`comb_all`) space rather than Experiment 14's raw z-table space, pooled across 11.1's 3 seeds/arm:

| Arm | Epoch | Closed / unconnected ratio |
|---|---:|---:|
| Trained or Frozen | 0000 (init) | 2.4793 ± 0.0031 |
| Trained | 0099 (final) | 2.2629 ± 0.0017 |
| Frozen | 0099 (final) | 2.2682 ± 0.0032 |

Both arms lose a nearly identical amount of discrimination (ratio 2.48 → ~2.26) **despite neither having any buddy-specific loss active** — ordinary retrieval training already erodes buddy-structure discrimination on its own, and the trained/frozen difference at epoch 99 (~0.005) is negligible next to that ~0.21 shared drop. Testing the three families directly (`lambda_buddy=0.1`, `lambda_buddy_con=0.3`, and the latter plus refresh) against this same ratio:

| Arm | Closed/unconnected ratio | Δ vs. baseline | mean/SEM |
|---|---:|---:|---:|
| baseline (buddy off) | 2.2629 ± 0.0017 | — | — |
| (d) Family #1 only | 2.2605 ± 0.0009 | −0.0024 | ≈1.3 (n.s.) |
| (e) Family #2 only | **2.2760 ± 0.0008** | **+0.0131** | **≈7.0** |
| (f) Family #2 + refresh | 2.2628 ± 0.0009 | +0.0001 vs. baseline | n.s. — cancels (e) |

**15.3 (typed Family #2, `b943a33`/`4ee9dfe`):** type-weighting and repair-exclusion added on top of 15.2(e)'s base value (`lambda_buddy_con=0.3`). A weighted-sampling CUDA out-of-bounds bug was caught by GPU smoke-testing (not just config parsing) and fixed with an explicit per-row clamp before the full run:

| Arm | Closed/unconnected ratio | Δ vs. buddy-off | mean/SEM |
|---|---:|---:|---:|
| 15.2(e) untyped | 2.2760 ± 0.0008 | +0.0131 | ≈7.0 |
| 15.3 typed + repair-excluded | 2.2795 ± 0.0014 | +0.0166 | ≈7.5 |

The typed refinement adds a further +0.0035 (mean/SEM≈2.2) over untyped Family #2 — real at this project's bar, but modest next to the gain from having *any* static, non-refreshed buddy supervision at all.

**15.4** (hub-mediated repulsion, an explicit new loss term pushing apart genuinely-unconnected hub pairs) is approved, per the plan's 2026-09-01 scope revision, but **not yet scoped or run** — its design is gated on 15.2/15.3's attribution, which is now in hand.

### Verdict

The false-transitivity erosion is mostly a byproduct of ordinary retrieval training, not something today's buddy-specific losses cause — of the three existing families, only the static (non-refreshed) contrastive term helps at all, and self-refresh actively cancels that gain, a measured regression consistent with the feedback-loop risk the original brainstorm flagged. Type-weighting that surviving term helps a little further. This reprioritizes 15.4: a targeted repulsion mechanism is motivated by a real, measured gap that reweighting alone only partially closes, not a theoretical concern.

## 8. Experiment 16 — mutual-KNN K ablation and a K(N) scaling law (16.1 + 16.2)

### What was tested

Every prior RedCaps finding (C5–C15) fixes K=30, inherited from a single early sweep on a different dataset that never checked K against real training. **Stage A (16.1)** was a training-free graph diagnostic: strict buddy (`A_img ∩ A_txt`) and union buddy (`E`) statistics across `K ∈ {10,20,30,50,75,100} × N ∈ {150k, 300k, 500k}` (18 cells), asking whether K should scale with N. **Stage B (16.2)**, gated on Stage A's shortlists, trained the predicted K values at 300k/500k against the K=30 anchor (trained arm only — the freeze-vs-trainable crossing was dropped by explicit decision).

### Key results — Stage A

Strict buddy is sparse everywhere: the zero-strict-degree fraction never drops below **60.9%** (150k, K=100, the densest cell tested) and reaches **94.8%** at 500k/K=10. Strict-buddy average degree rises with K but falls with N at fixed K (K=30: 0.371 → 0.305 → 0.266 across 150k/300k/500k). Holding the 150k/K=30 value as an invariant, inversion predicts **K≈35 at 300k** and **K≈39 at 500k** — sublinear scaling (an 8.7% increase in K for a 2× increase in N, not proportional). Subreddit-lift quality stayed flat (22.7–23.0×) across all 18 cells: the K(N) effect is pure structural sparsity, not a signal-quality collapse.

### Key results — Stage B

| N | K | metric | Δ vs. K=30 | mean/SEM |
|---|---:|---|---:|---:|
| 300k | 35 | t2i R1 | +0.20 | +2.0 |
| 300k | 50 | t2i R1 | +0.33 | +2.8 |
| 500k | 39 | i2t R1 | **+0.97** | **+14.5** |
| 500k | 39 | t2i R1 | +0.10 | +0.9 (n.s.) |

**500k/K=39 — exactly the K(N)-invariant-matched prediction — is the clearest, best-supported result in the whole sweep**: +0.97 i2t R1 over the project-wide K=30 default, clearing both the significance bar and the noise floor by a wide margin. But the effect is direction- and scale-specific, not a uniform "bigger K helps": 500k's win is i2t-only, 300k's (smaller, borderline) win is t2i-only, and neither scale shows both directions moving. 500k/K=50 has only 1 seed (sweep stopped early once the K=39 signal was clear) and is a directional data point only, not a paired claim.

### Verdict

K=30, used everywhere in this project's RedCaps work, is measurably suboptimal at 500k specifically for i2t retrieval, and the fix the structural K(N) rule predicted is the one that actually works — a rare case here of a training-free diagnostic landing exactly on the retrieval-optimal value. The per-(N, metric) heterogeneity means no blanket "increase K" claim is supported; any paper statement needs to be scoped per cell.

## 9. Combiner architecture ablation — a separate exploratory track

### What was tested

Independent of the publication-plan's numbered experiments, a literature-grounded search asked whether the production fusion module (`Combiner_new`: two MLP towers → concat → MLP decode → learned scalar gate) — never itself chosen by ablation — is actually a good choice for fusing a tiny buddy vector into a frozen CLIP embedding. A Codex-dispatched brainstorm (grounded in the actual repo code, not a generic prompt) produced a ranked shortlist from the composed-image-retrieval, FiLM, LoRA, and Highway/ResNet literature; three candidates were implemented behind a new `combiner_type` switch (default unchanged) and smoke-tested at `redcaps_150k`, 30 epochs, 3 seeds.

### Key results

| combiner_type | oracle i2t R1 Δ | pre_diff i2t R1 Δ (deployment) |
|---|---:|---:|
| residual_control (full-width, zero-init) | +0.63 (n.s.) | **−23.87 (mean/SEM −84)** |
| **lowrank (rank-16 residual)** | **+3.10 (mean/SEM +15)** | −3.63 (mean/SEM −11) |
| film (zero-init, bounded γ) | +2.97 (mean/SEM +20) | **−24.63 (mean/SEM −78)** |

Sweeping buddy dimension `{8,16,32}` for the `lowrank` winner moved nothing — every metric landed within noise of the existing `embedding_dim=16` default. Separately, a real bug was found: the production `Combiner_new` is not actually identity-init at t=0 (its gate zero-inits, but the two MLP towers feeding it are randomly initialized), contradicting the codebase's own identity-init convention and confounding prior reads of "does conditioning help" with "does a random perturbation at init help."

### Verdict

**Fusion family is the axis that matters, and it isn't close; buddy dimension is a non-factor at this scale.** `residual_control` and `film` are neutral-to-good on oracle retrieval but collapse once the fusion has to work from the *predicted* (not table) buddy vector — a reproducible fragility, not noise. `lowrank`'s contained rank-16 residual is the only family with both a real oracle gain and a comparatively small (1/5th–1/7th) deployment-tier cost, a clean empirical confirmation that restricting fusion to a small subspace, not letting it touch the full frozen embedding, is what makes conditioning deployment-safe. This is a smoke-test result (`redcaps_150k`, 30 epochs) meant to narrow the search space — `lowrank` should be validated at the project's real 300k/500k, 100-epoch scale before being adopted as the default combiner for the publication plan's own experiments; no config default or spec claim was changed this week.

## 10. Synthesis — what this week established

Two long-standing puzzles were substantially resolved, in the same direction. The i2t-vs-t2i freeze-ablation asymmetry from Experiment 11.1 — this project's headline post-init-training finding — is now understood as **substantially an architectural artifact of the one-sided combiner**, not an intrinsic property of buddy-graph training: it shrinks ~11× under `combine_side="txt"` and collapses to a noise-floor null under Experiment 13's symmetric conditioning. Separately, the "false transitivity" concern from Experiment 12 is now understood with a genuine positive control (Experiment 14: the embedding discriminates real edges from non-edges by a real ~31%) and a causal attribution (Experiment 15: ordinary retrieval training erodes that discrimination on its own, and only the static contrastive term recovers part of it). Both threads moved from "here is a concerning pattern" to "here is what causes it and how much it matters" — the kind of progress that strengthens rather than complicates the publication narrative, since neither resolution requires abandoning buddy-graph structure as the core initializer claim.

Two new tunable axes were also discovered this week, both with the same shape: a training-free structural diagnostic made a specific, falsifiable numeric prediction, and a follow-up training run confirmed it. Experiment 16's K(N) invariant predicted K≈39 at 500k and that value delivered the week's single cleanest retrieval win (+0.97 i2t R1, mean/SEM +14.5). The combiner-architecture search found an analogous asymmetry at the mechanism level: restricting fusion capacity (low-rank) rather than expanding it (full-width residual/FiLM) is what keeps a real oracle gain from collapsing under deployment-tier conditioning. Neither result yet changes a production default — both are flagged for validation at full scale before adoption.

## 11. Next steps

1. Scope and run Experiment 15.4 (hub-mediated repulsion), now that 15.2/15.3 have attributed the erosion and shown reweighting alone only partially closes the gap; use an evidence-conditioned margin (only push apart pairs also far in frozen CLIP space) per the brainstorm's own caution against a universal non-edge-implies-negative rule.
2. Complete 500k/K=50's remaining 2 seeds if the paper's final K(N) story needs a 3-seed cell there; otherwise treat 16.2 as closed pending a structural cross-reference explaining why the win is i2t-specific at 500k but t2i-specific at 300k.
3. Validate `lowrank` combiner at the project's real 300k/500k, 100-epoch operating point before considering it as a replacement default; add the residual-norm/cosine-shift/gate-value logging the smoke pass skipped.
4. Investigate the mechanistic *why* behind Experiment 13's result (gradient asymmetry into the shared table, or `other_proj`'s never-adapting identity pathway) — Experiment 13 shows removing the asymmetry removes the effect, not the causal path by which the asymmetry produced it.
5. Consider the mirror closed-triangle configuration (img-only hub, txt_only closure) that Experiment 14 flagged as untested, given C10's headline was precisely that its subgroup effect flips modality under `combine_side="txt"`.

## Appendix — source artifacts

- Prior weekly report: [2026-08-26 weekly report](2026-08-26_conditional_buddies.md)
- Experiment 12 (+12.2, 12.3): [cross-modal bridge/false-transitivity diagnostic](../auto/buddy/2026-08-26_polysemy_bridge_diagnostic.md)
- `combine_side="txt"` replication: [does it flip the asymmetry?](../auto/buddy/2026-08-27_combine_side_txt_replication.md)
- Experiment 13: [symmetric conditioning](../auto/buddy/2026-08-27_symmetric_conditioning_exp13.md)
- Experiment 14: [closed-triangle bridge diagnostic + addendum](../auto/buddy/2026-08-31_closed_triangle_bridge_diagnostic.md)
- Experiment 16.1: [buddy K scaling, Stage A](../auto/buddy/2026-09-01_buddy_k_scaling_stage_a.md)
- Experiment 16.2: [buddy K ablation, Stage B](../auto/buddy/2026-09-02_buddy_k_ablation_stage_b.md)
- Combiner architecture brainstorm: [literature search + candidate shortlist](../auto/buddy/2026-09-01_combiner_architecture_brainstorm.md)
- Combiner architecture ablation: [family vs. dimension results](../auto/buddy/2026-09-02_combiner_architecture_ablation.md)
- Master status and gates: [publication-plan design](../../superpowers/specs/2026-08-04-buddy-publication-plan-design.md)
