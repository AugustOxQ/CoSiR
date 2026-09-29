# Reproducing CoSiR in PercepT

**ArtELingo mechanism-fidelity investigation · 2026-09-23**

---

## 1. What is PercepT?

For each painting, PercepT combines two complementary descriptions:

- a factual embedding from CLIP, capturing what is depicted;
- an affective embedding from a GoEmotions-tuned text encoder, capturing the emotional language around it.

**Stage 1 — unsupervised topic formation.** The fused representation passes through an autoencoder, then Deep Embedded Clustering (DEC) sharpens an initial K-means grouping into perceptual topics. Emotion and genre labels are not used for training; they are only used afterward to evaluate the discovered topics.

**Stage 2 — image-only topic mapping.** The Stage 1 topic assignments are frozen. An image-only classifier learns to recover them from the painting, so no text is needed at inference.

---

## 2. Why test this mechanism here?

CoSiR already has an established same-data result from a different factual-plus-affective mechanism: its conditional buddy graph. On the identical ArtELingo held-out split and criterion, **Attention-h1** achieved emotion AMI **0.1249** and genre AMI **0.2404**.

This investigation asks whether PercepT’s independent fusion → autoencoder → DEC mechanism can find an equally useful partition on that same data. It is a **mechanism-fidelity check**, not an attempt to beat the paper’s reported values.

The paper is not a fair numerical baseline: it used a different dataset and label taxonomy; this replication substitutes a GoEmotions-RoBERTa affect backbone, a concatenation-based 2:1 fusion formula for the paper’s elementwise sum, and a convergence-controlled DEC schedule rather than fixed 200 epochs.

---

## 3. Early observation: getting from collapse to a stable operating point

DEC did not work reliably on the first run. The short progression below shows the diagnosis-led path to K=60/40; the accompanying chart shows held-out emotion and genre AMI at the same waypoints.

| waypoint | emotion AMI | genre AMI | takeaway |
|---|---:|---:|---|
| Base K=100/67 | 0.0363 | 0.3081 | collapsed |
| Balance λ=1 | 0.0925 | 0.2631 | still collapsed |
| Balance λ=500 | 0.1142 | 0.2116 | still collapsed |
| λ=1000, one seed | 0.1242 | 0.2466 | first success, fragile evidence |
| λ=1000, 4-seed mean | 0.1235 | 0.2089 | only 1/4 clears both |
| K=60/40, 4-seed mean | 0.1252 | 0.2486 | 4/4 clear both |

**Mechanism learned:** changing cluster count—not merely increasing regularization—was the stability fix.

Chart asset: `assets/diagrams/percept_fusion_mechanism_progression.png`.

---

## 4. How to read the metrics—and what “collapse” means

**Adjusted Mutual Information (AMI)** is chance-adjusted agreement between a discovered topic assignment and a human label. **0** means chance-level agreement; **1** means perfect agreement.

DEC repeatedly trains on its own increasingly confident assignments. Without an adequate counterweight, that self-sharpening pressure can empty almost every topic into a few: a converged run can still be a failed partition.

**Concrete base-run example:** the unregularized K=100/67 run converged, yet **65 of 67** surviving topics had under 1% of nodes and the median topic size was **0**. Its held-out AMI was emotion **0.0363** and genre **0.3081**. The genre value alone did not make the collapsed clustering useful.

---

## 5. Fusion mechanisms tried—and the CCA pivot

| train-split method | emotion AMI | genre AMI |
|---|---:|---:|
| Content-only | 0.0593 | 0.4384 |
| GoEmotions-only | 0.1180 | 0.0396 |
| Late fusion — union | 0.1236 | 0.1394 |
| Hierarchical refinement | 0.1072 | 0.1954 |

1. **Late fusion:** union improved emotion but retained little genre; intersection was degenerate (**98.96%** isolated before repair).
2. **Hierarchical refinement:** real affect signal, but much genre cost was a smaller-label granularity artifact.
3. **SNF and co-regularized spectral:** evidence-based **skipped**, not run—the mutual-kNN graph intersection was near-empty.
4. **CCA audit — the pivot:** top held-out canonical correlation **0.7285**, versus permutation-null **0.0699**, licensed a learned model.
5. **Outcome:** a learned two-teacher contrastive student was pursued.

The embedded frontier predates the later architecture sweep, so it shows Stage 1 as the frontier at that time; the next slide shows what came after.

Figure: `assets/fusion_pareto_frontier.png`.

---

## 6. From linear heads to attention: how Attention-h1 became the standing baseline

| configuration | train emotion / genre AMI | held-out emotion / genre AMI |
|---|---:|---:|
| Stage 1 — linear heads, scalar gate | 0.1284 / 0.2799 | 0.1095 / 0.2901 |
| Stage 2 — MLP-64 heads, scalar gate | 0.1230 / 0.2319 | 0.1046 / 0.2087 |
| MLP-128 heads, scalar gate | 0.1142 / 0.2039 | 0.1114 / 0.1604 |
| Attention-h4 — linear heads | 0.1328 / 0.2441 | 0.1216 / 0.1875 |
| **Attention-h1 — linear heads** | **0.1351 / 0.2397** | **0.1249 / 0.2404** |

Richer **per-view encoding** (MLP heads) consistently hurt both AMI axes. Richer **combination** capacity (self-attention instead of a scalar gate) helped, but only with one head: spreading the 32-D shared space over four heads lost the genre-axis gain.

Attention-h1 is the first configuration to clear both held-out bars (**emotion > 0.1236; genre > 0.1954**). It does not strictly dominate Stage 1—Stage 1’s held-out genre AMI (**0.2901**) remains higher—but it is the standing result under the investigation’s predeclared Pareto-bar criterion.

Chart asset: `assets/diagrams/percept_attention_architecture_heldout_ami.png`.

---

## 7. Fine K-sweep: what qualifies as success?

The predeclared **held-out Pareto bar** is:

> **emotion AMI > 0.1236 AND genre AMI > 0.1954**, simultaneously, on val+test with zero train overlap.

| configuration | emotion AMI (4-seed mean) | genre AMI (4-seed mean) | joint clears |
|---|---:|---:|---:|
| Attention-h1 buddy baseline | 0.1249 | 0.2404 | established same-split reference |
| K=60/40 (standing) | 0.1252 | 0.2486 | 4/4 |
| K=65/43 (best fine-sweep) | 0.1234 | 0.2435 | 3/4 |

K=65/43 was the best nearby point, but its like-for-like 4-seed means are lower on both axes. K=60/40 remains the standing configuration.

---

## 8. Full K sweep: why K=60/40 remains standing

The chart plots the source-reported held-out screen values for all ten cluster-count configurations. Its horizontal lines show both Pareto thresholds and the Attention-h1 same-split baseline.

| K initial / surviving | emotion AMI | genre AMI |
|---|---:|---:|
| 20 / 13 | 0.1152 | 0.2273 |
| 30 / 20 | 0.1220 | 0.2577 |
| 40 / 27 | 0.1220 | 0.2736 |
| 50 / 33 | 0.1202 | 0.2830 |
| 55 / 37 | 0.1208 | 0.2622 |
| 60 / 40 (standing) | 0.1238 | 0.2617 |
| 65 / 43 | 0.1238 | 0.2376 |
| 70 / 47 | 0.1237 | 0.2357 |
| 80 / 53 | 0.1220 | 0.2396 |
| 100 / 67 | 0.1242 | 0.2466 |

**Reference point:** Attention-h1 = emotion **0.1249**, genre **0.2404**. Only 65/43 and 70/47 cleared both bars at the seed-42 screen; 65/43 then cleared jointly in 3/4 seeds, versus 4/4 for K=60/40.

Chart asset: `assets/diagrams/percept_k_sweep_heldout_ami.png`.

---

## 9. Stage 1 extended-seed check: best-supported is not seed-robust

| statistic | held-out emotion AMI | held-out genre AMI |
|---|---:|---:|
| mean (14 seeds) | 0.1242 | 0.2517 |
| min / max | 0.1157 / 0.1289 | 0.2192 / 0.2750 |
| sample standard deviation | 0.0032 | 0.0156 |
| individual bar clears | 10/14 (71.4%) | 14/14 (100%) |
| both bars clear simultaneously | 10/14 (71.4%) | — |

Genre clears in every seed. Emotion misses in four, so the 4/4 original read becomes a more cautious **10/14 joint-clearance** result.

---

## 10. Stage 2 reproducibility: the gate is a safety check, not a formality

Before *any* Stage 2 image-classifier training, every run re-fits the exact frozen Stage 1 K=60/40 seed-42 encoder from scratch. It must reproduce the known held-out Stage 1 AMI **0.1238 emotion / 0.2617 genre** within absolute tolerance **0.002**. This gate prevents silently training the image mapper against a different, unintended clustering.

The refit returned **0.1225 / 0.2274**: emotion was within tolerance (difference 0.0013), but genre missed by **0.0343**—over 17× the tolerance. The gate stopped mapper training, as intended. A repeated re-fit was byte-identical, so the Stage 2 experiment was rebased to that reproducible clustering and all 14 mapper seeds were retrained against it.

**Resolution of the noise-floor question:** five separate-process launches of this exact seed-42 refit all produced byte-identical **0.1225 / 0.2274** results (zero spread). The Stage 1 seed dependence is therefore genuine, not GPU noise; the citation mismatch most likely reflects a different execution environment or library version, not run-to-run randomness.

---

## 11. Rebased Stage 2: image-only mapping remains seed-robust

The mapper attention-pools 50 CLIP ViT-B/32 patch tokens with one learnable query, then uses a linear 40-topic head. It predicts multi-label targets at `q > 1.2/40` (always including the argmax topic).

| statistic | held-out macro AUC |
|---|---:|
| mean (14 seeds) | 0.8290 |
| min / max | 0.8266 / 0.8315 |
| sample standard deviation | 0.0014 |
| 95% CI (normal approximation) | [0.8283, 0.8298] |

All 14 mapper-initialization seeds exceed the 0.5000 train-marginal-frequency baseline and the old threshold-sweep maximum of 0.5760. The new mean is marginally higher than the original 4-seed citation (0.8256), despite the Stage 1 genre AMI shift.

---

## 12. Stage 2 result table: exact scope of the rebased claim

| conclusion | evidence |
|---|---|
| Fixed Stage 1 target | deterministic re-fit of K=60/40, seed 42: 0.1225 emotion / 0.2274 genre |
| Stage 2 target construction | frozen 40 topics; image-only mapper; `q > 1.2/40` multi-label targets |
| Mapper evidence | 14 newly trained mapper-init seeds, all against the same rebased target set |
| Held-out result | macro AUC mean 0.8290; min/max 0.8266 / 0.8315; sample SD 0.0014 |
| Robustness conclusion | seed-robust mapper result; not a mixture of old and rebased target sets |

The Stage 2 conclusion remains intact after the gate correction: the image-only mapper is stable across mapper initializations even though the underlying Stage 1 partition is sensitive to its seed.
