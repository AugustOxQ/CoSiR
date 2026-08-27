# Cross-modal neighborhood-disagreement bridges: does the buddy graph already reflect them, and is any B-C pull real signal or a "false transitivity" artifact?

**Date:** 2026-08-26 · **Dataset:** RedCaps, 150,000 rows (matches C5/Exp 9-11's scale) · **Branch:** `experiment/condition_drift_retrieval_correlation`
**Code:** `scripts/analyze_polysemy_bridges.py`, `src/conditional_buddy/buddy_graph.py` (`classify_edges`/`bridge_node_stats`)
**Spec:** `docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md` §4 Experiment 12
**Precursor:** `src/test/20260824_buddy_graph_disagreement/diagnose_disagreement.py` (found ~80% of nodes are "bridge" nodes; this experiment measures what that structurally implies for the resulting embedding)

---

## TL;DR

**Cross-modal neighborhood-disagreement bridge structure is real and pervasive (80.2% of nodes), the buddy-init embedding does pull a bridge node's image-only and text-only neighbors (B, C) together — strongly and reliably (mean pull = +1.98 embedding-distance units, mean/SEM = +102.1, 91.4% of 5,000 sampled pairs pulled closer than baseline) — but that pull is only very weakly explained by real shared-neighbor overlap (Spearman rho = +0.076 between shared-neighbor Jaccard and pull magnitude; statistically significant at this sample size, p = 8.6e-8, but rho² ≈ 0.006 means shared-neighbor structure accounts for well under 1% of the pull's variance).** This lands closer to the spec's **second branch — "real + ungraded" — confirming Experiment 10's flagged "false transitivity" risk as an actual, measurable distortion**, not the "real + graded" branch: most of the pull looks like a broad, largely content-independent effect of the spectral embedding's global smoothing (consistent with Experiment 10's own finding that this method is near-invariant to within-class affinity structure once a dominant edge class is present — see that report's Laplacian-rescale-invariance argument), rather than the embedding faithfully tracking how much B and C's neighborhoods actually overlap. Separately, the per-node bridge/disagreement label does **not** significantly predict per-sample retrieval-rank change in this run's population (`corr(is_bridge_or_one_sided, |delta_rank|)` rho = +0.016, p = 0.37) — a null on the retrieval/drift cross-reference.

## Method

Reused the already-completed buddy-init template (`res/CoSiR_condition_freeze_ablation/redcaps_150k/template_embeddings/`, the same one 11.1/11.2/11.3 trained against) and rebuilt `A_img`/`A_txt`/`E` from cached RedCaps-150k features (K=30, alpha=0.5) via `classify_edges` to identify cross-modal neighborhood-disagreement bridge nodes and label every node (`img_only_only` / `txt_only_only` / `bridge` / `neither`). An **image-only** edge joins two samples that are mutual nearest neighbors in image-feature space but not in text-feature space; a **text-only** edge is the converse. A **bridge node** A has at least one image-only neighbor B and at least one text-only neighbor C, creating an indirect B–A–C path even though B and C are not directly linked. This is a graph-topology/provenance label, not evidence that an image has multiple captions or that a caption has multiple meanings. A **bridge pair** is the resulting B–C pair around A. Its **pull** is `baseline_dist - bc_dist`, where `baseline_dist` is the embedding distance from B to a degree-matched non-neighbor C′ and `bc_dist` is the embedding distance from B to C. Positive pull means the bridge pair is closer in the embedding than the matched baseline; it describes resulting embedding geometry, not a literal training force. Sampled 5,000 bridge-node (A, B, C) triples and compared each B–C pair's embedded L2 distance against a degree-decile-matched, non-adjacent baseline pair. Checked whether that pull correlates with each pair's shared-neighbor Jaccard overlap in `E` (the LINE/GraRep-style second-order-proximity score). Cross-referenced the per-node bridge/disagreement label against a 150k-population, 3,000-query-sample per-sample retrieval-rank/condition-drift dump from one already-completed 11.1 `trained` run (seed 1, paired against its seed-1 `frozen` counterpart), produced by extending `scripts/analyze_condition_retrieval_correlation.py`'s `analyze_pair()` with a new opt-in `--dump-per-sample` flag (Task 2 of this experiment's implementation plan).

## Results

### Bridge structure

```
bridge nodes: 120,324 / 150,000 (80.2% of nodes)
label counts: {'neither': 1737, 'img_only_only': 21765, 'txt_only_only': 6174, 'bridge': 120324}
```

![Counts of each cross-modal neighborhood-disagreement node label](assets/polysemy_bridges/node_label_counts.png)

The log-scaled count view makes the dominance of bridge nodes (80.2% of the 150,000-node graph) immediately visible while retaining the smaller label groups.

The 80.2% bridge-node fraction matches Experiment 10's own diagnostic figure on this same dataset almost exactly — a strong cross-check that this experiment's rebuilt graph is consistent with the earlier one. Very few nodes (1,737, 1.2%) have no img-only or txt-only edges at all (`neither`) — the large majority of the graph's non-`both`/`repair` edges belong to bridge nodes, not to nodes with only one edge type.

### False-transitivity audit

```
sampled bridge pairs: 5,000
pull (baseline_dist - bc_dist): mean=+1.9786 (n=5000, frac_pulled_closer=0.914)  mean/SEM=+102.1 *
grading check: corr(shared_neighbor_jaccard, pull) rho=+0.076 p=8.619e-08
```

![Distribution of bridge-pair embedding pull against degree-matched baselines](assets/polysemy_bridges/pull_distribution.png)

The pull distribution is overwhelmingly positive, with the zero reference and mean pull showing how consistently bridge-derived pairs sit closer than their matched baselines.

![Shared-neighbor Jaccard versus bridge-pair embedding pull](assets/polysemy_bridges/jaccard_vs_pull.png)

The fitted trend is positive but shallow against the wide vertical spread, visualizing the statistically real yet practically weak shared-neighbor correlation.

The pull is large and essentially universal: 91.4% of sampled (B, C) pairs sit closer together in the buddy-init embedding than their degree-matched baseline, and the effect clears the project's mean/SEM ≥ 2 significance bar by two orders of magnitude (+102.1). But the grading check — whether that pull scales with how much B and C's neighborhoods actually overlap — comes back statistically significant only because of the large sample size (p = 8.6e-8), not because the relationship is practically strong: rho = +0.076 means shared-neighbor Jaccard explains roughly 0.6% (rho²) of the variance in pull magnitude. The overwhelming majority of the pull is not accounted for by real shared-neighbor structure.

This does not mean the B–C pull is a random event unrelated to the construction. The buddy graph deliberately combines image- and text-derived neighbor edges, and the spectral embedding deliberately smooths graph connectivity, so a degree of closeness along the indirect B–A–C path is an expected consequence. The concern is narrower: the construction did not explicitly establish that B and C are content-sensitively related, and the audit finds that the induced pull is mostly not graded by evidence from their wider shared neighborhoods. Thus “false transitivity” names a risk of over-interpreting embedding closeness as semantic relatedness, rather than a demonstrated error in the graph construction itself.

### Retrieval/drift cross-reference

```
retrieval cross-reference (n_joined=3000):
  neither:        n=38   median|delta_rank|=10.0  median_drift=0.0830
  img_only_only:  n=422  median|delta_rank|=39.0  median_drift=0.0850
  txt_only_only:  n=139  median|delta_rank|=14.0  median_drift=0.0689
  bridge:         n=2401 median|delta_rank|=13.0  median_drift=0.0780
  corr(is_bridge_or_one_sided, |delta_rank|): rho=+0.016 p=3.680e-01
```

![Median absolute retrieval-rank change by bridge/disagreement label](assets/polysemy_bridges/retrieval_by_label.png)

The label-wise medians make the descriptively higher `img_only_only` value visible alongside subgroup sizes and the overall null correlation.

The formal test — is a sample a bridge or one-sided disagreement node *at all* (any of the three non-`neither` labels) correlated with how much its retrieval rank moved between the frozen and trained arms — is a clean null (rho = +0.016, p = 0.37, far from significant). `|delta_rank|` is the magnitude of rank change, not a quality score: a larger value means retrieval position changed more, without saying whether that change was an improvement or deterioration. `condition_drift` likewise records representation change between conditions. This cross-reference is an association check on one paired run, not evidence that the graph label causes retrieval or drift changes. Descriptively, the small `img_only_only` subgroup (n=422) shows a notably higher median `|delta_rank|` (39) than the other three groups (10-14), but this is a single subgroup observation, not the designed statistical test, and should not be read as a confirmed effect without a dedicated follow-up.

## Interpretation

This experiment answers the three questions from the brainstorming session concretely:

1. **Is cross-modal neighborhood disagreement reflected in the current graph structure and initialization?** Yes, structurally — `classify_edges` already labels the img-only/txt-only edge pattern (A-B, A-C in the running example) explicitly, and 80.2% of nodes exhibit it. But the *embedding* reflects this topology in a way that is largely non-specific: B and C get pulled together reliably, but mostly independent of whether their neighborhoods actually overlap in content-relevant ways. This should not be interpreted as evidence of semantic polysemy (for example, that one image has multiple distinct captions).
2. **Is the edge provenance modality-aware — do we know whether an edge is image-only or text-only?** Yes, trivially — `classify_edges`'s per-edge typing already carries this (confirmed again here), and the per-node label (`img_only_only`/`txt_only_only`/`bridge`) makes it queryable per sample. What this experiment adds is that knowing the direction doesn't yet buy anything downstream: the retrieval/drift cross-reference found no significant behavioral signature tied to bridge/disagreement type.
3. **What about B and C's own relation?** This is the experiment's central, previously-unmeasured finding: B and C — never directly connected in either modality's mutual-kNN graph — do end up measurably closer together than chance in the resulting spectral embedding, and that pull is strong and robust, but it is **not well explained by real second-order/shared-neighbor structure**. Per the spec's own framing, this is the **"false transitivity" branch, not the "real + graded" branch**: the pull looks like a broad byproduct of the spectral method's global smoothing (consistent with Experiment 10's finding that this pipeline's spectral step is near-invariant to fine-grained affinity reweighting once a dominant, structurally homogeneous edge class exists) rather than the embedding faithfully encoding *how much* B and C are actually related through A.

**Practical takeaway for the paper:** this reinforces rather than undercuts the "buddy-init geometry alone is fine" throughline from Experiments 11.1-11.3 — the bridge-pair pull is a measurable but behaviorally inert property of the current construction (no significant retrieval/drift consequence found), not a mechanism actively producing bad training signal. It is nonetheless a genuine, previously-unquantified methodological caveat worth stating plainly in the paper's limitations: the buddy graph's spectral embedding does not cleanly separate "genuinely related via shared context" from "coincidentally connected via one bridging sample," and a reader should not over-interpret embedded closeness between two samples as evidence of deep cross-modal relatedness without checking whether a bridge node is responsible. Per the spec's own discipline, this does **not** trigger a committed follow-up mechanism (e.g. a modality-aware dual-embedding representation) — that would need to be separately scoped and approved, and the retrieval-null result here means there is not yet evidence such a mechanism would move any measured outcome.

## Caveats

- This is a single-graph diagnostic (one buddy-init template, not repeated across training seeds) — statistical significance here comes from the number of sampled bridge-pairs (5,000), not from seed replication. This matches the precedent of Experiment 9's and Experiment 10's own diagnostic scripts, which are likewise single-graph analyses.
- The retrieval/drift cross-reference uses one specific 11.1 trained run's per-sample dump (seed 1, `20260825_161846_CoSiR_Experiment` vs. frozen `20260825_170212_CoSiR_Experiment`); it is in-sample and own-condition, the same scoping caveat 11.2's own report already states for that population. A second or third seed's dump was not checked here.
- The `img_only_only` subgroup's descriptively higher median `|delta_rank|` (39 vs. 10-14 elsewhere) is based on only 422 samples and was not the experiment's designed hypothesis test — treat as a lead for a possible future targeted check, not a finding.

## Reproduction

```bash
python scripts/analyze_condition_retrieval_correlation.py \
  --pair res/CoSiR_condition_freeze_ablation/redcaps_150k/20260825_170212_CoSiR_Experiment \
         res/CoSiR_condition_freeze_ablation/redcaps_150k/20260825_161846_CoSiR_Experiment \
  --dump-per-sample
python scripts/analyze_polysemy_bridges.py --n-bridge-sample 5000 --device cuda \
  --per-sample-npz res/CoSiR_condition_freeze_ablation/redcaps_150k/20260825_161846_CoSiR_Experiment/condition_geometry/per_sample_retrieval_correlation.npz
```

## Experiment 12.2 — Training-trajectory audit: does the i2t deficit hold from the start?

**Motivation:** Experiments 11.1–11.3 (`docs/reports/2026-08-25_condition_freeze_ablation.md`) established that the default post-init-`trained` checkpoint loses to a `frozen`-at-init baseline on i2t retrieval by a wide, seed-replicated margin, with no attempted fix (11.3's `pred_coupled`) recovering any of it. This experiment's own retrieval/drift cross-reference (above) used the final-epoch (99) `trained` checkpoint as "the outcome of post-init training." Before extending that cross-reference further (Experiment 12.3, next section), this audit checks a prior, cheaper question: is epoch 99 even a representative checkpoint to call "trained," or is the i2t deficit still actively growing at that point (making epoch 99 a worst-case, non-representative snapshot of an ongoing collapse)?

### TL;DR

**The i2t deficit is not present at initialization, but it is not still growing at epoch 99 either — it emerges over roughly the first half of training and then plateaus.** Both `trained` and `pred_coupled`'s i2t gap vs. `frozen` starts at ~0 at epoch 0 (shared init, as expected), declines steadily and almost identically across all 6 already-completed runs (3 `trained` seeds, 3 `pred_coupled` seeds) through epoch ~40–50 (reaching roughly −4.2 to −5.3 R1), then flattens out with no further systematic trend through epoch 99 (fluctuating between about −4.1 and −5.3, indistinguishable from noise around a stable level). t2i's gap stays within roughly ±0.8 R1 throughout, with no trend — consistent with 11.1's own t2i noise-floor null. **This means epoch 99 is a fair representative of the settled, post-acquisition-phase effect of continued training, not an outlier mid-collapse** — the cross-reference this report and Experiment 12.3 build on is measuring a plateaued end-state, not a transient still in motion.

### Method

New script `scripts/analyze_training_trajectory.py` pulls per-epoch `test_oracle/{t2i,i2t}_R1` history (not just the final-epoch summary 11.1/11.3's own scripts read) via the wandb API for all 9 already-completed runs (11.1's 3 `trained` + 3 `frozen`, 11.3's 3 `pred_coupled`; same `condition-freeze-ablation-redcaps_150k` / `pred-stopgrad-ablation-redcaps_150k` tags). Rows are joined on the `epoch` key logged in the *same* `wandb.log()` call as these metrics — **not** wandb's own `_step` counter, which is a global logging-call index that does not align across runs (different arms log a different number of things per epoch, e.g. `pred_coupled`'s extra `loss_pred` term, shifting `_step` at the "same" training epoch). A first implementation used `_step` and silently produced empty/misaligned gaps for every run; this was caught before any real numbers were reported (see Caveats) and fixed to use `epoch`, confirmed to share the identical `_step` value as `test_oracle/i2t_R1` within each logged row. Zero new training, zero new eval runs — reuses only already-logged wandb history.

### Results

```
i2t gap vs frozen (treatment - frozen), by (arm, seed, epoch):
  trained seed 1: e0=+0.00, e10=-0.80, e20=-2.50, e30=-3.70, e40=-4.20, e50=-4.50, e60=-5.00, e70=-4.90, e80=-4.80, e90=-5.10, e99=-4.90
  trained seed 2: e0=+0.00, e10=-0.30, e20=-3.00, e30=-4.00, e40=-4.30, e50=-4.80, e60=-4.90, e70=-4.70, e80=-4.50, e90=-4.30, e99=-4.40
  trained seed 3: e0=+0.00, e10=-0.50, e20=-3.00, e30=-4.10, e40=-4.70, e50=-5.30, e60=-4.60, e70=-4.80, e80=-4.10, e90=-4.70, e99=-4.70
  pred_coupled seed 1: e0=+0.00, e10=-0.90, e20=-2.50, e30=-3.70, e40=-4.40, e50=-4.30, e60=-4.40, e70=-4.80, e80=-4.60, e90=-4.70, e99=-4.70
  pred_coupled seed 2: e0=+0.00, e10=-0.40, e20=-3.00, e30=-3.70, e40=-4.90, e50=-4.90, e60=-5.00, e70=-4.70, e80=-4.70, e90=-4.40, e99=-4.50
  pred_coupled seed 3: e0=+0.10, e10=-0.50, e20=-2.80, e30=-4.20, e40=-4.90, e50=-5.20, e60=-4.70, e70=-4.90, e80=-4.20, e90=-4.70, e99=-5.10
```

![Per-epoch i2t gap vs. the frozen baseline, all 6 already-completed trained/pred_coupled seeds](assets/training_trajectory/i2t_gap_trajectory.png)

All 6 lines are nearly indistinguishable in shape: a sharp, monotonic decline from epoch 0 to roughly epoch 40–50, then a flat, noisy plateau through epoch 99 with no further systematic drift in either direction. t2i's gap (not plotted; see Reproduction to regenerate) stays within roughly ±0.8 R1 across every epoch and run, with no trend — a stable null, matching 11.1's own t2i noise-floor finding.

### Interpretation

The i2t deficit is **late-onset but saturating, not an ongoing collapse**: it is essentially zero at epoch 0/10 (shared buddy-init geometry, as expected — training hasn't had time to diverge yet), grows steadily through the first ~40–50% of the run, and then holds at a stable level through the final epoch. Because the trajectory has already flattened well before epoch 99, **the final checkpoint used throughout 11.1–11.3 and this report's own retrieval/drift cross-reference is a fair representative of the settled post-init-training effect**, not a worst-case sample of a still-accelerating failure. This narrows what any `trained`-vs-`frozen` per-sample diagnostic (this report's cross-reference, and Experiment 12.3 next) can claim: it characterizes the *plateaued* effect of continued training, not the *transient acquisition* dynamics of epochs 0–50, which would need the intermediate checkpoints Experiment 12.4 tentatively scopes. The near-identical shape between `trained` and `pred_coupled` — two different loss-stack configurations — also reinforces 11.3's own finding that whatever drives this decay is common to both recipes, not something the stop-gradient change specifically introduced or fixed.

### Caveats

- History is pulled via wandb's `run.history(keys=[...])`, which only returns rows where every requested key was logged at the identical step; this repo's training loop happens to log `epoch` in the same call as `test_oracle/*` (confirmed above), but this is an implementation detail of the current logging code, not a documented wandb guarantee — a future change to the logging call structure could silently break this join.
- This audits `test_oracle` only (the oracle-max-over-conditions metric 11.1's headline used), not `test_pre_diff`/predictor metrics.
- 11 logged points per run (epoch 0, 10, 20, ..., 99) describe the shape at the eval cadence training already uses, not a continuous curve — a short-lived spike or partial recovery between eval points would not be visible here.

### Reproduction

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR
python scripts/analyze_training_trajectory.py \
  --tag condition-freeze-ablation-redcaps_150k \
  --pred-coupled-tag pred-stopgrad-ablation-redcaps_150k \
  --out-fig docs/reports/assets/training_trajectory/i2t_gap_trajectory.png
```

## Documentation updates (2026-08-27)

- Reframed the diagnostic as cross-modal neighborhood-disagreement bridge structure, rather than semantic polysemy.
- Defined image-only/text-only edges, bridge nodes, and bridge-pair pull; stated that these are graph-topology and embedding-geometry concepts.
- Clarified that false transitivity is an expected smoothing consequence that is risky to over-interpret, not a demonstrated performance failure.
- Clarified that retrieval/drift cross-reference metrics measure association and magnitude of change, not retrieval quality or causality.
