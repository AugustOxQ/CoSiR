# Cross-modal polysemy: does the buddy graph already reflect it, and is any B-C pull real signal or a "false transitivity" artifact?

**Date:** 2026-08-26 · **Dataset:** RedCaps, 150,000 rows (matches C5/Exp 9-11's scale) · **Branch:** `experiment/condition_drift_retrieval_correlation`
**Code:** `scripts/analyze_polysemy_bridges.py`, `src/conditional_buddy/buddy_graph.py` (`classify_edges`/`bridge_node_stats`)
**Spec:** `docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md` §4 Experiment 12
**Precursor:** `src/test/20260824_buddy_graph_disagreement/diagnose_disagreement.py` (found ~80% of nodes are "bridge" nodes; this experiment measures what that structurally implies for the resulting embedding)

---

## TL;DR

**Bridge-node structure is real and pervasive (80.2% of nodes), the buddy-init embedding does pull a bridge node's image-only and text-only neighbors (B, C) together — strongly and reliably (mean pull = +1.98 embedding-distance units, mean/SEM = +102.1, 91.4% of 5,000 sampled pairs pulled closer than baseline) — but that pull is only very weakly explained by real shared-neighbor overlap (Spearman rho = +0.076 between shared-neighbor Jaccard and pull magnitude; statistically significant at this sample size, p = 8.6e-8, but rho² ≈ 0.006 means shared-neighbor structure accounts for well under 1% of the pull's variance).** This lands closer to the spec's **second branch — "real + ungraded" — confirming Experiment 10's flagged "false transitivity" risk as an actual, measurable distortion**, not the "real + graded" branch: most of the pull looks like a broad, largely content-independent effect of the spectral embedding's global smoothing (consistent with Experiment 10's own finding that this method is near-invariant to within-class affinity structure once a dominant edge class is present — see that report's Laplacian-rescale-invariance argument), rather than the embedding faithfully tracking how much B and C's neighborhoods actually overlap. Separately, the per-node polysemy label does **not** significantly predict per-sample retrieval-rank change in this run's population (`corr(is_polysemic, |delta_rank|)` rho = +0.016, p = 0.37) — a null on the retrieval/drift cross-reference.

## Method

Reused the already-completed buddy-init template (`res/CoSiR_condition_freeze_ablation/redcaps_150k/template_embeddings/`, the same one 11.1/11.2/11.3 trained against) and rebuilt `A_img`/`A_txt`/`E` from cached RedCaps-150k features (K=30, alpha=0.5) via `classify_edges` to identify bridge nodes and label every node (`img_only_only` / `txt_only_only` / `bridge` / `neither`). Sampled 5,000 bridge-node (A, B, C) triples — B an image-only neighbor of A, C a text-only neighbor — and compared each pair's embedded L2 distance against a degree-decile-matched, non-adjacent baseline pair. Checked whether that pull correlates with each pair's shared-neighbor Jaccard overlap in `E` (the LINE/GraRep-style second-order-proximity score). Cross-referenced the per-node polysemy label against a 150k-population, 3,000-query-sample per-sample retrieval-rank/condition-drift dump from one already-completed 11.1 `trained` run (seed 1, paired against its seed-1 `frozen` counterpart), produced by extending `scripts/analyze_condition_retrieval_correlation.py`'s `analyze_pair()` with a new opt-in `--dump-per-sample` flag (Task 2 of this experiment's implementation plan).

## Results

### Bridge structure

```
bridge nodes: 120,324 / 150,000 (80.2% of nodes)
label counts: {'neither': 1737, 'img_only_only': 21765, 'txt_only_only': 6174, 'bridge': 120324}
```

The 80.2% bridge-node fraction matches Experiment 10's own diagnostic figure on this same dataset almost exactly — a strong cross-check that this experiment's rebuilt graph is consistent with the earlier one. Very few nodes (1,737, 1.2%) have no img-only or txt-only edges at all (`neither`) — the large majority of the graph's non-`both`/`repair` edges belong to bridge nodes, not to nodes with only one edge type.

### False-transitivity audit

```
sampled bridge pairs: 5,000
pull (baseline_dist - bc_dist): mean=+1.9786 (n=5000, frac_pulled_closer=0.914)  mean/SEM=+102.1 *
grading check: corr(shared_neighbor_jaccard, pull) rho=+0.076 p=8.619e-08
```

The pull is large and essentially universal: 91.4% of sampled (B, C) pairs sit closer together in the buddy-init embedding than their degree-matched baseline, and the effect clears the project's mean/SEM ≥ 2 significance bar by two orders of magnitude (+102.1). But the grading check — whether that pull scales with how much B and C's neighborhoods actually overlap — comes back statistically significant only because of the large sample size (p = 8.6e-8), not because the relationship is practically strong: rho = +0.076 means shared-neighbor Jaccard explains roughly 0.6% (rho²) of the variance in pull magnitude. The overwhelming majority of the pull is not accounted for by real shared-neighbor structure.

### Retrieval/drift cross-reference

```
retrieval cross-reference (n_joined=3000):
  neither:        n=38   median|delta_rank|=10.0  median_drift=0.0830
  img_only_only:  n=422  median|delta_rank|=39.0  median_drift=0.0850
  txt_only_only:  n=139  median|delta_rank|=14.0  median_drift=0.0689
  bridge:         n=2401 median|delta_rank|=13.0  median_drift=0.0780
  corr(is_polysemic, |delta_rank|): rho=+0.016 p=3.680e-01
```

The formal test — is a sample being polysemic *at all* (any of the three non-`neither` labels) correlated with how much its retrieval rank moved between the frozen and trained arms — is a clean null (rho = +0.016, p = 0.37, far from significant). Descriptively, the small `img_only_only` subgroup (n=422) shows a notably higher median `|delta_rank|` (39) than the other three groups (10-14), but this is a single subgroup observation, not the designed statistical test, and should not be read as a confirmed effect without a dedicated follow-up.

## Interpretation

This experiment answers the three questions from the brainstorming session concretely:

1. **Is polysemy reflected in the current graph structure and initialization?** Yes, structurally — `classify_edges` already labels the img-only/txt-only edge pattern (A-B, A-C in the running example) explicitly, and 80.2% of nodes exhibit it. But the *embedding* reflects it in a way that is largely non-specific: B and C get pulled together reliably, but mostly independent of whether their neighborhoods actually overlap in content-relevant ways.
2. **Is this modality-aware — do we know it's image-side or text-side polysemic?** Yes, trivially — `classify_edges`'s per-edge typing already carries this (confirmed again here), and the per-node label (`img_only_only`/`txt_only_only`/`bridge`) makes it queryable per sample. What this experiment adds is that knowing the direction doesn't yet buy anything downstream: the retrieval/drift cross-reference found no significant behavioral signature tied to polysemy type.
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
