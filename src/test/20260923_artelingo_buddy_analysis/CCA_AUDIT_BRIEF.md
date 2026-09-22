# Brief: linear CCA + conditional-residual audit (go/no-go diagnostic)

Write a new script
`src/test/20260923_artelingo_buddy_analysis/run_cca_audit_pilot.py`.
Do NOT run it — execution happens separately, on GPU, outside this task.

## Context and why this run matters

Two brainstorms
(`fusion_brainstorm_codex_findings.md`, `joint_fusion_brainstorm_codex_findings.md`)
independently concluded that before building any heavier joint-fusion
mechanism (learned two-teacher student, SNF, co-regularized spectral
clustering), we should first cheaply measure whether there is ANY stable
linear relationship between content and affect at all — below the level of
"are they mutual top-20 kNN neighbors" (already measured as nearly empty:
98.96% of nodes had zero shared edge in the raw intersection). This is a
**diagnostic audit, not a new fusion method** — its purpose is to determine
whether pursuing any joint-representation approach further is worth the
effort, with a predeclared decision rule, not to produce a usable graph.

This has two parts:
1. **Linear CCA**: is there a reproducible (held-out-validated) linear
   subspace of CLIP content features and GoEmotions affect features that is
   correlated across the same paintings?
2. **Conditional residual**: after removing whatever affect content DOES
   explain (via linear regression), does the LEFTOVER (residual) affect
   signal still carry real, useful emotion structure? This distinguishes "no
   real signal" from "real signal that is specifically orthogonal to
   content" — the latter is consistent with the hierarchical-refinement
   pilot's own result (real affect signal, but not reconcilable with
   genre-preserving partitioning).

Read `run_pipeline.py`, `run_affect_pilot.py`, `run_single_modality_pilot.py`,
and `run_bert_heldout_pilot.py` in full first — this reuses conventions from
all four, especially `run_bert_heldout_pilot.py`'s pattern for redirecting
reused pipeline helpers at the held-out feature store (`artelingo_val_test.json`
/ `/data/SSD2/pre_extract/artelingo_heldout/features`, 9,365 unique
paintings, verified zero painting-level overlap with train).

## Implementation

### 1. Load train and held-out data

- Train: `pipeline.assert_extraction_complete()`, `pipeline.load_dedup_features()`
  → `(paintings, img_nodes, txt_nodes, emotion_counts)`. Get `majority_emotion`.
- Held-out: load a SEPARATE `run_pipeline.py` module instance (fresh
  `load_sibling_module` call), reassign `.STORAGE_DIR =
  "/data/SSD2/pre_extract/artelingo_heldout/features"` and `.TRAIN_JSON =
  "/data/PDD/artelingo/artelingo_val_test.json"` BEFORE calling its
  functions — exact pattern from `run_bert_heldout_pilot.py`. Get
  `(heldout_paintings, heldout_img_nodes, heldout_txt_nodes,
  heldout_emotion_counts)` and `heldout_majority_emotion`.
- Extract GoEmotions affect nodes for BOTH splits via
  `affect_pilot.extract_affect_nodes(json_path, paintings, device)` — train
  uses `pipeline.TRAIN_JSON` (the ORIGINAL train pipeline's constant, not the
  patched one), held-out uses the heldout JSON path directly.

### 2. Build content features and fit PCA (train only)

```python
CONTENT_PCA_DIM = 50
SEED = 42
```
- `content_train = concatenate(l2_normalize(img_nodes), l2_normalize(txt_nodes))`,
  same for `content_heldout` using the held-out arrays.
- Fit `sklearn.decomposition.PCA(n_components=CONTENT_PCA_DIM,
  random_state=SEED)` on `content_train` ONLY. Transform both train and
  held-out through this fitted PCA. Affect features are used at their
  native 28 dimensions (already low-dimensional relative to N — no PCA
  needed there).

### 3. Linear CCA (train-fit, held-out-evaluated)

```python
CCA_N_COMPONENTS = 10
N_PERMUTATIONS = 20
REAL_SIGNAL_CORR_THRESHOLD = 0.15
```
- Fit `sklearn.cross_decomposition.CCA(n_components=CCA_N_COMPONENTS)` on
  `(content_train_pca, affect_train)` ONLY.
- Transform held-out content and held-out affect through this fitted CCA.
  Compute the Pearson correlation between each of the 10 held-out canonical
  component pairs (`numpy.corrcoef` per column pair) — this is the real
  held-out canonical correlation per component.
- **Permutation null**: for `seed in range(N_PERMUTATIONS)` (seeds 0..19),
  randomly permute the ROW correspondence of `affect_train` relative to
  `content_train_pca` (breaking the true painting-to-painting pairing,
  `numpy.random.default_rng(seed).permutation`), refit a fresh CCA on this
  scrambled train pairing, transform the SAME (unpermuted, correctly-paired)
  held-out data through it, and record its per-component held-out canonical
  correlations. This gives a null distribution of 20 values per component.
- For each of the 10 components, report: real held-out correlation, the
  null distribution's mean/95th-percentile, and whether the real value
  exceeds the null's 95th percentile.
- **Decision rule**: "stable shared signal found" = the TOP (first)
  component's real held-out correlation both exceeds
  `REAL_SIGNAL_CORR_THRESHOLD` (0.15) AND exceeds its own null
  distribution's 95th percentile. State this explicitly; do not blur it
  with softer language.

### 4. Held-out edge-retrieval check

- Build held-out content and held-out affect mutual-kNN graphs
  independently: `mutual_knn(l2_normalize(content_heldout... )` — actually
  reuse `single_modality_pilot.build_single_modality_graph()` twice, once
  for held-out content (pass `content_heldout` — the raw concatenated,
  un-PCA'd content array, since this graph is a ground-truth reference, not
  part of the CCA pipeline) and once for held-out affect (pass the held-out
  GoEmotions array), both with `expected_nodes=len(heldout_paintings)`.
- Project held-out content and held-out affect into the fitted CCA space
  (via the train-fitted PCA+CCA pipeline). Build a THIRD mutual-kNN graph
  directly on the concatenation of the two held-out CCA projections (this is
  the "joint CCA space" graph).
- For a random sample of 2,000 held-out nodes (`numpy.random.default_rng(SEED).choice`),
  compute: what fraction of each sampled node's TRUE content-graph neighbors
  are ALSO neighbors in the joint-CCA-space graph (recall-style overlap),
  and same for TRUE affect-graph neighbors. Report both fractions, plus the
  same two fractions computed using RANDOM neighbor sets of matching size as
  a chance floor (same predeclared sample, `numpy.random.default_rng(SEED)`
  for the random comparison sets).

### 5. Conditional residual (partial CCA / residualization)

```python
```
- Fit ordinary least-squares linear regression `affect_train ~
  content_train_pca` (train only) via `sklearn.linear_model.LinearRegression`
  (`numpy.float64` inputs are fine here — small matrices). Report R²
  (`sklearn`'s `.score()`, on held-out: fit on train, score on held-out
  content/affect pair) as "fraction of affect variance explained by
  content."
- Compute `residual_affect_train = affect_train - model.predict(content_train_pca)`
  and `residual_affect_heldout = affect_heldout - model.predict(content_heldout_pca)`.
- Build a mutual-kNN graph from `residual_affect_train` alone via
  `single_modality_pilot.build_single_modality_graph()` (same K/repair
  convention as every other single-modality pilot), run Leiden seed=42,
  compute emotion/genre AMI via `pipeline.external_metrics()` +
  `pipeline.load_genre_map()`.
- **Decision rule**: "informative residual" = residual-affect-only emotion
  AMI retains at least 80% of the raw GoEmotions-affect-only reference
  (0.1180 × 0.8 = 0.0944). State plainly whether this holds.

## Report

Write `src/test/20260923_artelingo_buddy_analysis/cca_audit_pilot_report.md`:

- Explain both parts in plain language up front (what CCA measures, what
  the residual test measures, and why together they form a go/no-go gate
  for the joint-fusion methods evaluated in the second brainstorm).
- **CCA results table**: all 10 components' real held-out correlation, null
  mean, null 95th percentile, and pass/fail against both predeclared
  criteria.
- State the CCA decision rule's outcome plainly (stable shared signal found:
  yes/no).
- **Edge-retrieval table**: content-graph recall and affect-graph recall for
  the joint-CCA-space graph, versus the random-neighbor chance floor.
- **Residual results**: held-out R² (content explaining affect), and the
  residual-affect-only emotion/genre AMI table row next to the raw
  GoEmotions-affect-only reference (0.1180/0.0396).
- State the residual decision rule's outcome plainly (informative residual:
  yes/no).
- A final synthesis paragraph combining both parts' outcomes into one of
  four honest conclusions: (a) no shared signal and no informative residual
  — supports abandoning joint-fusion methods entirely; (b) shared signal
  found — licenses a small learned-student pilot per the second
  brainstorm's ranking; (c) no shared signal but an informative residual —
  supports the "affect is real but orthogonal to content" reading, pointing
  toward conditional/two-output representations rather than any single
  joint graph; (d) mixed/inconclusive — say so plainly, do not force a
  clean narrative if the two parts disagree.

Print clear timestamped progress logs matching the other scripts' `log()`
format throughout.
