# Conditional Buddies — Progress Report

**Date:** 2026-06-24 (updated 2026-07-07 with Family #3)
**Branch:** `experiment/conditional_buddy_train` (init/analysis work landed on `experiment/conditional_buddy`)
**Audience:** a CoSiR collaborator who knew the project *before* the buddy work began.

## Executive summary

Over roughly June 9 – July 7, 2026 we built a "conditional buddy" pipeline that gives each
sample's trainable condition vector a content-aware geometric starting point, validated
that the buddy signal is real (not a data artifact) on two datasets, and then began using
that signal *during* training rather than only at initialization. Three independent probes
— category lift, a held-out encoder (DINOv2), and a vision-language-model judge — all
confirm buddies connect samples that share specific multimodal content. **All three** planned
ways of using buddies during training are now implemented and gated behind default-off config
flags: Family #1 (a Laplacian smoothness regularizer on the condition embeddings), Family #2
(buddies as extra contrastive retrieval positives), and Family #3 (a self-refreshing buddy
graph that co-evolves with the model). The large-scale ablations for all three are
launched-but-pending-analysis — the framework is complete; the retrieval numbers are the
remaining work.

---

## 1. Starting point and motivation

**In one sentence:** before buddies, each sample's condition vector was initialized from a
generic CLIP-feature transform; buddies replace that with an initialization that places
samples sharing cross-modal context near each other.

In CoSiR, frozen CLIP image/text features are combined with a per-sample *trainable*
condition (a.k.a. label) embedding `z` of dimension `D = 16`. These `z` vectors are the
only learnable per-sample knobs and they need an initial value. The pre-existing strategy
(`initialization_strategy: imgtxt`) initialized them from `(image − text)` CLIP features
passed through PCA — a generic transform with no notion of which samples are *related*.

The hypothesis behind buddies: **if samples that share cross-modal context start close in
condition space, training has an easier optimization landscape.** Rather than scattering
conditions arbitrarily, seed them from the actual neighborhood structure of the data.

---

## 2. The buddy idea and graph construction

**In one sentence:** a "buddy" is a pair of samples that are mutual nearest neighbors in
*both* CLIP image space and CLIP text space; we build two graphs from this — a strict one
(B) and a broad one (E).

Definitions (introduced here, used throughout):

- **Mutual KNN (mutual-K-nearest-neighbors).** For one modality, draw an edge between
samples *i* and *j* only if *j* is in *i*'s top-K neighbors **and** *i* is in *j*'s
top-K (K=30 throughout). Mutuality filters out hub samples that everyone points to but
that don't point back. We do this once for image features (`A_img`) and once for text
features (`A_txt`).
- **Conditional buddy / the two graphs.** Combining the two per-modality mutual graphs
gives:
  - **B = A_img ∩ A_txt** (strict intersection) — an edge survives only if the pair is
  mutual-KNN in *both* modalities. Precise but sparse.
  - **E = A_img ∪ A_txt** (union) — an edge survives if the pair is mutual-KNN in *either*
  modality. Broad and well-connected. **E is the graph used for initialization** because
  it leaves few isolated nodes.

Edges are then weighted by cosine distance (`1 − cosine similarity`) computed per modality
on the existing edges only, rank-normalized within each modality so the two scales are
comparable, and mixed with a weight `α` (default `α=0.5`, equal image/text). Source notes:
`.claude/conditional_buddies_init.md`; implementation in `src/conditional_buddy/buddy_graph.py`
(`mutual_knn`, `union_graph`).

**Connectivity work (June 23).** On diverse 1:1 data, E can *fragment* into many
disconnected components (54 on RedCaps-150K). This matters because the spectral
initialization below reads the graph's Laplacian, and a disconnected graph wastes its
leading dimensions on "which component am I in?" indicator vectors instead of topical
structure. We added `ensure_connected` (design: `docs/superpowers/specs/2026-06-23-buddy-connectivity-design.md`):
label components, pick a per-component *medoid* in the mix-weighted concat feature
`[√α·img, √(1−α)·txt]`, build a minimum spanning tree over the medoids, and add the `C−1`
bridge edges. It is default-on and a no-op when E is already connected, so Impressions and
COCO are unchanged. On RedCaps it added 53 bridges over 54 components → a single connected
graph. Note the older `ensure_min_degree` guarantees degree ≥ 1 but **not** connectedness;
that gap is what `ensure_connected` closes.

---

## 3. Buddy-based initialization

**In one sentence:** embed graph E into 16 dimensions with Laplacian Eigenmaps, rank-
normalize, and write the result into the condition store as the new `buddies` init strategy.

**Spectral embedding / Laplacian Eigenmaps.** This is a graph-layout method: it places
connected nodes near each other by minimizing the *smoothness energy*

> `Σ_{(i,j)∈E} w_ij · ‖z_i − z_j‖²`

— in plain English, "neighbors in the graph should have similar coordinates." Solving this
(subject to a normalization) is an eigenvalue problem on the graph Laplacian; the lowest
non-trivial eigenvectors give the embedding. This produces the `[N, 16]` init, which is
normalized and copied into the condition store. It is a drop-in `initialization_strategy: buddies` sibling of `imgtxt`, and the (expensive) graph is computed once per dataset and
cached as a template.

**Two notable fixes that made the init usable:**

- **The collapse problem (June 9).** A first 2-D run looked numerically great
(buddies ~20× closer than random) but the plot showed nearly all 12k samples collapsed
to the origin. The cause was *eigenvector localization* on uneven-degree graphs combined
with z-score normalization dividing by an outlier-dominated standard deviation. The fix:
replace z-score with **per-dimension rank normalization** (map each dimension's values to
evenly-spaced ranks in [−1, 1]), which guarantees the points fill the space while
preserving neighborhood order. After the fix the embedding spreads out and buddies stay
clearly closer than random. Evidence: `docs/reports/assets/{collapse_comparison,collapse_zscore,fixed_rank}.png`;
narrative in `docs/reports/2026-06-09_weekly_conditional_buddies.md`.
- **The eigensolver fix (June 23).** ARPACK shift-invert never finished on the real,
fragmented RedCaps-150K Laplacian. We switched to a matrix-free `pyamg` (`amg`) solver
(`src/conditional_buddy/embedding_methods.py`), which handled 150K in seconds.

Engineering forks worth knowing (from the June 9 weekly): FAISS was replaced with exact GPU
brute-force `topk` (no new dependency, with a clean cuVS seam for scale); SMACOF/MDS was
dropped because scikit-learn 1.8 removed the missing-edge weight argument it needed.

---

## 4. Analysis and validation

**In one sentence:** across three independent signals and two datasets, buddies demonstrably
connect samples sharing real, specific content — strongest on the strict B graph.

### 4a. Dimensionality and hyperparameter study (June 9, Impressions)

Report: `docs/reports/2026-06-09_buddies_dim_hparam_study.md`. Key measured findings:

- **n_dim=16 is the sweet spot.** "kNN preservation" (fraction of a sample's graph
neighbors that remain its nearest neighbors in the embedding) rises from **6.7% at 2-D**
to **72% at 16-D**, and only **79% at 32-D** — diminishing returns past 16, which also
matches the model's `embedding_dim`. The flat buddy/random distance ratio (~0.06 at every
dim) is *misleading*; kNN preservation is the honest metric. Structure is genuinely
high-dimensional (participation ratio ≈ 0.83·n_dim).
- **α (image/text mix) dominates K.** On Impressions, image-weighted distances are cleaner
(α=1.0 metric-best) because each image carries four very different caption styles, making
text neighborhoods noisy. Caveat noted in the report: α=1.0 discards the cross-modal
premise, and the intrinsic metrics are biased toward image self-consistency, so the
recommendation is to keep **α≈0.5** and let downstream retrieval arbitrate.

### 4b. Are buddies meaningful? (June 22, Impressions)

Report: `docs/reports/2026-06-22_buddy_analysis.md`; assets under `assets/buddy_analysis/`.
The confound: Impressions has only **814 source photos behind 12,123 records**, so 80% of
strict-B edges literally connect the *same photo* in a different caption style. We controlled
for this by splitting every metric into within-photo vs **cross-photo** edges. Measured:

- **Type lift** (observed/chance co-occurrence of caption types over edges): 1.5× overall,
rising to **2–3× on cross-photo edges** — the signal gets *stronger*, not weaker, once
same-photo edges are removed.
- **Held-out DINOv2** (a self-supervised encoder the graph never saw): cross-photo buddies
sit at cosine distance **0.39 (B) / 0.65 (E)** vs **0.95 for random** different-photo
pairs — type-free confirmation.
- **VLM judge (Qwen2.5-VL-7B):** a strict buddy's caption describes the anchor image **74%**
of the time (B) vs ~1% for type-matched random captions.

Recurring theme: **B ≫ E** on every quality probe; E buys connectivity at the cost of
looser neighbors.

### 4c. RedCaps extension (June 23) — does it generalize off near-duplicates?

Report: `docs/reports/2026-06-23_redcaps_buddy.md`; assets under `assets/redcaps_buddy/`.
RedCaps is genuinely **1 image : 1 caption** (no near-duplicates), with **350 subreddits** as
free ground truth. Every buddy edge here is cross-content by construction. Measured:

- **B is rare on clean data:** 82.5% of samples have *no* strict buddy (mutual-NN-in-both is
a high bar when every image is unique), so the "prefer B" lever from Impressions is moot —
init genuinely needs E. E stays well-connected (avg degree 19).
- **Subreddit lift ~20×** (B 19.5×, E 22.8×) — an order of magnitude above Impressions' type
lift.
- **Held-out DINOv2:** buddy **0.39 (B) / 0.59 (E)** vs random 0.97 — almost exactly matching
Impressions' cross-photo numbers.
- **VLM judge:** strict-buddy caption describes the anchor image **81%** vs **7%** for a
*same-subreddit* (hard) negative and 1% for a random caption. The clean
buddy ≫ same-topic ≫ random gradient shows buddies capture **specific** content, not just
broad topic.

**Verdict:** the buddy signal is not a near-duplicate artifact; it generalizes.

One important correction the RedCaps run surfaced about the *init space* (vs the per-edge
signal): on the real, fragmented 150K graph, a 16-d spectral init was **component-dominated
and topically uninformative** (KMeans ARI vs subreddit ~0.02, per-dim MI ~0.15). The
`ensure_connected` bridging fix from §2 recovered it (ARI ~0.15, per-dim MI ~1.1). So the
per-edge buddy signal was always strong; it only reaches the init once connectivity is
handled. The corrected reading of the init space is a **smooth topical manifold** (related
subreddits blend rather than forming hard clusters; silhouette stays negative), not a
discrete cluster-per-subreddit partition.

---

## 5. Using buddies during training — Family #1 (smoothness regularizer)

**In one sentence:** keep the validated buddy geometry alive during training by re-minimizing
the *same* Laplacian smoothness energy that produced the init, so the init becomes a target
we hold rather than a starting point we drift from.

Design: `docs/superpowers/specs/2026-06-23-buddy-train-regularizer-design.md`; plan:
`docs/superpowers/plans/2026-06-23-buddy-train-regularizer.md`; change log:
`.claude/20260623_buddy_train_log.md`. Implemented in `src/metrics/regularizer.py` and
`src/hook/train_cosir.py`.

**Motivation.** Until now the buddy signal was used *only* to initialize `z`; afterward `z`
trains freely under the contrastive loss and the buddy geometry is free to wash out. This is
the first of three staged ways to use the signal beyond init — the easiest and most
self-contained.

**The term.**

> `L_buddy = (1/|S|) · Σ_{(i,j)∈S} ‖z_i − z_j‖²`

i.e. "on a sample S of buddy edges, keep paired conditions close" — exactly the Laplacian
Eigenmaps energy, on raw `z` with no re-normalization so it matches the init energy.

**How it's wired (the non-obvious part).** Random training batches almost never contain a
buddy *pair*, so an in-batch-only term would rarely fire. Instead, each step takes the
batch's anchor positions, looks up their E-neighbors via a persisted CSR neighbor index,
samples up to `s` buddies per anchor directly from the **full** `[N, D]` differentiable
embedding table (so gradients flow to both anchor and buddy rows, even buddies not in the
batch), and averages `‖z_i − z_j‖²`. Cost is negligible (`batch × s` gathers of 16-d
vectors). To make this possible, `compute_buddy_init(..., return_edges=True)` now also
returns E's edge list (remapped to z-table order), persisted as `buddy_edges.npy` next to the
init template and threaded through template copy/restore.

**Config / gating** (in `cfg.loss`, default-off → byte-for-byte backward compatible):


| key                 | default | meaning                             |
| ------------------- | ------- | ----------------------------------- |
| `lambda_buddy`      | `0.0`   | term weight; **0 = off**            |
| `buddy_reg_samples` | `4`     | buddies sampled per anchor per step |
| `buddy_reg_graph`   | `"E"`   | graph source (only E wired)         |


The term only fires when `embedding_manager.embeddings.requires_grad` (the condition-learning
phase). A `drift_from_init` diagnostic (mean `‖z − z_init‖`, logged for every run including
the baseline) was added so post-sweep analysis can tell "regularizer inert" from "active but
redundant."

**Run / ablate.** Smoke: `scripts/run_buddyreg_smoke.sh`. Full focused ablation:
`scripts/run_buddyreg_full.sh` (sweeps `lambda_buddy ∈ {0, 0.1, 0.3, 1.0}`, 0 = baseline).
Because `lambda_buddy` is **not** part of the template key, every arm reuses the *same* buddy
init, so 0 vs >0 differ only by the training term — a clean ablation. A wider grid
(`lambda_buddy × lr × lr_label`, 40 runs) lives in `scripts/sweep_config_v4.yaml`, analyzed
by `scripts/analyze_buddyreg_sweep.py` (paired per-cell ΔR1 table, λ × lr_label interaction,
drift-by-λ). Unit + smoke tests in `src/test/20260623_buddy_train_reg/`.

---

## 6. Using buddies during training — Family #2 (contrastive supervision)

**In one sentence:** instead of only keeping conditions smooth, use buddies as *extra
retrieval positives* in the combined/retrieval space where the evaluation metric actually
lives.

Design: `docs/superpowers/specs/2026-06-24-buddy-contrastive-supervision-design.md`; plan:
`docs/superpowers/plans/2026-06-24-buddy-contrastive-supervision.md`; change log:
`.claude/20260624_log.md`. Implemented in `src/metrics/regularizer.py`
(`buddy_contrastive_loss`, `reorder_features_to_z`) and `src/hook/train_cosir.py`.

**Why a second family, and how it differs from #1.** Family #1 pulls the *z-space* lever
(smoothness on the condition embeddings). A contrastive loss on `z` would just be a stronger
version of the same lever. Family #2 deliberately acts in **combined / retrieval space** —
the space the t2i/i2t R1 metric lives in — so it can move the metric in a way #1 may not.
Holding the graph (E) and the buddy init template constant across #1 and #2 isolates
"smoothness vs contrastive" as the only difference.

**Mechanism.** For each batch anchor, pull its fused feature `comb_emb` toward its buddies'
**projected other-side features** (`project_other(other-side pooled feature)`), using the
batch's in-batch `other_emb` as negatives (anchor's own row masked), via a temperature-scaled
multi-positive InfoNCE (SupCon):

> `L_i = −(1/K) Σ_k log[ exp(s_ik/τ) / (Σ_k exp(s_ik/τ) + Σ_neg exp(s_in/τ)) ]`

i.e. "make the anchor's fused feature retrieve its buddies' images alongside its own." The
buddy's underlying pooled feature is **frozen**; gradient flows into `z_i` (anchor),
`project_other`, and the combiner — never into buddy `z_j` — which keeps the supervision
stable and avoids collapse. The combine side determines the target: `combine_side=="img"` →
target is `project_other(txt_j)`; `=="txt"` → `project_other(img_j)`.

**Implementation reuse.** It reuses Family #1's persisted `buddy_edges.npy` and CSR (no new
persistence). It builds a one-time frozen `other_feat_table` `[N, Dfeat]` of the
non-combine-side pooled feature in z-table order. Streaming feature stores (no RAM feature
table) are out of scope (warned + disabled).

**Config / gating** (read via `getattr`, not in YAML → add on the CLI with `+`):


| key                     | default | meaning                    |
| ----------------------- | ------- | -------------------------- |
| `lambda_buddy_con`      | `0.0`   | term weight; **0 = off**   |
| `buddy_con_samples`     | `4`     | buddy positives per anchor |
| `buddy_con_temperature` | `0.07`  | InfoNCE temperature        |


Independent of `lambda_buddy`, so a clean 2×2 (#1 alone / #2 alone / both / neither) ablation
shares one init. A `buddy_con_alignment` diagnostic (mean cosine between each anchor's
`comb_emb` and its buddy positives) is the #2 analogue of #1's `drift_from_init`.

**Run / ablate.** `scripts/run_buddycon_full.sh` sweeps `lambda_buddy_con ∈ {0, 0.3}` with
Family #1 held off, isolating the contrastive term. Tests in
`src/test/20260624_buddy_contrastive/` (positive-gather correctness, gradient direction,
isolated-anchor zero, self-masking, index alignment, temperature sanity).

---

## 7. Using buddies during training — Family #3 (self-refreshing buddies)

**In one sentence:** stop freezing the buddy graph at init — periodically rebuild it from the
model's *own* current representations during training, so the supervision co-evolves with the
model instead of staying pinned to the original frozen-CLIP neighborhood.

Design: `docs/superpowers/specs/2026-07-07-buddy-self-refresh-design.md`; plan:
`docs/superpowers/plans/2026-07-07-buddy-self-refresh.md`; change log: `.claude/20260707_log.md`.
Implemented in `src/metrics/regularizer.py` (`refresh_buddy_graph`, `edge_jaccard`) and
`src/hook/train_cosir.py`.

**The gap it closes.** Families #1 and #2 both reuse *one* graph: the init-time cross-modal
CLIP mutual-KNN (E). That graph reflects what *frozen CLIP* judged similar before any training
happened. But once the model has learned, its own combined/retrieval space may know better
neighbors than CLIP did. Family #3 tests exactly that hypothesis: recompute buddies from the
*evolving* combined features and see whether a live graph beats the frozen one.

**Mechanism.** On a warm-up-then-periodic schedule (default: skip until epoch 50, then refresh
every 50 epochs), and entirely without gradients: (1) do one pass over all samples to get each
sample's *current* combined feature; (2) build a fresh mutual-KNN graph from those; (3) union it
with the frozen CLIP graph — the CLIP edges are **always kept**, the fresh graph only *adds*;
(4) hand the rebuilt graph to Family #2's contrastive term, which simply reads a new neighbor
index. Family #3 adds **no new loss** — it is "Family #2 with a live graph," which makes the
comparison clean: hold everything else fixed and flip only static-vs-refreshed.

**Two guards against a co-training trap.** Recomputing buddies from the very features a loss then
pulls together risks a feedback loop — already-close points keep re-selecting each other and the
representation can collapse. Two things prevent it. (a) The CLIP graph is retained as a permanent
anchor: refresh only *adds* edges, never abandons the validated init. A single "blend" knob
controls how much of the fresh graph is added, and **blend = 0 reproduces Family #2 exactly**, so
the ablation is a continuum with a built-in baseline. (b) The contrastive *target* stays the
buddy's **frozen** CLIP feature (inherited from #2), so the loss never pulls a moving feature
toward another moving feature — only the *choice of who your buddies are* is dynamic.

**A subtle bug worth flagging (caught in code review).** The graph must be rebuilt from the
model's *inference* features. In the first cut, the recompute accidentally ran with the model in
"training" mode, which leaves dropout active — so the graph was built from half-randomized
features and then frozen for 50 epochs, which would have quietly made the whole experiment
meaningless. The fix switches the model to eval mode for the recompute pass only; a regression
test now guarantees two recomputes of the same model produce the *identical* graph.

**Config / gating** (`cfg.loss`, read via `getattr`, default-off → byte-for-byte backward
compatible):


| key                    | default | meaning                                          |
| ---------------------- | ------- | ------------------------------------------------ |
| `buddy_refresh`        | `False` | master switch; **False = off**                   |
| `buddy_refresh_warmup` | `50`    | first refresh epoch (avoids the epoch-0 no-op)   |
| `buddy_refresh_period` | `50`    | refresh every R epochs                           |
| `buddy_refresh_blend`  | `1.0`   | fraction of fresh edges added (**0 = static #2**) |
| `buddy_refresh_k`      | `30`    | mutual-KNN K for the comb-space graph            |


Two diagnostics were added: `graph_churn` (how much the graph changes between refreshes —
thrashing vs stabilizing over training) and `graph_new_edge_frac` (how much the model's
neighborhood *disagrees* with CLIP).

**Run / ablate.** `scripts/run_buddyrefresh_full.sh` sweeps `buddy_refresh_blend ∈ {0, 1.0}` with
Family #1 held off and #2 on — `0` = static #2 baseline, `1.0` = full refresh — isolating
"static vs refreshed graph" as the only difference. Smoke: `scripts/run_buddyrefresh_smoke.sh`.
Unit + regression tests in `src/test/20260707_buddy_refresh/` (7 tests, incl. the eval-mode
determinism guard). Because no `buddy_refresh*` key is part of the buddy init template, every arm
reuses the *same* init — the full `{#1, #2, #3}` matrix shares one buddy initialization.

---

## 8. Current status and what's next

**Validated (with measured numbers, cited above):**

- The buddy signal is real and content-specific on both Impressions and RedCaps (lift,
DINOv2, VLM all agree; B ≫ E on quality, E needed for connectivity).
- n_dim=16 is the right init dimensionality; rank normalization and the connectivity bridge
are both necessary for the init to carry structure.

**Implemented but pending analysis:**

- **Family #1** (smoothness regularizer), **Family #2** (contrastive supervision), and
**Family #3** (self-refreshing graph) are all coded, unit-tested, and gated default-off. The
large-scale ablation sweeps are **launched but not yet analyzed** — there are no downstream
retrieval (R1) numbers to report yet for any family. `analyze_buddyreg_sweep.py` exists to
crunch the #1 sweep once it lands; its verdict guide explicitly anticipates the "active but
redundant → #2 is the real test" outcome, which is the motivation for #2. #2 and #3 reuse the
same init template as #1, so the whole `{#1, #2, #3}` matrix can be swept from one buddy
initialization.
- **The staged buddy program is now feature-complete.** All three planned ways of using the
signal beyond init exist behind flags; what remains is running the ablations and reading the
retrieval outcomes — not more implementation.

**Planned next:**

- **Analyze the three ablations** and decide which lever (if any) moves R1: does keeping the
buddy geometry smooth (#1), supervising retrieval with buddy positives (#2), or letting the
graph co-evolve (#3) help — and do they compound or overlap?
- The recurring **B ≫ E** finding suggests testing an init/supervision that leans on B where
it exists and falls back to E only for B-isolated nodes — noted as a follow-up in the
analysis reports.

---

## Appendix — key paths

- Pipeline source: `src/conditional_buddy/{buddy_graph,compute_buddies,embedding_methods,init_conditions,visualize}.py`
- Training-time terms: `src/metrics/regularizer.py` (families #1/#2/#3); wiring in
`src/hook/train_cosir.py`
- Init notes: `.claude/conditional_buddies_init.md`
- Designs: `docs/superpowers/specs/2026-06-{09,22,23,24}-*.md`, `docs/superpowers/specs/2026-07-07-buddy-self-refresh-design.md`
- Plans: `docs/superpowers/plans/2026-06-{23,24}-*.md`, `docs/superpowers/plans/2026-07-07-buddy-self-refresh.md`
- Reports: `docs/reports/2026-06-{09,22,23}_*.md`; figures under `docs/reports/assets/`
- Change logs: `.claude/{20260609,20260623,20260623_buddy_train,20260624,20260707}_log.md`
- Run/ablation: `scripts/run_buddyreg_{smoke,full}.sh`, `scripts/run_buddycon_full.sh`,
`scripts/run_buddyrefresh_{smoke,full}.sh`, `scripts/sweep_config_v4.yaml`,
`scripts/analyze_buddyreg_sweep.py`
- Tests: `src/test/{20260623_buddy_train_reg,20260624_buddy_contrastive,20260707_buddy_refresh}/`

