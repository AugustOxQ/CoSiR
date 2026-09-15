# Buddy-Graph Prototype Conditioning (Experiment 18)

**Date:** 2026-09-15
**Motivates from:** `docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md` §4 Experiment 17 (17.1's result), `docs/reports/2026-09-15_condition_space_audit.md` (full 17.1 write-up)
**Status:** proposed — pending user sign-off before `writing-plans` produces the task-by-task execution plan
**Branch:** to be forked from `experiment/condition_drift_retrieval_correlation` as `experiment/buddy_prototype_conditioning` once this spec is approved

---

## 1. Goal

Experiment 17.1 established that the buddy-graph signal, once actually probed, is **content-adjacent cluster structure**, not a clean linear semantic axis — RedCaps `warmth` is ~87–90% explained by raw content (companion-animal detection), with only a small (0.059 AUC), not-yet-robustly-established residual beyond content, and `register` has no raw-CLIP signal at all. Meanwhile the trained condition vectors *do* carry real signal beyond a capacity-matched neutral baseline (Addendum, `bb618e7`), so training clearly adds something — the current architecture just isn't the right *kind* of object to make that something interpretable.

This experiment replaces Experiment 17.2's originally-sketched "reserved concept subspace + steering-vector loss" (which assumed a linear axis exists to steer toward — an assumption 17.1 closed) with a design that treats the signal as what 17.1 actually found it to be: discrete, content-adjacent cluster structure. It is informed by PercepT (arXiv 2606.03345), which represents "perception" as a discovered set of visual-textual cluster prototypes with soft attention-pooled assignment, rather than a linear direction — and by a broader literature scan (DICE's content/style subspace decomposition, CoMatch-style neighbor-consistency regularization, noise-robust graph-contrastive learning) synthesized in this session's brainstorm.

## 2. Why not the old cluster-conditioning attempt

CoSiR previously tried cluster-based conditioning (`src/model/clustering.py`, `src/utils/condition_space_evaluator.py`'s `compute_condition_space_quality`): within one training epoch, (1) train with contrastive loss, updating the model and condition vectors, then (2) apply UMAP+HDBSCAN/KMeans clustering to the just-updated conditions and manually force them toward cluster assignments — a second, non-differentiable update the model never saw or adapted to.

This failed for three concrete reasons (per the user, 2026-09-15):
1. No good initialization existed at the time (pre-buddy-graph) — clustering noise gives messy clusters.
2. No way to tell whether the manual pull helped or hurt — it wasn't ablatable or gradient-checked.
3. **Structural bug, not a hyperparameter problem**: the model's combiner adapted to the stage-1 (contrastive-updated) condition vector, never to the stage-2 cluster-pulled one — so whatever the clustering step did was invisible to training.

Per the user's explicit direction: **do not reuse or adapt `clustering.py`** — this design uses fresh code built around fixing failure mode 3 specifically (see §3). It is not the old design with a better initializer; it is a different mechanism (single-stage, fully differentiable) that also happens to inherit a better initializer.

## 3. Architecture

**Single-stage, fully differentiable prototype assignment** — the defining difference from the old design. There is exactly one update per training step; the model never has to adapt to a value it didn't produce itself.

- **Prototype bank** (`PrototypeBank`, new `nn.Module`): P learnable key/value vector pairs in the condition embedding dimension (`embedding_dim: 16`, `configs/model/clip_base.yaml`, unchanged). P is a swept hyperparameter (§7), not fixed here.
- **Community-informed seeding**: run CPU-based community detection (Leiden, e.g. via `leidenalg`/`networkx` — not `cuml`, consistent with this project's documented `libllvmlite.so`/cuml import-chain workaround) once over the existing `union_graph` (`src/conditional_buddy/buddy_graph.py`, **unchanged** — only buddy-graph *construction* is fixed per the user's 2026-09-15 decision carried over from Experiment 17). Each detected community's mean image+text feature seeds one prototype — same spirit as `init_conditions.py`'s existing per-*sample* neighborhood-mean init, applied per-*prototype* instead.
- **Attention pooling**: for each sample, a query built from its frozen CLIP feature attends (single dot-product head, learnable temperature) over the prototype bank's keys; the sample's condition vector is the resulting softmax-weighted sum of prototype values. This vector is what feeds the combiner — replacing `TrainableEmbeddingManager`'s free per-sample lookup as the condition source.
- **Combiner**: `CombinerLowRankAdapter` (`src/model/combiner.py`, already implemented, identity-init) — **unchanged**. It receives the attention-pooled condition vector exactly as it previously received the free per-sample vector; this is a clean interface boundary.
- **Buddy-consistency loss**: `buddy_contrastive_loss` (`src/metrics/regularizer.py:270`) — **unchanged**. It operates on `comb_emb` (the combiner's output), not on the raw condition vector, so it is already agnostic to how that vector was produced. This is what replaces the old design's manual cluster-pull: a soft, ablatable term inside the *same* backward pass, not a second phase.

Net effect: one backward pass per step updates the prototype bank, the attention query projection, and the combiner together, consistently. The buddy graph's influence enters twice, both differentiably: once at prototype seeding (a one-time, non-trainable init) and once via the unchanged buddy-consistency loss (continuous, ablatable, trainable).

## 4. Components (files)

| File | Status | Responsibility |
|---|---|---|
| `src/conditional_buddy/buddy_graph.py` | unchanged | Graph construction (fixed per user constraint) |
| `src/conditional_buddy/prototype_seed.py` | **new** | CPU Leiden community detection over `union_graph`; per-community mean-feature computation |
| `src/model/prototype_bank.py` | **new** | `PrototypeBank(nn.Module)`: key/value store + `seed_from_communities()`; attention-pooling forward |
| `src/model/combiner.py` | unchanged | `CombinerLowRankAdapter` consumes the pooled vector as before |
| `src/metrics/regularizer.py` | unchanged | `buddy_contrastive_loss` operates on `comb_emb`, already agnostic to condition source |
| `src/utils/embedding_manager.py` | **modified** | Persistence path adapted for `O(P·D)` prototype-bank checkpoints instead of `O(N·D)` per-sample vectors (a large size reduction, not a rewrite) |

## 5. Data flow (one training step)

1. Load cached CLIP image/text features (`FeatureManager`, unchanged).
2. Attention-pool each sample's condition vector from the prototype bank.
3. `CombinerLowRankAdapter(text_features, condition_vector)` → conditioned text embedding (as today).
4. Contrastive loss (image vs. conditioned text) — unchanged family.
5. `buddy_contrastive_loss` on `comb_emb` over buddy-graph edges — unchanged function, now implicitly regularizing attention-pooled vectors.
6. One backward pass updates: prototype bank, attention projection, combiner (and the backbone too, only if `freeze_backbone: false` is tested per §7).

## 6. Failure modes introduced by this design (not present in the old free-vector design)

- **Prototype collapse**: attention weight concentrating on 1–2 prototypes for most samples, a known failure mode in codebook/prototype learning (cf. VQ-VAE). Mitigation: monitor per-prototype usage / attention entropy during the local-GPU smoke test (§8) before committing to DAS6 runs; if observed, standard fixes (commitment-style auxiliary loss, temperature annealing) are in scope as a follow-up, not designed preemptively (YAGNI).
- **Cold-start samples with no buddy edges**: `ensure_min_degree`/`ensure_connected` already guarantee every node has ≥1 union-graph edge (existing `buddy_graph.py` behavior, unchanged), so this is inherited, not new.
- **Community count ≠ prototype count P**: Leiden's natural community count over the 150k union graph may not match the swept P values. Mitigation: if natural community count exceeds P, coarsen via a second clustering pass over community-mean features down to P seeds before handing off to `PrototypeBank.seed_from_communities`; if it's fewer than P, the remaining prototype slots initialize from the largest communities' member variance (concrete choice deferred to implementation, not blocking this spec).

## 7. Evaluation & success criteria

- **Local-GPU smoke test first** (small subset, e.g. 1–2k samples): verify gradients flow end-to-end with no NaNs, and check attention-weight entropy / per-prototype usage isn't degenerate before any DAS6 run.
- **RedCaps-150k, 3 seeds each, two arms**, run concurrently on node411/node412: (a) architecture-fix-only baseline — current `CombinerLowRankAdapter` + free per-sample vector, matching 17.2's already-stated floor; (b) prototype-conditioning arm (this design).
- **Retrieval**: `test_oracle`/`test_pre_diff` t2i/i2t R1 vs. the baseline arm, read against this project's standard noise floor (§5 of the publication-plan spec) — a real win or a clean null are both reportable.
- **Interpretability**: reuse 17.1's control-task-gated probe harness (selectivity, not raw accuracy) on the trained condition vectors, **and** `condition_space_evaluator.py`'s existing `silhouette_score` machinery for prototype coherence — the same metric PercepT itself reports (0.97 vs. 0.37 baseline), giving a literature-aligned readout rather than an invented one.
- **P (prototype count)**: swept during the local-GPU smoke test (candidates: 8, 16, 32); the value carried into the 150k DAS6 runs is chosen from smoke-test community-count/usage evidence, not fixed a priori.

## 8. Compute plan

- Local GPU: debug, smoke-test, gradient/collapse sanity checks, P sweep at small scale.
- `node411` / `node412` (DAS6, both already booked): one arm each, run concurrently, 3 seeds per arm at RedCaps-150k.
- 300k/500k: explicitly out of scope for this experiment's first pass — reserved as a stretch goal contingent on the 150k result, matching this plan's existing scale-up discipline (Exp 16, 17.2).

## 9. Execution mode

Same pattern as Experiment 17.1: `subagent-driven-development`, implementation tasks routed through Codex via the `ccg` `codeagent-wrapper --backend codex` bridge, Claude reviewing diffs and running verification. GPU training runs themselves are never launched or held by Codex (per this project's established Codex-session-instability constraint) — only implementation, and read-only monitoring of runs Claude or the user starts directly.

## 10. Relationship to the publication-plan spec

This experiment **supersedes** Experiment 17.2's original sketch (reserved concept subspace + steering-vector regularization toward a linear axis) in `docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md` §4. A short pointer is added there rather than forking the whole spec (the `experiment/two_side_conditioning` spec fork caused real reconciliation cost later — avoided here deliberately). 17.2's architecture-fix-only arm (low-rank, symmetric combiner) is **not** superseded — it is this spec's own baseline arm (§7a), so it is pursued regardless of this experiment's outcome, exactly as originally scoped.

## Out of scope

- Backbone unfreezing: deferred to a secondary axis only if 150k results are inconclusive, per 17.2's original scoping (carried over unchanged).
- Impressions dataset: 17.1 found no usable raw-CLIP signal on either axis there; not a target for this experiment.
- Dynamic (data-driven) prototype count selection à la PercepT's formation stage: P is swept manually (§7), not learned — a fixed, small sweep is enough to answer whether this architecture works at all; automatic P selection is a refinement for later, not a blocker now.

## Self-review

- **Placeholder scan**: no TBD/TODO left; the two genuinely open parameters (P, coarsening-vs-underflow tie-break for community/prototype count mismatch) are explicitly flagged as implementation-time decisions with a stated default direction, not silently skipped.
- **Internal consistency**: §3's claim that `buddy_contrastive_loss` needs no change was verified directly against `src/metrics/regularizer.py:270` (operates on `comb_emb`, agnostic to condition source) before writing this spec, not assumed.
- **Scope**: single architectural change (condition-vector source + its seeding), reusing existing combiner and loss code — right-sized for one implementation plan.
- **Ambiguity check**: "attention pooling" is specified concretely (single dot-product head, learnable temperature, softmax over prototype keys) to avoid two engineers building different mechanisms from this doc.
