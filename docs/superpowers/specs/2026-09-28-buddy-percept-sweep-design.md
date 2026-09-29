# Buddy-graph (Stage 1 + Stage 2) comprehensive hyperparameter sweep — design spec

Generated 2026-09-28. Branch `experiment/percept_topic_pipeline`, worktree
`/project/CoSiR-buddy_prototype_conditioning`.

## 1. Goal

Find the best-performing buddy-graph configuration end-to-end (Stage 1
topic formation → Stage 2 mapper), via a single joint Weights & Biases
Bayesian sweep, distributed across 9 reserved DAS6 GPUs (3 nodes × 3 GPUs
each), running up to a 24-hour soft budget (extendable if not yet
converged).

This is the direct continuation of tonight's overnight investigation
(`docs/reports/2026-09-26_artelingo_buddy_vs_percept_stage1_report.md`),
which hand-tuned a handful of hyperparameters one at a time (candidates
1-6). This sweep replaces that manual process with a systematic search
over the full space at once, now that the per-trial cost is known to be
cheap (~35-90s once warmed up, measured directly from tonight's own pilot
run logs).

## 2. Scope and non-goals

**In scope:** the buddy-graph topic-formation pipeline used throughout
tonight's investigation — `LearnedStudent` (Stage 1, InfoNCE two-teacher
student) → Leiden clustering (post-hoc, on the trained embedding) → Stage
2 (`AttentionPoolingMapper`, patch-image classifier over the frozen topic
vocabulary).

**Explicitly out of scope:** CoSiR's own trainable-label-embedding
retrieval framework (`src/hook/train_cosir.py`, `scripts/run_sweep_agent.py`,
`scripts/sweep_config_v*.yaml`). That is a *different* system that
happens to also use the word "buddy" (a buddy-graph *regularizer* for
retrieval training) — this sweep does not touch it, reuse its sweep
agent, or share its W&B project. This mirrors the standing constraint
from earlier in this investigation.

## 3. Trial architecture

One W&B trial = one Python process invocation that runs the **entire
pipeline end-to-end**, for one sampled hyperparameter configuration, at a
fixed seed (see §6):

1. Load cached fixed inputs (see §3.1) — paid once per agent *process*,
   not per trial.
2. Build Stage 1 teacher graphs (content, affect) — `teacher_graph_K`/
   `teacher_graph_alpha` if swept (extra), else fixed at the pipeline's
   current defaults (`K=20, ALPHA=0.5`, `run_pipeline.py`).
3. Train `LearnedStudent` (`heads`, `lr`, `noise_std`, and any active
   extras) to plateau or `MAX_EPOCHS=200`, logging a per-checkpoint proxy
   metric every 5 epochs for hyperband.
4. Re-cluster the frozen trained embedding's own mutual-kNN graph via
   Leiden at the sampled `leiden_resolution` (reusing
   `leidenalg.RBConfigurationVertexPartition`, the exact pattern
   validated in `run_candidate3_k_sweep_pilot.py`).
5. Merge small train communities below `merge_small_threshold` (reusing
   `merge_small_communities` from `run_candidate1_min_occupancy_pilot.py`).
6. Assign held-out points onto the frozen train vocabulary via k-NN
   transfer (`assign_to_train_communities`,
   `run_heldout_label_transfer_pilot.py`, `k=transfer_k`).
7. Build Stage 2 targets per `target_cutoff` (single-label one-hot, or
   `cosine_vote_fractions` + relative threshold, reusing
   `run_candidate4_rich_multilabel_pilot.py`'s helpers), with or without
   class-balanced loss weighting.
8. Train `AttentionPoolingMapper` (`mapper_lr`, `mapper_epochs`, and any
   active capacity extras) on cached patch features.
9. Score: held-out Stage 1 emotion/genre AMI, held-out Stage 2 macro AUC
   (scored against **single-label** held-out targets always, regardless
   of what the model was trained on — this is the one hard-learned rule
   from tonight, §6f/§6b of the master report: never let a richer
   training-target convention leak into the evaluation-target
   convention).
10. Report to W&B: `objective` (see §5), plus every raw metric, plus
    wall-clock breakdown (stage1_seconds, stage2_seconds) for later
    per-trial cost analysis.

### 3.1 Cached, hyperparameter-independent inputs (loaded once per process)

- Deduplicated train/held-out CLIP features, majority emotion/genre
  labels (`pipeline.load_dedup_features`).
- GoEmotions affect embeddings for both splits (~5m40s measured — the
  dominant one-time cost).
- Train-only content PCA fit (only re-fit if `content_pca_dim` is an
  active swept extra — see §7 risk).
- Patch feature tensors (`[N, 50, 512]`) for both splits.

## 4. Reused building blocks (exact files, no reimplementation)

| piece | source |
|---|---|
| `LearnedStudent`, `AttentionFusion`, teacher-graph construction, InfoNCE, checkpoint eval | `src/test/20260923_artelingo_buddy_analysis/run_learned_student_arch_sweep_pilot.py` |
| noise/cosine-schedule training loop pattern | `run_attention_h1_noise_schedule_pilot.py` |
| Leiden re-clustering at arbitrary resolution | `run_candidate3_k_sweep_pilot.py::leiden_partition`, `src/conditional_buddy/buddy_graph.py::mutual_knn` |
| small-community merging | `run_candidate1_min_occupancy_pilot.py::merge_small_communities` |
| held-out k-NN label transfer | `run_heldout_label_transfer_pilot.py::assign_to_train_communities` |
| rich multi-label targets | `run_candidate4_rich_multilabel_pilot.py::cosine_vote_fractions`, `threshold_targets` |
| `AttentionPoolingMapper`, patch feature loading, `evaluate_auc`/`auc_summary` | `run_buddy_stage2_pilot.py` |
| class-balanced loss weighting | `run_candidate1_min_occupancy_pilot.py` (inverse-frequency pattern) |

The new sweep script imports these as sibling modules (the
`load_module`/`importlib.util` pattern used throughout this
investigation), rather than copy-pasting.

## 5. Search space

Two YAML-driven parameter groups. **Core is always swept. Extra is
toggle-able** via a single `SWEEP_EXTRAS=1/0` environment variable read
by the sweep script at agent-process start — this lets the same sweep
config file serve both "core-only" and "core+extra" without a second
file, per the request that extras be droppable independently.

### 5.1 Core — Stage 1
| param | range |
|---|---|
| `heads` | {mlp128, attn1, attn4} |
| `lr` | log-uniform [3e-4, 3e-3] |
| `noise_std` | {0.0, 0.02, 0.05, 0.1} |
| `leiden_resolution` | log-uniform [0.05, 2.0] |

### 5.2 Core — Stage 2
| param | range |
|---|---|
| `mapper_lr` | log-uniform [3e-4, 3e-2] |
| `mapper_epochs` | {100, 200, 400, 800} |
| `merge_small_threshold` | {0.0, 0.01, 0.02} |
| `class_balanced_loss` | {true, false} |
| `target_cutoff` | {"single_label", 0.5, 0.3, 0.15} |

### 5.3 Extra — Stage 1
| param | range |
|---|---|
| `num_heads` | {1, 2, 4, 8, 16} — ignored (no-op) when `heads=mlp128`, since that path has no attention module; generalizes attn1/attn4 when `heads` is attention-based |
| `d_shared` | {16, 32, 64, 128} |
| `content_pca_dim` | {30, 50, 80, 120} |
| `lambda_affect` | log-uniform [0.5, 2.0] |
| `teacher_graph_K` | {10, 15, 20, 30} |
| `teacher_graph_alpha` | {0.3, 0.5, 0.7} |
| `batch_size` | {512, 1024, 2048, 4096} |
| `weight_decay` (Stage 1, AdamW) | {0.0, 1e-5, 1e-4} |

### 5.4 Extra — Stage 2
| param | range |
|---|---|
| `num_queries` | {1, 2, 4, 8} |
| `mlp_head` | {"linear", "one_hidden"} |
| `mapper_epochs` (extends the core param, not a separate one) | adds 1600 to the core {100, 200, 400, 800} set when extras are on |
| `transfer_k` | {10, 20, 30, 40} |
| `weight_decay` (Stage 2, AdamW) | {0.0, 1e-5, 1e-4} |

22 dimensions total with extras on, 9 without.

## 6. Seeding strategy

Every sweep trial trains at a **fixed `seed=42`** — matching tonight's
own "screen at seed 42" convention, and necessary at this scale: seed
noise stacked on top of 22 hyperparameters would make the response
surface much harder for Bayesian search to model, and multi-seed
averaging per trial would divide the effective trial budget by the
number of seeds averaged.

After the sweep (soft-stopped at 24h or later), **re-run the top ~10
configurations by `objective` through the established 4-seed stress test
(42, 7, 123, 2024)** before declaring a final winner — this is a
separate, small follow-up script, not part of the sweep itself.

## 7. Objective and gating

Single scalar `objective`, `goal: maximize`:

```
objective = stage2_macro_auc if (emotion_ami > 0.1236 and genre_ami > 0.1954) else -1.0
```

(thresholds are this investigation's standing Pareto bar). Gate-failing
trials always rank below every valid trial, so bayes naturally avoids
that region once it has enough data, without needing constrained
optimization support from W&B (which the classic `bayes` method doesn't
have natively).

Raw `stage1_emotion_ami`, `stage1_genre_ami`, `stage1_fused_silhouette`,
`stage2_macro_auc`, `n_topics_after_merge` are logged separately for
post-hoc analysis (Pareto front, correlation plots) regardless of the
gate.

**Hyperband intermediate metric:** log `checkpoint_proxy` = mean of
held-out content_recall and affect_recall, every 5 epochs during Stage 1
training (already computed at each checkpoint in the reused training
loop) — lets `early_terminate: hyperband` kill clearly-bad Stage 1
trajectories before they reach Stage 2 at all.

```yaml
method: bayes
metric:
  name: objective
  goal: maximize
early_terminate:
  type: hyperband
  min_iter: 3
  eta: 2
```

## 8. Execution and dispatch

1. **Local smoke test first** (no cluster): run the sweep script directly
   (not via `wandb agent`) for 2-3 hardcoded configs on the local GPU,
   confirm the full pipeline completes and reports sane, non-degenerate
   metrics. Then `wandb sweep <config>.yaml --count 3` locally to confirm
   the W&B-driven path also works end-to-end before touching DAS6.
2. Create the real sweep once: `wandb sweep scripts/sweep_config_buddy_percept.yaml`
   → note the printed `sweep_id`.
3. Dispatch 9 agents via `cluster-run`, 3 per node, one `cluster launch
   --gpu-slots i -- <agent-launch-command>` per GPU (the documented
   pattern for multi-GPU nodes: "launch one job per GPU with
   `--gpu-slots`").
4. Each agent process runs `wandb agent <sweep_id>` bound to
   `scripts/run_buddy_percept_sweep_agent.py`, which does the one-time
   cache load (§3.1) then loops trials.
5. Monitor via `cluster watch <tag>` per job (9 tags total) plus the W&B
   sweep dashboard directly (progress, objective distribution, parameter
   importance — W&B computes this natively for `bayes` sweeps).
6. 24h soft budget: check in at that point; if `objective` is still
   improving, extend by simply not killing the agents (they keep pulling
   trials from the same queue) rather than redesigning anything.

## 9. New files

All under the worktree `/project/CoSiR-buddy_prototype_conditioning`:

- `scripts/sweep_config_buddy_percept.yaml` — the W&B sweep definition (§5, §7).
- `scripts/run_buddy_percept_sweep_agent.py` — the trial script (§3), the
  `program:` target for the sweep config.
- `src/test/20260928_buddy_percept_sweep/` — smoke-test scratch (local
  runs, not the sweep itself), plus the post-sweep top-10 4-seed stress
  script once the sweep has results.

## 10. Risks and mitigations

- **`content_pca_dim` as an extra invalidates the "fit PCA once" cache.**
  Mitigation: when `content_pca_dim` is active (extras on), re-fit PCA
  per-trial (cheap — it's fit on top of already-cached raw content
  features, not re-extracting CLIP features) rather than caching it
  process-wide.
- **W&B API volume at ~8,000-17,000 total trials.** Mitigation: log only
  checkpoint-level summaries (every 5 epochs) during training, not every
  step; this is well within W&B's normal sweep scale (documented sweeps
  routinely run into the tens of thousands of runs).
- **9 independent agent processes each pay the ~5m40s one-time warmup.**
  Accepted cost (~9 GPU-minutes total, negligible against a 24h budget).
- **DAS6 data availability.** Confirm the ArtELingo CLIP feature /
  patch-feature caches this pipeline depends on
  (`/data/SSD2/pre_extract/...` per tonight's PercepT scripts, and
  whatever path buddy's own pilots resolve to) are reachable from the
  reserved nodes, or covered by `cluster-run`'s `DATA_MAP` /
  data-sync convention, **before** dispatching — check this during the
  smoke-test step, not after launching all 9 agents.
- **Hyperband + non-monotonic checkpoint proxy risk killing good runs
  early** if `checkpoint_proxy` is noisy at low epoch counts. Mitigation:
  `min_iter: 3` (don't judge before at least 3 checkpoints = 15 epochs).

## 11. Deliverables checklist

- [ ] `scripts/run_buddy_percept_sweep_agent.py`
- [ ] `scripts/sweep_config_buddy_percept.yaml` (core + extras via env flag)
- [ ] Local smoke test passing (hardcoded configs, then a 3-run local `wandb agent`)
- [ ] Sweep created, 9 agents dispatched via `cluster-run`
- [ ] Post-sweep: top-10-by-objective 4-seed stress script
- [ ] Report folded into the master investigation report once a winner is confirmed
