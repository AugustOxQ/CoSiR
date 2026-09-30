# Matched PercepT-vs-buddy head-to-head (H2H) — design

Date: 2026-09-30. Branch `experiment/percept_topic_pipeline`
(worktree `/project/CoSiR-buddy_prototype_conditioning`).
Follows master report §6i
(`docs/reports/2026-09-26_artelingo_buddy_vs_percept_stage1_report.md`).

## 1. Question

§6g's Stage 2 headline is "symmetrically tuned PercepT 0.9226 vs buddy
0.8534". §6i's sweep found a buddy config at 0.9355, but only inside a
re-implemented harness whose Stage 1 differs from the pilots, at a different
topic count (K), with different held-out labels, and with ~2,267 trials of
buddy-only tuning. None of these numbers can be compared directly.

This project answers: **with both systems run through one validated harness,
at matched K, with the same held-out labelling and evaluation, and the same
tuning budget, which system gives better Stage 2 macro AUC — and how do their
Stage 1 AMIs compare on both yardsticks?**

It also closes the two §6i confirmation checks (QC1, QC2, §3).

## 2. Decisions

User-approved (2026-09-30):
- **Objective:** held-out Stage 2 macro AUC, no Stage 1 gate. Stage 1 AMIs
  are reported on both yardsticks for every selected config.
- **Matched K:** two levels, K≈16 and K≈40. Four cells: {buddy, PercepT} × {16, 40}.
- **Split:** the 9,365 held-out paintings are split 50/50 into *val* and
  *test*. All tuning and selection use val; final numbers use test only,
  at fresh seeds.
- **Budget:** equal trial count per cell, Bayesian W&B sweeps, each trial
  scored on the mean of 2 seeds; then 4-seed stress of each cell's top 5 on
  val; the cell winner goes to test.

Controller rulings (each overridable; cost if wrong in brackets):
- **R1 Native inputs.** Each system keeps its own published inputs: buddy
  uses PCA'd CLIP content + 28-d GoEmotions probabilities; PercepT uses its
  fused 2:1 content:affect vector with the 768-d GoEmotions embedding. The
  report states this as a system difference. [A reader who wants
  input-matched systems gets a system-level answer instead.]
- **R2 Held-out labels.** The primary held-out topic labels (Stage 2
  evaluation targets and transfer AMI) come from a k-NN vote into the
  system's own train topics in its own Stage 1 embedding space, with
  **k = 20 fixed, never tuned** (tuning it tunes the evaluation labels,
  §6i). PercepT's native DEC assignment is computed too and reported as a
  secondary AUC/AMI. [If native labels change the ranking, the report must
  say so; both are logged for every trial.]
- **R3 K control.** PercepT sets K directly (pruned survivors = target).
  Buddy hits the target by bisection over Leiden resolution on the trained
  embedding (after small-topic merging), accepting K within ±2; if bisection
  fails in 12 steps the trial returns objective −1 and is logged as a
  K miss. Leiden resolution therefore leaves buddy's search space. [Buddy
  loses the freedom to pick K inside the band; mitigated by ±2.]
- **R4 Budget sizing.** 9 GPUs (3 nodes × 3), not the 27 stated in the
  question. Trials per cell = the largest multiple of 50 (max 300, min 150)
  that the measured per-trial times let all four cells finish in ≤ 60 wall
  hours. Same number for every cell. [Fewer trials → noisier optima,
  equally for both systems.]
- **R5 Stage 2 space is identical for both systems**; Stage 1 spaces are
  system-specific (§6).
- **R7 Buddy Stage 1 implementations.** The four structural knobs of §4
  (`teacher_graph`, `infonce_negatives`, `stage1_optimizer`,
  `stage1_stopping`) are realised as one categorical `buddy_impl`
  {pilot, harness}: `pilot` is a faithful port of the snapshot pilot
  (union teacher graph + repair, in-batch InfoNCE, Adam, plateau stopping,
  PCA seeded by the run seed), `harness` is the §6i Stage 1. Mixed
  combinations are not searched. [A hybrid could beat both; not tested.]
- **R6 Seeds.** Search trials use seeds (1001, 1002); stress uses
  (42, 7, 123, 2024); test uses fresh (11, 23, 57, 101, 211). No seed is
  reused across these roles.

## 3. Confirmation checks (QC, Phase 0)

- **QC1:** `m8x7ifx4` (§6i winner) at fresh seeds (11, 23, 57, 101) with
  `transfer_k` 20 and 40, through the existing harness, also scoring the
  pilots' *independent* held-out re-clustering AMI.
- **QC2:** the pilots' unchanged Attention-h1 snapshot pilot at seeds
  42/7/123/2024, snapshots kept, scored with (a) independent AMI
  (must reproduce the §3 table), (b) transfer AMI k∈{10,20,30,40},
  (c) the sweep gate's exact measurement (merge 0.02 → transfer k 20/40).
- Output: one table putting both on both yardsticks; written into the
  master report as §6j. QC2's per-seed snapshots are also the reference
  data for validation V2.

## 4. Harness v2 architecture

New modules under `scripts/buddy_percept_sweep/` (existing modules keep
their behaviour; the §6i sweep stays reproducible):

| module | purpose |
|---|---|
| `inputs_store.py` | Builds once and loads from disk (npz under `$H2H_CACHE_DIR`, default `<feature_root>/artelingo_h2h_cache/`) every hyperparameter-independent input for both systems: buddy content raw + 28-d affect, pilot img/txt nodes (for the union teacher graph), PercepT fused inputs (768-d affect), emotion/genre labels, patch features. File lock + atomic rename so concurrent agents build it once per node. Kills the ~40% GoEmotions re-extraction waste. |
| `splits.py` | Deterministic val/test split of held-out painting indices (seed 0, stratified by majority emotion × has-genre). Saved alongside the cache; its hash is logged with every trial. |
| `stage1.py` (extended) | Buddy Stage 1 knobs, defaults = current harness behaviour: `teacher_graph` {concat_mknn, pilot_union}, `infonce_negatives` {full, in_batch}, `stage1_optimizer` {adamw_cosine, adam_constant}, `stage1_stopping` {fixed, plateau}. `pilot_union` calls the pilots' own `pipeline.build_buddy_graphs` / `build_single_modality_graph`; `plateau` reproduces the pilot's rule (checkpoint every 5 epochs, content+affect recall on held-out reference graphs built from the **val** subset only, window 5, rel. 0.01). |
| `clustering.py` (extended) | `leiden_graph` {mknn, pilot_repaired}; `pilot_repaired` + resolution 1.0 reproduces the pilots' `detect_communities`. `target_k_partition(...)` = R3 bisection. |
| `stage1_percept.py` | PercepT Stage 1 by calling the fixed pilot's functions (`run_percept_stage2_fixed_pilot.py` flow as reused by `run_percept_mapper_symmetric_sweep_pilot.fit_stage1_and_get_targets`), with knobs applied by setting the pilot modules' constants before the call (pretrain epochs/lr, DEC lr, λ_balance, λ_recon, N_initial, stability threshold). Returns train/held-out latent embeddings, train labels, native held-out labels. Frozen pilot files are imported, never edited. |
| `pilot_metrics.py` (from QC) | Independent re-clustering AMI with the pilots' own graph + Leiden code. |
| `evaluate.py` | Given a trial's embeddings/labels and a held-out subset (val or test): transfer labels (k=20), both AMI yardsticks, K, Stage 2 macro AUC with primary (transfer) and secondary (native, PercepT only) targets, skipped-topic count. |
| `h2h_trial.py` | `run_h2h_trial(config, store, subset, seeds) -> dict`: Stage 1 (system-dispatched) → K control → Stage 2 (shared mapper code) → `evaluate` per seed; returns per-seed and mean metrics. |
| `h2h_agent.py` | In-process W&B agent (`wandb.agent(sweep_id, function=...)`), one store per process, logs all metrics; objective = mean val AUC (primary labels) over the 2 search seeds. |
| `h2h_select.py` | Per cell: pull top-5 from W&B, freeze to JSON, 4-seed val stress, pick winner (max mean val AUC), then run winner on test at 5 fresh seeds. Line-oriented `H2H_RESULT {json}` output like `run_top10_stress.py`. |

Stage 2 is the existing harness mapper (`ParameterizedAttentionPoolingMapper`,
`train_stage2`), used identically for both systems.

## 5. Validation gates (Phase 1) — must pass before any sweep

- **V1 Stage 2 equivalence (snapshot-driven).** Feeding the pilots' saved
  topics through harness Stage 2 reproduces:
  (a) buddy §6f 0.8534 from `attention_h1_embedding_snapshot.npz` (Leiden
  K=19 → merge 3 smallest → K=16, transfer k=20, multi-label cutoff 0.15,
  class-balanced, mapper lr 1e-2, 400 epochs, 4 mapper seeds), and
  (b) PercepT from `percept_fixed_snapshot.npz` (K=40, native targets):
  its recorded 0.5925 at the untuned mapper (lr 1e-3, 100 epochs). The
  snapshot is the §6b fixed run, not §6g's refit, so 0.9226 is checked in
  V3 instead; V1 also reports lr 1e-2 / 400 epochs on the snapshot as
  information. Before any AUC comparison, the
  harness-built train and held-out target matrices must equal the pilot's
  own target matrices exactly (computed by importing the §6f / §6g pilot
  functions), so a mismatch is pinned to targets or to the mapper, not both.
  AUC tolerance |Δ mean AUC| ≤ 0.003. Validates mapper, targets, merge and
  AUC code against the published numbers on identical Stage 1 outputs.
- **V2 Buddy faithful Stage 1.** Harness with all pilot knobs on
  (`pilot_union`, `in_batch`, `adam_constant`, `plateau`, `pilot_repaired`
  at resolution 1.0, attn1/1 head/d_shared 32/pca 50/batch 1024/lr 1e-3) at
  seeds 42/7/123/2024 vs QC2's pilot snapshots on the same node type:
  per-seed |Δ independent emotion AMI| ≤ 0.005 and |Δ genre AMI| ≤ 0.015
  (genre rests on 159 paintings), train K within ±1. Plateau monitor uses
  the full held-out set for this check only (as the pilot did).
- **V3 PercepT Stage 1.** Harness PercepT at the fixed pilot's settings,
  seed 42: K=40, held-out native AMIs within 0.01 of the snapshot
  (0.1094 / 0.2798), and V1(b)'s Stage 2 path on the refit within 0.02 of
  0.9226 (§6g showed PercepT's refit is not bit-reproducible).
- A failed gate is debugged (systematic-debugging) and fixed before any
  sweep; if after two fix attempts it still fails, the controller records a
  ruling with the measured gap and proceeds only if the gap cannot change a
  head-to-head conclusion larger than the gap itself; the report states it.

## 6. Search spaces

Shared Stage 2 (both systems): `mapper_lr` log-uniform [1e-3, 3e-2];
`mapper_epochs` {100, 200, 400, 800}; `num_queries` {1, 2, 4, 8};
`mlp_head` {linear, one_hidden}; `weight_decay_stage2` {0, 1e-5, 1e-4};
`class_balanced_loss` {false, true}; `target_cutoff` {single_label, 0.15,
0.3, 0.5}; `train_target_k` {10, 20, 40} (k-NN vote fractions for
multi-label *training* targets only).

Buddy Stage 1: `heads` {mlp128, attn}; `num_heads` {1, 2, 4, 8};
`d_shared` {16, 32, 64, 128}; `lr` log-uniform [1e-4, 1e-2]; `noise_std`
{0, 0.05, 0.1, 0.2}; `lambda_affect` log-uniform [0.25, 4];
`batch_size` {512, 1024, 2048, 4096}; `weight_decay` {0, 1e-5, 1e-4};
`content_pca_dim` {30, 50, 80, 120}; `teacher_graph_K` {10, 15, 20, 30};
`merge_small_threshold` {0, 0.005, 0.01, 0.02}; the four structural knobs of
§4 (`teacher_graph`, `infonce_negatives`, `stage1_optimizer`,
`stage1_stopping`); `leiden_graph` {mknn, pilot_repaired}.

PercepT Stage 1: `pretrain_epochs` {50, 100, 200}; `pretrain_lr`
log-uniform [3e-4, 3e-3]; `dec_lr` log-uniform [3e-5, 1e-3];
`lambda_balance` log-uniform [10, 1e4]; `lambda_reconstruction`
log-uniform [0.1, 10]; `n_initial_factor` {1.0, 1.5, 2.0, 2.5}
(N_initial = round(factor × K)); `stability_threshold` {1e-3, 3e-3};
`latent_dim` if the pilot's autoencoder exposes it, else fixed.

## 7. Protocol

1. Phase 0 QC and Phase 1 validation (§3, §5).
2. Measure per-trial time for each cell; fix trials-per-cell by R4.
3. Four W&B sweeps (project `polysemic/CoSiR-h2h`), one per cell, `bayes`,
   no hyperband. Agents run in-process with the disk store; 9 agents,
   cells interleaved so all four progress together (e.g. round-robin agents
   across sweeps, rebalanced when a cell completes).
4. Stop each sweep at its trial count.
5. Selection (`h2h_select.py`): per cell top-5 by 2-seed val mean → 4-seed
   val stress → winner = max mean val AUC (primary labels).
6. Test: each cell winner at 5 fresh seeds on the test half; report mean ±
   std AUC (primary and secondary labels), both AMI yardsticks on test, K,
   skipped topics, GPU-hours per cell.
7. Also on test: §6i winner `m8x7ifx4` and the §6g PercepT config as fixed
   reference points (no tuning), so the new numbers connect to the old ones.

## 8. Reporting

- Master report: §6j (QC results) and §6k (H2H result, short, pointing at
  the full report).
- Full report: `docs/reports/auto/percept/2026-10-0X_matched_percept_buddy_h2h.md`
  on this branch (the user's report layout; promotion to main via
  `scripts/promote_reports.py` later, not now).
- Every claim cites the raw logs (`H2H_RESULT` lines) committed under
  `src/test/20260930_matched_h2h/`.
- Headline table: rows = 4 cells, columns = test AUC (primary),
  test AUC (secondary), independent AMI emo/genre, transfer AMI
  emo/genre, K, trials, GPU-h. The verdict per K level is "system A better
  by Δ (95% CI over test seeds)" or "no difference beyond seed noise".

## 9. Risks

- PercepT's refit nondeterminism (§6g) widens its seed variance; handled
  by multi-seed search and stress, and reported.
- K targeting for buddy may fail for some Stage 1 configs (degenerate
  embeddings); those trials return −1 and are counted.
- Genre AMI on ~80 labelled paintings per half is noisy; emotion AMI is
  the primary Stage 1 signal.
- DAS6 reservation ends in 5 days; sweeps are sized (R4) to leave ≥ 1 day
  for selection, test and reporting.

## 10. Constraints

Never edit `src/test/20260922_percept_topic_pipeline/`,
`src/test/20260923_artelingo_buddy_analysis/`,
`src/test/20260927_deep_stage_analysis/` (import/load only), nor
`src/hook/train_cosir.py`, `scripts/run_sweep_agent.py`,
`scripts/sweep_config_v*.yaml`. Cluster work only through the cluster-run
CLI, under `/local/wding/`. Implementation by Claude Code subagents sized
to difficulty (no Codex). New commits only; stage by explicit path. Final
whole-branch review before the work is called done.
