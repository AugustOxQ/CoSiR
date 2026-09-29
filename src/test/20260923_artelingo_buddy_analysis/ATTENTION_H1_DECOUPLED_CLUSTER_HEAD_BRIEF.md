# Brief: Attention-h1 + decoupled Euclidean clustering head (DEC on its own latent, not the InfoNCE embedding)

Write a new script
`src/test/20260923_artelingo_buddy_analysis/run_attention_h1_decoupled_cluster_head_pilot.py`.
Do NOT run it — execution happens separately, on GPU, outside this task.

## Context: the diagnosed failure mode this pilot is designed to fix

Read, in full, before writing any code:
- `run_attention_h1_dec_hybrid_pilot.py` and its report — DEC's Student's-t
  kernel (`soft_assignments`/`target_distribution`, imported unchanged from
  `../20260922_percept_topic_pipeline/run_percept_stage1_pilot.py`) applied
  **directly on Attention-h1's own L2-normalized fused embedding**. Result:
  decisive failure, negative held-out silhouette in 4/4 stress seeds
  (mean ≈ -0.0288).
- `run_attention_h1_vmf_dec_hybrid_pilot.py` and its report — the diagnosed
  root cause of that failure: DEC's Student's-t kernel measures **squared
  Euclidean distance**, which assumes cluster centers can spread freely
  through space. Attention-h1's embedding is L2-normalized (forced onto a
  unit hypersphere by the model's own `F.normalize` calls), which starves
  the Euclidean kernel of real distance variation to exploit. Replacing the
  kernel with a cosine/vMF-style softmax (this pilot) fixed the sign of the
  silhouette (0.0271–0.0324 across 4 seeds, positive) but still
  underperformed simpler buddy variants on genre AMI, and did not clear the
  held-out Pareto bar in any of 4 seeds.

**This pilot tests the other fix to the same diagnosis.** Instead of
bending DEC's kernel to fit the sphere (the vMF attempt), give DEC its own
**separate, unconstrained Euclidean latent space** to cluster in, produced
by a small extra head on top of Attention-h1's shared trunk. The two
InfoNCE losses keep training the existing L2-normalized fused embedding
exactly as before (untouched geometry, untouched teacher-graph fidelity).
A new `cluster_head` MLP maps that fused embedding into a second,
**not-normalized** representation used only by DEC. This directly restores
the free Euclidean spread DEC's kernel needs, without changing the
InfoNCE-facing embedding's geometry at all — a genuinely different
mechanism from both prior attempts, not a third parameter tweak on the
same one.

## Implementation

Reuse `run_attention_h1_dec_hybrid_pilot.py`'s structure wholesale: its
sibling-import pattern (including the `sys.path` fix and the
`percept_stage1_base_pilot_for_attention_dec`-style import of
`../20260922_percept_topic_pipeline/run_percept_stage1_pilot.py` for
`soft_assignments`, `target_distribution`, and `prune_centers` — reuse
these three unmodified, do not reimplement), teacher graph construction,
`LearnedStudent("attn1")`, `sample_positive_pairs`,
`content_batch_embeddings`, `symmetric_infonce`, the cosine-annealed
optimizer (`LR_START=1e-3`, `LR_FLOOR=1e-5`, `T_max=arch.MAX_EPOCHS`), the
plateau-based recall stopping rule, and `evaluate_checkpoint`/AMI/
silhouette scoring. Do **not** modify the shared `LearnedStudent` class in
`run_learned_student_arch_sweep_pilot.py` — define the new head as a
standalone module in this new script only, exactly the way the DEC hybrid
siblings added a standalone `centers = nn.Parameter(...)` without touching
shared code.

**New: a decoupled clustering head.**

1. Define `ClusterHead(nn.Module)`: `nn.Sequential(nn.Linear(D_SHARED,
   D_SHARED), nn.ReLU(), nn.Linear(D_SHARED, D_SHARED))` (import `D_SHARED`
   from the `arch` sibling module rather than hardcoding 32). No
   normalization anywhere in this head — its output is a free Euclidean
   vector, unlike every embedding elsewhere in this project's buddy
   pilots.
2. At the start of each run, after constructing `model = LearnedStudent
   ("attn1")` and this run's fresh `cluster_head = ClusterHead().to(device)`,
   compute the clean (no-noise) train fused embedding, pass it through
   `cluster_head` to get the initial clustering latent, and fit
   `N_INITIAL_CLUSTERS = 100` K-means centers
   (`sklearn.cluster.KMeans(n_clusters=100, n_init=10, random_state=seed)`)
   on **that latent**, not on the fused embedding directly. Register the
   centers as a trainable `torch.nn.Parameter` alongside both `model`'s and
   `cluster_head`'s parameters in one optimizer (`list(model.parameters())
   + list(cluster_head.parameters()) + [centers]`), so one Adam instance
   and one cosine scheduler update everything.
3. Each training epoch, after computing the clean full-dataset fused
   embedding already used for the affect InfoNCE loss
   (`affect_embeddings, _ = model(train_content_t, train_affect_t)`), pass
   it through `cluster_head` to get `cluster_latent = cluster_head
   (affect_embeddings)` (this stays inside the autograd graph — do not
   detach). Compute `q = soft_assignments(cluster_latent, centers)`,
   `p = target_distribution(q)`, `dec_loss = F.kl_div(q.log(), p,
   reduction="batchmean")` — identical formula to the DEC hybrid sibling.
   Total loss: `content_loss + affect_loss + LAMBDA_DEC * dec_loss`, with
   the same linear warm-up from 0 over the first 30 epochs
   (`DEC_WARMUP_EPOCHS = 30`) as the DEC hybrid sibling, for the same
   reason (avoid premature self-sharpening collapse before the fused
   embedding has learned any structure).
4. At the same cadence as `CHECKPOINT_EVERY`, log `dec_loss`'s raw value,
   `content_recall`/`affect_recall` (existing diagnostics, unchanged), and
   the same all-100-center cluster-size collapse diagnostic
   (`min`/`max`/`median`/`below_one_percent` of hard assignments across all
   100 centers) the DEC hybrid sibling logs.
5. At the end of training, call `prune_centers(centers)` (the norm-based
   PercepT helper, reused unmodified — a plain vector norm is a meaningful
   proxy again here, because `cluster_head`'s output is genuinely
   unconstrained, unlike the sphere-constrained embedding that made
   norm-based pruning meaningless in the very first DEC hybrid attempt;
   say this explicitly in a comment) to keep the 67 highest-norm of the
   100 centers. Assign final **hard** cluster labels via
   `soft_assignments(cluster_head(embedding), surviving_centers)
   .argmax(dim=1)` for both train and held-out (compute held-out fused
   embeddings the normal clean way, pass through the same trained
   `cluster_head`, then assign via the pruned centers — no Leiden pass for
   this variant, same as the DEC hybrid sibling).
6. **Report silhouette in both spaces**, using the same hard labels for
   both: (a) in the `cluster_head` latent (the space DEC actually
   optimized — this is the fairest look at whether decoupling worked at
   all) and (b) in the original L2-normalized fused embedding (the space
   every other pilot in this project reports silhouette in, and the one
   that matters for comparing against the established reference table).
   Use the existing seed-42 two-stage sampled `silhouette_score`
   convention (draw ≤6,000 points with `np.random.default_rng(42)`, score
   with `sample_size=min(4000, len(idx))`, `random_state=42`) identically
   for both spaces. Compute held-out emotion/genre AMI once, from the hard
   labels (label space doesn't depend on which embedding space silhouette
   is measured in).

## Single-seed screen, then stress only the winner

**Seed 42 screen across `LAMBDA_DEC` in `(0.1, 0.5, 1.0)`** — same bracket
and same rationale as the DEC hybrid sibling (no reconstruction term here,
so PercepT's own implicit KL-weight-1 convention doesn't transfer
directly; a fresh modest bracket is the right starting point). Report, for
each value: held-out emotion AMI, held-out genre AMI, held-out silhouette
in **both** spaces (cluster-head latent and fused embedding), the held-out
Pareto bar (emotion AMI > 0.1236 AND genre AMI > 0.1954, defined on the
fused-embedding side as always), the collapse diagnostic trajectory, and
the `content_recall`/`affect_recall` trajectory.

**Winner selection:** same rule as every other sibling in this
investigation — among values clearing the held-out Pareto bar, pick the
highest held-out **fused-embedding** silhouette (that's the number
comparable to the reference table); if none clears, pick the highest
held-out fused-embedding silhouette among all three and label it
explicitly "best available, does not clear the Pareto bar."

**Stress test:** the winning `LAMBDA_DEC` at `SEEDS = (7, 123, 2024)`, same
methodology as every other 4-seed stress in this investigation (fresh
model, fresh `cluster_head`, fresh K-means center init with
`random_state=seed`, fresh `torch.manual_seed`/`np.random.seed`).

## Report

Write
`src/test/20260923_artelingo_buddy_analysis/attention_h1_decoupled_cluster_head_pilot_report.md`
with: the 3-value screen table (both splits, both silhouette spaces, all
AMI metrics, collapse diagnostics, recall trajectory), the
winner-selection rationale, the four-seed stress table and summary
statistics (report both silhouette spaces throughout, but the headline
comparisons and Pareto-bar checks use the fused-embedding space, for
consistency with every earlier report), and an explicit comparison table
against ALL SEVEN established reference points already used in this
investigation's other buddy reports: (a) Attention-h1 baseline
(0.1249/0.2404, silhouette 0.0392), (b) Leiden pseudo-contrastive alone
(0.1165/0.2351, silhouette 0.0877), (c) noise+schedule alone (0.1306/0.1973
seed 42; four-seed mean 0.1277/0.2396, silhouette 0.0466), (d) the
noise-schedule+pseudo-contrastive combination (four-seed mean
0.1160/0.3011, silhouette 0.0739, 0/4 cleared the bar), (e) Euclidean DEC
hybrid (four-seed mean 0.1160/0.1321, silhouette -0.0288), (f) vMF DEC
hybrid (report its own four-seed numbers — read them from
`attention_h1_vmf_dec_hybrid_pilot_report.md` rather than guessing), (g)
PercepT-replication faithful recipe (0.1092/0.3288, silhouette 0.5120).
State plainly, in a clearly headed final section: does decoupling the
clustering latent from the InfoNCE embedding close a meaningful fraction
of the silhouette gap to the faithful-recipe PercepT replication (use the
same declared 10%-of-gap marker the vMF sibling's report used, for
comparability), does it do so while clearing the held-out AMI Pareto bar,
and is that result robust across 4 seeds or only a single-seed screen
artifact. Also state explicitly whether the cluster-head-space silhouette
and the fused-embedding-space silhouette move together or diverge — that
comparison is the direct evidence for or against the decoupling
hypothesis itself, independent of whether this particular pilot's numbers
happen to clear the Pareto bar. This investigation's established
convention is a blunt, numeric verdict — if this collapses or
underperforms every buddy variant tried so far, report that plainly,
exactly as prior negative results in this project have been reported.

Do not touch git, do not modify any other file in the repository.
