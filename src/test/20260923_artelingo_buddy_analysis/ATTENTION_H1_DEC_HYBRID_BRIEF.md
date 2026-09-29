# Brief: Attention-h1 + DEC's actual clustering loss (buddy-encoder / DEC-loss hybrid)

Write a new script
`src/test/20260923_artelingo_buddy_analysis/run_attention_h1_dec_hybrid_pilot.py`.
Do NOT run it — execution happens separately, on GPU, outside this task.

## Context: why this is a different idea from everything tried so far

Read, in full, before writing any code:
- `run_attention_h1_noise_schedule_pilot.py` and its report (cosine LR
  schedule alone was the only winner there; noise did not help).
- `run_attention_h1_leiden_pseudo_contrastive_pilot.py` and its report (a
  same-community InfoNCE loss from periodic Leiden re-clustering raised
  held-out silhouette 0.0392→0.0877 but slightly hurt AMI).
- `run_attention_h1_noise_schedule_pseudo_contrastive_pilot.py` and its
  report (combining the above two: silhouette 0.0789, held-out emotion AMI
  missed the Pareto bar by 0.0026).
- `run_percept_stage1_faithful_recipe_pilot.py` (in
  `../20260922_percept_topic_pipeline/`) — specifically its
  `soft_assignments`, `target_distribution`, `prune_centers`, and DEC
  training-loop functions in its imported `base` module
  (`run_percept_stage1_pilot.py`, same directory). This PercepT-replication
  pilot's DEC mechanism reached held-out silhouette 0.5120 — by far the
  highest measured anywhere in this project, roughly 6-10x every buddy/
  attention result above.

**None of the three buddy pilots above ever used DEC's actual mechanism.**
The Leiden pseudo-contrastive loss is a **discrete, one-shot, binary**
same/different-community signal, recomputed by re-running Leiden every 20
epochs. DEC's own loss is fundamentally different: **continuous** soft
cluster assignment via a Student's-t kernel against **learnable, jointly-
trained cluster centers**, sharpened into a self-reinforcing target
distribution every step, with no discrete re-clustering at all. This is the
literal mechanism that gives PercepT its huge measured silhouette gap over
every buddy variant tried so far, and it has never been tried on Attention-
h1's own embedding. This pilot tests it directly: keep Attention-h1's own
architecture and its two teacher-graph InfoNCE losses (content, affect) as
the embedding-formation mechanism — this is "the buddy" contribution — and
add DEC's own clustering loss on top, reusing the PercepT replication's own
functions rather than reimplementing a close variant.

## Implementation

Reuse `run_attention_h1_noise_schedule_pilot.py`'s structure wholesale: its
sibling-import pattern (including the `sys.path` fix), teacher graph
construction, `LearnedStudent("attn1")`, `sample_positive_pairs`,
`content_batch_embeddings`, `symmetric_infonce`, the cosine-annealed
optimizer (`LR_START=1e-3`, `LR_FLOOR=1e-5`, `T_max=arch.MAX_EPOCHS`, the
epoch-ceiling guard — reuse this unconditionally; the noise-schedule
sibling's own winner was schedule-only, `noise_std=0`, so do **not** add
embedding noise here, keep it out entirely rather than wiring in a
permanently-disabled parameter), the plateau-based recall stopping rule,
and `evaluate_checkpoint`/AMI/silhouette scoring.

**New: import PercepT's own DEC machinery as a sibling module** from
`../20260922_percept_topic_pipeline/run_percept_stage1_pilot.py` via the
same `load_module` pattern already used throughout this directory. Reuse,
unmodified: `soft_assignments(latent, centers)`, `target_distribution(q)`,
and `prune_centers(centers)`. Do not reimplement these — call them exactly
as PercepT's own script defines them, so the DEC mechanism is provably
identical to the one that produced silhouette 0.5120, not a lookalike.

**New training loss term, added to the existing two-teacher InfoNCE total:**

1. At the start of each run (after the embedding is randomly initialized,
   before epoch 1), fit `N_INITIAL_CLUSTERS = 100` K-means centers
   (`sklearn.cluster.KMeans(n_clusters=100, n_init=10, random_state=seed)`)
   on the clean (no-noise) train embedding, exactly mirroring how the
   PercepT pilot initializes its own centers, and register them as a
   trainable `torch.nn.Parameter` alongside the `LearnedStudent` model's
   own parameters (so the same optimizer/scheduler updates both).
2. Each training epoch, after computing the clean embedding used for the
   two existing InfoNCE losses, also compute `q = soft_assignments(embedding,
   centers)`, `p = target_distribution(q)`, and
   `dec_loss = F.kl_div(q.log(), p, reduction="batchmean")` — the identical
   formula PercepT's own DEC pilot uses. The DEC loss operates on the
   **clean, un-noised full-dataset embedding** (all 61,402 train nodes),
   which the code already computes once per epoch for the affect InfoNCE
   loss (`affect_embeddings, _ = model(train_content_t, train_affect_t)`) —
   reuse that same tensor, do not recompute the forward pass a third time.
3. Total loss becomes
   `content_loss + affect_loss + LAMBDA_DEC * dec_loss`. Introduce
   `LAMBDA_DEC` as a swept constant (see screen below) with a linear
   warm-up from 0 to its screened value over the first 30 epochs (mirror
   the pseudo-contrastive sibling's warm-up idea, since DEC's self-
   sharpening target can degenerate if it dominates before the embedding
   has learned any structure at all — state this reasoning in a comment).
4. At the same cadence as `CHECKPOINT_EVERY`, log `dec_loss`'s raw value,
   `content_recall`/`affect_recall` (existing diagnostics, unchanged), and
   the same all-100-center cluster-size collapse diagnostic PercepT's own
   pilot logs (`min`/`max`/`median`/`below_one_percent` of hard assignments
   across all 100 centers) — this combination could plausibly collapse the
   same way an earlier PercepT-side pilot's own balance-term combination
   did, and that collapse was only caught because it was logged this way.
5. At the end of training (same point as Leiden/AMI/silhouette are
   evaluated in the sibling pilots), call `prune_centers(centers)` to keep
   the 67 highest-norm of the 100 centers (identical rule to PercepT), then
   assign final **hard** cluster labels via `soft_assignments(embedding,
   surviving_centers).argmax(dim=1)` for both train and held-out (compute
   held-out embeddings the normal clean way, then assign via the same
   pruned centers — no separate Leiden pass is needed or wanted for this
   variant, since DEC's own centers ARE the topic definition; do not also
   run Leiden on top of this). Compute held-out emotion/genre AMI and
   silhouette from these DEC-derived hard labels, using the same
   `external_metrics`/`silhouette_score` conventions as every other pilot
   in this directory.

## Single-seed screen, then stress only the winner

**Seed 42 screen across `LAMBDA_DEC` in `(0.1, 0.5, 1.0)`** — bracket
chosen because PercepT's own joint loss is `L_KL + LAMBDA_RECONSTRUCTION *
L_R` with `LAMBDA_RECONSTRUCTION=1` fixed and no explicit weight on its own
KL term (i.e., PercepT's own KL is implicitly weight-1 against a
reconstruction term of comparable scale); here there is no reconstruction
term at all, only two InfoNCE losses whose scale is unrelated to KL's, so
a fresh, modest bracket is the right starting point, not a blind reuse of
a PercepT constant. Report, for each value: held-out emotion AMI, held-out
genre AMI, held-out silhouette, the held-out Pareto bar (emotion AMI >
0.1236 AND genre AMI > 0.1954), the collapse diagnostic trajectory, and the
existing `content_recall`/`affect_recall` trajectory (a large `LAMBDA_DEC`
could plausibly wreck teacher-graph fidelity even if clustering looks
fine — report both, do not only report clustering metrics).

**Winner selection:** same rule as the noise-schedule sibling — among
values clearing the held-out Pareto bar, pick the highest held-out
silhouette; if none clears, pick the highest held-out silhouette among all
three and label it explicitly as "best available, does not clear the
Pareto bar," per that sibling's exact wording convention.

**Stress test:** the winning `LAMBDA_DEC` at `SEEDS = (7, 123, 2024)`, same
methodology as every other 4-seed stress in this investigation (fresh
model, fresh K-means center init with `random_state=seed`, fresh
`torch.manual_seed`/`np.random.seed`).

## Report

Write
`src/test/20260923_artelingo_buddy_analysis/attention_h1_dec_hybrid_pilot_report.md`
with: the 3-value screen table (both splits, all metrics, collapse
diagnostics, recall trajectory), the winner-selection rationale, the
four-seed stress table and summary statistics, and an explicit comparison
table against ALL FIVE established reference points already used in this
investigation's other buddy reports: (a) Attention-h1 baseline
(0.1249/0.2404, silhouette 0.0392), (b) Leiden pseudo-contrastive alone
(0.1165/0.2351, silhouette 0.0877), (c) noise+schedule alone (0.1306/0.1973
seed 42; four-seed mean available in its own report), (d) the noise-
schedule+pseudo-contrastive combination (0.1210/0.2623, silhouette 0.0789,
missed the bar), (e) PercepT-replication standing balance-hack
(0.1252/0.2486, silhouette never measured), (f) PercepT-replication
faithful recipe (0.1092/0.3288, silhouette 0.5120). State plainly, in a
clearly headed final section: does this DEC-loss hybrid close a meaningful
fraction of the silhouette gap to the faithful-recipe PercepT replication,
does it do so while clearing the held-out AMI Pareto bar, and is that
result robust across 4 seeds or only a single-seed screen artifact. This
investigation's established convention is a blunt, numeric verdict — if
this collapses or underperforms every buddy variant tried so far, report
that plainly, exactly as prior negative results in this project have been
reported.

Do not touch git, do not modify any other file in the repository.
