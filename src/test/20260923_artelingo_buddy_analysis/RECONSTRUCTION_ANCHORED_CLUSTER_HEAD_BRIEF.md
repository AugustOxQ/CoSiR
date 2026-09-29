# Brief: a properly reconstruction-anchored clustering head, frozen buddy trunk (brainstorm candidate 4 / the "fifth attempt")

Write one new script. Do NOT run it — execution happens separately, on
GPU, outside this task.

## Context: why this design is different from every prior attempt tonight

Read, in full, before writing any code:
- `docs/reports/2026-09-26_buddy_silhouette_gap_brainstorm.md`, candidate 4
  — "Add a genuinely reconstruction-anchored clustering head to
  Attention-h1." Its own recommendation: pretrain an encoder/decoder to
  reconstruct the clean, frozen 32-D fused buddy vector, initialize
  K-means only afterward, then jointly optimize reconstruction and DEC KL
  **while keeping the buddy trunk frozen for this first diagnostic** (not
  jointly fine-tuning Attention-h1 itself — that is explicitly a separate,
  more expensive variant this pilot does not attempt).
- `run_attention_h1_decoupled_cluster_head_detached_pilot_report.md` — the
  prior attempt closest to this one. It used a bare `ClusterHead` MLP
  (32→32→32, unconstrained output) trained with **KL loss alone, no
  reconstruction term at all**, and its own cluster-latent silhouette
  never rose above ~0.002 mean — consistent with DEC's well-known
  tendency to collapse toward a degenerate, uninformative mapping without
  a reconstruction anchor.
- `percept_on_buddy_embedding_pilot_report.md` — the other close prior
  attempt (brainstorm candidate 2). It **did** have a reconstruction term
  (PercepT's own `lambda_R=1`), but used PercepT's unmodified 128-D-latent
  autoencoder unchanged, which is **overcomplete** relative to the 32-D
  input (128 > 32) — its own loss-scale audit showed the reconstruction/KL
  ratio shifted 433× from the original recipe, and the resulting DEC
  partition scored **negative** silhouette back in buddy's native space,
  consistent with an overcomplete decoder reconstructing well without the
  encoder needing to find well-separated clusters to do it.

**This pilot is deliberately different from both**: it has a real
reconstruction anchor (unlike the decoupled-head attempts), and its
cluster latent is **undercomplete** (smaller than the 32-D input, unlike
PercepT's unmodified autoencoder) — matching the classical rationale for
autoencoder-based clustering, where the bottleneck itself is what forces
the encoder to discard noise and keep only clustering-relevant structure.
Input and reconstruction target are both exactly 32-D, so there is no
PercepT-style dimensionality mismatch to audit for loss-scale distortion
this time.

## What to build

Write
`src/test/20260923_artelingo_buddy_analysis/run_reconstruction_anchored_cluster_head_pilot.py`.

**Frozen trunk, no retraining of Attention-h1**: load buddy's already-saved
`attention_h1_embedding_snapshot.npz` (`train_embedding_post`,
`heldout_embedding_post`, `train_emotion`/`genre`,
`heldout_emotion`/`genre`, `train_paintings`/`heldout_paintings`) and use
those frozen 32-D vectors directly as the fixed input to everything below
— no buddy model is instantiated or trained in this pilot at all.

**Architecture**: `nn.Sequential(nn.Linear(32, 24), nn.ReLU(), nn.Linear(24, 16))`
encoder, `nn.Sequential(nn.Linear(16, 24), nn.ReLU(), nn.Linear(24, 32))`
decoder (16-D cluster latent, deliberately undercomplete relative to the
32-D input — state this reasoning explicitly in the script's docstring and
the report's Method section).

**Two-phase training, seed 42, single screen (no multi-seed stress unless
a result looks promising enough to justify one, per this investigation's
established staged-gate discipline)**:
1. Pretrain phase: reconstruction-only (`F.mse_loss(decoder(encoder(x)), x)`)
   on the frozen train embeddings, Adam, a reasonable fixed epoch budget
   and learning rate (reuse PercepT's own `PRETRAIN_LEARNING_RATE`/
   `PRETRAIN_EPOCHS` constants from `run_percept_stage1_pilot.py` if they
   transfer sensibly at this much smaller input/output width — state
   whether you kept them or chose different values, and why).
2. K-means: fit `N_INITIAL_CLUSTERS = 100` centers
   (`sklearn.cluster.KMeans(n_clusters=100, n_init=10, random_state=42)`)
   on the pretrained encoder's clean train cluster-latent (16-D).
3. Joint phase: reuse PercepT's own `soft_assignments`/`target_distribution`
   (Student's-t kernel — appropriate here since the cluster latent is
   unconstrained Euclidean, not L2-normalized) and `prune_centers`
   (norm-based, also appropriate here for the same reason) imported
   unmodified from `run_percept_stage1_pilot.py`. Total loss:
   `LAMBDA_DEC * kl_loss + LAMBDA_RECONSTRUCTION * reconstruction_loss`,
   screening `LAMBDA_DEC` over `(0.1, 0.5, 1.0)` at seed 42 with
   `LAMBDA_RECONSTRUCTION = 1` fixed (matching PercepT's own convention,
   now scale-appropriate since input/output are both 32-D). Use the same
   30-epoch linear KL warm-up every other DEC-hybrid pilot tonight used,
   for the same reason (avoid premature self-sharpening collapse).
   Log the same collapse diagnostics (all-100-center min/max/median/
   below-1%) every other DEC-hybrid pilot tonight logs, at the same
   cadence.
4. Prune to 67 surviving centers, assign final hard labels via
   `soft_assignments(cluster_latent, surviving_centers).argmax(dim=1)`
   for both train and held-out.

**Evaluation** (reuse the exact conventions established across tonight's
DEC-hybrid pilots): held-out emotion/genre AMI against the Pareto bar
(emotion > 0.1236, genre > 0.1954); silhouette in **both** the 16-D
cluster latent and the original 32-D buddy embedding (same seed-42
two-stage sampling convention as every other pilot in this directory);
occupancy/collapse diagnostics for the 67 surviving centers on both
splits.

**Winner selection and stress**: among the three screened `LAMBDA_DEC`
values, pick the one clearing the Pareto bar with highest held-out
32-D-space silhouette; if none clears, pick the best available and label
it explicitly as such, per this investigation's established wording
convention. Stress only the winner at seeds 7, 123, 2024 (fresh K-means
init and pretrain per seed, same methodology as every other 4-seed stress
tonight).

## Report

Write
`src/test/20260923_artelingo_buddy_analysis/reconstruction_anchored_cluster_head_pilot_report.md`
with: the screen table, collapse diagnostics, the four-seed stress table,
and an explicit comparison against ALL prior DEC-hybrid attempts tonight
(Euclidean, vMF, decoupled un-detached, decoupled detached, and PercepT-
on-buddy-embedding) plus the plain buddy baseline and PercepT's faithful
recipe, using each one's own already-published numbers (read them from
their actual report files, do not guess). State plainly whether adding a
genuine, appropriately-sized reconstruction anchor is the missing
ingredient the four prior clustering-loss attempts lacked, or whether this
is now a sixth negative result — and if the latter, state explicitly that
this investigation has now tested every mechanism the brainstorm memo
identified for attaching a DEC-style clustering objective to Attention-h1
and all have failed, which is itself a strong, well-evidenced conclusion
worth stating in those terms rather than leaving open-ended.

Do not touch git, do not modify any other file in the repository.
