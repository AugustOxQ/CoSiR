# Brief: DEC-vs-Leiden pilot

Write a new script `src/test/20260923_artelingo_buddy_analysis/run_dec_pilot.py`.
Do NOT run it — execution happens separately, on GPU, outside this task.

## Context

This directory has three existing, working scripts:
- `run_pipeline.py`: baseline CLIP img+txt union graph, community-vs-emotion AMI=0.0593.
- `run_affect_pilot.py`: GoEmotions-fused sweep, best emotion AMI=0.1160 at weight=4,
  but genre AMI collapses at that weight.
- `run_single_modality_pilot.py`: GoEmotions-affect-ONLY graph via Leiden, emotion
  AMI=0.1180 — barely above the fused result, meaning Leiden clustering itself looks
  like the bottleneck, not signal dilution.

Read all three files in full first, for conventions: the `log()` timestamped-print
helper, the `load_sibling_module()` / `importlib.util.spec_from_file_location`
pattern used to import `run_pipeline.py` and `run_affect_pilot.py` as libraries
(reuse `load_dedup_features()`, `external_metrics()`, `majority()`, `load_genre_map()`
from `run_pipeline.py`; reuse `extract_affect_nodes()`, `l2_normalize()` from
`run_affect_pilot.py`), the `assert_extraction_complete()` call at the top of `main()`,
and the `device = "cuda" if torch.cuda.is_available() else "cpu"` convention.

## Purpose

Test whether PercepT's actual clustering mechanism (Deep Embedded Clustering, DEC)
does meaningfully better than one-shot Leiden on the EXACT SAME GoEmotions-affect-only
input (61402 x 28, mean-pooled GoEmotions sigmoid probabilities per painting — get this
via `extract_affect_nodes()`, same as `run_single_modality_pilot.py` already does).
This isolates clustering METHOD from input SIGNAL, since Leiden already caps at
AMI=0.1180 on this exact input.

## Implementation (fully specified — implement exactly this, do not redesign)

1. Small PyTorch autoencoder (28-dim input is much smaller than typical DEC use
   cases, so use a small architecture, not any paper's oversized one):
   - Encoder: `Linear(28,64) -> ReLU -> Linear(64,32) -> ReLU -> Linear(32,16)`,
     producing latent `z` (16-dim).
   - Decoder mirrors it: `Linear(16,32) -> ReLU -> Linear(32,64) -> ReLU -> Linear(64,28)`.
   - Both as simple `nn.Sequential`.

2. Pretrain stage: MSE reconstruction loss between input and `decoder(encoder(input))`.
   Adam, lr=1e-3, batch_size=1024, 50 epochs, shuffled each epoch. Log mean epoch loss
   every 10 epochs via `log()`.

3. Initialize K=28 cluster centers via `sklearn.cluster.KMeans(n_clusters=28, n_init=10,
   random_state=42)` fit on the pretrained encoder's latent `z` for all 61402 nodes
   (no batching needed).

4. DEC joint training, 100 epochs, full-batch (N=61402 is small enough — compute over
   the whole dataset each epoch on GPU, no minibatching in this stage):
   - Student's-t soft assignment (degrees of freedom = 1, standard DEC formulation):
     `q_ij = (1 + ||z_i - mu_j||^2)^-1`, then row-normalize over j so each row of Q
     sums to 1. `z_i = encoder(input_i)`, recomputed each epoch (not frozen — this is
     the "joint" part of DEC).
   - Target distribution (self-sharpening, standard DEC formula):
     `p_ij = (q_ij^2 / sum_i(q_ij))`, then row-normalize over j so each row of P sums
     to 1. Detach P from the autograd graph — it's a fixed target for this step,
     recomputed fresh each epoch from the current Q.
   - Loss = `KL(P || Q)` (`torch.nn.functional.kl_div` with `log_target=False`, using
     `Q.log()` as input and `P` as target, `batchmean` reduction) + `lambda_R *
     reconstruction_loss` (lambda_R=1.0, reconstruction computed the same way as
     pretraining, same forward pass). Cluster centers `mu_j` are also `nn.Parameter`,
     trained jointly via the same optimizer (Adam, lr=1e-3, one optimizer over
     encoder+decoder+centers).
   - Log mean epoch total loss, KL component, and reconstruction component every 10
     epochs.

5. After DEC training: hard-assign each node via `argmax_j(Q_ij)` (final Q). Compute:
   - Cluster-size distribution: min/max/median cluster size, and how many of the 28
     clusters have <1% of nodes (< 614 of 61402). This is a collapse-detection check —
     this exact failure mode (near-total collapse into 1-2 clusters) is what broke the
     original Experiment 18 prototype bank this whole investigation is following up
     on. Report it prominently.
   - Community-vs-emotion AMI/V-measure via `external_metrics()` and
     `majority()`/`load_dedup_features()` for `majority_emotion`, and
     community-vs-genre AMI/V-measure via `load_genre_map()`, exactly like the other
     three scripts do.
   - Silhouette score on the final latent `z` using the hard cluster assignments
     (`sklearn.metrics.silhouette_score` — try the full 61402 first, subsample to
     10000 only if too slow).

6. Write `src/test/20260923_artelingo_buddy_analysis/dec_pilot_report.md`: state the
   DEC pretrain and joint-training loss trajectories (a few checkpoint values, not
   every epoch), the cluster-size collapse-detection stats, then a comparison table
   with two rows — "Leiden on GoEmotions-only (reference)": emotion AMI 0.1180, genre
   AMI 0.0396 (hardcode these, don't recompute) — and "DEC on GoEmotions-only (this
   run)": the actual numbers just computed. Then one paragraph stating plainly whether
   DEC cleared the predeclared bar of AMI > 0.177 (50% relative gain over 0.1180) AND
   did not collapse (fewer than half of the 28 clusters holding <1% of nodes) — both
   conditions needed to call it a real win, matching how the earlier pilots defined
   their success bars.

Print clear timestamped progress logs matching the other three scripts' `log()`
format throughout (pretrain start/progress, DEC training start/progress, final
evaluation).
