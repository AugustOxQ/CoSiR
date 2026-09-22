# Brief: Stage 1 — simplest learned two-teacher gated student

Write a new script
`src/test/20260923_artelingo_buddy_analysis/run_learned_student_stage1_pilot.py`.
Do NOT run it — execution happens separately, on GPU, outside this task.

## Context and why this run matters

The CCA audit (`cca_audit_pilot_report.md`) found a strong, held-out-
validated shared linear signal between content and affect (top component
r=0.7285, far above both the 0.15 bar and its permutation null), which
per the second brainstorm's decision rule (`joint_fusion_brainstorm_codex_findings.md`)
licenses building a small learned two-teacher student. That brainstorm was
explicit: **start with linear projections and a scalar/vector gate;
cross-attention is an ablation after that baseline, not the first build.**
This is that first, simplest build.

Read `run_pipeline.py`, `run_affect_pilot.py`, `run_single_modality_pilot.py`,
`run_bert_heldout_pilot.py`, and `run_cca_audit_pilot.py` in full first — this
reuses conventions and helpers from all of them, especially
`run_cca_audit_pilot.py`'s held-out-loading pattern and reference graphs.

## Architecture

```python
D_SHARED = 32
CONTENT_PCA_DIM = 50   # same as the CCA audit, for a fair comparison
```

- Content input: `concat(l2_normalize(img_nodes), l2_normalize(txt_nodes))`,
  then PCA-reduced to `CONTENT_PCA_DIM` (fit on train only, same convention
  as `run_cca_audit_pilot.py`'s `content_features()`/PCA step — reuse that
  exact logic, do not diverge).
- Affect input: raw 28-d GoEmotions probabilities (no PCA, matching every
  prior pilot's convention).
- `proj_content = nn.Linear(CONTENT_PCA_DIM, D_SHARED)`, `proj_affect =
  nn.Linear(28, D_SHARED)`. Both L2-normalize their output.
- Gate network: `nn.Sequential(nn.Linear(2 * D_SHARED, 16), nn.ReLU(),
  nn.Linear(16, 1), nn.Sigmoid())`, input is the concatenation of the two
  projected (L2-normalized) views, output is a per-node scalar `g in (0, 1)`.
- Student embedding: `z = l2_normalize(g * content_proj + (1 - g) *
  affect_proj)`.

## Training data and positive-pair sampling

```python
BATCH_SIZE = 1024          # positive pairs per teacher per step
TEMPERATURE = 0.1
LEARNING_RATE = 1e-3
MAX_EPOCHS = 200
PLATEAU_WINDOW = 5         # checkpoints
PLATEAU_REL_IMPROVEMENT = 0.01
CHECKPOINT_EVERY = 5       # epochs
SEED = 42
```

- Build `E_content` (train, via `build_buddy_graphs`, same as every prior
  pilot) and `E_affect` (train, via `single_modality_pilot.
  build_single_modality_graph` on raw GoEmotions nodes) exactly as done
  throughout this investigation.
- Extract each graph's edge list once as an `(n_edges, 2)` array of node-index
  pairs (upper triangle only, since graphs are symmetric — reuse the
  `triu`-based key extraction idiom from `src.conditional_buddy.buddy_graph.
  _adj_to_keys` for a clean approach, or simply `graph.tocoo()` filtered to
  `row < col`).
- Each training step: sample `BATCH_SIZE` edges from `E_content`'s edge list
  and `BATCH_SIZE` edges from `E_affect`'s edge list (with replacement if the
  edge list is smaller than `BATCH_SIZE`, `numpy.random.default_rng` seeded
  per-epoch from `SEED + epoch` for reproducibility), producing endpoint
  index pairs `(a_content, b_content)` and `(a_affect, b_affect)`.
- For each teacher, run all nodes referenced in that teacher's sampled batch
  through the student network once (content nodes' `img_nodes`/`txt_nodes`
  rows for content teacher; ALL 61,402 affect rows are cheap to embed, so no
  special-casing needed for the affect teacher's inputs — content features
  for a content-teacher batch, affect features for an affect-teacher batch;
  BOTH `proj_content` and `proj_affect` and the gate always run together
  per node, since every node has both a content and an affect representation
  by construction — do not skip either projection head for either teacher's
  batch).
- InfoNCE loss per teacher, in-batch negatives: for teacher batch with
  anchor embeddings `Z_a` and positive embeddings `Z_b` (each `(BATCH_SIZE,
  D_SHARED)`, L2-normalized), `logits = Z_a @ Z_b.T / TEMPERATURE`, target
  is the diagonal (`torch.arange(BATCH_SIZE)`), loss is
  `F.cross_entropy(logits, target)` (standard symmetric InfoNCE — also
  compute and average the transposed version, `Z_b @ Z_a.T`, for a
  symmetric loss, matching standard practice).
- `total_loss = content_loss + affect_loss` (equal weight — do not tune this
  weight; the whole point of Stage 1 is to test the simplest, un-tuned
  version first).

## Predeclared convergence and collapse diagnostics (every `CHECKPOINT_EVERY` epochs)

Load the held-out data and reference graphs exactly as
`run_cca_audit_pilot.py` does (`heldout_pipeline` pattern, `content_heldout`,
`affect_heldout`, held-out `content_graph`/`affect_graph` via
`single_modality_pilot.build_single_modality_graph`).

At each checkpoint, with the model in eval mode and no gradient:
1. Embed all held-out nodes (`content_heldout` through the train-fit PCA,
   then `proj_content`; `affect_heldout` through `proj_affect`; combine via
   the gate exactly as in training) to get held-out student embeddings.
2. Build a mutual-kNN graph on these held-out student embeddings (K=20, same
   convention as every other pilot).
3. Compute recall@K-style retrieval for BOTH teachers using the SAME
   `graph_overlap_fraction()` helper pattern from `run_cca_audit_pilot.py`
   (reuse it — import that module as a sibling and call its function
   directly): what fraction of each of a fixed 2,000-node held-out sample's
   TRUE content-graph neighbors, and TRUE affect-graph neighbors, appear in
   this checkpoint's student graph. Use the SAME 2,000-node sample (seeded
   once, `numpy.random.default_rng(SEED)`) at every checkpoint for a fair
   trajectory.
4. Record both teachers' checkpoint loss values and their gradient-norm
   share: after `total_loss.backward()`, before `optimizer.step()`, compute
   `grad_norm_content` = the L2 norm of gradients flowing into `proj_content`
   and the gate from `content_loss` alone (call `content_loss.backward(retain_graph=True)`
   separately to isolate this, then re-zero grads and do the real combined
   backward pass for the actual optimizer step — do this diagnostic
   computation only at checkpoint epochs, not every step, to avoid slowing
   down training) — report `grad_norm_content / (grad_norm_content +
   grad_norm_affect)` as content's gradient share.
5. Effective rank: compute the covariance matrix of a random 5,000-node
   sample of held-out student embeddings, its eigenvalues, and report the
   top eigenvalue's fraction of total variance explained (`top_eigenvalue /
   sum(eigenvalues)`), plus the count of eigenvalues needed to explain 95% of
   variance ("effective rank at 95%").
6. Gate distribution: mean, std, and the fraction of held-out nodes with
   `g < 0.1` or `g > 0.9` ("saturated").

Log every checkpoint's full diagnostic line clearly.

## Stopping rule

Stop when BOTH teachers' held-out content-recall and affect-recall (from
diagnostic #3) have each improved by less than `PLATEAU_REL_IMPROVEMENT`
(1%) relative, checkpoint-over-checkpoint, for `PLATEAU_WINDOW` (5)
consecutive checkpoints — or at `MAX_EPOCHS` (200), whichever comes first.
Do NOT stop based on training loss alone. Log which condition triggered the
stop.

## Collapse determination (apply at the final/stopping checkpoint)

Using the thresholds and language from the second brainstorm's own spec:
- **Collapsed** if either teacher's held-out recall (diagnostic #3) is not
  meaningfully above ITS OWN first-checkpoint (epoch `CHECKPOINT_EVERY`)
  value (define "meaningfully above" as a relative improvement of at least
  20% over the first checkpoint — if a teacher never clears this, its signal
  never developed), OR if that teacher's final gradient share (diagnostic
  #4) is below 5% or above 95% at the final checkpoint, OR if the final
  gate distribution (diagnostic #6) has over 90% of nodes saturated at
  either extreme.
- **Merely a compromise** (not collapsed, but not a real win either) if
  neither teacher collapsed by the above rule, but the final community
  partition (see below) does not clear the Pareto bar.
- **Real success** requires: not collapsed, AND the final partition clears
  the Pareto bar below.

## Final evaluation

Build the final mutual-kNN graph from the TRAIN student embeddings at the
stopping checkpoint (K=20, same repair convention), run Leiden (seed=42),
compute emotion/genre AMI + V-measure via `pipeline.external_metrics()` +
`pipeline.load_genre_map()`.

**Predeclared Pareto bar**: this run counts as a real, reportable win only
if BOTH: `emotion_AMI > 0.1236` (beats late union, the best emotion AMI
found among fusion methods) AND `genre_AMI > 0.1954` (beats hierarchical
refinement, the best genre AMI found among methods with real, non-null
affect signal). Falling short of either is not a failure of this script —
report the actual numbers plainly either way.

## Report

Write
`src/test/20260923_artelingo_buddy_analysis/learned_student_stage1_pilot_report.md`:

- Explain the architecture and training setup in plain language.
- A full checkpoint trajectory table: epoch, content held-out recall,
  affect held-out recall, content loss, affect loss, content gradient
  share, top-eigenvalue variance fraction, effective rank at 95%, gate
  mean/std/saturated-fraction.
- State which stopping condition triggered, and at which epoch.
- State the collapse determination plainly (collapsed / compromise / real
  success), with the specific numbers that drove that call.
- The final comparison table against the existing frontier:

  | signal | emotion AMI | emotion V-measure | genre AMI | genre V-measure |
  |---|---:|---:|---:|---:|
  | Content-only (reference) | 0.0593 | 0.0600 | 0.4384 | 0.4572 |
  | GoEmotions-affect-only (reference) | 0.1180 | 0.1189 | 0.0396 | 0.0937 |
  | Late fusion — union (reference) | 0.1236 | 0.1241 | 0.1394 | 0.1677 |
  | Hierarchical refinement (reference) | 0.1072 | 0.1141 | 0.1954 | 0.3901 |
  | Learned student, Stage 1 (this run) | *computed* | *computed* | *computed* | *computed* |

- State the Pareto-bar verdict explicitly (cleared both / cleared one /
  cleared neither), and if a recommendation for escalating to a richer
  architecture (Stage 2: small MLP heads, or cross-attention ablation) is or
  is not warranted BY THE EVIDENCE — per the second brainstorm's own
  caution, do not recommend more architecture if the failure mode looks like
  genuinely incompatible teacher constraints (both teachers achieve good
  individual recall, i.e. not collapsed, but the final partition still can't
  satisfy both AMI targets) rather than insufficient model capacity.

Print clear timestamped progress logs matching the other scripts' `log()`
format throughout, including the full diagnostic line at every checkpoint.
