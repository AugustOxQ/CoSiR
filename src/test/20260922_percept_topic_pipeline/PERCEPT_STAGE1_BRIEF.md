# Brief: PercepT Stage 1 — P-Topic Formation (literal replication, ArtELingo)

Write a new script
`src/test/20260922_percept_topic_pipeline/run_percept_stage1_pilot.py`.
Do NOT run it — execution happens separately, on GPU, outside this task.
Also write `src/test/20260922_percept_topic_pipeline/PERCEPT_STAGE1_BRIEF_NOTES.md`
if you make any judgment call not fully pinned down below — record exactly what you
chose and why, one paragraph each, before writing the main script.

## Context: what this replicates

"Beyond Semantics: Modeling Factual and Affective Perceptual Experiences from
Vision-Language Data" (arXiv 2606.03345), method PercepT. Its Stage 1
("P-Topic Formation") fuses a factual (CLIP) and an affective (GoEmotions-tuned
text encoder) embedding, autoencodes the fusion into a 128-dim latent, and runs
Deep Embedded Clustering (DEC) to sharpen ~100 initial K-means clusters into
~67 final "P-Topics." Full mechanism spec is in
`src/test/20260922_percept_brainstorm/BRIEF.md` lines 41-60 — read that section
in full before writing any code; it is the ground truth for what the paper does.

This project (CoSiR / ArtELingo buddy-graph investigation) has already run 9
pilots exploring content/affect fusion, all documented in
`docs/reports/2026-09-22_artelingo_fusion_mechanism_investigation.md` (read in
full) and its companion `docs/reports/2026-09-22_artelingo_fusion_diagnostics_report.md`.
The current best result is **Attention-h1**
(`src/test/20260923_artelingo_buddy_analysis/learned_student_arch_sweep_pilot_report.md`):
held-out emotion AMI=0.1249, genre AMI=0.2404 — the first configuration to clear
this investigation's predeclared held-out Pareto bar (emotion AMI > 0.1236 AND
genre AMI > 0.1954, both simultaneously). This new PercepT-style pilot is a
genuinely different mechanism (learned reconstruction-grounded latent + DEC,
vs. a small contrastive two-teacher student) being tested against the exact
same bar for direct comparability.

Two prior DEC pilots already exist in the sibling directory
(`src/test/20260923_artelingo_buddy_analysis/run_dec_pilot.py` and
`run_dec_pilot_v2.py` — read both in full for reusable conventions), but they
ran DEC on the 28-dim GoEmotions-probability affect signal ALONE, not on a
PercepT-style fused content+affect embedding. This is the first pilot to build
the actual fused input.

**Read these sibling-directory files in full before writing anything, for
conventions and REUSE (import via the same `load_sibling_module()` /
`importlib.util.spec_from_file_location` pattern already used throughout this
investigation — do not re-derive what's already there):**
- `../20260923_artelingo_buddy_analysis/run_pipeline.py` — `log()`,
  `load_dedup_features()` (returns per-painting `img_all`/`txt_all` CLIP arrays;
  note `img_dim = img_all.shape[1]`, do not hardcode 768 — this repo's CLIP
  backbone is ViT-B/32, so the real dimension is smaller than the paper's 768,
  see "Dimension mismatch" below), `external_metrics()`, `majority()`,
  `load_genre_map()`, `assert_extraction_complete()`, `TRAIN_JSON`,
  `STORAGE_DIR`.
- `../20260923_artelingo_buddy_analysis/run_affect_pilot.py` —
  `extract_affect_nodes()` for the caption-grouping/batching pattern (you will
  write a new extraction function for embeddings rather than probabilities,
  see below, but reuse its row-to-painting grouping, `caption_text()` string-
  vs-list handling, and `np.add.at` mean-pool-by-painting-index logic
  verbatim).
- `../20260923_artelingo_buddy_analysis/run_dec_pilot_v2.py` — the
  convergence-controlled DEC training loop (stability criterion
  `fraction_changed < 0.001`, epoch ceiling 500, Student's-t soft assignment,
  self-sharpened target `P`, `KL(P||Q) + lambda_R * L_R` loss with
  `lambda_R=1.0`, full-batch training since N≈61k is small enough). **Reuse
  this DEC training loop's mechanics exactly**, including its convergence
  criterion — this project already found (dec_pilot_report.md vs
  dec_pilot_v2_report.md) that convergence-controlled DEC (lr=1e-4) gives a
  materially different, more trustworthy result than a fixed 100-epoch run
  (lr=1e-3). This is a deliberate, documented deviation from the paper's fixed
  200-epoch schedule, made because this project already validated the
  convergence-controlled approach as more reliable on this exact
  autoencoder+DEC combination — state this plainly in the report, do not
  present it as a silent change.
- `../20260923_artelingo_buddy_analysis/run_cca_audit_pilot.py` — the
  monkey-patch held-out evaluation pattern (`heldout_pipeline.STORAGE_DIR` /
  `heldout_pipeline.TRAIN_JSON` reassignment on a freshly-loaded sibling
  module) used to get a genuinely held-out (val+test, zero train overlap)
  evaluation split. **Reuse this exact pattern** — every pilot in this
  investigation fits on train only and reports both train and held-out
  metrics; this one must too.

## Fused embedding h (Eq. 2 of the paper, adapted for this repo's CLIP dims)

**Content component h_C:** For each painting, take the CLIP image feature and
CLIP text feature from `load_dedup_features()`, L2-normalize each
independently, then concatenate them (this matches the existing
`content_features()` convention already used in `run_cca_audit_pilot.py` —
reuse that function directly by importing it, do not reimplement). This gives
an L2-normalized-per-half content vector of dimension `2 * img_dim`. Call the
final L2-normalized-as-a-whole vector `h_C'` (i.e. L2-normalize the
concatenated result once more so the whole vector has unit norm — this is the
"norm-rescaled" step of Eq. 2).

**Affect component h_E:** The paper's `h_E` is a 768-dim TEXT EMBEDDING from a
GoEmotions-finetuned encoder — NOT a probability vector. Every earlier pilot
in this investigation (`run_affect_pilot.py`, `run_dec_pilot.py`,
`run_dec_pilot_v2.py`) instead used the 28-dim GoEmotions SIGMOID PROBABILITY
output. For literal fidelity to the paper here, extract the actual embedding:
load `AutoModel.from_pretrained("SamLowe/roberta-base-go_emotions")` (the BASE
encoder, not `AutoModelForSequenceClassification` — same checkpoint, different
head-stripped load path), run the same caption batching/tokenization as
`extract_affect_nodes()`, take `outputs.last_hidden_state`, mean-pool over the
attention-masked tokens (standard mean-pooling: sum `last_hidden_state *
attention_mask.unsqueeze(-1)` over the token dimension, divide by
`attention_mask.sum(dim=1, keepdim=True)`), then mean-pool by painting across
that painting's captions (same `np.add.at`-by-painting-index pattern as
`extract_affect_nodes()`). This gives one 768-dim embedding per painting
(roberta-base hidden size). L2-normalize it to get `h_E`. State explicitly in
the report that this pilot uses the embedding-level affect signal, a genuine
methodological upgrade over every prior affect pilot in this investigation
(which used 28-dim label probabilities), and that the backbone
(`SamLowe/roberta-base-go_emotions`, a RoBERTa) substitutes for the paper's
ModernBERT-family encoder — same GoEmotions fine-tuning objective, different
backbone family, because that is the encoder already validated and cached in
this project.

**Dimension mismatch and the combination rule.** The paper's Eq. 2 is an
elementwise weighted sum `h = (2*h_C' + h_E)/3`, which requires `h_C'` and
`h_E` to share a dimension (768 in the paper, because their CLIP variant
happens to also produce 768-dim features). This repo's CLIP backbone is
ViT-B/32 (see CLAUDE.md), so `h_C'` (dimension `2 * img_dim`, likely 1024) and
`h_E` (768) do NOT share a dimension — an elementwise sum is not directly
possible without inventing a projection layer, which would add an untrained
or separately-trained component the paper does not call for. **Resolve this
by concatenation-with-repetition instead of elementwise summation**: build
`h = L2_normalize(concat([h_C', h_C', h_E]))` — two copies of the (unit-norm)
content vector and one copy of the (unit-norm) affect vector, concatenated
then L2-normalized as a whole. This preserves the paper's 2:1 content:affect
weighting ratio (each content copy contributes equally to the final vector,
so doubling the copies doubles content's share of the combined vector's norm
budget, mirroring the coefficient 2 in front of `h_C'`) without requiring
matching dimensions. State this substitution plainly in the report as a
necessary, documented deviation from the literal Eq. 2 formula — do not
silently reinterpret "weighted sum" as "weighted concatenation" without
flagging it.

## Autoencoder (Stage 1, paper architecture)

- Encoder: `Linear(D_in, 500) -> ReLU -> Linear(500, 500) -> ReLU ->
  Linear(500, 2000) -> ReLU -> Linear(2000, 128)` (final layer, no
  activation — this is the latent `z`). `D_in` is whatever
  `h`'s dimension works out to (`2*img_dim*2 + 768`... i.e. compute it
  dynamically from the actual concatenated vector, do not hardcode).
- Decoder: exact mirror, `Linear(128, 2000) -> ReLU -> Linear(2000, 500) ->
  ReLU -> Linear(500, 500) -> ReLU -> Linear(500, D_in)`.
- Pretrain: MSE reconstruction loss `||h - h_hat||_2^2` (mean over batch and
  dims), Adam lr=1e-3, batch_size=1024, 100 epochs, shuffled each epoch. Log
  mean epoch loss every 10 epochs via `log()`, same as
  `run_dec_pilot_v2.py`'s pretrain logging.

## K-means init and DEC (reuse `run_dec_pilot_v2.py`'s exact training loop)

- `sklearn.cluster.KMeans(n_clusters=100, n_init=10, random_state=42)` fit on
  the pretrained encoder's latent `z` for all train-split nodes.
- DEC joint training exactly as `run_dec_pilot_v2.py`: Student's-t soft
  assignment (`q_ij = (1 + ||z_i - mu_j||^2)^-1`, row-normalized), self-
  sharpened target `p_ij = q_ij^2 / sum_i(q_ij)` (row-normalized, detached),
  loss `KL(P||Q) + 1.0 * L_R`, Adam lr=1e-4, full-batch, convergence via
  `fraction_changed < 0.001` stability criterion, epoch ceiling 500. Log the
  same trajectory fields `run_dec_pilot_v2.py` logs (pretrain loss every 10
  epochs; joint total/KL/recon and `fraction_changed` at the same cadence it
  uses).

## Norm-thresholding pruning (paper prunes 100 -> ~67 surviving topics)

The paper's exact pruning rule beyond "norm-thresholding" is not fully
specified in the brainstorm brief. Implement this concrete, paper-grounded
proxy: after DEC converges, compute the L2 norm of each of the 100 final
cluster centroids `mu_j` in latent space. Rank centroids by norm, descending.
Keep the top 67 (matching the paper's own reported retention rate of ~67/100)
and drop the bottom 33 as "noise centers." State this explicitly in the
report as an implemented approximation of the paper's pruning step, grounded
in its own reported retention ratio, not a literal reproduction of its
unstated exact rule.

## Evaluation (this investigation's established discipline, applied here for the first time to a DEC-based method)

Fit everything (autoencoder pretrain, K-means init, DEC joint training,
pruning) on the TRAIN split only. Then, using the monkey-patch
`heldout_pipeline.STORAGE_DIR`/`TRAIN_JSON` pattern from
`run_cca_audit_pilot.py`, load the genuinely held-out (val+test) painting
nodes, compute their `h` fused embeddings (same extraction functions, held-out
captions), pass them through the FROZEN trained encoder to get held-out `z`,
compute their soft assignment `Q` against the 67 surviving centroids only
(drop the 33 pruned columns before the row-normalization step), and take
argmax as each held-out painting's hard topic label. There is no fallback
needed for "assigned to a pruned topic" — pruned centroids are excluded from
the assignment computation entirely, so a held-out point can only be assigned
to a surviving topic.

Report, for BOTH train and held-out splits:
- Emotion AMI, emotion V-measure, genre AMI, genre V-measure (same
  `external_metrics()` / `load_genre_map()` calls every other pilot uses, same
  1144-painting genre-labelled overlap for the genre numbers).
- Cluster-size collapse check on the 67 surviving topics, using this
  investigation's predeclared rule (collapse if >50% of clusters have <1% of
  assigned nodes).
- Final latent silhouette score (all train nodes, and separately all held-out
  nodes) — report for paper-comparability context, but explicitly state that
  the paper's reported 0.97 was measured on their own held-out fused-embedding
  input directly (a much less lossy comparison point than a 128-dim DEC
  latent evaluated against label-based external metrics), so no attempt
  should be made to treat a large silhouette gap as a failure signal on its
  own — AMI against real labels is this investigation's primary criterion,
  not silhouette.

## Predeclared success criterion (same bar as the rest of this investigation)

State this explicitly, verbatim, before reporting results (do not derive
verdict language ad hoc): **held-out Pareto bar = emotion AMI > 0.1236 AND
genre AMI > 0.1954, both simultaneously.** This is the exact bar Attention-h1
cleared. Report whether this pilot clears it, using the same "Real success" /
"Merely a compromise" / "Collapsed" verdict language this investigation has
used throughout (verdict = "Collapsed" if the cluster-size collapse rule
fires; "Real success" if not collapsed AND both held-out AMI thresholds
clear; "Merely a compromise" otherwise).

## Report

Write `src/test/20260922_percept_topic_pipeline/percept_stage1_pilot_report.md`
with the same structure and rigor as
`../20260923_artelingo_buddy_analysis/learned_student_arch_sweep_pilot_report.md`
(read it for tone/format) — training trajectories, collapse detection, a
results table (architecture/split/emotion AMI/genre AMI/collapse
verdict/silhouette), and an explicit statement of every documented deviation
from the literal paper recipe (the concat-not-sum fusion rule, the
convergence-controlled DEC schedule instead of fixed 200 epochs, the norm-
threshold pruning proxy, the RoBERTa-vs-ModernBERT affect backbone).

This is Stage 1 only (P-Topic Formation). Stage 2 (P-Topic Mapping, the
supervised image-only classifier that predicts these frozen topic labels) is
a separate follow-up pilot, scoped only after these results are in — do not
build it now.
