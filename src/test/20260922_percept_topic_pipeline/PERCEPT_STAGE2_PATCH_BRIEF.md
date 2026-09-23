# Brief: PercepT Stage 2 — P-Topic Mapping with genuine patch-token attention pooling

This SUPERSEDES `PERCEPT_STAGE2_BRIEF.md` (that brief assumed no patch-level
features were available; the orchestrating session has since verified they
are). Ignore the old brief's "global CLIP embedding only" design entirely.

Write two new files:
1. `src/test/20260922_percept_topic_pipeline/run_percept_patch_feature_extraction.py`
2. `src/test/20260922_percept_topic_pipeline/run_percept_stage2_pilot.py`

Do NOT run either script — execution happens separately, on GPU, outside
this task.

## Context: patch features are real and verified working here

Read `src/test/20260922_percept_topic_pipeline/run_percept_stage1_pilot.py`,
`run_percept_stage1_cluster_count_sweep_pilot.py`, and
`run_percept_stage1_cluster_count_sweep_v2_pilot.py` in full first, for the
exact Stage 1 mechanics (fused embedding, autoencoder, K-means, DEC,
pruning) you must reuse unchanged.

`src/model/cosirmodel.py`'s `encode_img()` (lines ~162-172) computes BOTH a
pooled embedding and a full per-token sequence:
```python
img_output = self.vision_model(**images)
img_emb = self.visual_projection(img_output.pooler_output)       # pooled
img_full = self.visual_projection(img_output.last_hidden_state)  # patch tokens
```
**Do not import `src.model.cosirmodel`** — this project's memory records
that importing `src.model` breaks in this environment (a broken local
llvmlite dependency chain unrelated to this code). Replicate the exact same
logic directly via raw HuggingFace `transformers`, the established bypass
pattern used throughout this whole investigation:
```python
from transformers import CLIPModel, CLIPProcessor
backbone = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
vision_model = backbone.vision_model
visual_projection = backbone.visual_projection
```
This was smoke-tested by the orchestrating session on real ArtELingo images
and confirmed working: `vision_model(**inputs).last_hidden_state` has shape
`[batch, 50, 768]` (1 CLS + 49 patch tokens for ViT-B/32 at 224x224), and
`visual_projection(...)` maps the last dim to 512, matching the existing
cached `img_features` dimensionality. No NaNs, sane per-token norms.

## Script 1: `run_percept_patch_feature_extraction.py`

Extracts and caches patch-token embeddings for every unique ArtELingo
painting (train and held-out separately) — this is the expensive,
one-time step, kept separate from Stage 2 training so it is not repeated
if Stage 2's classifier design changes.

- Image paths: for each row in `TRAIN_JSON` (`/data/PDD/artelingo/artelingo_train.json`
  for train, `HELDOUT_JSON` (`/data/PDD/artelingo/artelingo_val_test.json`)
  for held-out — reuse the exact constants already defined in
  `run_percept_stage1_pilot.py`), the image resolves at
  `/data/PDD/wikiart_proj/wikiart/<record['image']>`. Verified: every row
  sharing the same `painting` value has an identical `image` field (checked
  across 42,593 train paintings, zero inconsistencies), so take one row per
  unique painting — reuse `load_dedup_features()`'s own painting-dedup
  ordering (`sorted(by_painting.keys())`) so painting index alignment with
  the already-cached CLIP `img_features`/`txt_features` arrays is exact and
  reusable downstream without reordering.
- Load `Image.open(path).convert("RGB")` per image, batch via
  `CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")` (batch
  size 64, tune down if you hit memory issues — log actual batch size
  used), run through `vision_model` then `visual_projection` on
  `last_hidden_state` to get `[batch, 50, 512]` patch-token embeddings per
  image. Log progress every ~20 batches via `log()` (reuse the `log()`
  helper from the base pilot module).
- Cache to disk as a single float32 tensor per split:
  `/data/SSD2/pre_extract/artelingo_percept_patch_features/train_patch_features.pt`
  and `.../heldout_patch_features.pt` (create the directory if it does not
  exist). Shape `[N_paintings, 50, 512]` in the SAME painting order as
  `load_dedup_features()` returns (critical for later alignment — verify
  this by asserting the saved tensor's first dim equals `len(paintings)`
  from `load_dedup_features()` before writing).
- **Skip re-extraction if the cache file already exists and its first
  dimension matches the expected painting count** — log this and exit early
  for that split. This makes the script idempotent/resumable, since this is
  a one-off expensive step that later runs should not blindly repeat.
- Any image that fails to load (corrupt file, path not found) should be
  logged with its painting name and path, and raise a `RuntimeError` listing
  all failures at the end rather than silently skipping — a partial/missing
  cache would silently corrupt Stage 2's later alignment with the
  DEC-derived pseudo-labels, so fail loudly rather than continuing with
  gaps.

## Script 2: `run_percept_stage2_pilot.py`

Reuse `run_percept_stage1_cluster_count_sweep_pilot.py`'s and the base
pilot's exact Stage 1 mechanics (`load_sibling_module` pattern, K-means
init, DEC training loop, pruning) to re-fit the winning `N_INITIAL=60`,
`N_SURVIVING=40`, `LAMBDA_BALANCE=1000`, `LAMBDA_RECONSTRUCTION=1`,
`SEED=42` configuration from scratch (there is no saved checkpoint from the
sweep — re-fitting deterministically at this exact seed is the only way to
reproduce it). **Before training Stage 2, verify this re-fit reproduces the
established numbers** (held-out emotion AMI=0.1238, genre AMI=0.2617,
within a small numerical tolerance, e.g. absolute difference < 0.002 on
each) — if it does not match closely, stop and report this as a
reproducibility failure rather than silently proceeding to train Stage 2 on
a different clustering than intended.

**Deriving frozen multi-label targets** (train and held-out, using the
frozen Stage-1 encoder + 40 surviving centers, via `soft_assignments()`
reused from the base pilot): for each painting, the target label set is
its `argmax` topic (always included, guaranteeing no all-zero rows) UNION
any topic where `q[topic] > 2.0/40`. Log the resulting label-count
distribution (mean/median/max labels per painting, fraction multi-labeled)
for both splits — this must appear in the report.

**Stage 2 model — genuine patch attention pooling, matching the paper's
better-performing mapper design** (load the cached `.pt` tensors from
Script 1 as this model's input, keyed by the same painting order):

```python
class AttentionPoolingMapper(nn.Module):
    def __init__(self, d_model=512, n_topics=40):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.classifier = nn.Linear(d_model, n_topics)

    def forward(self, patch_tokens):  # [batch, 50, d_model]
        query = self.query.expand(patch_tokens.shape[0], -1, -1)      # [batch, 1, d_model]
        attn_weights = (query @ patch_tokens.transpose(1, 2)) / (patch_tokens.shape[-1] ** 0.5)
        attn_weights = attn_weights.softmax(dim=-1)                    # [batch, 1, 50]
        pooled = (attn_weights @ patch_tokens).squeeze(1)               # [batch, d_model]
        return self.classifier(pooled)                                  # logits [batch, n_topics]
```
One learnable query vector attends over the 50 patch tokens (single-head,
scaled dot-product) to a single pooled vector, then a plain linear
multi-label head — this matches "attention-pooling over patches -> a single
pooled vector -> linear projection -> per-topic sigmoid score" from the
paper, and the paper's own ablation found this simpler design beats a
more complex multi-query cross-attention variant, so do not build anything
fancier than this.

- Loss: `nn.BCEWithLogitsLoss()` against the multi-hot train targets.
- Training: Adam lr=1e-3, 100 full-batch epochs (61,402 train paintings —
  full-batch is consistent with every other training loop in this
  investigation). Log mean epoch BCE loss every 10 epochs.
- No text or global pooled embedding is used as classifier input anywhere
  in this model — only the raw patch-token tensor, per the paper's
  image-only-at-inference constraint.

## Evaluation

On held-out (patch features + held-out multi-label targets derived from
the SAME frozen Stage-1 encoder/centers, projected forward, never refit):
compute per-topic AUC (`sklearn.metrics.roc_auc_score`) for all 40 topics,
report macro-averaged AUC as the headline number plus min/median/max
per-topic AUC. Skip (and explicitly log, do not silently drop) any topic
with zero positive or zero negative held-out examples.

**Baseline for context:** a trivial baseline scoring every held-out painting
with each topic's TRAIN-set marginal frequency (ignoring the image
entirely). Report the trained model's macro AUC next to this baseline's
macro AUC.

## Report

Write
`src/test/20260922_percept_topic_pipeline/percept_stage2_pilot_report.md`
with:
- Confirmation the Stage-1 re-fit reproduced the established K=60/40
  seed-42 numbers (state the actual re-fit values next to the established
  ones).
- Multi-label target statistics for both splits.
- The training loss trajectory.
- The held-out per-topic AUC results (macro/min/median/max, skipped-topic
  count) and the marginal-frequency baseline comparison.
- A plain, honest closing statement on whether the image-only attention-
  pooling mapper meaningfully beats the marginal-frequency baseline —
  this is the real test of whether it learned something from the image,
  not just topic base rates.
