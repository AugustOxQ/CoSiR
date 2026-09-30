# CoSiR v2 Block 1 extension: cross-encoder ablation (DINOv2 + e5)

> **For agentic workers:** function/class-formal code, no ad hoc scripts. Real GPU runs
> pre-authorized on the local machine. Checkbox steps.

**Goal:** test whether Block 1's "content-only Stage 1 plateaus at the raw-feature baseline"
finding (`docs/reports/auto/v2/2026-09-28_block1_stage1_validation.md`,
`docs/reports/2026-09-28_cosir_v2_..._raw_clip_baseline...`) is a property of buddy/InfoNCE
training itself, or a property of CLIP ViT-B/32's specific features. Reuse Block 1's exact
pipeline (`build_content_graph`, `train_stage1`, `detect_communities`) unchanged, only swapping
the input features from CLIP to **DINOv2 (image) + e5 (text)** — a meaningfully different,
non-CLIP encoder pairing.

**Context**: a prior RedCaps cross-VLM survival study
(`experiment/percept_topic_pipeline:src/test/20260716_buddy_cross_vlm/`, read its log at
`20260716_buddy_cross_vlm_log.md` for the full finding) found buddy-graph edges survive across 16
vision×text encoder combinations, but **vision encoder choice drives real structural variation** —
a DINOv2-based graph has real, measured disagreement with a CLIP-based graph (not identical
structure). That's necessary (not sufficient) for training against it to escape the "restates
CLIP" plateau. This has never been tested on ArtELingo or against real emotion AMI — this plan is
that test.

**Reference implementation for the encoders** (read for understanding, adapt fresh — do not
blindly copy, this project's convention all night): `HeldoutEncoder` class in
`experiment/percept_topic_pipeline:src/test/20260708_heldout_grid/heldout_models.py`. It already
has the exact model IDs, pooling functions, and the required `"query: "` prefix for e5 — reuse
those exact choices, they're already validated to load and run correctly in this environment.

## Global constraints

- Branch: `cosir-v2` (`/project/CoSiR-v2`).
- Reuse Block 1's unmodified pipeline: `build_content_graph`/`GraphConfig` (`src/model/graph.py`),
  `train_stage1`/`Stage1Config`/`AttentionFusionStudent` (`src/train/stage1.py`,
  `src/model/student.py`), `detect_communities`/`community_stats` (`src/model/communities.py`).
  Do not modify any of these — this ablation only changes what features feed in.
- `seed=42`. No `cuml`/`cugraph`.
- Codex is the default implementer, direct invocation
  (`codex e --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check -c mcp_servers={}
  -C /project/CoSiR-v2 --json -`). No `.ccg/tasks/` scaffolding, no secondary review layer —
  Claude reviews every diff.
- Local GPU real runs pre-authorized.

---

### Task 1: extract DINOv2 + e5 features for ArtELingo (train split, full scale)

**Files:**
- Create: `src/test/20260929_cross_encoder_stage1/extract_features.py`
- Create: `src/test/20260929_cross_encoder_stage1/README.md`

**What to do:**
1. Adapt (not copy verbatim — this repo's `src/` has none of the RedCaps-specific pipeline code
   the reference script depended on) a `HeldoutEncoder`-equivalent for exactly two models:
   `facebook/dinov2-small` (image, CLS-token pooling from `last_hidden_state[:, 0]`) and
   `intfloat/e5-base-v2` (text, attention-masked mean pooling, `"query: "` prefix on every
   caption). Output L2-normalized float32 arrays, matching the reference's `_norm` convention.
2. **Smoke test first, on a small subsample (e.g. 200-500 rows), before the full run.** Confirm:
   both models load, both encode without errors, output shapes/dims are as expected (384 for
   DINOv2, 768 for e5), no NaNs. Only proceed to the full run after this passes — say so
   explicitly in your report, don't skip straight to the full 308,723-row run on an unverified
   pipeline.
3. Full run: extract DINOv2 image features + e5 text features for **all 308,723 rows** of
   `/data/PDD/artelingo/artelingo_train.json` (same row order/positional convention as every prior
   task tonight — images resolve via `/data/PDD/wikiart_proj/wikiart/<image field>`). Batch
   reasonably (e.g. 128-256 images/texts per batch) for GPU throughput; this is real, fresh
   extraction over 308K images, expect it to be the most time-consuming step in this whole
   plan — report actual wall-clock time, don't estimate it.
4. Save both feature arrays to disk (e.g. `.npy`, under
   `src/test/20260929_cross_encoder_stage1/features/` — this directory should be gitignored as
   data, not committed; check `.gitignore` covers it or add a rule, don't commit large binary
   arrays to git).
5. Report actual extraction wall-clock time, shapes, and basic sanity stats (no NaNs, reasonable
   norm distribution) in the README.

- [ ] Smoke test on a subsample, confirm clean.
- [ ] Full extraction, save features, record wall-clock time.
- [ ] Commit: `git add src/test/20260929_cross_encoder_stage1/extract_features.py src/test/20260929_cross_encoder_stage1/README.md .gitignore && git commit -m "feat(cosir-v2): DINOv2+e5 feature extraction for ArtELingo cross-encoder ablation"`
  (only touch `.gitignore` if you needed to add a rule; do not commit the `.npy` feature files
  themselves).

---

### Task 2: re-run Block 1's pipeline on DINOv2+e5 features, real comparison

**Files:**
- Create: `src/test/20260929_cross_encoder_stage1/run_cross_encoder_validation.py`
- Create: `docs/reports/auto/v2/2026-09-29_block1_cross_encoder_ablation.md`

**What to do:**
1. Load the DINOv2/e5 features saved by Task 1 (do not re-extract).
2. Run Block 1's unmodified pipeline: `build_content_graph(dinov2_feat, e5_feat, GraphConfig())`
   → `train_stage1(dinov2_feat, e5_feat, graph, Stage1Config())` → `detect_communities(embeddings)`
   → `community_stats`.
3. **Also compute the raw-feature baseline** for this encoder pair (mirroring
   `raw_clip_baseline.py` from Block 1's own validation): L2-normalize and concatenate raw
   DINOv2+e5 features directly (no Stage 1 training), cluster with the same `detect_communities`,
   same defaults.
4. Compute the same real `emotion` label (from `artelingo_train.json`, same positional join) AMI
   for both: (a) DINOv2+e5-trained Stage 1, (b) raw DINOv2+e5 features clustered directly.
5. Report all four numbers side by side in a clear table: CLIP Stage 1 (0.036942, from
   `docs/reports/auto/v2/2026-09-28_block1_stage1_validation.md`), CLIP raw baseline (0.035781,
   same report), DINOv2+e5 Stage 1 (new), DINOv2+e5 raw baseline (new). Community counts/
   occupancy for all four too, where available.

**Interpretation to write plainly** (this is the actual point of the whole ablation, get this
right): does the **gap between trained-Stage1 and raw-baseline** widen for DINOv2+e5 compared to
CLIP (i.e., does buddy/InfoNCE training add more value when given different input features,
even if the absolute AMI numbers differ for other reasons)? Or does the same near-zero gap
persist regardless of encoder (suggesting the plateau is about the *training mechanism*, not the
*backbone*)? State this as the primary finding, not a footnote — it's the reason this ablation
exists. Also report the raw absolute numbers plainly (does DINOv2+e5 alone do better/worse/same
as CLIP alone at emotion correlation) as a secondary, genuinely interesting finding in its own
right.

- [ ] Run it for real, write the report with real numbers, plain verdict up front (matching this
  project's established report convention).
- [ ] Commit: `git add src/test/20260929_cross_encoder_stage1/run_cross_encoder_validation.py docs/reports/auto/v2/2026-09-29_block1_cross_encoder_ablation.md && git commit -m "docs(cosir-v2): cross-encoder (DINOv2+e5) ablation for Block 1 Stage 1"`

## Self-review

**Placeholder scan:** none. **Scope:** two tasks — extraction (with a smoke-test gate before the
expensive full run), then the actual comparison. Right-sized. **Honesty check carried from Block
1/Candidate A:** this plan states its actual falsifiable question up front (does the gap widen
or not) rather than assuming an outcome, matching the discipline already established tonight.
