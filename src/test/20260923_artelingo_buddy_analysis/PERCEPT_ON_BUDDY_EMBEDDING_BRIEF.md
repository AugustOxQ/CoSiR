# Brief: PercepT's unmodified autoencoder+DEC recipe, fed buddy's frozen embedding instead of raw CLIP+affect (brainstorm candidate 2)

Write one new script. Do NOT run it — execution happens separately, on
GPU, outside this task.

## Context

Read, in full, before writing any code:
- `docs/reports/2026-09-26_buddy_silhouette_gap_brainstorm.md`, candidate 2
  — "Feed frozen Attention-h1 vectors into PercepT's own autoencoder +
  reconstruction + DEC." Read its exact cautions: `build_autoencoder`
  mechanically accepts any `input_dim` (confirmed: `Linear(input_dim, 500)`
  in, `Linear(500, input_dim)` out — no redesign needed), but swapping a
  32-D unit-norm vector in for the original 2,816-D one makes the 128-D
  latent **overcomplete** (128 > 32) and changes the reconstruction loss's
  effective scale, so `lambda_R=1` cannot be assumed equivalent without
  checking. **Log and report actual reconstruction error, KL magnitude,
  and center occupancy — do not just assume the recipe transfers.**
- `../20260922_percept_topic_pipeline/run_percept_stage1_faithful_recipe_pilot.py`,
  in full — reuse its own `pretrain_autoencoder`, `initialize_cluster_centers`,
  `train_dec_until_stable`, `evaluate_assignments` (these are this pilot's
  *own* seed-threaded versions of the mechanism, already used correctly by
  tonight's `run_percept_stage1_faithful_recipe_snapshot_pilot.py` — reuse
  the same sibling-import pattern that script uses, not a fresh
  reimplementation) and its `VARIANTS`, `N_INITIAL_CLUSTERS=100`,
  `N_SURVIVING_CLUSTERS=67`, `EMOTION_PARETO_BAR`, `GENRE_PARETO_BAR`
  constants. Reuse `base.build_autoencoder`, `base.soft_assignments`,
  `base.prune_centers` from `run_percept_stage1_pilot.py` (via `faithful.base`,
  exactly as `run_percept_stage1_faithful_recipe_snapshot_pilot.py` already
  does — read that script too, it is the closest existing template for
  this one's painting-alignment and snapshot-saving conventions).
- `attention_h1_embedding_snapshot.npz` — buddy's frozen 32-D
  `train_embedding_post`/`heldout_embedding_post` is this pilot's new input
  `h`, replacing PercepT's usual `fused_embeddings(...)` call entirely.
  Confirm shape `(61402, 32)` / `(9365, 32)`.
- `run_attention_h1_embedding_snapshot_pilot.py`'s own report or the
  snapshot's saved `train_community_post`/`heldout_community_post` and
  emotion/genre arrays — this pilot's **baseline reference point (a)**:
  buddy's own Leiden labels scored in buddy's own 32-D embedding, at the
  *same seed 42* this new pilot uses (a single-seed baseline, since this
  pilot itself is single-seed screen-only — do not substitute the
  four-seed mean, cite the matching seed-42 numbers).
- `run_percept_stage1_faithful_recipe_pilot_report.md`'s Variant A
  held-out row — this pilot's **reference point for the original 2,816-D
  input**: emotion AMI 0.1092, genre AMI 0.3288, silhouette 0.5120,
  occupancy 50/67 below 1%, 0 minimum. Also note its final-epoch loss
  magnitudes (recon≈0.000080, KL≈0.391414 at epoch 154) as the baseline
  for this pilot's loss-scale audit.

## What to build

Write
`src/test/20260923_artelingo_buddy_analysis/run_percept_on_buddy_embedding_pilot.py`.

Run **Variant A only, seed 42** (a single-seed screen, matching this
investigation's established staged-gate discipline — no multi-seed stress
until a seed-42 result looks promising enough to justify one). Reuse
`faithful.pretrain_autoencoder`, `faithful.initialize_cluster_centers`,
`faithful.train_dec_until_stable`, `faithful.evaluate_assignments`
unmodified, called on buddy's 32-D `train_embedding_post`/
`heldout_embedding_post` as `train_h`/`heldout_h` in place of PercepT's
usual fused CLIP+affect vector. Painting alignment: obtain the canonical
painting order the same way `run_percept_stage1_faithful_recipe_snapshot_pilot.py`
does (via `pipeline.load_dedup_features()`/`heldout_pipeline.load_dedup_features()`),
verify it as a set against the buddy snapshot's own painting arrays, and
reindex buddy's embeddings/labels into that canonical order before
training — reuse that script's `assert_matching_paintings` pattern (or
import it) rather than reimplementing.

**Report all three evaluation points the memo specifies, using the same
seed-42 fit's resulting DEC labels throughout:**

(a) **Reference only** (no new computation beyond what's cited above):
buddy's own Leiden labels scored on buddy's own 32-D embedding at seed 42.

(b) **New DEC labels in the new 128-D latent**: standard PercepT-style
evaluation via `evaluate_assignments` — held-out emotion/genre AMI,
silhouette (128-D), and the standard 67-center occupancy diagnostic
(min/max/median/zero-count/below-1%/collapsed), exactly as
`run_percept_stage1_faithful_recipe_pilot.py` already reports these.

(c) **The same DEC hard labels, scored back in buddy's original 32-D
embedding space**: reuse the sampled-silhouette convention used throughout
tonight's buddy pilots (`np.random.default_rng(42)` draw of ≤6,000 points,
`silhouette_score(sample_size=min(4000, len(idx)), random_state=42)`),
applied to buddy's *original* 32-D `heldout_embedding_post` array using
the *DEC* hard labels from (b), not buddy's own Leiden labels. This tests
whether the DEC-induced partition is also sensible in the geometry
Attention-h1's own InfoNCE losses actually shaped, not only in the new
128-D latent PercepT's own autoencoder built for it.

**Loss-scale audit**: log and report, side by side with the original
2,816-D run's own final-epoch numbers cited above: this run's final
pretrain reconstruction loss, final DEC-phase reconstruction loss, final
DEC-phase KL loss, and their ratio. State plainly whether the ratio looks
similar in scale to the original run's ratio or has shifted substantially
— do not silently assume `lambda_R=1` behaves the same way here.

## Report

Write
`src/test/20260923_artelingo_buddy_analysis/percept_on_buddy_embedding_pilot_report.md`
with: a Method section explaining the input swap and the overcomplete-
latent caveat; the full pretrain and DEC training/collapse trajectories
(same format as `run_percept_stage1_faithful_recipe_pilot_report.md`); the
loss-scale audit; a combined table with all three evaluation points (a),
(b), (c) plus the original 2,816-D reference row, so a reader sees emotion
AMI, genre AMI, silhouette, and occupancy for all four side by side; and a
final verdict section stating plainly: does buddy's compressed 32-D
representation retain enough signal for PercepT's own successful recipe to
build separable, non-collapsed topics from it, does the resulting
partition beat buddy's own Leiden-based approach when judged in buddy's
own native embedding space (c), and does it approach the original 2,816-D
input's own numbers or fall well short. This investigation's established
convention is a blunt, numeric verdict, not a hedge.

Do not touch git, do not modify any other file in the repository.
