# Brief: wire buddy's frozen Stage 1 topics into a PercepT-Stage-2-comparable image-only mapper

Write one new script. Do NOT run it — execution happens separately, on
GPU, outside this task.

## Context

Read, in full, before writing any code:
- `../20260922_percept_topic_pipeline/run_percept_stage2_pilot.py`, in
  full — the pilot this one is structurally comparable to. Read
  particularly its `AttentionPoolingMapper`, `load_patch_features`,
  `evaluate_auc`, `auc_summary`, and `main()`'s target-construction and
  train-marginal-baseline logic (the last ~40 lines of `main()`, after
  Stage 1 is re-fit): it computes `train_marginals =
  train_targets.float().mean(dim=0)`, broadcasts it to the held-out
  target's shape, and scores it with the exact same `evaluate_auc` call
  as the trained mapper — reuse this baseline convention unmodified, it
  is the reference every "does the mapper add anything" claim in this
  project is measured against.
- `../20260922_percept_topic_pipeline/percept_stage2_pilot_report.md` —
  PercepT's own headline numbers to cite for comparison: macro AUC
  **0.5690** (min 0.3618, median 0.5536, max 0.8744, 0 skipped topics),
  train-marginal baseline macro AUC exactly **0.5000**. Also note its own
  "Frozen multi-label target statistics" table: mean/median/max labels are
  all **1.000/1.000/1**, 0.00% multi-labeled — PercepT's own target,
  despite being built by a multi-hot threshold rule, turned out to be
  purely single-label in practice. This makes buddy's genuinely
  single-label hard partition (Leiden communities) a structurally faithful
  comparison, not an apples-to-oranges one — state this explicitly in the
  new report.
- `run_buddy_percept_downstream_probe_pilot.py` — the closest prior buddy
  pilot (reuse its painting-alignment pattern: canonical order from
  `pipeline.load_dedup_features()`, reindex snapshot arrays by painting
  ID, assert exact set equality first). That pilot trained a similar
  mapper with **single-label cross-entropy** for a downstream human-label
  probe; this pilot is different in purpose and must instead use
  **`BCEWithLogitsLoss` against a one-hot-shaped target**, literally
  matching `run_percept_stage2_pilot.py`'s own training convention, so the
  two systems' Stage 2 reports are comparable to each other, not just
  internally consistent.
- `attention_h1_embedding_snapshot.npz` (buddy's frozen 32-D
  `train_embedding_post`, `train_community_post` — 19 contiguous labels
  0-18) and `run_heldout_label_transfer_pilot.py`'s
  `assign_to_train_communities` (k=20, reuse unmodified) — this is
  buddy's Stage 1, exactly as validated and recommended earlier tonight.

## What to build

Write
`src/test/20260923_artelingo_buddy_analysis/run_buddy_stage2_pilot.py`.

1. Load buddy's frozen train communities and held-out communities (via
   `assign_to_train_communities`, k=20), reindexed to the canonical
   patch-feature painting order (fresh `pipeline.load_dedup_features()`/
   `heldout_pipeline.load_dedup_features()` calls, exactly as the
   downstream-probe pilot does it — reuse that alignment pattern, do not
   reinvent it). Assert the buddy snapshot's painting-ID set matches the
   canonical order exactly before reindexing, the same discipline every
   pilot tonight has used.
2. Build one-hot targets: `torch.zeros(N, 19)` with a 1 at each painting's
   community index, for both train and held-out. Log the same "frozen
   multi-label target statistics" table PercepT's own report logs (mean/
   median/max labels, fraction multi-labeled) — it will trivially show
   1.000/1.000/1/0.00%, and the report should say so plainly rather than
   treat it as a surprise (buddy's communities are a hard partition by
   construction).
3. Reuse `AttentionPoolingMapper` (with `n_topics=19`), `load_patch_features`,
   and the patch-feature path constants **unmodified** via sibling import.
   Train with `BCEWithLogitsLoss`, `MAPPER_LEARNING_RATE`, `MAPPER_EPOCHS`
   reused unmodified from the sibling — matching PercepT's own Stage 2
   training convention exactly, not the cross-entropy variant from the
   downstream-probe pilot.
4. Evaluate held-out per-topic AUC via `evaluate_auc`/`auc_summary` reused
   unmodified, against buddy's held-out one-hot targets. Compute the same
   train-marginal-frequency baseline (mean of train one-hot targets,
   broadcast to held-out shape, scored the same way) as PercepT's own
   pilot does, and check `skipped_topics` matches between the mapper and
   baseline scoring exactly as the original does.

## Report

Write
`src/test/20260923_artelingo_buddy_analysis/buddy_stage2_pilot_report.md`,
structured to mirror `percept_stage2_pilot_report.md`'s sections (target
statistics, mapper training loss trajectory, full per-topic AUC table,
macro/min/median/max summary for both the mapper and the train-marginal
baseline) plus one new section: a direct side-by-side comparison against
PercepT's own cited Stage 2 numbers (macro AUC 0.5690, baseline 0.5000,
min/median/max as cited above). State plainly, using this investigation's
established blunt-verdict convention: does buddy's image-only Stage 2
mapper beat its own train-marginal baseline by a meaningful margin
(reuse PercepT's own predeclared 0.01 practical-margin convention), and
how does buddy's macro AUC compare numerically to PercepT's 0.5690 — is
buddy's Stage 2 topic space at least as predictable from images alone as
PercepT's, worse, or better, stated as a plain number comparison, not a
hedge.

Do not touch git, do not modify any other file in the repository.
