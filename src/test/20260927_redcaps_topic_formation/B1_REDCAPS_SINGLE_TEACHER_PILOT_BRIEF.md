# Brief: B1 — RedCaps single-teacher Stage 1 pilot (150k, local GPU, cheap-first gate)

Write the script(s) described below. Do NOT run them — execution happens
separately, on local GPU, outside this task. This is a brand-new pipeline
(first RedCaps application of tonight's buddy+Leiden method), so read
carefully and ask no code to write itself from assumption where the brief
gives you the exact function to reuse.

## Context

Read, in full, before writing any code:
- `docs/reports/2026-09-27_method_improvement_and_redcaps_brainstorm.md`,
  candidate B1 (and its "Exact transfer boundary" section, and open design
  point 1 about which graph anchors the subreddit-lift comparison) — this
  brief is B1's smallest-test description turned into an exact spec.
- `src/test/20260623_redcaps_buddy/redcaps_buddy.py`, in full — reuse
  `load_data`, `build_graphs`, `edges`, `subreddit_lift` **unmodified**.
  `load_data(storage_dir, annotation_path)` returns a `Data` dataclass
  (`img`, `txt`: L2-normalized CLIP features; `sample_ids`; `sub_id`;
  `sub_names`; `records`), positionally joined and already L2-normalized.
  Defaults point at the 150k scale.
- `src/conditional_buddy/buddy_graph.py` — reuse `mutual_knn`,
  `union_graph`, `ensure_min_degree` **unmodified**. `mutual_knn(features,
  K, device, ...)` and `union_graph(A_img, A_txt)` are exactly what
  `redcaps_buddy.build_graphs` already calls internally — you do not need
  to call them directly except to build the student's own post-training
  validation embedding graph (`build_graphs` only builds CLIP-feature
  graphs, not arbitrary-embedding graphs, so write a tiny 3-line wrapper
  that calls `mutual_knn`/`union_graph` directly on the student's
  validation embeddings instead of on raw CLIP features).
- `src/test/20260923_artelingo_buddy_analysis/run_learned_student_arch_sweep_pilot.py`
  — reuse `symmetric_infonce`, `sample_positive_pairs`, `detect_communities`,
  `upper_triangle_edges` **unmodified** via sibling import. These are
  already dataset-agnostic (they operate on embeddings/edge arrays, not
  ArtELingo-specific data), so importing this module is safe and does not
  pull in any ArtELingo-only assumption as long as you do not call its
  `LearnedStudent`/`main`/pipeline-loading functions.
- `run_heldout_label_transfer_pilot.py` — reuse `assign_to_train_communities`
  **unmodified**; it is already fully generic (embeddings + integer labels
  in, integer labels out).
- Confirm the exact 150k feature-store paths from `redcaps_buddy.py`'s own
  `STORAGE`/`ANNOT` module constants — do not hardcode different paths.

## Design (do not deviate without flagging it clearly in your final message)

**Split**: load the full 150k `Data` via `load_data()`. Build a stable,
subreddit-blind train/validation/test split by shuffling `np.arange(data.n)`
with `np.random.default_rng(42)` and cutting at roughly 120k/15k/15k. Save
the three index arrays in the output artifact (below) so the split is
reproducible and auditable. The test split is not touched anywhere in
this pilot — it exists only so a later, more committed pilot can use it
without re-splitting; do not evaluate on it here.

**Teacher graph**: build the union teacher graph via
`build_graphs(data_restricted_to_train, K=30, device=...)`, where
`data_restricted_to_train` is a `Data` instance (or an equivalent tuple)
containing only the train rows (`data.img[train_idx]`, `data.txt[train_idx]`,
etc., with `sub_id`/`sub_names` sliced consistently). Report
`subreddit_lift` on this **train-only teacher graph itself** first (both
its raw union, `E`, and after `ensure_min_degree` repair) as a sanity
check: this should be in the same ballpark as Experiment 16's own
published ~22-23x lift numbers (cite the exact figure from
`docs/reports/2026-09-01_buddy_k_scaling_stage_a.md` and compare) — if it
is wildly different, something is wrong with data loading or the split,
and you should say so plainly rather than proceeding silently.

**Student architecture** (write this new, small, in the pilot script —
there is no existing single-teacher two-token attention module to reuse):
a `RedCapsStudent(nn.Module)` with `proj_img = nn.Linear(512, 32)`,
`proj_txt = nn.Linear(512, 32)`, one-head `nn.MultiheadAttention`-based
fusion exactly mirroring `AttentionFusion` from the arch-sweep module
(read that class and replicate its structure for two 32-D tokens, with
its own `LayerNorm(32)` — do not import the ArtELingo `AttentionFusion`
class directly since it is defined inside a module you are only
selectively reusing pieces of; a small faithful copy here, with a comment
noting it mirrors that class, is acceptable). Forward pass takes
`(img_features, txt_features)` (raw 512-D CLIP, L2-normalized already by
`load_data`) and returns one 32-D, L2-normalized fused embedding — a
**single** output, unlike ArtELingo's content/affect dual output, since
there is only one teacher here.

**Architecture control**: a second, equally-capacity `RedCapsMeanStudent`
that mean-pools the same two 32-D projections instead of attention-fusing
them (no learned query/attention at all) — this isolates whether attention
does anything useful in this single-teacher setting, per the brainstorm
memo's own stated reasoning.

**Training**: one symmetric InfoNCE loss (reuse `symmetric_infonce`
unmodified) on positive pairs sampled from the **train-only union teacher
graph's edges** (reuse `upper_triangle_edges`/`sample_positive_pairs`
unmodified). Fixed Adam learning rate 1e-3 (no schedule — keep this pilot
minimal), a batch size matching this project's established convention
(`BATCH_SIZE` from the arch-sweep module, reused unmodified), and a
generous but bounded epoch ceiling (e.g. 200, `CHECKPOINT_EVERY=5`) with
the same recall-plateau stopping rule as the ArtELingo pilots
(`relative_improvement`/`PLATEAU_REL_IMPROVEMENT`/`PLATEAU_WINDOW`, reused
unmodified) — recall against the train teacher graph is the one
diagnostic available here (no second teacher to average against). Train
both the attention student and the mean-pool control, same seed (42),
same budget.

**Stage 1 evaluation** (train Leiden + validation transfer + subreddit
lift), for both students:
1. Compute each student's final train embeddings; partition via
   `detect_communities` (reused unmodified, same Leiden default as
   ArtELingo). Report community count and occupancy (min/max/median/
   below-1%, same convention as every ArtELingo pilot tonight).
2. Compute each student's validation embeddings; assign to the frozen
   train vocabulary via `assign_to_train_communities` (k=20, reused
   unmodified). Report transfer coverage (how many train communities
   appear in validation) and any degenerate-fallback count.
3. **Main result**: build a mutual-kNN graph directly on each student's
   validation embeddings (`mutual_knn(..., K=30, device=...)` then
   `union_graph` is not needed here — a single embedding space has only
   one modality, so build ONE mutual-kNN graph on the validation
   embeddings, not a union of two) and compute `subreddit_lift` on its
   edges, using the validation-restricted `Data` (same slicing pattern as
   train). This is the primary Stage 1 quality metric — does the trained
   embedding preserve or enhance subreddit-cohesive local structure on
   held-out data.
4. **Community-level lift** (the AMI-analog): for each student, build the
   edge list of all validation-point pairs sharing the same transferred
   Leiden community (this can be large for big communities — if any
   community exceeds 2,000 validation points, randomly subsample pairs
   within it to at most 2,000×2,000/2 pairs via `np.random.default_rng(42)`
   rather than enumerating a huge dense pair set, and say you did this)
   and compute `subreddit_lift` on those "same-community" pairs.
5. **Controls, at the same validation N and K=30**: (a) a mutual-kNN graph
   built directly on raw validation CLIP image features alone, (b) the
   same on raw validation CLIP text features alone, (c) a "graph-only"
   baseline — partition the **train teacher union graph itself** (no
   trained student at all) via `detect_communities`, transfer to
   validation via `assign_to_train_communities` using the raw train CLIP
   embeddings (image+text concatenated, or just image — pick one and
   state why) as the transfer-distance space, then compute both the
   validation mutual-kNN-graph lift and community-level lift for this
   no-training baseline exactly as in steps 3-4.

## Output

Write:
- `src/test/20260927_redcaps_topic_formation/run_b1_redcaps_single_teacher_pilot.py`
- Its report:
  `src/test/20260927_redcaps_topic_formation/b1_redcaps_single_teacher_pilot_report.md`,
  with: the split sizes and saved index-array path, the train-teacher-graph
  sanity-check lift (raw and repaired, compared against the cited
  Experiment 16 number), a combined table (attention student / mean-pool
  control / raw-image control / raw-text control / graph-only baseline ×
  {validation embedding-graph lift, community-level lift, Leiden
  occupancy, transfer coverage}), and a plain verdict: did the trained
  single-teacher student preserve or improve subreddit lift relative to
  every control, does attention help over mean-pooling, and is this
  encouraging enough to justify B2 (patch-feature Stage 2) and eventually
  300k/500k on DAS6 — or does it justify stopping here and diagnosing
  further before spending more compute. Save the train/val/test index
  arrays as a `.npz` alongside the report for reuse by any follow-up
  pilot.

Do not touch git, do not modify any other file in the repository.
