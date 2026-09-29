# Brief: deep Stage 1 / Stage 2 analysis (buddy vs. fixed PercepT)

## Context and goal

Tonight's investigation compares buddy (this project's own InfoNCE+Leiden
topic-formation method) against PercepT (a DEC-based baseline) on ArtELingo.
Two real PercepT implementation bugs were found (independent adversarial
review, verified against the actual paper arXiv:2606.03345) and fixed:
backwards center-pruning direction, and a reconstruction loss ~2,816x too
weak. Fixed, PercepT's Stage 2 macro AUC is 0.5925 vs. buddy's 0.5978 — a
near-tie (+0.0053), not the originally reported +0.0288 margin. Full
derivation:
[`docs/reports/2026-09-26_artelingo_buddy_vs_percept_stage1_report.md`](../../../docs/reports/2026-09-26_artelingo_buddy_vs_percept_stage1_report.md)
§6b.

So far all analysis has been top-line numbers (AMI, silhouette, macro AUC,
occupancy counts). **This task's job is to go deeper**: characterize what
the two systems' topics actually contain, why Stage 2 AUC is what it is
per-topic, whether the two systems' topic structures agree with each
other, and whether prediction errors cluster around anything meaningful
(genre, emotion). The goal is to surface concrete, evidenced ideas for
improving buddy's own Stage 2 AUC further — this analysis feeds directly
into a follow-up implementation round, so findings must be specific enough
to act on, not just descriptive.

Do this work in a new directory: `src/test/20260927_deep_stage_analysis/`.
Two parts, run in order.

## Part A — produce a fixed-PercepT snapshot (currently missing)

Buddy already has a rich snapshot with everything needed for analysis:
`src/test/20260923_artelingo_buddy_analysis/attention_h1_embedding_snapshot.npz`,
fields: `train_paintings, train_embedding_pre, train_embedding_post,
train_community_pre, train_community_post, train_emotion, train_genre,
heldout_paintings, heldout_embedding_pre, heldout_embedding_post,
heldout_community_pre, heldout_community_post, heldout_emotion,
heldout_genre, seed`. `*_post` is buddy's final trained embedding/community
assignment (use this one for analysis; `*_pre` is a pre-training reference,
ignore it here).

PercepT has no equivalent snapshot for the FIXED (bug-corrected) K=60/40
pipeline — the existing
`src/test/20260922_percept_topic_pipeline/percept_stage1_faithful_recipe_snapshot.npz`
is from the OLD BUGGY implementation and must not be used. Read
`src/test/20260922_percept_topic_pipeline/run_percept_stage2_fixed_pilot.py`
in full (this is the corrected K=60/40 Stage 1 + Stage 2 pipeline that
produced the 0.5925 number) and write a new script
`run_percept_fixed_snapshot_pilot.py` in this task's own directory
(`src/test/20260927_deep_stage_analysis/`) that runs the exact same
pipeline (same seed 42, same fixes, do not change any hyperparameter or
fix) but additionally saves a snapshot
`percept_fixed_snapshot.npz` (in this task's own directory) with fields
mirroring buddy's naming convention as closely as sensible:
`train_paintings, train_embedding (128-D Z), train_topic (hard argmax over
the 40 surviving centers), train_emotion, train_genre, heldout_paintings,
heldout_embedding, heldout_topic, heldout_emotion, heldout_genre,
heldout_stage2_scores (the trained mapper's held-out sigmoid outputs,
shape [n_heldout, 40]), heldout_stage2_targets (the frozen multi-hot
targets used for AUC, shape [n_heldout, 40]), per_topic_auc (dict/array of
the 40 per-topic AUCs, skip list included), seed`. Do not retrain anything
differently from `run_percept_stage2_fixed_pilot.py` — this is purely an
instrumentation pass over the identical pipeline, reusing its functions by
import where possible rather than copy-pasting logic that could drift.

Run Part A once it's written; confirm the resulting macro AUC matches
0.5925 (±0.001) as a sanity check before proceeding to Part B. If it
doesn't match, stop and report the discrepancy rather than continuing.

## Part B — the deep analysis itself

New script `run_deep_stage_analysis.py` in the same directory, consuming
both snapshots (buddy's existing one, and the new
`percept_fixed_snapshot.npz` from Part A). Also needs buddy's own Stage 2
predictions, which were never persisted — read
`src/test/20260923_artelingo_buddy_analysis/run_buddy_stage2_pilot.py` in
full and reuse its exact mapper/training/prediction code (import, don't
reimplement) to retrain buddy's Stage 2 mapper once more (seed 42, same as
that pilot) and capture held-out sigmoid scores + multi-hot targets +
per-topic AUC in memory (or save them to a small buddy-side snapshot
addition in this task's own directory — do not modify buddy's existing
snapshot file or pilot script).

Caption text for topic characterization comes from
`/data/PDD/artelingo/ArtELingo/ArtELingo/Dataset/artelingo_release_lite.csv`
(columns: `art_style,emotion,language,painting,split,utterance`) — filter
to `language == "english"` (case-insensitive), matching the convention
already used by `run_percept_stage1_pilot.py`'s `load_dedup_features()`
(read that function for the exact filtering/dedup logic and reuse it
rather than re-deriving it from scratch). A painting has multiple English
caption rows (multiple annotators); for representative-caption sampling,
any 3-5 per painting/topic is fine, seed-42 sampled.

Produce these analyses, each with its own clearly labeled section in the
final report:

1. **Topic characterization table**, one row per surviving topic, for
   BOTH systems separately (buddy: 19 communities; PercepT: up to 40
   topics, using held-out assignments): topic size, majority emotion label
   + its share (%), majority genre label + its share (% of the genre-
   labelled subset only, since genre coverage is much smaller), label
   entropy (Shannon entropy of the emotion distribution within that
   topic — lower means the topic is more emotionally homogeneous), and 3
   representative captions (seed-42 sampled from that topic's held-out
   members). This is the single most important deliverable — it turns
   "topic 17" into something a reader can understand semantically.

2. **Per-topic AUC vs. per-topic occupancy**, for both systems: a
   scatter plot (topic size on x, per-topic AUC on y) plus the Pearson and
   Spearman correlation coefficient. State plainly whether small/collapsed
   topics are dragging down macro AUC disproportionately, or whether AUC
   is roughly independent of topic size.

3. **Cross-system topic alignment**: both systems assign a topic label to
   the SAME 9,365 held-out paintings (confirm the painting ID sets match
   exactly before proceeding; if they don't, stop and report the
   discrepancy). Build a buddy-community x PercepT-topic contingency table
   on held-out data, compute normalized mutual information (NMI) between
   the two label sets, and report the single most "concentrated" pairing
   in each direction (e.g., "buddy community 4 maps most heavily onto
   PercepT topics {2, 9, 14}" and vice versa). This tells us whether the
   two independently-derived topic structures agree at all, agree
   partially (buddy's coarser topics splitting into several PercepT
   topics or vice versa), or are essentially uncorrelated.

4. **Error analysis by emotion** (use the full 9,365 held-out set, not the
   159-painting genre overlap, for statistical power — do the same for
   genre too but explicitly flag the genre analysis as thin-sample/
   indicative only): for each system, is the Stage 2 mapper's per-painting
   top-1 prediction accuracy (predicted topic with highest score == the
   painting's true hard-assigned topic) systematically higher or lower
   for certain emotion categories? Report a table: emotion category x
   [buddy top-1 accuracy, PercepT top-1 accuracy, support count].

5. **Synthesis — concrete next ideas**: given findings 1-4, write a short
   (under 500 words) section proposing specific, evidenced ideas for
   improving buddy's own Stage 2 AUC further (not PercepT's). This directly
   feeds a follow-up implementation round, so ideas must cite which
   specific finding above motivates them (e.g., "topic X's low AUC
   correlates with Y, suggesting Z"), not be generic suggestions
   disconnected from this analysis's own evidence.

## Output

- `percept_fixed_snapshot.npz` (Part A)
- `run_percept_fixed_snapshot_pilot.py`, `run_deep_stage_analysis.py`
  (scripts)
- `deep_stage_analysis_report.md` — the full report, all 5 analyses above,
  plots saved under `assets/` in this same directory and referenced with
  relative paths, every number traceable to a specific computation in the
  script (no hand-waved figures)

## Explicit non-goals / constraints

- Do not modify any existing shared/frozen file (`run_percept_stage1_pilot.py`,
  `run_percept_stage1_cluster_count_sweep_pilot.py`,
  `run_percept_stage2_fixed_pilot.py`, `run_buddy_stage2_pilot.py`,
  buddy's existing snapshot, or any report file) — read-only imports only.
- No new training beyond what's needed to capture buddy's Stage 2
  predictions (a single seed-42 retrain, matching the existing pilot
  exactly) and the Part A PercepT snapshot instrumentation pass.
- Do not attempt to improve either system's numbers here — this is
  characterization and error analysis only. Improvement ideas belong in
  §5's synthesis as proposals, not as new experiments run in this task.
- Do not touch DAS6 — this is small-scale, local GPU/CPU is sufficient.
