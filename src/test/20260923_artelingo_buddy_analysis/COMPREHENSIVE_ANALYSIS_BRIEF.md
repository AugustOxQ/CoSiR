# Brief: comprehensive statistical/visual analysis of buddy vs. PercepT, with examples

This task is implementation AND execution — this script only loads saved
snapshots and does CPU-side plotting/statistics (no training, no GPU
required), so you may run it yourself after writing it, unlike the
GPU-training pilots in this directory. Do NOT touch git.

## Context

Read, in full, before writing any code:
- `docs/reports/2026-09-26_artelingo_buddy_vs_percept_stage1_report.md` —
  the full investigation this analysis covers. Every number you plot or
  cite should be traceable to a report already in this directory or the
  snapshots below — do not invent or approximate numbers.
- `attention_h1_embedding_snapshot.npz` (buddy: `train_embedding_post`,
  `heldout_embedding_post`, `train_community_post`, `heldout_community_post`,
  train/heldout `emotion`/`genre`, `train_paintings`/`heldout_paintings`).
- `../20260922_percept_topic_pipeline/percept_stage1_faithful_recipe_snapshot.npz`
  (PercepT: `train_latent`/`heldout_latent` (128-D), `train_topic`/
  `heldout_topic`, emotion/genre, paintings).
- `attention_h1_baseline_seed_stress_pilot_report.md`,
  `attention_h1_noise_schedule_pilot_report.md`,
  `attention_h1_noise_schedule_pseudo_contrastive_stress_pilot_report.md`,
  `attention_h1_dec_hybrid_pilot_report.md`, `attention_h1_vmf_dec_hybrid_pilot_report.md`,
  `attention_h1_decoupled_cluster_head_pilot_report.md`,
  `attention_h1_decoupled_cluster_head_detached_pilot_report.md`,
  `reconstruction_anchored_cluster_head_pilot_report.md` — each has a
  4-seed (or seed-42) held-out emotion AMI / genre AMI / silhouette result;
  read each report's own "Held-out summary statistics" section for the
  per-seed values (not just the mean) so real per-seed scatter/error bars
  can be plotted, not just a bar with an invented error bar.
- `buddy_percept_matched_silhouette_audit_pilot_report.md` (occupancy
  histograms for both systems), `buddy_percept_downstream_probe_pilot_report.md`
  (the three-feature-set human-label probe results), `buddy_stage2_pilot_report.md`
  and `../20260922_percept_topic_pipeline/percept_stage2_pilot_report.md`
  (per-topic AUC tables for both systems).
- `/data/PDD/artelingo/ArtELingo/ArtELingo/Dataset/artelingo_release_lite.csv`
  — columns `art_style,emotion,language,painting,split,utterance`. This is
  the only place caption/utterance text lives in this environment; there
  are **no raw painting images anywhere in this environment** (confirmed
  by exhaustive search) — "good examples" must be presented as text
  (painting ID, style, caption, labels), not thumbnails. State this
  limitation plainly in the report rather than silently working around it.

## What to build

Write two files:
`src/test/20260923_artelingo_buddy_analysis/build_comprehensive_analysis.py`
(the script) and
`docs/reports/2026-09-27_comprehensive_analysis.md` (the report the script
writes, with a section on how to reproduce each figure). Save all figures
as PNGs under `docs/reports/assets/comprehensive_analysis/` (create the
directory), referenced from the report by relative path.

**Figures to produce** (matplotlib, no seaborn dependency needed, or
seaborn if already installed — check `pip show seaborn` first and fall
back to plain matplotlib if unavailable):

1. **Tournament bar chart**: held-out emotion AMI, genre AMI, and
   silhouette for every buddy variant tried tonight (baseline, noise+
   schedule, noise+schedule+pseudo-contrastive, Euclidean DEC, vMF DEC,
   decoupled un-detached, decoupled detached, reconstruction-anchored) plus
   PercepT's faithful recipe, as three grouped subplots (one per metric),
   with real per-seed scatter points overlaid on each bar (not just a mean
   with an invented error bar) wherever 4-seed data exists, and the
   Pareto bar drawn as a horizontal reference line on the AMI subplots.
2. **Occupancy histograms**: buddy's 19 train-community held-out counts
   vs. PercepT's 67 surviving-center held-out counts, as two side-by-side
   bar charts (log-scale y-axis if the PercepT one needs it given its
   near-zero counts), reusing the exact counts already in the matched-audit
   report's occupancy tables (read them from that report, do not
   recompute independently unless you must — if you do recompute, they
   must match that report's numbers exactly or you have a bug).
3. **Embedding visualization**: a 2D projection (UMAP if installed --
   check `pip show umap-learn` first --, otherwise PCA) of buddy's 32-D
   held-out embedding, three panels colored by (a) Leiden community, (b)
   majority emotion, (c) genre (only genre-labelled points, others grayed
   out). Do the same for PercepT's 128-D held-out latent if a 2D
   projection completes in reasonable time (a few minutes) -- if UMAP or
   even PCA is impractically slow at this scale, fall back to a random
   6,000-point sample (reuse the seed-42 `np.random.default_rng(42)`
   sampling convention already used everywhere in this directory) and say
   so in the report.
4. **Per-topic AUC comparison**: a grouped bar chart, buddy's 19 topics
   sorted by AUC descending vs. PercepT's 40 topics sorted by AUC
   descending (two separate panels, since topic counts differ and topic
   identity isn't comparable across systems), both with the 0.5
   random-baseline line drawn.
5. **Training curves**: content/affect recall vs. epoch for the plain
   Attention-h1 baseline (read the per-checkpoint trajectory from
   `attention_h1_baseline_seed_stress_pilot_report.md` if it logs one, or
   from `attention_h1_embedding_snapshot_pilot_report.md`/its underlying
   trajectory data if available), alongside PercepT's own DEC KL/
   reconstruction loss trajectory (from
   `percept_stage1_faithful_recipe_pilot_report.md`'s logged trajectory
   text) -- two curves on the same time axis is not meaningful since the
   losses are incomparable; use two side-by-side subplots instead, each
   with its own y-axis, captioned separately.

**Statistics**: for every 4-seed comparison above, compute and report the
per-seed values (already read from each report), the mean, the sample
standard deviation, and a 95% bootstrap confidence interval (resample the
4 seed values with replacement, 10,000 resamples, percentile method,
`np.random.default_rng(42)`) for each of emotion AMI, genre AMI, and
silhouette. State plainly that n=4 makes any CI very wide and should be
read as indicative, not as a rigorous significance claim -- do not
overstate precision that four seeds cannot support.

**Good examples**: using buddy's held-out Stage 2 mapper (retrain it
fresh at seed 42 exactly as `run_buddy_stage2_pilot.py` does -- reuse that
script's functions via sibling import rather than reimplementing) and
PercepT's own held-out per-topic scores if reproducible from its own
snapshot, select: (a) the 5 held-out paintings where buddy's Stage 2
mapper's top-1 predicted topic corresponds to a topic with high held-out
per-topic AUC (i.e., a topic the mapper is good at) AND whose true emotion
matches the topic's dominant emotion label, (b) 3 held-out paintings where
buddy and a hypothetical PercepT-style topic assignment would clearly
disagree (pick paintings whose buddy topic's dominant emotion differs
starkly from their own true emotion, illustrating a miss). For each
example, look up its `art_style`, `emotion`, and one `utterance` (its
first caption row) from the CSV by painting ID, and tabulate: painting ID,
art style, true emotion, true genre (if any), assigned buddy topic id,
that topic's held-out AUC, and the caption text. Present as a Markdown
table. State explicitly in the report that these are illustrative, not a
random or representative sample -- they were selected to show the method
working, and a reader should treat them as qualitative color, not
evidence.

## Report structure

`docs/reports/2026-09-27_comprehensive_analysis.md`: a short intro stating
scope and the no-raw-images limitation, then one section per figure above
(image embedded via Markdown, one-paragraph caption stating exactly what
it shows and citing the source report(s) for its numbers), a "Statistics"
section with the bootstrap CI table, a "Qualitative examples" section with
the two example tables, and a short "Reproduce" section naming the script
and confirming it was actually run (not just written) to produce these
exact artifacts.

Do not touch git, do not modify any other file in the repository.
