# Brief: RedCaps Leiden resolution sweep (B1 follow-up diagnostic)

## Context

`b1_redcaps_single_teacher_pilot_report.md` (same directory) found that the
completely **untrained** "graph-only baseline" — Leiden run directly on the
raw train teacher union graph, zero training or embeddings involved — is
~99% occupancy-collapsed: 1,404 of 1,423 detected communities hold under 1%
of members. This happens before any student architecture, loss, or CLIP
feature is involved, so the cause cannot be anything about the learned
student. `docs/reports/2026-09-27_redcaps_b1_result_and_diagnosis.md`
diagnoses this as **Leiden's default modularity resolution behaving very
differently on RedCaps' graph structure than on ArtELingo's**, and proposes
a resolution-aware sweep as the direct test of that diagnosis. This brief
scopes that sweep. It is CPU-only, requires no retraining, and should take
a few minutes total.

## What "default resolution" means here

`src/conditional_buddy/prototype_seed.py::detect_communities` uses
`leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, seed=seed)`
— plain modularity, no resolution knob. `ModularityVertexPartition` is
equivalent to `RBConfigurationVertexPartition` at `resolution_parameter=1.0`.
**Do not modify `detect_communities`** (it's shared, frozen infrastructure
used elsewhere). Instead, write a new, local, standalone function in this
pilot's own script that takes a `resolution` argument and calls
`leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, resolution_parameter=resolution, seed=seed)`
on the same igraph `Graph` construction `detect_communities` uses (copy that
few lines of graph-building logic locally — don't import a private helper
that doesn't exist).

## What to build

New script in this same directory:
`src/test/20260927_redcaps_topic_formation/run_leiden_resolution_sweep_pilot.py`.

1. **Reuse the exact same train split and graph as B1** — do not resample.
   Load `b1_redcaps_single_teacher_pilot_split.npz` (already saved by B1) to
   recover the identical train/val/test index arrays (seed-42
   `default_rng(42).shuffle`, 120k/15k/15k). Rebuild the train-only raw
   teacher union graph via `redcaps_buddy.load_data` +
   `redcaps_buddy.build_graphs(data, K=30, device=...)` restricted to the
   train indices, exactly as B1 did (read `run_b1_redcaps_single_teacher_pilot.py`
   in full first to copy the exact graph-construction call, K value, and any
   filtering/repair steps B1 applied before Leiden — match it exactly so this
   is a true apples-to-apples resolution sweep against B1's own graph-only
   baseline row, not a subtly different graph).

2. **Sweep resolution_parameter over**: `[1.0, 0.5, 0.25, 0.1, 0.05, 0.01, 0.005]`.
   (1.0 reproduces B1's existing graph-only-baseline row — use it as an
   in-pilot sanity check: community count and below-1% count at
   resolution=1.0 should closely match B1's reported 1423 communities /
   1404 below 1%. If it doesn't match closely, stop and report the
   discrepancy rather than continuing the sweep — something about the graph
   reconstruction differs from B1 and needs to be found first.)

3. **At each resolution, report:**
   - number of communities
   - occupancy: min / median / max / count-below-1%
   - the same **community-level lift** metric B1 used for its graph-only
     baseline row (`subreddit_lift` over same-community validation pairs,
     transferred via the same cosine k=20 nearest-train-neighbor assignment
     B1's graph-only baseline used, with the same 2,000-point subsampling
     cap for oversized communities) — reuse `assign_to_train_communities`
     and B1's own transfer/lift code paths, don't reimplement them.

4. **Output**: a markdown report
   `run_leiden_resolution_sweep_pilot_report.md` in this directory, with one
   results table (resolution × [communities, min/median/max occupancy,
   below-1% count, community lift]), plus a short verdict paragraph:
   does any resolution recover a much smaller, healthier-occupancy
   partition (say, closer to ArtELingo's ~19-way split) while keeping
   community lift comparable to or better than the resolution=1.0 baseline?
   If yes, name the best resolution and its numbers. If no resolution
   both shrinks community count materially and keeps lift up, say so
   plainly — this would mean the fragmentation is not simply a resolution
   miscalibration and the next diagnostic step is different (e.g. graph
   density/degree distribution itself).

## Explicit non-goals (do not do these)

- No training, no embeddings, no student architecture — Leiden runs
  directly on the raw graph adjacency only, same as B1's graph-only
  baseline.
- No DAS6 dispatch — local GPU/CPU only, this is small and cheap.
- No modification to `detect_communities`, `redcaps_buddy.py`, or any other
  shared file — this is a fully standalone script in this pilot's own
  directory, importing but never editing shared functions.
- Do not proceed to B2 (patch-feature extraction) or any RedCaps scale-up —
  this pilot's only job is to answer the resolution question.

## Deliverable checklist

- [ ] `run_leiden_resolution_sweep_pilot.py` written and run to completion
- [ ] Sanity check at resolution=1.0 matches B1's existing graph-only row
      (or a clear discrepancy is reported and the sweep is halted)
- [ ] `run_leiden_resolution_sweep_pilot_report.md` written with the full
      table and an explicit verdict
