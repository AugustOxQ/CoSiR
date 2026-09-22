# Brief: late (graph-level) fusion of the content and affect buddy graphs

Write a new script
`src/test/20260923_artelingo_buddy_analysis/run_late_fusion_pilot.py`.
Do NOT run it — execution happens separately, on GPU, outside this task.

## Context and why this run matters

Every affect-fusion attempt so far has been **early fusion**: concatenate
GoEmotions features with CLIP-text features into one vector, then build a
single mutual-kNN graph in that combined space (`run_affect_pilot.py`). That
approach needed an affect weight of 4.0 before emotion AMI moved much
(0.0593→0.1160), and paid for it by collapsing genre AMI (0.4384→0.0867) —
a real trade-off, not a clean win (`affect_pilot_report.md`).

This pilot tests **late fusion** instead: build each view's mutual-kNN graph
*independently*, in its own native feature space, then combine the resulting
**edge sets** rather than the features. This is exactly the same combination
principle the codebase already uses for image vs. text
(`src.conditional_buddy.buddy_graph.union_graph`) — this pilot applies it to
(content graph, affect graph) instead of (image graph, text graph). The
motivation: each view gets an undiluted vote in its own metric space, so
there's no cross-modality distance-scale competition to tune a weight
against, and a union only *adds* edges on top of the existing content
structure rather than reshaping it — plausibly preserving genre much better
than concatenation did. Test this; do not assume the outcome.

Read `run_pipeline.py`, `run_affect_pilot.py`, and `run_single_modality_pilot.py`
in full first, for conventions and reuse. Match their `log()` timestamped
progress-printing convention throughout.

## Implementation

1. Import `run_pipeline.py`, `run_affect_pilot.py`, and
   `run_single_modality_pilot.py` as sibling library modules via the same
   `load_sibling_module()` / `importlib.util.spec_from_file_location`
   pattern used throughout this directory.

2. Load and dedup features via `pipeline.assert_extraction_complete()` then
   `pipeline.load_dedup_features()` to get `(paintings, img_nodes, txt_nodes,
   emotion_counts)`. Get `majority_emotion` via `pipeline.majority()`.

3. **Build the content graph** exactly as the existing baseline does: import
   `build_buddy_graphs` from `src.conditional_buddy.compute_buddies` (same
   import `run_pipeline.py` itself uses) and call it as
   `A_img, A_txt, E_content = build_buddy_graphs(img_nodes, txt_nodes,
   K=pipeline.K, alpha=pipeline.ALPHA, device=device,
   connect_components=True)`. `E_content` is the already-repaired,
   already-connected img+txt union buddy graph — the same graph every prior
   pilot's "affect_weight=0.0" / "CLIP img+txt UNION baseline" row used.

4. **Build the affect graph**: extract GoEmotions nodes via
   `affect_pilot.extract_affect_nodes(pipeline.TRAIN_JSON, paintings,
   device)`, then build its single-modality graph via
   `single_modality_pilot.build_single_modality_graph("GoEmotions-affect-only",
   affect_nodes, pipeline, affect_pilot, device,
   expected_nodes=len(paintings))`. This already L2-normalizes internally and
   applies the same `ensure_min_degree`/`ensure_connected` repairs as the
   content graph, giving a fully comparable, already-repaired `E_affect`.

5. **Late union**: `from src.conditional_buddy.buddy_graph import
   union_graph` — call `E_late_union = union_graph(E_content, E_affect)`
   directly. This function is already fully generic (a binarised sparse
   add), not image/text-specific in any way, so no modification is needed.
   Log the resulting edge count (`E_late_union.nnz`) and how it compares to
   `E_content.nnz` and `E_affect.nnz`.

6. **Late intersection** (secondary diagnostic, cheap — compute and report
   it too, do not skip it even if it looks degenerate): mirror the exact
   idiom `buddy_graph.mutual_knn` itself uses to binarise a `.multiply()`
   result —
   ```python
   E_late_intersection = E_content.multiply(E_affect).tocsr()
   E_late_intersection.data[:] = 1.0
   ```
   Log the fraction of nodes with degree 0 in this raw intersection before
   any repair (`np.diff(E_late_intersection.indptr) == 0`) — this is
   expected to be large (GoEmotions' kNN structure is nearly orthogonal to
   content structure; its own genre AMI is only 0.0396), and that number is
   itself a reportable finding, not a bug to fix. Then still repair it with
   `ensure_min_degree` and `ensure_connected` (same `img_feats`/`txt_feats`
   argument-reuse trick as `build_single_modality_graph`: pass the
   L2-normalized content+affect concat, or more simply reuse whichever
   single feature array is easiest to pass into both repair-function
   argument slots — img_nodes is fine, since repair only uses these features
   to pick which isolated node to bridge to, not to change graph semantics)
   and run Leiden on it anyway, exactly like every other graph in this
   pilot. Do not silently drop this variant even though a large pre-repair
   isolated fraction is expected.

7. Run `detect_communities` (from `src.conditional_buddy.prototype_seed`,
   seed=42) on `E_late_union` and on the repaired `E_late_intersection`
   separately.

8. Compute emotion/genre AMI and V-measure for both via
   `pipeline.external_metrics()` and `pipeline.load_genre_map()`, same
   pattern as `run_single_modality_pilot.py`'s `evaluate_graph()` (reuse that
   function directly by importing it — do not reimplement it).

## Report

Write `src/test/20260923_artelingo_buddy_analysis/late_fusion_pilot_report.md`:

- Explain the late-fusion concept in plain language up front (each view
  builds its own graph independently in its own space; graphs are then
  combined at the edge level, not the feature level), and contrast it
  explicitly with the early-fusion approach already tried.
- Report `E_content.nnz`, `E_affect.nnz`, `E_late_union.nnz`, and the raw
  (pre-repair) isolated-node fraction of the intersection variant, before
  the results table.
- A comparison table with these exact reference rows (hardcode) plus this
  run's two new rows:

  | signal | emotion AMI | emotion V-measure | genre AMI | genre V-measure |
  |---|---:|---:|---:|---:|
  | Content-only (CLIP img+txt union, reference) | 0.0593 | 0.0600 | 0.4384 | 0.4572 |
  | GoEmotions-affect-only (reference, single-modality ceiling) | 0.1180 | 0.1189 | 0.0396 | 0.0937 |
  | Early fusion, best point (weight=4.0, reference) | 0.1160 | 0.1166 | 0.0867 | 0.1237 |
  | Late fusion — union (this run) | *computed* | *computed* | *computed* | *computed* |
  | Late fusion — intersection (this run) | *computed* | *computed* | *computed* | *computed* |

- A conclusion paragraph evaluating the union variant against TWO
  predeclared bars, stated separately and explicitly (do not collapse them
  into one verdict):
  (a) the original affect-pilot bar (AMI > 0.09, at least 50% relative
  improvement over the 0.0593 content-only baseline, while retaining at
  least 80% of the 0.4384 genre AMI baseline, i.e. genre AMI >= 0.3507) —
  state whether late-union clears this, for an apples-to-apples comparison
  with the early-fusion pilot's own criterion.
  (b) the stricter DEC-pilot bar (emotion AMI > 0.177, a 50% relative
  improvement over GoEmotions' own 0.1180 single-modality ceiling) — state
  whether late-union clears this too, as a check against the best-known
  ceiling from any method so far.
  Then state plainly whether late fusion preserved genre structure better
  than early fusion did (direct numeric comparison against the 0.0867 early-
  fusion genre AMI figure).
- A short paragraph on the intersection variant: state its pre-repair
  isolated-node fraction plainly, and whether its post-repair metrics (if
  meaningfully different from the union's) suggest the intersection is
  informative or just degenerate/noise-dominated after heavy repair.

Print clear timestamped progress logs matching the other scripts' `log()`
format throughout.
