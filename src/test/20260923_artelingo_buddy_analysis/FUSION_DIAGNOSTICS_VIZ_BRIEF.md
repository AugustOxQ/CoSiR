# Brief: fusion-investigation diagnostics and visualizations

Write a chart-generation script
`docs/reports/assets/build_2026-09-22_fusion_diagnostics_charts.py`
(matching this repo's existing convention of `docs/reports/assets/build_*.py`
scripts that generate PNGs for a report) and a companion markdown report
`docs/reports/2026-09-22_artelingo_fusion_diagnostics_report.md`. This is a
documentation/visualization task — no model training, no GPU needed. The
chart script should be runnable locally (matplotlib only; add `pandas` only
if genuinely helpful, no other new dependencies).

## Why this exists

The existing `docs/reports/2026-09-22_artelingo_fusion_mechanism_investigation.md`
is a text-and-tables narrative report. This is its deeper, visual companion:
statistics and charts across every fusion mechanism tried, so patterns that
are hard to see in markdown tables (the shape of the trade-off frontier, the
training dynamics that separate a real compromise from collapse) are visible
directly. Cross-link the two reports at the top of each (add one sentence
to the narrative report's own top, after reading it, pointing to this new
diagnostics report, and vice versa).

## Critical data-fidelity rule

**Do not hardcode any numeric result from memory or from this brief's prose.**
Every number that appears in a chart or table must be parsed directly out of
the actual committed markdown table in its source report file (listed
below), by reading the file and extracting values from its markdown tables
(a small, permissive parser — split on `|`, strip whitespace, skip
separator rows — is sufficient; do not use a heavy markdown library
dependency for this). This guarantees every chart is a faithful
reproduction of already-verified source data, and stays correct
automatically if a source report is ever corrected. If a number this brief
mentions in prose cannot be found in its cited source file's tables, treat
that as a bug to fix (find the right source), not as license to hardcode
the number.

## Source reports (read each one's tables in full)

1. `src/test/20260923_artelingo_buddy_analysis/single_modality_pilot_report.md` — content-only, GoEmotions-only references.
2. `src/test/20260923_artelingo_buddy_analysis/affect_pilot_report.md` — early fusion weight sweep (5 rows: weight 0.0/0.5/1.0/2.0/4.0).
3. `src/test/20260923_artelingo_buddy_analysis/dec_pilot_v2_report.md` — GoEmotions-only DEC reference.
4. `src/test/20260923_artelingo_buddy_analysis/late_fusion_pilot_report.md` — late union/intersection.
5. `src/test/20260923_artelingo_buddy_analysis/hierarchical_refinement_pilot_report.md` — hierarchical + Control A + Control B.
6. `src/test/20260923_artelingo_buddy_analysis/cca_audit_pilot_report.md` — CCA held-out canonical correlations (10 components) and edge-retrieval recall.
7. `src/test/20260923_artelingo_buddy_analysis/learned_student_stage1_pilot_report.md` — Stage 1 full checkpoint trajectory (epoch 0..200) and final train/held-out comparison.
8. `src/test/20260923_artelingo_buddy_analysis/learned_student_stage2_pilot_report.md` — Stage 2 full checkpoint trajectory and final train/held-out comparison.
9. `src/test/20260923_artelingo_buddy_analysis/learned_student_weight_sweep_pilot_report.md` — 4-weight x 2-split results table.

## Charts to generate (save each as its own PNG, descriptive filename, 150 dpi, consistent style: one shared matplotlib style block, e.g. a clean sans-serif font, gridlines at low alpha, a consistent color per method-family used identically across every chart)

1. **`fusion_pareto_frontier.png`** — scatter plot, x-axis emotion AMI, y-axis
   genre AMI. One point per method/variant, parsed from every source table
   above (include: content-only, GoEmotions-only Leiden, GoEmotions-only
   DEC, all 5 early-fusion weight points connected by a thin line in weight
   order, late union, late intersection, hierarchical + its two controls,
   Stage 1 train AND held-out, Stage 2 train AND held-out, all 4 weight-sweep
   points at both splits connected by a thin line in weight order). Color by
   method family (single-signal reference / early fusion / late fusion /
   hierarchical / learned student). Label the Stage 1 train point and Stage 1
   held-out point explicitly (they are the empirical frontier). Draw a
   dashed reference line or shaded region marking "Pareto-dominated by Stage
   1" (points with both emotion AMI <= Stage 1's and genre AMI <= Stage 1's
   train values) versus not, so the frontier is visually obvious, not just
   implied by point position.

2. **`fusion_method_bars.png`** — grouped bar chart, one group per
   method (not every weight-sweep point — just the headline methods:
   content-only, GoEmotions-only, early fusion best, late union, late
   intersection, hierarchical, Stage 1, Stage 2), two bars per group
   (emotion AMI, genre AMI), train-split values (the values every method
   reports as its primary number).

3. **`stage1_checkpoint_trajectory.png`** — 2x2 subplot grid from Stage 1's
   full checkpoint table: (a) content and affect held-out recall vs. epoch,
   two lines; (b) content and affect loss vs. epoch, two lines; (c) content
   gradient share vs. epoch, one line with a horizontal reference line at
   0.5; (d) gate mean vs. epoch with a shaded band showing gate mean ±
   gate std at each checkpoint.

4. **`stage1_vs_stage2_trajectory.png`** — 1x2 subplot: (a) content held-out
   recall vs. epoch for both stages overlaid (two lines, clearly
   legended); (b) affect held-out recall vs. epoch for both stages
   overlaid. This is the chart that makes the "MLP leans further toward
   affect at content's expense" finding visible directly.

5. **`weight_sweep_tradeoff.png`** — line plot, x-axis content_weight
   (1.0/1.5/2.0/3.0), two y-axes or two subplots: emotion AMI and genre AMI
   each vs. content_weight, train and held-out as separate lines (4 lines
   total, 2 per metric), showing the monotonic trade-off directly.

6. **`hierarchical_controls_bars.png`** — grouped bar chart: hierarchical vs.
   Control A vs. Control B, emotion AMI and genre AMI as two bars per
   group. This is the chart that makes the "genre loss is mostly a
   granularity artifact, not affect-specific" finding visible — annotate
   directly on the chart (a text annotation or bracket) that Control A used
   the same split sizes with no real signal.

7. **`cca_canonical_correlations.png`** — bar chart, one bar per canonical
   component (1..10) from the CCA audit's held-out correlation, with each
   bar's corresponding null-distribution 95th-percentile shown as an
   overlaid marker or thin error-bar-style tick, so the real-vs-null gap is
   visible per component.

## Report structure

`docs/reports/2026-09-22_artelingo_fusion_diagnostics_report.md`:

- One-paragraph intro cross-linking the narrative stage report.
- One section per chart: embed the image (relative path
  `assets/<filename>.png`), then 2-4 sentences of plain-language discussion
  of what the chart shows — referencing the same honest characterizations
  already established in the narrative report (e.g., late union's genre
  cost is from specific community merges, not smooth dilution; Stage 1's
  "Collapsed" label is better read as a healthy compromise given gradient
  share and gate saturation; hierarchical's genre loss is mostly
  granularity, not affect). Do not introduce new claims not already
  established in the narrative report — this report's job is to make
  existing, already-verified findings visually legible, not to draw new
  conclusions.
- A closing note stating explicitly that every number plotted was parsed
  from its source report's own tables (name the parsing approach briefly),
  not hand-transcribed, and pointing back to the narrative report for full
  methodology and caveats on each method.

## Verification requirement

After generating the charts, re-open each PNG's underlying data (print the
parsed values the script used, or re-derive them) and manually cross-check
at least three spot values per chart against the actual source markdown
file content, in your own final summary to the user, so the fidelity claim
is verified, not just asserted.
