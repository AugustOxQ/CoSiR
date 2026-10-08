# Research loop figure log

## Round 1

Rendered `research_loop.png` at 200 dpi and `research_loop.svg`, then opened and inspected the PNG.

Defects seen:
- The experiment steps were positioned left to right while their edges used ports intended for a right-to-left sequence; several connections bent around nodes and had unclear arrowheads.
- The `new results` return edge crossed the `yes` edge and passed through the experiment cluster near its title.
- The `no: discuss again` label overlapped its return edge.
- The experiment cluster was displaced far to the right, leaving substantial empty space below the start and decision steps.
- The distorted layout made the smaller text less comfortable to read at slide scale.

Changes for the next round:
- Use Graphviz's fixed-position layout to place two aligned rows: decisions left to right, then experiments right to left.
- Give every visible edge an explicit path, with the discussion return above the decision cluster and the results return on the left.
- Position edge labels next to their paths with clear spacing.
- Center a compact legend beneath the experiment cluster and reduce empty space.

## Round 2

Rendered both formats using `/root/miniconda3/bin/dot -Kneato -n2`, with `-Gdpi=200` for the PNG, then opened and inspected the PNG.

Defects seen: **none**.

Changes: none needed after this inspection. The fixed positions, explicit edge paths, and adjacent labels resolved the defects from round 1.

## Update 2026-10-08: wrap-up step and decision lines

### Round 1

Rendered both formats using the DOT render commands, with the PNG at 200 dpi, then opened and inspected `research_loop.png`.

Defects seen:
- The experiment row and legend were too far right, leaving avoidable empty space below the left side.

Checks: the new housekeeping node was outside both clusters, had the agents’ grey fill and border with a dashed outline, and both new edges were present. The three new supporting lines fit their boxes. No overlapping or clipped text, crossings, misplaced edge labels, incorrect colours, or missing edges were seen.

Changes for the next round:
- Center the experiment cluster and legend beneath the overall figure.
- Route the `yes` edge through the gap between clusters and into step 6, clear of the experiment title.

### Round 2

Rendered both formats using `/root/miniconda3/bin/dot -Kneato -n2`, with `-Gdpi=200` for the PNG, then opened and inspected `research_loop.png`.

Defects seen: **none**.

The centered experiment row and legend use the lower space efficiently. All text remains readable and inside its boxes. The paths and their labels are clear of other edges, nodes, text, and cluster titles. The housekeeping node remains outside both clusters with the requested dashed grey border. No further changes were needed.

## Final state

- Final render: update round 2 (2026-10-08); no visible defects remain.
- PNG: 4094 × 1349 pixels, rendered at 200 dpi; landscape. Dimensions were read from the PNG header and checked against the SVG dimensions and Graphviz’s 200/72 scale.
- SVG: valid XML; all text uses DejaVu Sans and `#111827`.
- All 12 required edges are present. The direct `1 → 2` edge has been replaced by `1 → Wrap up and hand off → 2`, with `next chat` on the second edge. The `no: discuss again`, `yes`, and `new results` edges and labels remain present and clear.
- The wrap-up node belongs to neither cluster and uses the agents’ fill `#f3f4f6` and border `#6b7280`, with a dashed outline.
- Steps 1, 2, and 4 have the requested decision lines. Node labels use 14 pt; all supporting lines use 11 pt; edge labels use 12 pt; cluster titles use 17 pt.
- Existing labels, fonts, sizes, colours, cluster titles, and the three legend swatches are preserved.
- Structural checks verified the exact edge set, requested supporting text and sizes, dashed outline, palette, labels, and legend.
- Unresolved defects: **none**.

The DOT file includes both reproducible render commands. Fixed-position rendering requires `-Kneato -n2`.
