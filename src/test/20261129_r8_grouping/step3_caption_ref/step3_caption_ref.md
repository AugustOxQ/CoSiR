# Step 3: how much of the told-true-style gain is lost to poor caption placement

Date 2026-10-12 (Amsterdam). Seed 42, 4,096 episodes per pair. Descriptive; no verdict code.

**Answer.** With each caption placed by its own painting's image head (P), the told margin on style x genre rises from
+3.92 (A, round 7's K1) to +10.87 R@1 points. The one-hot true-style ceiling T* gave +21.74. So better placement would
recover about half of the T* margin. P is a privileged reference, not a method: a caption borrows its own painting's
image posterior, which no real caption has (see Doubts).

## What ran

- `r8_caption_ref.py T` and `r8_caption_ref.py R0` (each about 1 minute, CPU). Round 7 modules are imported from the
  round 7 folder; round 7's `results/` is only read.
- A = `F.told` as round 7 ran it. P = the same call with each row's caption posterior set to that row's image
  posterior (`posteriors_override`). The told scorer and its matched counterpart are built from the same posterior
  stack, so both see the replacement (C2).
- Nothing was refit. The told route has no reader, no gate threshold and no head fitted on caption posteriors. The
  only fitted part is the 224-cell choice (cross-fit by parity half on the 12,288 episodes), which `F.told` redoes
  for P, for the scorer and for the counterpart. That is the faithful version. As a check I also scored P with A's
  cells kept (last rows of the table).
- The image posterior is a function of the painting: within a painting the largest difference is 3.1e-6 and the argmax
  never differs.

## Reproduction of A

- All 15 saved arrays in `arrays_ceiling.npz` (per-anchor metrics, cells, gates, sigma) are exactly equal.
- `ceiling_verdict.json`: Sum(4d) 643, point 3.9246, interval [3.3484, 4.4928], CLEAR; the rerun is identical.
- The margins in `descriptive_ceiling.json` (three pairs and pooled) are identical.
- R0: A equals the spike's told A1 file (`sp_cand_told_A1_224.npz`) exactly (fused R@1 and margin).

## Table: told margin over the matched counterpart (R@1 points, 95% interval)

| Grouping | Version | style x genre | emotion x style | emotion x genre | Sum(4d) style x genre |
|---|---|---|---|---|---|
| T | A (round 7) | +3.92 [+3.35, +4.49] | +2.78 [+2.18, +3.37] | +3.09 [+2.52, +3.66] | 643 |
| T | P | +10.87 [+10.07, +11.69] | +7.28 [+6.57, +7.98] | +3.60 [+3.05, +4.18] | 1781 |
| T | P, A's cells kept | +12.90 [+12.02, +13.77] | +9.45 [+8.68, +10.21] | +3.09 [+2.52, +3.66] | 2113 |
| T | T* (round 7) | +21.74 [+21.13, +22.35] | | | |
| R0 | A | +0.13 [-0.35, +0.60] | +2.53 [+1.97, +3.07] | +2.95 [+2.41, +3.50] | 22 |
| R0 | P | +0.59 [-0.11, +1.27] | +3.58 [+2.96, +4.19] | +3.33 [+2.79, +3.87] | 97 |
| R0 | P, A's cells kept | +1.58 [+0.88, +2.26] | +4.44 [+3.77, +5.10] | +2.95 [+2.41, +3.50] | 259 |

Style x genre fused and counterpart R@1 (points), re-selected cells: T A 25.87 and 21.95; T P 38.79 and 27.92;
R0 A 21.82 and 21.69; R0 P 25.85 and 25.26.

## Caption placement against the true style (T)

| Measure | Caption head (A) | Image head of own painting (P) |
|---|---|---|
| Round 7 check rows (10,000) | 26.49% | 63.72% |
| Seed-42 selection rows (32,413) | 25.43% | 60.92% |
| Rows used in the episodes (23,360) | 27.09% | 61.69% |
| Largest single class (chance reference, selection rows) | 15.97% | 15.97% |

The caption head and the image head agree on 29.4% of selection rows. For R0 there is no true code; they agree on
41.4%.

## Deviations and doubts

- P is a privileged reference, not a self-match (corrected by the controller, 2026-10-12 18:33). The direction term
  is s = p(query) . p(candidate) (`probe_dots`), and every row of an episode is from a different painting
  (`build_aspect_episodes` draws with `used`; `validate_aspect_episodes` asserts it). So under P a caption query's
  term is its own painting's image posterior dotted with *another* painting's image posterior, never with itself.
  What P adds is the painting's visual information, which a real caption does not carry; so +10.87 is a ceiling on
  caption placement, not a reachable number. The counterpart's rise (21.95 to 27.92 on style x genre) is the same
  sharper style agreement, used without the condition. (The worker's first draft read this as a self-match.)
- The 224-cell choice is shared across all three pairs, so re-selection moves emotion x genre too (+3.09 to +3.60), a
  pair where T is not used. The counterpart cells change from (39, 13) to (7, 7). With A's cells kept that pair is
  unchanged by construction; this is why the third row is shown.
- Raw CSD: the style x genre margin stays inside noise under P with re-selected cells (+0.59, interval includes 0).
  Only the cells-kept variant clears 0 (+1.58). Cheap (about 1 minute), so it ran.
- No true style code exists for R0, so no placement accuracy against truth.
- The comparators B1 and BT of the descriptive pass were not recomputed under P.

## Files

`r8_caption_ref.py`, `results_T.json`, `results_R0.json`, `results.json` (both merged), `run_T.log`, `run_R0.log`, this page.
