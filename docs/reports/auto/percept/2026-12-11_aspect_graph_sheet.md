# Aspect graph sheet, stages 0 and 1: the aspect graph alone keeps the best pair agreement, graphs that subtract content give the best heads, and no emotion or style head reaches the prompted view on its own side

> 2026-10-18 (Amsterdam), local GPU (extraction) and CPU; queue ended 18:20. The file name uses CoSiR's sequence
> date. Spec: [aspect graph sweep](../../../superpowers/specs/2026-10-18-aspect-graph-sweep-design.md) (draft;
> stages 0 and 1 started before approval on the user's word, D10). Decision rule: none (descriptive). Run log:
> [20261018_aspect_graph_sheet_log.md](../../../../src/test/20261018_aspect_graph_sheet/20261018_aspect_graph_sheet_log.md).
> Train side only; labels used only to score.

## Results brief

**What we tried.** Stage 0 rebuilt the pilot's graphs. Stage 1 built 21 graphs per aspect (emotion, style, genre):
the aspect graph alone, the aspect graph pruned by content, fused with content or with the other side, and guided by
the content overlap. We scored each graph on how often its pairs share a label (lift). Then we trained a small
cross-modal head on five or six graphs per aspect and clustered its outputs.
[Spec](../../../superpowers/specs/2026-10-18-aspect-graph-sweep-design.md).

**Outcome against the bar.** No bar (exploration). Stage 0 passed: the pilot's graphs and lifts matched exactly.
Lift is on 13,400 training paintings; AMI (k = 20) on 5,000 evaluation paintings. "Target AMI" is the mean of the
image-side and caption-side AMI for the aspect's own label.
- Lift: the aspect graph alone is the best of 21 graphs for emotion (2.42) and style (7.07). For genre, a graph
  that weighs images 0.75 and captions 0.25 has 5.35, the caption graph 4.41; the image graph alone also has 5.35.
- Head: graphs that subtract the content similarity give the best heads for emotion (0.137 ± 0.002, aspect graph
  0.120, one seed) and style (0.260 ± 0.003, aspect graph 0.218). For genre, the image-weighted graph gives
  0.420 ± 0.002 (caption graph 0.402 ± 0.005).
- Against the prompted view clustered directly (no training), per side: the best head is below it on emotion
  captions (0.194 vs 0.217), style images (0.348 vs 0.395) and genre images (0.454 vs 0.496). It is above it on
  style captions (0.172 vs 0.135) and genre captions (0.386 vs 0.353), and level on emotion images (0.091 vs 0.093).
- The 10% of aspect pairs with the most similar content agree most on every label (emotion lift 2.80, whole graph
  2.42).

**What it does not show.** Development data only: one frozen train-side sample, the 2B Qwen model, ArtELingo. The
heads take the default embeddings and stop early (best dev loss at epoch 1 to 4), so they are a cheap probe of each
graph, not the stage 2 model. Three seeds only for the top two graphs per aspect; the aspect-graph head for emotion
and style has one seed. The labels picked the rows: 28 head runs were scored on one evaluation set that earlier
pilots also read, so the seed sd does not cover the selection. The image-only genre graph got no head.

**Options for next.**
- a. Stage 2 as in the spec: one shared trunk with three heads on the best graphs per aspect, loss forms a, b, c,
  beside per-view clustering and PercepT core. A build of one to two days, then a night of runs. It tells whether
  one model keeps all three aspects, and where it stands against PercepT core.
- b. A short stage 1 follow-up first, on CPU (about an hour, plus a small code change): the missing seeds of the
  aspect-graph heads, a head on the image-only genre graph, and a head that takes the prompted views as input. It
  tells whether the gains hold, and whether a head can pass the prompted view at all.
- c. PercepT core first (a few hours; the September code on our features, not checked yet). It tells whether
  clustering the prompted views already beats PercepT core, which sets what the trunk must add.

**Your reading.** (user, before ours, 2026-10-18) "The exploration results are interesting and show a lot of
potential, so we first do (b) to make the pilot more robust, then go (c), then (a)."

<details>
<summary>Our reading and recommendation (open after writing yours)</summary>

**Our reading.** Content and aspect go together in this data, now on all three aspects: every pruning row we tried
lowers the own-label lift. Fairly sure. Still, the head gains from pairs that agree on the aspect and differ more in
the rest: for emotion and style, the lift ratio orders the heads nearly as their target AMI does (two exceptions
inside the noise); the own-label lift does not. For genre, the strongest structure of the default embeddings already,
more genre agreement helps. Medium sure (one seed for two references, early-stopped heads). Genre's lift gain comes
from the image side: our default took genre's reference graph from captions, which made the reference weak. The main
risk for stage 2 is its outcome (b): a trunk on default inputs may stay below the prompted views it learns from.

**Our recommendation.** Option a, with b and c as its first steps (they run on CPU while the trunk is built). Carry
per aspect the aspect graph and the best residual graph; for genre the image-weighted graph and the image graph alone.
Our view.

</details>

## What ran

- **Stages and types:** stage 0, regression check (gate: the pilot's three emotion lifts within 1e-9); stage 1,
  exploration (no bar). Stages 2 and 3 wait for the spec's approval.
- **Data and seeds:** the frozen sample of the premise check (19,889 paintings, train side). Graphs on 13,400 fit
  paintings and, apart, on 1,489 dev paintings (early stopping). Scores on the fixed 5,000 evaluation paintings.
  Head seed 0 for every selected graph; seeds 1 and 2 for the top two per aspect by seed-0 target AMI.
- **Views:** emotion from emotion_focus captions, style from style_word images, genre from genre_word captions
  (agent default; the image side enters through the other-side, side-fusion, rank-fusion and agreement rows).
  New on the GPU: style_word and genre_word for the 14,889 training images (about 23 min each).
- **Graphs:** 21 rows per aspect, 63 in all ([exact definitions](../../../../src/test/20261018_aspect_graph_sheet/ags_rows.md)):
  references (aspect graph R0, other side R0o, full pruning R1, CLIP pruning R1c), prune by content rank or by
  content-score quantile, fuse (residual `z_aspect − λ·s_content`, content-direction removal, side fusion
  `w·caption + (1−w)·image`, rank fusion), guided (down-weight shared pairs, keep a shared pair only if the other
  side agrees, residual with a per-painting λ).
- **Selection** (spec §3.4; eligible rows have at least 20% of R0's fit pairs and 200 dev pairs): per aspect R0,
  R1 and the best row by lift ratio of each family, plus the best row of all by own-label lift.
- **Head:** default Qwen image and default caption mean in, a 2-layer MLP per side, InfoNCE on four buddy terms
  (weighted per pair) plus two own-caption terms; the pilot's settings otherwise. AMI at k = 20, 3 k-means seeds.
- **Code and who:** Codex wrote tickets 01 (`ags_graphs.py`) and 02 (`ags_head.py`, `ags_queue.sh`); the controller
  read the diffs, ran the smokes and merged them at 16:41; results commit `ba75753` on main.

## Results

Stage 0: `STAGE0 PASS 2.4204897561 / 2.3849313416 / 2.5528809893`; the pilot's six graphs are pair-identical on fit
and dev.

**Graphs and heads.** Lift on fit paintings; lift ratio = own-label lift over the mean lift of the other two labels.
Head target AMI: mean ± population sd over 3 seeds, or one seed. "No pair": fit paintings without a pair (of 13,400).

| Aspect | Row | Fit pairs | No pair | Own lift | Lift ratio | Head target AMI |
|---|---|---:|---:|---:|---:|---|
| emotion | R0, aspect graph | 57,727 | 440 | 2.42 | 0.78 | 0.120 (1 seed) |
| emotion | R1, full pruning | 45,505 | 705 | 2.38 | 0.82 | 0.125 (1 seed) |
| emotion | Psim50, prune top half by content | 28,864 | 1,418 | 2.30 | 1.06 | 0.120 (1 seed) |
| emotion | Gres, residual, per-painting λ | 35,754 | 1,628 | 2.35 | 1.15 | 0.132 ± 0.000 |
| emotion | Fres1, residual λ = 1 | 13,810 | 7,578 | 2.19 | 3.30 | **0.137 ± 0.002** |
| style | R0, aspect graph | 67,963 | 178 | 7.07 | 2.10 | 0.218 (1 seed) |
| style | R1, full pruning | 47,717 | 446 | 6.61 | 2.05 | 0.220 (1 seed) |
| style | Psim50 | 33,982 | 1,228 | 6.66 | 2.27 | 0.222 (1 seed) |
| style | Gres | 32,450 | 2,601 | 6.59 | 2.47 | 0.244 ± 0.002 |
| style | Fres05, residual λ = 0.5 | 23,136 | 5,322 | 6.54 | 3.36 | **0.260 ± 0.003** |
| genre | R0, caption graph | 56,190 | 541 | 4.41 | 1.79 | 0.402 ± 0.005 |
| genre | R0o, image graph alone | 62,906 | 282 | 5.35 | 1.38 | — |
| genre | R1, full pruning | 36,328 | 1,092 | 4.23 | 1.89 | 0.406 (1 seed) |
| genre | Psim75, prune top quarter | 42,142 | 753 | 3.95 | 1.87 | 0.403 (1 seed) |
| genre | Fres025, residual λ = 0.25 | 45,393 | 990 | 3.31 | 1.90 | 0.386 (1 seed) |
| genre | Gres | 26,419 | 2,762 | 3.15 | 1.88 | 0.381 (1 seed) |
| genre | Fside025, image 0.75 + caption 0.25 | 59,869 | 348 | 5.35 | 1.38 | **0.420 ± 0.002** |

All 63 rows: [graphs/sheet.md](../../../../src/test/20261018_aspect_graph_sheet/graphs/sheet.md); per-side AMI for
every head: [ags_head_tables.md](../../../../src/test/20261018_aspect_graph_sheet/ags_head_tables.md).

**Heads against the views, per side** (AMI of the aspect's own label; views with no training, sd over k-means seeds
at most 0.007 for the default and prompted views shown).

| Aspect, side | Default view | Prompted view | Best head (row) |
|---|---:|---:|---|
| emotion, captions (its side) | 0.107 | 0.217 | 0.194 (Fres1) |
| emotion, images | 0.070 | 0.093 | 0.091 (Gres) |
| style, images (its side) | 0.222 | 0.395 | 0.348 (Fres05) |
| style, captions | 0.120 | 0.135 | 0.172 (Fres05) |
| genre, images | 0.449 | 0.496 | 0.454 (Fside025) |
| genre, captions | 0.338 | 0.353 | 0.386 (Fside025) |
| **mean of both sides** | emotion 0.089, style 0.171, genre 0.393 | emotion 0.155, style 0.265, genre 0.424 | the best target AMI above |

**Verdict under the rule:** none (descriptive). Stage 0: pass.

## Analysis

- **The most content-similar pairs agree most on every label.** The pairs that the 90% content-score prune drops
  (the top 10% by content score) have higher lift than the whole aspect graph on all three aspects: emotion 2.80 vs
  2.42, style 8.78 vs 7.07, genre 6.00 vs 4.41. So every prune row lowers the own-label lift. This extends the
  pilot's finding (shared pairs 2.55 vs 2.38) from emotion to style and genre.
- **The head follows the lift ratio for emotion and style more than the own-label lift.** The best residual row
  has the lowest own-label lift of the selected rows but the highest lift ratio and target AMI. The order has two
  exceptions, both inside the seed noise (emotion prune-top-half, style full pruning). The emotion λ = 1 pairs share
  style and genre less often than random pairs (lifts 0.64 and 0.69; aspect graph 2.28 and 3.95). Its head's image
  genre AMI falls to 0.304 (aspect graph 0.427). Our reading: the head drops the dominant structure when the pairs
  cross it. Not tested.
- **Low coverage did not stop the best head.** The emotion λ = 1 graph leaves 57% of fit paintings with no pair
  (7,578 of 13,400), and its head is still the best. The effect of coverage was not isolated. The lift ratio needs
  this coverage column beside it (spec §8).
- **Genre is different.** It is the strongest structure of the default embeddings (default image genre AMI 0.449),
  and the head with the most genre agreement (lift 5.35) wins, at the lowest lift ratio. The image graph alone has
  the same lift as the image-weighted fusion (5.35 to two decimals), so the lift gain over the caption graph comes
  from the image side. Whether fusion helps the head is not tested: the image graph alone got no head, since the
  selection rule picked the fusion by less than 0.01 of lift.
- **The cross-modal head moves the aspect to the other side for style and genre.** Caption-side style 0.172 vs
  0.135 for the prompted captions, caption-side genre 0.386 vs 0.353. For emotion the image side stays at the
  prompted image's level (0.091 vs 0.093), as in the pilot (0.094).
- **For emotion and style, the prompted view is still ahead on the aspect's own side** (by 0.023 and 0.047), and
  on genre's image side (by 0.042). Possible reasons, not isolated: the heads read only the default embeddings, and
  each graph keeps only mutual top-20 pairs under its own score. It is the main risk for stage 2's outcome (b).
- **The style instruction works on the training images.** Style lift 7.07 for the style_word image graph, 5.37 for
  the default image graph of the pilot (spec §5 assumption, at the graph level).
- **The heads fit little.** Every run had its lowest dev loss at epoch 1 to 4 and stopped after 16 to 19 epochs, as
  the pilot's students did. The seed sd of target AMI stays at 0.005 or less.

## Deviations

- Stages 0 and 1 started before the spec's approval (D10, user).
- Agent defaults set in the build (run log, "Agent defaults"): genre's aspect graph from captions; the other-side
  and CLIP-pruning reference rows added (21 rows per aspect, 63 in all); similarities divided by their sd before
  mixing; the 20% coverage floor for selection; the head's loss with own-caption terms; the target AMI definition.
  They go into the spec as amendments at its approval.
- The aspect-graph heads for emotion and style have one seed: the seed rule gives seeds 1 and 2 only to the top two
  rows by seed-0 target AMI.
- Run log slips: (1) it gives 16.8% of the aspect graph's pairs for style's λ = 1 row; that is 9,727 / 57,727
  (emotion's aspect graph). Against style's it is 14.3%; the row is ineligible either way. (2) It says 39 head runs;
  the head sheet lists 28 (16 rows on seed 0, 6 rows on seeds 1 and 2), with no output missing.

## Files and storage

- **Outputs:** `src/test/20261018_aspect_graph_sheet/`: scripts, run log, `ags_rows.md`, `graphs/sheet.md`,
  `ags_head_tables.md` (in git); `ags_head_results.json`, `graphs/sheet.json`, `graphs/selected.json`,
  `head_seed_rows.json`, `graphs/*.npz` (gitignored). The style_word and genre_word training images are in the
  aspect-prompt store under `/data/SSD2/pre_extract/artelingo_full/`.
- **Storage:** the aspect-prompt store (6.5 GB) is kept (user, 2026-10-18 18:05). `outputs/` (1.1 GB, the head
  output pairs) and `graphs/` (67 MB) wait for the storage summary at the wrap-up; see the disk ledger rows dated
  2026-10-18.
