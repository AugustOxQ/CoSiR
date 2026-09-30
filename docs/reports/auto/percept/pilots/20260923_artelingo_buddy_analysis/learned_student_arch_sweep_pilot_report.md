# ArtELingo learned two-teacher student — architecture sweep

Generated automatically, 2026-09-22 21:11:45.

## Context

This sweep asks whether self-attention can beat Stage 1's scalar gate while holding linear projection heads fixed, and whether Stage 2's MLP-underperforms finding remains true with 128 hidden units instead of 64. All new runs retain Stage 1's data, PCA, loss, sampling, learning rate, training budget, diagnostics, and stopping rule.

## Results

| architecture | split | emotion AMI | genre AMI | collapse verdict | final content gradient share | final gate/attention-weight mean | final gate/attention-weight saturated fraction |
|---|---|---:|---:|---|---:|---:|---:|
| Linear (reference, not rerun) | train | 0.1284 | 0.2799 | Collapsed | 0.4988 | 0.4926 | 0.0000 |
| Linear (reference, not rerun) | held-out | 0.1095 | 0.2901 | Collapsed | 0.4988 | 0.4926 | 0.0000 |
| MLP-64 (reference, not rerun) | train | 0.1230 | 0.2319 | Merely a compromise | 0.4823 | 0.4865 | 0.0000 |
| MLP-64 (reference, not rerun) | held-out | 0.1046 | 0.2087 | Merely a compromise | 0.4823 | 0.4865 | 0.0000 |
| MLP-128 | train | 0.1142 | 0.2039 | Collapsed | 0.4807 | 0.4794 | 0.0000 |
| MLP-128 | held-out | 0.1114 | 0.1604 | Collapsed | 0.4807 | 0.4794 | 0.0000 |
| Attention-h1 | train | 0.1351 | 0.2397 | Real success | 0.5008 | 0.4863 | 0.0000 |
| Attention-h1 | held-out | 0.1249 | 0.2404 | Real success | 0.5008 | 0.4863 | 0.0000 |
| Attention-h4 | train | 0.1328 | 0.2441 | Real success | 0.5090 | 0.4914 | 0.0000 |
| Attention-h4 | held-out | 0.1216 | 0.1875 | Real success | 0.5090 | 0.4914 | 0.0000 |

## Held-out Pareto verdict

The held-out Pareto bar is emotion AMI > 0.1236 and genre AMI > 0.1954 — the same predeclared, both-simultaneously bar used throughout this investigation (Stage 1, Stage 2, and the weight sweep all missed it on held-out). **Attention-h1 is the first and only configuration across the entire investigation to clear this bar on held-out data**: emotion AMI=0.1249, genre AMI=0.2404, both above their thresholds. It also clears the train bar (0.1351/0.2397). Its training trajectory is healthy, not fragile: content and affect held-out recall both developed well past their epoch-0 baselines (+27.5% and +597% respectively), content gradient share stayed centered within [0.47, 0.52] for all 200 epochs without drifting toward either extreme, the self-attention weight itself barely moved (0.4999 → 0.4863) rather than saturating, and effective rank grew from 12 to 20 rather than collapsing.

Attention-h1 does not *strictly dominate* Stage 1 on held-out — it trades some of Stage 1's genre AMI (0.2901) for a real gain in emotion AMI (0.1249 vs. 0.1095), landing at a lower summed AMI (0.3653 vs. Stage 1's 0.3996) despite clearing the bar Stage 1 missed. Both framings are worth stating plainly rather than picking one: by strict dominance, Stage 1 still has the higher genre number; by the predeclared Pareto-bar criterion this investigation has used as its primary success test since Stage 1, Attention-h1 is now the standing result.

## Attention-head comparison

Attention-h1 held-out summed AMI is 0.3653; Attention-h4 is 0.3092. On this measure, more heads hurt — and only Attention-h1 clears the held-out Pareto bar (Attention-h4's held-out genre AMI, 0.1875, falls just short of the 0.1954 bar). Splitting one already-small (32-dimensional) shared space across 4 attention heads likely leaves too little per-head capacity to combine two views well; a single head keeps the full shared space for one attention computation.

## MLP-capacity comparison

MLP-64 held-out summed AMI is 0.3133; MLP-128 is 0.2719. More MLP capacity does not improve the result — consistent with Stage 2's original finding that richer PER-VIEW encoding hurts, now confirmed to hold at a second, larger capacity too.

## Conclusion

Two different rankings answer two different questions here, and neither should be collapsed into the other. By held-out summed AMI, Linear (Stage 1) still ranks highest (0.3996). By the predeclared Pareto-bar criterion — the primary success test used throughout this investigation — **Attention-h1 is the new best result**: the first configuration to clear both held-out thresholds simultaneously. The mechanistic pattern across this whole sweep is now coherent: richer PER-VIEW encoding capacity (MLP heads, Stage 2 and MLP-128) consistently hurts, while richer COMBINATION capacity (attention vs. a scalar gate), holding per-view encoding at the simple linear heads that already work best, helps — provided that combination capacity isn't spread too thin (1 head, not 4).
