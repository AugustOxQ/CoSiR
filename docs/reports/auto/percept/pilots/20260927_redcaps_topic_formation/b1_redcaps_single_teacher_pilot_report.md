# B1 — RedCaps single-teacher Stage 1 pilot

Generated 2026-09-27 02:08:36; seed 42; 150k source `STORAGE=/data/SSD2/pre_extract/redcaps_150k/features` and `ANNOT=/data/PDD/redcaps/redcaps_plus/redcaps_150k.json`.

## Split and teacher sanity check

Train/validation/test sizes: **120,000/15,000/15,000**. The shuffled index arrays are saved at `/project/CoSiR-buddy_prototype_conditioning/src/test/20260927_redcaps_topic_formation/b1_redcaps_single_teacher_pilot_split.npz`. The split uses `default_rng(42).shuffle(arange(N))`, with no subreddit lookup; test rows are saved but never evaluated by this pilot.

Train raw union: **22.850× (0.3681/0.0161)**, 1,148,216 edges, 150 qualifying subreddits. Train degree-repaired union: **22.840× (0.3680/0.0161)**, 1,149,570 edges, 150 qualifying subreddits. Repair fixed 1,354 isolated nodes and added 1,354 undirected edges. Edge types: image-only 659,032, text-only 465,584, both 23,600, repair 1,354.

Experiment 16 Stage A reports **22.788×** for the **full 150k, K=30 repaired union** ([docs/reports/auto/buddy/2026-09-01_buddy_k_scaling_stage_a.md](/project/CoSiR-buddy_prototype_conditioning/docs/reports/2026-09-01_buddy_k_scaling_stage_a.md)). The train-only repaired union differs by +0.2% and the raw union by +0.3%. The repaired figure is the primary comparison because the cited protocol includes repair. This is within the declared 25% same-ballpark gate.

## Validation results

Lift entries show overall lift × (observed/expected same-subreddit fraction); expectation uses the edge-endpoint subreddit marginal.

| Method | Validation embedding-graph lift | Community-level lift | Train Leiden occupancy (C; min/median/max; below 1%) | Transfer coverage | Fallbacks |
|---|---:|---:|---:|---:|---:|
| Attention student | 18.391× (0.3088/0.0168) | 6.720× (0.1426/0.0212) | 326; 1/1.0/11126; 304 | 35/326 | 0 |
| Mean-pool control | 18.973× (0.3163/0.0167) | 6.865× (0.1462/0.0213) | 555; 1/1.0/10016; 529 | 39/555 | 0 |
| Raw image control | 24.314× (0.3485/0.0143) | n/a | n/a | n/a | n/a |
| Raw text control | 18.153× (0.3099/0.0171) | n/a | n/a | n/a | n/a |
| Graph-only baseline | 27.114× (0.4089/0.0151) | 4.773× (0.1140/0.0239) | 1423; 1/1.0/21482; 1404 | 27/1423 | 0 |

### Lift support and sampling

- Attention student: graph 149,614 edges, 57 qualifying subreddits; same-community 5,656,699 sampled pairs, 214 qualifying subreddits, 0 communities above 2,000 points subsampled.
- Mean-pool control: graph 147,663 edges, 57 qualifying subreddits; same-community 5,511,510 sampled pairs, 206 qualifying subreddits, 0 communities above 2,000 points subsampled.
- Raw image control: graph 101,335 edges, 45 qualifying subreddits.
- Raw text control: graph 51,528 edges, 31 qualifying subreddits.
- Graph-only baseline: graph 72,668 edges, 39 qualifying subreddits; same-community 7,738,212 sampled pairs, 214 qualifying subreddits, 1 communities above 2,000 points subsampled.

For any transferred community above 2,000 validation points, a seed-42 random sample of 2,000 members is selected before pairs are enumerated. Thus no large community creates its full dense pair set (maximum 1,999,000 pairs per community).

Training positive pairs and the graph-only Leiden partition use the raw train teacher union `E`; the repaired graph is reported for the Experiment 16 sanity comparison only. Graph-only cosine k=20 transfer uses unit-normalized, concatenated raw image+text CLIP features; its validation embedding graph uses the same features at K=30. Both modalities therefore contribute equally in this no-training control. Raw image/text controls have no Leiden labels.

## Training

Both students use seed 42, Adam at 0.001, 1024 sampled positive pairs per epoch, K=30, a 200-epoch ceiling, and a 5-epoch recall checkpoint with 5 consecutive relative gains below 0.01 as the stopping rule. Recall is mean per-node raw-train-teacher-neighbor recall on a fixed seed-42 sample of 2,000 train nodes.

- Attention student: 37,120 trainable parameters; stopped at epoch 200 (reached MAX_EPOCHS=200); final sampled train teacher recall 0.1787.
- Mean-pool control: 32,896 trainable parameters; stopped at epoch 175 (recall plateaued for 5 consecutive checkpoints); final sampled train teacher recall 0.1916.
Both controls have the same two 512→32 projections, 32-D output, and LayerNorm. Attention adds its own Q/K/V/output weights; the mean-pool control has no learned attention. Parameter counts are therefore shown rather than claimed to be identical.

## Verdict

The attention student does not match every listed validation embedding-graph lift control. Attention does not improve over mean pooling on this lift measure. Attention does not match both community-level lift controls. Stop before B2 patch extraction or 300k/500k DAS6 training and diagnose the teacher, student geometry, and transfer coverage.
This one-seed validation gate is not a sealed-test or downstream-topic claim.
