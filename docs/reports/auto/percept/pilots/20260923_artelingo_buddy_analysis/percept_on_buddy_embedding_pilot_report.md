# PercepT faithful Variant A on frozen buddy embeddings

Generated automatically, 2026-09-26 23:24:43.

## Method

Seed 42 only. Canonical train and held-out painting orders came from the respective `pipeline.load_dedup_features()` calls. The painting sets and uniqueness were checked against the buddy snapshot with the faithful snapshot template's `assert_matching_paintings`, then all saved buddy arrays were reindexed to those orders. For the native-space sampled silhouette, DEC labels were mapped back to the snapshot's saved held-out order so the seed-42 indices match the cited buddy baseline. The frozen float32, unit-normalized 32-D `embedding_post` vectors replaced PercepT's 2,816-D fused input h entirely; no buddy training or affect extraction was repeated.

The faithful recipe's own seed-threaded pretraining, K-means initialization, joint DEC training, and assignment evaluation functions were called unmodified. Variant A uses noise 0.1 in pretraining only (noise=0.1), lambda_R=1, 100 initial centers and 67 highest-norm survivors. The unchanged autoencoder maps 32→500→500→2000→128→2000→500→500→32. Its 128-D latent is overcomplete relative to the 32-D input. Raw per-coordinate MSE and the pretraining noise-to-signal scale can therefore differ; lambda_R=1 is audited below, not presumed equivalent to the original run.

Point (a) uses the seed-42 buddy Leiden AMIs from `attention_h1_embedding_snapshot_pilot_report.md` and the matching seed-42 sampled silhouette from `attention_h1_baseline_seed_stress_pilot_report.md`; buddy occupancy is counted from that snapshot's saved held-out labels. Point (b) uses the faithful full-held-out 128-D latent evaluator. Point (c) keeps exactly the labels from (b), but scores silhouette on original 32-D held-out vectors with a seeded draw of at most 6,000 and `silhouette_score(sample_size=min(4000, len(idx)), random_state=42)`. AMI and occupancy are identical in (b) and (c). The original reference uses the 2,816-D Variant A held-out row of `percept_stage1_faithful_recipe_pilot_report.md`. Full latent silhouettes and sampled native silhouettes have different protocols and spaces.

Held-out Pareto bar: emotion AMI > 0.1236 and genre AMI > 0.1954. The verdict below calls topics useful only when both bars clear, 67-center occupancy is not collapsed, and the full 128-D latent silhouette is positive. Whether the partition improves buddy's native geometry is a separate verdict below. For the numerical 'approaches original' comparison, all three 128-D metrics must reach at least 90% of the original row; this is an exploratory screen threshold.

## Training and cluster-size trajectories

### Variant A (pretrain-only noise), seed 42

- Pretraining reconstruction and LR: epoch 10: recon=0.000481, lr=0.00098034537; epoch 20: recon=0.000339, lr=0.00091440488; epoch 30: recon=0.000229, lr=0.00080838899; epoch 40: recon=0.000162, lr=0.00067267527; epoch 50: recon=0.000115, lr=0.00052054833; epoch 60: recon=0.000087, lr=0.0003668994; epoch 70: recon=0.000072, lr=0.00022676873; epoch 80: recon=0.000064, lr=0.00011387327; epoch 90: recon=0.000060, lr=3.9264019e-05; epoch 100: recon=0.000058, lr=1.0244253e-05.
- DEC stopped via **stability criterion** at epoch **133** (threshold `fraction_changed < 0.001`, ceiling 500).
- Joint DEC losses and LR: epoch 1: total=0.076322, KL=0.076279, recon=0.000043, fraction_changed=0.347106, lr=0.001; epoch 10: total=0.147265, KL=0.130254, recon=0.017012, fraction_changed=0.077945, lr=0.00099506171; epoch 20: total=0.193972, KL=0.184915, recon=0.009057, fraction_changed=0.050308, lr=0.00097811754; epoch 25: total=0.226961, KL=0.219668, recon=0.007293, fraction_changed=0.052360, lr=0.00096523936; epoch 30: total=0.257636, KL=0.251082, recon=0.006554, fraction_changed=0.044624, lr=0.00094952365; epoch 40: total=0.288652, KL=0.277631, recon=0.011021, fraction_changed=0.021693, lr=0.00090998411; epoch 50: total=0.277015, KL=0.264570, recon=0.012444, fraction_changed=0.012035, lr=0.00086047252; epoch 60: total=0.283771, KL=0.266676, recon=0.017094, fraction_changed=0.020113, lr=0.00080220801; epoch 70: total=0.255107, KL=0.240416, recon=0.014691, fraction_changed=0.005830, lr=0.00073662526; epoch 75: total=0.249658, KL=0.234868, recon=0.014790, fraction_changed=0.004886, lr=0.00070158821; epoch 80: total=0.238649, KL=0.223546, recon=0.015102, fraction_changed=0.006710, lr=0.00066533912; epoch 90: total=0.226468, KL=0.210784, recon=0.015684, fraction_changed=0.006156, lr=0.0005901049; epoch 100: total=0.216704, KL=0.200117, recon=0.016588, fraction_changed=0.006482, lr=0.00051277512; epoch 110: total=0.209947, KL=0.193665, recon=0.016282, fraction_changed=0.003681, lr=0.00043525389; epoch 120: total=0.204634, KL=0.188313, recon=0.016321, fraction_changed=0.001694, lr=0.00035945004; epoch 125: total=0.202662, KL=0.186340, recon=0.016322, fraction_changed=0.001335, lr=0.00032277835; epoch 130: total=0.201330, KL=0.185012, recon=0.016318, fraction_changed=0.001091, lr=0.00028723011; epoch 133: total=0.200423, KL=0.184118, recon=0.016305, fraction_changed=0.000977, lr=0.00026653193.
- All-100-center diagnostics (epoch 0 and every 25 epochs or stop): epoch 0: min=416, max=1,035, median=599.0, below_1pct=55/100; epoch 25: min=7, max=6,349, median=211.5, below_1pct=74/100; epoch 50: min=0, max=12,392, median=0.0, below_1pct=88/100; epoch 75: min=0, max=12,183, median=0.0, below_1pct=88/100; epoch 100: min=0, max=10,429, median=0.0, below_1pct=90/100; epoch 125: min=0, max=9,303, median=0.0, below_1pct=90/100; epoch 133: min=0, max=9,262, median=0.0, below_1pct=90/100.

## Loss-scale audit

MSE is mean per input coordinate. The ratio is final DEC reconstruction / final DEC KL; the original cited losses were from epoch 154 and rounded to six decimal places. For unit-norm inputs, MSE × input dimension is reconstruction squared error relative to the input's unit squared norm.

| input | final pretrain reconstruction | final DEC reconstruction | final DEC KL | reconstruction / KL | final DEC MSE × dim |
|---|---:|---:|---:|---:|---:|
| buddy 32-D | 0.00005831 | 0.01630477 | 0.18411827 | 0.088556 | 0.521753 |
| original 2,816-D | 0.000045 | 0.000080 | 0.391414 | 0.000204387 | 0.225280 |

The new ratio is 433.28× the original ratio: **shifted substantially** under a predeclared threefold scale check. These loss values alone do not measure gradient strength.

## Held-out comparison

| evaluation point (seed 42) | emotion AMI | genre AMI | silhouette | occupancy |
|---|---:|---:|---:|---|
| (a) buddy Leiden, native 32-D | 0.1249 | 0.2404 | 0.0377 (sampled) | min 85; max 955; median 521.5; zero 0; below 1% 2/18; Leiden (67-center rule n/a) |
| (b) DEC labels, PercepT 128-D latent | 0.1482 | 0.1765 | 0.4886 (full) | min 0; max 1,495; median 0.0; zero 44; below 1% 57/67; collapsed |
| (c) same DEC labels, buddy native 32-D | 0.1482 | 0.1765 | -0.0252 (sampled) | min 0; max 1,495; median 0.0; zero 44; below 1% 57/67; collapsed |
| original 2,816-D PercepT Variant A, 128-D latent | 0.1092 | 0.3288 | 0.5120 (full) | min 0; max 1,251; median 13.0; zero n/a; below 1% 50/67; collapsed |

New DEC train occupancy: min 0; max 8,277; median 0.0; zero 42; below 1% 56/67; collapsed. Held-out genre n=159. Original zero count was not reported; its minimum of zero does establish at least one empty center. Buddy's own Leiden labels use 18 observed held-out communities, so the 67-center collapse criterion does not apply.

## Verdict

- Compressed-input signal: **no** by the stated screen. DEC held-out AMI=0.1482/0.1765, 128-D silhouette=0.4886, and 57/67 centers below 1% (collapsed).
- Beats buddy's own Leiden partition in native space: **no** by noncollapse, both AMIs at least matching 0.1249/0.2404, and native silhouette exceeding 0.0377. Observed AMI differences=+0.0233/-0.0639; native silhouette difference=-0.0629.
- Approaches original 2,816-D result numerically: **no; falls short** on the 90%-of-each-metric screen. Differences in held-out AMI=+0.0390/-0.1523; full 128-D silhouette difference=-0.0234. The original itself was collapsed (50/67 below 1%).
