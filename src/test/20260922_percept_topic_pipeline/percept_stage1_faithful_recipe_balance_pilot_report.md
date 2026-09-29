# ArtELingo PercepT Stage 1 faithful recipe + balance pilot

Generated automatically, 2026-09-24 23:30:41.

## Controlled setup

The train-only input is the base pilot's 2816-dimensional fused vector: independently normalized CLIP content and GoEmotions-RoBERTa affect embeddings, with repeated content for a 2:1 norm budget. The 128-D autoencoder, Student's-t assignment, self-sharpened target, 100-center seeded K-means, and 67 highest-norm center pruning follow the faithful-recipe pilot.

Only Variant A is used: Gaussian latent noise with standard deviation 0.1 during the 100-epoch pretrain, and clean z for reconstruction, KL, and balance during joint DEC. Both phases use Adam at initial LR 1e-3 and cosine annealing to 1e-05; pretrain T_max=100, DEC T_max=200 and held at the floor thereafter. DEC keeps `fraction_changed < 0.001` and the 500-epoch ceiling. Joint loss is KL + 1.0 * MSE reconstruction + lambda_balance * KL(uniform || mean soft assignment).

Stage 1 isolates lambda_balance in {100, 300, 1000} at seed 42 after a shared fresh pretrain. Stage 2, when gated in, uses fresh autoencoders, noisy pretraining, and seeded K-means at seeds 7, 123, and 2024 for only the chosen lambda.

## Predeclared success criterion

**Held-out Pareto bar = emotion AMI > 0.1236 AND genre AMI > 0.1954, both simultaneously.** Real success additionally requires non-collapse under the surviving-center rule. Silhouette uses final 128-D Z and surviving-center assignments separately on both splits; it is n/a if only one cluster is assigned.

## Stage 1: seed 42 balance screen

### Training and cluster-size trajectories

### Lambda 100, seed 42

- Pretraining reconstruction and LR: epoch 10: recon=0.000091, lr=0.00098034537; epoch 20: recon=0.000073, lr=0.00091440488; epoch 30: recon=0.000059, lr=0.00080838899; epoch 40: recon=0.000053, lr=0.00067267527; epoch 50: recon=0.000050, lr=0.00052054833; epoch 60: recon=0.000048, lr=0.0003668994; epoch 70: recon=0.000047, lr=0.00022676873; epoch 80: recon=0.000046, lr=0.00011387327; epoch 90: recon=0.000045, lr=3.9264019e-05; epoch 100: recon=0.000045, lr=1.0244253e-05.
- DEC stopped via **stability criterion** at epoch **164** (threshold `fraction_changed < 0.001`, ceiling 500).
- Joint DEC losses and LR: epoch 1: total=0.885229, KL=0.241859, recon=0.000045, balance=0.006433, fraction_changed=0.927331, lr=0.001; epoch 10: total=0.038183, KL=0.000099, recon=0.000184, balance=0.000379, fraction_changed=0.082831, lr=0.00099506171; epoch 20: total=0.008527, KL=0.000007, recon=0.000188, balance=0.000083, fraction_changed=0.022524, lr=0.00097811754; epoch 25: total=0.005546, KL=0.000003, recon=0.000138, balance=0.000054, fraction_changed=0.021677, lr=0.00096523936; epoch 30: total=0.004008, KL=0.000002, recon=0.000134, balance=0.000039, fraction_changed=0.024071, lr=0.00094952365; epoch 40: total=0.002490, KL=0.000001, recon=0.000119, balance=0.000024, fraction_changed=0.033582, lr=0.00090998411; epoch 50: total=0.001791, KL=0.000001, recon=0.000111, balance=0.000017, fraction_changed=0.017654, lr=0.00086047252; epoch 60: total=0.001391, KL=0.000000, recon=0.000106, balance=0.000013, fraction_changed=0.017019, lr=0.00080220801; epoch 70: total=0.001137, KL=0.000000, recon=0.000104, balance=0.000010, fraction_changed=0.005684, lr=0.00073662526; epoch 75: total=0.001044, KL=0.000000, recon=0.000104, balance=0.000009, fraction_changed=0.002720, lr=0.00070158821; epoch 80: total=0.000957, KL=0.000000, recon=0.000104, balance=0.000009, fraction_changed=0.003388, lr=0.00066533912; epoch 90: total=0.000832, KL=0.000000, recon=0.000104, balance=0.000007, fraction_changed=0.011645, lr=0.0005901049; epoch 100: total=0.000729, KL=0.000000, recon=0.000104, balance=0.000006, fraction_changed=0.011856, lr=0.00051277512; epoch 110: total=0.000655, KL=0.000000, recon=0.000104, balance=0.000006, fraction_changed=0.006124, lr=0.00043525389; epoch 120: total=0.000601, KL=0.000000, recon=0.000104, balance=0.000005, fraction_changed=0.003909, lr=0.00035945004; epoch 125: total=0.000579, KL=0.000000, recon=0.000104, balance=0.000005, fraction_changed=0.002834, lr=0.00032277835; epoch 130: total=0.000559, KL=0.000000, recon=0.000104, balance=0.000005, fraction_changed=0.002948, lr=0.00028723011; epoch 140: total=0.000531, KL=0.000000, recon=0.000104, balance=0.000004, fraction_changed=0.002524, lr=0.0002203724; epoch 150: total=0.000508, KL=0.000000, recon=0.000104, balance=0.000004, fraction_changed=0.001743, lr=0.00016052317; epoch 160: total=0.000496, KL=0.000000, recon=0.000104, balance=0.000004, fraction_changed=0.001140, lr=0.00010915609; epoch 164: total=0.000491, KL=0.000000, recon=0.000104, balance=0.000004, fraction_changed=0.000993, lr=9.1275356e-05.
- All-100-center diagnostics (epoch 0 and every 25 epochs or stop): epoch 0: min=413, max=949, median=604.0, below_1pct=52/100; epoch 25: min=0, max=44,985, median=0.0, below_1pct=96/100; epoch 50: min=0, max=33,464, median=0.0, below_1pct=97/100; epoch 75: min=0, max=51,825, median=0.0, below_1pct=98/100; epoch 100: min=0, max=47,760, median=0.0, below_1pct=97/100; epoch 125: min=0, max=41,452, median=0.0, below_1pct=98/100; epoch 150: min=0, max=37,357, median=0.0, below_1pct=98/100; epoch 164: min=0, max=36,215, median=0.0, below_1pct=98/100.

### Lambda 300, seed 42

- Pretraining reconstruction and LR: epoch 10: recon=0.000091, lr=0.00098034537; epoch 20: recon=0.000073, lr=0.00091440488; epoch 30: recon=0.000059, lr=0.00080838899; epoch 40: recon=0.000053, lr=0.00067267527; epoch 50: recon=0.000050, lr=0.00052054833; epoch 60: recon=0.000048, lr=0.0003668994; epoch 70: recon=0.000047, lr=0.00022676873; epoch 80: recon=0.000046, lr=0.00011387327; epoch 90: recon=0.000045, lr=3.9264019e-05; epoch 100: recon=0.000045, lr=1.0244253e-05.
- DEC stopped via **stability criterion** at epoch **76** (threshold `fraction_changed < 0.001`, ceiling 500).
- Joint DEC losses and LR: epoch 1: total=2.169020, KL=0.241872, recon=0.000045, balance=0.006424, fraction_changed=0.927185, lr=0.001; epoch 10: total=0.110098, KL=0.000093, recon=0.000186, balance=0.000366, fraction_changed=0.079411, lr=0.00099506171; epoch 20: total=0.023874, KL=0.000006, recon=0.000163, balance=0.000079, fraction_changed=0.026758, lr=0.00097811754; epoch 25: total=0.015421, KL=0.000003, recon=0.000148, balance=0.000051, fraction_changed=0.026270, lr=0.00096523936; epoch 30: total=0.011007, KL=0.000002, recon=0.000130, balance=0.000036, fraction_changed=0.029168, lr=0.00094952365; epoch 40: total=0.006743, KL=0.000001, recon=0.000131, balance=0.000022, fraction_changed=0.031546, lr=0.00090998411; epoch 50: total=0.004781, KL=0.000001, recon=0.000115, balance=0.000016, fraction_changed=0.018012, lr=0.00086047252; epoch 60: total=0.003691, KL=0.000000, recon=0.000107, balance=0.000012, fraction_changed=0.017377, lr=0.00080220801; epoch 70: total=0.002979, KL=0.000000, recon=0.000105, balance=0.000010, fraction_changed=0.004186, lr=0.00073662526; epoch 75: total=0.002697, KL=0.000000, recon=0.000104, balance=0.000009, fraction_changed=0.001221, lr=0.00070158821; epoch 76: total=0.002652, KL=0.000000, recon=0.000104, balance=0.000008, fraction_changed=0.000717, lr=0.0006944283.
- All-100-center diagnostics (epoch 0 and every 25 epochs or stop): epoch 0: min=404, max=951, median=603.5, below_1pct=52/100; epoch 25: min=0, max=40,719, median=0.0, below_1pct=96/100; epoch 50: min=0, max=34,847, median=0.0, below_1pct=97/100; epoch 75: min=0, max=52,415, median=0.0, below_1pct=98/100; epoch 76: min=0, max=52,457, median=0.0, below_1pct=98/100.

### Lambda 1000, seed 42

- Pretraining reconstruction and LR: epoch 10: recon=0.000091, lr=0.00098034537; epoch 20: recon=0.000073, lr=0.00091440488; epoch 30: recon=0.000059, lr=0.00080838899; epoch 40: recon=0.000053, lr=0.00067267527; epoch 50: recon=0.000050, lr=0.00052054833; epoch 60: recon=0.000048, lr=0.0003668994; epoch 70: recon=0.000047, lr=0.00022676873; epoch 80: recon=0.000046, lr=0.00011387327; epoch 90: recon=0.000045, lr=3.9264019e-05; epoch 100: recon=0.000045, lr=1.0244253e-05.
- DEC stopped via **stability criterion** at epoch **76** (threshold `fraction_changed < 0.001`, ceiling 500).
- Joint DEC losses and LR: epoch 1: total=6.661076, KL=0.241845, recon=0.000045, balance=0.006419, fraction_changed=0.927771, lr=0.001; epoch 10: total=0.360522, KL=0.000088, recon=0.000217, balance=0.000360, fraction_changed=0.074460, lr=0.00099506171; epoch 20: total=0.076879, KL=0.000006, recon=0.000203, balance=0.000077, fraction_changed=0.024006, lr=0.00097811754; epoch 25: total=0.049186, KL=0.000003, recon=0.000158, balance=0.000049, fraction_changed=0.021025, lr=0.00096523936; epoch 30: total=0.034828, KL=0.000002, recon=0.000166, balance=0.000035, fraction_changed=0.023224, lr=0.00094952365; epoch 40: total=0.021012, KL=0.000001, recon=0.000112, balance=0.000021, fraction_changed=0.031856, lr=0.00090998411; epoch 50: total=0.014734, KL=0.000001, recon=0.000108, balance=0.000015, fraction_changed=0.021123, lr=0.00086047252; epoch 60: total=0.011222, KL=0.000000, recon=0.000105, balance=0.000011, fraction_changed=0.019332, lr=0.00080220801; epoch 70: total=0.008933, KL=0.000000, recon=0.000104, balance=0.000009, fraction_changed=0.004316, lr=0.00073662526; epoch 75: total=0.008020, KL=0.000000, recon=0.000104, balance=0.000008, fraction_changed=0.001107, lr=0.00070158821; epoch 76: total=0.007864, KL=0.000000, recon=0.000104, balance=0.000008, fraction_changed=0.000961, lr=0.0006944283.
- All-100-center diagnostics (epoch 0 and every 25 epochs or stop): epoch 0: min=414, max=951, median=603.5, below_1pct=53/100; epoch 25: min=0, max=43,128, median=0.0, below_1pct=96/100; epoch 50: min=0, max=36,221, median=0.0, below_1pct=97/100; epoch 75: min=0, max=56,221, median=0.0, below_1pct=98/100; epoch 76: min=0, max=56,280, median=0.0, below_1pct=98/100.

### Results on both splits

| lambda | seed | split | emotion AMI | genre AMI | silhouette (128-D Z) | verdict | held-out Pareto bar | surviving min | surviving max | surviving median | surviving below 1% | all-100 min | all-100 max | all-100 median | all-100 below 1% |
|---:|---:|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 42 | train | 0.0684 | 0.2504 | 0.0803 | Collapsed | n/a (train split) | 0 | 36,215 | 0.0 | 65/67 | 0 | 36,215 | 0.0 | 98/100 |
| 100 | 42 | held-out | 0.0536 | 0.2247 | 0.0703 | Collapsed | does not clear | 0 | 5,964 | 0.0 | 65/67 | 0 | 36,215 | 0.0 | 98/100 |
| 300 | 42 | train | 0.0524 | 0.1888 | -0.2191 | Collapsed | n/a (train split) | 0 | 52,457 | 0.0 | 65/67 | 0 | 52,457 | 0.0 | 98/100 |
| 300 | 42 | held-out | 0.0420 | 0.1413 | -0.2162 | Collapsed | does not clear | 0 | 8,228 | 0.0 | 65/67 | 0 | 52,457 | 0.0 | 98/100 |
| 1000 | 42 | train | 0.0373 | 0.1133 | 0.0261 | Collapsed | n/a (train split) | 0 | 56,280 | 0.0 | 65/67 | 0 | 56,280 | 0.0 | 98/100 |
| 1000 | 42 | held-out | 0.0317 | 0.0720 | 0.1499 | Collapsed | does not clear | 0 | 8,732 | 0.0 | 65/67 | 0 | 56,280 | 0.0 | 98/100 |

Surviving-center diagnostics are split-specific. All-100 diagnostics repeat the final train checkpoint before pruning.

## Stage 2

Skipped: none of the three screened values cleared both held-out Pareto bars at seed 42. All three are plain misses. The lambda range was not widened.

## Decision

**No held-out Pareto success.** All three prescribed lambda values missed at least one bar; no Stage-2 seed stress was run. This pilot does not establish a new standing PercepT Stage 1 configuration.
