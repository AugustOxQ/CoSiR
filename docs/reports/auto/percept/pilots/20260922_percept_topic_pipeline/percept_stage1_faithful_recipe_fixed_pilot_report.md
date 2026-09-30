# ArtELingo PercepT Stage 1 faithful recipe pilot (BUG-FIXED re-run)

Generated automatically, 2026-09-27 15:38:32.

Fixes two bugs found by an independent adversarial review (agy) and verified against the actual paper (arXiv:2606.03345) on 2026-09-27, documented in full in [`docs/reports/auto/percept/2026-09-27_agy_independent_percept_review.md`](../../2026-09-27_agy_independent_percept_review.md) and the module docstring of this file. In one sentence: the original pilot's center-pruning kept the wrong (high-norm, underused) centers and its reconstruction loss was ~2,816x too weak relative to the paper's own stated formula. See `percept_stage1_faithful_recipe_pilot_report.md` for the original, left unmodified for provenance.

## Controlled setup

Identical to the original faithful-recipe pilot except: (1) center pruning now keeps the 67 LOWEST-norm centers instead of the 67 highest (paper Algorithm 1 direction, arXiv:2606.03345 Sec 4.3: underused centers drift to large norms and should be discarded, not kept) -- an initial attempt at the paper's exact max-finite-difference threshold was tried first but was fooled by a single extreme-norm outlier (kept 99/100 centers); the fixed-67-count target is this project's own original, documented proxy for the paper's reported ~67/100 empirical retention rate, now applied in the correct direction; (2) reconstruction loss sums squared error over the feature dimension before averaging over the batch, matching the paper's unreduced `||h - h_hat||^2` rather than dividing by the 2,816-D feature dimension.

## Predeclared success criterion

**Held-out Pareto bar = emotion AMI > 0.1236 AND genre AMI > 0.1954, both simultaneously.** A Real success also requires non-collapse under the established surviving-center rule (now measured against the actual, data-dependent surviving count).

## Phase 1: seed 42, both variants

### Training and cluster-size trajectories

### Variant A (pretrain-only noise), seed 42 (67 surviving centers)

- Pretraining reconstruction and LR: epoch 10: recon=0.183434, lr=0.00098034537; epoch 20: recon=0.144672, lr=0.00091440488; epoch 30: recon=0.130178, lr=0.00080838899; epoch 40: recon=0.119873, lr=0.00067267527; epoch 50: recon=0.113449, lr=0.00052054833; epoch 60: recon=0.108926, lr=0.0003668994; epoch 70: recon=0.106330, lr=0.00022676873; epoch 80: recon=0.104900, lr=0.00011387327; epoch 90: recon=0.104203, lr=3.9264019e-05; epoch 100: recon=0.103957, lr=1.0244253e-05.
- DEC stopped via **stability criterion** at epoch **158** (threshold `fraction_changed < 0.001`, ceiling 500).
- Joint DEC losses and LR: epoch 1: total=0.250742, KL=0.147385, recon=0.103358, fraction_changed=0.507508, lr=0.001; epoch 10: total=0.287961, KL=0.097686, recon=0.190275, fraction_changed=0.110566, lr=0.00099506171; epoch 20: total=0.373339, KL=0.206556, recon=0.166783, fraction_changed=0.038321, lr=0.00097811754; epoch 25: total=0.378969, KL=0.226497, recon=0.152471, fraction_changed=0.028175, lr=0.00096523936; epoch 30: total=0.391419, KL=0.248355, recon=0.143063, fraction_changed=0.021237, lr=0.00094952365; epoch 40: total=0.419914, KL=0.286254, recon=0.133659, fraction_changed=0.018322, lr=0.00090998411; epoch 50: total=0.444494, KL=0.315356, recon=0.129138, fraction_changed=0.026872, lr=0.00086047252; epoch 60: total=0.470532, KL=0.337520, recon=0.133012, fraction_changed=0.037588, lr=0.00080220801; epoch 70: total=0.482346, KL=0.353490, recon=0.128855, fraction_changed=0.022442, lr=0.00073662526; epoch 75: total=0.488611, KL=0.359543, recon=0.129067, fraction_changed=0.013387, lr=0.00070158821; epoch 80: total=0.494123, KL=0.363391, recon=0.130732, fraction_changed=0.023631, lr=0.00066533912; epoch 90: total=0.513881, KL=0.368559, recon=0.145322, fraction_changed=0.036155, lr=0.0005901049; epoch 100: total=0.506682, KL=0.369614, recon=0.137068, fraction_changed=0.012166, lr=0.00051277512; epoch 110: total=0.505773, KL=0.368456, recon=0.137317, fraction_changed=0.007996, lr=0.00043525389; epoch 120: total=0.505040, KL=0.367207, recon=0.137832, fraction_changed=0.005521, lr=0.00035945004; epoch 125: total=0.505050, KL=0.366630, recon=0.138420, fraction_changed=0.003664, lr=0.00032277835; epoch 130: total=0.504834, KL=0.366072, recon=0.138762, fraction_changed=0.003404, lr=0.00028723011; epoch 140: total=0.504817, KL=0.365468, recon=0.139349, fraction_changed=0.002541, lr=0.0002203724; epoch 150: total=0.504948, KL=0.365210, recon=0.139737, fraction_changed=0.001791, lr=0.00016052317; epoch 158: total=0.505026, KL=0.365084, recon=0.139942, fraction_changed=0.000896, lr=0.00011868695.
- All-100-center diagnostics (epoch 0 and every 25 epochs or stop): epoch 0: min=355, max=939, median=608.0, below_1pct=54/100; epoch 25: min=197, max=1,952, median=528.0, below_1pct=60/100; epoch 50: min=46, max=2,742, median=501.0, below_1pct=67/100; epoch 75: min=2, max=4,683, median=321.5, below_1pct=73/100; epoch 100: min=0, max=6,925, median=185.5, below_1pct=77/100; epoch 125: min=0, max=8,172, median=142.5, below_1pct=77/100; epoch 150: min=0, max=8,640, median=116.5, below_1pct=78/100; epoch 158: min=0, max=8,710, median=111.0, below_1pct=79/100.

### Variant B (noise persists), seed 42 (67 surviving centers)

- Pretraining reconstruction and LR: epoch 10: recon=0.183434, lr=0.00098034537; epoch 20: recon=0.144672, lr=0.00091440488; epoch 30: recon=0.130178, lr=0.00080838899; epoch 40: recon=0.119873, lr=0.00067267527; epoch 50: recon=0.113449, lr=0.00052054833; epoch 60: recon=0.108926, lr=0.0003668994; epoch 70: recon=0.106330, lr=0.00022676873; epoch 80: recon=0.104900, lr=0.00011387327; epoch 90: recon=0.104203, lr=3.9264019e-05; epoch 100: recon=0.103957, lr=1.0244253e-05.
- DEC stopped via **stability criterion** at epoch **161** (threshold `fraction_changed < 0.001`, ceiling 500).
- Joint DEC losses and LR: epoch 1: total=0.251309, KL=0.147385, recon=0.103925, fraction_changed=0.511368, lr=0.001; epoch 10: total=0.306089, KL=0.119817, recon=0.186272, fraction_changed=0.141396, lr=0.00099506171; epoch 20: total=0.373220, KL=0.212524, recon=0.160695, fraction_changed=0.038794, lr=0.00097811754; epoch 25: total=0.384635, KL=0.236026, recon=0.148609, fraction_changed=0.026155, lr=0.00096523936; epoch 30: total=0.400530, KL=0.259022, recon=0.141508, fraction_changed=0.026563, lr=0.00094952365; epoch 40: total=0.428133, KL=0.295427, recon=0.132706, fraction_changed=0.018908, lr=0.00090998411; epoch 50: total=0.456379, KL=0.324033, recon=0.132346, fraction_changed=0.059900, lr=0.00086047252; epoch 60: total=0.475795, KL=0.344513, recon=0.131282, fraction_changed=0.028452, lr=0.00080220801; epoch 70: total=0.490184, KL=0.358300, recon=0.131884, fraction_changed=0.016074, lr=0.00073662526; epoch 75: total=0.499923, KL=0.362107, recon=0.137816, fraction_changed=0.048321, lr=0.00070158821; epoch 80: total=0.506049, KL=0.364767, recon=0.141283, fraction_changed=0.048647, lr=0.00066533912; epoch 90: total=0.507217, KL=0.366856, recon=0.140362, fraction_changed=0.017817, lr=0.0005901049; epoch 100: total=0.509356, KL=0.366028, recon=0.143328, fraction_changed=0.011058, lr=0.00051277512; epoch 110: total=0.510932, KL=0.364519, recon=0.146413, fraction_changed=0.007329, lr=0.00043525389; epoch 120: total=0.513536, KL=0.363769, recon=0.149767, fraction_changed=0.008501, lr=0.00035945004; epoch 125: total=0.514284, KL=0.363618, recon=0.150666, fraction_changed=0.006010, lr=0.00032277835; epoch 130: total=0.514550, KL=0.363479, recon=0.151071, fraction_changed=0.003583, lr=0.00028723011; epoch 140: total=0.516017, KL=0.363484, recon=0.152533, fraction_changed=0.001922, lr=0.0002203724; epoch 150: total=0.517084, KL=0.363617, recon=0.153467, fraction_changed=0.001612, lr=0.00016052317; epoch 160: total=0.518066, KL=0.363827, recon=0.154239, fraction_changed=0.001140, lr=0.00010915609; epoch 161: total=0.518143, KL=0.363853, recon=0.154290, fraction_changed=0.000896, lr=0.00010453659.
- All-100-center diagnostics (epoch 0 and every 25 epochs or stop): epoch 0: min=355, max=939, median=608.0, below_1pct=54/100; epoch 25: min=198, max=1,670, median=537.5, below_1pct=61/100; epoch 50: min=39, max=3,099, median=480.0, below_1pct=69/100; epoch 75: min=0, max=5,375, median=282.0, below_1pct=76/100; epoch 100: min=0, max=7,299, median=185.5, below_1pct=77/100; epoch 125: min=0, max=8,163, median=136.5, below_1pct=78/100; epoch 150: min=0, max=8,488, median=110.5, below_1pct=79/100; epoch 161: min=0, max=8,543, median=101.5, below_1pct=79/100.

### Final results

| variant | seed | split | n surviving | emotion AMI | genre AMI | silhouette (128-D Z) | verdict | held-out Pareto bar | surviving min | surviving max | surviving median | surviving below 1% | all-100 min | all-100 max | all-100 median | all-100 below 1% |
|---|---:|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A (pretrain-only noise) | 42 | train | 67 | 0.1330 | 0.4008 | 0.2298 | Collapsed | n/a (train split) | 0 | 11,413 | 109.0 | 46/67 | 0 | 8,710 | 111.0 | 79/100 |
| A (pretrain-only noise) | 42 | held-out | 67 | 0.1097 | 0.3764 | 0.2224 | Collapsed | does not clear | 0 | 1,668 | 13.0 | 45/67 | 0 | 8,710 | 111.0 | 79/100 |
| B (noise persists) | 42 | train | 67 | 0.1312 | 0.4032 | 0.2541 | Collapsed | n/a (train split) | 0 | 11,147 | 104.0 | 46/67 | 0 | 8,543 | 101.5 | 79/100 |
| B (noise persists) | 42 | held-out | 67 | 0.1078 | 0.3886 | 0.2405 | Collapsed | does not clear | 0 | 1,637 | 14.0 | 45/67 | 0 | 8,543 | 101.5 | 79/100 |

Surviving-center collapse numbers are split-specific and now use each run's own data-dependent surviving count (see the 'n surviving' column), not a fixed 67. The all-100 numbers repeat the final train checkpoint before pruning.

## Phase 2

Skipped: neither variant cleared the held-out Pareto bar in Phase 1.

## Decision

- **Variant A: Collapsed.** Phase 1 missed the held-out Pareto bar; Phase 2 was not run for this variant.
- **Variant B: Collapsed.** Phase 1 missed the held-out Pareto bar; Phase 2 was not run for this variant.

## Comparison to the original (buggy) faithful-recipe result

Original (Variant A, seed 42, held-out): emotion AMI 0.1092, genre AMI 0.3288, silhouette 0.5120, 67 fixed surviving centers, 50/67 below 1% occupancy, 21/67 empty. Compare directly against this run's own Phase 1 table above, which reports its own data-dependent surviving count and occupancy under the corrected pruning and loss scale.
