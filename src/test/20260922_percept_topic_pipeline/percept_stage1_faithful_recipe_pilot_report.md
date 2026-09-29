# ArtELingo PercepT Stage 1 faithful recipe pilot

Generated automatically, 2026-09-24 22:30:39.

## Controlled setup

Train-only input is the base pilot's 2816-dimensional fused vector: independently normalized CLIP content and GoEmotions-RoBERTa affect embeddings, with repeated content to preserve a 2:1 norm budget. The base 128-D autoencoder, Student's-t assignments, self-sharpened target, and 67 highest-norm of 100-center pruning are unchanged. RoBERTa and concatenation remain the documented adaptations to the paper's ModernBERT-family encoder and elementwise fusion.

Both variants reconstruct clean h from a latent z corrupted with Gaussian standard deviation 0.1 during pretraining. Variant A uses clean z for joint reconstruction; variant B continues the same corruption for joint reconstruction only. KL always uses clean z. Appendix E calls the reconstruction objective MSE, whereas the displayed Section 4.1 equation uses an unsquared L2 norm; this pilot follows Appendix E's MSE convention. No balance term is used.

Both phases use Adam at initial LR 1e-3 with cosine annealing to 1e-05: pretraining T_max=100 for 100 epochs, joint DEC T_max=200, held at the floor after epoch 200. Joint DEC retains the project's `fraction_changed < 0.001` stop rule and 500-epoch ceiling rather than a fixed 200 epochs. Reconstruction weight is 1.

## Predeclared success criterion

**Held-out Pareto bar = emotion AMI > 0.1236 AND genre AMI > 0.1954, both simultaneously.** A Real success also requires non-collapse under the established surviving-center rule.

Silhouette is computed on the final 128-D latent Z using the same surviving-center hard assignments as AMI, separately for train and held-out. The paper reports 0.97 for its full method; this pilot reports its own measured values without treating that reference number as the held-out decision rule. If all samples share one cluster, silhouette is undefined and reported as n/a.

## Phase 1: seed 42, both variants

### Training and cluster-size trajectories

### Variant A (pretrain-only noise), seed 42

- Pretraining reconstruction and LR: epoch 10: recon=0.000091, lr=0.00098034537; epoch 20: recon=0.000073, lr=0.00091440488; epoch 30: recon=0.000059, lr=0.00080838899; epoch 40: recon=0.000053, lr=0.00067267527; epoch 50: recon=0.000050, lr=0.00052054833; epoch 60: recon=0.000048, lr=0.0003668994; epoch 70: recon=0.000047, lr=0.00022676873; epoch 80: recon=0.000046, lr=0.00011387327; epoch 90: recon=0.000045, lr=3.9264019e-05; epoch 100: recon=0.000045, lr=1.0244253e-05.
- DEC stopped via **stability criterion** at epoch **154** (threshold `fraction_changed < 0.001`, ceiling 500).
- Joint DEC losses and LR: epoch 1: total=0.241904, KL=0.241859, recon=0.000045, fraction_changed=0.651689, lr=0.001; epoch 10: total=0.318070, KL=0.317980, recon=0.000090, fraction_changed=0.264161, lr=0.00099506171; epoch 20: total=0.380078, KL=0.379999, recon=0.000079, fraction_changed=0.091105, lr=0.00097811754; epoch 25: total=0.398222, KL=0.398147, recon=0.000075, fraction_changed=0.071203, lr=0.00096523936; epoch 30: total=0.402433, KL=0.402360, recon=0.000074, fraction_changed=0.048044, lr=0.00094952365; epoch 40: total=0.398013, KL=0.397939, recon=0.000074, fraction_changed=0.070877, lr=0.00090998411; epoch 50: total=0.399096, KL=0.399020, recon=0.000076, fraction_changed=0.078287, lr=0.00086047252; epoch 60: total=0.401793, KL=0.401716, recon=0.000077, fraction_changed=0.049184, lr=0.00080220801; epoch 70: total=0.405723, KL=0.405646, recon=0.000077, fraction_changed=0.036432, lr=0.00073662526; epoch 75: total=0.404978, KL=0.404900, recon=0.000078, fraction_changed=0.029022, lr=0.00070158821; epoch 80: total=0.410313, KL=0.410235, recon=0.000078, fraction_changed=0.046578, lr=0.00066533912; epoch 90: total=0.408021, KL=0.407943, recon=0.000078, fraction_changed=0.014104, lr=0.0005901049; epoch 100: total=0.406627, KL=0.406549, recon=0.000079, fraction_changed=0.008664, lr=0.00051277512; epoch 110: total=0.405463, KL=0.405384, recon=0.000079, fraction_changed=0.012426, lr=0.00043525389; epoch 120: total=0.401290, KL=0.401210, recon=0.000079, fraction_changed=0.004072, lr=0.00035945004; epoch 125: total=0.399463, KL=0.399383, recon=0.000079, fraction_changed=0.004642, lr=0.00032277835; epoch 130: total=0.397852, KL=0.397773, recon=0.000080, fraction_changed=0.004234, lr=0.00028723011; epoch 140: total=0.394931, KL=0.394851, recon=0.000080, fraction_changed=0.002410, lr=0.0002203724; epoch 150: total=0.392401, KL=0.392321, recon=0.000080, fraction_changed=0.001661, lr=0.00016052317; epoch 154: total=0.391494, KL=0.391414, recon=0.000080, fraction_changed=0.000993, lr=0.00013888261.
- All-100-center diagnostics (epoch 0 and every 25 epochs or stop): epoch 0: min=413, max=949, median=604.0, below_1pct=52/100; epoch 25: min=0, max=4,587, median=286.5, below_1pct=67/100; epoch 50: min=0, max=10,379, median=152.0, below_1pct=73/100; epoch 75: min=0, max=10,657, median=74.5, below_1pct=76/100; epoch 100: min=0, max=9,947, median=44.5, below_1pct=76/100; epoch 125: min=0, max=9,571, median=27.5, below_1pct=77/100; epoch 150: min=0, max=9,506, median=28.0, below_1pct=77/100; epoch 154: min=0, max=9,503, median=28.0, below_1pct=77/100.

### Variant B (noise persists), seed 42

- Pretraining reconstruction and LR: epoch 10: recon=0.000091, lr=0.00098034537; epoch 20: recon=0.000073, lr=0.00091440488; epoch 30: recon=0.000059, lr=0.00080838899; epoch 40: recon=0.000053, lr=0.00067267527; epoch 50: recon=0.000050, lr=0.00052054833; epoch 60: recon=0.000048, lr=0.0003668994; epoch 70: recon=0.000047, lr=0.00022676873; epoch 80: recon=0.000046, lr=0.00011387327; epoch 90: recon=0.000045, lr=3.9264019e-05; epoch 100: recon=0.000045, lr=1.0244253e-05.
- DEC stopped via **stability criterion** at epoch **158** (threshold `fraction_changed < 0.001`, ceiling 500).
- Joint DEC losses and LR: epoch 1: total=0.241918, KL=0.241872, recon=0.000045, fraction_changed=0.650321, lr=0.001; epoch 10: total=0.318220, KL=0.318133, recon=0.000086, fraction_changed=0.265089, lr=0.00099506171; epoch 20: total=0.380218, KL=0.380140, recon=0.000077, fraction_changed=0.091430, lr=0.00097811754; epoch 25: total=0.398206, KL=0.398132, recon=0.000074, fraction_changed=0.069949, lr=0.00096523936; epoch 30: total=0.402432, KL=0.402359, recon=0.000073, fraction_changed=0.051008, lr=0.00094952365; epoch 40: total=0.398167, KL=0.398093, recon=0.000074, fraction_changed=0.072636, lr=0.00090998411; epoch 50: total=0.404602, KL=0.404526, recon=0.000076, fraction_changed=0.102244, lr=0.00086047252; epoch 60: total=0.401036, KL=0.400959, recon=0.000077, fraction_changed=0.075812, lr=0.00080220801; epoch 70: total=0.403737, KL=0.403659, recon=0.000078, fraction_changed=0.043614, lr=0.00073662526; epoch 75: total=0.405943, KL=0.405865, recon=0.000078, fraction_changed=0.012687, lr=0.00070158821; epoch 80: total=0.406977, KL=0.406899, recon=0.000078, fraction_changed=0.029462, lr=0.00066533912; epoch 90: total=0.407141, KL=0.407062, recon=0.000079, fraction_changed=0.013501, lr=0.0005901049; epoch 100: total=0.408323, KL=0.408244, recon=0.000079, fraction_changed=0.021367, lr=0.00051277512; epoch 110: total=0.404520, KL=0.404440, recon=0.000079, fraction_changed=0.009723, lr=0.00043525389; epoch 120: total=0.402816, KL=0.402736, recon=0.000080, fraction_changed=0.005114, lr=0.00035945004; epoch 125: total=0.400959, KL=0.400879, recon=0.000080, fraction_changed=0.005554, lr=0.00032277835; epoch 130: total=0.399764, KL=0.399684, recon=0.000080, fraction_changed=0.002948, lr=0.00028723011; epoch 140: total=0.397268, KL=0.397188, recon=0.000080, fraction_changed=0.002199, lr=0.0002203724; epoch 150: total=0.395432, KL=0.395351, recon=0.000080, fraction_changed=0.001189, lr=0.00016052317; epoch 158: total=0.394216, KL=0.394135, recon=0.000080, fraction_changed=0.000765, lr=0.00011868695.
- All-100-center diagnostics (epoch 0 and every 25 epochs or stop): epoch 0: min=404, max=951, median=603.5, below_1pct=52/100; epoch 25: min=0, max=4,560, median=295.5, below_1pct=68/100; epoch 50: min=0, max=10,362, median=141.0, below_1pct=76/100; epoch 75: min=0, max=10,614, median=68.0, below_1pct=77/100; epoch 100: min=0, max=9,980, median=46.0, below_1pct=77/100; epoch 125: min=0, max=9,689, median=31.5, below_1pct=77/100; epoch 150: min=0, max=9,524, median=26.0, below_1pct=77/100; epoch 158: min=0, max=9,506, median=24.0, below_1pct=77/100.

### Final results

| variant | seed | split | emotion AMI | genre AMI | silhouette (128-D Z) | verdict | held-out Pareto bar | surviving min | surviving max | surviving median | surviving below 1% | all-100 min | all-100 max | all-100 median | all-100 below 1% |
|---|---:|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A (pretrain-only noise) | 42 | train | 0.1373 | 0.3902 | 0.5291 | Collapsed | n/a (train split) | 0 | 9,503 | 69.0 | 47/67 | 0 | 9,503 | 28.0 | 77/100 |
| A (pretrain-only noise) | 42 | held-out | 0.1092 | 0.3288 | 0.5120 | Collapsed | does not clear | 0 | 1,251 | 13.0 | 50/67 | 0 | 9,503 | 28.0 | 77/100 |
| B (noise persists) | 42 | train | 0.1364 | 0.3909 | 0.5149 | Collapsed | n/a (train split) | 0 | 9,506 | 80.0 | 48/67 | 0 | 9,506 | 24.0 | 77/100 |
| B (noise persists) | 42 | held-out | 0.1086 | 0.3412 | 0.4999 | Collapsed | does not clear | 0 | 1,287 | 11.0 | 50/67 | 0 | 9,506 | 24.0 | 77/100 |

Surviving-center collapse numbers are split-specific. The all-100 numbers repeat the final train checkpoint before pruning.

## Phase 2

Skipped: neither variant cleared the held-out Pareto bar in Phase 1. Both are plain misses at seed 42; no other noise strength or LR schedule was tested.

## Decision

- **Variant A: Collapsed.** Phase 1 missed the held-out Pareto bar; Phase 2 was not run for this variant.
- **Variant B: Collapsed.** Phase 1 missed the held-out Pareto bar; Phase 2 was not run for this variant.
