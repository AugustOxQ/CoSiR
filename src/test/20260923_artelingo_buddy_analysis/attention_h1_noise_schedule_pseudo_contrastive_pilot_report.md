# Attention-h1 noise, cosine schedule, and Leiden pseudo-contrastive pilot

Generated automatically, 2026-09-26 05:29:54.

## Method

The Attention-h1 student retains both teacher InfoNCE losses and adds the same-community symmetric InfoNCE term from the Leiden pseudo-contrastive pilot. All three training losses pass their embeddings through the noise/renormalization helper immediately before InfoNCE. The selected noise_std=0 makes that helper an identity in this run. Epoch-0, checkpoint diagnostics, reclustering, final Leiden, AMI, and silhouette use clean model forward passes. No noise value or cluster hyperparameter is screened in this pilot.

noise_std=0; recluster every 20 epochs starting at 1; LAMBDA_CLUSTER_MAX=1.0; CLUSTER_WARMUP_EPOCHS=50; DOMINANT_FRACTION_GUARD=0.8; batch=1024; temperature=0.1. Adam starts at 0.001; CosineAnnealingLR T_max=200, eta_min=1e-05; one step after each training epoch, never beyond T_max. LR rows show the rate used for that epoch's update. CHECKPOINT_EVERY=5. Recall-based plateau stopping matches the sibling pilots. Silhouette uses the pseudo pilot's seed-42 6,000-point draw and 4,000-point score sample.

## Seed 42 result

Held-out Pareto bar: emotion AMI > 0.1236 AND genre AMI > 0.1954. Seed 42 does not clear it.

| split | pre emotion AMI | pre genre AMI | pre silhouette | post emotion AMI | post genre AMI | post silhouette | pre Leiden communities | post Leiden communities | held-out Pareto bar |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| train | 0.0665 | 0.2734 | -0.0038 | 0.1326 | 0.2954 | 0.0603 | 20 | 18 | n/a |
| held-out | 0.0500 | 0.2234 | 0.0247 | 0.1210 | 0.2623 | 0.0789 | 13 | 15 | does not clear |

## Full training, LR, and loss trajectories

### Seed 42

Stopped: both recalls plateaued for 5 consecutive checkpoints.

| epoch | LR used | content loss | affect loss | cluster loss | cluster weight | pseudo communities | dominant fraction | content recall | affect recall | top eigen fraction | effective rank 95% | gate mean | gate std | gate saturated |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.00100000 | nan | nan | nan | 0.0000 | 0 | nan | 0.0717 | 0.0209 | 0.3338 | 12 | 0.4999 | 0.0029 | 0.0000 |
| 5 | 0.00099902 | 5.9570 | 6.4348 | 6.3995 | 0.1000 | 20 | 0.1357 | 0.0759 | 0.0270 | 0.3016 | 12 | 0.5008 | 0.0033 | 0.0000 |
| 10 | 0.00099506 | 5.5760 | 6.2160 | 6.6214 | 0.2000 | 20 | 0.1357 | 0.0802 | 0.0375 | 0.2213 | 12 | 0.5015 | 0.0036 | 0.0000 |
| 15 | 0.00098808 | 5.5610 | 5.8229 | 6.6222 | 0.3000 | 20 | 0.1357 | 0.0791 | 0.0493 | 0.1958 | 13 | 0.5023 | 0.0040 | 0.0000 |
| 20 | 0.00097812 | 5.4133 | 5.4697 | 6.9000 | 0.4000 | 20 | 0.1357 | 0.0811 | 0.0591 | 0.1861 | 13 | 0.5033 | 0.0043 | 0.0000 |
| 25 | 0.00096524 | 5.4269 | 5.3043 | 6.5096 | 0.5000 | 13 | 0.1569 | 0.0802 | 0.0692 | 0.1787 | 13 | 0.5041 | 0.0048 | 0.0000 |
| 30 | 0.00094952 | 5.4733 | 5.1031 | 6.4109 | 0.6000 | 13 | 0.1569 | 0.0759 | 0.0801 | 0.1782 | 13 | 0.5045 | 0.0053 | 0.0000 |
| 35 | 0.00093107 | 5.4665 | 5.0083 | 6.5369 | 0.7000 | 13 | 0.1569 | 0.0746 | 0.0886 | 0.1799 | 14 | 0.5045 | 0.0058 | 0.0000 |
| 40 | 0.00090998 | 5.3662 | 5.0658 | 6.4680 | 0.8000 | 13 | 0.1569 | 0.0745 | 0.0915 | 0.1835 | 14 | 0.5045 | 0.0061 | 0.0000 |
| 45 | 0.00088640 | 5.3850 | 5.0713 | 6.1854 | 0.9000 | 13 | 0.1324 | 0.0763 | 0.0930 | 0.1784 | 14 | 0.5044 | 0.0066 | 0.0000 |
| 50 | 0.00086047 | 5.2970 | 4.9944 | 6.2807 | 1.0000 | 13 | 0.1324 | 0.0773 | 0.0982 | 0.1710 | 14 | 0.5041 | 0.0072 | 0.0000 |
| 55 | 0.00083235 | 5.3837 | 4.8680 | 6.2417 | 1.0000 | 13 | 0.1324 | 0.0760 | 0.1019 | 0.1671 | 15 | 0.5035 | 0.0078 | 0.0000 |
| 60 | 0.00080221 | 5.2141 | 4.9118 | 6.2720 | 1.0000 | 13 | 0.1324 | 0.0774 | 0.1026 | 0.1658 | 15 | 0.5029 | 0.0085 | 0.0000 |
| 65 | 0.00077023 | 5.1564 | 4.9098 | 6.3088 | 1.0000 | 16 | 0.1320 | 0.0800 | 0.1022 | 0.1667 | 15 | 0.5023 | 0.0091 | 0.0000 |
| 70 | 0.00073663 | 5.2973 | 4.9336 | 6.1685 | 1.0000 | 16 | 0.1320 | 0.0817 | 0.1001 | 0.1692 | 15 | 0.5020 | 0.0096 | 0.0000 |
| 75 | 0.00070159 | 5.1414 | 4.8879 | 6.1583 | 1.0000 | 16 | 0.1320 | 0.0843 | 0.0986 | 0.1673 | 15 | 0.5015 | 0.0101 | 0.0000 |
| 80 | 0.00066534 | 5.1679 | 4.9191 | 6.2434 | 1.0000 | 16 | 0.1320 | 0.0846 | 0.0986 | 0.1639 | 15 | 0.5008 | 0.0106 | 0.0000 |
| 85 | 0.00062810 | 5.1774 | 4.9045 | 6.2390 | 1.0000 | 13 | 0.1570 | 0.0827 | 0.1021 | 0.1583 | 15 | 0.5000 | 0.0112 | 0.0000 |
| 90 | 0.00059010 | 5.1482 | 4.8338 | 6.3374 | 1.0000 | 13 | 0.1570 | 0.0818 | 0.1029 | 0.1556 | 16 | 0.4992 | 0.0118 | 0.0000 |
| 95 | 0.00055158 | 5.2255 | 4.8152 | 6.2584 | 1.0000 | 13 | 0.1570 | 0.0803 | 0.1060 | 0.1512 | 16 | 0.4983 | 0.0124 | 0.0000 |
| 100 | 0.00051278 | 5.2745 | 4.8304 | 6.2648 | 1.0000 | 13 | 0.1570 | 0.0801 | 0.1087 | 0.1495 | 16 | 0.4974 | 0.0129 | 0.0000 |
| 105 | 0.00047392 | 5.2468 | 4.8113 | 6.2690 | 1.0000 | 15 | 0.1281 | 0.0793 | 0.1075 | 0.1553 | 16 | 0.4968 | 0.0133 | 0.0000 |
| 110 | 0.00043525 | 5.2090 | 4.7330 | 6.2616 | 1.0000 | 15 | 0.1281 | 0.0797 | 0.1059 | 0.1584 | 16 | 0.4964 | 0.0137 | 0.0000 |
| 115 | 0.00039702 | 5.2611 | 4.7102 | 6.1993 | 1.0000 | 15 | 0.1281 | 0.0782 | 0.1057 | 0.1598 | 16 | 0.4962 | 0.0141 | 0.0000 |
| 120 | 0.00035945 | 5.2457 | 4.7100 | 6.1805 | 1.0000 | 15 | 0.1281 | 0.0768 | 0.1067 | 0.1590 | 16 | 0.4960 | 0.0144 | 0.0000 |
| 125 | 0.00032278 | 5.2709 | 4.6173 | 6.2957 | 1.0000 | 15 | 0.0945 | 0.0755 | 0.1093 | 0.1565 | 16 | 0.4960 | 0.0148 | 0.0000 |
| 130 | 0.00028723 | 5.2653 | 4.7151 | 6.2052 | 1.0000 | 15 | 0.0945 | 0.0744 | 0.1114 | 0.1542 | 17 | 0.4959 | 0.0152 | 0.0000 |
| 135 | 0.00025302 | 5.2510 | 4.6280 | 6.2425 | 1.0000 | 15 | 0.0945 | 0.0741 | 0.1136 | 0.1509 | 17 | 0.4957 | 0.0155 | 0.0000 |
| 140 | 0.00022037 | 5.2465 | 4.6293 | 6.2583 | 1.0000 | 15 | 0.0945 | 0.0738 | 0.1153 | 0.1482 | 17 | 0.4955 | 0.0158 | 0.0000 |
| 145 | 0.00018948 | 5.4083 | 4.5699 | 6.2735 | 1.0000 | 17 | 0.0914 | 0.0729 | 0.1168 | 0.1469 | 17 | 0.4954 | 0.0161 | 0.0000 |
| 150 | 0.00016052 | 5.2421 | 4.6010 | 6.2300 | 1.0000 | 17 | 0.0914 | 0.0733 | 0.1180 | 0.1464 | 17 | 0.4952 | 0.0163 | 0.0000 |
| 155 | 0.00013370 | 5.3408 | 4.6235 | 6.0814 | 1.0000 | 17 | 0.0914 | 0.0728 | 0.1192 | 0.1468 | 17 | 0.4950 | 0.0165 | 0.0000 |
| 160 | 0.00010916 | 5.2943 | 4.5860 | 6.1571 | 1.0000 | 17 | 0.0914 | 0.0727 | 0.1185 | 0.1470 | 17 | 0.4949 | 0.0166 | 0.0000 |
| 165 | 0.00008706 | 5.2939 | 4.5680 | 6.1868 | 1.0000 | 18 | 0.1379 | 0.0726 | 0.1190 | 0.1467 | 17 | 0.4948 | 0.0167 | 0.0000 |
| 170 | 0.00006754 | 5.3126 | 4.6196 | 6.1493 | 1.0000 | 18 | 0.1379 | 0.0725 | 0.1193 | 0.1460 | 17 | 0.4947 | 0.0168 | 0.0000 |
| 175 | 0.00005071 | 5.3873 | 4.5169 | 6.1598 | 1.0000 | 18 | 0.1379 | 0.0721 | 0.1194 | 0.1456 | 17 | 0.4947 | 0.0168 | 0.0000 |
| 180 | 0.00003669 | 5.3207 | 4.5390 | 6.2773 | 1.0000 | 18 | 0.1379 | 0.0721 | 0.1198 | 0.1454 | 17 | 0.4946 | 0.0169 | 0.0000 |

Successful recluster trajectory (rejected passes retain the prior labels): 0 skipped.

| epoch | communities | dominant fraction |
|---:|---:|---:|
| 1 | 20 | 0.1357 |
| 21 | 13 | 0.1569 |
| 41 | 13 | 0.1324 |
| 61 | 16 | 0.1320 |
| 81 | 13 | 0.1570 |
| 101 | 15 | 0.1281 |
| 121 | 15 | 0.0945 |
| 141 | 17 | 0.0914 |
| 161 | 18 | 0.1379 |

## Stress decision

Seed 42 missed the held-out Pareto bar. Stress was skipped; this is a plain miss. No cluster parameters were retuned.

## Reference comparison (seed 42)

| result | held-out emotion AMI | held-out genre AMI | held-out silhouette |
|---|---:|---:|---:|
| Attention-h1 baseline | 0.1249 | 0.2404 | 0.0392 |
| Leiden pseudo-contrastive alone | 0.1165 | 0.2351 | 0.0877 |
| Noise + schedule alone (winning seed 42) | 0.1306 | 0.1973 | 0.0488 |
| PercepT replication standing balance-hack | 0.1252 | 0.2486 | never measured |
| PercepT replication faithful recipe | 0.1092 | 0.3288 | 0.5120 |
| This combination | 0.1210 | 0.2623 | 0.0789 |

Noise + schedule alone, four-seed held-out results from its report:

| seed | emotion AMI | genre AMI | silhouette |
|---:|---:|---:|---:|
| 42 | 0.1306 | 0.1973 | 0.0488 |
| 7 | 0.1244 | 0.2583 | 0.0438 |
| 123 | 0.1334 | 0.2452 | 0.0487 |
| 2024 | 0.1222 | 0.2576 | 0.0451 |

## Final numeric verdict

Against Attention-h1 baseline: emotion AMI does not beat 0.1249 (-0.0039); genre AMI beats 0.2404 (+0.0219); silhouette beats 0.0392 (+0.0397).

Against Leiden pseudo-contrastive alone: emotion AMI beats 0.1165 (+0.0045); genre AMI beats 0.2351 (+0.0272); silhouette does not beat 0.0877 (-0.0088).

Against Noise + schedule alone (winning seed 42): emotion AMI does not beat 0.1306 (-0.0096); genre AMI beats 0.1973 (+0.0650); silhouette beats 0.0488 (+0.0301).

Against PercepT replication standing balance-hack: emotion AMI does not beat 0.1252 (-0.0042); genre AMI beats 0.2486 (+0.0137); silhouette not measured.

Against PercepT replication faithful recipe: emotion AMI beats 0.1092 (+0.0118); genre AMI does not beat 0.3288 (-0.0665); silhouette does not beat 0.5120 (-0.4331).

Combining beats only one individual buddy mechanism on held-out silhouette. The axis-by-axis comparisons above show AMI retention or loss against each individual buddy mechanism and whether either PercepT reference is beaten on a measured axis.
