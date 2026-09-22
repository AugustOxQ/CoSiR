# ArtELingo learned two-teacher student — Stage 1

Generated automatically, 2026-09-22 20:25:23.

## Architecture and training

The student projects train-only-PCA-reduced, independently L2-normalized CLIP image/text content (50 dimensions) and raw 28-dimensional GoEmotions probabilities into a shared 32-dimensional space. A small scalar gate combines the two normalized projections for every node. It is trained with equally weighted symmetric in-batch InfoNCE losses from content and affect buddy-graph edges; neither teacher is tuned or reweighted. Held-out retrieval uses a fixed 2,000-node sample, and all PCA fitting is restricted to train paintings.

## Checkpoint trajectory

| epoch | content held-out recall | affect held-out recall | content loss | affect loss | content gradient share | top-eigenvalue variance fraction | effective rank at 95% | gate mean | gate std | gate saturated fraction |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.1690 | 0.0325 | — | — | — | 0.1415 | 23 | 0.5135 | 0.0047 | 0.0000 |
| 5 | 0.1812 | 0.0321 | 5.8926 | 6.3563 | 0.7356 | 0.1423 | 23 | 0.5254 | 0.0046 | 0.0000 |
| 10 | 0.1922 | 0.0316 | 5.6527 | 6.3180 | 0.7187 | 0.1444 | 23 | 0.5343 | 0.0049 | 0.0000 |
| 15 | 0.1968 | 0.0337 | 5.4325 | 6.2443 | 0.6901 | 0.1473 | 23 | 0.5377 | 0.0055 | 0.0000 |
| 20 | 0.1907 | 0.0387 | 5.2399 | 6.0259 | 0.6671 | 0.1514 | 23 | 0.5325 | 0.0065 | 0.0000 |
| 25 | 0.1754 | 0.0466 | 5.1568 | 5.8442 | 0.6289 | 0.1582 | 23 | 0.5213 | 0.0076 | 0.0000 |
| 30 | 0.1525 | 0.0568 | 5.0901 | 5.5611 | 0.5932 | 0.1665 | 23 | 0.5073 | 0.0088 | 0.0000 |
| 35 | 0.1330 | 0.0701 | 5.1303 | 5.2952 | 0.5613 | 0.1733 | 23 | 0.4952 | 0.0098 | 0.0000 |
| 40 | 0.1232 | 0.0830 | 5.1888 | 5.1115 | 0.5389 | 0.1738 | 23 | 0.4909 | 0.0108 | 0.0000 |
| 45 | 0.1213 | 0.0861 | 5.1713 | 4.9962 | 0.5255 | 0.1684 | 23 | 0.4938 | 0.0117 | 0.0000 |
| 50 | 0.1240 | 0.0868 | 5.1070 | 4.8775 | 0.5217 | 0.1595 | 23 | 0.4997 | 0.0126 | 0.0000 |
| 55 | 0.1294 | 0.0850 | 5.0577 | 4.8504 | 0.5113 | 0.1497 | 24 | 0.5039 | 0.0133 | 0.0000 |
| 60 | 0.1300 | 0.0881 | 4.8696 | 4.8490 | 0.4945 | 0.1418 | 24 | 0.5015 | 0.0143 | 0.0000 |
| 65 | 0.1256 | 0.0922 | 4.9183 | 4.7533 | 0.4943 | 0.1349 | 24 | 0.4966 | 0.0154 | 0.0000 |
| 70 | 0.1248 | 0.0941 | 5.1026 | 4.6621 | 0.5221 | 0.1284 | 24 | 0.4932 | 0.0164 | 0.0000 |
| 75 | 0.1311 | 0.0922 | 4.8664 | 4.6458 | 0.5047 | 0.1223 | 25 | 0.4942 | 0.0174 | 0.0000 |
| 80 | 0.1304 | 0.0939 | 4.8797 | 4.6110 | 0.5143 | 0.1189 | 25 | 0.4926 | 0.0186 | 0.0000 |
| 85 | 0.1306 | 0.0963 | 4.8915 | 4.5565 | 0.5131 | 0.1158 | 25 | 0.4915 | 0.0199 | 0.0000 |
| 90 | 0.1318 | 0.0949 | 4.7861 | 4.5312 | 0.5065 | 0.1126 | 25 | 0.4931 | 0.0208 | 0.0000 |
| 95 | 0.1313 | 0.0977 | 4.8728 | 4.5748 | 0.4946 | 0.1103 | 25 | 0.4913 | 0.0218 | 0.0000 |
| 100 | 0.1321 | 0.0979 | 4.8908 | 4.5377 | 0.5108 | 0.1079 | 26 | 0.4913 | 0.0227 | 0.0000 |
| 105 | 0.1320 | 0.0982 | 4.8760 | 4.5175 | 0.5044 | 0.1066 | 26 | 0.4903 | 0.0235 | 0.0000 |
| 110 | 0.1311 | 0.0999 | 4.7646 | 4.3635 | 0.5116 | 0.1058 | 26 | 0.4890 | 0.0245 | 0.0000 |
| 115 | 0.1331 | 0.0994 | 4.8359 | 4.3773 | 0.5136 | 0.1047 | 26 | 0.4894 | 0.0251 | 0.0000 |
| 120 | 0.1371 | 0.0981 | 4.6793 | 4.3725 | 0.5007 | 0.1033 | 26 | 0.4902 | 0.0254 | 0.0000 |
| 125 | 0.1344 | 0.1008 | 4.7743 | 4.3222 | 0.5069 | 0.1025 | 26 | 0.4890 | 0.0261 | 0.0000 |
| 130 | 0.1328 | 0.1028 | 4.7034 | 4.3764 | 0.5067 | 0.1018 | 27 | 0.4876 | 0.0265 | 0.0000 |
| 135 | 0.1341 | 0.1035 | 4.7336 | 4.3552 | 0.4997 | 0.1008 | 27 | 0.4874 | 0.0273 | 0.0000 |
| 140 | 0.1304 | 0.1093 | 4.7140 | 4.3168 | 0.5007 | 0.0997 | 27 | 0.4851 | 0.0281 | 0.0000 |
| 145 | 0.1303 | 0.1092 | 4.8730 | 4.2112 | 0.5157 | 0.0985 | 27 | 0.4874 | 0.0287 | 0.0000 |
| 150 | 0.1367 | 0.1046 | 4.6379 | 4.3148 | 0.4948 | 0.0966 | 27 | 0.4923 | 0.0291 | 0.0000 |
| 155 | 0.1316 | 0.1104 | 4.7619 | 4.3197 | 0.5009 | 0.0962 | 27 | 0.4872 | 0.0294 | 0.0000 |
| 160 | 0.1353 | 0.1083 | 4.7097 | 4.2711 | 0.5077 | 0.0949 | 27 | 0.4900 | 0.0301 | 0.0000 |
| 165 | 0.1373 | 0.1085 | 4.6681 | 4.2561 | 0.5005 | 0.0945 | 27 | 0.4913 | 0.0310 | 0.0000 |
| 170 | 0.1304 | 0.1182 | 4.6999 | 4.2635 | 0.5092 | 0.0948 | 27 | 0.4856 | 0.0311 | 0.0000 |
| 175 | 0.1336 | 0.1141 | 4.8032 | 4.1962 | 0.5090 | 0.0942 | 28 | 0.4881 | 0.0312 | 0.0000 |
| 180 | 0.1368 | 0.1094 | 4.5742 | 4.2794 | 0.5001 | 0.0931 | 28 | 0.4929 | 0.0314 | 0.0000 |
| 185 | 0.1332 | 0.1125 | 4.7579 | 4.2350 | 0.4971 | 0.0934 | 28 | 0.4914 | 0.0319 | 0.0000 |
| 190 | 0.1313 | 0.1148 | 4.7461 | 4.2017 | 0.5085 | 0.0925 | 28 | 0.4917 | 0.0330 | 0.0000 |
| 195 | 0.1315 | 0.1139 | 4.6435 | 4.2351 | 0.4964 | 0.0910 | 28 | 0.4938 | 0.0340 | 0.0000 |
| 200 | 0.1314 | 0.1152 | 4.7189 | 4.1816 | 0.4988 | 0.0901 | 28 | 0.4926 | 0.0346 | 0.0000 |

## Stopping and collapse determination

Training stopped at epoch 200 because **reached MAX_EPOCHS=200**.

**Collapsed.** Here, “first” means epoch 0 (pre-training, before any gradient steps). First/final held-out content recall: 0.1690 / 0.1314; affect recall: 0.0325 / 0.1152; final content gradient share: 0.4988; final gate saturated fraction: 0.0000.

## Final comparison

| signal | emotion AMI | emotion V-measure | genre AMI | genre V-measure |
|---|---:|---:|---:|---:|
| Content-only (reference) | 0.0593 | 0.0600 | 0.4384 | 0.4572 |
| GoEmotions-affect-only (reference) | 0.1180 | 0.1189 | 0.0396 | 0.0937 |
| Late fusion — union (reference) | 0.1236 | 0.1241 | 0.1394 | 0.1677 |
| Hierarchical refinement (reference) | 0.1072 | 0.1141 | 0.1954 | 0.3901 |
| Learned student, Stage 1 (this run) | 0.1284 | 0.1289 | 0.2799 | 0.3014 |

## Held-out generalization

| signal | emotion AMI | emotion V-measure | genre AMI | genre V-measure |
|---|---:|---:|---:|---:|
| Learned student — TRAIN | 0.1284 | 0.1289 | 0.2799 | 0.3014 |
| Learned student — HELD-OUT | 0.1095 | 0.1123 | 0.2901 | 0.4240 |

The absolute train-to-held-out drop is 0.0189 points for emotion AMI and 0.0102 points for genre AMI. 
## Pareto-bar verdict

**Cleared both.** Held-out results do not clear the same Pareto bar. Because train results did clear it, this is evidence of the same train/held-out generalization gap seen in the BERT pilot, not a new failure mode.

At least one teacher failed the predeclared development/balance/collapse criterion; inspect the trajectory before considering a richer architecture.
