# ArtELingo learned two-teacher student — Stage 2

Generated automatically, 2026-09-22 20:32:07.

## Architecture and training

The student projects train-only-PCA-reduced, independently L2-normalized CLIP image/text content (50 dimensions) and raw 28-dimensional GoEmotions probabilities into a shared 32-dimensional space. A small scalar gate combines the two normalized projections for every node. It is trained with equally weighted symmetric in-batch InfoNCE losses from content and affect buddy-graph edges; neither teacher is tuned or reweighted. Held-out retrieval uses a fixed 2,000-node sample, and all PCA fitting is restricted to train paintings. This is a controlled capacity-only comparison against Stage 1: PCA dimensionality, shared embedding dimension, loss, training data and sampling, learning rate, epoch budget, checkpoint cadence, diagnostics, and Pareto bar are identical; only the projection heads change to small 2-layer MLPs with a 64-unit hidden layer.

## Checkpoint trajectory

| epoch | content held-out recall | affect held-out recall | content loss | affect loss | content gradient share | top-eigenvalue variance fraction | effective rank at 95% | gate mean | gate std | gate saturated fraction |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0735 | 0.0744 | — | — | — | 0.1880 | 20 | 0.4467 | 0.0019 | 0.0000 |
| 5 | 0.0623 | 0.1019 | 6.6713 | 6.5249 | 0.5263 | 0.2842 | 19 | 0.4378 | 0.0017 | 0.0000 |
| 10 | 0.0556 | 0.1192 | 6.4198 | 6.0159 | 0.4673 | 0.3624 | 17 | 0.4333 | 0.0019 | 0.0000 |
| 15 | 0.0608 | 0.1143 | 6.2843 | 5.2932 | 0.4760 | 0.3454 | 17 | 0.4392 | 0.0023 | 0.0000 |
| 20 | 0.0767 | 0.0855 | 5.7572 | 4.9527 | 0.5072 | 0.2609 | 18 | 0.4488 | 0.0028 | 0.0000 |
| 25 | 0.0875 | 0.0816 | 5.5526 | 5.2077 | 0.4792 | 0.2146 | 18 | 0.4492 | 0.0031 | 0.0000 |
| 30 | 0.0858 | 0.0959 | 5.4532 | 4.9848 | 0.5021 | 0.1734 | 18 | 0.4493 | 0.0034 | 0.0000 |
| 35 | 0.0825 | 0.1072 | 5.4859 | 4.7719 | 0.5026 | 0.1488 | 19 | 0.4530 | 0.0041 | 0.0000 |
| 40 | 0.0853 | 0.1105 | 5.4685 | 4.6836 | 0.5032 | 0.1362 | 20 | 0.4532 | 0.0045 | 0.0000 |
| 45 | 0.0904 | 0.1118 | 5.3605 | 4.6494 | 0.5042 | 0.1317 | 20 | 0.4536 | 0.0048 | 0.0000 |
| 50 | 0.0936 | 0.1119 | 5.2223 | 4.5079 | 0.4975 | 0.1295 | 21 | 0.4554 | 0.0054 | 0.0000 |
| 55 | 0.0984 | 0.1107 | 5.2515 | 4.3748 | 0.5117 | 0.1283 | 21 | 0.4559 | 0.0059 | 0.0000 |
| 60 | 0.0948 | 0.1195 | 5.0749 | 4.3863 | 0.4977 | 0.1245 | 22 | 0.4565 | 0.0065 | 0.0000 |
| 65 | 0.0926 | 0.1219 | 5.1094 | 4.2168 | 0.4932 | 0.1172 | 22 | 0.4632 | 0.0078 | 0.0000 |
| 70 | 0.0971 | 0.1211 | 5.2815 | 4.2454 | 0.5128 | 0.1134 | 23 | 0.4652 | 0.0085 | 0.0000 |
| 75 | 0.0973 | 0.1241 | 5.0080 | 4.1925 | 0.4985 | 0.1120 | 23 | 0.4658 | 0.0096 | 0.0000 |
| 80 | 0.0908 | 0.1360 | 5.1398 | 4.0480 | 0.5175 | 0.1095 | 24 | 0.4679 | 0.0107 | 0.0000 |
| 85 | 0.1012 | 0.1243 | 5.1101 | 4.0390 | 0.5096 | 0.1038 | 24 | 0.4734 | 0.0115 | 0.0000 |
| 90 | 0.0952 | 0.1373 | 5.0170 | 4.0754 | 0.5036 | 0.1008 | 24 | 0.4702 | 0.0127 | 0.0000 |
| 95 | 0.0964 | 0.1402 | 5.0541 | 3.9915 | 0.5038 | 0.0985 | 25 | 0.4749 | 0.0137 | 0.0000 |
| 100 | 0.1020 | 0.1379 | 5.0448 | 4.0666 | 0.4982 | 0.0944 | 25 | 0.4778 | 0.0149 | 0.0000 |
| 105 | 0.0992 | 0.1414 | 5.0875 | 4.0155 | 0.5020 | 0.0947 | 25 | 0.4804 | 0.0158 | 0.0000 |
| 110 | 0.1000 | 0.1435 | 4.9706 | 3.8683 | 0.5097 | 0.0955 | 26 | 0.4805 | 0.0172 | 0.0000 |
| 115 | 0.1008 | 0.1433 | 5.0026 | 3.8795 | 0.5035 | 0.0931 | 26 | 0.4818 | 0.0178 | 0.0000 |
| 120 | 0.1021 | 0.1435 | 4.8786 | 3.8776 | 0.5003 | 0.0913 | 26 | 0.4844 | 0.0187 | 0.0000 |
| 125 | 0.1036 | 0.1447 | 4.9563 | 3.8505 | 0.5004 | 0.0898 | 27 | 0.4848 | 0.0187 | 0.0000 |
| 130 | 0.0953 | 0.1556 | 4.9195 | 3.8015 | 0.5116 | 0.0896 | 27 | 0.4796 | 0.0191 | 0.0000 |
| 135 | 0.1008 | 0.1511 | 4.8235 | 3.8917 | 0.4816 | 0.0889 | 27 | 0.4853 | 0.0203 | 0.0000 |
| 140 | 0.1005 | 0.1485 | 4.8537 | 3.7878 | 0.4982 | 0.0856 | 27 | 0.4877 | 0.0209 | 0.0000 |
| 145 | 0.0971 | 0.1559 | 4.9943 | 3.7350 | 0.5130 | 0.0846 | 27 | 0.4844 | 0.0216 | 0.0000 |
| 150 | 0.0931 | 0.1641 | 4.8909 | 3.7977 | 0.4935 | 0.0842 | 27 | 0.4826 | 0.0215 | 0.0000 |
| 155 | 0.1018 | 0.1503 | 4.9114 | 3.8446 | 0.4920 | 0.0845 | 28 | 0.4873 | 0.0209 | 0.0000 |
| 160 | 0.1003 | 0.1577 | 4.9311 | 3.6872 | 0.5144 | 0.0834 | 28 | 0.4873 | 0.0215 | 0.0000 |
| 165 | 0.0983 | 0.1624 | 4.8952 | 3.6994 | 0.5053 | 0.0834 | 28 | 0.4872 | 0.0219 | 0.0000 |
| 170 | 0.0989 | 0.1632 | 4.9131 | 3.7497 | 0.5087 | 0.0822 | 28 | 0.4883 | 0.0216 | 0.0000 |
| 175 | 0.0981 | 0.1652 | 4.9752 | 3.6718 | 0.5072 | 0.0837 | 28 | 0.4853 | 0.0217 | 0.0000 |
| 180 | 0.0962 | 0.1720 | 4.7924 | 3.6660 | 0.5009 | 0.0815 | 28 | 0.4829 | 0.0218 | 0.0000 |
| 185 | 0.0987 | 0.1666 | 4.8934 | 3.6692 | 0.4936 | 0.0818 | 28 | 0.4906 | 0.0228 | 0.0000 |
| 190 | 0.0988 | 0.1694 | 4.9203 | 3.6386 | 0.5013 | 0.0801 | 28 | 0.4909 | 0.0234 | 0.0000 |
| 195 | 0.0920 | 0.1814 | 4.8638 | 3.6146 | 0.5044 | 0.0795 | 28 | 0.4862 | 0.0227 | 0.0000 |
| 200 | 0.1032 | 0.1656 | 4.7934 | 3.7358 | 0.4823 | 0.0803 | 29 | 0.4865 | 0.0224 | 0.0000 |

## Stopping and collapse determination

Training stopped at epoch 200 because **reached MAX_EPOCHS=200**.

**Merely a compromise.** Here, “first” means epoch 0 (pre-training, before any gradient steps). First/final held-out content recall: 0.0735 / 0.1032; affect recall: 0.0744 / 0.1656; final content gradient share: 0.4823; final gate saturated fraction: 0.0000.

## Final comparison

| signal | emotion AMI | emotion V-measure | genre AMI | genre V-measure |
|---|---:|---:|---:|---:|
| Content-only (reference) | 0.0593 | 0.0600 | 0.4384 | 0.4572 |
| GoEmotions-affect-only (reference) | 0.1180 | 0.1189 | 0.0396 | 0.0937 |
| Late fusion — union (reference) | 0.1236 | 0.1241 | 0.1394 | 0.1677 |
| Hierarchical refinement (reference) | 0.1072 | 0.1141 | 0.1954 | 0.3901 |
| Learned student, Stage 1 — linear heads (reference) | 0.1284 | 0.1289 | 0.2799 | 0.3014 |
| Learned student, Stage 2 (this run) | 0.1230 | 0.1235 | 0.2319 | 0.2553 |

## Held-out generalization

| signal | emotion AMI | emotion V-measure | genre AMI | genre V-measure |
|---|---:|---:|---:|---:|
| Learned student, Stage 2 — TRAIN | 0.1230 | 0.1235 | 0.2319 | 0.2553 |
| Learned student, Stage 1 — linear heads (reference) | 0.1095 | 0.1123 | 0.2901 | 0.4240 |
| Learned student, Stage 2 — HELD-OUT | 0.1046 | 0.1072 | 0.2087 | 0.3531 |

The absolute train-to-held-out drop is 0.0184 points for emotion AMI and 0.0231 points for genre AMI. 
## Pareto-bar verdict

**Cleared one.** Held-out results do not clear the same Pareto bar. 

Both teachers retained signal, but the partition did not clear both AMI targets; this indicates incompatible teacher constraints rather than insufficient capacity, so do not escalate architecture on this evidence alone.
