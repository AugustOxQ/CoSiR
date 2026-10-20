Diagnostics only; all computations followed the saved verdict.

| Row | Aspect | Seeds | k20 AMI | Leiden 3a AMI | Leiden count | k-means at c AMI | Leiden 3b AMI | Fit k20 AMI |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| K1 | emotion | 5 | 0.1008 | 0.0757 | 12.6000 | 0.0980 | 0.0678 | 0.1047 |
| B1 | emotion | 5 | 0.0799 | 0.0762 | 19.1333 | 0.0789 | 0.0748 | 0.0802 |
| B2 | emotion | 2 | 0.0565 | 0.0535 | 20.5000 | 0.0543 | 0.0526 | 0.0591 |
| B3 | emotion | 3 | 0.0606 | 0.0594 | 21.1111 | 0.0601 | 0.0598 | 0.0604 |
| B4 | emotion | 5 | 0.0699 | 0.0662 | 12.3333 | 0.0666 | 0.0660 | 0.0715 |
| B5 | emotion | 5 | 0.0644 | 0.0616 | 17.6000 | 0.0636 | 0.0625 | 0.0652 |
| K1 | style | 5 | 0.3685 | 0.3394 | 15.8000 | 0.3638 | 0.3402 | 0.3640 |
| B1 | style | 5 | 0.2280 | 0.2305 | 16.0000 | 0.2242 | 0.2322 | 0.2267 |
| K1 | genre | 5 | 0.4438 | 0.4824 | 16.7333 | 0.4396 | 0.4465 | 0.4569 |
| B1 | genre | 5 | 0.4230 | 0.4241 | 19.1333 | 0.4226 | 0.4200 | 0.4238 |
| target | emotion | 1 | 0.0921 | 0.0736 | 13.3333 | 0.0924 | — | — |
| encoder | emotion | 1 | 0.0708 | 0.0590 | 13.0000 | 0.0669 | — | — |
| target | style | 1 | 0.4026 | 0.3715 | 17.0000 | 0.3987 | — | — |
| encoder | style | 1 | 0.2233 | 0.2147 | 13.0000 | 0.2154 | — | — |
| target | genre | 1 | 0.5062 | 0.5048 | 18.0000 | 0.5043 | — | — |
| encoder | genre | 1 | 0.4521 | 0.4699 | 13.0000 | 0.4622 | — | — |

| Margin | d | Paired 95% interval |
|---|---:|---:|
| leiden_emotion | -0.0005 | [-0.0048, 0.0023] |
| style | 0.1388 | [0.1283, 0.1427] |
| genre | 0.0208 | [0.0103, 0.0310] |

All finite saved runs appear above, including collapsed paper runs; only the verdict’s eligible seeds enter margins.
JSON contains V, ARI, FMI and random baselines per partition, plus every paper run’s topic count, share, collapse and dev AUC.
