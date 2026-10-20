Diagnostics only; all computations followed the saved verdict.

| Row | Aspect | Seeds | k20 AMI | Leiden 3a AMI | Leiden count | k-means at c AMI | Leiden 3b AMI | Fit k20 AMI |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| K1 | emotion | 5 | 0.1001 | 0.0766 | 13.5333 | 0.1002 | 0.0727 | 0.1080 |
| B1 | emotion | 5 | 0.0779 | 0.0748 | 18.8000 | 0.0778 | 0.0745 | 0.0810 |
| B2 | emotion | 4 | 0.0532 | 0.0484 | 32.0000 | 0.0493 | 0.0476 | 0.0562 |
| B3 | emotion | 5 | 0.0628 | 0.0615 | 18.7333 | 0.0630 | 0.0623 | 0.0658 |
| B4 | emotion | 5 | 0.0693 | 0.0656 | 12.6667 | 0.0661 | 0.0666 | 0.0727 |
| B5 | emotion | 5 | 0.0652 | 0.0630 | 15.8667 | 0.0648 | 0.0629 | 0.0677 |
| K1 | style | 5 | 0.3719 | 0.3432 | 14.7333 | 0.3678 | 0.3457 | 0.3668 |
| B1 | style | 5 | 0.2283 | 0.2293 | 15.8667 | 0.2234 | 0.2332 | 0.2283 |
| K1 | genre | 5 | 0.4490 | 0.4785 | 18.9333 | 0.4496 | 0.4644 | 0.4592 |
| B1 | genre | 5 | 0.4411 | 0.4500 | 18.8000 | 0.4442 | 0.4272 | 0.4418 |
| target | emotion | 1 | 0.0921 | 0.0736 | 13.3333 | 0.0924 | — | — |
| encoder | emotion | 1 | 0.0708 | 0.0590 | 13.0000 | 0.0669 | — | — |
| target | style | 1 | 0.4026 | 0.3715 | 17.0000 | 0.3987 | — | — |
| encoder | style | 1 | 0.2233 | 0.2147 | 13.0000 | 0.2154 | — | — |
| target | genre | 1 | 0.5062 | 0.5048 | 18.0000 | 0.5043 | — | — |
| encoder | genre | 1 | 0.4521 | 0.4699 | 13.0000 | 0.4622 | — | — |

| Margin | d | Paired 95% interval |
|---|---:|---:|
| leiden_emotion | 0.0019 | [-0.0030, 0.0053] |
| style | 0.1436 | [0.1339, 0.1491] |
| genre | 0.0079 | [0.0002, 0.0153] |

All finite saved runs appear above, including collapsed paper runs; only the verdict’s eligible seeds enter margins.
JSON contains V, ARI, FMI and random baselines per partition, plus every paper run’s topic count, share, collapse and dev AUC.
