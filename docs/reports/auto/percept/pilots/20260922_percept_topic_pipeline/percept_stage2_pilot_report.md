# ArtELingo PercepT Stage 2 patch-attention pilot

Generated automatically, 2026-09-23 02:54:48.

## Stage-1 K=60/40 seed-42 reproduction

| metric | established | re-fit | absolute difference | status |
|---|---:|---:|---:|---|
| held-out emotion AMI | 0.1238 | 0.1238 | 0.0000 | reproduced |
| held-out genre AMI | 0.2617 | 0.2617 | 0.0000 | reproduced |

## Frozen multi-label target statistics

| split | mean labels | median labels | max labels | fraction multi-labeled |
|---|---:|---:|---:|---:|
| train | 1.000 | 1.000 | 1 | 0.00% |
| held-out | 1.000 | 1.000 | 1 | 0.00% |

## Attention-pooling mapper training

The mapper consumes only cached `[painting, 50, 512]` patch tokens. A single learned query performs scaled dot-product attention over patches, followed by a linear 40-topic head and BCE-with-logits loss.

| epoch | full-batch BCE loss |
|---:|---:|
| 10 | 0.497633 |
| 20 | 0.352473 |
| 30 | 0.262015 |
| 40 | 0.208292 |
| 50 | 0.176370 |
| 60 | 0.156757 |
| 70 | 0.144108 |
| 80 | 0.135523 |
| 90 | 0.129420 |
| 100 | 0.124905 |

## Held-out per-topic AUC

| topic | mapper AUC | marginal-frequency baseline AUC |
|---:|---:|---:|
| 0 | 0.6854 | 0.5000 |
| 1 | 0.5818 | 0.5000 |
| 2 | 0.4817 | 0.5000 |
| 3 | 0.4499 | 0.5000 |
| 4 | 0.5246 | 0.5000 |
| 5 | 0.5736 | 0.5000 |
| 6 | 0.4728 | 0.5000 |
| 7 | 0.8744 | 0.5000 |
| 8 | 0.8353 | 0.5000 |
| 9 | 0.5538 | 0.5000 |
| 10 | 0.6756 | 0.5000 |
| 11 | 0.4775 | 0.5000 |
| 12 | 0.5881 | 0.5000 |
| 13 | 0.3618 | 0.5000 |
| 14 | 0.4275 | 0.5000 |
| 15 | 0.5611 | 0.5000 |
| 16 | 0.4602 | 0.5000 |
| 17 | 0.7585 | 0.5000 |
| 18 | 0.6860 | 0.5000 |
| 19 | 0.4388 | 0.5000 |
| 20 | 0.5990 | 0.5000 |
| 21 | 0.3854 | 0.5000 |
| 22 | 0.7799 | 0.5000 |
| 23 | 0.3871 | 0.5000 |
| 24 | 0.5198 | 0.5000 |
| 25 | 0.4006 | 0.5000 |
| 26 | 0.7350 | 0.5000 |
| 27 | 0.3896 | 0.5000 |
| 28 | 0.8220 | 0.5000 |
| 29 | 0.4423 | 0.5000 |
| 30 | 0.4402 | 0.5000 |
| 31 | 0.4177 | 0.5000 |
| 32 | 0.5045 | 0.5000 |
| 33 | 0.6354 | 0.5000 |
| 34 | 0.6072 | 0.5000 |
| 35 | 0.7851 | 0.5000 |
| 36 | 0.5534 | 0.5000 |
| 37 | 0.6150 | 0.5000 |
| 38 | 0.5003 | 0.5000 |
| 39 | 0.7705 | 0.5000 |

| scorer | macro AUC | min | median | max | skipped topics |
|---|---:|---:|---:|---:|---:|
| patch-attention mapper | 0.5690 | 0.3618 | 0.5536 | 0.8744 | 0 |
| train-marginal baseline | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0 |

## Conclusion

**The image-only attention-pooling mapper meaningfully beats the marginal-frequency baseline.** Its macro AUC is higher by 0.0690, exceeding the predeclared 0.01 practical margin.
