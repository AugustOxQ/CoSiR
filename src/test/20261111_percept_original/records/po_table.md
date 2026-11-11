# r26 PercepT (CLIP L/14 + ModernBERT-GoEmotions, paper settings): held AMI

Held (test) paintings: 12281 (genre labelled: 9,949). Arm L1: Stage 2 attention pooling on CLIP L/14 patch tokens. Cell = mean over uncollapsed seeds (and 3 k-means states). Smoke: False.
Class counts for rclass: {'emotion': 9, 'style': 27, 'genre': 10}.

## Cut k40: 5 of 5 seeds in the mean (collapsed: 0, seeds [])

| reading | emotion | style | genre |
|---|---:|---:|---:|
| rclass (k = class count: 9 / 27 / 10), paper row | 0.064 | 0.262 | 0.474 |
| r20 (k = 20) | 0.068 | 0.256 | 0.460 |
| own (argmax) | 0.076 | 0.251 | 0.419 |

Full precision:

- rclass: emotion 0.06358430182818484, style 0.26217891118293324, genre 0.4744370236399559
- r20: emotion 0.06778573034359733, style 0.256385112816235, genre 0.460050871367174
- own: emotion 0.0761861331065032, style 0.25109937768915225, genre 0.41853470674784254

Kept topic counts per seed: {0: 40, 1: 40, 2: 40, 3: 40, 4: 40}; mean dev AUC (uncollapsed): 0.9654068857918283

| seed | collapsed | share (top topic) | K | dev AUC | rclass emo | rclass sty | rclass gen | r20 emo | own emo |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | False | 0.072 | 40 | 0.968 | 0.060 | 0.253 | 0.512 | 0.066 | 0.077 |
| 1 | False | 0.051 | 40 | 0.964 | 0.064 | 0.276 | 0.471 | 0.069 | 0.075 |
| 2 | False | 0.061 | 40 | 0.965 | 0.062 | 0.250 | 0.473 | 0.068 | 0.077 |
| 3 | False | 0.058 | 40 | 0.966 | 0.065 | 0.269 | 0.460 | 0.068 | 0.072 |
| 4 | False | 0.056 | 40 | 0.964 | 0.067 | 0.262 | 0.457 | 0.068 | 0.079 |

## Cut strict: 0 of 5 seeds in the mean (collapsed: 5, seeds [0, 1, 2, 3, 4])

| reading | emotion | style | genre |
|---|---:|---:|---:|
| rclass (k = class count: 9 / 27 / 10), paper row | n/a | n/a | n/a |
| r20 (k = 20) | n/a | n/a | n/a |
| own (argmax) | n/a | n/a | n/a |

Full precision:

- rclass: emotion n/a, style n/a, genre n/a
- r20: emotion n/a, style n/a, genre n/a
- own: emotion n/a, style n/a, genre n/a

Kept topic counts per seed: {0: 96, 1: 99, 2: 1, 3: 99, 4: 98}; mean dev AUC (uncollapsed): n/a

| seed | collapsed | share (top topic) | K | dev AUC | rclass emo | rclass sty | rclass gen | r20 emo | own emo |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | True | 0.031 | 96 | 0.971 | 0.060 | 0.267 | 0.477 | 0.064 | 0.077 |
| 1 | True | 0.018 | 99 | 0.968 | 0.059 | 0.279 | 0.478 | 0.064 | 0.077 |
| 2 | True | 1.000 | 1 | n/a | n/a | n/a | n/a | n/a | n/a |
| 3 | True | 0.019 | 99 | 0.971 | 0.066 | 0.277 | 0.482 | 0.068 | 0.075 |
| 4 | True | 0.019 | 98 | 0.970 | 0.065 | 0.270 | 0.479 | 0.066 | 0.078 |

