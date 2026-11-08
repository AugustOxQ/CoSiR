Read-out: R0 / none+k0, plain cosine on unit float32 rows; no map or query expansion. Seeds: [0, 1, 2, 3, 4]; shared query/seed bootstrap, 10000 draws (95%). Held list: 12,281 rows; scored rows: 12281. Labelled full rows: emotion 12,281, style 12,281, genre 9,949. Smoke: False. Queries exclude self; singleton queries are dropped.

### emotion (12,281 queries)

| arm | mAP | Δ [95%] |
|---|---:|---:|
| k11=40 | 0.2642 | — |
| k11=40-no_buddy | 0.2461 | -0.0181 [-0.0190, -0.0171] |
| k11=40-no_caption_endpoints | 0.2632 | -0.0010 [-0.0029, +0.0009] |
| k11=40-no_content_correction | 0.2573 | -0.0068 [-0.0077, -0.0061] |
| k11=40-no_direct | 0.2648 | +0.0006 [-0.0005, +0.0017] |
| k11=40-no_instructed_graphs | 0.2433 | -0.0208 [-0.0223, -0.0195] |
| k11=40-no_relational | 0.2645 | +0.0004 [-0.0007, +0.0014] |
| k11=40-no_shared_trunk | 0.2633 | -0.0009 [-0.0013, -0.0006] |
| k11=40-no_sharp_row | 0.2643 | +0.0001 [-0.0009, +0.0010] |
| plain image | 0.2424 | — |
| teacher | 0.2665 | — |

### style (12,281 queries)

| arm | mAP | Δ [95%] |
|---|---:|---:|
| k11=40 | 0.3110 | — |
| k11=40-no_buddy | 0.2129 | -0.0981 [-0.1002, -0.0961] |
| k11=40-no_caption_endpoints | 0.3154 | +0.0044 [+0.0034, +0.0055] |
| k11=40-no_content_correction | 0.2533 | -0.0577 [-0.0607, -0.0550] |
| k11=40-no_direct | 0.3115 | +0.0005 [-0.0007, +0.0016] |
| k11=40-no_instructed_graphs | 0.1887 | -0.1223 [-0.1246, -0.1200] |
| k11=40-no_relational | 0.3097 | -0.0013 [-0.0031, +0.0001] |
| k11=40-no_shared_trunk | 0.3111 | +0.0001 [-0.0012, +0.0010] |
| k11=40-no_sharp_row | 0.3106 | -0.0004 [-0.0023, +0.0011] |
| plain image | 0.1801 | — |
| teacher | 0.3091 | — |

### genre (9,949 queries)

| arm | mAP | Δ [95%] |
|---|---:|---:|
| k11=40 | 0.5896 | — |
| k11=40-no_buddy | 0.5897 | +0.0000 [-0.0012, +0.0011] |
| k11=40-no_caption_endpoints | 0.5879 | -0.0017 [-0.0038, +0.0004] |
| k11=40-no_content_correction | 0.5885 | -0.0011 [-0.0019, -0.0000] |
| k11=40-no_direct | 0.5883 | -0.0013 [-0.0053, +0.0017] |
| k11=40-no_instructed_graphs | 0.5906 | +0.0010 [+0.0000, +0.0018] |
| k11=40-no_relational | 0.4610 | -0.1287 [-0.1328, -0.1241] |
| k11=40-no_shared_trunk | 0.5889 | -0.0007 [-0.0013, -0.0002] |
| k11=40-no_sharp_row | 0.5820 | -0.0077 [-0.0099, -0.0047] |
| plain image | 0.4760 | — |
| teacher | 0.5860 | — |

Δ = removal minus c*. Context rows are unseeded.
