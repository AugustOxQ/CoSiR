# Spatial attention diagnostics

Medians [10th, 90th percentiles]. Descriptive results do not establish object grounding.
A/B checks completed on CPU float32 are provisional for the controller’s CUDA bf16 run.

## geometry

FAIL

Differential hidden[0]: true >= .95 for all layouts; a,b,d transpose >= .10 lower. Hidden[8] and random controls are reported, not gated.
Per-cell d = L2(h_Y - h_X); an exact 1-D two-means split of d alone predicts the high-distance group as changed. Agreement excludes boundary-straddling cells. Position terms cancel between paired inputs; vision/LLM mixing can spread differences. Figure pairs share a colour scale within each layer. Processor calibration is unchanged.

| Image | Hidden index | True | Transpose | Random |
|---|---:|---:|---:|---:|
| a_left_red | 0 | 0.5439 | 0.5175 | 0.5570 |
| a_left_red | 8 | 0.6623 | 0.6009 | 0.5526 |
| a_left_red (processor calibration only) | pixels | 1.0000 | 0.5439 | 0.5570 |
| b_top_white | 0 | 0.5792 | 0.5875 | 0.6125 |
| b_top_white | 8 | 0.6042 | 0.5458 | 0.5208 |
| b_top_white (processor calibration only) | pixels | 1.0000 | 0.6250 | 0.6417 |
| c_square | 0 | 0.7750 | 0.7583 | 0.7750 |
| c_square | 8 | 0.8125 | 0.7375 | 0.7458 |
| c_square (processor calibration only) | pixels | 1.0000 | 0.9250 | 0.9250 |
| d_quadrants | 0 | 0.7167 | 0.6417 | 0.6833 |
| d_quadrants | 8 | 0.6708 | 0.5375 | 0.6292 |
| d_quadrants (processor calibration only) | pixels | 1.0000 | 0.5333 | 0.2958 |

Figure: [geometry.png](geometry.png)

## recompute

FAIL

Max L1: 0.24192; viewer: {"status": "PASS", "url": "http://localhost:8765/api/maps?k=0&mod=img&layer=layers+16-27", "max_abs_errors": {"default": 4.9910234599945605e-08, "emotion_focus": 4.76447975711336e-07, "style_word": 4.862062013156337e-07, "genre_word": 4.6725881992087936e-07}}

| k | View | Band | L1 |
|---:|---|---|---:|
| 0 | default | layers 16-27 | 0.09958 |
| 0 | default | last layer | 0.14139 |
| 0 | default | all layers | 0.07188 |
| 0 | emotion_focus | layers 16-27 | 0.14109 |
| 0 | emotion_focus | last layer | 0.20204 |
| 0 | emotion_focus | all layers | 0.08930 |
| 1 | default | layers 16-27 | 0.08368 |
| 1 | default | last layer | 0.15487 |
| 1 | default | all layers | 0.06053 |
| 1 | emotion_focus | layers 16-27 | 0.17828 |
| 1 | emotion_focus | last layer | 0.24192 |
| 1 | emotion_focus | all layers | 0.10920 |
| 2 | default | layers 16-27 | 0.04802 |
| 2 | default | last layer | 0.08917 |
| 2 | default | all layers | 0.03771 |
| 2 | emotion_focus | layers 16-27 | 0.07489 |
| 2 | emotion_focus | last layer | 0.11363 |
| 2 | emotion_focus | all layers | 0.05184 |
| 3 | default | layers 16-27 | 0.03700 |
| 3 | default | last layer | 0.08846 |
| 3 | default | all layers | 0.02861 |
| 3 | emotion_focus | layers 16-27 | 0.06148 |
| 3 | emotion_focus | last layer | 0.15816 |
| 3 | emotion_focus | all layers | 0.04388 |

Query row is the final unpadded token, verified `<|endoftext|>` (151643).


## structure

DESCRIPTIVE

5000 paintings; 8 workers; 9.5 seconds.
Binary symmetric rook Moran I; ceil(percent*N) top masses; exact top-5 Jaccard; one independently seeded shuffle per map. Constant-map I is undefined.

| View | Band | Moran I | Shuffle I | Top 1% | Top 5% | Border mass | Border share | Border enrichment | Top-5 Jaccard/default |
|---|---|---|---|---|---|---|---|---|---|
| default | layers 16-27 | 0.1376 [0.0642, 0.2325] | -0.0027 [-0.0515, 0.0456] | 0.0572 [0.0436, 0.0753] | 0.1972 [0.1523, 0.2472] | 0.1919 [0.1295, 0.3601] | 0.1784 [0.1720, 0.2841] | 1.0392 [0.7394, 1.3302] | 1.0000 [1.0000, 1.0000] |
| default | last layer | 0.0939 [0.0295, 0.1812] | -0.0055 [-0.0494, 0.0481] | 0.1068 [0.0756, 0.1426] | 0.3244 [0.2373, 0.4041] | 0.2909 [0.2167, 0.4080] | 0.1784 [0.1720, 0.2841] | 1.4486 [1.1611, 1.8120] | 1.0000 [1.0000, 1.0000] |
| default | all layers | 0.2331 [0.1748, 0.2986] | -0.0052 [-0.0512, 0.0467] | 0.0710 [0.0597, 0.0830] | 0.2110 [0.1822, 0.2407] | 0.3090 [0.2611, 0.4291] | 0.1784 [0.1720, 0.2841] | 1.5863 [1.3993, 1.8076] | 1.0000 [1.0000, 1.0000] |
| emotion_focus | layers 16-27 | 0.0645 [0.0041, 0.1819] | -0.0076 [-0.0424, 0.0444] | 0.1578 [0.1069, 0.2192] | 0.3835 [0.2879, 0.4735] | 0.2257 [0.1488, 0.3324] | 0.1784 [0.1720, 0.2841] | 1.1047 [0.7709, 1.5628] | 0.0000 [0.0000, 0.2500] |
| emotion_focus | last layer | 0.0783 [0.0117, 0.1850] | -0.0073 [-0.0421, 0.0460] | 0.1905 [0.1301, 0.2744] | 0.4771 [0.3531, 0.5933] | 0.3425 [0.2444, 0.4684] | 0.1784 [0.1720, 0.2841] | 1.6900 [1.2298, 2.2154] | 0.2500 [0.1111, 0.6667] |
| emotion_focus | all layers | 0.2068 [0.1191, 0.2781] | -0.0074 [-0.0449, 0.0451] | 0.1188 [0.0905, 0.1548] | 0.3131 [0.2548, 0.3636] | 0.3604 [0.3052, 0.4420] | 0.1784 [0.1720, 0.2841] | 1.8303 [1.4156, 2.2029] | 0.2500 [0.1111, 0.6667] |
| style_word | layers 16-27 | 0.0879 [0.0291, 0.1677] | -0.0068 [-0.0441, 0.0450] | 0.1446 [0.0962, 0.2067] | 0.3743 [0.2728, 0.4745] | 0.2612 [0.1626, 0.4229] | 0.1784 [0.1720, 0.2841] | 1.3193 [0.9211, 1.7460] | 0.0000 [0.0000, 0.1111] |
| style_word | last layer | 0.0874 [0.0212, 0.1773] | -0.0078 [-0.0457, 0.0460] | 0.1556 [0.1032, 0.2166] | 0.4140 [0.2975, 0.5148] | 0.3274 [0.2284, 0.4519] | 0.1784 [0.1720, 0.2841] | 1.6025 [1.1996, 2.0930] | 0.2500 [0.1111, 0.6667] |
| style_word | all layers | 0.2237 [0.1430, 0.2994] | -0.0067 [-0.0493, 0.0463] | 0.1103 [0.0813, 0.1401] | 0.3151 [0.2522, 0.3700] | 0.3859 [0.3229, 0.5007] | 0.1784 [0.1720, 0.2841] | 1.9591 [1.6456, 2.2699] | 0.4286 [0.1111, 0.6667] |
| genre_word | layers 16-27 | 0.0847 [0.0239, 0.1598] | -0.0065 [-0.0460, 0.0475] | 0.1437 [0.0978, 0.2028] | 0.3944 [0.2928, 0.4979] | 0.2424 [0.1358, 0.3984] | 0.1784 [0.1720, 0.2841] | 1.2095 [0.7704, 1.6477] | 0.0000 [0.0000, 0.2500] |
| genre_word | last layer | 0.0714 [0.0128, 0.1661] | -0.0075 [-0.0424, 0.0424] | 0.1662 [0.1081, 0.2481] | 0.4051 [0.3034, 0.5253] | 0.3140 [0.2218, 0.4348] | 0.1784 [0.1720, 0.2841] | 1.5419 [1.1295, 2.0640] | 0.2500 [0.1111, 0.6667] |
| genre_word | all layers | 0.2016 [0.1281, 0.2728] | -0.0066 [-0.0498, 0.0467] | 0.1075 [0.0796, 0.1327] | 0.3166 [0.2511, 0.3702] | 0.3778 [0.3138, 0.4872] | 0.1784 [0.1720, 0.2841] | 1.9096 [1.5961, 2.2223] | 0.4286 [0.2500, 0.6667] |


## sinks

DESCRIPTIVE

Attention layer l uses hidden_states[l+1] output; layer 27 output has terminal RMSNorm. Mean L2 across outputs 16..27; max_abs is the maximum over their layers and channels. Text = instruction tokens.

Cross-view top-5 norm Jaccard: 0.4583 [0.2500, 0.8333].

| View | Norm rho | Maxabs rho | Top 1% norm mass | Top 5% norm mass | Top-5 intersection | Top norm border | Border share | Max image / text median | Max image / image median |
|---|---|---|---|---|---|---|---|---|---|
| default | 0.6780 [0.6780, 0.6780] | 0.2982 [0.2982, 0.2982] | 0.0229 [0.0229, 0.0229] | 0.1276 [0.1276, 0.1276] | 0.0000 [0.0000, 0.0000] | 1.0000 [1.0000, 1.0000] | 0.1785 [0.1785, 0.1785] | 1.4826 [1.4826, 1.4826] | 1.4275 [1.4275, 1.4275] |
| emotion_focus | 0.6858 [0.6858, 0.6858] | 0.4892 [0.4892, 0.4892] | 0.0233 [0.0233, 0.0233] | 0.1471 [0.1471, 0.1471] | 0.0000 [0.0000, 0.0000] | 0.4000 [0.4000, 0.4000] | 0.1785 [0.1785, 0.1785] | 1.2347 [1.2347, 1.2347] | 1.4564 [1.4564, 1.4564] |
| style_word | 0.4642 [0.4642, 0.4642] | 0.0356 [0.0356, 0.0356] | 0.0126 [0.0126, 0.0126] | 0.0648 [0.0648, 0.0648] | 0.0000 [0.0000, 0.0000] | 0.6000 [0.6000, 0.6000] | 0.1785 [0.1785, 0.1785] | 1.1450 [1.1450, 1.1450] | 1.4831 [1.4831, 1.4831] |
| genre_word | 0.4766 [0.4766, 0.4766] | 0.0688 [0.0688, 0.0688] | 0.0209 [0.0209, 0.0209] | 0.0865 [0.0865, 0.0865] | 0.0000 [0.0000, 0.0000] | 0.6000 [0.6000, 0.6000] | 0.1785 [0.1785, 0.1785] | 1.2582 [1.2582, 1.2582] | 1.4927 [1.4927, 1.4927] |

Per-head cells: Moran I / top-5% mass, each median [p10, p90].

| Head | default | emotion_focus | style_word | genre_word |
|---:|---|---|---|---|
| 0 | 0.0887 [0.0887, 0.0887] / 0.1885 [0.1885, 0.1885] | 0.1336 [0.1336, 0.1336] / 0.3363 [0.3363, 0.3363] | 0.0884 [0.0884, 0.0884] / 0.3453 [0.3453, 0.3453] | 0.0962 [0.0962, 0.0962] / 0.3363 [0.3363, 0.3363] |
| 1 | 0.1471 [0.1471, 0.1471] / 0.1753 [0.1753, 0.1753] | 0.1482 [0.1482, 0.1482] / 0.3098 [0.3098, 0.3098] | 0.1169 [0.1169, 0.1169] / 0.3692 [0.3692, 0.3692] | 0.0770 [0.0770, 0.0770] / 0.4475 [0.4475, 0.4475] |
| 2 | 0.1283 [0.1283, 0.1283] / 0.2133 [0.2133, 0.2133] | 0.0797 [0.0797, 0.0797] / 0.3536 [0.3536, 0.3536] | 0.0805 [0.0805, 0.0805] / 0.4058 [0.4058, 0.4058] | 0.0996 [0.0996, 0.0996] / 0.4082 [0.4082, 0.4082] |
| 3 | 0.1075 [0.1075, 0.1075] / 0.2103 [0.2103, 0.2103] | 0.0650 [0.0650, 0.0650] / 0.3174 [0.3174, 0.3174] | 0.1211 [0.1211, 0.1211] / 0.3105 [0.3105, 0.3105] | 0.1786 [0.1786, 0.1786] / 0.2820 [0.2820, 0.2820] |
| 4 | 0.1607 [0.1607, 0.1607] / 0.1915 [0.1915, 0.1915] | 0.1392 [0.1392, 0.1392] / 0.2906 [0.2906, 0.2906] | 0.0910 [0.0910, 0.0910] / 0.3716 [0.3716, 0.3716] | 0.0632 [0.0632, 0.0632] / 0.3816 [0.3816, 0.3816] |
| 5 | 0.1699 [0.1699, 0.1699] / 0.1905 [0.1905, 0.1905] | 0.1349 [0.1349, 0.1349] / 0.2900 [0.2900, 0.2900] | 0.0274 [0.0274, 0.0274] / 0.4257 [0.4257, 0.4257] | 0.0236 [0.0236, 0.0236] / 0.4138 [0.4138, 0.4138] |
| 6 | 0.0493 [0.0493, 0.0493] / 0.2036 [0.2036, 0.2036] | 0.0519 [0.0519, 0.0519] / 0.4840 [0.4840, 0.4840] | 0.0682 [0.0682, 0.0682] / 0.4415 [0.4415, 0.4415] | 0.0517 [0.0517, 0.0517] / 0.4284 [0.4284, 0.4284] |
| 7 | 0.0834 [0.0834, 0.0834] / 0.2109 [0.2109, 0.2109] | 0.0867 [0.0867, 0.0867] / 0.4243 [0.4243, 0.4243] | 0.0492 [0.0492, 0.0492] / 0.3805 [0.3805, 0.3805] | 0.0433 [0.0433, 0.0433] / 0.3749 [0.3749, 0.3749] |
| 8 | 0.1399 [0.1399, 0.1399] / 0.2058 [0.2058, 0.2058] | 0.0512 [0.0512, 0.0512] / 0.3919 [0.3919, 0.3919] | 0.0255 [0.0255, 0.0255] / 0.4356 [0.4356, 0.4356] | 0.0384 [0.0384, 0.0384] / 0.3828 [0.3828, 0.3828] |
| 9 | 0.0986 [0.0986, 0.0986] / 0.1877 [0.1877, 0.1877] | 0.0452 [0.0452, 0.0452] / 0.4028 [0.4028, 0.4028] | 0.0477 [0.0477, 0.0477] / 0.3847 [0.3847, 0.3847] | 0.1446 [0.1446, 0.1446] / 0.3248 [0.3248, 0.3248] |
| 10 | 0.1528 [0.1528, 0.1528] / 0.2271 [0.2271, 0.2271] | 0.1033 [0.1033, 0.1033] / 0.4224 [0.4224, 0.4224] | 0.0498 [0.0498, 0.0498] / 0.4429 [0.4429, 0.4429] | 0.0315 [0.0315, 0.0315] / 0.4214 [0.4214, 0.4214] |
| 11 | 0.1375 [0.1375, 0.1375] / 0.2667 [0.2667, 0.2667] | 0.0833 [0.0833, 0.0833] / 0.3798 [0.3798, 0.3798] | 0.0297 [0.0297, 0.0297] / 0.3855 [0.3855, 0.3855] | 0.0327 [0.0327, 0.0327] / 0.4063 [0.4063, 0.4063] |
| 12 | 0.1344 [0.1344, 0.1344] / 0.1952 [0.1952, 0.1952] | 0.1020 [0.1020, 0.1020] / 0.3357 [0.3357, 0.3357] | 0.0825 [0.0825, 0.0825] / 0.3624 [0.3624, 0.3624] | 0.0927 [0.0927, 0.0927] / 0.3656 [0.3656, 0.3656] |
| 13 | 0.1707 [0.1707, 0.1707] / 0.2338 [0.2338, 0.2338] | 0.1137 [0.1137, 0.1137] / 0.3266 [0.3266, 0.3266] | 0.1539 [0.1539, 0.1539] / 0.2921 [0.2921, 0.2921] | 0.1184 [0.1184, 0.1184] / 0.3011 [0.3011, 0.3011] |
| 14 | 0.1041 [0.1041, 0.1041] / 0.2349 [0.2349, 0.2349] | 0.1023 [0.1023, 0.1023] / 0.3988 [0.3988, 0.3988] | 0.0311 [0.0311, 0.0311] / 0.4414 [0.4414, 0.4414] | 0.0095 [0.0095, 0.0095] / 0.4524 [0.4524, 0.4524] |
| 15 | 0.0815 [0.0815, 0.0815] / 0.2345 [0.2345, 0.2345] | 0.1234 [0.1234, 0.1234] / 0.4526 [0.4526, 0.4526] | 0.0660 [0.0660, 0.0660] / 0.3430 [0.3430, 0.3430] | 0.0483 [0.0483, 0.0483] / 0.3420 [0.3420, 0.3420] |


## objects

DESCRIPTIVE

Object cells have their 32x32 cell centre inside the disc silhouette; fractional metrics weight each cell by silhouette pixel coverage.

| Object | View | Band | Object mass | Token share | Enrichment | Hottest r,c |
|---|---|---|---:|---:|---:|---|
| disc_0 | default | layers 16-27 | 0.0555 | 0.0250 | 2.219 | [3, 5] |
| disc_0 | default | last layer | 0.0098 | 0.0250 | 0.392 | [7, 12] |
| disc_0 | emotion_focus | layers 16-27 | 0.0343 | 0.0250 | 1.370 | [7, 12] |
| disc_0 | emotion_focus | last layer | 0.0077 | 0.0250 | 0.306 | [7, 17] |
| disc_1 | default | layers 16-27 | 0.0400 | 0.0250 | 1.600 | [3, 15] |
| disc_1 | default | last layer | 0.0098 | 0.0250 | 0.392 | [7, 12] |
| disc_1 | emotion_focus | layers 16-27 | 0.0254 | 0.0250 | 1.016 | [4, 19] |
| disc_1 | emotion_focus | last layer | 0.0064 | 0.0250 | 0.255 | [4, 19] |
| disc_2 | default | layers 16-27 | 0.0363 | 0.0250 | 1.451 | [13, 7] |
| disc_2 | default | last layer | 0.0104 | 0.0250 | 0.417 | [9, 14] |
| disc_2 | emotion_focus | layers 16-27 | 0.0209 | 0.0250 | 0.836 | [11, 18] |
| disc_2 | emotion_focus | last layer | 0.0080 | 0.0250 | 0.319 | [7, 15] |
| disc_3 | default | layers 16-27 | 0.0348 | 0.0250 | 1.390 | [10, 15] |
| disc_3 | default | last layer | 0.0129 | 0.0250 | 0.517 | [13, 5] |
| disc_3 | emotion_focus | layers 16-27 | 0.0215 | 0.0250 | 0.860 | [11, 18] |
| disc_3 | emotion_focus | last layer | 0.0081 | 0.0250 | 0.326 | [11, 18] |
| face_0 | default | layers 16-27 | 0.0266 | 0.0250 | 1.063 | [2, 1] |
| face_0 | default | last layer | 0.0088 | 0.0250 | 0.352 | [9, 6] |
| face_0 | emotion_focus | layers 16-27 | 0.0164 | 0.0250 | 0.658 | [0, 5] |
| face_0 | emotion_focus | last layer | 0.0236 | 0.0250 | 0.943 | [11, 0] |
| face_1 | default | layers 16-27 | 0.0251 | 0.0250 | 1.005 | [8, 17] |
| face_1 | default | last layer | 0.0103 | 0.0250 | 0.414 | [7, 12] |
| face_1 | emotion_focus | layers 16-27 | 0.0195 | 0.0250 | 0.781 | [0, 15] |
| face_1 | emotion_focus | last layer | 0.0213 | 0.0250 | 0.853 | [11, 0] |
| face_2 | default | layers 16-27 | 0.0178 | 0.0250 | 0.711 | [13, 1] |
| face_2 | default | last layer | 0.0107 | 0.0250 | 0.428 | [10, 17] |
| face_2 | emotion_focus | layers 16-27 | 0.0124 | 0.0250 | 0.495 | [0, 4] |
| face_2 | emotion_focus | last layer | 0.0182 | 0.0250 | 0.728 | [10, 17] |
| face_3 | default | layers 16-27 | 0.0173 | 0.0250 | 0.691 | [7, 12] |
| face_3 | default | last layer | 0.0135 | 0.0250 | 0.538 | [11, 0] |
| face_3 | emotion_focus | layers 16-27 | 0.0184 | 0.0250 | 0.735 | [13, 6] |
| face_3 | emotion_focus | last layer | 0.0186 | 0.0250 | 0.742 | [13, 14] |

Medians over the four positions; hottest object-cell rank among all cells is 1 + the number of cells with strictly greater attention (1 = hottest).

| Object | View | Band | Median mass | Median token share | Median hottest object rank |
|---|---|---|---:|---:|---:|
| disc | default | layers 16-27 | 0.0381 | 0.0250 | 1.0 |
| disc | default | last layer | 0.0101 | 0.0250 | 48.0 |
| disc | emotion_focus | layers 16-27 | 0.0234 | 0.0250 | 9.0 |
| disc | emotion_focus | last layer | 0.0078 | 0.0250 | 58.0 |
| face | default | layers 16-27 | 0.0215 | 0.0250 | 38.0 |
| face | default | last layer | 0.0105 | 0.0250 | 92.0 |
| face | emotion_focus | layers 16-27 | 0.0174 | 0.0250 | 50.5 |
| face | emotion_focus | last layer | 0.0199 | 0.0250 | 48.5 |

Figure: [objects.png](objects.png)

## sheet

DESCRIPTIVE

Paintings: [0, 625, 1250, 1875, 2500, 3125, 3750, 4375]
Prepared image JPEG quality 90; jet alpha .5, nearest; ranks use average ties.

Figure: [sheet_values.png](sheet_values.png)
Figure: [sheet_ranks.png](sheet_ranks.png)

Cleanup candidates (nothing deleted): generated PNGs, results.json, summary.md, run logs and the local Matplotlib cache under this output directory. No source or stored data is a removal candidate.
