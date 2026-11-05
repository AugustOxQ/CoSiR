# Text retrieval exploration

Validation picks and scores use the same rows; exploration numbers only.

## Table 1: label mAP

| row | emotion/t2i | emotion/i2t | emotion/t2t | style/t2i | style/i2t | style/t2t | genre/t2i | genre/i2t | genre/t2t |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cstar | 0.2650 (center_k10) | 0.2898 (center_k10) | 0.3455 (center_k10) | 0.1720 (center_k10) | 0.1359 (center_k10) | 0.1072 (center_k10) | 0.4877 (center_k10) | 0.3989 (center) | 0.3073 (center_k10) |
| cstar_free | 0.2577 (center_k10) | 0.2768 (center_k10) | 0.3168 (center_k10) | 0.1467 (center_k10) | 0.1116 (center_k10) | 0.0921 (center_k10) | 0.4537 (center_k10) | 0.3968 (center_k10) | 0.3049 (center_k10) |
| plain | 0.2236 (center) | 0.2407 (center_k10) | 0.2541 (center_k10) | 0.1342 (center_k10) | 0.1079 (center) | 0.0940 (center_k10) | 0.4179 (center_k10) | 0.3390 (center_k10) | 0.2669 (center_k10) |
| teacher | 0.2714 (center_k10) | 0.3036 (center_k10) | 0.3790 (center_k10) | 0.1578 (center_k10) | 0.1219 (center) | 0.0953 (center_k10) | 0.4470 (center_k10) | 0.3932 (center) | 0.2951 (center_k10) |
| clip_l14 | 0.2165 (center_k10) | 0.2232 (center_k10) | 0.2023 (center) | 0.1516 (center_k10) | 0.1076 (center_k10) | 0.0909 (center_k10) | 0.4168 (center_k10) | 0.3569 (center_k10) | 0.2557 (center_k10) |
| siglip2_so400m | 0.2116 (center_k10) | 0.2214 (center_k10) | 0.2029 (center) | 0.1495 (center_k10) | 0.1056 (center_k10) | 0.0893 (center_k10) | 0.4293 (center_k10) | 0.3459 (center_k10) | 0.2447 (center_k10) |
| siglip_so400m | 0.2143 (center_k10) | 0.2229 (center_k10) | 0.2065 (center) | 0.1529 (center_k10) | 0.1065 (center) | 0.0892 (center_k10) | 0.4231 (center_k10) | 0.3511 (center_k10) | 0.2471 (center_k10) |
| roberta | — | — | 0.3099 (r0) | — | — | — | — | — | — |
| chance | 0.1723 | 0.1723 | 0.1565 | 0.0756 | 0.0756 | 0.0753 | 0.1404 | 0.1404 | 0.1401 |

## Table 1b: macro mAP at the same picked readout

| row | emotion/t2i | emotion/i2t | emotion/t2t | style/t2i | style/i2t | style/t2t | genre/t2i | genre/i2t | genre/t2t |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cstar | 0.1911 (center_k10) | 0.2171 (center_k10) | 0.2962 (center_k10) | 0.1323 (center_k10) | 0.0915 (center_k10) | 0.0653 (center_k10) | 0.4119 (center_k10) | 0.3281 (center) | 0.2419 (center_k10) |
| cstar_free | 0.1865 (center_k10) | 0.2044 (center_k10) | 0.2632 (center_k10) | 0.0985 (center_k10) | 0.0708 (center_k10) | 0.0500 (center_k10) | 0.3834 (center_k10) | 0.3191 (center_k10) | 0.2287 (center_k10) |
| plain | 0.1525 (center) | 0.1615 (center_k10) | 0.1969 (center_k10) | 0.1014 (center_k10) | 0.0670 (center) | 0.0521 (center_k10) | 0.3477 (center_k10) | 0.2703 (center_k10) | 0.1987 (center_k10) |
| teacher | 0.1922 (center_k10) | 0.2193 (center_k10) | 0.3260 (center_k10) | 0.1104 (center_k10) | 0.0777 (center) | 0.0528 (center_k10) | 0.3765 (center_k10) | 0.3114 (center) | 0.2217 (center_k10) |
| clip_l14 | 0.1441 (center_k10) | 0.1460 (center_k10) | 0.1506 (center) | 0.1124 (center_k10) | 0.0657 (center_k10) | 0.0506 (center_k10) | 0.3608 (center_k10) | 0.2798 (center_k10) | 0.1878 (center_k10) |
| siglip2_so400m | 0.1393 (center_k10) | 0.1450 (center_k10) | 0.1515 (center) | 0.1103 (center_k10) | 0.0631 (center_k10) | 0.0489 (center_k10) | 0.3610 (center_k10) | 0.2727 (center_k10) | 0.1785 (center_k10) |
| siglip_so400m | 0.1425 (center_k10) | 0.1472 (center_k10) | 0.1544 (center) | 0.1136 (center_k10) | 0.0669 (center) | 0.0486 (center_k10) | 0.3633 (center_k10) | 0.2781 (center_k10) | 0.1801 (center_k10) |
| roberta | — | — | 0.2527 (r0) | — | — | — | — | — | — |

## Table 2: the rule

| aspect | direction | cstar readout | minus plain, 95% CI | minus teacher, 95% CI | above |
| --- | --- | --- | --- | --- | --- |
| emotion | t2i | center_k10 | 0.0414 [0.0406, 0.0421] | -0.0064 [-0.0071, -0.0058] | no |
| emotion | i2t | center_k10 | 0.0491 [0.0466, 0.0517] | -0.0139 [-0.0167, -0.0111] | no |
| emotion | t2t | center_k10 | 0.0914 [0.0899, 0.0930] | -0.0335 [-0.0343, -0.0326] | no |
| style | t2i | center_k10 | 0.0378 [0.0361, 0.0395] | 0.0142 [0.0125, 0.0160] | yes |
| style | i2t | center_k10 | 0.0280 [0.0271, 0.0289] | 0.0140 [0.0134, 0.0147] | yes |
| style | t2t | center_k10 | 0.0132 [0.0126, 0.0138] | 0.0119 [0.0113, 0.0126] | yes |
| genre | t2i | center_k10 | 0.0698 [0.0676, 0.0722] | 0.0407 [0.0375, 0.0437] | yes |
| genre | i2t | center | 0.0599 [0.0573, 0.0623] | 0.0057 [0.0044, 0.0071] | yes |
| genre | t2t | center_k10 | 0.0404 [0.0390, 0.0419] | 0.0122 [0.0106, 0.0138] | yes |

Count: 6; outcome: 1.

## Table 2b: free variant beside the rule

| aspect | direction | cstar readout | minus plain, 95% CI | minus teacher, 95% CI | above |
| --- | --- | --- | --- | --- | --- |
| emotion | t2i | center_k10 | 0.0340 [0.0332, 0.0349] | -0.0137 [-0.0145, -0.0130] | no |
| emotion | i2t | center_k10 | 0.0362 [0.0340, 0.0383] | -0.0268 [-0.0294, -0.0244] | no |
| emotion | t2t | center_k10 | 0.0627 [0.0615, 0.0639] | -0.0622 [-0.0633, -0.0610] | no |
| style | t2i | center_k10 | 0.0126 [0.0109, 0.0143] | -0.0111 [-0.0126, -0.0096] | no |
| style | i2t | center_k10 | 0.0037 [0.0031, 0.0043] | -0.0103 [-0.0108, -0.0098] | no |
| style | t2t | center_k10 | -0.0020 [-0.0022, -0.0017] | -0.0032 [-0.0035, -0.0029] | no |
| genre | t2i | center_k10 | 0.0358 [0.0332, 0.0383] | 0.0067 [0.0045, 0.0088] | yes |
| genre | i2t | center_k10 | 0.0577 [0.0551, 0.0602] | 0.0036 [0.0024, 0.0048] | yes |
| genre | t2t | center_k10 | 0.0380 [0.0367, 0.0394] | 0.0098 [0.0089, 0.0107] | yes |

Count: 3; outcome: 2.

## Table 3: pair hits, best MRR readout

| row | aspect | direction | readout | R@1 | R@5 | R@10 | MedR | MRR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clip_l14 | all | i2t | center | 0.1494 | 0.3215 | 0.4189 | 17.0000 | 0.2390 |
| clip_l14 | all | t2i | center | 0.0902 | 0.1987 | 0.2603 | 69.0000 | 0.1487 |
| clip_l14 | all | t2t | center | 0.0514 | 0.1128 | 0.1509 | 295.0000 | 0.0862 |
| cstar | emotion | i2t | center | 0.0348 | 0.1105 | 0.1666 | 107.0000 | 0.0805 |
| cstar | emotion | t2i | center | 0.0200 | 0.0615 | 0.0934 | 267.5000 | 0.0474 |
| cstar | emotion | t2t | r0 | 0.0187 | 0.0485 | 0.0686 | 875.0000 | 0.0373 |
| cstar | genre | i2t | r0 | 0.0865 | 0.2285 | 0.3186 | 30.0000 | 0.1646 |
| cstar | genre | t2i | center | 0.0571 | 0.1498 | 0.2118 | 88.0000 | 0.1098 |
| cstar | genre | t2t | r0 | 0.0597 | 0.1318 | 0.1756 | 198.0000 | 0.0998 |
| cstar | style | i2t | center | 0.0060 | 0.0221 | 0.0416 | 459.0000 | 0.0208 |
| cstar | style | t2i | center_k10 | 0.0036 | 0.0163 | 0.0278 | 727.0000 | 0.0148 |
| cstar | style | t2t | center | 0.0289 | 0.0649 | 0.0911 | 600.0000 | 0.0516 |
| cstar_free | emotion | i2t | center | 0.1403 | 0.2848 | 0.3708 | 26.0000 | 0.2169 |
| cstar_free | emotion | t2i | center | 0.0655 | 0.1470 | 0.1984 | 137.0000 | 0.1117 |
| cstar_free | emotion | t2t | center | 0.0448 | 0.0988 | 0.1318 | 405.0000 | 0.0761 |
| cstar_free | genre | i2t | center | 0.0792 | 0.1835 | 0.2508 | 67.0000 | 0.1380 |
| cstar_free | genre | t2i | center | 0.0531 | 0.1266 | 0.1738 | 163.0000 | 0.0955 |
| cstar_free | genre | t2t | center | 0.0468 | 0.1036 | 0.1394 | 320.0000 | 0.0796 |
| cstar_free | style | i2t | center | 0.0267 | 0.0787 | 0.1188 | 232.0000 | 0.0592 |
| cstar_free | style | t2i | center | 0.0202 | 0.0540 | 0.0792 | 549.0000 | 0.0419 |
| cstar_free | style | t2t | center | 0.0317 | 0.0709 | 0.0969 | 667.0000 | 0.0553 |
| plain | all | i2t | center | 0.2081 | 0.4064 | 0.5068 | 10.0000 | 0.3063 |
| plain | all | t2i | center | 0.1178 | 0.2442 | 0.3102 | 49.0000 | 0.1840 |
| plain | all | t2t | center | 0.0675 | 0.1397 | 0.1823 | 195.0000 | 0.1077 |
| roberta | all | t2t | center | 0.0041 | 0.0109 | 0.0155 | 2908.0000 | 0.0092 |
| siglip2_so400m | all | i2t | center | 0.2326 | 0.4358 | 0.5382 | 8.0000 | 0.3319 |
| siglip2_so400m | all | t2i | center | 0.1270 | 0.2514 | 0.3180 | 46.0000 | 0.1926 |
| siglip2_so400m | all | t2t | center | 0.0522 | 0.1118 | 0.1503 | 324.0000 | 0.0862 |
| siglip_so400m | all | i2t | center | 0.2154 | 0.4163 | 0.5179 | 9.0000 | 0.3158 |
| siglip_so400m | all | t2i | center | 0.1257 | 0.2512 | 0.3173 | 47.0000 | 0.1915 |
| siglip_so400m | all | t2t | center | 0.0533 | 0.1136 | 0.1516 | 311.0000 | 0.0878 |
| teacher | emotion | i2t | center | 0.0744 | 0.1775 | 0.2466 | 67.0000 | 0.1325 |
| teacher | emotion | t2i | center | 0.0328 | 0.0832 | 0.1183 | 302.0000 | 0.0633 |
| teacher | emotion | t2t | center | 0.0205 | 0.0483 | 0.0687 | 930.0000 | 0.0383 |
| teacher | genre | i2t | center | 0.0688 | 0.1681 | 0.2341 | 72.0000 | 0.1243 |
| teacher | genre | t2i | center | 0.0447 | 0.1137 | 0.1604 | 156.0000 | 0.0853 |
| teacher | genre | t2t | center | 0.0376 | 0.0902 | 0.1261 | 312.0000 | 0.0694 |
| teacher | style | i2t | center | 0.0426 | 0.1100 | 0.1614 | 109.0000 | 0.0852 |
| teacher | style | t2i | center | 0.0275 | 0.0788 | 0.1163 | 252.0000 | 0.0593 |
| teacher | style | t2t | center | 0.0303 | 0.0743 | 0.1050 | 435.0000 | 0.0570 |

## Table 4: leak sensitivity at the all-entry picked readout

| aspect | label-name caption share |
| --- | --- |
| emotion | 0.1146 |
| style | 0.0020 |
| genre | 0.0282 |

| aspect | class | n | share |
| --- | --- | --- | --- |
| emotion | amusement | 3076 | 0.1024 |
| emotion | anger | 479 | 0.2568 |
| emotion | awe | 5012 | 0.0621 |
| emotion | contentment | 8639 | 0.0615 |
| emotion | disgust | 1496 | 0.0675 |
| emotion | excitement | 2600 | 0.1777 |
| emotion | fear | 2854 | 0.2397 |
| emotion | sadness | 3263 | 0.3101 |
| emotion | something else | 3453 | 0.0000 |
| style | Abstract_Expressionism | 1084 | 0.0000 |
| style | Action_painting | 32 | 0.0000 |
| style | Analytical_Cubism | 27 | 0.0370 |
| style | Art_Nouveau_Modern | 1740 | 0.0000 |
| style | Baroque | 1693 | 0.0000 |
| style | Color_Field_Painting | 642 | 0.0016 |
| style | Contemporary_Realism | 210 | 0.0048 |
| style | Cubism | 859 | 0.0070 |
| style | Early_Renaissance | 588 | 0.0000 |
| style | Expressionism | 2630 | 0.0000 |
| style | Fauvism | 340 | 0.0000 |
| style | High_Renaissance | 496 | 0.0040 |
| style | Impressionism | 4570 | 0.0026 |
| style | Mannerism_Late_Renaissance | 540 | 0.0000 |
| style | Minimalism | 556 | 0.0072 |
| style | Naive_Art_Primitivism | 917 | 0.0022 |
| style | New_Realism | 126 | 0.0000 |
| style | Northern_Renaissance | 1040 | 0.0000 |
| style | Pointillism | 166 | 0.0181 |
| style | Pop_Art | 638 | 0.0094 |
| style | Post_Impressionism | 2408 | 0.0029 |
| style | Realism | 4033 | 0.0025 |
| style | Rococo | 817 | 0.0000 |
| style | Romanticism | 2621 | 0.0000 |
| style | Symbolism | 1474 | 0.0007 |
| style | Synthetic_Cubism | 124 | 0.0484 |
| style | Ukiyo_e | 501 | 0.0000 |
| genre | abstract_painting | 2028 | 0.0409 |
| genre | cityscape | 1733 | 0.0023 |
| genre | genre_painting | 3942 | 0.0000 |
| genre | illustration | 816 | 0.0037 |
| genre | landscape | 4907 | 0.0406 |
| genre | nude_painting | 762 | 0.1982 |
| genre | portrait | 5281 | 0.0413 |
| genre | religious_painting | 2672 | 0.0378 |
| genre | sketch_and_study | 1504 | 0.0525 |
| genre | still_life | 1241 | 0.0258 |

| row | aspect | direction | picked readout | no_leak mAP | cross_painting mAP |
| --- | --- | --- | --- | --- | --- |
| cstar | emotion | t2i | center_k10 | 0.2616 | 0.2647 |
| cstar | emotion | i2t | center_k10 | 0.2816 | 0.2895 |
| cstar | emotion | t2t | center_k10 | 0.3095 | 0.3455 |
| cstar | style | t2i | center_k10 | 0.1716 | 0.1713 |
| cstar | style | i2t | center_k10 | 0.1356 | 0.1353 |
| cstar | style | t2t | center_k10 | 0.1070 | 0.1067 |
| cstar | genre | t2i | center_k10 | 0.4789 | 0.4869 |
| cstar | genre | i2t | center | 0.3885 | 0.3977 |
| cstar | genre | t2t | center_k10 | 0.2974 | 0.3066 |
| plain | emotion | t2i | center | 0.2257 | 0.2232 |
| plain | emotion | i2t | center_k10 | 0.2400 | 0.2404 |
| plain | emotion | t2t | center_k10 | 0.2463 | 0.2541 |
| plain | style | t2i | center_k10 | 0.1340 | 0.1326 |
| plain | style | i2t | center | 0.1077 | 0.1063 |
| plain | style | t2t | center_k10 | 0.0940 | 0.0934 |
| plain | genre | t2i | center_k10 | 0.4095 | 0.4168 |
| plain | genre | i2t | center_k10 | 0.3291 | 0.3378 |
| plain | genre | t2t | center_k10 | 0.2607 | 0.2662 |
| teacher | emotion | t2i | center_k10 | 0.2673 | 0.2712 |
| teacher | emotion | i2t | center_k10 | 0.2936 | 0.3035 |
| teacher | emotion | t2t | center_k10 | 0.3351 | 0.3790 |
| teacher | style | t2i | center_k10 | 0.1573 | 0.1569 |
| teacher | style | i2t | center | 0.1213 | 0.1209 |
| teacher | style | t2t | center_k10 | 0.0951 | 0.0948 |
| teacher | genre | t2i | center_k10 | 0.4366 | 0.4463 |
| teacher | genre | i2t | center | 0.3796 | 0.3922 |
| teacher | genre | t2t | center_k10 | 0.2857 | 0.2945 |

## Table 5: capmean2i

| row | head | label | readout | mAP | macro mAP | chance |
| --- | --- | --- | --- | --- | --- | --- |
| plain | all | emotion | r0 | 0.2480 | 0.1548 | 0.2019 |
| plain | all | style | r0 | 0.1245 | 0.0916 | 0.0760 |
| plain | all | genre | r0 | 0.3651 | 0.3223 | 0.1406 |
| plain | all | emotion | center | 0.2622 | 0.1589 | 0.2019 |
| plain | all | style | center | 0.1383 | 0.1132 | 0.0760 |
| plain | all | genre | center | 0.4311 | 0.3667 | 0.1406 |
| cstar | emotion | emotion | r0 | 0.2871 | 0.1832 | 0.2019 |
| cstar | emotion | emotion | center | 0.2905 | 0.1867 | 0.2019 |
| cstar | style | style | r0 | 0.1971 | 0.1697 | 0.0760 |
| cstar | style | style | center | 0.2036 | 0.1741 | 0.0760 |
| cstar | genre | genre | r0 | 0.5215 | 0.4665 | 0.1406 |
| cstar | genre | genre | center | 0.5378 | 0.4778 | 0.1406 |
| teacher | emotion | emotion | r0 | 0.2966 | 0.1791 | 0.2019 |
| teacher | emotion | emotion | center | 0.3014 | 0.1911 | 0.2019 |
| teacher | style | style | r0 | 0.1381 | 0.0894 | 0.0760 |
| teacher | style | style | center | 0.1665 | 0.1260 | 0.0760 |
| teacher | genre | genre | r0 | 0.4346 | 0.3655 | 0.1406 |
| teacher | genre | genre | center | 0.5198 | 0.4435 | 0.1406 |

## Table 5b: capmean2i pair hits

| row | aspect | direction | readout | R@1 | R@5 | R@10 | MedR | MRR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cstar | emotion | capmean2i | center | 0.0927 | 0.2341 | 0.3230 | 31.0000 | 0.1702 |
| cstar | genre | capmean2i | r0 | 0.2570 | 0.5057 | 0.6174 | 5.0000 | 0.3768 |
| cstar | style | capmean2i | center | 0.0143 | 0.0486 | 0.0809 | 228.0000 | 0.0398 |
| plain | all | capmean2i | center | 0.4181 | 0.6391 | 0.7243 | 2.0000 | 0.5223 |
| teacher | emotion | capmean2i | center | 0.1520 | 0.3001 | 0.3835 | 24.0000 | 0.2299 |
| teacher | genre | capmean2i | center | 0.1678 | 0.3412 | 0.4324 | 16.0000 | 0.2573 |
| teacher | style | capmean2i | center | 0.1174 | 0.2508 | 0.3324 | 33.0000 | 0.1901 |

## Table 6: SemArt pairs

| row | aspect | direction | readout | R@1 | R@5 | R@10 | MedR | MRR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clip_b32 | all | i2t | center | 0.1506 | 0.3162 | 0.4200 | 17.0000 | 0.2373 |
| clip_b32 | all | t2i | center | 0.1450 | 0.3340 | 0.4331 | 16.0000 | 0.2402 |
| cstar | emotion | i2t | center | 0.0430 | 0.1207 | 0.1862 | 101.0000 | 0.0919 |
| cstar | emotion | t2i | center | 0.0412 | 0.1188 | 0.1787 | 103.0000 | 0.0874 |
| cstar | genre | i2t | center | 0.1113 | 0.2619 | 0.3452 | 28.0000 | 0.1913 |
| cstar | genre | t2i | center | 0.1310 | 0.3040 | 0.3976 | 21.0000 | 0.2181 |
| cstar | style | i2t | center | 0.0131 | 0.0580 | 0.1066 | 108.0000 | 0.0502 |
| cstar | style | t2i | center | 0.0150 | 0.0486 | 0.0926 | 107.0000 | 0.0469 |
| cstar_free | emotion | i2t | r0 | 0.1478 | 0.2947 | 0.3704 | 26.0000 | 0.2238 |
| cstar_free | emotion | t2i | center | 0.1646 | 0.3340 | 0.4107 | 23.0000 | 0.2490 |
| cstar_free | genre | i2t | center | 0.1366 | 0.3031 | 0.3994 | 24.0000 | 0.2201 |
| cstar_free | genre | t2i | center | 0.2002 | 0.3957 | 0.4687 | 13.0000 | 0.2925 |
| cstar_free | style | i2t | center | 0.0898 | 0.2142 | 0.2956 | 42.0000 | 0.1604 |
| cstar_free | style | t2i | center | 0.1057 | 0.2413 | 0.3358 | 31.0000 | 0.1806 |
| plain | all | i2t | center | 0.2516 | 0.4593 | 0.5341 | 8.0000 | 0.3483 |
| plain | all | t2i | center | 0.3143 | 0.5070 | 0.5968 | 5.0000 | 0.4095 |
| plain_raw | all | i2t | center | 0.2694 | 0.4780 | 0.5529 | 7.0000 | 0.3674 |
| plain_raw | all | t2i | center | 0.3433 | 0.5529 | 0.6268 | 4.0000 | 0.4394 |
| siglip2_so400m | all | i2t | center | 0.2142 | 0.3863 | 0.4808 | 11.0000 | 0.3049 |
| siglip2_so400m | all | t2i | center | 0.2413 | 0.4415 | 0.5154 | 9.0000 | 0.3351 |

skip clip_l14: SemArt store absent
skip siglip_so400m: SemArt store absent
## Table 7: MMArt

Pair R@10 (readout picked on MRR).

| row/head | e_narrative | e_formal | e_emotional | e_historical | e_unified |
| --- | --- | --- | --- | --- | --- |
| cstar/emotion | 0.3850 (center) | 0.1167 (center) | 0.1723 (center) | 0.0279 (center) | 0.3456 (center) |
| cstar/genre | 0.6523 (center) | 0.3280 (center) | 0.5526 (center) | 0.1182 (center) | 0.7339 (center) |
| cstar/style | 0.0979 (center) | 0.0530 (center) | 0.0950 (center) | 0.0499 (center) | 0.1408 (center) |
| cstar_free/emotion | 0.8306 (center) | 0.3165 (center) | 0.4304 (center) | 0.1201 (center) | 0.8228 (center) |
| cstar_free/genre | 0.7005 (center) | 0.3046 (center) | 0.4947 (center) | 0.1817 (center) | 0.7736 (center) |
| cstar_free/style | 0.4724 (center) | 0.1763 (center) | 0.3023 (center) | 0.1340 (center) | 0.5547 (center) |
| plain/all | 0.9142 (center) | 0.4614 (center) | 0.7780 (center) | 0.2054 (center) | 0.9393 (center) |

emotion mAP.

| row/head | e_narrative | e_formal | e_emotional | e_historical | e_unified |
| --- | --- | --- | --- | --- | --- |
| cstar/emotion | 0.2040 (r0) | 0.1920 (center) | 0.2648 (center) | 0.1854 (center) | 0.2180 (center) |
| cstar/genre | 0.1920 (r0) | 0.1877 (center) | 0.2029 (center) | 0.1827 (center) | 0.1941 (r0) |
| cstar/style | 0.1947 (r0) | 0.1916 (r0) | 0.1978 (r0) | 0.1892 (r0) | 0.1947 (r0) |
| cstar_free/emotion | 0.2025 (r0) | 0.1931 (center) | 0.2586 (center) | 0.1874 (center) | 0.2178 (center) |
| cstar_free/genre | 0.1976 (r0) | 0.1889 (center) | 0.2156 (center) | 0.1866 (center) | 0.1986 (r0) |
| cstar_free/style | 0.1942 (center) | 0.1871 (center) | 0.2197 (center) | 0.1850 (center) | 0.1954 (center) |
| plain/all | 0.1958 (r0) | 0.1889 (center) | 0.2230 (center) | 0.1851 (center) | 0.1991 (center) |

style mAP.

| row/head | e_narrative | e_formal | e_emotional | e_historical | e_unified |
| --- | --- | --- | --- | --- | --- |
| cstar/emotion | 0.1358 (center) | 0.1377 (center) | 0.1322 (center) | 0.1347 (center) | 0.1436 (center) |
| cstar/genre | 0.1738 (center) | 0.1894 (center) | 0.1873 (center) | 0.2757 (center) | 0.2431 (center) |
| cstar/style | 0.2232 (center) | 0.2263 (center) | 0.2252 (center) | 0.3383 (center) | 0.3085 (center) |
| cstar_free/emotion | 0.1432 (center) | 0.1431 (center) | 0.1323 (center) | 0.1720 (center) | 0.1627 (center) |
| cstar_free/genre | 0.1727 (center) | 0.1859 (center) | 0.1785 (center) | 0.3014 (center) | 0.2523 (center) |
| cstar_free/style | 0.1993 (center) | 0.2213 (center) | 0.2018 (center) | 0.4077 (center) | 0.3642 (center) |
| plain/all | 0.1549 (center) | 0.1671 (center) | 0.1585 (center) | 0.2957 (center) | 0.2166 (center) |

genre mAP.

| row/head | e_narrative | e_formal | e_emotional | e_historical | e_unified |
| --- | --- | --- | --- | --- | --- |
| cstar/emotion | 0.4000 (center) | 0.3621 (center) | 0.3079 (center) | 0.2602 (center) | 0.3821 (center) |
| cstar/genre | 0.4941 (center) | 0.4757 (center) | 0.4798 (center) | 0.3603 (center) | 0.5080 (center) |
| cstar/style | 0.3487 (center) | 0.3088 (center) | 0.3196 (center) | 0.3091 (center) | 0.3395 (center) |
| cstar_free/emotion | 0.4261 (center) | 0.3885 (center) | 0.3303 (center) | 0.2632 (center) | 0.4115 (center) |
| cstar_free/genre | 0.5580 (center) | 0.5332 (center) | 0.5185 (center) | 0.4219 (center) | 0.5590 (center) |
| cstar_free/style | 0.3524 (center) | 0.3302 (center) | 0.3064 (center) | 0.3085 (center) | 0.3458 (center) |
| plain/all | 0.4456 (center) | 0.4450 (center) | 0.4147 (center) | 0.3604 (center) | 0.4726 (center) |

e_emotional own-emotion word share: 0.7384.


## Table 8: word queries

| row | aspect | form | direction | readout | macro AP | chance |
| --- | --- | --- | --- | --- | --- | --- |
| plain | emotion | bare | t2i | r0 | 0.2027 | 0.1171 |
| plain | emotion | bare | t2i | center | 0.1782 | 0.1171 |
| plain | emotion | bare | t2t | r0 | 0.2618 | 0.1110 |
| plain | emotion | bare | t2t | center | 0.3049 | 0.1110 |
| plain | emotion | template | t2i | r0 | 0.2074 | 0.1171 |
| plain | emotion | template | t2i | center | 0.2197 | 0.1171 |
| plain | emotion | template | t2t | r0 | 0.2025 | 0.1110 |
| plain | emotion | template | t2t | center | 0.2412 | 0.1110 |
| plain | style | bare | t2i | r0 | 0.2309 | 0.0370 |
| plain | style | bare | t2i | center | 0.2319 | 0.0370 |
| plain | style | bare | t2t | r0 | 0.0581 | 0.0370 |
| plain | style | bare | t2t | center | 0.0631 | 0.0370 |
| plain | style | template | t2i | r0 | 0.1737 | 0.0370 |
| plain | style | template | t2i | center | 0.2069 | 0.0370 |
| plain | style | template | t2t | r0 | 0.0466 | 0.0370 |
| plain | style | template | t2t | center | 0.0495 | 0.0370 |
| plain | genre | bare | t2i | r0 | 0.5932 | 0.1000 |
| plain | genre | bare | t2i | center | 0.5735 | 0.1000 |
| plain | genre | bare | t2t | r0 | 0.2246 | 0.1000 |
| plain | genre | bare | t2t | center | 0.2725 | 0.1000 |
| plain | genre | template | t2i | r0 | 0.5400 | 0.1000 |
| plain | genre | template | t2i | center | 0.5444 | 0.1000 |
| plain | genre | template | t2t | r0 | 0.1722 | 0.1000 |
| plain | genre | template | t2t | center | 0.2013 | 0.1000 |
| clip_l14 | emotion | bare | t2i | r0 | 0.1719 | 0.1171 |
| clip_l14 | emotion | bare | t2i | center | 0.1305 | 0.1171 |
| clip_l14 | emotion | bare | t2t | r0 | 0.1367 | 0.1110 |
| clip_l14 | emotion | bare | t2t | center | 0.1623 | 0.1110 |
| clip_l14 | emotion | template | t2i | r0 | 0.1714 | 0.1171 |
| clip_l14 | emotion | template | t2i | center | 0.1346 | 0.1171 |
| clip_l14 | emotion | template | t2t | r0 | 0.1311 | 0.1110 |
| clip_l14 | emotion | template | t2t | center | 0.1513 | 0.1110 |
| clip_l14 | style | bare | t2i | r0 | 0.3357 | 0.0370 |
| clip_l14 | style | bare | t2i | center | 0.1494 | 0.0370 |
| clip_l14 | style | bare | t2t | r0 | 0.0493 | 0.0370 |
| clip_l14 | style | bare | t2t | center | 0.0586 | 0.0370 |
| clip_l14 | style | template | t2i | r0 | 0.2779 | 0.0370 |
| clip_l14 | style | template | t2i | center | 0.1375 | 0.0370 |
| clip_l14 | style | template | t2t | r0 | 0.0476 | 0.0370 |
| clip_l14 | style | template | t2t | center | 0.0533 | 0.0370 |
| clip_l14 | genre | bare | t2i | r0 | 0.5735 | 0.1000 |
| clip_l14 | genre | bare | t2i | center | 0.2513 | 0.1000 |
| clip_l14 | genre | bare | t2t | r0 | 0.1542 | 0.1000 |
| clip_l14 | genre | bare | t2t | center | 0.2127 | 0.1000 |
| clip_l14 | genre | template | t2i | r0 | 0.4665 | 0.1000 |
| clip_l14 | genre | template | t2i | center | 0.2141 | 0.1000 |
| clip_l14 | genre | template | t2t | r0 | 0.1285 | 0.1000 |
| clip_l14 | genre | template | t2t | center | 0.1635 | 0.1000 |
| siglip2_so400m | emotion | bare | t2i | r0 | 0.1827 | 0.1171 |
| siglip2_so400m | emotion | bare | t2i | center | 0.1341 | 0.1171 |
| siglip2_so400m | emotion | bare | t2t | r0 | 0.1532 | 0.1110 |
| siglip2_so400m | emotion | bare | t2t | center | 0.1900 | 0.1110 |
| siglip2_so400m | emotion | template | t2i | r0 | 0.1911 | 0.1171 |
| siglip2_so400m | emotion | template | t2i | center | 0.1392 | 0.1171 |
| siglip2_so400m | emotion | template | t2t | r0 | 0.1358 | 0.1110 |
| siglip2_so400m | emotion | template | t2t | center | 0.1579 | 0.1110 |
| siglip2_so400m | style | bare | t2i | r0 | 0.3664 | 0.0370 |
| siglip2_so400m | style | bare | t2i | center | 0.1387 | 0.0370 |
| siglip2_so400m | style | bare | t2t | r0 | 0.0489 | 0.0370 |
| siglip2_so400m | style | bare | t2t | center | 0.0565 | 0.0370 |
| siglip2_so400m | style | template | t2i | r0 | 0.3256 | 0.0370 |
| siglip2_so400m | style | template | t2i | center | 0.1435 | 0.0370 |
| siglip2_so400m | style | template | t2t | r0 | 0.0463 | 0.0370 |
| siglip2_so400m | style | template | t2t | center | 0.0480 | 0.0370 |
| siglip2_so400m | genre | bare | t2i | r0 | 0.6779 | 0.1000 |
| siglip2_so400m | genre | bare | t2i | center | 0.2939 | 0.1000 |
| siglip2_so400m | genre | bare | t2t | r0 | 0.1498 | 0.1000 |
| siglip2_so400m | genre | bare | t2t | center | 0.1918 | 0.1000 |
| siglip2_so400m | genre | template | t2i | r0 | 0.6237 | 0.1000 |
| siglip2_so400m | genre | template | t2i | center | 0.2946 | 0.1000 |
| siglip2_so400m | genre | template | t2t | r0 | 0.1478 | 0.1000 |
| siglip2_so400m | genre | template | t2t | center | 0.1852 | 0.1000 |
| siglip_so400m | emotion | bare | t2i | r0 | 0.1736 | 0.1171 |
| siglip_so400m | emotion | bare | t2i | center | 0.1451 | 0.1171 |
| siglip_so400m | emotion | bare | t2t | r0 | 0.1597 | 0.1110 |
| siglip_so400m | emotion | bare | t2t | center | 0.1857 | 0.1110 |
| siglip_so400m | emotion | template | t2i | r0 | 0.1950 | 0.1171 |
| siglip_so400m | emotion | template | t2i | center | 0.1598 | 0.1171 |
| siglip_so400m | emotion | template | t2t | r0 | 0.1390 | 0.1110 |
| siglip_so400m | emotion | template | t2t | center | 0.1616 | 0.1110 |
| siglip_so400m | style | bare | t2i | r0 | 0.3594 | 0.0370 |
| siglip_so400m | style | bare | t2i | center | 0.2459 | 0.0370 |
| siglip_so400m | style | bare | t2t | r0 | 0.0482 | 0.0370 |
| siglip_so400m | style | bare | t2t | center | 0.0541 | 0.0370 |
| siglip_so400m | style | template | t2i | r0 | 0.3011 | 0.0370 |
| siglip_so400m | style | template | t2i | center | 0.2220 | 0.0370 |
| siglip_so400m | style | template | t2t | r0 | 0.0442 | 0.0370 |
| siglip_so400m | style | template | t2t | center | 0.0467 | 0.0370 |
| siglip_so400m | genre | bare | t2i | r0 | 0.6640 | 0.1000 |
| siglip_so400m | genre | bare | t2i | center | 0.4516 | 0.1000 |
| siglip_so400m | genre | bare | t2t | r0 | 0.1678 | 0.1000 |
| siglip_so400m | genre | bare | t2t | center | 0.2109 | 0.1000 |
| siglip_so400m | genre | template | t2i | r0 | 0.5926 | 0.1000 |
| siglip_so400m | genre | template | t2i | center | 0.4105 | 0.1000 |
| siglip_so400m | genre | template | t2t | r0 | 0.1443 | 0.1000 |
| siglip_so400m | genre | template | t2t | center | 0.1770 | 0.1000 |
| cstar | emotion | bare | t2i | r0 | 0.2651 | 0.1171 |
| cstar | emotion | bare | t2i | center | 0.2671 | 0.1171 |
| cstar | emotion | bare | t2t | r0 | 0.4394 | 0.1110 |
| cstar | emotion | bare | t2t | center | 0.4417 | 0.1110 |
| cstar | emotion | template | t2i | r0 | 0.3074 | 0.1171 |
| cstar | emotion | template | t2i | center | 0.3060 | 0.1171 |
| cstar | emotion | template | t2t | r0 | 0.5205 | 0.1110 |
| cstar | emotion | template | t2t | center | 0.5193 | 0.1110 |
| cstar | style | bare | t2i | r0 | 0.2680 | 0.0370 |
| cstar | style | bare | t2i | center | 0.2373 | 0.0370 |
| cstar | style | bare | t2t | r0 | 0.0832 | 0.0370 |
| cstar | style | bare | t2t | center | 0.0853 | 0.0370 |
| cstar | style | template | t2i | r0 | 0.3115 | 0.0370 |
| cstar | style | template | t2i | center | 0.2535 | 0.0370 |
| cstar | style | template | t2t | r0 | 0.0867 | 0.0370 |
| cstar | style | template | t2t | center | 0.0886 | 0.0370 |
| cstar | genre | bare | t2i | r0 | 0.5947 | 0.1000 |
| cstar | genre | bare | t2i | center | 0.5368 | 0.1000 |
| cstar | genre | bare | t2t | r0 | 0.2809 | 0.1000 |
| cstar | genre | bare | t2t | center | 0.2997 | 0.1000 |
| cstar | genre | template | t2i | r0 | 0.6103 | 0.1000 |
| cstar | genre | template | t2i | center | 0.5685 | 0.1000 |
| cstar | genre | template | t2t | r0 | 0.3023 | 0.1000 |
| cstar | genre | template | t2t | center | 0.3318 | 0.1000 |
| cstar_free | emotion | bare | t2i | r0 | 0.2205 | 0.1171 |
| cstar_free | emotion | bare | t2i | center | 0.2261 | 0.1171 |
| cstar_free | emotion | bare | t2t | r0 | 0.3726 | 0.1110 |
| cstar_free | emotion | bare | t2t | center | 0.3861 | 0.1110 |
| cstar_free | emotion | template | t2i | r0 | 0.2678 | 0.1171 |
| cstar_free | emotion | template | t2i | center | 0.2739 | 0.1171 |
| cstar_free | emotion | template | t2t | r0 | 0.4384 | 0.1110 |
| cstar_free | emotion | template | t2t | center | 0.4430 | 0.1110 |
| cstar_free | style | bare | t2i | r0 | 0.3477 | 0.0370 |
| cstar_free | style | bare | t2i | center | 0.3384 | 0.0370 |
| cstar_free | style | bare | t2t | r0 | 0.0712 | 0.0370 |
| cstar_free | style | bare | t2t | center | 0.0689 | 0.0370 |
| cstar_free | style | template | t2i | r0 | 0.3345 | 0.0370 |
| cstar_free | style | template | t2i | center | 0.3346 | 0.0370 |
| cstar_free | style | template | t2t | r0 | 0.0668 | 0.0370 |
| cstar_free | style | template | t2t | center | 0.0639 | 0.0370 |
| cstar_free | genre | bare | t2i | r0 | 0.6762 | 0.1000 |
| cstar_free | genre | bare | t2i | center | 0.6849 | 0.1000 |
| cstar_free | genre | bare | t2t | r0 | 0.3131 | 0.1000 |
| cstar_free | genre | bare | t2t | center | 0.3266 | 0.1000 |
| cstar_free | genre | template | t2i | r0 | 0.6922 | 0.1000 |
| cstar_free | genre | template | t2i | center | 0.7138 | 0.1000 |
| cstar_free | genre | template | t2t | r0 | 0.3001 | 0.1000 |
| cstar_free | genre | template | t2t | center | 0.3174 | 0.1000 |
| teacher | emotion | bare | t2i | r0 | 0.2688 | 0.1171 |
| teacher | emotion | bare | t2i | center | 0.2718 | 0.1171 |
| teacher | emotion | bare | t2t | r0 | 0.5295 | 0.1110 |
| teacher | emotion | bare | t2t | center | 0.5481 | 0.1110 |
| teacher | emotion | template | t2i | r0 | 0.2801 | 0.1171 |
| teacher | emotion | template | t2i | center | 0.2796 | 0.1171 |
| teacher | emotion | template | t2t | r0 | 0.5428 | 0.1110 |
| teacher | emotion | template | t2t | center | 0.5536 | 0.1110 |
| teacher | style | bare | t2i | r0 | 0.3883 | 0.0370 |
| teacher | style | bare | t2i | center | 0.3780 | 0.0370 |
| teacher | style | bare | t2t | r0 | 0.0712 | 0.0370 |
| teacher | style | bare | t2t | center | 0.0771 | 0.0370 |
| teacher | style | template | t2i | r0 | 0.3655 | 0.0370 |
| teacher | style | template | t2i | center | 0.3747 | 0.0370 |
| teacher | style | template | t2t | r0 | 0.0721 | 0.0370 |
| teacher | style | template | t2t | center | 0.0785 | 0.0370 |
| teacher | genre | bare | t2i | r0 | 0.6894 | 0.1000 |
| teacher | genre | bare | t2i | center | 0.7063 | 0.1000 |
| teacher | genre | bare | t2t | r0 | 0.3218 | 0.1000 |
| teacher | genre | bare | t2t | center | 0.3518 | 0.1000 |
| teacher | genre | template | t2i | r0 | 0.6670 | 0.1000 |
| teacher | genre | template | t2i | center | 0.6875 | 0.1000 |
| teacher | genre | template | t2t | r0 | 0.2965 | 0.1000 |
| teacher | genre | template | t2t | center | 0.3298 | 0.1000 |


Outcome 1: cstar above in 6/9 cells; free variant 3/9.

The free variant changes the outcome; user decision required.

cstar wins label cells but loses pair MRR in some cells; user decision required on stating this limit.
