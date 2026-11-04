# r19 diagnostics (stage 4; descriptive, nothing here decides a verdict)

## r0

R0 = plain cosine retrieval mAP (no read-out map; every labelled row is a query and a gallery row).

### semart

| encoder | aspect | X-c* | plain image | matched (PercepT) | teacher | c* seed range |
|---|---|---|---|---|---|---|
| qwen | genre | 0.6910 | 0.5632 | 0.5419 | 0.6919 | 0.6902 to 0.6922 |
| siglip2 | genre | 0.6887 | 0.5659 | 0.5617 | 0.6919 | 0.6874 to 0.6907 |
| siglip1 | genre | 0.6916 | 0.5702 | 0.5828 | 0.6919 | 0.6903 to 0.6941 |
| clip_l14 | genre | 0.6763 | 0.5491 | 0.5556 | 0.6919 | 0.6736 to 0.6785 |
| dinov2_large | genre | 0.6401 | 0.5087 | 0.5360 | 0.6919 | 0.6357 to 0.6442 |
| clip_b32 | genre | 0.6347 | 0.5351 | 0.5334 | 0.6919 | 0.6338 to 0.6371 |
| clip_b16 | genre | 0.6578 | 0.5434 | 0.5360 | 0.6919 | 0.6564 to 0.6589 |

### mmart

| encoder | aspect | X-c* | plain image | matched (PercepT) | teacher | c* seed range |
|---|---|---|---|---|---|---|
| dinov2_large | emotion | 0.1921 | 0.1887 | 0.1861 | 0.2026 | 0.1917 to 0.1926 |
| dinov2_large | style | 0.2818 | 0.1906 | 0.1642 | 0.3347 | 0.2798 to 0.2839 |
| dinov2_large | genre | 0.5798 | 0.4455 | 0.5039 | 0.5918 | 0.5782 to 0.5814 |
| clip_b32 | emotion | 0.1942 | 0.1893 | 0.1856 | 0.2026 | 0.1936 to 0.1945 |
| clip_b32 | style | 0.3048 | 0.2305 | 0.1714 | 0.3347 | 0.3030 to 0.3064 |
| clip_b32 | genre | 0.5752 | 0.4351 | 0.5102 | 0.5918 | 0.5731 to 0.5765 |
| clip_b16 | emotion | 0.1962 | 0.1898 | 0.1863 | 0.2026 | 0.1958 to 0.1968 |
| clip_b16 | style | 0.3208 | 0.2419 | 0.1741 | 0.3347 | 0.3196 to 0.3225 |
| clip_b16 | genre | 0.5876 | 0.4472 | 0.5148 | 0.5918 | 0.5861 to 0.5891 |

### artelingo_held (new three; r14's `bs_diag.artelingo_held`)

| encoder | aspect | X-c* held | plain image held | X-c* validation |
|---|---|---|---|---|
| dinov2_large | emotion | 0.2482 | 0.2289 | 0.2527 |
| dinov2_large | style | 0.2552 | 0.1686 | 0.2584 |
| dinov2_large | genre | 0.5726 | 0.4395 | 0.5742 |
| clip_b32 | emotion | 0.2535 | 0.2384 | 0.2568 |
| clip_b32 | style | 0.2781 | 0.2088 | 0.2837 |
| clip_b32 | genre | 0.5686 | 0.4258 | 0.5757 |
| clip_b16 | emotion | 0.2571 | 0.2384 | 0.2601 |
| clip_b16 | style | 0.2944 | 0.2201 | 0.2984 |
| clip_b16 | genre | 0.5811 | 0.4368 | 0.5868 |

### affection_test (X photo model; validation R0 as r14 reports it)

| encoder | photo validation R0 mean |
|---|---|
| dinov2_large | 0.2804 |
| clip_b32 | 0.2994 |
| clip_b16 | 0.3095 |


## clustering_kclass

AMI of k-means, K = the class count beside K = 20 (mean over seeds and the three k-means states).

### semart

| encoder | aspect | K | c* K | c* 20 | plain K | plain 20 | matched K | matched 20 | teacher K | teacher 20 |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen | genre | 10 | 0.5531 | 0.4972 | 0.4471 | 0.4369 | 0.3882 | 0.3765 | 0.5583 | 0.4968 |
| siglip2 | genre | 10 | 0.5524 | 0.4983 | 0.4951 | 0.4630 | 0.4123 | 0.4022 | 0.5583 | 0.4968 |
| siglip1 | genre | 10 | 0.5546 | 0.4980 | 0.4786 | 0.4667 | 0.4346 | 0.4168 | 0.5583 | 0.4968 |
| clip_l14 | genre | 10 | 0.5466 | 0.4873 | 0.4784 | 0.4577 | 0.4123 | 0.4085 | 0.5583 | 0.4968 |
| dinov2_large | genre | 10 | 0.5059 | 0.4638 | 0.4253 | 0.4070 | 0.3948 | 0.3763 | 0.5583 | 0.4968 |
| clip_b32 | genre | 10 | 0.5036 | 0.4547 | 0.4607 | 0.4262 | 0.3834 | 0.3744 | 0.5583 | 0.4968 |
| clip_b16 | genre | 10 | 0.5238 | 0.4741 | 0.4790 | 0.4509 | 0.3922 | 0.3847 | 0.5583 | 0.4968 |

### mmart

| encoder | aspect | K | c* K | c* 20 | plain K | plain 20 | matched K | matched 20 | teacher K | teacher 20 |
|---|---|---|---|---|---|---|---|---|---|---|
| dinov2_large | emotion | 9 | 0.0449 | 0.0510 | 0.0328 | 0.0367 | 0.0372 | 0.0423 | 0.0492 | 0.0566 |
| dinov2_large | style | 20 | 0.3682 | 0.3682 | 0.2510 | 0.2510 | 0.1952 | 0.1952 | 0.4158 | 0.4158 |
| dinov2_large | genre | 10 | 0.5172 | 0.4978 | 0.4170 | 0.4255 | 0.4436 | 0.4299 | 0.5256 | 0.5071 |
| clip_b32 | emotion | 9 | 0.0460 | 0.0517 | 0.0350 | 0.0356 | 0.0373 | 0.0401 | 0.0492 | 0.0566 |
| clip_b32 | style | 20 | 0.3916 | 0.3916 | 0.3108 | 0.3108 | 0.2153 | 0.2153 | 0.4158 | 0.4158 |
| clip_b32 | genre | 10 | 0.5014 | 0.4924 | 0.4072 | 0.4128 | 0.4422 | 0.4284 | 0.5256 | 0.5071 |
| clip_b16 | emotion | 9 | 0.0473 | 0.0552 | 0.0353 | 0.0374 | 0.0366 | 0.0408 | 0.0492 | 0.0566 |
| clip_b16 | style | 20 | 0.4139 | 0.4139 | 0.3207 | 0.3207 | 0.2204 | 0.2204 | 0.4158 | 0.4158 |
| clip_b16 | genre | 10 | 0.5188 | 0.5051 | 0.4180 | 0.4409 | 0.4502 | 0.4378 | 0.5256 | 0.5071 |

## semart_close

D14: the SemArt set without the 217 paintings at nearest cosine >= 0.93 (10942 left). Reporting only: no bar changes and no verdict is replaced.

Any SemArt verdict would change: no.

| encoder | cell | without close: d [95% interval] | whole set: d [95% interval] | outcome whole -> without | rows |
|---|---|---|---|---|---|
| qwen | SC11 clu | +0.0594 [+0.0535, +0.0652] | +0.0603 [+0.0550, +0.0656] | WIN -> WIN | 10942 |
| qwen | SC13 clu | +0.1247 [+0.1136, +0.1351] | +0.1207 [+0.1111, +0.1295] | WIN -> WIN | 10942 |
| qwen | SC14 clu | +0.0036 [-0.0005, +0.0080] | +0.0004 [-0.0035, +0.0041] | LEVEL -> LEVEL | 10942 |
| qwen | SC12 ret | +0.1132 [+0.1110, +0.1154] | +0.1135 [+0.1114, +0.1156] | WIN -> WIN | 10942 |
| qwen | SC14 ret | -0.0009 [-0.0019, +0.0001] | -0.0009 [-0.0019, +0.0001] | LEVEL -> LEVEL | 10942 |
| siglip2 | SC11 clu | +0.0404 [+0.0356, +0.0455] | +0.0353 [+0.0310, +0.0398] | WIN -> WIN | 10942 |
| siglip2 | SC13 clu | +0.0921 [+0.0819, +0.1020] | +0.0962 [+0.0832, +0.1077] | WIN -> WIN | 10942 |
| siglip2 | SC14 clu | +0.0013 [-0.0027, +0.0056] | +0.0015 [-0.0027, +0.0052] | LEVEL -> LEVEL | 10942 |
| siglip2 | SC12 ret | +0.1049 [+0.1029, +0.1070] | +0.1052 [+0.1032, +0.1073] | WIN -> WIN | 10942 |
| siglip2 | SC14 ret | -0.0028 [-0.0043, -0.0013] | -0.0028 [-0.0042, -0.0013] | LEVEL -> LEVEL | 10942 |
| siglip1 | SC11 clu | +0.0401 [+0.0348, +0.0452] | +0.0312 [+0.0263, +0.0364] | WIN -> WIN | 10942 |
| siglip1 | SC13 clu | +0.0857 [+0.0749, +0.0967] | +0.0812 [+0.0675, +0.0936] | WIN -> WIN | 10942 |
| siglip1 | SC14 clu | +0.0059 [+0.0018, +0.0105] | +0.0011 [-0.0034, +0.0058] | LEVEL -> WIN | 10942 |
| siglip1 | SC12 ret | +0.1037 [+0.1015, +0.1059] | +0.1042 [+0.1020, +0.1064] | WIN -> WIN | 10942 |
| siglip1 | SC14 ret | -0.0007 [-0.0022, +0.0008] | -0.0008 [-0.0023, +0.0007] | LEVEL -> LEVEL | 10942 |
| clip_l14 | SC11 clu | +0.0297 [+0.0237, +0.0359] | +0.0296 [+0.0246, +0.0350] | WIN -> WIN | 10942 |
| clip_l14 | SC13 clu | +0.0833 [+0.0717, +0.0946] | +0.0788 [+0.0690, +0.0880] | WIN -> WIN | 10942 |
| clip_l14 | SC14 clu | -0.0053 [-0.0113, +0.0016] | -0.0095 [-0.0149, -0.0047] | LEVEL -> LEVEL | 10942 |
| clip_l14 | SC12 ret | +0.0845 [+0.0821, +0.0867] | +0.0845 [+0.0822, +0.0868] | WIN -> WIN | 10942 |
| clip_l14 | SC14 ret | -0.0135 [-0.0156, -0.0115] | -0.0133 [-0.0154, -0.0114] | BELOW -> BELOW | 10942 |
| dinov2_large | SC11 clu | +0.0596 [+0.0546, +0.0646] | +0.0568 [+0.0517, +0.0620] | WIN -> WIN | 10942 |
| dinov2_large | SC13 clu | +0.0838 [+0.0743, +0.0923] | +0.0875 [+0.0792, +0.0949] | WIN -> WIN | 10942 |
| dinov2_large | SC14 clu | -0.0321 [-0.0375, -0.0261] | -0.0330 [-0.0389, -0.0275] | BELOW -> BELOW | 10942 |
| dinov2_large | SC12 ret | +0.1064 [+0.1033, +0.1096] | +0.1063 [+0.1032, +0.1095] | WIN -> WIN | 10942 |
| dinov2_large | SC14 ret | -0.0445 [-0.0475, -0.0414] | -0.0442 [-0.0473, -0.0411] | BELOW -> BELOW | 10942 |
| clip_b32 | SC11 clu | +0.0286 [+0.0243, +0.0327] | +0.0285 [+0.0240, +0.0334] | WIN -> WIN | 10942 |
| clip_b32 | SC13 clu | +0.0808 [+0.0717, +0.0893] | +0.0803 [+0.0705, +0.0894] | WIN -> WIN | 10942 |
| clip_b32 | SC14 clu | -0.0404 [-0.0453, -0.0348] | -0.0422 [-0.0477, -0.0366] | BELOW -> BELOW | 10942 |
| clip_b32 | SC12 ret | +0.0764 [+0.0743, +0.0785] | +0.0762 [+0.0742, +0.0783] | WIN -> WIN | 10942 |
| clip_b32 | SC14 ret | -0.0505 [-0.0529, -0.0482] | -0.0504 [-0.0527, -0.0481] | BELOW -> BELOW | 10942 |
| clip_b16 | SC11 clu | +0.0295 [+0.0251, +0.0345] | +0.0231 [+0.0178, +0.0286] | WIN -> WIN | 10942 |
| clip_b16 | SC13 clu | +0.0914 [+0.0774, +0.1043] | +0.0894 [+0.0776, +0.1009] | WIN -> WIN | 10942 |
| clip_b16 | SC14 clu | -0.0215 [-0.0263, -0.0161] | -0.0228 [-0.0284, -0.0172] | BELOW -> BELOW | 10942 |
| clip_b16 | SC12 ret | +0.0910 [+0.0886, +0.0933] | +0.0910 [+0.0887, +0.0934] | WIN -> WIN | 10942 |
| clip_b16 | SC14 ret | -0.0291 [-0.0313, -0.0271] | -0.0289 [-0.0310, -0.0269] | BELOW -> BELOW | 10942 |

## semart_ap_type

Mean per-query AP per SemArt type at each arm's pick (c*: mean of seeds 0 to 4); queries are the paintings of the type.

### qwen

| type | queries | X-c* | plain image | teacher |
|---|---|---|---|---|
| genre | 884 | 0.2564 | 0.1845 | 0.2553 |
| historical | 344 | 0.1778 | 0.1263 | 0.1778 |
| interior | 323 | 0.2400 | 0.3021 | 0.2138 |
| landscape | 1548 | 0.7382 | 0.5784 | 0.7523 |
| mythological | 1175 | 0.4518 | 0.3725 | 0.4285 |
| other | 253 | 0.0487 | 0.0434 | 0.0470 |
| portrait | 1686 | 0.8614 | 0.7056 | 0.8729 |
| religious | 4100 | 0.8865 | 0.7595 | 0.8871 |
| still-life | 822 | 0.9206 | 0.8235 | 0.9252 |
| study | 24 | 0.0295 | 0.0168 | 0.0218 |

### siglip2

| type | queries | X-c* | plain image | teacher |
|---|---|---|---|---|
| genre | 884 | 0.2660 | 0.2380 | 0.2553 |
| historical | 344 | 0.1754 | 0.1233 | 0.1778 |
| interior | 323 | 0.2708 | 0.2469 | 0.2138 |
| landscape | 1548 | 0.7449 | 0.6087 | 0.7523 |
| mythological | 1175 | 0.4543 | 0.4026 | 0.4285 |
| other | 253 | 0.0487 | 0.0413 | 0.0470 |
| portrait | 1686 | 0.8636 | 0.6759 | 0.8729 |
| religious | 4100 | 0.8746 | 0.7667 | 0.8871 |
| still-life | 822 | 0.9125 | 0.8016 | 0.9252 |
| study | 24 | 0.0333 | 0.0275 | 0.0218 |

### siglip1

| type | queries | X-c* | plain image | teacher |
|---|---|---|---|---|
| genre | 884 | 0.2718 | 0.2532 | 0.2553 |
| historical | 344 | 0.1751 | 0.1103 | 0.1778 |
| interior | 323 | 0.2511 | 0.2456 | 0.2138 |
| landscape | 1548 | 0.7526 | 0.6190 | 0.7523 |
| mythological | 1175 | 0.4428 | 0.4000 | 0.4285 |
| other | 253 | 0.0478 | 0.0427 | 0.0470 |
| portrait | 1686 | 0.8651 | 0.6407 | 0.8729 |
| religious | 4100 | 0.8788 | 0.7735 | 0.8871 |
| still-life | 822 | 0.9189 | 0.8536 | 0.9252 |
| study | 24 | 0.0300 | 0.0267 | 0.0218 |

### clip_l14

| type | queries | X-c* | plain image | teacher |
|---|---|---|---|---|
| genre | 884 | 0.2643 | 0.2340 | 0.2553 |
| historical | 344 | 0.1588 | 0.1179 | 0.1778 |
| interior | 323 | 0.2613 | 0.2678 | 0.2138 |
| landscape | 1548 | 0.7415 | 0.6227 | 0.7523 |
| mythological | 1175 | 0.4281 | 0.4052 | 0.4285 |
| other | 253 | 0.0487 | 0.0449 | 0.0470 |
| portrait | 1686 | 0.8546 | 0.6735 | 0.8729 |
| religious | 4100 | 0.8601 | 0.7814 | 0.8871 |
| still-life | 822 | 0.9160 | 0.8376 | 0.9252 |
| study | 24 | 0.0345 | 0.0235 | 0.0218 |

### dinov2_large

| type | queries | X-c* | plain image | teacher |
|---|---|---|---|---|
| genre | 884 | 0.2453 | 0.1529 | 0.2553 |
| historical | 344 | 0.1252 | 0.0753 | 0.1778 |
| interior | 323 | 0.2613 | 0.2549 | 0.2138 |
| landscape | 1548 | 0.7149 | 0.6365 | 0.7523 |
| mythological | 1175 | 0.3804 | 0.2646 | 0.4285 |
| other | 253 | 0.0467 | 0.0400 | 0.0470 |
| portrait | 1686 | 0.8052 | 0.6466 | 0.8729 |
| religious | 4100 | 0.8322 | 0.7164 | 0.8871 |
| still-life | 822 | 0.8907 | 0.7886 | 0.9252 |
| study | 24 | 0.0223 | 0.0237 | 0.0218 |

### clip_b32

| type | queries | X-c* | plain image | teacher |
|---|---|---|---|---|
| genre | 884 | 0.2406 | 0.2021 | 0.2553 |
| historical | 344 | 0.1152 | 0.0983 | 0.1778 |
| interior | 323 | 0.2634 | 0.2688 | 0.2138 |
| landscape | 1548 | 0.7210 | 0.6344 | 0.7523 |
| mythological | 1175 | 0.3642 | 0.3165 | 0.4285 |
| other | 253 | 0.0456 | 0.0408 | 0.0470 |
| portrait | 1686 | 0.8243 | 0.6620 | 0.8729 |
| religious | 4100 | 0.8107 | 0.7598 | 0.8871 |
| still-life | 822 | 0.8956 | 0.7267 | 0.9252 |
| study | 24 | 0.0281 | 0.0243 | 0.0218 |

### clip_b16

| type | queries | X-c* | plain image | teacher |
|---|---|---|---|---|
| genre | 884 | 0.2575 | 0.2147 | 0.2553 |
| historical | 344 | 0.1377 | 0.1097 | 0.1778 |
| interior | 323 | 0.2660 | 0.2727 | 0.2138 |
| landscape | 1548 | 0.7327 | 0.6151 | 0.7523 |
| mythological | 1175 | 0.4014 | 0.3389 | 0.4285 |
| other | 253 | 0.0473 | 0.0408 | 0.0470 |
| portrait | 1686 | 0.8370 | 0.6172 | 0.8729 |
| religious | 4100 | 0.8392 | 0.7850 | 0.8871 |
| still-life | 822 | 0.9139 | 0.7672 | 0.9252 |
| study | 24 | 0.0299 | 0.0220 | 0.0218 |

## semart_inner

SemArt pairs inside the set (default Qwen rows) and the md5 step.

| set paintings | pairs at cosine >= 0.95 | paintings in a pair | md5 groups | md5 duplicates removed | before md5 |
|---|---|---|---|---|---|
| 11159 | 472 | 698 | 3 | 3 | 11162 |

## r6_5_held

R6-5 student (10-25 and 10-26 rounds; stored files) beside c* (this round, SC1 candidate value): ArtELingo held, AMI with 20 clusters, five seeds. DINOv2 R6-5 is the sample-scale run (the full run was skipped).

| encoder | aspect | R6-5 | c* | stored block |
|---|---|---|---|---|
| clip_b32 | emotion | 0.0885 | 0.0919 | full K3 S2-IN |
| clip_b32 | style | 0.3693 | 0.3743 | full K3 S2-IN |
| clip_b32 | genre | 0.4484 | 0.4892 | full K3 S2-IN |
| clip_b16 | emotion | 0.0945 | 0.0968 | full K6 S2-IN |
| clip_b16 | style | 0.3894 | 0.3922 | full K6 S2-IN |
| clip_b16 | genre | 0.4575 | 0.4952 | full K6 S2-IN |
| dinov2_large | emotion | 0.0814 | 0.0859 | sample K5 S1-NO |
| dinov2_large | style | 0.3342 | 0.3444 | sample K5 S1-NO |
| dinov2_large | genre | 0.4544 | 0.4866 | sample K5 S1-NO |

## gain_share

Share of the teacher's gain over the plain image that X-c* keeps: (c* - plain) / (teacher - plain). A gap of zero or below gives no share.

| encoder | set | aspect | task | plain | X-c* | teacher | gap | share |
|---|---|---|---|---|---|---|---|---|
| clip_b16 | affection_test | emotion | clu | 0.1147 | 0.1808 | 0.1666 | 0.0519 | 1.273 |
| clip_b16 | affection_test | emotion | ret | 0.2559 | 0.3125 | 0.2945 | 0.0386 | 1.464 |
| clip_b16 | artelingo_held | emotion | clu | 0.0631 | 0.0968 | 0.0921 | 0.0289 | 1.165 |
| clip_b16 | artelingo_held | emotion | ret | 0.2469 | 0.2624 | 0.2684 | 0.0215 | 0.721 |
| clip_b16 | artelingo_held | genre | clu | 0.4119 | 0.4952 | 0.5062 | 0.0943 | 0.884 |
| clip_b16 | artelingo_held | genre | ret | 0.5040 | 0.6081 | 0.6124 | 0.1084 | 0.960 |
| clip_b16 | artelingo_held | style | clu | 0.3066 | 0.3922 | 0.4026 | 0.0959 | 0.892 |
| clip_b16 | artelingo_held | style | ret | 0.2457 | 0.3086 | 0.3217 | 0.0760 | 0.828 |
| clip_b16 | mmart_test | emotion | clu | 0.0374 | 0.0552 | 0.0566 | 0.0192 | 0.930 |
| clip_b16 | mmart_test | emotion | ret | 0.1908 | 0.1978 | 0.2021 | 0.0113 | 0.622 |
| clip_b16 | mmart_test | genre | clu | 0.4409 | 0.5051 | 0.5071 | 0.0662 | 0.970 |
| clip_b16 | mmart_test | genre | ret | 0.5143 | 0.6143 | 0.6172 | 0.1029 | 0.972 |
| clip_b16 | mmart_test | style | clu | 0.3207 | 0.4139 | 0.4158 | 0.0951 | 0.980 |
| clip_b16 | mmart_test | style | ret | 0.2685 | 0.3344 | 0.3476 | 0.0790 | 0.833 |
| clip_b16 | semart_new | genre | clu | 0.4509 | 0.4741 | 0.4968 | 0.0459 | 0.504 |
| clip_b16 | semart_new | genre | ret | 0.5885 | 0.6795 | 0.7085 | 0.1200 | 0.759 |
| clip_b32 | affection_test | emotion | clu | 0.1116 | 0.1711 | 0.1666 | 0.0550 | 1.082 |
| clip_b32 | affection_test | emotion | ret | 0.2527 | 0.3049 | 0.2945 | 0.0418 | 1.248 |
| clip_b32 | artelingo_held | emotion | clu | 0.0642 | 0.0919 | 0.0921 | 0.0279 | 0.993 |
| clip_b32 | artelingo_held | emotion | ret | 0.2472 | 0.2592 | 0.2684 | 0.0212 | 0.564 |
| clip_b32 | artelingo_held | genre | clu | 0.4085 | 0.4892 | 0.5062 | 0.0977 | 0.826 |
| clip_b32 | artelingo_held | genre | ret | 0.4905 | 0.5981 | 0.6124 | 0.1220 | 0.882 |
| clip_b32 | artelingo_held | style | clu | 0.2978 | 0.3743 | 0.4026 | 0.1047 | 0.730 |
| clip_b32 | artelingo_held | style | ret | 0.2348 | 0.2928 | 0.3217 | 0.0869 | 0.667 |
| clip_b32 | mmart_test | emotion | clu | 0.0356 | 0.0517 | 0.0566 | 0.0209 | 0.768 |
| clip_b32 | mmart_test | emotion | ret | 0.1909 | 0.1958 | 0.2021 | 0.0112 | 0.443 |
| clip_b32 | mmart_test | genre | clu | 0.4127 | 0.4924 | 0.5071 | 0.0943 | 0.844 |
| clip_b32 | mmart_test | genre | ret | 0.5004 | 0.6037 | 0.6172 | 0.1168 | 0.884 |
| clip_b32 | mmart_test | style | clu | 0.3110 | 0.3916 | 0.4158 | 0.1049 | 0.769 |
| clip_b32 | mmart_test | style | ret | 0.2566 | 0.3188 | 0.3476 | 0.0910 | 0.684 |
| clip_b32 | semart_new | genre | clu | 0.4262 | 0.4547 | 0.4968 | 0.0707 | 0.403 |
| clip_b32 | semart_new | genre | ret | 0.5819 | 0.6581 | 0.7085 | 0.1266 | 0.602 |
| clip_l14 | semart_new | genre | clu | 0.4577 | 0.4873 | 0.4968 | 0.0392 | 0.757 |
| clip_l14 | semart_new | genre | ret | 0.6106 | 0.6951 | 0.7085 | 0.0979 | 0.864 |
| dinov2_large | affection_test | emotion | clu | 0.0875 | 0.1547 | 0.1666 | 0.0791 | 0.850 |
| dinov2_large | affection_test | emotion | ret | 0.2273 | 0.2870 | 0.2945 | 0.0672 | 0.888 |
| dinov2_large | artelingo_held | emotion | clu | 0.0602 | 0.0859 | 0.0921 | 0.0318 | 0.807 |
| dinov2_large | artelingo_held | emotion | ret | 0.2368 | 0.2549 | 0.2684 | 0.0317 | 0.573 |
| dinov2_large | artelingo_held | genre | clu | 0.4209 | 0.4866 | 0.5062 | 0.0852 | 0.771 |
| dinov2_large | artelingo_held | genre | ret | 0.5051 | 0.6007 | 0.6124 | 0.1074 | 0.890 |
| dinov2_large | artelingo_held | style | clu | 0.2354 | 0.3444 | 0.4026 | 0.1672 | 0.652 |
| dinov2_large | artelingo_held | style | ret | 0.1791 | 0.2657 | 0.3217 | 0.1426 | 0.607 |
| dinov2_large | mmart_test | emotion | clu | 0.0367 | 0.0510 | 0.0566 | 0.0199 | 0.718 |
| dinov2_large | mmart_test | emotion | ret | 0.1890 | 0.1939 | 0.2021 | 0.0131 | 0.375 |
| dinov2_large | mmart_test | genre | clu | 0.4255 | 0.4978 | 0.5071 | 0.0816 | 0.887 |
| dinov2_large | mmart_test | genre | ret | 0.4632 | 0.6078 | 0.6172 | 0.1540 | 0.939 |
| dinov2_large | mmart_test | style | clu | 0.2510 | 0.3682 | 0.4158 | 0.1649 | 0.711 |
| dinov2_large | mmart_test | style | ret | 0.1929 | 0.2924 | 0.3476 | 0.1546 | 0.643 |
| dinov2_large | semart_new | genre | clu | 0.4070 | 0.4638 | 0.4968 | 0.0899 | 0.632 |
| dinov2_large | semart_new | genre | ret | 0.5580 | 0.6642 | 0.7085 | 0.1505 | 0.706 |
| qwen | semart_new | genre | clu | 0.4369 | 0.4972 | 0.4968 | 0.0600 | 1.006 |
| qwen | semart_new | genre | ret | 0.5941 | 0.7075 | 0.7085 | 0.1144 | 0.992 |
| siglip1 | semart_new | genre | clu | 0.4667 | 0.4980 | 0.4968 | 0.0301 | 1.037 |
| siglip1 | semart_new | genre | ret | 0.6035 | 0.7077 | 0.7085 | 0.1050 | 0.992 |
| siglip2 | semart_new | genre | clu | 0.4630 | 0.4983 | 0.4968 | 0.0338 | 1.044 |
| siglip2 | semart_new | genre | ret | 0.6005 | 0.7057 | 0.7085 | 0.1080 | 0.974 |

## seed_spreads

Seed spread (max - min over seeds 0 to 4) of X-c* on test beside the validation range (the LEVEL unit r of the teacher cell, unrounded; "n/a" when the cell file is missing). ArtELingo held and Affection test: r14's `bs_diag` values against the stage 1 ranges.

| set | encoder | aspect | AMI spread (test) | AMI range (val) | ratio | mAP R0 spread (test) | mAP range (val) | ratio |
|---|---|---|---|---|---|---|---|---|
| semart | qwen | genre | 0.0068 | 0.0139 | 0.49 | 0.0020 | 0.0038 | 0.52 |
| semart | siglip2 | genre | 0.0030 | 0.0070 | 0.43 | 0.0033 | 0.0043 | 0.79 |
| semart | siglip1 | genre | 0.0075 | 0.0115 | 0.65 | 0.0038 | 0.0041 | 0.93 |
| semart | clip_l14 | genre | 0.0077 | 0.0159 | 0.48 | 0.0050 | 0.0017 | 3.00 |
| semart | dinov2_large | genre | 0.0075 | 0.0116 | 0.65 | 0.0085 | 0.0044 | 1.95 |
| semart | clip_b32 | genre | 0.0081 | 0.0108 | 0.75 | 0.0033 | 0.0025 | 1.30 |
| semart | clip_b16 | genre | 0.0096 | 0.0121 | 0.79 | 0.0025 | 0.0027 | 0.94 |
| mmart | dinov2_large | emotion | 0.0010 | 0.0021 | 0.49 | 0.0009 | 0.0009 | 1.02 |
| mmart | dinov2_large | style | 0.0068 | 0.0040 | 1.70 | 0.0041 | 0.0032 | 1.25 |
| mmart | dinov2_large | genre | 0.0132 | 0.0112 | 1.18 | 0.0033 | 0.0042 | 0.77 |
| mmart | clip_b32 | emotion | 0.0019 | 0.0037 | 0.51 | 0.0010 | 0.0010 | 0.95 |
| mmart | clip_b32 | style | 0.0039 | 0.0122 | 0.32 | 0.0035 | 0.0038 | 0.91 |
| mmart | clip_b32 | genre | 0.0106 | 0.0147 | 0.72 | 0.0034 | 0.0025 | 1.34 |
| mmart | clip_b16 | emotion | 0.0019 | 0.0053 | 0.36 | 0.0010 | 0.0012 | 0.84 |
| mmart | clip_b16 | style | 0.0085 | 0.0135 | 0.63 | 0.0029 | 0.0037 | 0.77 |
| mmart | clip_b16 | genre | 0.0094 | 0.0059 | 1.58 | 0.0029 | 0.0027 | 1.07 |


| set | encoder | aspect | AMI spread | AMI range (val) | ratio | mAP spread (R0; Affection at pick) | mAP range (val) | ratio |
|---|---|---|---|---|---|---|---|---|
| artelingo_held | dinov2_large | emotion | 0.0024 | 0.0043 | 0.55 | 0.0029 | 0.0032 | 0.91 |
| artelingo_held | dinov2_large | style | 0.0057 | 0.0110 | 0.52 | 0.0034 | 0.0032 | 1.07 |
| artelingo_held | dinov2_large | genre | 0.0069 | 0.0116 | 0.59 | 0.0036 | 0.0044 | 0.83 |
| artelingo_held | clip_b32 | emotion | 0.0040 | 0.0063 | 0.63 | 0.0018 | 0.0019 | 0.93 |
| artelingo_held | clip_b32 | style | 0.0083 | 0.0091 | 0.91 | 0.0020 | 0.0030 | 0.65 |
| artelingo_held | clip_b32 | genre | 0.0096 | 0.0108 | 0.89 | 0.0029 | 0.0025 | 1.14 |
| artelingo_held | clip_b16 | emotion | 0.0023 | 0.0031 | 0.74 | 0.0015 | 0.0017 | 0.86 |
| artelingo_held | clip_b16 | style | 0.0081 | 0.0138 | 0.59 | 0.0038 | 0.0037 | 1.03 |
| artelingo_held | clip_b16 | genre | 0.0139 | 0.0121 | 1.15 | 0.0029 | 0.0027 | 1.07 |
| affection_test | dinov2_large | emotion | 0.0095 | 0.0057 | 1.67 | 0.0019 (at pick) | 0.0021 | 0.89 |
| affection_test | clip_b32 | emotion | 0.0060 | 0.0121 | 0.49 | 0.0052 (at pick) | 0.0051 | 1.04 |
| affection_test | clip_b16 | emotion | 0.0041 | 0.0049 | 0.84 | 0.0064 (at pick) | 0.0062 | 1.03 |


## percept_strongest

Strongest PercepT entry per set and aspect (C1): the largest matched-arm value (the cell's comparator value, the larger of its two readings) over the encoders that have an SC3, SC9 or SC13 cell.

| set | aspect | entry | encoder | cell | AMI |
|---|---|---|---|---|---|
| artelingo_held | emotion | M6 | clip_b16 | SC3 | 0.0749 |
| artelingo_held | genre | M6 | clip_b16 | SC3 | 0.4309 |
| artelingo_held | style | M6 | clip_b16 | SC3 | 0.2228 |
| mmart_test | emotion | M5 | dinov2_large | SC9 | 0.0437 |
| mmart_test | genre | M6 | clip_b16 | SC9 | 0.4378 |
| mmart_test | style | M6 | clip_b16 | SC9 | 0.2274 |
| semart_new | genre | M8 | siglip1 | SC13 | 0.4168 |

## cost_rows

Cost table, other rows (from the probe: `stage3/cost_table.json`).

| pass | kind | ms/image | images/s | ms/image e2e | images/s e2e | peak MiB | parameters | vision tower share |
|---|---|---|---|---|---|---|---|---|
| qwen | model | 88.59 | 11.3 | 88.42 | 11.3 | 7389 | 2127532032 | {"total_seconds": 176.3419383582659, "visual_seconds": 62.88 |
| qwen_emotion_focus | model | 89.77 | 11.1 | n/a | n/a | 7476 | 2127532032 | n/a |
| qwen_style_word | model | 88.73 | 11.3 | n/a | n/a | 7414 | 2127532032 | n/a |
| qwen_genre_word | model | 88.34 | 11.3 | n/a | n/a | 7409 | 2127532032 | n/a |
| teacher | model | n/a | n/a | 267.61 | 3.7 | 7476 | n/a | n/a |
| siglip2 | model | 19.34 | 51.7 | 18.95 | 52.8 | 2842 | 1136008498 | n/a |
| siglip1 | model | 19.44 | 51.4 | 19.34 | 51.7 | 2350 | 877960498 | n/a |
| clip_l14 | model | 7.86 | 127.1 | 8.61 | 116.1 | 1124 | 427616513 | n/a |
| dinov2_large | model | 8.50 | 117.7 | 9.33 | 107.2 | 848 | 304368640 | n/a |
| clip_b32 | model | 5.15 | 194.0 | 6.90 | 144.9 | 376 | 151277313 | n/a |
| clip_b16 | model | 5.73 | 174.6 | 7.24 | 138.2 | 483 | 149620737 | n/a |
| heads_qwen | heads | 0.04 | 28358.5 | n/a | n/a | 82 | n/a | n/a |
| heads_siglip2 | heads | 0.03 | 28658.6 | n/a | n/a | 72 | n/a | n/a |
| heads_siglip1 | heads | 0.03 | 29770.6 | n/a | n/a | 72 | n/a | n/a |
| heads_clip_l14 | heads | 0.04 | 27356.0 | n/a | n/a | 68 | n/a | n/a |
| heads_dinov2_large | heads | 0.03 | 31076.8 | n/a | n/a | 71 | n/a | n/a |
| heads_clip_b32 | heads | 0.04 | 25591.4 | n/a | n/a | 65 | n/a | n/a |
| heads_clip_b16 | heads | 0.04 | 28504.1 | n/a | n/a | 64 | n/a | n/a |

Passes per aspect: {"cstar": 0.3333333333333333, "teacher": 1} (c* shares one encoder pass among three aspects; the teacher runs one instructed Qwen pass per aspect).

| encoder | c* s (16,384 paintings, model) | teacher s | c* s e2e | teacher s e2e |
|---|---|---|---|---|
| qwen | 1452.1 | 4371.8 | 1449.3 | 4384.5 |
| siglip2 | 317.4 | 4371.8 | 311.1 | 4384.5 |
| siglip1 | 319.1 | 4371.8 | 317.4 | 4384.5 |
| clip_l14 | 129.5 | 4371.8 | 141.7 | 4384.5 |
| dinov2_large | 139.8 | 4371.8 | 153.3 | 4384.5 |
| clip_b32 | 85.1 | 4371.8 | 113.7 | 4384.5 |
| clip_b16 | 94.4 | 4371.8 | 119.1 | 4384.5 |

Prep log Qwen (store extraction): {"qwen_default": {"seconds": 1448.5740305329673, "paintings": 16782, "ms_per_image": 86.31712731098601}, "emotion_focus": {"seconds": 1489.1076726652682, "paintings": 16782, "ms_per_image": 88.7324319309539}, "style_word": {"seconds": 1472.3615040858276, "paintings": 16782, "ms_per_image": 87.73456704122438}, "genre_word": {"seconds": 1466.1030863639899, "paintings": 16782, "ms_per_image": 87.36164261494397}}

## node_time

node402 time per run (`node/units.jsonl`; seconds of the unit, including its export).

| kind | encoder | runs | ok | failed | mean s | min s | max s | total s | host |
|---|---|---|---|---|---|---|---|---|---|
| cstar | clip_b16 | 5 | 5 | 0 | 531 | 503 | 560 | 2657 | node402 |
| cstar | clip_b32 | 5 | 5 | 0 | 501 | 478 | 521 | 2507 | node402 |
| cstar | dinov2_large | 5 | 5 | 0 | 449 | 435 | 464 | 2245 | node402 |
| ii_cstar | clip_l14 | 1 | 1 | 0 | 502 | 502 | 502 | 502 | node402 |
| ii_xphoto | clip_l14 | 1 | 1 | 0 | 264 | 264 | 264 | 264 | node402 |
| xphoto | clip_b16 | 5 | 5 | 0 | 207 | 172 | 267 | 1033 | node402 |
| xphoto | clip_b32 | 5 | 5 | 0 | 207 | 184 | 242 | 1036 | node402 |
| xphoto | dinov2_large | 5 | 5 | 0 | 195 | 180 | 240 | 976 | node402 |

## r17_metrics

r17's NMI, ARI, ACC and the number of non-empty clusters (`ns_metrics`, read only) on this round's clustering cells; mean over seeds (c*, matched) and the three k-means states. Context only (D13): no claim and no bar.

### semart

| encoder | aspect | K | arm | NMI | ARI | ACC | clusters |
|---|---|---|---|---|---|---|---|
| qwen | genre | 10 | cstar | 0.5539 | 0.4613 | 0.5877 | 10.0 |
| qwen | genre | 10 | matched | 0.3893 | 0.2670 | 0.4402 | 10.0 |
| qwen | genre | 10 | frozen | 0.4481 | 0.3017 | 0.4778 | 10.0 |
| qwen | genre | 10 | teacher | 0.5591 | 0.5039 | 0.6272 | 10.0 |
| qwen | genre | 20 | cstar | 0.4989 | 0.2805 | 0.3857 | 20.0 |
| qwen | genre | 20 | matched | 0.3785 | 0.1945 | 0.3091 | 20.0 |
| qwen | genre | 20 | frozen | 0.4387 | 0.2096 | 0.3240 | 20.0 |
| qwen | genre | 20 | teacher | 0.4985 | 0.2984 | 0.3999 | 20.0 |
| siglip2 | genre | 10 | cstar | 0.5532 | 0.4438 | 0.5833 | 10.0 |
| siglip2 | genre | 10 | matched | 0.4133 | 0.3154 | 0.4700 | 10.0 |
| siglip2 | genre | 10 | frozen | 0.4960 | 0.3563 | 0.5336 | 10.0 |
| siglip2 | genre | 10 | teacher | 0.5591 | 0.5039 | 0.6272 | 10.0 |
| siglip2 | genre | 20 | cstar | 0.5000 | 0.2864 | 0.3864 | 20.0 |
| siglip2 | genre | 20 | matched | 0.4041 | 0.2287 | 0.3368 | 20.0 |
| siglip2 | genre | 20 | frozen | 0.4647 | 0.2298 | 0.3545 | 20.0 |
| siglip2 | genre | 20 | teacher | 0.4985 | 0.2984 | 0.3999 | 20.0 |
| siglip1 | genre | 10 | cstar | 0.5554 | 0.4494 | 0.5865 | 10.0 |
| siglip1 | genre | 10 | matched | 0.4356 | 0.3412 | 0.4919 | 10.0 |
| siglip1 | genre | 10 | frozen | 0.4795 | 0.3341 | 0.4804 | 10.0 |
| siglip1 | genre | 10 | teacher | 0.5591 | 0.5039 | 0.6272 | 10.0 |
| siglip1 | genre | 20 | cstar | 0.4996 | 0.2752 | 0.3886 | 20.0 |
| siglip1 | genre | 20 | matched | 0.4187 | 0.2416 | 0.3409 | 20.0 |
| siglip1 | genre | 20 | frozen | 0.4685 | 0.2300 | 0.3433 | 20.0 |
| siglip1 | genre | 20 | teacher | 0.4985 | 0.2984 | 0.3999 | 20.0 |
| clip_l14 | genre | 10 | cstar | 0.5474 | 0.4471 | 0.5873 | 10.0 |
| clip_l14 | genre | 10 | matched | 0.4133 | 0.3085 | 0.4648 | 10.0 |
| clip_l14 | genre | 10 | frozen | 0.4793 | 0.3292 | 0.4855 | 10.0 |
| clip_l14 | genre | 10 | teacher | 0.5591 | 0.5039 | 0.6272 | 10.0 |
| clip_l14 | genre | 20 | cstar | 0.4890 | 0.2625 | 0.3779 | 20.0 |
| clip_l14 | genre | 20 | matched | 0.4105 | 0.2395 | 0.3495 | 20.0 |
| clip_l14 | genre | 20 | frozen | 0.4594 | 0.2196 | 0.3558 | 20.0 |
| clip_l14 | genre | 20 | teacher | 0.4985 | 0.2984 | 0.3999 | 20.0 |
| dinov2_large | genre | 10 | cstar | 0.5068 | 0.3999 | 0.5502 | 10.0 |
| dinov2_large | genre | 10 | matched | 0.3960 | 0.3010 | 0.4749 | 10.0 |
| dinov2_large | genre | 10 | frozen | 0.4263 | 0.3032 | 0.4808 | 10.0 |
| dinov2_large | genre | 10 | teacher | 0.5591 | 0.5039 | 0.6272 | 10.0 |
| dinov2_large | genre | 20 | cstar | 0.4656 | 0.2601 | 0.3672 | 20.0 |
| dinov2_large | genre | 20 | matched | 0.3783 | 0.2052 | 0.3228 | 20.0 |
| dinov2_large | genre | 20 | frozen | 0.4089 | 0.1993 | 0.3318 | 20.0 |
| dinov2_large | genre | 20 | teacher | 0.4985 | 0.2984 | 0.3999 | 20.0 |
| clip_b32 | genre | 10 | cstar | 0.5044 | 0.3911 | 0.5441 | 10.0 |
| clip_b32 | genre | 10 | matched | 0.3846 | 0.2841 | 0.4535 | 10.0 |
| clip_b32 | genre | 10 | frozen | 0.4617 | 0.3295 | 0.5203 | 10.0 |
| clip_b32 | genre | 10 | teacher | 0.5591 | 0.5039 | 0.6272 | 10.0 |
| clip_b32 | genre | 20 | cstar | 0.4565 | 0.2452 | 0.3644 | 20.0 |
| clip_b32 | genre | 20 | matched | 0.3765 | 0.2104 | 0.3283 | 20.0 |
| clip_b32 | genre | 20 | frozen | 0.4281 | 0.2015 | 0.3377 | 20.0 |
| clip_b32 | genre | 20 | teacher | 0.4985 | 0.2984 | 0.3999 | 20.0 |
| clip_b16 | genre | 10 | cstar | 0.5246 | 0.4198 | 0.5594 | 10.0 |
| clip_b16 | genre | 10 | matched | 0.3933 | 0.3024 | 0.4672 | 10.0 |
| clip_b16 | genre | 10 | frozen | 0.4799 | 0.3635 | 0.5464 | 10.0 |
| clip_b16 | genre | 10 | teacher | 0.5591 | 0.5039 | 0.6272 | 10.0 |
| clip_b16 | genre | 20 | cstar | 0.4758 | 0.2628 | 0.3705 | 20.0 |
| clip_b16 | genre | 20 | matched | 0.3867 | 0.2187 | 0.3312 | 20.0 |
| clip_b16 | genre | 20 | frozen | 0.4527 | 0.2302 | 0.3554 | 20.0 |
| clip_b16 | genre | 20 | teacher | 0.4985 | 0.2984 | 0.3999 | 20.0 |

### mmart

| encoder | aspect | K | arm | NMI | ARI | ACC | clusters |
|---|---|---|---|---|---|---|---|
| dinov2_large | emotion | 9 | cstar | 0.0458 | 0.0260 | 0.1931 | 9.0 |
| dinov2_large | emotion | 9 | matched | 0.0382 | 0.0200 | 0.1924 | 9.0 |
| dinov2_large | emotion | 9 | frozen | 0.0338 | 0.0185 | 0.1798 | 9.0 |
| dinov2_large | emotion | 9 | teacher | 0.0501 | 0.0309 | 0.2087 | 9.0 |
| dinov2_large | emotion | 20 | cstar | 0.0528 | 0.0199 | 0.1229 | 20.0 |
| dinov2_large | emotion | 20 | matched | 0.0442 | 0.0151 | 0.1223 | 20.0 |
| dinov2_large | emotion | 20 | frozen | 0.0385 | 0.0146 | 0.1144 | 20.0 |
| dinov2_large | emotion | 20 | teacher | 0.0584 | 0.0265 | 0.1331 | 20.0 |
| dinov2_large | style | 20 | cstar | 0.3707 | 0.1660 | 0.3112 | 20.0 |
| dinov2_large | style | 20 | matched | 0.1984 | 0.0611 | 0.1744 | 20.0 |
| dinov2_large | style | 20 | frozen | 0.2540 | 0.0827 | 0.2145 | 20.0 |
| dinov2_large | style | 20 | teacher | 0.4181 | 0.1945 | 0.3378 | 20.0 |
| dinov2_large | genre | 10 | cstar | 0.5179 | 0.4620 | 0.6272 | 10.0 |
| dinov2_large | genre | 10 | matched | 0.4443 | 0.3407 | 0.4897 | 10.0 |
| dinov2_large | genre | 10 | frozen | 0.4178 | 0.3134 | 0.5009 | 10.0 |
| dinov2_large | genre | 10 | teacher | 0.5262 | 0.4600 | 0.6206 | 10.0 |
| dinov2_large | genre | 20 | cstar | 0.4991 | 0.3385 | 0.4655 | 20.0 |
| dinov2_large | genre | 20 | matched | 0.4314 | 0.2613 | 0.3683 | 20.0 |
| dinov2_large | genre | 20 | frozen | 0.4269 | 0.2481 | 0.3813 | 20.0 |
| dinov2_large | genre | 20 | teacher | 0.5084 | 0.3662 | 0.5032 | 20.0 |
| clip_b32 | emotion | 9 | cstar | 0.0470 | 0.0245 | 0.1944 | 9.0 |
| clip_b32 | emotion | 9 | matched | 0.0382 | 0.0216 | 0.1944 | 9.0 |
| clip_b32 | emotion | 9 | frozen | 0.0359 | 0.0190 | 0.1862 | 9.0 |
| clip_b32 | emotion | 9 | teacher | 0.0501 | 0.0309 | 0.2087 | 9.0 |
| clip_b32 | emotion | 20 | cstar | 0.0535 | 0.0193 | 0.1213 | 20.0 |
| clip_b32 | emotion | 20 | matched | 0.0419 | 0.0139 | 0.1231 | 20.0 |
| clip_b32 | emotion | 20 | frozen | 0.0374 | 0.0115 | 0.1085 | 20.0 |
| clip_b32 | emotion | 20 | teacher | 0.0584 | 0.0265 | 0.1331 | 20.0 |
| clip_b32 | style | 20 | cstar | 0.3940 | 0.1832 | 0.3282 | 20.0 |
| clip_b32 | style | 20 | matched | 0.2184 | 0.0733 | 0.1820 | 20.0 |
| clip_b32 | style | 20 | frozen | 0.3136 | 0.1181 | 0.2472 | 20.0 |
| clip_b32 | style | 20 | teacher | 0.4181 | 0.1945 | 0.3378 | 20.0 |
| clip_b32 | genre | 10 | cstar | 0.5021 | 0.4391 | 0.6149 | 10.0 |
| clip_b32 | genre | 10 | matched | 0.4429 | 0.3489 | 0.5058 | 10.0 |
| clip_b32 | genre | 10 | frozen | 0.4080 | 0.2882 | 0.4599 | 10.0 |
| clip_b32 | genre | 10 | teacher | 0.5262 | 0.4600 | 0.6206 | 10.0 |
| clip_b32 | genre | 20 | cstar | 0.4937 | 0.3371 | 0.4668 | 20.0 |
| clip_b32 | genre | 20 | matched | 0.4299 | 0.2662 | 0.3678 | 20.0 |
| clip_b32 | genre | 20 | frozen | 0.4143 | 0.2314 | 0.3610 | 20.0 |
| clip_b32 | genre | 20 | teacher | 0.5084 | 0.3662 | 0.5032 | 20.0 |
| clip_b16 | emotion | 9 | cstar | 0.0482 | 0.0271 | 0.1989 | 9.0 |
| clip_b16 | emotion | 9 | matched | 0.0375 | 0.0192 | 0.1918 | 9.0 |
| clip_b16 | emotion | 9 | frozen | 0.0362 | 0.0194 | 0.1900 | 9.0 |
| clip_b16 | emotion | 9 | teacher | 0.0501 | 0.0309 | 0.2087 | 9.0 |
| clip_b16 | emotion | 20 | cstar | 0.0570 | 0.0204 | 0.1233 | 20.0 |
| clip_b16 | emotion | 20 | matched | 0.0426 | 0.0139 | 0.1227 | 20.0 |
| clip_b16 | emotion | 20 | frozen | 0.0392 | 0.0123 | 0.1158 | 20.0 |
| clip_b16 | emotion | 20 | teacher | 0.0584 | 0.0265 | 0.1331 | 20.0 |
| clip_b16 | style | 20 | cstar | 0.4163 | 0.1965 | 0.3521 | 20.0 |
| clip_b16 | style | 20 | matched | 0.2235 | 0.0744 | 0.1853 | 20.0 |
| clip_b16 | style | 20 | frozen | 0.3234 | 0.1204 | 0.2526 | 20.0 |
| clip_b16 | style | 20 | teacher | 0.4181 | 0.1945 | 0.3378 | 20.0 |
| clip_b16 | genre | 10 | cstar | 0.5195 | 0.4555 | 0.6181 | 10.0 |
| clip_b16 | genre | 10 | matched | 0.4510 | 0.3619 | 0.5191 | 10.0 |
| clip_b16 | genre | 10 | frozen | 0.4188 | 0.3112 | 0.4715 | 10.0 |
| clip_b16 | genre | 10 | teacher | 0.5262 | 0.4600 | 0.6206 | 10.0 |
| clip_b16 | genre | 20 | cstar | 0.5064 | 0.3540 | 0.4775 | 20.0 |
| clip_b16 | genre | 20 | matched | 0.4393 | 0.2707 | 0.3684 | 20.0 |
| clip_b16 | genre | 20 | frozen | 0.4423 | 0.2522 | 0.3939 | 20.0 |
| clip_b16 | genre | 20 | teacher | 0.5084 | 0.3662 | 0.5032 | 20.0 |

### artelingo_held (X-c*, K = 20)

| encoder | aspect | NMI | ARI | ACC | clusters |
|---|---|---|---|---|---|
| dinov2_large | emotion | 0.0883 | 0.0354 | 0.1496 | 20.0 |
| dinov2_large | style | 0.3492 | 0.1573 | 0.2920 | 20.0 |
| dinov2_large | genre | 0.4884 | 0.3236 | 0.4530 | 20.0 |
| clip_b32 | emotion | 0.0942 | 0.0361 | 0.1518 | 20.0 |
| clip_b32 | style | 0.3788 | 0.1755 | 0.3137 | 20.0 |
| clip_b32 | genre | 0.4910 | 0.3233 | 0.4591 | 20.0 |
| clip_b16 | emotion | 0.0992 | 0.0390 | 0.1579 | 20.0 |
| clip_b16 | style | 0.3966 | 0.1860 | 0.3209 | 20.0 |
| clip_b16 | genre | 0.4970 | 0.3410 | 0.4700 | 20.0 |

### affection_test (X photo model, K = 20)

| encoder | NMI | ARI | ACC | clusters |
|---|---|---|---|---|
| dinov2_large | 0.1579 | 0.0547 | 0.1806 | 20.0 |
| clip_b32 | 0.1742 | 0.0622 | 0.1934 | 20.0 |
| clip_b16 | 0.1839 | 0.0653 | 0.2020 | 20.0 |


