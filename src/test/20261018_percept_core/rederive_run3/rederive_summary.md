# Independent CoSiR r10 C13 re-derivation, run 3

Population standard deviations use ddof=0. Arm image-side statistics summarize each training seed's mean over the three k-means seeds. All displayed scores use six decimals; JSON retains full float precision.

## Stage 2: per arm and label

| Arm | Label | Seeds | Image-side mean | Image-side sd | Argmax mean | Argmax sd |
|---|---|---:|---:|---:|---:|---:|
| P1 | emotion | 3 | 0.052965 | 0.000714 | 0.057863 | 0.003962 |
| P1 | style | 3 | 0.167319 | 0.002911 | 0.153483 | 0.001543 |
| P1 | genre | 3 | 0.355833 | 0.000785 | 0.350627 | 0.005219 |
| P2 | emotion | 3 | 0.065547 | 0.001960 | 0.065808 | 0.002073 |
| P2 | style | 3 | 0.228560 | 0.004909 | 0.219576 | 0.004604 |
| P2 | genre | 3 | 0.400066 | 0.004603 | 0.383464 | 0.008929 |
| P3 | emotion | 3 | 0.077174 | 0.001244 | 0.081914 | 0.000970 |
| P3 | style | 3 | 0.188544 | 0.002844 | 0.191428 | 0.000973 |
| P3 | genre | 3 | 0.435880 | 0.003738 | 0.400170 | 0.001384 |
| P4 | emotion | 3 | 0.053502 | 0.000049 | 0.055000 | 0.001987 |
| P4 | style | 3 | 0.162478 | 0.003175 | 0.151542 | 0.005689 |
| P4 | genre | 3 | 0.361472 | 0.001190 | 0.359268 | 0.010396 |
| P5 | emotion | 3 | 0.073752 | 0.001993 | 0.078095 | 0.001912 |
| P5 | style | 3 | 0.180590 | 0.005117 | 0.187539 | 0.004551 |
| P5 | genre | 3 | 0.435054 | 0.006903 | 0.400774 | 0.002680 |

## Stage 2: per file

Each label cell is the mean ± population sd over k-means seeds 0, 1, 2.

| File | K | Emotion image-side | Style image-side | Genre image-side |
|---|---:|---:|---:|---:|
| P1__s0.npz | 40 | 0.052063 ± 0.001386 | 0.170014 ± 0.001031 | 0.356729 ± 0.002268 |
| P1__s1.npz | 40 | 0.053023 ± 0.001321 | 0.163277 ± 0.001389 | 0.354818 ± 0.003264 |
| P1__s2.npz | 40 | 0.053809 ± 0.000814 | 0.168666 ± 0.000732 | 0.355952 ± 0.000754 |
| P2__s0.npz | 40 | 0.067137 ± 0.000880 | 0.230533 ± 0.002308 | 0.402837 ± 0.005272 |
| P2__s1.npz | 40 | 0.062786 ± 0.000536 | 0.221810 ± 0.003282 | 0.393579 ± 0.002293 |
| P2__s2.npz | 40 | 0.066718 ± 0.002209 | 0.233338 ± 0.005623 | 0.403782 ± 0.001335 |
| P3__s0.npz | 40 | 0.078861 ± 0.000987 | 0.184522 ± 0.000800 | 0.433488 ± 0.004146 |
| P3__s1.npz | 40 | 0.076761 ± 0.003151 | 0.190576 ± 0.003697 | 0.441158 ± 0.003782 |
| P3__s2.npz | 40 | 0.075899 ± 0.001436 | 0.190535 ± 0.002863 | 0.432993 ± 0.004522 |
| P4__s0.npz | 40 | 0.053530 ± 0.001099 | 0.166965 ± 0.003091 | 0.360981 ± 0.003259 |
| P4__s1.npz | 40 | 0.053541 ± 0.000192 | 0.160103 ± 0.004830 | 0.363112 ± 0.004662 |
| P4__s2.npz | 40 | 0.053433 ± 0.000270 | 0.160366 ± 0.002267 | 0.360322 ± 0.004557 |
| P5__s0.npz | 40 | 0.071742 ± 0.000706 | 0.187758 ± 0.002270 | 0.443489 ± 0.000557 |
| P5__s1.npz | 40 | 0.073045 ± 0.000758 | 0.176149 ± 0.001710 | 0.426581 ± 0.002655 |
| P5__s2.npz | 40 | 0.076469 ± 0.001061 | 0.177863 ± 0.001734 | 0.435091 ± 0.001144 |

## Comparator: Established arms

Select the lowest-numbered arm within 0.003 (inclusive) of the highest mean.

| Aspect | Comparator | Comparator mean | Comparator sd | Highest arm | Highest mean |
|---|---|---:|---:|---|---:|
| emotion | P3 | 0.077174 | 0.001244 | P3 | 0.077174 |
| style | P2 | 0.228560 | 0.004909 | P2 | 0.228560 |
| genre | P3 | 0.435880 | 0.003738 | P3 | 0.435880 |

## Comparator: All arms with all three seeds present

Select the lowest-numbered arm within 0.003 (inclusive) of the highest mean.

| Aspect | Comparator | Comparator mean | Comparator sd | Highest arm | Highest mean |
|---|---|---:|---:|---|---:|
| emotion | P3 | 0.077174 | 0.001244 | P3 | 0.077174 |
| style | P2 | 0.228560 | 0.004909 | P2 | 0.228560 |
| genre | P3 | 0.435880 | 0.003738 | P3 | 0.435880 |

## Stage 1: checks and coverage

Fit-pair denominator: 67327 caption rows. Coverage uses topic[fit_rows]; the pre-cut share uses stored fit_counts. These come from separate model passes and are not compared as equal.

| File | Checks | Kept | Fit count sum | Top 40 match | With fit pair | With ≥1% | Share before cut | 40th | 41st | Gap |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| clip_roberta__s0.npz | PASS | 40 | 67327 | True | 40 | 40 | 0.568405 | 680 | 678 | 2 |
| clip_roberta__s1.npz | PASS | 40 | 67327 | True | 40 | 40 | 0.567425 | 669 | 668 | 1 |
| clip_roberta__s2.npz | PASS | 40 | 67327 | True | 40 | 40 | 0.581104 | 684 | 683 | 1 |
| qwen_qwen__s0.npz | PASS | 40 | 67327 | True | 40 | 40 | 0.611389 | 732 | 719 | 13 |
| qwen_qwen__s1.npz | PASS | 40 | 67327 | True | 40 | 40 | 0.625796 | 741 | 728 | 13 |
| qwen_qwen__s2.npz | PASS | 40 | 67327 | True | 40 | 40 | 0.627326 | 677 | 665 | 12 |
| qwen_roberta__s0.npz | PASS | 40 | 67327 | True | 40 | 40 | 0.593863 | 726 | 724 | 2 |
| qwen_roberta__s1.npz | PASS | 40 | 67327 | True | 40 | 40 | 0.593180 | 685 | 680 | 5 |
| qwen_roberta__s2.npz | PASS | 40 | 67327 | True | 40 | 40 | 0.586362 | 722 | 708 | 14 |

## Stage 1: argmax per version

| Version | Label | Seeds | Mean AMI | Population sd |
|---|---|---:|---:|---:|
| clip_roberta | emotion | 3 | 0.142800 | 0.004205 |
| clip_roberta | style | 3 | 0.160058 | 0.000498 |
| clip_roberta | genre | 3 | 0.305755 | 0.005304 |
| qwen_qwen | emotion | 3 | 0.132306 | 0.002679 |
| qwen_qwen | style | 3 | 0.172022 | 0.001364 |
| qwen_qwen | genre | 3 | 0.375419 | 0.003198 |
| qwen_roberta | emotion | 3 | 0.121870 | 0.004402 |
| qwen_roberta | style | 3 | 0.165880 | 0.005393 |
| qwen_roberta | genre | 3 | 0.366786 | 0.005291 |

## Stage 1: argmax per file

| File | Emotion | Style | Genre |
|---|---:|---:|---:|
| clip_roberta__s0.npz | 0.145772 | 0.159976 | 0.310994 |
| clip_roberta__s1.npz | 0.136853 | 0.159493 | 0.298486 |
| clip_roberta__s2.npz | 0.145776 | 0.160705 | 0.307786 |
| qwen_qwen__s0.npz | 0.130577 | 0.170403 | 0.372233 |
| qwen_qwen__s1.npz | 0.136091 | 0.173740 | 0.374230 |
| qwen_qwen__s2.npz | 0.130252 | 0.171924 | 0.379792 |
| qwen_roberta__s0.npz | 0.120682 | 0.172574 | 0.373527 |
| qwen_roberta__s1.npz | 0.127756 | 0.159368 | 0.360604 |
| qwen_roberta__s2.npz | 0.117171 | 0.165699 | 0.366226 |

## Label counts

| Label | Scored | Excluded (<0) | Class counts |
|---|---:|---:|---|
| emotion | 5000 | 0 | 0: 780, 1: 55, 2: 974, 3: 1758, 4: 149, 5: 215, 6: 410, 7: 373, 8: 286 |
| style | 5000 | 0 | 0: 170, 1: 6, 2: 7, 3: 278, 4: 276, 5: 96, 6: 27, 7: 105, 8: 76, 9: 438, 10: 45, 11: 80, 12: 756, 13: 88, 14: 76, 15: 141, 16: 20, 17: 152, 18: 26, 19: 98, 20: 448, 21: 687, 22: 135, 23: 436, 24: 253, 25: 14, 26: 66 |
| genre | 4053 | 947 | 0: 302, 1: 296, 2: 682, 3: 125, 4: 827, 5: 97, 6: 883, 7: 406, 8: 253, 9: 182 |

Emotion majority ties among evaluation paintings with nonempty votes: 1295; ties select the smallest class id.

## Every check

Only exact expected filenames were opened; no pending NPZ or JSON sidecar was read. File counts below refer to expected filenames present, not unrelated directory entries.

- PASS: sample sample_ids shape; observed: `{"shape": [99999], "dtype": "int64"}`
- PASS: sample pid shape; observed: `{"shape": [99999], "dtype": "int64"}`
- PASS: sample painting_ids shape; observed: `{"shape": [99999], "dtype": "<U125"}`
- PASS: sample emo shape; observed: `{"shape": [99999], "dtype": "int64"}`
- PASS: sample sty shape; observed: `{"shape": [99999], "dtype": "int64"}`
- PASS: sample gen shape; observed: `{"shape": [99999], "dtype": "int64"}`
- PASS: sample first shape; observed: `{"shape": [19889], "dtype": "int64"}`
- PASS: sample all_painting_ids shape; observed: `{"shape": [19889], "dtype": "<U125"}`
- PASS: sample subset_pid shape; observed: `{"shape": [5000], "dtype": "int64"}`
- PASS: sample sample_ids int64; observed: `"int64"`
- PASS: sample pid integer; observed: `"int64"`
- PASS: sample emo integer; observed: `"int64"`
- PASS: sample sty integer; observed: `"int64"`
- PASS: sample gen integer; observed: `"int64"`
- PASS: sample first integer; observed: `"int64"`
- PASS: sample subset_pid integer; observed: `"int64"`
- PASS: sample pid range; observed: `[0, 19888]`
- PASS: sample first range; observed: `[0, 87486]`
- PASS: sample subset_pid range; observed: `[3, 19888]`
- PASS: sample subset_pid unique; observed: `5000`
- PASS: sample emotion range; observed: `[0, 8]`
- PASS: sample first is actual first caption
- PASS: sample painting names agree with pid
- PASS: emotion label length; observed: `{"valid": 5000, "excluded": 0, "class_counts": {"0": 780, "1": 55, "2": 974, "3": 1758, "4": 149, "5": 215, "6": 410, "7": 373, "8": 286}}`
- PASS: style label length; observed: `{"valid": 5000, "excluded": 0, "class_counts": {"0": 170, "1": 6, "2": 7, "3": 278, "4": 276, "5": 96, "6": 27, "7": 105, "8": 76, "9": 438, "10": 45, "11": 80, "12": 756, "13": 88, "14": 76, "15": 141, "16": 20, "17": 152, "18": 26, "19": 98, "20": 448, "21": 687, "22": 135, "23": 436, "24": 253, "25": 14, "26": 66}}`
- PASS: genre label length; observed: `{"valid": 4053, "excluded": 947, "class_counts": {"0": 302, "1": 296, "2": 682, "3": 125, "4": 827, "5": 97, "6": 883, "7": 406, "8": 253, "9": 182}}`
- PASS: fit_rows one-dimensional int64; observed: `{"shape": [67327], "dtype": "int64"}`
- PASS: fit_rows nonempty; observed: `67327`
- PASS: fit_rows strictly ascending
- PASS: fit_rows caption range; observed: `[0, 99994]`
- PASS: fit_rows represent 13400 paintings; observed: `13400`
- PASS: fit and evaluation paintings disjoint; observed: `0`
- PASS: Stage 2 file exists P1__s0.npz; observed: `"/project/CoSiR/src/test/20261018_percept_core/outputs/stage2/P1__s0.npz"`
- PASS: P1__s0.npz keys; observed: `["eval_prob"]`
- PASS: P1__s0.npz shape; observed: `[5000, 40]`
- PASS: P1__s0.npz float32; observed: `"float32"`
- PASS: P1__s0.npz finite
- PASS: P1__s0.npz values in [0,1]; observed: `[0.020108897238969803, 0.5952078104019165]`
- PASS: Stage 2 file exists P1__s1.npz; observed: `"/project/CoSiR/src/test/20261018_percept_core/outputs/stage2/P1__s1.npz"`
- PASS: P1__s1.npz keys; observed: `["eval_prob"]`
- PASS: P1__s1.npz shape; observed: `[5000, 40]`
- PASS: P1__s1.npz float32; observed: `"float32"`
- PASS: P1__s1.npz finite
- PASS: P1__s1.npz values in [0,1]; observed: `[0.018868980929255486, 0.6051468253135681]`
- PASS: Stage 2 file exists P1__s2.npz; observed: `"/project/CoSiR/src/test/20261018_percept_core/outputs/stage2/P1__s2.npz"`
- PASS: P1__s2.npz keys; observed: `["eval_prob"]`
- PASS: P1__s2.npz shape; observed: `[5000, 40]`
- PASS: P1__s2.npz float32; observed: `"float32"`
- PASS: P1__s2.npz finite
- PASS: P1__s2.npz values in [0,1]; observed: `[0.021239766851067543, 0.5675209760665894]`
- PASS: Stage 2 file exists P2__s0.npz; observed: `"/project/CoSiR/src/test/20261018_percept_core/outputs/stage2/P2__s0.npz"`
- PASS: P2__s0.npz keys; observed: `["eval_prob"]`
- PASS: P2__s0.npz shape; observed: `[5000, 40]`
- PASS: P2__s0.npz float32; observed: `"float32"`
- PASS: P2__s0.npz finite
- PASS: P2__s0.npz values in [0,1]; observed: `[1.7836075585364597e-06, 0.9647647142410278]`
- PASS: Stage 2 file exists P2__s1.npz; observed: `"/project/CoSiR/src/test/20261018_percept_core/outputs/stage2/P2__s1.npz"`
- PASS: P2__s1.npz keys; observed: `["eval_prob"]`
- PASS: P2__s1.npz shape; observed: `[5000, 40]`
- PASS: P2__s1.npz float32; observed: `"float32"`
- PASS: P2__s1.npz finite
- PASS: P2__s1.npz values in [0,1]; observed: `[1.9532419628376374e-06, 0.9662895798683167]`
- PASS: Stage 2 file exists P2__s2.npz; observed: `"/project/CoSiR/src/test/20261018_percept_core/outputs/stage2/P2__s2.npz"`
- PASS: P2__s2.npz keys; observed: `["eval_prob"]`
- PASS: P2__s2.npz shape; observed: `[5000, 40]`
- PASS: P2__s2.npz float32; observed: `"float32"`
- PASS: P2__s2.npz finite
- PASS: P2__s2.npz values in [0,1]; observed: `[1.5573855307593476e-06, 0.9742817878723145]`
- PASS: Stage 2 file exists P3__s0.npz; observed: `"/project/CoSiR/src/test/20261018_percept_core/outputs/stage2/P3__s0.npz"`
- PASS: P3__s0.npz keys; observed: `["eval_prob"]`
- PASS: P3__s0.npz shape; observed: `[5000, 40]`
- PASS: P3__s0.npz float32; observed: `"float32"`
- PASS: P3__s0.npz finite
- PASS: P3__s0.npz values in [0,1]; observed: `[2.5722187046994804e-07, 0.9950915575027466]`
- PASS: Stage 2 file exists P3__s1.npz; observed: `"/project/CoSiR/src/test/20261018_percept_core/outputs/stage2/P3__s1.npz"`
- PASS: P3__s1.npz keys; observed: `["eval_prob"]`
- PASS: P3__s1.npz shape; observed: `[5000, 40]`
- PASS: P3__s1.npz float32; observed: `"float32"`
- PASS: P3__s1.npz finite
- PASS: P3__s1.npz values in [0,1]; observed: `[3.4185924846497073e-07, 0.9927934408187866]`
- PASS: Stage 2 file exists P3__s2.npz; observed: `"/project/CoSiR/src/test/20261018_percept_core/outputs/stage2/P3__s2.npz"`
- PASS: P3__s2.npz keys; observed: `["eval_prob"]`
- PASS: P3__s2.npz shape; observed: `[5000, 40]`
- PASS: P3__s2.npz float32; observed: `"float32"`
- PASS: P3__s2.npz finite
- PASS: P3__s2.npz values in [0,1]; observed: `[3.222364455268689e-07, 0.994835615158081]`
- PASS: Stage 2 file exists P4__s0.npz; observed: `"/project/CoSiR/src/test/20261018_percept_core/outputs/stage2/P4__s0.npz"`
- PASS: P4__s0.npz keys; observed: `["eval_prob"]`
- PASS: P4__s0.npz shape; observed: `[5000, 40]`
- PASS: P4__s0.npz float32; observed: `"float32"`
- PASS: P4__s0.npz finite
- PASS: P4__s0.npz values in [0,1]; observed: `[0.014708925038576126, 0.34326639771461487]`
- PASS: Stage 2 file exists P4__s1.npz; observed: `"/project/CoSiR/src/test/20261018_percept_core/outputs/stage2/P4__s1.npz"`
- PASS: P4__s1.npz keys; observed: `["eval_prob"]`
- PASS: P4__s1.npz shape; observed: `[5000, 40]`
- PASS: P4__s1.npz float32; observed: `"float32"`
- PASS: P4__s1.npz finite
- PASS: P4__s1.npz values in [0,1]; observed: `[0.01136218011379242, 0.3276519775390625]`
- PASS: Stage 2 file exists P4__s2.npz; observed: `"/project/CoSiR/src/test/20261018_percept_core/outputs/stage2/P4__s2.npz"`
- PASS: P4__s2.npz keys; observed: `["eval_prob"]`
- PASS: P4__s2.npz shape; observed: `[5000, 40]`
- PASS: P4__s2.npz float32; observed: `"float32"`
- PASS: P4__s2.npz finite
- PASS: P4__s2.npz values in [0,1]; observed: `[0.010131004266440868, 0.3739221692085266]`
- PASS: Stage 2 file exists P5__s0.npz; observed: `"/project/CoSiR/src/test/20261018_percept_core/outputs/stage2/P5__s0.npz"`
- PASS: P5__s0.npz keys; observed: `["eval_prob"]`
- PASS: P5__s0.npz shape; observed: `[5000, 40]`
- PASS: P5__s0.npz float32; observed: `"float32"`
- PASS: P5__s0.npz finite
- PASS: P5__s0.npz values in [0,1]; observed: `[4.871376972914732e-07, 0.9885849356651306]`
- PASS: Stage 2 file exists P5__s1.npz; observed: `"/project/CoSiR/src/test/20261018_percept_core/outputs/stage2/P5__s1.npz"`
- PASS: P5__s1.npz keys; observed: `["eval_prob"]`
- PASS: P5__s1.npz shape; observed: `[5000, 40]`
- PASS: P5__s1.npz float32; observed: `"float32"`
- PASS: P5__s1.npz finite
- PASS: P5__s1.npz values in [0,1]; observed: `[2.595257342363766e-07, 0.9927510023117065]`
- PASS: Stage 2 file exists P5__s2.npz; observed: `"/project/CoSiR/src/test/20261018_percept_core/outputs/stage2/P5__s2.npz"`
- PASS: P5__s2.npz keys; observed: `["eval_prob"]`
- PASS: P5__s2.npz shape; observed: `[5000, 40]`
- PASS: P5__s2.npz float32; observed: `"float32"`
- PASS: P5__s2.npz finite
- PASS: P5__s2.npz values in [0,1]; observed: `[2.3655137226796796e-07, 0.9930518865585327]`
- PASS: Stage 2 present file count; observed: `15`
- PASS: Stage 1 file exists clip_roberta__s0.npz; observed: `"/project/CoSiR/src/test/20261018_percept_core/outputs/stage1/clip_roberta__s0.npz"`
- PASS: clip_roberta__s0.npz center_norms shape; observed: `{"shape": [100], "dtype": "float32"}`
- PASS: clip_roberta__s0.npz fit_counts shape; observed: `{"shape": [100], "dtype": "int64"}`
- PASS: clip_roberta__s0.npz kept shape; observed: `{"shape": [100], "dtype": "bool"}`
- PASS: clip_roberta__s0.npz topic shape; observed: `{"shape": [99999], "dtype": "int16"}`
- PASS: clip_roberta__s0.npz q_paint_eval shape; observed: `{"shape": [5000, 40], "dtype": "float32"}`
- PASS: clip_roberta__s0.npz center_norms floating dtype; observed: `"float32"`
- PASS: clip_roberta__s0.npz q_paint_eval floating dtype; observed: `"float32"`
- PASS: clip_roberta__s0.npz fit_counts int64; observed: `"int64"`
- PASS: clip_roberta__s0.npz kept bool; observed: `"bool"`
- PASS: clip_roberta__s0.npz topic integer; observed: `"int16"`
- PASS: clip_roberta__s0.npz center_norms finite/nonnegative
- PASS: clip_roberta__s0.npz fit_counts nonnegative
- PASS: clip_roberta__s0.npz kept sum; observed: `40`
- PASS: clip_roberta__s0.npz fit_counts sum; observed: `67327`
- PASS: clip_roberta__s0.npz top40 matches
- PASS: clip_roberta__s0.npz topic range 0..39; observed: `[0, 39]`
- PASS: clip_roberta__s0.npz q_paint_eval finite
- PASS: Stage 1 file exists clip_roberta__s1.npz; observed: `"/project/CoSiR/src/test/20261018_percept_core/outputs/stage1/clip_roberta__s1.npz"`
- PASS: clip_roberta__s1.npz center_norms shape; observed: `{"shape": [100], "dtype": "float32"}`
- PASS: clip_roberta__s1.npz fit_counts shape; observed: `{"shape": [100], "dtype": "int64"}`
- PASS: clip_roberta__s1.npz kept shape; observed: `{"shape": [100], "dtype": "bool"}`
- PASS: clip_roberta__s1.npz topic shape; observed: `{"shape": [99999], "dtype": "int16"}`
- PASS: clip_roberta__s1.npz q_paint_eval shape; observed: `{"shape": [5000, 40], "dtype": "float32"}`
- PASS: clip_roberta__s1.npz center_norms floating dtype; observed: `"float32"`
- PASS: clip_roberta__s1.npz q_paint_eval floating dtype; observed: `"float32"`
- PASS: clip_roberta__s1.npz fit_counts int64; observed: `"int64"`
- PASS: clip_roberta__s1.npz kept bool; observed: `"bool"`
- PASS: clip_roberta__s1.npz topic integer; observed: `"int16"`
- PASS: clip_roberta__s1.npz center_norms finite/nonnegative
- PASS: clip_roberta__s1.npz fit_counts nonnegative
- PASS: clip_roberta__s1.npz kept sum; observed: `40`
- PASS: clip_roberta__s1.npz fit_counts sum; observed: `67327`
- PASS: clip_roberta__s1.npz top40 matches
- PASS: clip_roberta__s1.npz topic range 0..39; observed: `[0, 39]`
- PASS: clip_roberta__s1.npz q_paint_eval finite
- PASS: Stage 1 file exists clip_roberta__s2.npz; observed: `"/project/CoSiR/src/test/20261018_percept_core/outputs/stage1/clip_roberta__s2.npz"`
- PASS: clip_roberta__s2.npz center_norms shape; observed: `{"shape": [100], "dtype": "float32"}`
- PASS: clip_roberta__s2.npz fit_counts shape; observed: `{"shape": [100], "dtype": "int64"}`
- PASS: clip_roberta__s2.npz kept shape; observed: `{"shape": [100], "dtype": "bool"}`
- PASS: clip_roberta__s2.npz topic shape; observed: `{"shape": [99999], "dtype": "int16"}`
- PASS: clip_roberta__s2.npz q_paint_eval shape; observed: `{"shape": [5000, 40], "dtype": "float32"}`
- PASS: clip_roberta__s2.npz center_norms floating dtype; observed: `"float32"`
- PASS: clip_roberta__s2.npz q_paint_eval floating dtype; observed: `"float32"`
- PASS: clip_roberta__s2.npz fit_counts int64; observed: `"int64"`
- PASS: clip_roberta__s2.npz kept bool; observed: `"bool"`
- PASS: clip_roberta__s2.npz topic integer; observed: `"int16"`
- PASS: clip_roberta__s2.npz center_norms finite/nonnegative
- PASS: clip_roberta__s2.npz fit_counts nonnegative
- PASS: clip_roberta__s2.npz kept sum; observed: `40`
- PASS: clip_roberta__s2.npz fit_counts sum; observed: `67327`
- PASS: clip_roberta__s2.npz top40 matches
- PASS: clip_roberta__s2.npz topic range 0..39; observed: `[0, 39]`
- PASS: clip_roberta__s2.npz q_paint_eval finite
- PASS: Stage 1 file exists qwen_qwen__s0.npz; observed: `"/project/CoSiR/src/test/20261018_percept_core/outputs/stage1/qwen_qwen__s0.npz"`
- PASS: qwen_qwen__s0.npz center_norms shape; observed: `{"shape": [100], "dtype": "float32"}`
- PASS: qwen_qwen__s0.npz fit_counts shape; observed: `{"shape": [100], "dtype": "int64"}`
- PASS: qwen_qwen__s0.npz kept shape; observed: `{"shape": [100], "dtype": "bool"}`
- PASS: qwen_qwen__s0.npz topic shape; observed: `{"shape": [99999], "dtype": "int16"}`
- PASS: qwen_qwen__s0.npz q_paint_eval shape; observed: `{"shape": [5000, 40], "dtype": "float32"}`
- PASS: qwen_qwen__s0.npz center_norms floating dtype; observed: `"float32"`
- PASS: qwen_qwen__s0.npz q_paint_eval floating dtype; observed: `"float32"`
- PASS: qwen_qwen__s0.npz fit_counts int64; observed: `"int64"`
- PASS: qwen_qwen__s0.npz kept bool; observed: `"bool"`
- PASS: qwen_qwen__s0.npz topic integer; observed: `"int16"`
- PASS: qwen_qwen__s0.npz center_norms finite/nonnegative
- PASS: qwen_qwen__s0.npz fit_counts nonnegative
- PASS: qwen_qwen__s0.npz kept sum; observed: `40`
- PASS: qwen_qwen__s0.npz fit_counts sum; observed: `67327`
- PASS: qwen_qwen__s0.npz top40 matches
- PASS: qwen_qwen__s0.npz topic range 0..39; observed: `[0, 39]`
- PASS: qwen_qwen__s0.npz q_paint_eval finite
- PASS: Stage 1 file exists qwen_qwen__s1.npz; observed: `"/project/CoSiR/src/test/20261018_percept_core/outputs/stage1/qwen_qwen__s1.npz"`
- PASS: qwen_qwen__s1.npz center_norms shape; observed: `{"shape": [100], "dtype": "float32"}`
- PASS: qwen_qwen__s1.npz fit_counts shape; observed: `{"shape": [100], "dtype": "int64"}`
- PASS: qwen_qwen__s1.npz kept shape; observed: `{"shape": [100], "dtype": "bool"}`
- PASS: qwen_qwen__s1.npz topic shape; observed: `{"shape": [99999], "dtype": "int16"}`
- PASS: qwen_qwen__s1.npz q_paint_eval shape; observed: `{"shape": [5000, 40], "dtype": "float32"}`
- PASS: qwen_qwen__s1.npz center_norms floating dtype; observed: `"float32"`
- PASS: qwen_qwen__s1.npz q_paint_eval floating dtype; observed: `"float32"`
- PASS: qwen_qwen__s1.npz fit_counts int64; observed: `"int64"`
- PASS: qwen_qwen__s1.npz kept bool; observed: `"bool"`
- PASS: qwen_qwen__s1.npz topic integer; observed: `"int16"`
- PASS: qwen_qwen__s1.npz center_norms finite/nonnegative
- PASS: qwen_qwen__s1.npz fit_counts nonnegative
- PASS: qwen_qwen__s1.npz kept sum; observed: `40`
- PASS: qwen_qwen__s1.npz fit_counts sum; observed: `67327`
- PASS: qwen_qwen__s1.npz top40 matches
- PASS: qwen_qwen__s1.npz topic range 0..39; observed: `[0, 39]`
- PASS: qwen_qwen__s1.npz q_paint_eval finite
- PASS: Stage 1 file exists qwen_qwen__s2.npz; observed: `"/project/CoSiR/src/test/20261018_percept_core/outputs/stage1/qwen_qwen__s2.npz"`
- PASS: qwen_qwen__s2.npz center_norms shape; observed: `{"shape": [100], "dtype": "float32"}`
- PASS: qwen_qwen__s2.npz fit_counts shape; observed: `{"shape": [100], "dtype": "int64"}`
- PASS: qwen_qwen__s2.npz kept shape; observed: `{"shape": [100], "dtype": "bool"}`
- PASS: qwen_qwen__s2.npz topic shape; observed: `{"shape": [99999], "dtype": "int16"}`
- PASS: qwen_qwen__s2.npz q_paint_eval shape; observed: `{"shape": [5000, 40], "dtype": "float32"}`
- PASS: qwen_qwen__s2.npz center_norms floating dtype; observed: `"float32"`
- PASS: qwen_qwen__s2.npz q_paint_eval floating dtype; observed: `"float32"`
- PASS: qwen_qwen__s2.npz fit_counts int64; observed: `"int64"`
- PASS: qwen_qwen__s2.npz kept bool; observed: `"bool"`
- PASS: qwen_qwen__s2.npz topic integer; observed: `"int16"`
- PASS: qwen_qwen__s2.npz center_norms finite/nonnegative
- PASS: qwen_qwen__s2.npz fit_counts nonnegative
- PASS: qwen_qwen__s2.npz kept sum; observed: `40`
- PASS: qwen_qwen__s2.npz fit_counts sum; observed: `67327`
- PASS: qwen_qwen__s2.npz top40 matches
- PASS: qwen_qwen__s2.npz topic range 0..39; observed: `[0, 39]`
- PASS: qwen_qwen__s2.npz q_paint_eval finite
- PASS: Stage 1 file exists qwen_roberta__s0.npz; observed: `"/project/CoSiR/src/test/20261018_percept_core/outputs/stage1/qwen_roberta__s0.npz"`
- PASS: qwen_roberta__s0.npz center_norms shape; observed: `{"shape": [100], "dtype": "float32"}`
- PASS: qwen_roberta__s0.npz fit_counts shape; observed: `{"shape": [100], "dtype": "int64"}`
- PASS: qwen_roberta__s0.npz kept shape; observed: `{"shape": [100], "dtype": "bool"}`
- PASS: qwen_roberta__s0.npz topic shape; observed: `{"shape": [99999], "dtype": "int16"}`
- PASS: qwen_roberta__s0.npz q_paint_eval shape; observed: `{"shape": [5000, 40], "dtype": "float32"}`
- PASS: qwen_roberta__s0.npz center_norms floating dtype; observed: `"float32"`
- PASS: qwen_roberta__s0.npz q_paint_eval floating dtype; observed: `"float32"`
- PASS: qwen_roberta__s0.npz fit_counts int64; observed: `"int64"`
- PASS: qwen_roberta__s0.npz kept bool; observed: `"bool"`
- PASS: qwen_roberta__s0.npz topic integer; observed: `"int16"`
- PASS: qwen_roberta__s0.npz center_norms finite/nonnegative
- PASS: qwen_roberta__s0.npz fit_counts nonnegative
- PASS: qwen_roberta__s0.npz kept sum; observed: `40`
- PASS: qwen_roberta__s0.npz fit_counts sum; observed: `67327`
- PASS: qwen_roberta__s0.npz top40 matches
- PASS: qwen_roberta__s0.npz topic range 0..39; observed: `[0, 39]`
- PASS: qwen_roberta__s0.npz q_paint_eval finite
- PASS: Stage 1 file exists qwen_roberta__s1.npz; observed: `"/project/CoSiR/src/test/20261018_percept_core/outputs/stage1/qwen_roberta__s1.npz"`
- PASS: qwen_roberta__s1.npz center_norms shape; observed: `{"shape": [100], "dtype": "float32"}`
- PASS: qwen_roberta__s1.npz fit_counts shape; observed: `{"shape": [100], "dtype": "int64"}`
- PASS: qwen_roberta__s1.npz kept shape; observed: `{"shape": [100], "dtype": "bool"}`
- PASS: qwen_roberta__s1.npz topic shape; observed: `{"shape": [99999], "dtype": "int16"}`
- PASS: qwen_roberta__s1.npz q_paint_eval shape; observed: `{"shape": [5000, 40], "dtype": "float32"}`
- PASS: qwen_roberta__s1.npz center_norms floating dtype; observed: `"float32"`
- PASS: qwen_roberta__s1.npz q_paint_eval floating dtype; observed: `"float32"`
- PASS: qwen_roberta__s1.npz fit_counts int64; observed: `"int64"`
- PASS: qwen_roberta__s1.npz kept bool; observed: `"bool"`
- PASS: qwen_roberta__s1.npz topic integer; observed: `"int16"`
- PASS: qwen_roberta__s1.npz center_norms finite/nonnegative
- PASS: qwen_roberta__s1.npz fit_counts nonnegative
- PASS: qwen_roberta__s1.npz kept sum; observed: `40`
- PASS: qwen_roberta__s1.npz fit_counts sum; observed: `67327`
- PASS: qwen_roberta__s1.npz top40 matches
- PASS: qwen_roberta__s1.npz topic range 0..39; observed: `[0, 39]`
- PASS: qwen_roberta__s1.npz q_paint_eval finite
- PASS: Stage 1 file exists qwen_roberta__s2.npz; observed: `"/project/CoSiR/src/test/20261018_percept_core/outputs/stage1/qwen_roberta__s2.npz"`
- PASS: qwen_roberta__s2.npz center_norms shape; observed: `{"shape": [100], "dtype": "float32"}`
- PASS: qwen_roberta__s2.npz fit_counts shape; observed: `{"shape": [100], "dtype": "int64"}`
- PASS: qwen_roberta__s2.npz kept shape; observed: `{"shape": [100], "dtype": "bool"}`
- PASS: qwen_roberta__s2.npz topic shape; observed: `{"shape": [99999], "dtype": "int16"}`
- PASS: qwen_roberta__s2.npz q_paint_eval shape; observed: `{"shape": [5000, 40], "dtype": "float32"}`
- PASS: qwen_roberta__s2.npz center_norms floating dtype; observed: `"float32"`
- PASS: qwen_roberta__s2.npz q_paint_eval floating dtype; observed: `"float32"`
- PASS: qwen_roberta__s2.npz fit_counts int64; observed: `"int64"`
- PASS: qwen_roberta__s2.npz kept bool; observed: `"bool"`
- PASS: qwen_roberta__s2.npz topic integer; observed: `"int16"`
- PASS: qwen_roberta__s2.npz center_norms finite/nonnegative
- PASS: qwen_roberta__s2.npz fit_counts nonnegative
- PASS: qwen_roberta__s2.npz kept sum; observed: `40`
- PASS: qwen_roberta__s2.npz fit_counts sum; observed: `67327`
- PASS: qwen_roberta__s2.npz top40 matches
- PASS: qwen_roberta__s2.npz topic range 0..39; observed: `[0, 39]`
- PASS: qwen_roberta__s2.npz q_paint_eval finite
- PASS: Stage 1 present file count; observed: `9`

## Problems and absent files

- No validation problems.
- No absent files.
- Nothing deleted. Nothing proposed for removal.

## Reproducibility

Python 3.11.15; NumPy 2.2.6; scikit-learn 1.6.1.
OMP_NUM_THREADS=8; MKL_NUM_THREADS=8; PYTHONDONTWRITEBYTECODE=1; CUDA_VISIBLE_DEVICES is empty.
MiniBatchKMeans(n_clusters=20, random_state=r, n_init=3, batch_size=4096), r=0,1,2; other parameters retain installed sklearn defaults. One set of partitions is shared across all labels for each matrix. AMI uses installed sklearn defaults.
