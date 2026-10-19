# Independent CoSiR r10 / C13 clustering derivation

All reported AMIs were computed from the authorized NPZ inputs. Population SD uses ddof=0. Image-side arm SD is across model-seed means; per-file SD is across the three k-means seeds.

## Per arm and label

| Arm | Label | Model seeds | Image-side mean | Image-side SD | Argmax mean | Argmax SD |
|---|---|---|---:|---:|---:|---:|
| P1 | emotion | 0,1,2 | 0.028945 | 0.015762 | 0.025610 | 0.019082 |
| P1 | style | 0,1,2 | 0.136823 | 0.027724 | 0.099528 | 0.049759 |
| P1 | genre | 0,1,2 | 0.233993 | 0.083858 | 0.192033 | 0.099130 |
| P2 | emotion | 0,1,2 | 0.035515 | 0.019733 | 0.033487 | 0.019555 |
| P2 | style | 0,1,2 | 0.161855 | 0.078059 | 0.151286 | 0.075507 |
| P2 | genre | 0,1,2 | 0.216970 | 0.142606 | 0.202586 | 0.134460 |
| P3 | emotion | 1,2 | 0.071993 | 0.000015 | 0.075301 | 0.000083 |
| P3 | style | 1,2 | 0.194264 | 0.002754 | 0.196594 | 0.002538 |
| P3 | genre | 1,2 | 0.446199 | 0.005545 | 0.399691 | 0.005161 |
| P4 | emotion | 1,2 | 0.051798 | 0.001246 | 0.053463 | 0.000393 |
| P4 | style | 1,2 | 0.155862 | 0.000136 | 0.145716 | 0.006643 |
| P4 | genre | 1,2 | 0.348011 | 0.000531 | 0.330617 | 0.003110 |
| P5 | emotion | none | — | — | — | — |
| P5 | style | none | — | — | — | — |
| P5 | genre | none | — | — | — | — |

## Per-file image-side scores

| File | K | Emotion mean | Emotion SD | Style mean | Style SD | Genre mean | Genre SD |
|---|---:|---:|---:|---:|---:|---:|---:|
| P1__s0.npz | 99 | 0.051138 | 0.001575 | 0.165742 | 0.000991 | 0.349168 | 0.000034 |
| P1__s1.npz | 2 | 0.016042 | 0.000567 | 0.099436 | 0.000326 | 0.151927 | 0.001061 |
| P1__s2.npz | 3 | 0.019653 | 0.000128 | 0.145293 | 0.000085 | 0.200884 | 0.000638 |
| P2__s0.npz | 99 | 0.063390 | 0.001006 | 0.249997 | 0.004540 | 0.411668 | 0.003560 |
| P2__s1.npz | 2 | 0.022727 | 0.000479 | 0.060225 | 0.000485 | 0.074078 | 0.000377 |
| P2__s2.npz | 3 | 0.020429 | 0.000424 | 0.175342 | 0.001889 | 0.165165 | 0.001886 |
| P3__s1.npz | 99 | 0.071977 | 0.000578 | 0.191509 | 0.002858 | 0.440654 | 0.001790 |
| P3__s2.npz | 99 | 0.072008 | 0.000808 | 0.197018 | 0.003022 | 0.451744 | 0.002748 |
| P4__s1.npz | 99 | 0.050553 | 0.001124 | 0.155998 | 0.002069 | 0.348542 | 0.002814 |
| P4__s2.npz | 99 | 0.053044 | 0.000519 | 0.155726 | 0.001900 | 0.347480 | 0.005313 |

## Comparators

Only arms with all three valid model seeds are eligible. Choose the lowest arm number within 0.003 of the highest image-side mean.

| Aspect | Comparator | Comparator mean | Comparator SD | Highest arm(s) | Highest mean |
|---|---|---:|---:|---|---:|
| emotion | P2 | 0.035515 | 0.019733 | P2 | 0.035515 |
| style | P2 | 0.161855 | 0.078059 | P2 | 0.161855 |
| genre | P1 | 0.233993 | 0.083858 | P1 | 0.233993 |

## Checks

- Expected files: 15; present: 10; absent: 5; successfully scored: 10; failed validation/scoring: 0.
- Input paths were constructed from P1–P5 and model seeds 0–2. Only the specified NPZ files were read as scoring inputs; no pending NPZs, sidecars, external results, reports, pre-existing logs, or source code were read.
- Every input was loaded with allow_pickle=False; SHA-256 hashes are stored in JSON.
- Sample: six caption arrays each have 99,999 rows; two painting arrays each have 19,889 rows; subset_pid has 5,000 rows.
- Integer indices/labels, int64 sample IDs, unique sample IDs and subset indices, index bounds, actual first-caption indices, caption-to-painting names, and emotion range -1–8 were checked.
- Emotion uses nonnegative-caption majority and smallest-ID tie breaking; 1295 evaluation paintings have a nonempty tied majority. Empty votes receive -1.
- Style and genre use first[subset_pid], preserving evaluation row order. Only nonnegative labels enter AMI; all 5,000 rows enter k-means.
- Each output must contain only eval_prob, have shape (5000, K), K >= 2, float32 dtype, finite values, and range [0, 1].
- Normalization uses float32 x / max(row L2 norm, 1e-12); finite normalized values and unit-norm error <= 1e-5 were checked. Zero rows remain zero.
- Exactly three MiniBatchKMeans partitions per file use n_clusters=20, random_state=0/1/2, n_init=3, batch_size=4096, other defaults; each partition is shared by all labels.
- AMI uses sklearn default arguments. Argmax uses the original probabilities and first-maximum tie breaking. All AMIs and statistics must be finite.
- Arm aggregates use present, successfully scored model seeds; comparator eligibility requires all three. SD is population SD throughout.

### Label counts

| Label | Scored | Excluded | Classes | Counts (including negative labels) |
|---|---:|---:|---:|---|
| emotion | 5000 | 0 | 9 | {"0": 780, "1": 55, "2": 974, "3": 1758, "4": 149, "5": 215, "6": 410, "7": 373, "8": 286} |
| style | 5000 | 0 | 27 | {"0": 170, "1": 6, "10": 45, "11": 80, "12": 756, "13": 88, "14": 76, "15": 141, "16": 20, "17": 152, "18": 26, "19": 98, "2": 7, "20": 448, "21": 687, "22": 135, "23": 436, "24": 253, "25": 14, "26": 66, "3": 278, "4": 276, "5": 96, "6": 27, "7": 105, "8": 76, "9": 438} |
| genre | 4053 | 947 | 10 | {"-1": 947, "0": 302, "1": 296, "2": 682, "3": 125, "4": 827, "5": 97, "6": 883, "7": 406, "8": 253, "9": 182} |

### Per-file shapes and value ranges

| File | Shape | Min | Max | Zero norm rows | Argmax tied rows |
|---|---|---:|---:|---:|---:|
| P1__s0.npz | [5000, 99] | 0.006102881394326687 | 0.43085339665412903 | 0 | 0 |
| P1__s1.npz | [5000, 2] | 0.8334795832633972 | 0.9649614691734314 | 0 | 1 |
| P1__s2.npz | [5000, 3] | 0.042583152651786804 | 0.974626362323761 | 0 | 0 |
| P2__s0.npz | [5000, 99] | 2.4080579805740854e-06 | 0.8760653734207153 | 0 | 0 |
| P2__s1.npz | [5000, 2] | 0.470048189163208 | 0.9991416931152344 | 0 | 0 |
| P2__s2.npz | [5000, 3] | 7.381760042335372e-06 | 0.9999998807907104 | 0 | 0 |
| P3__s1.npz | [5000, 99] | 1.9037916842989944e-07 | 0.9699992537498474 | 0 | 0 |
| P3__s2.npz | [5000, 99] | 5.21022002430982e-07 | 0.9749599695205688 | 0 | 0 |
| P4__s1.npz | [5000, 99] | 0.003058579983189702 | 0.2497972697019577 | 0 | 0 |
| P4__s2.npz | [5000, 99] | 0.0037410941440612078 | 0.251486599445343 | 0 | 0 |

### Absent files

- /project/CoSiR/src/test/20261018_percept_core/outputs/stage2/P3__s0.npz
- /project/CoSiR/src/test/20261018_percept_core/outputs/stage2/P4__s0.npz
- /project/CoSiR/src/test/20261018_percept_core/outputs/stage2/P5__s0.npz
- /project/CoSiR/src/test/20261018_percept_core/outputs/stage2/P5__s1.npz
- /project/CoSiR/src/test/20261018_percept_core/outputs/stage2/P5__s2.npz

### Problems or observations

- P1__s1.npz: 1 argmax-tied rows; first maximum used as specified.
- P1: K varies across available model seeds: {'0': 99, '1': 2, '2': 3}. This is permitted by the supplied rule.
- P2: K varies across available model seeds: {'0': 99, '1': 2, '2': 3}. This is permitted by the supplied rule.
- 5 expected model files absent; see complete list above.

### Execution

- Python: 3.11.15; NumPy: 2.2.6; SciPy: 1.17.1; scikit-learn: 1.6.1.
- CPU execution; OMP_NUM_THREADS=8, MKL_NUM_THREADS=8, PYTHONDONTWRITEBYTECODE=1; CUDA_VISIBLE_DEVICES is empty.
- Elapsed seconds: 1.7952073070919141.
- Nothing deleted; no removal proposed.

## Serialized-output verification

- Re-read the saved JSON and checked all expected filenames are accounted for exactly once.
- Checked 30 partitions each have 20 cluster-count entries summing to 5,000, and 90 per-k-means-seed AMIs are finite.
- Independently recomputed 54 mean/SD pairs using Python math.fsum and population variance.
- Checked every numeric arm and per-file summary row matches the saved JSON rounded to six decimals.
- Recomputed comparator eligibility, maximum means, tolerance selection, and selected-arm SDs.
- Checked each label's scored and excluded counts and class counts sum to 5,000.
- All verification checks passed; details are in rederive_verification.json.
