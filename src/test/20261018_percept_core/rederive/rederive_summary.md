# Independent C13 re-derivation

Part A values are means and population SDs over run seeds. Each side score first averages AMI over k-means seeds 0, 1, 2. Target rows have label equal to aspect; the both-sides target aggregate is included in those rows and explicitly in JSON.

| Group | Aspect | Side | Label | Target label | Mean | SD over run seeds |
|---|---|---|---|---|---:|---:|
| row 6 | emotion | img | emotion | yes | 0.070932 | 0.002056 |
| row 6 | emotion | img | style |  | 0.156280 | 0.003668 |
| row 6 | emotion | img | genre |  | 0.416956 | 0.002890 |
| row 6 | emotion | cap | emotion | yes | 0.093751 | 0.003836 |
| row 6 | emotion | cap | style |  | 0.128852 | 0.002862 |
| row 6 | emotion | cap | genre |  | 0.359149 | 0.006348 |
| row 6 | emotion | both_sides | emotion | yes | 0.082342 | 0.002690 |
| row 6 | emotion | both_sides | style |  | 0.142566 | 0.003088 |
| row 6 | emotion | both_sides | genre |  | 0.388052 | 0.001737 |
| row 6 | style | img | emotion |  | 0.067340 | 0.002695 |
| row 6 | style | img | style | yes | 0.186106 | 0.005616 |
| row 6 | style | img | genre |  | 0.429405 | 0.005553 |
| row 6 | style | cap | emotion |  | 0.076653 | 0.001291 |
| row 6 | style | cap | style | yes | 0.139810 | 0.002120 |
| row 6 | style | cap | genre |  | 0.389320 | 0.008274 |
| row 6 | style | both_sides | emotion |  | 0.071997 | 0.001038 |
| row 6 | style | both_sides | style | yes | 0.162958 | 0.001826 |
| row 6 | style | both_sides | genre |  | 0.409363 | 0.004923 |
| row 6 | genre | img | emotion |  | 0.070932 | 0.002056 |
| row 6 | genre | img | style |  | 0.156280 | 0.003668 |
| row 6 | genre | img | genre | yes | 0.416956 | 0.002890 |
| row 6 | genre | cap | emotion |  | 0.093751 | 0.003836 |
| row 6 | genre | cap | style |  | 0.128852 | 0.002862 |
| row 6 | genre | cap | genre | yes | 0.359149 | 0.006348 |
| row 6 | genre | both_sides | emotion |  | 0.082342 | 0.002690 |
| row 6 | genre | both_sides | style |  | 0.142566 | 0.003088 |
| row 6 | genre | both_sides | genre | yes | 0.388052 | 0.001737 |
| row 10 | emotion | img | emotion | yes | 0.083168 | 0.001120 |
| row 10 | emotion | img | style |  | 0.185637 | 0.004122 |
| row 10 | emotion | img | genre |  | 0.431095 | 0.005998 |
| row 10 | emotion | cap | emotion | yes | 0.088169 | 0.000939 |
| row 10 | emotion | cap | style |  | 0.141914 | 0.001866 |
| row 10 | emotion | cap | genre |  | 0.392684 | 0.006267 |
| row 10 | emotion | both_sides | emotion | yes | 0.085668 | 0.000251 |
| row 10 | emotion | both_sides | style |  | 0.163776 | 0.001651 |
| row 10 | emotion | both_sides | genre |  | 0.411890 | 0.002057 |
| row 10 | style | img | emotion |  | 0.080188 | 0.001426 |
| row 10 | style | img | style | yes | 0.171682 | 0.001384 |
| row 10 | style | img | genre |  | 0.427099 | 0.004622 |
| row 10 | style | cap | emotion |  | 0.121486 | 0.005442 |
| row 10 | style | cap | style | yes | 0.137314 | 0.001268 |
| row 10 | style | cap | genre |  | 0.356782 | 0.008299 |
| row 10 | style | both_sides | emotion |  | 0.100837 | 0.002992 |
| row 10 | style | both_sides | style | yes | 0.154498 | 0.001204 |
| row 10 | style | both_sides | genre |  | 0.391940 | 0.004453 |
| row 13 | emotion | img | emotion | yes | 0.083527 | 0.000907 |
| row 13 | emotion | img | style |  | 0.168905 | 0.002625 |
| row 13 | emotion | img | genre |  | 0.417012 | 0.006536 |
| row 13 | emotion | cap | emotion | yes | 0.094245 | 0.002416 |
| row 13 | emotion | cap | style |  | 0.140115 | 0.001717 |
| row 13 | emotion | cap | genre |  | 0.376749 | 0.001053 |
| row 13 | emotion | both_sides | emotion | yes | 0.088886 | 0.001303 |
| row 13 | emotion | both_sides | style |  | 0.154510 | 0.002143 |
| row 13 | emotion | both_sides | genre |  | 0.396881 | 0.002987 |
| row 13 | style | img | emotion |  | 0.079104 | 0.001899 |
| row 13 | style | img | style | yes | 0.174726 | 0.000517 |
| row 13 | style | img | genre |  | 0.435965 | 0.003508 |
| row 13 | style | cap | emotion |  | 0.125917 | 0.003151 |
| row 13 | style | cap | style | yes | 0.138039 | 0.000481 |
| row 13 | style | cap | genre |  | 0.346101 | 0.009061 |
| row 13 | style | both_sides | emotion |  | 0.102511 | 0.002234 |
| row 13 | style | both_sides | style | yes | 0.156383 | 0.000321 |
| row 13 | style | both_sides | genre |  | 0.391033 | 0.005825 |
| row 13 | genre | img | emotion |  | 0.075482 | 0.002055 |
| row 13 | genre | img | style |  | 0.219508 | 0.004771 |
| row 13 | genre | img | genre | yes | 0.468331 | 0.002714 |
| row 13 | genre | cap | emotion |  | 0.085323 | 0.001385 |
| row 13 | genre | cap | style |  | 0.146017 | 0.000939 |
| row 13 | genre | cap | genre | yes | 0.400137 | 0.002602 |
| row 13 | genre | both_sides | emotion |  | 0.080402 | 0.001653 |
| row 13 | genre | both_sides | style |  | 0.182763 | 0.002636 |
| row 13 | genre | both_sides | genre | yes | 0.434234 | 0.002648 |

## Gate: genre / R0 / default / run seed 0

SD below is over the three k-means seeds. Both-sides SD uses paired img/cap seeds. With one run seed, the SD over run seeds is zero (recorded in JSON).

| Side | Label | Mean | SD over k-means seeds |
|---|---|---:|---:|
| img | emotion | 0.074848 | 0.003013 |
| img | style | 0.172430 | 0.006264 |
| img | genre | 0.436881 | 0.003784 |
| cap | emotion | 0.103434 | 0.007603 |
| cap | style | 0.143486 | 0.001892 |
| cap | genre | 0.375317 | 0.003723 |
| both_sides | emotion | 0.089141 | 0.005300 |
| both_sides | style | 0.157958 | 0.003983 |
| both_sides | genre | 0.406099 | 0.003604 |

## Part B: September encoders

SD is over the three k-means seeds; both-sides SD uses paired img/cap seeds.

| Side / encoder | Label | Mean | SD over k-means seeds |
|---|---|---:|---:|
| img_clip | emotion | 0.064741 | 0.002211 |
| img_clip | style | 0.300013 | 0.002735 |
| img_clip | genre | 0.411582 | 0.009767 |
| cap_clip | emotion | 0.066715 | 0.003342 |
| cap_clip | style | 0.109923 | 0.001693 |
| cap_clip | genre | 0.330445 | 0.019535 |
| cap_roberta | emotion | 0.189365 | 0.002252 |
| cap_roberta | style | 0.048655 | 0.001033 |
| cap_roberta | genre | 0.067952 | 0.000678 |
| both_clip | emotion | 0.065728 | 0.002017 |
| both_clip | style | 0.204968 | 0.000831 |
| both_clip | genre | 0.371014 | 0.005145 |
| both_roberta | emotion | 0.127053 | 0.001999 |
| both_roberta | style | 0.174334 | 0.001427 |
| both_roberta | genre | 0.239767 | 0.004732 |

## Checks

Largest head-row norm deviation after float32 cast, before any scoring: 3.737211227416992e-05; required tolerance: 0.01.

Each check and its observed counts are listed below. Class-count key -1 denotes excluded labels. The JSON also contains all 5,000 evaluation labels, per-file norm statistics, per-fit cluster counts, and all individual k-means-seed AMIs.

- **PASS sample_counts**: {"captions": 99999, "evaluation_paintings": 5000, "paintings": 19889}
- **PASS sample_id_dtype_and_uniqueness**: {"dtype": "int64", "unique_sample_ids": 99999}
- **PASS pid_range**: {"max": 19888, "min": 0}
- **PASS all_paintings_have_captions**: {"caption_count_distribution": {"10": 159, "11": 77, "12": 15, "3": 407, "4": 3378, "5": 12858, "6": 2782, "7": 113, "8": 24, "9": 76}, "max_captions": 12, "min_captions": 3, "sum": 99999}
- **PASS first_caption_indices**: {"checked_paintings": 19889}
- **PASS painting_names_by_pid**: {"checked_captions": 99999}
- **PASS unique_painting_names**: {"unique_names": 19889}
- **PASS subset_valid_unique**: {"unique_pids": 5000}
- **PASS subset_name_order**: {"checked_paintings": 5000}
- **PASS auxiliary_subset_image_indices**: {"matches_first_caption_row": 62, "note": "Field semantics are not specified by the task; use first[subset_pid] for image sample IDs.", "used_for_scoring": false}
- **PASS emotion_class_range**: {"0": 10180, "1": 1458, "2": 16237, "3": 28366, "4": 4820, "5": 8429, "6": 9100, "7": 10485, "8": 10924}
- **PASS emotion_majority**: {"all_paintings_with_tied_nonempty_majority": 5109, "all_paintings_without_emotion": 0, "evaluation_paintings_with_tied_nonempty_majority": 1295, "tie_rule": "smallest class id"}
- **PASS evaluation_label_counts**: {"emotion": {"class_counts_including_missing": {"0": 780, "1": 55, "2": 974, "3": 1758, "4": 149, "5": 215, "6": 410, "7": 373, "8": 286}, "excluded": 0, "valid": 5000}, "genre": {"class_counts_including_missing": {"-1": 947, "0": 302, "1": 296, "2": 682, "3": 125, "4": 827, "5": 97, "6": 883, "7": 406, "8": 253, "9": 182}, "excluded": 947, "valid": 4053}, "style": {"class_counts_including_missing": {"0": 170, "1": 6, "10": 45, "11": 80, "12": 756, "13": 88, "14": 76, "15": 141, "16": 20, "17": 152, "18": 26, "19": 98, "2": 7, "20": 448, "21": 687, "22": 135, "23": 436, "24": 253, "25": 14, "26": 66, "3": 278, "4": 276, "5": 96, "6": 27, "7": 105, "8": 76, "9": 438}, "excluded": 0, "valid": 5000}}
- **PASS head_inventory**: {"expected": 25, "found": 25, "missing": [], "pending_skipped": [], "unexpected": []}
- **PASS emotion__Ddef__s0.npz_arrays**: {"array_names": ["img", "cap"]}
- **PASS emotion__Ddef__s0.npz_img_stored_shape_dtype**: {"dtype": "float16", "shape": [5000, 2048]}
- **PASS emotion__Ddef__s0.npz_img_unit_norm**: {"max_abs_deviation_from_one": 3.737211227416992e-05, "max_norm": 1.0000300407409668, "min_norm": 0.9999626278877258, "rows": 5000, "rows_outside_0_01": 0}
- **PASS emotion__Ddef__s0.npz_cap_stored_shape_dtype**: {"dtype": "float16", "shape": [5000, 2048]}
- **PASS emotion__Ddef__s0.npz_cap_unit_norm**: {"max_abs_deviation_from_one": 2.849102020263672e-05, "max_norm": 1.0000284910202026, "min_norm": 0.9999740123748779, "rows": 5000, "rows_outside_0_01": 0}
- **PASS emotion__Ddef__s1.npz_arrays**: {"array_names": ["img", "cap"]}
- **PASS emotion__Ddef__s1.npz_img_stored_shape_dtype**: {"dtype": "float16", "shape": [5000, 2048]}
- **PASS emotion__Ddef__s1.npz_img_unit_norm**: {"max_abs_deviation_from_one": 2.5510787963867188e-05, "max_norm": 1.0000255107879639, "min_norm": 0.9999754428863525, "rows": 5000, "rows_outside_0_01": 0}
- **PASS emotion__Ddef__s1.npz_cap_stored_shape_dtype**: {"dtype": "float16", "shape": [5000, 2048]}
- **PASS emotion__Ddef__s1.npz_cap_unit_norm**: {"max_abs_deviation_from_one": 2.562999725341797e-05, "max_norm": 1.000024437904358, "min_norm": 0.9999743700027466, "rows": 5000, "rows_outside_0_01": 0}
- **PASS emotion__Ddef__s2.npz_arrays**: {"array_names": ["img", "cap"]}
- **PASS emotion__Ddef__s2.npz_img_stored_shape_dtype**: {"dtype": "float16", "shape": [5000, 2048]}
- **PASS emotion__Ddef__s2.npz_img_unit_norm**: {"max_abs_deviation_from_one": 2.9921531677246094e-05, "max_norm": 1.0000299215316772, "min_norm": 0.999971330165863, "rows": 5000, "rows_outside_0_01": 0}
- **PASS emotion__Ddef__s2.npz_cap_stored_shape_dtype**: {"dtype": "float16", "shape": [5000, 2048]}
- **PASS emotion__Ddef__s2.npz_cap_unit_norm**: {"max_abs_deviation_from_one": 2.9385089874267578e-05, "max_norm": 1.0000247955322266, "min_norm": 0.9999706149101257, "rows": 5000, "rows_outside_0_01": 0}
- **PASS emotion__R0o__prompted__s0.npz_arrays**: {"array_names": ["img", "cap"]}
- **PASS emotion__R0o__prompted__s0.npz_img_stored_shape_dtype**: {"dtype": "float16", "shape": [5000, 2048]}
- **PASS emotion__R0o__prompted__s0.npz_img_unit_norm**: {"max_abs_deviation_from_one": 3.600120544433594e-05, "max_norm": 1.0000313520431519, "min_norm": 0.9999639987945557, "rows": 5000, "rows_outside_0_01": 0}
- **PASS emotion__R0o__prompted__s0.npz_cap_stored_shape_dtype**: {"dtype": "float16", "shape": [5000, 2048]}
- **PASS emotion__R0o__prompted__s0.npz_cap_unit_norm**: {"max_abs_deviation_from_one": 3.0517578125e-05, "max_norm": 1.000026822090149, "min_norm": 0.999969482421875, "rows": 5000, "rows_outside_0_01": 0}
- **PASS emotion__R0o__prompted__s1.npz_arrays**: {"array_names": ["img", "cap"]}
- **PASS emotion__R0o__prompted__s1.npz_img_stored_shape_dtype**: {"dtype": "float16", "shape": [5000, 2048]}
- **PASS emotion__R0o__prompted__s1.npz_img_unit_norm**: {"max_abs_deviation_from_one": 3.1113624572753906e-05, "max_norm": 1.0000311136245728, "min_norm": 0.9999712705612183, "rows": 5000, "rows_outside_0_01": 0}
- **PASS emotion__R0o__prompted__s1.npz_cap_stored_shape_dtype**: {"dtype": "float16", "shape": [5000, 2048]}
- **PASS emotion__R0o__prompted__s1.npz_cap_unit_norm**: {"max_abs_deviation_from_one": 3.2067298889160156e-05, "max_norm": 1.0000320672988892, "min_norm": 0.9999732375144958, "rows": 5000, "rows_outside_0_01": 0}
- **PASS emotion__R0o__prompted__s2.npz_arrays**: {"array_names": ["img", "cap"]}
- **PASS emotion__R0o__prompted__s2.npz_img_stored_shape_dtype**: {"dtype": "float16", "shape": [5000, 2048]}
- **PASS emotion__R0o__prompted__s2.npz_img_unit_norm**: {"max_abs_deviation_from_one": 3.039836883544922e-05, "max_norm": 1.0000303983688354, "min_norm": 0.999973475933075, "rows": 5000, "rows_outside_0_01": 0}
- **PASS emotion__R0o__prompted__s2.npz_cap_stored_shape_dtype**: {"dtype": "float16", "shape": [5000, 2048]}
- **PASS emotion__R0o__prompted__s2.npz_cap_unit_norm**: {"max_abs_deviation_from_one": 3.0994415283203125e-05, "max_norm": 1.0000277757644653, "min_norm": 0.9999690055847168, "rows": 5000, "rows_outside_0_01": 0}
- **PASS emotion__R0o__s0.npz_arrays**: {"array_names": ["img", "cap"]}
- **PASS emotion__R0o__s0.npz_img_stored_shape_dtype**: {"dtype": "float16", "shape": [5000, 2048]}
- **PASS emotion__R0o__s0.npz_img_unit_norm**: {"max_abs_deviation_from_one": 2.872943878173828e-05, "max_norm": 1.0000280141830444, "min_norm": 0.9999712705612183, "rows": 5000, "rows_outside_0_01": 0}
- **PASS emotion__R0o__s0.npz_cap_stored_shape_dtype**: {"dtype": "float16", "shape": [5000, 2048]}
- **PASS emotion__R0o__s0.npz_cap_unit_norm**: {"max_abs_deviation_from_one": 2.777576446533203e-05, "max_norm": 1.0000277757644653, "min_norm": 0.9999740719795227, "rows": 5000, "rows_outside_0_01": 0}
- **PASS emotion__R0o__s1.npz_arrays**: {"array_names": ["img", "cap"]}
- **PASS emotion__R0o__s1.npz_img_stored_shape_dtype**: {"dtype": "float16", "shape": [5000, 2048]}
- **PASS emotion__R0o__s1.npz_img_unit_norm**: {"max_abs_deviation_from_one": 3.147125244140625e-05, "max_norm": 1.00002920627594, "min_norm": 0.9999685287475586, "rows": 5000, "rows_outside_0_01": 0}
- **PASS emotion__R0o__s1.npz_cap_stored_shape_dtype**: {"dtype": "float16", "shape": [5000, 2048]}
- **PASS emotion__R0o__s1.npz_cap_unit_norm**: {"max_abs_deviation_from_one": 2.5928020477294922e-05, "max_norm": 1.000024676322937, "min_norm": 0.9999740719795227, "rows": 5000, "rows_outside_0_01": 0}
- **PASS emotion__R0o__s2.npz_arrays**: {"array_names": ["img", "cap"]}
- **PASS emotion__R0o__s2.npz_img_stored_shape_dtype**: {"dtype": "float16", "shape": [5000, 2048]}
- **PASS emotion__R0o__s2.npz_img_unit_norm**: {"max_abs_deviation_from_one": 3.039836883544922e-05, "max_norm": 1.0000303983688354, "min_norm": 0.9999712705612183, "rows": 5000, "rows_outside_0_01": 0}
- **PASS emotion__R0o__s2.npz_cap_stored_shape_dtype**: {"dtype": "float16", "shape": [5000, 2048]}
- **PASS emotion__R0o__s2.npz_cap_unit_norm**: {"max_abs_deviation_from_one": 2.9325485229492188e-05, "max_norm": 1.0000293254852295, "min_norm": 0.9999709725379944, "rows": 5000, "rows_outside_0_01": 0}
- **PASS genre__Ddef__s0.npz_arrays**: {"array_names": ["img", "cap"]}
- **PASS genre__Ddef__s0.npz_img_stored_shape_dtype**: {"dtype": "float16", "shape": [5000, 2048]}
- **PASS genre__Ddef__s0.npz_img_unit_norm**: {"max_abs_deviation_from_one": 3.737211227416992e-05, "max_norm": 1.0000300407409668, "min_norm": 0.9999626278877258, "rows": 5000, "rows_outside_0_01": 0}
- **PASS genre__Ddef__s0.npz_cap_stored_shape_dtype**: {"dtype": "float16", "shape": [5000, 2048]}
- **PASS genre__Ddef__s0.npz_cap_unit_norm**: {"max_abs_deviation_from_one": 2.849102020263672e-05, "max_norm": 1.0000284910202026, "min_norm": 0.9999740123748779, "rows": 5000, "rows_outside_0_01": 0}
- **PASS genre__Ddef__s1.npz_arrays**: {"array_names": ["img", "cap"]}
- **PASS genre__Ddef__s1.npz_img_stored_shape_dtype**: {"dtype": "float16", "shape": [5000, 2048]}
- **PASS genre__Ddef__s1.npz_img_unit_norm**: {"max_abs_deviation_from_one": 2.5510787963867188e-05, "max_norm": 1.0000255107879639, "min_norm": 0.9999754428863525, "rows": 5000, "rows_outside_0_01": 0}
- **PASS genre__Ddef__s1.npz_cap_stored_shape_dtype**: {"dtype": "float16", "shape": [5000, 2048]}
- **PASS genre__Ddef__s1.npz_cap_unit_norm**: {"max_abs_deviation_from_one": 2.562999725341797e-05, "max_norm": 1.000024437904358, "min_norm": 0.9999743700027466, "rows": 5000, "rows_outside_0_01": 0}
- **PASS genre__Ddef__s2.npz_arrays**: {"array_names": ["img", "cap"]}
- **PASS genre__Ddef__s2.npz_img_stored_shape_dtype**: {"dtype": "float16", "shape": [5000, 2048]}
- **PASS genre__Ddef__s2.npz_img_unit_norm**: {"max_abs_deviation_from_one": 2.9921531677246094e-05, "max_norm": 1.0000299215316772, "min_norm": 0.999971330165863, "rows": 5000, "rows_outside_0_01": 0}
- **PASS genre__Ddef__s2.npz_cap_stored_shape_dtype**: {"dtype": "float16", "shape": [5000, 2048]}
- **PASS genre__Ddef__s2.npz_cap_unit_norm**: {"max_abs_deviation_from_one": 2.9385089874267578e-05, "max_norm": 1.0000247955322266, "min_norm": 0.9999706149101257, "rows": 5000, "rows_outside_0_01": 0}
- **PASS genre__R0__s0.npz_arrays**: {"array_names": ["img", "cap"]}
- **PASS genre__R0__s0.npz_img_stored_shape_dtype**: {"dtype": "float16", "shape": [5000, 2048]}
- **PASS genre__R0__s0.npz_img_unit_norm**: {"max_abs_deviation_from_one": 2.562999725341797e-05, "max_norm": 1.0000256299972534, "min_norm": 0.9999749064445496, "rows": 5000, "rows_outside_0_01": 0}
- **PASS genre__R0__s0.npz_cap_stored_shape_dtype**: {"dtype": "float16", "shape": [5000, 2048]}
- **PASS genre__R0__s0.npz_cap_unit_norm**: {"max_abs_deviation_from_one": 2.9206275939941406e-05, "max_norm": 1.0000249147415161, "min_norm": 0.9999707937240601, "rows": 5000, "rows_outside_0_01": 0}
- **PASS genre__R0o__prompted__s0.npz_arrays**: {"array_names": ["img", "cap"]}
- **PASS genre__R0o__prompted__s0.npz_img_stored_shape_dtype**: {"dtype": "float16", "shape": [5000, 2048]}
- **PASS genre__R0o__prompted__s0.npz_img_unit_norm**: {"max_abs_deviation_from_one": 2.7418136596679688e-05, "max_norm": 1.0000274181365967, "min_norm": 0.9999735355377197, "rows": 5000, "rows_outside_0_01": 0}
- **PASS genre__R0o__prompted__s0.npz_cap_stored_shape_dtype**: {"dtype": "float16", "shape": [5000, 2048]}
- **PASS genre__R0o__prompted__s0.npz_cap_unit_norm**: {"max_abs_deviation_from_one": 2.753734588623047e-05, "max_norm": 1.0000275373458862, "min_norm": 0.9999732375144958, "rows": 5000, "rows_outside_0_01": 0}
- **PASS genre__R0o__prompted__s1.npz_arrays**: {"array_names": ["img", "cap"]}
- **PASS genre__R0o__prompted__s1.npz_img_stored_shape_dtype**: {"dtype": "float16", "shape": [5000, 2048]}
- **PASS genre__R0o__prompted__s1.npz_img_unit_norm**: {"max_abs_deviation_from_one": 3.0517578125e-05, "max_norm": 1.000030517578125, "min_norm": 0.999972403049469, "rows": 5000, "rows_outside_0_01": 0}
- **PASS genre__R0o__prompted__s1.npz_cap_stored_shape_dtype**: {"dtype": "float16", "shape": [5000, 2048]}
- **PASS genre__R0o__prompted__s1.npz_cap_unit_norm**: {"max_abs_deviation_from_one": 3.1113624572753906e-05, "max_norm": 1.0000271797180176, "min_norm": 0.9999688863754272, "rows": 5000, "rows_outside_0_01": 0}
- **PASS genre__R0o__prompted__s2.npz_arrays**: {"array_names": ["img", "cap"]}
- **PASS genre__R0o__prompted__s2.npz_img_stored_shape_dtype**: {"dtype": "float16", "shape": [5000, 2048]}
- **PASS genre__R0o__prompted__s2.npz_img_unit_norm**: {"max_abs_deviation_from_one": 3.325939178466797e-05, "max_norm": 1.0000332593917847, "min_norm": 0.9999730587005615, "rows": 5000, "rows_outside_0_01": 0}
- **PASS genre__R0o__prompted__s2.npz_cap_stored_shape_dtype**: {"dtype": "float16", "shape": [5000, 2048]}
- **PASS genre__R0o__prompted__s2.npz_cap_unit_norm**: {"max_abs_deviation_from_one": 2.682209014892578e-05, "max_norm": 1.000026822090149, "min_norm": 0.9999748468399048, "rows": 5000, "rows_outside_0_01": 0}
- **PASS style__Ddef__s0.npz_arrays**: {"array_names": ["img", "cap"]}
- **PASS style__Ddef__s0.npz_img_stored_shape_dtype**: {"dtype": "float16", "shape": [5000, 2048]}
- **PASS style__Ddef__s0.npz_img_unit_norm**: {"max_abs_deviation_from_one": 3.218650817871094e-05, "max_norm": 1.0000239610671997, "min_norm": 0.9999678134918213, "rows": 5000, "rows_outside_0_01": 0}
- **PASS style__Ddef__s0.npz_cap_stored_shape_dtype**: {"dtype": "float16", "shape": [5000, 2048]}
- **PASS style__Ddef__s0.npz_cap_unit_norm**: {"max_abs_deviation_from_one": 2.8014183044433594e-05, "max_norm": 1.0000280141830444, "min_norm": 0.9999719858169556, "rows": 5000, "rows_outside_0_01": 0}
- **PASS style__Ddef__s1.npz_arrays**: {"array_names": ["img", "cap"]}
- **PASS style__Ddef__s1.npz_img_stored_shape_dtype**: {"dtype": "float16", "shape": [5000, 2048]}
- **PASS style__Ddef__s1.npz_img_unit_norm**: {"max_abs_deviation_from_one": 3.4868717193603516e-05, "max_norm": 1.0000282526016235, "min_norm": 0.9999651312828064, "rows": 5000, "rows_outside_0_01": 0}
- **PASS style__Ddef__s1.npz_cap_stored_shape_dtype**: {"dtype": "float16", "shape": [5000, 2048]}
- **PASS style__Ddef__s1.npz_cap_unit_norm**: {"max_abs_deviation_from_one": 3.325939178466797e-05, "max_norm": 1.0000332593917847, "min_norm": 0.9999712109565735, "rows": 5000, "rows_outside_0_01": 0}
- **PASS style__Ddef__s2.npz_arrays**: {"array_names": ["img", "cap"]}
- **PASS style__Ddef__s2.npz_img_stored_shape_dtype**: {"dtype": "float16", "shape": [5000, 2048]}
- **PASS style__Ddef__s2.npz_img_unit_norm**: {"max_abs_deviation_from_one": 3.331899642944336e-05, "max_norm": 1.0000274181365967, "min_norm": 0.9999666810035706, "rows": 5000, "rows_outside_0_01": 0}
- **PASS style__Ddef__s2.npz_cap_stored_shape_dtype**: {"dtype": "float16", "shape": [5000, 2048]}
- **PASS style__Ddef__s2.npz_cap_unit_norm**: {"max_abs_deviation_from_one": 2.6881694793701172e-05, "max_norm": 1.000025749206543, "min_norm": 0.9999731183052063, "rows": 5000, "rows_outside_0_01": 0}
- **PASS style__R0o__prompted__s0.npz_arrays**: {"array_names": ["img", "cap"]}
- **PASS style__R0o__prompted__s0.npz_img_stored_shape_dtype**: {"dtype": "float16", "shape": [5000, 2048]}
- **PASS style__R0o__prompted__s0.npz_img_unit_norm**: {"max_abs_deviation_from_one": 2.7060508728027344e-05, "max_norm": 1.000027060508728, "min_norm": 0.9999745488166809, "rows": 5000, "rows_outside_0_01": 0}
- **PASS style__R0o__prompted__s0.npz_cap_stored_shape_dtype**: {"dtype": "float16", "shape": [5000, 2048]}
- **PASS style__R0o__prompted__s0.npz_cap_unit_norm**: {"max_abs_deviation_from_one": 2.9027462005615234e-05, "max_norm": 1.0000274181365967, "min_norm": 0.9999709725379944, "rows": 5000, "rows_outside_0_01": 0}
- **PASS style__R0o__prompted__s1.npz_arrays**: {"array_names": ["img", "cap"]}
- **PASS style__R0o__prompted__s1.npz_img_stored_shape_dtype**: {"dtype": "float16", "shape": [5000, 2048]}
- **PASS style__R0o__prompted__s1.npz_img_unit_norm**: {"max_abs_deviation_from_one": 2.9921531677246094e-05, "max_norm": 1.0000267028808594, "min_norm": 0.9999700784683228, "rows": 5000, "rows_outside_0_01": 0}
- **PASS style__R0o__prompted__s1.npz_cap_stored_shape_dtype**: {"dtype": "float16", "shape": [5000, 2048]}
- **PASS style__R0o__prompted__s1.npz_cap_unit_norm**: {"max_abs_deviation_from_one": 3.063678741455078e-05, "max_norm": 1.0000306367874146, "min_norm": 0.9999743700027466, "rows": 5000, "rows_outside_0_01": 0}
- **PASS style__R0o__prompted__s2.npz_arrays**: {"array_names": ["img", "cap"]}
- **PASS style__R0o__prompted__s2.npz_img_stored_shape_dtype**: {"dtype": "float16", "shape": [5000, 2048]}
- **PASS style__R0o__prompted__s2.npz_img_unit_norm**: {"max_abs_deviation_from_one": 2.968311309814453e-05, "max_norm": 1.0000296831130981, "min_norm": 0.9999704957008362, "rows": 5000, "rows_outside_0_01": 0}
- **PASS style__R0o__prompted__s2.npz_cap_stored_shape_dtype**: {"dtype": "float16", "shape": [5000, 2048]}
- **PASS style__R0o__prompted__s2.npz_cap_unit_norm**: {"max_abs_deviation_from_one": 3.409385681152344e-05, "max_norm": 1.0000287294387817, "min_norm": 0.9999659061431885, "rows": 5000, "rows_outside_0_01": 0}
- **PASS style__R0o__s0.npz_arrays**: {"array_names": ["img", "cap"]}
- **PASS style__R0o__s0.npz_img_stored_shape_dtype**: {"dtype": "float16", "shape": [5000, 2048]}
- **PASS style__R0o__s0.npz_img_unit_norm**: {"max_abs_deviation_from_one": 2.6941299438476562e-05, "max_norm": 1.0000269412994385, "min_norm": 0.9999741315841675, "rows": 5000, "rows_outside_0_01": 0}
- **PASS style__R0o__s0.npz_cap_stored_shape_dtype**: {"dtype": "float16", "shape": [5000, 2048]}
- **PASS style__R0o__s0.npz_cap_unit_norm**: {"max_abs_deviation_from_one": 3.0875205993652344e-05, "max_norm": 1.0000308752059937, "min_norm": 0.9999740719795227, "rows": 5000, "rows_outside_0_01": 0}
- **PASS style__R0o__s1.npz_arrays**: {"array_names": ["img", "cap"]}
- **PASS style__R0o__s1.npz_img_stored_shape_dtype**: {"dtype": "float16", "shape": [5000, 2048]}
- **PASS style__R0o__s1.npz_img_unit_norm**: {"max_abs_deviation_from_one": 3.147125244140625e-05, "max_norm": 1.0000314712524414, "min_norm": 0.9999694228172302, "rows": 5000, "rows_outside_0_01": 0}
- **PASS style__R0o__s1.npz_cap_stored_shape_dtype**: {"dtype": "float16", "shape": [5000, 2048]}
- **PASS style__R0o__s1.npz_cap_unit_norm**: {"max_abs_deviation_from_one": 3.063678741455078e-05, "max_norm": 1.0000277757644653, "min_norm": 0.9999693632125854, "rows": 5000, "rows_outside_0_01": 0}
- **PASS style__R0o__s2.npz_arrays**: {"array_names": ["img", "cap"]}
- **PASS style__R0o__s2.npz_img_stored_shape_dtype**: {"dtype": "float16", "shape": [5000, 2048]}
- **PASS style__R0o__s2.npz_img_unit_norm**: {"max_abs_deviation_from_one": 3.248453140258789e-05, "max_norm": 1.0000274181365967, "min_norm": 0.9999675154685974, "rows": 5000, "rows_outside_0_01": 0}
- **PASS style__R0o__s2.npz_cap_stored_shape_dtype**: {"dtype": "float16", "shape": [5000, 2048]}
- **PASS style__R0o__s2.npz_cap_unit_norm**: {"max_abs_deviation_from_one": 3.2901763916015625e-05, "max_norm": 1.000032901763916, "min_norm": 0.9999722242355347, "rows": 5000, "rows_outside_0_01": 0}
- **PASS clip_metadata_counts**: {"metadata": {"available_features": ["img_features", "txt_features"], "backbone_model": "openai/clip-vit-base-patch32", "feature_dims": {"img_features": [512], "txt_features": [512]}, "hdf5_compression": true, "hdf5_compression_level": 4, "num_shards": 4, "shard_size": 100000, "total_samples": 308723}, "observed_rows": 308723, "observed_shards": 4}
- **PASS clip_caption_join_unique_ids**: {"required_rows": 99999, "stored_rows": 308723, "stored_unique_ids": 308723}
- **PASS clip_caption_join_all_required_ids_found**: {"extra_stored_ids": 208724, "matched": 99999, "missing": 0}
- **PASS clip_caption_join_join_order**: {"join_key": "sample_id", "required_rows_already_at_same_position": 0}
- **PASS clip_image_id_join**: {"matched": 5000, "requested_sample_ids": "sample_ids[first[subset_pid]]"}
- **PASS shard_00000_txt_features_shape_dtype**: {"dtype": "float32", "shape": [100000, 512]}
- **PASS shard_00000_img_features_shape_dtype**: {"dtype": "float32", "shape": [100000, 512]}
- **PASS shard_00001_txt_features_shape_dtype**: {"dtype": "float32", "shape": [100000, 512]}
- **PASS shard_00001_img_features_shape_dtype**: {"dtype": "float32", "shape": [100000, 512]}
- **PASS shard_00002_txt_features_shape_dtype**: {"dtype": "float32", "shape": [100000, 512]}
- **PASS shard_00002_img_features_shape_dtype**: {"dtype": "float32", "shape": [100000, 512]}
- **PASS shard_00003_txt_features_shape_dtype**: {"dtype": "float32", "shape": [8723, 512]}
- **PASS shard_00003_img_features_shape_dtype**: {"dtype": "float32", "shape": [8723, 512]}
- **PASS clip_evaluation_images_finite_nonzero**: {"dtype": "float32", "max_original_norm": 12.623445510864258, "min_original_norm": 7.89057731628418, "shape": [5000, 512]}
- **PASS clip_evaluation_images_normalized**: {"max_abs_deviation_from_one": 1.7881393432617188e-07, "max_norm": 1.0000001192092896, "min_norm": 0.9999998211860657, "rows": 5000, "rows_outside_0_01": 0}
- **PASS clip_captions_finite_nonzero**: {"dtype": "float32", "max_original_norm": 13.74708366394043, "min_original_norm": 6.151608467102051, "shape": [99999, 512]}
- **PASS clip_captions_normalized**: {"max_abs_deviation_from_one": 2.384185791015625e-07, "max_norm": 1.0000001192092896, "min_norm": 0.9999997615814209, "rows": 99999, "rows_outside_0_01": 0}
- **PASS clip_painting_means_finite_nonzero**: {"dtype": "float32", "max_original_norm": 0.9738513231277466, "min_original_norm": 0.7725581526756287, "shape": [19889, 512]}
- **PASS clip_painting_means_normalized**: {"max_abs_deviation_from_one": 1.7881393432617188e-07, "max_norm": 1.0000001192092896, "min_norm": 0.9999998211860657, "rows": 19889, "rows_outside_0_01": 0}
- **PASS roberta_shape**: {"id_shape": [99999], "painting_id_shape": [99999], "vectors_dtype": "float32", "vectors_shape": [99999, 768]}
- **PASS roberta_caption_join_unique_ids**: {"required_rows": 99999, "stored_rows": 99999, "stored_unique_ids": 99999}
- **PASS roberta_caption_join_all_required_ids_found**: {"extra_stored_ids": 0, "matched": 99999, "missing": 0}
- **PASS roberta_caption_join_join_order**: {"join_key": "sample_id", "required_rows_already_at_same_position": 99999}
- **PASS roberta_painting_ids_agree**: {"matched": 99999, "mismatched": 0}
- **PASS roberta_captions_finite_nonzero**: {"dtype": "float32", "max_original_norm": 22.929834365844727, "min_original_norm": 17.169126510620117, "shape": [99999, 768]}
- **PASS roberta_captions_normalized**: {"max_abs_deviation_from_one": 1.1920928955078125e-07, "max_norm": 1.0000001192092896, "min_norm": 0.9999998807907104, "rows": 99999, "rows_outside_0_01": 0}
- **PASS roberta_painting_means_finite_nonzero**: {"dtype": "float32", "max_original_norm": 0.9932155013084412, "min_original_norm": 0.5493723154067993, "shape": [19889, 768]}
- **PASS roberta_painting_means_normalized**: {"max_abs_deviation_from_one": 1.1920928955078125e-07, "max_norm": 1.0000001192092896, "min_norm": 0.9999998807907104, "rows": 19889, "rows_outside_0_01": 0}
- **PASS emotion__Ddef__s0.npz/img_scoring_matrix**: {"dtype": "float32", "shape": [5000, 2048]}
- **PASS emotion__Ddef__s0.npz/cap_scoring_matrix**: {"dtype": "float32", "shape": [5000, 2048]}
- **PASS emotion__Ddef__s1.npz/img_scoring_matrix**: {"dtype": "float32", "shape": [5000, 2048]}
- **PASS emotion__Ddef__s1.npz/cap_scoring_matrix**: {"dtype": "float32", "shape": [5000, 2048]}
- **PASS emotion__Ddef__s2.npz/img_scoring_matrix**: {"dtype": "float32", "shape": [5000, 2048]}
- **PASS emotion__Ddef__s2.npz/cap_scoring_matrix**: {"dtype": "float32", "shape": [5000, 2048]}
- **PASS style__Ddef__s0.npz/img_scoring_matrix**: {"dtype": "float32", "shape": [5000, 2048]}
- **PASS style__Ddef__s0.npz/cap_scoring_matrix**: {"dtype": "float32", "shape": [5000, 2048]}
- **PASS style__Ddef__s1.npz/img_scoring_matrix**: {"dtype": "float32", "shape": [5000, 2048]}
- **PASS style__Ddef__s1.npz/cap_scoring_matrix**: {"dtype": "float32", "shape": [5000, 2048]}
- **PASS style__Ddef__s2.npz/img_scoring_matrix**: {"dtype": "float32", "shape": [5000, 2048]}
- **PASS style__Ddef__s2.npz/cap_scoring_matrix**: {"dtype": "float32", "shape": [5000, 2048]}
- **PASS genre__Ddef__s0.npz/img_scoring_matrix**: {"dtype": "float32", "shape": [5000, 2048]}
- **PASS genre__Ddef__s0.npz/cap_scoring_matrix**: {"dtype": "float32", "shape": [5000, 2048]}
- **PASS genre__Ddef__s1.npz/img_scoring_matrix**: {"dtype": "float32", "shape": [5000, 2048]}
- **PASS genre__Ddef__s1.npz/cap_scoring_matrix**: {"dtype": "float32", "shape": [5000, 2048]}
- **PASS genre__Ddef__s2.npz/img_scoring_matrix**: {"dtype": "float32", "shape": [5000, 2048]}
- **PASS genre__Ddef__s2.npz/cap_scoring_matrix**: {"dtype": "float32", "shape": [5000, 2048]}
- **PASS emotion__R0o__s0.npz/img_scoring_matrix**: {"dtype": "float32", "shape": [5000, 2048]}
- **PASS emotion__R0o__s0.npz/cap_scoring_matrix**: {"dtype": "float32", "shape": [5000, 2048]}
- **PASS emotion__R0o__s1.npz/img_scoring_matrix**: {"dtype": "float32", "shape": [5000, 2048]}
- **PASS emotion__R0o__s1.npz/cap_scoring_matrix**: {"dtype": "float32", "shape": [5000, 2048]}
- **PASS emotion__R0o__s2.npz/img_scoring_matrix**: {"dtype": "float32", "shape": [5000, 2048]}
- **PASS emotion__R0o__s2.npz/cap_scoring_matrix**: {"dtype": "float32", "shape": [5000, 2048]}
- **PASS style__R0o__s0.npz/img_scoring_matrix**: {"dtype": "float32", "shape": [5000, 2048]}
- **PASS style__R0o__s0.npz/cap_scoring_matrix**: {"dtype": "float32", "shape": [5000, 2048]}
- **PASS style__R0o__s1.npz/img_scoring_matrix**: {"dtype": "float32", "shape": [5000, 2048]}
- **PASS style__R0o__s1.npz/cap_scoring_matrix**: {"dtype": "float32", "shape": [5000, 2048]}
- **PASS style__R0o__s2.npz/img_scoring_matrix**: {"dtype": "float32", "shape": [5000, 2048]}
- **PASS style__R0o__s2.npz/cap_scoring_matrix**: {"dtype": "float32", "shape": [5000, 2048]}
- **PASS emotion__R0o__prompted__s0.npz/img_scoring_matrix**: {"dtype": "float32", "shape": [5000, 2048]}
- **PASS emotion__R0o__prompted__s0.npz/cap_scoring_matrix**: {"dtype": "float32", "shape": [5000, 2048]}
- **PASS emotion__R0o__prompted__s1.npz/img_scoring_matrix**: {"dtype": "float32", "shape": [5000, 2048]}
- **PASS emotion__R0o__prompted__s1.npz/cap_scoring_matrix**: {"dtype": "float32", "shape": [5000, 2048]}
- **PASS emotion__R0o__prompted__s2.npz/img_scoring_matrix**: {"dtype": "float32", "shape": [5000, 2048]}
- **PASS emotion__R0o__prompted__s2.npz/cap_scoring_matrix**: {"dtype": "float32", "shape": [5000, 2048]}
- **PASS style__R0o__prompted__s0.npz/img_scoring_matrix**: {"dtype": "float32", "shape": [5000, 2048]}
- **PASS style__R0o__prompted__s0.npz/cap_scoring_matrix**: {"dtype": "float32", "shape": [5000, 2048]}
- **PASS style__R0o__prompted__s1.npz/img_scoring_matrix**: {"dtype": "float32", "shape": [5000, 2048]}
- **PASS style__R0o__prompted__s1.npz/cap_scoring_matrix**: {"dtype": "float32", "shape": [5000, 2048]}
- **PASS style__R0o__prompted__s2.npz/img_scoring_matrix**: {"dtype": "float32", "shape": [5000, 2048]}
- **PASS style__R0o__prompted__s2.npz/cap_scoring_matrix**: {"dtype": "float32", "shape": [5000, 2048]}
- **PASS genre__R0o__prompted__s0.npz/img_scoring_matrix**: {"dtype": "float32", "shape": [5000, 2048]}
- **PASS genre__R0o__prompted__s0.npz/cap_scoring_matrix**: {"dtype": "float32", "shape": [5000, 2048]}
- **PASS genre__R0o__prompted__s1.npz/img_scoring_matrix**: {"dtype": "float32", "shape": [5000, 2048]}
- **PASS genre__R0o__prompted__s1.npz/cap_scoring_matrix**: {"dtype": "float32", "shape": [5000, 2048]}
- **PASS genre__R0o__prompted__s2.npz/img_scoring_matrix**: {"dtype": "float32", "shape": [5000, 2048]}
- **PASS genre__R0o__prompted__s2.npz/cap_scoring_matrix**: {"dtype": "float32", "shape": [5000, 2048]}
- **PASS genre__R0__s0.npz/img_scoring_matrix**: {"dtype": "float32", "shape": [5000, 2048]}
- **PASS genre__R0__s0.npz/cap_scoring_matrix**: {"dtype": "float32", "shape": [5000, 2048]}
- **PASS row1/clip_img_scoring_matrix**: {"dtype": "float32", "shape": [5000, 512]}
- **PASS row1/clip_cap_scoring_matrix**: {"dtype": "float32", "shape": [5000, 512]}
- **PASS row1/roberta_cap_scoring_matrix**: {"dtype": "float32", "shape": [5000, 768]}

## Problems and file handling

- No problems detected.
- No files deleted. Would remove: none.
- No project source code, prior results, reports, or unlisted input files were read.
- Computation used CPU only, with OMP_NUM_THREADS=8 and MKL_NUM_THREADS=8.
- Versions: NumPy 2.2.6, scikit-learn 1.6.1, SciPy 1.17.1.

## Saved-output audit

- validate_results.py verified all 477 original AMIs, all reported means and population SDs, paired both-sides values, target aggregates, label counts, 159 partition counts, and the values and formatting of all 96 summary table rows.
