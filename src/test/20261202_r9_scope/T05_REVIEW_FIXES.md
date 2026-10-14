Ticket 05 review fixes. This hand-back supersedes the interface choices in the initial T05_HAND_BACK.md.

1. New-backbone style files retain CLIP's keys, shapes and dtypes, but Gram/random heads are NaN. Only `style_csd__img_src` and the selection indices are copied. Records identify `keys_unrefit` separately from `keys_copied`; CLIP retains its complete bit-equal file.
2. Affect-km labels are read from the frozen `parts(backbone)["affect_km"]` with the R6 SHA. Recipe `affect` must equal them. Both the existing label test and the review regression supply a differing recipe affect array before checking that it is rejected.
3. Explicit frozen pins for partition_L, style_csd labels, told_oracle, and affect-km must equal R6's exact `(Path, SHA-256)` tuple. The file bytes are also asserted against the frozen SHA.
4. `fit_heads_r9` captures the input pins, feature-store file hashes and loaded feature-array hashes before fitting, and stores that snapshot with every fitted head. The writer rechecks this snapshot; caller hashes are accepted only if equal to it. It asserts both data widths and every head/modality coefficient width against DIMENSIONS[backbone]. It no longer derives provenance from a fresh disk snapshot at write time.
5. The n6 record writes all five heads' `coef_sha256` and the affect head's `affect_prov`. A new-backbone selection check verifies the pinned record hash, its backbone, all coefficient hashes and affect provenance. The new manifest must declare the legacy n6 input key plus `.record.json`; CLIP's original 51 input entries remain unchanged.
6. Added regressions and single-line mutations for the review's routing, frozen-pin, draw-size, rule-hash, told-hash, copied-selection, exclusive-open and backbone-record cases. Mutation helpers preserve the tiny fit fixture so mutations are killed by the intended assertion, not an unrelated fit error. Guard markers already present remain intact.
7. `pins_path` now resolves in the round folder itself. Future generated pin files are to be committed with `git add -f`. Restored module-SHA coverage for r9 modules, runners and scripts, excluding DTS configuration. The check script reports copied and unrefitted arrays separately on new backbones and counts only the eight refit arrays. Fixed the requested stale docstrings.

Ticket 06's bundle/context/picks imports and R6 sys.path entry were left for that ticket.

Tests: 204 passed: 73 existing heads/common tests and 131 new review regressions/mutation tests. The existing suite had ten upstream SciPy deprecation warnings. Both Qwen and SigLIP are exercised. Commands:

```bash
timeout 235s env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python -m pytest -q -p no:cacheprovider --tb=short --basetemp=src/test/20261202_r9_scope/results/review-existing src/test/20261202_r9_scope/test_r9_heads.py src/test/20261202_r9_scope/test_r9_common.py
timeout 235s env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python -m pytest -q -p no:cacheprovider --tb=short --basetemp=src/test/20261202_r9_scope/results/review-final-mutants src/test/20261202_r9_scope/test_r9_review_fixes.py
```

Results: existing suite 73 passed in 79.39s; new suite 131 passed in 8.86s.

Mutations: 34 single-line forms, each killed on Qwen and SigLIP (68 kills); survived: 0.

- `unused_heads`: killed on both backbones (r9_heads.py; `test_new_backbone_unused_heads_are_nan_and_csd_source_is_copied`).
- `recipe_affect`: killed on both backbones (r9_heads.py; `test_recipe_affect_must_equal_the_frozen_partition`).
- `explicit_frozen_pin`: killed on both backbones (r9_heads.py; `test_explicit_frozen_partition_pin_rejected`).
- `caller_fit_hashes`: killed on both backbones (r9_heads.py; `test_writer_rejects_caller_hashes_unrelated_to_fit`).
- `feature_width`: killed on both backbones (r9_heads.py; `test_writer_rejects_wrong_feature_width`).
- `coefficient_width`: killed on both backbones (r9_heads.py; `test_writer_rejects_wrong_coefficient_width`).
- `fit_file_hash`: killed on both backbones (r9_heads.py; `test_writer_checks_feature_file_hash_from_fit`).
- `record_coefficients`: killed on both backbones (r9_heads.py; `test_n6_record_pins_all_coefficients_and_affect_provenance`).
- `record_affect_provenance`: killed on both backbones (r9_heads.py; `test_n6_record_pins_all_coefficients_and_affect_provenance`).
- `check_coefficients`: killed on both backbones (r9_heads.py; `test_new_backbone_selection_checks_affect_coefficients`).
- `check_affect_provenance`: killed on both backbones (r9_heads.py; `test_new_backbone_selection_checks_affect_provenance`).
- `load_data_backbone`: killed on both backbones (r9_common.py; `test_load_data_routes_the_requested_backbone`).
- `load_rows_backbone`: killed on both backbones (r9_common.py; `test_load_rows_routes_the_requested_backbone`).
- `writer_refit_rows`: killed on both backbones (r9_heads.py; `test_writer_refit_routes_the_requested_backbone`).
- `writer_refit_backbone`: killed on both backbones (r9_heads.py; `test_writer_refit_routes_the_requested_backbone`).
- `implicit_recipe_partitions`: killed on both backbones (r9_heads.py; `test_head_labels_without_explicit_pins_use_recipe`).
- `recipe_partition_role`: killed on both backbones (r9_heads.py; `test_recipe_pins_keep_partitions_distinct_from_frozen_affect`).
- `recipe_frozen_path`: killed on both backbones (r9_heads.py; `test_recipe_pins_reject_replaced_frozen_path`).
- `recipe_frozen_hash`: killed on both backbones (r9_heads.py; `test_recipe_pins_reject_changed_frozen_bytes`).
- `selection_recipe_arrays`: killed on both backbones (r9_heads.py; `test_new_backbone_selection_uses_its_own_arrays`).
- `full_draw`: killed on both backbones (r9_heads.py; `test_fit_heads_requires_full_draw`).
- `r9_rule_hash`: killed on both backbones (r9_common.py; `test_r9_rule_hash_cannot_be_bypassed`).
- `manifest_frozen_hash`: killed on both backbones (r9_common.py; `test_manifest_cannot_change_a_frozen_hash`).
- `told_before_labels`: killed on both backbones (r9_heads.py; `test_told_hash_is_checked_before_loading_labels`).
- `copied_selection`: killed on both backbones (r9_heads.py; `test_copied_style_selection_is_checked`).
- `exclusive_output`: killed on both backbones (r9_heads.py; `test_writer_exclusive_open_handles_a_racing_file`).
- `record_backbone`: killed on both backbones (r9_heads.py; `test_record_names_the_actual_backbone`).
- `pins_folder`: killed on both backbones (r9_common.py; `test_pins_files_live_in_round_folder`).
- `record_sha`: killed on both backbones (r9_heads.py; `test_selection_record_hash_is_pinned`).
- `manifest_record_pin`: killed on both backbones (r9_common.py; `test_new_manifest_requires_n6_record_hash`).
- `script_copies`: killed on both backbones (r9_check_heads.py; `test_check_script_reports_copies_and_unrefitted_arrays`).
- `script_unrefit`: killed on both backbones (r9_check_heads.py; `test_check_script_reports_copies_and_unrefitted_arrays`).
- `frozen_affect_source`: killed on both backbones (r9_heads.py; `test_recipe_affect_must_equal_the_frozen_partition`).
- `module_sha_coverage`: killed on both backbones (r9_common.py; `test_module_sha_coverage`).

CLIP replay command (run once in this fix round):

```bash
timeout 235s env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/CoSiR/bin/python src/test/20261202_r9_scope/r9_check_heads.py --backbone clip_b32
```

CLIP output (exit 0):

```text
[18:07:55] head affect done
[18:08:09] head image done
[18:08:28] head caption done
n6_posteriors/keys: pass
n6_posteriors/selection: pass
n6_posteriors/affect__img: pass
n6_posteriors/affect__txt: pass
n6_posteriors/image__img: pass
n6_posteriors/image__txt: pass
n6_posteriors/caption__img: pass
n6_posteriors/caption__txt: pass
step1_heads_style/keys: pass
step1_heads_style/selection: pass
step1_heads_style/style_csd__img: pass
step1_heads_style/style_csd__txt: pass
step1_heads_style/style_csd__img_src: pass
step1_heads_style/style_gram__img: pass
step1_heads_style/style_gram__txt: pass
step1_heads_style/style_gram__img_src: pass
step1_heads_style/style_rand__img: pass
step1_heads_style/style_rand__txt: pass
affect_identity: pass
clip_b32: 16 arrays, PASS (118.2s)
```

No command longer than four minutes is needed from the controller. No GPU or push.
