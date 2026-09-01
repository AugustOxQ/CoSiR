# Experiment 15.3 Review

## Review scope

- Default-path preservation
- Repair-free semantic CSR construction
- CSR-aligned weighted-with-replacement sampling
- Family #1/#2 wiring and Family #3 boundary
- Missing-provenance failure behavior

## Findings

- Critical: none found in the implementation review.
- Warning: `buddy_refresh` is deliberately rejected when either type-aware flag is active, because refreshed KNN edges lack persisted provenance. This prevents the feature from silently reverting to untyped sampling and leaves `refresh_buddy_graph` unchanged.
- Info: `buddy_type_weights` accepts optional `img_only`, `txt_only`, `both`, and `repair` keys. Unspecified semantic types are 1.0; unspecified `repair` is 0.0.

## Verification

- `src/test/20260623_buddy_train_reg/test_buddy_reg.py` passed, including repair-only degree-zero, exact default sampling, uniform-frequency, skewed weighted-frequency, missing-provenance, and Family #1 weighted skip tests.
- `src/test/20260707_buddy_refresh/test_buddy_refresh.py` passed.
- `python -m py_compile src/metrics/regularizer.py src/hook/train_cosir.py` passed.
- Hydra parse dry run passed with `+loss.buddy_exclude_repair=true +loss.buddy_type_weights='{both: 2.0, img_only: 1.0, txt_only: 1.0}'`.
