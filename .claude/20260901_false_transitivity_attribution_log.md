# `scripts/analyze_polysemy_bridges.py`

Experiment 15.2 adds a checkpoint source alongside the existing template source.
`--run-dir` and `--epoch` rebuild `comb_all` from a `condition_viz` snapshot using
the already-debugged geometry helpers, raw frozen features, and the snapshot's
`combine_side`. The reconstructed rows are explicitly remapped into the buddy
graph's sample-ID order; mismatched stores, duplicate IDs, and missing IDs fail
before the pull statistic is computed.

# `src/test/20260901_false_transitivity_attribution/test_comb_all_embedding_loader.py`

Adds focused regression coverage for reordered, duplicate, missing, and
feature-store-mismatched sample IDs. A real RedCaps-150k checkpoint smoke run
exercises the full reconstruction path because a compact synthetic
`FeatureManager`/checkpoint fixture is not available in the current test setup.
