# Requirements

- Inspect all `combine_side` references and adjacent modality-selection branches in the eight user-named Python files.
- Verify both `img` and `txt` configurations against the invariant: the named side is combined; the opposite side is projected as `other`.
- Cover training, evaluation/retrieval, condition drift/embedding shift, and persisted per-sample artifacts.
- Provide exact `file:line` findings without modifying source files.
- Write the full audit to `docs/superpowers/scratch/2026-08-27_codex_combine_side_review.md`.
