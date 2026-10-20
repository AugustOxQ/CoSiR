# Findings

- No held outputs or verdict exist in this worktree or the main round folder yet; full diagnostic execution must wait for the run.
- ct_score.paired_bootstrap validates both r20 and own for every comparator. Leiden needs only r20 (the three Leiden partitions), so extend validation with an optional comparator reading schema; keep its default and statistical implementation unchanged.
- D2 and PercepT production inference enforce the training device. Diagnostics must reconstruct the same saved modules on CPU, using their builders and hash checks, with a recorded CPU diagnostic inference flag.
- September 3b: mutual20, degree repair, component-medoid MST repair at alpha=.5; vote ties choose the nearest neighbor among winning labels. CPU graph math uses float32 instead of GPU half.
- Ticket 07/06 code is not present in this base. Consume their documented outputs directly; use the shared PercepT mapper builder and saved config.
- Focused tests: 51 passed (test_ct_diag.py + existing test_ct_score.py), 14.86 s.
- Synthetic 12,281×2,048 held / 13,400×2,048 fit benchmark: union graph 1.33 s; mutual graph + both repairs + three Leiden partitions 9.40 s; all three vote transfers with one neighbor search 6.11 s.
- Synthetic full-six-arm, five-seed Leiden bootstrap: 16 draws in 1.72 s at eight workers; simple extrapolation to 2,000 draws ≈3.6 min. These are component timings, not a complete-run guarantee.
- Diagnostics load every finite saved paper run, including collapsed ones; unavailable/nonfinite statuses and final eligibility stay visible in JSON. Fit and held checkpoint hashes must agree.
- Final validation: whole round 201 passed / 2 skipped (169.70 s); final diagnostics + scorer tests 52 passed (16.64 s). The additional test checks fit metrics and held-transfer metrics use their own populations.
- Public entry points: run/main orchestrate and write JSON/Markdown; diagnose_vectors scores one vector set; leiden_margin delegates to 03; FitInference reconstructs CPU saved-weight inference. Graph/vote/metric/load helpers are independently testable.
- Deviations: the scorer gains an optional r20-only comparator schema (default verdict behavior unchanged); CPU fit inference uses the shared builders instead of the production wrappers that enforce the training device; CPU mutual-graph math uses float32 instead of September's GPU half. All are recorded in output protocol/provenance. No open design questions; production timing awaits artifacts.
