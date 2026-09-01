# Experiment 15.2 review

- Codex review: no critical findings. It confirmed that the snapshot path uses
  raw features, the saved `combine_side`, Experiment 11.1's helper functions,
  explicit sample-ID alignment, and the unchanged template source path.
- Follow-up hardening addressed its provenance/legacy-snapshot concerns:
  matching configured graph store, unique feature IDs, and required
  `combine_side`.
- Claude wrapper review could not start because its wrapper passes
  `--dangerously-skip-permissions`, which Claude refuses as root. A direct
  read-only Claude invocation also could not authenticate in bare mode.

Verification: alignment unit test passed; `--epoch` without `--run-dir` was
rejected; a complete real RedCaps-150k frozen seed-1 epoch-99 checkpoint run
completed with finite, non-degenerate pulls. No training command was run.
