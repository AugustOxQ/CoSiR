# Review

## Root cause

Confirmed: with float32 `rand = nextafter(1, 0)`, `base + rand * total`
can round to the owning row's final CDF value. `searchsorted(..., right=True)`
then returns the first position after that row; for the final nonempty row this
is `indices.numel()` and causes an out-of-bounds gather.

## Fix review

- The weighted branch converts the global `searchsorted` result to a local CSR
  offset and clamps it to `[0, degree - 1]` before adding the row start back.
- The uniform branch is unchanged.
- `buddy_graph_smoothness_loss` and `buddy_contrastive_loss` both call the
  shared helper, so both weighted paths receive the fix.

## Verification

- Deterministic boundary test was red before the production change (CPU
  `IndexError: index 2 is out of bounds`) and green afterwards.
- Full buddy regularizer script passed, including the new 12,288-draw
  row-membership test.
- Exact two-epoch RedCaps-150k GPU smoke command completed and produced
  `epoch_0000.pt`, `epoch_0001.pt`, and the final phase-1 model artifact
  without a CUDA assertion.
- `git diff --check` and `python -m compileall` passed.

## External review

Dual reviewer invocation was attempted. The Claude wrapper cannot run under
root because its required `--dangerously-skip-permissions` flag is rejected;
the Codex wrapper did not return a report before final verification. The
findings above are the completed local review.
