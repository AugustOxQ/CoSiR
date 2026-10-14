# Scoped re-review of fx-rr3 (r9-fx-rr3 @ 50662cf, base 17ae05c)

Verdict: PASS (no HIGH, MEDIUM; one LOW note).

## LOW-1 (test for a _fix1 pass with its own sensitivity file)
- Met. `test_r9_apply_rule.py:723` builds a `_fix1` pass with a different `sensitivity_H6_fix1.json`. The original's SHA is refused; the `_fix1` file's SHA gives a verdict that names that file and SHA.
- Test fails under my mutants (scratch copy, one at a time, each restored):
  - M1: `spath` always `sensitivity_<Hn>.json` (`run_r9_apply_rule.py:658`). Fails.
  - M2: `sens_sha` set to the original file's SHA after `load_sensitivity`. Fails.
  - M3: `agreement-sensitivity` guard (`:669`) replaced by `if False`. Fails.
  - Only this test failed each time (the older `..._bound_to_the_sensitivity_file_it_uses` test did not catch M1/M2).

## LOW-2 (margins_not_compared guard, `run_r9_apply_rule.py:536-538`)
(a) Can a legitimate agreement carry the key? No.
- Only writer of the key in all `.py` files: `rederive/rd9_compare.py:259`, inside `if selection` (derived `mode == 'selection'`, i.e. `rd9_phase2.py --mode selection`, the C13 replay of round 6 seed 42).
- Held and smoke modes of `rd9_compare` never write it (`rd9_test_compare.py:605, 643` assert this).
- `run_r9_preread.py:268-283` `smoke_agreement` writes no such key (fields phase, smoke, agreement, read, pass_file, held_pass_sha256, sensitivity_sha256, rederived_counts, items, compared, failed, what).
- No other writer of `rd9_<Hn>_agreement*.json` found in `src` (grep for the key and for `rederived_counts`: only compare, preread, tests, probes).
- Met.

(b) `run_r9_descriptive.py:96` also calls `load_agreement`. Refusing there is right: the descriptive step binds the same agreement file that the apply step accepted, so a file with the key would have been refused at apply already; the guard keeps the two in line. Descriptive tests pass (below). Met.

(c) Guard proven. Replacing the condition with `if False` (M4): only `test_agreement_that_carries_margins_not_compared_is_refused` (`test_r9_apply_rule.py:738`) fails. The test also checks the same file without the key gives GO. Met.

## N5
`git diff --stat 17ae05c 50662cf`: 4 files only: `.claude/20261014_log.md`, `fx_rr3_handback.md`, `run_r9_apply_rule.py`, `test_r9_apply_rule.py`. No closure module, not `run_r9_stage2.py`. Met. The source diff is the guard (3 lines) and 2 docstring lines; no other behaviour change.

## Tests
From F on the worktree, CPU: `test_r9_apply_rule.py test_r9_descriptive.py`: 228 passed (6 s). Scratch copy is clean after the mutants (`git diff --stat` empty).

## New findings
- LOW-A: the second half of RR3 LOW-2 (the comparer cross-checking the derived `mode` against the reported file's `mode`) was not done; the hand-back says it is out of the ticket. Not needed for the apply guard to work. Left open.
- LOW-B (note): the guard refuses by key presence only, so it depends on `rd9_compare` being the only writer; a hand-edited agreement without the key would still pass, but the SHA bindings already cover hand edits to the pass file.

Scratch copy (not removed): `.../scratchpad/rr4_wt` (a git worktree of the ticket repo; remove with `git worktree remove` if wanted).
