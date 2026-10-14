# fx-desc-a28

Cause: the held runner starts H8–H10 with `mode=a28`, while
`r9_stats.pass_record` serializes their passes with `mode=held`. The descriptive
runner required literal equality, so every real A28 read stopped at that binding.

The runner now requires exactly started `a28` / pass `held` for real H8–H10.
Other modes retain exact equality, as do all eight other binding fields.
The audit also found H9/H10's H8 disclosure comparison used integer seed keys
from `a28_attempt_records` against string JSON keys from `loaded_parts`.
`loaded_parts` now returns integer seed keys after its existing exact seed-set
validation; record contents remain exact. Only `run_r9_descriptive.py` changed
in production code; `r9_descriptive.py` needed no change.

Read-only audit: `fx_desc_a28_binding_check.py` uses the main checkout's real
artifacts and the runner's `read_attempts`. Its JSON output records every binding
key, truncated unequal values, other comparisons, and source hashes.

| Binding key | H6 | H7 | H8 | H9 | H10 |
|---|---|---|---|---|---|
| mode | equal | equal | a28 / held | a28 / held | a28 / held |
| read | equal | equal | equal | equal | equal |
| backbone | equal | equal | equal | equal | equal |
| runner_sha256 | equal | equal | equal | equal | equal |
| module_sha256 | equal | equal | equal | equal | equal |
| input_sha256 | equal | equal | equal | equal | equal |
| pins_sha256 | equal | equal | equal | equal | equal |
| coef_sha256 | equal | equal | equal | equal | equal |
| episodes_sha256 | equal | equal | equal | equal | equal |

Other checked comparisons: H6/H7 each have 28, H8 has 45, H9/H10 each have 48.
Only H9/H10's raw H8 part-disclosure key comparison differs; both canonical
comparisons agree. Verdict/pass/agreement identities, hashes, seeds and counts;
array/sensitivity hashes; episode joins; pin hashes/declarations; A28 metadata,
set-aside disclosures and cross-backbone episode hashes all agree.

Validation: the pre-fix focused regression run gave 7 failed / 4 passed.
After both fixes, the whole `test_r9_descriptive.py` passed: **110 passed,
0 failed**, 7.95 s (`pytest -q -p no:cacheprovider --tb=short`, ticket Python/env).
The tests cover the four requested mode cases, H9/H10 and H6/H7 controls,
each other binding field, and disclosure keys/content.

Mutations in `fx_desc_a28_mutants.json` replace exact source in module memory
without editing the fixed file. All three were killed (pytest exit 1):

- Restore original mode equality: H8 started a28 / pass held fails at `mode`.
- Remove the read restriction: H6 started a28 / pass held reaches the forbidden
  array loader instead of refusing at `mode`.
- Restore JSON string part keys: H9's exact H8 disclosure comparison fails.

Not checked: full current input/backbone file-hash reconstruction; refitted heads;
rebuilt episodes/parts/anchor joins; numerical pass and score/direction
reproduction; paired anchor joins and language mask/kept-row checks; optional H5
and stage0b reproduction. The audit checks saved artifact bindings and pin
declarations. A full real-mode descriptive run and output write belong to the
controller after merge. No verdict was changed.
