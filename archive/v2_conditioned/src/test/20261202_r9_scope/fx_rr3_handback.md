# fx-rr3 hand-back (branch r9-fx-rr3, from r9-scope @ 17ae05c)

Both items done. No closure module and not `run_r9_stage2.py` touched.

## LOW-1 (test only), done
`test_r9_apply_rule.py:723` `test_fix_pass_with_its_own_sensitivity_file_is_bound_to_that_file`: a `_fix1` pass with its own
`sensitivity_H6_fix1.json` (`make_sens("H6", se=0.2)`, other bytes than the original). Agreement carrying the original file's
SHA is refused ("another sensitivity file"); with the `_fix1` file's SHA the verdict is written and
`sensitivity_sha256` of the verdict is that SHA. Uses `write_agreement(..., over=...)`.
Mutation: `sens_sha` forced to the SHA of the original `sensitivity_<Hn>.json` right after `load_sensitivity` (:664): only the
new test failed (156 passed, 1 failed). File restored.

## LOW-2 (guard plus test), done
`run_r9_apply_rule.py:536-538` (in `load_agreement`, before the return): `if "margins_not_compared" in agr:  # GUARD: margins-compared`
-> `refuse(...)` (exit 4, message names the key). Docstring AGREEMENT block (about :71) names it. No test counts guard markers
(none found), so nothing else to update.
Test: `test_r9_apply_rule.py:738` `test_agreement_that_carries_margins_not_compared_is_refused`: with the key, refused (exit 4,
message has the key); after deleting the key from the same file, verdict GO. Other tests already pass without the key.
Mutation: guard condition replaced by `if False:`: only the new test failed. File restored.

## Tests (from F, `-q -p no:cacheprovider`, CPU)
- `test_r9_apply_rule.py`: 157 passed (was 155).
- `test_r9_apply_rule.py test_r9_preread.py rederive/rd9_test_compare.py`: 377 passed.
After the restores, `git diff` of `run_r9_apply_rule.py` shows only the guard and the docstring lines.

## Choices / not done
- Guard placed in `load_agreement` (where all other agreement-key guards are), not next to `agreement-sensitivity` in `apply`.
- The other half of RR3 LOW-2 (comparer cross-checking `mode` against the reported file) was not done: not in the ticket.
- Nothing to delete.

Commit: see `git log -1` on r9-fx-rr3 (hash given in the final message).
