# Reader fix, round 3 (one-sided affect steering on R1, fresh-seed test): run log

Folder date 20261121 is a sequence number. Times are Amsterdam local time (`TZ=Europe/Amsterdam date`). The binding
rule is `DECISION_RULE.md` in this folder; the spec is `docs/superpowers/specs/2026-10-06-round3-affect-gate-design.md`.

## Timeline

| Time | Event |
|---|---|
| 2026-10-06 19:10 | Round-3 tab started from `docs/superpowers/handoffs/2026-10-06-round3-affect-gate-handoff.md` |
| 2026-10-06 19:11 to 19:21 | The handoff's seven open points settled with the user one at a time: affect frozen by name (redundancy criterion recorded); seed 42 = two regression checks (R1 = round-1 R-c, AFF = the brainstorm's numbers), D12 recorded; GO = round 2's seven checks; AFF minus R1 = pre-registered secondary check; R1 in full on the test seeds, descriptive; sensitivity projection logged only; random-share control added as a descriptive ride-along; disclosure plus a stated prior |
| 2026-10-06 19:22 | Spec committed (728f5d7) and sent to the user; the user approved it ("approved, write the rule") |
| 2026-10-06 19:28 | Rule drafting started from the spec, with round 2's rule as the template |
| 2026-10-06 19:33 | Fresh Opus checker dispatched on the draft rule (against the spec, round 2's rule and final review, the handoff, the brainstorm's code and results) |
| 2026-10-06 19:49 | Checker report (`rule_check/opus_rule_check.md`): 2 blocking (B1: the condition-a open share 80.90006709098816% is a float32 mean, the exact share is 9,941/12,288, compare counts; B2: the wiring smoke test was placed before §5 and needs `sensitivity.json`, moved after §6.1 and smoke runs print no metric at all), 7 should-fix (bar comparator scope, the bundle call sequence with the smoke flag never passed to `model_inputs` or `load_readers`, input asserts on every seed, per-check NO-GO reading, random-share E not 12,288, re-derivation scope and x, crashed build), 16 nits. All 36 SHA-256s verified; every seed-42 number of §5 items 2 and 3 reproduced at full precision through the float32 path (float32 and float64 integer statistics identical in all 224 × 12,288 entries for both readers) |
| 2026-10-06 19:51 | All 25 findings applied by the controller; plan updated to match (wiring smoke after the sensitivity projection) |
