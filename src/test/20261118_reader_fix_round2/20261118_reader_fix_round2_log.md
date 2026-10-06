# Reader fix, round 2 (follow-ups of the confidence-gated reader): run log

Folder date 20261118 is a sequence number. Times are Amsterdam local time. The binding rule is `DECISION_RULE.md` in
this folder; the spec is `docs/superpowers/specs/2026-10-06-reader-fix-round2-design.md`.

## Timeline

| Time | Event |
|---|---|
| 2026-10-06 15:43 | Round-2 tab started from `docs/superpowers/handoffs/2026-10-06-reader-fix-round2-handoff.md` |
| 2026-10-06 15:44 | Fresh Opus checker dispatched on the draft rule (against the spec, round 1's rule and code, the handoff's four ambiguities) |
| 2026-10-06 16:09 | Checker report (`rule_check/opus_rule_check.md`): 2 blocking (cross-fit ties depend on float rounding: compare integer sums; bank arrays are `pairs_{a,b}_{img,txt}`), 7 should-fix (re-derivation tolerances, numeric recipes, actions for code fixes and crashes, run_baselines exception in §9, top-k reading explicit, R3 bank SHA-256 records, D(k) disclosure), 9 nits; all 44 SHA-256s and every round-1 number verified; round 1's 224 cells reproduced bit for bit |
| 2026-10-06 16:13 | All 18 findings applied to the rule (Sonnet editor, controller reviewed the word diff; one sentence on float32 vs float64 made precise by the controller). Cell numbers checked: fused 116 = τ_2 (0, 2), 119 = τ_2 (0, 16); counterpart 58 = τ_1 (0, 0.5), 123 = τ_2 (0.5, 1) |
