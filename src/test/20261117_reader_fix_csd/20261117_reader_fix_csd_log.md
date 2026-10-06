# Reader fix with the CSD style grouping (plan (a)): run log

Folder date 20261117 is a sequence number. Times are Amsterdam local time. The binding rule is `DECISION_RULE.md` in
this folder; the review that shaped it is in `ars_review/`.

## Timeline

| Time | Event |
|---|---|
| 2026-10-06 02:28 | Run tab started from `docs/superpowers/handoffs/2026-10-06-reader-fix-run-handoff.md` |
| 2026-10-06 02:37 | `DECISION_RULE.md` drafted (letter items R1 to R5, S1 to S17, AR check, user decisions of 02:25) |
| 2026-10-06 02:38 | Fresh Opus checker dispatched on the rule against the letter's acceptance criteria |
| 2026-10-06 02:37 | User (going to sleep): run everything autonomously with subagents; node404 offered (not needed: CPU only) |
| 2026-10-06 02:50 | Checker report: 1 blocking finding (bank block order and seeds depend on the dict key names; keys and blocks now written out), 7 should-fix (B's affect-km made explicit, R-b expected's pick, order of work vs the 09:00 trigger, full SHA-256s, head-draw and CV details, S8 carried in the rule, re-derivation before the carry), 8 nits; all applied |
| 2026-10-06 02:51 | Commit f25c48f (review records, report, handoffs, index row, rule); rule sent to the user |
| 2026-10-06 02:53 | Implementation dispatched: stream 1 (Sonnet: `common.py`, `run_ra.py`, `run_rc.py`, `rc_core.py`, tests), stream 2 (Opus: R-b) |
| 2026-10-06 03:01 | Stream 1 done: 10 unit tests pass, smoke clean, regression check reproduces step 1 exactly (A0, A1, AR arrays, B, B′, input SHA-256s) |
| 2026-10-06 03:02 | Real `run_ra.py` launched by the main session; independent re-derivation agent (Opus, own code in `rederive/`) dispatched |
| 2026-10-06 03:05 | R-a done (pending re-derivation): σ_h affect 0.01108, image 0.04159, caption 0.02706, csd 0.11202, rand 0.00234. Ra_A1 bar margin −0.045 [−0.279, +0.195] (B′ 18.805), margin +0.252, gain statistic +0.850 [0.560, 1.139]; Ra_A0 bar +0.230 [+0.028, +0.437] (B′ 18.437), margin +0.299; Ra_AR bar +0.090. No R-a candidate clears the bar. AR check: R-a picks rand in 30.5% of (episode, condition) values against 20.3% for the step-1 arg-max reader (rand's tiny σ inflates its noise) |
