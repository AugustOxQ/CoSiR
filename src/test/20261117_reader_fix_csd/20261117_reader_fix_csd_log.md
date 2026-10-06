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
| 2026-10-06 03:11 | Independent re-derivation of R-a and the AR check (own code in `rederive/`, implementation not read): 786 scalars equal at full precision; reader terms, picks, fused and counterpart arrays, bar-margin vectors bit-identical; B and the three B′ equal step 1's stored arrays. Top-two margins differ by at most 1.6e-7 when σ is computed in float64 instead of float32 (no pick changes); to be checked at R-c's thresholds |
| 2026-10-06 03:12 | Stream 2 done: `rb_build.py` (halves, heads, bank, train), `rb_eval.py`, `rb_features.py`, 9 unit tests pass, smoke chain clean (A1, A0, AR). Real `halves` and `heads --grouping csd` (41 s) run by the agent |
| 2026-10-06 03:14 | Real R-b chain launched by the main session (`rb_run_all.sh`: heads and banks, train, eval A1 and A0, eval AR, summary; at most 3 processes) |
| 2026-10-06 03:26 | R-b chain complete (heads and banks 8 min, train 2 min, eval 3 min). Cross-fitted heads close to the standard heads (other-half accuracy image 88.7 to 89.5 against 92.7, csd 83.1 to 83.8 against 85.1; no convergence warnings). Readers: A1 out-of-fold bank accuracy 62.9 / 62.7% (chance 25%), chosen C 1 and 100 (log losses flat across C ≥ 1). Shift report A1: max abs SMD 0.51 (caption S and C); top probability mean 0.62 on the bank, 0.56 on seed 42 |
| 2026-10-06 03:26 | R-b candidates (pending re-derivation), bar margins: Rb_argmax_A1 +0.181 [−0.086, +0.462] (B′), Rb_expected_A1 +0.098 [−0.059, +0.253] (counterpart), Rb_argmax_A0 +0.144 [−0.033, +0.327] (counterpart), Rb_expected_A0 +0.313 [+0.076, +0.550] (B′); gain statistics all with lower bounds above 0. None clears (+0.5). AR: Rb_expected_AR +0.317, picks to rand 27.3% |
| 2026-10-06 03:27 | R-c trigger met (all six base candidates have numbers). Parent = Rb_expected_A0 (largest bar margin, +0.3133 at full precision; next Ra_A0 +0.2299). `run_rc.py --parent Rb_expected_A0` launched; re-derivation agent sent Phase 2 (R-b, then R-c) |
