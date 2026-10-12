# Round 8: grouping (main line) and encoder screen: run log

Folder date 20261129 is a sequence number. Times are Amsterdam local time (`TZ=Europe/Amsterdam date`). The binding
rule is `DECISION_RULE.md` in this folder (SHA-256 50eb65addbaeade7cd85b12ea3153da5143fe0f515c1648b8133229088a9d50e,
equal to `RULE_SHA256` in `r8_common.py`). The build record is `.scratch/r8-grouping/notes/build_log.md` (local, outside
git); the run handoff is `docs/superpowers/handoffs/2026-10-12-r8-run-handoff.md`.

## Decisions in the run chat

| Time | Decision | Who decided | Why |
|---|---|---|---|
| 2026-10-12 23:28 | M4 view check: keep the rule as written (handoff choice a). V100, V67 and V33 drop if the real check fails; NB and FC go on. | agent default, night mode, no question asked | `run-mode` was night. Rule §9 says the run chat does not wait for the user, and M4 covers this outcome ("a failure drops V100 to V33, reported as a fair mechanic"). Choice b is a rule change after a check result (C7), which only the user makes. On the smoke, 1 of 256 paintings failed (min cosine 0.99983 against 0.9999). |

## Timeline

| Time | Event |
|---|---|
| 2026-10-12 23:27 | `r8 run` chat started from the run handoff; run mode night |
| 2026-10-12 23:28 | Checks: rule SHA-256 equal to `RULE_SHA256`; `results/stage0_regression.json` pass, written 21:44 under the same rule SHA; GPU free; load 1.1; real dry-runs print 13 GPU and 81 CPU steps |
| 2026-10-12 23:29 | Both queues started from the main checkout (8e556d6 code, `R8_RESULTS` unset); wrapper output in `results/logs/wrapper_{gpu,cpu}.out` |
