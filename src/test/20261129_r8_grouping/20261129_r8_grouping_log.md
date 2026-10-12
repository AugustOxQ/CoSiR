# Round 8: grouping (main line) and encoder screen: run log

Folder date 20261129 is a sequence number. Times are Amsterdam local time (`TZ=Europe/Amsterdam date`). The binding
rule is `DECISION_RULE.md` in this folder (SHA-256 50eb65addbaeade7cd85b12ea3153da5143fe0f515c1648b8133229088a9d50e,
equal to `RULE_SHA256` in `r8_common.py`) until amendment 1 (2026-10-13 01:13); from the restart, SHA-256
18b4e06300bf6dfa39edcda1a30d2ddf5ff4fbc335313452f182a42988089830. The build record is `.scratch/r8-grouping/notes/build_log.md` (local, outside
git); the run handoff is `docs/superpowers/handoffs/2026-10-12-r8-run-handoff.md`.

## Decisions in the run chat

| Time | Decision | Who decided | Why |
|---|---|---|---|
| 2026-10-12 23:28 | (Superseded at 01:13 by amendment 1.) M4 view check: keep the rule as written (handoff choice a). V100, V67 and V33 drop if the real check fails; NB and FC go on. | agent default, night mode, no question asked | `run-mode` was night. Rule §9 says the run chat does not wait for the user, and M4 covers this outcome ("a failure drops V100 to V33, reported as a fair mechanic"). Choice b is a rule change after a check result (C7), which only the user makes. On the smoke, 1 of 256 paintings failed (min cosine 0.99983 against 0.9999). |
| 2026-10-13 01:13 | Amendment 1: M4's view-check threshold 0.9999 → 0.999, and a restart of the whole run from stage 0 under the new rule SHA-256 (18b4e063…). Recorded in spec A §4 "Amendments" and the rule header. | lowering: user; 0.999: user, after seeing the recommendation (flagged) | The user, in the `r8 build 5` chat (relayed to this chat at 01:13, confirmed by the user here): "I guess we lower the threshold, because this looks like noise to me." 255 of 256 cosines are ≥ 0.9999 and the minimum is 0.99983, where a real fault would give about 0.9 or lower on most paintings; 0.999 is a round value with a margin, not fitted to 0.99983. |

## Timeline

| Time | Event |
|---|---|
| 2026-10-12 23:27 | `r8 run` chat started from the run handoff; run mode night |
| 2026-10-12 23:28 | Checks: rule SHA-256 equal to `RULE_SHA256`; `results/stage0_regression.json` pass, written 21:44 under the same rule SHA; GPU free; load 1.1; real dry-runs print 13 GPU and 81 CPU steps |
| 2026-10-12 23:29 | Both queues started from the main checkout (8e556d6 code, `R8_RESULTS` unset); wrapper output in `results/logs/wrapper_{gpu,cpu}.out` |
| 2026-10-12 23:30 | Views step: the real M4 view check failed (256 paintings, min cosine 0.99983 against 0.9999, `/data/SSD2/pre_extract/artelingo/r8_clip_views/meta.json`, allow_tf32 False). No views written; V100, V67, V33 dropped under the rule (the CPU queue skipped their heads). FC3 stage 1 queued behind the load wait |
| 2026-10-13 00:52 | FC3 stage 1 ok (the old run; its record carries the old rule hash); FC1 started |
| 2026-10-13 01:13 | User decision relayed from the `r8 build 5` chat: amend M4 and restart; confirmed by the user in this chat |
| 2026-10-13 01:14 | Both queues stopped (queue processes, the FC1 tree, the CPU lock waiters); GPU and the three locks free. The old run's FC1 was partial, so it has no record |
| 2026-10-13 01:15 | Old real outputs moved, not deleted: `results/` → `results_m4_09999/` (2.1 GB), `/data/SSD2/pre_extract/artelingo/r8_clip_views` → `r8_clip_views_m4_09999` (2.4 MB, the write-once `meta.json`), `/data/SSD2/cosir_r8/fcmi/FC1` → `FC1_m4_09999` (3.2 GB, partial), `FC3` → `FC3_m4_09999` (4.8 GB); no encoder output existed. Listed on `.scratch/pending_deletions.md` |
| 2026-10-13 01:16 | Amendment edits: `DECISION_RULE.md` M4 and header; `RULE_SHA256` in `r8_common.py`; `CHECK_THRESHOLD` in `r8_views.py`; spec A amendment row. Checked in the archived record: 1 of 256 cosines below 0.9999, 0 below 0.999. `test_r8_common.py` 28 passed, 1 skipped; `test_r8_stage0.py` 10 passed. Stage 0 restarted under the new rule; fresh Opus review of the amendment dispatched (C7) |
| 2026-10-13 01:20 | Fresh Opus review of amendment 1 (C7): approve with minors, no rule edit needed (`.scratch/r8-grouping/notes/review_amendment1.md`). Minors fixed outside the rule: spec A §5 assumption line, row 1 wording, the `r8_common.py` comment, this log's header |

## For the report

- Name amendment 1 in the report's first lines and in the gate phone note as a change made after results (rule §7's
  flags do not list amendments; review of amendment 1).
- From the run handoff: D2 "also reported as a comparator" and ED18 "at most 3 h" are not enforced in code; 06 → 07
  and 04 → 07 run at full size for the first time in this run.
