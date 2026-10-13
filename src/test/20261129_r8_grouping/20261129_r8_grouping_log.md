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
| 2026-10-13 01:27 | Start the restarted run at once and let its CPU tail run past 11:00 (the 11:00 preference, `shared-resources.md`, added 2026-10-13) | agent default (run chat) | Estimate: the GPU queue ends about 09:15 (views about 10 min, FC3 stage 1 about 80 min with its waits, five FCMI runs about 5.5 h, encoders about 40 min); the CPU tail after the last GPU output (encoder parts, collect, placement, readers, scoring, C13 twice, NBC if triggered, descriptive) adds about 3 to 4.5 h, so the end falls about 12:30 to 14:00. A split over two nights would put the gate verdict a day later (D35 plans it for 14 October; held read by 16 October; cut-off 23 October). The GPU is free from about 09:15; if the CPU is needed earlier, the CPU queue can be stopped at a step boundary and restarted in the evening (finished steps are skipped). |

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

| 2026-10-13 01:25 | Stage 0 under the amended rule: pass (SC1a to SC1e), rule SHA-256 18b4e063… at start and end, inputs unchanged, 538 s |
| 2026-10-13 01:26 | Dry-runs: 13 GPU and 81 CPU steps; GPU free; load 5.0 (stage 0 just ended); run mode night |
| 2026-10-13 01:27 | Both queues restarted from the main checkout (fdaee39) |
| 2026-10-13 01:29 | Views step ok under amendment 1: check passed (256 paintings, min cosine 0.99983 ≥ 0.999); `views.npy` [36,518, 5, 512], 374 MB, in 3 min; V100, V67, V33 stay in the run |
| 2026-10-13 02:50 to 05:03 | FC3 stage 1 ok (SC2 pass, 3,330 s); FC1 ok (3,096 s); FC2 ok (3,133 s) |
| 2026-10-13 06:07 | FC4 (w 0.40) failed D21: killed at 3,600.9 s after 299 of 300 epochs (`results/fcmi/FC4/d21.json`, pass false, "construction-to-labels exceeded 60 minutes"). Dropped under the rule (M7 D21 test, M16); `r8_groupings.py part` records the drop, so the chain goes on. The queue logs it as "failed" (exit 247), not as a rule drop |
| 2026-10-13 06:15 | Cause read from the step logs: run time grows with the FCMI weight (w 0.05: 3,096 s, 10.3 s per epoch; 0.10: 3,133 s; 0.20: 3,330 s; 0.40: over 3,600 s, 9.7 s per epoch for epochs 0 to 18, then 11.5 to 12.6 s). During FC4 no heavy CPU step ran (the V parts waited on the turnstile until 06:07) and the 1-minute load was 2.6 at start and 2.0 at end. So FC5 (0.80) and FC6 (1.60) will likely fail D21 too. Decision: run on under the rule as written (agent default, night mode); raising D21's cap would be an amendment, which only the user makes, and would restart the whole run (every record carries the rule SHA-256). Phone note sent |
| 2026-10-13 07:11 and 08:11 | FC5 (w 0.80) and FC6 (w 1.60) failed D21 at 3,601 s (297 and 299 of 300 epochs done, median 12.3 and 12.2 s per epoch); dropped under the rule, as predicted at 06:15 |
| 2026-10-13 08:14 to 09:06 | Encoders: AL ok (08:14), DI ok (08:23), SG ok (09:06); `stageE_{AL,DI,SG}.json` pass true (ESC1) |
| 2026-10-13 09:06 | GPU queue done (exit 1: 10 ok; FC4, FC5, FC6 not ok, all D21 drops under the rule). GPU free from here; the CPU queue runs on (part_SG, then collect, placement, readers, scoring, C13) |
| 2026-10-13 11:20 | `gate_verdict.json` written (rule SHA-256 18b4e063…, smoke false). C13: `rederive/trigger_rederive.json` and `rederive/gate_rederive.json` both agree, 0 disagreements. NBC triggered (none of the six NB scorers passed; source R1_NB33, strength 0.33) and ran |
| 2026-10-13 11:20 | Gate verdict: no main pick, not promising (main line fails at the gate, D31); no screen pick. All 20 main scorers below AFF on SC3 (points −0.47 to −0.80, every upper bound below 0); FC4 to FC6 dropped (D21). Screen: best R1_DI +0.285 [−0.012, +0.592], SC4 style −63 against −49; R1_SG +0.059; AL below. Rule §8: no pick in either track, round 8 ends at the gate, AFF stays current best; no held read, no `HELD_RULE.md` |
| 2026-10-13 11:31 | CPU queue done: 81 of 81 steps ok (descriptive, label-free and harm steps included). Gate phone note sent 11:33 |

## For the report

- Name amendment 1 in the report's first lines and in the gate phone note as a change made after results (rule §7's
  flags do not list amendments; review of amendment 1).
- From the run handoff: D2 "also reported as a comparator" and ED18 "at most 3 h" are not enforced in code; 06 → 07
  and 04 → 07 run at full size for the first time in this run.
- FC4 dropped by D21 by about 7 s; run time grows with the FCMI weight (above), so FC5 and FC6 dropped too: the FC sweep
  holds only w 0.05 to 0.20 (FC1 to FC3). Each dropped run missed the cap by 1 to 3 epochs. The build's 54.5 min estimate came from the stage 1 weight only.
