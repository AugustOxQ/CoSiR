# Handoff: read the K = 40 PercepT core results with the user (decision C), then decide the comparator and stage 2

## Brief
- Done: the five PercepT core arms reran with K = 40; every GPU step exits 0, all five arms are established. Score, ablation table, exact re-derivation and both reports are committed.
- State: comparator (image-side AMI) emotion P3 0.077, style P2 0.229, genre P3 0.436, against our best network 0.081 / 0.348 / 0.458: level on emotion, above on style and genre.
- Next: the user's reading (decision C), then options a to c on the comparator, then the stage 2 decision.
- Blocked: nothing.

> Written 2026-10-19 17:01 by "r10 run 3". Loop step reached: step 9 (verify and report) of the amended r10 spec
> (A3) is done; the next chat starts at step 1 (decision C). Direction log entry: none yet (the wrap-up adds it).

## 1. Read in this order

1. The user-read report, [2026-10-19_percept_core_k40.md](../../user_read/2026-10-19_percept_core_k40.md): what the
   user reads. Its advice is folded away.
2. The auto report's results brief,
   [2026-12-15_percept_core_k40.md](../../reports/auto/percept/2026-12-15_percept_core_k40.md): show it to the user
   without the collapsed block, ask for their reading, then open ours and say where they differ
   (`research-loop.md`, "Decision C").
3. The direction log's "Now" and its 2026-10-19 entries ([direction_log.md](../direction_log.md)).
4. The spec [2026-10-18-percept-core-design.md](../specs/2026-10-18-percept-core-design.md): §6 (the licensed claim
   names K = 40), §8 (no K sweep in this spec, so option c needs an amendment).
5. Only if needed: the round log `src/test/20261018_percept_core/20261018_percept_core_log.md`, "Results, run 3".

Skills: `grilling` and `domain-modeling` for the stage 2 decision, if it becomes a new idea with a grill.

## 2. Decided by the user, not to be reopened

- The spec's §4 (D1 to D15), approved 2026-10-18 23:34; A1 and A2, 2026-10-19 08:20.
- A3: a fixed K = 40, the 40 of 100 centers with the most fit pairs (K = 40: user, after seeing the recommendation,
  10:40; which 40: agent default, shown to the user).
- Descriptive mode: no bar; the comparator per aspect is the best established arm (D15).

## 3. Where things stand

| What | Number | Beside |
|---|---|---|
| Stage 0 (amended) | every check PASS (51 s), 30 fix tests | snapshot 0.122469 / 0.227419, row 2 exact |
| Stage 1 repeat; training vs run 2 | bitwise equal; `center_norms` and γ counts 9 of 9 equal | — |
| Arms established | 5 of 5; seed sd at most 0.007; dev AUC 0.850 to 0.963 | run 2: 1 of 5, sd up to 0.143 |
| Comparator, image-side AMI | P3 0.077, P2 0.229, P3 0.436 (P5 0.435 within 0.003) | our best default-input network 0.081 / 0.348 / 0.458 |
| P3 (Qwen) | 0.077 / 0.189 / 0.436 | Qwen default images clustered 0.070 / 0.222 / 0.449 |
| P2 (CLIP vector) | 0.066 / 0.229 / 0.400 | CLIP images clustered 0.065 / 0.300 / 0.412 |
| K = 40 vs run 2's 99 topics, 6 seeds, same training | image-side AMI moves by at most 0.02 (emotion at most 0.005) | — |
| Kept topics | 57 to 63% of fit pairs before the cut; 40 of 40 hold at least 1%; 40th vs 41st differ by 1 to 14 pairs | — |
| Re-derivation (C13) | 90 per-file values, 60 aggregates, 9 Stage 1 kept checks equal (max abs diff 0) | — |

Numbers are emotion / style / genre, development data only (the same 5,000 paintings chose our rows and the arms).
Our network minus the comparator, unrounded: +0.003, +0.120, +0.022.

## 4. Open points for the next step

- Key: the user's reading (decision C), given before ours.
- Key: options a, b, c of the results brief. a: PercepT core with K = 40 as the comparator, then stage 2. b: as a,
  plus the image encoders clustered directly as a second bar per aspect (0.070 / 0.300 / 0.449). c: a K check first
  (for example K = 20 and 60): an amendment, a short build and about 35 GPU minutes. Our recommendation, in the
  collapsed block: b.
- Key: the stage 2 decision of our model (direction log "Now": one shared trunk with emotion, style and genre heads,
  B2). Emotion is level with PercepT core today, so stage 2 must raise image-side emotion.
- Wrap-up after decision C: direction log entry and "Now"; the storage summary with `.scratch/pending_deletions.md`
  (build 6 worktrees and branches, the deferred head outputs, `sept_stage0/`); `wrapup-check`, `disk-check`,
  `sync.sh`; the Codex ledger rows of this run are written.

## 5. Code and pitfalls

- Earlier §5 notes hold: the queues write the main checkout's round folder; `pc_queue_status.txt` is appended across
  runs (read the block after the last `PC_gpu_SUBMIT`); one GPU queue only; pass
  `--ignore=src/test/20261018_percept_core/sept_stage0` to pytest in the main checkout.
- A rerun overwrites `outputs/stage1/`, `stage2/`, `pc_results.json`, `pc_tables.md` and `pc_ablation_table.md`;
  copy them first (`*_run1*`, `*_run2*` exist). Run 3's files have no copy yet.
- A K check (option c) needs K as a parameter: `pc_stage1.py` calls `pc_fixes.top_used(fit_counts, 40)` and several
  readers assert 40 (`pc_score`, `kept_topic_stats`).
- The C13 brief `rederive_run3/task.txt` takes the established list and the fit rows file (`fit_rows.npy`, written
  from `pc_fixes.pair_rows`; sha256 equal to the sidecars' `fit_rows_sha256`).

## 6. State at handoff

- Running: nothing. GPU free (queue ended 16:51); no lock.
- Uncommitted: nothing (after this chat's commit).
- On disk over 1 GB: unchanged (the deferred head outputs); this run added about 19 MB.
