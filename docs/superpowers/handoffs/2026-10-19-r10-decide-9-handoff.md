# Handoff: read the PercepT core arm results with the user (decision C), then decide the topic count and stage 2

## Brief
- Done: PercepT core ran: the changed stage 0 check passes, a repeated run is bitwise equal, all 24 GPU steps exit 0. Score, ablation table, exact re-derivation and both reports are committed.
- State: the topic count keeps 1, 2, 3 or 99 topics by seed. Only P2 passes all checks; as the comparator its image-side AMI is 0.036 / 0.162 / 0.217, against 0.081 / 0.348 / 0.458 for our best network.
- Next: the user's reading (decision C), then options a to c on the topic count, then the stage 2 decision.
- Blocked: nothing.

> Written 2026-10-19 10:17 by "r10 run 2". Loop step reached: step 9 (verify and report) of the amended r10 spec is done;
> the next chat starts at step 1 (decision C). Direction log entry: none yet (the wrap-up adds it).

## 1. Read in this order

1. The user-read report, [2026-10-19_percept_core_arms.md](../../user_read/2026-10-19_percept_core_arms.md): what
   the user reads. Its advice is folded away.
2. The auto report's results brief,
   [2026-12-14_percept_core_arms.md](../../reports/auto/percept/2026-12-14_percept_core_arms.md): show it to the user
   without the collapsed block, ask for their reading, then open ours and say where they differ
   (`research-loop.md`, "Decision C").
3. The direction log's "Now" and its 2026-10-19 entry ([direction_log.md](../direction_log.md)).
4. The spec [2026-10-18-percept-core-design.md](../specs/2026-10-18-percept-core-design.md): §3.2 (arms, the four
   fixes, Algorithm 1), §6 ("established", licensed claims), §8 (the γ risk row: no rerun with another γ under this
   spec, so option b needs an amendment).
5. Only if needed: the round log `src/test/20261018_percept_core/20261018_percept_core_log.md`, "Results, run 2".

Skills: `grilling` and `domain-modeling` only if option b leads to a new grill (a short one: the count rule).

## 2. Decided by the user, not to be reopened

- The spec's §4 (D1 to D15), approved 2026-10-18 23:34.
- A1 (stage 0: only the checks that repeat exactly) and A2 (one-thread k-means start), 2026-10-19 08:20.
- Descriptive mode: no bar; the comparator per aspect is the best established arm (D15).

## 3. Where things stand

| What | Number | Beside |
|---|---|---|
| Stage 0 (amended) | every check PASS (51 s) | snapshot 0.122469 / 0.227419, row 2 exact, 21 fix tests |
| Stage 1 repeat | bitwise equal | — |
| Topics kept (γ = 2), seeds 0 / 1 / 2 | CLIP + RoBERTa 99 / 2 / 3; Qwen + Qwen 1 / 99 / 99; Qwen + RoBERTa 1 / 1 / 1 | the largest norm gap always at an end |
| Comparator (P2, established), image-side AMI | 0.036 / 0.162 / 0.217 | our best default-input network 0.081 / 0.348 / 0.458 |
| P3, the 2 seeds with 99 topics | 0.072 / 0.194 / 0.446 | Qwen default images clustered 0.070 / 0.222 / 0.449 |
| P2 seed 0 (99 topics) | 0.063 / 0.250 / 0.412 | CLIP images clustered 0.065 / 0.300 / 0.412 |
| P4 vs P3 (same Qwen topics, 2 seeds) | 0.052 / 0.156 / 0.348 | 0.072 / 0.194 / 0.446 |
| Re-derivation (C13) | 90 per-file values, 24 aggregates equal (max abs diff 0) | — |

Numbers are emotion / style / genre, development data only (the same 5,000 paintings chose our rows and the arm).

## 4. Open points for the next step

- Key: the user's reading (decision C), given before ours.
- Key: options a, b, c of the results brief. a: keep P2 and add the image encoders clustered directly as a second
  bar. b: amend the topic count (a fixed count or a cut away from the ends) and rerun the five arms (a short build,
  about 20 GPU minutes). c: keep the rule, run seeds 3 to 9 (about 15 GPU minutes). Our recommendation, in the
  collapsed block: b.
- Key, if b: which count rule. Candidates: a fixed K (September used 60 pruned to 40; at 99 topics about 40 topics
  hold at least 1% of pairs), all 100 topics, or a cut only away from the ends. It changes a fix of the spec (§3.2
  fix 3) and the §8 row, so it is an amendment (A3) with "who decided".
- Key: the stage 2 decision of our model, per the direction log's "Now", after the comparator question.
- Wrap-up after decision C: direction log entry and "Now"; the storage summary with `.scratch/pending_deletions.md`
  (build 4 and 5 worktrees and ticket branches, the first run's rows); `wrapup-check`, `disk-check`, `sync.sh`.

## 5. Code and pitfalls

- The first run handoff's §5 still holds: the queues write the main checkout's round folder; `pc_queue_status.txt`
  is appended across runs (read the block after the last `PC_gpu_SUBMIT`); start one GPU queue only.
- A rerun overwrites `outputs/stage1/`, `stage2/`, `pc_results.json`, `pc_tables.md` and `pc_ablation_table.md`;
  copy them first, as this run did with `*_run1.*`.
- `pc_score` drops K < 2 seeds with a cause line; an arm needs three trained seeds to get a mean.
- The C13 brief (`rederive_run2/task.txt`) has no "established" filter in its comparator; add one if reused.

## 6. State at handoff

- Running: nothing. GPU free (queue ended 10:08); no lock.
- Uncommitted: nothing (after this chat's commit).
- On disk over 1 GB: unchanged (the deferred head outputs); this run added about 18 MB.
