# Handoff: decision C on the stage 2 model sheet, the wrap-up, then decision A for the next step

## Brief
- Done: stage 2 model sheet run; stages 0 to 3 all exit 0; auto and user-read reports committed.
- State: emotion reaches the instructed-image target (best 0.0962 vs 0.0929; PercepT core 0.0772). Style 0.3700 vs
  0.3952 and genre 0.4891 vs 0.4956 do not.
- Next: show the results brief without our reading, record the user's reading (C), wrap up, then step 2.
- Blocked: nothing.

> Written 2026-10-19 22:50 by "r10 run 4". Loop step reached: step 9 done (reports committed); the next chat starts
> at step 1 (decision C). Direction log entry (last): [2026-10-19 · PercepT core with K = 40 read](../direction_log.md).

## 1. Read in this order

1. The auto report's results brief:
   [2026-12-16_model_sheet](../../reports/auto/percept/2026-12-16_model_sheet.md). Show the user the brief without
   the collapsed block, ask for their reading, then open ours and say where the two differ (`research-loop.md`,
   "Decision C").
2. The user-read report: [2026-10-19_model_sheet](../../user_read/2026-10-19_model_sheet.md). Its advice is folded
   away the same way.
3. The spec's brief, outcomes (a) to (d):
   [stage 2 model sheet spec](../specs/2026-10-19-stage2-model-sheet-design.md).
4. The round log's results and timeline: `src/test/20261019_model_sheet/20261019_model_sheet_log.md`; the full sheet:
   `src/test/20261019_model_sheet/ms_tables_final.md`.
5. `~/.claude/rules/research-loop.md` ("Wrap-up and handoff") and `storage.md` before the wrap-up.

Skills: none for step 1. If step 2 picks a medium loop, the grill uses `grilling` and `domain-modeling`.

## 2. Decided by the user, not to be reopened

- The stage 2 spec, approved 2026-10-19 19:58: target D3 (one pass on the default image reaches the instructed Qwen
  image clustered directly), gate timing D4 (10-29 emotion at least +0.01 over PercepT core; the instructed bar judged
  with retrieval on 11-10), default input carries the claim (D5), image-side picks (D6), V, ARI, FMI beside AMI (D7).
- Descriptive mode; time-box 2026-10-23 so the claim test spec fits before 10-29 (spec §9).
- Run mode day (user, 2026-10-19 20:51).

## 3. Where things stand

Image-side AMI of each head's own label (development data, 5,000 evaluation paintings, k = 20):

| Row | Seeds | Emotion | Style | Genre |
|---|---:|---:|---:|---:|
| Target: instructed Qwen image | — | 0.0929 | 0.3952 | 0.4956 |
| PercepT core bar | 3 | 0.0772 | 0.2286 | 0.4359 |
| Stage 1 network | 3 | 0.0805 | 0.3485 | 0.4583 |
| A1 separate heads | 3 | 0.0894 | 0.3539 | 0.4574 |
| T1 trunk | 3 | 0.0930 | 0.3591 | 0.4603 |
| D2 trunk + direct distillation 0.5 | 3 | 0.0962 | 0.3697 | 0.4540 |
| D5 trunk + relational distillation 1 | 3 | 0.0884 | 0.3700 | 0.4891 |
| D1 direct distillation only | 1 | 0.0825 | 0.2751 | 0.4366 |
| I1 instructed inputs (base D17) | 1 | 0.0953 | 0.3835 | 0.4852 |

- §6 line (0.0872): reached by nine default-input rows. Outcomes: (a) no; (b) yes, emotion only; (c) no; (d) no.
- D17 fired on form b (T4, one seed); D17 = D5 on form b, seed 0 only. I1's base = D17 (agent default 17: seed-0
  three-aspect mean 0.3151 vs D5 0.3140).
- The K = 40 back-fill (V, ARI, FMI) is folded into both K = 40 reports (`03946d2`).
- Options in the brief: (1) a short second loop for style and genre (relational distillation alone, at 0.5 and 2,
  and with direct at 0.5); (2) the claim test spec now with D2 or T1; (3) both.

## 4. Open points for the next step

- **Key:** decision C, the user's reading of the sheet, given before ours.
- **Key:** decision A, the next step (options 1 to 3), its type and loop size.
- **Key, later:** the claim test row (our first choice D2; T1 is the simpler row).
- **Minor:** the storage decisions (section 6).

## 5. Code and pitfalls

- `ms_train.py`, `ms_rows.py` and `ms_score.py` hashes are in every head's config and stage 0's PASS. A new row type
  (relational distillation alone, a combined pull) changes the trainer: stage 0 must run again, and F5 asserts one
  code version per row, so old rows retrain or go to a new outputs folder.
- `outputs/` is gitignored; `ms_tables_final.md` and `ms_backfill_k40.md` are the committed copies.
- T1's emotion tie band (0.0010) is narrower than most rows' spread (0.0019 to 0.0046); several rows have one seed.
- Held data: none read. The claim test is the first held read of this line (`held-data.md`).

## 6. State at handoff

- Running: nothing (no GPU lock, no background job, no Codex job, no subagent).
- Uncommitted: nothing of this chat. Untracked files from earlier chats (`archive/v2_conditioned/...` results,
  three old `queue_status.txt` files) are unchanged.
- On disk over 1 GB, for the wrap-up's storage summary:
  - `src/test/20261019_model_sheet/outputs/`, 3.4 GB, system disk: the sheet's arrays (recommendation: keep while the
    sheet is in use).
  - `src/test/20261018_aspect_graph_sheet/outputs_fu/`, 1.2 GB: no longer needed after the back-fill fold (on
    `.scratch/pending_deletions.md`).
  - The build's smoke folder, 4.9 GB, and the other build rows: on `.scratch/pending_deletions.md`.
