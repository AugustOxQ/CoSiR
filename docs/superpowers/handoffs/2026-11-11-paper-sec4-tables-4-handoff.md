# Handoff: score Table 3's VLM2Vec and PercepT (CLIP L/14) rows with the gated scorer and fill them in

## Brief
- Done: Table 5's four removals at the class count (as signed changes), the 4.3 paragraph, "five random seeds" wording, pair hits out of the paper (user).
- State: Table 3's VLM2Vec and PercepT (CLIP L/14) rows are still empty. r25's validation store was 1 of 3 aspects at 08:42.
- Next: tell the r26 tab this tab's name, run the VLM2Vec wait script, then score PercepT (CLIP L/14) when r26 sends its files.
- Blocked: the user's 11:00 meeting. Ask nothing about it until the user says it is over.

> Written 2026-11-11 08:45 by the "paper sec4 tables 3" tab, cut at about 210k tokens. Outside the research loop: no
> loop step, no direction log entry. PF means `/project/CoSiR/docs/paper/cvpr2027_aspect_spaces/`; REPO means
> `PF/CVPR2027-CoSiR-Paper/` (the Overleaf-synced LaTeX repo the co-authors see). T3 means
> `/project/CoSiR/src/test/20261111_table3_rows/`.

## 1. Read in this order

1. This handoff, all of it.
2. `T3/20261111_table3_rows_log.md`: the scorer, its commands, the reuse map and the two gates (both passed).
3. `PF/notes/trace/4_experiments.md`: "4.3 Table 3", "4.3 text, first paragraph on Table 3", the two Table 5
   subsections, "4.1 text, the two PercepT versions" and "4.2 text".
4. `REPO/Tables/main_retrieval.tex` (the two empty rows) and `REPO/Sections/4_experiments.tex` (4.1 to 4.4).
5. `2026-11-11-paper-sec4-tables-3-handoff.md`, sections 2, 4 and 5, and through it the earlier handoffs' sections 2
   and 5: their decisions and pitfalls stand.
6. Only when the user says the meeting is over: `PF/notes/after_meeting_2026-11-11.md` (project memory
   `paper-meeting-2026-11-11`).

Skills: none.

## 2. Decided by the user, not to be reopened

All of section 2 of the earlier handoffs stands. New, all 2026-11-11, recorded in the PF README:

- **Seed wording:** "means over five random seeds" in captions, "five models trained with different random seeds" in
  4.1. Never "seeds 0 to 4" in the paper (user).
- **Table 5:** the four removals' clustering AMI at the class count is filled (user: yes, now). The removal rows show
  the signed change from CoSiR (user), with three decimals (agent default; a value that rounds to zero prints as
  0.000). Row names are the part's name without "w/o" (user). No scaling: the table is at `\footnotesize` (user asked
  for normal scale; `\small` was 26pt too wide). Bold and intervals: later (user).
- **4.3 first paragraph:** from the user's four points, three number pairs only. The user's point 1 ("stable across
  seeds") holds for genre only, so the text says that (agent, told to the user). The one-pass clause is the user's
  yes after seeing the recommendation. The reason sentence is the user's guess, worded as "a possible reason".
- **Pair hits stay out of the paper**, and the own-caption term is not pursued (user, after a discussion with the
  co-authors: the model is not trained for exact pair retrieval). No own-caption or late-fusion row in Table 3.
  Closes the 2026-11-05 flag. Project memory `pair-hit-limit` is updated.
- **TODO under 4.4** (the user's, in REPO): "discuss how to write about MMArt and what else we need to do". Leave it
  until the user raises it.

## 3. Where things stand

- REPO (pushed): `2ca85c6` seed wording, `ad45676` Table 5 cells, `47ee0f4` 4.3 paragraph, `89d7ac0` signed changes,
  `57dfd1f` row names and size, `82b59b7` the TODO under 4.4. PF (pushed) up to `4ba1d94`. Main repo: `8c2122f5` the
  ablation scorer and record (`src/test/20261111_ablation_classcount/`, gate exact against r17's c* rows).
- Table 5, change from CoSiR, clustering then retrieval (E / S / G): instructed graphs -0.033 / -0.186 / +0.010 and
  -0.021 / -0.122 / +0.001; content correction -0.016 / -0.068 / 0.000 and -0.007 / -0.058 / -0.001; graph-positive
  loss -0.029 / -0.140 / +0.008 and -0.018 / -0.098 / 0.000; relational distillation +0.001 / -0.002 / -0.071 and
  0.000 / -0.001 / -0.129.
- **VLM2Vec row:** the test store is complete. The validation store `/data/SSD2/pre_extract/artelingo/vlm2vec_v2_val/`
  is written by r25's GPU job `r25:extract_val` (started 07:58). At 08:42 only `emotion/meta.json` existed. r25
  expects the end at about 09:15 to 09:20.
- **PercepT (CLIP L/14) row:** r26 had only extraction logs at 08:20. Its `r26:extract_patch` was queued for about
  09:23, and training comes after, so this row is hours away. At 08:15 this tab asked the r26 tab to send its test
  path, file pattern and kept seeds to "CoSiR paper sec4 tables 3". That name is now wrong: see section 4, item 1.
- 4.2's "at most 0.077 for the other methods" still holds (CRL 0.077, VLM2Vec 0.064, BERTopic 0.038). Re-check it when
  Table 2's PercepT (CLIP L/14) row fills (the "paper experiments 2" tab edits Table 2).

## 4. Open points for the next step

Key:
1. **First, message the r26 tab** ("CoSiR r26 PercepT original") with this tab's own name (`ListAgents` prints it), so
   that r26 sends its test path, file pattern and kept seeds here. Check `ReadNotifications` too.
2. **VLM2Vec row of Table 3.** Launch `T3/t3_wait_vlm2vec.sh` with Bash `run_in_background`. It waits for the six
   store `meta.json` files, then runs `t3_run.py vlm2vec` through `res-queue cpu --short` (about 4 to 8 min). Output
   `T3/out/vlm2vec.json`, log `T3/logs/vlm2vec.log`. Read the cells from the JSON, round to three decimals, fill the
   row in `REPO/Tables/main_retrieval.tex`, re-check the bold, compile, look at the page, commit, push, update the
   trace, and send the PDF. If VLM2Vec beats CoSiR in any image cell, the 4.3 sentence "slightly above the teacher on
   all three aspects" still holds, but check the bold.
3. **PercepT (CLIP L/14) row of Table 3.** When r26's message arrives, run `t3_run.py percept-clip --seeds <kept>
   --test-dir <r26 test dir> [--test-file-pattern ...]` through `res-queue cpu --short`. Then fill the row as in
   item 2.
4. **4.1 PercepT note:** add that the CLIP L/14 version is a mean over the seeds that did not collapse, with r26's
   count. Show the sentence to the user first.

Minor:
5. Table 5 bold and intervals, and the other four removals: later (user).
6. The MMArt TODO under 4.4: wait for the user.

## 5. Code and pitfalls

- All pitfalls of the earlier handoffs' section 5 stand: two chats share REPO and PF (check `git status` before you
  edit a shared file, commit only your own paths with `git commit -- <paths>`, push at once); no notes of ours in REPO
  (the 4.4 TODO is the user's own); paper names only; no semicolons between clauses; "test set", never "held-out".
- In the main repo, `*.json`, `out/` and `logs/` are gitignored. Keep a record under the round's `records/` and add
  it with `git add -f`, as r25 does.
- Signed numbers widen a table: after any change to Table 5, check `main.log` for "Overfull".
- The scorer's code names the test split `held`. Printed text and JSON field names say "test".
- `t3_run.py` asserts every store's key set and the VLM2Vec `emb_sha256`. A "missing ..." assertion means a store is
  not complete yet.
- The r23 text cells for any later text: `/data/SSD2/cosir_r23/20261106_text_claim/stage3/part4_macro_cells_leak.json`
  (entry `plain` for Table 3; `no_leak` drops texts that name their own label; `cross_painting` drops the query's own
  painting from the gallery).

## 6. State at handoff

- Running: nothing of this tab. Its VLM2Vec wait was stopped at 08:42 before it scored anything. Others: `r25:extract_val`
  (GPU), r21's jobs, `r26:extract_patch` queued. Other tabs: "paper experiments 2", r25 report, r26, r21.
- Uncommitted: in REPO, `Sections/2_related_work.tex` and `main.tex` (not ours). In PF, `CBMI_IMP_CAMREADY (2).pdf`
  (the user's). In the main repo, files not ours, plus `src/test/20261111_ablation_classcount/out/` (gitignored, the
  record's working copy and a smoke file) and `out_smoke_removal/` (8 KB, a smoke file). Nothing to delete now.
- In `/project/claude-config`, the memory commit `916fb7f` also took another chat's index line (the pointer to
  `project_paper-meeting-2026-11-11.md`, whose file that chat has not committed).
- On disk over 1 GB: nothing from this work.
