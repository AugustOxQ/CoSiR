# Handoff: score Table 3's VLM2Vec and PercepT (CLIP L/14) rows with the gated scorer and fill them in

## Brief
- Done: the MMArt table (4.4, own subsection), the first 4.2 paragraph on Table 2, the 4.1 sentences on the two PercepT versions, and a Table 3 scorer that passed both gates.
- State: ours leads Table 2 on all three aspects. Table 3's VLM2Vec and PercepT (CLIP L/14) rows are still empty.
- Next: run the VLM2Vec wait script now, then score PercepT (CLIP L/14) when r26 sends its test path and kept seeds.
- Blocked: the user's 11:00 meeting. Ask nothing about it until the user says it is over.

> Written 2026-11-11 08:15 by the "paper sec4 tables 2" tab, cut at about 200k tokens. Outside the research loop: no
> loop step, no direction log entry. PF means `/project/CoSiR/docs/paper/cvpr2027_aspect_spaces/`; REPO means
> `PF/CVPR2027-CoSiR-Paper/` (the Overleaf-synced LaTeX repo the co-authors see). T3 means
> `/project/CoSiR/src/test/20261111_table3_rows/`.

## 1. Read in this order

1. This handoff, all of it.
2. `T3/20261111_table3_rows_log.md`: what the scorer does, its commands, the reuse map and the two gates.
3. `PF/notes/trace/4_experiments.md`, sections "4.3 Table 3", "4.1 text, the two PercepT versions" and "4.2 text".
4. `REPO/Tables/main_retrieval.tex` (the two empty rows) and `REPO/Sections/4_experiments.tex` (4.1 and 4.2).
5. The earlier handoffs `2026-11-11-paper-sec4-tables-2-handoff.md` and `2026-11-11-paper-sec4-tables-handoff.md`,
   sections 2 and 5: their decisions and pitfalls stand.
6. Only when the user says the meeting is over: `PF/notes/after_meeting_2026-11-11.md` (project memory
   `paper-meeting-2026-11-11`).

Skills: none.

## 2. Decided by the user, not to be reopened

All of section 2 of the two earlier handoffs stands. New, all 2026-11-11, recorded in the PF README:

- **MMArt table:** in its own subsection after retrieval, 4.4 "Results on MMArt" (user, option b). The heading wording
  is the agent's default. Ablation and qualitative examples are now 4.5 and 4.6, and the ablation table is Table 5.
- **4.2 first paragraph:** from the user's three points. (a) Ours leads on emotion, above the teacher and all other
  methods. (b) TURTLE does not help. (c) Compare with PercepT on the same backbone. Wording: the agent's.
- **TURTLE sentence:** it states only the condition (default settings, no hyperparameter selection), not a cause
  (user, option i).
- **PercepT versions:** which version each row uses is stated in 4.1's "Compared methods", not in the caption (user).
  Wording: the agent's.
- **Table 2's VLM2Vec and BERTopic rows:** the user said this tab fills them. The "paper experiments 2" tab had already
  committed them (REPO `0e45e13`, 08:08). The values match r25's record exactly.

## 3. Where things stand

- REPO (pushed): `15e83e1` the MMArt table, `6cf981b` its subsection, `ca9a0b8` the 4.2 paragraph, `b48df06` the 4.1
  PercepT sentences. PF (pushed): README rows and traces up to `d2c91f5`.
- Table 2 at the class count: ours 0.102 / 0.411 / 0.510. Teacher 0.089 / 0.406 / 0.501. VLM2Vec 0.064 / 0.373 /
  0.459. BERTopic 0.038 / 0.105 / 0.260. The 4.2 sentence "at most 0.077 for the other methods" (CRL 0.077) still
  holds. Re-check it when Table 2's PercepT (CLIP L/14) row fills.
- **The scorer** (`T3`, main repo `e061d41c` and `18c3c682`) reuses the project's pick and test code. Both gates
  passed, run under `res-queue cpu --short`:
  - `gate-tpips`: pick `none+k10`, all six values exact (difference 0), 3.9 min.
  - `gate-percept`: r17's CLIP L/14 build, pick `center+k10`, max difference 2.8e-17, 5.0 min.
- **VLM2Vec row:** the test store is complete (`/data/SSD2/pre_extract/artelingo/vlm2vec_v2/<aspect>/`, final per r25).
  The validation store `.../vlm2vec_v2_val/` is being written by the GPU job `r25:extract_val`, which started 07:58.
  r25 expects it to end about 09:15 to 09:20. The pick unit is Qwen c*'s ranges, as for TPIPS (agent default; the unit
  source is written into the output).
- **PercepT (CLIP L/14) row:** the r26 tab sends the test-probability path and the kept seeds (seeds that did not
  collapse) to this tab's successor, about 09:00. Check messages. Validation probabilities:
  `/data/SSD2/cosir_r26/20261111_percept_original/outputs/val/k40/L1__s{0..4}.npz`. The pick unit is CLIP L/14 c*'s
  ranges, as r17's pick for this build.

## 4. Open points for the next step

Key:
1. **VLM2Vec row of Table 3.** First step: launch `T3/t3_wait_vlm2vec.sh` with Bash `run_in_background`. It waits for
   the six store `meta.json` files, then runs `t3_run.py vlm2vec` through `res-queue cpu --short` (about 4 to 8 min).
   Output `T3/out/vlm2vec.json`, log `T3/logs/vlm2vec.log`. Then read the cells from the JSON, round to three
   decimals, fill the row in `REPO/Tables/main_retrieval.tex`, re-check the bold, compile, look at the page, commit,
   push, update the trace, and send the PDF.
2. **PercepT (CLIP L/14) row of Table 3.** When r26's message arrives, run
   `t3_run.py percept-clip --seeds <kept> --test-dir <r26 test dir> [--test-file-pattern ...]` through
   `res-queue cpu --short`. Then fill the row as in item 1.
3. **4.1 PercepT note:** add that the CLIP L/14 version is a mean over the seeds that did not collapse, with r26's
   count. Show the sentence to the user first.

Minor:
4. Ablation clustering at the class count (user: later). See the earlier handoff, section 4, item 3.
5. Table 5 (ablation) polish and bold wait for its clustering cells. The other four removals and the intervals are
   for the user to decide later.

## 5. Code and pitfalls

- All pitfalls of the earlier handoffs' section 5 stand. Two chats share REPO and PF: the "paper experiments 2" tab
  edits `Tables/main_clustering.tex` and its own README rows. Check `git status` before you edit a shared file. Commit
  only your own paths (`git commit -- <paths>`) and push at once. No notes of ours in REPO. Use paper names, never
  internal ones (c* is `\method`). No semicolons between clauses. Write "test set", never "held-out".
- REPO has two uncommitted files that are not ours: `Sections/2_related_work.tex` and `main.tex`. Leave both.
- The scorer's code names the test split `held`. Printed text and JSON field names say "test".
- `t3_run.py` asserts the key sets of every store against the project's keys and checks the VLM2Vec `emb_sha256`.
  A "missing ..." assertion means a store is not complete yet.
- The scorer averages PercepT over the kept seeds one seed at a time, because `ns_ret.arm_ap` accepts only all five
  seeds or none. The unit tests show this equals the five-seed path when all five seeds are kept.
- r25's caveat for any later text on BERTopic: captions often name the emotion, so BERTopic's emotion cells see label
  words, and its topics are very unbalanced.

## 6. State at handoff

- Running: `r25:extract_val` (GPU, r25's job, started 07:58), then `r21:smoke` and `r26:extract_patch` in the GPU
  queue. Nothing of this tab runs: its VLM2Vec wait loop was stopped before the cut, and the next tab starts it again.
  Other tabs: "paper experiments 2", r25, r26, r21.
- Uncommitted: in REPO, `Sections/2_related_work.tex` and `main.tex` (not ours). In the main repo, files not ours
  (`docs/superpowers/constitution.md`, older untracked files).
- On disk over 1 GB: nothing from this work (`T3` is 80 KB).
