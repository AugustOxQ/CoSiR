# Handoff: r13 run: start the full queue on main tonight, watch it to the end, verify, then write the auto report and the user-read report

## Brief
- Done: the build is finished and merged into main (`269edae`). The final review (Opus `xhigh`) found no wrong number; one fix wave fixed 5 findings and 13 older notes; the re-review passed; smoke 2 passed (C13: 298 items equal).
- State: ready to run. Frozen c* is below its teacher on Affection test (AMI −0.0282, mAP −0.0190); r13 fine-tunes it on the train photos.
- Next: `bash src/test/20261031_photo_finetune/pf_queue.sh full` from `/project/CoSiR` (about 5 to 6 h, ends about 01:00), then verify, then the two reports.
- Blocked: nothing.

> Written 2026-11-01 18:45 by "r13 build 5". Loop step reached: step 7 done (build merged and pushed), step 8 next.
> Direction log entry: [2026-10-31](../direction_log.md). Run mode: night.

## 1. Read in this order

1. The spec, [2026-10-31-photo-finetune-design.md](../specs/2026-10-31-photo-finetune-design.md): §3.4 to §3.6 (measurement,
   stages and gates, diagnostics), §6 (the decision rule), §9 (deliverables; the report by 2026-11-03 23:00).
2. The round log `src/test/20261031_photo_finetune/20261031_photo_finetune_log.md`: smoke 1, the final review, the
   fix wave and smoke 2, with the step timings.
3. The final review's notes for the run and the report: `.scratch/r13-photo-finetune/notes/final_review.md` (rows
   B3, B4, B5, C-N1) and `notes/fix_wave_rereview.md` (N1). §5 below has what each means for the run.
4. `~/.claude/references/loop-chats.md` ("Unattended chats", "How to cut"), and before the reports the rules
   `report-kinds.md`, `report-writing.md`, `user-read-reports.md` and `~/.claude/templates/results_brief.md`.
5. For the agent defaults: §4 below (builds 1 to 5 together).

Skills: none.

## 2. Decided by the user, not to be reopened

The spec's §4 (D1 to D14), approved 2026-11-01 15:33. Nothing new was asked in build chats 1 to 5.

## 3. Where things stand

- **Code:** main at `269edae`, pushed; round folder `src/test/20261031_photo_finetune/` (prefix `pf_`), clean. Fast
  suite: 362 passed (`-m 'not slow'`). `pf_queue.sh full --dry-run` on main: 11 steps, start checks pass.
- **Queue, full mode:** extract_images, extract_texts, stage0a, stage1, commit (sheet and near-copy record, pushed),
  check_records, stage2, commit (stage 2 record, pushed), stage3, c13_derive, c13_compare, diag. GPU steps run under
  `flock -o -w 3600 -E 75 /tmp/gpu0.lock`; stage 3, C13 and diag on CPU. Status in `OUTPUTS/queue_status.txt`, logs
  in `OUTPUTS/logs/queue_<step>.log`, timings in `OUTPUTS/queue_seconds.txt`.
- **Outputs root:** `/data/SSD2/cosir_r13/20261031_photo_finetune/outputs/` (does not exist yet; SSD2 has 1.8 TB
  free). New views under `/data/SSD2/pre_extract/affection_local/qwen3vl_emb_2b_aspect_prompts/`
  (`emotion_focus_train`, `emotion_focus_txt`); default store `/data/SSD2/pre_extract/affection_local/qwen3vl_emb_2b/`.
- **Timing (smoke-based, with model load; not checked at full size):** images 72,255 at about 15.5 per second, about
  65 to 78 min; texts about 474,000 at about 210 per second, about 35 to 40 min; stage 0a about 10 min; stage 1 (up to
  9 sweep rows) and stage 2 (five runs) light GPU, about 1.2 h by the spec; stage 3, C13 and diag on CPU, about 2 h by
  the spec (diag about 6 to 10 min). Total about 5 to 6 h: started by 19:30 tonight (Sunday), it ends about 00:30 to
  01:30, well before 11:00, and the heavy extraction runs on a weekend evening (`shared-resources.md`). Estimate the
  end again at the start.
- **Smoke 2** (18:13 to 18:37, after the fix wave): every step exit 0; gates 10 of 10; C13 compare 298 items, 0
  mismatches; a stage 3 rerun reran compare and diag and gave the same verdict byte for byte.
- **Codex:** 5h 30% left at 18:12 (reset 20:36), weekly 45%. The ledger is up to date
  (`/project/claude-config/codex/jobs.md`).
- **GPU check at 18:13:** driver 595.99.02 answers, module and library match, GPU idle, no lock. Load 5.8, 54 GB
  RAM available.

## 4. Open points for the next step

None for the user. Agent defaults set in builds 1 to 5, to record in the run log and the report:

- Data: smoke mode subsets train only (`PF_SMOKE=N`, the first N sorted train keys). Train keys come from the
  index's text rows (equal to r12's rule, re-derived). The within-train near-copy count (10,114) counts photos;
  1,057 of them are also removed as near-copies of held photos.
- Gate (i) image parity uses the first 100 sorted validation keys plus the first 100 FI validation keys. Gate (ii)'s
  "its r12 mAP" for frozen c* is r12's five-seed c* `center+k10` mAP (0.28584595252536343); the seed-0 value is
  reported. The smoke runs stage 0a at the full draws.
- Training: sweep rows start from c* seed 0 with training seed 0; the five final runs start from c* seed s with
  training seed 100 + s. Row 8's random start uses c*'s factory in c*'s original aspect order under the run's
  training seed, the emotion view kept; it still records cstar_seed = s. A finished run (run.json ok or failed) is
  reused, never retrained; to retrain, move its folder aside.
- Records: the queue commits the sheet and the near-copy record in one commit after stage 1, and the stage 2 record
  after stage 2, each pushed. A rerun skips sealed steps and fails on a changed committed record. Stage 1 is sealed
  by the sheet and `stage1/rows.json`. Stage 2 refuses a sheet whose picked sweep run is missing, failed, of other
  settings or of another fit set. Stage 3 checks r', the read-out and the final runs' seeds, start hashes and fit
  set before any test export.
- Comparators' clustering partitions and per-query AP are read from r12's outputs (hash- and row-checked; their
  points and AMIs are checked against r12's records).
- Diagnostics: comparators' R0 and nDCG from r12's `diag/diag.json`; R0 mAP through the read-out `none+k0`; the
  untied verdict as r12's method; photo minus c* with a shared seed draw, each side its own read-out; gain share =
  (arm − plain) / (teacher − plain) on points; held spread from the test per-seed values; CLIP L/14 is r12's `clip`
  row and PercepT core comes from r12's diag.

## 5. Code and pitfalls

- **The verdict is final only after C13** (B3): `stage3/verdict.json` is written before C13 derive and compare, as in
  r12. Read it as the verdict only when `rederive/rederive.json` is `"ok": true` and newer. A failed compare is
  reported as a result, not fixed into agreement.
- **Check the text join on the full view** (B4): no gate checks the csv_row to explanation join of the new text view;
  on the smoke view it is right. Once `extract_texts` is done (CPU, while stage 0a runs), run
  `/root/miniconda3/envs/CoSiR/bin/python -I /project/CoSiR/.scratch/r13-photo-finetune/review_final/B/join_check.py
  /data/SSD2/pre_extract/affection_local/qwen3vl_emb_2b_aspect_prompts/emotion_focus_txt
  /data/SSD2/pre_extract/affection_local/qwen3vl_emb_2b` and compare with `join_check_out.txt` beside it (matched
  cosine about 0.6 against about 0.4 shifted, matched wins in 92 to 96% of rows per source). If the join is wrong,
  the run's text graph inputs are wrong: stop and note it at the top of the handoff.
- **Training is not deterministic across devices** (C-N1): the same row trained on GPU and on CPU differed by about
  0.6 r_clu in its sweep score, about the 0.5 r tie margin. Say so where the report gives the sweep sheet and the
  pick. r_mAP is small (0.00101 in smoke 1), so scores in r units look large.
- **Diag reruns on every queue call** (N1): compare always runs and rewrites rederive.json, so diag (6 to 10 min)
  runs again on any later queue call. No wrong number.
- **Reruns:** `gates.json` with any `"ok": false` is not a seal; the queue reruns all of stage 0a. To redo one part,
  `pf_run.py 0a --parts <part>`, then the queue again. The smoke-only flags (`--rows`, `--max-epochs`, the draw
  flags) are refused in full mode.
- **Full mode needs main** with no tracked changes in the round folder; the queue's `git push` pushes all of local
  main (other sessions' commits too). A failed commit after `git add` blocks the next start until it is committed or
  unstaged by hand (B5).
- **Stage 0a holds the GPU lock** for about 8 minutes of mostly CPU work (B5); no action.
- Builds 1 to 4's §5 still hold (keys `<source>/<image_name>`, never join by name or store position; the 4,422
  ignored train explanations stay out; the auto report's sequence date is 2027-01-01).
- **Storage:** `.scratch/pending_deletions.md` lists the build's leftovers (worktrees 2.7 GB, fix-wave scratch 10 GB,
  the two smoke folders); the next wrap-up decides them with the user.

## 6. State at handoff

- Running: nothing (no subagents, no Codex job, no GPU lock).
- Uncommitted: nothing of this chat's in the project (`.scratch/` is outside git by design).
- On disk over 1 GB: `.scratch/r13-photo-finetune/codex/fw1_scratch/` (9.4 GB) and the worktrees (2.7 GB), both on
  the pending-deletions list.
