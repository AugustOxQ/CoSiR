# Handoff: r26 run, run PercepT on CLIP L/14 + ModernBERT-GoEmotions (smoke, full run, scoring, exports, reports)

## Brief
- Done: the r26 build is merged on branch r26-percept-original; both feature stores are extracted and checked.
- State: no PercepT result on CLIP L/14 yet; the Table 2 comparators are PercepT (Qwen3) 0.080 / 0.188 / 0.423.
- Next: run the smoke now; queue the full run for 19:00 if the smoke draws over 200 W (weekday rule).
- Blocked: nothing.

> Written 2026-11-11 09:30 by the "r26 percept original" tab (build chat). Loop step reached: r26 (a small loop opened from
> a paper chat) after the build, before step 8 (run). Direction log entry: none yet; this chat adds a short one.

**Agent defaults of the build, flagged** (each has a line in the round's `decisions.md`; the report lists them):
- Stage 2 on CLIP L/14 patch tokens (257 x 768, fp16) with attention pooling, as arm B4 on B/32 patches.
- Both topic cuts: K = 40 (fills the row) and strict Algorithm 1 (reported with its collapse check).
- ModernBERT caption vector: attention-mask mean of the last hidden state, special tokens included, max length 64
  (the RoBERTa pooling of `pc_roberta.py`); 0 captions truncated.
- No emotion map in Stage 1 (B4's PCA map does not apply to a 768-wide emotion encoder; r26's E is not centred).
- r17's scorer with three readings: `rclass` (k = class count), `r20` (k = 20), `own` (argmax).

## 1. Read in this order

All code is in the worktree `/project/CoSiR-r26` (branch `r26-percept-original`), round folder
`src/test/20261111_percept_original/` (prefix `po_`).
1. This handoff.
2. `po_paths.py` (paths, sizes, stores) and `decisions.md` (every result-shaping choice so far).
3. `/project/CoSiR/.scratch/r26-percept-original/review.md`: the final whole-branch review. Nothing blocking; minors
   M1, M2, M4 are open (section 5).
4. The build handoff `docs/superpowers/handoffs/2026-11-11-r26-percept-original-handoff.md`, sections 2 to 5 (the
   user's decisions, the protocol, the report rules).
5. `~/.claude/references/run-planning.md`, before the full run starts.

Skills: none.

## 2. Decided by the user, not to be reopened

- The run itself: build handoff section 2 (2026-11-11, paper README rows of that date).
- **Table 2 reads the k-means AMI at each aspect's class count** (emotion 9, style 27, genre 10; r17's `class_count`)
  (user, after seeing the recommendation; paper README, cosir-paper commit 71be7b4). This is the `rclass` reading.
  k = 20 (`r20`) stays for the supplementary.
- Table 2 shows AMI with three decimals and no intervals (user). The paper chat fills the row; do not edit the LaTeX
  repo. Do not call this run "faithful" (GLOSSARY row 8).

## 3. Where things stand

- Stores, both checked (`done.json`: gate ok, full count, not smoke):
  - ModernBERT `/data/SSD2/pre_extract/artelingo_full/modernbert_go_emotions/`: 308,723 x 768 fp32, parity max diff
    3.8e-5 (tolerance 1e-4), 0 truncated, 250 s. `po.modernbert(po.split())` loads 246,979 split captions, all finite.
  - CLIP L/14 patch `/data/SSD2/pre_extract/artelingo_full/clip_l14_patch/`: 61,402 x 257 x 768 fp16 (23 GB), 316 s.
    Parity gate over all rows: cosine of the rebuilt global vector with `clip_l14/img.npy` mean 0.99999999, min
    0.99999986 (thresholds 0.999 and 0.99); all tokens finite. `po.PatchStore().rows(...)` joins all 42,969 fit,
    6,152 dev and 12,281 test names; 256-row samples per split are finite (mean token norm about 8.2 to 8.5).
- Scorer check (arm B4's archived held outputs re-scored, `records/b4_scorer_check.json`): emotion r20 0.06994 and own
  0.07109 reproduce the stage 2 report (0.0699, 0.0711; report 2026-12-19).
- Table 2 comparators at k = 20 (archived, r17): PercepT (Qwen3, core settings) 0.080 / 0.188 / 0.423; CoSiR 0.107 /
  0.407 / 0.500. Review finding S2 (sent to "CoSiR paper experiments 2"): the Qwen3 row is core settings, the L/14
  row paper settings; the caption should say so.
- 10 unit tests pass (`test_po_readers.py`, `test_po_train.py`, `test_po_score.py`).

## 4. Open points for this chat

Run mode is night unless `run-mode` says day; take the defaults.
1. **Smoke** (short jobs, any time): `po_queue_run.sh --smoke <scratchpad>/po_run_smoke` in a background Bash. During
   Stage 1 or 2, read `nvidia-smi --query-gpu=power.draw`. Check the summary line `R26_RUN smoke rc=0`.
2. **Full run:** about 1 h 45 min of GPU (Stage 1 about 55 min, Stage 2 25 to 40 min, held 5 to 10 min), then CPU
   scoring and exports. Today is a weekday. Default: heavy (over 200 W) → `po_queue_run.sh --at 19:00`, ends about
   21:15. Light → start at once. DAS6 only if `cluster status` shows a reserved node (it did not at 05:30); the
   stores are 24 GB, so local stays the default. Write the choice as a decision line.
3. **Results to the paper tabs** when the files are final (check names with `ListAgents`):
   - "CoSiR paper sec4 tables 5" (it asked at about 09:20; "tables 3" and "tables 4" were cut at 08:45 and 09:15, so
     the tab may move again: send to the highest "CoSiR paper sec4 tables N" in `ListAgents`). It scores Table 3's
     PercepT (CLIP L/14) row with `T3/t3_run.py percept-clip`.
     Send only (a) the test directory `OUT/test/k40/` and pattern `L1__s{seed}.npz` (keys `prob`, `names`; rows in
     r17's held order), (b) the k40 seeds that did not collapse, with their count (`OUT/score/po_scores.json`). OUT is
     `/data/SSD2/cosir_r26/20261111_percept_original/outputs`. The validation files `OUT/val/k40/L1__s{seed}.npz`
     are written too; the tab said it needs nothing else.
   - "CoSiR paper experiments 2": `records/po_table.md`, the `rclass` row for Table 2 and `r20` for the
     supplementary, three decimals. Say how many seeds the k40 mean uses if any collapsed.
4. **Merge** `r26-percept-original` into main and push (`matt-chain.md`).
5. **Reports:** the run log `src/test/20261111_percept_original/20261111_percept_original_log.md`; a short auto report
   under `docs/reports/auto/percept/` (CoSiR's sequence date, next after the latest, now 2027-01-13; r25 may add one,
   so check at write time) with its `reports_sum.md` row and `docs-check`; a half-page user-read report; a short
   direction-log entry; `wrapup-check`; the storage summary. Strict cut: report its collapse count.
6. **End:** r26 is a small loop from a paper chat, so no next decide chat. Close with the chat-end routine.

## 5. Code and pitfalls

- Commands, from the round folder: `po_queue_run.sh` runs `po_run.py stage1`, `stage2`, `held` (GPU, each through
  `res-queue`), `score --record` and `po_export_val.py --test` (CPU). Logs go to `OUT/logs/<step>.log`. Outputs are
  idempotent: a verified output is skipped, so a rerun after a failure resumes.
- Stage 2 puts the fit tokens on the GPU as fp16 (17 GB): it needs the GPU free, which `res-queue` gives.
- Smoke outputs go to the scratchpad (`PO_OUTPUTS`); `po_run.py` refuses `--smoke` on the default OUT.
- Review minors, open: M1, the B4 check compares at three decimals and skips own; M2, nothing refuses retraining after
  held outputs exist (do not retrain after scoring); M4, `score_cut_seed` has no test. None blocks the run.
- The test outputs `OUT/held/L1__<cut>__s<seed>.npz` hold only `prob` (`verify_held` asserts it); the copy with names
  is `OUT/test/<cut>/L1__s<seed>.npz` (`po_export_val.py --test`, tested on a fake file only).
- The validation export checked its keys (`--check-keys`): r17's validation keys are exactly the 6,152 dev paintings.
- GPU work through `res-queue`; never stop another session's job. Delete nothing; list removals in
  `.scratch/pending_deletions.md` (the build's items are listed there).

## 6. State at handoff

- Running: nothing of this chat (both extractions ended at 09:28, exit 0). Other sessions' GPU jobs wait in
  `res-queue` (r21); the smoke's short jobs go before normal jobs that waited under 60 minutes.
- Uncommitted: nothing of this chat (`docs/superpowers/constitution.md` was modified before; not ours).
- On disk over 1 GB: the two stores (CLIP L/14 patch about 24 GB, ModernBERT 0.9 GB) under
  `/data/SSD2/pre_extract/artelingo_full/`, kept: the run reads them.
