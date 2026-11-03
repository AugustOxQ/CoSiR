# r15 round log (the MMArt test)

Spec: `docs/superpowers/specs/2026-11-02-mmart-test-design.md`. Build: branch `r15-mmart-test`, merged into main
(bc27203); tickets, final review and smoke in the run handoff
(`docs/superpowers/handoffs/2026-11-03-r15-run-handoff.md`). Run chat "r15 run", started 2026-11-03 01:13 (Amsterdam),
run mode night. Outputs: `/data/SSD2/cosir_r15/20261102_mmart_test/outputs/` (`OUT`). CPU only. Times are Amsterdam.

## Build (summary; details in the handoff)

- 8 tickets (6 of verdict code, each with an Opus review and a fix round); the final whole-branch review (Opus `xhigh`,
  three parts) approved with notes after its fix wave (1ce7712, 72ef2d2, bc27203) and re-review. 274 round tests pass.
- End-to-end smoke (`MT_SMOKE=1`, `smoke_build3/`): `QUEUE smoke OK rc=0 seconds=5671`; stage 1 at full size, stage 2 at
  40 and 100 draws.
- Review notes carried into the run: finding 3 (a failed teacher-aspect gate stops stage 2 with exit 2), finding 8 (the
  frozen and teacher retrieval rows are normalised twice; the C13 brief does the same), finding 11 (the unseen-artist
  subset has 15 paintings, so it gets no cells).

## Run

- 01:13 `uptime` load 0.9 (5 min 5.0), 53 GB free; GPU not used. `bash mt_run_cpu.sh first` started (stage 0, then
  stage 1; `MT_PARALLEL` 3), log `OUT/first_queue.log`.
- 01:18 C13: Codex job (`gpt-6.1-sol`, high; `codex-usage --room rederivation`: room, five-hour window 72% left) writes
  `mt_c13.py` (derive, compare) and `test_mt_c13.py` from the spec's rule, r14's `bs_c13.py` and the frozen r10 to r13
  protocol code, with no `mt_*` import and without the r14 scorer modules the round's code calls. The brief carries
  finding 8. It smokes a subset on `smoke_build3/`; the full derive runs under this chat after stage 2. Brief:
  `.scratch/r15-mmart-test/c13/task.txt`.
- 02:05 `QUEUE first OK rc=0 seconds=3120`. **Stage 0: 22 steps, none failed; every gate ok** (`records/status.json`):
  - (i) stores and near-copy file pass; cut sets 16,384 (test, 0.95) and 10,403 (far, 0.85), as the spec's counts.
  - (ii) c* exports: all four encoders pass, max difference 6.1e-5 (bar 1e-3).
  - (iii) scoring replay at full draws (2,000 clustering, 10,000 retrieval): max difference 0 (bar 1e-9);
    clustering 1,312 s.
  - (iv) mappers on CPU: all 20 pass (a), max 8.3e-7 (Qwen P3; bar 1e-5); (b) passes. No retrain (D11 fallback unused).
  - (v) validation teacher views pass; (vi) read-out replay: max difference 0, 1,484 s.
  - Exports: c* about 72 s per encoder, matched arms about 14 s.
- **Stage 1: 13 steps, none failed** (ranges about 2 min per encoder, picks 540 to 957 s, teacher pick 121 s). Records
  committed and pushed before any test number: 8dd3857 (the runner's printed command has a literal `\n` in its message,
  so the same paths were committed with a proper message).
- 02:07 stage 2 started (`bash mt_run_cpu.sh stage2`, log `OUT/stage2_queue.log`).
- 02:02 Codex C13, first pass: `mt_c13.py`, 43 tests pass; subset smoke on `smoke_build3/` (cut, SigLIP 2 and teacher
  ranges and picks, SigLIP 2's 15 cells at 40 and 100 draws): 2,079 items, 16 failed, max 2.6e-5. Two layout slips fixed
  in compare only. The 16: the three SigLIP 2 emotion clustering cells. Cause: the k-means thread count. r14's C13
  helper fits with one thread; the round fits with 8 (its runner sets `OMP_NUM_THREADS=8`). One thread changes 212 of
  16,384 assignments in seed 2, state 1, so the c* emotion AMI moves 1.3e-5; at 8 threads C13 matches the round
  exactly (max difference 0). Outcomes unchanged. **Decision (agent default):** C13 fits k-means at 8 threads, the
  round's setting; the spec fixes no thread count. The report states the thread sensitivity. 02:12 fix round sent by
  resume (`--kmeans-threads`, default 8, recorded; the record refuses other values).
- 02:21 fix round back: 51 tests pass; smoke 2,079 items, 0 failed, max difference 0 (SigLIP 2 emotion cells included).
- 02:22 full C13 derive started (`mt_c13.py derive --workers 8 --kmeans-threads 8`, log `c13/derive_full.log`); it reads
  no round result and does not need stage 2. Its code is committed as f093fb7 (the exact bytes it loaded).
- 02:28 Claude review of `mt_c13.py` (Sonnet `high` worker): no path lets a wrong verdict pass (independence, full scope,
  compare logic and protocol checked). One must-fix: any BELOW teacher cell makes compare fail (C13 writes `'no claim'`,
  the round `null`), a false FAIL. Should-fix: the matched-arm sidecars are not identity-checked; `derived.json` is not
  tied to its code. Not applicable here: a voided encoder would block the record (every gate passed). 02:31 second fix
  round by resume, compare-side code only (the running derive stays valid): claim vocabulary, 20 matched-sidecar
  items, `--derive-commit f093fb7` source check, unique SC4 identities.
- 02:35 second fix round back: only compare-side functions changed (an AST diff against f093fb7: `compare`,
  `record_refusal`, `main`, three new compare helpers). Its source check hashed the bare file, but the derive records a
  composite (frozen sources, this file, `mp_stores`); the brief's wording was wrong. The controller rebuilt the composite
  with the committed file's bytes (three lines) and fixed its test: 95 tests pass. The smoke's `derived.json` comes from
  an earlier draft of the file, so its source item fails by design; the subset also lacks 15 matched-arm input
  bindings (only SigLIP 2 ran). Committed as c00fda2.
- 02:44 `QUEUE stage2 OK rc=0 seconds=2189`: 8 scorer steps (clustering 637 to 825 s, retrieval 250 to 488 s) and
  `decide`, none failed; 60 cells at 2,000 and 10,000 draws, n = 16,384. The round's outcomes (pending C13): all four
  encoders hold; against the teacher 18 WIN, 4 LEVEL, 2 BELOW (emotion retrieval, Qwen and CLIP L/14); every D7 cell
  WIN. Cell table: `.scratch/r15-mmart-test/stage2_table.txt`.
- 03:47 full C13 derive done: `C13 derived cells=60 seconds=5177.9 scope=full`. Its `source_sha256` (8fbc1c…) is the
  composite with the f093fb7 bytes of `mt_c13.py` in both parts, so the derive ran exactly f093fb7.
- 03:48 compare (`--write-record --derive-commit f093fb7`): 6,697 items, 1 failed, max difference 0; the record was
  refused. The one failure is the source item. Cause: the compare rebuilt only the file part of the composite;
  the frozen part (`bs_c13.frozen_identity()`) also hashes this file as a loaded module, and compare took it from
  the current file (c00fda2). Fix, compare only: `frozen_identity_with(bytes)` rebuilds the frozen part with the
  committed bytes, and a test checks it equals the frozen identity on the current bytes; 96 tests pass. This also
  corrects the 02:35 note: the smoke's `derived.json` came from the same code (8fbc1c…); its source item failed for
  this same reason.
- 03:51 **`C13 PASS items=6697 failed=0 max_abs=0 scope=full`**: every re-derived number equals the round's exactly
  (the cut, 24 ranges, 9 picks with their tables, 60 cells, verdicts and claims). `records/c13.json` written; its
  digests match `OUT/stage2/{outcomes,cells_clu,cells_ret}.json`.
- 03:51 `QUEUE verdicts OK`; `records/verdicts.json` committed (5c323d8). **All four encoders hold, GO**; every
  gate and stage 1 status ok, so no void (D17). SC3 wins on emotion, style and genre for each encoder.
- 03:53 no cut to a second run chat: the context meter's note came before the chat's compaction (about 03:00), which
  brought the context well under 200k, so a cut would only add a handoff (agent default). Stage 3 started
  (`bash mt_run_cpu.sh stage3`, log `OUT/stage3_queue.log`); load 0.4, 55 GB available.
- 04:43 `QUEUE stage3 OK rc=0 seconds=3053`: all ten diagnostics ok (`OUT/diag/summary.json`). The unseen-artist subset
  has 15 paintings of 12 artists, so it gets no cells (fewer paintings than the 20 clusters; final review finding 11).
- 04:45 a display bug in `mt_diag.py`, found in the controller's read of `far.md`: the lookup of stage 2 cells
  (`test_cells`, and `item_shared_held`'s join) keyed cells without the task, so for SC4 the retrieval cell replaced
  the clustering one. Effect: in `far.md` the "d on test" column of the 12 SC4 clustering rows showed the retrieval
  value; `shared_held.md` lacked its 12 SC4 clustering rows. The cells (JSON and cell files) and every other item
  were right; `teacher_gain_share` keys by task; `percept_strongest` reads SC3 only. Fix: the task joins both keys;
  370 round tests pass. The two items were re-rendered from their saved JSON
  (`.scratch/r15-mmart-test/report/diag_rerender.py`, no scorer run): JSON identical, the four cell files the same
  bytes. The old files are kept in `OUT/diag/pre_fix_0450/`.
- Storage (04:46): `OUT` 12 GB (exports 9.0 GB, stage 0 2.9 GB, the rest under 1 MB); C13 folder 19 MB (cache
  16 MB); C13 smoke 16 MB; the build's smoke `smoke_build3/` 12 GB (already on `.scratch/pending_deletions.md`).
- 04:47 auto report draft by Codex (`gpt-6.1-sol`, medium; brief `.scratch/r15-mmart-test/report/codex_auto_report_brief.md`):
  15.4 KB, index check OK. It caught a wrong example in the brief: a LEVEL teacher cell does license "level with or
  better than" (spec §6); the brief said no claim.
- 04:57 claim review (Sonnet `high` worker; `.scratch/r15-mmart-test/report/claim_review.md`): about 490 numbers
  checked, none wrong; the rule check passes (claims, D7 flags, counts, the four licensed sentences word for word).
  1 must-fix (two clock times, changed in this log after the draft) and 9 should-fix (the results brief's link,
  deciding number, terms and options; two readings that could be misread; commits; code labels in prose; the
  seed-spread reading). The controller applied all ten; the report is 16.2 KB.
- 05:01 user-read report draft by Codex (`gpt-6.1-sol`, medium), 150 lines. Checks in parallel: cold read (Haiku)
  answered the four reader questions right, and listed undefined terms (PercepT, holds, GO, the encoders) and two
  dense paragraphs; fidelity check (Sonnet `high`) found no wrong number in about 40, two claims too strong (the
  "better than" scope, the far-subset style line), five missing caveats (Qwen-only shared reading, "about half",
  small emotion values and narrow teacher wins, labelled counts and 20 clusters, the encoder-selection wording).
  One fix wave by the controller: all applied, background facts added (project aim, PercepT, earlier results). The
  narrow-win caveat was also missing from the auto report and was added there. No re-check (the reference's rule).
- 05:05 run chat done: handoff `docs/superpowers/handoffs/2026-11-03-r16-decide-handoff.md`; the next loop's decide
  chat ("r16 decide") opens for decision C. Nothing runs.
