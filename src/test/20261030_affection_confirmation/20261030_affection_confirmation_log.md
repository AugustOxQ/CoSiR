# r12 round log: frozen c* on Affection, CRL bars (build and run)

Spec: [2026-10-30-affection-confirmation-design.md](../../../docs/superpowers/specs/2026-10-30-affection-confirmation-design.md)
(approved 2026-10-30 21:17). Build handoff: `docs/superpowers/handoffs/2026-10-30-r12-build-handoff.md`.

## Build (2026-10-30, unattended, night mode)

Integration branch `r12-affection`; tickets in `.scratch/r12-affection/issues/` (outside git; the list is copied into
the run handoff).

### Build decisions (agent defaults; none changes a decision of spec §4)

- **Store merge without in-place writes.** FI is extracted alone into `qwen3vl_emb_2b_fi/`; `ac_store.py merge` writes
  the 85,007-image store into a new folder, checks gate 0a(i), then renames the old store to `qwen3vl_emb_2b_pre_fi/`
  and the merged one to `qwen3vl_emb_2b/`. Nothing is deleted; the old folder goes on `.scratch/pending_deletions.md`.
- **0a(ii), "the first 200 Affection validation keys including 100 FI images"**: read as the first 100 non-FI and the
  first 100 FI validation keys in sorted order (the sorted list puts FI, `you_aaai_affective`, last).
- **Committed records need `git add -f`.** The repo's `.gitignore` ignores `*.json`, so `stage0a_store_hashes.json`,
  `stage0b_ranges.json` and `stage1_picks.json` are force-added (r11 did the same for its ranges file). The queue's
  commit steps must use `git add -f`.

### Timeline

- 21:24 integration branch and worktree; 21:27 round skeleton (`conftest.py`, `ac_paths.py`).
- 21:30 wave 1: ticket 02 (Codex, `gpt-6.1-sol` high), 06 and 07 (Claude Sonnet `high` workers).
- 21:30 to 22:05 tickets 02, 06 and 07 built, each reviewed by an Opus `high` worker (pass, no must-fix), fixed and
  merged. Ticket 11 built by Codex and reviewed (pass after fixes); its fix round by Codex (89 tests), merged at 22:06.
- 22:06 build chat 2 opens (handoff `2026-10-30-r12-build-2-handoff.md`). Codex has no room (5-hour window 58% left,
  an implementer needs 80%), so every ticket of this wave goes to Claude workers.
- 22:08 wave 2: tickets 01, 04 (Sonnet `medium`) and 08 (Sonnet `high`, test-first); 03 and 05 (Sonnet `medium`)
  start after 01 merges, since 03 reuses 01's FI path code.
- 22:12 ticket 01 merged (diff read). `ac_store.py record` run on the real store and its hashes file committed
  (`cc2a102`) before any FI row exists. 22:13 ticket 04 merged (diff read). 22:17 ticket 03 merged (diff read).
- 22:26 ticket 05 merged (diff read). 22:28 ticket 08 merged after an Opus `high` review (pass, no must-fix; four
  should-fix test gaps for a fix round, listed in `.scratch/r12-affection/notes/08-review-fixes.md`). 433 fast tests
  pass on the integration branch.
- 22:31 the context meter passed 200k; cut to build chat 3. Found before the cut: ticket 12's planned smoke cannot run
  as written (the loaders refuse smoke views and exports, and full views need the run chat's heavy GPU steps); ticket
  13 (a smoke mode, verdict code) added to the tracker.
- 22:33 build chat 3 opens (handoff `2026-10-30-r12-build-3-handoff.md`). Codex still has no room (reset 23:52), so
  wave 3 goes to Claude workers: the 08 review fixes and ticket 13 (Sonnet `high`, test-first), ticket 09 (Sonnet
  `medium`).
- 22:41 ticket 09 built (diff read). 22:44 the 08 review fixes merged (`306aaf6`); all 128 `test_ac_stats.py` tests
  pass with the slow replays (217 s). 22:45 ticket 13 built. 22:46 ticket 12 (queue script, Sonnet `medium`) and
  22:48 ticket 10 (diagnostics and figure, Sonnet `medium`) built on 09's branch.
- 22:50 ticket 13's Opus `high` review: pass, no must-fix (four mutations each caught; full mode unchanged). 22:51
  tickets 13, 09, 10 and 12 merged; 515 fast tests pass. The queue's smoke FI extraction was moved to the scratch
  folder by the controller (`e86559b`). Cut to build chat 4 at the context meter (200k).

### Readings settled in the build (agent defaults; reviewers agreed)

- **"Within 0.001 r of the best" (CRL version) and "more than 0.001 r" (photo wording)**: 0.001 in the score's own
  units of r. On Affection the two readings are the same condition; on ArtELingo the score mixes three r_a, so only
  this reading is defined (r11's convention).
- **r11's guard on Affection read-outs**: kept; with one aspect it cannot change a pick.
- **A seeded clustering comparator** (PercepT core, descriptive only) draws its own seeds, as r10's d_R; SC1 and SC3
  have unseeded comparators, so no verdict number depends on it.

- **Smoke size: 2,100 keys per split** (ticket 12 said about 200): each cross-fit half must exceed the largest
  whitening size, 1,024 (r11's finding); `ac_vh.py`'s smoke default is 2,100 too. Smoke draws: 200 (clustering),
  500 (retrieval), passed to the runner and to vh alike.
- **Stage 0a**: the three replays of 0a(iii) run only when every other item is present and passes. In smoke, the
  replays, the FI merge, 0a(ii)(b) and the label counts (full-split numbers) are recorded "skipped in smoke".
- **Diagnostics, "every row not picked"**: all stage 1 validation rows with a `picked` flag, plus the test rows of the
  teacher wording and CRL versions not picked (from stages 2 and 3). Test numbers for read-outs not picked are not
  computed (the diag file says so).
- **Impressions (D13)**: the store holds 1,442 files of 822 photo IDs, not the spec's 827 (train plus test is 822;
  the annotation file has 823; the media folder 945). The figure uses the store's 822: 6 queries, 816 gallery. 427 IDs
  have a file without a headline overlay; 395 have only overlay files, so D13's fallback applies. The figure decides
  nothing.

### Checks run in the build (CPU)

- Labels: validation and test class and tie counts equal spec §3.4 exactly (ticket 02; re-derived by the reviewer).
- 0a(iii), r10 part: c*'s held emotion block through `ac_cluster` equals A_cand, A_T, d and the interval within 1e-9.
- 0a(iii), r11 stage 4 part: all seven arms' stage 4 scores within 1e-9 and the seven picks equal (23 minutes).
- 0a(i), the record: the old store's file, row and list hashes (63,199 images, 392,331 texts), committed in
  `stage0a_store_hashes.json` before FI is added.
- 0a(iv) (early run, scratch outputs): all 15 c* heads on ArtELingo validation re-inferred, max absolute difference
  6.1e-5 (bar 1e-3), 12 s on CPU.
- 0a(v) (early run, light GPU, scratch outputs): the new PercepT loader gives r11's held P3 arrays for all five seeds
  (sha256 equal; the gate is seed 0) and own-reading emotion AMI 0.08742922158584242 (equal), 7 s.
- 0a(ii) part (a) (early run, light GPU): the generalized extractor on the first 200 ArtELingo `emotion_focus` rows,
  min cosine 0.999817 (bar 0.999), max absolute difference 3.8e-3, 51 s with the model load. Part (b) needs the
  merged store and runs in the run chat.
- 0a(vi) (early run, light GPU): CLIP L/14 on 200 ArtELingo `clip_l14` rows, cosine 1.000000, max absolute
  difference 0.
- 64-image smokes of the three views sealed (instructed Qwen about 13 images/s on the smoke, CLIP L/14 faster).
  The early runs wrote to the scratchpad; the run chat's stage 0a reruns every item into the real outputs.
- 0a(vii) (early run, light GPU): CLIP-CRL validation vectors hash-equal to r11's for emotion, style and genre; the
  copied CLIP painting texts hash-equal; the Qwen text path on the first 200 store texts, min cosine 0.99952 (bar
  0.999), mean 0.99980. The six CRL text arrays take 12 s.
- 0a(iii), r11 stage 5 part (ticket 08's slow test, rerun by the controller): r11's per-arm and per-seed mAPs and
  SC10 to SC15's d and intervals within 1e-9, 250 s.
- Stage 4 (ticket 10, CPU, scratch outputs): the Impressions figure built from the real store and c*'s seed-0 emotion
  head; the PNG was looked at (three look-and-fix rounds by the worker; checked by the controller).

### Build chat 4 (2026-10-30 22:54 to 2026-10-31 00:10)

- 22:55 ticket 10's diff read by the controller (`ac_diag.py`, the two dispatch lines): no must-fix from the read.
- **End-to-end smoke** (`ac_queue.sh smoke`, 2,100 keys per split, draws 200 and 500), scratch folder of build chat 4.
  Step times (s): FI 13, parity a 27, smoke keys 6, `emotion_focus` 247, `emotion_focus_photo` 267, CLIP L/14 45,
  PercepT export 11 and check 10, c* export 23, re-infer 14, Impressions export 6, CRL texts 13 and check 20, parity c
  13, stage 0a 5, stage 0b 22, vh 0b 29, stage 1 379, vh 1 1,302. GPU steps about 12 minutes in all (instructed views
  about 8.5 images/s on 4,200 images each, under the lock, light).
- Contact fixes found by the smoke, each with a test: `record_copy` makes `OUTPUTS/records/` (`d3e054a`); stages 0b
  and 1 are sealed by the outputs file and the record copy, since a failed copy left a seal (`3f4fd5d`); vh accepts the
  runner's "skipped in smoke" label gates in smoke only, mutation-checked (`84f8e11`); vh reads r11's own sidecars for
  the frozen ArtELingo CLIP CRL arm, since the round writes CLIP painting texts for emotion only (`910e527`).
- Ticket 13's review nits fixed (Sonnet `high` worker, a test per fix, each failing without it; 528 tests with the
  slow replays pass on its branch) and merged (`931fae7`).
- **vh stage 1 mismatch, open**: for ArtELingo's fixed-pick arms vh scores the full read-out menu (c* and CLIP CRL),
  the runner only what §6 needs (CLIP CRL at R0 and `none+k10`; c* not at all). The 14 mismatches are all rows the
  runner does not hold; every shared row agrees. Fix in vh (compare the rule's rows; ideally also skip computing the
  rest, which is most of vh's 20 minutes per stage), next chat.
- **Final whole-branch review** (Opus `xhigh` worker at `910e527`): pass after fixes. Must-fix: `ac_diag.py:187`
  asserts six stage 2 arms, stage 2 writes ten, so diag always fails. Should-fix: no tests for `ac_diag.py` or
  `ac_queue.sh`; c* clustering in stages 0b and 2 uses re-normalized unit float32 rows, while r10, the 0a(iii) replay
  and vh use the stored float16 rows (equal on the smoke); C13 runs after a verdict file is written (reading: a verdict
  counts only once its `vh/<stage>.json` passes; the queue stops on a failed vh). 28 mutations, all caught. Traced the
  SC table, bootstraps, outcomes, verdict logic, stage 1 rule, seed ranges, held-data order, stage 0a and joins as
  matching the spec.
- Reviewer's independent re-derivation on the smoke (own code, numpy and scikit-learn only): r_clu, r_map, r_nDCG
  per seed equal (diff 0); teacher AMIs equal; clustering wording pick equal; teacher read-out picks equal, mAP within
  5e-8 (the score in units of r, about 0.001, then differs by about 5e-5).
- Fix wave (Codex `gpt-6.1-sol` high, branch `r12-fixwave`): the diag assert with a diag test, and c* clustering on
  its stored float16 rows (r10's protocol) in stage 0b, stage 2 and diag.

### Build chat 5 (2026-10-31 00:08 to 01:00)

- 00:12 the fix wave's diff read by the controller (verdict code): `ac_data.stored` returns the checked float16 export
  rows unchanged; stage 0b r_clu, stage 2 partitions (SC1, SC3) and diag cluster c* through it; r10's
  `r20_partition` and vh cast the same rows to float32, so all three paths match. 512 fast tests and the 26 slow and
  real-data tests pass. Merged into `r12-affection` (`5c93064`).
- 00:14 to 00:18 vh stage 1 fix (Codex `gpt-6.1-sol` high, by resume of the fix wave session, `90ff740`): for
  ArtELingo, vh no longer cross-fits c* (its pick is fixed) and scores CLIP CRL only at R0 and `none+k10`, the rows
  the runner writes (spec §6); the restricted cross-fit is bit-equal to the full menu on the same rows. Four tests;
  three fail with the full menu restored. 516 fast tests pass. Controller diff read: pass. Merged (`ba6ef6e`).
- **Fresh smoke** (`ac_queue.sh smoke`, 2,100 keys, draws 200 and 500; GPU-step outputs copied from build chat 4's
  smoke, everything from stage 0a on recomputed), 00:15 to 00:50: every step exit 0 (17 run, 12 skipped as sealed).
  Step times (s): FI 18, parity a 33, PercepT check 9, re-infer 13, CRL check 20, parity c 13, stage 0a 5, stage 0b
  22, vh 0b 19, stage 1 312, vh 1 443, stage 2 57, vh 2 513, stage 3 41, vh 3 517, diag 37, stage 4 8. vh agrees on
  every item: 0b 41, 1 598, 2 186, 3 188 (0 mismatches). vh 1 took 1,302 s before the fix.
- Stage 0b on stored float16 rows gives the same r_clu on the smoke as before (0.004444511160314857).
- The final reviewer's re-derivation script on the fresh smoke: stage 0b and 1 (`stage01.txt`): r_clu, r_map,
  r_nDCG per seed diff 0; teacher AMIs diff 0; both wording picks and the three read-out picks equal (mAP within
  5e-8).
- The same script, SC part (`sc.txt`): SC1 and SC2 d diff 0 (c* clustered on the stored float16 rows), SC8 d diff
  9.7e-9 (bar 1e-6); same outcomes and the same verdict line.
- **Scoped re-review** (Sonnet `high` worker, every change after `910e527`, at `ee39896`): pass, no must-fix or
  should-fix. Each fix's new test fails when that fix is reverted (scratch copies); vh imports no `ac_*` module; the
  restricted cross-fit is bit-equal; stages 2 and 3 of vh still compare picks strictly. Its note: the smoke gives the
  same r_clu with stored and re-normalized rows, so only the unit tests show the `stored` change. Report:
  `.scratch/r12-affection/build5/rereview_report.md`.
- `r12-affection` merged into main (`fd6a4b5`) and pushed; `ac_queue.sh full --dry-run` from main lists 26 steps.
  Worktrees, branches and smoke folders are on `.scratch/pending_deletions.md`.
