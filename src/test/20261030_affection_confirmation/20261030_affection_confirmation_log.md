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
  (`cc2a102`) before any FI row exists. 22:13 ticket 04 merged (diff read). 22:20 ticket 03 merged (diff read).

### Readings settled in the build (agent defaults; reviewers agreed)

- **"Within 0.001 r of the best" (CRL version) and "more than 0.001 r" (photo wording)**: 0.001 in the score's own
  units of r. On Affection the two readings are the same condition; on ArtELingo the score mixes three r_a, so only
  this reading is defined (r11's convention).
- **r11's guard on Affection read-outs**: kept; with one aspect it cannot change a pick.
- **A seeded clustering comparator** (PercepT core, descriptive only) draws its own seeds, as r10's d_R; SC1 and SC3
  have unseeded comparators, so no verdict number depends on it.

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
