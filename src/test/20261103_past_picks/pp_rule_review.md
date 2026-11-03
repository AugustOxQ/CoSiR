# Rule review: r19 spec (c* past our picks: three encoders, SemArt's new artists, the cost table)

> Reviewed 2026-11-03, finished 19:58, by a fresh Opus subagent (constitution C7). Spec:
> `docs/superpowers/specs/2026-11-03-past-picks-design.md` @ eeaf3b2 (clean). Read: `spec_core.md`, `spec_experiment.md`,
> `held-data.md`, the constitution, the r14 and r15 specs (inherited parts), their rule reviews, the facts brief, the
> SemArt check (log, `sc_check.py`, `sc_check.json`, `sc_nearcopy.parquet`), and the stored outputs named below
> (listings, sidecars, index and id files; no feature array loaded). Read-only checks on CPU; no GPU. This file is the
> only edit.

**Verdict: pass after fixes** (3 must-fix, 9 should-fix, 13 nit). The design is sound, every path it relies on exists
on disk, and nothing selects on test data. Three text errors would still make gates fail or let verdicts stand on
unchecked code: the SemArt set's unknown-author rule, the tolerance of the DINOv2 matched-arm rerun, and the export
gate's coverage. None of the fixes reopens D1 to D15.

## Must-fix

**M1. §3.3 "The SemArt set" (and gate (vi)): the unknown-author rule in the text is not the rule that gives 11,162.**
- The text says "its author is not 'UNKNOWN MASTER'". `sc_check.py` drops every author that starts with "UNKNOWN":
  20 author forms, among them unknown cabinetmakers and icon painters.
- Recomputed from `sc_nearcopy.parquet`: the code's rule gives 11,162 paintings by 2,717 artists. "Starts with
  UNKNOWN MASTER" gives 11,205 by 2,723 (the facts brief's hand filter hit this). "Equals UNKNOWN MASTER" gives
  11,657 by 2,737.
- A builder who follows the text fails gate (vi), and that voids every SemArt verdict and SC14.
- Fix: "and its author does not start with 'UNKNOWN' (20 author forms: unknown masters, icon painters, cabinetmakers
  and others; `sc_check.py`'s rule)". Add to gate (vi): "and its type counts equal `sc_check.json`'s
  `new_artist_types`". The second part is cheap and catches a wrong label join.

**M2. §3.5 (v) and D23: the DINOv2 rerun tolerance fails for numerical reasons about half the time.**
- The gate asks the rerun's held readings to "lie within the stored sample M5's range over seeds 0 to 4".
- In 10-26's `reported_stage3.json`, stored seed 0 is the lowest of the five seeds in both emotion readings: own
  0.07175 (min 0.07175), 20 clusters 0.06655 (min 0.06655). Genre at 20 clusters is 0.42854 against a min of 0.42809.
- The sample M5 trained on CUDA (sidecar `device: cuda`), so a node402 rerun is not bit-exact. Any rerun that lands
  below the stored seed 0 fails the gate, and the failure voids DINOv2's ArtELingo, MMArt and SemArt verdicts.
- Fix in (v) and D23: "its held readings (own and 20 clusters, each aspect) differ from the stored seed-0 readings by
  at most the stored arm's maximum minus minimum over seeds 0 to 4 of that reading; the differences are recorded".

**M3. §3.5 (iii) and its row in the voids table: the export gate misses the checkpoints that SemArt uses.**
- SemArt needs new exports of the four existing encoders' c*: r10's Qwen checkpoints and r14's X-c*. Gate (iii)
  replays only r14's CLIP L/14. r15's gate (ii) replayed both checkpoint families; this round's does not.
- The table maps a failed (iii) to "every verdict and teacher cell of the three new encoders" only. If the export
  code is broken, the four existing encoders' SemArt verdicts and SC14 still stand, on the same broken code.
- Fix (iii): "The export code, given r10's Qwen c* and r14's CLIP L/14 checkpoints, reproduces their stored ArtELingo
  held exports (and CLIP L/14's Affection test exports) within 1e-3, all five seeds."
- Fix the table: split the row into "(ii): every verdict and teacher cell of the three new encoders" and "(iii): the
  same, plus every SemArt verdict and SC14 cell of an encoder whose checkpoint family failed".

## Should-fix

**S1. §6 "Teacher cells" and D17: a D7 cell at LEVEL has no stated claim.** "LEVEL or WIN in any other cell" leaves
a D7 cell at LEVEL open. Use r15's wording: "a D7 cell at WIN gives 'better than'; a D7 cell at LEVEL, and any other
cell at LEVEL or WIN, give 'level with or better than'; BELOW gives no claim and keeps its row." In D17, "on MMArt as
r15" names no cell, because this round scores no MMArt teacher cell of the four existing encoders. Write: "WIN
licenses 'better than' only in the SemArt genre mAP cell of Qwen, SigLIP 2, SigLIP 1 and CLIP L/14 (r15's D7 cells)".

**S2. §3.4 "Cost" and SC15: the measurement is not defined well enough to implement or to re-derive (C13).**
- Repeats: say how they combine. "c*'s time in repeat r = the X pass in repeat r + the three networks in repeat r; its
  figure is the maximum over r. The teacher's time in repeat r = the sum of its three passes in repeat r; its figure
  is the minimum over r." Say which figures fill <a> and <b> in the licensed sentence.
- Resize: "decoded and resized beforehand" conflicts with "preprocessing to tensors" being timed. One shared resize
  for all encoders would also cut Qwen's dynamic image-token count. Write: "decoded to RGB at file size beforehand;
  timed: each encoder's own resize (Qwen: the extractor's pre-resize to at most 512 image tokens; the others: their
  processor's resize and crop), the rest of its processor, transfer, forward and pooling".
- Loader: "the extractor's loader with 8 workers" names no real loader. The 86.3 ms reference of gate (viii) comes
  from the MMArt prep's extractor, which loads on one prefetch thread (`ThreadPoolExecutor(1)` in r14's
  `encode_rows`). Only `scripts/extract_features.py` uses 8 threads. Use the MMArt prep's loader for the end-to-end
  form, so (viii) compares like with like.
- End to end: say whether the teacher's three passes share one decode ("each image decoded once per arm" implies it).
- Raw data: save per-batch timings per pass and repeat (the C13 input). A repeat during which another compute process
  shows in `nvidia-smi` is discarded and run again, and the log records it.

**S3. SC15: the comparison is unfair to a teacher that shares its image encoding.** Qwen's instruction is the system
prompt, before the image tokens, so the language model's work cannot be shared across the three instructions. The
vision tower's output can be shared, and a reviewer may ask about it. Add to "Not licensed": "cost against a teacher
that computes its image encoding once for the three instructions". Add a §3.6 row: the vision tower's share of one
Qwen pass (one extra timing). The claim very likely survives either way, since one c* pass is far below three Qwen
passes.

**S4. §3.5 (i): three of its checks have no row in the voids table, and two new stores have no extraction check.**
- Missing rows. A failed ArtELingo parity of X voids X's MMArt, Affection and SemArt verdicts and teacher cells. A
  failed SemArt CLIP B/32 parity voids every SemArt verdict and SC14. A failed Affection CLIP L/14 parity voids the
  new three's photo verdicts and their photo teacher cells.
- Missing checks. The new SemArt SigLIP 1 and CLIP L/14 stores have no parity check this round. Add "the first 100
  ArtELingo rows of the stored SigLIP 1 and CLIP L/14 stores". Also add "the first 100 SemArt keys of the stored
  SemArt Qwen default and SigLIP 2 stores", which are reused as inputs.
- Say that (i) runs through the code that writes the new stores, not the old extractor.

**S5. §3.5 (iv), the SemArt-path replay: SemArt's row order breaks a 1e-9 replay.** SemArt rows are ordered by the
sha256 of the key. k-means starts and bootstrap rows depend on row order, so fed in that order the path cannot give
r14's values. Fix: "fed ArtELingo validation and held genre inputs in r14's order (the order is an argument), with
r14's picks and the matched arm's stored held probabilities, gives r14's CLIP L/14 genre values and intervals of SC1
to SC3 and SC6". The SC6 part also covers SC14's teacher path and LEVEL unit, and the probabilities exercise the
matched-arm reading code.

**S6. §3.5 (vii): it misses two code paths of stage 1, and its row in the table voids too much.**
- Stage 1 also runs r12's rule (photo models' and new frozen images' Affection picks) and the seed-range code; (vii)
  replays only r11's rule.
- The table's "every mAP cell, so every verdict except SC15" voids paths that do not use the rule. The existing
  encoders' SemArt picks are reused, and the photo picks use r12's rule.
- Fix (vii): "fed r14's stored validation exports, the stage 1 code reproduces r14's `stage1_siglip1.json` and
  `stage1_clip_l14.json` (ranges; ArtELingo picks, the SigLIP 1 frozen pick included; Affection photo and frozen
  picks) and r15's `stage1_clip_l14.json` (MMArt ranges and picks), within 1e-9". The records exist.
- Fix the table: "(vii): the new three's verdicts and teacher cells that use a failed range or pick; AMI teacher claims
  stand; the existing encoders' SemArt verdicts stand".

**S7. §3.5 (v): the new full-scale DINOv2 mappers have no CPU replay.** They train on node402 (CUDA), and r15's
reading applies them on CPU to MMArt and SemArt. Add: "the new M5 mappers on CPU give their node402 held probabilities
within 1e-5 (as r15 (iv)(a))".

**S8. §3.1: the paintings prior does not add up.** With "SC1 and SC2 WIN in all six cells 0.8", holds plus qualified
must be 0.8 and fails 0.2. The spec gives 0.85, 0.85 and 0.75, with fails 0.15, 0.15 and 0.25. Write "SC1 and SC2 WIN
in all six cells 0.85, 0.85, 0.75". "MMArt as ArtELingo, 0.05 lower" should say which numbers move: holds 0.05 lower,
fails 0.05 higher.

**S9. §9 Budget: "after 19:00 on weekdays and ending by 11:00" breaks the machine's quiet hours.** The quiet hours are
weekdays 10:00 to 19:00. Write "ending by 10:00".

## Nit

1. D18 and §3.4: "the matched arm's genre mapper" and "genre outputs" do not exist as such. Each matched arm has one
   mapper per seed (`prob` [12281, 40]; partitions `percept__s<k>__{own,r20}`, with no aspect), scored against each
   aspect's labels. Write "the matched arm's mapper (one per seed), its readings scored against SemArt's type".
2. §6 "Licensed (item 16)": "R6-5" is an internal code name in a paper sentence. Write "… with an earlier version of
   our model".
3. §6 "Verdicts": D4 gives qualified results a limit sentence too. Add "A qualified verdict also gets a limit
   sentence naming its cells."
4. Brief Plan 3 and §6: the md5 step drops 3 paintings (3 groups of 2, checked by hash), so 11,159 are scored. Write
   "11,162 paintings (11,159 after byte duplicates)".
5. §3.3: "SemArt's other 10,220 paintings: not touched" conflicts with §9, which extracts all 21,382 per store. Write
   "extracted with the rest; never scored".
6. §3.3 timing row: "the first 2,000 keys in sha256 order" of which set? Name the 16,384-painting cut set.
7. Voids table, (viii): "until the cause is traced and fixed" has no end. Add "if no fix passes (viii) before the
   time-box, SC15 is inconclusive".
8. §5: the main extractor accepts `--image-only` (required for DINOv2) only for `artelingo_full` and `artelingo28`,
   and the round extractors (`bs_extract`, `mp_extract`) have no CLIP B/32, B/16 or DINOv2 view. The build adds
   these paths; name it so the tickets cover it.
9. §9 disk "about 6 GB": 15 × 0.38 + 15 × 0.18 = 8.4 GB before MMArt and SemArt exports. r14 per encoder: c* 1.9
   GB, photo 0.14 GB, Affection exports 0.26 GB. Say about 10 to 15 GB; SSD2 has 1.8 TB free, so there is no risk.
10. §3.3 evaluation order lists "cross-fit halves" for SemArt; nothing is picked on SemArt, so no cross-fit runs
    there. Drop the words.
11. Brief "Outcomes": say that SemArt's genre mAP cell can license "better than" for the four existing encoders (D17).
12. (ii): write "per head", and add "its validation export within 1e-3" (it feeds ranges, picks and SemArt's maps).
13. Stage 1 gate: "before any SemArt number" should read "before any SemArt score"; stage 0's count is a number.

## Checked and fine

- **Stores on disk.** ArtELingo DINOv2 (image only), CLIP B/32 and B/16 (image and text). SemArt Qwen default,
  SigLIP 2 and CLIP B/32, each with 21,382 rows and `image_file` and `type` in the index. SemArt images: 21,384 files.
  MMArt and Affection have no store for the three encoders, as §9 plans.
- **Matched arms.** M3 (10-25 round), M6 and the sample M5 (10-26 round, SSD2): held outputs and mappers (`.pt`) for
  seeds 0 to 4. Partitions are stored per seed. M3 is in 10-25's `reported_stage4`, M6 in 10-26's, and the sample M5
  has per-seed readings in 10-26's `reported_stage3`.
- **Claim test.** The full and sample Qwen Stage 1 topics (seeds 0 to 4) and the five P3 mappers exist. Stage 2
  trains a fixed 100 epochs, so the DINOv2 rerun stops on nothing.
- **r14 outputs.** CLIP L/14 c* weights, held exports and validation exports (6,152 rows); photo models s0 to s4;
  stage 1 records with ranges, ArtELingo picks (the SigLIP 1 frozen pick included) and Affection picks.
- **Teacher view.** `genre_word_s2img` holds all 6,152 ArtELingo validation paintings (checked by name), so the
  teacher's SemArt maps can be fitted as D20 says.
- **SemArt set.** With the code's rule it recomputes to 11,162 paintings by 2,717 artists; 217 at cosine ≥ 0.93;
  the md5 step drops 3.
- **SC15 inputs.** Qwen's extractor batch is 32, so D22's reason holds. CLIP extracts at batch 256, so batch 32 only
  slows c* (conservative). The SemArt B/32 store (10-03) predates the ArtELingo B/32 store (10-25), but the CLIP B/32
  encoder code did not change in between.
- **Verdict logic.** For ArtELingo, MMArt, photos, SemArt and cost, every cell pattern maps to one verdict. Inconclusive
  is never "qualified". It matches D4 and the four scenario checks of D15. Comparators and numeric bars are fixed for
  SC1 to SC15.
- **Held data.** Stopping, ranges and picks use validation only. SemArt selects nothing, and its units and maps come
  from ArtELingo validation. The matched arm's larger reading on test only makes the bar stricter. The probe reads no
  label.
- **Hard rules.** 1 to 4 and 7 hold: all sections present, no open question, every §4 row names who decided, "Check
  these" lists the picks taken after the recommendation, and the body (306 non-blank lines) is over target with a
  reason. Rule 6 applies at the tickets.
- **Budget.** node402 at about 3.4 GPU h matches r14's measured units. Local extraction of 1¾ to 2 h is plausible from
  the MMArt prep's rates.

Not done: no GPU use, no timing, no feature array loaded, no re-scoring of r14's or r15's records. The tickets do not
exist yet. Nothing to remove.
