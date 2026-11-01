# r13 rule review (C7)

> Reviewer: a fresh Opus `xhigh` worker, 2026-10-31, on the spec draft at `f0c77c2`
> ([spec](../../../docs/superpowers/specs/2026-10-31-photo-finetune-design.md), §6 with §3.4 and §3.5). Verdict: pass
> after fixes. Every must-fix and should-fix item was applied in `95e1f2e`, with these changes: S3's tolerance is 1e-4,
> not 1e-5 (GPU summation order); the rest as proposed. The near-copy evidence (scripts and outputs) is in
> `review_nearcopy/`; the scripts wrote their `.npz` files to a session temp folder that no longer exists.

## Must-fix

- **M1. The 0.98 near-copy cut misses about 97% of the real copies.** For each train photo, its nearest validation or
  test photo by default Qwen cosine; each pair compared by 32×32 grayscale thumbnails (pixel copy: correlation > 0.95).
  1,369 train photos have a nearest cosine ≥ 0.90; 923 of them are pixel copies; 22 of the 923 reach 0.98. Pixel
  copies by cosine band: 0.98 to 1.0, 22 of 24; 0.97 to 0.98, 40 of 52; 0.95 to 0.97, 393 of 461; 0.93 to 0.95, 325 of
  437; 0.90 to 0.93, 143 of 395; 0.85 to 0.90, 5 of 127 sampled; 0.80 to 0.85, 0 of 273 sampled. Lowest copy cosine
  0.8897. 877 of the 923 are Emotional Machines ↔ FI pairs: the same Flickr photo at 640 px in one set and 500 px in
  the other. About 625 test photos (7.4%) and 298 validation photos are touched. Fix: cut at 0.85 (about 2,328 train
  photos, 3.2%), float32 cosine on L2-normalized rows of the default store's `img.npy` joined by key; the photo and its
  explanations leave before any graph is built.
- **M2. The validation stopping graph cannot be built as written.** It needs E (the instructed explanation mean) on
  the 4,251 validation photos; no Affection instructed text view exists. Fix: extract the validation explanations too
  (26,371 rows, about 1.5 min).
- **M3. Row 9 was ambiguous** ("beats row 1": strictly or beyond the tie margin; "among 1 to 7" includes row 1).
  Fix: the highest-scoring row among 2 to 7 that changes the setting, if strictly above row 1; else row 1's value;
  equal scores to the lower row number.

## Should-fix

- **S1.** D9: "changed settings" = the number of the five grid columns that differ from row 1; the margin inclusive
  (score ≥ best − 0.5).
- **S2.** Teacher values in full, from r12's `stage1/scores.json`: `teacher_wording.clustering.artwork` =
  0.17189868120941146; `teacher_wording.retrieval.artwork` = 0.304397579447216 (equal to
  `readout.affection.teacher_artwork["center+k10"].map.emotion`).
- **S3.** No check of the trainer or the loss (C12). Gate (vi): the new loss code, fed c*'s full-scale ArtELingo
  inputs and validation graph, gives `best_dev_loss` = 6.161110545314756 of `emotion__s0.pt`. c*'s validation loss
  uses a fixed pair order with no shuffle (`src/test/20261020_claim_test/ct2_d2.py` lines 201 to 217).
- **S4.** Gate (iv) covered only k = 40 with the correction on. Add r10's `graphs/full/uncorrected__k40/d2/graphs/`
  `emotion.npz` and `graphs/sample/aspect__k10`. k applies to every mutual top-k in the builder, including the edges
  that define λ (`rs_graphs._partition_aspect`); the validation graph uses the row's k and correction.
- **S5.** Gate (ii) also replays r12's `stage0b/ranges.json`, c*'s 25 read-out scores and its pick, and frozen c*'s
  seed-0 sweep score (AMI 0.14268969071601648); frozen c* goes in the sheet as a reference row.
- **S6.** D10 reused the sweep winner's seed-0 run as a test seed. Fix: five new runs; run s starts from c*'s seed s
  and uses seed 100 + s for its training randomness.
- **S7.** Row 8's initialization: c*'s trainer's initializer under the run's seed.
- **S8.** 4,422 of the 447,680 train explanations are on "ignored" rows; E, D and the caption inputs use rows whose
  split is not "ignored", as r12's labels.
- **S9.** C13 re-derives from saved outputs (each row's seed-0 validation embeddings saved), not by retraining.
- **S10.** A row with a non-finite loss or a crash is recorded as failed, gets no score, and is left out of the pick
  and of row 9; if row 1 fails, stop.
- **S11.** Gate (i): the first 200 sorted validation keys are all `coco_affective`, so none is FI. Use the first 100
  sorted validation keys plus the first 100 sorted FI validation keys, batch 32; name the ArtELingo text store
  (`/data/SSD2/pre_extract/artelingo_full/qwen3vl_emb_2b_aspect_prompts/emotion_focus/txt.npy`).
- **S12.** The photo model's checkpoints and sidecars (sha256) go under the r13 outputs root and its exports check
  against them; only gate (iii) reads c*'s root and hashes.
- **S13.** r' spelled out: max minus min over the five test runs of validation AMI (three states, 20 clusters, all
  4,251 photos) and of R0 mAP and nDCG (all validation photos as queries and gallery); r'_clu for SC1 and SC3, r'_mAP
  for SC2, SC4, SC5 and the read-out pick.

## Nits (applied)

N1 claim-test outcomes mapped (holds GO, qualified GO for the qualified claim, fails NO-GO); N2 the licensed claim
says "without emotion labels, from the photos' free-text explanations" and "minus near-copies of held photos"; N3 the
brief's "Qualified" names both plain-image WINs; N4 the held-data row says the filter reads images, never labels; N5
fresh AdamW state, epoch 0 not a stopping candidate; N6 copies inside train stay in, count reported.

## Judged sound

D13 (the fixed sweep read-out: center+k10 was r12's pick for c*, the plain image and the teacher, so the gap compares
like with like); D14 (a paper claim; no tuning or selection reads test; the timing is disclosed); the two kinds of
range do not conflict (r12's ranges scale only the sweep score, r' only the outcomes and the read-out pick); every
result combination has a verdict.

Checked against files: r12's ranges, frozen c*'s per-seed AMI and mAP, the gaps of about −8 and −18 r, the split
counts, 447,680 train explanations, 1.83 GB of float16 explanation vectors, 19.4 images/s, report 12-31's shares and
gaps, c*'s graphs and `best_dev_loss` on disk. Not checked: the text rate of 310/s, the training time per row, the
graph build time at about 76k nodes.
