# r13 photo fine-tune: round log

Spec: `docs/superpowers/specs/2026-10-31-photo-finetune-design.md`. Queue: `pf_queue.sh` (full | smoke). Outputs of the
full run: `/data/SSD2/cosir_r13/20261031_photo_finetune/outputs/`.

## Build smoke 1 (2026-11-01 17:03 to 17:31, build chat 4)

`bash pf_queue.sh smoke <scratch>/smoke1` on the integration branch `r13-photo-finetune`: PF_SMOKE=2000 (the first
2,000 sorted train keys; validation and test full), rows 1 and 8, 2 epochs, stage 3 and C13 at 200/500 draws. Not a
result: it checks the wiring and the formats.

| Step | Seconds | Note |
|---|---|---|
| extract_images | 129 | 2,000 instructed train images, with model load (GPU) |
| extract_texts | 183 | 38,913 instructed explanations: the 2,000 train photos' rows and all validation rows (GPU) |
| stage0a | 529 | graphs (gate iv) 212, stage 2 replay 173, SC replay 58 (at 200/500), loss 31, sweep 22, rest under 20 |
| stage1 | 31 | rows 1 and 8; row 9 equal to row 1, skipped |
| stage2 | 161 | five final runs, 2 epochs |
| stage3 | 70 | |
| c13_derive | 185 | |
| c13_compare | 4 | 298 items, all equal within 1e-6 |
| diag | 75 | |

Two failures, both fixed and rerun on the same scratch folder (the queue's seals skipped the finished steps):

1. **Stage 0a stopped: 9 of 10 gate items ok.** The SC replay (gate 0a(ii)) matches r12's points, d and r, but its
   intervals match r12's within 1e-9 only at the full draws (2,000 / 10,000). Fix `0fb35fe`: the smoke runs stage 0a
   at the full draws (the SC part takes 72 s). Then `pf_run.py 0a --parts sc` on the smoke folder: 10 of 10 items ok.
2. **Diag crashed: "R0 mAP differs from the verdict row".** Diag computed R0 mAP with a bare
   `rt_metrics.retrieval(qe_k=0)`; the verdict rows (and r12's R0 rows) go through the read-out `none+k0`, whose
   re-normalization moves rows by about 1.5e-8. On seed 0 the two give 0.2912723187 and 0.2912723158 (3e-9 apart, above
   the 1e-9 check). Fix `85f54be`: diag computes R0 mAP through `none+k0`. Then diag passed.

Gates (all ok): parity images min cosine 0.99974 (200 rows), parity texts 0.99935 (200 rows); views 2,000 images,
38,913 texts; sweep replay AMI and mAP difference 0.0; stage 2 replay; SC replay at full draws; inference max abs diff
0.0; the three r10 graphs exact; near-copies 29 of 2,000 smoke train photos removed; loss 6.161110545314756 (difference
0.0).

GPU peak and power not measured (light: 2,000 images, 2-epoch runs).

## Final whole-branch review and fix wave (2026-11-01 17:33 to 18:20, build chat 5)

Opus `xhigh` review in four parts (a first reviewer stopped at its context limit; three parts went on in parallel).
Record: `.scratch/r13-photo-finetune/notes/final_review.md` (outside git), details in `review_final/part_{A,B,C}.md`.
No FIX BEFORE RUN; nothing gave a wrong number. Leakage ruled out for smoke 1's row 1 score (a retrain with random
validation arrays gives the same checkpoint bit for bit; a ridge map default to teacher image on the fit photos
reaches validation AMI 0.1561, mAP 0.2921, below the teacher's 0.1719, 0.3044). Hard rule 6: SC1 to SC5 and D1 to
D14 all covered.

Fix wave, two Codex jobs (`gpt-6.1-sol` high), merged `b56b86a` (W1) and `c35db35` (W2):
- W1: stage 2 refuses a sheet whose picked sweep run is missing, failed, of other settings or of another fit set, and
  a reused run of another fit set (B2); stage 1 is sealed by the sheet and `stage1/rows.json`; C13 compare always
  runs and diag reruns when older than the verdict or the compare (B1); stage 3 checks r', the read-out, and the
  final runs' seeds, start hashes and fit set before any test export (A1, D10, D12); wiring tests past the guards;
  full-run refusal of the smoke flags tested (A2); require_gates refuses a 0a of another smoke setting (A3); stage
  tracebacks go into the stage log; the dead smoke relaxation of the full-draws gate item removed.
- W2: the 13 deferred ticket-review notes (sweep sheet and pick guards, stage 2 and stage 3 guard tests, stage 3
  checks of each comparator's point and AMI against r12's records, near-copy cut and count tests, graph refusals).
- All targeted mutants fail a fast test (W1 14 of 14, W2 56 of 56). Fast suite 362 passed. Sonnet `high`
  re-review: pass (`notes/fix_wave_rereview.md`).

## Build smoke 2 (2026-11-01 18:13 to 18:37, build chat 5, after the fix wave)

`bash pf_queue.sh smoke <scratch>/smoke2` on `r13-photo-finetune` at `c35db35`, same settings as smoke 1. All steps
exit 0; the two commit steps are skipped in smoke, as designed.

| Step | Seconds |
|---|---|
| extract_images | 126 |
| extract_texts | 184 |
| stage0a | 580 |
| stage1 | 31 |
| stage2 | 169 |
| stage3 | 72 |
| c13_derive | 188 |
| c13_compare | 3 |
| diag | 76 |

Gates 10 of 10 ok; C13 compare 298 items, 0 mismatches; diag ok. The new stage 3 comparator checks pass on r12's
records. Rerun test of B1: with `stage3/verdict.json` moved aside, the queue reran stage 3, compare and diag (3 run,
8 skipped, 18:37 to 18:40), and the new verdict equals the first byte for byte.

## Full run (2026-11-01 18:43 to 21:24, r13 run chat)

`bash src/test/20261031_photo_finetune/pf_queue.sh full` on main at `949ad04`, local GPU (RTX 3090) under the lock,
Sunday evening. All 11 steps exit 0. Outputs: `/data/SSD2/cosir_r13/20261031_photo_finetune/outputs/` (849 MB); new
views `emotion_focus_train` (286 MB) and `emotion_focus_txt` (1.9 GB) under
`/data/SSD2/pre_extract/affection_local/qwen3vl_emb_2b_aspect_prompts/`.

| Step | Seconds | Note |
|---|---|---|
| extract_images | 3,630 | 72,255 instructed train images (about 20 per second); GPU about 350 W |
| extract_texts | 2,142 | 474,051 instructed explanations (train and validation) |
| stage0a | 646 | gates 10 of 10 ok |
| stage1 | 1,840 | 9 rows, none failed; sheet committed `ee67eb2` |
| stage2 | 910 | five final runs; record committed `dd51dde` |
| stage3 | 95 | verdict |
| c13_derive | 266 | |
| c13_compare | 4 | 378 items, 0 mismatches (max difference 1.1e-16) |
| diag | 115 | |

The whole queue took 2 h 41 min, against the 5 to 6 h estimate (the explanation and training steps were faster).

**Text join check (final review B4), 20:20, on the full view** (474,051 rows; script
`.scratch/r13-photo-finetune/review_final/B/join_check.py`): matched cosine 0.5972 against 0.3842 (row before), 0.3843
(row after) and 0.3843 (next view row); the matched row wins in 93.1% to 96.1% of rows per source (94.4% overall). The
key check passed. The join is right, as on the smoke view.

**Stage 0a gates:** all ok. Image parity min cosine 0.99974 (200 rows), text parity 0.99935 (200 rows); the r12 replays
within 1e-9; inference max difference 0.0; the three r10 graphs exact; loss 6.161110545314756 (difference 0.0).
Near-copies: 2,328 of 72,255 train photos removed (cut 0.85), so 69,927 fit photos.

**Stage 1 (validation, seed 0):** pick row 5 (k = 80), score 5.42 r_clu (validation AMI 0.1905, mAP 0.3280; the
teacher's 0.1719, 0.3044). Next best: row 2 (learning rate 0.0003) 4.49, row 4 (k = 20) 4.04, row 3 3.55, row 9 (three
changed settings: learning rate 0.0003, k = 80, teacher weight 1) 3.43, row 7 (teacher weight 1) 2.52, row 8 (random
start) 2.40, row 1 (c*'s settings) 1.45, row 6 (no content correction) -11.06; frozen c* (row 0, reference) -18.65. The
pick leads row 2 by 0.93 r_clu. Training is not deterministic across devices (final review C-N1: about 0.6 r_clu
between GPU and CPU for one row), so the margin is above that noise, but not by much. Best epochs: 1 to 4 for rows
1 to 6 and 8; 14 for row 7; 10 for row 9.

**Stage 2 (validation, five fresh runs):** all ok, all stopped at best epoch 1 (the validation loss rises from epoch 2).
r'_clu 0.00707 (r12's c* 0.00343), r'_mAP 0.00242 (r12's 0.00101); read-out centering plus k = 10.

**Stage 3 (Affection test, 8,501 photos): holds.** SC1 to SC5 all WIN; `stage3/verdict.json` 21:17, then C13 ok at
21:22 (newer, `"ok": true`), so the verdict is final (final review B3).

| ID | Photo model | Comparator | Difference [95%] | Outcome |
|---|---|---|---|---|
| SC1 AMI | 0.1838 | plain 0.1041 | +0.0797 [+0.0741, +0.0857] | WIN |
| SC2 mAP | 0.3172 | plain 0.2452 | +0.0720 [+0.0699, +0.0742] | WIN |
| SC3 AMI | 0.1838 | teacher 0.1666 | +0.0172 [+0.0121, +0.0225] | WIN |
| SC4 mAP | 0.3172 | teacher 0.2945 | +0.0227 [+0.0210, +0.0244] | WIN |
| SC5 mAP | 0.3172 | CRL 0.2739 | +0.0433 [+0.0411, +0.0455] | WIN |

**Diagnostics:** photo model minus frozen c*: AMI +0.0454 [+0.0408, +0.0501], mAP +0.0417 [+0.0399, +0.0436]. Share of
the teacher's gain over the plain image: 1.28 (AMI) and 1.46 (mAP), against frozen c*'s 0.55 and 0.61. With tied
images dropped (6,584 photos): holds. Row 8 against row 1 (seed 0): AMI +0.0033, mAP +0.00001.

**Agent defaults from builds 1 to 5** (run handoff §4): smoke subsets train only; train keys from the index's text
rows; the within-train near-copy count (10,114) counts photos, 1,057 of them also near-copies of held photos; gate (i)
uses the first 100 sorted validation keys plus the first 100 FI validation keys; gate (ii)'s frozen c* mAP is r12's
five-seed `center+k10` mAP; sweep rows start from c* seed 0 with training seed 0, final runs from c* seed s with
training seed 100 + s; row 8 uses c*'s factory under the run's seed; finished runs are reused, never retrained;
comparators' partitions and per-query AP come from r12's outputs (hash- and row-checked); diagnostics as listed in
`diag.json`'s `reading` block.
