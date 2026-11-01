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
