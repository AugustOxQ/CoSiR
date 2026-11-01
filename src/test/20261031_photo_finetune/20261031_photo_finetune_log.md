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
