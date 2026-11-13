# Independent SC2 handoff

Full candidate means, five-seed group values, level/status, and image-forward comparison are pending the controller CPU queue and candidate exports. No final `sc2_rederived.json` was written from partial data.

Selected arm from our completed `sc1_rederived.json`: **siglip1_prefix**, prefixed captions. At the final availability check, seed-0 and seed-1 weights exist for all three heads, but no candidate `val/siglip1_prefix/<aspect>__s<seed>.npz` exports exist. Do not wait for training in this session.

## Full run

After all five seeds' three checkpoints, validation exports and sidecars exist:

```bash
bash /project/CoSiR/src/test/20261112_siglip_only/rederive/r29rd_run_sc2.sh
```

This runs `/root/miniconda3/envs/CoSiR/bin/python r29rd_sc2.py run`, CPU only, eight OMP/MKL/OpenBLAS/Torch threads, HF offline, no bytecode. It writes `sc2_rederived.json` with the requested full schema, `sc2_rederived.run.json` with runtime/provenance/per-state AMI/query counts, and `r29rd_sc2_full.log`, all here. The full mode checks every required input before expensive work and requires the completed independent K0 regression report with the current source hash. It recomputes every candidate cell, and checks each seed-0 text cell against our completed SC1 within 1e-6.

Own completed SC1 provenance reports 335.1278000380844 seconds for the selected arm's full seed-0 text. Five text seeds plus 15 image files imply roughly 30 minutes for full SC2; unmeasured until the queued run. The interactive tests below were each capped at 230 seconds and used no GPU.

## K0 image regression (full, all five seeds and three heads)

Runtime: 85.0497019472532 seconds. All 30 per-seed image cells agree with r28's frozen records within 1e-6.

| Aspect | Largest AMI difference | Largest mAP difference |
|---|---:|---:|
| emotion | 0.0 | 8.300149234585774e-09 |
| style | 0.0 | 5.020673077282822e-08 |
| genre | 0.0 | 2.4487293859287718e-08 |

Overall largest differences: AMI **0.0**, mAP **5.020673077282822e-08**.

These computations are self-checks only. The pass-rule baseline still comes directly from frozen r28 records, without substituting the re-scored K0 values.

## Frozen baselines

| Group | m | r | best | lower |
|---|---:|---:|---:|---:|
| image | 0.38629170813482466 | 0.002978860218950219 | 0.38786969161861545 | 0.38331284791587444 |
| text | 0.259351288425063 | 0.0020107135605435533 | 0.2606346548204899 | 0.25734057486451944 |

All four specified 10-decimal mean/range checks pass. Text uses the nine validation picks' `means.r0` and `per_seed.r0` only; image m uses the recorded `score`. Ranges are group-level maximum minus minimum over seeds, without rounding.

## Candidate seed-0 text test

256 deterministic sampled queries per direction, full validation galleries, group **0.25461091934964497** (subset only). Runtime: **9.274409942794591 seconds**. Counts: emotion/style 256 per cell; genre t2i=215, i2t=202, t2t=215. It uses the actual K2 seed-0 checkpoints. No SC2 level is inferred from this subset.

Stored candidate image exports were unavailable, so forward agreement is **pending**, not measured. The full script independently projects every validation image for every aspect/seed, rounds its output to float16, reports numerical element equality count/share and maximum absolute float16 difference, and also reports the maximum pre-quantisation difference. Image cells always use stored rows rather than projected rows.

Six SC2-specific synthetic/scalar tests pass in 0.023 seconds. They check exact level boundaries, clustering all rows before label masking, lack of AMI input renormalisation, labelled-only image AP, scalar retrieval AP, weighted float16 forward diagnostics and frozen baselines.

## AMI population and implementation choices

1. **All 6,152 paintings enter clustering.** R17's `2026-11-03-new-scores-design.md` §3.3 explicitly says to cluster every row and mask missing aspect labels after clustering. Genre AMI is then scored on 4,960 labelled rows. Image retrieval queries/gallery are labelled paintings only.
2. **No extra AMI normalisation.** R28 valpick's permitted module docstring and the SC2 brief explicitly specify stored float16 rows cast to float32 with no further L2, resolving r17's more general unit-row wording. MiniBatchKMeans uses K=9/27/10, n_init=3, batch_size=4096, states 0/1/2, eight threads, other installed sklearn defaults, and arithmetic-normalised sklearn AMI. State values are averaged in float64; AMI configurations and sklearn version are recorded in provenance.
3. **Image retrieval numerics.** Promote float16 to float32, L2 once, float32 CPU cosines, stable descending order with ties in sorted painting order, remove the own painting, float64 AP and finite-query mean. No-positive queries are dropped. Every real K0 labelled query is scoreable.
4. **Text numerics.** Reuse our standalone SC1 stored-row ID joins, per-caption input, labels, exact GELU towers, output L2, stable ties and AP, without output float16 quantisation or read-out changes. Painting plurality includes all nine emotion classes and ties alphabetically; missing/conflicting genre values are -1. The lowercased prefix in store metadata is used as supplied, as in SC1.
5. **Forward agreement is a diagnostic.** Report numerical equality after float16 quantisation and differences without inventing a minimum equality-share threshold. Final pass/fail follows only the stated group comparisons.
6. **Aggregation and thresholds.** Equally average six image or nine text cells per seed, then five seed groups. Frozen K0 m/r/best/lower are unrounded. Level 3 uses strict `>` for both best seeds; level 2 requires both lower bounds and text at least its K0 mean; level 1 requires both lower bounds; otherwise level 0, status fail. No tolerance band is added.
7. **Bounded CPU work.** Ranking batch size 128 and forward batch size 1,024. Subset queries are sorted `default_rng(0).choice(..., replace=False)` positions and retain full galleries. Full mode has no sampling.

Only permitted specs, the first 20 lines requested for the r28 valpick protocol, permitted r28 baseline records, data/weights/sidecars and our own re-derivation artifacts were accessed. No forbidden scoring modules were imported, including transitively; imports are standard/public libraries and our own `r29rd_sc1.py`. Earlier SC1 artifacts remain intact. No commits, deletions, .ccg records or writes outside rederive/.

Cleanup candidates after review: `r29rd_sc2_k0_probe.json`, `r29rd_sc2_k0_probe.log`, `r29rd_sc2_inspection.json`, `r29rd_sc2_inspection.log`, `r29rd_sc2_text_smoke.json`, `r29rd_sc2_text_smoke.log`, `r29rd_sc2_tests.log`. Keep the full K0 regression report and source/tests/runner/notes as audit evidence. Existing SC1 cleanup candidates are listed in `r29rd_notes.md`. Nothing was deleted.
