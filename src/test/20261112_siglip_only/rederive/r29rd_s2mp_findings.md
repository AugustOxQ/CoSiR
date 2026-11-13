# Protocol findings
- r15 mt_c13: MMArt key SHA256 order, inclusive near-copy cut .95 after exact-duplicate filter, then first md5; 16,384 test rows. Validation 5,769. Store clustering unit(raw); store retrieval unit(unit(raw)); adapter clustering raw float16 promoted to float32, retrieval once normalized.
- r17 ns_c13: class-count MiniBatchKMeans 9/20/10, n_init=3, batch_size=4096, states 0..2, eight numerical threads on paintings, one on photos. AMI bootstrap 2,000 default_rng(0), all rows then 5 seed positions, partitions fixed; unlabelled masked at AMI.
- r15 plain-side MMArt mAP uses untold joined heads per r17 SC7, Q1-side uses aspect head. Each needs its own validation pick. Shared candidate range at R0 from aspect heads as r15/r17.
- r12 Affection: sorted keys; plurality emotion alphabetical ties; all nine emotions; photos one-thread k-means. r14 EmoSet: SHA256 order, frozen near-copy removal list; same retrieval bootstrap (10,000 queries then seed positions).
- D19 says rounded 0.00695 in prose but D18/r28 use full committed 0.0069460867449697705; take full stored r14 ranges.map.r and explain.
- Stage 2 MMArt and photo exports absent at initial inspection; forward check pending. Never score candidate held in this session.

## Initial verification
- Affection comparator rebuilds from keyed stores and frozen picks passed full-query 1e-9: largest absolute difference 5.120679991144073e-10 (2 comparators, 49.55 s CPU including CSV/store checksums).
- Small protocol suite: 13 tests passed in 1.617 s; checks sklearn-equivalent AMI, missing labels/repeated bootstrap rows, exact draw transcript, stable ties/self removal, strict WIN/LEVEL boundaries, D22 ladder and held CLI guard.
- Reduced seed diagnostics must compare only comparator means or target per-seed values; comparing a one-seed candidate to a five-seed target is invalid. Fixed before full AMI replays.

## Full AMI/retrieval replays and numeric correction
- MMArt emotion AMI: 13 checks, both r17 SC1/SC3 paired intervals and r28 point; max 6.689093723366568e-15, 79.73 s CPU.
- Affection mAP: 10 point/difference/interval checks vs r14 SC5/SC6, max 5.303496253272755e-10; comparator rebuild max 5.120679991144073e-10; 159.89 s CPU.
- Affection AMI: 12 checks against r17 SC4 plain/teacher, max 5.329070518200751e-15; 56.13 s CPU.
- Initial MMArt emotion plain mAP missed 1e-9 (2.1652351300804895e-9). A frozen-only probe of documented float64 normalization for the picked none map reduces error to 1.7726403678253178e-11. Implemented that normalization for every W, including none, and parity-half cross-fit; kept raw R0 range metrics separate. Historical failed probe preserved; no deletion.
- 14 fixture tests now pass (1.101 s), including a regression for none-map normalization.
- EmoSet frozen photo SigLIP1 seed-0 mAP (full test gallery and all queries) matches r28 photos cell a within 3.8402991897612537e-10. Plain SigLIP1 comparator mAP rebuild matches cell c within 1.137819838348264e-10. CPU 125.74 s for the two full-query scorings.
- 16 fixtures now pass in 1.613 s, including all 20 INCONCLUSIVE cell identities with no model-data reads, and union/completeness/failure of frozen replay evidence.
- Full EmoSet AMI replay: all five SigLIP1 photo seeds match r28 photos cell a; Qwen photo candidate and both 2000-draw r17 SC4 intervals match. 18 checks, largest absolute difference 3.7192471324942744e-15; 118.40 s CPU.
- All EmoSet comparator rebuilds passed: AMI against r28 cell c and r17 teacher max 3.552713678800501e-15; mAP against r28 cell c and r14 SC10 teacher max 1.137819838348264e-10. 56.56 s CPU (plain AP cached).
- Final production path MMArt emotion plain/teacher mAP rebuilds both pass 1e-9 after applying none-map normalization. Largest difference 8.48049408475049e-11; CPU 87.61 s with persistent float64 gallery reducing repeated conversion cost.
- MMArt style plain and teacher mAP comparator rebuilds pass: largest difference 5.657299073646982e-10; 91.27 s CPU. Genre follows as a separate bounded command.
- MMArt genre plain/teacher mAP rebuilds both pass: largest difference 2.1983914688661343e-10; 68.32 s CPU. Across all six MMArt rebuilt mAP comparator points, largest difference is 5.657299073646982e-10 (style). All photo comparator rebuilds also pass.
- Frozen MMArt validation smoke (5 seeds, 16 queries per gallery, reduced none/center/center+k10 menu) completed in 22.00 s; exercised all aspects and untold/told/plain/teacher picks, teacher ids.npz validation joins and R0 range code. This is a diagnostic only, not a complete pick/table replay.
- Frozen MMArt untold held retrieval smoke (seed 0, 16 queries, all three full test galleries) completed in 11.38 s. This exercises SC7's joined vector, its none+k1 pick, self removal, missing-label restriction and plain comparator path. No complete candidate mAP replay is claimed from a query subset.
