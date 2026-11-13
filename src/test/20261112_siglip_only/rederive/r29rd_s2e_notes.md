# Rule findings and progress
- r29 §§3.2–3.5, §6, D17–D19/D22–D24/A1/A2 read (brief skipped). r11 retrieval §3.4/§6, r17 new-scores §§3.3–3.4 and permitted C13 implementations establish protocol.
- Plain-side candidate retrieval is r17 untold: unit heads emotion/style/genre concatenated / sqrt(3). Q1 side is r14 told head. Two arm-level picks, repeated per aspect. Both use told R0 ranges.
- SC9 uses float16 exports promoted to float32 without re-normalizing for k-means, n_init=3, batch=4096, states 0..2, 8 threads. K* groups read from sc2_rederived.json.
- Read-out menu, parity crossfit, labelled retrieval gallery, stable ties, R0 range, guard and score tie rules inherited from r11/r14; r17 AMI clusters all paintings and masks labels afterwards. Fresh rng(0), rows then seed draw per comparison.
- Prefixed E2/E3 caption stores absent at inspection. No stage-2 held input has been read. Own image forward implements the architecture in allowed r29_model.py and validates checkpoint sidecar widths; avoids importing r29_model.py's forbidden scoring/trainer dependencies.
- No files removed or commits made. All new files use job prefix except explicitly requested names.

## Numeric correction and checks
- Allowed docstrings only (AST extraction; no scorer import) of r11 rt_metrics/rt_readout/rt_stats/rt_data establish float64 dot products, W and normalization, with a single float32 cast of mapped rows. This differs from the earlier C1 numerical helper; job-local implementations now honor it.
- r14 held sidecar `sha256` is the image-array digest, not the archive digest. The first bounded map attempt exposed this; fixed by validating the array digest and the `arrays.img` entry.
- 12 small fixtures pass (0.01 s test body; about 2 s interpreter startup): outcome boundaries, SC9 tie/seed semantics, D22, held guard, output confinement, untold order, repeated-row AMI, bootstrap RNG transcript, stable AP ties, float64 mapping, parity/label masking, selection guard/ties.
- Full frozen map attempt stopped after 153 s when five-seed/two-side runtime was clearly >4 min. No files deleted. Own AP caches retained. --sides allows a bounded full told-side replay; --frozen-units allows full bootstrap with frozen independent per-query AP plus separately recomputed query samples (never full-scoring certification).

## Completed frozen validation replays
- All 30 per-seed AMI and all 30 per-seed R0 mAP cells for SigLIP2/CLIP L14 exactly equal r28 valpick.json; both scores within 1.1102230246251565e-16. CPU 220.784 s (command bounded at 225 s).
- All six class-count AMI ranges and six told R0 mAP ranges derived from those independent cells exactly equal r17/r14 records (12 checks, max absolute difference 0).
- Availability-only report: 122 missing E2/E3 weights/validation exports/sidecars/prefixed caption indexes; no pending result was invented. This is the state at inspection, not a claim about later controller runs.
- Frozen independent C13 per-query AP files can be located by the original source/input hash algorithm, with no lookup by point value. Both encoders' candidate and plain comparator unit-cache address probes succeeded.

## Cleanup (nothing deleted)
No removal is needed. If desired after controller review, obsolete exploratory files are r29rd_s2e_groups_siglip2_emotion.json/.log (earlier float32 scorer smoke), r29rd_s2e_map_siglip2_emotion.log (interrupted full attempt), and r29rd_s2e_unit_probe.json (address probe). Keep the final replay reports and source-bound cache for reuse; no other job's files are included.

## Completed AMI interval subset
- Emotion at K=9, both encoders, against both plain and Q1: all four complete 2,000-draw intervals pass at 1e-9. Largest interval endpoint difference 3.587408148320037e-15; largest point/statistic difference 1.0803857808383555e-14. CPU 142.4 s, full five training seeds and all 12,281 painting rows.
- The previous all-aspects SigLIP2 AMI command reached the 225 s bound before finishing (no result JSON produced); its parts caches remain. Style/genre full intervals remain controller work.
- r28 valpick record explicitly orders candidates siglip1, siglip2, clip_l14; the otherwise unspecified E2/E3 exact-tie fallback follows that stable candidate order.

## Completed mAP bootstrap and sampled ranking replays
- All 12 frozen comparisons (two encoders × three aspects × plain/Q1) pass the full 10,000-draw bootstrap replay from frozen independent C13 per-query AP arrays, located solely by source/input hashes. Largest endpoint difference 3.2439329000766293e-16; largest point difference 1.1102230246251565e-16.
- Independently recomputed 64 query positions per seed/cell with full galleries, all five seeds, both candidate read-outs, and plain comparator: maximum per-query AP difference is exactly 0 for both candidates and comparators. CPU 221.774 s for this bounded replay.
- This report deliberately sets full_scoring=false. Frozen-unit bootstrap agreement plus sampled ranking agreement does not substitute for complete independent ranking. Complete map replay remains in the controller command and is required by the held certification gate.

## Final fixture/forward/smoke checks
- 14 fixtures pass in 0.011 s test body, including rejection of a sampled ranking report as held certification, and `emb` exports / row-order rejection before any array is read.
- Frozen seed-0 emotion image-tower checks: SigLIP2 share equal 0.9983391135504714, CLIP L14 0.998222519734436; max quantized difference 0.00006103515625 for both. All residuals are at most one adjacent float16 value: both gates pass. Uses old weights/validation rows only.
- Real frozen crossfit smoke, both encoders, all three aspects, seed 0, 16 queries/full galleries, truncated no-whitening menu: completes in 7.4 s; exercises told/untold paths, no final pick certification.
- Remaining rule interpretation: “seeds ahead” means strictly ahead in matched training-seed IDs. E2/E3 exact ties follow r28's listed candidate order. No unresolved ambiguity blocks the implementation; full whitening picks and complete independent ranking remain to be run, not assumed passed.
