# Stage 1c independent derivation: controller handoff

No K* held input has been read or scored during development. No GPU has been used. `c1_rederived.json` is reserved for the controller's held command after committed validation picks and a passed, unparked SC2.

Implementation: `r29rd_c1.py`, `r29rd_c1_core.py`, `r29rd_test_c1.py`, runner `r29rd_run_c1.sh`. Imports are public libraries and our own SC1/SC2 modules; no round scoring or old C13 module is imported. The permitted independent C13 sources were read and adapted locally. The same forbidden outputs, code and round logs remain excluded. All writes/caches stay in this directory. No commits, deletions or .ccg records.

## Development results

All 13 scientific invariant tests pass, including all SC3 level boundaries, D22 upper-level handling, read-out ties/guards, whitening against an independent eigendecomposition, bootstrap transcripts and held-access prevention. Spawn workers after a multithreaded Torch product match serial AMI within 1e-14. CLI held mode without the committed-pick assertion exits before data access.

- AMI: full six-cell/five-seed/2,000-draw K0 replay reached max point/interval difference **2.4924506902834764e-14**, runtime **149.580784317106 seconds**. This full check preceded the spawn-only process-start fix; arithmetic is unchanged. A final-source emotion point check reached **1.0526302052227265e-14** in **5.171768152154982 seconds**. The final-source all-aspect full run hit the 230-second cap; full recertification is queued.
- Image mAP: all six K0 cells with full five-seed APs and 10,000-draw intervals completed in three capped commands; total **505.55903855059296 seconds**. Largest difference **1.9594635425335127e-07**: within r29's 1e-6 tolerance, outside the historical 1e-9 gate.

| Aspect | Untold/P1 max statistic difference | Told/Q1 max statistic difference | CPU seconds |
|---|---:|---:|---:|
| emotion | 4.3560814655049795e-08 | 2.4680487098471815e-09 | 196.8444531573914 |
| style | 1.38439625319009e-07 | 1.9594635425335127e-07 | 188.0531589873135 |
| genre | 3.093433142975499e-08 | 2.285192934881053e-08 | 120.66142640588805 |

- Text: all 18 frozen-AP interval checks agree within **5.551115123125783e-17**; all nine P1 side-row points match exactly. Final-source seed-0 own-forward/ranking check uses 128 queries per cell and full galleries: max AP difference **1.4394521713256836e-05**, max sampled mean difference **1.210719347000122e-07**, runtime **42.57644591713324 seconds**. This sampled check does not certify full scoring at 1e-9. Queue the full five-seed GPU replay.
- Actual candidate validation exports appeared before handoff. Seed-0 validation test with 256 queries, full galleries and reduced image menu completed in **28.194816475268453 seconds**. Image forward agreement after float16 rounding: equality share **0.9985047048131367**, largest difference **6.103515625e-05**, pre-quantisation max difference **6.070733070373535e-05**. All final picks remain unset in this diagnostic.

The final image/text score kernels are unchanged by the process-start fix. Development reports retain their original source hashes. The recommended controller commands regenerate all full reports for one final source/device before held certification. No K* held number has been computed. Full validation picks and `c1_rederived.json` remain reserved for the queued commands.

## Two distinct image candidates

- **Against P1:** r17 SC7's untold vector: independently unit-normalise emotion/style/genre heads, concatenate in that order and divide by float32 sqrt(3); one arm-level r11 cross-fit pick.
- **Against Q1:** r14's told aspect head; a separate common arm-level r11 cross-fit pick.
- Both use the candidate's told R0 per-aspect validation mAP ranges as the gain units and LEVEL units.
- Requested output `picks.image_map[aspect].pick` is the told/Q1 pick; `p1_pick` is the untold/P1 pick. The common picks are repeated per aspect. All 25 image read-outs, per-seed text means, AMI ranges and image ranges are retained for audit.

## Controller commands

Use the same Python environment as development. The runner enforces eight OMP/MKL/OpenBLAS threads, offline HF and no bytecode; CPU is the default. `--device cuda` restores the queue's original CUDA visibility after importing our older CPU-only helpers, disables TF32 and uses CUDA for forward passes, cosines, ranking, AP and expansion. Clustering, covariance eigendecomposition and bootstrap statistics remain CPU work. The runner confines the CUDA driver cache to `r29rd_c1_cuda_cache/` here. GPU timings below are estimates: no GPU benchmark was performed.

Full frozen K0 replays, before declaring the new scoring path certified:

```bash
bash /project/CoSiR/src/test/20261112_siglip_only/rederive/r29rd_run_c1.sh replay --family ami --device cpu --output r29rd_c1_replay_ami_full.json
bash /project/CoSiR/src/test/20261112_siglip_only/rederive/r29rd_run_c1.sh replay --family image_map --device cuda --output r29rd_c1_replay_image_map.json
bash /project/CoSiR/src/test/20261112_siglip_only/rederive/r29rd_run_c1.sh replay --family text --device cuda --output r29rd_c1_replay_text_full.json
```

The text command without `--query-limit` recomputes all five Qwen3 c* seeds on full K0 held galleries and queries, then recomputes intervals from those own APs. A sampled replay instead checks forward/ranking only on selected queries, and checks full intervals from the permitted frozen APs; its scope is explicitly reported and it cannot certify full cell values. K0 text uses the permitted r23 independent path's double input normalisation and forward batch 4096.

Full K* validation picks/ranges, once all weights and stored validation exports exist:

```bash
bash /project/CoSiR/src/test/20261112_siglip_only/rederive/r29rd_run_c1.sh validate --device cuda
```

This writes `r29rd_c1_picks.json` with `complete=true`, both image pick tables, all text read-out means, five-seed ranges, forward checks and source hash. It reads validation only. The full command has no query sampling, reduced menu or forward-only fallback. No final pick can be written by a subset diagnostic.

**Controller step:** ensure SC2 passed and commit the required pick/range records before any K* held scoring. This script neither reads the forbidden records nor performs a commit. After that step, explicitly assert it when invoking held:

```bash
bash /project/CoSiR/src/test/20261112_siglip_only/rederive/r29rd_run_c1.sh held --device cuda --picks-committed --replay-reports r29rd_c1_replay_ami_full.json r29rd_c1_replay_image_map.json r29rd_c1_replay_text_full.json
```

This writes `c1_rederived.json`: 30 unique comparison cells, full unrounded statistics, all picks/ranges, SC3 counts/level, image-forward diagnostics and replay reports. It pins the picks-file SHA256 and rejects incomplete or stale picks/replay sources. Own image outputs are compared with stored validation/held exports at float16 resolution; image metrics use the stored rows. Text uses own caption/image tower output in float32.

Every comparison whose replay is uncertified is conservatively reported INCONCLUSIVE (an unresolved diagnostic rather than a claim that three failed traced attempts have already occurred); it contributes neither WIN nor LEVEL. The ladder reports whether undecided cells could raise its level. An out-of-tolerance replay is recorded as an unresolved gate, not silently described as passed. The diagnostic reports include both the older 1e-9 gate result and actual numerical differences so the controller can resolve CPU/GPU numeric-path differences before claiming certification.

## Runtime estimates

| Command | CPU, 8 threads | Controller GPU plus CPU statistics |
|---|---|---|
| AMI replay, all three aspects, 2,000 draws | about 5–10 minutes estimated for spawn-safe final source; final-source full replay exceeded the 230-second cap | same CPU work; GPU unnecessary |
| Image mAP replay, all three aspects, full five seeds | about 8–12 minutes; split per aspect for bounded development | about 1–5 minutes, unmeasured |
| Full text replay, all five seeds and all queries | about 3–6 hours | about 15–45 minutes, unmeasured |
| Full validation picks, 25 image read-outs | about 1–3 hours | about 10–30 minutes, unmeasured; eigendecomposition stays on CPU |
| K* held cells and intervals | about 3–6 hours | about 20–60 minutes, unmeasured; AMI/bootstrap still CPU |

CPU estimates use our completed SC1 selected-arm raw validation runtime (~335 seconds per seed), population scaling and the bounded replays. Cache hits can make repeated commands faster. Each interactive development command has a 230-second timeout; the full commands above are for the controller queue.

## Protocol choices

All choices are also written into JSON `notes`. In particular:

1. AMI fits all paintings, masks negative labels only when scoring, and samples all painting rows before masking in bootstrap. Use stored float16-to-float32 candidate rows without another L2; plain/teacher store vectors are unit-normalised according to the inherited comparator convention. Three MiniBatchKMeans states, n_init=3, batch_size=4096; mean states then seeds.
2. Image retrieval labels mask queries/gallery after label-free maps are fitted. Cross-fit halves are fixed by original sorted-name parity, with equally weighted half means. Covariance uses float64 and n-1; whitening eigenvalues are descending and regularised by 1e-6 times the largest. The final map fits all validation rows. Image input/read-out rows and cosines are float32; AP/means/intervals are float64.
3. R11 selects one read-out per arm over all three aspects using mean gain in candidate told-R0 range units, the no-aspect-loss guard and 0.001 score tie band. The tie priority is R0, no expansion, none/center/whiten, smaller k, larger whitening size. A 1e-12 comparison epsilon follows the permitted r17 independent selector. Zero range prevents a certified pick.
4. Text APs are cast to float32 to preserve r23's recorded per-unit convention; means and statistics are float64. This differs from SC1's retained double AP by a storage rounding step. Validation means use all rows, float64 average then float32 mean vector. Centering/expansion rank all rows, unlabelled rows leave only AP, own caption row leaves t2t only. Expansion includes the query plus ten gallery neighbours.
5. Exact score ties preserve gallery order: sorted painting names or ascending caption extraction IDs. Emotion plurality ties take the alphabetically earliest of all nine emotions, including `something else`; missing/conflicting genre is -1. Input caption IDs are joined explicitly with painting checks.
6. Fresh default_rng(0) per comparison. AMI draws rows then candidate seeds (2,000 draws). Image mAP draws labelled query indices then shared seeds (10,000). Text draws sorted painting units with at least one scored query, then shared seeds; repeats count in query sums and counts. Percentiles use numpy's linear method. No rounded threshold or outcome tie band is added.
7. P1's text form is fixed plain; per-cell r23 SigLIP side-row picks are used. Q1 text uses its own r23 teacher pick. Q1 image retrieval uses frozen r11 instructed per-query APs; P1 image retrieval is rebuilt from its stored plain image at the frozen r14 pick and checked in K0 replay.
8. The specified prefixed store records lowercase `represent the caption: `, while the prose capitalizes it. Score the specified stored vectors unchanged.
9. Ranking batches are 128; candidate CPU/GPU forward batches 1,024, K0 r23 replay batches 4,096. Subsets use sorted default_rng(0) sampled query positions, full galleries. `--forward-only-validation --quick --query-limit N` is a diagnostic fallback while validation exports are absent; its image rows are quantised own forwards and it reports forward/export comparison as pending. It can never supply final picks.

## Cleanup candidates

Nothing was deleted. After audit, candidates are `r29rd_c1_cache/` (content-keyed diagnostic/replay caches), `r29rd_c1_cuda_cache/` if created by the controller, `r29rd_c1_replay_text_impl.txt`, `r29rd_c1_validation_smoke.json`, `r29rd_c1_validation_smoke.log`, `r29rd_c1_replay_ami_probe.json`, `r29rd_c1_replay_ami_probe.log` `r29rd_c1_guard_check.log`, `r29rd_c1_mp_check.log`, `r29rd_c1_replay_ami_final_probe.json`, `r29rd_c1_replay_ami_final_probe.log`, `r29rd_c1_replay_ami_final.log` (capped run), and `r29rd_c1_tests.log`. Keep final replay reports, sources, tests, runner, handoff notes and any eventual picks/final output as evidence. Prior SC1/SC2 files remain intact.
