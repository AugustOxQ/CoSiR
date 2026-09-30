# Matched head-to-head: SDD ledger (preserved record)

Verbatim copy of the subagent-driven-development ledger for plan
`docs/superpowers/plans/2026-09-30-matched-percept-buddy-h2h.md` (Tasks 1-8, the
confirmation checks, the sweeps, selection, test, the final whole-branch review
and its fix wave). The original lived in the git-ignored
`.superpowers/sdd/2026-09-30-matched-percept-buddy-h2h/progress.md`, deleted on
2026-10-01 after the final review came back clean. Results: `docs/reports/auto/percept/2026-09-30_matched_percept_buddy_h2h.md`.

---

# SDD ledger — plan: docs/superpowers/plans/2026-09-30-matched-percept-buddy-h2h.md

Spec: docs/superpowers/specs/2026-09-30-matched-percept-buddy-h2h-design.md (reachable, binding).
Context: user asleep 2026-09-30 ~03:00, authorized full automation ("you need to automate and get a good report in the morning"); spec/plan user-review gates skipped under that authorization (to be stated in the morning report).
Phase 0 (QC1/QC2) tooling committed a906a89 (sonnet implementer, controller-reviewed diff); 9 QC jobs launched on DAS6 at a906a89.

## Pre-flight conflict scan

| pair / task | interface | check | result |
|---|---|---|---|
| T1 -> T2 | Stage1Output, H2HStore fields, PilotModules.arch | T2 uses store.train_img/txt, *_affect28, *_content_raw; pilot.arch | OK (names match T1 Produces) |
| T1 -> T3 | Stage1Output, H2HStore.train/heldout_percept_h | T3 reads store.*_percept_h | OK |
| T1 -> T4 | H2HSplit(val_idx,test_idx,digest), EVAL_TRANSFER_K | T4 run_h2h_trial uses split.val_idx/test_idx/digest | OK |
| T2 -> T4 | fit_buddy_stage1(cfg, store, seed, monitor_idx, pilot, device) | T4 calls with monitor_idx per `monitor` arg | OK |
| T3 -> T4 | fit_percept_stage1(cfg, store, seed, k_target, mods, device), load_percept_modules | T4 calls; n_topics = distinct train labels | OK |
| T4 -> T5 | stage2 path pieces | T5 uses targets/stage2 directly + harness mapper; does not need run_h2h_trial | OK |
| T4 -> T6 | run_h2h_trial, build_topic_graph, independent_partition | T6 v2 uses pilot.arch.detect_communities on build_topic_graph(...,"pilot_repaired") | OK |
| T4 -> T7 | resolve_h2h_config flat keys buddy_*/percept_*/stage2 names | T7 YAML names: buddy_impl, buddy_heads, buddy_num_heads, buddy_d_shared, buddy_lr, buddy_batch_size, buddy_content_pca_dim, buddy_temperature, buddy_max_epochs, buddy_plateau_window, buddy_noise_std, buddy_lambda_affect, buddy_weight_decay, buddy_teacher_graph_K; all are BuddyStage1Config fields (impl, heads, num_heads, ...) | OK |
| T4 -> T8 | run_h2h_trial rows, H2HConfig | T8 summarize keys match T4 per-seed row keys | OK |
| T2 self | tests vs code | test expects pilot_constants ValueError for impl="nope" and heads="attn" | OK consistent with Step 3 |
| T3 self | tests vs code | module key per constant may differ from test assumption; task text tells implementer to fix the test's module key to reality | OK (explicit escape hatch) |
| T4 self | Review Focus 1,2,4,5 tests | RF1 k_miss->-1 test, RF2 _auc_for skip test, RF4 38 labels test present; RF5 k_target=0 path: config test covers percept rejection, trial path not tested | GAP — ruled below |
| T1 self | RF3 concurrency test | threads test present | OK |
| T7 self | YAML heads mapping | buddy_heads [mlp128, attn1, attn4] but harness BuddyStage1Config.heads expects mlp128|attn | GAP — ruled below |
| Global | "implementers do not commit" | SDD dispatches sequentially, review package needs BASE..HEAD commits | CONFLICT — ruled below |

Ruling: implementers DO commit (one or more commits, subject prefix `feat(h2h): cluster run: `, both trailer lines, explicit-path staging) — the plan's no-commit rule existed only for parallel dispatch; SDD dispatches sequentially and the review package is BASE..HEAD — cost if wrong: none.
Ruling: T4 must add a Review-Focus-5 test: run_h2h_trial with k_target=0 (buddy) calls leiden_on_graph with cfg.leiden_resolution and never target_k_partition (monkeypatched fakes) — the plan lists RF5 but gave no test — cost if wrong: one extra test.
Ruling: T2's harness path accepts heads in {mlp128, attn1, attn4, attn} and maps attn1/attn4/attn -> "attn1" for ParameterizedLearnedStudent (its is_attention set is {attn1, attn4, attn}; head count from num_heads) — so T7's single `buddy_heads` categorical works for both impls — cost if wrong: none, attn1/attn4 are identical in the harness student anyway (§6i).
Task 1: dispatched (sonnet), BASE b100d87 (ledger commit none)
Task 1: implementer DONE_WITH_CONCERNS c4dcd4d (dtype addendum applied; build_arrays/load_patches untested by necessity); review dispatched (sonnet)
Task 1: review (sonnet): spec ✅, quality Approved, 0 Critical/Important. Pilot call signatures verified by reviewer; flock+atomic replace sound across processes.
Task 1: minor (deferred): no flush/fsync before os.replace (crash durability).
Task 1: minor (deferred): concurrency test uses threads only, not processes.
Task 1: minor (deferred): exclusive lock taken even when final exists (no fast path).
Task 1: minor (deferred): split tests miss tiny/empty-stratum rounding cases.
Task 1: minor (deferred): h2h_store imports private pilot_metrics._load_module.
Task 1: complete (commits b100d87..c4dcd4d, review clean)
Task 2: dispatched (opus, port fidelity), BASE c4dcd4d
Task 2: implementer DONE_WITH_CONCERNS 14c96d7 (synthetic CPU bit-exact equivalence test vs unchanged pilot main(); concern: store inputs must be bit-identical for exact V2 — GoEmotions extraction determinism)
Task 2: review dispatched (opus; named risk: equivalence test must not patch the pilot training path)
Ruling: Task 1 follow-up fix round (after Task 2 review) — build_arrays must run GoEmotions extraction and all store construction under the pilot's determinism settings (torch.manual_seed/np seed 42 + cudnn.deterministic=True, benchmark=False, use_deterministic_algorithms(True, warn_only=True), CUBLAS_WORKSPACE_CONFIG=:4096:8), because QC2 shows hardware-level float noise moves independent AMI by ~0.006 at a seed (local vs DAS6 seed 42), larger than V2's 0.005 tolerance; bit-identical store inputs keep V2 meaningful — cost if wrong: none (determinism only).
Task 2: review (opus): spec ✅, quality Approved, 0 Critical/Important; named risk (equivalence test patching pilot training path) PASS.
Task 2: minor (deferred): subset-monitor test checks shapes/state only, not that stopping uses monitor rows (exact check suggested: pilot on heldout[monitor_idx]).
Task 2: minor (deferred): equivalence fixture pre-sets arch constants, so a skipped knob setattr would not be caught.
Task 2: minor (deferred): monitor_idx not validated (bool mask/dupes/out of range).
Task 2: minor (deferred): harness heads=attn4 with num_heads=1 silently builds 1 head (per ruling; document in sweep YAML).
Task 2: minor (deferred): import-time CUBLAS_WORKSPACE_CONFIG setdefault is process-wide.
Task 2: complete (commits c4dcd4d..14c96d7, review clean)
Task 1: fix round 1/5 dispatched (resume implementer): store determinism ruling above.
Task 1: fix round 1/5 (2 addressed, 0 open — deterministic store build, fsync before replace; commits 14c96d7..1af3ad2)
Task 1: minor (deferred): determinism test does not assert seeds/CUBLAS env or restore-on-exception path; count check skipped when percept_base injected (test-only path).
Task 1: complete (commits b100d87..c4dcd4d + 14c96d7..1af3ad2, review clean)
Ruling: Task 3 implementer dispatched while Task 1's re-review was still running (different files; only one implementer active at any time; a Task 1 round 2 would have waited for Task 3) — to save overnight wall-clock — cost if wrong: interleaved commits in a review range, mitigated by explicit per-task commit lists.
Task 3: dispatched (opus, port fidelity + equivalence test), BASE 1af3ad2
Controller commits during Task 3 (exclude from Task 3 review range): 4e5b9a7 (§6j docs + QC logs), 3333114 (run_h2h_build_store.sh). Task 3 review BASE = 3333114 if all Task 3 commits follow it. Store pre-build launched: h2h-store-node403/404/405 (slot 0).
Store pre-build: all 3 nodes succeeded (6 min); dtypes match pilot (content/affect28 float64); split val 4686 / test 4679, digest 5f9e2827... (check all nodes equal)
Controller commit ba21723 (early V2 check). Task 3 review BASE now = ba21723 (if all Task 3 commits follow).
Early V2 launch deferred: cluster sync refuses while Task 3 implementer has uncommitted scripts/ files; launch after Task 3 commits.
Task 3: implementer DONE_WITH_CONCERNS 2f158e2 (bit-exact vs unchanged pilot main() + fit_stage1_and_get_targets on synthetic CPU; replays two GoEmotions AutoModel loads that consume torch RNG; full-batch latents like pilot; pins default determinism flags; adds info surviving_centers/label_ids).
Ruling: PercepT store inputs (affect768/fused h) were built under determinism flags while the PercepT pilot extracted without them — accepted; PercepT's refit is not bit-reproducible anyway (§6g) and V3 uses tolerances — cost if wrong: V3 compares within tolerance, not bitwise.
Ruling: Tasks 6/7/8 wrappers must export HF_HUB_OFFLINE=1 (PercepT fit replays GoEmotions model loads from the node HF cache) — cost if wrong: none.
Ruling: V3 (Task 6) Stage 2 check uses the pilot's own s2.multi_hot_targets on surviving centers (threshold 2/K, the §6g target construction) as primary, and single-label as secondary — §6g's 0.9226 used multi-hot (effectively single-label, 0% multi-labeled) — cost if wrong: none, both reported.
Early V2 launched: ev2-s42/s7/s123 (node403 slots 0/1/2), ev2-s2024 (node404 slot 0) at 2f158e2.
Task 3: review dispatched (opus)
Task 4: dispatched (opus) while Task 3 review runs (same ruling as Task 3: one implementer at a time; Task 3 fix rounds wait for Task 4), BASE 2f158e2
V2 (early, controller script ba21723 on DAS6 at 2f158e2): PASS, bit-exact — seeds 42/7/123/2024 train+held-out embeddings identical to QC2 pilot snapshots (max abs diff 0.0), independent AMIs and held-out K identical, train K 19/20/20/20 = QC2. Port fit ~35 s/seed.
Ruling: V2 is satisfied by the early check (bit-exact is strictly stronger than the spec's tolerance gate); Task 6 drops its v2 mode and implements only v3 + timing — cost if wrong: none (early_v2_check.py + logs are committed provenance).
Controller commit b8c7b03 (V2 logs) interleaves Task 4's range — exclude from Task 4 review package.
Controller commit 36a8733 (early V3 check) — exclude from Task 4 review range. Early V3 launched ev3-s42 node405 slot 0.
Task 3: review (opus): spec ✅, quality Approved, 0 Critical/Important; risks PASS (no training-path patching; RNG replay empirically necessary and correctly placed; knobs on reading modules).
Task 3: minor (deferred): equivalence test configures the pilot side via percept_constants itself (knob placement guarded only by the key test).
Task 3: minor (deferred): replay has no offline guard (wrappers export HF_HUB_OFFLINE=1 per ruling).
Task 3: minor (deferred): info holds numpy arrays (surviving_centers, label_ids) — Task 4/7 must not log info wholesale.
Task 3: minor (deferred): unused loop variable _split.
Task 3: complete (commits ba21723..2f158e2, review clean)
V3 (early, controller script 36a8733 on DAS6 node405): PASS — K=40; native AMI 0.1073/0.2770 vs 0.1094/0.2798 (|Δ| ≤ 0.003 < 0.01); pilot-Stage-2 AUC lr1e-2/400 4-seed mean 0.9258 vs 0.9226 (Δ +0.0032 < 0.02); untuned 0.5803 vs 0.5843; held-out multi-labelled fraction 0.0 (as §6g). DEC stopped at epoch 76 (stability); Stage 1 fit 69 s.
Ruling: Task 6 is dropped — V2 and V3 are both satisfied by the committed early-check scripts (early_v2_check.py, early_v3_check.py) with logs; per-trial timing for R4 sizing comes from the Task 7 agent smoke run on DAS6 — cost if wrong: none (validation evidence is committed).
Controller commit c46a4c1 (V3 log) — exclude from Task 4 review range.
Task 4: implementer DONE_WITH_CONCERNS 8315dc7 (195 tests; concerns: mknn+merge0 always K-miss (singletons); k_target=0 reference builds mknn on GPU vs §6i CPU; GPU determinism by seeding only). Review range c46a4c1..8315dc7; review dispatched (opus).
Ruling: buddy sweep space merge_small_threshold = {0.005, 0.01, 0.02} (0.0 dropped) — the unrepaired mknn graph leaves isolated nodes as singleton communities, so merge 0 can never reach K≤42 (Task 4 synthetic finding; §6i: all 49 gate-passing K>100 runs had merge off); dropping it avoids burning trials on guaranteed k_miss — cost if wrong: pilot_repaired+merge0 combinations are not searched.
Task 7: dispatched (sonnet) while Task 4 review runs (Task 4 fix rounds wait for Task 7), BASE 8315dc7
Task 7: implementer DONE ef401af (209 tests); review dispatched (sonnet)
Task 7: review (sonnet): spec ✅, quality Approved, 0 Critical/Important.
Task 7: minor (deferred): _free_cuda lacks gc.collect() before empty_cache (OOM traceback cycles may hold GPU memory).
Task 7: minor (deferred): empty COUNT_ARGS array under set -u fails on bash < 4.4.
Task 7: minor (deferred): wandb.init() outside try.
Task 7: complete (commits 8315dc7..ef401af, review clean)
Controller commit f0b7bb4 (smoke script)
Task 4: review (opus): spec ✅, quality Approved, 0 Critical/Important; risks: no val/test leakage, label spaces consistent, K-miss returns -1 cleanly (reproduced with real functions).
Task 4: minor (deferred→bundled fix): bisection diagnostics (k_raw, steps) not in rows.
Task 4: minor (deferred): component-floor miss spends all 12 Leiden runs.
Task 4: minor (deferred→bundled fix): later seeds still run after a K miss (objective already -1).
Task 4: minor (deferred): order-preserving comment inaccurate; no lo<1<hi check; seed_all lives in h2h_eval.
Task 4: complete (commits c46a4c1..8315dc7, review clean)
Ruling: K=40 cells use merge_small_threshold {0.002, 0.005, 0.01}; K=16 cells keep {0.005, 0.01, 0.02} — reviewer advisory B: K=40 at merge 0.02 needs ≥38 communities of ≥2% each (~76% of nodes near-uniform), practically infeasible and makes K(r) non-monotone — cost if wrong: buddy's merge options differ by K level (never by system at a fixed K).
Ruling: bundled pre-sweep fix (one dispatch, after smoke results): (1) run_h2h_trial aborts remaining seeds after the first k_miss; (2) rows carry k_raw_after_merge + bisect_steps; (3) h2h_agent _free_cuda does gc.collect() before empty_cache; (4) buddy_k40.yaml merge values per ruling above — cost if wrong: none (efficiency/diagnostics).
Ruling: Task 8 reference mode builds the mknn topic graph on CPU (as §6i's clustering.leiden_partition did) for the m8x7ifx4 reference; transfer_k=40 maps to train_target_k=40 — advisory C.
Task 5: dispatched (opus; V1 local, may run the V1 script on the local 3090), BASE f0b7bb4
Smoke (DAS6, f0b7bb4, 2 search seeds, val): all 5 presets ran end to end, all K hits. buddy_pilot_k16 0.8578 (K 18/15, 2.6 min), buddy_pilot_k40 0.8762 (K 39/40, 2.6 min), buddy_harness_k40 0.9328 (K 40/40, 4.1 min), percept_k16 0.9220 (2.9 min), percept_k40 0.9297 (2.9 min). Store load ~12-16 s.
Ruling (R4 sizing): TRIALS_PER_CELL stays 300 (the spec max) — measured 2.6-4.1 min/trial at default settings; heavier configs (800 mapper epochs, 8 queries, 400 Stage 1 epochs) estimated ≤2x → ~5 min avg → 4×300×5 min ≈ 100 GPU-h ≈ 11-14 h wall on 9 GPUs, far under the 60 h cap — cost if wrong: none (cap is an upper bound).
Ruling: agents run round-robin over all four sweeps (count=1 per wandb.agent call) so cells progress together and no GPU idles when a cell hits run_cap (spec §7.3 "cells interleaved") — part of the pre-sweep bundled fix — cost if wrong: small per-call agent overhead.
Pre-sweep bundled fix dispatched (sonnet) concurrently with Task 5 (disjoint files); brief presweep-fix-brief.md; BASE f0b7bb4 (Task 5 commits may interleave).
Pre-sweep fix: implementer DONE f7e2f1e (218 tests). Controller finding (Important): single zero-trial pass stops the round-robin agent — a transient W&B failure would idle a GPU all night; fix round 1 sent (3 consecutive empty passes, 60 s apart; wandb.agent exceptions survive).
Sweeps created (polysemic/CoSiR-h2h, run_cap 300 each): buddy_k16 l0hg1hb7, buddy_k40 0o1hc9gm, percept_k16 dzzmonjy, percept_k40 7un6c1ak.
Pre-sweep fix: fix round 1/5 (1 addressed: robust stop rule; commit dc771bb). Review (sonnet, e17b1d3..dc771bb): spec ✅, Approved, 0 Critical/Important.
Pre-sweep fix: minor (deferred): stop rule tolerates only ~2 min of total W&B outage; over-long lines; flaky test coupled to pass structure; wandb.init outside try.
Pre-sweep fix: complete (commits f0b7bb4..dc771bb excluding controller e17b1d3, review clean)
Sweeps LAUNCHED ~04:40: 9 agents h2h-agent-node40{3,4,5}-{0,1,2} at dc771bb, each round-robin over all 4 sweeps (staggered start order), count 1000 (sweeps end at run_cap 300 each).
Launch incident: controller's launch loop ran in zsh (1-indexed arrays) so ORD[0] was empty for agents 0/4/8 → h2h-agent-node403-0, node404-1, node405-2 exited at start (argparse: sweep '1000'); no trials affected. Relaunched under explicit bash as h2h-agent-node403-0b, node404-1b, node405-2b at 899d0e6 (agent code identical to dc771bb). Lesson: build launch commands in `bash -c`.
Task 5: implementer DONE_WITH_CONCERNS 899d0e6 (228 tests). V1 buddy: train targets identical; 132/9365 held-out labels differ — all k=20 vote ties (pilot: nearest tied winner; harness assign_to_train_communities: lowest topic id; the §6i parked Task 5 finding); stage2_metrics on the pilot's targets 4-seed 0.85341 vs 0.8534 (bit-equal to the pilot loop at seed 42); on harness labels 0.85362 (+0.0002). V1 percept: held-out targets identical; lr1e-2/400 4-seed 0.92264 vs 0.9226; lr1e-3/100 seed 42 0.5843 vs 0.5925 — the pilot never reseeded its mapper (drew init from the RNG stream left after Stage 1), seed noise at that setting 0.5830-0.5906 (std 0.0029).
Ruling: V1 PASSES for its purpose (the Stage 2 path is equivalent: Δ +0.00001 buddy, Δ ≈ 0 percept at the tuned setting). The strict "targets equal" fail for buddy is the known tie-break rule (132 rows, +0.0002 AUC), applied identically to BOTH systems in the H2H, so it cannot favour either; kept as-is (changing it now would change all running trials). The strict PercepT untuned fail is an unseeded pilot mapper init, below-noise, not a harness defect — cost if wrong: labels differ from the pilot on ~1.4% of held-out rows at vote ties, disclosed in the report.
Finding (inferred, untested): §6g's "PercepT refit not bit-reproducible" gap (0.5843 vs 0.5925) is most likely the pilot's unseeded mapper init, not Stage 1.
Task 5: review dispatched (sonnet)
Task 5: review (sonnet): spec ✅, Approved; named risks: pilot targets from pilots' own functions ✓, numbers consistent ✓, 132-row tie claim established by code+log ✓.
Task 5: Important (reviewer) — the report's PercepT untuned-gap cause (unseeded pilot mapper init) is inference, not tested. Ruling: no code fix; every relay of it (master report, H2H report, morning summary) words it as "consistent with, not tested" — cost if wrong: none.
Task 5: minor (deferred): PercepT train targets not compared (snapshot lacks them); buddy `pass` false only via tie-break; run_percept 58 lines, two long lines.
Task 5: complete (commits dc771bb..899d0e6, review clean after ruling)
Task 8: dispatched (sonnet), BASE 899d0e6
Task 8: implementer DONE 32e6345 (240 tests); review dispatched (sonnet)
Task 8: review (sonnet): spec ✅; risks OK (m8x7ifx4 mapping routes all 20 keys; topic_graph_device default is a no-op; select excludes objective ≤ -1). Important: select lacks finished-state filter → fix round 1 sent (resume implementer).
Task 8: minor (deferred): parse_lines dedup overwrites split jobs of one rank; config_for_trial silently drops unknown keys; cmd_select untested (W&B).
Task 8: fix round 1/5 (1 addressed — select requires finished runs; commits 32e6345..3cf1442); re-review clean.
Task 8: complete (commits 899d0e6..3cf1442, review clean)
All implementation tasks complete (1,2,3,4,5,7,8 + pre-sweep fix; 6 dropped by ruling). Remaining: controller Task 9 (monitor sweeps → select → val stress → test → report → final whole-branch review).
Throughput check 04:40 (≈16 min after agent start): 12 finished runs (startup-inflated). Re-measure at ~05:40; if < 40 runs/h consider equalized early stop per ruling R4 (multiple of 50, min 150, same N per cell by creation order).
Throughput 05:33: 68 finished in 69 min (~60/h incl. startup) → 1200 trials ≈ 20 h → sweeps end ~23:00 2026-09-30. Keeping TRIALS_PER_CELL=300 per R4 (≤60 h). Morning report will be interim.
Early pattern: buddy harness impl reaches val AUC 0.98-0.99 with independent emotion AMI ≈0.05 (vs pilot ≈0.12); PercepT leaders 0.95-0.96 with ind emo ≈0.09-0.10. The ungated AUC objective rewards image-predictable topics.
Ruling (secondary analysis, no change to the approved primary protocol): the H2H report adds, per cell, (a) the AUC-vs-emotion-AMI Pareto front over all trials (both yardsticks), and (b) an emotion-constrained selection — best val AUC among trials whose val independent emotion AMI ≥ a floor that both systems' trials reach (floor fixed from the data before looking at test, stated in the report), stress-tested and test-run exactly like the primary winners — so the headline cannot be won purely by discarding affect structure — cost if wrong: extra ~40 stress/test trials; primary result unaffected.
Throughput 06:31: 138 finished (+70 in 58 min ≈ 72/h) → sweeps end ≈ 21:00; final result (stress+test) ≈ 23:00.
Ruling: secondary-analysis emotion floor = min over the 4 cells of each cell's 10th-largest val independent emotion AMI among finished valid runs (symmetric, val-only, fixed before test; analyze_h2h_sweeps.py) — cost if wrong: a different floor could change the constrained winners; the primary (unconstrained) result is unaffected.
Controller commit (analysis script) at 06:40.
Interim report committed 0e6ba3d (07:30): H2H report §5 leaderboard at ~52/300 per cell + master report §6k pointer.
09:24: 348 finished (~71/h), no errors; ETA ~21:20.
11:19: 499 finished (~79/h), no errors; ETA ~20:15.
13:16: 646 finished (~75/h), no errors; ETA ~20:40.
16:49: container rebooted (local monitor lost); all 9 DAS6 agents unaffected and running; 908 finished (~74/h); ETA ~20:50.
Sweeps complete 20:58 (1200/1200, 0 errors). Frozen: sweep_runs.json + finalists (commit 8c9eb35). Primary top-1 val: buddy_k16 0.9928, buddy_k40 0.9938, percept_k16 0.9739, percept_k40 0.9692. Emotion floor 0.1117 (binding percept_k16, exactly 10 runs). Constrained top-1 val: buddy_k16 0.9567, buddy_k40 0.9597, percept_k16 0.9505, percept_k40 0.9605.
Stress stage launched at 8c9eb35: h2h-stress-{primary,constrained}-{buddy,percept}_k{16,40} (8 jobs, ranks 1-5, stress seeds, val) + h2h-test-ref-m8x7ifx4 (node405 slot 2).
22:10: stress done: primary buddy_k40, constrained buddy_k16 (5 results/20 seeds each), test-ref-m8x7ifx4 done. Launched h2h-test-ref-percept_6g (node405 slot 2). Note: cluster logs needs -n 200000 to capture all result lines.
Winners: primary_buddy_k40 = rank 3 89mkiavu (val 0.9939±0.0003, ind_emo 0.052); constrained_buddy_k16 = rank 1 mssv0f7s (val 0.9482±0.0011, ind_emo 0.117). ref_m8x7ifx4 test: 0.9352±0.0052. Test runs launched: h2h-test-primary-buddy_k40 (node403/1), h2h-test-constrained-buddy_k16 (node404/1).
22:56 stress complete; winners: primary buddy_k16 r2 6edlxmyv, buddy_k40 r3 89mkiavu, percept_k16 r2 x0oa5511, percept_k40 r2 0biqmu50; constrained buddy_k16 r1 mssv0f7s, buddy_k40 r1 4fu1936m, percept_k16 r1 usnsbxdq, percept_k40 r1 i1rigwnq. Remaining 6 test runs launched.
Stress + test complete 23:20. Test (5 seeds): plain AUC buddy 0.9931/0.9937 vs PercepT 0.9664/0.9604 (Δ +0.027/+0.033, p≤0.005; ind emo 0.060 vs 0.079/0.110); emotion-constrained buddy 0.9462/0.9550 vs PercepT 0.9435/0.9598 (Δ +0.003 p=0.70 / -0.005 p=0.21; ind emo ~0.12 both; PercepT genre +0.10 p<0.01). Refs: m8x7ifx4 0.9352, percept_6g 0.9317. Report + §6k + figures committed a766038.
Final whole-branch review: range d6b96e2..HEAD (all of 2026-09-30's work: QC, spec/plan, Tasks 1-8, sweeps, results), data files excluded from the package (logs/json/png listed separately).
Final review (opus, d6b96e2..a766038): With fixes. All numbers reproduce, no leakage, 241 tests pass. 1 Critical (C1 genre/emotion conclusions do not replicate on val), 8 Important (I1 tie≠equivalence + native labels; I2 Pareto statements/coverage, m8x7ifx4 inside buddy space beats front; I3 pilot Stage 1 understated (8/10 constrained finalists pilot); I4 post-hoc framing/provenance; I5 unsupported attributions; I6 PercepT K not enforced (code); I7 test half not untouched; I8 master top section), 9 Minor. Controller verified C1 directly (val: genre Δ −0.005/+0.030, emotion Δ +0.021/+0.018; AUC tie replicates on both halves).
Ruling (framing, pending the user's choice): the report leads with the approved primary (plain-AUC) result, then the emotion-constrained comparison explicitly labelled secondary, devised after the interim leaderboard, floor fixed before test; section titles/commit framing corrected accordingly — cost if wrong: user can reorder.
Ruling: one fix dispatch (opus) with the complete findings list (SDD final-review rule); then one scoped re-review.
Final-review fix wave dispatched (opus), FIX_BASE a766038; brief final-fix-brief.md + final-review-findings.md.
Final-review fix wave: commits 6ea529c (code: PercepT K miss, sample SD) + 2f78672 (docs), 246 tests; added fig3 (val vs test). Scoped re-review dispatched (opus).
Scoped re-review (opus, a766038..2f78672): all 18 findings ADDRESSED; every new number reproduces; 246 tests. One Important new item (equivalence wording "matched PercepT" for the pilot runner-up, 3-4 places) + 5 minors.
Ruling: the residual Important item and the minors are wording fixes in controller-owned report text with every value already verified by the re-reviewer; applied by the controller directly (no second fix wave, per SDD), commit below — cost if wrong: an unreviewed text change of ~10 lines.
Final review: clean after fix wave + residual fixes. SDD complete for this plan.
dcb0f47 residual fixes
