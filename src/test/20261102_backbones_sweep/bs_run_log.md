# r14 run log (backbones sweep)

Run chat "r14 run", started 2026-11-02 07:27 (Amsterdam). Spec: `docs/superpowers/specs/2026-11-02-backbones-sweep-design.md`.
Handoff in: `docs/superpowers/handoffs/2026-11-02-r14-run-handoff.md`. Run mode: night. Job tags are UTC (cluster CLI);
times here are Amsterdam.

## Node402

- 07:28 `cluster sync --node node402 --allow-any-branch`: HEAD 5f28b2b ("chat records"; code = 4519762, the reviewed
  state). The marker check was skipped on purpose: the last node-code change (fix A, 2284e2e, merged at 650f346) carries
  no "cluster run" subject, and no empty commit is made for the marker (handoff §4).
- 07:28 0(viii): `20261102-062858-5f28b2b`, slot 0.
- 07:29 prep: `20261102-062941-5f28b2b`, slot 1.
- 07:30 checks of part a for siglip2, siglip1, clip_l14: all ready (siglip1 and clip_l14 synced their Affection stores).
- 07:32 part a SigLIP 2: `20261102-063236-5f28b2b`, slot 2.
- 07:38 0(viii) succeeded, pulled (verified). c* seed 1 on node402 against the 3090: AMI difference emotion −0.0017,
  style +0.0008, genre +0.0024 (no gate). One full c* run (three heads, training plus export) took 444 s on the A6000.
- 07:39 part a SigLIP 1: `20261102-063850-5f28b2b`, slot 0.
- 07:50 prep succeeded ("BS_NODE prep done": the photo graphs k20 to k160 ready). Pull copied 603 MB but was not
  verified: the running part a jobs write to the same results folder (`node/units.jsonl`). Pull it again once the node
  is quiet.
- 07:51 part a CLIP L/14: `20261102-065108-5f28b2b`, slot 1. The sweep workers start as part a jobs free their GPUs.

## Local

- 07:34 stage 0 chain (one sequence, since `gate.json` is merged per process): CPU gates iii, vi, vii, x, xi, xii now;
  then, after the extraction queue's DONE line, `bs_run_gpu.sh` (i, iv, v, ix under the lock), then gate ii.
- About 08:03 the extraction queue ended: "QUEUE DONE ok in 11601s" (check step rc 0).
- 08:05 CPU gates so far all rc 0: iii, vi (5 items), vii-matched, vii-sha, vii-removals (828 s), x, xi.
- 08:05 part a timing on the A6000: X-c* about 516 s per seed, X photo about 215 s; part a ends about 08:35 to 08:55.
  Sweep estimate: 45 photo runs × 215 s + 120 knob runs × about 500 s ≈ 19 GPU-h, about 6.5 h on 3 GPUs (ends about
  15:30 today, before 11-04 18:06).
- 08:12 gate xii done (CPU part: 2,396 s in all). 08:12 to 08:25 GPU gates under the lock: i (six stores), iv (390 s),
  v (194 s), ix (c*, photo): all rc 0. 08:26 gate ii rc 0. **Stage 0: 26 of 26 gates pass, none failed or pending.**
  `status.json` waits for the three part a results and stage 2.
- 08:27 `exports-qwen`: photo_qwen and cstar_qwen EmoSet exports, rc 0, 18 s, none missing.
- 08:32 part a SigLIP 2 succeeded: 10 of 10 units ok, first attempt each (`part_a_siglip2.json` status ok). Pulled,
  verified, 2.96 GB.
- 08:33 sync of c5d7673 (log only; the launch refused the older node commit), then sweep worker 1:
  `20261102-073350-c5d7673`, slot 2.
- 08:35 `exports-part-a --enc siglip2` on the local CPU.
- 08:35 SigLIP 2 exports rc 0 (held, Affection photo, EmoSet photo; 32 s, none missing); stage 1 SigLIP 2 started.
- 08:36 part a SigLIP 1 succeeded: 10 of 10 ok. 08:39 sweep worker 2: `20261102-073909-c5d7673`, slot 0. Pulled,
  verified. 08:40 `exports-part-a --enc siglip1`.
- 08:41 SigLIP 1 exports rc 0 (46 s, none missing).
- 08:46 stage 1 SigLIP 2 rc 0 (640 s): ranges, ArtELingo retrieval pick, Affection ranges and pick; record committed
  (1493b87) before any held number. 08:46 stage 1 SigLIP 1 started.
- 08:51 part a CLIP L/14 succeeded: 10 of 10 ok. **Part a: 30 of 30 units ok, each on its first attempt.** Pulled,
  verified. 08:52 sync of 317ec92 (log), sweep worker 3: `20261102-075201-317ec92`, slot 1. 08:52
  `exports-part-a --enc clip_l14`.
- 08:53 CLIP L/14 exports rc 0 (44 s, none missing).
- 08:59 stage 1 SigLIP 1 rc 0 (736 s), record committed (a6af0f4). 08:59 stage 1 CLIP L/14, then stage 2, in one
  sequence (both commit records).
- 09:10 stage 1 CLIP L/14 rc 0 (655 s), record committed. **Stage 1 records of all three encoders committed** before
  any held number. Stage 2 started.
- 09:12 sweeps: 26 photo-sweep units done of 45 (about 200 s each), all ok.
- 09:02 C13: Codex job (gpt-6.1-sol, high) writes `bs_c13.py` (derive, compare) from the spec's rule and the frozen
  protocol code of r10 to r13, without the round's scoring code (r13's `pf_rederive.py` as the model). The full derive
  and compare run under this chat after stage 3. Brief: the run chat's scratchpad `c13_brief.txt`.
- Note on the 11:00 rule: the scoring, C13 and part b steps run on the local CPU (8 threads of 32) during the day, as
  the spec's budget plans (CPU about 4 h; report due 11-04 23:00). No heavy local GPU work runs in working hours.
- 09:23 C13 code back from Codex (23 min; 5h window 18 points, weekly about 3): `bs_c13.py` and `test_bs_c13.py`, 30
  tests pass; smoke compares of the near-copy sets (99 items) and the Affection validation ranges and picks (175
  items) match, 0 mismatches. Its compare also checks the full set of cell identities. Committed 79410bb.
  Codex's estimate for the full derive: 2 to 4 h on the CPU; compare 5 to 15 s.
- 09:26 stage 2 rc 0 (945 s); `records/stage2_emoset.json` committed (8cca6f1) before any EmoSet test number.
  `status.json` written: every encoder's c* and photo runs ok, stage 0 ok on both paths; part c stage 0 and stage 2 ok.
- 09:26 sweeps: 32 photo units of 45 done, none failed; the knob sweep has not started.
- 09:27 the chat cut at the context meter (200k); stage 3 and the C13 derive start in the next run chat.
- `res/r14_bs -> /data/SSD2/cosir_r14/20261102_backbones_sweep/outputs` created (plan step 7).

## Run chat 2 ("r14 run 2", from 09:27)

- 09:28 load 2.8 (15-min mean 20). Started in the background: `bs_run_cpu.sh stage3` (log
  `OUT/stage3_run2.log`) and the full `bs_c13.py derive` (CPU, 8 threads; log `OUT/c13_derive.log`).
- 09:29 node402: the three sweep workers running (slots 0 to 2); watchers started (`--stall-minutes 60`).
- 10:19 stage 3 rc 0 (3,060 s): the runner committed `records/status.json` (60c35d0), then the cells of all three
  encoders (ArtELingo clustering 390 to 520 s each, retrieval about 120 s, Affection about 90 s) and part c (EmoSet,
  1,098 s). Four cells files; hash map in `OUT/stage3/c13_sha256.json`. No verdict yet.
- 10:19 C13 derive: SigLIP 2 and SigLIP 1 sections done, CLIP L/14 started (75 clustering partitions so far).
- 10:57 C13 derive rc 0: 61 cells, partial=False, 5,334 s (CPU, 8 threads).
- 10:58 first full compare: 598 mismatches, max abs diff 1.1e-16. All in the compare adapter, none numeric: the round
  names SC3's comparator `matched_M2` (C13 `M2`) and Affection SC6's teacher `teacher_artwork` (C13
  `teacher_emotion_focus`, the same r12 artwork-wording view; `bs_affection.py` docstring); the stage 2 record keeps the
  descriptive arms under `descriptive` and the wording scores under `teacher`. Fixed in `compare` only (derive
  untouched; change log `.claude/20261102_log.md`); 30 C13 tests pass.
- 10:59 compare again: **items 2,676, mismatches 0, max abs diff 1.1e-16** (bar 1e-6). All ten hashes equal the stage 3
  map. `records/c13.json` written (pass true).
- 11:00 `decide` rc 0: `records/verdicts.json` committed (3e6ff30). Compare again: 45 mismatches, all adapter layout of
  `outcomes.json` (verdicts under `paintings`/`photos.<enc>`, SC6 claims as `sc6.<enc>` rows, the two cell names), none
  numeric. Fixed in `compare_outcomes` only; 30 C13 tests pass.
- 11:01 compare: **items 2,768, mismatches 0** (61 written outcomes, 24 SC6 claims, 6 encoder verdicts, part c).
  `c13.json` gets an `outcomes_compare` note (its hash map is unchanged).
- 11:01 early checks while the sweeps run (the runner's stage 4 and stage 5 run everything again later):
  `bs_partb.py ablation` (local r10 data only) and ten of the twelve diagnostics (all but `knob_curves` and
  `photo_length`, which need the sweeps); logs `OUT/stage4/logs/partb_ablation_early.log`, `diag_early_<item>.log`.
- Codex ledger row of the C13 job updated (claude-config 444d7d4).
- 11:47 early diagnostics: eight of ten ok (r0_rows 959 s, emoset_per_class_ap 810 s, held_spreads 852 s,
  r_e_vs_spread 126 s, the rest seconds). `photo_minus_cstar` and `emoset_uncut` failed alone (`ac_cluster` imported
  before `ctx()` puts its folder on the path; an `all` run is not hit). Fixed (import order; change log), rerun.
- 11:49 node402: photo sweep 45 of 45 units done; knob sweep 46 of 120 (about 20 units an hour on 3 workers): end
  about 15:30. Early ablation sheet: 3 of 8 removals done (about 500 s each).
- 11:55 part b spot re-derivation (spec §3.4, C13): five rows drawn before any sheet exists, `default_rng(14)`, in this
  order: knob `k2=16`, `k11=80` (of 39), photo `lr=0.003` (of 14), ablation `content_correction`, `relational` (of 8).
  Codex re-derives them from the stored partitions and per-query AP after stage 4.
- 11:59 the two fixed diagnostics rc 0 (photo_minus_cstar 342 s, emoset_uncut 305 s): ten of twelve ok early.
- 12:09 early ablation sheet rc 0 (4,063 s; 385 to 514 s per removal). Held relational term, genre AMI −0.0440
  [−0.0512, −0.0372], as report 12-28. The runner's stage 4 writes it again.
- 14:56 sweep worker 1 (`20261102-073350-c5d7673`) succeeded: 59 units, 0 failed. Worker 3 (`20261102-075201-317ec92`)
  ended "failed": 48 units ok, 1 failed: knob `k11=20` seed 1, twice, `AssertionError` in `rb_student.load_scale`:
  `CT_FULL/d2/graphs/manifest.json` missing on the node. Cause: `k11=20` is the base graph (20 neighbours), so it gets
  no graph patch and reads the claim test's own graphs
  (`/project/CoSiR/src/test/20261020_claim_test/outputs/stage2/full/d2/graphs`, 8.9 MB, gitignored), which
  `bs_node_inputs.json` does not list; every other knob value reads graph stores built in the run's output folder.
  A missing node input, not a training failure. Seeds 2 and 3 of `k11=20` had not run yet (161 units ok, 1 failed at
  15:00; worker 2 ended at 14:59 with 55 units, 0 failed: 162 of 165 ok). Next chat: add the folder to the node inputs, sync, rerun the `k11=20` units
  (spec D21: rows not done by 11-04 18:06 are "not run"; part b enters no verdict), as a deviation in the report.
- 15:00 the chat cuts at the context meter (200k).

## Run chat 3 ("r14 run 3", from 15:00)

- 15:02 node402 quiet (3 free GPUs, env ok). k11=20 fix: new node input `ct_full_d2_graphs` (the CT's
  `stage2/full/d2/graphs`, 8.9 MB, 4 files; group `knob_base`, added for the knob kind), its `cluster.conf` DATA_MAP
  entry, and `bs_node.py retry-failed` / `sweep --retry-failed`, which moves `failed.json` to `failed_attempt1.json` and
  the stale claim to `.claims_retired/` (nothing deleted). 340 round tests pass. Commits b280591, 91d79c3 ("cluster
  run"); change log `.claude/20261102_log.md`.
- 15:05 to 15:07 pulls, all verified, marked, code copies removed: sweep workers `20261102-073350-c5d7673`,
  `20261102-073909-c5d7673`, `20261102-075201-317ec92` and the prep job `20261102-062941-5f28b2b` (its earlier pull was
  not verified). Local: knob 117 units with `metrics.json`, photo 45.
- 15:07 local `bs_node.py retry-failed --kind knob photo`: moved `knob/k11=20/failed.json` and the claim
  `knob__k11=20__s1` aside (so `bs_partb` does not read the row as "not run" after the rerun's pull).
- 15:08 `cluster sync` (91d79c3); `cluster check --sync-data`: the graphs copied to `/local/wding/r14_in/ct_full_d2_graphs`,
  all four gates yes. 15:09 launch `20261102-140830-91d79c3` (slot 0): `sweep --kind knob --retry-failed`, one
  worker, the three k11=20 units.
- 15:09 early photo and knob sheets on the local CPU (`bs_partb.py photo`, `knob`; logs `OUT/stage4/logs/partb_*_early.log`),
  for timing and as the layout the spot re-derivation compares against; stage 4 writes all three again after the pull.
- 15:12 early photo sheet rc 0 (174 s). The early knob sheet (15 of 39 rows in 15 min, about 40 min in all) was stopped
  at 15:27, since it started before the k11=20 pull; stage 4 writes it again.
- 15:25 `20261102-140830-91d79c3` succeeded: `retry-failed` moved the two items aside, then `knob__k11=20__s1` to `s3`
  ok on the first attempt (282, 272, 263 s). 15:26 pulled, verified, marked, code copy removed. **Sweeps complete:
  knob 120 of 120 units, photo 45 of 45; no `failed.json` left** (the row's first record is kept as
  `failed_attempt1.json`).
- 15:15 Codex job (gpt-6.1-sol, high): the part b spot re-derivation, `bs_c13_partb.py` (derive, compare), separate code
  from the rule and frozen r10 to r13 code, without the round's sheet and scoring code (5h window 97% left at 15:06).
- 15:28 `bs_run_cpu.sh stage4` (the three sheets; log `OUT/stage4_run3.log`).
- 15:27 Codex back (12 min; 5h window 12 points, weekly about 2): `bs_c13_partb.py`, `test_bs_c13_partb.py` (12 tests),
  imports only frozen r10 to r13 code and `bs_paths`; committed c7c90ea. 15:29 to 15:49 full derive, the five rows
  (knob about 1 min each, ablation 722 s and 466 s).
- 15:50 early compare (knob sheet not yet rewritten): 313 items, 26 mismatches, all the ablation rows' validation mAP
  under the read-out (centering + k = 10). The sheet uses r11's cross-fit mAP (`rt_readout.crossfit_all`) for that
  point value; the derive scored it in-sample. A definition, not a numeric slip: the frozen r11 protocol is the
  cross-fit. Every other item matches (AMI both splits, R0 mAP, held read-out mAP, the photo row). One Codex fix round
  by resume (15:52).
- 15:52 Codex fix round (resume): validation read-out mAP by r11's `rt_readout.crossfit_all`; 12 tests; committed
  3c3c7e4. 15:53 to 16:14 full derive again (1,301 s).
- 16:07 stage 4 knob sheet rc 0 (2,448 s): **39 of 39 rows done**, k11=20 included. 16:11 photo sheet rc 0 (209 s).
- 16:15 compare: 0 numeric mismatches, but the adapter did not know the knob sheet's layout (written after Codex's
  job), so the knob cells were unpaired. Fixed in `adapt_sheet` only (change log); 72b499b.
- 16:42 stage 4 rc 0 (4,534 s; ablation 1,877 s): `stage4_status.json` ok, three sheets, none missing. Held relational
  term, genre AMI −0.0440 [−0.0512, −0.0372], as report 12-28.
- 16:43 final compare against the three final sheets: **items 429, mismatches 0, max abs diff 5.1e-14, nothing
  unpaired**. Record `records/c13_partb.json` (sheet and derive hashes).
- Node402: no r14 work left; nothing runs there (3 GPUs free at 16:17).

## Run chat 4 ("r14 run 4", from 16:46)

- 16:46 load 1.6, 52 GB free memory. 16:47 `bs_run_cpu.sh stage5` in the background (all twelve diagnostics; log
  `OUT/stage5_run4.log`). Codex 5h window 76% left (report job needs 14%).
- 17:11 stage 5 rc 0 (1,499 s): **all twelve diagnostics ok** (knob_curves and photo_length now too); `OUT/diag/`,
  figure `OUT/diag/knob_curves.png`. 17:13 Codex job (gpt-6.1-sol, medium): the auto report draft.
