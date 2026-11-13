# r29 run log: the SigLIP-only aspect adapter

Spec: `docs/superpowers/specs/2026-11-12-siglip-only-adapter-design.md`. Run chat "r29 run" (session c36e6513), night
mode. Outputs `/data/SSD2/cosir_r29/20261112_siglip_only/` (stage logs in its `logs/`). Decision lines: `decisions.md`.

## Run

- 04:05 Chat start. Queue idle, GPU free (RTX 3090, 33 MiB used). r21 run 4 has no GPU work planned (its reply), so
  r29 queues freely.
- 04:06 `r29_run_stage0.sh`: every gate record exists and is skipped; it ran the two diagnostic (e) inputs
  (`text_gate_i_a_new.json`, `groups_gate_i_a_new.json`) and merged `gates.json`. Commit 31507322.
- 04:07 r28's run reports are committed (bdabdd28), so r29's main chat opened (tab "r29 main", workspace "CoSiR r29").
- 04:07 Stage 1a started (`logs/run_stage1a.log`): K1 and K2 seed 0, their validation text scores, P1 in both forms,
  gate (iv) hash part, gates merge.
- 04:12 Codex (gpt-6.1-sol, high) re-derives SC1 from the rule and the data (brief `codex_logs/sc1_rederive_brief.txt`,
  code and output in `rederive/`), CPU only.
- End-time plan: stage 1 (1a, 1b, 1c, Table 8) about 3 to 4 hours, so done near 08:00; stage 2 about 6 GPU-hours
  plus CPU scoring, so it runs past 11:00 into the weekday daytime. The user allowed this for r29 (2026-11-13 02:19,
  `R29_ALLOW_DAYTIME=1` for stage 2); no other chat has GPU work queued.
- 04:21 Stage 1a done, no failed step (`logs/run_stage1a.log`). K1 trained 6.15 min, K2 5.97 min (local). Every gate
  in `gates.json` passes, gate (iv) included (hash part: the scorer's store hash equals the trainer's for both arms).
- 04:21 `r29_decide.py sc1`: text group, validation seed 0, r0: K1 (plain) 0.2552660252, K2 (prefixed) 0.2552826721.
  K* = K2 (`siglip1_prefix`, prefixed) by 0.0000167, a close call; the rule has no margin (D5: higher wins, only an
  exact tie goes to plain). P1 form (D20): plain, 0.1801105675 against prefixed 0.1794144870.
- 04:37 Codex's re-derivation (`rederive/r29rd_sc1.py`, own forward pass, own AP; 924 s on the CPU queue) gives the same
  pick and P1 form; every one of the 36 cells within 2.9e-8 of ours, groups within 1.7e-9 (tolerance 1e-6). SC1 holds.
- 04:36 Stage 1b started (`logs/run_stage1b.log`): K* seeds 1 to 4, validation exports and scores, picks, groups.
- 04:46 Codex (same session, resumed) built `rederive/r29rd_sc2.py`; on K0's 15 frozen validation exports it gives
  r28's recorded cells (AMI difference 0, mAP within 5.0e-8). The full SC2 run waits for 1b's exports.
- 04:48 Codex (same session) builds the 1c re-derivation (`codex_logs/c1_rederive_brief.txt`): validation only until
  our picks are committed; its held part runs in the controller's queue after the SC2 commit.
- 05:27 Stage 1b done, no failed step (`logs/run_stage1b.log`). K* seeds 1 to 4 trained 5.9 to 7.0 min each; image
  picks took 21.9 min on the CPU queue.
- 05:28 `r29_decide.py sc2`: level 0, fail. Five-seed means on validation: image 0.3882157161 (bound 0.3833128479,
  K0's best seed 0.3878696916: above it), text 0.2548565868 (bound m_T - r_T = 0.2573405749; K0's mean 0.2593512884).
  Per-seed text 0.2540 to 0.2558, all five below the bound. By D3 and §6, K* is an extra row; stage 2 does not run; 1c
  runs (D9) and Table 8 runs.
- 05:29 Codex's SC2 script runs on the CPU queue (`rederive/r29rd_run_sc2.sh`).
- 05:28 Table 8 (`r29_run_stage1.sh t8`, `logs/run_stage1t8.log`, 0.41 min GPU): SigLIP 1 text pass on K*'s caption
  form (prefixed) 1.138 ms per caption (repeats 1.132 to 1.145) against the Qwen3 pass 3.616 ms (r28); the three caption
  towers 0.017 ms. One CPU job (Codex's SC2 run, 8 threads) ran beside it.
- 05:42 Codex built `rederive/r29rd_c1.py` (1c: picks, ranges, 30 cells; from r17, r14 and r23's C13 modules).
  05:43 to 05:54 its replays and validation picks ran on the queue: replays of frozen older-round cells within 1e-6 (AMI
  2.5e-14, image mAP 3.6e-7, text 6.3e-7); picks in `rederive/r29rd_c1_picks.json`.
- 05:56 Codex's cache (25 GB) moved from the system disk to `/data/SSD2/cosir_r29/20261112_siglip_only/rd_c1_cache`;
  the link `rederive/r29rd_c1_cache` keeps the script's path (gitignored). On `.scratch/pending_deletions.md`.
- 06:08 Codex's SC2 re-derivation (`rederive/sc2_rederived.json`): level 0, fail, as ours. Per-seed groups within
  1.6e-8, cells within 7.8e-8, K0's m, r, best and lower identical (tolerance 1e-6). AMI clusters all 6,152 paintings and
  masks unlabelled rows at scoring (r17 §3.3). Its forward pass equals the stored exports in 99.8% of float16 elements.
  SC2 holds: K* is an extra row (D3); stage 2 does not run.
- 06:10 Cut: this chat passed 200k tokens. "r29 run 2" goes on with stage 1c
  (`docs/superpowers/handoffs/2026-11-13-r29-run-2-handoff.md`).

## Run 2 (chat "r29 run 2")

- 05:59 Chat start. Queue idle, GPU free. Run mode night.
- 06:00 Stage 1c started (`r29_run_stage1.sh c`, `logs/run_stage1c.log`): its order check passes (pass rule, prefix pick,
  text and image pick records committed). K* = `siglip1_prefix`, P1 form plain. Held export of the five seeds took about
  1 min.
- 06:01 Codex's held part queued on the GPU behind 1c's held text scoring (`rederive/r29rd_run_c1.sh held`, waits for
  the 15 held exports).
- 06:02 Codex's 1c validation picks (`rederive/r29rd_c1_picks.json`) against ours (`records/image_picks_siglip1_prefix.json`,
  `records/text_siglip1_prefix.json`): every pick the same (image told and untold `center+k10` for all aspects; text
  `center_k10` in eight cells, `center` for genre image to text). Ranges within 1e-6: AMI within 1e-14, image mAP within
  5.2e-8, text within 9.4e-7 (emotion text to text; its `center_k10` per-seed values differ by up to 9.4e-7, the
  expansion step on float16 against float32 tower rows).
- 06:16 Codex's held part done (`rederive/c1_rederived.json`, 9.0 min GPU queue): 30 cells. Its own outcome field marks
  the image mAP and text cells INCONCLUSIVE, because its replays of older cells differ by more than 1e-9 (its own flag;
  image 3.6e-7, text 6.3e-7, both within the spec's 1e-6). Its outcomes recomputed from its lo and r by §3.4's rule:
  14 of 15 WIN over P1 (style image mAP BELOW), 12 of 15 WIN or LEVEL over Q1 (the three emotion text cells BELOW).
- 06:17 Stage 1c done, no failed step (`logs/run_stage1c.log`): held text scoring 6.7 min GPU, text cells 0.3 min, image
  cells 10.1 min CPU.
- 06:18 C13 comparison (`r29_c1_compare.py`, `records/c1_compare.json`): every pick the same, every range within 9.4e-7,
  all 30 outcomes the same; 29 cells within 1e-6 in value, comparator value, d, lo, hi and r. One number over 1e-6:
  genre text to image against P1, upper interval end, ours 0.0682642562 against Codex 0.0682652749 (1.02e-6).
- 06:20 Cause (`r29_c1_cause_check.py`, `records/c1_cause_check.json`): Codex's interval code on our per-query APs gives
  our record within 6e-17, so the interval code agrees. The gap is in the per-query APs: the two forward passes (both
  float32, different code and batches) differ in about 1,150 of 61,744 queries per seed, up to 0.04 in a few queries
  (near-tied captions change rank). Cell means differ by 6e-8; one bootstrap percentile moves by 1.02e-6. No bug to fix,
  and the night chat does not choose a side or widen the tolerance: the cell is INCONCLUSIVE in the ladder (§3.4), with
  both sides' WIN kept in the record. It cannot change the SC3 level (below).
- 06:22 SC3 ladder (`records/SC3_siglip1_prefix.json`): level 1, "training adds over the plain embedding in 13 of 15
  cells". P1: 13 WIN, style image mAP BELOW (K*'s joined three-head vector 0.2872 against plain SigLIP 1 0.2962, d
  -0.0090 [-0.0107, -0.0074]), genre text to image INCONCLUSIVE (C13, above). Q1: 12 of 15 WIN; the three emotion text
  cells BELOW (t2i -0.0060, i2t -0.0229, t2t -0.0493). Level 2 fails, so level 3 does not hold. Counting the inconclusive
  cell as a WIN gives 14 of 15, the same level.
- 06:23 Validation diagnostics (`records/diagnostics_val.json`): (a) K* minus K0, image group cells -0.0009 to +0.0080
  (mean +0.0019), text cells -0.0245 (emotion t2t) to +0.0020 (mean -0.0045). (b) best epochs as K0's (style 1 to 3,
  genre 14 to 18, emotion 6 to 18). (c) caption tower over its own SigLIP 1 input, t2t r0: emotion +0.119, style +0.018,
  genre +0.057. (e) local minus node402 at seed 0 (gate (i)(a), Qwen3 caption input): image mean +0.0025 (genre AMI
  +0.0184), text mean -0.0001.
- 06:25 Report alternative (`r29_alt_told_vs_p1.py`, `records/alt_told_vs_p1.json`; exploration, decides nothing):
  image mAP against plain SigLIP 1 with K*'s own aspect head instead of the joined vector: WIN in all three (style
  +0.0588 [+0.0565, +0.0611]).
- 06:32 Diagnostic (d) (`records/diag_d_SC3_siglip1_prefix.json`, 489 s): K* minus K0 on test, paired intervals.
  Image (told head): AMI emotion +0.0033 [+0.0006, +0.0061], style +0.0057 [+0.0022, +0.0092], genre -0.0107 [-0.0209,
  -0.0013]; mAP emotion +0.0008, style -0.0005, genre +0.0002 (style and genre intervals cover 0). Text: emotion t2i
  -0.0023, i2t -0.0058, t2t -0.0191 [-0.0213, -0.0169]; style t2i -0.0040, i2t -0.0036, t2t -0.0029; genre t2i -0.0039,
  i2t -0.0022, t2t +0.0011 [-0.0025, +0.0044].
- 06:33 The one BELOW over P1 is inherited: K0's own cell in r17 SC7 (`20261103_new_scores/records/outcomes.json`,
  untold style mAP) is also BELOW, 0.2877 against plain 0.2962 (d -0.0085 [-0.0101, -0.0069]). The spec's prior (§3.1,
  "the current models win over the plain embedding in every cell") missed this cell.
- 06:34 Numbers sheet `paper_cells.md` written (`r29_ladder.py sheet`): the 1c cells, K0 beside them, Table 8; stage 2
  rows "not run".

## Run 3 (chat "r29 run 3"): reports

- 06:29 Chat start. Queue idle. Run mode night.
- 06:35 Auto report `docs/reports/auto/percept/2027-01-18_siglip_only.md` and user-read report
  `docs/user_read/2026-11-13_siglip_only.md` drafted; Claude's claim review recomputed six brief values from the records.
- 06:36 Three Codex checks (fresh sessions): the auto report's claims (gpt-6.1-terra, medium; 3 findings), the user-read
  fidelity check (gpt-6.1-terra, medium; 3 findings), a cold read (gpt-6-luna, medium). One fix wave applied.
- Correction to 06:23 above: diagnostic (b) is not "as K0's" for emotion. Genre equals K0's per seed (14 to 18), style
  is 1 to 3 for both; emotion K* 6 to 18 against K0 10 to 18 (seed 0: 6 against 12; seed 2: 6 against 17;
  `records/diagnostics_val.json`, key `b`).
