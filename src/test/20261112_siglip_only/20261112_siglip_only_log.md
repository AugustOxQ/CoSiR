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
