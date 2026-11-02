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
- `res/r14_bs -> /data/SSD2/cosir_r14/20261102_backbones_sweep/outputs` created (plan step 7).
