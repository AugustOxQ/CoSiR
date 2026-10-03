# 20261106 MLLM probe with Qwen3-VL-8B-Instruct: log

## Problem
Branch 2 test (spec §4): does an in-context Qwen3-VL-8B-Instruct reranker beat CLIP B/32 cosine on fresh seed-46
selection aspect episodes, under the rule of E3 PREREGISTRATION §9? Rules: `PREREGISTRATION.md` in this folder,
committed in de3bd45 before the 8B model scored anything.

## Setup steps
1. Model download: `hf download Qwen/Qwen3-VL-8B-Instruct` into `/data/SSD2/HF_home` (17 GB, 4 weight shards).
2. Runner: `src/test/20261102_mllm_probe/run_probe.py` gained `--model` (default unchanged) and provenance fields
   (runner and reranker SHA-256, peak GPU memory, wall time) in 238abf6. A 2B smoke (`--n 4`, seed 44) reproduced the
   default path (2.45 s per episode, model id unchanged in `probe.json` and in the resume fingerprint).
3. Reranker: the 8B checkpoint's `config.json` names `Qwen3VLForConditionalGeneration`, the class
   `src/eval/mllm_reranker.py` already loads; the reranker is unchanged (SHA-256 7779b605…).

## Timing smoke and the episode count (PREREGISTRATION §2; recorded before any seed-46 score)
- Command: `run_probe.py --n 4 --seed 44 --model Qwen/Qwen3-VL-8B-Instruct --out <scratchpad>/probe8b_smoke`
  (the 2B probe's 12 smoke episodes; their numbers are not read as evidence).
- Measured: 5.30 s per episode (4 prompts), model load 10.0 s, probe 73.6 s for 12 episodes, peak allocated GPU memory
  18.9 GB (fits the 24 GB RTX 3090; DAS6 not needed).
- Rule: 5.30 s × 1,800 episodes = 9,540 s = 2.65 h ≤ 6 h, so the run uses **600 episodes per pair (1,800 pooled)**.

## Run
- Command: `flock -n -o -E 75 /tmp/gpu0.lock run_probe.py --n 600 --seed 46 --model Qwen/Qwen3-VL-8B-Instruct
  --out src/test/20261106_mllm_probe_8b/results`
- Local start 2026-10-03 18:09 (`results/`, run log `run_probe_8b.log`): stopped by the controller at 18:23 after the
  150-episode checkpoint (5.36 s per episode), at the user's request, to free the local GPU for other projects. No
  verdict, summary or score was computed or read from the partial file (`results/probe_partial.npz`); it is kept
  unread as a record.
- Moved to DAS6 node404 (user's reservation). The node run starts fresh from episode 0 with identical rules, model,
  seed, episode count and code path; only data locations differ (environment-variable overrides; image paths therefore
  differ in the resume fingerprint, so the local partial could not be resumed there in any case). The node run's
  output is `outputs/mllm_probe_8b_seed46` in the job's code worktree, pulled back with `cluster pull --tag`.
- Results and verdict: to be added after the node run.
