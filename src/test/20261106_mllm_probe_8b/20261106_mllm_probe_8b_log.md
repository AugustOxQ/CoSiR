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
- Node job `mllm-probe-8b-seed46` (commit ea77ad1, node404 GPU slot 0, started 2026-10-03 16:58 UTC). Data reached the
  node with `scripts/das6_sync_mllm_probe_8b.py` (19.3 GB: 8B snapshot, features, annotations, genre labels, the 5,224
  WikiArt images the seed-46 episodes use). The wrapper's in-job checks passed (snapshot complete, 0 of 5,224 images
  missing). Speed on node404: about 3.6 s per episode.
- **Deviation found and checked: transformers 5.16.1 on the node versus 5.6.2 locally** (every earlier local check,
  including the v2 rendering check and the timing smoke, used 5.6.2). Check (b33d223; job `render-check-8b-node`,
  CPU only, no scores): the 48 prompts of the first 4 seed-46 episodes per pair, both conditions and directions, rendered
  through the reranker's processor path under both versions. Result IDENTICAL on every field: input_ids SHA-256,
  sequence length, image-token count, image_grid_thw, pixel_values (bf16) SHA-256, letter token ids and the token before
  each letter, processor image settings and the episodes' SHA-256. Same weight snapshot on both machines
  (0c351dd01ed87e9c1b53cbc748cba10e6187ff3b). Remaining difference: the model's forward code and kernels of the two
  library versions, which can change logits by numerical noise but not the inputs; disclosed.
- Finished 2026-10-03 (1,800/1,800 episodes, 3.59 s per episode, wall 6,512 s, peak allocated GPU memory 18.9 GB);
  pulled with `cluster pull --tag mllm-probe-8b-seed46 --node node404` into
  `res/cluster_jobs/mllm-probe-8b-seed46/code/outputs/mllm_probe_8b_seed46/`. The node's seed-46 episode arrays are
  identical to the ones saved locally before the move. Runner SHA-256 9751e818… (the path-override version, ea77ad1),
  reranker SHA-256 7779b605… (unchanged, as pre-registered).

## Verdict (PREREGISTRATION §3; re-derived by the controller from `per_anchor.npz` with `cluster_bootstrap`)
| Pooled, 1,800 seed-46 episodes (1,507 anchor paintings) | Qwen3-VL-8B | CLIP cosine | 8B minus cosine [95% CI] |
|---|---|---|---|
| R@1 | 14.21 | 13.14 | +1.07 [0.17, 1.93] |
| Condition gain | 0.21 | 0.00 | +0.21 [−0.51, 0.94] |
| Either aspect candidate first (descriptive) | 28.21 | 26.28 | +1.93 [0.25, 3.51] |

**MLLM works (pre-registered rule): False.** The R@1 bound clears zero but the condition-gain bound does not. The 8B model
put an aspect-sharing candidate first more often than the cosine, but did not choose the conditioned aspect over the
other one. Per pair (descriptive): emotion×style R@1 +1.88 [0.42, 3.33], gain +0.38 [−0.91, 1.66]; emotion×genre
+0.88 [−0.63, 2.42], gain −0.04 [−1.29, 1.23]; style×genre +0.46 [−1.17, 2.04], gain +0.29 [−1.00, 1.59]. Swap 12.67.
Context (not a paired comparison; different episodes): the 2B v2 probe on seed 44 gave R@1 +0.22 [−1.13, 1.53] and gain
−0.53 [−1.68, 0.66].
