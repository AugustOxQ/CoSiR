# 20261102 MLLM probe: setup and smoke log

## Problem
Early probe (plan Task 14, spec section 6): does an in-context Qwen3-VL-2B-Instruct reranker beat CLIP B/32 cosine on
seed-44 aspect episodes? Pre-registered rule: works iff compare(mllm, cosine, clusters=groups[anchor]) has a 95% CI
lower bound > 0 on both `r1` and `gain` (pooled).

## Setup
- Episodes: seed 44, `--n` per pair (default 300), pairs emotion__style (third genre), emotion__genre (third style),
  style__genre (third emotion), selection rows only (asserted), one shared PaintingValueIndex, validated; saved to
  `results/episodes_seed44.npz` (Ruling R4 layout) with SHA-256 in `probe.json`.
- Prompt: `src/eval/mllm_reranker.py::build_messages` (aspect never named). Condition 'a' shows P_A pairs as examples
  and P_B pairs as counter-examples, 'b' swaps (`ep.condition`). Each pair = image of the `*_img` row + caption of the
  `*_txt` row. Image path `/data/PDD/wikiart_proj/wikiart/<annotation image>`, annotation index = `data.sample_ids[row]`.
- Letter-position control: for every (episode, condition, direction) the 13 candidates are randomly permuted (seeded
  `default_rng([44, episode])`) before lettering; the 13 letter logits are written back to the episode's column order.
  This removes the confound that p_a sits at A and p_b at B. Permutations are stored in `probe_partial.npz`.
- Cosine baseline: CLIP B/32 features from `load_artelingo()`, NaN outside the selection rows, same episodes.
- Checkpoint every 50 episodes to `results/probe_partial.npz`; resume asserts the permutations match.

## Checks on real prompts (`inspect_prompts.py`)
- The 13 letters A..M are single tokens (ids 32..44).
- `apply_chat_template` accepts `{"type": "image", "image": <path>}` unchanged; no key adjustment needed.
- `AutoProcessor.from_pretrained(..., max_pixels=...)` is honoured in transformers 5.6.2 (it sets
  `image_processor.size.longest_edge`). A 1744x3000 image gives a 36x20 patch grid (180 tokens) with max_pixels set,
  against 188x108 (5,076 tokens) without it. Max pixels per image in the real prompts: 196,608 (i2t), 199,680 (t2i).
- i2t prompt: 9 images, 2,105 tokens (1,610 image tokens), 0.49 s. t2i prompt: 21 images, 4,126 tokens (3,765 image
  tokens), 0.96 s.

## Smoke (`--n 4`, 12 episodes, 48 prompts)
Per prompt 0.83 s (3.34 s per episode of 4 prompts); model load 13 s. No NaN scores. Pooled smoke numbers
(12 episodes, not informative): MLLM r1 12.5, gain -6.25; cosine r1 16.67, gain 0.0. Resume from the partial file works.
Projected full run (900 episodes, 3,600 prompts): about 50 min (t2i prompts are slower, so up to about 1 h).
