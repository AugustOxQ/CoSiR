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

## Fix round 1 (written BEFORE any v2 verdict exists)
Review finding: the Qwen3-VL chat template joins text parts with no separator, so the v1 prompt rendered as
`Candidates:A. <caption>.B. <caption>` and `...one letter.Example pairs:Pair 1: image...`. The tokenizer then merges the
label with its neighbours (`':A'`, `'.B'`), and for the 35% of captions without final punctuation the letter fuses with
the last word, so the letter the model reads is not the clean letter token we score. This biases the MLLM downward.
The prompt fix (explicit newlines, wording otherwise identical, aspect still unnamed) is justified by the rendering
alone, not by any result: no v2 number existed when it was made. The v1 full run (`results/`, no line breaks) finishes
as a pre-fix record and its result will be reported alongside v2 as the pre-fix run.
Other changes in this round:
- Letters are scored in fp32 from the last hidden state (`lm_head.weight[letter_ids].float() @ h.float()`), no
  full-vocabulary logits; this removes bf16 ties among the 13 letter logits.
- Resume fingerprint stored in the partial: per-pair episodes SHA-256, model id, max_pixels, SHA-256 of INSTRUCTION,
  of one rendered sample prompt per direction and of the permutations; asserted on resume. Partial is written
  atomically (temp file + rename); a complete partial skips model loading.
- `unpermute` helper with a CPU test (stub logit = shown column returns arange(13)).
- Rendering check (`inspect_render.py`): in both directions the 13 labels are the clean ids 32..44, each preceded by
  a token containing a newline (id 198 `\n`, 510 `:\n`, 624 `.\n`) and followed by `.`.
- fp32 check (`check_fp32.py`, 8 real prompts, both directions): max abs difference to the bf16 logits 0.062; the argmax
  is identical in 8/8; the full ranking is identical in all 5 prompts without bf16 ties, and differs only in the 2 of
  the 5 tie prompts where bf16 ties exist (fp32 breaks them).
- Smoke v2 (`--n 4`, `results/smoke_v2/`): 2.58 s per episode (4 prompts), no NaN; re-running on the complete partial
  skips model loading. Smoke numbers are 12 episodes and not informative.

## Post-hoc (final-review fix wave, 2026-10-03; not pre-registered): letter preference and probe resolution

Descriptive only, outside the decision map. Script `posthoc_letter_bias.py` (CPU, stored `probe_partial.npz` and
`probe.json`), output `results/posthoc_letter_bias.json`. The top-scored letter of each ranking is recovered through the
stored permutation (letter j showed column perms[j]).

- v2 (fixed prompt): 3,600 rankings, no top-score ties. The top score fell on letter C in 626 rankings (17.4%), on A in
  138 (3.8%), against 7.7% for a uniform choice (chi-square 857.9, df 12). v1 (pre-fix, bf16): C 14.4%, G 4.2%,
  94 top-score ties.
- Because the candidates were permuted at random before lettering, the preference cannot favour the target over the
  other aspect's candidate in expectation; it adds noise to both rates.
- Resolution: the pooled paired gain interval (MLLM minus cosine) has a half-width of 1.17 points in v2 (1.15 in v1),
  so a pooled gain had to exceed about 1.2 points for its interval to clear zero; 80% power needs about 1.7 points.
- Only Qwen3-VL-2B-Instruct was probed. Spec §6 allows the 8B model "if time allows"; it was not tried.
