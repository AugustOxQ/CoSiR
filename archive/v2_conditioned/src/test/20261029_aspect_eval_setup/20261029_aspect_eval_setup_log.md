# 20261029 aspect eval setup (E0): log

## Problem
E1 to E11 need one audited evaluation stack for aspect episodes (value-disjoint, third-aspect control, condition gain,
clustered bootstrap), a CUB third aspect group, and a Qwen3-VL-Embedding-2B port that matches the official pipeline.

## Steps
1. Tasks 1 to 5: ArtELingo split recompute and genre join, episode builder and validator, metrics and bootstrap,
   agreement rule and scorers, CUB loader, held ledger, CUB third-aspect probe (`cub_third_aspect.py`).
2. Task 4: `reproduce_spike.py` re-scores the aspect spike's stored episodes with the new modules.
3. Task 6: `qwen_fidelity.py` (CLIP against cached features; Qwen against the official embedder on CUB images and on
   large ArtELingo images).
4. Task 7: report `docs/reports/auto/v2/2026-10-29_aspect_eval_setup.md`, figures in
   `docs/reports/assets/2026-10-29_aspect_eval_setup/`, spec sync (§5.2, §7).

## Results
- Spike reproduction (re-run for the report): CLIP only 11.127, SE beta 0.3 R@1 10.950, swap 16.248 against the stored
  11.13, 10.95, 16.25. REPRODUCED. 37 tests pass across the four stack modules.
- Splits equal stage (d); selection labelled share emotion 0.892, style 1.000, genre 0.815.
- CUB third aspect: `has_wing_color`, min(image 0.430, caption 0.470) minus majority 0.276 = +0.154; others +0.008 to
  +0.055. Features are CLIP ViT-B/32.
- Qwen fidelity: PASS on CUB and on large images after the preprocessing fix; i2t 48/50 on both (gated), cosine minimum
  0.9996. max_pixels 524,288 (official default 1,843,200).

## Issues
- Plan's z-score code turned NaN rows into finite zeros; fixed so they stay non-finite (misses).
- Validator did not check the third aspect's labels and was never tested for rejection; fixed with tamper tests.
- Qwen port let the HF processor resize; the official path pre-resizes with `qwen_vl_utils`. Large-image stage failed
  (i2t 47/50) until the port copied that preprocessing.
- Cached ArtELingo features are not unit norm (about 9.9); consumers must normalise (`EvalInputs` does).
- has_wing_color rests on 532 labelled dev images; the Qwen gate passed at exactly 48/50.
