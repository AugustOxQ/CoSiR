"""
Final-review follow-up for the buddy distance-mode ablation
(docs/reports/2026-08-24_buddy_distance_mode_ablation.md).

The report's Interpretation section originally only offered two thin hypotheses for the
observed training-level null ("the spectral embedding is robust to this distortion" /
"the effect is too small for 3 seeds to detect") without ever checking whether the actual
saved init tensors used to train the 6 real runs differ at all. This script closes that
gap: it loads the two real, saved buddy-init templates that trained the 6 real runs --

    res/CoSiR_buddy_distance_mode_ablation/redcaps_150k/mode_blend/template_embeddings/embeddings.npy
    res/CoSiR_buddy_distance_mode_ablation/redcaps_150k/mode_typed/template_embeddings/embeddings.npy

-- (both (150000, 16) float32, rank-normalised to [-1, 1] per dimension, per their
template_config.json) and reports:

  1. mean |delta| between the two templates, elementwise, over all (150000 x 16) entries.
  2. that mean |delta| as a ratio to the mean |embedding value| (i.e. how big the change is
     relative to the typical magnitude of the embedding vectors themselves).
  3. per-dimension Pearson correlation between the two templates' columns (min, max, and
     the full sorted list of all 16 -- cheap enough to just print all of them).

No training, no GPU -- these are two ~9.6MB .npy files.

Usage
-----
  python measure_init_diff.py
"""
import os

import numpy as np

BLEND_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..",
    "res/CoSiR_buddy_distance_mode_ablation/redcaps_150k/mode_blend/template_embeddings/embeddings.npy",
))
TYPED_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..",
    "res/CoSiR_buddy_distance_mode_ablation/redcaps_150k/mode_typed/template_embeddings/embeddings.npy",
))


def main():
    blend = np.load(BLEND_PATH)
    typed = np.load(TYPED_PATH)
    print(f"blend: {BLEND_PATH}  shape={blend.shape} dtype={blend.dtype}")
    print(f"typed: {TYPED_PATH}  shape={typed.shape} dtype={typed.dtype}")
    assert blend.shape == typed.shape, "templates must be directly comparable (same N, same embedding_dim)"

    delta = typed - blend
    mean_abs_delta = np.abs(delta).mean()
    mean_abs_val = 0.5 * (np.abs(blend).mean() + np.abs(typed).mean())
    ratio = mean_abs_delta / mean_abs_val

    print(f"\nmean |typed - blend|                 = {mean_abs_delta:.6f}")
    print(f"mean |embedding value| (avg of both)  = {mean_abs_val:.6f}")
    print(f"ratio (mean |delta| / mean |value|)   = {ratio:.4f}x")

    n_dim = blend.shape[1]
    corrs = np.empty(n_dim)
    for j in range(n_dim):
        corrs[j] = np.corrcoef(blend[:, j], typed[:, j])[0, 1]

    print(f"\nper-dimension Pearson correlation (blend[:,j] vs typed[:,j]), {n_dim} dims:")
    for j in range(n_dim):
        print(f"  dim {j:2d}: r = {corrs[j]:+.4f}")
    print(f"\n  min r = {corrs.min():+.4f}  (dim {int(corrs.argmin())})")
    print(f"  max r = {corrs.max():+.4f}  (dim {int(corrs.argmax())})")
    print(f"  mean r = {corrs.mean():+.4f}")


if __name__ == "__main__":
    main()
