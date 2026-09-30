"""THROWAWAY: does the condition loss learn the plan's original weak world (style amplitude 0.25) with more steps?

Controller check cited by the selection report's "Training budget" caveat. It reuses the synthetic style world
of ``src/test/test_train_factors.py`` (480 rows, 240 two-row paintings, 4 styles carried by a one-hot block of
amplitude 0.25 on top of 12 content dimensions) and compares naive-rule R@1 on 200 mined condition episodes
(``_naive_r1``) for the no-condition run ("none") and the condition-loss run ("cond") at 300 and 2,000 steps.
Synthetic data only; no ArtELingo row is read. CPU, about a minute. Run from the repository root:

    python src/test/20261016_factor_learning_grid/weak_world_check.py
"""
import contextlib
import dataclasses
import io
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src/test"))
torch.set_num_threads(4)
from scipy.sparse import csr_matrix  # noqa: E402

from src.train.condition_sources import CommunitySource  # noqa: E402
from src.train.train_factors import R3_CONFIG, train_factors  # noqa: E402
import test_train_factors as T  # noqa: E402


def world(amp, seed=7):
    rng = np.random.default_rng(seed)
    paint = np.arange(480) // 2
    style = rng.integers(0, 4, 240)[paint]
    content = rng.standard_normal((240, 12))[paint]
    onehot = np.eye(4)[style] * amp
    img = np.hstack([content, onehot]) + 0.05 * rng.standard_normal((480, 16))
    txt = np.hstack([content + 0.3 * rng.standard_normal((480, 12)), onehot]) + 0.05 * rng.standard_normal((480, 16))
    left = np.arange(0, 480, 2)
    graph = csr_matrix((np.ones(480, dtype=np.float32), (np.concatenate((left, left + 1)), np.concatenate((left + 1, left)))), shape=(480, 480))
    return img.astype(np.float32), txt.astype(np.float32), graph, paint, style


for amp in (0.25,):
    img, txt, graph, paint, style = world(amp)
    source = CommunitySource(style, np.arange(480), min_group_rows=20)
    for steps in (300, 2000):
        base = dataclasses.replace(R3_CONFIG, num_factors=8, epochs=steps, batch_size=64, painting_batches=True, condition_episodes_per_step=16)
        out = {}
        for name, lam in (("none", 0.0), ("cond", 1.0)):
            h = {}
            with contextlib.redirect_stdout(io.StringIO()):
                model, _, _ = train_factors(img, txt, graph, dataclasses.replace(base, lambda_condition=lam), device="cpu", group_ids=paint,
                                            condition_source=source if lam > 0 else None, history=h, log_every=max(steps // 4, 1))
            out[name] = T._naive_r1(model, img, txt, source, paint)
            if lam > 0:
                out["cond_loss"] = [round(x, 3) for x in h["condition_loss"]]; out["tau"] = [round(x, 3) for x in h["tau"]]
        print(f"amp {amp} steps {steps}: {out}", flush=True)
