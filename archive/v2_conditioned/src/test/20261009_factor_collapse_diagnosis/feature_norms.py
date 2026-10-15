"""Read-only context: how much of each CLIP feature vector is a common mean direction."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.data.artelingo import load_artelingo  # noqa: E402
from src.data.splits import grouped_split, leakage_groups  # noqa: E402

data = load_artelingo()
train = grouped_split(leakage_groups(data.paintings, data.img_features), (0.7, 0.1, 0.2), 42).train
for name in ("img_features", "txt_features"):
    x = getattr(data, name)[train].astype(np.float64)
    norms = np.linalg.norm(x, axis=1)
    mean_sq = float((x ** 2).sum(axis=1).mean())
    centered_sq = float(((x - x.mean(axis=0)) ** 2).sum(axis=1).mean())
    print(f"{name} (train rows): row norm mean {norms.mean():.3f} min {norms.min():.3f} max {norms.max():.3f}; "
          f"mean-vector norm {np.linalg.norm(x.mean(axis=0)):.3f}; "
          f"share of mean squared norm in the common mean vector {1 - centered_sq / mean_sq:.3f}")
