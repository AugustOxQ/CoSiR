import numpy as np

from src.train.pseudo_partitions import build_episode_bank, kmeans_partition


def test_kmeans_partition_scope_and_bank():
    rng = np.random.default_rng(0)
    n_paint = 3000
    groups = np.repeat(np.arange(n_paint), 2)
    feats = rng.normal(size=(2 * n_paint, 8)).astype(np.float32)
    rows = np.arange(0, 2 * n_paint - 200)
    p1 = kmeans_partition(feats, rows, k=8, seed=0)
    assert (p1[rows] >= 0).all() and (p1[len(rows):] == -1).all()
    p2 = kmeans_partition(feats[:, ::-1].copy(), rows, k=8, seed=1)
    p3 = np.repeat(rng.integers(0, 8, n_paint), 2); p3[len(rows):] = -1
    bank = build_episode_bank({"x": p1, "y": p2, "z": p3}, groups, rows, n_per_pair=20, seed=2, min_paintings=5)
    assert len(bank.anchor) == 60 and np.isin(bank.rows(), rows).all()
