import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import checkpoint_probe as cp


def test_probe_selectivity_separable_data():
    rng = np.random.RandomState(0)
    X_pos = rng.normal(loc=2.0, scale=0.5, size=(60, 4))
    X_neg = rng.normal(loc=-2.0, scale=0.5, size=(60, 4))
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * 60 + [0] * 60)
    result = cp.probe_selectivity(X, y, seed=42, n_shuffles=10, n_folds=5)
    assert result["real_acc"] > 0.9
    assert result["z"] > 2.0


def test_probe_selectivity_random_data_has_low_selectivity():
    rng = np.random.RandomState(1)
    X = rng.normal(size=(120, 4))
    y = rng.randint(0, 2, size=120)
    result = cp.probe_selectivity(X, y, seed=42, n_shuffles=10, n_folds=5)
    assert abs(result["z"]) < 3.0


def test_load_checkpoint_roundtrip(tmp_path):
    emb_dir = tmp_path / "final_embeddings"
    emb_dir.mkdir()
    emb = np.random.rand(10, 16).astype(np.float32)
    sids = np.arange(10, dtype=np.int64)
    np.save(emb_dir / "embeddings.npy", emb)
    np.save(emb_dir / "sample_ids.npy", sids)
    loaded_emb, loaded_sids = cp.load_checkpoint(str(tmp_path))
    assert np.allclose(loaded_emb, emb)
    assert np.array_equal(loaded_sids, sids)
