import numpy as np
import torch

from scripts.buddy_percept_sweep.cache import FixedInputCache, RawInputs


def _fake_raw_loader_factory(call_counter):
    def _loader():
        call_counter["raw_calls"] += 1
        rng = np.random.default_rng(0)
        return RawInputs(
            train_content_raw=rng.normal(size=(40, 200)).astype(np.float32),
            heldout_content_raw=rng.normal(size=(8, 200)).astype(np.float32),
            train_affect=rng.normal(size=(40, 28)).astype(np.float32),
            heldout_affect=rng.normal(size=(8, 28)).astype(np.float32),
            train_emotion=["awe"] * 40,
            heldout_emotion=["awe"] * 8,
            train_genre=np.array(["landscape"] * 40, dtype=object),
            heldout_genre=np.array(["landscape"] * 8, dtype=object),
            train_patches=torch.zeros(40, 50, 512),
            heldout_patches=torch.zeros(8, 50, 512),
        )
    return _loader


def test_first_call_fits_pca_and_calls_raw_loader_once():
    counter = {"raw_calls": 0}
    cache = FixedInputCache()
    result = cache.get(content_pca_dim=10, raw_loader=_fake_raw_loader_factory(counter))
    assert counter["raw_calls"] == 1
    assert result.train_content.shape == (40, 10)
    assert result.content_pca_dim == 10


def test_same_pca_dim_reuses_cache_without_recalling_raw_loader():
    counter = {"raw_calls": 0}
    cache = FixedInputCache()
    loader = _fake_raw_loader_factory(counter)
    cache.get(content_pca_dim=10, raw_loader=loader)
    cache.get(content_pca_dim=10, raw_loader=loader)
    assert counter["raw_calls"] == 1


def test_different_pca_dim_refits_with_new_shape():
    counter = {"raw_calls": 0}
    cache = FixedInputCache()
    loader = _fake_raw_loader_factory(counter)
    first = cache.get(content_pca_dim=10, raw_loader=loader)
    second = cache.get(content_pca_dim=30, raw_loader=loader)
    assert first.train_content.shape == (40, 10)
    assert second.train_content.shape == (40, 30)
    # Raw loader is itself cached independently of PCA dim -- only the
    # PCA fit is redone, not the underlying feature extraction.
    assert counter["raw_calls"] == 1


def test_revisiting_pca_dim_reuses_earlier_fit():
    counter = {"raw_calls": 0}
    cache = FixedInputCache()
    loader = _fake_raw_loader_factory(counter)

    first = cache.get(content_pca_dim=10, raw_loader=loader)
    cache.get(content_pca_dim=30, raw_loader=loader)
    third = cache.get(content_pca_dim=10, raw_loader=loader)

    assert counter["raw_calls"] == 1
    assert third is first
    assert third.train_content.shape == (40, 10)
    assert third.content_pca_dim == 10
