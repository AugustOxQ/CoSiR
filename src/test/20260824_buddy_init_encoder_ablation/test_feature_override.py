"""
Test: TrainableEmbeddingManager's buddies init accepts a feature_override triple,
bypassing FeatureManager entirely (Experiment 8 — buddy-init encoder-pair ablation,
docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md).

Run:
    python src/test/20260824_buddy_init_encoder_ablation/test_feature_override.py
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np
import torch

from src.conditional_buddy.compute_buddies import compute_buddy_init
from src.utils.embedding_manager_nocache import TrainableEmbeddingManager

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_HALF = torch.cuda.is_available()
TMP_DIR = "/tmp/test_feature_override_embeddings"


def _synthetic_pair(n=40, dim=32, seed=1):
    rng = np.random.default_rng(seed)
    img = rng.normal(0, 1, (n, dim)).astype(np.float32)
    txt = rng.normal(0, 1, (n, dim)).astype(np.float32)
    # Deliberately NOT range(n) - CLAUDE.md's sample-ID-consistency trap.
    sample_ids = list(range(100, 100 + n))
    return img, txt, sample_ids


def test_feature_override_bypasses_feature_manager():
    img, txt, sample_ids = _synthetic_pair()
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    mgr = TrainableEmbeddingManager(
        sample_ids=sample_ids, embedding_dim=8, embeddings_dir=TMP_DIR,
        mode="ram", initialization_strategy="zeros", device=DEVICE,
    )
    # feature_manager=None: if the override branch didn't bypass it, this would raise
    # AttributeError on feature_manager.get_num_chunks().
    mgr.initialize(
        "buddies", feature_manager=None, model=None, device=DEVICE,
        k=10, feature_override=(img, txt, sample_ids),
    )
    emb = mgr.get_embeddings(sample_ids)
    assert emb.shape == (len(sample_ids), 8), f"unexpected shape {emb.shape}"
    assert not torch.allclose(emb, torch.zeros_like(emb)), "embeddings were not initialized"
    print("PASS test_feature_override_bypasses_feature_manager")


def test_feature_override_matches_direct_compute_buddy_init():
    """The override path must be numerically identical to calling compute_buddy_init
    directly with the same inputs — Task 1 must not add any reordering/reprocessing
    beyond what compute_buddy_init itself does."""
    img, txt, sample_ids = _synthetic_pair(seed=2)
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    mgr = TrainableEmbeddingManager(
        sample_ids=sample_ids, embedding_dim=8, embeddings_dir=TMP_DIR,
        mode="ram", initialization_strategy="zeros", device=DEVICE,
    )
    mgr.initialize(
        "buddies", feature_manager=None, model=None, device=DEVICE,
        k=10, seed=42, feature_override=(img, txt, sample_ids),
    )
    got = mgr.get_embeddings(sample_ids).detach().cpu().numpy()

    want = compute_buddy_init(
        img, txt, n_dim=8, K=10, device=DEVICE, seed=42, use_half=USE_HALF,
        input_sample_ids=sample_ids, output_sample_ids=sample_ids,
    )
    np.testing.assert_allclose(got, want, atol=1e-4)
    print("PASS test_feature_override_matches_direct_compute_buddy_init")


if __name__ == "__main__":
    test_feature_override_bypasses_feature_manager()
    test_feature_override_matches_direct_compute_buddy_init()
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    print("ALL TESTS PASSED")
