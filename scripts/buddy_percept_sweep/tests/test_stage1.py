import numpy as np
import torch

from scripts.buddy_percept_sweep.stage1 import ParameterizedLearnedStudent, train_stage1
from scripts.buddy_percept_sweep.cache import FixedInputs


def _tiny_fixed_inputs():
    rng = np.random.default_rng(0)
    n_train, n_heldout = 40, 16
    return FixedInputs(
        train_content=rng.normal(size=(n_train, 10)).astype(np.float32),
        train_affect=rng.normal(size=(n_train, 28)).astype(np.float32),
        heldout_content=rng.normal(size=(n_heldout, 10)).astype(np.float32),
        heldout_affect=rng.normal(size=(n_heldout, 28)).astype(np.float32),
        train_emotion=["awe"] * n_train,
        heldout_emotion=["awe"] * n_heldout,
        train_genre=np.array(["landscape"] * n_train, dtype=object),
        heldout_genre=np.array(["landscape"] * n_heldout, dtype=object),
        train_patches=torch.zeros(n_train, 50, 512),
        heldout_patches=torch.zeros(n_heldout, 50, 512),
        content_pca_dim=10,
    )


def test_mlp128_ignores_num_heads_and_runs():
    student = ParameterizedLearnedStudent(
        heads="mlp128", num_heads=4, d_shared=16, content_dim=10, affect_dim=28
    )
    content = torch.randn(5, 10)
    affect = torch.randn(5, 28)
    embedding, gate = student(content, affect)
    assert embedding.shape == (5, 16)
    # L2-normalized output.
    norms = embedding.norm(dim=1)
    assert torch.allclose(norms, torch.ones(5), atol=1e-5)


def test_attention_head_variants_produce_correct_shape():
    for heads, num_heads in (("attn1", 1), ("attn4", 4), ("attn1", 8)):
        student = ParameterizedLearnedStudent(
            heads=heads, num_heads=num_heads, d_shared=16, content_dim=10, affect_dim=28
        )
        embedding, _ = student(torch.randn(5, 10), torch.randn(5, 28))
        assert embedding.shape == (5, 16)


def test_train_stage1_runs_end_to_end_on_tiny_data():
    fixed = _tiny_fixed_inputs()
    student = ParameterizedLearnedStudent(
        heads="attn1", num_heads=1, d_shared=8, content_dim=10, affect_dim=28
    )
    checkpoints = []
    embedding = train_stage1(
        student, fixed, lr=1e-3, noise_std=0.0, lambda_affect=1.0,
        batch_size=8, weight_decay=0.0, seed=42, max_epochs=10,
        log_checkpoint=lambda epoch, proxy: checkpoints.append((epoch, proxy)),
    )
    assert embedding.shape == (40, 8)
    assert len(checkpoints) > 0
    assert np.isfinite(embedding).all()
