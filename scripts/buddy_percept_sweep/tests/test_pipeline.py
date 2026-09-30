import numpy as np
import torch

from scripts.buddy_percept_sweep.cache import FixedInputs
from scripts.buddy_percept_sweep.pipeline import TrialConfig, run_trial


def test_trial_config_has_defaults_including_content_pca_dim():
    config = TrialConfig()
    assert config.content_pca_dim == 50


def _tiny_fixed_inputs(n_train=60, n_heldout=24):
    rng = np.random.default_rng(0)
    # Two synthetic "emotion" and "genre" clusters so AMI has real signal.
    train_emotion = ["awe"] * (n_train // 2) + ["fear"] * (n_train // 2)
    heldout_emotion = ["awe"] * (n_heldout // 2) + ["fear"] * (n_heldout // 2)
    train_genre = np.array(["landscape"] * (n_train // 2) + ["portrait"] * (n_train // 2), dtype=object)
    heldout_genre = np.array(["landscape"] * (n_heldout // 2) + ["portrait"] * (n_heldout // 2), dtype=object)
    return FixedInputs(
        train_content=rng.normal(size=(n_train, 10)).astype(np.float32),
        train_affect=rng.normal(size=(n_train, 28)).astype(np.float32),
        heldout_content=rng.normal(size=(n_heldout, 10)).astype(np.float32),
        heldout_affect=rng.normal(size=(n_heldout, 28)).astype(np.float32),
        train_emotion=train_emotion,
        heldout_emotion=heldout_emotion,
        train_genre=train_genre,
        heldout_genre=heldout_genre,
        train_patches=torch.randn(n_train, 50, 16),
        heldout_patches=torch.randn(n_heldout, 50, 16),
        content_pca_dim=10,
    )


def test_run_trial_completes_and_returns_finite_result():
    fixed = _tiny_fixed_inputs()
    config = TrialConfig(
        heads="attn1", num_heads=1, d_shared=8, lr=1e-3, noise_std=0.0,
        lambda_affect=1.0, batch_size=16, weight_decay=0.0,
        teacher_graph_K=5,
        leiden_resolution=1.0, merge_small_threshold=0.0,
        mapper_lr=1e-2, mapper_epochs=20, num_queries=1, mlp_head="linear",
        transfer_k=5, target_cutoff="single_label", class_balanced_loss=False,
        weight_decay_stage2=0.0, max_epochs_stage1=20,
    )
    result = run_trial(config, fixed, log_checkpoint=None)
    assert np.isfinite(result.emotion_ami)
    assert np.isfinite(result.genre_ami)
    assert np.isfinite(result.stage2_macro_auc)
    assert result.objective in (-1.0,) or 0.0 <= result.objective <= 1.0
    assert result.n_topics_after_merge >= 1
    assert result.stage1_seconds > 0
    assert result.stage2_seconds > 0


def test_run_trial_handles_degenerate_leiden_resolution_without_crashing():
    fixed = _tiny_fixed_inputs()
    config = TrialConfig(
        heads="mlp128", num_heads=1, d_shared=8, lr=1e-3, noise_std=0.0,
        lambda_affect=1.0, batch_size=16, weight_decay=0.0,
        teacher_graph_K=5,
        leiden_resolution=0.001, merge_small_threshold=0.0,  # degenerate: likely K=1
        mapper_lr=1e-2, mapper_epochs=20, num_queries=1, mlp_head="linear",
        transfer_k=5, target_cutoff=0.15, class_balanced_loss=True,
        weight_decay_stage2=0.0, max_epochs_stage1=20,
    )
    result = run_trial(config, fixed, log_checkpoint=None)
    # Degenerate partitions should gate-fail, not raise.
    assert result.objective == -1.0


def test_run_trial_is_deterministic_for_same_seed_and_fixed_inputs():
    fixed = _tiny_fixed_inputs()
    config = TrialConfig(
        heads="attn1", num_heads=1, d_shared=8, lr=1e-3, noise_std=0.0,
        lambda_affect=1.0, batch_size=16, weight_decay=0.0,
        teacher_graph_K=5,
        leiden_resolution=1.0, merge_small_threshold=0.0,
        mapper_lr=1e-2, mapper_epochs=20, num_queries=1, mlp_head="linear",
        transfer_k=5, target_cutoff="single_label", class_balanced_loss=False,
        weight_decay_stage2=0.0, max_epochs_stage1=20, seed=123,
    )
    first = run_trial(config, fixed)
    # A trial's result must not depend on RNG draws made by a prior trial.
    torch.rand(100)
    np.random.rand(100)
    second = run_trial(config, fixed)
    assert first.n_topics_after_merge >= 2
    for field in ("emotion_ami", "genre_ami", "stage2_macro_auc", "objective", "n_topics_after_merge"):
        assert getattr(first, field) == getattr(second, field)


def test_run_trial_return_artifacts_matches_default_and_has_expected_shapes():
    fixed = _tiny_fixed_inputs()
    config = TrialConfig(
        heads="attn1", num_heads=1, d_shared=8, lr=1e-3, noise_std=0.0,
        lambda_affect=1.0, batch_size=16, weight_decay=0.0,
        teacher_graph_K=5,
        leiden_resolution=1.0, merge_small_threshold=0.0,
        mapper_lr=1e-2, mapper_epochs=20, num_queries=1, mlp_head="linear",
        transfer_k=5, target_cutoff="single_label", class_balanced_loss=False,
        weight_decay_stage2=0.0, max_epochs_stage1=20, seed=123,
    )
    plain = run_trial(config, fixed)
    with_artifacts = run_trial(config, fixed, return_artifacts=True)
    assert plain.artifacts is None
    artifacts = with_artifacts.artifacts
    assert set(artifacts) == {"train_embedding", "heldout_embedding", "raw_labels",
                              "merged_labels", "heldout_hard"}
    assert artifacts["train_embedding"].shape == (60, 8)
    assert artifacts["heldout_embedding"].shape == (24, 8)
    assert artifacts["train_embedding"].dtype == np.float32
    assert artifacts["heldout_embedding"].dtype == np.float32
    assert artifacts["raw_labels"].shape == (60,)
    assert artifacts["merged_labels"].shape == (60,)
    assert artifacts["heldout_hard"].shape == (24,)
    for field in ("emotion_ami", "genre_ami", "stage2_macro_auc", "objective", "n_topics_after_merge"):
        assert getattr(plain, field) == getattr(with_artifacts, field)


def test_run_trial_return_artifacts_on_degenerate_early_return():
    fixed = _tiny_fixed_inputs()
    config = TrialConfig(
        heads="mlp128", num_heads=1, d_shared=8, batch_size=16, teacher_graph_K=5,
        leiden_resolution=0.001, transfer_k=5, max_epochs_stage1=20,
    )
    result = run_trial(config, fixed, return_artifacts=True)
    assert result.objective == -1.0
    assert result.artifacts["heldout_hard"].shape == (24,)
