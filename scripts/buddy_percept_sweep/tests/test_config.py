from scripts.buddy_percept_sweep.config import resolve_trial_config


def _full_raw_config(**overrides):
    base = dict(
        heads="attn1", num_heads=1, d_shared=32, lr=1e-3, noise_std=0.0,
        lambda_affect=1.0, batch_size=1024, weight_decay=0.0,
        teacher_graph_K=20,
        leiden_resolution=1.0, merge_small_threshold=0.0,
        mapper_lr=1e-2, mapper_epochs=400, num_queries=1, mlp_head="linear",
        transfer_k=20, target_cutoff="single_label", class_balanced_loss=False,
        weight_decay_stage2=0.0,
    )
    base.update(overrides)
    return base


def test_resolve_trial_config_round_trips_all_core_fields():
    config = resolve_trial_config(_full_raw_config())
    assert config.heads == "attn1"
    assert config.mapper_epochs == 400
    assert config.target_cutoff == "single_label"


def test_resolve_trial_config_accepts_numeric_target_cutoff():
    config = resolve_trial_config(_full_raw_config(target_cutoff=0.15))
    assert config.target_cutoff == 0.15


def test_resolve_trial_config_coerces_bool_like_class_balanced():
    config = resolve_trial_config(_full_raw_config(class_balanced_loss=True))
    assert config.class_balanced_loss is True
