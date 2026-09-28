"""Translates a raw wandb.config dict (or plain dict, for tests) into a
validated TrialConfig (spec §5's 22 parameters)."""
from scripts.buddy_percept_sweep.pipeline import TrialConfig


def resolve_trial_config(raw: dict) -> TrialConfig:
    target_cutoff = raw["target_cutoff"]
    if target_cutoff != "single_label":
        target_cutoff = float(target_cutoff)
    return TrialConfig(
        heads=str(raw["heads"]),
        num_heads=int(raw["num_heads"]),
        d_shared=int(raw["d_shared"]),
        lr=float(raw["lr"]),
        noise_std=float(raw["noise_std"]),
        lambda_affect=float(raw["lambda_affect"]),
        batch_size=int(raw["batch_size"]),
        weight_decay=float(raw["weight_decay"]),
        teacher_graph_K=int(raw["teacher_graph_K"]),
        leiden_resolution=float(raw["leiden_resolution"]),
        merge_small_threshold=float(raw["merge_small_threshold"]),
        mapper_lr=float(raw["mapper_lr"]),
        mapper_epochs=int(raw["mapper_epochs"]),
        num_queries=int(raw["num_queries"]),
        mlp_head=str(raw["mlp_head"]),
        transfer_k=int(raw["transfer_k"]),
        target_cutoff=target_cutoff,
        class_balanced_loss=bool(raw["class_balanced_loss"]),
        weight_decay_stage2=float(raw["weight_decay_stage2"]),
        content_pca_dim=int(raw.get("content_pca_dim", 50)),
    )
