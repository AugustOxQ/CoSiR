"""Wandb-agnostic orchestration of one full Stage1+Stage2 trial (spec §3).
`run_trial` is the single function the wandb entrypoint (Task 8) calls.
"""
import time
from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np
import torch
from sklearn.metrics import adjusted_mutual_info_score

from scripts.buddy_percept_sweep.cache import FixedInputs
from scripts.buddy_percept_sweep.clustering import leiden_partition, merge_small_communities
from scripts.buddy_percept_sweep.objective import compute_objective
from scripts.buddy_percept_sweep.stage1 import ParameterizedLearnedStudent, train_stage1
from scripts.buddy_percept_sweep.stage2 import (
    ParameterizedAttentionPoolingMapper, auc_summary, evaluate_auc, train_stage2,
)
from scripts.buddy_percept_sweep.targets import (
    assign_to_train_communities, build_targets, cosine_vote_fractions, one_hot,
)


@dataclass
class TrialConfig:
    heads: str = "attn1"
    num_heads: int = 1
    d_shared: int = 32
    lr: float = 1e-3
    noise_std: float = 0.0
    lambda_affect: float = 1.0
    batch_size: int = 1024
    weight_decay: float = 0.0
    teacher_graph_K: int = 20
    leiden_resolution: float = 1.0
    merge_small_threshold: float = 0.0
    mapper_lr: float = 1e-2
    mapper_epochs: int = 400
    num_queries: int = 1
    mlp_head: str = "linear"
    transfer_k: int = 20
    target_cutoff: Union[str, float] = "single_label"
    class_balanced_loss: bool = False
    weight_decay_stage2: float = 0.0
    max_epochs_stage1: int = 200
    seed: int = 42
    content_pca_dim: int = 50


@dataclass
class TrialResult:
    emotion_ami: float
    genre_ami: float
    stage2_macro_auc: float
    objective: float
    n_topics_after_merge: int
    stage1_seconds: float
    stage2_seconds: float


def run_trial(config: TrialConfig, fixed_inputs: FixedInputs,
              log_checkpoint: Optional[Callable[[int, float], None]] = None) -> TrialResult:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    stage1_start = time.monotonic()
    student = ParameterizedLearnedStudent(
        heads=config.heads, num_heads=config.num_heads, d_shared=config.d_shared,
        content_dim=fixed_inputs.train_content.shape[1], affect_dim=fixed_inputs.train_affect.shape[1],
    )
    train_embedding = train_stage1(
        student, fixed_inputs, lr=config.lr, noise_std=config.noise_std,
        lambda_affect=config.lambda_affect, batch_size=config.batch_size,
        weight_decay=config.weight_decay, seed=config.seed, max_epochs=config.max_epochs_stage1,
        log_checkpoint=log_checkpoint, teacher_graph_K=config.teacher_graph_K,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student.eval()
    with torch.no_grad():
        heldout_embedding, _ = student(
            torch.as_tensor(fixed_inputs.heldout_content, dtype=torch.float32, device=device),
            torch.as_tensor(fixed_inputs.heldout_affect, dtype=torch.float32, device=device),
        )
    heldout_embedding = heldout_embedding.cpu().numpy()

    k_neighbors = max(2, min(20, len(train_embedding) - 1))
    raw_labels = leiden_partition(train_embedding, resolution=config.leiden_resolution,
                                   seed=config.seed, k_neighbors=k_neighbors)
    merged_labels, _ = merge_small_communities(train_embedding, raw_labels, config.merge_small_threshold)
    n_topics = int(merged_labels.max()) + 1

    transfer_k = min(config.transfer_k, len(train_embedding))
    heldout_hard = assign_to_train_communities(train_embedding, merged_labels, heldout_embedding, k=transfer_k)

    emotion_ami = float(adjusted_mutual_info_score(fixed_inputs.heldout_emotion, heldout_hard))
    genre_mask = fixed_inputs.heldout_genre != ""
    if genre_mask.any() and len(set(heldout_hard[genre_mask].tolist())) > 1:
        genre_ami = float(adjusted_mutual_info_score(fixed_inputs.heldout_genre[genre_mask], heldout_hard[genre_mask]))
    else:
        genre_ami = 0.0
    stage1_seconds = time.monotonic() - stage1_start

    stage2_start = time.monotonic()
    if n_topics < 2:
        # Degenerate partition: no valid multi-class target to train a
        # classifier on. Gate-fail via the objective rather than crash.
        return TrialResult(
            emotion_ami=emotion_ami, genre_ami=genre_ami, stage2_macro_auc=0.5,
            objective=compute_objective(emotion_ami, genre_ami, 0.5),
            n_topics_after_merge=n_topics, stage1_seconds=stage1_seconds,
            stage2_seconds=time.monotonic() - stage2_start,
        )

    if config.target_cutoff == "single_label":
        train_targets_np = one_hot(merged_labels, n_topics)
        baseline_heldout_targets = one_hot(heldout_hard, n_topics)
    else:
        train_fractions = cosine_vote_fractions(train_embedding, merged_labels, train_embedding,
                                                 n_topics, k=transfer_k)
        train_targets_np = build_targets(train_fractions, n_topics, config.target_cutoff)
        # Evaluation ALWAYS uses single-label held-out targets (Global
        # Constraints) regardless of the training target convention.
        baseline_heldout_targets = one_hot(heldout_hard, n_topics)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    mapper = ParameterizedAttentionPoolingMapper(
        n_topics=n_topics, num_queries=config.num_queries, mlp_head=config.mlp_head,
        d_model=fixed_inputs.train_patches.shape[-1],
    ).to(device)
    train_targets = torch.as_tensor(train_targets_np, dtype=torch.float32)
    train_stage2(
        mapper, fixed_inputs.train_patches, train_targets, mapper_lr=config.mapper_lr,
        mapper_epochs=config.mapper_epochs, weight_decay=config.weight_decay_stage2,
        class_balanced=config.class_balanced_loss, train_labels_for_weighting=merged_labels,
        seed=config.seed,
    )
    mapper.eval()
    with torch.no_grad():
        heldout_scores = torch.sigmoid(mapper(fixed_inputs.heldout_patches.to(device))).cpu().numpy()
    aucs, skipped = evaluate_auc(heldout_scores, baseline_heldout_targets)
    stage2_macro_auc = auc_summary(aucs)["macro"] if aucs else 0.5
    stage2_seconds = time.monotonic() - stage2_start

    objective = compute_objective(emotion_ami, genre_ami, stage2_macro_auc)
    return TrialResult(
        emotion_ami=emotion_ami, genre_ami=genre_ami, stage2_macro_auc=stage2_macro_auc,
        objective=objective, n_topics_after_merge=n_topics,
        stage1_seconds=stage1_seconds, stage2_seconds=stage2_seconds,
    )
