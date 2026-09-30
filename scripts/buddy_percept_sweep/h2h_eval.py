"""Evaluation for the matched-topic-count head-to-head on one held-out subset
(val or test): transfer labels (k-NN vote into the system's own train topics,
k = EVAL_TRANSFER_K fixed, spec R2), both Stage-1 AMI yardsticks, and one
Stage-2 mapper scored against one or more held-out label sets. The Stage-2
path mirrors `pipeline.run_trial` (lines 111-146) on store inputs.
"""
import math
import random
from typing import Optional

import numpy as np
import torch

from scripts.buddy_percept_sweep.h2h_types import EVAL_TRANSFER_K
from scripts.buddy_percept_sweep.pilot_metrics import ami_emotion_genre, independent_partition
from scripts.buddy_percept_sweep.stage2 import (
    ParameterizedAttentionPoolingMapper, auc_summary, evaluate_auc, train_stage2,
)
from scripts.buddy_percept_sweep.targets import (
    assign_to_train_communities, build_targets, cosine_vote_fractions, one_hot,
)


def eval_labels(train_embedding: np.ndarray, train_labels: np.ndarray, heldout_embedding: np.ndarray,
                subset_idx: np.ndarray) -> np.ndarray:
    """Primary held-out topic labels of the subset rows: majority topic of the
    EVAL_TRANSFER_K nearest train nodes (cosine) in the Stage-1 embedding."""
    return assign_to_train_communities(train_embedding, train_labels, heldout_embedding[subset_idx],
                                       EVAL_TRANSFER_K)


def stage1_metrics(train_embedding: np.ndarray, train_labels: np.ndarray, heldout_embedding: np.ndarray,
                   subset_idx: np.ndarray, heldout_emotion: np.ndarray, heldout_genre: np.ndarray, seed: int,
                   pilot, device: str, native_subset: Optional[np.ndarray] = None,
                   transfer_labels: Optional[np.ndarray] = None) -> dict:
    """Emotion/genre AMI of the subset under (a) transfer labels, (b) the
    pilots' independent re-clustering of the subset's held-out embedding
    (seeded by `seed`), and (c) the native labels when given.
    `transfer_labels` (= eval_labels(...) for this subset) skips recomputing them."""
    subset_idx = np.asarray(subset_idx)
    emotion, genre = np.asarray(heldout_emotion)[subset_idx], np.asarray(heldout_genre)[subset_idx]
    if transfer_labels is None:
        transfer_labels = eval_labels(train_embedding, train_labels, heldout_embedding, subset_idx)
    transfer_emo, transfer_genre = ami_emotion_genre(transfer_labels, emotion, genre)
    independent = independent_partition(heldout_embedding[subset_idx], pilot, "heldout", seed, device)
    ind_emo, ind_genre = ami_emotion_genre(independent, emotion, genre)
    out = {"transfer_emo": transfer_emo, "transfer_genre": transfer_genre,
           "ind_emo": ind_emo, "ind_genre": ind_genre, "ind_k": int(len(np.unique(independent)))}
    if native_subset is not None:
        out["native_emo"], out["native_genre"] = ami_emotion_genre(native_subset, emotion, genre)
    return out


def _auc_for(scores: np.ndarray, labels: np.ndarray, n_topics: int) -> tuple[float, int]:
    """Macro AUC of `scores` against single-label targets; topics with no
    positive (or no negative) row are skipped. NaN when every topic is skipped."""
    aucs, skipped = evaluate_auc(scores, one_hot(np.asarray(labels, dtype=np.int64), n_topics))
    macro = auc_summary(aucs)["macro"] if aucs else math.nan
    return macro, len(skipped)


def seed_all(seed: int) -> None:
    """Seed every global RNG a trial step may draw from."""
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stage2_metrics(s2cfg, store, train_embedding: np.ndarray, train_labels: np.ndarray, subset_idx: np.ndarray,
                   eval_label_sets: dict, seed: int, device: str) -> dict:
    """Trains ONE mapper on all train patches against the train topics and
    scores the subset's held-out patches against each label set.
    Returns auc_<name> (macro) and skipped_<name> (topics with no positive or
    no negative row in that set) for every name in `eval_label_sets`."""
    train_labels = np.asarray(train_labels, dtype=np.int64)
    n_topics = int(train_labels.max()) + 1
    if len(np.unique(train_labels)) != n_topics or train_labels.min() != 0:
        raise ValueError(f"train labels must be 0..n-1 with every topic used; got {len(np.unique(train_labels))} "
                         f"distinct labels in [{train_labels.min()}, {train_labels.max()}]")
    if s2cfg.target_cutoff == "single_label":
        train_targets_np = one_hot(train_labels, n_topics)
    else:
        fractions = cosine_vote_fractions(train_embedding, train_labels, train_embedding, n_topics,
                                          k=s2cfg.train_target_k)
        train_targets_np = build_targets(fractions, n_topics, float(s2cfg.target_cutoff))

    seed_all(seed)
    mapper = ParameterizedAttentionPoolingMapper(
        n_topics=n_topics, num_queries=s2cfg.num_queries, mlp_head=s2cfg.mlp_head,
        d_model=store.train_patches.shape[-1],
    ).to(device)
    train_stage2(
        mapper, store.train_patches, torch.as_tensor(train_targets_np, dtype=torch.float32),
        mapper_lr=s2cfg.mapper_lr, mapper_epochs=s2cfg.mapper_epochs, weight_decay=s2cfg.weight_decay_stage2,
        class_balanced=s2cfg.class_balanced_loss, train_labels_for_weighting=train_labels, seed=seed,
    )
    mapper.eval()
    subset_rows = torch.as_tensor(np.asarray(subset_idx), dtype=torch.long)
    with torch.no_grad():
        scores = torch.sigmoid(mapper(store.heldout_patches[subset_rows].to(device))).cpu().numpy()

    out = {}
    for name, labels in eval_label_sets.items():
        out[f"auc_{name}"], out[f"skipped_{name}"] = _auc_for(scores, labels, n_topics)
    return out
