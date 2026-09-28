"""Parameterized Stage 2 mapper (capacity extras: num_queries, mlp_head)
and training/eval helpers. Fresh implementation for the same reason as
Tasks 3-5: the original `AttentionPoolingMapper` (imported from
`run_percept_stage2_pilot.py` by `run_buddy_stage2_pilot.py`) hardcodes a
single query and a linear head; this generalizes both without touching
that file.
"""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn


class ParameterizedAttentionPoolingMapper(nn.Module):
    def __init__(self, n_topics: int, num_queries: int, mlp_head: str, d_model: int = 512) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.query = nn.Parameter(torch.randn(num_queries, d_model) * 0.02)
        pooled_dim = num_queries * d_model
        if mlp_head == "linear":
            self.classifier = nn.Linear(pooled_dim, n_topics)
        elif mlp_head == "one_hidden":
            self.classifier = nn.Sequential(
                nn.Linear(pooled_dim, pooled_dim), nn.ReLU(), nn.Linear(pooled_dim, n_topics)
            )
        else:
            raise ValueError(f"Unknown mlp_head: {mlp_head}")

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        batch = patch_tokens.shape[0]
        query = self.query.unsqueeze(0).expand(batch, -1, -1)  # [B, Q, D]
        attn_weights = (query @ patch_tokens.transpose(1, 2)) / (patch_tokens.shape[-1] ** 0.5)
        attn_weights = attn_weights.softmax(dim=-1)  # [B, Q, 50]
        pooled = (attn_weights @ patch_tokens).reshape(batch, -1)  # [B, Q*D]
        return self.classifier(pooled)


def train_stage2(mapper, train_patches: torch.Tensor, train_targets: torch.Tensor,
                  mapper_lr: float, mapper_epochs: int, weight_decay: float,
                  class_balanced: bool, train_labels_for_weighting: np.ndarray, seed: int) -> None:
    torch.manual_seed(seed)
    device = next(mapper.parameters()).device
    train_patches = train_patches.to(device)
    train_targets = train_targets.to(device)
    optimizer = torch.optim.AdamW(mapper.parameters(), lr=mapper_lr, weight_decay=weight_decay)
    if class_balanced:
        n_topics = int(train_labels_for_weighting.max()) + 1
        counts = np.bincount(train_labels_for_weighting, minlength=n_topics)
        inverse_freq = (len(train_labels_for_weighting) / n_topics) / counts[train_labels_for_weighting]
        weights = torch.as_tensor(inverse_freq / inverse_freq.mean(), dtype=torch.float32, device=device)
    else:
        weights = torch.ones(len(train_labels_for_weighting), device=device)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    mapper.train()
    for _ in range(mapper_epochs):
        logits = mapper(train_patches)
        per_sample_loss = bce(logits, train_targets).mean(dim=1)
        loss = (per_sample_loss * weights).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    mapper.eval()


def evaluate_auc(scores: np.ndarray, targets: np.ndarray) -> tuple[dict, list]:
    aucs, skipped = {}, []
    for topic in range(targets.shape[1]):
        topic_targets = targets[:, topic]
        positives = int(topic_targets.sum())
        negatives = len(topic_targets) - positives
        if positives == 0 or negatives == 0:
            skipped.append(topic)
            continue
        aucs[topic] = float(roc_auc_score(topic_targets, scores[:, topic]))
    return aucs, skipped


def auc_summary(aucs: dict) -> dict:
    values = np.asarray(list(aucs.values()))
    return {
        "macro": float(values.mean()), "min": float(values.min()),
        "median": float(np.median(values)), "max": float(values.max()),
    }
