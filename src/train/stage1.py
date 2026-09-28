"""Stage 1 student training against positive pairs from a content graph."""

from dataclasses import dataclass

import numpy as np
import torch
from scipy.sparse import csr_matrix, triu
from torch.nn import functional as F

from src.model.student import AttentionFusionStudent


@dataclass
class Stage1Config:
    lr: float = 1e-3
    epochs: int = 200
    batch_size: int = 1024
    temperature: float = 0.1
    seed: int = 42


def train_stage1(
    img_features: np.ndarray,
    txt_features: np.ndarray,
    graph: csr_matrix,
    config: Stage1Config,
    device: str | None = None,
) -> tuple[AttentionFusionStudent, np.ndarray]:
    """Fit one sampled edge batch per epoch and encode all samples at the end."""
    upper = triu(graph, k=1).tocoo()
    upper.eliminate_zeros()
    edges = np.column_stack((upper.row, upper.col)).astype(np.int64, copy=False)
    if len(edges) == 0:
        raise ValueError("Teacher graph contains no upper-triangle edges to train on")

    img = torch.as_tensor(np.asarray(img_features, dtype=np.float32))
    txt = torch.as_tensor(np.asarray(txt_features, dtype=np.float32))
    if img.shape != txt.shape or graph.shape != (len(img), len(img)):
        raise ValueError("Image/text features and teacher graph must have matching sample dimensions")

    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(config.seed)
    model = AttentionFusionStudent(feature_dim=img.shape[1]).to(selected_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_rng = np.random.default_rng(config.seed + epoch)
        sampled = edges[epoch_rng.choice(
            len(edges), size=config.batch_size, replace=len(edges) < config.batch_size
        )]
        node_ids, inverse = np.unique(sampled.reshape(-1), return_inverse=True)
        local_pairs = torch.as_tensor(
            inverse.reshape(-1, 2), dtype=torch.long, device=selected_device
        )
        embeddings = model(img[node_ids].to(selected_device), txt[node_ids].to(selected_device))
        anchors = embeddings[local_pairs[:, 0]]
        positives = embeddings[local_pairs[:, 1]]
        targets = torch.arange(config.batch_size, device=selected_device)
        logits = anchors @ positives.T / config.temperature
        reverse_logits = positives @ anchors.T / config.temperature
        loss = 0.5 * (
            F.cross_entropy(logits, targets) + F.cross_entropy(reverse_logits, targets)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        print(f"stage1 epoch={epoch} loss={loss.item():.6f}", flush=True)

    model.eval()
    full_embeddings = []
    with torch.no_grad():
        for start in range(0, len(img), 8192):
            batch = model(
                img[start : start + 8192].to(selected_device),
                txt[start : start + 8192].to(selected_device),
            )
            full_embeddings.append(batch.cpu().numpy())
    return model, np.concatenate(full_embeddings, axis=0)
