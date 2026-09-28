"""Train a shared factor dictionary from paired features and a content graph."""

from dataclasses import dataclass

import numpy as np
import torch
from scipy.sparse import csr_matrix, triu

from src.model.factors import SharedFactorEncoder
from src.train.factors import (
    anti_split_penalty,
    graph_neighbor_consistency_loss,
    paired_agreement_loss,
    reconstruction_loss,
    sparsity_penalty,
)


@dataclass
class FactorTrainingConfig:
    """Starting weights; this exact combination has no validated reference."""

    num_factors: int = 32
    lr: float = 1e-3
    epochs: int = 2000
    batch_size: int = 1024
    lambda_reconstruction: float = 1.0
    lambda_paired: float = 1.0
    lambda_graph: float = 1.0
    lambda_sparsity: float = 0.01
    lambda_anti_split: float = 0.1
    seed: int = 42


def train_factors(
    img_features: np.ndarray,
    txt_features: np.ndarray,
    graph: csr_matrix,
    config: FactorTrainingConfig,
    device: str | None = None,
) -> tuple[SharedFactorEncoder, np.ndarray, np.ndarray]:
    """Fit one edge-sampled unique-node batch per epoch; encode all rows once.

    Graph consistency compares the mean of each node's two modality codes on
    the induced graph of the sampled nodes. All other losses see the same nodes.
    """
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
    model = SharedFactorEncoder(feature_dim=img.shape[1], num_factors=config.num_factors)
    model = model.to(selected_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_rng = np.random.default_rng(config.seed + epoch)
        sampled = edges[epoch_rng.choice(
            len(edges), size=config.batch_size, replace=len(edges) < config.batch_size
        )]
        node_ids = np.unique(sampled.reshape(-1))
        img_batch = img[node_ids].to(selected_device)
        txt_batch = txt[node_ids].to(selected_device)
        img_codes = model.encode_image(img_batch)
        txt_codes = model.encode_text(txt_batch)
        local_graph = graph[node_ids][:, node_ids].tocsr()
        local_ids = np.arange(len(node_ids), dtype=np.int64)

        loss = (
            config.lambda_reconstruction * reconstruction_loss(model, img_batch, txt_batch)
            + config.lambda_paired * paired_agreement_loss(img_codes, txt_codes)
            + config.lambda_graph * graph_neighbor_consistency_loss(
                0.5 * (img_codes + txt_codes), local_graph, local_ids
            )
            + config.lambda_sparsity * 0.5 * (
                sparsity_penalty(img_codes) + sparsity_penalty(txt_codes)
            )
            + config.lambda_anti_split * anti_split_penalty(img_codes, txt_codes)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        print(f"factor epoch={epoch} loss={loss.item():.6f}", flush=True)

    model.eval()
    full_img_codes = []
    full_txt_codes = []
    with torch.no_grad():
        for start in range(0, len(img), 8192):
            full_img_codes.append(
                model.encode_image(img[start : start + 8192].to(selected_device)).cpu().numpy()
            )
            full_txt_codes.append(
                model.encode_text(txt[start : start + 8192].to(selected_device)).cpu().numpy()
            )
    return model, np.concatenate(full_img_codes), np.concatenate(full_txt_codes)
