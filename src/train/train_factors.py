"""Train a shared factor dictionary from paired features and a content graph."""

from dataclasses import asdict, dataclass

import numpy as np
import torch
from scipy.sparse import csr_matrix, triu

from src.model.factors import SharedFactorEncoder
from src.train.factors import (
    anti_split_penalty,
    cross_modal_infonce_loss,
    decorrelation_penalty,
    graph_neighbor_consistency_loss,
    paired_agreement_loss,
    reconstruction_loss,
    sparsity_penalty,
    usage_balance_penalty,
)


@dataclass
class FactorTrainingConfig:
    """Starting weights, not validated optima.

    WARNING: the defaults are the historical collapsed recipe R0 (cosine
    agreement; 32 factors collapse onto about one axis, see
    docs/reports/auto/v2/2026-10-09_candidate_a_factor_collapse_diagnosis.md).
    They are kept unchanged for reproducibility. New work should start from
    ``R3_CONFIG``, the recipe selected in
    docs/reports/auto/v2/2026-10-11_candidate_a_factor_repair.md (derive
    variants with ``dataclasses.replace(R3_CONFIG, ...)``; do not mutate it).

    Usage balance starts at 0.1, comparable to anti-split because both are
    secondary balance regularizers relative to reconstruction and agreement.

    The last six fields are collapse-fix mechanisms, all OFF by default so the
    default training computation is unchanged: ``agreement="infonce"`` swaps the
    cosine paired-agreement term for cross-modal InfoNCE at
    ``infonce_temperature``; ``lambda_decorrelation > 0`` adds a factor
    decorrelation penalty; ``activation="topk"`` (with ``topk``) keeps only the
    top-k factors per item; ``center_inputs`` subtracts per-modality feature
    means inside the encoder.
    """

    num_factors: int = 32
    lr: float = 1e-3
    epochs: int = 2000
    batch_size: int = 1024
    lambda_reconstruction: float = 1.0
    lambda_paired: float = 1.0
    lambda_graph: float = 1.0
    lambda_sparsity: float = 0.01
    lambda_anti_split: float = 0.1
    lambda_usage_balance: float = 0.1
    seed: int = 42
    agreement: str = "cosine"
    infonce_temperature: float = 0.1
    lambda_decorrelation: float = 0.0
    activation: str = "relu"
    topk: int | None = None
    center_inputs: bool = False


R3_CONFIG = FactorTrainingConfig(lambda_usage_balance=0.1, agreement="infonce", lambda_decorrelation=1.0)
"""The repaired recipe R3 (Task 6 selection, amended gates). Train it with ``group_ids`` = leakage-group
ids of the training rows, as Task 6 did; equals the config stored in Task 6's selected_seed42.pt."""


def encode_rows(model: SharedFactorEncoder, img_features, txt_features, rows=None,
                batch_size: int = 8192, device=None) -> tuple[np.ndarray, np.ndarray]:
    """Encode (a subset of) rows with a frozen model: eval mode, no_grad, fixed-size batches.

    ``rows`` selects rows first (``None`` = all). ``device`` is where batches are sent (default: the
    model's device; the model is not moved). The model's train/eval mode is restored afterwards.
    Returns float32 numpy ``(img_codes, txt_codes)``.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if rows is not None:
        img_features, txt_features = img_features[rows], txt_features[rows]
    if len(img_features) != len(txt_features):
        raise ValueError("Image and text features must have the same number of rows")
    target = torch.device(device) if device is not None else next(model.parameters()).device
    was_training = model.training
    model.eval()
    img_out, txt_out = [], []
    try:
        with torch.no_grad():
            for start in range(0, len(img_features), batch_size):
                stop = start + batch_size
                img_out.append(model.encode_image(torch.as_tensor(
                    img_features[start:stop], dtype=torch.float32, device=target)).cpu().numpy())
                txt_out.append(model.encode_text(torch.as_tensor(
                    txt_features[start:stop], dtype=torch.float32, device=target)).cpu().numpy())
    finally:
        model.train(was_training)
    return np.concatenate(img_out), np.concatenate(txt_out)


def train_factors(
    img_features: np.ndarray,
    txt_features: np.ndarray,
    graph: csr_matrix,
    config: FactorTrainingConfig,
    device: str | None = None,
    group_ids: np.ndarray | None = None,
) -> tuple[SharedFactorEncoder, np.ndarray, np.ndarray]:
    """Fit one edge-sampled unique-node batch per epoch; encode all rows once.

    Graph consistency compares the mean of each node's two modality codes on
    the induced graph of the sampled nodes. All other losses see the same nodes.
    ``group_ids`` (one integer per row, e.g. painting / leakage-group id) is used only by the
    ``"infonce"`` agreement term, to exclude same-group pairs as negatives. InfoNCE REQUIRES it:
    the content graph is a mutual-kNN graph on image features, so same-painting edges are common,
    and silently treating them as negatives trains a recipe that is not the validated R3. Pass
    ``group_ids=np.arange(n)`` to opt out of masking explicitly.
    """
    if config.agreement not in {"cosine", "infonce"}:
        raise ValueError(f"agreement must be 'cosine' or 'infonce', got {config.agreement!r}")
    if config.agreement == "infonce" and group_ids is None:
        raise ValueError(
            "agreement='infonce' needs group_ids (one integer group id per row, e.g. leakage-group ids) so "
            "same-painting pairs are not used as negatives; pass group_ids=np.arange(n) to disable masking explicitly"
        )
    upper = triu(graph, k=1).tocoo()
    upper.eliminate_zeros()
    edges = np.column_stack((upper.row, upper.col)).astype(np.int64, copy=False)
    if len(edges) == 0:
        raise ValueError("Teacher graph contains no upper-triangle edges to train on")

    img = torch.as_tensor(np.asarray(img_features, dtype=np.float32))
    txt = torch.as_tensor(np.asarray(txt_features, dtype=np.float32))
    if img.shape != txt.shape or graph.shape != (len(img), len(img)):
        raise ValueError("Image/text features and teacher graph must have matching sample dimensions")
    if group_ids is not None:
        group_ids = np.asarray(group_ids)
        if group_ids.shape != (len(img),):
            raise ValueError(
                f"group_ids must have one entry per row ({len(img)}), got shape {group_ids.shape}"
            )
        if not np.issubdtype(group_ids.dtype, np.integer):
            raise ValueError(
                f"group_ids must be integer ids (e.g. dense leakage-group ids), got dtype {group_ids.dtype}"
            )

    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(config.seed)
    model = SharedFactorEncoder(
        feature_dim=img.shape[1],
        num_factors=config.num_factors,
        activation=config.activation,
        topk=config.topk,
        image_mean=img.mean(0) if config.center_inputs else None,
        text_mean=txt.mean(0) if config.center_inputs else None,
    )
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

        if config.agreement == "cosine":
            agreement = paired_agreement_loss(img_codes, txt_codes)
        else:
            agreement = cross_modal_infonce_loss(
                img_codes,
                txt_codes,
                config.infonce_temperature,
                None
                if group_ids is None
                else torch.as_tensor(group_ids[node_ids], device=selected_device),
            )

        loss = (
            config.lambda_reconstruction * reconstruction_loss(model, img_batch, txt_batch)
            + config.lambda_paired * agreement
            + config.lambda_graph * graph_neighbor_consistency_loss(
                0.5 * (img_codes + txt_codes), local_graph, local_ids
            )
            + config.lambda_sparsity * 0.5 * (
                sparsity_penalty(img_codes) + sparsity_penalty(txt_codes)
            )
            + config.lambda_anti_split * anti_split_penalty(img_codes, txt_codes)
            + config.lambda_usage_balance * usage_balance_penalty(img_codes, txt_codes)
        )
        if config.lambda_decorrelation > 0:
            loss = loss + config.lambda_decorrelation * 0.5 * (
                decorrelation_penalty(img_codes) + decorrelation_penalty(txt_codes)
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        print(f"factor epoch={epoch} loss={loss.item():.6f}", flush=True)

    model.eval()
    full_img_codes, full_txt_codes = encode_rows(model, img, txt, device=selected_device)
    return model, full_img_codes, full_txt_codes


def save_factor_checkpoint(model: SharedFactorEncoder, config: FactorTrainingConfig, path) -> None:
    torch.save({"state_dict": model.state_dict(), "config": asdict(config),
                "feature_dim": model.image_encoder.in_features}, path)


def load_factor_checkpoint(path, device: str = "cpu") -> tuple[SharedFactorEncoder, FactorTrainingConfig]:
    payload = torch.load(path, map_location=device, weights_only=True)
    config = FactorTrainingConfig(**payload["config"])
    model = SharedFactorEncoder(payload["feature_dim"], config.num_factors,
                                activation=config.activation, topk=config.topk)
    model.load_state_dict(payload["state_dict"])     # restores image_mean / text_mean buffers
    return model.to(device).eval(), config
