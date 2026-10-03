"""Train a shared factor dictionary from paired features and a content graph."""

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
from scipy.sparse import csr_matrix, triu

from src.model.factors import SharedFactorEncoder
from src.train.aspect_loss import aspect_episode_loss, aspect_episode_scores
from src.train.condition_episodes import mine_condition_episodes
from src.train.factor_condition_loss import naive_episode_loss, naive_episode_scores
from src.train.factors import (
    anti_split_penalty,
    cross_modal_infonce_loss,
    decorrelation_penalty,
    graph_neighbor_consistency_loss,
    paired_agreement_loss,
    painting_infonce_loss,
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

    The last five fields (factor-learning spec 2026-09-30) are also OFF by default.
    ``painting_batches`` expands every edge-sampled batch to all rows of each sampled painting (group);
    ``agreement_level="painting"`` replaces the row InfoNCE with ``painting_infonce_loss`` (needs
    ``painting_batches`` and ``agreement="infonce"``); ``lambda_condition > 0`` adds the naive-rule
    condition-episode loss (``naive_episode_loss``) on ``condition_episodes_per_step`` episodes mined per
    step from a ``condition_source``, scored at the fixed ``condition_beta`` with 12 random negatives.
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
    agreement_level: str = "pair"
    painting_batches: bool = False
    lambda_condition: float = 0.0
    condition_episodes_per_step: int = 64
    condition_beta: float = 0.3
    lambda_aspect: float = 0.0
    aspect_episodes_per_step: int = 32
    aspect_beta: float = 0.3
    lambda_swap: float = 1.0
    aspect_tau_fixed: bool = False          # True: keep the aspect-loss temperature at its step-0 value (spec §15)


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


class GroupRows:
    """Row lookup by group id: ``expand(rows)`` returns every row of every group that ``rows`` touches, sorted."""

    def __init__(self, group_ids: np.ndarray) -> None:
        self._group_ids = np.asarray(group_ids)
        self._order = np.argsort(self._group_ids, kind="stable")
        self._sorted = self._group_ids[self._order]

    def expand(self, rows: np.ndarray) -> np.ndarray:
        groups = np.unique(self._group_ids[np.asarray(rows)])
        starts = np.searchsorted(self._sorted, groups, side="left")
        stops = np.searchsorted(self._sorted, groups, side="right")
        return np.sort(np.concatenate([self._order[a:b] for a, b in zip(starts, stops)]))


CONDITION_RANDOM_NEGATIVES = 12


def _mine_condition(source, group_ids: np.ndarray, config: FactorTrainingConfig, rng):
    return mine_condition_episodes(source, None, group_ids, config.condition_episodes_per_step, rng,
                                   num_hard=0, num_random=CONDITION_RANDOM_NEGATIVES)


def _episode_tensors(model: SharedFactorEncoder, img: torch.Tensor, txt: torch.Tensor, episodes, device):
    """Encode the unique rows of a batch of episodes; return their features, codes and row-local indices."""
    table = np.concatenate([episodes.anchor[:, None], episodes.supports, episodes.contrasts, episodes.candidates],
                           axis=1)
    rows, inverse = np.unique(table, return_inverse=True)
    inverse = torch.as_tensor(inverse.reshape(table.shape), device=device)
    s, c = episodes.supports.shape[1], episodes.contrasts.shape[1]
    index = {"anchor": inverse[:, 0], "supports": inverse[:, 1:1 + s], "contrasts": inverse[:, 1 + s:1 + s + c],
             "candidates": inverse[:, 1 + s + c:]}
    img_rows, txt_rows = img[rows].to(device), txt[rows].to(device)
    return img_rows, txt_rows, model.encode_image(img_rows), model.encode_text(txt_rows), index


def _aspect_tensors(model, img, txt, bank, take: np.ndarray, device):
    parts = (bank.anchor[take][:, None], bank.candidates[take], bank.pairs_a_img[take], bank.pairs_a_txt[take],
             bank.pairs_b_img[take], bank.pairs_b_txt[take])
    table = np.concatenate(parts, axis=1)
    rows, inverse = np.unique(table, return_inverse=True)
    inv = torch.as_tensor(inverse.reshape(table.shape), device=device)
    idx = {"anchor": inv[:, 0], "candidates": inv[:, 1:14], "pa_img": inv[:, 14:18], "pa_txt": inv[:, 18:22],
           "pb_img": inv[:, 22:26], "pb_txt": inv[:, 26:30]}
    img_rows, txt_rows = img[rows].to(device), txt[rows].to(device)
    return img_rows, txt_rows, model.encode_image(img_rows), model.encode_text(txt_rows), idx


def train_factors(
    img_features: np.ndarray,
    txt_features: np.ndarray,
    graph: csr_matrix,
    config: FactorTrainingConfig,
    device: str | None = None,
    group_ids: np.ndarray | None = None,
    condition_source=None,
    aspect_bank=None,
    history: dict | None = None,
    log_every: int = 50,
) -> tuple[SharedFactorEncoder, np.ndarray, np.ndarray]:
    """Fit one edge-sampled unique-node batch per epoch; encode all rows once.

    Graph consistency compares the mean of each node's two modality codes on
    the induced graph of the sampled nodes. All other losses see the same nodes.
    ``group_ids`` (one integer per row, e.g. painting / leakage-group id) is used only by the
    ``"infonce"`` agreement term, to exclude same-group pairs as negatives. InfoNCE REQUIRES it:
    the content graph is a mutual-kNN graph on image features, so same-painting edges are common,
    and silently treating them as negatives trains a recipe that is not the validated R3. Pass
    ``group_ids=np.arange(n)`` to opt out of masking explicitly.
    ``condition_source`` (rows 0 .. n-1 of the arrays passed in) is required exactly when
    ``config.lambda_condition > 0``. ``aspect_bank`` (an ``AspectEpisodes`` with local rows 0 .. n-1) is required
    exactly when ``config.lambda_aspect > 0`` and is not combined with the condition loss. ``history``, if a dict, receives per-step logs every ``log_every`` steps (plus
    the first and last).
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
    if config.agreement_level not in {"pair", "painting"}:
        raise ValueError(f"agreement_level must be 'pair' or 'painting', got {config.agreement_level!r}")
    if config.agreement_level == "painting" and not (config.painting_batches and config.agreement == "infonce"):
        raise ValueError("agreement_level='painting' needs painting_batches=True and agreement='infonce'")
    if (config.painting_batches or config.lambda_condition > 0) and group_ids is None:
        raise ValueError("painting_batches and the condition loss need group_ids (painting / leakage-group ids)")
    if config.lambda_condition < 0:
        raise ValueError("lambda_condition must be >= 0")
    if (config.lambda_condition > 0) != (condition_source is not None):
        raise ValueError("pass a condition_source exactly when lambda_condition > 0")
    if condition_source is not None and (condition_source.rows.min() < 0
                                         or condition_source.rows.max() >= len(img_features)):
        raise ValueError("condition_source rows must index the training rows (0 .. n-1), not global rows")
    if config.lambda_aspect < 0:
        raise ValueError("lambda_aspect must be >= 0")
    if config.aspect_tau_fixed and config.lambda_aspect <= 0:
        raise ValueError("aspect_tau_fixed needs lambda_aspect > 0")
    if (config.lambda_aspect > 0) != (aspect_bank is not None):
        raise ValueError("pass an aspect_bank exactly when lambda_aspect > 0")
    if config.lambda_aspect > 0 and config.lambda_condition > 0:
        raise ValueError("the value-condition loss and the aspect loss are not combined")
    if aspect_bank is not None and (aspect_bank.rows().min() < 0 or aspect_bank.rows().max() >= len(img_features)):
        raise ValueError("aspect_bank rows must index the training rows (0 .. n-1)")

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

    group_rows = GroupRows(group_ids) if config.painting_batches else None
    log_tau = None
    if config.lambda_condition > 0:
        condition_rng = np.random.default_rng([config.seed, 1])
        first_episodes = _mine_condition(condition_source, group_ids, config, condition_rng)
        with torch.no_grad():                                    # tau := std of step-0 scores (unit-scale logits)
            img_rows, txt_rows, ic, tc, index = _episode_tensors(model, img, txt, first_episodes, selected_device)
            scores = naive_episode_scores(img_rows, txt_rows, ic, tc, index["anchor"], index["supports"],
                                          index["contrasts"], index["candidates"], config.condition_beta)
            std = float(torch.cat([scores["i2t"].ravel(), scores["t2i"].ravel()]).std())
        log_tau = torch.nn.Parameter(torch.tensor(math.log(max(std, 1e-6)), device=selected_device))
        optimizer.add_param_group({"params": [log_tau]})
    if config.lambda_aspect > 0:
        aspect_rng = np.random.default_rng([config.seed, 2])
        first_take = aspect_rng.choice(len(aspect_bank.anchor), config.aspect_episodes_per_step, replace=False)
        with torch.no_grad():                                    # tau := std of step-0 scores (unit-scale logits)
            img_rows, txt_rows, ic, tc, idx = _aspect_tensors(model, img, txt, aspect_bank, first_take,
                                                              selected_device)
            scores = aspect_episode_scores(img_rows, txt_rows, ic, tc, idx, config.aspect_beta)
            std = float(torch.cat([scores[k].ravel() for k in sorted(scores)]).std())
        if config.aspect_tau_fixed:
            log_tau = torch.tensor(math.log(max(std, 1e-6)), device=selected_device)   # constant, not optimised
        else:
            log_tau = torch.nn.Parameter(torch.tensor(math.log(max(std, 1e-6)), device=selected_device))
            optimizer.add_param_group({"params": [log_tau]})

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_rng = np.random.default_rng(config.seed + epoch)
        sampled = edges[epoch_rng.choice(
            len(edges), size=config.batch_size, replace=len(edges) < config.batch_size
        )]
        node_ids = np.unique(sampled.reshape(-1))
        if group_rows is not None:
            node_ids = group_rows.expand(node_ids)
        img_batch = img[node_ids].to(selected_device)
        txt_batch = txt[node_ids].to(selected_device)
        img_codes = model.encode_image(img_batch)
        txt_codes = model.encode_text(txt_batch)
        local_graph = graph[node_ids][:, node_ids].tocsr()
        local_ids = np.arange(len(node_ids), dtype=np.int64)

        if config.agreement_level == "painting":
            agreement = painting_infonce_loss(
                img_codes, txt_codes, torch.as_tensor(group_ids[node_ids], device=selected_device),
                config.infonce_temperature)
        elif config.agreement == "cosine":
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
        condition_loss = None
        if config.lambda_condition > 0:
            episodes = first_episodes if epoch == 1 else _mine_condition(condition_source, group_ids, config,
                                                                         condition_rng)
            img_rows, txt_rows, ic, tc, index = _episode_tensors(model, img, txt, episodes, selected_device)
            condition_loss = naive_episode_loss(
                img_rows, txt_rows, ic, tc, **index,
                positive_mask=torch.as_tensor(episodes.positive_mask, device=selected_device),
                beta=config.condition_beta, log_tau=log_tau)
            loss = loss + config.lambda_condition * condition_loss
        aspect_loss = None
        if config.lambda_aspect > 0:
            take = first_take if epoch == 1 else aspect_rng.choice(len(aspect_bank.anchor),
                                                                    config.aspect_episodes_per_step, replace=False)
            img_rows, txt_rows, ic, tc, idx = _aspect_tensors(model, img, txt, aspect_bank, take, selected_device)
            aspect_loss = aspect_episode_loss(aspect_episode_scores(img_rows, txt_rows, ic, tc, idx,
                                                                    config.aspect_beta), log_tau, config.lambda_swap)
            loss = loss + config.lambda_aspect * aspect_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        print(f"factor epoch={epoch} loss={loss.item():.6f}", flush=True)
        if history is not None and (epoch % log_every == 0 or epoch in (1, config.epochs)):
            history.setdefault("step", []).append(epoch)
            history.setdefault("loss", []).append(loss.item())
            history.setdefault("agreement", []).append(agreement.item())
            history.setdefault("batch_rows", []).append(int(len(node_ids)))
            if condition_loss is not None:
                history.setdefault("condition_loss", []).append(condition_loss.item())
                history.setdefault("tau", []).append(float(log_tau.detach().exp()))
            if aspect_loss is not None:
                history.setdefault("aspect_loss", []).append(aspect_loss.item())
                history.setdefault("tau", []).append(float(log_tau.detach().exp()))

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
