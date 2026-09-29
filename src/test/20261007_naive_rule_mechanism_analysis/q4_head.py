"""Task 9 Q4: condition-head capacity, recovery objective, and ranking objective.

All input codes and episodes are supplied by the caller. No factor model is
fit here. Ranking and swap evaluations use Task 7's score_pool and pair pools.
"""

import copy
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from torch.nn import functional as F
from torch.utils.data import WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "20261005_condition_ranking_evaluation"))

from run_ranking_eval import episode_arrays, score_pool  # noqa: E402
from src.model.condition import ConditionEncoder  # noqa: E402

SEED = 42
EPOCHS = 300
BATCH_SIZE = 64
LR = 0.01
RANKING_TEMPERATURE = 0.1
CHECKPOINTS = frozenset(range(0, EPOCHS + 1, 25))


def l1_weights(weights: np.ndarray) -> np.ndarray:
    """Normalize nonnegative weights per episode, leaving zero rows at zero."""
    sums = np.abs(weights).sum(axis=-1, keepdims=True)
    return np.divide(weights, sums, out=np.zeros_like(weights), where=sums > 0)


def positive_ranks(scores: np.ndarray) -> np.ndarray:
    """Use Task 7's strict-greater positive-rank convention."""
    return 1 + (scores[:, 1:] > scores[:, :1]).sum(axis=1)


def _condition_tensors(img_codes: np.ndarray, txt_codes: np.ndarray,
                       episodes: list) -> tuple[tuple[torch.Tensor, ...], np.ndarray]:
    supports, contrasts, _, targets = episode_arrays(episodes)
    values = tuple(torch.as_tensor(array[index], dtype=torch.float32) for array, index in (
        (img_codes, supports), (txt_codes, supports),
        (img_codes, contrasts), (txt_codes, contrasts),
    ))
    return values, targets


def _predict(model: ConditionEncoder, inputs: tuple[torch.Tensor, ...]) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return np.concatenate([
            torch.vmap(model)(*(value[start:start + 256] for value in inputs)).cpu().numpy()
            for start in range(0, len(inputs[0]), 256)
        ], axis=0)


def _naive(img_codes: np.ndarray, txt_codes: np.ndarray, episodes: list) -> np.ndarray:
    supports, contrasts, _, _ = episode_arrays(episodes)
    pair = 0.5 * (img_codes + txt_codes)
    return np.maximum(pair[supports].mean(axis=1) - pair[contrasts].mean(axis=1), 0.0)


def _ranking(img_features: np.ndarray, txt_features: np.ndarray,
             img_codes: np.ndarray, txt_codes: np.ndarray, episodes: list,
             weights: np.ndarray, beta: float) -> dict:
    _, _, candidates, _ = episode_arrays(episodes)
    anchors = np.asarray([ep.anchor_idx for ep in episodes], dtype=np.int64)
    result = {}
    for direction, af, cf, ac, cc in (
        ("i2t", img_features, txt_features, img_codes, txt_codes),
        ("t2i", txt_features, img_features, txt_codes, img_codes),
    ):
        scores, _, _ = score_pool(af[anchors], cf[candidates], ac[anchors],
                                  cc[candidates], weights, beta)
        ranks = positive_ranks(scores)
        result[direction] = {
            "recall1": float(np.mean(ranks == 1)),
            "recall3": float(np.mean(ranks <= 3)),
            "count1": int(np.count_nonzero(ranks == 1)),
            "count3": int(np.count_nonzero(ranks <= 3)),
        }
    return result


def _swaps(img_features: np.ndarray, txt_features: np.ndarray,
           img_codes: np.ndarray, txt_codes: np.ndarray, episodes: list,
           pairs: list[tuple[int, int]], weights: np.ndarray, beta: float) -> dict:
    result = {}
    for direction, af, cf, ac, cc in (
        ("i2t", img_features, txt_features, img_codes, txt_codes),
        ("t2i", txt_features, img_features, txt_codes, img_codes),
    ):
        reversals = 0
        any_change = 0
        for a, b in pairs:
            ep_a, ep_b = episodes[a], episodes[b]
            pool = list(dict.fromkeys([
                ep_a.positive_idx, ep_b.positive_idx,
                *ep_a.hard_negative_idxs, *ep_a.condition_distractor_idxs,
                *ep_a.anchor_distractor_idxs, *ep_b.hard_negative_idxs,
                *ep_b.condition_distractor_idxs, *ep_b.anchor_distractor_idxs,
            ]))
            anchor_feat = np.broadcast_to(af[ep_a.anchor_idx], (2, af.shape[1]))
            pool_feat = np.broadcast_to(cf[pool], (2, len(pool), cf.shape[1]))
            anchor_code = np.broadcast_to(ac[ep_a.anchor_idx], (2, ac.shape[1]))
            pool_code = np.broadcast_to(cc[pool], (2, len(pool), cc.shape[1]))
            scores, _, _ = score_pool(anchor_feat, pool_feat, anchor_code, pool_code,
                                      weights[[a, b]], beta)
            any_change += int(not np.array_equal(np.argsort(-scores[0]),
                                                  np.argsort(-scores[1])))
            reversals += int(scores[0, 0] - scores[0, 1] > 1e-8
                             and scores[1, 1] - scores[1, 0] > 1e-8)
        result[direction] = {
            "appropriate_reversals": int(reversals),
            "reversal_rate": float(reversals / len(pairs)),
            "any_rank_change": float(any_change / len(pairs)),
        }
    return result


def _measure(img_features: np.ndarray, txt_features: np.ndarray,
             img_codes: np.ndarray, txt_codes: np.ndarray, episodes: list,
             pairs: list[tuple[int, int]], weights: np.ndarray, beta: float) -> dict:
    normalized = l1_weights(weights)
    return {
        "ranking": _ranking(img_features, txt_features, img_codes, txt_codes,
                            episodes, normalized, beta),
        "swaps": _swaps(img_features, txt_features, img_codes, txt_codes,
                        episodes, pairs, normalized, beta),
    }


def _recovery_loss(logits: np.ndarray, targets: np.ndarray) -> float:
    with torch.no_grad():
        return float(F.cross_entropy(torch.as_tensor(logits, dtype=torch.float32),
                                     torch.as_tensor(targets, dtype=torch.long)).item())


def _balanced_recovery_loss(logits: np.ndarray, targets: np.ndarray) -> float:
    with torch.no_grad():
        per_episode = F.cross_entropy(torch.as_tensor(logits, dtype=torch.float32),
                                      torch.as_tensor(targets, dtype=torch.long),
                                      reduction="none").numpy()
    return float(np.mean([per_episode[targets == target].mean()
                          for target in np.unique(targets)]))


def regression_r2(prediction: np.ndarray, target: np.ndarray) -> float | None:
    """Variance-weighted R² after centering each factor across episodes."""
    residual = float(np.square(prediction - target).sum())
    total = float(np.square(target - target.mean(axis=0, keepdims=True)).sum())
    return float(1 - residual / total) if total > 0 else None


def _calibrated_recovery(logits_train: np.ndarray, targets_train: np.ndarray,
                         logits_held: np.ndarray, targets_held: np.ndarray) -> dict:
    """Fit one positive logit scale on train episodes, evaluate held CE."""
    fit = minimize_scalar(
        lambda log_scale: _recovery_loss(logits_train * np.exp(log_scale), targets_train),
        bounds=(-8, 8), method="bounded",
    )
    scale = float(np.exp(fit.x))
    return {"train_fitted_logit_scale": scale,
            "held_ce": _recovery_loss(logits_held * scale, targets_held),
            "held_argmax_accuracy": float(np.mean(logits_held.argmax(axis=1) == targets_held))}


def _ranking_components(img_features: np.ndarray, txt_features: np.ndarray,
                        img_codes: np.ndarray, txt_codes: np.ndarray,
                        episodes: list) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Precompute only the fixed score terms; gradients flow through weights."""
    _, _, candidates, _ = episode_arrays(episodes)
    anchors = np.asarray([ep.anchor_idx for ep in episodes], dtype=np.int64)
    out = []
    for af, cf, ac, cc in (
        (img_features, txt_features, img_codes, txt_codes),
        (txt_features, img_features, txt_codes, img_codes),
    ):
        zero = np.zeros((len(episodes), img_codes.shape[1]), dtype=np.float32)
        _, cosine, _ = score_pool(af[anchors], cf[candidates], ac[anchors],
                                  cc[candidates], zero, 1.0)
        product = ac[anchors, None, :] * cc[candidates]
        out.append((torch.as_tensor(cosine, dtype=torch.float32),
                    torch.as_tensor(product, dtype=torch.float32)))
    return tuple(out)


def analyze_heads(img_features: np.ndarray, txt_features: np.ndarray,
                  img_codes: np.ndarray, txt_codes: np.ndarray,
                  train_episodes: list, held_episodes: list,
                  swap_pairs: list[tuple[int, int]], beta: float,
                  trained_head: ConditionEncoder) -> dict:
    """Fit and evaluate all three Q4 heads on caller-supplied Task 7 data."""
    if not train_episodes or not held_episodes or not swap_pairs:
        raise ValueError("Q4 requires training episodes, held episodes, and swap pairs")
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    train_inputs, train_targets = _condition_tensors(img_codes, txt_codes, train_episodes)
    held_inputs, held_targets = _condition_tensors(img_codes, txt_codes, held_episodes)
    naive_train = _naive(img_codes, txt_codes, train_episodes)
    naive_held = _naive(img_codes, txt_codes, held_episodes)
    naive_eval = _measure(img_features, txt_features, img_codes, txt_codes,
                          held_episodes, swap_pairs, naive_held, beta)

    # Q4a: literal raw-output MSE, without normalizing the regression target.
    mse_head = ConditionEncoder()
    mse_opt = torch.optim.Adam(mse_head.parameters(), lr=LR)
    naive_target = torch.as_tensor(naive_train, dtype=torch.float32)
    mse_generator = torch.Generator().manual_seed(SEED)
    for _ in range(EPOCHS):
        for indices in torch.randperm(len(train_episodes), generator=mse_generator).split(BATCH_SIZE):
            prediction = torch.vmap(mse_head)(*(value[indices] for value in train_inputs))
            loss = F.mse_loss(prediction, naive_target[indices])
            mse_opt.zero_grad(set_to_none=True)
            loss.backward()
            mse_opt.step()
    mse_held = _predict(mse_head, held_inputs)
    r2 = regression_r2(mse_held, naive_held)
    mse_eval = _measure(img_features, txt_features, img_codes, txt_codes,
                        held_episodes, swap_pairs, mse_held, beta)

    # Q4b: exactly Task 7's replacement-sampled inverse-frequency CE recipe.
    counts = np.bincount(train_targets, minlength=img_codes.shape[1])
    if np.any(counts == 0):
        raise ValueError("Recovery training needs episodes for every factor")
    sampler = WeightedRandomSampler(
        torch.as_tensor(1.0 / counts[train_targets], dtype=torch.double),
        num_samples=len(train_episodes), replacement=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    recovery_head = copy.deepcopy(mse_head)
    recovery_opt = torch.optim.Adam(recovery_head.parameters(), lr=LR)
    train_labels = torch.as_tensor(train_targets, dtype=torch.long)
    checkpoints = []
    trained_logits = _predict(copy.deepcopy(trained_head).cpu(), held_inputs)
    trained_train_logits = _predict(copy.deepcopy(trained_head).cpu(), train_inputs)
    naive_init_recovery = _calibrated_recovery(
        _predict(mse_head, train_inputs), train_targets, mse_held, held_targets)
    task7_recovery = _calibrated_recovery(
        trained_train_logits, train_targets, trained_logits, held_targets)
    for epoch in range(EPOCHS + 1):
        if epoch in CHECKPOINTS:
            train_logits = _predict(recovery_head, train_inputs)
            held_logits = _predict(recovery_head, held_inputs)
            measured = _measure(img_features, txt_features, img_codes, txt_codes,
                                held_episodes, swap_pairs, held_logits, beta)
            checkpoints.append({
                "epoch": epoch,
                "train_loss": _balanced_recovery_loss(train_logits, train_targets),
                "train_unweighted_loss": _recovery_loss(train_logits, train_targets),
                "held_loss": _recovery_loss(held_logits, held_targets),
                **measured,
            })
        if epoch == EPOCHS:
            break
        recovery_head.train()
        sampled = np.fromiter(sampler, dtype=np.int64)
        for indices in np.array_split(sampled, range(BATCH_SIZE, len(sampled), BATCH_SIZE)):
            prediction = torch.vmap(recovery_head)(*(value[indices] for value in train_inputs))
            loss = F.cross_entropy(prediction, train_labels[indices])
            recovery_opt.zero_grad(set_to_none=True)
            loss.backward()
            recovery_opt.step()

    # Q4c: fresh head, bidirectional 13-candidate InfoNCE with positive slot 0.
    torch.manual_seed(SEED)
    ranking_head = ConditionEncoder()
    ranking_opt = torch.optim.Adam(ranking_head.parameters(), lr=LR)
    components = _ranking_components(img_features, txt_features, img_codes,
                                     txt_codes, train_episodes)
    ranking_generator = torch.Generator().manual_seed(SEED)
    for _ in range(EPOCHS):
        ranking_head.train()
        for indices in torch.randperm(len(train_episodes), generator=ranking_generator).split(BATCH_SIZE):
            weights = torch.vmap(ranking_head)(*(value[indices] for value in train_inputs))
            weights = weights / weights.abs().sum(dim=1, keepdim=True).clamp_min(1e-12)
            targets = torch.zeros(len(indices), dtype=torch.long)
            direction_losses = []
            for cosine, product in components:
                scores = beta * cosine[indices] + (product[indices] * weights[:, None, :]).sum(dim=-1)
                direction_losses.append(F.cross_entropy(scores / RANKING_TEMPERATURE, targets))
            loss = 0.5 * (direction_losses[0] + direction_losses[1])
            ranking_opt.zero_grad(set_to_none=True)
            loss.backward()
            ranking_opt.step()
    ranking_held = _predict(ranking_head, held_inputs)
    ranking_eval = _measure(img_features, txt_features, img_codes, txt_codes,
                            held_episodes, swap_pairs, ranking_held, beta)

    return {
        "metadata": {
            "seed": SEED, "beta": float(beta), "train_episodes": len(train_episodes),
            "held_episodes": len(held_episodes), "swap_pairs": len(swap_pairs),
            "mse": {"optimizer": "Adam", "learning_rate": LR,
                    "epochs": EPOCHS, "batch_size": BATCH_SIZE, "sampling": "shuffle"},
            "recovery": {"optimizer": "Adam", "learning_rate": LR,
                         "epochs": EPOCHS, "batch_size": BATCH_SIZE,
                         "sampling": "inverse-class-frequency replacement", "checkpoint_every": 25,
                         "train_loss_definition": "class-macro-mean CE"},
            "ranking": {"optimizer": "Adam", "learning_rate": LR,
                        "epochs": EPOCHS, "batch_size": BATCH_SIZE,
                        "temperature": RANKING_TEMPERATURE, "sampling": "shuffle",
                        "direction_reduction": "mean", "target_slot": 0},
            "evaluation_weights": "L1-normalized per episode",
        },
        "naive": naive_eval,
        "q4a": {"held_r2": r2, "held_mse": float(np.mean(np.square(mse_held - naive_held))),
                **mse_eval},
        "q4b": {"naive_init_held_recovery_loss": checkpoints[0]["held_loss"],
                "task7_trained_head_held_recovery_loss": _recovery_loss(trained_logits, held_targets),
                "naive_init_calibrated_recovery": naive_init_recovery,
                "task7_trained_calibrated_recovery": task7_recovery,
                "checkpoints": checkpoints},
        "q4c": ranking_eval,
    }
