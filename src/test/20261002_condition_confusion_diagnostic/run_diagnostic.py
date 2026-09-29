"""Diagnose Task 3's held-out condition-recovery confusion on real ArtELingo codes.

Run from the repository root in the CoSiR environment. The targeted factor is
used only for the same validation-only recovery objective as Task 3.
"""

import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "20261001_condition_interface_validation"))

from run_validation import (  # noqa: E402
    NUM_EPISODES,
    RECOVERY_BATCH_SIZE,
    RECOVERY_EPOCHS,
    RECOVERY_LR,
    SEED,
    load_real_features,
    mine_and_measure,
    prepare_real_codes,
)
from src.model.condition import ConditionEncoder  # noqa: E402


def factor_column_cosines(pair_codes: np.ndarray) -> np.ndarray:
    """Return uncentered cosine similarities of complete real-data factor columns."""
    columns = np.asarray(pair_codes, dtype=np.float64)
    if columns.ndim != 2 or not np.isfinite(columns).all():
        raise ValueError("pair_codes must be a finite (samples, factors) matrix")
    norms = np.linalg.norm(columns, axis=0)
    if np.any(norms == 0):
        raise ValueError("Cannot compute cosine for a zero-activation factor")
    normalized = columns / norms
    return normalized.T @ normalized


def summarize_confusion(
    truth: np.ndarray, predictions: np.ndarray, cosine: np.ndarray
) -> dict:
    """Measure complete confusion, prediction skew, and dead-factor neighbors."""
    truth = np.asarray(truth, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    cosine = np.asarray(cosine)
    if cosine.ndim != 2 or cosine.shape[0] != cosine.shape[1]:
        raise ValueError("cosine must be square")
    n_factors = cosine.shape[0]
    if (truth.shape != predictions.shape or truth.ndim != 1 or not len(truth)
            or np.any((truth < 0) | (truth >= n_factors))
            or np.any((predictions < 0) | (predictions >= n_factors))):
        raise ValueError("truth and predictions must be nonempty, matching factor ID vectors")
    confusion = np.zeros((n_factors, n_factors), dtype=np.int64)
    np.add.at(confusion, (truth, predictions), 1)
    predicted_counts = confusion.sum(axis=0)
    top5 = sorted(range(n_factors), key=lambda factor: (-predicted_counts[factor], factor))[:5]
    dead_details = []
    for factor in range(n_factors):
        count = int(confusion[factor].sum())
        if count == 0 or confusion[factor, factor] != 0:
            continue
        wrong_factors = sorted(
            (other for other in range(n_factors) if confusion[factor, other] > 0),
            key=lambda other: (-confusion[factor, other], other),
        )
        neighbors = sorted(
            (other for other in range(n_factors) if other != factor),
            key=lambda other: (-cosine[factor, other], other),
        )[:3]
        dead_details.append({
            "factor": factor,
            "held_out_count": count,
            "wrong_predictions": [
                {"factor": other, "count": int(confusion[factor, other]),
                 "proportion": float(confusion[factor, other] / count)}
                for other in wrong_factors
            ],
            "top_wrong_factor": wrong_factors[0],
            "top3_correlated": [
                {"factor": other, "cosine": float(cosine[factor, other])}
                for other in neighbors
            ],
            "top_wrong_is_top3_correlated": wrong_factors[0] in neighbors,
        })
    dead_count = len(dead_details)
    correlated_count = sum(item["top_wrong_is_top3_correlated"] for item in dead_details)
    return {
        "held_out_count": len(truth),
        "correct": int(np.trace(confusion)),
        "confusion": confusion.tolist(),
        "predicted_counts": predicted_counts.tolist(),
        "top5_factors": top5,
        "top5_count": int(predicted_counts[top5].sum()),
        "top5_fraction": float(predicted_counts[top5].sum() / len(truth)),
        "dead_factors": [item["factor"] for item in dead_details],
        "dead_details": dead_details,
        "dead_top3_correlated_count": correlated_count,
        "dead_top3_correlated_fraction": correlated_count / dead_count if dead_count else None,
    }


def recover_and_diagnose(img_codes: np.ndarray, txt_codes: np.ndarray, episodes: list) -> dict:
    """Repeat Task 3's recovery training, then retain every held-out prediction."""
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    rng = np.random.default_rng(SEED)
    permutation = rng.permutation(len(episodes))
    train_count = int(0.8 * len(episodes))
    train_indices = permutation[:train_count]
    held_indices = permutation[train_count:]
    # Defaults give fixed-size nonempty sets, permitting vectorized public forward.
    if any(len(ep.support_idxs) != 4 or len(ep.contrast_idxs) != 4 for ep in episodes):
        raise ValueError("Recovery requires four support and four contrast pairs per episode")
    support_indices = np.asarray([ep.support_idxs for ep in episodes], dtype=np.int64)
    contrast_indices = np.asarray([ep.contrast_idxs for ep in episodes], dtype=np.int64)
    support_img = torch.as_tensor(img_codes[support_indices], dtype=torch.float32)
    support_txt = torch.as_tensor(txt_codes[support_indices], dtype=torch.float32)
    contrast_img = torch.as_tensor(img_codes[contrast_indices], dtype=torch.float32)
    contrast_txt = torch.as_tensor(txt_codes[contrast_indices], dtype=torch.float32)
    targets = torch.as_tensor([ep.targeted_factor for ep in episodes], dtype=torch.long)

    encoder = ConditionEncoder()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=RECOVERY_LR)
    def weights(indices: np.ndarray) -> torch.Tensor:
        return torch.vmap(encoder)(support_img[indices], support_txt[indices],
                                   contrast_img[indices], contrast_txt[indices])

    # Check vmap uses the same public forward behavior as one episode at a time.
    with torch.no_grad():
        first = int(train_indices[0])
        direct = encoder(support_img[first], support_txt[first],
                         contrast_img[first], contrast_txt[first])
        if not torch.allclose(weights(np.asarray([first]))[0], direct, atol=1e-6):
            raise AssertionError("Vectorized ConditionEncoder disagrees with direct forward")
    for epoch in range(RECOVERY_EPOCHS):
        encoder.train()
        epoch_indices = rng.permutation(train_indices)
        for start in range(0, len(epoch_indices), RECOVERY_BATCH_SIZE):
            batch_indices = epoch_indices[start:start + RECOVERY_BATCH_SIZE]
            loss = F.cross_entropy(weights(batch_indices), targets[batch_indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if epoch in (0, 24, 49, 74, 99):
            print(f"Recovery epoch {epoch + 1}/{RECOVERY_EPOCHS}: train_batch_loss={loss.item():.6f}", flush=True)

    encoder.eval()
    with torch.no_grad():
        predictions = weights(held_indices).argmax(dim=1).numpy()
    truth = targets[held_indices].numpy()
    pair_codes = 0.5 * (img_codes + txt_codes)
    cosine = factor_column_cosines(pair_codes)
    result = summarize_confusion(truth, predictions, cosine)
    result["held_out_predictions"] = [
        {"episode": int(index), "true_factor": int(actual), "predicted_factor": int(predicted)}
        for index, actual, predicted in zip(held_indices, truth, predictions)
    ]
    result["train_episodes"] = len(train_indices)
    result["seed"] = SEED
    print(f"Held-out recovery: {result['correct']}/{len(truth)}; "
          f"dead factors: {len(result['dead_factors'])}; "
          f"top-5 prediction share: {result['top5_fraction']:.2%}; "
          f"dead-factor neighbor hits: {result['dead_top3_correlated_count']}/{len(result['dead_factors'])}",
          flush=True)
    return result


def main() -> None:
    started = perf_counter()
    img_features, txt_features = load_real_features()
    img_codes, txt_codes, prerequisite = prepare_real_codes(img_features, txt_features)
    del img_features, txt_features
    episodes, mining = mine_and_measure(img_codes, txt_codes)
    if len(episodes) != NUM_EPISODES:
        raise AssertionError("Task 3 episode count changed")
    diagnostic = recover_and_diagnose(img_codes, txt_codes, episodes)
    print("RESULT_JSON=" + json.dumps({"samples": len(img_codes), "prerequisite": prerequisite,
                                        "mining": mining, "diagnostic": diagnostic,
                                        "seconds": perf_counter() - started}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
