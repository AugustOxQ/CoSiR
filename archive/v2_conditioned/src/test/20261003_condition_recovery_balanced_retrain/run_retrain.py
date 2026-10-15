"""Re-test real-data condition recovery with more episodes and balanced batches.

Run from the repository root in the CoSiR environment. The targeted factor is
used only for this validation objective, as in Tasks 3 and 4.
"""

import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import WeightedRandomSampler
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "20261001_condition_interface_validation"))

from run_validation import SEED, load_real_features, prepare_real_codes  # noqa: E402
from src.model.condition import ConditionEncoder  # noqa: E402
from src.train.episodes import EpisodeMiningConfig, mine_episodes  # noqa: E402

NUM_EPISODES = 4096
RECOVERY_EPOCHS = 300
RECOVERY_BATCH_SIZE = 64
RECOVERY_LR = 0.01
BASELINE_DEAD_FACTORS = {0, 1, 2, 4, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18, 20, 23, 24, 26, 31}


def make_balanced_sampler(train_targets: torch.Tensor, seed: int) -> tuple[WeightedRandomSampler, list[int]]:
    """Sample one training-set-sized epoch with inverse class-frequency weights."""
    counts = torch.bincount(train_targets).tolist()
    weights = torch.tensor([1.0 / counts[int(target)] for target in train_targets], dtype=torch.double)
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(weights, num_samples=len(train_targets), replacement=True,
                                    generator=generator)
    return sampler, counts


def summarize_recovery(truth: np.ndarray, predictions: np.ndarray, n_factors: int) -> dict:
    """Summarize held-out recovery and prediction distribution by factor."""
    confusion = np.zeros((n_factors, n_factors), dtype=np.int64)
    np.add.at(confusion, (truth, predictions), 1)
    per_factor = []
    for factor in range(n_factors):
        count = int(confusion[factor].sum())
        correct = int(confusion[factor, factor])
        per_factor.append({"factor": factor, "correct": correct, "count": count,
                           "accuracy": correct / count if count else None})
    predicted_counts = confusion.sum(axis=0)
    top5 = sorted(range(n_factors), key=lambda factor: (-predicted_counts[factor], factor))[:5]
    dead = [row["factor"] for row in per_factor if row["count"] and row["correct"] == 0]
    return {
        "held_out_count": len(truth),
        "correct": int(np.trace(confusion)),
        "accuracy": float(np.trace(confusion) / len(truth)),
        "distinct_predicted_factors": int(np.count_nonzero(predicted_counts)),
        "predicted_counts": predicted_counts.tolist(),
        "top5_factors": top5,
        "top5_count": int(predicted_counts[top5].sum()),
        "top5_fraction": float(predicted_counts[top5].sum() / len(truth)),
        "dead_factors": dead,
        "per_factor": per_factor,
    }


def recover_balanced(img_codes: np.ndarray, txt_codes: np.ndarray, episodes: list) -> dict:
    """Train the unchanged ConditionEncoder with balanced replacement sampling."""
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    permutation = np.random.default_rng(SEED).permutation(len(episodes))
    train_count = int(0.8 * len(episodes))
    train_indices = permutation[:train_count]
    held_indices = permutation[train_count:]
    if any(len(ep.support_idxs) != 4 or len(ep.contrast_idxs) != 4 for ep in episodes):
        raise ValueError("Recovery requires four support and four contrast pairs per episode")
    support_indices = np.asarray([ep.support_idxs for ep in episodes], dtype=np.int64)
    contrast_indices = np.asarray([ep.contrast_idxs for ep in episodes], dtype=np.int64)
    support_img = torch.as_tensor(img_codes[support_indices], dtype=torch.float32)
    support_txt = torch.as_tensor(txt_codes[support_indices], dtype=torch.float32)
    contrast_img = torch.as_tensor(img_codes[contrast_indices], dtype=torch.float32)
    contrast_txt = torch.as_tensor(txt_codes[contrast_indices], dtype=torch.float32)
    targets = torch.as_tensor([ep.targeted_factor for ep in episodes], dtype=torch.long)
    sampler, train_class_counts = make_balanced_sampler(targets[train_indices], SEED)
    encoder = ConditionEncoder()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=RECOVERY_LR)

    def weights(indices: np.ndarray) -> torch.Tensor:
        return torch.vmap(encoder)(support_img[indices], support_txt[indices],
                                   contrast_img[indices], contrast_txt[indices])

    def predict(indices: np.ndarray) -> np.ndarray:
        encoder.eval()
        with torch.no_grad():
            return np.concatenate([
                weights(indices[start:start + 512]).argmax(dim=1).numpy()
                for start in range(0, len(indices), 512)
            ])

    with torch.no_grad():
        first = int(train_indices[0])
        direct = encoder(support_img[first], support_txt[first],
                         contrast_img[first], contrast_txt[first])
        if not torch.allclose(weights(np.asarray([first]))[0], direct, atol=1e-6):
            raise AssertionError("Vectorized ConditionEncoder disagrees with direct forward")

    dynamics = []
    for epoch in range(1, RECOVERY_EPOCHS + 1):
        encoder.train()
        # Sampler indices are positions in the training split, not episode IDs.
        sampled_indices = train_indices[np.fromiter(sampler, dtype=np.int64)]
        for start in range(0, len(sampled_indices), RECOVERY_BATCH_SIZE):
            batch_indices = sampled_indices[start:start + RECOVERY_BATCH_SIZE]
            loss = F.cross_entropy(weights(batch_indices), targets[batch_indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if epoch % 50 == 0:
            train_predictions = predict(train_indices)
            held_predictions = predict(held_indices)
            point = {
                "epoch": epoch,
                "train_correct": int(np.count_nonzero(train_predictions == targets[train_indices].numpy())),
                "train_count": len(train_indices),
                "held_out_correct": int(np.count_nonzero(held_predictions == targets[held_indices].numpy())),
                "held_out_count": len(held_indices),
                "distinct_held_out_predictions": int(len(np.unique(held_predictions))),
            }
            dynamics.append(point)
            print(f"Epoch {epoch}/{RECOVERY_EPOCHS}: train={point['train_correct']}/{point['train_count']}, "
                  f"held-out={point['held_out_correct']}/{point['held_out_count']}, "
                  f"distinct={point['distinct_held_out_predictions']}/{img_codes.shape[1]}", flush=True)

    truth = targets[held_indices].numpy()
    result = summarize_recovery(truth, held_predictions, img_codes.shape[1])
    current_dead = set(result["dead_factors"])
    result.update({
        "seed": SEED,
        "train_episodes": len(train_indices),
        "training": {"epochs": RECOVERY_EPOCHS, "batch_size": RECOVERY_BATCH_SIZE,
                     "adam_lr": RECOVERY_LR, "objective": "cross_entropy(w(c), targeted_factor), validation only",
                     "sampler": "WeightedRandomSampler, inverse training-class frequency, replacement=True",
                     "samples_per_epoch": len(train_indices), "train_class_counts": train_class_counts},
        "dynamics": dynamics,
        "baseline_dead_still_dead": sorted(BASELINE_DEAD_FACTORS & current_dead),
        "newly_dead": sorted(current_dead - BASELINE_DEAD_FACTORS),
        "baseline_dead_recovered": sorted(BASELINE_DEAD_FACTORS - current_dead),
    })
    print(f"Held-out recovery: {result['correct']}/{result['held_out_count']}; "
          f"distinct: {result['distinct_predicted_factors']}/{img_codes.shape[1]}; "
          f"top-5 share: {result['top5_fraction']:.2%}; dead: {len(current_dead)}", flush=True)
    return result


def main() -> None:
    started = perf_counter()
    img_features, txt_features = load_real_features()
    img_codes, txt_codes, prerequisite = prepare_real_codes(img_features, txt_features)
    del img_features, txt_features
    config = EpisodeMiningConfig(seed=SEED)
    # Mining repeatedly ranks small candidate pools; avoid BLAS thread overhead.
    with threadpool_limits(limits=1, user_api="blas"):
        episodes = mine_episodes(img_codes, txt_codes, config, num_episodes=NUM_EPISODES)
    mining = {"episode_count": len(episodes), "config": vars(config),
              "chosen_factors": sorted({ep.targeted_factor for ep in episodes})}
    print(f"Mined {len(episodes)} episodes across {len(mining['chosen_factors'])} factors", flush=True)
    recovery = recover_balanced(img_codes, txt_codes, episodes)
    print("RESULT_JSON=" + json.dumps({"samples": len(img_codes), "prerequisite": prerequisite,
                                        "mining": mining, "recovery": recovery,
                                        "seconds": perf_counter() - started}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
