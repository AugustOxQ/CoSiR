"""Validate episode mining and condition recovery on real ArtELingo codes.

Run from the repository root in the CoSiR environment. The targeted factor is
used only for this validation objective; it is not a stage (d) training signal.
"""

import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.model.communities import community_stats, detect_communities
from src.model.condition import ConditionEncoder
from src.model.graph import GraphConfig, build_content_graph
from src.train.episodes import EpisodeMiningConfig, mine_episodes
from src.train.stage1 import Stage1Config, train_stage1
from src.train.train_factors import FactorTrainingConfig, train_factors
from src.utils import FeatureManager


FEATURE_DIR = "/data/SSD2/pre_extract/artelingo/features"
ANNOTATIONS_PATH = Path("/data/PDD/artelingo/artelingo_train.json")
EXPECTED_SAMPLES = 308_723
NUM_EPISODES = 1024
RECOVERY_EPOCHS = 100
RECOVERY_BATCH_SIZE = 64
RECOVERY_LR = 0.05
SEED = 42


def load_real_features() -> tuple[np.ndarray, np.ndarray]:
    """Apply the validated FeatureManager/annotation positional join."""
    manager = FeatureManager(storage_dir=FEATURE_DIR)
    with ANNOTATIONS_PATH.open() as file:
        annotations = json.load(file)
    if not (len(annotations) == manager.total_samples == EXPECTED_SAMPLES):
        raise ValueError("Annotation/feature count mismatch")
    features = manager.load_all_to_ram(["img_features", "txt_features"])
    sample_ids = np.asarray(manager.get_all_sample_ids(), dtype=np.int64)
    img_features = features["img_features"].numpy()
    txt_features = features["txt_features"].numpy()
    if not (len(sample_ids) == len(img_features) == len(txt_features) == EXPECTED_SAMPLES):
        raise ValueError("Feature rows and sample IDs do not match metadata")
    if (sample_ids.min() < 0 or sample_ids.max() >= len(annotations)
            or len(np.unique(sample_ids)) != len(sample_ids)):
        raise ValueError("Sample IDs must uniquely index the annotation list")
    emotions = np.asarray([annotations[int(index)]["emotion"] for index in sample_ids])
    print(f"Loaded {len(sample_ids):,} rows, {len(np.unique(emotions))} emotions", flush=True)
    return img_features, txt_features


def prepare_real_codes(img_features: np.ndarray, txt_features: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Reproduce Task 6's graph, Stage 1, communities, and raw-factor recipe."""
    graph = build_content_graph(img_features, txt_features, GraphConfig())
    print(f"Graph: {graph.nnz // 2:,} edges", flush=True)
    with redirect_stdout(io.StringIO()):
        _, embeddings = train_stage1(img_features, txt_features, graph, Stage1Config())
    if not np.isfinite(embeddings).all():
        raise ValueError("Stage 1 produced non-finite embeddings")
    labels = detect_communities(embeddings)
    communities = community_stats(labels)["num_communities"]
    print(f"Communities: {communities}", flush=True)

    config = FactorTrainingConfig(lambda_usage_balance=0.1)
    with redirect_stdout(io.StringIO()):
        _, img_codes, txt_codes = train_factors(img_features, txt_features, graph, config)
    if (img_codes.shape != (EXPECTED_SAMPLES, 32)
            or txt_codes.shape != img_codes.shape
            or not np.isfinite(img_codes).all()
            or not np.isfinite(txt_codes).all()):
        raise ValueError("Factor training produced invalid codes")
    print(f"Factor codes: {img_codes.shape}; raw features; usage balance {config.lambda_usage_balance}", flush=True)
    return img_codes, txt_codes, {"edges": int(graph.nnz // 2), "communities": communities, "factor_config": vars(config)}


def mine_and_measure(img_codes: np.ndarray, txt_codes: np.ndarray) -> tuple[list, dict]:
    """Mine episodes and measure factor eligibility, role shortfalls, and values."""
    config = EpisodeMiningConfig(seed=SEED)
    pair_codes = 0.5 * (img_codes + txt_codes)
    high_cutoffs = np.percentile(pair_codes, config.high_activation_percentile, axis=0)
    low_cutoffs = np.percentile(pair_codes, config.low_activation_percentile, axis=0)
    eligible = []
    pool_sizes = []
    for factor in range(pair_codes.shape[1]):
        column = pair_codes[:, factor]
        high_size = int(np.count_nonzero((column >= high_cutoffs[factor]) & (column > low_cutoffs[factor])))
        low_size = int(np.count_nonzero(column <= low_cutoffs[factor]))
        pool_sizes.append({"factor": factor, "high": high_size, "low": low_size})
        if high_size >= max(config.min_pool_size, 2) and low_size >= config.min_pool_size:
            eligible.append(factor)
    excluded = sorted(set(range(pair_codes.shape[1])) - set(eligible))
    episodes = mine_episodes(img_codes, txt_codes, config, NUM_EPISODES)
    chosen = sorted({episode.targeted_factor for episode in episodes})
    if not set(chosen).issubset(eligible):
        raise AssertionError("Mining selected an ineligible factor")

    requested = {
        "support": config.num_support,
        "contrast": config.num_contrast,
        "hard_negative": config.num_hard_negatives,
        "condition_distractor": config.num_condition_distractors,
        "anchor_distractor": config.num_anchor_distractors,
    }
    role_attrs = {
        "support": "support_idxs",
        "contrast": "contrast_idxs",
        "hard_negative": "hard_negative_idxs",
        "condition_distractor": "condition_distractor_idxs",
        "anchor_distractor": "anchor_distractor_idxs",
    }
    shortfalls = {role: [] for role in requested}
    for episode in episodes:
        roles = [[episode.anchor_idx], episode.support_idxs, episode.contrast_idxs,
                 [episode.positive_idx], episode.hard_negative_idxs,
                 episode.condition_distractor_idxs, episode.anchor_distractor_idxs]
        flattened = [index for role in roles for index in role]
        if len(flattened) != len(set(flattened)):
            raise AssertionError("Episode roles overlap")
        for role, attr in role_attrs.items():
            shortfalls[role].append(requested[role] - len(getattr(episode, attr)))
    if any(value < 0 for values in shortfalls.values() for value in values):
        raise AssertionError("Episode role exceeds requested count")
    all_shortfalls = np.asarray([value for values in shortfalls.values() for value in values])
    rng = np.random.default_rng(SEED)
    spot_indices = rng.choice(len(episodes), size=10, replace=False).tolist()
    spot_checks = []
    for episode_index in spot_indices:
        episode = episodes[episode_index]
        factor = episode.targeted_factor
        indices = {
            "anchor": episode.anchor_idx,
            "support": episode.support_idxs[0],
            "contrast": episode.contrast_idxs[0],
            "positive": episode.positive_idx,
            "hard_negative": episode.hard_negative_idxs[0],
            "condition_distractor": episode.condition_distractor_idxs[0],
            "anchor_distractor": episode.anchor_distractor_idxs[0],
        }
        values = {role: float(pair_codes[index, factor]) for role, index in indices.items()}
        for role in ("anchor", "support", "positive", "condition_distractor"):
            if values[role] <= low_cutoffs[factor] or values[role] < high_cutoffs[factor]:
                raise AssertionError(f"{role} was not in the high pool")
        for role in ("contrast", "hard_negative", "anchor_distractor"):
            if values[role] > low_cutoffs[factor]:
                raise AssertionError(f"{role} was not in the low pool")
        spot_checks.append({"episode": episode_index, "factor": factor, "values": values})

    result = {
        "episode_count": len(episodes),
        "mining_config": vars(config),
        "eligible_factors": eligible,
        "excluded_factors": excluded,
        "chosen_factors": chosen,
        "pool_sizes": pool_sizes,
        "shortfall": {
            "role_slots": int(all_shortfalls.size),
            "shortened_role_slots": int(np.count_nonzero(all_shortfalls)),
            "rate": float(np.mean(all_shortfalls > 0)),
            "mean_missing_per_role_slot": float(all_shortfalls.mean()),
            "mean_missing_when_shortened": float(all_shortfalls[all_shortfalls > 0].mean()) if np.any(all_shortfalls > 0) else 0.0,
            "distribution": {str(value): int(count) for value, count in zip(*np.unique(all_shortfalls, return_counts=True))},
            "by_role": {role: {"shortened_episodes": int(np.count_nonzero(values)),
                               "total_missing": int(np.sum(values))} for role, values in shortfalls.items()},
        },
        "spot_checks": spot_checks,
    }
    print(f"Episodes: {len(episodes)}; chosen factors: {len(chosen)}; excluded: {excluded}; "
          f"shortened roles: {result['shortfall']['shortened_role_slots']}/{all_shortfalls.size}", flush=True)
    return episodes, result


def recover_condition(img_codes: np.ndarray, txt_codes: np.ndarray, episodes: list) -> dict:
    """Train only the Task 1 validation objective, then evaluate held-out episodes."""
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
    per_factor = []
    for factor in sorted(np.unique(truth).tolist()):
        mask = truth == factor
        correct = int(np.count_nonzero(predictions[mask] == factor))
        per_factor.append({"factor": factor, "correct": correct,
                           "count": int(mask.sum()), "accuracy": float(correct / mask.sum())})
    accuracy = float(np.mean(predictions == truth))
    print(f"Held-out recovery: {accuracy:.4%} ({np.count_nonzero(predictions == truth)}/{len(truth)}); "
          f"chance={1 / img_codes.shape[1]:.4%}", flush=True)
    return {
        "seed": SEED,
        "split": "seed-42 random permutation of episodes; first floor(0.8*N) train, remainder held out",
        "train_episodes": int(len(train_indices)),
        "held_out_episodes": int(len(held_indices)),
        "training": {"epochs": RECOVERY_EPOCHS, "batch_size": RECOVERY_BATCH_SIZE,
                     "adam_lr": RECOVERY_LR, "objective": "cross_entropy(w(c), targeted_factor), validation only"},
        "held_out_correct": int(np.count_nonzero(predictions == truth)),
        "held_out_accuracy": accuracy,
        "chance_accuracy": float(1 / img_codes.shape[1]),
        "per_factor": per_factor,
    }


def main() -> None:
    started = perf_counter()
    img_features, txt_features = load_real_features()
    img_codes, txt_codes, prerequisite = prepare_real_codes(img_features, txt_features)
    del img_features, txt_features
    episodes, mining = mine_and_measure(img_codes, txt_codes)
    recovery = recover_condition(img_codes, txt_codes, episodes)
    print("RESULT_JSON=" + json.dumps({"samples": len(img_codes), "prerequisite": prerequisite,
                                        "mining": mining, "recovery": recovery,
                                        "seconds": perf_counter() - started}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
