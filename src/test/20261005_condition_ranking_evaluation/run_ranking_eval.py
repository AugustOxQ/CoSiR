"""Item-disjoint, read-only conditional ranking evaluation on ArtELingo.

Run with the CoSiR environment from the repository root. The only trained
component here is ConditionEncoder's earlier validation-only recovery head;
the stage-(d) retrieval score is evaluated without fitting it.
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
from torch.utils.data import WeightedRandomSampler
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "20261001_condition_interface_validation"))

from run_validation import EXPECTED_SAMPLES, SEED, load_real_features  # noqa: E402
from src.model.communities import community_stats, detect_communities  # noqa: E402
from src.model.condition import ConditionEncoder  # noqa: E402
from src.model.graph import GraphConfig, build_content_graph  # noqa: E402
from src.train.episodes import EpisodeMiningConfig, mine_episodes  # noqa: E402
from src.train.stage1 import Stage1Config, train_stage1  # noqa: E402
from src.train.train_factors import FactorTrainingConfig, train_factors  # noqa: E402

TRAIN_EPISODES = 4096
EVAL_EPISODES = 1024
EPOCHS = 300
BATCH_SIZE = 64
LR = 0.01
BETAS = (0.0, 0.03, 0.3)
SWAP_SAMPLE = 256
VARIANTS = ("trained", "naive", "uniform", "shuffled_trained", "oracle")


def split_items(n_items: int, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Split unique row positions once; all episode roles inherit this split."""
    perm = np.random.default_rng(seed).permutation(n_items)
    train_count = int(0.8 * n_items)
    train, held = perm[:train_count], perm[train_count:]
    assert len(train) + len(held) == n_items and not np.intersect1d(train, held).size
    return train, held


def prepare_item_disjoint_codes(img_features: np.ndarray, txt_features: np.ndarray,
                                train_items: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit the validated raw-factor recipe on training items, then encode all.

    The graph, Stage 1 diagnostic, and factor optimizer never receive held-out
    items. The returned frozen factor model is used only to encode held-out
    features, which is essential for a strict item-disjoint evaluation.
    """
    train_img, train_txt = img_features[train_items], txt_features[train_items]
    graph = build_content_graph(train_img, train_txt, GraphConfig())
    print(f"Train-only graph: {graph.nnz // 2:,} edges", flush=True)
    with redirect_stdout(io.StringIO()):
        _, embeddings = train_stage1(train_img, train_txt, graph, Stage1Config())
    if not np.isfinite(embeddings).all():
        raise ValueError("Stage 1 produced non-finite embeddings")
    communities = community_stats(detect_communities(embeddings))["num_communities"]
    print(f"Train-only communities: {communities}", flush=True)
    factor_config = FactorTrainingConfig(lambda_usage_balance=0.1)
    with redirect_stdout(io.StringIO()):
        model, train_img_codes, train_txt_codes = train_factors(
            train_img, train_txt, graph, factor_config
        )
    img_codes = np.empty((len(img_features), factor_config.num_factors), dtype=np.float32)
    txt_codes = np.empty_like(img_codes)
    img_codes[train_items] = train_img_codes
    txt_codes[train_items] = train_txt_codes
    del train_img_codes, train_txt_codes
    held_items = np.setdiff1d(np.arange(len(img_features)), train_items, assume_unique=True)
    device = next(model.parameters()).device
    with torch.no_grad():
        for start in range(0, len(held_items), 8192):
            ids = held_items[start:start + 8192]
            img = torch.as_tensor(img_features[ids], dtype=torch.float32, device=device)
            txt = torch.as_tensor(txt_features[ids], dtype=torch.float32, device=device)
            img_codes[ids] = model.encode_image(img).cpu().numpy()
            txt_codes[ids] = model.encode_text(txt).cpu().numpy()
    if (img_codes.shape != (EXPECTED_SAMPLES, 32) or txt_codes.shape != img_codes.shape
            or not np.isfinite(img_codes).all() or not np.isfinite(txt_codes).all()):
        raise ValueError("Factor preparation produced invalid codes")
    print(f"Train-only factor fit; encoded {len(held_items)} unseen items", flush=True)
    return img_codes, txt_codes, {"edges": int(graph.nnz // 2),
                                  "communities": communities, "factor_config": vars(factor_config),
                                  "factor_fit_items": len(train_items)}


def episode_arrays(episodes: list) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return fixed-size support, contrast, candidate, and target arrays."""
    supports = np.asarray([ep.support_idxs for ep in episodes], dtype=np.int64)
    contrasts = np.asarray([ep.contrast_idxs for ep in episodes], dtype=np.int64)
    candidates = np.asarray([
        [ep.positive_idx, *ep.hard_negative_idxs, *ep.condition_distractor_idxs,
         *ep.anchor_distractor_idxs] for ep in episodes
    ], dtype=np.int64)
    targets = np.asarray([ep.targeted_factor for ep in episodes], dtype=np.int64)
    if (supports.ndim != 2 or contrasts.ndim != 2 or candidates.ndim != 2
            or supports.shape[1] != 4 or contrasts.shape[1] != 4 or candidates.shape[1] != 13):
        raise ValueError("Default mining roles must all be present; no candidate may be padded")
    return supports, contrasts, candidates, targets


def validate_roles(episodes: list, allowed_items: np.ndarray) -> None:
    """Catch any miner indexing or role-overlap mistake before measurement."""
    allowed = set(map(int, allowed_items))
    for ep in episodes:
        roles = [ep.anchor_idx, *ep.support_idxs, *ep.contrast_idxs, ep.positive_idx,
                 *ep.hard_negative_idxs, *ep.condition_distractor_idxs, *ep.anchor_distractor_idxs]
        if len(roles) != len(set(roles)) or not set(roles).issubset(allowed):
            raise AssertionError("Episode roles overlap or cross the item split")


def train_recovery_head(img_codes: np.ndarray, txt_codes: np.ndarray, episodes: list) -> ConditionEncoder:
    """Reuse Task 5's 4096-episode, balanced, validation-only CE recipe."""
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    supports, contrasts, _, targets_np = episode_arrays(episodes)
    support_img = torch.as_tensor(img_codes[supports], dtype=torch.float32)
    support_txt = torch.as_tensor(txt_codes[supports], dtype=torch.float32)
    contrast_img = torch.as_tensor(img_codes[contrasts], dtype=torch.float32)
    contrast_txt = torch.as_tensor(txt_codes[contrasts], dtype=torch.float32)
    targets = torch.as_tensor(targets_np, dtype=torch.long)
    counts = np.bincount(targets_np, minlength=img_codes.shape[1])
    if np.any(counts == 0):
        raise ValueError("Balanced recovery needs training episodes for all factors")
    sampler = WeightedRandomSampler(
        torch.as_tensor(1.0 / counts[targets_np], dtype=torch.double),
        num_samples=len(episodes), replacement=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    model = ConditionEncoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    def weights(indices: np.ndarray) -> torch.Tensor:
        return torch.vmap(model)(support_img[indices], support_txt[indices],
                                 contrast_img[indices], contrast_txt[indices])

    with torch.no_grad():
        direct = model(support_img[0], support_txt[0], contrast_img[0], contrast_txt[0])
        if not torch.allclose(weights(np.asarray([0]))[0], direct, atol=1e-6):
            raise AssertionError("Vectorized ConditionEncoder disagrees with direct forward")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        sampled = np.fromiter(sampler, dtype=np.int64)
        for start in range(0, len(sampled), BATCH_SIZE):
            indices = sampled[start:start + BATCH_SIZE]
            loss = F.cross_entropy(weights(indices), targets[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if epoch % 50 == 0:
            print(f"Recovery epoch {epoch}/{EPOCHS}: loss={loss.item():.6f}", flush=True)
    return model.eval()


def make_weights(model: ConditionEncoder, img_codes: np.ndarray, txt_codes: np.ndarray,
                 episodes: list) -> tuple[dict[str, np.ndarray], np.ndarray, dict]:
    """Build all five literal w(c) variants and gap-rank diagnostics."""
    supports, contrasts, _, targets = episode_arrays(episodes)
    pair_codes = 0.5 * (img_codes + txt_codes)
    gap = pair_codes[supports].mean(axis=1) - pair_codes[contrasts].mean(axis=1)
    n, n_factors = gap.shape
    naive = np.maximum(gap, 0.0)
    with torch.no_grad():
        supp_i = torch.as_tensor(img_codes[supports], dtype=torch.float32)
        supp_t = torch.as_tensor(txt_codes[supports], dtype=torch.float32)
        cont_i = torch.as_tensor(img_codes[contrasts], dtype=torch.float32)
        cont_t = torch.as_tensor(txt_codes[contrasts], dtype=torch.float32)
        trained = np.concatenate([
            torch.vmap(model)(supp_i[start:start + 256], supp_t[start:start + 256],
                              cont_i[start:start + 256], cont_t[start:start + 256]).numpy()
            for start in range(0, n, 256)
        ])
    rng = np.random.default_rng(SEED)
    # A cyclic permutation of randomized episode order guarantees no fixed points.
    order = rng.permutation(n)
    shuffled_source = np.empty(n, dtype=np.int64)
    shuffled_source[order] = np.roll(order, 1)
    assert np.all(shuffled_source != np.arange(n))
    variants = {
        "trained": trained,
        "naive": naive,
        "uniform": np.full((n, n_factors), 1.0 / n_factors, dtype=np.float32),
        "shuffled_trained": trained[shuffled_source],
        "oracle": np.eye(n_factors, dtype=np.float32)[targets],
    }
    target_gap = gap[np.arange(n), targets]
    true_rank = 1 + (gap > target_gap[:, None]).sum(axis=1)
    diagnostic = {
        "ranks": true_rank.tolist(), "median": float(np.median(true_rank)),
        "top3": float(np.mean(true_rank <= 3)), "top10": float(np.mean(true_rank <= 10)),
        "rank_counts": np.bincount(true_rank, minlength=n_factors + 1)[1:].tolist(),
        "naive_target_argmax": float(np.mean(naive.argmax(axis=1) == targets)),
        "trained_target_argmax": float(np.mean(trained.argmax(axis=1) == targets)),
        "shuffled_source": shuffled_source.tolist(),
    }
    return variants, gap, diagnostic


def unit_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)


def score_pool(anchor: np.ndarray, candidates: np.ndarray, anchor_codes: np.ndarray,
               candidate_codes: np.ndarray, weights: np.ndarray, beta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the spec's symmetric score to one direction of a candidate pool."""
    cosine = np.sum(unit_rows(anchor)[:, None, :] * unit_rows(candidates), axis=-1)
    factor = np.sum(weights[:, None, :] * anchor_codes[:, None, :] * candidate_codes, axis=-1)
    return beta * cosine + factor, cosine, factor


def evaluate_ranking(img_features: np.ndarray, txt_features: np.ndarray,
                     img_codes: np.ndarray, txt_codes: np.ndarray, episodes: list,
                     variants: dict[str, np.ndarray]) -> dict:
    """Rank the positive first-column candidate in both retrieval directions."""
    _, _, candidates, _ = episode_arrays(episodes)
    anchors = np.asarray([ep.anchor_idx for ep in episodes], dtype=np.int64)
    out = {}
    for beta in BETAS:
        per_variant = {}
        for name in VARIANTS:
            weights = variants[name]
            directions = {}
            for direction, a_feat, c_feat, a_code, c_code in (
                ("i2t", img_features[anchors], txt_features[candidates],
                 img_codes[anchors], txt_codes[candidates]),
                ("t2i", txt_features[anchors], img_features[candidates],
                 txt_codes[anchors], img_codes[candidates]),
            ):
                scores, cosine, factor = score_pool(a_feat, c_feat, a_code, c_code, weights, beta)
                ranks = 1 + (scores[:, 1:] > scores[:, :1]).sum(axis=1)
                directions[direction] = {
                    "ranks": ranks.tolist(), "recall1": float(np.mean(ranks == 1)),
                    "recall3": float(np.mean(ranks <= 3)),
                    "mean_cosine": float(cosine.mean()),
                    "mean_abs_factor": float(np.abs(factor).mean()),
                }
            per_variant[name] = directions
        out[str(beta)] = per_variant
    return out


def choose_swap_pairs(pair_codes: np.ndarray, held_items: np.ndarray, episodes: list,
                      config: EpisodeMiningConfig) -> list[tuple[int, int]]:
    """Match conditions A/B valid for A's anchor, with selective positives.

    A's anchor is high for both factors. A's positive is high for A and low
    for B; B's positive is high for B and low for A. B's support/contrast is
    disjoint from A's anchor. The two condition-specific positives thus have
    an operationally testable preference under a fixed anchor/candidate union.
    """
    high = np.percentile(pair_codes[held_items], config.high_activation_percentile, axis=0)
    low = np.percentile(pair_codes[held_items], config.low_activation_percentile, axis=0)
    targets = np.asarray([ep.targeted_factor for ep in episodes])
    anchors = np.asarray([ep.anchor_idx for ep in episodes])
    positives = np.asarray([ep.positive_idx for ep in episodes])
    rng = np.random.default_rng(SEED)
    pairs = []
    for a in rng.permutation(len(episodes)):
        if len(pairs) >= SWAP_SAMPLE:
            break
        factor_a = targets[a]
        for b in rng.permutation(len(episodes)):
            factor_b = targets[b]
            if (a == b or factor_a == factor_b or positives[a] == positives[b]
                    or anchors[a] in [*episodes[b].support_idxs, *episodes[b].contrast_idxs,
                                       positives[b]]):
                continue
            if (pair_codes[anchors[a], factor_b] >= high[factor_b]
                    and pair_codes[anchors[a], factor_b] > low[factor_b]
                    and pair_codes[positives[a], factor_b] <= low[factor_b]
                    and pair_codes[positives[b], factor_a] <= low[factor_a]):
                pairs.append((int(a), int(b)))
                break
    if not pairs:
        raise ValueError("No valid two-condition swap pairs found")
    return pairs


def evaluate_swaps(img_features: np.ndarray, txt_features: np.ndarray,
                   img_codes: np.ndarray, txt_codes: np.ndarray, episodes: list,
                   variants: dict[str, np.ndarray], pairs: list[tuple[int, int]]) -> dict:
    """Require both positive preferences to reverse on the same candidate union."""
    out = {}
    for beta in BETAS:
        per_variant = {}
        for name in VARIANTS:
            by_direction = {}
            for direction, af, cf, ac, cc in (
                ("i2t", img_features, txt_features, img_codes, txt_codes),
                ("t2i", txt_features, img_features, txt_codes, img_codes),
            ):
                count = 0
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
                                               variants[name][[a, b]], beta)
                    if not np.array_equal(np.argsort(-scores[0]), np.argsort(-scores[1])):
                        any_change += 1
                    margin_a = scores[0, 0] - scores[0, 1]
                    margin_b = scores[1, 1] - scores[1, 0]
                    count += bool(margin_a > 1e-8 and margin_b > 1e-8)
                by_direction[direction] = {"appropriate_reversals": count,
                                           "reversal_rate": count / len(pairs),
                                           "any_rank_change": any_change / len(pairs)}
            per_variant[name] = by_direction
        out[str(beta)] = per_variant
    return out


def main() -> None:
    started = perf_counter()
    img_features, txt_features = load_real_features()
    train_items, held_items = split_items(EXPECTED_SAMPLES)
    img_codes, txt_codes, prerequisite = prepare_item_disjoint_codes(
        img_features, txt_features, train_items
    )
    if img_codes.shape != (EXPECTED_SAMPLES, 32):
        raise ValueError("Expected 308,723 paired items and 32 factors")
    config = EpisodeMiningConfig(seed=SEED)
    # Mine with split-local positions; then remap every role to global row IDs.
    with threadpool_limits(limits=1, user_api="blas"):
        train_episodes = mine_episodes(img_codes[train_items], txt_codes[train_items],
                                       config, TRAIN_EPISODES)
        eval_episodes = mine_episodes(img_codes[held_items], txt_codes[held_items],
                                      config, EVAL_EPISODES)
    for episodes, items in ((train_episodes, train_items), (eval_episodes, held_items)):
        for ep in episodes:
            ep.anchor_idx = int(items[ep.anchor_idx])
            ep.support_idxs = items[ep.support_idxs].astype(int).tolist()
            ep.contrast_idxs = items[ep.contrast_idxs].astype(int).tolist()
            ep.positive_idx = int(items[ep.positive_idx])
            ep.hard_negative_idxs = items[ep.hard_negative_idxs].astype(int).tolist()
            ep.condition_distractor_idxs = items[ep.condition_distractor_idxs].astype(int).tolist()
            ep.anchor_distractor_idxs = items[ep.anchor_distractor_idxs].astype(int).tolist()
        validate_roles(episodes, items)
        episode_arrays(episodes)
    print(f"Item split: {len(train_items)} train, {len(held_items)} held; "
          f"episodes: {len(train_episodes)} train, {len(eval_episodes)} held", flush=True)
    model = train_recovery_head(img_codes, txt_codes, train_episodes)
    variants, _, gap_diagnostic = make_weights(model, img_codes, txt_codes, eval_episodes)
    ranking = evaluate_ranking(img_features, txt_features, img_codes, txt_codes,
                               eval_episodes, variants)
    pairs = choose_swap_pairs(0.5 * (img_codes + txt_codes), held_items, eval_episodes, config)
    swaps = evaluate_swaps(img_features, txt_features, img_codes, txt_codes,
                           eval_episodes, variants, pairs)
    result = {
        "samples": EXPECTED_SAMPLES, "split_seed": SEED,
        "train_items": len(train_items), "held_items": len(held_items),
        "train_episodes": TRAIN_EPISODES, "eval_episodes": EVAL_EPISODES,
        "mining_config": vars(config), "prerequisite": prerequisite,
        "training": {"epochs": EPOCHS, "batch_size": BATCH_SIZE, "adam_lr": LR,
                     "sampler": "inverse-class-frequency WeightedRandomSampler, replacement=True",
                     "objective": "cross_entropy(w(c), targeted_factor), validation only"},
        "betas": BETAS, "gap": gap_diagnostic, "ranking": ranking,
        "swap_pairs": pairs, "swap_sample": len(pairs), "swaps": swaps,
        "seconds": perf_counter() - started,
    }
    print("RESULT_JSON=" + json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
