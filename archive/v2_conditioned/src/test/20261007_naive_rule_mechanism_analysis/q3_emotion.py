"""Q3b: human-emotion episodes, independent of factor-code episode mining.

The arrays passed to :func:`analyze_emotion` are in FeatureManager storage order.
Stored sample IDs are joined to ArtELingo annotation positions before mining.
"""

import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "20261005_condition_ranking_evaluation"))

from run_ranking_eval import score_pool  # noqa: E402
from src.utils import FeatureManager  # noqa: E402


FEATURE_DIR = "/data/SSD2/pre_extract/artelingo/features"
ANNOTATIONS_PATH = Path("/data/PDD/artelingo/artelingo_train.json")
SEED = 42
TRAIN_EPISODES = 4096
HELD_EPISODES = 1024
BOOTSTRAPS = 2000
BETAS = (0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
VARIANTS = ("naive", "uniform", "clip_only")
DIRECTIONS = ("i2t", "t2i")


@dataclass
class EmotionEpisode:
    anchor_idx: int
    emotion: str
    support_idxs: list[int]
    contrast_idxs: list[int]
    positive_idx: int
    random_idxs: list[int]
    nearest_idxs: dict[str, list[int]]


def _load_labels(n_items: int) -> tuple[np.ndarray, np.ndarray]:
    """Use exactly the feature-cache sample IDs as annotation list positions."""
    manager = FeatureManager(storage_dir=FEATURE_DIR)
    with ANNOTATIONS_PATH.open() as file:
        annotations = json.load(file)
    ids = np.asarray(manager.get_all_sample_ids(), dtype=np.int64)
    if (len(ids) != n_items or manager.total_samples != n_items
            or len(annotations) != n_items or len(np.unique(ids)) != n_items
            or ids.min() < 0 or ids.max() >= len(annotations)):
        raise ValueError("FeatureManager sample IDs do not uniquely join ArtELingo annotations")
    if any("emotion" not in row or "painting" not in row for row in annotations):
        raise ValueError("ArtELingo rows need emotion and painting fields")
    emotions = np.asarray([annotations[int(i)]["emotion"] for i in ids])
    paintings = np.asarray([annotations[int(i)]["painting"] for i in ids])
    if np.any(emotions == "") or np.any(paintings == ""):
        raise ValueError("Empty emotion or painting annotation")
    return emotions, paintings


def _unit_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)


def _pick_distinct(pool: np.ndarray, painting_codes: np.ndarray, used: set[int],
                   count: int, rng: np.random.Generator) -> list[int]:
    chosen = []
    # Each ArtELingo painting has few annotations, so rejection is cheap; the
    # bounded fallback makes sparse synthetic and skewed subsets reliable.
    for _ in range(count):
        for _attempt in range(200):
            idx = int(pool[rng.integers(len(pool))])
            painting = int(painting_codes[idx])
            if painting not in used:
                break
        else:
            eligible = pool[~np.isin(painting_codes[pool], list(used))]
            if not len(eligible):
                raise ValueError("Insufficient distinct paintings for emotion episode")
            idx = int(rng.choice(eligible))
            painting = int(painting_codes[idx])
        chosen.append(idx)
        used.add(painting)
    return chosen


def _top_distinct(scores: np.ndarray, pool: np.ndarray, painting_codes: np.ndarray,
                  used: set[int], count: int = 6) -> list[int]:
    """Return exact nearest eligible annotations with one per painting."""
    k = min(len(pool), max(64, count))
    while True:
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.lexsort((pool[top], -scores[top]))]
        seen = set(used)
        picked = []
        for pos in top:
            idx = int(pool[pos])
            painting = int(painting_codes[idx])
            if painting not in seen:
                picked.append(idx)
                seen.add(painting)
                if len(picked) == count:
                    return picked
        if k == len(pool):
            raise ValueError("Insufficient distinct paintings for nearest distractors")
        k = min(len(pool), 2 * k)


def mine_emotion_episodes(img_features: np.ndarray, txt_features: np.ndarray,
                          items: np.ndarray, emotions: np.ndarray, paintings: np.ndarray,
                          count: int, seed: int = SEED) -> list[EmotionEpisode]:
    """Mine label-defined episodes; nearest pools use cross-modal CLIP cosine.

    The six random distractors are drawn before nearest selection. The same
    anchor/support/contrast/positive/random roles are used in both directions;
    each direction gets its own six nearest eligible distractors.
    """
    rng = np.random.default_rng(seed)
    items = np.asarray(items, dtype=np.int64)
    if len(np.unique(items)) != len(items):
        raise ValueError("Item split contains duplicates")
    _, painting_codes = np.unique(paintings, return_inverse=True)
    labels = sorted(set(map(str, emotions[items])))
    by_emotion = {e: items[emotions[items] == e] for e in labels}
    others = {e: items[emotions[items] != e] for e in labels}
    valid = [e for e in labels if len(np.unique(painting_codes[by_emotion[e]])) >= 6
             and len(np.unique(painting_codes[others[e]])) >= 16]
    if not valid:
        raise ValueError("No emotion has enough distinct paintings for full episodes")
    episodes = []
    for _ in range(count):
        emotion = str(rng.choice(valid))
        same_pool, other_pool = by_emotion[emotion], others[emotion]
        used = set()
        anchor = _pick_distinct(same_pool, painting_codes, used, 1, rng)[0]
        support = _pick_distinct(same_pool, painting_codes, used, 4, rng)
        positive = _pick_distinct(same_pool, painting_codes, used, 1, rng)[0]
        contrast = _pick_distinct(other_pool, painting_codes, used, 4, rng)
        random = _pick_distinct(other_pool, painting_codes, used, 6, rng)
        episodes.append(EmotionEpisode(anchor, emotion, support, contrast,
                                       positive, random, {}))

    # Matrix multiplication batches all anchors of an emotion. Both direction
    # searches rank the entire split's eligible opposite-emotion annotations.
    for direction, anchor_features, candidate_features in (
        ("i2t", img_features, txt_features), ("t2i", txt_features, img_features)
    ):
        anchor_unit = _unit_rows(anchor_features)
        candidate_unit = _unit_rows(candidate_features)
        for emotion in valid:
            pool = others[emotion]
            eligible_codes = candidate_unit[pool]
            ep_ids = [i for i, ep in enumerate(episodes) if ep.emotion == emotion]
            for start in range(0, len(ep_ids), 32):
                batch = ep_ids[start:start + 32]
                similarity = anchor_unit[[episodes[i].anchor_idx for i in batch]] @ eligible_codes.T
                for row, ep_id in enumerate(batch):
                    ep = episodes[ep_id]
                    roles = [ep.anchor_idx, *ep.support_idxs, *ep.contrast_idxs,
                             ep.positive_idx, *ep.random_idxs]
                    used = set(map(int, painting_codes[roles]))
                    ep.nearest_idxs[direction] = _top_distinct(
                        similarity[row], pool, painting_codes, used)
    for ep in episodes:
        for direction in DIRECTIONS:
            roles = [ep.anchor_idx, *ep.support_idxs, *ep.contrast_idxs,
                     ep.positive_idx, *ep.random_idxs, *ep.nearest_idxs[direction]]
            if len(roles) != 22 or len(set(map(int, painting_codes[roles]))) != 22:
                raise AssertionError("Emotion episode has wrong shape or repeated painting")
            if not all(emotions[i] == ep.emotion for i in [ep.anchor_idx, *ep.support_idxs,
                                                           ep.positive_idx]):
                raise AssertionError("Positive emotion role mismatch")
            if not all(emotions[i] != ep.emotion for i in [*ep.contrast_idxs,
                                                           *ep.random_idxs,
                                                           *ep.nearest_idxs[direction]]):
                raise AssertionError("Negative emotion role mismatch")
    return episodes


def _weights(episodes: list[EmotionEpisode], img_codes: np.ndarray,
             txt_codes: np.ndarray) -> dict[str, np.ndarray]:
    pair_codes = 0.5 * (img_codes + txt_codes)
    support = np.asarray([ep.support_idxs for ep in episodes], dtype=np.int64)
    contrast = np.asarray([ep.contrast_idxs for ep in episodes], dtype=np.int64)
    naive = np.maximum(pair_codes[support].mean(axis=1) -
                       pair_codes[contrast].mean(axis=1), 0)
    sums = naive.sum(axis=1, keepdims=True)
    naive = np.divide(naive, sums, out=np.zeros_like(naive), where=sums > 0)
    n, factors = naive.shape
    return {"naive": naive, "uniform": np.full((n, factors), 1 / factors,
                                                dtype=np.float32),
            "clip_only": np.zeros((n, factors), dtype=np.float32)}


def _components(img_features: np.ndarray, txt_features: np.ndarray,
                img_codes: np.ndarray, txt_codes: np.ndarray,
                episodes: list[EmotionEpisode]) -> dict:
    anchors = np.asarray([ep.anchor_idx for ep in episodes], dtype=np.int64)
    weights = _weights(episodes, img_codes, txt_codes)
    result = {}
    for direction, af, cf, ac, cc in (
        ("i2t", img_features, txt_features, img_codes, txt_codes),
        ("t2i", txt_features, img_features, txt_codes, img_codes),
    ):
        candidates = np.asarray([[ep.positive_idx, *ep.nearest_idxs[direction],
                                  *ep.random_idxs] for ep in episodes], dtype=np.int64)
        by_variant = {}
        for variant in VARIANTS:
            _, cosine, factor = score_pool(af[anchors], cf[candidates], ac[anchors],
                                           cc[candidates], weights[variant], 0.0)
            by_variant[variant] = (cosine, factor)
        result[direction] = by_variant
    return result


def _recall(cosine: np.ndarray, factor: np.ndarray, beta: float) -> tuple[np.ndarray, np.ndarray]:
    scores = factor + beta * cosine
    positive = scores[:, :1]
    ranks = 1 + np.sum(scores[:, 1:] > positive, axis=1)
    ranks = ranks + 0.5 * np.sum(scores[:, 1:] == positive, axis=1)
    return (ranks <= 1).astype(np.float32), (ranks <= 3).astype(np.float32)


def _score_beta(components: dict, variant: str, beta: float) -> float:
    return float(np.mean([_recall(*components[direction][variant], beta)[0].mean()
                          for direction in DIRECTIONS]))


def _select_betas(train: dict) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    grid = list(BETAS)
    scores = {v: {str(beta): _score_beta(train, v, beta) for beta in grid}
              for v in VARIANTS}
    # Zero is the physical lower boundary. Extend the positive upper boundary
    # while an additional decade still improves a variant's training score.
    for upper in (10.0, 30.0, 100.0, 300.0, 1000.0):
        improving = [v for v in VARIANTS if scores[v][str(grid[-1])] >
                     max(scores[v][str(b)] for b in grid[:-1])]
        if not improving:
            break
        grid.append(upper)
        for variant in VARIANTS:
            scores[variant][str(upper)] = _score_beta(train, variant, upper)
    selected = {}
    for variant in VARIANTS:
        eligible = [b for b in grid if b > 0] if variant == "clip_only" else grid
        selected[variant] = float(max(eligible, key=lambda b: (scores[variant][str(b)], -b)))
    return selected, scores


def _bootstrap_diff(a: np.ndarray, b: np.ndarray,
                    draw: np.ndarray) -> dict[str, float]:
    difference = a - b
    means = difference[draw].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"difference": float(difference.mean()), "ci95": [float(lo), float(hi)]}


def analyze_emotion(img_features: np.ndarray, txt_features: np.ndarray,
                    img_codes: np.ndarray, txt_codes: np.ndarray,
                    train_items: np.ndarray, held_items: np.ndarray) -> dict:
    """Return compact, JSON-serializable Q3b results at train-selected betas."""
    n = len(img_features)
    if (txt_features.shape != img_features.shape or img_codes.shape != txt_codes.shape
            or img_codes.shape[0] != n or img_codes.ndim != 2
            or not all(np.isfinite(x).all() for x in (img_features, txt_features,
                                                      img_codes, txt_codes))):
        raise ValueError("Feature and factor arrays must have aligned finite rows")
    train_items = np.asarray(train_items, dtype=np.int64)
    held_items = np.asarray(held_items, dtype=np.int64)
    if (len(np.unique(train_items)) != len(train_items)
            or len(np.unique(held_items)) != len(held_items)
            or np.intersect1d(train_items, held_items).size
            or min(train_items.min(), held_items.min()) < 0
            or max(train_items.max(), held_items.max()) >= n):
        raise ValueError("Train and held item indices must be disjoint, unique, and in range")
    emotions, paintings = _load_labels(n)
    train_paintings = set(paintings[train_items])
    held_paintings = set(paintings[held_items])
    painting_overlap = len(train_paintings & held_paintings)
    held_rows_with_train_painting = int(sum(
        painting in train_paintings for painting in paintings[held_items]
    ))
    train_episodes = mine_emotion_episodes(img_features, txt_features, train_items,
                                           emotions, paintings, TRAIN_EPISODES, SEED)
    held_episodes = mine_emotion_episodes(img_features, txt_features, held_items,
                                          emotions, paintings, HELD_EPISODES, SEED)
    train = _components(img_features, txt_features, img_codes, txt_codes, train_episodes)
    selected, train_grid = _select_betas(train)
    held = _components(img_features, txt_features, img_codes, txt_codes, held_episodes)
    recall = {}
    episode_recall = {}
    for variant in VARIANTS:
        beta = selected[variant]
        per_direction = {}
        episode_recall[variant] = {}
        for direction in DIRECTIONS:
            r1, r3 = _recall(*held[direction][variant], beta)
            episode_recall[variant][direction] = (r1, r3)
            per_direction[direction] = {"r1": float(r1.mean()), "r3": float(r3.mean())}
        recall[variant] = {"beta": beta, "directions": per_direction,
                           "mean_bidirectional_r1": float(np.mean([
                               per_direction[d]["r1"] for d in DIRECTIONS]))}
    draw = np.random.default_rng(SEED).integers(0, len(held_episodes),
                                                size=(BOOTSTRAPS, len(held_episodes)))
    ci = {}
    for baseline in ("uniform", "clip_only"):
        comparison = {}
        for direction in DIRECTIONS:
            comparison[direction] = {
                metric: _bootstrap_diff(episode_recall["naive"][direction][j],
                                        episode_recall[baseline][direction][j], draw)
                for j, metric in enumerate(("r1", "r3"))}
        comparison["bidirectional"] = {
            metric: _bootstrap_diff(
                np.mean([episode_recall["naive"][d][j] for d in DIRECTIONS], axis=0),
                np.mean([episode_recall[baseline][d][j] for d in DIRECTIONS], axis=0),
                draw) for j, metric in enumerate(("r1", "r3"))}
        ci[f"naive_minus_{baseline}"] = comparison
    counts = Counter(ep.emotion for ep in held_episodes)
    return {
        "metadata": {"seed": SEED, "train_episodes": len(train_episodes),
                     "held_episodes": len(held_episodes), "bootstrap_resamples": BOOTSTRAPS,
                     "train_items": len(train_items), "held_items": len(held_items),
                     "painting_field": "painting", "annotation_join": "FeatureManager sample IDs",
                     "unique_train_paintings": len(train_paintings),
                     "unique_held_paintings": len(held_paintings),
                     "painting_overlap_count": painting_overlap,
                     "held_rows_with_train_painting": held_rows_with_train_painting,
                     "nearest": "six exact cross-modal cosine nearest among other emotions, per direction",
                     "candidate_order": "positive, six nearest, six random",
                     "beta_zero_rank": "midrank (half of tied competitors)",
                     "clip_only_beta_selection": "positive beta only; zero makes every score tie",
                     "held_emotion_counts": dict(sorted(counts.items()))},
        "beta_train_mean_bidirectional_r1": train_grid,
        "selected_beta": selected,
        "held_recall": recall,
        "paired_bootstrap": ci,
    }
