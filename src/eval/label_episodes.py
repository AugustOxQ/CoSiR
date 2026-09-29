"""Episodes whose condition is a human label (emotion, art style), not a factor.

Label-defined episodes are identical for every factor model, so two models can
be compared on the very same episodes (paired), and they do not reward a model
for agreeing with its own mined structure.
"""

from dataclasses import dataclass

import numpy as np
import torch

from src.model.conditioning import conditional_score, naive_condition_weights, pair_codes


@dataclass(frozen=True)
class LabelEpisodes:
    anchor: np.ndarray
    positive: np.ndarray
    supports: np.ndarray
    contrasts: np.ndarray
    distractors: np.ndarray
    labels: np.ndarray


def _draw_distinct(rng, pool, paintings, used, count):
    """Draw `count` rows from `pool` whose paintings are not yet in `used` (updates `used`)."""
    if count > 0 and len(pool) == 0:
        raise ValueError("Not enough distinct paintings to fill an episode")
    picked = []
    for _ in range(50 * count):
        if len(picked) == count:
            break
        row = int(pool[rng.integers(len(pool))])
        if paintings[row] not in used:
            used.add(paintings[row])
            picked.append(row)
    if len(picked) < count:
        eligible = pool[~np.isin(paintings[pool], list(used))]
        _, first = np.unique(paintings[eligible], return_index=True)    # one row per painting
        eligible = eligible[np.sort(first)]
        if len(eligible) < count - len(picked):
            raise ValueError("Not enough distinct paintings to fill an episode")
        for row in rng.choice(eligible, count - len(picked), replace=False):
            used.add(paintings[row])
            picked.append(int(row))
    return picked


def build_label_episodes(labels, paintings, rows, n_episodes, seed=42, num_support=4, num_contrast=4,
                         num_distractors=12, min_paintings_per_label=30,
                         exclude_target_labels=(),
                         exclude_target_paintings_from_negatives: bool = False) -> LabelEpisodes:
    """Label-defined retrieval episodes.

    `exclude_target_paintings_from_negatives`: for per-annotation labels (emotion) an other-label row
    often depicts a painting that another annotator gave the target label, which makes it an ambiguous
    negative. When True, contrasts and distractors come only from rows whose painting carries the
    target label in NO row of the full `labels` array (not just `rows`).
    """
    labels, paintings = np.asarray(labels), np.asarray(paintings)
    rows = np.asarray(rows, dtype=np.int64)
    rng = np.random.default_rng(seed)
    row_labels = labels[rows]
    members = {label: rows[row_labels == label] for label in np.unique(row_labels)}
    others = {label: rows[row_labels != label] for label in members}
    missing = sorted({str(x) for x in exclude_target_labels if x not in members})
    if missing:
        raise ValueError(f"exclude_target_labels not present in labels[rows]: {missing}")
    excluded = set(exclude_target_labels)
    eligible = sorted(label for label, m in members.items()
                      if label not in excluded
                      and len(np.unique(paintings[m])) >= min_paintings_per_label)
    if len(eligible) < 2:
        raise ValueError("Need at least two eligible target labels")
    negatives = others
    if exclude_target_paintings_from_negatives:
        negatives = {}
        for label in eligible:                              # once per label, not per episode
            target_paintings = np.unique(paintings[labels == label])
            negatives[label] = others[label][~np.isin(paintings[others[label]], target_paintings)]
    fields = {k: [] for k in ("anchor", "positive", "supports", "contrasts", "distractors", "labels")}
    for _ in range(n_episodes):
        label = eligible[rng.integers(len(eligible))]       # label-balanced targets
        used: set = set()
        same = _draw_distinct(rng, members[label], paintings, used, 2 + num_support)
        other = _draw_distinct(rng, negatives[label], paintings, used, num_contrast + num_distractors)
        fields["anchor"].append(same[0])
        fields["positive"].append(same[1])
        fields["supports"].append(same[2:])
        fields["contrasts"].append(other[:num_contrast])
        fields["distractors"].append(other[num_contrast:])
        fields["labels"].append(label)
    return LabelEpisodes(**{k: np.asarray(v, dtype=np.int64 if k != "labels" else None)
                            for k, v in fields.items()})


def tie_aware_rank(scores: torch.Tensor) -> torch.Tensor:
    positive, others = scores[:, :1], scores[:, 1:]
    rank = 1.0 + (others > positive).sum(dim=1) + 0.5 * (others == positive).sum(dim=1)
    non_finite = ~torch.isfinite(scores).all(dim=1)         # NaN/inf never counts as a success
    return torch.where(non_finite, torch.full_like(rank, float(scores.shape[1])), rank)


def _t(values) -> torch.Tensor:
    return torch.as_tensor(np.asarray(values), dtype=torch.float32)


def label_episode_recall(img_feat, txt_feat, img_codes, txt_codes, episodes: LabelEpisodes,
                         weights: torch.Tensor, beta: float) -> dict:
    candidates = np.concatenate([episodes.positive[:, None], episodes.distractors], axis=1)
    out = {}
    for direction, qf, cf, qc, cc in (("i2t", img_feat, txt_feat, img_codes, txt_codes),
                                      ("t2i", txt_feat, img_feat, txt_codes, img_codes)):
        scores = conditional_score(_t(qf[episodes.anchor]), _t(cf[candidates]),
                                   _t(qc[episodes.anchor]), _t(cc[candidates]), weights, beta)
        ranks = tie_aware_rank(scores)
        out[direction] = {
            "ranks": ranks.numpy(), "recall1": float((ranks <= 1).float().mean()),
            "recall3": float((ranks <= 3).float().mean()),
            "tied_episodes": int((scores[:, 1:] == scores[:, :1]).any(dim=1).sum()),
        }
    return out


def condition_lift(img_feat, txt_feat, img_codes, txt_codes, episodes: LabelEpisodes) -> dict:
    """Naive minus uniform R@1 at beta=0: the condition-specific benefit of the factor space."""
    support = pair_codes(_t(img_codes[episodes.supports]), _t(txt_codes[episodes.supports]))
    contrast = pair_codes(_t(img_codes[episodes.contrasts]), _t(txt_codes[episodes.contrasts]))
    naive = naive_condition_weights(support, contrast)
    uniform = torch.full_like(naive, 1.0 / naive.shape[-1])
    result = {name: label_episode_recall(img_feat, txt_feat, img_codes, txt_codes, episodes, w, 0.0)
              for name, w in (("naive", naive), ("uniform", uniform))}
    lift = {d: result["naive"][d]["recall1"] - result["uniform"][d]["recall1"] for d in ("i2t", "t2i")}
    return {**result, "lift": lift, "lift_mean": 0.5 * (lift["i2t"] + lift["t2i"])}
