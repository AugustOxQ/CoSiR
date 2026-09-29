"""Pass/fail geometry gates for a learned shared factor space.

The factor-discovery plan's gates (top-2 mass, dead, modality-private,
community spanning) measure usage; a dictionary of 32 copies of one axis
passes all of them. These gates add dimensionality, redundancy, linear
information, sparsity and pair-specificity checks. Dead / private / spanning
definitions are copied from
src/test/20260928_factor_discovery_validation/run_validation.py.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FactorGateThresholds:
    min_participation_ratio: float = 8.0      # per modality, eval rows; collapsed codes: 1.32
    max_pair_abs_corr: float = 0.90           # eval pair codes; collapsed: 374/496 pairs >= .9
    readout_pca_rank: int = 10                # code readout must beat CLIP PCA-10 per modality
    readout_ridge: float = 1e-3
    readout_fit_rows: int = 100_000
    max_active_fraction: float = 0.375        # <= 12 of 32 active per row; collapsed: ~0.71
    near_zero_mean: float = 1e-4              # Task 3 definition
    private_min_over_max: float = 0.05        # Task 3 definition
    single_community_mass: float = 0.50       # Task 3 definition
    max_dead: int = 0
    max_private: int = 1
    max_top2_mass_share: float = 0.20
    min_spanning_fraction: float = 0.75
    retrieval_pool: int = 1000
    retrieval_k: int = 10
    min_retrieval_ratio: float = 0.5          # code R@k / CLIP R@k, same pools
    seed: int = 42


def relative_l2(target: np.ndarray, reconstruction: np.ndarray) -> float:
    """Mean per-row ||target - reconstruction|| / ||target|| (Task 3/6 metric)."""
    target = np.asarray(target, dtype=np.float64)
    error = np.linalg.norm(target - reconstruction, axis=1)
    return float((error / np.maximum(np.linalg.norm(target, axis=1), 1e-12)).mean())


def participation_ratio(codes: np.ndarray) -> float:
    """(sum lambda)^2 / sum lambda^2 of the centered covariance; 0.0 if no variance."""
    codes = np.asarray(codes, dtype=np.float64)
    energy = np.linalg.svd(codes - codes.mean(axis=0), compute_uv=False) ** 2
    total = energy.sum()
    if total <= 0:
        return 0.0
    share = energy / total
    return float(1.0 / (share**2).sum())


def factor_correlation_summary(codes: np.ndarray, threshold: float = 0.9,
                               near_zero_std: float = 1e-8) -> dict:
    """Max off-diagonal |Pearson r| over varying factors; constant factors are listed."""
    codes = np.asarray(codes, dtype=np.float64)
    std = codes.std(axis=0)
    varying = np.flatnonzero(std > near_zero_std)
    constant = np.flatnonzero(std <= near_zero_std).tolist()
    if len(varying) < 2:
        return {"max_abs": 0.0, "pairs_at_or_above": 0, "pairs_total": 0,
                "constant_factors": constant}
    upper = np.abs(np.corrcoef(codes[:, varying].T)[np.triu_indices(len(varying), 1)])
    return {"max_abs": float(upper.max()), "pairs_at_or_above": int((upper >= threshold).sum()),
            "pairs_total": int(upper.size), "constant_factors": constant}


def active_fraction(codes: np.ndarray) -> float:
    return float((np.asarray(codes) > 0).mean())


def top2_mass_share(img_codes: np.ndarray, txt_codes: np.ndarray) -> float:
    mean = 0.5 * (np.asarray(img_codes, np.float64).mean(0) + np.asarray(txt_codes, np.float64).mean(0))
    total = mean.sum()
    return 1.0 if total <= 0 else float(np.sort(mean)[-2:].sum() / total)


def dead_and_private(img_codes, txt_codes, near_zero_mean, private_min_over_max):
    img_mean = np.asarray(img_codes).mean(axis=0, dtype=np.float64)
    txt_mean = np.asarray(txt_codes).mean(axis=0, dtype=np.float64)
    maximum, minimum = np.maximum(img_mean, txt_mean), np.minimum(img_mean, txt_mean)
    dead = maximum < near_zero_mean
    private = ~dead & (minimum / np.maximum(maximum, 1e-12) <= private_min_over_max)
    return np.flatnonzero(dead), np.flatnonzero(private)


def community_spanning(img_codes, txt_codes, labels, near_zero_mean, single_community_mass) -> dict:
    combined = 0.5 * (np.asarray(img_codes, np.float64) + np.asarray(txt_codes, np.float64))
    labels = np.asarray(labels, dtype=np.int64)
    n_communities = int(labels.max()) + 1
    largest = np.zeros(combined.shape[1])
    for factor in range(combined.shape[1]):
        mass = np.bincount(labels, weights=combined[:, factor], minlength=n_communities)
        if mass.sum() > 0:
            largest[factor] = mass.max() / mass.sum()
    dead_idx, _ = dead_and_private(img_codes, txt_codes, near_zero_mean, 0.0)
    dead = np.zeros(combined.shape[1], dtype=bool)
    dead[dead_idx] = True
    topic_like = ~dead & (largest >= single_community_mass)
    spanning = ~dead & ~topic_like
    return {"spanning_indices": np.flatnonzero(spanning).tolist(),
            "topic_like_indices": np.flatnonzero(topic_like).tolist(),
            "spanning_fraction": float(spanning.sum() / combined.shape[1])}


def linear_readout_rel_l2(fit_codes, fit_features, eval_codes, eval_features, ridge=1e-3) -> float:
    """Best affine map codes -> features (scale-invariant ridge), scored on eval rows."""
    x = np.asarray(fit_codes, np.float64)
    y = np.asarray(fit_features, np.float64)
    x_mean, y_mean = x.mean(0), y.mean(0)
    xc = x - x_mean
    gram = xc.T @ xc
    trace = np.trace(gram)
    if trace <= 0:
        prediction = np.broadcast_to(y_mean, np.shape(eval_features))
    else:
        penalty = ridge * trace / gram.shape[0] * np.eye(gram.shape[0])
        weights = np.linalg.solve(gram + penalty, xc.T @ (y - y_mean))
        prediction = (np.asarray(eval_codes, np.float64) - x_mean) @ weights + y_mean
    return relative_l2(eval_features, prediction)


def pca_rel_l2(fit_features, eval_features, rank: int) -> float:
    fit = np.asarray(fit_features, np.float64)
    mean = fit.mean(0)
    basis = np.linalg.svd(fit - mean, full_matrices=False)[2][:rank]
    centered = np.asarray(eval_features, np.float64) - mean
    return relative_l2(eval_features, centered @ basis.T @ basis + mean)


def _unit_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)


def paired_retrieval_recall(query, gallery, pool=1000, k=10, seed=42) -> float:
    """Row i of query must retrieve row i of gallery among `pool` rows; ties count against."""
    n = len(query)
    n_pools = n // pool
    if n_pools == 0 or len(gallery) != n:
        raise ValueError("Need matching query/gallery with at least one full pool")
    order = np.random.default_rng(seed).permutation(n)
    q, g = _unit_rows(query), _unit_rows(gallery)
    hits = 0
    for index in range(n_pools):
        rows = order[index * pool:(index + 1) * pool]
        scores = q[rows] @ g[rows].T
        rank = (scores >= np.diag(scores)[:, None]).sum(axis=1)
        hits += int((rank <= k).sum())
    return hits / (n_pools * pool)


@dataclass
class FactorGateReport:
    values: dict
    passed: dict

    @property
    def all_passed(self) -> bool:
        return all(self.passed.values())


def evaluate_factor_gates(*, fit_img_codes, fit_txt_codes, fit_img_features, fit_txt_features,
                          eval_img_codes, eval_txt_codes, eval_img_features, eval_txt_features,
                          community_img_codes, community_txt_codes, community_labels,
                          thresholds: FactorGateThresholds = FactorGateThresholds()) -> FactorGateReport:
    t = thresholds
    fit_rows = np.arange(len(fit_img_codes))
    if len(fit_rows) > t.readout_fit_rows:
        fit_rows = np.sort(np.random.default_rng(t.seed).choice(fit_rows, t.readout_fit_rows, replace=False))
    values: dict = {}
    for name, fit_codes, fit_feat, eval_codes, eval_feat in (
        ("img", fit_img_codes, fit_img_features, eval_img_codes, eval_img_features),
        ("txt", fit_txt_codes, fit_txt_features, eval_txt_codes, eval_txt_features),
    ):
        values[f"participation_ratio_{name}"] = participation_ratio(eval_codes)
        values[f"active_fraction_{name}"] = active_fraction(eval_codes)
        values[f"readout_{name}"] = linear_readout_rel_l2(
            np.asarray(fit_codes)[fit_rows], np.asarray(fit_feat)[fit_rows], eval_codes, eval_feat, t.readout_ridge)
        values[f"pca{t.readout_pca_rank}_{name}"] = pca_rel_l2(
            np.asarray(fit_feat)[fit_rows], eval_feat, t.readout_pca_rank)
    pair = 0.5 * (np.asarray(eval_img_codes, np.float64) + np.asarray(eval_txt_codes, np.float64))
    values["correlation"] = factor_correlation_summary(pair)
    dead, private = dead_and_private(eval_img_codes, eval_txt_codes, t.near_zero_mean, t.private_min_over_max)
    values["dead_indices"], values["private_indices"] = dead.tolist(), private.tolist()
    values["top2_mass_share"] = top2_mass_share(eval_img_codes, eval_txt_codes)
    values["community"] = community_spanning(community_img_codes, community_txt_codes, community_labels,
                                             t.near_zero_mean, t.single_community_mass)
    code_recall = 0.5 * (paired_retrieval_recall(eval_img_codes, eval_txt_codes, t.retrieval_pool, t.retrieval_k, t.seed)
                         + paired_retrieval_recall(eval_txt_codes, eval_img_codes, t.retrieval_pool, t.retrieval_k, t.seed))
    clip_recall = 0.5 * (paired_retrieval_recall(eval_img_features, eval_txt_features, t.retrieval_pool, t.retrieval_k, t.seed)
                         + paired_retrieval_recall(eval_txt_features, eval_img_features, t.retrieval_pool, t.retrieval_k, t.seed))
    if clip_recall <= 0:
        raise ValueError("CLIP pair retrieval is zero; the retrieval gate is undefined")
    values["code_retrieval_recall"], values["clip_retrieval_recall"] = code_recall, clip_recall
    values["retrieval_ratio"] = code_recall / clip_recall
    rank = t.readout_pca_rank
    passed = {
        "participation_ratio": bool(min(values["participation_ratio_img"], values["participation_ratio_txt"]) >= t.min_participation_ratio),
        "redundancy": bool(values["correlation"]["max_abs"] <= t.max_pair_abs_corr and not values["correlation"]["constant_factors"]),
        "readout": bool(values["readout_img"] <= values[f"pca{rank}_img"] and values["readout_txt"] <= values[f"pca{rank}_txt"]),
        "sparsity": bool(max(values["active_fraction_img"], values["active_fraction_txt"]) <= t.max_active_fraction),
        "dead": len(dead) <= t.max_dead,
        "modality_private": len(private) <= t.max_private,
        "usage_concentration": values["top2_mass_share"] <= t.max_top2_mass_share,
        "community_spanning": values["community"]["spanning_fraction"] >= t.min_spanning_fraction,
        "pair_retrieval": values["retrieval_ratio"] >= t.min_retrieval_ratio,
    }
    return FactorGateReport(values=values, passed=passed)
