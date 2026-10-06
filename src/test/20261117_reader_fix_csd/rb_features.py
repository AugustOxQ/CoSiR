"""Pure-array pieces of the learned reader R-b (DECISION_RULE.md §4.2), shared by rb_build.py, rb_eval.py and
test_rb.py. Numpy and scikit-learn only; no data is loaded here.

  episode_features   §4.2 item 4: six features per grouping, in configuration order
  bank_labels        §4.2 item 5: class of condition a (block's first grouping) and b (block's second grouping)
  episode_folds      §4.2 item 6: KFold(5, shuffle=True, random_state=0) over episode indices
  average_probs      §4.2 item 6: P^c(h) = mean over the two half-readers
  picks_and_margins  §4.2 item 7: arg max P (ties to the first grouping) and the top-two margin
  smd                §4.2 item 11(c): standardised mean difference per feature
"""
from itertools import combinations

import numpy as np
from sklearn.model_selection import KFold

FEATURES = ("S", "C", "Delta", "sd_support", "sd_contrast", "argmax_match")   # item 4, per grouping, this order
N_PAIRS = 4
C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)                                       # item 6
N_FOLDS, FOLD_SEED = 5, 0                                                    # item 6


def feature_names(parts):
    return [f"{h}__{f}" for h in parts for f in FEATURES]


def episode_features(post, parts, sup_img, sup_txt, con_img, con_txt):
    """(E, 6 * len(parts)) float64 features of one condition. ``post[h]['img']`` / ``post[h]['txt']`` are posterior
    arrays indexed by the same row space as the pair arrays (each (E, 4)). The agreement of a pair (i, t) is
    p_h(i) . p_h(t) (D3), computed as aspect_deltas does (float32 einsum, mean over the 4 pairs), so Delta equals
    src.eval.aspect_quick_checks.aspect_deltas bit for bit; the standard deviations (ddof 1) are taken in float64 of
    the same float32 agreements. Condition b's features are this function applied to the swapped sets."""
    sup_img, sup_txt = np.asarray(sup_img), np.asarray(sup_txt)
    con_img, con_txt = np.asarray(con_img), np.asarray(con_txt)
    if not (sup_img.shape == sup_txt.shape == con_img.shape == con_txt.shape and sup_img.shape[1] == N_PAIRS):
        raise ValueError(f"pair arrays must all be (E, {N_PAIRS})")
    cols = []
    for h in parts:
        pi, pt = post[h]["img"], post[h]["txt"]
        a_img, a_txt = pi[sup_img], pt[sup_txt]
        sup = np.einsum("nsc,nsc->ns", a_img, a_txt)
        con = np.einsum("nsc,nsc->ns", pi[con_img], pt[con_txt])
        s, c = sup.mean(axis=1), con.mean(axis=1)
        match = (a_img.argmax(axis=-1) == a_txt.argmax(axis=-1)).mean(axis=1)
        cols += [s, c, s - c, sup.astype(np.float64).std(axis=1, ddof=1),
                 con.astype(np.float64).std(axis=1, ddof=1), match]
    return np.stack([np.asarray(x, dtype=np.float64) for x in cols], axis=1)


def both_conditions(post, parts, ep):
    """{'a': features, 'b': features} of an AspectEpisodes-like object (fields pairs_a_img ... pairs_b_txt).
    Condition a: supports = pairs_a, contrasts = pairs_b; condition b: swapped (AspectEpisodes.condition)."""
    return {"a": episode_features(post, parts, ep.pairs_a_img, ep.pairs_a_txt, ep.pairs_b_img, ep.pairs_b_txt),
            "b": episode_features(post, parts, ep.pairs_b_img, ep.pairs_b_txt, ep.pairs_a_img, ep.pairs_a_txt)}


def bank_blocks(parts):
    """The builder's blocks: combinations of the sorted keys (src.train.pseudo_partitions.build_episode_bank)."""
    return list(combinations(sorted(parts), 2))


def bank_labels(block_pairs, block_size, parts):
    """(y_a, y_b): per bank episode, the class (position in configuration order ``parts``) of the grouping its
    condition-a supports share (the block's first grouping) and of condition b's (the block's second)."""
    parts = list(parts)
    ya = np.concatenate([np.full(block_size, parts.index(a), dtype=np.int64) for a, _ in block_pairs])
    yb = np.concatenate([np.full(block_size, parts.index(b), dtype=np.int64) for _, b in block_pairs])
    return ya, yb


def stack_conditions(Xa, Xb, ya, yb):
    """Training examples: rows 0..N-1 are condition a of episodes 0..N-1, rows N..2N-1 condition b; also returns
    the episode index of every row."""
    n = len(Xa)
    if not (len(Xb) == len(ya) == len(yb) == n):
        raise ValueError("condition arrays differ in length")
    return np.vstack([Xa, Xb]), np.concatenate([ya, yb]), np.concatenate([np.arange(n), np.arange(n)])


def episode_folds(n_episodes):
    """(n_episodes,) fold id of each episode: KFold(5, shuffle=True, random_state=0) over episode indices."""
    fold = np.full(n_episodes, -1, dtype=np.int64)
    for f, (_, test) in enumerate(KFold(N_FOLDS, shuffle=True, random_state=FOLD_SEED).split(np.arange(n_episodes))):
        fold[test] = f
    if (fold < 0).any():
        raise AssertionError("an episode got no fold")
    return fold


def choose_c(mean_losses, grid=C_GRID):
    """Index of the smallest mean log loss; exact ties go to the smaller C (grid ascending)."""
    if list(grid) != sorted(grid):
        raise ValueError("the C grid must be ascending")
    best = 0
    for i in range(1, len(grid)):
        if mean_losses[i] < mean_losses[best]:
            best = i
    return best


def average_probs(prob_list):
    """P(h) = mean over the half-readers' probabilities (each (N, H), rows sum to 1)."""
    stack = np.stack([np.asarray(p, dtype=np.float64) for p in prob_list])
    return stack.mean(axis=0)


def picks_and_margins(P):
    """(pick, margin): arg max_h P(h) with ties to the first grouping in configuration order (np.argmax), and the
    largest minus the second-largest P(h)."""
    P = np.asarray(P, dtype=np.float64)
    top = np.sort(P, axis=1)
    return P.argmax(axis=1).astype(np.int64), top[:, -1] - top[:, -2]


def smd(x_eval, x_bank):
    """Per column: (mean_eval - mean_bank) / sqrt((var_eval + var_bank) / 2), variances with ddof 1. A column with
    zero variance in both sets gives 0 when the means agree and NaN otherwise (reported as null)."""
    x_eval, x_bank = np.asarray(x_eval, dtype=np.float64), np.asarray(x_bank, dtype=np.float64)
    d = x_eval.mean(axis=0) - x_bank.mean(axis=0)
    s = np.sqrt((x_eval.var(axis=0, ddof=1) + x_bank.var(axis=0, ddof=1)) / 2)
    out = np.full(d.shape, np.nan)
    ok = s > 0
    out[ok] = d[ok] / s[ok]
    out[~ok & (d == 0)] = 0.0
    return out


def distribution(x):
    """Mean and deciles (10th to 90th percentiles, numpy linear) of a 1-d array."""
    x = np.asarray(x, dtype=np.float64)
    return {"n": int(len(x)), "mean": float(x.mean()),
            "deciles": {str(q): float(v) for q, v in zip(range(10, 100, 10), np.percentile(x, range(10, 100, 10)))}}
