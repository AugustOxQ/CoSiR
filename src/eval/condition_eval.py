"""Stage (d) evaluation: wrong-condition control, condition-use gain, human swap test, ceiling (spec §5-6)."""

from dataclasses import dataclass, replace

import numpy as np
import torch

from src.data.sampling import draw_distinct
from src.eval.label_episodes import EMOTION_CATCH_ALL, LabelEpisodes, tie_aware_rank
from src.model.conditioning import conditional_score, pair_codes


def _t(values, device=None):
    return torch.as_tensor(np.asarray(values), dtype=torch.float32, device=device)


def wrong_condition(episodes: LabelEpisodes, seed: int = 42) -> LabelEpisodes:
    """Give every episode another episode's supports and contrasts (a seeded derangement)."""
    n = len(episodes.anchor)
    if n < 2:
        raise ValueError("need at least two episodes for a derangement")
    order = np.random.default_rng(seed).permutation(n)
    source = np.empty(n, dtype=np.int64)
    source[order] = np.roll(order, 1)
    assert np.all(source != np.arange(n))
    return replace(episodes, supports=episodes.supports[source], contrasts=episodes.contrasts[source])


def _weights(scorer, img_codes, txt_codes, supports, contrasts, device):
    support = pair_codes(_t(img_codes[supports], device), _t(txt_codes[supports], device))
    contrast = pair_codes(_t(img_codes[contrasts], device), _t(txt_codes[contrasts], device))
    return scorer.weights(support, contrast)


def _scores(scorer, weights, direction, anchor, candidates, img_feat, txt_feat, img_codes, txt_codes, device):
    if direction == "i2t":
        qf, cf, qc, cc = img_feat, txt_feat, img_codes, txt_codes
    else:
        qf, cf, qc, cc = txt_feat, img_feat, txt_codes, img_codes
    return scorer.score(_t(qf[anchor], device), _t(cf[candidates], device), _t(qc[anchor], device),
                        _t(cc[candidates], device), weights)


def label_ranks(scorer, img_feat, txt_feat, img_codes, txt_codes, episodes: LabelEpisodes, device=None) -> dict:
    candidates = np.concatenate([episodes.positive[:, None], episodes.distractors], axis=1)
    with torch.no_grad():
        w = _weights(scorer, img_codes, txt_codes, episodes.supports, episodes.contrasts, device)
        return {d: tie_aware_rank(_scores(scorer, w, d, episodes.anchor, candidates, img_feat, txt_feat,
                                          img_codes, txt_codes, device)).cpu().numpy()
                for d in ("i2t", "t2i")}


def paired_bootstrap(values, n_boot: int = 5000, seed: int = 42) -> dict:
    values = np.asarray(values, dtype=np.float64)
    idx = np.random.default_rng(seed).integers(0, len(values), (n_boot, len(values)))
    boots = values[idx].mean(axis=1)
    return {"point": float(values.mean()),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}


def condition_use_gain(model_ranks, model_wrong_ranks, naive_ranks, naive_wrong_ranks, n_boot=5000, seed=42) -> dict:
    """Per episode: [hit(model) - hit(model | wrong c)] - [hit(naive) - hit(naive | wrong c)], R@1."""
    diffs = {}
    for d in ("i2t", "t2i"):
        hit = lambda r: (np.asarray(r[d]) <= 1).astype(np.float64)
        diffs[d] = (hit(model_ranks) - hit(model_wrong_ranks)) - (hit(naive_ranks) - hit(naive_wrong_ranks))
    out = {d: paired_bootstrap(v, n_boot, seed) for d, v in diffs.items()}
    out["mean"] = paired_bootstrap(0.5 * (diffs["i2t"] + diffs["t2i"]), n_boot, seed)
    return out


@dataclass(frozen=True)
class HumanSwapEpisodes:
    anchor: np.ndarray
    supports_emo: np.ndarray
    contrasts_emo: np.ndarray
    supports_style: np.ndarray
    contrasts_style: np.ndarray
    candidates: np.ndarray        # column 0 = p_emo, column 1 = p_style, then negatives
    emotions: np.ndarray
    styles: np.ndarray


def build_human_swap_episodes(data, groups, rows, n_episodes, seed=42, num_support=4, num_contrast=4,
                              num_negatives=11, min_paintings=30) -> HumanSwapEpisodes:
    emotions, styles = np.asarray(data.emotions), np.asarray(data.art_styles)
    groups, rows = np.asarray(groups), np.asarray(rows, dtype=np.int64)
    rng = np.random.default_rng(seed)
    has_emotion = {e: np.isin(groups, np.unique(groups[emotions == e])) for e in np.unique(emotions)}
    row_em, row_st = emotions[rows], styles[rows]

    def eligible(values, value):
        return len(np.unique(groups[rows[values == value]])) >= min_paintings

    ok_em = {e for e in np.unique(row_em) if e != EMOTION_CATCH_ALL and eligible(row_em, e)}
    ok_st = {s for s in np.unique(row_st) if eligible(row_st, s)}
    anchors = rows[np.isin(row_em, list(ok_em)) & np.isin(row_st, list(ok_st))]
    if len(anchors) == 0:
        raise ValueError("no eligible anchors")
    names = ("anchor", "supports_emo", "contrasts_emo", "supports_style", "contrasts_style", "candidates",
             "emotions", "styles")
    fields = {k: [] for k in names}
    failures = 0
    while len(fields["anchor"]) < n_episodes:
        a = int(anchors[rng.integers(len(anchors))])
        e, s = emotions[a], styles[a]
        clean_e = ~has_emotion[e][rows]                     # painting carries no annotation of e
        pools = {
            "sup_emo": rows[(row_em == e) & (row_st != s)],
            "con_emo": rows[clean_e],
            "sup_style": rows[(row_st == s) & clean_e],
            "con_style": rows[row_st != s],
            "p_emo": rows[(row_em == e) & (row_st != s)],
            "p_style": rows[(row_st == s) & clean_e],
            "neg": rows[(row_st != s) & clean_e],
        }
        used = {groups[a]}
        try:
            picked = [draw_distinct(rng, pools["sup_emo"], groups, used, num_support),
                      draw_distinct(rng, pools["con_emo"], groups, used, num_contrast),
                      draw_distinct(rng, pools["sup_style"], groups, used, num_support),
                      draw_distinct(rng, pools["con_style"], groups, used, num_contrast)]
            candidates = (draw_distinct(rng, pools["p_emo"], groups, used, 1)
                          + draw_distinct(rng, pools["p_style"], groups, used, 1)
                          + draw_distinct(rng, pools["neg"], groups, used, num_negatives))
        except ValueError:
            failures += 1
            if failures > 10 * n_episodes:
                raise RuntimeError("could not build enough human swap episodes")
            continue
        for key, value in zip(names, [a, *picked, candidates, e, s]):
            fields[key].append(value)
    return HumanSwapEpisodes(*(np.asarray(fields[k], dtype=np.int64) for k in names[:6]),
                             emotions=np.asarray(fields["emotions"]), styles=np.asarray(fields["styles"]))


def human_swap_success(scorer, img_feat, txt_feat, img_codes, txt_codes, episodes: HumanSwapEpisodes,
                       device=None) -> dict:
    """Success = p_emo above p_style under the emotion condition AND p_style above p_emo under the style one."""
    out = {}
    with torch.no_grad():
        w_emo = _weights(scorer, img_codes, txt_codes, episodes.supports_emo, episodes.contrasts_emo, device)
        w_style = _weights(scorer, img_codes, txt_codes, episodes.supports_style, episodes.contrasts_style, device)
        for d in ("i2t", "t2i"):
            s_emo = _scores(scorer, w_emo, d, episodes.anchor, episodes.candidates, img_feat, txt_feat,
                            img_codes, txt_codes, device)
            s_style = _scores(scorer, w_style, d, episodes.anchor, episodes.candidates, img_feat, txt_feat,
                              img_codes, txt_codes, device)
            ok = (s_emo[:, 0] > s_emo[:, 1]) & (s_style[:, 1] > s_style[:, 0])
            finite = torch.isfinite(s_emo).all(dim=1) & torch.isfinite(s_style).all(dim=1)
            out[d] = (ok & finite).cpu().numpy()
    return out


def swap_success_difference(model_success, naive_success, n_boot=5000, seed=42) -> dict:
    diff = {d: np.asarray(model_success[d], float) - np.asarray(naive_success[d], float) for d in ("i2t", "t2i")}
    out = {d: paired_bootstrap(v, n_boot, seed) for d, v in diff.items()}
    out["pooled"] = paired_bootstrap(0.5 * (diff["i2t"] + diff["t2i"]), n_boot, seed)
    out["rates"] = {"model": {d: float(np.mean(model_success[d])) for d in diff},
                    "naive": {d: float(np.mean(naive_success[d])) for d in diff}}
    return out


def ceiling_ranks(img_feat, txt_feat, img_codes, txt_codes, episodes: LabelEpisodes, beta=0.3, steps=100, lr=0.1,
                  device=None) -> dict:
    """Oracle upper bound: per-episode simplex weights optimised on the episode's own positive (diagnostic only)."""
    candidates = np.concatenate([episodes.positive[:, None], episodes.distractors], axis=1)
    out = {}
    for d in ("i2t", "t2i"):
        if d == "i2t":
            qf, cf, qc, cc = img_feat, txt_feat, img_codes, txt_codes
        else:
            qf, cf, qc, cc = txt_feat, img_feat, txt_codes, img_codes
        q, c = _t(qf[episodes.anchor], device), _t(cf[candidates], device)
        qcode, ccode = _t(qc[episodes.anchor], device), _t(cc[candidates], device)
        theta = torch.zeros(len(episodes.anchor), qcode.shape[1], device=q.device, requires_grad=True)
        optimizer = torch.optim.Adam([theta], lr=lr)
        for _ in range(steps):
            s = conditional_score(q, c, qcode, ccode, torch.softmax(theta, dim=1), beta)
            margin = s[:, 0] - 0.01 * torch.logsumexp(s[:, 1:] / 0.01, dim=1)
            optimizer.zero_grad()
            (-margin.mean()).backward()
            optimizer.step()
        with torch.no_grad():
            s = conditional_score(q, c, qcode, ccode, torch.softmax(theta, dim=1), beta)
            out[d] = tie_aware_rank(s).cpu().numpy()
    return out
