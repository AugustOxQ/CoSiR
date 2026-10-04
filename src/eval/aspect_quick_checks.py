"""Quick checks after the A′ repair (spec docs/superpowers/specs/2026-10-04-new-method-quick-checks-design.md §4):
N1's centered agreement rule, diagonal KISSME on factor codes, N2's find-then-select cascade and D0's label-probe
told / inferred scorers. Every score function returns scores[cond][dir] -> (E, K), like src.eval.aspect_scorers."""

import numpy as np

from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS
from src.eval.aspect_scorers import EvalInputs

ASPECTS = ("emotion", "style", "genre")


def _finite_rows(*arrays) -> np.ndarray:
    ok = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        ok &= np.isfinite(np.asarray(a).reshape(len(a), -1)).all(axis=1)
    return ok


# ---------------------------------------------------------------- N1 (spec §4.2)

def centered_agreement_weights(sup_img, sup_txt, con_img, con_txt) -> np.ndarray:
    """(E,S,F) x4 -> (E,F): ReLU(cov_S - cov_C), L1-normalized, where cov_S(l) is the covariance across the S support
    pairs between the image code and the caption code of factor l (divided by S). An all-zero row stays zero; an
    episode with any non-finite code gets an all-NaN row (a miss, never a silent hit)."""
    def cov(x, y):
        x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
        return ((x - x.mean(axis=1, keepdims=True)) * (y - y.mean(axis=1, keepdims=True))).mean(axis=1)

    w = np.maximum(cov(sup_img, sup_txt) - cov(con_img, con_txt), 0.0)
    total = w.sum(axis=1, keepdims=True)
    w = np.where(total > 0, w / np.where(total > 0, total, 1.0), 0.0)
    w[~_finite_rows(sup_img, sup_txt, con_img, con_txt)] = np.nan
    return w.astype(np.float32)


def _code_sides(inputs: EvalInputs, ep, d: str):
    """(query codes (E,F), candidate codes (E,K,F), the 8 example codes of the query's modality (E,8,F))."""
    if d == "i2t":
        q, c = inputs.img_codes[ep.anchor], inputs.txt_codes[ep.candidates]
        ex = np.concatenate([inputs.img_codes[ep.pairs_a_img], inputs.img_codes[ep.pairs_b_img]], axis=1)
    else:
        q, c = inputs.txt_codes[ep.anchor], inputs.img_codes[ep.candidates]
        ex = np.concatenate([inputs.txt_codes[ep.pairs_a_txt], inputs.txt_codes[ep.pairs_b_txt]], axis=1)
    return q.astype(np.float64), c.astype(np.float64), ex.astype(np.float64)


def centered_term(inputs: EvalInputs, ep, uniform: bool = False) -> dict:
    """N1: sum_l w_l (q_l - mu^q_l)(c_l - mu^c_l), with w from centered_agreement_weights (or 1/F: the condition
    removed), mu^c the mean over the candidates and mu^q the mean over the 8 example items of the query's modality
    (the same 8 items under both conditions)."""
    out = {c: {} for c in CONDITIONS}
    for cond in CONDITIONS:
        si, st, ci, ct, _ = ep.condition(cond)
        if uniform:
            f = inputs.img_codes.shape[1]
            w = np.full((len(ep.anchor), f), 1.0 / f)
        else:
            w = centered_agreement_weights(inputs.img_codes[si], inputs.txt_codes[st], inputs.img_codes[ci],
                                           inputs.txt_codes[ct]).astype(np.float64)
        for d in DIRECTIONS:
            q, c, ex = _code_sides(inputs, ep, d)
            out[cond][d] = np.einsum("nf,nf,nkf->nk", w, q - ex.mean(axis=1),
                                     c - c.mean(axis=1, keepdims=True)).astype(np.float32)
    return out


# ---------------------------------------------------------------- diagonal KISSME on codes (comparator)

def code_scale(img_codes_train, txt_codes_train) -> np.ndarray:
    """(F,) per-factor std of the pooled image and caption codes of training rows; a dead factor (std 0) gets 1."""
    pooled = np.concatenate([np.asarray(img_codes_train), np.asarray(txt_codes_train)]).astype(np.float64)
    if not np.isfinite(pooled).all():
        raise ValueError("training codes must be finite")
    std = pooled.std(axis=0)
    return np.where(std > 0, std, 1.0)


def kissme_diag_term(inputs: EvalInputs, ep, scale: np.ndarray) -> dict:
    """Diagonal KISSME (Köstinger et al. 2012) on codes divided by ``scale``: per factor, v = mean over the 4 pairs of
    the squared image-minus-caption difference, m_l = 1/(v_S,l + 1) - 1/(v_C,l + 1) (ridge 1, as E1's KISSME on
    unit-variance coordinates), score = -sum_l m_l (q_l - c_l)^2."""
    s = np.asarray(scale, dtype=np.float64)
    ic, tc = inputs.img_codes / s, inputs.txt_codes / s
    out = {c: {} for c in CONDITIONS}
    for cond in CONDITIONS:
        si, st, ci, ct, _ = ep.condition(cond)
        v_s = ((ic[si] - tc[st]) ** 2).mean(axis=1)
        v_c = ((ic[ci] - tc[ct]) ** 2).mean(axis=1)
        m = 1.0 / (v_s + 1.0) - 1.0 / (v_c + 1.0)
        for d in DIRECTIONS:
            q, c = (ic[ep.anchor], tc[ep.candidates]) if d == "i2t" else (tc[ep.anchor], ic[ep.candidates])
            out[cond][d] = (-np.einsum("nf,nkf->nk", m, (q[:, None, :] - c) ** 2)).astype(np.float32)
    return out


# ---------------------------------------------------------------- N2 (spec §4.3)

def _topk(s: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest scores per row, best first (exact ties: the lower column first)."""
    return np.argsort(-s, axis=1, kind="stable")[:, :k]


def cascade_scores(control: dict, rerank: dict, k: int) -> dict:
    """Keep the order of ``control`` outside its top k; reorder the top k by ``rerank`` (ties broken by the control
    score) and keep them above every other candidate. A row with a non-finite control score, or a non-finite rerank
    score inside its top k, becomes all NaN (a miss)."""
    out = {c: {} for c in CONDITIONS}
    for cond in CONDITIONS:
        for d in DIRECTIONS:
            s = np.asarray(control[cond][d], dtype=np.float64)
            r = np.asarray(rerank[cond][d], dtype=np.float64)
            if not 1 <= k <= s.shape[1]:
                raise ValueError(f"k must be in 1..{s.shape[1]}, got {k}")
            rows = np.arange(len(s))[:, None]
            top = _topk(np.where(np.isfinite(s), s, -np.inf), k)
            r_top, s_top = r[rows, top], s[rows, top]
            order = np.lexsort((-s_top, -r_top), axis=-1)          # primary: rerank desc; secondary: control desc
            ranked = np.take_along_axis(top, order, axis=1)
            o = s.copy()
            base = np.max(np.where(np.isfinite(s), s, -np.inf), axis=1)
            o[rows, ranked] = base[:, None] + 1.0 + (k - np.arange(k))[None, :]
            o[~(np.isfinite(s).all(axis=1) & np.isfinite(r_top).all(axis=1))] = np.nan
            out[cond][d] = o.astype(np.float32)
    return out


def both_in_topk(control: dict, k: int) -> np.ndarray:
    """Per anchor, the share of its four rankings (2 conditions x 2 directions) whose top k under ``control`` holds
    both p_a (column 0) and p_b (column 1); a ranking with a non-finite score counts as 0."""
    vals = []
    for cond in CONDITIONS:
        for d in DIRECTIONS:
            s = np.asarray(control[cond][d], dtype=np.float64)
            top = _topk(np.where(np.isfinite(s), s, -np.inf), k)
            hit = (top == 0).any(axis=1) & (top == 1).any(axis=1) & np.isfinite(s).all(axis=1)
            vals.append(hit.astype(np.float64))
    return np.mean(vals, axis=0)


# ---------------------------------------------------------------- D0 (spec §4.1; diagnostic only)

def probe_dots(post: dict, ep, aspects=ASPECTS) -> dict:
    """{aspect: {dir: (E,K) p_h(query) . p_h(candidate)}}, each item scored with its own modality's posterior."""
    return {h: {"i2t": np.einsum("nc,nkc->nk", post[h]["img"][ep.anchor], post[h]["txt"][ep.candidates]),
                "t2i": np.einsum("nc,nkc->nk", post[h]["txt"][ep.anchor], post[h]["img"][ep.candidates])}
            for h in aspects}


def aspect_deltas(post: dict, ep, cond: str, aspects=ASPECTS) -> np.ndarray:
    """(E,H): Delta_h = S_h - C_h, the mean within-pair agreement p_h(image) . p_h(caption) over the 4 support pairs of
    condition ``cond`` minus the same over its 4 contrast pairs."""
    si, st, ci, ct, _ = ep.condition(cond)
    cols = []
    for h in aspects:
        pi, pt = post[h]["img"], post[h]["txt"]
        cols.append(np.einsum("nsc,nsc->ns", pi[si], pt[st]).mean(axis=1)
                    - np.einsum("nsc,nsc->ns", pi[ci], pt[ct]).mean(axis=1))
    return np.stack(cols, axis=1)


def _stacked_dots(post, ep, aspects):
    dots = probe_dots(post, ep, aspects)
    return {d: np.stack([dots[h][d] for h in aspects], axis=1) for d in DIRECTIONS}      # (E, H, K)


def told_scores(post: dict, ep, aspect_a, aspect_b, aspects=ASPECTS) -> dict:
    """D0 Told: the conditioned aspect's probe dot product. ``aspect_a`` / ``aspect_b`` are (E,) indices into
    ``aspects``: the aspect that condition a (resp. b) is about, per episode."""
    stack = _stacked_dots(post, ep, aspects)
    out = {}
    for cond, idx in (("a", np.asarray(aspect_a)), ("b", np.asarray(aspect_b))):
        out[cond] = {d: stack[d][np.arange(len(idx)), idx] for d in DIRECTIONS}
    return out


def inferred_weights(post: dict, ep, cond: str, mode: str, aspects=ASPECTS) -> tuple:
    """(E,H) aspect weights from Delta and the (E,) soft-fallback flag. 'hard': one-hot argmax (ties to the first
    aspect). 'soft': max(Delta, 0) normalised to sum 1, uniform 1/H where every Delta <= 0 (flag set)."""
    delta = aspect_deltas(post, ep, cond, aspects)
    if mode == "hard":
        w = np.zeros_like(delta)
        w[np.arange(len(delta)), delta.argmax(axis=1)] = 1.0
        return w, np.zeros(len(delta), dtype=bool)
    if mode == "soft":
        pos = np.maximum(delta, 0.0)
        total = pos.sum(axis=1, keepdims=True)
        fallback = total[:, 0] <= 0
        w = np.where(fallback[:, None], 1.0 / delta.shape[1], pos / np.where(total > 0, total, 1.0))
        return w, fallback
    raise ValueError(f"mode must be 'hard' or 'soft', got {mode!r}")


def inferred_scores(post: dict, ep, mode: str, aspects=ASPECTS) -> tuple:
    """D0 Inferred: the Told scores of each aspect weighted by inferred_weights; also returns, per condition, the
    weights and the soft-fallback flags."""
    stack = _stacked_dots(post, ep, aspects)
    out, info = {}, {}
    for cond in CONDITIONS:
        w, fallback = inferred_weights(post, ep, cond, mode, aspects)
        out[cond] = {d: np.einsum("nh,nhk->nk", w, stack[d]) for d in DIRECTIONS}
        info[cond] = {"weights": w, "fallback": fallback}
    return out, info
