"""Quick checks after the A′ repair (spec docs/superpowers/specs/2026-10-04-new-method-quick-checks-design.md §4):
N1's centered agreement rule, diagonal KISSME on factor codes, N2's find-then-select cascade and D0's label-probe
told / inferred scorers. Every score function returns scores[cond][dir] -> (E, K), like src.eval.aspect_scorers."""

import numpy as np

from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS, per_anchor
from src.eval.aspect_nested import nested_cells, nested_scores
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


# ---------------------------------------------------------------- decision rule
# src/test/20261108_new_method_quick_checks/DECISION_RULE.md, committed before any check ran.

D0_FRACTION = 0.5          # §2: "close to Told" = Inferred keeps at least half of Told's condition gain
N2_RARE_SHARE = 10.0       # §5: both aspect candidates "rarely" in the top k = a share below 10%
CONFIG_ORDER = ("N1-nested-A3", "N1-nested-C0", "N1-nested-SE", "N2-2-agree", "N2-3-agree", "N2-5-agree",
                "N2-2-N1", "N2-3-N1", "N2-5-N1")
NEXT = {1: "fix the passing configuration and test it on fresh episode seeds 45, 47, 48 (GO rule, DECISION_RULE.md §6)",
        2: "build N6 (cross-modal heads on the label-free k-means partitions, aspect picked by D0's decision variant), "
           "run the same checks, then the same test",
        3: "stop method work; move to branch 3 with D0, N1 and N2 reported as analysis results"}


def d0_reading(told_gain: dict, gains: dict) -> dict:
    """§2. ``told_gain``: Told's pooled condition gain {'point', 'ci95'} (points); ``gains``: {'hard': point,
    'soft': point}. The decision variant has the larger gain (ties to 'hard'). 'close' if its gain >= 0.5 x Told's,
    'far' otherwise, 'unreadable' if Told's gain has a lower bound <= 0."""
    variant = "hard" if gains["hard"] >= gains["soft"] else "soft"
    threshold = D0_FRACTION * told_gain["point"]
    if told_gain["ci95"][0] <= 0:
        reading = "unreadable"
    else:
        reading = "close" if gains[variant] >= threshold else "far"
    return {"variant": variant, "gain": float(gains[variant]), "told_gain": float(told_gain["point"]),
            "threshold": float(threshold), "reading": reading}


def config_passes(r1_vs_control: dict, gain_vs_control: dict) -> bool:
    """§4: both paired differences against the configuration's own condition-free control have lower bounds > 0."""
    return bool(r1_vs_control["ci95"][0] > 0 and gain_vs_control["ci95"][0] > 0)


def n1_stop(gain_minus_current: float, either_n1: float, either_cos: float) -> dict:
    """§5: N1 stops on a checkpoint if its term-only gain is not above the current rule's (paired point) or its
    term-only either rate is below cosine's (points)."""
    reasons = []
    if gain_minus_current <= 0:
        reasons.append("term-only gain not above the current rule")
    if either_n1 < either_cos:
        reasons.append("term-only either rate below cosine")
    return {"stop": bool(reasons), "reasons": reasons}


def n2_reading(share_both: float, gain_vs_control: dict) -> dict:
    """§5 readings at one k: both aspect candidates rarely in the top k; the reordering adds no gain."""
    return {"rare": bool(share_both < N2_RARE_SHARE), "no_gain": bool(gain_vs_control["ci95"][0] <= 0)}


def decision_row(configs: list, d0: str) -> dict:
    """§4, first matching row wins. ``configs``: dicts with 'name', 'passes', 'm_r', 'm_g' (points), in
    CONFIG_ORDER (configurations that were not computed are left out). Several passing: the largest
    min(m_r, m_g), ties to the earlier one."""
    names = [c["name"] for c in configs]
    if any(n not in CONFIG_ORDER for n in names) or names != sorted(names, key=CONFIG_ORDER.index):
        raise ValueError(f"configs must be named from CONFIG_ORDER and kept in its order, got {names}")
    if d0 not in ("close", "far", "unreadable"):
        raise ValueError(f"unknown D0 reading {d0!r}")
    passing = [c for c in configs if c["passes"]]
    if passing:
        best = max(passing, key=lambda c: min(c["m_r"], c["m_g"]))      # max keeps the first of equal maxima
        return {"row": 1, "config": best["name"], "next": NEXT[1]}
    if d0 == "unreadable":
        return {"row": None, "config": None, "next": "D0 cannot be read (Told's gain is not reliably positive); "
                                                      "report to the user without applying a row"}
    row = 2 if d0 == "close" else 3
    return {"row": row, "config": None, "next": NEXT[row]}


GO_COMPARATORS = ("cosine", "rca", "control")


def go_verdict(pooled: dict, comparators=GO_COMPARATORS) -> dict:
    """§6 (and ADDENDUM_1 R3 with comparators=ADDENDUM_COMPARATORS): GO iff, on the pooled test seeds, the paired
    difference config − comparator has a 95% lower bound above 0 for R@1 and for condition gain against every
    comparator. ``pooled[comparator][metric]`` is a compare() result."""
    failed = [f"{c}/{m}" for c in comparators for m in ("r1", "gain") if not pooled[c][m]["ci95"][0] > 0]
    return {"go": not failed, "failed": failed}


# ---------------------------------------------------------------- ADDENDUM_1.md: matched condition-free control

ADDENDUM_COMPARATORS = (*GO_COMPARATORS, "matched")


def _require_condition_free(scores: dict, what: str) -> None:
    for d in DIRECTIONS:
        if not np.array_equal(np.asarray(scores["a"][d]), np.asarray(scores["b"][d]), equal_nan=True):
            raise ValueError(f"{what} must be identical under both conditions (condition-free)")


def crossfit_condition_free(cos: dict, t_u: dict, t_c: dict, parity) -> tuple:
    """ADDENDUM_1 R1: the matched condition-free control. Over the 56 nested cells z(cos) + λ_u·z(t_u) + λ_a·z(t_c),
    with t_u and t_c both condition-free, each episode-index parity half picks the cell with the highest R@1 (ties to
    the first cell in row-major order, λ_u outer); each half's pick scores the other half."""
    _require_condition_free(t_u, "t_u")
    _require_condition_free(t_c, "t_c")
    n = len(cos["a"]["i2t"])
    parity = np.asarray(parity)
    if parity.shape != (n,) or not np.isin(parity, (0, 1)).all() or not ((parity == 0).any() and (parity == 1).any()):
        raise ValueError(f"parity must be a length-{n} array of 0/1 with both halves non-empty")
    cache = {}

    def scored(cell):
        if cell not in cache:
            cache[cell] = nested_scores(cos, t_u, t_c, *cell)
        return cache[cell]

    def r1(cell, rows):
        s = scored(cell)
        return float(per_anchor({c: {d: s[c][d][rows] for d in DIRECTIONS} for c in CONDITIONS})["r1"].mean())

    out = {c: {d: np.empty(np.asarray(cos[c][d]).shape, np.float32) for d in DIRECTIONS} for c in CONDITIONS}
    picks = {}
    for half in (0, 1):
        tune, apply = parity == half, parity != half
        cell = max(nested_cells(), key=lambda c: r1(c, tune))        # max keeps the first of equal maxima
        picks[half] = [float(cell[0]), float(cell[1])]
        for c in CONDITIONS:
            for d in DIRECTIONS:
                out[c][d][apply] = scored(cell)[c][d][apply]
    return out, picks


# ---------------------------------------------------------------- N6 (ADDENDUM_2_N6.md)

def uniform_probe_scores(post: dict, ep, aspects=ASPECTS) -> dict:
    """The condition-free probe score: the mean over ``aspects`` of p_h(query) . p_h(candidate), identical under both
    conditions (N6's T_6u when ``post`` holds the partition heads)."""
    stack = _stacked_dots(post, ep, aspects)
    mean = {d: stack[d].mean(axis=1) for d in DIRECTIONS}
    return {c: {d: mean[d].copy() for d in DIRECTIONS} for c in CONDITIONS}
