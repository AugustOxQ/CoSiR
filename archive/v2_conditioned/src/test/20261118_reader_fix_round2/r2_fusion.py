"""The round-2 fusion family as pure functions (DECISION_RULE.md section 4.5 to 4.6): the top-k restriction, the 896
cells, exact integer cross-fit criteria and the assembly of each half's pick on the other half. Everything here takes
z-scored torch dicts / numpy arrays, so the unit tests run it on synthetic data. run_r2_fusion.py wires it to the bundle.

Scores are dicts {cond: {dir: (E, 13)}}. B is condition-free, so the top-k set K (taken from B only) is the same under
both conditions of an episode in each direction.
"""
import numpy as np

from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS, per_anchor
from src.eval.aspect_nested import NESTED_A, NESTED_U, _combine, control_sums
from src.eval.aspect_quick_checks import _require_condition_free

KTOPS = (13, 5, 3, 2)                       # k_top, outermost, in this order (rule 4.5 item 7)
N_TAU = 4
N_U, N_A = len(NESTED_U), len(NESTED_A)     # 7, 8
CELLS_PER_KAPPA = N_TAU * N_U * N_A         # 224
N_CELLS = len(KTOPS) * CELLS_PER_KAPPA      # 896


# ---------------------------------------------------------------- cells

def cell_number(kappa, t, u, a):
    """((kappa * 4 + t) * 7 + u) * 8 + a with zero-based kappa (k_top index), t (tau index), u, a (lambda indices)."""
    return ((kappa * N_TAU + t) * N_U + u) * N_A + a


def decode_cell(i):
    """-> (kappa, t, u, a), the inverse of cell_number."""
    if not 0 <= i < N_CELLS:
        raise ValueError(f"cell {i} out of range")
    a = i % N_A
    u = (i // N_A) % N_U
    t = (i // (N_A * N_U)) % N_TAU
    kappa = i // CELLS_PER_KAPPA
    return kappa, t, u, a


def cell_values(i):
    """-> (k_top, tau index, lambda_u, lambda_a) of cell number i."""
    kappa, t, u, a = decode_cell(i)
    return KTOPS[kappa], t, NESTED_U[u], NESTED_A[a]


def describe_cell(i, taus):
    k, t, lu, la = cell_values(i)
    return {"cell": int(i), "k_top": int(k), "tau_index": int(t), "tau": float(taus[t]),
            "lambda_u": float(lu), "lambda_a": float(la)}


# ---------------------------------------------------------------- top-k set and restriction

def rank_info(B):
    """{dir: {"order", "pos"}} from B alone (rule 4.5 item 5): order = argsort(-b, stable) per ranking row, so ties go to
    the lower candidate index; pos[j] = position of candidate j in order. B[a] == B[b] is asserted."""
    info = {}
    for d in DIRECTIONS:
        ba, bb = np.asarray(B["a"][d]), np.asarray(B["b"][d])
        if not np.array_equal(ba, bb, equal_nan=True):
            raise AssertionError("B must be identical under both conditions")
        order = np.argsort(-ba, axis=1, kind="stable")
        pos = np.empty_like(order)
        np.put_along_axis(pos, order, np.arange(order.shape[1])[None, :].repeat(order.shape[0], 0), axis=1)
        info[d] = {"order": order, "pos": pos}
    return info


def restrict(S, pos, k_top):
    """Rule 4.5 item 6. Inside K (pos < k_top) the scores are kept; the candidate at position p >= k_top gets
    min over K of S - 1 - (p - k_top), float64, so every outside candidate is at least 1 below the lowest in K and the
    outside candidates keep B's order. k_top = 13 returns S itself, bit-identical."""
    S = np.asarray(S)
    if k_top >= S.shape[1]:
        return S
    S64 = S.astype(np.float64)
    inK = pos < k_top
    low = np.where(inK, S64, np.inf).min(axis=1, keepdims=True)
    return np.where(inK, S64, low - 1.0 - (pos - k_top))


def restrict_scores(scores, info, k_top):
    return {c: {d: restrict(scores[c][d], info[d]["pos"], k_top) for d in DIRECTIONS} for c in CONDITIONS}


# ---------------------------------------------------------------- integer per-anchor metrics

def as_int4(x, what="metric"):
    """4 * (a per-anchor metric), as exact integers (every per-episode R@1 and gain is a multiple of 0.25)."""
    x = np.asarray(x, np.float64)
    r = np.rint(4.0 * x)
    if not np.array_equal(r / 4.0, x):
        raise AssertionError(f"{what} is not a multiple of 0.25")
    return r.astype(np.int8)


def int_metrics(scores):
    """(4*R@1, 4*gain) per episode as int8 from a restricted or unrestricted score dict."""
    pa = per_anchor(scores)
    return as_int4(pa["r1"], "r1"), as_int4(pa["gain"], "gain")


def cell_statistics(zB, info, gated, G, n_kappa=len(KTOPS)):
    """Per cell (row = cell number; the first n_kappa k_top values only): int8 4*R@1 and 4*gain of the restricted gated
    fused reader, int8 4*R@1 of the restricted gated counterpart (condition-free asserted per cell, with the same K).
    gated[t], G[t]: {cond: {dir: (E, 13) torch}}, t = tau index."""
    n_rows = n_kappa * CELLS_PER_KAPPA
    E = len(zB["a"]["i2t"])
    fri = np.zeros((n_rows, E), np.int8)
    fgi = np.zeros((n_rows, E), np.int8)
    cri = np.zeros((n_rows, E), np.int8)
    for t in range(N_TAU):
        for u, lu in enumerate(NESTED_U):
            for a, la in enumerate(NESTED_A):
                sf = _combine(zB, zB, gated[t], lu, la)
                sc = _combine(zB, zB, G[t], lu, la)
                for kappa in range(n_kappa):
                    k = KTOPS[kappa]
                    rf_ = restrict_scores(sf, info, k)
                    rc_ = restrict_scores(sc, info, k)
                    _require_condition_free(rc_, f"restricted counterpart (cell {cell_number(kappa, t, u, a)})")
                    i = cell_number(kappa, t, u, a)
                    fri[i], fgi[i] = int_metrics(rf_)
                    cri[i] = int_metrics(rc_)[0]
    return fri, fgi, cri


# ---------------------------------------------------------------- exact integer cross-fit (rule 4.5 item 8, 4.6)

def tune_mask(parity, half):
    return np.asarray(parity) == half


def control_choice(zB, parity):
    """Per tune half: sigma* = the smallest of the 30 control sums with the largest integer rho of (1+sigma) z(B)
    (unrestricted). -> {half: (sigma, rho_ctrl int)}."""
    sums = control_sums()
    ri = [int_metrics(_combine(zB, zB, zB, s, 0.0))[0] for s in sums]
    out = {}
    for half in (0, 1):
        tune = tune_mask(parity, half)
        best_i, best = 0, None
        for i, r in enumerate(ri):
            rho = int(r[tune].sum(dtype=np.int64))
            if best is None or rho > best:                    # strictly greater: ties keep the smaller sigma
                best_i, best = i, rho
        out[half] = (float(sums[best_i]), int(best))
    return out


def fused_criterion(fri, fgi, tune, rho_ctrl, allowed):
    """int64 min(rho - rho_ctrl, gamma) per allowed cell on the tune half."""
    rho = fri[allowed][:, tune].sum(axis=1, dtype=np.int64)
    gam = fgi[allowed][:, tune].sum(axis=1, dtype=np.int64)
    return np.minimum(rho - np.int64(rho_ctrl), gam), rho, gam


def select_fused(fri, fgi, ctrl, parity, allowed=None):
    """Min-margin pick per tune half, compared as integers, ties to the lowest cell number. -> {half: cell number}."""
    allowed = np.arange(len(fri)) if allowed is None else np.asarray(sorted(allowed))
    picks = {}
    for half in (0, 1):
        crit, _, _ = fused_criterion(fri, fgi, tune_mask(parity, half), ctrl[half][1], allowed)
        picks[half] = int(allowed[int(np.argmax(crit))])                  # argmax keeps the first maximum
    return picks


def select_cf(cri, parity, allowed=None):
    """Counterpart: per tune half the cell with the largest integer rho, ties to the lowest cell number."""
    allowed = np.arange(len(cri)) if allowed is None else np.asarray(sorted(allowed))
    out = {}
    for half in (0, 1):
        rho = cri[allowed][:, tune_mask(parity, half)].sum(axis=1, dtype=np.int64)
        out[half] = int(allowed[int(np.argmax(rho))])
    return out


def pick_details(fri, fgi, cri, ctrl, parity, fpick, cpick):
    """Integer criteria behind the chosen cells (diagnostic)."""
    out = {"fused": {}, "counterpart": {}}
    for half in (0, 1):
        tune = tune_mask(parity, half)
        out["fused"][str(half)] = {"rho": int(fri[fpick[half]][tune].sum(dtype=np.int64)),
                                   "gamma": int(fgi[fpick[half]][tune].sum(dtype=np.int64)),
                                   "rho_ctrl": int(ctrl[half][1]), "n_tune": int(tune.sum())}
        out["counterpart"][str(half)] = {"rho": int(cri[cpick[half]][tune].sum(dtype=np.int64)),
                                         "n_tune": int(tune.sum())}
    return out


def assemble(zB, info, terms, picks, parity):
    """Scores of each half's picked cell (restricted) applied to the OTHER half's episodes (parity != half).
    terms[t]: {cond: {dir: (E, 13)}}. -> {cond: {dir: (E, 13) float64}} (inside K the float32 values exactly)."""
    out = {c: {d: np.empty(np.asarray(zB[c][d]).shape, np.float64) for d in DIRECTIONS} for c in CONDITIONS}
    for half in (0, 1):
        apply = np.asarray(parity) != half
        kappa, t, u, a = decode_cell(picks[half])
        s = restrict_scores(_combine(zB, zB, terms[t], NESTED_U[u], NESTED_A[a]), info, KTOPS[kappa])
        for c in CONDITIONS:
            for d in DIRECTIONS:
                out[c][d][apply] = s[c][d][apply]
    return out
