"""Round 3 readers, gates and the 224-cell fusion families (DECISION_RULE.md D5, D6, D8, D9, section 4 items 3 to 5, 7 item 7).

Pure functions on a seed's bundle (a SimpleNamespace, interface of r3_bundle): n, cl, parity, pair_index, B and Bp
({c: {d: (n, 13) float32}}), pB, pBp, stack ({d: (n, 3, 13)}, A0 order) and F ({c: (n, 18) float64}). Everything here
is CPU, takes no file and no seed-42 array. Round 1's and round 2's pieces are reused, not reimplemented.

R1's gate is 1[m >= tau_t]; AFF's gate is R1's times 1[pick = affect] (pick = numpy.argmax of P, so an arg-max tie that
includes affect counts as affect, affect being index 0 of A0). Cell number (t*7 + u)*8 + a, 224 cells (round 2's
cells 0 to 223).
"""
import numpy as np

import r3_common as R

C, K, rb, rbe, rf, F = R.C, R.K, R.rb, R.rbe, R.rf, R.F
from src.eval.aspect_metrics import CONDITIONS, per_anchor  # noqa: E402
from src.eval.aspect_nested import _zdict  # noqa: E402

N_CELLS = F.CELLS_PER_KAPPA       # 224
N_GROUPINGS = len(R.A0)


# ---------------------------------------------------------------- reader

def reader(bundle, readers=None):
    """Frozen round-1 half-readers on the bundle's features (D5). -> {"P": {c: (n, 3) float64}, "T": {c: {d: (n, 13)
    float32}}, "m": {c: (n,) float64 top-two margins}, "pick": {c: (n,) int64 arg max, ties to the first grouping}}.
    readers: the pickle dict of rb.load_readers (injected by tests); default rb.load_readers("A0", False), never smoke."""
    if readers is None:
        readers = rb.load_readers("A0", False)[0]
    if readers["feature_names"] != rf.feature_names(R.A0):
        raise AssertionError("the readers were trained on another feature layout")
    P = {c: rf.average_probs(rbe.half_reader_probs(readers, bundle.F[c], N_GROUPINGS)) for c in CONDITIONS}
    for c in CONDITIONS:
        if not np.allclose(P[c].sum(axis=1), 1.0, rtol=0, atol=1e-12):
            raise AssertionError("averaged probabilities do not sum to 1")
    T = C.expected_term(bundle.stack, P)
    pm = {c: rf.picks_and_margins(P[c]) for c in CONDITIONS}
    m = {c: pm[c][1] for c in CONDITIONS}
    for c in CONDITIONS:
        if not np.array_equal(m[c], C.top_two_margin(P[c])):
            raise AssertionError("top-two margin differs from common.top_two_margin")
    return {"P": P, "T": T, "m": m, "pick": {c: pm[c][0] for c in CONDITIONS}}


# ---------------------------------------------------------------- gates

def gates_r1(m, taus):
    """[ {c: (n,) float32} ] x 4: g_t^c = 1[m^c >= tau_t] (float64 margins against float64 tau; D6)."""
    g = K.gates(m, list(taus))
    return [g[k] for k in range(len(taus))]


def gates_aff(m, pick, taus):
    """AFF's gate: R1's gate times 1[pick = affect] (D6). pick = arg max of P, so a tie that includes affect is affect."""
    out = []
    for gt in gates_r1(m, taus):
        out.append({c: (gt[c] * (np.asarray(pick[c]) == R.AFFECT).astype(np.float32)).astype(np.float32)
                    for c in CONDITIONS})
    return out


def open_count(g, c):
    """Integer number of open gates of condition c."""
    return int(np.count_nonzero(np.asarray(g[c])))


def affect_shares(g_aff):
    """share_c = (open AFF tau_0 gates of condition c) / E, float64 (rule 7 item 7)."""
    return {c: open_count(g_aff[0], c) / len(g_aff[0][c]) for c in CONDITIONS}


def gates_random(g_r1, share, rng_seed, n):
    """Random-share control (rule 7 item 7): keep^c = 1[u < share_c], u = default_rng(rng_seed).random(n), condition a
    drawn first, then b, from one generator. Gate = R1's gate at every tau index times keep^c."""
    gen = np.random.default_rng(rng_seed)
    keep = {}
    for c in CONDITIONS:                      # CONDITIONS order is ("a", "b"): a first
        keep[c] = (gen.random(n) < float(share[c])).astype(np.float32)
    return [{c: (gt[c] * keep[c]).astype(np.float32) for c in CONDITIONS} for gt in g_r1]


def open_shares(gates, pair_index):
    """Label-free open counts and shares per tau index. Integer counts are primary (rule 5 item 3 compares counts);
    shares (percent) are count / episodes in float64, never a float32 mean."""
    pair_index = np.asarray(pair_index)
    out = {}
    for k, g in enumerate(gates):
        E = len(g["a"])
        cnt = {c: open_count(g, c) for c in CONDITIONS}
        pp_cnt, pp = {}, {}
        for i, p in enumerate(C.POOLED_ORDER):
            msk = pair_index == i
            e_i = int(msk.sum())
            pp_cnt[p] = {c: int(np.count_nonzero(np.asarray(g[c])[msk])) for c in CONDITIONS}
            pp[p] = 100.0 * (pp_cnt[p]["a"] + pp_cnt[p]["b"]) / (2 * e_i) if e_i else float("nan")
        out[f"tau_{k}"] = {
            "open_count": cnt, "n_episodes": E,
            "overall": 100.0 * (cnt["a"] + cnt["b"]) / (2 * E),
            "a": 100.0 * cnt["a"] / E, "b": 100.0 * cnt["b"] / E,
            "per_pair_open_count": pp_cnt, "per_pair": pp}
    return out


# ---------------------------------------------------------------- the 224-cell family

def _terms(bundle, T, gates):
    zB, zT = _zdict(bundle.B), _zdict(T)
    gated = {t: K.gated_terms(zT, gates[t]) for t in range(len(gates))}
    G = {t: K.g_cf(gated[t]) for t in gated}
    return zB, gated, G


def _as_picks(cells):
    picks = ({int(h): int(c) for h, c in cells.items()} if isinstance(cells, dict)
             else {h: int(c) for h, c in enumerate(cells)})
    if sorted(picks) != [0, 1] or not all(0 <= c < N_CELLS for c in picks.values()):
        raise ValueError(f"cells must be one per tune half, each in 0..{N_CELLS - 1}: {picks}")
    return picks


def run_family(bundle, T, gates, fused_only=False):
    """The 224 cells of one gate set. Integer cross-fits (round 2's): control sigma* by tune half, min-margin fused pick,
    max-R@1 counterpart pick (ties to the lowest cell number, integer sums). -> {"fpick", "cpick" (None if fused_only),
    "sigma" {half: sigma*}, "fused" per-anchor dict, "cf" per-anchor dict (None if fused_only), "details"}.
    The counterpart's per-cell statistics are computed in the same pass either way; with fused_only the counterpart is
    not cross-fitted, assembled or reported (no cpick, no cf, no counterpart field in details)."""
    if len(gates) != F.N_TAU:
        raise ValueError(f"{len(gates)} gate sets, expected {F.N_TAU}")
    parity = np.asarray(bundle.parity)
    zB, gated, G = _terms(bundle, T, gates)
    info = F.rank_info(bundle.B)
    fri, fgi, cri = F.cell_statistics(zB, info, gated, G, 1)
    if fri.shape[0] != N_CELLS:
        raise AssertionError(f"{fri.shape[0]} cells, expected {N_CELLS}")
    ctrl = F.control_choice(zB, parity)
    fpick = F.select_fused(fri, fgi, ctrl, parity)
    fused = F.assemble(zB, info, gated, fpick, parity)
    C._assert_finite(fused, "fused")
    if fused_only:                      # rule 6.4: the counterpart is not cross-fitted, assembled or written
        details = {"fused": {}}
        for h in (0, 1):
            tune = F.tune_mask(parity, h)
            details["fused"][str(h)] = {"rho": int(fri[fpick[h]][tune].sum(dtype=np.int64)),
                                        "gamma": int(fgi[fpick[h]][tune].sum(dtype=np.int64)),
                                        "rho_ctrl": int(ctrl[h][1]), "n_tune": int(tune.sum())}
    else:
        cpick = F.select_cf(cri, parity)
        details = F.pick_details(fri, fgi, cri, ctrl, parity, fpick, cpick)
    out = {"fpick": fpick, "cpick": None, "sigma": {h: ctrl[h][0] for h in (0, 1)}, "fused": per_anchor(fused),
           "cf": None, "details": details, "ctrl": {h: {"sigma": ctrl[h][0], "rho_ctrl": ctrl[h][1]} for h in (0, 1)}}
    if not fused_only:
        cfs = F.assemble(zB, info, G, cpick, parity)
        C._assert_finite(cfs, "counterpart")
        out["cpick"], out["cf"] = cpick, per_anchor(cfs)
    return out


def score_frozen(bundle, T, gates, fused_cells, cf_cells):
    """Frozen-cell line (rule 6 item 9): the cell chosen on tune half h scores the episodes of parity 1 - h.
    fused_cells, cf_cells: {half: cell} or (cell_half0, cell_half1). -> {"fused", "cf"} per-anchor dicts."""
    parity = np.asarray(bundle.parity)
    zB, gated, G = _terms(bundle, T, gates)
    info = F.rank_info(bundle.B)
    fused = F.assemble(zB, info, gated, _as_picks(fused_cells), parity)
    cfs = F.assemble(zB, info, G, _as_picks(cf_cells), parity)
    C._assert_finite(fused, "frozen fused"), C._assert_finite(cfs, "frozen counterpart")
    return {"fused": per_anchor(fused), "cf": per_anchor(cfs)}


def describe(cell, taus=R.TAUS):
    """Cell number -> (tau index, tau, lambda_u, lambda_a) record (round 2's describe_cell, k_top 13)."""
    if not 0 <= cell < N_CELLS:
        raise ValueError(f"cell {cell} outside the 224")
    return F.describe_cell(cell, taus)
