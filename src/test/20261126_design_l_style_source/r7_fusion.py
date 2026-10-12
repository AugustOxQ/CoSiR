"""Round 7 scorers: DECISION_RULE.md L11–L13, G1–G3, §5 and §7.3; R3 D5–D9.

Readers use any ordered parts tuple. R3's 224-cell integer selection and parity
cross-fits are unchanged. Reused code is loaded from MAIN through r7_common.
All functions are CPU-only and write nothing.
"""
import r7_common as R
import r7_bundle as B

import pickle
from types import SimpleNamespace
import numpy as np

C, K = (R.load(name) for name in ("common", "rc_core"))
F = R.load("r2_fusion")
per_anchor = R.load("src.eval.aspect_metrics").per_anchor
_zdict = R.load("src.eval.aspect_nested")._zdict
TAUS = R.load("r3_common").TAUS
N_CELLS = F.CELLS_PER_KAPPA
CONDITIONS = ("a", "b")


def _canonical(name):
    """R0 and round 1's csd denote the same head; preserve feature column order."""
    head, sep, feature = name.partition("__")
    return ("R0" if head == "csd" else head) + sep + feature


def reader(bundle, parts, reader_pkl=None):
    """L11/R3 D5: dict of P, expected T, top-two m and argmax pick (ties first).

    reader_pkl is a loaded half-reader dict or a pickle path. Only A0 may default
    to bundle.readers. Frozen A0 stacks/features are used directly when present.
    """
    parts = tuple(parts)
    assert len(parts) >= 2 and len(set(parts)) == len(parts), "parts must be distinct groupings"
    if reader_pkl is None:
        assert parts == B.A0, "a non-A0 configuration requires its own reader pickle"
        reader_pkl = bundle.readers
    if not isinstance(reader_pkl, dict):
        R.assert_threads()
        with open(reader_pkl, "rb") as f:
            reader_pkl = pickle.load(f)
    rf, rbe = (R.load(name) for name in ("rb_features", "rb_eval"))
    pk = reader_pkl
    assert {"feature_names", "groupings", "halves", "sklearn_version"} <= set(pk), "reader pickle keys differ"
    import sklearn
    assert pk["sklearn_version"] == sklearn.__version__, "reader sklearn version differs"
    assert tuple(map(_canonical, pk["groupings"])) == tuple(map(_canonical, parts)), "reader grouping order differs"
    assert list(map(_canonical, pk["feature_names"])) == list(map(_canonical, rf.feature_names(parts))), \
        "the readers were trained on another feature layout"
    assert len(pk["halves"]) == 2, "two half-readers required"
    if parts == B.A0 and hasattr(bundle, "stack") and hasattr(bundle, "F"):
        stack, features = bundle.stack, bundle.F
    else:
        stack, features = B.stack_and_features(bundle, parts)
    for d in ("i2t", "t2i"):
        x = stack[d]
        assert x.shape == (bundle.n, len(parts), 13) and x.dtype == np.float32 and np.isfinite(x).all(), \
            f"stack/{d}: shape, dtype or finite values differ"
    for c in CONDITIONS:
        x = features[c]
        assert x.shape == (bundle.n, 6 * len(parts)) and x.dtype == np.float64 and np.isfinite(x).all(), \
            f"features/{c}: shape, dtype or finite values differ"
    P = {c: rf.average_probs(rbe.half_reader_probs(pk, features[c], len(parts))) for c in CONDITIONS}
    for c in CONDITIONS:
        assert P[c].shape == (bundle.n, len(parts)) and P[c].dtype == np.float64 and np.isfinite(P[c]).all(), \
            f"probabilities/{c}: shape, dtype or finite values differ"
        assert (P[c] >= 0).all() and np.allclose(P[c].sum(axis=1), 1., rtol=0, atol=1e-12), \
            "averaged probabilities do not sum to 1"
    T = C.expected_term(stack, P)
    pm = {c: rf.picks_and_margins(P[c]) for c in CONDITIONS}
    m = {c: pm[c][1] for c in CONDITIONS}
    for c in CONDITIONS:
        assert np.array_equal(m[c], C.top_two_margin(P[c])), "top-two margin differs from shared definition"
    return {"P": P, "T": T, "m": m, "pick": {c: pm[c][0] for c in CONDITIONS}}


def _margins(m):
    assert set(m) == set(CONDITIONS), "margins require both conditions"
    n = len(m["a"])
    assert n > 0, "empty margins"
    for c in CONDITIONS:
        x = np.asarray(m[c])
        assert x.shape == (n,) and x.dtype == np.float64 and np.isfinite(x).all(), \
            f"margin/{c}: shape, float64 dtype or finite values differ"
    return n


def thresholds(m):
    """L12: (four float64 linear percentiles over a then b, number of margins)."""
    _margins(m)
    return K.thresholds(m)


def gates_r1(m, taus):
    """Four float32 gate dicts; equality opens the gate (R3 D6, L13)."""
    _margins(m)
    taus = np.asarray(taus, dtype=np.float64)
    assert taus.shape == (4,) and np.isfinite(taus).all() and np.all(np.diff(taus) >= 0), \
        "taus must be four finite ordered thresholds"
    g = K.gates(m, list(taus))
    return [g[k] for k in range(4)]


def gates_x_aff(m, taus, pick, allowed):
    """L13: R1 gate times 1[pick in allowed], with positional class indices."""
    n = _margins(m)
    allowed = tuple(sorted(allowed))
    assert all(isinstance(i, (int, np.integer)) and i >= 0 for i in allowed), "invalid allowed positions"
    assert set(pick) == set(CONDITIONS), "picks require both conditions"
    for c in CONDITIONS:
        assert np.asarray(pick[c]).shape == (n,) and np.asarray(pick[c]).dtype == np.int64, \
            f"pick/{c}: shape or int64 dtype differs"
    return [{c: (gt[c] * np.isin(pick[c], allowed).astype(np.float32)).astype(np.float32)
             for c in CONDITIONS} for gt in gates_r1(m, taus)]


def _score(bundle, rd, gates, name):
    out = run_family(bundle, rd["T"], gates)
    out.update(reader=rd, gates=gates,
               fused_cells=np.array([out["fpick"][h] for h in (0, 1)], dtype=np.int64),
               cf_cells=np.array([out["cpick"][h] for h in (0, 1)], dtype=np.int64),
               gate_a=np.stack([g["a"] for g in gates]), gate_b=np.stack([g["b"] for g in gates]))
    arrays = {f"{name}{suffix}__{m}": x for who, suffix in (("fused", ""), ("cf", "_cf"))
              for m, x in out[who].items()}
    arrays.update({f"{name}__{k}": out[k] for k in ("fused_cells", "cf_cells", "gate_a", "gate_b")})
    arrays[f"{name}__sigma"] = np.array([out["sigma"][h] for h in (0, 1)], dtype=np.float64)
    out["arrays"] = arrays
    return out


def score_r1(bundle, parts, reader_pkl, taus, *, name=None):
    """L13/§7.3: R1 and matched CF; rerun picks on this seed's parity halves.

    Return R3's run_family dict plus reader, gates, fused_cells/cf_cells (int64),
    gate_a/gate_b (float32) and arrays, ready to update the index's arrays.npz.
    The default array prefix is R1_{parts[-1]}; name may override it.
    """
    parts = tuple(parts)
    rd = reader(bundle, parts, reader_pkl)
    return _score(bundle, rd, gates_r1(rd["m"], taus), name or f"R1_{parts[-1]}")


def score_x_aff(bundle, parts, reader_pkl, taus, allowed=None, *, name=None):
    """L13: extended AFF with allowed {position of affect, position of X}.

    Explicit allowed sets also support G1's {affect} and all-position tests.
    The counterpart retains these same gates; outputs follow score_r1.
    """
    parts = tuple(parts)
    if allowed is None:
        assert len(parts) == 4 and tuple(parts[:3]) == ("affect", "image", "caption"), \
            "default allowed set requires A0 followed by X"
        allowed = {parts.index("affect"), len(parts) - 1}
    allowed = set(allowed)
    assert all(isinstance(i, (int, np.integer)) and 0 <= i < len(parts) for i in allowed), \
        "allowed position outside parts"
    rd = reader(bundle, parts, reader_pkl)
    return _score(bundle, rd, gates_x_aff(rd["m"], taus, rd["pick"], allowed), name or f"XAFF_{parts[-1]}")


def score_aff(bundle):
    """G1: unchanged AFF via X-AFF with A0's frozen reader, R3 taus, {affect}."""
    return score_x_aff(bundle, B.A0, None, TAUS, allowed={B.A0.index("affect")}, name="AFF")


def told(bundle, parts, told_map, posteriors_override=None):
    """G2/§5: one-hot told probabilities, all gates open, the same 224 cells.

    posteriors_override is a partial {grouping: {img, txt}} dict of global-row
    float32 posteriors, NaN outside selection, with the standard head's shape.
    For T*, supply T's true-style one-hots for both sides. The bundle is unchanged.
    Outputs follow score_r1; array prefixes are told_A1, told_T or told_Tstar.
    """
    parts = tuple(parts)
    assert len(parts) >= 2 and len(set(parts)) == len(parts), "parts must be distinct groupings"
    assert set(told_map) == {"emotion", "style", "genre"} and set(told_map.values()) <= set(parts), \
        "told map must map all three aspects into parts"
    source = bundle
    if posteriors_override is not None:
        assert isinstance(posteriors_override, dict) and posteriors_override, "empty posterior override"
        assert set(posteriors_override) <= set(parts), "posterior override outside parts"
        for h, post in posteriors_override.items():
            assert set(post) == {"img", "txt"}, f"{h}: override requires img/txt keys"
            for side in ("img", "txt"):
                assert post[side].shape == bundle.post[h][side].shape, f"{h}/{side}: override class/row shape differs"
        source = SimpleNamespace(**{**vars(bundle), "post": {**bundle.post, **posteriors_override}})
    stack, _ = B.stack_and_features(source, parts)
    pair_index = np.asarray(bundle.pair_index)
    assert pair_index.shape == (bundle.n,) and pair_index.dtype == np.int64 and np.isin(pair_index, (0, 1, 2)).all(), \
        "told pair_index: shape, dtype or pair codes differ"
    pick = C.told_index(parts, told_map, pair_index)
    P = {c: np.eye(len(parts), dtype=np.float64)[pick[c]] for c in CONDITIONS}
    m = {c: C.top_two_margin(P[c]) for c in CONDITIONS}
    taus, n = thresholds(m)
    assert taus == [1.] * 4 and n == 2 * bundle.n, "one-hot told thresholds must be 1"
    gates = gates_r1(m, taus)
    assert all((g[c] == 1).all() for g in gates for c in CONDITIONS), "told gates must all open"
    rd = {"P": P, "T": C.expected_term(stack, P), "m": m, "pick": pick}
    label = "Tstar" if "T" in parts and posteriors_override is not None else (
        "T" if "T" in parts else "A1" if tuple(map(_canonical, parts)) == B.A1 else parts[-1])
    return _score(bundle, rd, gates, f"told_{label}")


# R3 run_family and its cross-fits, copied unchanged.
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


def describe(cell, taus=TAUS):
    """Cell number -> (tau index, tau, lambda_u, lambda_a) record (round 2's describe_cell, k_top 13)."""
    if not 0 <= cell < N_CELLS:
        raise ValueError(f"cell {cell} outside the 224")
    return F.describe_cell(cell, taus)
