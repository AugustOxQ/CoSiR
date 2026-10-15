"""Round 5 re-derivation: the placement extension (D5) and the candidates G-T and G-TF (D6, D7) for ANY placement Q.

Stage A runs this with Q = Q_CLIP only (item 4: it must reproduce AFF exactly). Stage B runs it with Q_GE after the
stage-A targets and the GoEmotions spot check have matched (the guard below)."""
from types import SimpleNamespace

import numpy as np

import rd5_core as core
from rd5_bundle import b_prime
from rd5_placement import fingerprint, swap_caption_side

COMPARATOR_ORDER = ("B'_Q", "B'(A0)", "counterpart", "B")      # D8 tie order (B'_Q is B'_G when Q = Q_GE)


class Guard:
    """Refuses a non-CLIP placement until stage A's targets and the spot check have been recorded as matched."""
    released = False

    @classmethod
    def check(cls, kind):
        if kind not in ("clip", "ge"):
            raise AssertionError(f"unknown placement kind {kind!r}")
        if kind == "ge" and not cls.released:
            raise AssertionError("GE placement refused: stage-A targets and the spot check have not passed")


def extend(bundle, Q, kind, A) -> SimpleNamespace:
    """post_Q (a new dict), stack_Q, F_Q and B'_Q; asserts the image and caption parts unchanged and the bundle's
    post untouched (identity and value fingerprints before and after)."""
    Guard.check(kind)
    n_total = len(bundle.ctx.groups)
    if Q.shape != (n_total, 41) or Q.dtype != np.float32:
        raise AssertionError("placement must be float32 (N, 41)")
    sel, in_sel = bundle.ctx.selection, bundle.ctx.in_sel
    if not (np.isfinite(Q[sel]).all() and np.isnan(Q[~in_sel]).all()):
        raise AssertionError("placement must be finite on selection rows and NaN elsewhere")
    if kind == "clip" and not (Q is bundle.post["affect"]["txt"]
                               or np.array_equal(Q, bundle.post["affect"]["txt"], equal_nan=True)):
        raise AssertionError("a clip placement must be the bundle's own Q_CLIP")
    before = fingerprint(bundle.post)
    post_Q = swap_caption_side(bundle.post, Q)
    if post_Q is bundle.post or post_Q["affect"] is bundle.post["affect"]:
        raise AssertionError("post_Q must be a new dict")
    stack_Q = core.grouping_stack(post_Q, bundle.ep, core.A0)
    F_Q = core.reader_features(post_Q, bundle.ep, core.A0)
    for d in core.DIRECTIONS:
        if not np.array_equal(stack_Q[d][:, 1:], bundle.stack[d][:, 1:]):
            raise AssertionError("image/caption slices of stack_Q differ from the bundle's")
    for c in core.CONDITIONS:
        if not np.array_equal(F_Q[c][:, 6:], bundle.F[c][:, 6:]):
            raise AssertionError("feature columns 6..17 of F_Q differ from the bundle's")
    if not np.array_equal(F_Q["b"][:, 2], -F_Q["a"][:, 2]):
        raise AssertionError("Delta^b != -Delta^a on F_Q")
    Bq, pBq = b_prime(bundle, post_Q, A)
    if fingerprint(bundle.post) != before:
        raise AssertionError("the bundle's post changed during the extension")
    return SimpleNamespace(kind=kind, Q=Q, post=post_Q, stack=stack_Q, F=F_Q, B=Bq, pB=pBq)


def candidates(bundle, ext, aff_reader, taus, A) -> dict:
    """G-T: AFF's P, m, pi and gates, T on stack_Q. G-TF: the A0 half-readers on F_Q, T on stack_Q, tau' from its own
    margins (condition a first), AFF's gate form at tau'. Returns the readers, gates, tau' and the families."""
    Guard.check(ext.kind)
    out = {}
    rd_t = {c: {"P": aff_reader[c]["P"], "m": aff_reader[c]["m"], "pi": aff_reader[c]["pi"],
                "T": {d: core.expected_term(aff_reader[c]["P"], ext.stack[d]) for d in core.DIRECTIONS}}
            for c in core.CONDITIONS}
    g_t = {c: core.gates_aff(rd_t[c]["m"], rd_t[c]["pi"], taus) for c in core.CONDITIONS}
    out["G-T"] = {"reader": rd_t, "gates": g_t, "taus": np.asarray(taus, np.float64)}
    rd_tf = core.reader(ext.F, ext.stack, bundle.halves)
    tau_p = core.taus_from_margins(rd_tf["a"]["m"], rd_tf["b"]["m"])
    g_tf = {c: core.gates_aff(rd_tf[c]["m"], rd_tf[c]["pi"], tau_p) for c in core.CONDITIONS}
    out["G-TF"] = {"reader": rd_tf, "gates": g_tf, "taus": tau_p}
    for name, v in out.items():
        T = {c: v["reader"][c]["T"] for c in core.CONDITIONS}
        v["family"] = core.run_family(bundle.B, T, v["gates"], bundle.parity, A["zscore_rows"])
        v["open_tau0"] = {c: int(v["gates"][c][0].sum()) for c in core.CONDITIONS}
    return out


def comparators(ext, bundle, fam) -> list:
    return [("B'_Q", ext.pB), ("B'(A0)", bundle.pBp0), ("counterpart", fam.pa_cf), ("B", bundle.pB)]
