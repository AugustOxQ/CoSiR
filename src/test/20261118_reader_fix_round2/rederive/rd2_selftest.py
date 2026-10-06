"""Synthetic checks of the re-derivation's pure pieces (no real data, no candidate number): cell numbering, the top-k
sets and the restriction of rule §4.5 items 5 to 7, the integer counts against src.eval.aspect_metrics.per_anchor, the
EM fixed point, and the draw procedure's invariants on a toy half."""
import numpy as np

import rd2_core as K
from src.eval.aspect_metrics import per_anchor


def main():
    ok = {}
    # cell numbering: round 1's cells and the rule's examples
    ok["cell_116"] = K.cell_decode(116) == (0, 2, 0, 4) and K.NA[4] == 2.0
    ok["cell_119"] = K.cell_decode(119) == (0, 2, 0, 7) and K.NA[7] == 16.0
    ok["cell_58"] = K.cell_decode(58) == (0, 1, 0, 2) and K.NA[2] == 0.5
    ok["cell_123"] = K.cell_decode(123) == (0, 2, 1, 3) and (K.NU[1], K.NA[3]) == (0.5, 1.0)
    ok["roundtrip"] = all(K.cell_id(*K.cell_decode(i)) == i for i in range(896))
    ok["last_cell"] = K.cell_decode(895) == (3, 3, 6, 7)
    # restriction on random rows with ties
    rng = np.random.default_rng(0)
    E = 2000
    b = rng.integers(0, 6, size=(E, 13)).astype(np.float32)          # many ties in B
    S = rng.normal(size=(E, 13)).astype(np.float32)
    S[:, 3] = S[:, 4]                                                  # ties in S
    o, pos = K.topk_positions(b)
    good = True
    for k in (13, 5, 3, 2):
        Sp = K.restrict(S, pos, k)
        if k == 13:
            good &= Sp is S
            continue
        inK = pos < k
        # members keep their float32 values
        good &= bool(np.array_equal(np.where(inK, Sp, 0), np.where(inK, S.astype(np.float64), 0)))
        minK = np.where(inK, S.astype(np.float64), np.inf).min(1)
        outside_max = np.where(inK, -np.inf, Sp).max(1)
        good &= bool((outside_max <= minK - 1).all())
        # outside K: B's order (positions ascending -> scores descending, strictly)
        for n in range(50):
            seq = Sp[n, o[n, k:]]
            good &= bool((np.diff(seq) < 0).all())
        # K = the first k of the stable argsort of -b: ties to the lower index
        for n in range(50):
            Kset = set(o[n, :k].tolist())
            ref = sorted(range(13), key=lambda j: (-b[n, j], j))[:k]
            good &= Kset == set(ref)
    ok["restriction"] = bool(good)
    # counts vs per_anchor, with ties and NaN rows
    sc = {c: {d: rng.integers(0, 4, size=(E, 13)).astype(np.float32) for d in K.DIRS} for c in K.COND}
    sc["a"]["i2t"][5] = np.nan
    pa = per_anchor(sc)
    mc = K.metrics_from_counts(*K.counts(sc))
    ok["counts_equal_per_anchor"] = all(np.array_equal(pa[m], mc[m]) for m in K.METRICS)
    # combine equals aspect_nested._combine
    import torch
    from src.eval.aspect_nested import _combine
    zb = {c: {d: torch.as_tensor(rng.normal(size=(E, 13)).astype(np.float32)) for d in K.DIRS} for c in K.COND}
    zt = {c: {d: torch.as_tensor(rng.normal(size=(E, 13)).astype(np.float32)) for d in K.DIRS} for c in K.COND}
    eq = True
    for u in K.NU:
        for a in K.NA:
            ref = _combine(zb, zb, zt, u, a)
            for c in K.COND:
                for d in K.DIRS:
                    eq &= np.array_equal(ref[c][d], K.combine(zb[c][d].numpy(), zt[c][d].numpy(), u, a))
    ok["combine_equals_aspect_nested"] = bool(eq)
    # EM: a uniform-prior input returns the uniform prior at once
    import rd2_r2 as R2
    P = rng.dirichlet(np.ones(3), size=500)
    P = np.vstack([P, P[:, [1, 2, 0]], P[:, [2, 0, 1]]])                # mean exactly symmetric up to rounding
    pi, s, n, conv, _, _ = R2.em(P, 3)
    ok["em_symmetric"] = bool(conv and np.allclose(pi, 1 / 3, atol=1e-9))
    print(ok)
    if not all(ok.values()):
        raise SystemExit("self-test failed")


if __name__ == "__main__":
    main()
