"""Synthetic self-tests of the re-derivation's own code (no real data): cell numbering of the rule's named cells,
integer counts against src.eval.aspect_metrics.per_anchor on rows with ties and NaN, the one-way decomposition on a
balanced toy design against the textbook formulas, and the gate definitions."""
import numpy as np

import rd3_core as K
import rd3_family as FAM
from src.eval.aspect_metrics import per_anchor


def main():
    named = {116: (2, 0.0, 2.0), 119: (2, 0.0, 16.0), 58: (1, 0.0, 0.5), 123: (2, 0.5, 1.0), 39: (0, 4.0, 16.0),
             149: (2, 4.0, 4.0), 10: (0, 0.5, 0.5)}
    for cid, (t, lu, la) in named.items():
        d = K.cell_desc(cid, K.RULE_TAUS)
        assert (d["tau_index"], d["lambda_u"], d["lambda_a"]) == (t, lu, la), (cid, d)
        assert K.cell_id(t, K.NU.index(lu), K.NA.index(la)) == cid
    assert [K.cell_id(*K.cell_decode(i)) for i in range(K.N_CELLS)] == list(range(K.N_CELLS))

    rng = np.random.default_rng(0)
    E = 4000
    sc = {c: {d: rng.integers(0, 4, size=(E, 13)).astype(np.float32) for d in K.DIRS} for c in K.COND}  # many ties
    sc["a"]["i2t"][::97] = np.nan
    ours, ref = K.metrics(sc), per_anchor(sc)
    for m in K.METRICS:
        assert np.array_equal(ours[m], ref[m]), m

    # balanced one-way design: n0 = m, MS formulas
    P, m = 50, 4
    g = np.repeat(np.arange(P), m)
    a = rng.normal(0, 2.0, P)[g]
    v = a + rng.normal(0, 1.0, P * m)
    s = K.sensitivity(v, g)
    means = v.reshape(P, m).mean(1)
    msb = m * ((means - v.mean()) ** 2).sum() / (P - 1)
    msw = ((v.reshape(P, m) - means[:, None]) ** 2).sum() / (P * (m - 1))
    assert np.isclose(s["n0"], m) and np.isclose(s["ms_between"], msb) and np.isclose(s["ms_within"], msw)
    assert np.isclose(s["sigma_a2"], max(0.0, (msb - msw) / m))
    n = P * m
    se2 = (s["sigma_a2"] * (9 * P * m * m - 6 * n) + msw * 3 * n) / (3 * n) ** 2
    assert np.isclose(s["SE"] ** 2, se2) and np.isclose(s["x"], 2.8 * s["SE"])

    margins = {"a": np.array([0.1, 0.5, 0.9]), "b": np.array([0.2, 0.6, 0.95])}
    picks = {"a": np.array([0, 1, 0]), "b": np.array([0, 0, 2])}
    taus = [0.0, 0.5, 0.9, 1.0]
    g1, ga = FAM.gates_r1(margins, taus), FAM.gates_aff(margins, picks, taus)
    assert g1[1]["a"].tolist() == [False, True, True] and ga[1]["a"].tolist() == [False, False, True]
    assert ga[0]["b"].tolist() == [True, True, False] and g1[3]["b"].tolist() == [False, False, False]
    print("rd3 self-tests passed")


if __name__ == "__main__":
    main()
