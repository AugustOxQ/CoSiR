"""Synthetic checks of the round-4 pieces of rd4_core (no real data): the carry rule (E, M, the integer tie band 24,
the order V4, V2, V24), the four-way bar comparator with ties to the earliest, the gate factors and their asserts,
as_int4, the decoding of the rule's named cells, and the first-grouping tie rule of the picks.
Usage: python rd4_selftest.py
"""
import numpy as np

import rd4_core as K
from rd4_core import K3


def dev(clear, dk):
    return {"D10_clauses": {"clears_bar": clear}, "Delta_k": {"int": dk}}


def main():
    # ---- carry (§5 item 8)
    c = K.carry({"V4": dev(True, 10), "V2": dev(True, 30), "V24": dev(True, 34)})
    assert c["E"] == ["V4", "V2", "V24"] and c["M"] == 34 and c["tied"] == ["V4", "V2", "V24"] and c["carried"] == "V4"
    assert c["gap_exactly_24"] == ["V4"], c
    c = K.carry({"V4": dev(True, 9), "V2": dev(True, 30), "V24": dev(True, 34)})
    assert c["tied"] == ["V2", "V24"] and c["carried"] == "V2", c            # gap 25 is outside the band
    c = K.carry({"V4": dev(False, 100), "V2": dev(True, 0), "V24": dev(True, -3)})
    assert c["E"] == [] and c["kill"] and c["carried"] is None, c          # Delta_k = 0 is not > 0
    c = K.carry({"V4": dev(True, 1), "V2": dev(False, 50), "V24": dev(True, 26)})
    assert c["E"] == ["V4", "V24"] and c["M"] == 26 and c["tied"] == ["V24"] and c["carried"] == "V24", c

    # ---- bar comparator (D8): largest mean R@1, ties to the earliest in the given order
    mk = lambda xs: {"r1": np.asarray(xs, np.float64)}  # noqa: E731
    order = [("Bprime_A1", mk([0.25, 0.5])), ("Bprime_A0", mk([0.5, 0.25])), ("counterpart", mk([0.75, 0.0])),
             ("B", mk([0.0, 0.5]))]
    assert K.bar_comparator(order)[0] == "Bprime_A1"                        # four-way tie at 0.75 -> first
    order[2] = ("counterpart", mk([0.75, 0.25]))
    assert K.bar_comparator(order)[0] == "counterpart"
    order[3] = ("B", mk([0.75, 0.25]))
    assert K.bar_comparator(order)[0] == "counterpart"                      # tie with B -> earlier counterpart
    assert K.bar_comparator(order[1:])[0] == "counterpart"

    # ---- as_int4 (D9)
    assert np.array_equal(K.as_int4([0.0, 0.25, 1.0]), [0, 1, 4])
    try:
        K.as_int4([0.1])
        raise AssertionError("as_int4 accepted 0.1")
    except AssertionError as e:
        assert "multiple of 0.25" in str(e)

    # ---- gate factors (D6)
    E = 6
    base = [{"a": np.array([1, 1, 0, 1, 0, 1], np.float32), "b": np.array([0, 1, 1, 1, 0, 0], np.float32)}
            for _ in range(4)]
    ones = K.factor_ones(E)
    g = K.apply_factors(base, ones, ones)
    assert K.gates_equal(g, base)
    av = K.factor_abstention(np.array([0.0, 0.5, 0.01, 0.03, 0.02, 0.021043562795966864]), K.V75)
    assert np.array_equal(av["a"], [1, 0, 1, 0, 1, 0]) and np.array_equal(av["a"], av["b"])   # v = v75 is not below
    a1 = K.factor_a1_pick({"a": np.array([0, 1, 0, 0, 3, 0]), "b": np.array([2, 0, 0, 1, 0, 0])})
    g4, g2, g24 = K.apply_factors(base, av), K.apply_factors(base, a1), K.apply_factors(base, a1, av)
    for t in range(4):
        for c in K.COND:
            assert np.array_equal(g24[t][c], g4[t][c] * g2[t][c])
            assert g4[t][c].dtype == np.float32
            assert not np.any(g24[t][c][base[t][c] == 0])
    try:
        K.apply_factors([{"a": np.array([2.0], np.float32), "b": np.array([1.0], np.float32)}], K.factor_ones(1))
        raise AssertionError("apply_factors accepted a gate value 2")
    except AssertionError as e:
        assert "0s and 1s" in str(e)
    gr = K.gates_r1({"a": np.array([0.1, 0.5]), "b": np.array([0.2, 0.0])}, [0.0, 0.2, 0.3, 0.6])
    assert np.array_equal(gr[1]["a"], [0, 1]) and np.array_equal(gr[1]["b"], [1, 0]) and gr[0]["a"].dtype == np.float32
    ga = K.gates_aff({"a": np.array([0.1, 0.5]), "b": np.array([0.2, 0.0])}, {"a": np.array([0, 1]), "b": np.array([0, 0])},
                     [0.0, 0.2, 0.3, 0.6])
    assert np.array_equal(ga[0]["a"], [1, 0]) and np.array_equal(ga[0]["b"], [1, 1])

    # ---- the rule's named cells decode to the rule's settings
    named = {116: (2, 0.0, 2.0), 119: (2, 0.0, 16.0), 58: (1, 0.0, 0.5), 123: (2, 0.5, 1.0), 39: (0, 4.0, 16.0),
             149: (2, 4.0, 4.0), 10: (0, 0.5, 0.5), 117: (2, 0.0, 4.0), 67: (1, 0.5, 1.0)}
    for cid, (t, lu, la) in named.items():
        tt, u, a = K3.cell_decode(cid)
        assert (tt, K3.NU[u], K3.NA[a]) == (t, lu, la), (cid, tt, u, a)
        assert K3.cell_id(tt, u, a) == cid

    # ---- tie rule of the picks: numpy.argmax takes the first grouping; exact ties counted
    P = np.array([[0.4, 0.4, 0.1, 0.1], [0.1, 0.3, 0.3, 0.3], [0.7, 0.1, 0.1, 0.1]])
    assert np.array_equal(np.argmax(P, axis=1), [0, 1, 0]) and K.exact_argmax_ties(P) == 2
    print("rd4_selftest: all checks passed")


if __name__ == "__main__":
    main()
