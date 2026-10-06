"""Unit tests of the round-2 fusion stream (synthetic data, fast). Run from /project/CoSiR:
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python -m pytest src/test/20261118_reader_fix_round2/test_r2_fusion.py -q -p no:cacheprovider
Each guard has a test that fails when the guard is deleted (the mutations are listed in the task report)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r2_common as R  # noqa: E402
import r2_fusion as F  # noqa: E402
import run_r2_fusion as RUN  # noqa: E402

from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS, per_anchor  # noqa: E402
from src.eval.aspect_nested import NESTED_A, NESTED_U, _combine, _zdict  # noqa: E402

K = R.K


# ---------------------------------------------------------------- helpers

def scores(E=30, seed=0, cond_dependent=True, dtype=np.float32):
    rng = np.random.default_rng(seed)
    out = {c: {} for c in CONDITIONS}
    for d in DIRECTIONS:
        base = rng.normal(size=(E, 13)).astype(dtype)
        for c in CONDITIONS:
            out[c][d] = (rng.normal(size=(E, 13)).astype(dtype) if cond_dependent else base.copy())
    return out


def cond_free(E=30, seed=0):
    return scores(E, seed, cond_dependent=False)


def naive_restrict_row(s, b, k):
    """Rule 4.5 item 6 for one 13-score row, in plain python."""
    order = sorted(range(13), key=lambda j: (-float(b[j]), j))           # stable: ties to the lower index
    if k >= 13:
        return [float(x) for x in s]
    inK = order[:k]
    low = min(float(s[j]) for j in inK)
    out = [0.0] * 13
    for p, j in enumerate(order):
        out[j] = float(s[j]) if p < k else low - 1.0 - (p - k)
    return out


def naive_ints(sc):
    """(4*R@1, 4*gain) per episode by definition: 4 rankings (2 conditions x 2 directions), target strictly first."""
    E = len(sc["a"]["i2t"])
    r, g = np.zeros(E, np.int64), np.zeros(E, np.int64)
    for i in range(E):
        for d in DIRECTIONS:
            for cond, tgt, oth in (("a", 0, 1), ("b", 1, 0)):
                row = np.asarray(sc[cond][d][i], np.float64)
                hit = all(row[tgt] > row[j] for j in range(13) if j != tgt)
                other = all(row[oth] > row[j] for j in range(13) if j != oth)
                r[i] += hit
                g[i] += int(hit) - int(other)
    return r, g


def pipeline_inputs(E=30, seed=3):
    B = cond_free(E, seed)
    T = scores(E, seed + 1)
    margins = {c: np.random.default_rng(seed + 2 + j).random(E) for j, c in enumerate(CONDITIONS)}
    taus, _ = K.thresholds(margins)
    zB, zT = _zdict(B), _zdict(T)
    g = K.gates(margins, taus)
    gated = {t: K.gated_terms(zT, g[t]) for t in g}
    G = {t: K.g_cf(gated[t]) for t in g}
    return B, zB, gated, G, F.rank_info(B)


def row_with_sum(n, total, lo=0, hi=4):
    """An int8 vector of length n with entries in [lo, hi] summing to total."""
    v = np.full(n, lo, np.int64)
    rest = total - lo * n
    for i in range(n):
        step = min(hi - lo, rest)
        v[i] += step
        rest -= step
    assert rest == 0
    return v.astype(np.int8)


# ---------------------------------------------------------------- top-k set and restriction

def test_topk_from_B_ties_to_lower_index():
    b = np.zeros((1, 13), np.float32)
    b[0, [1, 2, 3]] = 3.0                      # three-way tie for the top
    b[0, 4] = 2.0
    B = {c: {d: b.copy() for d in DIRECTIONS} for c in CONDITIONS}
    info = F.rank_info(B)
    order = info["i2t"]["order"][0]
    assert list(order[:5]) == [1, 2, 3, 4, 0]                     # ties 1, 2, 3 in index order; then 4; then the zeros
    S = np.arange(13, dtype=np.float32)[None, :]
    r = F.restrict(S, info["i2t"]["pos"], 2)
    assert r[0, 1] == S[0, 1] and r[0, 2] == S[0, 2]              # K = {1, 2}; candidate 3 (tied with them) is outside
    assert r[0, 3] < min(S[0, 1], S[0, 2])


def test_rank_info_asserts_B_condition_free():
    B = scores(10, 1, cond_dependent=True)
    with pytest.raises(AssertionError):
        F.rank_info(B)


@pytest.mark.parametrize("k", [5, 3, 2])
def test_restriction_keeps_S_inside_K_and_B_order_outside(k):
    B, S = cond_free(40, 5), scores(40, 6)
    info = F.rank_info(B)
    for c in CONDITIONS:
        for d in DIRECTIONS:
            s, b = S[c][d], B[c][d]
            r = F.restrict(s, info[d]["pos"], k)
            assert r.dtype == np.float64
            for i in range(len(s)):
                order = sorted(range(13), key=lambda j: (-float(b[i, j]), j))
                inK, outK = order[:k], order[k:]
                assert all(r[i, j] == float(s[i, j]) for j in inK)                   # S kept inside K, exactly
                low = min(float(s[i, j]) for j in inK)
                assert max(r[i, j] for j in outK) <= low - 1.0                        # at least 1 below the lowest in K
                assert all(r[i, outK[p]] > r[i, outK[p + 1]] for p in range(len(outK) - 1))   # B's order outside
                assert list(r[i]) == naive_restrict_row(s[i], b[i], k)


def test_k13_leaves_scores_bit_identical():
    B, S = cond_free(20, 1), scores(20, 2)
    info = F.rank_info(B)
    r = F.restrict_scores(S, info, 13)
    for c in CONDITIONS:
        for d in DIRECTIONS:
            assert r[c][d].dtype == np.float32 and np.array_equal(r[c][d], S[c][d])
            assert r[c][d].tobytes() == S[c][d].tobytes()


def test_restricted_first_place_comes_from_K():
    B, S = cond_free(60, 7), scores(60, 8)
    info = F.rank_info(B)
    r = F.restrict_scores(S, info, 3)
    for c in CONDITIONS:
        for d in DIRECTIONS:
            top = r[c][d].argmax(1)
            assert all(info[d]["pos"][i, top[i]] < 3 for i in range(len(top)))


# ---------------------------------------------------------------- cells

def test_cell_numbering_and_order():
    seq = [(k, t, u, a) for k in F.KTOPS for t in range(4) for u in NESTED_U for a in NESTED_A]
    assert F.KTOPS == (13, 5, 3, 2) and F.N_CELLS == 896 == len(seq)
    for i, (k, t, u, a) in enumerate(seq):
        assert F.cell_values(i) == (k, t, u, a)
        kappa, ti, ui, ai = F.decode_cell(i)
        assert F.cell_number(kappa, ti, ui, ai) == i == ((kappa * 4 + ti) * 7 + ui) * 8 + ai
    assert F.cell_values(0) == (13, 0, 0.0, 0.0) and F.cell_values(223) == (13, 3, 16.0, 16.0)
    assert F.cell_values(224) == (5, 0, 0.0, 0.0) and F.cell_values(895) == (2, 3, 16.0, 16.0)
    for i, (t, u, a) in enumerate(K.rc_cells()):                 # cells 0 to 223 are round 1's, in round 1's order
        assert F.cell_values(i) == (13, t, u, a)
    with pytest.raises(ValueError):
        F.decode_cell(896)


def test_round1_picks_are_cells_116_and_119():
    assert F.cell_values(116)[1:] == (2, 0.0, 2.0) and F.cell_values(119)[1:] == (2, 0.0, 16.0)
    assert F.cell_values(58)[1:] == (1, 0.0, 0.5) and F.cell_values(123)[1:] == (2, 0.5, 1.0)


# ---------------------------------------------------------------- per-cell statistics against a naive reference

def test_cell_statistics_equal_naive_reference():
    B, zB, gated, G, info = pipeline_inputs(E=24, seed=11)
    fri, fgi, cri = F.cell_statistics(zB, info, gated, G)
    assert fri.shape == (896, 24) and fri.dtype == np.int8
    rng = np.random.default_rng(0)
    for i in sorted(set(rng.choice(896, 60, replace=False).tolist() + [0, 223, 224, 500, 895])):
        k, t, lu, la = F.cell_values(i)
        sf = _combine(zB, zB, gated[t], lu, la)
        sc = _combine(zB, zB, G[t], lu, la)
        rf_ = {c: {d: np.array([naive_restrict_row(sf[c][d][j], B[c][d][j], k) for j in range(24)])
                   for d in DIRECTIONS} for c in CONDITIONS}
        rc_ = {c: {d: np.array([naive_restrict_row(sc[c][d][j], B[c][d][j], k) for j in range(24)])
                   for d in DIRECTIONS} for c in CONDITIONS}
        r, g = naive_ints(rf_)
        assert np.array_equal(fri[i], r) and np.array_equal(fgi[i], g), i
        assert np.array_equal(cri[i], naive_ints(rc_)[0]), i


def test_k13_cells_equal_unrestricted_per_anchor():
    B, zB, gated, G, info = pipeline_inputs(E=24, seed=12)
    fri, fgi, cri = F.cell_statistics(zB, info, gated, G, n_kappa=1)
    assert fri.shape == (224, 24)
    for i in (0, 57, 116, 223):
        _, t, lu, la = F.cell_values(i)
        pa = per_anchor(_combine(zB, zB, gated[t], lu, la))
        assert np.array_equal(fri[i] / 4.0, pa["r1"]) and np.array_equal(fgi[i] / 4.0, pa["gain"])


def test_counterpart_is_condition_free_and_K_comes_from_B():
    """The restricted counterpart is identical under both conditions only because K comes from B. This test calls the
    real per-cell code with condition-dependent gates; with K taken from the fused score it raises (or differs)."""
    B, zB, gated, G, info = pipeline_inputs(E=24, seed=13)
    F.cell_statistics(zB, info, gated, G, n_kappa=2)                      # asserts condition-free per cell
    for t in range(4):
        for k in (5, 3, 2):
            sc = F.restrict_scores(_combine(zB, zB, G[t], 0.5, 1.0), info, k)
            for d in DIRECTIONS:
                assert np.array_equal(sc["a"][d], sc["b"][d])
                top_k = np.argsort(info[d]["pos"], axis=1)[:, :k]       # candidates at B's positions 0..k-1
                assert (np.take_along_axis(sc["a"][d], top_k, 1)
                        == np.take_along_axis(_combine(zB, zB, G[t], 0.5, 1.0)["a"][d], top_k, 1)).all()
    # the fused scores differ by condition, so a K built from them would differ by condition
    sf = _combine(zB, zB, gated[3], 0.5, 1.0)
    assert not np.array_equal(sf["a"]["i2t"], sf["b"]["i2t"])
    pos_fa = np.argsort(np.argsort(-sf["a"]["i2t"], axis=1, kind="stable"), axis=1)
    pos_fb = np.argsort(np.argsort(-sf["b"]["i2t"], axis=1, kind="stable"), axis=1)
    assert not np.array_equal(pos_fa < 3, pos_fb < 3)


# ---------------------------------------------------------------- exact integer cross-fit

def ctrl_of(rho0, rho1=0):
    return {0: (0.0, rho0), 1: (0.0, rho1)}


def test_integer_crossfit_ties_go_to_lowest_cell():
    m = 20
    parity = np.array([0] * m + [1] * m)
    fri = np.zeros((12, 2 * m), np.int8)
    fgi = np.zeros((12, 2 * m), np.int8)
    for i in (3, 7, 10):                                           # three exactly tied cells (rho 40, gamma 20)
        fri[i, :m], fgi[i, :m] = row_with_sum(m, 40), row_with_sum(m, 20, -4, 4)
    fri[5, :m], fgi[5, :m] = row_with_sum(m, 39), row_with_sum(m, 20, -4, 4)       # one worse
    assert F.select_fused(fri, fgi, ctrl_of(10), parity)[0] == 3
    cri = fri.copy()
    assert F.select_cf(cri, parity)[0] == 3
    assert F.select_cf(cri, parity, allowed=[7, 10, 5])[0] == 7


def test_integer_criterion_exact_where_float_means_misorder():
    """A (lower cell) and B tie exactly in integers: A's min is its gamma, B's is rho - rho_ctrl. Float means of round 1's
    criterion can order B above A in the last bit; the integer criteria tie and the lower cell wins."""
    found = None
    for m in range(40, 400):
        for rho_c in range(1, 4 * m // 2, 7):
            for d in range(1, 30, 3):
                fa = float(np.float64(d / 4) / m)                                    # A: mean gain = gamma / (4 m)
                fb = float(np.float64((rho_c + d) / 4) / m) - float(np.float64(rho_c / 4) / m)    # B: mean r1 - r_ctrl
                if fb > fa:
                    found = (m, rho_c, d)
                    break
            if found:
                break
        if found:
            break
    assert found is not None
    m, rho_c, d = found
    parity = np.array([0] * m + [1] * m)
    fri = np.zeros((2, 2 * m), np.int8)
    fgi = np.zeros((2, 2 * m), np.int8)
    fri[0, :m], fgi[0, :m] = row_with_sum(m, rho_c + d + 3), row_with_sum(m, d, -4, 4)          # A: rho - rho_c = d + 3 > gamma = d
    fri[1, :m], fgi[1, :m] = row_with_sum(m, rho_c + d), row_with_sum(m, d + 5, -4, 4)          # B: rho - rho_c = d < gamma
    # integers: both criteria are d, a tie, so the lowest cell (0) wins ...
    crit, _, _ = F.fused_criterion(fri, fgi, parity == 0, rho_c, np.arange(2))
    assert list(crit) == [d, d]
    assert F.select_fused(fri, fgi, ctrl_of(rho_c), parity)[0] == 0
    # ... while the float means of round 1's criterion rank B above A
    mean = lambda x: float(np.mean(x[:m].astype(np.float64) / 4.0))                           # noqa: E731
    r_ctrl = rho_c / 4.0 / m
    crit_float = [min(mean(fri[i]) - r_ctrl, mean(fgi[i])) for i in range(2)]
    assert crit_float[1] > crit_float[0]


def test_min_margin_criterion():
    m = 10
    parity = np.array([0] * m + [1] * m)
    fri = np.zeros((3, 2 * m), np.int8)
    fgi = np.zeros((3, 2 * m), np.int8)
    # cell 0: big R@1 gain over the control but small condition gain; cell 1: balanced; cell 2: big gain, no R@1 gain
    fri[0, :m], fgi[0, :m] = row_with_sum(m, 30), row_with_sum(m, 2, -4, 4)
    fri[1, :m], fgi[1, :m] = row_with_sum(m, 22), row_with_sum(m, 12, -4, 4)
    fri[2, :m], fgi[2, :m] = row_with_sum(m, 10), row_with_sum(m, 30, -4, 4)
    crit, rho, gam = F.fused_criterion(fri, fgi, parity == 0, 10, np.arange(3))
    assert list(crit) == [2, 12, 0]                                  # min(rho - 10, gamma)
    assert F.select_fused(fri, fgi, ctrl_of(10), parity)[0] == 1
    # the pick is made on the tune half only: half 1 has nothing, so its criterion ties at 0 -> lowest cell
    assert F.select_fused(fri, fgi, ctrl_of(10), parity)[1] == 0


def test_control_picks_smallest_sigma_among_ties():
    B = cond_free(30, 21)
    zB = _zdict(B)
    parity = np.arange(30) % 2
    ctrl = F.control_choice(zB, parity)
    assert ctrl[0][0] == 0.0 and ctrl[1][0] == 0.0                  # (1 + sigma) z(B) ranks as B for every sigma: all tie
    assert ctrl[0][1] == int(F.int_metrics(_combine(zB, zB, zB, 0.0, 0.0))[0][parity == 0].sum())


def test_as_int4_refuses_non_multiples():
    assert list(F.as_int4(np.array([0.0, 0.25, 1.0, -0.75]))) == [0, 1, 4, -3]
    with pytest.raises(AssertionError):
        F.as_int4(np.array([0.3]))


# ---------------------------------------------------------------- assembly

def test_assembly_applies_each_halfs_pick_to_the_other_half():
    B, zB, gated, G, info = pipeline_inputs(E=30, seed=31)
    parity = np.arange(30) % 2
    picks = {0: F.cell_number(1, 2, 3, 4), 1: F.cell_number(2, 1, 5, 2)}
    out = F.assemble(zB, info, gated, picks, parity)
    for half in (0, 1):
        kappa, t, u, a = F.decode_cell(picks[half])
        s = F.restrict_scores(_combine(zB, zB, gated[t], NESTED_U[u], NESTED_A[a]), info, F.KTOPS[kappa])
        for c in CONDITIONS:
            for d in DIRECTIONS:
                other = parity != half
                assert np.array_equal(out[c][d][other], s[c][d][other])         # half's pick scores the OTHER half
                assert not np.array_equal(out[c][d][parity == half], s[c][d][parity == half])


# ---------------------------------------------------------------- rule SHA and run guards

def test_rule_sha_assertion_refuses_a_modified_rule(tmp_path, monkeypatch):
    R.assert_rule()                                                    # the committed rule passes
    bad = tmp_path / "DECISION_RULE.md"
    bad.write_bytes(R.RULE.read_bytes() + b"\nedited\n")
    monkeypatch.setattr(R, "RULE", bad)
    with pytest.raises(SystemExit):
        R.assert_rule()


def test_candidate_run_refuses_without_a_passed_regression(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "RES", tmp_path)
    with pytest.raises(SystemExit):
        RUN.regression_gate(False)                                     # no file
    (tmp_path / "regression_check.json").write_text(json.dumps({"passed": False,
                                                                "provenance": {"rule_sha256": R.RULE_SHA}}))
    with pytest.raises(SystemExit):
        RUN.regression_gate(False)                                     # failed check
    (tmp_path / "regression_check.json").write_text(json.dumps({"passed": True,
                                                                "provenance": {"rule_sha256": R.RULE_SHA}}))
    RUN.regression_gate(False)                                         # passes
    RUN.regression_gate(True)                                          # smoke is never gated


def _stream_files(tmp_path, r3_is_r1=False, wrong_sha=False, E=8, H=3):
    d = tmp_path / "smoke"
    d.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    P = {c: rng.dirichlet(np.ones(H), size=E) for c in CONDITIONS}
    npz = d / "probs_R2_A0.npz"
    np.savez(npz, P__a=P["a"], P__b=P["b"])
    rec = {"reader": "R2", "config": "A0", "smoke": True, "n_episodes": E, "groupings": ["affect", "image", "caption"],
           "npz_sha256": "0" * 64 if wrong_sha else R.sha_file(npz), "r3_is_r1": r3_is_r1,
           "provenance": {"rule_sha256": R.RULE_SHA}}
    (d / "probs_R2_A0.json").write_text(json.dumps(rec))
    return P


def test_stream_probs_loading_and_refusals(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "RES", tmp_path)
    P = _stream_files(tmp_path)
    got, _ = RUN.load_stream_probs("R2", "A0", True, 8, 3)
    assert all(np.array_equal(got[c], P[c]) for c in CONDITIONS)
    _stream_files(tmp_path, wrong_sha=True)
    with pytest.raises(SystemExit):
        RUN.load_stream_probs("R2", "A0", True, 8, 3)                  # npz differs from its json
    _stream_files(tmp_path, r3_is_r1=True)
    with pytest.raises(SystemExit):
        RUN.load_stream_probs("R2", "A0", True, 8, 3)                  # R3 is R1: no second evaluation
    _stream_files(tmp_path)
    with pytest.raises(SystemExit):
        RUN.load_stream_probs("R2", "A0", True, 9, 3)                  # other episodes


# ---------------------------------------------------------------- rule application (r2_apply_rule.py)

def _row(bar, lo, glo, comparator="counterpart"):
    import r2_apply_rule as A
    rec = {"bar": {"r1": {"point": bar, "ci95": [lo, bar + 0.3]}, "comparator": comparator},
           "gain_statistic": {"point": 2.0, "ci95": [glo, 3.0]}, "margin": {"r1": {"point": 0.3}},
           "r1_means": {"fused": 19.0, "counterpart": 18.5}}
    return A.row_of(rec)


def test_apply_rule_carry_tie_and_kill():
    import r2_apply_rule as A
    # D12 is conjunctive and applied at full precision
    assert _row(0.5, 0.1, 0.1)["clears"] and not _row(0.4999999999, 0.1, 0.1)["clears"]
    assert not _row(0.6, 0.0, 0.1)["clears"] and not _row(0.6, 0.1, 0.0)["clears"]
    r = {"R1": _row(0.60, 0.1, 0.1), "R2": _row(0.64, 0.1, 0.1), "R3": _row(0.66, 0.1, 0.1)}
    d = A.apply_rule(r)
    assert d["eligible"] == ["R1", "R2", "R3"] and d["carried"] == "R2" and d["tied"] == ["R2", "R3"]    # M = 0.66; R1 is 0.06 short
    r["R3"] = _row(0.80, 0.1, 0.1)                                                       # M = 0.80: only R3 within 0.05
    assert A.apply_rule(r)["carried"] == "R3"
    r["R3"] = _row(0.66, 0.1, 0.1)
    r["R1"] = _row(0.61, 0.1, 0.1)                                                       # 0.61 >= 0.66 - 0.05: tied, earliest wins
    assert A.apply_rule(r)["carried"] == "R1"
    r = {"R1": _row(0.1, 0.0, 0.0), "R2": _row(0.2, 0.0, 0.0), "R3": {**_row(0.9, 0.1, 0.1), "identical_to": "R1"}}
    assert A.apply_rule(r)["carried"] is None                                            # R3 = R1 is never carried
    none = A.apply_rule({"R1": _row(0.3, 0.1, 0.1), "R2": None, "R3": _row(0.45, 0.1, 0.1)})
    assert none["carried"] is None and none["ablation_candidate"] == "R3"
    assert none["ablation_label"] == "best development candidate, not carried"
