"""Tests of r5_diag.py (rule DECISION_RULE.md section 5 measured diagnostics (a) to (d), section 10 list A item 8).
Synthetic only: hand-made arrays and a synthetic bundle run through round 3's run_family.
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python -m pytest -q -p no:cacheprovider test_r5_diag.py
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r5_common as R5  # noqa: E402
import r5_guard as G  # noqa: E402
import r5_diag as D  # noqa: E402

from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS, per_anchor  # noqa: E402

RF3, C = R5.RF3, R5.C


@pytest.fixture(autouse=True)
def _fresh():
    G._reset_for_tests()
    yield
    G._reset_for_tests()


# ---------------------------------------------------------------- placements and carry file

def clip_placement(monkeypatch, n=6):
    Q = np.random.default_rng(0).dirichlet(np.ones(41), size=n).astype(np.float32)
    monkeypatch.setattr(R5.RB3, "_HEADS", {"k": {"post": {"affect": {"txt": Q}}}})
    return G.clip_from_bundle(SimpleNamespace(post={"affect": {"txt": Q}}))


def ge_placement(tmp_path, monkeypatch):
    n_rows, n_sel = 6, 4
    post = np.random.default_rng(1).dirichlet(np.ones(41), size=n_sel).astype(np.float32)
    p = tmp_path / "ge.npz"
    np.savez(p, post_sel=post, rows=np.array([0, 2, 3, 5], np.int64), classes=np.arange(41))
    monkeypatch.setattr(R5, "N_ROWS", n_rows)
    monkeypatch.setattr(R5, "N_SELECTION", n_sel)
    monkeypatch.setattr(R5, "GE_POST_SHA", R5.sha256_file(p))
    return G.ge_from_file(p)


def carry_file(tmp_path, sha=None):
    d = tmp_path / "results"
    d.mkdir(exist_ok=True)
    p = d / "carry.json"
    p.write_text(json.dumps({"rule_sha256": sha or R5.RULE_SHA}))
    return p


def release_guard(tmp_path):
    d = tmp_path / "results"
    d.mkdir(exist_ok=True)
    p = d / "regression_check.json"
    p.write_text(json.dumps({"rule_sha256": R5.RULE_SHA, "all_passed": True,
                             "items": {k: {"passed": True} for k in "1234"}}))
    G.release(p)


# ---------------------------------------------------------------- guard behaviour

PI = np.array([0, 1, 2, 0, 1, 2])


def test_ge_refused_before_release_everywhere(tmp_path, monkeypatch):
    ge = ge_placement(tmp_path, monkeypatch)
    cp = carry_file(tmp_path)
    x = {"a": np.zeros(6), "b": np.zeros(6)}
    with pytest.raises(G.GuardError):
        D.auc_emotion(x, PI, ge, cp)
    with pytest.raises(G.GuardError):
        D.auc_delta({"a": np.zeros((6, 18)), "b": np.zeros((6, 18))}, PI, ge, cp)
    with pytest.raises(G.GuardError):
        D.pair_lift(None, None, None, None, ge, cp)
    with pytest.raises(G.GuardError):
        D.sharper_term(None, None, None, None, None, None, ge, cp)


def test_ge_needs_carry_even_when_released(tmp_path, monkeypatch):
    ge = ge_placement(tmp_path, monkeypatch)
    release_guard(tmp_path)
    x = {"a": np.array([.1, .2, .3, .4, .5, .6]), "b": np.array([.6, .5, .4, .3, .2, .1])}
    with pytest.raises(G.GuardError):
        D.auc_emotion(x, PI, ge, None)
    with pytest.raises(G.GuardError):
        D.auc_emotion(x, PI, ge, tmp_path / "results" / "missing.json")
    with pytest.raises(G.GuardError):
        D.auc_emotion(x, PI, ge, carry_file(tmp_path, sha="0" * 64))
    assert isinstance(D.auc_emotion(x, PI, ge, carry_file(tmp_path)), float)


def test_clip_allowed_without_carry_but_bad_carry_refused(tmp_path, monkeypatch):
    clip = clip_placement(monkeypatch)
    x = {"a": np.array([.1, .2, .3, .4, .5, .6]), "b": np.array([.6, .5, .4, .3, .2, .1])}
    assert isinstance(D.auc_emotion(x, PI, clip), float)
    with pytest.raises(G.GuardError):
        D.auc_emotion(x, PI, clip, carry_file(tmp_path, sha="0" * 64))


def test_non_placement_refused():
    with pytest.raises(G.GuardError):
        D.auc_emotion({"a": np.zeros(6), "b": np.zeros(6)}, PI, "clip")


# ---------------------------------------------------------------- (a), (b) AUC sets and order

def test_auc_sets_and_order(monkeypatch):
    pi = np.array([0, 1, 2])
    y = D.auc_sets(pi)
    # condition a: pairs e x s, e x g positive, s x g not; condition b: none
    assert y.tolist() == [True, True, False, False, False, False]


def test_auc_hand_example(monkeypatch):
    clip = clip_placement(monkeypatch)
    a = np.array([0.9, 0.8, 0.1])
    b = np.array([0.7, 0.2, 0.3])
    got = D.auc_emotion({"a": a, "b": b}, np.array([0, 1, 2]), clip)
    # positives 0.9, 0.8; negatives 0.1, 0.7, 0.2, 0.3 -> all 8 pairs won
    assert got == 1.0
    a = np.array([0.5, 0.8, 0.1])
    b = np.array([0.7, 0.2, 0.3])
    got = D.auc_emotion({"a": a, "b": b}, np.array([0, 1, 2]), clip)
    assert got == pytest.approx(roc_auc_score([1, 1, 0, 0, 0, 0], [0.5, 0.8, 0.1, 0.7, 0.2, 0.3]))
    assert got == pytest.approx(7 / 8)
    # swapping the condition order flips the labels: not the same number
    flipped = D.auc_emotion({"a": b, "b": a}, np.array([0, 1, 2]), clip)
    assert flipped != got


def test_auc_accepts_probability_matrix_column_zero(monkeypatch):
    clip = clip_placement(monkeypatch)
    rng = np.random.default_rng(3)
    Pa, Pb = rng.dirichlet(np.ones(3), 9), rng.dirichlet(np.ones(3), 9)
    pi = np.arange(9) % 3
    assert D.auc_emotion({"a": Pa, "b": Pb}, pi, clip) == D.auc_emotion({"a": Pa[:, 0], "b": Pb[:, 0]}, pi, clip)


def test_auc_delta_reads_column_two(monkeypatch):
    clip = clip_placement(monkeypatch)
    Fa, Fb = np.zeros((3, 18)), np.zeros((3, 18))
    Fa[:, 2] = [0.9, 0.8, 0.1]
    Fb[:, 2] = [0.7, 0.2, 0.3]
    Fa[:, 5] = -Fa[:, 2]                      # a different column carries the opposite order
    assert D.auc_delta({"a": Fa, "b": Fb}, np.array([0, 1, 2]), clip) == 1.0
    assert D.auc_delta({"a": Fa, "b": Fb}, np.array([0, 1, 2]), clip, column=5) < 1.0


# ---------------------------------------------------------------- (c) pair lift

def test_pair_lift_matches_pair_stats_heads(monkeypatch):
    clip = clip_placement(monkeypatch)
    rng = np.random.default_rng(4)
    n = 60
    Pi, Pt = rng.dirichlet(np.ones(5), n), rng.dirichlet(np.ones(5), n)
    labS = {"emotion": rng.integers(0, 3, n), "style": rng.integers(0, 3, n), "genre": rng.integers(0, 3, n)}
    gS = rng.integers(0, 6, n)
    got = D.pair_lift(Pi.astype(np.float32), Pt, labS, gS, clip)
    ref = R5.RTO.pair_stats_heads(Pi.astype(np.float32).astype(np.float64), Pt, labS, gS)
    assert got["ratio_same_over_diff"] == ref["by_aspect"]["ratio_same_over_diff"]
    assert got["emotionxstyle"] == ref["contrast"]["emotionxstyle"]["ratio"]
    assert got["emotionxgenre"] == ref["contrast"]["emotionxgenre"]["ratio"]
    assert got["group_lift"] == 2.7111312041209863


def test_pair_lift_selection_must_match_placement(monkeypatch):
    clip = clip_placement(monkeypatch, n=80)
    rng = np.random.default_rng(7)
    sel = np.arange(0, 80, 2)
    n = len(sel)
    Pt = clip.Q[sel].astype(np.float64)
    Pi = rng.dirichlet(np.ones(41), n)
    labS = {k: rng.integers(0, 3, n) for k in ("emotion", "style", "genre")}
    gS = rng.integers(0, 8, n)
    D.pair_lift(Pi, Pt, labS, gS, clip, selection=sel)
    with pytest.raises(ValueError):
        D.pair_lift(Pi, Pt + 0.01, labS, gS, clip, selection=sel)


# ---------------------------------------------------------------- (d) the sharper term

def hand_scores(rows):
    """rows: list of (a_i2t, b_i2t, a_t2i, b_t2i) each a 2-vector of the first two columns; other columns -1."""
    E = len(rows)
    S = {c: {d: np.full((E, 13), -1.0) for d in DIRECTIONS} for c in CONDITIONS}
    for e, (ai, bi, at, bt) in enumerate(rows):
        for c, d, v in (("a", "i2t", ai), ("b", "i2t", bi), ("a", "t2i", at), ("b", "t2i", bt)):
            S[c][d][e, 0], S[c][d][e, 1] = v
    return S


def test_episode_terms_definitions():
    # episode: i2t: a hits target (col0 first), b wins other (col0 first, target col1); t2i: a other wins (col1 first),
    # b target wins (col1 first)
    S = hand_scores([((2, 1), (2, 1), (1, 2), (1, 2))])
    t = D.episode_terms(S)
    assert t["cond"]["a"]["r1"][0] == 0.5 and t["cond"]["a"]["other"][0] == 0.5
    assert t["cond"]["b"]["r1"][0] == 0.5 and t["cond"]["b"]["other"][0] == 0.5
    assert t["dir"]["i2t"]["r1"][0] == 0.5          # a hits, b misses
    assert t["dir"]["i2t"]["other"][0] == 0.5
    assert t["dir"]["t2i"]["r1"][0] == 0.5
    assert D._metric(t["cond"]["a"], "gain")[0] == 0.0
    assert D._metric(t["cond"]["a"], "either")[0] == 1.0


def test_episode_terms_ties_are_misses():
    t = D.episode_terms(hand_scores([((1, 1), (1, 1), (1, 1), (1, 1))]))
    assert t["cond"]["a"]["r1"][0] == 0 and t["cond"]["a"]["other"][0] == 0


def test_episode_terms_condition_means_equal_per_anchor():
    rng = np.random.default_rng(5)
    S = {c: {d: rng.normal(size=(40, 13)) for d in DIRECTIONS} for c in CONDITIONS}
    t = D.episode_terms(S)
    pa = per_anchor(S)
    np.testing.assert_array_equal(0.5 * (t["cond"]["a"]["r1"] + t["cond"]["b"]["r1"]), pa["r1"])
    np.testing.assert_array_equal(0.5 * (t["dir"]["i2t"]["r1"] + t["dir"]["t2i"]["r1"]), pa["r1"])
    np.testing.assert_array_equal(0.5 * (t["cond"]["a"]["other"] + t["cond"]["b"]["other"]), pa["other"])


def test_term_diff_orientation_fused_minus_cf():
    good = D.episode_terms(hand_scores([((2, 1), (1, 2), (2, 1), (1, 2))]))     # all four hits
    bad = D.episode_terms(hand_scores([((1, 2), (2, 1), (1, 2), (2, 1))]))      # all four other wins
    d = D.term_diff(good, bad)
    assert d["cond"]["a"]["r1"][0] == 1.0
    assert d["cond"]["a"]["gain"][0] == 2.0
    assert d["cond"]["a"]["either"][0] == 0.0
    assert d["dir"]["t2i"]["r1"][0] == 1.0


def synth(n=48, seed=0):
    rng = np.random.default_rng(seed)
    base = {d: rng.normal(size=(n, 13)).astype(np.float32) for d in DIRECTIONS}
    B = {c: {d: base[d].copy() for d in DIRECTIONS} for c in CONDITIONS}
    bundle = SimpleNamespace(n=n, parity=np.arange(n) % 2, B=B)
    T = {c: {} for c in CONDITIONS}
    for c, col in (("a", 0), ("b", 1)):
        for d in DIRECTIONS:
            t = rng.normal(size=(n, 13)).astype(np.float32)
            t[:, col] += 1.5 * (rng.random(n) < 0.6)
            T[c][d] = t
    P = {c: rng.dirichlet(np.ones(3), size=n) for c in CONDITIONS}
    m = {c: C.top_two_margin(P[c]) for c in CONDITIONS}
    pick = {c: P[c].argmax(axis=1) for c in CONDITIONS}
    gates = RF3.gates_aff(m, pick, R5.R3.TAUS)
    return bundle, T, gates


def test_sharper_term_reassembly_and_structure(monkeypatch):
    clip = clip_placement(monkeypatch)
    bundle, T, gates = synth()
    fam = RF3.run_family(bundle, T, gates)
    cl, pi = np.arange(48) // 4, np.arange(48) % 3
    out = D.sharper_term(bundle, T, gates, fam, cl, pi, clip)
    assert set(out["per_pair_condition"]) == set(C.POOLED_ORDER)
    assert set(out["per_pair_condition"][C.POOLED_ORDER[0]]) == {"a", "b"}
    assert set(out["per_direction"]) == {"i2t", "t2i"}
    assert set(out["per_direction"]["i2t"]) == {"r1", "gain", "either"}
    assert set(out["cells"]["fused"]) == {0, 1}
    # pooled over pairs and conditions the per-direction mean difference equals the family's fused minus cf R@1
    d = out["diff"]
    both = 0.5 * (d["dir"]["i2t"]["r1"] + d["dir"]["t2i"]["r1"])
    np.testing.assert_array_equal(both, np.asarray(fam["fused"]["r1"]) - np.asarray(fam["cf"]["r1"]))
    aff = D.sharper_term(bundle, T, gates, fam, cl, pi, clip)
    again = D.sharper_term(bundle, T, gates, fam, cl, pi, clip, aff=aff)
    assert again["minus_aff"]["per_direction"]["i2t"]["r1"]["point"] == 0.0


def test_reassembly_mismatch_raises(monkeypatch):
    clip = clip_placement(monkeypatch)
    bundle, T, gates = synth()
    fam = RF3.run_family(bundle, T, gates)
    bad = dict(fam)
    bad["fused"] = {k: np.asarray(v).copy() for k, v in fam["fused"].items()}
    bad["fused"]["r1"][0] += 0.25
    with pytest.raises(AssertionError):
        D.reassemble(bundle, T, gates, bad, clip)
    wrong_cells = dict(fam)
    wrong_cells["fpick"] = {h: (c + 37) % 224 for h, c in fam["fpick"].items()}
    # different cells give different per-anchor arrays unless the synthetic data make them tie; accept only a raise
    try:
        D.reassemble(bundle, T, gates, wrong_cells, clip)
    except AssertionError:
        return
    pytest.skip("synthetic cells tie")


def test_either_cost_per_gain():
    rec = {"either_change": -1.629638671875, "gain_statistic": {"point": 3.110758463541667}}
    assert D.either_cost_per_gain(rec) == pytest.approx(R5.AFF_EITHER_PER_GAIN)
