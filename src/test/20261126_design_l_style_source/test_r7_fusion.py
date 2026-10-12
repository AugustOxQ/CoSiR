"""Ticket 08: L12/L13 gates, R3 integer cross-fits, G1/G2/G3 and the §5 told route.

Seed 42 is used only with the frozen A0/A1 objects for stage-0 regressions.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r7_common as R  # noqa: E402
import r7_bundle as B  # noqa: E402
import r7_fusion as F  # noqa: E402
from test_r7_bundle import frozen_heads, seed42  # noqa: E402, F401


def test_thresholds_pool_both_conditions_with_linear_percentiles():
    taus, n = F.thresholds({"a": np.array([0., .2, .4, .6]), "b": np.array([.8, 1., .8, .4])})
    assert n == 8
    assert taus == pytest.approx([0., .35, .5, .8], abs=1e-15)


def test_gates_include_equality_and_close_disallowed_picks():
    m = {c: np.array([0., .25, .5, .75, 1.]) for c in ("a", "b")}
    pick = {"a": np.array([0, 1, 2, 3, 0]), "b": np.array([3, 2, 1, 0, 3])}
    taus = [0., .25, .5, .75]
    r1 = F.gates_r1(m, taus)
    xa = F.gates_x_aff(m, taus, pick, {0, 3})
    all_allowed = F.gates_x_aff(m, taus, pick, {0, 1, 2, 3})
    for k in range(4):
        for c in ("a", "b"):
            assert np.array_equal(r1[k][c], [int(i >= k) for i in range(5)])
            assert np.array_equal(all_allowed[k][c], r1[k][c])
            assert xa[k][c].dtype == np.float32
    assert np.array_equal(xa[0]["a"], [1, 0, 0, 1, 1])
    assert np.array_equal(xa[2]["b"], [0, 0, 0, 1, 1])


def synthetic_bundle():
    """Noncontiguous global rows and a fourth head opposed to the other three."""
    n = 6
    selection = np.array([1, 3, 5, 7], dtype=np.int64)
    p = np.full((8, 2), np.nan, dtype=np.float32)
    p[selection] = [[1, 0], [1, 0], [0, 1], [0, 1]]
    ep = SimpleNamespace(anchor=np.full(n, 1), candidates=np.tile([3, 5, 7, 3, 5, 7, 3, 5, 7, 3, 5, 7, 3], (n, 1)),
                         pairs_a_img=np.full((n, 4), 1), pairs_a_txt=np.full((n, 4), 3),
                         pairs_b_img=np.full((n, 4), 1), pairs_b_txt=np.full((n, 4), 5))
    ep.condition = lambda c: ((ep.pairs_a_img, ep.pairs_a_txt, ep.pairs_b_img, ep.pairs_b_txt, None)
                              if c == "a" else
                              (ep.pairs_b_img, ep.pairs_b_txt, ep.pairs_a_img, ep.pairs_a_txt, None))
    post = {h: {"img": p.copy(), "txt": p.copy()} for h in B.A0 + ("V1",)}
    post["V1"]["txt"][selection] = [[0, 1], [0, 1], [1, 0], [1, 0]]
    zero = {c: {d: np.zeros((n, 13), dtype=np.float32) for d in ("i2t", "t2i")} for c in ("a", "b")}
    return SimpleNamespace(n=n, ctx=SimpleNamespace(pooled=ep, groups=np.zeros(8), selection=selection),
                           post=post, B=zero, parity=np.arange(n) % 2,
                           pair_index=np.repeat(np.arange(3), 2))


def uniform_reader(parts):
    """A real balanced logistic regression gives tied probabilities on every feature row."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import sklearn
    X = np.zeros((2 * len(parts), 6 * len(parts)))
    scaler = StandardScaler().fit(X)
    model = LogisticRegression().fit(scaler.transform(X), np.tile(np.arange(len(parts)), 2))
    return {"feature_names": R.load("rb_features").feature_names(parts), "groupings": list(parts),
            "halves": [{"scaler": scaler, "model": model}] * 2, "sklearn_version": sklearn.__version__}


@pytest.mark.parametrize("parts", [B.A0, B.A0 + ("V1",), ("V1", "caption", "affect", "image")])
def test_reader_expected_term_and_first_grouping_ties(parts):
    bundle = synthetic_bundle()
    out = F.reader(bundle, parts, uniform_reader(parts))
    same = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1])
    expected = same if len(parts) == 3 else .25 + .5 * same
    for c in ("a", "b"):
        assert np.array_equal(out["P"][c], np.full((bundle.n, len(parts)), 1 / len(parts)))
        assert np.array_equal(out["pick"][c], np.zeros(bundle.n, dtype=np.int64))
        assert np.array_equal(out["m"][c], np.zeros(bundle.n))
        for d in ("i2t", "t2i"):
            assert np.array_equal(out["T"][c][d], np.tile(expected, (bundle.n, 1)))
            assert out["T"][c][d].dtype == np.float32


def test_reader_rejects_wrong_feature_order():
    parts = B.A0 + ("V1",)
    pk = uniform_reader(parts)
    pk["feature_names"] = pk["feature_names"][::-1]
    with pytest.raises(AssertionError, match="feature"):
        F.reader(synthetic_bundle(), parts, pk)


def test_family_ties_choose_first_cells_and_smallest_sigma():
    b = synthetic_bundle()
    out = F.run_family(b, b.B, [{c: np.ones(b.n, np.float32) for c in ("a", "b")}] * 4)
    assert out["fpick"] == out["cpick"] == {0: 0, 1: 0}
    assert out["sigma"] == {0: 0., 1: 0.}
    assert all(np.array_equal(x, np.zeros(b.n)) for x in out["fused"].values())
    assert all(np.array_equal(x, np.zeros(b.n)) for x in out["cf"].values())


def test_family_tunes_on_each_half_and_applies_its_pick_to_the_other_half():
    b = synthetic_bundle()
    # Correct on even rows, swapped on odd rows. B is tied everywhere.
    terms = {c: {d: np.zeros((b.n, 13), dtype=np.float32) for d in ("i2t", "t2i")}
             for c in ("a", "b")}
    for c, target in (("a", 0), ("b", 1)):
        for d in ("i2t", "t2i"):
            terms[c][d][b.parity == 0, target] = 1
            terms[c][d][b.parity == 1, 1 - target] = 1
    gates = [{c: np.ones(b.n, np.float32) for c in ("a", "b")}] * 4
    out = F.run_family(b, terms, gates)
    # Cell 1 is the first positive lambda_a. All positive cells tie on the even half.
    assert out["fpick"] == {0: 1, 1: 0}
    assert out["cpick"] == {0: 0, 1: 0}
    assert np.array_equal(out["fused"]["r1"], np.zeros(b.n))
    assert np.array_equal(out["fused"]["gain"], [0, -1, 0, -1, 0, -1])


def test_x_aff_all_positions_equals_r1_including_counterparts():
    b, parts = synthetic_bundle(), B.A0 + ("V1",)
    pk, taus = uniform_reader(parts), [0., 0., 0., 0.]
    r1 = F.score_r1(b, parts, pk, taus)
    xa = F.score_x_aff(b, parts, pk, taus, allowed={0, 1, 2, 3})
    for field in ("fpick", "cpick", "sigma"):
        assert r1[field] == xa[field]
    for field in ("fused", "cf"):
        assert all(np.array_equal(r1[field][m], xa[field][m]) for m in r1[field])
    assert np.array_equal(r1["gate_a"], xa["gate_a"])
    assert np.array_equal(r1["gate_b"], xa["gate_b"])
    arrays = xa["arrays"]
    assert set(arrays) == {f"XAFF_V1{suffix}__{m}" for suffix in ("", "_cf")
                           for m in ("r1", "gain", "other", "swap", "strict")} | {
                               f"XAFF_V1__{m}" for m in ("fused_cells", "cf_cells", "sigma", "gate_a", "gate_b")}
    assert arrays["XAFF_V1__fused_cells"].dtype == np.int64
    assert arrays["XAFF_V1__sigma"].dtype == np.float64
    assert arrays["XAFF_V1__gate_a"].shape == (4, b.n)


def test_x_aff_default_uses_affect_and_x_positions_and_closes_other_picks():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    b, parts = synthetic_bundle(), B.A0 + ("V1",)
    b.ctx.groups = np.zeros(10)
    b.ctx.selection = np.array([1, 3, 5, 7, 9], dtype=np.int64)
    # Each support row agrees with exactly one grouping, in parts order.
    for j, h in enumerate(parts):
        p = np.full((10, 2), np.nan, dtype=np.float32)
        p[b.ctx.selection] = [0, 1]
        p[1] = [1, 0]
        b.post[h] = {"img": p.copy(), "txt": p.copy()}
        b.post[h]["txt"][3 + 2 * j] = [1, 0]
    b.ctx.pooled.pairs_a_txt = np.repeat(np.array([3, 5, 7, 9, 3, 7])[:, None], 4, axis=1)
    b.ctx.pooled.pairs_b_txt = np.repeat(np.array([9, 7, 5, 3, 9, 5])[:, None], 4, axis=1)
    # A real reader learns to select the grouping whose support feature is 1.
    y = np.tile(np.arange(4), 2)
    X = np.zeros((8, 24))
    X[np.arange(8), 6 * y] = 1
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(C=10).fit(scaler.transform(X), y)
    pk = uniform_reader(parts)
    pk["halves"] = [{"scaler": scaler, "model": model}] * 2
    out = F.score_x_aff(b, parts, pk, [0., .1, .2, .3])
    for c, picks, gates in (("a", [0, 1, 2, 3, 0, 2], [1, 0, 0, 1, 1, 0]),
                            ("b", [3, 2, 1, 0, 3, 1], [1, 0, 0, 1, 1, 0])):
        assert np.array_equal(out["reader"]["pick"][c], picks)
        assert (out["reader"]["m"][c] > .3).all(), "R1 gates must be open for every pick"
        assert np.array_equal(out[f"gate_{c}"], np.tile(gates, (4, 1)))


@pytest.mark.parametrize("parts", [B.A0, ("image", "affect", "caption", "V1")])
def test_x_aff_default_rejects_layouts_other_than_a0_plus_x(parts):
    with pytest.raises(AssertionError, match="default allowed set requires A0 followed by X"):
        F.score_x_aff(synthetic_bundle(), parts, uniform_reader(parts), [0.] * 4)


def checked_archive(key):
    path = R.path_of(key)
    assert R.sha_file(path) == R.INPUTS[key], "frozen regression input hash differs"
    return np.load(path, allow_pickle=False)


def test_seed42_aff_matches_round3_per_anchor_cells_sigma_and_gates(seed42):
    out = F.score_aff(seed42)
    assert out["fpick"] == {0: 39, 1: 119}
    assert out["cpick"] == {0: 149, 1: 10}
    assert out["sigma"] == {0: 0., 1: 0.}
    arrays = out["arrays"]
    assert arrays["AFF__fused_cells"].dtype == np.int64
    assert arrays["AFF__cf_cells"].dtype == np.int64
    assert np.array_equal(arrays["AFF__fused_cells"], [39, 119])
    assert np.array_equal(arrays["AFF__cf_cells"], [149, 10])
    assert np.array_equal(arrays["AFF__sigma"], [0, 0])
    with checked_archive("20261121_round3_affect_gate/results/seed42_arrays.npz") as z:
        assert np.array_equal(seed42.cl, z["cl"])
        assert np.array_equal(seed42.pair_index, z["pair_index"])
        assert np.array_equal(seed42.parity, z["parity"])
        for who, old in (("fused", "aff_fused"), ("cf", "aff_cf")):
            for m in ("r1", "gain", "other", "swap", "strict"):
                assert np.array_equal(out[who][m], z[f"{old}__{m}"])
        for c in ("a", "b"):
            assert np.array_equal(out[f"gate_{c}"], z[f"aff_gate__{c}"])
        assert np.array_equal(out["arrays"]["AFF__r1"], z["aff_fused__r1"])
        assert np.array_equal(out["arrays"]["AFF_cf__gain"], z["aff_cf__gain"])


def test_told_uses_aspect_map_opens_all_gates_and_preserves_gates_for_cf():
    b, parts = synthetic_bundle(), B.A0 + ("V1",)
    out = F.told(b, parts, {"emotion": "affect", "style": "V1", "genre": "image"})
    assert np.array_equal(out["reader"]["pick"]["a"], [0, 0, 0, 0, 3, 3])
    assert np.array_equal(out["reader"]["pick"]["b"], [3, 3, 1, 1, 1, 1])
    for c in ("a", "b"):
        assert np.array_equal(out["reader"]["m"][c], np.ones(b.n))
        assert np.array_equal(out[f"gate_{c}"], np.ones((4, b.n), dtype=np.float32))
    assert np.array_equal(out["cf"]["gain"], np.zeros(b.n))
    assert np.all((0 <= out["fused_cells"]) & (out["fused_cells"] < 224))


def test_tstar_overrides_posteriors_without_changing_bundle():
    b = synthetic_bundle()
    b.post["T"] = b.post.pop("V1")
    parts = B.A0 + ("T",)
    told_map = {"emotion": "affect", "style": "T", "genre": "image"}
    original = b.post["T"]["txt"].copy()
    override = {"T": {side: b.post["affect"][side].copy() for side in ("img", "txt")}}
    predicted = F.told(b, parts, told_map)
    truth = F.told(b, parts, told_map, posteriors_override=override)
    same = [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]
    opposite = [0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0]
    for d in ("i2t", "t2i"):
        assert np.array_equal(predicted["reader"]["T"]["b"][d][:2], [opposite] * 2)
        assert np.array_equal(truth["reader"]["T"]["b"][d][:2], [same] * 2)
    assert np.array_equal(b.post["T"]["txt"], original, equal_nan=True)
    assert "told_Tstar__r1" in truth["arrays"]


def test_seed42_told_a1_matches_spike_per_anchor(seed42):
    out = F.told(seed42, B.A1, {"emotion": "affect", "style": "R0", "genre": "image"})
    with checked_archive("20261119_no_caption_csd_spike/results/sp_cand_told_A1_224.npz") as z:
        assert np.array_equal(seed42.cl, z["anchor_group"])
        assert np.array_equal(seed42.pair_index, z["pair_index"])
        assert np.array_equal(out["fused"]["r1"], z["fused_r1"])
        assert np.array_equal(out["fused"]["r1"] - out["cf"]["r1"], z["margin_r1"])
    assert "told_A1__r1" in out["arrays"]


def test_seed42_a1_round1_reader_probabilities_and_r1_match_spike(seed42):
    key = "20261117_reader_fix_csd/results/rb_reader_A1.pkl"
    path = R.path_of(key)
    assert R.sha_file(path) == R.INPUTS[key], "frozen A1 reader hash differs"
    out = F.reader(seed42, B.A1, path)
    # Independently use round 1's original csd names/features in their original column order.
    import pickle
    with open(path, "rb") as f:
        pk = pickle.load(f)
    legacy = SimpleNamespace(**{**vars(seed42), "post": {**seed42.post, "csd": seed42.post["R0"]}})
    rbe, rf = R.load("rb_eval"), R.load("rb_features")
    features, _ = rbe.seed42_features(legacy, B.A0 + ("csd",))
    for c in ("a", "b"):
        expected = rf.average_probs(rbe.half_reader_probs(pk, features[c], 4))
        assert np.allclose(out["P"][c], expected, rtol=0, atol=1e-12)
        assert np.array_equal(out["pick"][c], np.argmax(expected, axis=1))
    taus, n = F.thresholds(out["m"])
    assert n == 24576
    scored = F.score_r1(seed42, B.A1, pk, taus)
    with checked_archive("20261119_no_caption_csd_spike/results/sp_cand_R1_A1_224.npz") as z:
        assert np.array_equal(seed42.cl, z["anchor_group"])
        assert np.array_equal(seed42.pair_index, z["pair_index"])
        assert np.array_equal(scored["fused"]["r1"], z["fused_r1"])
        assert np.array_equal(scored["fused"]["r1"] - scored["cf"]["r1"], z["margin_r1"])
