"""Synthetic-array tests for ft_eval (no real scorer value is ever used here)."""
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))

import ft_eval as E  # noqa: E402


def _sel_setup():
    n = 10
    selection = np.array([1, 3, 4, 7])
    val = np.array([0, 5])
    return n, selection, val


def test_place_features_nan_outside_selection_and_val_ignored():
    n, sel, val = _sel_setup()
    rows = np.concatenate([val, sel])[::-1].copy()          # arbitrary order
    img = np.arange(len(rows) * 3, dtype=np.float32).reshape(-1, 3)
    txt = -img
    fi, ft = E.place_features(rows, img, txt, n, sel, allowed_extra=val)
    assert fi.shape == (n, 3) and ft.shape == (n, 3)
    out = np.setdiff1d(np.arange(n), sel)
    assert np.isnan(fi[out]).all() and np.isnan(ft[out]).all()         # val rows too
    for k, r in enumerate(rows):
        if r in sel:
            assert (fi[r] == img[k]).all() and (ft[r] == txt[k]).all()


def test_place_features_rejects_bad_input():
    n, sel, val = _sel_setup()
    rows = np.concatenate([val, sel])
    img = np.ones((len(rows), 2), np.float32)
    with pytest.raises(AssertionError):                    # a selection row missing
        E.place_features(rows[:-1], img[:-1], img[:-1], n, sel)
    with pytest.raises(AssertionError):                    # duplicate row
        E.place_features(np.concatenate([rows, rows[:1]]), np.ones((len(rows) + 1, 2), np.float32),
                         np.ones((len(rows) + 1, 2), np.float32), n, sel)
    bad = img.copy()
    bad[-1, 0] = np.nan                                    # non-finite selection feature
    with pytest.raises(AssertionError):
        E.place_features(rows, bad, img, n, sel)
    with pytest.raises(AssertionError):                    # a row that is neither selection nor val (a held row)
        E.place_features(np.concatenate([rows, [9]]), np.ones((len(rows) + 1, 2), np.float32),
                         np.ones((len(rows) + 1, 2), np.float32), n, sel, allowed_extra=val)


def test_place_features_val_extra_allowed():
    n, sel, val = _sel_setup()
    rows = np.concatenate([val, sel])
    img = np.ones((len(rows), 2), np.float32)
    E.place_features(rows, img, img, n, sel, allowed_extra=val)


def test_ci_is_in_percentage_points():
    cl = np.arange(40) // 2
    r = E.ci(np.full(40, 0.25), cl)
    assert r["point"] == pytest.approx(25.0)
    assert r["n_clusters"] == 20


def _fake(seeds, per_seed=24, n_paint=6):
    """data[seed] with painting ids shared across seeds; scorers with known constant per-anchor values."""
    data = {}
    for s in seeds:
        cl = np.arange(per_seed) % n_paint
        pair = np.repeat(np.arange(3), per_seed // 3)
        rng = np.random.default_rng(s)
        jitter = rng.normal(0, 0.01, per_seed)
        def sc(r1, gain=0.0, other=0.1):
            return {"r1": np.full(per_seed, r1) + jitter, "gain": np.full(per_seed, gain) + jitter,
                    "other": np.full(per_seed, other) + jitter}
        scorers = {"plain": sc(0.40), "B": sc(0.45, 0.1), "Bp0": sc(0.50, 0.0), "AFF": sc(0.70, 0.3, 0.05),
                   "ft:x": sc(0.60)}
        if s == 42:
            scorers["Bp1"] = sc(0.52)
        data[s] = {"cl": cl, "pair_index": pair, "scorers": scorers}
    return data


def test_difference_signs_and_definitions():
    rep = E.build_report(_fake([42, 49, 50, 51]), ["x"], pooled_seeds=(49, 50, 51))
    pooled = rep["pooled"]["comparisons"]["x"]
    assert pooled["AFF_minus_ft"]["r1"]["point"] == pytest.approx(10.0, abs=1e-9)
    assert pooled["ft_minus_plain"]["r1"]["point"] == pytest.approx(20.0, abs=1e-9)
    assert pooled["ft_minus_Bp0"]["r1"]["point"] == pytest.approx(10.0, abs=1e-9)
    assert pooled["AFF_minus_ft"]["r1"]["ci95"][0] > 0
    assert pooled["ft_minus_plain"]["gain"]["point"] == pytest.approx(0.0, abs=1e-9)
    assert pooled["AFF_minus_ft"]["other"]["point"] == pytest.approx(-5.0, abs=1e-9)   # AFF 0.05 - ft 0.10
    # either = r1 + other: AFF 0.75, ft 0.70, plain 0.50, Bp0 0.60
    sc = rep["pooled"]["scorers"]
    assert sc["AFF"]["either"]["point"] == pytest.approx(75.0, abs=1.0)   # jitter enters r1 and other
    assert sc["AFF"]["either"]["point"] == pytest.approx(sc["AFF"]["r1"]["point"] + sc["AFF"]["other"]["point"])
    assert pooled["AFF_minus_ft"]["either"]["point"] == pytest.approx(5.0, abs=1e-9)
    assert pooled["ft_minus_plain"]["either"]["point"] == pytest.approx(20.0, abs=1e-9)
    assert pooled["ft_minus_Bp0"]["either"]["point"] == pytest.approx(10.0, abs=1e-9)
    lo, hi = pooled["AFF_minus_ft"]["either"]["ci95"]
    assert lo <= 5.0 <= hi


def test_pooled_clusters_are_paintings_across_seeds():
    rep = E.build_report(_fake([42, 49, 50, 51]), ["x"], pooled_seeds=(49, 50, 51))
    assert rep["pooled"]["n_episodes"] == 72
    assert rep["pooled"]["scorers"]["AFF"]["r1"]["n_clusters"] == 6        # not 18: same painting on three seeds
    assert rep["seeds"]["49"]["scorers"]["AFF"]["r1"]["n_clusters"] == 6


def test_pooled_concatenation_order_matters_for_alignment():
    data = _fake([49, 50, 51])
    cat = E.pool(data, (49, 50, 51), "AFF", "r1")
    assert np.array_equal(cat[:24], data[49]["scorers"]["AFF"]["r1"])
    assert np.array_equal(cat[48:], data[51]["scorers"]["AFF"]["r1"])


def test_bp1_only_on_seed42_and_pair_split():
    rep = E.build_report(_fake([42, 49, 50, 51]), ["x"], pooled_seeds=(49, 50, 51))
    assert "Bp1" in rep["seeds"]["42"]["scorers"]
    assert "Bp1" not in rep["seeds"]["49"]["scorers"] and "Bp1" not in rep["pooled"]["scorers"]
    assert set(rep["pairs"]) == {"0", "1", "2"} and set(rep["pairs_seed42"]) == {"0", "1", "2"}
    assert rep["pairs_seed42"]["0"]["n_episodes"] == 8
    assert rep["pairs"]["1"]["n_episodes"] == 24
    assert rep["pairs"]["2"]["comparisons"]["x"]["ft_minus_plain"]["r1"]["point"] == pytest.approx(20.0, abs=1e-9)


def test_per_anchor_alignment_assert():
    E.assert_aligned(np.array([1, 2, 3]), np.array([0, 1, 1]), np.array([1, 2, 3]), np.array([0, 1, 1]), "t")
    with pytest.raises(AssertionError):
        E.assert_aligned(np.array([1, 2, 3]), np.array([0, 1, 1]), np.array([1, 2, 4]), np.array([0, 1, 1]), "t")
    with pytest.raises(AssertionError):
        E.assert_aligned(np.array([1, 2, 3]), np.array([0, 1, 1]), np.array([1, 2, 3]), np.array([0, 1, 2]), "t")


def test_score_features_matches_cosine_on_tiny_episodes():
    from src.eval.aspect_episodes import AspectEpisodes
    rng = np.random.default_rng(0)
    img = rng.normal(size=(12, 4)).astype(np.float32)
    txt = rng.normal(size=(12, 4)).astype(np.float32)
    n = 5
    anchor = np.arange(n)
    cand = np.stack([np.arange(n), (np.arange(n) + 1) % 12, (np.arange(n) + 2) % 12, (np.arange(n) + 3) % 12], 1)
    pa = np.arange(n)
    ep = AspectEpisodes("a", "b", anchor, cand, pa, pa, pa, pa)
    r = E.score_features(img, txt, ep)
    assert set(r) == {"r1", "gain", "other", "swap", "strict"} and r["r1"].shape == (n,)
    assert np.array_equal(r["gain"], np.zeros(n))                          # cosine: gain 0 by construction


def test_hand_computed_episode_top1_and_either():
    from src.eval.aspect_episodes import AspectEpisodes
    # unit 2-d features. Image row 0 points along x; captions: row 0 = x (candidate p_a), row 1 = y (p_b),
    # row 2 = -x. i2t from anchor image 0: cosines [1, 0, -1] -> p_a is top-1 (hit), p_b is not.
    # t2i from anchor caption 0 (x): images rows 0..2 = x, y, -x -> cosines [1, 0, -1] -> p_a top-1 again.
    img = np.array([[1, 0], [0, 1], [-1, 0]], np.float32)
    txt = img.copy()
    ep = AspectEpisodes("a", "b", np.array([0]), np.array([[0, 1, 2]]), np.array([[0]]), np.array([[0]]),
                        np.array([[1]]), np.array([[1]]))
    r = E.score_features(img, txt, ep)
    # cosine ignores the condition: p_a wins under both. r1 = 0.5*(hit_aa=1 + hit_bb=0) = 0.5;
    # other = 0.5*(hit_ba=0 + hit_ab=1) = 0.5; gain = r1 - other = 0; either = r1 + other = 1.
    assert r["r1"][0] == 0.5 and r["other"][0] == 0.5 and r["gain"][0] == 0.0
    assert E.metric_values(r, "either")[0] == 1.0
    assert r["swap"][0] == 0.0 and r["strict"][0] == 0.0
