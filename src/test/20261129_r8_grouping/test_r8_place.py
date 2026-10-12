"""Tests for ticket 08 (r8_place): what the verdict rests on, per DECISION_RULE.md M11, M12, M13, M16.

Synthetic data throughout. Run from the round folder:
python -m pytest -q -p no:cacheprovider --rootdir=. test_r8_place.py
"""
from types import SimpleNamespace

import numpy as np
import pytest

import r8_groupings as G
import r8_place as P


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def toy(n_p=40, k=3, d=4, seed=0):
    """n_p paintings in k groups (round robin), halves alternate by a fixed shuffle, unit float32 vectors."""
    rng = np.random.default_rng(seed)
    labels = np.arange(n_p) % k
    halves = rng.permutation(np.arange(n_p) % 2)
    phi = unit(rng.normal(size=(n_p, d)) + 3 * np.eye(d)[labels % d])
    return phi, labels.astype(np.int64), halves.astype(np.int64)


# ---------------------------------------------------------------- M11: nearest centre

def test_centres_are_unit_mean_in_float64():
    phi, labels, _ = toy()
    mu = P.unit_centres(phi, labels, 3)
    assert mu.dtype == np.float64 and mu.shape == (3, 4)
    for g in range(3):
        m = phi[labels == g].astype(np.float64).mean(axis=0)
        assert np.allclose(mu[g], m / np.linalg.norm(m), rtol=0, atol=1e-15)
    assert np.allclose(np.linalg.norm(mu, axis=1), 1, atol=1e-12)


def test_nearest_centre_argmax_and_ties_to_lowest_group():
    mu = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 1.0, 0], [0, 0, 1.0]])         # groups 1 and 2 share a centre
    phi = np.array([[0.9, 0.1, 0], [0.1, 0.9, 0], [0, 0.2, 0.8], [0.5, 0, 0.5]], dtype=np.float32)
    m = P.nearest_centre(phi, mu)
    assert m.tolist() == [0, 1, 3, 0]            # row 1: tie between groups 1 and 2 -> 1; row 3: tie 0 and 3 -> 0
    assert m.dtype == np.int64


def test_nearest_centre_uses_float64():
    # a difference of 1e-9 in the score is invisible in float32 but decides the argmax in float64
    mu = np.array([[1.0, 0.0], [1.0 + 1e-9, 0.0]])
    phi = np.array([[1.0, 0.0]], dtype=np.float32)
    assert P.nearest_centre(phi, mu).tolist() == [1]


def test_selection_images_use_all_painting_centres():
    phi, labels, halves = toy(n_p=60, seed=3)
    rng = np.random.default_rng(5)
    sel = unit(rng.normal(size=(30, 4)) + 3 * np.eye(4)[rng.integers(0, 3, 30)])
    got = P.own_map(sel, phi, labels, 3)
    mu_all = P.unit_centres(phi, labels, 3)
    assert np.array_equal(got, np.argmax(sel.astype(np.float64) @ mu_all.T, axis=1))
    # the all-painting centres differ from each half's centres, so a half-centre map would give another answer here
    for h in (0, 1):
        mu_h = P.unit_centres(phi, labels, 3, members=halves == h)
        assert not np.allclose(mu_h, mu_all)


def test_one_hot_float32():
    o = P.onehot(np.array([2, 0, 1, 2]), 3)
    assert o.dtype == np.float32 and o.shape == (4, 3)
    assert np.array_equal(o, np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32))


# ---------------------------------------------------------------- M12: cross-fitted bank inputs

def test_half_centres_never_use_the_other_half():
    phi, labels, halves = toy(n_p=60, seed=1)
    base = P.crossfit_centres(phi, labels, halves, 3)
    for h in (0, 1):
        # centres used for half-h paintings come from half 1-h only: replacing every half-h vector changes nothing
        changed = phi.copy()
        changed[halves == h] = unit(np.random.default_rng(9).normal(size=((halves == h).sum(), 4)))
        again = P.crossfit_centres(changed, labels, halves, 3)
        assert np.array_equal(again[h], base[h])                    # mu^(1-h), built from half 1-h members
        assert not np.array_equal(again[1 - h], base[1 - h])        # while the other half's centres did move
    # and base[h] is exactly the unit mean of the half-(1-h) members
    for h in (0, 1):
        for g in range(3):
            m = phi[(labels == g) & (halves == 1 - h)].astype(np.float64).mean(axis=0)
            assert np.allclose(base[h][g], m / np.linalg.norm(m), rtol=0, atol=1e-15)


def test_crossfit_labels_use_other_half_centres_and_full_grouping_labels():
    phi, labels, halves = toy(n_p=60, seed=2)
    got = P.crossfit_painting_labels(phi, labels, halves, 3)
    for h in (0, 1):
        mu = P.unit_centres(phi, labels, 3, members=halves == 1 - h)
        idx = np.flatnonzero(halves == h)
        assert np.array_equal(got[idx], np.argmax(phi[idx].astype(np.float64) @ mu.T, axis=1))
    # a painting whose own half-h label would be wrong under half-h centres but right under the other half's:
    # poison half h so that its own centres point elsewhere; the map of a half-h painting must not change
    h = 0
    poisoned = phi.copy()
    poisoned[(halves == h) & (labels == 0)] = unit(np.ones((int(((halves == h) & (labels == 0)).sum()), 4)))
    base = P.crossfit_painting_labels(phi, labels, halves, 3)
    again = P.crossfit_painting_labels(poisoned, labels, halves, 3)
    keep = (halves == h) & (labels != 0)
    assert np.array_equal(base[keep], again[keep])


def test_group_without_member_in_a_half_drops_x_with_recorded_reason():
    phi, labels, halves = toy(n_p=40, seed=4)
    halves = np.where(labels == 2, 1, halves)                      # group 2 has no member in half 0
    reason = P.m12_reason(labels, halves, 3)
    assert reason is not None and "M12" in reason and "group 2" in reason and "half 0" in reason
    assert P.m12_reason(*toy(n_p=40, seed=4)[1:], 3) is None
    out = P.image_side(phi, labels, halves, np.arange(40), phi[:5])
    assert out == {"dropped": True, "reason": reason}
    rec = P.drop_record("X", reason, smoke=True)
    assert rec["dropped"] is True and rec["reason"] == reason and rec["grouping"] == "X" and rec["smoke"] is True
    # the group-less half also stops centres being built (no silent fallback)
    with pytest.raises(AssertionError):
        P.unit_centres(phi, labels, 3, members=halves == 0)


def test_image_side_rows_and_format():
    phi, labels, halves = toy(n_p=40, seed=6)
    g_st = np.repeat(np.arange(40), 2)                              # two rows per painting
    sel = phi[:7]
    out = P.image_side(phi, labels, halves, g_st, sel)
    assert out["img_sel"].shape == (7, 3) and out["img_sel"].dtype == np.float32
    assert out["img_cf"].shape == (80, 3) and out["img_cf"].dtype == np.float32
    assert (out["img_cf"].sum(1) == 1).all() and (out["img_sel"].sum(1) == 1).all()
    assert np.array_equal(out["img_cf"][::2], out["img_cf"][1::2])  # both rows of a painting agree
    assert np.array_equal(out["img_cf"].argmax(1)[::2], P.crossfit_painting_labels(phi, labels, halves, 3))
    assert np.array_equal(out["img_sel"].argmax(1), P.own_map(sel, phi, labels, 3))
    assert 0 <= out["agreement_fit"] <= 1 and 0 <= out["agreement_crossfit"] <= 1


def test_tp_space_is_one_hot_so_own_map_reproduces_the_labels():
    labels = np.random.default_rng(0).integers(0, 5, 100)
    labels[:5] = np.arange(5)
    phi = np.eye(5, dtype=np.float32)[labels]                       # 07's TP space: one-hot of the merged TP group
    assert np.array_equal(P.own_map(phi, phi, labels, 5), labels)
    mu = P.unit_centres(phi, labels, 5)
    assert np.array_equal(mu, np.eye(5))


# ---------------------------------------------------------------- M13: random controls

def test_control_draws_follow_shares_and_seeds():
    labels_p = np.repeat(np.arange(4), [400, 300, 200, 100])
    g_st = np.repeat(np.arange(1000), 2)
    sel_codes = np.repeat([5, 9, 12, 40], [3, 1, 2, 2])             # 4 paintings in ascending code, 8 rows
    i = 5
    train_rows, sel_rows = P.control_image_labels(labels_p, i, g_st, sel_codes)
    shares = np.bincount(labels_p, minlength=4) / 1000
    exp_train = np.random.default_rng(7200 + i).choice(4, size=1000, p=shares)
    exp_sel = np.random.default_rng(7300 + i).choice(4, size=4, p=shares)
    assert np.array_equal(train_rows, exp_train[g_st])              # same draw for every row of a painting
    assert np.array_equal(sel_rows, exp_sel[[0, 0, 0, 1, 2, 2, 3, 3]])
    # it is r8_groupings.control_draws that makes the draws (the two cannot drift)
    tr, se = G.control_draws(labels_p, i, 4)
    assert np.array_equal(exp_train, tr) and np.array_equal(exp_sel, se)
    # the shares: a large draw reproduces the group-size shares
    big = np.repeat(np.arange(4), [40_000, 30_000, 20_000, 10_000])
    tr_big, _ = G.control_draws(big, 1, 10)
    assert np.allclose(np.bincount(tr_big, minlength=4) / len(big), [0.4, 0.3, 0.2, 0.1], atol=0.01)


def test_control_draw_streams_differ_by_index_and_by_side():
    labels_p = np.repeat(np.arange(3), 50)
    g_st = np.arange(150)
    a, _ = P.control_image_labels(labels_p, 1, g_st, np.arange(10))
    b, _ = P.control_image_labels(labels_p, 2, g_st, np.arange(10))
    assert not np.array_equal(a, b)
    a_train, a_sel = P.control_image_labels(labels_p, 1, g_st, np.arange(150))
    assert not np.array_equal(a_train, a_sel)                       # 7200 + i and 7300 + i are separate streams


def test_control_side_arrays():
    labels_p = np.repeat(np.arange(3), 20)
    g_st = np.repeat(np.arange(60), 2)
    out = P.control_side(labels_p, 4, g_st, np.repeat([2, 8, 9], 2))
    assert out["img_cf"].shape == (120, 3) and out["img_sel"].shape == (6, 3)
    assert out["img_cf"].dtype == np.float32 and (out["img_sel"].sum(1) == 1).all()


# ---------------------------------------------------------------- R7 format

def _ctx(n=50, sel=(3, 4, 9, 20, 21, 33)):
    return SimpleNamespace(groups=np.arange(n), selection=np.array(sel, dtype=np.int64))


def test_post_matches_r7_posterior_ok_and_xheads_match_the_reader_checks(tmp_path):
    import r8_bundle as B
    import r8_reader as Q
    ctx = _ctx()
    k = 3
    img_sel = P.onehot(np.array([0, 1, 2, 0, 1, 2]), k)
    txt = np.full((50, k), np.nan, dtype=np.float32)
    txt[ctx.selection] = np.full((6, k), 1 / 3, dtype=np.float32)
    post = P.post_arrays(50, ctx.selection, img_sel, txt)
    B._posterior_ok({m: post[m] for m in ("img", "txt")}, ctx, "X")
    assert post["rows"].dtype == np.int64 and np.array_equal(post["rows"], ctx.selection)
    assert np.allclose(post["img"][ctx.selection].sum(1), 1) and np.isnan(post["img"][0]).all()
    # a NaN-free image side outside the selection, or a txt side on the wrong rows, is refused
    bad = txt.copy()
    bad[0] = 1 / 3
    with pytest.raises(AssertionError):
        P.post_arrays(50, ctx.selection, img_sel, bad)
    # xheads: rows = scorer-train, filled_by_half = 1 - half, classes arange(K)
    half = np.array([0, 1, 1, 0, 0, 1, 0, 1], dtype=np.int8)
    img_cf = P.onehot(np.array([0, 1, 2, 0, 1, 2, 0, 1]), k)
    txt_cf = np.full((8, k), 1 / 3, dtype=np.float32)
    halves = {"scorer_train": np.arange(8), "half_of_local_row": half}
    x = P.xheads_arrays(img_cf, txt_cf, half, k)
    assert x["filled_by_half"].dtype == np.int8 and np.array_equal(x["filled_by_half"], 1 - half)
    assert all(x[f"classes_half{j}"].dtype == np.int64 and np.array_equal(x[f"classes_half{j}"], np.arange(k))
               for j in (0, 1))
    root = tmp_path
    (root / "heads").mkdir()
    np.savez(root / "heads" / "xheads_X.npz", **x)
    got = Q.load_crossfit("X", np.arange(k).repeat(3)[:8], halves, root)
    assert set(got) == {"img", "txt"}
    with pytest.raises(AssertionError):
        P.xheads_arrays(img_cf * 0.5, txt_cf, half, k)              # rows must sum to 1
