"""Unit tests of R-b's pure pieces (rb_features.py) on synthetic arrays. Run from /project/CoSiR:
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python -m pytest -q src/test/20261117_reader_fix_csd/test_rb.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rb_features as rf  # noqa: E402

PARTS = ("affect", "image", "caption", "csd")


def synthetic(n_rows=300, n_ep=50, seed=0):
    rng = np.random.default_rng(seed)
    post = {}
    for j, h in enumerate(PARTS):
        k = 5 + j
        post[h] = {m: rng.dirichlet(np.ones(k), size=n_rows).astype(np.float32) for m in ("img", "txt")}
    ep = SimpleNamespace(**{f: rng.integers(0, n_rows, size=(n_ep, 4)) for f in
                            ("pairs_a_img", "pairs_a_txt", "pairs_b_img", "pairs_b_txt")})
    return post, ep


def test_feature_layout_and_values():
    post, ep = synthetic()
    F = rf.both_conditions(post, PARTS, ep)["a"]
    assert F.shape == (50, 6 * len(PARTS)) and F.dtype == np.float64
    assert rf.feature_names(PARTS)[:6] == [f"affect__{f}" for f in rf.FEATURES]
    h = 2                                                   # caption block: columns 12..17
    pi, pt = post["caption"]["img"], post["caption"]["txt"]
    sup = np.einsum("nsc,nsc->ns", pi[ep.pairs_a_img], pt[ep.pairs_a_txt]).astype(np.float64)
    con = np.einsum("nsc,nsc->ns", pi[ep.pairs_b_img], pt[ep.pairs_b_txt]).astype(np.float64)
    blk = F[:, 6 * h: 6 * h + 6]
    np.testing.assert_allclose(blk[:, 0], sup.mean(1), rtol=1e-6)
    np.testing.assert_allclose(blk[:, 1], con.mean(1), rtol=1e-6)
    np.testing.assert_allclose(blk[:, 3], sup.std(1, ddof=1), rtol=1e-12)
    np.testing.assert_allclose(blk[:, 4], con.std(1, ddof=1), rtol=1e-12)
    match = (pi[ep.pairs_a_img].argmax(-1) == pt[ep.pairs_a_txt].argmax(-1)).mean(1)
    np.testing.assert_array_equal(blk[:, 5], match)
    assert set(np.unique(blk[:, 5])) <= {0, 0.25, 0.5, 0.75, 1.0}


def test_delta_equals_aspect_deltas_formula():
    post, ep = synthetic(seed=3)
    F = rf.both_conditions(post, PARTS, ep)["a"]
    for j, h in enumerate(PARTS):                           # aspect_deltas' float32 computation, bit for bit
        pi, pt = post[h]["img"], post[h]["txt"]
        d = (np.einsum("nsc,nsc->ns", pi[ep.pairs_a_img], pt[ep.pairs_a_txt]).mean(axis=1)
             - np.einsum("nsc,nsc->ns", pi[ep.pairs_b_img], pt[ep.pairs_b_txt]).mean(axis=1))
        np.testing.assert_array_equal(F[:, 6 * j + 2], d.astype(np.float64))


def test_condition_swap_symmetry():
    post, ep = synthetic(seed=1)
    F = rf.both_conditions(post, PARTS, ep)
    a, b = F["a"], F["b"]
    for j in range(len(PARTS)):
        s, c, d, sds, sdc, _ = (6 * j + i for i in range(6))
        np.testing.assert_array_equal(b[:, s], a[:, c])      # condition b's S = condition a's C
        np.testing.assert_array_equal(b[:, c], a[:, s])
        np.testing.assert_array_equal(b[:, d], -a[:, d])     # Delta^b = -Delta^a exactly (D3)
        np.testing.assert_array_equal(b[:, sds], a[:, sdc])
        np.testing.assert_array_equal(b[:, sdc], a[:, sds])
    # the match share of b is taken on b's supports (a's contrasts)
    pi, pt = post["image"]["img"], post["image"]["txt"]
    np.testing.assert_array_equal(b[:, 6 + 5], (pi[ep.pairs_b_img].argmax(-1) == pt[ep.pairs_b_txt].argmax(-1)).mean(1))


def test_bank_labels_from_block_positions():
    blocks = rf.bank_blocks(PARTS)
    assert blocks == [("affect", "caption"), ("affect", "csd"), ("affect", "image"), ("caption", "csd"),
                      ("caption", "image"), ("csd", "image")]
    ya, yb = rf.bank_labels(blocks, 3, PARTS)
    # configuration order: affect 0, image 1, caption 2, csd 3
    np.testing.assert_array_equal(ya, np.repeat([0, 0, 0, 2, 2, 3], 3))
    np.testing.assert_array_equal(yb, np.repeat([2, 3, 1, 3, 1, 1], 3))
    y = np.concatenate([ya, yb])
    assert (np.bincount(y, minlength=4) == 9).all()          # balanced by construction
    a0 = ("affect", "image", "caption")
    ya, yb = rf.bank_labels(rf.bank_blocks(a0), 2, a0)
    np.testing.assert_array_equal(ya, [0, 0, 0, 0, 2, 2])
    np.testing.assert_array_equal(yb, [2, 2, 1, 1, 1, 1])


def test_folds_keep_both_conditions_together():
    n = 103
    Xa, Xb = np.zeros((n, 2)), np.ones((n, 2))
    X, y, epi = rf.stack_conditions(Xa, Xb, np.zeros(n, int), np.ones(n, int))
    assert X.shape == (2 * n, 2) and (epi[:n] == np.arange(n)).all() and (epi[n:] == np.arange(n)).all()
    fold = rf.episode_folds(n)
    ex_fold = fold[epi]
    assert (ex_fold[:n] == ex_fold[n:]).all()                # condition a and b of an episode share a fold
    assert sorted(np.bincount(fold).tolist()) == [20, 20, 21, 21, 21]
    from sklearn.model_selection import KFold
    for f, (_, te) in enumerate(KFold(5, shuffle=True, random_state=0).split(np.arange(n))):
        assert (fold[te] == f).all()


def test_choose_c_ties_to_smaller():
    assert rf.choose_c([0.5, 0.4, 0.4, 0.6, 0.7]) == 1
    assert rf.choose_c([0.3, 0.3, 0.3, 0.3, 0.3]) == 0
    assert rf.choose_c([0.9, 0.8, 0.7, 0.6, 0.5]) == 4


def test_average_probs_sum_to_one_and_picks():
    rng = np.random.default_rng(2)
    p0, p1 = rng.dirichlet(np.ones(4), size=200), rng.dirichlet(np.ones(4), size=200)
    P = rf.average_probs([p0, p1])
    np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(P, (p0 + p1) / 2)
    pick, margin = rf.picks_and_margins(P)
    assert (pick == P.argmax(1)).all() and (margin >= 0).all()
    tie = np.array([[0.4, 0.4, 0.2], [0.1, 0.45, 0.45]])
    pick, margin = rf.picks_and_margins(tie)
    np.testing.assert_array_equal(pick, [0, 1])              # ties to the first grouping in configuration order
    np.testing.assert_allclose(margin, [0.0, 0.0])


def test_smd():
    rng = np.random.default_rng(4)
    a, b = rng.normal(1.0, 1.0, size=(5000, 2)), rng.normal(0.0, 1.0, size=(8000, 2))
    s = rf.smd(a, b)
    assert np.all(np.abs(s - 1.0) < 0.06)
    const = rf.smd(np.ones((10, 1)), np.ones((10, 1)))
    assert const[0] == 0.0
    assert np.isnan(rf.smd(np.ones((10, 1)), np.zeros((10, 1)))[0])


def test_shape_errors():
    post, ep = synthetic()
    with pytest.raises(ValueError):
        rf.episode_features(post, PARTS, ep.pairs_a_img[:, :3], ep.pairs_a_txt[:, :3], ep.pairs_b_img[:, :3],
                            ep.pairs_b_txt[:, :3])
