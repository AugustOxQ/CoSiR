"""Tests of om_clay (ticket 01): synthetic rows for the math, the reference files for gate (i)."""
import numpy as np
import pytest

import om_clay as oc
import om_paths as op


def unit(a):
    a = np.asarray(a, dtype=np.float64)
    return (a / np.linalg.norm(a, axis=1, keepdims=True)).astype(np.float32)


def rows(n, d, seed, shift=0.0):
    g = np.random.default_rng(seed)
    x = g.standard_normal((n, d))
    x[:, 0] += shift
    return unit(x)


def test_reflections_map_mean_to_mu_and_are_isometries():
    img, txt = rows(60, 16, 0, shift=2.0), rows(12, 16, 1, shift=-1.0)
    mu = unit(txt.mean(0, keepdims=True))[0]
    mean = unit(img.mean(0, keepdims=True))[0]
    out = oc.align(img, mu, mean)
    assert np.allclose(unit(out.mean(0, keepdims=True))[0], mu, atol=1e-5)
    # the transform is linear and orthogonal: it keeps all inner products
    assert np.allclose(out @ out.T, img @ img.T, atol=1e-5)
    # the mean itself lands on mu
    assert np.allclose(oc.align(mean[None], mu, mean)[0], mu, atol=1e-5)


def test_alignment_is_not_a_single_reflection_or_identity():
    # two reflections: determinant +1 (a rotation); one reflection would give -1
    d = 8
    img, txt = rows(30, d, 2, shift=1.5), rows(9, d, 3, shift=-1.5)
    mu = unit(txt.mean(0, keepdims=True))[0]
    mean = unit(img.mean(0, keepdims=True))[0]
    m = oc.align(np.eye(d, dtype=np.float32), mu, mean)
    assert np.linalg.det(m.astype(np.float64)) == pytest.approx(1.0, abs=1e-4)
    assert not np.allclose(m, np.eye(d), atol=1e-3)


def test_alignment_identity_when_mean_equals_mu():
    mu = unit(np.arange(1, 9, dtype=np.float64)[None])[0]
    x = rows(5, 8, 4)
    assert np.allclose(oc.align(x, mu, mu.copy()), x, atol=1e-6)


def test_log_map_of_mu_is_zero_and_norm_is_angle():
    mu = np.random.default_rng(5).standard_normal(10)
    mu /= np.linalg.norm(mu)                      # float64 unit vector: mu itself is then exactly on the sphere
    assert np.abs(oc.log_map(mu, mu[None])).max() < 1e-6
    x = rows(7, 10, 6)
    lg = oc.log_map(mu, x)
    ang = np.arccos(np.clip(x @ mu, -1, 1))
    assert np.allclose(np.linalg.norm(lg, axis=1), ang, atol=1e-4)
    assert np.abs(lg @ mu).max() < 1e-5          # tangent vectors are orthogonal to mu


def test_subspace_is_rank_r_orthogonal_projector_and_mu_is_mean_direction():
    txt = rows(20, 32, 7, shift=1.0)
    P, mu = oc.subspace(txt, 7)
    assert P.shape == (32, 32) and P.dtype == np.float32 and mu.shape == (32,)
    assert np.allclose(mu, unit(txt.mean(0, keepdims=True))[0], atol=1e-6)
    assert np.allclose(P, P.T, atol=1e-5) and np.allclose(P @ P, P, atol=1e-5)
    assert np.trace(P) == pytest.approx(7.0, abs=1e-4)
    # the span is the top right singular space of the centred tangent vectors
    t = oc.log_map(mu, txt)
    t = t - t.mean(0, keepdims=True)
    vt = np.linalg.svd(t.astype(np.float64), full_matrices=False)[2]
    assert np.allclose(P, vt[:7].T @ vt[:7], atol=1e-4)


def test_subspace_asserts():
    txt = rows(10, 16, 8)
    with pytest.raises(AssertionError):
        oc.subspace(txt, 10)                      # rank must be <= n - 1
    with pytest.raises(AssertionError):
        oc.subspace(txt * 2.0, 4)                 # unit rows only


def test_condition_unit_rows_depends_on_every_row_passed_and_lies_in_subspace():
    txt = rows(25, 24, 9, shift=1.0)
    P, mu = oc.subspace(txt, 6)
    img = rows(40, 24, 10, shift=-1.0)
    out = oc.condition(img, P, mu)
    assert out.shape == (40, 24) and out.dtype == np.float32
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
    assert np.allclose(out @ P, out, atol=1e-5)   # rows sit in the projector's range
    # the alignment mean is over every row passed: a subset gives other vectors for the same images
    sub = oc.condition(img[:20], P, mu)
    assert np.abs(sub - out[:20]).max() > 1e-3
    # row order does not change a row's vector
    perm = np.random.default_rng(0).permutation(40)
    assert np.allclose(oc.condition(img[perm], P, mu), out[perm], atol=1e-5)


def test_text_set_a_and_b(tmp_path, monkeypatch):
    a = oc.text_set('A', 'emotion')
    assert a.shape == (50, 768) and a.dtype == np.float32
    assert np.array_equal(a, np.load(op.CLAY_TEXT_A['emotion']))
    b = rows(30, 768, 11)
    np.save(tmp_path / 'style_text.npy', b)
    monkeypatch.setattr(op, 'CLAY_B_DIR', tmp_path)
    assert np.array_equal(oc.text_set('B', 'style'), b)
    with pytest.raises(AssertionError):
        oc.text_set('B', 'genre')                 # no file: that aspect fell back (no .npy)
    with pytest.raises(AssertionError):
        oc.text_set('C', 'style')


def test_vectors_uses_clay_rank_per_set(tmp_path, monkeypatch):
    b = rows(30, 768, 12)
    np.save(tmp_path / 'genre_text.npy', b)
    monkeypatch.setattr(op, 'CLAY_B_DIR', tmp_path)
    img = rows(100, 768, 13, shift=1.0)
    got = oc.vectors(img, 'genre', 'B')
    want = oc.condition(img, *oc.subspace(b, op.clay_rank(30)))
    assert np.array_equal(got, want)
    P, _ = oc.subspace(b, 29)
    assert np.trace(P) == pytest.approx(29.0, abs=1e-2)


# gate (i) on the official reference (made once by the official code; read only)
def test_gate_i_against_reference(tmp_path):
    res = oc.gate_i(out_dir=tmp_path)
    assert res['pass'], res
    assert res['max_abs']['P'] < op.TOL_CLAY and res['max_abs']['mu'] < op.TOL_CLAY
    assert res['max_abs']['query'] < op.TOL_CLAY and res['max_abs']['gallery'] < op.TOL_CLAY
    assert res['text_equals_set_a'] is True
    assert (tmp_path / 'stage0' / 'gate_i.json').exists()
    assert res['shapes']['P'] == [768, 768]


def test_gate_i_fails_when_the_math_is_broken(tmp_path, monkeypatch):
    # a wrong alignment (coordinates reversed) must fail the gate
    real = oc.align
    monkeypatch.setattr(oc, 'align', lambda x, mu, mean: real(x, mu, mean)[:, ::-1].copy())
    res = oc.gate_i(out_dir=tmp_path)
    assert not res['pass']
