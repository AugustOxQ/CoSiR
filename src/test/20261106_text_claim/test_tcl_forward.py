"""Ticket 04 tests: late fusion algebra on real rows, one stored c* export, no label read."""
import re
from pathlib import Path

import numpy as np
import pytest

import tcl_paths as P
import tcl_forward as F
import rt_data

N = 400   # first paintings' rows


@pytest.fixture(scope='module')
def art():
    """art_val emotion seed 0: c*, plain, late rows (full), text and image."""
    out = {r: F.rows('art_val', 'caption', r, 'emotion' if r != 'plain' else 'all', 0 if r != 'plain' else 'x', 'cpu')
           for r in ('cstar', 'late', 'plain')}
    return out


def test_fuse_endpoints_and_mixture(art):
    c_t, c_i = art['cstar']
    p_t, p_i = art['plain']
    for c, p in ((c_t[:N], p_t[:N]), (c_i[:N], p_i[:N])):
        assert np.allclose(np.linalg.norm(c, axis=1), 1, atol=1e-5) and np.allclose(np.linalg.norm(p, axis=1), 1, atol=1e-5)
        zero, one = F.fuse(c, p, 0.0), F.fuse(c, p, 1.0)
        assert zero.shape[1] == 4096
        assert np.array_equal(zero[:, :2048], c) and not zero[:, 2048:].any()      # a = 0: c* up to the weight 1
        assert np.array_equal(one[:, 2048:], p) and not one[:, :2048].any()        # a = 1: plain
        assert np.allclose(zero @ zero[::-1].T, c @ c[::-1].T, atol=1e-6)
        assert np.allclose(one @ one[::-1].T, p @ p[::-1].T, atol=1e-6)


def test_late_rows_cosine_is_the_mixture(art):
    a = P.A
    assert a == 0.7
    (c_t, c_i), (p_t, p_i), (l_t, l_i) = art['cstar'], art['plain'], art['late']
    assert l_t.shape == (len(c_t), 4096) and l_i.shape == (len(c_i), 4096) and l_t.dtype == l_i.dtype == np.float32
    q, g = slice(0, N), slice(1000, 1000 + N)
    for lq, lg, cq, cg, pq, pg in ((l_t, l_i, c_t, c_i, p_t, p_i), (l_i, l_t, c_i, c_t, p_i, p_t),
                                   (l_t, l_t, c_t, c_t, p_t, p_t)):
        late = lq[q] @ lg[g].T
        want = (1 - a) * (cq[q] @ cg[g].T) + a * (pq[q] @ pg[g].T)
        assert np.abs(late - want).max() < 1e-5
        assert np.allclose(lq[q, :2048], np.sqrt(1 - a) * cq[q]) and np.allclose(lq[q, 2048:], np.sqrt(a) * pq[q])


def test_cstar_image_rows_match_saved_export(art):
    got = art['cstar'][1]
    saved = rt_data.embed('cstar', 'val', 'emotion', 0)
    assert got.shape == saved.shape == (P.N_PAINTINGS['art_val'], 2048)
    assert np.abs(got - saved).max() < 1e-3


def test_rows_refuse_bad_arguments():
    with pytest.raises(AssertionError):
        F.rows('art_val', 'caption', 'cstar', 'all', 0, 'cpu')
    with pytest.raises(AssertionError):
        F.rows('art_val', 'caption', 'plain', 'emotion', 'x', 'cpu')
    with pytest.raises(AssertionError):
        F.rows('art_val', 'caption', 'late', 'emotion', 'x', 'cpu')
    with pytest.raises(AssertionError):
        F.rows('mm_val', 'narrative', 'free', 'emotion', 0, 'cpu')       # free is ArtELingo only
    with pytest.raises(AssertionError):
        F.rows('art_val', 'narrative', 'plain', 'all', 'x', 'cpu')


def test_forward_reads_no_label():
    """tcl_forward.py names no label reader; gate (ii) takes images by name or key from the stores."""
    src = Path(F.__file__).read_text()
    code = '\n'.join(re.sub(r'#.*', '', line) for line in src.splitlines())
    body = code.split('"""', 2)[2]                      # drop the module doc
    for banned in ('labels', 'label_share', 'names_label', 'frame(', '.labels'):
        assert banned not in body.replace("'labels'", ''), banned
