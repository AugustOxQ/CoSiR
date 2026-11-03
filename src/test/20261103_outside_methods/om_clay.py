"""CLAY conditioning, written from the method's math (spec §3.4 CLAY, D4, D6, D12, D23; ticket 01).

For one aspect: a subspace (P, mu) from unit CLIP L/14 text rows, then each image set is turned by two Householder
reflections so that its normalized mean image sits on mu, mapped to the tangent space at mu (log map), projected by P
and normalized. No label enters: the text rows and the mean of the images passed are the only inputs.

CLI: `python om_clay.py gate-i` writes OUT/stage0/gate_i.json.
"""
import sys
from pathlib import Path

import numpy as np

import om_paths as op

D = 768
EPS_COS = 1e-7           # dot products are clamped to [-1 + EPS_COS, 1 - EPS_COS] before the arc cosine
EPS_NORM = 1e-9          # lower bound of a tangent vector's norm in the division
EPS_REFLECT = 1e-12      # a reflection vector with squared norm below this is the identity


def _unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), EPS_NORM)


def log_map(mu, x):
    """Log map at the unit vector `mu` of the unit rows `x` on the sphere: tangent vectors, float64 [n, d]."""
    mu = np.asarray(mu, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    dot = x @ mu
    along = x - dot[:, None] * mu[None]               # the part of x orthogonal to mu (zero for x = mu)
    norm = np.maximum(np.linalg.norm(along, axis=1, keepdims=True), EPS_NORM)
    return np.arccos(np.clip(dot, -1.0 + EPS_COS, 1.0 - EPS_COS))[:, None] * along / norm


def subspace(text, rank):
    """(P [d, d] float32, mu [d] float32) from unit text rows [n_t, d]: mu is the normalized mean, P projects onto the
    top `rank` right singular vectors of the centred tangent vectors at mu."""
    text = np.asarray(text)
    assert text.ndim == 2 and text.shape[0] >= 2, text.shape
    assert np.abs(np.linalg.norm(text.astype(np.float64), axis=1) - 1.0).max() < 1e-4, 'text rows must be unit rows'
    assert 1 <= rank <= text.shape[0] - 1, f'rank {rank} needs rank <= n_t - 1 = {text.shape[0] - 1}'
    t64 = text.astype(np.float64)
    mu = _unit(t64.mean(0))
    tangent = log_map(mu, t64)
    tangent = tangent - tangent.mean(0, keepdims=True)
    vt = np.linalg.svd(tangent, full_matrices=False)[2][:rank]
    return (vt.T @ vt).astype(np.float32), mu.astype(np.float32)


def _reflect(x, v):
    """Householder reflection of the rows of `x` in the hyperplane orthogonal to `v` (identity if v is ~0)."""
    n2 = float(v @ v)
    if n2 < EPS_REFLECT:
        return x
    return x - (2.0 / n2) * np.outer(x @ v, v)


def align(images, mu, mean):
    """Rows of `images` turned so that the unit vector `mean` goes to the unit vector `mu`: reflect `mean` to the
    normalized midpoint of mean and mu, then the midpoint to mu. Float64 [n, d]; an isometry of the rows."""
    mu = np.asarray(mu, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    mid = mean + mu
    assert np.linalg.norm(mid) > 1e-6, 'mean image is opposite to mu: the midpoint is undefined'
    mid = mid / np.linalg.norm(mid)
    x = np.asarray(images, dtype=np.float64)
    return _reflect(_reflect(x, mean - mid), mid - mu)


def condition(images, P, mu):
    """Conditioned vectors, float32 [n, d] unit rows. The alignment uses the normalized mean of every row passed
    (the whole scored split, whatever its labels)."""
    images = np.asarray(images)
    assert images.ndim == 2 and images.shape[1] == np.shape(mu)[0] == np.shape(P)[0], (images.shape, np.shape(P))
    x = images.astype(np.float64)
    mean = x.mean(0)
    assert np.isfinite(mean).all() and np.linalg.norm(mean) > 0, 'degenerate mean image'
    mean = mean / np.linalg.norm(mean)
    mu64 = np.asarray(mu, dtype=np.float64)
    y = log_map(mu64, align(x, mu64, mean)) @ np.asarray(P, dtype=np.float64)
    out = _unit(y)
    assert np.isfinite(out).all()
    return out.astype(np.float32)


def text_set(name, aspect):
    """Unit text rows float32 [n_t, 768]: 'A' = r11's lists (50 rows), 'B' = ticket 05's generated lists."""
    assert aspect in op.ASPECTS, aspect
    if name == 'A':
        path = op.CLAY_TEXT_A[aspect]
    else:
        assert name == 'B', name
        path = op.CLAY_B_DIR / f'{aspect}_text.npy'
    assert Path(path).exists(), f'{path}: missing (set B has no file for an aspect that fell back)'
    t = np.load(path)
    assert t.dtype == np.float32 and t.ndim == 2 and t.shape[1] == D, (t.dtype, t.shape)
    return t


def vectors(images, aspect, set_name):
    """Conditioned vectors of `images` [n, 768] for `aspect` with text set `set_name` ('A' or 'B')."""
    t = text_set(set_name, aspect)
    return condition(images, *subspace(t, op.clay_rank(t.shape[0])))


def gate_i(out_dir=None):
    """Gate (i), D23: ours against the official code's reference outputs, largest absolute difference < TOL_CLAY.
    The reference conditions the 2,000 test rows with the 50 emotion-set-A texts at rank 49."""
    out_dir = Path(out_dir) if out_dir is not None else op.OUT
    inp = np.load(op.parity_file('inputs'))
    ref = np.load(op.parity_file('clay'))
    clip, text = inp['clip'], inp['text']
    assert clip.shape == (op.PARITY_N, D) and text.shape == (50, D), (clip.shape, text.shape)
    same_text = bool(np.array_equal(text, text_set('A', 'emotion')))
    P, mu = subspace(text, op.clay_rank(text.shape[0]))
    ours = {'P': P, 'mu': mu, 'query': condition(clip, P, mu), 'gallery': condition(clip, P, mu)}
    max_abs = {k: float(np.abs(ours[k].astype(np.float64) - ref[k].astype(np.float64)).max()) for k in ours}
    ok_shapes = all(ours[k].shape == ref[k].shape for k in ours)
    res = {'pass': bool(same_text and ok_shapes and all(v < op.TOL_CLAY for v in max_abs.values())),
           'tol': op.TOL_CLAY, 'max_abs': max_abs, 'shapes': {k: list(ours[k].shape) for k in ours},
           'ref_shapes': {k: list(ref[k].shape) for k in ours}, 'text_equals_set_a': same_text,
           'rank': op.clay_rank(text.shape[0]), 'dtype': 'float64 math, float32 outputs', 'device': 'cpu'}
    op.write_json(out_dir / 'stage0' / 'gate_i.json', res)
    return res


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == 'gate-i':
        r = gate_i()
        print(r)
        sys.exit(0 if r['pass'] else 1)
    sys.exit('usage: om_clay.py gate-i')
