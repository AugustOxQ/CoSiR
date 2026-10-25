"""End-to-end test of rb_probe on small synthetic data."""
import json

import numpy as np

import rb_probe


def unit(x):
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def make(n, w, rng, centers, nonlin=True):
    y = rng.integers(0, 3, n)
    default = unit(centers[y] + 0.8 * rng.normal(size=(n, w)))
    # the instructed image is a (nonlinear) function of the default image: the information is present
    instr = unit(default @ np.eye(w)[:, ::-1] + (0.5 * np.tanh(3 * default) if nonlin else 0))
    return default, instr, y.astype(np.int64)


def test_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    rng = np.random.default_rng(0)
    w = 16
    centers = rng.normal(size=(3, w)) * 2
    xf, yf, _ = make(600, w, rng, centers)
    xs, ys, _ = make(150, w, rng, centers)
    sets, ref = {}, None
    for name, n in (('P', 400), ('old5000', 300)):
        d, i, y = make(n, w, rng, centers)
        lab = {a: np.where(rng.random(n) < 0.1, -1, y).astype(np.int64) for a in rb_probe.ASPECTS}
        sets[name] = dict(default=d, instr={a: i for a in rb_probe.ASPECTS}, labels=lab)
    # R0-5 export exists for P only: old5000 must come out "missing"
    folder = tmp_path / 'part1' / 'train' / 'R0-5'
    folder.mkdir(parents=True)
    for a in rb_probe.ASPECTS:
        np.savez(folder / f'{a}__s0__P.npz', img=sets['P']['instr'][a].astype(np.float16))
    fit = {a: (xf, yf) for a in rb_probe.ASPECTS}
    stop = {a: (xs, ys) for a in rb_probe.ASPECTS}
    monkeypatch.setattr(rb_probe, "MAX_EPOCHS", 30)
    probe = rb_probe.run(fit, stop, sets, tmp_path, smoke=False)
    path = rb_probe.write_probe(tmp_path, probe)
    back = json.loads(path.read_text())
    assert (tmp_path / 'part1' / 'probe.md').exists()
    for a in rb_probe.ASPECTS:
        r = back['aspects'][a]
        assert r['mlp']['best_epoch'] >= 1 and r['ridge']['alpha_over_n'] in rb_probe.RIDGE_GRID
        P, O = r['ami']['P'], r['ami']['old5000']
        assert O['R0-5'] is None and P['R0-5'] is not None
        assert abs(P['R0-5'] - P['instructed']) < 0.02  # same vectors up to float16
        assert P['labelled'] == int((sets['P']['labels'][a] >= 0).sum())
        # the mapping is learnable: both probes beat the loss of predicting the mean (standardized target loss ~1)
        assert r['mlp']['stop_loss'] < 0.6 and r['ridge']['stop_loss'] < 0.6
        for s in (P, O):
            assert 0 <= s['default'] <= 1 and 0 <= s['mlp'] <= 1


def test_ridge_recovers_linear_map():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(200, 8)).astype(np.float32)
    y = (x @ rng.normal(size=(8, 5))).astype(np.float32)
    ws = rb_probe.ridge_fit(x - x.mean(0), y, grid=(1e-8,))
    pred = rb_probe.ridge_predict(ws[1e-8], x - x.mean(0))
    assert np.abs(pred - y).max() < 1e-2
