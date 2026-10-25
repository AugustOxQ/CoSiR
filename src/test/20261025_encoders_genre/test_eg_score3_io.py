"""Stage iii scorer: effects against R6-5 and reading a run, on small synthetic inputs."""
import json

import numpy as np
import pytest

import eg_paths
import eg_score3 as s3
import eg_variants as V
import rb_pick


def synthetic(n=240, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 4, size=n)
    y[rng.random(n) < 0.2] = -1                                       # masked rows
    informative = np.where(y >= 0, y, 0)

    def part(noise):
        return np.stack([np.where(rng.random(n) < noise, rng.integers(0, 5, size=n), informative)
                         for _ in range(3)]).astype(np.int64)
    return y.astype(np.int64), {'REF': part(0.6), 'GOOD': part(0.1), 'BAD': part(1.0)}


# ------------------------------------------------------------------ effects against R6-5
def test_effects_match_the_rb_function_when_the_reference_is_its_own():
    y, parts = synthetic()
    parts = {'R0-5': parts['REF'], 'GOOD': parts['GOOD'], 'BAD': parts['BAD']}
    mine = s3.aspect_effects(y, parts, 'R0-5', n_draws=40, workers=1)
    theirs = rb_pick.aspect_effects(y, parts, 40, 1)
    for n in parts:
        for k in ('A', 'e', 'lo', 'hi'):
            assert mine[n][k] == pytest.approx(theirs[n][k], abs=1e-12), (n, k)


def test_effects_use_the_named_reference():
    y, parts = synthetic()
    eff = s3.aspect_effects(y, parts, 'REF', n_draws=40, workers=1)
    assert eff['REF']['e'] == 0 and eff['REF']['lo'] == 0 and eff['REF']['hi'] == 0
    assert eff['GOOD']['e'] == pytest.approx(eff['GOOD']['A'] - eff['REF']['A'])
    assert eff['GOOD']['e'] > 0 > eff['BAD']['e']
    assert eff['GOOD']['lo'] > 0 > eff['BAD']['hi']


def test_cached_bootstrap_scores_are_reused_and_arms_are_independent(monkeypatch):
    y, parts = synthetic()
    store = {}

    class Cache:
        def get(self, n):
            return store.get(n)

        def put(self, n, b):
            store[n] = b
    first = s3.aspect_effects(y, parts, 'REF', n_draws=30, workers=1, cache=Cache())
    assert set(store) == set(parts)
    with monkeypatch.context() as m:
        m.setattr(rb_pick, 'bootstrap_scores', lambda *a, **k: pytest.fail('bootstrap recomputed'))
        assert s3.aspect_effects(y, parts, 'REF', n_draws=30, workers=1, cache=Cache()) == first
    del store['GOOD']                  # one arm computed alone gives the numbers it had when computed with the others
    assert s3.aspect_effects(y, parts, 'REF', n_draws=30, workers=1, cache=Cache()) == first


# ------------------------------------------------------------------ reading a run (VG3, failed runs, VG4 folders)
def write_run(out, variant, status='ok', cause=None, n=29232, norm=1.0, finite=True):
    folder = eg_paths.iii_run(out, variant, 0)
    folder.mkdir(parents=True)
    (folder / 'run.json').write_text(json.dumps(dict(variant=variant, seed=0, switches=V.switches(variant), status=status,
                                                     cause=cause, best_epoch={a: 3 for a in eg_paths.ASPECTS})))
    if status == 'ok':
        x = np.zeros((n, 2048), np.float16)
        x[:, 0] = norm
        if not finite:
            x[0, 0] = np.nan
        for a in eg_paths.ASPECTS:
            np.savez(folder / f'{a}__P.npz', img=x)


def test_read_run_ok_failed_declared_and_vg3(tmp_path):
    write_run(tmp_path, 'GT-2')
    r = s3.read_run(tmp_path, 'GT-2', 'sha')
    assert r['status'] == 'ok' and set(r['img']) == {'emotion', 'style', 'genre'} and r['best_epoch']['genre'] == 3
    write_run(tmp_path, 'GT-4', status='failed', cause='non-finite loss at epoch 2')
    r = s3.read_run(tmp_path, 'GT-4', 'sha')
    assert r['status'] == 'failed' and r['img'] is None and 'non-finite' in r['cause']
    write_run(tmp_path, 'GR-2', norm=0.5)                              # fails VG3 (not unit norm)
    with pytest.raises(AssertionError, match='VG3'):
        s3.read_run(tmp_path, 'GR-2', 'sha')
    r = s3.read_run(tmp_path, 'GR-2', 'sha', {'GR-2': 'norm 0.5 after the rerun'})
    assert r['status'] == 'failed' and r['cause'] == 'norm 0.5 after the rerun'
    write_run(tmp_path, 'GA-1', n=100)
    with pytest.raises(AssertionError, match='VG3'):
        s3.read_run(tmp_path, 'GA-1', 'sha')
    write_run(tmp_path, 'GR-0.5', finite=False)
    with pytest.raises(AssertionError, match='VG3'):
        s3.read_run(tmp_path, 'GR-0.5', 'sha')


def test_variant_folders_outside_the_allowed_set_stop_the_scorer(tmp_path):
    write_run(tmp_path, 'GT-2')
    write_run(tmp_path, 'R6-5')
    s3.check_variant_dirs(tmp_path, ['GT-2', 'GT-4'])
    write_run(tmp_path, 'GT-6')
    with pytest.raises(AssertionError, match='VG4'):
        s3.check_variant_dirs(tmp_path, ['GT-2', 'GT-4'])
    s3.check_variant_dirs(tmp_path, ['GT-2', 'GT-4', 'GT-6'])
