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
PSHA = eg_paths.P_SHA256


def write_run(out, variant, status='ok', cause=None, n=29232, norm=1.0, finite=True, p_sha=PSHA):
    folder = eg_paths.iii_run(out, variant, 0)
    folder.mkdir(parents=True)
    run = dict(variant=variant, seed=0, switches=V.switches(variant), status=status, cause=cause,
               best_epoch={a: 3 for a in eg_paths.ASPECTS}, p_sha256=p_sha)
    if p_sha is None:
        del run['p_sha256']
    (folder / 'run.json').write_text(json.dumps(run))
    if status == 'ok':
        x = np.zeros((n, 2048), np.float16)
        x[:, 0] = norm
        if not finite:
            x[0, 0] = np.nan
        for a in eg_paths.ASPECTS:
            np.savez(folder / f'{a}__P.npz', img=x)


def test_read_run_ok_failed_declared_and_vg3(tmp_path):
    write_run(tmp_path, 'GT-2')
    r = s3.read_run(tmp_path, 'GT-2', PSHA)
    assert r['status'] == 'ok' and set(r['img']) == {'emotion', 'style', 'genre'} and r['best_epoch']['genre'] == 3
    write_run(tmp_path, 'GT-4', status='failed', cause='non-finite loss at epoch 2')
    r = s3.read_run(tmp_path, 'GT-4', PSHA)
    assert r['status'] == 'failed' and r['img'] is None and 'non-finite' in r['cause']
    write_run(tmp_path, 'GR-2', norm=0.5)                              # fails VG3 (not unit norm)
    with pytest.raises(AssertionError, match='VG3'):
        s3.read_run(tmp_path, 'GR-2', PSHA)
    r = s3.read_run(tmp_path, 'GR-2', PSHA, {'GR-2': 'norm 0.5 after the rerun'})
    assert r['status'] == 'failed' and r['cause'] == 'norm 0.5 after the rerun'
    assert set(r['vg3_problem']) == {'emotion', 'style', 'genre'} and r['img'] is None     # the problem beside the cause
    write_run(tmp_path, 'GA-1', n=100)
    with pytest.raises(AssertionError, match='VG3'):
        s3.read_run(tmp_path, 'GA-1', PSHA)
    write_run(tmp_path, 'GR-0.5', finite=False)
    with pytest.raises(AssertionError, match='VG3'):
        s3.read_run(tmp_path, 'GR-0.5', PSHA)


def test_declared_failure_of_a_run_that_passes_vg3_is_refused(tmp_path):
    write_run(tmp_path, 'GT-2')                                        # exports pass VG3 on all three aspects
    with pytest.raises(AssertionError, match='VG3'):
        s3.read_run(tmp_path, 'GT-2', PSHA, {'GT-2': 'no good reason'})


def test_declared_failure_needs_one_aspect_failing_vg3(tmp_path):
    write_run(tmp_path, 'GT-2')
    folder = eg_paths.iii_run(tmp_path, 'GT-2', 0)
    bad = np.zeros((29232, 2048), np.float16)                          # genre alone is not unit norm
    np.savez(folder / 'genre__P.npz', img=bad)
    with pytest.raises(AssertionError, match='VG3'):
        s3.read_run(tmp_path, 'GT-2', PSHA)
    r = s3.read_run(tmp_path, 'GT-2', PSHA, {'GT-2': 'genre norm 0 after the rerun'})
    assert r['status'] == 'failed' and list(r['vg3_problem']) == ['genre'] and 'norm 0' in r['cause']


def test_run_json_must_carry_the_p_hash_of_12_24(tmp_path):
    write_run(tmp_path, 'GT-2', p_sha=None)
    with pytest.raises(AssertionError, match='p_sha256'):
        s3.read_run(tmp_path, 'GT-2', PSHA)
    write_run(tmp_path, 'GT-4', status='failed', cause='non-finite loss', p_sha=None)
    with pytest.raises(AssertionError, match='p_sha256'):
        s3.read_run(tmp_path, 'GT-4', PSHA)
    write_run(tmp_path, 'GR-2', p_sha='0' * 64)
    with pytest.raises(AssertionError, match='p_sha256'):
        s3.read_run(tmp_path, 'GR-2', PSHA)


def test_failed_option_refuses_a_variant_outside_the_scored_set():
    first = list(V.FIRST)
    assert s3.parse_failed(['GT-2=bad norm'], first) == {'GT-2': 'bad norm'}
    with pytest.raises(AssertionError, match='scored'):
        s3.parse_failed(['GT-6=bad norm'], first)                      # a valid variant, but not run in this stage
    assert s3.parse_failed(['GT-6=bad norm'], first + ['GT-6']) == {'GT-6': 'bad norm'}


def test_scorer_refuses_a_declared_variant_outside_the_variants_it_scores(tmp_path):
    with pytest.raises(AssertionError, match='scored'):
        s3.score_variants(tmp_path, ['GT-2'], 20, 1, {'GT-4': 'x'}, ctx=dict(p_sha=PSHA, labels={}))


def test_bootstrap_cache_identity_names_the_code_it_runs():
    ident = s3._identity('GT-2', 'genre', 'p', 'i')
    assert set(ident['code_sha256']) == {'rb_ami.py', 'rb_pick.py'}


def test_variant_folders_outside_the_allowed_set_stop_the_scorer(tmp_path):
    write_run(tmp_path, 'GT-2')
    write_run(tmp_path, 'R6-5')
    s3.check_variant_dirs(tmp_path, ['GT-2', 'GT-4'])
    write_run(tmp_path, 'GT-6')
    with pytest.raises(AssertionError, match='VG4'):
        s3.check_variant_dirs(tmp_path, ['GT-2', 'GT-4'])
    s3.check_variant_dirs(tmp_path, ['GT-2', 'GT-4', 'GT-6'])
