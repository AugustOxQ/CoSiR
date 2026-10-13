"""The per-read agreement file carries what the apply-rule step (ticket 19) reads: phase, smoke, pass_file,
held_pass_sha256, rederived_counts, next to ticket 25's own agreement/compared/failed/items. Small synthetic passes."""
import copy
import hashlib
import json
import sys
import numpy as np
import pytest
import rd9_compare as C
import rd9_inputs
import rd9_stats as S

ARMS = ('aff_fused', 'aff_cf', 'cosine', 'rca', 'B', 'B0')
METRICS = ('r1', 'gain', 'other', 'swap', 'strict')


def make_arrays(a28=False):    # a28: level 1 passes, so level 2 is a real object
    n = 40
    rng = np.random.default_rng(3)
    arrays = {'cl': np.repeat(np.arange(10), 4), 'pair_index': np.tile(np.arange(4), 10), 'seed_index': np.zeros(n, int)}
    for arm in ARMS:
        for m in METRICS:
            arrays[f'{arm}__{m}'] = np.zeros(n)
    arrays['aff_fused__r1'] = rng.integers(0, 3, n) / 4
    arrays['aff_fused__gain'] = rng.integers(0, 3, n) / 4
    for arm, hi in (('cosine', 2), ('rca', 3), ('B', 4), ('B0', 1 if a28 else 3), ('aff_cf', 3)):    # other counts per check
        arrays[f'{arm}__r1'] = rng.integers(0, hi, n) / 4
    arrays['rca__gain'] = rng.integers(0, 2, n) / 4
    return arrays


def make_stats(a28=False):
    stats = S.pass_record(make_arrays(a28), a28=a28)
    stats.update(seeds=[55, 56, 57] if a28 else [52, 53, 54], episodes_sha256={'x': 'a' * 64})
    return stats


def write(path, value):
    path.write_text(json.dumps(rd9_inputs.jsonable(value)))
    return path


@pytest.fixture
def work(tmp_path, monkeypatch):
    monkeypatch.setattr(rd9_inputs, 'RD', tmp_path)   # outputs are allowed only below RD
    return tmp_path


def stage(work, mode='held', a28=False, read='H6', name='held_pass_H6.json', tweak=None):
    stats = make_stats(a28)
    own = write(work / 'own.json', dict(read=read, backbone='clip_b32', stats=stats))
    arrays = make_arrays(a28)
    np.savez(work / 'own.npz', **arrays)
    np.savez(work / 'rep.npz', **arrays)
    reported = copy.deepcopy(stats)
    if mode is not None: reported['mode'] = mode
    if tweak: tweak(reported)
    rep = write(work / name, reported)
    return stats, own, rep


def run(work, own, rep, monkeypatch, out='agr.json', extra=()):
    argv = ['rd9_compare.py', '--derived', str(own), '--derived-arrays', str(work / 'own.npz'), '--reported', str(rep),
            '--reported-arrays', str(work / 'rep.npz'), '--out', str(work / out), *extra]
    monkeypatch.setattr(sys, 'argv', argv)
    C.main()
    return json.loads((work / out).read_text())


def test_schema_and_values(work, monkeypatch):
    stats, own, rep = stage(work)
    got = run(work, own, rep, monkeypatch)
    assert set(got) == {'agreement', 'compared', 'failed', 'items', 'independent_identity', 'read', 'backbone',
                        'phase', 'smoke', 'pass_file', 'held_pass_sha256', 'rederived_counts'}
    assert got['agreement'] is True and got['failed'] == 0 and got['compared'] == len(got['items']) >= 1
    assert got['phase'] == 2 and type(got['phase']) is int
    assert got['smoke'] is False
    assert got['read'] == 'H6'
    assert got['pass_file'] == 'held_pass_H6.json'
    assert got['held_pass_sha256'] == hashlib.sha256(rep.read_bytes()).hexdigest()
    assert list(got['rederived_counts']) == ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7']
    assert got['rederived_counts'] == {j: stats['checks'][j]['n'] for j in S.CHECK_IDS}
    assert all(type(v) is int for v in got['rederived_counts'].values())


def test_smoke_comes_from_the_pass_files_mode(work, monkeypatch):
    _, own, rep = stage(work, mode='smoke')
    assert run(work, own, rep, monkeypatch)['smoke'] is True


def test_two_level_read_has_the_same_keys(work, monkeypatch):
    stats, own, rep = stage(work, a28=True, read='H8', name='held_pass_H8_fix1.json')
    got = run(work, own, rep, monkeypatch)
    assert got['pass_file'] == 'held_pass_H8_fix1.json' and got['read'] == 'H8'
    assert got['rederived_counts'] == {j: stats['checks'][j]['n'] for j in S.CHECK_IDS}


def test_hash_is_of_the_file_bytes(work, monkeypatch):
    _, own, rep = stage(work)
    rep.write_bytes(rep.read_bytes() + b'\n  ')           # the same JSON, other bytes
    got = run(work, own, rep, monkeypatch)
    assert got['held_pass_sha256'] == hashlib.sha256(rep.read_bytes()).hexdigest()


def test_counts_are_the_independent_passes_not_the_reported(work, monkeypatch):
    stats, own, rep = stage(work, tweak=lambda r: r['checks']['P3'].update(n=r['checks']['P3']['n'] + 1))
    with pytest.raises(SystemExit) as e:
        run(work, own, rep, monkeypatch)
    assert e.value.code == 1
    got = json.loads((work / 'agr.json').read_text())       # a disagreement is still written, with provenance
    assert got['agreement'] is False and got['failed'] >= 1
    assert got['rederived_counts']['P3'] == stats['checks']['P3']['n']
    assert got['held_pass_sha256'] == hashlib.sha256(rep.read_bytes()).hexdigest()


@pytest.mark.parametrize('mode', [None, 'held ', 'Held', 'dev', True, 1])
def test_missing_or_unknown_mode_crashes(work, monkeypatch, mode):
    _, own, rep = stage(work, mode=None, tweak=None if mode is None else (lambda r: r.update(mode=mode)))
    with pytest.raises(ValueError, match='mode'):
        run(work, own, rep, monkeypatch)
    assert not (work / 'agr.json').exists()


def test_mode_inside_a_stats_wrapper_is_read(work, monkeypatch):
    stats, own, rep = stage(work, mode=None)
    write(rep, dict(stats=dict(json.loads(rep.read_text()), mode='smoke')))
    assert run(work, own, rep, monkeypatch)['smoke'] is True


def test_missing_reported_inputs_crash(work, monkeypatch):
    _, own, rep = stage(work)
    with pytest.raises(AssertionError, match='runner scalars AND arrays'):
        monkeypatch.setattr(sys, 'argv', ['x', '--derived', str(own), '--derived-arrays', str(work / 'own.npz'),
                                          '--out', str(work / 'agr.json')])
        C.main()
    with pytest.raises(FileNotFoundError):
        run(work, own, work / 'no_such_pass.json', monkeypatch)
    assert not (work / 'agr.json').exists()


@pytest.mark.parametrize('bad', ['drop', 'float', 'bool', 'extra'])
def test_derived_counts_must_be_seven_integers(work, monkeypatch, bad):
    stats, own, rep = stage(work)
    record = json.loads(own.read_text())
    n = record['stats']['checks']
    if bad == 'drop': del n['P7']
    elif bad == 'float': n['P2']['n'] = 3.0
    elif bad == 'bool': n['P2']['n'] = True
    else: n['P8'] = dict(n['P1'])
    write(own, record)
    with pytest.raises((ValueError, KeyError, AssertionError)):
        run(work, own, rep, monkeypatch)
    assert not (work / 'agr.json').exists()
