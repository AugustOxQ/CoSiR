"""Runner guards, provenance and wiring, using temporary git records and small arrays; no real data or GPU."""
import argparse
import json
import subprocess

import pytest

import pf_paths as P
import pf_run

ARGS = argparse.Namespace(device='cpu', workers=1, parts=None, rows=None, max_epochs=None, draws_clu=2000, draws_ret=10000)


def git(repo, *args):
    subprocess.run(['git', '-c', 'user.name=t', '-c', 'user.email=t@t', *args], cwd=repo, check=True,
                   capture_output=True)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    git(tmp_path, 'init', '-q')
    monkeypatch.setattr(P, 'SMOKE', 0)
    monkeypatch.setattr(P, 'SHEET_FILE', tmp_path / 'stage1_sheet.json')
    monkeypatch.setattr(P, 'STAGE2_FILE', tmp_path / 'stage2_records.json')
    monkeypatch.setattr(P, 'OUTPUTS', tmp_path / 'outputs')
    write_gates(tmp_path / 'outputs', True)
    return tmp_path


def write_gates(outputs, ok):
    (outputs / 'stage0a').mkdir(parents=True, exist_ok=True)
    (outputs / 'stage0a/gates.json').write_text(json.dumps({'items': {}, 'ok': ok, 'smoke': P.SMOKE}))


def commit(repo, f):
    git(repo, 'add', f.name)
    git(repo, 'commit', '-q', '-m', f.name)


def test_guard_reason_states(repo):
    f = repo / 'stage1_sheet.json'
    assert 'does not exist' in pf_run.guard_reason(f)
    f.write_text('{}')
    assert 'not in git HEAD' in pf_run.guard_reason(f)
    commit(repo, f)
    assert pf_run.guard_reason(f) == ''
    f.write_text('{"x": 1}')
    assert 'differs from its content in git HEAD' in pf_run.guard_reason(f)


def test_stage2_refuses_without_committed_sheet(repo):
    P.SHEET_FILE.write_text('{}')                                   # present, not committed
    with pytest.raises(SystemExit, match='REFUSED: stage 2: sheet'):
        pf_run.stage2(ARGS)
    commit(repo, P.SHEET_FILE)
    P.SHEET_FILE.write_text('{"edited": 1}')                        # committed, then changed
    with pytest.raises(SystemExit, match='REFUSED: stage 2: sheet'):
        pf_run.stage2(ARGS)


def test_stage2_passes_the_guard_with_a_committed_sheet(repo):
    P.SHEET_FILE.write_text('{}')
    commit(repo, P.SHEET_FILE)
    with pytest.raises(KeyError):                                   # past the guard: the junk sheet has no pick
        pf_run.stage2(ARGS)


def test_stage3_refuses_without_committed_stage2_record(repo):
    P.SHEET_FILE.write_text('{}')
    commit(repo, P.SHEET_FILE)
    P.STAGE2_FILE.write_text('{}')                                  # present, not committed
    with pytest.raises(SystemExit, match='REFUSED: stage 3: stage 2 record'):
        pf_run.stage3(ARGS)
    P.STAGE2_FILE.rename(P.STAGE2_FILE.with_suffix('.aside'))
    with pytest.raises(SystemExit, match='REFUSED: stage 3: stage 2 record'):
        pf_run.stage3(ARGS)


def test_stage3_refuses_without_committed_sheet(repo):
    P.SHEET_FILE.write_text('{}')
    P.STAGE2_FILE.write_text('{}')
    commit(repo, P.STAGE2_FILE)
    with pytest.raises(SystemExit, match='REFUSED: stage 3: sheet'):
        pf_run.stage3(ARGS)


def test_stage3_passes_the_guard_with_both_committed(repo):
    for f in (P.SHEET_FILE, P.STAGE2_FILE):
        f.write_text('{}')
        commit(repo, f)
    with pytest.raises(KeyError):
        pf_run.stage3(ARGS)


def commit_records(repo):
    for f in (P.SHEET_FILE, P.STAGE2_FILE):
        f.write_text('{}')
        commit(repo, f)


@pytest.mark.parametrize('stage', ['1', '2', '3'])
def test_stages_refuse_without_gates_json(repo, stage):
    commit_records(repo)                                            # so only the gates guard can refuse
    path = P.OUTPUTS / 'stage0a/gates.json'
    path.rename(path.with_suffix('.aside'))
    with pytest.raises(SystemExit, match=f'REFUSED: stage {stage}: .*gates.json does not exist'):
        getattr(pf_run, f'stage{stage}')(ARGS)


@pytest.mark.parametrize('stage', ['1', '2', '3'])
def test_stages_refuse_when_gates_not_ok(repo, stage):
    commit_records(repo)
    write_gates(P.OUTPUTS, False)
    with pytest.raises(SystemExit, match=f'REFUSED: stage {stage}: stage 0a did not pass'):
        getattr(pf_run, f'stage{stage}')(ARGS)


@pytest.fixture
def picked_run(repo, monkeypatch):
    import rs_vh
    keys = ['source/fit_a', 'other/fit_b']
    monkeypatch.setattr(pf_run.pf_data, 'fit_keys', lambda: list(keys))
    pick = dict(row=4, settings=dict(P.GRID[4]))
    pf_run.write_json(P.SHEET_FILE, dict(pick=pick))
    commit(repo, P.SHEET_FILE)
    run = dict(status='ok', settings=pick['settings'], cstar_seed=0, train_seed=0,
               fit_keys_sha256=rs_vh.names_sha(keys))
    path = P.OUTPUTS / 'runs/sweep/row4/run.json'
    pf_run.write_json(path, run)
    return path, run


@pytest.mark.parametrize('bad', ['missing', 'status', 'settings', 'fit_keys_sha256'])
def test_stage2_refuses_stale_sheet_before_arrays(picked_run, monkeypatch, bad):
    path, run = picked_run
    if bad == 'missing':
        path.rename(path.with_suffix('.aside'))
    else:
        run[bad] = {'status': 'failed', 'settings': dict(P.GRID[1]), 'fit_keys_sha256': 'stale'}[bad]
        pf_run.write_json(path, run)
    monkeypatch.setattr(pf_run, 'load_arrays', lambda: pytest.fail('loaded arrays before stale-sheet refusal'))
    monkeypatch.setattr(pf_run, 'train', lambda *a: pytest.fail('trained before stale-sheet refusal'))
    with pytest.raises(SystemExit, match='REFUSED: stage 2:.*sweep/row4'):
        pf_run.stage2(ARGS)


@pytest.mark.parametrize('status', ['ok', 'failed'])
def test_train_refuses_finished_run_with_other_fit_keys(picked_run, status):
    path, run = picked_run
    run.update(status=status, fit_keys_sha256='stale')
    pf_run.write_json(path, run)
    with pytest.raises(SystemExit, match='REFUSED:.*fit_keys_sha256'):
        pf_run.train('1', 'sweep/row4', run['settings'], 0, 0, None, {}, {}, ARGS)


@pytest.mark.parametrize('status', ['ok', 'failed'])
def test_train_reuses_finished_run_with_current_fit_keys(picked_run, status):
    path, run = picked_run
    run['status'] = status
    pf_run.write_json(path, run)
    assert pf_run.train('1', 'sweep/row4', run['settings'], 0, 0, None, {}, {}, ARGS) == run


@pytest.fixture
def wiring(picked_run, monkeypatch, repo):
    """Small valid records past the real git guards; spies enforce every orchestration boundary."""
    from types import SimpleNamespace
    import numpy as np
    import rs_vh

    fit_keys = pf_run.pf_data.fit_keys()
    val_keys = ['source/val_a', 'other/val_b']
    monkeypatch.setattr(pf_run, 'N_VAL', len(val_keys))
    monkeypatch.setattr(pf_run.pf_data, 'keys', lambda split: val_keys if split == 'val' else fit_keys + ['source/removed'])
    settings = picked_run[1]['settings']
    ranges = {'clu': {'r': 0.02}, 'map': {'r': 0.03}}
    readout = 'center+k20'
    record = dict(ranges=ranges, readout={'pick': readout},
                  r12_ranges={'clu': {'r': P.R12_R_CLU}, 'map': {'r': P.R12_R_MAP}})
    pf_run.write_json(P.STAGE2_FILE, record)
    commit(repo, P.STAGE2_FILE)
    monkeypatch.setattr(P, 'CSTAR_WEIGHTS', repo / 'cstar')
    for s in range(5):
        pf_run.write_json(P.CSTAR_WEIGHTS / f'emotion__s{s}.json', {'weights_sha256': f'cstar-{s}'})
        run = dict(status='ok', settings=settings, cstar_seed=s, train_seed=100 + s,
                   cstar_weights_sha256=f'cstar-{s}', fit_keys_sha256=rs_vh.names_sha(fit_keys))
        pf_run.write_json(P.OUTPUTS / f'runs/final/s{s}/run.json', run)

    calls = SimpleNamespace(arrays=[], trains=[], exports=[], graphs=[], ranges=0, readouts=0, scored=0, written=[])
    stored = np.array([[0.6, 0.799], [0.799, 0.6]], dtype=np.float16)
    embed = stored.astype(np.float32)
    embed /= np.linalg.norm(embed, axis=1, keepdims=True)

    def photo_arrays(keys):
        calls.arrays.append(list(keys))
        assert keys in (fit_keys, val_keys), 'fit arrays must exclude near-copies'
        return {name: embed.copy() for name in ('I', 'D', 'E', 'T')}

    monkeypatch.setattr(pf_run.pf_data, 'photo_arrays', photo_arrays)
    monkeypatch.setattr(pf_run.pf_graphs, '_folder', lambda *a: repo / 'graph')

    def build(k, correction, fit, val):
        assert (k, correction) == (settings['k'], settings['correction'])
        assert fit['keys'] == fit_keys and val['keys'] == val_keys
        calls.graphs.append('build')

    def load(k, correction, fk, vk):
        assert (k, correction, fk, vk) == (settings['k'], settings['correction'], fit_keys, val_keys)
        calls.graphs.append('load')
        return 'graph-pairs'

    monkeypatch.setattr(pf_run.pf_graphs, 'build', build)
    monkeypatch.setattr(pf_run.pf_graphs, 'load', load)

    def train(stage, run_id, cfg, cstar_seed, train_seed, graphs, fit, val, args):
        calls.trains.append((stage, run_id, dict(cfg), cstar_seed, train_seed))
        assert fit['keys'] == fit_keys and val['keys'] == val_keys
        assert graphs.get(cfg['k'], cfg['correction']) == 'graph-pairs'
        return {'status': 'ok'}

    monkeypatch.setattr(pf_run, 'train', train)
    monkeypatch.setattr(pf_run.pf_export, 'export', lambda *a: calls.exports.append(a))

    def rows(run_id, split, array):
        assert run_id in [f'final/s{s}' for s in range(5)] and split in ('val', 'test')
        return array

    monkeypatch.setattr(pf_run.pf_export, 'stored', lambda run_id, split: rows(run_id, split, stored))
    monkeypatch.setattr(pf_run.pf_export, 'embed', lambda run_id, split: rows(run_id, split, embed))

    def get_ranges(get_stored, get_embed, workers):
        assert workers == ARGS.workers
        for s in range(5):
            np.testing.assert_array_equal(get_stored(s), stored)
            np.testing.assert_array_equal(get_embed(s), embed)
            assert get_stored(s).dtype == np.float16 and get_embed(s).dtype == np.float32
        calls.ranges += 1
        return ranges

    def get_readout(xs, r_map, workers):
        assert r_map == 0.03 and r_map != P.R12_R_MAP and workers == ARGS.workers
        assert set(xs) == set(range(5))
        for x in xs.values():
            np.testing.assert_array_equal(x, embed)
        calls.readouts += 1
        return record['readout']

    def stage2_record(actual_ranges, selected):
        assert actual_ranges == ranges and selected == record['readout']
        return record

    monkeypatch.setattr(pf_run.pf_stage2, 'ranges', get_ranges)
    monkeypatch.setattr(pf_run.pf_stage2, 'readout', get_readout)
    monkeypatch.setattr(pf_run.pf_stage2, 'record', stage2_record)

    def candidate(name, pick, loader):
        assert name == 'photo' and pick == readout
        test_stored = np.stack([loader.stored('test', s) for s in range(5)])
        test_embed = np.stack([loader.embed('test', s) for s in range(5)])
        val_embed = np.stack([loader.embed('val', s) for s in range(5)])
        for x in test_stored:
            np.testing.assert_array_equal(x, stored)
        for x in (*test_embed, *val_embed):
            np.testing.assert_array_equal(x, embed)
            np.testing.assert_allclose(np.linalg.norm(x, axis=1), 1, atol=1e-6)
        assert test_stored.dtype == np.float16 and test_embed.dtype == val_embed.dtype == np.float32
        return SimpleNamespace(readout=pick, val_embed=val_embed, test_stored=test_stored, test_embed=test_embed)

    def score(cand, r, **kwargs):
        assert r == {'clu': 0.02, 'map': 0.03} and cand.readout == readout
        assert kwargs == dict(workers=ARGS.workers, draws_clu=2000, draws_ret=10000)
        calls.scored += 1
        return {'doc': dict(r=r, sc={}, verdict='holds')}

    monkeypatch.setattr(pf_run.pf_stage3, 'candidate_from_data', candidate)
    monkeypatch.setattr(pf_run.pf_stage3, 'stage3', score)
    monkeypatch.setattr(pf_run.pf_stage3, 'write_stage3', calls.written.append)
    return calls


def test_stage2_wiring_after_committed_guard(wiring):
    assert pf_run.stage2(ARGS)['ok']
    assert wiring.trains == [('2', f'final/s{s}', dict(P.GRID[4]), s, 100 + s) for s in range(5)]
    assert wiring.arrays == [pf_run.pf_data.fit_keys(), ['source/val_a', 'other/val_b']]
    assert wiring.graphs == ['build', 'load']
    assert wiring.exports == [(f'final/s{s}', 'val') for s in range(5)]
    assert wiring.ranges == wiring.readouts == 1


def test_stage3_wiring_after_committed_guards(wiring):
    assert pf_run.stage3(ARGS)['ok']
    assert wiring.exports == [(f'final/s{s}', 'test') for s in range(5)]
    assert wiring.scored == 1 and len(wiring.written) == 1
    assert not wiring.arrays and not wiring.trains


@pytest.mark.parametrize('seed', range(5))
@pytest.mark.parametrize('field,value', [('status', 'failed'), ('settings', dict(P.GRID[1])), ('cstar_seed', 9),
                                        ('train_seed', 9), ('cstar_weights_sha256', 'wrong'),
                                        ('fit_keys_sha256', 'stale')])
def test_stage3_refuses_bad_run_before_test_exports(wiring, seed, field, value):
    path = P.OUTPUTS / f'runs/final/s{seed}/run.json'
    run = pf_run.read_json(path)
    run[field] = value
    pf_run.write_json(path, run)
    with pytest.raises((AssertionError, SystemExit)):
        pf_run.stage3(ARGS)
    assert wiring.exports == [] and wiring.scored == 0 and wiring.written == []


def test_stage3_refuses_r12_ranges_before_test_exports(wiring, repo):
    rec = pf_run.read_json(P.STAGE2_FILE)
    rec['ranges'] = rec['r12_ranges']
    pf_run.write_json(P.STAGE2_FILE, rec)
    commit(repo, P.STAGE2_FILE)
    with pytest.raises(AssertionError, match='photo-model ranges'):
        pf_run.stage3(ARGS)
    assert wiring.exports == [] and wiring.scored == 0


def test_stage3_random_start_provenance(wiring, repo):
    sheet = pf_run.read_json(P.SHEET_FILE)
    sheet['pick']['settings'] = dict(P.GRID[8])
    pf_run.write_json(P.SHEET_FILE, sheet)
    commit(repo, P.SHEET_FILE)
    for seed in range(5):
        path = P.OUTPUTS / f'runs/final/s{seed}/run.json'
        run = pf_run.read_json(path)
        run.update(settings=dict(P.GRID[8]), cstar_weights_sha256=None)
        pf_run.write_json(path, run)
    assert pf_run.stage3(ARGS)['ok']


@pytest.mark.parametrize('bad', ['readout', 'ranges'])
def test_stage3_preexport_argument_checks(wiring, bad):
    rec = pf_run.read_json(P.STAGE2_FILE)
    pick = 'none+k0' if bad == 'readout' else rec['readout']['pick']
    r = {'clu': P.R12_R_CLU, 'map': P.R12_R_MAP} if bad == 'ranges' else {'clu': 0.02, 'map': 0.03}
    with pytest.raises(AssertionError, match='read-out|photo-model ranges'):
        pf_run.require_final_runs(pf_run.read_json(P.SHEET_FILE), rec, pick, r)
    assert wiring.exports == []


@pytest.mark.parametrize('stage,flag,value', [('1', '--rows', '1,8'), ('2', '--max-epochs', '2'),
                                             ('3', '--draws-clu', '200'), ('3', '--draws-ret', '500')])
def test_main_refuses_smoke_flags_in_full_mode(repo, monkeypatch, stage, flag, value):
    # A spy makes removing the parser refusal observable even when later stage guards would refuse as well.
    called = []
    monkeypatch.setattr(pf_run, f'stage{stage}', lambda args: called.append(args) or dict(ok=True, summary='ran'))
    with pytest.raises(SystemExit) as exc:
        pf_run.main([stage, flag, value])
    assert exc.value.code == 2 and not called


@pytest.mark.parametrize('stage', ['1', '2', '3'])
@pytest.mark.parametrize('current,recorded', [(0, 2000), (2000, 0), (2000, 123), (0, None)])
def test_stages_refuse_gates_from_another_smoke_setting(repo, monkeypatch, stage, current, recorded):
    monkeypatch.setattr(P, 'SMOKE', current)
    path = P.OUTPUTS / 'stage0a/gates.json'
    pf_run.write_json(path, dict(ok=True, smoke=recorded))
    with pytest.raises(SystemExit, match=f'REFUSED: stage {stage}:.*smoke'):
        getattr(pf_run, f'stage{stage}')(ARGS)


@pytest.mark.parametrize('smoke', [0, 2000])
def test_require_gates_accepts_matching_smoke(repo, monkeypatch, smoke):
    monkeypatch.setattr(P, 'SMOKE', smoke)
    pf_run.write_json(P.OUTPUTS / 'stage0a/gates.json', dict(ok=True, smoke=smoke))
    pf_run.require_gates('2')


def test_stage_crash_traceback_is_saved_in_stage_log(repo, monkeypatch, capsys):
    import sys

    def crash(args):
        print('before crash stdout')
        print('before crash stderr', file=sys.stderr)
        raise RuntimeError('stage regression sentinel')

    monkeypatch.setattr(pf_run, 'stage2', crash)
    streams = sys.stdout, sys.stderr
    with pytest.raises(RuntimeError, match='stage regression sentinel'):
        pf_run.main(['2'])
    assert (sys.stdout, sys.stderr) == streams
    log = (P.OUTPUTS / 'logs/2.log').read_text()
    assert 'Traceback (most recent call last):' in log
    assert 'in crash' in log and 'RuntimeError: stage regression sentinel' in log
    assert 'stage 2 FAILED' in log
    assert 'before crash stdout' in log and 'before crash stderr' in log
    captured = capsys.readouterr()
    assert 'before crash stdout' in captured.out and 'before crash stderr' in captured.err
    print('after restoration')
    assert 'after restoration' not in (P.OUTPUTS / 'logs/2.log').read_text()
