"""Tests for pf_run (ticket 10): only the held guard. Stage 2 refuses without a committed sheet, stage 3 without a
committed stage 2 record (and sheet). The records live in a temporary git repo; `pf_paths.SHEET_FILE`,
`STAGE2_FILE` and `SMOKE` are monkeypatched (the stages read them at call time). The files hold junk ('{}'), so a
stage whose guard was removed fails with a KeyError right after the guard, before any data, run or export is touched."""
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
    (outputs / 'stage0a/gates.json').write_text(json.dumps({'items': {}, 'ok': ok}))


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
    P.STAGE2_FILE.unlink()
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
    (P.OUTPUTS / 'stage0a/gates.json').unlink()
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
