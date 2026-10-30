"""Tests of ac_run's guard (spec 3.5: no Affection test array and no ArtELingo held array before the ranges and picks
are committed) and of the 0a stop rule. A temporary git repo holds the records; every data loader is patched to raise,
so a stage that loads data before the guard fails the test by a different error than the guard's SystemExit."""
import json
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

import ac_paths
import ac_data
import ac_run
import rt_data

RANGES = {k: {'per_seed': [0.1, 0.2, 0.3, 0.4, 0.5], 'r': 0.4} for k in ('clu', 'map', 'ndcg')}
PICKS = {'affection': {}, 'artelingo': {}}
ARGS = Namespace(workers=1, draws_clu=2, draws_ret=2, smoke=None)


def git(repo, *a):
    subprocess.run(['git', '-C', str(repo), '-c', 'user.name=t', '-c', 'user.email=t@t', *a], check=True,
                   capture_output=True)


@pytest.fixture
def world(tmp_path, monkeypatch):
    repo, out = tmp_path / 'round', tmp_path / 'out'
    repo.mkdir()
    git(repo, 'init', '-q')
    monkeypatch.setattr(ac_paths, 'SMOKE', 0, raising=False)
    monkeypatch.setattr(ac_paths, 'OUTPUTS', out)
    monkeypatch.setattr(ac_paths, 'RANGES_FILE', repo / 'stage0b_ranges.json')
    monkeypatch.setattr(ac_paths, 'PICKS_FILE', repo / 'stage1_picks.json')
    loaded = []

    def boom(name):
        def f(*a, **k):
            loaded.append(name)
            raise RuntimeError(f'data loaded before the guard: {name}')
        return f
    for mod, names in ((ac_data, ('labels', 'keys', 'embed', 'emotion_dist', 'percept_prob', 'artelingo_crl_qwen')),
                       (rt_data, ('labels', 'embed'))):
        for n in names:
            monkeypatch.setattr(mod, n, boom(f'{mod.__name__}.{n}'))
    monkeypatch.setattr(ac_data, 'smoke_rows', boom('smoke_rows'), raising=False)
    for kind, doc, rel in (('ranges', RANGES, 'stage0b/ranges.json'), ('picks', PICKS, 'stage1/picks.json')):
        text = json.dumps(doc, indent=1) + '\n'
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        (out / rel).write_text(text)
        ac_run.record_copy(kind).write_text(text)
    git(repo, 'add', '-A')
    git(repo, 'commit', '-qm', 'records')
    return dict(repo=repo, out=out, loaded=loaded)


def break_record(w, kind, defect):
    f, out = ac_run.record_copy(kind), ac_run.record_output(kind)
    if defect == 'missing':
        f.unlink()
    elif defect == 'uncommitted':
        git(w['repo'], 'rm', '-q', '--cached', f.name)          # on disk and in the outputs, not in HEAD
        git(w['repo'], 'commit', '-qm', 'untrack')
    elif defect == 'changed_from_outputs':
        f.write_text(f.read_text() + ' ')
    elif defect == 'changed_from_head':           # outputs and record agree, HEAD holds something else
        f.write_text(f.read_text() + ' ')
        out.write_text(f.read_text())
    else:
        raise AssertionError(defect)


def test_clean_records_pass_the_guard(world):
    assert ac_run.guard_reason('ranges') == '' and ac_run.guard_reason('picks') == ''


DEFECTS = ('missing', 'uncommitted', 'changed_from_outputs', 'changed_from_head')
CASES = [(stage, kind, d) for stage, kinds in (('1', ('ranges',)), ('2', ('ranges', 'picks')), ('3', ('ranges', 'picks')))
         for kind in kinds for d in DEFECTS]


@pytest.mark.parametrize('stage,kind,defect', CASES)
def test_a_bad_record_stops_the_stage_before_any_data_is_loaded(world, stage, kind, defect):
    break_record(world, kind, defect)
    with pytest.raises(SystemExit) as e:
        getattr(ac_run, f'stage{stage}')(ARGS)
    assert 'REFUSED' in str(e.value) and kind in str(e.value)
    assert world['loaded'] == []


def test_smoke_guard_checks_the_copy_against_the_outputs_and_skips_git(world, monkeypatch, capsys):
    monkeypatch.setattr(ac_paths, 'SMOKE', 5)
    assert ac_run.record_copy('ranges') == world['out'] / 'records/stage0b_ranges.json'
    assert 'does not exist' in ac_run.guard_reason('ranges')
    rec = ac_run.record_copy('ranges')
    rec.parent.mkdir(parents=True)
    rec.write_text((world['out'] / 'stage0b/ranges.json').read_text())
    assert ac_run.guard_reason('ranges') == ''                       # not in git: the git part is skipped
    assert 'skipped' in capsys.readouterr().out
    rec.write_text('{}\n')
    assert 'differs' in ac_run.guard_reason('ranges')


def test_stage_0b_refuses_without_a_passing_gates_file(world):
    with pytest.raises(SystemExit):
        ac_run.stage0b(ARGS)
    ac_run.write_json(world['out'] / 'stage0a/gates.json', {'items': {}, 'ok': False, 'smoke': 0})
    with pytest.raises(SystemExit):
        ac_run.stage0b(ARGS)
    assert world['loaded'] == []


def test_cli_stage_0a_with_empty_outputs_exits_nonzero_naming_missing_items_and_runs_no_replay(tmp_path):
    out = tmp_path / 'out'
    env = dict(os.environ, AC_OUTPUTS=str(out), PYTHONDONTWRITEBYTECODE='1')
    env.pop('AC_SMOKE', None)
    r = subprocess.run([sys.executable, str(Path(ac_run.__file__)), '--stage', '0a'], env=env, cwd=tmp_path,
                       capture_output=True, text=True, timeout=200)
    assert r.returncode == 1, r.stdout + r.stderr
    gates = json.loads((out / 'stage0a/gates.json').read_text())
    assert gates['ok'] is False
    for name in ('0a(i)_store', '0a(iv)_cstar_reinfer', '0a(v)_percept_loader', '0a(vii)_crl'):
        assert name in gates['missing'] and name in r.stderr
    for name in ('0a(iii)_r10_stage4_emotion', '0a(iii)_r11_stage4', '0a(iii)_r11_stage5'):
        assert gates['items'][name]['value'].startswith('not run')       # replays come after the item check


def test_smoke_never_writes_the_committed_record_files(world, monkeypatch):
    import ac_select
    import ac_stats
    monkeypatch.setattr(ac_paths, 'SMOKE', 5)
    for kind in ('ranges', 'picks'):                       # the runner's own copies are redirected and pass
        assert ac_run.record_copy(kind).parent == world['out'] / 'records'
    ranges = {k: dict(v, per_seed=[0.1, 0.2, 0.3, 0.4, 0.5], r=0.4) for k, v in RANGES.items()}
    before = {f.name: f.read_bytes() for f in world['repo'].glob('*.json')}
    with pytest.raises(AssertionError, match='committed record'):
        ac_stats.write_ranges(ranges, world['out'] / 'x', copy_to=ac_paths.RANGES_FILE)
    with pytest.raises(AssertionError, match='committed record'):
        ac_select.write_picks(PICKS, {}, world['out'] / 'x', copy_to=ac_paths.PICKS_FILE)
    assert {f.name: f.read_bytes() for f in world['repo'].glob('*.json')} == before and not (world['out'] / 'x').exists()
    monkeypatch.setattr(ac_paths, 'SMOKE', 0)
    ac_paths.check_record_write(ac_paths.PICKS_FILE)       # a full run may write them
