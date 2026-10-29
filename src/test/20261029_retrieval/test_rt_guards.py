"""The held-data guards: rt_run.guard on a real temporary git repo, the committed-ranges comparison, the two
validation-only data gates (rt_run.Loader, rt_stage0.Inputs) and the rerun-stable RANGES_FILE. No real data is read."""
import copy
import json
import subprocess
from types import SimpleNamespace

import pytest

import rt_data
import rt_paths
import rt_run
import rt_stage0


def _doc(seconds=1.0):
    """A stage 0a doc as stage0a writes it to <outputs>/stage0a/ranges.json (the timings are the part a rerun changes)."""
    return dict(map={a: dict(per_seed=[0.1, 0.2], r=0.1) for a in rt_paths.ASPECTS},
                ndcg={'emotion': dict(per_seed=[0.3, 0.4], r=0.1)},
                inputs={'labels/val/emotion': 'abc'}, smoke=None,
                seed_seconds={a: [seconds] * 2 for a in rt_paths.ASPECTS}, total_seconds=seconds * 7)


def _dump(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + '\n')


def _git(repo, *cmd):
    subprocess.run(['git', *cmd], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A git repo holding one commit (no ranges yet), RANGES_FILE inside it, and an outputs folder with a stage 0a doc."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    _git(repo, 'init', '-q')
    _git(repo, 'config', 'user.name', 'test')
    _git(repo, 'config', 'user.email', 'test@example.com')
    (repo / 'other.txt').write_text('x\n')
    _git(repo, 'add', 'other.txt')
    _git(repo, 'commit', '-q', '-m', 'first')
    ranges = repo / 'stage0a_ranges.json'
    monkeypatch.setattr(rt_paths, 'RANGES_FILE', ranges)
    outputs = tmp_path / 'out'
    _dump(outputs / 'stage0a/ranges.json', _doc())
    args = SimpleNamespace(outputs=outputs, smoke=None, arms=())
    return SimpleNamespace(repo=repo, ranges=ranges, outputs=outputs, args=args)


def _commit_ranges(e, doc=None):
    _dump(e.ranges, rt_paths.committed_ranges(_doc() if doc is None else doc))
    _git(e.repo, 'add', e.ranges.name)
    _git(e.repo, 'commit', '-q', '-m', 'ranges')


# ---- guard ----------------------------------------------------------------------------------------------------------

def test_guard_allows_committed_equal_ranges(env):
    _commit_ranges(env)
    assert rt_run.guard(env.args) == ''


def test_guard_allows_when_only_the_timings_differ(env):
    """F1: a rerun of 0a gives other wall-clock times; the committed keys are the same, so the guard allows."""
    _commit_ranges(env)
    _dump(env.outputs / 'stage0a/ranges.json', _doc(seconds=99.0))
    assert rt_run.guard(env.args) == ''


def test_guard_refuses_missing_outputs_ranges(env):
    _commit_ranges(env)
    (env.outputs / 'stage0a/ranges.json').unlink()
    assert 'is missing' in rt_run.guard(env.args)


def test_guard_refuses_missing_ranges_file(env):
    assert 'does not exist' in rt_run.guard(env.args)


def test_guard_refuses_untracked_ranges_file(env):
    _dump(env.ranges, rt_paths.committed_ranges(_doc()))
    assert 'not tracked' in rt_run.guard(env.args)


def test_guard_refuses_staged_but_not_committed(env):
    _dump(env.ranges, rt_paths.committed_ranges(_doc()))
    _git(env.repo, 'add', env.ranges.name)
    assert 'differs from HEAD' in rt_run.guard(env.args)


def test_guard_refuses_committed_but_changed_in_the_working_copy(env):
    _commit_ranges(env)
    changed = _doc()
    changed['map']['emotion']['r'] = 0.5
    _dump(env.ranges, rt_paths.committed_ranges(changed))
    _dump(env.outputs / 'stage0a/ranges.json', changed)          # equal to the working copy, not to HEAD
    assert 'differs from HEAD' in rt_run.guard(env.args)


def test_guard_refuses_committed_but_different_from_outputs(env):
    _commit_ranges(env)
    other = _doc()
    other['map']['style']['r'] = 0.5
    _dump(env.outputs / 'stage0a/ranges.json', other)
    assert 'differ' in rt_run.guard(env.args)


def test_guard_refuses_stage5_without_picks(env):
    _commit_ranges(env)
    assert 'picks.json' in rt_run.guard(env.args, need_picks=True)
    _dump(env.outputs / 'stage4/picks.json', {})
    assert rt_run.guard(env.args, need_picks=True) == ''


def test_guard_smoke_skips_the_git_checks_only(env):
    env.args.smoke = 100                                          # RANGES_FILE is missing, untracked, nothing committed
    assert rt_run.guard(env.args) == ''
    assert 'picks.json' in rt_run.guard(env.args, need_picks=True)
    (env.outputs / 'stage0a/ranges.json').unlink()
    assert 'is missing' in rt_run.guard(env.args)


# ---- the committed-ranges comparison --------------------------------------------------------------------------------

def test_ranges_match_equal_and_unequal(env):
    path = env.outputs / 'stage0a/ranges.json'
    _commit_ranges(env)
    assert rt_paths.ranges_match(path)
    _dump(path, _doc(seconds=5.0))                                # timings are not compared
    assert rt_paths.ranges_match(path)
    for key, edit in (('map', lambda d: d['map']['genre'].update(r=0.2)),
                      ('ndcg', lambda d: d['ndcg']['emotion'].update(r=0.2)),
                      ('inputs', lambda d: d['inputs'].update(extra='z')),
                      ('smoke', lambda d: d.update(smoke=100))):
        other = copy.deepcopy(_doc())
        edit(other)
        _dump(path, other)
        assert not rt_paths.ranges_match(path), key


def test_stage4_full_mode_asserts_the_same_comparison(env, monkeypatch):
    def stop(*a, **k):
        raise RuntimeError('data read reached')
    monkeypatch.setattr(rt_data, 'labels', stop)
    _commit_ranges(env)
    with pytest.raises(RuntimeError, match='data read reached'):          # equal: the assert passes, data is next
        rt_run.stage4(env.args)
    other = _doc()
    other['ndcg']['emotion']['r'] = 0.5
    _dump(env.outputs / 'stage0a/ranges.json', other)
    with pytest.raises(AssertionError, match='RANGES_FILE differs'):
        rt_run.stage4(env.args)


# ---- the validation-only data gates ---------------------------------------------------------------------------------

def _no_store(monkeypatch):
    def stop(*a, **k):
        raise RuntimeError('store read')
    for name in ('labels', 'emotion_dist', 'embed', 'raw', 'names'):
        monkeypatch.setattr(rt_data, name, stop)


def test_loader_refuses_held_reads_before_any_store_read(monkeypatch):
    _no_store(monkeypatch)
    loader = rt_run.Loader(None, ('val',))
    for call in (lambda: loader.labels('held'), lambda: loader.emotion_dist('held'),
                 lambda: loader.embed('held', 'cstar', 'emotion', 0)):
        with pytest.raises(AssertionError, match='may read only'):
            call()
    with pytest.raises(RuntimeError, match='store read'):         # the gate is the only thing that stops a held read
        loader.labels('val')


def test_inputs_validation_only_refuses_held_reads_before_any_store_read(monkeypatch):
    _no_store(monkeypatch)
    data = rt_stage0.Inputs(validation_only=True)
    for args in (('labels', 'held'), ('names', 'held'), ('embed', 'held', 'cstar', 'emotion', 0)):
        with pytest.raises(AssertionError, match='validation only'):
            data.read(*args)
    with pytest.raises(RuntimeError, match='store read'):
        data.read('labels', 'val')


# ---- stage 0a and the committed file --------------------------------------------------------------------------------

def test_stage0a_writes_a_byte_identical_ranges_file_on_a_rerun(env, monkeypatch):
    class FakeInputs:
        def __init__(self, smoke, validation_only):
            self.hashes = {'names/val': 'h'}

        def read(self, kind, split, *args):
            assert split == 'val'
            return {a: None for a in rt_paths.ASPECTS} if kind == 'labels' else None

    def seed(data, aspect, s, labels, dist, workers):
        base = 0.1 * (1 + rt_paths.ASPECTS.index(aspect)) + 0.01 * s
        return base, (base + 0.3 if aspect == 'emotion' else None)

    def run_with_step(step):
        t = [0.0]

        def tick():
            t[0] += step
            return t[0]
        monkeypatch.setattr(rt_stage0, 'time', SimpleNamespace(perf_counter=tick))
        assert rt_stage0.stage0a(env.outputs) == 0
        return env.ranges.read_bytes(), json.loads((env.outputs / 'stage0a/ranges.json').read_text())

    monkeypatch.setattr(rt_stage0, 'Inputs', FakeInputs)
    monkeypatch.setattr(rt_stage0, 'validation_seed', seed)
    first, doc1 = run_with_step(1.0)
    second, doc2 = run_with_step(7.0)
    assert doc1['total_seconds'] != doc2['total_seconds'] and doc1['seed_seconds'] != doc2['seed_seconds']
    assert first == second
    assert tuple(json.loads(first)) == rt_paths.RANGES_KEYS
    assert json.loads(first) == rt_paths.committed_ranges(doc2)
