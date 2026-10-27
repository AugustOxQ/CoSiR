"""The sample-scale trainer's run logic (resume, V1d rerun, failures, records) and V1a's comparison, with the cohort
replaced by a recorder; the cohort itself is `rg_train.train_cohort`, checked by V1a on the GPU."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

import rs_paths as paths
import rs_sheet as S
import rs_train as T
import rg_train
import rg_variants as V
from ab_paths import ms_train


SCRATCH = paths.SMOKE_ROOT / 'r10-rs-02'
SCRATCH.mkdir(parents=True, exist_ok=True)


class FakeData:
    hashes = dict(sample_sha256='s', default_inputs={}, teachers={})

    def get(self, *, export_p=True):
        assert export_p
        return 'sample', {'img': None, 'cap': None}, {'img': None}, paths.P_SHA256


@pytest.fixture
def cohort(monkeypatch, tmp_path):
    """Replace the graph loader and `train_cohort`; `calls` records what the cohort saw (the patches in force)."""
    calls = []
    outcomes = []          # statuses to return, in order: 'ok', 'nonfinite', 'va3'
    manifests = {a: {'archives': {a: 'x'}, 'identity': 'i'} for a in T.ASPECTS}
    monkeypatch.setattr(rg_train, 'load_graphs', lambda s, recipe: ({a: {} for a in T.ASPECTS}, manifests))
    monkeypatch.setattr(rg_train, 'verify_cached_run', lambda folder, record, export_p: calls.append('verify'))
    monkeypatch.setattr(T, 'code_commit', lambda: ('abc', False))

    def train_cohort(s, inputs, graphs, mans, recipe, variant, seed, device, run_dir, *, p, p_sha256, smoke,
                     max_epochs, rerun_of, input_hashes, diagnostic=False):
        outcome = outcomes.pop(0) if outcomes else 'ok'
        calls.append(dict(variant=variant, seed=seed, run_dir=Path(run_dir), lr=ms_train.CONFIG['lr'],
                          arm=dict(V.ARMS[variant]), weight=recipe['base']['distill_weight'], rerun_of=rerun_of,
                          smoke=smoke, max_epochs=max_epochs))
        record = dict(variant=variant, seed=seed, switches=V.switches(variant), status='ok',
                      best_epoch={a: 3 for a in T.ASPECTS}, epochs_run={a: 18 for a in T.ASPECTS},
                      p_sha256=p_sha256, smoke=smoke, max_epochs=max_epochs, code_commit='abc', code_dirty=False,
                      rg_code_sha256={}, input_hashes=input_hashes)
        if rerun_of is not None:
            record['rerun_of'] = str(rerun_of)
        if outcome != 'ok':
            record.update(status='failed', failure_kind={'nonfinite': 'nonfinite_loss', 'va3': 'va3'}[outcome],
                          cause=f'cause of {outcome}')
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        rg_train.atomic_record(Path(run_dir) / 'run.json', record)
        return record

    monkeypatch.setattr(rg_train, 'train_cohort', train_cohort)
    return calls, outcomes


def run(cfg=None, seed=0, outputs=None, **kwargs):
    return T.run_config(FakeData(), cfg or S.config(), seed, outputs, 'cpu', **kwargs)


def trained(calls):
    return [c for c in calls if c != 'verify']


def test_g_runs_the_frozen_arm_with_no_patch_and_writes_the_index_fields(cohort, tmp_path):
    calls, _ = cohort
    before = (dict(ms_train.CONFIG), dict(V.ARMS['GS-0.02']))
    record = run(outputs=tmp_path)
    [call] = trained(calls)
    assert (call['variant'], call['seed'], call['lr'], call['arm'], call['weight']) == ('GS-0.02', 0, 1e-3, V.ARMS['GS-0.02'], 0.25)
    assert call['run_dir'] == paths.run_dir(tmp_path, 'G', 0).resolve()
    assert (dict(ms_train.CONFIG), dict(V.ARMS['GS-0.02'])) == before
    on_disk = json.loads((call['run_dir'] / 'run.json').read_text())
    assert on_disk == json.loads(json.dumps(record))
    assert (on_disk['name'], on_disk['variant'], on_disk['base_arm'], on_disk['seed'], on_disk['status']) == (
        'G', 'G', 'GS-0.02', 0, 'ok')
    assert on_disk['config'] == T.jsonable(S.config()) and on_disk['knobs'] == {}
    assert set(on_disk) >= {'best_epoch', 'epochs_run', 'code_sha256', 'input_hashes', 'machine', 'host'}
    assert on_disk['code_sha256']['rs_knobs.py'] and on_disk['code_sha256']['rg_train.py']


def test_a_knob_run_sees_its_patches_and_its_own_folder(cohort, tmp_path):
    calls, _ = cohort
    cfg = S.config({'k6': 5e-4, 'k5': 0.125, 'k3': 12})
    record = run(cfg, seed=2, outputs=tmp_path)
    [call] = trained(calls)
    assert call['run_dir'] == paths.run_dir(tmp_path, 'k3=12+k5=0.125+k6=0.0005', 2).resolve()
    assert (call['lr'], call['weight'], call['arm']['lam'], call['arm']['tau_t']) == (5e-4, 0.125, 12, 0.02)
    assert set(record['knobs']) == {'k3', 'k5', 'k6'} and record['switches']['lam'] == 12
    assert ms_train.CONFIG['lr'] == 1e-3 and V.ARMS['GS-0.02']['lam'] == 8


def test_finished_runs_are_verified_and_skipped(cohort, tmp_path):
    calls, _ = cohort
    run(outputs=tmp_path)
    again = run(outputs=tmp_path)
    assert len(trained(calls)) == 1 and calls[-1] == 'verify' and again['status'] == 'ok'


def test_a_changed_input_configuration_or_code_is_refused_on_resume(cohort, tmp_path, monkeypatch):
    run(S.config({'k6': 5e-4}), outputs=tmp_path)
    path = paths.run_dir(tmp_path, 'k6=0.0005', 0) / 'run.json'
    good = json.loads(path.read_text())
    for field, value, message in (('config', dict(good['config'], k7=0.0), 'configuration mismatch'),
                                  ('smoke', True, 'configuration mismatch'),
                                  ('input_hashes', {'x': 1}, 'inputs changed'),
                                  ('code_sha256', dict(good['code_sha256'], **{'rs_knobs.py': '0'}), 'code_sha256 changed')):
        path.write_text(json.dumps(dict(good, **{field: value})))
        with pytest.raises(AssertionError, match=message):
            run(S.config({'k6': 5e-4}), outputs=tmp_path)
    path.write_text(json.dumps(good))
    run(S.config({'k6': 5e-4}), outputs=tmp_path)


def test_nonfinite_loss_is_a_recorded_failure_and_is_not_rerun(cohort, tmp_path):
    calls, outcomes = cohort
    outcomes.append('nonfinite')
    record = run(S.config({'k6': 2e-3}), outputs=tmp_path)
    assert (record['status'], record['failure_kind'], record['cause']) == ('failed', 'nonfinite_loss', 'cause of nonfinite')
    assert len(trained(calls)) == 1
    assert run(S.config({'k6': 2e-3}), outputs=tmp_path)['status'] == 'failed' and len(trained(calls)) == 1


def test_a_v1b_failure_is_rerun_once_with_the_first_run_moved_aside(cohort, tmp_path):
    calls, outcomes = cohort
    outcomes.extend(['va3', 'ok'])
    record = run(outputs=tmp_path)
    assert record['status'] == 'ok' and len(trained(calls)) == 2 and trained(calls)[0]['rerun_of'] is None
    [aside] = sorted((tmp_path / 'replaced' / 'G').glob('s0-*'))
    assert record['rerun_of'] == str(aside)
    assert json.loads((aside / 'run.json').read_text())['failure_kind'] == 'va3'
    # a second V1b failure stays failed: one rerun only
    outcomes.extend(['va3', 'va3'])
    record = run(S.config({'k7': 0.01}), outputs=tmp_path)
    assert record['status'] == 'failed' and record['rerun_of'] and len(trained(calls)) == 4
    assert run(S.config({'k7': 0.01}), outputs=tmp_path)['status'] == 'failed' and len(trained(calls)) == 4


def test_seeds_scratch_and_frozen_outputs_are_checked(cohort, tmp_path):
    with pytest.raises(AssertionError):
        run(seed=5, outputs=tmp_path)
    with pytest.raises(AssertionError, match='/tmp/claude-0'):
        run(outputs=tmp_path, smoke=True)
    with pytest.raises(AssertionError, match='frozen input'):
        run(outputs=paths.RG_OUT / 'x')


def test_k11_goes_through_rs_graphs_with_the_loaded_sample(monkeypatch, tmp_path):
    """The k11 run's graphs are rs_graphs.sample_store(k, 'aspect', s=, outputs=); no gate of this module."""
    seen = []

    class Fake:
        @staticmethod
        def sample_store(k, variant='aspect', *, s=None, outputs=None):
            seen.append((k, variant, s, outputs))
            return {a: {} for a in T.ASPECTS}, {a: {'archives': {a: 'x'}, 'identity': 'i'} for a in T.ASPECTS}

    monkeypatch.setitem(sys.modules, 'rs_graphs', Fake)
    monkeypatch.setattr(rg_train, 'verify_cached_run', lambda *a, **k: None)
    monkeypatch.setattr(T, 'code_commit', lambda: ('abc', False))
    got = []

    def stop(*args, **kwargs):
        got.append(args[2])
        raise KeyboardInterrupt

    monkeypatch.setattr(rg_train, 'train_cohort', stop)
    with pytest.raises(KeyboardInterrupt):
        run(S.config({'k11': 30}), outputs=tmp_path)
    assert seen == [(30, 'aspect', 'sample', tmp_path.resolve())] and set(got[0]) == set(T.ASPECTS)


def test_code_dirty_sees_the_round_folder_and_12_27(tmp_path, monkeypatch):
    def git(*args):
        subprocess.run(['git', *args], cwd=tmp_path, check=True, capture_output=True)

    git('init', '-q')
    git('config', 'user.email', 'a@b')
    git('config', 'user.name', 'n')
    for folder in ('20261027_recipe_sweep', '20261027_relational_genre', '20261026_other'):
        (tmp_path / 'src/test' / folder).mkdir(parents=True)
        (tmp_path / 'src/test' / folder / 'x.py').write_text('1\n')
    git('add', '-A')
    git('commit', '-qm', 'a')
    monkeypatch.setattr(paths, 'ROUND', tmp_path / 'src/test/20261027_recipe_sweep')
    head, dirty = T.code_commit()
    assert len(head) == 40 and not dirty
    for folder, expected in (('20261026_other', False), ('20261027_recipe_sweep', True), ('20261027_relational_genre', True)):
        path = tmp_path / 'src/test' / folder / 'x.py'
        path.write_text('2\n')
        assert T.code_commit()[1] is expected, folder
        git('checkout', '--', str(path))
    (tmp_path / 'src/test/20261027_recipe_sweep/new.py').write_text('1\n')
    assert T.code_commit()[1]


def test_an_interrupted_v1b_rerun_checks_the_origin_code(cohort, tmp_path):
    calls, outcomes = cohort
    outcomes.extend(['va3', 'ok'])
    run(outputs=tmp_path)
    [aside] = (tmp_path / 'replaced' / 'G').glob('s0-*')
    done = paths.run_dir(tmp_path, 'G', 0)
    done.rename(tmp_path / 'finished_rerun')                    # as if the rerun was interrupted before its folder
    origin = json.loads((aside / 'run.json').read_text())
    (aside / 'run.json').write_text(json.dumps(dict(origin, code_sha256=dict(origin['code_sha256'], **{'rs_knobs.py': '0'}))))
    with pytest.raises(AssertionError, match='code_sha256 changed'):
        run(outputs=tmp_path)
    (aside / 'run.json').write_text(json.dumps(origin))
    assert run(outputs=tmp_path)['rerun_of'] == str(aside)


def test_smoke_runs_two_epochs_in_scratch(cohort):
    calls, _ = cohort
    out = Path(tempfile.mkdtemp(dir=SCRATCH))
    record = run(S.config({'k13': 5}), outputs=out, smoke=True)
    assert (trained(calls)[0]['smoke'], trained(calls)[0]['max_epochs'], record['smoke']) == (True, 2, True)
    assert trained(calls)[0]['run_dir'].is_relative_to(out)


# ------------------------------------------------------------------ V1a's comparison
def test_v1a_compares_exports_and_states_bit_for_bit(tmp_path, monkeypatch):
    ref_root = tmp_path / 'ref'
    rng = np.random.default_rng(0)
    arrays = {(seed, a, label): rng.standard_normal((4, 3)).astype(np.float16)
              for seed in (0, 1) for a in T.ASPECTS for label in ('P', 'old5000')}
    for seed in (0, 1):
        ref = ref_root / 'GS-0.02' / f's{seed}'
        ref.mkdir(parents=True)
        (ref / 'run.json').write_text(json.dumps(dict(status='ok', variant='GS-0.02', seed=seed, p_sha256=paths.P_SHA256)))
        for a in T.ASPECTS:
            torch.save({'w': torch.full((2,), float(seed))}, ref / f'{a}.pt')
            for label in ('P', 'old5000'):
                np.savez(ref / f'{a}__{label}.npz', img=arrays[(seed, a, label)])
    monkeypatch.setattr(paths, 'RG_TRAIN', ref_root)
    monkeypatch.setattr(rg_train, 'read_export', lambda path, n: np.load(path)['img'])
    monkeypatch.setattr(T, 'code_commit', lambda: ('abc', False))
    state = dict(damage=None)

    def replay(data, cfg, seed, outputs, device, *, run_dir, smoke, machine, name):
        assert name == 'G-v1a'
        assert cfg == S.config()
        run_dir.mkdir(parents=True, exist_ok=True)
        for a in T.ASPECTS:
            bad = state['damage'] == (seed, a)
            torch.save({'w': torch.full((2,), float(seed) + bad)}, run_dir / f'{a}.pt')
            for label in ('P', 'old5000'):
                np.savez(run_dir / f'{a}__{label}.npz', img=arrays[(seed, a, label)])
        return dict(status='ok')

    monkeypatch.setattr(T, 'run_config', replay)
    out = tmp_path / 'out'
    record = T.regress_v1a(FakeData(), out, 'cpu', smoke=False)
    assert record['pass'] and json.loads((out / 'gates/v1a_train.json').read_text())['pass']
    assert set(record['code_sha256']) >= {'rs_train.py', 'rs_knobs.py', 'rg_train.py', 'ms_train.py'}
    state['damage'] = (1, 'style')
    record = T.regress_v1a(FakeData(), out, 'cpu', smoke=False)
    assert not record['pass'] and record['items']['s1']['aspects']['style']['state_equal'] is False
    assert record['items']['s0']['pass'] and not record['items']['s1']['pass']
    assert not T.regress_v1a(FakeData(), Path(tempfile.mkdtemp(dir=SCRATCH)), 'cpu', smoke=True)['pass']


def test_g_repeat_is_g_in_its_own_folder_and_machine_is_written_as_given(cohort, tmp_path):
    calls, _ = cohort
    record = run(outputs=tmp_path, name='G-repeat', machine='node402')
    [call] = trained(calls)
    assert (call['variant'], call['arm'], call['lr']) == ('GS-0.02', V.ARMS['GS-0.02'], 1e-3)
    assert call['run_dir'] == paths.run_dir(tmp_path, 'G-repeat', 0).resolve()
    assert (record['name'], record['config'], record['machine']) == ('G-repeat', T.jsonable(S.config()), 'node402')
    for name, cfg in (('G-repeat', S.config({'k6': 5e-4})), ('other', S.config())):
        with pytest.raises(AssertionError):
            run(cfg, outputs=tmp_path, name=name)
    with pytest.raises(AssertionError):
        run(outputs=tmp_path, machine='node401')
