"""Queue failure propagation, V1 and isolated write destinations (CPU only)."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ct_run
import ct_smoke


@pytest.mark.parametrize('name', ['ct_queue_stage0.sh', 'ct_queue_stage1.sh', 'ct_queue_cpu.sh'])
def test_queue_shell_syntax(name):
    subprocess.run(['bash', '-n', str(HERE / name)], check=True)


def test_stage1_stops_at_missing_gate_and_records_one_failed_step(tmp_path):
    # Run the actual shell queue against a copied source-only queue directory.
    root = tmp_path / 'round'
    root.mkdir()
    for name in ('ct_queue_common.sh', 'ct_queue_stage1.sh', 'ct_stage0.py'):
        (root / name).write_bytes((HERE / name).read_bytes())
    result = subprocess.run(['bash', str(root / 'ct_queue_stage1.sh')], capture_output=True, text=True)
    assert result.returncode == 1
    assert result.stdout.splitlines()[0].startswith('require_stage0=1 seconds=')
    status = (root / 'outputs/queue_stage1_status.txt').read_text()
    assert 'require_stage0=1' in status and 'DONE failed=1' in status
    assert not (root / 'outputs/held').exists()
    assert list((root / 'outputs/logs').glob('*.log')) == [root / 'outputs/logs/stage1_require_stage0.log']


def test_smoke_refuses_production_shared_existing_and_parent_roots(tmp_path):
    for root in (HERE / 'outputs', HERE, Path('/data/new-smoke'), Path('/project/CoSiR/src/test'), tmp_path):
        with pytest.raises(AssertionError):
            ct_smoke.scratch_root(root)
    assert ct_smoke.scratch_root(tmp_path / 'new-smoke') == tmp_path / 'new-smoke'


def test_full_adapter_refuses_cpu_before_data_or_training():
    with pytest.raises(AssertionError, match='GPU'):
        ct_run.main(['d2', '--stage0-training', '--device', 'cpu'])


def test_images_adapter_keeps_the_shared_cache_and_only_redirects_the_held_list(tmp_path, monkeypatch):
    import ct_extract
    import ct_stage0
    monkeypatch.setattr(ct_stage0, 'require_pass', lambda *args: None)
    monkeypatch.setattr(ct_run, 'configure', lambda path: tmp_path)
    captured = {}

    def extract(argv):
        captured['held_list'] = ct_extract.HELD_LIST
        captured['write_store'] = ct_extract.write_store
        captured['output'] = ct_extract.OUTPUT
    monkeypatch.setattr(ct_extract, 'main', extract)
    # Restore all adapter globals in the shared pytest process afterward.
    for name in ('OUTPUT', 'HELD_LIST', 'write_store'):
        monkeypatch.setattr(ct_extract, name, getattr(ct_extract, name))
    writer = ct_extract.write_store
    ct_run.main(['images'])
    assert captured['held_list'] == tmp_path / 'held/held_paintings.json'
    assert captured['write_store'] is writer
    assert captured['output'] == SHARED_CACHE
    assert ct_extract.write_store.__defaults__[-1] == SHARED_CACHE


def test_diagnostics_adapter_reads_instructed_images_from_the_shared_cache(tmp_path, monkeypatch):
    import ct_diag
    import ct_stage0
    monkeypatch.setattr(ct_stage0, 'require_pass', lambda *args: None)
    monkeypatch.setattr(ct_run, 'configure', lambda path: tmp_path)
    monkeypatch.setattr(ct_diag, 'run', lambda *args, **kw: None)
    original = ct_diag.target_images
    monkeypatch.setattr(ct_diag, 'target_images', original)
    ct_run.main(['diagnostics'])
    assert ct_diag.target_images is original and ct_diag.TARGETS == SHARED_CACHE
    assert original.__defaults__[-1] == SHARED_CACHE


SHARED_CACHE = Path('/data/SSD2/pre_extract/artelingo_full/qwen3vl_emb_2b_aspect_prompts')
OLD, NEW = {'ct_run.py': 'older code'}, {'ct_run.py': 'current code'}


def _fake_stage0_artifacts(root):
    """Seed 0 and seed 1 files of both Stage 0 steps, plus an unrelated file."""
    names = ['d2/train/D2/emotion__Gres__s0.npz', 'd2/train/D2/emotion__Gres__s0.json',
             'd2/weights/emotion__s0.pt', 'd2/weights/emotion__s0.json',
             'percept/core/stage1/clip_roberta__s0.npz', 'percept/core/stage1/clip_roberta__s0.pt',
             'percept/core/stage1/clip_roberta__s0.json', 'percept/core/stage2/k40/P1__s0.npz',
             'percept/core/stage2/k40/P1__s0.json']
    keep = ['d2/train/D2/emotion__Gres__s1.npz', 'percept/core/stage2/k40/P1__s1.npz', 'logs/note.txt']
    for name in names + keep:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(name)
    return names, keep


def _stage0_main(monkeypatch, tmp_path, step, train):
    import ct_d2
    import ct_stage0
    monkeypatch.setattr(ct_stage0, 'code_identity', lambda: NEW)
    monkeypatch.setattr(ct_stage0, 'require_pass', lambda *args: None)
    monkeypatch.setattr(ct_run, 'configure', lambda path: tmp_path)
    monkeypatch.setattr(ct_d2, 'train', train)
    monkeypatch.setattr(ct_run, 'train_percept', train)
    ct_run.main([step, '--stage0-training'])


@pytest.mark.parametrize(('step', 'moved'), [('d2', slice(0, 4)), ('core', slice(4, 9))])
def test_stage0_training_moves_seed0_artifacts_of_older_code_aside_and_retrains(
        tmp_path, monkeypatch, step, moved):
    import re
    import ct_stage0
    names, keep = _fake_stage0_artifacts(tmp_path)
    ct_stage0.write_training_code(tmp_path, 'd2', OLD)
    ct_stage0.write_training_code(tmp_path, 'core', OLD)
    seen = []

    def train(*args, **kwargs):
        seen.append([n for n in names[moved] if (tmp_path / n).exists()])
    _stage0_main(monkeypatch, tmp_path, step, train)
    assert seen == [[]]                                   # gone before training starts
    (stamp,) = (tmp_path / 'stale').iterdir()
    assert re.fullmatch(r'\d{8}_\d{6}', stamp.name)
    for name in names[moved]:
        assert not (tmp_path / name).exists() and (stamp / name).read_text() == name
    other = [n for n in names if n not in names[moved]]
    assert all((tmp_path / n).read_text() == n for n in other + keep)
    assert ct_stage0.read_training_code(tmp_path) == {'d2': OLD if step == 'core' else NEW,
                                                      'core': OLD if step == 'd2' else NEW}


def test_stage0_training_keeps_artifacts_of_the_current_code(tmp_path, monkeypatch):
    import ct_stage0
    names, keep = _fake_stage0_artifacts(tmp_path)
    ct_stage0.write_training_code(tmp_path, 'd2', NEW)
    calls = []
    _stage0_main(monkeypatch, tmp_path, 'd2', lambda *a, **k: calls.append(a))
    assert len(calls) == 1 and not (tmp_path / 'stale').exists()
    assert all((tmp_path / n).exists() for n in names + keep)


def test_stage0_training_without_a_record_treats_existing_artifacts_as_stale(tmp_path, monkeypatch):
    import ct_stage0
    names, keep = _fake_stage0_artifacts(tmp_path)
    _stage0_main(monkeypatch, tmp_path, 'd2', lambda *a, **k: None)
    assert not (tmp_path / names[0]).exists() and list((tmp_path / 'stale').rglob('*__s0.npz'))
    assert ct_stage0.read_training_code(tmp_path) == {'d2': NEW}


def test_stage0_retraining_stays_refused_once_held_output_exists(tmp_path, monkeypatch):
    import ct_stage0
    names, _ = _fake_stage0_artifacts(tmp_path)
    ct_stage0.write_training_code(tmp_path, 'd2', OLD)
    (tmp_path / 'held/d2').mkdir(parents=True)
    (tmp_path / 'held/d2/emotion__s0.npz').write_bytes(b'held')
    calls = []
    with pytest.raises(AssertionError, match='V4'):
        _stage0_main(monkeypatch, tmp_path, 'd2', lambda *a, **k: calls.append(a))
    assert not calls and not (tmp_path / 'stale').exists() and (tmp_path / names[0]).exists()
    assert ct_stage0.read_training_code(tmp_path) == {'d2': OLD}


def test_run_main_sets_cublas_workspace_before_any_cuda_work(monkeypatch):
    import os
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', 'placeholder')
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG')
    with pytest.raises(AssertionError, match='GPU'):
        ct_run.main(['d2', '--stage0-training', '--device', 'cpu'])
    assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'


# --- wiring: every ct_run step calls the merged modules with the right names and arguments ---------

def test_configure_points_every_artifact_writer_at_the_outputs_root(monkeypatch):
    import ct_d2
    import ct_percept
    import ct_rcol
    import ct_score
    for module, name in ((ct_d2, 'OUTPUTS'), (ct_percept, 'OUTPUTS'), (ct_score, 'OUTPUTS'),
                         (ct_rcol, 'STAGE1'), (ct_rcol, 'OUTPUT')):
        monkeypatch.setattr(module, name, getattr(module, name))
    root = HERE / 'wiring_outputs_not_created'
    assert ct_run.configure(root) == root
    assert ct_d2.OUTPUTS == ct_percept.OUTPUTS == ct_score.OUTPUTS == root
    assert ct_rcol.STAGE1 == root / 'percept/paper/stage1'
    assert ct_rcol.OUTPUT == root / 'percept/rcol_stage1.json'
    with pytest.raises(AssertionError, match='worktree'):
        ct_run.configure(Path('/tmp/elsewhere'))


@pytest.fixture
def wired(tmp_path):
    """Replace every heavy callable ct_run reaches with an autospec mock (signature is enforced)."""
    from unittest import mock
    import ct_d2
    import ct_diag
    import ct_extract
    import ct_held
    import ct_percept as pt
    import ct_rcol
    import ct_score as sc
    import ct_stage0
    real_write_verdict = sc.write_verdict
    nonfinite = {('paper', 'k40', 'P1', 0)}
    names, sha = ['painting-a', 'painting-b'], 'held-sha'
    labels = {'emotion': 'emotion labels'}
    eligibility = dict(eligible_seeds=[0, 1])
    patches = {
        'configure': mock.patch.object(ct_run, 'configure', autospec=True, return_value=tmp_path),
        'require_pass': mock.patch.object(ct_stage0, 'require_pass', autospec=True),
        'frozen_training': mock.patch.object(ct_run, 'frozen_training', autospec=True),
        'frozen_held': mock.patch.object(ct_run, 'frozen_held', autospec=True, return_value=nonfinite),
        'train_percept': mock.patch.object(ct_run, 'train_percept', autospec=True),
        'd2_train': mock.patch.object(ct_d2, 'train', autospec=True),
        'd2_write_held': mock.patch.object(ct_d2, 'write_held', autospec=True),
        'run_held': mock.patch.object(pt, 'run_held', autospec=True),
        'write_held_list': mock.patch.object(ct_held, 'write_held_list', autospec=True),
        'read_held_list': mock.patch.object(ct_held, 'read_held_list', autospec=True, return_value=(names, sha)),
        'held_names': mock.patch.object(ct_held, 'held_names', autospec=True, return_value=names),
        'painting_labels': mock.patch.object(ct_held, 'painting_labels', autospec=True, return_value=labels),
        'write_stage1': mock.patch.object(ct_rcol, 'write_stage1', autospec=True),
        'read_rcol': mock.patch.object(sc, 'read_rcol', autospec=True,
                                       return_value=('rcol record', 'eligible', 'widths')),
        'prepare_held': mock.patch.object(sc, 'prepare_held', autospec=True,
                                          return_value=('fixed', eligibility, 'sources')),
        'paired_bootstrap': mock.patch.object(sc, 'paired_bootstrap', autospec=True, return_value='result'),
        'write_verdict': mock.patch.object(sc, 'write_verdict', autospec=True),
        'extract_main': mock.patch.object(ct_extract, 'main', autospec=True),
        'diag_run': mock.patch.object(ct_diag, 'run', autospec=True),
        'file_sha256': mock.patch.object(ct_stage0, 'file_sha256', autospec=True, return_value='rcol-sha'),
        'check_output': mock.patch('subprocess.check_output', autospec=True, return_value='abc123\n'),
    }
    started = {name: p.start() for name, p in patches.items()}
    started.update(tmp=tmp_path, names=names, sha=sha, labels=labels, eligibility=eligibility,
                   nonfinite=nonfinite, real_write_verdict=real_write_verdict)
    yield started
    for p in patches.values():
        p.stop()


REPORT = HERE / 'stage0.json'


def test_wiring_frozen_and_held_list(wired):
    tmp = wired['tmp']
    ct_run.main(['frozen'])
    wired['configure'].assert_called_once_with(HERE / 'outputs')
    wired['require_pass'].assert_called_once_with(REPORT, 'cuda')
    wired['frozen_training'].assert_called_once_with(tmp, 'cuda')
    wired['frozen_training'].reset_mock()
    ct_run.main(['held-list'])
    wired['write_held_list'].assert_called_once_with(tmp / 'held/held_paintings.json')
    wired['frozen_training'].assert_not_called()


def test_wiring_d2_uses_report_device_and_seeds(wired):
    ct_run.main(['d2', '--report', str(REPORT), '--device', 'cuda:1', '--outputs', str(HERE / 'o')])
    wired['configure'].assert_called_once_with(HERE / 'o')
    wired['require_pass'].assert_called_once_with(REPORT, 'cuda:1')
    wired['frozen_training'].assert_called_once_with(wired['tmp'], 'cuda:1')
    wired['d2_train'].assert_called_once_with([0, 1, 2, 3, 4], 'cuda:1', wired['tmp'])


@pytest.mark.parametrize('step', ['core', 'paper'])
def test_wiring_percept_training(wired, step):
    ct_run.main([step])
    wired['frozen_training'].assert_called_once_with(wired['tmp'], 'cuda')
    wired['train_percept'].assert_called_once_with(step, [0, 1, 2, 3, 4], 'cuda')


def test_wiring_stage0_training_skips_the_gate_and_uses_seed_zero(wired, monkeypatch):
    import ct_stage0
    monkeypatch.setattr(ct_stage0, 'code_identity', lambda: NEW)
    ct_run.main(['d2', '--stage0-training'])
    wired['require_pass'].assert_not_called()
    wired['d2_train'].assert_called_once_with([0], 'cuda', wired['tmp'])
    ct_run.main(['core', '--stage0-training'])
    wired['train_percept'].assert_called_once_with('core', [0], 'cuda')


def test_wiring_rcol(wired):
    import ct_rcol
    ct_run.main(['rcol'])
    wired['write_stage1'].assert_called_once_with(ct_rcol.STAGE1, ct_rcol.OUTPUT)


def test_wiring_held_writes_every_arm_and_preserves_nonfinite(wired):
    from unittest import mock
    tmp = wired['tmp']
    ct_run.main(['held'])
    wired['frozen_held'].assert_called_once_with(tmp, 'cuda')
    wired['d2_write_held'].assert_called_once_with([0, 1, 2, 3, 4], outdir=tmp)
    expected = [mock.call('core', 'k40', arm, seed) for arm in ('P1', 'P2', 'P3', 'P4') for seed in range(5)]
    expected += [mock.call('paper', cut, arm, seed) for cut in ('strict', 'k40') for arm in ('P1', 'P4')
                 for seed in range(5) if ('paper', cut, arm, seed) not in wired['nonfinite']]
    assert sorted(map(str, wired['run_held'].call_args_list)) == sorted(map(str, expected))
    assert len(expected) == 39


def test_wiring_images_keeps_cache_defaults(wired, monkeypatch):
    import ct_extract
    for name in ('OUTPUT', 'HELD_LIST', 'write_store'):
        monkeypatch.setattr(ct_extract, name, getattr(ct_extract, name))
    seen = {}
    wired['extract_main'].side_effect = lambda argv: seen.update(held=ct_extract.HELD_LIST)
    ct_run.main(['images', '--device', 'cuda:1'])
    wired['extract_main'].assert_called_once_with(['--device', 'cuda:1'])
    assert seen == dict(held=wired['tmp'] / 'held/held_paintings.json')


def test_wiring_verdict(wired):
    import ct_rcol
    tmp = wired['tmp']
    ct_run.main(['verdict'])
    wired['read_rcol'].assert_called_once_with(ct_rcol.OUTPUT)
    wired['read_held_list'].assert_called_once_with(tmp / 'held/held_paintings.json')
    wired['painting_labels'].assert_called_once_with(wired['names'])
    wired['prepare_held'].assert_called_once_with(wired['names'], wired['sha'], 'eligible', 'widths',
                                                  output_root=tmp)
    wired['paired_bootstrap'].assert_called_once_with('fixed', 'emotion labels', [0, 1], workers=8)
    wired['write_verdict'].assert_called_once_with(
        HERE / 'verdict_stage1.json', 'result', wired['eligibility'], wired['sha'], wired['labels'], 'abc123',
        dict(held_outputs='sources', rcol=dict(path=str(ct_rcol.OUTPUT), sha256='rcol-sha',
                                               record='rcol record')), replace=None)


def test_verdict_replacement_goes_through_ct_run_only_with_a_reason(wired, monkeypatch):
    import numpy as np
    import ct_score as sc
    monkeypatch.setattr(ct_run, 'HERE', wired['tmp'])
    path = wired['tmp'] / 'verdict_stage1.json'
    wired['write_verdict'].side_effect = wired['real_write_verdict']
    wired['painting_labels'].return_value = {'emotion': np.array([0, 1, -1])}

    def bootstrap(d):
        return dict(d=d, interval=[d - .1, d + .1], bootstrap_d=[d, d])

    wired['paired_bootstrap'].return_value = bootstrap(.05)
    ct_run.main(['verdict'])
    before = path.read_bytes()
    wired['paired_bootstrap'].return_value = bootstrap(.07)
    with pytest.raises(AssertionError, match='refusing to replace'):
        ct_run.main(['verdict'])
    assert path.read_bytes() == before
    ct_run.main(['verdict', '--replace', 'V3 rerun: corrected row order'])
    assert json.loads(path.read_text())['replacement']['reason'] == 'V3 rerun: corrected row order'
    assert json.loads(path.read_text())['d'] == .07
    with pytest.raises(AssertionError, match='verdict'):
        ct_run.main(['d2', '--replace', 'not a verdict step'])
    wired['d2_train'].assert_not_called()


def test_wiring_diagnostics(wired, monkeypatch):
    import ct_diag
    monkeypatch.setattr(ct_diag, 'target_images', ct_diag.target_images)
    before = ct_diag.target_images
    ct_run.main(['diagnostics'])
    wired['diag_run'].assert_called_once_with(HERE / 'verdict_stage1.json', wired['tmp'],
                                              HERE / 'diag_stage1.json', workers=8)
    assert ct_diag.target_images is before


def test_nonfinite_paper_held_output_is_preserved_for_v4_collapse(tmp_path, monkeypatch):
    import numpy as np
    import ct_held
    import ct_percept as pt
    monkeypatch.setattr(pt, 'OUTPUTS', tmp_path)
    names = [f'painting-{i:05d}' for i in range(12281)]
    sha = ct_held.names_sha256(names)
    held_list = tmp_path / 'held/held_paintings.json'
    held_list.parent.mkdir(parents=True)
    held_list.write_text(json.dumps(dict(names=names, sha256=sha)))
    config = dict(settings='paper', cut='k40', arm='P1', seed=0, smoke=False,
                  device='cuda', mapper='attention', input_dim=512, eval_paintings=5000)
    mapper = pt.build_mapper(config, 40)
    # verify_output checks the paper Stage 2 width against its Stage 1 kept count (ticket 07).
    stage1 = pt.stage1_path('paper', pt.ARMS['P1'][0], 0)
    stage1.parent.mkdir(parents=True)
    np.savez(stage1, kept_k40=np.ones(40, dtype=bool))
    stage1.with_suffix('.json').write_text(json.dumps(dict(K_by_cut=dict(k40=40))))
    path = pt.stage2_path('paper', 'k40', 'P1', 0)
    pt.save_output(path, dict(eval_prob=np.full((5000, 40), .5, dtype=np.float32)),
                   dict(mapper=pt.cpu_state(mapper)), dict(config=config, K=40, nonfinite=False))
    source = pt.verify_output(path)
    held_path = tmp_path / 'held/percept/paper__k40/P1__s0.npz'
    held_path.parent.mkdir(parents=True)
    prob = np.full((12281, 40), .5, dtype=np.float32)
    prob[0, 0] = np.nan
    np.savez_compressed(held_path, prob=prob)
    meta = dict(config=config, weights_sha256=source['weights_sha256'], held_sha256=sha,
                rows=len(names), inference_device='cuda', array_sha256s=dict(prob=pt.pc_stage1.array_sha256(prob)),
                nonfinite=True)  # V4 fix: paper sidecars record the flag
    held_path.with_suffix('.json').write_text(json.dumps(meta))
    before = held_path.read_bytes()
    assert ct_run.frozen_held(tmp_path) == {('paper', 'k40', 'P1', 0)}
    assert held_path.read_bytes() == before
