"""Repeatable September artifact checks, without importing September modules."""
import hashlib
import json
import re
import subprocess
import sys

import numpy as np
import pytest
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads

import pc_stage0
from pc_fixes import FIX_TESTS


def test_fix_test_count_matches_collected_tests():
    collected = subprocess.run(
        [sys.executable, '-m', 'pytest', '--collect-only', '-q',
         str(pc_stage0.HERE / 'test_pc_fixes.py')],
        capture_output=True, text=True, timeout=30)
    assert collected.returncode == 0, collected.stdout + collected.stderr
    count = re.search(r'(\d+) tests? collected', collected.stdout)
    assert count is not None, collected.stdout
    assert int(count.group(1)) == FIX_TESTS


@pytest.mark.parametrize('commit', pc_stage0.RECORD_COMMITS)
def test_record_blobs_match_reference_and_cover_known_scripts(commit):
    record = pc_stage0.python_blobs(commit)
    reference = pc_stage0.python_blobs(pc_stage0.COMMIT)
    result = pc_stage0.compare_blobs(record, reference)
    assert result['status'] == 'PASS' and result['actual'] == result['expected'] == 0
    assert result['files_compared'] == len(record) > 8
    known = [pc_stage0.PILOT + name for name in (
        'run_percept_stage1_embedding_snapshot_pilot.py',
        'run_percept_stage2_extended_seed_pilot.py', 'run_percept_stage1_pilot.py',
        'run_percept_stage1_cluster_count_sweep_pilot.py', 'run_percept_stage2_sweep_pilot.py')]
    known += [pc_stage0.BUDDY + name for name in (
        'run_pipeline.py', 'run_affect_pilot.py', 'run_cca_audit_pilot.py')]
    if commit == '3b3ce61':
        known.remove(pc_stage0.PILOT + 'run_percept_stage1_embedding_snapshot_pilot.py')
    assert set(known) <= record.keys()
    all_python = {entry.split('\t')[1] for entry in pc_stage0.git(
        'ls-tree', '-r', '-z', commit, '--', 'src/').split('\0')
        if entry and entry.endswith('.py')}
    assert {path for path in all_python if not path.startswith('src/test/')} <= record.keys()
    assert all(not path.startswith('src/test/') or
               path.startswith((pc_stage0.PILOT, pc_stage0.BUDDY)) for path in record)


@pytest.mark.parametrize('damage', ['changed', 'missing'])
def test_record_blob_change_or_removal_fails(damage):
    record = pc_stage0.python_blobs('4e4480b')
    reference = pc_stage0.python_blobs(pc_stage0.COMMIT)
    path = pc_stage0.PILOT + 'run_percept_stage1_embedding_snapshot_pilot.py'
    if damage == 'changed':
        reference[path] = '0' * 40
    else:
        del reference[path]
    result = pc_stage0.compare_blobs(record, reference)
    assert result['status'] == 'FAIL' and result['actual'] == 1
    assert result['differences'] == [path]


def test_source_copies_match_mapped_git_text():
    result = pc_stage0.check_source_copies((pc_stage0.HERE / 'pc_sept.py').read_text())
    assert result == dict(status='PASS', expected=0, actual=0, differences=[])


@pytest.mark.parametrize('damage, name', [
    ('character', 'PRETRAIN_EPOCHS'), ('extra', 'UNMAPPED'), ('missing', 'PRETRAIN_EPOCHS'),
])
def test_source_copy_change_extra_or_missing_name_fails(damage, name):
    source = (pc_stage0.HERE / 'pc_sept.py').read_text()
    if damage == 'character':
        source = source.replace('PRETRAIN_EPOCHS = 100', 'PRETRAIN_EPOCHS = 101')
    elif damage == 'extra':
        source += '\nUNMAPPED = 1\n'
    else:
        source = source.replace('PRETRAIN_EPOCHS = 100\n', '')
    result = pc_stage0.check_source_copies(source)
    assert result['status'] == 'FAIL' and result['actual'] == 1
    assert result['differences'] == [name]


@pytest.mark.parametrize('old, new, difference', [
    ('def build_autoencoder(', '@torch.no_grad()\ndef build_autoencoder(', 'build_autoencoder'),
    ('class AttentionPoolingMapper(',
     '@torch.no_grad()\nclass AttentionPoolingMapper(', 'AttentionPoolingMapper'),
    ('def build_autoencoder(',
     '@(\n    torch.no_grad()\n)\n@torch.no_grad()\ndef build_autoencoder(', 'build_autoencoder'),
    ('', '\nLAMBDA_BALANCE += 1\n', 'top-level AugAssign'),
    ('', '\nif True:\n    LAMBDA_BALANCE = 0\n', 'top-level If'),
    ('', '\n"extra string outside the module docstring"\n', 'top-level Expr'),
    ('from sklearn.cluster import KMeans',
     'from sklearn.cluster import MiniBatchKMeans as KMeans', 'imports'),
    ('import numpy as np\nimport torch', 'import torch\nimport numpy as np', 'imports'),
    ('base = sys.modules[__name__]', 'base = __import__("math")', 'base'),
    ('base = sys.modules[__name__]', '', 'base'),
    ('', '\nLAMBDA_BALANCE = 1000\n', 'duplicate name LAMBDA_BALANCE'),
    ('', '\nbase.LAMBDA_BALANCE = 0\n', 'non-name assignment target'),
    ('', '\nbase.LAMBDA_BALANCE: int = 0\n', 'non-name assignment target'),
    ('', '\nleft, right = 1, 2\n', 'non-name assignment target'),
])
def test_source_behaviour_edits_return_fail_entries(old, new, difference):
    source = (pc_stage0.HERE / 'pc_sept.py').read_text()
    changed = source.replace(old, new) if old else source + new
    result = pc_stage0.check_source_copies(changed)
    assert result['status'] == 'FAIL' and result['actual'] >= 1
    assert any(difference in entry for entry in result['differences'])


@pytest.mark.parametrize('name, marker, filename', [
    ('build_autoencoder', 'def build_autoencoder(', 'run_percept_stage1_pilot.py'),
    ('AttentionPoolingMapper', 'class AttentionPoolingMapper(', 'run_percept_stage2_sweep_pilot.py'),
])
def test_source_comparison_includes_september_decorators(monkeypatch, name, marker, filename):
    source = (pc_stage0.HERE / 'pc_sept.py').read_text()
    decorated = '@(\n    torch.no_grad()\n)\n' + marker
    git = pc_stage0.git

    def decorated_git(*args):
        text = git(*args)
        return text.replace(marker, decorated) if args[-1].endswith(filename) else text

    monkeypatch.setattr(pc_stage0, 'git', decorated_git)
    result = pc_stage0.check_source_copies(source)
    assert result['status'] == 'FAIL' and result['differences'] == [name]
    result = pc_stage0.check_source_copies(source.replace(marker, decorated))
    assert result == dict(status='PASS', expected=0, actual=0, differences=[])


@pytest.mark.parametrize('alias', pc_stage0.ALIASES)
def test_sweep_aliases_must_match_verbatim(monkeypatch, alias):
    git = pc_stage0.git

    def changed_git(*args):
        text = git(*args)
        if args[-1].endswith('run_percept_stage1_cluster_count_sweep_pilot.py'):
            text = text.replace(f'{alias} = base.{alias}', f'{alias} = 0')
        return text

    monkeypatch.setattr(pc_stage0, 'git', changed_git)
    result = pc_stage0.check_source_copies((pc_stage0.HERE / 'pc_sept.py').read_text())
    assert result['status'] == 'FAIL' and result['actual'] == 1
    assert result['differences'] == [f'sweep alias {alias}']


def test_snapshot_rescores_real_values_and_counts():
    with np.load(pc_stage0.SNAPSHOT, allow_pickle=True) as snapshot:
        result = pc_stage0.check_snapshot(snapshot)
    assert result['status'] == 'PASS'
    assert result['actual']['emotion'] == pytest.approx(.12246902855745419, abs=1e-6)
    assert result['actual']['genre'] == pytest.approx(.22741920423971046, abs=1e-6)
    assert result['actual']['heldout_paintings'] == 9365
    assert result['actual']['genre_rows'] == 159


@pytest.mark.parametrize('field, shift', [
    ('emotion', 2e-6), ('genre', 2e-6), ('heldout_paintings', 1), ('genre_rows', 1),
])
@pytest.mark.parametrize('direction', [-1, 1])
def test_snapshot_rejects_small_metric_or_single_row_count_drift(monkeypatch, field, shift,
                                                              direction):
    expected = dict(pc_stage0.SNAPSHOT_EXPECTED)
    expected[field] += direction * shift
    monkeypatch.setattr(pc_stage0, 'SNAPSHOT_EXPECTED', expected)
    with np.load(pc_stage0.SNAPSHOT, allow_pickle=True) as snapshot:
        result = pc_stage0.check_snapshot(snapshot)
    assert result['status'] == 'FAIL'
    assert result['expected'][field] - result['actual'][field] == pytest.approx(direction * shift)


def test_snapshot_genre_ami_masks_unlabelled_paintings():
    snapshot = dict(heldout_emotion=np.array(['a', 'a', 'b', 'b', 'b', 'a']),
                    heldout_topic=np.array([0, 0, 1, 1, 0, 1]),
                    heldout_genre=np.array(['a', 'a', 'b', 'b', '', '']),
                    heldout_paintings=np.arange(6))
    result = pc_stage0.score_snapshot(snapshot)
    assert result['genre'] == pytest.approx(1.)
    assert result['genre_rows'] == 4 and result['heldout_paintings'] == 6
    snapshot['heldout_topic'][4:] = [1, 0]
    assert pc_stage0.score_snapshot(snapshot)['genre'] == result['genre']
    assert pc_stage0.check_snapshot(snapshot)['status'] == 'FAIL'


@pytest.mark.parametrize('changed', [False, True])
def test_kmeans_gate_requires_every_label_to_match(monkeypatch, changed):
    labels = np.arange(61402) % 60
    expected = labels.copy()
    if changed:
        expected[7] = (expected[7] + 1) % 60
    latents = np.broadcast_to(np.zeros(128, dtype=np.float32), (61402, 128))

    class FakeKMeans:
        def __init__(self, **kwargs):
            assert kwargs == dict(n_clusters=60, n_init=10, random_state=42)

        def fit(self, inputs):
            assert inputs is latents and _openmp_effective_n_threads() == 1
            self.labels_ = labels
            return self

    monkeypatch.setattr(pc_stage0, 'KMeans', FakeKMeans)
    result = pc_stage0.check_kmeans(dict(train_latent_pre=latents, train_initial_label=expected,
                                       n_initial_clusters=np.array(60), seed=np.array(42)))
    assert result['status'] == ('FAIL' if changed else 'PASS')
    assert result['actual'] == int(changed) and result['expected'] == 0
    assert result['seconds'] >= 0


@pytest.mark.parametrize('smoke, failure, fix_line', [
    (False, '', None), (True, '', None), (True, 'kmeans', None),
    (False, 'row2', None), (False, 'fix_tests', None),
    (False, 'source_duplicate', None), (False, 'source_target', None),
    (False, '', f'{FIX_TESTS} passed in 1.1s'), (False, 'fix_summary', f'{FIX_TESTS - 1} passed in 1s'),
    (False, 'fix_summary', f'{FIX_TESTS} passed, 1 skipped in 1s'),
    (False, 'fix_summary', f'{FIX_TESTS + 1} passed in 1s'),
    (False, 'fix_summary', f'{FIX_TESTS} passed, 1 failed in 1s'),
    (False, 'fix_summary', f'{FIX_TESTS} passed, 1 error in 1s'),
    (False, 'fix_summary', f'{FIX_TESTS} passed, 1 deselected in 1s'),
])
def test_run_schema_commands_and_exit_status(tmp_path, monkeypatch, smoke, failure, fix_line):
    calls = []
    kmeans = dict(status='FAIL' if failure == 'kmeans' else 'PASS', expected=0,
                  actual=int(failure == 'kmeans'), seconds=.25)
    monkeypatch.setattr(pc_stage0, 'check_kmeans', lambda snapshot: kmeans)
    if failure.startswith('source_'):
        check_source = pc_stage0.check_source_copies
        addition = ('\nLAMBDA_BALANCE = 1000\n' if failure == 'source_duplicate' else
                    '\nbase.LAMBDA_BALANCE = 0\n')
        monkeypatch.setattr(pc_stage0, 'check_source_copies',
                            lambda source: check_source(source + addition))

    def fake_command(name, command, folder, cwd, env):
        calls.append(name)
        assert cwd == pc_stage0.REPO
        assert env['CUDA_VISIBLE_DEVICES'] == ''
        assert env['OMP_NUM_THREADS'] == env['MKL_NUM_THREADS'] == '8'
        assert env['PYTHONDONTWRITEBYTECODE'] == '1'
        assert env['TMPDIR'] == str(tmp_path / 'tmp')
        assert folder.is_relative_to(tmp_path)
        if name == 'row2':
            assert command == [sys.executable, str(pc_stage0.HERE / 'pc_score.py'),
                               '--check-row2', '--outdir', str(tmp_path)]
            (tmp_path / 'row2_check.json').write_text(json.dumps(dict(both_mean=dict(
                emotion=.2 if failure == 'row2' else .08855639551158387,
                style=.17095358715401665, genre=.3930761231906966))))
        else:
            assert name == 'fix_tests'
            assert command == [sys.executable, '-m', 'pytest', '-q',
                               str(pc_stage0.HERE / 'test_pc_fixes.py')]
            line = fix_line or ('1 failed in 1.26s' if failure == name else f'{FIX_TESTS} passed in 1.26s')
            (folder / 'fix_tests.log').write_text('.....................\n' + line + '\n')
        return dict(status='FAIL' if failure == name == 'fix_tests' else 'PASS',
                    expected=0, actual=int(failure == name == 'fix_tests'), seconds=.01)

    monkeypatch.setattr(pc_stage0, 'run_command', fake_command)
    # An old success must never survive a failed or interrupted new invocation.
    (tmp_path / 'stage0.json').write_text('{"status": "PASS"}\n')
    code = pc_stage0.run(tmp_path, smoke=smoke)
    result = json.loads((tmp_path / 'stage0.json').read_text())
    assert code == int(bool(failure))
    assert result['status'] == ('FAIL' if failure else 'SMOKE' if smoke else 'PASS')
    assert result['fix_tests'] == (fix_line or ('1 failed in 1.26s' if failure == 'fix_tests'
                                             else f'{FIX_TESTS} passed in 1.26s'))
    assert calls == ['row2', 'fix_tests']
    assert result['commit'] == pc_stage0.COMMIT and result['smoke'] is smoke
    with pc_stage0.SNAPSHOT.open('rb') as stream:
        assert result['snapshot_sha256'] == hashlib.file_digest(stream, 'sha256').hexdigest()
    assert result['files_compared'] == {
        commit: len(pc_stage0.python_blobs(commit)) for commit in pc_stage0.RECORD_COMMITS}
    assert result['kmeans_seconds'] == .25
    assert result['kmeans_differing_labels'] == int(failure == 'kmeans')
    if failure.startswith('source_'):
        assert result['checks']['source_copies']['status'] == 'FAIL'
        assert result['checks']['source_copies']['actual'] == 1
        assert 'reason' not in result
    assert set(result['checks']) == {
        'september_blobs_3b3ce61', 'september_blobs_4e4480b', 'source_copies', 'snapshot',
        'kmeans_labels', 'row2_command', 'row2_emotion', 'row2_style', 'row2_genre', 'fix_tests'}
    assert all({'status', 'expected', 'actual'} <= check.keys()
               for check in result['checks'].values())
    assert not (tmp_path / 'stage0.pending.json').exists()


@pytest.mark.parametrize('aspect', ['emotion', 'style', 'genre'])
@pytest.mark.parametrize('shift, status', [(2e-6, 'FAIL'), (5e-7, 'PASS')])
def test_run_checks_small_row2_drift(tmp_path, monkeypatch, aspect, shift, status):
    values = dict(emotion=.08855639551158387, style=.17095358715401665,
                  genre=.3930761231906966)
    values[aspect] += shift
    monkeypatch.setattr(pc_stage0, 'check_kmeans', lambda snapshot: dict(
        status='PASS', expected=0, actual=0, seconds=.01))

    def command(name, args, folder, cwd, env):
        if name == 'row2':
            (tmp_path / 'row2_check.json').write_text(json.dumps(dict(both_mean=values)))
        else:
            (folder / 'fix_tests.log').write_text(f'{FIX_TESTS} passed in 1.1s\n')
        return dict(status='PASS', expected=0, actual=0, seconds=.01)

    monkeypatch.setattr(pc_stage0, 'run_command', command)
    assert pc_stage0.run(tmp_path) == int(status == 'FAIL')
    result = json.loads((tmp_path / 'stage0.json').read_text())
    assert result['status'] == status
    assert result['checks'][f'row2_{aspect}']['status'] == status


def test_interrupted_run_replaces_previous_pass(tmp_path, monkeypatch):
    (tmp_path / 'stage0.json').write_text('{"status": "PASS"}\n')

    def interrupted(commit):
        raise RuntimeError('interrupted git read')

    monkeypatch.setattr(pc_stage0, 'python_blobs', interrupted)
    with pytest.raises(RuntimeError, match='interrupted git read'):
        pc_stage0.run(tmp_path)
    assert json.loads((tmp_path / 'stage0.json').read_text())['status'] == 'FAIL'
