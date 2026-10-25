"""Fix round i-ii b: the P-export code hash (N2), the stage 5 replace (N1), V5 drop at start (N4)."""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

import eg_encoders
import eg_paths
import eg_pinfer
import eg_run
import eg_v5
from test_eg_final_i_ii import bare_run, put, stage3

RB = eg_paths.TEST / '20261025_recipe_backbones'


# ------------------------------------------------------------------ N2: content hash of the code that makes the export
@pytest.fixture
def code_tree(tmp_path, monkeypatch):
    """A copy of this round's and RB's python files; `eg_pinfer.code_sha256` reads the copy."""
    here, rb = tmp_path / 'code/20261025_encoders_genre', tmp_path / 'code/20261025_recipe_backbones'
    for src, dst in ((eg_paths.ROUND, here), (RB, rb)):
        dst.mkdir(parents=True)
        for f in src.glob('*.py'):
            shutil.copy(f, dst / f.name)
    monkeypatch.setattr(eg_pinfer, 'CODE_ROOT', here / 'eg_pinfer.py')
    monkeypatch.setattr(eg_pinfer, 'CODE_FOLDERS', (here, rb))
    return here, rb


def test_code_hash_ignores_unrelated_files_and_the_driver(code_tree):
    here, rb = code_tree
    before = eg_pinfer.code_sha256()
    assert len(before) == 64
    for path in (here / 'eg_score.py', here / 'eg_run.py', here / 'test_eg_fix_b.py', rb / 'rb_score3.py'):
        if path.exists():
            path.write_text(path.read_text() + '\n# unrelated\n')
    (here / 'handoff.md').write_text('a report')
    (here / 'eg_new_report.py').write_text('x = 1\n')
    assert eg_pinfer.code_sha256() == before


@pytest.mark.parametrize('name', ['eg_pinfer.py', 'eg_encoders.py', 'eg_paths.py', 'rb_student.py', 'rb_matched.py',
                                  'rb_encoders.py', 'rb_paths.py'])
def test_code_hash_follows_the_closure_in_both_folders_by_content(code_tree, name):
    here, rb = code_tree
    path = (here if name.startswith('eg_') else rb) / name
    before = eg_pinfer.code_sha256()
    original = path.read_text()
    path.write_text(original + '\n# dirty tree, not committed\n')
    assert eg_pinfer.code_sha256() != before
    path.write_text(original)
    assert eg_pinfer.code_sha256() == before


def test_code_hash_includes_file_names(code_tree):
    here, rb = code_tree
    before = eg_pinfer.code_sha256()
    (rb / 'rb_student.py').rename(rb / 'rb_student_moved.py')
    assert eg_pinfer.code_sha256() != before


def test_default_code_hash_runs_on_the_real_tree():
    assert len(eg_pinfer.code_sha256()) == 64


def rerun_world(tmp_path, monkeypatch, kind, sidecar, *, complete=False, forced='VL3 nonfinite', code='CODE'):
    """One export file (bad bytes) plus saved weights, ready for a rerun. `sidecar` is a dict, or None for none."""
    import rb_matched
    import rb_student
    run = bare_run(tmp_path, forced=forced)
    run.code_sha = code
    put(tmp_path, 'stage_ii/p/index.json', dict(complete=complete))
    enc = eg_encoders.BY_STUDENT['K6']
    out = (eg_paths.p_student(tmp_path, 'K6', 'emotion', 0) if kind == 'student'
           else eg_paths.p_own(tmp_path, 'M6', 0))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b'failed VL3')
    if sidecar is not None:
        put(tmp_path, str(out.with_suffix('.json').relative_to(tmp_path)), sidecar)
    monkeypatch.setattr(run, 'store_images', lambda *a: np.ones((1, enc.width), np.float32))
    monkeypatch.setattr(rb_student, 'weights_path', lambda *a: tmp_path / 'weights.pt')
    put(tmp_path, 'weights.json', dict(weights_sha256='W'))
    monkeypatch.setattr(rb_student.ms_train.head, 'sha_file', lambda *a: 'W')
    img = np.zeros((1, 2048), np.float16)
    img[0, 0] = 1
    monkeypatch.setattr(rb_student, 'infer', lambda *a: img)
    monkeypatch.setattr(rb_matched.base, 'verify_output', lambda *a: dict(weights_sha256='W', config={'device': 'cpu'}))
    monkeypatch.setattr(rb_matched, 'infer', lambda *a: np.zeros((1, 40), np.float32))
    call = (lambda: run.student_one(enc, 'emotion', 0, run.args.forced_vl3)) if kind == 'student' \
        else (lambda: run.own_one(enc, 0, run.args.forced_vl3))
    return run, out, call


@pytest.mark.parametrize('kind', ['student', 'own'])
def test_unrelated_commit_does_not_refuse_the_forced_rerun(tmp_path, monkeypatch, kind):
    run, out, call = rerun_world(tmp_path, monkeypatch, kind,
                                 dict(weights_sha256='W', code_commit='FIRST COMMIT', code_sha256='CODE'))
    assert run.commit == 'COMMIT'                      # HEAD moved since the first inference
    assert call() == 'wrote'
    side = json.loads(out.with_suffix('.json').read_text())
    assert side['code_sha256'] == 'CODE' and side['code_commit'] == 'COMMIT'
    log = json.loads((tmp_path / 'stage_ii/vl4_log.jsonl').read_text())
    assert log['code_sha256'] == 'CODE'


@pytest.mark.parametrize('kind', ['student', 'own'])
def test_changed_closure_file_refuses_the_forced_rerun(tmp_path, monkeypatch, kind):
    run, out, call = rerun_world(tmp_path, monkeypatch, kind,
                                 dict(weights_sha256='W', code_commit='COMMIT', code_sha256='OLD CODE'))
    with pytest.raises(AssertionError, match='VL4.*same weights and code'):
        call()
    assert out.read_bytes() == b'failed VL3'


@pytest.mark.parametrize('kind', ['student', 'own'])
def test_old_sidecar_without_code_hash_is_refused_with_a_message(tmp_path, monkeypatch, kind):
    run, out, call = rerun_world(tmp_path, monkeypatch, kind, dict(weights_sha256='W', code_commit='COMMIT'))
    with pytest.raises(AssertionError, match='no code_sha256'):
        call()
    assert out.read_bytes() == b'failed VL3'


@pytest.mark.parametrize('kind', ['student', 'own'])
@pytest.mark.parametrize('forced', [None, 'failed VL3'])
def test_missing_sidecar_before_the_index_is_complete_is_reinferred_and_logged(tmp_path, monkeypatch, kind, forced):
    run, out, call = rerun_world(tmp_path, monkeypatch, kind, None, forced=forced)
    assert call() == 'wrote'
    assert json.loads(out.with_suffix('.json').read_text())['code_sha256'] == 'CODE'
    log = json.loads((tmp_path / 'stage_ii/vl4_log.jsonl').read_text())
    assert log['cause'] == 'sidecar missing; re-inferred' and log['file'] == str(out)


@pytest.mark.parametrize('kind', ['student', 'own'])
@pytest.mark.parametrize('forced', [None, 'failed VL3'])
def test_missing_sidecar_after_the_index_is_complete_is_refused(tmp_path, monkeypatch, kind, forced):
    run, out, call = rerun_world(tmp_path, monkeypatch, kind, None, complete=True, forced=forced)
    with pytest.raises(AssertionError, match='VL4'):
        call()
    assert out.read_bytes() == b'failed VL3'
    assert not (tmp_path / 'stage_ii/vl4_log.jsonl').exists()


# ------------------------------------------------------------------ N1: replace also replaces a "no stage 3 outcome" stage 5 block
def replace_world(tmp_path, monkeypatch, s5_block):
    stage3(tmp_path, K6={'outcome_id': None, 'v_failure': 'V3: held output missing'})
    if s5_block is not None:
        put(tmp_path, 'part2/verdict_stage5.json', dict(stage=5, blocks={'K6': s5_block}))
    calls = []
    monkeypatch.setattr(eg_run, 'call', lambda cmd, print_only: calls.append([str(c) for c in cmd]) or 0)
    return calls


def stages_called(calls):
    return [c[c.index('--stage') + 1] for c in calls]


def test_replace_stage_3_also_replaces_a_no_stage_3_outcome_stage_5_block(tmp_path, monkeypatch):
    calls = replace_world(tmp_path, monkeypatch, dict(outcome_id=None, v_failure='no stage 3 outcome: V3: held output missing'))
    assert eg_run.replace(tmp_path, 3, 'K6', 'cause fixed', False, 2) == 0
    assert stages_called(calls) == ['3', '5']
    assert calls[1][calls[1].index('--replace') + 1] == 'cause fixed (stage 3 block of K6 replaced)'
    assert calls[1][calls[1].index('--candidates') + 1] == 'K6'


@pytest.mark.parametrize('s5', [None, dict(outcome_id='A-GO', d=0.1),
                                dict(outcome_id=None, v_failure='V5: artelingo28/clip_b16 failed')])
def test_replace_stage_3_leaves_other_stage_5_blocks_alone(tmp_path, monkeypatch, s5):
    calls = replace_world(tmp_path, monkeypatch, s5)
    eg_run.replace(tmp_path, 3, 'K6', 'cause fixed', False, 2)
    assert stages_called(calls) == ['3']


def test_replace_stage_3_print_only_prints_both_calls(tmp_path, monkeypatch, capsys):
    stage3(tmp_path, K6={'outcome_id': None, 'v_failure': 'x'})
    assert eg_run.replace(tmp_path, 3, 'K6', 'cause fixed', True, 2) == 0
    out = capsys.readouterr().out
    assert out.count('EG_CALL') == 2 and '--stage 3' in out and '--stage 5' in out


# ------------------------------------------------------------------ N4: V5 drops the blocks about to be checked first
def test_v5_drops_old_blocks_at_the_start_so_a_hard_kill_leaves_none(tmp_path, monkeypatch):
    old = {'pass': True, 'checks': {'rows': True}}
    put(tmp_path, 'stores/v5.json', dict(calibration={s: .2 for s in eg_v5.CALIBRATION},
                                         stores={'artelingo_full/clip_b16': old, 'artelingo28/clip_b16': old,
                                                 'artelingo_full/other': old}))

    class Killed(BaseException):
        pass

    def check(dataset, *args, **kw):
        if dataset == eg_v5.A28:
            raise Killed()                              # SIGKILL: no `except Exception` can run
        return {'pass': True, 'checks': {'rows': True}}
    monkeypatch.setattr(eg_v5, 'check_store', check)
    monkeypatch.setattr(eg_paths, 'check_imports', lambda: None)
    monkeypatch.setattr('sys.argv', ['eg_v5.py', '--outputs', str(tmp_path), '--store', 'clip_b16',
                                   '--dataset', 'artelingo_full', '--dataset', 'artelingo28'])
    with pytest.raises(Killed):
        eg_v5.main()
    saved = json.loads((tmp_path / 'stores/v5.json').read_text())['stores']
    assert saved['artelingo_full/clip_b16']['pass']             # the first dataset was checked and saved
    assert 'artelingo28/clip_b16' not in saved                  # the old block is gone: the queue reruns V5
    assert saved['artelingo_full/other'] == old                 # another store's block stays
