"""Tiny CPU checks of the full-scale trainer's loss wiring and the held guard (ticket 06; verdict code)."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

import subprocess

import ab_train
import rb_encoders
import rb_rows
import rb_student
import rg_paths
import rg_variants as V
from ab_paths import ms_train


# --- the held guard ---------------------------------------------------------------------------------------------

def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


ASPECTS = ('emotion', 'style', 'genre')


def put_weights(root, arm, seeds=(0,), *, rg_code='rg-code', rb_code='rb-code', max_epochs=None, smoke=False):
    """Fake saved best views (a few bytes and the record the trainer writes) of `arm` under `root/weights/<arm>/`."""
    for seed in seeds:
        for aspect in ASPECTS:
            path = rb_student.weights_path(root, arm, aspect, seed)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f'{arm}{aspect}{seed}'.encode())
            config = dict(aspect=aspect, seed=seed, smoke=smoke, device='cpu',
                          max_epochs=ms_train.CONFIG['max_epochs'] if max_epochs is None else max_epochs,
                          rb=dict(scale='full', encoder='qwen3vl_emb_2b', width=2048, recipe='R6-5', code_sha256=rb_code),
                          rg=dict(arm=arm, switches=V.switches(arm), code_sha256=rg_code))
            verdict = root.parents[1] / 'stage_b/verdict.json'
            if verdict.is_file():
                config['rg']['stage_b_verdict_sha256'] = ms_train.head.sha_file(verdict)
            put(path.with_suffix('.json'), dict(weights_sha256=ms_train.head.sha_file(path), config=config))


def vc1_train_record(tmp_path, **overrides):
    """A passing full-run `vc1_train.json` whose per-aspect weights checksums are those of the fake VC1 run's files."""
    run = tmp_path / 'stage_c/regression/vc1_run'
    items = {aspect: {'pass': True, 'weights': {'pass': True, 'ours_sha256': ms_train.head.sha_file(
        rb_student.weights_path(run, 'R6-5', aspect, 0))}} for aspect in ASPECTS}
    return {'pass': True, 'identity': {'smoke': False}, 'items': items, **overrides}


@pytest.fixture
def out(tmp_path, monkeypatch):
    import rg_full
    # the current code's hashes are fixed to the fake records' (tests of the code on disk change them again)
    monkeypatch.setattr(rg_full, '_code_sha256', lambda: 'rg-code')
    monkeypatch.setattr(rg_full, '_rb_code_sha256', lambda: 'rb-code')
    put(tmp_path / 'stage_b/verdict.json', dict(outcome='B-PICK', v_held='GR-8', v_held_switches=V.switches('GR-8'), smoke=False))
    put_vb2(tmp_path)
    put_weights(tmp_path / 'stage_c/regression/vc1_run', 'R6-5')
    put(tmp_path / 'stage_c/regression/vc1_train.json', vc1_train_record(tmp_path))
    import rg_held
    put(tmp_path / 'stage_c/regression/vc1_scorer.json', {'pass': True, 'identity': rg_held.code_provenance()})
    put_weights(rg_paths.full_root(tmp_path), 'GR-8', V.FULL_SEEDS)
    return tmp_path


def yes():
    return True


def test_guard_passes_when_every_condition_holds(out):
    import rg_full
    rg_full.require_guard(out, 'GR-8', ancestor=yes)


@pytest.mark.parametrize('name', ['vc1_train', 'vc1_scorer'])
def test_guard_refuses_a_missing_vc1_file(out, name):
    import rg_full
    (out / f'stage_c/regression/{name}.json').unlink()
    with pytest.raises(AssertionError, match='VC1'):
        rg_full.require_guard(out, 'GR-8', ancestor=yes)


@pytest.mark.parametrize('name', ['vc1_train', 'vc1_scorer'])
@pytest.mark.parametrize('value', [False, 'true', 1, None])
def test_guard_refuses_a_vc1_file_that_does_not_pass(out, name, value):
    import rg_full
    put(out / f'stage_c/regression/{name}.json', {'pass': value})
    with pytest.raises(AssertionError, match='VC1'):
        rg_full.require_guard(out, 'GR-8', ancestor=yes)


def test_guard_refuses_a_missing_verdict(out):
    import rg_full
    (out / 'stage_b/verdict.json').unlink()
    with pytest.raises(AssertionError, match='verdict'):
        rg_full.require_guard(out, 'GR-8', ancestor=yes)


def test_guard_refuses_b_none(out):
    import rg_full
    put(out / 'stage_b/verdict.json', dict(outcome='B-NONE', v_held=None, v_held_switches=None))
    with pytest.raises(AssertionError, match='B-PICK'):
        rg_full.require_guard(out, 'GR-8', ancestor=yes)


def test_guard_refuses_another_arm_than_v_held(out):
    import rg_full
    with pytest.raises(AssertionError, match='v_held is'):
        rg_full.require_guard(out, 'GR-16', ancestor=yes)


def test_guard_refuses_other_switches_than_the_arms(out):
    import rg_full
    put(out / 'stage_b/verdict.json',
        dict(outcome='B-PICK', v_held='GR-8', v_held_switches=V.switches('GR-16')))
    with pytest.raises(AssertionError, match='switches'):
        rg_full.require_guard(out, 'GR-8', ancestor=yes)


@pytest.mark.parametrize('arm', ['K4', 'R6-5'])
def test_guard_refuses_an_arm_outside_the_round_and_the_base_recipe(out, arm):
    import rg_full
    with pytest.raises(AssertionError, match='not a candidate arm'):
        rg_full.require_guard(out, arm, ancestor=yes)


def test_guard_refuses_when_the_rule_commit_is_not_an_ancestor(out):
    import rg_full
    with pytest.raises(AssertionError, match='rule commit'):
        rg_full.require_guard(out, 'GR-8', ancestor=lambda: False)


def test_rule_commit_ancestry_checks_git_for_real():
    import rg_full
    assert rg_full.rule_commit_is_ancestor() is True            # this branch descends from the rule's commit
    assert rg_full.rule_commit_is_ancestor('0' * 40) is False   # an unknown commit is not an ancestor


def test_guard_refuses_an_empty_outputs_folder(tmp_path):
    import rg_full
    with pytest.raises(AssertionError):
        rg_full.require_guard(tmp_path, 'GR-8', ancestor=yes)


# --- the switch wiring ------------------------------------------------------------------------------------------

def rows(n, width=5, seed=7):
    generator = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(n, width, generator=generator), dim=1)


def models():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        return nn.ModuleDict({s: nn.Sequential(nn.Linear(5, 5), nn.Tanh(), nn.Dropout(0.1)) for s in ('img', 'cap')})


def batch(n, aspect='genre', teacher=None):
    return dict(models=models(), img=rows(n), cap=rows(n, seed=9), teacher=rows(n, seed=11) if teacher is None else teacher,
                pairs=np.array([[0, 1], [2, 3], [1, 4], [5, 6]]), weights=np.ones(4, dtype=np.float32),
                row=rb_rows.recipe('R6-5')['base'], aspect=aspect)


def seeded(fn):
    """The loss's value with the dropout stream reset, so two calls draw the same masks."""
    torch.manual_seed(5)
    return fn()


def run_with_rng(fn):
    rng = torch.get_rng_state()
    value = fn()
    after = torch.get_rng_state()
    torch.set_rng_state(rng)
    return value, after


def test_switches_off_use_the_ab_loss_exactly():
    import rg_full
    fit_teacher = rows(8, seed=11)
    loss = rg_full.head_loss_for('R6-5', seed=0, fit_teacher=fit_teacher, n_dev=3)
    for aspect in ('emotion', 'style', 'genre'):
        b = batch(8, aspect)
        got, after = run_with_rng(lambda: loss(**b))
        # the AB loss as rb_student.loss_patch patches it: partial(ab_train.switched_head_loss, switches=...)
        want, after_want = run_with_rng(
            lambda: ab_train.switched_head_loss(**b, switches=rb_rows.recipe('R6-5')['switches']))
        assert torch.equal(got, want) and torch.equal(after, after_want)


def test_a_gr_arm_changes_only_the_genre_loss():
    import rg_full
    loss = rg_full.head_loss_for('GR-8', seed=0, fit_teacher=rows(8, seed=11), n_dev=3)
    ab = rb_rows.recipe('R6-5')['switches']
    b = batch(8, 'genre')
    assert not torch.equal(seeded(lambda: loss(**b)), seeded(lambda: ab_train.switched_head_loss(**b, switches=ab)))
    for aspect in ('emotion', 'style'):
        b = batch(8, aspect)
        assert torch.equal(seeded(lambda: loss(**b)), seeded(lambda: ab_train.switched_head_loss(**b, switches=ab)))


def test_gr_loss_is_the_genre_loss_of_ticket_01():
    import rg_full
    import rg_loss
    b = batch(8, 'genre')
    loss = rg_full.head_loss_for('GN-8', seed=0, fit_teacher=rows(8, seed=11), n_dev=3)
    got = seeded(lambda: loss(**b))
    want = seeded(lambda: rg_loss.switched_head_loss(**b, switches=rb_rows.recipe('R6-5')['switches'],
                                                     genre=rg_loss.GenreLoss(V.switches('GN-8'), 0)))
    assert torch.equal(got, want)


N_FIT, N_DEV = 4200, 4150      # anchors need >= 4096 fit rows; the dev count must differ from the fit count


def gq_batch(teacher, n):
    b = batch(n, 'genre', teacher=teacher)
    b['pairs'] = np.array([[0, 1], [2, 3], [1, 4], [5, 6]])
    return b


def test_gq_anchors_are_the_fit_teacher_rows_at_the_drawn_fit_local_positions():
    import rg_full
    import rg_loss
    fit_teacher = rows(N_FIT, seed=21)
    genre = rg_full.genre_for('GQ-8', seed=3, fit_teacher=fit_teacher, n_dev=N_DEV)
    ids = rg_loss.draw_anchors(N_FIT, 3)
    assert torch.equal(genre.anchor_ids, ids) and torch.equal(genre.anchors, fit_teacher[ids])
    assert (genre.n_fit, genre.n_dev) == (N_FIT, N_DEV)


def test_gq_fit_and_dev_calls_run_and_a_mismatched_fit_teacher_is_refused():
    import rg_full
    fit_teacher = rows(N_FIT, seed=21)
    loss = rg_full.head_loss_for('GQ-8', seed=3, fit_teacher=fit_teacher, n_dev=N_DEV)
    assert torch.isfinite(loss(**gq_batch(fit_teacher, N_FIT)))
    dev = rows(N_DEV, seed=33)
    assert torch.isfinite(loss(**gq_batch(dev, N_DEV), include_mask_penalty=False))
    with pytest.raises(AssertionError):                       # a fit call whose teacher is not the anchors' source
        loss(**gq_batch(rows(N_FIT, seed=22), N_FIT))
    with pytest.raises(AssertionError):                       # a dev call carrying fit-sized tensors
        loss(**gq_batch(fit_teacher, N_FIT), include_mask_penalty=False)


def test_gq_anchors_do_not_move_the_global_random_streams():
    import rg_full
    before = torch.get_rng_state().clone()
    rg_full.genre_for('GQ-8', seed=3, fit_teacher=rows(N_FIT, seed=21), n_dev=N_DEV)
    assert torch.equal(before, torch.get_rng_state())


def test_other_arms_take_no_anchors_and_gq_needs_the_fit_teacher():
    import rg_full
    assert rg_full.genre_for('GR-8', seed=0, fit_teacher=rows(8), n_dev=3).anchors is None
    with pytest.raises(AssertionError):
        rg_full.genre_for('GQ-8', seed=0, fit_teacher=None, n_dev=N_DEV)


def test_a_non_finite_loss_stops_the_run_with_its_aspect():
    import rg_full
    loss = rg_full.head_loss_for('GR-8', seed=0, fit_teacher=rows(8, seed=11), n_dev=3)
    b = batch(8, 'genre')
    b['teacher'] = torch.full((8, 5), float('nan'))
    with pytest.raises(rg_full.NonFiniteLoss) as info:
        loss(**b)
    assert info.value.args[0] == 'genre'


# --- the guard's VC1 code identity ------------------------------------------------------------------------------

def test_guard_refuses_a_smoke_vc1_train_record(out):
    import rg_full
    put(out / 'stage_c/regression/vc1_train.json', {'pass': True, 'identity': {'smoke': True}})
    with pytest.raises(AssertionError, match='smoke'):
        rg_full.require_guard(out, 'GR-8', ancestor=yes)


def test_guard_refuses_a_vc1_train_record_without_identity(out):
    import rg_full
    put(out / 'stage_c/regression/vc1_train.json', {'pass': True})
    with pytest.raises(AssertionError, match='smoke'):
        rg_full.require_guard(out, 'GR-8', ancestor=yes)


@pytest.mark.parametrize('which', ['rg', 'rb'])
def test_guard_refuses_weights_of_code_that_vc1_did_not_check(out, which):
    import rg_full
    put_weights(rg_paths.full_root(out), 'GR-8', (3,), **{f'{which}_code': 'changed-after-vc1'})
    with pytest.raises(AssertionError, match='code'):
        rg_full.require_guard(out, 'GR-8', ancestor=yes)


def test_guard_refuses_when_the_vc1_run_has_no_weights_record(out):
    import rg_full
    (out / 'stage_c/regression/vc1_run/weights/R6-5/genre__s0.json').unlink()
    with pytest.raises(AssertionError, match='missing'):
        rg_full.require_guard(out, 'GR-8', ancestor=yes)


def test_rg_full_itself_counts_as_code_that_alters_a_weight():
    import rg_full
    assert 'rg_full.py' in rg_full.CODE_FILES


def test_guard_refuses_a_vc1_run_weights_record_without_a_code_hash(out):
    import rg_full
    path = out / 'stage_c/regression/vc1_run/weights/R6-5/genre__s0.json'
    record = json.loads(path.read_text())
    del record['config']['rg']['code_sha256']
    put(path, record)
    with pytest.raises(AssertionError, match='rg code'):
        rg_full.require_guard(out, 'GR-8', ancestor=yes)


def test_guard_refuses_a_trained_record_without_a_code_hash(out):
    import rg_full
    path = rb_student.weights_path(rg_paths.full_root(out), 'GR-8', 'style', 0).with_suffix('.json')
    record = json.loads(path.read_text())
    del record['config']['rb']['code_sha256']
    put(path, record)
    with pytest.raises(AssertionError, match='rb code'):
        rg_full.require_guard(out, 'GR-8', ancestor=yes)


# --- the guard checks the code on disk now, and ties the VC1 run to vc1_train.json ---------------------------------

def test_guard_refuses_when_the_rg_code_on_disk_changed_after_vc1(out, monkeypatch):
    import rg_full
    monkeypatch.setattr(rg_full, '_code_sha256', lambda: 'edited-after-vc1')
    with pytest.raises(AssertionError, match='rg code on disk'):
        rg_full.require_guard(out, 'GR-8', ancestor=yes)


def test_guard_refuses_when_the_rb_code_on_disk_changed_after_vc1(out, monkeypatch):
    import rg_full
    monkeypatch.setattr(rg_full, '_rb_code_sha256', lambda: 'edited-after-vc1')
    with pytest.raises(AssertionError, match='rb code on disk'):
        rg_full.require_guard(out, 'GR-8', ancestor=yes)


def test_guard_refuses_a_vc1_run_that_vc1_train_json_does_not_describe(out):
    import rg_full
    # `regress vc1` retrained vc1_run and stopped before it rewrote vc1_train.json: the old pass is still there
    run = out / 'stage_c/regression/vc1_run'
    put_weights(run, 'R6-5', rg_code='rg-code')
    (run / 'weights/R6-5/style__s0').write_bytes(b'retrained')
    path = run / 'weights/R6-5/style__s0.json'
    record = json.loads(path.read_text())
    record['weights_sha256'] = ms_train.head.sha_file(run / 'weights/R6-5/style__s0')
    put(path, record)
    with pytest.raises(AssertionError, match='vc1_train.json'):
        rg_full.require_guard(out, 'GR-8', ancestor=yes)


def test_guard_refuses_a_vc1_train_record_without_the_aspect_checksum(out):
    import rg_full
    put(out / 'stage_c/regression/vc1_train.json', vc1_train_record(out, items={}))
    with pytest.raises(AssertionError, match='vc1_train.json'):
        rg_full.require_guard(out, 'GR-8', ancestor=yes)


def test_the_current_rb_code_hash_is_the_one_rb_student_writes_into_a_config():
    import rg_full
    rb = rb_student._rb_block('full', rg_full.ENCODER, rg_full.RECIPE, rb_rows.recipe(rg_full.RECIPE), np.zeros((2, 3)))
    assert rg_full._rb_code_sha256() == rb['code_sha256']


def test_an_edit_to_a_code_file_changes_the_code_hash(tmp_path, monkeypatch):
    import shutil
    import rg_full
    for name in rg_full.CODE_FILES:
        shutil.copy(rg_paths.ROUND / name, tmp_path / name)
    monkeypatch.setattr(rg_paths, 'ROUND', tmp_path)
    before = rg_full._code_sha256()
    assert before == rg_full._code_sha256()
    for name in rg_full.CODE_FILES:
        original = (tmp_path / name).read_text()
        (tmp_path / name).write_text(original + '# edited\n')
        assert rg_full._code_sha256() != before, name
        (tmp_path / name).write_text(original)
        assert rg_full._code_sha256() == before, name


# --- held inference: VC3, a partial run, the guard's call site --------------------------------------------------

N_ROWS = 4


@pytest.fixture
def fake(monkeypatch):
    """Inference without data or a model: `infer` returns the queued arrays (1.0 marks a VC3 failure)."""
    import rg_full
    state = dict(calls=0, queue=[])
    monkeypatch.setattr(rb_student, 'held_names', lambda: ([f'p{i}' for i in range(N_ROWS)], 'names-sha'))
    monkeypatch.setattr(rb_encoders, 'full_images',
                        lambda encoder, names: np.zeros((len(names), rb_encoders.WIDTH[encoder]), dtype=np.float32))
    monkeypatch.setattr(rg_paths, 'N_HELD', N_ROWS)

    def infer(weights, aspect, width, x, device):
        state['calls'] += 1
        return np.full((N_ROWS, 2048), state['queue'].pop(0) if state['queue'] else 0.0, dtype=np.float16)

    monkeypatch.setattr(rb_student, 'infer', infer)
    monkeypatch.setattr(rg_full, 'vc3_problem', lambda x: 'bad output' if x[0, 0] == 1 else None)
    return state


@pytest.fixture
def root(tmp_path):
    put_weights(tmp_path, 'GR-8', V.FULL_SEEDS)
    return tmp_path


def test_a_second_vc3_failure_is_recorded_and_a_second_call_does_not_try_again(root, fake):
    import rg_full
    fake['queue'] = [1.0, 1.0]
    with pytest.raises(AssertionError, match='VC3 failed twice'):
        rg_full.infer_held('GR-8', [0], 'cpu', root)
    failed = root / 'held/GR-8/failed__emotion__s0.json'
    assert json.loads(failed.read_text())['problems'] == ['bad output', 'bad output'] and fake['calls'] == 2
    with pytest.raises(AssertionError, match='VC4'):             # a good third attempt must not hide the failure
        rg_full.infer_held('GR-8', [0], 'cpu', root)
    assert fake['calls'] == 2 and not list((root / 'held/GR-8').glob('*.npz'))


def test_a_stopped_training_run_leaves_no_held_output(root, fake):
    import rg_full
    put(rg_full.failed_path(root, 'GR-8', 1), dict(failure_kind='nonfinite_loss'))
    with pytest.raises(AssertionError, match='VC4'):
        rg_full.infer_held('GR-8', [0, 1], 'cpu', root)
    assert fake['calls'] == 0 and not (root / 'held').exists()


def test_a_vc3_rerun_that_passes_lists_the_first_problem_in_the_output(root, fake):
    import rg_full
    fake['queue'] = [1.0]
    rg_full.infer_held('GR-8', [0], 'cpu', root)
    _, info = rb_student.read_outputs(root / 'held/GR-8/emotion__s0.npz')
    assert info['vc3_attempts'] == 2 and info['vc3_problems'] == ['bad output']
    _, info = rb_student.read_outputs(root / 'held/GR-8/style__s0.npz')
    assert info['vc3_attempts'] == 1 and info['vc3_problems'] == []


def test_weights_trained_with_fewer_epochs_than_the_ct_are_refused_outside_a_smoke(tmp_path, fake):
    import rg_full
    put_weights(tmp_path, 'GR-8', (0,), max_epochs=3)
    with pytest.raises(AssertionError, match='max_epochs'):
        rg_full.infer_held('GR-8', [0], 'cpu', tmp_path)
    assert fake['calls'] == 0


@pytest.fixture
def infer_calls(monkeypatch):
    import rg_full
    calls = []
    monkeypatch.setattr(rg_full, 'infer_held', lambda *args, **kwargs: calls.append((args, kwargs)) or {})
    return calls


def test_held_refuses_before_any_inference_when_the_guard_fails(out, infer_calls):
    import rg_full
    put(out / 'stage_b/verdict.json', dict(outcome='B-NONE', v_held=None, v_held_switches=None))
    with pytest.raises(AssertionError, match='B-PICK'):
        rg_full.held('GR-8', list(V.FULL_SEEDS), out, ancestor=yes)
    assert infer_calls == []


def test_held_runs_inference_on_the_full_root_when_the_guard_passes(out, infer_calls):
    import rg_full
    rg_full.held('GR-8', list(V.FULL_SEEDS), out, ancestor=yes)
    assert len(infer_calls) == 1 and infer_calls[0][0][:2] == ('GR-8', list(V.FULL_SEEDS))
    assert infer_calls[0][0][3] == rg_paths.full_root(out) and infer_calls[0][1] == dict(smoke=False)


def test_held_smoke_outside_the_smoke_folder_is_refused(infer_calls):
    import rg_full
    with pytest.raises(AssertionError, match='claude-0'):
        rg_full.held('GR-8', [0], '/nonexistent/outputs', smoke=True)
    assert infer_calls == []


def test_held_refuses_a_partial_seed_list_before_any_output(out, infer_calls):
    import rg_full
    with pytest.raises(AssertionError, match='seeds'):
        rg_full.held('GR-8', [0, 1], out, ancestor=yes)
    assert infer_calls == []


def test_held_refuses_missing_weights_of_a_later_seed_before_any_output(out, infer_calls):
    import rg_full
    rb_student.weights_path(rg_paths.full_root(out), 'GR-8', 'genre', 4).unlink()
    with pytest.raises(AssertionError, match='missing'):
        rg_full.held('GR-8', list(V.FULL_SEEDS), out, ancestor=yes)
    assert infer_calls == []


def test_held_refuses_a_weights_checksum_mismatch_of_a_later_seed(out, infer_calls):
    import rg_full
    rb_student.weights_path(rg_paths.full_root(out), 'GR-8', 'style', 3).write_bytes(b'other')
    with pytest.raises(AssertionError, match='checksum'):
        rg_full.held('GR-8', list(V.FULL_SEEDS), out, ancestor=yes)
    assert infer_calls == []


def test_held_refuses_when_any_seed_stopped_on_a_non_finite_loss(out, infer_calls):
    import rg_full
    put(rg_full.failed_path(rg_paths.full_root(out), 'GR-8', 2), dict(failure_kind='nonfinite_loss'))
    with pytest.raises(AssertionError, match='VC4'):
        rg_full.held('GR-8', list(V.FULL_SEEDS), out, ancestor=yes)
    assert infer_calls == []


def test_held_refuses_when_any_seed_has_a_recorded_vc3_failure(out, infer_calls):
    import rg_full
    put(rg_paths.full_root(out) / 'held/GR-8/failed__genre__s4.json', dict(failure_kind='vc3'))
    with pytest.raises(AssertionError, match='VC4'):
        rg_full.held('GR-8', list(V.FULL_SEEDS), out, ancestor=yes)
    assert infer_calls == []


# --- provenance -------------------------------------------------------------------------------------------------

def test_code_provenance_counts_an_untracked_py_file_as_dirty(tmp_path, monkeypatch):
    import rg_full
    git = ['git', '-c', 'user.name=t', '-c', 'user.email=t@t']
    subprocess.run(git + ['init', '-q'], cwd=tmp_path, check=True)
    (tmp_path / 'rg_loss.py').write_text('x = 1\n')
    subprocess.run(['git', 'add', '.'], cwd=tmp_path, check=True)
    subprocess.run(git + ['commit', '-qm', 'a'], cwd=tmp_path, check=True)
    monkeypatch.setattr(rg_paths, 'ROUND', tmp_path)
    assert rg_full.code_provenance()['dirty'] is False
    (tmp_path / 'notes.txt').write_text('not code\n')
    assert rg_full.code_provenance()['dirty'] is False
    (tmp_path / 'new_module.py').write_text('y = 2\n')
    assert rg_full.code_provenance()['dirty'] is True


@pytest.mark.parametrize('field', ['rg_held_sha256', 'rb_score_sha256'])
def test_guard_refuses_a_stale_vc1_scorer(out, field):
    import rg_full
    path = out / 'stage_c/regression/vc1_scorer.json'
    record = json.loads(path.read_text())
    record['identity'][field] = 'older-code'
    put(path, record)
    with pytest.raises(AssertionError, match='scorer.*other code'):
        rg_full.require_guard(out, 'GR-8', ancestor=yes)


def put_vb2(out):
    verdict = out / 'stage_b/verdict.json'
    put(out / 'stage_b/rederive_vb2.json', dict(agree=True, complete=True, smoke=False,
        input_hashes={str(verdict): ms_train.head.sha_file(verdict)}))


def call_gate(gate, out):
    import rg_full
    import rg_held
    return (rg_full.require_guard(out, 'GR-8', ancestor=yes) if gate == 'held'
            else rg_held.require_gate(out, 'GR-8'))


@pytest.mark.parametrize('gate', ['held', 'score'])
@pytest.mark.parametrize('bad', ['missing', 'disagree', 'incomplete', 'smoke', 'no_smoke',
                                'stale', 'no_verdict_hash', 'unrelated_hash', 'smoke_verdict'])
def test_stage_c_gates_refuse_invalid_vb2(out, gate, bad):
    path = out / 'stage_b/rederive_vb2.json'
    rec = json.loads(path.read_text())
    if bad == 'missing':
        path.rename(path.with_suffix('.aside'))
    elif bad == 'disagree':
        rec['agree'] = False
    elif bad == 'incomplete':
        rec['complete'] = False
    elif bad == 'smoke':
        rec['smoke'] = True
    elif bad == 'no_smoke':
        del rec['smoke']
    elif bad == 'stale':
        rec['input_hashes'][str(out / 'stage_b/verdict.json')] = 'old-verdict'
    elif bad == 'no_verdict_hash':
        rec['input_hashes'] = {}
    elif bad == 'unrelated_hash':
        rec['input_hashes'] = {str(out / 'other/verdict.json'): ms_train.head.sha_file(out / 'stage_b/verdict.json')}
    else:
        verdict = out / 'stage_b/verdict.json'
        v = json.loads(verdict.read_text())
        v['smoke'] = True
        put(verdict, v)
        rec['input_hashes'][str(verdict)] = ms_train.head.sha_file(verdict)
    if bad != 'missing':
        put(path, rec)
    with pytest.raises(AssertionError, match='VB2'):
        call_gate(gate, out)


@pytest.mark.parametrize('gate', ['held', 'score'])
def test_stage_c_gates_accept_vb2_through_outputs_symlink_and_record_its_hash(out, gate):
    link = out / 'outputs_link'
    link.symlink_to(out, target_is_directory=True)
    rec = json.loads((out / 'stage_b/rederive_vb2.json').read_text())
    rec['input_hashes'] = {str(link / 'stage_b/verdict.json'): ms_train.head.sha_file(out / 'stage_b/verdict.json')}
    put(out / 'stage_b/rederive_vb2.json', rec)
    shas = call_gate(gate, out)
    assert shas['rederive_vb2'] == ms_train.head.sha_file(out / 'stage_b/rederive_vb2.json')



@pytest.fixture
def tiny_training(monkeypatch):
    """No data load or optimizer: capture configs at the trainer's save boundary."""
    import rg_full
    monkeypatch.setattr(rb_student, 'load_scale', lambda scale: (
        dict(first=[0, 1], fit=np.array([0]), dev=np.array([1])),
        {a: dict(cap=np.zeros((2, 2048), dtype=np.float32)) for a in ASPECTS}, {}, {}))
    monkeypatch.setattr(rb_student, 'training_images', lambda *args: np.zeros((2, 2048), dtype=np.float32))
    saved = []
    def run(*args, **kwargs):
        config = rg_full.ct2_d2.make_config()
        saved.append(config)
        root = Path(args[9]).parents[1]
        for aspect in ASPECTS:
            file = rb_student.weights_path(root, config['rg']['arm'], aspect, args[7])
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(b'synthetic best weights')
            put(file.with_suffix('.json'), dict(config=config, weights_sha256=ms_train.head.sha_file(file)))
        return dict(status='ok')
    monkeypatch.setattr(rb_student, 'CT2_MAKE_CONFIG', lambda: {})
    monkeypatch.setattr(rg_full.ct2_d2, 'run', run)
    return saved


@pytest.mark.parametrize('bad', ['missing', 'B-NONE', 'wrong-arm'])
@pytest.mark.parametrize('custom_root', [False, True])
def test_candidate_train_requires_the_stage_b_pick(out, tiny_training, bad, custom_root):
    import rg_full
    (rg_paths.full_root(out) / 'weights').rename(rg_paths.full_root(out) / 'weights.aside')
    path = out / 'stage_b/verdict.json'
    verdict = json.loads(path.read_text())
    if bad == 'missing':
        path.rename(path.with_suffix('.aside'))
    else:
        verdict.update(outcome='B-NONE') if bad == 'B-NONE' else verdict.update(v_held='GR-4')
        put(path, verdict)
    with pytest.raises(AssertionError, match='verdict|B-PICK|v_held'):
        rg_full.train('GR-8', [0], 'cpu', out, root=out / 'custom' if custom_root else None)
    assert tiny_training == []


def test_candidate_train_records_the_stage_b_verdict_in_every_saved_config(out, tiny_training):
    import rg_full
    # No candidate weights yet: training produces a new cohort, rather than resuming fake records.
    (rg_paths.full_root(out) / 'weights').rename(rg_paths.full_root(out) / 'weights.aside')
    rg_full.train('GR-8', [0, 1], 'cpu', out)
    assert len(tiny_training) == 2
    assert all(rec['rg']['stage_b_verdict_sha256'] == ms_train.head.sha_file(out / 'stage_b/verdict.json')
               for rec in tiny_training)
    for seed in (0, 1):
        for aspect in ASPECTS:
            file = rb_student.weights_path(rg_paths.full_root(out), 'GR-8', aspect, seed).with_suffix('.json')
            assert json.loads(file.read_text())['config']['rg']['stage_b_verdict_sha256'] == ms_train.head.sha_file(out / 'stage_b/verdict.json')
    # Resuming must accept the same recorded config and skip the already verified cohorts.
    assert rg_full.train('GR-8', [0, 1], 'cpu', out) == {}
    assert len(tiny_training) == 2


def test_vc1_training_does_not_require_a_stage_b_pick(tmp_path, tiny_training):
    import rg_full
    rg_full.train('R6-5', [0], 'cpu', tmp_path, root=tmp_path / 'vc1_run')
    assert len(tiny_training) == 1
    assert 'stage_b_verdict_sha256' not in tiny_training[0]['rg']


def test_smoke_training_does_not_require_a_stage_b_pick(tmp_path, tiny_training, monkeypatch):
    import rg_full
    monkeypatch.setattr(rb_student, 'SMOKE_ROOT', tmp_path)
    rg_full.train('GR-8', [0], 'cpu', tmp_path, smoke=True)
    assert len(tiny_training) == 1


def test_smoke_trains_every_seed_given_but_at_most_two_epochs(tmp_path, tiny_training, monkeypatch):
    # the held scorer reads all five seeds, also in a smoke
    import rg_full
    monkeypatch.setattr(rb_student, 'SMOKE_ROOT', tmp_path)
    rg_full.train('GR-8', [0, 1, 2, 3, 4], 'cpu', tmp_path, smoke=True)
    assert len(tiny_training) == 5
    with pytest.raises(AssertionError, match='at most 2 epochs'):
        rg_full.train('GR-8', [0], 'cpu', tmp_path, smoke=True, max_epochs=3)


@pytest.mark.parametrize('command', ['train', 'held'])
def test_cli_smoke_passes_the_seeds_through(tmp_path, monkeypatch, command):
    import rg_full
    calls = []
    monkeypatch.setattr(rg_full, command, lambda arm, seeds, *a, **k: calls.append(seeds) or {})
    rg_full.main([command, '--arm', 'GR-8', '--seeds', '0', '1', '2', '3', '4', '--outputs', str(tmp_path), '--smoke'])
    assert calls == [[0, 1, 2, 3, 4]]


@pytest.mark.parametrize('bad', ['missing', 'stale'])
def test_guard_refuses_weights_without_the_current_b_verdict(out, bad):
    import rg_full
    path = rb_student.weights_path(rg_paths.full_root(out), 'GR-8', 'genre', 4).with_suffix('.json')
    record = json.loads(path.read_text())
    if bad == 'missing':
        del record['config']['rg']['stage_b_verdict_sha256']
    else:
        record['config']['rg']['stage_b_verdict_sha256'] = 'old-verdict'
    put(path, record)
    with pytest.raises(AssertionError, match='B verdict'):
        rg_full.require_guard(out, 'GR-8', ancestor=yes)


@pytest.mark.parametrize('failed_seed', [0, 1])
def test_train_cli_exits_nonzero_with_the_vc4_cause_when_any_seed_fails(out, tiny_training, monkeypatch, capsys, failed_seed):
    import rg_full
    (rg_paths.full_root(out) / 'weights').rename(rg_paths.full_root(out) / 'weights.aside')
    def run(*args, **kwargs):
        if args[7] == failed_seed:
            raise rg_full.NonFiniteLoss('genre')
        return dict(status='ok')
    monkeypatch.setattr(rg_full.ct2_d2, 'run', run)
    with pytest.raises(SystemExit) as error:
        rg_full.main(['train', '--arm', 'GR-8', '--seeds', '0', '1', '--device', 'cpu', '--outputs', str(out)])
    assert error.value.code == 1
    text = capsys.readouterr().out
    assert f'GR-8 s{failed_seed}' in text and 'non-finite loss in the genre head' in text and 'VC4' in text
    record = json.loads(rg_full.failed_path(rg_paths.full_root(out), 'GR-8', failed_seed).read_text())
    assert record['status'] == 'failed' and record['failure_kind'] == 'nonfinite_loss'
