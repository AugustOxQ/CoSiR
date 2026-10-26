"""Tiny CPU checks of the full-scale trainer's loss wiring and the held guard (ticket 06; verdict code)."""
import json

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

import ab_train
import rb_rows
import rg_variants as V
from ab_paths import ms_train


# --- the held guard ---------------------------------------------------------------------------------------------

def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def out(tmp_path):
    put(tmp_path / 'stage_b/verdict.json', dict(outcome='B-PICK', v_held='GR-8', v_held_switches=V.switches('GR-8')))
    for name in ('vc1_train', 'vc1_scorer'):
        put(tmp_path / f'stage_c/regression/{name}.json', {'pass': True})
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
    with pytest.raises(AssertionError, match='v_held'):
        rg_full.require_guard(out, 'GR-16', ancestor=yes)


def test_guard_refuses_other_switches_than_the_arms(out):
    import rg_full
    put(out / 'stage_b/verdict.json',
        dict(outcome='B-PICK', v_held='GR-8', v_held_switches=V.switches('GR-16')))
    with pytest.raises(AssertionError, match='switches'):
        rg_full.require_guard(out, 'GR-8', ancestor=yes)


def test_guard_refuses_an_arm_outside_the_round(out):
    import rg_full
    with pytest.raises(AssertionError):
        rg_full.require_guard(out, 'K4', ancestor=yes)


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
