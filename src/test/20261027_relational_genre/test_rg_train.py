"""Synthetic CPU checks of ticket 02's build and trainer patch seams."""
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

import rg_loss
import rg_train as train
import rg_variants as V


def test_anchor_build_preserves_sheet_parameters_and_global_rng():
    recipe = train.resolved(0)
    row, aspects = recipe['base'], list(train.ASPECTS)
    torch.manual_seed(19)
    expected, _ = train.ms_train.build_model(row, aspects, torch.device('cpu'))
    expected_rng = torch.get_rng_state().clone()
    teacher = F.normalize(torch.arange(4100 * 4, dtype=torch.float32).reshape(4100, 4) + 1, dim=1)
    genre = rg_loss.GenreLoss(V.switches('GQ-8'), seed=19)
    torch.manual_seed(19)
    actual, _ = train.build_model_with(genre, train.ms_train.build_model, lambda: teacher)(
        row, aspects, torch.device('cpu'))
    assert list(actual.state_dict()) == list(expected.state_dict())
    assert all(torch.equal(value, expected.state_dict()[key]) for key, value in actual.state_dict().items())
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert torch.equal(genre.anchors, teacher[rg_loss.draw_anchors(len(teacher), 19)])
    assert torch.equal(genre.anchor_ids, rg_loss.draw_anchors(len(teacher), 19))


def test_anchors_follow_graph_fit_order_not_global_teacher_positions(monkeypatch):
    rng = np.random.default_rng(5)
    teacher = rng.standard_normal((4200, 6)).astype(np.float32)
    teacher /= np.linalg.norm(teacher, axis=1, keepdims=True)
    fit = rng.permutation(4200)[:4100].astype(np.int64)
    graph = {'fit_node_ids': fit}
    monkeypatch.setattr(train.ms_train.ags_trainimg, 'aspect_images', lambda *_: teacher)
    genre = rg_loss.GenreLoss(V.switches('GQ-8'), seed=3)

    def factory(*_):
        return nn.ModuleDict(), {'genre': nn.ModuleDict()}

    train.build_model_with(genre, factory, lambda: train.fit_teacher_rows({}, graph, 'cpu'))(
        {}, ['genre'], torch.device('cpu'))
    expected = torch.from_numpy(teacher[fit])[rg_loss.draw_anchors(len(fit), 3)]
    assert torch.equal(genre.anchors, expected)
    assert not torch.equal(genre.anchors, torch.from_numpy(teacher)[genre.anchor_ids])
    teacher[fit[0]] *= 2
    with pytest.raises(AssertionError, match='unit norm'):
        train.fit_teacher_rows({}, graph, 'cpu')


def test_cohort_routes_every_fit_and_dev_loss_through_rg_and_restores_hooks(monkeypatch):
    monkeypatch.setattr(train, 'atomic_record', lambda *_: None)
    run_dir = Path(__file__).parent / 'unused_test_run'
    recipe = train.resolved(0)
    graph = {'rows': {recipe['base']['graphs']['genre']: {'fit': (np.array([[0, 1]]), np.ones(1))}}}
    calls = []
    original_loss, original_build, original_save = (train.ms_train.head_loss, train.ms_train.build_model,
                                                     train.ms_train.save_output)

    def switched(*args, **kwargs):
        calls.append((args[6], args[9]))
        return torch.tensor(float('nan') if len(calls) == 6 else 1., requires_grad=True)

    def frozen_loop(*args):
        for aspect in train.ASPECTS:
            for fit in (True, False):
                train.ms_train.head_loss(models={'img': nn.Identity()}, img=torch.ones(2, 3),
                                         cap=torch.ones(2, 3), teacher=torch.ones(2, 3),
                                         pairs=np.array([[0, 1]]), weights=np.ones(1),
                                         row=recipe['base'], aspect=aspect, include_mask_penalty=fit)
        pytest.fail('nonfinite loss must stop the cohort')

    monkeypatch.setattr(rg_loss, 'switched_head_loss', switched)
    monkeypatch.setattr(train.ms_train, 'run', frozen_loop)
    record = train.train_cohort({}, {}, {'genre': graph}, {}, recipe, 'GN-8', 0, 'cpu', run_dir,
                                p=None, p_sha256='a' * 64, max_epochs=2)
    assert calls == [(a, fit) for a in train.ASPECTS for fit in (True, False)]
    assert record['status'] == 'failed' and record['failure_kind'] == 'nonfinite_loss'
    assert train.ms_train.head_loss is original_loss
    assert train.ms_train.build_model is original_build
    assert train.ms_train.save_output is original_save
