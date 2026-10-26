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
    original_loss, original_build, original_save = (
        train.ms_train.head_loss, train.ms_train.build_model, train.ms_train.save_output)

    def switched(*args, **kwargs):
        assert kwargs['switches'] == recipe['switches']
        assert kwargs['genre'].switches == V.switches('GN-8')
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


def test_cohort_gq_uses_fit_anchors_with_the_reviewed_dev_tensor_guard(monkeypatch):
    rng = np.random.default_rng(11)
    teacher = rng.standard_normal((4200, 6)).astype(np.float32)
    teacher /= np.linalg.norm(teacher, axis=1, keepdims=True)
    order = rng.permutation(len(teacher)).astype(np.int64)
    recipe = train.resolved(0)
    pairs, weights = np.array([[0, 1], [0, 2]], dtype=np.int64), np.ones(2, dtype=np.float32)
    graph = dict(fit_node_ids=order[:4100], dev_node_ids=order[4100:],
                 rows={recipe['base']['graphs']['genre']: {'fit': (pairs, weights)}})
    observations = []

    def factory(_row, aspects, _device):
        models = nn.ModuleDict({s: nn.Sequential(nn.Linear(6, 6), nn.Dropout(0.1)) for s in ('img', 'cap')})
        return models, {a: models for a in aspects}

    def frozen_loop(*_args):
        models, views = train.ms_train.build_model(recipe['base'], list(train.ASPECTS), torch.device('cpu'))
        for part in ('fit', 'dev'):
            x = torch.from_numpy(teacher[graph[f'{part}_node_ids']])
            if part == 'fit':
                models.train()
                value = train.ms_train.head_loss(views['genre'], x, x, pairs, weights, recipe['base'],
                                                 'genre', x)
            else:
                models.eval()
                with torch.no_grad():
                    value = train.ms_train.head_loss(views['genre'], x, x, pairs, weights, recipe['base'],
                                                     'genre', x, include_mask_penalty=False)
            observations.append(float(value.detach()))
        # No optimizer, file exports or real training: stop after exercising the loss seam.
        raise train.NonFiniteLoss('synthetic stop')

    monkeypatch.setattr(train, 'atomic_record', lambda *_: None)
    monkeypatch.setattr(train.ms_train, 'build_model', factory)
    monkeypatch.setattr(train.ms_train, 'run', frozen_loop)
    monkeypatch.setattr(train.ms_train.ags_trainimg, 'aspect_images', lambda *_: teacher)
    record = train.train_cohort({}, {}, {'genre': graph}, {}, recipe, 'GQ-8', 0, 'cpu',
                                Path(__file__).parent / 'unused_test_run', p=None,
                                p_sha256='a' * 64, max_epochs=2)
    assert len(observations) == 2 and np.isfinite(observations).all()
    assert record['term_means'][0]['calls'] == 1
    assert record['anchors']['count'] == 4096
