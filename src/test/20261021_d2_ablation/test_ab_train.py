"""CPU loss arithmetic, scoped overrides, and frozen runner plumbing."""
import copy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

import ab_paths
import ab_rows
import ab_train

ms = ab_paths.ms_train


@pytest.fixture
def batch():
    torch.manual_seed(91)
    models = nn.ModuleDict({s: nn.Sequential(nn.Linear(6, 6), nn.Tanh())
                            for s in ('img', 'cap')})
    return dict(models=models, img=F.normalize(torch.randn(7, 6), dim=1),
                cap=F.normalize(torch.randn(7, 6), dim=1),
                teacher=F.normalize(torch.randn(7, 6), dim=1),
                pairs=np.array([[0, 1], [2, 3], [1, 4]], dtype=np.int64),
                weights=np.array([0.2, 1.0, 0.7], dtype=np.float32),
                row=copy.deepcopy(ab_rows.ROWS['R0']['base']), aspect='emotion')


@pytest.mark.parametrize('base', ['D2', 'T1'])
def test_default_loss_is_bitwise_original(batch, base):
    batch['row'] = copy.deepcopy(ab_paths.ms_rows.ROWS[base])
    expected = ms.head_loss(**batch)
    actual = ab_train.switched_head_loss(**batch)
    assert torch.equal(actual, expected)


def symmetric(x, y, w=None):
    logits = x @ y.T / ms.CONFIG['temperature']
    ids = torch.arange(len(x))
    losses = (F.cross_entropy(logits, ids, reduction='none') +
              F.cross_entropy(logits.T, ids, reduction='none')) / 2
    return losses.mean() if w is None else (losses * w).sum() / w.sum()


@pytest.mark.parametrize('name', ['R3', 'R5', 'R6', 'R7'])
@pytest.mark.parametrize('dev', [False, True])
def test_switched_loss_is_sum_of_kept_terms(batch, name, dev):
    switches = ab_rows.ROWS[name]['switches']
    a, b = torch.from_numpy(batch['pairs']).unbind(1)
    ids = torch.cat((a, b))
    z = batch['models']['img'](batch['img'][ids])
    ia, ib = z.chunk(2)
    ca, cb = batch['models']['cap'](batch['cap'][ids]).chunk(2)
    w = torch.from_numpy(batch['weights'])
    buddy = symmetric(ia, ib, w)
    if switches['endpoints'] == 'both':
        buddy = (buddy + symmetric(ia, cb, w) + symmetric(ca, ib, w) +
                 symmetric(ca, cb, w)) / 4
    logits = z @ batch['teacher'][ids].T / 0.05
    duplicate = ids[:, None] == ids[None, :]
    duplicate.fill_diagonal_(False)
    expected = 0.5 * F.cross_entropy(logits.masked_fill(duplicate, -torch.inf),
                                   torch.arange(len(ids)))
    if switches['buddy']:
        expected = buddy + expected
    if switches['endpoints'] == 'both' and switches['own']:
        expected = expected + (symmetric(ia, ca) + symmetric(ib, cb)) / 2
    actual = ab_train.switched_head_loss(**batch, switches=switches,
                                         include_mask_penalty=not dev)
    torch.testing.assert_close(actual, expected)


def test_image_only_never_calls_caption_model(batch):
    def forbidden(*args):
        raise AssertionError('caption forward was called')
    batch['models']['cap'].register_forward_pre_hook(forbidden)
    loss = ab_train.switched_head_loss(**batch, switches=ab_rows.ROWS['R7']['switches'])
    loss.backward()
    assert all(p.grad is None for p in batch['models']['cap'].parameters())
    assert all(p.grad is not None for p in batch['models']['img'].parameters())


@pytest.mark.parametrize('name', list(ab_rows.ROWS))
@pytest.mark.parametrize('fail', [False, True])
def test_override_scope_and_teacher(name, fail):
    original_loss, original_teacher = ms.head_loss, ms.ags_trainimg.aspect_images
    inputs = {a: {'img': np.full((7, 6), i, dtype=np.float32)}
              for i, a in enumerate(ab_paths.ms_rows.ASPECTS)}
    recipe = ab_rows.resolve(name, 0)
    def run(*args, **kwargs):
        switched = recipe['switches'] != ab_rows.DEFAULT_SWITCHES
        assert (ms.head_loss is not original_loss) == switched
        assert (ms.ags_trainimg.aspect_images is not original_teacher) == (name == 'R3')
        if name == 'R3':
            for a, key in ms.head.ASPECTS.items():
                    assert ms.ags_trainimg.aspect_images(args[0], key) is inputs[a]['img']
        if fail:
            raise RuntimeError('runner failed')
        return 'ran'
    with patch.object(ms, 'run', run):
        if fail:
            with pytest.raises(RuntimeError, match='runner failed'):
                ab_train.run_cohort({}, inputs, {}, {}, recipe, name,
                                    list(inputs), 0, torch.device('cpu'), Path('/unused'))
        else:
            assert ab_train.run_cohort({}, inputs, {}, {}, recipe, name,
                                       list(inputs), 0, torch.device('cpu'), Path('/unused')) == 'ran'
    assert ms.head_loss is original_loss
    assert ms.ags_trainimg.aspect_images is original_teacher


def test_paths_and_imports():
    assert ab_paths.ROUND == Path(__file__).resolve().parent
    assert ab_paths.MAIN_ROUND == Path('/project/CoSiR/src/test/20261021_d2_ablation')
    assert ab_paths.OUTPUTS == ab_paths.MAIN_ROUND / 'outputs'
    ab_paths.check_imports()


@pytest.mark.parametrize('name', ['R9', 'R10', 'R11'])
def test_missing_graph_store_fails_clearly(tmp_path, monkeypatch, name):
    monkeypatch.setattr(ab_train, 'MAIN_ROUND', tmp_path / 'missing')
    with pytest.raises(AssertionError, match=str(tmp_path / 'missing/graphs')):
        ab_train.require_graph_store(ab_rows.resolve(name, 0))


def test_frozen_sample_path_is_scoped(monkeypatch):
    original = ab_paths.pav_sample.SAMPLE
    def load():
        assert ab_paths.pav_sample.SAMPLE == ab_paths.SAMPLE
        return 'sample'
    monkeypatch.setattr(ms.head, 'load_frozen', load)
    assert ab_train.load_sample() == 'sample'
    assert ab_paths.pav_sample.SAMPLE == original


def test_runner_cohorts_seed_graphs_and_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(ab_train, 'MAIN_ROUND', tmp_path)
    (tmp_path / 'graphs').mkdir()
    for file in ['manifest.json', 'emotion.npz', 'style.npz', 'genre.npz']:
        (tmp_path / 'graphs' / file).touch()
    monkeypatch.setattr(ab_train, 'load_sample', lambda: {})
    monkeypatch.setattr(ms, 'disjoint_split', lambda s: ('fit', 'dev'))
    monkeypatch.setattr(ms.head, 'load_inputs', lambda s, a, k: dict(kind=k, aspect=a))
    loads, runs = [], []
    def graph(s, base, aspect, rows):
        loads.append((base, aspect, rows))
        return {}, {}
    monkeypatch.setattr(ms.head, 'load_graph', graph)
    def run(*args, **kwargs):
        runs.append(args)
        return {}
    monkeypatch.setattr(ab_train, 'run_cohort', run)
    ab_train.train(['R1', 'R4', 'R7', 'R10', 'R12'], [0, 2], 'cpu', tmp_path / 'out',
                   max_epochs=1)
    assert len(runs) == 14  # separate: six, other four rows: two each
    assert all(len(args[6]) == 1 for args in runs if args[5] == 'R1')
    assert all(len(args[6]) == 3 for args in runs if args[5] != 'R1')
    assert all(args[9] == tmp_path / 'out/train' for args in runs)
    assert all(args[1]['emotion']['kind'] == 'prompted' for args in runs if args[5] == 'R12')
    assert {row[0] for base, a, row in loads if base == tmp_path} == {'rand_s0', 'rand_s2'}
    meta = json.loads((tmp_path / 'out/train/R10/ab_row.json').read_text())
    assert meta['store'] == 'ab'
    assert meta['graphs_by_seed'] == {str(s): dict.fromkeys(ab_paths.ms_rows.ASPECTS, f'rand_s{s}')
                                      for s in (0, 2)}
    meta = json.loads((tmp_path / 'out/train/R7/ab_row.json').read_text())
    assert meta['caption_side'] == 'not scored'
    for row, side in (('R4', 'not scored'), ('R1', 'scored'), ('R12', 'scored')):  # R4: no loss on the caption net
        assert json.loads((tmp_path / f'out/train/{row}/ab_row.json').read_text())['caption_side'] == side


def test_metadata_refuses_changed_switches(tmp_path):
    ab_train.write_row_metadata(tmp_path, 'R5', [0])
    path = tmp_path / 'train/R5/ab_row.json'
    metadata = json.loads(path.read_text())
    metadata['switches']['buddy'] = 1.0
    path.write_text(json.dumps(metadata))
    with pytest.raises(AssertionError, match='row metadata mismatch'):
        ab_train.write_row_metadata(tmp_path, 'R5', [0])


def test_output_guards_and_smoke_epochs(tmp_path, monkeypatch):
    for root in (ab_paths.MAIN_ROUND.parent, ab_paths.ROUND.parent):
        with pytest.raises(AssertionError, match='earlier round'):
            ab_train.validate_outputs(root / '20261019_model_sheet/outputs', False)
    monkeypatch.setattr(ab_train, 'SMOKE_ROOT', tmp_path)
    assert ab_train.validate_outputs(tmp_path / 'out', True) == tmp_path / 'out'
    with pytest.raises(AssertionError, match='scratchpad'):
        ab_train.validate_outputs(tmp_path.parent / 'elsewhere', True)
    with pytest.raises(AssertionError, match='2 epochs'):
        ab_train.train(['R0'], [0], 'cpu', tmp_path / 'out', smoke=True, max_epochs=1)
