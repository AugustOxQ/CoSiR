"""Synthetic CPU checks of ticket 02's build and trainer patch seams."""
import hashlib
import json
import subprocess
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
    genre = rg_loss.GenreLoss(V.switches('GQ-8'), seed=19, n_dev=100)
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
    genre = rg_loss.GenreLoss(V.switches('GQ-8'), seed=3, n_dev=100)

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
    assert record['rg_code_sha256'] == train.rg_code_sha256()
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


@pytest.mark.parametrize('mismatch', [False, True])
def test_va1_training_pass_requires_every_aspects_p_bits(tmp_path, monkeypatch, mismatch):
    monkeypatch.setattr(train, 'check_outputs', lambda out, _: out)
    monkeypatch.setattr(train, 'run_variant', lambda *a, **k: {'status': 'ok'})
    monkeypatch.setattr(train, 'states_equal', lambda *_: False)  # reported only
    monkeypatch.setattr(train.ms_train.head, 'sha_file', lambda *_: 'state')
    monkeypatch.setattr(train.paths, 'RB_PART1', tmp_path / 'rb')
    monkeypatch.setattr(train.paths, 'frozen_run', lambda arm, seed: tmp_path / 'eg' / arm / f's{seed}')
    for arm, seed in ((train.BASE, 0), ('GR-8', 0), (train.BASE, 1)):
        folder = tmp_path / 'rb/train' / train.BASE if (arm, seed) == (train.BASE, 0) else train.paths.frozen_run(arm, seed)
        folder.mkdir(parents=True)
        record = dict(status='ok', variant=arm, seed=seed, p_sha256=train.paths.P_SHA256)
        (folder / 'run.json').write_text(json.dumps(record))
        for aspect in train.ASPECTS:
            (folder / f'{aspect}__s0.json').write_text(json.dumps(record))

    def read_export(path, _n):
        changed = mismatch and path.name == 'genre__P.npz' and path.parent.name == 'va1_GR-8_s0'
        return np.array([[1., 1e-7 if changed else 0.]], dtype=np.float32)
    monkeypatch.setattr(train, 'read_export', read_export)
    result = train.regress_va1(None, tmp_path / 'out', 'cpu')
    assert result['pass'] is (not mismatch)
    assert result['items']['GR-8_s0']['pass'] is (not mismatch)
    assert result['rg_code_sha256'] == train.rg_code_sha256()


def test_rg_code_hashes_are_the_three_files_bytes():
    expected = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                [Path(train.__file__), Path(rg_loss.__file__), Path(V.__file__)]}
    assert train.rg_code_sha256() == expected


@pytest.mark.parametrize('status', ['ok', 'failed'])
def test_cached_run_cannot_be_relabelled_as_current_code(tmp_path, monkeypatch, status):
    monkeypatch.setattr(train, 'check_device', lambda _: torch.device('cpu'))
    monkeypatch.setattr(train, 'load_graphs', lambda *_: ({}, {}))
    monkeypatch.setattr(train, 'input_identity', lambda *_: {})
    monkeypatch.setattr(train, 'verify_cached_run', lambda *a, **k: None)
    data = type('Data', (), {'get': lambda self, **_: ({}, {}, None, train.paths.P_SHA256)})()
    folder = train.paths.run_dir(tmp_path, 'GR-16', 0)
    folder.mkdir(parents=True)
    record = dict(variant='GR-16', seed=0, switches=V.switches('GR-16'), smoke=False,
                  max_epochs=train.ms_train.CONFIG['max_epochs'], p_sha256=train.paths.P_SHA256,
                  input_hashes={}, status=status, failure_kind='nonfinite_loss', rg_code_sha256={'old': 'code'})
    (folder / 'run.json').write_text(json.dumps(record))
    with pytest.raises(AssertionError, match='rg_code_sha256.*move.*aside'):
        train.run_variant(data, 'GR-16', 0, tmp_path, 'cpu')


def eq_case(tmp_path, monkeypatch):
    pairs = np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int64)
    recipe = train.resolved(0)
    graph = dict(fit_node_ids=np.arange(3), rows={recipe['base']['graphs']['genre']: {'fit': (pairs, np.ones(3))}})
    x = torch.eye(3, dtype=torch.float32)
    monkeypatch.setattr(train, 'check_device', lambda _: torch.device('cpu'))
    monkeypatch.setattr(train, 'load_graphs', lambda *_: ({'genre': graph}, {}))
    monkeypatch.setattr(train, 'input_identity', lambda *_: {})
    monkeypatch.setattr(train, 'fit_teacher_rows', lambda *_: x)
    monkeypatch.setattr(train.ms_train, 'build_model', lambda *_: (nn.Identity(), {'genre': {'img': nn.Identity()}}))
    data = type('Data', (), {'get': lambda self, **_: ({}, {'img': x.numpy()}, None, train.paths.P_SHA256)})()
    return data


def test_eq_records_current_source_hashes_and_keeps_matching_file(tmp_path, monkeypatch):
    data = eq_case(tmp_path, monkeypatch)
    record = train.equivalence(data, tmp_path)
    assert record['pass'] is True
    assert record['eq_code_sha256'] == train.eq_code_sha256()
    path = tmp_path / 'regression/eq.json'
    before = path.read_bytes()
    assert train.equivalence(data, tmp_path) == record
    assert path.read_bytes() == before


@pytest.mark.parametrize('file', ['rg_loss.py', 'reference', 'missing'])
def test_eq_refuses_stale_kept_source_hashes(tmp_path, monkeypatch, file):
    data = eq_case(tmp_path, monkeypatch)
    train.equivalence(data, tmp_path)
    path = tmp_path / 'regression/eq.json'
    old = json.loads(path.read_text())
    if file == 'missing':
        old.pop('eq_code_sha256')
    else:
        key = 'rg_loss.py' if file == 'rg_loss.py' else next(k for k in old['eq_code_sha256'] if k != 'rg_loss.py')
        old['eq_code_sha256'][key] = 'stale'
    path.write_text(json.dumps(old))
    before = path.read_bytes()
    with pytest.raises(AssertionError, match='EQ.*code.*move.*aside.*rerun'):
        train.equivalence(data, tmp_path)
    assert path.read_bytes() == before


@pytest.mark.parametrize('seed', [3, 19])
def test_anchor_build_uses_the_model_seed_and_a_private_cpu_generator(seed):
    teacher = F.normalize(torch.arange(4200 * 4, dtype=torch.float32).reshape(4200, 4) + 1, dim=1)
    genre = rg_loss.GenreLoss(V.switches('GQ-8'), seed=seed, n_dev=100)
    factory = lambda *_: (nn.ModuleDict(), {'genre': nn.ModuleDict()})
    torch.manual_seed(999)  # distinct from the model seed
    before = torch.get_rng_state().clone()
    train.build_model_with(genre, factory, lambda: teacher)({}, ['genre'], 'cpu')
    expected = torch.randperm(len(teacher), generator=torch.Generator(device='cpu').manual_seed(seed))[:4096]
    other = torch.randperm(len(teacher), generator=torch.Generator(device='cpu').manual_seed(seed + 1))[:4096]
    assert torch.equal(genre.anchor_ids, expected)
    assert not torch.equal(genre.anchor_ids, other)
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.parametrize('problem', ['no_dev', 'same_rows', 'duplicate_ids'])
def test_build_hook_runs_anchor_init_checks(monkeypatch, problem):
    teacher = F.normalize(torch.ones(4100, 3), dim=1)
    genre = rg_loss.GenreLoss(V.switches('GQ-8'), seed=3, n_dev=4100 if problem == 'same_rows' else None if problem == 'no_dev' else 100)
    if problem == 'duplicate_ids':
        monkeypatch.setattr(rg_loss, 'draw_anchors', lambda *_: torch.zeros(4096, dtype=torch.long))
    factory = lambda *_: (nn.ModuleDict(), {'genre': nn.ModuleDict()})
    with pytest.raises(AssertionError):
        train.build_model_with(genre, factory, lambda: teacher)({}, ['genre'], 'cpu')


def test_term_means_group_calls_and_pairs_by_epoch(monkeypatch):
    genre = rg_loss.GenreLoss(V.switches('GR-16'), term_log={})
    values = iter([2., 4., 100., 8., 12.])
    def switched(*a, **k):
        genre.term_log['relational'] = torch.tensor(next(values))
        return torch.tensor(1., requires_grad=True)
    monkeypatch.setattr(rg_loss, 'switched_head_loss', switched)
    loss = train.TrainingLoss(genre, {}, 3)
    def call(n, fit=True):
        loss({}, torch.ones(3, 2), None, np.zeros((n, 2), dtype=np.int64), None, {}, 'genre', include_mask_penalty=fit)
    call(2); call(1); call(2, fit=False); call(1); call(2)
    means = loss.means()
    assert [(e['epoch'], e['calls'], e['pairs'], e['relational']) for e in means] == [(1, 2, 3, 3.), (2, 2, 3, 10.)]


@pytest.mark.parametrize('trained_nonfinite', [False, True])
def test_diagnostic_nonfinite_terms_are_counted_but_trained_loss_stops(monkeypatch, trained_nonfinite):
    genre = rg_loss.GenreLoss(V.LOSS_SCALE_SWITCHES, term_log={})
    genre.anchors, genre.anchor_ids = torch.ones(2, 3), torch.arange(2)
    def switched(models, img, *a, **k):
        models['img'](img)
        genre.term_log.update(direct=torch.tensor(1.), relational=torch.tensor(2.))
        return torch.tensor(float('nan') if trained_nonfinite else 3., requires_grad=True)
    monkeypatch.setattr(rg_loss, 'switched_head_loss', switched)
    monkeypatch.setattr(rg_loss, 'anchors_loss', lambda *_: torch.tensor(float('inf')))
    loss = train.TrainingLoss(genre, {}, 1, diagnostic=True)
    def call():
        return loss({'img': nn.Identity()}, torch.ones(2, 3), None, np.array([[0, 1]]), None, {}, 'genre', torch.ones(2, 3))
    if trained_nonfinite:
        with pytest.raises(train.NonFiniteLoss):
            call()
        assert loss.steps == []
    else:
        assert float(call().detach()) == 3.
        assert float(call().detach()) == 3.  # continues to another epoch
        assert len(loss.steps) == 2
        assert all(step['anchors'] is None and step['nonfinite_counts']['anchors'] == 1 for step in loss.steps)
        assert all(e['anchors'] is None and e['nonfinite_counts']['anchors'] == 1 for e in loss.means())
        json.dumps(loss.steps, allow_nan=False)
        json.dumps(loss.means(), allow_nan=False)


@pytest.mark.parametrize('reader', ['train', 'score'])
def test_dirty_flags_only_include_round_python_files(tmp_path, monkeypatch, reader):
    import rg_score
    def git(*args):
        return subprocess.run(['git', '-C', str(tmp_path), *args], check=True, capture_output=True, text=True).stdout.strip()
    git('init', '-q')
    folder = tmp_path / 'round'
    folder.mkdir()
    (folder / 'module.py').write_text('x = 1\n')
    doc = tmp_path / 'notes.md'
    doc.write_text('notes\n')
    other = tmp_path / 'other.py'
    other.write_text('x = 1\n')
    git('add', '.')
    git('-c', 'user.name=Test', '-c', 'user.email=test@example.com', 'commit', '-qm', 'fixture')
    head = git('rev-parse', 'HEAD')
    monkeypatch.setattr(train, '__file__', str(folder / 'rg_train.py'))
    monkeypatch.setattr(rg_score.rg_paths, 'ROUND', folder)
    monkeypatch.setattr(rg_score, 'scorer_code_sha256', lambda: 'hash')
    def state():
        return train.code_commit() if reader == 'train' else (rg_score.code_state()['code_commit'], rg_score.code_state()['code_dirty'])
    assert state() == (head, False)
    doc.write_text('edited\n')
    other.write_text('x = 2\n')
    assert state() == (head, False)
    new = folder / 'untracked.py'
    new.write_text('x = 2\n')
    assert state() == (head, True)
    new.rename(folder / 'untracked.txt')
    (folder / 'module.py').write_text('x = 3\n')
    assert state() == (head, True)



def test_interrupted_va3_rerun_requires_its_origin_code_hashes(tmp_path, monkeypatch):
    monkeypatch.setattr(train, 'check_device', lambda _: torch.device('cpu'))
    monkeypatch.setattr(train, 'load_graphs', lambda *_: ({}, {}))
    monkeypatch.setattr(train, 'input_identity', lambda *_: {})
    monkeypatch.setattr(train, 'code_commit', lambda: ('head', False))
    monkeypatch.setattr(train, 'train_cohort', lambda *a, **k: {'status': 'ok'})
    data = type('Data', (), {'get': lambda self, **_: ({}, {}, None, train.paths.P_SHA256)})()
    folder = tmp_path / 'replaced/GR-16/s0-origin'
    folder.mkdir(parents=True)
    record = dict(variant='GR-16', seed=0, switches=V.switches('GR-16'), smoke=False,
                  max_epochs=train.ms_train.CONFIG['max_epochs'], input_hashes={}, status='failed',
                  failure_kind='va3', code_commit='head', code_dirty=False, rg_code_sha256={'old': 'code'})
    (folder / 'run.json').write_text(json.dumps(record))
    with pytest.raises(AssertionError, match='origin rg_code_sha256'):
        train.run_variant(data, 'GR-16', 0, tmp_path, 'cpu')


@pytest.mark.parametrize('arm, rerun_of', [('GR-16', None), ('GR-16', '/old/va3'), (V.LOSS_SCALE, None)])
def test_every_run_kind_records_training_code_hashes(tmp_path, arm, rerun_of):
    genre = rg_loss.GenreLoss(V.LOSS_SCALE_SWITCHES if arm == V.LOSS_SCALE else V.switches(arm), term_log={})
    loss = train.TrainingLoss(genre, {}, 1)
    exporter = type('Exporter', (), {'info': {}, 'exports': {}})()
    record = train.write_run(tmp_path, arm, 0, 'failed', genre=genre, loss=loss, exporter=exporter,
                             p_sha256=train.paths.P_SHA256, smoke=False, max_epochs=1, seconds=0,
                             input_hashes={}, cause='nonfinite', failure_kind='nonfinite_loss', rerun_of=rerun_of)
    assert record['rg_code_sha256'] == train.rg_code_sha256()
    assert json.loads((tmp_path / 'run.json').read_text())['rg_code_sha256'] == record['rg_code_sha256']



def test_eq_hashes_cover_both_defining_files():
    expected = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                (Path(rg_loss.__file__), Path(train.ms_train.__file__))}
    assert train.eq_code_sha256() == expected
