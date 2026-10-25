"""Stage iii trainer (ticket 11): the genre-loss switches, the GA adaptor, and a tiny CPU cohort end to end."""
import copy
import json
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

import eg_paths  # noqa: F401
import eg_train3
import eg_variants as V
import ab_rows
import ab_train
import rb_part1
import rb_probe
import rb_rows
from ab_paths import ms_train

DIM = 6


# ------------------------------------------------------------------ the loss switches
@pytest.fixture
def batch():
    torch.manual_seed(7)
    models = nn.ModuleDict({s: nn.Sequential(nn.Linear(DIM, DIM), nn.Tanh()) for s in ('img', 'cap')})
    return dict(models=models, img=F.normalize(torch.randn(8, DIM), dim=1),
                cap=F.normalize(torch.randn(8, DIM), dim=1),
                teacher=F.normalize(torch.randn(8, DIM), dim=1),
                pairs=np.array([[0, 1], [2, 3], [1, 4], [5, 6]], dtype=np.int64),
                weights=np.array([0.2, 1.0, 0.7, 0.5], dtype=np.float32),
                row=rb_rows.recipe('R6-5')['base'], aspect='genre')


R65 = rb_rows.recipe('R6-5')['switches']


def make_genre(batch, variant, with_adaptor=True):
    """GenreLoss of a variant with a tiny adaptor and the standardizer of the batch's teacher."""
    genre = eg_train3.GenreLoss(V.switches(variant), seed=3)
    if genre.w:
        torch.manual_seed(11)
        genre.adaptor = nn.Sequential(nn.Linear(DIM, DIM), nn.GELU(), nn.Linear(DIM, DIM))
        genre.std = eg_train3.TorchStd(rb_probe.Standardizer(batch['teacher'].numpy()), torch.device('cpu'))
    return genre


def loss(batch, genre, **changes):
    args = dict(batch, **changes)
    return eg_train3.switched_head_loss(**args, switches=R65, genre=genre)


def original(batch, **changes):
    return ab_train.switched_head_loss(**dict(batch, **changes), switches=R65)


def head_output(batch):
    a, b = torch.as_tensor(batch['pairs']).unbind(1)
    ids = torch.cat((a, b))
    return batch['models']['img'](batch['img'][ids]), ids


def test_all_switches_off_is_the_original_loss_bit_for_bit(batch):
    for genre in (None, make_genre(batch, 'R6-5')):
        assert torch.equal(loss(batch, genre), original(batch))


def test_switches_off_with_integer_or_float_multiplier_still_original(batch):
    genre = eg_train3.GenreLoss(dict(m=1.0, lam=0.0, w=0.0), seed=0)
    assert torch.equal(loss(batch, genre), original(batch))


@pytest.mark.parametrize('variant', ['GT-2', 'GT-4', 'GR-0.5', 'GR-2', 'GA-1'])
@pytest.mark.parametrize('aspect', ['emotion', 'style'])
def test_each_switch_leaves_emotion_and_style_as_they_were(batch, variant, aspect):
    genre = make_genre(batch, variant)
    assert torch.equal(loss(batch, genre, aspect=aspect), original(batch, aspect=aspect))


def test_teacher_multiplier_scales_only_the_direct_term(batch):
    z, ids = head_output(batch)
    direct = ms_train.distill_loss(z, batch['teacher'][ids], ids, 'direct')
    for variant, m in (('GT-2', 2), ('GT-4', 4), ('GT-1.5', 1.5)):
        actual = loss(batch, make_genre(batch, variant))
        expected = original(batch) + (0.25 * m - 0.25) * direct
        torch.testing.assert_close(actual, expected)
    assert not torch.equal(loss(batch, make_genre(batch, 'GT-2')), original(batch))


def test_relational_term_is_added_beside_the_direct_term_on_the_same_output(batch):
    z, ids = head_output(batch)
    relational = ms_train.distill_loss(z, batch['teacher'][ids], ids, 'relational')
    for variant, lam in (('GR-0.5', 0.5), ('GR-2', 2)):
        actual = loss(batch, make_genre(batch, variant))
        torch.testing.assert_close(actual, original(batch) + lam * relational)


def test_one_forward_pass_serves_every_term(batch):
    calls = []
    batch['models']['img'].register_forward_hook(lambda *a: calls.append(1))
    loss(batch, make_genre(batch, 'GR-2'))
    assert len(calls) == 1


def test_adaptor_term_is_the_probe_loss_on_the_standardized_teacher(batch):
    genre = make_genre(batch, 'GA-1')
    z, ids = head_output(batch)
    std = rb_probe.Standardizer(batch['teacher'].numpy())
    target = torch.from_numpy(std.fwd(batch['teacher'].numpy()))[ids]
    expected = original(batch) + 1 * rb_probe.loss_fn(genre.adaptor(z), target)
    torch.testing.assert_close(loss(batch, genre), expected)


def test_adaptor_weight_scales_the_term_and_trains_the_adaptor(batch):
    genre = make_genre(batch, 'GA-0.5')
    z, ids = head_output(batch)
    target = torch.from_numpy(rb_probe.Standardizer(batch['teacher'].numpy()).fwd(batch['teacher'].numpy()))[ids]
    expected = original(batch) + 0.5 * rb_probe.loss_fn(genre.adaptor(z), target)
    value = loss(batch, genre)
    torch.testing.assert_close(value, expected)
    value.backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in genre.adaptor.parameters())


def grads(params, value):
    """Gradients of `value` on `params` (None where there is none), then cleared."""
    for p in params:
        p.grad = None
    value.backward()
    out = [None if p.grad is None else p.grad.clone() for p in params]
    for p in params:
        p.grad = None
    return out


@pytest.mark.parametrize('variant,term', [('GR-0.5', 'relational'), ('GR-2', 'relational'), ('GA-1', 'adaptor'),
                                          ('GT-2', 'direct')])
def test_switch_gradients_flow_through_the_head_output_like_the_independent_sum(batch, variant, term):
    """The extra term must reach the image head and the adaptor: a detached z would pass the value tests."""
    genre = make_genre(batch, variant)
    params = list(batch['models'].parameters())
    n_models = len(params)
    if genre.adaptor is not None:
        params += list(genre.adaptor.parameters())
    actual = grads(params, loss(batch, genre))
    z, ids = head_output(batch)
    teacher = batch['teacher'][ids]
    if term == 'relational':
        extra = genre.lam * ms_train.distill_loss(z, teacher, ids, 'relational')
    elif term == 'direct':
        extra = (0.25 * genre.m - 0.25) * ms_train.distill_loss(z, teacher, ids, 'direct')
    else:
        target = torch.from_numpy(rb_probe.Standardizer(batch['teacher'].numpy()).fwd(batch['teacher'].numpy()))[ids]
        extra = genre.w * rb_probe.loss_fn(genre.adaptor(z), target)
    expected = grads(params, original(batch) + extra)
    assert len(actual) == len(expected) and all(a is not None for a in actual)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e)
    plain = grads(params[:n_models], original(batch))
    # the extra term really moves the head (otherwise the comparison above is empty)
    assert any(not torch.allclose(a, p) for a, p in zip(actual[:n_models], plain))


def test_adaptor_is_not_used_when_w_is_zero(batch):
    genre = make_genre(batch, 'GT-2')
    assert genre.adaptor is None
    loss(batch, genre).backward()


def test_torch_standardizer_equals_the_probe_standardizer_bit_for_bit():
    x = np.random.default_rng(0).standard_normal((50, 12)).astype(np.float32)
    std = rb_probe.Standardizer(x)
    t = eg_train3.TorchStd(std, torch.device('cpu'))
    assert torch.equal(t.fwd(torch.from_numpy(x)), torch.from_numpy(std.fwd(x)))


def test_non_finite_loss_raises_the_run_stopping_error(batch):
    batch['img'][0, 0] = float('nan')
    with pytest.raises(eg_train3.NonFiniteLoss, match='genre'):
        eg_train3.checked_head_loss(**batch, switches=R65, genre=make_genre(batch, 'GT-2'))


def test_other_switch_combinations_are_refused(batch):
    with pytest.raises(AssertionError):
        eg_train3.switched_head_loss(**batch, switches=ab_rows.DEFAULT_SWITCHES, genre=make_genre(batch, 'GT-2'))


# ------------------------------------------------------------------ the adaptor
def test_adaptor_build_leaves_the_global_random_streams_unchanged():
    torch.manual_seed(5)
    state = torch.get_rng_state()
    eg_train3.make_adaptor(0, torch.device('cpu'))
    assert torch.equal(torch.get_rng_state(), state)
    expected = torch.rand(4)
    torch.manual_seed(5)
    assert torch.equal(expected, torch.rand(4))      # the next draw is the one a run without the adaptor makes


def test_adaptor_shape_and_seed():
    a, b, c = (eg_train3.make_adaptor(s, torch.device('cpu')) for s in (0, 0, 1))
    assert [type(m).__name__ for m in a] == ['Linear', 'GELU', 'Linear']
    assert a[0].weight.shape == a[2].weight.shape == (2048, 2048)
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))
    assert not torch.equal(a[0].weight, c[0].weight)
    bound = 1 / 2048 ** 0.5
    assert a[0].weight.abs().max() <= bound and a[2].bias.abs().max() <= bound
    assert a[0].weight.std().item() == pytest.approx(bound / 3 ** 0.5, rel=0.02)


def test_adaptor_comes_after_the_models_parameters_and_in_the_genre_view():
    row = rb_rows.recipe('R6-5')['base']
    genre = eg_train3.GenreLoss(V.switches('GA-1'), seed=0)
    torch.manual_seed(0)
    base_models, base_views = ms_train.build_model(row, list(eg_train3.ASPECTS), torch.device('cpu'))
    torch.manual_seed(0)
    models, views = eg_train3.build_model_with(genre, ms_train.build_model)(row, list(eg_train3.ASPECTS),
                                                                             torch.device('cpu'))
    base_params, params = list(base_models.parameters()), list(models.parameters())
    assert len(params) == len(base_params) + 4
    assert all(torch.equal(p, q) for p, q in zip(params, base_params))      # the model is built as without the adaptor
    assert all(p is q for p, q in zip(params[-4:], genre.adaptor.parameters()))
    opt = torch.optim.AdamW(models.parameters(), lr=1e-3, weight_decay=0.05)
    ids = {id(p) for g in opt.param_groups for p in g['params']}
    assert all(id(p) in ids for p in genre.adaptor.parameters())
    assert len(opt.param_groups) == 1 and len(opt.param_groups[0]['params']) == len(params)
    genre_keys = [k for k in views['genre'].state_dict() if k.startswith('adaptor.')]
    assert genre_keys == ['adaptor.0.weight', 'adaptor.0.bias', 'adaptor.2.weight', 'adaptor.2.bias']
    assert not any(k.startswith('adaptor.') for a in ('emotion', 'style') for k in views[a].state_dict())


def test_without_the_adaptor_the_model_is_the_original_object_pair():
    row = rb_rows.recipe('R6-5')['base']
    genre = eg_train3.GenreLoss(V.switches('GR-2'), seed=0)
    models, views = eg_train3.build_model_with(genre, ms_train.build_model)(row, list(eg_train3.ASPECTS),
                                                                             torch.device('cpu'))
    assert list(models) == ['img', 'cap'] and genre.adaptor is None
    assert not any(k.startswith('adaptor.') for k in views['genre'].state_dict())


# ------------------------------------------------------------------ bit comparison
def test_bits_equal_tells_signed_zero_and_nan_payloads_apart():
    a = np.array([0.0, 1.0], dtype=np.float16)
    assert eg_train3.bits_equal(a, a.copy())
    assert not eg_train3.bits_equal(a, np.array([-0.0, 1.0], dtype=np.float16))
    assert not eg_train3.bits_equal(a, a[:1])
    assert not eg_train3.bits_equal(a, a.astype(np.float32))


# ------------------------------------------------------------------ a tiny cohort on the CPU
N_SAMPLE, N_FAKE_P = 19889, 300


def unit(rng, n):
    x = rng.standard_normal((n, 2048)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def fake_pairs(rng, n_nodes, m):
    code = np.unique(rng.integers(0, n_nodes * n_nodes, size=4 * m))
    a, b = code // n_nodes, code % n_nodes
    keep = a < b
    p = np.stack([a[keep], b[keep]], axis=1)[:m].astype(np.int64)
    return p, rng.uniform(0.2, 1.0, size=len(p)).astype(np.float32)


@pytest.fixture(scope='module')
def cohort():
    rng = np.random.default_rng(5)
    subset = np.sort(rng.permutation(N_SAMPLE)[:5000]).astype(np.int64)
    s = dict(first=np.arange(N_SAMPLE), subset_pid=subset, sample_ids=np.arange(3), pid=np.arange(3),
             all_painting_ids=np.array([str(i) for i in range(N_SAMPLE)]), subset_image_indices=np.arange(3))
    fit, dev = ms_train.disjoint_split(s)
    x = dict(img=unit(rng, N_SAMPLE), cap=unit(rng, N_SAMPLE))
    graphs, manifests = {}, {}
    for aspect, row in (('emotion', 'Gres'), ('style', 'Fres05'), ('genre', 'R0o')):
        fp, fw = fake_pairs(rng, len(fit), 1100)
        dp, dw = fake_pairs(rng, len(dev), 300)
        graphs[aspect] = dict(fit_node_ids=fit, dev_node_ids=dev, rows={row: dict(fit=(fp, fw), dev=(dp, dw))})
        manifests[aspect] = dict(identity=dict(method='fake'), archives={aspect: 'x'},
                                 counts={aspect: dict(fit={row: len(fp)}, dev={row: len(dp)})})
    teacher = unit(rng, N_SAMPLE)
    p = dict(img=unit(rng, N_FAKE_P), sha256='f' * 64)
    return s, x, graphs, manifests, teacher, p


def run_cohort(cohort, tmp, variant, seed=0, max_epochs=3):
    s, x, graphs, manifests, teacher, p = cohort
    recipe = rb_part1.resolved('R6-5')
    with patch.object(ms_train.ags_trainimg, 'aspect_images', lambda s_, key: teacher), \
            patch.object(eg_train3, 'N_P', N_FAKE_P):
        return eg_train3.train_cohort(s, {a: x for a in eg_train3.ASPECTS}, graphs, manifests, recipe, variant, seed,
                                      torch.device('cpu'), tmp, p=p, p_sha256='f' * 64, max_epochs=max_epochs)


@pytest.fixture(scope='module')
def runs(cohort, tmp_path_factory):
    out = {}
    for variant in ('R6-5', 'GA-1', 'GT-2'):
        out[variant] = tmp_path_factory.mktemp(variant.replace('-', ''))
        run_cohort(cohort, out[variant], variant)
    return out


def part1_reference(cohort, tmp_path_factory):
    s, x, graphs, manifests, teacher, p = cohort
    tmp = tmp_path_factory.mktemp('part1')
    with patch.object(ms_train.ags_trainimg, 'aspect_images', lambda s_, key: teacher), \
            patch.object(rb_part1, 'N_P', N_FAKE_P):
        torch.manual_seed(0)
        rb_part1.train_cohort(s, {a: x for a in eg_train3.ASPECTS}, graphs, manifests, rb_part1.resolved('R6-5'),
                              'R6-5', list(eg_train3.ASPECTS), torch.device('cpu'), tmp,
                              p=dict(p, cap=p['img'], names=[], counts={}), max_epochs=3, snapshot_epoch=3)
    return tmp / 'part1/train/R6-5'


def load(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def test_base_run_equals_the_part_1_trainer_bit_for_bit(cohort, runs, tmp_path_factory):
    ref = part1_reference(cohort, tmp_path_factory)
    for aspect in eg_train3.ASPECTS:
        for label in ('P', 'old5000'):
            mine = load(runs['R6-5'] / f'{aspect}__{label}.npz')
            assert set(mine) == {'img'}
            assert eg_train3.bits_equal(mine['img'], load(ref / f'{aspect}__s0__{label}.npz')['img']), (aspect, label)
        mine = torch.load(runs['R6-5'] / f'{aspect}.pt', weights_only=True)
        old = torch.load(ref / f'{aspect}__s0.pt', weights_only=True)
        assert list(mine) == list(old) and all(torch.equal(mine[k], old[k]) for k in old)


def test_run_folder_holds_what_iii_run_promises(runs):
    for variant, folder in runs.items():
        record = json.loads((folder / 'run.json').read_text())
        assert record['status'] == 'ok' and record['variant'] == variant and record['seed'] == 0
        assert record['switches'] == V.switches(variant)
        assert set(record['best_epoch']) == set(eg_train3.ASPECTS)
        assert all(1 <= e <= 3 for e in record['best_epoch'].values())
        assert record['p_sha256'] == 'f' * 64 and record['code_commit']
        for aspect in eg_train3.ASPECTS:
            for label, n in (('P', N_FAKE_P), ('old5000', 5000)):
                z = load(folder / f'{aspect}__{label}.npz')
                assert z['img'].dtype == np.float16 and z['img'].shape == (n, 2048)
        assert (folder / 'genre__P_adaptor.npz').exists() == (variant == 'GA-1')


def test_adaptor_is_trained_exported_and_saved_with_the_genre_best_state(runs):
    folder = runs['GA-1']
    state = torch.load(folder / 'genre.pt', weights_only=True)
    keys = [k for k in state if k.startswith('adaptor.')]
    assert keys == ['adaptor.0.weight', 'adaptor.0.bias', 'adaptor.2.weight', 'adaptor.2.bias']
    initial = eg_train3.make_adaptor(0, torch.device('cpu'))
    assert not torch.equal(state['adaptor.0.weight'], initial[0].weight)     # the optimizer moved it
    for aspect in ('emotion', 'style'):
        assert not any(k.startswith('adaptor.') for k in torch.load(folder / f'{aspect}.pt', weights_only=True))
    z = load(folder / 'genre__P_adaptor.npz')
    assert set(z) == {'adaptor'} and z['adaptor'].dtype == np.float16 and z['adaptor'].shape == (N_FAKE_P, 2048)
    adaptor = eg_train3.make_adaptor(0, torch.device('cpu'))
    adaptor.load_state_dict({k[len('adaptor.'):]: v for k, v in state.items() if k.startswith('adaptor.')})
    head = torch.from_numpy(load(folder / 'genre__P.npz')['img'].astype(np.float32))
    with torch.no_grad():
        np.testing.assert_allclose(adaptor(head).numpy(), z['adaptor'].astype(np.float32), rtol=0.05, atol=0.05)


def test_variant_changes_only_what_the_switch_reaches(runs):
    base = load(runs['R6-5'] / 'genre__old5000.npz')['img']
    assert not np.array_equal(base, load(runs['GT-2'] / 'genre__old5000.npz')['img'])
    assert not np.array_equal(base, load(runs['GA-1'] / 'genre__old5000.npz')['img'])


def test_a_non_finite_loss_stops_the_run_with_a_failed_record(cohort, tmp_path):
    def broken(*args, **kwargs):
        raise eg_train3.NonFiniteLoss('genre')
    with patch.object(eg_train3, 'checked_head_loss', broken):
        record = run_cohort(cohort, tmp_path, 'GT-2')
    saved = json.loads((tmp_path / 'run.json').read_text())
    assert record['status'] == saved['status'] == 'failed' and 'non-finite' in saved['cause']
    assert saved['switches'] == V.switches('GT-2') and saved['best_epoch'] == {}
    assert not list(tmp_path.glob('*__P.npz'))


def test_a_vg3_failure_marks_the_run_failed_and_says_which_export(cohort, tmp_path):
    with patch.object(eg_train3, 'vg3_problem', lambda x, n: 'norm error 2'):
        record = run_cohort(cohort, tmp_path, 'GR-2')
    assert record['status'] == 'failed' and record['failure_kind'] == 'vg3' and 'VG3' in record['cause']
    assert 'emotion' in record['cause']


def test_a_vg3_failure_on_the_old_5000_export_is_recorded_without_failing_the_run(cohort, tmp_path):
    """Rule VG3 covers P only; the old-5,000 exports are reported in run.json."""
    with patch.object(eg_train3, 'vg3_problem', lambda x, n: 'norm error 2' if n == eg_train3.N_OLD else None):
        record = run_cohort(cohort, tmp_path, 'GR-2')
    saved = json.loads((tmp_path / 'run.json').read_text())
    assert record['status'] == saved['status'] == 'ok' and 'cause' not in saved
    assert saved['old5000_problems'] and all('old5000' in p for p in saved['old5000_problems'])
    assert len(saved['old5000_problems']) == len(eg_train3.ASPECTS)


def test_a_clean_run_has_an_empty_old_5000_problem_list(runs):
    assert json.loads((runs['GT-2'] / 'run.json').read_text())['old5000_problems'] == []


# ------------------------------------------------------------------ VG4: one rerun after a VG3 failure
class StubData:
    def get(self):
        return None, {}, {}, 'f' * 64


@pytest.fixture
def vg3_runs(tmp_path):
    """run_variant with a fake trainer that fails VG3 every time; `calls` counts the trainings."""
    calls = []
    state = dict(commit=('c0ffee', False))

    def fake_train(s, inputs, graphs, manifests, recipe, variant, seed, device, run_dir, *, max_epochs, rerun_of=None,
                   **kwargs):
        calls.append(rerun_of)
        return eg_train3.write_run(run_dir, variant, seed, 'failed', failure_kind='vg3', cause='VG3: fake',
                                   p_sha256='f' * 64, max_epochs=max_epochs, rerun_of=rerun_of)

    def run(rerun_failed):
        return eg_train3.run_variant(StubData(), 'GR-2', 1, tmp_path, 'cpu', rerun_failed=rerun_failed)

    with patch.object(eg_train3, 'train_cohort', fake_train), patch.object(eg_paths, 'check_imports', lambda: None), \
            patch.object(ab_train, 'require_graph_store', lambda recipe: None), \
            patch.object(ms_train.head, 'load_graph', lambda *a: (None, None)), \
            patch.object(eg_train3, 'code_commit', lambda: state['commit']):
        yield tmp_path, calls, run, state


def test_a_second_rerun_after_a_vg3_rerun_is_refused(vg3_runs):
    outputs, calls, run, _ = vg3_runs
    assert run(False)['status'] == 'failed' and len(calls) == 1
    again = run(True)                                   # VG4: the one rerun
    assert len(calls) == 2 and again['status'] == 'failed'
    moved = list((outputs / 'stage_iii/replaced/GR-2').glob('s1-*'))
    assert len(moved) == 1 and again['rerun_of'] == str(moved[0])
    with pytest.raises(AssertionError, match='already rerun'):
        run(True)
    assert len(calls) == 2 and len(list((outputs / 'stage_iii/replaced/GR-2').glob('s1-*'))) == 1
    assert json.loads((eg_paths.iii_run(outputs, 'GR-2', 1) / 'run.json').read_text())['rerun_of'] == str(moved[0])


def test_a_rerun_is_refused_when_a_vg3_failed_run_was_already_moved_aside(vg3_runs):
    """The record may lack `rerun_of` (an interrupted rerun was retrained as a first run): replaced/ still counts."""
    outputs, calls, run, _ = vg3_runs
    run(False)
    run(True)
    path = eg_paths.iii_run(outputs, 'GR-2', 1) / 'run.json'
    record = json.loads(path.read_text())
    del record['rerun_of']
    path.write_text(json.dumps(record))
    with pytest.raises(AssertionError, match='already rerun'):
        run(True)
    assert len(calls) == 2


def test_a_rerun_needs_the_old_run_to_come_from_the_current_clean_code(vg3_runs):
    outputs, calls, run, state = vg3_runs
    run(False)
    state['commit'] = ('newer', False)
    with pytest.raises(AssertionError, match='same code'):
        run(True)
    state['commit'] = ('c0ffee', True)
    with pytest.raises(AssertionError, match='same code'):
        run(True)
    assert len(calls) == 1 and (eg_paths.iii_run(outputs, 'GR-2', 1) / 'run.json').exists()
    state['commit'] = ('c0ffee', False)
    assert run(True)['rerun_of'] and len(calls) == 2


def test_parity_keeps_the_code_commit_of_every_seed(tmp_path):
    state = dict(commit=('aaa', False))
    def fake_run(data, variant, seed, outputs, device):
        return dict(status='failed', cause='x')
    with patch.object(eg_train3, 'run_variant', fake_run), patch.object(eg_train3, 'code_commit', lambda: state['commit']):
        eg_train3.parity(None, tmp_path, 'cpu', [1])
        state['commit'] = ('bbb', False)
        record = eg_train3.parity(None, tmp_path, 'cpu', [2])
    assert record['seeds']['1']['code_commit'] == 'aaa' and record['seeds']['2']['code_commit'] == 'bbb'
    assert record['code_commit'] == 'bbb'
