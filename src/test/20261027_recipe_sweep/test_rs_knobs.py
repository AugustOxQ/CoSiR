"""The knob layer: G enters no patch; each knob changes only its own part (tiny CPU tensors, patched objects)."""
import inspect
import re
import sys
from contextlib import ExitStack
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

import rs_knobs as K
import rs_sheet as S
import ab_train
import eg_train3 as eg
import rg_loss
import rg_train
import rg_variants as V
from ab_paths import ms_train

CPU = torch.device('cpu')
ALL_KNOBS = [k for k in S.KNOBS]


def state():
    """Every object the layer may patch, by value or identity."""
    return dict(config=dict(ms_train.CONFIG), arm=dict(V.ARMS['GS-0.02']), weight=V.TEACHER_WEIGHT,
                switched=ab_train.switched_head_loss, resolved=rg_train.resolved, load_graphs=rg_train.load_graphs,
                infer_img=eg.infer_img, trunk=ms_train.TrunkSide)


def touched(cfg):
    """The names of `state()` that differ inside the patches of `cfg`; and that they are restored after."""
    before = state()
    with ExitStack() as stack:
        K.patches(cfg, stack, scale='sample', outputs='o')
        inside = state()
    assert state() == before
    return {name for name in before if inside[name] != before[name]}


def test_g_enters_no_patch():
    stack = ExitStack()
    assert K.patches(S.config(), stack, scale='sample') == {}
    assert not stack._exit_callbacks
    assert K.describe(S.config()) == {}
    assert K.patches(S.config(), ExitStack(), scale='full') == {}
    assert touched(S.config()) == set()


@pytest.mark.parametrize('knob,parts', [
    ('k1', {'switched'}), ('k2', {'switched'}), ('k3', {'arm'}), ('k4', {'arm'}), ('k5', {'resolved', 'weight'}),
    ('k6', {'config'}), ('k7', {'config'}), ('k8', {'config', 'infer_img'}), ('k9', {'config', 'trunk'}),
    ('k10', {'config', 'trunk'}), ('k11', {'load_graphs'}), ('k12', {'config'}), ('k13', {'config'})])
def test_each_knob_patches_only_its_part(knob, parts):
    for value in S.VALUES[knob]:
        cfg = S.config({knob: value})
        assert touched(cfg) == parts
        assert set(K.describe(cfg)) == {knob}


def test_config_patch_changes_only_the_named_keys():
    names = dict(k6='lr', k7='weight_decay', k8='batch_size', k12='temperature', k13='patience')
    for knob, key in names.items():
        value = S.VALUES[knob][0]
        before = dict(ms_train.CONFIG)
        with ExitStack() as stack:
            K.patches(S.config({knob: value}), stack, scale='sample')
            assert {k for k in before if ms_train.CONFIG[k] != before[k]} == {key}
            assert ms_train.CONFIG[key] == value
    before = dict(ms_train.CONFIG)
    with ExitStack() as stack:
        K.patches(S.config({'k9': 512}), stack, scale='sample')
        assert {k for k in before if ms_train.CONFIG[k] != before[k]} == {'architecture'}
    with ExitStack() as stack:
        K.patches(S.config({'k10': 0.2}), stack, scale='sample')
        assert {k for k in before if ms_train.CONFIG[k] != before[k]} == {'dropout'}


def test_two_knobs_patch_the_union():
    cfg = S.config({'k2': 8, 'k6': 5e-4})
    assert touched(cfg) == {'switched', 'config'}
    assert set(K.describe(cfg)) == {'k2', 'k6'}
    cfg = S.config({'k3': 12, 'k4': 0.025})
    assert touched(cfg) == {'arm'} and set(K.describe(cfg)) == {'k3', 'k4'}


def test_patches_unwind_on_error():
    before = state()
    with pytest.raises(RuntimeError):
        with ExitStack() as stack:
            K.patches(S.config({'k6': 5e-4, 'k5': 0.5}), stack, scale='sample')
            raise RuntimeError
    assert state() == before
    # full scale (ticket 08): the same configurations patch; its tests are in test_rs_full.py
    with ExitStack() as stack:
        assert set(K.patches(S.config({'k6': 5e-4}), stack, scale='full')) == {'k6'}
    assert state() == before


# ------------------------------------------------------------------ the loss terms, on tiny tensors
def batch(n=6, seed=0):
    """A tiny graph batch: n pairs, 2n distinct paintings; random unit rows; the fit tensors are the 2n rows."""
    g = torch.Generator().manual_seed(seed)
    img, cap, teacher = (F.normalize(torch.randn(2 * n, 2048, generator=g), dim=1) for _ in range(3))
    pairs = torch.stack((torch.arange(n), torch.arange(n, 2 * n)), dim=1).numpy()
    weights = torch.rand(n, generator=g).numpy().astype('float32') * 0.5 + 0.5
    return img, cap, teacher, pairs, weights


def head_loss(cfg, aspect, *, eval_mode=True, seed=3, fit=True):
    """The loss rg_train's TrainingLoss route computes for one head under `cfg` (patched as the cohort is). `fit`
    False is a stopping call: include_mask_penalty False, under no_grad, as ms_train.run makes it."""
    img, cap, teacher, pairs, weights = batch()
    with ExitStack() as stack:
        K.patches(cfg, stack, scale='sample')
        recipe = rg_train.resolved(0)
        switches = V.switches('GS-0.02')
        genre = rg_loss.GenreLoss(switches, 0, term_log={})
        torch.manual_seed(seed)
        models, views = ms_train.build_model(recipe['base'], list(rg_train.ASPECTS), CPU)
        models.eval() if eval_mode else models.train()
        with torch.set_grad_enabled(fit):
            loss = rg_loss.switched_head_loss(views[aspect], img, cap, pairs, weights, recipe['base'], aspect,
                                              teacher, None, fit, switches=recipe['switches'], genre=genre)
        z = views[aspect]['img'](img[torch.as_tensor(pairs).T.reshape(-1)]) if eval_mode else None
    return float(loss.detach()), z, (img, cap, teacher, pairs)


def ids_of(pairs):
    return torch.as_tensor(pairs).T.reshape(-1)


def relational(z, teacher, pairs, tau_t):
    ids = ids_of(pairs)
    return float(rg_loss.relational_loss(z, teacher[ids], ids, tau_t).detach())


@pytest.mark.parametrize('fit', [True, False], ids=['fit call', 'stopping call'])
@pytest.mark.parametrize('knob,aspect,other', [('k1', 'emotion', 'style'), ('k2', 'style', 'emotion')])
def test_k1_k2_add_the_relational_term_to_their_own_head_only(knob, aspect, other, fit):
    # the stopping loss carries the term too: the loss that is trained is the loss that stops
    base, z, (_, _, teacher, pairs) = head_loss(S.config(), aspect, fit=fit)
    for weight in (2, 16):
        for tau_t in (0.02, 0.035):
            cfg = S.config({knob: weight, 'k4': tau_t} if tau_t != 0.02 else {knob: weight})
            loss, _, _ = head_loss(cfg, aspect, fit=fit)
            assert loss == pytest.approx(base + weight * relational(z, teacher, pairs, tau_t), rel=1e-5)
            assert head_loss(cfg, other, fit=fit)[0] == head_loss(S.config(), other, fit=fit)[0]
            assert head_loss(cfg, 'genre', fit=fit)[0] == head_loss(
                S.config({'k4': tau_t} if tau_t != 0.02 else None), 'genre', fit=fit)[0]


@pytest.mark.parametrize('knob,aspect', [('k1', 'emotion'), ('k2', 'style')])
def test_k1_k2_relational_term_carries_gradient_to_the_head(knob, aspect):
    # grad(k1 loss) = grad(G loss) + weight * grad(relational term on the head's own z): the term is attached
    img, cap, teacher, pairs, weights = batch()
    recipe = rg_train.resolved(0)
    ids = ids_of(pairs)
    weight, tau_t = 8, 0.025

    def grads(cfg, extra=False):
        with ExitStack() as stack:
            K.patches(cfg, stack, scale='sample')
            torch.manual_seed(3)
            models, views = ms_train.build_model(recipe['base'], list(rg_train.ASPECTS), CPU)
            models.eval()
            params = list(views[aspect]['img'].parameters())
            genre = rg_loss.GenreLoss(V.switches('GS-0.02'), 0, term_log={})
            loss = rg_loss.switched_head_loss(views[aspect], img, cap, pairs, weights, recipe['base'], aspect,
                                              teacher, None, True, switches=recipe['switches'], genre=genre)
            if extra:
                z = views[aspect]['img'](img[ids])
                loss = weight * rg_loss.relational_loss(z, teacher[ids], ids, tau_t)
            return torch.autograd.grad(loss, params)

    g_k, g_g = grads(S.config({knob: weight, 'k4': tau_t})), grads(S.config({'k4': tau_t}))
    g_rel = grads(S.config(), extra=True)
    assert sum(float(g.norm()) for g in g_rel) > 1e-4
    for a, b, c in zip(g_k, g_g, g_rel):
        assert torch.allclose(a, b + c, atol=1e-6, rtol=1e-4)


def test_k1_relational_term_is_on_the_frozen_forward_pass_not_a_second_one():
    # Training mode: dropout draws make a second forward differ; the term must see the frozen call's z.
    img, cap, teacher, pairs, weights = batch()
    recipe = rg_train.resolved(0)
    seen = []
    cfg = S.config({'k1': 2})
    with ExitStack() as stack:
        K.patches(cfg, stack, scale='sample')
        torch.manual_seed(1)
        models, views = ms_train.build_model(recipe['base'], list(rg_train.ASPECTS), CPU)
        models.train()
        views['emotion']['img'].register_forward_hook(lambda _m, _a, z: seen.append(z))
        genre = rg_loss.GenreLoss(V.switches('GS-0.02'), 0, term_log={})
        torch.manual_seed(9)
        loss = rg_loss.switched_head_loss(views['emotion'], img, cap, pairs, weights, recipe['base'], 'emotion',
                                          teacher, None, True, switches=recipe['switches'], genre=genre)
    assert len(seen) == 1 and loss.requires_grad
    loss.backward()
    assert views['emotion']['img'].trunk[0].weight.grad is not None


@pytest.mark.parametrize('fit', [True, False], ids=['fit call', 'stopping call'])
def test_k3_and_k4_set_the_genre_relational_weight_and_teacher_temperature(fit):
    base, z, (_, _, teacher, pairs) = head_loss(S.config(), 'genre', fit=fit)
    rel = relational(z, teacher, pairs, 0.02)
    for lam in (4, 12, 16):
        loss, _, _ = head_loss(S.config({'k3': lam}), 'genre', fit=fit)
        assert loss == pytest.approx(base + (lam - 8) * rel, rel=1e-5)
    for tau_t in (0.015, 0.035):
        loss, _, _ = head_loss(S.config({'k4': tau_t}), 'genre', fit=fit)
        assert loss == pytest.approx(base + 8 * (relational(z, teacher, pairs, tau_t) - rel), rel=1e-5)
    # the student row stays at 0.05 and the direct term is untouched: k3 = 8 and k4 = G's value change nothing
    assert head_loss(S.config(), 'emotion')[0] == head_loss(S.config({'k3': 12}), 'emotion')[0]


def test_k5_sets_the_direct_weight_on_all_heads_and_only_k5_relaxes_the_assert():
    img, cap, teacher, pairs, weights = batch()
    ids = ids_of(pairs)
    for aspect in rg_train.ASPECTS:
        base, z, _ = head_loss(S.config(), aspect)
        direct = float(ms_train.distill_loss(z, teacher[ids], ids, 'direct'))
        for weight in S.VALUES['k5']:
            loss, _, _ = head_loss(S.config({'k5': weight}), aspect)
            assert loss == pytest.approx(base + (weight - 0.25) * direct, rel=1e-5), (aspect, weight)
    # without k5 the frozen assert on 0.25 still guards the genre body
    recipe = rg_train.resolved(0)
    recipe['base']['distill_weight'] = 0.125
    models, views = ms_train.build_model(recipe['base'], list(rg_train.ASPECTS), CPU)
    with pytest.raises(AssertionError):
        rg_loss.switched_head_loss(views['genre'], img, cap, pairs, weights, recipe['base'], 'genre', teacher,
                                   None, True, switches=recipe['switches'],
                                   genre=rg_loss.GenreLoss(V.switches('GS-0.02'), 0))
    with ExitStack() as stack:
        K.patches(S.config({'k5': 0.125}), stack, scale='sample')
        assert rg_train.resolved(0)['base']['distill_weight'] == 0.125
        assert rg_train.resolved(0)['switches'] == recipe['switches']
    assert rg_train.resolved(0)['base']['distill_weight'] == 0.25


def test_k12_changes_the_four_buddy_terms_only():
    img, cap, teacher, pairs, weights = batch()
    x, y = img[:6], cap[:6]
    ids = ids_of(pairs)
    z = F.normalize(torch.randn(12, 2048, generator=torch.Generator().manual_seed(4)), dim=1)
    before = (float(ms_train.symmetric(x, y)), float(ms_train.distill_loss(z, teacher[ids], ids, 'direct')),
              float(ms_train.distill_loss(z, teacher[ids], ids, 'relational')),
              float(rg_loss.relational_loss(z, teacher[ids], ids, 0.02)))
    with ExitStack() as stack:
        K.patches(S.config({'k12': 0.1}), stack, scale='sample')
        buddy = ms_train.symmetric(x, y)
        expected = (F.cross_entropy(x @ y.T / 0.1, torch.arange(6)) + F.cross_entropy(y @ x.T / 0.1, torch.arange(6))) / 2
        assert float(buddy) == pytest.approx(float(expected), rel=1e-5) and float(buddy) != before[0]
        after = (float(ms_train.distill_loss(z, teacher[ids], ids, 'direct')),
                 float(ms_train.distill_loss(z, teacher[ids], ids, 'relational')),
                 float(rg_loss.relational_loss(z, teacher[ids], ids, 0.02)))
    assert after == before[1:]


def test_temperature_is_read_by_the_buddy_terms_only():
    """The frozen readers of CONFIG['temperature']; the direct and relational terms hard-code 0.05."""
    readers = {}
    for module in ('ms_train', 'ags_head', 'itp_train'):
        source = inspect.getsource(sys.modules[module])
        functions = re.split(r'\n(?=def |class )', source)
        readers[module] = sorted(re.match(r'(?:def|class) (\w+)', f).group(1) for f in functions
                                 if "CONFIG['temperature']" in f and re.match(r'(?:def|class) (\w+)', f))
    assert readers == {'ms_train': ['extra_symmetric', 'symmetric'], 'ags_head': ['symmetric'],
                       'itp_train': ['loss_fn']}
    for fn in (ms_train.distill_loss, rg_loss.relational_loss, rg_loss.anchors_loss):
        assert "CONFIG['temperature']" not in inspect.getsource(fn)
    assert ms_train.CONFIG is sys.modules['itp_train'].CONFIG is sys.modules['ags_head'].CONFIG


def test_config_keys_are_read_where_the_knobs_assume():
    run = inspect.getsource(ms_train.run)
    for key in ('lr', 'weight_decay', 'batch_size', 'patience'):
        assert f"CONFIG['{key}']" in run, key
    # one batch size trains and computes the stopping loss
    assert run.count("CONFIG['batch_size']") == 1 and 'range(0, len(st[\'dev_pairs\']), batch_size)' in run
    assert 'CONFIG' not in inspect.getsource(ms_train.save_output).replace("CONFIG['batch_size']", '')


def test_k8_pins_the_export_batch_but_not_training():
    class View:
        def __init__(self):
            self.sizes = []

        def eval(self):
            pass

        def __getitem__(self, key):
            return lambda x: (self.sizes.append(len(x)), x)[1]

    x = torch.ones(1300, 2048).numpy()
    for cfg, training_batch in ((S.config(), 512), (S.config({'k8': 2048}), 2048)):
        view = View()
        with ExitStack() as stack:
            K.patches(cfg, stack, scale='sample')
            assert ms_train.CONFIG['batch_size'] == training_batch
            eg.infer_img(view, x, CPU)
            assert ms_train.CONFIG['batch_size'] == training_batch      # restored after the pinned call
        assert view.sizes == [512, 512, 276]


@pytest.mark.parametrize('width,dropout', [(512, 0.1), (1536, 0.1), (2048, 0.1), (1024, 0.0), (1024, 0.3), (512, 0.2)])
def test_k9_k10_build_the_trunk(width, dropout):
    cfg = S.config({'k9': width, 'k10': dropout})
    recipe = rg_train.resolved(0)
    with ExitStack() as stack:
        K.patches(cfg, stack, scale='sample')
        models, views = ms_train.build_model(recipe['base'], list(rg_train.ASPECTS), CPU)
        for side in ('img', 'cap'):
            assert [tuple(p.shape) for p in models[side].trunk.parameters()] == [(width, 2048), (width,)]
            assert models[side].trunk[2].p == dropout
            assert all(tuple(h.weight.shape) == (2048, width) for h in models[side].heads.values())
        out = views['genre']['img'](torch.randn(3, 2048))
        assert out.shape == (3, 2048) and torch.allclose(out.norm(dim=1), torch.ones(3), atol=1e-5)


def test_trunk_at_g_values_equals_the_frozen_trunk_under_the_same_seed():
    recipe = rg_train.resolved(0)
    torch.manual_seed(5)
    frozen, _ = ms_train.build_model(recipe['base'], list(rg_train.ASPECTS), CPU)
    torch.manual_seed(5)
    ours = ms_train.TrunkSide  # replaced below by the layer's class at G's values
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ms_train, 'TrunkSide', K.trunk_side(1024, 0.1))
        mine, _ = ms_train.build_model(recipe['base'], list(rg_train.ASPECTS), CPU)
    assert ours is ms_train.TrunkSide
    assert list(mine.state_dict()) == list(frozen.state_dict())
    assert all(torch.equal(v, frozen.state_dict()[k]) for k, v in mine.state_dict().items())


# ------------------------------------------------------------------ k11
def test_k11_calls_rs_graphs_with_the_loaded_sample_and_k20_never_does(monkeypatch, tmp_path):
    seen = []

    class Fake:
        @staticmethod
        def sample_store(k, variant='aspect', *, s=None, outputs=None):
            seen.append((k, variant, s, outputs))
            return 'graphs', 'manifests'

    monkeypatch.setattr(K, 'rs_graphs', Fake)
    recipe = rg_train.resolved(0)
    with ExitStack() as stack:
        K.patches(S.config({'k11': 30}), stack, scale='sample', outputs=tmp_path)
        assert rg_train.load_graphs('the sample', recipe) == ('graphs', 'manifests')
    assert seen == [(30, 'aspect', 'the sample', tmp_path)]
    assert rg_train.load_graphs.__module__ == 'rg_train'          # restored: the frozen loader
    with pytest.raises(AssertionError, match='needs the outputs'):
        K.patches(S.config({'k11': 30}), ExitStack(), scale='sample')
    with pytest.raises(AssertionError, match='frozen store'):
        K.store_graphs('s', 20, tmp_path)
    assert len(seen) == 1
