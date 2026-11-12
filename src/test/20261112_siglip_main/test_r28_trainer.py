"""CPU checks against r10's actual patch/loss paths and r14's frozen ref path."""
import copy
import difflib
import inspect
from contextlib import ExitStack, contextmanager

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

import r28_trainer as T

B, K, S, G, V = T.B, T.K, T.S, T.G, T.V
RG = B.rg_full
R10_LOSS = RG.rg_loss.switched_head_loss
R10_GENRE = K.genre_loss
ROWS = (*T.KNOBS, *T.REMOVALS)


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    torch.set_num_threads(2)
    # Config/loss tests do not read graph artifacts. Separate tests below
    # check gate forwarding and selection of the graph store.
    monkeypatch.setattr(T, 'BASE_REQUIRE_GATE', lambda *a: None)


def expected_config(row):
    if row == 'ref':
        return S.config({'k11': 40})
    if row in T.REMOVALS:
        return S.config({'k11': 40}, remove=(row,))
    knob, value = row.split('=')
    return S.config({'k11': 40, knob: float(value)})


@contextmanager
def r10_context(row):
    """Independent oracle: the contexts entered by rs_full.train, without its run gates."""
    cfg = expected_config(row)
    with ExitStack() as stack:
        # Readiness of external artifacts is irrelevant to numeric unit tests.
        from unittest.mock import patch
        stack.enter_context(patch.object(G, 'require_gate', lambda *a: None))
        K.patches(cfg, stack, scale='full', outputs=T.OUTPUTS / 'unit-graphs')
        stack.enter_context(B.rs_full.arm_alias(B.row_arm(row)))
        yield cfg


def snapshot(row, width):
    recipe = B.rb_rows.recipe(RG.RECIPE)
    torch.manual_seed(19)
    models, _ = B.rb_student.build_model(recipe['base'], list(B.ASPECTS), 'cpu', width=width)
    return dict(recipe=recipe, config=copy.deepcopy(B.ms_train.CONFIG),
                switches=V.switches(B.row_arm(row)),
                state={k: v.detach().clone() for k, v in models.state_dict().items()})


@pytest.mark.parametrize('row', ROWS)
def test_siglip_config_switches_and_trunk_equal_r10_except_image_width(row):
    with r10_context(row):
        expected = snapshot(row, 2048)
    with T.training_context('siglip1', row) as ctx:
        actual = snapshot(row, 1152)
        assert ctx.cfg == expected_config(row)
        assert ctx.arm == row and ctx.backbone == 'siglip_so400m' and ctx.width == 1152
    for key in ('recipe', 'config', 'switches'):
        assert actual[key] == expected[key], key
    assert actual['state'].keys() == expected['state'].keys()
    for key, tensor in actual['state'].items():
        shape = (1024, 1152) if key == 'img.trunk.0.weight' else expected['state'][key].shape
        assert tensor.shape == shape, key
    assert actual['recipe']['base']['inputs'] == 'default'
    assert actual['recipe']['switches']['teacher'] == 'instructed'
    if row == 'relational':
        assert actual['switches']['lam'] == 0
    if row in K.GRAPH_REMOVALS:
        assert actual['recipe']['base']['graphs'] == G.row_names(K.GRAPH_REMOVALS[row])


class ImageSide(nn.Module):
    """Ignore zero padding used solely to satisfy r10's obsolete shape assert."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1152, 2048)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        return F.normalize(self.dropout(self.linear(x[:, :1152].contiguous())), dim=1)


class CaptionSide(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2048, 2048)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        return F.normalize(self.dropout(self.linear(x)), dim=1)


def batch():
    torch.manual_seed(71)
    models = nn.ModuleDict({'img': ImageSide(), 'cap': CaptionSide()})
    img = F.normalize(torch.randn(6, 1152), dim=1)
    cap = F.normalize(torch.randn(6, 2048), dim=1)
    teacher = F.normalize(torch.randn(6, 2048), dim=1)
    pairs = np.array([[0, 1], [2, 3], [4, 5]], dtype=np.int64)
    weights = np.array([1.0, 0.7, 0.3], dtype=np.float32)
    return models, img, cap, teacher, pairs, weights


def evaluate(data, aspect, row, *, padded=False, dev=False):
    models, img, cap, teacher, pairs, weights = data
    models.zero_grad(set_to_none=True)
    recipe = B.rb_rows.recipe(RG.RECIPE)
    log = {}
    genre = RG.rg_loss.GenreLoss(V.switches(B.row_arm(row)), term_log=log)
    # Direct and relational tensors, outputs, RNG draws, pairs and weights are
    # identical; only unused image columns differ to satisfy r10's assertion.
    torch.manual_seed(97)
    loss = RG.checked_head_loss(models, F.pad(img, (0, 896)) if padded else img, cap,
                                pairs, weights, recipe['base'], aspect, teacher,
                                include_mask_penalty=not dev, switches=recipe['switches'], genre=genre)
    loss.backward()
    gradients = {k: p.grad.detach().clone() if p.grad is not None else None
                 for k, p in models.named_parameters()}
    return loss.detach(), dict(log), gradients


@pytest.mark.parametrize('row', ROWS)
@pytest.mark.parametrize('aspect', B.ASPECTS)
@pytest.mark.parametrize('dev', (False, True))
def test_siglip_loss_terms_and_gradients_equal_r10(row, aspect, dev):
    data = batch()
    with r10_context(row):
        expected = evaluate(data, aspect, row, padded=True, dev=dev)
    with T.training_context('siglip1', row):
        actual = evaluate(data, aspect, row, dev=dev)
    assert torch.equal(actual[0], expected[0])
    assert actual[1].keys() == expected[1].keys()
    for key in actual[1]:
        assert torch.equal(actual[1][key], expected[1][key]), key
    for key, grad in actual[2].items():
        if grad is None:
            assert expected[2][key] is None
        else:
            assert torch.equal(grad, expected[2][key]), key


@pytest.mark.parametrize('aspect', B.ASPECTS)
def test_buddy_removal_changes_loss_and_stops_caption_gradients(aspect):
    data = batch()
    with T.training_context('siglip1', 'ref'):
        ref = evaluate(data, aspect, 'ref')
    with T.training_context('siglip1', 'buddy'):
        removed = evaluate(data, aspect, 'buddy')
    assert not torch.equal(ref[0], removed[0])
    assert any(g is not None for k, g in ref[2].items() if k.startswith('cap.'))
    assert all(g is None for k, g in removed[2].items() if k.startswith('cap.'))


def test_buddy_body_differs_from_r10_only_at_the_teacher_shape_assert():
    original = inspect.getsource(R10_GENRE).splitlines(keepends=True)
    adapted = inspect.getsource(T.L.genre_loss).splitlines(keepends=True)
    changes = [line for line in difflib.ndiff(original, adapted) if line.startswith(('- ', '+ '))]
    assert changes == [
        '-     assert teacher is not None and teacher.shape == img.shape and not teacher.requires_grad\n',
        '+     assert teacher is not None and teacher.shape == (len(img), rb_student.OUT) and not teacher.requires_grad\n',
    ]


@pytest.mark.parametrize('shape,requires_grad', (((6, 1152), False), ((7, 2048), False), ((6, 2048), True)))
def test_buddy_rejects_invalid_teacher_before_forward(shape, requires_grad):
    models, img, cap, _, pairs, weights = batch()
    with T.training_context('siglip1', 'buddy'):
        recipe = B.rb_rows.recipe(RG.RECIPE)
        genre = RG.rg_loss.GenreLoss(V.switches('buddy'))
        handle = models['img'].register_forward_pre_hook(lambda *a: pytest.fail('teacher was not checked'))
        try:
            with pytest.raises(AssertionError):
                RG.checked_head_loss(models, img, cap, pairs, weights, recipe['base'], 'genre',
                                     torch.zeros(shape, requires_grad=requires_grad),
                                     switches=recipe['switches'], genre=genre)
        finally:
            handle.remove()


@pytest.mark.parametrize('aspect', B.ASPECTS)
def test_siglip_ref_config_model_loss_are_exactly_r14(aspect):
    data = batch()
    with T.BASE_CONTEXT('siglip1', 'ref') as reference:
        expected_model = snapshot('ref', 1152)
        expected_loss = evaluate(data, aspect, 'ref')
        expected_fn = RG.rg_loss.switched_head_loss
    with T.training_context('siglip1', 'ref') as actual:
        actual_model = snapshot('ref', 1152)
        actual_loss = evaluate(data, aspect, 'ref')
        assert RG.rg_loss.switched_head_loss is expected_fn is B.bs_cstar_loss.switched_head_loss
        assert actual == reference
    assert actual_model['recipe'] == expected_model['recipe']
    assert actual_model['config'] == expected_model['config']
    for key, value in actual_model['state'].items():
        assert torch.equal(value, expected_model['state'][key]), key
    assert torch.equal(actual_loss[0], expected_loss[0])


@pytest.mark.parametrize('row', ('ref', *B.knob_rows(), *T.REMOVALS))
def test_qwen_switches_and_model_remain_r10(row):
    with r10_context(row):
        expected = snapshot(row, 2048)
    with T.training_context('qwen', row):
        actual = snapshot(row, 2048)
    for key in ('recipe', 'config', 'switches'):
        assert actual[key] == expected[key]
    for key, value in actual['state'].items():
        assert torch.equal(value, expected['state'][key])
    assert RG.rg_loss.switched_head_loss is R10_LOSS
    assert K.genre_loss is R10_GENRE


@pytest.mark.parametrize('row', ROWS)
def test_actual_trainer_receives_row_config_and_encoder(monkeypatch, row):
    observations = []
    records = []
    monkeypatch.setattr(B, 'code_state', lambda: ('test-commit', False))
    monkeypatch.setattr(RG, '_cohort_state', lambda *a: None)
    monkeypatch.setattr(B.eg, 'atomic_record', lambda path, record: records.append((path, record)))

    def load_data(ctx):
        assert ctx.cfg == expected_config(row)
        s = dict(fit=np.array([0, 1]), dev=np.array([2]), all_painting_ids=np.array(['p9', 'p2', 'p8']))
        return s, {}, {}, {}, np.zeros((3, ctx.width), dtype=np.float32)

    def cohort(s, inputs, graphs, manifests, images, arm, seed, device, root, **kw):
        observations.append((RG.ENCODER, images.shape, arm, snapshot(row, images.shape[1])))
        return {'status': 'ok'}

    def record(enc, row_, ctx, seed, device, keep, smoke, max_epochs, root, seconds, state):
        return dict(config=ctx.cfg, row=row_, name=ctx.arm, seed=seed, status='ok', seconds=seconds)

    monkeypatch.setattr(B, 'load_data', load_data)
    monkeypatch.setattr(RG, 'train_cohort', cohort)
    monkeypatch.setattr(T, 'BASE_RECORD', record)
    root = T.SMOKE / 'unit-no-write' / row
    runs = T.train('siglip1', row, [0], 'cpu', root, smoke=True)
    assert len(observations) == len(records) == 1
    assert observations[0][:3] == ('siglip_so400m', (3, 1152), row)
    assert runs[0].record['config'] == expected_config(row)
    assert records[0][0] == root / 'train' / row / 'run__s0.json'
    assert runs[0].record['r28_code_sha256'] == T.code_hashes()


@pytest.mark.parametrize('row', T.REMOVALS)
def test_removal_export_context_keeps_graph_rows_and_switches(row):
    with T.training_context('siglip1', row) as trained:
        expected = (trained.cfg, V.switches(row), B.rb_rows.recipe(RG.RECIPE))
    with T.training_context('siglip1', row, export=True) as exported:
        assert (exported.cfg, V.switches(row), B.rb_rows.recipe(RG.RECIPE)) == expected


@pytest.mark.parametrize('variant', ('aspect', 'uncorrected', 'default_view'))
def test_graph_seam_keeps_qwen_variant_and_never_writes_r10(monkeypatch, variant):
    seen = []
    monkeypatch.setattr(B, 'require_graph_gate', lambda: seen.append('k11_gate'))
    monkeypatch.setattr(T, 'BASE_REQUIRE_GATE', lambda root, name, item: seen.append((root, name, item)))
    monkeypatch.setattr(G, '_materialize', lambda scale, k, v, root: (None, (scale, k, v, root)))
    result = T.graph_data('full', 40, T.SMOKE / 'unit-graphs', variant)
    source = T.graph_source(40, variant, T.SMOKE / 'unit-graphs')
    assert result == ('full', 40, variant, source)
    assert seen[0] == 'k11_gate'
    if variant != 'aspect':
        assert seen[1] == (T.P.R10_OUT, 'ablation_graphs', variant)


@pytest.mark.parametrize('present', ('r10', 'r14', 'neither'))
def test_graph_cache_reuse_and_r28_build_destination(monkeypatch, present):
    from pathlib import Path
    verified = []
    root = T.SMOKE / 'unit-graphs'
    cache = {'r10': T.P.R10_OUT, 'r14': T.P.DEFAULT_OUT}.get(present)
    monkeypatch.setattr(Path, 'exists', lambda p: cache is not None and p.is_relative_to(cache))
    monkeypatch.setattr(G, '_require_own_store', lambda *a: verified.append(a))
    assert T.graph_source(80, 'aspect', root) == (cache if cache is not None else root)
    assert len(verified) == (0 if cache is None else 1)
    if verified:
        assert verified[0] == (G.store_root(cache, 'full', 80, 'aspect'), 'full', 80, 'aspect')


def test_patch_unwind_and_output_guards():
    with pytest.raises(RuntimeError, match='unwind'):
        with T.training_context('siglip1', 'buddy'):
            raise RuntimeError('unwind')
    assert RG.rg_loss.switched_head_loss is R10_LOSS and K.genre_loss is R10_GENRE
    assert RG.ENCODER == 'qwen3vl_emb_2b'
    assert T.check_root(T.SMOKE / 'unit', True) == T.SMOKE / 'unit'
    for root, smoke in ((T.P.DEFAULT_OUT, False), (T.OUTPUTS / 'trainer', True), (T.SMOKE / 'unit', False)):
        with pytest.raises(AssertionError):
            T.check_root(root, smoke)
    with pytest.raises(AssertionError, match='do not retrain'):
        T.train('siglip1', 'ref', [0], 'cpu', T.OUTPUTS / 'trainer')
