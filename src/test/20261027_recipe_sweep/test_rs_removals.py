"""The eight removal switches (ticket 07, rule §4.5): each removes only its part, at both scales. On tiny CPU tensors
(the loss terms present on each head, the gradient that reaches each network), on the patched objects (what is touched,
which graphs are read) and on the built models (the trunk count). Every other knob and every other part stay as in F.
The training itself is the sample and full-scale trainers' (their `--smoke` runs); G stays patch-free (test_rs_knobs)."""
import json
import sys
from contextlib import ExitStack

import pytest
import torch
from torch.nn import functional as Fn

import rs_graphs
import rs_knobs as K
import rs_sheet as S
import ab_train
import rb_rows
import rb_student
import rg_full
import rg_loss
import rg_train
import rg_variants as V
from ab_paths import ms_train

CPU = torch.device('cpu')
ASPECTS = list(rg_train.ASPECTS)
SCALES = ('sample', 'full')
TAU = V.TAU


def removal(name, **changes):
    return S.config(changes, remove=[name])


def patched(cfg, scale, outputs=None):
    """The description `K.patches` returns; the patches it enters are undone on return."""
    with ExitStack() as stack:
        return K.patches(cfg, stack, scale=scale, outputs=outputs)


# ------------------------------------------------------------------------------------ what each removal touches
def state():
    """Every object the removals may patch, by value or identity (the knob objects of both scales included)."""
    return dict(config=dict(ms_train.CONFIG), arm=dict(V.ARMS['GS-0.02']), weight=V.TEACHER_WEIGHT,
                switched=ab_train.switched_head_loss, resolved=rg_train.resolved, load_graphs=rg_train.load_graphs,
                recipe=rb_rows.recipe, load_scale=rb_student.load_scale, infer_img=rb_student.infer,
                head_loss=rg_loss.switched_head_loss, trunk=ms_train.TrunkSide, trunk_head=ms_train.TrunkHead)


def touched(cfg, scale):
    before = state()
    with ExitStack() as stack:
        K.patches(cfg, stack, scale=scale, outputs='o')
        inside = state()
    assert state() == before
    return {name for name in before if inside[name] != before[name]}


@pytest.fixture
def gate(tmp_path, monkeypatch):
    """A passing ablation_graphs gate for the graph code on disk, in tmp_path (the outputs of these tests); the stores
    are not built: the loaders are recorders."""
    def write(**items):
        (tmp_path / 'gates').mkdir(exist_ok=True)
        passing = {'default_view': dict(**{'pass': True}), 'uncorrected': dict(**{'pass': True})}
        for name, ok in items.items():
            passing[name] = {'pass': ok}
        (tmp_path / 'gates/ablation_graphs.json').write_text(json.dumps(
            {'pass': all(i['pass'] for i in passing.values()), 'items': passing,
             'code_sha256': rs_graphs.code_sha256()}))
    seen = []
    monkeypatch.setattr(rs_graphs, 'sample_store',
                        lambda k, variant='aspect', *, s=None, outputs=None: (seen.append(('sample', k, variant, s, outputs)),
                                                                              ('graphs', 'manifests'))[1])
    monkeypatch.setattr(rs_graphs, 'full_training_data',
                        lambda k, variant='aspect', *, outputs=None: (seen.append(('full', k, variant, outputs)),
                                                                      ('s', 'inputs', 'graphs', 'manifests'))[1])
    write()
    return tmp_path, seen, write


EXPECTED = {   # removal -> the objects it patches, per scale
    'relational': {s: {'arm'} for s in SCALES},
    'sharp_row': {s: {'arm'} for s in SCALES},
    'direct': {s: {'arm', 'head_loss'} for s in SCALES},
    'buddy': {s: {'head_loss'} for s in SCALES},
    'caption_endpoints': {s: {'head_loss'} for s in SCALES},
    'content_correction': dict(sample={'load_graphs', 'resolved'}, full={'load_scale', 'recipe'}),
    'instructed_graphs': dict(sample={'load_graphs', 'resolved'}, full={'load_scale', 'recipe'}),
    'shared_trunk': {s: {'trunk', 'trunk_head'} for s in SCALES},
}


@pytest.mark.parametrize('scale', SCALES)
@pytest.mark.parametrize('name', S.REMOVALS)
def test_each_removal_patches_only_its_part(name, scale, gate):
    outputs, _, _ = gate
    cfg = removal(name)
    before = state()
    with ExitStack() as stack:
        description = K.patches(cfg, stack, scale=scale, outputs=outputs)
        inside = state()
    assert {n for n in before if inside[n] != before[n]} == EXPECTED[name][scale]
    assert set(description) == {f'no_{name}'} == set(K.describe(cfg, scale))
    assert state() == before and ms_train.CONFIG == before['config']      # the optimiser's and the loop's knobs: G's


def test_the_removals_are_the_sheets_eight_and_all_are_tested():
    assert set(EXPECTED) == set(S.REMOVALS) and len(S.REMOVALS) == 8


def test_the_arm_diffs_are_one_switch_each():
    arm = dict(V.ARMS['GS-0.02'])
    for name, change in (('relational', dict(lam=0)), ('sharp_row', dict(tau_t=TAU)), ('direct', dict(m=0))):
        with ExitStack() as stack:
            K.patches(removal(name), stack, scale='sample')
            assert V.ARMS['GS-0.02'] == dict(arm, **change)
    assert V.ARMS['GS-0.02'] == arm


# ------------------------------------------------------------------------------------ the loss, on tiny tensors
def batch(n=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    img, cap, teacher = (Fn.normalize(torch.randn(2 * n, 2048, generator=g), dim=1) for _ in range(3))
    pairs = torch.stack((torch.arange(n), torch.arange(n, 2 * n)), dim=1).numpy()
    weights = torch.rand(n, generator=g).numpy().astype('float32') * 0.5 + 0.5
    return img, cap, teacher, pairs, weights


def route(cfg, scale, aspect, *, fit=True, train=False, seed=3, before=None):
    """(loss, models, views, tensors): the loss one head gets under `cfg` as the scale's trainer routes it. The model is
    built under G's patches only (the removals do not touch the build), so every cfg starts from the same weights."""
    img, cap, teacher, pairs, weights = batch()
    with ExitStack() as stack:
        K.patches(cfg, stack, scale=scale, outputs='o')
        row = rg_train.resolved(0)['base'] if scale == 'sample' else rb_rows.recipe(V.BASE)['base']
        torch.manual_seed(seed)
        models, views = ms_train.build_model(row, ASPECTS, CPU)
        models.train() if train else models.eval()
        if scale == 'sample':
            switches = rg_train.resolved(0)['switches']
            genre = rg_loss.GenreLoss(V.switches('GS-0.02'), 0, term_log={})
            call = lambda: rg_loss.switched_head_loss(views[aspect], img, cap, pairs, weights, row, aspect, teacher,
                                                      None, fit, switches=switches, genre=genre)
        else:
            head_loss = rg_full.head_loss_for('GS-0.02', 0, fit_teacher=None, n_dev=1)
            call = lambda: head_loss(views[aspect], img, cap, pairs, weights, row, aspect, teacher, None, fit)
        if before is not None:
            before(views)
        torch.manual_seed(9)
        with torch.set_grad_enabled(fit):
            loss = call()
    return loss, models, views, (img, cap, teacher, pairs, weights)


def terms(aspect, tensors):
    """The pieces of the loss of one head of the seed-3 model, computed apart from the trainer's code."""
    img, cap, teacher, pairs, weights = tensors
    torch.manual_seed(3)
    row = rg_train.resolved(0)['base']
    _, views = ms_train.build_model(row, ASPECTS, CPU)
    for side in views[aspect].values():
        side.eval()
    ids = torch.as_tensor(pairs).T.reshape(-1)
    z = views[aspect]['img'](img[ids])
    ia, ib = z.chunk(2)
    ca, cb = views[aspect]['cap'](cap[ids]).chunk(2)
    w = torch.as_tensor(weights)
    sym = ms_train.symmetric
    return dict(
        buddy=float((sym(ia, ib, w) + sym(ia, cb, w) + sym(ca, ib, w) + sym(ca, cb, w)) / 4),
        image_image=float(sym(ia, ib, w)),
        direct=float(ms_train.distill_loss(z, teacher[ids], ids, 'direct')),
        relational=lambda tau: float(rg_loss.relational_loss(z, teacher[ids], ids, tau)))


def g_loss(aspect, t, k1=0, k2=0):
    """G's loss on a head, from the pieces: four buddy terms + 0.25 direct (+ the relational term where there is one)."""
    value = t['buddy'] + 0.25 * t['direct']
    if aspect == 'genre':
        value += 8 * t['relational'](0.02)
    weight = dict(emotion=k1, style=k2).get(aspect, 0)
    return value + weight * t['relational'](0.02) if weight else value


@pytest.mark.parametrize('fit', [True, False], ids=['fit call', 'stopping call'])
@pytest.mark.parametrize('scale', SCALES)
def test_g_decomposes_into_the_pieces(scale, fit):
    # the reference for the removal tests below: the trainer's G loss equals the pieces, on every head
    for aspect in ASPECTS:
        loss, _, _, tensors = route(S.config(), scale, aspect, fit=fit)
        assert float(loss) == pytest.approx(g_loss(aspect, terms(aspect, tensors)), rel=1e-5), aspect


@pytest.mark.parametrize('fit', [True, False], ids=['fit call', 'stopping call'])
@pytest.mark.parametrize('scale', SCALES)
def test_relational_removes_the_relational_term_of_every_head_that_has_one(scale, fit):
    cfg = removal('relational', k1=8, k2=2)           # F with relational terms on emotion and style as well
    for aspect in ASPECTS:
        loss, _, _, tensors = route(cfg, scale, aspect, fit=fit)
        t = terms(aspect, tensors)
        assert float(loss) == pytest.approx(t['buddy'] + 0.25 * t['direct'], rel=1e-5), aspect
    # on G itself only the genre head has one; the other two are G's loss exactly
    for aspect in ('emotion', 'style'):
        assert float(route(removal('relational'), scale, aspect, fit=fit)[0]) == \
            float(route(S.config(), scale, aspect, fit=fit)[0])


@pytest.mark.parametrize('fit', [True, False], ids=['fit call', 'stopping call'])
@pytest.mark.parametrize('scale', SCALES)
def test_sharp_row_moves_every_relational_teacher_row_to_the_students_temperature(scale, fit):
    cfg = removal('sharp_row', k1=8, k2=2, k4=0.035)
    for aspect, weight in (('emotion', 8), ('style', 2), ('genre', 8)):
        loss, _, _, tensors = route(cfg, scale, aspect, fit=fit)
        t = terms(aspect, tensors)
        assert float(loss) == pytest.approx(t['buddy'] + 0.25 * t['direct'] + weight * t['relational'](TAU), rel=1e-5)
    # G: genre's row goes from 0.02 to 0.05 (= the frozen GR-8 arm); the heads without a term are G's exactly
    loss, _, _, tensors = route(removal('sharp_row'), scale, 'genre', fit=fit)
    t = terms('genre', tensors)
    assert float(loss) == pytest.approx(t['buddy'] + 0.25 * t['direct'] + 8 * t['relational'](TAU), rel=1e-5)
    for aspect in ('emotion', 'style'):
        assert float(route(removal('sharp_row'), scale, aspect, fit=fit)[0]) == \
            float(route(S.config(), scale, aspect, fit=fit)[0])


@pytest.mark.parametrize('fit', [True, False], ids=['fit call', 'stopping call'])
@pytest.mark.parametrize('scale', SCALES)
def test_direct_removes_the_teacher_term_on_all_three_heads_and_keeps_the_relational_term(scale, fit):
    cfg = removal('direct', k1=8)
    for aspect in ASPECTS:
        loss, _, _, tensors = route(cfg, scale, aspect, fit=fit)
        t = terms(aspect, tensors)
        expected = t['buddy'] + (8 * t['relational'](0.02) if aspect in ('emotion', 'genre') else 0)
        assert float(loss) == pytest.approx(expected, rel=1e-5), aspect


@pytest.mark.parametrize('scale', SCALES)
def test_direct_drops_the_term_and_leaves_the_recipe_row_alone(scale):
    # the row keeps its distill and weight (the frozen genre body asserts on them, and the teacher rows are still
    # loaded for the relational terms); the term is absent, not multiplied by 0
    frozen = rg_train.resolved(0) if scale == 'sample' else rb_rows.recipe(V.BASE)
    img, cap, teacher, pairs, weights = batch()
    with ExitStack() as stack:
        K.patches(removal('direct'), stack, scale=scale)
        recipe = rg_train.resolved(0) if scale == 'sample' else rb_rows.recipe(V.BASE)
        assert recipe == frozen and recipe['base']['distill'] == 'direct' and recipe['base']['distill_weight'] == 0.25
        genre = rg_loss.GenreLoss(V.switches('GS-0.02'), 0, term_log={})
        _, views = ms_train.build_model(recipe['base'], ASPECTS, CPU)
        rg_loss.switched_head_loss(views['genre'], img, cap, pairs, weights, recipe['base'], 'genre', teacher, None,
                                   True, switches=recipe['switches'], genre=genre)
        assert set(genre.term_log) == {'relational'}
        # absent, not weighted 0: a non-finite teacher row (0 * inf = nan) cannot reach the loss of the other two heads
        bad = teacher.clone()
        bad[0] = float('inf')
        for aspect in ('emotion', 'style'):
            loss = rg_loss.switched_head_loss(views[aspect], img, cap, pairs, weights, recipe['base'], aspect, bad, None,
                                              True, switches=recipe['switches'], genre=genre)
            assert torch.isfinite(loss), aspect
    loss, _, views, _ = route(removal('direct'), scale, 'emotion', train=True)
    loss.backward()
    assert all(p.grad is not None for p in views['emotion']['img'].parameters())


@pytest.mark.parametrize('fit', [True, False], ids=['fit call', 'stopping call'])
@pytest.mark.parametrize('scale', SCALES)
def test_buddy_removes_the_four_buddy_terms_on_all_three_heads(scale, fit):
    cfg = removal('buddy', k1=8)
    for aspect in ASPECTS:
        loss, _, _, tensors = route(cfg, scale, aspect, fit=fit)
        t = terms(aspect, tensors)
        expected = 0.25 * t['direct'] + (8 * t['relational'](0.02) if aspect in ('emotion', 'genre') else 0)
        assert float(loss) == pytest.approx(expected, rel=1e-5), aspect


@pytest.mark.parametrize('fit', [True, False], ids=['fit call', 'stopping call'])
@pytest.mark.parametrize('scale', SCALES)
def test_caption_endpoints_keeps_only_the_image_image_buddy_term_unaveraged(scale, fit):
    # D2's image-only endpoints (R7): the one term, as it is (not divided by four); the other terms stay
    cfg = removal('caption_endpoints', k1=8)
    for aspect in ASPECTS:
        loss, _, _, tensors = route(cfg, scale, aspect, fit=fit)
        t = terms(aspect, tensors)
        expected = t['image_image'] + 0.25 * t['direct'] + (8 * t['relational'](0.02) if aspect in ('emotion', 'genre') else 0)
        assert float(loss) == pytest.approx(expected, rel=1e-5), aspect


@pytest.mark.parametrize('name,cap_grad', [(None, True), ('buddy', False), ('caption_endpoints', False)])
@pytest.mark.parametrize('aspect', ASPECTS)
def test_the_caption_network_gets_gradient_only_through_the_buddy_terms(name, cap_grad, aspect):
    cfg = removal(name) if name else S.config()
    loss, models, views, _ = route(cfg, 'sample', aspect, train=True)
    loss.backward()
    cap = list(views[aspect]['cap'].parameters())
    assert all((p.grad is not None) == cap_grad for p in cap)
    assert all(p.grad is not None for p in views[aspect]['img'].parameters())     # the image side always trains


@pytest.mark.parametrize('scale', SCALES)
@pytest.mark.parametrize('name,calls', [(None, 1), ('buddy', 1), ('caption_endpoints', 0)])
@pytest.mark.parametrize('aspect', ASPECTS)
def test_the_caption_network_still_runs_when_only_the_buddy_terms_are_off(name, calls, aspect, scale):
    # as in `ab_train`: buddy off keeps the caption forward pass (the same dropout draws on every head, so the image
    # side of the three heads is unchanged by the switch); image-only endpoints skip it
    seen = []

    def hook(views):
        views[aspect]['cap'].register_forward_hook(lambda *args: seen.append(1))

    route(removal(name) if name else S.config(), scale, aspect, before=hook)
    assert len(seen) == calls, (name, aspect, scale)


@pytest.mark.parametrize('scale', SCALES)
def test_the_genre_copy_equals_the_frozen_genre_body_at_the_frozen_switches(scale):
    # `genre_loss` is a copy of rg_loss's genre body with ab_train's two buddy switches; with them at R6-5's values it
    # must be the frozen body, bit for bit, in training mode (same dropout draws)
    img, cap, teacher, pairs, weights = batch()
    row = rg_train.resolved(0)['base']
    switches = rg_train.resolved(0)['switches']
    for fit in (True, False):
        values = []
        for compute in ('frozen', 'copy'):
            torch.manual_seed(3)
            models, views = ms_train.build_model(row, ASPECTS, CPU)
            models.train()
            genre = rg_loss.GenreLoss(V.switches('GS-0.02'), 0, term_log={})
            torch.manual_seed(9)
            with torch.set_grad_enabled(fit):
                if compute == 'frozen':
                    loss = rg_loss.switched_head_loss(views['genre'], img, cap, pairs, weights, row, 'genre', teacher,
                                                      None, fit, switches=switches, genre=genre)
                else:
                    loss = K.genre_loss(views['genre'], img, cap, pairs, weights, row, teacher, switches, genre)
            values.append((float(loss), {k: float(v) for k, v in genre.term_log.items()}))
        assert values[0] == values[1] and set(values[0][1]) == {'direct', 'relational'}


def test_the_genre_copy_logs_the_terms_it_has():
    img, cap, teacher, pairs, weights = batch()
    row = rg_train.resolved(0)['base']
    switches = dict(rg_train.resolved(0)['switches'], buddy=0.0)
    torch.manual_seed(3)
    _, views = ms_train.build_model(row, ASPECTS, CPU)
    genre = rg_loss.GenreLoss(V.switches('GS-0.02'), 0, term_log={'stale': 1})
    K.genre_loss(views['genre'], img, cap, pairs, weights, row, teacher, switches, genre)
    assert set(genre.term_log) == {'direct', 'relational'}
    genre = rg_loss.GenreLoss(V.switches('GS-0.02'), 0, term_log={})
    genre.m = genre.lam = 0
    with pytest.raises(AssertionError, match='left no loss term'):
        K.genre_loss(views['genre'], img, cap, pairs, weights, row, teacher, switches, genre)


@pytest.mark.parametrize('name', ['buddy', 'caption_endpoints', 'direct'])
def test_a_removal_changes_no_loss_the_sample_and_full_routes_do_not_share(name):
    # both trainers reach the wrapped function through `rg_loss.switched_head_loss`: the same numbers at both scales
    for aspect in ASPECTS:
        assert float(route(removal(name), 'sample', aspect)[0]) == float(route(removal(name), 'full', aspect)[0])


# ------------------------------------------------------------------------------------ the graphs
GRAPH = {'content_correction': ('uncorrected', dict(emotion='R0', style='R0', genre='R0o')),
         'instructed_graphs': ('default_view', dict(emotion='Dview', style='Dview', genre='Dview'))}


@pytest.mark.parametrize('name', GRAPH)
@pytest.mark.parametrize('k', [20, 30])
def test_graph_removals_read_their_variant_at_the_runs_k_at_the_sample_scale(name, k, gate):
    outputs, seen, _ = gate
    variant, rows = GRAPH[name]
    frozen = rg_train.resolved(0)
    cfg = removal(name, k11=k)
    with ExitStack() as stack:
        description = K.patches(cfg, stack, scale='sample', outputs=outputs)
        assert rg_train.load_graphs('the sample', frozen) == ('graphs', 'manifests')
        recipe = rg_train.resolved(0)
    assert seen == [('sample', k, variant, 'the sample', outputs)]
    assert recipe['base']['graphs'] == rows
    # nothing else of the recipe row changes
    assert recipe['switches'] == frozen['switches'] and recipe['store'] == frozen['store']
    assert dict(recipe['base'], graphs=None) == dict(frozen['base'], graphs=None)
    assert set(description) == {f'no_{name}'} | ({'k11'} if k != 20 else set())
    assert rg_train.load_graphs.__module__ == 'rg_train' and rg_train.resolved(0)['base']['graphs'] != rows


@pytest.mark.parametrize('name', GRAPH)
@pytest.mark.parametrize('k', [20, 30])
def test_graph_removals_read_their_variant_at_the_runs_k_at_full_scale(name, k, gate):
    outputs, seen, _ = gate
    variant, rows = GRAPH[name]
    frozen = rb_rows.recipe(V.BASE)
    frozen_other = rb_rows.recipe('R0-5')
    cfg = removal(name, k11=k)
    with ExitStack() as stack:
        K.patches(cfg, stack, scale='full', outputs=outputs)
        assert rb_student.load_scale('full') == ('s', 'inputs', 'graphs', 'manifests')
        recipe = rb_rows.recipe(V.BASE)
        assert rb_rows.recipe('R0-5') == frozen_other            # held inference builds from R0-5's row: untouched
    assert seen == [('full', k, variant, outputs)]
    assert recipe['base']['graphs'] == rows
    assert recipe['switches'] == frozen['switches'] and dict(recipe['base'], graphs=None) == dict(frozen['base'], graphs=None)
    assert rb_rows.recipe(V.BASE) == frozen


def test_graph_removal_with_k11_has_one_loader_not_two(gate):
    outputs, seen, _ = gate
    cfg = removal('content_correction', k11=30)
    with ExitStack() as stack:
        description = K.patches(cfg, stack, scale='sample', outputs=outputs)
        rg_train.load_graphs('s', rg_train.resolved(0))
    assert len(seen) == 1 and set(description) == {'k11', 'no_content_correction'}


@pytest.mark.parametrize('scale', SCALES)
@pytest.mark.parametrize('name', GRAPH)
def test_graph_removals_refuse_without_a_passing_ablation_graphs_gate(name, scale, gate):
    outputs, seen, write = gate
    variant = GRAPH[name][0]
    cfg = removal(name)
    (outputs / 'gates/ablation_graphs.json').unlink()
    with pytest.raises(rs_graphs.GraphRefusal, match='ablation_graphs.json is missing'):
        patched(cfg, scale=scale, outputs=outputs)
    write(**{variant: False})
    with pytest.raises(rs_graphs.GraphRefusal, match='failed'):
        patched(cfg, scale=scale, outputs=outputs)
    # a gate written by other graph code does not count
    path = outputs / 'gates/ablation_graphs.json'
    path.write_text(json.dumps(dict(json.loads(path.read_text()), code_sha256='0' * 64, **{'pass': True})))
    with pytest.raises(rs_graphs.GraphRefusal, match='other graph code'):
        patched(cfg, scale=scale, outputs=outputs)
    assert not seen


def test_each_graph_removal_needs_its_own_item_of_the_gate(gate):
    outputs, _, write = gate
    write(uncorrected=False)
    patched(removal('instructed_graphs'), scale='sample', outputs=outputs)
    with pytest.raises(rs_graphs.GraphRefusal):
        patched(removal('content_correction'), scale='sample', outputs=outputs)
    write(default_view=False)
    patched(removal('content_correction'), scale='full', outputs=outputs)
    with pytest.raises(rs_graphs.GraphRefusal):
        patched(removal('instructed_graphs'), scale='full', outputs=outputs)


def test_the_non_graph_removals_do_not_read_the_gate_and_need_no_outputs(tmp_path):
    for name in ('relational', 'sharp_row', 'direct', 'buddy', 'caption_endpoints', 'shared_trunk'):
        patched(removal(name), scale='sample')                 # no outputs, no gates/ folder
        patched(removal(name), scale='full', outputs=tmp_path)
    for name in GRAPH:
        with pytest.raises(AssertionError, match='needs the outputs'):
            patched(removal(name), scale='sample')


# ------------------------------------------------------------------------------------ the trunk
@pytest.mark.parametrize('scale', SCALES)
def test_shared_trunk_gives_each_aspect_its_own_trunk_on_both_sides(scale):
    row = rg_train.resolved(0)['base'] if scale == 'sample' else rb_rows.recipe(V.BASE)['base']

    def build(cfg):
        with ExitStack() as stack:
            K.patches(cfg, stack, scale=scale)
            torch.manual_seed(5)
            if scale == 'full':
                return rb_student.build_model(row, ASPECTS, CPU, width=2048)         # the full path's factory
            return ms_train.build_model(row, ASPECTS, CPU)

    shared_models, shared = build(S.config())
    own_models, own = build(removal('shared_trunk'))
    for side in ('img', 'cap'):
        assert len({id(shared[a][side].trunk) for a in ASPECTS}) == 1       # G: one trunk
        assert len({id(own[a][side].trunk) for a in ASPECTS}) == 3          # removal: one per aspect
        assert len({id(own[a][side].output) for a in ASPECTS}) == 3         # heads: one per aspect either way
        assert len({id(p) for a in ASPECTS for p in own[a][side].trunk.parameters()}) == 6   # (weight, bias) x 3
        assert len({id(p) for a in ASPECTS for p in shared[a][side].trunk.parameters()}) == 2
    # each view keeps G's state keys and shapes, so the saved best states and the held inference are unchanged in form
    for a in ASPECTS:
        mine, frozen = own[a].state_dict(), shared[a].state_dict()
        assert list(mine) == list(frozen) and all(mine[k].shape == frozen[k].shape for k in mine)
    # the cohort's parameters: the optimiser sees every trunk once
    assert len(list(own_models.parameters())) == len(list(shared_models.parameters())) + 2 * 2 * 2   # 2 more trunks x 2 params x 2 sides
    # the trunks start different (independent initialisation), not copies of one
    assert not torch.equal(own['emotion']['img'].trunk[0].weight, own['style']['img'].trunk[0].weight)
    out = own['genre']['img'](torch.randn(3, 2048))
    assert out.shape == (3, 2048) and torch.allclose(out.norm(dim=1), torch.ones(3), atol=1e-5)


@pytest.mark.parametrize('scale', SCALES)
def test_shared_trunk_creates_the_three_trunks_before_the_three_heads(scale):
    # the order of creation decides the initial draws; the docstring of `separate_trunks` names it
    row = rg_train.resolved(0)['base'] if scale == 'sample' else rb_rows.recipe(V.BASE)['base']
    with ExitStack() as stack:
        K.patches(removal('shared_trunk'), stack, scale=scale)
        models, _ = (rb_student.build_model(row, ASPECTS, CPU, width=2048) if scale == 'full' else
                     ms_train.build_model(row, ASPECTS, CPU))
    for side in ('img', 'cap'):
        names = [n for n, _ in models[side].named_parameters()]
        kinds = [n.split('.')[0] for n in names]
        assert kinds == ['trunks'] * 6 + ['heads'] * 6, names


def test_shared_trunk_keeps_the_trunks_width_and_dropout_of_the_run():
    for width, dropout in ((1024, 0.1), (512, 0.0), (2048, 0.3)):
        cfg = S.config({'k9': width, 'k10': dropout}, remove=['shared_trunk'])
        with ExitStack() as stack:
            K.patches(cfg, stack, scale='sample')
            models, views = ms_train.build_model(rg_train.resolved(0)['base'], ASPECTS, CPU)
        for a in ASPECTS:
            for side in ('img', 'cap'):
                trunk = views[a][side].trunk
                assert [tuple(p.shape) for p in trunk.parameters()] == [(width, 2048), (width,)]
                assert trunk[2].p == dropout and tuple(views[a][side].output.weight.shape) == (2048, width)


def test_a_head_trains_only_its_own_trunk_when_the_trunk_is_not_shared():
    # the point of the removal: the loss of one aspect moves no parameter of another aspect's trunk (shared: it does)
    def grads(cfg):
        loss, _, views, _ = route(cfg, 'sample', 'emotion', train=True)
        loss.backward()
        return {a: all(p.grad is None for p in views[a]['img'].trunk.parameters()) for a in ('style', 'genre')}

    assert grads(S.config()) == dict(style=False, genre=False)      # shared: the emotion loss reaches the common trunk
    assert grads(removal('shared_trunk')) == dict(style=True, genre=True)


def test_shared_trunk_changes_nothing_else_of_the_model_or_the_loss():
    # the loss terms and the optimiser's settings are G's: the same head loss on the same pieces
    before = state()
    loss, _, _, tensors = route(removal('shared_trunk'), 'sample', 'genre')
    assert loss.requires_grad
    with ExitStack() as stack:
        K.patches(removal('shared_trunk'), stack, scale='sample')
        inside = state()
    assert {n for n in before if inside[n] != before[n]} == {'trunk', 'trunk_head'}


# ------------------------------------------------------------------------------------ refusals and the other knobs
def test_a_removal_with_nothing_to_remove_refuses():
    # relational: every lambda is 0
    cfg = S.config({'k3': 0}, remove=['relational'])
    with pytest.raises(AssertionError, match='no relational term'):
        patched(cfg, scale='sample')
    with pytest.raises(AssertionError, match='no relational term'):
        patched(cfg, scale='full')
    with pytest.raises(AssertionError, match='no relational term'):
        K.describe(cfg)
    # sharp_row: no relational term, or every teacher row is already at 0.05
    with pytest.raises(AssertionError, match='no relational term'):
        patched(S.config({'k3': 0}, remove=['sharp_row']), scale='sample')
    with pytest.raises(AssertionError, match='already at 0.05'):
        patched(S.config({'k4': 0.05}, remove=['sharp_row']), scale='sample')
    # a relational term on emotion alone is enough to remove
    patched(S.config({'k1': 2, 'k3': 0}, remove=['relational']), scale='sample')
    # direct: the teacher weight is 0
    for scale in SCALES:
        with pytest.raises(AssertionError, match='no direct term'):
            patched(S.config({'k5': 0}, remove=['direct']), scale=scale)
    with pytest.raises(AssertionError, match='no direct term'):
        K.describe(S.config({'k5': 0}, remove=['direct']))


@pytest.mark.parametrize('scale', SCALES)
def test_the_recorded_description_names_only_the_patches_that_are_entered(scale):
    # `ab_train.switched_head_loss` is wrapped only for a k1 or k2 term, and by sharp_row alone (relational adds none);
    # the record names it exactly then, with what happens to it
    target = 'ab_train.switched_head_loss'
    frozen = ab_train.switched_head_loss
    text = dict(relational='k1/k2 term not added', sharp_row=f'teacher row at {TAU}')
    for name in text:
        for k in ({}, {'k1': 2}, {'k2': 2}):
            with ExitStack() as stack:
                description = K.patches(removal(name, **k), stack, scale=scale)[f'no_{name}']
                wrapped = ab_train.switched_head_loss is not frozen
            assert wrapped == (name == 'sharp_row' and bool(k)), (name, k)
            assert (description.get(target) == text[name]) == bool(k) and (target in description) == bool(k), (name, k)


def test_one_removal_at_a_time():
    with pytest.raises(AssertionError, match='one removal'):
        patched(S.config(remove=['buddy', 'direct']), scale='sample')


@pytest.mark.parametrize('scale', SCALES)
def test_a_removal_keeps_the_other_knobs_of_f_as_they_are(scale, gate):
    outputs, _, _ = gate
    knobs = {'k5': 0.125, 'k6': 5e-4, 'k9': 512, 'k12': 0.07, 'k13': 10}
    for name in S.REMOVALS:
        plain = K.describe(S.config(knobs), scale)
        cfg = S.config(knobs, remove=[name])
        with ExitStack() as stack:
            description = K.patches(cfg, stack, scale=scale, outputs=outputs)
            assert ms_train.CONFIG['lr'] == 5e-4 and ms_train.CONFIG['patience'] == 10
            assert ms_train.CONFIG['temperature'] == 0.07 and ms_train.CONFIG['architecture'] == [2048, 512, 2048]
            assert V.TEACHER_WEIGHT == 0.125
        assert {k: v for k, v in description.items() if k != f'no_{name}'} == plain, name


@pytest.mark.parametrize('scale', SCALES)
def test_relational_and_sharp_row_act_on_the_runs_own_values(scale):
    # F with k1 = 8, k3 = 12, k4 = 0.025: the removal sets the lambdas to 0 / the row to 0.05, whatever F had
    f = dict(k1=8, k3=12, k4=0.025)
    frozen_loss = ab_train.switched_head_loss
    with ExitStack() as stack:
        K.patches(S.config(f, remove=['relational']), stack, scale=scale)
        assert V.ARMS['GS-0.02']['lam'] == 0 and V.ARMS['GS-0.02']['tau_t'] == 0.025
        assert ab_train.switched_head_loss is frozen_loss          # emotion's term is gone, not wrapped
    with ExitStack() as stack:
        K.patches(S.config(f, remove=['sharp_row']), stack, scale=scale)
        assert V.ARMS['GS-0.02']['lam'] == 12 and V.ARMS['GS-0.02']['tau_t'] == TAU
    loss, _, _, tensors = route(S.config(f, remove=['sharp_row']), scale, 'emotion')
    t = terms('emotion', tensors)
    assert float(loss) == pytest.approx(t['buddy'] + 0.25 * t['direct'] + 8 * t['relational'](TAU), rel=1e-5)


def test_the_full_run_record_names_the_graphs_the_run_reads():
    import rs_full
    assert rs_full.graph_choice(S.config()) == dict(k=20, variant='aspect')
    assert rs_full.graph_choice(S.config({'k11': 30})) == dict(k=30, variant='aspect')
    for name, (variant, _) in GRAPH.items():
        assert rs_full.graph_choice(removal(name, k11=30)) == dict(k=30, variant=variant)
    assert rs_full.graph_choice(removal('buddy')) == dict(k=20, variant='aspect')
