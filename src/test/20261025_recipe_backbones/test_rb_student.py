"""CPU tests of the width-opened student on tiny fake data (no GPU, no real stores)."""
import hashlib
import os
import json

import numpy as np
import pytest
import torch

import rb_paths
import rb_rows
import ms_train

ASPECTS = ('emotion', 'style', 'genre')
TRUNK_ROW = rb_rows.recipe('R0-5')['base']


def sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def models_for(width, seed=0):
    import rb_student
    torch.manual_seed(seed)
    return rb_student.build_model(TRUNK_ROW, list(ASPECTS), torch.device('cpu'), width=width)


# --- slice 1: the encoder width reaches only the image trunk's first Linear -------------------------------------

def test_width_reaches_only_the_image_trunks_first_linear():
    models, views = models_for(512)
    sheet, _ = models_for(2048)
    shapes = {k: tuple(v.shape) for k, v in models.state_dict().items()}
    reference = {k: tuple(v.shape) for k, v in sheet.state_dict().items()}
    assert shapes.keys() == reference.keys()
    changed = {k for k in shapes if shapes[k] != reference[k]}
    assert changed == {'img.trunk.0.weight'}
    assert shapes['img.trunk.0.weight'] == (1024, 512) and reference['img.trunk.0.weight'] == (1024, 2048)


def test_caption_side_and_outputs_stay_2048_wide():
    models, views = models_for(1152)
    assert tuple(models['cap'].trunk[0].weight.shape) == (1024, 2048)
    for side in ('img', 'cap'):
        for aspect in ASPECTS:
            assert tuple(models[side].heads[aspect].weight.shape) == (2048, 1024)
    z = views['emotion']['img'](torch.randn(5, 1152))
    c = views['emotion']['cap'](torch.randn(5, 2048))
    assert z.shape == c.shape == (5, 2048)
    with pytest.raises(RuntimeError):
        views['emotion']['cap'](torch.randn(5, 1152))


def test_width_2048_is_the_sheets_model_bit_for_bit():
    """The Qwen regression path: same layers, same initial weights, same state keys as the sheet's factory."""
    import rb_student
    torch.manual_seed(3)
    ours, _ = rb_student.build_model(TRUNK_ROW, list(ASPECTS), torch.device('cpu'), width=2048)
    torch.manual_seed(3)
    sheet, _ = rb_student.SHEET_BUILD_MODEL(TRUNK_ROW, list(ASPECTS), torch.device('cpu'))
    assert list(ours.state_dict()) == list(sheet.state_dict())
    for key, value in ours.state_dict().items():
        assert torch.equal(value, sheet.state_dict()[key]), key
    # the wide class itself (used for every width but 2048) consumes the generator as the sheet's TrunkSide does
    torch.manual_seed(3)
    wide = torch.nn.ModuleDict({side: rb_student.WideTrunkSide(list(ASPECTS), 2048) for side in ('img', 'cap')})
    for key, value in wide.state_dict().items():
        assert torch.equal(value, sheet.state_dict()[key]), key


# --- slice 2: the recipe's switches change the loss as the AB code does -------------------------------------------

def loss_inputs(width=64, n=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    unit = lambda x: x / x.norm(dim=1, keepdim=True)
    img = unit(torch.randn(n, width, generator=g))
    cap = unit(torch.randn(n, 2048, generator=g))
    teacher = unit(torch.randn(n, 2048, generator=g))
    pairs = np.array([[0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [6, 7], [8, 9]])
    return dict(img=img, cap=cap, teacher=teacher, pairs=pairs, weights=np.ones(len(pairs), np.float32))


def hand_loss(view, data, row, own=True, endpoints='both'):
    """Buddy (mean of four) + own-caption (mean of two) + w * teacher, from the primitives, written out by hand."""
    ids = torch.cat([torch.as_tensor(data['pairs'][:, 0]), torch.as_tensor(data['pairs'][:, 1])])
    w = torch.as_tensor(data['weights'])
    z = view['img'](data['img'][ids])
    ia, ib = z.chunk(2)
    sym = ms_train.symmetric
    teacher = row['distill_weight'] * ms_train.distill_loss(z, data['teacher'][ids], ids, 'direct')
    if endpoints == 'image':
        return sym(ia, ib, w) + teacher
    ca, cb = view['cap'](data['cap'][ids]).chunk(2)
    buddy = (sym(ia, ib, w) + sym(ia, cb, w) + sym(ca, ib, w) + sym(ca, cb, w)) / 4
    total = buddy + teacher
    if own:
        total = total + (sym(ia, ca) + sym(ib, cb)) / 2
    return total


@pytest.mark.parametrize('name', rb_rows.ORDER)
def test_recipe_switches_give_the_rules_loss(name):
    import rb_student
    recipe = rb_rows.recipe(name)
    row, switches = recipe['base'], recipe['switches']
    torch.manual_seed(0)
    _, views = rb_student.build_model(row, list(ASPECTS), torch.device('cpu'), width=64)
    view = views['style']
    view.eval()
    data = loss_inputs()
    with rb_student.loss_patch(switches):
        got = ms_train.head_loss(view, **data, row=row, aspect='style')
    want = hand_loss(view, data, row, own=bool(switches['own']), endpoints=switches['endpoints'])
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)


def test_switches_actually_change_the_loss_and_are_unwound():
    import rb_student
    losses = {}
    for name in ('R0-5', 'R6-5', 'R7-5'):
        recipe = rb_rows.recipe(name)
        torch.manual_seed(0)
        _, views = rb_student.build_model(recipe['base'], list(ASPECTS), torch.device('cpu'), width=64)
        views['style'].eval()
        # one weight w for all three, so only the switches differ
        row = dict(recipe['base'], distill_weight=0.5)
        with rb_student.loss_patch(recipe['switches']):
            losses[name] = float(ms_train.head_loss(views['style'], **loss_inputs(), row=row, aspect='style').detach())
    assert len({round(v, 6) for v in losses.values()}) == 3
    assert ms_train.head_loss is rb_student.SHEET_HEAD_LOSS     # the context leaves the sheet's loss in place


def test_default_switches_leave_the_sheets_loss_untouched():
    import rb_student
    with rb_student.loss_patch(rb_rows.recipe('R0-5')['switches']):
        assert ms_train.head_loss is rb_student.SHEET_HEAD_LOSS
    with rb_student.loss_patch(rb_rows.recipe('R6-3')['switches']):
        assert ms_train.head_loss is not rb_student.SHEET_HEAD_LOSS


# --- slice 3: the training core on tiny fake data (full-scale loop, CPU) ------------------------------------------

def teacher_map(x):
    return {ms_train.head.ASPECTS[a]: np.roll(x, 3 * i + 1, axis=1).copy() for i, a in enumerate(ASPECTS)}


def tiny_full(width, n=12, seed=31):
    """A full-scale split of n paintings: images of `width`, 2048-wide captions and teachers, the CT's graph layout."""
    import ct_held
    rng = np.random.default_rng(seed)
    unit = lambda x: (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)
    qwen, cap = unit(rng.standard_normal((n, 2048))), unit(rng.standard_normal((n, 2048)))
    images = qwen if width == 2048 else unit(rng.standard_normal((n, width)))
    fit, dev = np.array([8, 1, 7, 2, 6, 3], dtype=np.int64), np.array([11, 9, 10, 0], dtype=np.int64)
    empty = np.empty(0, dtype=np.int64)
    s = dict(mode='full', first=np.arange(n), sample_ids=np.arange(n), pid=np.arange(n),
             all_painting_ids=np.array([f'p{i:05d}' for i in range(n)]), subset_pid=empty,
             subset_image_indices=empty.copy(), fit=fit, dev=dev, fit_pairs=fit.copy())
    s['names_sha256'] = {role: ct_held.names_sha256(s['all_painting_ids'][ids])
                         for role, ids in [('fit', fit), ('dev', dev), ('eval', empty)]}
    rows = rb_rows.recipe('R0-5')['base']['graphs']
    graphs = {a: dict(fit_node_ids=fit[::-1].copy(), dev_node_ids=dev[::-1].copy(),
                      rows={rows[a]: {
                          'fit': (np.array([[0, 1], [0, 2], [1, 3], [2, 4], [3, 5]]), np.ones(5, np.float32)),
                          'dev': (np.array([[0, 1], [0, 2], [1, 3]]), np.ones(3, np.float32))}})
              for a in ASPECTS}
    manifest = dict(identity=dict(method='tiny CPU fixture'), archives={a: a for a in ASPECTS},
                    counts={a: {p: {rows[a]: len(graphs[a]['rows'][rows[a]][p][0])} for p in ('fit', 'dev')}
                            for a in ASPECTS})
    teachers = teacher_map(qwen)
    s['data_sha256'] = dict(img=sha(qwen), cap=sha(cap), fit=sha(fit), dev=sha(dev))
    for a in ASPECTS:
        s['data_sha256'][f'{a}.img'] = sha(teachers[ms_train.head.ASPECTS[a]])
    inputs = {a: dict(img=qwen, cap=cap) for a in ASPECTS}
    return s, inputs, graphs, dict.fromkeys(ASPECTS, manifest), images, teachers


@pytest.fixture
def tiny(monkeypatch):
    import ct2_data
    torch.set_num_threads(8)
    monkeypatch.setitem(ms_train.CONFIG, 'patience', 2)

    def make(width):
        data = tiny_full(width)
        teachers = data[-1]
        monkeypatch.setattr(ct2_data, 'instructed_images', lambda split, key: teachers[key])
        return data[:-1]
    return make


def test_student_at_2048_and_r0_5_equals_the_cts_loop_bit_for_bit(tmp_path, tiny):
    """The regression's premise: Qwen with R0-5 is the CT's D2 computation, outputs and best weights included."""
    import ct2_d2
    import rb_student
    s, inputs, graphs, manifests, images = tiny(2048)
    row = rb_rows.recipe('R0-5')['base']
    cpu = torch.device('cpu')
    want = ct2_d2.run(s, inputs, graphs, manifests, row, 'D2', list(ASPECTS), 0, cpu, tmp_path / 'ct', max_epochs=4)
    got = rb_student.train_cohort('full', s, inputs, graphs, manifests, images, 'qwen3vl_emb_2b', 'R0-5', 0, cpu,
                                  tmp_path / 'rb', 'regression', max_epochs=4)
    for aspect in ASPECTS:
        assert got[aspect]['best_epoch'] == want[aspect]['best_epoch']
        assert got[aspect]['best_dev_loss'] == want[aspect]['best_dev_loss']
        name = f"{aspect}__{row['graphs'][aspect]}__s0.npz"
        with np.load(tmp_path / 'ct/D2' / name) as a, np.load(tmp_path / 'rb/train/regression/R0-5' / name) as b:
            for side in ('img', 'cap'):
                np.testing.assert_array_equal(a[side].view(np.uint16), b[side].view(np.uint16))
    assert (tmp_path / 'rb/weights/regression/emotion__s0.pt').is_file()


@pytest.mark.parametrize('name', ['R0-5', 'R6-5', 'R7-5'])
def test_cpu_run_with_width_512_saves_best_weights_with_only_the_first_image_layer_opened(tmp_path, tiny, name):
    import rb_student
    s, inputs, graphs, manifests, images = tiny(512)
    got = rb_student.train_cohort('full', s, inputs, graphs, manifests, images, 'clip_b32', name, 0,
                                  torch.device('cpu'), tmp_path, 'K3', max_epochs=3)
    assert set(got) == set(ASPECTS)
    for aspect in ASPECTS:
        state = torch.load(tmp_path / f'weights/K3/{aspect}__s0.pt', weights_only=True)
        assert tuple(state['img.trunk.0.weight'].shape) == (1024, 512)
        assert tuple(state['cap.trunk.0.weight'].shape) == (1024, 2048)
        assert tuple(state['img.output.weight'].shape) == (2048, 1024)
        info = json.loads((tmp_path / f'weights/K3/{aspect}__s0.json').read_text())
        assert info['weights_sha256'] == ms_train.head.sha_file(tmp_path / f'weights/K3/{aspect}__s0.pt')
        rb = info['config']['rb']
        assert (rb['encoder'], rb['width'], rb['recipe'], rb['scale']) == ('clip_b32', 512, name, 'full')
        assert rb['switches'] == rb_rows.recipe(name)['switches'] and rb['image_sha256'] == sha(images)
        assert info['config']['row'] == rb_rows.recipe(name)['base']
        with np.load(tmp_path / f"train/K3/{name}/{aspect}__{info['config']['graph_row']}__s0.npz") as z:
            assert z['img'].shape == z['cap'].shape == (len(s['dev']), 2048)


def test_sample_scale_patches_the_sheet_loop_like_the_full_scale(tmp_path, tiny, monkeypatch):
    """Sample scale = ms_train.run with the width-opened factory, the recipe's loss, the rb config and the saver."""
    import rb_student
    s, inputs, graphs, manifests, images = tiny(512)
    seen = {}

    def recorder(*args, **kwargs):
        seen.update(args=args, kwargs=kwargs, build=ms_train.build_model, loss=ms_train.head_loss,
                    make_config=ms_train.make_config, save=ms_train.save_output)
        models, _ = ms_train.build_model(args[4], list(ASPECTS), torch.device('cpu'))
        seen['img_weight'] = tuple(models['img'].trunk[0].weight.shape)
        seen['config'] = ms_train.make_config(args[0], args[4], args[5], 'style', list(ASPECTS), args[3], 0,
                                              torch.device('cpu'), False, 200)
        return 'ran'

    monkeypatch.setattr(ms_train, 'run', recorder)
    # the sample config reads the sample fingerprint and the sheet's graph manifest: the tiny split has both
    s['sample_ids'], s['subset_pid'] = s['sample_ids'].astype(np.int64), s['subset_pid']
    out = rb_student.train_cohort('sample', s, inputs, graphs, manifests, images, 'clip_b32', 'R6-3', 1,
                                  torch.device('cpu'), tmp_path, 'K3', max_epochs=7)
    assert out == 'ran'
    args, kwargs = seen['args'], seen['kwargs']
    assert args[4] == rb_rows.recipe('R6-3')['base'] and args[5] == 'R6-3' and args[7] == 1 and kwargs['max_epochs'] == 7
    assert args[2] is graphs and all(args[1][a]['img'] is images for a in ASPECTS)
    assert args[9] == tmp_path / 'train' / 'K3'
    assert seen['img_weight'] == (1024, 512)
    assert seen['loss'] is not rb_student.SHEET_HEAD_LOSS and seen['build'] is not rb_student.SHEET_BUILD_MODEL
    assert seen['config']['rb']['encoder'] == 'clip_b32' and seen['config']['rb']['switches']['own'] == 0.0
    assert ms_train.run is recorder and ms_train.build_model is rb_student.SHEET_BUILD_MODEL
    assert ms_train.make_config is rb_student.SHEET_MAKE_CONFIG and ms_train.save_output is rb_student.SHEET_SAVE_OUTPUT


@pytest.mark.parametrize('scale', ['full', 'sample'])
@pytest.mark.parametrize('name', ['R6-3', 'R7-5'])
def test_the_loop_runs_under_the_switched_loss_at_both_scales(tmp_path, tiny, monkeypatch, scale, name):
    """Wrap the loop itself (no training): what it sees as `ms_train.head_loss` is the recipe's switched loss."""
    import ct2_d2
    import rb_student
    s, inputs, graphs, manifests, images = tiny(512)
    seen = []

    def spy(*args, **kwargs):
        seen.append(ms_train.head_loss)
        return 'ran'

    monkeypatch.setattr(ms_train, 'run', spy)
    monkeypatch.setattr(ct2_d2, 'run', spy)
    rb_student.train_cohort(scale, s, inputs, graphs, manifests, images, 'clip_b32', name, 0, torch.device('cpu'),
                            tmp_path, 'K3', max_epochs=2)
    (loss,) = seen
    assert loss is not rb_student.SHEET_HEAD_LOSS and loss.func is rb_student.ab_train.switched_head_loss
    assert loss.keywords['switches'] == rb_rows.recipe(name)['switches']
    assert ms_train.head_loss is rb_student.SHEET_HEAD_LOSS
    seen.clear()
    rb_student.train_cohort(scale, s, inputs, graphs, manifests, images, 'clip_b32', 'R0-5', 0, torch.device('cpu'),
                            tmp_path, 'K3', max_epochs=2)
    assert seen == [rb_student.SHEET_HEAD_LOSS]


def test_train_cohort_refuses_a_recipe_with_other_inputs(tmp_path, tiny, monkeypatch):
    import rb_student
    s, inputs, graphs, manifests, images = tiny(512)
    real = rb_rows.recipe

    def other(name):
        recipe = real(name)
        recipe['base']['inputs'] = 'instructed'
        return recipe

    monkeypatch.setattr(rb_student.rb_rows, 'recipe', other)
    with pytest.raises(AssertionError, match='inputs'):
        rb_student.train_cohort('full', s, inputs, graphs, manifests, images, 'clip_b32', 'R0-5', 0,
                                torch.device('cpu'), tmp_path, 'K3', max_epochs=2)


@pytest.fixture
def driver(tiny, monkeypatch):
    import rb_student

    def make(width):
        s, inputs, graphs, manifests, images = tiny(width)
        monkeypatch.setattr(rb_student, 'load_scale', lambda scale: (s, inputs, graphs, manifests))
        monkeypatch.setattr(rb_student, 'training_images', lambda encoder, split, inp: images)
        return s, images
    return make


def test_train_resumes_replaces_incomplete_and_refuses_after_held(tmp_path, driver):
    import rb_student
    driver(512)
    first = rb_student.train('full', 'clip_b32', 'R0-5', [0], 'cpu', tmp_path, max_epochs=2)
    weights = tmp_path / 'part2/full/weights/K3/emotion__s0.pt'
    stamp = weights.stat().st_mtime_ns
    assert set(first) == {0} and weights.is_file()
    assert rb_student.train('full', 'clip_b32', 'R0-5', [0], 'cpu', tmp_path, max_epochs=2) == {}   # verified; skip
    assert weights.stat().st_mtime_ns == stamp
    (tmp_path / 'part2/full/weights/K3/genre__s0.pt').unlink()                                       # incomplete cohort
    again = rb_student.train('full', 'clip_b32', 'R0-5', [0], 'cpu', tmp_path, max_epochs=2)
    assert set(again) == {0} and (tmp_path / 'part2/full/weights/K3/genre__s0.pt').is_file()
    assert list((tmp_path / 'part2/full/replaced/K3').glob('s0-*/weights/K3/emotion__s0.pt'))
    with pytest.raises(AssertionError, match='another recipe'):
        rb_student.train('full', 'clip_b32', 'R6-5', [0], 'cpu', tmp_path, max_epochs=2, arm='K3')
    held = tmp_path / 'part2/full/held/K3'
    held.mkdir(parents=True)
    (held / 'emotion__s0.npz').write_bytes(b'x')
    (tmp_path / 'part2/full/weights/K3/genre__s0.pt').unlink()
    with pytest.raises(AssertionError, match='V4'):
        rb_student.train('full', 'clip_b32', 'R0-5', [0], 'cpu', tmp_path, max_epochs=2)
    (held / 'emotion__s0.npz').rename(held / 'emotion__s3.npz')                  # a held output of another seed
    with pytest.raises(AssertionError, match='V4'):
        rb_student.train('full', 'clip_b32', 'R0-5', [0], 'cpu', tmp_path, max_epochs=2)


def test_arms_and_smoke_rules(tmp_path):
    import rb_student
    assert rb_student.arm_for('siglip2_so400m', 'R0-5') == 'K2' and rb_student.arm_for('clip_b32', 'R7-3') == 'K3'
    assert rb_student.arm_for('qwen3vl_emb_2b', 'R6-3') == 'K4' and rb_student.arm_for('dinov2_large', 'R7-5') == 'K5'
    with pytest.raises(AssertionError, match='K4'):
        rb_student.arm_for('qwen3vl_emb_2b', 'R0-5')
    with pytest.raises(AssertionError, match='K5'):
        rb_student.arm_for('dinov2_large', 'R6-3')
    with pytest.raises(AssertionError, match='smoke'):
        rb_student.train('full', 'clip_b32', 'R0-5', [0], 'cpu', tmp_path, smoke=True)    # outputs outside /tmp/claude-0


# --- slice 4: held and ArtELingo-28 inference, output files, the regression gate ----------------------------------

def test_infer_is_the_best_view_on_the_encoder_image(tmp_path, driver):
    import torch.nn.functional as F
    import rb_student
    s, images = driver(512)
    rb_student.train('full', 'clip_b32', 'R0-5', [0], 'cpu', tmp_path, max_epochs=2)
    path = tmp_path / 'part2/full/weights/K3/style__s0.pt'
    x = np.random.default_rng(5).standard_normal((700, 512)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    got = rb_student.infer(path, 'style', 512, x, torch.device('cpu'))
    w = torch.load(path, weights_only=True)
    hidden = F.gelu(torch.from_numpy(x) @ w['img.trunk.0.weight'].T + w['img.trunk.0.bias'])
    want = F.normalize(hidden @ w['img.output.weight'].T + w['img.output.bias'], dim=1)
    assert got.shape == (700, 2048) and got.dtype == np.float16
    np.testing.assert_allclose(got.astype(np.float32), want.numpy(), atol=2e-3)
    with pytest.raises(AssertionError):
        rb_student.infer(path, 'style', 512, x[:, :256], torch.device('cpu'))
    with pytest.raises(RuntimeError):
        rb_student.infer(path, 'style', 2048, np.zeros((3, 2048), np.float32) + 1, torch.device('cpu'))


def test_outputs_carry_shapes_dtypes_hashes_and_detect_changes(tmp_path):
    import rb_student
    x = np.random.default_rng(0).standard_normal((4, 2048)).astype(np.float16)
    path = tmp_path / 'a/b.npz'
    rb_student.write_outputs(path, dict(img=x), dict(arm='K2'))
    info = json.loads(path.with_suffix('.json').read_text())
    assert info['arrays']['img'] == dict(shape=[4, 2048], dtype='float16', sha256=sha(x))
    assert info['arm'] == 'K2' and set(info['code_sha256']) >= {'rb_student.py', 'rb_rows.py', 'rb_encoders.py'}
    arrays, got = rb_student.read_outputs(path)
    np.testing.assert_array_equal(arrays['img'], x)
    other = x.copy()
    other[0, 0] += 1
    np.savez(path, img=other)
    with pytest.raises(AssertionError, match='sha256'):
        rb_student.read_outputs(path)


@pytest.fixture
def served(driver, monkeypatch, tmp_path):
    """Trained K3 weights plus fake held and ArtELingo-28 providers (width 512)."""
    import rb_encoders
    import rb_student
    s, images = driver(512)
    rb_student.train('full', 'clip_b32', 'R6-5', [0], 'cpu', tmp_path, max_epochs=2)
    rng = np.random.default_rng(9)
    unit = lambda x: (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)
    held = unit(rng.standard_normal((9, 512)))
    a28 = unit(rng.standard_normal((5, 512)))
    names = [f'h{i}' for i in range(9)]
    monkeypatch.setattr(rb_student, 'held_names', lambda: (names, rb_paths_sha('held')))
    monkeypatch.setattr(rb_encoders, 'full_images', lambda enc, ns: held if list(ns) == names else pytest.fail('names'))
    monkeypatch.setattr(rb_encoders, 'a28_names', lambda: [f'a{i}' for i in range(5)])
    monkeypatch.setattr(rb_encoders, 'a28_sha256', lambda: rb_paths_sha('a28'))
    monkeypatch.setattr(rb_encoders, 'a28_images', lambda enc: a28)
    return held, a28


def rb_paths_sha(kind):
    return hashlib.sha256(kind.encode()).hexdigest()


def test_held_inference_needs_the_stage_gate_and_writes_the_index_files(tmp_path, served):
    import rb_student
    held, a28 = served
    with pytest.raises(AssertionError, match='V1'):
        rb_student.infer_arm('full', 'K3', [0], ('held',), tmp_path, 'cpu')
    gate = tmp_path / 'part2/regression/stage3a.json'
    gate.parent.mkdir(parents=True)
    gate.write_text(json.dumps(dict(pas=True, items={})))
    with pytest.raises((AssertionError, KeyError)):                            # no 'pass' key (shared rule: KeyError)
        rb_student.infer_arm('full', 'K3', [0], ('held',), tmp_path, 'cpu')
    gate.write_text(json.dumps({'pass': True, 'items': {}}))                     # no passing student/matched/scorer item
    with pytest.raises(AssertionError, match='V1'):
        rb_student.infer_arm('full', 'K3', [0], ('held',), tmp_path, 'cpu')
    items = {f'{group}x': {'pass': True} for group in ('student/', 'matched/', 'scorer/')}
    gate.write_text(json.dumps({'pass': True, 'items': dict(items, **{'matched/bad': {'pass': False}})}))
    with pytest.raises(AssertionError, match='V1'):                                # file says pass, an item fails
        rb_student.infer_arm('full', 'K3', [0], ('held',), tmp_path, 'cpu')
    gate.write_text(json.dumps({'pass': True, 'items': {k: v for k, v in items.items() if k != 'scorer/x'}}))
    with pytest.raises(AssertionError, match='V1'):                                # a group has no passing item
        rb_student.infer_arm('full', 'K3', [0], ('held',), tmp_path, 'cpu')
    gate.write_text(json.dumps({'pass': True, 'items': items}))
    (tmp_path / 'part2/verdict_stage3.json').write_text(json.dumps(dict(blocks={})))   # stage 4's V1 reads it too
    rb_student.infer_arm('full', 'K3', [0], ('held', 'a28'), tmp_path, 'cpu')
    for aspect in ASPECTS:
        for kind, rows, names in (('held', 9, 'held'), ('a28', 5, 'a28')):
            arrays, info = rb_student.read_outputs(tmp_path / f'part2/full/{kind}/K3/{aspect}__s0.npz')
            assert set(arrays) == {'img'} and arrays['img'].shape == (rows, 2048) and arrays['img'].dtype == np.float16
            assert (info['rows'], info['width'], info['dtype'], info['scale'], info['seed']) == (rows, 2048, 'float16', 'full', 0)
            assert info['sha256'] == sha(arrays['img'])
            assert info['list_sha256'] == rb_paths_sha(names) and info['arm'] == 'K3' and info['aspect'] == aspect
            winfo = json.loads((tmp_path / f'part2/full/weights/K3/{aspect}__s0.json').read_text())
            assert info['weights_sha256'] == winfo['weights_sha256']
    # the A28 output is the best view on the A28 image, not on the held image
    path = tmp_path / 'part2/full/weights/K3/emotion__s0.pt'
    arrays, _ = rb_student.read_outputs(tmp_path / 'part2/full/a28/K3/emotion__s0.npz')
    np.testing.assert_array_equal(arrays['img'], rb_student.infer(path, 'emotion', 512, a28, torch.device('cpu')))
    # a second run verifies and keeps the files
    stamp = (tmp_path / 'part2/full/a28/K3/emotion__s0.npz').stat().st_mtime_ns
    rb_student.infer_arm('full', 'K3', [0], ('a28',), tmp_path, 'cpu')
    assert (tmp_path / 'part2/full/a28/K3/emotion__s0.npz').stat().st_mtime_ns == stamp


def test_inference_refuses_a_device_other_than_the_training_device(tmp_path, served):
    import rb_student
    with pytest.raises(AssertionError, match='device'):
        rb_student.infer_arm('full', 'K3', [0], ('a28',), tmp_path, 'cuda')


# --- slice 5: K1 from the CT's saved weights; the regressions ------------------------------------------------------

@pytest.fixture
def ct_reference(tmp_path, driver, monkeypatch):
    """A CT-layout D2 (weights, held outputs) trained by the CT's own `ct2_d2.train` and `ct_d2.infer` on tiny data."""
    import sys
    from types import SimpleNamespace
    import ct2_d2
    import ct_d2
    import ct_held
    import rb_encoders
    import rb_student
    s, images = driver(2048)
    monkeypatch.setitem(ms_train.CONFIG, 'max_epochs', 4)
    monkeypatch.setitem(sys.modules, 'ct2_stage2a', SimpleNamespace(require_pass=lambda **kw: None))
    monkeypatch.setattr(ct2_d2, 'load_training_data', lambda mode, root: rb_student.load_scale('full'))
    ct = tmp_path / 'ct'
    ct2_d2.train('full', [0], 'cpu', ct)
    held = np.random.default_rng(4).standard_normal((9, 2048)).astype(np.float32)
    held /= np.linalg.norm(held, axis=1, keepdims=True)
    names = [f'h{i}' for i in range(9)]
    monkeypatch.setattr(ct_held, 'qwen_default_images', lambda ns: held)
    for aspect in ASPECTS:
        out = ct_d2.infer(aspect, 0, names, outdir=ct)
        (ct / 'held/d2').mkdir(parents=True, exist_ok=True)
        np.savez(ct / 'held/d2' / f'{aspect}__s0.npz', img=out)
    monkeypatch.setattr(rb_student, 'ct_root', lambda scale: ct)
    monkeypatch.setattr(rb_student, 'held_names', lambda: (names, rb_paths.HELD_SHA256))
    monkeypatch.setattr(rb_encoders, 'full_images', lambda enc, ns: held)
    a28 = np.random.default_rng(6).standard_normal((5, 2048)).astype(np.float32)
    a28 /= np.linalg.norm(a28, axis=1, keepdims=True)
    monkeypatch.setattr(rb_encoders, 'a28_names', lambda: [f'a{i}' for i in range(5)])
    monkeypatch.setattr(rb_encoders, 'a28_sha256', lambda: rb_paths_sha('a28'))
    monkeypatch.setattr(rb_encoders, 'a28_images', lambda enc: a28)
    return ct, held, a28


def test_regress_passes_when_the_student_is_the_cts_d2_and_merges_items(tmp_path, ct_reference):
    import rb_student
    ct, held, a28 = ct_reference
    out = tmp_path / 'rb'
    gate = out / 'part2/regression/stage3a.json'
    gate.parent.mkdir(parents=True)
    gate.write_text(json.dumps({'pass': True, 'items': {'matched/P3': {'pass': True, 'note': 'other ticket'}}}))
    record = rb_student.regress('stage3a', 'cpu', out)
    assert record['pass'] is True and 'matched/P3' in record['items']
    for aspect in ASPECTS:
        item = record['items'][f'student/{aspect}']
        assert item['pass'] is True
        for name in ('weights', 'best_epoch', 'held', 'k1_replay_held'):
            assert item[name]['pass'] is True
    assert json.loads(gate.read_text()) == record
    # the regression's own files stay out of the arms' folders
    assert not (out / 'part2/full/weights').exists() and (out / 'part2/regression/stage3a_run/held/regression').is_dir()


def test_regress_writes_through_the_shared_locked_writer(tmp_path, ct_reference, monkeypatch):
    import rb_score
    import rb_student
    calls = []
    original = rb_score.write_regression_items
    monkeypatch.setattr(rb_score, 'write_regression_items', lambda *a, **k: calls.append(a[0]) or original(*a, **k))
    out = tmp_path / 'rb'
    record = rb_student.regress('stage3a', 'cpu', out)
    path = out / 'part2/regression/stage3a.json'
    assert calls == [path] and json.loads(path.read_text()) == record
    assert path.with_name('stage3a.json.lock').is_file()                      # the shared writer's fcntl lock
    assert not hasattr(rb_student, 'write_regression_items')


def test_regress_fails_on_one_flipped_bit(tmp_path, ct_reference):
    import rb_student
    ct, held, a28 = ct_reference
    path = ct / 'held/d2/style__s0.npz'
    with np.load(path) as z:
        x = z['img'].copy()
    x.view(np.uint16)[3, 7] ^= 1
    np.savez(path, img=x)
    record = rb_student.regress('stage3a', 'cpu', tmp_path / 'rb')
    assert record['pass'] is False
    failing = sorted(k for k, v in record['items'].items() if not v['pass'])
    assert failing == ['student/style']
    item = record['items']['student/style']
    assert [k for k in ('weights', 'best_epoch', 'held', 'k1_replay_held') if not item[k]['pass']] == ['held', 'k1_replay_held']


def test_regress_must_run_on_the_cts_device(tmp_path, ct_reference):
    import rb_student
    with pytest.raises(AssertionError, match='must run on'):
        rb_student.regress('stage3a', 'cuda', tmp_path / 'rb')


def test_k1_infers_artelingo28_from_the_cts_saved_weights(tmp_path, ct_reference, monkeypatch):
    import ct_d2
    import ct_held
    import rb_student
    ct, held, a28 = ct_reference
    rb_student.infer_arm('full', 'K1', [0], ('a28',), tmp_path / 'rb', None)
    monkeypatch.setattr(ct_held, 'qwen_default_images', lambda ns: a28)      # the CT's own inference, as the oracle
    for aspect in ASPECTS:
        arrays, info = rb_student.read_outputs(tmp_path / f'rb/part2/full/a28/K1/{aspect}__s0.npz')
        assert arrays['img'].shape == (5, 2048) and info['arm'] == 'K1' and info['recipe'] is None
        assert info['encoder'] == 'qwen3vl_emb_2b' and info['image_sha256'] == sha(a28)
        np.testing.assert_array_equal(arrays['img'], ct_d2.infer(aspect, 0, [f'a{i}' for i in range(5)], outdir=ct))
    with pytest.raises(AssertionError, match='K1 held'):
        rb_student.infer_arm('full', 'K1', [0], ('held',), tmp_path / 'rb', None)


# --- slice 6: the CLI ------------------------------------------------------------------------------------------------

def test_picked_recipe_reads_pick_json_and_checks_outcome(tmp_path):
    import rb_student
    pick = tmp_path / 'part1/pick.json'
    pick.parent.mkdir(parents=True)
    pick.write_text(json.dumps(dict(outcome='P-KEEP', recipe='R0-5')))
    assert rb_student.picked_recipe(tmp_path) == 'R0-5'
    pick.write_text(json.dumps(dict(outcome='P-R6-3', recipe='R6-3')))
    assert rb_student.picked_recipe(tmp_path) == 'R6-3'
    for bad in (dict(outcome='P-KEEP', recipe='R6-3'), dict(outcome='P-R6-3', recipe='R7-3'), dict(outcome='P-R9', recipe='R9')):
        pick.write_text(json.dumps(bad))
        with pytest.raises(AssertionError):
            rb_student.picked_recipe(tmp_path)


def test_k4_and_k5_follow_the_pick_when_pick_json_exists(tmp_path, driver):
    import rb_student
    driver(1024)                                      # DINOv2-large width
    # without pick.json nothing changes: any recipe trains
    rb_student.train('full', 'dinov2_large', 'R7-3', [0], 'cpu', tmp_path / 'free', max_epochs=2)
    assert (tmp_path / 'free/part2/full/weights/K5/emotion__s0.pt').is_file()
    pick = tmp_path / 'part1/pick.json'
    pick.parent.mkdir(parents=True)
    pick.write_text(json.dumps(dict(outcome='P-R7-5', recipe='R7-5')))
    for encoder, recipe in (('dinov2_large', 'R7-3'), ('qwen3vl_emb_2b', 'R6-3')):
        with pytest.raises(AssertionError, match='pick.json'):
            rb_student.train('full', encoder, recipe, [0], 'cpu', tmp_path, max_epochs=2)
    assert not (tmp_path / 'part2').exists()
    rb_student.train('full', 'dinov2_large', 'R7-5', [0], 'cpu', tmp_path, max_epochs=2)        # the picked recipe
    assert (tmp_path / 'part2/full/weights/K5/emotion__s0.pt').is_file()
    # K2 and K3 are not tied to the pick
    driver(512)
    rb_student.train('full', 'clip_b32', 'R0-5', [0], 'cpu', tmp_path, max_epochs=2)
    assert (tmp_path / 'part2/full/weights/K3/emotion__s0.pt').is_file()


def test_cli_train_sets_the_ct_determinism_env_and_passes_the_pick(tmp_path, monkeypatch):
    import rb_student
    pick = tmp_path / 'part1/pick.json'
    pick.parent.mkdir(parents=True)
    pick.write_text(json.dumps(dict(outcome='P-R7-5', recipe='R7-5')))
    calls = []
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG', raising=False)
    monkeypatch.setattr(rb_student, 'train', lambda *a, **k: calls.append((a, k)))
    rb_student.main(['train', '--scale', 'sample', '--encoder', 'dinov2_large', '--pick', '--seeds', '0', '2',
                     '--device', 'cpu', '--outputs', str(tmp_path)])
    assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
    (args, kwargs), = calls
    assert args[:4] == ('sample', 'dinov2_large', 'R7-5', [0, 2]) and kwargs['smoke'] is False
    with pytest.raises(SystemExit):     # a recipe and --pick together, or neither
        rb_student.main(['train', '--scale', 'sample', '--encoder', 'clip_b32', '--pick', '--recipe', 'R0-5'])
    with pytest.raises(SystemExit):
        rb_student.main(['train', '--scale', 'sample', '--encoder', 'clip_b32'])


def test_cli_infer_and_regress_dispatch(tmp_path, monkeypatch):
    import rb_student
    calls = []
    monkeypatch.setattr(rb_student, 'infer_arm', lambda *a, **k: calls.append(('infer', a, k)))
    monkeypatch.setattr(rb_student, 'regress', lambda *a, **k: calls.append(('regress', a, k)))
    rb_student.main(['infer', '--scale', 'full', '--arm', 'K1', '--kinds', 'a28', '--outputs', str(tmp_path)])
    rb_student.main(['regress', 'stage2', '--device', 'cuda', '--outputs', str(tmp_path)])
    assert calls[0][1][:4] == ('full', 'K1', [0, 1, 2, 3, 4], ['a28']) and calls[0][2]['smoke'] is False
    assert calls[1][0] == 'regress' and calls[1][1] == ('stage2', 'cuda', tmp_path)


def test_r0_5_is_the_cts_d2_row():
    import ct_d2
    assert rb_rows.recipe('R0-5')['base'] == ct_d2.ROWS['D2']
    assert rb_rows.recipe('R0-5')['switches'] == dict(teacher='instructed', buddy=1.0, own=1.0, endpoints='both')


# --- real data (RB_REAL=1): the sample scale of the CT, read only -------------------------------------------------

@pytest.mark.skipif(not os.environ.get('RB_REAL'), reason='reads the CT sample, graphs and stores; RB_REAL=1')
def test_real_sample_qwen_student_inputs_and_config_are_the_cts():
    import ct_d2
    import rb_student
    s, inputs, graphs, manifests = rb_student.load_scale('sample')
    images = rb_student.training_images('qwen3vl_emb_2b', s, inputs)        # asserts equality with the CT array
    assert images.shape == (19889, 2048) and images.dtype == np.float32
    for aspect in ASPECTS:
        base = rb_rows.recipe('R0-5')['base']
        config = rb_student._with_rb(rb_student.SHEET_MAKE_CONFIG, dict(x=1))(
            s, base, 'D2', aspect, list(ASPECTS), manifests, 0, torch.device('cuda'), False, 200)
        saved = json.loads((rb_paths.CT_OUT / f'd2/weights/{aspect}__s0.json').read_text())['config']
        assert {k: v for k, v in config.items() if k != 'rb'} == saved
        info = rb_student._arm_weights('sample', 'K1', aspect, 0, None, False)
        assert info[2] == 'qwen3vl_emb_2b' and info[0].is_file()
    assert rb_student.held_names()[1] == rb_paths.HELD_SHA256


@pytest.mark.skipif(not os.environ.get('RB_REAL'), reason='loads the full-scale CT data, about a minute; RB_REAL=1')
def test_real_full_qwen_student_inputs_and_config_are_the_cts():
    import ct2_d2
    import rb_student
    s, inputs, graphs, manifests = rb_student.load_scale('full')
    images = rb_student.training_images('qwen3vl_emb_2b', s, inputs)
    assert images.shape == (49121, 2048) and images.dtype == np.float32
    for aspect in ASPECTS:
        config = ct2_d2.make_config(s, rb_rows.recipe('R0-5')['base'], 'D2', aspect, list(ASPECTS), manifests, 0,
                                    torch.device('cuda'), False, 200)
        saved = json.loads((rb_paths.CT_FULL / f'd2/weights/{aspect}__s0.json').read_text())['config']
        assert config == saved
