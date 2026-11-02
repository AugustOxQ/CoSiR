"""Tests of bs_cstar (r14 ticket 03): rows, the encoder wiring, keep modes, graph seam, exports, the stage 0 comparisons."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import bs_cstar as C
import bs_paths as P
from bs_cstar import S, V, ms_train, rb_encoders, rb_rows, rb_student, rg_full, rs_graphs, rs_knobs

CPU = torch.device('cpu')
R10_WEIGHTS = P.R10_OUT / 'full/weights/k11=40'
R10_HELD = P.R10_OUT / 'full/held/k11=40/emotion__s0.json'
needs_r10 = pytest.mark.skipif(not R10_WEIGHTS.exists(), reason='r10 outputs are not on this machine')


# ------------------------------------------------------------------------------------------------------ rows
def test_knob_rows_are_r10s_with_neighbours_10_20_80():
    rows = C.knob_rows()
    assert len(rows) == 39 and len(set(rows)) == 39
    by_knob = {k: [r.split('=')[1] for r in rows if r.split('=')[0] == k] for k in S.KNOBS}
    assert by_knob['k11'] == ['10', '20', '80']
    for k in S.KNOBS:
        if k != 'k11':
            assert [float(v) for v in by_knob[k]] == [float(v) for v in S.VALUES[k]], k
    assert rows[:3] == ['k1=2', 'k1=8', 'k1=16'] and rows[-3:] == ['k13=5', 'k13=10', 'k13=30']


def test_every_row_moves_exactly_one_knob_from_cstar():
    assert C.STAR == S.config({'k11': 40}) and S.name(C.STAR) == 'k11=40'
    assert C.row_config('ref') == C.STAR and C.row_arm('ref') == 'k11=40'
    for row in C.knob_rows():
        knob, value = row.split('=')
        cfg = C.row_config(row)
        assert [k for k in S.KNOBS if cfg[k] != C.STAR[k]] == [knob], row
        assert cfg[knob] == float(value) and C.row_arm(row) == row


@pytest.mark.parametrize('row', ['k11=40', 'k6=0.01', 'k11=30', 'k14=1', 'bogus', 'k6=', '=3'])
def test_rows_outside_the_sweep_are_refused(row):
    with pytest.raises(AssertionError):
        C.row_config(row)


def test_knob_rows_are_qwen_only():
    with pytest.raises(AssertionError, match='Qwen only'):
        with C.training_context('siglip2', 'k6=0.0005'):
            pass


# -------------------------------------------------------------------------------- the encoder in the model
def synthetic(width, n=6):
    rng = np.random.default_rng(0)
    unit = lambda x: (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)
    s = dict(first=np.arange(n), fit=np.arange(4), dev=np.arange(4, n), all_painting_ids=np.array(list('abcdefgh'[:n])))
    images = unit(rng.normal(size=(n, width)))
    inputs = {a: dict(img=unit(rng.normal(size=(n, 2048))), cap=unit(rng.normal(size=(n, 2048)))) for a in C.ASPECTS}
    return s, inputs, images


@pytest.fixture
def spy(monkeypatch, tmp_path):
    """Run `rg_full.train_cohort` for an encoder and row with `ct2_d2.run` replaced by a probe that builds the model the
    way the loop does and records what the loop would read."""
    seen = {}

    def probe(s, inputs, graphs, manifests, row, name, aspects, seed, device, outdir, smoke=False, max_epochs=200):
        models, views = ms_train.build_model(row, aspects, CPU)
        seen.update(row=row, name=name, config=dict(ms_train.CONFIG), arm_switches=None,
                    models={k: {n: tuple(p.shape) for n, p in m.state_dict().items()} for k, m in models.items()},
                    view={n: tuple(p.shape) for n, p in views['emotion'].state_dict().items()},
                    img_width={a: inputs[a]['img'].shape[1] for a in aspects}, load_scale=rb_student.load_scale,
                    dropout={k: [m.p for m in models[k].modules() if isinstance(m, torch.nn.Dropout)] for k in models})
        return {}
    monkeypatch.setattr(rg_full.ct2_d2, 'run', probe)

    def run(enc, row='ref'):
        with C.training_context(enc, row) as ctx:
            s, inputs, images = synthetic(ctx.width)
            rb = rb_student._rb_block(rg_full.SCALE, ctx.backbone, rg_full.RECIPE, rb_rows.recipe(rg_full.RECIPE), images)
            rg = rg_full.rg_block(ctx.arm, 0, len(s['fit']))
            rg_full.train_cohort(s, inputs, {}, {}, images, ctx.arm, 0, CPU, tmp_path, rb=rb)
            seen.update(ctx=ctx, rb=rb, rg=rg, arm_switches=V.switches(ctx.arm))
        return dict(seen)
    return run


@pytest.mark.parametrize('enc', ['siglip2', 'siglip1', 'clip_l14'])
def test_another_width_changes_only_the_image_trunks_first_layer(spy, enc):
    qwen, other = spy('qwen'), spy(enc)
    width = P.ENCODERS[enc]['width']
    assert other['img_width'] == dict.fromkeys(C.ASPECTS, width) and qwen['img_width'] == dict.fromkeys(C.ASPECTS, 2048)
    assert other['rb']['width'] == width and other['rb']['encoder'] == P.ENCODERS[enc]['art']
    first = 'trunk.0.weight'
    assert list(other['models']) == list(qwen['models'])
    for side in qwen['models']:
        assert list(other['models'][side]) == list(qwen['models'][side]), side      # same layers, same order
        for name, shape in qwen['models'][side].items():
            if (side, name) == ('img', first):
                assert shape == (1024, 2048) and other['models'][side][name] == (1024, width)
            else:
                assert other['models'][side][name] == shape, (side, name)
    # everything else that resolves is c*'s: the training row, the loop's settings, the loss switches, the output width
    assert other['row'] == qwen['row'] and other['config'] == qwen['config'] and other['arm_switches'] == qwen['arm_switches']
    assert other['view'][f'img.{first}'] == (1024, width) and other['view']['img.output.weight'] == (2048, 1024)
    assert other['dropout'] == qwen['dropout']


@needs_r10
def test_the_qwen_path_is_r10s_cstar_model_and_configuration(spy):
    got = spy('qwen')
    info = json.loads((R10_WEIGHTS / 'emotion__s0.json').read_text())['config']
    theirs = torch.load(R10_WEIGHTS / 'emotion__s0.pt', map_location='cpu', weights_only=True)
    assert {k: tuple(v.shape) for k, v in theirs.items()} == got['view']                 # keys, order and shapes
    assert list(theirs) == list(got['view'])
    for key, value in got['config'].items():
        assert info[key] == value, key                                                  # architecture, lr, patience, ...
    assert info['row'] == got['row']
    assert (info['rg']['arm'], info['rg']['switches']) == ('k11=40', got['arm_switches']) == (got['ctx'].arm, got['rg']['switches'])
    assert info['rg']['anchors'] == got['rg']['anchors']
    for key in ('scale', 'encoder', 'width', 'recipe', 'switches'):
        assert info['rb'][key] == got['rb'][key], key
    assert got['ctx'].knobs == {'k11': {'rb_student.load_scale': "rs_graphs.full_training_data(k=40, 'aspect')"}}
    assert json.loads((P.R10_OUT / 'full/train/k11=40/run__s0.json').read_text())['knobs'] == got['ctx'].knobs


def test_the_graph_seam_is_ours_and_asks_for_the_rows_neighbours(spy, monkeypatch):
    calls = []
    monkeypatch.setattr(C, 'graph_data', lambda *a: calls.append(a) or 'data')
    for row, k in (('ref', 40), ('k11=10', 10), ('k11=80', 80)):
        with C.training_context('qwen', row, graph_out=Path('/nowhere')):
            assert rb_student.load_scale('full') == 'data'
        assert calls[-1] == ('full', k, Path('/nowhere'), 'aspect'), row
    # k = 20 is G's: the CT's own graphs, no seam
    frozen = rb_student.load_scale
    with C.training_context('qwen', 'k11=20'):
        assert rb_student.load_scale is frozen
    assert rb_student.load_scale is frozen        # and the patches are gone after a context


def test_contexts_leave_the_frozen_modules_as_they_were():
    before = (rg_full.ENCODER, dict(rb_encoders.WIDTH), dict(ms_train.CONFIG), {k: dict(v) for k, v in V.ARMS.items()})
    with C.training_context('siglip2', 'ref'):
        assert rg_full.ENCODER == 'siglip2_so400m' and rb_encoders.WIDTH['siglip2_so400m'] == 1152
    with C.training_context('qwen', 'k6=0.0005'), pytest.raises(RuntimeError):
        assert ms_train.CONFIG['lr'] == 0.0005
        raise RuntimeError
    assert before == (rg_full.ENCODER, dict(rb_encoders.WIDTH), dict(ms_train.CONFIG), {k: dict(v) for k, v in V.ARMS.items()})


@pytest.mark.parametrize('enc', ['siglip1', 'clip_l14', 'siglip2'])
def test_the_width_patch_is_removed_on_exit_and_qwens_width_is_never_touched(enc):
    r10_table = {'siglip2_so400m': 1152, 'clip_b32': 512, 'qwen3vl_emb_2b': 2048, 'dinov2_large': 1024}   # rb_encoders, as r10 has it
    before = dict(rb_encoders.WIDTH)
    assert before == r10_table                  # no earlier test or context left a patch behind
    with C.training_context(enc, 'ref'):
        assert rb_encoders.WIDTH[P.ENCODERS[enc]['art']] == P.ENCODERS[enc]['width']
        assert rb_encoders.WIDTH['qwen3vl_emb_2b'] == 2048
    assert rb_encoders.WIDTH == before and rb_encoders.WIDTH['qwen3vl_emb_2b'] == 2048


# ------------------------------------------------------------------------------------------------- graph stores
def test_graphs_come_from_r10_when_it_has_them_and_are_built_here_otherwise(monkeypatch, tmp_path):
    monkeypatch.setattr(C.P, 'OUT', tmp_path)
    monkeypatch.setattr(C, 'require_graph_gate', lambda: None)
    seen = []
    monkeypatch.setattr(rs_graphs, '_materialize', lambda scale, k, variant, out, **kw: seen.append((scale, k, variant, Path(out))) or (None, 'data'))
    store40 = rs_graphs.store_root(P.R10_OUT, 'full', 40, 'aspect') / 'd2/graphs/manifest.json'
    if store40.exists():
        assert C.graph_data('full', 40, tmp_path) == 'data' and seen[-1] == ('full', 40, 'aspect', P.R10_OUT)
    assert C.graph_data('full', 80, tmp_path) == 'data' and seen[-1] == ('full', 80, 'aspect', tmp_path.resolve())
    assert C.graph_data('full', 10, tmp_path) == 'data' and seen[-1][3] == tmp_path.resolve()
    assert all(call[3] != P.R10_OUT or call[1] == 40 for call in seen)           # r10's folder is only ever read, at 40
    with pytest.raises(AssertionError):
        C.graph_data('full', 20, tmp_path)                                         # k = 20 is the CT's own
    with pytest.raises(AssertionError, match='inside'):
        C.graph_data('full', 80, P.R10_OUT / 'graphs')                              # never built into r10's folder


def test_the_graph_gate_must_pass_for_the_code_on_disk(monkeypatch, tmp_path):
    gate = json.loads((P.R10_OUT / 'gates/k11_full.json').read_text())
    monkeypatch.setattr(C.P, 'R10_OUT', tmp_path)
    (tmp_path / 'gates').mkdir()
    for edit in (dict(), dict(pass_=False), dict(code_sha256='0' * 64)):
        g = dict(gate, **{'pass': edit.get('pass_', True)}, code_sha256=edit.get('code_sha256', gate['code_sha256']))
        (tmp_path / 'gates/k11_full.json').write_text(json.dumps(g))
        if edit:
            with pytest.raises(AssertionError):
                C.require_graph_gate()
        else:
            C.require_graph_gate()


# --------------------------------------------------------------------------------------------- training modes
def fake_cohort(monkeypatch, calls, fail=None):
    """`rg_full.train_cohort` replaced by one that writes r10's files for the three heads (as the real saver does)."""
    def fake(s, inputs, graphs, manifests, images, arm, seed, device, root, *, rb, smoke=False, max_epochs=None):
        calls.append((arm, seed, Path(root)))
        if fail:
            return dict(status='failed', cause=fail)
        for k, aspect in enumerate(C.ASPECTS):
            weights, val, _ = C.paths_of(root, arm, aspect, seed)
            weights.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'w': torch.zeros(2)}, weights)
            weights.with_suffix('.json').write_text(json.dumps(dict(
                weights_sha256=ms_train.head.sha_file(weights), best_epoch=3 + k, best_dev_loss=6.0 + k,
                config=dict(rb=rb, rg=rg_full.rg_block(arm, seed, len(s['fit']))))))
            val.parent.mkdir(parents=True, exist_ok=True)
            np.savez(val, img=np.full((len(s['dev']), 2048), k, np.float16), cap=np.full((len(s['dev']), 2048), -k, np.float16))
        return {}
    monkeypatch.setattr(rg_full, 'train_cohort', fake)
    s, inputs, images = synthetic(2048)
    monkeypatch.setattr(C, 'load_data', lambda ctx: (s, inputs, {}, {}, images))
    return s


def files(root):
    return sorted(str(p.relative_to(root)) for p in Path(root).rglob('*') if p.is_file())


def test_keep_all_writes_r10s_layout_and_skips_verified_seeds(monkeypatch, tmp_path):
    calls = []
    fake_cohort(monkeypatch, calls)
    run = C.train('qwen', 'ref', [0], 'cpu', tmp_path)[0]
    assert files(tmp_path) == sorted([
        'train/k11=40/run__s0.json', *(f'weights/k11=40/{a}__s0.{e}' for a in C.ASPECTS for e in ('pt', 'json')),
        *(f'train/k11=40/R6-5/{a}__{r}__s0.npz' for a, r in C.GRAPH_ROWS.items())])
    r = run.record
    assert (r['encoder'], r['name'], r['seed'], r['status'], r['device']) == ('qwen', 'k11=40', 0, 'ok', 'cpu')
    assert r['best_epoch'] == dict(emotion=3, style=4, genre=5) and r['best_dev_loss']['genre'] == 8.0
    assert r['seconds'] >= 0 and r['host'] and r['config']['k11'] == 40 and run.dev is None
    assert json.loads(C.record_path(tmp_path, 'k11=40', 0).read_text()) == r
    again = C.train('qwen', 'ref', [0], 'cpu', tmp_path)[0]
    assert len(calls) == 1 and again.record == r                                    # verified weights: not trained again


def test_keep_none_leaves_only_the_record_and_returns_the_validation_outputs(monkeypatch, tmp_path):
    calls = []
    fake_cohort(monkeypatch, calls)
    run = C.train('qwen', 'k6=0.0005', [1], 'cpu', tmp_path, keep='none')[1]
    assert files(tmp_path) == ['train/k6=0.0005/run__s1.json']                      # no weights, no exports, no scratch
    assert calls[0][2] != tmp_path and not calls[0][2].exists()
    assert list(run.dev) == list(C.ASPECTS) and run.dev_names == ['e', 'f']
    assert (run.dev['genre']['img'] == 2).all() and (run.dev['genre']['cap'] == -2).all() and run.dev['emotion']['img'].shape == (2, 2048)
    assert run.record['config']['k6'] == 0.0005 and run.record['name'] == 'k6=0.0005'
    assert run.record['best_epoch'] == dict(emotion=3, style=4, genre=5)


def test_a_non_finite_loss_raises_and_writes_no_run_record(monkeypatch, tmp_path):
    fake_cohort(monkeypatch, [], fail='non-finite loss in the style head')
    for keep in C.KEEP:
        with pytest.raises(C.CstarNonFiniteLoss, match='style head'):
            C.train('qwen', 'ref', [0], 'cpu', tmp_path, keep=keep)
    assert not list(tmp_path.rglob('run__s*.json')) and not list(tmp_path.rglob('*.pt'))


def test_the_real_cohort_wiring_turns_a_non_finite_loss_into_the_error(monkeypatch, tmp_path, spy):
    def bad(*args, **kwargs):
        raise rg_full.NonFiniteLoss('genre')
    monkeypatch.setattr(rg_full.ct2_d2, 'run', bad)
    s, inputs, images = synthetic(1152)
    monkeypatch.setattr(C, 'load_data', lambda ctx: (s, inputs, {}, {}, images))
    with pytest.raises(C.CstarNonFiniteLoss, match='genre head'):
        C.train('siglip2', 'ref', [0], 'cpu', tmp_path)
    assert (tmp_path / 'train/k11=40/failed__s0.json').exists()


def test_inputs_are_checked(monkeypatch, tmp_path):
    fake_cohort(monkeypatch, [])
    for kwargs in (dict(seeds=[5]), dict(seeds=[0, 0]), dict(seeds=[]), dict(seeds=[0], keep='some'),
                   dict(seeds=[0], max_epochs=5), dict(seeds=[0], smoke=True, max_epochs=5)):
        with pytest.raises(AssertionError):
            C.train('qwen', 'ref', kwargs.pop('seeds'), 'cpu', tmp_path, **kwargs)
    with pytest.raises(AssertionError, match='inside'):
        C.train('qwen', 'ref', [0], 'cpu', P.R10_OUT / 'full')                      # never into r10's folder
    with pytest.raises(AssertionError, match='real folder'):
        C.train('qwen', 'ref', [0], 'cpu', P.DEFAULT_OUT / 'cstar/qwen', smoke=True)
    with pytest.raises(AssertionError):
        C.train('nonesuch', 'ref', [0], 'cpu', tmp_path)


# --------------------------------------------------------------------------------------------------- exports
def make_checkpoint(root, arm, enc, aspect, seed, torch_seed=0):
    width = P.ENCODERS[enc]['width']
    torch.manual_seed(torch_seed)
    _, views = rb_student.build_model(rb_rows.recipe(rb_rows.REFERENCE)['base'], list(C.ASPECTS), CPU, width=width)
    path = rb_student.weights_path(root, arm, aspect, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(views[aspect].state_dict(), path)
    path.with_suffix('.json').write_text(json.dumps(dict(
        weights_sha256=ms_train.head.sha_file(path), best_epoch=1, best_dev_loss=1.0,
        config=dict(aspect=aspect, seed=seed, smoke=False, max_epochs=ms_train.CONFIG['max_epochs'], device='cpu',
                    rb=dict(scale='full', encoder=P.ENCODERS[enc]['art'], width=width, recipe='R6-5'),
                    rg=dict(arm=arm, switches=dict(V.ARMS[rs_knobs.rs_paths.G_ARM]))))))
    return views[aspect]


def unit(rng, n, d):
    x = rng.normal(size=(n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


@pytest.mark.parametrize('enc', ['qwen', 'siglip2', 'siglip1', 'clip_l14'])
def test_held_export_of_a_checkpoint_of_any_width(monkeypatch, tmp_path, enc):
    rng = np.random.default_rng(1)
    width, n = P.ENCODERS[enc]['width'], 600                                         # more than one inference batch
    x = unit(rng, n, width)
    names = [f'p{i}' for i in range(n)]
    monkeypatch.setattr(rb_student, 'held_names', lambda: (names, 'f' * 64))
    monkeypatch.setattr(rb_encoders, 'full_images', lambda b, ns: x if (b, ns) == (P.ENCODERS[enc]['art'], names) else 1 / 0)
    views = {(a, s): make_checkpoint(tmp_path / 'w', 'k11=40', enc, a, s, torch_seed=7 * s + i)
             for s in (0, 1) for i, a in enumerate(C.ASPECTS)}
    out = C.export_held(enc, 'ref', [0, 1], tmp_path / 'w', out_root=tmp_path / 'o')
    assert sorted(out) == sorted((a, s) for a in C.ASPECTS for s in (0, 1))
    for (aspect, seed), arrays in out.items():
        views[aspect, seed].eval()
        with torch.no_grad():
            want = views[aspect, seed]['img'](torch.from_numpy(x)).numpy()
        got = arrays['img']
        assert got.shape == (n, 2048) and got.dtype == np.float16
        assert np.abs(got.astype(np.float32) - want).max() < 2e-3
        path = tmp_path / 'o/held/k11=40' / f'{aspect}__s{seed}.npz'
        read, info = rb_student.read_outputs(path)
        assert (info['kind'], info['arm'], info['aspect'], info['seed'], info['width'], info['rows'], info['list_sha256']) == (
            'held', 'k11=40', aspect, seed, 2048, n, 'f' * 64)
        assert info['sha256'] == rb_student.sha_array(got) and info['encoder'] == P.ENCODERS[enc]['art']
        assert info['vc3_attempts'] == 1 and info['vc3_problems'] == []                    # r10's readers index these
        if R10_HELD.exists():                                                              # r10's own held sidecar: same keys, same types
            theirs = json.loads(R10_HELD.read_text())
            assert set(info) - set(theirs) == {'export_device'} and set(theirs) <= set(info)
            for key in ('vc3_attempts', 'vc3_problems', 'kind', 'rows', 'width', 'dtype', 'switches', 'recipe'):
                assert type(info[key]) is type(theirs[key]), key
    assert files(tmp_path / 'w') == sorted(files(tmp_path / 'w'))                    # the weights folder got no export
    assert not list((tmp_path / 'w').rglob('*.npz'))
    again = C.export_held(enc, 'ref', [0], tmp_path / 'w', out_root=tmp_path / 'o')   # an existing file must match
    assert np.array_equal(again['emotion', 0]['img'], out['emotion', 0]['img'])


def test_export_refuses_wrong_width_changed_weights_and_r10s_folder(monkeypatch, tmp_path):
    rng = np.random.default_rng(2)
    names = [f'p{i}' for i in range(5)]
    monkeypatch.setattr(rb_student, 'held_names', lambda: (names, 'f' * 64))
    monkeypatch.setattr(rb_encoders, 'full_images', lambda b, ns: unit(rng, 5, 768))   # a CLIP L/14 store for a SigLIP checkpoint
    make_checkpoint(tmp_path / 'w', 'k11=40', 'siglip2', 'emotion', 0)
    with pytest.raises(AssertionError):
        C.export_held('siglip2', 'ref', [0], tmp_path / 'w', out_root=tmp_path / 'o')
    with pytest.raises(AssertionError, match='inside'):
        C.export_held('siglip2', 'ref', [0], tmp_path / 'w', out_root=P.R10_OUT / 'full')
    # a checkpoint of another encoder than asked for
    make_checkpoint(tmp_path / 'w2', 'k11=40', 'clip_l14', 'emotion', 0)
    monkeypatch.setattr(rb_encoders, 'full_images', lambda b, ns: unit(rng, 5, 1152))
    with pytest.raises(AssertionError):
        C.export_held('siglip2', 'ref', [0], tmp_path / 'w2', out_root=tmp_path / 'o')
    # weights changed after the sidecar was written
    make_checkpoint(tmp_path / 'w3', 'k11=40', 'siglip2', 'emotion', 0)
    path = rb_student.weights_path(tmp_path / 'w3', 'k11=40', 'emotion', 0)
    torch.save({k: v + 1 for k, v in torch.load(path, weights_only=True).items()}, path)
    with pytest.raises(AssertionError, match='checksum'):
        C.export_held('siglip2', 'ref', [0], tmp_path / 'w3', out_root=tmp_path / 'o')


@pytest.mark.parametrize('enc', ['qwen', 'siglip1'])
def test_validation_export_has_both_sides(monkeypatch, tmp_path, enc):
    rng = np.random.default_rng(3)
    width, n = P.ENCODERS[enc]['width'], 7
    img, cap = unit(rng, n, width), unit(rng, n, 2048)
    names = [f'q{i}' for i in range(n)]
    monkeypatch.setattr(C, '_validation_inputs', lambda b: (names, img, cap))
    views = {a: make_checkpoint(tmp_path / 'w', 'k11=40', enc, a, 2, torch_seed=i) for i, a in enumerate(C.ASPECTS)}
    out = C.export_validation(enc, 'ref', [2], tmp_path / 'w', out_root=tmp_path / 'o')
    for aspect in C.ASPECTS:
        views[aspect].eval()
        with torch.no_grad():
            want = {'img': views[aspect]['img'](torch.from_numpy(img)).numpy(), 'cap': views[aspect]['cap'](torch.from_numpy(cap)).numpy()}
        arrays, info = rb_student.read_outputs(tmp_path / 'o/val/k11=40' / f'{aspect}__s2.npz')
        assert sorted(arrays) == ['cap', 'img'] and info['kind'] == 'val' and info['rows'] == n
        for side in ('img', 'cap'):
            assert arrays[side].shape == (n, 2048) and np.abs(arrays[side].astype(np.float32) - want[side]).max() < 2e-3
            assert np.array_equal(out[aspect, 2][side], arrays[side])
    assert not (tmp_path / 'o/held').exists()


# ------------------------------------------------------------------------------------ stage 0 comparisons
def sidecar(root, arm, aspect, seed, epoch, loss):
    path = rb_student.weights_path(root, arm, aspect, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix('.json').write_text(json.dumps(dict(best_epoch=epoch, best_dev_loss=loss)))


@pytest.mark.parametrize('style_epoch,delta,ok', [(1, 0.0, True), (1, 9e-5, True), (1, -9e-5, True), (1, 2e-4, False),
                                                   (2, 0.0, False), (0, 5e-5, False)])
def test_0iv_needs_the_same_best_epoch_and_a_loss_within_1e4(tmp_path, style_epoch, delta, ok):
    for aspect, e, loss in (('emotion', 4, 6.16), ('style', 1, 6.53), ('genre', 18, 6.88)):
        sidecar(tmp_path / 'ref', 'k11=40', aspect, 0, e, loss)
        sidecar(tmp_path / 'ours', 'k11=40', aspect, 0, e, loss)
    sidecar(tmp_path / 'ours', 'k11=40', 'style', 0, style_epoch, 6.53 + delta)
    record = C.compare_training(tmp_path / 'ours', tmp_path / 'ref')
    assert record['pass'] is ok and record['items']['emotion']['pass'] and record['items']['genre']['pass']
    assert record['items']['style']['pass'] is ok


def test_0ix_compares_held_and_both_validation_sides_within_1e3(tmp_path):
    def build(shift_cap=0.0, shift_held=0.0):
        for seed in (0,):
            for aspect in C.ASPECTS:
                rng = np.random.default_rng(5)
                img, cap = (rng.uniform(-.2, .2, size=(4, 8)).astype(np.float16) for _ in range(2))
                held_theirs, val_theirs = (tmp_path / 'ref/held/k11=40' / f'{aspect}__s{seed}.npz',
                                           tmp_path / 'ref/train/k11=40/R6-5' / f'{aspect}__{C.GRAPH_ROWS[aspect]}__s{seed}.npz')
                held_ours, val_ours = (tmp_path / 'ours/held/k11=40' / f'{aspect}__s{seed}.npz',
                                       tmp_path / 'ours/val/k11=40' / f'{aspect}__s{seed}.npz')
                for p in (held_theirs, val_theirs, held_ours, val_ours):
                    p.parent.mkdir(parents=True, exist_ok=True)
                np.savez(held_theirs, img=img)
                np.savez(val_theirs, img=img, cap=cap)
                np.savez(held_ours, img=img + np.float16(shift_held))
                np.savez(val_ours, img=img, cap=cap + np.float16(shift_cap))
        return C.compare_exports(tmp_path / 'ours', tmp_path / 'ref', seeds=[0])
    assert build()['pass']
    assert build(shift_cap=0.0005, shift_held=0.0005)['pass']
    bad_cap = build(shift_cap=0.004)
    assert not bad_cap['pass'] and not bad_cap['items']['val/emotion__s0/cap']['pass'] and bad_cap['items']['val/emotion__s0/img']['pass']
    bad_held = build(shift_held=0.004)
    assert not bad_held['pass'] and not bad_held['items']['held/genre__s0/img']['pass']


@pytest.mark.parametrize('ours_shape,ours_dtype', [((3, 8), np.float16), ((4, 8), np.float32), ((1, 8), np.float16)])
def test_0ix_fails_on_another_shape_or_dtype_and_does_not_broadcast(tmp_path, ours_shape, ours_dtype):
    for aspect in C.ASPECTS:
        held_theirs = tmp_path / 'ref/held/k11=40' / f'{aspect}__s0.npz'
        val_theirs = tmp_path / 'ref/train/k11=40/R6-5' / f'{aspect}__{C.GRAPH_ROWS[aspect]}__s0.npz'
        held_ours, val_ours = (tmp_path / 'ours' / d / 'k11=40' / f'{aspect}__s0.npz' for d in ('held', 'val'))
        for p in (held_theirs, val_theirs, held_ours, val_ours):
            p.parent.mkdir(parents=True, exist_ok=True)
        ref = np.zeros((4, 8), np.float16)
        mine = np.zeros(ours_shape, ours_dtype)
        np.savez(held_theirs, img=ref)
        np.savez(val_theirs, img=ref, cap=ref)
        np.savez(held_ours, img=ref)
        np.savez(val_ours, img=ref, cap=mine)
    record = C.compare_exports(tmp_path / 'ours', tmp_path / 'ref', seeds=[0])
    assert not record['pass'] and not record['items']['val/style__s0/cap']['pass'] and record['items']['val/style__s0/img']['pass']


# ------------------------------------------------------------------------------------ review fixes
def test_a_knob_rows_arm_carries_the_rows_lambda_and_teacher_temperature():
    """`rs_full.arm_alias` copies G's arm: entered before the knob patches it would give the row G's lam and tau_t."""
    base = V.switches(rs_knobs.rs_paths.G_ARM)
    with C.training_context('qwen', 'k3=4'):
        assert V.switches(C.row_arm('k3=4'))['lam'] == 4 and V.switches(C.row_arm('k3=4'))['tau_t'] == base['tau_t']
    with C.training_context('qwen', 'k4=0.035'):
        assert V.switches(C.row_arm('k4=0.035'))['tau_t'] == 0.035 and V.switches(C.row_arm('k4=0.035'))['lam'] == base['lam']
    assert V.switches(rs_knobs.rs_paths.G_ARM) == base and 'k3=4' not in V.ARMS


def test_the_run_record_names_its_keep_mode_and_a_record_of_the_other_mode_is_not_skipped(monkeypatch, tmp_path):
    calls = []
    fake_cohort(monkeypatch, calls)
    assert C.train('qwen', 'ref', [1], 'cpu', tmp_path, keep='all')[1].record['keep'] == 'all'
    assert C.train('qwen', 'ref', [1], 'cpu', tmp_path, keep='none')[1].record['keep'] == 'none'
    assert json.loads(C.record_path(tmp_path, 'k11=40', 1).read_text())['keep'] == 'none'     # (it overwrote: the CLI refuses this root)
    with pytest.raises(AssertionError, match='another run'):                                    # keep-all finds the keep-none record
        C.train('qwen', 'ref', [1], 'cpu', tmp_path, keep='all')


def test_the_command_line_needs_a_root_for_keep_none_and_refuses_the_default_roots(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(C, 'train', lambda enc, row, seeds, device, root, **kw: seen.append((enc, row, root, kw['keep'])) or {})
    argv = lambda *a: ['train', '--encoder', 'qwen', *a]
    with pytest.raises(AssertionError, match='--root'):
        C.main(argv('--keep', 'none'))
    for enc in P.ENCODERS:                                                   # the default keep-all root of any encoder
        for root in (C.default_root(enc), C.default_root(enc) / '.' / 'x' / '..'):
            with pytest.raises(AssertionError, match='keep-all root'):
                C.main(argv('--keep', 'none', '--root', str(root)))
    assert not seen
    assert C.main(argv('--keep', 'none', '--root', str(tmp_path / 'sweep'))) == 0 and seen[-1][2] == tmp_path / 'sweep'
    assert C.main(argv('--keep', 'all')) == 0 and seen[-1] == ('qwen', 'ref', C.default_root('qwen'), 'all')   # keep-all: default root
    assert C.main(argv('--row', 'k6=0.0005', '--keep', 'none', '--root', str(tmp_path / 'sweep'))) == 0


def test_the_code_state_is_read_before_training_and_is_the_one_in_the_record(monkeypatch, tmp_path):
    calls = []
    fake_cohort(monkeypatch, calls)
    def broken():
        raise RuntimeError('git failed')
    monkeypatch.setattr(C, 'code_state', broken)
    with pytest.raises(RuntimeError, match='git failed'):
        C.train('qwen', 'ref', [0], 'cpu', tmp_path)
    assert not calls and not list(tmp_path.rglob('*'))                          # nothing trained, nothing to clean up after
    states = iter([('c0ffee', True), ('other', False)])
    monkeypatch.setattr(C, 'code_state', lambda: next(states))
    record = C.train('qwen', 'ref', [0, 1], 'cpu', tmp_path)
    assert (record[0].record['code_commit'], record[0].record['code_dirty']) == ('c0ffee', True)
    assert record[1].record['code_commit'] == 'c0ffee'                          # read once for the call, not per seed


def test_train_checks_the_imports_as_r10s_trainer_does(monkeypatch, tmp_path):
    fake_cohort(monkeypatch, [])
    def refused():
        raise AssertionError('rs_x: imported from another folder')
    monkeypatch.setattr(C.rs_paths, 'check_imports', refused)
    with pytest.raises(AssertionError, match='imported from another folder'):
        C.train('siglip2', 'ref', [0], 'cpu', tmp_path)


@pytest.mark.parametrize('field,other', [('encoder', 'siglip2'), ('row', 'k6=0.0005'), ('seed', 2), ('smoke', True),
                                         ('keep', 'none'), ('config', dict(k11=80))])
def test_verified_weights_are_skipped_only_with_this_runs_record(monkeypatch, tmp_path, field, other):
    calls = []
    fake_cohort(monkeypatch, calls)
    run = C.train('qwen', 'ref', [0], 'cpu', tmp_path)[0]
    assert len(calls) == 1
    assert C.train('qwen', 'ref', [0], 'cpu', tmp_path)[0].record == run.record and len(calls) == 1      # matching record: skipped
    path = C.record_path(tmp_path, 'k11=40', 0)
    path.write_text(json.dumps(dict(run.record, **{field: other})))                                       # a record of another run
    with pytest.raises(AssertionError, match='another run'):
        C.train('qwen', 'ref', [0], 'cpu', tmp_path)
    assert len(calls) == 1                                                                                 # and nothing was retrained
    path.unlink()
    with pytest.raises(AssertionError, match='without a run record'):
        C.train('qwen', 'ref', [0], 'cpu', tmp_path)
    assert len(calls) == 1
