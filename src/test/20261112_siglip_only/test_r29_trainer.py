"""Tests of r29_trainer (ticket 03): gate (i)(b) on a fixed batch through r14's real train path, the caption input
substitution, sidecar and record fields, the export code against r14's, the comparisons of gate (i), the guards."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r29_trainer as T  # noqa: E402

B, P, C, M = T.B, T.P, T.C, T.M
RG, ms_train = B.rg_full, B.ms_train
UNIT = P.SMOKE_OUT / 'unit-no-write'          # nothing is written under it (every writer is stubbed)
N = 6


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    torch.set_num_threads(2)
    monkeypatch.setattr(B, 'code_state', lambda: ('test-commit', False))
    monkeypatch.setattr(B.eg, 'atomic_record', lambda path, record: None)
    monkeypatch.setattr(B, '_record', lambda *a, **k: dict(status='ok', seconds=0.0, best_epoch={}))
    monkeypatch.setattr(C, 'store_hash', lambda source: f'hash-{source}')


def unit(x):
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def data(img_width=1152):
    rng = np.random.default_rng(5)
    s = dict(first=np.arange(N), fit=np.array([0, 1, 2]), dev=np.array([3, 4, 5]),
             all_painting_ids=np.array([f'p{i}' for i in range(N)]))
    img, cap = unit(rng.normal(size=(N, img_width))), unit(rng.normal(size=(N, 2048)))
    inputs = dict.fromkeys(T.ASPECTS, dict(img=img, cap=cap))
    return s, inputs, img


def siglip_means(width=1152):
    return unit(np.random.default_rng(6).normal(size=(N, width)))


def fixed_batch_run(log):
    """Stands for `ct2_d2.run`: builds the model the way the loop does (`ms_train.build_model`, patched by
    `rg_full.train_cohort`), then evaluates the real patched loss on a fixed batch of the inputs it was handed."""
    def run(s, inputs, graphs, manifests, row, name, aspects, seed, device, outdir, smoke=False, max_epochs=200):
        torch.manual_seed(seed)
        models, views = ms_train.build_model(row, aspects, device)
        log['state'] = {k: v.detach().clone() for k, v in models.state_dict().items()}
        log['inputs'] = {a: {side: x.copy() for side, x in inputs[a].items()} for a in aspects}
        log['loss'], log['grads'] = {}, {}
        teacher = torch.nn.functional.normalize(torch.randn(N, 2048, generator=torch.Generator().manual_seed(3)), dim=1)
        pairs, weights = np.array([[0, 1], [2, 3], [4, 5]], dtype=np.int64), np.array([1.0, 0.7, 0.3], dtype=np.float32)
        for aspect in aspects:
            models.zero_grad(set_to_none=True)
            tensors = {side: torch.from_numpy(x) for side, x in inputs[aspect].items()}
            torch.manual_seed(97)
            loss = ms_train.head_loss(views[aspect], **tensors, pairs=pairs, weights=weights, row=row, aspect=aspect,
                                      teacher=teacher)
            loss.backward()
            log['loss'][aspect] = loss.detach().clone()
            log['grads'][aspect] = {k: p.grad.detach().clone() if p.grad is not None else None
                                    for k, p in models.named_parameters()}
        return {}
    return run


def run_path(monkeypatch, source=None):
    """One seed through B.train: r14's own (source None) or this module's, with the fixed-batch run and stub data."""
    log = {}
    s, inputs, img = data()
    monkeypatch.setattr(B, 'load_data', lambda ctx: (s, inputs, {}, {}, img))
    monkeypatch.setattr(T.ct2_d2, 'run', fixed_batch_run(log))
    if source is None:
        B.train('siglip1', 'ref', [0], 'cpu', UNIT, smoke=True)
    else:
        T.train('siglip1', source, [0], 'cpu', UNIT, smoke=True)
    return log


def test_qwen_source_equals_r14_state_loss_and_gradients_on_a_fixed_batch(monkeypatch):
    expected = run_path(monkeypatch, None)
    actual = run_path(monkeypatch, 'qwen')
    assert actual['state'].keys() == expected['state'].keys()
    for k, v in actual['state'].items():
        assert torch.equal(v, expected['state'][k]), k
    for aspect in T.ASPECTS:
        assert torch.equal(actual['loss'][aspect], expected['loss'][aspect])
        for k, g in actual['grads'][aspect].items():
            if g is None:
                assert expected['grads'][aspect][k] is None
            else:
                assert torch.equal(g, expected['grads'][aspect][k]), (aspect, k)
        for side in ('img', 'cap'):
            assert np.array_equal(actual['inputs'][aspect][side], expected['inputs'][aspect][side])
    assert actual['state']['cap.trunk.0.weight'].shape == (1024, 2048)


def test_siglip_source_changes_only_the_caption_input_and_its_trunk_layer(monkeypatch):
    expected = run_path(monkeypatch, None)
    monkeypatch.setattr(C, 'painting_means', lambda source, names: siglip_means())
    actual = run_path(monkeypatch, 'siglip1')
    for aspect in T.ASPECTS:
        assert np.array_equal(actual['inputs'][aspect]['img'], expected['inputs'][aspect]['img'])
        assert np.array_equal(actual['inputs'][aspect]['cap'], siglip_means())
        assert actual['inputs'][aspect]['cap'].shape == (N, 1152)
    assert actual['state'].keys() == expected['state'].keys()
    for k, v in actual['state'].items():
        want = (1024, 1152) if k == 'cap.trunk.0.weight' else expected['state'][k].shape
        assert v.shape == want, k
    assert all(torch.isfinite(v) for v in actual['loss'].values())
    # the image side and the heads are built from the same seed in the same order: the image trunk is r14's
    assert torch.equal(actual['state']['img.trunk.0.weight'], expected['state']['img.trunk.0.weight'])


def test_painting_means_are_asked_in_the_cts_painting_order(monkeypatch):
    asked = []
    monkeypatch.setattr(C, 'painting_means', lambda source, names: asked.append((source, list(names))) or siglip_means())
    run_path(monkeypatch, 'siglip1')
    assert asked == [('siglip1', [f'p{i}' for i in range(N)])]


def test_sidecar_rb_block_and_run_record_carry_the_caption_fields(monkeypatch):
    seen = {}
    s, inputs, img = data()
    monkeypatch.setattr(B, 'load_data', lambda ctx: (s, inputs, {}, {}, img))
    monkeypatch.setattr(C, 'painting_means', lambda source, names: siglip_means())

    def cohort(s, inputs, graphs, manifests, images, arm, seed, device, root, *, rb, **kw):
        seen.update(rb=rb, arm=arm, v_arm=B.V.switches(arm))
        return {}

    monkeypatch.setattr(RG, 'train_cohort', cohort)
    monkeypatch.setattr(B, '_record', lambda *a, **k: dict(status='ok', seconds=0.0))
    records = []
    monkeypatch.setattr(B.eg, 'atomic_record', lambda path, record: records.append((path, record)))
    runs = T.train('siglip1', 'siglip1', [0], 'cpu', UNIT, smoke=True)
    rb = seen['rb']
    assert (rb['cap_source'], rb['cap_width'], rb['cap_store_sha256']) == ('siglip1', 1152, 'hash-siglip1')
    assert rb['cap_sha256'] == B.rb_student.sha_array(siglip_means()) and rb['r29_code_sha256'] == T.code_hashes()
    assert rb['width'] == 1152 and rb['encoder'] == 'siglip_so400m'                # r14's own fields stay
    assert seen['arm'] == 'siglip1' and seen['v_arm'] == dict(B.V.ARMS[B.rs_full.G_ARM])
    record = runs[0].record
    assert (record['cap_source'], record['cap_width'], record['cap_store_sha256']) == ('siglip1', 1152, 'hash-siglip1')
    assert record['r29_code_sha256'] == T.code_hashes() and set(T.code_hashes()) == set(T.CODE_FILES)
    assert records[0][0] == UNIT / 'train' / 'siglip1' / 'run__s0.json'


def test_qwen_rb_block_keeps_the_cts_own_caption_hash(monkeypatch):
    seen = {}
    s, inputs, img = data()
    monkeypatch.setattr(B, 'load_data', lambda ctx: (s, inputs, {}, {}, img))
    monkeypatch.setattr(RG, 'train_cohort', lambda *a, rb, **k: seen.update(rb=rb) or {})
    T.train('siglip1', 'qwen', [0], 'cpu', UNIT, smoke=True)
    assert seen['rb']['cap_width'] == 2048 and seen['rb']['cap_sha256'] == B.rb_student.sha_array(inputs['emotion']['cap'])


def test_non_finite_outcome_raises(monkeypatch):
    s, inputs, img = data()
    monkeypatch.setattr(B, 'load_data', lambda ctx: (s, inputs, {}, {}, img))
    monkeypatch.setattr(RG, 'train_cohort', lambda *a, **k: dict(status='failed', cause='non-finite loss in the genre head'))
    with pytest.raises(B.CstarNonFiniteLoss, match='non-finite'):
        T.train('siglip1', 'qwen', [0], 'cpu', UNIT, smoke=True)


def test_patches_unwind_on_failure(monkeypatch):
    before = (B.load_data, B.row_arm, B._record, B.rb_student._rb_block, B.rb_student.build_model, T.ct2_d2.run)
    with pytest.raises(RuntimeError, match='unwind'):
        with T.caption_patches('siglip1', 1152):
            assert B.rb_student.build_model is not before[4] and B.row_arm('ref') == 'siglip1'
            raise RuntimeError('unwind')
    assert (B.load_data, B.row_arm, B._record, B.rb_student._rb_block, B.rb_student.build_model, T.ct2_d2.run) == before
    with T.caption_patches('qwen', 2048):
        assert B.rb_student.build_model is before[4]          # r14's own model objects at 2048


def test_root_and_stage_guards(monkeypatch, tmp_path):
    monkeypatch.setattr(P, 'OUT', tmp_path / 'out')
    monkeypatch.setattr(P, 'SMOKE_OUT', tmp_path / 'out' / 'smoke')
    monkeypatch.setattr(P, 'RECORDS', tmp_path / 'records')
    assert T.check_root(P.SMOKE_OUT / 'a', True) == (P.SMOKE_OUT / 'a').resolve()
    assert T.check_root(P.OUT / 'cstar' / 'siglip1') == (P.OUT / 'cstar' / 'siglip1').resolve()
    for root, smoke in ((P.OUT / 'cstar', True), (P.SMOKE_OUT / 'a', False), (tmp_path / 'elsewhere', False),
                        (B.P.R10_OUT / 'x', False)):
        with pytest.raises(AssertionError):
            T.check_root(root, smoke)
    with pytest.raises(AssertionError, match='stage 2'):
        T.train('siglip2', 'siglip2', [0], 'cpu', P.OUT / 'cstar' / 'siglip2')
    (tmp_path / 'records').mkdir()
    (tmp_path / 'records' / 'pass_rule.json').write_text(json.dumps(dict(status='fail')))
    with pytest.raises(AssertionError, match='stage 2'):
        T.require_pass('clip_l14')
    (tmp_path / 'records' / 'pass_rule.json').write_text(json.dumps(dict(status='pass')))
    T.require_pass('clip_l14')
    T.require_pass('siglip1')
    with pytest.raises(AssertionError):
        T.check_args('qwen', 'siglip1')
    with pytest.raises(AssertionError):
        T.check_args('siglip1', 'nope')


def test_train_refuses_records_of_other_code(monkeypatch, tmp_path):
    def no_real_data(ctx):
        pytest.fail('code guard must reject the record before loading data')

    monkeypatch.setattr(B, 'load_data', no_real_data)
    monkeypatch.setattr(P, 'OUT', tmp_path)
    root = tmp_path / 'cstar' / 'siglip1'
    path = B.record_path(root, 'siglip1', 0)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(dict(r29_code_sha256={'r29_trainer.py': 'old'})))
    with pytest.raises(AssertionError, match='code changed'):
        T.train('siglip1', 'siglip1', [0], 'cpu', root)


@pytest.mark.parametrize('command', ['train', 'export'])
@pytest.mark.parametrize('encoder,source', [
    ('siglip1', 'siglip2'), ('siglip1', 'clipl14_prefix'),
    ('siglip2', 'siglip1'), ('siglip2', 'clipl14'),
    ('clip_l14', 'siglip2'), ('clip_l14', 'siglip1_prefix'),
])
def test_commands_refuse_caption_source_of_another_encoder(monkeypatch, tmp_path, command, encoder, source):
    monkeypatch.setattr(P, 'OUT', tmp_path)
    monkeypatch.setattr(P, 'RECORDS', tmp_path / 'records')
    P.RECORDS.mkdir()
    (P.RECORDS / 'pass_rule.json').write_text(json.dumps(dict(status='pass')))
    monkeypatch.setattr(B, 'training_context', lambda *a, **k: pytest.fail('must reject before loading data'))
    with pytest.raises(AssertionError, match='caption source.*encoder'):
        if command == 'train':
            T.train(encoder, source, [0], 'cpu', tmp_path / 'model')
        else:
            T.export(encoder, source, [0], tmp_path / 'model')


# ------------------------------------------------------------------------------------------------- the exports
ARM_SWITCHES = dict(B.V.ARMS[B.rs_full.G_ARM])


def save_head(root, arm, aspect, seed, cap_width, rb_extra, *, width=1152, seed_model=1):
    torch.manual_seed(seed_model)
    _, views = M.build_model(list(T.ASPECTS), torch.device('cpu'), width=width, cap_width=cap_width)
    path = B.rb_student.weights_path(root, arm, aspect, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(views[aspect].state_dict(), path)
    rb = dict(scale='full', encoder='siglip_so400m', width=width, recipe=RG.RECIPE, **rb_extra)
    config = dict(aspect=aspect, seed=seed, smoke=True, max_epochs=2, device='cpu', rb=rb,
                  rg=dict(arm=arm, switches=ARM_SWITCHES))
    path.with_suffix('.json').write_text(json.dumps(dict(weights_sha256=ms_train.head.sha_file(path), config=config,
                                                         best_epoch=1, best_dev_loss=0.5)))
    return path


def stub_inputs(monkeypatch, n=700, cap_width=1152):
    rng = np.random.default_rng(8)
    names = [f'q{i}' for i in range(n)]
    img = unit(rng.normal(size=(n, 1152)))
    cap = unit(rng.normal(size=(n, cap_width)))
    monkeypatch.setattr(T, 'held_inputs', lambda backbone: (names, 'sha-held', dict(img=img)))
    monkeypatch.setattr(T, 'validation_inputs', lambda source, backbone: (names, 'sha-val', dict(img=img, cap=cap)))
    return names, img, cap


def test_export_rebuilds_each_view_at_its_recorded_widths(monkeypatch, tmp_path):
    monkeypatch.setattr(P, 'SMOKE_OUT', tmp_path / 'smoke')
    names, img, cap = stub_inputs(monkeypatch)
    fields = dict(cap_source='siglip1', cap_width=1152, cap_store_sha256='hash-siglip1', cap_sha256='x')
    w = {a: save_head(tmp_path / 'w', 'siglip1', a, 0, 1152, fields) for a in T.ASPECTS}
    out = T.export('siglip1', 'siglip1', [0], tmp_path / 'w', kinds=('val', 'held'), out_root=P.SMOKE_OUT / 'o', smoke=True)
    for aspect in T.ASPECTS:
        view = M.load_view(w[aspect], expect_store_sha256='hash-siglip1')
        with torch.no_grad():
            want = {side: view[side](torch.from_numpy(x)).numpy().astype(np.float16)
                    for side, x in (('img', img), ('cap', cap))}
        got, info = B.rb_student.read_outputs(P.SMOKE_OUT / 'o' / 'val' / 'siglip1' / f'{aspect}__s0.npz')
        assert np.array_equal(got['img'], want['img']) and np.array_equal(got['cap'], want['cap'])
        assert (info['cap_source'], info['cap_width'], info['cap_store_sha256']) == ('siglip1', 1152, 'hash-siglip1')
        assert info['list_sha256'] == 'sha-val' and info['cap_sha256'] == B.rb_student.sha_array(cap)
        held, hinfo = B.rb_student.read_outputs(P.SMOKE_OUT / 'o' / 'held' / 'siglip1' / f'{aspect}__s0.npz')
        assert sorted(held) == ['img'] and np.array_equal(held['img'], want['img']) and hinfo['rows'] == len(names)
    assert out[('genre', 0)].keys() == {'img'}


def test_export_refuses_other_caption_source_or_store(monkeypatch, tmp_path):
    monkeypatch.setattr(P, 'SMOKE_OUT', tmp_path / 'smoke')
    stub_inputs(monkeypatch)
    fields = dict(cap_source='siglip1', cap_width=1152, cap_store_sha256='hash-OLD', cap_sha256='x')
    for a in T.ASPECTS:
        save_head(tmp_path / 'w', 'siglip1', a, 0, 1152, fields)
    with pytest.raises(AssertionError, match='another caption source or store'):
        T.export('siglip1', 'siglip1', [0], tmp_path / 'w', kinds=('held',), out_root=P.SMOKE_OUT / 'o', smoke=True)


@pytest.mark.parametrize('change', ['missing_hash', 'changed_code'])
def test_export_reuses_only_outputs_of_current_code(monkeypatch, tmp_path, change):
    monkeypatch.setattr(P, 'SMOKE_OUT', tmp_path / 'smoke')
    stub_inputs(monkeypatch, n=5)
    root = P.SMOKE_OUT / 'model'
    fields = dict(cap_source='siglip1', cap_width=1152, cap_store_sha256='hash-siglip1')
    for aspect in T.ASPECTS:
        save_head(root, 'siglip1', aspect, 0, 1152, fields)
    T.export('siglip1', 'siglip1', [0], root, kinds=('held',), smoke=True)
    path = root / 'held' / 'siglip1' / 'emotion__s0.json'
    info = json.loads(path.read_text())
    assert info['r29_code_sha256'] == T.code_hashes()
    monkeypatch.setattr(T, 'infer_sides', lambda *a: pytest.fail('existing output must be reused or refused'))
    T.export('siglip1', 'siglip1', [0], root, kinds=('held',), smoke=True)
    if change == 'missing_hash':
        info.pop('r29_code_sha256')
        path.write_text(json.dumps(info))
    else:
        monkeypatch.setattr(T, 'code_hashes', lambda: dict(info['r29_code_sha256'], **{'r29_trainer.py': 'changed'}))
    with pytest.raises(AssertionError, match='code changed|code hash'):
        T.export('siglip1', 'siglip1', [0], root, kinds=('held',), smoke=True)


def validation_model(monkeypatch, tmp_path):
    monkeypatch.setattr(P, 'SMOKE_OUT', tmp_path / 'smoke')
    stub_inputs(monkeypatch, n=5)
    root = P.SMOKE_OUT / 'model'
    fields = dict(cap_source='siglip1', cap_width=1152, cap_store_sha256='hash-siglip1')
    arrays = {side: np.zeros((5, 2048), dtype=np.float16) for side in ('img', 'cap')}
    arrays['img'][:, 0], arrays['cap'][:, 1] = 1, 1
    for aspect in T.ASPECTS:
        save_head(root, 'siglip1', aspect, 0, 1152, fields)
        write_export(root, 'val', 'siglip1', aspect, 0, arrays, layout='train')
    B.record_path(root, 'siglip1', 0).write_text(json.dumps(dict(device='cpu')))
    devices = []
    def infer(weights, inputs, device):
        devices.append(str(device))
        return {side: x.copy() for side, x in arrays.items()}

    monkeypatch.setattr(T, 'infer_sides', infer)
    return root, arrays, devices


def test_own_root_validation_uses_run_device_and_equals_training_arrays(monkeypatch, tmp_path):
    root, arrays, devices = validation_model(monkeypatch, tmp_path)
    T.export('siglip1', 'siglip1', [0], root, kinds=('val',), smoke=True)
    assert devices == ['cpu'] * 3
    for aspect in T.ASPECTS:
        got, info = B.rb_student.read_outputs(root / 'val' / 'siglip1' / f'{aspect}__s0.npz')
        assert info['export_device'] == 'cpu'
        assert all(got[side].tobytes() == x.tobytes() for side, x in arrays.items())
    monkeypatch.setattr(T, 'infer_sides', lambda *a: pytest.fail('verified export should be reused'))
    T.export('siglip1', 'siglip1', [0], root, kinds=('val',), device='cpu', smoke=True)


@pytest.mark.parametrize('mismatch', ['img', 'cap', 'signed_zero', 'reused'])
def test_own_root_validation_refuses_arrays_different_from_training(monkeypatch, tmp_path, mismatch):
    root, arrays, devices = validation_model(monkeypatch, tmp_path)
    if mismatch == 'reused':
        T.export('siglip1', 'siglip1', [0], root, kinds=('val',), smoke=True)
    changed = {side: x.copy() for side, x in arrays.items()}
    if mismatch == 'signed_zero':
        changed['cap'][0, 2] = np.float16(-0.0)
    else:
        changed['img' if mismatch == 'img' else 'cap'][0, 0] += np.float16(0.01)
    write_export(root, 'val', 'siglip1', 'emotion', 0, changed, layout='train')
    with pytest.raises(AssertionError, match='training-time validation'):
        T.export('siglip1', 'siglip1', [0], root, kinds=('val',), smoke=True)
    if mismatch != 'reused':
        assert not (root / 'val' / 'siglip1' / 'emotion__s0.npz').exists()


@pytest.mark.parametrize('mismatch', ['override', 'run_record'])
def test_own_root_validation_refuses_device_different_from_run(monkeypatch, tmp_path, mismatch):
    root, arrays, devices = validation_model(monkeypatch, tmp_path)
    if mismatch == 'run_record':
        B.record_path(root, 'siglip1', 0).write_text(json.dumps(dict(device='cuda:7')))
    with pytest.raises(AssertionError, match='recorded training device'):
        T.export('siglip1', 'siglip1', [0], root, kinds=('val',), smoke=True,
                 device='cuda:7' if mismatch == 'override' else None)
    assert devices == []


def test_k0_sidecar_without_caption_fields_is_qwen_at_2048_only_under_k0_root(monkeypatch, tmp_path):
    monkeypatch.setattr(P, 'SMOKE_OUT', tmp_path / 'smoke')
    monkeypatch.setattr(P, 'K0_ROOT', tmp_path / 'k0')
    names, img, cap = stub_inputs(monkeypatch, cap_width=2048)
    for a in T.ASPECTS:
        save_head(tmp_path / 'k0', 'k11=40', a, 0, 2048, {})
        save_head(tmp_path / 'other', 'k11=40', a, 0, 2048, {})
    T.export('siglip1', 'qwen', [0], tmp_path / 'k0', kinds=('val',), out_root=P.SMOKE_OUT / 'o', smoke=True, arm='k11=40')
    got, _ = B.rb_student.read_outputs(P.SMOKE_OUT / 'o' / 'val' / 'k11=40' / 'emotion__s0.npz')
    assert got['cap'].shape == (len(names), 2048)
    with pytest.raises(AssertionError, match='lacks the caption fields'):
        T.export('siglip1', 'qwen', [0], tmp_path / 'other', kinds=('val',), out_root=P.SMOKE_OUT / 'o2', smoke=True,
                 arm='k11=40')


def test_infer_sides_is_bit_equal_to_r14s_on_k0_style_weights(monkeypatch, tmp_path):
    monkeypatch.setattr(P, 'K0_ROOT', tmp_path / 'k0')
    names, img, cap = stub_inputs(monkeypatch, cap_width=2048)
    path = save_head(tmp_path / 'k0', 'k11=40', 'style', 0, 2048, {})
    ours = T.infer_sides(path, dict(img=img, cap=cap), 'cpu')
    theirs = B._infer_sides(path, 'style', 1152, dict(img=img, cap=cap), 'cpu')
    assert ours.keys() == theirs.keys() == {'img', 'cap'}
    for side in ours:
        assert ours[side].dtype == np.float16 and np.array_equal(ours[side], theirs[side])


# ------------------------------------------------------------------------------------------- gate (i) comparisons
def make_run(root, arm, seed, state_of, best_epoch=3):
    for k, aspect in enumerate(T.ASPECTS):
        path = B.rb_student.weights_path(root, arm, aspect, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state_of(k), path)
        path.with_suffix('.json').write_text(json.dumps(dict(weights_sha256=T.file_sha(path), best_epoch=best_epoch,
                                                             best_dev_loss=0.25 + k)))


def test_compare_weights_equal_and_different(tmp_path):
    same = lambda k: {'a': torch.arange(4.) + k, 'b': torch.ones(2, 2) * k}      # noqa: E731
    make_run(tmp_path / 'x', 'qwen', 0, same)
    make_run(tmp_path / 'y', 'k11=40', 0, same)
    r = T.compare_weights(tmp_path / 'x', 'qwen', tmp_path / 'y', 'k11=40')
    assert r['equal'] and all(h['bytes_equal'] and h['tensors_equal'] for h in r['heads'].values())
    assert r['heads']['emotion']['best_epoch'] == [3, 3]
    nudged = lambda k: {'a': torch.arange(4.) + k + (1e-3 if k == 2 else 0), 'b': torch.ones(2, 2) * k}  # noqa: E731
    make_run(tmp_path / 'z', 'k11=40', 0, nudged)
    r = T.compare_weights(tmp_path / 'x', 'qwen', tmp_path / 'z', 'k11=40')
    assert not r['equal'] and [a for a, h in r['heads'].items() if not h['tensors_equal']] == ['genre']


def test_gate_i_a_compare_writes_record_with_both_comparisons(monkeypatch, tmp_path):
    monkeypatch.setattr(P, 'OUT', tmp_path)
    monkeypatch.setattr(P, 'GATES', tmp_path / 'gates')
    same = lambda k: {'a': torch.arange(4.) + k}                                  # noqa: E731
    other = lambda k: {'a': torch.arange(4.) + k + 1}                             # noqa: E731
    make_run(T.gate_i_a_root('r14'), 'k11=40', 0, same)
    make_run(T.gate_i_a_root('new'), 'qwen', 0, same)
    make_run(T.gate_i_a_root('r14_b'), 'k11=40', 0, other)
    result = T.gate_i_a_compare()
    record = json.loads((tmp_path / 'gates' / 'i_a_weights.json').read_text())
    assert record['gate'] == 'i_a_weights' and record['status'] == 'pass' == result['status'] and record['tries'] == 1
    cmp = record['details']['comparisons']
    assert cmp['new_vs_r14']['equal'] and not cmp['r14_vs_r14_b']['equal']
    assert T.gate_i_a_compare()['details']['comparisons'].keys() == cmp.keys()
    assert json.loads((tmp_path / 'gates' / 'i_a_weights.json').read_text())['tries'] == 2


def write_export(root, kind, arm, aspect, seed, arrays, layout='export'):
    if kind == 'held':
        path = Path(root) / 'held' / arm / f'{aspect}__s{seed}.npz'
    elif layout == 'train':
        path = B.paths_of(root, arm, aspect, seed)[1]
    else:
        path = Path(root) / 'val' / arm / f'{aspect}__s{seed}.npz'
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def test_compare_exports_is_bit_exact_and_reads_r14s_training_time_validation_file(tmp_path):
    rng = np.random.default_rng(2)
    arr = {a: dict(img=rng.normal(size=(5, 4)).astype(np.float16), cap=rng.normal(size=(5, 4)).astype(np.float16))
           for a in T.ASPECTS}
    for a in T.ASPECTS:
        write_export(tmp_path / 'new', 'val', 'k11=40', a, 0, arr[a])
        write_export(tmp_path / 'new', 'held', 'k11=40', a, 0, dict(img=arr[a]['img']))
        write_export(tmp_path / 'old', 'val', 'k11=40', a, 0, arr[a], layout='train')
        write_export(tmp_path / 'old', 'held', 'k11=40', a, 0, dict(img=arr[a]['img']))
    r = T.compare_exports(tmp_path / 'new', tmp_path / 'old', 'k11=40', [0])
    assert r['equal'] and len(r['items']) == 3 * (2 + 1)
    bad = arr['style']['cap'].copy()
    bad[0, 0] = np.nextafter(bad[0, 0], np.float16(10))
    write_export(tmp_path / 'old', 'val', 'k11=40', 'style', 0, dict(img=arr['style']['img'], cap=bad), layout='train')
    r = T.compare_exports(tmp_path / 'new', tmp_path / 'old', 'k11=40', [0])
    assert not r['equal'] and r['items']['val/style__s0/cap']['differing_values'] == 1
    assert r['items']['val/style__s0/img']['equal'] and r['items']['held/style__s0/img']['equal']


def test_gate_i_c_runs_export_then_compares_on_the_stored_devices(monkeypatch, tmp_path):
    monkeypatch.setattr(P, 'OUT', tmp_path)
    monkeypatch.setattr(P, 'GATES', tmp_path / 'gates')
    monkeypatch.setattr(P, 'K0_ROOT', tmp_path / 'k0')
    for a in T.ASPECTS:
        save_head(tmp_path / 'k0', 'k11=40', a, 0, 2048, {})
        (tmp_path / 'k0' / 'held' / 'k11=40').mkdir(parents=True, exist_ok=True)
        (tmp_path / 'k0' / 'held' / 'k11=40' / f'{a}__s0.json').write_text(json.dumps(dict(export_device='cpu')))
    calls = []
    def export(*a, **k):
        calls.append((a, k))
        (k['out_root'] / k['kinds'][0]).mkdir(parents=True)

    monkeypatch.setattr(T, 'export', export)
    ref_roots = []
    def reference(*a, **k):
        ref_roots.append(k['out_root'])

    monkeypatch.setattr(B, 'export_validation', reference)
    monkeypatch.setattr(B, 'export_held', reference)
    monkeypatch.setattr(T, 'compare_exports', lambda *a, **k: dict(equal=True, items={'x': dict(equal=True)}))
    result = T.gate_i_c()
    assert result['status'] == 'pass' and result['details']['devices'] == dict(val='cpu', held='cpu')
    assert [(c[1]['kinds'], c[1]['device'], c[1]['arm']) for c in calls] == [(('val',), 'cpu', 'k11=40'),
                                                                              (('held',), 'cpu', 'k11=40')]
    record = json.loads((tmp_path / 'gates' / 'i_c.json').read_text())
    assert record['status'] == 'pass' and record['details']['r29_code_sha256'] == T.code_hashes()
    assert record['details']['out_root'] == str(tmp_path / 'gate_i_c' / 'try1')
    assert ref_roots == [tmp_path / 'gate_i_c' / 'try1' / 'r14_code'] * 2
    T.gate_i_c(r14_code=False, write=False)
    assert calls[-1][1]['out_root'] == tmp_path / 'gate_i_c' / 'try2'
    assert json.loads((tmp_path / 'gates' / 'i_c.json').read_text())['tries'] == 1
    T.gate_i_c(r14_code=False)
    record = json.loads((tmp_path / 'gates' / 'i_c.json').read_text())
    assert record['tries'] == 2 and record['details']['out_root'].endswith('/try3')


@pytest.mark.parametrize('write', [True, False])
def test_gate_i_c_refuses_nonempty_output_before_loading_or_comparing(monkeypatch, tmp_path, write):
    monkeypatch.setattr(P, 'OUT', tmp_path)
    out = tmp_path / 'prior'
    out.mkdir()
    (out / 'old.npz').write_bytes(b'previous try')
    monkeypatch.setattr(T, 'stored_devices', lambda *a: pytest.fail('must reject the folder before reading K0'))
    with pytest.raises(AssertionError, match='non-empty'):
        T.gate_i_c(out_root=out, write=write)
    assert (out / 'old.npz').read_bytes() == b'previous try'
