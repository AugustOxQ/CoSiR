"""Stage 2a regression check and the stage 2 V1 gate (CPU only, fake trees, no held data)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ct2_stage2a as ct

K = 20
SAMPLE_SHA = 'sample-fingerprint'
CLOSE = 'traced kernel order'
ALL_PASS = lambda: {key: {'status': 'PASS'} for key in ct.required_items()}


# --- comparisons: equal inputs pass, a one-bit change fails -----------------------------------------

def flip_bit(x):
    y = x.copy()
    y.view(np.uint8).reshape(-1)[0] ^= 1
    return y


def arrays():
    rng = np.random.default_rng(0)
    return dict(center_norms=rng.random(100, dtype=np.float32), fit_counts=np.arange(100, dtype=np.int64),
                kept=np.arange(100) < 40, topic=np.arange(50, dtype=np.int16),
                q_paint_eval=rng.random((5, 40), dtype=np.float32))


def test_arrays_pass_by_hash_when_equal():
    result = ct.compare_arrays(arrays(), arrays())
    assert result['status'] == 'PASS' and result['mode'] == 'hash'


@pytest.mark.parametrize('key', list(arrays()))
def test_one_bit_change_in_any_array_fails_without_a_named_cause(key):
    actual = arrays()
    actual[key] = flip_bit(actual[key])
    result = ct.compare_arrays(actual, arrays())
    assert result['status'] == 'FAIL' and result['mode'] == 'ami'


@pytest.mark.parametrize('damage', ['dtype', 'shape', 'missing_key', 'extra_key'])
def test_array_comparison_sees_dtype_shape_and_key_changes(damage):
    actual = arrays()
    if damage == 'dtype':
        actual['topic'] = actual['topic'].astype(np.int32)
    elif damage == 'shape':
        actual['topic'] = actual['topic'].reshape(5, 10)
    elif damage == 'missing_key':
        del actual['topic']
    else:
        actual['extra'] = np.zeros(1)
    assert ct.compare_arrays(actual, arrays())['status'] == 'FAIL'


def test_hash_mismatch_passes_only_with_cause_and_three_close_amis():
    actual = arrays()
    actual['topic'] = flip_bit(actual['topic'])
    expected_ami = dict(emotion=.1, style=.2, genre=.3)
    near = dict(expected_ami, genre=.3 + 5e-5)
    far = dict(expected_ami, genre=.3 + 2e-4)
    assert ct.compare_arrays(actual, arrays(), near, expected_ami, CLOSE)['status'] == 'PASS'
    assert ct.compare_arrays(actual, arrays(), near, expected_ami, '')['status'] == 'FAIL'
    assert ct.compare_arrays(actual, arrays(), far, expected_ami, CLOSE)['status'] == 'FAIL'
    assert ct.compare_arrays(actual, arrays(), {'emotion': .1}, {'emotion': .1}, CLOSE)['status'] == 'FAIL'


def judgments(**change):
    runs = [dict(version=v, seed=0, cut=c, kept_count=n, share=s, collapsed=x)
            for v, c, n, s, x in (('clip_roberta', 'strict', 1, 1., True),
                                  ('clip_roberta', 'k40', 40, .048, False))]
    for run in runs:
        run.update(change.get(run['cut'], {}))
    return runs


def test_rcol_judgments_pass_when_equal():
    assert ct.compare_judgments(judgments(), judgments())['status'] == 'PASS'


@pytest.mark.parametrize('change', [
    {'strict': dict(kept_count=2)}, {'k40': dict(collapsed=True)}, {'k40': dict(share=.048 + 2e-4)}])
def test_rcol_judgment_change_fails(change):
    assert ct.compare_judgments(judgments(**change), judgments())['status'] == 'FAIL'


def test_rcol_share_within_stage1_topic_tolerance_passes_and_missing_cut_fails():
    assert ct.compare_judgments(judgments(k40=dict(share=.048 + 5e-5)), judgments())['status'] == 'PASS'
    assert ct.compare_judgments(judgments()[:1], judgments())['status'] == 'FAIL'
    assert ct.compare_judgments(judgments() + judgments()[:1], judgments())['status'] == 'FAIL'


@pytest.mark.parametrize('damage', ['row_order', 'dtype', 'value', 'nonfinite'])
def test_replay_requires_exact_finite_arrays(damage):
    expected = np.asarray([[1., 0.], [0., 1.]], dtype=np.float32)
    assert ct.compare_replay(expected.copy(), expected)['status'] == 'PASS'
    actual = expected.copy()
    if damage == 'row_order':
        actual = actual[::-1]
    elif damage == 'dtype':
        actual = actual.astype(np.float16)
    elif damage == 'value':
        actual = flip_bit(actual)
    else:
        actual[0, 0] = np.nan
    assert ct.compare_replay(actual, expected)['status'] == 'FAIL'


# --- required items ---------------------------------------------------------------------------------

def test_required_items_cover_stage0s_checks_at_stage2_arms():
    items = ct.required_items()
    assert {f'regression/d2/{a}' for a in ('emotion', 'style', 'genre')} <= items
    assert {f'regression/stage1/{s}/{v}' for s in ('core', 'paper') for v in ('clip_roberta', 'qwen_qwen')} <= items
    mappers = {f'{s}/{c}/{a}' for s, c, a in [('core', 'k40', 'P1'), ('core', 'k40', 'P2'), ('core', 'k40', 'P3'),
                                              ('core', 'k40', 'P4'), ('paper', 'k40', 'P1'), ('paper', 'k40', 'P4'),
                                              ('paper', 'strict', 'P4')]}
    assert {f'regression/stage2/{m}' for m in mappers} <= items
    assert {f'replay/percept/{m}' for m in mappers} <= items
    assert {f'replay/d2/{a}' for a in ('emotion', 'style', 'genre')} <= items
    assert {f'regression/rcol/{v}' for v in ('clip_roberta', 'qwen_qwen')} <= items
    assert 'training/code' in items
    assert not any('strict/P1' in key for key in items)    # absent in stage 1: skipped as stage 1 did
    assert len(items) == 2 + 3 + 4 + 7 + 2 + 3 + 7


# --- the gate ---------------------------------------------------------------------------------------

@pytest.fixture
def s2a_root(tmp_path, monkeypatch):
    root = tmp_path / 'outputs/stage2/s2a'
    root.mkdir(parents=True)
    monkeypatch.setattr(ct, 'S2A', root)
    return root


def test_gate_passes_a_complete_report_and_binds_device_host_and_code(tmp_path, s2a_root, monkeypatch):
    path = tmp_path / 'stage2a.json'
    source = tmp_path / 'output.npz'
    source.write_bytes(b'checked artifact')
    ct.write_report(path, ALL_PASS(), 'cuda', {str(source): ct.file_sha256(source)}, s2a_root)
    report = ct.require_pass(path, 'cuda')
    assert report['status'] == 'PASS' and report['stage'] == '2a' and report['smoke'] is False
    with pytest.raises(AssertionError, match='device'):
        ct.require_pass(path, 'cuda:1')
    monkeypatch.setattr(ct, 'training_host', lambda: 'node408')
    with pytest.raises(AssertionError, match='host'):
        ct.require_pass(path, 'cuda')


def test_gate_default_path_and_device_mirror_stage0():
    import inspect
    assert inspect.signature(ct.require_pass).parameters['path'].default == ct.HERE / 'stage2a.json'
    assert inspect.signature(ct.require_pass).parameters['device'].default == 'cuda'


def test_gate_fails_on_a_code_edit(tmp_path, s2a_root, monkeypatch):
    path = tmp_path / 'stage2a.json'
    ct.write_report(path, ALL_PASS(), 'cuda', {}, s2a_root)
    assert ct.require_pass(path, 'cuda')
    identity = ct.code_identity()
    monkeypatch.setattr(ct, 'code_identity', lambda: dict(identity, **{'ct2_verdict.py': 'edited'}))
    with pytest.raises(AssertionError, match='code'):
        ct.require_pass(path, 'cuda')


def test_code_identity_binds_ct2_modules_ct_code_rule_and_reference_folders():
    identity = ct.code_identity()
    names = {Path(key).name for key in identity}
    assert {'ct2_stage2a.py', 'ct2_d2.py', 'ct2_percept.py', 'ct2_verdict.py', 'ct_percept.py',
            'ct_stage0.py', 'DECISION_RULE.md', 'ms_train.py', 'pc_stage1.py', 'ags_graphs.py'} <= names
    assert not any(name.startswith('test_') for name in names)


def test_ct2_identity_changes_when_a_ct2_file_changes(tmp_path, monkeypatch):
    (tmp_path / 'ct2_a.py').write_text('x = 1\n')
    (tmp_path / 'test_ct2_a.py').write_text('tests are not code under check\n')
    monkeypatch.setattr(ct, 'HERE', tmp_path)
    before = ct.ct2_identity()
    assert set(before) == {f'{tmp_path.name}/ct2_a.py'}
    (tmp_path / 'ct2_a.py').write_text('x = 2\n')
    assert ct.ct2_identity() != before


@pytest.mark.parametrize('drop', ['missing', 'failed', 'extra'])
def test_gate_fails_on_a_missing_failed_or_unknown_item(tmp_path, s2a_root, drop):
    path = tmp_path / 'stage2a.json'
    items = ALL_PASS()
    if drop == 'missing':
        items.pop('replay/d2/style')
    elif drop == 'failed':
        items['regression/stage2/core/k40/P3'] = {'status': 'FAIL'}
    else:
        items['regression/stage2/paper/strict/P1'] = {'status': 'PASS'}
    report = ct.write_report(path, items, 'cuda', {}, s2a_root)
    assert (report['status'] == 'PASS') == (drop == 'extra')
    with pytest.raises(AssertionError):
        ct.require_pass(path, 'cuda')


def test_gate_fails_when_a_checked_artifact_changes_or_report_is_missing(tmp_path, s2a_root):
    path = tmp_path / 'stage2a.json'
    with pytest.raises((AssertionError, FileNotFoundError)):
        ct.require_pass(path, 'cuda')
    source = tmp_path / 'output.npz'
    source.write_bytes(b'checked')
    ct.write_report(path, ALL_PASS(), 'cuda', {str(source): ct.file_sha256(source)}, s2a_root)
    source.write_bytes(b'changed after the check')
    with pytest.raises(AssertionError, match='source'):
        ct.require_pass(path, 'cuda')


def test_gate_is_not_opened_by_a_stage0_report_or_a_scratch_root(tmp_path, s2a_root):
    path = tmp_path / 'stage2a.json'
    ct.write_report(path, ALL_PASS(), 'cuda', {}, tmp_path / 'scratch')
    with pytest.raises(AssertionError, match='root'):
        ct.require_pass(path, 'cuda')
    ct.write_report(path, ALL_PASS(), 'cuda', {}, s2a_root)
    report = json.loads(path.read_text())
    report.update(stage=0)
    path.write_text(json.dumps(report))
    with pytest.raises(AssertionError):
        ct.require_pass(path, 'cuda')


def test_gate_is_the_one_ct2_d2_calls(tmp_path, s2a_root, monkeypatch):
    import ct2_d2
    monkeypatch.setattr(ct.require_pass, '__defaults__', (tmp_path / 'missing.json', 'cuda'))
    with pytest.raises(FileNotFoundError):
        ct2_d2._require_pass('cuda')


# --- writing rules ----------------------------------------------------------------------------------

def test_artifacts_go_only_to_the_s2a_folder_or_a_scratch_folder(tmp_path, monkeypatch):
    outputs = tmp_path / 'outputs'
    monkeypatch.setattr(ct, 'HERE', tmp_path)
    monkeypatch.setattr(ct, 'S2A', outputs / 'stage2/s2a')
    monkeypatch.setattr(ct, 'REFERENCE', tmp_path / 'main/outputs')
    assert ct.check_root(outputs / 'stage2/s2a') == (outputs / 'stage2/s2a').resolve()
    assert ct.check_root(tmp_path / 'scratch')
    assert ct.check_root(ct.chunked_root(outputs / 'stage2/s2a')) == (outputs / 'stage2/s2a_chunked').resolve()
    for bad in (outputs, outputs / 'd2', outputs / 'stage2/full', outputs / 'stage2',
                tmp_path / 'main/outputs', tmp_path / 'main/outputs/stage2/s2a'):
        with pytest.raises(AssertionError, match='outputs'):
            ct.check_root(bad)


def test_default_report_needs_the_s2a_root(tmp_path, monkeypatch):
    monkeypatch.setattr(ct, 'S2A', tmp_path / 'outputs/stage2/s2a')
    with pytest.raises(AssertionError, match='report'):
        ct.main(['--root', str(tmp_path / 'scratch'), '--check-only'])


# --- training record --------------------------------------------------------------------------------

def test_training_must_have_run_under_the_current_code(tmp_path):
    def run():
        return ct.training_code_item(tmp_path)['status']
    assert run() == 'FAIL'                                          # no record
    current = ct.code_identity()
    for step in ('d2', 'core'):
        ct.write_training_code(tmp_path, step, current)
    assert run() == 'FAIL'                                          # paper not recorded
    ct.write_training_code(tmp_path, 'paper', dict(current, **{'ct2_percept.py': 'older code'}))
    assert run() == 'FAIL'                                          # paper trained by older code
    ct.write_training_code(tmp_path, 'paper', current)
    assert run() == 'PASS'


def test_stale_seed0_artifacts_are_renamed_never_deleted(tmp_path):
    old = tmp_path / 'percept/core/stage1/clip_roberta__s0.npz'
    old.parent.mkdir(parents=True)
    old.write_bytes(b'older code')
    moved = ct.prepare_training(tmp_path, 'core')
    assert moved and not old.exists()
    assert [p.read_bytes() for p in (tmp_path / 'stale').rglob('*.npz')] == [b'older code']
    fresh = tmp_path / 'percept/core/stage1/qwen_qwen__s0.npz'
    fresh.parent.mkdir(parents=True, exist_ok=True)
    fresh.write_bytes(b'this code')
    assert ct.prepare_training(tmp_path, 'core') == [] and fresh.exists()   # same identity: kept
    assert ct.training_code_item(tmp_path)['stale_steps'] == ['d2', 'paper']


def test_training_refuses_after_held_outputs_exist(tmp_path):
    (tmp_path / 'held/d2').mkdir(parents=True)
    (tmp_path / 'held/d2/emotion__s0.npz').write_bytes(b'held')
    with pytest.raises(AssertionError, match='V4'):
        ct.train(tmp_path, 'cuda')


def test_train_runs_every_seed0_cell_in_sample_mode(tmp_path, monkeypatch):
    import ct2_d2
    import ct2_percept
    calls = []
    monkeypatch.setattr(ct2_d2, 'train', lambda *a, **k: calls.append(('d2', a, k)))
    monkeypatch.setattr(ct2_percept, 'run_stage1', lambda *a, **k: calls.append(('s1', a, k)))
    monkeypatch.setattr(ct2_percept, 'run_stage2', lambda *a, **k: calls.append(('s2', a, k)))
    ct.train(tmp_path, 'cuda')
    assert not any('dec_chunk_rows' in k for kind, a, k in calls)     # gating runs keep the mode's default
    assert calls[0][0] == 'd2' and calls[0][1][:3] == ('sample', [0], 'cuda')
    stage1 = {a[1:4] for kind, a, k in calls if kind == 's1'}
    assert stage1 == {(s, v, 0) for s in ('core', 'paper') for v in ('clip_roberta', 'qwen_qwen')}
    stage2 = {a[1:5] for kind, a, k in calls if kind == 's2'}
    assert stage2 == ({('core', 'k40', arm, 0) for arm in ('P1', 'P2', 'P3', 'P4')} |
                      {('paper', cut, arm, 0) for cut in ('strict', 'k40') for arm in ('P1', 'P4')})
    assert all(a[0] == 'sample' and k['root'] == tmp_path.resolve() and k['device'] == 'cuda'
               for kind, a, k in calls if kind != 'd2')
    assert ct.training_code_item(tmp_path)['status'] == 'PASS'
    order = [kind for kind, *_ in calls]
    assert order.index('s1') < order.index('s2')


# --- the check on fake trees (stage 1 layout in both) ----------------------------------------------

def sha(path):
    return ct.file_sha256(path)


def put_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def put_npz(path, **arrays_):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays_)
    return sha(path)


def d2_arrays(aspect):
    rng = np.random.default_rng(len(aspect))
    x = rng.standard_normal((5000, 64)).astype(np.float32)
    x = np.pad(x / np.linalg.norm(x, axis=1, keepdims=True), ((0, 0), (0, 2048 - 64)))
    return dict(img=x.astype(np.float16), cap=x[::-1].astype(np.float16))


def probs(seed, k=K):
    return np.random.default_rng(seed).random((5000, k), dtype=np.float32)


def stage1_arrays(version):
    rng = np.random.default_rng(len(version))
    return dict(center_norms=rng.random(100, dtype=np.float32), fit_counts=np.arange(100, dtype=np.int64),
                kept=np.arange(100) < 40, topic=np.arange(50, dtype=np.int16), q_paint_eval=probs(7, 40))


D2_RECIPE = dict(row=ct.ROWS['D2'], smoke=False, seed=0, lr=.001, batch_size=512,
                 fit_pair_count=35754, dev_pair_count=4926, sample_sha256=SAMPLE_SHA)
GRAPH_KEYS = dict(graph_identity='x', graph_sha256='x', graph_archives={'emotion': 'x'})


def build_d2(root, aspect, *, current, device='cuda', data=None, config=None):
    graph = ct.GRAPHS[aspect]
    data = data or d2_arrays(aspect)
    path = root / f'd2/train/D2/{aspect}__{graph}__s0.npz'
    digest = put_npz(path, **data)
    cfg = dict(D2_RECIPE, aspect=aspect, graph_row=graph, device=device, **GRAPH_KEYS)
    if current:
        cfg.update(mode='sample', graph_identity='ct2', graph_sha256='y', graph_archives={aspect: 'y'})
    cfg.update(config or {})
    put_json(path.with_suffix('.json'), dict(output_sha256=digest, config=cfg))
    if current:
        weight = root / f'd2/weights/{aspect}__s0.pt'
        weight.parent.mkdir(parents=True, exist_ok=True)
        weight.write_bytes(b'weights')
        put_json(weight.with_suffix('.json'), dict(weights_sha256=sha(weight), output_sha256=digest, config=cfg))


PIPE = dict(core=[('k40', 'P1'), ('k40', 'P2'), ('k40', 'P3'), ('k40', 'P4')],
            paper=[('k40', 'P1'), ('k40', 'P4'), ('strict', 'P4')])
VERSION = dict(P1='clip_roberta', P2='clip_roberta', P3='qwen_qwen', P4='qwen_qwen')
RECIPE = dict(seed=0, smoke=False, lr=.003, epochs=100)


def build_stage1(root, settings, version, *, current, device='cuda', data=None, config=None):
    path = root / f'percept/{settings}/stage1/{version}__s0.npz'
    data = data or stage1_arrays(version)
    put_npz(path, **data)
    path.with_suffix('.pt').write_bytes(b'stage1 weights')
    cfg = dict(RECIPE, version=version, settings=settings, device=device)
    if current:
        cfg.update(mode='sample', names_sha256={'fit': 'n'})
    cfg.update(config or {})
    put_json(path.with_suffix('.json'), dict(config=cfg, sample_sha256=SAMPLE_SHA,
             array_sha256s={k: ct.array_sha256(v) for k, v in data.items()}, weights_sha256=sha(path.with_suffix('.pt'))))


def build_stage2(root, settings, cut, arm, *, current, device='cuda', data=None, config=None, broken=None):
    path = root / f'percept/{settings}/stage2/{cut}/{arm}__s0.npz'
    data = probs(hash((settings, cut, arm)) % 1000) if data is None else data
    put_npz(path, eval_prob=data)
    path.with_suffix('.pt').write_bytes(b'mapper weights')
    source = root / f'percept/{settings}/stage1/{VERSION[arm]}__s0.json'
    source_meta = json.loads(source.read_text())
    cfg = dict(RECIPE, arm=arm, settings=settings, cut=cut, device=device)
    if current:
        cfg.update(mode='sample', names_sha256={'fit': 'n'})
    cfg.update(config or {})
    meta = dict(config=cfg, K=data.shape[1], sample_sha256=SAMPLE_SHA,
                stage1_sha256=sha(source.with_suffix('.npz')), stage1_array_sha256s=source_meta['array_sha256s'],
                stage1_weights_sha256=source_meta['weights_sha256'])
    if broken:
        meta[broken] = 'stale'
    put_json(path.with_suffix('.json'), meta)


def build_tree(root, *, current, aspects=(), stage1=(), mappers=(), device='cuda', **kw):
    for aspect in aspects:
        build_d2(root, aspect, current=current, device=device, **kw.get('d2', {}))
    for settings, version in stage1:
        build_stage1(root, settings, version, current=current, device=device, **kw.get('stage1', {}))
    for settings, cut, arm in mappers:
        build_stage2(root, settings, cut, arm, current=current, device=device, **kw.get('stage2', {}))


@pytest.fixture
def labels():
    y = np.tile(np.arange(K), 5000 // K).astype(np.int64)
    return {a: y for a in ('emotion', 'style', 'genre')}


@pytest.fixture
def world(tmp_path, labels, monkeypatch):
    """A reference tree and a current tree on disk; loaders that need the GPU or held-side stores are faked."""
    reference, root = tmp_path / 'reference', tmp_path / 'outputs/stage2/s2a'
    names = [f'painting-{i}' for i in range(5000)]
    monkeypatch.setattr(ct, 'S2A', root)
    monkeypatch.setattr(ct, 'context', lambda: (names, labels))
    monkeypatch.setattr(ct, 'verify_percept', lambda path, root: json.loads(Path(path).with_suffix('.json').read_text()))
    state = dict(d2=None, mapper=None)
    monkeypatch.setattr(ct, 'infer_d2', lambda root, aspect, names, device: state['d2'] if state['d2'] is not None
                        else np.load(root / f'd2/train/D2/{aspect}__{ct.GRAPHS[aspect]}__s0.npz')['img'])
    monkeypatch.setattr(ct, 'infer_mapper', lambda root, settings, cut, arm, names, device: state['mapper']
                        if state['mapper'] is not None else
                        np.load(root / f'percept/{settings}/stage2/{cut}/{arm}__s0.npz')['eval_prob'])
    return dict(reference=reference, root=root, tmp=tmp_path, state=state, names=names)


def run(world, item, causes=None, device='cuda'):
    report = ct.check(world['root'], world['tmp'] / 'stage2a.json', device=device, causes=causes or {},
                      reference=world['reference'], only=(item,))
    return report['items'][item]


D2_ITEM, S1_ITEM, S2_ITEM = 'regression/d2/emotion', 'regression/stage1/core/clip_roberta', 'regression/stage2/core/k40/P1'
RCOL_ITEM = 'regression/rcol/clip_roberta'


def test_d2_identical_npz_passes_by_hash(world):
    for tree, current in ((world['reference'], False), (world['root'], True)):
        build_tree(tree, current=current, aspects=['emotion'])
    result = run(world, D2_ITEM)
    assert result['status'] == 'PASS' and result['mode'] == 'hash'


def test_d2_one_bit_change_fails_without_cause_and_passes_with_cause_and_close_amis(world):
    build_tree(world['reference'], current=False, aspects=['emotion'])
    changed = d2_arrays('emotion')
    changed['img'] = flip_bit(changed['img'])
    build_tree(world['root'], current=True, aspects=['emotion'], d2=dict(data=changed))
    first = run(world, D2_ITEM)
    assert first['status'] == 'FAIL' and first['mode'] == 'ami'


def test_d2_ami_escape_with_the_real_scorer(world, labels, monkeypatch):
    """Both sides give the same partition, so the AMIs agree to 1e-4 while the bytes differ."""
    y = labels['emotion']
    x = np.pad(np.eye(K, dtype=np.float32)[y], ((0, 0), (0, 2048 - K)))
    other = np.pad(np.eye(K, dtype=np.float32)[y] * 1.0, ((0, 0), (0, 2048 - K)))
    other[0, K] = 1e-3                                 # different bytes, identical partition
    data = lambda img: dict(img=img.astype(np.float16), cap=x.astype(np.float16))
    build_tree(world['reference'], current=False, aspects=['emotion'], d2=dict(data=data(x)))
    build_tree(world['root'], current=True, aspects=['emotion'], d2=dict(data=data(other)))
    assert run(world, D2_ITEM)['status'] == 'FAIL'                          # no named cause
    passed = run(world, D2_ITEM, {D2_ITEM: CLOSE})
    assert passed['status'] == 'PASS' and passed['mode'] == 'ami' and passed['cause'] == CLOSE
    assert set(passed['image_ami']['emotion']['absolute_errors']) == {'emotion', 'style', 'genre'}
    # A far AMI fails even with a cause: swap two clusters' worth of labels in the current partition.
    far = np.pad(np.eye(K, dtype=np.float32)[np.arange(5000) % 3], ((0, 0), (0, 2048 - K)))
    build_tree(world['root'], current=True, aspects=['emotion'], d2=dict(data=data(far)))
    assert run(world, D2_ITEM, {D2_ITEM: CLOSE})['status'] == 'FAIL'


@pytest.mark.parametrize(('field', 'value', 'text'), [
    ('device', 'cpu', 'device'), ('mode', 'full', 'mode'), ('seed', 1, 'recipe'), ('smoke', True, 'recipe'),
    ('lr', .002, 'recipe'), ('fit_pair_count', 35755, 'recipe')])
def test_d2_wrong_device_mode_or_recipe_fails_even_with_identical_bytes(world, field, value, text):
    build_tree(world['reference'], current=False, aspects=['emotion'])
    build_tree(world['root'], current=True, aspects=['emotion'], d2=dict(config={field: value}))
    result = run(world, D2_ITEM)
    assert result['status'] == 'FAIL' and text in result['error']


def test_d2_graph_identity_is_the_only_config_difference_allowed(world):
    build_tree(world['reference'], current=False, aspects=['emotion'])
    build_tree(world['root'], current=True, aspects=['emotion'])
    assert run(world, D2_ITEM)['status'] == 'PASS'      # graph_identity/sha/archives differ by construction


def test_d2_sidecar_hash_must_match_the_npz(world):
    build_tree(world['reference'], current=False, aspects=['emotion'])
    build_tree(world['root'], current=True, aspects=['emotion'])
    path = next(world['root'].glob('d2/train/D2/*.json'))
    meta = json.loads(path.read_text())
    meta['output_sha256'] = '0' * 64
    path.write_text(json.dumps(meta))
    assert run(world, D2_ITEM)['status'] == 'FAIL'


ALL_S1 = [(s, v) for s in ('core', 'paper') for v in ('clip_roberta', 'qwen_qwen')]


def test_stage1_legacy_arrays_pass_by_hash(world):
    for tree, current in ((world['reference'], False), (world['root'], True)):
        build_tree(tree, current=current, stage1=[('core', 'clip_roberta')])
    result = run(world, S1_ITEM)
    assert result['status'] == 'PASS' and result['mode'] == 'hash'
    assert set(result['actual']) == set(ct.LEGACY)


def build_pair(world, settings, arms, **stage1_change):
    """Reference and current trees for one settings' clip_roberta stage 1 and the named dependent mappers."""
    for tree, current in ((world['reference'], False), (world['root'], True)):
        data = stage1_change.get('data') if current else None
        build_stage1(tree, settings, 'clip_roberta', current=current, data=data)
        for arm in arms:
            build_stage2(tree, settings, 'k40', arm, current=current,
                         data=np.eye(K, dtype=np.float32)[np.tile(np.arange(K), 250)])


@pytest.mark.parametrize('key', ct.LEGACY)
def test_stage1_one_bit_change_in_a_legacy_array_fails(world, key):
    changed = stage1_arrays('clip_roberta')
    changed[key] = flip_bit(changed[key])
    build_pair(world, 'core', ('P1', 'P2'), data=changed)
    result = run(world, S1_ITEM)
    assert result['status'] == 'FAIL' and result['mode'] == 'ami'


@pytest.mark.parametrize('settings', ['core', 'paper'])
def test_stage1_difference_checks_all_dependent_mappers(world, settings):
    arms = ('P1', 'P2') if settings == 'core' else ('P1',)
    changed = stage1_arrays('clip_roberta')
    changed['topic'] = flip_bit(changed['topic'])
    build_pair(world, settings, arms, data=changed)
    item = f'regression/stage1/{settings}/clip_roberta'
    assert run(world, item)['status'] == 'FAIL'
    result = run(world, item, {item: CLOSE})
    assert result['status'] == 'PASS' and set(result['image_ami']) == set(arms)


def test_stage1_difference_fails_when_one_dependent_mapper_moved_far(world):
    changed = stage1_arrays('clip_roberta')
    changed['topic'] = flip_bit(changed['topic'])
    build_pair(world, 'core', ('P1', 'P2'), data=changed)
    far = np.eye(K, dtype=np.float32)[np.roll(np.tile(np.arange(K), 250), 1) % 7]
    build_stage2(world['root'], 'core', 'k40', 'P2', current=True, data=far)
    item = 'regression/stage1/core/clip_roberta'
    assert run(world, item, {item: CLOSE})['status'] == 'FAIL'


def test_stage1_current_must_have_run_in_sample_mode_on_the_checked_device(world):
    build_tree(world['reference'], current=False, stage1=[('core', 'clip_roberta')])
    build_stage1(world['root'], 'core', 'clip_roberta', current=True, device='cpu')
    assert 'device' in run(world, S1_ITEM)['error']
    build_stage1(world['root'], 'core', 'clip_roberta', current=True, config={'mode': 'full'})
    assert 'mode' in run(world, S1_ITEM)['error']


def test_stage1_recipe_must_equal_the_reference_recipe(world):
    build_tree(world['reference'], current=False, stage1=[('core', 'clip_roberta')])
    build_stage1(world['root'], 'core', 'clip_roberta', current=True, config={'lr': .5})
    result = run(world, S1_ITEM)
    assert result['status'] == 'FAIL' and 'recipe' in result['error']


def test_mapper_eval_prob_passes_by_hash_and_fails_on_one_bit(world):
    for tree, current in ((world['reference'], False), (world['root'], True)):
        build_tree(tree, current=current, stage1=[('core', 'clip_roberta')], mappers=[('core', 'k40', 'P1')])
    assert run(world, S2_ITEM)['mode'] == 'hash'
    changed = flip_bit(probs(hash(('core', 'k40', 'P1')) % 1000))
    build_stage2(world['root'], 'core', 'k40', 'P1', current=True, data=changed)
    result = run(world, S2_ITEM)
    assert result['status'] == 'FAIL' and result['mode'] == 'ami'


def test_mapper_with_a_named_cause_and_close_amis_passes_but_not_with_far_amis(world):
    perfect = np.eye(K, dtype=np.float32)[np.tile(np.arange(K), 250)]
    near = perfect.copy()
    near[0, 0] = .999
    build_tree(world['reference'], current=False, stage1=[('core', 'clip_roberta')])
    build_stage2(world['reference'], 'core', 'k40', 'P1', current=False, data=perfect)
    build_tree(world['root'], current=True, stage1=[('core', 'clip_roberta')])
    build_stage2(world['root'], 'core', 'k40', 'P1', current=True, data=near)
    assert run(world, S2_ITEM)['status'] == 'FAIL'
    assert run(world, S2_ITEM, {S2_ITEM: CLOSE})['status'] == 'PASS'
    build_stage2(world['root'], 'core', 'k40', 'P1', current=True,
                 data=np.eye(K, dtype=np.float32)[np.arange(5000) % 3])
    assert run(world, S2_ITEM, {S2_ITEM: CLOSE})['status'] == 'FAIL'


@pytest.mark.parametrize('field', ['stage1_array_sha256s', 'stage1_weights_sha256', 'stage1_sha256'])
def test_mapper_must_descend_from_the_checked_stage1_run(world, field):
    for tree, current in ((world['reference'], False), (world['root'], True)):
        build_tree(tree, current=current, stage1=[('core', 'clip_roberta')], mappers=[('core', 'k40', 'P1')])
    build_stage2(world['root'], 'core', 'k40', 'P1', current=True, broken=field,
                 data=np.load(world['reference'] / 'percept/core/stage2/k40/P1__s0.npz')['eval_prob'])
    result = run(world, S2_ITEM)
    assert result['status'] == 'FAIL' and 'Stage 1' in result['error']


def test_paper_strict_p1_is_not_an_item_and_strict_p4_is(world):
    for tree, current in ((world['reference'], False), (world['root'], True)):
        build_tree(tree, current=current, stage1=[('paper', 'qwen_qwen')], mappers=[('paper', 'strict', 'P4')])
    assert run(world, 'regression/stage2/paper/strict/P4')['status'] == 'PASS'


def test_replay_of_inference_must_equal_the_saved_array_bit_for_bit(world):
    for tree, current in ((world['reference'], False), (world['root'], True)):
        build_tree(tree, current=current, aspects=['emotion'], stage1=[('core', 'clip_roberta')],
                   mappers=[('core', 'k40', 'P1')])
    assert run(world, 'replay/d2/emotion')['status'] == 'PASS'
    assert run(world, 'replay/percept/core/k40/P1')['status'] == 'PASS'
    saved = np.load(next(world['root'].glob('d2/train/D2/*.npz')))['img']
    world['state']['d2'] = flip_bit(saved)
    world['state']['mapper'] = flip_bit(probs(hash(('core', 'k40', 'P1')) % 1000))
    assert run(world, 'replay/d2/emotion')['status'] == 'FAIL'
    assert run(world, 'replay/percept/core/k40/P1')['status'] == 'FAIL'
    world['state']['mapper'] = probs(hash(('core', 'k40', 'P1')) % 1000)[:4999]
    assert run(world, 'replay/percept/core/k40/P1')['status'] == 'FAIL'


def judgments_for(version, **change):
    runs = [dict(version=version, seed=0, cut='strict', kept_count=98, share=.022, collapsed=True),
            dict(version=version, seed=0, cut='k40', kept_count=40, share=.059, collapsed=False),
            dict(version=version, seed=1, cut='k40', kept_count=40, share=.059, collapsed=False)]
    for run_ in runs:
        run_.update(change.get(run_['cut'], {}))
    return runs


def test_rcol_item_compares_seed0_judgments_with_stage1s_report(world, monkeypatch):
    (world['reference'] / 'percept').mkdir(parents=True)
    put_json(world['reference'] / 'percept/rcol_stage1.json',
             dict(runs=judgments_for('clip_roberta') + judgments_for('qwen_qwen'), entries={}))
    build_stage1(world['root'], 'paper', 'clip_roberta', current=True)
    state = dict(runs=judgments_for('clip_roberta'))
    monkeypatch.setattr(ct, 'read_rcol_runs', lambda path: state['runs'])
    assert run(world, RCOL_ITEM)['status'] == 'PASS'
    state['runs'] = judgments_for('clip_roberta', k40=dict(collapsed=True))
    assert run(world, RCOL_ITEM)['status'] == 'FAIL'
    state['runs'] = judgments_for('clip_roberta', strict=dict(kept_count=97))
    assert run(world, RCOL_ITEM)['status'] == 'FAIL'


def test_rcol_reader_is_the_stage2_reader_at_the_sample_pair_count(monkeypatch):
    import ct2_verdict
    seen = []
    monkeypatch.setattr(ct2_verdict, 'read_stage1', lambda path, pairs: seen.append((path, pairs)) or [])
    ct.read_rcol_runs(Path('/x/percept/paper/stage1/clip_roberta__s0.npz'))
    assert seen == [(Path('/x/percept/paper/stage1/clip_roberta__s0.npz'), 67327)]


# --- the whole check, held data, writes -------------------------------------------------------------

def build_all(tree, *, current):
    build_tree(tree, current=current, aspects=ct.ASPECTS, stage1=ALL_S1,
               mappers=[(s, c, a) for s, cuts in PIPE.items() for c, a in cuts])


@pytest.fixture
def full_world(world, monkeypatch):
    world['reference'].mkdir(parents=True)
    build_all(world['reference'], current=False)
    build_all(world['root'], current=True)
    runs = [r for v in ('clip_roberta', 'qwen_qwen') for r in judgments_for(v)]
    put_json(world['reference'] / 'percept/rcol_stage1.json', dict(runs=runs, entries={}))
    monkeypatch.setattr(ct, 'read_rcol_runs', lambda path: judgments_for(Path(path).stem.split('__s')[0]))
    current = ct.code_identity()
    for step in ct.TRAINING_STEPS:
        ct.write_training_code(world['root'], step, current)
    return world


def test_complete_fake_trees_pass_and_the_report_gates_the_next_stage(full_world, monkeypatch):
    # No held list, held feature or held writer may be touched by the check.
    import ct_held
    for name in ('read_held_list', 'held_names', 'write_held_list', 'qwen_default_images', 'clip_patches',
                 'clip_global_images'):
        monkeypatch.setattr(ct_held, name, lambda *a, **k: (_ for _ in ()).throw(AssertionError('held read')))
    report = ct.check(full_world['root'], full_world['tmp'] / 'stage2a.json', device='cuda',
                      reference=full_world['reference'])
    assert report['status'] == 'PASS', report['failed']
    assert set(report['items']) == ct.required_items() and not report['missing']
    assert report['stage'] == '2a' and report['device'] == 'cuda' and report['training_host']
    assert ct.require_pass(full_world['tmp'] / 'stage2a.json', 'cuda')['status'] == 'PASS'
    assert str(full_world['root'].resolve()) == report['root']


def test_check_writes_only_the_report_and_never_the_reference_tree(full_world):
    def snapshot(folder):
        return {str(p): sha(p) for p in sorted(Path(folder).rglob('*')) if p.is_file()}
    before_reference, before_root = snapshot(full_world['reference']), snapshot(full_world['root'])
    ct.check(full_world['root'], full_world['tmp'] / 'stage2a.json', device='cuda',
             reference=full_world['reference'])
    assert snapshot(full_world['reference']) == before_reference
    assert snapshot(full_world['root']) == before_root
    assert {p.name for p in full_world['tmp'].iterdir()} == {'reference', 'outputs', 'stage2a.json'}


@pytest.mark.parametrize('where', ['HERE', 'REFERENCE'])
def test_check_refuses_to_run_on_the_stage1_outputs_or_another_outputs_folder(full_world, where):
    bad = ct.HERE / 'outputs/stage2/full' if where == 'HERE' else ct.REFERENCE / 'stage2/s2a'
    with pytest.raises(AssertionError, match='outputs'):
        ct.check(bad, full_world['tmp'] / 'r.json', reference=full_world['reference'])


def test_missing_artifacts_make_failed_items_and_a_closed_gate(world):
    world['reference'].mkdir(parents=True)
    report = ct.check(world['root'], world['tmp'] / 'stage2a.json', device='cuda', reference=world['reference'])
    assert report['status'] == 'FAIL' and set(report['items']) == ct.required_items()
    assert report['items']['regression/stage0/code']['status'] == 'PASS'
    assert all(v['status'] == 'FAIL' for k, v in report['items'].items() if k != 'regression/stage0/code')
    with pytest.raises(AssertionError):
        ct.require_pass(world['tmp'] / 'stage2a.json', 'cuda')


def test_a_code_edit_after_the_check_closes_the_gate(full_world, monkeypatch):
    ct.check(full_world['root'], full_world['tmp'] / 'stage2a.json', device='cuda', reference=full_world['reference'])
    assert ct.require_pass(full_world['tmp'] / 'stage2a.json', 'cuda')
    identity = ct.code_identity()
    monkeypatch.setattr(ct, 'ct2_identity', lambda: {**identity, 'x/ct2_percept.py': 'edited'})
    with pytest.raises(AssertionError, match='code'):
        ct.require_pass(full_world['tmp'] / 'stage2a.json', 'cuda')


def test_cli_require_pass_reports_failure_without_raising(tmp_path, monkeypatch):
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG', raising=False)
    assert ct.main(['--require-pass', '--report', str(tmp_path / 'missing.json')]) == 1
    import os
    assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'


# --- the non-gating chunked item --------------------------------------------------------------------

def test_chunked_training_runs_core_stage1_only_with_1024_rows_in_its_own_root(tmp_path, monkeypatch):
    import ct2_percept
    calls = []
    monkeypatch.setattr(ct2_percept, 'run_stage1', lambda *a, **k: calls.append((a, k)))
    monkeypatch.setattr(ct2_percept, 'run_stage2', lambda *a, **k: calls.append('stage 2'))
    root = tmp_path / 'chunked'
    ct.train_chunked(root, 'cuda')
    assert sorted(a for a, k in calls) == [('sample', 'core', v, 0) for v in ('clip_roberta', 'qwen_qwen')]
    assert all(k == dict(root=root.resolve(), device='cuda', dec_chunk_rows=1024) for a, k in calls)


@pytest.mark.parametrize('changed', [None, 'q_paint_eval'])
def test_chunked_item_reports_hash_match_or_the_amis_and_never_gates(world, labels, changed):
    root = ct.chunked_root(world['root'])
    for version in ('clip_roberta', 'qwen_qwen'):
        build_stage1(world['reference'], 'core', version, current=False)
        data = stage1_arrays(version)
        if changed:
            data = dict(data, q_paint_eval=np.eye(40, dtype=np.float32)[np.arange(5000) % 3])
        build_stage1(root, 'core', version, current=True, data=data)
    item = ct.chunked_item(root, world['reference'], 'clip_roberta')
    assert item['gating'] is False
    if changed is None:
        assert item['status'] == 'MATCH' and item['hash_match'] is True
    else:
        assert item['status'] == 'DIFFERENT' and item['hash_match'] is False
        assert item['differing_arrays'] == ['q_paint_eval']
        assert set(item['actual_ami']) == set(item['expected_ami']) == set(item['ami_differences']) == {
            'emotion', 'style', 'genre'}
        assert item['topics'] == dict(actual=dict(kept=40, used=50), expected=dict(kept=40, used=50))


def test_chunked_item_not_run_when_artifacts_are_missing(world):
    item = ct.chunked_item(ct.chunked_root(world['root']), world['reference'], 'clip_roberta')
    assert item['status'] == 'NOT RUN' and item['gating'] is False


def test_chunked_results_are_reported_but_never_change_pass_or_fail(full_world):
    root = ct.chunked_root(full_world['root'])
    for version in ('clip_roberta', 'qwen_qwen'):
        data = dict(stage1_arrays(version), center_norms=np.ones(100, np.float32))
        build_stage1(root, 'core', version, current=True, data=data)
    report = ct.check(full_world['root'], full_world['tmp'] / 'stage2a.json', device='cuda',
                      reference=full_world['reference'], chunked=True)
    assert report['status'] == 'PASS'
    assert report['chunked']['gating'] is False
    assert {v['status'] for v in report['chunked']['items'].values()} == {'DIFFERENT'}
    assert not any('chunked' in key for key in report['items']) and ct.require_pass(
        full_world['tmp'] / 'stage2a.json', 'cuda')
    skipped = ct.check(full_world['root'], full_world['tmp'] / 'stage2a.json', device='cuda',
                       reference=full_world['reference'], chunked=False)
    assert skipped['chunked'] == dict(gating=False, skipped=True)
    assert skipped['status'] == 'PASS'


def test_static_arm_versions_match_core():
    import ct_percept
    assert ct.VERSION_OF == {arm: ct_percept.ARMS[arm][0] for arm in ct.ARMS}


# --- review fixes (ticket 07) -----------------------------------------------------------------------

def test_real_infer_d2_and_infer_mapper_reach_stage1_inference_with_the_stage1_signatures(tmp_path, monkeypatch):
    """Only the stage 1 inference functions are stubbed; every ct2 wrapper in front of them runs for real."""
    import ct2_d2
    import ct2_percept
    import ct_d2
    import ct_percept
    root = tmp_path / 'outputs/stage2/s2a'
    weight = root / 'd2/weights/emotion__s0.pt'
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b'weights')
    config = dict(aspect='emotion', seed=0, row=ct.ROWS['D2'], name='D2', active_aspects=list(ct.ASPECTS),
                  smoke=False, max_epochs=ct_d2.ms_train.CONFIG['max_epochs'], mode='sample', device='cuda',
                  names_sha256='n', data_sha256='d', stage2_code_sha256=ct_d2.ms_train.head.sha_file(Path(ct2_d2.__file__)))
    put_json(weight.with_suffix('.json'), dict(weights_sha256=sha(weight), config=config))
    names, seen = ['a', 'b'], {}

    def stub_d2(aspect, seed, names, side='img', *, device=None, outdir=None):
        seen['d2'] = (aspect, seed, names, side, device, outdir)
        return 'd2 out'

    def stub_verify(path):
        seen['verify'] = Path(path)
        return dict(config=dict(mode='sample'))

    def stub_mapper(settings, cut, arm, seed, names, *, device=None):
        seen['mapper'] = (settings, cut, arm, seed, names, device, ct_percept.OUTPUTS)
        return 'mapper out'

    monkeypatch.setattr(ct_d2, 'infer', stub_d2)
    monkeypatch.setattr(ct_percept, 'verify_output', stub_verify)
    monkeypatch.setattr(ct_percept, 'infer_stage2', stub_mapper)
    assert ct.infer_d2(root, 'emotion', names, 'cuda') == 'd2 out'
    assert seen['d2'] == ('emotion', 0, names, 'img', 'cuda', root)
    assert ct.infer_mapper(root, 'core', 'k40', 'P1', names, 'cuda') == 'mapper out'
    assert seen['mapper'] == ('core', 'k40', 'P1', 0, names, 'cuda', root)
    # The mode check of ct2_d2's weights info still guards the replay.
    put_json(weight.with_suffix('.json'), dict(weights_sha256=sha(weight), config=dict(config, mode='full')))
    with pytest.raises(AssertionError, match='mode'):
        ct.infer_d2(root, 'emotion', names, 'cuda')


def test_diagnostics_main_needs_the_stage2a_gate_before_any_other_read(tmp_path, monkeypatch):
    import ct2_diag
    monkeypatch.setattr(ct2_diag, 'run', lambda *a, **k: pytest.fail('diagnostics ran without the gate'))
    with pytest.raises(OSError):
        ct2_diag.main(['--verdict', str(tmp_path / 'v.json'), '--stage2a', str(tmp_path / 'missing.json')])
    put_json(tmp_path / 'failed.json', dict(schema=1, stage='2a', smoke=False, status='FAIL', missing=[], failed=['x']))
    with pytest.raises(AssertionError, match='did not pass'):
        ct2_diag.main(['--verdict', str(tmp_path / 'v.json'), '--stage2a', str(tmp_path / 'failed.json')])
    ran = []
    monkeypatch.setattr(ct, 'require_pass', lambda *a, **k: ran.append('gate'))
    monkeypatch.setattr(ct2_diag, 'run', lambda *a, **k: ran.append('run'))
    ct2_diag.main([])
    assert ran == ['gate', 'run']


def test_a_crashing_chunked_training_is_recorded_and_the_gating_report_is_still_written(full_world, monkeypatch):
    report_path = full_world['tmp'] / 'stage2a.json'
    monkeypatch.setattr(ct, 'train', lambda *a, **k: None)
    check = ct.check
    monkeypatch.setattr(ct, 'check', lambda *a, **k: check(*a, reference=full_world['reference'], **k))

    def crash(*a, **k):
        raise RuntimeError('CUDA out of memory')
    monkeypatch.setattr(ct, 'train_chunked', crash)
    assert ct.main(['--root', str(full_world['root']), '--report', str(report_path)]) == 0
    report = json.loads(report_path.read_text())
    assert report['status'] == 'PASS' and not report['failed'] and not report['missing']
    assert report['chunked']['gating'] is False and 'RuntimeError: CUDA out of memory' in report['chunked']['error']
    assert {v['status'] for v in report['chunked']['items'].values()} == {'ERROR'}
    assert ct.require_pass(report_path, 'cuda')['status'] == 'PASS'


def test_a_crashing_chunked_comparison_is_recorded_and_never_gates(world, monkeypatch):
    for version in ('clip_roberta', 'qwen_qwen'):
        build_stage1(world['reference'], 'core', version, current=False)
        build_stage1(ct.chunked_root(world['root']), 'core', version, current=True)
    monkeypatch.setattr(ct, 'compare_arrays', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('boom')))
    item = ct.chunked_item(ct.chunked_root(world['root']), world['reference'], 'clip_roberta')
    assert item['gating'] is False and item['status'] == 'ERROR' and 'RuntimeError: boom' in item['error']


@pytest.mark.parametrize('item', [S1_ITEM, S2_ITEM])
def test_a_gating_percept_item_trained_with_chunked_dec_fails(world, item):
    build_tree(world['reference'], current=False, stage1=[('core', 'clip_roberta')], mappers=[('core', 'k40', 'P1')])

    def build_current(chunk_rows):
        config = dict(dec_chunk_rows=chunk_rows)
        build_stage1(world['root'], 'core', 'clip_roberta', current=True, config=config)
        build_stage2(world['root'], 'core', 'k40', 'P1', current=True, config=config)
    build_current(1024)
    result = run(world, item)
    assert result['status'] == 'FAIL' and 'dec_chunk_rows' in result['error']
    build_current(None)
    assert run(world, item)['status'] == 'PASS'


@pytest.mark.parametrize('changed', [False, True], ids=['equal', 'one-hash-differs'])
def test_stage2a_binds_the_code_verified_by_stage0(full_world, monkeypatch, changed):
    verified = json.loads((ct.HERE / 'stage0.json').read_text())['code']
    current = dict(verified)
    if changed:
        current[next(iter(current))] = 'changed hash'
    monkeypatch.setattr(ct.ct_stage0, 'code_identity', lambda: current)
    report = ct.check(full_world['root'], full_world['tmp'] / 'stage2a.json',
                      reference=full_world['reference'], only=['regression/stage0/code'], chunked=False)
    assert 'regression/stage0/code' in ct.required_items()
    assert report['items']['regression/stage0/code']['status'] == ('FAIL' if changed else 'PASS')
    assert str((ct.HERE / 'stage0.json').resolve()) in report['sources']


@pytest.mark.parametrize('name', ['ct2_queue_cpu.sh', 'ct2_queue_gpu.sh', 'ct2_queue_common.sh'])
def test_queue_script_edits_change_the_code_identity(tmp_path, monkeypatch, name):
    path = tmp_path / name
    path.write_text('#!/bin/bash\necho before\n')
    monkeypatch.setattr(ct, 'HERE', tmp_path)
    before = ct.ct2_identity()
    assert f'{tmp_path.name}/{name}' in before
    path.write_text('#!/bin/bash\necho after\n')
    assert ct.ct2_identity() != before
