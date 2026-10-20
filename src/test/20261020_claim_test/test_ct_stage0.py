"""Stage 0 verdict checks at the queue's public gate."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ct_stage0 as ct


def test_bitwise_regression_passes_without_metric_escape():
    result = ct.compare_hashes({'img': 'same'}, {'img': 'same'})
    assert result['status'] == 'PASS' and result['mode'] == 'hash'


@pytest.mark.parametrize(('cause', 'delta', 'status'), [
    ('', 0., 'FAIL'), ('CUDA kernel rounding traced in replay', .000099, 'PASS'),
    ('CUDA kernel rounding traced in replay', .000101, 'FAIL')])
def test_hash_mismatch_needs_named_cause_and_all_three_amis(cause, delta, status):
    expected = dict(emotion=.1, style=.2, genre=.3)
    actual = dict(expected, genre=.3 + delta)
    result = ct.compare_hashes({'img': 'new'}, {'img': 'old'}, actual, expected, cause)
    assert result['status'] == status
    assert result['mode'] == 'ami' and result['cause'] == cause


def test_missing_aspect_cannot_pass_fallback():
    result = ct.compare_hashes({'x': 'a'}, {'x': 'b'}, {'emotion': .1}, {'emotion': .1}, 'rounding')
    assert result['status'] == 'FAIL'


def test_scorer_replay_uses_six_decimal_means_on_every_aspect():
    values = {a: [.1000001, .1, .0999999] for a in ct.ASPECTS}
    expected = dict.fromkeys(ct.ASPECTS, .1)
    assert ct.compare_scores(values, expected)['status'] == 'PASS'
    values['style'] = [.100001] * 3
    assert ct.compare_scores(values, expected)['status'] == 'FAIL'
    del values['style']
    assert ct.compare_scores(values, expected)['status'] == 'FAIL'


@pytest.mark.parametrize('damage', ['row_order', 'dtype', 'value', 'nonfinite'])
def test_held_path_replay_requires_exact_finite_training_arrays(damage):
    expected = np.asarray([[1., 0.], [0., 1.]], dtype=np.float32)
    assert ct.compare_replay(expected.copy(), expected)['status'] == 'PASS'
    actual = expected.copy()
    if damage == 'row_order':
        actual = actual[::-1]
    elif damage == 'dtype':
        actual = actual.astype(np.float16)
    elif damage == 'value':
        actual[0, 0] = np.nextafter(actual[0, 0], np.float32(0))
    else:
        actual[0, 0] = np.nan
    assert ct.compare_replay(actual, expected)['status'] == 'FAIL'


def test_gate_fails_closed_for_missing_partial_failed_stale_or_wrong_device(tmp_path):
    path = tmp_path / 'stage0.json'
    with pytest.raises((AssertionError, FileNotFoundError)):
        ct.require_pass(path, 'cuda')
    source = tmp_path / 'output.npz'
    source.write_bytes(b'checked training output')
    items = {key: {'status': 'PASS'} for key in ct.required_items()}
    ct.write_report(path, items, 'cuda', {str(source): ct.file_sha256(source)})
    assert ct.require_pass(path, 'cuda')['status'] == 'PASS'
    with pytest.raises(AssertionError, match='device'):
        ct.require_pass(path, 'cuda:1')
    source.write_bytes(b'changed after check')
    with pytest.raises(AssertionError, match='source'):
        ct.require_pass(path, 'cuda')
    ct.write_report(path, items, 'cuda', {str(source): ct.file_sha256(source)})
    items.pop(next(iter(items)))
    assert ct.write_report(path, items, 'cuda', {})['status'] == 'FAIL'
    with pytest.raises(AssertionError):
        ct.require_pass(path, 'cuda')
    items = {key: {'status': 'PASS'} for key in ct.required_items()}
    items['regression/d2/emotion']['status'] = 'FAIL'
    assert ct.write_report(path, items, 'cuda', {})['status'] == 'FAIL'
    with pytest.raises(AssertionError):
        ct.require_pass(path, 'cuda')


def test_gate_rejects_code_changes(tmp_path, monkeypatch):
    path = tmp_path / 'stage0.json'
    ct.write_report(path, {k: {'status': 'PASS'} for k in ct.required_items()}, 'cuda', {})
    monkeypatch.setattr(ct, 'code_identity', lambda: {'changed.py': 'new hash'})
    with pytest.raises(AssertionError, match='code'):
        ct.require_pass(path, 'cuda')


def test_missing_training_artifacts_write_failed_items_and_cannot_open_v1(tmp_path):
    report = ct.check(tmp_path / 'outputs', tmp_path / 'stage0.json', device='cuda')
    assert report['status'] == 'FAIL'
    assert set(report['items']) == ct.required_items()
    assert all(report['items'][f'scorer/{kind}/{name}']['status'] == 'PASS'
               for kind, names in [('d2', ct.ASPECTS), ('percept', ct.ARMS)] for name in names)
    assert all(report['items'][f'replay/d2/{a}']['status'] == 'FAIL' for a in ct.ASPECTS)
    with pytest.raises(AssertionError):
        ct.require_pass(tmp_path / 'stage0.json', 'cuda')


def test_real_development_scorer_matches_known_perfect_partition():
    import ct_score
    y = np.tile(np.arange(20), 250).astype(np.int64)
    x = np.eye(20, dtype=np.float32)[y]
    values = ct_score.score_dev(x, {a: y for a in ct.ASPECTS}, kind='percept')
    assert ct.compare_scores(values, dict.fromkeys(ct.ASPECTS, 1.))['status'] == 'PASS'


def test_gate_cannot_move_between_training_hosts(tmp_path, monkeypatch):
    path = tmp_path / 'stage0.json'
    monkeypatch.setattr(ct, 'training_host', lambda: 'local-gpu-host')
    ct.write_report(path, {k: {'status': 'PASS'} for k in ct.required_items()}, 'cuda', {})
    monkeypatch.setattr(ct, 'training_host', lambda: 'node408')
    with pytest.raises(AssertionError, match='host'):
        ct.require_pass(path, 'cuda')


@pytest.mark.parametrize('changed', ['stage1_array_sha256s', 'stage1_weights_sha256', 'stage1_sha256'])
def test_stage2_must_use_the_checked_stage1_artifact(changed):
    source = dict(array_sha256s={'topic': 'checked-topics'}, weights_sha256='checked-weights')
    mapper = dict(stage1_array_sha256s=source['array_sha256s'],
                  stage1_weights_sha256=source['weights_sha256'], stage1_sha256='checked-file')
    ct.verify_dependency(mapper, source, 'checked-file')
    mapper[changed] = 'stale'
    with pytest.raises(AssertionError, match='Stage 1'):
        ct.verify_dependency(mapper, source, 'checked-file')


def test_scorer_replay_tolerance_is_absolute_not_rounded():
    # Two means 4e-10 apart straddle a six-decimal rounding boundary: equal under abs().
    expected = dict.fromkeys(ct.ASPECTS, .1234565 + 2e-10)
    values = {a: [.1234565 - 2e-10] * 3 for a in ct.ASPECTS}
    assert ct.compare_scores(values, expected)['status'] == 'PASS'
    # Two means 9.998e-7 apart share one rounded value but differ by more than 5e-7.
    expected = dict.fromkeys(ct.ASPECTS, .1234564999)
    values = {a: [.1234555001] * 3 for a in ct.ASPECTS}
    assert ct.compare_scores(values, expected)['status'] == 'FAIL'
    for delta, status in ((4e-7, 'PASS'), (6e-7, 'FAIL')):
        values = {a: [.25 + delta] * 3 for a in ct.ASPECTS}
        assert ct.compare_scores(values, dict.fromkeys(ct.ASPECTS, .25))['status'] == status


def test_stage0_main_sets_cublas_workspace_before_any_cuda_work(monkeypatch, tmp_path):
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', 'placeholder')
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG')
    assert ct.main(['--require-pass', '--report', str(tmp_path / 'missing.json')]) == 1
    import os
    assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':16:8')
    ct.main(['--require-pass', '--report', str(tmp_path / 'missing.json')])
    assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':16:8'


# --- gate logic on tiny fake reference trees (no GPU, no held data) -------------------------------

GATE_ITEM = 'regression/stage2/P1'


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return ct.file_sha256(path)


@pytest.fixture(scope='module')
def frozen():
    import ct_held
    import ct_percept as pt
    s = pt.load_frozen()
    names = s['all_painting_ids'][s['subset_pid']].tolist()
    return dict(fingerprint=pt.pc_data.fingerprint(s), labels=ct_held.painting_labels(names))


def _probs(seed):
    rng = np.random.default_rng(seed)
    return rng.random((5000, 40), dtype=np.float32)


def _recorded_means(labels, eval_prob):
    import ct_score
    values = ct_score.score_dev(eval_prob, labels, kind='percept')
    return {a: float(np.mean(v)) for a, v in values.items()}


def build_percept_trees(root, frozen, *, same_arrays, ami_delta=0., broken=None, device='cuda'):
    """Fake MS/PC references and a fake current core run for Stage 2 arm P1 (and its Stage 1)."""
    ms, pc, outputs = root / 'ms', root / 'pc', root / 'outputs'
    old_prob = _probs(1)
    new_prob = old_prob if same_arrays else _probs(2)
    fp = frozen['fingerprint']
    s1_arrays = dict(center_norms=np.ones(100, np.float32), fit_counts=np.ones(100, np.int64),
                     kept=np.ones(100, bool), topic=np.zeros(10, np.int16),
                     q_paint_eval=_probs(3))
    s1_old = _npz(pc / 'stage1/clip_roberta__s0.npz', **s1_arrays)
    s1_new = _npz(outputs / 'percept/core/stage1/clip_roberta__s0.npz', **s1_arrays)
    s2_old = _npz(pc / 'stage2/P1__s0.npz', eval_prob=old_prob)
    s2_new = _npz(outputs / 'percept/core/stage2/k40/P1__s0.npz', eval_prob=new_prob)
    import ct_percept as pt
    s1_hashes = {k: pt.pc_stage1.array_sha256(v) for k, v in s1_arrays.items()}
    _json(pc / 'stage1/clip_roberta__s0.json', dict(config=dict(seed=0, smoke=False), sample_sha256=fp))
    _json(pc / 'stage2/P1__s0.json', dict(config=dict(seed=0, smoke=False)))
    means = _recorded_means(frozen['labels'], new_prob)
    recorded = dict(means, genre=means['genre'] + ami_delta)
    _json(pc / 'pc_results.json', dict(arms=dict(P1=dict(seeds={'0': dict(
        source_sha256=dict(stage1=s1_old, stage2=s2_old),
        img={a: dict(mean=recorded[a]) for a in ct.ASPECTS})}))))
    _json(ms / 'ms_results.json', dict(runs={}))
    (outputs / 'percept/core/stage1/clip_roberta__s0.pt').write_bytes(b'stage1 weights')
    (outputs / 'percept/core/stage2/k40/P1__s0.pt').write_bytes(b'stage2 weights')
    _json(outputs / 'percept/core/stage1/clip_roberta__s0.json', dict(
        config=dict(seed=0, smoke=False, device=device), sample_sha256=fp,
        array_sha256s=s1_hashes, weights_sha256='stage1-weights'))
    mapper = dict(config=dict(seed=0, smoke=False, device=device), stage1_array_sha256s=s1_hashes,
                  stage1_weights_sha256='stage1-weights', stage1_sha256=s1_new)
    if broken:
        mapper[broken] = 'stale'
    _json(outputs / 'percept/core/stage2/k40/P1__s0.json', mapper)
    return ms, pc, outputs


@pytest.fixture
def run_gate(tmp_path, monkeypatch, frozen):
    import ct_percept as pt
    monkeypatch.setattr(pt, 'verify_output', lambda p: json.loads(Path(p).with_suffix('.json').read_text()))

    def run(item=GATE_ITEM, causes=None, **kw):
        ms, pc, outputs = build_percept_trees(tmp_path, frozen, **kw)
        report = ct.check(outputs, tmp_path / 'stage0.json', device='cuda', causes=causes or {},
                          ms=ms, pc=pc, only=(item,))
        return report['items'][item]
    return run


def test_gate_fake_tree_identical_hashes_pass_by_hash(run_gate):
    result = run_gate(same_arrays=True)
    assert result['status'] == 'PASS' and result['mode'] == 'hash'


def test_gate_hash_mismatch_without_cause_fails(run_gate):
    result = run_gate(same_arrays=False)
    assert result['status'] == 'FAIL' and result['mode'] == 'ami' and result['cause'] == ''
    assert all(r['absolute_errors'] and max(r['absolute_errors'].values()) < 1e-4
               for r in result['image_ami'].values())


def test_gate_hash_mismatch_with_cause_and_close_amis_passes(run_gate):
    result = run_gate(same_arrays=False, ami_delta=5e-5, causes={GATE_ITEM: 'traced kernel order'})
    assert result['status'] == 'PASS' and result['cause'] == 'traced kernel order'


def test_gate_hash_mismatch_with_cause_and_one_far_ami_fails(run_gate):
    result = run_gate(same_arrays=False, ami_delta=2e-4, causes={GATE_ITEM: 'traced kernel order'})
    assert result['status'] == 'FAIL'
    assert max(result['image_ami']['P1']['absolute_errors'].values()) > 1e-4


@pytest.mark.parametrize('field', ['stage1_array_sha256s', 'stage1_weights_sha256', 'stage1_sha256'])
def test_gate_broken_stage1_dependency_fails_even_with_identical_hashes(run_gate, field):
    result = run_gate(same_arrays=True, broken=field)
    assert result['status'] == 'FAIL' and 'Stage 1' in result['error']


def build_d2_trees(root, frozen, *, device):
    ms, outputs = root / 'ms', root / 'outputs'
    graph = ct.GRAPHS['emotion']
    x = np.tile(np.eye(2048, dtype=np.float16)[:1], (5000, 1))
    import ct_d2
    config = dict(aspect='emotion', seed=0, graph_row=graph, row=ct_d2.ROWS['D2'], smoke=False,
                  sample_sha256=frozen['fingerprint'], device=device)
    sha = _npz(ms / f'D2/emotion__{graph}__s0.npz', img=x, cap=x)
    _json(ms / f'D2/emotion__{graph}__s0.json', dict(output_sha256=sha, config=dict(config, device='cuda')))
    (outputs / 'd2/train/D2').mkdir(parents=True)
    (outputs / f'd2/train/D2/emotion__{graph}__s0.npz').write_bytes((ms / f'D2/emotion__{graph}__s0.npz').read_bytes())
    _json(outputs / f'd2/train/D2/emotion__{graph}__s0.json', dict(output_sha256=sha, config=config))
    weight = outputs / 'd2/weights/emotion__s0.pt'
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b'weights')
    _json(weight.with_suffix('.json'), dict(weights_sha256=ct.file_sha256(weight), output_sha256=sha,
                                            config=config))
    _json(ms / 'ms_results.json', dict(runs={'D2/emotion/s0': dict(identity=dict(output_sha256=sha))}))
    _json(root / 'pc/pc_results.json', dict(arms={}))
    return ms, root / 'pc', outputs


@pytest.mark.parametrize(('device', 'status'), [('cuda', 'PASS'), ('cpu', 'FAIL')])
def test_gate_d2_must_be_trained_on_the_checked_device(tmp_path, frozen, device, status):
    ms, pc, outputs = build_d2_trees(tmp_path, frozen, device=device)
    item = 'regression/d2/emotion'
    report = ct.check(outputs, tmp_path / 'stage0.json', device='cuda', ms=ms, pc=pc, only=(item,))
    assert report['items'][item]['status'] == status
    if status == 'FAIL':
        assert 'D2 training device' in report['items'][item]['error']


def test_gate_requires_training_under_the_current_code(tmp_path):
    outputs = tmp_path / 'outputs'
    item = 'training/code'

    def run():
        return ct.check(outputs, tmp_path / 'stage0.json', ms=tmp_path, pc=tmp_path, only=(item,))['items'][item]
    assert run()['status'] == 'FAIL'                     # no record
    current = ct.code_identity()
    ct.write_training_code(outputs, 'd2', current)
    assert run()['status'] == 'FAIL'                     # core not recorded
    ct.write_training_code(outputs, 'core', dict(current, **{'ct_run.py': 'older code'}))
    assert run()['status'] == 'FAIL'                     # core trained by older code
    ct.write_training_code(outputs, 'core', current)
    assert run()['status'] == 'PASS'
    assert item in ct.required_items()
