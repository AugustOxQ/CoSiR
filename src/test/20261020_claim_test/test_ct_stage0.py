"""Stage 0 verdict checks at the queue's public gate."""
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
