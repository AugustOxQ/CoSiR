"""Stage 1 repeat comparisons report differences without failing the run."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


HERE = Path(__file__).resolve().parent
NAME = 'stage1/clip_roberta__s0.npz'


@pytest.fixture
def roots(tmp_path):
    first, second = tmp_path / 'first', tmp_path / 'second'
    for root in (first, second):
        (root / 'stage1').mkdir(parents=True)
        np.savez_compressed(root / NAME,
                            topic=np.array([0., 1., np.nan], dtype=np.float32),
                            kept=np.array([True, False]))
        (root / NAME).with_suffix('.json').write_text(json.dumps(dict(
            config=dict(version='clip_roberta', seed=0), K={'2': 60},
            seconds=dict(total=12.5))))
    return first, second


def run_repeat(roots, tmp_path):
    first, second = roots
    out = tmp_path / 'stage1_repeat.json'
    result = subprocess.run([sys.executable, str(HERE / 'pc_repeat.py'),
                             '--first', str(first), '--second', str(second),
                             '--out', str(out)], capture_output=True, text=True, timeout=30)
    report = json.loads(out.read_text()) if out.exists() else None
    return result, report


def test_identical_arrays_and_sidecars(roots, tmp_path):
    result, report = run_repeat(roots, tmp_path)
    assert result.returncode == 0, result.stderr
    assert report == dict(
        bitwise_equal=True, arrays=dict(kept=True, topic=True),
        sidecar=dict(K=True, config=True), first=str(roots[0]), second=str(roots[1]),
        first_npz_sha256=hashlib.sha256((roots[0] / NAME).read_bytes()).hexdigest(),
        second_npz_sha256=hashlib.sha256((roots[1] / NAME).read_bytes()).hexdigest())


def test_one_float32_bit_difference_names_array_and_exits_zero(roots, tmp_path):
    path = roots[1] / NAME
    with np.load(path) as saved:
        arrays = dict(saved)
    arrays['topic'].view(np.uint32)[1] ^= 1
    np.savez_compressed(path, **arrays)
    result, report = run_repeat(roots, tmp_path)
    assert result.returncode == 0, result.stderr
    assert report['bitwise_equal'] is False
    assert report['arrays'] == dict(kept=True, topic=False)
    assert all(report['sidecar'].values())
    assert report['first_npz_sha256'] != report['second_npz_sha256']


def test_seconds_are_ignored(roots, tmp_path):
    path = (roots[1] / NAME).with_suffix('.json')
    sidecar = json.loads(path.read_text())
    sidecar['seconds'] = dict(total=99., extra_phase=.5)
    path.write_text(json.dumps(sidecar))
    result, report = run_repeat(roots, tmp_path)
    assert result.returncode == 0, result.stderr
    assert report['bitwise_equal'] is True
    assert 'seconds' not in report['sidecar']


@pytest.mark.parametrize('root_index,suffix', [(0, '.npz'), (0, '.json'),
                                              (1, '.npz'), (1, '.json')])
def test_missing_file_exits_nonzero(roots, tmp_path, root_index, suffix):
    (roots[root_index] / NAME).with_suffix(suffix).unlink()
    result, report = run_repeat(roots, tmp_path)
    assert result.returncode != 0
    assert report is None
    assert 'FileNotFoundError' in result.stderr


def test_missing_input_removes_previous_report(roots, tmp_path):
    result, report = run_repeat(roots, tmp_path)
    assert result.returncode == 0, result.stderr
    assert report['bitwise_equal'] is True
    (roots[1] / NAME).with_suffix('.json').unlink()
    result, report = run_repeat(roots, tmp_path)
    assert result.returncode != 0
    assert report is None
    assert not (tmp_path / 'stage1_repeat.json').exists()


@pytest.mark.parametrize('change', ['key', 'dtype', 'shape'])
def test_array_key_dtype_and_shape_must_match(roots, tmp_path, change):
    path = roots[1] / NAME
    with np.load(path) as saved:
        arrays = dict(saved)
    if change == 'key':
        arrays['other'] = arrays.pop('topic')
    elif change == 'dtype':
        arrays['topic'] = arrays['topic'].view(np.uint32)
    else:
        arrays['topic'] = arrays['topic'].reshape(1, 3)
    np.savez_compressed(path, **arrays)
    result, report = run_repeat(roots, tmp_path)
    assert result.returncode == 0, result.stderr
    assert report['bitwise_equal'] is False
    assert report['arrays']['topic'] is False
    if change == 'key':
        assert report['arrays']['other'] is False


@pytest.mark.parametrize('change', ['value', 'key', 'type'])
def test_sidecar_values_and_keys_must_match(roots, tmp_path, change):
    path = (roots[1] / NAME).with_suffix('.json')
    sidecar = json.loads(path.read_text())
    if change == 'value':
        sidecar['config']['seed'] = 1
    elif change == 'type':
        sidecar['config']['seed'] = False
    else:
        del sidecar['config']
        sidecar['other'] = None
    path.write_text(json.dumps(sidecar))
    result, report = run_repeat(roots, tmp_path)
    assert result.returncode == 0, result.stderr
    assert report['bitwise_equal'] is False
    assert report['sidecar']['config'] is False
    if change == 'key':
        assert report['sidecar']['other'] is False


def test_archive_bytes_are_reported_but_do_not_gate_array_equality(roots, tmp_path):
    path = roots[1] / NAME
    with np.load(path) as saved:
        arrays = dict(saved)
    np.savez(path, **arrays)
    result, report = run_repeat(roots, tmp_path)
    assert result.returncode == 0, result.stderr
    assert report['bitwise_equal'] is True
    assert report['first_npz_sha256'] != report['second_npz_sha256']
