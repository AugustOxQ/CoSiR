"""Tests for ns_mappers (r17 ticket 05): the validation-probability file format and the gate (iv) judgement.

Run: PYTHONDONTWRITEBYTECODE=1 python -m pytest -q src/test/20261103_new_scores/test_ns_mappers.py
The real-data run (CPU mappers against stored probabilities) is the CLI, not a unit test.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ns_paths as NP  # noqa: E402
import ns_mappers as M  # noqa: E402

N = NP.N_ROWS[('artelingo', 'val')]


@pytest.fixture
def names():
    return [f'painting_{i:05d}' for i in range(N)]


@pytest.fixture
def probs():
    return np.random.default_rng(0).random((N, 40)).astype(np.float32)


@pytest.fixture(autouse=True)
def fake_mapper(monkeypatch):
    """The current mapper's weights hash, per (encoder, seed), without the real mapper files."""
    monkeypatch.setattr(M, 'mapper_weights_sha', lambda enc, seed: f'w-{enc}-{seed}')


def write(tmp_path, names, probs, enc='qwen', seed=0, **kw):
    args = dict(mapper=f'/some/{enc}/{seed}.pt', mapper_sha256=f'w-{enc}-{seed}', store='/some/store')
    args.update(kw)
    return M.write_val(tmp_path, enc, seed, names, probs, **args)


# ------------------------------------------------------------------------------------------------ format
def test_round_trip(tmp_path, names, probs):
    path = write(tmp_path, names, probs)
    assert path == M.val_path(tmp_path, 'qwen', 0)
    assert path.parent.name == 'qwen' and path.parent.parent.name == 'percept_val'
    got_names, got = M.read_val(tmp_path, 'qwen', 0)
    assert got_names == names
    assert got.dtype == np.float32 and got.shape == (N, 40)
    assert np.array_equal(got, probs)
    side = json.loads(path.with_suffix('.json').read_text())
    assert side['n'] == N and side['mapper_sha256'] == 'w-qwen-0' and side['input_store'] == '/some/store'
    assert side['sha256'] == M.sha_file(path)


def test_read_joins_by_name(tmp_path, names, probs):
    write(tmp_path, names, probs)
    want = [names[5], names[2], names[9]]
    got_names, got = M.read_val(tmp_path, 'qwen', 0, names=want)
    assert got_names == want
    assert np.array_equal(got, probs[[5, 2, 9]])
    with pytest.raises(AssertionError, match='not exported'):
        M.read_val(tmp_path, 'qwen', 0, names=['no_such_painting'])


def test_stale_sidecar_refused(tmp_path, names, probs):
    path = write(tmp_path, names, probs)
    other = probs.copy()
    other[0, 0] = 1.0 - other[0, 0]
    np.savez_compressed(path, names=np.asarray(names), probs=other)      # file replaced, sidecar not updated
    with pytest.raises(AssertionError, match='sha256'):
        M.read_val(tmp_path, 'qwen', 0)


def test_missing_sidecar_refused(tmp_path, names, probs):
    path = write(tmp_path, names, probs)
    path.with_suffix('.json').unlink()
    with pytest.raises(FileNotFoundError):
        M.read_val(tmp_path, 'qwen', 0)


def test_wrong_row_count_refused_on_write(tmp_path, names, probs):
    with pytest.raises(AssertionError):
        write(tmp_path, names[:-1], probs[:-1])
    with pytest.raises(AssertionError):
        write(tmp_path, names, probs[:-1])


def test_wrong_row_count_refused_on_read(tmp_path, names, probs):
    """A short file whose sidecar is consistent with it (sha256 and n agree) is still refused: it is not the 6,152."""
    path = M.val_path(tmp_path, 'qwen', 0)
    path.parent.mkdir(parents=True)
    np.savez_compressed(path, names=np.asarray(names[:-1]), probs=probs[:-1])
    side = dict(n=N - 1, sha256=M.sha_file(path), mapper_sha256='w-qwen-0', encoder='qwen', seed=0)
    path.with_suffix('.json').write_text(json.dumps(side))
    with pytest.raises(AssertionError, match='rows'):
        M.read_val(tmp_path, 'qwen', 0)


def test_sidecar_n_must_match_file(tmp_path, names, probs):
    path = write(tmp_path, names, probs)
    side = json.loads(path.with_suffix('.json').read_text())
    side['n'] = N - 1
    path.with_suffix('.json').write_text(json.dumps(side))
    with pytest.raises(AssertionError):
        M.read_val(tmp_path, 'qwen', 0)


def test_sidecar_must_name_this_encoder_and_seed(tmp_path, names, probs):
    write(tmp_path, names, probs, enc='qwen', seed=0)
    src, dst = M.val_path(tmp_path, 'qwen', 0), M.val_path(tmp_path, 'qwen', 1)
    dst.write_bytes(src.read_bytes())
    dst.with_suffix('.json').write_text(src.with_suffix('.json').read_text())     # seed 0's files under seed 1's name
    with pytest.raises(AssertionError, match='seed'):
        M.read_val(tmp_path, 'qwen', 1)


def test_changed_mapper_refused(tmp_path, names, probs, monkeypatch):
    write(tmp_path, names, probs)
    monkeypatch.setattr(M, 'mapper_weights_sha', lambda enc, seed: 'another-hash')
    with pytest.raises(AssertionError, match='mapper'):
        M.read_val(tmp_path, 'qwen', 0)


def test_write_rejects_bad_probs(tmp_path, names, probs):
    bad = probs.copy()
    bad[3, 3] = 1.5
    with pytest.raises(AssertionError):
        write(tmp_path, names, bad)
    bad[3, 3] = np.nan
    with pytest.raises(AssertionError):
        write(tmp_path, names, bad)
    with pytest.raises(AssertionError):
        write(tmp_path, names, probs.astype(np.float64))


def test_write_rejects_unsorted_or_duplicate_names(tmp_path, names, probs):
    swapped = list(names)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    with pytest.raises(AssertionError):
        write(tmp_path, swapped, probs)
    dup = list(names)
    dup[1] = dup[0]
    with pytest.raises(AssertionError):
        write(tmp_path, dup, probs)


# ------------------------------------------------------------------------------------------------ gate (iv)
def test_max_abs_diff():
    a = np.zeros((4, 40), np.float32)
    b = a.copy()
    b[2, 7] = 3e-6
    assert M.max_abs_diff(a, b) == pytest.approx(3e-6, rel=1e-6)
    b[0, 0] = np.nan
    assert M.max_abs_diff(a, b) == float('inf')              # a NaN is never "within tolerance"
    with pytest.raises(AssertionError):
        M.max_abs_diff(a, np.zeros((5, 40), np.float32))


def all_ok(value=1e-7):
    return {e: {s: value for s in NP.SEEDS} for e in NP.ENCODERS}


def test_judge_all_ok():
    out = M.judge(all_ok(), {s: 1e-7 for s in NP.SEEDS})
    assert set(out) == set(NP.ENCODERS)
    assert all(v['status'] == 'ok' for v in out.values())
    assert out['qwen']['evidence']['held_max_abs_diff'] == pytest.approx(1e-7)
    assert out['qwen']['evidence']['val_max_abs_diff'] == pytest.approx(1e-7)
    assert out['clip_l14']['evidence']['val_max_abs_diff'] is None     # r11 stored validation exists only for Qwen


def test_judge_tolerance_is_ns_paths_tol():
    held = all_ok(NP.TOL_MAPPER)                              # at the tolerance: passes
    assert M.judge(held, {s: 0.0 for s in NP.SEEDS})['siglip1']['status'] == 'ok'
    held['siglip1'][3] = NP.TOL_MAPPER * 1.01                 # just above: fails, and only that encoder
    out = M.judge(held, {s: 0.0 for s in NP.SEEDS})
    assert out['siglip1']['status'] == 'failed'
    assert out['siglip1']['evidence']['failed_seeds'] == [3]
    assert [e for e, v in out.items() if v['status'] == 'failed'] == ['siglip1']


def test_judge_qwen_validation_fails_qwen_only():
    val = {s: 0.0 for s in NP.SEEDS}
    val[4] = 2e-5
    out = M.judge(all_ok(), val)
    assert out['qwen']['status'] == 'failed' and out['qwen']['evidence']['val_failed_seeds'] == [4]
    assert all(out[e]['status'] == 'ok' for e in NP.ENCODERS if e != 'qwen')


def test_judge_missing_values_fail():
    held = all_ok()
    del held['siglip2'][2]
    out = M.judge(held, {s: 0.0 for s in NP.SEEDS})
    assert out['siglip2']['status'] == 'failed'
    del held['clip_l14']
    assert M.judge(held, {s: 0.0 for s in NP.SEEDS})['clip_l14']['status'] == 'failed'
    assert M.judge(all_ok(), None)['qwen']['status'] == 'failed'          # no Qwen validation comparison: not ok
    assert M.judge(all_ok(), {0: 0.0})['qwen']['status'] == 'failed'


def test_judge_nan_and_inf_fail():
    held = all_ok()
    held['qwen'][0] = float('nan')
    assert M.judge(held, {s: 0.0 for s in NP.SEEDS})['qwen']['status'] == 'failed'
    held = all_ok()
    held['qwen'][0] = float('inf')
    assert M.judge(held, {s: 0.0 for s in NP.SEEDS})['qwen']['status'] == 'failed'


def _replace_npz(path, names, probs):
    """Replace the npz and bring the sidecar's sha256 and n up to date, but not its names hash."""
    np.savez_compressed(path, names=np.asarray(names), probs=probs)
    side = json.loads(path.with_suffix('.json').read_text())
    side['sha256'], side['n'] = M.sha_file(path), len(names)
    path.with_suffix('.json').write_text(json.dumps(side))


def test_unsorted_names_refused_on_read(tmp_path, names, probs):
    path = write(tmp_path, names, probs)
    swapped = list(names)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    _replace_npz(path, swapped, probs)
    with pytest.raises(AssertionError, match='not sorted'):
        M.read_val(tmp_path, 'qwen', 0)


def test_renamed_rows_refused_on_read(tmp_path, names, probs):
    path = write(tmp_path, names, probs)
    renamed = list(names)
    renamed[-1] = 'zzz_other_painting'                  # still sorted and unique, but not the exported names
    _replace_npz(path, renamed, probs)
    with pytest.raises(AssertionError, match='names hash'):
        M.read_val(tmp_path, 'qwen', 0)
