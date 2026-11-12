"""Tests of r29_captions on small synthetic stores (index.json + txt.npy, scripts/extract_features.py's format)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r29_captions as C  # noqa: E402

W = 6
# Extraction IDs deliberately not in row order and painting groups interleaved: a position join would be wrong.
IDS = [40, 7, 12, 3, 25, 18, 31]
NAMES = ['b', 'a', 'b', 'c', 'a', 'b', 'c']


def make_store(path, ids=IDS, names=NAMES, seed=0, width=W):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(len(ids), width)).astype(np.float16)
    path.mkdir(parents=True, exist_ok=True)
    texts = [dict(sample_id=i, painting=n, img=0) for i, n in zip(ids, names)]
    (path / 'index.json').write_text(json.dumps(dict(dim=width, n_texts=len(ids), texts=texts, images=[])))
    np.save(path / 'txt.npy', x)
    return x


@pytest.fixture
def store(tmp_path, monkeypatch):
    x = make_store(tmp_path / 'sig')
    monkeypatch.setitem(C.SOURCES, 'toy', (tmp_path / 'sig', W))
    return x


def test_caption_rows_join_by_sample_id_not_position(store):
    ids, paintings = [3, 40, 18], ['c', 'b', 'b']      # asked in an order unlike the store's
    got = C.caption_rows('toy', ids, paintings)
    assert got.dtype == np.float32 and got.shape == (3, W)
    np.testing.assert_array_equal(got, store[[3, 0, 5]].astype(np.float32))


def test_caption_rows_wrong_painting_raises(store):
    with pytest.raises(AssertionError, match='painting'):
        C.caption_rows('toy', [3, 40], ['c', 'a'])


def test_caption_rows_missing_id_raises(store):
    with pytest.raises(AssertionError, match='missing'):
        C.caption_rows('toy', [3, 999], ['c', 'c'])


def test_painting_means_in_given_order_mean_then_l2(store):
    names = ['c', 'a', 'b']                              # not the store's order of first appearance
    got = C.painting_means('toy', names)
    x = store.astype(np.float64)
    # independent expectation: members by ascending extraction ID, mean of raw rows, then L2
    members = {'a': [7, 25], 'b': [12, 18, 40], 'c': [3, 31]}
    row = {i: k for k, i in enumerate(IDS)}
    want = np.stack([x[[row[i] for i in members[n]]].mean(0) for n in names])
    want = (want / np.linalg.norm(want, axis=1, keepdims=True)).astype(np.float32)
    assert got.dtype == np.float32 and got.shape == (3, W)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(got, axis=1), 1.0, atol=1e-6)


def test_painting_means_is_the_cts_rule_bit_for_bit(store):
    """The CT's rule (`apc_data.paint_mean` then `l2`) on rows ordered by painting then extraction ID."""
    from apc_data import l2, paint_mean
    names = ['a', 'b', 'c']
    order = sorted(range(len(IDS)), key=lambda k: (names.index(NAMES[k]), IDS[k]))
    pid = np.asarray([names.index(NAMES[k]) for k in order])
    want = l2(paint_mean(store[order].astype(np.float32), pid, 3))
    np.testing.assert_array_equal(C.painting_means('toy', names), want)


def test_painting_means_unknown_or_repeated_painting_raises(store):
    with pytest.raises(AssertionError, match='no captions'):
        C.painting_means('toy', ['a', 'zzz'])
    with pytest.raises(AssertionError, match='repeated'):
        C.painting_means('toy', ['a', 'a'])


def test_store_checks_width_and_row_count(tmp_path, monkeypatch):
    make_store(tmp_path / 's')
    monkeypatch.setitem(C.SOURCES, 'wide', (tmp_path / 's', W + 2))
    with pytest.raises(AssertionError):
        C.caption_rows('wide', [3], ['c'])
    make_store(tmp_path / 'dup', ids=[1, 1, 2, 3, 4, 5, 6])
    monkeypatch.setitem(C.SOURCES, 'dup', (tmp_path / 'dup', W))
    with pytest.raises(AssertionError, match='duplicate'):
        C.caption_rows('dup', [2], ['b'])


def test_store_hash_is_sha256_of_index_bytes(store, tmp_path):
    import hashlib
    want = hashlib.sha256((tmp_path / 'sig' / 'index.json').read_bytes()).hexdigest()
    assert C.store_hash('toy') == want


def test_qwen_has_no_caption_rows(store):
    with pytest.raises(AssertionError):
        C.caption_rows('qwen', [3], ['c'])


# --- gate (iii) ------------------------------------------------------------------------------------------------

def _stores():
    q_ids = np.asarray(IDS)
    return dict(sig_ids=q_ids, sig_names=np.asarray(NAMES), qwen_ids=q_ids[::-1].copy(),
                qwen_names=np.asarray(NAMES)[::-1].copy(),
                roles=dict(train=(np.asarray([40, 12, 7, 25]), np.asarray(['b', 'b', 'a', 'a'])),
                           val=(np.asarray([3, 31]), np.asarray(['c', 'c'])),
                           held=(np.asarray([18]), np.asarray(['b']))))


def test_gate_iii_checks_pass_on_matching_stores():
    out = C.gate_iii_checks(**_stores())
    assert out['status'] == 'pass', out
    assert out['details']['n_captions'] == len(IDS)
    assert out['details']['roles'] == dict(train=4, val=2, held=1)


def test_gate_iii_checks_fail_on_painting_count_difference():
    s = _stores()
    s['qwen_names'] = s['qwen_names'].copy()
    s['qwen_names'][0] = 'a'                 # one Qwen caption under another painting: the counts and the join differ
    out = C.gate_iii_checks(**s)
    assert out['status'] == 'fail'


def test_gate_iii_checks_fail_on_uncovered_caption():
    s = _stores()
    s['roles']['held'] = (np.asarray([999]), np.asarray(['b']))
    assert C.gate_iii_checks(**s)['status'] == 'fail'


def test_gate_iii_checks_fail_on_wrong_role_painting():
    s = _stores()
    s['roles']['val'] = (np.asarray([3, 31]), np.asarray(['c', 'a']))
    assert C.gate_iii_checks(**s)['status'] == 'fail'


def test_write_gate_record(tmp_path):
    C.write_gate_record(tmp_path, dict(status='pass', details={'x': 1}))
    rec = json.loads((tmp_path / 'iii.json').read_text())
    assert rec['gate'] == 'iii' and rec['status'] == 'pass' and rec['tries'] == 1 and rec['details'] == {'x': 1}
    assert len(rec['written']) == 16
    C.write_gate_record(tmp_path, dict(status='fail', details={}))
    assert json.loads((tmp_path / 'iii.json').read_text())['tries'] == 2
