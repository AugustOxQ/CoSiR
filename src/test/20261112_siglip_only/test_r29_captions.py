"""Tests of r29_captions on small synthetic stores (index.json + txt.npy, scripts/extract_features.py's format)."""
import hashlib
import json
import os
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


BACKBONE = 'toy_backbone'


def make_store(path, ids=IDS, names=NAMES, seed=0, width=W, backbone=BACKBONE, smoke=False, txt_done=None, gate_ok=True,
               bad_sha=False):
    """A store as scripts/extract_features.py seals it: index.json, txt.npy, progress.json (with the index's sha256)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(len(ids), width)).astype(np.float16)
    path.mkdir(parents=True, exist_ok=True)
    texts = [dict(sample_id=i, painting=n, img=0) for i, n in zip(ids, names)]
    raw = json.dumps(dict(backbone=backbone, smoke=smoke, dim=width, n_texts=len(ids), texts=texts, images=[]))
    (path / 'index.json').write_text(raw)
    np.save(path / 'txt.npy', x)
    sha = hashlib.sha256(raw.encode()).hexdigest()
    (path / 'progress.json').write_text(json.dumps(dict(
        txt_done=len(ids) if txt_done is None else txt_done, gate=dict(ok=gate_ok),
        index_sha256='0' * 64 if bad_sha else sha)))
    return x


def annotations(ids=IDS, names=NAMES, extra=None):
    """Annotation painting of every sample ID (row i = ID i): the store's captions plus unrelated ones in the gaps."""
    ann = np.asarray(['other'] * (max(ids) + 1), dtype=object)
    ann[list(ids)] = names
    for i, n in (extra or {}).items():
        ann[i] = n
    return ann.astype(str)


@pytest.fixture
def store(tmp_path, monkeypatch):
    x = make_store(tmp_path / 'sig')
    monkeypatch.setitem(C.SOURCES, 'toy', (tmp_path / 'sig', W))
    monkeypatch.setitem(C.BACKBONE, 'toy', BACKBONE)
    monkeypatch.setattr(C, 'annotation_paintings', lambda: annotations())
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


def _toy(tmp_path, monkeypatch, name, **kw):
    make_store(tmp_path / name, **kw)
    monkeypatch.setitem(C.SOURCES, name, (tmp_path / name, W))
    monkeypatch.setitem(C.BACKBONE, name, BACKBONE)


def test_painting_means_store_missing_a_caption_raises(tmp_path, monkeypatch):
    keep = [k for k, i in enumerate(IDS) if i != 18]                      # the store lacks caption 18 of painting b
    _toy(tmp_path, monkeypatch, 'miss', ids=[IDS[k] for k in keep], names=[NAMES[k] for k in keep])
    monkeypatch.setattr(C, 'annotation_paintings', lambda: annotations())
    with pytest.raises(AssertionError, match='missing'):
        C.painting_means('miss', ['a', 'b', 'c'])


def test_painting_means_caption_relabelled_to_another_painting_raises(tmp_path, monkeypatch):
    names = list(NAMES)
    names[IDS.index(18)] = 'a'                                           # annotation says b, the store says a
    _toy(tmp_path, monkeypatch, 'relab', names=names)
    monkeypatch.setattr(C, 'annotation_paintings', lambda: annotations())
    with pytest.raises(AssertionError, match='painting'):
        C.painting_means('relab', ['a', 'b', 'c'])


def test_painting_means_extra_store_caption_of_a_wanted_painting_raises(tmp_path, monkeypatch):
    _toy(tmp_path, monkeypatch, 'extra', ids=IDS + [99], names=NAMES + ['a'])    # annotations have no caption 99 for a
    monkeypatch.setattr(C, 'annotation_paintings', lambda: annotations())
    with pytest.raises(AssertionError, match='extra'):
        C.painting_means('extra', ['a', 'b', 'c'])
    # an extra caption of a painting that is not asked for is fine
    C.painting_means('extra', ['b', 'c'])


def test_painting_means_asks_only_for_the_given_paintings(store):
    got = C.painting_means('toy', ['b'])
    assert got.shape == (1, W)


def _read_store(tmp_path, monkeypatch, name='bad', **kw):
    _toy(tmp_path, monkeypatch, name, **kw)
    return C.caption_rows(name, [3], ['c'])


def test_store_check_accepts_a_sealed_store(tmp_path, monkeypatch):
    assert _read_store(tmp_path, monkeypatch).shape == (1, W)


@pytest.mark.parametrize('kw, match', [
    (dict(smoke=True), 'smoke'),
    (dict(backbone='some_other_backbone'), 'backbone'),
    (dict(txt_done=3), 'txt_done'),
    (dict(gate_ok=False), 'gate'),
    (dict(bad_sha=True), 'index_sha256'),
])
def test_store_check_refuses_unsealed_or_foreign_store(tmp_path, monkeypatch, kw, match):
    with pytest.raises(AssertionError, match=match):
        _read_store(tmp_path, monkeypatch, **kw)


def test_store_cache_follows_a_rewritten_index(tmp_path, monkeypatch):
    _toy(tmp_path, monkeypatch, 'rw')
    assert C.caption_rows('rw', [3], ['c']).shape == (1, W)
    old = (tmp_path / 'rw' / 'index.json').stat()
    make_store(tmp_path / 'rw', ids=IDS[:-1], names=NAMES[:-1])           # same folder, one caption fewer (ID 31 gone)
    os.utime(tmp_path / 'rw' / 'index.json', ns=(old.st_atime_ns, old.st_mtime_ns))   # even an unchanged mtime: size differs
    with pytest.raises(AssertionError, match='missing'):
        C.caption_rows('rw', [31], ['c'])


def test_every_real_source_has_a_backbone():
    assert set(C.BACKBONE) == {k for k, (f, _) in C.SOURCES.items() if f is not None}


def test_store_checks_width_and_row_count(tmp_path, monkeypatch):
    make_store(tmp_path / 's')
    monkeypatch.setitem(C.SOURCES, 'wide', (tmp_path / 's', W + 2))
    monkeypatch.setitem(C.BACKBONE, 'wide', BACKBONE)
    with pytest.raises(AssertionError):
        C.caption_rows('wide', [3], ['c'])
    make_store(tmp_path / 'dup', ids=[1, 1, 2, 3, 4, 5, 6])
    monkeypatch.setitem(C.SOURCES, 'dup', (tmp_path / 'dup', W))
    monkeypatch.setitem(C.BACKBONE, 'dup', BACKBONE)
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


def test_gate_iii_does_not_write_a_record_unless_asked():
    import inspect
    assert inspect.signature(C.gate_iii).parameters['write'].default is False
