"""Readers join by stored IDs on small synthetic stores (verdict code: never by position)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import po_percept as po
import po_paths as P

W = 4
NAMES = ['b-paint', 'a-paint', 'd-paint', 'c-paint', 'e-paint']        # store order is not sorted
SAMPLE_IDS = [30, 10, 50, 20, 40, 60, 70, 80]                           # store row order of the captions
CAP_PAINT = ['d-paint', 'b-paint', 'a-paint', 'c-paint', 'b-paint', 'e-paint', 'a-paint', 'd-paint']


def split_view(sample_ids, painting_ids, all_names):
    painting_ids = np.asarray(painting_ids)
    first = np.asarray([list(painting_ids).index(n) for n in all_names])
    return dict(sample_ids=np.asarray(sample_ids, dtype=np.int64), painting_ids=painting_ids,
                all_painting_ids=np.asarray(all_names), first=first)


@pytest.fixture
def clip_store(tmp_path):
    rng = np.random.default_rng(0)
    img = rng.normal(size=(len(NAMES), W)).astype(np.float16)
    txt = rng.normal(size=(len(SAMPLE_IDS), W)).astype(np.float16)
    index = dict(backbone='clip_l14', smoke=False, dtype='float16', dim=W, n_images=len(NAMES), n_texts=len(SAMPLE_IDS),
                 images=[dict(painting=n) for n in NAMES],
                 texts=[dict(sample_id=i, painting=p) for i, p in zip(SAMPLE_IDS, CAP_PAINT)])
    (tmp_path / 'index.json').write_text(json.dumps(index))
    np.save(tmp_path / 'img.npy', img)
    np.save(tmp_path / 'txt.npy', txt)
    return tmp_path, img, txt, index


def test_clip_image_rows_follow_requested_names(clip_store):
    root, img, _, _ = clip_store
    wanted = ['c-paint', 'a-paint', 'e-paint']
    s = split_view([1, 2, 3], wanted, wanted)
    x = po.clipl14_image(s, root, n_images=5, n_texts=8, width=W)
    assert x.dtype == np.float32 and x.shape == (3, W)
    for row, name in enumerate(wanted):
        assert np.array_equal(x[row], img[NAMES.index(name)].astype(np.float32))


def test_clip_image_missing_name_and_count_mismatch(clip_store):
    root, *_ = clip_store
    s = split_view([1, 2], ['a-paint', 'zzz'], ['a-paint', 'zzz'])
    with pytest.raises(AssertionError, match='missing'):
        po.clipl14_image(s, root, n_images=5, n_texts=8, width=W)
    s = split_view([1], ['a-paint'], ['a-paint'])
    with pytest.raises(AssertionError):
        po.clipl14_image(s, root, n_images=6, n_texts=8, width=W)


def test_clip_text_rows_follow_sample_ids(clip_store):
    root, _, txt, _ = clip_store
    ids = [20, 70, 30]
    paint = [CAP_PAINT[SAMPLE_IDS.index(i)] for i in ids]
    s = split_view(ids, paint, sorted(set(paint)))
    x = po.clipl14_text(s, root, n_images=5, n_texts=8, width=W)
    for row, i in enumerate(ids):
        assert np.array_equal(x[row], txt[SAMPLE_IDS.index(i)].astype(np.float32))
    bad = split_view(ids, ['a-paint', 'a-paint', 'a-paint'], ['a-paint'])
    with pytest.raises(AssertionError, match='painting mismatch'):
        po.clipl14_text(bad, root, n_images=5, n_texts=8, width=W)
    with pytest.raises(AssertionError, match='missing'):
        po.clipl14_text(split_view([999], ['a-paint'], ['a-paint']), root, n_images=5, n_texts=8, width=W)


def write_modernbert(root, smoke=False, revision=P.MODERNBERT_REVISION, ok=True):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(len(SAMPLE_IDS), W)).astype(np.float32)
    np.save(root / 'txt.npy', x)
    np.savez(root / 'ids.npz', sample_ids=np.asarray(SAMPLE_IDS, dtype=np.int64), painting_ids=np.asarray(CAP_PAINT))
    (root / 'done.json').write_text(json.dumps(dict(
        model_id=P.MODERNBERT_MODEL, revision=revision, count=len(SAMPLE_IDS), shape=[len(SAMPLE_IDS), W],
        dtype='float32', parity=dict(ok=ok), smoke=smoke)))
    return x


def test_modernbert_join_and_identity_checks(tmp_path):
    x = write_modernbert(tmp_path)
    ids = [60, 10, 80]
    paint = [CAP_PAINT[SAMPLE_IDS.index(i)] for i in ids]
    s = split_view(ids, paint, sorted(set(paint)))
    got = po.modernbert(s, tmp_path, n_captions=8, width=W)
    for row, i in enumerate(ids):
        assert np.array_equal(got[row], x[SAMPLE_IDS.index(i)])
    with pytest.raises(AssertionError, match='painting mismatch'):
        po.modernbert(split_view(ids, ['a-paint'] * 3, ['a-paint']), tmp_path, n_captions=8, width=W)
    for kwargs in (dict(smoke=True), dict(revision='0' * 40), dict(ok=False)):
        write_modernbert(tmp_path, **kwargs)
        with pytest.raises(AssertionError):
            po.modernbert(s, tmp_path, n_captions=8, width=W)


def test_patch_store_rows_and_fetch(tmp_path):
    rng = np.random.default_rng(2)
    tokens = rng.normal(size=(len(NAMES), 3, W)).astype(np.float16)
    np.save(tmp_path / 'tokens.npy', tokens)
    np.savez(tmp_path / 'ids.npz', painting_names=np.asarray(NAMES))
    (tmp_path / 'done.json').write_text(json.dumps(dict(
        model_id=P.CLIP_L14_MODEL, revision=P.CLIP_L14_REVISION, count=len(NAMES), shape=[len(NAMES), 3, W],
        dtype='float16', parity_gate=dict(ok=True), smoke=False)))
    store = po.PatchStore(tmp_path, n=len(NAMES), tokens=3, width=W)
    wanted = ['e-paint', 'b-paint', 'c-paint']
    rows = store.rows(wanted)
    block = store.fetch(rows)
    for r, name in enumerate(wanted):
        assert np.array_equal(block[r], tokens[NAMES.index(name)])
    with pytest.raises(AssertionError, match='missing'):
        store.rows(['nope'])
    dev = po.load_tokens_to_device(store, rows, 'cpu', chunk=2)
    assert dev.dtype == po.torch.float16 and np.array_equal(dev.numpy(), block)
    with pytest.raises(AssertionError):
        po.PatchStore(tmp_path, n=len(NAMES) + 1, tokens=3, width=W)
