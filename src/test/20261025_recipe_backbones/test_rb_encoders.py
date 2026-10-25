import json

import numpy as np
import pytest

import rb_paths
import rb_encoders


def make_store(root, dataset, backbone, names, dim, seed=0, texts=None, **index):
    """A generic store root/dataset/backbone: img.npy float16 and index.json (images, texts) as the extractor writes."""
    path = root / dataset / backbone
    path.mkdir(parents=True)
    x = np.random.default_rng(seed).normal(size=(len(names), dim)).astype(np.float16)
    np.save(path / 'img.npy', x)
    head = dict(dataset=dataset, backbone=backbone, smoke=False, dtype='float16', dim=dim, n_images=len(names))
    head.update(index)
    head['images'] = [{'painting': n, 'image': f'images/{n}.jpg'} for n in names]
    head['texts'] = texts or []
    head['n_texts'] = len(head['texts'])
    (path / 'index.json').write_text(json.dumps(head))
    return x


def expected(x, rows):
    """float32 store row divided by its float32 norm, written out independently of the module."""
    v = x[rows].astype(np.float32)
    return v / np.sqrt((v * v).sum(axis=1, keepdims=True))


def test_read_store_joins_by_name_in_caller_order(tmp_path):
    names = ['b', 'a', 'd', 'c']
    x = make_store(tmp_path, 'artelingo_full', 's', names, 8)
    got = rb_encoders.read_store(tmp_path / 'artelingo_full' / 's', 8, ['c', 'b', 'a'], 4)
    assert got.dtype == np.float32 and got.shape == (3, 8)
    np.testing.assert_allclose(got, expected(x, [3, 0, 1]), rtol=1e-6)


@pytest.fixture
def store(tmp_path):
    names = ['b', 'a', 'd', 'c']
    make_store(tmp_path, 'artelingo_full', 's', names, 8)
    return tmp_path / 'artelingo_full' / 's'


def test_read_store_fails_loudly(store):
    with pytest.raises(AssertionError, match='duplicate requested'):
        rb_encoders.read_store(store, 8, ['a', 'a'], 4)
    with pytest.raises(AssertionError, match='missing'):
        rb_encoders.read_store(store, 8, ['a', 'zzz'], 4)
    with pytest.raises(AssertionError):
        rb_encoders.read_store(store, 8, ['a'], 5)
    with pytest.raises(AssertionError):
        rb_encoders.read_store(store, 16, ['a'], 4)


def test_read_store_rejects_bad_payload(store):
    x = np.load(store / 'img.npy')
    x[1, 2] = np.inf
    np.save(store / 'img.npy', x)
    with pytest.raises(AssertionError, match='nonfinite'):
        rb_encoders.read_store(store, 8, ['a'], 4)
    x[1, 2] = 0
    x[1] = 0
    np.save(store / 'img.npy', x)
    with pytest.raises(AssertionError, match='zero'):
        rb_encoders.read_store(store, 8, ['a'], 4)
    rb_encoders.read_store(store, 8, ['b', 'd'], 4)


def test_read_store_rejects_duplicate_stored_names(tmp_path):
    make_store(tmp_path, 'artelingo_full', 's', ['a', 'a', 'b'], 8)
    with pytest.raises(AssertionError, match='duplicate painting'):
        rb_encoders.read_store(tmp_path / 'artelingo_full' / 's', 8, ['b'], 3)


def test_read_store_rejects_smoke_store(tmp_path):
    make_store(tmp_path, 'artelingo_full', 's', ['a', 'b'], 8, smoke=True)
    with pytest.raises(AssertionError):
        rb_encoders.read_store(tmp_path / 'artelingo_full' / 's', 8, ['a'], 2)


def test_full_images_reads_each_encoders_store_by_name(tmp_path, monkeypatch):
    names = ['p3', 'p1', 'p2', 'p0']
    xs = {}
    for k, backbone in enumerate(('siglip2_so400m', 'clip_b32', 'dinov2_large')):
        xs[backbone] = make_store(tmp_path, 'artelingo_full', backbone, names, rb_encoders.WIDTH[backbone], seed=k)
    monkeypatch.setattr(rb_encoders, 'FULL_ROOT', tmp_path / 'artelingo_full')
    monkeypatch.setattr(rb_encoders, 'FULL_COUNT', 4)
    for backbone, x in xs.items():
        got = rb_encoders.full_images(backbone, ['p0', 'p3', 'p2'])
        assert got.shape == (3, rb_encoders.WIDTH[backbone]) and got.dtype == np.float32
        np.testing.assert_allclose(got, expected(x, [3, 0, 2]), rtol=1e-6)
    with pytest.raises(AssertionError):
        rb_encoders.full_images('resnet50', ['p0'])


def test_full_qwen_goes_through_the_ct_loader(monkeypatch):
    calls = []
    sentinel = np.zeros((2, 2048), np.float32)

    def fake(s):
        calls.append(s)
        return sentinel

    monkeypatch.setattr(rb_encoders.ct2_data, 'default_images', fake)
    assert rb_encoders.full_images('qwen3vl_emb_2b', ['x', 'y']) is sentinel
    assert len(calls) == 1 and list(calls[0]['all_painting_ids']) == ['x', 'y']


# ---- ArtELingo-28 ----

A28 = ['zed_one', 'amy_two', 'moe_three', 'kim_four', 'bob_five']   # unsorted on purpose


def sha256_of(names):
    import hashlib
    return hashlib.sha256('\n'.join(names).encode('utf-8')).hexdigest()


@pytest.fixture
def a28(tmp_path, monkeypatch):
    """Fake read-set parquet (rows in file order, row_id ascending) and A28 stores in sorted-name order."""
    import pandas as pd
    paintings = ['amy_two', 'amy_two', 'bob_five', 'kim_four', 'zed_one', 'moe_three', 'zed_one']
    frame = pd.DataFrame({'row_id': [5, 6, 7, 9, 10, 11, 12], 'painting': paintings, 'language': 'en',
                          'emotion': 0, 'style': 0, 'genre': 0, 'held': True, 'dropped': False})
    frame.to_parquet(tmp_path / 'read_set_a28.parquet')
    texts = [dict(row_id=int(r), painting=p, img=0) for r, p in zip(frame.row_id, frame.painting)]
    srt = sorted(A28)
    for k, backbone in enumerate(('siglip2_so400m', 'clip_b32', 'dinov2_large')):
        make_store(tmp_path, 'artelingo28', backbone, srt, rb_encoders.WIDTH[backbone], seed=k, texts=texts)
    for attr, value in (('A28_ROOT', tmp_path / 'artelingo28'), ('READ_SET', tmp_path / 'read_set_a28.parquet'),
                        ('A28_COUNT', 5), ('A28_SHA256', sha256_of(srt))):
        monkeypatch.setattr(rb_encoders, attr, value)
    rb_encoders._a28_list.cache_clear()
    return tmp_path


def rewrite_index(root, backbone, edit):
    path = root / 'artelingo28' / backbone / 'index.json'
    index = json.loads(path.read_text())
    edit(index)
    path.write_text(json.dumps(index))
    rb_encoders._a28_list.cache_clear()


def test_a28_names_sorted_and_hash(a28):
    names = rb_encoders.a28_names()
    assert names == sorted(A28) and isinstance(names, list)
    assert rb_encoders.a28_sha256() == sha256_of(sorted(A28))


def test_a28_names_fail_when_store_index_and_parquet_disagree(a28):
    rewrite_index(a28, 'siglip2_so400m', lambda i: i['images'][0].update(painting='other'))
    with pytest.raises(AssertionError):
        rb_encoders.a28_names()


def test_a28_names_fail_when_text_rows_and_parquet_disagree(a28):
    rewrite_index(a28, 'siglip2_so400m', lambda i: i['texts'][2].update(painting='zed_one'))
    with pytest.raises(AssertionError, match='row_id'):
        rb_encoders.a28_names()


def test_a28_names_fail_on_wrong_count_or_hash(a28, monkeypatch):
    monkeypatch.setattr(rb_encoders, 'A28_SHA256', '0' * 64)
    rb_encoders._a28_list.cache_clear()
    with pytest.raises(AssertionError, match='sha256'):
        rb_encoders.a28_names()
    monkeypatch.setattr(rb_encoders, 'A28_SHA256', sha256_of(sorted(A28)))
    monkeypatch.setattr(rb_encoders, 'A28_COUNT', 746)
    rb_encoders._a28_list.cache_clear()
    with pytest.raises(AssertionError):
        rb_encoders.a28_names()


def test_a28_images_in_sorted_name_order_whatever_the_store_order(a28):
    srt = sorted(A28)
    order = ['bob_five', 'amy_two', 'zed_one', 'kim_four', 'moe_three']      # a store not in sorted order
    import shutil
    shutil.rmtree(a28 / 'artelingo28' / 'clip_b32')
    x = make_store(a28, 'artelingo28', 'clip_b32', order, 512, seed=7,
                   texts=json.loads((a28 / 'artelingo28' / 'siglip2_so400m' / 'index.json').read_text())['texts'])
    got = rb_encoders.a28_images('clip_b32')
    assert got.dtype == np.float32 and got.shape == (5, 512)
    np.testing.assert_allclose(got, expected(x, [order.index(n) for n in srt]), rtol=1e-6)
    assert rb_encoders.a28_images('siglip2_so400m').shape == (5, 1152)
    assert rb_encoders.a28_images('dinov2_large').shape == (5, 1024)


def test_a28_images_reject_a_store_of_other_paintings(a28):
    import shutil
    shutil.rmtree(a28 / 'artelingo28' / 'dinov2_large')
    make_store(a28, 'artelingo28', 'dinov2_large', ['amy_two', 'bob_five', 'kim_four', 'moe_three', 'other'], 1024)
    with pytest.raises(AssertionError, match='missing'):
        rb_encoders.a28_images('dinov2_large')


def make_instructed(root, key, names, instruction=None, seed=3, **done):
    import itp_data
    path = root / 'artelingo28' / 'qwen3vl_emb_2b_aspect_prompts' / key
    path.mkdir(parents=True)
    x = np.random.default_rng(seed).normal(size=(len(names), 2048)).astype(np.float16)
    np.save(path / 'img.npy', x)
    np.savez(path / 'ids.npz', painting_names=np.asarray(names))
    import ct2_data
    meta = dict(store=f'{key}_img', instruction=instruction or itp_data.PROMPTS[key], smoke=False,
                model_revision=ct2_data.QWEN_REVISION, count=len(names),
                ids_sha256=ct2_data.ids_sha256({'painting_names': list(names)}))
    meta.update(done)
    (path / 'done.json').write_text(json.dumps(meta))
    return x


def test_a28_instructed_joins_by_name_in_a28_order(a28):
    order = ['kim_four', 'zed_one', 'amy_two', 'moe_three', 'bob_five']
    x = make_instructed(a28, 'emotion_focus', order)
    got = rb_encoders.a28_instructed('emotion_focus')
    assert got.dtype == np.float32 and got.shape == (5, 2048)
    np.testing.assert_allclose(got, expected(x, [order.index(n) for n in sorted(A28)]), rtol=1e-6)


def test_a28_instructed_refuses_unsealed_wrong_or_partial_stores(a28):
    with pytest.raises(AssertionError):
        rb_encoders.a28_instructed('emotion_word')
    make_instructed(a28, 'style_word', sorted(A28), instruction='some other prompt')
    with pytest.raises(AssertionError, match='instruction'):
        rb_encoders.a28_instructed('style_word')
    make_instructed(a28, 'genre_word', sorted(A28)[:4])
    with pytest.raises(AssertionError):
        rb_encoders.a28_instructed('genre_word')


def test_a28_instructed_refuses_wrong_revision_count_or_ids(a28):
    make_instructed(a28, 'emotion_focus', sorted(A28), model_revision='0' * 40)
    with pytest.raises(AssertionError, match='revision'):
        rb_encoders.a28_instructed('emotion_focus')
    make_instructed(a28, 'style_word', sorted(A28), count=len(A28) + 1)
    with pytest.raises(AssertionError):
        rb_encoders.a28_instructed('style_word')
    make_instructed(a28, 'genre_word', sorted(A28), ids_sha256='0' * 64)
    with pytest.raises(AssertionError, match='ids hash'):
        rb_encoders.a28_instructed('genre_word')


# ---- real stores (read only; seconds) ----

def _have(*parts):
    return (rb_paths.PRE.joinpath(*parts) / 'index.json').exists()


def test_real_a28_list_is_the_746_in_both_places():
    import pandas as pd
    names = rb_encoders.a28_names()
    assert len(names) == 746 and names == sorted(set(names))
    assert rb_encoders.a28_sha256() == 'c1f1b27ed2af3057192f7684e25a327ca4be42127eacbe5ab0134f89f6ba2b70'
    parquet = pd.read_parquet(rb_paths.PRE / 'artelingo28' / 'read_set_a28.parquet', columns=['painting'])
    assert names == sorted(parquet.painting.unique())
    for backbone in ('siglip2_so400m', 'qwen3vl_emb_2b', 'clip_b32'):
        index = json.loads((rb_paths.PRE / 'artelingo28' / backbone / 'index.json').read_text())
        assert [row['painting'] for row in index['images']] == names     # store index agrees, same order


@pytest.mark.parametrize('backbone', list(rb_encoders.WIDTH))
def test_real_a28_images(backbone):
    if not _have('artelingo28', backbone):
        pytest.skip(f'{backbone}: no ArtELingo-28 store yet (ticket 08)')
    x = rb_encoders.a28_images(backbone)
    assert x.shape == (746, rb_encoders.WIDTH[backbone]) and x.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(x, axis=1), 1, atol=1e-5)


def test_real_a28_instructed_if_extracted():
    if not (rb_paths.PRE / 'artelingo28' / 'qwen3vl_emb_2b_aspect_prompts' / 'emotion_focus' / 'done.json').exists():
        pytest.skip('instructed A28 store not extracted yet (ticket 08)')
    for key in rb_encoders.A28_KEYS:
        assert rb_encoders.a28_instructed(key).shape == (746, 2048)


def test_real_siglip_full_images_sample():
    index = json.loads((rb_paths.PRE / 'artelingo_full' / 'siglip2_so400m' / 'index.json').read_text())
    names = [row['painting'] for row in index['images']]
    pick = [1000, 5, 61401, 77]
    x = rb_encoders.full_images('siglip2_so400m', [names[i] for i in pick])
    raw = np.load(rb_paths.PRE / 'artelingo_full' / 'siglip2_so400m' / 'img.npy', mmap_mode='r')
    assert len(names) == 61402 and x.shape == (4, 1152)
    np.testing.assert_allclose(x, expected(np.asarray(raw), pick), rtol=1e-6)


def test_real_generic_reader_equals_the_ct_expression_on_qwen_rows():
    """The generic path is the CT's expression: same floats as ct2_data.default_images on a Qwen sample."""
    qwen = rb_encoders.FULL_ROOT / 'qwen3vl_emb_2b'
    index = json.loads((qwen / 'index.json').read_text())
    names = [row['painting'] for row in index['images']]
    sample = [names[i] for i in np.random.default_rng(0).choice(len(names), 300, replace=False)]
    via_ct = rb_encoders.full_images('qwen3vl_emb_2b', sample)
    generic = rb_encoders.read_store(qwen, 2048, sample, rb_encoders.FULL_COUNT)
    assert via_ct.shape == generic.shape == (300, 2048)
    assert np.array_equal(via_ct, generic)
