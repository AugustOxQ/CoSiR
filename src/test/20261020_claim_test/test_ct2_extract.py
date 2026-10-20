"""CPU extraction plumbing and old-store parity with deterministic tiny encoders."""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

import ct2_extract as ct


def features(rows, dim=2048, dtype=np.float16):
    x = np.zeros((len(rows), dim), dtype=dtype)
    x[np.arange(len(rows)), np.asarray(rows) % dim] = 1
    return x


class Image:
    def __init__(self, row):
        self.row = row

    def close(self):
        pass


class Encoder:
    def encode_images(self, images, batch_size):
        assert batch_size == 32
        return features([im.row for im in images], dtype=np.float32)

    def encode_texts(self, texts, batch_size):
        assert batch_size == 128
        return features([int(text) for text in texts], dtype=np.float32)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    from scripts import extract_features
    monkeypatch.setattr(ct.data, 'ASPECT', tmp_path / 'old')
    monkeypatch.setattr(ct.data, 'ROBERTA', tmp_path / 'old_roberta')
    # Non-positional old IDs ensure the parity gate joins to the right rows.
    old_ids = np.arange(32, dtype=np.int64)[::-1]
    old_names = np.asarray([f'p{i:03}' for i in old_ids])
    for key in ('emotion_focus', 'style_word', 'genre_word'):
        path = ct.data.ASPECT / key
        path.mkdir(parents=True)
        np.savez(path / 'ids.npz', image_painting_ids=old_names,
                 sample_ids=old_ids, caption_painting_ids=old_names)
        np.save(path / 'img.npy', features(old_ids))
        np.save(path / 'txt.npy', features(old_ids))
        (path / 'done.json').write_text(json.dumps(dict(instruction=ct.PROMPTS[key])))
    ct.data.ROBERTA.mkdir()
    np.savez(ct.data.ROBERTA / 'ids.npz', sample_ids=old_ids, painting_ids=old_names)
    np.save(ct.data.ROBERTA / 'txt.npy', features(old_ids, 768, np.float32))
    missing_ids = np.asarray([35, 40, 47], dtype=np.int64)
    names = np.asarray([f'p{i:03}' for i in missing_ids])
    missing = dict(sample_ids=missing_ids, painting_names=names,
                   train=dict(sample_ids=missing_ids[:2], painting_names=names[:2]),
                   val=dict(sample_ids=missing_ids[2:], painting_names=names[2:]))
    calls = []

    def query(paths):
        calls.append(paths)
        return missing

    monkeypatch.setattr(ct.data, 'missing_ids', query)

    def images(names):
        # Different adapter order, as in the real name-joined image_spec.
        order = names[::-1]
        spec = SimpleNamespace(images=[dict(painting=n) for n in order],
                               load_image=lambda row: Image(int(order[row][1:])))
        return spec, np.asarray([order.index(n) for n in names])

    def captions(*, sample_ids, expected):
        assert expected is None
        return SimpleNamespace(texts=[str(i) for i in sample_ids], text_meta=[
            dict(sample_id=int(i), painting=f'p{i:03}') for i in sample_ids])

    monkeypatch.setattr(ct, 'image_spec', images)
    monkeypatch.setattr(extract_features, 'adapter_artelingo_full', captions)
    monkeypatch.setattr(ct, 'load_encoder', lambda store, device: (
        (None, None, device) if store == 'roberta' else Encoder(),
        dict(model_id='fake', model_revision='a' * 40)))
    import pc_roberta
    monkeypatch.setattr(pc_roberta, 'encode', lambda texts, *args: features(
        [int(text) for text in texts], 768, np.float32))
    return missing, calls


@pytest.mark.parametrize('store', ct.STORES)
def test_extract_ids_order_shape_dtype_seal_and_refusal(tmp_path, monkeypatch, setup, store):
    missing, calls = setup
    dest = tmp_path / 'smoke'
    path = ct.store_path(store, dest)
    pending = path.with_name(path.name + '.pending')
    original = ct.encode_batches

    def batches(key, enc, ids):
        if len(next(iter(ids.values()))) != 32:
            assert pending.is_dir() and not path.exists()
        yield from original(key, enc, ids)

    monkeypatch.setattr(ct, 'encode_batches', batches)
    assert ct.main(['--store', store, '--device', 'cpu', '--limit', '2', '--dest', str(dest)]) == path
    assert calls == [ct.old_paths(store)]
    assert not pending.exists()
    image = store.endswith('_img')
    with np.load(path / 'ids.npz', allow_pickle=False) as z:
        key = 'painting_names' if image else 'sample_ids'
        assert np.array_equal(z[key], missing[key][:2])
        if not image:
            name_key = 'painting_ids' if store == 'roberta' else 'caption_painting_ids'
            assert z[name_key].tolist() == ['p035', 'p040']
    values = np.load(path / ('img.npy' if image else 'txt.npy'))
    assert values.shape == (2, 768 if store == 'roberta' else 2048)
    assert values.dtype == (np.float32 if store == 'roberta' else np.float16)
    assert values.argmax(axis=1).tolist() == [35, 40]
    done = json.loads((path / 'done.json').read_text())
    primary = missing[key][:2]
    assert done['ids_sha256'] == hashlib.sha256('\n'.join(map(str, primary)).encode()).hexdigest()
    assert done['parity_gate']['n'] == 32 and done['parity_gate']['passed']
    assert done['count'] == 2 and done['counts'] == dict(train=2, val=0)
    assert done['model_revision'] == 'a' * 40 and done['items_per_second'] > 0 and done['smoke']
    before = {p.name: p.read_bytes() for p in path.iterdir()}
    monkeypatch.setattr(ct, 'load_encoder', lambda *a: pytest.fail('existing store constructed encoder'))
    with pytest.raises(AssertionError, match='existing store'):
        ct.extract(store, 'cpu', 2, dest)
    assert before == {p.name: p.read_bytes() for p in path.iterdir()}


@pytest.mark.parametrize('store', ct.STORES)
def test_parity_failure_never_seals(tmp_path, monkeypatch, setup, store):
    original = ct.encode_batches

    def damaged(*args):
        for start, x in original(*args):
            x = x.copy()
            # Only one bad row: a mean-only cosine gate would incorrectly pass.
            if store == 'roberta':
                x[0, 0] += .002
            else:
                x[0] *= .998
                x[0, 500] = np.sqrt(1 - .998 ** 2)
            yield start, x

    monkeypatch.setattr(ct, 'encode_batches', damaged)
    dest = tmp_path / 'smoke'
    with pytest.raises(AssertionError, match='parity failed'):
        ct.extract(store, 'cpu', 2, dest)
    assert not ct.store_path(store, dest).exists()


def test_full_counts_pending_refusal_and_failure(tmp_path, monkeypatch, setup):
    dest = tmp_path / 'smoke'
    path = ct.extract('roberta', 'cpu', dest=dest)
    done = json.loads((path / 'done.json').read_text())
    assert done['count'] == 3 and done['counts'] == dict(train=2, val=1) and not done['smoke']
    path = ct.store_path('emotion_focus_img', dest)
    pending = path.with_name(path.name + '.pending')
    pending.mkdir()
    (pending / 'partial').write_text('keep')
    with pytest.raises(AssertionError, match='pending extraction'):
        ct.extract('emotion_focus_img', 'cpu', dest=dest)
    assert (pending / 'partial').read_text() == 'keep'
    original = ct.encode_batches

    def fail(store, enc, ids):
        if len(next(iter(ids.values()))) == 32:
            yield from original(store, enc, ids)
        else:
            raise RuntimeError('interrupted')

    monkeypatch.setattr(ct, 'encode_batches', fail)
    with pytest.raises(RuntimeError, match='interrupted'):
        ct.extract('style_word_img', 'cpu', dest=dest)
    path = ct.store_path('style_word_img', dest)
    assert not path.exists() and path.with_name(path.name + '.pending').exists()


def test_limit_requires_isolated_root(setup):
    with pytest.raises(AssertionError, match='isolated --dest'):
        ct.extract('roberta', 'cpu', limit=2)


def test_qwen_loads_model_and_processor_at_frozen_revision(monkeypatch):
    from src.data import feature_extract
    from transformers import AutoModel, Qwen3VLProcessor
    calls = []

    class Model:
        config = SimpleNamespace(_commit_hash=ct.QWEN_REVISION)

        def to(self, device):
            assert device == 'cpu'
            return self

        def eval(self):
            return self

    def model(repo, **kwargs):
        calls.append(('model', repo, kwargs))
        return Model()

    def processor(repo, **kwargs):
        calls.append(('processor', repo, kwargs))
        return SimpleNamespace(image_processor=SimpleNamespace())

    monkeypatch.setattr(AutoModel, 'from_pretrained', model)
    monkeypatch.setattr(Qwen3VLProcessor, 'from_pretrained', processor)
    enc, identity = ct.load_encoder('emotion_focus_img', 'cpu')
    assert enc.instruction == ct.PROMPTS['emotion_focus']
    assert identity == dict(model_id=feature_extract.QWEN_REPO, model_revision=ct.QWEN_REVISION)
    assert [kind for kind, _, _ in calls] == ['model', 'processor']
    for _, repo, kwargs in calls:
        assert repo == feature_extract.QWEN_REPO
        assert kwargs['revision'] == ct.QWEN_REVISION and kwargs['local_files_only']
    assert AutoModel.from_pretrained is model and Qwen3VLProcessor.from_pretrained is processor


@pytest.mark.parametrize('store,count,lengths', [
    ('emotion_focus_img', 33, [32, 1]),
    ('emotion_focus_txt', 4097, [4096, 1]),
    ('roberta', 257, [256, 1])])
def test_unchanged_extraction_chunks(setup, store, count, lengths):
    rows = np.arange(count, dtype=np.int64)
    names = np.asarray([f'p{i:03}' for i in rows])
    ids = dict(painting_names=names) if store.endswith('_img') else dict(
        sample_ids=rows, **{'painting_ids' if store == 'roberta' else 'caption_painting_ids': names})
    enc = (None, None, 'cpu') if store == 'roberta' else Encoder()
    batches = list(ct.encode_batches(store, enc, ids))
    assert [len(x) for _, x in batches] == lengths
    assert [offset for offset, _ in batches] == [0, lengths[0]]
    assert np.concatenate([x for _, x in batches]).argmax(axis=1).tolist() == (
        rows % (768 if store == 'roberta' else 2048)).tolist()
