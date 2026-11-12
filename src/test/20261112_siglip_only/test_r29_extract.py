"""CPU tests using the production text path with a tiny tokenizer/model."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile

import numpy as np
import pytest
import torch

import r29_extract as extract
from src.data.feature_extract import ClipL14, SiglipSo400m, Siglip2So400m


@pytest.fixture
def work():
    # Keep even pytest's test data inside the authorized worktree; delete nothing.
    root = extract.paths.REPO / '.scratch/r29-t02-tests'
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(dir=root))


class Tokens(dict):
    def to(self, device):
        return self


class Tokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        ids = [[sum(word.encode()) % 100 + 1 for word in text.split()] + [101] for text in texts]
        if kwargs.get('return_tensors') == 'pt':
            limit = kwargs['max_length']
            ids = [row[:limit] for row in ids]
            width = limit if kwargs['padding'] == 'max_length' else max(map(len, ids))
            return Tokens(input_ids=torch.tensor([row + [0] * (width - len(row)) for row in ids]))
        assert kwargs == dict(truncation=False, add_special_tokens=True)
        return dict(input_ids=ids)


class Model:
    def get_text_features(self, input_ids):
        ids = input_ids.float()
        markers = [sum(word.encode()) % 100 + 1 for word in ('represent', 'Represent')]
        prefixed = (ids[:, 0] == markers[0]) | (ids[:, 0] == markers[1])
        # A prefix rotates the direction substantially, making a wrong replay fail.
        return torch.stack([torch.ones(len(ids)), 2 * prefixed.float(), ids.sum(1) % 17 / 100], dim=1)


def encoder(cls=SiglipSo400m):
    # Exercise the actual encode_texts method, including normalization and padding.
    enc = object.__new__(cls)
    enc.dim, enc.device = 3, 'cpu'
    enc.p, enc.m = SimpleNamespace(tokenizer=Tokenizer()), Model()
    return enc


def reference(n=3):
    return dict(dataset='artelingo_full', n_texts=n, n_images=2,
                images=[dict(painting='a', image='a.jpg'), dict(painting='b', image='b.jpg')],
                texts=[dict(sample_id=1000 + i, painting='ab'[i % 2], img=i % 2) for i in range(n)])


def write_store(work, *, n=3, cls=SiglipSo400m, prefix=extract.paths.PREFIX):
    ref, enc, folder = reference(n), encoder(cls), work / 'store'
    captions = ['LOUD Red ' + str(i) for i in range(n)]
    extract.extract_store(ref, captions, enc, folder, prefix=prefix)
    return folder, ref, enc, captions


def reseal(folder, change):
    index = extract._json(folder / 'index.json')
    change(index)
    extract._write(folder / 'index.json', index)
    progress = extract._json(folder / 'progress.json')
    progress['index_sha256'] = extract._sha(folder / 'index.json')
    extract._write(folder / 'progress.json', progress)


@pytest.mark.parametrize('cls', [SiglipSo400m, Siglip2So400m, ClipL14])
def test_prefix_once_before_encoder_normalization(work, cls):
    folder, ref, enc, captions = write_store(work, cls=cls)
    calls = enc.p.tokenizer.calls
    expected = [extract.paths.PREFIX + caption for caption in captions]
    if cls is not ClipL14:
        expected = [text.lower() for text in expected]
    assert calls[0][0] == calls[1][0] == expected
    assert all(text.lower().count(extract.paths.PREFIX.lower()) == 1 for text in calls[1][0])
    assert calls[1][1]['truncation'] is True
    assert calls[1][1]['max_length'] == (77 if cls is ClipL14 else 64)
    assert calls[1][1]['padding'] == (True if cls is ClipL14 else 'max_length')
    index, progress, rows = extract._read(folder)
    assert index['prefix'] == (extract.paths.PREFIX if cls is ClipL14 else extract.paths.PREFIX.lower())
    assert index['backbone'] == enc.name and index['smoke'] is False
    assert index['dim'] == 3 and index['n_texts'] == len(captions)
    assert index['texts'] == ref['texts'] and index['images'] == ref['images']
    assert index['text_only'] is True and not (folder / 'img.npy').exists()
    assert index['truncation_limit'] == calls[1][1]['max_length']
    assert index['n_texts_truncated'] == progress['n_truncated'] == 0
    assert progress['txt_done'] == len(captions) and progress['gate']['ok'] is True
    assert progress['index_sha256'] == hashlib.sha256((folder / 'index.json').read_bytes()).hexdigest()
    assert rows.dtype == np.float16 and rows.shape == (3, 3)


@pytest.mark.parametrize('cls,limit', [(SiglipSo400m, 64), (Siglip2So400m, 64), (ClipL14, 77)])
def test_count_special_tokens_strictly_over_limit_and_keep_every_row(work, cls, limit):
    enc, ref = encoder(cls), reference(3)
    # One special token: exactly at limit, over by one, short. Never drop a row.
    texts = ['x ' * (limit - 1), 'x ' * limit, 'short']
    extract.extract_store(ref, texts, enc, work / 'store', prefix='')
    index, progress, rows = extract._read(work / 'store')
    assert index['n_texts_truncated'] == progress['n_truncated'] == 1
    assert rows.shape == (3, 3) and index['texts'] == ref['texts']


def test_join_by_id_accepts_permuted_reference(work):
    folder, ref, _, _ = write_store(work)
    assert extract.check_store(folder, dict(ref, texts=ref['texts'][::-1]), expected_prefix=extract.paths.PREFIX)['ok']


def test_wrong_painting_is_rejected_after_resealing(work):
    folder, ref, _, _ = write_store(work)
    reseal(folder, lambda index: index['texts'][0].update(painting='wrong'))
    with pytest.raises(AssertionError, match='painting mismatch'):
        extract.check_store(folder, ref, expected_prefix=extract.paths.PREFIX)


@pytest.mark.parametrize('bad', [0.0, 1.002, 2.0, float('nan'), float('inf')])
def test_bad_norm_or_nonfinite_row_is_rejected(work, bad):
    folder, ref, _, _ = write_store(work)
    rows = np.load(folder / 'txt.npy', mmap_mode='r+')
    rows[0] = [bad, 0, 0]
    rows.flush()
    with pytest.raises(AssertionError, match='finite/norm'):
        extract.check_store(folder, ref, expected_prefix=extract.paths.PREFIX)


@pytest.mark.parametrize('fault,message', [
    ('duplicate', 'duplicate sample_id'), ('missing', 'sample_id coverage'),
    ('count', 'text row count'), ('truncation', 'n_texts_truncated'), ('smoke', 'smoke store'),
])
def test_manifest_rejections(work, fault, message):
    folder, ref, _, _ = write_store(work)

    def change(index):
        if fault == 'duplicate':
            index['texts'][0]['sample_id'] = index['texts'][1]['sample_id']
        elif fault == 'missing':
            index['texts'][0]['sample_id'] = -7
        elif fault == 'count':
            index['n_texts'] += 1
        elif fault == 'truncation':
            index.pop('n_texts_truncated')
        else:
            index['smoke'] = True

    reseal(folder, change)
    with pytest.raises((AssertionError, KeyError), match=message):
        extract.check_store(folder, ref, expected_prefix=extract.paths.PREFIX)


def test_reject_changed_index_bytes(work):
    folder, ref, _, _ = write_store(work)
    with (folder / 'index.json').open('a') as f:
        f.write('\n')
    with pytest.raises(AssertionError, match='index hash'):
        extract.check_store(folder, ref, expected_prefix=extract.paths.PREFIX)


def test_refuse_nonempty_folder_without_changing_it(work):
    folder, ref, enc, captions = write_store(work)
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    with pytest.raises(AssertionError, match='non-empty output'):
        extract.extract_store(ref, captions, enc, folder)
    assert {p.name: p.read_bytes() for p in folder.iterdir()} == before


def test_annotation_join_checks_painting(work):
    ref = reference(2)
    ref['texts'][0]['sample_id'], ref['texts'][1]['sample_id'] = 1, 0
    annotations = [dict(painting='b', image='b.jpg', caption='Second'),
                   dict(painting='a', image='a.jpg', caption='First')]
    path = work / 'annotations.json'
    path.write_text(json.dumps(annotations))
    assert extract.caption_spec(ref, path).texts == ['First', 'Second']
    annotations[1]['painting'] = 'wrong'
    path.write_text(json.dumps(annotations))
    with pytest.raises(AssertionError, match='alignment'):
        extract.caption_spec(ref, path)


def test_gate_iv_replay_and_failure_records(work):
    folder, ref, enc, captions = write_store(work, n=100)
    plain = work / 'plain'
    extract.extract_store(ref, captions, enc, plain, prefix='')
    record = work / 'records/gates/iv_store.json'
    result = extract.gate_iv(folder, plain, captions, enc, record=record)
    assert set(result) == {'gate', 'status', 'tries', 'details', 'written'}
    assert result['gate'] == 'iv_store' and result['status'] == 'pass' and result['tries'] == 1
    assert 'hash' not in result['details']
    assert result['details']['replay']['min_cosine'] >= 0.999
    assert result['details']['prefixed_replay']['min_cosine'] >= 0.999
    assert len(result['written']) == 16
    rows = np.load(plain / 'txt.npy', mmap_mode='r+')
    rows[0] = [0, 0, 1]
    rows.flush()
    with pytest.raises(AssertionError, match='empty-prefix replay'):
        extract.gate_iv(folder, plain, captions, enc, record=record)
    failed = extract._json(record)
    assert failed['status'] == 'fail' and failed['tries'] == 2


def test_smoke_does_not_write_shared_record(work):
    folder, ref, enc, captions = write_store(work, n=100)
    plain = work / 'plain'
    extract.extract_store(ref, captions, enc, plain, prefix='')
    reseal(folder, lambda index: index.update(smoke=True))
    record = folder / 'records/gates/iv_store.json'
    assert extract.gate_iv(folder, plain, captions, enc, record=record, allow_smoke=True)['status'] == 'pass'
    with pytest.raises(AssertionError, match='smoke records stay'):
        extract.gate_iv(folder, plain, captions, enc, record=work / 'elsewhere.json', allow_smoke=True)


@pytest.mark.parametrize('cls', [SiglipSo400m, Siglip2So400m, ClipL14])
def test_check_store_expected_prefix(work, cls):
    folder, ref, _, _ = write_store(work, cls=cls)
    assert extract.check_store(folder, ref, expected_prefix=extract.paths.PREFIX)['ok']
    with pytest.raises(AssertionError, match='stored prefix'):
        extract.check_store(folder, ref, expected_prefix=None)
    with pytest.raises(AssertionError, match='stored prefix'):
        extract.check_store(folder, ref, expected_prefix='Wrong prefix: ')
    normalized = extract.paths.PREFIX.lower()
    if cls is ClipL14:
        with pytest.raises(AssertionError, match='stored prefix'):
            extract.check_store(folder, ref, expected_prefix=normalized)
    else:
        assert extract.check_store(folder, ref, expected_prefix=normalized)['ok']
    reseal(folder, lambda index: index.pop('prefix'))
    with pytest.raises(AssertionError, match='stored prefix'):
        extract.check_store(folder, ref, expected_prefix=extract.paths.PREFIX)


@pytest.mark.parametrize('legacy', [False, True])
def test_plain_store_expected_prefix_none(work, legacy):
    folder, ref, _, _ = write_store(work, prefix='')
    if legacy:
        reseal(folder, lambda index: index.pop('prefix'))
    assert extract.check_store(folder, ref, expected_prefix=None)['ok']
    with pytest.raises(AssertionError, match='stored prefix'):
        extract.check_store(folder, ref, expected_prefix=extract.paths.PREFIX)


def gate_inputs(work):
    folder, ref, enc, captions = write_store(work, n=100)
    plain = work / 'plain'
    extract.extract_store(ref, captions, enc, plain, prefix='')
    return folder, plain, enc, captions


@pytest.mark.parametrize('target,label', [('plain', 'empty-prefix'), ('prefixed', 'prefixed')])
def test_gate_rejects_cosine_between_point9_and_point999(work, target, label):
    folder, plain, enc, captions = gate_inputs(work)
    rows = np.load((plain if target == 'plain' else folder) / 'txt.npy', mmap_mode='r+')
    x = rows[0].astype(np.float32)
    x /= np.linalg.norm(x)
    orthogonal = np.cross(x, [0, 0, 1])
    orthogonal /= np.linalg.norm(orthogonal)
    rows[0] = 0.95 * x + np.sqrt(1 - 0.95 ** 2) * orthogonal
    rows.flush()
    with pytest.raises(AssertionError, match=f'{label} replay'):
        extract.gate_iv(folder, plain, captions, enc, record=work / 'iv_store.json')


def test_gate_requires_zero_truncated_in_entire_store(work):
    folder, plain, enc, captions = gate_inputs(work)
    reseal(folder, lambda index: index.update(n_texts_truncated=1))
    progress = extract._json(folder / 'progress.json')
    progress['n_truncated'] = 1
    extract._write(folder / 'progress.json', progress)
    with pytest.raises(AssertionError, match='zero truncated'):
        extract.gate_iv(folder, plain, captions, enc, record=work / 'iv_store.json')


def test_fake_encoder_distinguishes_empty_and_prefixed_replays(work):
    folder, plain, enc, captions = gate_inputs(work)
    prefixed = np.load(folder / 'txt.npy').astype(np.float32)
    unprefixed = np.load(plain / 'txt.npy').astype(np.float32)
    cosine = np.sum(prefixed * unprefixed, axis=1)
    assert (cosine < 0.9).all()
    assert extract.gate_iv(folder, plain, captions, enc, record=work / 'iv_store.json')['status'] == 'pass'


def test_gate_part_filename_and_independent_try_count(work, monkeypatch):
    folder, plain, enc, captions = gate_inputs(work)
    gates = work / 'records/gates'
    gates.mkdir(parents=True)
    shared, hash_part = gates / 'iv.json', gates / 'iv_hash.json'
    extract._write(shared, dict(tries=3))
    extract._write(hash_part, dict(tries=2))
    before = shared.read_bytes(), hash_part.read_bytes()
    monkeypatch.setattr(extract.paths, 'GATES', gates)
    for tries in range(1, 4):
        result = extract.gate_iv(folder, plain, captions, enc)
        assert result['gate'] == 'iv_store' and result['tries'] == tries
    assert extract._json(gates / 'iv_store.json') == result
    with pytest.raises(AssertionError, match='three tries'):
        extract.gate_iv(folder, plain, captions, enc)
    assert (shared.read_bytes(), hash_part.read_bytes()) == before


def test_gate_checks_stored_order(work):
    folder, plain, enc, captions = gate_inputs(work)
    reseal(folder, lambda index: index['texts'].reverse())
    with pytest.raises(AssertionError, match='extraction order'):
        extract.gate_iv(folder, plain, captions, enc, record=work / 'iv_store.json')


def test_help_documents_integration_worktree(capsys):
    with pytest.raises(SystemExit) as exit:
        extract.main(['--help'])
    assert exit.value.code == 0
    help_text = capsys.readouterr().out
    assert 'cd /project/CoSiR-r29\n' in help_text
    assert 'iv_store.json' in help_text
    assert '--expected-prefix' in help_text
