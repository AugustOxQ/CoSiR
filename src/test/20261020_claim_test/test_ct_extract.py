"""CPU smoke coverage for name joins, train parity gates and store sealing."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

import ct_extract as ct
from src.data import feature_extract


class Image:
    def __init__(self, row):
        self.row = row
        self.closed = False

    def close(self):
        self.closed = True


class Encoder:
    def __init__(self, device='cpu', instruction=ct.PROMPTS['emotion_focus']):
        assert device == 'cpu'
        self.instruction = instruction
        self.m = SimpleNamespace(config=SimpleNamespace(_commit_hash='a' * 40))
        self.calls = []

    def encode_images(self, images, batch_size):
        assert batch_size == 32  # The unchanged train-image encoder call.
        self.calls.append(list(images))
        x = np.zeros((len(images), 2048), dtype=np.float32)
        x[np.arange(len(images)), [im.row for im in images]] = 1
        return x


@pytest.fixture
def inputs():
    names = ['a', 'b', 'c']
    loaded = []

    def load(row):
        im = Image(row)
        loaded.append(im)
        return im

    spec = SimpleNamespace(images=[dict(painting=n) for n in ['c', 'a', 'b']], load_image=load)
    return names, spec, np.asarray([1, 2, 0]), loaded


def test_image_spec_joins_names_and_disables_full_count_gate(monkeypatch):
    from scripts import extract_features
    monkeypatch.setattr(ct.ct_held, '_annotations', lambda: [
        dict(painting='b'), dict(painting='a'), dict(painting='b')])

    def adapter(*, sample_ids, expected):
        assert sample_ids.dtype == np.int64
        assert sample_ids.tolist() == [1, 0]
        assert expected is None
        return SimpleNamespace(images=[dict(painting='b'), dict(painting='a')])

    monkeypatch.setattr(extract_features, 'adapter_artelingo_full', adapter)
    _, rows = ct.image_spec(['a', 'b'])
    assert rows.tolist() == [1, 0]


def test_pending_store_smoke_and_immutable_resume(tmp_path, inputs, monkeypatch):
    names, spec, rows, loaded = inputs
    enc = Encoder()
    sha = ct.ct_held.names_sha256(names)
    original_encode = enc.encode_images

    def encode(images, batch_size):
        assert (tmp_path / 'emotion_focus_held.pending').is_dir()
        assert not (tmp_path / 'emotion_focus_held').exists()
        return original_encode(images, batch_size)

    enc.encode_images = encode
    path = ct.write_store(enc, 'emotion_focus', names, sha, spec, rows, batch=2, root=tmp_path)
    assert [len(call) for call in enc.calls] == [2, 1]
    assert all(im.closed for im in loaded)
    assert not path.with_name(path.name + '.pending').exists()
    x = np.load(path / 'img.npy')
    assert x.dtype == np.float16 and x.argmax(axis=1).tolist() == [1, 2, 0]
    meta = ct.verify_store(path, 'emotion_focus', names, sha)
    assert meta['model_id'] == feature_extract.QWEN_REPO
    assert meta['model_revision'] == 'a' * 40
    before = {p.name: p.read_bytes() for p in path.iterdir()}
    with pytest.raises(AssertionError, match='existing store'):
        ct.write_store(enc, 'emotion_focus', names, sha, spec, rows, root=tmp_path)
    monkeypatch.setattr(ct, 'OUTPUT', tmp_path)
    monkeypatch.setattr(ct.ct_held, 'read_held_list', lambda path: (names, sha))
    monkeypatch.setattr(feature_extract, 'Qwen3VLEmb', lambda **kw: pytest.fail('resume built model'))
    assert ct.main(['--keys', 'emotion_focus']) == [path]
    assert before == {p.name: p.read_bytes() for p in path.iterdir()}


def test_pending_refused_without_overwrite(tmp_path, inputs):
    names, spec, rows, _ = inputs
    pending = tmp_path / 'emotion_focus_held.pending'
    pending.mkdir()
    (pending / 'partial').write_text('keep')
    with pytest.raises(FileExistsError):
        ct.write_store(Encoder(), 'emotion_focus', names, ct.ct_held.names_sha256(names),
                       spec, rows, root=tmp_path)
    assert (pending / 'partial').read_text() == 'keep'


def test_failed_encoding_leaves_only_pending(tmp_path, inputs):
    names, spec, rows, loaded = inputs
    enc = Encoder()

    def fail(images, batch_size):
        raise RuntimeError('interrupted')

    enc.encode_images = fail
    with pytest.raises(RuntimeError, match='interrupted'):
        ct.write_store(enc, 'emotion_focus', names, ct.ct_held.names_sha256(names),
                       spec, rows, root=tmp_path)
    assert (tmp_path / 'emotion_focus_held.pending').is_dir()
    assert not (tmp_path / 'emotion_focus_held').exists()
    assert all(im.closed for im in loaded)


@pytest.mark.parametrize('damage', ['names', 'hash', 'instruction', 'values'])
def test_verify_rejects_mismatched_or_corrupt_store(tmp_path, inputs, damage):
    names, spec, rows, _ = inputs
    sha = ct.ct_held.names_sha256(names)
    path = ct.write_store(Encoder(), 'emotion_focus', names, sha, spec, rows, root=tmp_path)
    if damage == 'names':
        np.savez(path / 'ids.npz', painting_names=np.asarray(names[::-1]))
    elif damage == 'values':
        x = np.load(path / 'img.npy')
        np.save(path / 'img.npy', x[::-1])
    else:
        meta = json.loads((path / 'meta.json').read_text())
        meta['held_sha256' if damage == 'hash' else 'instruction'] = 'wrong'
        (path / 'meta.json').write_text(json.dumps(meta))
    with pytest.raises(AssertionError):
        ct.verify_store(path, 'emotion_focus', names, sha)


def test_check_requires_each_row_not_mean(monkeypatch, tmp_path, inputs, capsys):
    names, spec, rows, _ = inputs
    sample = tmp_path / 'sample.npz'
    np.savez(sample, unused=np.zeros(1))
    monkeypatch.setattr(ct, 'SAMPLE', sample)
    reference = np.zeros((3, 2048), np.float16)
    reference[np.arange(3), rows] = 1
    monkeypatch.setattr(ct, 'validate_train_images', lambda s, key: (
        reference, dict(painting_names=np.asarray(names))))
    monkeypatch.setattr(ct, 'image_spec', lambda requested: (spec, rows))
    assert ct.check_train_images(Encoder(), 'emotion_focus', 3, batch=2)['min_cosine'] == 1
    # Mean remains above .999, but the worst row fails the ticket's gate.
    reference[0, rows[0]] = .998
    reference[0, 9] = np.sqrt(1 - .998 ** 2)
    with pytest.raises(AssertionError, match='per-row'):
        ct.check_train_images(Encoder(), 'emotion_focus', 3)
    assert 'min_cosine=' in capsys.readouterr().out


def test_cpu_check_cli_all_instructions_no_held_reads(monkeypatch):
    calls = []
    monkeypatch.setattr(ct.ct_held, 'read_held_list', lambda path: pytest.fail('check read held'))
    monkeypatch.setattr(feature_extract, 'Qwen3VLEmb', Encoder)
    monkeypatch.setattr(ct, 'check_train_images', lambda enc, key, n, batch: calls.append(
        (key, enc.instruction, n, batch)))
    ct.main(['--check', '2', '--device', 'cpu', '--batch', '1'])
    assert calls == [(key, ct.PROMPTS[key], 2, 1) for key in ct.KEYS]


def test_cli_cuda_default_and_cpu_full_refusal(monkeypatch, inputs, tmp_path):
    names, spec, rows, _ = inputs
    monkeypatch.setattr(ct, 'OUTPUT', tmp_path)
    monkeypatch.setattr(ct.ct_held, 'read_held_list', lambda path: (names, ct.ct_held.names_sha256(names)))
    monkeypatch.setattr(ct, 'image_spec', lambda names: (spec, rows))
    calls = []

    def encoder(device, instruction):
        assert device == 'cuda'
        return Encoder(instruction=instruction)

    monkeypatch.setattr(feature_extract, 'Qwen3VLEmb', encoder)
    monkeypatch.setattr(ct, 'write_store', lambda enc, key, *args: calls.append((key, enc.instruction)))
    ct.main([])
    assert calls == [(key, ct.PROMPTS[key]) for key in ct.KEYS]
    with pytest.raises(AssertionError, match='requires GPU'):
        ct.main(['--device', 'cpu'])


def test_real_development_adapter_rgb_smoke(monkeypatch):
    """Use real development IDs and RGB files; substitute only the neural encoder."""
    names = np.load(ct.OUTPUT / 'emotion_focus_trainimg/ids.npz', allow_pickle=False)['painting_names'][:2].tolist()
    spec, rows = ct.image_spec(names)
    assert [spec.images[int(row)]['painting'] for row in rows] == names
    with spec.load_image(int(rows[0])) as image:
        assert image.mode == 'RGB' and min(image.size) > 0
    ref = np.load(ct.OUTPUT / 'emotion_focus_trainimg/img.npy', mmap_mode='r')[:2]

    class Replay(Encoder):
        def encode_images(self, images, batch_size):
            assert batch_size == 32
            assert all(im.mode == 'RGB' for im in images)
            return ref.astype(np.float32)

    monkeypatch.setattr(feature_extract, 'Qwen3VLEmb', Replay)
    assert ct.main(['--check', '2', '--keys', 'emotion_focus', '--device', 'cpu']) == []
