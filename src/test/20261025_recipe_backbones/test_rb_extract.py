"""Ticket 08 wiring, no GPU and no model: the writers' metadata passes the real readers in rb_encoders."""
import sys
import types

import numpy as np
import pytest
from PIL import Image

import rb_paths
import rb_encoders
import rb_extract_a28 as a28
from scripts import extract_features as ef

NAMES = ['a_one', 'b_two', 'c_three', 'd_four']


class FakeQwen:
    instruction = None

    def encode_images(self, images, batch_size=32):
        assert self.instruction is not None
        rng = np.random.default_rng(len(images) + len(self.instruction))
        x = rng.normal(size=(len(images), 2048)).astype(np.float32)
        return x / np.linalg.norm(x, axis=1, keepdims=True)


def fake_spec(names):
    images = [{'painting': n, 'image': f'images/{n}.jpg'} for n in names]
    return ef.Spec(images, lambda i: Image.new('RGB', (4, 4)), [], [])


def test_instructed_store_passes_reader(tmp_path, monkeypatch):
    root = tmp_path / 'artelingo28'
    monkeypatch.setattr(rb_encoders, 'A28_ROOT', root)
    monkeypatch.setattr(rb_encoders, 'A28_COUNT', len(NAMES))
    monkeypatch.setattr(rb_encoders, 'a28_names', lambda: list(NAMES))
    store_root = root / 'qwen3vl_emb_2b_aspect_prompts'
    identity = dict(model_id='m', model_revision=a28.ct2_data.QWEN_REVISION)
    spec, rows = fake_spec(NAMES), np.arange(len(NAMES))
    for key in a28.KEYS:
        a28.write_store(FakeQwen(), identity, key, NAMES, spec, rows, store_root, smoke=False)
        x = rb_encoders.a28_instructed(key)
        assert x.shape == (len(NAMES), 2048) and x.dtype == np.float32
    with pytest.raises(AssertionError, match='existing store'):
        a28.write_store(FakeQwen(), identity, 'genre_word', NAMES, spec, rows, store_root, smoke=False)
    # a smoke store is refused by the reader
    a28.write_store(FakeQwen(), identity, 'emotion_focus', NAMES, spec, rows, tmp_path / 'smoke', smoke=True)
    monkeypatch.setattr(rb_encoders, 'A28_ROOT', tmp_path / 'smoke_root')
    (tmp_path / 'smoke_root').mkdir()
    (tmp_path / 'smoke_root' / 'qwen3vl_emb_2b_aspect_prompts').symlink_to(tmp_path / 'smoke')
    with pytest.raises(AssertionError, match='smoke'):
        rb_encoders.a28_instructed('emotion_focus')


class FakeDino:
    name, dim = 'dinov2_large', 1024

    def encode_images(self, images, batch_size=128):
        x = np.random.default_rng(0).normal(size=(len(list(images)), 1024)).astype(np.float32)
        return x / np.linalg.norm(x, axis=1, keepdims=True)


def run_main(monkeypatch, tmp_path, argv, dataset='artelingo28'):
    monkeypatch.setitem(ef.ADAPTERS, dataset, lambda **kw: ef.Spec(
        [{'painting': n, 'image': f'images/{n}.jpg'} for n in NAMES], lambda i: Image.new('RGB', (4, 4)),
        ['t'] * 8, [{'row_id': j, 'painting': NAMES[j // 2], 'img': j // 2} for j in range(8)]))
    mod = types.ModuleType('src.data.feature_extract')
    mod.load_encoder = lambda name, device: FakeDino()
    monkeypatch.setitem(sys.modules, 'src.data.feature_extract', mod)
    ef.main(['--dataset', dataset, '--out-root', str(tmp_path)] + argv)


def test_allow_list():
    assert 'clip_b32' in ef.ALLOWED['artelingo_full']
    assert 'dinov2_large' in ef.ALLOWED['artelingo_full'] and 'dinov2_large' in ef.ALLOWED['artelingo28']
    assert all('dinov2_large' not in v for k, v in ef.ALLOWED.items() if k not in ef.IMAGE_ONLY_DATASETS)


def test_dino_image_only_store_passes_reader(tmp_path, monkeypatch):
    run_main(monkeypatch, tmp_path, ['--backbone', 'dinov2_large', '--image-only'])
    path = tmp_path / 'artelingo28' / 'dinov2_large'
    assert not (path / 'txt.npy').exists()
    x = rb_encoders.read_store(path, 1024, NAMES[::-1], len(NAMES))
    assert x.shape == (4, 1024)
    assert ef.verify('artelingo28', 'dinov2_large', root=tmp_path)['ok']
    with pytest.raises(SystemExit):     # sealed: never overwritten
        run_main(monkeypatch, tmp_path, ['--backbone', 'dinov2_large', '--image-only'])


def test_dino_needs_image_only_and_limit_goes_to_smoke(tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        run_main(monkeypatch, tmp_path, ['--backbone', 'dinov2_large'])
    run_main(monkeypatch, tmp_path, ['--backbone', 'dinov2_large', '--image-only', '--limit', '2'])
    assert (tmp_path / '_smoke' / 'artelingo28' / 'dinov2_large' / 'img.npy').exists()
    assert not (tmp_path / 'artelingo28').exists()
