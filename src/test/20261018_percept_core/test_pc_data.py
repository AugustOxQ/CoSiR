"""CPU checks of frozen IDs against independent feature-store reads."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import pc_data

CLIP = Path('/data/SSD2/pre_extract/artelingo/features')
QWEN = Path('/data/SSD2/pre_extract/artelingo_full/qwen3vl_emb_2b_aspect_prompts')
ANNOTATIONS = Path('/data/PDD/artelingo/artelingo_train.json')
PATCHES = Path('/data/SSD2/pre_extract/artelingo_percept_patch_features/train_patch_features.pt')


@pytest.mark.parametrize('fault', [None, 'order', 'id_count', 'count', 'fingerprint',
                                  'names', 'feature_count', 'dimension', 'dtype', 'nonfinite'])
def test_roberta_store_identity_and_array_boundaries(tmp_path, monkeypatch, fault):
    s = dict(sample_ids=np.array([12, 5, 38], dtype=np.int64),
             painting_ids=np.array(['b', 'a', 'b']), pid=np.array([1, 0, 1]),
             first=np.array([1, 0]), all_painting_ids=np.array(['a', 'b']),
             subset_pid=np.array([0]), subset_image_indices=np.array([7]))
    ids, names = s['sample_ids'].copy(), s['painting_ids'].copy()
    expected = np.arange(3 * 768, dtype=np.float32).reshape(3, 768)
    x = expected.copy()
    done = dict(count=3, sample_sha256=pc_data.fingerprint(s))
    if fault == 'order':
        ids, names, x = ids[::-1], names[::-1], x[::-1]
    elif fault == 'id_count':
        ids, names = ids[:-1], names[:-1]
    elif fault == 'count':
        done['count'] = 2
    elif fault == 'fingerprint':
        done['sample_sha256'] = 'different sample'
    elif fault == 'names':
        names[0] = 'c'
    elif fault == 'feature_count':
        x = x[:-1]
    elif fault == 'dimension':
        x = x[:, :-1]
    elif fault == 'dtype':
        x = x.astype(np.float16)
    elif fault == 'nonfinite':
        x[1, 2] = np.nan
    np.save(tmp_path / 'txt.npy', x)
    np.savez(tmp_path / 'ids.npz', sample_ids=ids, painting_ids=names)
    (tmp_path / 'done.json').write_text(json.dumps(done))
    monkeypatch.setattr(pc_data, 'ROBERTA', tmp_path)
    if fault is None:
        np.testing.assert_array_equal(pc_data.caption_features(s, 'roberta'), expected)
    else:
        with pytest.raises(AssertionError):
            pc_data.caption_features(s, 'roberta')


def test_caption_texts_match_annotation_sample_ids():
    s = pc_data.load_frozen()
    texts = pc_data.caption_texts(s)
    assert isinstance(texts, list) and len(texts) == 99999
    records = json.loads(ANNOTATIONS.read_text())
    for pos in np.random.default_rng(12).choice(len(texts), 20, replace=False):
        assert texts[pos] == records[int(s['sample_ids'][pos])]['caption']


def test_patch_tokens_match_independently_found_painting_names():
    s = pc_data.load_frozen()
    records = json.loads(ANNOTATIONS.read_text())
    names = sorted({record['painting'] for record in records})
    chosen = np.random.default_rng(11).choice(len(s['first']), 5, replace=False)
    # Include a duplicate and retain request order across the gather chunk boundary.
    pids = np.concatenate([np.tile(chosen, 60), chosen[:1]])
    x = pc_data.patch_tokens(s, pids)
    assert x.shape == (len(pids), 50, 512) and x.dtype == np.float32
    raw = torch.load(PATCHES, map_location='cpu', mmap=True, weights_only=True)
    for pos, pid in enumerate(pids):
        row = names.index(s['all_painting_ids'][pid])
        np.testing.assert_array_equal(x[pos], raw[row].numpy())


@pytest.mark.parametrize('key', ['default', 'emotion_focus'])
def test_qwen_captions_match_extraction_ids(key):
    s = pc_data.load_frozen()
    chosen = np.random.default_rng(9).choice(len(s['sample_ids']), 20, replace=False)
    x = pc_data.caption_features(s, 'qwen_' + key)
    assert x.shape == (99999, 2048) and x.dtype == np.float32
    with np.load(QWEN / key / 'ids.npz') as ids:
        raw = np.load(QWEN / key / 'txt.npy', mmap_mode='r')
        for pos in chosen:
            row, = np.flatnonzero(ids['sample_ids'] == s['sample_ids'][pos])
            np.testing.assert_array_equal(x[pos], raw[row].astype(np.float32))


def test_qwen_images_match_painting_names():
    s = pc_data.load_frozen()
    store = Path('/data/SSD2/pre_extract/artelingo_full/qwen3vl_emb_2b')
    names = [record['painting'] for record in json.loads((store / 'index.json').read_text())['images']]
    raw = np.load(store / 'img.npy', mmap_mode='r')
    x = pc_data.image_features(s, 'qwen_default')
    assert x.shape == (19889, 2048) and x.dtype == np.float32
    for pos in np.random.default_rng(10).choice(len(s['first']), 20, replace=False):
        expected = raw[names.index(s['all_painting_ids'][pos])].astype(np.float32)
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(x[pos], expected, rtol=1e-6, atol=1e-7)


def test_clip_captions_and_first_caption_images_match_shard_ids():
    s = pc_data.load_frozen()
    chosen = np.random.default_rng(7).choice(len(s['sample_ids']), 20, replace=False)
    paintings = np.random.default_rng(8).choice(len(s['first']), 20, replace=False)
    # Storage positions are not the identity: require an extraction-ID join.
    s['rows'] = np.zeros_like(s['rows'])
    text = pc_data.caption_features(s, 'clip_text')
    image = pc_data.image_features(s, 'clip_image')
    assert text.shape == (99999, 512) and text.dtype == np.float32
    assert image.shape == (19889, 512) and image.dtype == np.float32
    for result, positions, ids, filename in (
            (text, chosen, s['sample_ids'][chosen], 'txt_features.npy'),
            (image, paintings, s['sample_ids'][s['first'][paintings]], 'img_features.npy')):
        found = 0
        for folder in sorted((CLIP / 'shards').iterdir()):
            stored = np.load(folder / 'sample_ids.npy', mmap_mode='r')
            raw = np.load(folder / filename, mmap_mode='r')
            for pos, sample_id in zip(positions, ids):
                rows = np.flatnonzero(stored == sample_id)
                if len(rows):
                    assert len(rows) == 1
                    np.testing.assert_array_equal(result[pos], raw[rows[0]])
                    found += 1
        assert found == len(ids)


def test_partitions_are_disjoint_and_follow_paintings():
    s = pc_data.load_frozen()
    part = pc_data.partitions(s)
    assert part.shape == (19889,)
    assert {key: int((part == key).sum()) for key in ('fit', 'dev', 'eval')} == {
        'fit': 13400, 'dev': 1489, 'eval': 5000}
    _, fit, dev = pc_data.split(s)
    assert (part[fit] == 'fit').all()
    assert (part[dev] == 'dev').all()
    assert (part[s['subset_pid']] == 'eval').all()
    assert (part[s['pid']][np.isin(s['pid'], s['subset_pid'])] == 'eval').all()
