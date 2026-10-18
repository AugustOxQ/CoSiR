"""Frozen-caption and painting joins for the PercepT core feature readers."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '20261017_image_transfer_pilot'))
from itp_data import default_images, fingerprint, load_frozen, prompted, split
from src.utils.feature_manager import FeatureManager

CLIP = Path('/data/SSD2/pre_extract/artelingo/features')
ANNOTATIONS = Path('/data/PDD/artelingo/artelingo_train.json')
PATCHES = Path('/data/SSD2/pre_extract/artelingo_percept_patch_features/train_patch_features.pt')
ROBERTA = Path('/data/SSD2/pre_extract/artelingo_full/roberta_go_emotions')


def check_array(x, shape):
    assert x.shape == shape and x.dtype == np.float32
    for start in range(0, len(x), 256):
        assert np.isfinite(x[start:start + 256]).all(), 'nonfinite features'
    return x


def clip_features(sample_ids, feature):
    """Join actual extraction IDs, keeping only one FeatureManager shard in RAM."""
    manager = FeatureManager(str(CLIP))
    assert manager.metadata['backbone_model'] == 'openai/clip-vit-base-patch32'
    stored = np.asarray(manager.get_all_sample_ids(), dtype=np.int64)
    assert stored.shape == (manager.total_samples,)
    assert len(np.unique(stored)) == len(stored)
    lookup = {int(sample_id): row for row, sample_id in enumerate(stored)}
    rows = np.asarray([lookup[int(sample_id)] for sample_id in sample_ids], dtype=np.int64)
    x = np.empty((len(rows), 512), dtype=np.float32)
    offset = 0
    for shard_id in range(manager.num_shards):
        shard = manager.load_shard_to_ram(shard_id, [feature])
        ids = shard['sample_ids'].numpy()
        raw = shard[feature].numpy()
        assert np.array_equal(ids, stored[offset:offset + len(ids)])
        assert raw.shape == (len(ids), 512) and raw.dtype == np.float32
        take = (rows >= offset) & (rows < offset + len(ids))
        x[take] = raw[rows[take] - offset]
        offset += len(ids)
        del shard, raw, ids
    assert offset == len(stored)
    return check_array(x, (len(sample_ids), 512))


def caption_features(s, kind):
    assert kind in ('clip_text', 'roberta', 'qwen_default', 'qwen_emotion_focus'), kind
    if kind == 'clip_text':
        return clip_features(s['sample_ids'], 'txt_features')
    if kind == 'roberta':
        done = json.loads((ROBERTA / 'done.json').read_text())
        assert done['count'] == len(s['sample_ids']), 'RoBERTa count mismatch'
        assert done['sample_sha256'] == fingerprint(s), 'RoBERTa sample mismatch'
        with np.load(ROBERTA / 'ids.npz', allow_pickle=False) as ids:
            assert ids['sample_ids'].dtype == np.int64
            assert np.array_equal(ids['sample_ids'], s['sample_ids']), 'RoBERTa ID order mismatch'
            assert np.array_equal(ids['painting_ids'], s['painting_ids']), 'RoBERTa painting mismatch'
        return check_array(np.array(np.load(ROBERTA / 'txt.npy', mmap_mode='r')),
                           (len(s['sample_ids']), 768))
    # prompted asserts exact extraction-ID and painting-name order before reading.
    raw = prompted(s, kind.removeprefix('qwen_'))[1]
    return check_array(np.asarray(raw, dtype=np.float32), (len(s['sample_ids']), 2048))


def image_features(s, kind):
    assert kind in ('clip_image', 'qwen_default'), kind
    if kind == 'qwen_default':
        return check_array(default_images(s)[0], (len(s['first']), 2048))
    first = s['first']
    assert np.array_equal(s['painting_ids'][first], s['all_painting_ids'])
    return clip_features(s['sample_ids'][first], 'img_features')


def caption_texts(s):
    with ANNOTATIONS.open() as f:
        records = json.load(f)
    ids = s['sample_ids']
    assert ids.dtype == np.int64 and ids.ndim == 1
    assert len(np.unique(ids)) == len(ids) and ((ids >= 0) & (ids < len(records))).all()
    selected = [records[int(sample_id)] for sample_id in ids]
    assert np.array_equal([record['painting'] for record in selected], s['painting_ids'])
    texts = [record['caption'] for record in selected]
    assert all(isinstance(text, str) and text.strip() for text in texts)
    return texts


def patch_tokens(s, pids):
    """Name-joined patch rows; remap every 256 paintings to bound resident cache RAM."""
    pids = np.asarray(pids)
    assert pids.ndim == 1 and np.issubdtype(pids.dtype, np.integer)
    assert ((pids >= 0) & (pids < len(s['first']))).all()
    with ANNOTATIONS.open() as f:
        names = sorted({record['painting'] for record in json.load(f)})
    assert len(names) == 61402
    lookup = {name: row for row, name in enumerate(names)}
    rows = np.asarray([lookup[str(name)] for name in s['all_painting_ids'][pids]], dtype=np.int64)
    x = np.empty((len(pids), 50, 512), dtype=np.float32)
    # A single long-lived mapping would retain touched cache pages alongside the
    # 2.04 GB output. Releasing each mapping keeps peak RAM near the output size.
    for start in range(0, len(rows), 256):
        cache = torch.load(PATCHES, map_location='cpu', mmap=True, weights_only=True)
        assert cache.shape == (61402, 50, 512) and cache.dtype == torch.float32
        torch.index_select(cache, 0, torch.from_numpy(rows[start:start + 256]),
                           out=torch.from_numpy(x[start:start + 256]))
        del cache
    return check_array(x, (len(pids), 50, 512))


def partitions(s):
    """Painting partition labels; captions inherit labels through s['pid']."""
    _, fit, dev = split(s)
    part = np.full(len(s['first']), '', dtype='<U4')
    for key, ids, count in (('fit', fit, 13400), ('dev', dev, 1489),
                            ('eval', s['subset_pid'], 5000)):
        assert len(ids) == len(np.unique(ids)) == count
        assert (part[ids] == '').all(), 'overlapping partitions'
        part[ids] = key
    assert (part != '').all()
    return part
