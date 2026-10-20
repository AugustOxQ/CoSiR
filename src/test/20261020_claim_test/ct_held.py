"""One name order and name-joined inputs for the claim test paintings."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
for folder in ('20261017_aspect_premise_checks', '20261018_percept_core'):
    sys.path.insert(0, str(HERE.parent / folder))

from apc_data import l2, paint_first, paint_majority
from pc_data import check_array, clip_features, patch_tokens
from src.data.artelingo_splits import encode_labels
from src.data.splits import grouped_split
from src.data.wikiart_genre import load_wikiart_genre
from src.utils.feature_manager import FeatureManager

ANNOTATIONS = Path('/data/PDD/artelingo/artelingo_train.json')
CLIP = Path('/data/SSD2/pre_extract/artelingo/features')
QWEN = Path('/data/SSD2/pre_extract/artelingo_full/qwen3vl_emb_2b')
HELD_LIST = HERE / 'outputs/held/held_paintings.json'


def _annotations():
    records = json.loads(ANNOTATIONS.read_text())
    assert len(records) == 308723
    return records


def _clip_index(records):
    manager = FeatureManager(str(CLIP))
    assert manager.metadata['backbone_model'] == 'openai/clip-vit-base-patch32'
    ids = np.asarray(manager.get_all_sample_ids())
    assert ids.dtype == np.int64 and ids.shape == (manager.total_samples,) == (len(records),)
    assert len(np.unique(ids)) == len(ids) and ((ids >= 0) & (ids < len(records))).all()
    names = np.asarray([records[int(i)]['painting'] for i in ids])
    assert len(np.unique(names)) == 61402
    return ids, names


def held_names():
    """Rebuild the seed-42 grouped held split from extraction-ID metadata."""
    _, names = _clip_index(_annotations())
    split = grouped_split(pd.factorize(names, sort=False)[0], seed=42)
    parts = [getattr(split, part) for part in ('train', 'val', 'held')]
    assert tuple(len(rows) for rows in parts) == (216107, 30872, 61744)
    assert tuple(len(np.unique(names[rows])) for rows in parts) == (42969, 6152, 12281)
    return sorted(set(names[split.held].tolist()))


def names_sha256(names):
    """Hash names joined by newlines, with no final newline (rule §3)."""
    return hashlib.sha256('\n'.join(names).encode('utf-8')).hexdigest()


def write_held_list(path=HELD_LIST):
    """Freeze the canonical list; an existing different list is never replaced."""
    path = Path(path)
    names = held_names()
    sha256 = names_sha256(names)
    if path.exists():
        assert read_held_list(path) == (names, sha256), 'held list differs'
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(dict(names=names, sha256=sha256), handle, indent=2)
        handle.write('\n')


def read_held_list(path=HELD_LIST):
    """Read the saved sorted list and verify its count, uniqueness and hash."""
    value = json.loads(Path(path).read_text())
    assert set(value) == {'names', 'sha256'}
    names, sha256 = value['names'], value['sha256']
    assert isinstance(names, list) and all(isinstance(name, str) for name in names)
    assert len(names) == len(set(names)) == 12281 and names == sorted(names)
    assert names_sha256(names) == sha256, 'held list hash mismatch'
    return names, sha256


def painting_labels(names):
    """Use full-annotation codes and each painting's captions in annotation order."""
    records = _annotations()
    paintings = np.asarray([record['painting'] for record in records])
    unique, pid = np.unique(paintings, return_inverse=True)
    assert len(unique) == 61402
    emotion, emotion_names = encode_labels([record['emotion'] for record in records])
    assert (emotion >= 0).all(), 'every caption has an emotion (rule §3)'
    style, style_names = encode_labels([record['art_style'] for record in records])
    assert len(emotion_names) == 9 and len(style_names) == 27
    first_style = paint_first(style, pid, len(unique))
    assert np.array_equal(style, first_style[pid]), 'style must be constant per painting'
    genre = load_wikiart_genre(paintings.astype(object))
    labels = dict(emotion=paint_majority(emotion, pid, len(unique), 9),
                  style=first_style, genre=paint_first(genre, pid, len(unique)))
    lookup = {name: row for row, name in enumerate(unique)}
    rows = np.asarray([lookup[name] for name in names], dtype=np.int64)
    return {aspect: values[rows] for aspect, values in labels.items()}


def qwen_default_images(names):
    """Join the unique-image store's names, then use the training path's L2."""
    images = json.loads((QWEN / 'index.json').read_text())['images']
    stored_names = [record['painting'] for record in images]
    assert len(stored_names) == len(set(stored_names)) == 61402
    lookup = {name: row for row, name in enumerate(stored_names)}
    rows = np.asarray([lookup[name] for name in names], dtype=np.int64)
    raw = np.load(QWEN / 'img.npy', mmap_mode='r')
    assert raw.shape == (61402, 2048) and raw.dtype == np.float16
    x = np.asarray(raw[rows], dtype=np.float32)
    assert (np.linalg.norm(x, axis=1) > 0).all()
    return check_array(l2(x), (len(names), 2048))


def clip_global_images(names):
    """Join painting -> extraction sample ID -> CLIP row, normalized as Stage 2."""
    ids, paintings = _clip_index(_annotations())
    unique, first = np.unique(paintings, return_index=True)
    lookup = {name: ids[row] for name, row in zip(unique, first)}
    sample_ids = np.asarray([lookup[name] for name in names], dtype=np.int64)
    x = clip_features(sample_ids, 'img_features')
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    assert (norms > 0).all()
    return check_array(x / norms, (len(names), 512))


def clip_patches(names):
    """Reuse the core's sorted-name join and bounded-memory patch gather."""
    rows = np.arange(len(names), dtype=np.int64)
    s = dict(first=rows, all_painting_ids=np.asarray(names))
    return patch_tokens(s, rows)
