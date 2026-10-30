"""Affection labels and keyed frozen inputs (r12, spec 2026-10-30).

Keys use rs_vh.names_sha (UTF-8, newline joined, no trailing newline); array
hashes use rs_vh.array_sha (contiguous array bytes, not the .npy/.npz file).
Only annotation aggregates are cached; feature arrays are read per call.
"""
import csv
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

import ac_paths
import rs_vh
import rt_crl
import rt_data

ARMS = ('cstar', 'plain', 'teacher_artwork', 'teacher_photo', 'clip', 'percept',
        'crl_clip_painting', 'crl_clip_photo', 'crl_qwen_painting', 'crl_qwen_photo')
SEEDED = {'cstar', 'percept'}


@lru_cache(maxsize=4)
def _annotations(path, mtime_ns, size):
    del mtime_ns, size  # Cache identity includes file changes and synthetic fixture paths.
    emotions = ac_paths.EMOTIONS
    assert tuple(emotions) == rs_vh.EMOTIONS == tuple(sorted(emotions)), 'r10 emotion codes'
    codes = {emotion: i for i, emotion in enumerate(emotions)}
    membership, counts = {}, {}
    with Path(path).open(newline='') as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        assert fields and len(fields) == len(set(fields)), 'annotation columns'
        assert {'dataset', 'image_name', 'split', 'emotion'} <= set(fields), 'annotation columns'
        for row in reader:
            assert None not in row and None not in row.values(), 'malformed annotation row'
            split = row['split']
            assert split in ('train', 'val', 'test', 'ignored'), 'unknown annotation split'
            # Spec §3.4 L130–132: ignored rows enter neither votes nor distributions.
            if split == 'ignored':
                continue
            source, image = row['dataset'], row['image_name']
            assert source and image and '/' not in source, 'annotation key fields'
            key = f'{source}/{image}'
            # Spec §3.3 L121–124: one non-ignored split per image key.
            assert membership.setdefault(key, split) == split, f'multiple splits: {key}'
            assert row['emotion'] in codes, 'unknown annotation emotion'
            if split in ac_paths.SPLITS:
                if key not in counts:
                    counts[key] = np.zeros(9, dtype=np.int64)
                counts[key][codes[row['emotion']]] += 1
    return membership, counts


def _split(split):
    assert split in ac_paths.SPLITS, split
    path = Path(ac_paths.ANNOTATIONS)
    stat = path.stat()
    membership, counts = _annotations(path, stat.st_mtime_ns, stat.st_size)
    # Spec §3.3 L121–124: Python-sorted keys fix every evaluation row.
    ns = sorted(key for key, part in membership.items() if part == split)
    assert len(ns) == len(set(ns)) == ac_paths.N[split], 'split key count'
    votes = np.asarray([counts[key] for key in ns], dtype=np.int64)
    assert votes.shape == (len(ns), 9) and (votes.sum(axis=1) > 0).all(), 'annotation votes'
    return ns, votes


def keys(split):
    """Unique `<dataset>/<image_name>` keys, in Python sorted order."""
    return _split(split)[0]


def labels(split):
    """Plurality labels, int64 [n]; alphabetical codes break ties."""
    _, votes = _split(split)
    # Spec §3.4 L130–132: argmax chooses the first (lowest alphabetical) code.
    result = votes.argmax(axis=1).astype(np.int64)
    assert result.shape == (ac_paths.N[split],) and result.dtype == np.int64
    return result


def ties(split):
    """True where multiple emotions share the largest non-ignored vote count."""
    _, votes = _split(split)
    # Spec §3.4 L131–134: ties concern the plurality, not any repeated vote count.
    result = (votes == votes.max(axis=1, keepdims=True)).sum(axis=1) > 1
    assert result.shape == (ac_paths.N[split],) and result.dtype == np.bool_
    return result


def emotion_dist(split):
    """Non-ignored emotion shares, float64 [n, 9], alphabetical columns."""
    _, votes = _split(split)
    # Spec §3.4 L132: graded relevance uses exactly the label's annotation rows.
    result = votes / votes.sum(axis=1, keepdims=True)
    assert result.shape == (ac_paths.N[split], 9) and result.dtype == np.float64
    assert np.allclose(result.sum(axis=1), 1, rtol=0, atol=1e-15)
    return result


def sources(split):
    """Annotation dataset names, in keys(split) order."""
    return [key.split('/', 1)[0] for key in keys(split)]


def label_counts(split):
    """JSON-ready class counts, plurality-tie count and image count (§3.4 gate)."""
    y = labels(split)
    return {'classes': np.bincount(y, minlength=9).tolist(),
            'ties': int(ties(split).sum()), 'n': len(y)}


def _json(path, required=()):
    result = json.loads(Path(path).read_text())
    assert isinstance(result, dict), f'expected JSON object: {path}'
    assert set(required) <= set(result), f'missing metadata fields: {path}'
    return result


def _key_list(stored):
    assert isinstance(stored, list) and all(isinstance(key, str) and key for key in stored), 'key list'
    assert len(stored) == len(set(stored)), 'duplicate stored keys'


def _gather(stored, requested):
    _key_list(stored)
    _key_list(requested)
    lookup = {key: i for i, key in enumerate(stored)}
    missing = [key for key in requested if key not in lookup]
    assert not missing, f'missing keys: {missing[:3]}'
    rows = np.asarray([lookup[key] for key in requested], dtype=np.int64)
    # Spec §3.3 L124 (C6): confirm identities after joining, not just row counts.
    assert [stored[i] for i in rows] == requested, 'joined key order'
    return rows


def _array(x, count, width, dtype, *, unit=False):
    assert x.shape == (count, width) and x.dtype == dtype, (x.shape, x.dtype, count, width, dtype)
    assert np.isfinite(x).all(), 'nonfinite features'
    if unit:
        assert np.all(np.abs(np.linalg.norm(x.astype(np.float32), axis=1) - 1) < 1e-3), 'nonunit features'
    return x


def _unit(x):
    x = x.astype(np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    assert np.isfinite(norms).all() and (norms > 0).all(), 'zero/nonfinite feature norm'
    return _array(x / norms, len(x), x.shape[1], np.float32, unit=True)


def _plain(requested):
    path = Path(ac_paths.STORE)
    side = _json(path / 'index.json', ('dataset', 'backbone', 'smoke', 'dtype', 'dim', 'n_images', 'images'))
    assert side['dataset'] == 'affection_local' and side['backbone'] == 'qwen3vl_emb_2b', 'store identity'
    assert side['smoke'] is False and side['dtype'] == 'float16' and side['dim'] == 2048, 'store metadata'
    stored = []
    assert isinstance(side['images'], list), 'store image list'
    for row in side['images']:
        assert isinstance(row, dict) and {'source', 'image_name'} <= set(row), 'store key fields'
        assert isinstance(row['source'], str) and row['source'], 'store source'
        assert isinstance(row['image_name'], str) and row['image_name'], 'store image name'
        stored.append(f"{row['source']}/{row['image_name']}")
    assert len(stored) == side['n_images'], 'store row count'
    rows = _gather(stored, requested)
    x = np.load(path / 'img.npy', mmap_mode='r', allow_pickle=False)
    assert x.shape == (len(stored), 2048) and x.dtype == np.float16, 'store array shape/dtype'
    # Spec §3.3 L117: gather only the requested val/test rows; never read train features.
    return _array(x[rows], len(requested), 2048, np.float16, unit=True)


def _view(arm, requested):
    is_clip = arm == 'clip'
    path = Path(ac_paths.CLIP_VIEW) if is_clip else Path(ac_paths.VIEWS) / ac_paths.WORDINGS[arm[8:]]
    side = _json(path / 'meta.json', ('instruction', 'model_id', 'model_revision', 'dtype', 'shape',
                                     'keys_sha256', 'img_sha256', 'gate', 'smoke'))
    stored = json.loads((path / 'keys.json').read_text())
    _key_list(stored)
    # Spec §3.3 L122–124; index: views carry validation keys followed by test keys.
    assert stored == keys('val') + keys('test'), 'view key order'
    assert side['keys_sha256'] == rs_vh.names_sha(stored), 'view keys hash'
    gate = side['gate']
    assert side['smoke'] is False and isinstance(gate, dict) and gate.get('ok') is True, 'view extraction gate'
    width = 768 if is_clip else 2048
    assert side['dtype'] == 'float16' and side['shape'] == [len(stored), width], 'view shape/dtype'
    model = rt_crl.CLIP_ID if is_clip else 'Qwen/Qwen3-VL-Embedding-2B'
    revision = rt_crl.CLIP_REVISION if is_clip else rt_data.ct2_data.QWEN_REVISION
    assert side['model_id'] == model and side['model_revision'] == revision, 'view encoder'
    instruction = None if is_clip else (ac_paths.PHOTO_INSTRUCTION if arm == 'teacher_photo' else
                                        rt_data.ct2_data.itp_data.PROMPTS['emotion_focus'])
    # Spec §3.2 L102–103: wordings define different arms, with exact instructions.
    assert side['instruction'] == instruction, 'view instruction'
    x = np.load(path / 'img.npy', mmap_mode='r', allow_pickle=False)
    _array(x, len(stored), width, np.float16, unit=True)
    assert side['img_sha256'] == rs_vh.array_sha(x), 'view image array hash'
    return _array(x[_gather(stored, requested)], len(requested), width, np.float16, unit=True)


def _export(arm, split, seed, head, requested):
    percept = arm == 'percept'
    name, key = ('P3', 'prob') if percept else (head, 'img')
    path = Path(ac_paths.OUTPUTS) / 'exports' / arm / split / f'{name}__s{seed}.npz'
    side = _json(path.with_suffix('.json'), ('keys_sha256', 'array_sha256', 'weights_sha256',
                                           'split', 'seed', 'smoke', 'device' if percept else 'aspect'))
    # Spec §3.3 L124 (C6): the positional export must attest to the full key list.
    assert side['keys_sha256'] == rs_vh.names_sha(requested), 'export keys hash'
    assert side['split'] == split and side['seed'] == seed and side['smoke'] is False, 'export identity'
    weights = side['weights_sha256']
    # Ticket 04 gates the actual frozen weights; this boundary requires its hash record.
    assert isinstance(weights, str) and len(weights) == 64 and all(c in '0123456789abcdef' for c in weights), 'weights hash'
    if percept:
        assert side['device'] == 'cuda', 'PercepT training device (spec §3.7 L190)'
    else:
        assert side['aspect'] == head, 'cstar head'
    width, dtype = (40, np.float32) if percept else (2048, np.float16)
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == {key}, 'export array keys'
        x = _array(archive[key], len(requested), width, dtype, unit=not percept)
    assert side['array_sha256'] == rs_vh.array_sha(x), 'export array hash'
    if percept:
        assert ((x >= 0) & (x <= 1)).all(), 'PercepT probability range'
    return x


def _crl_text(encoder, wording, aspect):
    assert encoder in ('clip', 'qwen') and wording in ('painting', 'photo') and aspect in ac_paths.ASPECTS
    lists = _json(ac_paths.R11_CRL_LISTS_FILE, ('aspects',))
    assert set(lists['aspects']) == set(ac_paths.ASPECTS), 'frozen CRL aspects'
    entry = lists['aspects'][aspect]
    assert isinstance(entry, dict) and {'values', 'count'} <= set(entry), 'frozen CRL value fields'
    values = entry['values']
    _key_list(values)
    assert entry['count'] == len(values) > 0, 'frozen CRL value count'
    path = Path(ac_paths.OUTPUTS) / 'crl' / f'{encoder}_{wording}_{aspect}.npy'
    side = _json(path.with_suffix('.json'), ('texts', 'encoder', 'model_revision'))
    # Spec §3.2 L104–105, §3.7 L189: use r11's frozen values in their original order.
    assert side['texts'] == [f'a {wording} with the {aspect} of {value}' for value in values], 'CRL value texts/order'
    assert side['encoder'] == encoder, 'CRL text encoder'
    revision = rt_crl.CLIP_REVISION if encoder == 'clip' else rt_data.ct2_data.QWEN_REVISION
    assert side['model_revision'] == revision, 'CRL text revision'
    width = 768 if encoder == 'clip' else 2048
    x = np.load(path, allow_pickle=False)
    _array(x, len(values), width, np.float32, unit=True)
    # The index promises no hash field for CRL texts. Validate either conventional
    # array hash if supplied; exact frozen texts and pinned revision are mandatory.
    for field in ('array_sha256', 'sha256'):
        if field in side:
            assert side[field] == rs_vh.array_sha(x), 'CRL text array hash'
    if 'lists_sha256' in side:
        assert side['lists_sha256'] == rs_vh.file_sha(ac_paths.R11_CRL_LISTS_FILE), 'CRL frozen lists hash'
    if 'shape' in side:
        assert side['shape'] == list(x.shape), 'CRL text shape'
    if 'dtype' in side:
        assert side['dtype'] == 'float32', 'CRL text dtype'
    return x


def _crl(img, encoder, wording, aspect):
    # Ticket 05 owns this operation: spec §3.2 L104, float64 product/normalization,
    # one float32 cast, exactly as r11. Never normalize its float32 result again.
    import ac_crl
    text = _crl_text(encoder, wording, aspect)
    _array(img, len(img), text.shape[1], np.float32, unit=True)
    return _array(ac_crl.vectors(img, text), len(img), len(text), np.float32, unit=True)


def embed(arm, split, seed=None, *, head='emotion'):
    """Unit float32 rows in keys(split) order; cstar supports the switch-column head."""
    assert arm in ARMS and split in ac_paths.SPLITS, (arm, split)
    assert head in ac_paths.ASPECTS and (arm == 'cstar' or head == 'emotion'), (arm, head)
    if arm in SEEDED:
        assert isinstance(seed, (int, np.integer)) and not isinstance(seed, (bool, np.bool_)), (arm, seed)
        assert seed in ac_paths.SEEDS, (arm, seed)
    else:
        assert seed is None, (arm, seed)
    requested = keys(split)
    if arm == 'plain':
        return _unit(_plain(requested))
    if arm in ('teacher_artwork', 'teacher_photo', 'clip'):
        return _unit(_view(arm, requested))
    if arm in SEEDED:
        return _unit(_export(arm, split, seed, head, requested))
    _, encoder, wording = arm.split('_')
    return _crl(embed('clip' if encoder == 'clip' else 'plain', split), encoder, wording, 'emotion')


def artelingo_crl_qwen(split, aspect):
    """Painting-worded Qwen CRL in frozen r11 names(split) order, for all aspects."""
    assert split in ('val', 'held') and aspect in ac_paths.ASPECTS, (split, aspect)
    requested = rt_data.names(split)
    _key_list(requested)
    assert requested == sorted(requested), 'ArtELingo name order'
    # Spec §3.2 L105: r11 owns the frozen key/hash join for every old ArtELingo arm.
    img = _array(rt_data.embed('plain', split, aspect), len(requested), 2048, np.float32, unit=True)
    return _crl(img, 'qwen', 'painting', aspect)
