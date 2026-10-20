"""Stage 2 painting roles and name/extraction-ID joined Qwen features.

Providers take a split2 dictionary. Images and painting caption means are L2
normalized float32; per_caption=True returns raw float32 captions, as PercepT
normalizes them during fusion. Store paths are globals for model-free tests.
"""
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '20261017_image_transfer_pilot'))
import itp_data
import pav_sample
from apc_data import l2, paint_mean
from src.data.artelingo_splits import encode_labels
from src.data.splits import grouped_split
from src.data.wikiart_genre import load_wikiart_genre
from src.utils.feature_manager import FeatureManager

SAMPLE = Path('/project/CoSiR/src/test/20261017_prompted_aspect_views/sample.npz')
QWEN = Path('/data/SSD2/pre_extract/artelingo_full/qwen3vl_emb_2b')
ASPECT = Path('/data/SSD2/pre_extract/artelingo_full/qwen3vl_emb_2b_aspect_prompts')
ANNOTATIONS = Path('/data/PDD/artelingo/artelingo_train.json')
CLIP = Path('/data/SSD2/pre_extract/artelingo/features')
ROBERTA = Path('/data/SSD2/pre_extract/artelingo_full/roberta_go_emotions')
QWEN_REVISION = '9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda'
ROBERTA_REVISION = 'd75048347613a25d77de8cf6412eaae9fa7b26be'
FULL_COUNTS = ((42969, 216107), (6152, 30872), (12281, 61744))
FULL_NAMES_SHA256 = (
    '921b6b7c18ff2cf3c41bf06acf9b1f787d8b9f927ed728710944403e3e35f118',
    '1430d4b56e4a76dbe3e1c4d699e1db5c92967c5f2048b74b6cbcdf1589809e2e',
    'a0b1139c6056fd858ddbd2ee1dd4bafd1453e590818c6aa59883364c0288a876')


def _names_sha256(names):
    return hashlib.sha256('\n'.join(names).encode('utf-8')).hexdigest()


def split2(mode):
    """Frozen sample roles, or sorted train names followed by sorted val names.

    Full captions ascend by extraction ID within each painting. Full subset_pid
    is empty: validation is dev, never an evaluation subset. Sheet writers must
    explicitly select dev instead. rows retains the CLIP loader row mapping.
    """
    assert mode in ('sample', 'full'), mode
    if mode == 'sample':
        with patch.object(pav_sample, 'SAMPLE', SAMPLE):
            s = pav_sample.load_frozen()
        _, fit, dev = itp_data.split(s)
    else:
        records = json.loads(ANNOTATIONS.read_text())
        manager = FeatureManager(str(CLIP))
        assert manager.metadata['backbone_model'] == 'openai/clip-vit-base-patch32'
        ids = np.asarray(manager.get_all_sample_ids())
        assert ids.dtype == np.int64 and ids.shape == (len(records),)
        assert len(np.unique(ids)) == len(ids) and ((ids >= 0) & (ids < len(records))).all()
        names = np.asarray([records[int(i)]['painting'] for i in ids])
        # Q9 and ct_held: exact-image groups were verified to equal painting
        # groups; preserve their first-occurrence order in the CLIP ID index.
        parts = grouped_split(pd.factorize(names, sort=False)[0], seed=42)
        roles = [getattr(parts, part) for part in ('train', 'val', 'held')]
        role_names = [np.asarray(sorted(set(names[rows]))) for rows in roles]
        for rows, ns, counts, digest in zip(roles, role_names, FULL_COUNTS, FULL_NAMES_SHA256):
            assert (len(ns), len(rows)) == counts
            assert _names_sha256(ns) == digest, 'split name hash mismatch'
        for a, b in ((0, 1), (0, 2), (1, 2)):
            assert not np.intersect1d(role_names[a], role_names[b]).size
            assert not np.intersect1d(ids[roles[a]], ids[roles[b]]).size
        all_names = np.concatenate(role_names[:2])
        lookup = {name: row for row, name in enumerate(all_names)}
        rows = np.concatenate(roles[:2])
        pid = np.asarray([lookup[name] for name in names[rows]], dtype=np.int64)
        order = np.lexsort((ids[rows], pid))
        rows, pid = rows[order], pid[order]
        first = np.flatnonzero(np.r_[True, pid[1:] != pid[:-1]])
        emotion, _ = encode_labels([r['emotion'] for r in records])
        style, _ = encode_labels([r['art_style'] for r in records])
        selected_ids = ids[rows]
        empty = np.empty(0, dtype=np.int64)
        s = dict(sample_ids=selected_ids, rows=rows, pid=pid, first=first,
                 all_painting_ids=all_names, painting_ids=names[rows],
                 subset_pid=empty, subset_image_indices=empty.copy(),
                 subset_painting_ids=np.empty(0, dtype=all_names.dtype),
                 emo=emotion[selected_ids], sty=style[selected_ids],
                 gen=load_wikiart_genre(names[rows].astype(object)))
        assert np.array_equal(s['painting_ids'][first], all_names)
        fit = np.arange(len(role_names[0]), dtype=np.int64)
        dev = np.arange(len(fit), len(all_names), dtype=np.int64)
    s.update(mode=mode, fit=fit, dev=dev,
             fit_pairs=np.flatnonzero(np.isin(s['pid'], fit)),
             names_sha256={role: _names_sha256(s['all_painting_ids'][rows])
                           for role, rows in (('fit', fit), ('dev', dev), ('eval', s['subset_pid']))})
    return s


def _image_store(path):
    if (path / 'index.json').exists():
        images = json.loads((path / 'index.json').read_text())['images']
        names = np.asarray([row['painting'] for row in images])
    else:
        with np.load(path / 'ids.npz', allow_pickle=False) as ids:
            key = 'image_painting_ids' if 'image_painting_ids' in ids else 'painting_names'
            names = ids[key]
    assert names.ndim == 1 and len(np.unique(names)) == len(names), 'duplicate painting names'
    raw = np.load(path / 'img.npy', mmap_mode='r')
    assert raw.shape == (len(names), 2048) and raw.dtype == np.float16
    return names, raw


def _cosine_gate(a, b):
    cosine = np.sum(l2(a) * l2(b), axis=1)
    assert np.isfinite(cosine).all()
    assert cosine.mean() > 0.999 and cosine.min() > 0.99, 'overlap cosine gate failed'


def default_images(s):
    """Full default store by painting name, gated against A/default's overlap."""
    names, raw = _image_store(QWEN)
    lookup = {name: row for row, name in enumerate(names)}
    wanted = s['all_painting_ids']
    assert all(name in lookup for name in wanted), 'missing default images'
    rows = np.asarray([lookup[name] for name in wanted], dtype=np.int64)
    x = np.asarray(raw[rows], dtype=np.float32)
    assert np.isfinite(x).all(), 'nonfinite features'
    assert (np.linalg.norm(x, axis=1) > 0).all(), 'zero default images'
    x = l2(x)
    assert np.isfinite(x).all()
    names, reference = _image_store(_sealed(ASPECT / 'default', 'default'))
    target = {name: row for row, name in enumerate(wanted)}
    overlap = [(target[name], row) for row, name in enumerate(names) if name in target]
    if overlap:
        a, b = np.asarray(overlap).T
        _cosine_gate(x[a], np.asarray(reference[b], dtype=np.float32))
    return x


def _caption_store(path):
    if (path / 'index.json').exists():
        texts = json.loads((path / 'index.json').read_text())['texts']
        sample_ids = np.asarray([row['sample_id'] for row in texts], dtype=np.int64)
        names = np.asarray([row['painting'] for row in texts])
    else:
        with np.load(path / 'ids.npz', allow_pickle=False) as ids:
            sample_ids = ids['sample_ids']
            names = ids['caption_painting_ids']
    assert sample_ids.dtype == np.int64 and sample_ids.ndim == 1
    assert len(np.unique(sample_ids)) == len(sample_ids), 'duplicate caption IDs'
    assert names.shape == sample_ids.shape
    raw = np.load(path / 'txt.npy', mmap_mode='r')
    assert raw.shape == (len(sample_ids), 2048) and raw.dtype == np.float16
    return sample_ids, names, raw


def _sealed(path, key):
    done = json.loads((path / 'done.json').read_text())
    assert done['instruction'] == itp_data.PROMPTS[key], 'instruction mismatch'
    assert not done.get('smoke', False), 'smoke store'
    if path.name.endswith(('_s2img', '_s2txt')):
        check_s2_store(path)
    return path


def ids_sha256(ids):
    """Hash the primary ordered IDs as newline-joined names or decimal IDs (Q1)."""
    values = ids['painting_names'] if 'painting_names' in ids else ids['sample_ids']
    return hashlib.sha256('\n'.join(map(str, values)).encode('utf-8')).hexdigest()


def check_s2_store(path):
    """Verify an extraction's parity, revision, row counts and ordered ID hash.

    Accepts Qwen s2img/s2txt and RoBERTa _s2 stores; returns their done metadata.
    Legacy Qwen training manifests omit revisions, so use the extraction's frozen
    revision there; a recorded legacy revision takes precedence over the fallback.
    """
    path = Path(path)
    done = json.loads((path / 'done.json').read_text())
    assert done['parity_gate']['passed'] is True, 'stage 2 parity gate failed'
    assert not done.get('smoke', False), 'smoke store'
    store = done['store']
    image = store.endswith('_img')
    if store == 'roberta':
        references, fallback = [ROBERTA], ROBERTA_REVISION
        width, dtype = 768, np.float32
    else:
        key, kind = store.rsplit('_', 1)
        assert key in ('emotion_focus', 'style_word', 'genre_word') and kind in ('img', 'txt')
        references, fallback = [ASPECT / key], QWEN_REVISION
        if image:
            references.append(ASPECT / f'{key}_trainimg')
        width, dtype = 2048, np.float16
    revisions = set()
    for reference in references:
        for name in ('done.json', 'meta.json'):
            manifest = reference / name
            if manifest.exists():
                legacy = json.loads(manifest.read_text())
                revision = legacy.get('model_revision', legacy.get('snapshot_revision'))
                if revision is not None:
                    revisions.add(revision)
    assert len(revisions) <= 1, 'legacy model revision mismatch'
    expected = next(iter(revisions)) if revisions else fallback
    assert done['model_revision'] == expected, 'stage 2 model revision mismatch'
    raw = np.load(path / ('img.npy' if image else 'txt.npy'), mmap_mode='r')
    with np.load(path / 'ids.npz', allow_pickle=False) as z:
        ids = {key: z[key] for key in z.files}
    primary = ids['painting_names' if image else 'sample_ids']
    assert raw.ndim == 2 and raw.shape[1] == width and raw.dtype == dtype, 'stage 2 array shape/dtype'
    assert done['count'] == len(primary) == raw.shape[0], 'stage 2 row count mismatch'
    assert all(value.ndim == 1 and len(value) == len(primary) for value in ids.values()), 'stage 2 ID row count mismatch'
    assert done['shape'] == list(raw.shape), 'stage 2 shape mismatch'
    assert done['ids_sha256'] == ids_sha256(ids), 'stage 2 ID hash mismatch'
    return done


def _caption_union(s, paths):
    wanted = s['sample_ids']
    assert len(np.unique(wanted)) == len(wanted)
    lookup = {int(sid): row for row, sid in enumerate(wanted)}
    x = np.empty((len(wanted), 2048), dtype=np.float32)
    seen = np.zeros(len(wanted), dtype=bool)
    for path in paths:
        ids, names, raw = _caption_store(path)
        source = np.asarray([row for row, sid in enumerate(ids) if int(sid) in lookup], dtype=np.int64)
        target = np.asarray([lookup[int(ids[row])] for row in source], dtype=np.int64)
        assert np.array_equal(names[source], s['painting_ids'][target]), 'caption painting mismatch'
        cosines = []
        for start in range(0, len(source), 4096):
            src, dst = source[start:start + 4096], target[start:start + 4096]
            values = np.asarray(raw[src], dtype=np.float32)
            assert np.isfinite(values).all(), 'nonfinite features'
            assert (np.linalg.norm(values, axis=1) > 0).all(), 'zero caption features'
            overlap = seen[dst]
            if overlap.any():
                cosines.append(np.sum(l2(x[dst[overlap]]) * l2(values[overlap]), axis=1))
            x[dst[~overlap]] = values[~overlap]
            seen[dst] = True
        if cosines:
            cosine = np.concatenate(cosines)
            assert cosine.mean() > 0.999 and cosine.min() > 0.99, 'overlap cosine gate failed'
    assert seen.all(), f'missing captions: {int((~seen).sum())}'
    return x


def default_captions(s, *, per_caption=False):
    """Prefer A/default's raw captions; use Qwen full for remaining IDs."""
    x = _caption_union(s, [_sealed(ASPECT / 'default', 'default'), QWEN])
    return x if per_caption else l2(paint_mean(x, s['pid'], len(s['first'])))


def instructed_images(s, key):
    """Prefer A/key, then key_trainimg, then key_s2img, joined by name."""
    assert key in ('emotion_focus', 'style_word', 'genre_word'), key
    paths = [ASPECT / key, ASPECT / f'{key}_trainimg']
    extra = ASPECT / f'{key}_s2img'
    if extra.exists():
        paths.append(extra)
    wanted = s['all_painting_ids']
    assert len(np.unique(wanted)) == len(wanted)
    lookup = {name: row for row, name in enumerate(wanted)}
    x = np.empty((len(wanted), 2048), dtype=np.float32)
    seen = np.zeros(len(wanted), dtype=bool)
    for path in paths:
        names, raw = _image_store(_sealed(path, key))
        source = np.asarray([row for row, name in enumerate(names) if name in lookup], dtype=np.int64)
        target = np.asarray([lookup[names[row]] for row in source], dtype=np.int64)
        values = np.asarray(raw[source], dtype=np.float32)
        assert np.isfinite(values).all(), 'nonfinite features'
        assert (np.linalg.norm(values, axis=1) > 0).all(), 'zero instructed images'
        overlap = seen[target]
        if overlap.any():
            _cosine_gate(x[target[overlap]], values[overlap])
        x[target[~overlap]] = values[~overlap]
        seen[target] = True
    assert seen.all(), f'missing {key} images: {int((~seen).sum())}'
    return l2(x)


def instructed_captions(s, key, *, per_caption=False):
    """Raw instructed captions or normalized means; only emotion has an S2 store."""
    assert key in ('emotion_focus', 'style_word', 'genre_word'), key
    paths = [_sealed(ASPECT / key, key)]
    extra = ASPECT / 'emotion_focus_s2txt'
    if key == 'emotion_focus' and extra.exists():
        paths.append(_sealed(extra, key))
    x = _caption_union(s, paths)
    return x if per_caption else l2(paint_mean(x, s['pid'], len(s['first'])))


def missing_ids(store):
    """Sorted missing painting_names/sample_ids for train+val, and per role.

    store is a key (emotion_focus/style_word/genre_word/default/roberta), a
    directory, or an explicit list of directories. Keys and base directories
    include existing trainimg/s2img/s2txt siblings (RoBERTa includes its _s2).
    Explicit lists inspect exactly those stores, useful for auditing Q1's old
    coverage. Caption-only stores contribute their caption painting names.
    Return keys: painting_names, sample_ids, train, val; each role contains the
    same two arrays. IDs are extraction IDs, sorted numerically, never row IDs.
    This metadata-only query reads neither feature values nor held artifacts.
    """
    if isinstance(store, (list, tuple)):
        paths = [Path(path) for path in store]
    else:
        if str(store) == 'roberta':
            base = ROBERTA
        elif str(store) in ('default', 'emotion_focus', 'style_word', 'genre_word'):
            base = ASPECT / str(store)
        else:
            base = Path(store)
        paths = [base]
        if base.parent == ASPECT and base.name in ('emotion_focus', 'style_word', 'genre_word'):
            paths.extend(path for suffix in ('trainimg', 's2img', 's2txt')
                         if (path := ASPECT / f'{base.name}_{suffix}').exists())
        elif base == ASPECT / 'default':
            paths.append(QWEN)
        elif base == ROBERTA and (extra := ROBERTA.with_name(ROBERTA.name + '_s2')).exists():
            paths.append(extra)
    image_names, caption_names, known_ids = set(), set(), set()
    has_images = False
    for path in paths:
        if (path / 'index.json').exists():
            index = json.loads((path / 'index.json').read_text())
            has_images = True
            image_names.update(row['painting'] for row in index['images'])
            known_ids.update(int(row['sample_id']) for row in index['texts'])
        else:
            with np.load(path / 'ids.npz', allow_pickle=False) as ids:
                for key in ('image_painting_ids', 'painting_names'):
                    if key in ids:
                        has_images = True
                        names = ids[key]
                        assert names.ndim == 1 and len(np.unique(names)) == len(names)
                        image_names.update(names.tolist())
                if 'sample_ids' in ids:
                    sample_ids = ids['sample_ids']
                    key = 'caption_painting_ids' if 'caption_painting_ids' in ids else 'painting_ids'
                    assert sample_ids.dtype == np.int64 and sample_ids.ndim == 1
                    assert len(np.unique(sample_ids)) == len(sample_ids)
                    assert ids[key].shape == sample_ids.shape
                    known_ids.update(sample_ids.tolist())
                    caption_names.update(ids[key].tolist())
    known_names = image_names if has_images else caption_names
    s = split2('full')
    result = {}
    for role, rows in (('train', s['fit']), ('val', s['dev'])):
        pairs = np.flatnonzero(np.isin(s['pid'], rows))
        result[role] = dict(
            painting_names=np.asarray(sorted(set(s['all_painting_ids'][rows]) - known_names), dtype=str),
            sample_ids=np.asarray(sorted(set(s['sample_ids'][pairs]) - known_ids), dtype=np.int64))
    for key in ('painting_names', 'sample_ids'):
        result[key] = np.sort(np.concatenate([result['train'][key], result['val'][key]]))
    return result
