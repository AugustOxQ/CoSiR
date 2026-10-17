"""Shared paths, positional validation and label-free student data construction."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PAV = HERE.parent / '20261017_prompted_aspect_views'
APC = HERE.parent / '20261017_aspect_premise_checks'
for path in (ROOT, APC, PAV):
    sys.path.insert(0, str(path))

from pav_sample import load_frozen
from pav_extract import OUTPUT, identity, validate_key
from pav_prompts import PROMPTS
from apc_data import l2, paint_mean

STORE = Path('/data/SSD2/pre_extract/artelingo_full/qwen3vl_emb_2b')
TRAIN_IMAGES = OUTPUT / 'emotion_focus_trainimg'
SMOKE = Path('/tmp/claude-0/-project-CoSiR/itp_smoke')


def fingerprint(s):
    """Identity only; labels never enter a training identity or decision."""
    return hashlib.sha256(b''.join(np.ascontiguousarray(s[k]).tobytes() for k in
        ('sample_ids', 'pid', 'first', 'all_painting_ids', 'subset_pid', 'subset_image_indices'))).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + '.pending.json')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def split(s):
    """Ascending non-evaluation paintings; fixed 10% dev, independent of run seed."""
    train_all = np.setdiff1d(np.arange(len(s['first'])), s['subset_pid'])
    assert len(train_all) == 14889
    order = np.random.default_rng(0).permutation(train_all)
    dev, fit = order[:1489], order[1489:]
    assert len(fit) == 13400 and not np.intersect1d(fit, dev).size
    assert not np.intersect1d(train_all, s['subset_pid']).size
    return train_all, fit, dev


def prompted(s, key):
    return validate_key(OUTPUT / key, identity(s, 5000, 99999), PROMPTS[key])


def default_images(s, reference=None):
    """Name join for all paintings; subset always uses this same source."""
    with (STORE / 'index.json').open() as f:
        images = json.load(f)['images']
    names = [m['painting'] for m in images]
    assert len(names) == len(set(names)), 'duplicate painting names in default store'
    lookup = {name: i for i, name in enumerate(names)}
    indices = np.asarray([lookup[str(name)] for name in s['all_painting_ids']], dtype=np.int64)
    raw = np.load(STORE / 'img.npy', mmap_mode='r')
    assert raw.shape == (61402, 2048) and raw.dtype == np.float16
    x = l2(np.asarray(raw[indices], dtype=np.float32))
    assert np.isfinite(x).all()
    if reference is None:
        reference = prompted(s, 'default')[0]
    cosine = np.sum(x[s['subset_pid']] * l2(np.asarray(reference, dtype=np.float32)), axis=1)
    gate = dict(mean=float(cosine.mean()), min=float(cosine.min()))
    print(f"default store join: cosine mean={gate['mean']:.8f} min={gate['min']:.8f}", flush=True)
    assert gate['mean'] > 0.999 and gate['min'] > 0.99, gate
    return x, gate


def targets(s, key, txt=None):
    if txt is None:
        txt = prompted(s, key)[1]
    return l2(paint_mean(np.asarray(txt, dtype=np.float32), s['pid'], len(s['first'])))


def train_image_ids(s, adapter_indices):
    dense = split(s)[0]
    return dict(dense_indices=dense, painting_names=s['all_painting_ids'][dense],
                image_indices=np.asarray(adapter_indices)[dense])


def validate_train_images(s, path=TRAIN_IMAGES, expected=None, smoke=False):
    """Verify sealed extraction without constructing an adapter or a model."""
    from scripts.extract_features import check_arrays
    path = Path(path)
    done = json.loads((path / 'done.json').read_text())
    assert done['instruction'] == PROMPTS['emotion_focus']
    assert done['sample_sha256'] == fingerprint(s)
    assert done['smoke'] == smoke
    assert done['subset_gate']['mean_cosine'] > 0.999
    dense = split(s)[0][:16] if smoke else split(s)[0]
    with np.load(path / 'ids.npz', allow_pickle=False) as z:
        ids = {k: z[k] for k in z.files}
    assert np.array_equal(ids['dense_indices'], dense)
    assert np.array_equal(ids['painting_names'], s['all_painting_ids'][dense])
    assert len(np.unique(ids['image_indices'])) == len(dense)
    assert np.issubdtype(ids['image_indices'].dtype, np.integer) and (ids['image_indices'] >= 0).all()
    if expected is not None:
        for key, value in expected.items():
            assert np.array_equal(ids[key], value), f'{path}: {key} mismatch'
    # Bind the full positional adapter mapping even on a model-free resume.
    assert done['ids_sha256'] == hashlib.sha256(b''.join(
        np.ascontiguousarray(ids[k]).tobytes() for k in sorted(ids))).hexdigest()
    x = np.load(path / 'img.npy', mmap_mode='r')
    assert x.shape == (len(dense), 2048) and x.dtype == np.float16
    gate = check_arrays(x, np.empty((0, 2048), dtype=np.float16))
    assert gate['ok'], gate
    return x, ids


def emotion_images(s):
    train, ids = validate_train_images(s)
    image = prompted(s, 'emotion_focus')[0]
    x = np.empty((len(s['first']), 2048), dtype=np.float32)
    x[ids['dense_indices']] = l2(np.asarray(train, dtype=np.float32))
    x[s['subset_pid']] = l2(np.asarray(image, dtype=np.float32))
    assert np.isfinite(x).all()
    return x
