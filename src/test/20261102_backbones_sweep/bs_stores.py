"""r14 store reader: `load(dataset, view_or_encoder, keys) -> float32 [len(keys), d]`, joined by key (spec §3.5 0(ii)).

Datasets: 'affection' (keys `<source>/<image_name>`), 'emoset' (keys are the split-file path strings, for example
`image/awe/awe_08973.jpg`), 'artelingo' (key = the `image` path string of the store's index.json; default-image
encoder stores only). Views: 'qwen_default', 'emotion_focus', 'emotion_focus_photo', 'clip_l14', 'siglip2', 'siglip1'.

Two folder formats: ac format (img.npy float16 + keys.json) and index format (img.npy + index.json with an `images`
list, the Qwen Affection store and the ArtELingo stores). A view may span several folders (Affection CLIP L/14: val and
test, then train; Affection emotion_focus: val and test, then train); a folder that does not exist yet is skipped, and
every requested key must be found in exactly one of the folders that exist, else the load asserts.

New stores (r14 extractions) live under the store root: /data/SSD2/pre_extract, or BS_STORE_ROOT for a smoke run
(`<root>/affection_local/...`, `<root>/emoset/...`). Stores of older rounds are always read from /data/SSD2/pre_extract.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bs_paths as BP  # noqa: E402

VIEWS = ('qwen_default', 'emotion_focus', 'emotion_focus_photo', 'clip_l14', 'siglip2', 'siglip1')
QWEN_VIEWS = ('qwen_default', 'emotion_focus', 'emotion_focus_photo')
WIDTH = {'qwen_default': 2048, 'emotion_focus': 2048, 'emotion_focus_photo': 2048, 'clip_l14': 768, 'siglip2': 1152,
         'siglip1': 1152}
AFF_VIEWS = BP.AFF / 'qwen3vl_emb_2b_aspect_prompts'   # r12/r13 instructed Affection views (old stores)


def store_root():
    """Root of the NEW stores: the real pre_extract folder, or the scratch root of a smoke run."""
    return Path(os.environ.get('BS_STORE_ROOT') or BP.PRE)


def new_folder(dataset, view):
    """The folder an r14 extraction of (dataset, view) writes, under the store root; None where r14 extracts none."""
    root = store_root()
    if dataset == 'emoset':
        if view in QWEN_VIEWS:
            return root / 'emoset' / BP.EMO_QWEN_VIEWS['default' if view == 'qwen_default' else view]
        return root / 'emoset' / BP.ENCODERS[view]['emo']
    assert dataset == 'affection'
    if view in ('siglip2', 'siglip1'):
        return root / 'affection_local' / BP.ENCODERS[view]['aff'][0]
    if view == 'clip_l14':
        return root / 'affection_local' / BP.ENCODERS['clip_l14']['aff'][1]    # train keys (valtest is an old store)
    return None


def folders(dataset, view):
    """List of store folders (in lookup order) that hold `view` of `dataset`."""
    assert view in VIEWS, view
    if dataset == 'emoset':
        return [new_folder('emoset', view)]
    if dataset == 'artelingo':
        assert view in ('qwen_default', 'clip_l14', 'siglip2', 'siglip1'), f'no index-format ArtELingo store for {view}'
        enc = 'qwen' if view == 'qwen_default' else view
        return [BP.ART / BP.ENCODERS[enc]['art']]
    assert dataset == 'affection', dataset
    if view == 'qwen_default':
        return [BP.AFF / BP.ENCODERS['qwen']['aff'][0]]
    if view == 'emotion_focus':
        return [AFF_VIEWS / 'emotion_focus', AFF_VIEWS / 'emotion_focus_train']
    if view == 'emotion_focus_photo':
        return [AFF_VIEWS / 'emotion_focus_photo']
    if view == 'clip_l14':
        return [BP.AFF / BP.ENCODERS['clip_l14']['aff'][0], new_folder('affection', 'clip_l14')]
    return [new_folder('affection', view)]


def folder_keys(folder, dataset):
    """Keys of a store folder in row order (keys.json, or index.json images)."""
    folder = Path(folder)
    if (folder / 'keys.json').exists():
        keys = json.loads((folder / 'keys.json').read_text())
    else:
        images = json.loads((folder / 'index.json').read_text())['images']
        if dataset == 'affection':
            keys = [f"{m['source']}/{m['image_name']}" for m in images]
        else:
            assert dataset == 'artelingo'
            keys = [m['image'] for m in images]
    assert len(set(keys)) == len(keys), f'duplicate keys in {folder}'
    return keys


def load(dataset, view, keys, folder_list=None):
    """float32 [len(keys), d] rows of `view`, joined by key. Asserts: unique keys, each found in exactly one folder,
    finite values, row count and width. `folder_list` overrides the folders (a check of one store)."""
    keys = list(keys)
    assert keys and len(set(keys)) == len(keys), 'keys must be non-empty and unique'
    fl = [Path(f) for f in (folder_list or folders(dataset, view)) if f is not None and Path(f).exists()]
    assert fl, f'no store folder exists for {dataset}/{view}'
    found = np.full(len(keys), -1, dtype=np.int64)
    out = None
    for fi, folder in enumerate(fl):
        fk = folder_keys(folder, dataset)
        lookup = {k: i for i, k in enumerate(fk)}
        want = [(j, lookup[k]) for j, k in enumerate(keys) if k in lookup]
        if not want:
            continue
        js = np.asarray([w[0] for w in want])
        assert (found[js] < 0).all(), f'keys found in more than one folder (second: {folder})'
        found[js] = fi
        img = np.load(folder / 'img.npy', mmap_mode='r')
        assert img.ndim == 2 and img.shape[0] == len(fk), f'{folder}: rows {img.shape} vs keys {len(fk)}'
        assert img.shape[1] == WIDTH[view], f'{folder}: width {img.shape[1]}'
        if out is None:
            out = np.empty((len(keys), img.shape[1]), dtype=np.float32)
        rows = np.asarray([w[1] for w in want])
        order = np.argsort(rows)                       # sorted reads from the memmap
        out[js[order]] = np.asarray(img[rows[order]], dtype=np.float32)
    missing = [k for k, f in zip(keys, found) if f < 0]
    assert out is not None and not missing, f'{dataset}/{view}: {len(missing)} of {len(keys)} keys not found, e.g. {missing[:2]}'
    assert out.shape == (len(keys), WIDTH[view]) and np.isfinite(out).all()
    return out
