"""Aspect-instruction training images (14,889 non-evaluation paintings); the itp extraction, any instruction.

Controller-run on the GPU. Help, import and `aspect_images` are model-free.
"""
import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ITP = HERE.parent / '20261017_image_transfer_pilot'
sys.path.insert(0, str(ITP))
from itp_data import (OUTPUT, PROMPTS, atomic_json, fingerprint, l2, load_frozen, prompted, split,
                      train_image_ids)
from itp_extract import encode

KEYS = ('emotion_focus', 'style_word', 'genre_word')
SMOKE = Path('/tmp/claude-0/-project-CoSiR/c1d9190f-c197-43c9-bb25-2692e4200e97/scratchpad/ags_smoke')


def train_path(key, smoke=False):
    return SMOKE / f'{key}_trainimg' if smoke else OUTPUT / f'{key}_trainimg'


def validate_train_images(s, key, smoke=False, expected=None):
    """itp_data.validate_train_images with the instruction as a parameter."""
    import json
    from scripts.extract_features import check_arrays
    path = train_path(key, smoke)
    done = json.loads((path / 'done.json').read_text())
    assert done['instruction'] == PROMPTS[key]
    assert done['sample_sha256'] == fingerprint(s)
    assert done['smoke'] == smoke
    assert done['subset_gate']['mean_cosine'] > 0.999
    dense = split(s)[0][:16] if smoke else split(s)[0]
    with np.load(path / 'ids.npz', allow_pickle=False) as z:
        ids = {k: z[k] for k in z.files}
    assert np.array_equal(ids['dense_indices'], dense)
    assert np.array_equal(ids['painting_names'], s['all_painting_ids'][dense])
    assert len(np.unique(ids['image_indices'])) == len(dense)
    if expected is not None:
        for k, v in expected.items():
            assert np.array_equal(ids[k], v), f'{path}: {k} mismatch'
    assert done['ids_sha256'] == hashlib.sha256(b''.join(
        np.ascontiguousarray(ids[k]).tobytes() for k in sorted(ids))).hexdigest()
    x = np.load(path / 'img.npy', mmap_mode='r')
    assert x.shape == (len(dense), 2048) and x.dtype == np.float16
    gate = check_arrays(x, np.empty((0, 2048), dtype=np.float16))
    assert gate['ok'], gate
    return x, ids


def aspect_images(s, key):
    """All 19,889 paintings in painting order: training images from <key>_trainimg, evaluation from the store."""
    train, ids = validate_train_images(s, key)
    x = np.empty((len(s['first']), 2048), dtype=np.float32)
    x[ids['dense_indices']] = l2(np.asarray(train, dtype=np.float32))
    x[s['subset_pid']] = l2(np.asarray(prompted(s, key)[0], dtype=np.float32))
    assert np.isfinite(x).all()
    return x


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--key', required=True, choices=KEYS[1:])
    p.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    p.add_argument('--smoke', action='store_true', help='16 training images into the scratchpad')
    args = p.parse_args()
    s = load_frozen()
    path = train_path(args.key, args.smoke)
    if (path / 'done.json').exists():
        validate_train_images(s, args.key, smoke=args.smoke)
        print(f'{path}: verified sealed output; skip', flush=True)
        return
    from scripts.extract_features import adapter_artelingo_full, check_arrays
    spec = adapter_artelingo_full(sample_ids=s['sample_ids'], expected=None)
    assert np.array_equal([m['sample_id'] for m in spec.text_meta], s['sample_ids'])
    adapter_indices = np.asarray([m['img'] for m in spec.text_meta])[s['first']]
    assert np.array_equal([spec.images[int(i)]['painting'] for i in adapter_indices], s['all_painting_ids'])
    assert len(np.unique(adapter_indices)) == len(s['first'])
    assert np.array_equal(adapter_indices[s['subset_pid']], s['subset_image_indices'])
    ids = train_image_ids(s, adapter_indices)
    if args.smoke:
        ids = {k: v[:16] for k, v in ids.items()}
    reference = prompted(s, args.key)[0]
    from src.data.feature_extract import Qwen3VLEmb
    start = time.monotonic()
    enc = Qwen3VLEmb(device=args.device, instruction=PROMPTS[args.key])
    # Gate: the first 32 evaluation images re-encode to the stored prompted vectors.
    test = np.asarray(encode(enc, spec, s['subset_image_indices'][:32]), dtype=np.float32)
    ref = np.asarray(reference[:32], dtype=np.float32)
    cosine = np.sum(test * ref, axis=1) / (np.linalg.norm(test, axis=1) * np.linalg.norm(ref, axis=1))
    subset_gate = dict(mean_cosine=float(cosine.mean()), min_cosine=float(cosine.min()), n=32)
    print(f"{args.key} subset re-encode gate: mean_cosine={subset_gate['mean_cosine']:.8f}", flush=True)
    assert subset_gate['mean_cosine'] > 0.999, subset_gate
    path.mkdir(parents=True, exist_ok=True)
    ni = len(ids['dense_indices'])
    img = np.lib.format.open_memmap(path / 'img.npy', mode='w+', dtype=np.float16, shape=(ni, 2048))
    image_start = time.monotonic()
    for a in range(0, ni, 32):
        batch = encode(enc, spec, ids['image_indices'][a:a + 32])
        img[a:a + len(batch)] = batch
        if a % 3200 == 0:
            print(f'{args.key}: {a + len(batch)}/{ni} [{time.monotonic() - image_start:.0f}s]', flush=True)
    img.flush()
    seconds = time.monotonic() - image_start
    gate = check_arrays(img, np.empty((0, 2048), dtype=np.float16))
    assert gate['ok'], gate
    np.savez(path / 'ids.npz', **ids)
    done = dict(instruction=PROMPTS[args.key], sample_sha256=fingerprint(s),
                ids_sha256=hashlib.sha256(b''.join(np.ascontiguousarray(ids[k]).tobytes() for k in sorted(ids))).hexdigest(),
                smoke=args.smoke, n_images=ni, image_seconds=seconds, total_seconds=time.monotonic() - start,
                image_items_per_second=ni / seconds, subset_gate=subset_gate, gate=gate)
    atomic_json(path / 'done.json', done)
    validate_train_images(s, args.key, smoke=args.smoke, expected=ids)
    print(f'{args.key}: {ni} images; {seconds:.1f}s; {ni / seconds:.2f} images/s; sealed {path}', flush=True)


if __name__ == '__main__':
    main()
