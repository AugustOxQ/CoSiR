"""Instructed Qwen images (emotion_focus, style_word, genre_word) of the 746 ArtELingo-28 paintings (ticket 08).

Store `PRE/artelingo28/qwen3vl_emb_2b_aspect_prompts/<key>/`: img.npy float16 [746, 2048], ids.npz (painting_names
only, `a28_names()` order), done.json, in the form `rb_encoders.a28_instructed` reads. Same encoder, prompts,
pinned Qwen revision, batch 32 and float16 cast as the CT's extraction (`ct2_extract.load_encoder`,
`ct_extract.encode_rows`); the images come from the ArtELingo-28 adapter. Written to `<store>.pending`, checked,
then renamed; an existing store or pending folder is never touched.
--limit N (first N sorted paintings) requires --dest, an isolated root; its stores have smoke true and are refused
by the reader.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

import rb_paths
import rb_encoders
import ct2_data
import ct2_extract
import ct_extract
from itp_data import PROMPTS

KEYS = rb_encoders.A28_KEYS
ROOT = rb_encoders.A28_ROOT / 'qwen3vl_emb_2b_aspect_prompts'
BATCH = 32
WIDTH = 2048


def a28_spec(names):
    """The adapter's A28 image list, checked to be the pinned 746 sorted names; returns (spec, rows) in `names` order."""
    from scripts.extract_features import adapter_artelingo28
    spec = adapter_artelingo28()
    stored = [m['painting'] for m in spec.images]
    assert stored == sorted(stored) and len(set(stored)) == len(stored)
    lookup = {name: row for row, name in enumerate(stored)}
    rows = np.asarray([lookup[name] for name in names], dtype=np.int64)
    assert [stored[row] for row in rows] == list(names)
    return spec, rows


def verify_store(path, key, names, smoke):
    """Re-read a written folder the way `rb_encoders.a28_instructed` does (done.json, ids.npz, img.npy)."""
    path = Path(path)
    done = json.loads((path / 'done.json').read_text())
    assert done['instruction'] == PROMPTS[key] and done['smoke'] is smoke
    assert done['model_revision'] == ct2_data.QWEN_REVISION
    assert not (path / 'index.json').exists()
    with np.load(path / 'ids.npz', allow_pickle=False) as z:
        assert z.files == ['painting_names']
        stored = z['painting_names']
    assert stored.dtype.kind == 'U' and stored.tolist() == list(names) and len(set(stored.tolist())) == len(names)
    assert done['count'] == len(names)
    assert done['ids_sha256'] == ct2_data.ids_sha256({'painting_names': stored})
    x = np.load(path / 'img.npy', mmap_mode='r')
    assert x.shape == (len(names), WIDTH) and x.dtype == np.float16
    x = np.asarray(x, dtype=np.float32)
    assert np.isfinite(x).all() and (np.linalg.norm(x, axis=1) > 0).all()
    assert np.all(np.abs(np.linalg.norm(x, axis=1) - 1) < 1e-3)
    return done


def write_store(enc, identity, key, names, spec, rows, root, smoke, batch=BATCH):
    """Encode `names` with instruction PROMPTS[key] and seal root/<key>; refuses an existing store or pending folder."""
    assert key in KEYS and len(names) > 0 and len(set(names)) == len(names)
    assert [spec.images[int(row)]['painting'] for row in rows] == list(names)
    assert identity['model_revision'] == ct2_data.QWEN_REVISION
    path = Path(root) / key
    pending = path.with_name(path.name + '.pending')
    assert not path.exists(), f'existing store: {path}'
    assert not pending.exists(), f'pending extraction exists: {pending}'
    enc.instruction = PROMPTS[key]
    pending.mkdir(parents=True)
    x = np.lib.format.open_memmap(pending / 'img.npy', mode='w+', dtype=np.float16, shape=(len(names), WIDTH))
    start = time.monotonic()
    written = 0
    for offset, value in ct_extract.encode_rows(enc, spec, rows, batch):
        assert offset == written and value.shape[1:] == (WIDTH,)
        x[offset:offset + len(value)] = value
        written += len(value)
        print(f'{key}: {written}/{len(names)}', flush=True)
    assert written == len(names)
    x.flush()
    del x
    seconds = time.monotonic() - start
    painting_names = np.asarray(names)
    np.savez(pending / 'ids.npz', painting_names=painting_names)
    done = dict(store=key, instruction=PROMPTS[key], **identity, count=written, n_images=written, n_captions=0,
                ids_sha256=ct2_data.ids_sha256({'painting_names': painting_names}), dtype='float16',
                shape=[written, WIDTH], batch_size=batch, seconds=seconds, smoke=smoke,
                img_sha256=hashlib.sha256((pending / 'img.npy').read_bytes()).hexdigest())
    (pending / 'done.json').write_text(json.dumps(done, indent=2, allow_nan=False) + '\n')
    verify_store(pending, key, names, smoke)
    assert not path.exists(), f'existing store: {path}'
    pending.rename(path)
    print(f'{key}: sealed {written} images at {path}', flush=True)
    return path


def extract(keys=KEYS, device='cuda', limit=None, dest=None):
    assert limit is None or (limit > 0 and dest is not None), '--limit requires an isolated --dest'
    root = ROOT if dest is None else Path(dest)
    assert dest is None or Path(dest).resolve().is_relative_to('/tmp/claude-0'), 'smoke root must be under /tmp/claude-0'
    for key in keys:
        assert not (root / key).exists() and not (root / (key + '.pending')).exists(), f'store exists: {root / key}'
    names = rb_encoders.a28_names()
    if limit is not None:
        names = names[:limit]
    spec, rows = a28_spec(names)
    enc, identity = ct2_extract.load_encoder(f'{keys[0]}_img', device)
    return [write_store(enc, identity, key, names, spec, rows, root, smoke=limit is not None) for key in keys]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--keys', nargs='+', choices=KEYS, default=list(KEYS))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--limit', type=int, metavar='N')
    parser.add_argument('--dest', type=Path, help='isolated smoke root (under /tmp/claude-0); stores go to DEST/<key>')
    args = parser.parse_args(argv)
    assert len(set(args.keys)) == len(args.keys)
    return extract(args.keys, args.device, args.limit, args.dest)


if __name__ == '__main__':
    main()
