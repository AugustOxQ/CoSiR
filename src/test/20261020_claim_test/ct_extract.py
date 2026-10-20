"""Held instructed Qwen images for the target diagnostic (not model inputs).

The controller supplies V1 and runs GPU work under flock -o -w on
/tmp/gpu0.lock. Full extraction requires CUDA; --check N may use CPU.
Check mode only re-encodes development paintings and writes nothing.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
for folder in ('20261017_image_transfer_pilot', '20261018_aspect_graph_sheet'):
    sys.path.insert(0, str(HERE.parent / folder))

import ct_held
from ags_trainimg import KEYS, OUTPUT, PROMPTS, validate_train_images
from itp_extract import encode

HELD_LIST = Path('/project/CoSiR/src/test/20261020_claim_test/outputs/held/held_paintings.json')
SAMPLE = Path('/project/CoSiR/src/test/20261017_prompted_aspect_views/sample.npz')


def image_spec(names):
    """Use the train-image adapter and join painting names, never store row IDs."""
    from scripts.extract_features import adapter_artelingo_full
    assert names and len(names) == len(set(names))
    records = ct_held._annotations()
    first = {}
    for sid, record in enumerate(records):
        first.setdefault(record['painting'], sid)
    ids = np.asarray([first[name] for name in names], dtype=np.int64)
    spec = adapter_artelingo_full(sample_ids=ids, expected=None)
    lookup = {record['painting']: row for row, record in enumerate(spec.images)}
    assert len(lookup) == len(spec.images) == len(names)
    rows = np.asarray([lookup[name] for name in names], dtype=np.int64)
    assert [spec.images[int(row)]['painting'] for row in rows] == list(names)
    return spec, rows


def encode_rows(enc, spec, rows, batch=32):
    """Reuse train-image RGB loading, preprocessing, encoding and float16 cast."""
    assert batch > 0 and len(rows) > 0
    for start in range(0, len(rows), batch):
        value = np.asarray(encode(enc, spec, rows[start:start + batch]), dtype=np.float16)
        assert value.shape == (min(batch, len(rows) - start), 2048)
        assert np.isfinite(value).all()
        assert np.all(np.abs(np.linalg.norm(value.astype(np.float32), axis=1) - 1) < 1e-3)
        yield start, value


def check_train_images(enc, key, n, batch=32):
    """Require every selected development row to match its train-image store."""
    assert key in KEYS and 0 < n <= 14889
    with np.load(SAMPLE, allow_pickle=False) as z:
        sample = {name: z[name] for name in z.files}
    reference, ids = validate_train_images(sample, key)
    names = ids['painting_names'][:n].tolist()
    spec, rows = image_spec(names)
    cosines = []
    for start, value in encode_rows(enc, spec, rows, batch):
        x = value.astype(np.float32)
        ref = np.asarray(reference[start:start + len(x)], dtype=np.float32)
        cosine = np.sum(x * ref, axis=1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(ref, axis=1))
        cosines.extend(cosine.tolist())
    minimum = float(min(cosines))
    print(f'{key}: check n={n} min_cosine={minimum:.8f}', flush=True)
    assert np.all(np.asarray(cosines) > .999), f'{key}: per-row cosine must exceed 0.999'
    return dict(n=n, min_cosine=minimum)


def model_identity(enc):
    """Record the resolved Hugging Face commit, rather than a moving revision."""
    from src.data.feature_extract import QWEN_REPO
    revision = enc.m.config._commit_hash
    assert isinstance(revision, str) and len(revision) == 40
    return dict(model_id=QWEN_REPO, model_revision=revision)


def verify_store(path, key, names, held_sha256):
    """Validate a complete store without changing it or constructing an encoder."""
    path = Path(path)
    meta = json.loads((path / 'meta.json').read_text())
    assert meta['instruction'] == PROMPTS[key]
    assert meta['held_sha256'] == held_sha256 == ct_held.names_sha256(names)
    assert meta['dtype'] == 'float16' and meta['shape'] == [len(names), 2048]
    from src.data.feature_extract import QWEN_REPO
    assert meta['model_id'] == QWEN_REPO and len(meta['model_revision']) == 40
    with np.load(path / 'ids.npz', allow_pickle=False) as z:
        assert z.files == ['painting_names']
        assert z['painting_names'].tolist() == names
    x = np.load(path / 'img.npy', mmap_mode='r')
    assert x.shape == (len(names), 2048) and x.dtype == np.float16
    assert np.isfinite(x).all()
    assert np.all(np.abs(np.linalg.norm(x.astype(np.float32), axis=1) - 1) < 1e-3)
    assert meta['img_sha256'] == hashlib.sha256(x.tobytes()).hexdigest()
    return meta


def write_store(enc, key, names, held_sha256, spec, rows, batch=32, root=OUTPUT):
    """Seal one new directory; existing stores and pending work are never replaced."""
    assert key in KEYS and enc.instruction == PROMPTS[key]
    assert held_sha256 == ct_held.names_sha256(names)
    assert [spec.images[int(row)]['painting'] for row in rows] == names
    path = Path(root) / f'{key}_held'
    assert not path.exists(), f'existing store: {path}'
    pending = path.with_name(path.name + '.pending')
    pending.mkdir(parents=True, exist_ok=False)
    x = np.lib.format.open_memmap(pending / 'img.npy', mode='w+', dtype=np.float16,
                                  shape=(len(names), 2048))
    for start, value in encode_rows(enc, spec, rows, batch):
        x[start:start + len(value)] = value
    x.flush()
    meta = dict(instruction=PROMPTS[key], **model_identity(enc), held_sha256=held_sha256,
                dtype=str(x.dtype), shape=list(x.shape), img_sha256=hashlib.sha256(x.tobytes()).hexdigest())
    del x
    np.savez(pending / 'ids.npz', painting_names=np.asarray(names))
    (pending / 'meta.json').write_text(json.dumps(meta, indent=2, allow_nan=False) + '\n')
    verify_store(pending, key, names, held_sha256)
    assert not path.exists(), f'existing store: {path}'
    pending.rename(path)
    print(f'{key}: sealed {len(names)} images at {path}', flush=True)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--keys', nargs='+', choices=KEYS, default=list(KEYS))
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--check', type=int, metavar='N', help='compare N development rows; no writes')
    args = parser.parse_args(argv)
    assert len(args.keys) == len(set(args.keys)) and args.batch > 0
    assert args.check is None or 0 < args.check <= 14889
    assert args.check is not None or (args.device == 'cuda' or args.device.startswith('cuda:')), \
        'full extraction requires GPU'
    names = sha256 = spec = rows = None
    todo = args.keys
    paths = []
    if args.check is None:
        names, sha256 = ct_held.read_held_list(HELD_LIST)
        todo = []
        for key in args.keys:
            path = OUTPUT / f'{key}_held'
            if path.exists():
                verify_store(path, key, names, sha256)
                print(f'{key}: verified existing store {path}', flush=True)
                paths.append(path)
            else:
                assert not path.with_name(path.name + '.pending').exists(), 'pending extraction exists'
                todo.append(key)
        if not todo:
            return paths
        spec, rows = image_spec(names)
    from src.data.feature_extract import Qwen3VLEmb
    import torch
    torch.set_num_threads(8)
    enc = Qwen3VLEmb(device=args.device, instruction=PROMPTS[todo[0]])
    for key in todo:
        enc.instruction = PROMPTS[key]
        if args.check is not None:
            check_train_images(enc, key, args.check, args.batch)
        else:
            paths.append(write_store(enc, key, names, sha256, spec, rows, args.batch))
    return paths


if __name__ == '__main__':
    main()
