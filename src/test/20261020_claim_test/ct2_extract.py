"""Missing train+validation features, sealed in new shared-cache stores.

--dest is an isolated root containing the same store basenames as the cache.
--limit requires --dest; smoke stores retain smoke=True and cannot feed trainers.
Failed work remains in <store>.pending for inspection; no resume or overwrite.
"""
import argparse
import hashlib
import json
import sys
import time
from functools import partial
from pathlib import Path
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent / '20261018_percept_core'))
import ct2_data as data
from ct_extract import encode_rows, image_spec, model_identity
from itp_data import PROMPTS

STORES = ('emotion_focus_img', 'style_word_img', 'genre_word_img', 'emotion_focus_txt', 'roberta')
QWEN_REVISION = '9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda'


def store_path(store, dest=None):
    """New cache directory, or its basename below a smoke root."""
    assert store in STORES
    if store == 'roberta':
        path = data.ROBERTA.with_name(data.ROBERTA.name + '_s2')
    else:
        key, kind = store.rsplit('_', 1)
        path = data.ASPECT / f'{key}_s2{kind}'
    return path if dest is None else Path(dest) / path.name


def old_paths(store):
    if store == 'roberta':
        return [data.ROBERTA]
    key, kind = store.rsplit('_', 1)
    paths = [data.ASPECT / key]
    if kind == 'img':
        paths.append(data.ASPECT / f'{key}_trainimg')
    return paths


def load_encoder(store, device):
    """Construct unchanged encoders and require the old model revision."""
    import torch
    torch.set_num_threads(8)
    if store == 'roberta':
        import pc_roberta as roberta
        tokenizer, model = roberta.load_encoder(device)
        identity = dict(model_id=roberta.MODEL_NAME, model_revision=roberta.REVISION,
                        pooling='attention-mask mean of last_hidden_state, including special tokens',
                        max_length=roberta.MAX_LENGTH)
        return (tokenizer, model, device), identity
    from src.data.feature_extract import Qwen3VLEmb
    from transformers import AutoModel, Qwen3VLProcessor
    # The shared constructor has no revision parameter. Bind both loaders at
    # runtime without changing stage 1's hashed source or its encoding math.
    with patch.object(AutoModel, 'from_pretrained', partial(
            AutoModel.from_pretrained, revision=QWEN_REVISION, local_files_only=True)), \
            patch.object(Qwen3VLProcessor, 'from_pretrained', partial(
                Qwen3VLProcessor.from_pretrained, revision=QWEN_REVISION, local_files_only=True)):
        enc = Qwen3VLEmb(device=device, instruction=PROMPTS[store.rsplit('_', 1)[0]])
    identity = model_identity(enc)
    assert identity['model_revision'] == QWEN_REVISION, 'Qwen revision differs from old stores'
    return enc, identity


def caption_spec(ids):
    from scripts.extract_features import adapter_artelingo_full
    spec = adapter_artelingo_full(sample_ids=ids['sample_ids'], expected=None)
    assert np.array_equal([m['sample_id'] for m in spec.text_meta], ids['sample_ids'])
    names = ids['painting_ids'] if 'painting_ids' in ids else ids['caption_painting_ids']
    assert np.array_equal([m['painting'] for m in spec.text_meta], names)
    assert all(isinstance(text, str) and text.strip() for text in spec.texts)
    return spec


def encode_batches(store, enc, ids):
    """Preserve image batch 32, Qwen text chunks 4096/batch 128, RoBERTa 256."""
    if store.endswith('_img'):
        spec, rows = image_spec(ids['painting_names'].tolist())
        yield from encode_rows(enc, spec, rows)
        return
    spec = caption_spec(ids)
    if store == 'roberta':
        import pc_roberta as roberta
        for start in range(0, len(spec.texts), roberta.BATCH_SIZE):
            yield start, roberta.encode(spec.texts[start:start + roberta.BATCH_SIZE], *enc)
    else:
        for start in range(0, len(spec.texts), 4096):
            yield start, np.asarray(enc.encode_texts(spec.texts[start:start + 4096], batch_size=128),
                                    dtype=np.float16)


def parity_gate(store, enc):
    """Re-encode 32 old-store IDs, joined explicitly to their saved rows."""
    path = old_paths(store)[0]
    if store.endswith('_img'):
        names, raw = data._image_store(data._sealed(path, store.rsplit('_', 1)[0]))
        rows = np.argsort(names)[:32]
        ids = dict(painting_names=names[rows])
    elif store == 'roberta':
        with np.load(path / 'ids.npz', allow_pickle=False) as z:
            sample_ids, names = z['sample_ids'], z['painting_ids']
        assert sample_ids.dtype == np.int64 and sample_ids.ndim == 1
        assert len(np.unique(sample_ids)) == len(sample_ids) and names.shape == sample_ids.shape
        raw = np.load(path / 'txt.npy', mmap_mode='r')
        assert raw.shape == (len(sample_ids), 768) and raw.dtype == np.float32
        rows = np.argsort(sample_ids)[:32]
        ids = dict(sample_ids=sample_ids[rows], painting_ids=names[rows])
    else:
        sample_ids, names, raw = data._caption_store(data._sealed(path, 'emotion_focus'))
        rows = np.argsort(sample_ids)[:32]
        ids = dict(sample_ids=sample_ids[rows], caption_painting_ids=names[rows])
    assert len(rows) == 32, 'parity needs 32 old rows'
    expected = np.asarray(raw[rows], dtype=np.float32)
    actual = np.concatenate([value for _, value in encode_batches(store, enc, ids)]).astype(np.float32)
    assert actual.shape == expected.shape and np.isfinite(actual).all() and np.isfinite(expected).all()
    report = dict(n=32, ids_sha256=ids_sha256(ids))
    if store == 'roberta':
        difference = float(np.max(np.abs(actual - expected)))
        report.update(max_abs_difference=difference, threshold=1e-3)
        assert difference <= 1e-3, f'RoBERTa parity failed: {report}'
    else:
        cosine = np.sum(actual * expected, axis=1) / (np.linalg.norm(actual, axis=1) *
                                                     np.linalg.norm(expected, axis=1))
        report.update(min_cosine=float(cosine.min()), mean_cosine=float(cosine.mean()), threshold=.999)
        assert np.isfinite(cosine).all() and (cosine >= .999).all(), f'cosine parity failed: {report}'
    return dict(passed=True, **report)


def ids_sha256(ids):
    """Hash the primary ordered IDs as newline-joined names or decimal IDs (Q1)."""
    values = ids['painting_names'] if 'painting_names' in ids else ids['sample_ids']
    return hashlib.sha256('\n'.join(map(str, values)).encode('utf-8')).hexdigest()


def extract(store, device='cuda', limit=None, dest=None):
    """Select missing IDs, pass old-store parity, write pending, then seal."""
    path = store_path(store, dest)
    assert not path.exists(), f'existing store: {path}'
    pending = path.with_name(path.name + '.pending')
    assert not pending.exists(), f'pending extraction exists: {pending}'
    assert limit is None or (limit > 0 and dest is not None), '--limit requires an isolated --dest'
    missing = data.missing_ids(old_paths(store))
    image = store.endswith('_img')
    id_key = 'painting_names' if image else 'sample_ids'
    wanted = missing[id_key]
    assert wanted.ndim == 1 and len(wanted) and np.array_equal(wanted, np.unique(wanted))
    if not image:
        assert wanted.dtype == np.int64
    selected = wanted if limit is None else wanted[:limit]
    ids = {id_key: selected}
    if not image:
        # Adapter metadata joins extraction IDs to painting names; no store-row join.
        from scripts.extract_features import adapter_artelingo_full
        spec = adapter_artelingo_full(sample_ids=selected, expected=None)
        assert np.array_equal([m['sample_id'] for m in spec.text_meta], selected)
        name_key = 'painting_ids' if store == 'roberta' else 'caption_painting_ids'
        ids[name_key] = np.asarray([m['painting'] for m in spec.text_meta])
    enc, identity = load_encoder(store, device)
    parity = parity_gate(store, enc)
    pending.mkdir(parents=True, exist_ok=False)
    dtype, dim = (np.float32, 768) if store == 'roberta' else (np.float16, 2048)
    x = np.lib.format.open_memmap(pending / ('img.npy' if image else 'txt.npy'), mode='w+',
                                  dtype=dtype, shape=(len(selected), dim))
    start = time.monotonic()
    written = 0
    for offset, value in encode_batches(store, enc, ids):
        assert offset == written and value.shape[1:] == (dim,) and len(value) > 0
        assert value.dtype == dtype and np.isfinite(value).all()
        assert written + len(value) <= len(selected)
        if store != 'roberta':
            assert np.all(np.abs(np.linalg.norm(value.astype(np.float32), axis=1) - 1) < 1e-3)
        x[offset:offset + len(value)] = value
        written += len(value)
        print(f'{store}: {written}/{len(selected)}', flush=True)
    assert written == len(selected)
    x.flush()
    del x
    seconds = time.monotonic() - start
    np.savez(pending / 'ids.npz', **ids)
    done = dict(store=store, instruction=None if store == 'roberta' else PROMPTS[store.rsplit('_', 1)[0]],
                **identity, count=written, n_images=written if image else 0, n_captions=0 if image else written,
                counts={role: int(np.isin(selected, missing[role][id_key]).sum()) for role in ('train', 'val')},
                ids_sha256=ids_sha256(ids), dtype=np.dtype(dtype).name, shape=[written, dim],
                batch_size=32 if image else (256 if store == 'roberta' else 128),
                seconds=seconds, items_per_second=written / seconds, parity_gate=parity, smoke=limit is not None)
    (pending / 'done.json').write_text(json.dumps(done, indent=2, allow_nan=False) + '\n')
    assert not path.exists(), f'existing store: {path}'
    pending.rename(path)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--store', required=True, choices=STORES)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--limit', type=int, metavar='N')
    parser.add_argument('--dest', type=Path, help='isolated smoke root; appends the cache store basename')
    args = parser.parse_args(argv)
    return extract(args.store, args.device, args.limit, args.dest)


if __name__ == '__main__':
    main()
