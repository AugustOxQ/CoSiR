"""Text-only r29 stores; reuse the frozen extractor's caption and encoder paths.

CLI: extract [--limit N] [--gate-iv], check --store DIR, gate-iv --store DIR.
--limit counts captions (not images), marks smoke=true, and requires --out.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

import r29_paths as paths

sys.path.insert(0, str(paths.REPO))
from scripts import extract_features as extractor


ENCODERS = (extractor.SIGLIP1, extractor.SIGLIP2, extractor.CLIP_L14)
NORM_TOL = 1e-3  # D23; stricter than extract_features.check_arrays' default.
REPLAY_COSINE = 0.999


def _json(path):
    with Path(path).open() as f:
        value = json.load(f)
    assert isinstance(value, dict), path
    return value


def _write(path, value):
    with Path(path).open('w') as f:
        json.dump(value, f)  # Same bytes/serialization convention as extract_features.


def _sha(path):
    return extractor._sha256(path)


def _keys(index):
    assert isinstance(index['texts'], list)
    assert index['n_texts'] == len(index['texts']) > 0, 'text row count'
    by_id = {}
    for row in index['texts']:
        sid, painting = row['sample_id'], row['painting']
        assert type(sid) in (int, str) and isinstance(painting, str) and painting
        assert sid not in by_id, f'duplicate sample_id: {sid}'
        by_id[sid] = painting
    return by_id


def _read(folder, *, allow_smoke=False):
    """Cheap seal checks before exposing a caption array, including legacy stores."""
    folder = Path(folder)
    index, progress = _json(folder / 'index.json'), _json(folder / 'progress.json')
    assert type(index['smoke']) is bool
    assert allow_smoke or index['smoke'] is False, 'smoke store'
    assert index['backbone'] in ENCODERS
    assert type(index['dim']) is int and index['dim'] > 0
    _keys(index)
    assert progress['txt_done'] == index['n_texts'], 'incomplete text store'
    assert progress['gate']['ok'] is True, 'failed extraction gate'
    assert progress['index_sha256'] == _sha(folder / 'index.json'), 'index hash'
    rows = np.load(folder / 'txt.npy', mmap_mode='r', allow_pickle=False)
    assert rows.dtype == np.float16 and index['dtype'] == 'float16'
    assert rows.shape == (index['n_texts'], index['dim']), 'array shape'
    return index, progress, rows


def check_store(folder, reference_index, *, allow_smoke=False):
    """Gate viii. Exact key coverage; reference order may differ (C6).

    reference_index is an index dict or an index.json path. For a smoke check,
    pass an explicitly sliced reference and allow_smoke=True.
    """
    reference = reference_index if isinstance(reference_index, dict) else _json(reference_index)
    expected = _keys(reference)
    index, progress, rows = _read(folder, allow_smoke=allow_smoke)
    actual = _keys(index)
    assert actual.keys() == expected.keys(), 'sample_id coverage'
    for sid, painting in actual.items():
        assert painting == expected[sid], f'painting mismatch for sample_id {sid}'
    count = index['n_texts_truncated']
    assert type(count) is int and 0 <= count <= index['n_texts'], 'truncation count'
    assert progress['n_truncated'] == count, 'truncation count mismatch'
    assert index['truncation_limit'] == (77 if index['backbone'] == extractor.CLIP_L14 else 64)
    gate = extractor.check_arrays(rows, None, tol=NORM_TOL)
    assert gate['ok'], f'finite/norm gate: {gate}'
    return dict(gate, n_texts=index['n_texts'], n_truncated=count,
                index_sha256=_sha(Path(folder) / 'index.json'))


def caption_spec(reference, annotations=extractor.ARTELINGO_ANN):
    """Resolve raw captions by reference IDs, never by annotation position alone."""
    _keys(reference)
    ids = np.asarray([r['sample_id'] for r in reference['texts']], dtype=np.int64)
    spec = extractor.adapter_artelingo_full(annotations_path=annotations, sample_ids=ids, expected=None)
    assert spec.text_meta == reference['texts'], 'annotation ID/painting/img alignment'
    assert spec.images == reference['images'][:len(spec.images)], 'annotation image alignment'
    assert all(isinstance(t, str) for t in spec.texts)
    return spec


def stored_prefix(encoder, prefix):
    assert encoder in ENCODERS and isinstance(prefix, str)
    return prefix.lower() if encoder in (extractor.SIGLIP1, extractor.SIGLIP2) else prefix


def _encode_chunk(enc, captions, prefix, batch_size):
    # Exactly one concatenation, BEFORE the extractor's own lowercasing.
    texts = [prefix + caption for caption in captions]
    # Includes special tokens, no truncation while counting, strict > limit (D14).
    count = extractor.count_truncated(enc, texts)
    start = time.time()
    features = enc.encode_texts(texts, batch_size)
    assert features.shape == (len(texts), enc.dim), 'encoder shape'
    return features.astype(np.float16), count, time.time() - start


def extract_store(reference, texts, enc, out, *, prefix=paths.PREFIX, smoke=False, batch_size=None):
    """Write a new text store. `texts` must be raw captions, never pre-prefixed.

    Keep image metadata and text `img` links unchanged; no image array is written.
    The encoder owns normalisation, token cutting, pooling and L2 normalisation.
    """
    _keys(reference)
    assert len(texts) == reference['n_texts'] and all(isinstance(t, str) for t in texts)
    assert enc.name in ENCODERS
    stored_prefix(enc.name, prefix)
    assert type(smoke) is bool
    batch_size = batch_size or extractor.BATCH[enc.name]['txt']
    assert batch_size > 0
    out = Path(out)
    assert not out.exists() or (out.is_dir() and not any(out.iterdir())), f'non-empty output: {out}'
    out.mkdir(parents=True, exist_ok=True)
    n = len(texts)
    rows = np.lib.format.open_memmap(out / 'txt.npy', mode='w+', dtype=np.float16, shape=(n, enc.dim))
    progress = dict(img_done=0, txt_done=0, n_truncated=0, img_seconds=0.0, txt_seconds=0.0,
                    n_img=reference['n_images'], n_txt=n)
    _write(out / 'progress.json', progress)
    for a in range(0, n, extractor.TXT_CHUNK):
        b = min(a + extractor.TXT_CHUNK, n)
        features, count, seconds = _encode_chunk(enc, texts[a:b], prefix, batch_size)
        progress['n_truncated'] += count
        rows[a:b] = features
        progress['txt_seconds'] += seconds
        progress['txt_done'] = b
        rows.flush()
        _write(out / 'progress.json', progress)
        print(f'[{enc.name}] texts {b}/{n}', flush=True)
    progress['gate'] = extractor.check_arrays(rows, None, tol=NORM_TOL)
    _write(out / 'progress.json', progress)
    assert progress['gate']['ok'], f"finite/norm gate: {progress['gate']}"
    index = dict(dataset=reference['dataset'], backbone=enc.name, smoke=smoke, dtype='float16', dim=enc.dim,
                 n_images=reference['n_images'], n_texts=n, n_texts_truncated=progress['n_truncated'],
                 truncation_limit=77 if enc.name == extractor.CLIP_L14 else 64,
                 extra=reference.get('extra', {}), image_only=False, text_only=True,
                 images=reference['images'], texts=reference['texts'], prefix=stored_prefix(enc.name, prefix))
    _write(out / 'index.json', index)
    progress['index_sha256'] = _sha(out / 'index.json')
    _write(out / 'progress.json', progress)
    return check_store(out, reference, allow_smoke=smoke)


def _first(index, n):
    assert 0 < n <= index['n_texts']
    return dict(index, texts=index['texts'][:n], n_texts=n)


def gate_iv(folder, plain, texts, enc, *, record=None, allow_smoke=False):
    """Validate prefix store and replay 100 raw captions with an empty prefix.

    details.hash is reserved for tickets 03/04's trainer/scorer hash comparison.
    A pass here covers this ticket's store/replay portion only.
    """
    folder, plain = Path(folder), Path(plain)
    record = Path(record) if record is not None else paths.GATES / 'iv.json'
    assert not allow_smoke or record.resolve().is_relative_to(folder.resolve()), 'smoke records stay with smoke'
    tries = _json(record)['tries'] + 1 if record.exists() else 1
    assert 1 <= tries <= 3, 'gate iv exhausted its three tries'
    details = {'hash': None, 'scope': 'store and empty-prefix replay; hash comparison pending tickets 03/04'}
    status = 'fail'
    try:
        reference, _, plain_rows = _read(plain)
        index = _json(folder / 'index.json')
        assert enc.name == index['backbone'] == reference['backbone'] == extractor.SIGLIP1
        assert index['prefix'] == stored_prefix(enc.name, paths.PREFIX), 'stored prefix'
        selected = _first(reference, index['n_texts']) if allow_smoke else reference
        assert index['texts'] == selected['texts'], 'extraction order'
        details['store'] = check_store(folder, selected, allow_smoke=allow_smoke)
        assert details['store']['n_truncated'] == 0, 'gate iv requires zero truncated captions'
        assert len(texts) >= 100 and len(plain_rows) >= 100, 'gate iv needs 100 captions'
        # Same text path, count rule and float16 storage rounding, with the empty prefix.
        replay, count, _ = _encode_chunk(enc, texts[:100], '', extractor.BATCH[enc.name]['txt'])
        assert count == 0
        assert replay.shape == (100, index['dim'])
        assert extractor.check_arrays(replay, None, tol=NORM_TOL)['ok']
        x, y = replay.astype(np.float32), np.asarray(plain_rows[:100], dtype=np.float32)
        cosine = np.sum(x * y, axis=1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1))
        details['replay'] = dict(n_texts=100, min_cosine=float(cosine.min()), threshold=REPLAY_COSINE)
        assert np.isfinite(cosine).all() and (cosine >= REPLAY_COSINE).all(), 'empty-prefix replay'
        status = 'pass'
    except Exception as error:
        details['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        record.parent.mkdir(parents=True, exist_ok=True)
        written = subprocess.check_output(['date', '+%F %H:%M'], env=dict(os.environ, TZ='Europe/Amsterdam'),
                                          text=True).strip()
        _write(record, dict(gate='iv', status=status, tries=tries, details=details, written=written))
    return _json(record)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('extract', 'check', 'gate-iv'))
    parser.add_argument('--encoder', choices=ENCODERS, default=extractor.SIGLIP1)
    parser.add_argument('--reference', type=Path, default=paths.CAPTION_SOURCES['siglip1'][0])
    parser.add_argument('--annotations', type=Path, default=extractor.ARTELINGO_ANN)
    parser.add_argument('--prefix', default=paths.PREFIX)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--store', type=Path)
    parser.add_argument('--limit', type=int, help='first N captions; smoke only, explicit --out required')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--gate-iv', action='store_true', help='also run gate iv after extraction')
    args = parser.parse_args(argv)
    reference, _, _ = _read(args.reference)
    if args.command == 'check':
        assert args.store is not None and args.limit is None
        print(json.dumps(check_store(args.store, reference)))
        return
    if args.limit is not None:
        assert args.command == 'extract' and args.out is not None, 'smoke requires extract --out'
        reference = _first(reference, args.limit)
    if args.command == 'extract':
        source = {extractor.SIGLIP1: 'siglip1', extractor.SIGLIP2: 'siglip2', extractor.CLIP_L14: 'clipl14'}[args.encoder]
        assert args.out is not None or args.prefix == paths.PREFIX, 'custom prefix needs explicit --out'
        out = args.out or paths.CAPTION_SOURCES[source + '_prefix'][0]
        assert not out.exists() or (out.is_dir() and not any(out.iterdir())), f'non-empty output: {out}'
    else:
        assert args.store is not None and args.limit is None
        out = args.store
    spec = caption_spec(reference, args.annotations)
    from src.data.feature_extract import load_encoder
    enc = load_encoder(args.encoder, args.device)
    if args.command == 'extract':
        result = extract_store(reference, spec.texts, enc, out, prefix=args.prefix, smoke=args.limit is not None)
        print(json.dumps(result))
    if args.command == 'gate-iv' or args.gate_iv:
        record = out / 'records/gates/iv.json' if args.limit is not None else None
        print(json.dumps(gate_iv(out, args.reference, spec.texts, enc, record=record,
                                 allow_smoke=args.limit is not None)))


if __name__ == '__main__':
    main()
