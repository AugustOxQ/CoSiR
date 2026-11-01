"""Extraction of r13's two new views and gate (i) parity (spec 2026-10-31-photo-finetune-design.md §3.5 0a(i), D8).

Views (artwork wording = c*'s own instruction, `PROMPTS['emotion_focus']`; both written through `<view>.pending`, then
renamed; an existing or pending target is refused):
  images  TRAIN_IMG_VIEW  img.npy float16 [n_train, 2048] unit + keys.json + meta.json; every train key, batch 32.
  texts   TXT_VIEW        txt.npy float16 [n_rows, 2048] unit + rows.json + meta.json; every explanation row of the
                          train photos then of the val photos (`pf_data.text_rows`), text = the CSV `utterance` at
                          `csv_row` (only that column is read), chunks of 4096, batch 128.
  parity  gate (i): OUTPUTS/stage0a/parity_images.json, parity_texts.json (r12's report form, every cosine >= 0.999).
  check   OUTPUTS/stage0a/views.json: both views re-read from disk (rows, keys, finite, unit norm, hashes, gate).

Production identity: the views are full views with `smoke: false`. PF_SMOKE=N (with PF_OUTPUTS set) makes the smoke
views under OUTPUTS/smoke_views (first N sorted train keys, all val texts); then `--smoke` is required, and without
PF_SMOKE it is refused. Run from anywhere, env /root/miniconda3/envs/CoSiR, PYTHONDONTWRITEBYTECODE=1, GPU under
`flock -n -o -E 75 /tmp/gpu0.lock`:
  python pf_extract.py parity [images|texts|both]
  python pf_extract.py images [--smoke]
  python pf_extract.py texts [--smoke]
  python pf_extract.py check
Reuses r12's ac_extract (spec_for, load_encoder, encode_rows, check_target, parity_report) and ct2_extract's pinned
text encoder; ac_extract.seal and write_parity are copied here (changed: view name, widths and the output folder),
because they are bound to r12's view names and to r12's outputs folder.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import pf_paths   # first: puts r12 and every older round on sys.path
import ac_data
import ac_extract
import pf_data
import rs_vh

WIDTH = 2048
IMG_BATCH = 32
TXT_CHUNK = 4096
TXT_BATCH = 128
COSINE_MIN = pf_paths.TOL['parity_cos']
UNIT_TOL = pf_paths.TOL['unit_norm']
N_PARITY = 200


# ---------------------------------------------------------------- selection
def image_keys():
    """Every train key (smoke: the first PF_SMOKE sorted), joined to images by key."""
    keys = pf_data.keys('train')
    ac_extract.check_keys(keys)
    assert keys == sorted(keys)
    return keys


def text_selection():
    """(csv_row list[int], key list[str]) of the train photos' rows then the val photos' rows, in `text_rows` order."""
    csv_rows, keys = [], []
    for split in ('train', 'val'):
        rows = pf_data.text_rows(split)
        csv_rows += rows['csv_row'].tolist()
        keys += rows['key']
    assert len(set(csv_rows)) == len(csv_rows) == len(keys), 'csv_row must be unique'
    if pf_paths.SMOKE == 0:
        assert len(csv_rows) == sum(pf_paths.N_TEXTS.values()), 'row count'
    return csv_rows, keys


def utterances(csv_rows):
    """The CSV `utterance` of each `csv_row` (position in the preprocessed CSV, as the adapter counts it); only that
    column is read. Returns list[str] in `csv_rows` order."""
    import pandas as pd
    frame = pd.read_csv(Path(pf_paths.ANNOTATIONS), usecols=['utterance'], dtype=str)
    return lookup_texts(frame['utterance'].tolist(), csv_rows)


def lookup_texts(column, csv_rows):
    """`column[csv_row]` for each row; every text must be a non-blank string (the index holds only such rows)."""
    csv_rows = [int(c) for c in csv_rows]
    assert csv_rows and min(csv_rows) >= 0 and max(csv_rows) < len(column), 'csv_row out of range'
    texts = [column[c] for c in csv_rows]
    assert all(isinstance(t, str) and t.strip() for t in texts), 'blank or missing utterance at a csv_row'
    return texts


# ---------------------------------------------------------------- sealing
def check_smoke_mode(smoke):
    assert smoke == (pf_paths.SMOKE > 0), \
        f'--smoke must be given exactly when PF_SMOKE is set (flag {smoke}, PF_SMOKE {pf_paths.SMOKE})'
    return smoke


def pending_of(out):
    return ac_extract.check_target(out)


def _meta(out, x, identity, batch, seconds, smoke, **hashes):
    """Gate the array (finite, unit within 1e-3) and build the meta dict (r12's fields)."""
    from extract_features import check_arrays
    gate = check_arrays(x, None, tol=UNIT_TOL)
    assert gate['ok'], f'extraction gate failed: {gate}'
    assert x.dtype == np.float16 and x.ndim == 2 and x.shape[1] == WIDTH
    return dict(view=Path(out).name, instruction=ac_extract.instruction_for('emotion_focus'), **identity,
                dtype='float16', shape=list(x.shape), **hashes, gate=gate, smoke=bool(smoke), batch_size=batch,
                seconds=seconds)


def _publish(pending, out, meta):
    pending, out = Path(pending), Path(out)
    (pending / 'meta.json').write_text(json.dumps(meta, indent=2, allow_nan=False) + '\n')
    assert not out.exists(), f'existing view: {out}'
    pending.rename(out)


def seal_images(pending, out, keys, x, identity, smoke, seconds):
    """Adapted from ac_extract.seal (changed: view name from the folder, width 2048 fixed)."""
    assert x.shape[0] == len(keys)
    (Path(pending) / 'keys.json').write_text(json.dumps(keys))
    meta = _meta(out, x, identity, IMG_BATCH, seconds, smoke, keys_sha256=rs_vh.names_sha(keys),
                 img_sha256=rs_vh.array_sha(x))
    _publish(pending, out, meta)
    return meta


def rows_sha(csv_rows, keys):
    """The index's `rows_sha256`: names_sha over 'csv_row<TAB>key' in rows.json's order."""
    return rs_vh.names_sha([f'{int(c)}\t{k}' for c, k in zip(csv_rows, keys)])


def seal_texts(pending, out, csv_rows, keys, x, identity, smoke, seconds):
    assert x.shape[0] == len(keys) == len(csv_rows)
    (Path(pending) / 'rows.json').write_text(json.dumps({'csv_row': [int(c) for c in csv_rows], 'key': list(keys)}))
    meta = _meta(out, x, identity, TXT_BATCH, seconds, smoke, txt_sha256=rs_vh.array_sha(x),
                 rows_sha256=rows_sha(csv_rows, keys))
    _publish(pending, out, meta)
    return meta


# ---------------------------------------------------------------- extraction
def extract_images(device='cuda', smoke=False):
    check_smoke_mode(smoke)
    keys = image_keys()
    out = Path(pf_paths.TRAIN_IMG_VIEW)
    pending = pending_of(out)
    spec, rows = ac_extract.spec_for(keys)
    enc, identity = ac_extract.load_encoder('emotion_focus', device)
    pending.mkdir(parents=True)
    x = np.lib.format.open_memmap(pending / 'img.npy', mode='w+', dtype=np.float16, shape=(len(keys), WIDTH))
    t0, done = time.monotonic(), 0
    for start, value in ac_extract.encode_rows(enc, spec, rows, IMG_BATCH, WIDTH):
        assert start == done
        x[start:start + len(value)] = value
        done += len(value)
        if (start // IMG_BATCH) % 50 == 0 or done == len(keys):
            print(f'images: {done}/{len(keys)} {done / (time.monotonic() - t0):.1f}/s', flush=True)
    x.flush()
    meta = seal_images(pending, out, keys, x, identity, smoke, time.monotonic() - t0)
    print(f'images: sealed {len(keys)} at {out}', flush=True)
    return meta


def extract_texts(device='cuda', smoke=False):
    check_smoke_mode(smoke)
    csv_rows, keys = text_selection()
    out = Path(pf_paths.TXT_VIEW)
    pending = pending_of(out)
    texts = utterances(csv_rows)
    import ct2_extract
    enc, identity = ct2_extract.load_encoder('emotion_focus_txt', device)
    assert enc.instruction == ac_extract.instruction_for('emotion_focus')
    pending.mkdir(parents=True)
    x = np.lib.format.open_memmap(pending / 'txt.npy', mode='w+', dtype=np.float16, shape=(len(texts), WIDTH))
    t0, done = time.monotonic(), 0
    for start in range(0, len(texts), TXT_CHUNK):
        value = np.asarray(enc.encode_texts(texts[start:start + TXT_CHUNK], batch_size=TXT_BATCH), dtype=np.float16)
        assert value.shape == (len(texts[start:start + TXT_CHUNK]), WIDTH), value.shape
        x[start:start + len(value)] = value
        done += len(value)
        print(f'texts: {done}/{len(texts)} {done / (time.monotonic() - t0):.1f}/s', flush=True)
    x.flush()
    meta = seal_texts(pending, out, csv_rows, keys, x, identity, smoke, time.monotonic() - t0)
    print(f'texts: sealed {len(texts)} at {out}', flush=True)
    return meta


# ---------------------------------------------------------------- parity (gate (i))
def write_parity(name, report, **info):
    """Adapted from ac_extract.write_parity (changed: writes under r13's OUTPUTS, not r12's)."""
    path = Path(pf_paths.OUTPUTS) / 'stage0a' / f'{name}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(name=name, **info, **report), indent=2, allow_nan=False) + '\n')
    print(f'{name}: min_cosine={report["min_cosine"]:.6f} max_abs_diff={report["max_abs_diff"]:.3e} '
          f'ok={report["ok"]} -> {path}', flush=True)
    return path


def parity_image_keys(n_each=100):
    """The first 100 sorted val keys plus the first 100 sorted FI val keys (no repeats)."""
    val = ac_data.keys('val')
    first = val[:n_each]
    fi = [k for k in val if ac_extract.split_key(k)[0] == pf_paths.ac_paths.FI_SOURCE][:n_each]
    assert len(fi) == n_each and not set(first) & set(fi), 'parity keys: FI keys must follow the first 100'
    return first + fi


def parity_images(device='cuda'):
    """New instructed-image path vs r12's `emotion_focus` rows (val/test view), by key."""
    keys = parity_image_keys()
    expected = ac_data._view('teacher_artwork', keys)
    spec, rows = ac_extract.spec_for(keys)
    enc, identity = ac_extract.load_encoder('emotion_focus', device)
    actual = np.concatenate([v for _, v in ac_extract.encode_rows(enc, spec, rows, IMG_BATCH, WIDTH)])
    report = ac_extract.parity_report(actual, expected)
    write_parity('parity_images', report, view='emotion_focus', store=str(Path(pf_paths.VAL_TEST_TEACHER_VIEW)),
                 keys=keys, keys_sha256=rs_vh.names_sha(keys), **identity, batch_size=IMG_BATCH)
    return report


def parity_texts(device='cuda'):
    """New text path vs ArtELingo's instructed caption store: its first 200 rows in the store's own order (ids.npz)."""
    import ct2_extract
    store = Path(pf_paths.ART_TXT_STORE)
    with np.load(store / 'ids.npz', allow_pickle=False) as z:
        sample_ids = z['sample_ids'][:N_PARITY]
    expected = np.asarray(np.load(store / 'txt.npy', mmap_mode='r')[:N_PARITY])
    from scripts.extract_features import adapter_artelingo_full
    spec = adapter_artelingo_full(sample_ids=sample_ids, expected=None)
    assert np.array_equal([m['sample_id'] for m in spec.text_meta], sample_ids)
    texts = spec.texts
    enc, identity = ct2_extract.load_encoder('emotion_focus_txt', device)
    assert enc.instruction == ac_extract.instruction_for('emotion_focus')
    actual = np.concatenate([np.asarray(enc.encode_texts(texts[a:a + TXT_CHUNK], batch_size=TXT_BATCH),
                                        dtype=np.float16) for a in range(0, len(texts), TXT_CHUNK)])
    report = ac_extract.parity_report(actual, expected)
    write_parity('parity_texts', report, view='emotion_focus_txt', store=str(store),
                 sample_ids_sha256=rs_vh.names_sha([str(s) for s in sample_ids.tolist()]), **identity,
                 batch_size=TXT_BATCH)
    return report


# ---------------------------------------------------------------- check
def _norms(x, chunk=65536):
    bad, dev, nonfinite = 0, 0.0, 0
    for a in range(0, len(x), chunk):
        v = np.asarray(x[a:a + chunk], dtype=np.float32)
        fin = np.isfinite(v).all(1)
        nonfinite += int((~fin).sum())
        d = np.abs(np.linalg.norm(np.where(np.isfinite(v), v, 0), axis=1) - 1)
        bad += int((d > UNIT_TOL).sum())
        dev = max(dev, float(d[fin].max()) if fin.any() else 0.0)
    return dict(n_nonfinite=nonfinite, n_bad_norm=bad, max_norm_dev=dev)


def check_images():
    stored, x = pf_data._train_view()          # meta, key list vs the train keys, hashes, shape, unit rows
    assert stored == pf_data.keys('train'), 'train view keys differ from the index train keys, in order'
    side = json.loads((Path(pf_paths.TRAIN_IMG_VIEW) / 'meta.json').read_text())
    norms = _norms(x)
    assert norms['n_nonfinite'] == 0 and norms['n_bad_norm'] == 0 and side['gate']['ok'] is True
    return dict(view=Path(pf_paths.TRAIN_IMG_VIEW).name, rows=len(stored), shape=list(x.shape), **norms,
                gate=side['gate'], smoke=side['smoke'], keys_sha256=side['keys_sha256'],
                img_sha256=side['img_sha256'])


def check_texts():
    path = Path(pf_paths.TXT_VIEW)
    side = json.loads((path / 'meta.json').read_text())
    rows = json.loads((path / 'rows.json').read_text())
    csv_rows, keys = text_selection()
    assert rows['csv_row'] == csv_rows and rows['key'] == keys, 'rows.json differs from pf_data.text_rows'
    pf_data._check_meta(side, ('view', 'instruction', 'model_id', 'model_revision', 'dtype', 'shape', 'txt_sha256',
                               'rows_sha256', 'gate', 'smoke'), (len(csv_rows), WIDTH), path)
    x = np.load(path / 'txt.npy', mmap_mode='r', allow_pickle=False)
    assert x.dtype == np.float16 and x.shape == (len(csv_rows), WIDTH)
    assert side['txt_sha256'] == rs_vh.array_sha(x), 'text array hash'
    assert side['rows_sha256'] == rows_sha(csv_rows, keys), 'rows hash'
    norms = _norms(x)
    assert norms['n_nonfinite'] == 0 and norms['n_bad_norm'] == 0
    return dict(view=path.name, rows=len(csv_rows), shape=list(x.shape), **norms, gate=side['gate'],
                smoke=side['smoke'], txt_sha256=side['txt_sha256'], rows_sha256=side['rows_sha256'])


def check():
    report = dict(images=check_images(), texts=check_texts(), ok=True)
    path = Path(pf_paths.OUTPUTS) / 'stage0a' / 'views.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f'check ok -> {path}', flush=True)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('images', 'texts'):
        p = sub.add_parser(name)
        p.add_argument('--smoke', action='store_true', help='required exactly when PF_SMOKE is set')
        p.add_argument('--device', default='cuda')
    p = sub.add_parser('parity')
    p.add_argument('which', nargs='?', default='both', choices=['images', 'texts', 'both'])
    p.add_argument('--device', default='cuda')
    sub.add_parser('check')
    args = ap.parse_args(argv)
    if args.cmd == 'images':
        extract_images(args.device, args.smoke)
    elif args.cmd == 'texts':
        extract_texts(args.device, args.smoke)
    elif args.cmd == 'check':
        check()
    else:
        ok = True
        if args.which in ('images', 'both'):
            ok &= parity_images(args.device)['ok']
        if args.which in ('texts', 'both'):
            ok &= parity_texts(args.device)['ok']
        return 0 if ok else 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
