"""FI in the Affection Qwen default store (r12 ticket 01; spec §3.3, §3.5 stage 0a(i), D11).

Commands (repo root; env /root/miniconda3/envs/CoSiR, PYTHONDONTWRITEBYTECODE=1):
  1. record the old store (once, before FI is added; commit the file afterwards):
       python src/test/20261030_affection_confirmation/ac_store.py record
  2. FI-only extraction (GPU, under flock; smoke first: --limit writes under _smoke/affection_fi/):
       python scripts/extract_features.py --dataset affection_fi --backbone qwen3vl_emb_2b --limit 64
       python scripts/extract_features.py --dataset affection_fi --backbone qwen3vl_emb_2b
     (full output: /data/SSD2/pre_extract/affection_local/qwen3vl_emb_2b_fi, 21,808 images, 134,418 texts)
  3. merge (no --swap leaves the store alone; the gate and OUTPUTS/stage0a/store.json are written either way):
       python src/test/20261030_affection_confirmation/ac_store.py merge \\
           --fi /data/SSD2/pre_extract/affection_local/qwen3vl_emb_2b_fi \\
           --out /data/SSD2/pre_extract/affection_local/qwen3vl_emb_2b_merged [--swap]
     --swap (only when every check passes) renames the old folder to qwen3vl_emb_2b_pre_fi and the merged one to
     qwen3vl_emb_2b. Nothing is deleted.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))
import ac_paths as P  # noqa: E402
from extract_features import check_arrays  # noqa: E402

CHUNK = 16384
TOL = 1e-3


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def sha_rows(arr, n=None):
    """sha256 of the raw bytes of the first n rows (C order); independent of the .npy header."""
    n = len(arr) if n is None else n
    h = hashlib.sha256()
    for i in range(0, n, CHUNK):
        h.update(np.ascontiguousarray(arr[i:min(i + CHUNK, n)]).tobytes())
    return h.hexdigest()


def sha_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()


def summarize(store):
    store = Path(store)
    idx = json.load(open(store / 'index.json'))
    img = np.load(store / 'img.npy', mmap_mode='r')
    txt = np.load(store / 'txt.npy', mmap_mode='r')
    assert img.shape[0] == idx['n_images'] == len(idx['images']) and txt.shape[0] == idx['n_texts'] == len(idx['texts'])
    return {'img_file_sha256': sha_file(store / 'img.npy'), 'txt_file_sha256': sha_file(store / 'txt.npy'),
            'index_file_sha256': sha_file(store / 'index.json'),
            'img_rows_sha256': sha_rows(img), 'txt_rows_sha256': sha_rows(txt),
            'images_sha256': sha_json(idx['images']), 'texts_sha256': sha_json(idx['texts']),
            'img_shape': list(img.shape), 'txt_shape': list(txt.shape), 'dtype': str(img.dtype),
            'n_images': idx['n_images'], 'n_texts': idx['n_texts']}


def record(store=P.STORE, out_file=P.STORE_HASHES_FILE):
    """Write the old store's hashes (file hashes, row-data hashes, canonical-JSON hashes of the lists)."""
    out_file = Path(out_file)
    if out_file.exists():
        raise SystemExit(f'{out_file} exists; record refuses to overwrite')
    rec = summarize(store)
    rec['store'] = str(store)
    out_file.write_text(json.dumps(rec, indent=1) + '\n')
    return rec


def merge(old, fi, out, hashes_file=P.STORE_HASHES_FILE, stage_file=None, swap=False,
          expected_fi=(P.N_FI['images'], P.N_FI['texts'])):
    old, fi, out = Path(old), Path(fi), Path(out)
    stage_file = Path(stage_file or P.OUTPUTS / 'stage0a/store.json')
    rec = json.load(open(hashes_file))
    items = {}

    def item(name, value, reference, ok):
        items[name] = {'value': value, 'reference': reference, 'ok': bool(ok)}

    oi, fx = json.load(open(old / 'index.json')), json.load(open(fi / 'index.json'))
    o_img, o_txt = np.load(old / 'img.npy', mmap_mode='r'), np.load(old / 'txt.npy', mmap_mode='r')
    f_img, f_txt = np.load(fi / 'img.npy', mmap_mode='r'), np.load(fi / 'txt.npy', mmap_mode='r')
    n_oi, n_ot, n_fi, n_ft = len(o_img), len(o_txt), len(f_img), len(f_txt)
    assert (n_oi, n_ot) == (oi['n_images'], oi['n_texts']) and (n_fi, n_ft) == (fx['n_images'], fx['n_texts'])
    assert fx.get('dataset') == 'affection_fi' and not fx['smoke'] and not fx['image_only'], 'not a full FI-only store'
    assert fx['backbone'] == oi['backbone'] and fx['dim'] == oi['dim'] and fx['dtype'] == oi['dtype'] == 'float16'
    assert o_img.dtype == f_img.dtype == o_txt.dtype == f_txt.dtype == np.float16
    assert all(m['source'] == P.FI_SOURCE for m in fx['images']), 'FI-only store holds another source'
    assert all(0 <= t['img'] < n_fi for t in fx['texts'])

    for name, fname, key in (('old_store_unchanged_img_file', 'img.npy', 'img_file_sha256'),
                             ('old_store_unchanged_txt_file', 'txt.npy', 'txt_file_sha256'),
                             ('old_store_unchanged_index_file', 'index.json', 'index_file_sha256')):
        v = sha_file(old / fname)
        item(name, v, rec[key], v == rec[key])
    item('fi_counts', [n_fi, n_ft], list(expected_fi), (n_fi, n_ft) == tuple(expected_fi))
    old_keys = {(m['source'], m['image_name']) for m in oi['images']}
    dup = sum((m['source'], m['image_name']) in old_keys for m in fx['images'])
    item('fi_keys_new', dup, 0, dup == 0)
    gate = check_arrays(f_img, f_txt, tol=TOL)
    item('fi_extraction_gate', gate, {'n_nonfinite': 0, 'n_bad_norm': 0}, gate['ok'])

    ex, fe = oi['extra'], fx['extra']
    extra = dict(ex)
    extra['sources'] = list(ex['sources']) + list(fe['sources'])
    for k in ('n_images_per_source', 'n_texts_per_source'):
        extra[k] = {**ex[k], **fe[k]}
    extra['n_empty_skipped'] = ex['n_empty_skipped'] + fe['n_empty_skipped']
    extra['n_rows_dropped_other_sources'] = max(0, ex['n_rows_dropped_other_sources'] - n_ft - fe['n_empty_skipped'])
    texts = oi['texts'] + [{**t, 'img': t['img'] + n_oi} for t in fx['texts']]
    index = {**oi, 'n_images': n_oi + n_fi, 'n_texts': n_ot + n_ft,
             'n_texts_truncated': oi['n_texts_truncated'] + fx['n_texts_truncated'], 'extra': extra,
             'images': oi['images'] + fx['images'], 'texts': texts}

    if not all(v['ok'] for v in items.values()):  # nothing is written when the inputs already fail
        return _finish(items, stage_file, out, False, old)
    if out.exists():
        raise SystemExit(f'{out} exists; merge refuses to write into it')
    out.mkdir(parents=True)
    from numpy.lib.format import open_memmap
    for name, a, b in (('img.npy', o_img, f_img), ('txt.npy', o_txt, f_txt)):
        m = open_memmap(out / name, mode='w+', dtype=np.float16, shape=(len(a) + len(b), a.shape[1]))
        for src, off in ((a, 0), (b, len(a))):
            for i in range(0, len(src), CHUNK):
                j = min(i + CHUNK, len(src))
                m[off + i:off + j] = src[i:j]
        m.flush()
        del m
    json.dump(index, open(out / 'index.json', 'w'))
    json.dump({'old': str(old), 'fi': str(fi), 'hashes_file': str(hashes_file), 'n_old': [n_oi, n_ot]},
              open(out / 'merge.json', 'w'))

    # the same checks on the written result, against the recorded hashes
    m_img, m_txt = np.load(out / 'img.npy', mmap_mode='r'), np.load(out / 'txt.npy', mmap_mode='r')
    mi = json.load(open(out / 'index.json'))
    for name, v, ref in (('merged_old_img_rows', sha_rows(m_img, rec['n_images']), rec['img_rows_sha256']),
                         ('merged_old_txt_rows', sha_rows(m_txt, rec['n_texts']), rec['txt_rows_sha256']),
                         ('merged_old_index_images', sha_json(mi['images'][:rec['n_images']]), rec['images_sha256']),
                         ('merged_old_index_texts', sha_json(mi['texts'][:rec['n_texts']]), rec['texts_sha256'])):
        item(name, v, ref, v == ref)
    fi_eq = [sha_rows(m_img[rec['n_images']:]) == sha_rows(f_img), sha_rows(m_txt[rec['n_texts']:]) == sha_rows(f_txt)]
    item('merged_fi_rows_equal', fi_eq, [True, True], all(fi_eq))
    want = [rec['n_images'] + n_fi, rec['n_texts'] + n_ft]
    got = [len(m_img), len(m_txt), mi['n_images'], mi['n_texts']]
    item('merged_counts', got, want * 2, got == want * 2)
    off_ok = all(n_oi <= t['img'] < n_oi + n_fi for t in mi['texts'][rec['n_texts']:])
    item('merged_fi_text_img_offset', off_ok, True, off_ok)
    return _finish(items, stage_file, out, swap, old)


def _finish(items, stage_file, out, swap, old):
    ok = all(v['ok'] for v in items.values())
    res = {'items': items, 'ok': ok, 'swapped': False}
    if swap and ok:
        pre = old.with_name(old.name + '_pre_fi')
        if pre.exists():
            raise SystemExit(f'{pre} exists; refusing to swap')
        old.rename(pre)
        out.rename(old)
        res['swapped'] = True
        res['pre_fi_folder'] = str(pre)
    stage_file.parent.mkdir(parents=True, exist_ok=True)
    stage_file.write_text(json.dumps(res, indent=1) + '\n')
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('record')
    r.add_argument('--store', type=Path, default=P.STORE)
    r.add_argument('--out-file', type=Path, default=P.STORE_HASHES_FILE)
    m = sub.add_parser('merge')
    m.add_argument('--fi', type=Path, default=P.STORE_FI)
    m.add_argument('--out', type=Path, required=True)
    m.add_argument('--old', type=Path, default=P.STORE)
    m.add_argument('--hashes-file', type=Path, default=P.STORE_HASHES_FILE)
    m.add_argument('--stage-file', type=Path, default=None)
    m.add_argument('--swap', action='store_true')
    a = ap.parse_args(argv)
    if a.cmd == 'record':
        print(json.dumps(record(a.store, a.out_file)))
    else:
        res = merge(a.old, a.fi, a.out, a.hashes_file, a.stage_file, a.swap)
        print(json.dumps({k: v['ok'] for k, v in res['items'].items()}), 'ok =', res['ok'], 'swapped =', res['swapped'])
        raise SystemExit(0 if res['ok'] else 1)


if __name__ == '__main__':
    main()
