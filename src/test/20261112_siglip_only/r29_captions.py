"""r29 caption input layer, data side (ticket 01; spec §3.2, §3.5 gate (iii), C6).

`caption_rows` and `painting_means` read the per-caption text rows of a store in scripts/extract_features.py's format
(index.json, txt.npy) for the sources in `r29_paths.CAPTION_SOURCES`. Rows join to ArtELingo captions by extraction
sample ID with the painting checked, never by position. `painting_means` follows the CT's Qwen loader
(`ct2_data.default_captions`): raw float32 rows of a painting's captions in ascending sample ID, float64 sum and mean
(`apc_data.paint_mean`), then L2 (`apc_data.l2`). The `qwen` source is K0's own input; it is read through the CT's loaders.
"""
import hashlib
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r29_paths as P  # noqa: E402

P.use_older_rounds()   # r28's folder for `import r28_trainer` (r14's folders), used by the Qwen paths and the gate
from apc_data import l2, paint_mean  # noqa: E402  (the CT's own mean and normalisation)
P.check_older_imports()  # every older-round module loaded so far is the main checkout's copy

SOURCES = P.CAPTION_SOURCES              # tests patch this dict; name -> (store folder or None, input width)
CHUNK = 8192
FULL_CAPTIONS = 308723                   # ArtELingo full: every annotation caption has a row in every store
ROLE_COUNTS = dict(train=216107, val=30872, held=61744)   # ct2_data.FULL_COUNTS (train, val, held captions)


def _folder(source):
    assert source in SOURCES, f'unknown caption source {source!r}'
    folder, width = SOURCES[source]
    assert folder is not None, f'{source}: no per-caption store (K0 reads the CT loaders)'
    return Path(folder), width


# `index['backbone']` of each source's store (scripts/extract_features.py writes `enc.name`; a prefixed store is made by
# the same encoder as its base store, so it carries the base's name).
BACKBONE = {'siglip1': 'siglip_so400m', 'siglip1_prefix': 'siglip_so400m',
            'siglip2': 'siglip2_so400m', 'siglip2_prefix': 'siglip2_so400m',
            'clipl14': 'clip_l14', 'clipl14_prefix': 'clip_l14'}


def _read(source):
    """(sample_ids int64 [n], painting names [n], txt.npy memmap float16 [n, width]) of a source's store, checked.
    Cached on the store files' size and mtime, so a rewritten store is read again."""
    folder, width = _folder(source)
    assert source in BACKBONE, f'no backbone name for {source!r}'
    stat = tuple((f.stat().st_size, f.stat().st_mtime_ns)
                 for f in (folder / 'index.json', folder / 'progress.json', folder / 'txt.npy'))
    return _read_cached(folder, width, BACKBONE[source], stat)


@lru_cache(maxsize=8)
def _read_cached(folder, width, backbone, stat):
    index_bytes = (folder / 'index.json').read_bytes()
    index = json.loads(index_bytes)
    progress = json.loads((folder / 'progress.json').read_text())
    texts = index['texts']
    ids = np.asarray([r['sample_id'] for r in texts], dtype=np.int64)
    names = np.asarray([r['painting'] for r in texts])
    assert index['smoke'] is False, f'{folder}: a smoke store'
    assert index['backbone'] == backbone, f'{folder}: backbone {index["backbone"]!r}, expected {backbone!r}'
    assert progress['txt_done'] == index['n_texts'], f'{folder}: txt_done {progress["txt_done"]} != n_texts'
    assert progress['gate']['ok'] is True, f'{folder}: extraction gate not ok: {progress["gate"]}'
    assert progress['index_sha256'] == hashlib.sha256(index_bytes).hexdigest(), f'{folder}: index_sha256 differs from index.json'
    assert len(np.unique(ids)) == len(ids), f'{folder}: duplicate caption IDs'
    raw = np.load(folder / 'txt.npy', mmap_mode='r')
    assert raw.dtype == np.float16 and raw.shape == (len(ids), width), f'{folder}: txt.npy {raw.shape} {raw.dtype}'
    assert index['dim'] == width and index['n_texts'] == len(ids), f'{folder}: index.json disagrees with the width/rows'
    return ids, names, raw


def _values(raw, rows):
    """float32 rows of `raw` (float16 memmap) at `rows`, finite and non-zero."""
    x = np.empty((len(rows), raw.shape[1]), dtype=np.float32)
    for a in range(0, len(rows), CHUNK):
        x[a:a + CHUNK] = raw[rows[a:a + CHUNK]]
    assert np.isfinite(x).all(), 'nonfinite caption features'
    assert (np.linalg.norm(x, axis=1) > 0).all(), 'zero caption features'
    return x


def _join(ids, names, sample_ids, paintings):
    """Store row of each wanted sample ID; every ID present and its painting equal."""
    sample_ids = np.asarray(sample_ids, dtype=np.int64)
    paintings = np.asarray(paintings)
    assert sample_ids.ndim == 1 and paintings.shape == sample_ids.shape
    order = np.argsort(ids)
    pos = np.searchsorted(ids[order], sample_ids)
    pos[pos == len(ids)] = 0
    rows = order[pos]
    assert (ids[rows] == sample_ids).all(), f'missing caption IDs: {int((ids[rows] != sample_ids).sum())}'
    assert (names[rows] == paintings).all(), f'caption painting mismatch: {int((names[rows] != paintings).sum())}'
    return rows


def caption_rows(source, sample_ids, paintings):
    """float32 [n, w] stored rows (not re-normalised) of the captions `sample_ids`, in that order; the painting of each
    ID is checked against `paintings` (C6)."""
    ids, names, raw = _read(source)
    return _values(raw, _join(ids, names, sample_ids, paintings))


@lru_cache(maxsize=1)
def annotation_paintings():
    """Painting of every ArtELingo caption, by extraction sample ID (annotation row i is sample ID i, as `split2`)."""
    import r28_trainer  # noqa: F401  (puts the CT's folder on sys.path)
    import ct2_data
    records = json.loads(Path(ct2_data.ANNOTATIONS).read_text())
    assert len(records) == FULL_CAPTIONS
    return np.asarray([r['painting'] for r in records])


def painting_means(source, painting_names):
    """float32 [n, w]: for each painting in the given order, the L2-normalised mean of all its captions' rows, as
    `ct2_data` makes Qwen's. The captions of a painting are the annotations' (ascending sample ID), joined to the store
    by ID with the painting checked (C6): a missing or relabelled caption, or a store caption the annotations do not
    give that painting, raises. `qwen` goes through the CT's loaders."""
    names_in = [str(n) for n in painting_names]
    assert len(set(names_in)) == len(names_in), 'repeated painting names'
    if source == 'qwen':
        return _qwen_means(names_in)
    ids, names, raw = _read(source)
    ann = annotation_paintings()
    lookup = {n: k for k, n in enumerate(names_in)}
    pid_ann = np.asarray([lookup.get(str(n), -1) for n in ann], dtype=np.int64)
    wanted = np.flatnonzero(pid_ann >= 0)                        # ascending sample ID
    wanted = wanted[np.argsort(pid_ann[wanted], kind='stable')]    # painting, then ascending ID, as split2
    pid = pid_ann[wanted]
    counts = np.bincount(pid, minlength=len(names_in))
    assert (counts > 0).all(), f'no captions for {[names_in[k] for k in np.flatnonzero(counts == 0)[:5]]}'
    rows = _join(ids, names, wanted, ann[wanted])
    n_store = int(np.isin(names, np.asarray(names_in)).sum())
    assert n_store == len(wanted), f'{source}: {n_store - len(wanted)} extra store captions for the wanted paintings'
    return l2(paint_mean(_values(raw, rows), pid, len(names_in)))


def _qwen_means(names_in):
    """K0's painting means: `ct2_data.default_captions` of the CT's full split, selected by name."""
    import r28_trainer  # noqa: F401  (puts r14's and the CT's folders on sys.path)
    import ct2_data
    s = ct2_data.split2('full')
    lookup = {str(n): k for k, n in enumerate(s['all_painting_ids'])}
    assert all(n in lookup for n in names_in), 'painting outside the CT\'s train and validation sets'
    return ct2_data.default_captions(s)[[lookup[n] for n in names_in]]


def store_hash(source):
    """sha256 of the store's index.json bytes (`qwen`: the Qwen full store the CT falls back to for unsealed IDs)."""
    if source == 'qwen':
        import r28_trainer  # noqa: F401
        import ct2_data
        return hashlib.sha256((Path(ct2_data.QWEN) / 'index.json').read_bytes()).hexdigest()
    folder, _ = _folder(source)
    return hashlib.sha256((folder / 'index.json').read_bytes()).hexdigest()


# --- gate (iii) ------------------------------------------------------------------------------------------------

def _counts(names):
    u, c = np.unique(names, return_counts=True)
    return dict(zip(u.tolist(), c.tolist()))


def gate_iii_checks(*, sig_ids, sig_names, qwen_ids, qwen_names, roles):
    """Gate (iii) on arrays: `roles` {role: (sample IDs, annotation paintings)}. Passes when (1) every Qwen caption has
    a SigLIP row with the same sample ID and painting, and the stores hold the same IDs; (2) the per-painting caption
    counts are equal; (3) every role's captions join to the SigLIP store by ID with painting equal, the roles are
    disjoint and together are exactly the store's captions."""
    sig_ids, qwen_ids = np.asarray(sig_ids, dtype=np.int64), np.asarray(qwen_ids, dtype=np.int64)
    sig_names, qwen_names = np.asarray(sig_names), np.asarray(qwen_names)
    checks = {}
    checks['same_ids'] = bool(np.array_equal(np.sort(sig_ids), np.sort(qwen_ids)))
    try:
        _join(sig_ids, sig_names, qwen_ids, qwen_names)
        checks['qwen_ids_join_with_painting'] = True
    except AssertionError:
        checks['qwen_ids_join_with_painting'] = False
    checks['painting_counts_equal'] = _counts(sig_names) == _counts(qwen_names)
    seen = []
    for role, (ids, paintings) in roles.items():
        try:
            _join(sig_ids, sig_names, ids, paintings)
            checks[f'{role}_joins'] = True
        except AssertionError:
            checks[f'{role}_joins'] = False
        seen.append(np.asarray(ids, dtype=np.int64))
    union = np.concatenate(seen)
    checks['roles_disjoint_and_cover_store'] = bool(len(np.unique(union)) == len(union)
                                                    and np.array_equal(np.sort(union), np.sort(sig_ids)))
    details = dict(checks=checks, n_captions=len(sig_ids), n_paintings=len(_counts(sig_names)),
                   roles={r: len(v[0]) for r, v in roles.items()})
    return dict(status='pass' if all(checks.values()) else 'fail', details=details)


def write_gate_record(folder, result, gate='iii'):
    """`<folder>/<gate>.json` = {gate, status, tries, details, written}; tries counts the file's earlier runs."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    assert result['status'] in ('pass', 'fail', 'fallback', 'parked')
    path = Path(folder) / f'{gate}.json'
    tries = json.loads(path.read_text())['tries'] + 1 if path.exists() else 1
    record = dict(gate=gate, status=result['status'], tries=tries, details=result['details'],
                  written=datetime.now(ZoneInfo('Europe/Amsterdam')).strftime('%Y-%m-%d %H:%M'))
    Path(folder).mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + '\n')
    return record


def gate_iii(source='siglip1', write=False):
    """Gate (iii) on the real stores (writes `records/gates/iii.json` only with `write=True`: a try is used up by each write): SigLIP 1 against the Qwen full store (the CT's fallback store), the annotations,
    the CT's train and validation captions (`split2('full')`) and the held paintings' captions; also builds the
    painting means of train and validation and checks they are one per painting in the CT's order, finite, unit."""
    import r28_trainer as R
    import ct2_data
    folder, width = _folder(source)
    sig_ids, sig_names, _ = _read(source)
    qwen = json.loads((Path(ct2_data.QWEN) / 'index.json').read_text())['texts']
    qwen_ids = np.asarray([r['sample_id'] for r in qwen], dtype=np.int64)
    qwen_names = np.asarray([r['painting'] for r in qwen])
    ann = annotation_paintings()                               # annotation row i is extraction sample ID i (split2)
    assert len(ann) == FULL_CAPTIONS == len(sig_ids)
    s = ct2_data.split2('full')
    in_dev = np.isin(s['pid'], s['dev'])
    held_names, _ = R.B.rb_student.held_names()
    held_ids = np.flatnonzero(np.isin(ann, np.asarray(held_names)))
    roles = dict(train=(s['sample_ids'][~in_dev], s['painting_ids'][~in_dev]),
                 val=(s['sample_ids'][in_dev], s['painting_ids'][in_dev]),
                 held=(held_ids, ann[held_ids]))
    out = gate_iii_checks(sig_ids=sig_ids, sig_names=sig_names, qwen_ids=qwen_ids, qwen_names=qwen_names, roles=roles)
    # annotations against the store, and the roles' sizes against the CT's counts
    out['details']['checks']['annotation_painting_equals_store'] = bool(
        (ann[sig_ids] == sig_names).all())
    out['details']['checks']['role_sizes_equal_ct_counts'] = all(
        len(roles[r][0]) == n for r, n in ROLE_COUNTS.items())
    means = painting_means(source, s['all_painting_ids'])
    counts = _counts(sig_names)
    out['details']['checks']['train_val_painting_means_ok'] = bool(
        means.shape == (len(s['all_painting_ids']), width) and np.isfinite(means).all()
        and np.allclose(np.linalg.norm(means, axis=1), 1.0, atol=1e-5)
        and all(counts[n] == k for n, k in zip(s['all_painting_ids'].tolist(), np.diff(np.r_[s['first'], len(s['pid'])]).tolist())))
    out['status'] = 'pass' if all(out['details']['checks'].values()) else 'fail'
    out['details']['source'] = source
    out['details']['store_sha256'] = store_hash(source)
    if write:
        write_gate_record(P.GATES, out)
    return out


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='gate (iii) on the real stores')
    ap.add_argument('--write', action='store_true', help='write records/gates/iii.json (adds one try)')
    result = gate_iii(write=ap.parse_args().write)
    print(result['status'], json.dumps(result['details']['checks']))
