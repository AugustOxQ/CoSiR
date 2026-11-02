"""EmoSet keys, the hashed evaluation order, labels for scoring and the near-copy cut (r14 spec §3.3, D9, D17, D20).

Keys (C6): the image path string of each split-file entry (`image/awe/awe_08973.jpg`), validation then test file order
before any cut. The path names the class, so NO function here derives a class from a key; `labels(keys)` reads the
first field of the split entry and is the only label reader (scoring code calls it; nothing else does).

Evaluation order (D17): `order(keys)` sorts by the sha256 hex digest of the key string. Bootstrap rows, query and
gallery order (with its ties) and cross-fit halves by parity all use this order.

Near-copy cut (D9, D20): an EmoSet validation or test photo leaves its set when the cosine between its default Qwen
image and that of any of the 72,255 Affection train photos is >= NEAR_COPY_CUT (inclusive). The computation is r13's
(`pf_nearcopy`): float32 unit rows of the default store joined by key, blockwise CPU products, the float32 cosine
against the cut in float64, the first maximum on ties. Images only; labels are never read. `keys(split)` returns the
cut set (needs the record), `keys(split, cut=False)` the uncut set. `replay_r13` replays r13's cut
(queried = Affection train, reference = Affection val and test) through `near_copy_record` and `bs_stores.load` as gate 0(x).

CLI (run by the controller; CPU, several minutes):
  python bs_emoset.py nearcopy [--out FILE]   # the real cut -> RECORDS/nearcopy_emoset.json (+ run evidence in OUT/stage0)
  python bs_emoset.py gate0x                  # gate 0(x) -> OUT/stage0/gate_0x.json
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bs_paths as BP  # noqa: E402

ROOT = BP.EMOSET_ROOT                    # tests point these at a synthetic split folder
COUNTS = BP.EMOSET_SPLITS
SPLITS = ('val', 'test')
HIST_EDGES = [round(0.80 + 0.01 * i, 2) for i in range(21)]     # 0.80, 0.81, ..., 1.00 (spec §3.3)
BLOCK = 2048                             # queried rows per block: 2048 x 72,255 float32 = 0.6 GB at most
AFFECTION_TRAIN_N = 72255                 # D20: the reference of the real cut
_ENTRIES = {}


def record_file():
    return BP.RECORDS / 'nearcopy_emoset.json'


# ------------------------------------------------------------------------------------------------ split files
def _entries(split):
    """The split file's entries in file order; each `[label, 'image/<class>/<class>_<n>.jpg', 'annotation/...']`."""
    assert split in SPLITS, split
    path = Path(ROOT) / f'{split}.json'
    stamp = (str(path), path.stat().st_mtime_ns)
    if stamp not in _ENTRIES:
        entries = json.loads(path.read_text())
        assert len(entries) == COUNTS[split], f'{split}: {len(entries)} entries, expected {COUNTS[split]}'
        assert all(len(e) == 3 and e[1].startswith('image/') and e[2].startswith('annotation/') for e in entries), \
            f'{split}: entry shape'
        assert len({e[1] for e in entries}) == len(entries), f'{split}: duplicate keys'
        _ENTRIES[stamp] = entries
    return _ENTRIES[stamp]


def _split_keys(split):
    return [e[1] for e in _entries(split)]


def order(keys):
    """Keys sorted by the sha256 hex digest of the key string (D17). Keys must be unique."""
    keys = list(keys)
    assert len(set(keys)) == len(keys), 'order: keys must be unique'
    return sorted(keys, key=lambda k: hashlib.sha256(k.encode()).hexdigest())


def labels(keys):
    """Class label of each key: the FIRST FIELD of its split entry (validation or test file). The only label reader;
    called only by scoring code. Never looks at the key's path."""
    table = {}
    for split in SPLITS:
        for e in _entries(split):
            assert e[1] not in table, f'key in both splits: {e[1]}'
            table[e[1]] = e[0]
    keys = list(keys)
    missing = [k for k in keys if k not in table]
    assert not missing, f'{len(missing)} keys are in no split file, e.g. {missing[:2]}'
    return [table[k] for k in keys]


def keys(split, cut=True, record=None):
    """Keys of an EmoSet split in sha256 order. `cut=True`: without the near-copies of the record (asserts it exists
    and fits the split file); `cut=False`: the uncut set (diagnostics, §3.6). `record` overrides the record file."""
    every = _split_keys(split)
    if not cut:
        return order(every)
    path = Path(record) if record is not None else record_file()
    assert path.exists(), f'near-copy record missing: {path} (run `bs_emoset.py nearcopy`)'
    rec = json.loads(path.read_text())
    assert rec['cut'] == BP.NEAR_COPY_CUT, f"record cut {rec['cut']} != {BP.NEAR_COPY_CUT}"
    ref = rec['reference']
    assert ref['n'] == AFFECTION_TRAIN_N and ref['view'] == 'qwen_default', f'record reference block: {ref}'
    part = rec['splits'][split]
    removed = set(part['removed_keys'])
    assert part['n'] == len(every) and len(removed) == part['n_removed'] == len(part['removed_keys']), 'record counts'
    assert removed <= set(every), 'record names keys that are not in the split file'
    return order([k for k in every if k not in removed])


# ------------------------------------------------------------------------------------------------ near-copy
def nearest(query, reference, block=BLOCK):
    """Per queried row: the max cosine to any reference row (float32) and the first reference row reaching it (int64).
    Rows are unit float32. Of equal cosines the earlier reference row counts (r13's `_nearest`)."""
    assert query.dtype == reference.dtype == np.float32 and query.ndim == reference.ndim == 2, 'float32 matrices'
    assert query.shape[1] == reference.shape[1] and len(query) and len(reference), 'shapes'
    best = np.empty(len(query), dtype=np.float32)
    arg = np.empty(len(query), dtype=np.int64)
    for a in range(0, len(query), block):
        sim = query[a:a + block] @ reference.T
        j = sim.argmax(axis=1)
        arg[a:a + block] = j
        best[a:a + block] = sim[np.arange(len(j)), j]
    return best, arg


def removed_mask(best, cut=BP.NEAR_COPY_CUT):
    """True where the float32 cosine reaches the cut (inclusive), compared in float64 as r13 did."""
    return best.astype(np.float64) >= cut


def histogram(best):
    """Counts of the max cosines in the 20 bins 0.80..1.00 (last bin closed); float32 cosines above 1 go in the last."""
    b = np.minimum(best.astype(np.float64), 1.0)
    counts, _ = np.histogram(b, bins=HIST_EDGES)
    assert counts.sum() == int((b >= HIST_EDGES[0]).sum()), 'histogram counts'
    return dict(edges=HIST_EDGES, counts=[int(c) for c in counts], n_below=int((b < HIST_EDGES[0]).sum()))


def _assert_unit(x, name):
    norms = np.linalg.norm(x.astype(np.float64), axis=1)
    assert np.isfinite(norms).all() and np.abs(norms - 1).max() < 1e-3, f'{name}: rows must be unit'


def unit(x):
    """float32, L2-normalized rows (as ac_data._unit)."""
    x = np.asarray(x).astype(np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    assert np.isfinite(norms).all() and (norms > 0).all(), 'zero/nonfinite feature norm'
    return x / norms


def near_copy_record(ref_keys, load_ref, query_keys, load_query, *, cut=BP.NEAR_COPY_CUT, block=BLOCK):
    """The near-copy record. `ref_keys`: reference photos (all Affection train); `query_keys`: {split: keys in split-file
    order}. `load_ref(keys)` / `load_query(keys)` return float32 rows [len(keys), d] joined BY KEY (not position); rows
    are normalized here. Labels are not read. Returns the record (no timings: it is committed)."""
    ref_keys = list(ref_keys)
    assert len(set(ref_keys)) == len(ref_keys), 'reference keys must be unique'
    ref = unit(load_ref(ref_keys))
    assert ref.shape[0] == len(ref_keys)
    _assert_unit(ref, 'reference')
    out = {}
    for split, qkeys in query_keys.items():
        qkeys = list(qkeys)
        assert len(set(qkeys)) == len(qkeys) and not set(qkeys) & set(ref_keys), 'keys must be distinct'
        q = unit(load_query(qkeys))
        assert q.shape == (len(qkeys), ref.shape[1])
        _assert_unit(q, split)
        best, _ = nearest(q, ref, block)
        gone = removed_mask(best, cut)
        out[split] = dict(n=len(qkeys), n_removed=int(gone.sum()), n_kept=int((~gone).sum()),
                          removed_keys=sorted(k for k, g in zip(qkeys, gone) if g), hist=histogram(best))
    sha = hashlib.sha256('\n'.join(sorted(ref_keys)).encode()).hexdigest()
    return dict(cut=cut, rule='cosine >= cut (inclusive); float32 unit rows; max over reference; labels not read',
                reference=dict(dataset='affection', split='train', view='qwen_default', n=len(ref_keys),
                               keys_sha256=sha), queried_view='qwen_default', splits=out)


def real_record(block=BLOCK):
    """The real record: queried EmoSet val and test, reference all 72,255 Affection train photos (default Qwen)."""
    import bs_stores
    train_keys = _affection_train_keys()
    assert len(train_keys) == AFFECTION_TRAIN_N, f'{len(train_keys)} Affection train keys, expected 72,255 (D20)'
    return near_copy_record(
        train_keys, lambda ks: bs_stores.load('affection', 'qwen_default', ks),
        {s: _split_keys(s) for s in SPLITS}, lambda ks: bs_stores.load('emoset', 'qwen_default', ks), block=block)


def _affection_train_keys():
    """Affection train photos `<source>/<image_name>` as r13 defines them (pf_data.keys('train'), sorted)."""
    _r13_on_path()
    import pf_data
    return pf_data.keys('train')


def _r13_on_path():
    p = str(BP.R13_ROUND)
    if p not in sys.path:
        sys.path.insert(0, p)


def replay_r13(block=BLOCK):
    """Gate 0(x): r13's cut replayed through the code that makes the real EmoSet cut: `near_copy_record` with
    `bs_stores.load('affection', 'qwen_default', keys)` as both loaders (so `bs_stores.load` and `unit` run too).
    Roles as r13's: reference = Affection val then test, queried = Affection train. Compares the removed set and the
    histogram with r13's committed record. Returns a dict with `match` (removed keys identical), counts and seconds."""
    import bs_stores
    started = time.time()
    _r13_on_path()
    import pf_data
    train_keys = pf_data.keys('train')
    held_keys = pf_data.keys('val') + pf_data.keys('test')
    load = lambda ks: bs_stores.load('affection', 'qwen_default', ks)
    rec = near_copy_record(held_keys, load, {'train': train_keys}, load, block=block)
    part = rec['splits']['train']
    theirs = json.loads((BP.R13_ROUND / 'stage0a_nearcopy.json').read_text())
    mine = part['removed_keys']
    return dict(match=mine == theirs['removed_keys'], n_removed=part['n_removed'], r13_n_removed=theirs['n_removed'],
                n_train=len(train_keys), hist_match=part['hist']['counts'] == theirs['hist']['counts'],
                missing_from_mine=sorted(set(theirs['removed_keys']) - set(mine))[:5],
                extra_in_mine=sorted(set(mine) - set(theirs['removed_keys']))[:5],
                seconds=round(time.time() - started, 1))


def main(argv):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=('nearcopy', 'gate0x'))
    ap.add_argument('--out', default=None, help='record file (nearcopy); default RECORDS/nearcopy_emoset.json')
    args = ap.parse_args(argv)
    stage0 = BP.OUT / 'stage0'
    stage0.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if args.what == 'gate0x':
        res = replay_r13()
        (stage0 / 'gate_0x.json').write_text(json.dumps(res, indent=1) + '\n')
        print(res)
        assert res['match'] and res['n_removed'] == 2328, 'gate 0(x) failed'
        return
    path = Path(args.out) if args.out else record_file()
    rec = real_record()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=1) + '\n')
    evidence = dict(record=str(path), seconds=round(time.time() - started, 1),
                    **{s: {k: v for k, v in p.items() if k not in ('removed_keys', 'hist')} for s, p in rec['splits'].items()})
    (stage0 / 'nearcopy_emoset_run.json').write_text(json.dumps(evidence, indent=1) + '\n')
    print(evidence)


if __name__ == '__main__':
    main(sys.argv[1:])
