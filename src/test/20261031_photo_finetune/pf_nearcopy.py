"""Near-copy rule of r13 (spec 2026-10-31-photo-finetune-design.md §3.3 D12, §3.5 0a(v)).

A train photo leaves training when the cosine between its default Qwen image and that of any validation or test photo
is >= NEAR_COPY_CUT (inclusive), in float32 on unit rows of the default store's `img.npy`, joined by key (C6). Images
only: no label is read. Blockwise on CPU so that no full train x held matrix is ever held.
"""
import json
import time

import numpy as np

import pf_paths
import pf_data

HIST_EDGES = [round(0.80 + 0.01 * i, 2) for i in range(21)]     # 0.80, 0.81, ..., 1.00 (spec §3.3)
BLOCK = 2048                     # train rows per block: 2048 x 72,255 float32 = 0.6 GB at most


def _nearest(train, held, block):
    """Per train row: the max cosine to any held row (float32) and the first held row reaching it (int64)."""
    best = np.empty(len(train), dtype=np.float32)
    arg = np.empty(len(train), dtype=np.int64)
    for a in range(0, len(train), block):
        sim = train[a:a + block] @ held.T
        j = sim.argmax(axis=1)                  # first maximum: a tie goes to the earlier held row
        arg[a:a + block] = j
        best[a:a + block] = sim[np.arange(len(j)), j]
    return best, arg


def _within_train_copies(train, cut, block):
    """Train photos whose nearest other train photo has cosine >= cut (self-pairs excluded). Reported only."""
    count = 0
    for a in range(0, len(train), block):
        sim = train[a:a + block] @ train.T
        rows = np.arange(len(sim))
        sim[rows, a + rows] = -np.inf               # a photo is not its own copy
        count += int((sim.max(axis=1).astype(np.float64) >= cut).sum())
    return count


def compute(train_keys, train, held_keys, held_split, held, *, cut=pf_paths.NEAR_COPY_CUT, block=BLOCK):
    """The near-copy record and per-photo detail.

    `train` [n, d] and `held` [m, d] are float32 unit rows; `held_keys` / `held_split` ('val' or 'test') name the held
    rows. Held rows are ordered as given: of equal cosines the earlier row is the nearest.
    Returns (record, detail): the record has the ticket's fields, the detail has `cosine` (float32 [n]),
    `nearest_key` and `nearest_split` (per train photo).
    """
    assert train.dtype == held.dtype == np.float32 and train.ndim == held.ndim == 2, 'float32 matrices'
    assert train.shape == (len(train_keys), train.shape[1]) and held.shape == (len(held_keys), train.shape[1]), 'shapes'
    assert len(held_split) == len(held_keys) and set(held_split) <= {'val', 'test'}, 'held splits'
    assert len(set(train_keys)) == len(train_keys) and not set(train_keys) & set(held_keys), 'keys must be distinct'
    best, arg = _nearest(train, held, block)
    removed = best.astype(np.float64) >= cut     # the float32 cosine against the cut, exactly
    sources = sorted({k.split('/', 1)[0] for k in train_keys})
    by_source = {s: {'val': 0, 'test': 0} for s in sources}
    for key, gone, j in zip(train_keys, removed, arg):
        if gone:
            by_source[key.split('/', 1)[0]][held_split[j]] += 1
    # Cosines can round to just above 1 in float32; clip so that the closed last bin takes them.
    counts, _ = np.histogram(np.minimum(best.astype(np.float64), 1.0), bins=HIST_EDGES)
    assert counts.sum() == int((best.astype(np.float64) >= HIST_EDGES[0]).sum()), 'histogram counts'
    detail = dict(cosine=best, nearest_key=[held_keys[j] for j in arg], nearest_split=[held_split[j] for j in arg])
    record = dict(cut=cut, n_train=len(train_keys), n_removed=int(removed.sum()),
                  removed_keys=sorted(k for k, r in zip(train_keys, removed) if r), by_source=by_source,
                  hist=dict(edges=HIST_EDGES, counts=[int(c) for c in counts]),
                  within_train_copies=_within_train_copies(train, cut, block))
    assert sum(v['val'] + v['test'] for v in by_source.values()) == record['n_removed'], 'by_source counts'
    return record, detail


def run():
    """Compute the near-copy record on the default store and write NEARCOPY_FILE and OUTPUTS/stage0a/nearcopy.json.

    Reads images only (joined by key through `pf_data.default_images`), never labels. Returns the record.
    """
    started = time.time()
    train_keys = pf_data.keys('train')
    held_keys, held_split = [], []
    for split in ('val', 'test'):
        split_keys = pf_data.keys(split)
        held_keys += split_keys
        held_split += [split] * len(split_keys)
    train, held = pf_data.default_images(train_keys), pf_data.default_images(held_keys)
    record, _ = compute(train_keys, train, held_keys, held_split, held)
    pf_paths.NEARCOPY_FILE.parent.mkdir(parents=True, exist_ok=True)
    pf_paths.NEARCOPY_FILE.write_text(json.dumps(record, indent=1) + '\n')
    evidence = pf_paths.OUTPUTS / 'stage0a' / 'nearcopy.json'
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(json.dumps(dict(record, seconds=time.time() - started), indent=1) + '\n')
    return record


if __name__ == '__main__':
    result = run()
    print({k: v for k, v in result.items() if k != 'removed_keys'})
