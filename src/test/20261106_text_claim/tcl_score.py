"""r23 scoring runner (ticket 05; spec 2026-11-06 text-claim design §3.4, D11, D12).

    python tcl_score.py --set art_val [--kind caption] [--rows plain cstar ...] [--smoke]

One score file per row, head, seed, direction and read-out (`tcl_paths.score_path`): `q_pid` (int64), `first_rank` (float32,
NaN = no hit) and `ap__<entry>` (float32, NaN = not scored) for each entry `<aspect>`, `<aspect>__no_leak`,
`<aspect>__cross_painting`. A head row reads its head's aspect; a one-vector row all three; `roberta` emotion, text to text.
A validation set fits the means of every row (labelled or not) and writes `means_path` before it ranks; a held set reads
its validation set's means of the same row, head, seed and kind. Run from the round folder.
"""
import argparse
import gc
import os
import time
from pathlib import Path

import numpy as np
import torch

import tcl_paths as P
from tcl_metrics import evaluate, fit_means

BLOCK = 1024


def query_gallery_means(direction, text_mean, image_mean):
    """(query mean, gallery mean) for `evaluate(means=)` from `fit_means`'s (text, image). The one place the pair is ordered."""
    if direction == 't2i':
        return text_mean, image_mean
    if direction == 'i2t':
        return image_mean, text_mean
    assert direction == 't2t'
    return text_mean, text_mean


def _entries(direction, aspects, fr, flags):
    """r22's `entries_for`: label entry per aspect, `__no_leak` (text sides drop texts that name their label) and
    `__cross_painting` (the query's own painting leaves the gallery). New arrays; the frame is not written."""
    q_text, g_text = direction[0] == 't', direction[2] == 't'
    entries = {}
    for a in aspects:
        lab = fr['labels'][a]
        entry = dict(q=lab['t'] if q_text else lab['i'], g=lab['t'] if g_text else lab['i'])
        keep = ~flags[a]
        assert keep.shape == (fr['n_t'],) and keep.dtype == bool
        entries[a] = entry
        entries[a + '__no_leak'] = dict(entry, q_keep=keep if q_text else None, g_keep=keep if g_text else None)
        entries[a + '__cross_painting'] = dict(entry, exclude_same_painting=True)
    return entries


def _save(path, **arrays):
    """Write to a temporary name in the same folder, then rename: a killed job leaves no partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'wb') as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def _complete(path, aspects, q_pid):
    """A score file is complete when its keys, query IDs, dtypes and column lengths match the frame."""
    if not Path(path).exists():
        return False
    with np.load(path, allow_pickle=False) as z:
        keys = {'q_pid', 'first_rank'} | {f'ap__{a}{s}' for a in aspects
                                         for s in ('', '__no_leak', '__cross_painting')}
        return (set(z.files) == keys and np.array_equal(z['q_pid'], q_pid)
                and all(z[k].shape == q_pid.shape and z[k].dtype == (np.int64 if k == 'q_pid' else np.float32)
                        for k in keys))


def _plan(row, head):
    """(aspects, directions) of a row: a head row reads its head, `roberta` emotion text to text, the rest all three."""
    if row == 'roberta':
        return ('emotion',), ('t2t',)
    return ((head,) if row in P.HEAD_ROWS else P.ASPECTS), P.DIRECTIONS


def plan(set_name, rows=None):
    """[(row, head, seed)] to score on a set: the bar rows on every set, the side rows on the ArtELingo sets (D12). `rows`
    narrows to some of them."""
    allowed = P.BAR_ROWS + (P.SIDE_ROWS if P.GROUP[set_name] == 'art' else ())
    assert rows is None or set(rows) <= set(allowed), (set_name, rows)
    out = []
    for row in allowed:
        if rows is not None and row not in rows:
            continue
        for head in (P.ASPECTS if row in P.HEAD_ROWS else ('all',)):
            for seed in (P.SEEDS if row in P.SEEDED_ROWS else ('x',)):
                out.append((row, head, seed))
    return out


def unit_done(set_name, kind, row, head, seed, fr, out=None):
    """Every score file of the row, head and seed is complete (and, on validation, so are its means)."""
    aspects, directions = _plan(row, head)
    if set_name in ('art_val', 'mm_val'):
        path = P.means_path(set_name, kind, row, head, seed, out=out)
        if not path.exists():
            return False
        with np.load(path, allow_pickle=False) as z:
            if set(z.files) != {'t', 'i', 'n_t', 'n_i'}:
                return False
            if (int(z['n_t']), int(z['n_i'])) != (fr['n_t'], 0 if row == 'roberta' else fr['n_i']):
                return False
    return all(_complete(P.score_path(set_name, kind, row, head, seed, d, r, out=out), aspects,
                         fr['tp'] if d[0] == 't' else fr['ip'])
               for d in directions for r in P.READOUTS)


def score_unit(set_name, kind, row, head, seed, text, image, fr, flags, device, out=None, block=BLOCK, val_n=None):
    """Rank one row, head and seed in every direction and read-out; one file each. `text`, `image` are the rows of
    `tcl_forward.rows` in frame order (`image` None for roberta); `fr` is `tcl_data.frame`; `flags` maps aspect to
    `names_label`. A validation set fits and writes its means first; a held set reads its validation set's, whose row
    counts must be `val_n` = (n_t, n_i)."""
    assert set_name in P.SETS and row in P.BAR_ROWS + P.SIDE_ROWS
    assert (head in P.ASPECTS) == (row in P.HEAD_ROWS) and (seed in P.SEEDS) == (row in P.SEEDED_ROWS), (row, head, seed)
    text_only = row == 'roberta'
    assert text.shape[0] == fr['n_t'] and (image is None) == text_only
    assert text_only or image.shape[0] == fr['n_i']
    aspects, directions = _plan(row, head)
    if unit_done(set_name, kind, row, head, seed, fr, out=out):
        return
    mt, mi = unit_means(set_name, kind, row, head, seed, text, image, out, val_n)
    pid = dict(t=fr['tp'], i=fr['ip'])
    for direction in directions:
        entries = _entries(direction, aspects, fr, flags)
        q, g = (text if direction[0] == 't' else image), (text if direction[2] == 't' else image)
        for readout in P.READOUTS:
            path = P.score_path(set_name, kind, row, head, seed, direction, readout, out=out)
            if _complete(path, aspects, pid[direction[0]]):
                continue
            started = time.monotonic()
            result = evaluate(q, g, pid[direction[0]], pid[direction[2]], entries, direction == 't2t', readout, device,
                              block=block, means=query_gallery_means(direction, mt, mi))
            arrays = {f'ap__{k}': v for k, v in result['ap'].items()}
            _save(path, q_pid=pid[direction[0]], first_rank=result['first_rank'], **arrays)
            print(f'{set_name}/{kind}/{row}/{head}/s{seed}/{direction}/{readout}: {time.monotonic() - started:.2f}s', flush=True)


def unit_means(set_name, kind, row, head, seed, text, image, out, val_n):
    """(text mean, image mean) of a row. Validation: fitted on every row and written before any ranking. Held: read from
    the validation set's file of the same row, head, seed and kind. `roberta` has no image: its image mean is empty."""
    text_only = image is None
    if set_name in ('art_val', 'mm_val'):
        mt, mi = fit_means(text, text if text_only else image)
        if text_only:
            mi = np.zeros(0, np.float32)
        _save(P.means_path(set_name, kind, row, head, seed, out=out), t=mt, i=mi, n_t=np.int64(len(text)),
              n_i=np.int64(0 if text_only else len(image)))
        return mt, mi
    assert val_n is not None, 'a held set needs the row counts of its validation set'
    path = P.means_path(P.VAL_OF[set_name], kind, row, head, seed, out=out)
    with np.load(path, allow_pickle=False) as z:
        assert set(z.files) == {'t', 'i', 'n_t', 'n_i'}, path
        mt, mi, n_t, n_i = z['t'], z['i'], int(z['n_t']), int(z['n_i'])
    assert (n_t, n_i) == (val_n[0], 0 if text_only else val_n[1]), (path, n_t, n_i, val_n)
    assert mt.dtype == np.float32 and mt.shape == (text.shape[1],), path
    assert mi.dtype == np.float32 and mi.shape == (0 if text_only else text.shape[1],), path
    return mt, mi


def release(device):
    """Free rows and cached GPU memory between rows, as r22's `release`."""
    gc.collect()
    if torch.device(device).type == 'cuda':
        torch.cuda.empty_cache()


def held_guard(set_name):
    """Refuse to score a held set (§3.5: stage 1 is committed before any held number). A validation set needs nothing.
    Reads the set's own group's stage 1 record and status.json, and the other group's stage 1 record when that file exists,
    through `read_committed_record`: a missing file, or a file that differs from git HEAD, raises. A missing other-group
    record is allowed, since §3.5 and §6 void only that group's verdicts."""
    if set_name not in ('art_held', 'mm_test'):
        return
    own = P.GROUP[set_name]
    other = 'mm' if own == 'art' else 'art'
    P.read_committed_record(f'stage1_{own}.json')
    P.read_committed_record('status.json')
    if P.record_path(f'stage1_{other}.json').exists():
        P.read_committed_record(f'stage1_{other}.json')


def run(set_name, kinds=None, rows=None, device='cuda', block=BLOCK, out=None):
    """Score a set: every kind (default all of its group), every planned row, head and seed. Rows are built and released
    per head and seed."""
    held_guard(set_name)
    import tcl_data
    import tcl_forward
    assert set_name in P.SETS
    kinds = tuple(kinds) if kinds else P.kinds(P.GROUP[set_name])
    assert set(kinds) <= set(P.kinds(P.GROUP[set_name])), (set_name, kinds)
    cuda = torch.device(device).type == 'cuda'
    for kind in kinds:
        fr = tcl_data.frame(set_name, kind)
        todo = [u for u in plan(set_name, rows) if not unit_done(set_name, kind, *u, fr, out=out)]
        print(f'{set_name}/{kind}: {len(todo)} of {len(plan(set_name, rows))} units to score', flush=True)
        if not todo:
            continue
        flags = {a: tcl_data.names_label(set_name, kind, a) for a in P.ASPECTS}
        val_n = None
        if P.VAL_OF[set_name] != set_name:
            val = tcl_data.frame(P.VAL_OF[set_name], kind)
            val_n = (val['n_t'], val['n_i'])
        for row, head, seed in todo:
            if cuda:
                torch.cuda.reset_peak_memory_stats(device)
            text, image = tcl_forward.rows(set_name, kind, row, head, seed, device)
            score_unit(set_name, kind, row, head, seed, text, image, fr, flags, device, out=out, block=block, val_n=val_n)
            del text, image
            release(device)
            if cuda:
                print(f'  peak GPU memory {torch.cuda.max_memory_allocated(device) / 2**30:.2f} GiB', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--set', required=True, choices=P.SETS, dest='set_name')
    p.add_argument('--kind', help='one text kind (default: all kinds of the set)')
    p.add_argument('--rows', nargs='+', help='some of the set\'s rows (default: all)')
    p.add_argument('--smoke', action='store_true', help='must agree with TCL_SMOKE=N and TCL_OUTPUTS in the environment')
    p.add_argument('--device', default='cuda')
    p.add_argument('--block', type=int, default=BLOCK)
    args = p.parse_args()
    assert args.smoke == bool(P.SMOKE), '--smoke and TCL_SMOKE=N go together'
    run(args.set_name, [args.kind] if args.kind else None, args.rows, args.device, args.block)


if __name__ == '__main__':
    main()
