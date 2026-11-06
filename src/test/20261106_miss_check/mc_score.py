"""Picked read-outs only, saved validation means only, two label entries per ranking."""
import numpy as np

from mc_common import OUT, P, R23, record
from mc_metrics import evaluate
from tcl_score import query_gallery_means, _save


def context(text):
    return ('art', 'caption', 'art_val') if text == 'A' else ('mm', 'narrative', 'mm_val')


def path(text, row, head, seed, direction, readout):
    return OUT / 'scores' / text / row / f'{head}__s{seed}__{direction}__{readout}.npz'


def means(text, row, head, seed):
    _, kind, val = context(text)
    with np.load(P.means_path(val, kind, row, head, seed, out=R23), allow_pickle=False) as z:
        assert int(z['n_i']) == P.N_PAINTINGS[val]
        assert int(z['n_t']) == (P.N_CAPTIONS[val] if text == 'A' else P.N_PAINTINGS[val])
        return z['t'], z['i']


def score(text, row, head, seed, t, i, fr, directions, readouts, device, block):
    mt, mi = means(text, row, head, seed)
    vectors, pids = dict(t=t, i=i), dict(t=fr['tp'], i=fr['ip'])
    aspects = (head,) if row in P.HEAD_ROWS else P.ASPECTS
    for d in directions:
        for ro in readouts(d):
            entries = {f'{s}__{a}': dict(q=lab[a][d[0]], g=lab[a][d[2]])
                       for s, lab in fr['labels'].items() for a in aspects}
            dest = path(text, row, head, seed, d, ro)
            qpid = pids[d[0]]
            if dest.exists():
                with np.load(dest, allow_pickle=False) as z:
                    assert set(z.files) == {'q_pid'} | {f'ap__{e}' for e in entries}
                    assert np.array_equal(z['q_pid'], qpid)
                    assert all(z[f'ap__{e}'].shape == qpid.shape for e in entries)
                continue
            aps = evaluate(vectors[d[0]], vectors[d[2]], entries, d == 't2t', ro, device,
                           block, query_gallery_means(d, mt, mi))
            _save(dest, q_pid=qpid, **{f'ap__{k}': v for k, v in aps.items()})


def run(text, fr, rows, device, block):
    from tcl_score import plan, release
    group, kind, _ = context(text)
    picks = record(group)['picks'][kind]
    for row, head, seed in plan('art_held' if text == 'A' else 'mm_test', P.BAR_ROWS):
        aspects = (head,) if row in P.HEAD_ROWS else P.ASPECTS
        readouts = lambda d: sorted({picks[row][a][d] for a in aspects})
        t, i = rows.rows(row, head, seed, device)
        score(text, row, head, seed, t, i, fr, P.DIRECTIONS, readouts, device, block)
        del t, i
        release(device)
    print(f'score {text}: picked label AP complete ({fr["n_i"]} paintings, {fr["n_t"]} texts)', flush=True)
