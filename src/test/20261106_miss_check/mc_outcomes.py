"""Four fixed comparisons over nine cells of each available crossed set."""
import json

import numpy as np

from mc_common import OUT, P, R23, ROUND, interval, matched, record, write
from mc_score import context, path
from tcl_metrics import outcome


def load(text, entry, row, head, seed, a, d, ro, fr):
    with np.load(path(text, row, head, seed, d, ro), allow_pickle=False) as z:
        pid, ap = z['q_pid'], z[f'ap__{entry}__{a}'].astype(np.float64)
    assert np.array_equal(pid, fr['tp'] if d[0] == 't' else fr['ip'])
    return pid, ap


def run(frames, smoke):
    cells = []
    available = [t for t in ('A', 'M') if (OUT / 'scores' / t).exists()]
    assert available, 'run score first'
    if not smoke:
        assert available == ['A', 'M'], 'full outcomes require both text sets'
    for text in available:
        fr = frames[text]
        group, kind, _ = context(text)
        rec = record(group)
        picks = rec['picks'][kind]
        for entry in fr['labels']:
            for candidate in ('cstar', 'late'):
                for comparator in ('plain', 'teacher'):
                    for a in P.ASPECTS:
                        for d in P.DIRECTIONS:
                            cr, mr = picks[candidate][a][d], picks[comparator][a][d]
                            mh = a if comparator == 'teacher' else 'all'
                            pid, m = load(text, entry, comparator, mh, 'x', a, d, mr, fr)
                            cs = []
                            for seed in P.SEEDS:
                                cp, c = load(text, entry, candidate, a, seed, a, d, cr, fr)
                                assert np.array_equal(cp, pid)
                                cs.append(c)
                            c = np.stack(cs)
                            mask = matched(c, m)
                            result = interval(c[:, mask], m[mask], pid[mask])
                            seed_range = float(rec['ranges'][kind][candidate][a][d])
                            cells.append(dict(set=entry, candidate=candidate, comparator=comparator, aspect=a,
                                              direction=d, kind=kind, candidate_readout=cr, comparator_readout=mr,
                                              seed_range=seed_range, outcome=outcome(result['lower'], seed_range), **result))
    # Preserve each frozen stage-2 MMArt label cell verbatim, under the E wrapper.
    frozen = json.loads((R23 / 'stage2/cells.json').read_text())
    e = [c for c in frozen if c['group'] == 'mm' and c['cell'] == 'label']
    assert len(e) == 36 and all(c['kind'] == 'narrative' for c in e)
    cells.extend(dict(set='E', **c) for c in e)
    counts = {}
    for s in sorted({c['set'] for c in cells}):
        counts[s] = {}
        for candidate in ('cstar', 'late'):
            for comparator in ('plain', 'teacher'):
                own = [c for c in cells if (c['set'], c['candidate'], c['comparator']) == (s, candidate, comparator)]
                assert len(own) == 9
                counts[s][candidate + '-' + comparator] = sum(c['outcome'] == 'WIN' for c in own)
    result = dict(smoke=smoke, draws=10000, counts=counts, cells=cells,
                  E_source=str(R23 / 'stage2/cells.json'))
    dest = (OUT / 'records' if smoke else ROUND / 'records') / 'c_cells.json'
    write(dest, result)
    print('| Set | c* − plain WIN | c* − teacher WIN | late − plain WIN | late − teacher WIN |')
    print('|---|---:|---:|---:|---:|')
    for s, values in counts.items():
        print('| ' + s + ' | ' + ' | '.join(str(n) for n in values.values()) + ' |')
    print('Genre t2t (difference [95% interval], word):')
    for s in counts:
        own = [c for c in cells if c['set'] == s and c['aspect'] == 'genre' and c['direction'] == 't2t']
        print(s + ': ' + '; '.join(f"{c['candidate']}-{c['comparator']} {c['point']:+.4f} [{c['lower']:+.4f}, {c['upper']:+.4f}] {c['outcome']}" for c in own))
    return result
