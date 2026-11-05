"""Robustness of the nine-cell rule: fixed read-outs, cross-fit picks, emotion gap per class, RoBERTa sensitivity.

Reads only saved per-query AP; no ranking runs. Added after the first read of the outcome; checks only, the rule stays as written.
"""
import json
from pathlib import Path

import numpy as np

import tx_paths as paths
import tx_data as data
from tx_core import write_json
from tx_metrics import painting_bootstrap

CELLS = [(a, d) for a in paths.ASPECTS for d in ('t2i', 'i2t', 't2t')]
READOUTS = ('r0', 'center', 'center_k10')
OUT = paths.OUT
FREE = ('plain',)


def load(row, a, d, ro, entry=None):
    tag = 'all' if row in FREE else a
    with np.load(OUT / 'perquery' / f'{row}__{tag}__{d}__{ro}.npz', allow_pickle=False) as z:
        return z['ap__' + (entry or a)], z['q_pid']


def compare(row, a, d, get):
    c, pid = get(row, a, d)
    out = {}
    for other in ('plain', 'teacher'):
        b, p = get(other, a, d)
        assert np.array_equal(pid, p)
        mean, lo, hi = painting_bootstrap(c - b, pid, paths.N_PAINTINGS)
        out[other] = dict(mean=mean, lo=lo, hi=hi)
    return dict(aspect=a, direction=d, above=all(v['lo'] > 0 for v in out.values()), **out)


def fixed():
    return {ro: [compare('cstar', a, d, lambda r, x, y: load(r, x, y, ro)) for a, d in CELLS] for ro in READOUTS}


def crossfit_rows(row, a, d):
    """Per query, the AP at the read-out picked on the other half of the paintings (even and odd painting index)."""
    arrays = {ro: load(row, a, d, ro) for ro in READOUTS}
    pid = arrays['r0'][1]
    half = pid % 2 == 0
    picks, out = {}, np.full(len(pid), np.nan, dtype=np.float32)
    for name, fit, score in (('fit_even', half, ~half), ('fit_odd', ~half, half)):
        pick = max(READOUTS, key=lambda ro: np.nanmean(arrays[ro][0][fit]))
        out[score], picks[name] = arrays[pick][0][score], pick
    return out, pid, picks


def crossfit():
    cells = []
    for a, d in CELLS:
        rec = compare('cstar', a, d, lambda r, x, y: crossfit_rows(r, x, y)[:2])
        rec['picks'] = {r: crossfit_rows(r, a, d)[2] for r in ('cstar', 'plain', 'teacher')}
        cells.append(rec)
    return cells


def emotion_classes():
    names, labels = data.label_names('emotion'), data.caption_labels()['emotion']
    picks = {r['row'] + r['direction']: r['readout'] for r in json.loads(
        (paths.ROUND / 'results/summary.json').read_text())['picked'] if r['entry'] == 'emotion'}
    result = {}
    for d in ('t2i', 't2t'):
        rows = {r: load(r, 'emotion', d, picks[r + d])[0] for r in ('cstar', 'plain', 'teacher')}
        result[d] = {n: dict(n=int((labels == i).sum()), **{r: float(np.nanmean(v[labels == i])) for r, v in rows.items()})
                     for i, n in enumerate(names)}
    return result


def roberta():
    core = json.loads((OUT / 'results/core.json').read_text())
    return [{k: r[k] for k in ('entry', 'readout', 'map', 'macro_map', 'chance')}
            for r in core['records'] if r['row'] == 'roberta']


def main():
    doc = dict(fixed=fixed(), crossfit=crossfit(), emotion_classes=emotion_classes(), roberta=roberta())
    write_json(paths.ROUND / 'results/checks.json', doc)
    for ro, cells in doc['fixed'].items():
        print(ro, sum(c['above'] for c in cells), [f"{c['aspect'][0]}/{c['direction']}" for c in cells if c['above']])
    print('crossfit', sum(c['above'] for c in doc['crossfit']), [f"{c['aspect'][0]}/{c['direction']}" for c in doc['crossfit'] if c['above']])
    for c in doc['crossfit']:
        print(c['aspect'], c['direction'], f"plain {c['plain']['mean']:+.4f} [{c['plain']['lo']:+.4f}, {c['plain']['hi']:+.4f}]",
              f"teacher {c['teacher']['mean']:+.4f} [{c['teacher']['lo']:+.4f}, {c['teacher']['hi']:+.4f}]", c['picks']['cstar'])
    for ro in READOUTS:
        print(ro, 'g/t2t', {k: round(doc['fixed'][ro][8]['teacher'][k], 4) for k in ('mean', 'lo', 'hi')})
    for d, v in doc['emotion_classes'].items():
        print(d)
        for n, x in v.items():
            print(f"  {n:15s} {x['n']:6d} c* {x['cstar']:.4f} plain {x['plain']:.4f} teacher {x['teacher']:.4f}")
    for r in doc['roberta']:
        if r['entry'] != 'emotion__cross_painting':
            print(r)


if __name__ == '__main__':
    main()
