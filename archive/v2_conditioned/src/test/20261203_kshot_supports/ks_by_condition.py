"""Follow-up (user, 2026-10-14): the k-shot results split by condition, i.e. by the aspect the example pairs show.

Condition a tells aspect A (target candidate 0), condition b tells aspect B (target candidate 1). R@1 per condition
averages the two directions. Per aspect pools the conditions that tell it, across the two pairs it occurs in.
Same frozen scorers and episodes as run_ks.py; CPU only; development seed 42.
"""
import json
from pathlib import Path

import ks_core as K
import numpy as np
from src.eval.aspect_metrics import cluster_bootstrap, first_place

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results/ks_by_condition.json'
TXT = HERE / 'results/ks_by_condition.txt'
SCORERS = ('AFF', 'B0', 'CF')
TARGET = {'a': 0, 'b': 1}


def raw_scores(env, b):
    """The scores of K.score, before per_anchor averages the conditions."""
    frozen = {n: K.S.frozen_nested(b.cos, b.t_n1u, getattr(b, K.P.TERM_OF[n]), env.picks[n], b.parity)
              for n in ('B', 'B0')}
    rd = K.reader(b, env.readers)
    gates = K.R.RF.gates_aff(rd['m'], rd['pick'], K.R.R3.assert_taus())
    aff, cf = K.S.frozen_cells(frozen['B'], b.parity, rd['T'], gates,
                               K.S.CELLS['aff']['fused'], K.S.CELLS['aff']['cf'])
    return {'B0': frozen['B0'], 'AFF': aff, 'CF': cf}


def hits(raw, c):
    """Per episode: R@1 under condition c, and the rate at which the other aspect's candidate wins."""
    t = TARGET[c]
    r1 = np.mean([first_place(raw[c][d], t) for d in K.DIRECTIONS], axis=0)
    other = np.mean([first_place(raw[c][d], 1 - t) for d in K.DIRECTIONS], axis=0)
    return r1, other


def ci(v, cl):
    r = cluster_bootstrap(v, cl)
    return [100 * r['point'], 100 * r['ci95'][0], 100 * r['ci95'][1]]


def main():
    env = K.load()
    pi = env.eps.pair_index
    aspects = {}
    for i, ep in enumerate(env.eps.per_pair):
        assert K.R.PAIR_NAMES[i] == f'{ep.aspect_a}__{ep.aspect_b}'
        for c, asp in (('a', ep.aspect_a), ('b', ep.aspect_b)):
            aspects.setdefault(asp, []).append((i, c))
    rec = {'purpose': 'exploratory follow-up; no decision', 'units': 'R@1 points', 'k': {}}
    for k in (4, 3, 2, 1):
        b = K.build(env, k)
        raw = raw_scores(env, b)
        h = {n: {c: hits(raw[n], c) for c in TARGET} for n in SCORERS}
        rk = {'per_pair_condition': {}, 'per_aspect': {}}
        for i, ep in enumerate(env.eps.per_pair):
            m = pi == i
            for c, asp in (('a', ep.aspect_a), ('b', ep.aspect_b)):
                rk['per_pair_condition'][f'{K.R.PAIR_NAMES[i]}|{asp}'] = {
                    **{n: 100 * float(h[n][c][0][m].mean()) for n in SCORERS},
                    'AFF_other': 100 * float(h['AFF'][c][1][m].mean()),
                    'AFF_minus_B0': ci(h['AFF'][c][0][m] - h['B0'][c][0][m], env.cl[m])}
        for asp, cells in aspects.items():
            v = {n: np.concatenate([h[n][c][0][pi == i] for i, c in cells]) for n in SCORERS}
            o = np.concatenate([h['AFF'][c][1][pi == i] for i, c in cells])
            cl = np.concatenate([env.cl[pi == i] for i, _ in cells])
            rk['per_aspect'][asp] = {**{n: 100 * float(v[n].mean()) for n in SCORERS},
                                     'AFF_other': 100 * float(o.mean()),
                                     'AFF_minus_B0': ci(v['AFF'] - v['B0'], cl),
                                     'AFF_minus_CF': ci(v['AFF'] - v['CF'], cl)}
        rec['k'][str(k)] = rk
        print(f'k={k} done', flush=True)
    # k = 4 must reproduce run_ks's pooled AFF R@1 (both conditions averaged)
    pooled = np.mean([v['AFF'] for v in rec['k']['4']['per_pair_condition'].values()])
    assert abs(pooled - 19.136555989583336) < 1e-9, pooled
    OUT.write_text(json.dumps(rec, indent=2) + '\n')
    f = lambda x: f'{x[0]:+.2f} [{x[1]:+.2f}, {x[2]:+.2f}]'
    lines = ['Per aspect told (pooled over its two pairs), R@1 points: AFF | B0 | AFF-B0 [95%] | AFF-CF [95%] | AFF other-aspect win']
    for k in ('4', '3', '2', '1'):
        for asp, r in rec['k'][k]['per_aspect'].items():
            lines.append(f"k={k} {asp:8s} AFF {r['AFF']:.2f} | B0 {r['B0']:.2f} | {f(r['AFF_minus_B0'])} | "
                         f"{f(r['AFF_minus_CF'])} | other {r['AFF_other']:.2f}")
    lines += ['', 'Per pair and told aspect: AFF | B0 | AFF-B0 [95%]']
    for k in ('4', '3', '2', '1'):
        for key, r in rec['k'][k]['per_pair_condition'].items():
            lines.append(f"k={k} {key:22s} AFF {r['AFF']:.2f} | B0 {r['B0']:.2f} | {f(r['AFF_minus_B0'])}")
    TXT.write_text('\n'.join(lines) + '\n')
    print(f'written {TXT}', flush=True)


if __name__ == '__main__':
    main()
