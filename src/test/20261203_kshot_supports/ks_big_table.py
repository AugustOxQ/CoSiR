"""Descriptive big table for the outside-view page (user's meeting, 2026-10-14): k = 4 only, development seed 42.

Scorers: cosine, RCA, B, B0 (= B'(A0)), B1 (= B'(A1)), CF, R1 (the gated reader without AFF's one-sided rule), AFF.
Metrics (percent, two directions averaged): R@1, other-aspect rate, condition gain = R@1 - other-aspect rate.
Scopes: all episodes; each aspect pair (both conditions); each told aspect (pooled over its two pairs, as per_aspect
in ks_by_condition.py). AFF - scorer with 95% cluster-bootstrap intervals for all and per told aspect.
Same frozen scorers and episodes as run_ks.py; CPU only. No decision rests on this.
"""
import json
from pathlib import Path

import ks_core as K
import numpy as np
from src.eval.aspect_metrics import cluster_bootstrap, first_place
from src.eval.aspect_quick_checks import uniform_probe_scores

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results/ks_big_table.json'
TXT = HERE / 'results/ks_big_table.txt'
SCORERS = ('cosine', 'RCA', 'B', 'B0', 'B1', 'CF', 'R1', 'AFF')
FREE = ('cosine', 'B', 'B0', 'B1', 'CF')
TARGET = {'a': 0, 'b': 1}
LABEL = {'B0': "B'(A0)", 'B1': "B'(A1)"}


def csd_post(env):
    """Round 4's csd posteriors (step1_heads_style.npz), as test_r6_bundle's reference builds them."""
    K.R.assert_input(K.H.CSD_HEADS_REL)
    sel = np.flatnonzero(np.isfinite(env.post['affect']['img'][:, 0]))
    out = {}
    with np.load(K.R.INPUT_PATHS[K.H.CSD_HEADS_REL]) as z:
        assert np.array_equal(z['selection'], sel)
        for m in ('img', 'txt'):
            a = np.asarray(z[f'style_csd__{m}'], dtype=np.float32)
            full = np.full((K.R.N_ROWS, a.shape[1]), np.nan, np.float32)
            full[sel] = a
            out[m] = full
    return out


def raw_scores(env, b):
    frozen = {n: K.S.frozen_nested(b.cos, b.t_n1u, getattr(b, t), env.picks[n], b.parity)
              for n, t in (('B', 't6u_B'), ('B0', 't6u_B0'), ('B1', 't6u_B1'))}
    rd = K.reader(b, env.readers)
    taus = K.R.R3.assert_taus()
    g_aff = K.R.RF.gates_aff(rd['m'], rd['pick'], taus)
    g_r1 = K.R.RF.gates_r1(rd['m'], taus)
    aff, cf = K.S.frozen_cells(frozen['B'], b.parity, rd['T'], g_aff, K.S.CELLS['aff']['fused'], K.S.CELLS['aff']['cf'])
    r1, _ = K.S.frozen_cells(frozen['B'], b.parity, rd['T'], g_r1, K.S.CELLS['r1']['fused'])
    rca = K.S.frozen_fused(b.cos, b.pm_terms['rca'], env.lambdas['rca'], b.parity)
    return {'cosine': b.cos, 'RCA': rca, **frozen, 'CF': cf, 'R1': r1, 'AFF': aff}


def hits(raw, c):
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
    b = K.build(env, 4)
    # B'(A1): csd posteriors enter the uniform-probe term (round 4's A1 order); it depends on the k = 4 supports
    post_a1 = {**env.post, 'csd': csd_post(env)}
    assert tuple(post_a1) == K.B.A1
    b.t6u_B1 = uniform_probe_scores(post_a1, b.pooled, K.B.A1)
    raw = raw_scores(env, b)
    H = {n: {c: hits(raw[n], c) for c in TARGET} for n in SCORERS}      # per episode: (R@1, other)
    cells_of = {}
    pairs = []
    for i, ep in enumerate(env.eps.per_pair):
        pairs.append((i, K.R.PAIR_NAMES[i]))
        for c, asp in (('a', ep.aspect_a), ('b', ep.aspect_b)):
            cells_of.setdefault(asp, []).append((i, c))
    scopes = {'all': [(i, c) for i, _ in pairs for c in TARGET]}
    scopes.update({name: [(i, c) for c in TARGET] for i, name in pairs})
    scopes.update({f'told_{a}': cells for a, cells in cells_of.items()})

    def vec(n, cells, j):
        return np.concatenate([H[n][c][j][pi == i] for i, c in cells])

    clus = {s: np.concatenate([env.cl[pi == i] for i, _ in cells]) for s, cells in scopes.items()}
    rec = {'purpose': 'descriptive table for the outside-view page; no decision', 'k': 4, 'seed': 42,
           'units': 'percent', 'scorers': {}, 'aff_minus': {}}
    for n in SCORERS:
        rec['scorers'][n] = {}
        for s, cells in scopes.items():
            r1, ot = 100 * vec(n, cells, 0).mean(), 100 * vec(n, cells, 1).mean()
            rec['scorers'][n][s] = {'r1': float(r1), 'other': float(ot), 'gain': float(r1 - ot)}
    for n in SCORERS:
        if n == 'AFF':
            continue
        rec['aff_minus'][n] = {s: ci(vec('AFF', scopes[s], 0) - vec(n, scopes[s], 0), clus[s])
                               for s in ('all', 'told_emotion', 'told_style', 'told_genre')}
    # ---- asserts
    ref = json.loads((HERE / 'results/kshot_seed42.json').read_text())['checks']['C1']['pooled_r1_targets']
    for n, v in ref.items():
        assert abs(rec['scorers'][n]['all']['r1'] - v) < 1e-9, (n, rec['scorers'][n]['all']['r1'], v)
    asserts = [f'pooled R@1 equals kshot_seed42.json C1 for {sorted(ref)} (1e-9)']
    for n in FREE:
        for s in ['all'] + [name for _, name in pairs]:
            assert abs(rec['scorers'][n][s]['gain']) < 1e-9, (n, s, rec['scorers'][n][s]['gain'])
    asserts.append(f'condition gain is 0 (pooled and per pair) for {list(FREE)}')
    bc = json.loads((HERE / 'results/ks_by_condition.json').read_text())['k']['4']['per_aspect']
    for a in ('emotion', 'style', 'genre'):
        for n in ('AFF', 'B0'):
            assert abs(rec['scorers'][n][f'told_{a}']['r1'] - bc[a][n]) < 1e-9, (a, n)
    asserts.append('per told aspect AFF and B0 R@1 equal ks_by_condition.json at k = 4 (1e-9)')
    # round 6 seed-42 references: the k = 4 prefix is the full episode, so B1 and R1 equal round 6's pooled values
    assert abs(rec['scorers']['B1']['all']['r1'] - 18.804931640625) < 1e-9, rec['scorers']['B1']['all']['r1']
    asserts.append("B'(A1) pooled R@1 = 18.804931640625 (r6_picks MEAN_R1_TARGETS B1)")
    assert abs(rec['scorers']['R1']['all']['r1'] - 18.918863932291664) < 1e-9, rec['scorers']['R1']['all']['r1']
    asserts.append('R1 pooled R@1 = 18.918863932291664 (regression_seed42.json r1_fused_r1)')
    rec['asserts_passed'] = asserts
    OUT.write_text(json.dumps(rec, indent=2) + '\n')
    # ---- text
    pn = [name for _, name in pairs]
    cols = ['all'] + pn + ['told_emotion', 'told_style', 'told_genre']
    f = lambda x: f'{x[0]:+.2f} [{x[1]:+.2f}, {x[2]:+.2f}]'
    L = ['k = 4 example pairs, development seed 42, percent. R@1 and gain = R@1 - other-aspect rate.',
         'Gain of a condition-free scorer (cosine, B, B0, B1, CF) is 0 for the all and pair scopes; in the told-aspect',
         'scopes it is NOT 0 and measures a lean toward one aspect, not use of the condition.', '',
         'Table 1: R@1 (gain in brackets)', f"{'scorer':8s}" + ''.join(f'{c:>22s}' for c in cols)]
    for n in SCORERS:
        L.append(f'{LABEL.get(n, n):8s}' + ''.join(
            f"{rec['scorers'][n][c]['r1']:>12.2f} ({rec['scorers'][n][c]['gain']:+7.2f})" for c in cols))
    L += ['', 'Table 2: AFF - scorer in R@1 points [95% cluster bootstrap]',
          f"{'scorer':8s}{'all':>26s}{'told emotion':>26s}{'told style':>26s}{'told genre':>26s}"]
    for n, r in rec['aff_minus'].items():
        L.append(f'{LABEL.get(n, n):8s}' + ''.join(f"{f(r[s]):>26s}" for s in ('all', 'told_emotion', 'told_style', 'told_genre')))
    L += ['', 'Other-aspect rate:', f"{'scorer':8s}" + ''.join(f'{c:>14s}' for c in cols)]
    for n in SCORERS:
        L.append(f'{LABEL.get(n, n):8s}' + ''.join(f"{rec['scorers'][n][c]['other']:>14.2f}" for c in cols))
    L += ['', 'Asserts passed:'] + ['- ' + a for a in asserts]
    TXT.write_text('\n'.join(L) + '\n')
    print('\n'.join(L), flush=True)


if __name__ == '__main__':
    main()
