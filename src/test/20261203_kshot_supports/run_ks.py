"""Run the one CPU-only exploratory k-shot check; fail closed on C1/C2/C3."""
import json
import time
from pathlib import Path

START = time.perf_counter()
import ks_core as K
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results/kshot_seed42.json'
LOG = HERE / '20261203_kshot_supports_log.md'


def points(pa, mask=None):
    return {n: {m: 100 * float(np.asarray(pa[n][m])[mask].mean() if mask is not None else pa[n][m].mean())
                for m in ('r1', 'gain')} for n in K.NAMES}


def interval(v, cl):
    draws, _ = K.ST.bootstrap_draws(v, cl, n_boot=5000, seed=42)
    return K.ST.point_ci_from_draws(v, draws)


def c1(env, b, pa, frozen):
    base_rec = json.loads(K.R.INPUT_PATHS[K.B.BASELINES42_REL].read_text())
    with np.load(K.R.R3D / 'results/seed42_arrays.npz') as z:
        assert np.array_equal(z['cl'], env.cl) and np.array_equal(z['pair_index'], b.pair_index)
        assert np.array_equal(z['parity'], b.parity)
        for name, stored in (('AFF', 'aff_fused'), ('CF', 'aff_cf')):
            for m in pa[name]:
                assert np.array_equal(pa[name][m], z[f'{stored}__{m}']), f'C1 {name}/{m} differs from r6 regression source'
        cf_target = 100 * float(z['aff_cf__r1'].mean())
    with np.load(K.R.AB / 'results/per_anchor_seed42.npz') as z:
        assert np.array_equal(z['anchor_group'], env.cl) and np.array_equal(z['pair_index'], b.pair_index)
        for name, stored in (('cosine', 'cosine'), ('RCA', 'rca')):
            for m in pa[name]:
                assert np.array_equal(pa[name][m], z[f'{stored}__{m}']), f'C1 {name}/{m} differs from r6 regression source'
    targets = {'AFF': 19.136555989583336, 'B': 18.341064453125, 'B0': 18.436686197916664,
               'CF': cf_target, 'cosine': base_rec['scorers']['cosine']['overall']['r1']['point'],
               'RCA': base_rec['scorers']['rca']['overall']['r1']['point']}
    for name, want in targets.items():
        got = 100 * float(pa[name]['r1'].mean())
        assert got == want, f'C1 {name}: {got} != {want}'
    # Verification only, at k=4: original crossfit path and frozen-equals-crossfit
    # guard. The computed picks are compared with the frozen picks, never used.
    scores = {}
    for name in ('B', 'B0'):
        s, p = K.P.crossfit_condition_free(b.cos, b.t_n1u, getattr(b, K.P.TERM_OF[name]), b.parity)
        assert K.P.as_nested_pick(p, name) == env.picks[name], f'C1 {name} pick differs'
        scores[name] = s
    with K.patch.object(K.P, 'NESTED', ('B', 'B0')):
        equal = K.P.frozen_equals_crossfit(b, env.picks, scores)
    assert all(equal.values()), f'C1 frozen_equals_crossfit: {equal}'
    return {'passed': True, 'pooled_r1_targets': targets, 'frozen_equals_crossfit': equal,
            'AFF_CF_cosine_RCA_recorded_per_episode_metrics_exact': True,
            'reader_features_equal_original': True}


def fmt(v):
    return f"{v['point']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]"


def report(rec):
    lines = ['# K-shot supports: exploratory seed-42 check', '',
             'Purpose: describe frozen AFF and comparator sensitivity to fewer example pairs. These development numbers decide nothing.', '',
             'Design: the exact seed-42 12,288 development episodes (4,096 per pair), checked against round 6’s per-pair SHA-256s. '
             'Keep first k support and first k contrast pairs; queries, 13 candidates and targets stay identical. '
             'Condition b swaps the sets. Frozen tune-half h picks score parity 1−h; AFF cells 39/119 and CF cells 149/10. '
             'Frozen A0 readers, heads and A3 checkpoint; all four thresholds unchanged. '
             'R@1 averages both conditions and directions; condition gain is R@1 minus other-aspect retrieval, in percentage points.', '',
             'Run from `/project/CoSiR`:', '', '```bash',
             'PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= \\',
             '/root/miniconda3/envs/CoSiR/bin/python src/test/20261203_kshot_supports/run_ks.py \\',
             '> src/test/20261203_kshot_supports/results/ks_run.log 2>&1', '```', '',
             'C1 PASS: all six pooled k=4 R@1 values equal recorded round 6 values exactly; AFF/CF/cosine/RCA '
             'per-episode metrics and original reader features match exactly. B/B0 pass the original '
             '`r6_picks.frozen_equals_crossfit` check; verification picks equal the frozen picks.',
             'C2 PASS: cosine’s four raw condition/direction score arrays are identical at all k.',
             'C3 PASS: each episode and reader agreement input has exactly k pairs in each set; '
             'factor centring uses 2k items and RCA projection statistics use k pairs. Nested prefixes and unchanged targets/candidates are asserted.', '',
             'Pooled points (R@1 / condition gain, percentage points):', '',
             '| Scorer | k=2 | k=3 | k=4 |', '|---|---:|---:|---:|']
    for n in K.NAMES:
        cells = [f"{rec['k'][str(k)]['pooled'][n]['r1']:.3f} / {rec['k'][str(k)]['pooled'][n]['gain']:.3f}" for k in (2,3,4)]
        lines.append('| ' + n + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', 'Requested pooled differences; point [95% CI], percentage points. '
              '5,000 paired anchor-painting cluster resamples, seed 42, round 6 `r6_stats.bootstrap_draws`; '
              f"{rec['n_anchor_paintings']} clusters across all three pairs, shared draws for every difference. No decision test.", '',
              '| k | AFF−B0 R@1 | AFF−CF R@1 | AFF−CF gain |', '|---|---:|---:|---:|']
    for k in (2,3,4):
        dif = rec['k'][str(k)]['differences']
        lines.append(f"| {k} | {fmt(dif['AFF_minus_B0_r1'])} | {fmt(dif['AFF_minus_CF_r1'])} | {fmt(dif['AFF_minus_CF_gain'])} |")
    lines += ['', '| k−4 | AFF R@1 | AFF gain | B0 R@1 | B0 gain |', '|---|---:|---:|---:|---:|']
    for k in (2,3,4):
        dif = rec['k'][str(k)]['versus_k4']
        lines.append(f"| {k}−4 | " + ' | '.join(fmt(dif[n][m]) for n in ('AFF','B0') for m in ('r1','gain')) + ' |')
    lines += ['', 'AFF gate-open share (fractions):', '', '| k | τ0 a | τ0 b | τ2 a | τ2 b |', '|---|---:|---:|---:|---:|']
    for k in (2,3,4):
        g = rec['k'][str(k)]['AFF_gate_open_share']
        lines.append(f'| {k} | ' + ' | '.join(f'{g[t][c]:.6f}' for t in ('tau_0','tau_2') for c in ('a','b')) + ' |')
    for pair in K.R.PAIR_NAMES:
        lines += ['', f'Aspect pair `{pair}`; points only (R@1 / condition gain):', '', '| Scorer | k=2 | k=3 | k=4 |', '|---|---:|---:|---:|']
        for n in K.NAMES:
            cells = [f"{rec['k'][str(k)]['per_pair'][pair][n]['r1']:.3f} / {rec['k'][str(k)]['per_pair'][pair][n]['gain']:.3f}" for k in (2,3,4)]
            lines.append('| ' + n + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', f"Runtime: {rec['runtime_s']:.1f} seconds (CPU, eight threads), including input loading, k=4 validation, all k and bootstraps.", '',
              'Adapters: `ks_core.prefix` replaces the four-pair episode validator with exact-prefix/shape validation '
              'of the hash-verified, previously validated full episodes; `ks_core.features` implements the original '
              '18-feature arithmetic with a k-aware guard. Factor `_code_sides`/`centered_term` and RCA '
              '`_proj_pairs`/`_cov` already derive the count from array shapes and are imported unchanged. '
              'The bundle/scorer adapters follow the relevant round 6 calls and omit unrequested scorers.', '',
              'Deviation/interpretation: there is no saved RCA PCA basis. Its original deterministic PCA calculation is reconstructed '
              'once from the exact frozen 60,000-row scorer-train draw, unit normalisation and seed; the resulting basis is fixed '
              'across k. No head/reader/checkpoint/threshold or episode-based parameter is fitted or picked. '
              'The later saved affect classifier is checked against round 6’s coefficient hashes; other heads use the '
              'saved round 6 verified E2 posteriors. The k=4 grid run is solely the explicitly requested reproduction check. '
              'No held features or annotations were read; only train/selection feature values are selected from read-only memmaps. '
              'Git-dependent import path discovery is replaced in memory with `/project/CoSiR`.', '',
              'Terminology: the brief describes examples sharing the query’s value. The actual frozen episodes use '
              'value-disjoint examples that share the aspect with one another and exclude the anchor’s value; '
              'the required existing episodes are preserved.']
    return '\n'.join(lines) + '\n'


def main():
    checks = {}
    try:
        print('Loading frozen development inputs', flush=True)
        env = K.load()
        data, diag = {}, {}
        for k in (4, 3, 2):
            b = K.build(env, k)
            data[k], diag[k], frozen = K.score(env, b)
            if k == 4:
                checks['C1'] = c1(env, b, data[k], frozen)
                ref_cos = b.cos
                print('C1 PASS: k=4 exact reproduction', flush=True)
            else:
                assert all(np.array_equal(b.cos[c][d], ref_cos[c][d]) for c in K.CONDITIONS for d in K.DIRECTIONS), f'C2 k={k}'
            print(f'k={k} scored; pair dimensions checked', flush=True)
        checks.update(C2={'passed': True, 'raw_cosine_arrays_exact_all_k': True},
                      C3={'passed': True, 'pair_counts': [2, 3], 'factor_example_items': [4, 6]})
        rec = {'purpose': 'exploratory only; no decision', 'seed': 42, 'n_episodes': env.eps.n,
               'n_anchor_paintings': int(len(np.unique(env.cl))), 'episodes_sha256': env.eps.sha,
               'units': 'percentage points (gate shares are fractions)',
               'bootstrap': {'resamples': 5000, 'seed': 42, 'cluster': 'anchor painting', 'paired': True},
               'checks': checks, 'k': {}}
        for k in (2,3,4):
            pa = data[k]
            diffs = {f'AFF_minus_{n}_{m}': interval(pa['AFF'][m]-pa[n][m], env.cl)
                     for n,m in (('B0','r1'),('CF','r1'),('CF','gain'))}
            paired = {n: {m: interval(pa[n][m]-data[4][n][m], env.cl) for m in ('r1','gain')} for n in ('AFF','B0')}
            rec['k'][str(k)] = {'pooled': points(pa), 'per_pair': {p: points(pa, env.eps.pair_index==i)
                                for i,p in enumerate(K.R.PAIR_NAMES)}, 'differences': diffs,
                                'versus_k4': paired, 'AFF_gate_open_share': diag[k]}
        rec['runtime_s'] = time.perf_counter() - START
        OUT.write_text(json.dumps(rec, indent=2, allow_nan=False) + '\n')
        LOG.write_text(report(rec))
        print(f"C1/C2/C3 PASS; runtime {rec['runtime_s']:.1f}s; results: {OUT}", flush=True)
    except Exception as exc:
        elapsed = time.perf_counter() - START
        LOG.write_text(LOG.read_text() + f'\nStopped after {elapsed:.1f}s: {type(exc).__name__}: {exc}. '
                       f'Completed checks: {json.dumps(checks)}. No k-shot result file published.\n')
        raise


if __name__ == '__main__':
    main()
