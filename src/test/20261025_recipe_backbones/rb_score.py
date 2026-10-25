"""Part 2 scorer: the CT rule §3 statistic for any candidate, entry list and row count (CPU).

Arms are named by ID, in rule order: the candidate first (K1 for the scorer check, K2 to K5), then B1 to B6, then
its matched arm (M2, M3, M5; none for K4). The candidate's reading is r20 only; every other arm has r20 and own.
Reuses the CT's partition code and AMI; the CT's own scorer hard-codes K1 and its arm order, so the statistic,
the draw order and the verdict files are generalized here (RB rule §3.2, §4.2).
"""
import hashlib
import inspect
import json
import multiprocessing
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import adjusted_mutual_info_score as ami
from threadpoolctl import threadpool_limits

import rb_paths
import apc_data
import ct_score
import pav_score

SEEDS = list(range(5))


def draw_bootstrap(n_rows, eligible, arms, n_resamples=2000, seedless=()):
    """The whole RNG transcript of one candidate: I_b, then S_b per arm with eligible seeds, in `arms` order.

    `seedless` arms (the frozen encoder image Fk, reported margins only) have one partition set: they draw nothing
    and enter every resample with the pseudo-seed 0."""
    assert len(set(arms)) == len(arms) and set(eligible) <= set(arms) and set(seedless) <= set(arms)
    assert not set(seedless) & set(eligible) and arms[0] not in seedless
    assert n_rows > 0 and n_resamples > 0
    assert all(list(ss) == sorted(set(ss)) for ss in eligible.values())
    rng = np.random.default_rng(0)
    draws = []
    for _ in range(n_resamples):
        rows = rng.integers(0, n_rows, size=n_rows)
        seeds = {arm: np.array([0]) if arm in seedless else
                 rng.choice(eligible[arm], size=len(eligible[arm]), replace=True)
                 for arm in arms if arm in seedless or len(eligible.get(arm, []))}
        draws.append((rows, seeds))
    return draws


def score_indices(fixed, labels, rows, seeds, arms):
    """A_cand, every A_e and d for one paired row/seed draw of fixed partitions.

    fixed[arm][seed] = {reading: int [3 or 1, n]}; seeds[arm] = the drawn seeds (repeats weigh); arms[0] is the
    candidate (r20 only), the others give entries `<arm>-<reading>`. Rows are indexed first, then masked.
    """
    assert len(arms) >= 2 and set(seeds) <= set(arms) and len(seeds.get(arms[0], ())) > 0
    y = labels[rows]
    mask = y >= 0
    per_seed, means = {}, {}
    for arm in arms:
        chosen = seeds.get(arm, ())
        if not len(chosen):
            continue
        per_seed[arm] = {
            int(seed): {reading: [float(ami(y[mask], p[rows][mask])) for p in ps]
                        for reading, ps in fixed[arm][seed].items()}
            for seed in np.unique(chosen)}
        means[arm] = {
            reading: float(np.mean([np.mean(per_seed[arm][int(s)][reading]) for s in chosen]))
            for reading in per_seed[arm][int(chosen[0])]}
    cand = means.pop(arms[0])
    assert set(cand) == {'r20'}
    entries = {f'{arm}-{reading}': value for arm, readings in means.items() for reading, value in readings.items()}
    assert entries
    return dict(d=cand['r20'] - max(entries.values()), A_cand=cand['r20'], A_e=entries, per_seed=per_seed)


def _init_worker(fixed, labels, arms):
    global _worker
    _worker = (fixed, labels, arms)
    threadpool_limits(limits=1)


def _draw_difference(draw):
    rows, seeds = draw
    fixed, labels, arms = _worker
    return score_indices(fixed, labels, rows, seeds, arms)['d']


def paired_bootstrap(fixed, labels, eligible, arms, n_resamples=2000, workers=8, seedless=()):
    """Point estimate (I_0 = 0..n-1, every eligible seed) and the percentile interval of one candidate.

    Draws come from `draw_bootstrap` before any AMI is scheduled, so the result does not depend on scheduling.
    """
    assert 1 <= workers <= 8
    assert labels.ndim == 1 and np.issubdtype(labels.dtype, np.integer)
    assert set(eligible) <= set(arms) and eligible.get(arms[0])
    all_seeds = dict(eligible, **{arm: [0] for arm in seedless})
    for arm, seeds in all_seeds.items():
        for seed in seeds:
            readings = fixed[arm][seed]
            assert set(readings) == ({'r20'} if arm == arms[0] or arm in seedless else {'r20', 'own'}), (arm, seed)
            for reading, ps in readings.items():
                assert ps.shape == (3 if reading == 'r20' else 1, len(labels)), (arm, seed, reading)
                assert np.issubdtype(ps.dtype, np.integer)
    draws = draw_bootstrap(len(labels), eligible, arms, n_resamples, seedless)
    point = score_indices(fixed, labels, np.arange(len(labels)), all_seeds, arms)
    if workers == 1:
        differences = [score_indices(fixed, labels, rows, seeds, arms)['d'] for rows, seeds in draws]
    else:
        with multiprocessing.Pool(workers, initializer=_init_worker, initargs=(fixed, labels, arms)) as pool:
            differences = pool.map(_draw_difference, draws)
    point.update(interval=np.percentile(differences, [2.5, 97.5]).tolist(), bootstrap_d=differences)
    return point


def outcome(d, lower, scale):
    """Rule §4.2 outcome ID on full-precision values: S1-* at the sample scale (stage 3), S2-* at full (stage 4)."""
    assert scale in ('sample', 'full') and np.isfinite(d) and np.isfinite(lower)
    kind = ('GO' if lower > 0 else 'MM') if d >= .01 else ('IN' if d > 0 else 'NO')
    return f'S{1 if scale == "sample" else 2}-{kind}'


CT_ARMS = ('K1', 'B1', 'B2', 'B3', 'B4', 'B5')   # the CT's list: the scorer checks use it


def arms_for(candidate):
    """Rule §2 arm order for a candidate: itself, B1 to B6, then its matched arm (K2, K3, K5; K4's is B1)."""
    assert candidate in ('K1', 'K2', 'K3', 'K4', 'K5')
    matched = {'K2': ('M2',), 'K3': ('M3',), 'K5': ('M5',)}.get(candidate, ())
    return (candidate, 'B1', 'B2', 'B3', 'B4', 'B5', 'B6') + matched


def eligible_seeds(arms, rcol_eligible, nonfinite=None):
    """Eligible seeds per arm: B2 to B5 from the R-COL record (minus non-finite paper runs, V4); every other
    arm (candidate, B1, B6, matched) all seeds 0 to 4."""
    assert set(rcol_eligible) == {'B2', 'B3', 'B4', 'B5'}
    nonfinite = nonfinite or {}
    assert set(nonfinite) <= set(rcol_eligible)
    assert all(ss == sorted(set(ss)) and set(ss) <= set(SEEDS) for ss in rcol_eligible.values())
    return {arm: [s for s in rcol_eligible[arm] if s not in nonfinite.get(arm, ())] if arm in rcol_eligible
            else SEEDS.copy() for arm in arms}


def read_rcol_eligible(scale, path=None):
    """(eligible seeds of B2..B5, record path) from the CT's frozen R-COL record of the scale."""
    path = Path(path) if path else {'sample': rb_paths.CT_OUT / 'percept/rcol_stage1.json',
                                    'full': rb_paths.CT_FULL / 'percept/rcol.json'}[scale]
    record, eligible, _ = ct_score.read_rcol(path)
    return eligible, path


HELD_ROWS = 12281
CT_ROOT = {'sample': rb_paths.CT_OUT, 'full': rb_paths.CT_FULL}
CT_STEMS = dict(K1='held/d2/{aspect}', B1='held/percept/core__k40/P3', B2='held/percept/paper__strict/P1',
                B3='held/percept/paper__strict/P4', B4='held/percept/paper__k40/P1',
                B5='held/percept/paper__k40/P4', B6='held/percept/core__k40/P2')
PAPER_ARMS = ('B2', 'B3', 'B4', 'B5')   # may be dropped when non-finite (V4); no other arm may be


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def output_path(scale, arm, seed, aspect='emotion', outputs=None):
    """Saved held output of an arm: the CT's files for K1 and B1 to B6, RB's part 2 outputs for K2.. and M.."""
    assert scale in CT_ROOT and seed in SEEDS
    if arm in CT_STEMS:
        return CT_ROOT[scale] / f'{CT_STEMS[arm].format(aspect=aspect)}__s{seed}.npz'
    outputs = Path(rb_paths.OUTPUTS if outputs is None else outputs)
    if arm in ('K2', 'K3', 'K4', 'K5'):
        return outputs / f'part2/{scale}/held/{arm}/{aspect}__s{seed}.npz'
    assert arm in ('M2', 'M3', 'M5')
    return outputs / f'part2/{scale}/held/{arm}/P3__s{seed}.npz'


def load_output(path, arm, n_rows=HELD_ROWS, held_sha256=None):
    """A saved output as float32, checked as V3 (rows, width, dtype, unit norm for K, [0, 1] for PercepT).

    Non-finite paper arms (B2 to B5) return None (V4: they leave the entry); any other arm fails.
    CT sidecars must carry the held list's hash when `held_sha256` is given.
    """
    path = Path(path)
    is_k = arm[0] == 'K'
    if held_sha256 is not None and arm in CT_STEMS:
        meta = json.loads(path.with_suffix('.json').read_text())
        assert meta['held_sha256'] == held_sha256 and meta['rows'] == n_rows, path
    with np.load(path, allow_pickle=False) as z:
        assert z.files == (['img'] if is_k else ['prob']), path
        x = z['img' if is_k else 'prob']
    assert x.ndim == 2 and x.shape[0] == n_rows, path
    if is_k:
        assert x.shape[1] == 2048 and x.dtype == np.float16, path
    else:
        assert x.dtype == np.float32, path
        if arm not in ('B2', 'B3'):
            assert x.shape[1] == 40, path
    x = x.astype(np.float32)
    if arm in PAPER_ARMS and not np.isfinite(x).all():
        return None
    assert np.isfinite(x).all(), path
    if is_k:
        assert np.all(np.abs(np.linalg.norm(x, axis=1) - 1) < 1e-3), path
    else:
        assert ((x >= 0) & (x <= 1)).all(), path
    return x


def _identity(n_rows, kind, held_sha256, source_sha256):
    code = {Path(m.__file__).name: sha_file(m.__file__) for m in (pav_score, apc_data)}
    code['output_partitions'] = hashlib.sha256(inspect.getsource(ct_score.output_partitions).encode()).hexdigest()
    return dict(held_sha256=held_sha256, source_sha256=source_sha256, kind=kind, rows=n_rows,
                sklearn=sklearn.__version__, code_sha256=code)


def fixed_partitions(x, arm, folder, aspect, seed, held_sha256, source_sha256):
    """Partitions of one output set, computed once on all rows then fixed (rule §3), saved as
    `<folder>/<aspect>__s{seed}__<r20|own>.npz` (key `labels`) with a sidecar binding source and code."""
    kind = 'k1' if arm[0] == 'K' else 'percept'
    readings = ('r20',) if kind == 'k1' else ('r20', 'own')
    folder = Path(folder)
    identity = _identity(len(x), kind, held_sha256, source_sha256)
    paths = {r: folder / f'{aspect}__s{seed}__{r}.npz' for r in readings}
    if all(p.exists() for p in paths.values()):
        result = {}
        for reading, path in paths.items():
            meta = json.loads(path.with_suffix('.json').read_text())
            assert meta['identity'] == identity, f'{path}: fixed partition inputs changed'
            assert meta['partitions_sha256'] == sha_file(path), path
            with np.load(path, allow_pickle=False) as z:
                result[reading] = z['labels']
    else:
        assert not any(p.exists() for p in paths.values()), f'{folder}: partial partition set'
        result = ct_score.output_partitions(x, kind)
        folder.mkdir(parents=True, exist_ok=True)
        for reading, path in paths.items():
            tmp = path.with_suffix('.pending.npz')
            np.savez_compressed(tmp, labels=result[reading])
            tmp.replace(path)
            path.with_suffix('.json').write_text(json.dumps(
                dict(identity=identity, reading=reading, partitions_sha256=sha_file(path)), indent=2) + '\n')
    assert set(result) == set(readings)
    for reading, ps in result.items():
        assert ps.shape == (3 if reading == 'r20' else 1, len(x)) and np.issubdtype(ps.dtype, np.integer)
    return result


CT_VERDICTS = {'sample': rb_paths.MAIN_TEST / '20261020_claim_test/verdict_stage1.json',
               'full': rb_paths.MAIN_TEST / '20261020_claim_test/verdict_stage2.json'}
CT_HELD_LIST = rb_paths.CT_OUT / 'held/held_paintings.json'


def held_setup():
    """(names, sha256, emotion labels) of the 12,281 held paintings; the list must be the CT's frozen one (V3)."""
    import ct_held
    names, sha256 = ct_held.read_held_list(CT_HELD_LIST)
    assert sha256 == rb_paths.HELD_SHA256 and ct_held.names_sha256(names) == sha256
    assert len(names) == HELD_ROWS and names == sorted(set(names))
    labels = ct_held.painting_labels(names)['emotion']
    assert labels.shape == (HELD_ROWS,) and np.issubdtype(labels.dtype, np.integer) and (labels >= 0).any()
    return names, sha256, labels


def ct_cached_partitions(scale, arms, eligible):
    """The CT's saved fixed partitions of every eligible arm and seed (K1 and B1 to B5 only)."""
    fixed = {}
    for arm in arms:
        fixed[arm] = {}
        for seed in eligible[arm]:
            with np.load(CT_ROOT[scale] / f'held/partitions/{arm}__s{seed}.npz', allow_pickle=False) as z:
                fixed[arm][seed] = {k: z[k] for k in z.files}
    return fixed


def prepare_arms(scale, arms, rcol_eligible, held_sha256, outputs=None, aspect='emotion', n_rows=HELD_ROWS):
    """Load and validate every needed output (V3), fix its partitions once; return (fixed, eligible, sources).

    The candidate (arms[0]) is read for `aspect`; the other arms are aspect-free PercepT outputs. A missing file of
    an eligible arm fails; a non-finite paper output leaves its entry (V4) and is listed in the eligibility.
    """
    outputs = Path(rb_paths.OUTPUTS if outputs is None else outputs)
    eligible = eligible_seeds(arms, rcol_eligible)
    nonfinite = {arm: [] for arm in PAPER_ARMS}
    fixed, sources = {}, {}
    for arm in arms:
        fixed[arm], sources[arm] = {}, {}
        folder = outputs / f'part2/partitions/{scale}/{arm}'
        for seed in eligible[arm]:
            path = output_path(scale, arm, seed, aspect if arm[0] == 'K' else 'emotion', outputs)
            assert path.exists(), f'missing eligible output: {path}'
            x = load_output(path, arm, n_rows, held_sha256)
            sources[arm][str(seed)] = dict(path=str(path), output_sha256=sha_file(path))
            if x is None:
                nonfinite[arm].append(seed)
                sources[arm][str(seed)]['status'] = 'nonfinite_held'
                continue
            fixed[arm][seed] = fixed_partitions(x, arm, folder, aspect if arm[0] == 'K' else 'percept', seed,
                                                held_sha256, sources[arm][str(seed)]['output_sha256'])
    nonfinite = {arm: ss for arm, ss in nonfinite.items() if ss}
    return fixed, eligible_seeds(arms, rcol_eligible, nonfinite), dict(outputs=sources, nonfinite_held=nonfinite)


def _json_value(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.pending.json')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def candidate_block(fixed, labels, eligible, arms, scale, n_resamples=2000, workers=8, eligibility=None,
                    sources=None):
    """One candidate's verdict block: point estimate, interval, entries, outcome (rule §3.2, §4.2).
    Its own `default_rng(0)` bootstrap over `arms` in rule order."""
    result = paired_bootstrap(fixed, labels, eligible, arms, n_resamples, workers)
    values, counts = np.unique(labels[labels >= 0], return_counts=True)
    return dict(
        candidate=arms[0], arms=list(arms), scale=scale, **result,
        eligibility=eligibility if eligibility is not None else dict(eligible_seeds=eligible),
        sources=sources or {},
        label_counts=dict(classes={str(v): int(n) for v, n in zip(values, counts)},
                          missing=int((labels < 0).sum())),
        outcome_id=outcome(result['d'], result['interval'][0], scale),
        bootstrap=dict(resamples=n_resamples, rng='numpy.random.default_rng(0) per candidate',
                       arm_order=list(arms), n_rows=len(labels), percentiles=[2.5, 97.5], method='linear'))


def write_verdict(path, header, blocks, replace=None):
    """Write blocks into a verdict file, merging by candidate or model name.

    The header must equal the file's. A block that differs from the saved one is refused unless `replace` gives a
    V-check reason (recorded with the file's old hash); identical content is left untouched.
    """
    assert replace is None or replace.strip(), 'replacement needs a non-empty V-check reason'
    header, blocks = _json_value(header), _json_value(blocks)
    path = Path(path)
    record = dict(schema=1, **header, blocks={})
    if path.exists():
        existing = json.loads(path.read_text())
        assert {k: v for k, v in existing.items() if k not in ('blocks', 'replacement')} == \
            {k: v for k, v in record.items() if k != 'blocks'}, f'{path}: header differs'
        record = existing
    changed = {name for name, block in blocks.items() if name in record['blocks'] and record['blocks'][name] != block}
    assert not changed or replace, f'{path}: refusing to replace {sorted(changed)}; give a V-check reason (replace)'
    if not changed and blocks.items() <= record['blocks'].items():
        return record
    if changed:
        record['replacement'] = dict(reason=replace, blocks=sorted(changed), previous_sha256=sha_file(path))
    record['blocks'].update(blocks)
    _atomic_json(path, record)
    return record


def write_regression_items(path, items):
    """Merge items {name: {'pass': bool, ...}} into a regression file {pass, items}; pass = every item passes."""
    assert all(isinstance(item.get('pass'), bool) for item in items.values())
    path = Path(path)
    record = json.loads(path.read_text()) if path.exists() else dict(items={})
    record['items'].update(_json_value(items))
    record['pass'] = all(item['pass'] for item in record['items'].values())
    _atomic_json(path, record)
    return record


TOLERANCE = 1e-6


def _item(ours, theirs):
    diff = float(abs(ours - theirs))
    return {'pass': bool(diff < TOLERANCE), 'ours': float(ours), 'ct': float(theirs), 'abs_diff': diff}


def compare_with_ct(result, ct):
    """Regression items of a scorer result against a CT verdict: each A_e, A_K1, d, both bounds within 1e-6, the
    entry sets equal, and every bootstrap d (a draw-order check)."""
    items = {'scorer/A_K1': _item(result['A_cand'], ct['A_K1']),
             'scorer/entries': {'pass': sorted(result['A_e']) == sorted(ct['A_e']),
                                'ours': sorted(result['A_e']), 'ct': sorted(ct['A_e'])}}
    for entry in sorted(ct['A_e']):
        items[f'scorer/A_e/{entry}'] = (_item(result['A_e'][entry], ct['A_e'][entry])
                                        if entry in result['A_e'] else {'pass': False, 'ours': None})
    items['scorer/d'] = _item(result['d'], ct['d'])
    items['scorer/interval_lo'] = _item(result['interval'][0], ct['interval'][0])
    items['scorer/interval_hi'] = _item(result['interval'][1], ct['interval'][1])
    ours, theirs = np.asarray(result['bootstrap_d']), np.asarray(ct['bootstrap_d'])
    same = ours.shape == theirs.shape
    items['scorer/bootstrap_d'] = {'pass': bool(same and np.abs(ours - theirs).max() < TOLERANCE),
                                   'n_ours': len(ours), 'n_ct': len(theirs),
                                   'max_abs_diff': float(np.abs(ours - theirs).max()) if same else None}
    return items


CHECKS = {'stage2': 'sample', 'stage3a': 'full'}   # spec §3.5: scorer checks against the CT's stage 1 / stage 2 verdicts
STAGES = {3: 'sample', 4: 'full'}


def _code_identity():
    import subprocess
    here = Path(__file__).resolve().parent
    return dict(rb_score_sha256=sha_file(__file__), commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=here, text=True).strip())


def run_check(stage, outputs=None, workers=8, n_resamples=2000):
    """Scorer check (spec §3.5 stage 2 / 3a): the CT's arm list on the CT's saved outputs at one scale must give
    `CT/verdict_stage1.json` (sample) or `verdict_stage2.json` (full): each A_e, d, both bounds within 1e-6.
    Partitions are recomputed here (or read back from this checkout's own validated cache) and also compared with
    the CT's fixed partitions. Items go into `part2/regression/<stage>.json`."""
    scale = CHECKS[stage]
    outputs = Path(rb_paths.OUTPUTS if outputs is None else outputs)
    ct = json.loads(CT_VERDICTS[scale].read_text())
    assert ct['held_sha256'] == rb_paths.HELD_SHA256 and len(ct['bootstrap_d']) == n_resamples == 2000
    names, sha, labels = held_setup()
    rcol, rcol_path = read_rcol_eligible(scale)
    fixed, eligible, sources = prepare_arms(scale, CT_ARMS, rcol, sha, outputs)
    assert eligible == ct['eligibility']['eligible_seeds']
    ct_fixed = ct_cached_partitions(scale, CT_ARMS, eligible)
    items = {}
    for arm in CT_ARMS:
        same = {seed: all(np.array_equal(fixed[arm][seed][r], ct_fixed[arm][seed][r]) for r in ct_fixed[arm][seed])
                for seed in eligible[arm]}
        if same:
            items[f'scorer/partitions/{arm}'] = {'pass': all(same.values()), 'equal': same}
    result = paired_bootstrap(fixed, labels, eligible, CT_ARMS, n_resamples, workers)
    items.update(compare_with_ct(result, ct))
    ours_outcome = outcome(result['d'], result['interval'][0], scale)
    items['scorer/outcome'] = {'pass': ours_outcome == ct['outcome_id'],
                               'ours': ours_outcome, 'ct': ct['outcome_id']}
    return write_regression_items(outputs / f'part2/regression/{stage}.json', items)


def check_v3_aspects(scale, candidate, outputs, held_sha256, n_rows=HELD_ROWS):
    """V3 on all three aspects of every seed of a candidate's output set (style and genre are only reported)."""
    for aspect in ('emotion', 'style', 'genre'):
        for seed in SEEDS:
            load_output(output_path(scale, candidate, seed, aspect, outputs), candidate, n_rows, held_sha256)


def require_gate(outputs, stage, candidates):
    """V1: stage 3 needs a passed stage 2, stage 4 a passed stage 3a and, per candidate, a stage 3 block not S1-NO."""
    reg = json.loads((outputs / f'part2/regression/{"stage2" if stage == 3 else "stage3a"}.json').read_text())
    assert reg['pass'] is True, f'V1: the regression check of stage {"2" if stage == 3 else "3a"} has not passed'
    if stage == 4:
        v3 = json.loads((outputs / 'part2/verdict_stage3.json').read_text())
        for c in candidates:
            assert c in v3['blocks'], f'{c}: no stage 3 outcome yet'
            assert v3['blocks'][c]['outcome_id'] != 'S1-NO', f'{c}: S1-NO, stage 4 does not run'


def run_stage(stage, candidates, outputs=None, workers=8, n_resamples=2000, gate=True, replace=None):
    """Stage 3 (sample) or 4 (full): each candidate's statistic, written to `part2/verdict_stage{3,4}.json`."""
    scale = STAGES[stage]
    outputs = Path(rb_paths.OUTPUTS if outputs is None else outputs)
    if gate:
        require_gate(outputs, stage, candidates)
    names, sha, labels = held_setup()
    rcol, rcol_path = read_rcol_eligible(scale)
    header = dict(stage=stage, scale=scale, held_sha256=sha)
    path = outputs / f'part2/verdict_stage{stage}.json'
    for candidate in candidates:
        arms = arms_for(candidate)
        check_v3_aspects(scale, candidate, outputs, sha, len(labels))
        fixed, eligible, sources = prepare_arms(scale, arms, rcol, sha, outputs, n_rows=len(labels))
        eligibility = dict(eligible_seeds=eligible, rcol_eligible_seeds=rcol, nonfinite_held=sources['nonfinite_held'])
        block = candidate_block(fixed, labels, eligible, arms, scale, n_resamples, workers, eligibility,
                                dict(held_outputs=sources['outputs'], rcol=dict(path=str(rcol_path), sha256=sha_file(rcol_path)),
                                     code=_code_identity()))
        record = write_verdict(path, header, {candidate: block}, replace)
        print(f'{candidate}: d={block["d"]:.4f} interval={block["interval"]} {block["outcome_id"]}', flush=True)
    return record


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--outputs', type=Path, default=rb_paths.OUTPUTS)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--workers', type=int, default=8, choices=range(1, 9))
    what = p.add_mutually_exclusive_group(required=True)
    what.add_argument('--check', choices=sorted(CHECKS), help='scorer check on the CT\'s saved outputs (2,000 draws)')
    what.add_argument('--stage', type=int, choices=sorted(STAGES), help='verdict of stage 3 (sample) or 4 (full)')
    p.add_argument('--candidates', nargs='+', choices=('K2', 'K3', 'K4', 'K5'))
    p.add_argument('--replace', metavar='REASON', help='V-check reason for replacing a changed block')
    args = p.parse_args(argv)
    if args.smoke:
        assert str(args.outputs.resolve()).startswith('/tmp/claude-0/'), 'smoke needs --outputs under /tmp/claude-0/'
        assert args.stage, 'a regression check is never shrunk: no --smoke with --check'
    if args.check:
        record = run_check(args.check, args.outputs, args.workers)
        bad = [k for k, v in record['items'].items() if not v['pass'] and k.startswith('scorer/')]
        print(f'{args.check}: pass={record["pass"]}; failing scorer items: {bad}', flush=True)
        raise SystemExit(0 if not bad else 1)
    assert args.candidates, '--stage needs --candidates'
    run_stage(args.stage, args.candidates, args.outputs, args.workers, n_resamples=20 if args.smoke else 2000,
              gate=not args.smoke, replace=args.replace)


if __name__ == '__main__':
    with threadpool_limits(limits=8):
        main()
