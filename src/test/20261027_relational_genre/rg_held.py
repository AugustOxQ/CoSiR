"""Stage C's held scorer (rule §3.2, §4.3; ticket 07). CPU.

`check vc1`  the scorer half of VC1: on 12-24's saved K4 genre outputs with K4 in G's place, this scorer gives
             `reported_stage4.json`'s K4 genre margins "T", "a_B1_to_B6" and "b_matched" (B1 in R's place) within 1e-6
             -> `stage_c/regression/vc1_scorer.json`.
`score --arm G`  G's held genre outputs (`full_root(out)/held/<G>/genre__s{0..4}.npz`) against T and R: A_G, A_T, A_R,
             d_T, d_R at I_0 and the full seed lists, the two bootstraps, G_half, PC and the outcome G-GO, G-MM,
             G-PART or G-NO -> `stage_c/verdict.json`. Refuses unless `stage_b/verdict.json` names G as v_held and
             both VC1 files pass. It states nothing else: the diagnostics (spec §3.6) run after the verdict exists.
`report --arm G`  stage C's diagnostics (spec §3.6; ticket 08), only after `stage_c/verdict.json` exists: G's emotion,
             style and genre on H with 12-24's reported margins (`rb_part2_report`), Leiden, seed spreads, best
             epochs, K4's frozen block beside G's -> `stage_c/reported.{json,md}`. Decides nothing.

The statistic is 12-24's: `rb_score.score_indices` (indexed rows, then the unlabelled rows masked inside the AMI; the
mean over the drawn seeds of the mean over a set's three r20 partitions). The draws are this module's own, written
as the rule's transcript and tested against `rb_score.draw_bootstrap`.
"""
import argparse
import hashlib
import json
import multiprocessing
import subprocess
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import rg_paths
import rg_variants
import rb_score
import rb_student
import rb_part2_report

SEEDS = list(rg_variants.FULL_SEEDS)
N_LABELLED = 9949                                   # rule §3.2: genre labels over H
N_RESAMPLES = 2000
UNIT_NORM_TOL = 1e-3                                # as `rb_score.load_output`
PARTITION_ARM = 'K'                                 # `rb_score.fixed_partitions` takes only arm[0] == 'K' (the r20-only kind)


# --- the draws (rule §3.2 "Intervals") ------------------------------------------------------------------------

def draws_d_t(n_rows, n_resamples=N_RESAMPLES, seeds=SEEDS):
    """d_T: `default_rng(0)`; per draw I_b = integers(0, n, size=n), then S_b = choice(seeds, size=5, replace=True) for
    G. T draws nothing: it enters every draw with its one partition set (pseudo-seed 0)."""
    rng = np.random.default_rng(0)
    draws = []
    for _ in range(n_resamples):
        rows = rng.integers(0, n_rows, size=n_rows)
        s_g = rng.choice(seeds, size=len(seeds), replace=True)
        draws.append((rows, {'G': s_g, 'T': np.array([0])}))
    return draws


def draws_d_r(n_rows, n_resamples=N_RESAMPLES, seeds=SEEDS):
    """d_R: its own `default_rng(0)`; per draw I_b, then S_b for G, then S_R,b for R, in that order."""
    rng = np.random.default_rng(0)
    draws = []
    for _ in range(n_resamples):
        rows = rng.integers(0, n_rows, size=n_rows)
        s_g = rng.choice(seeds, size=len(seeds), replace=True)
        s_r = rng.choice(seeds, size=len(seeds), replace=True)
        draws.append((rows, {'G': s_g, 'R': s_r}))
    return draws


# --- the statistics ---------------------------------------------------------------------------------------------

def _check_fixed(fixed, seeds, n_rows, what):
    assert sorted(fixed) == list(seeds), f'{what}: seeds {sorted(fixed)}, expected {list(seeds)}'
    for seed, readings in fixed.items():
        assert set(readings) == {'r20'}, (what, seed)
        p = readings['r20']
        assert p.shape == (3, n_rows) and np.issubdtype(p.dtype, np.integer), (what, seed, p.shape, p.dtype)


def _bootstrap(fixed, labels, arms, draws, workers):
    """Point estimate (I_0, every seed of the lists) and the 95% interval (`numpy.percentile` default) of
    A(arms[0]) - A(arms[1]) over the given draws; draws are made before any AMI runs."""
    assert 1 <= workers <= 8 and len(arms) == 2
    n = len(labels)
    seeds = {arm: sorted(fixed[arm]) for arm in arms}
    point = rb_score.score_indices(fixed, labels, np.arange(n), seeds, arms)
    if workers == 1:
        diffs = [rb_score.score_indices(fixed, labels, rows, s, arms)['d'] for rows, s in draws]
    else:
        with multiprocessing.Pool(workers, initializer=rb_score._init_worker, initargs=(fixed, labels, arms)) as pool:
            diffs = pool.map(rb_score._draw_difference, draws)
    other = arms[1]
    per_seed = {arm: {int(s): float(np.mean(v['r20'])) for s, v in point['per_seed'][arm].items()} for arm in arms}
    return dict(A_G=point['A_cand'], **{f'A_{other}': point['A_e'][f'{other}-r20']}, d=point['d'],
                interval=np.percentile(diffs, [2.5, 97.5]).tolist(), bootstrap_d=[float(d) for d in diffs],
                per_seed=per_seed)


def bootstrap_d_t(fixed_g, fixed_t, labels, n_resamples=N_RESAMPLES, workers=8):
    """d_T = A_G - A_T; fixed_t = {0: {'r20': three partitions}}."""
    n = len(labels)
    _check_fixed(fixed_g, SEEDS, n, 'G')
    _check_fixed(fixed_t, [0], n, 'T')
    return _bootstrap({'G': fixed_g, 'T': fixed_t}, labels, ('G', 'T'), draws_d_t(n, n_resamples), workers)


def bootstrap_d_r(fixed_g, fixed_r, labels, n_resamples=N_RESAMPLES, workers=8):
    """d_R = A_G - A_R; fixed_r = {seed: {'r20': ...}} over seeds 0 to 4."""
    n = len(labels)
    _check_fixed(fixed_g, SEEDS, n, 'G')
    _check_fixed(fixed_r, SEEDS, n, 'R')
    return _bootstrap({'G': fixed_g, 'R': fixed_r}, labels, ('G', 'R'), draws_d_r(n, n_resamples), workers)


def g_half(a_t, a_r):
    """Half the held gap: (A_T(I_0) - A_R(I_0, {0..4})) / 2."""
    return (a_t - a_r) / 2


def require_g_half(value):
    """The scorer's G_half must equal 12-24's frozen value within 1e-6, else refuse (rule §3.2)."""
    assert abs(value - rg_variants.G_HALF) <= rg_variants.TOL, \
        f'G_half {value!r} differs from the frozen {rg_variants.G_HALF!r} by more than {rg_variants.TOL}'
    return value


def statistics(fixed_g, fixed_t, fixed_r, labels, n_resamples=N_RESAMPLES, workers=8):
    """A_G, A_T, A_R, d_T, d_R, their intervals and G_half. labels: int [n], < 0 masked."""
    assert labels.ndim == 1 and np.issubdtype(labels.dtype, np.integer) and (labels >= 0).any()
    d_t = bootstrap_d_t(fixed_g, fixed_t, labels, n_resamples, workers)
    d_r = bootstrap_d_r(fixed_g, fixed_r, labels, n_resamples, workers)
    assert abs(d_t['A_G'] - d_r['A_G']) < 1e-12                      # the same G point estimate in both
    return dict(d_T=d_t, d_R=d_r, G_half=g_half(d_t['A_T'], d_r['A_R']), n_resamples=n_resamples)


# --- the outcome (rule §4.3) ------------------------------------------------------------------------------------

def outcome(d_t, lo_t, d_r, lo_r, g_half_value, x=rg_variants.X):
    """G-GO: lo_T > -X ("matches or exceeds" when lo_T > 0). G-MM: d_T >= -X and lo_T <= -X, reported with PC.
    G-PART: not GO, not MM, PC holds. G-NO otherwise. PC: d_R >= G_half and lo_R > 0. Unrounded values."""
    assert all(np.isfinite(v) for v in (d_t, lo_t, d_r, lo_r, g_half_value))
    pc = bool(d_r >= g_half_value and lo_r > 0)
    if lo_t > -x:
        kind = 'G-GO'
    elif d_t >= -x:
        kind = 'G-MM'
    elif pc:
        kind = 'G-PART'
    else:
        kind = 'G-NO'
    return dict(outcome_id=kind, matches_or_exceeds=bool(kind == 'G-GO' and lo_t > 0), pc=pc,
                pc_parts=dict(d_R_ge_G_half=bool(d_r >= g_half_value), lower_gt_0=bool(lo_r > 0)),
                partial_sentence=pc if kind in ('G-MM', 'G-PART') else None, X=x)   # G-PART is the partial sentence itself


# --- files ------------------------------------------------------------------------------------------------------

def sha_file(path):
    return rb_score.sha_file(path)


def code_provenance():
    """Provenance only: git HEAD of this checkout and whether the tree has uncommitted changes."""
    here = Path(__file__).resolve().parent
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=here, capture_output=True, text=True, check=True).stdout.strip()
    # this round's .py files, untracked ones included (a new module not yet committed counts as dirty)
    dirty = bool(subprocess.run(['git', 'status', '--porcelain', '--', '*.py'], cwd=here,
                                capture_output=True, text=True, check=True).stdout.strip())
    return dict(code_commit=head, dirty=dirty, rg_held_sha256=sha_file(__file__),
                rb_score_sha256=sha_file(rb_score.__file__))


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.pending.json')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def load_held_output(path, arm, seed, list_sha256, n_rows=rg_paths.N_HELD, width=rg_paths.WIDTH, aspect='genre'):
    """A held output as float32, VC3: key `img`, [n_rows, width] float16, finite, unit norm, in H's order (the
    sidecar binds H's hash, the arm, the seed, the aspect, the scale `full` and the array's SHA-256)."""
    path = Path(path)
    arrays, info = rb_student.read_outputs(path)
    assert list(arrays) == ['img'], path
    x = arrays['img']
    assert x.ndim == 2 and x.shape == (n_rows, width) and x.dtype == np.float16, (path, x.shape, x.dtype)
    expected = dict(kind='held', arm=arm, scale='full', aspect=aspect, seed=seed, rows=n_rows, width=width,
                    dtype='float16', list_sha256=list_sha256, sha256=rb_student.sha_array(x))
    for key, value in expected.items():
        assert info.get(key) == value, f'{path}: sidecar {key} is {info.get(key)!r}, expected {value!r}'
    x = x.astype(np.float32)
    assert np.isfinite(x).all(), f'{path}: non-finite values (VC3)'
    assert np.all(np.abs(np.linalg.norm(x, axis=1) - 1) < UNIT_NORM_TOL), f'{path}: rows are not unit norm (VC3)'
    return x


def held_output_source(path):
    """Provenance of a checked held output, including its VC3 attempts (frozen smoke fixtures predate VC3)."""
    info = _record(Path(path).with_suffix('.json'))
    return dict(path=str(path), output_sha256=sha_file(path), vc3_attempts=info.get('vc3_attempts', 1))


def read_frozen_partitions(path, n_rows=rg_paths.N_HELD, source_file=None, n_sets=3):
    """A frozen r20 partition file (key `labels`, int [3, n]) with its sidecar: the file's SHA-256, H's hash, the row
    count, and, when `source_file` is given, the SHA-256 of the output file the partitions came from. `n_sets` is 3
    for r20 and 1 for a PercepT arm's `own` reading."""
    path = Path(path)
    meta = json.loads(path.with_suffix('.json').read_text())
    assert meta['partitions_sha256'] == sha_file(path), f'{path}: partition file changed'
    ident = meta['identity']
    assert ident['held_sha256'] == rg_paths.HELD_SHA256 and ident['rows'] == n_rows, path
    if source_file is not None:
        assert ident['source_sha256'] == sha_file(source_file), f'{path}: source output changed'
    with np.load(path, allow_pickle=False) as z:
        assert z.files == ['labels'], path
        labels = z['labels']
    assert labels.shape == (n_sets, n_rows) and np.issubdtype(labels.dtype, np.integer), (path, labels.shape)
    return labels


def g_partitions(x, folder, seed, held_sha, source_sha):
    """r20 partitions of one output set, cached under `folder` as `genre__s{seed}__r20.npz` with 12-24's sidecar
    (`rb_score.fixed_partitions`; the 'K' name only selects the r20-only reading)."""
    return rb_score.fixed_partitions(x, PARTITION_ARM, folder, 'genre', seed, held_sha, source_sha)['r20']


# --- the gate ---------------------------------------------------------------------------------------------------

def _record(path):
    path = Path(path)
    assert path.is_file(), f'{path}: missing'
    return json.loads(path.read_text())


def require_vb2(out, smoke=False):
    """VB2 stops stage C unless its complete agreement covers the current B verdict bytes and mode."""
    out = Path(out)
    path = out / 'stage_b/rederive_vb2.json'
    assert path.is_file(), f'VB2: {path} missing'
    record = json.loads(path.read_text())
    assert record.get('agree') is True, f'VB2: {path} does not agree'
    assert record.get('complete') is True, f'VB2: {path} is incomplete'
    assert record.get('smoke') is smoke, f'VB2: {path} smoke mode differs'
    verdict = out / 'stage_b/verdict.json'
    assert _record(verdict).get('smoke') is smoke, f'VB2: {verdict} smoke mode differs'
    compared = [sha for name, sha in record.get('input_hashes', {}).items()
                if Path(name).resolve() == verdict.resolve()]
    assert compared and all(sha == sha_file(verdict) for sha in compared), f'VB2: {path} did not check the current B verdict'
    return sha_file(path)


def require_gate(out, arm):
    """`score --arm G`: `stage_b/verdict.json` is B-PICK naming G as v_held with G's switches, and both VC1 files
    pass, and VB2 agrees with the current B verdict. Returns the input files' SHA-256."""
    out = Path(out)
    assert arm in rg_variants.ARMS, f'{arm}: not an arm of this round'
    verdict = _record(out / 'stage_b/verdict.json')
    assert verdict.get('outcome') == 'B-PICK' and verdict.get('v_held') == arm, \
        f'stage_b/verdict.json: outcome {verdict.get("outcome")!r}, v_held {verdict.get("v_held")!r}; G is {arm!r}'
    assert verdict.get('v_held_switches') == rg_variants.switches(arm), 'stage_b/verdict.json: switches differ'
    shas = dict(stage_b_verdict=sha_file(out / 'stage_b/verdict.json'))
    shas['rederive_vb2'] = require_vb2(out)
    for name in ('vc1_train', 'vc1_scorer'):
        path = out / f'stage_c/regression/{name}.json'
        assert _record(path).get('pass') is True, f'VC1: {path} does not pass'
        shas[name] = sha_file(path)
    return shas


def vc1_scorer_is_current(record):
    """The scorer half of VC1 counts only for this scorer: its recorded code hashes equal the current ones."""
    ident = record.get('identity', {})
    now = code_provenance()
    return ident.get('rg_held_sha256') == now['rg_held_sha256'] and ident.get('rb_score_sha256') == now['rb_score_sha256']


# --- check vc1 --------------------------------------------------------------------------------------------------

def _close(ours, theirs):
    diff = float(abs(ours - theirs))
    return {'pass': bool(diff < rg_variants.TOL), 'ours': float(ours), 'reported': float(theirs), 'abs_diff': diff}


def _margin_items(prefix, ours, reported):
    """Items of one reported margin: d, both bounds, A_cand and every entry, each within 1e-6."""
    items = {f'{prefix}/d': _close(ours['d'], reported['d']),
             f'{prefix}/interval_lo': _close(ours['interval'][0], reported['interval'][0]),
             f'{prefix}/interval_hi': _close(ours['interval'][1], reported['interval'][1]),
             f'{prefix}/A_cand': _close(ours['A_cand'], reported['A_cand'])}
    assert sorted(ours['A_e']) == sorted(reported['A_e']), (prefix, sorted(ours['A_e']), sorted(reported['A_e']))
    for entry, value in reported['A_e'].items():
        items[f'{prefix}/A_e/{entry}'] = _close(ours['A_e'][entry], value)
    return items


def _frozen_r20(arm, key, seeds, source_files=None):
    """{seed: {'r20': ...}} read with `read_frozen_partitions` (sidecar checks; source files when given)."""
    root = rg_paths.RB_PART2 / f'partitions/full/{arm}'
    return {s: {'r20': read_frozen_partitions(root / f'{key}__s{s}__r20.npz',
                                              source_file=None if source_files is None else source_files[s])}
            for s in seeds}


def held_setup_genre():
    """(names, H's SHA-256, genre labels) with 9,949 labelled paintings; H must be the CT's frozen list."""
    names, sha, labels = rb_part2_report.held_labels()
    assert sha == rg_paths.HELD_SHA256 and len(names) == rg_paths.N_HELD
    y = labels['genre']
    assert y.shape == (rg_paths.N_HELD,) and np.issubdtype(y.dtype, np.integer)
    assert int((y >= 0).sum()) == N_LABELLED, f'{int((y >= 0).sum())} labelled genre rows, expected {N_LABELLED}'
    return names, sha, y


def r_sources():
    """12-24's saved K4 genre outputs (R, and G's place in the scorer check): {seed: path}."""
    return {s: rg_paths.RB_PART2 / f'full/held/K4/genre__s{s}.npz' for s in SEEDS}


def run_check_vc1(out, workers=8, n_resamples=N_RESAMPLES, force=False):
    """The scorer check of §3.2. Writes `stage_c/regression/vc1_scorer.json` ({pass, items, identity})."""
    out = Path(out)
    path = out / 'stage_c/regression/vc1_scorer.json'
    if path.exists() and not force:
        record = json.loads(path.read_text())
        if record.get('pass') is True and vc1_scorer_is_current(record):
            print(f'{path}: exists, passes and matches this code; nothing recomputed')
            return record
    assert n_resamples == N_RESAMPLES, 'a regression check is never shrunk'
    names, sha, y = held_setup_genre()
    reported = json.loads((rg_paths.RB_PART2 / 'reported_stage4.json').read_text())
    assert reported['held_sha256'] == sha
    block = reported['blocks']['K4']['aspects']['genre']['margins']
    sources = r_sources()
    items = {}
    # K4 in G's place: partitions recomputed from the saved outputs here, compared with the frozen ones
    k4_frozen = _frozen_r20('K4', 'genre', SEEDS, sources)
    fixed_g = {}
    for s in SEEDS:
        x = load_held_output(sources[s], 'K4', s, sha)
        fixed_g[s] = {'r20': g_partitions(x, out / 'stage_c/regression/vc1_partitions/K4', s, sha, sha_file(sources[s]))}
    items['partitions/K4_recomputed_equals_frozen'] = {
        'pass': all(np.array_equal(fixed_g[s]['r20'], k4_frozen[s]['r20']) for s in SEEDS)}
    t_frozen = _frozen_r20('T', 'genre', [0])
    # margin "T": d_T with K4 in G's place
    d_t = bootstrap_d_t(fixed_g, t_frozen, y, n_resamples, workers)
    items.update(_margin_items('T', dict(d=d_t['d'], interval=d_t['interval'], A_cand=d_t['A_G'],
                                         A_e={'T-r20': d_t['A_T']}), block['T']))
    # G_half from the frozen files (the same K4 partitions as R)
    a_r = rb_score.score_indices({'G': fixed_g, 'R': k4_frozen}, y, np.arange(len(y)), {'G': SEEDS, 'R': SEEDS},
                                 ('G', 'R'))['A_e']['R-r20']
    gh = g_half(d_t['A_T'], a_r)
    items['G_half'] = _close(gh, rg_variants.G_HALF)
    items['G_half_from_reported_stage4'] = _close(
        gh, (block['T']['A_e']['T-r20'] - block['T']['A_cand']) / 2)
    # margin "b_matched": K4 against B1 with B1 in R's place (exercises d_R's draw order)
    b1 = _frozen_r20('B1', 'percept', SEEDS)
    d_b1 = bootstrap_d_r(fixed_g, b1, y, n_resamples, workers)
    mb = block['b_matched']
    assert mb['arms'] == ['K4', 'B1'] and mb['max_entry'] == 'B1-r20', 'b_matched: d is against B1 r20 only'
    items.update(_margin_items('b_matched', dict(d=d_b1['d'], interval=d_b1['interval'], A_cand=d_b1['A_G'],
                                                 A_e={'B1-r20': d_b1['A_R']}),
                               dict(mb, A_e={'B1-r20': mb['A_e']['B1-r20']})))
    # margin "a_B1_to_B6": 12-24's own bootstrap on the frozen partitions (arms with eligible seeds only)
    arms = rb_part2_report.margin_arms('K4')['a']
    rcol, _ = rb_score.read_rcol_eligible('full')
    eligible = {a: list(v) for a, v in rb_score.eligible_seeds(arms, rcol).items()}
    ma = block['a_B1_to_B6']
    assert eligible == {a: v for a, v in reported['blocks']['K4']['eligible_seeds'].items()}, 'eligible seeds differ'
    fixed = {'K4': fixed_g}
    for arm in arms[1:]:
        fixed[arm] = {s: {r: read_frozen_partitions(rg_paths.RB_PART2 / f'partitions/full/{arm}/percept__s{s}__{r}.npz',
                                                                n_sets=3 if r == 'r20' else 1)
                          for r in ('r20', 'own')} for s in eligible[arm]}
    res = rb_score.paired_bootstrap(fixed, y, eligible, arms, n_resamples, workers)
    items.update(_margin_items('a_B1_to_B6', res, ma))
    record = dict(items=items,
                  identity=dict(held_sha256=sha, n_resamples=n_resamples, **code_provenance()),
                  reported_stage4_sha256=sha_file(rg_paths.RB_PART2 / 'reported_stage4.json'))
    record['pass'] = all(item['pass'] for item in items.values())
    record['failing'] = sorted(k for k, v in items.items() if not v['pass'])
    _atomic_json(path, record)
    print(f'check vc1: pass={record["pass"]}; failing: {record["failing"]}', flush=True)
    return record


# --- score ------------------------------------------------------------------------------------------------------

def run_score(out, arm, workers=8, n_resamples=N_RESAMPLES, smoke=False):
    """G's held outcome. Writes `stage_c/verdict.json` once; a second run must reproduce it (no overwrite)."""
    out = Path(out)
    if smoke:
        assert rg_paths.smoke_ok(out), 'smoke needs --outputs under /tmp/claude-0/'
        gate = {}
    else:
        assert n_resamples == N_RESAMPLES
        gate = require_gate(out, arm)
        assert vc1_scorer_is_current(_record(out / 'stage_c/regression/vc1_scorer.json')), \
            'VC1: vc1_scorer.json was written by other code; rerun `check vc1`'
    names, sha, y = held_setup_genre()
    root = rg_paths.full_root(out) / 'held' / arm
    g_files = {s: root / f'genre__s{s}.npz' for s in SEEDS}
    fixed_g, g_sources = {}, {}
    for s, file in g_files.items():
        assert file.is_file(), f'missing held output of G: {file}'
        x = load_held_output(file, arm, s, sha)                                  # VC3
        g_sources[str(s)] = held_output_source(file)
        fixed_g[s] = {'r20': g_partitions(x, out / 'stage_c/partitions' / arm, s, sha, g_sources[str(s)]['output_sha256'])}
    r_files = r_sources()
    fixed_r = _frozen_r20('K4', 'genre', SEEDS, r_files)
    fixed_t = _frozen_r20('T', 'genre', [0])
    stats = statistics(fixed_g, fixed_t, fixed_r, y, n_resamples, workers)
    gh = require_g_half(stats['G_half'])
    d_t, d_r = stats['d_T'], stats['d_R']
    result = outcome(d_t['d'], d_t['interval'][0], d_r['d'], d_r['interval'][0], gh)
    record = dict(
        schema=1, stage='C', arm=arm, held_sha256=sha, rule_commit=rg_paths.RULE_COMMIT, smoke=bool(smoke),
        **result, G_half=gh, G_half_frozen=rg_variants.G_HALF, labelled_rows=int((y >= 0).sum()),
        d_T=d_t, d_R=d_r,
        bootstrap=dict(resamples=n_resamples, rng='numpy.random.default_rng(0) per statistic',
                       d_T_draws='I_b, S_b (G); T draws nothing', d_R_draws='I_b, S_b (G), S_R,b (R)',
                       n_rows=len(y), percentiles=[2.5, 97.5], method='linear'),
        sources=dict(
            G_outputs=g_sources, R_outputs={str(s): dict(path=str(p), output_sha256=sha_file(p)) for s, p in r_files.items()},
            T_partitions=str(rg_paths.RB_PART2 / 'partitions/full/T/genre__s0__r20.npz'),
            R_partitions=str(rg_paths.RB_PART2 / 'partitions/full/K4'), gate_files=gate),
        code=code_provenance())
    path = out / 'stage_c/verdict.json'
    keep = lambda r: {k: v for k, v in r.items() if k != 'code'}
    if path.exists():
        assert keep(json.loads(path.read_text())) == keep(json.loads(json.dumps(record))), \
            f'{path}: exists and differs from this run; a verdict is not overwritten (VC4)'
    else:
        _atomic_json(path, record)
    print(f'{arm}: d_T {d_t["d"]:+.4f} [{d_t["interval"][0]:+.4f}, {d_t["interval"][1]:+.4f}]; '
          f'd_R {d_r["d"]:+.4f} [{d_r["interval"][0]:+.4f}, {d_r["interval"][1]:+.4f}]; G_half {gh:.4f}; '
          f'PC {result["pc"]}; {result["outcome_id"]}', flush=True)
    return record


# --- report (spec §3.6 "Stage C"; ticket 08; reported only, decides nothing) ---------------------------------------

COMPARATORS = rb_part2_report.COMPARATORS           # B1 to B6
FROZEN_IMAGE = 'F4'                                 # K4's default Qwen image (rb_part2_report.FROZEN['K4'])


def _frozen_comparator_partitions(eligible):
    """B1 to B6's frozen part 2 partitions {arm: {seed: {'r20', 'own'}}} for each arm's eligible seeds (the same files
    `check vc1` reads); an arm with no eligible seed has none."""
    return {arm: {s: {r: read_frozen_partitions(rg_paths.RB_PART2 / f'partitions/full/{arm}/percept__s{s}__{r}.npz',
                                                n_sets=3 if r == 'r20' else 1) for r in ('r20', 'own')}
                  for s in eligible[arm]} for arm in COMPARATORS}


def _comparator_sources(eligible):
    """{arm: {seed: {path, output_sha256}}}: the CT's saved B1 to B6 outputs the partitions came from."""
    return {arm: {str(s): dict(path=str(rb_score.output_path('full', arm, s)),
                               output_sha256=sha_file(rb_score.output_path('full', arm, s))) for s in eligible[arm]}
            for arm in COMPARATORS}


def _leiden_identity(held_sha, source_sha):
    import ct_diag
    return dict(held_sha256=held_sha, source_sha256=source_sha, k=rb_part2_report.LEIDEN_K,
                code_sha256=rb_score.sha_file(ct_diag.__file__))


def _read_leiden(path, identity, n_rows):
    path = Path(path)
    meta = json.loads(path.with_suffix('.json').read_text())
    assert meta['identity'] == identity and meta['partitions_sha256'] == sha_file(path), f'{path}: Leiden inputs changed'
    with np.load(path, allow_pickle=False) as z:
        labels = z['labels']
    assert labels.shape == (3, n_rows) and np.issubdtype(labels.dtype, np.integer), path
    return labels


def _leiden_g(x, path, identity):
    """G's Leiden partitions of one output set, cached with a sidecar under the round's own outputs (a copy of the
    caching branch of `rb_part2_report.leiden_fixed`, whose loader reads only 12-24's arm names)."""
    path = Path(path)
    if path.exists():
        return _read_leiden(path, identity, len(x))
    labels = rb_part2_report.leiden_reading(x)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.pending.npz')
    np.savez_compressed(tmp, labels=labels)
    tmp.replace(path)
    path.with_suffix('.json').write_text(json.dumps(dict(identity=identity, partitions_sha256=sha_file(path)),
                                                    indent=2) + '\n')
    return labels


def _leiden_comparators(eligible, sources, sha, n_rows):
    """B1 to B6's frozen Leiden partitions (12-24's `part2/leiden/full/<arm>/percept__s{seed}.npz`), read only."""
    return {arm: {s: dict(r20=_read_leiden(
        rg_paths.RB_PART2 / f'leiden/full/{arm}/percept__s{s}.npz',
        _leiden_identity(sha, sources[arm][str(s)]['output_sha256']), n_rows)) for s in eligible[arm]}
        for arm in COMPARATORS}


def best_epochs_g(out, arm):
    """Best epoch and best dev loss per aspect and seed from G's full-scale weight sidecars."""
    root = rg_paths.full_root(out) / 'weights' / arm
    result = {}
    for aspect in rg_paths.ASPECTS:
        result[aspect] = {}
        for s in SEEDS:
            meta = json.loads((root / f'{aspect}__s{s}.json').read_text())
            result[aspect][str(s)] = dict(best_epoch=int(meta['best_epoch']), best_dev_loss=float(meta['best_dev_loss']))
    return result


def report_g(out, arm, ctx):
    """The reported block of G: for each aspect the margins of 12-24's K4 block, Leiden, seed spreads, best epochs."""
    sha, n_rows, nr, workers = ctx['sha'], ctx['n_rows'], ctx['n_resamples'], ctx['workers']
    labels = ctx['labels']
    eligible = dict(ctx['eligible'], **{arm: list(SEEDS)})
    groups = rb_part2_report.margin_arms('K4')                 # K4's lists with G in the candidate's place
    arms_a = (arm,) + groups['a'][1:]
    arms_b = (arm,) + groups['b'][1:]
    assert arms_a == (arm,) + COMPARATORS and arms_b == (arm, 'B1') and groups['ab'][1:] == groups['a'][1:]
    f4 = {0: {'r20': read_frozen_partitions(rg_paths.RB_PART2 / f'partitions/full/{FROZEN_IMAGE}/frozen__s0__r20.npz')}}
    everything = np.arange(n_rows)
    root = rg_paths.full_root(out) / 'held' / arm
    report = dict(candidate=arm, arms=list(arms_a), eligible_seeds=eligible, frozen_arm=FROZEN_IMAGE, aspects={},
                  best_epochs=best_epochs_g(out, arm), sources=dict(G_outputs={}, comparators=ctx['comp_sources']))
    for aspect in rg_paths.ASPECTS:
        g_fixed, g_leiden, src = {}, {}, {}
        for s in SEEDS:
            file = root / f'{aspect}__s{s}.npz'
            assert file.is_file(), f'missing held output of G: {file}'
            x = load_held_output(file, arm, s, sha, aspect=aspect)                 # VC3
            src[str(s)] = held_output_source(file)
            g_fixed[s] = {'r20': rb_score.fixed_partitions(                       # 'K': the r20-only kind, as g_partitions
                x, PARTITION_ARM, out / 'stage_c/partitions' / arm, aspect, s, sha, src[str(s)]['output_sha256'])['r20']}
            g_leiden[s] = {'r20': _leiden_g(x, out / f'stage_c/leiden/{arm}/{aspect}__s{s}.npz', _leiden_identity(
                sha, src[str(s)]['output_sha256']))}
        report['sources']['G_outputs'][aspect] = src
        y = labels[aspect]
        t = read_frozen_partitions(rg_paths.RB_PART2 / f'partitions/full/T/{aspect}__s0__r20.npz')
        fx = dict(ctx['comp_fixed'], **{arm: g_fixed, FROZEN_IMAGE: f4, 'T': {0: {'r20': t}}})
        res = dict(labels_masked=int((y < 0).sum()), margins={},
                   point_spread=rb_part2_report.seed_spread(rb_score.score_indices(fx, y, everything, eligible, arms_a)))
        m = res['margins']
        if aspect != 'emotion':                                 # emotion: against T and F4 only (spec §3.6)
            m['a_B1_to_B6'] = rb_part2_report.margin(fx, y, eligible, arms_a, nr, workers)
            m['b_matched'] = rb_part2_report.margin(fx, y, eligible, arms_b, nr, workers)
            m['max_a_b'] = rb_part2_report.margin(fx, y, eligible, arms_a, nr, workers)   # K4's 'ab' list = 'a'
        m['c_frozen'] = rb_part2_report.margin(fx, y, eligible, (arm, FROZEN_IMAGE), nr, workers, seedless=(FROZEN_IMAGE,))
        m['T'] = rb_part2_report.margin(fx, y, eligible, (arm, 'T'), nr, workers, seedless=('T',))
        lf = dict(ctx['comp_leiden'], **{arm: g_leiden})
        m['leiden_vs_all'] = rb_part2_report.margin(lf, y, eligible, arms_a, nr, workers, lenient=True)
        report['aspects'][aspect] = res
        print(f'{arm} {aspect}: ' + ', '.join(f'{k} d={v["d"]:+.4f}' for k, v in m.items()), flush=True)
    return report


def _k4_beside(reported_stage4, sha):
    """12-24's frozen K4 block (full scale), cut to what the comparison shows."""
    assert reported_stage4['held_sha256'] == sha
    block = reported_stage4['blocks']['K4']
    return dict(source=str(rg_paths.RB_PART2 / 'reported_stage4.json'), best_epochs=block['best_epochs'],
                aspects={a: dict(margins=block['aspects'][a]['margins'], point_spread=block['aspects'][a]['point_spread'])
                         for a in rg_paths.ASPECTS})


def _fmt(m):
    lo, hi = m['interval']
    return f'{m["d"]:+.4f} [{lo:+.4f}, {hi:+.4f}]'


def report_markdown(record):
    g, k4, arm = record['G'], record['K4_frozen'], record['arm']
    v = record['verdict_summary']
    L = [f'# Stage C reported readings: {arm} on H' + (' (smoke)' if record['smoke'] else ''), '',
         'Decides nothing. Margins are 12-24\'s (`rb_part2_report.margin`): d = the score of the student minus the '
         'largest comparator entry, with the 95% interval of its own bootstrap over the arms named. K4 is 12-24\'s '
         'frozen block, for comparison.', '',
         f'Verdict (stage_c/verdict.json): {v["outcome_id"]}; d_T {v["d_T"]:+.4f}, d_R {v["d_R"]:+.4f}, '
         f'G_half {v["G_half"]:.4f}.', '',
         '| aspect | margin | G d [interval] | G score | G largest entry | K4 d [interval] | K4 score |',
         '|---|---|---|---|---|---|---|']
    for aspect, res in g['aspects'].items():
        for name, m in res['margins'].items():
            k = k4['aspects'][aspect]['margins'].get(name)
            L.append(f'| {aspect} | {name} | {_fmt(m)} | {m["A_cand"]:.4f} | {m["max_entry"]} '
                     f'{m["A_e"][m["max_entry"]]:.4f} | ' + (f'{_fmt(k)} | {k["A_cand"]:.4f} |' if k else ' | |'))
    L += ['', 'Seed spread of each aspect\'s score (G, then K4; mean, min, max, sd over seeds):', '',
          '| aspect | student | mean | min | max | sd |', '|---|---|---|---|---|---|']
    for aspect in g['aspects']:
        for who, spread, name in (('G', g['aspects'][aspect]['point_spread'], arm),
                                  ('K4', k4['aspects'][aspect]['point_spread'], 'K4')):
            s = spread[name]['r20']
            L.append(f'| {aspect} | {who} | {s["mean"]:.4f} | {s["min"]:.4f} | {s["max"]:.4f} | {s["sd"]:.4f} |')
    L += ['', 'Best epoch per seed:', '']
    for who, be in (('G', g['best_epochs']), ('K4', k4['best_epochs'])):
        for aspect, seeds in be.items():
            L.append(f'- {who} {aspect}: ' + ', '.join(f's{s} {x["best_epoch"]}' for s, x in seeds.items()))
    return '\n'.join(L) + '\n'


def run_report(out, arm, workers=8, n_resamples=N_RESAMPLES, smoke=False):
    """Stage C's reported readings -> `stage_c/reported.{json,md}`. Refuses before `stage_c/verdict.json` exists."""
    out = Path(out)
    if smoke:
        assert rg_paths.smoke_ok(out), 'smoke needs --outputs under /tmp/claude-0/'
    else:
        assert n_resamples == N_RESAMPLES
    verdict_path = out / 'stage_c/verdict.json'
    assert verdict_path.is_file(), f'{verdict_path}: the reported readings run only after the verdict file exists'
    verdict = json.loads(verdict_path.read_text())
    assert verdict['stage'] == 'C' and verdict['arm'] == arm and verdict['smoke'] == bool(smoke), \
        f'{verdict_path}: stage {verdict["stage"]!r}, arm {verdict["arm"]!r}, smoke {verdict["smoke"]!r}'
    if not smoke:
        vc2_path = out / 'stage_c/rederive_vc2.json'
        assert vc2_path.is_file(), f'VC2: {vc2_path} missing'
        vc2 = json.loads(vc2_path.read_text())
        assert vc2.get('agree') is True, f'VC2: {vc2_path} does not agree'
        assert vc2.get('config', {}).get('comparison_sha256') == sha_file(verdict_path), f'VC2: {vc2_path} did not check the current C verdict'
    names, sha, labels = rb_part2_report.held_labels()
    assert sha == rg_paths.HELD_SHA256 == verdict['held_sha256'] and len(names) == rg_paths.N_HELD
    for s, rec in verdict['sources']['G_outputs'].items():                     # G's genre outputs are the verdict's
        assert sha_file(rec['path']) == rec['output_sha256'], f'{rec["path"]}: changed since the verdict'
    rcol, rcol_path = rb_score.read_rcol_eligible('full')
    eligible = {a: v for a, v in rb_score.eligible_seeds((arm,) + COMPARATORS, rcol).items() if a != arm}
    sources = _comparator_sources(eligible)
    reported_stage4 = json.loads((rg_paths.RB_PART2 / 'reported_stage4.json').read_text())
    assert {a: list(v) for a, v in reported_stage4['blocks']['K4']['eligible_seeds'].items() if a != 'K4'} == eligible
    ctx = dict(sha=sha, n_rows=len(names), n_resamples=n_resamples, workers=workers, labels=labels,
               eligible=eligible, comp_sources=sources, comp_fixed=_frozen_comparator_partitions(eligible),
               comp_leiden=_leiden_comparators(eligible, sources, sha, len(names)))
    g = report_g(out, arm, ctx)
    record = dict(schema=1, stage='C', arm=arm, held_sha256=sha, rule_commit=rg_paths.RULE_COMMIT, smoke=bool(smoke),
                  n_resamples=n_resamples,
                  verdict_summary=dict(outcome_id=verdict['outcome_id'], d_T=verdict['d_T']['d'],
                                       d_R=verdict['d_R']['d'], G_half=verdict['G_half'],
                                       G_outputs=verdict['sources']['G_outputs']),
                  G=g, K4_frozen=_k4_beside(reported_stage4, sha),
                  inputs=dict(verdict_sha256=sha_file(verdict_path), rcol=str(rcol_path),
                              reported_stage4_sha256=sha_file(rg_paths.RB_PART2 / 'reported_stage4.json')),
                  code=dict(code_provenance(), rb_part2_report_sha256=sha_file(rb_part2_report.__file__)))
    record = json.loads(json.dumps(record, default=lambda o: o.item() if hasattr(o, 'item') else o.tolist()))
    path = out / 'stage_c/reported.json'
    keep = lambda r: {k: v for k, v in r.items() if k != 'code'}
    if path.exists():
        assert keep(json.loads(path.read_text())) == keep(record), f'{path}: exists and differs; not overwritten'
    else:
        _atomic_json(path, record)
    md = path.with_suffix('.md')
    if not md.exists():
        md.write_text(report_markdown(record))
    print(f'wrote {path} and {md}', flush=True)
    return record


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--outputs', type=Path, default=rg_paths.OUTPUTS)
    p.add_argument('--smoke', action='store_true', help='20 draws; outputs must be under /tmp/claude-0/; no gate')
    p.add_argument('--workers', type=int, default=8, choices=range(1, 9))
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('check').add_argument('what', choices=['vc1'])
    s = sub.add_parser('score')
    s.add_argument('--arm', required=True, help='G: the stage B pick (smoke: also K4)')
    r = sub.add_parser('report')
    r.add_argument('--arm', required=True, help='G, as for score; needs stage_c/verdict.json')
    args = p.parse_args(argv)
    rg_paths.check_imports()
    if args.smoke:
        assert rg_paths.smoke_ok(args.outputs), 'smoke needs --outputs under /tmp/claude-0/'
    n = 20 if args.smoke else N_RESAMPLES
    if args.cmd == 'check':
        assert not args.smoke, 'a regression check is never shrunk: no --smoke with check'
        record = run_check_vc1(args.outputs, args.workers)
        raise SystemExit(0 if record['pass'] else 1)
    assert args.arm in rg_variants.ARMS or (args.smoke and args.arm == 'K4'), args.arm
    if args.cmd == 'report':
        run_report(args.outputs, args.arm, args.workers, n, args.smoke)
        return
    run_score(args.outputs, args.arm, args.workers, n, args.smoke)


if __name__ == '__main__':
    with threadpool_limits(limits=8):
        main()
