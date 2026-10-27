"""Stage 3 of the recipe sweep (rule §2.3, §3.3, §4.3): TOP's configurations at seeds 1 to 3 against G's frozen runs,
the joint bootstrap, the qualifiers, PICK or KEEP-G, and the V3a scorer gate.

Modes (`python rs_stage3.py MODE`):
  check v3a                       the joint bootstrap on 12-27's stage B files, R6-5 in G's place and GS-0.02 in c's, gives
                                  `RG/outputs/stage_b/verdict.json`: Ā_a and ē_a within 1e-6 (the three aspects), the
                                  emotion and style intervals in the joint mode, all three intervals in the EG rule's
                                  per-aspect mode; records the SHA-256 of G's seed 1 to 3 exports -> gates/v3a.json
  verdict --top-from PATH         Ā, ē, M̄ with intervals for each TOP configuration of a stage 2 file (PATH is relative to
                                  --outputs, e.g. sweep/local/stage2.json), the qualifiers, PICK or KEEP-G
                                  -> stage3/verdict.json (c* is written here before any full-scale run)

The statistic is 12-27's stage B (`eg_score3.fresh_aspect`): A_a(v, I, s) is `rb_ami.score` of the run's three fixed
partitions (`rs_sweep.fixed_partitions`: MiniBatchKMeans k = 20, states 0, 1, 2) over the rows of I that carry an a-label.
New here is the draw (rule §3.3): per configuration one `default_rng(0)`; for b in order I_b = `integers(0, 29232,
size=29232)` over all rows of P, then S_b = `choice([1, 2, 3], size=3)`; the same draws serve the three aspects, c and
G, and M. `joint_bootstrap` takes the arm and the reference as parameters: the ablation scorer (ticket 10) calls it with
the final model F in G's place. The pure rule is above the line "files"; decisions use unrounded values.
"""
import argparse
import ast
import hashlib
import json
import multiprocessing
import socket
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from threadpoolctl import threadpool_limits

import rs_paths
import rs_sheet
import rs_sweep as S
import eg_score3 as E
import rb_ami
import rb_pick
import rg_paths

ASPECTS = rs_sheet.ASPECTS
RANGES = rs_sheet.RANGES                 # r_a (rule §3)
SEEDS = rs_sheet.DEV_SEEDS               # (1, 2, 3)
G_ARM = rs_paths.G_ARM                   # 'GS-0.02': G's 12-27 name; the key of G's partitions and boots
R_ARM = 'R6-5'                           # V3a puts 12-25's R6-5 in G's place
N_DRAWS = rb_ami.N_DRAWS                 # 2,000
SMOKE_DRAWS = 20
TOL = 1e-6
Z_LOW, Z_HIGH = 2.5, 97.5                # 95% interval: numpy.percentile(·, [2.5, 97.5])
MODES = ('joint', 'per_aspect')


# ================================================================== the rule (pure)
def mean_gain_bar(e):
    """M̄ = (1/3) sum_a ē_a / r_a (rule §3.3), in seed ranges (the same formula as stage 1's M)."""
    return S.mean_gain(e)


def _finite(rec):
    values = [rec['M'][k] for k in ('M', 'lo', 'hi')] + [rec['effects'][a][k] for a in ASPECTS for k in ('e', 'lo', 'hi')]
    assert all(np.isfinite(v) for v in values), f'non-finite number in an ok record: {rec}'


def conditions(rec):
    """{aspect: lower bound of ē_a > -r_a, 'M': lower bound of M̄ > 0} of a scored record (rule §4.3 "Qualifies"; both
    strict); None for a failed one."""
    if rec['status'] == 'failed':
        return None
    assert rec['status'] == 'ok', rec['status']
    _finite(rec)
    out = {a: bool(rec['effects'][a]['lo'] > -RANGES[a]) for a in ASPECTS}
    out['M'] = bool(rec['M']['lo'] > 0)
    return out


def qualifies(rec):
    """Every aspect's lower bound above -r_a and M̄'s above 0. A configuration with a failed seed (a non-finite loss, or
    V3b still failing after its rerun, V3d) does not qualify."""
    c = conditions(rec)
    return c is not None and all(c.values())


def decide(records, order):
    """PICK or KEEP-G from {name: record} of TOP's configurations and `order`, TOP's order (rule §4.3): c* is the qualifier
    with the largest unrounded M̄, ties to TOP's order. -> dict(outcome, pick, qualifiers (best first), not_qualifying,
    failed, lead). `lead`: c*'s M̄ minus the next qualifier's, against the half width of c*'s M̄ interval (reported)."""
    order = list(order)
    assert len(set(order)) == len(order) and set(records) == set(order), \
        f'records {sorted(records)} are not TOP\'s configurations {order}'
    q = sorted((n for n in order if qualifies(records[n])), key=lambda n: (-records[n]['M']['M'], order.index(n)))
    lead = None
    if len(q) >= 2:
        top, nxt = records[q[0]]['M'], records[q[1]]['M']
        diff, half = top['M'] - nxt['M'], (top['hi'] - top['lo']) / 2
        lead = dict(over=q[1], lead=diff, half_width=half, less_than_half_width=bool(diff < half))
    return dict(outcome='PICK' if q else 'KEEP-G', pick=q[0] if q else None, qualifiers=q,
                not_qualifying=[n for n in order if records[n]['status'] == 'ok' and n not in q],
                failed=[n for n in order if records[n]['status'] == 'failed'], lead=lead)


# ------------------------------------------------------------------ the joint bootstrap
def joint_draws(n_rows, n_draws, seeds=SEEDS):
    """(I int32 [n_draws, n_rows], S int64 [n_draws, 3]) from one `default_rng(0)`: for b in order, I_b =
    `integers(0, n_rows, size=n_rows)`, then S_b = `choice(seeds, size=3, replace=True)` (rule §3.3). S_b sits between
    the I_b in the stream; it is `eg_score3.fresh_draws` with the rows of P in place of L_a (same stream when L_a is all of P)."""
    assert n_rows > 0 and n_draws > 0 and len(seeds) == 3
    rng = np.random.default_rng(0)
    I = np.empty((n_draws, n_rows), dtype=np.int32)
    chosen = np.empty((n_draws, 3), dtype=np.int64)
    for b in range(n_draws):
        I[b] = rng.integers(0, n_rows, size=n_rows)
        chosen[b] = rng.choice(list(seeds), size=3, replace=True)
    return I, chosen


_STATE = {}


def _init_boot(state):
    _STATE.update(state)
    threadpool_limits(limits=1)


def _draw_scores(b):
    """A(v, I_b) of every part set, over the rows of I_b that carry a label (repeats kept, unlabelled rows dropped)."""
    I, labels = _STATE['draws'][b], _STATE['labels']
    idx = I[labels[I] >= 0]
    return np.array([rb_ami.score(labels, p, idx) for p in _STATE['parts']])


def boot_scores(labels, parts, I, workers):
    """{name: float64 [n_draws]}: A(v, I_b) for each draw (row b of `I`, rows of P), per part set alone. `parts` =
    {name: int [3, n]}, `labels` the aspect's vector over P (-1 masked)."""
    labels = np.asarray(labels)
    names = list(parts)
    state = dict(labels=labels, parts=[parts[n] for n in names], draws=I)
    if workers == 1 or len(I) < 2:
        _STATE.update(state)
        rows = [_draw_scores(b) for b in range(len(I))]
    else:
        with multiprocessing.get_context('fork').Pool(workers, initializer=_init_boot, initargs=(state,)) as pool:
            rows = pool.map(_draw_scores, range(len(I)), chunksize=max(1, len(I) // (workers * 8)))
    scores = np.stack(rows)
    assert scores.shape == (len(I), len(names)) and np.isfinite(scores).all()
    return {n: scores[:, i] for i, n in enumerate(names)}


class Boots:
    """Per-run bootstrap scores of one aspect on one seed, cached next to the run's partitions (`<root>/<name>/s<seed>/`;
    the format of `eg_score3.DiskBoots`). The identity holds the draws' tag and the hash of this module's code (which
    makes the draws and the scores), so scores of other draws or other code are refused."""

    def __init__(self, root, aspect, seed, p_sha, label_sha, img_sha, n_draws):
        self.root, self.aspect, self.seed, self.n_draws = Path(root), aspect, seed, n_draws
        self.p_sha, self.label_sha, self.img_sha = p_sha, label_sha, img_sha
        self.draw_code = code_sha256()

    def _where(self, name, tag):
        path = self.root / name / f's{self.seed}' / f'{self.aspect}__boot_{tag}.npz'
        return path, E._identity(name, self.aspect, self.p_sha, self.img_sha[name], self.seed,
                                 label_sha256=self.label_sha, n_draws=self.n_draws, draws=f'joint-v1:{tag}',
                                 draw_code_sha256=self.draw_code)

    def get(self, name, tag):
        path, identity = self._where(name, tag)
        return E._read_npz_with_record(path, 'boot', identity)

    def put(self, name, tag, boot):
        path, identity = self._where(name, tag)
        E._write_npz_with_record(path, 'boot', np.asarray(boot, dtype=np.float64), dict(identity=identity))


class _NoCache:
    def get(self, name, tag):
        return None

    def put(self, name, tag, boot):
        pass


def joint_bootstrap(labels, parts, arm, ref, seeds, n_draws, workers=8, caches=None, mode='joint'):
    """Ā, ē and the intervals of `arm` against `ref` (rule §3.3), for all aspects from the same draws.
    labels {aspect: int [n]}; parts {aspect: {name: {seed: int [3, n]}}} for the two names, the same seeds; caches
    {aspect: {seed: Boots}} (optional). mode 'joint': one (I_b, S_b) stream over all n rows for the three aspects and M.
    mode 'per_aspect': the EG rule's draw (`eg_score3.fresh_aspect`): each aspect its own stream over its labelled rows,
    no M (S_b differs between aspects).
    -> dict(effects={aspect: dict(A {name: {seed: A}}, Abar {name: mean}, e, lo, hi)}, M=dict(M, lo, hi) or None)."""
    assert mode in MODES, mode
    seeds, names = tuple(seeds), (arm, ref)
    n_rows = len(labels[ASPECTS[0]])
    assert arm != ref and all(len(labels[a]) == n_rows for a in ASPECTS)
    assert all(set(parts[a]) == set(names) and all(sorted(parts[a][n]) == sorted(seeds) for n in names) for a in ASPECTS)
    caches = caches or {}
    shared = joint_draws(n_rows, n_draws, seeds) if mode == 'joint' else None
    effects, e_b = {}, {}
    for a in ASPECTS:
        y = np.asarray(labels[a])
        L = np.flatnonzero(y >= 0)
        A = {n: {s: float(rb_ami.score(y, parts[a][n][s], L)) for s in seeds} for n in names}
        Abar = {n: float(np.mean([A[n][s] for s in seeds])) for n in names}
        if mode == 'joint':
            (I, Sb), tag = shared, f'P-n{n_rows}'
        else:
            I, Sb = joint_draws(len(L), n_draws, seeds)
            I = L[I]                                             # positions in L_a -> rows of P
            tag = f'P-n{n_rows}' if len(L) == n_rows else f'L-n{len(L)}'      # the same draws when L_a is all of P
        boot = {n: {} for n in names}
        for s in seeds:
            cache = caches.get(a, {}).get(s, _NoCache())
            got = {n: cache.get(n, tag) for n in names}
            missing = {n: parts[a][n][s] for n in names if got[n] is None}
            if missing:
                for n, b in boot_scores(y, missing, I, workers).items():
                    cache.put(n, tag, b)
                    got[n] = b
            for n in names:
                assert got[n].shape == (n_draws,) and np.isfinite(got[n]).all()
                boot[n][s] = got[n]
        e_b[a] = E.paired_gains(boot[arm], boot[ref], Sb)
        lo, hi = np.percentile(e_b[a], [Z_LOW, Z_HIGH])
        effects[a] = dict(A=A, Abar=Abar, e=float(Abar[arm] - Abar[ref]), lo=float(lo), hi=float(hi))
        print(f'{a}: {arm} minus {ref}: e {effects[a]["e"]:.6f} [{lo:.6f}, {hi:.6f}] ({mode})', flush=True)
    if mode == 'per_aspect':
        return dict(effects=effects, M=None)
    M_b = sum(e_b[a] / RANGES[a] for a in ASPECTS) / len(ASPECTS)
    lo, hi = np.percentile(M_b, [Z_LOW, Z_HIGH])
    return dict(effects=effects, M=dict(M=float(mean_gain_bar({a: effects[a]['e'] for a in ASPECTS})), lo=float(lo), hi=float(hi)))


# ================================================================== files
_clean = S._clean
_file_sha256 = S._file_sha256


def code_sha256():
    """Hash this module and its transitive top-level imports in the round, RG, EG and RB folders (copied from
    `rs_sweep.scorer_code_sha256`, which starts from rs_sweep.py; same method). Names as well as contents are hashed; lazy
    imports inside functions are excluded."""
    folders = (rs_paths.ROUND, rg_paths.ROUND, Path(E.__file__).parent, Path(rb_ami.__file__).parent)
    pending = [rs_paths.ROUND / 'rs_stage3.py']
    files = set()
    while pending:
        path = pending.pop()
        if path in files:
            continue
        files.add(path)
        for node in ast.parse(path.read_bytes(), filename=str(path)).body:
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                for folder in folders:
                    dependency = folder / f'{name}.py'
                    if dependency.is_file():
                        pending.append(dependency)
                        break
    manifest = {f'{p.parent.name}/{p.name}': _file_sha256(p) for p in files}
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def code_state():
    return dict(S.code_state(), code_sha256=code_sha256())


def cache_root(outputs):
    """Partitions and bootstrap scores of every run stage 3 and V3a score: `<root>/<name>/s<seed>/`."""
    return Path(outputs) / 'stage3' / 'cache'


def gate_path(outputs):
    return Path(outputs) / 'gates' / 'v3a.json'


def verdict_path(outputs):
    return Path(outputs) / 'stage3' / 'verdict.json'


def read_frozen_run(folder, variant, seed):
    """{run_sha256, img {aspect: array}, file_sha256 {aspect}, array_sha256 {aspect}} of a frozen 12-25 or 12-27 run: its
    run.json (variant, seed, status ok, P's SHA-256, not a smoke) and exports that pass V1b."""
    folder = Path(folder)
    path = folder / 'run.json'
    assert path.exists(), f'{path} missing'
    run = json.loads(path.read_text())
    assert run['variant'] == variant and run['seed'] == seed and run['status'] == 'ok', (folder, run['variant'], run['seed'], run['status'])
    assert run['p_sha256'] == rs_paths.P_SHA256 and not run.get('smoke'), f'{path}: other P, or a smoke run'
    img = S.read_exports(folder)
    problems = S.v1b_problems(img)
    assert not problems, f'V1b {variant} s{seed}: {problems}'
    return dict(run_sha256=_file_sha256(path), img=img, file_sha256={a: _file_sha256(folder / f'{a}__P.npz') for a in ASPECTS},
                array_sha256={a: rb_pick.array_sha256(img[a]) for a in ASPECTS})


def g_folder(seed):
    return rs_paths.RG_TRAIN / G_ARM / f's{seed}'


def read_g_runs(gate):
    """G's seed 1 to 3 runs, only if every file has the SHA-256 the V3a gate recorded (rule §4.3)."""
    out = {}
    for s in SEEDS:
        run = read_frozen_run(g_folder(s), G_ARM, s)
        want = gate['g_exports'][str(s)]
        assert gate['g_run_sha256'][str(s)] == run['run_sha256'], f'G s{s}: run.json differs from the one V3a hashed'
        for a in ASPECTS:
            assert want[a]['file_sha256'] == run['file_sha256'][a] and want[a]['array_sha256'] == run['array_sha256'][a], \
                f'G s{s} {a}: the export differs from the one V3a hashed; stage 3 reads only hashed files'
        out[s] = run
    return out


def run_partitions(outputs, name, runs, p_sha):
    """{aspect: {seed: int [3, n]}} of a run's three seeds, cached under `cache_root`."""
    return {a: {s: S.fixed_partitions(cache_root(outputs) / name / f's{s}', name, a, runs[s]['img'][a], p_sha, s)
                for s in runs} for a in ASPECTS}


def caches_for(outputs, names, runs, ctx, n_draws):
    """{aspect: {seed: Boots}} for the named runs; `runs` = {name: {seed: {img}}}."""
    label_sha = {a: rb_pick.array_sha256(ctx['labels'][a]) for a in ASPECTS}
    return {a: {s: Boots(cache_root(outputs), a, s, ctx['p_sha'], label_sha[a],
                         {n: rb_pick.array_sha256(runs[n][s]['img'][a]) for n in names}, n_draws) for s in SEEDS} for a in ASPECTS}


def now_amsterdam():
    """Amsterdam local time to the minute, `YYYY-MM-DD HH:MM` (the verdict's `written`)."""
    return datetime.now(ZoneInfo('Europe/Amsterdam')).strftime('%Y-%m-%d %H:%M')


def compare(items, name, value, want):
    items[name] = dict(value=float(value), expected=float(want), diff=float(abs(value - want)),
                       **{'pass': bool(abs(value - want) <= TOL)})


# ------------------------------------------------------------------ V3a
def v3a_item_names(smoke):
    """The items a V3a gate must hold (names, in the order `run_check_v3a` writes them): per aspect Ā of both arms (the
    arm, then its three seeds) and ē; unless a smoke, the joint bounds (emotion and style: genre's labelled rows are
    fewer than P's, so its joint draw is not the EG rule's) and the per-aspect-mode bounds of all three; the two hash
    and partition equalities."""
    names = []
    for a in ASPECTS:
        for n in (G_ARM, R_ARM):
            names += [f'{a}_Abar_{n}'] + [f'{a}_A_{n}_s{s}' for s in SEEDS]
        names.append(f'{a}_e')
        if not smoke:
            if a != 'genre':
                names += [f'{a}_lo_joint', f'{a}_hi_joint']
            names += [f'{a}_lo_per_aspect', f'{a}_hi_per_aspect']
    return names + ['export_hashes_equal_stage_b', 'partitions_equal_to_stage_b']


def partition_item(equal):
    """The gate item for `equal` = {name: {seed: {aspect: bool}}}: the run's partitions equal 12-27's stage B files."""
    bad = [f'{n} s{s} {a}' for n in equal for s in equal[n] for a in equal[n][s] if not equal[n][s][a]]
    return {'mismatches': bad, 'pass': not bad}


def run_check_v3a(outputs, workers, smoke=False):
    """V3a (rule §4.3) -> gates/v3a.json. 12-27's stage B files with R6-5 in G's place and GS-0.02 in c's: Ā_a of both and ē_a
    equal `RG/outputs/stage_b/verdict.json` within 1e-6 (joint mode), so do the emotion and style bounds (all 29,232 rows
    carry both labels, so the joint draw is the EG rule's for them), and in the EG rule's per-aspect mode all three
    aspects' bounds. Also: both arms' exports hash (array) to the hashes in that verdict's identity. Records the SHA-256 of
    G's seed 1 to 3 exports (files and arrays) and run.json: stage 3 later reads only these. A smoke (20 draws) checks
    the point values only, not the bounds."""
    n_draws = SMOKE_DRAWS if smoke else N_DRAWS
    code = code_state()                                   # taken before the scoring: the record names the code that computed it
    ref_path = rs_paths.RG_STAGE_B / 'verdict.json'
    ref = json.loads(ref_path.read_text())
    ctx = E.load_context()
    assert not ref['smoke'] and ref['n_draws'] == N_DRAWS and ref['reference'] == R_ARM and ref['seeds'] == list(SEEDS)
    assert ref['identity']['p_sha256'] == ctx['p_sha'] == rs_paths.P_SHA256
    want = ref['results'][G_ARM]['effects']
    arms = (G_ARM, R_ARM)
    runs = {G_ARM: {s: read_frozen_run(g_folder(s), G_ARM, s) for s in SEEDS},
            R_ARM: {s: read_frozen_run(rg_paths.frozen_run(R_ARM, s), R_ARM, s) for s in SEEDS}}
    parts = {a: {n: {} for n in arms} for a in ASPECTS}
    equal = {n: {} for n in arms}
    for n in arms:
        for s in SEEDS:
            equal[n][str(s)] = {}
            for a in ASPECTS:
                parts[a][n][s] = S.fixed_partitions(cache_root(outputs) / n / f's{s}', n, a, runs[n][s]['img'][a], ctx['p_sha'], s)
                with np.load(rs_paths.RG_STAGE_B / 'partitions' / n / f's{s}' / f'{a}__partitions.npz', allow_pickle=False) as z:
                    equal[n][str(s)][a] = bool(np.array_equal(z['labels'], parts[a][n][s]))
        print(f'{n}: partitions done', flush=True)
    caches = caches_for(outputs, arms, runs, ctx, n_draws)
    joint = joint_bootstrap(ctx['labels'], parts, G_ARM, R_ARM, SEEDS, n_draws, workers, caches, 'joint')
    per = joint_bootstrap(ctx['labels'], parts, G_ARM, R_ARM, SEEDS, n_draws, workers, caches, 'per_aspect')
    items = {}
    for a in ASPECTS:
        for n in arms:
            compare(items, f'{a}_Abar_{n}', joint['effects'][a]['Abar'][n], want[a]['Abar'][n])
            for s in SEEDS:
                compare(items, f'{a}_A_{n}_s{s}', joint['effects'][a]['A'][n][s], want[a]['A'][n][str(s)])
        compare(items, f'{a}_e', joint['effects'][a]['e'], want[a]['e'])
        if not smoke:
            if a != 'genre':                                  # genre's labelled rows are fewer than P's: its joint draw differs
                compare(items, f'{a}_lo_joint', joint['effects'][a]['lo'], want[a]['lo'])
                compare(items, f'{a}_hi_joint', joint['effects'][a]['hi'], want[a]['hi'])
            compare(items, f'{a}_lo_per_aspect', per['effects'][a]['lo'], want[a]['lo'])
            compare(items, f'{a}_hi_per_aspect', per['effects'][a]['hi'], want[a]['hi'])
    bad = [f'{n} s{s} {a}' for n in arms for s in SEEDS for a in ASPECTS
           if runs[n][s]['array_sha256'][a] != ref['identity']['img_sha256'][n][str(s)][a]]
    items['export_hashes_equal_stage_b'] = {'mismatches': bad, 'pass': not bad}
    items['partitions_equal_to_stage_b'] = partition_item(equal)
    assert list(items) == v3a_item_names(smoke), 'V3a items differ from the item list the verdict requires'
    for k, v in items.items():
        if 'value' in v:
            print(f'{k}: {v["value"]:.10f} (12-27 {v["expected"]:.10f}, diff {v["diff"]:.1e})', flush=True)
    record = dict(stage='v3a', smoke=bool(smoke), **{'pass': all(i['pass'] for i in items.values())}, tol=TOL, n_draws=n_draws,
                  bounds_checked=not smoke, items=items, partitions_equal_to_stage_b=equal,
                  joint_genre_reading=dict(lo=joint['effects']['genre']['lo'], hi=joint['effects']['genre']['hi']),
                  joint_M_reading=joint['M'],
                  g_arm=G_ARM, g_exports={str(s): {a: dict(file_sha256=runs[G_ARM][s]['file_sha256'][a],
                                                          array_sha256=runs[G_ARM][s]['array_sha256'][a]) for a in ASPECTS}
                                          for s in SEEDS},
                  g_run_sha256={str(s): runs[G_ARM][s]['run_sha256'] for s in SEEDS},
                  reference=str(ref_path), reference_sha256=_file_sha256(ref_path), p_sha256=ctx['p_sha'],
                  label_sha256={a: rb_pick.array_sha256(ctx['labels'][a]) for a in ASPECTS}, versions=rb_ami.versions(),
                  host=socket.gethostname(), rule_commit=rs_paths.RULE_COMMIT, **code)
    return S._keep_or_write(gate_path(outputs), _clean(record))


def require_v3a(outputs, smoke):
    """The gate file: exists, a V3a record of the same kind (smoke or real), passed, written by this code, on P, with every
    V3a item (and, unless a smoke, the bounds checked with 2,000 draws)."""
    path = gate_path(outputs)
    assert path.exists(), f'{path} missing: run `rs_stage3 check v3a` first (V3a)'
    gate = json.loads(path.read_text())
    assert gate['stage'] == 'v3a' and gate['smoke'] is bool(smoke), f'{path}: not a V3a record of this kind (smoke flag differs)'
    assert gate['pass'] is True, f'V3a did not pass ({path}); stage 3 stops, trace the cause'
    assert gate['code_sha256'] == code_sha256(), f'{path} was written by other stage 3 code; rerun `check v3a`'
    assert gate['p_sha256'] == rs_paths.P_SHA256, f'{path}: other P'
    missing = [k for k in v3a_item_names(smoke) if k not in gate['items']]
    assert not missing, f'{path}: items missing from the V3a record: {missing}'
    assert gate['bounds_checked'] is (not smoke) and gate['n_draws'] == (SMOKE_DRAWS if smoke else N_DRAWS), \
        f'{path}: bounds not checked, or not with {N_DRAWS} draws'
    return gate


# ------------------------------------------------------------------ the verdict
def read_top(outputs, path, smoke):
    """({names in TOP's order}, record, SHA-256) of a stage 2 file."""
    path = Path(path)
    path = path if path.is_absolute() else Path(outputs) / path
    assert path.name == 'stage2.json' and path.exists(), f'{path}: not a stage2.json'
    s2 = json.loads(path.read_text())
    assert s2['stage'] == 'stage2' and s2['smoke'] is bool(smoke), f'{path}: not a stage 2 record of this kind (smoke flag differs)'
    assert s2['p_sha256'] == rs_paths.P_SHA256, f'{path}: other P'
    assert s2['outcome'] == 'TOP', f'stage 2 gave {s2["outcome"]}: stages 3 and 4 do not run; G stays the recipe'
    top = sorted(s2['top'], key=lambda t: t['rank'])
    assert [t['rank'] for t in top] == list(range(1, len(top) + 1)) and 1 <= len(top) <= S.TOP_N, top
    names = [t['name'] for t in top]
    assert len(set(names)) == len(names) and all(rs_sheet.name(rs_sheet.parse(n)) == n for n in names), names
    return names, s2, _file_sha256(path)


def read_config_runs(outputs, names, smoke):
    """{name: {seed: run record of `rs_sweep.read_run`}} on seeds 1 to 3, trained locally. A TOP configuration without a
    run.json on some seed stops stage 3 (V3d: only TOP's configurations at seeds 1 to 3 run, and each of them must have
    run). A failed run (a non-finite loss, or V3b after its rerun) is read as failed; the exports of ok runs pass V1b."""
    missing = [f'{n} s{s}' for n in names for s in SEEDS if not (rs_paths.run_dir(outputs, n, s) / 'run.json').exists()]
    assert not missing, f'V3d: TOP configurations lack runs (the seeds 1 to 3 runs have not finished): {missing}'
    return {n: {s: S.read_run(outputs, n, s, 'local', smoke) for s in SEEDS} for n in names}


def failed_record(runs):
    causes = [f'seed {s}: {r["cause"]}' for s, r in sorted(runs.items()) if r['status'] == 'failed']
    return dict(status='failed', cause='; '.join(causes), effects=None, M=None, conditions=None, qualifies=False)


def score_configurations(outputs, names, config_runs, gate, ctx, n_draws, workers):
    """{name: record}: for a configuration with all three seeds ok, `joint_bootstrap` against G's seed 1 to 3 runs; one with
    a failed seed is a failed record (no numbers, does not qualify)."""
    records = {n: failed_record(config_runs[n]) for n in names if any(r['status'] == 'failed' for r in config_runs[n].values())}
    todo = [n for n in names if n not in records]
    if not todo:
        return records, {}
    g_runs = read_g_runs(gate)
    runs = dict({G_ARM: g_runs}, **{n: config_runs[n] for n in todo})
    for n in runs:
        print(f'{n}: partitions', flush=True)
    for n in todo:
        parts = {a: {} for a in ASPECTS}
        for name in (n, G_ARM):
            p = run_partitions(outputs, name, runs[name], ctx['p_sha'])
            for a in ASPECTS:
                parts[a][name] = p[a]
        caches = caches_for(outputs, (n, G_ARM), runs, ctx, n_draws)
        out = joint_bootstrap(ctx['labels'], parts, n, G_ARM, SEEDS, n_draws, workers, caches, 'joint')
        rec = dict(status='ok', cause=None, effects=out['effects'], M=out['M'])
        records[n] = dict(rec, conditions=conditions(rec), qualifies=qualifies(rec))
    return records, runs


def run_verdict(outputs, top_path, workers=8, smoke=False):
    """TOP's configurations at seeds 1 to 3 against G -> stage3/verdict.json (rule §4.3): Ā_a, ē_a, M̄ with intervals,
    the qualifiers, PICK (c*) or KEEP-G. Written before any full-scale run of c*. Stops without V3a's gate for this code,
    on a TOP configuration with a missing seed (V3d), or when stage 2 gave TOP-NONE."""
    names, s2, s2_sha = read_top(outputs, top_path, smoke)
    gate = require_v3a(outputs, smoke)
    config_runs = read_config_runs(outputs, names, smoke)
    code = code_state()                                   # taken before the scoring: the record names the code that computed it
    n_draws = SMOKE_DRAWS if smoke else N_DRAWS
    ctx = E.load_context() if any(r['status'] == 'ok' for rs_ in config_runs.values() for r in rs_.values()) else None
    records, runs = score_configurations(outputs, names, config_runs, gate, ctx, n_draws, workers)
    d = decide(records, names)
    hashes = {n: {str(s): dict(run_sha256=r['run_sha256'], img_sha256=None if r['img'] is None else
                              {a: rb_pick.array_sha256(r['img'][a]) for a in ASPECTS}) for s, r in config_runs[n].items()}
              for n in names}
    record = dict(stage='stage3', smoke=bool(smoke), written=now_amsterdam(), stage2_machine=s2['machine'], rule_commit=rs_paths.RULE_COMMIT,
                  ranges=dict(RANGES), seeds=list(SEEDS), n_draws=n_draws, draw_mode='joint', reference=dict(name='G', arm=G_ARM),
                  top=names, **d, pick_config=json.loads(json.dumps(rs_sheet.parse(d['pick']))) if d['pick'] else None,
                  results={n: records[n] for n in names},
                  inputs=dict(stage2_sha256=s2_sha, v3a_sha256=_file_sha256(gate_path(outputs)), runs=hashes,
                              g_exports=gate['g_exports'], g_run_sha256=gate['g_run_sha256']),
                  p_sha256=rs_paths.P_SHA256, versions=rb_ami.versions(), host=socket.gethostname(), **code)
    return S._keep_or_write(verdict_path(outputs), _clean(record))


# ================================================================== CLI
def main(argv=None):
    parser = argparse.ArgumentParser(description='Stage 3 of the recipe sweep: joint bootstrap, qualifiers, PICK or KEEP-G; V3a.')
    parser.add_argument('mode', choices=['check', 'verdict'])
    parser.add_argument('what', nargs='?', choices=['v3a'], help='for `check`')
    parser.add_argument('--top-from', metavar='PATH', help='stage 2 file, relative to --outputs (sweep/local/stage2.json)')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--outputs', type=Path, default=None,
                        help=f'default {rs_paths.OUTPUTS}; with --smoke {rs_paths.SMOKE_ROOT}/r10-rs-smoke')
    parser.add_argument('--smoke', action='store_true', help='outputs under /tmp/claude-0/; 20 draws; reads smoke runs')
    args = parser.parse_args(argv)
    assert (args.mode == 'check') == (args.what is not None), 'usage: check v3a | verdict --top-from PATH'
    assert (args.mode == 'verdict') == (args.top_from is not None), '--top-from goes with verdict'
    assert 1 <= args.workers <= 8, '--workers at most 8'
    outputs = args.outputs or (rs_paths.SMOKE_ROOT / 'r10-rs-smoke' if args.smoke else rs_paths.OUTPUTS)
    outputs = rs_paths.check_outputs(outputs, args.smoke)
    rs_paths.check_imports()
    if args.mode == 'check':
        rec = run_check_v3a(outputs, args.workers, args.smoke)
        print(f"{gate_path(outputs)}: V3a {'PASS' if rec['pass'] else 'FAIL'}; "
              f"{sum(i['pass'] for i in rec['items'].values())} of {len(rec['items'])} items within {TOL}")
        return 0 if rec['pass'] else 1
    rec = run_verdict(outputs, args.top_from, args.workers, args.smoke)
    print(f"{verdict_path(outputs)}: {rec['outcome']}" + (f", c* = {rec['pick']}" if rec['pick'] else '') +
          f", qualifiers {rec['qualifiers']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
