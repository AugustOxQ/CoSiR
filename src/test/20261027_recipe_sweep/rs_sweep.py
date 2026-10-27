"""Stages 1 and 2 of the recipe sweep (rule §3.2, §4.2): the scorer on P, the gains, the guard, the knob gains, the
grid, the pool, TOP; the V1a scorer half; the node route's N3.

Modes (`python rs_sweep.py MODE`):
  score --machine M --runs NAME...   partitions and A on P of each run (rule §3.2): V1b per export, P's SHA-256 = 12-25's
                                     -> <outputs>/scores/M/<name>__s<seed>.json and scores/M/partitions/<name>/s<seed>/
  stage1 --machine M                 knob curves, guards, g_k, grid knobs, outcome, the grid -> sweep/M/stage1.json
  stage2 --machine M                 pool, TOP, outcome -> sweep/M/stage2.json
  n3 --machine node402               the node's repeat check and its readings (rule §4.2 "The node route") -> gates/n3.json
  check v1a                          the scorer on 12-27's GS-0.02 seed-0 exports gives its stage A values within 1e-6
                                     -> gates/v1a_scorer.json

The scoring kernel is 12-27's stage A scorer, unchanged: `rb_ami.partitions` (MiniBatchKMeans k = 20, states 0, 1, 2),
`rb_ami.score` against the labels L_a on P (`eg_score3.load_context`), the partition cache in the format of
`eg_score3.fixed_partitions` (copied below with a cache folder of its own; the original's sits under stage_iii/). A run's
A_a is the mean over the three states. The stage logic is below the line "the rule's stage logic": pure functions over a
table {run name: row} of A values, no file is read there. Decisions use unrounded values. A machine's decisions read only
that machine's scores (rule §4.0).
"""
import argparse
import ast
import hashlib
import itertools
import json
import socket
import subprocess
from pathlib import Path

import numpy as np

import rs_paths
import rs_sheet
import eg_score3 as E
import rb_ami
import rb_data
import rb_pick
import rg_paths

ASPECTS = rs_sheet.ASPECTS
RANGES = rs_sheet.RANGES                 # r_a (rule §3)
TOL = 1e-6                               # V1a
G_NAME = 'G'                             # G₀: the run named G (the sheet's name of G's configuration)
REPEAT = 'G-repeat'                      # G₀′: G at seed 0 again, right after G₀ (rule §2.2); its own run folder
MACHINES = ('local', 'node402')
TOP_N = 3
MAX_GRID_KNOBS = 4
MAX_LEVELS = 2                           # counting values with M > 0 kept per grid knob, besides G's value

_json_dump = E._json_dump
_file_sha256 = E._file_sha256


# ================================================================== the rule's stage logic (pure)
def gain_of(A, ref):
    """gain_a = A_a(c) - A_a(G₀) per aspect (rule §3.2)."""
    return {a: A[a] - ref[a] for a in ASPECTS}


def mean_gain(gain):
    """M = (1/3) sum_a gain_a / r_a (rule §3.2), in seed ranges."""
    return sum(gain[a] / RANGES[a] for a in ASPECTS) / len(ASPECTS)


def guard_ok(gain):
    """The guard: gain_a >= -r_a / 2 for every aspect (rule §3.2); the edge passes."""
    return all(gain[a] >= -RANGES[a] / 2 for a in ASPECTS)


def judge(row, ref):
    """A table row {status, cause, A, best_epoch} against G₀'s A `ref`: gains, M, guard, counting (a run is counting when
    it passed V1b, i.e. has status ok, and the guard). A failed run has no gain and does not count."""
    if row['status'] == 'failed':
        return dict(status='failed', cause=row['cause'], A=None, gain=None, M=None, guard=None, counting=False,
                    best_epoch=row.get('best_epoch'))
    assert row['status'] == 'ok' and row['A'] is not None, row
    gain = gain_of(row['A'], ref)
    guard = guard_ok(gain)
    return dict(status='ok', cause=None, A=dict(row['A']), gain=gain, M=mean_gain(gain), guard=guard, counting=guard,
                best_epoch=row.get('best_epoch'))


def judge_all(table, ref):
    """{name: judged row} in the table's order."""
    return {name: dict(judge(row, ref), name=name) for name, row in table.items()}


def value_name(k, v):
    return rs_sheet.name(rs_sheet.config({k: v}))


def knob_curve(judged, k):
    """The knob's stage 1 runs in the sheet's order of values: {value, name, M, gain, guard, counting, status}; a run that
    is not in `judged` has status None and no numbers."""
    out = []
    for v in rs_sheet.VALUES[k]:
        name = value_name(k, v)
        j = judged.get(name)
        out.append(dict(value=v, name=name, status=None, cause=None, M=None, gain=None, guard=None, counting=False)
                   if j is None else dict(value=v, name=name, **{x: j[x] for x in
                                                                 ('status', 'cause', 'M', 'gain', 'guard', 'counting')}))
    return out


def knob_gain(judged, k):
    """g_k = the largest M over the knob's counting values; None when no value counts (rule §4.2 Stage 1)."""
    ms = [c['M'] for c in knob_curve(judged, k) if c['counting']]
    return max(ms) if ms else None


def knob_gains(judged):
    return {k: knob_gain(judged, k) for k in rs_sheet.KNOBS}


def grid_knobs(gains):
    """The knobs with g_k > 0; the 4 with the largest g_k when more qualify, ties to the sheet's order (k1 first). Listed
    by g_k, largest first (ties by the sheet's order)."""
    order = {k: i for i, k in enumerate(rs_sheet.KNOBS)}
    qualify = sorted((k for k, g in gains.items() if g is not None and g > 0), key=lambda k: (-gains[k], order[k]))
    return qualify[:MAX_GRID_KNOBS]


def stage1_outcome(knobs):
    return f'GRID-{len(knobs)}' if knobs else 'SWEEP-NONE'


def levels(judged, k):
    """A grid knob's levels: G's value, then its counting values with M > 0, at most two (the largest M; ties to the
    listed order of values), in the sheet's order."""
    candidates = [(i, c) for i, c in enumerate(knob_curve(judged, k)) if c['counting'] and c['M'] > 0]
    best = sorted(candidates, key=lambda ic: (-ic[1]['M'], ic[0]))[:MAX_LEVELS]
    return [rs_sheet.G[k]] + [c['value'] for _, c in sorted(best, key=lambda ic: ic[0])]


def grid_points(knobs, level_of):
    """Every combination of the grid knobs' levels, all other knobs at G's value: configurations in the order of the
    sheet's knobs (the first knob changes slowest), each knob's levels as listed, G's value first."""
    knobs = sorted(knobs, key=rs_sheet.KNOBS.index)
    return [rs_sheet.config(dict(zip(knobs, combo))) for combo in itertools.product(*(level_of[k] for k in knobs))]


def split_points(points):
    """The grid's points by origin: the all-G point (G₀), the one-knob points (stage 1 runs, reused), the rest (new runs
    to train, at seed 0 on the same machine)."""
    order = [rs_sheet.name(p) for p in points]
    n = {name: len(rs_sheet.changed(p)) for name, p in zip(order, points)}
    return dict(order=order, g0=[x for x in order if n[x] == 0], reused=[x for x in order if n[x] == 1],
                new=[x for x in order if n[x] >= 2])


def pool(judged):
    """Every counting run with M > 0, G₀ and G₀′ excluded (rule §4.2 Stage 2), in the table's order."""
    return [n for n, j in judged.items() if n not in (G_NAME, REPEAT) and j['counting'] and j['M'] > 0]


def tie_key(judged, name):
    """Rule §4.2 TOP: larger M first; ties to fewer changed knobs, then the sorted tuple of the changed knobs' numbers
    (lexicographic), then the tuple of their values' positions in the sheet."""
    cfg = rs_sheet.parse(name)
    return (-judged[name]['M'], len(rs_sheet.changed(cfg)), rs_sheet.knob_numbers(cfg), rs_sheet.positions(cfg))


def top(judged, names, n=TOP_N):
    """The n runs of `names` (the pool) with the largest M, all of them when fewer."""
    return sorted(names, key=lambda x: tie_key(judged, x))[:n]


def stage2_outcome(pool_names):
    return 'TOP' if pool_names else 'TOP-NONE'


def n3_items(a_g0, a_g0p):
    """N3 (rule §4.2): |A_a(G₀′) - A_a(G₀)| <= r_a / 2 per aspect; the edge passes."""
    return {a: dict(G0=a_g0[a], G0_repeat=a_g0p[a], diff=abs(a_g0p[a] - a_g0[a]), limit=RANGES[a] / 2,
                    **{'pass': bool(abs(a_g0p[a] - a_g0[a]) <= RANGES[a] / 2)}) for a in ASPECTS}


# ================================================================== the scorer (12-27's stage A kernel)
def v1b_problems(img):
    """V1b per export: {aspect: first problem} for the exports that are not 29,232 finite unit-norm float16 rows."""
    problems = {}
    for a, x in img.items():
        problem = rb_pick.vp2_problem(x)
        if problem is not None:
            problems[a] = problem
    return problems


def partitions_dir(outputs, machine, name, seed):
    return Path(outputs) / 'scores' / machine / 'partitions' / name / f's{seed}'


def fixed_partitions(folder, name, aspect, img, p_sha, seed):
    """int64 [3, n]: the run's fixed partitions of one aspect (RB rule §3.1), computed once and cached in `folder`
    (copied from `eg_score3.fixed_partitions`, whose cache sits under stage_iii/; same file format and identity)."""
    path = Path(folder) / f'{aspect}__partitions.npz'
    identity = E._identity(name, aspect, p_sha, rb_pick.array_sha256(img), seed)
    parts = E._read_npz_with_record(path, 'labels', identity)
    if parts is None:
        parts = rb_ami.partitions(img)
        E._write_npz_with_record(path, 'labels', parts, dict(identity=identity, versions=rb_ami.versions(),
                                                             k=rb_ami.K, states=list(rb_ami.STATES)))
    assert parts.shape == (3, len(img)) and parts.dtype == np.int64
    return parts


def score_images(img, folder, name, seed, ctx):
    """({aspect: A}, {aspect: partitions}) of a run's three exports: A_a = rb_ami.score over the paintings with an
    a-label (the climb's point score of 12-27's stage A, `rg_score.score_arms` with n_draws None)."""
    A, parts = {}, {}
    for a in ASPECTS:
        parts[a] = fixed_partitions(folder, name, a, img[a], ctx['p_sha'], seed)
        A[a] = float(rb_ami.score(ctx['labels'][a], parts[a], np.flatnonzero(ctx['labels'][a] >= 0)))
        print(f'{name} s{seed} {a}: A {A[a]:.10f}', flush=True)
    return A, parts


def read_exports(folder):
    img = {}
    for a in ASPECTS:
        with np.load(Path(folder) / f'{a}__P.npz', allow_pickle=False) as z:
            assert z.files == ['img'], (folder, a, z.files)
            img[a] = z['img']
    return img


def config_of(name):
    return rs_sheet.config() if name == REPEAT else rs_sheet.parse(name)


def frozen_folder():
    """12-27's GS-0.02 seed-0 run (G₀ of the local fallback; V1a's input)."""
    return rs_paths.RG_TRAIN / rs_paths.G_ARM / 's0'


def read_run(outputs, name, seed, machine, smoke, frozen_g0=False):
    """{folder, source, status, cause, best_epoch, run_sha256, img} of one run. The run's own record is checked (name,
    config, seed, machine, P's SHA-256 = 12-25's, smoke flag); a run with status 'failed' has no exports read; the exports
    of a run with status 'ok' must pass V1b (a run that fails V1b is rerun by the trainer, rule V1d, never scored)."""
    cfg = config_of(name)
    if frozen_g0:
        assert machine == 'local' and name == G_NAME and seed == 0, '--frozen-g0 is G at seed 0 on the local machine only'
        folder = frozen_folder()
    else:
        folder = rs_paths.run_dir(outputs, name, seed)
    path = folder / 'run.json'
    assert path.exists(), f'{name} s{seed}: {path} (run.json) missing; the run has not finished'
    run = json.loads(path.read_text())
    if frozen_g0:
        assert run['variant'] == rs_paths.G_ARM and run['seed'] == 0, (run['variant'], run['seed'])
    else:
        assert run['name'] in {name, rs_sheet.name(cfg)} and run['seed'] == seed, (name, run['name'], run['seed'])
        assert run['config'] == json.loads(json.dumps(cfg)), f'{name}: run.json config differs from the name\'s'
        assert run['machine'] == machine, f'{name}: trained on machine {run["machine"]!r}, scored as {machine!r}'
    assert run['status'] in ('ok', 'failed'), (name, run['status'])
    assert run['p_sha256'] == rs_paths.P_SHA256, f'{name}: run.json says P differs from 12-25\'s'
    assert bool(run.get('smoke', False)) is bool(smoke), f'{name}: smoke flag of the run differs from this scoring\'s'
    rec = dict(folder=folder, source='12-27' if frozen_g0 else 'rs_trainer', status=run['status'],
               cause=run.get('cause'), best_epoch=run.get('best_epoch'), run_sha256=_file_sha256(path), img=None)
    if run['status'] == 'failed':
        assert run.get('cause'), f'{name}: failed run without a cause'
        # as rg_score.read_run: a run fails by a non-finite loss, or by V1b after its one rerun (rule V1d)
        assert run.get('failure_kind') in ('nonfinite_loss', 'va3'), (name, 'failure_kind', run.get('failure_kind'))
        assert run['failure_kind'] != 'va3' or 'rerun_of' in run, f'{name} s{seed}: failed V1b and has not been rerun; rerun it once first'
        return dict(rec, **({'rerun_of': run['rerun_of']} if 'rerun_of' in run else {}))
    img = read_exports(folder)
    problems = v1b_problems(img)
    assert not problems, f'V1b {name} s{seed}: {problems} (the trainer reruns it once; a second failure is status failed)'
    return dict(rec, img=img)


# ------------------------------------------------------------------ files
def scorer_code_sha256():
    """Hash this scorer and its transitive top-level imports in the round, RG, EG and RB folders (copied from
    `rg_score.scorer_code_sha256`, which starts from rg_score.py; same method). Names as well as contents are hashed;
    lazy imports inside functions are excluded."""
    folders = (rs_paths.ROUND, rg_paths.ROUND, Path(E.__file__).parent, Path(rb_ami.__file__).parent)
    pending = [rs_paths.ROUND / 'rs_sweep.py']
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
    """git HEAD, round-local Python changes and the scorer/dependency hash."""
    def git(*a):
        return subprocess.run(['git', '-C', str(rs_paths.ROUND), *a], capture_output=True, text=True, check=True).stdout
    return dict(code_commit=git('rev-parse', 'HEAD').strip(), code_dirty=bool(git('status', '--porcelain', '--', '*.py').strip()),
                code_sha256=scorer_code_sha256())


def _clean(record):
    return json.loads(json.dumps(record, allow_nan=False))


def score_path(outputs, machine, name, seed):
    assert machine in MACHINES, machine
    return Path(outputs) / 'scores' / machine / f'{name}__s{seed}.json'


def read_score(outputs, machine, name, seed):
    """A score file, after the checks every reader needs: kind, name, seed, machine, P's SHA-256."""
    path = score_path(outputs, machine, name, seed)
    assert path.exists(), f'{path} missing'
    rec = json.loads(path.read_text())
    assert rec['stage'] == 'score' and rec['name'] == name and rec['seed'] == seed, path
    assert rec['machine'] == machine, f'{path}: machine {rec["machine"]!r}, read as {machine!r}'
    assert rec['p_sha256'] == rs_paths.P_SHA256, f'{path}: P differs from 12-25\'s'
    return rec


def _keep_or_write(path, record, ignore=('code_commit', 'code_dirty', 'host')):
    """Write `record`; an existing file with the same values (up to the code commit, the dirty flag and the host) and the
    same scorer code is kept, any other one stops the run. Never overwrites or deletes."""
    path = Path(path)
    if path.exists():
        old = json.loads(path.read_text())
        a = {k: v for k, v in old.items() if k not in ignore}
        b = {k: v for k, v in record.items() if k not in ignore}
        assert a == b, f'{path} exists with other inputs, values or code; move it aside, then rerun'
        print(f'{path}: exists with the same inputs and values, kept', flush=True)
        return old
    _json_dump(path, record)
    return record


def run_score(outputs, machine, names, seed, smoke=False, frozen_g0=False):
    """{name: score record}: partitions and A on P of each run -> scores/<machine>/<name>__s<seed>.json. A failed run
    (status failed in its run.json) gets a record with no A. Finished scores of the same inputs are kept."""
    assert machine in MACHINES, machine
    assert len(set(names)) == len(names), names
    runs = {n: read_run(outputs, n, seed, machine, smoke, frozen_g0) for n in names}
    ctx = E.load_context()
    assert ctx['p_sha'] == rs_paths.P_SHA256, 'P differs from 12-25\'s'
    code = code_state()                                   # taken before the scoring: the record names the code that computed it
    out = {}
    for n, run in runs.items():
        A = img_sha = None
        if run['status'] == 'ok':
            A, _ = score_images(run['img'], partitions_dir(outputs, machine, n, seed), n, seed, ctx)
            img_sha = {a: rb_pick.array_sha256(run['img'][a]) for a in ASPECTS}
        rec = dict(stage='score', name=n, seed=seed, machine=machine, host=socket.gethostname(), source=run['source'],
                   smoke=bool(smoke), status=run['status'], cause=run['cause'], best_epoch=run['best_epoch'], A=A,
                   **({'rerun_of': run['rerun_of']} if 'rerun_of' in run else {}),
                   img_sha256=img_sha, run_sha256=run['run_sha256'], p_sha256=ctx['p_sha'],
                   label_sha256={a: rb_pick.array_sha256(ctx['labels'][a]) for a in ASPECTS},
                   versions=rb_ami.versions(), rule_commit=rs_paths.RULE_COMMIT, **code)
        out[n] = _keep_or_write(score_path(outputs, machine, n, seed), _clean(rec))
    return out


def load_table(outputs, machine, names, smoke=False, seed=rs_sheet.SWEEP_SEED):
    """{name: row} of the named runs' score files; stops with the list of the missing ones."""
    missing = [n for n in names if not score_path(outputs, machine, n, seed).exists()]
    assert not missing, f'scores missing on {machine}: {missing}'
    table = {}
    for n in names:
        rec = read_score(outputs, machine, n, seed)
        assert rec['smoke'] is bool(smoke), f'{n}: smoke flag differs'
        table[n] = {k: rec[k] for k in ('status', 'cause', 'A', 'best_epoch', 'source')}
    return table


def input_hashes(outputs, machine, names, seed=rs_sheet.SWEEP_SEED):
    return {n: _file_sha256(score_path(outputs, machine, n, seed)) for n in names}


def _row_out(j):
    return {k: j[k] for k in ('name', 'status', 'cause', 'A', 'gain', 'M', 'guard', 'counting', 'best_epoch')}


# ================================================================== stages 1 and 2 (files)
def sweep_path(outputs, machine, stage):
    return Path(outputs) / 'sweep' / machine / f'{stage}.json'


def stage1_names():
    return [rs_sheet.name(c) for c in rs_sheet.stage1()]


def decide_stage1(table):
    """The stage 1 decision from the table of G₀ and the 39 runs (pure)."""
    ref = table[G_NAME]
    assert ref['status'] == 'ok', f'G₀ failed ({ref["cause"]}): stage 1 has no reference'
    judged = judge_all({n: table[n] for n in stage1_names()}, ref['A'])
    gains = knob_gains(judged)
    knobs = grid_knobs(gains)
    level_of = {k: levels(judged, k) for k in knobs}
    split = split_points(grid_points(knobs, level_of)) if knobs else dict(order=[], g0=[], reused=[], new=[])
    kind = {n: k for k in ('g0', 'reused', 'new') for n in split[k]}
    return dict(judged=judged, ref=ref, knob_gains=gains, grid_knobs=knobs, outcome=stage1_outcome(knobs),
                levels=level_of, grid=dict(points=[dict(name=n, kind=kind[n]) for n in split['order']], **split))


REF_SOURCE = dict(local='12-27', node402='rs_trainer')    # G₀: 12-27's local run on the fallback (rule §4.2), the round's own on the node


def k11_gate(outputs):
    """None, or the failed k11 sample graph check (gates/k11_sample.json with pass false): rule §2.2, the k11 runs do not
    run, are not counting, and the cause is reported. A missing file is not a failure (the check has not run)."""
    path = Path(outputs) / 'gates' / 'k11_sample.json'
    if not path.exists():
        return None
    gate = json.loads(path.read_text())
    if gate['pass'] is True:
        return None
    return dict(cause=gate.get('cause') or 'the k = 20 rebuild differs from the frozen sample graphs (gates/k11_sample.json)',
                sha256=_file_sha256(path))


def stage1_table(outputs, machine, smoke):
    """({name: row}, {input: SHA-256}) of G₀ and the 39 runs. G₀ must be the machine's own kind of run (REF_SOURCE). When the
    k11 gate failed, k11's three runs are entered as failed (not counting) with its cause and the gate's SHA-256 is an input;
    a score file of such a run is refused."""
    gate = k11_gate(outputs)
    k11 = [rs_sheet.name(c) for c in rs_sheet.stage1() if rs_sheet.changed(c) == ('k11',)] if gate else []
    names = [n for n in [G_NAME] + stage1_names() if n not in k11]
    table = load_table(outputs, machine, names, smoke)
    assert table[G_NAME]['source'] == REF_SOURCE[machine], \
        f'G₀ on {machine} must come from {REF_SOURCE[machine]} (rule §4.2), not {table[G_NAME]["source"]}'
    inputs = input_hashes(outputs, machine, names)
    if gate:
        extra = [n for n in k11 if score_path(outputs, machine, n, 0).exists()]
        assert not extra, f'k11 gate failed but scores exist: {extra}'
        for n in k11:
            table[n] = dict(status='failed', cause=f'k11 sample graph check failed: {gate["cause"]}', A=None, best_epoch=None,
                            source=None)
        inputs['gates/k11_sample.json'] = gate['sha256']
    return {n: table[n] for n in [G_NAME] + stage1_names()}, inputs


def run_stage1(outputs, machine, smoke=False):
    """G₀ and the 39 runs of `machine` -> sweep/<machine>/stage1.json (rule §4.2 Stage 1). Stops when a run is missing."""
    code = code_state()
    table, inputs = stage1_table(outputs, machine, smoke)
    d = decide_stage1(table)
    judged = d['judged']
    record = dict(stage='stage1', machine=machine, smoke=bool(smoke), rule_commit=rs_paths.RULE_COMMIT,
                  ranges=dict(RANGES), reference=dict(name=G_NAME, source=d['ref']['source'], A=d['ref']['A'],
                                                      best_epoch=d['ref']['best_epoch']),
                  runs=[_row_out(judged[n]) for n in stage1_names()],
                  failed=[dict(name=n, cause=j['cause']) for n, j in judged.items() if j['status'] == 'failed'],
                  curves={k: knob_curve(judged, k) for k in rs_sheet.KNOBS}, knob_gains=d['knob_gains'],
                  grid_knobs=d['grid_knobs'], outcome=d['outcome'], levels=d['levels'], grid=d['grid'],
                  inputs=inputs, p_sha256=rs_paths.P_SHA256, **code)
    return _keep_or_write(sweep_path(outputs, machine, 'stage1'), _clean(record))


def run_stage2(outputs, machine, smoke=False):
    """The pool, TOP and the outcome -> sweep/<machine>/stage2.json (rule §4.2 Stage 2). Stops when stage 1 gave SWEEP-NONE,
    when stage 1's file is stale, or when a new grid point has no score."""
    s1_path = sweep_path(outputs, machine, 'stage1')
    assert s1_path.exists(), f'{s1_path} missing: run stage1 first'
    s1 = json.loads(s1_path.read_text())
    assert s1['machine'] == machine and s1['smoke'] is bool(smoke), f'{s1_path}: machine or smoke flag differs'
    assert s1['outcome'] != 'SWEEP-NONE', 'stage 1 gave SWEEP-NONE: stages 2, 3 and 4 do not run; G stays the recipe'
    code = code_state()
    new = s1['grid']['new']
    names = [G_NAME] + stage1_names() + new
    table1, now = stage1_table(outputs, machine, smoke)
    assert now == s1['inputs'], f'{s1_path} is stale: stage 1\'s score files changed; rerun stage1 (move the old file aside)'
    table = dict(table1, **load_table(outputs, machine, new, smoke))
    assert _clean({k: decide_stage1(table)[k] for k in ('outcome', 'grid_knobs', 'levels', 'grid')}) == \
        {k: s1[k] for k in ('outcome', 'grid_knobs', 'levels', 'grid')}, f'{s1_path} differs from the scores\' decision'
    judged = judge_all({n: table[n] for n in names}, table[G_NAME]['A'])
    members = pool(judged)
    chosen = top(judged, members)
    record = dict(stage='stage2', machine=machine, smoke=bool(smoke), rule_commit=rs_paths.RULE_COMMIT,
                  stage1_sha256=_file_sha256(s1_path), ranges=dict(RANGES),
                  new_runs=[_row_out(judged[n]) for n in new],
                  failed=[dict(name=n, cause=judged[n]['cause']) for n in new if judged[n]['status'] == 'failed'],
                  grid_table=[dict(kind=p['kind'], **(_row_out(judged[p['name']]) if p['name'] != G_NAME else dict(name=G_NAME)))
                              for p in s1['grid']['points']],
                  pool=members, top=[dict(rank=i + 1, **_row_out(judged[n])) for i, n in enumerate(chosen)],
                  outcome=stage2_outcome(members), inputs=dict(now, **input_hashes(outputs, machine, new)),
                  p_sha256=rs_paths.P_SHA256, **code)
    return _keep_or_write(sweep_path(outputs, machine, 'stage2'), _clean(record))


# ================================================================== the gates: V1a scorer half, N3
def frozen_g_scores(outputs, tag):
    """12-27's GS-0.02 seed-0 exports scored by this scorer, partitions cached under <outputs>/gates/<tag>/: ({aspect: A},
    {aspect: img sha256}, {aspect: partitions equal to 12-27's stage A cache}). The exports are the frozen run's, read
    through `read_run` (V1b, P's SHA-256)."""
    run = read_run(outputs, G_NAME, 0, 'local', False, frozen_g0=True)
    assert run['status'] == 'ok', 'the frozen G run is not ok'
    ctx = E.load_context()
    A, parts = score_images(run['img'], Path(outputs) / 'gates' / tag, rs_paths.G_ARM, 0, ctx)
    same = {}
    for a in ASPECTS:
        with np.load(rs_paths.RG_STAGE_A / 'partitions' / rs_paths.G_ARM / 's0' / f'{a}__partitions.npz', allow_pickle=False) as z:
            same[a] = bool(np.array_equal(z['labels'], parts[a]))
    return A, {a: rb_pick.array_sha256(run['img'][a]) for a in ASPECTS}, same, ctx


def stage_a_values():
    """12-27's stage A seed-0 A_a of GS-0.02 (RG/outputs/stage_a/verdict.json)."""
    path = rs_paths.RG_STAGE_A / 'verdict.json'
    return {a: json.loads(path.read_text())['results'][rs_paths.G_ARM]['effects'][a]['A'] for a in ASPECTS}, path


def run_check_v1a(outputs):
    """V1a, scorer half (rule §4.2): the scorer on RG_TRAIN/GS-0.02/s0 gives G's seed-0 A_a of stage A's verdict within
    1e-6 -> gates/v1a_scorer.json."""
    code = code_state()
    A, img_sha, same, ctx = frozen_g_scores(outputs, 'v1a_scorer_partitions')
    want, ref_path = stage_a_values()
    items = {}
    for a in ASPECTS:
        diff = abs(A[a] - want[a])
        items[f'G_{a}_A'] = dict(value=A[a], expected=want[a], diff=diff, **{'pass': bool(diff <= TOL)})
        print(f'{a}: A {A[a]:.10f} (12-27 {want[a]:.10f}, diff {diff:.1e}); partitions equal to 12-27\'s: {same[a]}', flush=True)
    record = dict(stage='v1a_scorer', **{'pass': all(i['pass'] for i in items.values())}, tol=TOL, items=items,
                  partitions_equal_to_12_27=same, reference=str(ref_path), reference_sha256=_file_sha256(ref_path),
                  img_sha256=img_sha, p_sha256=ctx['p_sha'], versions=rb_ami.versions(), host=socket.gethostname(),
                  rule_commit=rs_paths.RULE_COMMIT, **code)
    return _keep_or_write(Path(outputs) / 'gates' / 'v1a_scorer.json', _clean(record))


def run_n3(outputs, machine='node402', smoke=False):
    """N3 (rule §4.2 "The node route") -> gates/n3.json: G₀ and G₀′ pass V1b (their scores are status ok) and
    |A_a(G₀′) - A_a(G₀)| <= r_a / 2 per aspect. Readings, no gate: G₀ minus 12-27's local seed-0 A (trainer parity); this
    scorer on 12-27's exports minus 12-27's A (scorer parity: run this on the node's CPU, the record names the host);
    whether G₀′ equals G₀ bit for bit (exports)."""
    assert machine == 'node402', 'N3 is the node route\'s check'
    code = code_state()
    g0, g0p = (read_score(outputs, machine, n, 0) for n in (G_NAME, REPEAT))
    for rec in (g0, g0p):
        assert rec['source'] == 'rs_trainer' and rec['smoke'] is bool(smoke), (rec['name'], rec['source'])
    v1b = {rec['name']: rec['status'] == 'ok' for rec in (g0, g0p)}
    items = n3_items(g0['A'], g0p['A']) if all(v1b.values()) else {}
    want, ref_path = stage_a_values()
    scorer_A, _, _, ctx = frozen_g_scores(outputs, 'n3_scorer_parity_partitions')
    bit = {a: g0['img_sha256'][a] == g0p['img_sha256'][a] for a in ASPECTS} if all(v1b.values()) else None
    readings = dict(trainer_parity={a: g0['A'][a] - want[a] for a in ASPECTS} if g0['status'] == 'ok' else None,
                    scorer_parity={a: scorer_A[a] - want[a] for a in ASPECTS}, scorer_host=socket.gethostname(),
                    g0_repeat_equals_g0=dict(exports=bit, all=None if bit is None else all(bit.values()),
                                             best_epoch=g0['best_epoch'] == g0p['best_epoch']))
    record = dict(stage='n3', machine=machine, smoke=bool(smoke),
                  **{'pass': bool(all(v1b.values()) and all(i['pass'] for i in items.values()))}, v1b=v1b, items=items,
                  readings=readings, reference=str(ref_path), reference_sha256=_file_sha256(ref_path),
                  inputs=input_hashes(outputs, machine, [G_NAME, REPEAT]), p_sha256=ctx['p_sha'],
                  versions=rb_ami.versions(), host=socket.gethostname(), rule_commit=rs_paths.RULE_COMMIT, **code)
    return _keep_or_write(Path(outputs) / 'gates' / 'n3.json', _clean(record))


# ================================================================== CLI
def main(argv=None):
    parser = argparse.ArgumentParser(description='Stages 1 and 2 of the recipe sweep: scorer, gains, grid, TOP.')
    parser.add_argument('mode', choices=['score', 'stage1', 'stage2', 'n3', 'check'])
    parser.add_argument('what', nargs='?', choices=['v1a'], help='for `check`')
    parser.add_argument('--machine', choices=MACHINES)
    parser.add_argument('--runs', nargs='+', default=[], metavar='NAME', help='run folder names (G, G-repeat, k6=0.0005, ...)')
    parser.add_argument('--seed', type=int, default=rs_sheet.SWEEP_SEED)
    parser.add_argument('--frozen-g0', action='store_true',
                        help='score 12-27\'s GS-0.02 seed-0 run as G (the local fallback\'s G₀, rule §4.2)')
    parser.add_argument('--outputs', type=Path, default=rs_paths.OUTPUTS)
    parser.add_argument('--smoke', action='store_true', help='outputs under /tmp/claude-0/; reads smoke runs')
    args = parser.parse_args(argv)
    assert (args.mode == 'check') == (args.what is not None), 'usage: score | stage1 | stage2 | n3 | check v1a'
    assert (args.mode in ('score', 'stage1', 'stage2', 'n3')) == (args.machine is not None), '--machine goes with score, stage1, stage2, n3'
    assert bool(args.runs) == (args.mode == 'score'), '--runs goes with score'
    assert not args.frozen_g0 or args.mode == 'score', '--frozen-g0 goes with score'
    outputs = rs_paths.check_outputs(args.outputs, args.smoke)
    rs_paths.check_imports()
    if args.mode == 'score':
        out = run_score(outputs, args.machine, args.runs, args.seed, args.smoke, args.frozen_g0)
        for n, rec in out.items():
            print(f'{score_path(outputs, args.machine, n, args.seed)}: {rec["status"]}' +
                  (f', A {rec["A"]}' if rec['A'] else f' ({rec["cause"]})'))
    elif args.mode == 'stage1':
        rec = run_stage1(outputs, args.machine, args.smoke)
        print(f"{sweep_path(outputs, args.machine, 'stage1')}: {rec['outcome']}, grid knobs {rec['grid_knobs']}, "
              f"{len(rec['grid']['new'])} new points, {len(rec['failed'])} failed runs")
    elif args.mode == 'stage2':
        rec = run_stage2(outputs, args.machine, args.smoke)
        print(f"{sweep_path(outputs, args.machine, 'stage2')}: {rec['outcome']}, pool {len(rec['pool'])}, "
              f"top {[t['name'] for t in rec['top']]}")
    elif args.mode == 'n3':
        rec = run_n3(outputs, args.machine, args.smoke)
        print(f"{outputs / 'gates/n3.json'}: N3 {'PASS' if rec['pass'] else 'FAIL'}")
        return 0 if rec['pass'] else 1
    else:
        assert not args.smoke, '`check v1a` has no smoke'
        rec = run_check_v1a(outputs)
        print(f"{outputs / 'gates/v1a_scorer.json'}: V1a scorer half {'PASS' if rec['pass'] else 'FAIL'}; "
              f"{sum(i['pass'] for i in rec['items'].values())} of {len(rec['items'])} items within {TOL}")
        return 0 if rec['pass'] else 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
