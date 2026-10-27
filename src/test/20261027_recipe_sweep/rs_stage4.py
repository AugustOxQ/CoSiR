"""Stage 4's scorer of the recipe sweep (rule §3.4, §4.4): the candidate's held outcomes, d_T per aspect against the
instructed image T, the emotion margin over PercepT (PE-*), the reported margins; the scorer half of V4a. CPU.

The candidate is c*, the `pick` of `stage3/verdict.json`, or G where no stage 3 file is written (rule §4.4, last
paragraph: stage 1 SWEEP-NONE, stage 2 TOP-NONE) or stage 3 gave KEEP-G. Its held outputs are `full_root(out)/held/<name>/
{aspect}__s{seed}.npz` (ticket 08; G's are 12-27's `RG_FULL/held/GS-0.02/`).

Statistics (all 12,281 rows of H, labels < 0 masked inside the AMI, 2,000 draws, a `default_rng(0)` per statistic):
  d_T,a    A_c(I0, S0) - A_T,a(I0) with 12-27's kernels (`rg_held.draws_d_t`, `bootstrap_d_t`, `read_frozen_partitions`)
           for each aspect: I_b = integers(0, 12281, size=12281), then S_b = choice(0..4, size=5) for c; T draws nothing.
           T-WIN lower bound > 0; T-LEVEL not T-WIN and lower bound > -r_a; T-BELOW otherwise.
  PE-*     the emotion margin over B1 to B6 at full scale through stage 0's adapter (`rs_stage0.score_scale`, c in G's place).
  reported d_G,a (c minus G; I_b, then one S_b shared by both), margins against each of B1 to B6, B1 to B6 together,
           Leiden on genre. Computed only after `stage4/verdict.json` exists.

Files (under --outputs, default `rs_paths.OUTPUTS`):
  gates/v4a_scorer.json              the scorer check (pass, smoke, items, code_sha256, max_abs_diff)
  stage4/partitions/full/<name>/     the candidate's fixed r20 partitions (G's, for the reported d_G, under .../G/)
  stage4/verdict.json                the outcomes per aspect and the emotion margin (written once)
  stage4/reported.json, leiden/      the reported block

CLI (`python rs_stage4.py ...`):
  check v4a [--smoke]    G's saved held outputs in c's place against 12-27's `stage_c/verdict.json` (genre) and
                         `reported.json` (emotion and style, margin "T") within 1e-6 (a smoke: 20 draws, point values only)
  verdict --candidate NAME   stage4/verdict.json; refuses unless NAME is c* (or G where G is the candidate)
  report                 stage4/reported.json, only after the verdict file exists
"""
import argparse
import ast
import importlib.util
import json
import multiprocessing
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import rs_paths as paths
import rs_sheet as sheet
import rs_stage0 as S0
import rs_sweep
import rb_part2_report
import rb_score
import rg_held

ASPECTS = tuple(paths.ASPECTS)
SEEDS = S0.SEEDS                           # (0, 1, 2, 3, 4)
BASELINES = S0.BASELINES                   # B1 to B6
RANGES = sheet.RANGES                      # r_a (rule §3)
N_RESAMPLES = rg_held.N_RESAMPLES          # 2000
TOL = S0.TOL                               # V4a: 1e-6
HELD_SHA256, N_HELD = paths.HELD_SHA256, paths.N_HELD
PE_PREFIX = {'full': 'PE'}   # PE-GO, PE-MM, PE-IN, PE-NO (rule §4.4: the thresholds of §4.1)
G_NAME = 'G'
Refusal = S0.Refusal
# Report/verdict parity binds scoring, labels and partitions; the current V4a gate still binds the full closure.
SCORER_FILES = ('rs_stage4.py', 'rs_stage0.py', 'rs_sheet.py', 'rs_paths.py', 'rg_held.py', 'rb_score.py',
                'rb_part2_report.py', 'ct_held.py', 'ct_score.py', 'ct_diag.py', 'pav_score.py')


# ================================================================== the rule's outcome logic (pure)
def t_outcome(aspect, lower):
    """T-WIN: lower bound of d_T,a > 0. T-LEVEL: not T-WIN and lower > -r_a. T-BELOW otherwise (unrounded values)."""
    assert np.isfinite(lower)
    r = RANGES[aspect]
    if lower > 0:
        return 'T-WIN'
    return 'T-LEVEL' if lower > -r else 'T-BELOW'


def pe_outcome(d, lower):
    return S0.outcome_id(d, lower, 'full', PE_PREFIX)


# ================================================================== the draws and the statistics
def draws_d_g(n_rows, n_resamples=N_RESAMPLES, seeds=SEEDS):
    """d_G: its own `default_rng(0)`; per draw I_b, then one S_b shared by c ('C') and G (rule §3.4 "Reported")."""
    rng = np.random.default_rng(0)
    draws = []
    for _ in range(n_resamples):
        rows = rng.integers(0, n_rows, size=n_rows)
        s = rng.choice(list(seeds), size=len(seeds), replace=True)
        draws.append((rows, {'C': s, 'G': s}))
    return draws


def d_t_block(fixed_c, fixed_t, labels, aspect, n_resamples=N_RESAMPLES, workers=8):
    """d_T,a with its interval and outcome. fixed_c = {seed: {'r20': three partitions}} for seeds 0 to 4, fixed_t =
    {0: {'r20': ...}}; labels < 0 are masked by the statistic (`rb_score.score_indices`)."""
    res = rg_held.bootstrap_d_t(fixed_c, fixed_t, labels, n_resamples, workers)
    lo, hi = res['interval']
    return dict(d=res['d'], lo=lo, hi=hi, A_cand=res['A_G'], A_T=res['A_T'], d_in_ranges=res['d'] / RANGES[aspect],
                outcome=t_outcome(aspect, lo), labelled_rows=int((labels >= 0).sum()),
                labels_masked=int((labels < 0).sum()), per_seed=res['per_seed'], bootstrap_d=res['bootstrap_d'])


def d_g_block(fixed_c, fixed_g, labels, n_resamples=N_RESAMPLES, workers=8):
    """d_G = A_c - A_G over seeds 0 to 4 of both, with `draws_d_g`."""
    for fixed in (fixed_c, fixed_g):
        rg_held._check_fixed(fixed, SEEDS, len(labels), 'd_G')
    fixed = {'C': fixed_c, 'G': fixed_g}
    seeds = {arm: list(SEEDS) for arm in fixed}
    point = rb_score.score_indices(fixed, labels, np.arange(len(labels)), seeds, ('C', 'G'))
    draws = draws_d_g(len(labels), n_resamples)
    if workers == 1:
        diffs = [rb_score.score_indices(fixed, labels, rows, s, ('C', 'G'))['d'] for rows, s in draws]
    else:
        with multiprocessing.Pool(workers, initializer=rb_score._init_worker,
                                  initargs=(fixed, labels, ('C', 'G'))) as pool:
            diffs = pool.map(rb_score._draw_difference, draws)
    lo, hi = np.percentile(diffs, [2.5, 97.5]).tolist()
    return dict(d=point['d'], lo=lo, hi=hi, A_cand=point['A_cand'], A_G=point['A_e']['G-r20'],
                bootstrap_d=[float(d) for d in diffs])


def _margin_out(m):
    return dict(d=m['d'], lo=m['interval'][0], hi=m['interval'][1], A_cand=m['A_cand'], A_e=m['A_e'],
                max_entry=m['max_entry'], arms=m['arms'])


def per_baseline_margins(fixed, labels, eligible, label, n_resamples=N_RESAMPLES, workers=8):
    """The candidate's margin against each of B1 to B6 alone (12-24's `margin`: d = A_c minus the larger of the
    baseline's r20 and own readings; its own `default_rng(0)` over (c, Bk)); None for a baseline with no eligible seed.
    Single-baseline margins beyond B1 (`b_matched`) are extra readings, deciding nothing."""
    result = {}
    for arm in BASELINES:
        if not eligible[arm]:
            result[arm] = None
            continue
        m = rb_part2_report.margin(fixed, labels, eligible, (label, arm), n_resamples, workers)
        result[arm] = dict(_margin_out(m), eligible_seeds=list(eligible[arm]))
    return result


# ================================================================== files
sha_file = S0.sha_file


def _clean(value):
    return S0._clean(value)


def _git_state():
    return S0._git_state()


def _under_src_test(path):
    parts = Path(path).resolve().parts
    return any(a == 'src' and b == 'test' for a, b in zip(parts, parts[1:]))


def _imported_names(path):
    """Top-level module names imported anywhere in a file (also inside functions: lazy imports count)."""
    names = set()
    for node in ast.walk(ast.parse(Path(path).read_text())):
        if isinstance(node, ast.Import):
            names |= {a.name.split('.')[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split('.')[0])
    return names


def code_sha256():
    """{file name: SHA-256} of rs_stage4.py and every module it imports (directly, through the ones it imports, and
    lazily inside functions) whose file is under src/test/, plus stage 0's `CODE_MODULES` (loaded by name there).
    By static imports, not `sys.modules`, so the value does not depend on what else the process has loaded. The same
    function stamps `gates/v4a_scorer.json`."""
    todo, seen = ['rs_stage4', 'rg_held', 'rb_part2_report', 'ct_held', 'ct_diag', *S0.CODE_MODULES], {}
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        spec = importlib.util.find_spec(name)
        if spec is None or not spec.has_location or not str(spec.origin).endswith('.py') or not _under_src_test(spec.origin):
            seen[name] = None
            continue
        seen[name] = Path(spec.origin)
        todo += sorted(_imported_names(spec.origin))
    files = [p for p in seen.values() if p is not None]
    result = {p.name: sha_file(p) for p in files}
    assert len(result) == len(files), 'two files with one name'
    return dict(sorted(result.items()))


def gate_path(outputs, name='v4a_scorer'):
    return S0.gate_path(outputs, name)


def scorer_sha256(stamp=None):
    stamp = code_sha256() if stamp is None else stamp
    return {name: stamp[name] for name in SCORER_FILES}


def require_v4a_scorer(outputs):
    """The scorer half of V4a (ticket 08's `require_v4a` calls this): the gate passes, is not a smoke, and was written
    by the code on disk now. Returns the gate file's SHA-256."""
    path = gate_path(outputs)
    if not path.is_file():
        raise Refusal(f'V4a scorer gate {path} is missing: run `rs_stage4 check v4a` first')
    record = json.loads(path.read_text())
    if record.get('pass') is not True:
        raise Refusal(f'V4a scorer gate {path} does not pass')
    if record.get('smoke') is not False:
        raise Refusal(f'V4a scorer gate {path} is a smoke run')
    if record.get('code_sha256') != code_sha256():
        raise Refusal(f'V4a scorer gate {path} was written by other code than is on disk now: rerun `check v4a`')
    return sha_file(path)


# ================================================================== the data
def held_labels():
    """{aspect: int labels over H}: emotion by the CT's `painting_labels` in annotation order, style and genre the first
    caption's label, < 0 unlabelled (rule §3.4)."""
    names, sha, labels = rb_part2_report.held_labels()
    assert sha == HELD_SHA256 and len(names) == N_HELD
    assert int((labels['genre'] >= 0).sum()) == rg_held.N_LABELLED
    return labels


def t_fixed(aspect):
    """({0: {'r20': T_a's three partitions}}, source record): 12-24's frozen r20 partitions of the instructed image."""
    path = paths.RB_PART2 / f'partitions/full/T/{aspect}__s0__r20.npz'
    return {0: {'r20': rg_held.read_frozen_partitions(path)}}, dict(path=str(path), sha256=sha_file(path))


def candidate(outputs, name):
    """The held outputs of c* (ticket 08's layout; the sidecars name the run) or of G (12-27's)."""
    if name == G_NAME:
        return S0.FULL_G
    return S0.Candidate(name, paths.full_root(outputs) / 'held' / name, name)


def score_aspect(cand, aspect, labels, cache, n_resamples=N_RESAMPLES, workers=8, recorded=None):
    """The same per-aspect read and statistic for the held verdict and V4a's G replay."""
    fixed_c, src = S0.candidate_fixed(cand, 'full', cache, aspect, recorded=recorded)
    fixed_t, t_src = t_fixed(aspect)
    res = d_t_block(fixed_c, fixed_t, labels[aspect], aspect, n_resamples, workers)
    return res, fixed_c, dict(candidate_outputs=src, T_partitions=t_src)


def stage_candidate(outputs):
    """Who stage 4 scores: {name, kind, upstream: {file, sha256, outcome}}. `stage3/verdict.json` decides (PICK: its
    `pick`; KEEP-G: G). Where that file does not exist, the candidate is G only when stage 1 gave SWEEP-NONE or stage 2
    TOP-NONE on every machine with a stage 1 file, using `rs_full.sweep_ended_without_candidates`."""
    import rs_full
    outputs = Path(outputs)
    stage3 = outputs / 'stage3/verdict.json'
    if stage3.is_file():
        v = json.loads(stage3.read_text())
        if v.get('smoke') is not False:
            raise Refusal(f'{stage3} is a smoke record (smoke={v.get("smoke")!r}): stage 4 reads the real one')
        if v.get('outcome') == 'PICK' and isinstance(v.get('pick'), str) and v['pick']:
            try:
                rs_full.require_stage3(outputs, sheet.parse(v['pick']))
            except (AssertionError, ValueError, KeyError) as error:
                raise Refusal(f'{stage3}: {error}') from error
            name, kind = v['pick'], 'PICK'
        elif v.get('outcome') == 'KEEP-G' and v.get('pick') is None:
            name, kind = G_NAME, 'KEEP-G'
        else:
            raise Refusal(f'{stage3}: outcome {v.get("outcome")!r} with pick {v.get("pick")!r} is neither PICK with a '
                          f'run name nor KEEP-G with no pick')
        return dict(name=name, kind=kind, upstream=dict(file='stage3/verdict.json', sha256=sha_file(stage3),
                                                        outcome=v['outcome'], pick=v['pick']))
    ended = rs_full.sweep_ended_without_candidates(outputs)
    if not ended:
        raise Refusal(f'{stage3} is missing and not every machine ended in SWEEP-NONE or TOP-NONE')
    machines = {}
    for machine, outcome in ended.items():
        stage = 'stage1' if outcome == 'SWEEP-NONE' else 'stage2'
        path = rs_sweep.sweep_path(outputs, machine, stage)
        machines[machine] = dict(file=str(path.relative_to(outputs)), sha256=sha_file(path), outcome=outcome)
    kind = 'SWEEP-NONE' if all(v == 'SWEEP-NONE' for v in ended.values()) else 'TOP-NONE'
    return dict(name=G_NAME, kind=kind, upstream=dict(next(iter(machines.values())), machines=machines))


# ================================================================== the verdict
def _keep_or_write(path, record):
    path = Path(path)
    record = _clean(record)
    if path.exists():
        saved = json.loads(path.read_text())
        assert saved == record, f'{path}: refusing to replace a different record'
        return saved
    S0._atomic_json(path, record)
    return record


def pick_validity(outputs, cand):
    """PICK requires both V4a halves and V4c. Terminal seed failures leave all held claims without an outcome."""
    import rs_full
    try:
        gates = rs_full.require_v4a(outputs)
        rs_full.refuse_second_candidate(paths.full_root(outputs), cand.label, sheet.parse(cand.label))
    except AssertionError as error:
        raise Refusal(str(error)) from error
    failures = {}
    for seed in SEEDS:
        files = [rs_full.rg_full.failed_path(paths.full_root(outputs), cand.label, seed)]
        files += [S0.failed_path(cand.held_dir, aspect, seed) for aspect in ASPECTS]
        for path in files:
            if path.is_file():
                failures[str(path)] = dict(sha256=sha_file(path), record=json.loads(path.read_text()))
    path = cand.held_dir / 'v4c.json'
    v4c = json.loads(path.read_text()) if path.is_file() else None
    if v4c is not None:
        gates['v4c'] = sha_file(path)
    if failures:
        return gates, None, f'V4d: no outcome after terminal seed failure: {failures}', failures
    if v4c is None:
        raise Refusal(f'V4c record {path} is missing')
    if v4c.get('pass') is not True:
        return gates, None, f'V4c: no outcome: {v4c}', {}
    expected = {f'{a}__s{s}' for a in ASPECTS for s in SEEDS}
    if (v4c.get('name'), v4c.get('rows'), v4c.get('list_sha256')) != (cand.label, N_HELD, HELD_SHA256):
        raise Refusal(f'V4c: {path} does not name this candidate and H')
    if set(v4c.get('outputs', {})) != expected or set(v4c.get('output_files', {})) != expected:
        raise Refusal(f'V4c: {path} does not bind all five seeds of every aspect')
    recorded = {a: {str(s): v4c['output_files'][f'{a}__s{s}'] for s in SEEDS} for a in ASPECTS}
    return gates, recorded, None, {}


def verdict(outputs, name, n_resamples=N_RESAMPLES, workers=8):
    """stage4/verdict.json: the gated per-aspect d_T and emotion margin, or no outcome with the V4d cause.
    Written once: a second call must agree."""
    outputs = Path(outputs)
    assert n_resamples == N_RESAMPLES or paths.smoke_ok(outputs), 'a held verdict requires 2,000 draws outside scratch'
    gate_sha = require_v4a_scorer(outputs)
    who = stage_candidate(outputs)
    if name != who['name']:
        raise Refusal(f'{name!r} is not c*: {who["upstream"]["file"]} names {who["name"]!r} ({who["kind"]}); '
                      f'`verdict --candidate` takes only that candidate')
    cand = candidate(outputs, name)
    gates, recorded, failure, failures = dict(v4a_scorer=gate_sha), None, None, {}
    if who['kind'] == 'PICK':
        gates, recorded, failure, failures = pick_validity(outputs, cand)
    aspects, sources = {}, {}
    if failure is not None:
        aspects = {a: dict(outcome=None, v_failure=failure) for a in ASPECTS}
        margin = dict(outcome=None, v_failure=failure)
    else:
        labels = held_labels()
        cache = outputs / 'stage4/partitions'
        # The emotion margin also writes the partitions that d_T,emotion reads.
        block = S0.score_scale(cand, 'full', cache, n_resamples, workers,
                               recorded=None if recorded is None else recorded['emotion'], prefix=PE_PREFIX)
        for aspect in ASPECTS:
            res, _, src = score_aspect(cand, aspect, labels, cache / 'full' / cand.label, n_resamples, workers,
                                       recorded=None if recorded is None else recorded[aspect])
            res.pop('bootstrap_d')
            res.pop('per_seed')
            aspects[aspect] = res
            sources[aspect] = src
        entries = block['A_e']
        margin = dict(d=block['d'], lo=block['interval'][0], hi=block['interval'][1], A_cand=block['A_cand'], A_e=entries,
                      max_entry=max(entries, key=entries.get), outcome=block['outcome_id'], arms=block['arms'],
                      eligible_seeds=block['eligibility']['eligible_seeds'])
    stamp = code_sha256()
    record = dict(schema=1, stage=4, rule_commit=paths.RULE_COMMIT, held_sha256=HELD_SHA256, candidate=name,
                  kind=who['kind'], n_resamples=n_resamples, smoke=n_resamples != N_RESAMPLES,
                  ranges=dict(RANGES), aspects=aspects,
                  emotion_margin=margin, sources=dict(upstream=who['upstream'], by_aspect=sources, failures=failures),
                  gates=gates, code_sha256=stamp, scorer_sha256=scorer_sha256(stamp), v_failure=failure,
                  v4b='pending (ticket 11, the re-derivation from the rule)')
    return _keep_or_write(outputs / 'stage4/verdict.json', record)


# ================================================================== the reported block (after the outcomes are written)
def report_blocks(label, fixed_by_aspect, comp_fixed, eligible, labels, verdict_margin, g_fixed_by_aspect,
                  n_resamples=N_RESAMPLES, workers=8):
    """Per aspect: the margin against B1 to B6 together ('vs_B1_to_B6'; emotion's is the verdict's) and against each
    alone ('per_baseline'), and d_G,a when the candidate is not G (`g_fixed_by_aspect` is None then)."""
    out = {}
    arms = (label,) + BASELINES
    for aspect in ASPECTS:
        fixed = {**comp_fixed, label: fixed_by_aspect[aspect]}
        y = labels[aspect]
        block = dict(labels_masked=int((y < 0).sum()))
        if aspect == 'emotion':
            block['vs_B1_to_B6'] = verdict_margin
        else:
            block['vs_B1_to_B6'] = _margin_out(rb_part2_report.margin(fixed, y, eligible, arms, n_resamples, workers))
        block['per_baseline'] = per_baseline_margins(fixed, y, eligible, label, n_resamples, workers)
        block['b_matched'] = block['per_baseline']['B1']
        block['d_G'] = (None if g_fixed_by_aspect is None else
                        {k: v for k, v in d_g_block(fixed_by_aspect[aspect], g_fixed_by_aspect[aspect], y,
                                                    n_resamples, workers).items() if k != 'bootstrap_d'})
        out[aspect] = block
    return out


def genre_leiden(outputs, cand, eligible, labels, n_resamples, workers):
    """Leiden on genre as 12-27's readings (`rg_held._leiden_g` and the frozen comparators' Leiden partitions): the
    candidate against B1 to B6 together, `rb_part2_report.margin(..., lenient=True)`."""
    sources = rg_held._comparator_sources(eligible)
    comp_leiden = rg_held._leiden_comparators(eligible, sources, HELD_SHA256, N_HELD)
    cand_leiden = {}
    for seed in SEEDS:
        x, sha = S0.load_candidate(cand, 'full', 'genre', seed)
        cand_leiden[seed] = {'r20': rg_held._leiden_g(x, outputs / f'stage4/leiden/{cand.label}/genre__s{seed}.npz',
                                                      rg_held._leiden_identity(HELD_SHA256, sha))}
    fixed = {**comp_leiden, cand.label: cand_leiden}
    arms = (cand.label,) + BASELINES
    return _margin_out(rb_part2_report.margin(fixed, labels['genre'], eligible, arms, n_resamples, workers, lenient=True))


def report(outputs, n_resamples=N_RESAMPLES, workers=8):
    """stage4/reported.json, only after stage4/verdict.json exists (and the gate holds for this code)."""
    outputs = Path(outputs)
    vpath = outputs / 'stage4/verdict.json'
    if not vpath.is_file():
        raise Refusal(f'{vpath} is missing: the reported block runs only after the verdict file exists')
    require_v4a_scorer(outputs)
    v = json.loads(vpath.read_text())
    stamp = code_sha256()
    if scorer_sha256(v['code_sha256']) != scorer_sha256(stamp):
        raise Refusal(f'{vpath} was written by other scorer code than is on disk now')
    if v.get('v_failure') is not None:
        raise Refusal(f'{vpath}: no outcome: {v["v_failure"]}')
    cand = candidate(outputs, v['candidate'])
    labels = held_labels()
    cache = outputs / 'stage4/partitions'
    rcol, rcol_path = rb_score.read_rcol_eligible('full')
    eligible = rb_score.eligible_seeds((cand.label,) + BASELINES, rcol)
    assert eligible == v['emotion_margin']['eligible_seeds'], 'eligible seeds differ from the verdict\'s'
    comp_fixed = rg_held._frozen_comparator_partitions({a: eligible[a] for a in BASELINES})
    fixed_c = {a: S0.candidate_fixed(cand, 'full', cache / 'full' / cand.label, a)[0] for a in ASPECTS}
    is_g = cand.label == G_NAME
    fixed_g = None if is_g else {a: S0.candidate_fixed(S0.FULL_G, 'full', cache / 'full' / G_NAME, a)[0]
                                 for a in ASPECTS}
    aspects = report_blocks(cand.label, fixed_c, comp_fixed, eligible, labels, v['emotion_margin'], fixed_g,
                            n_resamples, workers)
    aspects['genre']['leiden_vs_B1_to_B6'] = genre_leiden(outputs, cand, eligible, labels, n_resamples, workers)
    record = dict(schema=1, stage=4, candidate=v['candidate'], held_sha256=HELD_SHA256, n_resamples=n_resamples,
                  aspects=aspects, d_G_note=('the candidate is G: d_G is None' if is_g else
                                             'd_G,a = c minus G (12-27\'s held outputs), one S_b shared'),
                  inputs=dict(verdict_sha256=sha_file(vpath), rcol=str(rcol_path), rcol_sha256=sha_file(rcol_path)),
                  code_sha256=stamp, verdict_code_sha256=v['code_sha256'], scorer_sha256=scorer_sha256(stamp))
    return _keep_or_write(outputs / 'stage4/reported.json', record)


# ================================================================== the scorer check (V4a scorer half)
def frozen_d_t():
    """12-27's G values as {aspect: {d, interval, A_G, A_T, [bootstrap_d]}}: genre from `stage_c/verdict.json` (d_T),
    emotion and style from `stage_c/reported.json`'s margin "T" (the margin against the instructed image)."""
    verdict = json.loads(S0.G_FULL_VERDICT.read_text())
    assert verdict['arm'] == paths.G_ARM and verdict['held_sha256'] == HELD_SHA256
    d = verdict['d_T']
    result = {'genre': dict(d=d['d'], interval=d['interval'], A_G=d['A_G'], A_T=d['A_T'],
                            bootstrap_d=d['bootstrap_d'])}
    reported = json.loads(S0.G_FULL_RECORD.read_text())
    assert reported['arm'] == paths.G_ARM and reported['held_sha256'] == HELD_SHA256
    for aspect in ('emotion', 'style'):
        m = reported['G']['aspects'][aspect]['margins']['T']
        assert m['arms'] == [paths.G_ARM, 'T'] and m['seedless'] == ['T'] and m['n_resamples'] == N_RESAMPLES
        result[aspect] = dict(d=m['d'], interval=m['interval'], A_G=m['A_cand'], A_T=m['A_e']['T-r20'])
    return result


def frozen_partitions_equal(aspect, fixed):
    """The candidate's partitions (G's held outputs) equal 12-27's cached ones, per seed."""
    equal = {}
    for seed in SEEDS:
        with np.load(paths.RG_STAGE_C / f'partitions/{paths.G_ARM}/{aspect}__s{seed}__r20.npz', allow_pickle=False) as z:
            equal[str(seed)] = bool(np.array_equal(z['labels'], fixed[seed]['r20']))
    return {'pass': all(equal.values()), 'equal': equal}


def compare_d_t(aspect, ours, theirs, point_only=False):
    """Items of one aspect against 12-27's values: d, both bounds, A_cand and A_T within 1e-6, and every bootstrap d
    where 12-27 saved them (genre). `point_only` (a smoke) leaves out what depends on the draws."""
    items = {f'd_T/{aspect}/d': S0._close(ours['d'], theirs['d']),
             f'd_T/{aspect}/A_cand': S0._close(ours['A_cand'], theirs['A_G']),
             f'd_T/{aspect}/A_T': S0._close(ours['A_T'], theirs['A_T'])}
    if not point_only:
        items[f'd_T/{aspect}/lo'] = S0._close(ours['lo'], theirs['interval'][0])
        items[f'd_T/{aspect}/hi'] = S0._close(ours['hi'], theirs['interval'][1])
        if 'bootstrap_d' in theirs:
            a, b = np.asarray(ours['bootstrap_d']), np.asarray(theirs['bootstrap_d'])
            same = a.shape == b.shape
            diff = float(np.abs(a - b).max()) if same else float('inf')
            items[f'd_T/{aspect}/bootstrap_d'] = {'pass': bool(same and diff < TOL), 'abs_diff': diff,
                                                  'n_ours': len(a), 'n_theirs': len(b)}
    return items


def check_v4a(outputs, workers=8, n_resamples=N_RESAMPLES, smoke=False):
    """The scorer half of V4a: G's saved held outputs in c*'s place give 12-27's d_T,genre (`stage_c/verdict.json`) and
    d_T,emotion and d_T,style (`stage_c/reported.json`, margin "T"), with intervals, within 1e-6. A smoke runs the
    same code on `n_resamples` draws and compares point values only; `require_v4a_scorer` refuses it."""
    assert smoke or n_resamples == N_RESAMPLES, 'a regression check is never shrunk: --smoke only'
    outputs = Path(outputs)
    labels = held_labels()
    frozen = frozen_d_t()
    cache = outputs / 'stage4/regression/partitions/full' / G_NAME
    items, ours_all = {}, {}
    for aspect in ASPECTS:
        ours, fixed_c, _ = score_aspect(S0.FULL_G, aspect, labels, cache, n_resamples, workers)
        items[f'partitions/{aspect}'] = frozen_partitions_equal(aspect, fixed_c)
        ours_all[aspect] = {k: v for k, v in ours.items() if k not in ('bootstrap_d', 'per_seed')}
        items.update(compare_d_t(aspect, ours, frozen[aspect], point_only=smoke))
    diffs = [i['abs_diff'] for i in items.values() if 'abs_diff' in i]
    record = dict(name='v4a_scorer', items=items, ours=ours_all,
                  n_resamples=n_resamples, smoke=bool(smoke), max_abs_diff=max(diffs), tolerance=TOL,
                  held_sha256=HELD_SHA256, code_sha256=code_sha256(), **_git_state(),
                  sources=dict(verdict=str(S0.G_FULL_VERDICT), verdict_sha256=sha_file(S0.G_FULL_VERDICT),
                               reported=str(S0.G_FULL_RECORD), reported_sha256=sha_file(S0.G_FULL_RECORD)))
    record['pass'] = all(i['pass'] for i in items.values())
    record['failing'] = sorted(k for k, v in items.items() if not v['pass'])
    S0._atomic_json(gate_path(outputs), record)
    return _clean(record)


# ================================================================== CLI
@threadpool_limits.wrap(limits=8)
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--outputs', type=Path, default=paths.OUTPUTS)
    parser.add_argument('--workers', type=int, default=8, choices=range(1, 9))
    sub = parser.add_subparsers(dest='command', required=True)
    c = sub.add_parser('check')
    c.add_argument('gate', choices=['v4a'])
    c.add_argument('--smoke', action='store_true', help='20 draws, point values only; outputs under /tmp/claude-0/')
    v = sub.add_parser('verdict')
    v.add_argument('--candidate', required=True, help='c* (the stage 3 pick), or G where G is the candidate')
    sub.add_parser('report')
    args = parser.parse_args(argv)
    smoke = getattr(args, 'smoke', False)
    outputs = paths.check_outputs(args.outputs, smoke)
    paths.check_imports()
    started = time.monotonic()
    try:
        if args.command == 'check':
            record = check_v4a(outputs, args.workers, 20 if smoke else N_RESAMPLES, smoke)
            print(f'check v4a: {"PASS" if record["pass"] else "FAIL"} failing {record["failing"]} max_abs_diff '
                  f'{record["max_abs_diff"]:.3g} smoke={smoke} ({time.monotonic() - started:.0f}s) '
                  f'{gate_path(outputs)}', flush=True)
            return 0 if record['pass'] else 1
        if args.command == 'verdict':
            record = verdict(outputs, args.candidate, workers=args.workers)
            if record.get('v_failure') is not None:
                print(f'stage 4 ({record["candidate"]}): no outcome: {record["v_failure"]}', flush=True)
                return 0
            print(f'stage 4 ({record["candidate"]}, {record["kind"]}): '
                  + '; '.join(f'{a} d {b["d"]:+.4f} [{b["lo"]:+.4f}, {b["hi"]:+.4f}] {b["outcome"]}'
                              for a, b in record['aspects'].items())
                  + f'; emotion over PercepT d {record["emotion_margin"]["d"]:+.4f} {record["emotion_margin"]["outcome"]} '
                  f'({time.monotonic() - started:.0f}s)', flush=True)
            return 0
        report(outputs, workers=args.workers)
        print(f'wrote {outputs / "stage4/reported.json"} ({time.monotonic() - started:.0f}s)', flush=True)
        return 0
    except Refusal as error:
        raise SystemExit(f'REFUSED: {error}')


if __name__ == '__main__':
    raise SystemExit(main())
