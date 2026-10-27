"""The ablation scorer of the recipe sweep (rule §2.5, §3.5, §4.5; ticket 10): development and held effects of the eight
removals of the final model F. It decides nothing.

Modes (`python rs_ablation.py MODE --final NAME`):
  dev   each removal x at seeds 1 to 3 on the frozen sample, exports on P: ē_a(x) = Ā_a(x) - Ā_a(F) with ticket 06's joint
        bootstrap (`rs_stage3.joint_bootstrap`, F in the reference's place, its own `default_rng(0)` per removal), the
        arms' and F's absolute AMI beside                                                         -> ablation/dev.json
  held  each removal x at full scale, seeds 0 to 4, on H: Δ_a(x) = A_x(I_0, S_0) - A_F(I_0, S_0), per aspect its own
        `default_rng(0)`: I_b, then one S_b over seeds 0 to 4 shared by x and F (rule §3.5), the absolute AMI beside
                                                                                                  -> ablation/held.json

F (rule §2.5) is c* of `stage3/verdict.json` under PICK and G under KEEP-G; with no stage 3 file it is G after stage 1's
SWEEP-NONE or stage 2's TOP-NONE (checked in the stage file), and a missing stage 3 file without that outcome is refused.
`--final NAME` must equal F ('G' for G). F's development runs are the ones stage 3 scored (checked against the SHA-256s its
verdict recorded) or, for G, 12-27's seed 1 to 3 runs through `rs_stage3.read_g_runs` (its V3a hash binding). F's held
outputs are stage 4's candidate's (`stage4/verdict.json` must name c*) or 12-27's `RG/outputs/stage_c/full/held/GS-0.02`.
Removal arms are `rs_sheet.config(F's changes, remove=(x,))`, folder `rs_sheet.name(...)` ('G-no_buddy').

An arm that failed (status failed in a run record, or a non-finite loss; a held output that failed VC3 twice; a graph removal
whose sample-scale graph check failed) is listed under `failed` with its cause and gets no effect; the other arms go on. An
arm with files missing and no failure record is refused (it has not finished). The statistics are ticket 06's
(development) and 12-27's `rg_held._bootstrap` kernel with this rule's draws (held); the pure draw code is above "files".
"""
import argparse
import ast
import hashlib
import json
import socket
from pathlib import Path

import numpy as np

import rs_paths
import rs_full
import rs_stage0
import rs_sheet
import rs_stage3 as T
import rs_sweep as S
import eg_score3 as E
import rb_ami
import rb_part2_report
import rb_pick
import rb_score
import rg_held
import rg_paths

ASPECTS = rs_sheet.ASPECTS
REMOVALS = rs_sheet.REMOVALS
DEV_SEEDS = rs_sheet.DEV_SEEDS               # (1, 2, 3)
FULL_SEEDS = rs_sheet.FULL_SEEDS             # (0, 1, 2, 3, 4)
G_NAME = 'G'                                 # G's sheet name; the user-facing --final
G_ARM = rs_paths.G_ARM                       # 'GS-0.02': G's 12-27 name; the key of G's partitions, boots and held sidecars
N_DRAWS = T.N_DRAWS                          # 2,000
SMOKE_DRAWS = T.SMOKE_DRAWS
# the two removals whose arms read graphs built by this round's code (rule §2.5): the variant whose sample-scale check must pass
GRAPH_VARIANT = dict(content_correction='uncorrected', instructed_graphs='default_view')
HELD_ARMS = ('X', 'F')                       # the keys of the two arms inside `rg_held._bootstrap`


# ================================================================== F and the arms (pure of the data)
def check_removals(removals, smoke):
    """All eight removals (a real run), or, only in a smoke, a subset in the sheet's order."""
    if removals is None:
        return list(REMOVALS)
    assert smoke, '--removals (a subset of the eight) is for a smoke only; a real run scores all eight'
    assert removals and set(removals) <= set(REMOVALS), f'removals {removals} are not in {list(REMOVALS)}'
    return [x for x in REMOVALS if x in set(removals)]


def _stage_record(path, stage, smoke):
    rec = json.loads(Path(path).read_text())
    assert rec['stage'] == stage and rec['smoke'] is bool(smoke), f'{path}: not a {stage} record of this kind (smoke flag differs)'
    return rec


def _final(name, outcome, pick, cfg, stage3_sha256, source_sha256, ended=None, stage_files=None):
    return dict(name=name, outcome=outcome, pick=pick, config=cfg, is_g=pick is None, run_key=G_ARM if pick is None else pick,
                stage3_sha256=stage3_sha256, source_sha256=source_sha256, ended=ended or {}, stage_files=stage_files or {})


def resolve_final(outputs, smoke):
    """F (rule §2.5): c* under PICK, G under KEEP-G of `stage3/verdict.json`; with no stage 3 file, G after stage 1's
    SWEEP-NONE or stage 2's TOP-NONE on every machine with a stage 1 file (the same rule as the full trainer)."""
    outputs = Path(outputs)
    path = T.verdict_path(outputs)
    if path.exists():
        v = _stage_record(path, 'stage3', smoke)
        assert v['p_sha256'] == rs_paths.P_SHA256, f'{path}: other P'
        assert v['outcome'] in ('PICK', 'KEEP-G'), f'{path}: outcome {v["outcome"]!r}'
        sha = S._file_sha256(path)
        if v['outcome'] == 'KEEP-G':
            assert v['pick'] is None, f'{path}: KEEP-G with a pick'
            return _final(G_NAME, 'KEEP-G', None, rs_sheet.config(), sha, sha)
        cfg = rs_sheet.parse(v['pick'])
        assert rs_sheet.name(cfg) == v['pick'] and not cfg['remove'] and cfg != rs_sheet.config(), f'{path}: pick {v["pick"]!r}'
        return _final(v['pick'], 'PICK', v['pick'], cfg, sha, sha)
    stage_files = {}
    for m in S.MACHINES:
        if not S.sweep_path(outputs, m, 'stage1').is_file():
            continue
        stage_files[m] = {}
        for stage in ('stage1', 'stage2'):
            p = S.sweep_path(outputs, m, stage)
            if p.is_file():
                _stage_record(p, stage, smoke)
                stage_files[m][stage] = S._file_sha256(p)
    ended = rs_full.sweep_ended_without_candidates(outputs)
    assert ended, (f'no stage 3 verdict ({path}), and not every machine with a stage 1 file ended in '
                   'SWEEP-NONE or TOP-NONE: stage 3 must have run (a missing file is not a KEEP-G)')
    outcomes = set(ended.values())
    outcome = 'SWEEP-NONE' if outcomes == {'SWEEP-NONE'} else 'TOP-NONE'
    source = None
    if len(ended) == 1:
        m, result = next(iter(ended.items()))
        source = stage_files[m]['stage1' if result == 'SWEEP-NONE' else 'stage2']
    return _final(G_NAME, outcome, None, rs_sheet.config(), None, source, ended, stage_files)


def check_final(final, name):
    """`--final NAME` must be F."""
    assert name == final['name'], (f'--final {name!r} is not the final model F = {final["name"]!r} '
                                   f'({final["outcome"]}, from the stage files)')


def require_stage4_candidate(outputs, final):
    """Held mode, F = c*: `stage4/verdict.json` names c* as its candidate (ticket 09 writes it). Returns its SHA-256; None
    for F = G (12-27's held outputs)."""
    if final['is_g']:
        return None
    path = Path(outputs) / 'stage4' / 'verdict.json'
    assert path.exists(), f'{path} missing: stage 4 (F = c*) has not written its verdict'
    candidate = json.loads(path.read_text())['candidate']
    assert candidate == final['pick'], f'stage4 verdict names candidate {candidate!r}, not c* {final["pick"]!r}'
    return S._file_sha256(path)


def arm_config(final, removal):
    """F's settings with the one removal and nothing retuned (spec §3.2 table)."""
    cfg = final['config']
    return rs_sheet.config({k: cfg[k] for k in rs_sheet.changed(cfg)}, remove=(removal,))


def arm_names(final, removals=REMOVALS):
    """{removal: run folder name}, in the sheet's order."""
    return {x: rs_sheet.name(arm_config(final, x)) for x in removals}


def graph_gate_cause(outputs, removal):
    """None, or why a graph removal does not run: `gates/ablation_graphs.json` (ticket 03) has its variant's check failing
    (rule §2.5: otherwise those two removals do not run, and the cause is reported). A missing file is not a failure."""
    variant = GRAPH_VARIANT.get(removal)
    path = Path(outputs) / 'gates' / 'ablation_graphs.json'
    if variant is None or not path.exists():
        return None
    item = json.loads(path.read_text())['items'].get(variant)
    if item is not None and item['pass'] is False:
        return (f'the sample-scale check of the {variant} graphs failed ({path.relative_to(outputs)}): '
                f'the {removal} removal does not run (rule §2.5)')
    return None


def graph_gate_sha256(outputs):
    path = Path(outputs) / 'gates' / 'ablation_graphs.json'
    return S._file_sha256(path) if path.is_file() else None


def failed_list(results):
    return [dict(removal=x, arm=r['arm'], cause=r['cause']) for x, r in results.items() if r['status'] == 'failed']


def _failed(name, cause):
    return dict(arm=name, status='failed', cause=cause, aspects=None)


# ================================================================== development
def cache_root(outputs):
    """This scorer's partitions and bootstrap scores: `<outputs>/ablation/cache/<name>/s<seed>/` (held: `.../held/<arm>/`)."""
    return Path(outputs) / 'ablation' / 'cache'


def part_folder(outputs, name, seed, aspect, shared):
    """Where a run's fixed partitions are cached: F's in stage 3's cache when they are there (the same files and identity
    stage 3 or V3a scored), everything else in this scorer's cache. Stage 3's folder is only read."""
    if name == shared:
        folder = T.cache_root(outputs) / name / f's{seed}'
        if (folder / f'{aspect}__partitions.npz').exists():
            return folder
    return cache_root(outputs) / name / f's{seed}'


class _Routed:
    """The cache of `rs_stage3.joint_bootstrap`: F's scores are read from stage 3's `Boots` when they exist, anything else
    from this scorer's own; new scores are written only to the own cache."""

    def __init__(self, own, shared, shared_name):
        self.own, self.shared, self.shared_name = own, shared, shared_name

    def get(self, name, tag):
        if name == self.shared_name:
            got = self.shared.get(name, tag)
            if got is not None:
                return got
        return self.own.get(name, tag)

    def put(self, name, tag, boot):
        self.own.put(name, tag, boot)


def run_hashes(runs):
    """{seed: {run_sha256, img_sha256 {aspect}}}: what a record says about the runs it scored."""
    return {str(s): dict(run_sha256=r['run_sha256'], img_sha256={a: rb_pick.array_sha256(r['img'][a]) for a in ASPECTS})
            for s, r in sorted(runs.items())}


def read_final_dev_runs(outputs, final, smoke):
    """{seed: run} of F on seeds 1 to 3. G: 12-27's runs through stage 3's reader with its V3a hash binding. c*: the runs
    stage 3 scored, accepted only if their run.json and exports have the SHA-256s its verdict recorded."""
    gate = T.require_v3a(outputs, smoke)
    if final['is_g']:
        return T.read_g_runs(gate)
    name = final['name']
    runs = T.read_config_runs(outputs, [name], smoke)[name]
    failed = [s for s, r in runs.items() if r['status'] != 'ok']
    assert not failed, f'F = {name} has failed seed(s) {failed}, which cannot be: stage 3 picked it'
    want = json.loads(T.verdict_path(outputs).read_text())['inputs']['runs'][name]
    for s, r in sorted(runs.items()):
        assert want[str(s)]['run_sha256'] == r['run_sha256'], f'F {name} s{s}: run.json differs from the one stage 3 scored'
        for a in ASPECTS:
            assert want[str(s)]['img_sha256'][a] == rb_pick.array_sha256(r['img'][a]), \
                f'F {name} s{s} {a}: the export differs from the one stage 3 scored'
    return runs


def arm_dev_failure(outputs, name, smoke):
    """None, or the causes ('seed 2: ...') of an arm's failed seeds (read from the run records, no export read). A seed
    without a run.json is an unfinished arm: refused."""
    causes = []
    for s in DEV_SEEDS:
        path = rs_paths.run_dir(outputs, name, s) / 'run.json'
        assert path.exists(), f'{name} s{s}: {path} missing; the arm has not finished'
        if json.loads(path.read_text())['status'] == 'failed':
            rec = S.read_run(outputs, name, s, 'local', smoke)                    # the reader's own checks of the record
            causes.append(f'seed {s}: {rec["cause"]}')
    return '; '.join(causes) or None


def dev_arms(outputs, final, removals, smoke):
    """{removal: dict(name, cause, runs)}: `runs` {seed: run} when the arm is ok on seeds 1 to 3, else None with the cause."""
    arms = {}
    for x, name in arm_names(final, removals).items():
        cause = graph_gate_cause(outputs, x) or arm_dev_failure(outputs, name, smoke)
        runs = None if cause else {s: S.read_run(outputs, name, s, 'local', smoke) for s in DEV_SEEDS}
        reruns = []
        for seed in DEV_SEEDS:
            path = rs_paths.run_dir(outputs, name, seed) / 'run.json'
            if path.is_file():
                rerun_of = json.loads(path.read_text()).get('rerun_of')
                if rerun_of is not None:
                    reruns.append(dict(removal=x, arm=name, seed=seed, rerun_of=rerun_of))
        arms[x] = dict(name=name, cause=cause, runs=runs, reruns=reruns)
    return arms


def _parts(outputs, name, runs, p_sha, shared):
    return {a: {s: S.fixed_partitions(part_folder(outputs, name, s, a, shared), name, a, runs[s]['img'][a], p_sha, s)
                for s in DEV_SEEDS} for a in ASPECTS}


def _caches(outputs, name, key, runs, f_runs, ctx, n_draws):
    label_sha = {a: rb_pick.array_sha256(ctx['labels'][a]) for a in ASPECTS}
    caches = {}
    for a in ASPECTS:
        caches[a] = {}
        for s in DEV_SEEDS:
            img = {name: rb_pick.array_sha256(runs[s]['img'][a]), key: rb_pick.array_sha256(f_runs[s]['img'][a])}
            own = T.Boots(cache_root(outputs), a, s, ctx['p_sha'], label_sha[a], img, n_draws)
            shared = T.Boots(T.cache_root(outputs), a, s, ctx['p_sha'], label_sha[a], img, n_draws)
            caches[a][s] = _Routed(own, shared, key)
    return caches


def score_dev(outputs, final, f_runs, arms, ctx, n_draws, workers):
    """{removal: record}: for an ok arm, ticket 06's joint bootstrap with the arm as c and F as G (each its own
    `default_rng(0)`): per aspect effect (ē_a), its interval, the arms' Ā and per-seed A beside; M̄ with its interval. A failed
    arm is a failed record and is never bootstrapped."""
    key = final['run_key']
    f_parts = None
    results = {}
    for x, arm in arms.items():
        if arm['cause']:
            results[x] = dict(_failed(arm['name'], arm['cause']), M=None)
            continue
        name = arm['name']
        if f_parts is None:
            f_parts = _parts(outputs, key, f_runs, ctx['p_sha'], key)
        x_parts = _parts(outputs, name, arm['runs'], ctx['p_sha'], key)
        parts = {a: {name: x_parts[a], key: f_parts[a]} for a in ASPECTS}
        print(f'{x}: {name} against {key}', flush=True)
        out = T.joint_bootstrap(ctx['labels'], parts, name, key, DEV_SEEDS, n_draws, workers,
                                _caches(outputs, name, key, arm['runs'], f_runs, ctx, n_draws), 'joint')
        aspects = {}
        for a in ASPECTS:
            e = out['effects'][a]
            aspects[a] = dict(effect=e['e'], lo=e['lo'], hi=e['hi'], arm_ami=e['Abar'][name], final_ami=e['Abar'][key],
                              arm_seed_ami={str(s): v for s, v in e['A'][name].items()},
                              final_seed_ami={str(s): v for s, v in e['A'][key].items()})
        assert all(np.isfinite(v) for r in aspects.values() for k, v in r.items() if np.isscalar(v)) and np.isfinite(out['M']['M'])
        results[x] = dict(arm=name, status='ok', cause=None, aspects=aspects, M=out['M'])
    return results


# ================================================================== held: the draws and the statistic (pure)
def held_draws(n_rows, n_draws, seeds=FULL_SEEDS):
    """[(I_b, {'X': S_b, 'F': S_b})]: one `default_rng(0)` (per aspect): for b in order I_b = `integers(0, n_rows,
    size=n_rows)`, then one S_b = `choice(seeds, size=5, replace=True)` shared by the removal arm and F (rule §3.5)."""
    assert n_rows > 0 and n_draws > 0 and list(seeds) == list(FULL_SEEDS)
    rng = np.random.default_rng(0)
    draws = []
    for _ in range(n_draws):
        rows = rng.integers(0, n_rows, size=n_rows)
        s = rng.choice(list(seeds), size=len(seeds), replace=True)
        draws.append((rows, {'X': s, 'F': s}))
    return draws


def held_effect(fixed_x, fixed_f, labels, n_draws, workers):
    """Δ(x) = A_x(I_0, S_0) - A_F(I_0, S_0) with the 95% interval of its bootstrap: `fixed_*` = {seed: {'r20': int [3, n]}}
    over seeds 0 to 4, labels int [n] (< 0 masked inside the AMI). The score is 12-27's (`rb_score.score_indices`, run by
    `rg_held._bootstrap`): the mean over the drawn seeds of the mean over a set's three partitions."""
    assert sorted(fixed_x) == sorted(fixed_f) == list(FULL_SEEDS)
    res = rg_held._bootstrap({'X': fixed_x, 'F': fixed_f}, labels, HELD_ARMS, held_draws(len(labels), n_draws), workers)
    return dict(effect=res['d'], lo=res['interval'][0], hi=res['interval'][1], arm_ami=res['A_G'], final_ami=res['A_F'],
                arm_seed_ami={str(s): v for s, v in res['per_seed']['X'].items()},
                final_seed_ami={str(s): v for s, v in res['per_seed']['F'].items()})


# ================================================================== held: files
def held_final_source(outputs, final):
    """(folder of F's held outputs, the arm name their sidecars carry): 12-27's G files, or stage 4's candidate's."""
    if final['is_g']:
        return rs_paths.RG_FULL / 'held' / G_ARM, G_ARM
    return rs_paths.full_root(outputs) / 'held' / final['name'], final['name']


def held_arm_failure(outputs, name):
    """None, or the causes of a full-scale arm's failed seeds, from the trainer's own records: `train/<name>/failed__s<seed>.json`
    (non-finite loss), a `run__s<seed>.json` with status failed, or a `held/<name>/failed__<aspect>__s<seed>.json` (VC3 twice)."""
    root = rs_paths.full_root(outputs)
    causes = []
    for s in FULL_SEEDS:
        failed, run = root / 'train' / name / f'failed__s{s}.json', root / 'train' / name / f'run__s{s}.json'
        if failed.exists():
            causes.append(f'seed {s}: {json.loads(failed.read_text())["cause"]}')
        elif run.exists() and json.loads(run.read_text())['status'] == 'failed':
            causes.append(f'seed {s}: {json.loads(run.read_text())["cause"]}')
        for a in ASPECTS:
            vc3 = root / 'held' / name / f'failed__{a}__s{s}.json'
            if vc3.exists():
                causes.append(f'{a} seed {s}: held output failed VC3 twice ({"; ".join(json.loads(vc3.read_text())["problems"])})')
    return '; '.join(causes) or None


def require_held_outputs(outputs, name):
    for a in ASPECTS:
        for s in FULL_SEEDS:
            path = rs_paths.full_root(outputs) / 'held' / name / f'{a}__s{s}.npz'
            assert path.is_file(), f'{name}: held output {path} missing; the arm has not finished'


def require_v4c(outputs, name):
    """The completed guarded held path, bound to this arm, H and every sidecar's array SHA-256. The array and
    sidecar are checked together again by `held_partitions`; a failed arm has no stamp and never calls this."""
    root = rs_paths.full_root(outputs) / 'held' / name
    path = root / 'v4c.json'
    assert path.is_file(), f'V4c: {path} missing; guarded held inference has not finished'
    gate = json.loads(path.read_text())
    assert gate['pass'] is True and gate['name'] == name, f'V4c: {path}: not a passing record for {name}'
    assert gate['list_sha256'] == rs_paths.HELD_SHA256, f'V4c: {path}: other H'
    keys = {f'{a}__s{s}' for a in ASPECTS for s in FULL_SEEDS}
    assert set(gate['outputs']) == keys, f'V4c: {path}: incomplete output set'
    for key in sorted(keys):
        sidecar = root / f'{key}.json'
        assert sidecar.is_file(), f'V4c: {sidecar} missing'
        assert gate['outputs'][key] == json.loads(sidecar.read_text())['sha256'], f'V4c: {path}: {key} differs from sidecar'
    return dict(sha256=S._file_sha256(path), reruns=gate['reruns'])


def held_arms(outputs, final, removals):
    """{removal: dict(name, cause)}: cause set for a failed arm (graph check, training, VC3), None for an arm with all
    its 15 held outputs; an arm with neither is refused."""
    arms = {}
    for x, name in arm_names(final, removals).items():
        cause = graph_gate_cause(outputs, x) or held_arm_failure(outputs, name)
        if not cause:
            require_held_outputs(outputs, name)
            require_v4c(outputs, name)
        arms[x] = dict(name=name, cause=cause)
    return arms


def held_partitions(outputs, root, arm, aspect, seed, sha):
    """(int [3, n] r20 partitions, source record) of one held output: sidecar and VC3 checks (`rg_held.load_held_output`),
    partitions fixed once and cached under `ablation/cache/held/<arm>/` (12-24's `rb_score.fixed_partitions`)."""
    file = Path(root) / f'{aspect}__s{seed}.npz'
    assert file.is_file(), f'{file}: missing'
    x = rg_held.load_held_output(file, arm, seed, sha, aspect=aspect)
    source = rg_held.held_output_source(file)
    parts = rb_score.fixed_partitions(x, rg_held.PARTITION_ARM, cache_root(outputs) / 'held' / arm, aspect, seed, sha,
                                      source['output_sha256'])['r20']
    return parts, source


def _held_sets(outputs, root, arm, sha):
    """({aspect: {seed: {'r20': parts}}}, {aspect: {seed: source}}) of one arm's 15 held outputs."""
    fixed, sources = {}, {}
    for a in ASPECTS:
        fixed[a], sources[a] = {}, {}
        for s in FULL_SEEDS:
            parts, source = held_partitions(outputs, root, arm, a, s, sha)
            fixed[a][s], sources[a][str(s)] = {'r20': parts}, source
        print(f'{arm} {a}: partitions done', flush=True)
    return fixed, sources


def score_held(outputs, final, arms, labels, sha, n_draws, workers):
    """({removal: record}, {arm name: {aspect: {seed: source}}}): per aspect Δ_a(x) with its interval and the absolute A beside;
    a failed arm gets no effect and its outputs are never read. Sources include F's."""
    f_root, f_arm = held_final_source(outputs, final)
    if not final['is_g']:
        require_v4c(outputs, f_arm)
    f_fixed = f_sources = None
    results, sources = {}, {}
    for x, arm in arms.items():
        if arm['cause']:
            results[x] = _failed(arm['name'], arm['cause'])
            continue
        name = arm['name']
        if f_fixed is None:
            f_fixed, f_sources = _held_sets(outputs, f_root, f_arm, sha)
            sources[f_arm] = f_sources
        x_fixed, sources[name] = _held_sets(outputs, rs_paths.full_root(outputs) / 'held' / name, name, sha)
        aspects = {}
        for a in ASPECTS:
            aspects[a] = held_effect(x_fixed[a], f_fixed[a], labels[a], n_draws, workers)
            print(f'{x} {a}: effect {aspects[a]["effect"]:+.6f} [{aspects[a]["lo"]:+.6f}, {aspects[a]["hi"]:+.6f}]', flush=True)
        results[x] = dict(arm=name, status='ok', cause=None, aspects=aspects)
    return results, sources


# ================================================================== the files
def code_sha256():
    """Hash this module and its transitive top-level imports in the round, RG, EG and RB folders (copied from
    `rs_stage3.code_sha256`, which starts from rs_stage3.py; same method)."""
    folders = (rs_paths.ROUND, rg_paths.ROUND, Path(E.__file__).parent, Path(rb_score.__file__).parent)
    pending = [rs_paths.ROUND / 'rs_ablation.py']
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
    manifest = {f'{p.parent.name}/{p.name}': S._file_sha256(p) for p in files}
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def code_state():
    return dict(S.code_state(), code_sha256=code_sha256())


def final_record(final):
    return {k: final[k] for k in ('name', 'outcome', 'pick', 'config', 'stage3_sha256', 'source_sha256', 'ended', 'stage_files')}


def dev_path(outputs):
    return Path(outputs) / 'ablation' / 'dev.json'


def held_path(outputs):
    return Path(outputs) / 'ablation' / 'held.json'


def run_dev(outputs, final_name, smoke=False, workers=8, removals=None):
    """The development ablation -> ablation/dev.json (rule §3.5). F and `--final` are checked before any file is read."""
    removals = check_removals(removals, smoke)
    final = resolve_final(outputs, smoke)
    check_final(final, final_name)
    code = code_state()                                   # taken before the scoring: the record names the code that computed it
    n_draws = SMOKE_DRAWS if smoke else N_DRAWS
    f_runs = read_final_dev_runs(outputs, final, smoke)
    arms = dev_arms(outputs, final, removals, smoke)
    ctx = E.load_context()                                # asserts P's SHA-256 is 12-24's
    results = score_dev(outputs, final, f_runs, arms, ctx, n_draws, workers)
    record = dict(stage='ablation_dev', smoke=bool(smoke), written=T.now_amsterdam(), rule_commit=rs_paths.RULE_COMMIT,
                  ranges=dict(rs_sheet.RANGES), seeds=list(DEV_SEEDS), n_draws=n_draws, draw_mode='joint',
                  draws='per removal: default_rng(0); I_b over the rows of P, then S_b over seeds 1 to 3, shared by the arm, F and the aspects',
                  final=final_record(final), final_key=final['run_key'], removals=removals, results=results,
                  failed=failed_list(results), reruns=[r for arm in arms.values() for r in arm.get('reruns', [])],
                  inputs=dict(ablation_graphs_sha256=graph_gate_sha256(outputs), final_runs=run_hashes(f_runs),
                              v3a_sha256=S._file_sha256(T.gate_path(outputs)),
                              arm_runs={x: None if arm['runs'] is None else run_hashes(arm['runs']) for x, arm in arms.items()}),
                  p_sha256=rs_paths.P_SHA256, versions=rb_ami.versions(), host=socket.gethostname(), **code)
    return S._keep_or_write(dev_path(outputs), S._clean(record))


def run_held(outputs, final_name, smoke=False, workers=8, removals=None):
    """The held ablation -> ablation/held.json (rule §3.5)."""
    removals = check_removals(removals, smoke)
    final = resolve_final(outputs, smoke)
    check_final(final, final_name)
    stage4_sha = require_stage4_candidate(outputs, final)
    code = code_state()
    n_draws = SMOKE_DRAWS if smoke else N_DRAWS
    names, sha, labels = rb_part2_report.held_labels()
    assert sha == rs_paths.HELD_SHA256 and len(names) == rs_paths.N_HELD, 'H differs from the CT\'s'
    T.require_v3a(outputs, smoke)
    g_full = rs_stage0.g_full_outputs_check() if final['is_g'] else None
    assert g_full is None or g_full['pass'] is True, '12-27 G held outputs differ from its recorded hashes'
    arms = held_arms(outputs, final, removals)
    v4c_names = ([final['name']] if not final['is_g'] else []) + [arm['name'] for arm in arms.values() if not arm['cause']]
    v4c = {name: require_v4c(outputs, name) for name in v4c_names}
    results, sources = score_held(outputs, final, arms, labels, sha, n_draws, workers)
    record = dict(stage='ablation_held', smoke=bool(smoke), written=T.now_amsterdam(), rule_commit=rs_paths.RULE_COMMIT,
                  ranges=dict(rs_sheet.RANGES), seeds=list(FULL_SEEDS), n_draws=n_draws, held_sha256=sha,
                  labelled_rows={a: int((labels[a] >= 0).sum()) for a in ASPECTS},
                  draws='per aspect: default_rng(0); I_b over the 12,281 rows of H, then one S_b over seeds 0 to 4 shared by the arm and F',
                  final=final_record(final), final_key=held_final_source(outputs, final)[1], removals=removals, results=results,
                  failed=failed_list(results),
                  reruns=[dict(removal=next((x for x, arm in arms.items() if arm['name'] == name), None),
                               arm=name, aspect=a, seed=int(seed), vc3_attempts=source['vc3_attempts'])
                          for name, aspects in sources.items() for a, seeds in aspects.items() for seed, source in seeds.items()
                          if source['vc3_attempts'] > 1],
                  inputs=dict(ablation_graphs_sha256=graph_gate_sha256(outputs), stage4_sha256=stage4_sha,
                              v3a_sha256=S._file_sha256(T.gate_path(outputs)), v4c=v4c, g_full_outputs=g_full, outputs=sources),
                  versions=rb_ami.versions(), host=socket.gethostname(), **code)
    return S._keep_or_write(held_path(outputs), S._clean(record))


# ================================================================== CLI
def main(argv=None):
    parser = argparse.ArgumentParser(description='The ablation scorer of the recipe sweep: development and held effects of the removals.')
    parser.add_argument('mode', choices=['dev', 'held'])
    parser.add_argument('--final', required=True, metavar='NAME', help="F's run name ('G' when G stays); must equal the stage files'")
    parser.add_argument('--removals', nargs='+', help=f'a subset of {list(REMOVALS)} (smoke only)')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--outputs', type=Path, default=None,
                        help=f'default {rs_paths.OUTPUTS}; with --smoke {rs_paths.SMOKE_ROOT}/r10-rs-smoke')
    parser.add_argument('--smoke', action='store_true', help='outputs under /tmp/claude-0/; 20 draws; reads smoke runs')
    args = parser.parse_args(argv)
    assert 1 <= args.workers <= 8, '--workers at most 8'
    outputs = args.outputs or (rs_paths.SMOKE_ROOT / 'r10-rs-smoke' if args.smoke else rs_paths.OUTPUTS)
    outputs = rs_paths.check_outputs(outputs, args.smoke)
    rs_paths.check_imports()
    run = run_dev if args.mode == 'dev' else run_held
    rec = run(outputs, args.final, args.smoke, args.workers, args.removals)
    path = dev_path(outputs) if args.mode == 'dev' else held_path(outputs)
    print(f'{path}: {len(rec["results"]) - len(rec["failed"])} arms scored, {len(rec["failed"])} failed'
          + ''.join(f'\n  failed {f["removal"]} ({f["arm"]}): {f["cause"]}' for f in rec['failed']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
