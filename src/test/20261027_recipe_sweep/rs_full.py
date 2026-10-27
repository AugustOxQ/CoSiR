"""Full-scale trainer and held inference of the recipe sweep (ticket 08): any `rs_sheet` configuration trained as 12-27's
G at full scale (42,969 train paintings, the CT's graphs, stopping on the 6,152 validation paintings), with the knob
layer's patches (`rs_knobs`, scale 'full'); held outputs of the three heads on H. `regress v4a` is V4a's training half.

G enters no patch: `train` reaches `rg_full.train_cohort` (-> `ct2_d2.run`) and `rg_full.infer_held` exactly as 12-27's
stage C did. Only two things differ from 12-27's `rg_full.train` / `held`, both outside the computation:
  * the run is named by its configuration (`rs_sheet.name`): the weights of 'k2=8' are `<full>/weights/k2=8/...`. The
    frozen functions key everything by an arm name that must be in `rg_variants.ARMS`, so the run's name is added to
    ARMS as a copy of G's arm (taken after the knob patches, so with k3 and k4 applied) for the length of the call;
  * the guards are this round's (rule §4.4): V4a's two gates and the stage 3 verdict instead of 12-27's VC1 and stage B.
Files, as 12-27's stage C layout (index, ticket 08): `<root>/weights/<name>/{aspect}__s{seed}.pt|json`,
`<root>/train/<name>/{R6-5/..., run__s{seed}.json}`, `<root>/held/<name>/{aspect}__s{seed}.npz|json` and `v4c.json`.
`run__s{seed}.json` is this module's record of the run: configuration, what was patched, the SHA-256 of the frozen
modules (checked at commit 68e8081 before the run), of the training code, and of the stage 3 verdict.

CLI: rs_full train --config NAME [NAME ...] [--seeds ...]   rs_full held --config NAME [NAME ...]   rs_full regress v4a
"""
import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch
from threadpoolctl import threadpool_limits

import rs_paths as paths
import rs_sheet as S
import rs_knobs
import rs_train
import ab_train
import ct2_d2  # noqa: F401  (the frozen modules whose hashes are checked)
import ct2_data  # noqa: F401
import ct2_graphs  # noqa: F401
import eg_train3 as eg
import rb_encoders
import rb_rows
import rb_student
import rg_full
import rg_loss
import rg_variants as V
from ab_paths import ms_train

ASPECTS = tuple(paths.ASPECTS)
G_ARM = paths.G_ARM
SEEDS = rg_full.SEEDS
V4A = 'G-v4a'                         # V4a's replay of G, named apart from G's frozen run (as rs_train's 'G-v1a')
REFERENCE = paths.RG_FULL             # 12-27's stage C full tree: weights/GS-0.02, held/GS-0.02
STAGE3 = 'stage3/verdict.json'
ROUND_FILES = ('rs_full.py', 'rs_knobs.py', 'rs_sheet.py', 'rs_paths.py', 'rs_graphs.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------- provenance of the frozen code
def frozen_modules():
    """{module name: {path, sha256}} of the `ms_` and `ct2_` modules imported now, and RG's `rg_loss` and `rg_full`."""
    result = {}
    for name, module in sorted(sys.modules.items()):
        if '.' in name or not getattr(module, '__file__', None):
            continue
        if name.startswith(('ms_', 'ct2_')) or name in ('rg_loss', 'rg_full'):
            path = Path(module.__file__).resolve()
            result[name] = dict(path=str(path), sha256=sha(path))
    return result


def git_show(commit, relative):
    return subprocess.run(['git', 'show', f'{commit}:{relative}'], cwd=paths.ROUND, capture_output=True,
                          check=True).stdout


def require_frozen_code(show=git_show):
    """V4a: every module of `frozen_modules()` equals its file at `rs_paths.FROZEN_COMMIT` (read with `git show`).
    Returns {name: sha256}; refuses with the module's name otherwise."""
    top = Path(subprocess.run(['git', 'rev-parse', '--show-toplevel'], cwd=paths.ROUND, capture_output=True, text=True,
                              check=True).stdout.strip()).resolve()
    shas = {}
    for name, record in frozen_modules().items():
        relative = Path(record['path']).relative_to(top)
        try:
            frozen = show(paths.FROZEN_COMMIT, str(relative))
        except subprocess.CalledProcessError:
            raise AssertionError(f'{name}: {relative} is not in commit {paths.FROZEN_COMMIT[:7]}') from None
        assert hashlib.sha256(frozen).hexdigest() == record['sha256'], \
            f'{name}: {relative} differs from commit {paths.FROZEN_COMMIT[:7]}'
        shas[name] = record['sha256']
    return shas


def code_sha256():
    """SHA-256, by file name, of what decides a full-scale run: this round's files, the frozen modules, and the
    other modules under them (the loss, the recipe rows, the encoders' loader, 12-24's student)."""
    files = [paths.ROUND / name for name in ROUND_FILES]
    files += [Path(record['path']) for record in frozen_modules().values()]
    files += [Path(m.__file__) for m in (rb_student, rb_rows, rb_encoders, ab_train, eg, V, ms_train.itp_train,
                                         ms_train.head)]
    result = {Path(p).name: sha(p) for p in files}
    assert len(result) == len(set(map(Path, files))), 'two files with one name'
    return result


def jsonable(cfg):
    return dict(cfg, remove=list(cfg['remove']))


# ----------------------------------------------------------------------------------------------- names and verdicts
def resolve(name):
    """The configuration of a run name: a sheet name, or V4a's replay of G. G itself is 12-27's frozen run."""
    if name == V4A:
        return S.config()
    assert name != 'G', (f"run name 'G': G's full-scale weights and held outputs are 12-27's frozen files "
                         "(V4a compares with them); only `regress v4a` retrains it")
    try:
        return S.parse(name)
    except (ValueError, KeyError, AssertionError):
        raise AssertionError(f'run name {name!r} is not a sheet name ({V4A} aside)') from None


def graph_choice(cfg):
    """Which graphs the run reads: the CT's aspect graphs at the run's k11. Ticket 07's graph removals change `variant`."""
    return dict(k=cfg['k11'], variant='aspect')


def stage3_problem(cfg, verdict):
    """None, or why `cfg` may not run at full scale under this stage 3 verdict: a stage 4 candidate is c* of a PICK; an
    ablation arm (a removal) is a removal from the final model F (c* under PICK, G under KEEP-G)."""
    final = S.parse(verdict['c_star']) if verdict['outcome'] == 'PICK' else S.config()
    if not cfg['remove']:
        if verdict['outcome'] != 'PICK':
            return 'the stage 3 verdict is KEEP-G: stage 4 does not run'
        if cfg != final:
            return f'the stage 3 verdict names {verdict["c_star"]}, not {S.name(cfg)}'
        return None
    if dict(cfg, remove=()) != final:
        return f'{S.name(cfg)} is not a removal from the final model {S.name(final)}'
    return None


def require_stage3(outputs, cfg):
    """The stage 3 verdict names `cfg` (see `stage3_problem`); returns its SHA-256. The file's fields, as ticket 06
    writes them: outcome 'PICK' or 'KEEP-G', and c_star (a run name) under PICK."""
    path = Path(outputs) / STAGE3
    verdict = rg_full._record(path)
    assert verdict.get('outcome') in ('PICK', 'KEEP-G'), f'{path}: outcome {verdict.get("outcome")!r}'
    assert verdict['outcome'] == 'KEEP-G' or isinstance(verdict.get('c_star'), str), f'{path}: PICK without c_star'
    problem = stage3_problem(cfg, verdict)
    assert problem is None, f'{path}: {problem}'
    return sha(path)


def arm_alias(name):
    """The run's name as an arm of `rg_variants` with G's (patched) switches, for the frozen functions that key by arm."""
    assert name not in V.ARMS, name
    return patch.dict(V.ARMS, {name: dict(V.ARMS[G_ARM])})


def load_data():
    """The CT's full-scale training data (`rb_student.load_scale`, replaced by k11's rebuilt graphs under the knob
    layer), as `rg_full.train` reads it."""
    s, inputs, graphs, manifests = rb_student.load_scale(rg_full.SCALE)
    return s, inputs, graphs, manifests, rb_student.training_images(rg_full.ENCODER, s, inputs)


# ------------------------------------------------------------------------------------------------------- training
def record_path(root, name, seed):
    return Path(root) / 'train' / name / f'run__s{seed}.json'


def weights_complete(root, name, seed):
    return all(rb_student.weights_path(root, name, a, seed).exists() for a in ASPECTS)


def check_resumed(root, name, seed, cfg, smoke, shas):
    """A cohort with verified weights is skipped only if its record is this configuration's, trained by this code."""
    path = record_path(root, name, seed)
    assert path.is_file(), f'{path}: weights without a run record; move them aside'
    record = json.loads(path.read_text())
    assert (record['name'], record['seed'], record['config'], record['smoke']) == (name, seed, jsonable(cfg), smoke), \
        f'{path}: configuration mismatch'
    assert record['frozen_sha256'] == shas, f'{path}: frozen modules changed'
    assert all(record['code_sha256'].get(k) == v for k, v in code_sha256().items()), \
        f'{path}: code_sha256 changed; move it aside and rerun'
    return record


@threadpool_limits.wrap(limits=8)
def train(name, seeds, device, outputs, *, smoke=False, max_epochs=None, root=None):
    """Train the configuration `name` for each seed at full scale into `root` (default `full_root(outputs)`; V4a's
    replay names its own). A seed with verified weights is skipped; a configuration with any held output is not
    trained further (V4d); a seed that stopped on a non-finite loss is not rerun. Returns {seed: run record}."""
    cfg, seeds = resolve(name), list(seeds)
    assert seeds and len(set(seeds)) == len(seeds) and all(s in SEEDS for s in seeds), f'seeds: {seeds}'
    assert (name == V4A) == (root is not None), 'regression root: only the V4a replay names its folder'
    assert not smoke or (max_epochs or 2) <= 2, '--smoke: at most 2 epochs'
    assert max_epochs is None or smoke, 'a full run trains to the CT\'s 200 epochs and its patience'
    paths.check_imports()
    outputs, device = paths.check_outputs(outputs, smoke), torch.device(device)
    root = paths.check_outputs(root if root is not None else paths.full_root(outputs), smoke)
    assert root.is_relative_to(outputs), 'run folder must be under --outputs'
    if device.type == 'cuda':
        assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'set CUBLAS_WORKSPACE_CONFIG=:4096:8 (CT queue)'
    shas = require_frozen_code()                        # before any data is read
    verdict_sha = require_stage3(outputs, cfg) if not smoke and name != V4A else None
    for seed in seeds:
        assert not rg_full.failed_path(root, name, seed).exists(), \
            f'V4d: {name} s{seed} stopped on a non-finite loss ({rg_full.failed_path(root, name, seed)}); no rerun'
    held = sorted((root / 'held' / name).rglob('*.npz')) if (root / 'held' / name).exists() else []
    assert not held or all(weights_complete(root, name, seed) for seed in seeds), \
        f'V4d: no new training of {name} once held outputs exist (any seed): {held[0]}'
    torch.set_num_threads(8)
    result = {}
    with ExitStack() as stack:
        knobs = rs_knobs.patches(cfg, stack, scale='full', outputs=outputs)
        stack.enter_context(arm_alias(name))
        s, inputs, graphs, manifests, images = load_data()
        rb = rb_student._rb_block(rg_full.SCALE, rg_full.ENCODER, rg_full.RECIPE, rb_rows.recipe(rg_full.RECIPE), images)
        for seed in seeds:
            rg = rg_full.rg_block(name, seed, len(s['fit']))
            state = rg_full._cohort_state(root, name, seed, rb, rg)
            if state == 'done':
                result[seed] = check_resumed(root, name, seed, cfg, smoke, shas)
                print(f'{name} s{seed}: verified weights; skip', flush=True)
                continue
            assert not held, f'V4d: no new training of {name} once held outputs exist (any seed): {held[0]}'
            if state == 'broken':
                rb_student._move_aside(root, name, seed, rg_full.RECIPE)
            start = time.monotonic()
            outcome = rg_full.train_cohort(s, inputs, graphs, manifests, images, name, seed, device, root, rb=rb,
                                           smoke=smoke, max_epochs=max_epochs)
            commit, dirty = rs_train.code_commit()
            record = dict(name=name, seed=seed, config=jsonable(cfg), smoke=smoke, max_epochs=max_epochs,
                          device=str(device), status='ok', cause=None, knobs=knobs, graphs=graph_choice(cfg),
                          stage3_verdict_sha256=verdict_sha, frozen_commit=paths.FROZEN_COMMIT, frozen_sha256=shas,
                          code_sha256=code_sha256(), code_commit=commit, code_dirty=dirty, host=socket.gethostname(),
                          seconds=time.monotonic() - start)
            if outcome.get('status') == 'failed':
                record.update(status='failed', cause=outcome['cause'])
            else:
                infos = {a: json.loads(rb_student.weights_path(root, name, a, seed).with_suffix('.json').read_text())
                         for a in ASPECTS}
                record.update(weights_sha256={a: i['weights_sha256'] for a, i in infos.items()},
                              best_epoch={a: i['best_epoch'] for a, i in infos.items()},
                              best_dev_loss={a: i['best_dev_loss'] for a, i in infos.items()})
            eg.atomic_record(record_path(root, name, seed), record)
            result[seed] = record
    return result


# ------------------------------------------------------------------------------------------------ held inference
def require_v4a(outputs):
    """V4a first (rule §4.4): the training gate (a full run, not a smoke, by the code that is on disk now) and the
    scorer gate pass. Returns the gate files' SHA-256."""
    shas = {}
    for gate in ('v4a_train', 'v4a_scorer'):
        path = Path(outputs) / f'gates/{gate}.json'
        record = rg_full._record(path)
        assert record.get('pass') is True, f'V4a: {path} does not pass'
        if gate == 'v4a_train':
            assert record.get('smoke') is False, f'V4a: {path} is a smoke run'
            assert record.get('code_sha256') == code_sha256(), \
                f'V4a: {path} was written by other code than is on disk now; rerun `regress v4a`'
        shas[gate] = sha(path)
    return shas


def rule_commit_is_ancestor(commit=paths.RULE_COMMIT):
    return rg_full.rule_commit_is_ancestor(commit)


def check_runs(root, name, cfg, seeds, outputs, shas):
    """Every seed's run record is ok, was written by the code on disk now, names the stage 3 verdict that is there now
    (it was written before the first full run and has not changed since), and agrees with its weights."""
    verdict_sha = sha(Path(outputs) / STAGE3)
    for seed in seeds:
        record = json.loads(record_path(root, name, seed).read_text())
        assert record['status'] == 'ok' and record['config'] == jsonable(cfg), f'{name} s{seed}: not an ok run of {name}'
        assert record['stage3_verdict_sha256'] == verdict_sha, \
            f'{name} s{seed}: the stage 3 verdict changed after the first full run (or the run was a smoke)'
        assert record['frozen_sha256'] == shas, f'{name} s{seed}: trained by other frozen modules than are on disk now'
        assert all(record['code_sha256'].get(k) == v for k, v in code_sha256().items()), \
            f'{name} s{seed}: trained by other code (code_sha256 changed); retrain'
        for aspect in ASPECTS:
            weights = rb_student.weights_path(root, name, aspect, seed)
            assert record['weights_sha256'][aspect] == ms_train.head.sha_file(weights), \
                f'{weights}: weights differ from the run record'


def v4c_record(root, name, seeds):
    """V4c: each held output set has H's rows (12,281, finite and unit norm: `rg_full.vc3_problem` ran on each when it
    was written) and H's SHA-256. Lists the outputs that needed a second try (V4d). Returns the record."""
    outputs, reruns = {}, []
    for seed in seeds:
        for aspect in ASPECTS:
            path = Path(root) / 'held' / name / f'{aspect}__s{seed}.json'
            assert path.is_file(), f'V4c: {path} missing'
            info = json.loads(path.read_text())
            assert info['rows'] == paths.N_HELD, f'V4c: {path}: rows {info["rows"]}, not {paths.N_HELD}'
            assert info['list_sha256'] == paths.HELD_SHA256, f"V4c: {path}: held list SHA-256 differs from the CT's"
            outputs[f'{aspect}__s{seed}'] = info['sha256']
            if info['vc3_attempts'] > 1:
                reruns.append(dict(output=f'{aspect}__s{seed}', problems=info['vc3_problems']))
    return {'pass': True, 'name': name, 'rows': paths.N_HELD, 'list_sha256': paths.HELD_SHA256, 'outputs': outputs,
            'reruns': reruns}


@threadpool_limits.wrap(limits=8)
def held(name, seeds, outputs, device=None, *, smoke=False, ancestor=rule_commit_is_ancestor):
    """Guarded held inference of the configuration `name`: the guards first, before any data or weights are read; all
    five seeds at once (a partial run would leave held outputs that block training the rest, V4d). Writes
    `held/<name>/` and its `v4c.json`."""
    assert name != V4A, f'{name} is the V4a replay: `regress v4a` writes its held outputs'
    cfg, seeds = resolve(name), list(seeds)
    paths.check_imports()
    outputs = paths.check_outputs(outputs, smoke)
    root = rg_full._check_root(paths.full_root(outputs))
    if not smoke:
        assert sorted(seeds) == list(S.FULL_SEEDS), f'held runs all seeds {S.FULL_SEEDS} of a configuration at once, not {seeds}'
        require_v4a(outputs)
        require_stage3(outputs, cfg)
        shas = require_frozen_code()
        rg_full._refuse_failed(root, name, seeds)
    torch.set_num_threads(8)
    with ExitStack() as stack:
        rs_knobs.patches(cfg, stack, scale='full', outputs=outputs)
        stack.enter_context(arm_alias(name))
        if not smoke:
            # nothing is written before every seed's weights are there and intact
            for seed in seeds:
                for aspect in ASPECTS:
                    rg_full._arm_weights(root, name, aspect, seed, False)
            check_runs(root, name, cfg, seeds, outputs, shas)
            assert ancestor(), f'the rule commit {paths.RULE_COMMIT} is not an ancestor of HEAD'
        rg_full.infer_held(name, seeds, device, root, smoke=smoke)
    record = v4c_record(root, name, seeds)
    eg.atomic_record(root / 'held' / name / 'v4c.json', record)
    return record


# ------------------------------------------------------------------------------------------------------ V4a
def v4a_items(reg_root, reference=None, seed=0):
    """Compare the replay of G (`reg_root`, arm V4A) with 12-27's files (`reference`, arm GS-0.02), bit for bit, per
    aspect: the weights (same keys in the same order, every tensor `torch.equal`), the best epoch and dev loss, and the
    held `img` (shape, dtype and bits). As `rg_full.vc1_items`, which names its arms."""
    reference = REFERENCE if reference is None else reference
    items = {}
    for aspect in ASPECTS:
        ours_pt = rb_student.weights_path(reg_root, V4A, aspect, seed)
        theirs_pt = rb_student.weights_path(reference, G_ARM, aspect, seed)
        a, b = (torch.load(p, map_location='cpu', weights_only=True) for p in (ours_pt, theirs_pt))
        same = list(a) == list(b) and all(torch.equal(a[k], b[k]) for k in a)
        oi, ti = (json.loads(p.with_suffix('.json').read_text()) for p in (ours_pt, theirs_pt))
        parts = {'weights': {'pass': bool(same), 'keys': list(a) == list(b), 'ours_sha256': ms_train.head.sha_file(ours_pt),
                             'reference_sha256': ms_train.head.sha_file(theirs_pt)},
                 'best_epoch': {'pass': bool(oi['best_epoch'] == ti['best_epoch']
                                             and oi['best_dev_loss'] == ti['best_dev_loss']),
                                'ours': [oi['best_epoch'], oi['best_dev_loss']],
                                'reference': [ti['best_epoch'], ti['best_dev_loss']]}}
        ours, _ = rb_student.read_outputs(Path(reg_root) / 'held' / V4A / f'{aspect}__s{seed}.npz')
        theirs, _ = rb_student.read_outputs(Path(reference) / 'held' / G_ARM / f'{aspect}__s{seed}.npz')
        ours, theirs = ours['img'], theirs['img']
        equal = (ours.shape == theirs.shape and ours.dtype == theirs.dtype
                 and bool((rb_student._bits(ours) == rb_student._bits(theirs)).all()))
        parts['held'] = {'pass': bool(equal), 'rows': int(len(ours)), 'ours_sha256': rb_student.sha_array(ours),
                         'reference_sha256': rb_student.sha_array(theirs)}
        items[aspect] = dict(**{'pass': all(v['pass'] for v in parts.values())}, **parts)
    return items


def regress_v4a(device, outputs, *, smoke=False):
    """V4a, training half (rule §4.4): G at seed 0 through this module's trainer and held inference, into
    `regression/v4a_run/`, equals 12-27's weights and held outputs bit for bit. Writes `gates/v4a_train.json`. With
    --smoke the run is 2 epochs and cannot equal 12-27's: the record is written (pass false) only to prove the path."""
    outputs = paths.check_outputs(outputs, smoke)
    recorded = json.loads(rb_student.weights_path(REFERENCE, G_ARM, ASPECTS[0], 0).with_suffix('.json').read_text())
    assert str(torch.device(device)) == recorded['config']['device'], \
        f'the reference weights were trained on {recorded["config"]["device"]}; the regression must run on it'
    reg_root = (outputs / 'regression/v4a_run').resolve()
    runs = train(V4A, [0], device, outputs, smoke=smoke, root=reg_root)
    with ExitStack() as stack:
        rs_knobs.patches(S.config(), stack, scale='full', outputs=outputs)       # none: G
        stack.enter_context(arm_alias(V4A))
        rg_full.infer_held(V4A, [0], device, reg_root, smoke=smoke)
    items = v4a_items(reg_root)
    commit, dirty = rs_train.code_commit()
    record = dict(items=items, smoke=smoke, device=str(device), reference=str(REFERENCE), run_dir=str(reg_root),
                  run_status=runs[0]['status'], code_commit=commit, code_dirty=dirty, code_sha256=code_sha256(),
                  frozen_commit=paths.FROZEN_COMMIT, frozen_sha256=require_frozen_code())
    record['pass'] = runs[0]['status'] == 'ok' and all(i['pass'] for i in items.values()) and not smoke
    eg.atomic_record(outputs / 'gates/v4a_train.json', record)
    print(f'V4a (training half): pass={record["pass"]}; failing: {[k for k, v in items.items() if not v["pass"]]}',
          flush=True)
    return record


# ------------------------------------------------------------------------------------------------ command line
def main(argv=None):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')    # the CT queue's setting, before any CUDA work
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('train', 'held', 'regress'):
        cmd = sub.add_parser(command)
        cmd.add_argument('--outputs', type=Path, default=None,
                         help=f'default {paths.OUTPUTS}; with --smoke {paths.SMOKE_ROOT}/r10-rs-smoke')
        cmd.add_argument('--device', default=None, help='train: cuda; held: the device the weights were trained on')
        cmd.add_argument('--smoke', action='store_true', help='scratch outputs only; 5000 fit pairs, 2 epochs')
        if command == 'regress':
            cmd.add_argument('gate', choices=['v4a'])
        else:
            cmd.add_argument('--config', nargs='+', required=True, help='run names, e.g. k6=0.0005 k2=8+k6=0.0005')
            cmd.add_argument('--seeds', type=int, nargs='+', choices=SEEDS, default=list(SEEDS))
    args = parser.parse_args(argv)
    outputs = args.outputs or (paths.SMOKE_ROOT / 'r10-rs-smoke' if args.smoke else paths.OUTPUTS)
    if args.command == 'regress':
        record = regress_v4a(args.device or 'cuda', outputs, smoke=args.smoke)
        print(f'RS_FULL V4A pass={record["pass"]}', flush=True)
        return 0 if record['pass'] else 1
    status = 0
    for name in args.config:
        if args.command == 'train':
            result = train(name, args.seeds, args.device or 'cuda', outputs, smoke=args.smoke)
            for seed, record in result.items():
                print(f'RS_FULL train {name} s{seed} {record["status"]} {record.get("cause") or ""}'.rstrip(), flush=True)
                status |= record['status'] != 'ok'       # a failed stage 4 run leaves stage 4 with no outcome (V4d)
        else:
            held(name, args.seeds, outputs, args.device, smoke=args.smoke)
            print(f'RS_FULL held {name} done', flush=True)
    return int(status)


if __name__ == '__main__':
    raise SystemExit(main())
