"""Stage 2: run the fixed recipe CLIs, record seed 42, then pin one backbone.

CPU steps use eight threads. Groups and A3 run locally under flock; no DAS6 job
is launched. Each step is exclusive and resumable only at step boundaries.

  run_r9_stage2.py --backbone B --step {groups,a3,reader,baselines,cells,record,pins,all}
  run_r9_stage2.py --backbone clip_b32 --step record --dry-run --replay DIR --out TMP
  run_r9_stage2.py --backbone clip_b32 --step pins --dry-run --replay DIR --out TMP

The record step takes about 8-12 minutes on CLIP; run it separately from fits.
Dry run scores the read-only replay through the same chain and requires exact
CLIP numbers, counts, arrays and sensitivity components. Writes go only to TMP.
"""
import argparse
import contextlib
import hashlib
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r9_common as R  # noqa: E402
import r9_backbone as BB  # noqa: E402
import r9_bundle as B  # noqa: E402
import r9_context as X  # noqa: E402
import r9_heads as H  # noqa: E402
import r9_picks as P  # noqa: E402
import r9_score as S  # noqa: E402
import run_r9_stage0a as S0A  # noqa: E402
import run_r9_stage0b as S0B  # noqa: E402
import numpy as np  # noqa: E402

STEPS = ('groups', 'a3', 'reader', 'baselines', 'cells', 'record', 'pins')
EXEMPT = frozenset(('picked', 'picks_seed42', 'cells_seed42', 'baselines_seed42'))
STEP_SCRIPTS = dict(zip(STEPS[:5], S0B.RECIPE_SCRIPTS))
STEP_ROLES = {
    'groups': ('partitions', 'graph', 'bank_AIC', 'n6_posteriors', 'step1_heads_style'),
    'a3': ('A3_seed42', 'picked'),
    'reader': tuple(r for r in sorted(BB.RECIPE_ROLES) if r.startswith('rb_')) + ('rc_tau',),
    'baselines': ('baselines_seed42', 'picks_seed42'),
    'cells': ('cells_seed42',),
}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_new(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        f.write(json.dumps(S0B.plain(record), indent=1, allow_nan=False) + '\n')
    return path


def path_record(path):
    return {'path': str(Path(path).resolve()), 'sha256': R._sha_cached(path)}


def assert_array_provenance(got, clip, made, copied, *, replay=False):
    """Assert each fitted array differs, and each frozen array equals (dtype included)."""
    require(not set(made) & set(copied), 'made/copied keys overlap')
    for key in list(made) + list(copied):
        require(key in got and key in clip, f'provenance key missing: {key}')
        a, b = np.asarray(got[key]), np.asarray(clip[key])
        diff = S0B.array_diff(a, b, key)
        if key in copied or replay:
            require(not diff, f'copied {key} differs from CLIP: {diff}')
        else:
            require(bool(diff), f'made {key} equals CLIP')  # guard:made_differs


FROZEN_KEYS = {'prepare': ('groups', 'split_train', 'scorer_train', 'selection'),
               'held_codes': ('held_rows',), 'partition_L': ('partition_L',),
               'affect_km': ('affect', 'local_groups'),
               'step1_group_style': ('scorer_train', 'style_csd')}


def assert_frozen(role, path, clip):
    if Path(path).suffix == '.npz':
        with np.load(path, allow_pickle=False) as a, np.load(clip, allow_pickle=False) as b:
            keys = FROZEN_KEYS.get(role, b.files)
            assert_array_provenance(a, b, [], keys)
    elif role == 'pm_fit_rows':
        a, b = read_json(path), read_json(clip)
        require(a['fit_rows_sha256'] == b['fit_rows_sha256'], 'frozen PM row hash differs')
        require(a['fit_rows'] == b['fit_rows'], 'frozen PM fit rows differ')
    else:
        require(read_json(path) == read_json(clip), f'frozen {role} differs from CLIP')


def assert_script(record, stage0b, scripts_dir=HERE):
    require(stage0b.get('passed') is True, 'stage 0b has not passed')
    item = stage0b.get('item4', {})
    require(item.get('passed') is True, 'stage 0b script checks have not passed')
    name = Path(record.get('script', '')).name
    want = item.get('scripts', {}).get(name, {}).get('sha256')
    path = Path(scripts_dir) / name
    require(want and path.is_file() and record.get('script_sha256') == want == R.sha256_file(path),
            f'{name}: script SHA-256 differs from the passed stage 0b')  # guard:script_sha
    return want


def require_pinnable(backbone, parts, stage, *, recipe_roles=BB.RECIPE_ROLES):
    for role in recipe_roles:
        path = Path(parts[role])
        require(path.is_file() and Path(str(path) + '.record.json').is_file(), f'{role}: file or record missing')
    require(stage.get('passed') is True and stage.get('backbone') == backbone,
            f'{backbone}: stage 2 has not passed')  # guard:stage2_passed
    require(read_json(parts['picks_seed42']).get('passed') is True, 'picks file has not passed')  # guard:picks_passed


def _arrays(obj, prefix=''):
    """Numerical arrays of a checkpoint or sklearn pickle, retaining their names."""
    import torch
    if torch.is_tensor(obj):
        yield prefix, obj.detach().cpu().numpy()
    elif isinstance(obj, np.ndarray):
        yield prefix, obj
    elif isinstance(obj, dict):
        for key, value in obj.items():
            yield from _arrays(value, f'{prefix}/{key}' if prefix else str(key))
    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            yield from _arrays(value, f'{prefix}/{i}')
    elif hasattr(obj, '__dict__'):
        yield from _arrays(vars(obj), prefix)


def assert_part(role, path, clip, rec, *, replay=False):
    made, copied = rec['keys_made'], rec['keys_copied']
    if path.suffix == '.npz':
        with np.load(path, allow_pickle=False) as a, np.load(clip, allow_pickle=False) as b:
            unrefit = rec.get('keys_unrefit', [])
            require(set(a.files) == set(made) | set(copied) | set(unrefit), f'{role}: incomplete provenance keys')
            for k in unrefit:
                require(np.isnan(a[k]).all(), f'{role}/{k}: unused head must be NaN')
            # scipy's sparse container shape/format describe the frozen row space,
            # rather than a fitted array; the three numerical graph arrays are fitted.
            if role == 'graph':
                copied = list(copied) + [k for k in made if k in ('shape', 'format')]
                made = [k for k in made if k not in ('shape', 'format')]
            assert_array_provenance(a, b, made, copied, replay=replay)
    elif role == 'A3_seed42' or path.suffix == '.pkl':
        if role == 'A3_seed42':
            from r9_recipe_a3 import _state
            a, b = _state(path), _state(clip)
        else:
            with path.open('rb') as f, Path(clip).open('rb') as g:
                a, b = pickle.load(f), pickle.load(g)
        a, b = dict(_arrays(a)), dict(_arrays(b))
        require(a and set(a) == set(b), f'{role}: fitted array names differ')
        # classes and sample counts describe the frozen target/layout, not a fit.
        copied = [k for k in a if k.endswith(('/classes_', '/n_samples_seen_'))]
        made = [k for k in a if k not in copied and a[k].ndim > 0]
        assert_array_provenance(a, b, made, copied, replay=replay)
    elif role == 'rc_tau':
        assert_array_provenance({'taus': read_json(path)['taus']}, {'taus': read_json(clip)['taus']},
                                ['taus'], [], replay=replay)
    elif role == 'rb_reader_A0_json':
        # Coefficients and scaler arrays are checked in the paired pickle; this
        # JSON carries the fit's CV summaries and provenance.
        require(read_json(path)['feature_names'] == read_json(clip)['feature_names'], 'reader feature layout differs')
    else:
        require(role in EXEMPT, f'no provenance comparison for {role}')
    return {'keys_made': list(made), 'keys_copied': list(copied), 'exempt': role in EXEMPT,
            'keys_unrefit': rec.get('keys_unrefit', [])}


def _input_files(record, parts):
    """Validate actual file inputs, including A3's nested store/upstream schema."""
    def visit(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, dict):
                    yield from visit(value)
                elif isinstance(value, str) and len(value) == 64:
                    p = Path(key)
                    if p.is_absolute():
                        require(p.is_file() and R._sha_cached(p) == value, f'input changed: {p}')
                        yield str(p), value
                    elif key in R.INPUT_SHA256:
                        p = R.TEST / key
                        candidates = [p] + [Path(v) for r, v in parts.items() if BB.CLIP_PARTS[r] == p]
                        require(any(q.is_file() and R._sha_cached(q) == value for q in candidates),
                                f'logical input changed: {key}')
                        yield key, value
                    elif key in parts or key in {Path(v).name for v in parts.values()}:
                        candidates = [Path(v) for r, v in parts.items() if key in (r, Path(v).name)]
                        require(any(q.is_file() and R._sha_cached(q) == value for q in candidates),
                                f'upstream input changed: {key}')
                        yield key, value
    return dict(visit(record.get('input_sha256', {})))


def provenance(backbone, parts, stage0b, *, replay=False):
    from r9_recipe_groups import store_shas
    stores = store_shas(backbone)
    records = {r: read_json(str(parts[r]) + '.record.json') for r in BB.RECIPE_ROLES}
    by_hash = {R._sha_cached(parts[r]): r for r in BB.RECIPE_ROLES}

    def feature_lineage(role, seen=frozenset()):
        require(role not in seen, f'{role}: cyclic input provenance')
        observed = {}

        def visit(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, dict):
                        visit(value)
                    elif isinstance(value, str):
                        if key in stores:
                            require(value == stores[key], f'{role}: feature store hash differs: {key}')
                            observed[key] = value
                        elif value in by_hash:
                            observed.update(feature_lineage(by_hash[value], seen | {role}))
        visit(records[role].get('input_sha256', {}))
        visit(records[role].get('upstream_sha256', {}))
        return observed

    out = {}
    for role in sorted(BB.FROZEN_ROLES):
        assert_frozen(role, parts[role], BB.CLIP_PARTS[role])
    for role in sorted(BB.RECIPE_ROLES):
        path = Path(parts[role])
        recpath = Path(str(path) + '.record.json')
        rec = records[role]
        sha = R.sha256_file(path)
        require(rec.get('backbone', backbone) == backbone, f'{role}: record names another backbone')
        if 'file_sha256' in rec:
            require(rec['file_sha256'] == sha, f'{role}: file SHA differs from its record')
        expected_script = next(STEP_SCRIPTS[s] for s, roles in STEP_ROLES.items() if role in roles)
        require(Path(rec.get('script', '')).name == expected_script, f'{role}: record names the wrong producer')
        script = assert_script(rec, stage0b)
        require(bool(rec.get('input_sha256')), f'{role}: record has no inputs')
        inputs = _input_files(rec, parts)
        for upstream, want in rec.get('upstream_sha256', {}).items():
            require(upstream in parts and R._sha_cached(parts[upstream]) == want,
                    f'{role}: upstream {upstream} changed')
        features = feature_lineage(role)
        require(features == stores, f'{role}: feature-store provenance is incomplete')
        keys = assert_part(role, path, BB.CLIP_PARTS[role], rec, replay=replay)
        out[role] = {**path_record(path), 'record': path_record(recpath),
                     'feature_store_sha256': features, 'script': Path(rec['script']).name,
                     'script_sha256': script, 'input_sha256': rec['input_sha256'],
                     'checked_file_inputs': inputs, **keys}
    return out


def recipe_input_manifest(backbone):
    """Bootstrap the chain before final pins, keeping every frozen R6 hash."""
    clip = BB.CLIP_PARTS
    files = R.input_paths(backbone)
    replacements = {clip[r] for r in BB.RECIPE_ROLES}
    return {rel: {'path': str(path), 'sha256': (R.sha256_file(path) if R.TEST / rel in replacements
                  or rel.endswith('.record.json') else R.INPUT_SHA256[rel])}
            for rel, path in files.items()}


@contextlib.contextmanager
def input_manifest(backbone, manifest):
    """A recipe-only override, restored on exit; final pins are never provisional."""
    saved = R.input_pins
    resolved = R.input_paths(backbone)
    require(set(manifest) == set(resolved), 'stage 2 input manifest keys differ')
    pins = {}
    for rel, path in resolved.items():
        entry = manifest[rel]
        require(Path(entry['path']).resolve() == Path(path).resolve(), f'{rel}: stage 2 input path differs')
        sha = entry['sha256']
        require(R.sha256_file(path) == sha, f'{rel}: stage 2 input SHA differs')
        if path == R.TEST / rel:
            require(sha == R.INPUT_SHA256[rel], f'{rel}: frozen input SHA differs')
        pins[rel] = (path, sha)
    R.input_pins = lambda b='clip_b32': pins if b == backbone else saved(b)
    try:
        yield
    finally:
        R.input_pins = saved


def round6_statistics():
    """The actual R6 producers, not independently implemented statistics."""
    R.to_front((R.MAIN, R.R6D, R.R3D, R.R2D, R.R1D))
    import run_r6_held as RH
    import run_r6_sensitivity as RS
    import r6_stats as ST
    return RH, RS, ST


def seed42_record(env, bundle, scored, paths):
    RH, RS, ST = round6_statistics()
    extra = {'episodes_sha256': {R.DEV_SEED: dict(bundle.episodes_sha256)},
             'runner_sha256': R.sha256_file(__file__), 'module_sha256': R.r9_module_shas()}
    counts = ST.pass_record([scored], 'regression', [R.DEV_SEED], extra)
    counts.update(input_sha256=env.inputs, coef_sha256=env.coef_sha256)
    sigma = RS.sensitivity_record(RH.per_episode_arrays(scored))
    # Use stage 0a's own array construction and number functions (including the
    # mean computation), with its cell constants routed to this backbone's picks.
    saved = S.CELLS
    S.CELLS = S.cells_from_record(read_json(paths['cells_seed42']))
    try:
        nums, bar = S0A.number_items(scored, bundle.cl, bundle.pair_index)
        arrays = {f'seed42_arrays/{k}': v for k, v in S0A.round3_arrays(scored, bundle, bar).items()}
    finally:
        S.CELLS = saved
    arrays.update({'per_anchor_seed42/anchor_group': np.asarray(bundle.cl),
                   'per_anchor_seed42/pair_index': np.asarray(bundle.pair_index)})
    for s in S0A.AB_SCORERS:
        for m in S0A.METRICS:
            arrays[f'per_anchor_seed42/{s}__{m}'] = np.asarray(scored[s][m])
    require(len(arrays) == 96, f'stage 0a array keys: {len(arrays)}, expected 96')
    hashes = {k: {'sha256': hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest(),
                  'dtype': str(a.dtype), 'shape': list(a.shape)} for k, a in arrays.items()}
    means = {n: nums[f'{n}_mean_r1']['got'] for n in P.NESTED}
    means.update(AFF=nums['aff_fused_r1']['got'], CF=nums['aff_cf_r1']['got'])
    record = {'seed': 42, 'mean_r1': means, 'checks': {c: {'point': counts['checks'][c]['point'],
                'n_j': counts['checks'][c]['n']} for c in ST.CHECKS}, 'arrays': hashes,
              'episodes_sha256': dict(bundle.episodes_sha256), 'coef_sha256': env.coef_sha256,
              'pass_counts': counts, 'statistics_code': {Path(m.__file__).name: R.sha256_file(m.__file__)
                                                        for m in (RH, RS, ST)}}
    return record, sigma, arrays


def _score(backbone, paths):
    H.require_threads()
    env_rows = R.load_rows(backbone)
    heads = H.fit_heads_r9(env_rows.data, env_rows.split.groups, env_rows.split.scorer_train, backbone=backbone)
    check = H.check_selection(heads, env_rows.data, env_rows.split.selection, backbone)
    require(check['passed'], 'head selection check failed')
    pm = B.fit_pm(env_rows.data, env_rows.split.scorer_train, backbone)
    readers = B.load_readers(backbone)
    env = SimpleNamespace(**vars(env_rows), inputs=R.assert_inputs(backbone), heads=heads,
                          coef_sha256=H.coef_sha256(heads), readers=readers)
    ctx = X.RowContext('selection', 42, env.data, env.split, env.labels, heads, env.value_sets, R.N_PER_PAIR)
    require(dict(ctx.episode_sha) == S0A.E.identity_targets()[42]['episodes_sha256'], 'seed 42 episodes differ')
    bundle = B.build_bundle_r6(ctx, readers, pm, backbone)
    P.check_seed42(bundle)
    scored = S.score_seed(bundle, P.load_picks(paths['picks_seed42'], backbone),
                           P.parse_lambda_picks(read_json(paths['baselines_seed42'])), readers, include_pm=True,
                           cells=S.cells_from_record(read_json(paths['cells_seed42'])),
                           taus=tuple(read_json(paths['rc_tau'])['taus']))
    return seed42_record(env, bundle, scored, paths)


def clip_exact(record, sigma, arrays):
    refs = S0A.load_references()
    require(not S0A.compare_counts(record['pass_counts'], refs[S0A.COUNTS_NAME]), 'CLIP seed 42 pass counts differ')
    want = S0B.pinned_numbers(refs[S0A.COUNTS_NAME])['r1']
    require(record['mean_r1'] == want, 'CLIP seed 42 means differ')
    stored = read_json(R.R6D / 'results/sensitivity_seed42.json')
    require(sigma == {k: stored[k] for k in sigma}, 'CLIP sigma components differ')
    for prefix, rel in (('seed42_arrays/', S0A.R3_ARRAYS_REL), ('per_anchor_seed42/', S0A.PER_ANCHOR42_REL)):
        with np.load(R.INPUT_PATHS[rel]) as z:
            for name, array in arrays.items():
                if name.startswith(prefix):
                    key = name[len(prefix):]
                    require(not S0B.array_diff(array, z[key], name), f'CLIP seed 42 array differs: {name}')


def run_record(backbone, paths, results, stage0b_path, *, dry_run=False):
    stage_path = results / f'stage2_{backbone}.json'
    seed_path = results / f'stage2_seed42_{backbone}.json'
    sigma_path = results / f'sensitivity_seed42_{backbone}.json'
    for path in (stage_path, seed_path, sigma_path):
        require(not path.exists(), f'refusing to overwrite {path}')
    stage0b = read_json(stage0b_path)
    require_pinnable(backbone, paths, {'passed': True, 'backbone': backbone})
    if not dry_run:
        check_receipts(backbone)
    prov = provenance(backbone, paths, stage0b, replay=dry_run)
    cells = P.read_cells(paths['cells_seed42'])
    if dry_run:
        with S0B.routed(paths, cells=S.cells_from_record(cells)):
            seed, sigma, arrays = _score(backbone, paths)
    else:
        with input_manifest(backbone, recipe_input_manifest(backbone)):
            seed, sigma, arrays = _score(backbone, paths)
    if dry_run:
        clip_exact(seed, sigma, arrays)
    for c in ('P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7'):
        for key in ('sigma_a2', 'sigma_eps2'):
            value = sigma[c][key]
            require(np.isfinite(value) and value >= 0, f'{c}: invalid {key}')
    write_new(seed_path, seed)
    if dry_run:
        # Computation and exact component equality were checked above. Preserve
        # CLIP's original provenance fields/format too, so this dry-run file is
        # byte-for-byte the frozen sensitivity file, not merely numerically equal.
        with sigma_path.open('xb') as f:
            f.write((R.R6D / 'results/sensitivity_seed42.json').read_bytes())
    else:
        write_new(sigma_path, sigma)
    stage = {'backbone': backbone, 'passed': True, 'dry_run': dry_run,
             'status': 'not run: switched off' if cells['switched_off'] else 'ready',
             'switched_off': cells['switched_off'], 'gate_share': cells['gate_share'],
             'provenance': prov, 'frozen_parts': {r: path_record(paths[r]) for r in sorted(BB.FROZEN_ROLES)},
             'stage0b': path_record(stage0b_path), 'seed42_record': path_record(seed_path),
             'sensitivity_seed42': path_record(sigma_path), 'runner_sha256': R.sha256_file(__file__),
             'rule_sha256': R.RULE_SHA256, 'time': R.amsterdam_now()}
    if not dry_run:
        stage['recipe_receipts'] = {step: path_record(paths['partitions'].parent / f'stage2_{step}_files.json')
                                    for step in STEP_SCRIPTS}
    write_new(stage_path, stage)
    print(f"stage 2 {backbone}: {stage['status']}; record {stage_path}", flush=True)
    return stage


def pins_evidence(backbone, paths, results=None, stage0b_path=None, *, dry_run=False):
    results = Path(results or R.RESULTS)
    stage_path = results / f'stage2_{backbone}.json'
    require(stage_path.is_file(), f'{backbone}: stage 2 record missing')
    stage = read_json(stage_path)
    require_pinnable(backbone, paths, stage)
    require(stage.get('runner_sha256') == R.sha256_file(__file__) and stage.get('rule_sha256') == R.RULE_SHA256,
            'stage 2 runner or rule changed: rerun the record')
    require(stage['dry_run'] is dry_run, 'dry-run stage 2 cannot authorise a real backbone')
    recipe_files = {}
    if not dry_run:
        check_receipts(backbone)
        for step, receipt in stage['recipe_receipts'].items():
            require(path_record(receipt['path']) == receipt, f'{step}: receipt changed after stage 2')
            recipe_files[step] = read_json(receipt['path'])
    # A stage-2 pass names exact files and exact record bytes. Reject changes
    # after the assertions ran, rather than blessing whatever is now on disk.
    for role in BB.RECIPE_ROLES:
        p = stage['provenance'][role]
        require(path_record(paths[role]) == {k: p[k] for k in ('path', 'sha256')}, f'{role}: stage 2 file changed')
        require(path_record(str(paths[role]) + '.record.json') == p['record'], f'{role}: stage 2 record changed')
        for file, sha in p['feature_store_sha256'].items():
            require(R._sha_cached(file) == sha, f'feature store changed: {file}')
        _input_files(read_json(str(paths[role]) + '.record.json'), paths)
    for role in BB.FROZEN_ROLES:
        require(path_record(paths[role]) == stage['frozen_parts'][role], f'{role}: frozen file changed')
    stage0b_path = Path(stage0b_path or stage['stage0b']['path'])
    require(path_record(stage0b_path) == stage['stage0b'], 'stage 0b changed')
    gate = read_json(stage0b_path)
    for role in BB.RECIPE_ROLES:
        assert_script(read_json(str(paths[role]) + '.record.json'), gate)
    for name in ('seed42_record', 'sensitivity_seed42'):
        require(path_record(stage[name]['path']) == stage[name], f'{name}: changed after stage 2')
    return {'stage2': {**path_record(stage_path), 'passed': True, 'status': stage['status'],
                      'switched_off': stage['switched_off'], 'gate_share': stage['gate_share']},
            'provenance': stage['provenance'], 'seed42': read_json(stage['seed42_record']['path']),
            'seed42_record': stage['seed42_record'], 'sensitivity_seed42': stage['sensitivity_seed42'],
            'recipe_files': recipe_files}


def run_pins(backbone, paths, results, stage0b_path, *, dry_run=False):
    import r9_pins as PINS
    out = results / f'pins_{backbone}.json' if dry_run else R.pins_path(backbone)
    require(not out.exists(), f'refusing to overwrite {out}')
    pins = PINS.build_pins(backbone, part_paths=paths if dry_run else None,
                           stage2_results=results, stage0b_path=stage0b_path, dry_run=dry_run)
    write_new(out, pins)
    print(f'written {out} (SHA-256 {R.sha256_file(out)})', flush=True)
    return pins


def step_command(backbone, step):
    out = R.RESULTS / 'recipe' / backbone
    args = [sys.executable, str(HERE / STEP_SCRIPTS[step]), '--backbone', backbone, '--out', str(out)]
    if step == 'a3':
        args += ['--run', 'main', '--inputs', str(out)]
    elif step == 'reader':
        args += ['--stage', 'all', '--upstream', str(out)]
    elif step == 'baselines':
        args += ['--partitions-dir', str(out), '--a3-dir', str(out)]
    elif step == 'cells':
        # Execute the script's own __main__/argparse CLI, with its inputs routed
        # before import. The producer is unchanged, so its 0b SHA stays binding.
        bootstrap = (
            "import sys, runpy; from pathlib import Path; "
            "sys.argv=sys.argv[1:]; sys.path.insert(0,str(Path(sys.argv[0]).parent)); "
            "import run_r9_stage2 as D; "
            "b=sys.argv[sys.argv.index('--backbone')+1]; "
            "out=Path(sys.argv[sys.argv.index('--out')+1]); "
            "m=D.read_json(out/'stage2_inputs.json'); "
            "\nwith D.input_manifest(b,m): runpy.run_path(sys.argv[0],run_name='__main__')"
        )
        args = [sys.executable, '-c', bootstrap, *args[1:]]
    if step in ('groups', 'a3'):
        args = ['flock', '-n', '-o', '-E', '75', '/tmp/gpu0.lock', *args]
    return args


def run_step(backbone, step):
    out = R.RESULTS / 'recipe' / backbone
    receipt = out / f'stage2_{step}_files.json'
    require(not receipt.exists(), f'refusing to overwrite {receipt}')
    check_receipts(backbone, before=step)
    if step == 'cells':
        # The cells CLI consumes these hashes before final pins exist.
        write_new(out / 'stage2_inputs.json', recipe_input_manifest(backbone))
    env = dict(os.environ, OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', PYTHONDONTWRITEBYTECODE='1',
               CUDA_VISIBLE_DEVICES='0' if step in ('groups', 'a3') else '')
    code = subprocess.run(step_command(backbone, step), env=env).returncode
    if code:
        return code
    # Some legacy recipe records omit their output hash; capture it immediately
    # after the producer exits, alongside the producer's immutable record bytes.
    paths = BB.parts(backbone)
    rec = {r: {'file': path_record(paths[r]), 'record': path_record(str(paths[r]) + '.record.json')}
           for r in STEP_ROLES[step]}
    extras = {}
    if step == 'groups':
        extras['build_record.json'] = {'file': path_record(out / 'build_record.json')}
    elif step == 'a3':
        from r9_recipe_a3 import OUTPUTS
        for name in OUTPUTS:
            if name not in {Path(paths[r]).name for r in STEP_ROLES[step]}:
                extras[name] = {'file': path_record(out / name),
                                'record': path_record(out / (name + '.record.json'))}
    elif step == 'cells':
        extras['stage2_inputs.json'] = {'file': path_record(out / 'stage2_inputs.json')}
    write_new(receipt, {**rec, 'extra_files': extras})
    return 0


def check_receipts(backbone, before=None):
    """File and record hashes captured when each preceding CLI finished."""
    steps = list(STEP_SCRIPTS)
    for step in steps[:steps.index(before)] if before is not None else steps:
        receipt = R.RESULTS / 'recipe' / backbone / f'stage2_{step}_files.json'
        require(receipt.is_file(), f'{step}: driver receipt missing')
        rec = read_json(receipt)
        require(set(rec) == set(STEP_ROLES[step]) | {'extra_files'}, f'{step}: receipt roles differ')
        for role in STEP_ROLES[step]:
            entry = rec[role]
            path = BB.parts(backbone)[role]
            require(path_record(path) == entry['file'], f'{role}: output changed since it was made')
            require(path_record(str(path) + '.record.json') == entry['record'], f'{role}: producer record changed')
        for name, entry in rec['extra_files'].items():
            for kind, reference in entry.items():
                require(path_record(reference['path']) == reference, f'{name}: extra {kind} changed')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--backbone', choices=BB.BACKBONES, required=True)
    ap.add_argument('--step', choices=STEPS + ('all',), required=True)
    ap.add_argument('--stage0b', type=Path, default=R.RESULTS / 'stage0b.json')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--replay', type=Path)
    ap.add_argument('--out', type=Path, help='dry-run outputs only; recipe paths are fixed')
    a = ap.parse_args(argv)
    if a.dry_run:
        require(a.backbone == 'clip_b32' and a.step in ('record', 'pins') and a.replay and a.out,
                'dry run requires CLIP, --step record or pins, --replay, and --out')
        results = a.out.resolve()
        require(results.is_relative_to(HERE) and not results.is_relative_to(a.replay.resolve()),
                'dry-run writes must stay in this round folder and outside the replay')
        paths = {**BB.CLIP_PARTS, **S0B.replay_files(a.replay)}
    else:
        require(a.backbone != 'clip_b32' and not a.replay and not a.out, 'CLIP is only a read-only replay dry run')
        results, paths = R.RESULTS, BB.parts(a.backbone)
        gate = read_json(a.stage0b)
        require(gate.get('passed') is True, 'stage 0b has not passed: recipe cannot run')
        for name in S0B.RECIPE_SCRIPTS:
            assert_script({'script': name, 'script_sha256': R.sha256_file(HERE / name)}, gate)
    for step in STEPS if a.step == 'all' else (a.step,):
        if step in STEP_SCRIPTS:
            code = run_step(a.backbone, step)
            if code:
                return code
        elif step == 'record':
            run_record(a.backbone, paths, results, a.stage0b, dry_run=a.dry_run)
        else:
            run_pins(a.backbone, paths, results, a.stage0b, dry_run=a.dry_run)
    return 0


if __name__ == '__main__':
    sys.exit(main())
