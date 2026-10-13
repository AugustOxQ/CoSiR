"""Rule §6 items 1–5, in order, and the required C13 phase-1 agreement.

CPU only; prints check names, never metrics. --results controls all new outputs;
stage certificates and recipe files are read from their owners' normal locations.
A failed run is recorded once too. After a fix, use --fix 1 (R6 naming),
--after-crash, or --reserve; no evidence is overwritten.

Phase 1 is read, not launched: run rd9_compare.py --phase1 --backbone B first,
writing AR.AGREEMENT_DIR/rd9_<B>_phase1_agreement.json. Smoke agreements are
explicit R6-style stand-ins, copied from AR.AGREEMENT_DIR to the smoke output.
CLIP sensitivity is read directly from R6, exactly as the held runner does.
"""
import argparse
import contextlib
import io
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r9_common as R
import run_r9_held as RH
import run_r9_stage0a as S0A
import run_r9_stage0b as S0B
import run_r9_stage2 as S2
import run_r9_apply_rule as AR
import numpy as np

CHECK_ORDER = ('inputs', 'heads', 'regression', 'sensitivity', 'phase1', 'smoke')
VARIANTS = ('smoke', 'fix1', 'crash1', 'reserve')


def require(cond, msg):
    if not cond:
        raise RH.Refused(msg)


def write_once(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    AR.write_once(path, RH._plain(record))


def checked_json(path, evidence):
    path = Path(path).resolve()
    raw = path.read_bytes()
    evidence[str(path)] = R.sha256_bytes(raw)
    return json.loads(raw)


def remember_certified(record, evidence):
    """Retain the owner's already-validated provenance and receipt digests."""
    if not isinstance(record, dict):
        return
    if isinstance(record.get('path'), str) and Path(record['path']).is_absolute() and 'sha256' in record:
        evidence[str(Path(record['path']).resolve())] = record['sha256']
    for name, value in record.items():
        if name == 'feature_store_sha256':
            evidence.update({str(Path(p).resolve()): sha for p, sha in value.items()})
        elif isinstance(value, dict):
            remember_certified(value, evidence)


def preread_name(read, kind='smoke'):
    return f"preread_{read}{'' if kind == 'smoke' else '_' + kind}.json"


def input_read_set(pins=RH.AD.PINS_PATH, expected_languages=RH.AD.LANGUAGES):
    """Label-free projection for the existing A28 loader, scoped to inputs_only.

    It retains the owner's table/row/join checks. Sentinel columns satisfy that
    loader's full-table schema; they are discarded before A28Inputs receives the
    result. No emotion/style/genre column is decoded from parquet, and no sentinel
    can reach an episode: setup_a28 returns at the inputs_only boundary.
    """
    import pandas as pd
    import pyarrow.parquet as pq
    AD = RH.AD
    path = AD._pinned(AD._pin_files(pins), 'results/read_set_a28.parquet')
    require(pq.read_schema(path).names == list(AD.TABLE_COLUMNS), 'A28 parquet schema')  # guard:preread_projection_schema
    columns = [c for c in AD.TABLE_COLUMNS if c not in R.ASPECTS]
    tab = pd.read_parquet(path, columns=columns)
    for a in R.ASPECTS:
        tab[a] = np.full(len(tab), -1, dtype=np.int64)
    tab = tab[list(AD.TABLE_COLUMNS)]
    AD._check_table(tab, expected_languages)
    return tab


def inputs_gate(read, evidence):
    """Only inputs/certificates; pins are checked before any pick/cell loader."""
    backbone = RH.READ_BACKBONES[read]
    inputs = R.assert_inputs(backbone)
    evidence.update({str(Path(p).resolve()): sha for _, (p, sha) in R.input_pins(backbone).items()})
    evidence[str(AR.RULE_PATH.resolve())] = R.RULE_SHA256
    reg = checked_json(R.RESULTS / S0A.OUT_NAME, evidence)
    require(reg.get('passed') is True, 'stage 0a did not pass')  # guard:preread_stage0a_passed
    current = R.r9_module_shas()
    recorded = reg.get('module_sha256', {})
    require(recorded == current,
            'stage 0a modules changed')  # guard:preread_stage0a_modules
    require(reg.get('runner_sha256') == S0A.runner_sha256(),
            'stage 0a runner changed')  # guard:preread_stage0a_runner
    pins = checked_json(R.pins_path(backbone), evidence)
    paths = RH.BB.parts(backbone)
    # build_pins (the owner's full check) reads cells. Authenticate ALL parts
    # first so even its internal cell read happens after the pins SHA check.
    require(set(pins.get('parts', {})) == set(paths), 'incomplete parts')  # guard:preread_parts
    for role, path in paths.items():
        ent = pins['parts'][role]
        require(ent == {'path': RH.PINS.rel_path(path), 'sha256': R.sha256_file(path)},
                f'pinned part changed: {role}')  # guard:preread_part_sha
        evidence[str(Path(path).resolve())] = ent['sha256']
    RH.check_pins(backbone)  # guard:preread_pins
    remember_certified(pins, evidence)
    if backbone != 'clip_b32':
        gate = checked_json(R.RESULTS / S0B.OUT_NAME, evidence)
        S2.assert_modules(gate)  # guard:preread_stage0b
        stage = checked_json(pins['stage2']['path'], evidence)
        remember_certified(stage, evidence)
        require(stage.get('passed') is True and stage.get('backbone') == backbone
                and stage.get('switched_off') is False and stage.get('dry_run') is False,
                'stage 2 did not pass or is switched off')  # guard:preread_stage2
        for name in ('seed42_record', 'sensitivity_seed42'):
            entry = stage[name]
            checked_json(entry['path'], evidence)
            require(evidence[str(Path(entry['path']).resolve())] == entry['sha256'],
                    'stage 2 evidence changed')  # guard:preread_stage2_evidence
    if read in RH.A28_READS:
        RH.AD.check_pins(RH.AD.PINS_PATH, backbones=(backbone,))  # guard:preread_a28_pins
        a28 = checked_json(RH.AD.PINS_PATH, evidence)
        evidence.update({str(Path(v['path']).resolve()): v['sha256'] for v in a28['files'].values()})
    # Also required for the A28 smoke on H6/H7: no A28 rows may be scored.
    return SimpleNamespace(read=read, backbone=backbone, pins=pins, paths=paths, inputs=inputs)


def heads_gate(state, evidence):
    env = RH.setup(state.backbone)
    env.inputs = state.inputs
    require(env.head_check.get('passed') is True, 'selection head refit differs')  # guard:preread_heads
    RH.head_guard(env, RH.refit_reference(state.backbone))  # guard:preread_arm_l
    if state.backbone == 'clip_b32':
        checked_json(S0A.REFERENCES[S0A.REFIT_NAME][0], evidence)
    state.env = env
    return {'passed': True, 'coef_sha256': env.coef_sha256, 'selection': env.head_check}


def regression_gate(state, evidence):
    env, backbone, paths = state.env, state.backbone, state.paths
    picks = RH.P.load_picks(paths['picks_seed42'], backbone)
    lambdas = RH.P.frozen_lambdas(backbone)
    cell_record = RH.P.read_cells(paths['cells_seed42'])
    cells, taus = RH.S.cells_from_record(cell_record), RH.S.taus_of(backbone)
    RH.read_guard(state.read, backbone, cell_record)
    bundle = RH.read_seed_bundle(env, 'selection', R.DEV_SEED, R.N_PER_PAIR, backbone)
    RH.P.check_seed42(bundle)
    scored = RH.S.score_seed(bundle, picks, lambdas, env.readers, include_pm=True, cells=cells, taus=taus)
    ident = RH.E.identity_check(env.labels, env.split.groups, env.split.selection, env.index, env.value_sets)
    require(ident['passed'] is True, 'selection episode identity differs')  # guard:preread_episode_hashes
    if backbone == 'clip_b32':
        refs = S0A.load_references()
        for path, _ in S0A.REFERENCES.values():
            checked_json(path, evidence)
        # load_picks returns only numerical picks; the coefficient certificate
        # lives in the authenticated file, not in that parsed scoring interface.
        pick_record = checked_json(paths['picks_seed42'], evidence)
        items, _ = S0A.regression_items(env, bundle, scored, {'refit': refs[S0A.REFIT_NAME], 'picks': pick_record})
        extra = {'episodes_sha256': {42: bundle.episodes_sha256}, 'runner_sha256': RH.runner_sha256(),
                 'module_sha256': R.r9_module_shas()}
        counts = RH.ST.pass_record([scored], 'regression', [42], extra)
        counts.update(input_sha256=env.inputs, coef_sha256=env.coef_sha256)
        require(len(items) == S0A.N_ITEMS and all(v['equal'] is True for v in items.values())
                and not S0A.compare_counts(counts, refs[S0A.COUNTS_NAME]),
                'CLIP seed-42 regression differs')  # guard:preread_clip_regression
        hashes = {k: v for k, v in items.items() if '/' in k}
    else:
        got, _, _ = S2.seed42_record(env, bundle, scored, paths)
        want = state.pins['seed42']
        fields = ('seed', 'mean_r1', 'checks', 'arrays', 'episodes_sha256', 'coef_sha256')
        require(all(RH._plain(got[k]) == RH._plain(want[k]) for k in fields),
                'backbone seed-42 regression differs')  # guard:preread_regression
        hashes = got['arrays']
    state.scored, state.bundle = scored, bundle
    state.picks, state.lambdas, state.cells, state.taus = picks, lambdas, cells, taus
    return {'passed': True, 'episodes': ident, 'per_anchor': hashes}


def sensitivity_gate(state, evidence):
    diffs = RH.ST.check_diffs([state.scored], secondary=False)
    sigma = {c: RH.ST.sigma_split(*diffs[c]) for c in RH.ST.CHECKS}
    source = RH.sigma_guard(state.backbone, R.RESULTS)
    require(sigma == source['sigma'], 'seed-42 sensitivity differs')  # guard:preread_sigma
    # Same owner arithmetic used by the held runner, checked on actual seed-42
    # painting counts as well as on the variance components themselves.
    _, counts = np.unique(state.scored['cl'], return_counts=True)
    margins = RH.ST.sensitivity_record(sigma, counts, len(state.scored['cl']),
                                      'two_level' if state.read in RH.A28_READS else 'seven')
    evidence[str(Path(source['file']).resolve())] = source['sha256']
    state.sigma = source
    return {'passed': True, 'source': source, 'seed42_margins': margins}


def phase1_gate(state, evidence):
    prefix = AR.AGREEMENT_DIR / f'rd9_{state.backbone}'
    path = next((Path(str(prefix) + f'_phase1_agreement{s}.json') for s in ('_fix1', '_crash1', '')
                 if Path(str(prefix) + f'_phase1_agreement{s}.json').is_file()),
                Path(str(prefix) + '_phase1_agreement.json'))
    agr = checked_json(path, evidence)
    items = agr.get('items', [])
    require(agr.get('agreement') is True and agr.get('phase') == 1 and agr.get('backbone') == state.backbone
            and agr.get('failed') == 0 and bool(items) and agr.get('compared') == len(items)
            and all(v.get('equal') is True for v in items),
            'phase-1 agreement missing or failed')  # guard:preread_phase1
    require(agr.get('pins_sha256') == R.sha256_file(R.pins_path(state.backbone)),
            'phase-1 agreement pins changed')  # guard:preread_phase1_pins
    files = agr.get('independent_files', {'record': str(prefix) + '_phase1.json',
                                        'arrays': str(prefix) + '_seed42_arrays.npz'})
    record, arrays = Path(files['record']).resolve(), Path(files['arrays']).resolve()
    require(record.is_relative_to(AR.AGREEMENT_DIR.resolve()) and arrays.is_relative_to(AR.AGREEMENT_DIR.resolve()),
            'phase-1 evidence paths outside the owner output directory')  # guard:preread_phase1_paths
    derived = checked_json(record, evidence)
    evidence[str(arrays.resolve())] = R.sha256_file(arrays)
    require(agr.get('independent_identity') == {'record_sha256': evidence[str(record)],
                                              'arrays_sha256': evidence[str(arrays.resolve())]},
            'phase-1 derived evidence changed')  # guard:preread_phase1_identity
    require(derived.get('backbone') == state.backbone and derived.get('stats', {}).get('seeds') == [42]
            and derived.get('inputs', {}).get(str(R.pins_path(state.backbone)), {}).get('sha256') == agr['pins_sha256'],
            'phase-1 inputs do not bind current pins')  # guard:preread_phase1_inputs
    return {'passed': True, 'agreement': str(path), 'sha256': evidence[str(path.resolve())]}


def wiring_mutation(bundle, state, out):
    """Execute an isolated score-module copy, changing CF's G_cf to AFF's gated term.

    Uses the actual selection smoke bundle, not a second refit. The certified
    module is never edited. The isolated source is kept as mutation evidence.
    """
    source = Path(RH.S.__file__).read_text()
    old, new = '    G = cf_terms(gated)\n', '    G = gated  # smoke wiring mutation: AFF in CF\n'
    require(source.count(old) == 1, 'wiring mutation source anchor missing')  # guard:preread_wiring_anchor
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'r9_score.py'
    with path.open('x') as f:
        f.write(source.replace(old, new))
    mod = ModuleType('r9_score_mutation')
    mod.__file__ = str(path)
    saved_path = list(sys.path)
    try:
        exec(compile(path.read_text(), str(path), 'exec'), mod.__dict__)
    finally:
        sys.path[:] = saved_path
    fired = False
    try:
        mod.score_seed(bundle, state.picks, state.lambdas, state.env.readers, include_pm=False,
                       secondary=False, cells=state.cells, taus=state.taus)
    except (AssertionError, ValueError) as exc:
        fired = 'the CF term of cell' in str(exc) and 'condition-free' in str(exc)
    require(fired, 'wiring mutation did not fire the CF assertion')  # guard:preread_mutation
    return {'passed': True, 'fired': True, 'source': str(path), 'sha256': R.sha256_file(path)}


def smoke_agreement(out, read, tag):
    """R6's stand-in: an explicitly smoke-only phase-2 record, not a C13 result."""
    names = RH.read_names(read, 'smoke')
    rec = json.loads((out / names.pass_).read_text())
    agr = {'phase': 2, 'smoke': True, 'agreement': True, 'read': read, 'pass_file': names.pass_,
           'held_pass_sha256': R.sha256_file(out / names.pass_),
           'rederived_counts': {c: rec['checks'][c]['n'] for c in RH.ST.CHECKS},
           'items': [{'key': 'smoke_chain_stand_in', 'equal': True}], 'compared': 1, 'failed': 0,
           'what': 'R6-style smoke stand-in; no independent phase-2 re-derivation ran'}
    # Producer stays under the apply step's owned directory. Every attempt has
    # a distinct subdirectory; the apply step itself selects the consumer name.
    src = AR.AGREEMENT_DIR / 'smoke' / tag / f'rd9_{read}_agreement.json'
    write_once(src, agr)
    write_once(out / src.name, agr)
    return {'source': str(src), 'copy': str(out / src.name), 'sha256': R.sha256_file(src)}


def smoke_gate(state, evidence, results, kind):
    """Both selection smoke paths, apply, descriptive, real label-free inputs, mutation."""
    import run_r9_descriptive as RD
    results = Path(results)
    RH.AD.check_pins(RH.AD.PINS_PATH, backbones=(state.backbone,))  # guard:preread_smoke_a28_pins
    a28 = checked_json(RH.AD.PINS_PATH, evidence)
    evidence.update({str(Path(v['path']).resolve()): v['sha256'] for v in a28['files'].values()})
    checked_json(RH.VALUE_SETS_PATH, evidence)
    with contextlib.redirect_stdout(io.StringIO()):
        RH.setup_a28(state.env, state.backbone, smoke=False, inputs_only=True)  # guard:preread_real_inputs
    runs, first_bundle = {}, []
    scoring = RH.S.score_seed
    original_a28_report = RD.a28_report
    def projected_a28_report(*args):
        # The descriptive pass consumes synthetic selection scores. Stage-1
        # metadata use the same non-label projection as the inputs-only loader.
        # The owner's thin-value disclosures require real labels, so omit them
        # explicitly from this smoke artifact instead of reporting sentinel counts.
        with patch.object(RH.AD, 'load_read_set', input_read_set):
            report = original_a28_report(*args)
        if 'per_language' in report:
            for language in report['per_language'].values():
                language.pop('values_on_fewer_than_30_paintings', None)
            report['per_language_note'] = ('Synthetic selection episodes (row ID mod 27); real non-label stage-1 '
                                           'metadata. Thin-value disclosures omitted: no real labels decoded.')
        return report
    def capture(bundle, *args, **kwargs):
        if not first_bundle:
            first_bundle.append(bundle)
        return scoring(bundle, *args, **kwargs)
    held_read = next((r for r in ('H6', 'H7') if RH.READ_BACKBONES[r] == state.backbone), 'H6')
    a28_read = next(r for r in RH.A28_READS if RH.READ_BACKBONES[r] == state.backbone)
    for read in (held_read, a28_read):
        tag = f'{state.read}_{kind}_{read}'
        out = results / 'smoke' / tag
        require(not out.exists(), 'smoke folder exists')  # guard:preread_smoke_once
        with patch.object(RH.S, 'score_seed', capture), contextlib.redirect_stdout(io.StringIO()):
            # The new-backbone sigma is pinned in the owners' results; copy its
            # exact bytes once into scratch for RH.run_smoke's --results boundary.
            if state.backbone != 'clip_b32':
                target = results / Path(state.sigma['file']).name
                if target.exists():
                    require(R.sha256_file(target) == state.sigma['sha256'], 'scratch sigma differs')  # guard:preread_scratch_sigma
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open('xb') as f:
                        f.write(Path(state.sigma['file']).read_bytes())
                evidence[str(target.resolve())] = state.sigma['sha256']
            code = RH.run_smoke(state.backbone, read, results=results, subdir=tag, env=state.env)
        require(code == 0, 'held runner smoke failed')  # guard:preread_smoke_runner
        names = RH.read_names(read, 'smoke')
        if read in RH.A28_READS:
            # The held producer owns this unpinned metadata path. Retain the
            # exact digest it used rather than reconstructing its location.
            source = checked_json(out / names.pass_, {})['near_duplicates_source']
            evidence[str(Path(source['file']).resolve())] = source['sha256']
        agreement = smoke_agreement(out, read, tag)
        with contextlib.redirect_stdout(io.StringIO()):
            AR.apply(read, smoke=True, results=out)
            # RD's output root is its own module binding; its owner functions
            # retain their normal input roots. No global R.RESULTS is changed.
            with patch.object(RD, 'R', SimpleNamespace(**{**vars(R), 'RESULTS': results})), \
                    patch.object(RH, 'setup', lambda b: state.env), \
                    patch.object(RD, 'a28_report', projected_a28_report):
                descriptive, _ = RD.run(read, smoke=True, smoke_dir=out, smoke_tag=tag)
        runs[read] = {'passed': True, 'agreement': agreement,
                      'outputs': {str(p): R.sha256_file(p) for p in
                                  (out / names.pass_, out / names.arrays, out / names.verdict, descriptive)}}
    mutation = wiring_mutation(first_bundle[0], state, results / 'smoke' / f'{state.read}_{kind}_mutation')
    return {'passed': True, 'runs': runs, 'real_inputs_only': True, 'mutation': mutation}


def publish_smoke(record, results, kind):
    """Keep an existing current canonical record; each read also keeps its own proof."""
    path = Path(results) / AR.SMOKE_RECORDS[kind]
    copy = HERE / path.name
    for target in (path, copy):
        if target.exists():
            old = json.loads(target.read_text())
            require(old.get('passed') is True and old.get('module_sha256') == record['module_sha256']
                    and old.get('runner_sha256') == record['runner_sha256'],
                    'canonical smoke stale; use a fresh R6 variant')  # guard:preread_smoke_current
        else:
            write_once(target, record)
    AR.smoke_guard(Path(results), 'reserve' if kind == 'reserve' else None)
    return {'name': path.name, 'sha256': R.sha256_file(path)}


def run(read, results=None, kind='smoke'):
    results = Path(results or R.RESULTS).resolve()
    require(results.is_relative_to(R.CHECKOUT), '--results must stay inside this worktree')  # guard:preread_output_scope
    require(read in RH.READ_BACKBONES and kind in VARIANTS, 'unknown read/variant')  # guard:preread_read_identity
    path = results / preread_name(read, kind)
    require(not path.exists(), 'pre-read record exists; use a fresh R6 variant')  # guard:preread_once
    modules = R.r9_module_shas()
    rec = {'passed': False, 'read': read, 'backbone': RH.READ_BACKBONES[read], 'kind': kind,
           'runner_sha256': RH.runner_sha256(), 'preread_runner_sha256': R.sha256_file(__file__),
           'module_sha256': modules, 'input_file_sha256': {}, 'checks': {k: {'status': 'not_run'} for k in CHECK_ORDER},
           'time': R.amsterdam_now()}
    evidence = rec['input_file_sha256']
    active = 'inputs'
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            state = inputs_gate(read, evidence)
        rec['checks']['inputs'] = {'passed': True, 'status': 'passed', 'sha256': dict(evidence)}
        for active, check in (('heads', heads_gate), ('regression', regression_gate),
                              ('sensitivity', sensitivity_gate), ('phase1', phase1_gate)):
            print(f'pre-read {read}: {active}', flush=True)
            with contextlib.redirect_stdout(io.StringIO()):
                result = check(state, evidence)
            rec['checks'][active] = {**RH._plain(result), 'status': 'passed', 'sha256': dict(evidence)}
        active = 'smoke'
        print(f'pre-read {read}: smoke', flush=True)
        result = smoke_gate(state, evidence, results, kind)
        require(R.r9_module_shas() == modules and all(R.sha256_file(p) == s for p, s in evidence.items()),
                'bytes changed during pre-read')  # guard:preread_bytes
        rec['checks']['smoke'] = {**result, 'status': 'passed', 'sha256': dict(evidence)}
        smoke = {**rec, 'passed': True}
        own = results / f'preread_{read}_{kind}_smoke_record.json'
        write_once(own, smoke)
        rec['smoke_proof'] = {'path': str(own), 'sha256': R.sha256_file(own)}
        rec['smoke_record'] = publish_smoke(smoke, results, kind)
        rec['passed'] = True
    except (Exception, SystemExit) as exc:
        # Any owner failure must leave a failed immutable proof, including fit
        # errors or malformed record types; none may fall through to a launch.
        rec['checks'][active] = {'passed': False, 'status': 'failed', 'error_type': type(exc).__name__,
                                 'error': str(exc), 'sha256': dict(evidence)}
        rec['stopped_at'] = active
    write_once(path, rec)
    print(f'pre-read {read}: {"passed" if rec["passed"] else "stopped at " + active}; {path.name}')
    return 0 if rec['passed'] else RH.EXIT_REFUSE


def preread_guard(results, read, variant='held', after_crash=False, here=None):
    """Held launch gate, with the exact successful proof and its file/code bytes."""
    if variant in ('reserve', 'fix1'):
        path = Path(results) / preread_name(read, variant)
    else:
        # A failed pre-read's corrected pass can authorise the first launch too;
        # it is not a held --fix (which requires an already completed held pass).
        path = next((Path(results) / preread_name(read, k) for k in ('fix1', 'crash1')
                     if (Path(results) / preread_name(read, k)).is_file()),
                    Path(results) / preread_name(read))
    rec = json.loads(path.read_text()) if path.is_file() else {}
    require(rec.get('passed') is True and rec.get('read') == read
            and rec.get('backbone') == RH.READ_BACKBONES[read]
            and all(rec.get('checks', {}).get(k, {}).get('status') == 'passed' for k in CHECK_ORDER),
            'successful pre-read required before launch')  # guard:preread_launch_passed
    require(rec.get('module_sha256') == R.r9_module_shas(here)
            and rec.get('runner_sha256') == RH.runner_sha256(),
            'pre-read code changed')  # guard:preread_launch_code
    require(bool(rec.get('input_file_sha256'))
            and all(Path(p).is_file() and R.sha256_file(p) == sha for p, sha in rec['input_file_sha256'].items()),
            'pre-read inputs changed')  # guard:preread_launch_inputs
    proof = rec.get('smoke_proof', {})
    p = Path(proof.get('path', ''))
    require(p.is_file() and R.sha256_file(p) == proof.get('sha256'),
            'pre-read smoke proof changed')  # guard:preread_launch_smoke
    return {'name': path.name, 'sha256': R.sha256_file(path)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--read', required=True, choices=tuple(RH.READ_BACKBONES))
    ap.add_argument('--results', type=Path, default=R.RESULTS)
    flags = ap.add_mutually_exclusive_group()
    flags.add_argument('--fix', type=int, choices=(1,))
    flags.add_argument('--after-crash', action='store_true')
    flags.add_argument('--reserve', action='store_true')
    a = ap.parse_args(argv)
    kind = 'fix1' if a.fix else 'crash1' if a.after_crash else 'reserve' if a.reserve else 'smoke'
    try:
        return run(a.read, a.results, kind)
    except (RH.Refused, AR.Stop) as exc:
        print('pre-read refused before checks: ' + str(exc))
        return RH.EXIT_REFUSE


if __name__ == '__main__':
    sys.exit(main())
