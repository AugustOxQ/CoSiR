"""Lean stop scenarios; heavy fits/reads are stubbed, file boundaries use tmp_path.

Each named scenario has an actual deletion-mutant run in t22_mutation_results.json.
The wiring scenario uses the real scoring and CF assertion on small synthetic terms.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
HELD_GUARDS = {'preread_launch'}
sys.path.insert(0, str(HERE))
import run_r9_preread as U


def put(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))
    return path


def fail(message):
    raise U.RH.Refused(message)


@pytest.fixture
def case(tmp_path, monkeypatch):
    mp, root = monkeypatch, tmp_path
    mp.setattr(U.R, 'RESULTS', root)
    mp.setattr(U.R, 'CHECKOUT', root)
    mp.setattr(U.R, 'HERE', root)
    mp.setattr(U, 'HERE', root / 'folder')
    mp.setattr(U.R, 'r9_module_shas', lambda *a: {'recipe.py': 'a'})
    mp.setattr(U.S2, 'recipe_module_shas', lambda: {'recipe.py': 'a'})
    mp.setattr(U.R, 'assert_inputs', lambda b: {})
    mp.setattr(U.R, 'input_pins', lambda b: {})
    (root / 'DECISION_RULE.md').write_text('rule')
    part = root / 'part.json'
    part.write_text('{}')
    pins_path = root / 'pins.json'
    mp.setattr(U.R, 'pins_path', lambda b: pins_path)
    mp.setattr(U.RH.BB, 'parts', lambda b: {'cells': part})
    mp.setattr(U.RH.PINS, 'rel_path', lambda p: str(p))
    mp.setattr(U.RH, 'check_pins', lambda b: None)
    mp.setattr(U.S2, 'assert_modules', lambda gate: U.require(gate['passed'], 'stage 0b'))
    mp.setattr(U.RH.AD, 'PINS_PATH', root / 'pins_a28.json')
    mp.setattr(U.RH.AD, 'check_pins', lambda *a, **k: None)
    put(root / 'pins_a28.json', {'files': {}})
    reg = {'passed': True, 'module_sha256': {'recipe.py': 'a'}, 'runner_sha256': U.S0A.runner_sha256()}
    put(root / U.S0A.OUT_NAME, reg)
    pins = {'parts': {'cells': {'path': str(part), 'sha256': U.R.sha256_file(part)}}, 'seed42': {'seed': 42},
            'stage2': {'path': str(root / 'stage2_qwen3vl_emb_2b.json')}}
    put(pins_path, pins)
    put(root / U.S0B.OUT_NAME, {'passed': True})
    evidence = put(root / 'seed.json', {})
    entry = {'path': str(evidence), 'sha256': U.R.sha256_file(evidence)}
    stage = {'passed': True, 'backbone': 'qwen3vl_emb_2b', 'switched_off': False, 'dry_run': False,
             'seed42_record': entry.copy(), 'sensitivity_seed42': entry.copy()}
    put(root / 'stage2_qwen3vl_emb_2b.json', stage)
    return NS(root=root, reg=reg, pins=pins, stage=stage, part=part, pins_path=pins_path, mp=mp)


INPUT_GUARDS = ('stage0a_passed', 'stage0a_modules', 'stage0a_runner', 'parts', 'part_sha', 'pins',
                'stage0b', 'stage2', 'stage2_switched_off', 'stage2_dry_run', 'stage2_backbone',
                'stage2_evidence', 'a28_pins')


@pytest.mark.parametrize('guard', INPUT_GUARDS)
def test_input_stop(case, guard):
    c, read = case, 'H6'
    if guard == 'stage0a_passed':
        c.reg['passed'] = False
        put(c.root / U.S0A.OUT_COUNTS_NAME, {'passed': True})
    if guard == 'stage0a_modules': c.reg['module_sha256'] = {'recipe.py': 'bad'}
    if guard == 'stage0a_runner': c.reg['runner_sha256'] = 'bad'
    if guard == 'parts': c.pins['parts']['extra'] = {}
    if guard == 'part_sha': c.part.write_text('{"changed":true}')
    if guard == 'pins': c.mp.setattr(U.RH, 'check_pins', lambda b: fail('pins'))
    if guard == 'stage0b': put(c.root / U.S0B.OUT_NAME, {'passed': False})
    if guard == 'stage2': c.stage['passed'] = False
    if guard == 'stage2_switched_off': c.stage['switched_off'] = True
    if guard == 'stage2_dry_run': c.stage['dry_run'] = True
    if guard == 'stage2_backbone': c.stage['backbone'] = 'siglip2_so400m'
    if guard == 'stage2_evidence': c.stage['seed42_record']['sha256'] = 'bad'
    if guard == 'a28_pins':
        read = 'H8'
        c.mp.setattr(U.RH.AD, 'check_pins', lambda *a, **k: fail('A28 pins'))
    put(c.root / U.S0A.OUT_NAME, c.reg)
    put(c.pins_path, c.pins)
    put(c.root / 'stage2_qwen3vl_emb_2b.json', c.stage)
    calls = []
    c.mp.setattr(U.RH.P, 'load_picks', lambda *a: calls.append('picks'))
    c.mp.setattr(U.RH.P, 'read_cells', lambda *a: calls.append('cells'))
    with pytest.raises(U.RH.Refused):
        U.inputs_gate(read, {})
    assert not calls


@pytest.mark.parametrize('guard', ('heads', 'arm_l'))
def test_head_stop(case, guard):
    env = NS(head_check={'passed': guard != 'heads'}, coef_sha256={})
    case.mp.setattr(U.RH, 'setup', lambda b: env)
    case.mp.setattr(U.RH, 'refit_reference', lambda b: {})
    case.mp.setattr(U.RH, 'head_guard', lambda *a: fail('arm L') if guard == 'arm_l' else None)
    case.mp.setattr(U, 'checked_json', lambda *a: {})
    with pytest.raises(U.RH.Refused):
        U.heads_gate(NS(backbone='clip_b32', inputs={}), {})


def regression_state(c, clip=False):
    state = U.inputs_gate('H8' if clip else 'H6', {})
    state.paths = {k: c.part for k in ('picks_seed42', 'cells_seed42')}
    state.env = NS(labels={}, split=NS(groups=[], selection=[]), index=None, value_sets={},
                   readers={}, inputs={}, coef_sha256={}, head_check={'passed': True})
    c.mp.setattr(U.RH.P, 'load_picks', lambda *a: {})
    c.mp.setattr(U.RH.P, 'frozen_lambdas', lambda *a: {})
    c.mp.setattr(U.RH.P, 'read_cells', lambda *a: {'switched_off': False})
    c.mp.setattr(U.RH.P, 'check_seed42', lambda *a: None)
    c.mp.setattr(U.RH.S, 'cells_from_record', lambda r: {})
    c.mp.setattr(U.RH.S, 'taus_of', lambda b: ())
    c.mp.setattr(U.RH, 'read_seed_bundle', lambda *a: NS(episodes_sha256={}, cl=[], pair_index=[]))
    c.mp.setattr(U.RH.S, 'score_seed', lambda *a, **k: {})
    c.mp.setattr(U.RH.E, 'identity_check', lambda *a: {'passed': True})
    c.mp.setattr(U.S2, 'seed42_record', lambda *a: ({k: 'wrong' for k in
                 ('seed', 'mean_r1', 'checks', 'arrays', 'episodes_sha256', 'coef_sha256')}, None, None))
    state.pins['seed42'] = {k: 'right' for k in ('seed', 'mean_r1', 'checks', 'arrays', 'episodes_sha256', 'coef_sha256')}
    c.mp.setattr(U.S0A, 'load_references', lambda: {U.S0A.REFIT_NAME: {}, U.S0A.COUNTS_NAME: {}})
    c.mp.setattr(U, 'checked_json', lambda *a: {})
    c.mp.setattr(U.S0A, 'regression_items', lambda *a: ({'x': {'equal': False}}, None))
    c.mp.setattr(U.RH.ST, 'pass_record', lambda *a, **k: {})
    return state


SEED42_FIELDS = ('seed', 'mean_r1', 'checks', 'arrays', 'episodes_sha256', 'coef_sha256')


@pytest.mark.parametrize('guard', ('episode_hashes', 'clip_regression', 'regression'))
def test_regression_stop(case, guard):
    state = regression_state(case, guard == 'clip_regression')
    if guard == 'episode_hashes':
        case.mp.setattr(U.RH.E, 'identity_check', lambda *a: {'passed': False})
        case.mp.setattr(U.S2, 'seed42_record', lambda *a: (state.pins['seed42'], None, None))
    with pytest.raises(U.RH.Refused):
        U.regression_gate(state, {})


@pytest.mark.parametrize('field', SEED42_FIELDS)
def test_regression_stop_one_field_wrong(case, field):
    # Only this one pinned seed-42 field differs from the re-run; the other five agree.
    state = regression_state(case)
    got = dict(state.pins['seed42'])
    got[field] = 'wrong'
    case.mp.setattr(U.S2, 'seed42_record', lambda *a: (got, None, None))
    with pytest.raises(U.RH.Refused, match='backbone seed-42 regression differs'):
        U.regression_gate(state, {})


def test_regression_all_fields_equal_passes(case):
    state = regression_state(case)
    case.mp.setattr(U.S2, 'seed42_record', lambda *a: (dict(state.pins['seed42']), None, None))
    assert U.regression_gate(state, {})['passed'] is True


@pytest.mark.parametrize('wrong', ('items_unequal', 'items_short', 'counts'))
def test_clip_regression_stop_one_half_wrong(case, wrong):
    # CLIP: the item comparison and the counts comparison each stop alone.
    state = regression_state(case, clip=True)
    items = {str(i): {'equal': True} for i in range(U.S0A.N_ITEMS)}
    if wrong == 'items_unequal': items['0'] = {'equal': False}
    if wrong == 'items_short': items.pop('0')
    case.mp.setattr(U.S0A, 'regression_items', lambda *a: (items, None))
    case.mp.setattr(U.S0A, 'compare_counts', lambda *a: ['differs'] if wrong == 'counts' else [])
    with pytest.raises(U.RH.Refused, match='CLIP seed-42 regression differs'):
        U.regression_gate(state, {})


def test_sigma_stop(case):
    case.mp.setattr(U.RH.ST, 'check_diffs', lambda *a, **k: {c: ([], []) for c in U.RH.ST.CHECKS})
    case.mp.setattr(U.RH.ST, 'sigma_split', lambda *a: {'sigma_a2': 1, 'sigma_eps2': 2})
    case.mp.setattr(U.RH, 'sigma_guard', lambda *a: {'sigma': {}})
    with pytest.raises(U.RH.Refused, match='sensitivity differs'):
        U.sensitivity_gate(NS(scored={}, backbone='clip_b32'), {})


def phase1(case):
    c = case
    c.mp.setattr(U.AR, 'AGREEMENT_DIR', c.root)
    backbone = 'clip_b32'
    source = c.root / f'rd9_{backbone}_phase1.json'
    arrays = c.root / f'rd9_{backbone}_seed42_arrays.npz'
    arrays.write_bytes(b'array')
    pinsha = U.R.sha256_file(c.pins_path)
    derived = {'backbone': backbone, 'stats': {'seeds': [42]}, 'inputs': {str(c.pins_path): {'sha256': pinsha}}}
    put(source, derived)
    agr = {'agreement': True, 'phase': 1, 'backbone': backbone, 'failed': 0, 'compared': 1,
           'items': [{'equal': True}], 'pins_sha256': pinsha,
           'independent_identity': {'record_sha256': U.R.sha256_file(source), 'arrays_sha256': U.R.sha256_file(arrays)}}
    return NS(backbone=backbone), agr, derived, source


@pytest.mark.parametrize('guard', ('phase1', 'phase1_phase', 'phase1_backbone', 'phase1_failed', 'phase1_item',
                                   'phase1_no_items', 'phase1_compared', 'phase1_pins', 'phase1_identity',
                                   'phase1_inputs', 'phase1_paths'))
def test_phase1_stop(case, guard):
    state, agr, derived, source = phase1(case)
    if guard == 'phase1': agr['agreement'] = False
    if guard == 'phase1_phase': agr['phase'] = 2
    if guard == 'phase1_backbone': agr['backbone'] = 'qwen3vl_emb_2b'
    if guard == 'phase1_failed': agr['failed'] = 1
    if guard == 'phase1_item': agr.update(items=[{'equal': True}, {'equal': False}], compared=2)
    if guard == 'phase1_no_items': agr.update(items=[], compared=0)
    if guard == 'phase1_compared': agr['compared'] = 2
    if guard == 'phase1_pins':
        agr['pins_sha256'] = 'bad'
        derived['inputs'][str(case.pins_path)]['sha256'] = 'bad'
        put(source, derived)
        agr['independent_identity']['record_sha256'] = U.R.sha256_file(source)
    if guard == 'phase1_identity': agr['independent_identity']['arrays_sha256'] = 'bad'
    if guard == 'phase1_inputs':
        derived['inputs'] = {}
        put(source, derived)
        agr['independent_identity']['record_sha256'] = U.R.sha256_file(source)
    if guard == 'phase1_paths':
        agr['independent_files'] = {'record': str(case.root.parent / 'outside_phase1.json'),
                                    'arrays': str(source.with_suffix('.npz'))}
    put(case.root / 'rd9_clip_b32_phase1_agreement.json', agr)
    with pytest.raises(U.RH.Refused):
        U.phase1_gate(state, {})


def test_once_stop(case):
    run_stubs(case)
    path = put(case.root / U.preread_name('H8'), {'old': True})
    old = path.read_bytes()
    with pytest.raises(U.RH.Refused, match='record exists'):
        U.run('H8', case.root)
    assert path.read_bytes() == old


@pytest.mark.parametrize('guard', ('real_inputs', 'smoke_once', 'smoke_runner', 'smoke_a28_pins', 'scratch_sigma'))
def test_smoke_stop(case, guard):
    c = case
    state = NS(read='H8', backbone='clip_b32', env=None)
    case.mp.setattr(U.RH.AD, 'results_dir', lambda: case.root)
    put(case.root / 'near_duplicates_a28.json', {})
    case.mp.setattr(U.RH, 'VALUE_SETS_PATH', put(case.root / 'values.json', {}))
    if guard == 'smoke_a28_pins':
        c.mp.setattr(U.RH.AD, 'check_pins', lambda *a, **k: fail('A28 pins'))
    if guard == 'scratch_sigma':
        state.backbone = 'qwen3vl_emb_2b'
        sigma = put(c.root / 'sensitivity_seed42_qwen3vl_emb_2b.json', {})
        state.sigma = {'file': str(sigma), 'sha256': 'bad'}
    if guard == 'smoke_once': (c.root / 'smoke/H8_smoke_H6').mkdir(parents=True)
    c.mp.setattr(U.RH, 'setup_a28', lambda *a, **k: fail('real inputs') if guard == 'real_inputs' else None)
    c.mp.setattr(U.RH, 'run_smoke', lambda *a, **k: 4)
    if guard == 'scratch_sigma':
        c.mp.setattr(U.RH, 'run_smoke', lambda *a, **k: 0)
    if guard not in ('smoke_runner', 'scratch_sigma'):
        class Traversed(Exception): pass
        c.mp.setattr(U.RH, 'run_smoke', lambda *a, **k: (_ for _ in ()).throw(Traversed()))
    c.mp.setattr(U, 'smoke_agreement', lambda *a: {})
    c.mp.setattr(U.AR, 'apply', lambda *a, **k: None)
    import run_r9_descriptive as RD
    c.mp.setattr(RD, 'run', lambda *a, **k: (c.part, {}))
    # If runner assertion is removed, output processing must proceed without an
    # unrelated filesystem failure; expose that traversal with a sentinel.
    if guard in ('smoke_runner', 'scratch_sigma'):
        class Reached(Exception): pass
        c.mp.setattr(U, 'smoke_agreement', lambda *a: (_ for _ in ()).throw(Reached()))
    with pytest.raises(U.RH.Refused):
        U.smoke_gate(state, {}, c.root, 'smoke')


def test_smoke_current_stop(case):
    put(case.root / U.AR.SMOKE_RECORDS['smoke'], {'passed': True, 'module_sha256': {'recipe.py': 'old'}})
    with pytest.raises(U.RH.Refused, match='canonical smoke stale'):
        U.publish_smoke({'module_sha256': {'recipe.py': 'a'}, 'runner_sha256': 'a'}, case.root, 'smoke')


def run_stubs(c):
    state = NS()
    c.mp.setattr(U, 'inputs_gate', lambda *a: state)
    for name in ('heads_gate', 'regression_gate', 'sensitivity_gate', 'phase1_gate'):
        c.mp.setattr(U, name, lambda *a: {'passed': True})
    c.mp.setattr(U, 'smoke_gate', lambda *a: {'passed': True})
    c.mp.setattr(U, 'publish_smoke', lambda *a: {'name': 'record', 'sha256': 'a'})


def test_bytes_stop(case):
    run_stubs(case)
    def smoke(*a):
        case.mp.setattr(U.R, 'r9_module_shas', lambda *a: {'recipe.py': 'changed'})
        return {'passed': True}
    case.mp.setattr(U, 'smoke_gate', smoke)
    assert U.run('H8', case.root) == U.RH.EXIT_REFUSE
    rec = json.loads((case.root / U.preread_name('H8')).read_text())
    assert rec['stopped_at'] == 'smoke'


@pytest.mark.parametrize('stop', U.CHECK_ORDER)
def test_order_stops_and_records_skips(case, stop):
    run_stubs(case)
    calls = []
    functions = ('inputs_gate', 'heads_gate', 'regression_gate', 'sensitivity_gate', 'phase1_gate', 'smoke_gate')
    for name, label in zip(functions, U.CHECK_ORDER):
        def check(*args, label=label):
            calls.append(label)
            if label == stop: fail(label)
            return NS() if label == 'inputs' else {'passed': True}
        case.mp.setattr(U, name, check)
    assert U.run('H8', case.root) == U.RH.EXIT_REFUSE
    i = U.CHECK_ORDER.index(stop)
    assert calls == list(U.CHECK_ORDER[:i + 1])
    rec = json.loads((case.root / U.preread_name('H8')).read_text())
    assert rec['checks'][stop]['status'] == 'failed'
    assert all(rec['checks'][k]['status'] == 'not_run' for k in U.CHECK_ORDER[i + 1:])


def own_files(case, a28=True):
    """This checkout's pins, stage 0a record (and the A28 pins), as a record keys them: resolved path -> SHA-256."""
    paths = [case.pins_path, case.root / U.S0A.OUT_NAME] + ([case.root / 'pins_a28.json'] if a28 else [])
    return {str(p.resolve()): U.R.sha256_file(p) for p in paths}


def launch_record(case):
    proof = put(case.root / 'proof.json', {'passed': True})
    rec = {'passed': True, 'read': 'H8', 'backbone': 'clip_b32',
           'checks': {k: {'status': 'passed'} for k in U.CHECK_ORDER},
           'module_sha256': U.R.r9_module_shas(), 'runner_sha256': U.RH.runner_sha256(),
           'input_file_sha256': {str(case.part): U.R.sha256_file(case.part), **own_files(case)},
           'smoke_proof': {'path': str(proof), 'sha256': U.R.sha256_file(proof)}}
    return rec


LAUNCH_GUARDS = ('launch_passed', 'launch_code', 'launch_inputs', 'launch_smoke', 'launch_modules', 'launch_read',
                 'launch_backbone', 'launch_check_not_run', 'launch_no_inputs', 'launch_other_worktree_inputs',
                 'launch_no_pins', 'launch_no_stage0a', 'launch_no_a28_pins')


@pytest.mark.parametrize('guard', LAUNCH_GUARDS)
def test_launch_stop(case, guard):
    rec = launch_record(case)
    if guard == 'launch_passed': rec['passed'] = False
    if guard == 'launch_code': rec['runner_sha256'] = 'bad'
    if guard == 'launch_inputs': case.part.write_text('changed')
    if guard == 'launch_smoke': rec['smoke_proof']['sha256'] = 'bad'
    # C MEDIUM-1: one sub-condition at a time, each with the others true.
    if guard == 'launch_modules': rec['module_sha256'] = {'recipe.py': 'older'}
    if guard == 'launch_read': rec['read'] = 'H6'
    if guard == 'launch_backbone': rec['backbone'] = 'qwen3vl_emb_2b'
    if guard == 'launch_check_not_run': rec['checks']['smoke'] = {'status': 'not_run'}
    if guard == 'launch_no_inputs': rec['input_file_sha256'] = {}
    # C LOW-4: a passing record copied from another worktree keys that worktree's files.
    if guard == 'launch_other_worktree_inputs':
        rec['input_file_sha256'] = {str(case.part): U.R.sha256_file(case.part)}
        for path, sha in own_files(case).items():
            other = case.root / 'other_worktree' / Path(path).name
            put(other, json.loads(Path(path).read_text()))
            rec['input_file_sha256'][str(other.resolve())] = U.R.sha256_file(other)
    own = {'launch_no_pins': case.pins_path, 'launch_no_stage0a': case.root / U.S0A.OUT_NAME,
           'launch_no_a28_pins': case.root / 'pins_a28.json'}
    if guard in own: rec['input_file_sha256'].pop(str(own[guard].resolve()))
    put(case.root / U.preread_name('H8'), rec)
    with pytest.raises(U.RH.Refused):
        U.preread_guard(case.root, 'H8')


def test_launch_a28_pins_not_needed_for_a_held_read(case):
    # H6 reads no ArtELingo-28 file: its record need not (and does not) list pins_a28.
    rec = launch_record(case)
    rec.update(read='H6', backbone='qwen3vl_emb_2b')
    rec['input_file_sha256'] = {str(case.part): U.R.sha256_file(case.part), **own_files(case, a28=False)}
    put(case.root / U.preread_name('H6'), rec)
    assert U.preread_guard(case.root, 'H6')['name'] == U.preread_name('H6')


def wiring(case):
    # Real assembly, metric calculation, and per-cell assertion, with synthetic
    # reader output. Only the trained reader is stubbed; no real data is opened.
    S = U.RH.S
    n, k = 3 * U.R.N_SMOKE, 13
    rng = np.random.default_rng(22)
    def term(free):
        x = {d: rng.normal(size=(n, k)).astype(np.float32) for d in S.DIRECTIONS}
        return {c: {d: x[d].copy() if free or c == 'a' else -x[d] for d in S.DIRECTIONS} for c in S.CONDITIONS}
    base, conditional = term(True), term(False)
    bundle = NS(n=n, mode='selection', seed=9001, parity=np.arange(n) % 2, cos=base,
                pm_terms={'rca': conditional}, t_n1u=base, t6u_B=base, t6u_B0=base,
                cl=np.arange(n), pair_index=np.repeat(np.arange(3), U.R.N_SMOKE))
    rd = {'T': conditional, 'P': {}, 'm': {c: np.ones(n) for c in S.CONDITIONS},
          'pick': {c: np.zeros(n, dtype=np.int64) for c in S.CONDITIONS}}
    case.mp.setattr(S.B, 'check_readers', lambda *a: None)
    case.mp.setattr(S.RF, 'reader', lambda *a: rd)
    state = NS(picks={x: {0: [0., 0.], 1: [0., 0.]} for x in ('B', 'B0')},
               lambdas={'rca': {0: 0., 1: 0.}}, env=NS(readers={}), cells=S.CELLS,
               taus=(-1., 0., 0.5, 0.9))
    # Unmutated control really scores and its CF gain is zero on every episode.
    scored = S.score_seed(bundle, state.picks, state.lambdas, {}, include_pm=False,
                          secondary=False, cells=state.cells, taus=state.taus)
    assert np.all(scored['aff_cf']['gain'] == 0)
    return bundle, state


def test_wiring_mutation_fires(case):
    bundle, state = wiring(case)
    result = U.wiring_mutation(bundle, state, case.root / 'mutation')
    assert result['fired'] is True


def test_mutation_stop(case):
    bundle, state = wiring(case)
    # Remove the mutation itself; an unchanged scoring copy must be refused.
    source = Path(U.RH.S.__file__).read_text()
    case.mp.setattr(Path, 'read_text', lambda p, *a, **k: source if p.name == 'r9_score.py' else '{}')
    with pytest.raises(U.RH.Refused, match='mutation did not fire'):
        U.wiring_mutation(bundle, state, case.root / 'control')


def test_held_launch_calls_preread_before_read(case):
    RH = U.RH
    case.mp.setattr(RH, 'refuse_or_go', lambda *a: NS())
    seen = []
    class Traversed(Exception): pass
    case.mp.setattr(RH, 'smoke_guard', lambda *a: (_ for _ in ()).throw(Traversed()))
    case.mp.setattr(RH, 'run_read', lambda *a: seen.append('read'))
    assert RH.run_held('clip_b32', 'H8', results=case.root) == RH.EXIT_REFUSE
    assert seen == []


def test_real_branch_projects_no_label_and_only_builds_inputs(case):
    import pandas as pd
    import pyarrow.parquet as pq
    AD = U.RH.AD
    table = pd.DataFrame({'row_id': [0, 1], 'painting': ['p', 'q'], 'language': ['Burmese', 'Darija'],
                          'emotion': [99, 98], 'style': [97, 96], 'genre': [95, 94],
                          'held': [False, True], 'dropped': [False, True]})[list(AD.TABLE_COLUMNS)]
    path = case.root / 'table.parquet'
    table.to_parquet(path)
    pins = {'files': {'results/read_set_a28.parquet': {'path': str(path), 'sha256': U.R.sha256_file(path)}}}
    original = pd.read_parquet
    calls = []
    def projected(path, *, columns):
        assert not set(columns) & set(U.R.ASPECTS)
        calls.append(columns)
        return original(path, columns=columns)
    case.mp.setattr(pd, 'read_parquet', projected)
    def load(*args):
        tab = AD.load_read_set(pins, ['Burmese', 'Darija'])
        assert all((tab[a] == -1).all() for a in U.R.ASPECTS)
        return NS(**{a: tab[a].to_numpy() for a in U.R.ASPECTS}, n=2)
    case.mp.setattr(AD, 'load_a28', load)
    case.mp.setattr(U.RH.AC, 'A28Inputs', lambda real, heads: real)
    case.mp.setattr(U.RH.B, 'model_inputs', lambda inputs, b: ('codes',))
    result = U.RH.setup_a28(NS(heads={}), 'clip_b32', False, inputs_only=True)
    assert result.codes == ('codes',)
    assert not any(hasattr(result.inputs, a) for a in U.R.ASPECTS)
    assert len(calls) == 1
    assert AD.load_read_set is not U.input_read_set


def comparer():
    sys.path.insert(0, str(HERE / 'rederive'))
    import rd9_compare
    return rd9_compare


@pytest.mark.parametrize('guard', ('compare_pins', 'compare_sources'))
def test_comparer_stop(case, guard):
    C = comparer()
    inputs = {str(case.pins_path): {'sha256': U.R.sha256_file(case.pins_path)},
              str(case.part): {'sha256': U.R.sha256_file(case.part)}}
    if guard == 'compare_pins': inputs.pop(str(case.pins_path))
    else: case.part.write_text('changed')
    with pytest.raises(AssertionError):
        C.phase1_certificate({'inputs': inputs}, case.pins_path)


def test_comparer_accepts_stage2_namespaces_and_detects_changed_hash():
    C = comparer()
    from rd9_inputs import sha_bytes
    arrays = {f'{arm}__{m}': np.array([.25, .5]) for arm in ('B', 'B0', 'B1', 'aff_fused', 'aff_cf', 'cosine', 'rca')
              for m in ('r1', 'gain', 'other', 'swap', 'strict')}
    arrays.update({k: np.array([0, 1], dtype=np.int64) for k in
                   ('cl', 'pair_index', 'parity', 'aff_fused_cells', 'aff_cf_cells')})
    arrays.update({f'{prefix}__{c}': np.array([.25, .5]) for prefix in ('aff_gate', 'P', 'margin', 'pick') for c in ('a', 'b')})
    own = {'picks': {'aff': [1, 2], 'cf': [3, 4], 'sigma_star': [0., 0.], 'switched_off': False,
                    'gate_share': .5, 'B': [[0., 0.]] * 2, 'B0': [[0., 0.]] * 2,
                    'rca_lambda': [0., 0.], 'taus': [-1., 0., .5, .9]},
           'stats': {'checks': {c: {'point': 1., 'n': 0} for c in U.RH.ST.CHECKS}, 'n_episodes': 2, 'n_clusters': 2,
                     'sensitivity': {c: {'sigma_a2': .5, 'sigma_e2': .25} for c in U.RH.ST.CHECKS}}}
    sigma = {c: {'sigma_a2': .5, 'sigma_eps2': .25} for c in U.RH.ST.CHECKS} | {'N': 2, 'n_paintings': 2}
    cells = {k: own['picks'][k] for k in ('aff', 'cf', 'sigma_star', 'switched_off', 'gate_share')}
    picks = {k: {'0': [0., 0.], '1': [0., 0.]} for k in ('B', 'B0')}
    seed = {'mean_r1': {k: 37.5 for k in ('B', 'B0', 'B1', 'AFF', 'CF')},
            'checks': {c: {'point': 1., 'n_j': 0} for c in U.RH.ST.CHECKS},
            'arrays': {('per_anchor_seed42/anchor_group' if k == 'cl' else
                        'per_anchor_seed42/' + k if k.startswith(('cosine__', 'rca__')) else 'seed42_arrays/' + k):
                       {'sha256': sha_bytes(v), 'dtype': str(v.dtype), 'shape': list(v.shape)}
                       for k, v in arrays.items() if not k.startswith(('B__', 'B0__', 'B1__'))}}
    def check():
        comp = C.Comparison()
        C.compare_phase1(comp, own, arrays, cells, picks, {'scorers': {'rca': {'lambda_picks': {'0': 0., '1': 0.}}}},
                         {'taus': own['picks']['taus']}, seed, sigma)
        return comp.record()
    assert check()['agreement'] is True
    sigma['P1']['sigma_a2'] = 9.    # fx-c13: the seed-42 sigma inputs are compared
    assert check()['agreement'] is False
    sigma['P1']['sigma_a2'] = .5
    seed['arrays']['seed42_arrays/aff_fused__r1']['sha256'] = '0' * 64
    assert check()['agreement'] is False


@pytest.mark.parametrize('kind', ('fix1', 'crash1', 'reserve'))
def test_corrected_preread_names_launch(case, kind):
    rec = launch_record(case)
    put(case.root / U.preread_name('H8'), {'passed': False})
    put(case.root / U.preread_name('H8', kind), rec)
    variant = 'reserve' if kind == 'reserve' else 'held'
    got = U.preread_guard(case.root, 'H8', variant)
    assert got['name'] == U.preread_name('H8', kind)


def test_new_failed_fix_preread_blocks_old_pass(case):
    put(case.root / U.preread_name('H8'), launch_record(case))
    put(case.root / U.preread_name('H8', 'fix1'), {'passed': False})
    with pytest.raises(U.RH.Refused):
        U.preread_guard(case.root, 'H8')


def test_projection_schema_stop(case):
    import pandas as pd
    AD = U.RH.AD
    tab = pd.DataFrame({'row_id': [0], 'painting': ['p'], 'language': ['Burmese'],
                        'emotion': [0], 'style': [0], 'genre': [0], 'held': [False], 'dropped': [False],
                        'extra': [1]})
    path = case.root / 'bad_schema.parquet'
    tab.to_parquet(path)
    pins = {'files': {'results/read_set_a28.parquet': {'path': str(path), 'sha256': U.R.sha256_file(path)}}}
    with pytest.raises(U.RH.Refused, match='parquet schema'):
        U.input_read_set(pins, ['Burmese'])


def test_wiring_anchor_stop(case):
    bundle, state = wiring(case)
    source = Path(U.RH.S.__file__).read_text().replace('G = cf_terms(gated)', 'G = (cf_terms(gated))')
    case.mp.setattr(Path, 'read_text', lambda p, *a, **k: source if p.name == 'r9_score.py' else '{}')
    with pytest.raises(U.RH.Refused, match='anchor missing'):
        U.wiring_mutation(bundle, state, case.root / 'anchor')


def test_output_scope_stop(case):
    run_stubs(case)
    with pytest.raises(U.RH.Refused, match='inside this worktree'):
        U.run('H8', case.root.parent / 'outside_case')


def test_read_identity_stop(case):
    run_stubs(case)
    with pytest.raises(U.RH.Refused, match='unknown read'):
        U.run('H11', case.root)


def test_clip_regression_uses_authenticated_raw_pick_certificate(case):
    state = regression_state(case, clip=True)
    raw = {'coef_sha256': {'affect': 'certificate'}}
    put(state.paths['picks_seed42'], raw)
    case.mp.setattr(U, 'checked_json', lambda path, evidence: json.loads(Path(path).read_text())
                   if path == state.paths['picks_seed42'] else {})
    def items(env, bundle, scored, orders):
        assert orders['picks'] == raw
        return ({str(i): {'equal': True} for i in range(U.S0A.N_ITEMS)}, None)
    case.mp.setattr(U.S0A, 'regression_items', items)
    case.mp.setattr(U.S0A, 'compare_counts', lambda *a: [])
    assert U.regression_gate(state, {})['passed'] is True


def test_success_records_order_and_hashes(case):
    run_stubs(case)
    def inputs(read, evidence):
        evidence[str(case.part)] = U.R.sha256_file(case.part)
        evidence.update(own_files(case))
        return NS()
    case.mp.setattr(U, 'inputs_gate', inputs)
    assert U.run('H8', case.root) == 0
    record = json.loads((case.root / U.preread_name('H8')).read_text())
    assert record['passed'] is True
    assert tuple(record['checks']) == U.CHECK_ORDER
    assert all(c['status'] == 'passed' for c in record['checks'].values())
    assert record['input_file_sha256'][str(case.part)] == U.R.sha256_file(case.part)
    assert U.preread_guard(case.root, 'H8')['name'] == U.preread_name('H8')


def test_phase1_fix_name_keeps_failed_agreement(case):
    state, agr, _, _ = phase1(case)
    failed = put(case.root / 'rd9_clip_b32_phase1_agreement.json', {'agreement': False})
    old = failed.read_bytes()
    put(case.root / 'rd9_clip_b32_phase1_agreement_fix1.json', agr)
    assert U.phase1_gate(state, {})['passed'] is True
    assert failed.read_bytes() == old


@pytest.mark.parametrize('error', (RuntimeError, TypeError, AttributeError))
def test_owner_errors_leave_failed_record(case, error):
    run_stubs(case)
    def fit(*args):
        raise error('owner failure')
    case.mp.setattr(U, 'heads_gate', fit)
    assert U.run('H8', case.root) == U.RH.EXIT_REFUSE
    rec = json.loads((case.root / U.preread_name('H8')).read_text())
    assert rec['checks']['heads']['error_type'] == error.__name__
    assert rec['stopped_at'] == 'heads'
    assert rec['checks']['smoke']['status'] == 'not_run'


def test_scratch_sigma_is_recorded(case):
    source = put(case.root / 'source/sensitivity_seed42_qwen3vl_emb_2b.json',
                 {c: {'sigma_a2': 1., 'sigma_eps2': 2.} for c in U.RH.ST.CHECKS})
    state = NS(read='H6', backbone='qwen3vl_emb_2b', env=None,
               sigma={'file': str(source), 'sha256': U.R.sha256_file(source)})
    case.mp.setattr(U.RH, 'VALUE_SETS_PATH', put(case.root / 'values.json', {}))
    case.mp.setattr(U.RH, 'setup_a28', lambda *a, **k: None)
    case.mp.setattr(U.RH, 'run_smoke', lambda *a, **k: 4)
    evidence = {}
    out = case.root / 'outputs'
    with pytest.raises(U.RH.Refused, match='runner smoke failed'):
        U.smoke_gate(state, evidence, out, 'smoke')
    target = out / source.name
    assert target.read_bytes() == source.read_bytes()
    assert evidence[str(target.resolve())] == U.R.sha256_file(target)


def test_finite_scratch_sigma_change_blocks_launch(case):
    sigma = {c: {'sigma_a2': 1., 'sigma_eps2': 2.} for c in U.RH.ST.CHECKS}
    path = put(case.root / 'sensitivity_seed42_qwen3vl_emb_2b.json', sigma)
    rec = launch_record(case)
    rec.update(read='H9', backbone='qwen3vl_emb_2b',
               input_file_sha256={str(path): U.R.sha256_file(path), **own_files(case)})
    put(case.root / U.preread_name('H9'), rec)
    U.preread_guard(case.root, 'H9')
    sigma['P1']['sigma_a2'] = 9.
    put(path, sigma)
    # The old finite/non-negative check accepts these changed bytes.
    assert U.RH.sigma_guard('qwen3vl_emb_2b', case.root)['sigma']['P1']['sigma_a2'] == 9.
    with pytest.raises(U.RH.Refused, match='inputs changed'):
        U.preread_guard(case.root, 'H9')


# ---- the held launch gate through run_held (moved from final_review/fr_c_gate_test.py, C MEDIUM-1) ----
# Each scenario calls the real RH.run_held with the real preread_guard and smoke_guard; only the ledger step
# (refuse_or_go), h5_hashes, a28_reference and run_read are stubbed. A launch = run_read called.

def arm(c, read='H8'):
    launched = []
    names = U.RH.read_names(read, 'held')
    c.mp.setattr(U.RH, 'refuse_or_go', lambda *a: NS(names=names, attempts=[], want={}, ledger={}))
    c.mp.setattr(U.RH, 'h5_hashes', lambda *a: {})
    c.mp.setattr(U.RH, 'a28_reference', lambda *a: None)
    c.mp.setattr(U.RH, 'run_read', lambda *a: launched.append(read) or 0)
    return launched


def good_smoke(c, sha=None, kind='smoke'):
    put(c.root / U.AR.SMOKE_RECORDS[kind], {'passed': True, 'module_sha256': sha or U.R.r9_module_shas()})


def go(c, read='H8', **kw):
    launched = arm(c, read)
    code = U.RH.run_held(U.RH.READ_BACKBONES[read], read, results=c.root, **kw)
    return code, launched


def test_gate_valid_record_launches(case):
    put(case.root / U.preread_name('H8'), launch_record(case)); good_smoke(case)
    code, launched = go(case)
    assert code == 0 and launched == ['H8']


def test_gate_missing_record(case):
    good_smoke(case)
    assert go(case) == (U.RH.EXIT_REFUSE, [])


def test_gate_failed_record(case):
    rec = launch_record(case); rec['passed'] = False
    put(case.root / U.preread_name('H8'), rec); good_smoke(case)
    assert go(case) == (U.RH.EXIT_REFUSE, [])


def test_gate_check_not_run_but_passed_true(case):
    rec = launch_record(case); rec['checks']['smoke'] = {'status': 'not_run'}
    put(case.root / U.preread_name('H8'), rec); good_smoke(case)
    assert go(case) == (U.RH.EXIT_REFUSE, [])


def test_gate_stale_module_bytes(case):
    rec = launch_record(case); rec['module_sha256'] = {'recipe.py': 'older'}
    put(case.root / U.preread_name('H8'), rec); good_smoke(case)
    assert go(case) == (U.RH.EXIT_REFUSE, [])


def test_gate_other_read_same_backbone(case):
    # H6's passing record placed under H9's name (both Qwen)
    rec = launch_record(case); rec.update(read='H6', backbone='qwen3vl_emb_2b')
    put(case.root / U.preread_name('H9'), rec); good_smoke(case)
    assert go(case, 'H9') == (U.RH.EXIT_REFUSE, [])


def test_gate_other_backbone(case):
    rec = launch_record(case); rec.update(read='H9', backbone='clip_b32')
    put(case.root / U.preread_name('H9'), rec); good_smoke(case)
    assert go(case, 'H9') == (U.RH.EXIT_REFUSE, [])


def test_gate_smoke_from_other_bytes(case):
    put(case.root / U.preread_name('H8'), launch_record(case)); good_smoke(case, {'recipe.py': 'older'})
    assert go(case) == (U.RH.EXIT_REFUSE, [])


def test_gate_smoke_missing(case):
    put(case.root / U.preread_name('H8'), launch_record(case))
    assert go(case) == (U.RH.EXIT_REFUSE, [])


def test_gate_empty_inputs(case):
    rec = launch_record(case); rec['input_file_sha256'] = {}
    put(case.root / U.preread_name('H8'), rec); good_smoke(case)
    assert go(case) == (U.RH.EXIT_REFUSE, [])


def test_gate_reserve_needs_reserve_preread(case):
    put(case.root / U.preread_name('H8'), launch_record(case)); good_smoke(case, kind='reserve')
    assert go(case, reserve=True) == (U.RH.EXIT_REFUSE, [])


def test_gate_fix_needs_held_fix_preread(case):
    # The base and the `_fix1` pre-reads pass; neither may serve a held --fix 1.
    put(case.root / U.preread_name('H8'), launch_record(case))
    put(case.root / U.preread_name('H8', 'fix1'), launch_record(case))
    good_smoke(case, kind='heldfix1')
    assert go(case, fix=1) == (U.RH.EXIT_REFUSE, [])


def test_gate_malformed_record_does_not_launch(case):
    (case.root / U.preread_name('H8')).write_text('[1, 2]'); good_smoke(case)
    try:
        code, launched = go(case)
    except Exception:
        launched = []
    assert launched == []


# ---- C LOW-2: a held --fix 1 has its own pre-read and smoke names ----

def test_held_fix_launches_on_its_own_preread_and_smoke(case):
    put(case.root / U.preread_name('H8', 'heldfix1'), launch_record(case)); good_smoke(case, kind='heldfix1')
    code, launched = go(case, fix=1)
    assert code == 0 and launched == ['H8']


def test_held_fix_preread_is_free_when_fix1_authorised_the_first_launch(case):
    # Base pre-read failed; `_fix1` passed on older bytes and authorised the first launch, with its own smoke record.
    put(case.root / U.preread_name('H8'), {'passed': False})
    put(case.root / U.preread_name('H8', 'fix1'), launch_record(case))
    put(case.root / U.AR.SMOKE_RECORDS['fix1'], {'passed': True, 'module_sha256': {'recipe.py': 'older'}})
    # The held fix runs on newer bytes: its smoke record and its pre-read are new files beside the old ones.
    smoke = {'module_sha256': {'recipe.py': 'a'}, 'runner_sha256': 'a', 'passed': True}
    assert U.publish_smoke(smoke, case.root, 'heldfix1')['name'] == 'smoke_record_heldfix1.json'
    with pytest.raises(U.RH.Refused, match='canonical smoke stale'):
        U.publish_smoke(smoke, case.root, 'fix1')       # the old name stays taken: the very deadlock
    run_stubs(case)
    assert U.run('H8', case.root, 'heldfix1') == 0
    assert (case.root / U.preread_name('H8', 'heldfix1')).is_file()
    assert json.loads((case.root / U.AR.SMOKE_RECORDS['fix1']).read_text())['module_sha256'] == {'recipe.py': 'older'}


def test_held_fix_smoke_record_takes_precedence_in_both_latest_functions(case):
    for kind in ('smoke', 'fix1', 'heldfix1'):
        put(case.root / U.AR.SMOKE_RECORDS[kind], {'passed': True})
    assert U.AR.latest_smoke_record(case.root, None).name == 'smoke_record_heldfix1.json'
    assert U.RH.latest_smoke_record(case.root, 'held').name == 'smoke_record_heldfix1.json'
    assert U.RH.latest_smoke_record(case.root, 'fix1').name == 'smoke_record_heldfix1.json'
    assert U.RH.read_names('H8', 'fix1').smoke_record == 'smoke_record_heldfix1.json'
    assert U.RH.read_names('H8', 'held').smoke_record == 'smoke_record.json'


def test_cli_held_fix_flag_runs_the_heldfix1_kind(case):
    seen = []
    case.mp.setattr(U, 'run', lambda read, results, kind: seen.append((read, kind)) or 0)
    assert U.main(['--read', 'H8', '--held-fix', '1', '--results', str(case.root)]) == 0
    assert U.main(['--read', 'H8', '--fix', '1', '--results', str(case.root)]) == 0
    assert seen == [('H8', 'heldfix1'), ('H8', 'fix1')]


# ---- C LOW-1: a refused gate, run_r9_held.py as a script, exits 4 ----

def test_script_run_of_held_runner_exits_4_on_a_refused_gate(case):
    # Execute the runner as `__main__` in its own namespace (the way `python run_r9_held.py` does), with the real
    # `if __name__ == '__main__':` block. The module run_r9_preread imports by name must be this same namespace.
    source = (HERE / 'run_r9_held.py').read_text()
    head, marker, tail = source.partition("if __name__ == '__main__':")
    assert marker
    mod = type(sys)('__main__')
    mod.__file__ = str(HERE / 'run_r9_held.py')
    case.mp.setitem(sys.modules, '__main__', mod)
    case.mp.delitem(sys.modules, 'run_r9_held')
    case.mp.delitem(sys.modules, 'run_r9_preread')
    case.mp.setattr(sys, 'argv', ['run_r9_held.py', '--backbone', 'clip_b32', '--read', 'H8'])
    exec(compile(head, mod.__file__, 'exec'), mod.__dict__)
    mod.refuse_or_go = lambda *a: NS(names=mod.read_names('H8', 'held'), attempts=[], want={}, ledger={})
    with pytest.raises(SystemExit) as stop:
        exec(compile('\n' * head.count('\n') + marker + tail, mod.__file__, 'exec'), mod.__dict__)
    assert stop.value.code == mod.EXIT_REFUSE == 4


# ---- C LOW-3: a scratch smoke run writes nothing into committed paths ----

def test_smoke_agreement_writes_only_into_the_smoke_folder(case):
    out = case.root / 'smoke/tag'
    names = U.RH.read_names('H8', 'smoke')
    put(out / names.pass_, {'checks': {c: {'n': 3} for c in U.RH.ST.CHECKS}})
    rederive_out = case.root / 'rederive_out'
    case.mp.setattr(U.AR, 'AGREEMENT_DIR', rederive_out)
    got = U.smoke_agreement(out, 'H8', 'tag')
    assert Path(got['copy']) == out / 'rd9_H8_agreement.json' and (out / 'rd9_H8_agreement.json').is_file()
    assert not rederive_out.exists()


def test_scratch_smoke_record_has_no_committed_copy(case):
    record = {'module_sha256': {'recipe.py': 'a'}, 'runner_sha256': 'a', 'passed': True}
    scratch = case.root / 'scratch'
    U.publish_smoke(record, scratch, 'smoke')
    assert (scratch / U.AR.SMOKE_RECORDS['smoke']).is_file()
    assert not (case.root / 'folder').exists()
    U.publish_smoke(record, case.root, 'smoke')       # the real results folder: the committed copy too
    assert (case.root / 'folder' / U.AR.SMOKE_RECORDS['smoke']).is_file()
