"""Ticket 13: provenance refusals and the final pins gate; CPU only."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_r9_stage2 as D


def test_refitted_array_equal_to_clip_fails():
    a = np.arange(8, dtype=np.float32)
    with pytest.raises(AssertionError, match="made.*equals CLIP"):
        D.assert_array_provenance({"post": a}, {"post": a.copy()}, ["post"], [])


def test_copied_array_changed_fails():
    with pytest.raises(AssertionError, match="copied.*differs"):
        D.assert_array_provenance({"rows": np.array([1, 3])}, {"rows": np.array([1, 2])}, [], ["rows"])


def test_a_key_in_both_made_and_copied_fails():
    # equal arrays: without the overlap guard the key would count as copied and pass (rr fixwave 2, LOW-5)
    a = np.arange(4, dtype=np.float32)
    with pytest.raises(AssertionError, match="made/copied keys overlap"):
        D.assert_array_provenance({"k": a, "m": a + 1}, {"k": a.copy(), "m": a.copy()}, ["k", "m"], ["k"])


def test_frozen_array_changed_fails(tmp_path, monkeypatch):
    a, b = tmp_path / "new.npz", tmp_path / "clip.npz"
    np.savez(a, **{k: np.array([0, 2]) for k in D.FROZEN_KEYS['prepare']})
    np.savez(b, **{k: np.array([0, 1]) for k in D.FROZEN_KEYS['prepare']})
    pin = tmp_path / 'pins_clip.json'
    pin.write_text(json.dumps({'parts': {'prepare': D.path_record(a)}}))
    monkeypatch.setattr(D.R, 'pins_path', lambda backbone: pin)
    with pytest.raises(AssertionError, match="copied.*differs"):
        D.assert_frozen("prepare", a, b)


def test_changed_script_sha_fails(tmp_path):
    script = tmp_path / "r9_recipe_groups.py"
    script.write_text("old\n")
    sha = D.R.sha256_file(script)
    stage0b = {"passed": True, "item4": {"passed": True, "scripts": {script.name: {"sha256": sha}}}}
    script.write_text("changed\n")
    with pytest.raises(AssertionError, match="script SHA-256"):
        D.assert_script({"script": str(script), "script_sha256": sha}, stage0b, tmp_path)


@pytest.mark.parametrize("failure", ["missing_record", "failed_stage2", "failed_picks"])
def test_pins_refuses_unrecorded_part_or_failed_stage(tmp_path, failure):
    part = tmp_path / "picks_seed42.json"
    part.write_text(json.dumps({"passed": failure != "failed_picks"}))
    if failure != "missing_record":
        Path(str(part) + ".record.json").write_text('{}')
    stage = {"passed": failure != "failed_stage2", "backbone": "qwen3vl_emb_2b"}
    with pytest.raises(AssertionError, match={"missing_record": "record", "failed_stage2": "stage 2", "failed_picks": "picks"}[failure]):
        D.require_pinnable("qwen3vl_emb_2b", {"picks_seed42": part}, stage, recipe_roles={"picks_seed42"})


@pytest.mark.parametrize('failure', ['missing_record', 'failed_stage2', 'failed_picks'])
def test_pins_step_refuses_before_writing(tmp_path, monkeypatch, failure):
    backbone = 'qwen3vl_emb_2b'
    paths = {r: tmp_path / (r + '.json') for r in D.BB.CLIP_PARTS}
    for role, path in paths.items():
        path.write_text(json.dumps({'passed': failure != 'failed_picks'}))
        if role in D.BB.RECIPE_ROLES and not (role == 'n6_posteriors' and failure == 'missing_record'):
            Path(str(path) + '.record.json').write_text('{}')
    D.write_new(tmp_path / f'stage2_{backbone}.json',
                {'passed': failure != 'failed_stage2', 'backbone': backbone})
    target = tmp_path / 'final_pins.json'
    monkeypatch.setattr(D.R, 'pins_path', lambda b: target)
    monkeypatch.setattr(D.BB, 'parts', lambda b: paths)
    with pytest.raises(AssertionError, match={'missing_record': 'record', 'failed_stage2': 'stage 2',
                                           'failed_picks': 'picks'}[failure]):
        D.run_pins(backbone, paths, tmp_path, tmp_path / 'stage0b.json')
    assert not target.exists()


def test_clip_sensitivity_uses_round6_function_exactly():
    RH, RS, ST = D.round6_statistics()
    with np.load(D.R.R6D / 'results/seed42_per_episode.npz') as z:
        got = RS.sensitivity_record(z)
    stored = D.read_json(D.R.R6D / 'results/sensitivity_seed42.json')
    assert got == {k: stored[k] for k in got}


def test_made_differs_copied_equal_passes():
    D.assert_array_provenance({"coef": np.array([2.]), "rows": np.array([1])},
                              {"coef": np.array([1.]), "rows": np.array([1])}, ["coef"], ["rows"])


# Small file-boundary world: real provenance, receipts, record and pins calls;
# only feature loading/scoring and the subprocess are substituted.
import contextlib
import copy
import hashlib
import pickle
from types import SimpleNamespace
import r9_pins as PINS
import r9_recipe_groups as GROUPS


@pytest.fixture
def world(tmp_path, monkeypatch):
    b = 'qwen3vl_emb_2b'
    out = tmp_path / 'results'
    recipe = out / 'recipe' / b
    clip = tmp_path / 'clip'
    recipe.mkdir(parents=True)
    clip.mkdir()
    roles = ('prepare', 'partitions', 'rc_tau', 'baselines_seed42', 'picks_seed42', 'cells_seed42')
    paths = {r: (clip if r == 'prepare' else recipe) / (r + ('.npz' if r in ('prepare', 'partitions') else '.json')) for r in roles}
    pinned = {r: clip / paths[r].name for r in roles}
    np.savez(pinned['prepare'], **{k: np.array([1, 2]) for k in D.FROZEN_KEYS['prepare']})
    np.savez(pinned['partitions'], fit=np.array([1.], np.float32), rows=np.array([1, 2]), unused=np.array([0.]))
    np.savez(paths['partitions'], fit=np.array([2.], np.float32), rows=np.array([1, 2]), unused=np.array([np.nan]))
    cells = D.read_json(D.BB.CLIP_PARTS['cells_seed42'])
    picks = {'passed': True, 'mean_r1': {n: float(i + 1) for i, n in enumerate(D.P.NESTED)},
             'targets': None, 'convention': 'frozen', **{n: {'0': [1, 1.], '1': [1, 1.]} for n in D.P.NESTED}}
    payloads = {'rc_tau': {'taus': [1., 2., 3., 4.]}, 'picks_seed42': picks,
                'baselines_seed42': {'scorers': {n: {'lambda_picks': {'0': 0., '1': 1.}} for n in D.P.LAMBDA_SCORERS},
                                    'episodes_sha256': {'pair': 'frozen'}, 'fit_rows': 60000},
                'cells_seed42': cells}
    for role, body in payloads.items():
        paths[role].write_text(json.dumps(body))
        reference = copy.deepcopy(body)
        if role == 'rc_tau':
            reference['taus'] = [0., 1., 2., 3.]
        pinned[role].write_text(json.dumps(reference))
    feature = tmp_path / 'features.npy'
    feature.write_bytes(b'feature store')
    stores = {str(feature): D.R.sha256_file(feature)}
    recipe_roles = frozenset(roles) - {'prepare'}
    step_roles = {'groups': ('partitions',), 'a3': (), 'reader': ('rc_tau',),
                  'baselines': ('baselines_seed42', 'picks_seed42'), 'cells': ('cells_seed42',)}
    monkeypatch.setattr(D.BB, 'CLIP_PARTS', pinned)
    monkeypatch.setattr(D.BB, 'RECIPE_ROLES', recipe_roles)
    monkeypatch.setattr(D.BB, 'FROZEN_ROLES', frozenset({'prepare'}))
    monkeypatch.setattr(D.BB, 'parts', lambda name: pinned if name == 'clip_b32' else paths)
    monkeypatch.setattr(D, 'STEP_ROLES', step_roles)
    monkeypatch.setattr(D.R, 'RESULTS', out)
    monkeypatch.setattr(D.R, 'input_paths', lambda name: {'recipe-baselines': paths['baselines_seed42']})
    monkeypatch.setattr(D.R, 'pins_path', lambda name: clip / 'pins_clip.json' if name == 'clip_b32' else out / f'pins_{name}.json')
    monkeypatch.setattr(PINS, 'rel_path', lambda p: str(p))
    monkeypatch.setattr(GROUPS, 'store_shas', lambda name: stores)
    gate = {'passed': True, 'item4': {'passed': True, 'scripts': {
        name: {'sha256': D.R.sha256_file(HERE / name)} for name in D.STEP_SCRIPTS.values()}},
        'module_sha256': D.recipe_module_shas()}
    gate_path = out / 'stage0b.json'
    D.write_new(gate_path, gate)
    D.write_new(clip / 'pins_clip.json', {'parts': {r: D.path_record(p) for r, p in pinned.items()}})
    records = {}
    for role in recipe_roles:
        step = next(s for s, rr in step_roles.items() if role in rr)
        rec = {'backbone': b, 'script': str(HERE / D.STEP_SCRIPTS[step]),
               'script_sha256': gate['item4']['scripts'][D.STEP_SCRIPTS[step]]['sha256'],
               'input_sha256': stores, 'keys_made': ['fit'] if role == 'partitions' else list(payloads[role]),
               'keys_copied': ['rows'] if role == 'partitions' else [],
               'keys_unrefit': ['unused'] if role == 'partitions' else []}
        if role == 'baselines_seed42':
            rec['keys_copied'] = ['episodes_sha256', 'fit_rows']
            rec['keys_made'] = ['scorers']
        records[role] = rec
        D.write_new(str(paths[role]) + '.record.json', rec)
    for step, rr in step_roles.items():
        D.write_new(recipe / f'stage2_{step}_files.json', {
            **{r: {'file': D.path_record(paths[r]), 'record': D.path_record(str(paths[r]) + '.record.json')} for r in rr},
            'extra_files': {}, 'module_sha256': gate['module_sha256']})
    seed = {'mean_r1': picks['mean_r1'], 'arrays': {}, 'checks': {}}
    sigma = {c: {'sigma_a2': float(i + 10), 'sigma_eps2': float(i + 20)} for i, c in enumerate(('P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7'))}
    monkeypatch.setattr(D, 'input_manifest', lambda *args: contextlib.nullcontext())
    monkeypatch.setattr(D, 'recipe_input_manifest', lambda *args: {})
    monkeypatch.setattr(D, '_score', lambda *args: (seed, sigma, {}))
    w = SimpleNamespace(b=b, out=out, recipe=recipe, paths=paths, pinned=pinned, gate=gate, gate_path=gate_path,
                        stores=stores, records=records, seed=seed, sigma=sigma)
    return w


def rewrite(path, data):
    Path(path).write_text(json.dumps(data))


def refresh_receipts(w):
    for step, roles in D.STEP_ROLES.items():
        p = w.recipe / f'stage2_{step}_files.json'
        rec = D.read_json(p)
        for role in roles:
            rec[role] = {'file': D.path_record(w.paths[role]), 'record': D.path_record(str(w.paths[role]) + '.record.json')}
        rewrite(p, rec)


def record_world(w):
    return D.run_record(w.b, w.paths, w.out, w.gate_path)


def test_synthetic_provenance_record_and_pins(world):
    w = world
    prov = D.provenance(w.b, w.paths, w.gate)
    assert prov['partitions']['keys_tested_differ'] == ['fit']
    assert prov['partitions']['keys_tested_equal'] == ['rows']
    assert prov['baselines_seed42']['keys_tested_differ'] == []
    assert prov['baselines_seed42']['keys_tested_equal'] == ['episodes_sha256', 'fit_rows']
    stage = record_world(w)
    assert stage['status'] == 'ready'
    assert D.read_json(w.out / f'sensitivity_seed42_{w.b}.json') == w.sigma
    assert stage['module_sha256'] == D.recipe_module_shas()
    pins = D.run_pins(w.b, w.paths, w.out, w.gate_path)
    assert pins['provenance'] == prov
    assert pins['module_sha256'] == stage['module_sha256']
    assert D.read_json(D.R.pins_path(w.b)) == pins
    with pytest.raises(AssertionError, match='overwrite'):
        D.run_pins(w.b, w.paths, w.out, w.gate_path)
    with pytest.raises(AssertionError, match='overwrite'):
        record_world(w)


@pytest.mark.parametrize('bad', ['frozen', 'script', 'module', 'made_equal', 'copied_changed', 'unrefit', 'lineage', 'exempt_copied'])
def test_provenance_call_sites_refuse(world, bad):
    w = world
    if bad == 'frozen':
        np.savez(w.paths['prepare'], **{k: np.array([1, 3]) for k in D.FROZEN_KEYS['prepare']})
    elif bad == 'module':
        w.gate['module_sha256'][next(iter(w.gate['module_sha256']))] = '0' * 64
    elif bad == 'script':
        rec = D.read_json(str(w.paths['partitions']) + '.record.json')
        rec['script_sha256'] = '0' * 64
        rewrite(str(w.paths['partitions']) + '.record.json', rec)
    elif bad in ('made_equal', 'copied_changed', 'unrefit'):
        np.savez(w.paths['partitions'], fit=np.array([1. if bad == 'made_equal' else 2.], np.float32),
                 rows=np.array([1, 3 if bad == 'copied_changed' else 2]), unused=np.array([0. if bad == 'unrefit' else np.nan]))
    elif bad == 'lineage':
        rec = D.read_json(str(w.paths['partitions']) + '.record.json')
        rec['input_sha256'] = {str(w.paths['prepare']): D.R.sha256_file(w.paths['prepare'])}
        rewrite(str(w.paths['partitions']) + '.record.json', rec)
    else:
        rec = D.read_json(w.paths['baselines_seed42'])
        rec['episodes_sha256'] = {'pair': 'changed'}
        rewrite(w.paths['baselines_seed42'], rec)
    with pytest.raises(AssertionError):
        D.provenance(w.b, w.paths, w.gate)


@pytest.mark.parametrize('bad', ['script', 'dry_run', 'file', 'runner', 'rule', 'module', 'receipt'])
def test_pins_evidence_call_sites_refuse(world, bad):
    w = world
    record_world(w)
    stage_path = w.out / f'stage2_{w.b}.json'
    stage = D.read_json(stage_path)
    if bad == 'script':
        rec_path = Path(str(w.paths['partitions']) + '.record.json')
        rec = D.read_json(rec_path)
        rec['script_sha256'] = '0' * 64
        rewrite(rec_path, rec)
        refresh_receipts(w)
        stage['provenance']['partitions']['record'] = D.path_record(rec_path)
        stage['recipe_receipts'] = {s: D.path_record(w.recipe / f'stage2_{s}_files.json') for s in D.STEP_SCRIPTS}
    elif bad in ('runner', 'rule'):
        stage[bad + '_sha256'] = '0' * 64
    elif bad == 'dry_run':
        stage['dry_run'] = True
    elif bad == 'file':
        # Refresh the receipt so only the stage's immutable snapshot catches it.
        w.paths['rc_tau'].write_text('{"taus": [2.0, 3.0, 4.0, 5.0]}')
        refresh_receipts(w)
        stage['recipe_receipts'] = {s: D.path_record(w.recipe / f'stage2_{s}_files.json') for s in D.STEP_SCRIPTS}
    elif bad == 'module':
        w.gate['module_sha256'][next(iter(w.gate['module_sha256']))] = '0' * 64
        rewrite(w.gate_path, w.gate)
        stage['stage0b'] = D.path_record(w.gate_path)
    else:
        w.paths['partitions'].write_bytes(b'changed')
    rewrite(stage_path, stage)
    with pytest.raises(AssertionError):
        D.run_pins(w.b, w.paths, w.out, w.gate_path)
    assert not D.R.pins_path(w.b).exists()


def test_switched_off_stops_without_numbers_or_pins(world, monkeypatch):
    w = world
    cells = D.read_json(w.paths['cells_seed42'])
    cells.update(aff=[0, 0], switched_off=True)
    rewrite(w.paths['cells_seed42'], cells)
    refresh_receipts(w)
    monkeypatch.setattr(D, '_score', lambda *args: pytest.fail('switched-off backbone scored'))
    stage = record_world(w)
    assert stage['status'] == 'not run: switched off' and stage['gate_share'] == cells['gate_share']
    assert 'seed42_record' not in stage and 'sensitivity_seed42' not in stage
    assert not (w.out / f'stage2_seed42_{w.b}.json').exists()
    assert not (w.out / f'sensitivity_seed42_{w.b}.json').exists()
    with pytest.raises(AssertionError, match='switched off'):
        D.run_pins(w.b, w.paths, w.out, w.gate_path)


@pytest.mark.parametrize('bad', [float('nan'), -1., float('inf')])
def test_run_record_refuses_bad_sigma_before_writes(world, bad):
    world.sigma['P3']['sigma_a2'] = bad
    with pytest.raises(AssertionError, match='invalid sigma_a2'):
        record_world(world)
    assert not (world.out / f'stage2_{world.b}.json').exists()


def test_run_record_checks_clip_exact_and_preserves_bytes(world, monkeypatch):
    w = world
    calls = []
    monkeypatch.setattr(D, 'provenance', lambda *args, **kwargs: {})
    monkeypatch.setattr(D.S0B, 'routed', lambda *args, **kwargs: contextlib.nullcontext())
    monkeypatch.setattr(D, 'clip_exact', lambda *args: calls.append(args))
    D.run_record('clip_b32', w.paths, w.out, w.gate_path, dry_run=True)
    assert len(calls) == 1
    assert (w.out / 'sensitivity_seed42_clip_b32.json').read_bytes() == (D.R.R6D / 'results/sensitivity_seed42.json').read_bytes()


@pytest.mark.parametrize('step', list(D.STEP_SCRIPTS))
def test_step_command_lock_and_arguments(step):
    cmd = D.step_command('qwen3vl_emb_2b', step)
    if step in ('groups', 'a3'):
        assert cmd[:7] == ['flock', '-n', '-o', '-E', '75', '/tmp/gpu0.lock', sys.executable]
    else:
        assert cmd[0] == sys.executable and 'flock' not in cmd
    assert '--backbone' in cmd and '--out' in cmd
    if step == 'a3':
        assert cmd[cmd.index('--run') + 1] == 'main' and '--inputs' in cmd
    elif step == 'reader':
        assert cmd[cmd.index('--stage') + 1] == 'all' and '--upstream' in cmd
    elif step == 'baselines':
        assert '--partitions-dir' in cmd and '--a3-dir' in cmd
    elif step == 'cells':
        assert cmd[1] == '-c' and 'runpy.run_path' in cmd[2]


@pytest.mark.parametrize('step', ['groups', 'a3', 'reader', 'baselines', 'cells'])
def test_run_step_env_receipt_and_exit(world, monkeypatch, step):
    w = world
    (w.recipe / f'stage2_{step}_files.json').unlink()
    captured = []
    monkeypatch.setattr(D.subprocess, 'run', lambda cmd, env: captured.append((cmd, env)) or SimpleNamespace(returncode=75))
    assert D.run_step(w.b, step, stage0b_path=w.gate_path) == 75
    assert not (w.recipe / f'stage2_{step}_files.json').exists()
    assert captured[0][1]['CUDA_VISIBLE_DEVICES'] == ('0' if step in ('groups', 'a3') else '')
    assert captured[0][1]['OMP_NUM_THREADS'] == captured[0][1]['MKL_NUM_THREADS'] == '8'
    if step == 'cells':
        # Failed run left the bootstrap; equal bootstrap must permit retry.
        assert (w.recipe / 'stage2_inputs.json').is_file()
        assert D.run_step(w.b, step, stage0b_path=w.gate_path) == 75
        rewrite(w.recipe / 'stage2_inputs.json', {'changed': True})
        with pytest.raises(AssertionError, match='manifest'):
            D.run_step(w.b, step, stage0b_path=w.gate_path)
    if step == 'groups':
        D.write_new(w.recipe / 'build_record.json', {})
        monkeypatch.setattr(D.subprocess, 'run', lambda cmd, env: SimpleNamespace(returncode=0))
        assert D.run_step(w.b, step, stage0b_path=w.gate_path) == 0
        receipt = D.read_json(w.recipe / 'stage2_groups_files.json')
        assert receipt['module_sha256'] == D.recipe_module_shas()
        assert receipt['partitions']['file'] == D.path_record(w.paths['partitions'])


def test_receipts_refuse_changed_file(world):
    world.paths['partitions'].write_bytes(b'changed')
    with pytest.raises(AssertionError, match='output changed'):
        D.check_receipts(world.b, module_sha256=world.gate['module_sha256'])


def test_exclusive_creation(tmp_path):
    p = tmp_path / 'record.json'
    D.write_new(p, {'one': 1})
    with pytest.raises(FileExistsError):
        D.write_new(p, {'two': 2})
    assert D.read_json(p) == {'one': 1}


def test_main_all_returns_first_failure(world, monkeypatch):
    w = world
    seen = []
    monkeypatch.setattr(D, 'run_step', lambda b, step, **kw: seen.append(step) or (75 if step == 'a3' else 0))
    monkeypatch.setattr(D, 'run_record', lambda *a, **kw: pytest.fail('continued after failure'))
    assert D.main(['--backbone', w.b, '--step', 'all', '--stage0b', str(w.gate_path)]) == 75
    assert seen == ['groups', 'a3']


def test_main_all_stops_after_switched_off_record(world, monkeypatch):
    w = world
    seen = []
    monkeypatch.setattr(D, 'run_step', lambda b, step, **kw: seen.append(step) or 0)
    monkeypatch.setattr(D, 'run_record', lambda *a, **kw: {'switched_off': True})
    monkeypatch.setattr(D, 'run_pins', lambda *a, **kw: pytest.fail('switched-off numeric pins'))
    assert D.main(['--backbone', w.b, '--step', 'all', '--stage0b', str(w.gate_path)]) == 0
    assert seen == list(D.STEP_SCRIPTS)


@pytest.mark.parametrize('bad', ['dry_backbone', 'dry_step', 'live_clip', 'failed_gate', 'module', 'script'])
def test_main_refuses_before_any_step(world, monkeypatch, bad):
    w = world
    monkeypatch.setattr(D, 'run_step', lambda *a, **kw: pytest.fail('ran before gate'))
    monkeypatch.setattr(D, 'run_record', lambda *a, **kw: pytest.fail('scored before gate'))
    args = ['--backbone', w.b, '--step', 'groups', '--stage0b', str(w.gate_path)]
    if bad in ('dry_backbone', 'dry_step'):
        args = ['--backbone', w.b if bad == 'dry_backbone' else 'clip_b32', '--step', 'record' if bad == 'dry_backbone' else 'groups',
                '--dry-run', '--replay', str(w.recipe), '--out', str(w.out)]
        monkeypatch.setattr(D, 'HERE', w.out.parent)
        monkeypatch.setattr(D, 'recipe_module_shas', lambda *args, **kwargs: w.gate['module_sha256'])
    elif bad == 'live_clip':
        args[1] = 'clip_b32'
    elif bad == 'failed_gate':
        w.gate['passed'] = False
    elif bad == 'module':
        w.gate['module_sha256'][next(iter(w.gate['module_sha256']))] = '0' * 64
    else:
        w.gate['item4']['scripts']['r9_recipe_groups.py']['sha256'] = '0' * 64
    rewrite(w.gate_path, w.gate)
    with pytest.raises(AssertionError):
        D.main(args)


def test_rc_tau_equal_is_not_exempt(tmp_path):
    a, b = tmp_path / 'tau.json', tmp_path / 'clip.json'
    for p in (a, b):
        rewrite(p, {'taus': [1., 2., 3., 4.]})
    with pytest.raises(AssertionError, match='made taus equals CLIP'):
        D.assert_part('rc_tau', a, b, {'keys_made': ['taus'], 'keys_copied': []})


def test_reader_coefficients_differ_iterations_are_metadata(tmp_path):
    a, b = tmp_path / 'reader.pkl', tmp_path / 'clip.pkl'
    for p, coef in ((a, 2.), (b, 1.)):
        with p.open('wb') as f:
            pickle.dump({'halves': [{'model': SimpleNamespace(coef_=np.array([coef]), classes_=np.array([0, 1, 2]), n_iter_=np.array([54]))}]}, f)
    rec = D.assert_part('rb_reader_A0', a, b, {'keys_made': ['halves'], 'keys_copied': []})
    assert rec['keys_tested_differ'] == ['halves/0/model/coef_']
    assert rec['keys_tested_equal'] == ['halves/0/model/classes_']
    assert rec['keys_metadata_not_compared'] == ['halves/0/model/n_iter_']


def test_reader_json_claims_only_feature_names(tmp_path):
    a, b = tmp_path / 'reader.json', tmp_path / 'clip.json'
    for p in (a, b):
        rewrite(p, {'feature_names': ['one'], 'runtime_s': 2})
    rec = D.assert_part('rb_reader_A0_json', a, b, {'keys_made': ['feature_names', 'runtime_s'], 'keys_copied': []})
    assert rec['keys_made'] == ['feature_names', 'runtime_s']
    assert rec['keys_tested_equal'] == ['feature_names'] and rec['keys_tested_differ'] == []


def test_import_closure_is_runtime_transitive_and_excludes_unrelated(tmp_path):
    (tmp_path / 'r9_recipe_groups.py').write_text('import r9_first\n')
    (tmp_path / 'r9_first.py').write_text('from r9_second import x\n')
    (tmp_path / 'r9_second.py').write_text('x = 1\n')
    (tmp_path / 'run_r9_held.py').write_text('x = 1\n')
    shas = D.recipe_module_shas(tmp_path, roots=['r9_recipe_groups.py'])
    assert {Path(p).name for p in shas} == {'r9_recipe_groups.py', 'r9_first.py', 'r9_second.py'}
    gate = {'passed': True, 'module_sha256': shas}
    D.assert_modules(gate, here=tmp_path, roots=['r9_recipe_groups.py'])
    (tmp_path / 'run_r9_held.py').write_text('x = 2\n')
    D.assert_modules(gate, here=tmp_path, roots=['r9_recipe_groups.py'])
    (tmp_path / 'r9_second.py').write_text('x = 2\n')
    with pytest.raises(AssertionError, match='r9_second.py differs from the bytes stage 0b passed on: rerun the stage 0b compare'):
        D.assert_modules(gate, here=tmp_path, roots=['r9_recipe_groups.py'])


def test_recipe_closure_has_the_certified_modules():
    names = {Path(p).name for p in D.recipe_module_shas()}
    assert names == set(D.STEP_SCRIPTS.values()) | {n + '.py' for n in (
        'r9_backbone', 'r9_bundle', 'r9_common', 'r9_context', 'r9_data', 'r9_episodes', 'r9_heads', 'r9_picks',
        'r9_pins', 'r9_score', 'r9_stats', 'run_r9_stage0a')}


def test_input_files_reject_clip_recipe_hash_and_unknown(world, monkeypatch):
    w = world
    rel = w.pinned['rc_tau'].name
    monkeypatch.setattr(D.R, 'TEST', w.pinned['rc_tau'].parent)
    monkeypatch.setattr(D.R, 'INPUT_SHA256', {rel: D.R.sha256_file(w.pinned['rc_tau'])})
    with pytest.raises(AssertionError, match='logical input changed'):
        D._input_files({'input_sha256': {rel: D.R.sha256_file(w.pinned['rc_tau'])}}, w.paths)
    with pytest.raises(AssertionError, match='CLIP recipe input path'):
        D._input_files({'input_sha256': {str(w.pinned['rc_tau']): D.R.sha256_file(w.pinned['rc_tau'])}}, w.paths)
    with pytest.raises(AssertionError, match='unknown input'):
        D._input_files({'input_sha256': {'unknown-input': '0' * 64}}, w.paths)


def test_input_files_replay_accepts_clip_logical_file_only(world, monkeypatch):
    w = world
    rel = w.pinned['baselines_seed42'].name
    clip_sha = D.R.sha256_file(w.pinned['baselines_seed42'])
    w.paths['baselines_seed42'].write_text(json.dumps(D.read_json(w.paths['baselines_seed42']), indent=2))
    own_sha = D.R.sha256_file(w.paths['baselines_seed42'])
    assert own_sha != clip_sha
    monkeypatch.setattr(D.R, 'TEST', w.pinned['baselines_seed42'].parent)
    monkeypatch.setattr(D.R, 'INPUT_SHA256', {rel: clip_sha})
    rec = {'input_sha256': {rel: clip_sha}}
    assert D._input_files(rec, w.paths, replay=True) == rec['input_sha256']
    with pytest.raises(AssertionError, match='logical input changed'):
        D._input_files(rec, w.paths)
    for replay in (False, True):
        assert D._input_files({'input_sha256': {rel: own_sha}}, w.paths, replay=replay) == {rel: own_sha}
        with pytest.raises(AssertionError, match='logical input changed'):
            D._input_files({'input_sha256': {rel: '0' * 64}}, w.paths, replay=replay)
    with pytest.raises(AssertionError, match='unknown input'):
        D._input_files({'input_sha256': {'unknown-input': clip_sha}}, w.paths, replay=True)


def replay_world(w, monkeypatch):
    # Byte-equal partitions name their frozen source as an input. Cells name
    # frozen baselines, while the replay baselines have different JSON bytes.
    np.savez(w.pinned['partitions'], fit=np.array([1.], np.float32),
             rows=np.array([1, 2]), unused=np.array([np.nan]))
    for role in D.BB.RECIPE_ROLES:
        w.paths[role].write_bytes(w.pinned[role].read_bytes())
    w.paths['baselines_seed42'].write_text(json.dumps(D.read_json(w.paths['baselines_seed42']), indent=2))
    rel = w.pinned['baselines_seed42'].name
    clip_sha = D.R.sha256_file(w.pinned['baselines_seed42'])
    monkeypatch.setattr(D.R, 'TEST', w.pinned['baselines_seed42'].parent)
    monkeypatch.setattr(D.R, 'INPUT_SHA256', {rel: clip_sha})
    for role, key, sha in (('partitions', 'partitions', D.R.sha256_file(w.paths['partitions'])),
                           ('cells_seed42', rel, clip_sha)):
        rec = copy.deepcopy(w.records[role])
        rec['input_sha256'] = {**w.stores, key: sha}
        rewrite(str(w.paths[role]) + '.record.json', rec)
    monkeypatch.setattr(D.S0B, 'routed', lambda *args, **kw: contextlib.nullcontext())
    monkeypatch.setattr(D, 'clip_exact', lambda *args: None)


def test_replay_self_hash_and_logical_input_reach_record_and_pins(world, monkeypatch):
    w = world
    replay_world(w, monkeypatch)
    stage = D.run_record(w.b, w.paths, w.out, w.gate_path, dry_run=True)
    skipped = {'partitions': D.R.sha256_file(w.paths['partitions'])}
    assert stage['provenance']['partitions']['skipped_self_input_sha256'] == skipped
    assert stage['provenance']['cells_seed42']['skipped_self_input_sha256'] == {}
    evidence = D.pins_evidence(w.b, w.paths, w.out, w.gate_path, dry_run=True)
    assert evidence['provenance'] == stage['provenance']


def test_self_hash_still_refuses_live_provenance(world, monkeypatch):
    w = world
    replay_world(w, monkeypatch)
    # Remove the unrelated replay-only baseline input to reach the live cycle.
    rewrite(str(w.paths['cells_seed42']) + '.record.json', w.records['cells_seed42'])
    with pytest.raises(AssertionError, match='partitions: cyclic input provenance'):
        D.provenance(w.b, w.paths, w.gate)


@pytest.mark.parametrize('entry', ['run_step', 'run_record'])
def test_receipts_checked_at_step_and_record_call_sites(world, monkeypatch, entry):
    w = world
    rewrite(w.paths['rc_tau'], {'taus': [2., 3., 4., 5.]})
    if entry == 'run_step':
        (w.recipe / 'stage2_baselines_files.json').unlink()
        monkeypatch.setattr(D.subprocess, 'run', lambda *args, **kw: pytest.fail('ran after an earlier output changed'))
    else:
        monkeypatch.setattr(D, '_score', lambda *args: pytest.fail('scored after an earlier output changed'))
    with pytest.raises(AssertionError, match='rc_tau: output changed since it was made'):
        if entry == 'run_step':
            D.run_step(w.b, 'baselines', stage0b_path=w.gate_path)
        else:
            record_world(w)
    assert not (w.out / f'stage2_{w.b}.json').exists()


def test_closure_module_missing_from_stage0b_refuses(world):
    module = next(m for m in world.gate['module_sha256'] if Path(m).name == 'r9_common.py')
    del world.gate['module_sha256'][module]
    with pytest.raises(AssertionError, match='r9_common.py differs from the bytes stage 0b passed on: rerun the stage 0b compare'):
        D.assert_modules(world.gate)


def test_frozen_pin_also_requires_round6_hash(world, monkeypatch):
    w = world
    monkeypatch.setattr(D.R, 'TEST', w.paths['prepare'].parent)
    monkeypatch.setattr(D.R, 'INPUT_SHA256', {w.paths['prepare'].name: '0' * 64})
    with pytest.raises(AssertionError, match='round 6 pin'):
        D.assert_frozen('prepare', w.paths['prepare'], w.pinned['prepare'])


def test_input_manifest_refuses_rewritten_frozen_pin(tmp_path, monkeypatch):
    # Use original context manager, not the world's substituted scoring context.
    p = tmp_path / 'frozen.npz'
    p.write_bytes(b'new bytes')
    monkeypatch.setattr(D.R, 'TEST', tmp_path)
    monkeypatch.setattr(D.R, 'INPUT_SHA256', {'frozen.npz': '0' * 64})
    monkeypatch.setattr(D.R, 'input_paths', lambda b: {'frozen.npz': p})
    with pytest.raises(AssertionError, match='frozen input SHA differs'):
        with D.input_manifest('qwen3vl_emb_2b', {'frozen.npz': D.path_record(p)}):
            pass


def test_seed_record_routes_cells_and_checks_pick_means(tmp_path, monkeypatch):
    # Call the real seed-record plumbing; statistics are replaced with fixed
    # outputs so this test observes routing without a feature or a bootstrap.
    cells_path = tmp_path / 'cells.json'
    cells = {'aff': [1, 2], 'cf': [3, 4], 'r1_fused': [5, 6], 'r1_counterpart': [7, 8]}
    rewrite(cells_path, cells)
    picks_path = tmp_path / 'picks.json'
    means = {n: float(i + 1) for i, n in enumerate(D.P.NESTED)}
    rewrite(picks_path, {'mean_r1': means})
    f = __file__
    checks = {f'P{i}': {'point': 1., 'n': 1} for i in range(1, 8)}
    rh = SimpleNamespace(__file__=f, per_episode_arrays=lambda s: s)
    rs = SimpleNamespace(__file__=f, sensitivity_record=lambda s: {})
    st = SimpleNamespace(__file__=f, CHECKS=tuple(checks), pass_record=lambda *a: {'checks': checks})
    monkeypatch.setattr(D, 'round6_statistics', lambda: (rh, rs, st))
    nums = {f'{n}_mean_r1': {'got': v} for n, v in means.items()}
    nums.update(aff_fused_r1={'got': 4.}, aff_cf_r1={'got': 3.})
    monkeypatch.setattr(D.S0A, 'number_items', lambda *a: (nums, {}))
    monkeypatch.setattr(D.S0A, 'round3_arrays', lambda *a: {
        f'key{i}': np.array(D.S.CELLS['aff']['fused'], dtype=np.int64) for i in range(39)})
    env = SimpleNamespace(inputs={}, coef_sha256={})
    bundle = SimpleNamespace(episodes_sha256={}, cl=np.array([1]), pair_index=np.array([0]))
    scored = {s: {m: np.array([1.]) for m in D.S0A.METRICS} for s in D.S0A.AB_SCORERS}
    original = D.S.CELLS
    rec, _, _ = D.seed42_record(env, bundle, scored, {'cells_seed42': cells_path, 'picks_seed42': picks_path})
    assert D.S.CELLS is original
    assert rec['arrays']['seed42_arrays/key0']['sha256'] == hashlib.sha256(np.array([1, 2], dtype=np.int64).tobytes()).hexdigest()
    rewrite(picks_path, {'mean_r1': dict(means, B=99.)})
    with pytest.raises(AssertionError, match='picks.*mean'):
        D.seed42_record(env, bundle, scored, {'cells_seed42': cells_path, 'picks_seed42': picks_path})


def test_copied_picks_keys_and_targets_metadata(tmp_path):
    a, b = tmp_path / 'picks.json', tmp_path / 'clip.json'
    rewrite(a, {'targets': None, 'convention': 'same'})
    rewrite(b, {'targets': {'B': 18.}, 'convention': 'same'})
    producer = {'keys_made': [], 'keys_copied': ['targets', 'convention']}
    rec = D.assert_part('picks_seed42', a, b, producer)
    assert rec['keys_copied'] == ['targets', 'convention']
    assert rec['keys_tested_equal'] == ['convention']
    assert rec['keys_tested_backbone_metadata'] == ['targets']
    rewrite(a, {'targets': None, 'convention': 'changed'})
    with pytest.raises(AssertionError, match='copied picks_seed42/convention'):
        D.assert_part('picks_seed42', a, b, producer)
    rewrite(a, {'targets': {'B': 18.}, 'convention': 'same'})
    with pytest.raises(AssertionError, match='targets must be null'):
        D.assert_part('picks_seed42', a, b, producer)


def test_picked_is_tied_to_backbones_checkpoint(tmp_path):
    checkpoint = tmp_path / 'A3_seed42.pt'
    checkpoint.write_bytes(b'checkpoint')
    a, b = tmp_path / 'picked.json', tmp_path / 'clip.json'
    rewrite(a, {'run': 'A3', 'seed': 42, 'checkpoint_sha256': D.R.sha256_file(checkpoint)})
    rewrite(b, {'run': 'A3', 'seed': 42, 'checkpoint_sha256': '0' * 64})
    producer = {'keys_made': ['checkpoint_sha256'], 'keys_copied': ['run', 'seed']}
    rec = D.assert_part('picked', a, b, producer, parts={'A3_seed42': checkpoint})
    assert rec['keys_tested_equal'] == ['run', 'seed']
    assert rec['keys_tested_backbone_metadata'] == ['checkpoint_sha256']
    rewrite(a, {'run': 'A3', 'seed': 42, 'checkpoint_sha256': '0' * 64})
    with pytest.raises(AssertionError, match='checkpoint_sha256'):
        D.assert_part('picked', a, b, producer, parts={'A3_seed42': checkpoint})


def test_input_files_checks_a3_build_record_and_groups(tmp_path):
    partitions = tmp_path / 'partitions.npz'
    np.savez(partitions, local_groups=np.array([0, 1], dtype=np.int64))
    build = tmp_path / 'build_record.json'
    build.write_text('{}')
    sha = hashlib.sha256(np.array([0, 1], dtype=np.int64).tobytes()).hexdigest()
    rec = {'input_sha256': {'parts': {'partitions': str(partitions), 'build_record': str(build)},
                           'part_sha256': {'build_record.json': D.R.sha256_file(build), 'local_groups': sha}}}
    assert D._input_files(rec, {'partitions': partitions}) == rec['input_sha256']['part_sha256']
    rec['input_sha256']['part_sha256']['local_groups'] = '0' * 64
    with pytest.raises(AssertionError, match='local_groups input array changed'):
        D._input_files(rec, {'partitions': partitions})


@pytest.mark.parametrize('bad', ['module', 'script'])
def test_run_step_checks_code_before_subprocess(world, monkeypatch, bad):
    w = world
    (w.recipe / 'stage2_groups_files.json').unlink()
    if bad == 'module':
        w.gate['module_sha256'][next(iter(w.gate['module_sha256']))] = '0' * 64
    else:
        w.gate['item4']['scripts']['r9_recipe_groups.py']['sha256'] = '0' * 64
    rewrite(w.gate_path, w.gate)
    monkeypatch.setattr(D.subprocess, 'run', lambda *args, **kwargs: pytest.fail('stale recipe ran'))
    with pytest.raises(AssertionError):
        D.run_step(w.b, 'groups', stage0b_path=w.gate_path)


def test_main_rechecks_closure_between_steps(world, monkeypatch):
    w = world
    seen = []
    def first_step(b, step, **kwargs):
        seen.append(step)
        assert len(seen) == 1, 'stale second step ran'
        w.gate['module_sha256'][next(iter(w.gate['module_sha256']))] = '0' * 64
        rewrite(w.gate_path, w.gate)
        return 0
    monkeypatch.setattr(D, 'run_step', first_step)
    with pytest.raises(AssertionError, match='rerun the stage 0b compare'):
        D.main(['--backbone', w.b, '--step', 'all', '--stage0b', str(w.gate_path)])
    assert seen == ['groups']


# ---- fix wave fx-tests (B2 M-3, 13 fix 2 L1): one test per guard the final review's mutations left alive ----

def edit_record(w, role, **changes):
    path = Path(str(w.paths[role]) + '.record.json')
    rec = D.read_json(path)
    rec.update(changes)
    rewrite(path, rec)


@pytest.mark.parametrize('bad, message', [
    ('other_backbone', 'rc_tau: record names another backbone'),            # run_r9_stage2.py :360
    ('file_sha', 'rc_tau: file SHA differs from its record'),               # :362
    ('wrong_producer', 'rc_tau: record names the wrong producer'),          # :364
    ('upstream_hash', 'baselines_seed42: upstream rc_tau changed'),         # :370
    ('upstream_unknown_role', 'baselines_seed42: upstream nothing changed'),
    ('key_missing_from_record', 'partitions: incomplete provenance keys'),  # :200
    ('key_missing_from_file', 'partitions: incomplete provenance keys'),
])
def test_provenance_record_guards_refuse(world, bad, message):
    w = world
    if bad == 'other_backbone':
        edit_record(w, 'rc_tau', backbone='clip_b32')
    elif bad == 'file_sha':
        edit_record(w, 'rc_tau', file_sha256='0' * 64)
    elif bad == 'wrong_producer':
        edit_record(w, 'rc_tau', script=str(HERE / D.STEP_SCRIPTS['groups']))
    elif bad == 'upstream_hash':
        edit_record(w, 'baselines_seed42', upstream_sha256={'rc_tau': '0' * 64})
    elif bad == 'upstream_unknown_role':
        edit_record(w, 'baselines_seed42', upstream_sha256={'nothing': D.R.sha256_file(w.paths['rc_tau'])})
    elif bad == 'key_missing_from_record':
        edit_record(w, 'partitions', keys_unrefit=[])
    else:
        np.savez(w.paths['partitions'], fit=np.array([2.], np.float32), rows=np.array([1, 2]),
                 unused=np.array([np.nan]), extra=np.array([0.]))
    with pytest.raises(AssertionError, match=message):
        D.provenance(w.b, w.paths, w.gate)


def test_provenance_accepts_the_correct_upstream_hash(world):
    w = world
    edit_record(w, 'baselines_seed42', upstream_sha256={'rc_tau': D.R.sha256_file(w.paths['rc_tau'])})
    assert set(D.provenance(w.b, w.paths, w.gate)) == set(D.BB.RECIPE_ROLES)


def test_reader_json_feature_names_must_equal_clips(tmp_path):
    a, b = tmp_path / 'reader.json', tmp_path / 'clip.json'
    rewrite(a, {'feature_names': ['two']})
    rewrite(b, {'feature_names': ['one']})
    with pytest.raises(AssertionError, match='reader feature layout differs'):          # :226
        D.assert_part('rb_reader_A0_json', a, b, {'keys_made': ['feature_names'], 'keys_copied': []})


def _stage_world(w):
    record_world(w)
    stage_path = w.out / f'stage2_{w.b}.json'
    return stage_path, D.read_json(stage_path)


@pytest.mark.parametrize('bad, message', [
    ('stage0b_changed', 'stage 0b changed'),                      # :558
    ('closure_changed', 'recipe module closure changed'),         # :561
    ('receipt_changed', 'receipt changed after stage 2'),         # :566
    ('record_changed', 'partitions: stage 2 record changed'),     # :573
    ('store_changed', 'feature store changed'),                   # :575
    ('frozen_changed', 'prepare: frozen file changed'),           # :578
    ('frozen_reassert', 'frozen prepare differs from pins_clip.json'),   # :579
    ('seed42_changed', 'seed42_record: changed after stage 2'),   # :583
    ('sensitivity_changed', 'sensitivity_seed42: changed after stage 2'),
])
def test_pins_evidence_stage_snapshot_guards_refuse(world, bad, message):
    w = world
    stage_path, stage = _stage_world(w)
    if bad == 'stage0b_changed':
        stage['stage0b']['sha256'] = '0' * 64
    elif bad == 'closure_changed':
        stage['module_sha256'][next(iter(stage['module_sha256']))] = '0' * 64
    elif bad == 'receipt_changed':
        step = next(iter(stage['recipe_receipts']))
        stage['recipe_receipts'][step]['sha256'] = '0' * 64
    elif bad == 'record_changed':
        stage['provenance']['partitions']['record']['sha256'] = '0' * 64
    elif bad == 'store_changed':
        file = next(iter(stage['provenance']['rc_tau']['feature_store_sha256']))
        stage['provenance']['rc_tau']['feature_store_sha256'][file] = '0' * 64
    elif bad == 'frozen_changed':
        stage['frozen_parts']['prepare']['sha256'] = '0' * 64
    elif bad == 'frozen_reassert':
        # The stage snapshot is refreshed, so only the re-assert against pins_clip.json can notice.
        np.savez(w.paths['prepare'], **{k: np.array([1, 3]) for k in D.FROZEN_KEYS['prepare']})
        stage['frozen_parts']['prepare'] = D.path_record(w.paths['prepare'])
    else:
        name = 'sensitivity_seed42' if bad == 'sensitivity_changed' else 'seed42_record'
        Path(stage[name]['path']).write_text(json.dumps(D.read_json(stage[name]['path']), indent=3))
    rewrite(stage_path, stage)
    with pytest.raises(AssertionError, match=message):
        D.run_pins(w.b, w.paths, w.out, w.gate_path)
    assert not D.R.pins_path(w.b).exists()


def test_replay_skip_is_only_the_roles_own_file_hash(world, monkeypatch):
    # rc_tau reaches the feature stores only through another role's file hash (partitions). A replay skip widened
    # from "this role's own file hash" to "any role's file hash" would cut that edge (13 fix 2 re-review, L1).
    w = world
    replay_world(w, monkeypatch)
    edit_record(w, 'rc_tau', input_sha256={'partitions': D.R.sha256_file(w.paths['partitions'])})
    out = D.provenance(w.b, w.paths, w.gate, replay=True)
    assert out['rc_tau']['feature_store_sha256'] == w.stores
    assert out['rc_tau']['skipped_self_input_sha256'] == {}
    assert out['partitions']['skipped_self_input_sha256'] == {'partitions': D.R.sha256_file(w.paths['partitions'])}
