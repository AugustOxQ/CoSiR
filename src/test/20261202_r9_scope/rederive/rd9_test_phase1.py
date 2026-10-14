"""fx-c13: the C13 phase-1 comparison on the stage-2 pins schema (namespaced `seed42.arrays`).

One self-consistent scenario agrees. Each pinned quantity of rule section 7.5 is then changed on its own, and the test
asserts that exactly its key disagrees: deleting the comparison of a quantity from rd9_compare fails its test (B2 M-2).
The minimum array sets (C MEDIUM-3), the CLIP branch's B/B0/B1 means and the real pins files (B2 M-1) are covered too.
"""
import copy
import hashlib
import json
import pickle
import sys
import numpy as np
import pytest
import rd9_compare as C
import rd9_inputs as I
import rd9_stats as S
from rd9_test_compare import make_arrays, write

GATES = ('aff_gate', 'P', 'margin', 'pick')
METRICS = ('r1', 'gain', 'other', 'swap', 'strict')


def entry(v):
    return {'sha256': I.sha_bytes(v), 'dtype': str(v.dtype), 'shape': list(v.shape)}


def scenario():
    arrays = make_arrays()
    n = len(arrays['cl'])
    rng = np.random.default_rng(5)
    arrays['B1__r1'] = rng.integers(0, 3, n) / 4
    arrays['parity'] = np.arange(n) % 2
    arrays['aff_fused_cells'] = np.array([39, 119])
    arrays['aff_cf_cells'] = np.array([149, 10])
    arrays['taus'] = np.array([0., .2, .4, .6])
    arrays['aff_sigma'] = np.array([0., 0.])
    for prefix in GATES:
        for side in 'ab':
            arrays[f'{prefix}__{side}'] = rng.random((n, 3))
    stats = S.pass_record(arrays, sensitivity=True)
    picks = {'aff': [39, 119], 'cf': [149, 10], 'sigma_star': [0., 0.], 'switched_off': False, 'gate_share': 0.25,
             'B': [[16., 8.], [8., 2.]], 'B0': [[16., 16.], [16., 8.]], 'rca_lambda': [0.5, 1.],
             'taus': [3.8684538364530674e-05, 0.21702129490553143, 0.47973989883399526, 0.7502585816077211]}
    own = {'backbone': 'qwen3vl_emb_2b', 'picks': picks, 'stats': stats}
    pinned = {}
    for name, v in arrays.items():
        if name in ('cl', 'pair_index'):
            pinned['seed42_arrays/' + name] = entry(v)
        elif name.startswith(('cosine__', 'rca__')):
            pinned['per_anchor_seed42/' + name] = entry(v)
        elif name.startswith(('aff_fused__', 'aff_cf__', 'aff_gate__', 'P__', 'margin__', 'pick__')) or \
                name in ('parity', 'aff_fused_cells', 'aff_cf_cells'):
            pinned['seed42_arrays/' + name] = entry(v)
    pinned['per_anchor_seed42/anchor_group'] = entry(arrays['cl'])
    pinned['per_anchor_seed42/pair_index'] = entry(arrays['pair_index'])
    pinned['seed42_arrays/r1_fused__r1'] = entry(arrays['aff_fused__r1'])    # pinned, never derived here: not compared
    mean = lambda key: 100 * float(np.mean(arrays[key + '__r1']))
    record = {'mean_r1': {'B': mean('B'), 'B0': mean('B0'), 'B1': mean('B1'), 'AFF': mean('aff_fused'), 'CF': mean('aff_cf')},
              'checks': {j: {'point': stats['checks'][j]['point'], 'n_j': stats['checks'][j]['n']} for j in S.CHECK_IDS},
              'arrays': pinned}
    sigma = {j: {'quantity': S.QUANTITIES[j], 'sigma_a2': stats['sensitivity'][j]['sigma_a2'],
                 'sigma_eps2': stats['sensitivity'][j]['sigma_e2']} for j in S.CHECK_IDS}
    sigma.update(S1={'sigma_a2': 1., 'sigma_eps2': 2.}, S2={'sigma_a2': 0., 'sigma_eps2': 3.}, seed=42,
                 N=stats['n_episodes'], n_paintings=stats['n_clusters'])
    pinned_side = copy.deepcopy(dict(    # the pinned files must not share objects with the derived record
        tau={'taus': list(picks['taus'])},
        cells={k: picks[k] for k in ('aff', 'cf', 'sigma_star', 'switched_off', 'gate_share')},
        picks={name: {str(h): picks[name][h] for h in (0, 1)} for name in ('B', 'B0')},
        base={'lambda_picks': {'rca': {'0': 0.5, '1': 1.}}}))
    return dict(own=own, arrays=arrays, record=record, sigma=sigma, **pinned_side)


def compare(s):
    c = C.Comparison()
    C.compare_phase1(c, s['own'], s['arrays'], s['cells'], s['picks'], s['base'], s['tau'], s['record'], s['sigma'])
    record = c.record()
    return record['agreement'], {i['key'] for i in record['items'] if not i['equal']}


def test_a_consistent_scenario_agrees():
    agreement, failed = compare(scenario())
    assert agreement and not failed


def edit(path, value):
    def apply(s):
        target = s
        for step in path[:-1]: target = target[step]
        target[path[-1]] = value(target[path[-1]]) if callable(value) else value
    return apply


# Each case changes one pinned value; the key that must fail is a prefix of the comparer's keys.
CASES = [
    ('sigma_star', edit(('cells', 'sigma_star', 0), 0.1), 'picks.sigma_star'),
    ('aff_cell', edit(('cells', 'aff', 0), 40), 'picks.aff'),
    ('cf_cell', edit(('cells', 'cf', 1), 11), 'picks.cf'),
    ('switched_off', edit(('cells', 'switched_off'), True), 'picks.switched_off'),
    ('gate_share', edit(('cells', 'gate_share'), 0.26), 'picks.gate_share'),
    ('B_pick', edit(('picks', 'B', '1'), [8., 4.]), 'picks.B.'),
    ('B0_pick', edit(('picks', 'B0', '0'), [16., 8.]), 'picks.B0.'),
    ('rca_lambda', edit(('base', 'lambda_picks', 'rca', '0'), 0.75), 'picks.rca_lambda'),
    ('tau', edit(('tau', 'taus', 1), lambda v: v * (1 + 1e-6)), 'picks.taus'),
    ('mean_B', edit(('record', 'mean_r1', 'B'), lambda v: v + 1e-6), 'seed42.mean_r1.B'),
    ('mean_B0', edit(('record', 'mean_r1', 'B0'), lambda v: v + 1e-6), 'seed42.mean_r1.B0'),
    ('mean_B1', edit(('record', 'mean_r1', 'B1'), lambda v: v + 1e-6), 'seed42.mean_r1.B1'),
    ('mean_AFF', edit(('record', 'mean_r1', 'AFF'), lambda v: v + 1e-6), 'seed42.mean_r1.AFF'),
    ('mean_CF', edit(('record', 'mean_r1', 'CF'), lambda v: v + 1e-6), 'seed42.mean_r1.CF'),
    ('P1_point', edit(('record', 'checks', 'P1', 'point'), lambda v: v + 1e-6), 'seed42.checks.P1.point'),
    ('P7_point', edit(('record', 'checks', 'P7', 'point'), lambda v: v + 1e-6), 'seed42.checks.P7.point'),
    ('P4_n_j', edit(('record', 'checks', 'P4', 'n_j'), lambda v: v + 1), 'seed42.checks.P4.n'),
    ('array_hash', edit(('record', 'arrays', 'seed42_arrays/aff_fused__r1', 'sha256'), '0' * 64), 'seed42.arrays.aff_fused__r1'),
    ('per_anchor_hash', edit(('record', 'arrays', 'per_anchor_seed42/rca__swap', 'sha256'), '0' * 64), 'seed42.arrays.rca__swap'),
    ('anchor_group_hash', edit(('record', 'arrays', 'per_anchor_seed42/anchor_group', 'sha256'), '0' * 64), 'seed42.arrays.cl'),
    # the seed-42 sigma inputs (A1 M1)
    ('sigma_a2', edit(('sigma', 'P3', 'sigma_a2'), lambda v: v + 1e-6), 'seed42.sigma.P3.sigma_a2'),
    ('sigma_e2', edit(('sigma', 'P6', 'sigma_eps2'), lambda v: v * 1.001), 'seed42.sigma.P6.sigma_e2'),
    ('sigma_N', edit(('sigma', 'N'), 12289), 'seed42.sigma.N'),
    ('sigma_clusters', edit(('sigma', 'n_paintings'), 4603), 'seed42.sigma.n_paintings'),
]


@pytest.mark.parametrize('name,change,key', CASES, ids=[c[0] for c in CASES])
def test_one_changed_pinned_quantity_disagrees(name, change, key):
    s = scenario()
    change(s)
    agreement, failed = compare(s)
    assert not agreement
    assert failed and all(k.startswith(key) for k in failed), failed


def test_picks_are_compared_exactly_and_taus_to_a_relative_tolerance():
    s = scenario()
    s['tau']['taus'][1] *= 1 + 1e-12       # inside rtol 1e-9
    assert compare(s)[0]
    s = scenario()
    s['picks']['B0']['1'] = [16., 8. + 1e-12]    # a pick is a grid value: no tolerance
    assert not compare(s)[0]


def test_absent_sigma_file_or_check_disagrees():
    s = scenario()
    s['sigma'] = None
    assert not compare(s)[0]
    s = scenario()
    del s['sigma']['P2']
    agreement, failed = compare(s)
    assert not agreement and failed == {'seed42.sigma.P2.sigma_a2', 'seed42.sigma.P2.sigma_e2'}


# ---- the minimum array sets (C MEDIUM-3)

FLOOR = ['cl', 'pair_index', 'parity', 'aff_fused_cells', 'aff_cf_cells'] + \
        [f'{p}__{side}' for p in GATES for side in 'ab'] + \
        [f'{arm}__{m}' for arm in ('aff_fused', 'aff_cf', 'cosine', 'rca') for m in METRICS]


@pytest.mark.parametrize('name', FLOOR)
def test_a_derivation_without_a_floor_array_disagrees(name):
    s = scenario()
    del s['arrays'][name]
    agreement, failed = compare(s)
    assert not agreement
    assert f'seed42.arrays.{name}.present' in failed


def test_the_means_need_their_arrays():
    for arm in ('B', 'B0', 'B1', 'aff_fused', 'aff_cf'):
        s = scenario()
        del s['arrays'][arm + '__r1']
        agreement, failed = compare(s)
        assert not agreement and any(k.startswith('seed42.mean_r1.') for k in failed)


def test_pinned_arrays_the_derivation_does_not_emit_are_not_compared():
    s = scenario()
    assert 'seed42_arrays/r1_fused__r1' in s['record']['arrays']
    assert compare(s)[0]


# ---- main(): the stage-2 pins schema, a fake Inputs that serves the part files

class FakeInputs:
    def __init__(self, backbone, pins, s):
        self.backbone, self.pin_record = backbone, json.loads(open(pins).read())
        self.s = s
    def json(self, role, keys=()):
        s = self.s
        return {'cells_seed42': s['cells'], 'picks_seed42': s['picks'], 'baselines_seed42': s['base'], 'rc_tau': s['tau']}[role]


def stage_main(tmp_path, monkeypatch, s):
    monkeypatch.setattr(I, 'RD', tmp_path)
    sigma_path = write(tmp_path / 'sensitivity_seed42_qwen3vl_emb_2b.json', s['sigma'])
    pins = write(tmp_path / 'pins_qwen3vl_emb_2b.json', {
        'backbone': 'qwen3vl_emb_2b', 'parts': {}, 'seed42': s['record'],
        'sensitivity_seed42': {'path': str(sigma_path), 'sha256': hashlib.sha256(sigma_path.read_bytes()).hexdigest()}})
    s['own']['inputs'] = {str(pins): {'sha256': I.sha_file(pins)}}
    own = write(tmp_path / 'own.json', s['own'])
    np.savez(tmp_path / 'own.npz', **s['arrays'])
    monkeypatch.setattr(I, 'Inputs', lambda backbone, pinned: FakeInputs(backbone, pinned, s))
    return pins, own, sigma_path


def run_main(tmp_path, monkeypatch, pins, own, out='agr.json'):
    monkeypatch.setattr(sys, 'argv', ['rd9_compare.py', '--phase1', '--backbone', 'qwen3vl_emb_2b', '--pins', str(pins),
                                      '--derived', str(own), '--derived-arrays', str(tmp_path / 'own.npz'),
                                      '--out', str(tmp_path / out)])
    C.main()
    return json.loads((tmp_path / out).read_text())


def test_main_reads_the_sigma_file_the_pins_name(tmp_path, monkeypatch):
    s = scenario()
    pins, own, _ = stage_main(tmp_path, monkeypatch, s)
    got = run_main(tmp_path, monkeypatch, pins, own)
    assert got['agreement'] is True and got['phase'] == 1
    assert any(i['key'] == 'seed42.sigma.P1.sigma_a2' for i in got['items'])
    assert any(i['key'].startswith('seed42.arrays.P__a') for i in got['items'])


def test_main_refuses_a_sigma_file_that_differs_from_its_pin(tmp_path, monkeypatch):
    s = scenario()
    pins, own, sigma_path = stage_main(tmp_path, monkeypatch, s)
    changed = json.loads(sigma_path.read_text())
    changed['P1']['sigma_a2'] += 1.
    write(sigma_path, changed)
    with pytest.raises(AssertionError, match='sensitivity_seed42_qwen3vl_emb_2b.json'):
        run_main(tmp_path, monkeypatch, pins, own)


def test_main_requires_the_sigma_record_in_the_pins(tmp_path, monkeypatch):
    s = scenario()
    pins, own, _ = stage_main(tmp_path, monkeypatch, s)
    record = json.loads(pins.read_text())
    del record['sensitivity_seed42']
    pins.write_text(json.dumps(record))
    own_record = json.loads(own.read_text())
    own_record['inputs'] = {str(pins): {'sha256': I.sha_file(pins)}}
    own.write_text(json.dumps(own_record))
    with pytest.raises(AssertionError, match='sensitivity_seed42'):
        run_main(tmp_path, monkeypatch, pins, own)


# ---- main(), CLIP branch (R6's files): a tree of synthetic round files under a monkeypatched T

def stage_clip(tmp_path, monkeypatch):
    s = scenario()
    s['own']['backbone'] = 'clip_b32'
    stats = s['own']['stats']
    stats.update(seeds=[42], episodes_sha256={'42': {'x': 'a' * 64}})
    monkeypatch.setattr(I, 'RD', tmp_path)
    monkeypatch.setattr(C, 'T', tmp_path / 'T')
    res = tmp_path / 'T/20261125_artelingo_held_test/results'
    report = copy.deepcopy({k: v for k, v in stats.items() if k not in ('sensitivity', 'decision')})
    for j, check in report['checks'].items(): check['quantity'] = S.QUANTITIES[j]
    res.mkdir(parents=True)
    write(res / 'seed42_pass_counts.json', report)
    picks = s['own']['picks']
    write(res / 'picks_seed42.json', {'B': s['picks']['B'], 'B0': s['picks']['B0'], 'lambda_picks': {'rca': s['base']['lambda_picks']['rca']},
                                      'mean_r1': {k: s['record']['mean_r1'][k] for k in ('B', 'B0', 'B1')}})
    sigma_path = write(res / 'sensitivity_seed42.json', s['sigma'])
    monkeypatch.setattr(I, 'R6_SENSITIVITY_SEED42_SHA256', hashlib.sha256(sigma_path.read_bytes()).hexdigest())
    tau = tmp_path / 'T/20261117_reader_fix_csd/results'
    tau.mkdir(parents=True)
    write(tau / 'rc_tau.json', {'taus': picks['taus']})
    r3 = tmp_path / 'T/20261121_round3_affect_gate/results'
    r3.mkdir(parents=True)
    np.savez(r3 / 'seed42_arrays.npz', **s['arrays'])
    pins = write(tmp_path / 'pins_clip.json', {'backbone': 'clip_b32'})
    s['own']['inputs'] = {str(pins): {'sha256': I.sha_file(pins)}}
    return s, pins


def run_clip(tmp_path, monkeypatch, s, pins, arrays=None):
    own = write(tmp_path / 'own.json', s['own'])
    np.savez(tmp_path / 'own.npz', **(s['arrays'] if arrays is None else arrays))
    monkeypatch.setattr(sys, 'argv', ['rd9_compare.py', '--clip-phase1', '--pins', str(pins), '--derived', str(own),
                                      '--derived-arrays', str(tmp_path / 'own.npz'), '--out', str(tmp_path / 'agr.json')])
    try:
        C.main()
    except SystemExit:
        pass
    return json.loads((tmp_path / 'agr.json').read_text())


def test_clip_branch_agrees_on_a_consistent_scenario(tmp_path, monkeypatch):
    s, pins = stage_clip(tmp_path, monkeypatch)
    got = run_clip(tmp_path, monkeypatch, s, pins)
    assert got['agreement'] is True
    keys = {i['key'] for i in got['items']}
    assert {'seed42.mean_r1.B', 'seed42.mean_r1.B0', 'seed42.mean_r1.B1', 'seed42.sigma.P1.sigma_a2'} <= keys


@pytest.mark.parametrize('name', ['B', 'B0', 'B1'])
def test_clip_branch_compares_the_B_means(tmp_path, monkeypatch, name):
    s, pins = stage_clip(tmp_path, monkeypatch)
    arrays = dict(s['arrays'])
    arrays[name + '__r1'] = arrays[name + '__r1'] + 0.25
    got = run_clip(tmp_path, monkeypatch, s, pins, arrays)
    failed = {i['key'] for i in got['items'] if not i['equal']}
    assert got['agreement'] is False and f'seed42.mean_r1.{name}' in failed


CLIP_FLOOR = [name for name in FLOOR if not name.startswith(('cosine__', 'rca__'))] + ['taus', 'aff_sigma']


@pytest.mark.parametrize('name', CLIP_FLOOR)
def test_clip_branch_derivation_without_a_floor_array_disagrees(tmp_path, monkeypatch, name):
    s, pins = stage_clip(tmp_path, monkeypatch)
    arrays = {k: v for k, v in s['arrays'].items() if k != name}
    got = run_clip(tmp_path, monkeypatch, s, pins, arrays)
    failed = {i['key'] for i in got['items'] if not i['equal']}
    assert got['agreement'] is False and 'arrays.' + name in failed


def test_clip_branch_compares_the_sigma_file(tmp_path, monkeypatch):
    s, pins = stage_clip(tmp_path, monkeypatch)
    s['own']['stats']['sensitivity']['P5']['sigma_e2'] += 1e-6
    got = run_clip(tmp_path, monkeypatch, s, pins)
    assert {i['key'] for i in got['items'] if not i['equal']} == {'seed42.sigma.P5.sigma_e2'}


# ---- B2 M-1: the real pins files are readable by Inputs (the checkout that has the files)

@pytest.mark.parametrize('backbone', ['clip_b32', 'qwen3vl_emb_2b'])
def test_inputs_read_the_real_pins_files(backbone):
    pins = I.HERE / ('pins_clip.json' if backbone == 'clip_b32' else f'pins_{backbone}.json')
    record = json.loads(pins.read_text())
    entries = I.parts_entries(record)
    assert set(entries.values()) == {p['sha256'] for p in record['parts'].values()}
    # logical keys of input_sha256 (relative to src/test, carrying the backbone's SHA-256) are not read
    assert not any(key in entries for key in record['input_sha256'])
    missing = [p for p in entries if not I.resolve_pin(p).exists()]
    if missing: pytest.skip(f'the pinned files are not in this checkout: {missing[0]}')
    inputs = I.Inputs(backbone)
    assert all(inputs.paths[role].exists() for role in I.ROLES)
    assert inputs.paths['cells_seed42'].resolve() == I.resolve_pin(record['parts']['cells_seed42']['path']).resolve()


def test_parts_entries_need_path_and_sha256():
    assert I.parts_entries({'parts': {'a': {'path': 'x/a.npz', 'sha256': '1' * 64, 'extra': 1}}}) == {'x/a.npz': '1' * 64}
    with pytest.raises(AssertionError):
        I.parts_entries({'parts': {'a': {'path': 'x/a.npz'}}})
    with pytest.raises(AssertionError):
        I.parts_entries({'parts': {}})


# ---- A1 L5: unpickling must not import the chain under test

def test_unpickle_refuses_a_pickle_that_imports_an_r9_module(tmp_path, monkeypatch):
    module = tmp_path / 'r9_fx_probe_module.py'
    module.write_text('class Thing:\n    pass\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    import r9_fx_probe_module
    raw = pickle.dumps(r9_fx_probe_module.Thing())
    del sys.modules['r9_fx_probe_module']
    with pytest.raises(AssertionError, match='r9_fx_probe_module'):
        I.unpickle(raw)
    sys.modules.pop('r9_fx_probe_module', None)


def test_unpickle_reads_plain_data():
    assert I.unpickle(pickle.dumps({'a': np.arange(3)}))['a'].tolist() == [0, 1, 2]


def test_the_reader_pickle_is_loaded_through_unpickle(tmp_path, monkeypatch):
    module = tmp_path / 'r9_fx_probe_reader.py'
    module.write_text('class Thing:\n    pass\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    import r9_fx_probe_reader
    pkl = tmp_path / 'reader.pkl'
    pkl.write_bytes(pickle.dumps({'config': 'A0', 'chain': r9_fx_probe_reader.Thing()}))
    del sys.modules['r9_fx_probe_reader']
    inputs = I.Inputs.__new__(I.Inputs)
    inputs.json = lambda role, keys=(): {}
    inputs.path = lambda role, keys=(): pkl
    try:
        with pytest.raises(AssertionError, match='r9_fx_probe_reader'):
            inputs.readers()
    finally:
        sys.modules.pop('r9_fx_probe_reader', None)


def test_no_rederive_script_unpickles_without_the_check():
    from pathlib import Path
    for path in sorted(Path(I.RD).glob('rd9_*.py')):
        if path.name.startswith('rd9_test') or path.name in ('rd9_mutations.py', 'rd9_inputs.py'):    # inputs: the helper
            continue
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if 'pickle.load' in line:
                pytest.fail(f'{path.name}:{number} unpickles directly; use rd9_inputs.unpickle')
