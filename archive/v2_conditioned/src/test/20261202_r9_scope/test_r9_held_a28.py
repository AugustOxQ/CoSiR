"""Ticket 21: C5 boundaries, deletion mutants and A28 orchestration; tmp_path ledgers only."""
import json
import re
from types import SimpleNamespace as NS

import numpy as np
import pytest

import run_r9_held as U
from test_r9_held import case, configure, hashes, mutant, put, stub_read

GUARDS = {}


@pytest.mark.parametrize('read', ['H8', 'H9', 'H10'])
def test_a28_read_names(read):
    for variant in ('held', 'fix1', 'reserve', 'smoke'):
        n = U.read_names(read, variant)
        suffix = '_reserve' if variant == 'reserve' else ''
        assert n.verdict == f'held_verdict_{read}{suffix}.json'
        assert n.started == f'held_started_{read}{suffix}.json'
        assert n.pass_ == f"held_pass_{read}{'_fix1' if variant == 'fix1' else suffix}.json"


def test_stage3_verdicts_required(case):
    with pytest.raises(U.Refused, match='H6.*H7'):
        U.a28_order_guard(case.res, case.ledger, 'H8')

GUARDS.update({
    'ledger_a28_sha': (U.Refused, 'pins_a28'),
    'a28_stage3': (U.Refused, 'H6.*H7'),
    'a28_h8_verdict': (U.Refused, 'H8 real verdict'),
    'a28_h8_started': (U.Refused, 'H8 started file'),
    'a28_h8_hashes': (U.Refused, 'differ from H8'),
    'a28_rerun_parts': (U.Refused, 'part records differ'),
    'a28_records': (AssertionError, 'records must exist'),
    'a28_label_join': (AssertionError, 'row space differs'),
    'a28_not_smoke': (AssertionError, 'must not be smoke'),
    'a28_coef': (AssertionError, 'coefficients differ'),
    'a28_set_aside': (U.AE.SetAsideStop, 'more than two'),
    'a28_pins': (SystemExit, 'A28 pins checked'),
    'a28_seeds': (AssertionError, 'builder seeds checked'),
    'a28_pass_shape': (U.AR.Stop, 'near_duplicates'),
})


def verdict(mod, c, read, value=None, status=None):
    v = dict(read=read, backbone=mod.LEDGER_NAMES[mod.READ_BACKBONES[read]], mode='held',
             rule_sha256=mod.R.RULE_SHA256,
             verdict=value or ('GO' if read in ('H6', 'H7') else 'full claim'))
    if status:
        v.update(verdict=None, status=status)
    put(c.res / mod.read_names(read).verdict, v)


def preflight(mod, c, mp, read='H8', bad=None):
    from test_r9_held import ledger, append_h5
    configure(mod, c, mp)
    c.backbone = mod.READ_BACKBONES[read]
    c.a28pins = c.res / 'pins_a28.json'
    put(c.a28pins, {'files': {}})
    mp.setattr(mod.AD, 'PINS_PATH', c.a28pins)
    ledger(c, row=read, scripts=f"run_r9_held.py {'a'*64}; {c.pins.name} {mod.R.sha256_file(c.pins)}; "
                              f"pins_a28.json {('b'*64) if bad == 'ledger_a28_sha' else mod.R.sha256_file(c.a28pins)}")
    append_h5(c, mod)
    for r in ('H6', 'H7'):
        with c.ledger.open('a') as f:
            f.write(f'| {r} | date | held | {mod.LEDGER_NAMES[mod.READ_BACKBONES[r]]} | AFF | pending | script | GO |\n')
        if bad != 'a28_stage3' or r != 'H7':
            verdict(mod, c, r)
    if read != 'H8' and bad != 'a28_h8_verdict':
        verdict(mod, c, 'H8')
    mp.setattr(mod, 'smoke_guard', lambda *a: {})
    mp.setattr(mod, 'run_read', lambda *a: 0)
    if bad == 'a28_h8_started':
        put(c.res / mod.read_names('H8').started,
            {'rule_sha256': mod.R.RULE_SHA256, 'attempts': [{'flags': {}, 'episodes_sha256': {}}]})
    else:
        mp.setattr(mod, 'a28_reference', lambda *a: None)
    code = mod.run_held(c.backbone, read, results=c.res, ledger=c.ledger, folder=c.folder)
    if code == 4:
        raise mod.Refused({g: regex for g, (_, regex) in GUARDS.items()}.get(bad, 'refused'))
    return code


def eps_of(mod, seed):
    per_pair = [NS(anchor=np.array([0]), candidates=np.zeros((1, 13), dtype=np.int64)) for _ in mod.R.PAIRS]
    return NS(seed=seed, sha=hashes(seed), per_pair=per_pair,
              part_sha={p: {nm: f'{seed * 1000 + j * 30 + i:064x}' for i, nm in enumerate(mod.AE.LANGUAGES)}
                        for j, p in enumerate(mod.R.PAIR_NAMES)},
              shares={p: {nm: 1 for nm in mod.AE.LANGUAGES} for p in mod.R.PAIR_NAMES},
              language={p: np.array([1], dtype=np.int64) for p in mod.R.PAIR_NAMES},
              set_aside={p: [] for p in mod.R.PAIR_NAMES}, set_aside_reasons={p: {} for p in mod.R.PAIR_NAMES})


def stub_a28(mod, c, mp, read='H8', variant='held', bad=None, record_bad='eps'):
    spec, env, events, scores, _ = stub_read(mod, c, mp, read, variant)
    spec.mode, spec.ctx_mode, spec.seeds, spec.h5, spec.reference = 'a28', 'a28', (55, 56, 57), {}, None
    mp.setattr(mod.R, 'N_PER_PAIR', 1)
    mp.setattr(mod.ST.R6C, 'N_PER_PAIR', 1)
    env.heads, env.pm = {}, object()
    data = NS(a28=NS(read_set_sha256='c'*64, **{a: np.array([0, 1]) for a in mod.R.ASPECTS}),
              prepared=object(), values={}, value_record={'a28': 'value record'}, flagged_languages=[np.str_('Urdu')],
              near_duplicates={'cos_099': np.int64(3), 'cos_098': np.int64(7)},
              near_source={'file': 'near_duplicates_a28.json', 'sha256': 'e' * 64})
    pins = c.res / 'pins_a28.json'
    put(pins, {})
    mp.setattr(mod.AD, 'PINS_PATH', pins)
    def checked_pins(path, backbones=None):
        events.append('a28_pins')
        assert path == pins and backbones == (spec.backbone,)
        if bad == 'a28_pins':
            raise SystemExit('A28 pins checked')
        return {}
    mp.setattr(mod.AD, 'check_pins', checked_pins)
    mp.setattr(mod, 'setup_a28', lambda *a: data)
    def seeds(seeds):
        events.append('builder_seeds')
        assert seeds == mod.AE.builder_seeds(spec.seeds, 27)
        if bad == 'a28_seeds':
            raise AssertionError('builder seeds checked')
    mp.setattr(mod.AE, 'assert_seeds_free', seeds)
    def build(ctx, values, seed, n):
        assert (c.res / spec.names.started).read_bytes() == spec.copy_to.read_bytes()
        events.append(f'build:{seed}')
        ep = eps_of(mod, seed)
        if bad == 'a28_label_join':
            ep.per_pair[0].candidates[0, 0] = 1
        if bad == 'a28_records' and record_bad == 'eps':
            ep.shares = None
        if bad == 'set_aside_stop' and seed == 57:
            raise mod.AE.SetAsideStop('three languages set aside')
        ep.set_aside[mod.R.PAIR_NAMES[0]] = [np.str_('Hindi')]
        if bad == 'a28_set_aside':
            ep.set_aside[mod.R.PAIR_NAMES[0]] = ['Hindi', 'Urdu', 'Thai']
        return ep
    mp.setattr(mod.AE, 'build_seed_a28', build)
    def context(a28, eps, seed, n, heads):
        # Recorder must already hold all part hashes before the context can compute cosine/posteriors.
        att = mod.read_attempts(c.res / spec.names.started)[-1]
        assert att['a28_episodes'][str(seed)]['part_sha'] == mod._plain(eps.part_sha)
        events.append(f'context:{seed}')
        return NS(eps=eps, seed=seed)
    mp.setattr(mod.AC, 'A28Context', context)
    def bundle(ctx, readers, pm, backbone):
        events.append(f'bundle:{ctx.seed}')
        b = NS(**vars(ctx.eps), episodes_sha256=ctx.eps.sha, cl=np.array([0, 1, 0]),
               coef_sha256={'wrong': 'coef'} if bad == 'a28_coef' else {}, smoke=bad == 'a28_not_smoke')
        if bad == 'a28_records' and record_bad == 'bundle':
            b.part_sha = None
        return b
    mp.setattr(mod.B, 'build_bundle_a28', bundle)
    mp.setattr(mod.ST, 'bootstrap_draws', lambda d, cl: (np.zeros(5000), np.ones(5000, dtype=bool)))
    mp.setattr(mod.ST.C, 'point_ci', lambda d, cl: mod.ST.point_ci_from_draws(d, np.zeros(5000)))
    real_pass = mod.ST.pass_record_a28
    passes = []
    def passed(scored, mode, seeds, extra, **kw):
        events.append('a28_pass')
        passes.append((mode, seeds, kw))
        rec = real_pass(scored, mode, seeds, extra, **kw)
        if bad == 'a28_pass_shape':
            rec['near_duplicates']['cos_099'] = '3'
        return rec
    mp.setattr(mod.ST, 'pass_record_a28', passed)
    if bad == 'a28_set_aside':
        # Isolate the pre-score guard from the consumer's later redundant validation.
        mp.setattr(mod.AR, 'load_pass', lambda *a: {})
    real_sens = mod.ST.sensitivity_record
    sensitivities = []
    def sensitive(sig, m_p, N, design):
        events.append('sensitivity')
        sensitivities.append((m_p.copy(), N, design))
        return real_sens(sig, m_p, N, design)
    mp.setattr(mod.ST, 'sensitivity_record', sensitive)
    if bad == 'a28_h8_hashes':
        hs = {s: hashes(s) for s in spec.seeds}
        parts = {s: mod.a28_episode_record(eps_of(mod, s)) for s in spec.seeds}
        hs[55][mod.R.PAIR_NAMES[0]] = 'd'*64
        spec.reference = NS(hashes=hs, records=parts)
    if bad == 'a28_rerun_parts':
        ep = eps_of(mod, 55)
        ep.part_sha[mod.R.PAIR_NAMES[0]]['Hindi'] = 'd'*64
        spec.attempts = [{'attempt': 1, 'flags': {}, 'episodes_sha256': {},
                          'a28_episodes': {'55': mod.a28_episode_record(ep)}}]
        put(c.res / spec.names.started, {'rule_sha256': mod.R.RULE_SHA256, 'attempts': spec.attempts})
    return spec, env, events, scores, passes, sensitivities


def scenario(guard, mod, c, mp):
    if guard in ('ledger_a28_sha', 'a28_stage3', 'a28_h8_verdict', 'a28_h8_started'):
        return preflight(mod, c, mp, 'H9' if guard in ('a28_h8_verdict', 'a28_h8_started') else 'H8', guard)
    spec, env, *_ = stub_a28(mod, c, mp, bad=guard)
    return mod.run_read(spec, c.res, env=env)


@pytest.mark.parametrize('guard', GUARDS)
def test_a28_guard(guard, case, monkeypatch):
    exc, regex = GUARDS[guard]
    with pytest.raises(exc, match=regex):
        scenario(guard, U, case, monkeypatch)


@pytest.mark.parametrize('guard', GUARDS)
def test_a28_guard_mutant(guard, case, monkeypatch, tmp_path):
    mod = mutant(tmp_path, guard)
    assert scenario(guard, mod, case, monkeypatch) == 0


def test_a28_guard_table_covers_all_new_markers():
    from test_r9_held import GUARDS as HELD_GUARDS, HERE
    from test_r9_preread import HELD_GUARDS as PREREAD_GUARDS
    markers = re.findall(r'# guard:(\w+)', (HERE / 'run_r9_held.py').read_text())
    assert len(markers) == len(set(markers))
    assert set(markers) - set(HELD_GUARDS) - PREREAD_GUARDS == set(GUARDS)


@pytest.mark.parametrize('read', ['H8', 'H9', 'H10'])
@pytest.mark.parametrize('variant', ['held', 'fix1', 'reserve'])
def test_a28_order_and_producer_calls(case, monkeypatch, capsys, read, variant):
    spec, env, events, scores, passes, sensitivities = stub_a28(U, case, monkeypatch, read, variant)
    assert U.run_read(spec, case.res, env=env) == 0
    assert events == ['inputs', 'pins', 'a28_pins', 'picks', 'cells', 'builder_seeds',
                      *[f'build:{s}' for s in spec.seeds],
                      *[e for s in spec.seeds for e in (f'context:{s}', f'bundle:{s}')],
                      'sensitivity', *[f'score:{s}' for s in spec.seeds], 'a28_pass']
    assert len(scores) == 3
    for kw in scores:
        assert kw == dict(include_pm=False, secondary=False, cells={'aff': {'fused': (1, 2), 'cf': (3, 4)}},
                          taus=(0., 1., 2., 3.), verdict=case.res / spec.names.verdict)
    assert len(passes) == 1
    mode, seeds, kw = passes[0]
    assert mode == 'a28' and seeds == [55, 56, 57]
    assert kw == dict(set_aside=[dict(language='Hindi', pair=U.R.PAIR_NAMES[0], seed=s) for s in seeds],
                      flagged_languages=['Urdu'], near_duplicates={'cos_099': 3, 'cos_098': 7})
    assert all(type(x['seed']) is int and type(x['language']) is str for x in kw['set_aside'])
    assert all(type(x) is int for x in kw['near_duplicates'].values())
    assert len(sensitivities) == 1
    m_p, N, design = sensitivities[0]
    assert np.array_equal(m_p, [6, 3]) and N == 9 and design == 'two_level'
    att = U.read_attempts(case.res / spec.names.started)[-1]
    assert set(att['a28_episodes']) == {'55', '56', '57'}
    assert len(att['a28_episodes']['55']['part_sha'][U.R.PAIR_NAMES[0]]) == 27
    loaded = U.AR.load_pass((case.res / spec.names.pass_).read_bytes(), spec.names.pass_, read, False)
    assert loaded['design'] == 'two_level' and 'secondary' not in loaded
    assert capsys.readouterr().out == f'held pass written {spec.names.pass_} {U.R.sha256_file(case.res / spec.names.pass_)}\n'
    assert not (case.res / spec.names.verdict).exists()
    with np.load(case.res / spec.names.arrays) as z:
        assert not any('share' in k or 'rate' in k or k.startswith(('B1__', 'r1_fused__')) for k in z.files)


def test_three_set_asides_stop_before_any_score(case, monkeypatch):
    spec, env, events, scores, passes, _ = stub_a28(U, case, monkeypatch, bad='set_aside_stop')
    with pytest.raises(U.AE.SetAsideStop, match='three languages'):
        U.run_read(spec, case.res, env=env)
    assert scores == passes == []
    assert 'sensitivity' not in events
    assert not any(e.startswith(('context:', 'bundle:')) for e in events)
    assert not (case.res / spec.names.pass_).exists()



@pytest.mark.parametrize('read', ['H8', 'H9', 'H10'])
@pytest.mark.parametrize('prior', ['H6', 'H7'])
@pytest.mark.parametrize('bad', ['empty', 'null', 'mismatched_status', 'wrong_read', 'wrong_rule', 'wrong_mode', 'wrong_backbone'])
def test_stage3_requires_a_completed_verdict(case, monkeypatch, read, prior, bad):
    preflight(U, case, monkeypatch, read)
    path = case.res / U.read_names(prior).verdict
    v = json.loads(path.read_text())
    if bad == 'empty':
        v = {}
    elif bad == 'null':
        v['verdict'] = None
    elif bad == 'mismatched_status':
        v.update(verdict=None, status='dropped under the rule')
    else:
        v['rule_sha256' if bad == 'wrong_rule' else bad.removeprefix('wrong_')] = 'wrong'
    put(path, v)
    with pytest.raises(U.Refused, match='H6.*H7'):
        U.a28_order_guard(case.res, case.ledger, read)


@pytest.mark.parametrize('status', ['not run: switched off', 'dropped under the rule'])
@pytest.mark.parametrize('prior', ['H6', 'H7'])
def test_stage3_null_verdict_matches_exact_ledger_status(case, monkeypatch, status, prior):
    preflight(U, case, monkeypatch)
    verdict(U, case, prior, status=status)
    rows = case.ledger.read_text().splitlines()
    case.ledger.write_text('\n'.join(row.replace('| GO |', f'| {status} |') if row.startswith(f'| {prior} |') else row
                                    for row in rows) + '\n')
    U.a28_order_guard(case.res, case.ledger, 'H8')
    case.ledger.write_text(case.ledger.read_text().replace(f'| {status} |', f'| quoted {status} |'))
    with pytest.raises(U.Refused, match='H6.*H7'):
        U.a28_order_guard(case.res, case.ledger, 'H8')


@pytest.mark.parametrize('read', ['H9', 'H10'])
@pytest.mark.parametrize('value', [{}, {'verdict': None}, {'verdict': 'GO'}, {'verdict': 'small claim', 'read': 'H9'}])
def test_h8_order_refuses_a_non_verdict(case, monkeypatch, read, value):
    preflight(U, case, monkeypatch, read)
    v = json.loads((case.res / U.read_names('H8').verdict).read_text())
    v.update(value)
    if not value:
        v = {}
    put(case.res / U.read_names('H8').verdict, v)
    with pytest.raises(U.Refused, match='H8 real verdict'):
        U.a28_order_guard(case.res, case.ledger, read)


@pytest.mark.parametrize('read', ['H9', 'H10'])
@pytest.mark.parametrize('changed', ['pooled', 'part'])
def test_rebuilt_h8_hashes_checked_before_scores(case, monkeypatch, read, changed):
    rebuilt_reference_case(U, case, monkeypatch, read, changed)


def rebuilt_reference_case(mod, c, mp, read, changed):
    spec, env, _, scores, passes, _ = stub_a28(mod, c, mp, read)
    records = {}
    for s in spec.seeds:
        ep = eps_of(mod, s)
        ep.set_aside[mod.R.PAIR_NAMES[0]] = ['Hindi']
        records[s] = mod.a28_episode_record(ep)
    spec.reference = NS(hashes={s: hashes(s) for s in spec.seeds}, records=records)
    if changed == 'pooled':
        spec.reference.hashes[55][mod.R.PAIR_NAMES[0]] = 'd'*64
    else:
        spec.reference.records[55]['part_sha'][mod.R.PAIR_NAMES[0]]['Hindi'] = 'd'*64
    try:
        mod.run_read(spec, c.res, env=env)
    except mod.Refused as exc:
        assert 'differ from H8' in str(exc)
        assert scores == passes == []
        return
    pytest.fail(f'{changed} hash change reached scores')


@pytest.mark.parametrize('changed,old,new', [
    ('pooled', 'dict(eps.sha) == self.reference.hashes.get(s)', 'True'),
    ('part', 'part == self.reference.records.get(s)', 'True'),
])
def test_each_h8_hash_comparison_mutant(case, monkeypatch, tmp_path, changed, old, new):
    mod = mutant(tmp_path, replacement=(old, new))
    with pytest.raises(pytest.fail.Exception, match='hash change reached scores'):
        rebuilt_reference_case(mod, case, monkeypatch, 'H10', changed)


@pytest.mark.parametrize('pair', range(3))
@pytest.mark.parametrize('column', [0, 1])
def test_both_anchor_label_joins_before_context(case, monkeypatch, pair, column):
    label_case(U, case, monkeypatch, pair, column)


def label_case(mod, c, mp, pair, column):
    spec, env, events, scores, _, _ = stub_a28(mod, c, mp)
    build = mod.AE.build_seed_a28
    def shifted(*args):
        ep = build(*args)
        ep.per_pair[pair].candidates[0, column] = 1  # same-language shift; labels differ at position 1
        return ep
    mp.setattr(mod.AE, 'build_seed_a28', shifted)
    try:
        mod.run_read(spec, c.res, env=env)
    except AssertionError as exc:
        assert 'row space differs' in str(exc)
        assert not any(e.startswith('context:') for e in events)
        assert scores == []
        return
    pytest.fail('shifted labels reached context')


@pytest.mark.parametrize('column', [0, 1])
def test_each_label_clause_mutant(case, monkeypatch, tmp_path, column):
    old = ('np.array_equal(getattr(a28, a)[ep.candidates[:, 0]], getattr(a28, a)[ep.anchor])' if column == 0 else
           'np.array_equal(getattr(a28, b)[ep.candidates[:, 1]], getattr(a28, b)[ep.anchor])')
    mod = mutant(tmp_path, replacement=(old, 'True'))
    with pytest.raises(pytest.fail.Exception, match='shifted labels reached context'):
        label_case(mod, case, monkeypatch, 0, column)


@pytest.mark.parametrize('site', ['eps', 'bundle'])
@pytest.mark.parametrize('field', ['language', 'part_sha', 'shares', 'set_aside'])
def test_all_episode_and_bundle_records_required(case, monkeypatch, site, field):
    spec, env, _, scores, _, _ = stub_a28(U, case, monkeypatch)
    target = U.AE if site == 'eps' else U.B
    name = 'build_seed_a28' if site == 'eps' else 'build_bundle_a28'
    original = getattr(target, name)
    def lost_record(*args):
        obj = original(*args)
        setattr(obj, field, None)
        return obj
    monkeypatch.setattr(target, name, lost_record)
    with pytest.raises(AssertionError, match='records must exist'):
        U.run_read(spec, case.res, env=env)
    assert scores == []


@pytest.mark.parametrize('old,new,bad', [
    ('        a28_order_guard(res, ledger, read)', '        pass', 'a28_stage3'),
    ("reference=a28_reference(results) if read in ('H9', 'H10') else None", 'reference=None', 'a28_h8_started'),
    ('    a28_label_guard(data.a28, eps)', '    pass', 'a28_label_join'),
    ('    a28_records_guard(eps)', '    pass', 'eps_records'),
    ('    a28_records_guard(bundle)', '    pass', 'bundle_records'),
    ('rec = A28EpisodeRecorder(', 'rec = EpisodeRecorder(', 'a28_h8_hashes'),
])
def test_a28_guard_call_removal_detected(case, monkeypatch, tmp_path, old, new, bad):
    if bad == 'a28_h8_hashes':
        mod = mutant(tmp_path, replacement=(
            'rec = A28EpisodeRecorder(started, spec.copy_to, k, spec.want, recorded, out, spec.overwrite,\n'
            '                                 read=spec.read, reference=spec.reference)',
            'rec = EpisodeRecorder(started, spec.copy_to, k, spec.want, recorded, out, spec.overwrite,\n'
            '                              read=spec.read, h5=None)'))
        spec, env, *_ = stub_a28(mod, case, monkeypatch)
        # The context tripwire requires the recorder's A28 metadata before cosine/posteriors.
        with pytest.raises(KeyError, match='a28_episodes'):
            mod.run_read(spec, case.res, env=env)
        return
    mod = mutant(tmp_path, replacement=(old, new))
    if bad in ('eps_records', 'bundle_records'):
        site = bad.removesuffix('_records')
        spec, env, _, scores, _, _ = stub_a28(mod, case, monkeypatch, bad='a28_records', record_bad=site)
        if site == 'eps':
            original = mod.B.build_bundle_a28
            def repaired(*a):
                bundle = original(*a)
                bundle.shares = {}  # the later boundary cannot mask the missing episode call
                return bundle
            monkeypatch.setattr(mod.B, 'build_bundle_a28', repaired)
        assert mod.run_read(spec, case.res, env=env) == 0
        assert len(scores) == 3
    else:
        assert scenario(bad, mod, case, monkeypatch) == 0


@pytest.mark.parametrize('name', ['started_guard', 'sensitivity_guard', 'head_guard'])
def test_a28_reuses_shared_guard_call_sites(case, monkeypatch, name):
    spec, env, *_ = stub_a28(U, case, monkeypatch)
    original = getattr(U, name)
    calls = []
    def spy(*a):
        calls.append(name)
        return original(*a)
    monkeypatch.setattr(U, name, spy)
    assert U.run_read(spec, case.res, env=env) == 0
    assert calls == [name]


def h5_collision_case(mod, c, mp):
    spec, env, _, scores, _, _ = stub_a28(mod, c, mp)
    spec.h5 = {52: hashes(55)}
    try:
        mod.run_read(spec, c.res, env=env)
    except AssertionError as exc:
        assert 'episode hashes equal recorded ones' in str(exc)
        assert scores == []
        return
    pytest.fail('H5 hash collision reached scores')


def test_a28_hashes_differ_from_h5(case, monkeypatch):
    h5_collision_case(U, case, monkeypatch)


def test_a28_h5_binding_mutant(case, monkeypatch, tmp_path):
    mod = mutant(tmp_path, replacement=('        recorded = {**recorded, **spec.h5}', '        pass'))
    with pytest.raises(pytest.fail.Exception, match='H5 hash collision reached scores'):
        h5_collision_case(mod, case, monkeypatch)


def test_smoke_builds_selection_space_and_checks_real_inputs_without_episodes(case, monkeypatch):
    import pandas as pd
    rows = np.arange(2, 110, 2, dtype=np.int64)  # global row IDs; different from positions k
    n = 112
    real = object()
    env = NS(heads=object(), split=NS(selection=rows),
             data=NS(paintings=np.array([f'p{i // 3:03}' for i in range(n)]),
                     img_features=np.ones((n, 2), dtype=np.float32), txt_features=np.ones((n, 2), dtype=np.float32)),
             labels={a: np.arange(n) % 3 for a in U.R.ASPECTS})
    values = {a: [{'code': 0, 'name': 'zero'}] for a in U.R.ASPECTS}
    vp, fp = case.res / 'value_sets_a28.json', case.res / 'language_flags_a28.json'
    put(vp, values)
    put(fp, {'flagged': ['Urdu']})
    put(case.res / 'near_duplicates_a28.json', {'cos_099': 3, 'cos_098': 7, 'cos_092': 20})
    monkeypatch.setattr(U, 'NEAR_DUPLICATES_SHA256', U.R.sha256_file(case.res / 'near_duplicates_a28.json'))  # the pin
    pins = case.res / 'pins_a28.json'
    put(pins, {'files': {'results/value_sets_a28.json': {'path': str(vp)},
                         'results/language_flags_a28.json': {'path': str(fp)}}})
    monkeypatch.setattr(U.AD, 'PINS_PATH', pins)
    monkeypatch.setattr(U.AD, 'load_a28', lambda b: real)  # default pins API; no custom pins argument
    monkeypatch.setattr(U.AD, 'results_dir', lambda: case.res)
    calls = []
    def inputs(a28, heads):
        assert a28 is real and heads is env.heads
        calls.append('real_inputs')
        return 'inputs-only'
    monkeypatch.setattr(U.AC, 'A28Inputs', inputs)
    def model(inp, backbone):
        assert inp == 'inputs-only' and backbone == 'clip_b32'
        calls.append('codes_masks')
    monkeypatch.setattr(U.B, 'model_inputs', model)
    prepared = []
    def prepare(tab):
        assert isinstance(tab, pd.DataFrame)
        prepared.append(tab)
        return 'prepared'
    monkeypatch.setattr(U.AE, 'prepare', prepare)
    def forbidden(*a, **kw):
        pytest.fail('smoke must not build or score an episode on real A28 rows')
    monkeypatch.setattr(U.AE, 'build_seed_a28', forbidden)
    monkeypatch.setattr(U.AC, 'A28Context', forbidden)
    monkeypatch.setattr(U.S, 'score_seed', forbidden)
    out = U.setup_a28(env, 'clip_b32', True)
    assert calls == ['real_inputs', 'codes_masks']
    assert out.prepared == 'prepared' and out.values == {a: [0] for a in U.R.ASPECTS}
    assert np.array_equal(out.a28.row_id, rows)
    assert np.array_equal(out.a28.language, rows % 27 + 1)
    assert np.array_equal(out.a28.painting_index, np.unique(out.a28.painting, return_inverse=True)[1])
    assert prepared[0]['language'].tolist() == [U.AE.LANGUAGES[r % 27] for r in rows]
    for a in U.R.ASPECTS:
        assert np.array_equal(getattr(out.a28, a), env.labels[a][rows])
    assert out.near_duplicates == {'cos_099': 3, 'cos_098': 7}
    # LOW-5 (ticket 21c): near_duplicates_a28.json is in no pin, so this source SHA is its only trace
    path = case.res / 'near_duplicates_a28.json'
    assert out.near_source == {'file': str(path), 'sha256': U.R.sha256_file(path)}
    assert out.flagged_languages == ['Urdu'] and out.value_record == values
    # B1 LOW-2: the file is read pinned. An edited file (same counts) is refused before anything is taken from it.
    put(path, {'cos_099': 3, 'cos_098': 7, 'cos_092': 21})
    with pytest.raises(AssertionError, match='near_duplicates_a28.json'):
        U.setup_a28(env, 'clip_b32', True)


def test_near_duplicates_pin_equals_the_comparers():
    import hashlib
    rederive = (U.HERE / 'rederive/rd9_inputs.py').read_text()
    assert re.search(r"NEAR_DUPLICATES_SHA256\s*=\s*'([0-9a-f]{64})'", rederive).group(1) == U.NEAR_DUPLICATES_SHA256
    real = U.R.MAIN / 'src/test/20261202_r9_scope/results/near_duplicates_a28.json'
    if real.exists():    # the stage-1 file exists only in the integration checkout
        assert hashlib.sha256(real.read_bytes()).hexdigest() == U.NEAR_DUPLICATES_SHA256


@pytest.mark.parametrize('read', ['H8', 'H9', 'H10'])
def test_smoke_uses_a28_seeds_mode_and_no_reference(case, monkeypatch, read):
    def run(spec, results, here, env):
        assert spec.seeds == (9001, 9002, 9003) and spec.n_per_pair == 64
        assert spec.mode == 'smoke' and spec.ctx_mode == 'a28'
        assert spec.reference is None and spec.h5 == {} and spec.overwrite is True
        assert spec.out == case.res / 'smoke' / 't21'
        return 0
    monkeypatch.setattr(U, 'run_read', run)
    assert U.run_smoke('clip_b32', read, results=case.res, subdir='t21') == 0


@pytest.mark.parametrize('read', ['H8', 'H9', 'H10'])
def test_first_and_reserve_reads_obey_amsterdam_deadline(case, monkeypatch, read):
    from datetime import date
    preflight(U, case, monkeypatch, read)
    monkeypatch.setattr(U, 'amsterdam_today', lambda: date(2026, 10, 19))
    for variant in ('held', 'reserve'):
        if variant == 'reserve':
            case.ledger.write_text(case.ledger.read_text().replace(f'| {read} |', f'| {read}-R |'))
        with pytest.raises(U.Refused, match='Amsterdam date'):
            U.refuse_or_go(case.res, case.ledger, read, U.READ_BACKBONES[read], variant, folder=case.folder)


@pytest.mark.parametrize('read', ['H8', 'H9', 'H10'])
@pytest.mark.parametrize('variant', ['after_crash', 'fix1', 'reserve'])
def test_a28_continuations_use_all_nine_a28_hashes(case, monkeypatch, read, variant):
    preflight(U, case, monkeypatch, read)
    base = U.read_names(read)
    hs = {str(s): hashes(s) for s in (55, 56, 57)}
    att = {'attempt': 1, 'flags': {}, 'episodes_sha256': hs,
           'a28_episodes': {str(s): U.a28_episode_record(eps_of(U, s)) for s in (55, 56, 57)}}
    started = {'rule_sha256': U.R.RULE_SHA256, 'attempts': [att]}
    put(case.res / base.started, started)
    put(case.folder / base.started, started)
    if variant == 'reserve':
        verdict(U, case, read)
        case.ledger.write_text(case.ledger.read_text().replace(f'| {read} |', f'| {read}-R |'))
    elif variant == 'fix1':
        put(case.res / base.pass_, {})
        put(case.res / 'smoke_record_heldfix1.json', {})
        case.ledger.write_text(case.ledger.read_text().replace('| (pending) |',
                              '| ' + '; '.join(h for hsp in hs.values() for h in hsp.values()) + ' |', 1))
    seen = []
    def read_spec(spec, *args):
        seen.append(spec)
        assert spec.mode == spec.ctx_mode == 'a28'
        assert spec.seeds == (55, 56, 57)
        assert spec.want == {int(s): h for s, h in hs.items()}
        return 0
    monkeypatch.setattr(U, 'run_read', read_spec)
    assert U.run_held(U.READ_BACKBONES[read], read, after_crash=variant == 'after_crash',
                      fix=1 if variant == 'fix1' else None, reserve=variant == 'reserve',
                      results=case.res, ledger=case.ledger, folder=case.folder) == 0
    assert len(seen) == 1


def test_h8_reference_recovers_complete_parts_and_hashes(case):
    att = {'attempt': 1, 'flags': {}, 'episodes_sha256': {str(s): hashes(s) for s in (55, 56, 57)},
           'a28_episodes': {str(s): U.a28_episode_record(eps_of(U, s)) for s in (55, 56, 57)}}
    put(case.res / U.read_names('H8').started, {'rule_sha256': U.R.RULE_SHA256, 'attempts': [att]})
    ref = U.a28_reference(case.res)
    assert ref.hashes == {s: hashes(s) for s in (55, 56, 57)}
    assert ref.records == {s: U.a28_episode_record(eps_of(U, s)) for s in (55, 56, 57)}


@pytest.mark.parametrize('bad', [None, 'a28_pass_shape'])
def test_consumer_checks_exact_pass_bytes_before_write(case, monkeypatch, bad):
    spec, env, _, _, _, _ = stub_a28(U, case, monkeypatch, bad=bad)
    load = U.AR.load_pass
    seen = []
    def validate(raw, name, read, smoke):
        assert not (case.res / spec.names.pass_).exists()
        assert (case.res / spec.names.arrays).is_file() and (case.res / spec.names.sensitivity).is_file()
        assert name == spec.names.pass_ and read == 'H8' and smoke is False
        seen.append(raw)
        return load(raw, name, read, smoke)
    monkeypatch.setattr(U.AR, 'load_pass', validate)
    if bad:
        with pytest.raises(U.AR.Stop, match='near_duplicates'):
            U.run_read(spec, case.res, env=env)
        assert not (case.res / spec.names.pass_).exists()
    else:
        assert U.run_read(spec, case.res, env=env) == 0
        assert (case.res / spec.names.pass_).read_bytes() == seen[0]
    assert len(seen) == 1


def test_a28_smoke_uses_two_level_writers_and_smoke_consumer(case, monkeypatch):
    spec, env, _, scores, passes, sensitivities = stub_a28(U, case, monkeypatch, variant='smoke')
    spec.mode, spec.seeds, spec.overwrite = 'smoke', (9001, 9002, 9003), True
    monkeypatch.setattr(U.R, 'N_SMOKE', 1)
    monkeypatch.setattr(U.ST.R6C, 'N_SMOKE', 1)
    assert U.run_read(spec, case.res, env=env) == 0
    assert passes[0][0:2] == ('smoke', [9001, 9002, 9003])
    assert sensitivities[0][2] == 'two_level'
    assert all(kw['verdict'] == case.res / spec.names.verdict for kw in scores)
    U.AR.load_pass((case.res / spec.names.pass_).read_bytes(), spec.names.pass_, 'H8', True)


# ---------------------------------------------------------------------------------------------------------------
# Ticket 21c: the LOW test gaps of the check of ticket 21 (review_t21.md LOW-1 to LOW-5 and LOW-7).
# ---------------------------------------------------------------------------------------------------------------
def sha_of(mod, path):
    return mod.R.sha256_file(path)


def a28_row(mod, c, mp, row, label_sha='good', label=True):
    """One ledger row ``row`` with the runner and pins labels and, for ``label``, a pins_a28.json label."""
    configure(mod, c, mp)
    base = row.removesuffix('-R')
    backbone = mod.READ_BACKBONES[base]
    from test_r9_held import ledger
    c.a28pins = c.res / 'pins_a28.json'
    put(c.a28pins, {'files': {}})
    mp.setattr(mod.AD, 'PINS_PATH', c.a28pins)
    scripts = f"run_r9_held.py {'a' * 64}; {c.pins.name} {sha_of(mod, c.pins)}"
    if label:
        shown = sha_of(mod, c.a28pins) if label_sha == 'good' else label_sha
        scripts += f"; pins_a28.json {shown}"
    ledger(c, row=row, backbone=mod.LEDGER_NAMES[backbone], scripts=scripts)
    return backbone


def guard_row(mod, c, row, backbone):
    return mod.ledger_guard(c.ledger, row, backbone, 'a' * 64, sha_of(mod, c.pins))


# LOW-1: the pins_a28.json SHA check of the ledger row, for every A28 read and its reserve row.
A28_ROWS = ['H8', 'H9', 'H10', 'H8-R', 'H9-R', 'H10-R']


@pytest.mark.parametrize('row', A28_ROWS)
def test_ledger_a28_sha_good_row_passes(case, monkeypatch, row):
    backbone = a28_row(U, case, monkeypatch, row)
    assert guard_row(U, case, row, backbone)['row'] == row


@pytest.mark.parametrize('row', A28_ROWS)
@pytest.mark.parametrize('bad', ['wrong_sha', 'no_label', 'missing_file', 'stale_last'])
def test_ledger_a28_sha_refuses_every_a28_row(case, monkeypatch, row, bad):
    backbone = a28_row(U, case, monkeypatch, row, label_sha={'wrong_sha': 'b' * 64, 'no_label': 'x',
                                                             'missing_file': 'c' * 64,
                                                             'stale_last': 'good'}[bad], label=bad != 'no_label')
    if bad == 'missing_file':
        monkeypatch.setattr(U.AD, 'PINS_PATH', case.res / 'absent_pins_a28.json')
    if bad == 'stale_last':  # the last labelled entry wins: a stale entry after the good one refuses
        text = case.ledger.read_text().replace(f"{sha_of(U, case.a28pins)} |",
                                               f"{sha_of(U, case.a28pins)}; pins_a28.json {'d' * 64} |")
        case.ledger.write_text(text)
    with pytest.raises(U.Refused, match='pins_a28.json SHA differs'):
        guard_row(U, case, row, backbone)


@pytest.mark.parametrize('row', A28_ROWS)
def test_ledger_a28_sha_last_entry_wins(case, monkeypatch, row):
    backbone = a28_row(U, case, monkeypatch, row)
    good = sha_of(U, case.a28pins)
    text = case.ledger.read_text().replace(f"pins_a28.json {good}", f"pins_a28.json {'d' * 64}; pins_a28.json {good}")
    assert text != case.ledger.read_text()
    case.ledger.write_text(text)
    assert guard_row(U, case, row, backbone)['row'] == row  # a stale first entry, a good last entry


@pytest.mark.parametrize('row', ['H6', 'H7', 'H6-R', 'H7-R'])
def test_ledger_a28_sha_not_asked_of_held_rows(case, monkeypatch, row):
    configure(U, case, monkeypatch)
    from test_r9_held import ledger
    backbone = U.READ_BACKBONES[row.removesuffix('-R')]
    ledger(case, row=row, backbone=U.LEDGER_NAMES[backbone])
    monkeypatch.setattr(U.AD, 'PINS_PATH', case.res / 'absent_pins_a28.json')
    assert guard_row(U, case, row, backbone)['row'] == row


# LOW-2: the A28 order guard runs for the reserve, after-crash and fix1 variants too.
def continuation(mod, c, mp, read, variant):
    preflight(mod, c, mp, read)
    base = mod.read_names(read)
    hs = {str(s): hashes(s) for s in (55, 56, 57)}
    att = {'attempt': 1, 'flags': {}, 'episodes_sha256': hs,
           'a28_episodes': {str(s): mod.a28_episode_record(eps_of(mod, s)) for s in (55, 56, 57)}}
    started = {'rule_sha256': mod.R.RULE_SHA256, 'attempts': [att]}
    put(c.res / base.started, started)
    put(c.folder / base.started, started)
    if variant == 'reserve':
        verdict(mod, c, read)
        c.ledger.write_text(c.ledger.read_text().replace(f'| {read} |', f'| {read}-R |'))
    elif variant == 'fix1':
        put(c.res / base.pass_, {})
        put(c.res / 'smoke_record_heldfix1.json', {})
        c.ledger.write_text(c.ledger.read_text().replace('| (pending) |',
                            '| ' + '; '.join(h for hsp in hs.values() for h in hsp.values()) + ' |', 1))


def go_continuation(mod, c, read, variant):
    return mod.refuse_or_go(c.res, c.ledger, read, mod.READ_BACKBONES[read],
                            'held' if variant == 'after_crash' else variant, variant == 'after_crash', c.folder)


@pytest.mark.parametrize('variant', ['after_crash', 'fix1', 'reserve'])
@pytest.mark.parametrize('read,missing', [('H8', 'H6'), ('H8', 'H7'), ('H9', 'H6'), ('H9', 'H7'), ('H9', 'H8'),
                                          ('H10', 'H6'), ('H10', 'H7'), ('H10', 'H8')])
def test_order_guard_runs_for_continuations(case, monkeypatch, read, missing, variant):
    continuation(U, case, monkeypatch, read, variant)
    assert go_continuation(U, case, read, variant).names.read == read  # every verdict present: the variant is fine
    (case.res / U.read_names(missing).verdict).unlink()
    with pytest.raises(U.Refused, match='H8 real verdict' if missing == 'H8' else 'H6.*H7'):
        go_continuation(U, case, read, variant)


# LOW-3: prior_verdict, branch by branch.
def edit_row(c, read, **cells):
    out = []
    for line in c.ledger.read_text().splitlines():
        if line.startswith(f'| {read} |'):
            parts = [p.strip() for p in line.strip()[1:-1].split('|')]
            for name, value in cells.items():
                parts[U.LEDGER_CELLS.index(name)] = value
            line = '| ' + ' | '.join(parts) + ' |'
        out.append(line)
    c.ledger.write_text('\n'.join(out) + '\n')


def not_run_case(c, mp, read, status='not run: switched off'):
    preflight(U, c, mp, 'H8')
    edit_row(c, read, Report=status)
    verdict(U, c, read, status=status)


def pv(c, read, allow=True):
    return U.prior_verdict(c.res, c.ledger, read, allow)


@pytest.mark.parametrize('status', ['not run: switched off', 'dropped under the rule'])
@pytest.mark.parametrize('read', ['H6', 'H7', 'H8'])
def test_prior_verdict_null_needs_not_run_allowance_and_wording(case, monkeypatch, read, status):
    not_run_case(case, monkeypatch, read, status)
    assert pv(case, read, True) is True
    assert pv(case, read, False) is False  # H8 as the reference of H9 and H10 is never a null verdict


@pytest.mark.parametrize('read', ['H9', 'H10'])
def test_h8_null_verdict_does_not_open_h9_or_h10(case, monkeypatch, read):
    preflight(U, case, monkeypatch, read)
    with case.ledger.open('a') as f:
        f.write(f'| H8 | date | held | {U.LEDGER_NAMES["clip_b32"]} | AFF | pending | script | '
                f'not run: switched off |\n')
    verdict(U, case, 'H8', status='not run: switched off')
    assert pv(case, 'H8', True) is True  # the same file and row would open stage 3
    with pytest.raises(U.Refused, match='H8 real verdict'):
        U.a28_order_guard(case.res, case.ledger, read)


@pytest.mark.parametrize('read', ['H6', 'H7'])
def test_prior_verdict_file_without_verdict_key(case, monkeypatch, read):
    not_run_case(case, monkeypatch, read)
    assert pv(case, read) is True
    path = case.res / U.read_names(read).verdict
    v = json.loads(path.read_text())
    del v['verdict']
    put(path, v)
    assert pv(case, read) is False


@pytest.mark.parametrize('read', ['H6', 'H7'])
@pytest.mark.parametrize('status,other', [('not run: switched off', 'dropped under the rule'),
                                          ('dropped under the rule', 'not run: switched off')])
@pytest.mark.parametrize('cell', ['Script SHA-256', 'Purpose'])
def test_prior_verdict_other_wording_in_another_cell(case, monkeypatch, read, status, other, cell):
    not_run_case(case, monkeypatch, read, status)
    assert pv(case, read) is True
    edit_row(case, read, **{cell: other})
    assert pv(case, read) is False


@pytest.mark.parametrize('read', ['H6', 'H7'])
def test_prior_verdict_status_must_be_a_rule_wording(case, monkeypatch, read):
    not_run_case(case, monkeypatch, read, 'not run: because')  # verdict file and ledger agree, wording is not the rule's
    assert pv(case, read) is False


@pytest.mark.parametrize('read', ['H6', 'H7'])
def test_prior_verdict_null_with_a_missing_ledger_row(case, monkeypatch, read):
    not_run_case(case, monkeypatch, read)
    assert pv(case, read) is True
    case.ledger.write_text(''.join(line for line in case.ledger.read_text().splitlines(keepends=True)
                                   if not line.startswith(f'| {read} |')))
    assert pv(case, read) is False
    with pytest.raises(U.Refused, match='H6.*H7'):
        U.a28_order_guard(case.res, case.ledger, 'H8')


@pytest.mark.parametrize('read', ['H6', 'H7', 'H8'])
@pytest.mark.parametrize('content', ['[]', '"GO"', 'null', '3', '[{"verdict": "GO"}]'])
def test_prior_verdict_non_dict_file_is_not_a_verdict(case, monkeypatch, read, content):
    preflight(U, case, monkeypatch, 'H8')
    (case.res / U.read_names(read).verdict).write_text(content)
    assert pv(case, read) is False
    assert pv(case, read, False) is False


@pytest.mark.parametrize('read,word,ok', [
    ('H6', 'GO', True), ('H6', 'NO-GO', True), ('H6', 'full claim', False), ('H6', 'small claim', False),
    ('H7', 'GO', True), ('H7', 'NO-GO', True), ('H7', 'full claim', False), ('H7', 'small claim', False),
    ('H8', 'GO', False), ('H8', 'NO-GO', True), ('H8', 'full claim', True), ('H8', 'small claim', True)])
def test_prior_verdict_words_of_each_stage(case, monkeypatch, read, word, ok):
    preflight(U, case, monkeypatch, 'H8')
    verdict(U, case, read, word)
    assert pv(case, read, False) is ok


# LOW-4: a28_reference, each clause alone.
def h8_started(c, mutate=None):
    hs = {str(s): hashes(s) for s in (55, 56, 57)}
    records = {str(s): U.a28_episode_record(eps_of(U, s)) for s in (55, 56, 57)}
    att = {'attempt': 1, 'flags': {}, 'episodes_sha256': hs, 'a28_episodes': records}
    if mutate:
        mutate(att)
    put(c.res / U.read_names('H8').started, {'rule_sha256': U.R.RULE_SHA256, 'attempts': [att]})


def test_a28_reference_control_is_complete(case):
    h8_started(case)
    assert set(U.a28_reference(case.res).records) == {55, 56, 57}


REFERENCE_SHORT = {
    'hash_seed_missing': lambda a: a['episodes_sha256'].pop('57'),
    'record_seed_missing': lambda a: a['a28_episodes'].pop('57'),
    'no_records_at_all': lambda a: a.pop('a28_episodes'),
    'record_not_a_dict': lambda a: a['a28_episodes'].update({'56': ['part_sha']}),
    **{f'{k}_missing': (lambda k: lambda a: a['a28_episodes']['55'].pop(k))(k)
       for k in ('part_sha', 'shares', 'set_aside', 'language')},
    **{f'{k}_not_a_dict': (lambda k: lambda a: a['a28_episodes']['55'].update({k: list(U.R.PAIR_NAMES)}))(k)
       for k in ('part_sha', 'shares', 'set_aside', 'language')},
    **{f'{k}_pair_missing': (lambda k: lambda a: a['a28_episodes']['56'][k].pop(U.R.PAIR_NAMES[2]))(k)
       for k in ('part_sha', 'shares', 'set_aside', 'language')},
    **{f'{k}_pair_extra': (lambda k: lambda a: a['a28_episodes']['56'][k].update({'extra__pair': {}}))(k)
       for k in ('part_sha', 'shares', 'set_aside', 'language')},
}


@pytest.mark.parametrize('short', REFERENCE_SHORT)
def test_a28_reference_refuses_each_short_file(case, short):
    h8_started(case, REFERENCE_SHORT[short])
    with pytest.raises(U.Refused, match='H8 started file needs all nine hashes and A28 part records'):
        U.a28_reference(case.res)


# LOW-5: provenance in the started and pass files.
@pytest.mark.parametrize('read', ['H8', 'H9', 'H10'])
def test_a28_started_and_pass_files_carry_provenance(case, monkeypatch, read):
    spec, env, *_ = stub_a28(U, case, monkeypatch, read)
    monkeypatch.setattr(U, 'value_sets_guard', lambda *a: {'clip': 'value sets'})  # the CLIP value sets, not A28's
    assert U.run_read(spec, case.res, env=env) == 0
    pins = sha_of(U, case.res / 'pins_a28.json')
    started = U.read_attempts(case.res / spec.names.started)[-1]
    passed = json.loads((case.res / spec.names.pass_).read_text())
    for rec in (started, passed):
        assert rec['pins_a28_sha256'] == pins
        assert rec['input_sha256'] == {'pins_a28.json': pins}
        assert rec['read_set_sha256'] == 'c' * 64
        assert rec['near_duplicates_source'] == {'file': 'near_duplicates_a28.json', 'sha256': 'e' * 64}
        assert rec['value_sets'] == {'a28': 'value record'}


def test_held_started_and_pass_files_have_no_a28_provenance(case, monkeypatch):
    spec, env, *_ = stub_read(U, case, monkeypatch, 'H6', 'held')
    assert U.run_read(spec, case.res, env=env) == 0
    started = U.read_attempts(case.res / spec.names.started)[-1]
    absent = {'pins_a28_sha256', 'read_set_sha256', 'near_duplicates_source'}
    assert not absent & set(started)
    assert 'pins_a28.json' not in started['input_sha256']


# LOW-7: READ_BACKBONES against the rule's table (section 7.1), not against the code.
RULE_LEDGER_NAMES = {'CLIP ViT-B/32': 'clip_b32', 'Qwen3-VL-Embedding-2B': 'qwen3vl_emb_2b',
                     'SigLIP 2 so400m/14-384': 'siglip2_so400m'}


def test_read_backbones_equal_the_rules_table():
    from test_r9_held import HERE
    text = (HERE / 'DECISION_RULE.md').read_text()
    rule = dict(re.findall(r'^\s*\| (H\d+) \| [^|]*\| ([^|]*?) \| [^|]*\| [^|]*\|$', text, re.M))
    assert set(rule) == {'H6', 'H7', 'H8', 'H9', 'H10'}
    assert {read: RULE_LEDGER_NAMES[name] for read, name in rule.items()} == U.READ_BACKBONES
    # and literally, so a wrong table edit is not mirrored
    assert U.READ_BACKBONES == {'H6': 'qwen3vl_emb_2b', 'H7': 'siglip2_so400m', 'H8': 'clip_b32',
                                'H9': 'qwen3vl_emb_2b', 'H10': 'siglip2_so400m'}
    assert {k: U.LEDGER_NAMES[v] for k, v in U.READ_BACKBONES.items()} == dict(rule)
