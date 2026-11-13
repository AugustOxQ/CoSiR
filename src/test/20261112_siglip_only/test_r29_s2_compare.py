"""Tests of r29_s2_compare.py on small synthetic fixtures (ours and Codex's files agree; each test breaks one thing)."""
import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r29_s2_compare as S

ASP = S.ASPECTS
ENC = S.ENCODERS
SEEDS = [0, 1, 2, 3, 4]


def w(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))
    return path


def lst(base, step=0.001):
    return [base + step * ((i * 3) % 5) for i in range(5)]


def ptp(x):
    return max(x) - min(x)


# ------------------------------------------------------------------------------------------------ val fixtures
def cells_of(enc_i):
    return {f'{m}/{a}': lst(0.1 * (enc_i + 1) + 0.01 * j, 0.001 * (j + 1))
            for j, (a, m) in enumerate((a, m) for a in ASP for m in ('ami', 'map'))}


def group_of(cells):
    per_seed = [sum(v[i] for v in cells.values()) / 6 for i in range(5)]
    return dict(per_seed=per_seed, mean=sum(per_seed) / 5)


def build_val_e2e3_sc9(rec, rd):
    cx_enc, kstar = {}, [0.30, 0.31, 0.29, 0.32, 0.30]
    for i, (E, arm) in enumerate(ENC.items()):
        cells = cells_of(i)
        g = group_of(cells)
        ours = dict(arm=arm, encoder=E,
                    ami_ranges={a: dict(k=9, per_seed={str(s): v for s, v in enumerate(cells[f'ami/{a}'])}, r=ptp(cells[f'ami/{a}']))
                                for a in ASP},
                    map_ranges={a: dict(per_seed={str(s): v for s, v in enumerate(cells[f'map/{a}'])}, r=ptp(cells[f'map/{a}']))
                                for a in ASP},
                    readouts=dict(cstar=dict(pick='center+k10', kept=True, score=5.5, map={a: 0.4 for a in ASP},
                                             table={'center+k10': dict(kept=True, score=5.5, map={a: 0.4 for a in ASP})}),
                          untold=dict(pick='center+k3', kept=True, score=3.5, map={a: 0.3 for a in ASP}),
                          plain=dict(pick='x'), teacher=dict(pick='y')))
        w(rec / f'image_picks_{arm}.json', ours)
        w(rec / f'image_groups_{arm}.json', dict(arm=arm, encoder=E, image=dict(cells=cells, per_seed=g['per_seed'], mean=g['mean'])))
        cx_enc[E] = dict(groups=dict(cells=cells, **g),
                         picks=dict(image_map={a: dict(pick_plain_side='center+k3', pick_q1_side='center+k10',
                                                       r=ptp(cells[f'map/{a}'])) for a in ASP},
                                    ami_range={a: ptp(cells[f'ami/{a}']) for a in ASP}),
                         image_tables=dict(q1=dict(pick='center+k10', table={'center+k10': {a: 0.4 for a in ASP}},
                                                   scores={'center+k10': 5.5}, kept={'center+k10': True}),
                                           plain=dict(pick='center+k3', table={'center+k3': {a: 0.3 for a in ASP}},
                                                      scores={'center+k3': 3.5}, kept={'center+k3': True})))
    per = {'siglip1': kstar, **{E: cx_enc[E]['groups']['per_seed'] for E in ENC}}
    means = {e: sum(v) / 5 for e, v in per.items()}
    ranking = sorted(means, key=lambda e: -means[e])
    lead, run = ranking[:2]
    margin = means[lead] - means[run]
    ahead = [s for s in range(5) if per[lead][s] > per[run][s]]
    rng = ptp(per[lead])
    w(rec / 'encoder_pick.json', dict(pick=lead, ranking=ranking, tie=None, margin=margin, seeds_ahead=ahead, close_call=margin < rng,
                                      encoders={**{e: dict(score=means[e], per_seed=per[e], seed_range=ptp(per[e])) for e in per},
                                                'qwen': dict(score=0.37, per_seed=[0.37] * 5, seed_range=0.0, role='information')}))
    w(rd / 's2e_val.json', dict(kind='validation', complete=True, encoders=cx_enc, replay_gates={e: True for e in ENC},
                                forward_check={E: {a: dict(**{'pass': True}) for a in ASP} for E in ENC},
                                sc9=dict(kstar_per_seed=kstar, means=means, pick=lead, runner_up=run, margin=margin, seeds_ahead=ahead,
                                         n_seeds_ahead=len(ahead), leader_range=rng, close_call=margin < rng, qwen_info=0.37),
                                notes=['seeds ahead = seeds where the leader is strictly above the runner-up', 'other']))


def mp_ranges(base):
    return {m: dict(per_seed={str(s): v for s, v in enumerate(lst(base + (0.01 if m == 'map' else 0)))}, r=ptp(lst(base)), n=10)
            for m in ('ami', 'map')}


def build_val_mmart_photo(rec, rd):
    ours_m = dict(arm=S.KSTAR, ami_ranges={}, map_ranges={}, readouts=dict(
        untold=dict(pick='none+k5', score=2.0, kept=True, map={a: 0.2 for a in ASP}, table={
            'none+k5': dict(score=2.0, kept=True, map={a: 0.2 for a in ASP})}),
        cstar=dict(pick='center+k1', score=1.0, kept=True, map={a: 0.25 for a in ASP}, table={
            'center+k1': dict(score=1.0, kept=True, map={a: 0.25 for a in ASP})}),
        plain=dict(pick='p'), teacher=dict(pick='t')))
    ranges = {}
    for j, a in enumerate(ASP):
        a_vals, m_vals = lst(0.1 + 0.01 * j), lst(0.3 + 0.01 * j, 0.002)
        ours_m['ami_ranges'][a] = dict(r=ptp(a_vals), per_seed={str(s): v for s, v in enumerate(a_vals)}, k=9)
        ours_m['map_ranges'][a] = dict(r=ptp(m_vals), per_seed=m_vals)            # a list on our side: normalised
        ranges[a] = dict(ami=dict(r=ptp(a_vals), per_seed={str(s): v for s, v in enumerate(a_vals)}),
                         map=dict(r=ptp(m_vals), per_seed={str(s): v for s, v in enumerate(m_vals)}))
    w(rec / f'mmart_picks_{S.KSTAR}.json', ours_m)
    ours_p = dict(arm=S.KSTAR, r_E=dict(value=0.00695), sets={}, comparator_picks={})
    cx_p = dict(picks={}, ranges={})
    for d, base in (('affection', 0.2), ('emoset', 0.4)):
        a_vals = lst(base)
        m_vals = lst(base + 0.1, 0.002)
        ours_p['sets'][d] = dict(ami_range=dict(r=ptp(a_vals), per_seed={str(s): v for s, v in enumerate(a_vals)}),
                                 r_map=ptp(m_vals) if d == 'affection' else 0.00695,
                                 map_range=dict(r=ptp(m_vals), per_seed={str(s): v for s, v in enumerate(m_vals)}) if d == 'affection' else None,
                                 readout=dict(pick='center+k3', val_scores={'center+k3': 0.5}))
        cx_p['picks'][d] = dict(pick='center+k3', table={'center+k3': dict(map=dict(emotion=0.5))})
        cx_p['ranges'][d] = dict(ami=dict(r=ptp(a_vals), per_seed={str(s): v for s, v in enumerate(a_vals)}),
                                 map=dict(r=ptp(m_vals), per_seed={str(s): v for s, v in enumerate(m_vals)}) if d == 'affection'
                                 else dict(r=0.00695, own_r0_range=ptp(m_vals), r_source='r14'))
    w(rec / f'photo_picks_{S.KSTAR}.json', ours_p)
    w(rd / 's2mp_val.json', dict(mode='validation', complete=True, notes=[],
                                 mmart=dict(picks=dict(p1=dict(pick='none+k5', table={'none+k5': dict(score=2.0, kept=True, map={a: 0.2 for a in ASP})}),
                                                       q1=dict(pick='center+k1', table={'center+k1': dict(score=1.0, kept=True, map={a: 0.25 for a in ASP})})),
                                            ranges=ranges, complete=True),
                                 photo=dict(picks=cx_p['picks'], ranges=cx_p['ranges'], complete=True)))


@pytest.fixture
def val(tmp_path):
    rec, rd = tmp_path / 'records', tmp_path / 'rederive'
    build_val_e2e3_sc9(rec, rd)
    build_val_mmart_photo(rec, rd)
    return rec, rd


def run_val(rec, rd, pieces=None):
    doc, ran, code = S.run('val', pieces or list(S.VAL_PIECES), rec, rd)
    return doc, ran, code


def test_val_all_pieces_pass(val):
    rec, rd = val
    doc, ran, code = run_val(*val)
    assert {n: r['status'] for n, r in ran.items()} == {n: 'pass' for n in S.VAL_PIECES}, {n: r.get('mismatches') for n, r in ran.items()}
    assert code == 0
    assert doc['mismatches']['val']['sc9'] == [] and doc['missing']['val'] == {}
    assert all(r['max_abs_diff'] == 0.0 for r in ran.values())
    assert len(ran['e2e3']['inputs']) == 3 and all(len(v) == 64 for v in ran['e2e3']['inputs'].values())
    saved = json.loads((rec / 's2_compare.json').read_text())
    assert saved['pieces']['val']['photo']['status'] == 'pass' and len(saved['written']['val']) == 16
    # the SC9 definition notes of Codex are recorded
    assert ran['sc9']['codex_status']['definition_notes'][0].startswith('seeds ahead')
    # the emoset r_E and Codex's own R0 range are handled
    assert any(u['side'] == 'codex' and 'own_r0_range' in u['field'] for u in ran['photo']['unpaired'])


def test_val_within_tolerance_passes_beyond_fails(val):
    rec, rd = val
    p = rec / f'image_picks_{ENC["siglip2"]}.json'
    d = json.loads(p.read_text())
    d['ami_ranges']['style']['r'] += 9e-7
    w(p, d)
    _, ran, code = run_val(rec, rd, ['e2e3'])
    assert ran['e2e3']['status'] == 'pass' and code == 0 and 8e-7 < ran['e2e3']['max_abs_diff'] < 1e-6
    d['ami_ranges']['style']['r'] += 3e-7
    w(p, d)
    _, ran, code = run_val(rec, rd, ['e2e3'])
    (m,) = ran['e2e3']['mismatches']
    assert code == 1 and m['id'] == 'siglip2/range/ami/style' and m['diff'] == pytest.approx(1.2e-6, rel=1e-3)
    assert m['ours'] != m['codex']


def test_val_pick_differs(val):
    rec, rd = val
    p = rd / 's2e_val.json'
    d = json.loads(p.read_text())
    d['encoders']['clip_l14']['picks']['image_map']['genre']['pick_plain_side'] = 'none+k0'
    w(p, d)
    doc, ran, code = run_val(rec, rd, ['e2e3'])
    assert code == 1
    assert [m['id'] for m in ran['e2e3']['mismatches']] == ['clip_l14/pick/image_map_plain_side/genre']
    assert doc['mismatches']['val']['e2e3'] == ran['e2e3']['mismatches']


def test_val_sc9_fields(val):
    rec, rd = val
    p = rd / 's2e_val.json'
    d = json.loads(p.read_text())
    d['sc9']['seeds_ahead'] = d['sc9']['seeds_ahead'][:-1] or [4]
    d['sc9']['close_call'] = not d['sc9']['close_call']
    d['sc9']['means']['clip_l14'] += 1e-3
    del d['sc9']['leader_range']
    w(p, d)
    _, ran, code = run_val(rec, rd, ['sc9'])
    ids = {m['id'] for m in ran['sc9']['mismatches']}
    assert code == 1
    assert {'sc9/close_call', 'sc9/mean/clip_l14', 'sc9/leader_range'} <= ids
    miss = [m for m in ran['sc9']['mismatches'] if m['id'] == 'sc9/leader_range'][0]
    assert 'missing on codex' in miss['note']


def test_val_sc9_pick_and_group(val):
    rec, rd = val
    p = rec / 'encoder_pick.json'
    d = json.loads(p.read_text())
    d['pick'] = 'siglip1' if d['pick'] != 'siglip1' else 'siglip2'
    w(p, d)
    g = rec / f'image_groups_{ENC["siglip2"]}.json'
    gd = json.loads(g.read_text())
    gd['image']['cells']['map/genre'][2] += 0.01
    w(g, gd)
    _, ran, code = run_val(rec, rd, ['sc9'])
    ids = {m['id'] for m in ran['sc9']['mismatches']}
    assert code == 1 and 'sc9/pick' in ids and 'siglip2/group/cell/map/genre[2]' in ids


def test_val_info_fields_never_mismatch(val):
    rec, rd = val
    p = rd / 's2e_val.json'
    d = json.loads(p.read_text())
    d['sc9']['qwen_info'] = 0.5
    d['encoders']['siglip2']['image_tables']['q1']['scores']['center+k10'] = 9.9
    w(p, d)
    _, ran, code = run_val(rec, rd, ['e2e3', 'sc9'])
    assert code == 0
    assert any(i['item'] == 'sc9/qwen_info' and i['equal'] is False for i in ran['sc9']['info'])
    assert any('score' in i['item'] and i['equal'] is False for i in ran['e2e3']['info'])


def test_val_mmart_photo_mismatches(val):
    rec, rd = val
    p = rd / 's2mp_val.json'
    d = json.loads(p.read_text())
    d['mmart']['picks']['q1']['pick'] = 'none+k0'
    d['photo']['ranges']['emoset']['map']['r'] = 0.007
    w(p, d)
    _, ran, code = run_val(rec, rd, ['mmart', 'photo'])
    assert code == 1
    assert [m['id'] for m in ran['mmart']['mismatches']] == ['mmart/pick/q1_side']
    assert {m['id'] for m in ran['photo']['mismatches']} == {'photo/emoset/range/map', 'photo/emoset/r_E'}


def test_val_codex_incomplete_or_pending_is_missing_not_mismatch(val):
    rec, rd = val
    p = rd / 's2e_val.json'
    d = json.loads(p.read_text())
    d['complete'] = False
    w(p, d)
    q = rd / 's2mp_val.json'
    m = json.loads(q.read_text())
    m['mmart'] = dict(picks={}, ranges={}, complete=False, pending='/x/emotion__s0.json')
    w(q, m)
    doc, ran, code = run_val(rec, rd)
    assert code == 0
    assert ran['e2e3']['status'] == ran['sc9']['status'] == ran['mmart']['status'] == 'missing'
    assert ran['photo']['status'] == 'pass'
    assert 'pending' in ran['mmart']['reason'] and ran['mmart']['codex_status']['pending'] == '/x/emotion__s0.json'
    assert set(doc['missing']['val']) == {'e2e3', 'sc9', 'mmart'}


def test_val_one_side_missing_and_sc9_without_key(val):
    rec, rd = val
    (rec / f'image_picks_{ENC["clip_l14"]}.json').unlink()
    p = rd / 's2e_val.json'
    d = json.loads(p.read_text())
    del d['sc9']
    w(p, d)
    doc, ran, code = run_val(rec, rd)
    assert code == 0 and ran['e2e3']['status'] == 'missing' and 'image_picks_clipl14_prefix' in ran['e2e3']['reason']
    assert ran['sc9']['status'] == 'missing' and 'no `sc9`' in ran['sc9']['reason']


def test_run_keeps_other_sections_and_phases(val, tmp_path):
    rec, rd = val
    run_val(rec, rd, ['e2e3', 'sc9'])
    (rd / 's2e_val.json').unlink()
    doc, ran, code = run_val(rec, rd, ['mmart'])
    saved = json.loads((rec / 's2_compare.json').read_text())
    assert saved['pieces']['val']['e2e3']['status'] == 'pass'          # untouched by the mmart run
    assert saved['pieces']['val']['mmart']['status'] == 'pass'
    assert saved['mismatches']['val']['e2e3'] == []
    # a held run touches only the held phase
    S.run('held', ['mmart'], rec, rd)
    saved = json.loads((rec / 's2_compare.json').read_text())
    assert saved['pieces']['val']['mmart']['status'] == 'pass' and saved['pieces']['held']['mmart']['status'] == 'missing'
    assert 'mmart' in saved['missing']['held'] and 'mmart' not in saved['missing']['val']


# ------------------------------------------------------------------------------------------------ held fixtures
def cell_numbers(i, outcome=None):
    """A cell's numbers with a fixed outcome by the rule: lo > 0 WIN."""
    lo = [0.02, -0.003, -0.05][i % 3]
    r = 0.01
    return dict(value=0.5 + 0.01 * i, comparator_value=0.4 + 0.01 * i, d=0.1 + lo, lo=lo, hi=lo + 0.05, r=r,
                outcome=S.rule(lo, r), n_draws=2000)


def build_held_e2e3(rec, rd):
    cx = {}
    i = 0
    for E, arm in ENC.items():
        ours_cells, cx_cells = [], []
        for a in ASP:
            for m in ('ami', 'map'):
                for ours_c, cx_c in (('plain', 'plain'), ('teacher', 'q1')):
                    n = cell_numbers(i)
                    i += 1
                    k = {'emotion': 9, 'style': 27, 'genre': 10}[a]
                    ours_cells.append(dict(id=f'artelingo/{E}/{arm}/{a}/{m}{k if m == "ami" else ""}/{ours_c}', n_rows=100, **n))
                    cx_cells.append(dict(id=f'{E}/{m}/{a}/{cx_c}', encoder=E, metric=m, aspect=a, side=cx_c, n=100, **n))
        w(rec / f'image_cells_{arm}.json', dict(arm=arm, cells=ours_cells))
        wins = sum(c['outcome'] == 'WIN' for c in cx_cells if c['side'] == 'plain')
        wq = sum(c['outcome'] in ('WIN', 'LEVEL') for c in cx_cells if c['side'] == 'q1')
        cx[E] = dict(cells=cx_cells, sc4=dict(wins_plain=wins, wins_or_level_q1=wq, level=0, inconclusive=0, inconclusive_above=False))
        w(rec / f'SC4_{E}_{arm}.json', dict(level=0, inconclusive_above=False, counts=dict(p1_win=wins, q1_win_or_level=wq,
                                                                                        p1_inconclusive=0, q1_inconclusive=0)))
    w(rd / 's2e_held.json', dict(kind='held', complete=True, encoders=cx, picks_sha256='ab'))


def build_held_mp(rec, rd, name):
    ours_cells, cx_cells = [], []
    i = 0
    if name == 'mmart':
        keys = [(a, m, c) for a in ASP for m in ('ami', 'map') for c in ('plain', 'teacher')]
    else:
        keys = [(d, m, c) for d in ('affection', 'emoset') for m in ('ami', 'map') for c in ('plain', 'teacher')]
    for x, m, c in keys:
        n = cell_numbers(i)
        i += 1
        k = 9 if m == 'ami' else ''
        pq = {'plain': 'p1', 'teacher': 'q1'}[c]
        if name == 'mmart':
            ours_cells.append(dict(id=f'mmart/siglip1/{S.KSTAR}/{x}/{m}{k}/{c}', n_rows=50, **n))
            cx_cells.append(dict(id=f'mmart/{m}/{x}/{pq}', n=50, **n))
        else:
            ours_cells.append(dict(id=f'{x}/siglip1/{S.KSTAR}/emotion/{m}{k}/{c}', n_rows=50, **n))
            cx_cells.append(dict(id=f'{x}/{m}/{pq}', n=50, **n))
    w(rec / f'{name}_cells_{S.KSTAR}.json', dict(arm=S.KSTAR, cells=ours_cells))
    lad = 'sc5' if name == 'mmart' else 'sc6'
    w(rd / 's2mp_held.json', dict(mode='held', **{name: dict(cells=cx_cells, **{lad: dict(level=1, wins_p1=3, wins_or_level_q1=2,
                                                                                          inconclusive_p1=0, inconclusive_q1=0,
                                                                                          inconclusive_above=False)})}))


def build_held_ablation(rec, rd):
    arms = {S.KSTAR: {}}
    ours_cells, cx_cells = [], []
    for ri, row in enumerate(S.ROWS):
        arms[row] = {}
        for a in ASP:
            ks = {m: lst(0.3 + 0.01 * ASP.index(a) + (0.1 if m == 'map' else 0), 0.001) for m in ('ami', 'map')}
            rs = {m: lst(0.2 + 0.01 * ri, 0.002) for m in ('ami', 'map')}
            arms[S.KSTAR][a] = {f'{m}_per_seed': ks[m] for m in ks}
            arms[row][a] = {**{f'{m}_per_seed': rs[m] for m in rs}, 'n_queries': 70}
            for m in ('ami', 'map'):
                km, rm = sum(ks[m]) / 5, sum(rs[m]) / 5
                dps = [r - k for r, k in zip(rs[m], ks[m])]
                ours_cells.append(dict(id=f'artelingo/siglip1/{row}/{a}/{m}{9 if m == "ami" else ""}/{S.KSTAR}', value=rm,
                                       comparator_value=km, d=rm - km, lo=rm - km - 0.01, hi=rm - km + 0.01, r=None, outcome=None,
                                       per_seed=dps, n_rows=100, n_draws=2000))
                cx_cells.append(dict(id=f'ablation/{row}/{a}/{m}', kstar_mean=km, removal_mean=rm, kstar_per_seed=ks[m],
                                     removal_per_seed=rs[m], d=rm - km, d_per_seed=dps, lo=rm - km - 0.01, hi=rm - km + 0.01,
                                     n_draws=2000, n=100 if m == 'ami' else 70))
    w(rec / 'ablation.json', dict(kstar=S.KSTAR, complete=True, status='ok', arms=arms, cells=ours_cells, inconclusive_removals={}))
    w(rd / 's2abl_held.json', dict(mode='held', complete=True, rows=list(S.ROWS), cells=cx_cells, kstar_check=dict(ok=True)))


@pytest.fixture
def held(tmp_path):
    rec, rd = tmp_path / 'records', tmp_path / 'rederive'
    build_held_e2e3(rec, rd)
    # the two s2mp pieces share one Codex file: build mmart then merge photo
    build_held_mp(rec, rd, 'mmart')
    mm = json.loads((rd / 's2mp_held.json').read_text())
    build_held_mp(rec, rd, 'photo')
    ph = json.loads((rd / 's2mp_held.json').read_text())
    w(rd / 's2mp_held.json', dict(mode='held', mmart=mm['mmart'], photo=ph['photo']))
    build_held_ablation(rec, rd)
    return rec, rd


def run_held(rec, rd, pieces=None):
    return S.run('held', pieces or list(S.HELD_PIECES), rec, rd)


def test_held_all_pieces_pass(held):
    rec, rd = held
    doc, ran, code = run_held(*held)
    assert {n: r['status'] for n, r in ran.items()} == {n: 'pass' for n in S.HELD_PIECES}, {n: r.get('mismatches') for n, r in ran.items()}
    assert code == 0
    assert len(ran['e2e3']['cells']) == 24 and len(ran['mmart']['cells']) == 12 and len(ran['photo']['cells']) == 8
    assert len(ran['ablation']['cells']) == 24
    assert ran['e2e3']['max_abs_diff_cells'] == 0.0
    # ladder information was compared and equal
    assert all(i['equal'] for i in ran['e2e3']['info'] if i['item'].startswith('ladder/SC4'))
    assert any(i['item'].startswith('ladder/SC5') and i['equal'] is None for i in ran['mmart']['info'])   # ours SC5 absent
    assert any(k.endswith('SC4_clip_l14_clipl14_prefix.json') for k in ran['e2e3']['inputs'])


def test_held_cell_number_and_outcome(held):
    rec, rd = held
    p = rd / 's2e_held.json'
    d = json.loads(p.read_text())
    c = next(c for c in d['encoders']['siglip2']['cells'] if c['id'] == 'siglip2/ami/emotion/plain')
    c['lo'] = -0.5           # the rule on Codex's lo gives BELOW, ours says WIN
    c['hi'] = 5.0
    c['d'] += 2e-6
    w(p, d)
    _, ran, code = run_held(rec, rd, ['e2e3'])
    assert code == 1
    ids = [m['id'] for m in ran['e2e3']['mismatches']]
    assert 'cell emotion/ami/plain lo' in ids and 'cell emotion/ami/plain d' in ids and 'cell emotion/ami/plain outcome' in ids
    assert 'cell emotion/ami/plain hi' in ids
    row = next(r for r in ran['e2e3']['cells'] if r['cell'] == 'emotion/ami/plain' and r['max_diff'] > 1e-3)
    assert row['codex_outcome_rule'] == 'BELOW'


def test_held_codex_inconclusive_field_is_information(held):
    rec, rd = held
    p = rd / 's2e_held.json'
    d = json.loads(p.read_text())
    c = d['encoders']['clip_l14']['cells'][3]
    c['outcome'] = 'INCONCLUSIVE'
    c['reason'] = 'no complete replay'
    w(p, d)
    _, ran, code = run_held(rec, rd, ['e2e3'])
    assert code == 0
    assert ran['e2e3']['codex_status']['inconclusive']['clip_l14'] == {c['id']: 'no complete replay'}
    assert any(r['codex_outcome_field'] == 'INCONCLUSIVE' for r in ran['e2e3']['cells'])


def test_held_our_outcome_must_be_the_rule_word(held):
    rec, rd = held
    p = rec / f'image_cells_{ENC["siglip2"]}.json'
    d = json.loads(p.read_text())
    d['cells'][0]['outcome'] = 'LEVEL' if d['cells'][0]['outcome'] != 'LEVEL' else 'WIN'
    w(p, d)
    _, ran, code = run_held(rec, rd, ['e2e3'])
    ids = [m['id'] for m in ran['e2e3']['mismatches']]
    assert code == 1 and any(i.endswith('outcome') for i in ids) and any(i.endswith('outcome (ours)') for i in ids)


def test_held_missing_cell_on_one_side(held):
    rec, rd = held
    p = rd / 's2mp_held.json'
    d = json.loads(p.read_text())
    d['photo']['cells'] = [c for c in d['photo']['cells'] if c['id'] != 'emoset/map/q1']
    w(p, d)
    _, ran, code = run_held(rec, rd, ['photo', 'mmart'])
    assert code == 1 and ran['mmart']['status'] == 'pass'
    (m,) = ran['photo']['mismatches']
    assert m['id'] == 'cell emoset/map/q1' and 'no cell of Codex' in m['note']


def test_held_mmart_pending_placeholder_is_missing(held):
    rec, rd = held
    w(rd / 's2mp_held.json', dict(mode='controller_pending', mmart=dict(cells=[], sc5=None), photo=dict(cells=[], sc6=None),
                                  pending='Controller must first commit'))
    doc, ran, code = run_held(rec, rd, ['mmart', 'photo'])
    assert code == 0 and ran['mmart']['status'] == ran['photo']['status'] == 'missing'
    assert ran['mmart']['codex_status']['mode'] == 'controller_pending' and set(doc['missing']['held']) == {'mmart', 'photo'}


def test_held_photo_parked_cells(held):
    rec, rd = held
    po, pc = rec / f'photo_cells_{S.KSTAR}.json', rd / 's2mp_held.json'
    d = json.loads(po.read_text())
    for c in d['cells']:
        if c['id'].startswith('emoset/') and '/map/' in c['id'] and c['id'].endswith('/plain'):
            c.update(value=None, comparator_value=None, d=None, lo=None, hi=None, outcome='INCONCLUSIVE', reason='gate (v)')
    w(po, d)
    _, ran, code = run_held(rec, rd, ['photo'])
    assert code == 1                                        # ours parked, Codex has numbers: one side has no numbers
    (m,) = ran['photo']['mismatches']
    assert m['id'] == 'cell emoset/map/p1' and 'ours cell has no numbers' in m['note']
    cd = json.loads(pc.read_text())
    for c in cd['photo']['cells']:
        if c['id'] == 'emoset/map/p1':
            c.update(value=None, comparator_value=None, d=None, lo=None, hi=None, r=None, outcome='INCONCLUSIVE', reason='rebuild')
    w(pc, cd)
    _, ran, code = run_held(rec, rd, ['photo'])
    assert code == 0 and any(r.get('parked') is True for r in ran['photo']['cells'])


def test_held_ablation_mismatch_and_missing_rows(held):
    rec, rd = held
    p = rd / 's2abl_held.json'
    d = json.loads(p.read_text())
    c = d['cells'][0]
    c['removal_per_seed'][2] += 5e-6
    c['hi'] += 1e-3
    w(p, d)
    _, ran, code = run_held(rec, rd, ['ablation'])
    ids = {m['id'] for m in ran['ablation']['mismatches']}
    assert code == 1 and ids == {f'{c["id"]} removal_per_seed[2]', f'{c["id"]} hi'}
    # Codex scored only two rows (--rows present only); ours lists the other two as inconclusive removals
    d['cells'] = [c for c in json.loads((rd / 's2abl_held.json').read_text())['cells']]
    keep = {'instructed_graphs', 'content_correction'}
    dd = json.loads(p.read_text())
    dd['cells'] = [c for c in dd['cells'] if c['id'].split('/')[1] in keep]
    dd['cells'][0]['removal_per_seed'][2] -= 5e-6
    dd['cells'][0]['hi'] -= 1e-3
    w(p, dd)
    od = json.loads((rec / 'ablation.json').read_text())
    od['cells'] = [c for c in od['cells'] if c['id'].split('/')[2] in keep]
    od['inconclusive_removals'] = {'buddy': 'not trained', 'relational': 'journal says crashed'}
    w(rec / 'ablation.json', od)
    _, ran, code = run_held(rec, rd, ['ablation'])
    assert code == 0 and ran['ablation']['status'] == 'pass' and len(ran['ablation']['cells']) == 12
    assert sum(u['side'] == 'both' for u in ran['ablation']['unpaired']) == 12
    # a row scored by Codex but absent in ours (and not listed) is a mismatch
    od['inconclusive_removals'] = {}
    od['cells'] = [c for c in od['cells'] if c['id'].split('/')[2] != 'content_correction']
    w(rec / 'ablation.json', od)
    _, ran, code = run_held(rec, rd, ['ablation'])
    assert code == 1 and len(ran['ablation']['mismatches']) == 6


def test_held_ablation_ours_without_cells_is_missing(held):
    rec, rd = held
    w(rec / 'ablation.json', dict(kind='r29_ablation', complete=False, status='inconclusive', cause='gate vii_ablation is parked'))
    _, ran, code = run_held(rec, rd, ['ablation'])
    assert code == 0 and ran['ablation']['status'] == 'missing' and 'parked' in ran['ablation']['reason']


def test_cli_exit_codes_and_default_pieces(held, tmp_path, capsys):
    rec, rd = held
    args = ['--phase', 'held', '--records', str(rec), '--rederive', str(rd)]
    assert S.main(args) == 0
    out = capsys.readouterr().out
    assert 'held/ablation: pass' in out
    saved = json.loads((rec / 's2_compare.json').read_text())
    assert set(saved['pieces']['held']) == set(S.HELD_PIECES)
    (rd / 's2e_held.json').unlink()                                    # one side missing: reported, not a mismatch
    assert S.main(args) == 0
    assert 'held/e2e3: MISSING' in capsys.readouterr().out
    p = rd / 's2abl_held.json'
    d = json.loads(p.read_text())
    d['cells'][5]['d'] += 1.0
    w(p, d)
    assert S.main([*args, '--pieces', 'ablation', 'mmart']) == 1
    assert 'held/ablation: mismatch' in capsys.readouterr().out
    with pytest.raises(SystemExit):
        S.main(['--phase', 'val', '--pieces', 'ablation', '--records', str(rec), '--rederive', str(rd)])


def test_rule_edges():
    assert S.rule(1e-12, 0.01) == 'WIN' and S.rule(0.0, 0.01) == 'LEVEL' and S.rule(-0.01, 0.01) == 'BELOW'
