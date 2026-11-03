"""Tests of ns_outcome (r17 ticket 02): the 412 cells, outcome words at the boundaries, every row of §3.5's gate table
(exactly the cells it names), stage 1 ranges and picks, the refusals, the licensed sentences, the summary, the file
round trip. The expected sets are written here from the spec, not taken from the module's tables."""
import copy
import json
import re
from collections import Counter

import pytest

import ns_paths as P
import ns_outcome as O

SPEC = P.REPO / 'docs/superpowers/specs/2026-11-03-new-scores-design.md'
ENCS, SETS = P.ENCODERS, ('artelingo', 'mmart')
ASP = P.ASPECTS
R = 0.01


# ------------------------------------------------------------------------------------------------ helpers
def mk(cc, lo=0.02, r=R, n_draws=None):
    """A stage 2 cell for claim cell `cc` with lower bound `lo` and range `r`."""
    return dict(id=cc['id'], sc=cc['sc'], set=cc['set'], encoder=cc['encoder'], aspect=cc['aspect'], task=cc['task'],
                metric=cc['metric'], k=cc['k'], candidate=cc['candidate'], comparator=cc['comparator'],
                cand=0.5, comp=0.4, d=lo + 0.01, lo=lo, hi=lo + 0.02, r=r, r_source='test',
                n_draws=P.N_DRAWS[cc['task']] if n_draws is None else n_draws, n_rows=1000)


def all_cells(lo=0.02, r=R):
    return [mk(c, lo, r) for c in O.claim_cells()]


def ok_status():
    ranges = {rk: 'ok' for rk in {O.range_key(c) for c in O.claim_cells()} if rk}
    picks = {pk: 'ok' for pk in {O.pick_key(c) for c in O.claim_cells()} if pk}
    return dict(
        gates=dict(i_a='ok', i_b='ok', i_d='ok', ii_scorer='ok', ii_crl='ok', ii_percept='ok', vi='ok',
                   i_c={s: 'ok' for s in P.PHOTO_SETS},
                   iii_metric={m: 'ok' for m in P.CLASS_METRICS},
                   iii_count={f'{s}/{a}': 'ok' for (s, a) in P.CLASS_COUNT},
                   iv={e: 'ok' for e in ENCS}, v={e: 'ok' for e in ENCS}),
        stage1=dict(ranges=ranges, picks=picks))


def fail(status, *path, value='failed'):
    s = copy.deepcopy(status)
    d = s
    for p in path[:-1]:
        d = d[p]
    assert path[-1] in d, path
    d[path[-1]] = value
    return s


def voided(rows):
    return {r['id'] for r in rows if r['void']}


def ids(pred):
    return {c['id'] for c in O.claim_cells() if pred(c)}


def run(cells=None, status=None, **kw):
    return O.outcomes(all_cells() if cells is None else cells, ok_status() if status is None else status, **kw)


def row_of(rows, cid):
    return next(r for r in rows if r['id'] == cid)


def find(**kw):
    out = [c for c in O.claim_cells() if all(c[k] == v for k, v in kw.items())]
    assert len(out) == 1, (kw, len(out))
    return out[0]


def is_ami20(c):
    return c['id'].split('/')[4] == 'ami@20'


# ------------------------------------------------------------------------------------------------ the 412 cells
def test_412_cells_unique_ids_counts_per_sc():
    cells = O.claim_cells()
    assert len(cells) == 412
    assert len({c['id'] for c in cells}) == 412
    assert Counter(c['sc'] for c in cells) == {'SC1': 96, 'SC2': 96, 'SC3': 96, 'SC4': 40, 'SC5': 30, 'SC6': 6,
                                               'SC7': 24, 'SC8': 24}


def test_id_format_seven_fields():
    for c in O.claim_cells():
        parts = c['id'].split('/')
        assert len(parts) == 7, c['id']
        sc, s, e, a, m, k, comp = parts
        assert (sc, s, e, a, comp) == (c['sc'], c['set'], c['encoder'], c['aspect'], c['comparator'])
        assert k == ('-' if c['k'] is None else str(c['k']))
        assert m == c['metric'] or (c['sc'] == 'SC5' and m == 'ami@20' and c['metric'] == 'ami')


def test_candidates_comparators_tasks():
    want = {'SC1': ('cstar', {'plain'}, 'clu'), 'SC2': ('cstar', {'matched'}, 'clu'),
            'SC3': ('cstar', {'teacher'}, 'clu'), 'SC4': ('photo', {'plain', 'teacher'}, 'clu'),
            'SC5': ('cstar', {'crl'}, 'clu'), 'SC6': ('cstar', {'crl'}, 'ret'), 'SC7': ('untold', {'plain'}, 'ret'),
            'SC8': ('cstar', {'matched'}, 'ret')}
    for c in O.claim_cells():
        cand, comps, task = want[c['sc']]
        assert (c['candidate'], c['task']) == (cand, task), c['id']
        assert c['comparator'] in comps, c['id']


def test_scope_per_sc():
    cells = O.claim_cells()
    for sc in ('SC1', 'SC2', 'SC3'):
        got = {(c['set'], c['encoder'], c['aspect'], c['metric']) for c in cells if c['sc'] == sc}
        assert got == {(s, e, a, m) for s in SETS for e in ENCS for a in ASP for m in P.CLASS_METRICS}
    got = {(c['set'], c['encoder'], c['comparator'], c['metric']) for c in cells if c['sc'] == 'SC4'}
    sets_enc = [('affection', e) for e in ENCS] + [('emoset', 'qwen')]
    assert got == {(s, e, comp, m) for s, e in sets_enc for comp in ('plain', 'teacher') for m in P.CLASS_METRICS}
    assert {c['aspect'] for c in cells if c['sc'] == 'SC4'} == {'emotion'}
    for sc in ('SC5', 'SC6'):
        assert {c['encoder'] for c in cells if c['sc'] == sc} == {'clip_l14'}
    assert {(c['set'], c['aspect']) for c in cells if c['sc'] == 'SC6'} == {(s, a) for s in SETS for a in ASP}
    for sc in ('SC7', 'SC8'):
        got = {(c['set'], c['encoder'], c['aspect']) for c in cells if c['sc'] == sc}
        assert got == {(s, e, a) for s in SETS for e in ENCS for a in ASP}
        assert all(c['metric'] == 'map' for c in cells if c['sc'] == sc)


def test_k_values():
    for c in O.claim_cells():
        if c['task'] == 'ret':
            assert c['k'] is None, c['id']
        elif c['sc'] == 'SC5' and is_ami20(c):
            assert c['k'] == 20
        else:
            assert c['k'] == P.CLASS_COUNT[(c['set'], c['aspect'])], c['id']
    sc5 = [c for c in O.claim_cells() if c['sc'] == 'SC5']
    assert sum(1 for c in sc5 if is_ami20(c)) == 6
    assert sum(1 for c in sc5 if c['kind'] == 'class_count') == 24


def test_bars():
    for c in O.claim_cells():
        teacher_win = (c['comparator'] == 'teacher' and
                       ((c['sc'] == 'SC3' and c['set'] == 'mmart' and c['aspect'] in ('emotion', 'style'))
                        or (c['sc'] == 'SC4' and c['set'] == 'emoset')))
        if c['comparator'] == 'teacher' and not teacher_win:
            assert c['bar'] == 'LEVEL_OR_WIN', c['id']
        else:
            assert c['bar'] == 'WIN', c['id']
    assert sum(1 for c in O.claim_cells() if c['bar'] == 'LEVEL_OR_WIN') == (96 - 32) + 16


def test_kinds():
    kinds = Counter(c['kind'] for c in O.claim_cells())
    assert kinds == {'class_count': 96 * 3 + 40 + 24, 'fixed20': 6, 'map': 54}


# ------------------------------------------------------------------------------------------------ outcome words
@pytest.mark.parametrize('lo,word', [(0.001, 'WIN'), (1e-12, 'WIN'), (0.0, 'LEVEL'), (-0.0099, 'LEVEL'),
                                     (-0.01, 'BELOW'), (-0.0101, 'BELOW'), (-0.5, 'BELOW')])
def test_word_boundaries(lo, word):
    cc = find(sc='SC1', set='mmart', encoder='qwen', aspect='genre', metric='ari')
    rows = run([mk(cc, lo=lo, r=0.01)])
    assert row_of(rows, cc['id'])['outcome'] == word


def test_word_matches_rt_stats_rule():
    import rt_stats
    cc = find(sc='SC1', set='mmart', encoder='qwen', aspect='genre', metric='ari')
    for lo in (-0.02, -0.01, -0.005, 0.0, 1e-9, 0.3):
        assert row_of(run([mk(cc, lo=lo)]), cc['id'])['outcome'] == rt_stats.outcome(lo, R)


def test_win_bar_meets_only_at_win():
    cc = find(sc='SC1', set='mmart', encoder='qwen', aspect='genre', metric='ari')
    for lo, meets in ((0.01, True), (0.0, False), (-0.005, False), (-0.5, False)):
        r = row_of(run([mk(cc, lo=lo)]), cc['id'])
        assert r['meets_bar'] is meets and r['status'] == 'decided' and r['void'] == []


def test_level_or_win_bar():
    cc = find(sc='SC3', set='artelingo', encoder='qwen', aspect='genre', metric='ami')
    assert cc['bar'] == 'LEVEL_OR_WIN'
    for lo, word, meets in ((0.01, 'WIN', True), (0.0, 'LEVEL', True), (-0.0099, 'LEVEL', True),
                            (-0.01, 'BELOW', False), (-0.3, 'BELOW', False)):
        r = row_of(run([mk(cc, lo=lo)]), cc['id'])
        assert (r['outcome'], r['meets_bar']) == (word, meets)


def test_row_carries_cell_numbers():
    cc = find(sc='SC6', set='mmart', aspect='style')
    c = mk(cc, lo=0.03)
    r = row_of(run([c]), cc['id'])
    for k in ('cand', 'comp', 'd', 'lo', 'hi', 'r', 'n_draws', 'n_rows'):
        assert r[k] == c[k]
    assert r['bar'] == 'WIN' and r['present'] is True


# ------------------------------------------------------------------------------------------------ gate table (D18)
def check_gate(status, expected):
    rows = run(status=status)
    assert len(rows) == 412
    assert voided(rows) == expected
    for r in rows:
        assert (r['status'] == 'inconclusive') == (r['id'] in expected), r['id']
        assert (r['meets_bar'] is None) if r['id'] in expected else (r['meets_bar'] is True)
        assert r['present'] is True


def test_gate_i_a():
    check_gate(fail(ok_status(), 'gates', 'i_a'), ids(lambda c: c['sc'] in ('SC1', 'SC2', 'SC3', 'SC5')))


def test_gate_i_b():
    check_gate(fail(ok_status(), 'gates', 'i_b'), ids(lambda c: c['sc'] == 'SC2'))


@pytest.mark.parametrize('s', P.PHOTO_SETS)
def test_gate_i_c(s):
    check_gate(fail(ok_status(), 'gates', 'i_c', s), ids(lambda c: c['sc'] == 'SC4' and c['set'] == s))


def test_gate_i_d_every_class_count_cell():
    exp = ids(lambda c: c['sc'] in ('SC1', 'SC2', 'SC3', 'SC4') or (c['sc'] == 'SC5' and not is_ami20(c)))
    assert len(exp) == 96 * 3 + 40 + 24
    check_gate(fail(ok_status(), 'gates', 'i_d'), exp)


def test_gate_i_d_spares_ami_at_20_and_retrieval():
    rows = run(status=fail(ok_status(), 'gates', 'i_d'))
    for r in rows:
        if r['sc'] in ('SC6', 'SC7', 'SC8') or r['id'].split('/')[4] == 'ami@20':
            assert not r['void'], r['id']


def test_gate_ii_scorer():
    check_gate(fail(ok_status(), 'gates', 'ii_scorer'), ids(lambda c: c['sc'] in ('SC6', 'SC7', 'SC8')))


def test_gate_ii_crl():
    check_gate(fail(ok_status(), 'gates', 'ii_crl'), ids(lambda c: c['sc'] in ('SC5', 'SC6')))


def test_gate_ii_percept():
    check_gate(fail(ok_status(), 'gates', 'ii_percept'), ids(lambda c: c['sc'] == 'SC8'))


@pytest.mark.parametrize('m', P.CLASS_METRICS)
def test_gate_iii_metric(m):
    exp = ids(lambda c: c['metric'] == m)
    assert len(exp) > 0
    check_gate(fail(ok_status(), 'gates', 'iii_metric', m), exp)


def test_gate_iii_metric_voids_ami_at_20_too():
    rows = run(status=fail(ok_status(), 'gates', 'iii_metric', 'ami'))
    assert row_of(rows, find(sc='SC5', set='artelingo', aspect='genre', metric='ami', k=20, kind='fixed20')['id'])['void']


@pytest.mark.parametrize('key', [f'{s}/{a}' for (s, a) in P.CLASS_COUNT])
def test_gate_iii_count(key):
    s, a = key.split('/')
    exp = ids(lambda c: c['set'] == s and c['aspect'] == a and c['kind'] == 'class_count')
    assert exp
    check_gate(fail(ok_status(), 'gates', 'iii_count', key), exp)


@pytest.mark.parametrize('e', ENCS)
def test_gate_iv(e):
    # X's SC8 cells with a new pick (all of X's, but Qwen on ArtELingo) and X's MMArt SC2 cells
    exp = ids(lambda c: c['encoder'] == e and ((c['sc'] == 'SC8' and not (e == 'qwen' and c['set'] == 'artelingo'))
                                                 or (c['sc'] == 'SC2' and c['set'] == 'mmart')))
    assert len(exp) == (6 + 12 if e != 'qwen' else 3 + 12)
    check_gate(fail(ok_status(), 'gates', 'iv', e), exp)


def test_gate_iv_qwen_spares_artelingo_sc8_and_artelingo_sc2():
    rows = run(status=fail(ok_status(), 'gates', 'iv', 'qwen'))
    assert not row_of(rows, find(sc='SC8', set='artelingo', encoder='qwen', aspect='style')['id'])['void']
    assert not row_of(rows, find(sc='SC2', set='artelingo', encoder='qwen', aspect='style', metric='ami')['id'])['void']
    assert row_of(rows, find(sc='SC8', set='mmart', encoder='qwen', aspect='style')['id'])['void']


@pytest.mark.parametrize('e', ENCS)
def test_gate_v(e):
    check_gate(fail(ok_status(), 'gates', 'v', e), ids(lambda c: c['sc'] == 'SC7' and c['encoder'] == e))


def test_gate_vi():
    exp = ids(lambda c: (c['sc'] == 'SC6' and c['set'] == 'mmart') or c['sc'] == 'SC7'
              or (c['sc'] == 'SC8' and not (c['encoder'] == 'qwen' and c['set'] == 'artelingo')))
    assert len(exp) == 3 + 24 + 21
    check_gate(fail(ok_status(), 'gates', 'vi'), exp)


def test_two_gates_union_and_reasons_name_the_gates():
    s = fail(fail(ok_status(), 'gates', 'ii_crl'), 'gates', 'i_b')
    rows = run(status=s)
    assert voided(rows) == ids(lambda c: c['sc'] in ('SC2', 'SC5', 'SC6'))
    both = row_of(rows, find(sc='SC5', set='mmart', aspect='style', metric='nmi')['id'])
    assert len(both['void']) == 1 and 'ii_crl' in both['void'][0]
    sc2 = row_of(rows, find(sc='SC2', set='mmart', encoder='qwen', aspect='style', metric='nmi')['id'])
    assert len(sc2['void']) == 1 and 'i_b' in sc2['void'][0]
    s = fail(fail(ok_status(), 'gates', 'i_a'), 'gates', 'i_b')
    r = row_of(run(status=s), find(sc='SC2', set='mmart', encoder='qwen', aspect='style', metric='nmi')['id'])
    assert len(r['void']) == 2


def test_gate_value_missing_voids_like_failed():
    check_gate(fail(ok_status(), 'gates', 'i_b', value='missing'), ids(lambda c: c['sc'] == 'SC2'))


# ------------------------------------------------------------------------------------------------ stage 1
def test_range_keys():
    cc = find(sc='SC1', set='artelingo', encoder='siglip1', aspect='genre', metric='nmi')
    assert O.range_key(cc) == 'cstar/artelingo/siglip1/genre/nmi'
    cc = find(sc='SC4', set='emoset', comparator='plain', metric='acc')
    assert O.range_key(cc) == 'photo/emoset/qwen/emotion/acc'
    cc = find(sc='SC5', set='mmart', aspect='style', metric='ari')
    assert O.range_key(cc) == 'cstar/mmart/clip_l14/style/ari'
    assert O.range_key(find(sc='SC6', set='mmart', aspect='style')) is None
    assert O.range_key(find(sc='SC5', set='mmart', aspect='style', metric='ami', k=20, kind='fixed20')) is None
    assert len({O.range_key(c) for c in O.claim_cells()} - {None}) == 96 + 20     # SC5 shares clip_l14's SC1 keys


def test_range_key_shared_by_comparators():
    rk = 'cstar/mmart/siglip2/style/ami'
    assert {c['sc'] for c in O.claim_cells() if O.range_key(c) == rk} == {'SC1', 'SC2', 'SC3'}
    rk = 'cstar/mmart/clip_l14/style/ami'
    assert {c['sc'] for c in O.claim_cells() if O.range_key(c) == rk} == {'SC1', 'SC2', 'SC3', 'SC5'}
    rk = 'photo/affection/siglip2/emotion/ari'
    assert {c['comparator'] for c in O.claim_cells() if O.range_key(c) == rk} == {'plain', 'teacher'}


def test_pick_keys():
    assert O.pick_key(find(sc='SC7', set='artelingo', encoder='qwen', aspect='genre')) == 'untold/artelingo/qwen'
    assert O.pick_key(find(sc='SC8', set='mmart', encoder='qwen', aspect='genre')) == 'matched/mmart/qwen'
    assert O.pick_key(find(sc='SC8', set='artelingo', encoder='qwen', aspect='genre')) is None
    assert O.pick_key(find(sc='SC8', set='artelingo', encoder='siglip1', aspect='genre')) == 'matched/artelingo/siglip1'
    assert O.pick_key(find(sc='SC6', set='mmart', aspect='genre')) == 'crl/mmart/clip_l14'
    assert O.pick_key(find(sc='SC6', set='artelingo', aspect='genre')) is None
    assert O.pick_key(find(sc='SC1', set='mmart', encoder='qwen', aspect='genre', metric='ami')) is None
    keys = {O.pick_key(c) for c in O.claim_cells()} - {None}
    assert len(keys) == 8 + 7 + 1


def test_range_missing_win_bar_cell_drops_only_the_level_word():
    cc = find(sc='SC1', set='mmart', encoder='qwen', aspect='genre', metric='ari')
    s = fail(ok_status(), 'stage1', 'ranges', O.range_key(cc), value='missing')
    # the range failed: the cell may still carry a number r; it is not read
    for lo, word, meets in ((0.01, 'WIN', True), (0.0, 'not WIN', False), (-0.005, 'not WIN', False),
                            (-0.5, 'not WIN', False)):
        r = row_of(run([mk(cc, lo=lo, r=0.01)], status=s), cc['id'])
        assert (r['outcome'], r['meets_bar'], r['status'], r['void']) == (word, meets, 'decided', [])
    r = row_of(run([mk(cc, lo=-0.5, r=None)], status=s), cc['id'])        # also when the cell has no r at all
    assert r['outcome'] == 'not WIN' and r['status'] == 'decided'


def test_range_missing_voids_nothing_else():
    cc = find(sc='SC1', set='mmart', encoder='qwen', aspect='genre', metric='ari')
    s = fail(ok_status(), 'stage1', 'ranges', O.range_key(cc), value='failed')
    rows = run(status=s)       # all cells at lo=0.02: WIN everywhere
    assert voided(rows) == set()
    assert Counter(r['outcome'] for r in rows) == {'WIN': 412}


def test_range_missing_teacher_cells_inconclusive_unless_win():
    cc = find(sc='SC3', set='artelingo', encoder='qwen', aspect='genre', metric='ami')      # level-or-win bar
    s = fail(ok_status(), 'stage1', 'ranges', O.range_key(cc), value='missing')
    r = row_of(run([mk(cc, lo=0.02)], s), cc['id'])
    assert (r['outcome'], r['status'], r['meets_bar'], r['void']) == ('WIN', 'decided', True, [])
    for lo in (0.0, -0.005, -0.5):
        r = row_of(run([mk(cc, lo=lo)], s), cc['id'])
        assert r['status'] == 'inconclusive' and r['meets_bar'] is None and r['outcome'] == 'not WIN'
        assert len(r['void']) == 1 and 'range' in r['void'][0]
    cw = find(sc='SC3', set='mmart', encoder='qwen', aspect='style', metric='ami')          # WIN bar
    assert cw['bar'] == 'WIN'
    s = fail(ok_status(), 'stage1', 'ranges', O.range_key(cw), value='missing')
    assert row_of(run([mk(cw, lo=0.02)], s), cw['id'])['status'] == 'decided'
    assert row_of(run([mk(cw, lo=-0.005)], s), cw['id'])['status'] == 'inconclusive'
    for kw in (dict(set='affection', encoder='siglip2'), dict(set='emoset', encoder='qwen')):   # SC4 teacher
        ct = find(sc='SC4', comparator='teacher', metric='nmi', **kw)
        s = fail(ok_status(), 'stage1', 'ranges', O.range_key(ct), value='missing')
        assert row_of(run([mk(ct, lo=0.02)], s), ct['id'])['status'] == 'decided'
        assert row_of(run([mk(ct, lo=-0.001)], s), ct['id'])['status'] == 'inconclusive'
    cp = find(sc='SC4', comparator='plain', metric='nmi', set='affection', encoder='siglip2')   # not a teacher cell
    s = fail(ok_status(), 'stage1', 'ranges', O.range_key(cp), value='missing')
    r = row_of(run([mk(cp, lo=-0.001)], s), cp['id'])
    assert r['status'] == 'decided' and r['outcome'] == 'not WIN'


def test_range_missing_sc5_class_count_cell():
    cc = find(sc='SC5', set='artelingo', aspect='genre', metric='acc')
    s = fail(ok_status(), 'stage1', 'ranges', O.range_key(cc), value='missing')
    r = row_of(run([mk(cc, lo=-0.001)], s), cc['id'])
    assert r['outcome'] == 'not WIN' and r['status'] == 'decided'


def test_ranges_do_not_touch_ami_at_20_or_retrieval():
    # their r is a recorded range, not a stage 1 range: every stage 1 range missing changes nothing there
    s = ok_status()
    for k in s['stage1']['ranges']:
        s['stage1']['ranges'][k] = 'missing'
    rows = run([mk(c, lo=-0.005) for c in O.claim_cells()], s)
    for r in rows:
        cc = find(id=r['id'])
        assert r['outcome'] == ('LEVEL' if O.range_key(cc) is None else 'not WIN'), r['id']


def test_absent_range_key_is_missing():
    s = ok_status()
    cc = find(sc='SC1', set='mmart', encoder='qwen', aspect='genre', metric='ari')
    del s['stage1']['ranges'][O.range_key(cc)]
    r = row_of(run([mk(cc, lo=-0.005)], s), cc['id'])
    assert r['outcome'] == 'not WIN'


def test_unknown_range_or_pick_key_refused():
    s = ok_status()
    s['stage1']['ranges']['cstar/mmart/qwen/genre/amiii'] = 'ok'
    with pytest.raises(AssertionError, match='range'):
        run(status=s)
    s = ok_status()
    s['stage1']['picks']['untold/mmart/nope'] = 'ok'
    with pytest.raises(AssertionError, match='pick'):
        run(status=s)


@pytest.mark.parametrize('pk', sorted({O.pick_key(c) for c in O.claim_cells()} - {None}))
def test_pick_missing_voids_that_arms_cells(pk):
    arm, s, e = pk.split('/')
    if arm == 'untold':
        exp = ids(lambda c: c['sc'] == 'SC7' and c['set'] == s and c['encoder'] == e)
        assert len(exp) == 3
    elif arm == 'matched':
        exp = ids(lambda c: c['sc'] == 'SC8' and c['set'] == s and c['encoder'] == e)
        assert len(exp) == 3
    else:
        assert pk == 'crl/mmart/clip_l14'
        exp = ids(lambda c: c['sc'] == 'SC6' and c['set'] == 'mmart')
        assert len(exp) == 3
    check_gate(fail(ok_status(), 'stage1', 'picks', pk, value='missing'), exp)


def test_absent_pick_key_voids():
    s = ok_status()
    del s['stage1']['picks']['untold/mmart/siglip1']
    rows = run(status=s)
    assert voided(rows) == ids(lambda c: c['sc'] == 'SC7' and c['set'] == 'mmart' and c['encoder'] == 'siglip1')


# ------------------------------------------------------------------------------------------------ refusals
def test_refuses_unknown_id():
    cells = all_cells()
    cells[0]['id'] = 'SC1/artelingo/qwen/emotion/ami/9/nope'
    with pytest.raises(AssertionError, match='not in claim_cells'):
        run(cells)


def test_refuses_duplicate():
    cells = all_cells()
    with pytest.raises(AssertionError, match='duplicate'):
        run(cells + [copy.deepcopy(cells[5])])


def test_missing_cells_reported_not_dropped():
    cells = all_cells()
    gone = {cells[3]['id'], cells[200]['id']}
    rows = run([c for c in cells if c['id'] not in gone])
    assert len(rows) == 412
    for cid in gone:
        r = row_of(rows, cid)
        assert (r['status'], r['present'], r['outcome'], r['meets_bar']) == ('missing', False, None, None)
    assert Counter(r['status'] for r in rows) == {'decided': 410, 'missing': 2}
    s = O.summary(rows)
    assert sum(v['missing'] for sc in s.values() for v in sc.values()) == 2


def test_missing_and_voided_cell_is_inconclusive_with_reason():
    cells = [c for c in all_cells() if c['sc'] != 'SC2']
    rows = run(cells, fail(ok_status(), 'gates', 'i_b'))
    r = row_of(rows, find(sc='SC2', set='mmart', encoder='qwen', aspect='style', metric='ami')['id'])
    assert r['status'] == 'inconclusive' and r['present'] is False and r['void'] and r['outcome'] is None


@pytest.mark.parametrize('field,bad', [('set', 'mmart'), ('encoder', 'siglip1'), ('aspect', 'genre'),
                                       ('metric', 'nmi'), ('k', 3), ('comparator', 'crl'), ('candidate', 'photo'),
                                       ('task', 'ret'), ('sc', 'SC2')])
def test_refuses_cell_whose_fields_disagree_with_its_id(field, bad):
    cc = find(sc='SC1', set='artelingo', encoder='qwen', aspect='emotion', metric='ami')
    c = mk(cc)
    c[field] = bad
    with pytest.raises(AssertionError, match='disagrees'):
        run([c])


@pytest.mark.parametrize('field,bad', [('lo', float('nan')), ('hi', float('inf')), ('d', None), ('cand', 'x'),
                                       ('n_rows', 0)])
def test_refuses_bad_numbers(field, bad):
    cc = find(sc='SC1', set='artelingo', encoder='qwen', aspect='emotion', metric='ami')
    c = mk(cc)
    c[field] = bad
    with pytest.raises(AssertionError):
        run([c])


def test_refuses_lo_above_hi_wrong_draws_and_missing_fields():
    cc = find(sc='SC1', set='artelingo', encoder='qwen', aspect='emotion', metric='ami')
    c = mk(cc)
    c['lo'], c['hi'] = 0.3, 0.1
    with pytest.raises(AssertionError, match='lo > hi'):
        run([c])
    with pytest.raises(AssertionError, match='n_draws'):
        run([mk(cc, n_draws=40)], smoke=False)
    run([mk(cc, n_draws=40)], smoke=True)          # smoke allows other draws
    c = mk(cc)
    del c['r']
    with pytest.raises(AssertionError, match='lacks'):
        run([c])


def test_refuses_missing_or_bad_r_when_the_range_is_used():
    cc = find(sc='SC1', set='artelingo', encoder='qwen', aspect='emotion', metric='ami')
    for bad in (None, -0.1, float('nan')):
        with pytest.raises(AssertionError, match='r '):
            run([mk(cc, r=bad)])
    cr = find(sc='SC6', set='mmart', aspect='style')
    with pytest.raises(AssertionError, match='r '):
        run([mk(cr, r=None)])
    cf = find(sc='SC5', set='mmart', aspect='style', kind='fixed20')
    with pytest.raises(AssertionError, match='r '):
        run([mk(cf, r=None)])


def test_status_shape_refused():
    def broken(fn):
        s = ok_status()
        fn(s)
        with pytest.raises(AssertionError, match='status'):
            run(status=s)
    broken(lambda s: s['gates'].pop('vi'))
    broken(lambda s: s['gates']['iv'].pop('qwen'))
    broken(lambda s: s['gates'].__setitem__('i_a', 'maybe'))
    broken(lambda s: s['gates'].__setitem__('extra', 'ok'))
    broken(lambda s: s['gates']['iii_count'].pop('mmart/style'))
    broken(lambda s: s['gates']['iii_metric'].pop('acc'))
    broken(lambda s: s.pop('stage1'))


# ------------------------------------------------------------------------------------------------ licensed sentences
def lic(cid_kw, lo, r=R, status=None):
    cc = find(**cid_kw)
    row = row_of(run([mk(cc, lo=lo, r=r)], status), cc['id'])
    return O.licensed(row)


SC3_KW = dict(sc='SC3', set='artelingo', encoder='qwen', aspect='style', metric='ami')


def test_licensed_none_unless_decided_and_meets_bar():
    kw = dict(sc='SC1', set='artelingo', encoder='qwen', aspect='style', metric='ami')
    assert lic(kw, lo=0.0) is None
    assert lic(kw, lo=-0.5) is None
    assert lic(kw, lo=0.5, status=fail(ok_status(), 'gates', 'i_a')) is None
    assert lic(kw, lo=0.5) is not None
    cc = find(**kw)
    assert O.licensed(row_of(O.outcomes([], ok_status()), cc['id'])) is None


def test_teacher_wording_level_or_win_cell():
    assert lic(SC3_KW, 0.01) == ('On ArtELingo held, c* on Qwen3-VL-Embedding-2B clusters paintings by style level '
                                 'with or better than the Qwen image instructed for style with as many clusters as '
                                 'classes (27), in AMI (5 seeds).')
    assert 'level with or better than' in lic(SC3_KW, -0.005)
    assert lic(SC3_KW, -0.01) is None
    assert lic(SC3_KW, -0.4) is None


def test_teacher_wording_win_bar_cell():
    kw = dict(sc='SC3', set='mmart', encoder='siglip2', aspect='emotion', metric='nmi')
    assert find(**kw)['bar'] == 'WIN'
    assert lic(kw, 0.01) == ('On MMArt test, c* on SigLIP 2 clusters paintings by emotion better than the Qwen image '
                             'instructed for emotion with as many clusters as classes (9), in NMI (5 seeds).')
    assert lic(kw, 0.0) == ('On MMArt test, c* on SigLIP 2 clusters paintings by emotion level with or better than '
                            'the Qwen image instructed for emotion with as many clusters as classes (9), in NMI '
                            '(5 seeds).')
    assert lic(kw, -0.5) is None


def test_teacher_wording_sc4():
    kw = dict(sc='SC4', set='emoset', encoder='qwen', comparator='teacher', metric='acc')
    assert lic(kw, 0.01) == ('On EmoSet test, the photo model on Qwen3-VL-Embedding-2B clusters photos by emotion '
                             'better than the Qwen image instructed for emotion with as many clusters as classes (8), '
                             'in ACC (5 seeds).')
    assert 'level with or better than' in lic(kw, 0.0)
    kw = dict(sc='SC4', set='affection', encoder='siglip1', comparator='teacher', metric='acc')
    assert find(**kw)['bar'] == 'LEVEL_OR_WIN'
    assert lic(kw, 0.01).count('level with or better than') == 1       # another teacher cell, even at WIN
    assert lic(kw, -0.005).count('level with or better than') == 1
    assert lic(kw, -0.02) is None


def test_non_teacher_wording_is_better_than():
    kw = dict(sc='SC1', set='artelingo', encoder='qwen', aspect='style', metric='ami')
    assert 'level with' not in lic(kw, 0.01)


def test_sentences_sc1_sc2_sc4_plain():
    assert lic(dict(sc='SC1', set='artelingo', encoder='qwen', aspect='style', metric='ami'), 0.01) == (
        'On ArtELingo held, c* on Qwen3-VL-Embedding-2B clusters paintings by style better than the plain '
        'Qwen3-VL-Embedding-2B image with as many clusters as classes (27), in AMI (5 seeds).')
    assert lic(dict(sc='SC2', set='mmart', encoder='siglip1', aspect='genre', metric='ari'), 0.01) == (
        'On MMArt test, c* on SigLIP 1 clusters paintings by genre better than PercepT on SigLIP 1, its k-means '
        'reading, with as many clusters as classes (10), in ARI (5 seeds).')
    assert lic(dict(sc='SC4', set='affection', encoder='clip_l14', comparator='plain', metric='ami'), 0.01) == (
        'On Affection test, the photo model on CLIP L/14 clusters photos by emotion better than the plain CLIP L/14 '
        'image with as many clusters as classes (9), in AMI (5 seeds).')


def test_sentences_sc5_to_sc8():
    assert lic(dict(sc='SC5', set='artelingo', aspect='genre', metric='ami', k=20, kind='fixed20'), 0.01) == (
        'c* on CLIP L/14 clusters paintings by genre better than CRL on CLIP L/14 on ArtELingo held (AMI with 20 '
        'clusters).')
    assert lic(dict(sc='SC5', set='mmart', aspect='genre', metric='nmi'), 0.01) == (
        'c* on CLIP L/14 clusters paintings by genre better than CRL on CLIP L/14 on MMArt test (NMI with as many '
        'clusters as classes, 10).')
    assert lic(dict(sc='SC6', set='mmart', aspect='style'), 0.01) == (
        'c* on CLIP L/14 retrieves paintings by style better than CRL on CLIP L/14 on MMArt test (mAP).')
    assert lic(dict(sc='SC7', set='artelingo', encoder='siglip1', aspect='genre'), 0.01) == (
        "On ArtELingo held, when the query names no aspect, c*'s joined heads on SigLIP 1 retrieve paintings of the "
        "same genre better than the plain SigLIP 1 image (mAP; read-outs picked on validation).")
    assert lic(dict(sc='SC8', set='mmart', encoder='clip_l14', aspect='style'), 0.01) == (
        "On MMArt test, c*'s style head on CLIP L/14 retrieves paintings by style better than PercepT's topic "
        "probabilities on CLIP L/14 (mAP).")


def test_family_sentences_merge_passing_metrics():
    cells = []
    for c in O.claim_cells():
        lo = 0.02
        if (c['sc'], c['set'], c['encoder'], c['aspect'], c['metric']) == ('SC1', 'artelingo', 'qwen', 'style', 'acc'):
            lo = -0.5
        cells.append(mk(c, lo=lo))
    fam = O.licensed_families(run(cells))
    got = [f for f in fam if f['sc'] == 'SC1' and f['set'] == 'artelingo' and f['encoder'] == 'qwen'
           and f['aspect'] == 'style']
    assert len(got) == 1
    assert got[0]['sentence'] == ('On ArtELingo held, c* on Qwen3-VL-Embedding-2B clusters paintings by style better '
                                  'than the plain Qwen3-VL-Embedding-2B image with as many clusters as classes (27), '
                                  'in AMI, NMI and ARI (5 seeds).')
    assert got[0]['metrics'] == ['ami', 'nmi', 'ari']
    f5 = [f for f in fam if f['sc'] == 'SC5' and f['set'] == 'mmart' and f['aspect'] == 'genre']
    assert len(f5) == 1 and f5[0]['sentence'] == (
        'c* on CLIP L/14 clusters paintings by genre better than CRL on CLIP L/14 on MMArt test (AMI with 20 '
        'clusters; AMI, NMI, ARI and ACC with as many clusters as classes, 10).')
    assert len([f for f in fam if f['sc'] == 'SC7']) == 24      # retrieval cells stay single
    assert not [f for f in O.licensed_families(run(cells, fail(ok_status(), 'gates', 'i_a'))) if f['sc'] == 'SC1']


def test_family_teacher_splits_by_wording():
    kw = dict(sc='SC3', set='mmart', encoder='qwen', aspect='style')
    cells = []
    for c in O.claim_cells():
        lo = 0.02
        if all(c[k] == v for k, v in kw.items()) and c['metric'] in ('ari', 'acc'):
            lo = -0.005            # LEVEL
        cells.append(mk(c, lo=lo))
    fam = [f for f in O.licensed_families(run(cells)) if all(f[k] == v for k, v in kw.items())]
    assert {f['wording']: f['metrics'] for f in fam} == {'better than': ['ami', 'nmi'],
                                                         'level with or better than': ['ari', 'acc']}


def test_spec_fragments_in_spec_text():
    flat = re.sub(r'\s+', ' ', SPEC.read_text())
    for frag in ('clusters paintings by', 'with as many clusters as classes', 'better than CRL on CLIP L/14 on',
                 '(AMI with 20 clusters;', 'retrieves paintings by', 'When the query names no aspect',
                 "c*'s joined heads on", 'retrieve paintings of the same', '(mAP; read-outs picked on validation)',
                 "better than PercepT's topic probabilities on", 'the photo model on', 'clusters photos by emotion',
                 'the Qwen image instructed for', 'level with or better than', '(5 seeds)'):
        assert frag in flat, frag


# ------------------------------------------------------------------------------------------------ summary
def test_summary_counts_per_sc_and_metric():
    cells = []
    for c in O.claim_cells():
        lo = 0.02
        if c['sc'] == 'SC1' and c['metric'] == 'acc':
            lo = -0.5 if c['encoder'] == 'qwen' else 0.0       # BELOW for qwen, LEVEL otherwise
        cells.append(mk(c, lo=lo))
    rows = run(cells, fail(ok_status(), 'gates', 'vi'))
    s = O.summary(rows)
    assert set(s) == {'SC1', 'SC2', 'SC3', 'SC4', 'SC5', 'SC6', 'SC7', 'SC8'}
    assert set(s['SC1']) == {'ami', 'nmi', 'ari', 'acc'} and set(s['SC6']) == {'map'}
    assert set(s['SC5']) == {'ami@20', 'ami', 'nmi', 'ari', 'acc'}
    a = s['SC1']['acc']
    assert (a['cells'], a['WIN'], a['LEVEL'], a['BELOW'], a['meets_bar'], a['inconclusive'], a['missing']) == \
        (24, 0, 18, 6, 0, 0, 0)
    assert s['SC1']['ami']['WIN'] == 24 and s['SC1']['ami']['meets_bar'] == 24
    assert s['SC7']['map']['inconclusive'] == 24 and s['SC7']['map']['meets_bar'] == 0
    assert (s['SC8']['map']['cells'], s['SC8']['map']['inconclusive'], s['SC8']['map']['meets_bar']) == (24, 21, 3)
    assert s['SC6']['map']['inconclusive'] == 3 and s['SC6']['map']['meets_bar'] == 3
    assert sum(v['cells'] for sc in s.values() for v in sc.values()) == 412
    for sc in s.values():
        for v in sc.values():
            assert v['cells'] == v['decided'] + v['inconclusive'] + v['missing']


# ------------------------------------------------------------------------------------------------ files
def test_decide_round_trip(tmp_path):
    cells_p, status_p, out_p = tmp_path / 'cells.json', tmp_path / 'status.json', tmp_path / 'outcomes.json'
    cells_p.write_text(json.dumps(all_cells()))
    status_p.write_text(json.dumps(ok_status()))
    res = O.decide(cells_p, status_p, out_p, smoke=True)
    back = json.loads(out_p.read_text())
    assert len(back['cells']) == 412 and back['summary']['SC1']['ami']['WIN'] == 24
    assert len(back['licensed']) > 0 and res['cells'][0]['id'] == back['cells'][0]['id']
    assert back['status'] == ok_status()
    cells_p.write_text(json.dumps({'a': 1}))
    with pytest.raises(AssertionError, match='list'):
        O.decide(cells_p, status_p, out_p, smoke=True)
