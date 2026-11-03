"""Tests of pp_outcome (r19 ticket 11): cell outcomes at the boundaries, the five verdict kinds, the D17 teacher claims,
every row of the spec's section 3.5 voids table, the five-runs rule, the refusals, the D15 scenario checks, the licensed
and limit sentences against the spec text, the CLI.

The cell universe below is written out here from the spec's section 6 table, independently of pp_outcome."""
import copy
import json
import re

import pytest

import pp_paths as PP
import pp_outcome as O

SPEC = PP.REPO / 'docs/superpowers/specs/2026-11-03-past-picks-design.md'
NEW = ('dinov2_large', 'clip_b32', 'clip_b16')
OLD = ('qwen', 'siglip2', 'siglip1', 'clip_l14')
ALL7 = NEW + OLD
ASP = ('emotion', 'style', 'genre')
AE, AF, MM, SA = 'artelingo_held', 'affection_test', 'mmart_test', 'semart_new'
WIN, LEVEL, BELOW, NOT_WIN = 'WIN', 'LEVEL', 'BELOW', 'not WIN'
LO = {WIN: 0.01, LEVEL: -0.002, BELOW: -0.05, NOT_WIN: -0.05}
R = 0.01
COUNTS = dict(mmart_n=16384, semart_n=11159, semart_artists=2717)
TEACHER = ('SC6', 'SC10', 'SC14')
TASK = {'SC1': 'clu', 'SC2': 'ret', 'SC3': 'clu', 'SC4': 'clu', 'SC5': 'ret', 'SC7': 'clu', 'SC8': 'ret', 'SC9': 'clu',
        'SC11': 'clu', 'SC12': 'ret', 'SC13': 'clu'}


def full_keys():
    """(sc, enc, aspect, task, data) of every verdict and teacher cell of the round (137 cells)."""
    keys = []
    for e in NEW:
        keys += [(sc, e, a, TASK[sc], AE) for sc in ('SC1', 'SC2', 'SC3') for a in ASP]
        keys += [('SC6', e, a, t, AE) for a in ASP for t in ('clu', 'ret')]
        keys += [('SC4', e, 'emotion', 'clu', AF), ('SC5', e, 'emotion', 'ret', AF),
                 ('SC6', e, 'emotion', 'clu', AF), ('SC6', e, 'emotion', 'ret', AF)]
        keys += [(sc, e, a, TASK[sc], MM) for sc in ('SC7', 'SC8', 'SC9') for a in ASP]
        keys += [('SC10', e, a, t, MM) for a in ASP for t in ('clu', 'ret')]
    for e in ALL7:
        keys += [(sc, e, 'genre', TASK[sc], SA) for sc in ('SC11', 'SC12', 'SC13')]
        keys += [('SC14', e, 'genre', t, SA) for t in ('clu', 'ret')]
    return keys


FULL = full_keys()


def mkcell(key, outcome=WIN, **kw):
    sc, enc, aspect, task, data = key
    r = R if sc in TEACHER else None
    c = dict(sc=sc, encoder=enc, aspect=aspect, task=task, data=data, candidate='c', comparator='k',
             candidate_value=0.5, comparator_value=0.4, d=LO[outcome] + 0.01, lo=LO[outcome], hi=LO[outcome] + 0.02,
             r=r, r_source='t', n_draws=PP.N_DRAWS[task], seeds=[0, 1, 2, 3, 4],
             n_rows={MM: 16384, SA: 11159}.get(data, 12281))
    c.update(kw)
    return c


def mkcost(enc, c=(10.0, 11.0, 12.0), t=(200.0, 210.0, 220.0), **kw):
    d = dict(sc='SC15', encoder=enc, cstar_ms=list(c), teacher_ms=list(t), cstar_ms_max=max(c),
             teacher_ms_min=min(t), cstar_ms_mean=sum(c) / 3, teacher_ms_mean=sum(t) / 3, cstar_ms_e2e=15.0,
             teacher_ms_e2e=260.0, repeats_discarded=0)
    d.update(kw)
    return d


def cells_of(over=None, drop=(), cost_over=None):
    """All 137 cells plus 7 cost cells, all WIN; `over` {(sc, enc, aspect, task, data) | (sc, enc, aspect): outcome}."""
    over = over or {}
    out = []
    for k in FULL:
        if k in drop:
            continue
        o = over.get(k, over.get(k[:3], WIN))
        out.append(mkcell(k, o))
    for e in ALL7:
        if ('SC15', e) not in drop:
            out.append(mkcost(e, **(cost_over or {}).get(e, {})))
    return out


def ok_status():
    ok = 'ok'
    return {
        'gates': {
            'i_store': {'dinov2_large': dict(mmart=ok, affection=ok, semart=ok), 'clip_b32': dict(mmart=ok, affection=ok),
                        'clip_b16': dict(mmart=ok, affection=ok, semart=ok), 'siglip1': dict(semart=ok),
                        'clip_l14': dict(semart=ok)},
            'i_art_parity': {e: ok for e in NEW + ('siglip1', 'clip_l14')},
            'i_semart_parity': {'clip_b32': ok, 'qwen': ok, 'siglip2': ok},
            'i_semart_genre_word': ok, 'i_aff_l14_parity': ok, 'ii': ok, 'iii': {'r10': ok, 'r14': ok}, 'iv': ok,
            'v': {e: ok for e in NEW}, 'vi': ok, 'vii': {'artelingo': ok, 'affection': ok, 'mmart': ok}, 'viii': ok},
        'runs': {e: dict(cstar=ok, photo=ok, matched=ok) for e in NEW}}


def failed(*path):
    s = ok_status()
    d = s
    for p in path[:-1]:
        d = d[p]
    d[path[-1]] = 'failed'
    return s


def decide_all(cs, status=None, counts=COUNTS, smoke=False):
    """O.decide with the cost cells split off the list."""
    return O.decide([c for c in cs if c['sc'] != 'SC15'], status or ok_status(), [c for c in cs if c['sc'] == 'SC15'],
                    counts=counts, smoke=smoke)


def run(over=None, status=None, drop=(), cost_over=None, smoke=False, counts=COUNTS):
    return decide_all(cells_of(over, drop, cost_over), status, counts, smoke)


def kind_of(key):
    return {AE: 'artelingo', AF: 'photos', MM: 'mmart', SA: 'semart'}[key[4]]


def void_view(res):
    """(verdicts that are inconclusive as {(kind, enc)}, cells whose status is inconclusive as a set of keys)."""
    v = {(k, e) for k, d in res['verdicts'].items() for e, x in d.items() if x['verdict'] == 'inconclusive'}
    c = {(x['sc'], x['encoder'], x['aspect'], x['task'], x['data']) for x in res['cells'] if x['status'] == 'inconclusive'}
    c |= {('SC15', x['encoder']) for x in res['cost_cells'] if x['status'] == 'inconclusive'}
    return v, c


def vset(kinds, encs):
    return {(k, e) for k in kinds for e in encs}


def cset(*, encs, datas, scs=None, tasks=None):
    return {k for k in FULL if k[1] in encs and k[4] in datas and (scs is None or k[0] in scs)
            and (tasks is None or k[3] in tasks)}


ALL_DATA = (AE, AF, MM, SA)
NEW_VERDICTS = vset(('artelingo', 'photos', 'mmart', 'semart'), NEW)


# ------------------------------------------------------------------------------------------------ cell outcome
def test_cell_outcome_boundaries():
    assert O.cell_outcome(dict(lo=1e-9, r=None)) == WIN
    assert O.cell_outcome(dict(lo=0.0, r=None)) == NOT_WIN              # lo > 0 is strict, and no unit means no LEVEL
    assert O.cell_outcome(dict(lo=0.0, r=0.01)) == LEVEL
    assert O.cell_outcome(dict(lo=-0.0099, r=0.01)) == LEVEL
    assert O.cell_outcome(dict(lo=-0.01, r=0.01)) == BELOW              # lo > -r is strict
    assert O.cell_outcome(dict(lo=-0.02, r=0.01)) == BELOW


# ------------------------------------------------------------------------------------------------ verdict kinds
@pytest.mark.parametrize('kind,scs,data', [('artelingo', ('SC1', 'SC2', 'SC3'), AE), ('mmart', ('SC7', 'SC8', 'SC9'), MM)])
def test_paintings_kinds_all_branches(kind, scs, data):
    s1, s2, s3 = scs
    e = 'clip_b32'
    assert run()['verdicts'][kind][e]['verdict'] == 'holds'
    for a in ASP:                                           # one PercepT cell misses -> qualified
        assert run({(s3, e, a): NOT_WIN})['verdicts'][kind][e]['verdict'] == 'qualified'
    r = run({(s3, e, a): NOT_WIN for a in ASP})
    assert r['verdicts'][kind][e]['verdict'] == 'qualified'         # all three PercepT cells miss, frozen cells win
    for sc in (s1, s2):                                     # one frozen-image cell misses -> fails
        for a in ASP:
            assert run({(sc, e, a): NOT_WIN})['verdicts'][kind][e]['verdict'] == 'fails'
    assert run({(s1, e, 'style'): NOT_WIN, (s3, e, 'style'): NOT_WIN})['verdicts'][kind][e]['verdict'] == 'fails'
    assert run({(s3, e, 'style'): NOT_WIN})['verdicts'][kind][e]['go'] == 'GO for the qualified claim'
    assert run({(s2, e, 'genre'): NOT_WIN})['verdicts'][kind][e]['go'] == 'NO-GO'
    assert run()['verdicts'][kind][e]['go'] == 'GO'


def test_mmart_cells_are_not_read_as_artelingo_cells():
    # a miss on an ArtELingo cell leaves the MMArt verdict alone, and the other way round (D15 (4))
    r = run({('SC1', 'clip_b32', 'style'): NOT_WIN})
    assert r['verdicts']['artelingo']['clip_b32']['verdict'] == 'fails'
    assert r['verdicts']['mmart']['clip_b32']['verdict'] == 'holds'
    r = run({('SC8', 'clip_b32', 'style'): NOT_WIN})
    assert r['verdicts']['mmart']['clip_b32']['verdict'] == 'fails'
    assert r['verdicts']['artelingo']['clip_b32']['verdict'] == 'holds'


def test_photos_kind():
    e = 'dinov2_large'
    assert run()['verdicts']['photos'][e]['verdict'] == 'holds'
    assert run({('SC4', e, 'emotion'): NOT_WIN})['verdicts']['photos'][e]['verdict'] == 'fails'
    assert run({('SC5', e, 'emotion'): NOT_WIN})['verdicts']['photos'][e]['verdict'] == 'fails'
    assert run({('SC5', e, 'emotion'): NOT_WIN})['verdicts']['photos'][e]['go'] == 'NO-GO'


def test_semart_kind_all_branches():
    e = 'siglip1'
    assert run()['verdicts']['semart'][e]['verdict'] == 'holds'
    assert run({('SC13', e, 'genre'): NOT_WIN})['verdicts']['semart'][e]['verdict'] == 'qualified'
    assert run({('SC11', e, 'genre'): NOT_WIN})['verdicts']['semart'][e]['verdict'] == 'fails'
    assert run({('SC12', e, 'genre'): NOT_WIN})['verdicts']['semart'][e]['verdict'] == 'fails'
    assert run({('SC11', e, 'genre'): NOT_WIN, ('SC13', e, 'genre'): NOT_WIN})['verdicts']['semart'][e]['verdict'] == 'fails'
    assert run({('SC12', e, 'genre'): NOT_WIN, ('SC13', e, 'genre'): NOT_WIN})['verdicts']['semart'][e]['verdict'] == 'fails'
    assert set(run()['verdicts']['semart']) == set(ALL7)


def test_cost_kind_boundaries():
    r = run()
    assert {e: r['verdicts']['cost'][e]['verdict'] for e in ALL7} == {e: 'holds' for e in ALL7}
    # c* max 100 against teacher min 100: strict, so it fails
    r = run(cost_over={'qwen': dict(c=(90.0, 95.0, 100.0), t=(100.0, 110.0, 120.0))})
    assert r['verdicts']['cost']['qwen']['verdict'] == 'fails'
    # one slow c* repeat decides, although the mean is far below
    r = run(cost_over={'qwen': dict(c=(10.0, 10.0, 201.0))})
    assert r['verdicts']['cost']['qwen']['verdict'] == 'fails'
    # one fast teacher repeat decides
    r = run(cost_over={'qwen': dict(t=(12.5, 300.0, 300.0))})
    assert r['verdicts']['cost']['qwen']['verdict'] == 'holds'
    r = run(cost_over={'qwen': dict(t=(11.5, 300.0, 300.0))})
    assert r['verdicts']['cost']['qwen']['verdict'] == 'fails'
    assert r['verdicts']['cost']['qwen']['go'] == 'NO-GO'


def test_each_set_has_its_own_verdict():
    # D15 (4): MMArt fails for clip_b16 and SemArt qualified for it; ArtELingo and photos hold
    r = run({('SC7', 'clip_b16', 'genre'): NOT_WIN, ('SC13', 'clip_b16', 'genre'): NOT_WIN})
    v = {k: r['verdicts'][k]['clip_b16']['verdict'] for k in r['verdicts']}
    assert v == {'artelingo': 'holds', 'photos': 'holds', 'mmart': 'fails', 'semart': 'qualified', 'cost': 'holds'}


# ------------------------------------------------------------------------------------------------ D15 scenarios
def test_scenario_1_dinov2_level_with_percept_in_style_clustering_is_qualified_with_limit_sentence():
    # the cell is a WIN-only bar (r null): a lower bound of -0.002 is 'not WIN'
    r = run({('SC3', 'dinov2_large', 'style'): NOT_WIN})
    v = r['verdicts']['artelingo']['dinov2_large']
    assert v['verdict'] == 'qualified' and v['go'] == 'GO for the qualified claim'
    assert v['sentence'] and 'DINOv2' in v['sentence']
    assert 'DINOv2' in v['limit'] and 'SC3' in v['limit'] and 'style' in v['limit'] and 'PercepT' in v['limit']
    assert 'emotion' not in v['limit'] and 'SC1' not in v['limit']          # names the missing cell only


def test_scenario_3_b32_holds_on_paintings_and_fails_on_photos():
    r = run({('SC4', 'clip_b32', 'emotion'): NOT_WIN})
    assert r['verdicts']['artelingo']['clip_b32']['verdict'] == 'holds'
    assert r['verdicts']['mmart']['clip_b32']['verdict'] == 'holds'
    p = r['verdicts']['photos']['clip_b32']
    assert p['verdict'] == 'fails' and p['sentence'] is None
    assert 'CLIP B/32' in p['limit'] and 'SC4' in p['limit'] and 'photos' in p['limit']


# ------------------------------------------------------------------------------------------------ teacher claims (D17)
def claim_of(res, key):
    sc, enc, aspect, task, data = key
    rows = [x for x in res['claims'][enc] if (x['sc'], x['aspect'], x['task'], x['data']) == (sc, aspect, task, data)]
    assert len(rows) == 1
    return rows[0]


@pytest.mark.parametrize('enc', OLD)
def test_d17_cells_win_is_better_than_level_is_level_with_or_better(enc):
    key = ('SC14', enc, 'genre', 'ret', SA)
    assert key[1:] + () and (enc, 'genre', 'ret', SA) in PP.D17
    c = claim_of(run({key: WIN}), key)
    assert c['claim'] == 'better than' and c['d17'] is True and 'is better than the Qwen image' in c['sentence']
    c = claim_of(run({key: LEVEL}), key)
    assert c['claim'] == 'level with or better than' and 'level with or better than' in c['sentence']
    c = claim_of(run({key: BELOW}), key)
    assert c['claim'] is None and c['sentence'] is None and c['outcome'] == BELOW and c['status'] == 'decided'


def test_non_d17_cells_never_license_better_than():
    keys = [k for k in FULL if k[0] in TEACHER and (k[1], k[2], k[3], k[4]) not in PP.D17]
    assert len(keys) == 3 * (6 + 2 + 6) + 3 * 2 + 4 * 1                     # new three: SC6 8, SC10 6; SC14 clu x 7, ret x 3
    for k in keys:
        assert claim_of(run({k: WIN}), k)['claim'] == 'level with or better than', k
        assert claim_of(run({k: LEVEL}), k)['claim'] == 'level with or better than', k
        assert claim_of(run({k: BELOW}), k)['claim'] is None, k


def test_semart_genre_clustering_of_old_encoder_is_not_a_d17_cell():
    k = ('SC14', 'qwen', 'genre', 'clu', SA)
    assert claim_of(run({k: WIN}), k)['claim'] == 'level with or better than'


def test_new_three_semart_genre_map_is_not_a_d17_cell():
    for e in NEW:
        k = ('SC14', e, 'genre', 'ret', SA)
        assert claim_of(run({k: WIN}), k)['claim'] == 'level with or better than'


def test_teacher_claim_sentences_name_model_aspect_task_and_set():
    c = claim_of(run(), ('SC6', 'clip_b32', 'style', 'clu', AE))
    assert c['sentence'] == ('c* on CLIP B/32 is level with or better than the Qwen image instructed for style in '
                             'clustering on 12,281 held ArtELingo paintings.')
    c = claim_of(run(), ('SC6', 'clip_b32', 'emotion', 'ret', AF))
    assert c['sentence'] == ('The photo model on CLIP B/32 is level with or better than the Qwen image instructed for '
                             'emotion in retrieval on 8,501 held Affection photos.')
    c = claim_of(run(), ('SC10', 'dinov2_large', 'genre', 'ret', MM))
    assert c['sentence'] == ('c* on DINOv2 is level with or better than the Qwen image instructed for genre in '
                             'retrieval on MMArt.')
    c = claim_of(run({('SC14', 'siglip2', 'genre', 'ret', SA): WIN}), ('SC14', 'siglip2', 'genre', 'ret', SA))
    assert c['sentence'] == ("c* on SigLIP 2 is better than the Qwen image instructed for genre in retrieval on "
                             "SemArt's new artists.")


def test_teacher_claims_do_not_move_verdicts():
    keys = [k for k in FULL if k[0] in TEACHER]
    r = run({k: BELOW for k in keys})
    assert all(d[e]['verdict'] in ('holds',) for d in r['verdicts'].values() for e in d)
    assert all(c['claim'] is None for rows in r['claims'].values() for c in rows)


# ------------------------------------------------------------------------------------------------ voids table
def test_all_ok_voids_nothing_and_counts_cells():
    r = run()
    assert void_view(r) == (set(), set())
    assert len(r['cells']) == 137 and len(r['cost_cells']) == 7
    assert sum(len(v) for v in r['claims'].values()) == 3 * 14 + 7 * 2


def check_void(status, verdicts, cells, drop_voided=False):
    """Expected inconclusive verdicts and cells, and the run with the voided cells left out must still work."""
    r = run(status=status)
    assert void_view(r) == (verdicts, cells)
    if drop_voided:
        r2 = run(status=status, drop=tuple(cells))
        assert void_view(r2) == (verdicts, set()) or void_view(r2)[0] == verdicts
    return r


@pytest.mark.parametrize('enc,s,kind,datas', [
    ('dinov2_large', 'mmart', 'mmart', (MM,)), ('dinov2_large', 'affection', 'photos', (AF,)),
    ('dinov2_large', 'semart', 'semart', (SA,)), ('clip_b32', 'mmart', 'mmart', (MM,)),
    ('clip_b32', 'affection', 'photos', (AF,)), ('clip_b16', 'mmart', 'mmart', (MM,)),
    ('clip_b16', 'affection', 'photos', (AF,)), ('clip_b16', 'semart', 'semart', (SA,)),
    ('siglip1', 'semart', 'semart', (SA,)), ('clip_l14', 'semart', 'semart', (SA,))])
def test_void_row_i_store_voids_that_encoder_on_that_set_only(enc, s, kind, datas):
    check_void(failed('gates', 'i_store', enc, s), {(kind, enc)}, cset(encs=(enc,), datas=datas), drop_voided=True)


def test_void_row_i_semart_genre_word_voids_sc14_only():
    r = check_void(failed('gates', 'i_semart_genre_word'), set(), cset(encs=ALL7, datas=(SA,), scs=('SC14',)),
                   drop_voided=True)
    assert all(x['verdict'] == 'holds' for x in r['verdicts']['semart'].values())      # SemArt verdicts stand
    assert all(c['claim'] == 'inconclusive' for e in ALL7 for c in r['claims'][e] if c['sc'] == 'SC14')


@pytest.mark.parametrize('enc,kinds,datas', [
    ('dinov2_large', ('mmart', 'photos', 'semart'), (MM, AF, SA)),
    ('clip_b32', ('mmart', 'photos'), (MM, AF)),
    ('clip_b16', ('mmart', 'photos', 'semart'), (MM, AF, SA)),
    ('siglip1', ('semart',), (SA,)), ('clip_l14', ('semart',), (SA,))])
def test_void_row_i_art_parity_voids_every_set_with_a_new_store_of_x(enc, kinds, datas):
    # ArtELingo's own set stands: it reads the old ArtELingo stores
    check_void(failed('gates', 'i_art_parity', enc), vset(kinds, (enc,)), cset(encs=(enc,), datas=datas),
               drop_voided=True)


@pytest.mark.parametrize('enc', ['clip_b32', 'qwen', 'siglip2'])
def test_void_row_i_semart_parity_voids_every_semart_verdict_and_sc14(enc):
    check_void(failed('gates', 'i_semart_parity', enc), vset(('semart',), ALL7), cset(encs=ALL7, datas=(SA,)),
               drop_voided=True)


def test_void_row_i_aff_l14_parity_voids_the_new_three_photos():
    check_void(failed('gates', 'i_aff_l14_parity'), vset(('photos',), NEW), cset(encs=NEW, datas=(AF,)),
               drop_voided=True)


def test_void_row_ii_voids_everything_of_the_new_three():
    check_void(failed('gates', 'ii'), NEW_VERDICTS, cset(encs=NEW, datas=ALL_DATA), drop_voided=True)


@pytest.mark.parametrize('fam,old', [('r10', ('qwen',)), ('r14', ('siglip2', 'siglip1', 'clip_l14'))])
def test_void_row_iii_voids_new_three_and_semart_of_the_failed_family(fam, old):
    check_void(failed('gates', 'iii', fam), NEW_VERDICTS | vset(('semart',), old),
               cset(encs=NEW, datas=ALL_DATA) | cset(encs=old, datas=(SA,)), drop_voided=True)


def test_void_row_iv_voids_everything_except_cost():
    verdicts = NEW_VERDICTS | vset(('semart',), OLD)
    r = check_void(failed('gates', 'iv'), verdicts, set(FULL), drop_voided=True)
    assert all(x['verdict'] == 'holds' for x in r['verdicts']['cost'].values())


@pytest.mark.parametrize('enc', NEW)
def test_void_row_v_voids_x_verdicts_but_not_photos_nor_teacher_claims(enc):
    r = check_void(failed('gates', 'v', enc), vset(('artelingo', 'mmart', 'semart'), (enc,)),
                   cset(encs=(enc,), datas=(AE, MM, SA), scs=('SC3', 'SC9', 'SC13')), drop_voided=True)
    assert r['verdicts']['photos'][enc]['verdict'] == 'holds'
    assert all(c['claim'] == 'level with or better than' and c['status'] == 'decided' for c in r['claims'][enc])


def test_void_row_vi_voids_every_semart_verdict_and_sc14():
    check_void(failed('gates', 'vi'), vset(('semart',), ALL7), cset(encs=ALL7, datas=(SA,)), drop_voided=True)


def test_void_row_vii_artelingo_voids_retrieval_of_new_three_on_artelingo_and_semart_not_ami_not_old_encoders():
    r = check_void(failed('gates', 'vii', 'artelingo'), vset(('artelingo', 'semart'), NEW),
                   cset(encs=NEW, datas=(AE, SA), tasks=('ret',)), drop_voided=True)
    assert r['verdicts']['semart']['qwen']['verdict'] == 'holds'
    # AMI teacher claims stand
    assert claim_of(r, ('SC6', 'clip_b32', 'style', 'clu', AE))['claim'] == 'level with or better than'
    assert claim_of(r, ('SC14', 'clip_b32', 'genre', 'clu', SA))['claim'] == 'level with or better than'
    assert claim_of(r, ('SC6', 'clip_b32', 'style', 'ret', AE))['claim'] == 'inconclusive'


def test_void_row_vii_affection_voids_photo_retrieval_of_new_three():
    r = check_void(failed('gates', 'vii', 'affection'), vset(('photos',), NEW),
                   cset(encs=NEW, datas=(AF,), tasks=('ret',)), drop_voided=True)
    assert claim_of(r, ('SC6', 'clip_b16', 'emotion', 'clu', AF))['claim'] == 'level with or better than'


def test_void_row_vii_mmart_voids_mmart_retrieval_of_new_three():
    r = check_void(failed('gates', 'vii', 'mmart'), vset(('mmart',), NEW), cset(encs=NEW, datas=(MM,), tasks=('ret',)),
                   drop_voided=True)
    assert claim_of(r, ('SC10', 'clip_b16', 'style', 'clu', MM))['claim'] == 'level with or better than'


def test_void_row_viii_voids_cost_only():
    r = check_void(failed('gates', 'viii'), vset(('cost',), ALL7), {('SC15', e) for e in ALL7})
    r = run(status=failed('gates', 'viii'), drop=tuple(('SC15', e) for e in ALL7))      # cost cells may be absent then
    assert all(x['verdict'] == 'inconclusive' and x['sentence'] is None for x in r['verdicts']['cost'].values())


@pytest.mark.parametrize('enc', NEW)
def test_runs_row_cstar_failed_voids_x_everywhere_except_cost(enc):
    s = ok_status()
    s['runs'][enc].update(cstar='failed', photo='failed')           # the photo model starts from the c* checkpoint
    check_void(s, vset(('artelingo', 'photos', 'mmart', 'semart'), (enc,)), cset(encs=(enc,), datas=ALL_DATA),
               drop_voided=True)


@pytest.mark.parametrize('enc', NEW)
def test_runs_row_photo_failed_voids_the_photo_verdict_only(enc):
    check_void(failed('runs', enc, 'photo'), {('photos', enc)}, cset(encs=(enc,), datas=(AF,)), drop_voided=True)


def test_runs_row_cstar_failed_with_photo_runs_ok_is_refused():
    with pytest.raises(AssertionError, match='photo'):
        run(status=failed('runs', 'clip_b32', 'cstar'))


@pytest.mark.parametrize('enc', NEW)
def test_runs_row_matched_failed_acts_like_gate_v(enc):
    check_void(failed('runs', enc, 'matched'), vset(('artelingo', 'mmart', 'semart'), (enc,)),
               cset(encs=(enc,), datas=(AE, MM, SA), scs=('SC3', 'SC9', 'SC13')), drop_voided=True)


def test_two_failures_add_their_reasons():
    s = failed('gates', 'ii')
    s['gates']['vi'] = 'failed'
    r = run(status=s)
    reasons = r['verdicts']['semart']['clip_b32']['reason']
    assert len(reasons) == 2 and any('ii' in x for x in reasons) and any('vi' in x for x in reasons)
    assert r['verdicts']['semart']['qwen']['reason'] and len(r['verdicts']['semart']['qwen']['reason']) == 1


def test_voided_verdict_is_inconclusive_never_qualified_and_gets_no_sentence():
    # the cells alone would give 'qualified' (SC3 style misses); a failed gate makes the verdict inconclusive
    r = run({('SC3', 'clip_b16', 'style'): NOT_WIN}, status=failed('gates', 'ii'))
    v = r['verdicts']['artelingo']['clip_b16']
    assert v['verdict'] == 'inconclusive' and v['go'] != 'GO for the qualified claim'
    assert v['sentence'] is None and v['limit'] is None
    # a voided fails stays inconclusive, not NO-GO
    r = run({('SC1', 'clip_b16', 'style'): NOT_WIN}, status=failed('gates', 'ii'))
    v = r['verdicts']['artelingo']['clip_b16']
    assert v['verdict'] == 'inconclusive' and 'NO-GO' not in v['go'] and v['limit'] is None


def test_voided_cells_keep_their_rows_with_outcome_and_status_inconclusive():
    r = run({('SC1', 'clip_b32', 'style'): NOT_WIN}, status=failed('gates', 'ii'))
    rows = [x for x in r['cells'] if (x['sc'], x['encoder'], x['aspect']) == ('SC1', 'clip_b32', 'style')]
    assert len(rows) == 1 and rows[0]['status'] == 'inconclusive' and rows[0]['outcome'] == NOT_WIN
    assert rows[0]['void'] and 'ii' in rows[0]['void'][0]
    ok = [x for x in r['cells'] if x['encoder'] == 'qwen' and x['sc'] == 'SC11'][0]
    assert ok['status'] == 'decided' and ok['void'] == []


def test_cell_missing_without_a_failed_gate_is_refused():
    for k in (('SC1', 'clip_b32', 'style', 'clu', AE), ('SC6', 'dinov2_large', 'genre', 'ret', AE),
              ('SC12', 'qwen', 'genre', 'ret', SA), ('SC14', 'siglip1', 'genre', 'clu', SA),
              ('SC5', 'clip_b16', 'emotion', 'ret', AF), ('SC10', 'clip_b16', 'style', 'ret', MM)):
        with pytest.raises(AssertionError, match='missing'):
            run(drop=(k,))
    with pytest.raises(AssertionError, match='missing'):
        run(drop=(('SC15', 'qwen'),))


def test_cell_missing_is_excused_only_by_the_gate_that_covers_it():
    # gate v covers SC3/9/13, not SC1
    with pytest.raises(AssertionError, match='missing'):
        run(status=failed('gates', 'v', 'clip_b32'), drop=(('SC1', 'clip_b32', 'style', 'clu', AE),))
    run(status=failed('gates', 'v', 'clip_b32'), drop=(('SC3', 'clip_b32', 'style', 'clu', AE),))
    # gate vii (mmart) covers retrieval cells, not AMI cells
    with pytest.raises(AssertionError, match='missing'):
        run(status=failed('gates', 'vii', 'mmart'), drop=(('SC7', 'clip_b32', 'style', 'clu', MM),))
    # the genre_word gate covers SC14, not SC13
    with pytest.raises(AssertionError, match='missing'):
        run(status=failed('gates', 'i_semart_genre_word'), drop=(('SC13', 'qwen', 'genre', 'clu', SA),))


# ------------------------------------------------------------------------------------------------ refusals
def test_refuses_duplicate_cells():
    cs = cells_of()
    cs.append(copy.deepcopy(cs[3]))
    with pytest.raises(AssertionError, match='duplicate'):
        decide_all(cs)
    cs = cells_of() + [mkcost('qwen')]
    with pytest.raises(AssertionError, match='duplicate'):
        decide_all(cs)


def test_refuses_cells_outside_the_rule():
    extra = mkcell(('SC1', 'qwen', 'emotion', 'clu', AE))                  # Qwen has no ArtELingo cell in this round
    with pytest.raises(AssertionError):
        decide_all(cells_of() + [extra])
    extra = mkcell(('SC11', 'qwen', 'emotion', 'clu', SA))                 # SemArt is genre only
    with pytest.raises(AssertionError):
        decide_all(cells_of() + [extra])


def test_refuses_wrong_draw_counts_outside_smoke_and_allows_them_in_smoke():
    k = ('SC1', 'clip_b32', 'style', 'clu', AE)
    for n in (1999, 10000, 40):
        cs = [mkcell(x, n_draws=n) if x == k else mkcell(x) for x in FULL]
        cs += [mkcost(e) for e in ALL7]
        with pytest.raises(AssertionError, match='n_draws'):
            decide_all(cs)
    k = ('SC2', 'clip_b32', 'style', 'ret', AE)
    for n in (2000, 9999):
        cs = [mkcell(x, n_draws=n) if x == k else mkcell(x) for x in FULL] + [mkcost(e) for e in ALL7]
        with pytest.raises(AssertionError, match='n_draws'):
            decide_all(cs)
    cs = [mkcell(x, n_draws=40 if x[3] == 'clu' else 100) for x in FULL] + [mkcost(e) for e in ALL7]
    decide_all(cs, smoke=True)


def test_refuses_r_on_win_only_bars_and_missing_r_on_teacher_cells():
    for k in FULL:
        if k[0] in TEACHER:
            with pytest.raises(AssertionError, match='r'):
                O.check_cell(mkcell(k, r=None))
            with pytest.raises(AssertionError, match='r'):
                O.check_cell(mkcell(k, r=0.0))
            with pytest.raises(AssertionError, match='r'):
                O.check_cell(mkcell(k, r=float('nan')))
            O.check_cell(mkcell(k))
        else:
            with pytest.raises(AssertionError, match='r'):
                O.check_cell(mkcell(k, r=0.01))
            O.check_cell(mkcell(k))


def test_refuses_malformed_cells():
    k = ('SC1', 'clip_b32', 'style', 'clu', AE)
    c = mkcell(k)
    del c['n_rows']
    with pytest.raises(AssertionError, match='lacks'):
        O.check_cell(c)
    for bad in (dict(lo=float('nan')), dict(d=float('inf')), dict(lo=1.0, hi=0.5), dict(task='ret'),
                dict(data=MM), dict(sc='SC99'), dict(encoder='nope'), dict(aspect='school')):
        with pytest.raises(AssertionError):
            O.check_cell(mkcell(k, **bad))
    with pytest.raises(AssertionError):
        O.check_cell(mkcell(('SC4', 'clip_b32', 'style', 'clu', AF)))      # photos are emotion only
    with pytest.raises(AssertionError):
        O.check_cell(mkcell(('SC6', 'clip_b32', 'style', 'clu', AF)))      # photo teacher cells are emotion only
    with pytest.raises(AssertionError):
        O.check_cell(mkcell(('SC11', 'clip_b32', 'style', 'clu', SA)))     # SemArt is genre only
    with pytest.raises(AssertionError):
        O.check_cell(mkcell(('SC1', 'qwen', 'style', 'clu', AE)))          # SC1 to SC10 are the new three only
    with pytest.raises(AssertionError):
        O.check_cell(mkcell(('SC6', 'clip_b32', 'style', 'clu', MM)))      # SC6 is ArtELingo or Affection


def test_refuses_malformed_cost_cells():
    for bad in (dict(cstar_ms=[1.0, 2.0]), dict(teacher_ms=[1.0, 2.0, float('nan')]), dict(cstar_ms_max=1.0),
                dict(teacher_ms_min=1.0), dict(cstar_ms_mean=1.0), dict(teacher_ms=[1.0, 2.0, -3.0]),
                dict(cstar_ms_e2e=float('nan'))):
        with pytest.raises(AssertionError):
            O.check_cost_cell(mkcost('qwen', **bad))
    O.check_cost_cell(mkcost('qwen'))


def test_refuses_malformed_status():
    for mutate in (lambda s: s['gates'].pop('viii'), lambda s: s['gates'].update(extra='ok'),
                   lambda s: s['gates'].update(ii='maybe'), lambda s: s['runs'].pop('clip_b16'),
                   lambda s: s['gates']['v'].pop('dinov2_large'), lambda s: s['gates']['iii'].pop('r10'),
                   lambda s: s['gates']['i_store'].pop('clip_b32'), lambda s: s['gates']['i_store']['clip_b32'].pop('mmart'),
                   lambda s: s['gates']['i_store'].update(qwen={'semart': 'ok'}),
                   lambda s: s['gates']['i_art_parity'].pop('clip_b16'),
                   lambda s: s['gates']['i_semart_parity'].pop('qwen'),
                   lambda s: s['gates']['vii'].pop('mmart'), lambda s: s['runs']['clip_b32'].pop('matched'),
                   lambda s: s.pop('runs')):
        s = ok_status()
        mutate(s)
        with pytest.raises(AssertionError):
            O.check_status(s)
    O.check_status(ok_status())


def test_refuses_counts_that_differ_from_the_cells_or_the_spec():
    with pytest.raises(AssertionError, match='mmart'):
        run(counts=dict(COUNTS, mmart_n=16000))
    with pytest.raises(AssertionError, match='semart'):
        run(counts=dict(COUNTS, semart_n=11162))
    cs = [mkcell(k, n_rows=16000) if k[4] == MM else mkcell(k) for k in FULL] + [mkcost(e) for e in ALL7]
    with pytest.raises(AssertionError, match='mmart_n'):                # cells and count agree, but not with the spec
        decide_all(cs, counts=dict(COUNTS, mmart_n=16000))
    decide_all(cs, counts=dict(COUNTS, mmart_n=16000), smoke=True)
    run(counts=dict(COUNTS, mmart_n=16000, semart_n=11162), smoke=True)


# ------------------------------------------------------------------------------------------------ licensed sentences
def norm(text):
    return re.sub(r'\s+', ' ', text.replace('≥', '>=').replace('>=', '>=')).strip()


SPEC_TEXT = norm(SPEC.read_text())
R14_TEXT = norm((PP.MAIN / 'docs/superpowers/specs/2026-11-02-backbones-sweep-design.md').read_text())
R15_TEXT = norm((PP.MAIN / 'docs/superpowers/specs/2026-11-02-mmart-test-design.md').read_text())


def test_semart_holds_sentence_follows_the_spec():
    s = run()['verdicts']['semart']['clip_l14']['sentence']
    assert s == (
        "On 11,159 paintings by 2,717 artists who have no painting in ArtELingo's train set (SemArt, from the Web "
        "Gallery of Art; artists matched by name; images at cosine ≥ 0.95 to an ArtELingo train or validation "
        "painting removed), frozen c* on CLIP L/14 clusters and retrieves paintings by type (SemArt's 10 classes, "
        "close to genre) better than the plain CLIP L/14 image, and clusters them better than PercepT on the CLIP L/14 "
        "image (AMI with 20 clusters; mAP with read-outs picked on ArtELingo validation; 5 seeds).")
    for frag in ("On <n> paintings by <m> artists who have no painting in ArtELingo's train set (SemArt, from the Web "
                 "Gallery of Art; artists matched by name; images at cosine ≥ 0.95 to an ArtELingo train or "
                 "validation painting removed), frozen c* on <X> clusters and retrieves paintings by type (SemArt's 10 "
                 "classes, close to genre) better than the plain <X> image, and clusters them better than PercepT on "
                 "the <X> image (AMI with 20 clusters; mAP with read-outs picked on ArtELingo validation; 5 seeds).",):
        assert norm(frag) in SPEC_TEXT


def test_semart_qualified_sentence_says_not_better_against_percept():
    r = run({('SC13', 'clip_b16', 'genre'): NOT_WIN})
    v = r['verdicts']['semart']['clip_b16']
    assert v['verdict'] == 'qualified'
    assert 'better than the plain CLIP B/16 image; against PercepT on the CLIP B/16 image it is not better' in v['sentence']
    assert 'clusters them better than PercepT' not in v['sentence']
    assert "against PercepT on the <X> image it is not better" in SPEC_TEXT
    assert 'CLIP B/16' in v['limit'] and 'SC13' in v['limit'] and 'SemArt' in v['limit']


def test_semart_fails_has_a_limit_sentence_and_no_licensed_sentence():
    v = run({('SC12', 'siglip2', 'genre'): NOT_WIN})['verdicts']['semart']['siglip2']
    assert v['verdict'] == 'fails' and v['sentence'] is None
    assert 'SigLIP 2' in v['limit'] and 'SC12' in v['limit'] and 'plain SigLIP 2 image' in v['limit']


def test_paintings_sentence_uses_the_item_16_encoder_sentence():
    enc_sentence = ("<X> is one of three encoders that failed or were unclear in an earlier held emotion test on "
                    "ArtELingo with an earlier version of our model")
    assert norm(enc_sentence) in SPEC_TEXT
    s = run()['verdicts']['artelingo']['dinov2_large']['sentence']
    assert s.startswith('On 12,281 held ArtELingo paintings, c*\'s recipe with the DINOv2 image as its only input at test')
    assert 'clusters and retrieves paintings by emotion, style and genre better than the plain DINOv2 image, and ' \
           'clusters them better than PercepT on the DINOv2 image' in s
    assert s.endswith('(AMI with 20 clusters; mAP with read-outs picked on validation; 5 seeds). DINOv2 is one of three '
                      'encoders that failed or were unclear in an earlier held emotion test on ArtELingo with an '
                      'earlier version of our model.')
    assert 'passed an earlier' not in s
    # the body is r14's text
    assert norm("clusters and retrieves paintings by emotion, style and genre better than the plain <X> image, and "
                "clusters them better than PercepT on the <X> image") in R14_TEXT


def test_paintings_qualified_sentence_names_the_aspects_where_percept_is_beaten():
    r = run({('SC3', 'clip_b32', 'style'): NOT_WIN})
    s = r['verdicts']['artelingo']['clip_b32']['sentence']
    assert 'better than the plain CLIP B/32 image; against PercepT on the CLIP B/32 image, better in emotion and genre' in s
    r = run({('SC3', 'clip_b32', a): NOT_WIN for a in ASP})
    s = r['verdicts']['artelingo']['clip_b32']['sentence']
    assert 'against PercepT on the CLIP B/32 image it is better in no aspect' in s


def test_mmart_sentence_uses_r15_head_and_the_item_16_encoder_sentence():
    s = run()['verdicts']['mmart']['clip_b16']['sentence']
    assert s.startswith('On 16,384 WikiArt paintings outside ArtELingo (MMArt; exact and near copies of ArtELingo train '
                        'and validation paintings removed), c*\'s recipe with the CLIP B/16 image as its only input at '
                        'test (frozen; never trained or picked on these paintings) clusters and retrieves paintings by '
                        'emotion (MMArt\'s labels), style and genre better than the plain CLIP B/16 image, and '
                        'clusters them better than PercepT on the CLIP B/16 image')
    assert s.endswith('(AMI with 20 clusters; mAP with read-outs picked on validation paintings with MMArt labels; 5 '
                      'seeds). CLIP B/16 is one of three encoders that failed or were unclear in an earlier held '
                      'emotion test on ArtELingo with an earlier version of our model.')
    assert 'one of four' not in s
    assert norm("On <n> WikiArt paintings outside ArtELingo (MMArt; exact and near copies of ArtELingo train and "
                "validation paintings removed), c*'s recipe with the <X> image as its only input at test (frozen; "
                "never trained or picked on these paintings) clusters and retrieves paintings by emotion (MMArt's "
                "labels), style and genre") in R15_TEXT
    q = run({('SC9', 'clip_b16', 'genre'): NOT_WIN})['verdicts']['mmart']['clip_b16']
    assert 'against PercepT on the CLIP B/16 image, better in emotion and style' in q['sentence']
    assert 'SC9' in q['limit'] and 'genre' in q['limit'] and 'MMArt' in q['limit']


def test_photos_sentence_is_r14s():
    s = run()['verdicts']['photos']['clip_b32']['sentence']
    assert s == ('On 8,501 held Affection photos, the same model on CLIP B/32, fine-tuned on Affection\'s unlabeled '
                 'train photos with the settings picked for Qwen, clusters and retrieves photos by emotion better '
                 'than the plain CLIP B/32 image.')
    assert norm("On 8,501 held Affection photos, the same model on <X>, fine-tuned on Affection's unlabeled train "
                "photos with the settings picked for Qwen, clusters and retrieves photos by emotion better than the "
                "plain <X> image.") in R14_TEXT


def test_cost_sentence_and_limit():
    r = run(cost_over={'qwen': dict(c=(20.0, 21.0, 22.0), t=(250.0, 260.0, 270.0), cstar_ms_e2e=30.0,
                                   teacher_ms_e2e=300.0)})
    s = r['verdicts']['cost']['qwen']['sentence']
    assert s == ("On one RTX 3090 at batch 32, c* on Qwen3-VL-Embedding-2B embeds a painting for emotion, style and "
                 "genre in 21.0 ms (one Qwen3-VL-Embedding-2B pass and three small heads), against 260.0 ms for the "
                 "Qwen image instructed per aspect (three passes); model time, images decoded beforehand (30.0 and "
                 "300.0 ms with decoding).")
    assert norm("On one RTX 3090 at batch 32, c* on <X> embeds a painting for emotion, style and genre in <a> ms (one "
                "<X> pass and three small heads), against <b> ms for the Qwen image instructed per aspect (three "
                "passes); model time, images decoded beforehand (<c> and <d> ms with decoding).") in SPEC_TEXT
    f = run(cost_over={'dinov2_large': dict(c=(10.0, 10.0, 201.0))})['verdicts']['cost']['dinov2_large']
    assert f['sentence'] is None and 'DINOv2' in f['limit'] and 'SC15' in f['limit']


def test_every_fails_and_qualified_verdict_has_a_limit_sentence_and_holds_has_none():
    over = {('SC3', 'clip_b32', 'style'): NOT_WIN, ('SC5', 'clip_b32', 'emotion'): NOT_WIN,
            ('SC8', 'clip_b32', 'genre'): NOT_WIN, ('SC13', 'clip_b32', 'genre'): NOT_WIN}
    r = run(over, cost_over={'clip_b32': dict(c=(300.0, 1.0, 1.0))})
    seen = set()
    for kind, d in r['verdicts'].items():
        for e, v in d.items():
            seen.add(v['verdict'])
            if v['verdict'] in ('fails', 'qualified'):
                assert v['limit'] and O.ENCODER_NAME[e] in v['limit'], (kind, e)
            else:
                assert v['limit'] is None
            if v['verdict'] in ('holds', 'qualified'):
                assert v['sentence']
            else:
                assert v['sentence'] is None
    assert seen == {'holds', 'qualified', 'fails'}


def test_limit_sentence_lists_every_missing_cell():
    r = run({('SC1', 'clip_b16', 'emotion'): NOT_WIN, ('SC2', 'clip_b16', 'genre'): NOT_WIN,
             ('SC3', 'clip_b16', 'style'): NOT_WIN})
    lim = r['verdicts']['artelingo']['clip_b16']['limit']
    for frag in ('SC1', 'SC2', 'SC3', 'emotion', 'genre', 'style', 'CLIP B/16', 'ArtELingo'):
        assert frag in lim
    assert 'fails' in lim


def test_placeholders_are_all_filled():
    r = run({('SC3', 'clip_b32', 'style'): NOT_WIN, ('SC13', 'qwen', 'genre'): NOT_WIN,
             ('SC9', 'dinov2_large', 'emotion'): NOT_WIN, ('SC4', 'clip_b16', 'emotion'): NOT_WIN})
    texts = []
    for d in r['verdicts'].values():
        for v in d.values():
            texts += [v['sentence'], v['limit']]
    texts += [c['sentence'] for rows in r['claims'].values() for c in rows]
    assert all('<' not in t and '>' not in t for t in texts if t)


# ------------------------------------------------------------------------------------------------ CLI and files
def test_cli_reads_the_cell_files_and_writes_stage4_outcomes(tmp_path, monkeypatch):
    parts = {'artelingo_clu': [], 'artelingo_ret': [], 'affection': [], 'mmart_clu': [], 'mmart_ret': [],
             'semart_clu': [], 'semart_ret': []}
    for k in FULL:
        c = mkcell(k, n_draws=40 if k[3] == 'clu' else 100)
        if k[4] == AE:
            parts['artelingo_' + k[3]].append(c)
        elif k[4] == AF:
            parts['affection'].append(c)
        elif k[4] == MM:
            parts['mmart_' + k[3]].append(c)
        else:
            parts['semart_' + k[3]].append(c)
    (tmp_path / 'stage2').mkdir()
    (tmp_path / 'stage3').mkdir()
    for p, cs in parts.items():
        (tmp_path / 'stage2' / f'cells_{p}.json').write_text(json.dumps(cs))
    (tmp_path / 'stage3' / 'cells_cost.json').write_text(json.dumps([mkcost(e) for e in ALL7]))
    status = tmp_path / 'status.json'
    status.write_text(json.dumps(ok_status()))
    O.main(['decide', '--root', str(tmp_path), '--status', str(status), '--smoke'])
    out = json.loads((tmp_path / 'stage4' / 'outcomes.json').read_text())
    assert out['smoke'] is True and len(out['cells']) == 137 and len(out['cost_cells']) == 7
    assert out['verdicts']['semart']['qwen']['verdict'] == 'holds'
    assert sorted(out['cell_files']) == sorted([f'cells_{p}.json' for p in parts] + ['cells_cost.json'])


def test_cli_refuses_a_stray_cells_file(tmp_path):
    (tmp_path / 'stage2').mkdir()
    (tmp_path / 'stage2' / 'cells_extra.json').write_text('[]')
    with pytest.raises(AssertionError, match='cells_extra'):
        O.load_cells(tmp_path)
