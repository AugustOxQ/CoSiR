"""Tests of mt_outcome (r15 ticket 06): cell outcomes at the boundaries, verdicts, every D17 row, the SC4 claims (D7
and non-D7), the refusals, the licensed sentences against the spec text, the CLI."""
import copy
import json
import re

import pytest

import mt_outcome as O
import mt_paths as MP

SPEC = MP.REPO / 'docs/superpowers/specs/2026-11-02-mmart-test-design.md'
ENCS = MP.ENCODERS
ASP = MP.ASPECTS
WIN, LEVEL, BELOW, NOT_WIN = O.WIN, O.LEVEL, O.BELOW, O.NOT_WIN
LO = {WIN: 0.01, LEVEL: -0.002, BELOW: -0.05, NOT_WIN: -0.05}     # lo for an outcome; r = 0.01 where used
R = 0.01


def cell(sc, enc, aspect, task, outcome=WIN, r='auto', n_draws=None, data='mmart_test'):
    if r == 'auto':
        r = R if sc == 'SC4' else None
    lo = LO[outcome]
    return dict(sc=sc, encoder=enc, aspect=aspect, task=task, data=data, candidate='c', comparator='k',
                candidate_value=0.5, comparator_value=0.4, d=lo + 0.01, lo=lo, hi=lo + 0.02, r=r, r_source='t',
                n_draws=MP.N_DRAWS[task] if n_draws is None else n_draws, seeds=[0, 1, 2, 3, 4], n_rows=16384)


def enc_cells(enc, over=None):
    """The 15 cells of an encoder, all WIN; `over` {(sc, aspect, task): outcome} overrides."""
    over = over or {}
    return [cell(sc, enc, a, t, over.get((sc, a, t), WIN)) for (sc, _, a, t, _) in O.required_keys(enc)]


def all_cells(over=None):
    over = over or {}
    return sum((enc_cells(e, over.get(e)) for e in ENCS), [])


def ok_status():
    return dict(
        gates=dict(i_nearcopy='ok', iii='ok', vi='ok',
                   i_default={e: 'ok' for e in ENCS}, ii={e: 'ok' for e in ENCS}, iv={e: 'ok' for e in ENCS},
                   i_teacher={a: 'ok' for a in ASP}, v={a: 'ok' for a in ASP}),
        stage1=dict(ranges={e: 'ok' for e in ENCS}, picks={e: 'ok' for e in ENCS}, teacher_pick='ok'))


def st(**fail):
    """ok_status with failures: st(i_nearcopy=True), st(ii='siglip2'), st(v='style'), st(picks='qwen')."""
    s = ok_status()
    for k, v in fail.items():
        if k in ('i_nearcopy', 'iii', 'vi'):
            s['gates'][k] = 'failed'
        elif k == 'teacher_pick':
            s['stage1']['teacher_pick'] = 'failed'
        elif k in ('ranges', 'picks'):
            s['stage1'][k][v] = 'failed'
        else:
            s['gates'][k][v] = 'failed'
    return s


def run(cells=None, status=None, n=16384, smoke=False):
    return O.decide(all_cells() if cells is None else cells, status or ok_status(), n, smoke=smoke)


def rows(res, **kw):
    return [r for r in res['cells'] if all(r[k] == v for k, v in kw.items())]


def statuses(res, **kw):
    return {r['status'] for r in rows(res, **kw)}


# ------------------------------------------------------------------------------------------ cell outcome
def mk(lo, r):
    return dict(lo=lo, r=r)


def test_cell_outcome_boundaries():
    assert O.cell_outcome(mk(1e-12, 0.01)) == WIN
    assert O.cell_outcome(mk(0.0, 0.01)) == LEVEL            # lo = 0 is not WIN (strict) and 0 > -r
    assert O.cell_outcome(mk(0.0, None)) == NOT_WIN
    assert O.cell_outcome(mk(-0.01, 0.01)) == BELOW          # lo = -r is not LEVEL (strict)
    assert O.cell_outcome(mk(-0.0099999, 0.01)) == LEVEL
    assert O.cell_outcome(mk(-1.0, None)) == NOT_WIN


def test_unrounded_unit_decides():
    r = 0.0046213
    assert O.cell_outcome(mk(-0.0046, r)) == LEVEL           # BELOW if r were rounded to 0.0046
    assert O.cell_outcome(mk(-0.0046214, r)) == BELOW


def test_rows_carry_outcomes_sc1_to_3_win_or_not_win_only():
    over = {'qwen': {('SC1', 'emotion', 'clu'): NOT_WIN, ('SC2', 'style', 'ret'): BELOW,
                     ('SC3', 'genre', 'clu'): LEVEL,
                     ('SC4', 'emotion', 'clu'): LEVEL, ('SC4', 'style', 'ret'): BELOW}}
    res = run(all_cells(over))
    for r in rows(res, encoder='qwen'):
        if r['sc'] != 'SC4':
            assert r['outcome'] in (WIN, NOT_WIN)
    assert rows(res, encoder='qwen', sc='SC1', aspect='emotion')[0]['outcome'] == NOT_WIN
    assert rows(res, encoder='qwen', sc='SC2', aspect='style')[0]['outcome'] == NOT_WIN
    assert rows(res, encoder='qwen', sc='SC3', aspect='genre')[0]['outcome'] == NOT_WIN
    assert rows(res, encoder='qwen', sc='SC4', aspect='emotion', task='clu')[0]['outcome'] == LEVEL
    assert rows(res, encoder='qwen', sc='SC4', aspect='style', task='ret')[0]['outcome'] == BELOW


def test_sixty_rows_when_all_present():
    res = run()
    assert len(res['cells']) == 60 and all(r['status'] == 'decided' for r in res['cells'])


# ------------------------------------------------------------------------------------------ verdicts
def verdict_of(over, enc='siglip2'):
    return run(all_cells({enc: over}))['verdicts'][enc]


def test_nine_wins_holds():
    v = run()['verdicts']
    assert all(v[e]['verdict'] == O.HOLDS and v[e]['go'] == 'GO' for e in ENCS)


@pytest.mark.parametrize('bad', [['genre'], ['emotion', 'style'], list(ASP)])
@pytest.mark.parametrize('worse', [LEVEL, BELOW, NOT_WIN])
def test_sc3_misses_with_six_wins_is_qualified(bad, worse):
    v = verdict_of({('SC3', a, 'clu'): worse for a in bad})
    assert v['verdict'] == O.QUALIFIED and v['go'] == 'GO for the qualified claim'
    assert v['sc3_win_aspects'] == [a for a in ASP if a not in bad]


@pytest.mark.parametrize('sc,task', [('SC1', 'clu'), ('SC2', 'ret')])
@pytest.mark.parametrize('aspect', ASP)
@pytest.mark.parametrize('sc3_all_win', [True, False])
def test_any_sc1_sc2_miss_fails(sc, task, aspect, sc3_all_win):
    over = {(sc, aspect, task): NOT_WIN}
    if not sc3_all_win:
        over[('SC3', 'genre', 'clu')] = NOT_WIN
    v = verdict_of(over)
    assert v['verdict'] == O.FAILS and v['go'] == 'NO-GO' and v['sentence'] is None


def test_verdicts_are_per_encoder():
    res = run(all_cells({'siglip1': {('SC3', 'style', 'clu'): NOT_WIN}, 'clip_l14': {('SC2', 'genre', 'ret'): NOT_WIN}}))
    got = {e: res['verdicts'][e]['verdict'] for e in ENCS}
    assert got == {'qwen': O.HOLDS, 'siglip2': O.HOLDS, 'siglip1': O.QUALIFIED, 'clip_l14': O.FAILS}


def test_sc4_does_not_enter_the_verdict():
    over = {('SC4', a, t): BELOW for a in ASP for t in ('clu', 'ret')}
    assert verdict_of(over)['verdict'] == O.HOLDS


# ------------------------------------------------------------------------------------------ SC4 claims
BETTER, LEVEL_OR = 'better than', 'level with or better than'


def claim(enc, aspect, task, outcome):
    res = run(all_cells({enc: {('SC4', aspect, task): outcome}}))
    (c,) = [c for c in res['sc4'][enc] if c['aspect'] == aspect and c['task'] == task]
    return c


@pytest.mark.parametrize('enc', ENCS)
@pytest.mark.parametrize('aspect', ASP)
@pytest.mark.parametrize('task', ['clu', 'ret'])
def test_sc4_claims_d7_and_non_d7(enc, aspect, task):
    d7 = (aspect, task) in MP.D7[enc]
    assert claim(enc, aspect, task, WIN)['claim'] == (BETTER if d7 else LEVEL_OR)
    assert claim(enc, aspect, task, LEVEL)['claim'] == LEVEL_OR
    assert claim(enc, aspect, task, BELOW)['claim'] is None
    assert claim(enc, aspect, task, WIN)['d7'] is d7


def test_sc4_d7_differs_by_encoder():
    assert claim('qwen', 'style', 'ret', WIN)['claim'] == LEVEL_OR          # not a D7 cell for Qwen
    assert claim('siglip2', 'style', 'ret', WIN)['claim'] == BETTER         # a D7 cell for the others
    assert claim('qwen', 'genre', 'clu', WIN)['claim'] == LEVEL_OR
    assert claim('qwen', 'genre', 'ret', WIN)['claim'] == BETTER


def test_sc4_below_keeps_its_row():
    res = run(all_cells({'qwen': {('SC4', 'emotion', 'clu'): BELOW}}))
    assert len(res['sc4']['qwen']) == 6 and len(rows(res, encoder='qwen', sc='SC4')) == 6
    c = [c for c in res['sc4']['qwen'] if (c['aspect'], c['task']) == ('emotion', 'clu')][0]
    assert c['claim'] is None and c['sentence'] is None and c['outcome'] == BELOW and c['status'] == 'decided'


def test_sc4_sentence_text():
    c = claim('siglip2', 'style', 'ret', WIN)
    assert c['sentence'] == 'c* on SigLIP 2 is better than the Qwen image instructed for style in retrieval on MMArt.'
    c = claim('qwen', 'genre', 'clu', LEVEL)
    assert c['sentence'] == ('c* on Qwen3-VL-Embedding-2B is level with or better than the Qwen image instructed '
                             'for genre in clustering on MMArt.')


# ------------------------------------------------------------------------------------------ D17 rows
def sc4_claims_stand(res, enc, **kw):
    return all(c['claim'] == BETTER or c['claim'] == LEVEL_OR for c in res['sc4'][enc]
               if all(c[k] == v for k, v in kw.items()))


def inconclusive_encoders(res):
    return {e for e in ENCS if res['verdicts'][e]['verdict'] == O.INCONCLUSIVE}


@pytest.mark.parametrize('fail', [dict(i_nearcopy=True), dict(iii=True), dict(i_default='qwen')])
def test_d17_every_encoder_gates(fail):
    res = run(status=st(**fail))
    assert inconclusive_encoders(res) == set(ENCS)
    assert statuses(res) == {'inconclusive'}
    for e in ENCS:
        assert res['verdicts'][e]['sentence'] is None and res['verdicts'][e]['reason']
        assert all(c['claim'] == 'inconclusive' and c['sentence'] is None for c in res['sc4'][e])


@pytest.mark.parametrize('fail', [dict(i_default='siglip2'), dict(ii='siglip2')])
def test_d17_encoder_gates_void_that_encoders_verdict_and_every_cell(fail):
    res = run(status=st(**fail))
    assert inconclusive_encoders(res) == {'siglip2'}
    assert statuses(res, encoder='siglip2') == {'inconclusive'}
    assert all(c['claim'] == 'inconclusive' for c in res['sc4']['siglip2'])
    assert statuses(res, encoder='qwen') == statuses(res, encoder='siglip1') == {'decided'}
    assert res['verdicts']['qwen']['verdict'] == O.HOLDS


@pytest.mark.parametrize('fail', [dict(ranges='siglip2'), dict(picks='siglip2')])
def test_d17_stage1_failure_voids_verdict_sc2_sc4_only_sc1_sc3_keep_outcomes(fail):
    # D17: a failure voids only the cells that use the failed part (ranges or picks: SC2 and SC4, not SC1 or SC3)
    res = run(status=st(**fail))
    assert inconclusive_encoders(res) == {'siglip2'}
    for r in rows(res, encoder='siglip2'):
        assert (r['status'] == 'inconclusive') == (r['sc'] in ('SC2', 'SC4')), r
    assert all(c['claim'] == 'inconclusive' for c in res['sc4']['siglip2'])
    assert statuses(res, encoder='qwen') == statuses(res, encoder='siglip1') == {'decided'}
    assert res['verdicts']['qwen']['verdict'] == O.HOLDS
    assert res['verdicts']['siglip2']['reason']


def test_stage1_failure_excuses_missing_sc2_sc4_but_not_sc1_sc3():
    s = st(ranges='siglip2')
    gone = drop(all_cells(), lambda c: c['encoder'] == 'siglip2' and c['sc'] in ('SC2', 'SC4'))
    assert len(run(gone, s)['cells']) == 60 - 9
    for sc in ('SC1', 'SC3'):
        gone = drop(all_cells(), lambda c: c['encoder'] == 'siglip2' and c['sc'] == sc and c['aspect'] == 'genre')
        with pytest.raises(AssertionError, match='missing'):
            run(gone, s)


@pytest.mark.parametrize('gate', ['i_teacher', 'v'])
@pytest.mark.parametrize('aspect', ASP)
def test_d17_teacher_gates_void_that_aspects_sc4_cells_only(gate, aspect):
    res = run(status=st(**{gate: aspect}))
    assert all(res['verdicts'][e]['verdict'] == O.HOLDS for e in ENCS)
    for e in ENCS:
        for r in rows(res, encoder=e):
            assert (r['status'] == 'inconclusive') == (r['sc'] == 'SC4' and r['aspect'] == aspect), r
        for c in res['sc4'][e]:
            assert (c['claim'] == 'inconclusive') == (c['aspect'] == aspect)
            assert (c['status'] == 'inconclusive') == (c['aspect'] == aspect)


def test_d17_iv_voids_verdict_and_sc3_only_sc4_claims_stand():
    res = run(status=st(iv='clip_l14'))
    assert inconclusive_encoders(res) == {'clip_l14'}
    for r in rows(res, encoder='clip_l14'):
        assert (r['status'] == 'inconclusive') == (r['sc'] == 'SC3'), r
    assert all(c['claim'] in (BETTER, LEVEL_OR) for c in res['sc4']['clip_l14'])
    assert statuses(res, encoder='qwen') == {'decided'}


def test_d17_vi_voids_sc2_and_sc4_map_cells_all_encoders_and_every_verdict():
    res = run(status=st(vi=True))
    assert inconclusive_encoders(res) == set(ENCS)
    for r in res['cells']:
        assert (r['status'] == 'inconclusive') == (r['sc'] == 'SC2' or (r['sc'] == 'SC4' and r['task'] == 'ret')), r
    for e in ENCS:
        for c in res['sc4'][e]:
            assert (c['claim'] == 'inconclusive') == (c['task'] == 'ret')
            if c['task'] == 'clu':
                assert c['claim'] in (BETTER, LEVEL_OR)


def test_d17_teacher_pick_voids_sc4_map_cells_verdicts_stand():
    res = run(status=st(teacher_pick=True))
    assert all(res['verdicts'][e]['verdict'] == O.HOLDS for e in ENCS)
    for r in res['cells']:
        assert (r['status'] == 'inconclusive') == (r['sc'] == 'SC4' and r['task'] == 'ret'), r


def test_d17_qwen_ranges_failure_takes_the_teacher_pick_with_it():
    # the teacher pick uses Qwen's ranges (D12): it cannot stand when they are missing
    res = run(status=st(ranges='qwen'))
    assert inconclusive_encoders(res) == {'qwen'}
    for r in rows(res, sc='SC4', task='ret'):
        assert r['status'] == 'inconclusive'
    assert statuses(res, encoder='siglip2', sc='SC4', task='clu') == {'decided'}
    assert res['verdicts']['siglip2']['verdict'] == O.HOLDS


def test_inconclusive_is_never_qualified_even_with_sc3_failing_or_absent():
    cells = [c for c in all_cells({'siglip2': {('SC3', 'genre', 'clu'): NOT_WIN}})]
    res = run(cells, st(iv='siglip2'))
    assert res['verdicts']['siglip2']['verdict'] == O.INCONCLUSIVE and res['verdicts']['siglip2']['sentence'] is None
    cells = [c for c in all_cells() if not (c['encoder'] == 'siglip2' and c['sc'] == 'SC3')]
    res = run(cells, st(iv='siglip2'))
    assert res['verdicts']['siglip2']['verdict'] == O.INCONCLUSIVE
    assert len(res['cells']) == 57


def test_inconclusive_notes_whether_sc1_and_sc2_won():
    res = run(all_cells({'siglip2': {('SC1', 'style', 'clu'): NOT_WIN}}), st(iv='siglip2'))
    assert res['verdicts']['siglip2']['sc1_sc2_all_win'] is False
    res = run(all_cells(), st(iv='siglip2'))
    assert res['verdicts']['siglip2']['sc1_sc2_all_win'] is True


def test_sc1_sc2_note_counts_only_cells_the_same_failure_did_not_void():
    # gate vi voids SC2: a not-WIN SC2 cell is no evidence, so the note is None (not False)
    res = run(all_cells({'siglip2': {('SC2', 'style', 'ret'): NOT_WIN}}), st(vi=True))
    assert res['verdicts']['siglip2']['sc1_sc2_all_win'] is None
    # gate ii voids every cell of the encoder
    res = run(all_cells({'siglip2': {('SC1', 'style', 'clu'): NOT_WIN}}), st(ii='siglip2'))
    assert res['verdicts']['siglip2']['sc1_sc2_all_win'] is None
    # stage 1 ranges voids SC2 only: SC2 voided gives None; with an SC1 miss it is still None
    res = run(all_cells({'siglip2': {('SC1', 'style', 'clu'): NOT_WIN}}), st(ranges='siglip2'))
    assert res['verdicts']['siglip2']['sc1_sc2_all_win'] is None
    # gate iv voids neither SC1 nor SC2: the note stands
    res = run(all_cells({'siglip2': {('SC2', 'style', 'ret'): NOT_WIN}}), st(iv='siglip2'))
    assert res['verdicts']['siglip2']['sc1_sc2_all_win'] is False


def test_two_voids_give_both_reasons():
    res = run(status=st(ii='qwen', iv='qwen'))
    assert len(res['verdicts']['qwen']['reason']) == 2


# ------------------------------------------------------------------------------------------ missing cells
def drop(cells, pred):
    return [c for c in cells if not pred(c)]


def test_missing_cell_raises_unless_status_excuses_it():
    cells = drop(all_cells(), lambda c: (c['encoder'], c['sc'], c['aspect']) == ('qwen', 'SC1', 'genre'))
    with pytest.raises(AssertionError, match='missing'):
        run(cells)
    with pytest.raises(AssertionError, match='missing'):
        run(cells, st(iv='qwen'))                  # iv excuses SC3 only
    with pytest.raises(AssertionError, match='missing'):
        run(cells, st(vi=True))                    # vi excuses SC2 and SC4 map cells only
    assert run(cells, st(ii='qwen'))['verdicts']['qwen']['verdict'] == O.INCONCLUSIVE


def test_excused_missing_cells():
    sc4_style = drop(all_cells(), lambda c: c['sc'] == 'SC4' and c['aspect'] == 'style')
    with pytest.raises(AssertionError, match='missing'):
        run(sc4_style)
    assert len(run(sc4_style, st(v='style'))['cells']) == 60 - 8
    no_ret = drop(all_cells(), lambda c: c['sc'] == 'SC4' and c['task'] == 'ret')
    assert len(run(no_ret, st(teacher_pick=True))['cells']) == 60 - 12
    no_sc2 = drop(all_cells(), lambda c: c['sc'] == 'SC2')
    with pytest.raises(AssertionError, match='missing'):
        run(no_sc2)
    assert len(run(no_sc2, st(vi=True))['cells']) == 60 - 12
    assert run([], st(i_nearcopy=True))['cells'] == []


def test_a_decided_encoder_cannot_hide_a_missing_cell_behind_another_encoders_failure():
    cells = drop(all_cells(), lambda c: c['encoder'] == 'qwen' and c['sc'] == 'SC1')
    with pytest.raises(AssertionError, match='missing'):
        run(cells, st(ii='siglip2'))


# ------------------------------------------------------------------------------------------ refusals
def with_cell(**mut):
    cells = all_cells()
    target = [c for c in cells if (c['encoder'], c['sc'], c['aspect'], c['task']) == ('qwen', 'SC4', 'emotion', 'ret')][0]
    target.update(mut)
    return cells


def test_n_draws_other_than_protocol_refused_outside_smoke():
    with pytest.raises(AssertionError, match='n_draws'):
        run(with_cell(n_draws=9999))
    with pytest.raises(AssertionError, match='n_draws'):
        run(with_cell(n_draws=2000))                # clustering's draws on a retrieval cell
    cells = all_cells()
    [c for c in cells if c['sc'] == 'SC1'][0]['n_draws'] = 10000
    with pytest.raises(AssertionError, match='n_draws'):
        run(cells)
    assert run(with_cell(n_draws=100), smoke=True)['cells']          # smoke lets them differ


@pytest.mark.parametrize('data', ['mmart_far', 'mmart_unseen_artist', 'mmart_shared_held', 'artelingo_held', None])
def test_data_other_than_mmart_test_refused(data):
    with pytest.raises(AssertionError, match='data'):
        run(with_cell(data=data))
    with pytest.raises(AssertionError, match='data'):
        run(with_cell(data=data), smoke=True)


def test_duplicate_cells_refused():
    cells = all_cells()
    with pytest.raises(AssertionError, match='duplicate'):
        run(cells + [copy.deepcopy(cells[3])])


def test_sc1_to_sc3_must_not_carry_r_and_sc4_must():
    cells = all_cells()
    [c for c in cells if c['sc'] == 'SC3'][0]['r'] = 0.01
    with pytest.raises(AssertionError, match=r'\br\b'):
        run(cells)
    with pytest.raises(AssertionError, match=r'\br\b'):
        run(with_cell(r=None))
    with pytest.raises(AssertionError, match=r'\br\b'):
        run(with_cell(r=0.0))
    with pytest.raises(AssertionError, match=r'\br\b'):
        run(with_cell(r=float('nan')))


@pytest.mark.parametrize('mut', [dict(sc='SC5'), dict(encoder='vit'), dict(aspect='mood'), dict(task='cluster'),
                                 dict(lo=float('nan')), dict(lo=1.0, hi=0.0), dict(d=None)])
def test_malformed_cells_refused(mut):
    with pytest.raises(AssertionError):
        run(with_cell(**mut))


@pytest.mark.parametrize('sc,task', [('SC1', 'ret'), ('SC2', 'clu'), ('SC3', 'ret')])
def test_sc_task_pairing_refused(sc, task):
    cells = all_cells()
    c = [c for c in cells if c['sc'] == sc][0]
    c['task'] = task
    c['n_draws'] = MP.N_DRAWS[task]
    with pytest.raises(AssertionError):
        run(cells)


def test_cell_missing_a_field_refused():
    cells = all_cells()
    del cells[0]['n_rows']
    with pytest.raises(AssertionError, match='lacks'):
        run(cells)


# ------------------------------------------------------------------------------------------ status
def test_status_absent_keys_refused():
    for path in (('gates', 'iii'), ('gates', 'vi'), ('gates', 'i_nearcopy'), ('gates', 'ii'), ('gates', 'iv'),
                 ('gates', 'i_default'), ('gates', 'i_teacher'), ('gates', 'v'), ('stage1', 'ranges'),
                 ('stage1', 'picks'), ('stage1', 'teacher_pick'), ('gates',), ('stage1',)):
        s = ok_status()
        d = s
        for k in path[:-1]:
            d = d[k]
        del d[path[-1]]
        with pytest.raises(AssertionError, match='status'):
            run(status=s)


def test_status_inner_keys_and_values_refused():
    s = ok_status()
    del s['gates']['ii']['qwen']
    with pytest.raises(AssertionError, match='status'):
        run(status=s)
    s = ok_status()
    del s['gates']['v']['genre']
    with pytest.raises(AssertionError, match='status'):
        run(status=s)
    s = ok_status()
    s['stage1']['picks']['siglip1'] = 'maybe'
    with pytest.raises(AssertionError, match='status'):
        run(status=s)
    s = ok_status()
    s['gates']['iii'] = True
    with pytest.raises(AssertionError, match='status'):
        run(status=s)
    s = ok_status()
    s['gates']['extra'] = 'ok'
    with pytest.raises(AssertionError, match='status'):
        run(status=s)


def test_ranges_as_one_string_applies_to_every_encoder():
    s = ok_status()
    s['stage1']['ranges'] = 'failed'
    res = run(status=s)
    assert inconclusive_encoders(res) == set(ENCS)


# ------------------------------------------------------------------------------------------ sentences
def squash(text):
    return re.sub(r'\s+', ' ', text.replace('\n> ', ' '))


SPEC_TEXT = squash(SPEC.read_text())


@pytest.mark.parametrize('frag', [O.HEAD, O.HOLDS_BODY, O.QUAL_BODY, O.TAIL, O.SC4_SENTENCE])
def test_sentence_fragments_are_in_the_spec(frag):
    assert frag.replace('<n>', '<n>') in SPEC_TEXT, frag


def test_holds_sentence():
    s = run()['verdicts']['siglip2']['sentence']
    assert s == ("On 16,384 WikiArt paintings outside ArtELingo (MMArt; exact and near copies of ArtELingo train and "
                 "validation paintings removed), c*'s recipe with the SigLIP 2 image as its only input at test "
                 "(frozen; never trained or picked on these paintings) clusters and retrieves paintings by emotion "
                 "(MMArt's labels), style and genre better than the plain SigLIP 2 image, and clusters them better "
                 "than PercepT on the SigLIP 2 image (AMI with 20 clusters; mAP with read-outs picked on validation "
                 "paintings with MMArt labels; 5 seeds). SigLIP 2 is one of four encoders; the three besides Qwen "
                 "passed an earlier held emotion test on ArtELingo.")


def test_n_comes_from_the_argument_with_thousands_separator():
    assert 'On 12,345 WikiArt' in run(n=12345, smoke=True)['verdicts']['qwen']['sentence']
    assert run(n=16384)['n'] == 16384


def test_n_must_be_the_protocol_test_count_outside_smoke_when_a_sentence_is_licensed():
    with pytest.raises(AssertionError, match='N_TEST|16384|test.n'):
        run(n=12345)
    run(n=12345, smoke=True)
    # no sentence licensed: n is not checked
    run(status=st(i_nearcopy=True), cells=[], n=12345)


def test_qualified_sentence_names_the_sc3_aspects_that_win():
    s = verdict_of({('SC3', 'genre', 'clu'): LEVEL})['sentence']
    assert s.endswith("passed an earlier held emotion test on ArtELingo.")
    assert "better than the plain SigLIP 2 image; against PercepT on the SigLIP 2 image, better in emotion and style (" in s
    assert 'clusters them better' not in s
    s = verdict_of({('SC3', 'emotion', 'clu'): BELOW, ('SC3', 'genre', 'clu'): BELOW})['sentence']
    assert 'better in style (' in s
    s = verdict_of({('SC3', 'style', 'clu'): BELOW})['sentence']
    assert 'better in emotion and genre (' in s
    s = verdict_of({('SC3', a, 'clu'): NOT_WIN for a in ASP})['sentence']
    assert 'it is better in no aspect' in s


def test_qwen_name_in_sentence():
    assert 'the Qwen3-VL-Embedding-2B image as its only input' in run()['verdicts']['qwen']['sentence']


def test_no_sentence_unless_holds_or_qualified():
    res = run(all_cells({'qwen': {('SC1', 'style', 'clu'): NOT_WIN}}), st(ii='siglip1'))
    assert res['verdicts']['qwen']['sentence'] is None
    assert res['verdicts']['siglip1']['sentence'] is None
    assert res['verdicts']['clip_l14']['sentence']


def test_n_needed_only_when_a_sentence_is_licensed():
    with pytest.raises(AssertionError, match=r'<n>|cutsets'):
        run(n=None)
    res = run(status=st(i_nearcopy=True), cells=[], n=None)
    assert inconclusive_encoders(res) == set(ENCS)
    for bad in (0, -5, 12.5, '16384', True):
        with pytest.raises(AssertionError):
            run(n=bad)


# ------------------------------------------------------------------------------------------ CLI and files
@pytest.fixture
def tree(tmp_path, monkeypatch):
    root = tmp_path / 'out'
    (root / 'stage2').mkdir(parents=True)
    rec = tmp_path / 'records'
    rec.mkdir()
    monkeypatch.setattr(MP, 'record_path', lambda name: rec / name)
    reads = []

    def committed(name):
        reads.append(name)
        return json.loads((rec / name).read_text())
    monkeypatch.setattr(MP, 'read_committed_record', committed)

    def put(cells=None, status=None, n=16384, split=True):
        cells = all_cells() if cells is None else cells
        if split:
            (root / 'stage2' / 'cells_clu.json').write_text(json.dumps([c for c in cells if c['task'] == 'clu']))
            (root / 'stage2' / 'cells_ret.json').write_text(json.dumps([c for c in cells if c['task'] == 'ret']))
        if status is not None:
            (rec / 'status.json').write_text(json.dumps(status))
        if n is not None:
            (rec / 'cutsets.json').write_text(json.dumps({'test': {'n': n, 'sha256': 'x'}, 'far': {'n': 1, 'sha256': 'y'}}))
        return root
    put.root, put.rec, put.reads = root, rec, reads
    return put


def test_cli_writes_outcomes_json(tree, capsys):
    root = tree(status=ok_status())
    O.main(['decide', '--root', str(root)])
    res = json.loads((root / 'stage2' / 'outcomes.json').read_text())
    assert len(res['cells']) == 60 and res['n'] == 16384
    assert {e: res['verdicts'][e]['verdict'] for e in ENCS} == {e: 'holds' for e in ENCS}
    assert 'holds' in capsys.readouterr().out


def test_cli_reads_status_and_cutsets_through_the_committed_record_outside_smoke(tree, monkeypatch):
    root = tree(status=ok_status())
    O.main(['decide', '--root', str(root)])
    assert sorted(tree.reads) == ['cutsets.json', 'status.json']
    tree.reads.clear()
    monkeypatch.setattr(MP, 'SMOKE', 1)                    # smoke reads the plain files
    O.main(['decide', '--root', str(root)])
    assert tree.reads == []


def test_cli_refuses_a_cutsets_n_other_than_n_test_outside_smoke(tree, monkeypatch):
    root = tree(status=ok_status(), n=12345)
    with pytest.raises(AssertionError):
        O.main(['decide', '--root', str(root)])
    monkeypatch.setattr(MP, 'SMOKE', 1)
    O.main(['decide', '--root', str(root)])


def test_cli_out_option(tree, tmp_path):
    root = tree(status=ok_status())
    out = tmp_path / 'x.json'
    O.main(['decide', '--root', str(root), '--out', str(out)])
    assert out.exists() and not (root / 'stage2' / 'outcomes.json').exists()


def test_cli_status_absent_is_an_error(tree):
    root = tree(status=None)
    with pytest.raises((AssertionError, FileNotFoundError)):
        O.main(['decide', '--root', str(root)])


def test_cli_cutsets_absent_is_an_error_when_a_sentence_is_needed(tree):
    root = tree(status=ok_status(), n=None)
    with pytest.raises(AssertionError):
        O.main(['decide', '--root', str(root)])


def test_cli_cutsets_without_test_n_refused(tree):
    root = tree(status=ok_status())
    (tree.rec / 'cutsets.json').write_text(json.dumps({'far': {'n': 3}}))
    with pytest.raises(AssertionError, match='cutsets'):
        O.main(['decide', '--root', str(root)])


def test_cli_cells_in_the_wrong_file_refused(tree):
    cells = all_cells()
    root = tree(cells=cells, status=ok_status())
    (root / 'stage2' / 'cells_clu.json').write_text(json.dumps(cells))     # retrieval cells in the clustering file
    with pytest.raises(AssertionError):
        O.main(['decide', '--root', str(root)])


def test_cli_ignores_pending_files_and_diag_folder(tree):
    root = tree(status=ok_status())
    (root / 'stage2' / 'cells_clu.pending.json').write_text('not json')
    (root / 'diag').mkdir()
    (root / 'diag' / 'cells_clu.json').write_text(json.dumps([cell('SC1', 'qwen', 'emotion', 'clu', data='mmart_far')]))
    O.main(['decide', '--root', str(root)])
    assert len(json.loads((root / 'stage2' / 'outcomes.json').read_text())['cells']) == 60


def test_cli_smoke_allows_other_draws(tree, monkeypatch):
    cells = all_cells()
    for c in cells:
        c['n_draws'] = 40
    root = tree(cells=cells, status=ok_status())
    with pytest.raises(AssertionError, match='n_draws'):
        O.main(['decide', '--root', str(root)])
    monkeypatch.setattr(MP, 'SMOKE', 1)
    O.main(['decide', '--root', str(root)])
    assert json.loads((root / 'stage2' / 'outcomes.json').read_text())['smoke'] is True


def test_cli_no_cells_at_all_with_everything_voided(tree):
    root = tree(cells=[], status=st(i_nearcopy=True), n=None, split=False)
    O.main(['decide', '--root', str(root)])
    res = json.loads((root / 'stage2' / 'outcomes.json').read_text())
    assert res['cells'] == [] and res['n'] is None
