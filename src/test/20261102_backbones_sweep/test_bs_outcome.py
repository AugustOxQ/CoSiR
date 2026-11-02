"""Tests of bs_outcome (r14 ticket 09): outcomes at the boundaries, the verdict patterns of rule review Q4, the
inconclusive paths (D18), missing cells, the n_draws refusal, the licensed sentences against the spec text."""
import copy
import json
import re

import pytest

import bs_outcome as O
import bs_paths as BP

SPEC = BP.REPO / 'docs/superpowers/specs/2026-11-02-backbones-sweep-design.md'
ENCS = ('siglip2', 'siglip1', 'clip_l14')
LO = {O.WIN: 0.01, O.LEVEL: -0.002, O.BELOW: -0.05, O.NOT_WIN: -0.05}     # lo for an outcome; r = 0.01 where used
R = 0.01


def cell(sc, enc, aspect, task, data, outcome=O.WIN, r='auto', n_draws=None):
    if r == 'auto':
        r = R if sc in O.NEEDS_R else None
    lo = LO[outcome]
    return dict(sc=sc, encoder=enc, aspect=aspect, task=task, data=data, candidate='c', comparator='k',
                candidate_value=0.5, comparator_value=0.4, d=lo + 0.01, lo=lo, hi=lo + 0.02, r=r, r_source='t',
                n_draws=O.N_DRAWS[task] if n_draws is None else n_draws, seeds=[0, 1, 2, 3, 4])


def enc_cells(enc, over=None):
    """All 15 painting + 4 photo cells of an encoder, all WIN; `over` {(sc, aspect, task): outcome} overrides."""
    over = over or {}
    cs = []
    for k in O.required_keys(enc, 'paintings') + O.required_keys(enc, 'photos'):
        sc, _, a, t, d = k
        cs.append(cell(sc, enc, a, t, d, over.get((sc, a, t), O.WIN)))
    return cs


def partc_cells(o7=O.WIN, o8=O.WIN, o9=O.WIN, o10=O.WIN, enc='qwen'):
    return [cell(sc, enc, 'emotion', t, 'emoset_test', o) for sc, t, o in
            (('SC7', 'clu', o7), ('SC8', 'ret', o8), ('SC9', 'clu', o9), ('SC10', 'ret', o10))]


def ok_status():
    return dict(encoders={e: dict(cstar_runs='ok', photo_runs='ok', stage0=dict(paintings='ok', photos='ok'))
                          for e in ENCS}, part_c=dict(stage0='ok', stage2='ok'))


def all_cells(over=None, **pc):
    over = over or {}
    return sum((enc_cells(e, over.get(e)) for e in ENCS), []) + partc_cells(**pc)


def run(cells=None, status=None, n=17000):
    return O.decide(cells if cells is not None else all_cells(), status or ok_status(), n)


# ------------------------------------------------------------------------------------------ cell outcome
def mk(lo, r):
    return dict(lo=lo, r=r)


def test_boundary_lo_zero_is_not_a_win():
    assert O.cell_outcome(mk(0.0, 0.01)) == O.LEVEL      # lo = 0 is not WIN (strict), and 0 > -r
    assert O.cell_outcome(mk(1e-12, 0.01)) == O.WIN
    assert O.cell_outcome(mk(0.0, None)) == O.NOT_WIN


def test_boundary_lo_minus_r_is_below():
    assert O.cell_outcome(mk(-0.01, 0.01)) == O.BELOW    # lo = -r is not LEVEL (strict)
    assert O.cell_outcome(mk(-0.0099999, 0.01)) == O.LEVEL
    assert O.cell_outcome(mk(-0.0128, 0.0128)) == O.BELOW


def test_unrounded_unit_decides():
    r = 0.0046213
    assert O.cell_outcome(mk(-0.0046, r)) == O.LEVEL     # would be BELOW with r rounded to 0.0046
    assert O.cell_outcome(mk(-0.0046214, r)) == O.BELOW


def test_win_only_cell_reports_win_or_not_win():
    assert O.cell_outcome(mk(-1.0, None)) == O.NOT_WIN and O.cell_outcome(mk(0.3, None)) == O.WIN


# ------------------------------------------------------------------------------------------ verdict patterns (Q4)
def pv(over):
    o = {(sc, a): O.WIN for sc in ('SC1', 'SC2', 'SC3') for a in O.ASPECTS}
    o.update(over)
    return O.paintings_verdict(o)


def test_paintings_nine_wins_holds():
    assert pv({}) == O.HOLDS


@pytest.mark.parametrize('bad', [[('SC3', 'genre')], [('SC3', 'emotion'), ('SC3', 'style')],
                                 [('SC3', a) for a in O.ASPECTS]])
@pytest.mark.parametrize('worse', [O.LEVEL, O.BELOW, O.NOT_WIN])
def test_paintings_sc3_misses_with_six_wins_is_qualified(bad, worse):
    assert pv({k: worse for k in bad}) == O.QUALIFIED


@pytest.mark.parametrize('sc', ['SC1', 'SC2'])
@pytest.mark.parametrize('aspect', O.ASPECTS)
@pytest.mark.parametrize('sc3_all_win', [True, False])
def test_paintings_any_sc1_sc2_miss_fails(sc, aspect, sc3_all_win):
    over = {(sc, aspect): O.LEVEL}
    if not sc3_all_win:
        over[('SC3', 'genre')] = O.BELOW
    assert pv(over) == O.FAILS


def test_photos_pattern():
    assert O.photos_verdict(O.WIN, O.WIN) == O.HOLDS
    assert O.photos_verdict(O.WIN, O.LEVEL) == O.FAILS
    assert O.photos_verdict(O.NOT_WIN, O.WIN) == O.FAILS
    assert O.photos_verdict(O.BELOW, O.BELOW) == O.FAILS


@pytest.mark.parametrize('o,v', [
    ((O.WIN, O.WIN, O.WIN, O.WIN), O.HOLDS),
    ((O.WIN, O.WIN, O.WIN, O.LEVEL), O.QUALIFIED),
    ((O.WIN, O.WIN, O.LEVEL, O.WIN), O.QUALIFIED),
    ((O.WIN, O.WIN, O.LEVEL, O.LEVEL), O.FAILS),
    ((O.WIN, O.WIN, O.WIN, O.BELOW), O.FAILS),
    ((O.WIN, O.WIN, O.BELOW, O.WIN), O.FAILS),
    ((O.WIN, O.WIN, O.BELOW, O.LEVEL), O.FAILS),
    ((O.LEVEL, O.WIN, O.WIN, O.LEVEL), O.FAILS),     # a plain-image miss is not rescued by the rest
    ((O.WIN, O.LEVEL, O.LEVEL, O.WIN), O.FAILS),
    ((O.BELOW, O.WIN, O.WIN, O.WIN), O.FAILS),
    ((O.NOT_WIN, O.WIN, O.WIN, O.WIN), O.FAILS),
])
def test_part_c_patterns(o, v):
    assert O.part_c_verdict(*o) == v


def test_go_mapping():
    assert O.GO_OF[O.HOLDS] == 'GO' and O.GO_OF[O.QUALIFIED] == 'GO for the qualified claim'
    assert O.GO_OF[O.FAILS] == 'NO-GO'


def test_sc6_claim_level_or_win_below_none():
    assert O.sc6_claim(O.WIN) and O.sc6_claim(O.LEVEL) and not O.sc6_claim(O.BELOW)


# ------------------------------------------------------------------------------------------ decide, end to end
def test_all_win_everything_holds_with_sentences():
    res = run()
    for e in ENCS:
        assert res['paintings'][e]['verdict'] == O.HOLDS and res['paintings'][e]['go'] == 'GO'
        assert res['photos'][e]['verdict'] == O.HOLDS and res['photos'][e]['sentence']
        assert len(res['sc6'][e]) == 8 and all(r['claim'] and r['sentence'] for r in res['sc6'][e])
    assert res['part_c']['verdict'] == O.HOLDS and '17,000' in res['part_c']['sentence']
    assert len(res['cells']) == 3 * 19 + 4 and all(c['outcome'] == O.WIN for c in res['cells'])


def test_decide_keeps_every_cell_with_outcome():
    res = run(all_cells({'siglip1': {('SC3', 'genre', 'clu'): O.BELOW}}))
    c = [c for c in res['cells'] if (c['sc'], c['encoder'], c['aspect']) == ('SC3', 'siglip1', 'genre')][0]
    assert c['outcome'] == O.NOT_WIN and c['status'] == 'decided'      # SC3 is a WIN-only bar (r null)
    assert res['paintings']['siglip1']['verdict'] == O.QUALIFIED
    assert res['paintings']['siglip1']['sc3_win_aspects'] == ['emotion', 'style']
    assert res['paintings']['siglip2']['verdict'] == O.HOLDS


def test_sc6_below_gives_no_claim_but_keeps_row():
    res = run(all_cells({'clip_l14': {('SC6', 'genre', 'clu'): O.BELOW, ('SC6', 'emotion', 'ret'): O.LEVEL}}))
    rows = {(r['aspect'], r['task'], r['data']): r for r in res['sc6']['clip_l14']}
    g = rows[('genre', 'clu', 'artelingo_held')]
    assert g['outcome'] == O.BELOW and not g['claim'] and g['sentence'] is None
    lv = rows[('emotion', 'ret', 'artelingo_held')]
    assert lv['outcome'] == O.LEVEL and lv['claim']
    assert sum(r['claim'] for r in res['sc6']['clip_l14']) == 7
    assert res['paintings']['clip_l14']['verdict'] == O.HOLDS         # SC6 does not enter the verdict


def test_sc6_photo_cells_are_their_own_rows():
    cs = all_cells()                     # the painting and the photo SC6 cell of emotion/ret are separate rows
    for c in cs:
        if (c['sc'], c['encoder'], c['task'], c['data']) == ('SC6', 'siglip2', 'ret', 'affection_test'):
            c['lo'] = LO[O.BELOW]
    res = run(cs)
    rows = {(r['aspect'], r['task'], r['data']): r for r in res['sc6']['siglip2']}
    assert not rows[('emotion', 'ret', 'affection_test')]['claim']
    assert rows[('emotion', 'ret', 'artelingo_held')]['claim']


def test_part_c_qualified_end_to_end():
    res = run(all_cells(o9=O.LEVEL))
    pc = res['part_c']
    assert pc['verdict'] == O.QUALIFIED and pc['go'] == 'GO for the qualified claim'
    assert 'better than the instructed image in retrieval and level with it in clustering' in pc['sentence']
    res = run(all_cells(o10=O.LEVEL))
    assert 'better than the instructed image in clustering and level with it in retrieval' in res['part_c']['sentence']


def test_part_c_fail_has_no_sentence_and_needs_no_n():
    res = O.decide(all_cells(o9=O.BELOW), ok_status(), None)
    assert res['part_c']['verdict'] == O.FAILS and res['part_c']['sentence'] is None and res['part_c']['go'] == 'NO-GO'


def test_part_c_holds_without_n_raises():
    with pytest.raises(AssertionError, match='<n>'):
        O.decide(all_cells(), ok_status(), None)


def test_part_c_sc9_sc10_need_a_unit():
    cs = all_cells()
    for c in cs:
        if c['sc'] == 'SC9':
            c['r'] = None
    with pytest.raises(AssertionError, match='needs a LEVEL unit'):
        O.check_cell([c for c in cs if c['sc'] == 'SC9'][0])


# ------------------------------------------------------------------------------------------ inconclusive paths (D18)
def test_cstar_runs_failed_gives_no_paintings_verdict_and_no_sc6_claims():
    st = ok_status()
    st['encoders']['siglip1']['cstar_runs'] = 'failed'
    st['encoders']['siglip1']['photo_runs'] = 'failed'
    cs = [c for c in all_cells() if not (c['encoder'] == 'siglip1' and c['sc'] in ('SC2', 'SC3'))]   # rows that exist
    res = O.decide(cs, st, 17000)
    assert res['paintings']['siglip1']['verdict'] == O.INCONCLUSIVE and res['paintings']['siglip1']['sentence'] is None
    assert not any(r['claim'] for r in res['sc6']['siglip1']) and len(res['sc6']['siglip1']) == 8
    assert all(r['status'] == 'inconclusive' for r in res['sc6']['siglip1'])
    assert res['photos']['siglip1']['verdict'] == O.INCONCLUSIVE        # photo runs failed with it
    assert res['paintings']['siglip2']['verdict'] == O.HOLDS and res['paintings']['clip_l14']['verdict'] == O.HOLDS
    assert all(r['claim'] for r in res['sc6']['siglip2'])
    rows = [c for c in res['cells'] if c['encoder'] == 'siglip1']
    assert rows and all(c['status'] == 'inconclusive' for c in rows if c['sc'] in ('SC1', 'SC6'))   # reported, not decided


def test_paintings_stage0_gate_failed_same_as_runs_failed():
    st = ok_status()
    st['encoders']['clip_l14']['stage0']['paintings'] = 'failed'
    res = run(status=st)
    assert res['paintings']['clip_l14']['verdict'] == O.INCONCLUSIVE
    assert not any(r['claim'] for r in res['sc6']['clip_l14'])
    assert res['photos']['clip_l14']['verdict'] == O.HOLDS             # its own path did not fail


def test_photo_runs_failed_removes_only_photos_and_photo_sc6():
    st = ok_status()
    st['encoders']['siglip2']['photo_runs'] = 'failed'
    cs = [c for c in all_cells() if not (c['encoder'] == 'siglip2' and c['data'] == 'affection_test' and c['sc'] == 'SC5')]
    res = O.decide(cs, st, 17000)
    assert res['photos']['siglip2']['verdict'] == O.INCONCLUSIVE and res['photos']['siglip2']['sentence'] is None
    assert res['paintings']['siglip2']['verdict'] == O.HOLDS           # "a failed photo run leaves the paintings verdict"
    rows = res['sc6']['siglip2']
    assert sum(r['claim'] for r in rows) == 6 and len(rows) == 8
    assert all(not r['claim'] and r['status'] == 'inconclusive' for r in rows if r['data'] == 'affection_test')
    assert all(r['claim'] for r in rows if r['data'] == 'artelingo_held')


def test_photos_stage0_gate_failed():
    st = ok_status()
    st['encoders']['siglip1']['stage0']['photos'] = 'failed'
    res = run(status=st)
    assert res['photos']['siglip1']['verdict'] == O.INCONCLUSIVE
    assert res['paintings']['siglip1']['verdict'] == O.HOLDS


@pytest.mark.parametrize('key', ['stage0', 'stage2'])
def test_part_c_inconclusive_on_gate(key):
    st = ok_status()
    st['part_c'][key] = 'failed'
    cs = [c for c in all_cells() if c['sc'] not in ('SC9', 'SC10')]     # cells that exist are reported
    res = O.decide(cs, st, None)
    assert res['part_c']['verdict'] == O.INCONCLUSIVE and res['part_c']['sentence'] is None
    assert [c['status'] for c in res['cells'] if c['sc'] == 'SC7'] == ['inconclusive']
    assert res['paintings']['siglip2']['verdict'] == O.HOLDS            # part a untouched


def test_failed_part_c_gate_is_not_overridden_by_wins():
    st = ok_status()
    st['part_c']['stage2'] = 'failed'
    assert O.decide(all_cells(), st, 17000)['part_c']['verdict'] == O.INCONCLUSIVE


def test_cstar_failed_with_photo_runs_ok_is_impossible():
    st = ok_status()
    st['encoders']['siglip2']['cstar_runs'] = 'failed'
    with pytest.raises(AssertionError, match='cannot be ok'):
        O.check_status(st)


@pytest.mark.parametrize('mutate', [lambda s: s.pop('part_c'), lambda s: s['encoders'].pop('siglip1'),
                                    lambda s: s['encoders']['siglip2'].update(cstar_runs='maybe'),
                                    lambda s: s['part_c'].pop('stage2')])
def test_status_schema_asserted(mutate):
    st = ok_status()
    mutate(st)
    with pytest.raises((AssertionError, KeyError)):
        O.check_status(st)


# ------------------------------------------------------------------------------------------ missing cells, refusal
@pytest.mark.parametrize('drop', [('SC1', 'siglip2', 'genre'), ('SC2', 'clip_l14', 'style'), ('SC3', 'siglip1', 'emotion'),
                                  ('SC4', 'siglip2', 'emotion'), ('SC5', 'siglip1', 'emotion'), ('SC7', 'qwen', 'emotion'),
                                  ('SC10', 'qwen', 'emotion')])
def test_missing_cell_raises(drop):
    cs = [c for c in all_cells() if (c['sc'], c['encoder'], c['aspect']) != drop]
    with pytest.raises(AssertionError, match='missing'):
        run(cs)


def test_missing_sc6_cell_raises():
    cs = [c for c in all_cells() if not (c['sc'] == 'SC6' and c['encoder'] == 'siglip2' and c['aspect'] == 'style'
                                          and c['task'] == 'ret')]
    with pytest.raises(AssertionError, match='missing'):
        run(cs)


def test_missing_cell_excused_only_for_failed_path():
    st = ok_status()
    st['encoders']['siglip2']['photo_runs'] = 'failed'
    cs = [c for c in all_cells() if not (c['encoder'] == 'siglip2' and c['data'] == 'affection_test')]
    O.decide(cs, st, 17000)                                              # fine: photos failed
    cs2 = [c for c in cs if not (c['encoder'] == 'siglip2' and c['sc'] == 'SC1' and c['aspect'] == 'genre')]
    with pytest.raises(AssertionError, match='missing'):
        O.decide(cs2, st, 17000)                                         # a paintings cell is still needed


def test_duplicate_cell_raises():
    cs = all_cells()
    with pytest.raises(AssertionError, match='duplicate'):
        run(cs + [copy.deepcopy(cs[0])])


def test_unexpected_cell_raises():
    extra = cell('SC4', 'siglip2', 'genre', 'clu', 'affection_test')
    with pytest.raises(AssertionError, match='not part of the rule'):
        run(all_cells() + [extra])
    with pytest.raises(AssertionError, match='unknown encoders'):
        run(all_cells() + [cell('SC4', 'dino', 'emotion', 'clu', 'affection_test')])


@pytest.mark.parametrize('task,bad', [('clu', 1999), ('clu', 10000), ('ret', 2000), ('ret', 9999)])
def test_n_draws_refused(task, bad):
    sc = 'SC1' if task == 'clu' else 'SC2'
    c = cell(sc, 'siglip2', 'emotion', task, 'artelingo_held', n_draws=bad)
    with pytest.raises(AssertionError, match='n_draws'):
        O.check_cell(c)
    O.check_cell(c, smoke=True)                                          # the smoke flag lets it through
    O.check_cell(cell(sc, 'siglip2', 'emotion', task, 'artelingo_held'))


def test_n_draws_refused_when_loading(tmp_path):
    (tmp_path / 'cells_x.json').write_text(json.dumps([cell('SC1', 'siglip2', 'emotion', 'clu', 'artelingo_held', n_draws=50)]))
    with pytest.raises(AssertionError, match='n_draws'):
        O.load_cells(tmp_path)
    cells, files = O.load_cells(tmp_path, smoke=True)
    assert len(cells) == 1 and files == ['cells_x.json']


def test_check_cell_other_asserts():
    base = cell('SC1', 'siglip2', 'emotion', 'clu', 'artelingo_held')
    for k, v in (('lo', float('nan')), ('task', 'ret'), ('data', 'emoset_test'), ('sc', 'SC11')):
        c = dict(base, **{k: v})
        with pytest.raises(AssertionError):
            O.check_cell(c)
    c = dict(base)
    del c['r']
    with pytest.raises(AssertionError, match='lacks'):
        O.check_cell(c)
    with pytest.raises(AssertionError, match='needs a LEVEL unit'):
        O.check_cell(cell('SC6', 'siglip2', 'emotion', 'clu', 'artelingo_held', r=None))


def test_load_cells_reads_every_file_and_flags_duplicates(tmp_path):
    a, b = enc_cells('siglip2')[:3], enc_cells('siglip1')[:2]
    (tmp_path / 'cells_one.json').write_text(json.dumps(a))
    (tmp_path / 'cells_two.json').write_text(json.dumps(b))
    cells, files = O.load_cells(tmp_path)
    assert len(cells) == 5 and files == ['cells_one.json', 'cells_two.json']
    (tmp_path / 'cells_three.json').write_text(json.dumps(a[:1]))
    with pytest.raises(AssertionError, match='duplicate'):
        O.load_cells(tmp_path)


def test_load_cells_skips_pending_files(tmp_path):
    a = enc_cells('siglip2')[:3]
    (tmp_path / 'cells_one.json').write_text(json.dumps(a))
    (tmp_path / 'cells_one.pending.json').write_text(json.dumps(a))      # a writer's temporary file: same cells
    cells, files = O.load_cells(tmp_path)
    assert len(cells) == 3 and files == ['cells_one.json']


@pytest.mark.parametrize('field,bad', [('aspect', 'style'), ('encoder', 'siglip2')])
@pytest.mark.parametrize('sc,task', [('SC7', 'clu'), ('SC8', 'ret'), ('SC9', 'clu'), ('SC10', 'ret')])
def test_part_c_cells_need_emotion_and_qwen(sc, task, field, bad):
    good = cell(sc, 'qwen', 'emotion', task, 'emoset_test')
    O.check_cell(good)
    with pytest.raises(AssertionError, match='part c'):
        O.check_cell(dict(good, **{field: bad}))
    cs = [dict(c, **{field: bad}) if c['sc'] == sc else c for c in partc_cells()]
    with pytest.raises(AssertionError, match='part c'):
        O.decide(sum((enc_cells(e) for e in ENCS), []) + cs, ok_status(), 17000)


# ------------------------------------------------------------------------------------------ licensed sentences
def spec_text():
    return re.sub(r'\s+', ' ', SPEC.read_text())


@pytest.mark.parametrize('frag', [
    O.PAINT_HEAD + O.PAINT_HOLDS_BODY + O.PAINT_TAIL,
    O.PAINT_QUAL_BODY,
    O.PAINT_QUAL_NONE.split('image; ', 1)[1],
    O.PHOTO_SENTENCE,
    O.SC6_SENTENCE,
    O.PARTC_HEAD + O.PARTC_HOLDS_BODY + O.PARTC_TAIL,
    O.PARTC_QUAL_BODY,
])
def test_sentence_fragments_are_the_spec_text(frag):
    assert re.sub(r'\s+', ' ', frag) in spec_text()


def test_paintings_sentences_filled():
    h = O.paintings_sentence('clip_l14', O.HOLDS)
    assert h.startswith('On 12,281 held ArtELingo paintings, c*\'s recipe with the CLIP L/14 image as its only input')
    assert 'better than the plain CLIP L/14 image, and clusters them better than PercepT on the CLIP L/14 image' in h
    assert h.endswith('CLIP L/14 is one of three encoders that passed an earlier held emotion test.') and '<' not in h
    q = O.paintings_sentence('siglip2', O.QUALIFIED, ['emotion', 'genre'])
    assert 'better than the plain SigLIP 2 image; against PercepT on the SigLIP 2 image, better in emotion and genre' in q
    assert 'clusters them' not in q and '<' not in q
    q1 = O.paintings_sentence('siglip1', O.QUALIFIED, ['style'])
    assert 'better in style (AMI' in q1
    q3 = O.paintings_sentence('siglip1', O.QUALIFIED, ['emotion', 'style', 'genre'])
    assert 'better in emotion, style and genre (AMI' in q3
    q0 = O.paintings_sentence('siglip1', O.QUALIFIED, [])
    assert 'against PercepT on the SigLIP 1 image it is better in no aspect' in q0 and '<' not in q0


def test_other_sentences_filled():
    p = O.photos_sentence('siglip1')
    assert 'the same model on SigLIP 1, fine-tuned' in p and p.endswith('better than the plain SigLIP 1 image.')
    s = O.sc6_sentence('siglip2', 'artelingo_held', 'genre', 'ret')
    assert s == ('c* on SigLIP 2 is level with or better than the Qwen image instructed for genre in retrieval '
                 'on 12,281 held ArtELingo paintings.')
    s = O.sc6_sentence('clip_l14', 'affection_test', 'emotion', 'clu')
    assert s == ('The photo model on CLIP L/14 is level with or better than the Qwen image instructed for emotion '
                 'in clustering on 8,501 held Affection photos.')
    c = O.part_c_sentence(O.HOLDS, 17716)
    assert c.startswith('On 17,716 EmoSet-118K test photos (photos with a near-copy') and '<' not in c
    q = O.part_c_sentence(O.QUALIFIED, 17716, 'ret')
    assert ('better than the plain Qwen3-VL-Embedding-2B image and better than the instructed image in retrieval '
            'and level with it in clustering') in q and '<' not in q


def test_aspect_list_order_and_form():
    assert O.aspect_list(['genre', 'emotion']) == 'emotion and genre'
    assert O.aspect_list(['genre', 'style', 'emotion']) == 'emotion, style and genre'


# ------------------------------------------------------------------------------------------ CLI
def test_cli_decide_writes_outcomes(tmp_path, capsys):
    stage3 = tmp_path / 'stage3'
    stage3.mkdir()
    cs = all_cells()
    (stage3 / 'cells_a.json').write_text(json.dumps([c for c in cs if c['data'] == 'artelingo_held']))
    (stage3 / 'cells_b.json').write_text(json.dumps([c for c in cs if c['data'] != 'artelingo_held']))
    (tmp_path / 'status.json').write_text(json.dumps(ok_status()))
    (tmp_path / 'nc.json').write_text(json.dumps({'splits': {'test': {'n_kept': 17000}}}))
    O.main(['decide', '--root', str(tmp_path), '--status', str(tmp_path / 'status.json'),
            '--nearcopy', str(tmp_path / 'nc.json')])
    res = json.loads((stage3 / 'outcomes.json').read_text())
    assert res['part_c']['verdict'] == O.HOLDS and res['smoke'] is False and len(res['cells']) == 61
    assert 'part c' in capsys.readouterr().out
