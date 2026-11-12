"""Tests of r29_ladder (ticket 08): the claim ladders of spec §6, the numbers sheet, diagnostic (d).

Cell records in the tests are synthetic, shaped from ticket 06's `_cell` (r29_image_cells.py) and ticket 07's brief."""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r29_ladder as L  # noqa: E402

ASP = ('emotion', 'style', 'genre')
K = {'emotion': 9, 'style': 27, 'genre': 10}
W, LV, B, I = 'WIN', 'LEVEL', 'BELOW', 'INCONCLUSIVE'


# ------------------------------------------------------------------------------------------------ synthetic records
def lo_of(outcome, r=0.01):
    return {'WIN': 0.02, 'LEVEL': -0.005, 'BELOW': -0.5, 'INCONCLUSIVE': 0.0}[outcome]


def img_cell(aspect, measure, comp, outcome, arm='siglip1', enc='siglip1', set_='artelingo', r=0.01):
    """Ticket 06's cell: fields as `r29_image_cells._cell`, comparator plain / teacher, lo and hi."""
    metric = 'ami' if measure == 'ami' else 'map'
    mid = f'ami{K[aspect]}' if measure == 'ami' else 'map'
    lo = lo_of(outcome, r)
    type_ = 'ami' if measure == 'ami' else ('map_plain' if comp == 'plain' else 'map_teacher')
    return dict(id=f'{set_}/{enc}/{arm}/{aspect}/{mid}/{comp}', type=type_, protocol='x', set=set_, encoder=enc, arm=arm,
                aspect=aspect, metric=metric, k=None, comparator=comp, value=0.3, comparator_value=0.25, d=lo + 0.01, lo=lo,
                hi=lo + 0.02, r=r, outcome=outcome, n_draws=10)


def txt_cell(aspect, direction, comp, outcome, arm='siglip1', enc='siglip1', style='lohi', r=0.01):
    """Ticket 07's cell: comparator p1 / q1, direction, lo and hi (no `interval` key)."""
    lo = lo_of(outcome, r)
    c = dict(id=f'artelingo/{enc}/{arm}/{aspect}/{direction}/{comp}', type='text', protocol='r23', comparator=comp, set='artelingo', encoder=enc, direction=direction,
             aspect=aspect, arm=arm, value=0.3, d=lo + 0.01, r=r, outcome=outcome)
    c.update(lo=lo, hi=lo + 0.02)
    return c


def art_cells(p1, q1, arm='siglip1'):
    """15 cells x 2 comparators from two outcome lists in cell order (aspect-major, then ami, map, t2i, i2t, t2t)."""
    keys = [(a, m) for a in ASP for m in ('ami', 'map', 't2i', 'i2t', 't2t')]
    img, txt = [], []
    for (a, m), po, qo in zip(keys, p1, q1):
        for comp_img, comp_txt, o in (('plain', 'p1', po), ('teacher', 'q1', qo)):
            if o is None:
                continue
            (img if m in ('ami', 'map') else txt).append(
                img_cell(a, m, comp_img, o, arm) if m in ('ami', 'map') else txt_cell(a, m, comp_txt, o, arm))
    return img, txt


def write_cells(records, arm, img=None, txt=None, other=None):
    records.mkdir(parents=True, exist_ok=True)
    if img is not None:
        (records / f'image_cells_{arm}.json').write_text(json.dumps(dict(arm=arm, encoder='siglip1', cells=img)))
    if txt is not None:
        (records / f'text_cells_{arm}.json').write_text(json.dumps(dict(arm=arm, encoder='siglip1', cells=txt)))
    for name, cells in (other or {}).items():
        (records / f'{name}_cells_{arm}.json').write_text(json.dumps(dict(arm=arm, encoder='siglip1', cells=cells)))


# ================================================================================================ the ladder (pure)
def outs(n_win, n_other, other=B, n_inc=0):
    return [W] * n_win + [other] * n_other + [I] * n_inc


@pytest.mark.parametrize('n_win, level', [(11, 0), (12, 1), (14, 1), (15, 2)])
def test_sc3_level_edges_on_p1(n_win, level):
    r = L.ladder(outs(n_win, 15 - n_win), [B] * 15, most=12)
    assert r['level'] == level and not r['inconclusive_above']


@pytest.mark.parametrize('n_q, level', [(11, 2), (12, 3), (15, 3)])
def test_sc3_level_3_edge_on_q1(n_q, level):
    q = [W] * (n_q // 2) + [LV] * (n_q - n_q // 2) + [B] * (15 - n_q)      # WIN or LEVEL both count
    r = L.ladder([W] * 15, q, most=12)
    assert r['level'] == level and r['counts']['q1_win_or_level'] == n_q


def test_level_3_needs_level_2_even_with_every_q1_cell_won():
    assert L.ladder(outs(14, 1), [W] * 15, most=12)['level'] == 1


@pytest.mark.parametrize('n_win, level', [(2, 0), (3, 1), (4, 2)])
def test_sc6_four_cell_edges(n_win, level):
    assert L.ladder(outs(n_win, 4 - n_win), [B] * 4, most=3)['level'] == level


@pytest.mark.parametrize('n_q, level', [(2, 2), (3, 3), (4, 3)])
def test_sc6_level_3_edge(n_q, level):
    assert L.ladder([W] * 4, outs(n_q, 4 - n_q, other=B), most=3)['level'] == level


@pytest.mark.parametrize('n_win, level', [(4, 0), (5, 1), (6, 2)])
def test_six_cell_edges(n_win, level):
    assert L.ladder(outs(n_win, 6 - n_win), [B] * 6, most=5)['level'] == level


def test_ladder_counts_only_the_cells_given():
    with pytest.raises(AssertionError):
        L.ladder([W] * 14, [W] * 15, most=12)


# --- the inconclusive rule (D22)
def test_inconclusive_counts_as_neither():
    r = L.ladder(outs(12, 0, n_inc=3), [B] * 15, most=12)           # 12 WIN on the decided cells, 3 not run
    assert r['level'] == 1 and r['counts']['p1_win'] == 12 and r['counts']['p1_inconclusive'] == 3


def test_inconclusive_raises_the_level_when_counted_as_wins():
    r = L.ladder(outs(11, 3, n_inc=1), [B] * 15, most=12)            # 12 if the one counted: level 1
    assert r['level'] == 0 and r['inconclusive_above'] and r['report'] == 'level 0, inconclusive above'
    r = L.ladder(outs(14, 0, n_inc=1), [B] * 15, most=12)            # 15 if counted: level 2
    assert r['level'] == 1 and r['inconclusive_above'] and r['report'] == 'level 1, inconclusive above'
    r = L.ladder([W] * 15, outs(11, 3, n_inc=1), most=12)            # Q1: 12 if counted: level 3
    assert r['level'] == 2 and r['inconclusive_above'] and r['report'] == 'level 2, inconclusive above'


def test_inconclusive_does_not_raise_when_it_would_not_help():
    r = L.ladder(outs(10, 3, n_inc=2), [B] * 15, most=12)            # 12 if both counted?  10 + 2 = 12: raises
    assert r['inconclusive_above']
    r = L.ladder(outs(9, 4, n_inc=2), [B] * 15, most=12)             # 11 even if both counted: no
    assert r['level'] == 0 and not r['inconclusive_above'] and r['report'] == 'level 0'
    r = L.ladder([W] * 15, outs(10, 3, n_inc=2) , most=12)           # Q1: 10 + 2 = 12 raises
    assert r['inconclusive_above']
    r = L.ladder([W] * 15, outs(9, 4, n_inc=2), most=12)             # 11 even if counted: no
    assert r['level'] == 2 and not r['inconclusive_above']


def test_inconclusive_p1_cell_blocks_level_2_and_flags_it():
    r = L.ladder(outs(14, 0, n_inc=1), [W] * 15, most=12)
    assert r['level'] == 1 and r['inconclusive_above']


def test_level_3_holds_on_decided_q1_cells():
    r = L.ladder([W] * 15, [W] * 12 + [I] * 3, most=12)
    assert r['level'] == 3 and not r['inconclusive_above']


def test_each_q1_cell_gets_its_claim_word():
    assert L.q1_claim(W) == 'above' and L.q1_claim(LV) == 'level with'
    assert L.q1_claim(B) is None and L.q1_claim(I) is None


# --- the sentences (row 8: only the reached level's slots)
def test_sentences_fill_only_their_slots():
    s1 = L.sentence(1, n=15, w1=13, wq=None)
    assert s1 == 'With no Qwen3 pass at test time, training adds over the plain embedding in 13 of 15 cells.'
    s2 = L.sentence(2, n=15, w1=15, wq=None)
    assert s2 == 'With no Qwen3 pass at test time, training adds over the plain embedding in every cell.'
    s3 = L.sentence(3, n=4, w1=4, wq=3)
    assert s3 == ('With no Qwen3 pass at test time, training adds over the plain embedding in every cell, and it is level '
                  'with or above the instructed Qwen3 in 3 of 4 cells.')
    assert L.sentence(0, n=15, w1=5, wq=None) is None
    with pytest.raises(AssertionError):
        L.sentence(2, n=15, w1=15, wq=14)          # a level-2 sentence takes no Q1 slot


# ================================================================================================ records
def test_claim_tables_match_the_spec():
    assert {c: (len(v['cells']), v['most']) for c, v in L.CLAIMS.items()} == {
        'SC3': (15, 12), 'SC4': (6, 5), 'SC5': (6, 5), 'SC6': (4, 3)}


def test_sc3_record_from_image_and_text_cells(tmp_path):
    img, txt = art_cells([W] * 15, [W] * 8 + [LV] * 4 + [B] * 3)
    write_cells(tmp_path, 'siglip1', img, txt)
    rec = L.build_record('SC3', 'siglip1', records=tmp_path)
    assert rec['level'] == 3 and rec['n_cells'] == 15 and rec['most'] == 12
    assert rec['counts'] == dict(p1_win=15, p1_inconclusive=0, q1_win_or_level=12, q1_win=8, q1_level=4, q1_inconclusive=0)
    assert rec['sentence'].endswith('in 12 of 15 cells.') and 'every cell' in rec['sentence']
    assert len(rec['counted']['p1_win']) == 15 and len(rec['counted']['q1_win_or_level']) == 12
    assert 'artelingo/emotion/ami' in rec['counted']['p1_win']
    claims = {c['cell']: c['q1_claim'] for c in rec['cells']}
    assert claims['artelingo/emotion/ami'] == 'above' and claims['artelingo/genre/t2t'] is None
    assert rec['cell_files']['image']['sha256'] and rec['cell_files']['text']['sha256']


def test_each_cell_file_uses_its_own_comparator_form(tmp_path):
    img, txt = art_cells([W] * 15, [W] * 15)
    assert {c['comparator'] for c in txt} == {'p1', 'q1'} and {c['comparator'] for c in img} == {'plain', 'teacher'}
    swapped = [dict(c, comparator='plain' if c['comparator'] == 'p1' else 'teacher',
                    id=c['id'].rsplit('/', 1)[0] + ('/plain' if c['comparator'] == 'p1' else '/teacher')) for c in txt]
    write_cells(tmp_path, 'siglip1', img, swapped)                      # text cells in the image form: refused
    with pytest.raises(AssertionError, match='p1 / q1'):
        L.build_record('SC3', 'siglip1', records=tmp_path)
    swapped = [dict(c, comparator='p1' if c['comparator'] == 'plain' else 'q1',
                    id=c['id'].rsplit('/', 1)[0] + ('/p1' if c['comparator'] == 'plain' else '/q1')) for c in img]
    write_cells(tmp_path, 'siglip1', swapped, txt)                      # image cells in the text form: refused
    with pytest.raises(AssertionError, match='p1 / q1'):
        L.build_record('SC3', 'siglip1', records=tmp_path)


def test_a_text_direction_field_must_match_the_id(tmp_path):
    img, txt = art_cells([W] * 15, [W] * 15)
    txt[0]['direction'] = 'i2t' if txt[0]['direction'] != 'i2t' else 't2t'
    write_cells(tmp_path, 'siglip1', img, txt)
    with pytest.raises(AssertionError):
        L.build_record('SC3', 'siglip1', records=tmp_path)


def test_a_missing_cell_record_is_not_run_and_counts_as_neither(tmp_path):
    img, txt = art_cells([W] * 15, [W] * 15)
    write_cells(tmp_path, 'siglip1', img, None)                      # text cells not written
    rec = L.build_record('SC3', 'siglip1', records=tmp_path)
    assert rec['sources']['text'] == 'not run'
    assert rec['counts']['p1_win'] == 6 and rec['counts']['p1_inconclusive'] == 9
    assert rec['level'] == 0 and rec['inconclusive_above']
    assert all(c['p1'] == 'INCONCLUSIVE (not run)' for c in rec['cells'] if c['cell'].endswith(('t2i', 'i2t', 't2t')))
    assert rec['sentence'] is None


def test_a_missing_single_cell_is_not_run(tmp_path):
    img, txt = art_cells([W] * 15, [W] * 15)
    img = [c for c in img if c['id'] != 'artelingo/siglip1/siglip1/style/map/plain']
    write_cells(tmp_path, 'siglip1', img, txt)
    rec = L.build_record('SC3', 'siglip1', records=tmp_path)
    assert rec['counts']['p1_win'] == 14 and rec['counts']['p1_inconclusive'] == 1
    assert rec['level'] == 1 and rec['inconclusive_above']


def test_recorded_inconclusive_outcome_counts_as_neither(tmp_path):
    img, txt = art_cells([W] * 14 + [I], [W] * 15)
    write_cells(tmp_path, 'siglip1', img, txt)
    rec = L.build_record('SC3', 'siglip1', records=tmp_path)
    assert rec['counts']['p1_inconclusive'] == 1 and rec['level'] == 1


def test_a_cell_with_the_wrong_fields_or_id_is_refused(tmp_path):
    img, txt = art_cells([W] * 15, [W] * 15)

    def bad(mut):
        i2 = [dict(c) for c in img]
        mut(i2[0])
        write_cells(tmp_path, 'siglip1', i2, txt)
        with pytest.raises(AssertionError):
            L.build_record('SC3', 'siglip1', records=tmp_path)
    bad(lambda c: c.update(arm='siglip1_prefix'))                                  # another arm than the file's
    bad(lambda c: c.update(id=c['id'].rsplit('/', 1)[0]))                          # short id
    bad(lambda c: c.update(outcome='WIN' if c['outcome'] != 'WIN' else 'BELOW'))     # not the rule's word for its lo and r
    bad(lambda c: c.pop('r'))
    bad(lambda c: c.update(comparator='crl', id=c['id'].rsplit('/', 1)[0] + '/crl'))
    bad(lambda c: c.update(outcome='MAYBE'))
    bad(lambda c: c.update(id='mmart' + c['id'][len('artelingo'):], set='mmart'))   # a set the claim does not hold
    bad(lambda c: c.update(encoder='siglip2'))
    bad(lambda c: c.update(lo=float('nan')))


def test_duplicate_cell_ids_are_refused(tmp_path):
    img, txt = art_cells([W] * 15, [W] * 15)
    write_cells(tmp_path, 'siglip1', img + [img[0]], txt)
    with pytest.raises(AssertionError):
        L.build_record('SC3', 'siglip1', records=tmp_path)


def test_the_record_arm_must_match_the_asked_arm(tmp_path):
    img, txt = art_cells([W] * 15, [W] * 15)
    write_cells(tmp_path, 'siglip1', img, txt)
    (tmp_path / 'image_cells_siglip1_prefix.json').write_text((tmp_path / 'image_cells_siglip1.json').read_text())
    with pytest.raises(AssertionError):
        L.build_record('SC3', 'siglip1_prefix', records=tmp_path)


def test_ladder_record_is_written_once(tmp_path):
    img, txt = art_cells([W] * 15, [W] * 15)
    write_cells(tmp_path, 'siglip1', img, txt)
    path, rec = L.write_ladder('SC3', 'siglip1', records=tmp_path)
    assert path.name == 'SC3_siglip1.json' and json.loads(path.read_text())['level'] == 3
    with pytest.raises(FileExistsError):
        L.write_ladder('SC3', 'siglip1', records=tmp_path)


def _sc4_cells(arm, enc, outcomes_p1, outcomes_q1):
    out = []
    for (a, m), po, qo in zip([(a, m) for a in ASP for m in ('ami', 'map')], outcomes_p1, outcomes_q1):
        out.append(img_cell(a, m, 'plain', po, arm, enc))
        out.append(img_cell(a, m, 'teacher', qo, arm, enc))
    return out


def test_sc4_is_per_encoder_and_refuses_without_a_pass(tmp_path):
    cells = _sc4_cells('siglip2', 'siglip2', [W] * 5 + [B], [W] * 6)
    write_cells(tmp_path, 'siglip2', cells)
    (tmp_path / 'image_cells_siglip2.json').write_text(json.dumps(dict(arm='siglip2', encoder='siglip2', cells=cells)))
    with pytest.raises(RuntimeError, match='pass_rule'):
        L.build_record('SC4', 'siglip2', encoder='siglip2', records=tmp_path)
    (tmp_path / 'pass_rule.json').write_text(json.dumps(dict(status='fail')))
    with pytest.raises(RuntimeError, match='pass_rule'):
        L.build_record('SC4', 'siglip2', encoder='siglip2', records=tmp_path)
    (tmp_path / 'pass_rule.json').write_text(json.dumps(dict(status='pass')))
    rec = L.build_record('SC4', 'siglip2', encoder='siglip2', records=tmp_path)
    assert rec['level'] == 1 and rec['n_cells'] == 6 and rec['encoder'] == 'siglip2'
    path, _ = L.write_ladder('SC4', 'siglip2', encoder='siglip2', records=tmp_path)
    assert path.name == 'SC4_siglip2_siglip2.json'
    with pytest.raises(AssertionError):                                  # the encoder must be the arm's
        L.build_record('SC4', 'siglip2', encoder='clip_l14', records=tmp_path)


def test_sc3_needs_no_pass_rule_record(tmp_path):
    img, txt = art_cells([W] * 15, [W] * 15)
    write_cells(tmp_path, 'siglip1', img, txt)
    assert L.build_record('SC3', 'siglip1', records=tmp_path)['level'] == 3


def test_sc6_four_cell_ladder_with_photo_cells(tmp_path):
    (tmp_path / 'pass_rule.json').write_text(json.dumps(dict(status='pass')))

    def pc(set_, measure, comp, o, aspect='emotion'):
        c = img_cell('emotion', 'ami' if measure == 'ami' else 'map', comp, o, 'siglip1', 'siglip1', set_)
        c['id'] = f'{set_}/siglip1/siglip1/{aspect}/{measure}{9 if measure == "ami" else ""}/{comp}'
        c['aspect'] = aspect
        return c
    cells = [pc(s, m, c, o) for (s, m), o in zip([('affection', 'ami'), ('affection', 'map'), ('emoset', 'ami'), ('emoset', 'map')],
                                                [W, W, W, B]) for c in ('plain', 'teacher')]
    write_cells(tmp_path, 'siglip1', other={'photo': cells})
    rec = L.build_record('SC6', 'siglip1', records=tmp_path)
    assert rec['n_cells'] == 4 and rec['level'] == 1 and rec['counts']['p1_win'] == 3
    assert 'in 3 of 4 cells' in rec['sentence']


def test_gate_parked_makes_that_parts_cells_inconclusive(tmp_path):
    img, txt = art_cells([W] * 15, [W] * 15)
    write_cells(tmp_path, 'siglip1', img, txt)
    (tmp_path / 'gates').mkdir()
    (tmp_path / 'gates' / 'v_text.json').write_text(json.dumps(dict(gate='v_text', status='parked', tries=3)))
    rec = L.build_record('SC3', 'siglip1', records=tmp_path)
    assert rec['counts']['p1_win'] == 6 and rec['counts']['p1_inconclusive'] == 9 and rec['gates']['v_text'] == 'parked'
    (tmp_path / 'gates' / 'v_text.json').write_text(json.dumps(dict(gate='v_text', status='fail', tries=2)))
    with pytest.raises(RuntimeError, match='still open'):
        L.build_record('SC3', 'siglip1', records=tmp_path)
    (tmp_path / 'gates' / 'v_text.json').write_text(json.dumps(dict(gate='v_text', status='fail', tries=3)))
    assert L.build_record('SC3', 'siglip1', records=tmp_path)['counts']['p1_inconclusive'] == 9
    (tmp_path / 'gates' / 'v_text.json').write_text(json.dumps(dict(gate='v_text', status='pass', tries=1)))
    assert L.build_record('SC3', 'siglip1', records=tmp_path)['level'] == 3


# ================================================================================================ the sheet
R28_SHEET = Path('/project/CoSiR/src/test/20261112_siglip_main/paper_cells.md')


def _r28_row(label_start, section):
    """The `SigLIP 1` column(s) of one row of r28's sheet: the numbers of the third cell group."""
    text = R28_SHEET.read_text()
    block = text.split(section, 1)[1].split('\n## ', 1)[0]
    (line,) = [ln for ln in block.splitlines() if ln.startswith(f'| {label_start}')]
    cols = [c.strip() for c in line.strip().strip('|').split('|')]
    return cols


def _nums(cell):
    return [float(x) for x in re.findall(r'-?\d+\.\d+', cell)]


def test_k0_cells_reproduce_the_r28_sheet_rows():
    k0 = L.k0_values()
    def art(measure):
        return [round(k0[('artelingo', a, measure)]['value'], 3) for a in ASP]
    assert art('ami') == _nums(_r28_row('CoSiR (ours), Qwen3 -> SigLIP 1', '## Table 2')[2])
    sec3 = '## Table 3'
    assert art('map') == _nums(_r28_row('CoSiR (ours) i2i', sec3)[2])
    assert art('t2i') == _nums(_r28_row('Text to image: CoSiR (ours)', sec3)[2])
    assert art('i2t') == _nums(_r28_row('Text to image: CoSiR (ours)'.replace('Text to image', 'Image to text'), sec3)[2])
    assert art('t2t') == _nums(_r28_row('Text to text: CoSiR (ours)', sec3)[2])
    r4 = _r28_row('CoSiR (ours)', '## Table 4')
    assert [round(k0[('mmart', a, 'ami')]['value'], 3) for a in ASP] == _nums(r4[2])
    assert [round(k0[('mmart', a, 'map')]['value'], 3) for a in ASP] == _nums(r4[5])
    r7 = _r28_row('CoSiR (ours), Qwen3 -> SigLIP 1', '## Table 7')
    got = [round(k0[(s, None, m)]['value'], 3) for s, m in (('affection', 'ami'), ('affection', 'map'), ('emoset', 'ami'), ('emoset', 'map'))]
    assert got == _nums(r7[2])


def test_sheet_with_no_new_records_shows_k0_rows_and_not_run(tmp_path):
    out = tmp_path / 'paper_cells.md'
    L.write_sheet(records=tmp_path / 'records', out=out)
    text = out.read_text()
    k0 = L.k0_values()
    assert f"{k0[('artelingo', 'emotion', 'ami')]['value']:.3f}" in text and 'not run' in text
    assert text.count('not run') >= 15 * 2
    assert 'SC3' in text and 'SC6' in text and 'Table 8' in text


def test_sheet_shows_a_new_cell_beside_k0_with_both_outcomes(tmp_path):
    img, txt = art_cells([W] * 15, [W] * 8 + [LV] * 4 + [B] * 3)
    write_cells(tmp_path / 'records', 'siglip1', img, txt)
    out = tmp_path / 'paper_cells.md'
    L.write_sheet(records=tmp_path / 'records', out=out)
    text = out.read_text()
    row = [ln for ln in text.splitlines() if ln.startswith('| emotion ami')][0]
    assert 'WIN' in row and 'above' in row and f"{L.k0_values()[('artelingo', 'emotion', 'ami')]['value']:.3f}" in row
    row = [ln for ln in text.splitlines() if ln.startswith('| genre t2t')][0]
    assert 'BELOW' in row


def test_sheet_names_the_ladder_level_when_the_record_exists(tmp_path):
    img, txt = art_cells([W] * 15, [W] * 15)
    write_cells(tmp_path / 'records', 'siglip1', img, txt)
    L.write_ladder('SC3', 'siglip1', records=tmp_path / 'records')
    L.write_sheet(records=tmp_path / 'records', out=tmp_path / 'p.md')
    assert 'level 3' in (tmp_path / 'p.md').read_text()


# ================================================================================================ diagnostic (d)
def test_diag_d_refuses_without_the_ladder_record(tmp_path):
    with pytest.raises(FileNotFoundError, match='ladder'):
        L.diag_d('SC3', 'siglip1', records=tmp_path, differs={})


def test_diag_d_writes_one_row_per_test_cell_after_the_ladder(tmp_path):
    img, txt = art_cells([W] * 15, [W] * 15)
    write_cells(tmp_path, 'siglip1', img, txt)
    L.write_ladder('SC3', 'siglip1', records=tmp_path)
    seen = []

    def differ(key, arm, k0):
        seen.append(key)
        return dict(value=k0['value'] - 0.01, d=-0.01, lo=-0.02, hi=0.0, n_draws=7, protocol='fake')
    path, rec = L.diag_d('SC3', 'siglip1', records=tmp_path, differs={m: differ for m in ('ami', 'map', 't2i', 'i2t', 't2t')})
    assert path.name == 'diag_d_SC3_siglip1.json' and len(rec['cells']) == 15 and len(seen) == 15
    c = rec['cells'][0]
    assert c['cell'] == 'artelingo/emotion/ami' and c['d'] == -0.01 and c['lo'] == -0.02
    assert abs(c['k0'] - L.k0_values()[('artelingo', 'emotion', 'ami')]['value']) < 1e-12
    assert 'sheet' not in rec and rec['ladder_level'] == 3        # it decides nothing; it names the level it follows


def test_diag_d_rejects_a_differ_whose_point_value_is_not_k_star_minus_k0(tmp_path):
    img, txt = art_cells([W] * 15, [W] * 15)
    write_cells(tmp_path, 'siglip1', img, txt)
    L.write_ladder('SC3', 'siglip1', records=tmp_path)
    bad = lambda key, arm, k0: dict(value=0.5, d=0.0, lo=-1, hi=1, n_draws=1, protocol='x')       # value - k0 != d
    with pytest.raises(AssertionError):
        L.diag_d('SC3', 'siglip1', records=tmp_path, differs={m: bad for m in ('ami', 'map', 't2i', 'i2t', 't2t')})


# --- the painting-unit interval of two seeded arms (text cells of diagnostic (d))
def test_painting_interval_two_seeded_carries_all_queries_of_a_painting():
    rng = np.random.default_rng(1)
    pid = np.repeat(np.arange(40), 5).astype(np.int64)                  # 40 paintings, 5 queries each
    base = rng.random(200)
    cand = np.stack([base + 0.1] * 5).astype(np.float64)
    comp = np.stack([base] * 5).astype(np.float64)
    d, lo, hi, m = L.painting_interval_seeded(cand, comp, pid, draws=500)
    assert m == 40 and abs(d - 0.1) < 1e-9 and lo > 0.09 and hi < 0.11
    # a painting-level shift: the interval is wide because paintings are the unit, not queries
    shift = np.repeat(rng.normal(size=40), 5)
    d2, lo2, hi2, _ = L.painting_interval_seeded(np.stack([base + shift] * 5), comp, pid, draws=500)
    assert hi2 - lo2 > 5 * (hi - lo)


def test_painting_interval_two_seeded_equal_arms_straddle_zero():
    rng = np.random.default_rng(2)
    pid = np.repeat(np.arange(30), 4).astype(np.int64)
    x = rng.random((5, 120))
    d, lo, hi, _ = L.painting_interval_seeded(x, x.copy(), pid, draws=400)
    assert d == 0.0 and lo < 0 < hi


def test_painting_interval_two_seeded_is_deterministic():
    pid = np.repeat(np.arange(10), 3).astype(np.int64)
    x, y = np.random.default_rng(3).random((5, 30)), np.random.default_rng(4).random((5, 30))
    assert L.painting_interval_seeded(x, y, pid, draws=200) == L.painting_interval_seeded(x, y, pid, draws=200)


# ================================================================================================ outcome rule at its edges
@pytest.mark.parametrize('lo, r, want', [(1e-12, 0.01, W), (0.0, 0.01, LV), (-0.0099999, 0.01, LV), (-0.01, 0.01, B), (0.0, 0.0, B)])
def test_outcome_edges_match_the_rule(lo, r, want):
    assert L.outcome(lo, r) == want


def test_a_cell_at_the_edge_is_checked_against_the_rule(tmp_path):
    img, txt = art_cells([W] * 15, [W] * 15)
    img[0].update(lo=0.0, hi=0.01, outcome=W)                       # lo = 0 is not above 0: the word must be LEVEL
    write_cells(tmp_path, 'siglip1', img, txt)
    with pytest.raises(AssertionError):
        L.build_record('SC3', 'siglip1', records=tmp_path)
