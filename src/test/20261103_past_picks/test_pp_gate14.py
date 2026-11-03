"""Tests of pp_gate14 (gates (iv) and (vii), r14 part): the compare helper, the encoder-table context manager, the parts
fed r14's stored inputs and stored outputs, the merge. The scorers themselves are replaced by r14's stored values (the full
replays run under the controller); each test fails when the guarded logic is broken."""
import copy
import json

import numpy as np
import pytest

import pp_paths as PP
import pp_register  # noqa: F401
import pp_gate14 as G
import pp_score14 as S

import bs_artret


def cell(sc='SC1', aspect='emotion', task='clu', d=0.05, n_draws=2000, **extra):
    return dict(sc=sc, encoder='clip_l14', aspect=aspect, task=task, data='artelingo_held', candidate='cstar_clip_l14',
                comparator='frozen_clip_l14', candidate_value=0.3, comparator_value=0.3 - d, d=d, lo=d - 0.01, hi=d + 0.01,
                r=None, r_source=None, n_draws=n_draws, seeds=[0, 1, 2, 3, 4], **extra)


# ------------------------------------------------------------------------------------------------ compare
def test_compare_passes_a_difference_of_1e_10_and_catches_1e_8():
    ok = G.compare({'a': {'d': 0.5 + 1e-10}}, {'a': {'d': 0.5}})
    assert ok['passed'] and ok['max_abs_diff'] == pytest.approx(1e-10, abs=1e-12)
    bad = G.compare({'a': {'d': 0.5 + 1e-8}}, {'a': {'d': 0.5}})
    assert not bad['passed'] and bad['max_abs_diff'] == pytest.approx(1e-8, abs=1e-11)


def test_compare_strings_booleans_none_are_exact():
    assert G.compare({'x': 'M7-own', 'b': True, 'n': None}, {'x': 'M7-own', 'b': True, 'n': None})['passed']
    assert not G.compare({'x': 'M7-r20'}, {'x': 'M7-own'})['passed']
    assert not G.compare({'b': False}, {'b': True})['passed']
    assert not G.compare({'n': 0.0}, {'n': None})['passed']          # a number where the stored value is None
    assert not G.compare({'b': True}, {'b': 1.0})['passed']          # a boolean is not a number


def test_compare_refuses_nan_and_a_missing_key_and_a_length_change():
    assert not G.compare({'v': float('nan')}, {'v': float('nan')})['passed']
    assert not G.compare({}, {'v': 1.0})['passed']
    assert not G.compare({'l': [1.0, 2.0]}, {'l': [1.0, 2.0, 3.0]})['passed']
    assert not G.compare({'l': [1.0, 2.5]}, {'l': [1.0, 2.0]})['passed']


def test_compare_ignores_the_written_key_anywhere():
    assert G.compare({'written': 'x', 'a': {'written': 'y', 'v': 1.0}}, {'written': 'z', 'a': {'v': 1.0}})['passed']


def test_compare_extra_keys_of_ours_only_when_allowed():
    ours, theirs = {'v': 1.0, 'n_rows': 12281}, {'v': 1.0}
    assert G.compare(ours, theirs, extra_keys=('n_rows',))['passed']
    assert not G.compare(ours, theirs)['passed']


def test_compare_cells_tolerate_only_n_rows_and_n_labelled_at_any_depth():
    base = [cell()]
    ok = [dict(cell(), n_rows=7, n_labelled=5)]
    assert G.compare_cells(ok, base)['passed']
    for bad in (dict(cell(), n_rows=7, extra=1), dict(cell(), n_rows=7, seeds_note={'x': 1})):
        assert not G.compare_cells([bad], base)['passed']
    nested = dict(cell(), reading={'v': 1.0, 'extra': 2.0})
    assert not G.compare_cells([nested], [dict(cell(), reading={'v': 1.0})])['passed']


def test_compare_skips_intervals_and_draw_counts_only_at_reduced_draws():
    ours, theirs = cell(n_draws=40), cell(n_draws=2000)
    ours['lo'] += 0.01
    ours['hi'] -= 0.01
    reduced = G.compare(ours, theirs, full_draws=False)
    assert reduced['wiring_ok'] and not reduced['intervals_compared'] and not reduced['passed']
    assert not any(k.endswith(('/lo', '/hi', '/n_draws')) or k in ('lo', 'hi', 'n_draws') for k in reduced['items'])
    full = G.compare(ours, theirs, full_draws=True)
    assert full['intervals_compared'] and not full['wiring_ok'] and not full['passed']
    assert G.compare(theirs, copy.deepcopy(theirs), full_draws=True)['passed']


def test_compare_passes_only_at_full_draws():
    same = G.compare(cell(), cell(), full_draws=False)
    assert same['wiring_ok'] and not same['passed']
    assert G.compare(cell(), cell(), full_draws=True)['passed']


def test_compare_names_the_item_that_differs():
    res = G.compare([cell(d=0.05)], [cell(d=0.06)], full_draws=True)
    assert [k for k, v in res['items'].items() if not v['ok']] == ['[0]/comparator_value', '[0]/d', '[0]/lo', '[0]/hi']


# ------------------------------------------------------------------------------------------------ the encoder table
def test_computed_frozen_sets_the_r11_arm_none_and_restores_it():
    before = bs_artret.ENC['clip_l14']['r11_arm']
    assert before == 'clip'
    with G.computed_frozen('clip_l14'):
        assert bs_artret.ENC['clip_l14']['r11_arm'] is None
    assert bs_artret.ENC['clip_l14']['r11_arm'] == before


def test_computed_frozen_restores_the_table_when_the_body_raises():
    with pytest.raises(ValueError):
        with G.computed_frozen('clip_l14'):
            raise ValueError('boom')
    assert bs_artret.ENC['clip_l14']['r11_arm'] == 'clip'


def test_computed_frozen_leaves_other_encoders_alone():
    others = {e: bs_artret.ENC[e]['r11_arm'] for e in bs_artret.ENC if e != 'clip_l14'}
    with G.computed_frozen('clip_l14'):
        assert {e: bs_artret.ENC[e]['r11_arm'] for e in others} == others


# ------------------------------------------------------------------------------------------------ gate (iv) parts
def stored_world(part):
    """r14's stored cells of the part for clip_l14 (as the file holds them) and the cells that a scorer would give."""
    spec = {'clu': [(sc, a, 'clu') for a in S.ASPECTS for sc in ('SC1', 'SC3', 'SC6')],
            'ret': [(sc, a, 'ret') for a in S.ASPECTS for sc in ('SC2', 'SC6')],
            'affection': [('SC4', 'emotion', 'clu'), ('SC5', 'emotion', 'ret'), ('SC6', 'emotion', 'clu'),
                          ('SC6', 'emotion', 'ret')]}[part]
    cells = []
    for i, (sc, a, task) in enumerate(spec):
        n = 2000 if task == 'clu' else 10000
        extra = {'comparator_reading': 'M7-own', 'comparator_readings': {'M7-r20': 0.2, 'M7-own': 0.21}} if sc == 'SC3' else {}
        c = cell(sc, a, task, d=0.01 * (i + 1), n_draws=n, **extra)
        c['data'] = 'affection_test' if part == 'affection' else 'artelingo_held'
        if sc == 'SC6':
            c['r'], c['r_source'] = 0.01, 'X-c* own validation seed range'
        cells.append(c)
    other = dict(cells[0], encoder='siglip2', d=0.9)                    # another encoder's cell: not compared
    return cells, [other] + cells


@pytest.fixture
def world(monkeypatch, tmp_path):
    """The gate's inputs replaced: stage 1 record, stored cells, the three scorers (returning r14's stored cells + n_rows)."""
    calls = []
    state = {'perturb': None, 'draws': None}
    rec = {'encoder': 'clip_l14', 'marker': 'r14 record'}
    monkeypatch.setattr(PP, 'OUT', tmp_path)
    monkeypatch.setattr(G, 'r14_record', lambda enc: rec)
    monkeypatch.setattr(S, 'check_record', lambda enc, r: calls.append(('check_record', enc, bs_artret.ENC[enc]['r11_arm'])) or r)
    monkeypatch.setattr(G, 'stored_cells', lambda part: stored_world(part)[1])

    def make(part):
        def run(enc, r, *a, **k):
            assert r is rec and enc == 'clip_l14'
            calls.append((part, bs_artret.ENC['clip_l14']['r11_arm'], k))
            got = copy.deepcopy(stored_world(part)[0])
            for c in got:
                c['n_rows'] = 7
                if state['draws']:
                    c['n_draws'] = state['draws']
            if state['perturb']:
                state['perturb'](got)
            return {'cells': got} if part == 'affection' else got
        return run
    monkeypatch.setattr(S, 'run_artelingo_clu', make('clu'))
    monkeypatch.setattr(S, 'run_artelingo_ret', make('ret'))
    monkeypatch.setattr(S, 'run_affection', make('affection'))
    return dict(calls=calls, state=state, out=tmp_path)


@pytest.mark.parametrize('part', ('clu', 'ret', 'affection'))
def test_gate_iv_part_passes_on_r14s_own_values_at_full_draws_and_writes_its_file(world, part):
    res = G.gate_iv(part)
    assert res['passed'] and res['intervals_compared'] and res['max_abs_diff'] == 0.0
    assert res['n_cells'] == {'clu': 9, 'ret': 6, 'affection': 4}[part]
    path = world['out'] / 'stage0' / f'gate_iv_r14_{part}.json'
    assert json.loads(path.read_text())['passed'] is True
    assert world['calls'][0][0] == 'check_record'          # the record is checked before any scorer runs


@pytest.mark.parametrize('part', ('clu', 'ret', 'affection'))
def test_gate_iv_part_fails_when_one_stored_value_differs_by_1e_8(world, part):
    world['state']['perturb'] = lambda cells: cells[1].update(d=cells[1]['d'] + 1e-8)
    res = G.gate_iv(part)
    assert not res['passed'] and res['max_abs_diff'] > 5e-9
    assert json.loads((world['out'] / 'stage0' / f'gate_iv_r14_{part}.json').read_text())['passed'] is False


@pytest.mark.parametrize('part', ('clu', 'ret', 'affection'))
def test_gate_iv_part_fails_on_a_changed_label_or_a_missing_cell(world, part):
    world['state']['perturb'] = lambda cells: cells[0].update(comparator='frozen_other')
    assert not G.gate_iv(part)['passed']
    world['state']['perturb'] = lambda cells: cells.pop()
    with pytest.raises(AssertionError):
        G.gate_iv(part)


def test_gate_iv_clu_reading_of_the_matched_comparator_is_compared():
    stored = stored_world('clu')[0]
    ours = copy.deepcopy(stored)
    ours[1]['comparator_reading'] = 'M7-r20'
    assert not G.compare_cells(ours, stored, full_draws=True)['passed']


def test_gate_iv_ret_scores_the_computed_frozen_path_and_restores_the_table(world):
    G.gate_iv('ret')
    (_, arm, _), = [c for c in world['calls'] if c[0] == 'ret']
    assert arm is None                                          # inside the call: r14's computed-frozen branch
    assert bs_artret.ENC['clip_l14']['r11_arm'] == 'clip'       # afterwards: r14's table again
    # the record was checked with r14's table (its stored frozen pick form), not the computed form
    assert ('check_record', 'clip_l14', 'clip') in world['calls']


def test_gate_iv_clu_and_affection_do_not_touch_the_encoder_table(world):
    G.gate_iv('clu')
    G.gate_iv('affection')
    assert [c[1] for c in world['calls'] if c[0] in ('clu', 'affection')] == ['clip', 'clip']


def test_gate_iv_affection_reads_r14s_exports_and_writes_partitions_under_stage0(world, monkeypatch):
    seen = {}
    real = S.run_affection

    def spy(enc, rec, **k):
        seen.update(k)
        return real(enc, rec, **k)
    monkeypatch.setattr(S, 'run_affection', spy)
    G.gate_iv('affection')
    assert seen['root'] == PP.R14_OUT
    assert str(seen['cache']).startswith(str(world['out'] / 'stage0'))


def test_gate_iv_reduced_draws_wires_but_does_not_pass(world, monkeypatch):
    monkeypatch.setattr(PP, 'SMOKE', 1)
    world['state']['draws'] = 40
    res = G.gate_iv('clu', n_draws=40)
    assert res['wiring_ok'] and not res['intervals_compared'] and not res['passed']


def test_gate_iv_refuses_other_draw_counts_outside_smoke(world):
    with pytest.raises(AssertionError):
        G.gate_iv('clu', n_draws=40)


# ------------------------------------------------------------------------------------------------ gate (iv) merge
def part_file(out, part, **over):
    d = dict(gate='iv_r14', part=part, passed=True, wiring_ok=True, intervals_compared=True, max_abs_diff=0.0,
             n_draws={'clu': 2000, 'ret': 10000, 'affection': {'clu': 2000, 'ret': 10000}}[part], n_cells=1)
    d.update(over)
    (out / 'stage0').mkdir(parents=True, exist_ok=True)
    (out / 'stage0' / f'gate_iv_r14_{part}.json').write_text(json.dumps(d))


def test_gate_iv_merge_passes_with_all_three_parts_passed(tmp_path, monkeypatch):
    monkeypatch.setattr(PP, 'OUT', tmp_path)
    for p in ('clu', 'ret', 'affection'):
        part_file(tmp_path, p)
    res = G.merge_iv()
    assert res['passed'] and (tmp_path / 'stage0' / 'gate_iv_r14.json').exists()


@pytest.mark.parametrize('missing', ('clu', 'ret', 'affection'))
def test_gate_iv_merge_needs_every_part(tmp_path, monkeypatch, missing):
    monkeypatch.setattr(PP, 'OUT', tmp_path)
    for p in ('clu', 'ret', 'affection'):
        if p != missing:
            part_file(tmp_path, p)
    res = G.merge_iv()
    assert not res['passed'] and missing in res['missing']


@pytest.mark.parametrize('bad', (dict(passed=False), dict(intervals_compared=False), dict(wiring_ok=False)))
def test_gate_iv_merge_refuses_a_failed_or_reduced_part(tmp_path, monkeypatch, bad):
    monkeypatch.setattr(PP, 'OUT', tmp_path)
    for p in ('clu', 'ret', 'affection'):
        part_file(tmp_path, p, **(bad if p == 'ret' else {}))
    assert not G.merge_iv()['passed']


def test_gate_iv_merge_refuses_a_part_run_at_other_draws_even_if_it_says_passed(tmp_path, monkeypatch):
    monkeypatch.setattr(PP, 'OUT', tmp_path)
    for p in ('clu', 'ret', 'affection'):
        part_file(tmp_path, p, **(dict(n_draws=40) if p == 'clu' else {}))
    assert not G.merge_iv()['passed']


# ------------------------------------------------------------------------------------------------ gate (vii)
def stored_stage1(enc):
    return {'encoder': enc, 'kind': 'stage1_record', 'written': 'x',
            'artclu': {'encoder': enc, 'ranges': {'emotion': 0.004, 'style': 0.008, 'genre': 0.016}},
            'artret_ranges': {'encoder': enc, 'ranges': {'emotion': 0.003, 'style': 0.006, 'genre': 0.009}},
            'artret_picks': {'encoder': enc, 'cstar': {'pick': 'center+k10'},
                             'frozen': {'pick': 'center', 'source': 'computed'}, 'teacher': {'pick': 'center+k10'}},
            'affection': {'encoder': enc, 'ranges': {'clu': 0.01, 'map': 0.02}, 'frozen': {'pick': 'center+k10', 'r': None}}}


@pytest.fixture
def vii(monkeypatch, tmp_path):
    seen = {}
    state = {'perturb': None}
    monkeypatch.setattr(PP, 'OUT', tmp_path)
    monkeypatch.setattr(G, 'r14_record', lambda enc: stored_stage1(enc))

    def art(enc, root=None, workers=8):
        seen['art'] = (enc, root)
        rec = stored_stage1(enc)
        out = {k: rec[k] for k in ('artclu', 'artret_ranges', 'artret_picks')}
        if state['perturb']:
            state['perturb'](out)
        return out

    def aff(enc, root=None, workers=8):
        seen['aff'] = (enc, root)
        return stored_stage1(enc)['affection']
    monkeypatch.setattr(S, 'stage1_artelingo', art)
    monkeypatch.setattr(S, 'stage1_affection', aff)
    return dict(seen=seen, state=state, out=tmp_path)


@pytest.mark.parametrize('enc', ('siglip1', 'clip_l14'))
def test_gate_vii_replays_the_stored_record_from_r14s_validation_exports(vii, enc):
    res = G.gate_vii(enc)
    assert res['passed'] and res['max_abs_diff'] == 0.0
    assert vii['seen']['art'] == (enc, PP.R14_OUT / 'cstar' / enc)
    assert vii['seen']['aff'] == (enc, PP.R14_OUT)
    assert json.loads((vii['out'] / 'stage0' / f'gate_vii_r14_{enc}.json').read_text())['passed'] is True


def test_gate_vii_fails_when_the_frozen_pick_or_a_range_differs(vii):
    vii['state']['perturb'] = lambda o: o['artret_picks']['frozen'].update(pick='center+k10')
    assert not G.gate_vii('siglip1')['passed']
    vii['state']['perturb'] = lambda o: o['artclu']['ranges'].update(emotion=0.004 + 1e-8)
    res = G.gate_vii('clip_l14')
    assert not res['passed'] and res['max_abs_diff'] > 5e-9


def test_gate_vii_refuses_an_encoder_without_an_r14_record(vii):
    with pytest.raises(AssertionError):
        G.gate_vii('clip_b32')


def test_gate_vii_compares_the_affection_block_too(vii, monkeypatch):
    def aff(enc, root=None, workers=8):
        block = stored_stage1(enc)['affection']
        block['ranges']['map'] += 1e-8
        return block
    monkeypatch.setattr(S, 'stage1_affection', aff)
    assert not G.gate_vii('siglip1')['passed']


def test_gate_vii_merge_needs_both_encoders_and_passes_only_when_both_passed(vii):
    assert not G.merge_vii()['passed']
    G.gate_vii('siglip1')
    res = G.merge_vii()
    assert not res['passed'] and 'clip_l14' in res['missing']
    G.gate_vii('clip_l14')
    assert G.merge_vii()['passed']
    assert (vii['out'] / 'stage0' / 'gate_vii_r14.json').exists()


def test_gate_vii_merge_fails_when_one_encoder_failed(vii):
    vii['state']['perturb'] = lambda o: o['artclu']['ranges'].update(style=1.0)
    G.gate_vii('siglip1')
    vii['state']['perturb'] = None
    G.gate_vii('clip_l14')
    assert not G.merge_vii()['passed']
