"""rs_vh (ticket 11): the held side re-derived from the rule's text. Tiny synthetic cases; one integration test on K4.

Expected values come from the rule text, from literals taken from numpy's default_rng(0) stream, or from sklearn
called directly, never from rs_vh's own helpers.
"""
import json

import numpy as np
import pytest
from sklearn.metrics import adjusted_mutual_info_score

import rs_paths
import rs_vh as V


def ami(labels, part):
    return float(adjusted_mutual_info_score(labels, part))


def fake_parts(rng, n, labels, noise):
    """Three r20-shaped partitions and one own partition: the label with `noise` of the rows reassigned."""
    def one():
        p = labels.copy()
        flip = rng.random(n) < noise
        p[flip] = rng.integers(0, 4, size=int(flip.sum()))
        return p
    return {'r20': np.stack([one() for _ in range(3)]), 'own': one()[None, :]}


# ---- the draw order (rule §3.1, §3.4, §3.5) -----------------------------------------------------------------

def test_draw_plan_order_and_empty_pools():
    # numpy default_rng(0), n = 7: I_b, then S_b for G, B1, (B2 has no seeds: nothing drawn), B4
    idx, seeds = V.draw_plan(7, {'G': [0, 1, 2, 3, 4], 'B1': [0, 1, 2, 3, 4], 'B2': [], 'B4': [0, 1, 2, 3, 4]}, 2)
    assert idx.tolist() == [[5, 4, 3, 1, 2, 0, 0], [4, 0, 2, 6, 3, 0, 5]]
    assert [{k: v.tolist() for k, v in s.items()} for s in seeds] == [
        {'G': [0, 0, 4, 3, 4], 'B1': [2, 3, 4, 3, 3], 'B4': [2, 2, 4, 1, 4]},
        {'G': [3, 4, 0, 0, 4], 'B1': [0, 2, 0, 1, 2], 'B4': [2, 2, 0, 0, 0]}]


def test_draw_plan_one_arm_is_d_t_stream():
    # d_T,a: I_b, then S_b for c*; T draws nothing (rule §3.4)
    idx, seeds = V.draw_plan(7, {'C': [0, 1, 2, 3, 4]}, 2)
    assert idx.tolist() == [[5, 4, 3, 1, 2, 0, 0], [3, 4, 6, 5, 4, 3, 3]]
    assert [s['C'].tolist() for s in seeds] == [[0, 0, 4, 3, 4], [4, 1, 4, 3, 0]]


# ---- the statistic against the entries: the maximum is retaken per draw (rule §3.1) ---------------------------

def make_case(n=48):
    rng = np.random.default_rng(7)
    labels = np.arange(n) % 4
    cand = {s: fake_parts(rng, n, labels, 0.35) for s in (0, 1)}
    good_first, good_last = labels.copy(), labels.copy()
    good_first[n // 2:] = rng.integers(0, 4, size=n - n // 2)       # E1 is right on the first half only
    good_last[:n // 2] = rng.integers(0, 4, size=n // 2)            # E2 on the second half only
    e1 = {s: {'r20': np.stack([good_first] * 3), 'own': good_first[None, :]} for s in (0, 1)}
    e2 = {s: {'r20': np.stack([good_last] * 3), 'own': good_last[None, :]} for s in (0, 1)}
    return labels, cand, {'E1': e1, 'E2': e2}


def test_entries_statistic_retakes_the_maximum_per_draw():
    labels, cand, entries = make_case()
    res = V.entries_statistic(labels, cand, entries, draws=40, workers=1)
    idx, seeds = V.draw_plan(len(labels), {'C': [0, 1], 'E1': [0, 1], 'E2': [0, 1]}, 40)
    expected, leaders = [], set()
    for b in range(40):
        def mean(parts, name, reading):
            return np.mean([np.mean([ami(labels[idx[b]], p[idx[b]]) for p in parts[s][reading]])
                            for s in seeds[b][name]])
        a = mean(cand, 'C', 'r20')
        values = {(e, r): mean(entries[e], e, r) for e in entries for r in ('r20', 'own')}
        leaders.add(max(values, key=values.get))
        expected.append(a - max(values.values()))
    assert len(leaders) > 1, 'the case must make the leading entry change between draws'
    assert np.allclose(res['bootstrap_d'], expected, atol=1e-12)
    assert np.allclose(res['interval'], np.percentile(expected, [2.5, 97.5]), atol=1e-12)


def test_entries_point_values_are_means_over_seeds_and_states():
    labels, cand, entries = make_case()
    res = V.entries_statistic(labels, cand, entries, draws=2, workers=1)
    a = np.mean([np.mean([ami(labels, p) for p in cand[s]['r20']]) for s in (0, 1)])
    e1 = np.mean([np.mean([ami(labels, p) for p in entries['E1'][s]['own']]) for s in (0, 1)])
    assert res['A_cand'] == pytest.approx(a, abs=1e-12)
    assert res['A_e']['E1-own'] == pytest.approx(e1, abs=1e-12)
    assert set(res['A_e']) == {'E1-r20', 'E1-own', 'E2-r20', 'E2-own'}
    assert res['d'] == pytest.approx(a - max(res['A_e'].values()), abs=1e-12)


def test_entry_without_seeds_is_left_out():
    labels, cand, entries = make_case()
    res = V.entries_statistic(labels, cand, {**entries, 'E3': {}}, draws=2, workers=1)
    assert not any(k.startswith('E3') for k in res['A_e'])


def test_unlabelled_rows_are_dropped_after_the_draw():
    labels, cand, entries = make_case()
    labels = labels.copy()
    labels[::5] = -1
    fixed = {0: {'r20': entries['E1'][0]['r20']}}
    res = V.fixed_statistic(labels, {0: cand[0]}, fixed, draws=6, workers=1)
    idx, seeds = V.draw_plan(len(labels), {'C': [0]}, 6)
    for b in range(6):
        rows = idx[b][labels[idx[b]] >= 0]
        left = np.mean([ami(labels[rows], p[rows]) for p in cand[0]['r20']])
        right = np.mean([ami(labels[rows], p[rows]) for p in fixed[0]['r20']])
        assert res['bootstrap_d'][b] == pytest.approx(left - right, abs=1e-12)
    keep = labels >= 0
    assert res['A_fixed'] == pytest.approx(np.mean([ami(labels[keep], p[keep]) for p in fixed[0]['r20']]), abs=1e-12)


def test_fixed_statistic_draws_only_for_the_candidate_seeds():
    labels, cand, entries = make_case()
    res = V.fixed_statistic(labels, cand, {0: entries['E2'][0]}, draws=5, workers=1)
    idx, seeds = V.draw_plan(len(labels), {'C': [0, 1]}, 5)
    for b in range(5):
        left = np.mean([np.mean([ami(labels[idx[b]], p[idx[b]]) for p in cand[s]['r20']]) for s in seeds[b]['C']])
        right = np.mean([ami(labels[idx[b]], p[idx[b]]) for p in entries['E2'][0]['r20']])
        assert res['bootstrap_d'][b] == pytest.approx(left - right, abs=1e-12)


def test_paired_statistic_shares_one_seed_draw():
    labels, cand, entries = make_case()
    other = {s: fake_parts(np.random.default_rng(100 + s), len(labels), labels, 0.5) for s in (0, 1)}
    res = V.paired_statistic(labels, {'X': other, 'F': cand}, cand, draws=6, workers=1)
    idx, seeds = V.draw_plan(len(labels), {'S': [0, 1]}, 6)
    for b in range(6):
        d = np.mean([np.mean([ami(labels[idx[b]], p[idx[b]]) for p in other[s]['r20']]) for s in seeds[b]['S']]) \
            - np.mean([np.mean([ami(labels[idx[b]], p[idx[b]]) for p in cand[s]['r20']]) for s in seeds[b]['S']])
        assert res['X']['bootstrap_d'][b] == pytest.approx(d, abs=1e-12)
    assert res['F']['d'] == 0 and res['F']['interval'] == [0.0, 0.0]      # F against itself, in the same draws
    assert res['X']['A_arm'] == pytest.approx(np.mean([np.mean([ami(labels, p) for p in other[s]['r20']])
                                                       for s in (0, 1)]), abs=1e-12)
    assert res['X']['A_final'] == pytest.approx(res['F']['A_arm'], abs=1e-12)


# ---- outcome edges (rule §4.1, §4.4) ------------------------------------------------------------------------

@pytest.mark.parametrize('d, lower, suffix', [
    (0.01, 1e-9, 'GO'), (0.0123, 0.0004, 'GO'), (0.01, 0.0, 'MM'), (0.02, -0.003, 'MM'),
    (0.0099999, 0.005, 'IN'), (1e-9, -0.01, 'IN'), (0.0, -0.01, 'NO'), (-0.004, -0.02, 'NO'), (0.0, 0.0, 'NO')])
def test_claim_outcome_edges(d, lower, suffix):
    assert V.claim_suffix(d, lower) == suffix
    assert V.stage0_outcome('sample', d, lower) == f'G-ES-{suffix}'
    assert V.stage0_outcome('full', d, lower) == f'G-EF-{suffix}'
    assert V.emotion_margin_outcome(d, lower) == f'PE-{suffix}'


@pytest.mark.parametrize('lower, r, expected', [
    (1e-9, 0.0128, 'T-WIN'), (0.0, 0.0128, 'T-LEVEL'), (-0.0127, 0.0128, 'T-LEVEL'),
    (-0.0128, 0.0128, 'T-BELOW'), (-0.0129, 0.0128, 'T-BELOW'), (-0.0045, 0.0046, 'T-LEVEL'), (-0.0046, 0.0046, 'T-BELOW')])
def test_t_outcome_edges(lower, r, expected):
    aspect = {0.0128: 'genre', 0.0046: 'emotion'}[r]
    assert V.t_outcome(aspect, lower) == expected


def test_seed_ranges_are_the_rules():
    assert V.RANGES == dict(emotion=0.0046, style=0.0082, genre=0.0128)


def test_e0_needs_full_go_and_a_sample_outcome_other_than_no():
    assert V.licensed_e0('G-ES-MM', 'G-EF-GO') is True
    assert V.licensed_e0('G-ES-GO', 'G-EF-GO') is True
    assert V.licensed_e0('G-ES-NO', 'G-EF-GO') is False
    assert V.licensed_e0('G-ES-GO', 'G-EF-MM') is False
    assert V.licensed_e0(None, 'G-EF-GO') is False
    assert V.licensed_e0('G-ES-GO', None) is False


# ---- labels (CT rule §3, RB rule §3.2, RG rule §3.2) ----------------------------------------------------------

ROWS = [dict(painting='b', emotion='awe', art_style='Pop Art'), dict(painting='a', emotion='fear', art_style='Cubism'),
        dict(painting='a', emotion='awe', art_style='Cubism'), dict(painting='b', emotion='awe', art_style='Pop Art'),
        dict(painting='c', emotion='something else', art_style='Cubism'),
        dict(painting='c', emotion='anger', art_style='Cubism')]


def test_emotion_plurality_ties_to_smallest_code(monkeypatch):
    monkeypatch.setattr(V, 'EMOTIONS', ('anger', 'awe', 'fear', 'something else'))
    # a: awe and fear tie -> awe (code 1); b: awe; c: anger and 'something else' tie -> anger (code 0)
    assert V.emotion_labels(['a', 'b', 'c'], ROWS).tolist() == [1, 1, 0]


def test_style_label_is_the_first_caption_and_constant():
    labels = V.style_labels(['a', 'b', 'c'], ROWS)
    assert labels.tolist() == [0, 1, 0]            # names sorted: Cubism 0, Pop Art 1
    bad = ROWS + [dict(painting='a', emotion='awe', art_style='Pop Art')]
    with pytest.raises(AssertionError):
        V.style_labels(['a', 'b', 'c'], bad)


def test_genre_label_first_caption_missing_and_conflict_masked(tmp_path):
    t = tmp_path / 'g.csv'
    t.write_text('x/a.jpg,3\nx/b.jpg,4\nx/b.jpg,5\n')
    v = tmp_path / 'v.csv'
    v.write_text('x/a.jpg,3\n')
    labels = V.genre_labels(['a', 'b', 'c'], ROWS, (t, v))
    assert labels.tolist() == [3, -1, -1]


# ---- eligibility (CT rule §4 R-COL, RB rule §2) -------------------------------------------------------------

def test_rcol_eligibility_cuts():
    runs = [dict(version='clip_roberta', cut='strict', seed=0, kept_count=9, share=0.1),
            dict(version='clip_roberta', cut='strict', seed=1, kept_count=10, share=0.5),
            dict(version='clip_roberta', cut='strict', seed=2, kept_count=91, share=0.1),
            dict(version='clip_roberta', cut='strict', seed=3, kept_count=40, share=0.51),
            dict(version='clip_roberta', cut='k40', seed=0, kept_count=1, share=0.5),
            dict(version='clip_roberta', cut='k40', seed=1, kept_count=1, share=0.51),
            dict(version='other', cut='k40', seed=2, kept_count=1, share=0.9)]
    assert V.rcol_eligible_seeds(runs, 'strict') == [1]
    assert V.rcol_eligible_seeds(runs, 'k40') == [0]
    assert V.rcol_eligible_seeds(runs, 'k40', nonfinite=(0,)) == []


# ---- the output sets (V0c) ----------------------------------------------------------------------------------

def unit_rows(n, width=8, seed=0):
    x = np.random.default_rng(seed).normal(size=(n, width)).astype(np.float32)
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float16)


def test_v0c_problems():
    x = unit_rows(10)
    assert V.v0c_problems(x, 10, 8) == []
    assert V.v0c_problems(x[:9], 10, 8)
    assert V.v0c_problems(x.astype(np.float32), 10, 8)
    bad = x.copy()
    bad[3, 0] = np.nan
    assert V.v0c_problems(bad, 10, 8)
    assert V.v0c_problems(x * np.float16(1.01), 10, 8)


def test_partition_refuses_rows_that_are_not_unit_norm():
    with pytest.raises(AssertionError):
        V.r20_partition(unit_rows(40) * np.float16(1.05))


def test_sidecar_is_checked_against_the_array(tmp_path):
    x = unit_rows(10)
    side = dict(rows=10, list_sha256='h', aspect='emotion', seed=2, sha256=V.array_sha(x))
    V.check_sidecar(side, x, 'emotion', 2, 10, 'h')
    for key, wrong in (('rows', 9), ('list_sha256', 'z'), ('aspect', 'style'), ('seed', 3), ('sha256', '0')):
        with pytest.raises(AssertionError):
            V.check_sidecar({**side, key: wrong}, x, 'emotion', 2, 10, 'h')


def test_held_dir_names(tmp_path):
    assert V.held_dir(tmp_path, 'G') == rs_paths.RG_FULL / 'held' / 'GS-0.02'
    assert V.held_dir(tmp_path, 'k6=0.0005') == rs_paths.full_root(tmp_path) / 'held' / 'k6=0.0005'


def test_removal_arm_names():
    assert V.removal_names('G') == {r: f'G-no_{r}' for r in V.REMOVALS}
    assert V.removal_names('k2=8')['buddy'] == 'k2=8-no_buddy'
    assert len(V.REMOVALS) == 8


# ---- comparison and files -----------------------------------------------------------------------------------

def block(d=0.0123, lower=0.004):
    return dict(candidate='G', scale='sample', d=d, A_cand=0.1, A_e={'B1-r20': 0.08, 'B1-own': 0.087},
                interval=[lower, 0.02], bootstrap_d=[0.01, 0.02], eligible_seeds={'G': [0, 1], 'B2': []},
                per_seed={'G': {'0': {'r20': [0.1, 0.2, 0.3]}}}, outcome_id='G-ES-GO', v_failure=None)


def verdict_from(mine):
    out = dict(blocks={s: {**b, 'eligibility': {'eligible_seeds': b.get('eligible_seeds', {})}} for s, b in mine['blocks'].items()},
               outcomes={s: b['outcome_id'] for s, b in mine['blocks'].items()}, E0=mine['E0'], n_resamples=2000,
               held_sha256=V.HELD_SHA256)
    return json.loads(json.dumps(out))      # a copy: the verdict file shares nothing with the record


def mine_pair():
    sample, full = block(), {**block(0.0164, 0.01), 'scale': 'full', 'outcome_id': 'G-EF-GO'}
    return dict(blocks={'sample': sample, 'full': full}, E0=True, n_resamples=2000, held_sha256=V.HELD_SHA256)


def test_compare_stage0_agrees_and_flags_each_kind_of_difference():
    mine = mine_pair()
    items = V.compare_stage0(mine, verdict_from(mine))
    assert items and all(i['ok'] for i in items)
    assert {'sample.d', 'full.interval_lower', 'sample.A_e[B1-own]', 'sample.outcome_id', 'E0'} <= {i['name'] for i in items}
    other = verdict_from(mine)
    other['blocks']['sample']['d'] += 2e-6
    bad = [i for i in V.compare_stage0(mine, other) if not i['ok']]
    assert [i['name'] for i in bad] == ['sample.d'] and bad[0]['diff'] == pytest.approx(2e-6)
    other = verdict_from(mine)
    other['blocks']['sample']['d'] += 5e-7
    assert all(i['ok'] for i in V.compare_stage0(mine, other))
    other = verdict_from(mine)
    other['outcomes']['full'] = 'G-EF-MM'
    other['blocks']['full']['outcome_id'] = 'G-EF-MM'
    assert not all(i['ok'] for i in V.compare_stage0(mine, other))
    other = verdict_from(mine)
    other['blocks']['sample']['eligibility']['eligible_seeds']['B2'] = [0]
    assert not all(i['ok'] for i in V.compare_stage0(mine, other))
    other = verdict_from(mine)
    other['E0'] = False
    assert not all(i['ok'] for i in V.compare_stage0(mine, other))


def test_a_scale_without_outcome_must_match_the_verdict():
    mine = mine_pair()
    mine['blocks']['full'] = dict(candidate='G', scale='full', outcome_id=None, v_failure=['seed 3: non-finite rows'])
    mine['E0'] = False
    theirs = verdict_from(mine)
    theirs['blocks']['full'] = dict(outcome_id=None, v_failure='seed 3 failed V0c')
    assert all(i['ok'] for i in V.compare_stage0(mine, theirs))
    theirs['blocks']['full']['outcome_id'] = 'G-EF-GO'
    assert not all(i['ok'] for i in V.compare_stage0(mine, theirs))


def test_atomic_json_and_exit_code(tmp_path, monkeypatch):
    target = tmp_path / 'sub' / 'rederive_x.json'
    V.write_json(target, {'a': 1})
    assert json.loads(target.read_text()) == {'a': 1}
    assert not list(target.parent.glob('*.tmp*'))
    assert V.exit_code([dict(ok=True)]) == 0 and V.exit_code([dict(ok=True), dict(ok=False)]) == 1 and V.exit_code([]) == 1


def test_report_is_written_beside_the_checked_file(tmp_path):
    verdict = tmp_path / 'verdict.json'
    verdict.write_text('{}')
    items = [V.item('x', 1.0, 1.0 + 5e-7), V.item('y', 'a', 'a')]
    code = V.finish('stage0', verdict, items, inputs={str(verdict): 'abc'}, extra=dict(mine={'k': 1}))
    out = json.loads((tmp_path / 'rederive_stage0.json').read_text())
    assert code == 0 and out['pass'] is True and out['tolerance'] == 1e-6
    assert out['items'][0]['diff'] == pytest.approx(5e-7) and out['inputs_sha256'] == {str(verdict): 'abc'}
    assert V.finish('stage0', verdict, items + [V.item('z', 1.0, 1.1)], inputs={}, extra={}) == 1
    assert json.loads((tmp_path / 'rederive_stage0.json').read_text())['pass'] is False


def test_cli_modes_exist():
    p = V.build_parser()
    assert p.parse_args(['stage0']).mode == 'stage0'
    a = p.parse_args(['stage4', '--candidate', 'k6=0.0005', '--outputs', '/tmp/x'])
    assert a.candidate == 'k6=0.0005' and str(a.outputs) == '/tmp/x'
    assert p.parse_args(['ablation', '--final', 'G']).final == 'G'
    assert p.parse_args(['selfcheck']).mode == 'selfcheck'
    assert p.parse_args(['stage0']).outputs == rs_paths.OUTPUTS
    with pytest.raises(SystemExit):
        p.parse_args(['stage4'])


# ---- integration: K4's sample-scale block from the frozen files (point values only) ---------------------------

def test_k4_sample_point_values_equal_the_frozen_block():
    res = V.stage0_scale(V.Ctx(), 'sample', V.RB_K4_DIRS['sample'], cand_name='K4', draws=0, workers=8)
    frozen = json.loads((rs_paths.RB_PART2 / 'verdict_stage3.json').read_text())['blocks']['K4']
    assert res['d'] == pytest.approx(frozen['d'], abs=1e-6)
    assert res['A_cand'] == pytest.approx(frozen['A_cand'], abs=1e-6)
    for k, v in frozen['A_e'].items():
        assert res['A_e'][k] == pytest.approx(v, abs=1e-6), k
    assert res['eligible_seeds'] == frozen['eligibility']['eligible_seeds']
