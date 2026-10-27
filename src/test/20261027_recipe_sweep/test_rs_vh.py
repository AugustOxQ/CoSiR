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
    side = dict(rows=10, list_sha256='h', aspect='emotion', seed=2, arm='G', sha256=V.array_sha(x))
    V.check_sidecar(side, x, 'emotion', 2, 10, 'h', 'G')
    for key, wrong in (('rows', 9), ('list_sha256', 'z'), ('aspect', 'style'), ('seed', 3), ('arm', 'K4'), ('sha256', '0')):
        with pytest.raises(AssertionError):
            V.check_sidecar({**side, key: wrong}, x, 'emotion', 2, 10, 'h', 'G')


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
                per_seed={'G': {'0': {'r20': [0.1, 0.2, 0.3]}}, 'B1': {'0': {'r20': [0.1, 0.2, 0.3]}}}, outcome_id='G-ES-GO',
                v_failure=None, partitions=dict(checked=7, unchecked=0, mismatches=[]))


def verdict_from(mine):
    out = dict(blocks={s: {**b, 'eligibility': {'eligible_seeds': b.get('eligible_seeds', {})}} for s, b in mine['blocks'].items()},
               outcomes={s: b['outcome_id'] for s, b in mine['blocks'].items()}, E0=mine['E0'], n_resamples=2000,
               held_sha256=V.HELD_SHA256)
    return json.loads(json.dumps(out, default=V._plain))      # a copy: the verdict file shares nothing with the record


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


def test_compare_stage0_partitions_item_per_scale():
    mine = mine_pair()
    items = V.compare_stage0(mine, verdict_from(mine))
    assert {'sample.partitions.mismatches', 'sample.partitions.unchecked', 'full.partitions.mismatches',
            'full.partitions.unchecked'} <= {i['name'] for i in items} and all(i['ok'] for i in items)
    mine['blocks']['full']['partitions'] = dict(checked=6, unchecked=1, mismatches=[])
    assert names_of(V.compare_stage0(mine, verdict_from(mine))) == ['full.partitions.unchecked']
    mine['blocks']['full']['partitions'] = dict(checked=7, unchecked=0, mismatches=['G s0 r20: differs from x'])
    assert names_of(V.compare_stage0(mine, verdict_from(mine))) == ['full.partitions.mismatches']


@pytest.mark.parametrize('drop', [('blocks', 'sample'), ('blocks',), ('outcomes',), ('blocks', 'full', 'bootstrap_d'),
                                  ('blocks', 'full', 'per_seed')])
def test_compare_stage0_missing_key_is_an_item_not_a_traceback(drop):
    mine = mine_pair()
    verdict = verdict_from(mine)
    node = verdict
    for k in drop[:-1]:
        node = node[k]
    del node[drop[-1]]
    assert names_of(V.compare_stage0(mine, verdict))


def test_per_seed_items_cover_every_arm_i_computed():
    mine = mine_pair()
    verdict = verdict_from(mine)
    del verdict['blocks']['sample']['per_seed']['B1']                     # an arm the verdict lacks
    assert 'sample.per_seed_max_abs_diff' in names_of(V.compare_stage0(mine, verdict))
    verdict = verdict_from(mine)
    verdict['blocks']['full']['per_seed']['B1']['0']['r20'][1] += 1e-3
    assert 'full.per_seed_max_abs_diff' in names_of(V.compare_stage0(mine, verdict))


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


# ---- stage 4 against ticket 09's verdict.json (keys of the example file, build 10) ----------------------------

def aspect_record(a, d, lower, upper, outcome, labelled):
    return dict(T_equals_frozen=True, d=d, A_cand=0.4, A_fixed=0.4 - d, interval=[lower, upper], bootstrap_d=[lower, upper],
                outcome_id=outcome, size_in_seed_ranges=d / V.RANGES[a], labelled_rows=labelled, v_failure=None,
                partitions=dict(checked=5, unchecked=0, mismatches=[]))


def mine_stage4():
    margin = {**block(0.0164, 0.0107), 'scale': 'full', 'outcome_id': 'PE-GO', 'eligible_seeds': {'G': [0, 1], 'B1': [0, 1], 'B2': []}}
    return dict(candidate='G', n_resamples=2000, held_sha256=V.HELD_SHA256, emotion_margin=margin, aspects=dict(
        emotion=aspect_record('emotion', 0.0118, 0.0083, 0.015, 'T-WIN', V.N_HELD),
        style=aspect_record('style', 0.0032, -0.0019, 0.0085, 'T-LEVEL', V.N_HELD),
        genre=aspect_record('genre', -0.006, -0.0105, -0.0014, 'T-LEVEL', V.N_GENRE_LABELLED)))


def verdict_stage4(mine):
    """Ticket 09's keys: aspects{a: d, lo, hi, outcome, A_cand, A_T, d_in_ranges, labelled_rows, labels_masked},
    emotion_margin{d, lo, hi, outcome, A_cand, A_e, max_entry, arms, eligible_seeds}; the rest is extra."""
    out = dict(schema=1, stage=4, kind='KEEP-G', candidate=mine['candidate'], n_resamples=2000, held_sha256=V.HELD_SHA256,
               ranges=dict(V.RANGES), sources={'x': 1}, gates={}, code_sha256={}, v4b='pending', aspects={})
    for a, r in mine['aspects'].items():
        out['aspects'][a] = dict(d=r['d'], lo=r['interval'][0], hi=r['interval'][1], outcome=r['outcome_id'], A_cand=r['A_cand'],
                                 A_T=r['A_fixed'], d_in_ranges=r['size_in_seed_ranges'], labelled_rows=r['labelled_rows'],
                                 labels_masked=V.N_HELD - r['labelled_rows'])
    m = mine['emotion_margin']
    out['emotion_margin'] = dict(d=m['d'], lo=m['interval'][0], hi=m['interval'][1], outcome=m['outcome_id'], A_cand=m['A_cand'],
                                 A_e=dict(m['A_e']), max_entry='B1-own', arms=list(m['eligible_seeds']),
                                 eligible_seeds=json.loads(json.dumps(m['eligible_seeds'])), eligible=None)
    return json.loads(json.dumps(out))


def names_of(items, ok=False):
    return [i['name'] for i in items if i['ok'] is ok]


def test_short_comparison_checks_the_prefix_of_the_draws_and_not_the_bounds():
    # the self-check with fewer draws: the first draws of a longer file are the same draws (one stream), the bounds are not
    mine = block()
    mine.update(bootstrap_d=[0.01, 0.02], interval=[0.0105, 0.0195])
    theirs = {**mine, 'bootstrap_d': [0.01, 0.02, 0.03, 0.04], 'interval': [0.011, 0.039]}
    full = V.compare_block('b', mine, theirs)
    assert {'b.bootstrap_d_max_abs_diff', 'b.interval_lower'} <= set(names_of(full))
    short = V.compare_block('b', mine, theirs, short=True)
    assert all(i['ok'] for i in short) and 'b.bootstrap_d_prefix_max_abs_diff' in {i['name'] for i in short}
    assert not {'b.interval_lower', 'b.interval_upper', 'b.outcome_id'} & {i['name'] for i in short}
    theirs['bootstrap_d'][1] = 0.0201
    assert names_of(V.compare_block('b', mine, theirs, short=True)) == ['b.bootstrap_d_prefix_max_abs_diff']
    t = dict(aspect_record('style', 0.0032, -0.0019, 0.0085, 'T-LEVEL', V.N_HELD), A_fixed=0.4 - 0.0032)
    t_theirs = dict(d=0.0032, A_cand=0.4, A_T=0.4 - 0.0032, lo=-0.002, hi=0.009, bootstrap_d=[-0.0019, 0.0085, 0.1], outcome='T-WIN')
    assert all(i['ok'] for i in V.compare_t_block('s', t, t_theirs, 'style', short=True))
    assert 's.interval_lower' in names_of(V.compare_t_block('s', t, t_theirs, 'style'))


def test_compare_stage4_agrees_with_09s_keys():
    mine = mine_stage4()
    items = V.compare_stage4(mine, verdict_stage4(mine))
    assert items and all(i['ok'] for i in items)
    named = {i['name'] for i in items}
    assert {'emotion.d_T', 'style.interval_lower', 'genre.A_T', 'genre.labels_masked', 'emotion.d_in_ranges',
            'genre.outcome_id', 'emotion_margin.d', 'emotion_margin.interval_upper', 'emotion_margin.max_entry',
            'emotion_margin.arms', 'emotion_margin.outcome_id', 'emotion_margin.A_e[B1-own]',
            'emotion_margin.eligible_seeds[B2]', 'ranges.genre', 'candidate', 'n_resamples'} <= named


@pytest.mark.parametrize('where, key, value, flagged', [
    (('aspects', 'style'), 'd', 0.0032 + 3e-6, 'style.d_T'),
    (('aspects', 'style'), 'lo', -0.0019 + 3e-6, 'style.interval_lower'),
    (('aspects', 'genre'), 'hi', 0.0, 'genre.interval_upper'),
    (('aspects', 'genre'), 'A_T', 0.5, 'genre.A_T'),
    (('aspects', 'emotion'), 'outcome', 'T-LEVEL', 'emotion.outcome_id'),
    (('aspects', 'emotion'), 'd_in_ranges', 1.0, 'emotion.d_in_ranges'),
    (('aspects', 'genre'), 'labels_masked', 0, 'genre.labels_masked'),
    (('aspects', 'genre'), 'labelled_rows', 12281, 'genre.labelled_rows'),
    (('emotion_margin',), 'lo', 0.0, 'emotion_margin.interval_lower'),
    (('emotion_margin',), 'outcome', 'PE-MM', 'emotion_margin.outcome_id'),
    (('emotion_margin',), 'max_entry', 'B1-r20', 'emotion_margin.max_entry'),
    (('emotion_margin',), 'arms', ['G', 'B1'], 'emotion_margin.arms'),
    ((), 'candidate', 'K9', 'candidate'),
    (('ranges',), 'style', 0.01, 'ranges.style'),
])
def test_compare_stage4_flags_each_value_the_rule_names(where, key, value, flagged):
    mine = mine_stage4()
    verdict = verdict_stage4(mine)
    node = verdict
    for k in where:
        node = node[k]
    node[key] = value
    assert names_of(V.compare_stage4(mine, verdict)) == [flagged]


@pytest.mark.parametrize('where, key', [
    (('aspects', 'emotion'), 'A_T'), (('aspects', 'style'), 'lo'), (('aspects', 'genre'), 'outcome'),
    (('aspects', 'genre'), 'labels_masked'), (('emotion_margin',), 'hi'), (('emotion_margin',), 'A_e'),
    (('emotion_margin',), 'max_entry'), (('emotion_margin',), 'outcome'), (('ranges',), 'genre'), ((), 'candidate')])
def test_compare_stage4_fails_on_a_missing_key(where, key):
    mine = mine_stage4()
    verdict = verdict_stage4(mine)
    node = verdict
    for k in where:
        node = node[k]
    del node[key]
    assert names_of(V.compare_stage4(mine, verdict))


def test_compare_stage4_ignores_extra_keys():
    mine = mine_stage4()
    verdict = verdict_stage4(mine)
    verdict['smoke'] = True
    verdict['aspects']['style']['note'] = 'extra'
    verdict['emotion_margin']['whatever'] = [1, 2]
    assert all(i['ok'] for i in V.compare_stage4(mine, verdict))


def failed_stage4(mine, cause='V4d: training failed, seed 2'):
    """My record when stage 4 has no outcome (V4d): every aspect and the margin without one, the cause at the top."""
    out = dict(mine, v_failure=cause)
    out['aspects'] = {a: dict(outcome_id=None, v_failure=cause) for a in V.ASPECTS}
    out['emotion_margin'] = dict(candidate='G', scale='full', claim=None, outcome_id=None, v_failure=cause)
    return out


def failed_verdict_stage4(cause='V4d: seed 2 training failed twice'):
    """The scorer's verdict with no outcome: top-level v_failure, outcome None in every aspect and in the margin."""
    return dict(schema=1, stage=4, candidate='G', held_sha256=V.HELD_SHA256, n_resamples=2000, smoke=False, v_failure=cause,
                aspects={a: dict(outcome=None, v_failure=cause) for a in V.ASPECTS},
                emotion_margin=dict(outcome=None, v_failure=cause))


def test_compare_stage4_no_outcome_both_sides_compares_only_the_outcomes():
    mine = failed_stage4(mine_stage4())
    items = V.compare_stage4(mine, failed_verdict_stage4())
    assert all(i['ok'] for i in items)
    assert {i['name'] for i in items} >= {'emotion.outcome_id', 'style.outcome_id', 'genre.outcome_id', 'emotion_margin.outcome_id'}
    assert not any('d_T' in i['name'] or 'interval' in i['name'] for i in items)


def test_compare_stage4_one_side_without_outcome_is_a_disagreement():
    verdict = failed_verdict_stage4()
    assert not all(i['ok'] for i in V.compare_stage4(mine_stage4(), verdict))              # the scorer failed, I did not
    assert not all(i['ok'] for i in V.compare_stage4(failed_stage4(mine_stage4()), verdict_stage4(mine_stage4())))
    half = failed_verdict_stage4()
    half['aspects']['style'] = dict(outcome='T-LEVEL', d=0.0)                               # one aspect still has an outcome
    assert 'style.outcome_id' in names_of(V.compare_stage4(failed_stage4(mine_stage4()), half))
    del half['aspects']['style']['outcome']
    assert 'style.outcome_id' in names_of(V.compare_stage4(failed_stage4(mine_stage4()), half))   # no key: not a stated failure
    no_top = failed_verdict_stage4()
    del no_top['v_failure']
    assert not all(i['ok'] for i in V.compare_stage4(failed_stage4(mine_stage4()), no_top))


def test_stage4_notes_print_the_cause():
    mine, verdict = failed_stage4(mine_stage4()), failed_verdict_stage4()
    assert 'V4d: training failed, seed 2' in ' '.join(V.stage4_notes(mine, verdict))
    assert 'V4d: seed 2 training failed twice' in ' '.join(V.stage4_notes(mine, verdict))
    assert V.stage4_notes(mine_stage4(), verdict_stage4(mine_stage4())) == []


def test_draws_follow_a_smoke_verdict_only():
    assert V.draws_for(dict(smoke=True, n_draws=20), 'n_draws') == 20
    assert V.draws_for(dict(smoke=False, n_draws=20), 'n_draws') == V.DRAWS
    assert V.draws_for(dict(n_resamples=20), 'n_resamples') == V.DRAWS


# ---- the ablation wiring: synthetic outputs tree, one removal present, seven missing ---------------------------

def write_set(folder, n, seed0, bad_norm=False, arm='x', skip=()):
    """Five seeds of three aspects in the saved layout (npz key img, sidecar with rows, list hash, aspect, seed, arm,
    hash); the (aspect, seed) pairs in `skip` are left out."""
    folder.mkdir(parents=True)
    for a in V.ASPECTS:
        for s in V.SEEDS:
            if (a, s) in skip:
                continue
            x = unit_rows(n, 8, seed0 + 10 * V.ASPECTS.index(a) + s)
            x = x * np.float16(1.05) if bad_norm and (a, s) == ('style', 3) else x
            np.savez(folder / f'{a}__s{s}.npz', img=x)
            (folder / f'{a}__s{s}.json').write_text(json.dumps(dict(rows=n, list_sha256='h', aspect=a, seed=s, arm=arm,
                                                                  sha256=V.array_sha(x))))


class FakeCtx:
    def __init__(self, n):
        self.sources = {}
        rng = np.random.default_rng(3)
        self.labels = {a: rng.integers(0, 5, size=n) for a in V.ASPECTS}
        self.labels['genre'][::7] = -1

    def record(self, path):
        self.sources[str(path)] = 'x'
        return path


@pytest.fixture
def tree(tmp_path, monkeypatch):
    monkeypatch.setattr(V, 'N_HELD', 150)
    monkeypatch.setattr(V, 'WIDTH', 8)
    monkeypatch.setattr(V, 'HELD_SHA256', 'h')
    V._CACHE.clear()
    final = 'k6=0.0005'
    root = rs_paths.full_root(tmp_path) / 'held'
    write_set(root / final, 150, 1, arm=final)
    write_set(root / f'{final}-no_direct', 150, 500, arm=f'{final}-no_direct')
    yield tmp_path, final
    V._CACHE.clear()


def test_ablation_wiring_one_arm_present_seven_missing(tree):
    out, final = tree
    ctx = FakeCtx(150)
    res = V.ablation(ctx, out, final, draws=6, workers=1)
    assert res['removals']['direct'] == dict(name=f'{final}-no_direct', status='ok', problems={})
    assert [x for x, r in res['removals'].items() if r['status'] == 'missing'] == [x for x in V.REMOVALS if x != 'direct']
    assert set(res['effects']) == {'direct'} and set(res['effects']['direct']) == set(V.ASPECTS)
    assert res['n_resamples'] == 6 and res['labelled_rows'] == {a: int((ctx.labels[a] >= 0).sum()) for a in V.ASPECTS}
    # one aspect by hand: the arm's and F's point AMIs over the labelled rows, per seed (mean of the 3 r20 partitions)
    a = 'style'
    keep = ctx.labels[a] >= 0
    cache = {}
    for name, seed0 in (('arm', 500), ('final', 1)):
        for s in V.SEEDS:
            x = unit_rows(150, 8, seed0 + 10 * V.ASPECTS.index(a) + s)
            cache[name, s] = np.mean([ami(ctx.labels[a][keep], p[keep]) for p in V.r20_partition(x)])
    r = res['effects']['direct'][a]
    assert r['A_arm'] == pytest.approx(np.mean([cache['arm', s] for s in V.SEEDS]), abs=1e-12)
    assert r['A_final'] == pytest.approx(np.mean([cache['final', s] for s in V.SEEDS]), abs=1e-12)
    assert r['d'] == pytest.approx(r['A_arm'] - r['A_final'], abs=1e-12)
    assert r['arm_seed_ami'] == pytest.approx({str(s): cache['arm', s] for s in V.SEEDS}, abs=1e-12)
    assert r['final_seed_ami'] == pytest.approx({str(s): cache['final', s] for s in V.SEEDS}, abs=1e-12)
    assert len(r['bootstrap_d']) == 6
    assert ctx.sources, 'every file read is recorded'


def test_ablation_arm_failing_v0c_has_no_effect(tmp_path, monkeypatch, tree):
    out, final = tree
    write_set(rs_paths.full_root(out) / 'held' / f'{final}-no_buddy', 150, 900, bad_norm=True, arm=f'{final}-no_buddy')
    res = V.ablation(FakeCtx(150), out, final, draws=3, workers=1)
    assert res['removals']['buddy']['status'] == 'failed V0c' and 'style' in res['removals']['buddy']['problems']
    assert set(res['effects']) == {'direct'}


def test_ablation_final_must_pass_v0c(tree):
    out, final = tree
    write_set(rs_paths.full_root(out) / 'held' / 'k6=0.0007', 150, 77, bad_norm=True, arm='k6=0.0007')
    with pytest.raises(AssertionError):
        V.ablation(FakeCtx(150), out, 'k6=0.0007', draws=3, workers=1)


# ---- the held ablation against ticket 10's held.json (keys of the example file, build 10) ------------------------

def effect_record(d, lower, upper):
    return dict(A_arm=0.3 + d, A_final=0.3, d=d, interval=[lower, upper], bootstrap_d=[lower, upper],
                arm_seed_ami={str(s): 0.3 + d + 0.01 * s for s in V.SEEDS}, final_seed_ami={str(s): 0.3 + 0.01 * s for s in V.SEEDS})


def mine_ablation():
    status = {x: dict(name=f'F-no_{x}', status='missing') for x in V.REMOVALS}
    status['direct'] = dict(name='F-no_direct', status='ok', problems={})
    status['buddy'] = dict(name='F-no_buddy', status='failed V0c', problems={'style': {3: ['non-finite rows']}})
    return dict(final='F', removals=status, n_resamples=2000, held_sha256=V.HELD_SHA256,
                labelled_rows=dict(emotion=V.N_HELD, style=V.N_HELD, genre=V.N_GENRE_LABELLED),
                effects={'direct': {'emotion': effect_record(0.001, -0.0005, 0.002), 'style': effect_record(-0.01, -0.02, 0.0),
                                    'genre': effect_record(0.0, -0.001, 0.001)}})


def verdict_ablation(mine, listed=('direct', 'buddy'), smoke=True):
    """Ticket 10's keys: final_key, removals (list), n_draws, seeds, labelled_rows, results{x: arm, status, cause,
    aspects{a: effect, lo, hi, arm_ami, final_ami, arm_seed_ami, final_seed_ami}}; the rest is extra."""
    results = {}
    for x in listed:
        if x in mine['effects']:
            asp = {a: dict(effect=r['d'], lo=r['interval'][0], hi=r['interval'][1], arm_ami=r['A_arm'], final_ami=r['A_final'],
                           arm_seed_ami=dict(r['arm_seed_ami']), final_seed_ami=dict(r['final_seed_ami']))
                   for a, r in mine['effects'][x].items()}
            results[x] = dict(arm=mine['removals'][x]['name'], status='ok', cause=None, aspects=asp)
        else:
            results[x] = dict(arm=mine['removals'][x]['name'], status='failed', cause='seed 3: non-finite loss', aspects=None)
    final_key = rs_paths.G_ARM if mine['final'] == 'G' else mine['final']       # the scorer names G's folder by its 12-27 arm
    return json.loads(json.dumps(dict(stage='ablation_held', smoke=smoke, seeds=list(V.SEEDS), n_draws=2000, held_sha256=V.HELD_SHA256,
                                      labelled_rows=mine['labelled_rows'], final_key=final_key, removals=list(listed),
                                      results=results, extra=1)))


def test_compare_ablation_agrees_with_10s_keys():
    mine = mine_ablation()
    items = V.compare_ablation(mine, verdict_ablation(mine, listed=tuple(V.REMOVALS), smoke=False))
    assert items and all(i['ok'] for i in items), names_of(items)
    named = {i['name'] for i in items}
    assert {'direct.emotion.effect', 'direct.style.interval_lower', 'direct.genre.arm_ami', 'direct.genre.final_ami',
            'direct.style.arm_seed_ami[3]', 'direct.style.final_seed_ami[0]', 'buddy.status', 'direct.status',
            'direct.arm', 'final', 'n_draws', 'removals listed', 'labelled_rows.genre'} <= named


@pytest.mark.parametrize('x, a, key, value, flagged', [
    ('direct', 'emotion', 'effect', 0.001 + 4e-6, 'direct.emotion.effect'),
    ('direct', 'style', 'hi', 0.5, 'direct.style.interval_upper'),
    ('direct', 'genre', 'arm_ami', 0.9, 'direct.genre.arm_ami'),
    ('direct', 'genre', 'final_ami', 0.9, 'direct.genre.final_ami'),
])
def test_compare_ablation_flags_each_number(x, a, key, value, flagged):
    mine = mine_ablation()
    verdict = verdict_ablation(mine)
    verdict['results'][x]['aspects'][a][key] = value
    assert flagged in names_of(V.compare_ablation(mine, verdict))


def test_compare_ablation_flags_seed_values_status_and_names():
    mine = mine_ablation()
    verdict = verdict_ablation(mine)
    verdict['results']['direct']['aspects']['style']['arm_seed_ami']['3'] = 0.0
    assert names_of(V.compare_ablation(mine, verdict)) == ['direct.style.arm_seed_ami[3]']
    verdict = verdict_ablation(mine)
    verdict['results']['buddy']['status'] = 'ok'
    assert 'buddy.status' in names_of(V.compare_ablation(mine, verdict))
    verdict = verdict_ablation(mine)
    verdict['results']['direct']['arm'] = 'F-no_directt'
    assert names_of(V.compare_ablation(mine, verdict)) == ['direct.arm']
    verdict = verdict_ablation(mine)
    verdict['final_key'] = 'G'
    assert names_of(V.compare_ablation(mine, verdict)) == ['final']
    verdict = verdict_ablation(mine)
    verdict['n_draws'] = 20
    assert 'n_draws' in names_of(V.compare_ablation(mine, verdict))


def test_compare_ablation_final_g_is_named_gs_002_in_the_verdict():
    mine = dict(mine_ablation(), final='G')
    verdict = verdict_ablation(mine)
    assert verdict['final_key'] == 'GS-0.02'
    assert names_of(V.compare_ablation(mine, verdict)) == []
    verdict['final_key'] = 'G'                                           # the scorer never writes this when F = G
    assert names_of(V.compare_ablation(mine, verdict)) == ['final']
    other = mine_ablation()                                              # any other final: the run's own name
    assert names_of(V.compare_ablation(other, verdict_ablation(other))) == []
    assert V.arm_name('G') == 'GS-0.02' and V.arm_name('k6=0.0005') == 'k6=0.0005'


def test_compare_ablation_fails_on_a_missing_key_and_ignores_extras():
    mine = mine_ablation()
    for where, key in ((('results', 'direct', 'aspects', 'style'), 'lo'), (('results', 'direct', 'aspects', 'genre'), 'arm_seed_ami'),
                       (('results', 'direct'), 'status'), ((), 'final_key'), (('labelled_rows',), 'genre')):
        verdict = verdict_ablation(mine)
        node = verdict
        for k in where:
            node = node[k]
        del node[key]
        assert names_of(V.compare_ablation(mine, verdict)), (where, key)
    verdict = verdict_ablation(mine)
    verdict['results']['direct']['aspects']['style']['extra'] = 1
    verdict['host'] = 'x'
    assert names_of(V.compare_ablation(mine, verdict)) == []


def test_compare_ablation_removals_listed():
    mine = mine_ablation()
    smoke = verdict_ablation(mine, smoke=True)                   # a smoke may run fewer than eight removals
    assert all(i['ok'] for i in V.compare_ablation(mine, smoke))
    assert 'removals listed' in names_of(V.compare_ablation(mine, verdict_ablation(mine, smoke=False)))   # a real run lists all eight
    hidden = verdict_ablation(mine, listed=('buddy',), smoke=True)      # my outputs have the direct arm: the verdict must list it
    assert any(not i['ok'] and i['name'].startswith('direct') for i in V.compare_ablation(mine, hidden))


# ---- the comparators against 12-24's partition cache (rule V0b: equal partitions) ----------------------------------

def write_percept_tree(tmp_path, monkeypatch, scale='sample'):
    """B1 and B6 with five seeds each (B2 to B5 have no eligible seeds), their probabilities and 12-24's partition cache."""
    monkeypatch.setattr(V, 'N_HELD', 150)
    monkeypatch.setattr(V, 'HELD_SHA256', 'h')
    root, cache = tmp_path / 'ct', tmp_path / 'rb_partitions'
    rcol = tmp_path / 'rcol.json'
    rcol.write_text(json.dumps(dict(runs=[], entries={a: dict(seeds=[]) for a, (_, _, cut) in V.B_ARMS.items() if cut})))
    monkeypatch.setitem(V.CT_ROOT, scale, root)
    monkeypatch.setitem(V.RCOL, scale, rcol)
    monkeypatch.setattr(V, 'RB_PARTITIONS', cache)
    V._CACHE.clear()
    for name in ('B1', 'B6'):
        folder, arm, _ = V.B_ARMS[name]
        for s in V.SEEDS:
            prob = np.random.default_rng(100 * s + len(name)).random((150, 20)).astype(np.float32)
            path = root / 'held/percept' / folder / f'{arm}__s{s}.npz'
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, prob=prob)
            path.with_suffix('.json').write_text(json.dumps(dict(held_sha256='h', rows=150, array_sha256s=dict(prob=V.array_sha(prob)))))
            target = cache / scale / name
            target.mkdir(parents=True, exist_ok=True)
            for reading, part in (('r20', V.r20_partition(V.l2_probabilities(prob))), ('own', V.own_partition(prob))):
                np.savez(target / f'percept__s{s}__{reading}.npz', labels=part)
    return cache / scale


def test_entry_partitions_are_compared_with_the_rb_cache_including_b6(tmp_path, monkeypatch):
    cache = write_percept_tree(tmp_path, monkeypatch)
    ctx = FakeCtx(150)
    res = V.entry_parts(ctx, 'sample', 1)
    assert res['saved'] == dict(checked=20, unchecked=0, mismatches=[])               # B1 and B6: five seeds, two readings
    assert any(k.endswith('B6/percept__s4__own.npz') for k in ctx.sources), 'partition files read are recorded'
    V._CACHE.clear()
    (cache / 'B6/percept__s2__r20.npz').unlink()
    assert V.entry_parts(FakeCtx(150), 'sample', 1)['saved']['unchecked'] == 1
    V._CACHE.clear()
    path = cache / 'B1/percept__s0__r20.npz'
    with np.load(path) as z:
        labels = z['labels'].copy()
    np.savez(path, labels=(labels + 1) % 20)                                           # a permuted partition
    saved = V.entry_parts(FakeCtx(150), 'sample', 1)['saved']
    assert len(saved['mismatches']) == 1 and 'B1 s0 r20' in saved['mismatches'][0]
    V._CACHE.clear()


# ---- whole modes on tiny trees (150 rows): partitions, no-outcome paths, smoke ---------------------------------

def fake_entries(ctx, scale, workers):
    rng = np.random.default_rng(11)
    parts = {'B1': {s: fake_parts(rng, 150, ctx.labels['emotion'], 0.6) for s in (0, 1)}, 'B2': {}}
    return dict(parts=parts, eligible={'B1': [0, 1], 'B2': []}, dropped={}, notes=[],
                saved=dict(checked=4, unchecked=0, mismatches=[]))


def fake_t(ctx, aspect):
    return fake_parts(np.random.default_rng(12), 150, ctx.labels[aspect], 0.5)['r20'], True


@pytest.fixture
def small(tmp_path, monkeypatch):
    monkeypatch.setattr(V, 'N_HELD', 150)
    monkeypatch.setattr(V, 'WIDTH', 8)
    monkeypatch.setattr(V, 'HELD_SHA256', 'h')
    monkeypatch.setattr(rs_paths, 'RG_FULL', tmp_path / 'rg_full')
    monkeypatch.setattr(rs_paths, 'SMOKE_ROOT', tmp_path)
    monkeypatch.setattr(V, 'Ctx', lambda: FakeCtx(150))
    monkeypatch.setattr(V, 'entry_parts', fake_entries)
    monkeypatch.setattr(V, 't_partitions', fake_t)
    V._CACHE.clear()
    yield tmp_path / 'outputs'
    V._CACHE.clear()


def write_cache(folder, seed0, aspects=('emotion',)):
    """The scorer's partition cache for a set written by write_set(seed0): {aspect}__s{seed}__r20.npz, key labels."""
    folder.mkdir(parents=True, exist_ok=True)
    for a in aspects:
        for s in V.SEEDS:
            x = unit_rows(150, 8, seed0 + 10 * V.ASPECTS.index(a) + s)
            np.savez(folder / f'{a}__s{s}__r20.npz', labels=V.r20_partition(x).astype(np.int32))


def stage0_tree(out, skip=()):
    write_set(out / 'stage0/sample/held/G', 150, 1, arm='G', skip=skip)
    write_set(rs_paths.RG_FULL / 'held' / rs_paths.G_ARM, 150, 2, arm=rs_paths.G_ARM)
    for scale, seed0 in (('sample', 1), ('full', 2)):
        write_cache(out / 'stage0/partitions' / scale / 'G', seed0)


def write_verdict(path, verdict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(verdict))


def stage0_verdict(out):
    verdict = verdict_from(V.stage0(FakeCtx(150), out, 6, 1))
    verdict.update(smoke=True, n_resamples=6)
    write_verdict(out / 'stage0/verdict.json', verdict)
    return verdict


def run_stage0(out):
    return V.main(['stage0', '--outputs', str(out), '--workers', '1'])


def report(out, mode):
    return json.loads((out / mode / f'rederive_{mode}.json').read_text())


def test_stage0_whole_run_agrees_and_the_report_has_smoke_and_the_partition_files(small):
    stage0_tree(small)
    stage0_verdict(small)
    assert run_stage0(small) == 0
    doc = report(small, 'stage0')
    assert doc['pass'] is True and doc['smoke'] is True
    assert any(k.endswith('partitions/sample/G/emotion__s0__r20.npz') for k in doc['inputs_sha256'])
    assert {'sample.partitions.mismatches', 'full.partitions.unchecked'} <= {i['name'] for i in doc['items']}


def swap_clusters(path):
    with np.load(path) as z:
        labels = z['labels'].copy()
    first, second = labels == 0, labels == 1
    labels[first], labels[second] = 1, 0
    np.savez(path, labels=labels)


def test_stage0_permuted_partition_fails_the_run(small):
    stage0_tree(small)
    stage0_verdict(small)
    swap_clusters(small / 'stage0/partitions/sample/G/emotion__s0__r20.npz')
    assert run_stage0(small) == 1
    bad = [i['name'] for i in report(small, 'stage0')['items'] if not i['ok']]
    assert bad == ['sample.partitions.mismatches']


def test_stage0_missing_partition_file_fails_the_run(small):
    stage0_tree(small)
    stage0_verdict(small)
    (small / 'stage0/partitions/full/G/emotion__s3__r20.npz').unlink()
    assert run_stage0(small) == 1
    assert [i['name'] for i in report(small, 'stage0')['items'] if not i['ok']] == ['full.partitions.unchecked']


@pytest.mark.parametrize('aspect, seed, how', [('emotion', 3, 'record'), ('style', 2, 'record'), ('genre', 0, 'missing'),
                                              ('style', 4, 'record')])
def test_stage0_failed_seed_leaves_the_scale_with_no_outcome_in_both_runs(small, aspect, seed, how):
    stage0_tree(small, skip=((aspect, seed),))
    folder = small / 'stage0/sample/held/G'
    if how == 'record':                                       # the scorer's failed__ record instead of the output files
        (folder / f'failed__{aspect}__s{seed}.json').write_text(json.dumps(dict(problems=['shape'])))
    verdict = stage0_verdict(small)                           # my own record is the scorer's here: both have no outcome
    assert verdict['outcomes']['sample'] is None and verdict['E0'] is False and verdict['outcomes']['full']
    assert verdict['blocks']['sample']['v_failure']
    assert run_stage0(small) == 0                             # no traceback: a rederive file, pass
    doc = report(small, 'stage0')
    assert doc['pass'] is True and doc['mine']['blocks']['sample']['outcome_id'] is None
    assert any('sample' in n and 'no outcome' in n for n in doc['notes'])
    claimed = json.loads((small / 'stage0/verdict.json').read_text())
    claimed['blocks']['sample'] = dict(candidate='G', scale='sample', outcome_id='G-ES-GO')    # the scorer found an outcome
    claimed['outcomes']['sample'] = 'G-ES-GO'
    write_verdict(small / 'stage0/verdict.json', claimed)
    assert run_stage0(small) == 1


def test_stage0_scorer_without_outcome_while_mine_has_one_fails(small):
    stage0_tree(small)
    verdict = stage0_verdict(small)
    verdict['blocks']['full'] = dict(candidate='G', scale='full', outcome_id=None, v_failure='V0c: emotion s3 failed twice')
    verdict['outcomes']['full'] = None
    verdict['E0'] = False
    write_verdict(small / 'stage0/verdict.json', verdict)
    assert run_stage0(small) == 1


def stage4_tree(out, name='c', skip=()):
    write_set(rs_paths.full_root(out) / 'held' / name, 150, 3, arm=name, skip=skip)
    write_cache(out / 'stage4/partitions/full' / name, 3, V.ASPECTS)


def stage4_verdict(out, name='c'):
    mine = V.stage4(FakeCtx(150), out, name, 6, 1)
    verdict = verdict_stage4(mine)
    m = mine['emotion_margin']
    verdict['emotion_margin']['max_entry'] = max(m['A_e'], key=m['A_e'].get)
    verdict.update(smoke=True, n_resamples=6, candidate=name)
    write_verdict(out / 'stage4/verdict.json', verdict)
    return verdict


def run_stage4(out, name='c'):
    return V.main(['stage4', '--candidate', name, '--outputs', str(out), '--workers', '1'])


def test_stage4_whole_run_agrees_and_compares_every_aspects_partitions(small):
    stage4_tree(small)
    stage4_verdict(small)
    assert run_stage4(small) == 0
    doc = report(small, 'stage4')
    names = {i['name'] for i in doc['items']}
    assert doc['pass'] and doc['smoke'] and {f'{a}.partitions.mismatches' for a in V.ASPECTS} <= names
    assert {'emotion_margin.partitions.unchecked', 'genre.partitions.unchecked'} <= names
    assert any(k.endswith('stage4/partitions/full/c/genre__s4__r20.npz') for k in doc['inputs_sha256'])


@pytest.mark.parametrize('aspect, flagged', [('emotion', ['emotion.partitions.mismatches', 'emotion_margin.partitions.mismatches']),
                                             ('style', ['style.partitions.mismatches']), ('genre', ['genre.partitions.mismatches'])])
def test_stage4_permuted_candidate_partition_fails_the_run(small, aspect, flagged):
    stage4_tree(small)
    stage4_verdict(small)
    swap_clusters(small / f'stage4/partitions/full/c/{aspect}__s1__r20.npz')
    assert run_stage4(small) == 1
    assert [i['name'] for i in report(small, 'stage4')['items'] if not i['ok']] == flagged


@pytest.mark.parametrize('aspect, flagged', [('emotion', ['emotion.partitions.unchecked', 'emotion_margin.partitions.unchecked']),
                                             ('style', ['style.partitions.unchecked'])])
def test_stage4_missing_partition_file_fails_the_run(small, aspect, flagged):
    stage4_tree(small)
    stage4_verdict(small)
    (small / f'stage4/partitions/full/c/{aspect}__s2__r20.npz').unlink()
    assert run_stage4(small) == 1
    assert [i['name'] for i in report(small, 'stage4')['items'] if not i['ok']] == flagged


def inject_failure(out, kind, name='c'):
    held = rs_paths.full_root(out) / 'held' / name
    if kind == 'train':
        folder = rs_paths.full_root(out) / 'train' / name
        folder.mkdir(parents=True)
        (folder / 'failed__s2.json').write_text(json.dumps(dict(cause='non-finite loss')))
    elif kind == 'v4c':
        (held / 'v4c.json').write_text(json.dumps(dict(**{'pass': False}, problems=['x'])))
    elif kind == 'held':
        for suffix in ('.npz', '.json'):
            (held / f'style__s1{suffix}').unlink()
        (held / 'failed__style__s1.json').write_text(json.dumps(dict(problems=['shape'])))
    else:
        for suffix in ('.npz', '.json'):                      # an output missing without any record
            (held / f'genre__s4{suffix}').unlink()


@pytest.mark.parametrize('kind', ['train', 'v4c', 'held', 'missing'])
def test_stage4_any_failure_leaves_every_aspect_and_the_margin_without_outcome(small, kind):
    stage4_tree(small)
    healthy = stage4_verdict(small)
    inject_failure(small, kind)
    V._CACHE.clear()                                           # the healthy run's reads must not stand in for the new files
    mine = V.stage4(FakeCtx(150), small, 'c', 6, 1)
    assert mine['v_failure'] and all(mine['aspects'][a]['outcome_id'] is None for a in V.ASPECTS)
    assert mine['emotion_margin']['outcome_id'] is None
    assert run_stage4(small) == 1                              # the verdict still states outcomes: a disagreement
    scorer = failed_verdict_stage4(f'{kind}: the scorer cause')
    scorer['candidate'] = 'c'
    write_verdict(small / 'stage4/verdict.json', scorer)
    assert run_stage4(small) == 0                              # no traceback; the rederive file says pass
    doc = report(small, 'stage4')
    assert doc['pass'] is True and doc['mine']['v_failure'] and any('the scorer cause' in n for n in doc['notes'])
    assert healthy['aspects']['emotion']['outcome']


def test_stage4_scorer_failure_while_my_outputs_are_fine_fails(small):
    stage4_tree(small)
    stage4_verdict(small)
    scorer = failed_verdict_stage4('V4d: seed 2 failed twice')
    scorer.update(candidate='c', smoke=True, n_resamples=6)       # a smoke: my outputs are re-derived with 6 draws, not 2,000
    write_verdict(small / 'stage4/verdict.json', scorer)
    assert run_stage4(small) == 1


# ---- smoke verdicts ---------------------------------------------------------------------------------------------

def test_smoke_verdict_outside_scratch_is_refused_and_leaves_no_report(small, monkeypatch):
    stage4_tree(small)
    stage4_verdict(small)
    monkeypatch.setattr(rs_paths, 'SMOKE_ROOT', small.parent / 'elsewhere')       # the outputs are not under scratch now
    with pytest.raises(AssertionError, match='smoke'):
        run_stage4(small)
    assert not (small / 'stage4/rederive_stage4.json').exists()


def test_report_says_when_the_verdict_is_not_a_smoke(small):
    scorer = failed_verdict_stage4()
    scorer['candidate'] = 'c'
    write_verdict(small / 'stage4/verdict.json', scorer)
    stage4_tree(small)
    inject_failure(small, 'train')
    assert run_stage4(small) == 0 and report(small, 'stage4')['smoke'] is False


# ---- integration: K4's sample-scale block from the frozen files (point values only) ---------------------------

def test_k4_sample_point_values_equal_the_frozen_block():
    res = V.stage0_scale(V.Ctx(), 'sample', V.RB_K4_DIRS['sample'], cand_name='K4', draws=0, workers=8)
    frozen = json.loads((rs_paths.RB_PART2 / 'verdict_stage3.json').read_text())['blocks']['K4']
    assert res['d'] == pytest.approx(frozen['d'], abs=1e-6)
    assert res['A_cand'] == pytest.approx(frozen['A_cand'], abs=1e-6)
    for k, v in frozen['A_e'].items():
        assert res['A_e'][k] == pytest.approx(v, abs=1e-6), k
    assert res['eligible_seeds'] == frozen['eligibility']['eligible_seeds']
