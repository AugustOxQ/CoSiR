"""Tests of rb_vp1 (VP1: independent re-derivation of part 1). Expected values are worked by hand from rule §3.1."""
import copy
import hashlib
import json
import os

import numpy as np
import pytest

import rb_vp1

ROWS = [  # annotation order matters: style is the first caption's
    dict(painting='p1', emotion='fear', art_style='Baroque'),
    dict(painting='p1', emotion='awe', art_style='Baroque'),
    dict(painting='p1', emotion='awe', art_style='Baroque'),
    dict(painting='p2', emotion='fear', art_style='Cubism'),
    dict(painting='p2', emotion='awe', art_style='Cubism'),
    dict(painting='p3', emotion='something else', art_style='Baroque'),
    dict(painting='p3', emotion='fear', art_style='Baroque'),
    dict(painting='p4', emotion='something else', art_style='Cubism'),
    dict(painting='p4', emotion='something else', art_style='Cubism'),
    dict(painting='p4', emotion='amusement', art_style='Cubism'),
    dict(painting='p5', emotion='amusement', art_style='Baroque'),   # not in the asked names
]


def test_labels_by_the_rule_with_hand_checked_codes():
    # emotion codes: amusement 0, awe 1, fear 2, something else 3 (names sorted); styles: Baroque 0, Cubism 1
    out = rb_vp1.labels_from_annotations(['p4', 'p1', 'p3', 'p2'], ROWS, genre_ids=np.array([5, -1, 0, 9]))
    assert out['emotion'].tolist() == [3, 1, 2, 1]      # p4 plurality; p1 plurality awe; p3 tie fear<something else; p2 tie awe<fear
    assert out['style'].tolist() == [1, 0, 0, 1]
    assert out['genre'].tolist() == [5, -1, 0, 9]       # the CSV id; -1 stays masked
    assert all(v.dtype == np.int64 for v in out.values())


def test_a_name_without_annotations_is_an_error():
    with pytest.raises(AssertionError):
        rb_vp1.labels_from_annotations(['p1', 'nope'], ROWS, genre_ids=np.array([0, 0]))


def test_style_is_the_first_captions():
    rows = [dict(painting='q', emotion='awe', art_style='B'), dict(painting='q', emotion='awe', art_style='A')]
    out = rb_vp1.labels_from_annotations(['q'], rows + [dict(painting='z', emotion='awe', art_style='A')],
                                         genre_ids=np.array([1]))
    assert out['style'].tolist() == [1]                 # first caption B; codes A 0, B 1


def circle(n_points=20, reps=10, dim=2):
    angle = 2 * np.pi * np.arange(n_points) / n_points
    pts = np.zeros((n_points, dim), np.float32)
    pts[:, 0], pts[:, 1] = np.cos(angle), np.sin(angle)
    return np.repeat(pts, reps, axis=0), np.repeat(np.arange(n_points), reps)


def test_partitions_find_twenty_separated_groups_and_are_repeatable():
    x, truth = circle()
    parts = rb_vp1.partitions(x.astype(np.float16))
    assert parts.shape == (3, 200) and parts.dtype == np.int64
    for row in parts:   # k-means++ on 20 distinct points picks each once: the partition is the grouping
        assert len(np.unique(row)) == 20
        assert all(len(np.unique(row[truth == j])) == 1 for j in range(20))
    assert np.array_equal(parts, rb_vp1.partitions(x.astype(np.float16)))


def test_partitions_refuse_rows_that_are_not_unit_norm():
    x, _ = circle()
    with pytest.raises(AssertionError):
        rb_vp1.partitions((x * 1.01).astype(np.float16))


def test_mean_ami_is_the_mean_over_the_three_partitions():
    labels = np.array([0, 0, 1, 1, 2, 2, 0, 1])
    parts = np.stack([labels, labels + 5, np.zeros(8, int)])   # AMI 1, 1 (renamed), 0 (one cluster)
    assert rb_vp1.mean_ami(labels, parts, np.arange(8)) == pytest.approx(2 / 3, abs=1e-12)
    idx = np.array([0, 2, 4, 4, 0])                              # repeats allowed; labels 0,1,2,2,0
    assert rb_vp1.mean_ami(labels, parts, idx) == pytest.approx(2 / 3, abs=1e-12)


R = rb_vp1.SEED_RANGES


def eff(e_em, e_st, e_ge, lo=(0.001, 0.001, 0.001)):
    """One candidate's effects; `lo` is the lower bound per aspect (hi is not used by the decision)."""
    return {a: dict(e=e, lo=l, hi=e + 1) for a, e, l in zip(rb_vp1.ASPECTS, (e_em, e_st, e_ge), lo)}


def all_flat():
    return {c: eff(0.0, 0.0, 0.0) for c in rb_vp1.CANDIDATES}


def test_gain_needs_a_positive_lower_bound_and_e_at_least_the_seed_range():
    assert rb_vp1.gain(e=R['style'], lo=1e-9, aspect='style') is True          # e = r counts
    assert rb_vp1.gain(e=R['style'] - 1e-9, lo=1e-9, aspect='style') is False
    assert rb_vp1.gain(e=R['style'] * 5, lo=0.0, aspect='style') is False       # lo must be > 0


def test_guard_is_the_point_estimate_at_minus_r_or_better_on_all_aspects():
    e = eff(0.0, -R['style'], 0.0)
    assert rb_vp1.evaluate(e)['guard'] is True                                    # exactly -r is allowed
    assert rb_vp1.evaluate(eff(0.0, -R['style'] - 1e-9, 0.0))['guard'] is False
    out = rb_vp1.evaluate(eff(5 * R['emotion'], -1.01 * R['style'], 0.0))
    assert out['guard'] is False and out['pass'] is False                        # a gain does not rescue the guard


def test_mean_gain_averages_all_three_aspects_in_seed_ranges_losses_included():
    out = rb_vp1.evaluate(eff(R['emotion'], -0.5 * R['style'], R['genre']))     # (1 - 0.5 + 1) / 3
    assert out['g'] == pytest.approx(0.5) and out['pass'] is True and out['flag_g_le_0'] is False
    out = rb_vp1.evaluate(eff(R['emotion'], -R['style'], -R['genre']))          # (1 - 1 - 1) / 3: passes, flagged
    assert out['g'] == pytest.approx(-1 / 3) and out['pass'] is True and out['flag_g_le_0'] is True


def test_no_passing_candidate_is_p_keep():
    cands, outcome, recipe = rb_vp1.decide(all_flat())
    assert (outcome, recipe) == ('P-KEEP', 'R0-5') and not any(c['pass'] for c in cands.values())


def test_largest_rounded_g_wins_and_the_first_in_order_wins_a_tie():
    effects = all_flat()
    effects['R6-5'] = eff(1.49 * R['emotion'], 0, 0)    # g 0.4967 -> 0.50
    effects['R0-3'] = eff(1.51 * R['emotion'], 0, 0)    # g 0.5033 -> 0.50, first in order
    effects['R7-3'] = eff(1.0 * R['emotion'], 0, 0)     # g 0.3333 -> 0.33
    assert rb_vp1.decide(effects)[1:] == ('P-R0-3', 'R0-3')
    effects['R7-5'] = eff(1.6 * R['emotion'], 0, 0)     # 0.5333 -> 0.53 beats 0.50
    assert rb_vp1.decide(effects)[1:] == ('P-R7-5', 'R7-5')


def test_an_excluded_candidate_never_passes_and_has_empty_fields():
    effects = all_flat()
    effects['R6-3'] = None
    effects['R7-3'] = eff(9 * R['emotion'], 0, 0)
    cands, outcome, recipe = rb_vp1.decide(effects)
    assert cands['R6-3'] == dict(guard=None, **{'pass': False, 'g': None, 'flag_g_le_0': None})
    assert (outcome, recipe) == ('P-R7-3', 'R7-3')
    effects['R7-3'] = None
    assert rb_vp1.decide(effects)[1:] == ('P-KEEP', 'R0-5')


def test_triggers():
    assert rb_vp1.triggers('R0-5') == {'K4': False, 'K5': False}
    assert rb_vp1.triggers('R6-3') == {'K4': True, 'K5': False}
    assert rb_vp1.triggers('R7-5') == {'K4': True, 'K5': True}
    assert rb_vp1.triggers('R7-3') == {'K4': True, 'K5': True}


def toy(n=90, seed=3):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 3, n)
    noise = lambda: np.stack([rng.integers(0, 4, n) for _ in range(3)])
    return labels, noise


def test_effects_of_a_perfect_candidate_against_a_constant_reference_are_exactly_one():
    labels, _ = toy()
    parts = {'R0-5': np.zeros((3, 90), int), 'R6-3': np.stack([labels] * 3), 'R7-3': np.zeros((3, 90), int)}
    out = rb_vp1.score_effects(labels, parts, n_draws=7, workers=1)
    assert out['R0-5'] == dict(A=0.0, e=0.0, lo=0.0, hi=0.0)
    assert out['R6-3'] == dict(A=1.0, e=1.0, lo=1.0, hi=1.0)
    assert out['R7-3'] == dict(A=0.0, e=0.0, lo=0.0, hi=0.0)


def test_interval_uses_default_rng_0_draws_of_the_labelled_positions_and_linear_percentiles():
    labels, noise = toy()
    ref = noise()
    parts = {'R0-5': ref, 'R6-3': np.stack([labels] * 3)}
    out = rb_vp1.score_effects(labels, parts, n_draws=5, workers=1)
    rng = np.random.default_rng(0)
    e_b = []
    for _ in range(5):
        i = rng.integers(0, 90, size=90)
        a_ref = np.mean([rb_vp1.adjusted_mutual_info_score(labels[i], p[i]) for p in ref])
        e_b.append(1.0 - a_ref)                        # the candidate's AMI is 1 on every draw
    lo, hi = np.percentile(e_b, [2.5, 97.5])
    assert out['R6-3']['lo'] == pytest.approx(lo, abs=1e-12) and out['R6-3']['hi'] == pytest.approx(hi, abs=1e-12)
    a_all = np.mean([rb_vp1.adjusted_mutual_info_score(labels, p) for p in ref])
    assert out['R6-3']['e'] == pytest.approx(1.0 - a_all, abs=1e-12) and out['R0-5']['A'] == pytest.approx(a_all)


def test_interval_does_not_depend_on_the_number_of_workers():
    labels, noise = toy()
    parts = {'R0-5': noise(), 'R6-3': noise(), 'R7-3': noise()}
    one = rb_vp1.score_effects(labels, parts, n_draws=9, workers=1)
    two = rb_vp1.score_effects(labels, parts, n_draws=9, workers=2)
    assert one == two


# ---- files: derive, compare, run -----------------------------------------------------------------------------

N = 120
NAMES = sorted(f'paint{i:03d}' for i in range(N))
EMO = ['amusement', 'awe', 'fear', 'sadness']
STY = ['Baroque', 'Cubism', 'Fauvism']


def synth_annotations():
    rng = np.random.default_rng(5)
    rows = []
    for p in NAMES:
        style = STY[rng.integers(0, 3)]
        for _ in range(3):
            rows.append(dict(painting=p, emotion=EMO[rng.integers(0, 4)], art_style=style))
    return rows


def synth_genre(names):
    return np.array([(-1 if i % 7 == 0 else i % 5) for i in range(len(names))], np.int64)


def unit(x):
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float16)


def write_p(tmp_path, names=NAMES):
    d = tmp_path / 'part1'
    d.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256('\n'.join(names).encode()).hexdigest()
    (d / 'P.json').write_text(json.dumps(dict(names=names, sha256=sha, counts=dict(train=90, val=30))))
    return sha


W = 16


def write_exports(tmp_path, labels, signal):
    """Exports whose cluster structure follows the labels with strength signal[recipe] (the rest is noise), each with
    its training record naming P."""
    sha = json.loads((tmp_path / 'part1' / 'P.json').read_text())['sha256']
    rng = np.random.default_rng(11)
    for r, strength in signal.items():
        for a in rb_vp1.ASPECTS:
            lab = labels[a]
            centers = rng.normal(size=(int(lab.max()) + 2, 16))
            x = strength * centers[lab] + rng.normal(size=(N, 16))
            d = tmp_path / 'part1' / 'train' / r
            d.mkdir(parents=True, exist_ok=True)
            np.savez(d / f'{a}__s0__P.npz', img=unit(x), cap=unit(x))
            (d / f'{a}__s0.json').write_text(json.dumps(dict(p_sha256=sha)))


SIGNAL = {'R0-5': 0.5, 'R0-3': 0.5, 'R6-5': 3.0, 'R6-3': 0.5, 'R7-5': 0.5, 'R7-3': 0.5}


def derive_all(tmp_path, excluded=(), signal=SIGNAL):
    sha = write_p(tmp_path)
    labels = rb_vp1.labels_from_annotations(NAMES, synth_annotations(), synth_genre(NAMES))
    write_exports(tmp_path, labels, signal)
    mine = rb_vp1.derive(tmp_path, annotations=synth_annotations(), genre_loader=synth_genre, excluded=excluded,
                         n_draws=8, workers=1, n_rows=N, width=W)
    return sha, labels, mine


def rederive(tmp_path, excluded=()):
    return rb_vp1.derive(tmp_path, annotations=synth_annotations(), genre_loader=synth_genre, excluded=excluded,
                         n_draws=8, workers=1, n_rows=N, width=W)


def scale_export(tmp_path, recipe, aspect, factor=1.1):
    f = tmp_path / 'part1' / 'train' / recipe / f'{aspect}__s0__P.npz'
    x = np.load(f)
    np.savez(f, img=x['img'] * np.float16(factor), cap=x['cap'])


def make_pick(mine, excluded=()):
    """What ticket 03's file looks like (index format), filled from `mine`; the tests then plant differences."""
    recipes = {r: (None if r in excluded else {a: dict(A=v['A'], e=v['e'], lo=v['lo'], hi=v['hi'], gain=v['gain'])
                                              for a, v in mine['effects'][r].items()}) for r in rb_vp1.RECIPES}
    return copy.deepcopy(dict(p_sha256=mine['p_sha256'], seed_ranges=rb_vp1.SEED_RANGES, versions={'sklearn': 'x', 'numpy': 'y'},
                label_counts=mine['label_counts'], label_sha256=mine['label_sha256'], recipes=recipes,
                candidates={c: dict(mine['candidates'][c], vp2_ok=c not in excluded, vp2_problems=[])
                            for c in rb_vp1.CANDIDATES},
                outcome=mine['outcome'], recipe=mine['recipe'], triggers=mine['triggers'], n_draws=mine['n_draws'],
                vp2_failed={c: 'norm error' for c in excluded}))


def write_saved_partitions(tmp_path, mine, excluded=()):
    for r in rb_vp1.RECIPES:
        if r in excluded:
            continue
        d = tmp_path / 'part1' / 'partitions' / r
        d.mkdir(parents=True, exist_ok=True)
        for a in rb_vp1.ASPECTS:
            np.savez(d / f'{a}.npz', labels=mine['parts'][r][a].astype(np.int64))


def test_derive_gives_the_labels_hashes_counts_and_the_decision(tmp_path):
    sha, labels, mine = derive_all(tmp_path)
    assert mine['p_sha256'] == sha
    assert mine['label_sha256'] == {a: hashlib.sha256(labels[a].tobytes()).hexdigest() for a in rb_vp1.ASPECTS}
    assert mine['label_counts'] == {a: int((labels[a] >= 0).sum()) for a in rb_vp1.ASPECTS}
    assert mine['label_counts']['genre'] == N - len(range(0, N, 7))
    assert mine['parts']['R6-5']['emotion'].shape == (3, N)
    assert set(mine['effects']) == set(rb_vp1.RECIPES)
    assert mine['effects']['R0-5']['style'] == dict(A=mine['effects']['R0-5']['style']['A'], e=0.0, lo=0.0, hi=0.0,
                                                    gain=False)
    # R6-5 carries a strong label signal in its exports, the others a weak one: the pick follows
    assert mine['effects']['R6-5']['emotion']['e'] > 0.05 > abs(mine['effects']['R7-3']['emotion']['e'])
    assert mine['outcome'] == 'P-R6-5' and mine['recipe'] == 'R6-5'
    assert mine['triggers'] == {'K4': True, 'K5': False}


def test_genre_is_scored_only_on_labelled_paintings(tmp_path):
    _, labels, mine = derive_all(tmp_path)
    keep = labels['genre'] >= 0
    parts = {r: mine['parts'][r]['genre'][:, keep] for r in rb_vp1.RECIPES}
    again = rb_vp1.score_effects(labels['genre'][keep], parts, n_draws=8, workers=1)
    assert again['R6-5']['e'] == mine['effects']['R6-5']['genre']['e']
    assert again['R6-5']['lo'] == mine['effects']['R6-5']['genre']['lo']


def agree_all(tmp_path, mine, pick=None, excluded=()):
    write_saved_partitions(tmp_path, mine, excluded)
    (tmp_path / 'part1' / 'pick.json').write_text(json.dumps(pick or make_pick(mine, excluded)))
    return rb_vp1.compare(tmp_path, mine)


def test_identical_files_agree(tmp_path):
    _, _, mine = derive_all(tmp_path)
    res = agree_all(tmp_path, mine)
    assert res['agree'] is True and res['disagreements'] == []
    assert res['partitions']['R6-5']['emotion'] == dict(equal=True, n_diff=0)
    assert res['outcome'] == dict(mine='P-R6-5', pick='P-R6-5', agree=True)


@pytest.mark.parametrize('what', ['partition', 'e', 'lo', 'hi', 'bound_just_over_tol', 'outcome', 'label_hash',
                                  'trigger', 'p_sha', 'g', 'gain_flag', 'n_draws'])
def test_a_planted_difference_is_reported(tmp_path, what):
    _, _, mine = derive_all(tmp_path)
    pick = make_pick(mine)
    if what == 'partition':
        write_saved_partitions(tmp_path, mine)
        f = tmp_path / 'part1' / 'partitions' / 'R7-3' / 'style.npz'
        lab = np.load(f)['labels'].copy()
        lab[0, 5] = (lab[0, 5] + 1) % 20
        np.savez(f, labels=lab)
    elif what == 'e':
        pick['recipes']['R6-3']['genre']['e'] += 1e-4
    elif what == 'lo':
        pick['recipes']['R6-5']['emotion']['lo'] -= 1e-3
    elif what == 'hi':
        pick['recipes']['R7-5']['style']['hi'] += 1e-3
    elif what == 'bound_just_over_tol':
        pick['recipes']['R6-5']['emotion']['hi'] += 2e-6
    elif what == 'outcome':
        pick['outcome'], pick['recipe'] = 'P-KEEP', 'R0-5'
    elif what == 'label_hash':
        pick['label_sha256']['genre'] = '0' * 64
    elif what == 'trigger':
        pick['triggers']['K5'] = True
    elif what == 'p_sha':
        pick['p_sha256'] = '1' * 64
    elif what == 'g':
        pick['candidates']['R0-3']['g'] += 0.01
    elif what == 'n_draws':
        pick['n_draws'] = 20
    elif what == 'gain_flag':
        pick['recipes']['R6-5']['emotion']['gain'] = not pick['recipes']['R6-5']['emotion']['gain']
    if what != 'partition':
        write_saved_partitions(tmp_path, mine)
    (tmp_path / 'part1' / 'pick.json').write_text(json.dumps(pick))
    res = rb_vp1.compare(tmp_path, mine)
    assert res['agree'] is False and len(res['disagreements']) >= 1


def test_a_difference_inside_the_tolerance_is_not_a_disagreement(tmp_path):
    _, _, mine = derive_all(tmp_path)
    pick = make_pick(mine)
    pick['recipes']['R6-5']['emotion']['e'] += 5e-7
    assert agree_all(tmp_path, mine, pick)['agree'] is True


def test_a_candidate_excluded_by_vp2_is_excluded_not_a_disagreement(tmp_path):
    derive_all(tmp_path, excluded=('R6-5',))
    scale_export(tmp_path, 'R6-5', 'genre')           # its exports really fail VP2
    mine = rederive(tmp_path, excluded=('R6-5',))
    chk = mine['excluded_exports']['R6-5']
    assert chk['present'] is True and len(chk['problems']) == 1 and chk['problems'][0].startswith('genre: norm')
    assert mine['effects']['R6-5'] is None and mine['candidates']['R6-5']['pass'] is False
    assert mine['outcome'] != 'P-R6-5'
    res = agree_all(tmp_path, mine, excluded=('R6-5',))
    assert res['agree'] is True and res['excluded'] == ['R6-5']
    # everything else is still compared: plant a difference in another candidate
    pick = make_pick(mine, ('R6-5',))
    pick['recipes']['R7-3']['emotion']['e'] += 1e-3
    assert agree_all(tmp_path, mine, pick, ('R6-5',))['agree'] is False


def test_an_excluded_candidate_whose_exports_pass_is_a_disagreement(tmp_path):
    _, _, mine = derive_all(tmp_path, excluded=('R6-5',))
    assert mine['excluded_exports']['R6-5'] == dict(present=True, problems=[])
    res = agree_all(tmp_path, mine, excluded=('R6-5',))
    assert res['agree'] is False and 'pick excluded R6-5, its exports pass VP2' in res['disagreements']


def test_an_excluded_candidate_without_its_exports_is_not_a_disagreement(tmp_path):
    derive_all(tmp_path, excluded=('R6-5',))
    (tmp_path / 'part1' / 'train' / 'R6-5' / 'style__s0__P.npz').unlink()   # training stopped before this export
    mine = rederive(tmp_path, excluded=('R6-5',))
    assert mine['excluded_exports']['R6-5'] == dict(present=False, problems=[])
    assert agree_all(tmp_path, mine, excluded=('R6-5',))['agree'] is True


def test_an_export_whose_record_names_another_p_is_refused(tmp_path):
    derive_all(tmp_path)
    (tmp_path / 'part1' / 'train' / 'R7-5' / 'emotion__s0.json').write_text(json.dumps(dict(p_sha256='3' * 64)))
    with pytest.raises(AssertionError, match='another P'):
        rederive(tmp_path)


@pytest.mark.parametrize('bad', ['float32', 'width'])
def test_an_export_of_the_wrong_dtype_or_width_fails_vp2(bad):
    x = unit(np.random.default_rng(0).normal(size=(N, W)))
    x = x.astype(np.float32) if bad == 'float32' else x[:, :-1]
    assert rb_vp1._check_export(x, N, W) is not None
    assert rb_vp1._check_export(unit(np.random.default_rng(0).normal(size=(N, W))), N, W) is None


def test_a_candidate_the_pick_scored_but_mine_found_bad_is_a_disagreement(tmp_path):
    _, _, mine = derive_all(tmp_path)
    pick = make_pick(mine)
    x = np.load(tmp_path / 'part1' / 'train' / 'R6-3' / 'style__s0__P.npz')
    np.savez(tmp_path / 'part1' / 'train' / 'R6-3' / 'style__s0__P.npz', img=x['img'] * np.float16(1.1), cap=x['cap'])
    mine2 = rederive(tmp_path)
    assert mine2['vp2_problems'] and 'R6-3' in mine2['vp2_problems']
    write_saved_partitions(tmp_path, mine)
    (tmp_path / 'part1' / 'pick.json').write_text(json.dumps(pick))
    res = rb_vp1.compare(tmp_path, mine2)
    assert res['agree'] is False and any('R6-3' in d for d in res['disagreements'])


def test_run_writes_the_json_and_md_and_refuses_a_changed_p_hash(tmp_path):
    _, _, mine = derive_all(tmp_path)
    agree_all(tmp_path, mine)
    res = rb_vp1.run(tmp_path, annotations=synth_annotations(), genre_loader=synth_genre, n_draws=8, workers=1,
                     n_rows=N, width=W)
    out = json.loads((tmp_path / 'part1' / 'rederive_vp1.json').read_text())
    assert res['agree'] is True and out['agree'] is True and out['n_draws'] == 8
    assert (tmp_path / 'part1' / 'rederive_vp1.md').read_text().startswith('# VP1')
    p = json.loads((tmp_path / 'part1' / 'P.json').read_text())
    p['sha256'] = '2' * 64
    (tmp_path / 'part1' / 'P.json').write_text(json.dumps(p))
    with pytest.raises(AssertionError):
        rb_vp1.run(tmp_path, annotations=synth_annotations(), genre_loader=synth_genre, n_draws=8, workers=1,
                   n_rows=N, width=W)


def test_smoke_refuses_an_outputs_path_outside_the_scratch_area():
    with pytest.raises(SystemExit):
        rb_vp1.main(['--smoke', '--outputs', '/project/CoSiR/nowhere'])


# ---- real data (slow: RB_SLOW=1) -------------------------------------------------------------------------------

slow = pytest.mark.skipif(not os.environ.get('RB_SLOW'), reason='reads the raw annotations: set RB_SLOW=1')


@slow
def test_real_annotations_give_the_rules_codes_and_p_json_counts_when_it_exists():
    from pathlib import Path
    annotations = rb_vp1._default_annotations()
    emotions = sorted({r['emotion'] for r in annotations})
    assert len(emotions) == 9 and emotions[8] == 'something else'              # rule §3.3: "something else" = 8
    names = sorted({r['painting'] for r in annotations})
    genre = rb_vp1._default_genre_loader(names)
    labels = rb_vp1.labels_from_annotations(names, annotations, genre)
    assert (labels['emotion'] >= 0).all() and (labels['style'] >= 0).all() and labels['emotion'].max() == 8
    assert labels['style'].max() == len({r['art_style'] for r in annotations}) - 1 == 26
    assert set(np.unique(labels['genre'])) <= set(range(-1, 10)) and (labels['genre'] >= 0).any()
    print({a: int((v >= 0).sum()) for a, v in labels.items()}, 'of', len(names))
    out = Path(os.environ.get('RB_OUTPUTS') or '/project/CoSiR/src/test/20261025_recipe_backbones/outputs')
    f = out / 'part1' / 'P.json'
    if f.exists():
        p = json.loads(f.read_text())
        assert p['counts'] == {'train': 23080, 'val': 6152} and len(p['names']) == 29232
        assert rb_vp1.sha256_names(p['names']) == p['sha256'] and set(p['names']) <= set(names)
        sub = rb_vp1.labels_from_annotations(p['names'], annotations, rb_vp1._default_genre_loader(p['names']))
        assert all(v.shape == (29232,) for v in sub.values())
