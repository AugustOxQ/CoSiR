"""Part 1 pick (ticket 03, rule §3.1 and §4.1): gain, guard, pass, g, ties, P-KEEP; bootstrap; files; hand-built cases."""
import json

import numpy as np
import pytest

import rb_paths  # noqa: F401  (sys.path)
import rb_pick
import rb_rows

R = rb_pick.SEED_RANGES


def eff(e_emotion, e_style, e_genre, lo=(-1.0, -1.0, -1.0)):
    """Effects of one recipe; `lo` the interval lower bounds in aspect order (negative: no gain)."""
    return {a: dict(e=e, lo=l, hi=e + 1.0) for a, e, l in zip(rb_pick.ASPECTS, (e_emotion, e_style, e_genre), lo)}


def test_seed_ranges_are_the_users_values_exactly():
    assert R == {'emotion': 0.0046, 'style': 0.0082, 'genre': 0.0128}
    assert rb_pick.ASPECTS == ('emotion', 'style', 'genre')


def test_gain_needs_a_positive_lower_bound_and_a_point_estimate_at_the_seed_range():
    r = R['style']
    assert rb_pick.gain(1e-9, r, 'style') is True            # e = r exactly counts ("at least small")
    assert rb_pick.gain(1e-9, np.nextafter(r, 0), 'style') is False
    assert rb_pick.gain(0.0, 10 * r, 'style') is False        # lower bound must be > 0
    assert rb_pick.gain(-0.001, 10 * r, 'style') is False


def test_guard_allows_exactly_minus_the_seed_range_on_every_aspect():
    ok = {a: -R[a] for a in R}
    assert rb_pick.guard(ok) is True
    bad = dict(ok, genre=float(np.nextafter(-R['genre'], -1)))
    assert rb_pick.guard(bad) is False


def test_mean_gain_counts_losses_in_seed_ranges():
    g = rb_pick.mean_gain({'emotion': 2 * R['emotion'], 'style': -R['style'], 'genre': 0.5 * R['genre']})
    assert g == pytest.approx((2 - 1 + 0.5) / 3)


def test_candidate_passes_with_one_gain_and_the_guard_only():
    one = eff(R['emotion'], 0.0, 0.0, lo=(0.001, -1, -1))
    assert rb_pick.evaluate(one)['pass'] is True
    no_gain = eff(R['emotion'], 0.0, 0.0, lo=(-0.001, -1, -1))              # point high enough, interval spans 0
    assert rb_pick.evaluate(no_gain)['pass'] is False
    small = eff(0.9 * R['emotion'], 0.0, 0.0, lo=(0.001, -1, -1))           # interval above 0, effect below range
    assert rb_pick.evaluate(small)['pass'] is False
    guard_fails = eff(5 * R['emotion'], -1.01 * R['style'], 0.0, lo=(0.001, -1, -1))
    out = rb_pick.evaluate(guard_fails)
    assert out['guard'] is False and out['pass'] is False
    assert out['g'] == pytest.approx((5 - 1.01 + 0) / 3)


def test_a_passing_candidate_with_g_at_most_zero_still_passes_and_is_flagged():
    e = eff(R['emotion'], -R['style'], -R['genre'], lo=(0.001, -1, -1))
    out = rb_pick.evaluate(e)
    assert out['pass'] is True and out['g'] == pytest.approx(-1 / 3) and out['flag_g_le_0'] is True
    assert rb_pick.evaluate(eff(2 * R['emotion'], 0.0, 0.0, lo=(0.001, -1, -1)))['flag_g_le_0'] is False


def test_pick_is_the_largest_two_decimal_g_with_python_round_and_ties_in_rule_order():
    assert rb_pick.pick({}) is None
    assert rb_pick.pick({'R6-3': 0.5}) == 'R6-3'
    # 1.236 rounds to 1.24 and beats 1.234 (1.23) although it is later in the order
    assert rb_pick.pick({'R0-3': 1.234, 'R7-3': 1.236}) == 'R7-3'
    # equal at two decimals: the first in R0-3, R6-5, R6-3, R7-5, R7-3 wins, even with the smaller raw g
    assert rb_pick.pick({'R7-5': 1.231, 'R6-5': 1.234, 'R7-3': 1.2349}) == 'R6-5'
    assert rb_pick.pick({'R7-3': 1.234, 'R0-3': 1.2301}) == 'R0-3'
    # Python's round: 0.125 -> 0.12 (a half-up rule would give 0.13 and choose R6-5)
    assert rb_pick.pick({'R0-3': 0.12, 'R6-5': 0.125}) == 'R0-3'
    # negative g still competes
    assert rb_pick.pick({'R6-5': -0.4, 'R7-3': -0.1}) == 'R7-3'


def test_tie_order_is_the_rows_module_order_without_the_reference():
    assert rb_pick.CANDIDATES == rb_rows.TIE_ORDER == ('R0-3', 'R6-5', 'R6-3', 'R7-5', 'R7-3')
    assert rb_pick.REFERENCE == 'R0-5'


def test_decide_gives_p_keep_when_nothing_passes_and_p_recipe_otherwise():
    nothing = {c: eff(0.0, 0.0, 0.0) for c in rb_pick.CANDIDATES}
    cands, outcome, recipe = rb_pick.decide(nothing)
    assert outcome == 'P-KEEP' and recipe == 'R0-5'
    assert all(not v['pass'] and v['vp2_ok'] for v in cands.values())
    wins = dict(nothing)
    wins['R6-3'] = eff(2 * R['emotion'], 0.0, 0.0, lo=(0.001, -1, -1))
    wins['R7-5'] = eff(3 * R['emotion'], 0.0, 0.0, lo=(0.001, -1, -1))
    cands, outcome, recipe = rb_pick.decide(wins)
    assert (outcome, recipe) == ('P-R7-5', 'R7-5')
    assert cands['R6-3']['pass'] and cands['R7-5']['pass'] and not cands['R0-3']['pass']


def test_decide_never_lets_a_candidate_that_failed_vp2_pass():
    effects = {c: eff(0.0, 0.0, 0.0) for c in rb_pick.CANDIDATES}
    effects['R7-3'] = eff(9 * R['emotion'], 0.0, 0.0, lo=(0.001, -1, -1))
    effects['R6-5'] = None                                                  # VP2 failed: no scores
    cands, outcome, _ = rb_pick.decide(effects, vp2_problems={'R6-5': ['emotion: nonfinite rows']})
    assert outcome == 'P-R7-3'
    assert cands['R6-5'] == dict(vp2_ok=False, vp2_problems=['emotion: nonfinite rows'], guard=None, **{
        'pass': False, 'g': None, 'flag_g_le_0': None})
    effects['R7-3'] = None
    cands, outcome, recipe = rb_pick.decide(effects, vp2_problems={'R6-5': ['x'], 'R7-3': ['y']})
    assert (outcome, recipe) == ('P-KEEP', 'R0-5')


# ------------------------------------------------------------------ the bootstrap
from sklearn.metrics import adjusted_mutual_info_score as _ami   # noqa: E402


def planted(n=300, noise=0.3, seed=3):
    """Labels with 4 classes (some masked) and three recipes' partitions: R0-5 noisy, R6-3 better, R7-3 the same as R0-5."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 4, n)
    labels = np.where(rng.random(n) < 0.1, -1, y)

    def part(flip):
        p = np.stack([np.where(rng.random(n) < flip, rng.integers(0, 4, n), y) for _ in range(3)])
        return p.astype(np.int64)
    base = part(0.6)
    return labels, {'R0-5': base, 'R6-3': part(0.1), 'R7-3': base.copy()}


def test_bootstrap_scores_follow_the_rule_formula_and_parallel_equals_serial():
    labels, parts = planted()
    labelled = np.flatnonzero(labels >= 0)
    n_draws = 6
    serial = rb_pick.bootstrap_scores(labels, parts, n_draws, workers=1)
    parallel = rb_pick.bootstrap_scores(labels, parts, n_draws, workers=4)
    assert set(serial) == set(parts)
    for name in parts:
        np.testing.assert_array_equal(serial[name], parallel[name])      # identical, not just close
        assert serial[name].shape == (n_draws,)
    rng = np.random.default_rng(0)                                         # the rule's formula, written out
    for b in range(n_draws):
        idx = labelled[rng.integers(0, len(labelled), size=len(labelled))]
        for name, p in parts.items():
            expect = np.mean([_ami(labels[idx], p[j][idx]) for j in range(3)])
            assert serial[name][b] == pytest.approx(expect, abs=1e-12)


def test_effects_are_point_estimates_on_all_labelled_rows_and_percentile_intervals():
    labels, parts = planted()
    labelled = np.flatnonzero(labels >= 0)
    out = rb_pick.aspect_effects(labels, parts, n_draws=40, workers=2)
    ref = np.mean([_ami(labels[labelled], parts['R0-5'][j][labelled]) for j in range(3)])
    better = np.mean([_ami(labels[labelled], parts['R6-3'][j][labelled]) for j in range(3)])
    assert out['R0-5']['A'] == pytest.approx(ref, abs=1e-12) and out['R0-5']['e'] == 0.0
    assert (out['R0-5']['lo'], out['R0-5']['hi']) == (0.0, 0.0)
    assert out['R6-3']['e'] == pytest.approx(better - ref, abs=1e-12)
    assert out['R6-3']['e'] > 0.1 and out['R6-3']['lo'] > 0
    assert set(out['R6-3']) == {'A', 'e', 'lo', 'hi'}
    assert out['R7-3']['e'] == 0.0 and out['R7-3']['lo'] == out['R7-3']['hi'] == 0.0           # same partitions: zero in every draw
    boot = rb_pick.bootstrap_scores(labels, parts, 40, workers=1)
    lo, hi = np.percentile(boot['R6-3'] - boot['R0-5'], [2.5, 97.5])
    assert (out['R6-3']['lo'], out['R6-3']['hi']) == (lo, hi)


def test_bootstrap_needs_the_reference_and_three_partitions_each():
    labels, parts = planted()
    with pytest.raises(AssertionError):
        rb_pick.aspect_effects(labels, {k: v for k, v in parts.items() if k != 'R0-5'}, n_draws=2)
    with pytest.raises(AssertionError):
        rb_pick.aspect_effects(labels, dict(parts, **{'R6-3': parts['R6-3'][:2]}), n_draws=2)


# ------------------------------------------------------------------ files: synthetic exports in tmp_path
import hashlib   # noqa: E402
import shutil   # noqa: E402

import rb_ami   # noqa: E402
import rb_data  # noqa: E402

N = 400
CLASSES = {'emotion': 4, 'style': 5, 'genre': 3}


def synthetic_labels(seed=11):
    rng = np.random.default_rng(seed)
    out = {a: rng.integers(0, c, N).astype(np.int64) for a, c in CLASSES.items()}
    out['genre'][rng.random(N) < 0.2] = -1                          # genre has masked paintings
    return out


def sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def write_exports(outputs, labels, signal, p_sha, names=None, bad=None):
    """Exports of every recipe: unit rows whose class signal is `signal[recipe][aspect]` (0: pure noise)."""
    rng = np.random.default_rng(5)
    centres = {a: rng.standard_normal((CLASSES[a], 2048)) for a in CLASSES}
    for recipe in rb_rows.ORDER:
        folder = outputs / 'part1' / 'train' / recipe
        folder.mkdir(parents=True, exist_ok=True)
        for aspect in rb_pick.ASPECTS:
            y = np.where(labels[aspect] >= 0, labels[aspect], 0)
            x = signal[recipe][aspect] * centres[aspect][y] + rng.standard_normal((N, 2048))
            x = (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float16)
            if bad and bad[0] == recipe and bad[1] == aspect:
                x[7, 3] = np.nan
            cap = x[::-1].copy()
            np.savez(folder / f'{aspect}__s0__P.npz', img=x, cap=cap)
            record = dict(recipe=recipe, aspect=aspect, seed=0, p_exports=True, p_sha256=p_sha, files={
                f'{aspect}__s0__P.npz': {k: dict(shape=list(v.shape), dtype=str(v.dtype), sha256=sha(v))
                                         for k, v in (('img', x), ('cap', cap))}})
            (folder / f'{aspect}__s0.json').write_text(json.dumps(record))


def flat(level):
    return {a: level for a in CLASSES}


def setup(tmp_path, signal, bad=None):
    names = [f'p{i:04d}' for i in range(N)]
    rb_data.write_p_json(tmp_path, names, {'train': 300, 'val': 100})
    p_sha = rb_data.names_sha256(names)
    labels = synthetic_labels()
    write_exports(tmp_path, labels, signal, p_sha, bad=bad)
    return labels, p_sha


@pytest.fixture(scope='module')
def planted_run(tmp_path_factory):
    """R6-3 strong on all three aspects, R7-5 on emotion only; the rest as the reference (weak signal)."""
    tmp_path = tmp_path_factory.mktemp('run')
    signal = {r: flat(0.02) for r in rb_rows.ORDER}
    signal['R6-3'] = flat(0.08)
    signal['R7-5'] = dict(flat(0.02), emotion=0.2)
    labels, p_sha = setup(tmp_path, signal)
    pick = rb_pick.run(tmp_path, labels, n_draws=20, workers=2, n_rows=N)
    return tmp_path, labels, p_sha, pick


def test_run_writes_pick_json_with_the_index_keys(planted_run):
    tmp_path, labels, p_sha, pick = planted_run
    on_disk = json.loads((tmp_path / 'part1' / 'pick.json').read_text())
    assert on_disk == pick
    for key in ('p_sha256', 'versions', 'seed_ranges', 'recipes', 'candidates', 'outcome', 'recipe', 'label_counts'):
        assert key in pick
    assert pick['p_sha256'] == p_sha and pick['seed_ranges'] == R
    assert pick['label_counts'] == {a: int((labels[a] >= 0).sum()) for a in CLASSES}
    assert set(pick['versions']) >= {'sklearn', 'numpy'}
    assert set(pick['recipes']) == set(rb_rows.ORDER) and set(pick['candidates']) == set(rb_rows.TIE_ORDER)
    for name in rb_rows.ORDER:
        for aspect in rb_pick.ASPECTS:
            assert set(pick['recipes'][name][aspect]) == {'A', 'e', 'lo', 'hi', 'gain'}
    for block in pick['candidates'].values():
        assert {'guard', 'pass', 'g', 'flag_g_le_0'} <= set(block)
    ref = pick['recipes']['R0-5']
    assert all(ref[a]['e'] == 0 and ref[a]['lo'] == ref[a]['hi'] == 0 and ref[a]['gain'] is False for a in CLASSES)


def test_run_values_match_a_direct_computation_and_the_pick_follows_the_rule(planted_run):
    tmp_path, labels, p_sha, pick = planted_run
    for aspect in rb_pick.ASPECTS:
        with np.load(tmp_path / 'part1' / 'train' / 'R6-3' / f'{aspect}__s0__P.npz') as z:
            parts = rb_ami.partitions(z['img'])
        with np.load(tmp_path / 'part1' / 'train' / 'R0-5' / f'{aspect}__s0__P.npz') as z:
            ref = rb_ami.partitions(z['img'])
        everyone = np.flatnonzero(labels[aspect] >= 0)
        a_c, a_r = rb_ami.score(labels[aspect], parts, everyone), rb_ami.score(labels[aspect], ref, everyone)
        got = pick['recipes']['R6-3'][aspect]
        assert got['A'] == pytest.approx(a_c, abs=1e-12) and got['e'] == pytest.approx(a_c - a_r, abs=1e-12)
        assert got['gain'] == rb_pick.gain(got['lo'], got['e'], aspect)
    assert pick['candidates']['R6-3']['pass'] is True
    assert pick['candidates']['R0-3']['pass'] is False
    passing = {n: b['g'] for n, b in pick['candidates'].items() if b['pass']}
    assert pick['recipe'] == rb_pick.pick(passing) and pick['outcome'] == f'P-{pick["recipe"]}'
    assert pick['outcome'] != 'P-KEEP'


def test_partitions_are_saved_with_their_index_and_reused_not_recomputed(planted_run, monkeypatch):
    tmp_path, labels, p_sha, pick = planted_run
    path = tmp_path / 'part1' / 'partitions' / 'R6-3' / 'emotion.npz'
    with np.load(path, allow_pickle=False) as z:
        assert z.files == ['labels'] and z['labels'].dtype == np.int64 and z['labels'].shape == (3, N)
        saved = z['labels']
    meta = json.loads(path.with_suffix('.json').read_text())
    assert meta['arrays']['labels']['sha256'] == sha(saved) and meta['arrays']['labels']['shape'] == [3, N]
    assert 'rb_ami.py' in meta['code_sha256'] and meta['source']['p_sha256'] == p_sha
    assert {'sklearn', 'numpy'} <= set(meta['versions'])
    before = (path.stat().st_mtime_ns, (tmp_path / 'part1' / 'pick.json').stat().st_mtime_ns)

    def boom(*a, **k):
        raise AssertionError('partitions recomputed')
    monkeypatch.setattr(rb_ami, 'partitions', boom)
    again = rb_pick.run(tmp_path, labels, n_draws=20, workers=1, n_rows=N)       # same pick, nothing recomputed
    assert again == pick
    assert path.stat().st_mtime_ns == before[0]


def test_a_cached_partition_of_other_inputs_is_refused(planted_run):
    tmp_path, labels, p_sha, _ = planted_run
    with np.load(tmp_path / 'part1' / 'train' / 'R6-3' / 'emotion__s0__P.npz') as z:
        img = z['img']
    other = img.copy()
    other[0] = other[1]
    with pytest.raises(AssertionError, match='other inputs'):
        rb_pick.fixed_partitions(tmp_path, 'R6-3', 'emotion', other, p_sha)
    with pytest.raises(AssertionError, match='other inputs'):
        rb_pick.fixed_partitions(tmp_path, 'R6-3', 'emotion', img, 'f' * 64)


def test_p_sha_mismatch_and_changed_exports_stop_the_run(tmp_path):
    signal = {r: flat(0.02) for r in rb_rows.ORDER}
    labels, p_sha = setup(tmp_path, signal)
    record = tmp_path / 'part1' / 'train' / 'R7-3' / 'style__s0.json'
    value = json.loads(record.read_text())
    record.write_text(json.dumps(dict(value, p_sha256='0' * 64)))
    with pytest.raises(AssertionError, match='P sha256'):
        rb_pick.run(tmp_path, labels, n_draws=2, workers=1, n_rows=N)
    record.write_text(json.dumps(value))
    npz = tmp_path / 'part1' / 'train' / 'R7-3' / 'style__s0__P.npz'
    with np.load(npz) as z:
        img, cap = z['img'].copy(), z['cap']
    img[0, 0] = img[0, 0] + np.float16(0.01)                    # not the array the trainer recorded
    np.savez(npz, img=img, cap=cap)
    with pytest.raises(AssertionError, match='sha256'):
        rb_pick.run(tmp_path, labels, n_draws=2, workers=1, n_rows=N)
    assert not (tmp_path / 'part1' / 'pick.json').exists()


def test_a_candidate_failing_vp2_does_not_pass_and_a_failing_reference_stops(tmp_path):
    signal = {r: flat(0.02) for r in rb_rows.ORDER}
    signal['R6-3'] = flat(0.08)
    labels, p_sha = setup(tmp_path, signal, bad=('R6-3', 'style'))      # one NaN in R6-3's style export
    pick = rb_pick.run(tmp_path, labels, n_draws=10, workers=1, n_rows=N)
    block = pick['candidates']['R6-3']
    assert block['vp2_ok'] is False and block['pass'] is False and block['g'] is None
    assert block['vp2_problems'] and block['vp2_problems'][0].startswith('style')
    assert pick['recipes']['R6-3'] is None
    assert not (tmp_path / 'part1' / 'partitions' / 'R6-3').exists()
    assert pick['outcome'] == 'P-KEEP' and pick['recipe'] == 'R0-5'
    other = tmp_path / 'ref'
    labels, _ = setup(other, signal, bad=('R0-5', 'genre'))
    with pytest.raises(AssertionError, match='R0-5'):
        rb_pick.run(other, labels, n_draws=2, workers=1, n_rows=N)


def test_pick_json_is_not_overwritten_by_a_run_with_other_results(planted_run):
    tmp_path, labels, p_sha, pick = planted_run
    with pytest.raises(AssertionError, match='pick.json'):
        rb_pick.run(tmp_path, labels, n_draws=21, workers=1, n_rows=N)   # other draw count: other bytes


# ------------------------------------------------------------------ real data (slow: RB_SLOW=1)
import os   # noqa: E402

slow = pytest.mark.skipif(not os.environ.get('RB_SLOW'), reason='real P labels: set RB_SLOW=1')


@slow
def test_real_p_labels_at_full_size_parallel_equals_serial():
    labels = rb_data.p_labels()
    rng = np.random.default_rng(0)
    parts = {}
    for name in ('R0-5', 'R6-3'):
        x = rng.standard_normal((rb_pick.N_P, 64)).astype(np.float32)
        parts[name] = rb_ami.partitions((x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float16))
    for aspect in rb_pick.ASPECTS:
        serial = rb_pick.bootstrap_scores(labels[aspect], parts, 4, workers=1)
        parallel = rb_pick.bootstrap_scores(labels[aspect], parts, 4, workers=8)
        for name in parts:
            np.testing.assert_array_equal(serial[name], parallel[name])


# ------------------------------------------------------------------ review fixes (ticket 03, build 17)
def test_label_hashes_and_triggers_are_in_pick_json(planted_run):
    tmp_path, labels, p_sha, pick = planted_run
    assert pick['label_sha256'] == {a: sha(labels[a].astype(np.int64)) for a in CLASSES}
    assert pick['outcome'] != 'P-KEEP'
    assert pick['triggers'] == {'K4': True, 'K5': pick['recipe'] in ('R7-5', 'R7-3')}
    assert pick['vp2_failed'] == {}


def test_triggers_follow_the_rule_line():
    assert rb_pick.triggers('P-KEEP', 'R0-5') == {'K4': False, 'K5': False}
    assert rb_pick.triggers('P-R0-3', 'R0-3') == {'K4': True, 'K5': False}
    assert rb_pick.triggers('P-R6-5', 'R6-5') == {'K4': True, 'K5': False}
    assert rb_pick.triggers('P-R7-5', 'R7-5') == {'K4': True, 'K5': True}
    assert rb_pick.triggers('P-R7-3', 'R7-3') == {'K4': True, 'K5': True}


def test_a_candidate_still_failing_vp2_after_its_rerun_is_excluded_without_loading_its_exports(tmp_path):
    signal = {r: flat(0.02) for r in rb_rows.ORDER}
    signal['R6-3'] = flat(0.08)
    labels, p_sha = setup(tmp_path, signal)
    shutil.rmtree(tmp_path / 'part1' / 'train' / 'R6-3')                  # the export was never written
    with pytest.raises(FileNotFoundError):                                # not on the list: still an error
        rb_pick.run(tmp_path, labels, n_draws=2, workers=1, n_rows=N)
    pick = rb_pick.run(tmp_path, labels, n_draws=10, workers=1, n_rows=N, vp2_failed={'R6-3': 'norm error 0.5'})
    block = pick['candidates']['R6-3']
    assert block == {'vp2_ok': False, 'vp2_problems': ['norm error 0.5'], 'guard': None, 'pass': False, 'g': None,
                     'flag_g_le_0': None}
    assert pick['recipes']['R6-3'] is None and pick['vp2_failed'] == {'R6-3': 'norm error 0.5'}
    assert 'R6-3' not in pick['sources']
    assert not (tmp_path / 'part1' / 'partitions' / 'R6-3').exists()
    assert pick['outcome'] == 'P-KEEP' and pick['recipe'] == 'R0-5' and pick['triggers'] == {'K4': False, 'K5': False}
    assert pick['recipes']['R7-3'] is not None                            # the others are scored


def test_vp2_failed_refuses_the_reference_and_unknown_recipes(tmp_path):
    signal = {r: flat(0.02) for r in rb_rows.ORDER}
    labels, _ = setup(tmp_path, signal)
    with pytest.raises(AssertionError, match='R0-5'):
        rb_pick.run(tmp_path, labels, n_draws=2, workers=1, n_rows=N, vp2_failed={'R0-5': 'x'})
    with pytest.raises(AssertionError, match='R9-9'):
        rb_pick.run(tmp_path, labels, n_draws=2, workers=1, n_rows=N, vp2_failed={'R9-9': 'x'})
    assert not (tmp_path / 'part1' / 'pick.json').exists()


def test_vp2_failed_command_line_form():
    assert rb_pick.parse_vp2_failed([]) == {}
    assert rb_pick.parse_vp2_failed(['R6-3=norm error 0.5 = bad', 'R7-5=nan']) == {'R6-3': 'norm error 0.5 = bad',
                                                                                  'R7-5': 'nan'}
    with pytest.raises(AssertionError):
        rb_pick.parse_vp2_failed(['R6-3'])
    with pytest.raises(AssertionError):
        rb_pick.parse_vp2_failed(['R6-3=a', 'R6-3=b'])
