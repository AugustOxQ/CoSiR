"""Tests of rs_vp (ticket 12): the independent re-derivation of the development side. Expected values come from the
rule's text (DECISION_RULE.md §3.2, §3.3, §4.2, §4.3) or from hand-built cases, never from the scorer."""
import json

import numpy as np
import pytest

import rs_sheet
import rs_vp

RA = dict(emotion=0.0046, style=0.0082, genre=0.0128)     # rule §3, written out again here on purpose
G_A = dict(emotion=0.10, style=0.40, genre=0.50)


def runs_for(deltas, failed=()):
    """Runs for the 39 stage 1 names plus G. `deltas`: {name: (d_emotion, d_style, d_genre)} added to G's A; the
    other stage 1 runs sit exactly at G (gain 0). `failed`: names whose run failed (no A)."""
    out = {'G': dict(status='ok', A=dict(G_A))}
    for cfg in rs_sheet.stage1():
        name = rs_sheet.name(cfg)
        d = deltas.get(name, (0.0, 0.0, 0.0))
        out[name] = dict(status='ok', A={a: G_A[a] + x for a, x in zip(rs_vp.ASPECTS, d)})
    for name in failed:
        out[name] = dict(status='failed', A=None)
    return out


def with_grid(deltas, new=None, failed=()):
    """Stage 1 runs plus every new grid point: at G unless `new` ({name: deltas}) says otherwise."""
    runs = runs_for(deltas)
    for name in rs_vp.derive_stage1(runs)['grid']['new']:
        d = (new or {}).get(name, (0.0, 0.0, 0.0))
        runs[name] = dict(status='failed', A=None) if name in failed else dict(
            status='ok', A={a: G_A[a] + x for a, x in zip(rs_vp.ASPECTS, d)})
    return runs


def ranges(x=1.0):
    """A run that is `x` seed ranges above G on every aspect (M = x)."""
    return tuple(RA[a] * x for a in rs_vp.ASPECTS)


# ---- constants from the rule ------------------------------------------------------------------------------------

def test_constants_are_the_rules_and_the_sheets():
    assert rs_vp.R_A == RA and rs_vp.TOL == 1e-6
    assert rs_vp.R_A == rs_sheet.RANGES        # cross-check against the sheet; rs_vp itself does not import it for r
    assert (rs_vp.N_P, rs_vp.N_DRAWS, rs_vp.K) == (29232, 2000, 20)
    assert rs_vp.DEV_SEEDS == (1, 2, 3)


# ---- gains, mean gain, guard (rule §3.2) ------------------------------------------------------------------------

def test_mean_gain_is_the_mean_of_gains_in_seed_ranges():
    g = rs_vp.gains(dict(emotion=0.1046, style=0.4082, genre=0.5256), G_A)
    assert g == pytest.approx(dict(emotion=0.0046, style=0.0082, genre=0.0256))
    assert rs_vp.mean_gain(g) == pytest.approx((1 + 1 + 2) / 3)


def test_guard_edges():
    # exactly r_a / 2 below G passes; a hair below fails; one aspect is enough to fail
    edge = {a: -RA[a] / 2 for a in RA}
    assert rs_vp.guard_ok(edge)
    for a in RA:
        below = dict(edge, **{a: -RA[a] / 2 - 1e-9})
        assert not rs_vp.guard_ok(below)
    assert rs_vp.guard_ok(dict(emotion=0.0, style=0.0, genre=0.0))


# ---- stage 1: knob gains, grid knobs, levels, grid (rule §4.2) -----------------------------------------------------

def test_stage1_row_counting_and_failed_run():
    runs = runs_for({'k6=0.0005': (RA['emotion'], 0.0, 0.0), 'k7=0.0': (0.0, -RA['style'] / 2 - 1e-6, 0.1)},
                    failed=('k5=0.5',))
    rows = rs_vp.derive_stage1(runs)['rows']
    assert rows['k6=0.0005']['counting'] and rows['k6=0.0005']['M'] == pytest.approx(1 / 3)
    assert rows['k7=0.0']['guard'] is False and rows['k7=0.0']['counting'] is False
    assert rows['k5=0.5']['status'] == 'failed' and rows['k5=0.5']['counting'] is False


def test_no_gain_is_sweep_none_and_a_knob_with_no_counting_value_has_no_gain():
    s1 = rs_vp.derive_stage1(runs_for({}))          # every run equals G: M = 0 everywhere, g_k = 0, not > 0
    assert all(g == 0 for g in s1['knob_gains'].values())
    assert s1['grid_knobs'] == [] and s1['outcome'] == 'SWEEP-NONE'
    # all three values of k13 fail the guard: its knob gain is none
    bad = {f'k13={v}': (0.0, 0.0, -RA['genre']) for v in (5, 10, 30)}
    assert rs_vp.derive_stage1(runs_for(bad))['knob_gains']['k13'] is None


def test_grid_knobs_largest_four_ties_to_sheet_order_and_the_grid():
    # six knobs gain; k4 and k5 tie at M 1 behind k3 (3) and k6 (2): the four are k3, k6 and, by sheet order, k4, k5
    g = RA['genre']
    d = {'k2=8': (0, 0, g * 0.5), 'k3=12': (0, 0, g * 3), 'k4=0.025': (0, 0, g), 'k5=0.5': (0, 0, g),
         'k6=0.0005': (0, 0, g * 2), 'k7=0.01': (0, 0, g * 0.25)}
    s1 = rs_vp.derive_stage1(runs_for(d))
    assert s1['grid_knobs'] == ['k3', 'k4', 'k5', 'k6'] and s1['outcome'] == 'GRID-4'
    assert s1['levels']['k3'] == [8, 12]            # G's value, then its one positive counting value
    points = s1['grid']['points']
    assert len(points) == 2 ** 4
    by_kind = lambda k: sorted(p['name'] for p in points if p['kind'] == k)    # noqa: E731
    assert by_kind('g0') == ['G']
    assert by_kind('reused') == sorted(['k3=12', 'k4=0.025', 'k5=0.5', 'k6=0.0005'])
    assert len(by_kind('new')) == 11 and sorted(s1['grid']['new']) == by_kind('new')
    assert 'k3=12+k4=0.025' in by_kind('new') and 'k3=12+k4=0.025+k5=0.5+k6=0.0005' in by_kind('new')


def test_grid_of_one_knob_with_two_levels_has_no_new_point():
    s1 = rs_vp.derive_stage1(runs_for({'k3=12': ranges(1)}))
    assert s1['outcome'] == 'GRID-1' and s1['grid']['new'] == []


def test_levels_take_the_two_largest_m_ties_to_listed_order():
    g = RA['genre']
    # k3: 12 and 16 tie at M 1, 4 has 0.5: the two largest are 12 and 16
    d = {'k3=4': (0, 0, g * 1.5), 'k3=12': (0, 0, g * 3), 'k3=16': (0, 0, g * 3)}
    assert rs_vp.derive_stage1(runs_for(d))['levels']['k3'] == [8, 12, 16]
    # three positive values: the two largest M, in descending order
    d = {'k3=4': (0, 0, g * 3), 'k3=12': (0, 0, g * 4.5), 'k3=16': (0, 0, g * 6)}
    assert rs_vp.derive_stage1(runs_for(d))['levels']['k3'] == [8, 16, 12]
    # a counting value with M below zero is no level
    d = {'k3=4': (0, 0, g * 3), 'k3=12': (-RA['emotion'] / 4, 0, 0)}
    assert rs_vp.derive_stage1(runs_for(d))['levels']['k3'] == [8, 4]
    # exactly three ties: listed order decides which two
    d = {'k3=4': (0, 0, g * 3), 'k3=12': (0, 0, g * 3), 'k3=16': (0, 0, g * 3)}
    assert rs_vp.derive_stage1(runs_for(d))['levels']['k3'] == [8, 4, 12]


def test_stage1_needs_every_run():
    runs = runs_for({})
    del runs['k9=512']
    with pytest.raises(AssertionError):
        rs_vp.derive_stage1(runs)


# ---- stage 2: the pool and TOP (rule §4.2) ---------------------------------------------------------------------

def test_pool_excludes_g_and_g_repeat_and_top_is_three_by_m():
    runs = with_grid({'k3=12': ranges(1), 'k6=0.0005': ranges(1), 'k9=1536': ranges(0.3)},
                     new={'k3=12+k6=0.0005': ranges(2), 'k3=12+k6=0.0005+k9=1536': ranges(0.5)})
    runs['G-repeat'] = dict(status='ok', A={a: G_A[a] + RA[a] * 9 for a in RA})      # would win, but is G0'
    s2 = rs_vp.derive_stage2(runs)
    assert 'G' not in s2['pool'] and 'G-repeat' not in s2['pool']
    assert sorted(s2['pool']) == sorted(['k3=12', 'k6=0.0005', 'k9=1536', 'k3=12+k6=0.0005',
                                         'k3=12+k6=0.0005+k9=1536'])
    assert s2['top'] == ['k3=12+k6=0.0005', 'k3=12', 'k6=0.0005'] and s2['outcome'] == 'TOP'
    assert s2['rows']['k3=12+k6=0.0005']['M'] == pytest.approx(2.0)


def test_top_ties_fewer_changed_knobs_then_knob_numbers_then_positions():
    one = ranges(1)
    # equal M: one-knob runs before two-knob runs; among them the smaller knob number
    runs = with_grid({'k2=8': one, 'k3=12': one, 'k6=0.0005': one}, new={'k2=8+k3=12': one})
    assert rs_vp.derive_stage2(runs)['top'] == ['k2=8', 'k3=12', 'k6=0.0005']
    # two-knob ties: the sorted tuples (2, 5) and (3, 4) compare lexicographically
    d = {'k2=8': one, 'k3=12': one, 'k4=0.025': one, 'k5=0.5': one}
    runs = with_grid(d, new={'k2=8+k5=0.5': ranges(2), 'k3=12+k4=0.025': ranges(2)})
    assert rs_vp.derive_stage2(runs)['top'] == ['k2=8+k5=0.5', 'k3=12+k4=0.025', 'k2=8']
    # the same knob: the value's position in the sheet (k9 = 512 is position 0, 1536 is position 1)
    runs = with_grid({'k9=1536': one, 'k9=512': one, 'k3=12': ranges(0.5)})
    assert rs_vp.derive_stage2(runs)['top'] == ['k9=512', 'k9=1536', 'k3=12']


def test_top_with_fewer_than_three_and_a_failed_new_point():
    runs = with_grid({'k3=12': ranges(1), 'k6=0.0005': ranges(2)}, failed=('k3=12+k6=0.0005',))
    s2 = rs_vp.derive_stage2(runs)
    assert s2['top'] == ['k6=0.0005', 'k3=12'] and 'k3=12+k6=0.0005' not in s2['pool']
    row = s2['rows']['k3=12+k6=0.0005']
    assert row['status'] == 'failed' and row['counting'] is False


def test_stage2_refuses_a_missing_new_run_and_sweep_none():
    runs = runs_for({'k3=12': ranges(1), 'k6=0.0005': ranges(1)})
    with pytest.raises(AssertionError):
        rs_vp.derive_stage2(runs)            # the new point k3+k6 has no run
    with pytest.raises(AssertionError):
        rs_vp.derive_stage2(runs_for({}))    # SWEEP-NONE: stage 2 does not run


# ---- stage 3: the qualifiers and the pick (rule §4.3) ------------------------------------------------------------

def test_worked_example_of_the_rule():
    e = dict(emotion=-0.0010, style=0.0100, genre=0.0080)
    assert rs_vp.mean_gain(e) == pytest.approx(0.542, abs=5e-4)               # (-0.217 + 1.220 + 0.625) / 3
    lo = dict(emotion=-0.0040, style=-0.0081, genre=-0.0127)
    assert rs_vp.qualifies(lo, 0.1)                                           # emotion: -0.0040 > -0.0046


def test_qualifier_edges_are_strict():
    lo = {a: -RA[a] + 1e-12 for a in RA}
    assert rs_vp.qualifies(lo, 1e-12)
    for a in RA:
        assert not rs_vp.qualifies(dict(lo, **{a: -RA[a]}), 1.0)              # lower bound = -r_a does not pass
    assert not rs_vp.qualifies(lo, 0.0)                                       # M-bar's lower bound must be above 0


def stat(M, ok=True, status='ok'):
    return dict(status=status, M=dict(M=M, lo=0.1, hi=1.0), qualifies=ok and status == 'ok')


def test_pick_is_the_largest_m_bar_among_qualifiers_ties_to_top_order():
    order = ['a', 'b', 'c']
    pick = rs_vp.choose_pick(order, dict(a=stat(0.4), b=stat(0.9, ok=False), c=stat(0.6)))
    assert pick == dict(pick='c', outcome='PICK', qualifiers=['a', 'c'])
    pick = rs_vp.choose_pick(order, dict(a=stat(0.5), b=stat(0.5), c=stat(0.5)))
    assert pick['pick'] == 'a' and pick['qualifiers'] == ['a', 'b', 'c']
    pick = rs_vp.choose_pick(order, dict(a=stat(0.5, ok=False), b=stat(0.5, status='failed'), c=stat(0.5, ok=False)))
    assert pick == dict(pick=None, outcome='KEEP-G', qualifiers=[])


# ---- the joint bootstrap (rule §3.3) --------------------------------------------------------------------------------

def toy(n=40, unlabelled=True, seed=5):
    """Labels (some -1) and, per arm name, parts[aspect][seed] = int64 [3, n]."""
    rng = np.random.default_rng(seed)
    labels = {a: rng.integers(0, 4, size=n).astype(np.int64) for a in rs_vp.ASPECTS}
    if unlabelled:
        labels['genre'][rng.random(n) < 0.3] = -1
    arms = {name: {a: {s: rng.integers(0, 4, size=(3, n)).astype(np.int64) for s in rs_vp.DEV_SEEDS}
                   for a in rs_vp.ASPECTS} for name in ('ref', 'c')}
    return labels, arms


def test_draw_order_is_rows_then_seeds_from_a_fresh_rng_per_configuration():
    draws = list(rs_vp.draw_stream(10, 4))
    rng = np.random.default_rng(0)
    for I, S in draws:
        assert np.array_equal(I, rng.integers(0, 10, size=10))                # I_b first ...
        assert np.array_equal(S, rng.choice([1, 2, 3], size=3, replace=True))  # ... then S_b
    again = list(rs_vp.draw_stream(10, 4))
    assert all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) for a, b in zip(draws, again))


def brute(labels, arms, n_draws):
    """The rule's text, one draw at a time: A_a(v, I_b, s) over the labelled rows of I_b, repeats kept."""
    from sklearn.metrics import adjusted_mutual_info_score as ami
    rng = np.random.default_rng(0)
    e = {a: [] for a in rs_vp.ASPECTS}
    for _ in range(n_draws):
        I = rng.integers(0, len(labels['emotion']), size=len(labels['emotion']))
        S = rng.choice([1, 2, 3], size=3, replace=True)
        for a in rs_vp.ASPECTS:
            J = I[labels[a][I] >= 0]
            vals = {v: [np.mean([ami(labels[a][J], arms[v][a][s][t][J]) for t in range(3)]) for s in S]
                    for v in ('ref', 'c')}
            e[a].append(np.mean(vals['c']) - np.mean(vals['ref']))
    return e


def test_joint_bootstrap_equals_the_rules_text_on_a_toy_case():
    labels, arms = toy()
    out = rs_vp.effects(labels, 'ref', arms, n_draws=7, workers=1)['c']
    e_b = brute(labels, arms, 7)
    for a in rs_vp.ASPECTS:
        lo, hi = np.percentile(e_b[a], [2.5, 97.5])
        assert out['e'][a]['lo'] == pytest.approx(lo, abs=1e-12) and out['e'][a]['hi'] == pytest.approx(hi, abs=1e-12)
    m_b = [np.mean([e_b[a][b] / RA[a] for a in rs_vp.ASPECTS]) for b in range(7)]
    lo, hi = np.percentile(m_b, [2.5, 97.5])
    assert out['M']['lo'] == pytest.approx(lo, abs=1e-9) and out['M']['hi'] == pytest.approx(hi, abs=1e-9)


def test_point_values_use_all_labelled_rows_and_the_three_seeds():
    from sklearn.metrics import adjusted_mutual_info_score as ami
    labels, arms = toy()
    out = rs_vp.effects(labels, 'ref', arms, n_draws=2, workers=1)
    for a in rs_vp.ASPECTS:
        idx = np.flatnonzero(labels[a] >= 0)
        abar = {v: np.mean([np.mean([ami(labels[a][idx], arms[v][a][s][t][idx]) for t in range(3)])
                            for s in rs_vp.DEV_SEEDS]) for v in ('ref', 'c')}
        assert out['ref']['Abar'][a] == pytest.approx(abar['ref'], abs=1e-12)
        assert out['c']['Abar'][a] == pytest.approx(abar['c'], abs=1e-12)
        assert out['c']['e'][a]['e'] == pytest.approx(abar['c'] - abar['ref'], abs=1e-12)
    m = np.mean([out['c']['e'][a]['e'] / RA[a] for a in rs_vp.ASPECTS])
    assert out['c']['M']['M'] == pytest.approx(m, abs=1e-12)


def test_joint_mode_equals_per_aspect_mode_when_every_row_carries_the_label():
    labels, arms = toy(unlabelled=False)
    joint = rs_vp.effects(labels, 'ref', arms, n_draws=6, workers=1, mode='joint')['c']['e']
    per = rs_vp.effects(labels, 'ref', arms, n_draws=6, workers=1, mode='aspect')['c']['e']
    for a in rs_vp.ASPECTS:
        assert joint[a] == pytest.approx(per[a], abs=1e-12)
    labels, arms = toy(unlabelled=True)       # genre has unlabelled rows: only genre's draws differ
    joint = rs_vp.effects(labels, 'ref', arms, n_draws=6, workers=1, mode='joint')['c']['e']
    per = rs_vp.effects(labels, 'ref', arms, n_draws=6, workers=1, mode='aspect')['c']['e']
    assert joint['emotion'] == pytest.approx(per['emotion'], abs=1e-12)
    assert joint['genre']['lo'] != per['genre']['lo']


def test_workers_do_not_change_the_draws():
    labels, arms = toy()
    one = rs_vp.effects(labels, 'ref', arms, n_draws=8, workers=1)['c']
    four = rs_vp.effects(labels, 'ref', arms, n_draws=8, workers=4)['c']
    assert one == four          # same draws, same AMIs: identical, not just close


# ---- files: exports, partitions, labels (rule §3.2, RB §3.1) -------------------------------------------------------

N_T, W_T = 120, 24       # a tiny P: rows, width


def unit_rows(seed, n=N_T, w=W_T):
    x = np.random.default_rng(seed).normal(size=(n, w)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x.astype(np.float16)


def tiny_ctx(seed=3):
    rng = np.random.default_rng(seed)
    labels = {a: rng.integers(0, 5, size=N_T).astype(np.int64) for a in rs_vp.ASPECTS}
    labels['genre'][::4] = -1
    return rs_vp.Context(names=[f'p{i:03d}' for i in range(N_T)], labels=labels, n_rows=N_T, width=W_T)


def write_run(folder, seed, status='ok', bad=None, rows=None):
    """A run folder with run.json and three exports on the tiny P."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'run.json').write_text(json.dumps(dict(status=status, cause='non-finite loss' if status == 'failed' else None)))
    if status == 'failed':
        return
    for i, a in enumerate(rs_vp.ASPECTS):
        x = unit_rows(seed * 10 + i) if rows is None else rows[a]
        if bad == a:
            x = (x.astype(np.float32) * 1.1).astype(np.float16)           # norm 1.1: fails V1b
        np.savez(folder / f'{a}__P.npz', img=x)


def test_export_checks_name_the_problem(tmp_path):
    ctx, rec = tiny_ctx(), rs_vp.Recorder()
    ok = tmp_path / 'ok.npz'
    np.savez(ok, img=unit_rows(1))
    x, sha = rs_vp.read_export(rec, ok, ctx)
    assert x.dtype == np.float32 and x.shape == (N_T, W_T) and str(ok) in rec.hashes
    for name, img in (('f32', unit_rows(1).astype(np.float32)), ('shape', unit_rows(1)[:-1]),
                      ('norm', (unit_rows(1).astype(np.float32) * 1.01).astype(np.float16)),
                      ('nan', np.full((N_T, W_T), np.nan, np.float16))):
        p = tmp_path / f'{name}.npz'
        np.savez(p, img=img)
        with pytest.raises(rs_vp.ExportProblem):
            rs_vp.read_export(rec, p, ctx)


def test_partitions_are_three_states_of_20_clusters_and_repeatable():
    x = unit_rows(2).astype(np.float32)
    p1, p2 = rs_vp.partitions(x, threads=1), rs_vp.partitions(x, threads=1)
    assert p1.shape == (3, N_T) and p1.dtype == np.int64 and p1.min() >= 0 and p1.max() < 20
    assert np.array_equal(p1, p2)
    from sklearn.cluster import MiniBatchKMeans
    for state in (0, 1, 2):
        km = MiniBatchKMeans(n_clusters=20, random_state=state, n_init=3, batch_size=4096)
        assert np.array_equal(p1[state], km.fit_predict(x))


def test_labels_plurality_ties_to_smallest_code_first_caption_style_genre_given():
    ann = [dict(painting='a', emotion='sadness', art_style='Baroque'), dict(painting='a', emotion='awe', art_style='Pop'),
           dict(painting='a', emotion='sadness', art_style='Pop'), dict(painting='b', emotion='fear', art_style='Pop'),
           dict(painting='b', emotion='awe', art_style='Baroque')]
    lab = rs_vp.labels_from_annotations(['a', 'b'], ann, np.array([3, -1]))
    # emotion codes sorted: awe 0, fear 1, sadness 2; style codes: Baroque 0, Pop 1
    assert lab['emotion'].tolist() == [2, 0]          # a: sadness twice; b: awe and fear tie -> smallest code
    assert lab['style'].tolist() == [0, 1]            # first caption's
    assert lab['genre'].tolist() == [3, -1]


# ---- part (a): partitions and A of the runs vs the score files ------------------------------------------------------

def score_file(outputs, machine, name, ctx, folder_seed, status='ok', delta=0.0):
    """What a correct scorer writes, built here with sklearn directly (not through rs_vp)."""
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import adjusted_mutual_info_score as ami
    d = outputs / 'scores' / machine
    (d / 'partitions' / name / 's0').mkdir(parents=True, exist_ok=True)
    A = {}
    for i, a in enumerate(rs_vp.ASPECTS):
        x = unit_rows(folder_seed * 10 + i).astype(np.float32)
        parts = np.stack([MiniBatchKMeans(n_clusters=20, random_state=s, n_init=3, batch_size=4096).fit(x).labels_
                          for s in (0, 1, 2)]).astype(np.int64)
        np.savez(d / 'partitions' / name / 's0' / f'{a}__partitions.npz', labels=parts)
        idx = np.flatnonzero(ctx.labels[a] >= 0)
        A[a] = float(np.mean([ami(ctx.labels[a][idx], p[idx]) for p in parts])) + delta
    (d / f'{name}__s0.json').write_text(json.dumps(dict(stage='1', name=name, seed=0, machine=machine, status=status,
                                                         A=A if status == 'ok' else None)))


def test_part_a_agrees_with_a_correct_scorer_and_catches_a_wrong_a_or_partition(tmp_path):
    ctx = tiny_ctx()
    out = tmp_path / 'out'
    write_run(out / 'train' / 'G' / 's0', 1)
    write_run(out / 'train' / 'k6=0.0005' / 's0', 2)
    write_run(out / 'train' / 'k6=0.0005-no_buddy' / 's0', 3)           # an ablation arm: not a stage 1 or 2 run
    write_run(out / 'train' / 'k3=12' / 's1', 4)                        # seed 1 only: no seed 0 run
    write_run(out / 'train' / 'G-repeat' / 's0', 1)                     # G0' is a stage 1 and 2 run too
    score_file(out, 'local', 'G', ctx, 1)
    score_file(out, 'local', 'G-repeat', ctx, 1)
    score_file(out, 'local', 'k6=0.0005', ctx, 2)
    rep = rs_vp.part_a(out, 'local', ctx, threads=1)
    assert rep['pass'] is True and sorted(rep['runs']) == ['G', 'G-repeat', 'k6=0.0005']
    saved = json.loads((out / 'scores' / 'local' / 'rederive_partitions.json').read_text())
    assert saved['pass'] is True and saved['items'] and all(i['pass'] for i in saved['items'].values())
    assert any(k.startswith('A/G/') for k in saved['items']) and saved['input_sha256']
    # a score file whose A is off by 2e-6 disagrees; by 1e-7 does not
    score_file(out, 'local', 'G', ctx, 1, delta=2e-6)
    assert rs_vp.part_a(out, 'local', ctx, threads=1)['pass'] is False
    score_file(out, 'local', 'G', ctx, 1, delta=1e-7)
    assert rs_vp.part_a(out, 'local', ctx, threads=1)['pass'] is True
    # a different partition file disagrees even when A is right
    pfile = out / 'scores' / 'local' / 'partitions' / 'G' / 's0' / 'genre__partitions.npz'
    labels = np.load(pfile)['labels'].copy()
    labels[0, 0] = (labels[0, 0] + 1) % 20
    np.savez(pfile, labels=labels)
    assert rs_vp.part_a(out, 'local', ctx, threads=1)['pass'] is False


def test_part_a_failed_run_and_export_failing_v1b_and_missing_score_file(tmp_path):
    ctx = tiny_ctx()
    out = tmp_path / 'out'
    write_run(out / 'train' / 'G' / 's0', 1)
    write_run(out / 'train' / 'k1=2' / 's0', 2, status='failed')
    write_run(out / 'train' / 'k2=2' / 's0', 3, bad='style')
    score_file(out, 'local', 'G', ctx, 1)
    for name in ('k1=2', 'k2=2'):
        (out / 'scores' / 'local' / f'{name}__s0.json').write_text(json.dumps(dict(name=name, status='failed', A=None,
                                                                                   cause='x')))
    rep = rs_vp.part_a(out, 'local', ctx, threads=1)
    assert rep['pass'] is True
    assert rep['runs']['k1=2']['status'] == 'failed' and rep['runs']['k2=2']['status'] == 'failed'
    assert rep['runs']['k2=2']['A'] is None and 'norm' in rep['runs']['k2=2']['cause']
    (out / 'scores' / 'local' / 'k2=2__s0.json').unlink()
    assert rs_vp.part_a(out, 'local', ctx, threads=1)['pass'] is False     # no score file for a run


def test_part_a_uses_frozen_g_when_the_machine_has_no_g_run(tmp_path):
    ctx = tiny_ctx()
    out, frozen = tmp_path / 'out', tmp_path / 'frozen' / 's0'
    write_run(frozen, 1)
    write_run(out / 'train' / 'k6=0.0005' / 's0', 2)
    score_file(out, 'local', 'G', ctx, 1)
    score_file(out, 'local', 'k6=0.0005', ctx, 2)
    rep = rs_vp.part_a(out, 'local', ctx, threads=1, frozen_g=frozen)
    assert rep['pass'] is True and rep['runs']['G']['source'] == str(frozen)


# ---- part (b): the sweep's decisions from both sets of A values ------------------------------------------------------

def stage1_json(d1):
    """A stage 1 file in the shape the ticket names (rows as a list, curves omitted)."""
    return dict(stage='1', ranges=RA, reference=dict(name='G', A=d1['rows']['G']['A']),
                runs=[dict(r, best_epoch=None, cause=None) for r in d1['rows'].values()],
                knob_gains=d1['knob_gains'], grid_knobs=d1['grid_knobs'], outcome=d1['outcome'], levels=d1['levels'],
                grid=dict(points=d1['grid']['points'], order=[], g0='G', reused=d1['grid']['reused'],
                          new=d1['grid']['new']))


def stage2_json(d2):
    return dict(stage='2', new_runs=[dict(r, best_epoch=None, cause=None) for r in d2['rows'].values()],
                pool=d2['pool'], top=[dict(rank=i + 1, name=n) for i, n in enumerate(d2['top'])], outcome=d2['outcome'])


def sweep_outputs(tmp_path, runs, mine=None, stage1=True, stage2=True):
    """Score files for `runs` (the scorer's A), rederive_partitions.json with `mine` (default: the same A), and the
    scorer's stage files as a correct scorer writes them."""
    out = tmp_path / 'out'
    sdir = out / 'scores' / 'node402'
    sdir.mkdir(parents=True)
    for name, r in runs.items():
        (sdir / f'{name}__s0.json').write_text(json.dumps(dict(name=name, seed=0, status=r['status'], A=r['A'])))
    mine = runs if mine is None else mine
    (sdir / 'rederive_partitions.json').write_text(json.dumps(dict(
        **{'pass': True}, runs={n: dict(status=r['status'], A=r['A']) for n, r in mine.items()})))
    (out / 'sweep' / 'node402').mkdir(parents=True)
    d1 = rs_vp.derive_stage1(runs)
    if stage1:
        (out / 'sweep' / 'node402' / 'stage1.json').write_text(json.dumps(stage1_json(d1)))
    if stage2 and d1['grid_knobs']:
        (out / 'sweep' / 'node402' / 'stage2.json').write_text(json.dumps(stage2_json(rs_vp.derive_stage2(runs))))
    return out


def sweep_case():
    return with_grid({'k3=12': ranges(1), 'k6=0.0005': ranges(2), 'k9=1536': ranges(0.3)},
                     new={'k3=12+k6=0.0005': ranges(2.5)})


def test_sweep_agrees_with_a_correct_scorer_and_writes_next_to_the_stage_files(tmp_path):
    out = sweep_outputs(tmp_path, sweep_case())
    reps = rs_vp.sweep(out, 'node402')
    assert [r['pass'] for r in reps] == [True, True]
    for name in ('rederive_stage1.json', 'rederive_stage2.json'):
        saved = json.loads((out / 'sweep' / 'node402' / name).read_text())
        assert saved['pass'] is True and saved['items'] and saved['input_sha256']
    assert json.loads((out / 'sweep' / 'node402' / 'rederive_stage2.json').read_text())['mine']['top'][0] \
        == 'k3=12+k6=0.0005'


def test_sweep_catches_a_wrong_outcome_knob_gain_pool_or_top(tmp_path):
    out = sweep_outputs(tmp_path, sweep_case())
    f1, f2 = out / 'sweep/node402/stage1.json', out / 'sweep/node402/stage2.json'
    good1, good2 = json.loads(f1.read_text()), json.loads(f2.read_text())
    for edit in (lambda d: d.update(outcome='GRID-4'), lambda d: d['knob_gains'].update(k6=0.5),
                 lambda d: d.update(grid_knobs=['k3', 'k6']), lambda d: d['grid'].update(new=[]),
                 lambda d: d['runs'][3].update(counting=not d['runs'][3]['counting']),
                 lambda d: d['runs'][5].update(M=d['runs'][5]['M'] + 1e-5)):
        bad = json.loads(json.dumps(good1))
        edit(bad)
        f1.write_text(json.dumps(bad))
        assert rs_vp.sweep(out, 'node402')[0]['pass'] is False
    f1.write_text(json.dumps(good1))
    for edit in (lambda d: d.update(top=d['top'][::-1]), lambda d: d.update(pool=d['pool'][1:]),
                 lambda d: d.update(outcome='TOP-NONE')):
        bad = json.loads(json.dumps(good2))
        edit(bad)
        f2.write_text(json.dumps(bad))
        assert rs_vp.sweep(out, 'node402')[1]['pass'] is False


def test_sweep_catches_two_sets_of_a_that_give_another_top_and_refuses_a_failed_part_a(tmp_path):
    runs = sweep_case()
    mine = {n: dict(r) for n, r in runs.items()}
    mine['k3=12'] = dict(status='ok', A={a: runs['k3=12']['A'][a] + 5 * RA[a] for a in RA})   # my A: k3=12 leads
    out = sweep_outputs(tmp_path, runs, mine=mine)
    assert rs_vp.sweep(out, 'node402')[0]['pass'] is False
    p = out / 'scores/node402/rederive_partitions.json'
    d = json.loads(p.read_text())
    d['pass'] = False
    p.write_text(json.dumps(d))
    with pytest.raises(AssertionError, match='part \\(a\\)'):
        rs_vp.sweep(out, 'node402')


def test_sweep_without_stage2_checks_stage1_only_and_sweep_none_has_no_stage2(tmp_path):
    out = sweep_outputs(tmp_path, runs_for({}))
    reps = rs_vp.sweep(out, 'node402')
    assert len(reps) == 1 and reps[0]['pass'] is True and reps[0]['mine']['outcome'] == 'SWEEP-NONE'


# ---- stage 3 and the development ablation: files to verdicts --------------------------------------------------------

def parts_direct(ctx, folder):
    """Partitions of a run folder's exports, by sklearn directly (the test's own route)."""
    from sklearn.cluster import MiniBatchKMeans
    out = {}
    for a in rs_vp.ASPECTS:
        x = np.load(folder / f'{a}__P.npz')['img'].astype(np.float32)
        out[a] = np.stack([MiniBatchKMeans(n_clusters=20, random_state=s, n_init=3, batch_size=4096).fit(x).labels_
                           for s in (0, 1, 2)]).astype(np.int64)
    return out


def write_arm(ctx, folders, tag):
    """Runs for seeds 1 to 3 of one arm; returns parts[aspect][seed]."""
    parts = {a: {} for a in rs_vp.ASPECTS}
    for s, folder in folders.items():
        write_run(folder, tag * 10 + s)
        for a, p in parts_direct(ctx, folder).items():
            parts[a][s] = p
    return parts


def brute_pair(labels, arms, ref, c, n_draws):
    from sklearn.metrics import adjusted_mutual_info_score as ami
    rng = np.random.default_rng(0)
    n = len(labels['emotion'])
    e = {a: [] for a in rs_vp.ASPECTS}
    for _ in range(n_draws):
        I = rng.integers(0, n, size=n)
        S = rng.choice([1, 2, 3], size=3, replace=True)
        for a in rs_vp.ASPECTS:
            J = I[labels[a][I] >= 0]
            f = lambda v: np.mean([np.mean([ami(labels[a][J], arms[v][a][s][t][J]) for t in range(3)])    # noqa: E731
                                   for s in S])
            e[a].append(f(c) - f(ref))
    return e


def stage3_case(tmp_path, ctx):
    out = tmp_path / 'out'
    frozen = {s: tmp_path / 'frozen' / f's{s}' for s in rs_vp.DEV_SEEDS}
    arms = {'G': write_arm(ctx, frozen, 1)}
    for tag, name in ((2, 'k3=12'), (3, 'k6=0.0005')):
        arms[name] = write_arm(ctx, {s: out / 'train' / name / f's{s}' for s in rs_vp.DEV_SEEDS}, tag)
    return out, frozen, arms


def test_stage3_core_on_a_tiny_case_matches_the_rules_text(tmp_path):
    ctx = tiny_ctx()
    out, frozen, arms = stage3_case(tmp_path, ctx)
    gparts = tmp_path / 'g_partitions'
    for s in rs_vp.DEV_SEEDS:
        (gparts / f's{s}').mkdir(parents=True)
        for a in rs_vp.ASPECTS:
            np.savez(gparts / f's{s}' / f'{a}__partitions.npz', labels=arms['G'][a][s])
    mine = rs_vp.stage3_core(out, ['k3=12', 'k6=0.0005'], ctx, rs_vp.Recorder(), n_draws=5, workers=2, threads=1,
                             g_folders=frozen, g_partitions=gparts)
    assert mine['items_pass'] is False                  # no score files of the scorer for the TOP runs: reported
    e_b = brute_pair(ctx.labels, arms, 'G', 'k3=12', 5)
    c = mine['configs']['k3=12']
    for a in rs_vp.ASPECTS:
        assert c['e'][a]['lo'] == pytest.approx(np.percentile(e_b[a], 2.5), abs=1e-12)
        assert c['e'][a]['hi'] == pytest.approx(np.percentile(e_b[a], 97.5), abs=1e-12)
    assert set(mine) >= {'pick', 'outcome', 'qualifiers', 'configs', 'top'}
    assert mine['outcome'] in ('PICK', 'KEEP-G') and (mine['pick'] is None) == (mine['outcome'] == 'KEEP-G')


def write_scorer_files(out, ctx, arms, names):
    """What a correct scorer leaves for seeds 1 to 3 of `names`: partitions and score files under scores/local/."""
    from sklearn.metrics import adjusted_mutual_info_score as ami
    for name in names:
        for s in rs_vp.DEV_SEEDS:
            d = out / 'scores' / 'local'
            (d / 'partitions' / name / f's{s}').mkdir(parents=True, exist_ok=True)
            A = {}
            for a in rs_vp.ASPECTS:
                np.savez(d / 'partitions' / name / f's{s}' / f'{a}__partitions.npz', labels=arms[name][a][s])
                idx = np.flatnonzero(ctx.labels[a] >= 0)
                A[a] = float(np.mean([ami(ctx.labels[a][idx], p[idx]) for p in arms[name][a][s]]))
            (d / f'{name}__s{s}.json').write_text(json.dumps(dict(name=name, seed=s, status='ok', A=A)))


def write_g_partitions(path, arms):
    for s in rs_vp.DEV_SEEDS:
        (path / f's{s}').mkdir(parents=True, exist_ok=True)
        for a in rs_vp.ASPECTS:
            np.savez(path / f's{s}' / f'{a}__partitions.npz', labels=arms['G'][a][s])


def test_stage3_core_compares_partitions_and_score_files(tmp_path):
    ctx = tiny_ctx()
    out, frozen, arms = stage3_case(tmp_path, ctx)
    gparts = tmp_path / 'g_partitions'
    write_scorer_files(out, ctx, arms, ('k3=12', 'k6=0.0005'))
    write_g_partitions(gparts, arms)
    kw = dict(n_draws=3, workers=1, threads=1, g_folders=frozen, g_partitions=gparts)
    assert rs_vp.stage3_core(out, ['k3=12', 'k6=0.0005'], ctx, rs_vp.Recorder(), **kw)['items_pass'] is True
    f = out / 'scores' / 'local' / 'k3=12__s2.json'
    d = json.loads(f.read_text())
    d['A']['style'] += 3e-6
    f.write_text(json.dumps(d))
    assert rs_vp.stage3_core(out, ['k3=12', 'k6=0.0005'], ctx, rs_vp.Recorder(), **kw)['items_pass'] is False
    d['A']['style'] -= 3e-6
    f.write_text(json.dumps(d))
    g = gparts / 's1' / 'emotion__partitions.npz'
    lab = np.load(g)['labels'].copy()
    lab[1, 3] = (lab[1, 3] + 1) % 20
    np.savez(g, labels=lab)
    assert rs_vp.stage3_core(out, ['k3=12', 'k6=0.0005'], ctx, rs_vp.Recorder(), **kw)['items_pass'] is False


def test_stage3_a_config_with_a_failed_or_bad_seed_does_not_qualify_and_has_no_numbers(tmp_path):
    ctx = tiny_ctx()
    out, frozen, arms = stage3_case(tmp_path, ctx)
    write_run(out / 'train' / 'k3=12' / 's2', 0, status='failed')
    write_run(out / 'train' / 'k6=0.0005' / 's3', 0, bad='genre')
    mine = rs_vp.stage3_core(out, ['k3=12', 'k6=0.0005'], ctx, rs_vp.Recorder(), n_draws=3, workers=1, threads=1,
                             g_folders=frozen, g_partitions=None)
    for name in ('k3=12', 'k6=0.0005'):
        c = mine['configs'][name]
        assert c['status'] == 'failed' and c['qualifies'] is False and c['cause'] and 'e' not in c
    assert mine['outcome'] == 'KEEP-G' and mine['qualifiers'] == []


def verdict_like(mine):
    """A verdict file in the layout compare_stage3 assumes."""
    return dict(pick=mine['pick'], outcome=mine['outcome'], qualifiers=mine['qualifiers'],
                configs={n: ({k: c[k] for k in ('status', 'Abar', 'e', 'M', 'qualifies')} if c['status'] == 'ok'
                             else dict(status='failed', cause='x', qualifies=False))
                         for n, c in mine['configs'].items()})


def fake_mine():
    ok = dict(status='ok', cause=None, qualifies=True, Abar=dict(emotion=0.1, style=0.4, genre=0.5),
              e=dict(emotion=dict(e=0.001, lo=-0.001, hi=0.003), style=dict(e=0.002, lo=0.0, hi=0.004),
                     genre=dict(e=0.003, lo=0.001, hi=0.005)), M=dict(M=0.5, lo=0.1, hi=0.9))
    return dict(top=['a', 'b'], pick='a', outcome='PICK', qualifiers=['a'],
                configs={'a': ok, 'b': dict(ok, qualifies=False, M=dict(M=0.4, lo=-0.1, hi=0.9))})


def test_compare_stage3_agrees_and_catches_each_kind_of_difference():
    mine = fake_mine()
    items = {}
    rs_vp.compare_stage3(items, mine, verdict_like(mine))
    assert items and all(i['pass'] for i in items.values())
    edits = (lambda v: v.update(pick='b'), lambda v: v.update(outcome='KEEP-G'), lambda v: v.update(qualifiers=['a', 'b']),
             lambda v: v['configs']['a']['Abar'].update(genre=0.5001),
             lambda v: v['configs']['a']['e']['style'].update(lo=-0.0001),
             lambda v: v['configs']['a']['M'].update(hi=0.95), lambda v: v['configs']['b'].update(qualifies=True))
    for edit in edits:
        v = json.loads(json.dumps(verdict_like(mine)))
        edit(v)
        items = {}
        rs_vp.compare_stage3(items, mine, v)
        assert not all(i['pass'] for i in items.values())
    items = {}
    v = verdict_like(mine)
    v['configs']['a']['Abar']['genre'] += 5e-7                       # inside 1e-6
    rs_vp.compare_stage3(items, mine, v)
    assert all(i['pass'] for i in items.values())


def test_ablation_core_points_e_a_against_the_final_model(tmp_path):
    ctx = tiny_ctx()
    frozen = {s: tmp_path / 'frozen' / f's{s}' for s in rs_vp.DEV_SEEDS}
    arms = {'G': write_arm(ctx, frozen, 1)}
    out = tmp_path / 'out'
    for i, x in enumerate(rs_vp.REMOVALS):
        name = f'G-no_{x}'
        if x == 'direct':
            write_run(out / 'train' / name / 's1', 0, status='failed')       # a failed arm: reported, no effect
            for s in (2, 3):
                write_run(out / 'train' / name / f's{s}', 0, status='failed')
            continue
        arms[name] = write_arm(ctx, {s: out / 'train' / name / f's{s}' for s in rs_vp.DEV_SEEDS}, 2 + i)
    mine = rs_vp.ablation_core(out, 'G', ctx, rs_vp.Recorder(), n_draws=0, workers=1, threads=1, g_folders=frozen,
                               g_partitions=None)
    assert mine['final'] == 'G' and sorted(mine['arms']) == sorted(f'G-no_{x}' for x in rs_vp.REMOVALS)
    assert mine['arms']['G-no_direct']['status'] == 'failed' and 'e' not in mine['arms']['G-no_direct']
    from sklearn.metrics import adjusted_mutual_info_score as ami
    for a in rs_vp.ASPECTS:
        idx = np.flatnonzero(ctx.labels[a] >= 0)
        ab = lambda v: np.mean([np.mean([ami(ctx.labels[a][idx], p[idx]) for p in arms[v][a][s]])      # noqa: E731
                                for s in rs_vp.DEV_SEEDS])
        assert mine['arms']['G-no_buddy']['e'][a]['e'] == pytest.approx(ab('G-no_buddy') - ab('G'), abs=1e-12)
        assert mine['arms']['G-no_buddy']['e'][a]['lo'] is None            # points only (n_draws 0)
    del_dir = out / 'train' / 'G-no_buddy'
    import shutil
    shutil.rmtree(del_dir)
    with pytest.raises(AssertionError, match='arm'):
        rs_vp.ablation_core(out, 'G', ctx, rs_vp.Recorder(), n_draws=0, workers=1, threads=1, g_folders=frozen,
                            g_partitions=None)


def test_compare_ablation_checks_points_and_the_failed_arms():
    arm = dict(status='ok', cause=None, Abar=dict(emotion=0.1, style=0.4, genre=0.5),
               e=dict(emotion=dict(e=0.001, lo=None, hi=None), style=dict(e=0.002, lo=None, hi=None),
                      genre=dict(e=-0.003, lo=None, hi=None)))
    mine = dict(final='G', arms={'G-no_buddy': arm, 'G-no_direct': dict(status='failed', cause='nan loss')})
    theirs = dict(final='G', results={'G-no_buddy': dict(status='ok', effects={a: dict(e=arm['e'][a]['e'], lo=-1, hi=1)
                                                                              for a in rs_vp.ASPECTS}),
                                      'G-no_direct': dict(status='failed', cause='non-finite loss')})
    items = {}
    rs_vp.compare_ablation(items, mine, theirs)
    assert items and all(i['pass'] for i in items.values())
    theirs['results']['G-no_buddy']['effects']['genre']['e'] += 2e-6
    items = {}
    rs_vp.compare_ablation(items, mine, theirs)
    assert not all(i['pass'] for i in items.values())
    theirs['results']['G-no_buddy']['effects']['genre']['e'] -= 2e-6
    theirs['final'] = 'k3=12'
    items = {}
    rs_vp.compare_ablation(items, mine, theirs)
    assert not all(i['pass'] for i in items.values())


def test_stage3_wrapper_takes_top_from_my_a_values_and_compares_with_the_verdict_file(tmp_path):
    ctx = tiny_ctx()
    runs = with_grid({'k3=12': ranges(1), 'k6=0.0005': ranges(2)}, new={'k3=12+k6=0.0005': ranges(2.5)})
    top = rs_vp.derive_stage2(runs)['top']
    assert top == ['k3=12+k6=0.0005', 'k6=0.0005', 'k3=12']
    out = tmp_path / 'out'
    frozen = {s: tmp_path / 'frozen' / f's{s}' for s in rs_vp.DEV_SEEDS}
    arms = {'G': write_arm(ctx, frozen, 1)}
    for tag, name in enumerate(top, start=2):
        arms[name] = write_arm(ctx, {s: out / 'train' / name / f's{s}' for s in rs_vp.DEV_SEEDS}, tag)
    write_scorer_files(out, ctx, arms, top)
    gparts = tmp_path / 'g_partitions'
    write_g_partitions(gparts, arms)
    sdir = out / 'scores' / 'node402'
    sdir.mkdir(parents=True)
    (sdir / 'rederive_partitions.json').write_text(json.dumps(dict(
        **{'pass': True}, runs={n: dict(status=r['status'], A=r['A']) for n, r in runs.items()})))
    kw = dict(n_draws=3, workers=1, threads=1, g_folders=frozen)
    core = rs_vp.stage3_core(out, top, ctx, rs_vp.Recorder(), g_partitions=gparts, **kw)
    verdict = verdict_like(core)
    (out / 'stage3').mkdir()
    vfile = out / 'stage3' / 'verdict.json'
    vfile.write_text(json.dumps(verdict))
    rep = rs_vp.stage3(out, 'node402', ctx=ctx, g_partitions=gparts, **kw)
    assert rep['pass'] is True and rep['mine']['top'] == top
    assert json.loads((out / 'stage3' / 'rederive_stage3.json').read_text())['pass'] is True
    verdict['configs'][top[1]]['M']['lo'] += 1e-5
    vfile.write_text(json.dumps(verdict))
    assert rs_vp.stage3(out, 'node402', ctx=ctx, g_partitions=gparts, **kw)['pass'] is False


def test_command_line_parses_every_mode_and_refuses_outputs_in_a_frozen_folder(tmp_path):
    with pytest.raises(SystemExit):
        rs_vp.main(['partitions'])                                     # --machine is required
    with pytest.raises(AssertionError, match='frozen'):
        rs_vp.main(['sweep', '--machine', 'local', '--outputs', str(rs_vp.paths.RG_OUT / 'x')])
