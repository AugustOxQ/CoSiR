"""Tests for ticket 07 (r8_groupings): what the verdict rests on, per DECISION_RULE.md M3, M7 (labels), M8, M9, M13, M16.

Synthetic data throughout, except the one test that reads R0's frozen labels from the real files.
Run from the round folder: python -m pytest -q -p no:cacheprovider --rootdir=. test_r8_groupings.py
"""
import sys

import numpy as np
import pytest

import r8_groupings as G


# ---------------------------------------------------------------- helpers

def stub(table, default=None):
    """K(r) from a dict keyed by round(r, 6); records the calls."""
    calls = []

    def k_of_r(r):
        calls.append(round(r, 6))
        key = round(r, 6)
        if callable(default) and key not in table:
            return default(r)
        return table.get(key, default)
    k_of_r.calls = calls
    return k_of_r


GRID = [round(j / 10, 6) for j in range(1, 31)]


def pairs(rec):
    return [(e['r'], e['K']) for e in rec['evaluated']]


# ---------------------------------------------------------------- M3: resolution search order and ties

def test_search_stops_at_r1_when_k_is_17():
    f = stub({1.0: 17})
    rec = G.search_resolution(f)
    assert rec['chosen_r'] == 1.0 and rec['K'] == 17 and rec['stage'] == 'r1'
    assert f.calls == [1.0] and pairs(rec) == [(1.0, 17)]


def test_search_grid_picks_k17_nearest_one_ties_to_smaller_r():
    # 17 at 0.7 and 1.3 (both 0.3 from 1.0) and 2.5: the tie goes to the smaller r
    table = {r: 15 for r in GRID}
    table.update({0.7: 17, 1.3: 17, 2.5: 17})
    rec = G.search_resolution(stub(table))
    assert rec['stage'] == 'grid' and rec['chosen_r'] == 0.7 and rec['K'] == 17
    assert [e['r'] for e in rec['evaluated']][:2] == [1.0, 0.1]          # r = 1.0 first, then the grid
    assert sorted(set(e['r'] for e in rec['evaluated'])) == GRID          # every grid point recorded once
    assert len(rec['evaluated']) == 30


def test_search_grid_nearer_point_beats_smaller_r():
    table = {r: 15 for r in GRID}
    table.update({0.5: 17, 1.1: 17})
    assert G.search_resolution(stub(table))['chosen_r'] == 1.1


def monotone(jump_lo, jump_hi, below=16, inside=17, above=18):
    """K = below for r < jump_lo, inside for jump_lo <= r < jump_hi, above beyond (a monotone K(r))."""
    return lambda r: below if r < jump_lo else (inside if r < jump_hi else above)


def test_search_bisects_the_bracketing_pair_with_midpoint_nearest_one():
    # K = 16 below 1.2, 18 above: bracketing pairs (1.2 = 16, 1.3 = 18) only (K(1.2) = 16 since 1.2 < 1.2000001)
    f = stub({}, default=monotone(1.2000001, 1.2000001))
    rec = G.search_resolution(f)
    bis = [e['r'] for e in rec['evaluated']][30:]
    # keeping the half that brackets 17: 1.25 -> 18 keeps [1.2, 1.25]; 1.225 -> 18 keeps [1.2, 1.225]; ...
    assert bis[0] == 1.25 and bis[1] == 1.225 and bis[2] == pytest.approx(1.2125)
    assert len(bis) == 10                                                  # at most 10 bisections
    # never 17: nearest count, |16 - 17| = |18 - 17|: ties to r nearest 1.0 -> the largest r below the jump (16 side)
    assert rec['K'] in (16, 18)
    ks = {e['r']: e['K'] for e in rec['evaluated']}
    cand = [r for r, k in ks.items() if abs(k - 17) == 1]
    best = min(cand, key=lambda r: (abs(r - 1.0), r))
    assert rec['chosen_r'] == best and rec['stage'] == 'nearest'


def test_search_bisection_keeps_upper_half_and_stops_at_first_17():
    # 17 only inside [1.2301, 1.2399): 1.25 -> 18 keep [1.2, 1.25]; 1.225 -> 16 keep [1.225, 1.25]; 1.2375 -> 17 stop
    f = stub({}, default=monotone(1.2301, 1.2399))
    rec = G.search_resolution(f)
    assert rec['stage'] == 'bisect' and rec['K'] == 17 and rec['chosen_r'] == pytest.approx(1.2375)
    assert [e['r'] for e in rec['evaluated']][30:] == [1.25, 1.225, 1.2375]


def test_search_bracket_choice_prefers_midpoint_nearest_one():
    # no 17 anywhere; K = 15 except 0.7 -> 19 and 1.6 -> 19: brackets (0.6,0.7) mid 0.65, (0.7,0.8) mid 0.75,
    # (1.5,1.6) mid 1.55, (1.6,1.7) mid 1.65 -> the midpoint nearest 1.0 is 0.75
    table = {r: 15 for r in GRID}
    table.update({0.7: 19, 1.6: 19})
    rec = G.search_resolution(stub(table))
    assert [e['r'] for e in rec['evaluated']][30] == pytest.approx(0.75)


def test_search_grid_tie_is_decided_after_rounding_the_distance_from_one():
    # 0.6 and 1.4 are both 0.4 from 1.0; in floats abs(1.4 - 1.0) = 0.3999999999999999 would win. Rule: smaller r.
    table = {r: 15 for r in GRID}
    table.update({0.6: 17, 1.4: 17})
    rec = G.search_resolution(stub(table))
    assert rec['stage'] == 'grid' and rec['chosen_r'] == 0.6
    # the other grid pairs that float noise breaks: (0.1, 1.9), (0.4, 1.6), (0.9, 1.1)
    for lo, hi in ((0.1, 1.9), (0.4, 1.6), (0.9, 1.1)):
        t = {r: 15 for r in GRID}
        t.update({lo: 17, hi: 17})
        assert G.search_resolution(stub(t))['chosen_r'] == lo, (lo, hi)


def test_search_bracket_midpoint_tie_goes_to_the_smaller_pair_after_rounding():
    # brackets only (0.3, 0.4) mid 0.35 and (1.6, 1.7) mid 1.65: both 0.65 from 1.0 (floats: the larger would win);
    # the rule's tie goes to the smaller midpoint, so the first bisection point is 0.35
    f = stub({}, default=lambda r: 19 if 0.35 < r < 1.65 else 15)
    rec = G.search_resolution(f)
    assert [e['r'] for e in rec['evaluated']][30] == pytest.approx(0.35)
    f2 = stub({}, default=lambda r: 19 if 0.35 < r < 1.65 else 15)
    # the tie is between exactly these two pairs: the other pairs (e.g. (0.2, 0.3)) are not brackets
    assert G.search_resolution(f2)['stage'] == 'nearest'


def test_search_nearest_count_tie_is_decided_after_rounding_the_distance_from_one():
    table = {r: 25 for r in GRID}
    table.update({0.6: 19, 1.4: 19})                  # no bracket; |K - 17| = 2 at both; 0.4 from 1.0 at both
    rec = G.search_resolution(stub(table))
    assert rec['stage'] == 'nearest' and rec['chosen_r'] == 0.6 and rec['K'] == 19


def test_search_bracket_needs_consecutive_grid_points_with_both_k_defined():
    # rule gap 1 (decided): 0.6 and 0.7 are undefined, so 0.5 (K 15) and 0.8 (K 19) are not an adjacent grid pair
    # (they were under the first reading): no bisection at 0.65, the nearest count decides
    table = {r: 15 for r in GRID}
    table.update({0.6: None, 0.7: None, **{r: 19 for r in GRID if r >= 0.8}})
    rec = G.search_resolution(stub(table))
    assert rec['stage'] == 'nearest' and len(rec['evaluated']) == 30
    assert [e['r'] for e in rec['evaluated'] if e['K'] is None] == [0.6, 0.7]


def test_search_bisection_stops_when_k_of_the_midpoint_is_undefined():
    # rule gap 2 (accepted): the only bracket is (1.2, 1.3); K(1.25) is undefined, so bisection stops and step (4) runs
    f = stub({}, default=lambda r: None if abs(r - 1.25) < 1e-9 else (16 if r < 1.25 else 18))
    rec = G.search_resolution(f)
    assert [e['r'] for e in rec['evaluated']][30:] == [1.25] and rec['evaluated'][30]['K'] is None
    assert rec['stage'] == 'nearest' and rec['K'] in (16, 18)


def test_search_nearest_count_when_k_never_brackets():
    table = {r: 20 for r in GRID}
    table.update({0.8: 19, 1.2: 19, 2.0: 19})        # |K-17| = 2 at three points; 0.8 and 1.2 tie, smaller r wins
    rec = G.search_resolution(stub(table))
    assert rec['stage'] == 'nearest' and rec['chosen_r'] == 0.8 and rec['K'] == 19
    assert len(rec['evaluated']) == 30                # no bracket, so no bisection


def test_search_skips_undefined_k_and_records_it():
    table = {r: 15 for r in GRID}
    table[1.0] = None                                  # merge_small raised SystemExit at r = 1.0
    table[0.6] = 17
    rec = G.search_resolution(stub(table))
    assert rec['chosen_r'] == 0.6 and rec['K'] == 17
    skipped = [e for e in rec['evaluated'] if e['K'] is None]
    assert [e['r'] for e in skipped] == [1.0] and skipped[0]['reason']


def test_search_all_undefined_is_dropped_with_reason():
    rec = G.search_resolution(stub({}, default=None))
    assert rec['dropped'] and rec['reason'] and rec['chosen_r'] is None
    assert len(rec['evaluated']) == 30


def test_leiden_labels_equal_r7_run_sweep_leiden_at_resolution_one_seed_42():
    phi, g = planted()
    xp = G.unit(phi)
    graph = G._sweep().knn_union_graph(xp, 20)
    assert np.array_equal(G.leiden_labels(graph, 1.0, 42), G._sweep().leiden(graph, 1.0)[0])


# ---------------------------------------------------------------- M3 / M7: labels, merge_small at 200 rows

def planted(n_clusters=6, per=120, rows_each=3, dim=16, seed=0, sep=8.0):
    """Paintings in well-separated clusters; every painting has rows_each rows. Returns phi_p, g (row -> painting)."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(n_clusters, dim)) * sep
    phi = np.concatenate([centres[c] + rng.normal(size=(per, dim)) for c in range(n_clusters)]).astype(np.float32)
    g = np.repeat(np.arange(len(phi)), rows_each).astype(np.int64)
    return phi, g


def check_labels(local, g):
    p = local[np.unique(g, return_index=True)[1]]
    assert np.array_equal(p[g], local), 'labels differ within a painting'
    assert np.array_equal(np.unique(local), np.arange(local.max() + 1)), 'labels not compact'


def test_group_space_record_has_every_evaluated_pair_and_valid_labels():
    phi, g = planted()
    content = np.arange(len(phi)) % 5
    local, rec = G.group_space(phi, g, content, target=6, min_rows=200)
    check_labels(local, g)
    assert np.bincount(local).min() >= 200
    assert rec['K'] == len(np.unique(local)) == rec['n_groups']
    assert rec['evaluated'] and all({'r', 'K'} <= set(e) for e in rec['evaluated'])
    assert (rec['chosen_r'], rec['K']) in pairs(rec)
    assert rec['seed'] == 42 and set(rec['stability_seeds']) == {'43', '44'}
    assert rec['merge_min_rows'] == 200 and rec['graph']['k'] == 20
    assert not rec['dropped']
    # sizes are in paintings and rows and agree with the labels
    assert sum(rec['sizes_rows']) == len(g) and sum(rec['sizes_paintings']) == len(phi)


def test_group_space_with_one_group_is_dropped():
    phi, g = planted(n_clusters=1, per=150)
    local, rec = G.group_space(phi, g, np.zeros(len(phi), dtype=np.int64), target=17, min_rows=200)
    assert local is None and rec['dropped'] and 'fewer than 2 groups' in rec['reason']


def test_merge_small_counts_rows_not_paintings():
    # 4 groups of paintings; the small group has 150 paintings: 2 rows each = 300 rows (kept) but 1 row each = 150 (merged)
    rng = np.random.default_rng(1)
    centres = np.eye(4, 8) * 10
    sizes = [300, 300, 300, 150]
    phi = np.concatenate([centres[c] + 0.1 * rng.normal(size=(s, 8)) for c, s in enumerate(sizes)]).astype(np.float32)
    labels_p = np.repeat(np.arange(4), sizes)
    content = np.zeros(len(phi), dtype=np.int64)
    for rows_small, expect in ((2, 4), (1, 3)):
        g = np.concatenate([np.repeat(np.arange(900), 1), np.repeat(np.arange(900, 1050), rows_small)]).astype(np.int64)
        local, rec = G.native_group(labels_p, phi, g, content)
        assert rec['n_groups'] == expect, rows_small
        check_labels(local, g)
        assert np.bincount(local).min() >= 200
    # the merged group's rows went to the nearest centroid (group 3 sits on axis 3: all others are equidistant -> the first)
    assert rec['merge_moves'] and rec['merge_moves'][0]['rows'] == 150


def test_native_group_compacts_empty_clusters_and_records_raw_sizes():
    phi, g = planted(n_clusters=3, per=150, rows_each=2)
    raw = np.repeat(np.array([4, 9, 20]), 150)        # gaps: labels 0..3, 5..8, ... are empty
    local, rec = G.native_group(raw, phi, g, np.zeros(len(phi), dtype=np.int64))
    check_labels(local, g)
    assert rec['n_groups'] == 3 and rec['raw_n_clusters'] == 3
    assert np.array_equal(local, np.repeat(np.arange(3), 300))


def test_native_group_merge_failure_is_a_drop_with_reason():
    phi, g = planted(n_clusters=3, per=50, rows_each=1)       # 150 rows in total: no group reaches 200 rows
    local, rec = G.native_group(np.repeat(np.arange(3), 50), phi, g, np.zeros(len(phi), dtype=np.int64))
    assert local is None and rec['dropped'] and rec['reason']


def test_compact_removes_gaps_and_keeps_order_of_first_value():
    out = G.compact(np.array([7, 7, 2, 9, 2]))
    assert out.dtype == np.int64 and out.tolist() == [1, 1, 0, 2, 0]


# ---------------------------------------------------------------- R0own

def test_r0own_equals_r0_frozen_labels():
    import r8_common as C
    ctx = C.ctx()
    labels, rec = G.r0_frozen_labels(ctx.scorer_train, ctx.g_st)
    with np.load(C.path_of('20261116_grouping_step1_style/results/step1_group_style.npz'), allow_pickle=False) as z:
        assert np.array_equal(labels, z['style_csd'])
    with np.load(C.path_of('20261126_design_l_style_source/results/groupings/r7_groupings.npz'),
                 allow_pickle=False) as z:
        assert np.array_equal(labels, z['R0'])
    assert len(np.unique(labels)) == 17 == rec['n_groups'] and not rec['dropped']
    assert labels.dtype == np.int64 and labels.shape == ctx.scorer_train.shape


# ---------------------------------------------------------------- M8 NRK

def test_nrk_keeps_clustering_with_lower_row_level_ami_with_content_ties_to_first():
    n = 400
    content = np.repeat(np.arange(8), 50)
    g = np.repeat(np.arange(n), 2)
    same, rnd = content.copy(), np.random.default_rng(0).integers(0, 8, n)
    assert G.nrk_choose(np.stack([same, rnd], 1), content, g)[0] == 1
    assert G.nrk_choose(np.stack([rnd, same], 1), content, g)[0] == 0
    assert G.nrk_choose(np.stack([same, same], 1), content, g)[0] == 0        # tie -> first
    # the choice is made over rows: a painting with many rows weighs more
    rows_g = np.concatenate([np.arange(n - 1), np.full(500, n - 1)])
    a, b = same.copy(), same.copy()
    a[:3], b[-1] = (a[:3] + 1) % 8, (b[-1] + 1) % 8                           # a: 3 wrong paintings (3 rows), b: 1 wrong painting (500 rows)
    c_rows = G.nrk_choose(np.stack([a, b], 1), content, rows_g)
    c_paint = G.nrk_choose(np.stack([a, b], 1), content, np.arange(n))
    assert c_paint[0] == 0 and c_rows[0] == 1


def test_nrk_space_is_unit_projection_on_the_kept_subspace():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(30, 6)).astype(np.float32)
    V = np.linalg.qr(rng.normal(size=(6, 6)))[0]
    P = [np.array([0, 2, 5]), np.array([1, 3, 4])]
    for s in (0, 1):
        z = G.nrk_space(x, V, P, s)
        raw = x @ V[:, P[s]]
        np.testing.assert_allclose(z, (raw / np.linalg.norm(raw, axis=1, keepdims=True)).astype(np.float32), atol=1e-6)
        assert z.dtype == np.float32 and z.shape == (30, 3)


def test_nrk_child_over_the_time_limit_is_dropped(tmp_path):
    x = np.zeros((20, 4), dtype=np.float32)
    cmd = [sys.executable, '-c', 'import time; time.sleep(60)']
    labels, rec = G.fit_child(x, tmp_path, timeout_seconds=1, command=cmd)
    assert labels is None and rec['dropped'] and 'exceeded' in rec['reason'] and rec['timeout_seconds'] == 1


def test_nrk_child_failure_is_dropped_with_reason(tmp_path):
    x = np.zeros((20, 4), dtype=np.float32)
    labels, rec = G.fit_child(x, tmp_path, timeout_seconds=30, command=[sys.executable, '-c', 'raise SystemExit(3)'])
    assert labels is None and rec['dropped'] and rec['reason']


# ---------------------------------------------------------------- M9 TP

def test_tp_prompts_from_sorted_top_level_folders_with_underscores_as_spaces():
    paintings = {f'id{i}': {'index': i, 'image': f'{s}/id{i}.jpg'} for i, s in
                 enumerate(['Post_Impressionism', 'Baroque', 'Baroque', 'Art_Nouveau_Modern'])}
    names, prompts = G.tp_prompts(paintings, expect=3)
    assert names == ['Art_Nouveau_Modern', 'Baroque', 'Post_Impressionism']
    assert prompts == ['a Art Nouveau Modern painting', 'a Baroque painting', 'a Post Impressionism painting']
    with pytest.raises(AssertionError):
        G.tp_prompts(paintings, expect=27)


def test_tp_argmax_ties_to_lowest_index():
    t = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    x = np.array([[1, 0], [0, 1], [.5, .5]], dtype=np.float32)
    # classes 0 and 1 are identical: x0 -> 0 (tie, lowest), x1 -> 2, x2 ties all three -> 0
    assert G.tp_argmax(x, t).tolist() == [0, 2, 0]


def test_tp_merge_map_applies_to_gallery_images():
    rng = np.random.default_rng(3)
    d = 6
    t = np.eye(d, dtype=np.float32)[:4]                                   # 4 classes
    sizes = [300, 300, 300, 150]                                           # class 3 has 150 rows (1 row each) -> merged
    raw = np.repeat(np.arange(4), sizes)
    x = (t[raw] + 0.05 * rng.normal(size=(len(raw), d))).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    g = np.arange(len(raw), dtype=np.int64)
    local, rec, mmap = G.tp_group(x, t, g)
    assert rec['n_groups'] == 3 and mmap.shape == (4,) and np.bincount(local).min() >= 200
    check_labels(local, g)
    # the gallery goes through the same map: images near each class axis get the merged label of that class's paintings
    gallery = (t[[0, 1, 2, 3, 3]] + 0.01).astype(np.float32)
    gallery /= np.linalg.norm(gallery, axis=1, keepdims=True)
    out, n_fb = G.tp_apply(gallery, t, mmap)
    assert n_fb == 0
    assert out.tolist() == [local[0], local[300], local[600], local[900], local[900]]
    assert out[3] == out[4] != -1 and out[3] in (out[0], out[1], out[2])    # the small class sits inside a big group
    assert rec['merge_map'] == mmap.tolist()


def test_tp_apply_class_without_paintings_needs_centres_then_takes_nearest_centre():
    mmap = np.array([0, -1, 1])
    t = np.eye(3, dtype=np.float32)
    x = np.array([[0, 1, 0], [1, 0, 0]], dtype=np.float32)          # image 0 has class 1 (no paintings), image 1 class 0
    with pytest.raises(AssertionError):
        G.tp_apply(x, t, mmap)
    centres = np.array([[0.6, 0.8, 0], [0, 0, 1.0]])
    out, n_fb = G.tp_apply(x, t, mmap, centres)
    assert out.tolist() == [0, 0] and n_fb == 1


# ---------------------------------------------------------------- M13 random controls

def test_random_control_labels_use_default_rng_7100_plus_i_and_x_sizes():
    rng = np.random.default_rng(5)
    labels_p = rng.integers(0, 5, 1000)
    labels_p = np.unique(labels_p, return_inverse=True)[1].astype(np.int64)
    g = np.repeat(np.arange(1000), 3).astype(np.int64)
    sizes = np.bincount(labels_p)
    for i in (1, 7, 16):
        z = G.random_control_labels(labels_p[g], g, i)
        # M13 / R7 L8 written out: permute the painting indices with default_rng(7100 + i), consecutive blocks
        perm = np.random.default_rng(7100 + i).permutation(1000)
        expect = np.empty(1000, dtype=np.int64)
        expect[perm] = np.repeat(np.arange(len(sizes)), sizes)
        assert np.array_equal(z, expect[g])
        assert np.array_equal(np.bincount(z[np.unique(g, return_index=True)[1]]), sizes)
        assert np.array_equal(z, G.random_control_labels(labels_p[g], g, i))            # seed-stable
    assert not np.array_equal(G.random_control_labels(labels_p[g], g, 1), G.random_control_labels(labels_p[g], g, 2))


def test_random_control_matches_r7_function_with_the_shifted_seed():
    import r8_common as C
    r7 = C.load_r7('r7_groupings')
    labels_p = np.repeat(np.arange(4), [200, 300, 250, 250]).astype(np.int64)
    g = np.repeat(np.arange(1000), 2).astype(np.int64)
    for i in (3, 12):
        assert np.array_equal(G.random_control_labels(labels_p[g], g, i), r7.random_control(labels_p[g], g, 100 + i))


def test_control_draws_exact_calls_and_seeds():
    labels_p = np.repeat(np.arange(3), [5, 3, 2]).astype(np.int64)
    i = 7
    train, sel = G.control_draws(labels_p, i, n_sel=6)
    shares = np.bincount(labels_p, minlength=3) / 10
    assert shares.dtype == np.float64
    assert np.array_equal(train, np.random.default_rng(7207).choice(3, size=10, p=shares))
    assert np.array_equal(sel, np.random.default_rng(7307).choice(3, size=6, p=shares))
    # pinned literals (numpy 2.2.6): the draws cannot drift with a changed call order or seed
    assert train.tolist() == [0, 2, 0, 0, 0, 2, 2, 0, 2, 2] and sel.tolist() == [2, 2, 0, 0, 1, 2]
    assert G.random_control_labels(labels_p, np.arange(10), 7).tolist() == [2, 1, 0, 2, 0, 1, 0, 0, 0, 1]
    t2, s2 = G.control_draws(labels_p, i, n_sel=6)
    assert np.array_equal(t2, train) and np.array_equal(s2, sel)


# ---------------------------------------------------------------- M16: stage and D21 records

def write_rec(path, **kw):
    """A stage record as ticket 04 / 06 write it: pass, smoke, rule_sha256."""
    import json
    import r8_common as C
    rec = {'pass': True, 'smoke': False, 'rule_sha256': C.sha_file(C.RULE), **kw}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({k: v for k, v in rec.items() if v is not Ellipsis}))
    return path


def test_fc_drops_from_stage1_and_d21_records(tmp_path, monkeypatch):
    import r8_common as C
    monkeypatch.setattr(C, 'RESULTS', tmp_path)
    # FC3 (stage 1) failed SC2: every FC goes, with that record's reason
    write_rec(tmp_path / 'stage1_fcmi.json', **{'pass': False, 'tests': {'process': True, 'finite_loss': False},
                                                 'error': 'non-finite total loss'})
    for name in G.FC_IDS:
        lab, rec, n = G.part_fc(name, smoke=False)
        assert lab is None and rec['dropped'] and 'finite_loss' in rec['reason'] and 'non-finite' in rec['reason']
    # stage 1 passed; FC2 failed D21 and published no space file: only FC2 drops, reason from its d21.json
    (tmp_path / 'stage1_fcmi.json').unlink()
    write_rec(tmp_path / 'stage1_fcmi.json')
    write_rec(tmp_path / 'fcmi' / 'FC2' / 'd21.json', **{'pass': False, 'tests': {'wall_time': False},
                                                         'error': 'construction-to-labels exceeded 60 minutes'})
    lab, rec, n = G.part_fc('FC2', smoke=False)
    assert lab is None and rec['dropped'] and 'wall_time' in rec['reason'] and 'FC2' in rec['reason']
    # a stray space file does not rescue a failed D21 (m-2): the record is read whether or not the space is there
    (tmp_path / 'spaces').mkdir()
    (tmp_path / 'spaces' / 'FC2.npz').write_bytes(b'stray')
    lab, rec, n = G.part_fc('FC2', smoke=False)
    assert lab is None and rec['dropped'] and 'wall_time' in rec['reason']
    # no D21 record: the call stops instead of guessing; D21 passed but no space: stops
    with pytest.raises(AssertionError, match='is missing'):
        G.part_fc('FC4', smoke=False)
    write_rec(tmp_path / 'fcmi' / 'FC4' / 'd21.json')
    with pytest.raises(AssertionError, match='space file'):
        G.part_fc('FC4', smoke=False)


def test_fc_stage1_record_needs_bool_pass_not_smoke_and_the_rules_sha(tmp_path, monkeypatch):
    import r8_common as C
    monkeypatch.setattr(C, 'RESULTS', tmp_path)
    p = tmp_path / 'stage1_fcmi.json'
    with pytest.raises(AssertionError, match='is missing'):
        G.part_fc('FC1', smoke=False)
    write_rec(p, **{'pass': Ellipsis, 'passed': True})                    # the removed `passed` key is not read
    with pytest.raises(AssertionError, match='is not a bool'):
        G.part_fc('FC1', smoke=False)
    p.unlink()
    write_rec(p, **{'pass': None})
    with pytest.raises(AssertionError, match='is not a bool'):
        G.part_fc('FC1', smoke=False)
    p.unlink()
    write_rec(p, smoke=True)
    with pytest.raises(AssertionError, match='smoke is not false'):
        G.part_fc('FC1', smoke=False)
    p.unlink()
    write_rec(p, rule_sha256='0' * 64)
    with pytest.raises(AssertionError, match='rule SHA-256 differs'):
        G.part_fc('FC1', smoke=False)


def test_encoder_stage_record_missing_raises_false_drops(tmp_path, monkeypatch):
    import r8_common as C
    monkeypatch.setattr(C, 'RESULTS', tmp_path)
    p = tmp_path / 'stageE_AL.json'
    with pytest.raises(AssertionError, match='is missing'):
        G.part_search('AL', smoke=False, n=None)               # no record: never a silent, write-once drop
    write_rec(p, **{'pass': False, 'reason': 'weight load failed',
                    'esc1': {'a_strict_load': {'ok': False}, 'c_finite': {'ok': True}, 'e_repeat': {'ok': False}}})
    lab, rec, n = G.part_search('AL', smoke=False, n=None)
    assert lab is None and rec['dropped'] and 'weight load failed' in rec['reason']
    assert 'a_strict_load' in rec['reason'] and 'e_repeat' in rec['reason'] and 'c_finite' not in rec['reason']
    for bad, match in (({'pass': None}, 'is not a bool'), ({'pass': 'true'}, 'is not a bool'),
                       ({'smoke': True}, 'smoke is not false'), ({'rule_sha256': 'f' * 64}, 'rule SHA-256 differs')):
        p.unlink()
        write_rec(p, **bad)
        with pytest.raises(AssertionError, match=match):
            G.part_search('AL', smoke=False, n=None)


def test_views_meta_check_failure_drops_v_spaces_with_the_cosines(tmp_path, monkeypatch):
    import json
    import r8_common as C
    monkeypatch.setattr(C, 'RESULTS', tmp_path)
    meta = tmp_path / 'meta.json'
    monkeypatch.setattr(G, 'VIEWS_META', meta)
    cosines = {'passed': False, 'mean_cosine': 0.123, 'min_cosine': 0.0456}
    meta.write_text(json.dumps({'painting_ids': [], 'views_sha256': 'x', 'view_check': cosines}))
    for name in ('V100', 'V67', 'V33'):
        lab, rec, n = G.part_search(name, smoke=False, n=None)
        assert lab is None and rec['dropped'] and rec['reason'].startswith('M4 view check failed')
        assert '0.123' in rec['reason'] and '0.0456' in rec['reason'] and rec['view_check'] == cosines
    # a passing check requires the space file (ticket 05 publishes it)
    meta.write_text(json.dumps({'painting_ids': [], 'views_sha256': 'x', 'view_check': {**cosines, 'passed': True}}))
    with pytest.raises(AssertionError, match='space file'):
        G.part_search('V100', smoke=False, n=None)
    # keys ticket 05 asserts, and a bool `passed`
    meta.write_text(json.dumps({'painting_ids': [], 'view_check': {'passed': True}}))
    with pytest.raises(AssertionError, match='views meta keys'):
        G.part_search('V100', smoke=False, n=None)
    meta.write_text(json.dumps({'painting_ids': [], 'views_sha256': 'x', 'view_check': {'passed': 1}}))
    with pytest.raises(AssertionError, match='is not a bool'):
        G.part_search('V100', smoke=False, n=None)


def test_nbc_part_needs_a_triggered_probe_record(tmp_path, monkeypatch):
    import r8_common as C
    monkeypatch.setattr(C, 'RESULTS', tmp_path)
    with pytest.raises(AssertionError, match='probe_trigger.json is missing'):
        G.part_search('NBC', smoke=False, n=None)
    p = write_rec(tmp_path / 'probe_trigger.json', triggered=False, strength=None)
    with pytest.raises(AssertionError, match='triggered is not true'):
        G.part_search('NBC', smoke=False, n=None)
    p.unlink()
    write_rec(p, triggered=True, strength=0.67)
    with pytest.raises(AssertionError, match='space file'):          # passes the trigger, then needs NBC's space
        G.part_search('NBC', smoke=False, n=None)


# ---------------------------------------------------------------- encoder files, painting join (C6)

def fake_encoder(tmp_path, order='ok', dim=8):
    """An encoder directory as ticket 04 writes it (real 49,121 image order, small dim)."""
    import json
    import r8_common as C
    d = tmp_path / 'enc' / 'r8_enc_aladin'
    d.mkdir(parents=True)
    nh = C.nonheld_images().astype(np.int64)
    emb = np.random.default_rng(0).normal(size=(len(nh), dim)).astype(np.float32)
    stored = nh if order == 'ok' else nh[::-1].copy()
    np.save(d / 'embeddings.npy', emb)
    np.save(d / 'image_index.npy', stored)
    meta = {'image_index_file': str(d / 'image_index.npy'), 'image_index_sha256': G.sha_array(nh),
            'embeddings_sha256': C.sha_file(d / 'embeddings.npy'), 'n': len(nh), 'smoke': False}
    (d / 'meta.json').write_text(json.dumps(meta))
    return tmp_path / 'enc', d, meta


def test_encoder_space_asserts_image_order_and_both_sha256(tmp_path):
    import json
    import r8_common as C
    root, d, meta = fake_encoder(tmp_path)
    space = G.encoder_space('AL', False, tmp_path / 'spaces', enc_root=root)
    with np.load(space) as z:
        assert np.array_equal(z['image_index'], C.nonheld_images()) and z['phi'].shape == (49121, 8)
        assert np.allclose(np.linalg.norm(z['phi'], axis=1), 1, atol=1e-5)
    root2, d2, _ = fake_encoder(tmp_path / 'rev', order='reversed')                   # other image order
    with pytest.raises(AssertionError, match='another image order'):
        G.encoder_space('AL', False, tmp_path / 'spaces2', enc_root=root2)
    root3, d3, m3 = fake_encoder(tmp_path / 'badimg')                                  # meta image SHA differs
    (d3 / 'meta.json').write_text(json.dumps({**m3, 'image_index_sha256': '0' * 64}))
    with pytest.raises(AssertionError, match='image_index SHA-256 differs'):
        G.encoder_space('AL', False, tmp_path / 'spaces3', enc_root=root3)
    root4, d4, m4 = fake_encoder(tmp_path / 'bademb')                                  # meta embeddings SHA differs
    (d4 / 'meta.json').write_text(json.dumps({**m4, 'embeddings_sha256': '0' * 64}))
    with pytest.raises(AssertionError, match='embeddings SHA-256 differs'):
        G.encoder_space('AL', False, tmp_path / 'spaces4', enc_root=root4)


def test_painting_vectors_joins_paintings_to_their_image_rows(monkeypatch):
    import r8_common as C
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(1000, size=200, replace=False)).astype(np.int32)         # sorted superset of the images
    phi = rng.normal(size=(200, 5)).astype(np.float32)
    img = rng.permutation(idx)[:40].astype(np.int32)                                   # non-monotone painting -> image
    assert not np.all(np.diff(img) > 0)
    monkeypatch.setattr(C, 'painting_image', lambda: img)
    out = G.painting_vectors(phi, idx, 25)
    assert out.shape == (25, 5)
    for p in range(25):
        assert np.array_equal(out[p], phi[np.flatnonzero(idx == img[p])[0]])
    assert not np.array_equal(out, phi[:25])
    # a painting whose image is not in the space file stops the call
    idx2 = np.delete(idx, np.flatnonzero(idx == img[3]))
    with pytest.raises(AssertionError, match='not in the space file'):
        G.painting_vectors(phi[:len(idx2)], idx2, 25)
    G.painting_vectors(phi[:len(idx2)], idx2, 3)                                       # paintings 0..2 are fine


def test_assert_same_scorer_train_names_the_part():
    st = np.arange(10, dtype=np.int64)
    parts = {'V100': {'record': {'scorer_train_sha256': G.sha_array(st)}},
             'AL': {'record': {'scorer_train_sha256': G.sha_array(st[::-1])}}}
    G.assert_same_scorer_train({'V100': parts['V100']}, st)
    with pytest.raises(AssertionError, match='AL'):
        G.assert_same_scorer_train(parts, st)
    with pytest.raises(AssertionError, match='AL'):
        G.assert_same_scorer_train({'AL': {'record': {}}}, st)


def test_tp_apply_fallback_only_with_centres_and_real_runs_pass_none():
    # rule gap 4 (decided): the real run calls tp_apply without centres, so an image of a class without paintings stops it
    mmap = np.array([0, -1, 1])
    t = np.eye(3, dtype=np.float32)
    x = np.array([[0, 1, 0]], dtype=np.float32)
    with pytest.raises(AssertionError, match='without scorer-train'):
        G.tp_apply(x, t, mmap)


# ---------------------------------------------------------------- M16 drops, collect

def part(labels, **kw):
    return {'labels': labels, 'record': {'dropped': labels is None, 'reason': kw.get('reason'), **kw.get('extra', {})}}


def test_assemble_adds_controls_per_candidate_and_records_drops_with_reasons():
    n_rows = 600
    g = np.repeat(np.arange(300), 2).astype(np.int64)
    st = np.arange(n_rows, dtype=np.int64)
    parts = {
        'V100': part(np.repeat(np.arange(3), 200)),
        'FC1': part(None, reason='SC2 failed: D21 test'),
        'AL': part(np.repeat(np.arange(2), 300)),
        'R0own': part(np.repeat(np.arange(2), 300)),
        'NRK': part(None, reason='NrKmeans exceeded 1800 seconds'),
        'TP': part(np.repeat(np.arange(5), 120)),
    }
    arrays, rec = G.assemble(parts, st, g, ids=list(parts))
    assert {'scorer_train', 'g_st'} <= set(arrays)
    assert {'V100', 'ZV100', 'AL', 'ZAL', 'R0own', 'TP'} <= set(arrays)
    assert 'ZR0own' not in arrays and 'ZTP' not in arrays and 'ZFC1' not in arrays and 'FC1' not in arrays
    assert set(rec['groupings']) == {'V100', 'ZV100', 'AL', 'ZAL', 'R0own', 'TP'}
    assert set(rec['dropped']) == {'FC1', 'ZFC1', 'NRK'}
    assert all(isinstance(r, str) and r for r in rec['dropped'].values())
    assert 'FC1' in rec['dropped']['ZFC1']
    for k in ('V100', 'ZV100', 'AL', 'ZAL', 'R0own', 'TP'):
        check_labels(arrays[k], g)
        assert arrays[k].dtype == np.int64
    # M13 index and seed: V100 is 1, AL is 14
    assert rec['groupings']['ZV100']['seed'] == 7101 and rec['groupings']['ZAL']['seed'] == 7114
    assert np.array_equal(np.bincount(arrays['ZAL']), np.bincount(arrays['AL']))


def test_assemble_refuses_a_part_with_one_group():
    g = np.arange(100, dtype=np.int64)
    with pytest.raises(AssertionError):
        G.assemble({'V100': part(np.zeros(100, dtype=np.int64))}, np.arange(100, dtype=np.int64), g, ids=['V100'])
