"""Unit tests (tiny arrays) of the qualitative-figure rules."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qf_cand as Q  # noqa: E402


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


# ---------------------------------------------------------------- panel (a)
def test_select_a_filter_rank_cap():
    keys = ['k0', 'k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'k7']
    emo = np.array([0, 0, 0, 1, 1, 1, 2, 2])
    e_ours = np.array([5, 4, 5, 5, 3, 4, 5, 5])     # k4 fails e_ours
    e_plain = np.array([0, 1, 2, 0, 0, 1, 0, 0])     # k2 fails e_plain
    g_plain = np.array([3, 5, 5, 3, 5, 2, 3, 5])     # k5 fails g_plain
    g_ours = np.array([2, 0, 0, 4, 0, 0, 1, 1])
    kept, ranked = Q.select_a(keys, emo, e_ours, e_plain, g_plain, g_ours, total=18)
    # passing: k0 (5), k1 (3), k3 (5), k6 (5), k7 (5); ranked by diff desc, g_ours asc, key asc
    assert [keys[i] for i in ranked] == ['k6', 'k7', 'k0', 'k3', 'k1']       # g_ours 1, 1, 2, 4 among diff 5; then k1
    assert [keys[i] for i in kept] == ['k6', 'k7', 'k0', 'k3', 'k1']
    # cap of 1 per class and total of 2
    kept, _ = Q.select_a(keys, emo, e_ours, e_plain, g_plain, g_ours, per_class=1, total=2)
    assert [keys[i] for i in kept] == ['k6', 'k0']
    kept, _ = Q.select_a(keys, emo, e_ours, e_plain, g_plain, g_ours, per_class=1, total=18)
    assert [keys[i] for i in kept] == ['k6', 'k0', 'k3']        # k7 (class 2) and k1 (class 0) capped


def test_select_a_cap_two_per_class_skips_later():
    n = 6
    keys = [f'q{i}' for i in range(n)]
    emo = np.zeros(n, dtype=int)
    kept, ranked = Q.select_a(keys, emo, [5] * n, [0] * n, [3] * n, [0] * n)
    assert len(ranked) == n and [keys[i] for i in kept] == ['q0', 'q1']


def test_build_a_counts_on_tiny_set():
    # 12 paintings; plain space and emotion head chosen so that the query's top 5 differ
    class S:  # minimal Set stand-in
        pass
    s = S()
    n = 12
    s.keys = [f'p{i:02d}' for i in range(n)]
    s.class_names = dict(emotion=['e0', 'e1'], style=['s0', 's1'], genre=['g0', 'g1'])
    s.labels = dict(emotion=np.array([0] * 6 + [1] * 6), style=np.zeros(n, dtype=int),
                    genre=np.array([0] * 6 + [0] * 3 + [1] * 3))
    rng = np.random.default_rng(0)
    plain = unit(rng.normal(size=(n, 8)))
    emo = unit(np.eye(2)[s.labels['emotion']] + 0.01 * rng.normal(size=(n, 2)))
    use = np.ones(n, dtype=bool)
    use[11] = False                                   # not usable: never a neighbour, never a query
    cands, info = Q.build_a(s, plain, emo, use)
    assert info['queries'] == 11 and info['gallery'] == 11
    assert all('p11' not in [x['key'] for x in c['ours'] + c['plain']] + [c['query']] for c in cands)
    for c in cands:
        assert len(c['ours']) == 5 and len(c['plain']) == 5
        assert c['counts']['e_ours'] >= 4 and c['counts']['e_plain'] <= 1 and c['counts']['g_plain'] >= 3


# ---------------------------------------------------------------- panel (b)
def test_ranks_of_percentile_and_exclusion():
    # x on a line: the query 0 at angle 0, others at growing angles; row 1 is a near copy (plain) of row 0
    ang = np.array([0, 1, 10, 20, 30, 40, 50, 60]) * np.pi / 180
    x = unit(np.stack([np.cos(ang), np.sin(ang)], 1))
    plain = x.copy()                                  # cos(0,1) = 0.9998 >= 0.95: near copy; cos(0,2) = 0.985 too
    plain = unit(np.stack([np.cos(ang * 3), np.sin(ang * 3), np.zeros(8)], 1))   # 3x angles: row 1 at 3 deg -> .9986
    g = np.arange(8)
    rank, gsz, cos = Q.ranks_of([0], [5], x, plain, g)
    # excluded for query 0: itself (row 0) and row 1 (cos .9986), row 2 at 30 deg -> .866 not excluded
    assert gsz[0] == 6
    assert rank[0] == 4                                # rows 2, 3, 4 are nearer than row 5
    assert abs(cos[0] - np.cos(np.radians(40))) < 1e-5
    # chunk size does not change the result
    rank2, gsz2, _ = Q.ranks_of([0, 3, 0], [5, 7, 2], x, plain, g, chunk=1)
    assert rank2[0] == 4 and gsz2[0] == 6 and rank2[2] == 1


def mk_pair_world():
    """16 paintings in 2-d spaces. A = angle, B = a different angle; labels set to give known candidates."""
    n = 16
    rng = np.random.default_rng(1)
    plain = unit(rng.normal(size=(n, 12)))
    xa = unit(rng.normal(size=(n, 4)))
    xb = unit(rng.normal(size=(n, 4)))
    laba = rng.integers(0, 2, n)
    labb = rng.integers(0, 3, n)
    return n, plain, xa, xb, laba, labb


def test_pair_candidates_rule_and_percentile_brute_force():
    n, plain, xa, xb, laba, labb = mk_pair_world()
    rows = np.arange(n)
    cands, nP = Q.pair_candidates(xa, xb, plain, laba, labb, rows)
    assert nP == n and cands
    for c in cands:
        i, j = c['i'], c['j']
        assert laba[i] == laba[j] and labb[i] != labb[j]
        # brute force: i's list in A over the gallery without i and its near copies
        ok = [g for g in rows if g != i and float(plain[i] @ plain[g]) < Q.NEAR]
        order_a = sorted(ok, key=lambda g: (-float(xa[i] @ xa[g]), g))
        assert j in order_a[:5] and c['a_rank'] == order_a.index(j) + 1
        order_b = sorted(ok, key=lambda g: (-float(xb[i] @ xb[g]), g))
        assert c['b_rank'] == order_b.index(j) + 1 and c['gallery'] == len(ok)
        assert abs(c['b_percentile'] - c['b_rank'] / len(ok)) < 1e-12
        order_p = sorted(ok, key=lambda g: (-float(plain[i] @ plain[g]), g))
        assert c['plain_rank'] == order_p.index(j) + 1
    # every (i, top-5 j) with the label condition appears
    want = 0
    for i in rows:
        ok = [g for g in rows if g != i and float(plain[i] @ plain[g]) < Q.NEAR]
        for j in sorted(ok, key=lambda g: (-float(xa[i] @ xa[g]), g))[:5]:
            want += int(laba[j] == laba[i] and labb[j] != labb[i])
    assert len(cands) == want


def test_rank_dedup_and_order():
    keys = ['a', 'b', 'c', 'd']
    mk = lambda i, j, p, ac: dict(i=i, j=j, b_percentile=p, a_cos=ac)  # noqa: E731
    cands = [mk(0, 1, 0.5, 0.9), mk(1, 0, 0.9, 0.8), mk(2, 3, 0.9, 0.95), mk(3, 2, 0.1, 0.99), mk(0, 2, 0.5, 0.91)]
    out = Q.rank_dedup(cands, keys)
    # order: (2,3) pct .9 cos .95; (1,0) pct .9 cos .8; (0,2) pct .5 cos .91; (0,1) dup of (1,0); (3,2) dup of (2,3)
    assert [(c['i'], c['j']) for c in out] == [(2, 3), (1, 0), (0, 2)]
    # ties on percentile and cos: key order
    t = Q.rank_dedup([mk(1, 2, .5, .5), mk(0, 3, .5, .5)], keys)
    assert [(c['i'], c['j']) for c in t] == [(0, 3), (1, 2)]


def test_pick_b_no_reuse_and_cap():
    mk = lambda i, j: dict(i=i, j=j)  # noqa: E731
    l1 = [mk(0, 1), mk(2, 3), mk(4, 5), mk(6, 7)]
    l2 = [mk(1, 8), mk(9, 10), mk(11, 12), mk(13, 14)]     # (1, 8) reuses painting 1: skipped
    l3 = [mk(2, 20)]                                       # reuses 2
    out = Q.pick_b([l1, l2, l3], per_pair=3)
    assert [(c['i'], c['j']) for c in out[0]] == [(0, 1), (2, 3), (4, 5)]
    assert [(c['i'], c['j']) for c in out[1]] == [(9, 10), (11, 12), (13, 14)]
    assert out[2] == []
    used = [r for lst in out for c in lst for r in (c['i'], c['j'])]
    assert len(used) == len(set(used))


def test_build_b_ids_and_no_reuse():
    n, plain, xa, xb, laba, labb = mk_pair_world()

    class S:
        pass
    s = S()
    s.keys = [f'k{i:02d}' for i in range(n)]
    s.class_names = dict(emotion=['e0', 'e1', 'e2'], style=['s0', 's1', 's2'], genre=['g0', 'g1', 'g2'])
    s.labels = dict(emotion=laba, style=labb, genre=np.where(np.arange(n) % 5 == 0, -1, laba))
    heads = dict(emotion=xa, style=xb, genre=unit(np.random.default_rng(5).normal(size=(n, 4))))
    use = np.ones(n, dtype=bool)
    out, info = Q.build_b(s, plain, heads, use)
    assert [c['id'] for c in out] == [f'b{i + 1}' for i in range(len(out))]
    keys = [k for c in out for k in (c['key_i'], c['key_j'])]
    assert len(keys) == len(set(keys))
    assert set(info['pairs']) == {f'{a}/{b}' for a, b in Q.PAIRS}
    for c in out:
        assert c['labels_i'][c['A']] == c['labels_j'][c['A']] and c['labels_i'][c['B']] != c['labels_j'][c['B']]
        assert 1 <= c['a_rank'] <= 5 and 0 < c['b_percentile'] <= 1


# ---------------------------------------------------------------- teaser exclusion and clusters
def test_teaser_exclusion():
    keys = ['x', Q.TEASER_KEY, 'y', 'z']
    pd = np.array([True, True, False, True])
    assert Q.usable_mask(keys, pd).tolist() == [True, False, False, True]


def test_teaser_never_a_neighbour_or_query():
    n = 14
    rng = np.random.default_rng(2)
    plain = unit(rng.normal(size=(n, 10)))
    x = unit(rng.normal(size=(n, 3)))
    keys = [f'k{i}' for i in range(n)]
    keys[4] = Q.TEASER_KEY
    use = Q.usable_mask(keys, np.ones(n, dtype=bool))
    rows = np.flatnonzero(use)
    idx, _ = Q.RC.topk_neighbours(rows, rows, dict(a=x), plain, n=5)
    assert 4 not in idx['a'] and 4 not in rows


def test_cluster_gain_sort():
    mk = lambda c, sh, size: dict(**{'class': c}, class_share=sh, size=size, cluster=1, labelled=size,  # noqa: E731
                                  paintings=[f'{c}{i}' for i in range(6)])
    cs = [mk('a', 0.5, 10), mk('b', 0.9, 20), mk('c', 0.3, 30)]
    pl = [mk('c', 0.1, 5), mk('a', 0.45, 6), mk('b', 0.1, 7)]      # other order: joined by class
    out = Q.cluster_gain(cs, pl)
    assert [r['class'] for r in out] == ['b', 'c', 'a']
    assert [r['rank'] for r in out] == [1, 2, 3]
    assert out[0]['gain'] == 0.8 and out[1]['gain'] == 0.2 and out[2]['gain'] == 0.05
    assert out[0]['cstar']['size'] == 20 and out[0]['plain']['size'] == 7
    assert out[0]['cstar']['in_figure'] == out[0]['cstar']['paintings'][:4] and len(out[0]['cstar']['paintings']) == 6
    # ties keep the order of the c* rows
    t = Q.cluster_gain([mk('x', 0.5, 1), mk('y', 0.5, 1)], [mk('x', 0.2, 1), mk('y', 0.2, 1)])
    assert [r['class'] for r in t] == ['x', 'y']


def test_r20_published_picks_are_first():
    """The rule's pick per aspect on r20's real candidates equals the published 20-cluster figure's classes."""
    import json
    if not Q.R20_CANDIDATES.exists():
        return
    doc = json.loads(Q.R20_CANDIDATES.read_text())
    cl = Q.build_clusters(doc)
    assert cl['emotion'][0]['id'] == 'c-emotion-1'
    assert [len(cl[a]) for a in Q.ASPECTS] == [9, 27, 10]
