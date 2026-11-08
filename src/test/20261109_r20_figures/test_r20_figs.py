"""Fast tests of the r20 figure rules on synthetic data (no real data, no network)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import r20_cand as C  # noqa: E402
import r20_data as D  # noqa: E402
import r20_pd as PD  # noqa: E402
import r20_view as V  # noqa: E402


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


# ---------------------------------------------------------------------------------------------- cluster rules
def _toy_clusters():
    # 3 clusters, 3 classes (0 most frequent). Labels -1 are ignored in every share.
    #   cluster 0: 10 rows, labelled 10: class0 x6, class1 x4        -> share(c0) .6
    #   cluster 1: 8 rows,  labelled 6 : class0 x3, class2 x3, 2 unl  -> share(c0) .5, share(c2) .5
    #   cluster 2: 6 rows,  labelled 6 : class1 x5, class2 x1         -> share(c1) .833
    part = np.array([0] * 10 + [1] * 8 + [2] * 6)
    lab = np.array([0] * 6 + [1] * 4 + [0] * 3 + [2] * 3 + [-1] * 2 + [1] * 5 + [2])
    return part, lab


def test_per_class_picks_largest_share_among_labelled():
    part, lab = _toy_clusters()
    st = C.cluster_stats(part, lab, 3, np.ones(len(part), bool))
    order = C.class_order(lab, 3)
    assert order == [0, 1, 2]                                  # class counts 9, 9, 4: tie 0 vs 1 -> lower id first
    got = dict(C.pick_per_class(st, order))
    assert got[0] == 0            # .6 beats .5
    assert got[1] == 2            # .833 beats .4
    assert got[2] == 1            # class 2: cluster 1 has .5 (3 of 6 labelled; the 2 unlabelled rows do not count), cluster 2 has .167


def test_per_class_ties_go_to_the_larger_cluster_then_lower_id():
    part = np.array([0] * 4 + [1] * 6 + [2] * 6)
    lab = np.array([0, 0, 1, 1] + [0, 0, 0, 1, 1, 1] + [0, 0, 0, 1, 1, 1])
    st = C.cluster_stats(part, lab, 2, np.ones(len(part), bool))
    # class 0: shares .5, .5, .5 -> all tied; larger clusters are 1 and 2 (6 rows), lower id wins -> 1
    assert dict(C.pick_per_class(st, [0, 1]))[0] == 1
    assert dict(C.pick_per_class(st, [0, 1]))[1] == 1


def test_per_class_marks_a_cluster_serving_two_classes():
    part, lab = _toy_clusters()
    labels = lab.copy()
    names = ['a', 'b', 'c']
    rows = C.panel_rows('per_class', C.cluster_stats(part, labels, 3, np.ones(len(part), bool)), labels, names, [0, 1, 2])
    by = {r['class']: r for r in rows}
    assert [r['class'] for r in rows] == ['a', 'b', 'c'] and not any(r['dup'] for r in rows)
    # force a duplicate: class c has only cluster 1 as its best, make class a also best at cluster 1
    part2 = part.copy()
    part2[:10] = 1
    st2 = C.cluster_stats(part2, labels, 3, np.ones(len(part), bool))
    rows2 = C.panel_rows('per_class', st2, labels, names, [0, 1, 2])
    ids = [r['cluster'] for r in rows2]
    assert any(r['dup'] for r in rows2) == (len(set(ids)) < len(ids))
    assert by['a']['class_share'] == 0.6 and by['a']['majority'] == 'a' and by['a']['labelled'] == 10


def test_purity_needs_20_labelled_and_ranks_descending():
    rng = np.random.default_rng(1)
    part = np.repeat(np.arange(8), 30)
    lab = np.concatenate([np.where(rng.random(30) < p, 0, 1) for p in (.95, .9, .8, .7, .6, .55, .99, .5)])
    lab[:30][10:] = -1                                         # cluster 0: only 10 labelled -> excluded
    st = C.cluster_stats(part, lab, 2, np.ones(len(part), bool))
    top = C.pick_purity(st)
    assert 0 not in top and len(top) == 6
    pur = [st['counts'][k].max() / st['labelled'][k] for k in top]
    assert pur == sorted(pur, reverse=True)


def test_random_clusters_are_seeded_and_eligible():
    part = np.array([0] * 30 + [1] * 5 + [2] * 40 + [3] * 25)
    st = C.cluster_stats(part, np.zeros(len(part), int), 1, np.ones(len(part), bool))
    a, b = C.pick_random_clusters(st, n=3), C.pick_random_clusters(st, n=3)
    assert a == b and 1 not in a and sorted(a) == [0, 2, 3]


# ---------------------------------------------------------------------------------------------- paintings
def test_centre_order_is_by_cosine_to_the_cluster_centre():
    x = unit([[1, 0], [0.9, 0.1], [0.5, 0.5], [0.0, 1.0], [0.8, 0.2]])
    part = np.zeros(5, int)
    cen = C.centres(x, part, 1)[0]
    plain = unit(np.eye(5) + 0.01)                              # no near copies
    got = C.paintings_centre(np.arange(5), x, cen, plain, n=5)
    cos = x @ cen
    assert got == [int(i) for i in np.argsort(-cos, kind='stable')]
    # the centre is the mean of ALL members' unit rows, not only the public-domain ones
    only = C.paintings_centre(np.array([3, 4]), x, cen, plain, n=2)
    assert only == [4, 3] and np.allclose(np.linalg.norm(cen), 1)


def test_near_copies_are_skipped_and_the_next_painting_is_taken():
    plain = unit([[1, 0, 0], [0.999, 0.01, 0], [0, 1, 0], [0, 0, 1]])
    assert float(plain[0] @ plain[1]) >= C.NEAR
    assert C.skip_near_copies([0, 1, 2, 3], plain, n=3) == [0, 2, 3]       # 1 skipped (copy of 0), 3 taken instead
    assert C.skip_near_copies([1, 0, 2, 3], plain, n=3) == [1, 2, 3]       # the earlier one stays
    # random rule: same permutation each call, near copy never next to its twin
    x = plain
    a = C.paintings_random(np.arange(4), plain, n=4)
    assert a == C.paintings_random(np.arange(4), plain, n=4) and not ({0, 1} <= set(a))


def test_only_public_domain_members_are_shown():
    rng = np.random.default_rng(0)
    x = unit(rng.normal(size=(60, 6)))
    part = np.arange(60) % 3
    lab = np.arange(60) % 2
    pdm = rng.random(60) < .5
    rows = C.build_panels(part, x, x, lab, ['a', 'b'], pdm, 'purity', 'centre')
    shown = [i for r in rows for i in r['rows']]
    assert shown and pdm[shown].all()
    assert all(r['pd_members'] == int(pdm[part == r['cluster']].sum()) for r in rows)


# ---------------------------------------------------------------------------------------------- teaser
def _brute_teaser(heads, plain, labels, pdm, keys, keep, n, min_agree, thr):
    asp = list(heads)
    q_all = [i for i in range(len(keys)) if pdm[i] and all(labels[a][i] >= 0 for a in asp)]
    out = []
    for q in q_all:
        gal = [g for g in range(len(keys)) if pdm[g] and float(plain[g] @ plain[q]) < thr]
        nb, agree = {}, {}
        for a in asp:
            s = sorted(gal, key=lambda g: (-float(heads[a][g] @ heads[a][q]), g))[:n]
            nb[a] = s
            agree[a] = sum(1 for g in s if labels[a][g] == labels[a][q] and labels[a][g] >= 0)
        if min(agree.values()) < min_agree:
            continue
        distinct = len({g for a in asp for g in nb[a]})
        out.append((-distinct, -np.mean(list(agree.values())) / n, keys[q], q, nb))
    out.sort(key=lambda t: t[:3])
    return out[:keep]


def test_teaser_ranking_follows_the_rule_and_matches_a_brute_force_reference():
    rng = np.random.default_rng(3)
    n = 70
    keys = [f'k{i:03d}' for i in range(n)]
    heads = {a: unit(rng.normal(size=(n, 5))) for a in D.ASPECTS}
    plain = unit(rng.normal(size=(n, 12)))
    plain[10] = plain[11]                                       # a near copy pair
    labels = {a: rng.integers(0, 2, n) for a in D.ASPECTS}
    labels['genre'][rng.random(n) < .2] = -1                    # unlabelled genre: never matches, never a query
    pdm = rng.random(n) < .85
    got, info = C.teaser_candidates(heads, plain, labels, pdm, keys, keep=8, n=5, min_agree=2, chunk=16)
    want = _brute_teaser(heads, plain, labels, pdm, keys, 8, 5, 2, C.NEAR)
    assert len(got) == len(want) > 2
    for g, w in zip(got, want):
        assert keys[g['row']] == w[2]
        for a in D.ASPECTS:
            assert g['heads'][a]['rows'] == w[4][a]
    # rule: distinct paintings (more first), then mean agreement (higher), then key
    sk = [(-g['distinct'], -g['mean_agreement'], keys[g['row']]) for g in got]
    assert sk == sorted(sk) and [g['rank'] for g in got] == list(range(1, len(got) + 1))
    assert all(min(h['agree'] for h in g['heads'].values()) >= 2 for g in got)
    assert all(pdm[g['row']] and all(labels[a][g['row']] >= 0 for a in D.ASPECTS) for g in got)


def test_teaser_gallery_excludes_the_query_and_its_near_copies():
    rng = np.random.default_rng(5)
    n = 30
    x = unit(rng.normal(size=(n, 6)))
    plain = unit(rng.normal(size=(n, 10)))
    plain[1] = plain[0]                                         # row 1 is a near copy of row 0
    x[1] = x[0]                                                 # ... and sits right next to it in the head space
    idx, _ = C.topk_neighbours(np.array([0]), np.arange(n), {'h': x}, plain, n=5)
    assert 0 not in idx['h'][0] and 1 not in idx['h'][0]


def test_agreement_counts_unlabelled_neighbours_as_no_match():
    lab = np.array([2, 2, -1, 0, 2])
    m = C.agreement(np.array([[1, 2, 3, 4]]), np.array([0]), lab)
    assert m.tolist() == [[True, False, False, True]]


# ---------------------------------------------------------------------------------------------- pd and viewer helpers
def test_slug_normalisation_and_public_domain_flag():
    assert D.norm_slug("Pierre-Paul-Prud'hon") == 'pierre-paul-prudhon'
    assert D.norm_slug('a.y.-jackson') == 'ay-jackson' and D.norm_slug('Émile  Bernard') == 'emile-bernard'
    assert PD.year_of('1983-01-01T00:00:00Z') == 1983 and PD.year_of('-0500-01-01T00:00:00Z') == -500
    b = [dict(id=dict(value='claude-monet'), d=dict(type='literal', value='1926-12-05T00:00:00Z')),
         dict(id=dict(value='peter-busa'), d=dict(type='literal', value='1983-01-01T00:00:00Z')),
         dict(id=dict(value='peter-busa'), d=dict(type='literal', value='1985-01-01T00:00:00Z')),
         dict(id=dict(value='henri-matisse'), d=dict(type='literal', value='1955-01-01T00:00:00Z')),
         dict(id=dict(value='alive-artist')), dict(id=dict(value='n-c-wyeth'), d=dict(type='literal', value='1945-01-01T00:00:00Z'))]
    years, raw = PD.death_years(b)
    assert years['peter-busa'] == 1985                          # several values -> the latest
    rows = {r['slug']: r for r in PD.artist_table(['claude-monet', 'peter-busa', 'henri-matisse', 'alive-artist', 'nobody',
                                                   'n.c.-wyeth'], years, raw)}
    assert rows['claude-monet']['public_domain'] and rows['henri-matisse']['public_domain']      # 1955 counts
    assert not rows['peter-busa']['public_domain'] and not rows['alive-artist']['public_domain']
    assert rows['nobody']['matched_id'] is None and not rows['nobody']['public_domain']
    assert rows['n.c.-wyeth']['match'] == 'dotdash' and rows['n.c.-wyeth']['public_domain']


def test_thumbnail_packing_stays_under_the_file_limit():
    items = [(f'k{i}', 'x' * 1000) for i in range(50)]
    chunks = V.pack(items, limit=10_000)
    assert len(chunks) > 1 and sum(len(c) for c in chunks) == 50
    import json
    assert all(len(json.dumps(c)) <= 10_000 for c in chunks)
