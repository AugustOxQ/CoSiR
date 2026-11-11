"""Step `candidates` of the qualitative-figure round (paper section 4.6): candidates by fixed rules (no choosing).

Panel (a) "same subject, different feeling", panel (b) "close in one aspect, far in another", and the cluster figure's
class ranking (from r20's finished candidates, no new compute). Pure functions on arrays first, then `run`.
r20's code is imported, not copied: data door (`r20_data`), public-domain table (`r20_pd`), top-k with near-copy
exclusion (`r20_cand.topk_neighbours`), `agreement`, `painting_record`.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROUND = Path(__file__).resolve().parent
R20 = Path('/project/CoSiR/src/test/20261109_r20_figures')
if str(R20) not in sys.path:
    sys.path.insert(0, str(R20))

import r20_cand as RC  # noqa: E402
import r20_data as D  # noqa: E402
import r20_pd as PD  # noqa: E402

OUT = Path(os.environ.get('QF_OUTPUTS', ROUND / 'outputs'))
R20_CANDIDATES = Path('/data/SSD2/cosir_r20/20261109_r20_figures/candidates/candidates.json')
NEAR = RC.NEAR
ASPECTS = D.ASPECTS
TEASER_KEY = 'childe-hassam_old-lyme-bridge'
SMOKE_ROWS = 800
NB = 5
A_MIN_E_OURS, A_MAX_E_PLAIN, A_MIN_G_PLAIN = 4, 1, 3
A_PER_CLASS, A_TOTAL = 2, 18
B_PER_PAIR = 3
PAIRS = (('emotion', 'style'), ('emotion', 'genre'), ('style', 'emotion'), ('style', 'genre'),
         ('genre', 'emotion'), ('genre', 'style'))
CLASS_COUNT = dict(emotion=9, style=27, genre=10)
IN_FIGURE = 4

RULES_TEXT = [
    'Shared: ArtELingo test set only (12,281 paintings). c* = our model, seed 0, one head per aspect, compared with the plain image only (frozen Qwen3-VL-Embedding-2B image, no instruction). Rows are unit-normalised (float32), similarity is cosine.',
    'Shared: public-domain paintings only (artist died 1955 or earlier, Wikidata, r20 table). Near copies are skipped (plain cosine 0.95 or more with the query; the query itself is included). The teaser\'s query childe-hassam_old-lyme-bridge is never a query, a neighbour or a pair member.',
    'Panel (a), step 1: queries are public-domain paintings with an emotion label and a genre label. Gallery: public-domain paintings (minus the teaser painting); the query and its near copies are dropped from its list.',
    'Panel (a), step 2: for each query take the top 5 in the plain image and the top 5 in our emotion head. e_ours = our top 5 with the query\'s emotion; e_plain = plain top 5 with the query\'s emotion; g_plain = plain top 5 with the query\'s genre; g_ours = our top 5 with the query\'s genre. An unlabelled neighbour never matches.',
    'Panel (a), step 3: keep a query when e_ours >= 4 and e_plain <= 1 and g_plain >= 3.',
    'Panel (a), step 4: rank by (e_ours - e_plain) descending, then g_ours ascending, then key ascending.',
    'Panel (a), step 5: walk the ranked list; keep at most 2 queries per query emotion class, up to 18 in total. IDs a1, a2, ... in that order.',
    'Panel (b): six ordered aspect pairs (A near, B far) in this order: (emotion, style), (emotion, genre), (style, emotion), (style, genre), (genre, emotion), (genre, style).',
    'Panel (b), step 1: P = public-domain paintings (minus the teaser painting) with labels in both A and B. For each first painting i in P take its top 5 in our A head over the gallery P (i and its near copies excluded).',
    'Panel (b), step 2: a candidate (i, j) has j in i\'s A top 5, label_A(j) == label_A(i) and label_B(j) != label_B(i).',
    'Panel (b), step 3: far in B = the rank of j in i\'s list in our B head over the same gallery (i and its near copies excluded; rank 1 = nearest; ties count as not nearer), as a percentile = rank / gallery size for i. Rank candidates by this percentile descending, then A cosine descending, then (key_i, key_j) ascending. (i, j) and (j, i) are the same pair: keep the first.',
    'Panel (b), step 4: keep the top 3 per ordered pair, with no painting used twice across all of panel (b) (the six lists are processed in the fixed order; a candidate that uses an already used painting is skipped). IDs b1 ... b18. The plain-image cosine of (i, j) and the plain rank percentile of j in i\'s plain list (same gallery) are stored beside.',
    'Cluster figure at the class count (emotion 9, style 27, genre 10), no new compute: from r20\'s candidates (k-means on all rows, per-class rule, 6 paintings nearest the cluster centre per row), join the c* and plain rows by class. gain = c* class share - plain class share; classes sorted by gain descending; the first per aspect is the rule\'s pick (as the published 20-cluster figure). Each row keeps r20\'s 6 paintings, the first 4 are marked "in figure". IDs c-<aspect>-<rank by gain>.',
]


# ------------------------------------------------------------------------------------------------ masks and ranks
def usable_mask(keys, pd_mask, exclude=(TEASER_KEY,)):
    """Public-domain rows minus the excluded keys (the teaser painting)."""
    ex = set(exclude)
    return np.asarray(pd_mask, dtype=bool) & np.array([k not in ex for k in keys], dtype=bool)


def ranks_of(q, t, x, plain, g_rows, thr=NEAR, chunk=500):
    """For each pair (query row q[m], target row t[m]; the target is a gallery row): rank of the target in the query's
    list over `g_rows` in space x (rank 1 = nearest; only strictly nearer rows count, query and its near copies
    (plain cosine >= thr) excluded), the gallery size for that query (excluded rows removed), and the cosine.
    Chunked: one [chunk, gallery] float32 block at a time."""
    q = np.asarray(q, dtype=np.int64)
    t = np.asarray(t, dtype=np.int64)
    g_rows = np.asarray(g_rows, dtype=np.int64)
    col = np.full(len(x), -1, dtype=np.int64)
    col[g_rows] = np.arange(len(g_rows))
    assert (col[t] >= 0).all(), 'target outside the gallery'
    xg, pg = x[g_rows], plain[g_rows]
    rank = np.zeros(len(q), dtype=np.int64)
    gsz = np.zeros(len(q), dtype=np.int64)
    cos = np.zeros(len(q), dtype=np.float32)
    uq = np.unique(q)
    for a in range(0, len(uq), chunk):
        qs = uq[a:a + chunk]
        sel = np.flatnonzero((q >= qs[0]) & (q <= qs[-1]) & np.isin(q, qs))
        bad = (plain[qs] @ pg.T) >= thr
        m = x[qs] @ xg.T
        m[bad] = -np.inf
        r = np.searchsorted(qs, q[sel])
        c = m[r, col[t[sel]]]
        rank[sel] = 1 + (m[r] > c[:, None]).sum(1)
        gsz[sel] = len(g_rows) - bad.sum(1)[r]
        cos[sel] = c
    return rank, gsz, cos


# ------------------------------------------------------------------------------------------------ panel (a)
def select_a(keys_q, q_emotion, e_ours, e_plain, g_plain, g_ours, min_e_ours=A_MIN_E_OURS, max_e_plain=A_MAX_E_PLAIN,
             min_g_plain=A_MIN_G_PLAIN, per_class=A_PER_CLASS, total=A_TOTAL):
    """Filter, rank and cap. All arguments are per query (same order). Returns (positions of the kept queries in
    final order, positions that passed the filter in ranked order)."""
    e_ours, e_plain, g_plain, g_ours = (np.asarray(v) for v in (e_ours, e_plain, g_plain, g_ours))
    ok = np.flatnonzero((e_ours >= min_e_ours) & (e_plain <= max_e_plain) & (g_plain >= min_g_plain))
    ranked = sorted(ok.tolist(), key=lambda i: (-(int(e_ours[i]) - int(e_plain[i])), int(g_ours[i]), keys_q[i]))
    used, kept = {}, []
    for i in ranked:
        c = int(q_emotion[i])
        if used.get(c, 0) >= per_class:
            continue
        used[c] = used.get(c, 0) + 1
        kept.append(i)
        if len(kept) == total:
            break
    return kept, ranked


def _nb_list(s, rows, cos):
    return [dict(key=s.keys[int(r)], cos=round(float(c), 4),
                 labels={a: (s.class_names[a][int(s.labels[a][r])] if s.labels[a][r] >= 0 else None) for a in ASPECTS})
            for r, c in zip(rows, cos)]


def build_a(s, plain, x_emotion, use):
    lab = s.labels
    q_rows = np.flatnonzero(use & (lab['emotion'] >= 0) & (lab['genre'] >= 0))
    g_rows = np.flatnonzero(use)
    idx, sim = RC.topk_neighbours(q_rows, g_rows, dict(ours=x_emotion, plain=plain), plain, n=NB)
    cnt = lambda sp, a: RC.agreement(idx[sp], q_rows, lab[a]).sum(1)  # noqa: E731
    e_ours, e_plain, g_plain, g_ours = cnt('ours', 'emotion'), cnt('plain', 'emotion'), cnt('plain', 'genre'), cnt('ours', 'genre')
    keys_q = [s.keys[r] for r in q_rows]
    kept, ranked = select_a(keys_q, lab['emotion'][q_rows], e_ours, e_plain, g_plain, g_ours)
    names = s.class_names['emotion']
    per_rule = {}
    for i in ranked:
        n = names[int(lab['emotion'][q_rows[i]])]
        per_rule[n] = per_rule.get(n, 0) + 1
    per_kept = {}
    cands = []
    for rank, i in enumerate(kept, 1):
        n = names[int(lab['emotion'][q_rows[i]])]
        per_kept[n] = per_kept.get(n, 0) + 1
        cands.append(dict(
            id=f'a{rank}', query=keys_q[i], query_emotion=n,
            counts=dict(e_ours=int(e_ours[i]), e_plain=int(e_plain[i]), g_plain=int(g_plain[i]), g_ours=int(g_ours[i])),
            ours=_nb_list(s, idx['ours'][i], sim['ours'][i]), plain=_nb_list(s, idx['plain'][i], sim['plain'][i])))
    info = dict(queries=int(len(q_rows)), gallery=int(len(g_rows)), kept_by_rule=int(len(ranked)),
                kept_by_rule_per_emotion_class=dict(sorted(per_rule.items())),
                final=int(len(cands)), final_per_emotion_class=dict(sorted(per_kept.items())))
    return cands, info


# ------------------------------------------------------------------------------------------------ panel (b)
def pair_candidates(xa, xb, plain, laba, labb, rows, n=NB, thr=NEAR, chunk=500):
    """All candidates (i, j) of one ordered pair (A near, B far), unsorted. `rows` = P (rows with labels in both).
    Returns a list of dicts with rows i, j and the A / B / plain numbers, and |P|."""
    rows = np.asarray(rows, dtype=np.int64)
    idx, sim = RC.topk_neighbours(rows, rows, dict(a=xa), plain, n=n, thr=thr, chunk=chunk)
    qi = np.repeat(rows, n)
    j = idx['a'].reshape(-1)
    cosa = sim['a'].reshape(-1)
    ranka = np.tile(np.arange(1, n + 1), len(rows))
    ok = np.isfinite(cosa) & (laba[j] == laba[qi]) & (labb[j] != labb[qi])
    qi, j, cosa, ranka = qi[ok], j[ok], cosa[ok], ranka[ok]
    if len(qi) == 0:
        return [], int(len(rows))
    rb, gb, cb = ranks_of(qi, j, xb, plain, rows, thr, chunk)
    rp, gp, cp = ranks_of(qi, j, plain, plain, rows, thr, chunk)
    assert (gb == gp).all()
    out = [dict(i=int(qi[m]), j=int(j[m]), a_cos=float(cosa[m]), a_rank=int(ranka[m]), b_cos=float(cb[m]),
                b_rank=int(rb[m]), gallery=int(gb[m]), b_percentile=float(rb[m]) / float(gb[m]),
                plain_cos=float(cp[m]), plain_rank=int(rp[m]), plain_percentile=float(rp[m]) / float(gb[m]))
           for m in range(len(qi))]
    return out, int(len(rows))


def rank_dedup(cands, keys):
    """Sort by b_percentile desc, a_cos desc, (key_i, key_j) asc; keep the first of (i, j) / (j, i)."""
    srt = sorted(cands, key=lambda c: (-c['b_percentile'], -c['a_cos'], keys[c['i']], keys[c['j']]))
    seen, out = set(), []
    for c in srt:
        p = frozenset((c['i'], c['j']))
        if p in seen:
            continue
        seen.add(p)
        out.append(c)
    return out


def pick_b(lists, per_pair=B_PER_PAIR):
    """lists: ranked, deduplicated candidate lists in the fixed pair order. Keep per_pair per list, no painting (row)
    used twice across all lists. Returns one list of picks per input list."""
    used, out = set(), []
    for lst in lists:
        got = []
        for c in lst:
            if c['i'] in used or c['j'] in used:
                continue
            got.append(c)
            used.update((c['i'], c['j']))
            if len(got) == per_pair:
                break
        out.append(got)
    return out


def _labels(s, r):
    return {a: (s.class_names[a][int(s.labels[a][r])] if s.labels[a][r] >= 0 else None) for a in ASPECTS}


def build_b(s, plain, heads, use):
    lists, info = [], dict(pairs={})
    for A, B in PAIRS:
        rows = np.flatnonzero(use & (s.labels[A] >= 0) & (s.labels[B] >= 0))
        raw, nP = pair_candidates(heads[A], heads[B], plain, s.labels[A], s.labels[B], rows)
        ded = rank_dedup(raw, s.keys)
        lists.append(ded)
        info['pairs'][f'{A}/{B}'] = dict(P=nP, candidates_raw=len(raw), candidates_after_pair_dedup=len(ded))
    picks = pick_b(lists)
    out, n = [], 0
    for (A, B), got in zip(PAIRS, picks):
        for c in got:
            n += 1
            out.append(dict(
                id=f'b{n}', A=A, B=B, key_i=s.keys[c['i']], key_j=s.keys[c['j']],
                labels_i=_labels(s, c['i']), labels_j=_labels(s, c['j']),
                a_cos=round(c['a_cos'], 4), a_rank=c['a_rank'], b_cos=round(c['b_cos'], 4), b_rank=c['b_rank'],
                gallery=c['gallery'], b_percentile=round(c['b_percentile'], 4), plain_cos=round(c['plain_cos'], 4),
                plain_percentile=round(c['plain_percentile'], 4), plain_rank=c['plain_rank']))
    info['final'] = len(out)
    return out, info


# ------------------------------------------------------------------------------------------------ clusters
def cluster_gain(cstar_rows, plain_rows, n_in_figure=IN_FIGURE):
    """Join the per-class rows of c* and plain by class; gain = c* class_share - plain class_share; sort by gain
    descending (ties: the order of the c* rows). Returns a list of dicts (rank 1 = the rule's pick)."""
    pl = {r['class']: r for r in plain_rows}
    assert set(pl) == {r['class'] for r in cstar_rows} and len(pl) == len(plain_rows), 'class sets differ'
    rows = []
    for pos, r in enumerate(cstar_rows):
        p = pl[r['class']]
        rows.append((-(r['class_share'] - p['class_share']), pos, r, p))
    rows.sort(key=lambda t: (t[0], t[1]))
    out = []
    for rank, (_, pos, r, p) in enumerate(rows, 1):
        def side(x):
            return dict(share=x['class_share'], size=x['size'], cluster=x['cluster'], labelled=x['labelled'],
                        paintings=list(x['paintings']), in_figure=list(x['paintings'][:n_in_figure]))
        out.append(dict(rank=rank, **{'class': r['class']}, gain=round(r['class_share'] - p['class_share'], 4),
                        cstar=side(r), plain=side(p)))
    return out


def build_clusters(r20_doc):
    S = r20_doc['sets']['artelingo']
    out = {}
    for a in ASPECTS:
        pc = S['panels'][a][str(CLASS_COUNT[a])]['per_class']['centre']
        rows = cluster_gain(pc['cstar'], pc['plain'])
        for r in rows:
            r['id'] = f'c-{a}-{r["rank"]}'
        out[a] = rows
    return out


# ------------------------------------------------------------------------------------------------ run
def input_shas(s):
    """sha256 of every input file of c* and plain rows (not cached: r20's cache sits in r20's outputs)."""
    files = list(s.sources['c_star'].values()) + [s.sources['plain']]
    return {f: D.sha256_file(f) for f in files}


def painting_table(s, death, keys):
    pos = {k: i for i, k in enumerate(s.keys)}
    return {k: RC.painting_record(s, pos[k], death) for k in sorted(keys)}


def run(smoke=False):
    t0 = time.time()
    root = OUT / ('smoke' if smoke else '')
    outd = root / 'candidates'
    outd.mkdir(parents=True, exist_ok=True)
    full = D.load('artelingo')
    assert TEASER_KEY in full.keys
    death_full, pd_full = PD.load_pd('artelingo', full.artists)
    s = full.smoke(SMOKE_ROWS) if smoke else full
    death, pd_mask = PD.load_pd('artelingo', s.artists)
    use = usable_mask(s.keys, pd_mask)
    plain = s.plain()
    heads = {a: s.cstar(a) for a in ASPECTS}
    cands_a, info_a = build_a(s, plain, heads['emotion'], use)
    cands_b, info_b = build_b(s, plain, heads, use)
    r20_doc = json.loads(R20_CANDIDATES.read_text())
    clusters = build_clusters(r20_doc)
    keys = set()
    for c in cands_a:
        keys.add(c['query'])
        keys.update(n['key'] for n in c['ours'] + c['plain'])
    for c in cands_b:
        keys.update((c['key_i'], c['key_j']))
    for rows in clusters.values():
        for r in rows:
            keys.update(r['cstar']['paintings'])
            keys.update(r['plain']['paintings'])
    assert TEASER_KEY not in keys
    pos = {k: i for i, k in enumerate(full.keys)}
    assert all(pd_full[pos[k]] for k in keys if k in pos), 'a shown painting is not public domain'
    doc = dict(
        header=dict(rules=RULES_TEXT, smoke=bool(smoke), made=time.strftime('%F %H:%M'), near_copy_cosine=NEAR,
                    excluded=[TEASER_KEY], seeds=dict(cstar='seed 0'),
                    r20_candidates=str(R20_CANDIDATES), smoke_rows=len(s) if smoke else None,
                    pairs=[list(p) for p in PAIRS], class_count=CLASS_COUNT,
                    painting_table='paintings[key] = img path, labels in the three aspects, artist, death year',
                    input_sha256=dict(files=input_shas(s) if not smoke else 'not computed in smoke runs',
                                      keys=s.keys_sha256())),
        counts=dict(set_size=len(s), public_domain=int(pd_mask.sum()), usable=int(use.sum()),
                    panel_a=info_a, panel_b=info_b,
                    clusters={a: len(r) for a, r in clusters.items()}),
        panel_a=cands_a, panel_b=cands_b, clusters=clusters,
        paintings=painting_table(full, death_full, keys))
    (outd / 'candidates.json').write_text(json.dumps(doc, separators=(',', ':')))
    doc['seconds'] = round(time.time() - t0, 1)
    return doc
