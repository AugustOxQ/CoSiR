"""Step `candidates`: every candidate panel and teaser, by fixed rules (no choosing). Pure functions on arrays first,
then `build_set` / `run`, which read the partitions saved by `r20_part`.

Rules (stated again in the header of candidates.json):
- clusters come from all rows; only public-domain paintings (artist died 1955 or earlier, Wikidata) are shown;
- cluster rules: per_class, purity, random; painting rules: centre, random; near copies skipped (plain cosine >= .95).
"""
import json
import time

import numpy as np
import scipy.sparse as sp

import r20_data as D
import r20_part as PART
import r20_pd as PD

NEAR = 0.95
N_PAINT = 6
N_CLUSTERS = 6
MIN_LABELLED = 20
MIN_MEMBERS = 20
TEASER_KEEP = 12
TEASER_NB = 5
TEASER_MIN_AGREE = 3
CLUSTER_RULES = ('per_class', 'purity', 'random')
PAINTING_RULES = ('centre', 'random')

RULES_TEXT = [
    'Sets: ArtELingo held (12,281 paintings) and MMArt test (16,384). c* = our model, seed 0, one head per aspect. Plain = frozen Qwen3-VL-Embedding-2B image without instruction. Rows are unit-normalised (float32).',
    'Clusters: r17 ns_metrics.partitions (MiniBatchKMeans, n_init 3, batch 4096), state 0, on all rows, at k = 20 and k = the aspect class count (MMArt style: 20, one partition serves both).',
    'Public domain: the painting\'s artist has a death year of 1955 or earlier in Wikidata (WikiArt ID P6002, P570 latest year); unmatched or no death date = not public domain. Only public-domain paintings are shown.',
    'Each rule is applied to c* (the aspect\'s head) and to the plain image separately, with the aspect\'s labels; unlabelled rows are ignored in every share.',
    'Cluster rule per_class: for each class, in descending class frequency, the cluster with the largest share of that class among its labelled members (ties: larger cluster, then lower id); one row per class; a cluster serving two classes appears twice, marked.',
    'Cluster rule purity: the 6 clusters with the highest purity (share of the majority label among labelled members) among clusters with at least 20 labelled members, ranked (ties: more labelled members, then lower id).',
    'Cluster rule random: 6 clusters drawn with numpy default_rng(0) without replacement among clusters with at least 20 members (all rows), in drawn order.',
    'Painting rule centre: the 6 public-domain members with the highest cosine to the cluster centre (mean of the unit rows of all members, in that model\'s space), descending.',
    'Painting rule random: 6 public-domain members drawn with numpy default_rng(0), the same draw for a given cluster.',
    'Both painting rules: skip a painting whose plain-image cosine with an earlier painting of the same row is 0.95 or more (near copy) and take the next one.',
    'Teaser queries: public-domain paintings with all three labels. Gallery: public-domain paintings except the query and its near copies (plain cosine >= 0.95). Top 5 neighbours by cosine in each c* head (emotion, style, genre) and in the plain image.',
    'Teaser agreement per aspect: share of that head\'s top 5 that carry the query\'s label in that aspect (an unlabelled genre neighbour counts as no match). Keep queries with at least 3 of 5 in all three aspects; rank by distinct paintings over the three head lists (more first), then mean agreement (higher), then key; keep the top 12. The plain image\'s top 5 and its agreement per aspect are stored beside.',
]


# ------------------------------------------------------------------------------------------------ cluster statistics
def cluster_stats(part, labels, n_classes, pd_mask):
    """Per cluster id 0..K-1: size, labelled count, class counts [K, C], public-domain member count. Joins are by row
    of the same set; labels < 0 are ignored in every share."""
    part = np.asarray(part)
    labels = np.asarray(labels)
    K = int(part.max()) + 1
    size = np.bincount(part, minlength=K)
    lab = labels >= 0
    counts = np.zeros((K, n_classes), dtype=np.int64)
    np.add.at(counts, (part[lab], labels[lab]), 1)
    return dict(K=K, size=size, labelled=counts.sum(1), counts=counts,
                pd=np.bincount(part[np.asarray(pd_mask, dtype=bool)], minlength=K))


def class_order(labels, n_classes):
    """Class ids by descending frequency among labelled rows (ties: lower id); classes with no labelled row last."""
    c = np.bincount(np.asarray(labels)[np.asarray(labels) >= 0], minlength=n_classes)
    return [int(i) for i in sorted(range(n_classes), key=lambda i: (-c[i], i))]


def pick_per_class(st, order):
    """[(class, cluster)] one per class in `order`: largest share of that class among a cluster's labelled members;
    ties: larger cluster, then lower id. Clusters without labelled members have no share."""
    lab = st['labelled']
    out = []
    for c in order:
        share = np.where(lab > 0, st['counts'][:, c] / np.maximum(lab, 1), -1.0)
        best = max(range(st['K']), key=lambda k: (round(share[k], 12), st['size'][k], -k))
        out.append((c, int(best)))
    return out


def pick_purity(st, n=N_CLUSTERS, min_labelled=MIN_LABELLED):
    """The n clusters with the highest purity among those with at least `min_labelled` labelled members, ranked
    (ties: more labelled members, then lower id)."""
    ok = [k for k in range(st['K']) if st['labelled'][k] >= min_labelled]
    pur = {k: st['counts'][k].max() / st['labelled'][k] for k in ok}
    ok.sort(key=lambda k: (-round(pur[k], 12), -st['labelled'][k], k))
    return ok[:n]


def pick_random_clusters(st, n=N_CLUSTERS, min_members=MIN_MEMBERS, seed=0):
    """n clusters drawn with default_rng(seed) without replacement among those with at least `min_members` rows, in
    drawn order."""
    ok = np.flatnonzero(st['size'] >= min_members)
    rng = np.random.default_rng(seed)
    return [int(k) for k in rng.choice(ok, size=min(n, len(ok)), replace=False)]


# ------------------------------------------------------------------------------------------------ paintings of a cluster
def centres(x, part, K):
    """Unit cluster centres [K, d]: the mean of the unit rows of all members, divided by its norm."""
    part = np.asarray(part)
    onehot = sp.csr_matrix((np.ones(len(part), dtype=np.float64), (part, np.arange(len(part)))), shape=(K, len(part)))
    c = onehot @ x.astype(np.float64)
    n = np.linalg.norm(c, axis=1, keepdims=True)
    return (c / np.maximum(n, 1e-12)).astype(np.float32)


def skip_near_copies(order, plain, n=N_PAINT, thr=NEAR):
    """Walk `order` (row indices); take a row unless its plain cosine with an already taken row is >= thr."""
    taken = []
    for i in order:
        if taken and float((plain[taken] @ plain[i]).max()) >= thr:
            continue
        taken.append(int(i))
        if len(taken) == n:
            break
    return taken


def paintings_centre(members, x, centre, plain, n=N_PAINT, thr=NEAR):
    """Public-domain `members` (row indices) by descending cosine to `centre` (ties: lower row), near copies skipped."""
    members = np.asarray(members, dtype=np.int64)
    if len(members) == 0:
        return []
    cos = x[members] @ centre
    order = members[np.lexsort((members, -cos))]
    return skip_near_copies(order, plain, n, thr)


def paintings_random(members, plain, n=N_PAINT, thr=NEAR, seed=0):
    """Public-domain `members` in the order of a default_rng(seed) permutation (the same for a given cluster), near
    copies skipped."""
    members = np.sort(np.asarray(members, dtype=np.int64))
    if len(members) == 0:
        return []
    order = members[np.random.default_rng(seed).permutation(len(members))]
    return skip_near_copies(order, plain, n, thr)


# ------------------------------------------------------------------------------------------------ rows
def row_info(st, k, class_names):
    """Cluster facts: size, labelled count, majority label and its share, public-domain member count."""
    lab = int(st['labelled'][k])
    maj = int(st['counts'][k].argmax()) if lab else None
    return dict(cluster=int(k), size=int(st['size'][k]), labelled=lab,
                majority=class_names[maj] if lab else None,
                majority_share=round(float(st['counts'][k][maj]) / lab, 4) if lab else None,
                pd_members=int(st['pd'][k]))


def panel_rows(rule, st, labels, class_names, order):
    """The chosen clusters of one cluster rule as rows (without paintings): per_class rows carry the row's class, its
    share in the cluster and `dup` (the cluster also serves another class)."""
    if rule == 'per_class':
        pairs = pick_per_class(st, order)
        used = {}
        for c, k in pairs:
            used[k] = used.get(k, 0) + 1
        rows = []
        for c, k in pairs:
            r = row_info(st, k, class_names)
            lab = int(st['labelled'][k])
            r.update({'class': class_names[c], 'class_share': round(float(st['counts'][k][c]) / lab, 4) if lab else None,
                      'dup': used[k] > 1})
            rows.append(r)
        return rows
    ks = pick_purity(st) if rule == 'purity' else pick_random_clusters(st)
    return [row_info(st, k, class_names) for k in ks]


def build_panels(part, x, plain, labels, class_names, pd_mask, rule_cluster, rule_paint):
    """Rows with their paintings (row indices) for one model, one count, one cluster rule and one painting rule."""
    st = cluster_stats(part, labels, len(class_names), pd_mask)
    rows = panel_rows(rule_cluster, st, labels, class_names, class_order(labels, len(class_names)))
    cen = centres(x, part, st['K']) if rule_paint == 'centre' else None
    for r in rows:
        k = r['cluster']
        members = np.flatnonzero((part == k) & pd_mask)
        r['rows'] = (paintings_centre(members, x, cen[k], plain) if rule_paint == 'centre'
                     else paintings_random(members, plain))
    return rows


# ------------------------------------------------------------------------------------------------ teaser
def topk_neighbours(q_rows, g_rows, spaces, plain, n=TEASER_NB, thr=NEAR, chunk=500):
    """For each query row index and each space (name -> unit rows over all rows): the n gallery rows with the highest
    cosine, near copies of the query (plain cosine >= thr, the query itself included) excluded. Returns
    {space: int [nq, n]} of row indices (ties: lower row) and {space: float [nq, n]} cosines."""
    q_rows = np.asarray(q_rows)
    g_rows = np.asarray(g_rows)
    pg = plain[g_rows]
    idx = {s: np.zeros((len(q_rows), n), dtype=np.int64) for s in spaces}
    sim = {s: np.zeros((len(q_rows), n), dtype=np.float32) for s in spaces}
    for a in range(0, len(q_rows), chunk):
        qs = q_rows[a:a + chunk]
        bad = (plain[qs] @ pg.T) >= thr
        for s, x in spaces.items():
            m = x[qs] @ x[g_rows].T
            m[bad] = -np.inf
            part = np.argpartition(-m, n - 1, axis=1)[:, :n]
            vals = np.take_along_axis(m, part, 1)
            gr = g_rows[part]
            order = np.stack([np.lexsort((gr[i], -vals[i])) for i in range(len(qs))])
            idx[s][a:a + chunk] = np.take_along_axis(gr, order, 1)
            sim[s][a:a + chunk] = np.take_along_axis(vals, order, 1)
    return idx, sim


def agreement(nb_rows, q_rows, labels):
    """Matches [nq, n] bool: the neighbour carries the query's label in this aspect (labels < 0 never match)."""
    ql = labels[q_rows][:, None]
    nl = labels[nb_rows]
    return (nl == ql) & (nl >= 0) & (ql >= 0)


def teaser_candidates(heads, plain, labels, pd_mask, keys, keep=TEASER_KEEP, n=TEASER_NB, min_agree=TEASER_MIN_AGREE,
                      chunk=500):
    """The teaser candidates by the fixed rule. heads: aspect -> unit rows; labels: aspect -> int array. Returns a list
    of dicts (rank order) with the query row, each head's neighbour rows, agreement per aspect (matches over the
    head's own aspect), the plain image's neighbours and its agreement in all three aspects."""
    aspects = list(heads)
    full = np.all([labels[a] >= 0 for a in aspects], axis=0)
    q_rows = np.flatnonzero(pd_mask & full)
    g_rows = np.flatnonzero(pd_mask)
    spaces = dict(heads, plain=plain)
    idx, sim = topk_neighbours(q_rows, g_rows, spaces, plain, n=n, chunk=chunk)
    own = {a: agreement(idx[a], q_rows, labels[a]) for a in aspects}          # head a's top n on aspect a
    ok = np.all([own[a].sum(1) >= min_agree for a in aspects], axis=0)
    cands = []
    for j in np.flatnonzero(ok):
        distinct = len(set(np.concatenate([idx[a][j] for a in aspects]).tolist()))
        mean_agree = float(np.mean([own[a][j].sum() for a in aspects])) / n
        cands.append((distinct, mean_agree, keys[q_rows[j]], j))
    cands.sort(key=lambda t: (-t[0], -t[1], t[2]))
    out = []
    for rank, (distinct, mean_agree, key, j) in enumerate(cands[:keep], 1):
        out.append(dict(
            rank=rank, row=int(q_rows[j]), distinct=int(distinct), mean_agreement=round(mean_agree, 4),
            heads={a: dict(rows=idx[a][j].tolist(), cos=[round(float(v), 4) for v in sim[a][j]],
                           agree=int(own[a][j].sum())) for a in aspects},
            plain=dict(rows=idx['plain'][j].tolist(), cos=[round(float(v), 4) for v in sim['plain'][j]],
                       agree={a: int(agreement(idx['plain'][j][None], q_rows[j:j + 1], labels[a]).sum())
                              for a in aspects})))
    return out, dict(queries=int(len(q_rows)), gallery=int(len(g_rows)), kept_by_rule=int(len(cands)))


# ------------------------------------------------------------------------------------------------ one set
def painting_record(s, i, death):
    lab = {}
    for a in D.ASPECTS:
        v = int(s.labels[a][i])
        lab[a] = s.class_names[a][v] if v >= 0 else None
    d = death[i]
    return dict(img=s.image_paths[i], labels=lab, artist=s.artists[i],
                death_year=None if np.isnan(d) else int(d))


def build_set(s, root, smoke=False):
    """Everything for one set: panels, teaser, painting table. Needs the partitions of `r20_part` under `root`."""
    death, pd_mask = PD.load_pd(s.name, s.artists)
    plain = s.plain()
    heads = {a: s.cstar(a) for a in D.ASPECTS}
    painting_rows = set()
    panels, counts = {}, {}
    for a in D.ASPECTS:
        ks = PART.counts_for(s, a)
        cnt = int(len(np.unique(s.labels[a][s.labels[a] >= 0])))
        counts[a] = dict(ks=ks, class_count=cnt, class_count_is_20=(cnt == 20))
        panels[a] = {}
        for k in ks:
            panels[a][str(k)] = {}
            parts = dict(cstar=(PART.load_partition(root, s.name, 'cstar', a, k), heads[a]),
                         plain=(PART.load_partition(root, s.name, 'plain', 'plain', k), plain))
            for rc in CLUSTER_RULES:
                panels[a][str(k)][rc] = {}
                for rp in PAINTING_RULES:
                    panels[a][str(k)][rc][rp] = {}
                    for model, (part, x) in parts.items():
                        rows = build_panels(part, x, plain, s.labels[a], s.class_names[a], pd_mask, rc, rp)
                        for r in rows:
                            r['paintings'] = [s.keys[i] for i in r.pop('rows')]
                            for kk in r['paintings']:
                                painting_rows.add(kk)
                        panels[a][str(k)][rc][rp][model] = rows
    teaser, tinfo = teaser_candidates(heads, plain, s.labels, pd_mask, s.keys)
    pos = {k: i for i, k in enumerate(s.keys)}
    for t in teaser:
        for rr in [t['row']] + sum([h['rows'] for h in t['heads'].values()], []) + t['plain']['rows']:
            painting_rows.add(s.keys[rr])
        t['query'] = s.keys[t['row']]
        for h in t['heads'].values():
            h['neighbours'] = [s.keys[r] for r in h.pop('rows')]
        t['plain']['neighbours'] = [s.keys[r] for r in t['plain'].pop('rows')]
        del t['row']
    paintings = {k: painting_record(s, pos[k], death) for k in sorted(painting_rows)}
    return dict(n=len(s), classes=s.class_names, counts=counts, public_domain_paintings=int(pd_mask.sum()),
                teaser_info=tinfo, panels=panels, teaser=teaser, paintings=paintings)


def run(smoke=False, sets=None):
    root = D.OUT / ('smoke' if smoke else '')
    out = root / 'candidates'
    out.mkdir(parents=True, exist_ok=True)
    doc = dict(header=dict(rules=RULES_TEXT, seeds=dict(cstar='seed 0', kmeans_state=0, random_rule='numpy default_rng(0)'),
                           painting_table='sets.<set>.paintings[key] = img path, labels in the three aspects, artist, death year; '
                                          'panel rows and teaser entries refer to paintings by key',
                           smoke=bool(smoke), made=time.strftime('%F %H:%M'), input_sha256={}),
               sets={})
    for name in (sets or D.SETS):
        s = D.load(name)
        if smoke:
            s = s.smoke()
            PART.run(smoke=True, sets=[name], all_jobs=True)
        else:
            check_partitions(root, s)
        doc['header']['input_sha256'][name] = dict(files=D.input_shas(s), keys=s.keys_sha256())
        doc['sets'][name] = build_set(s, root, smoke)
    raw = D.OUT / ('smoke' if smoke else '') / 'pd' / 'wikidata_raw.json'
    tab = D.OUT / 'pd' / 'wikidata_raw.json'
    doc['header']['input_sha256']['wikidata_raw'] = D.sha256_file(tab if tab.exists() else raw)
    (out / 'candidates.json').write_text(json.dumps(doc, separators=(',', ':')))
    return doc


def check_partitions(root, s):
    """Refuse to run on missing or stale partitions (the full `partitions` step must have run on these keys)."""
    missing, stale = [], []
    for model, aspect, k in PART.jobs(s):
        p = PART.path_for(root, s.name, model, aspect, k)
        if not p.exists() or not p.with_suffix('.json').exists():
            missing.append((model, aspect, k))
        elif json.loads(p.with_suffix('.json').read_text())['keys_sha256'] != s.keys_sha256():
            stale.append((model, aspect, k))
    assert not missing and not stale, f'run the partitions step first; {s.name} missing {missing[:3]} stale {stale[:3]}'
