"""r23 stage 3 diagnostics (ticket 10; spec 2026-11-06 text-claim design §3.6, D10, D12).

    python tcl_diag.py [--device D] [--parts N ...]          default: all 8 parts, device cuda; exit 0 on success

Descriptive only: it decides nothing and writes numbers, no wording. Writes OUT/stage3/<part>.json and OUT/stage3/summary.md;
nothing under records/ or stage2/. Without records/verdicts.json it refuses (exit 2), except under smoke. Picks and seed
ranges come from the committed stage 1 records, never recomputed. Every number is read from a score file (`tcl_paths.score_path`),
from the stated rows, or from the store manifests; the ranking, the means, the late vector and the metrics are `tcl_metrics`,
`tcl_forward` and `tcl_score`'s own.

    part1_why_a             pair and label four-way split with late fusion's hit rate; cosine spreads; Spearman (D10)
    part2_a_curve           validation curve of a (0.0 to 1.0, seeds 1 to 4) at the picked read-outs and pair picks
    part3_pair_all_rows     pair R@1, R@5, R@10, median rank of every row, head and read-out
    part4_macro_cells_leak  label mAP and macro mAP, every cell and read-out, plain / no-leak / cross-painting; emotion per class
    part5_side_rows         ArtELingo held side rows
    part6_mmart_other_texts MMArt test's four other texts, no-leak rows for narrative and unified, label-word shares (D9)
    part7_seed_spreads      held seed spreads beside stage 1's validation ranges
    part8_truncation        truncation counts from the 39 store manifests

Every record carries set, kind, row, head, seed ('x' unseeded) or mean_over_seeds with `seeds` and `<value>_per_seed`,
direction, readout and n_query. A record that mixes rows (part 1) names them under `sources`. A value that cannot exist is
null. A score file that a part needs but is absent crashes, except under smoke: the entry is skipped and listed in the
part's `missing`.

Part 1 pair ranking has no pick in the spec: c* is read at the late fusion pair pick's head and read-out, plain Qwen at its
pair pick, late fusion at its pair pick. Shares and late fusion's hit rate per group are computed per seed (seeds 0 to 4) and
then averaged; a group empty in a seed has no hit rate in that seed and the average runs over the seeds that have one.
Label cells: c* and late fusion at their picked read-outs for that aspect and direction, plain Qwen at its own; a query is
"above" when its AP is strictly above that row's own median over the scored queries of the cell (a tie is not above).
"""
import argparse
import json
import sys
import time

import numpy as np
from scipy.stats import rankdata

import tcl_paths as P
import tcl_data
import tcl_forward
import tcl_score
from tcl_metrics import evaluate, fit_means, macro_map, summarize_pair

ENTRIES = ('', '__no_leak', '__cross_painting')
GROUPS4 = ('both', 'only_cstar', 'only_plain', 'neither')
A_GRID = tuple(i / 10 for i in range(11))
CURVE_SEEDS = P.SEEDS[1:]
PART1_SETS = ('art_val', 'art_held', 'mm_test')
N_SAMPLE = 500
SELF_CHECK_TOL = 1e-6
PAIR_RANK = 10
SIDE_OF_ART = P.SIDE_ROWS


# ------------------------------------------------------------------------------------------------ pure helpers
def clean(x):
    """JSON-ready copy: numpy scalars and arrays to Python, NaN and infinity to None."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return float(x) if np.isfinite(x) else None
    return x


def mean_defined(values):
    """(mean over the finite values, how many were finite); (nan, 0) when none."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    return (float(v.mean()) if len(v) else float('nan')), int(len(v))


def hit(first_rank, k=PAIR_RANK):
    """R@k hit per query: first hit rank at most k; NaN (no hit) is a miss."""
    r = np.asarray(first_rank)
    return np.isfinite(r) & (r <= k)


def four_way(flag_cstar, flag_plain, flag_late):
    """Split queries by c* right / plain right (bool arrays of one length) into both, only_cstar, only_plain, neither.
    Per group: n, share of all queries, and late fusion's rate of `flag_late` inside the group (nan for an empty group)."""
    a, b, c = (np.asarray(x, dtype=bool) for x in (flag_cstar, flag_plain, flag_late))
    assert a.shape == b.shape == c.shape and a.ndim == 1 and len(a)
    masks = dict(both=a & b, only_cstar=a & ~b, only_plain=~a & b, neither=~a & ~b)
    out = {}
    for name in GROUPS4:
        m = masks[name]
        n = int(m.sum())
        out[name] = dict(n=n, share=n / len(a), late_rate=float(c[m].mean()) if n else float('nan'))
    assert sum(g['n'] for g in out.values()) == len(a)
    return out


def above_median(ap):
    """(above, scored, median): scored = finite AP; above = scored and AP strictly above the median over scored queries."""
    ap = np.asarray(ap)
    scored = np.isfinite(ap)
    assert scored.any()
    med = float(np.median(ap[scored].astype(np.float64)))
    return scored & (ap > med), scored, med


def label_split(ap_cstar, ap_plain, ap_late):
    """four_way on 'above the row's own median' flags, over the scored queries (the three rows must score the same ones)."""
    above_c, scored, med_c = above_median(ap_cstar)
    above_p, scored_p, med_p = above_median(ap_plain)
    above_l, scored_l, med_l = above_median(ap_late)
    assert np.array_equal(scored, scored_p) and np.array_equal(scored, scored_l), 'rows score different queries'
    out = four_way(above_c[scored], above_p[scored], above_l[scored])
    return out, dict(cstar=med_c, plain=med_p, late=med_l), int(scored.sum())


def average_groups(per_seed):
    """Per-seed four_way dicts to {group: share, late_rate, and their per-seed lists}. A seed with an empty group has no
    late rate; that rate averages over the seeds that have one (`late_rate_n_seeds`)."""
    out = {}
    for name in GROUPS4:
        shares = [s[name]['share'] for s in per_seed]
        rates = [s[name]['late_rate'] for s in per_seed]
        rate, n_def = mean_defined(rates)
        out[name] = dict(share=float(np.mean(shares)), share_per_seed=shares, n_per_seed=[s[name]['n'] for s in per_seed],
                         late_rate=rate, late_rate_per_seed=rates, late_rate_n_seeds=n_def)
    return out


def spread_ratio(std_plain, std_cstar, a=P.A):
    """a * std_plain / ((1 - a) * std_cstar): late cosine = (1 - a) c* cosine + a plain cosine, so a scales the plain
    spread and (1 - a) the c* spread. Nan when the c* spread is 0."""
    return float(a * std_plain / ((1 - a) * std_cstar)) if std_cstar > 0 else float('nan')


def sample_queries(n, size=N_SAMPLE):
    """The fixed query sample: default_rng(0), `size` distinct queries (all when fewer), sorted."""
    return np.sort(np.random.default_rng(0).choice(n, size=min(size, n), replace=False))


def row_ranks(x, chunk=50):
    """Average ranks of each row (ties share their mean rank), float32, chunked to bound memory."""
    out = np.empty(x.shape, dtype=np.float32)
    for lo in range(0, len(x), chunk):
        out[lo:lo + chunk] = rankdata(x[lo:lo + chunk], axis=1, method='average')
    return out


def row_pearson(a, b):
    """Pearson correlation of each row of a with the same row of b, float64; nan for a constant row."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a, b = a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True)
    den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 0, (a * b).sum(1) / den, np.nan)


def spearman_rows(x, y):
    """Spearman correlation per row: Pearson of the rows' average ranks."""
    assert x.shape == y.shape and x.ndim == 2
    return row_pearson(row_ranks(x), row_ranks(y))


def quartiles(values):
    """(mean, q25, q50, q75) over the finite values; nan when none."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if not len(v):
        return (float('nan'),) * 4
    return (float(v.mean()), *(float(q) for q in np.percentile(v, [25, 50, 75])))


# ------------------------------------------------------------------------------------------------ provenance and records
def seeds_of(row):
    return P.SEEDS if row in P.SEEDED_ROWS else ('x',)


def heads_of(row):
    return P.ASPECTS if row in P.HEAD_ROWS else ('all',)


def meta(set_name, kind, row, head, direction, readout, n_query, seeds=('x',)):
    """What produced a value. Seeded rows: mean_over_seeds with the seeds listed; unseeded: seed 'x'."""
    seeded = tuple(seeds) != ('x',)
    out = dict(set=set_name, kind=kind, row=row, head=head, seed=None if seeded else 'x', mean_over_seeds=seeded,
               direction=direction, readout=readout, n_query=int(n_query))
    if seeded:
        out['seeds'] = list(seeds)
    return out


def put(rec, name, per_seed, seeds):
    """Store a value: unseeded as `name`; seeded as `name` (mean over seeds) and `name_per_seed`."""
    if tuple(seeds) == ('x',):
        assert len(per_seed) == 1
        rec[name] = per_seed[0]
    else:
        assert len(per_seed) == len(seeds)
        rec[name] = float(np.mean(np.asarray(per_seed, dtype=np.float64)))
        rec[name + '_per_seed'] = [float(v) for v in per_seed]


def mean_ap(ap):
    """Label mAP of one AP column: the mean over its finite values (nan when none)."""
    v = ap[np.isfinite(ap)]
    return float(v.mean(dtype=np.float64)) if len(v) else float('nan')


def n_scored(ap):
    return int(np.isfinite(ap).sum())


# ------------------------------------------------------------------------------------------------ reading
def _expected(set_name, direction, pid):
    n_i = min(P.SMOKE or P.N_PAINTINGS[set_name], P.N_PAINTINGS[set_name])
    if P.GROUP[set_name] == 'art' and direction != 'i2t':
        assert len(pid) == (len(pid) if P.SMOKE else P.N_CAPTIONS[set_name])
    else:
        assert len(pid) == n_i
        assert np.array_equal(pid, np.arange(n_i)), 'painting order'
    assert ((pid >= 0) & (pid < n_i)).all()


def read_score(path, set_name, direction, cols=()):
    """{'q_pid', 'first_rank', and the requested ap columns}: keys, dtypes, shapes, ranges and query count asserted."""
    with np.load(path, allow_pickle=False) as z:
        want = ('q_pid', 'first_rank') + tuple(cols)
        assert set(want) <= set(z.files), f'missing score columns: {path}'
        out = {k: z[k] for k in want}
    pid, rank = out['q_pid'], out['first_rank']
    assert pid.ndim == 1 and pid.dtype == np.int64, f'q_pid: {path}'
    _expected(set_name, direction, pid)
    assert rank.shape == pid.shape and rank.dtype == np.float32, f'first_rank: {path}'
    ok = np.isfinite(rank)
    assert not np.isinf(rank).any() and ((rank[ok] >= 1) & (rank[ok] == np.floor(rank[ok]))).all(), f'ranks: {path}'
    for k in cols:
        ap = out[k]
        assert ap.shape == pid.shape and ap.dtype == np.float32, f'{k}: {path}'
        ok = np.isfinite(ap)
        assert not np.isinf(ap).any() and ((ap[ok] >= 0) & (ap[ok] <= 1)).all(), f'{k}: {path}'
    return out


class Reader:
    """Reads score files for one part. Absent file: crash, except under smoke, where it is listed in `missing`."""

    def __init__(self):
        self.missing = []

    def score(self, set_name, kind, row, head, seed, direction, readout, cols=()):
        path = P.score_path(set_name, kind, row, head, seed, direction, readout)
        if not path.exists():
            if not P.SMOKE:
                raise FileNotFoundError(path)
            self.missing.append(str(path.relative_to(P.OUT)))
            return None
        return read_score(path, set_name, direction, cols)

    def seeds(self, set_name, kind, row, head, direction, readout, cols=()):
        """The row's files over its seeds, in seed order, all checked to share q_pid; None when any is missing."""
        out = [self.score(set_name, kind, row, head, s, direction, readout, cols) for s in seeds_of(row)]
        if any(o is None for o in out):
            return None
        for o in out[1:]:
            assert np.array_equal(o['q_pid'], out[0]['q_pid']), f'q_pid differs across seeds: {row} {head} {direction}'
        return out


def same_queries(*groups):
    """Cheap assert: all score files of one comparison have the same q_pid."""
    ref = groups[0][0]['q_pid']
    for g in groups:
        for s in g:
            assert np.array_equal(s['q_pid'], ref), 'q_pid differs between rows of one comparison'


class Context:
    def __init__(self, recs, device):
        self.recs, self.device, self._frames = recs, device, {}

    def sets(self, names):
        """The sets of `names` whose group has a stage 1 record; a group without one is void (§6)."""
        return [s for s in names if P.GROUP[s] in self.recs]

    def frame(self, set_name, kind):
        if (set_name, kind) not in self._frames:
            self._frames[set_name, kind] = tcl_data.frame(set_name, kind)
        return self._frames[set_name, kind]

    def has_kind(self, R, set_name, kind):
        """True when the stage 1 record has the kind. Under smoke a record may hold fewer kinds: listed in `missing`."""
        ok = kind in self.recs[P.GROUP[set_name]]['picks']
        if not ok and not P.SMOKE:
            raise KeyError(f'stage 1 record has no kind {kind}')
        if not ok:
            R.missing.append(f'stage1_{P.GROUP[set_name]}.json: kind {kind}')
        return ok

    def pick(self, set_name, kind, row, aspect, direction):
        return self.recs[P.GROUP[set_name]]['picks'][kind][row][aspect][direction]

    def pair_pick(self, set_name, kind, direction):
        pp = self.recs[P.GROUP[set_name]]['pair_picks'][kind][direction]
        return pp['late']['head'], pp['late']['readout'], pp['plain']['readout']


def result(n, name, R, sections, **extra):
    return dict(part=n, name=name, smoke=bool(P.SMOKE), **extra, sections=sections, missing=sorted(set(R.missing)))


def bar_kind(set_name):
    return P.BAR_KIND[P.GROUP[set_name]]


# ------------------------------------------------------------------------------------------------ part 1
def sample_cosines(q, g, idx, self_match, device):
    """Raw cosines [len(idx), n_gallery] (rows are unit) of the sampled queries against the whole gallery; for text to text
    the query's own column is dropped. Returns a torch tensor on `device`."""
    import torch
    qt = torch.as_tensor(q[idx], dtype=torch.float32, device=device)
    gt = torch.as_tensor(g, dtype=torch.float32, device=device)
    sim = qt @ gt.T
    if self_match:
        keep = torch.ones_like(sim, dtype=torch.bool)
        keep[torch.arange(len(idx), device=device), torch.as_tensor(idx, device=device)] = False
        sim = sim[keep].view(len(idx), -1)
    return sim


def spreads(sim):
    """(pooled std over all sampled cosines, mean over queries of each query's std), float64, population std."""
    s = sim.double()
    return float(s.std(unbiased=False)), float(s.std(dim=1, unbiased=False).mean())


def part1_cosines(ctx, set_name, kind, sections):
    fr = ctx.frame(set_name, kind)
    dev = ctx.device
    plain = dict(zip('ti', tcl_forward.rows(set_name, kind, 'plain', 'all', 'x', dev)))
    idx = {s: sample_queries(n) for s, n in (('t', fr['n_t']), ('i', fr['n_i']))}
    plain_cos = {}
    for d in P.DIRECTIONS:
        sim = sample_cosines(plain[d[0]], plain[d[2]], idx[d[0]], d == 't2t', dev)
        plain_cos[d] = dict(std=spreads(sim), ranks=row_ranks(sim.cpu().numpy()))
        del sim
    per = {(h, d): dict(std=[], within=[], ratio=[], ratio_within=[], corr=[]) for h in P.ASPECTS for d in P.DIRECTIONS}
    for head in P.ASPECTS:
        for seed in P.SEEDS:
            cstar = dict(zip('ti', tcl_forward.rows(set_name, kind, 'cstar', head, seed, dev)))
            for d in P.DIRECTIONS:
                sim = sample_cosines(cstar[d[0]], cstar[d[2]], idx[d[0]], d == 't2t', dev)
                std, within = spreads(sim)
                p_std, p_within = plain_cos[d]['std']
                corr = row_pearson(plain_cos[d]['ranks'], row_ranks(sim.cpu().numpy()))
                cell = per[head, d]
                cell['std'].append(std)
                cell['within'].append(within)
                cell['ratio'].append(spread_ratio(p_std, std))
                cell['ratio_within'].append(spread_ratio(p_within, within))
                cell['corr'].append(quartiles(corr))
                del sim
            del cstar
    for head in P.ASPECTS:
        for d in P.DIRECTIONS:
            cell, (p_std, p_within) = per[head, d], plain_cos[d]['std']
            n_q = len(idx[d[0]])
            rec = meta(set_name, kind, 'cstar', head, d, 'r0', n_q, P.SEEDS)
            rec.update(a=P.A, sources=dict(cstar=dict(row='cstar', head=head), plain=dict(row='plain', head='all')),
                       std_plain=p_std, std_within_query_plain=p_within)
            put(rec, 'std_cstar', cell['std'], P.SEEDS)
            put(rec, 'std_within_query_cstar', cell['within'], P.SEEDS)
            put(rec, 'ratio', cell['ratio'], P.SEEDS)
            put(rec, 'ratio_within_query', cell['ratio_within'], P.SEEDS)
            sections['spread'].append(rec)
            rc = meta(set_name, kind, 'cstar', head, d, 'r0', n_q, P.SEEDS)
            rc.update(sources=dict(cstar=dict(row='cstar', head=head), plain=dict(row='plain', head='all')))
            for j, name in enumerate(('mean', 'q25', 'median', 'q75')):
                put(rc, 'spearman_' + name, [c[j] for c in cell['corr']], P.SEEDS)
            sections['rank_correlation'].append(rc)


def part1(ctx):
    R = Reader()
    sections = dict(pair_split=[], label_split=[], spread=[], rank_correlation=[])
    for set_name in ctx.sets(PART1_SETS):
        kind = bar_kind(set_name)
        for d in P.PAIR_DIRECTIONS:
            head, ro_late, ro_plain = ctx.pair_pick(set_name, kind, d)
            plain = R.score(set_name, kind, 'plain', 'all', 'x', d, ro_plain)
            cs = R.seeds(set_name, kind, 'cstar', head, d, ro_late)
            late = R.seeds(set_name, kind, 'late', head, d, ro_late)
            if plain is None or cs is None or late is None:
                continue
            same_queries([plain], cs, late)
            hp = hit(plain['first_rank'])
            hc = [hit(s['first_rank']) for s in cs]
            hl = [hit(s['first_rank']) for s in late]
            rec = meta(set_name, kind, None, head, d, None, len(hp), P.SEEDS)
            rec.update(cell='pair', sources=dict(cstar=dict(row='cstar', head=head, readout=ro_late),
                                                 plain=dict(row='plain', head='all', readout=ro_plain),
                                                 late=dict(row='late', head=head, readout=ro_late)),
                       groups=average_groups([four_way(c, hp, l) for c, l in zip(hc, hl)]))
            for name, per_seed in (('cstar', hc), ('plain', [hp] * len(P.SEEDS)), ('late', hl)):
                put(rec, f'r10_{name}', [float(h.mean()) for h in per_seed], P.SEEDS)
            sections['pair_split'].append(rec)
        for aspect in P.ASPECTS:
            col = f'ap__{aspect}'
            for d in P.DIRECTIONS:
                ro = {r: ctx.pick(set_name, kind, r, aspect, d) for r in ('cstar', 'late', 'plain')}
                plain = R.score(set_name, kind, 'plain', 'all', 'x', d, ro['plain'], (col,))
                cs = R.seeds(set_name, kind, 'cstar', aspect, d, ro['cstar'], (col,))
                late = R.seeds(set_name, kind, 'late', aspect, d, ro['late'], (col,))
                if plain is None or cs is None or late is None:
                    continue
                same_queries([plain], cs, late)
                splits = [label_split(c[col], plain[col], l[col]) for c, l in zip(cs, late)]
                n_q = splits[0][2]
                assert all(s[2] == n_q for s in splits)
                rec = meta(set_name, kind, None, aspect, d, None, n_q, P.SEEDS)
                rec.update(cell='label', aspect=aspect,
                           sources=dict(cstar=dict(row='cstar', head=aspect, readout=ro['cstar']),
                                        plain=dict(row='plain', head='all', readout=ro['plain']),
                                        late=dict(row='late', head=aspect, readout=ro['late'])),
                           groups=average_groups([s[0] for s in splits]))
                for name in ('cstar', 'plain', 'late'):
                    put(rec, f'median_ap_{name}', [s[1][name] for s in splits], P.SEEDS)
                sections['label_split'].append(rec)
        part1_cosines(ctx, set_name, kind, sections)
    return result(1, 'part1_why_a', R, sections, a=P.A, n_sample=N_SAMPLE, sample='default_rng(0) choice without replacement, '
                  'per query side; the same queries for c* and plain', definitions=dict(
        pair_choice='c* at the late fusion pair pick head and read-out; plain at its pair pick; late fusion at its pair pick',
        average='shares and late rate per seed, then averaged over seeds 0 to 4; an empty group has no rate in that seed',
        label='c* and late at their picked read-outs for the aspect and direction, plain at its own; above = AP > own median '
              'over scored queries of the cell',
        spread='std of raw cosines, sampled queries by the whole gallery; std_* pooled over all values, '
               'std_within_query_* the mean over queries of each query std; ratio = a std_plain / ((1 - a) std_cstar)',
        rank_correlation='Spearman (average ranks) between c* and plain cosines over the gallery, per sampled query; '
                         'mean and quartiles over queries, per seed, then the mean over seeds'))


# ------------------------------------------------------------------------------------------------ part 2
def curve_value(first_rank):
    return summarize_pair(first_rank.astype(np.float64))['r10']


def part2(ctx):
    R = Reader()
    labels_rec, pair_rec = [], []
    checks = dict(tolerance=SELF_CHECK_TOL, n_checked=0, max_abs_diff=0.0, a=P.A)
    n_rank, started = 0, time.monotonic()
    for set_name in ctx.sets(('art_val', 'mm_val')):
        kind = bar_kind(set_name)
        fr = ctx.frame(set_name, kind)
        pid = dict(t=fr['tp'], i=fr['ip'])
        plain_t, plain_i = tcl_forward.rows(set_name, kind, 'plain', 'all', 'x', ctx.device)
        value = {}      # (a, head, direction, 'label' or 'pair') -> {seed: (value, n_query, readout)}
        for head in P.ASPECTS:
            lab = fr['labels'][head]
            for seed in CURVE_SEEDS:
                c_t, c_i = tcl_forward.rows(set_name, kind, 'cstar', head, seed, ctx.device)
                for a in A_GRID:
                    t, i = tcl_forward.fuse(c_t, plain_t, a), tcl_forward.fuse(c_i, plain_i, a)
                    mt, mi = fit_means(t, i)
                    rows = dict(t=t, i=i)
                    for d in P.DIRECTIONS:
                        label_ro = ctx.pick(set_name, kind, 'late', head, d)
                        pair_head, pair_ro, _ = (ctx.pair_pick(set_name, kind, d) if d in P.PAIR_DIRECTIONS
                                                 else (None, None, None))
                        want = {label_ro} | ({pair_ro} if pair_head == head else set())
                        entry = {head: dict(q=lab[d[0]], g=lab[d[2]])}
                        for ro in sorted(want):
                            res = evaluate(rows[d[0]], rows[d[2]], pid[d[0]], pid[d[2]], entry if ro == label_ro else {},
                                           d == 't2t', ro, ctx.device, block=tcl_score.BLOCK,
                                           means=tcl_score.query_gallery_means(d, mt, mi))
                            n_rank += 1
                            if ro == label_ro:
                                ap = res['ap'][head]
                                value.setdefault((a, head, d, 'label'), {})[seed] = (mean_ap(ap), n_scored(ap), ro)
                            if pair_head == head and ro == pair_ro:
                                value.setdefault((a, head, d, 'pair'), {})[seed] = (
                                    curve_value(res['first_rank']), len(res['first_rank']), ro)
                            if a == P.A:
                                _self_check(R, checks, set_name, kind, head, seed, d, ro, ro == label_ro,
                                            pair_head == head and ro == pair_ro, res)
                print(f'part 2: {set_name} {head} seed {seed} done, {n_rank} rankings, '
                      f'{(time.monotonic() - started) / 60:.1f} min', flush=True)
        for (a, head, d, cell), by_seed in sorted(value.items(), key=lambda kv: (kv[0][0], P.ASPECTS.index(kv[0][1]),
                                                                              P.DIRECTIONS.index(kv[0][2]), kv[0][3])):
            vals = [by_seed[s] for s in CURVE_SEEDS]
            assert len({v[1] for v in vals}) == 1 and len({v[2] for v in vals}) == 1
            rec = meta(set_name, kind, 'late', head, d, vals[0][2], vals[0][1], CURVE_SEEDS)
            rec.update(a=a)
            if cell == 'label':
                rec.update(cell='label', aspect=head)
                put(rec, 'map', [v[0] for v in vals], CURVE_SEEDS)
                labels_rec.append(rec)
            else:
                rec.update(cell='pair')
                put(rec, 'r10', [v[0] for v in vals], CURVE_SEEDS)
                pair_rec.append(rec)
    minutes = (time.monotonic() - started) / 60
    return result(2, 'part2_a_curve', R, dict(label=labels_rec, pair=pair_rec), a_grid=list(A_GRID),
                  seeds=list(CURVE_SEEDS), rankings_run=n_rank, minutes=minutes, self_check=checks,
                  definitions=dict(
                      label='late vector of that a (tcl_forward.fuse), means refitted on it over every validation row, '
                            'ranked by tcl_metrics.evaluate at the picked read-out of the late row for the aspect and direction; '
                            'map = mean of finite AP',
                      pair='first hit rank at most 10, at the late fusion pair pick head and read-out per direction'))


def _self_check(R, checks, set_name, kind, head, seed, d, ro, is_label, is_pair, res):
    """At a = 0.7 the curve's value must equal the stored late fusion score file's, within 1e-6 (skipped, and listed in
    `missing`, when the file is absent under smoke)."""
    cols = (f'ap__{head}',) if is_label else ()
    stored = R.score(set_name, kind, 'late', head, seed, d, ro, cols)
    if stored is None:
        return
    assert len(stored['q_pid']) == len(res['first_rank'])
    diffs = []
    if is_label:
        diffs.append(abs(mean_ap(stored[cols[0]]) - mean_ap(res['ap'][head])))
    if is_pair:
        diffs.append(abs(curve_value(stored['first_rank']) - curve_value(res['first_rank'])))
    for diff in diffs:
        assert diff <= SELF_CHECK_TOL, f'a = 0.7 differs from the stored late file: {set_name} {head} s{seed} {d} {ro}: {diff}'
        checks['n_checked'] += 1
        checks['max_abs_diff'] = max(checks['max_abs_diff'], diff)


# ------------------------------------------------------------------------------------------------ part 3
def pair_record(set_name, kind, row, head, direction, readout, scs):
    summaries = [summarize_pair(s['first_rank'].astype(np.float64)) for s in scs]
    assert len({s['n_query'] for s in summaries}) == 1
    seeds = seeds_of(row)
    rec = meta(set_name, kind, row, head, direction, readout, summaries[0]['n_query'], seeds)
    for key, name in (('r1', 'r1'), ('r5', 'r5'), ('r10', 'r10'), ('medr', 'median_rank')):
        put(rec, name, [s[key] for s in summaries], seeds)
    return rec


def part3(ctx):
    R = Reader()
    records = []
    for set_name in ctx.sets(P.SETS):
        kind = bar_kind(set_name)
        art = P.GROUP[set_name] == 'art'
        units = list(dict.fromkeys((row, head) for row, head, _ in tcl_score.plan(set_name)))
        for row, head in units:
            _, directions = tcl_score._plan(row, head)
            for d in directions:
                if d == 't2t' and not art:
                    continue                      # MMArt has no text to text pair (spec §3.4)
                for ro in P.READOUTS:
                    scs = R.seeds(set_name, kind, row, head, d, ro)
                    if scs is not None:
                        records.append(pair_record(set_name, kind, row, head, d, ro, scs))
    return result(3, 'part3_pair_all_rows', R, dict(pair=records), definitions=dict(
        pair='R@k = share of queries with first hit rank at most k (no hit counts as a miss); median_rank over queries with a '
             'hit; text to text on ArtELingo sets only'))


# ------------------------------------------------------------------------------------------------ part 4
def class_names(set_name):
    import mt_paths
    return list(tcl_data.r22_names('emotion')) if P.GROUP[set_name] == 'art' else list(mt_paths.CLASSES['emotion'])


def per_class_ap(ap, labels, n_class):
    """Per class: mean AP over the scored queries of that class and their count (nan, 0 for none)."""
    assert ap.shape == labels.shape
    ok = np.isfinite(ap)
    out = []
    for c in range(n_class):
        m = ok & (labels == c)
        out.append((float(ap[m].mean(dtype=np.float64)) if m.any() else float('nan'), int(m.sum())))
    return out


def part4(ctx):
    R = Reader()
    cells, classes = [], []
    for set_name in ctx.sets(P.SETS):
        kind = bar_kind(set_name)
        fr = ctx.frame(set_name, kind)
        for row in P.BAR_ROWS:
            seeds = seeds_of(row)
            for head in heads_of(row):
                for aspect in ((head,) if row in P.HEAD_ROWS else P.ASPECTS):
                    cols = tuple(f'ap__{aspect}{e}' for e in ENTRIES)
                    for d in P.DIRECTIONS:
                        q_labels = fr['labels'][aspect][d[0]]
                        for ro in P.READOUTS:
                            scs = R.seeds(set_name, kind, row, head, d, ro, cols)
                            if scs is None:
                                continue
                            for entry, col in zip(ENTRIES, cols):
                                aps = [s[col] for s in scs]
                                n_q = {n_scored(a) for a in aps}
                                assert len(n_q) == 1, f'scored queries differ across seeds: {row} {head} {d} {ro} {col}'
                                rec = meta(set_name, kind, row, head, d, ro, n_q.pop(), seeds)
                                rec.update(aspect=aspect, entry=entry[2:] or 'plain')
                                put(rec, 'map', [mean_ap(a) for a in aps], seeds)
                                put(rec, 'macro_map', [macro_map(a, q_labels) for a in aps], seeds)
                                cells.append(rec)
                if head in ('emotion', 'all'):
                    ro = ctx.pick(set_name, kind, row, 'emotion', 't2i')
                    scs = R.seeds(set_name, kind, row, head, 't2i', ro, ('ap__emotion',))
                    if scs is not None:
                        names = class_names(set_name)
                        lab = fr['labels']['emotion']['t']
                        per = [per_class_ap(s['ap__emotion'], lab, len(names)) for s in scs]
                        assert all(p[c][1] == per[0][c][1] for p in per for c in range(len(names)))
                        for c, name in enumerate(names):
                            rec = meta(set_name, kind, row, head, 't2i', ro, per[0][c][1], seeds)
                            rec.update(aspect='emotion', entry='plain', class_index=c, class_name=name)
                            put(rec, 'ap', [p[c][0] for p in per], seeds)
                            classes.append(rec)
    return result(4, 'part4_macro_cells_leak', R, dict(cells=cells, emotion_t2i_per_class=classes), definitions=dict(
        map='mean of finite AP', macro_map='tcl_metrics.macro_map: mean over classes of the class mean AP',
        entry='plain = the aspect; no_leak = texts that name their own label leave the text side; '
              'cross_painting = the query\'s own painting leaves the gallery',
        per_class='emotion text to image, plain entry, at the row\'s picked read-out; AP mean per class over scored queries'))


# ------------------------------------------------------------------------------------------------ part 5
def part5(ctx):
    R = Reader()
    label, pair = [], []
    set_name = 'art_held'
    kind = bar_kind(set_name)
    for row in SIDE_OF_ART:
        seeds = seeds_of(row)
        for head in heads_of(row):
            aspects, directions = tcl_score._plan(row, head)
            for aspect in aspects:
                for d in directions:
                    ro = ctx.pick(set_name, kind, row, aspect, d)
                    scs = R.seeds(set_name, kind, row, head, d, ro, (f'ap__{aspect}',))
                    if scs is None:
                        continue
                    rec = meta(set_name, kind, row, head, d, ro, n_scored(scs[0][f'ap__{aspect}']), seeds)
                    rec.update(aspect=aspect)
                    put(rec, 'map', [mean_ap(s[f'ap__{aspect}']) for s in scs], seeds)
                    label.append(rec)
            for d in P.PAIR_DIRECTIONS:
                if d not in directions:
                    continue                      # roberta: text to text only
                for ro in P.READOUTS:
                    scs = R.seeds(set_name, kind, row, head, d, ro)
                    if scs is not None:
                        rec = meta(set_name, kind, row, head, d, ro, len(scs[0]['q_pid']), seeds)
                        put(rec, 'r10', [summarize_pair(s['first_rank'].astype(np.float64))['r10'] for s in scs], seeds)
                        pair.append(rec)
    return result(5, 'part5_side_rows', R, dict(label=label, pair=pair), definitions=dict(
        label='map at the stage 1 pick of the row, aspect and direction',
        pair='R@10 at each of the three read-outs (side rows have no pair pick); roberta has no pair file'))


# ------------------------------------------------------------------------------------------------ part 6
def part6(ctx):
    R = Reader()
    label, pair, no_leak, share = [], [], [], []
    set_name = 'mm_test'
    others = tuple(k for k in P.KINDS if k != P.BAR_KIND['mm'])
    rows = ('cstar', 'plain', 'teacher', 'late')
    for kind in others:
        if not ctx.has_kind(R, set_name, kind):
            continue
        for row in rows:
            seeds = seeds_of(row)
            for aspect in P.ASPECTS:
                for d in P.DIRECTIONS:
                    ro = ctx.pick(set_name, kind, row, aspect, d)
                    col = f'ap__{aspect}'
                    h = aspect if row in P.HEAD_ROWS else 'all'
                    scs = R.seeds(set_name, kind, row, h, d, ro, (col,))
                    if scs is None:
                        continue
                    rec = meta(set_name, kind, row, h, d, ro, n_scored(scs[0][col]), seeds)
                    rec.update(aspect=aspect)
                    put(rec, 'map', [mean_ap(s[col]) for s in scs], seeds)
                    label.append(rec)
        for d in P.PAIR_DIRECTIONS:
            head, ro_late, ro_plain = ctx.pair_pick(set_name, kind, d)
            for row, ro in (('cstar', ro_late), ('late', ro_late), ('plain', ro_plain)):
                h = 'all' if row == 'plain' else head
                scs = R.seeds(set_name, kind, row, h, d, ro)
                if scs is not None:
                    rec = meta(set_name, kind, row, h, d, ro, len(scs[0]['q_pid']), seeds_of(row))
                    put(rec, 'r10', [summarize_pair(s['first_rank'].astype(np.float64))['r10'] for s in scs], seeds_of(row))
                    pair.append(rec)
            for head in P.ASPECTS:
                for ro in P.READOUTS:
                    scs = R.seeds(set_name, kind, 'teacher', head, d, ro)
                    if scs is not None:
                        rec = meta(set_name, kind, 'teacher', head, d, ro, len(scs[0]['q_pid']), seeds_of('teacher'))
                        put(rec, 'r10', [summarize_pair(s['first_rank'].astype(np.float64))['r10'] for s in scs],
                            seeds_of('teacher'))
                        pair.append(rec)
    for kind in (P.BAR_KIND['mm'], 'unified'):
        if not ctx.has_kind(R, set_name, kind):
            continue
        for row in rows:
            seeds = seeds_of(row)
            for aspect in P.ASPECTS:
                h = aspect if row in P.HEAD_ROWS else 'all'
                for d in P.DIRECTIONS:
                    ro = ctx.pick(set_name, kind, row, aspect, d)
                    col = f'ap__{aspect}__no_leak'
                    scs = R.seeds(set_name, kind, row, h, d, ro, (col,))
                    if scs is None:
                        continue
                    rec = meta(set_name, kind, row, h, d, ro, n_scored(scs[0][col]), seeds)
                    rec.update(aspect=aspect, entry='no_leak')
                    put(rec, 'map', [mean_ap(s[col]) for s in scs], seeds)
                    no_leak.append(rec)
    for s in P.SETS:
        for kind in P.kinds(P.GROUP[s]):
            for aspect in P.ASPECTS:
                labelled = int((ctx.frame(s, kind)['labels'][aspect]['t'] >= 0).sum())
                share.append(dict(set=s, kind=kind, aspect=aspect, n_labelled=labelled,
                                  share=tcl_data.label_share(s, kind, aspect)))
    return result(6, 'part6_mmart_other_texts', R,
                  dict(label=label, pair=pair, no_leak=no_leak, label_word_share=share), definitions=dict(
                      label='map at the stage 1 pick of the row, kind, aspect and direction',
                      pair='R@10 for c* and late fusion at the late fusion pair pick of the kind, plain at its pair pick; '
                           'teacher at each head and read-out (no pair pick)',
                      no_leak='map of the no_leak entry at the same picks',
                      label_word_share='tcl_data.label_share: share of labelled rows whose text names its own label'))


# ------------------------------------------------------------------------------------------------ part 7
def part7(ctx):
    R = Reader()
    spreads_ = []
    for set_name in ctx.sets(('art_held', 'mm_test')):
        group = P.GROUP[set_name]
        for kind in P.kinds(group):
            if not ctx.has_kind(R, set_name, kind):
                continue
            for row in P.SEEDED_ROWS:
                if row == 'free' and group != 'art':
                    continue
                for aspect in P.ASPECTS:
                    col = f'ap__{aspect}'
                    for d in P.DIRECTIONS:
                        ro = ctx.pick(set_name, kind, row, aspect, d)
                        scs = R.seeds(set_name, kind, row, aspect, d, ro, (col,))
                        if scs is None:
                            continue
                        maps = [mean_ap(s[col]) for s in scs]
                        rec = meta(set_name, kind, row, aspect, d, ro, n_scored(scs[0][col]), P.SEEDS)
                        put(rec, 'map', maps, P.SEEDS)
                        rec.update(aspect=aspect, seed_spread=max(maps) - min(maps),
                                   stage1_range=ctx.recs[group]['ranges'][kind][row][aspect][d])
                        spreads_.append(rec)
    return result(7, 'part7_seed_spreads', R, dict(seed_spread=spreads_), definitions=dict(
        seed_spread='max minus min over seeds 0 to 4 of label mAP on the held set at the picked read-out',
        stage1_range='the same on the validation set, from stage 1\'s `ranges`, unrounded'))


# ------------------------------------------------------------------------------------------------ part 8
def part8(ctx):
    import tcl_extract
    R = Reader()
    stores = []
    for key, s, k, i, r, path, _ in tcl_extract.stores_to_check():
        done = path / 'done.json'
        if not done.exists():
            if not P.SMOKE:
                raise FileNotFoundError(done)
            R.missing.append(str(done))
            continue
        d = json.loads(done.read_text())
        for field in ('count', 'n_truncated', 'longest_tokens', 'token_limit', 'smoke'):
            assert field in d, f'{done} lacks {field}'
        stores.append(dict(key=key, set=s, kind=k, row=r, count=d['count'], n_truncated=d['n_truncated'],
                           share_truncated=d['n_truncated'] / d['count'], longest_tokens=d['longest_tokens'],
                           token_limit=d['token_limit'], manifest_smoke=d['smoke']))
    return result(8, 'part8_truncation', R, dict(stores=stores), definitions=dict(
        stores='the 39 new stores of tcl_extract.stores_to_check, numbers from each done.json'))


PARTS = {1: part1, 2: part2, 3: part3, 4: part4, 5: part5, 6: part6, 7: part7, 8: part8}
NAMES = {1: 'part1_why_a', 2: 'part2_a_curve', 3: 'part3_pair_all_rows', 4: 'part4_macro_cells_leak', 5: 'part5_side_rows',
         6: 'part6_mmart_other_texts', 7: 'part7_seed_spreads', 8: 'part8_truncation'}


# ------------------------------------------------------------------------------------------------ output
def stage3_dir():
    return P.OUT / 'stage3'


def write_part(res, voided):
    folder = stage3_dir()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'{NAMES[res["part"]]}.json'
    path.write_text(json.dumps(clean(dict(res, voided=voided)), indent=1, allow_nan=False) + '\n')
    return path


def _f(x, n=4):
    return 'null' if x is None else f'{x:.{n}f}'


def write_summary():
    """OUT/stage3/summary.md from the part files that exist: counts and the headline tables, numbers only."""
    folder = stage3_dir()
    parts = {n: json.loads((folder / f'{name}.json').read_text()) for n, name in NAMES.items()
             if (folder / f'{name}.json').exists()}
    voided = []
    for p in parts.values():
        for entry in p['voided']:
            if entry not in voided:
                voided.append(entry)
    lines = ['# Stage 3 diagnostics: numbers only', '', f'smoke: {bool(P.SMOKE)}', '', '## Voided (§6)', '']
    lines += [f'- {v["what"]}: {v["key"]}: {v["why"]}' for v in voided] or ['none']
    lines += ['', '| part | file | records | missing |', '|---|---|---|---|']
    for n, p in parts.items():
        lines.append(f'| {n} | {NAMES[n]}.json | {sum(len(v) for v in p["sections"].values())} | {len(p["missing"])} |')
    if 1 in parts:
        lines += ['', '## Part 1: pair split, mean over seeds 0 to 4 (share / late fusion hit rate)', '',
                  '| set | direction | head | both | only c* | only plain | neither |', '|---|---|---|---|---|---|---|']
        for r in parts[1]['sections']['pair_split']:
            g = r['groups']
            lines.append(f'| {r["set"]} | {r["direction"]} | {r["head"]} | ' + ' | '.join(
                f'{_f(g[k]["share"])} / {_f(g[k]["late_rate"])}' for k in GROUPS4) + ' |')
        lines += ['', '## Part 1: a std_plain / ((1 - a) std_cstar), mean over seeds', '',
                  '| set | direction | head | ratio | ratio within query |', '|---|---|---|---|---|']
        for r in parts[1]['sections']['spread']:
            lines.append(f'| {r["set"]} | {r["direction"]} | {r["head"]} | {_f(r["ratio"])} | {_f(r["ratio_within_query"])} |')
    if 2 in parts:
        lines += ['', '## Part 2: pair R@10 of the a curve, mean over seeds 1 to 4', '',
                  '| set | direction | head | ' + ' | '.join(f'{a:.1f}' for a in A_GRID) + ' |',
                  '|---|---|---|' + '---|' * len(A_GRID)]
        by = {}
        for r in parts[2]['sections']['pair']:
            by.setdefault((r['set'], r['direction'], r['head']), {})[r['a']] = r['r10']
        for (s, d, h), v in by.items():
            lines.append(f'| {s} | {d} | {h} | ' + ' | '.join(_f(v.get(a)) for a in A_GRID) + ' |')
        lines += ['', f'self check at a = {P.A}: {parts[2]["self_check"]["n_checked"]} values, largest difference '
                      f'{parts[2]["self_check"]["max_abs_diff"]:.2e}; rankings run {parts[2]["rankings_run"]}']
    if 8 in parts:
        lines += ['', '## Part 8: truncated texts', '', '| store | count | truncated | longest tokens |', '|---|---|---|---|']
        for r in parts[8]['sections']['stores']:
            lines.append(f'| {r["key"]} | {r["count"]} | {r["n_truncated"]} | {r["longest_tokens"]} |')
    (folder / 'summary.md').write_text('\n'.join(lines) + '\n')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--device', default='cuda')
    p.add_argument('--parts', nargs='+', type=int, choices=sorted(PARTS), default=sorted(PARTS))
    args = p.parse_args(argv)
    groups = [g for g in ('art', 'mm') if P.record_path(f'stage1_{g}.json').exists()] or ['art', 'mm']
    for set_name in ('art_held', 'mm_test'):
        if P.GROUP[set_name] in groups:
            tcl_score.held_guard(set_name)
    if not P.SMOKE and not P.record_path('verdicts.json').exists():
        print(f'refused: {P.record_path("verdicts.json")} is absent; stage 3 runs after the verdicts', file=sys.stderr)
        return 2
    outcomes = P.OUT / 'stage2/outcomes.json'
    assert P.SMOKE or outcomes.is_file(), outcomes
    voided = json.loads(outcomes.read_text())['voided'] if outcomes.exists() else []
    recs = {g: P.read_committed_record(f'stage1_{g}.json') for g in groups}
    ctx = Context(recs, args.device)
    for n in sorted(set(args.parts)):
        if {5: 'art', 6: 'mm'}.get(n, groups[0]) not in groups:
            print(f'part {n}: skipped, its group has no stage 1 record', flush=True)
            continue
        started = time.monotonic()
        res = PARTS[n](ctx)
        path = write_part(res, voided)
        print(f'part {n}: {path.name}, {sum(len(v) for v in res["sections"].values())} records, {len(res["missing"])} missing, '
              f'{time.monotonic() - started:.1f}s', flush=True)
    write_summary()
    return 0


if __name__ == '__main__':
    sys.exit(main())
