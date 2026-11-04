"""r18 ticket 12: diagnostics (spec §3.6, D22). Descriptive: nothing here decides anything and no output carries an
outcome word. Model: r17's `ns_diag.py`.

Reads saved outputs only (partition files, TPIPS stores, CLAY vectors, stage 1 picks, r17's sweep files) and runs after
the labels. A missing input gives a null in that row with a reason (`missing` lists), never a crash, so a partial run
still writes every file. Files: OUT/diag/<name>.json, one per diagnostic. Each item is its own plain process
(`ns_clu.paired_cell` opens a fork pool: never call an item from a pool worker).

  sweep   [--set S]   AMI and ARI at r17's counts (plain + TURTLE, c* + TURTLE where fits exist, TPIPS k-means, CLAY
          k-means) with mean, min, max over starts or seeds, beside r17's k-means rows (plain and c*, from r17's sweep
          files); [3, n] arms retain all three start values -> sweep_<set>.json
  adds    [--set S]   X-c* + TURTLE against X-c* k-means, plain + TURTLE against plain k-means, at the class count (four
          metrics) and 20 (AMI): d with its paired interval over r17's draws; the note records independent c* seed draws
          -> turtle_adds_<set>.json
  fits    non-empty clusters per TURTLE (and TEMI, where run) fit; TURTLE's final loss and entropy -> fits.json
  clay    text set picked per task, both sets' validation scores, class names found verbatim in each list (D22:
          listed, not removed) -> clay.json
  tpips   mean cosine between an image's three factor vectors -> tpips_cosine.json
  r0      R0 rows (`none+k0`, held split) of TPIPS and CLAY beside their validation R0 -> r0_rows.json
Default: every item. Smoke: OM_SMOKE=1 OM_OUTPUTS=<folder>, with `--aspects/--encs/--counts` to cut the work.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import om_paths as op  # noqa: E402
import om_cells  # noqa: E402
import om_data  # noqa: E402

import ns_clu  # noqa: E402
import ns_clu_cells  # noqa: E402
import ns_metrics  # noqa: E402

DIAG = 'diag'
SOFT = (om_cells.MissingInput, FileNotFoundError, AssertionError, KeyError, OSError, ValueError)
BANNED = re.compile(r'\b(WIN|LEVEL|BELOW)\b|inconclusive')   # outcome words (spec §3.4): none may reach a file
ASPECTS = op.ASPECTS


def _out(name):
    return op.OUT / DIAG / name


def write(name, doc):
    text = json.dumps(doc, sort_keys=True)
    hit = BANNED.search(text)
    assert hit is None, f'{name}: outcome word {hit.group(0)!r} in a diagnostic'
    op.write_json(_out(name), doc)
    return doc


def _reason(e):
    return f'{type(e).__name__}: {str(e)[:300]}'


def _draws():
    return op.SMOKE_DRAWS['clu'] if op.SMOKE else op.N_DRAWS['clu']


def _picks():
    p = op.OUT / 'stage1' / 'om_picks.json'
    return json.loads(p.read_text()) if p.exists() else None


def _text_set(picks, kind, set_):
    """(text set, source note): stage 1's choice; 'A' with a note when the record is absent."""
    try:
        return picks[f'clay_textset/{kind}/{set_}']['set'], 'stage1'
    except (TypeError, KeyError):
        return 'A', 'no stage 1 pick: set A used'


def _ks(set_, aspect, counts):
    """The counts of the sweep: r17's grid of the aspect plus its class count; `counts` (smoke) cuts it."""
    ks = sorted(set(op.SWEEP_COUNTS[aspect]) | {op.CLASS_COUNT[(set_, aspect)]})
    return [k for k in ks if counts is None or k in counts]


def _stat(per):
    v = [float(x) for x in per]
    return dict(per_run=v, mean=float(np.mean(v)), min=min(v), max=max(v))


# ================================================================================================ count sweep
def _r17_rows(set_, enc):
    """r17's k-means rows (c*, plain) of the (set, encoder) sweep file, or (None, reason)."""
    f = op.R17_OUT / 'stage3' / 'sweep' / f'{set_}__{enc}.json'
    if not f.exists():
        return None, f'{f} is missing'
    doc = json.loads(f.read_text())
    return [r for r in doc['rows'] if r['arm'] in ('cstar', 'plain')], None


def sweep(set_, aspects=None, encs=None, counts=None, ld=None):
    ld = ld or om_cells.Loaders()
    aspects, encs = tuple(aspects or ASPECTS), tuple(encs or op.ENCODERS)
    split, picks = op.HELD_SPLIT[set_], _picks()
    clu_ts, clu_src = _text_set(picks, 'clu', set_)
    rows, missing, r17, plain_cache = [], [], {}, {}

    def add(arm, enc, aspect, k, parts, y, seeded):
        if not seeded:
            parts = np.asarray(parts)[:, None]
        for t in ns_clu.metric_table(arm, aspect, parts, y, k, metrics=op.SWEEP_METRICS):
            rows.append(dict(arm=arm, enc=enc, aspect=aspect, k=int(k), metric=t['metric'], seeded=seeded,
                             **_stat(t['per_seed'])))

    def skip(arm, enc, aspect, k, e):
        missing.append(dict(arm=arm, enc=enc, aspect=aspect, k=int(k), reason=_reason(e)))

    for aspect in aspects:
        y = np.asarray(ld.labels(set_, aspect))
        for k in _ks(set_, aspect, counts):
            for arm, loader in (('tpips', lambda: ld.tpips_parts(set_, aspect, k)),
                                ('clay', lambda: ld.clay_parts(set_, aspect, k, clu_ts))):
                try:
                    add(arm, None, aspect, k, loader(), y, False)
                except SOFT as e:
                    skip(arm, None, aspect, k, e)
            for enc in encs:
                try:
                    add('plain_turtle', enc, aspect, k, ld.turtle_plain(set_, enc, k), y, False)
                except SOFT as e:
                    skip('plain_turtle', enc, aspect, k, e)
                if k in op.label_ks(set_, aspect):      # c* + TURTLE exists at the class count and 20 only
                    try:
                        add('cstar_turtle', enc, aspect, k, om_cells.repeat_states(ld.turtle_cstar(set_, enc, aspect, k)),
                            y, True)
                    except SOFT as e:
                        skip('cstar_turtle', enc, aspect, k, e)
    for enc in encs:
        rows17, why = _r17_rows(set_, enc)
        r17[enc] = dict(rows=[r for r in (rows17 or []) if r['aspect'] in aspects and
                              (counts is None or r['k'] in counts)], missing=why)
    return write(f'sweep_{set_}.json', dict(
        set=set_, split=split, aspects=list(aspects), encoders=list(encs), counts_given=counts,
        clay_text_set=clu_ts, clay_text_set_source=clu_src, class_count={a: op.CLASS_COUNT[(set_, a)] for a in aspects},
        rows=rows, r17_kmeans=r17, missing=missing, smoke=bool(op.SMOKE),
        note='per_run: one value per start (plain, TPIPS, CLAY: k-means states or TURTLE starts) or per seed (c*)'))


# ================================================================================================ what TURTLE adds
def _cell_rows(cand, comp, y, k, metrics, draws):
    res = ns_clu.paired_cell(cand, comp, y, k, r=None, metrics=metrics, draws=draws)
    return {m: dict(turtle=v['cand'], kmeans=v['comp'], d=v['d'], lo=v['lo'], hi=v['hi'], n_draws=v['n_draws'])
            for m, v in res.items()}


def adds(set_, aspects=None, encs=None, counts=None, ld=None, draws=None):
    """d = TURTLE minus k-means, interval from r17's paired draws (`ns_clu.paired_cell`, r None). Plain + TURTLE ([3, n])
    is tiled to five identical 'seeds' to fit the paired form: its draws are over rows (and k-means states)."""
    ld, draws = ld or om_cells.Loaders(), draws or _draws()
    aspects, encs = tuple(aspects or ASPECTS), tuple(encs or op.ENCODERS)
    split, cells, missing, plain_km = op.HELD_SPLIT[set_], [], [], {}
    for aspect in aspects:
        y = np.asarray(ld.labels(set_, aspect))
        K = op.CLASS_COUNT[(set_, aspect)]
        for k in op.label_ks(set_, aspect):
            if counts is not None and k not in counts:
                continue
            metrics = op.CLASS_METRICS if k == K else ('ami',)
            for enc in encs:
                for arm in ('cstar_turtle', 'plain_turtle'):
                    cell = dict(set=set_, enc=enc, aspect=aspect, k=int(k), arm=arm,
                                against='cstar k-means' if arm == 'cstar_turtle' else 'plain k-means',
                                is_class_count=bool(k == K), result=None, reason=None)
                    try:
                        if arm == 'cstar_turtle':
                            cand = om_cells.repeat_states(ld.turtle_cstar(set_, enc, aspect, k))
                            comp = ld.cstar_parts(set_, enc, aspect, k)
                        else:
                            t = ld.turtle_plain(set_, enc, k)
                            cand = np.repeat(np.asarray(t)[None], len(op.SEEDS), axis=0)
                            if (enc, k) not in plain_km:
                                plain_km[(enc, k)] = ns_clu_cells._comp_parts(set_, split, enc, aspect, 'plain', k)
                            comp = plain_km[(enc, k)]
                        cell['result'] = _cell_rows(cand, comp, y, k, metrics, draws)
                    except SOFT as e:
                        cell['reason'] = _reason(e)
                        missing.append(f'{arm}/{enc}/{aspect}/{k}')
                    cells.append(cell)
    return write(f'turtle_adds_{set_}.json', dict(set=set_, split=split, cells=cells, missing=missing, draws=draws,
                                                  smoke=bool(op.SMOKE), sign='d = TURTLE minus k-means',
                                                  note="For c* + TURTLE against c* k-means, c*'s seeds are drawn twice, "
                                                       'so the pair is on rows only and the interval is probably wider '
                                                       '(size not measured).'))


# ================================================================================================ fits
def _fit_rows(method, set_, enc, arm, aspect, k):
    starts = op.TURTLE_PLAIN_STARTS if arm == 'plain' else (op.TURTLE_CSTAR_START,)
    seeds = (None,) if arm == 'plain' else op.SEEDS
    side, why = [], []
    for st in starts:
        for sd in seeds:
            p = om_data._path(None, method, set_, enc, arm, om_data._name(arm, k, aspect, st, sd)).with_suffix('.json')
            if not p.exists():
                why.append(f'{p.name} missing')
                continue
            side.append(json.loads(p.read_text()))
    row = dict(method=method, set=set_, enc=enc, arm=arm, aspect=aspect, K=int(k), n_fits=len(side),
               n_expected=len(starts) * len(seeds), missing=why)
    for f in ('n_clusters', 'loss', 'entropy'):
        v = [s.get(f) for s in side]
        row[f] = v if v and all(x is not None for x in v) else None
        if row[f] is not None:
            row[f + '_mean'], row[f + '_min'], row[f + '_max'] = float(np.mean(v)), min(v), max(v)
    row['empty_clusters_max'] = (max(int(k) - c for c in row['n_clusters']) if row['n_clusters'] else None)
    row['seconds_total'] = float(sum(s.get('seconds') or 0 for s in side)) if side else None
    return row


def fits(sets=None, encs=None):
    sets, encs = tuple(sets or op.SETS), tuple(encs or op.ENCODERS)
    rows = []
    for method in ('turtle', 'temi'):
        for s in sets:
            for e in encs:
                for arm in om_data.ARMS:
                    pairs = ([(None, k) for k in op.plain_ks(s)] if arm == 'plain' else
                             [(a, k) for a in ASPECTS for k in op.label_ks(s, a)])
                    if method == 'temi' and not (op.OUT / 'parts' / 'temi' / s / e / arm).exists():
                        continue            # TEMI is a later build: rows only where it ran
                    rows += [_fit_rows(method, s, e, arm, a, k) for a, k in pairs]
    return write('fits.json', dict(rows=rows, smoke=bool(op.SMOKE),
                                   n_rows_without_fits=sum(1 for r in rows if r['n_fits'] == 0)))


# ================================================================================================ CLAY
def _norm(s):
    return re.sub(r'\s+', ' ', str(s).lower().replace('_', ' ')).strip()


def class_names(set_, aspect):
    """Class names of (set, aspect): MMArt from r15's `mt_paths.CLASSES`; ArtELingo emotion from r10's `rs_vh.EMOTIONS`,
    style from the sorted `art_style` values of the annotations, genre = the 10 WikiArt genres of `mt_paths.CLASSES`."""
    import mt_paths
    if set_ == 'mmart' or aspect == 'genre':
        return list(mt_paths.CLASSES[aspect])
    import rs_vh
    if aspect == 'emotion':
        return list(rs_vh.EMOTIONS)
    rows = json.loads(Path(rs_vh.ANNOTATIONS).read_text())
    return sorted({r['art_style'] for r in rows})


def _lists(text_set):
    p = op.CLAY_LISTS_A if text_set == 'A' else op.CLAY_B_DIR / 'lists.json'
    if not Path(p).exists():
        return None, f'{p} is missing'
    return json.loads(Path(p).read_text())['aspects'], None


def clay(sets=None):
    sets, picks = tuple(sets or op.SETS), _picks()
    doc = dict(picks_found=picks is not None, per_set={}, class_names_in_lists={}, smoke=bool(op.SMOKE))
    for s in sets:
        entry = {}
        for kind in ('clu', 'ret'):
            rec = (picks or {}).get(f'clay_textset/{kind}/{s}')
            entry[kind] = (None if rec is None else
                           dict(chosen=rec['set'], scores=rec.get('scores'), reason=rec.get('reason'),
                                margin=rec.get('margin')))
            if rec is None:
                entry[kind + '_reason'] = 'no stage 1 record' if picks is None else f'key clay_textset/{kind}/{s} absent'
        entry['pick'] = (picks or {}).get(f'clay/{s}', {}).get('pick')
        doc['per_set'][s] = entry
    for ts in ('A', 'B'):
        lists, why = _lists(ts)
        res = {}
        for s in sets:
            for a in ASPECTS:
                key = f'{s}/{a}'
                if lists is None or a not in lists:
                    res[key] = dict(reason=why or f'aspect {a} absent from the lists')
                    continue
                try:
                    names = class_names(s, a)
                except Exception as e:      # noqa: BLE001  (class names come from older rounds' modules)
                    res[key] = dict(reason=_reason(e))
                    continue
                vals = lists[a].get('values') or []
                exact = [n for n in names if n in vals]
                loose = [n for n in names if _norm(n) in {_norm(v) for v in vals}]
                res[key] = dict(n_classes=len(names), n_values=len(vals), fallback=lists[a].get('fallback'),
                                exact=exact, normalised=loose, n_exact=len(exact), n_normalised=len(loose),
                                note='normalised: lower case, underscores as spaces')
        doc['class_names_in_lists'][ts] = res
    return write('clay.json', doc)


# ================================================================================================ TPIPS
def tpips_cosine(sets=None, aspects=None):
    """Mean cosine between an image's three factor vectors (rows are unit, so the dot product): the mean over images of
    the mean over the three pairs, and each pair's mean. Splits: the set's held split and the development split."""
    sets = tuple(sets or op.SETS)
    rows = []
    for s in sets:
        for split in (op.HELD_SPLIT[s], op.DEV_SPLIT):
            row = dict(set=s, split=split, mean_cosine=None, pairs=None, reason=None)
            try:
                x = [om_data.tpips(s, split, a) for a in ASPECTS]
                pairs = {f'{ASPECTS[i]}-{ASPECTS[j]}': np.einsum('nd,nd->n', x[i].astype(np.float64), x[j].astype(np.float64))
                         for i, j in ((0, 1), (0, 2), (1, 2))}
                per_image = np.mean(list(pairs.values()), axis=0)
                row.update(mean_cosine=float(per_image.mean()), min_image=float(per_image.min()),
                           max_image=float(per_image.max()), n=int(len(per_image)),
                           pairs={k: float(v.mean()) for k, v in pairs.items()})
            except SOFT as e:
                row['reason'] = _reason(e)
            rows.append(row)
    return write('tpips_cosine.json', dict(rows=rows, smoke=bool(op.SMOKE)))


# ================================================================================================ R0 rows
def r0_rows(sets=None, aspects=None, ld=None):
    """Held-split R0 mAP (`none+k0`, r17's key) per aspect of TPIPS and CLAY (set A and B vectors where they exist),
    beside the validation R0 and pick that stage 1 recorded. Retrieval read-out as `om_cells.Loaders.ap`."""
    ld, sets, aspects = ld or om_cells.Loaders(), tuple(sets or op.SETS), tuple(aspects or ASPECTS)
    picks, rows = _picks(), []
    for s in sets:
        for arm, ts in (('tpips', None), ('clay', 'A'), ('clay', 'B')):
            row = dict(set=s, arm=arm, text_set=ts, split=op.HELD_SPLIT[s], aspects=None, reason=None, stage1=None)
            rec = (picks or {}).get(f'{arm}/{s}')
            if rec is not None and (ts is None or rec.get('text_set') == ts):
                row['stage1'] = dict(pick=rec.get('pick'), r0_map=rec['scores'].get('r0_map'), map=rec['scores'].get('map'))
            elif picks is not None and ts is not None:
                try:
                    sc = picks[f'clay_textset/ret/{s}']['scores'][ts]
                    row['stage1'] = dict(pick=sc['pick'], r0_map=sc['r0_map'], map=sc['map'])
                except (KeyError, TypeError):
                    pass
            try:
                out = {}
                for a in aspects:
                    ap = ld.ap(arm, s, 'clip_l14' if arm == 'clay' else 'qwen', a, 'none+k0', ts)
                    out[a] = dict(mean=float(ap['ap'].mean()), n_queries=int(len(ap['rows'])))
                row['aspects'] = out
            except SOFT as e:
                row['reason'] = _reason(e)
            rows.append(row)
    return write('r0_rows.json', dict(rows=rows, pick='none+k0', smoke=bool(op.SMOKE)))


# ================================================================================================ CLI
ITEMS = ('sweep', 'adds', 'fits', 'clay', 'tpips', 'r0')


def _list(s, cast=str):
    return None if s is None else tuple(cast(x) for x in s.split(','))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('items', nargs='*', choices=list(ITEMS) + ['all'], default=['all'])
    ap.add_argument('--set', dest='sets')
    ap.add_argument('--aspects')
    ap.add_argument('--encs')
    ap.add_argument('--counts', help='smoke: only these cluster counts (comma list)')
    a = ap.parse_args(argv)
    items = ITEMS if 'all' in a.items else a.items
    sets, aspects, encs, counts = _list(a.sets) or op.SETS, _list(a.aspects), _list(a.encs), _list(a.counts, int)
    t0 = time.monotonic()
    for it in items:
        t1 = time.monotonic()
        if it in ('sweep', 'adds'):
            ld = om_cells.Loaders()
            for s in sets:
                (sweep if it == 'sweep' else adds)(s, aspects, encs, counts, ld)
        elif it == 'fits':
            fits(sets, encs)
        elif it == 'clay':
            clay(sets)
        elif it == 'tpips':
            tpips_cosine(sets)
        else:
            r0_rows(sets, aspects)
        print(f'{it}: {time.monotonic() - t1:.0f}s -> {_out("")}', flush=True)
    print(f'done in {time.monotonic() - t0:.0f}s')


if __name__ == '__main__':
    main()
