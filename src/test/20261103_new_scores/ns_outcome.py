"""r17 outcome code (ticket 02): the 412 claim cells, bars, outcome words, voids (§3.5's table) and licensed sentences.

Spec: docs/superpowers/specs/2026-11-03-new-scores-design.md, §6 (table, per cell, inconclusive, licensed claims),
§3.4 "Outcomes" (WIN lo > 0; LEVEL not WIN and lo > -r; BELOW otherwise; no multiplicity correction), the gate table of
§3.5 (D18), D4, D5, D15. The word rule is r11's `rt_stats.outcome` (imported). Model: r15's `mt_outcome`.

CLI (reads files only, no GPU):
  python ns_outcome.py decide [--root OUT] [--out FILE]
    reads  OUT/stage2/cells.json     the cells written by the scorers (tickets 07, 08; written by 10)
           records/status.json       `ns_paths.record_path('status.json')`; outside smoke it must equal git HEAD
    writes OUT/stage2/outcomes.json  every claim cell as a row, the summary, the licensed sentences

CELL IDS. `SC/set/encoder/aspect/metric/k/comparator` (k `-` for map). One id differs from the plain metric name:
SC5's AMI-at-20 cells carry metric token `ami@20` (their `metric` field is `ami`, `k` 20, `kind` 'fixed20'). Without it
MMArt style (class count 20) would give the same id twice. Ticket 07 builds its SC5 cell ids with `claim_cells()`.
`kind`: 'class_count' (k = the class count; the class-count gates and stage 1 ranges apply), 'fixed20' (SC5 AMI at 20),
'map' (retrieval).

STATUS (ticket 10 writes it; values 'ok' | 'failed' | 'missing'; anything but 'ok' voids):
  {"gates": {"i_a","i_b","i_d","ii_scorer","ii_crl","ii_percept","vi": s; "i_c": {set: s} (affection, emoset);
             "iii_metric": {ami, nmi, ari, acc: s}; "iii_count": {"<set>/<aspect>": s}; "iv","v": {enc: s}},
   "stage1": {"ranges": {range key: s}, "picks": {pick key: s}}}
  The gate keys are exact (an absent or extra key raises). A range or pick key that is absent counts as 'missing' (a
  voided cell, never a silent pass); a key that no claim cell uses raises.
  Range key `<cstar|photo>/<set>/<enc>/<aspect>/<metric>`: the class-count cells (SC1 to SC4, SC5's four class-count
  metrics) use it for r. Pick key `<arm>/<set>/<enc>`: `untold` (SC7), `matched` (SC8; not Qwen on ArtELingo, which keeps
  r11's pick), `crl/mmart/clip_l14` (SC6 on MMArt). SC5's AMI at 20 and the retrieval cells take r from a recorded
  range, carried in the cell.

STATUS OF A ROW: 'decided', 'inconclusive' (a failed gate or stage 1 item voids it; `void` lists the reasons) or
'missing' (no stage 2 cell and no failed gate excuses it). Missing cells are rows, never dropped. `meets_bar` is None
unless decided. Teacher cells (SC3, SC4): `meets_bar` is True at WIN or LEVEL for either bar; `wording` tells which
sentence (WIN: "better than", LEVEL: "level with or better than"), as r15's SC4 rule. A non-teacher cell meets its
(WIN) bar only at WIN. `bar_met` is the strict reading (None unless decided): True only when the outcome reaches the
cell's own bar (WIN bar: WIN; LEVEL_OR_WIN bar: WIN or LEVEL), so a WIN-bar teacher cell at LEVEL has `meets_bar`
True (its weaker sentence is licensed) and `bar_met` False (r17 review of ticket 02, finding 1).
"""
import argparse
import functools
import json
import math
import sys
from pathlib import Path

import ns_paths as P

P.add_rounds()
import rt_stats                                            # noqa: E402  r11: the outcome word rule

WIN, LEVEL, BELOW, NOT_WIN = 'WIN', 'LEVEL', 'BELOW', 'not WIN'
BAR_WIN, BAR_LEVEL_OR_WIN = 'WIN', 'LEVEL_OR_WIN'
BETTER, LEVEL_WORDING = 'better than', 'level with or better than'
OK = 'ok'
STATUS_VALUES = (OK, 'failed', 'missing')
AMI20 = 'ami@20'
CLASS_COUNT, FIXED20, MAP = 'class_count', 'fixed20', 'map'
METRIC_NAME = {'ami': 'AMI', 'nmi': 'NMI', 'ari': 'ARI', 'acc': 'ACC', 'map': 'mAP', AMI20: 'AMI'}
SET_NAME = {'artelingo': 'ArtELingo held', 'mmart': 'MMArt test', 'affection': 'Affection test',
            'emoset': 'EmoSet test'}
CELL_KEYS = ('id', 'sc', 'set', 'encoder', 'aspect', 'task', 'metric', 'k', 'candidate', 'comparator', 'cand', 'comp',
             'd', 'lo', 'hi', 'r', 'r_source', 'n_draws', 'n_rows')
IDENTITY_KEYS = ('sc', 'set', 'encoder', 'aspect', 'task', 'metric', 'k', 'candidate', 'comparator')
NUMBERS = ('cand', 'comp', 'd', 'lo', 'hi')
GATE_SINGLE = ('i_a', 'i_b', 'i_d', 'ii_scorer', 'ii_crl', 'ii_percept', 'vi')


# ------------------------------------------------------------------------------------------------ the 412 cells
def _id(sc, s, e, a, tag, k, comp):
    return '/'.join((sc, s, e, a, tag, '-' if k is None else str(k), comp))


@functools.lru_cache(maxsize=1)
def _claim_cells():
    cells = []

    def add(sc, s, e, a, metric, kind, cand, comp, task, bar, tag=None):
        k = None if kind == MAP else (20 if kind == FIXED20 else P.CLASS_COUNT[(s, a)])
        cells.append(dict(id=_id(sc, s, e, a, tag or metric, k, comp), sc=sc, set=s, encoder=e, aspect=a, task=task,
                          metric=metric, k=k, kind=kind, candidate=cand, comparator=comp, bar=bar))

    def teacher_bar(s, e, a):
        return BAR_WIN if (s, e, a) in P.TEACHER_WIN_BAR else BAR_LEVEL_OR_WIN

    for sc, comp in (('SC1', 'plain'), ('SC2', 'matched'), ('SC3', 'teacher')):
        for s in P.PAINTING_SETS:
            for e in P.ENCODERS:
                for a in P.ASPECTS:
                    for m in P.CLASS_METRICS:
                        bar = teacher_bar(s, e, a) if comp == 'teacher' else BAR_WIN
                        add(sc, s, e, a, m, CLASS_COUNT, 'cstar', comp, 'clu', bar)
    for s in P.PHOTO_SETS:
        for e in P.PHOTO_ENCODERS[s]:
            for comp in ('plain', 'teacher'):
                for m in P.CLASS_METRICS:
                    bar = teacher_bar(s, e, 'emotion') if comp == 'teacher' else BAR_WIN
                    add('SC4', s, e, 'emotion', m, CLASS_COUNT, 'photo', comp, 'clu', bar)
    for s in P.PAINTING_SETS:
        for a in P.ASPECTS:
            add('SC5', s, P.CRL_ENCODER, a, 'ami', FIXED20, 'cstar', 'crl', 'clu', BAR_WIN, tag=AMI20)
            for m in P.CLASS_METRICS:
                add('SC5', s, P.CRL_ENCODER, a, m, CLASS_COUNT, 'cstar', 'crl', 'clu', BAR_WIN)
    for s in P.PAINTING_SETS:
        for a in P.ASPECTS:
            add('SC6', s, P.CRL_ENCODER, a, 'map', MAP, 'cstar', 'crl', 'ret', BAR_WIN)
    for sc, cand, comp in (('SC7', 'untold', 'plain'), ('SC8', 'cstar', 'matched')):
        for s in P.PAINTING_SETS:
            for e in P.ENCODERS:
                for a in P.ASPECTS:
                    add(sc, s, e, a, 'map', MAP, cand, comp, 'ret', BAR_WIN)
    assert len(cells) == 412 and len({c['id'] for c in cells}) == 412
    return tuple(cells)


def claim_cells():
    """The 412 claim cells (fresh dicts): id, sc, set, encoder, aspect, task, metric, k, kind, candidate, comparator,
    bar ('WIN' | 'LEVEL_OR_WIN'; teacher bars from `ns_paths.TEACHER_WIN_BAR`)."""
    return [dict(c) for c in _claim_cells()]


@functools.lru_cache(maxsize=1)
def _by_id():
    return {c['id']: c for c in _claim_cells()}


def metric_tag(cell):
    """The metric token of the cell id (`ami@20` for SC5's AMI at 20)."""
    return AMI20 if (cell['sc'] == 'SC5' and cell['kind'] == FIXED20) else cell['metric']


def range_key(cell):
    """The stage 1 range key of a class-count cell, else None."""
    if cell['kind'] != CLASS_COUNT:
        return None
    cand = 'photo' if cell['set'] in P.PHOTO_SETS else 'cstar'
    return f"{cand}/{cell['set']}/{cell['encoder']}/{cell['aspect']}/{cell['metric']}"


def pick_key(cell):
    """The key of the new stage 1 pick the cell uses, else None."""
    sc, s, e = cell['sc'], cell['set'], cell['encoder']
    if sc == 'SC7':
        return f'untold/{s}/{e}'
    if sc == 'SC8' and not (s == 'artelingo' and e == 'qwen'):       # Qwen on ArtELingo keeps r11's pick and probabilities
        return f'matched/{s}/{e}'
    if sc == 'SC6' and s == 'mmart':                                  # CRL on ArtELingo keeps r11's pick
        return 'crl/mmart/clip_l14'
    return None


def _is_teacher(cell):
    return cell['comparator'] == 'teacher'


# ------------------------------------------------------------------------------------------------ status
def check_status(status):
    """The status record's shape (exact gate keys; values ok | failed | missing; no range or pick key unused)."""
    def keys(d, want, what):
        assert isinstance(d, dict) and set(d) == set(want), (
            f'status {what}: keys {sorted(d) if isinstance(d, dict) else d!r}, need {sorted(want)}')

    def val(v, what):
        assert v in STATUS_VALUES, f'status {what}: {v!r} is not one of {STATUS_VALUES}'

    keys(status, ('gates', 'stage1'), 'top level')
    g, s1 = status['gates'], status['stage1']
    keys(g, GATE_SINGLE + ('i_c', 'iii_metric', 'iii_count', 'iv', 'v'), 'gates')
    for k in GATE_SINGLE:
        val(g[k], f'gates.{k}')
    for k, want in (('i_c', P.PHOTO_SETS), ('iii_metric', P.CLASS_METRICS),
                    ('iii_count', [f'{s}/{a}' for (s, a) in P.CLASS_COUNT]), ('iv', P.ENCODERS), ('v', P.ENCODERS)):
        keys(g[k], want, f'gates.{k}')
        for x, v in g[k].items():
            val(v, f'gates.{k}.{x}')
    keys(s1, ('ranges', 'picks'), 'stage1')
    for what, used in (('ranges', {range_key(c) for c in _claim_cells()} - {None}),
                       ('picks', {pick_key(c) for c in _claim_cells()} - {None})):
        assert isinstance(s1[what], dict), f'status stage1.{what}: not a dict'
        extra = set(s1[what]) - used
        assert not extra, f'status stage1 {what[:-1]}: keys no claim cell uses: {sorted(extra)}'
        for x, v in s1[what].items():
            val(v, f'stage1.{what}.{x}')
    return status


def _stage1(status, what, key):
    return status['stage1'][what].get(key, 'missing')


def gate_voids(cell, status):
    """Reasons from the failed gates and missing picks that void `cell` (§3.5's table); [] when none."""
    g = status['gates']
    sc, s, e, a, metric = cell['sc'], cell['set'], cell['encoder'], cell['aspect'], cell['metric']
    cc = cell['kind'] == CLASS_COUNT
    new_pick = pick_key(cell) is not None
    rules = [
        ('i_a', g['i_a'], sc in ('SC1', 'SC2', 'SC3', 'SC5')),
        ('i_b', g['i_b'], sc == 'SC2'),
        (f'i_c[{s}]', g['i_c'].get(s), sc == 'SC4'),
        ('i_d', g['i_d'], cc),
        ('ii_scorer', g['ii_scorer'], sc in ('SC6', 'SC7', 'SC8')),
        ('ii_crl', g['ii_crl'], sc in ('SC5', 'SC6')),
        ('ii_percept', g['ii_percept'], sc == 'SC8'),
        (f'iii_metric[{metric}]', g['iii_metric'].get(metric), metric in P.CLASS_METRICS),   # retrieval (map) has no metric gate
        (f'iii_count[{s}/{a}]', g['iii_count'].get(f'{s}/{a}'), cc),
        (f'iv[{e}]', g['iv'].get(e), (sc == 'SC8' and new_pick) or (sc == 'SC2' and s == 'mmart')),
        (f'v[{e}]', g['v'].get(e), sc == 'SC7'),
        ('vi', g['vi'], new_pick),
    ]
    out = [f'gate {name} {v}' for name, v, applies in rules if applies and v != OK]
    pk = pick_key(cell)
    if pk is not None and _stage1(status, 'picks', pk) != OK:
        out.append(f'stage 1 pick {pk} {_stage1(status, "picks", pk)}')
    return out


# ------------------------------------------------------------------------------------------------ rows
def _finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def check_cell(c, smoke):
    """Asserts at the file boundary; returns the claim cell the id names."""
    missing = [k for k in CELL_KEYS if k not in c]
    assert not missing, f'cell lacks {missing}: {c.get("id")}'
    cc = _by_id().get(c['id'])
    assert cc is not None, f'cell id {c["id"]!r} is not in claim_cells()'
    for k in IDENTITY_KEYS:
        assert c[k] == cc[k], f'cell {c["id"]}: field {k} = {c[k]!r} disagrees with its id ({cc[k]!r})'
    for k in NUMBERS:
        assert _finite(c[k]), f'cell {c["id"]}: {k} = {c[k]!r} is not a finite number'
    assert c['lo'] <= c['hi'], f'cell {c["id"]}: lo > hi ({c["lo"]} > {c["hi"]})'
    assert isinstance(c['n_rows'], int) and not isinstance(c['n_rows'], bool) and c['n_rows'] > 0, \
        f'cell {c["id"]}: n_rows {c["n_rows"]!r}'
    if not smoke:
        assert c['n_draws'] == P.N_DRAWS[cc['task']], (
            f'cell {c["id"]}: n_draws {c["n_draws"]} != {P.N_DRAWS[cc["task"]]} (the protocol; smoke allows others)')
    return cc


def word_of(lo, r):
    """WIN lo > 0; without r: 'not WIN'; else LEVEL (lo > -r) or BELOW (r11's rule, strict, unrounded)."""
    if lo > 0:
        return WIN
    if r is None:
        return NOT_WIN
    return rt_stats.outcome(lo, r)


def claim_wording(cell, outcome):
    """The sentence's comparison words. Teacher cells (r15's SC4 rule): a WIN-bar cell at WIN 'better than', at LEVEL
    'level with or better than'; another teacher cell at LEVEL or WIN 'level with or better than'; BELOW none. Other
    cells: 'better than' at WIN, else none."""
    if _is_teacher(cell):
        if outcome == WIN:
            return BETTER if cell['bar'] == BAR_WIN else LEVEL_WORDING
        return LEVEL_WORDING if outcome == LEVEL else None
    return BETTER if outcome == WIN else None


def _row(cc, c, status, smoke):
    """One output row for claim cell `cc` and stage 2 cell `c` (None when absent)."""
    void = gate_voids(cc, status)
    rk = range_key(cc)
    row = dict(cc, present=c is not None, void=void, outcome=None, meets_bar=None, bar_met=None, wording=None, status=None,
               range_key=rk, r_used=None, cand=None, comp=None, d=None, lo=None, hi=None, r=None, r_source=None,
               n_draws=None, n_rows=None)
    if c is not None:
        range_ok = rk is None or _stage1(status, 'ranges', rk) == OK
        r = c['r'] if range_ok else None
        if range_ok:
            assert _finite(r) and r >= 0, (
                f'cell {cc["id"]}: r is {c["r"]!r}; it needs a finite range >= 0 ({"stage 1 " + rk if rk else "recorded"})')
        row.update({k: c[k] for k in ('cand', 'comp', 'd', 'lo', 'hi', 'r', 'r_source', 'n_draws', 'n_rows')},
                   r_used=r)
        row['outcome'] = word_of(c['lo'], r)
        if not range_ok and _is_teacher(cc) and row['outcome'] != WIN:
            void.append(f'stage 1 range {rk} {_stage1(status, "ranges", rk)}: a teacher cell needs its LEVEL word')
    if void:
        row['status'] = 'inconclusive'
    elif c is None:
        row['status'] = 'missing'
    else:
        row['status'] = 'decided'
        wording = claim_wording(cc, row['outcome'])
        row['wording'] = wording
        row['meets_bar'] = wording is not None
        row['bar_met'] = row['outcome'] == WIN if cc['bar'] == BAR_WIN else row['outcome'] in (WIN, LEVEL)
    return row


def outcomes(cells, status, smoke=None):
    """One row per claim cell, in claim order: outcome word (WIN / LEVEL / BELOW, or 'not WIN' when the cell's stage 1
    range is missing), meets_bar, void reasons, status. Refuses ids outside `claim_cells()`, duplicates, fields that
    disagree with the id, bad numbers. A cell not given becomes a 'missing' row (or 'inconclusive' if a gate voids it)."""
    smoke = bool(P.SMOKE) if smoke is None else smoke
    check_status(status)
    given = {}
    for c in cells:
        check_cell(c, smoke)
        assert c['id'] not in given, f'duplicate cell {c["id"]}'
        given[c['id']] = c
    return [_row(cc, given.get(cc['id']), status, smoke) for cc in _claim_cells()]


# ------------------------------------------------------------------------------------------------ sentences
def _join(words):
    return words[0] if len(words) == 1 else ', '.join(words[:-1]) + ' and ' + words[-1]


def _sentence(sc, s, e, a, comp, wording, tags):
    """§6's sentence for one family: `tags` the metric tokens that pass (ordered)."""
    S, X = SET_NAME[s], P.ENCODER_NAME[e]
    K = P.CLASS_COUNT.get((s, a))
    M = _join([METRIC_NAME[t] for t in tags])
    if sc in ('SC1', 'SC2', 'SC3', 'SC4'):
        if sc == 'SC4':
            head = f'On {S}, the photo model on {X} clusters photos by emotion'
        else:
            head = f'On {S}, c* on {X} clusters paintings by {a}'
        target = {'plain': f'the plain {X} image', 'matched': f'PercepT on {X}, its k-means reading,',
                  'teacher': f'the Qwen image instructed for {a}'}[comp]
        return f'{head} {wording} {target} with as many clusters as classes ({K}), in {M} (5 seeds).'
    if sc == 'SC5':
        parts = []
        if AMI20 in tags:
            parts.append('AMI with 20 clusters')
        rest = [t for t in tags if t != AMI20]
        if rest:
            parts.append(f'{_join([METRIC_NAME[t] for t in rest])} with as many clusters as classes, {K}')
        return (f'c* on CLIP L/14 clusters paintings by {a} {wording} CRL on CLIP L/14 on {S} '
                f'({"; ".join(parts)}).')
    if sc == 'SC6':
        return f'c* on CLIP L/14 retrieves paintings by {a} {wording} CRL on CLIP L/14 on {S} (mAP).'
    if sc == 'SC7':
        return (f"On {S}, when the query names no aspect, c*'s joined heads on {X} retrieve paintings of the same {a} "
                f'{wording} the plain {X} image (mAP; read-outs picked on validation).')
    assert sc == 'SC8', sc
    return (f"On {S}, c*'s {a} head on {X} retrieves paintings by {a} {wording} PercepT's topic probabilities on {X} "
            f'(mAP).')


def licensed(row):
    """§6's sentence for one cell, or None (not decided, or the bar is missed)."""
    if row['status'] != 'decided' or not row['meets_bar']:
        return None
    return _sentence(row['sc'], row['set'], row['encoder'], row['aspect'], row['comparator'], row['wording'],
                     [metric_tag(row)])


def licensed_families(rows):
    """The paper's form: cells of one family (SC, set, encoder, aspect, comparator, wording) with the metrics that
    pass listed in one sentence (SC5: AMI at 20 joins the class-count metrics; retrieval cells stay single)."""
    fams = {}
    for r in rows:
        if licensed(r) is None:
            continue
        key = (r['sc'], r['set'], r['encoder'], r['aspect'], r['comparator'], r['wording'])
        fams.setdefault(key, []).append(metric_tag(r))
    out = []
    for (sc, s, e, a, comp, wording), tags in fams.items():
        out.append(dict(sc=sc, set=s, encoder=e, aspect=a, comparator=comp, wording=wording, metrics=tags,
                        sentence=_sentence(sc, s, e, a, comp, wording, tags)))
    return out


# ------------------------------------------------------------------------------------------------ summary, files
def summary(rows):
    """{sc: {metric tag: counts}}: cells, decided, inconclusive, missing, and over the decided rows WIN, LEVEL,
    BELOW, NOT_WIN, meets_bar, bar_met."""
    out = {}
    for r in rows:
        d = out.setdefault(r['sc'], {}).setdefault(metric_tag(r), dict(
            cells=0, decided=0, inconclusive=0, missing=0, WIN=0, LEVEL=0, BELOW=0, NOT_WIN=0, meets_bar=0, bar_met=0))
        d['cells'] += 1
        d[r['status']] += 1
        if r['status'] == 'decided':
            d['NOT_WIN' if r['outcome'] == NOT_WIN else r['outcome']] += 1
            d['meets_bar'] += int(r['meets_bar'])
            d['bar_met'] += int(r['bar_met'])
    return out


def decide(cells_path, status_path, out_path, smoke=None):
    """Read the stage 2 cells and the status record, write the outcomes file; returns the dict written."""
    smoke = bool(P.SMOKE) if smoke is None else smoke
    cells = json.loads(Path(cells_path).read_text())
    assert isinstance(cells, list), f'{cells_path}: the cells file must hold a list'
    status = json.loads(Path(status_path).read_text())
    rows = outcomes(cells, status, smoke=smoke)
    res = dict(smoke=smoke, status=status, cells=rows, summary=summary(rows), licensed=licensed_families(rows))
    P.write_json(out_path, res)
    return res


def table(res):
    lines = ['sc   metric   cells decided inconcl missing  WIN LEVEL BELOW notWIN meets']
    for sc, ms in res['summary'].items():
        for m, d in ms.items():
            lines.append(f'{sc:<4} {m:<8} {d["cells"]:>5} {d["decided"]:>7} {d["inconclusive"]:>7} {d["missing"]:>7} '
                         f'{d["WIN"]:>4} {d["LEVEL"]:>5} {d["BELOW"]:>5} {d["NOT_WIN"]:>6} {d["meets_bar"]:>5}')
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('decide', help='stage 2 cells + records/status.json -> OUT/stage2/outcomes.json')
    p.add_argument('--root', default=str(P.OUT))
    p.add_argument('--out', default=None)
    args = ap.parse_args(argv)
    stage2 = Path(args.root) / 'stage2'
    status_path = P.record_path('status.json')
    if not P.SMOKE:
        P.read_committed_record('status.json')             # refuses unless equal to git HEAD
    res = decide(stage2 / 'cells.json', status_path, args.out or stage2 / 'outcomes.json')
    print(table(res))
    print(f'{len(res["cells"])} cells, {len(res["licensed"])} licensed sentences')


if __name__ == '__main__':
    main()
