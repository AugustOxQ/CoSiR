"""r29 stage 1c, C13 check (spec §3.4 "Order and C13"): our picks, ranges and cells against Codex's independent
re-derivation (`rederive/r29rd_c1_picks.json`, `rederive/c1_rederived.json`), tolerance 1e-6.

Codex's own `outcome` field reads INCONCLUSIVE wherever its replay of older cells differed by more than 1e-9 (its own
flag, stricter than the spec's 1e-6). Here the outcome of each Codex cell is recomputed from its lo and r with §3.4's
rule (WIN: lo > 0; LEVEL: not WIN and lo > -r; BELOW otherwise) and compared with ours; its replay flags are recorded.
A number that differs by more than 1e-6, or a pick or an outcome that differs, is listed under `mismatches`; that pick,
range or cell is inconclusive until the cause is fixed (the ladder is not run while the list is non-empty).

usage: python r29_c1_compare.py [--allow-missing-image]   writes records/c1_compare.json
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
REC = HERE / 'records'
RD = HERE / 'rederive'
TOL = 1e-6
ASPECTS = ('emotion', 'style', 'genre')
DIRS = ('t2i', 'i2t', 't2t')
NUM_KEYS = ('value', 'comparator_value', 'd', 'lo', 'hi', 'r')


def rule(lo, r):
    return 'WIN' if lo > 0 else ('LEVEL' if lo > -r else 'BELOW')


def load(p):
    return json.loads(Path(p).read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--allow-missing-image', action='store_true')
    args = ap.parse_args()
    cpk = load(RD / 'r29rd_c1_picks.json')['picks']
    cx = load(RD / 'c1_rederived.json')
    ip = load(REC / 'image_picks_siglip1_prefix.json')
    tp = load(REC / 'text_siglip1_prefix.json')['picks']
    items, mism = [], []

    def num(name, ours, theirs):
        diff = abs(ours - theirs)
        items.append(dict(item=name, ours=ours, codex=theirs, diff=diff))
        if not diff <= TOL:
            mism.append(f'{name}: ours {ours!r} codex {theirs!r} diff {diff:.3e}')
        return diff

    def same(name, ours, theirs):
        items.append(dict(item=name, ours=ours, codex=theirs, equal=ours == theirs))
        if ours != theirs:
            mism.append(f'{name}: ours {ours!r} codex {theirs!r}')

    # picks and ranges (validation)
    assert cx['picks'] == cpk, 'held run must use the committed-time validation picks'
    for a in ASPECTS:
        num(f'range/ami/{a}', ip['ami_ranges'][a]['r'], cpk['ami_range'][a])
        num(f'range/image_map/{a}', ip['map_ranges'][a]['r'], cpk['image_map'][a]['range'])
        same(f'pick/image_map_told/{a}', ip['readouts']['cstar']['pick'], cpk['image_map'][a]['pick'])
        same(f'pick/image_map_untold/{a}', ip['readouts']['untold']['pick'], cpk['image_map'][a]['p1_pick'])
        for d in DIRS:
            o, x = tp[a][d], cpk['text'][f'{a}/{d}']
            same(f'pick/text/{a}/{d}', o['pick'], x['pick'])
            num(f'range/text/{a}/{d}', o['range'], x['range'])
            num(f'mean/text/{a}/{d}', o['mean'], x['means'][x['pick']])

    # cells (test): ours by (aspect, measure, comparator)
    ours = {}
    for f in ('text_cells_siglip1_prefix.json', 'image_cells_siglip1_prefix.json'):
        p = REC / f
        if not p.exists():
            if f.startswith('image') and args.allow_missing_image:
                continue
            sys.exit(f'missing {p}')
        for c in load(p)['cells']:
            parts = c['id'].split('/')
            aspect, measure, comp = parts[3], parts[4], parts[5]
            comp = {'plain': 'p1', 'teacher': 'q1'}.get(comp, comp)
            measure = 'ami' if measure.startswith('ami') else measure
            ours[(aspect, measure, comp)] = c
    cells = []
    for c in cx['cells']:
        parts = c['id'].split('/')
        if parts[0] == 'text':
            key = (parts[1], parts[2], parts[3])
        else:
            key = (parts[1], 'ami' if parts[0] == 'ami' else 'map', parts[2])
        if key not in ours:
            if args.allow_missing_image and key[1] in ('ami', 'map'):
                continue
            mism.append(f'cell {c["id"]}: no cell of ours')
            continue
        o = ours[key]
        row = dict(cell='/'.join(key), codex_id=c['id'], ours_outcome=o['outcome'], codex_outcome_field=c['outcome'],
                   codex_outcome_rule=rule(c['lo'], c['r']), ours_outcome_rule=rule(o['lo'], o['r']))
        diffs = {k: abs(o[k] - c[k]) for k in NUM_KEYS}
        row['max_diff'] = max(diffs.values())
        row['diffs'] = diffs
        for k, v in diffs.items():
            if not v <= TOL:
                mism.append(f'cell {row["cell"]} {k}: ours {o[k]!r} codex {c[k]!r} diff {v:.3e}')
        if o['outcome'] != row['codex_outcome_rule']:
            mism.append(f'cell {row["cell"]} outcome: ours {o["outcome"]} codex (rule) {row["codex_outcome_rule"]}')
        if o['outcome'] != row['ours_outcome_rule']:
            mism.append(f'cell {row["cell"]}: our outcome {o["outcome"]} is not the rule word for our lo and r')
        cells.append(row)
    n_expected = 18 + (0 if args.allow_missing_image and not (REC / 'image_cells_siglip1_prefix.json').exists() else 12)
    complete = len(cells) == 30
    out = dict(
        tolerance=TOL, complete=complete, n_cells_compared=len(cells), n_cells_expected=n_expected,
        status=('pass' if complete and not mism else ('mismatch' if mism else 'partial')),
        mismatches=mism,
        max_diff_picks_ranges=max(i.get('diff', 0.0) for i in items),
        max_diff_cells=max((c['max_diff'] for c in cells), default=None),
        codex_replays={fam: {k: v for k, v in item.items() if not isinstance(v, (dict, list))}
                       for r in cx['replays'] for fam, item in r['replays'].items()},
        codex_sc3_own_field=cx['sc3'],
        note=('Codex outcomes recomputed from its lo and r by the §3.4 rule; its outcome field is INCONCLUSIVE where its '
              'replay of older cells differed by more than 1e-9 (its own flag; the spec tolerance is 1e-6, and every '
              'replay is within 1e-6).'),
        items=items, cells=cells,
        written=datetime.now(ZoneInfo('Europe/Amsterdam')).strftime('%Y-%m-%d %H:%M'))
    (REC / 'c1_compare.json').write_text(json.dumps(out, indent=1))
    print(f"status {out['status']}; cells {len(cells)}; max diff picks/ranges {out['max_diff_picks_ranges']:.2e}; "
          f"cells {out['max_diff_cells']}; mismatches {len(mism)}")
    for m in mism:
        print('  ', m)


if __name__ == '__main__':
    main()
