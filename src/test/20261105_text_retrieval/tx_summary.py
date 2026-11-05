"""Render exploration tables and the prespecified nine-cell painting-bootstrap rule."""
import argparse
from pathlib import Path

import numpy as np

import tx_paths as paths
import tx_data as data
from tx_core import DIRECTIONS, write_json
from tx_metrics import painting_bootstrap

CELLS = [(a, d) for a in paths.ASPECTS for d in DIRECTIONS[:3]]
ROWS = ('cstar', 'cstar_free', 'plain', 'teacher', 'clip_l14', 'siglip2_so400m',
        'siglip_so400m', 'roberta')


def number(x):
    return '—' if x is None or not np.isfinite(x) else f'{x:.4f}'


def table(headers, rows):
    lines = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    lines.extend('| ' + ' | '.join(str(v) for v in row) + ' |' for row in rows)
    return '\n'.join(lines) + '\n'


def best(records, score, **where):
    candidates = [r for r in records if all(r.get(k) == v for k, v in where.items())
                  and r.get(score) is not None and np.isfinite(r[score])]
    return max(candidates, key=lambda r: r[score]) if candidates else None


def pick_labels(records):
    picks = {}
    for row in ROWS:
        for a, d in CELLS:
            rec = best(records, 'map', row=row, entry=a, direction=d)
            if rec is not None:
                picks[(row, a, d)] = rec
    return picks


def perquery(out, rec):
    path = out / 'perquery' / f'{rec["row"]}__{rec["aspect"] or "all"}__{rec["direction"]}__{rec["readout"]}.npz'
    with np.load(path, allow_pickle=False) as z:
        return z['ap__' + rec['entry']], z['q_pid']


def rule(out, picks, row, n_pid, smoke=False):
    cells = []
    for a, d in CELLS:
        candidate = picks.get((row, a, d))
        comparisons = {}
        for comparator in ('plain', 'teacher'):
            baseline = picks.get((comparator, a, d))
            if candidate is None or baseline is None or smoke:
                comparisons[comparator] = None
                continue
            ca, cp = perquery(out, candidate)
            ba, bp = perquery(out, baseline)
            assert ca.shape == ba.shape and np.array_equal(cp, bp), 'paired query join'
            mean, lo, hi = painting_bootstrap(ca - ba, cp, n_pid)
            comparisons[comparator] = dict(mean=mean, lo=lo, hi=hi,
                                           readout=baseline['readout'], n_paired=int(np.isfinite(ca - ba).sum()))
        complete = all(v is not None and np.isfinite(v['lo']) for v in comparisons.values())
        above = all(v['lo'] > 0 for v in comparisons.values()) if complete else None
        cells.append(dict(aspect=a, direction=d, readout=None if candidate is None else candidate['readout'],
                          comparisons=comparisons, above=above))
    complete = all(c['above'] is not None for c in cells)
    count = sum(c['above'] is True for c in cells)
    return dict(row=row, cells=cells, count=count if complete else None,
                outcome=(1 if count >= 5 else 2) if complete else None,
                complete=complete, smoke=smoke)


def interval(value):
    if value is None:
        return '—'
    return f'{number(value["mean"])} [{number(value["lo"])}, {number(value["hi"])}]'


def pair_table(pairs):
    groups = sorted({(r['row'], r['aspect'] or '', r['direction']) for r in pairs})
    rows, picks = [], []
    for row, a, d in groups:
        rec = best(pairs, 'mrr', row=row, aspect=a or None, direction=d)
        if rec is None:
            continue
        picks.append(rec)
        rows.append([row, a or 'all', d, rec['readout'], *[number(rec[k]) for k in ('r1', 'r5', 'r10', 'medr', 'mrr')]])
    return table(['row', 'aspect', 'direction', 'readout', 'R@1', 'R@5', 'R@10', 'MedR', 'MRR'], rows), picks


def label_tables(picks):
    headers = ['row'] + [a + '/' + d for a, d in CELLS]
    blocks = []
    for field in ('map', 'macro_map'):
        rows = []
        for row in ROWS:
            values = []
            for a, d in CELLS:
                rec = picks.get((row, a, d))
                values.append('—' if rec is None else number(rec[field]) + f' ({rec["readout"]})')
            rows.append([row, *values])
        if field == 'map':
            rows.append(['chance', *[number(picks.get(('plain', a, d), {}).get('chance')) for a, d in CELLS]])
        blocks.append(table(headers, rows))
    return blocks


def sensitivity_table(records, picks, leaks):
    rows = []
    for row in ('cstar', 'plain', 'teacher'):
        for a, d in CELLS:
            pick = picks.get((row, a, d))
            if pick is None:
                continue
            values = []
            for entry in (a + '__no_leak', a + '__cross_painting'):
                rec = best(records, 'map', row=row, aspect=pick['aspect'], direction=d,
                           readout=pick['readout'], entry=entry)
                values.append(number(None if rec is None else rec['map']))
            rows.append([row, a, d, pick['readout'], *values])
    shares = table(['aspect', 'label-name caption share'], [[a, number(leaks.get(a, {}).get('share'))] for a in paths.ASPECTS])
    per_class = []
    for a in paths.ASPECTS:
        for name, value in leaks.get(a, {}).get('per_class', {}).items():
            per_class.append([a, name, value['n'], number(value['share'])])
    return shares + '\n' + table(['aspect', 'class', 'n', 'share'], per_class) + '\n' + table(
        ['row', 'aspect', 'direction', 'picked readout', 'no_leak mAP', 'cross_painting mAP'], rows)


def diagnostic_table(records):
    rows = [[r['row'], r['aspect'] or 'all', r['entry'], r['readout'], number(r['map']),
             number(r['macro_map']), number(r['chance'])] for r in records if r['direction'] == 'capmean2i']
    return table(['row', 'head', 'label', 'readout', 'mAP', 'macro mAP', 'chance'], rows)


def mmart_tables(doc):
    pairs, records = doc.get('pairs', []), doc.get('records', [])
    groups = sorted({(r['row'], r['aspect'] or '') for r in pairs})
    headers = ['row/head'] + list(data.MMART_COLUMNS)
    rows = []
    for row, a in groups:
        cells = []
        for col in data.MMART_COLUMNS:
            rec = best(pairs, 'mrr', row=row, aspect=a or None, column=col)
            cells.append('—' if rec is None else f'{number(rec["r10"])} ({rec["readout"]})')
        rows.append([row + '/' + (a or 'all'), *cells])
    blocks = ['Pair R@10 (readout picked on MRR).\n\n' + table(headers, rows)]
    for label in paths.ASPECTS:
        rows = []
        for row, a in groups:
            cells = []
            for col in data.MMART_COLUMNS:
                rec = best(records, 'map', row=row, aspect=a or None, column=col, entry=label)
                cells.append('—' if rec is None else f'{number(rec["map"])} ({rec["readout"]})')
            rows.append([row + '/' + (a or 'all'), *cells])
        blocks.append(label + ' mAP.\n\n' + table(headers, rows))
    blocks.append('e_emotional own-emotion word share: ' + number(doc.get('emotional_leak_share')) + '.\n')
    return '\n'.join(blocks)


def render(out):
    core = data.read_json(out / 'results/core.json')
    records, pairs = core['records'], core['pairs']
    docs = {}
    for name in ('leak', 'semart', 'mmart', 'words'):
        path = out / 'results' / f'{name}.json'
        docs[name] = data.read_json(path) if path.exists() else {}
        if name != 'leak' and docs[name]:
            assert docs[name]['smoke'] == core['smoke'], 'mixed full/smoke results'
    picks = pick_labels(records)
    rules = [rule(out, picks, row, core['n_paintings'], core['smoke']) for row in ('cstar', 'cstar_free')]
    label, macro = label_tables(picks)
    pair_text, pair_picks = pair_table([r for r in pairs if r['direction'] not in ('capmean2i', 'i2i')])
    semart_text, semart_picks = pair_table(docs['semart'].get('pairs', []))
    sections = ['# Text retrieval exploration\n',
                'Validation picks and scores use the same rows; exploration numbers only.\n',
                '## Table 1: label mAP\n\n' + label,
                '## Table 1b: macro mAP at the same picked readout\n\n' + macro]
    for result in rules:
        rows = [[c['aspect'], c['direction'], c['readout'] or '—',
                 interval(c['comparisons']['plain']), interval(c['comparisons']['teacher']),
                 '—' if c['above'] is None else ('yes' if c['above'] else 'no')] for c in result['cells']]
        title = 'Table 2: the rule' if result['row'] == 'cstar' else 'Table 2b: free variant beside the rule'
        sections.append('## ' + title + '\n\n' + table(
            ['aspect', 'direction', 'cstar readout', 'minus plain, 95% CI', 'minus teacher, 95% CI', 'above'], rows))
        sections.append(f'Count: {result["count"]}; outcome: {result["outcome"]}.\n')
    sections.extend(['## Table 3: pair hits, best MRR readout\n\n' + pair_text,
                     '## Table 4: leak sensitivity at the all-entry picked readout\n\n' + sensitivity_table(records, picks, docs['leak']),
                     '## Table 5: capmean2i\n\n' + diagnostic_table(records),
                     '## Table 5b: capmean2i pair hits\n\n' + pair_table([r for r in pairs if r['direction'] == 'capmean2i'])[0],
                     '## Table 6: SemArt pairs\n\n' + semart_text,
                     '\n'.join(docs['semart'].get('notes', ['SemArt has not been run.'])),
                     '## Table 7: MMArt\n\n' + mmart_tables(docs['mmart']),
                     '\n'.join(docs['mmart'].get('notes', ['MMArt has not been run.'])),
                     '## Table 8: word queries\n'])
    word_rows = [[r['row'], r['aspect'], r['form'], r['direction'], r['readout'],
                  number(r['macro_map']), number(r['chance'])] for r in docs['words'].get('records', [])]
    sections.append(table(['row', 'aspect', 'form', 'direction', 'readout', 'macro AP', 'chance'], word_rows))
    sections.append('\n'.join(docs['words'].get('notes', ['Word extraction has not been run.'])))
    outcome = rules[0]['outcome']
    line = ('Smoke only; rule not evaluated.' if core['smoke'] else
            ('Rule incomplete: missing cells.' if outcome is None else
             f'Outcome {outcome}: cstar above in {rules[0]["count"]}/9 cells; free variant {rules[1]["count"]}/9.'))
    free_changes = rules[1]['outcome'] is not None and outcome is not None and rules[1]['outcome'] != outcome
    pair_limits = []
    for a, d in CELLS:
        cstar = next((r for r in pair_picks if r['row'] == 'cstar' and r['aspect'] == a and r['direction'] == d), None)
        plain = next((r for r in pair_picks if r['row'] == 'plain' and r['direction'] == d), None)
        cell = next(c for c in rules[0]['cells'] if c['aspect'] == a and c['direction'] == d)
        if cell['above'] and cstar and plain and cstar['mrr'] < plain['mrr']:
            pair_limits.append(dict(aspect=a, direction=d, cstar_mrr=cstar['mrr'], plain_mrr=plain['mrr']))
    sections.append(line + '\n')
    if free_changes:
        sections.append('The free variant changes the outcome; user decision required.\n')
    if pair_limits:
        sections.append('cstar wins label cells but loses pair MRR in some cells; user decision required on stating this limit.\n')
    summary = dict(smoke=core['smoke'], outcome=outcome, outcome_line=line, rules=rules,
                   picked=list(picks.values()), pair_picked=pair_picks, semart_picked=semart_picks,
                   leak=docs['leak'], capmean=[r for r in records if r['direction'] == 'capmean2i'],
                   mmart=docs['mmart'], words=docs['words'], free_changes_outcome=free_changes,
                   pair_limits=pair_limits)
    return '\n'.join(sections), summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=paths.OUT)
    args = p.parse_args()
    text, summary = render(args.out)
    directory = paths.ROUND / 'results'
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'tables.md').write_text(text)
    write_json(directory / 'summary.json', summary)
    print(summary['outcome_line'])


if __name__ == '__main__':
    main()
