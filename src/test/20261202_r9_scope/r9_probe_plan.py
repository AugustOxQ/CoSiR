"""One real, assertion-checked development corpus run; writes choices before scores.

Run with the build's CPU environment; no scores, fitting or test-split reads.
"""

import json
from datetime import datetime
from zoneinfo import ZoneInfo

from r9_probe_data import HERE, load_probe_data


def main():
    output = HERE / 'results/probes/plan.json'
    assert not output.exists(), f'Refusing to overwrite {output}'
    log = HERE / '20261202_r9_scope_log.md'
    previous = log.read_text()
    assert '## Probe corpora (ticket 26)' not in previous
    timestamp = datetime.now(ZoneInfo('Europe/Amsterdam')).strftime('%F %H:%M')
    plan = {'created_at': timestamp, 'seed': 42, 'minimum_scoring_units': 30,
            'stage': 'development exploration; no bar or claim', 'datasets': {}}
    text = ['\n## Probe corpora (ticket 26)\n\n',
            f'- {timestamp}: choices and counts fixed before any probe score. Seed 42 only; '
            'three unordered aspect pairs per dataset. Values need ≥ 30 distinct scoring units.\n',
            '- CUB: xlsa17 trainval species; dev_species defaults (30 species, seed 42), '
            'remaining 120 species fit. Units are images, grouped by species; ten caption rows per unit. '
            'The 50 test species are excluded before feature selection or caption/attribute loading.\n',
            '- SemArt: official train fit, val scoring, one painting/caption per unit. '
            'TYPE, SCHOOL and TIMEFRAME are used verbatim, with empty strings missing; '
            'TIMEFRAME is not inferred from DATE. Test CSV is never opened. Captions use the '
            'extraction script’s scrub_semart rule.\n',
            '- Full-store headers/index metadata identify allowed rows; mmap selections materialise only '
            'allowed features. SHA-256 covers complete img.npy, txt.npy and index.json byte streams, '
            'including excluded rows, for provenance; no excluded feature is decoded or scored.\n',
            '\n| Dataset / split | Species or painting groups | Units | Caption rows |\n',
            '|---|---:|---:|---:|\n']
    for name in ('cub', 'semart'):
        corpus = load_probe_data(name)
        p = corpus.plan
        plan['datasets'][name] = p
        print(name, p['splits'], flush=True)
        for split, count in p['splits'].items():
            text.append(f"| {name} / {split} | {count['groups']} | {count['units']} | {count['rows']} |\n")
    text += ['\nCUB aspect rule (same for all three): ' +
             plan['datasets']['cub']['aspects']['primary_colour']['rule'] + '\n',
             '\nThis is the simplest majority interpretation supported by the data: '
             'a unique maximum among qualified per-image labels, with ties missing. '
             'The run asserts one annotation per allowed image/attribute (2,752,152 records). '
             'Five- and six-field annotation records are accepted; labels use the first four '
             'fields as in the existing loaders (606 allowed records have six fields).\n',
             '\n| Dataset | Aspect | Values with ≥ 30 scoring units (count) | Missing scoring units |\n',
             '|---|---|---|---:|\n']
    for name, p in plan['datasets'].items():
        for aspect, a in p['aspects'].items():
            entries = ', '.join(f"{v['value']} ({v['count']})" for v in a['values'])
            text.append(f"| {name} | {aspect} | {entries} | {a['scoring_missing_units']} |\n")
    for name, p in plan['datasets'].items():
        text.append(f"\n{name} pairs: " + ', '.join(f'{a} × {b}' for a, b in p['pairs']) + '.\n')
        text.append(f"\n{name} join asserted: {p['join_check']}\n")
    text.append('\nFeature-file paths and SHA-256s are pinned in results/probes/plan.json. '
                'No probe score was read in this ticket.\n')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + '\n')
    log.write_text(previous + ''.join(text))
    print(output, flush=True)


if __name__ == '__main__':
    main()
