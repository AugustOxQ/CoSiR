"""r19 report tables: every cell, claim and cost row, rendered from OUT/stage4/outcomes.json (no number is typed by hand).

Also saves what the reports rest on under records/: outcomes.json, the check-outcomes summary and the diagnostics tables.
Usage: python pp_report_tables.py
"""
import hashlib
import json
import shutil
from pathlib import Path

ROUND = Path(__file__).resolve().parent
BASE = Path('/data/SSD2/cosir_r19/20261103_past_picks')
OUT = BASE / 'outputs'
RECORDS = ROUND / 'records'
NAME = dict(qwen='Qwen', siglip2='SigLIP 2', siglip1='SigLIP 1', clip_l14='CLIP L/14', dinov2_large='DINOv2', clip_b32='CLIP B/32',
            clip_b16='CLIP B/16')
ORDER = list(NAME)
ASPECTS = ('emotion', 'style', 'genre')
SHORT = {'better than': 'B', 'level with or better than': 'L', None: 'None'}
SETS = dict(artelingo='ArtELingo held', photos='Affection test', mmart='MMArt test', semart='SemArt new artists', cost='Cost')


def row(c):
    return f"{c['candidate_value']:.4f}/{c['comparator_value']:.4f}; {c['d']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]"


def plain_table(cells, scs, title, encoders):
    lines = [f'### {title}', '', 'Entries: **c*/comparator; d [95%]**; every entry is a WIN unless marked.', '',
             '|Encoder, aspect|' + '|'.join(scs) + '|', '|---|' + '---|' * len(scs)]
    index = {(c['sc'], c['encoder'], c['aspect']): c for c in cells}
    for e in encoders:
        for a in ASPECTS:
            found = [index.get((sc, e, a)) for sc in scs]
            if not any(found):
                continue
            text = [(row(c) + ('' if c['outcome'] == 'WIN' else f" **{c['outcome']}**")) if c else '' for c in found]
            lines.append(f'|{NAME[e]}, {a}|' + '|'.join(text) + '|')
    return lines + ['']


def teacher_table(cells, claims, sc, title):
    lines = [f'### {title}', '', '|Encoder|Data|Aspect|Task|c*/teacher|d [95%]|r|Outcome|Set for "better than"|Claim|',
             '|---|---|---|---|---|---|---|---|---|---|']
    rows = [c for c in cells if c['sc'] == sc]
    rows.sort(key=lambda c: (ORDER.index(c['encoder']), c['data'], ASPECTS.index(c['aspect']), c['task']))
    for c in rows:
        k = claims[(c['sc'], c['encoder'], c['aspect'], c['task'], c['data'])]
        task = 'AMI' if c['task'] == 'clu' else 'mAP'
        lines.append(f"|{NAME[c['encoder']]}|{c['data']}|{c['aspect']}|{task}|{c['candidate_value']:.4f}/{c['comparator_value']:.4f}"
                     f"|{c['d']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]|{c['r']:.4f}|{c['outcome']}|{'yes' if k['d17'] else 'no'}"
                     f"|{SHORT[k['claim']]}|")
    return lines + ['']


def main():
    data = (OUT / 'stage4/outcomes.json').read_bytes()
    res = json.loads(data)
    shutil.copyfile(OUT / 'stage4/outcomes.json', RECORDS / 'outcomes.json')
    shutil.copyfile(OUT / 'stage4/diag/tables.md', RECORDS / 'diag_tables.md')
    check = json.loads((BASE / 'c13/check_outcomes.json').read_text())
    assert check['pass'] is True and check['sha256']['stage4/outcomes.json'] == hashlib.sha256(data).hexdigest()
    (RECORDS / 'outcomes_check.json').write_text(json.dumps({k: v for k, v in check.items() if k != 'items'}, indent=1) + '\n')
    cells = res['cells']
    claims = {(k['sc'], e, k['aspect'], k['task'], k['data']): k for e, rows in res['claims'].items() for k in rows}
    new, every = ORDER[4:], ORDER
    out = ['# r19 full tables', '', 'Rendered by `pp_report_tables.py` from `records/outcomes.json`. Qwen = Qwen3-VL-Embedding-2B.', '',
           '## Verdicts', '', '|Encoder|' + '|'.join(SETS.values()) + '|', '|---|' + '---|' * len(SETS)]
    for e in ORDER:
        out.append(f'|{NAME[e]}|' + '|'.join(res['verdicts'][k].get(e, {}).get('verdict', 'not in this round') for k in SETS) + '|')
    out += ['', '## Cells', '']
    out += plain_table(cells, ['SC1', 'SC2', 'SC3'], 'ArtELingo held (12,281 paintings)', new)
    out += plain_table(cells, ['SC4', 'SC5'], 'Affection test, photo model', new)
    out += plain_table(cells, ['SC7', 'SC8', 'SC9'], 'MMArt test (16,384 paintings)', new)
    out += plain_table(cells, ['SC11', 'SC12', 'SC13'], 'SemArt new artists (11,159 paintings)', every)
    out += ['## Teacher cells', '', 'B = "better than"; L = "level with or better than"; None = no claim.', '']
    out += teacher_table(cells, claims, 'SC6', 'ArtELingo held and Affection test')
    out += teacher_table(cells, claims, 'SC10', 'MMArt test')
    out += teacher_table(cells, claims, 'SC14', 'SemArt new artists')
    out += ['## Cost (ms per image, model time, 2,000 MMArt paintings, batch 32)', '',
            '|Encoder|c* mean|c* slowest|Teacher mean|Teacher fastest|Ratio of means|c* with decoding|Teacher with decoding|',
            '|---|---|---|---|---|---|---|---|']
    for c in sorted(res['cost_cells'], key=lambda c: ORDER.index(c['encoder'])):
        out.append(f"|{NAME[c['encoder']]}|{c['cstar_ms_mean']:.1f}|{c['cstar_ms_max']:.1f}|{c['teacher_ms_mean']:.1f}"
                   f"|{c['teacher_ms_min']:.1f}|{c['teacher_ms_mean'] / c['cstar_ms_mean']:.1f}|{c['cstar_ms_e2e']:.1f}|{c['teacher_ms_e2e']:.1f}|")
    (RECORDS / 'report_tables.md').write_text('\n'.join(out) + '\n')


if __name__ == '__main__':
    main()
