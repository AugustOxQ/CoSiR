"""Read original JSON and fail closed if plotted values disagree with report tables."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DATA = Path('/data/SSD2/cosir_r10/20261027_recipe_sweep/outputs')
REPORT = ROOT / 'docs/reports/auto/percept/2026-12-28_recipe_sweep.md'
ASPECTS = ('emotion', 'style', 'genre')
ROWS = [('instructed_graphs', 'Instructed graphs'), ('buddy', 'Buddy terms'),
        ('content_correction', 'Content correction'), ('relational', 'Relational term'),
        ('sharp_row', 'Sharper teacher row'), ('direct', 'Direct teacher'),
        ('caption_endpoints', 'Caption endpoints'), ('shared_trunk', 'Shared trunk')]
INK, SECONDARY, GRID, REFERENCE = '#0b0b0b', '#52514e', '#e4e3df', '#8a8984'
BLUE, LIGHT, GREEN, ORANGE = '#1d6fd1', '#8fb8ea', '#4a9c5b', '#c98b2e'


def load(name):
    return json.loads((DATA / name).read_text())


def numbers(cell):
    return [float(v) for v in re.findall(r'[+-]?\d+\.\d+', cell)]


def check(actual, expected, context):
    assert len(actual) == len(expected), context
    for a, b in zip(actual, expected):
        if round(a, 4) != round(b, 4):
            raise ValueError(f'{context}: JSON {a:.4f} differs from report {b:.4f}')


def table(text, label):
    return {cells[0]: cells[1:] for line in text.splitlines() if line.startswith('| ')
            for cells in [[c.strip() for c in line.strip('|').split('|')]]
            if cells[0] not in ('Aspect', 'Removal', '---')}


def held_data():
    reported, verdict = load('stage4/reported.json'), load('stage4/verdict.json')
    section = REPORT.read_text().split('### Stage 4: held, full scale')[1].split('### Ablation')[0]
    primary = table(section.split('Reported comparisons:')[0], '')
    comparisons = table(section.split('Reported comparisons:')[1], '')
    bars = {}
    for a in ASPECTS:
        v, r = verdict['aspects'][a], reported['aspects'][a]
        b = r['vs_B1_to_B6']
        bars[a] = [b['A_e'][b['max_entry']], v['A_T'], r['d_G']['A_G'], v['A_cand']]
        check([v['A_cand'], v['A_T']], numbers(primary[a][0]), a+' c*/T')
        check([v[k] for k in ('d','lo','hi')], numbers(primary[a][1]), a+' paired interval')
        check([bars[a][2]], numbers(comparisons[a][0]), a+' G')
        check([bars[a][0]], numbers(comparisons[a][2]), a+' PercepT')
        check([b['A_cand']], [v['A_cand']], a+' reported/verdict')
        outcome = 'T-WIN' if v['lo'] > 0 else 'T-LEVEL'
        assert v['outcome'] == outcome
    # This reference is also explicitly stated in the checked report's seed-range key.
    assert f"genre {verdict['ranges']['genre']:.4f}" in REPORT.read_text()
    return bars, verdict


def ablation_data():
    section = REPORT.read_text().split('### Ablation')[1].split('## Analysis')[0]
    result = {}
    for split, marker, end in [('dev', '**Development.**', '**Held.**'), ('held','**Held.**', None)]:
        d = load('ablation/'+split+'.json')
        text = section.split(marker)[1]
        if end: text = text.split(end)[0]
        t = table(text, '')
        for key, label in ROWS:
            row = t[label.lower()]
            assert d['results'][key]['status'] == 'ok'
            for j,a in enumerate(ASPECTS):
                v = d['results'][key]['aspects'][a]
                vals = [v[k] for k in ('effect','lo','hi')]
                if split == 'held': vals.insert(0, v['arm_ami'])
                check(vals, numbers(row[j]), f'{split}/{key}/{a}')
                assert v['lo'] <= v['effect'] <= v['hi']
        result[split] = d
    assert result['dev']['seeds'] == [1,2,3]
    assert result['held']['seeds'] == [0,1,2,3,4]
    return result


def style(plt):
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':14,
        'text.color':INK, 'axes.labelcolor':INK, 'xtick.color':SECONDARY,
        'ytick.color':INK, 'figure.facecolor':'white', 'axes.facecolor':'white',
        'savefig.facecolor':'white'})


def clean(ax):
    for side in ('top','right','left'): ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(REFERENCE)
    ax.tick_params(length=0, pad=8)
