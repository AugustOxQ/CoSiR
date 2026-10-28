"""Read-only source checks shared by the three figure builders. No GPU required."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
REPORT = ROOT / 'docs/reports/auto/percept/2026-12-28_recipe_sweep.md'
LOG = ROOT / 'src/test/20261027_recipe_sweep/20261027_recipe_sweep_log.md'
OUTPUTS = Path('/data/SSD2/cosir_r10/20261027_recipe_sweep/outputs')
INK, SECONDARY, GRID, REFERENCE = '#0b0b0b', '#52514e', '#e4e3df', '#8a8984'
BLUE, OLD_BLUE = '#1d6fd1', '#8fb8ea'
NAMES = dict(zip([f'k{i}' for i in range(1,14)], [
    'emotion relational weight', 'style relational weight', 'genre relational weight',
    'teacher row temperature', 'direct teacher weight', 'learning rate', 'weight decay',
    'batch size (pairs)', 'trunk width', 'dropout', 'graph neighbours', 'buddy temperature', 'patience']))


def numeric(text):
    return [float(x) for x in re.findall(r'[-+]?\d+(?:\.\d+)?', text)]


def equal(actual, expected, digits, context):
    assert round(actual, digits) == round(expected, digits), (context, actual, expected)


def stage1():
    d = json.loads((OUTPUTS / 'sweep/node402/stage1.json').read_text())
    log, report = LOG.read_text(), REPORT.read_text()
    for k, curve in d['curves'].items():
        row = next(s for s in log.splitlines() if s.startswith(f'| {k} /'))
        cells = row.split('|')
        tests = [numeric(s) for s in cells[3].split(';')]
        failures = numeric(cells[4])
        for point, (value, gain) in zip(curve, tests, strict=True):
            assert value == point['value'], (k, value)
            equal(point['M'], gain, 3, point['name'])
            assert point['guard'] == (value not in failures), point['name']
        if k in d['grid_knobs']:
            assert f"{d['knob_gains'][k]:+.3f}" in report
    assert len(d['curves']) == 13 and sum(map(len, d['curves'].values())) == 39
    return d


def stage3():
    d = json.loads((OUTPUTS / 'stage3/verdict.json').read_text())
    report = REPORT.read_text().split('### Stage 3: local confirmation')[1].split('### Stage 4')[0]
    log = LOG.read_text().split('## Stage 3: confirmation')[1].split('## Stage 4')[0]
    for run in d['top']:
        result = d['results'][run]
        for source in (report, log):
            row = next(s for s in source.splitlines() if s.startswith(f'| {run} |'))
            cells = row.split('|')
            for aspect, cell in zip(('emotion', 'style', 'genre'), cells[2:5]):
                nums = numeric(cell)
                e = result['effects'][aspect]
                actual = [e['Abar'][run], e['e']] + ([e['lo'], e['hi']] if len(nums) == 4 else [])
                for a,b in zip(actual, nums, strict=True):
                    equal(a,b,4,(run,aspect))
            for a,b in zip([result['M'][v] for v in ('M','lo','hi')],numeric(cells[5]),strict=True):
                equal(a,b,3,(run,'mean'))
            assert cells[6].strip().startswith('yes') == result['qualifies']
    assert d['seeds'] == [1,2,3]
    assert '29,232' in REPORT.read_text()
    return d


def style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':13,
        'text.color':INK, 'axes.labelcolor':INK, 'xtick.color':SECONDARY,
        'ytick.color':INK, 'figure.facecolor':'white', 'axes.facecolor':'white',
        'savefig.facecolor':'white'})


def quiet(ax):
    for side in ('left','right','top'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(REFERENCE)
    ax.tick_params(axis='y',length=0,pad=12)
