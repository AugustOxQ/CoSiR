"""The follow-up's ablation table with PercepT image-side arms and rows 1, 6, 10, 13 filled in.

Reads the stored results only: the follow-up's ags_fu_results.json (rows 2, 3, 7 to 12), this round's
pc_results.json (row 1 and arms) and pc_rows_results.json (rows 6, 10, 13). Prints markdown at 3 decimals;
differences use the unrounded means.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ASPECTS = ('emotion', 'style', 'genre')
PROMPT = dict(emotion='emotion_focus', style='style_word', genre='genre_word')
PICK = dict(emotion='Fres1', style='Fres05', genre='Fside025')
ARM_ROWS = {'4a': 'P1', '4b': 'P2', '5a': 'P3', '5b': 'P4', '5c': 'P5'}


def load():
    fu = json.loads((HERE.parent / '20261018_aspect_graph_sheet' / 'ags_fu_results.json').read_text())
    result = json.loads((HERE / 'outputs' / 'pc_results.json').read_text())
    rows = json.loads((HERE / 'outputs' / 'pc_rows_results.json').read_text())['aggregates']
    return fu, result, rows


def head(fu, kind, graph, aspect):
    a = fu['aggregates'][aspect][kind][graph]
    img, cap = a['ami']['img'][aspect]['mean'], a['ami']['cap'][aspect]['mean']
    return dict(img=img, cap=cap, mean=a['target_ami']['mean'], sd=a['target_ami']['sd'])


def view(fu, key, aspect):
    img = fu['references'][f'{key}/img']['ami'][aspect]['mean']
    cap = fu['references'][f'{key}/cap']['ami'][aspect]['mean']
    return dict(img=img, cap=cap, mean=(img + cap) / 2, sd=None)


def row1(result, kind, aspect):
    r = result['row1'][kind]
    return dict(img=r['img'][aspect]['mean'], cap=r['cap'][aspect]['mean'],
                mean=r['both_mean'][aspect], sd=None)


def ours(rows, name, aspect):
    a = rows[name].get(aspect)
    if a is None:
        return None
    assert a['complete']
    return dict(img=a['ami']['img'][aspect]['mean'], cap=a['ami']['cap'][aspect]['mean'],
                mean=a['target_ami']['mean'], sd=a['target_ami']['sd'])


def percept(arms, key, aspect):
    r = arms[key]
    if r['img'] is None:
        return None
    a = r['img'][aspect]
    return dict(img=a['mean'], cap=None, mean=None, sd=a['sd'],
                star='' if r['established'] else '*')


DIFFS = [('Instruction in the graph (default graph to plain aspect graph)', '7', '6'),
         ('Network on the default graph vs default embedding', '6', '2'),
         ('Other side alone instead of the plain aspect graph', '10', '7'),
         ('Prompted input instead of default, other side alone', '13', '10'),
         ('Encoder: Qwen default vs CLIP image + CLIP text', '2', '1a'),
         ('Encoder: Qwen default vs CLIP image + RoBERTa', '2', '1b')]
IMAGE_DIFFS = [('Stage 2 reader, CLIP', '4b', '4a'),
               ('Stage 2 reader, Qwen topics', '5b', '5a'),
               ('E: RoBERTa vs Qwen emotion', '5c', '5a'),
               ('Qwen vs CLIP: I, T and Stage 2 input (RoBERTa E)', '5c', '4b')]


def main(fu=None, result=None, rows=None):
    if fu is None:
        fu, result, rows = load()
    arms = result['arms']
    assert set(ARM_ROWS.values()) <= arms.keys()
    for r in arms.values():
        assert r['img'] is None or set(ASPECTS) <= r['img'].keys()
    table = {
        '1a': lambda a: row1(result, 'clip_text', a),
        '1b': lambda a: row1(result, 'roberta', a),
        '2': lambda a: view(fu, 'default', a),
        '3': lambda a: view(fu, PROMPT[a], a),
        **{row: lambda a, key=key: percept(arms, key, a) for row, key in ARM_ROWS.items()},
        '6': lambda a: ours(rows, 'row6', a),
        '7': lambda a: head(fu, 'default', 'R0', a),
        '8': lambda a: head(fu, 'default', 'R1', a),
        '9': lambda a: head(fu, 'default', PICK[a], a),
        '10': lambda a: head(fu, 'default', 'R0o', a) if a == 'genre' else ours(rows, 'row10', a),
        '11': lambda a: head(fu, 'prompted', 'R0', a),
        '12': lambda a: head(fu, 'prompted', PICK[a], a),
        '13': lambda a: ours(rows, 'row13', a),
    }
    cells = {row: {a: f(a) for a in ASPECTS} for row, f in table.items()}
    print('| # | Emotion | Style | Genre |\n|---|---:|---:|---:|')
    for row, by in cells.items():
        print(f'| {row} | ' + ' | '.join('—' if by[a] is None or by[a]['mean'] is None else f"{by[a]['mean']:.3f}" +
              ('' if by[a]['sd'] is None else f" ± {by[a]['sd']:.3f}") for a in ASPECTS) + ' |')
    print('\n| # | Emotion img | Emotion cap | Style img | Style cap | Genre img | Genre cap |')
    print('|---|---:|---:|---:|---:|---:|---:|')
    for row, by in cells.items():
        print(f'| {row} | ' + ' | '.join('—' if by[a] is None or by[a][s] is None else
              f"{by[a][s]:.3f}" + by[a].get('star', '')
              for a in ASPECTS for s in ('img', 'cap')) + ' |')
    if any(r['img'] is not None and not r['established'] for r in arms.values()):
        print('\n\\* not established (reasons in pc_tables.md)')
    print('\n### Image side, PercepT arms\n\n| # | Emotion | Style | Genre |\n|---|---:|---:|---:|')
    for row in ARM_ROWS:
        by = cells[row]
        print(f'| {row} | ' + ' | '.join('—' if by[a] is None else
              f"{by[a]['img']:.3f} ± {by[a]['sd']:.3f}" + by[a]['star'] for a in ASPECTS) + ' |')
    print('\n| Change | Rows | Emotion | Style | Genre |\n|---|---|---:|---:|---:|')
    for name, x, y in DIFFS:
        vals = ['—' if cells[x][a] is None or cells[y][a] is None else
                f"{cells[x][a]['mean'] - cells[y][a]['mean']:+.3f}" for a in ASPECTS]
        print(f'| {name} | {x} vs {y} | ' + ' | '.join(vals) + ' |')
    print('\n### Image-side changes\n\n| Change | Rows | Emotion | Style | Genre |\n|---|---|---:|---:|---:|')
    for name, x, y in IMAGE_DIFFS:
        vals = ['—' if cells[x][a] is None or cells[y][a] is None else
                f"{cells[x][a]['img'] - cells[y][a]['img']:+.3f}" for a in ASPECTS]
        print(f'| {name} | {x} vs {y} | ' + ' | '.join(vals) + ' |')


if __name__ == '__main__':
    main()
