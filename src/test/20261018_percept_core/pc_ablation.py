"""The follow-up's ablation table with rows 1, 6, 10, 13 filled in (target AMI, both-side mean; img and cap).

Reads the stored results only: the follow-up's ags_fu_results.json (rows 2, 3, 7 to 12), this round's
pc_results.json (row 1) and pc_rows_results.json (rows 6, 10, 13). Prints markdown at 3 decimals; differences use
the unrounded means.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FU = json.loads((HERE.parent / '20261018_aspect_graph_sheet' / 'ags_fu_results.json').read_text())
ROW1 = json.loads((HERE / 'outputs' / 'pc_results.json').read_text())['row1']
ROWS = json.loads((HERE / 'outputs' / 'pc_rows_results.json').read_text())['aggregates']
ASPECTS = ('emotion', 'style', 'genre')
PROMPT = dict(emotion='emotion_focus', style='style_word', genre='genre_word')
PICK = dict(emotion='Fres1', style='Fres05', genre='Fside025')


def head(kind, graph, aspect):
    a = FU['aggregates'][aspect][kind][graph]
    img, cap = a['ami']['img'][aspect]['mean'], a['ami']['cap'][aspect]['mean']
    return dict(img=img, cap=cap, mean=a['target_ami']['mean'], sd=a['target_ami']['sd'])


def view(key, aspect):
    img = FU['references'][f'{key}/img']['ami'][aspect]['mean']
    cap = FU['references'][f'{key}/cap']['ami'][aspect]['mean']
    return dict(img=img, cap=cap, mean=(img + cap) / 2, sd=None)


def row1(kind, aspect):
    r = ROW1[kind]
    return dict(img=r['img'][aspect]['mean'], cap=r['cap'][aspect]['mean'],
                mean=r['both_mean'][aspect], sd=None)


def ours(name, aspect):
    a = ROWS[name].get(aspect)
    if a is None:
        return None
    assert a['complete']
    return dict(img=a['ami']['img'][aspect]['mean'], cap=a['ami']['cap'][aspect]['mean'],
                mean=a['target_ami']['mean'], sd=a['target_ami']['sd'])


TABLE = {
    '1a': lambda a: row1('clip_text', a),
    '1b': lambda a: row1('roberta', a),
    '2': lambda a: view('default', a),
    '3': lambda a: view(PROMPT[a], a),
    '6': lambda a: ours('row6', a),
    '7': lambda a: head('default', 'R0', a),
    '8': lambda a: head('default', 'R1', a),
    '9': lambda a: head('default', PICK[a], a),
    '10': lambda a: head('default', 'R0o', a) if a == 'genre' else ours('row10', a),
    '11': lambda a: head('prompted', 'R0', a),
    '12': lambda a: head('prompted', PICK[a], a),
    '13': lambda a: ours('row13', a),
}
DIFFS = [('Instruction in the graph (default graph to plain aspect graph)', '7', '6'),
         ('Network on the default graph vs default embedding', '6', '2'),
         ('Other side alone instead of the plain aspect graph', '10', '7'),
         ('Prompted input instead of default, other side alone', '13', '10'),
         ('Encoder: Qwen default vs CLIP image + CLIP text', '2', '1a'),
         ('Encoder: Qwen default vs CLIP image + RoBERTa', '2', '1b')]


def main():
    cells = {row: {a: f(a) for a in ASPECTS} for row, f in TABLE.items()}
    print('| # | Emotion | Style | Genre |\n|---|---:|---:|---:|')
    for row, by in cells.items():
        print(f'| {row} | ' + ' | '.join('—' if by[a] is None else f"{by[a]['mean']:.3f}" +
              ('' if by[a]['sd'] is None else f" ± {by[a]['sd']:.3f}") for a in ASPECTS) + ' |')
    print('\n| # | Emotion img | Emotion cap | Style img | Style cap | Genre img | Genre cap |')
    print('|---|---:|---:|---:|---:|---:|---:|')
    for row, by in cells.items():
        print(f'| {row} | ' + ' | '.join('—' if by[a] is None else f"{by[a][s]:.3f}"
              for a in ASPECTS for s in ('img', 'cap')) + ' |')
    print('\n| Change | Rows | Emotion | Style | Genre |\n|---|---|---:|---:|---:|')
    for name, x, y in DIFFS:
        vals = ['—' if cells[x][a] is None or cells[y][a] is None else
                f"{cells[x][a]['mean'] - cells[y][a]['mean']:+.3f}" for a in ASPECTS]
        print(f'| {name} | {x} vs {y} | ' + ' | '.join(vals) + ' |')


if __name__ == '__main__':
    main()
