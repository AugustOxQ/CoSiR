"""Compare the Codex re-derivation with our stored scores (controller check)."""
import json
import sys
R = sys.argv[1]
O = '/project/CoSiR/src/test/20261018_percept_core/outputs'
c = json.load(open(f'{R}/rederive_results.json'))
ours = json.load(open(f'{O}/pc_rows_results.json'))
row1 = json.load(open(f'{O}/pc_results.json'))['row1']
L = ('emotion', 'style', 'genre')
print('img keys', [k for k in c['part_a']['runs'][0]['img'] if k != 'partitions'])
worst = 0.0
n = 0
for run in c['part_a']['runs']:
    stem = run['file'].removesuffix('.npz')
    for side in ('img', 'cap'):
        for lab in L:
            theirs = run[side]['scores'][lab]['mean'] if 'scores' in run[side] else run[side][lab]['mean']
            mine = ours['outputs'][stem]['ami'][side][lab]['mean']
            worst = max(worst, abs(theirs - mine)); n += 1
print(f'part A per-run side scores: {n} values, max abs diff {worst:.2e}')
w1 = 0.0
for kind, ckey, bkey in (('clip_text', 'cap_clip', 'both_clip'), ('roberta', 'cap_roberta', 'both_roberta')):
    for lab in L:
        w1 = max(w1, abs(c['part_b']['img_clip']['scores'][lab]['mean'] if 'scores' in c['part_b']['img_clip'] else 0) * 0)
        ti = c['part_b']['img_clip'].get('scores', c['part_b']['img_clip'])[lab]['mean']
        tc = c['part_b'][ckey].get('scores', c['part_b'][ckey])[lab]['mean']
        tb = c['part_b'][bkey][lab]['mean'] if isinstance(c['part_b'][bkey][lab], dict) else c['part_b'][bkey][lab]
        r = row1[kind]
        w1 = max(w1, abs(ti - r['img'][lab]['mean']), abs(tc - r['cap'][lab]['mean']), abs(tb - r['both_mean'][lab]))
print(f'part B row 1: max abs diff {w1:.2e}')
