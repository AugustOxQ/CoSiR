"""Summarize the r11 retrieval queue outputs as markdown tables for the reports (reads files only).

  python rt_summary.py <outputs>   (the full run: /data/SSD2/cosir_r11/20261029_retrieval/outputs)
"""
import json
import sys
from pathlib import Path

O = Path(sys.argv[1])
A = ('emotion', 'style', 'genre')
ARMS = ('cstar', 'plain', 'instructed', 'clip', 'siglip2', 'percept', 'crl')


def rd(p):
    return json.loads((O / p).read_text())


def f4(x):
    return f'{x:.4f}'


def iv(c):
    return f"{c['d']:+.4f} [{c['interval'][0]:+.4f}, {c['interval'][1]:+.4f}]"


def lab(x):
    return '-' if x is None else x


print('## C13 re-derivations')
for st in ('stage0a', 'stage1', 'stage2', 'stage3', 'stage4', 'stage5'):
    p = O / st / f'rederive_{st}.json'
    if p.is_file():
        d = rd(f'{st}/rederive_{st}.json')
        print(f"- {st}: agree={d['agree']} n_checked={d['n_checked']} disagreements={len(d['disagreements'])} tol={d['tol']}")
    else:
        print(f'- {st}: missing')

if (O / 'stage0b/check.json').is_file():
    d = rd('stage0b/check.json')
    print('\n## Stage 0b')
    for it in d['items']:
        v = it['value']
        diff = '' if not isinstance(v, (int, float)) or not isinstance(it.get('reference'), (int, float)) \
            else f" diff={abs(v - it['reference']):.2e}"
        print(f"- {it['name']}: pass={it['pass']} value={v}{diff}")
    print({k: v for k, v in d.items() if k != 'items'})

r = rd('stage0a/ranges.json')
print('\n## r_a:', {a: round(r['map'][a]['r'], 7) for a in A}, 'nDCG', round(r['ndcg']['emotion']['r'], 8))


def sc_table(stage, ids):
    v = rd(f'stage{stage}/verdict.json')
    print(f"\n## Stage {stage}: GO={v.get('go', v.get('switch_go'))} draws={v['draws']} seconds={v['seconds']:.0f}")
    if 'map' in v:
        print('| arm | ' + ' | '.join(A) + ' |\n|---|---|---|---|')
        for arm in v['arms']:
            row = [f4(v['map'][arm][a]['mean']) for a in A]
            print(f'| {arm} | ' + ' | '.join(row) + ' |')
    print('\n| SC | aspect | candidate | comparator | d [95%] | r | outcome | bar | pass |\n|---|---|---|---|---|---|---|---|---|')
    for sid in ids:
        s = v['sc'][sid]
        for c in s['comparisons']:
            print(f"| {sid} | {c['aspect']} | {'/'.join(map(lab, c['candidate']))} | {'/'.join(map(lab, c['comparator']))} "
                  f"| {iv(c)} | {c['r']:.4f} | {c['outcome']} | {c.get('bar', '')} | {s['pass']} |")
    if 'readouts' in v:
        print('picks:', v['readouts'])
    return v


sc_table(1, [f'SC{i}' for i in range(1, 7)])
v2 = sc_table(2, ['SC7', 'SC8', 'SC9'])
sw = rd('stage2/switch.json')
print('\nSwitch tables (rows: head, columns: label aspect):')
for arm in ARMS:
    if arm not in sw:
        continue
    for h, cols in sw[arm].items():
        print(f'  {arm:10s} head {h:8s} ' + '  '.join(f'{a} {f4(cols[a])}' for a in A))
print('  chance', {a: f4(sw['chance'][a]) for a in A})

v3 = rd('stage3/verdict.json')
print(f"\n## Stage 3: reading={v3['reading']} stage1={v3['stage1_outcomes']}")
for arm, b in v3['ndcg'].items():
    print(f"  {arm}: nDCG {b['emotion']['mean']:.4f}")
for sid, s in v3['sc'].items():
    c = s['comparisons'][0]
    print(f"  {sid}: {iv(c)} r={c['r']:.5f} {c['outcome']}")

s4, pk = rd('stage4/scores.json'), rd('stage4/picks.json')
print(f"\n## Stage 4: seconds={s4['seconds']:.0f} dropped={s4['dropped']}")
print('| arm | pick | pick score | best score (name) | R0 mAP e/s/g | pick mAP e/s/g | n kept |\n|---|---|---|---|---|---|---|')
for arm in ARMS:
    if arm not in s4:
        continue
    sc = s4[arm]
    best = max(sc.items(), key=lambda kv: kv[1]['score'] if kv[1]['kept'] else -1e9)
    p = sc[pk[arm]]
    r0 = sc['none+k0']
    print(f"| {arm} | {pk[arm]} | {p['score']:+.2f} | {best[1]['score']:+.2f} ({best[0]}) | "
          + '/'.join(f4(r0['map'][a]) for a in A) + ' | ' + '/'.join(f4(p['map'][a]) for a in A)
          + f" | {sum(x['kept'] for x in sc.values())}/{len(sc)} |")
print('Top 5 per arm by score (kept only):')
for arm in ARMS:
    if arm not in s4:
        continue
    top = sorted(((k, x) for k, x in s4[arm].items() if x['kept']), key=lambda kv: -kv[1]['score'])[:5]
    print(f'  {arm}: ' + ', '.join(f"{k} {x['score']:+.2f}" for k, x in top))

sc_table(5, [f'SC{i}' for i in range(10, 16)])

d = rd('diag/diag.json')
print('\n## Diagnostics')
print('chance', {a: f4(d['chance'][a]) for a in A})
print('c* seed spread (held) vs r:', {a: (round(d['cstar_seed_spread'][a]['spread'], 5), round(d['cstar_seed_spread'][a]['r'], 5)) for a in A})
for st in ('stage1', 'stage5'):
    print(f'P@10 {st}:')
    for arm, b in d['p10'][st].items():
        print(f'  {arm:10s} ' + '  '.join(f'{a} {f4(b[a])}' for a in A))
print('Per-class AP, c* minus instructed (stage 1):')
for a in A:
    pc_c, pc_i = d['per_class_ap']['cstar'][a], d['per_class_ap']['instructed'][a]
    rows = sorted(((k, pc_c[k]['ap'] - pc_i[k]['ap'], pc_c[k]['n'], pc_c[k]['name']) for k in pc_c), key=lambda t: -t[1])
    print(f'  {a}: ' + ', '.join(f"{(nm or k)}:{dd:+.4f}(n={n})" for k, dd, n, nm in rows))
if 'crl' in d:
    print('CRL:', {a: (d['crl'][a]['count'], d['crl'][a]['label_name_matches']) for a in A})
    print('CRL overlap between aspects:', {k: len(v) for k, v in d['crl_overlap_between_aspects'].items()})
