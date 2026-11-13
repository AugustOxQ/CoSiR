"""Compare D16's re-derivation of the lambda = 0 row with stage2_lam0.json (run chat 3, 2026-11-13; as the 11-11 compare)."""
import json

R = '/data/SSD2/cosir_r21/20261111_dec_heads/'
ARM = 'dec-plain_l0_w5_S2'
TOL = 1e-6
METRIC = {'ami_class': 'ami_class_count', 'ami20': 'ami_20', 'map': 'map'}
d16 = json.load(open(R + 'd16/lam0/d16_table.json'))
ours = json.load(open(R + 'scores/stage2_lam0.json'))
failed, missing, n_num, n_word, worst, seen = [], [], 0, 0, 0.0, set()
for c in d16['cells']:
    if c['arm'] != ARM:
        continue
    key = METRIC[c['metric']]
    seen.add((c['aspect'], key))
    o = ours['aspects'][c['aspect']]['arms'][ARM][key]
    pairs = [('d', c['d'], o['change']), ('lo', c['lo'], o['lo']), ('hi', c['hi'], o['hi']),
             ('arm_mean', c['arm_mean'], o['arm']), ('cstar_mean', c['cstar_mean'], o['cstar'])]
    pairs += [(f'arm_s{s}', v, o['arm_seed'][str(s)]) for s, v in enumerate(c['arm_seeds'])]
    pairs += [(f'cstar_s{s}', v, o['cstar_seed'][str(s)]) for s, v in enumerate(c['cstar_seeds'])]
    for name, a, b in pairs:
        n_num += 1
        worst = max(worst, abs(a - b))
        if abs(a - b) > TOL:
            failed.append(f"{c['aspect']} {key} {name}: d16 {a} ours {b}")
    n_word += 1
    if c['word'] != o['word']:
        failed.append(f"{c['aspect']} {key} word: d16 {c['word']} ours {o['word']}")
for asp in ('emotion', 'style', 'genre'):
    for key in METRIC.values():
        if (asp, key) not in seen:
            missing.append(f'{asp} {key}')
doc = dict(status='PASS' if not failed and not missing and len(seen) == 9 else 'FAIL', arm=ARM, cells=len(seen), numbers=n_num,
           words=n_word, max_abs=worst, failed=failed, missing=missing, tolerance=TOL,
           ours='scores/stage2_lam0.json', d16='d16/lam0/d16_table.json')
json.dump(doc, open(R + 'd16/lam0/compare.json', 'w'), indent=1)
print(f"D16 lambda = 0 row: {doc['status']} cells={doc['cells']} numbers={n_num} words={n_word} max_abs={worst:.2e} "
      f"failed={len(failed)} missing={len(missing)}")
