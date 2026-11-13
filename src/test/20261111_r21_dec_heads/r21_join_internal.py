"""Join r21 diag (SI, DBI, CHI; paired by seed against c*) with the stage 2 changes (AMI at class count, mAP).
Since run chat 3 (2026-11-13) it reads `stage2_lam0.json` (the stage 2 arms plus the lambda = 0 control) and checks
that the old arms' internal scores equal `diag_before_lam0.json`. Section 4 keeps the three DEC forms only (15 pairs).
Run: python r21_join_internal.py > scores/internal_join_lam0.txt"""
import json
import numpy as np

S = '/data/SSD2/cosir_r21/20261111_dec_heads/scores/'
diag = json.load(open(S + 'diag.json'))
tab = json.load(open(S + 'stage2_lam0.json'))
old = json.load(open(S + 'diag_before_dbi_chi.json'))
before_lam0 = json.load(open(S + 'diag_before_lam0.json'))
ASP = ('emotion', 'style', 'genre')
FORMS = {'dec-plain_l10_w5_S2': 'plain', 'dec-start_l10_w5_S2': 'start', 'dec-teacher_l10_w5_S2': 'teacher'}
ARMS = {**FORMS, 'dec-plain_l0_w5_S2': 'lam0'}               # the lambda = 0 control (W 5, S2), rows 2 and 3 only

# 1. the silhouette must be unchanged against the run of 2026-11-11
worst = 0.0
for src, per in old['sources'].items():
    for a in ASP:
        for sd, row in per[a]['seeds'].items():
            worst = max(worst, abs(row['silhouette'] - diag['sources'][src][a]['seeds'][sd]['silhouette']))
print(f'silhouette vs the 2026-11-11 run: max |diff| {worst:.2e}')
worst = 0.0
for src, per in before_lam0['sources'].items():
    for a in ASP:
        for sd, row in per[a]['seeds'].items():
            for key in ('silhouette', 'dbi', 'chi'):
                worst = max(worst, abs(row[key] - diag['sources'][src][a]['seeds'][sd][key]))
print(f'SI, DBI, CHI of the old sources vs the 4-arm diag (before lambda = 0): max |diff| {worst:.2e}')

# 2. levels (seed mean)
print('\nlevels, seed mean: SI / DBI / CHI')
for src in ('cstar', *ARMS, 'plain', 'teacher'):
    cells = [f"{diag['sources'][src][a]['mean']['silhouette']:.4f} / {diag['sources'][src][a]['mean']['dbi']:.3f} / "
             f"{diag['sources'][src][a]['mean']['chi']:.0f}" for a in ASP]
    print(f"{ARMS.get(src, src):8s} | " + ' | '.join(cells))

# 3. changes from c*, paired by seed; n tighter = seeds where the arm is tighter than c*
print('\nchange from c* (paired by seed): dSI [n tighter] / dDBI [n tighter] / dCHI [n tighter] || dAMI word || dmAP word')
for a in ASP:
    for arm, f in ARMS.items():
        ref = diag['sources']['cstar'][a]['seeds']
        me = diag['sources'][arm][a]['seeds']
        out = []
        for key, tighter in (('silhouette', 1), ('dbi', -1), ('chi', 1)):
            d = np.array([me[s][key] - ref[s][key] for s in sorted(me)])
            fmt = '+.1f' if key == 'chi' else '+.4f'
            out.append(f"{d.mean():{fmt}} [{int((tighter * d > 0).sum())}/5]")
        c = tab['aspects'][a]['arms'][arm]
        ami, mp = c['ami_class_count'], c['map']
        print(f"{a:7s} {f:7s} | " + ' / '.join(out)
              + f" || {ami['change']:+.4f} [{ami['lo']:+.4f}, {ami['hi']:+.4f}] {ami['word']}"
              + f" || {mp['change']:+.4f} [{mp['lo']:+.4f}, {mp['hi']:+.4f}] {mp['word']}")
    rel = {k: diag['sources']['cstar'][a]['mean'][k] for k in ('silhouette', 'dbi', 'chi')}
    print(f"{a:7s} c* level: SI {rel['silhouette']:.4f}, DBI {rel['dbi']:.3f}, CHI {rel['chi']:.0f}")

# 4. per seed: does a tighter test partition go with a higher AMI? (15 arm-seed pairs per aspect)
print('\nper arm and seed, Spearman of dSI, dDBI, dCHI with dAMI (class count; 15 pairs per aspect)')
from scipy.stats import spearmanr
for a in ASP:
    rows = []
    for arm in FORMS:
        c = tab['aspects'][a]['arms'][arm]['ami_class_count']
        ref = diag['sources']['cstar'][a]['seeds']
        me = diag['sources'][arm][a]['seeds']
        for s in sorted(me):
            dami = c['arm_seed'][s] - c['cstar_seed'][s]
            rows.append([me[s]['silhouette'] - ref[s]['silhouette'], me[s]['dbi'] - ref[s]['dbi'],
                         me[s]['chi'] - ref[s]['chi'], dami])
    r = np.array(rows)
    rh = [spearmanr(r[:, i], r[:, 3]).statistic for i in range(3)]
    print(f"{a:7s}: rho(dSI, dAMI) {rh[0]:+.2f}, rho(dDBI, dAMI) {rh[1]:+.2f}, rho(dCHI, dAMI) {rh[2]:+.2f}")
