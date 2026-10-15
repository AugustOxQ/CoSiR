"""fx-desc item 2: run_r9_descriptive's copy of H5's loader and pass check gives the same results as round 6's.

Run from the round folder: python fx_desc_h5_equiv.py  (CPU, about 1-2 minutes: two bootstraps of 9 checks x 5,000).
This script (not the runner) imports r6_descriptive, from the main checkout. Writes fx_desc_h5_equiv.json.
"""
import copy
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r9_common as R
import run_r9_descriptive as RD
import numpy as np

sys.path.append(str(R.R6D))
import r6_common as R6
import r6_descriptive as D6
import r6_score as S6
import r6_stats as ST6

out = {}
out['core_scorers_equal'] = tuple(S6.CORE_SCORERS) == RD.H5_CORE
out['array_extra_equal'] = tuple(D6.ARRAY_EXTRA) == RD.H5_EXTRA
out['rule_sha_equal'] = R6.RULE_SHA256 == R.R6_RULE_SHA256
out['quantities_equal'] = {c: ST6.QUANTITIES[c][2] == RD.ST.QUANTITIES[c][2] for c in ST6.CHECKS + ST6.SECONDARY}
out['checks_equal'] = tuple(ST6.CHECKS) == tuple(RD.ST.CHECKS) and tuple(ST6.SECONDARY) == tuple(RD.ST.SECONDARY)

res = R.R6D / 'results'
rec = json.loads((res / 'held_pass.json').read_text())
a = D6.load_core(res / 'held_arrays.npz', rec['seeds'], R.N_PER_PAIR)
b = RD.h5_load_core(res / 'held_arrays.npz', rec['seeds'], R.N_PER_PAIR)
same = a.keys() == b.keys()
for s in a:
    same &= a[s].keys() == b[s].keys()
    same &= all(np.array_equal(a[s][k], b[s][k]) and a[s][k].dtype == b[s][k].dtype for k in ('cl', 'pair_index'))
    for sc in a[s]['pa']:
        for m in a[s]['pa'][sc]:
            x, y = a[s]['pa'][sc][m], b[s]['pa'][sc][m]
            same &= bool(x.dtype == y.dtype and np.array_equal(x, y))
out['load_core_equal_real_h5_held'] = bool(same)
D6.check_pass(a, rec)
RD.h5_check_pass(b, rec)
out['check_pass_both_accept_real_h5_held'] = True
neg = {}
for name, edit in {'n_P3': lambda r: r['checks']['P3'].__setitem__('n', r['checks']['P3']['n'] + 1),
                   'point_S1': lambda r: r['secondary']['S1'].__setitem__('point', r['secondary']['S1']['point'] + 1e-9),
                   'ci_P7': lambda r: r['checks']['P7']['ci95'].__setitem__(0, r['checks']['P7']['ci95'][0] + 1e-9),
                   'quantity_P1': lambda r: r['checks']['P1'].__setitem__('quantity', 'x'),
                   'n_clusters': lambda r: r.__setitem__('n_clusters', r['n_clusters'] + 1)}.items():
    r2 = copy.deepcopy(rec)
    edit(r2)
    res2 = []
    for f, core in ((D6.check_pass, a), (RD.h5_check_pass, b)):
        try:
            f(core, r2)
            res2.append('accepted')
        except AssertionError:
            res2.append('refused')
    neg[name] = res2
out['negative_controls_[r6, r9copy]'] = neg
(HERE / 'fx_desc_h5_equiv.json').write_text(json.dumps(out, indent=1) + '\n')
print(json.dumps(out, indent=1))
