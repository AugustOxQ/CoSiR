"""Diagnostic: is the raw-map gap between the shortcut and plain passes bf16 rounding?
Reference: the shortcut with the model in float32 (exact against plain float32 passes, att_diag_fp32.py).
Compares L1(shortcut bf16, ref) with L1(plain bf16, ref): equal sizes mean the shortcut adds no error of its own."""
import json
import numpy as np
import att_extract as X
import att_core as C
VIEWS, KS = ['default', 'style_word'], range(4)
ex = X.Extractor('cuda', VIEWS)
st = lambda: dict(t_wait=0, t_prefix=0, t_suffix=0, t_txt=0, n_prefix=0, n_img=0, n_joint=0, n_txt=0, rowsum_dev=0, t_full=0)
preps = {k: ex.prep(k, True) for k in KS}
short, plain = {}, {}
for k in KS:
    _, data, _ = ex.process(preps[k], st(), False)
    short[k] = data
    plain[k] = {}
    for v in VIEWS:
        vd = preps[k]['views'][v]
        items = [(p, None) for p in data[v]['img']] + \
                [(p, (int(np.nonzero(p['ttype'] == 3)[0][0]), int(np.nonzero(p['ttype'] == 3)[0][-1]) + 1)) for p in data[v]['joint']]
        plain[k][v] = [dict(ex.full_pass(p['ids'], vd, rng), ttype=p['ttype']) for p, rng in items]
ex.enc.m.float(); ex.dtype = ex.torch.float32
ref = {k: ex.process(preps[k], st(), False)[1] for k in KS}
out = {name: {b[0]: [] for b in C.BANDS} for name in ('img_short', 'img_plain', 'img_short_vs_plain', 'cross_short', 'cross_plain')}
for k in KS:
    for v in VIEWS:
        r_items = ref[k][v]['img'] + ref[k][v]['joint']
        s_items = short[k][v]['img'] + short[k][v]['joint']
        for i, (r, s, p) in enumerate(zip(r_items, s_items, plain[k][v])):
            for bi, band in enumerate(C.BANDS):
                rm, sm, pm = (C.band_map(x, band)[0] for x in (r, s, dict(layers=p['layers'], ttype=p['ttype'])))
                out['img_short'][band[0]].append(float(C.l1_norm_maps(sm, rm)))
                out['img_plain'][band[0]].append(float(C.l1_norm_maps(pm, rm)))
                out['img_short_vs_plain'][band[0]].append(float(C.l1_norm_maps(sm, pm)))
                if i > 0:
                    rc = np.asarray(r['cross'][bi], np.float64)
                    out['cross_short'][band[0]] += C.l1_norm_maps(np.asarray(s['cross'][bi], np.float64), rc).tolist()
                    out['cross_plain'][band[0]] += C.l1_norm_maps(np.asarray(p['cross'][bi], np.float64), rc).tolist()
res = {name: {b: dict(mean=round(float(np.mean(x)), 4), max=round(float(np.max(x)), 4)) for b, x in d.items()} for name, d in out.items()}
print(json.dumps(res, indent=1))
