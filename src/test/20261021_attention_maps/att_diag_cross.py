"""Diagnostic: magnitude of cross-map values and the float16 quantisation error (shortcut vs plain pass)."""
import numpy as np
import att_extract as X
import att_core as C
ex = X.Extractor('cuda', ['default'])
d = ex.prep(0, False); vd = d['views']['default']; P = vd['P']
cache, RP = ex.prefix_pass(vd)
res = ex.suffix_batch(vd, cache, RP, [vd['ids_img'][P:]] + [f[P:] for f in vd['joint']])
for i in (1, 2):
    f = vd['joint'][i - 1]; tt = C.token_types(f, 'joint'); cap = np.nonzero(tt == 3)[0]
    sc = res[i]['R_rows'][cap - P].numpy()
    full = ex.full_pass(f, vd, (int(cap[0]), int(cap[-1]) + 1))['cross']
    print('caption', i, 'n_cap_tok', len(cap), 'rowsum shortcut', np.round(sc.sum(1)[:6], 5), 'full', np.round(full.sum(1)[:6], 5))
    print(' value range', sc.min(), sc.max(), 'median', np.median(sc))
    print(' L1 per row shortcut-vs-full', np.round(C.l1_norm_maps(full, sc), 3)[:8])
    print(' L1 fp16 vs fp32 shortcut', np.round(C.l1_norm_maps(sc.astype(np.float16).astype(np.float32), sc), 3)[:8])
    print(' abs diff max', np.abs(full - sc).max(), 'full max', full.max())
