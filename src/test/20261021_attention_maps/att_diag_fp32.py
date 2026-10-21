"""Diagnostic: shortcut against plain full passes with the model in float32 (no bf16 noise).
A difference at 1e-5 level means the shortcut (cache, positions, suffix rollout) is exact; the bf16 gap is rounding."""
import json
import numpy as np
import att_extract as X
import att_core as C
ex = X.Extractor('cuda', ['default', 'style_word'])
ex.enc.m.float(); ex.dtype = ex.torch.float32
recs = []
for k in range(3):
    d = ex.prep(k, False)
    st = dict(t_wait=0, t_prefix=0, t_suffix=0, t_txt=0, n_prefix=0, n_img=0, n_joint=0, n_txt=0, rowsum_dev=0, t_full=0)
    painting, data, gb = ex.process(d, st, True)
    recs += gb
print(json.dumps(X.gate_b.__globals__['summarise'] and {key: X.summarise(sum([r[key] for r in recs], [])) for key in ('cos', 'roll_maxabs', 'img_l1', 'cross_l1')}))
