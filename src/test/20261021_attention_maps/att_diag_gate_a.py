"""Diagnostic: where does the image-embedding gap to the stored arrays come from?
Compares, per view, cosine of stored vs (shortcut | plain eager pass | plain sdpa pass | plain eager bf16 batch-of-1)."""
import sys, json
import numpy as np
import att_extract as X
import att_core as C

ex = X.Extractor('cuda', ['default', 'emotion_focus', 'style_word', 'genre_word'])
torch = ex.torch
res = {}
for k in range(16):
    d = ex.prep(k, False)
    for v, vd in d['views'].items():
        stored = np.asarray(ex.stored_arrays(v)[0][k], np.float32); stored /= np.linalg.norm(stored)
        cache, RP = ex.prefix_pass(vd)
        r = ex.suffix_batch(vd, cache, RP, [vd['ids_img'][vd['P']:]])[0]
        f_eager = ex.full_pass(vd['ids_img'], vd, None)['emb']
        ex.enc.m.set_attn_implementation('sdpa')
        ids = torch.from_numpy(vd['ids_img']).long()[None].cuda()
        mm = (ids == C.IMAGE_PAD).long()
        with torch.no_grad():
            h = ex.enc.m(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=vd['pix'].cuda().bfloat16(),
                         image_grid_thw=vd['thw'].cuda(), mm_token_type_ids=mm).last_hidden_state[0, -1]
        f_sdpa = ex.pool(h).cpu().numpy()
        ex.enc.m.set_attn_implementation('eager')
        for name, e in (('shortcut', r['emb']), ('eager_full', f_eager), ('sdpa_full', f_sdpa)):
            res.setdefault(v, {}).setdefault(name, []).append(float(np.dot(e, stored)))
        res[v].setdefault('sdpa_vs_eager', []).append(float(np.dot(f_sdpa, f_eager)))
for v in res:
    print(v, {n: (round(float(np.mean(x)), 5), round(float(np.min(x)), 5)) for n, x in res[v].items()})
