#!/bin/bash
# usage: att_viewer_test.sh <port>   (server must already run); checks endpoints for two paintings
P=${1:-8765}; U=http://127.0.0.1:$P
PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/CoSiR/bin/python - "$U" <<'PY'
import sys, json, urllib.request as r, numpy as np
U = sys.argv[1]; g = lambda p: r.urlopen(U + p).read()
m = json.loads(g('/api/meta')); print('meta', len(m['ks']), list(m['views']), m['bands'], len(m['layers']))
assert g('/').startswith(b'<!doctype') and len(g('/att_viewer.js')) > 1000 and len(g('/att_viewer.css')) > 100
for k in m['ks'][:2]:
    pi = json.loads(g(f'/api/painting?k={k}')); gr, gc = pi['grid_hw']; nc = len(pi['captions'])
    im = g(f'/api/image?k={k}'); assert im[:2] == b'\xff\xd8'
    print(k, pi['style'], pi['genre'], pi['captions'][0]['emotion'], nc, 'caps', len(im), 'jpeg bytes')
    for mod in ('img', 'txt', 'joint'):
        for layer in ('layers 16-27', 'last layer', 'all layers', '0', '13', '27'):
            for cap in range(min(nc, 2)):
                q = json.loads(g(f'/api/maps?k={k}&mod={mod}&cap={cap}&layer=' + layer.replace(' ', '%20')))
                assert len(q['views']) == 4
                for v, d in q['views'].items():
                    sh = d['shares']; assert abs(sum(sh.values()) - 1) < 2e-3, sh
                    if mod == 'txt': assert d['image_map'] is None
                    else:
                        a = np.array(d['image_map']); assert a.shape == (gr, gc) and abs(a.sum() - 1) < 1e-3, a.sum()
                    if mod == 'img': assert d['caption_tokens'] is None
                    else:
                        w = [t['weight'] for t in d['caption_tokens']]; assert abs(sum(w) - 1) < 1e-3
                    if mod == 'joint':
                        wm = np.array(d['word_maps']); assert wm.shape == (len(w), gr, gc)
                        assert np.abs(wm.sum((1, 2)) - 1).max() < 1e-3
                        assert d['word_maps_band'] == ('layers 16-27' if layer in ('0', '13', '27') else layer if layer == 'layers 16-27' else d['word_maps_band'])
    print('ok', k)
q = json.loads(g(f'/api/maps?k={k}&mod=joint&cap=0&layer=layers%2016-27'))['views']['default']
print('example shares', {a: round(b, 3) for a, b in q['shares'].items()}, 'tokens', ''.join(t['text'] for t in q['caption_tokens'])[:60])
PY
