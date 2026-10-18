"""CLIP B/32 content graphs vs Qwen content graphs; labels only for fit graph statistics."""
import json
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '20261017_image_transfer_pilot'))
sys.path.insert(0, str(HERE.parent / '20261017_prompted_aspect_views'))
sys.path.insert(0, str(HERE))
from itp_data import atomic_json, load_frozen, split
from pav_score import mutual_pairs, references
from apc_data import l2, paint_mean
import pbp_graphs as pg


def main():
    t0 = time.monotonic()
    s = load_frozen()
    _, fit, _ = split(s)
    with np.load(HERE / 'graphs/fit.npz', allow_pickle=False) as z:
        node_ids = z['node_ids']
        saved = {k: set(map(tuple, z[k].tolist())) for k in pg.GRAPH_NAMES}
    assert len(fit) == 13400 and np.array_equal(node_ids, fit), 'fit order mismatch'
    print('fit order matches saved node_ids', flush=True)

    # CLIP caption text (caption level, aligned to s['rows'] / s['pid'])
    name, txt, _ = next(references(s))
    assert name == 'clip_text' and len(txt) == len(s['pid']) == len(s['rows'])
    pc = len(s['first'])
    assert np.array_equal(s['pid'][s['first']], np.arange(pc)), 'first/pid inconsistent'
    # CLIP image at the painting's first caption row (same source as pav's clip_image)
    from src.data.artelingo import load_artelingo
    d = load_artelingo()
    assert np.array_equal(d.sample_ids[s['rows']], s['sample_ids'])
    img = l2(d.img_features[s['rows'][s['first']]])
    txtp = l2(paint_mean(txt, s['pid'], pc))
    assert img.shape[0] == txtp.shape[0] == pc and np.isfinite(img).all() and np.isfinite(txtp).all()
    print(f'join ok: {len(s["rows"])} captions, {pc} paintings, img {img.shape}, txt {txtp.shape}', flush=True)

    gi = mutual_pairs(np.ascontiguousarray(img[fit])); print('Ic', len(gi), time.monotonic() - t0, flush=True)
    gd = mutual_pairs(np.ascontiguousarray(txtp[fit])); print('Dc', len(gd), time.monotonic() - t0, flush=True)
    gc = gd | gi
    e, c = saved['E'], saved['C']
    graphs = dict(Ic=gi, Dc=gd, Cc=gc, E_and_Cc=e & gc, E_minus_Cc=e - gc, Cc_minus_E=gc - e)
    labels = pg.label_vectors(s, fit)
    n = len(fit)
    out = dict(n_nodes=n, graphs={}, fractions={
        'E_intersect_Cc_over_E': len(e & gc) / len(e),
        'E_intersect_Dc_over_E': len(e & gd) / len(e),
        'E_intersect_Ic_over_E': len(e & gi) / len(e),
        'Cc_intersect_C_over_Cc': len(gc & c) / len(gc),
        'Cc_intersect_C_over_C': len(gc & c) / len(c)})
    for k, v in graphs.items():
        out['graphs'][k] = pg.graph_stats(v, labels)
    qwen = json.loads((HERE / 'graphs/stats.json').read_text())
    out['qwen'] = {k: qwen['graphs'][k] for k in ('E', 'C', 'P', 'EC')}
    out['qwen_fractions'] = qwen['fractions']
    atomic_json(HERE / 'clip_content_stats.json', out)

    def row(k, g):
        l = [g['labels'][x]['lift'] for x in ('emotion', 'genre', 'style')]
        return f"| {k} | {g['n_pairs']} | " + ' | '.join(f'{v:.4f}' for v in l) + ' |'
    lines = ['| Graph | Pairs | Emotion lift | Genre lift | Style lift |', '|---|---:|---:|---:|---:|']
    lines += [row(k + ' (Qwen)', g) for k, g in out['qwen'].items()]
    lines += [row(k + ' (CLIP)', g) for k, g in out['graphs'].items()]
    lines += ['', 'Overlap: ' + '; '.join(f'{k}={v:.6f}' for k, v in out['fractions'].items()),
              'Qwen overlap: ' + '; '.join(f'{k}={v:.6f}' for k, v in qwen['fractions'].items()), '']
    (HERE / 'clip_content_stats.md').write_text('\n'.join(lines))
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=8):
        main()
