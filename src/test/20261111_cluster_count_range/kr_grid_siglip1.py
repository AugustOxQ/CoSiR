"""Cluster-count grid for the SigLIP 1 rows of the new Table 2 that the r17 sweep file does not hold.

Plumbing only. `+ TURTLE` on plain SigLIP 1 reuses `kr_grid.Grid` with the encoder switched to `siglip1`: r18's saved
fits (`parts/turtle/artelingo/siglip1/plain/k{K}__start{0,1,2}.npz`, no refit), scored by r17's
`ns_metrics.score_partitions`, mean over the three starts. Gate: r18's six recorded cells
(`SC5/artelingo/siglip1/{aspect}/ami/{k}/plain_turtle`, class count and 20), max |diff| <= 1e-6.

Plain, PercepT (matched) and ours on SigLIP 1 need no scoring: the r17 sweep file
`/data/SSD2/cosir_r17/20261103_new_scores/outputs/stage3/sweep/artelingo__siglip1.json` holds every grid count.
CLAY and CRL on SigLIP 1 come from r28 (`20261112_siglip_main`) and are added later.

    python kr_grid_siglip1.py --rows turtle --out records/cells_siglip1.json
"""
import argparse
import json
import time
from pathlib import Path

import kr_grid  # noqa: E402  (puts r17 and r18 round folders on sys.path)

ENC = 'siglip1'
ROWS = ('turtle',)
REF_ID = {'turtle': 'SC5/artelingo/siglip1/{a}/ami/{k}/plain_turtle'}


def references():
    lab = {c['id']: c['comp'] for c in json.loads(kr_grid.LAB.read_text())['cells']}
    ref = {}
    for a, kc in kr_grid.CLASS_COUNT.items():
        for k in (kc, 20):
            i = REF_ID['turtle'].format(a=a, k=k)
            ref[('turtle', a, k)] = (i, lab[i])
    assert len(ref) == 6
    return ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rows', default=','.join(ROWS))
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    rows = a.rows.split(',')
    assert set(rows) <= set(ROWS), rows
    kr_grid.ENC = ENC                       # read at call time by Grid.parts and Grid.inputs
    ref = references()
    g = kr_grid.Grid()
    out, t0 = [], time.time()
    for row in rows:
        for asp, ks in kr_grid.GRID.items():
            for k in ks:
                c = g.cell(row, asp, k)
                c['id'] = f'{row}_{ENC}/{asp}/ami/{k}'
                c['row'] = f'{row}_{ENC}'
                c['encoder'] = ENC
                r = ref.get((row, asp, k))
                c['gate'] = r is not None
                if r:
                    c['reference_id'], c['reference'], c['diff'] = r[0], r[1], c['mean'] - r[1]
                out.append(c)
                print(f"{c['id']:34s} mean={c['mean']:.10f} " + (f"ref={r[1]:.10f} diff={c['diff']:+.2e} " if r else '')
                      + f"used={c['n_clusters_used']}", flush=True)
    hdr = dict(protocol='TURTLE: r18 saved fits on plain SigLIP 1 (starts 0,1,2, no refit), AMI on rows with y >= 0, '
                        'mean over the 3 starts (ns_metrics.score_partitions)',
               code_commit=kr_grid.git_head(), encoder=ENC, held_split=kr_grid.HELD, total_seconds=time.time() - t0)
    Path(a.out).write_text(json.dumps(dict(header=hdr, cells=out), indent=1) + '\n')
    gated = [c for c in out if c['gate']]
    m = max(abs(c['diff']) for c in gated)
    print(f'GATE: {len(gated)} cells, max |diff| = {m:.3e}, {"PASS" if m <= 1e-6 and len(gated) == 6 else "FAIL"}')


if __name__ == '__main__':
    main()
