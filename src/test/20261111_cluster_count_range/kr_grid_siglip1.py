"""Cluster-count grid for the SigLIP 1 rows of the new Table 2 that the r17 sweep file does not hold.

Plumbing only; every loader and scorer is imported unchanged.
- `+ TURTLE` on plain SigLIP 1 reuses `kr_grid.Grid` with the encoder switched to `siglip1`: r18's saved fits
  (`parts/turtle/artelingo/siglip1/plain/k{K}__start{0,1,2}.npz`, no refit), scored by r17's `ns_metrics.score_partitions`,
  mean over the three starts. Gate: r18's six recorded cells (`SC5/artelingo/siglip1/{aspect}/ami/{k}/plain_turtle`,
  class count and 20).
- CLAY and CRL on SigLIP 1 come from r28 (`20261112_siglip_main/r28_clay_crl.py`, read only): its `clay_vectors` /
  `crl_vectors` (SigLIP 1 image rows and SigLIP 1 text rows) and its `clu_cell` (k-means `ns_metrics.partitions`, states
  0 to 2, AMI on labelled rows) at each K. CLAY's text set is the one r28 used for its clustering cell. Gate: r28's
  class-count cells (`records/clay_crl.json`, `cells[artelingo/{aspect}/ami/{method}].siglip1`).
Gate tolerance 1e-6 on every reference cell.

Plain, PercepT (matched) and ours on SigLIP 1 need no scoring: the r17 sweep file
`/data/SSD2/cosir_r17/20261103_new_scores/outputs/stage3/sweep/artelingo__siglip1.json` holds every grid count.

    python kr_grid_siglip1.py --rows turtle,clay,crl --out records/cells_siglip1.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import kr_grid  # noqa: E402  (puts r17 and r18 round folders on sys.path)

ENC = 'siglip1'
ROWS = ('turtle', 'clay', 'crl')
R28 = kr_grid.ROUNDS / '20261112_siglip_main'
R28_CELLS = R28 / 'records' / 'clay_crl.json'
TURTLE_ID = 'SC5/artelingo/siglip1/{a}/ami/{k}/plain_turtle'
TOL = 1e-6


def r28_module():
    if str(R28) not in sys.path:
        sys.path.insert(0, str(R28))
    import r28_clay_crl
    return r28_clay_crl


def references(rows):
    """{(row, aspect, k): (id, value)} of the recorded cells for `rows`."""
    ref = {}
    if 'turtle' in rows:
        lab = {c['id']: c['comp'] for c in json.loads(kr_grid.LAB.read_text())['cells']}
        for a, kc in kr_grid.CLASS_COUNT.items():
            for k in (kc, 20):
                i = TURTLE_ID.format(a=a, k=k)
                ref[('turtle', a, k)] = (i, lab[i])
    r28 = json.loads(R28_CELLS.read_text())['cells']
    for m in ('clay', 'crl'):
        if m in rows:
            for a, kc in kr_grid.CLASS_COUNT.items():
                i = f'artelingo/{a}/ami/{m}'
                ref[(m, a, kc)] = (f'r28 {R28_CELLS.name}:{i}.siglip1', r28[i]['siglip1'])
    return ref


def clay_text_set(a):
    ts = json.loads(R28_CELLS.read_text())['cells'][f'artelingo/{a}/ami/clay']['text_set']
    assert ts in ('A', 'B'), ts
    return ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rows', default=','.join(ROWS))
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    rows = a.rows.split(',')
    assert set(rows) <= set(ROWS), rows
    kr_grid.ENC = ENC                       # read at call time by Grid.parts and Grid.inputs
    ref = references(rows)
    g = kr_grid.Grid()
    r28 = r28_module() if {'clay', 'crl'} & set(rows) else None
    out, t0 = [], time.time()
    for row in rows:
        for asp, ks in kr_grid.GRID.items():
            if row == 'clay':
                ts = clay_text_set(asp)
                x = r28.clay_vectors('artelingo', kr_grid.HELD, asp, ts, ENC)
                src = f'r28_clay_crl.clay_vectors(artelingo, {kr_grid.HELD}, {asp}, text set {ts}, siglip1)'
            elif row == 'crl':
                x = r28.crl_vectors('artelingo', kr_grid.HELD, asp, ENC)
                src = f'r28_clay_crl.crl_vectors(artelingo, {kr_grid.HELD}, {asp}, siglip1)'
            for k in ks:
                if row == 'turtle':
                    c = g.cell(row, asp, k)
                else:
                    t1 = time.time()
                    r = r28.clu_cell(x, 'artelingo', asp, k)
                    c = dict(row=row, aspect=asp, k=k, mean=r['value'], per_state=r['per_state'],
                             source_inputs=src, n_clusters_used=r.get('n_clusters_used'), seconds=time.time() - t1)
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
    hdr = dict(protocol=('TURTLE: r18 saved fits on plain SigLIP 1 (starts 0,1,2, no refit), AMI on rows with y >= 0, mean '
                         'over the 3 starts (ns_metrics.score_partitions). CLAY, CRL: r28_clay_crl vectors on SigLIP 1 and '
                         'r28_clay_crl.clu_cell (ns_metrics.partitions, states 0,1,2, mean AMI on labelled rows)'),
               code_commit=kr_grid.git_head(), encoder=ENC, held_split=kr_grid.HELD, rows=rows,
               total_seconds=time.time() - t0)
    Path(a.out).write_text(json.dumps(dict(header=hdr, cells=out), indent=1) + '\n')
    gated = [c for c in out if c['gate']]
    m = max(abs(c['diff']) for c in gated)
    print(f'GATE: {len(gated)} of {len(ref)} cells, max |diff| = {m:.3e}, '
          f'{"PASS" if m <= TOL and len(gated) == len(ref) else "FAIL"}')


if __name__ == '__main__':
    main()
