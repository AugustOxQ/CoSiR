"""Cluster-count grid for the last Table 2 row: PercepT (CLIP L/14 + ModernBERT-GoEmotions, paper settings; r26).

Plumbing only: r17's `ns_metrics.percept_partitions` (k-means, MiniBatchKMeans n_init 3, batch 4096, states 0 to 2, on
the L2-normalised Stage 2 probabilities) and `ns_metrics.ami` are used unchanged, as r26's `po_score.score_probs` does.
Inputs: r26's test probabilities `OUT/held/L1__k40__s{seed}.npz` (key `prob`, [12281, 40], r17's held order), seeds
from r26's `po_scores.json` (cut k40, seeds in the mean). Cell = mean over seeds of the mean over the 3 states; AMI on
rows with label >= 0 (genre 9,949).

    python kr_grid_r26.py --append records/cells.json --list records/percept_clip.json

The 6 cells that r26 recorded (class count and K = 20) are the gate: |diff| <= 1e-6, else nothing is written.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kr_grid as KG            # noqa: E402  (r17/r18 sys.path set-up, GRID, CLASS_COUNT)
import ns_data                  # noqa: E402
import ns_metrics               # noqa: E402

ROW = 'percept_clip'
GRID, CLASS_COUNT = KG.GRID, KG.CLASS_COUNT
SET, HELD = KG.SET, KG.HELD
R26_OUT = Path('/data/SSD2/cosir_r26/20261111_percept_original/outputs')
R26_SCORES = Path('/project/CoSiR-r26/src/test/20261111_percept_original/records/po_scores.json')
TOL = 1e-6


def masked_ami(y, c):
    keep = y >= 0
    return float(ns_metrics.ami(y[keep], np.asarray(c)[keep]))


def load_probs(seeds, names):
    probs = {}
    for s in seeds:
        p = np.load(R26_OUT / 'held' / f'L1__k40__s{s}.npz')['prob']
        ex = np.load(R26_OUT / 'test' / 'k40' / f'L1__s{s}.npz', allow_pickle=False)
        assert ex['names'].tolist() == list(names), f'seed {s}: test export names differ from r17 held order'
        assert np.array_equal(ex['prob'], p), f'seed {s}: test export differs from held probabilities'
        probs[s] = p
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--append', required=True)
    ap.add_argument('--list', required=True)
    a = ap.parse_args()
    t0 = time.time()
    r26 = json.loads(R26_SCORES.read_text())['cuts']['k40']
    seeds = [int(s) for s in r26['seeds_in_mean']]
    assert r26['collapsed_count'] == 0 and seeds == [0, 1, 2, 3, 4], r26['seeds_in_mean']
    ref = {}
    for a_, k in CLASS_COUNT.items():
        ref[(a_, k)] = ('cuts.k40.table.rclass.' + a_, r26['table']['rclass'][a_])
        ref.setdefault((a_, 20), ('cuts.k40.table.r20.' + a_, r26['table']['r20'][a_]))
    assert len(ref) == 6
    names = ns_data.keys(SET, HELD)
    y = {a_: ns_data.labels(SET, HELD, a_) for a_ in CLASS_COUNT}
    probs = load_probs(seeds, names)
    parts = {}                                   # (seed, k) -> int32 [3, n]
    cells = []
    for asp, ks in GRID.items():
        for k in ks:
            t1 = time.time()
            per_seed = {}
            for s in seeds:
                if (s, k) not in parts:
                    parts[(s, k)] = np.asarray(ns_metrics.percept_partitions(probs[s], k, SET))
                per_seed[s] = [masked_ami(y[asp], p) for p in parts[(s, k)]]
            seed_means = [float(np.mean(v)) for v in per_seed.values()]
            c = dict(id=f'{ROW}/{asp}/ami/{k}', row=ROW, aspect=asp, k=k, mean=float(np.mean(seed_means)),
                     per_seed_mean=seed_means, per_seed_state={str(s): v for s, v in per_seed.items()},
                     n_labelled=int((y[asp] >= 0).sum()),
                     source_inputs=f'{R26_OUT}/held/L1__k40__s{{{",".join(map(str, seeds))}}}.npz',
                     n_clusters_used=[[int(len(np.unique(p))) for p in parts[(s, k)]] for s in seeds],
                     seconds=time.time() - t1)
            r = ref.get((asp, k))
            c['gate'] = r is not None
            if r:
                c['reference_id'], c['reference'], c['diff'] = r[0], r[1], c['mean'] - r[1]
            cells.append(c)
            print(f"{c['id']:30s} mean={c['mean']:.10f} " + (f"diff={c['diff']:+.2e} " if r else '')
                  + f"{c['seconds']:.1f}s", flush=True)
    gated = [c for c in cells if c['gate']]
    m = max(abs(c['diff']) for c in gated)
    ok = len(gated) == 6 and m <= TOL
    print(f'GATE: {len(gated)} r26 cells, max |diff| = {m:.3e}; {"PASS" if ok else "FAIL"}; {time.time() - t0:.0f}s')
    if not ok:
        sys.exit(1)
    path = Path(a.append)
    doc = json.loads(path.read_text())
    before = json.dumps(doc['cells'], sort_keys=True)
    assert not {c['id'] for c in cells} & {c['id'] for c in doc['cells']}, 'cells already present'
    doc['cells'] = doc['cells'] + cells
    assert json.dumps(doc['cells'][:-len(cells)], sort_keys=True) == before
    doc['header_r26_row'] = dict(
        row=ROW, seeds=seeds, total_seconds=time.time() - t0, code_commit=KG.git_head(), gate_max_abs_diff=m,
        reference_file=str(R26_SCORES),
        protocol=('ns_metrics.percept_partitions (L2-normalised probabilities, MiniBatchKMeans n_init 3, batch 4096, '
                  'states 0,1,2) per seed per K; masked ns_metrics.ami (y >= 0); mean over states, then over seeds'))
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(doc, indent=1) + '\n')
    tmp.replace(path)
    Path(a.list).write_text(json.dumps([dict(aspect=c['aspect'], k=c['k'], mean=c['mean']) for c in cells], indent=1)
                            + '\n')


if __name__ == '__main__':
    main()
