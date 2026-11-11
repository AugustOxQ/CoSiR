"""Cluster-count grid for four Table 2 rows (TURTLE on plain Qwen3, TPIPS, CLAY, CRL) on ArtELingo held.

Plumbing only: every loader and scorer is the project's own, imported unchanged (r17 `ns_*`, r18 `om_*`).
Mean AMI over the three k-means states (TURTLE: the three starts), per-state values too. Clustering runs on all
12,281 held rows; the mask y >= 0 is applied by `ns_metrics.score_partitions`.

    python kr_grid.py --only-existing --out gate.json      # the 24 cells with a recorded value (gate)
    python kr_grid.py --rows turtle,tpips --out cells.json # full grid
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROUNDS = Path(__file__).resolve().parents[1]
for sub in ('20261103_outside_methods', '20261103_new_scores'):
    if str(ROUNDS / sub) not in sys.path:
        sys.path.insert(0, str(ROUNDS / sub))

import om_data  # noqa: E402
import om_paths as op  # noqa: E402
import ns_data  # noqa: E402
import ns_metrics  # noqa: E402
import om_cells  # noqa: E402  (check_pin)

SET, ENC = 'artelingo', 'qwen'
HELD = op.HELD_SPLIT[SET]
ROWS = ('turtle', 'tpips', 'clay', 'crl')
GRID = {'emotion': (5, 8, 9, 16, 20), 'style': (10, 20, 27, 40), 'genre': (5, 10, 16, 20)}
CLASS_COUNT = {'emotion': 9, 'style': 27, 'genre': 10}
LAB = ROUNDS / '20261103_outside_methods/records/labels.json'
VER = ROUNDS / '20261103_new_scores/records/verdicts.json'
PICKS = op.OUT / 'stage1' / 'om_picks.json'
LAB_ID = {'turtle': 'SC5/artelingo/qwen/{a}/ami/{k}/plain_turtle', 'tpips': 'SC1/artelingo/qwen/{a}/ami/{k}/tpips',
          'clay': 'SC3/artelingo/qwen/{a}/ami/{k}/clay'}


def references():
    """{(row, aspect, k): (id, value)} of the 24 recorded cells."""
    lab = {c['id']: c['comp'] for c in json.loads(LAB.read_text())['cells']}
    ver = {c['id']: c['comp'] for c in json.loads(VER.read_text())['cells']}
    ref = {}
    for a, kc in CLASS_COUNT.items():
        for k in (kc, 20):
            for row in ('turtle', 'tpips', 'clay'):
                i = LAB_ID[row].format(a=a, k=k)
                ref[(row, a, k)] = (i, lab[i])
            i = (f'SC5/artelingo/clip_l14/{a}/ami/{k}/crl' if k == kc else f'SC5/artelingo/clip_l14/{a}/ami@20/20/crl')
            ref[('crl', a, k)] = (i, ver[i])
    assert len(ref) == 24
    return ref


def _no_write(*a, **k):
    raise RuntimeError('kr_grid: a loader tried to write a file; refusing')


def git_head():
    return subprocess.run(['git', '-C', str(ROUNDS), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()


class Grid:
    def __init__(self):
        om_data._atomic = _no_write                   # CLAY's loader would write a cache npz on a sidecar mismatch
        self.picks = json.loads(PICKS.read_text())
        self.text_set = om_cells._clay_text_set(self.picks, 'clu', SET)
        self.y = {a: ns_data.labels(SET, HELD, a) for a in CLASS_COUNT}
        self.cache = {}

    def inputs(self, row, a):
        return {'turtle': f'{op.OUT}/parts/turtle/{SET}/{ENC}/plain/k{{K}}__start{{0,1,2}}.npz',
                'tpips': f'/data/SSD2/pre_extract/artelingo_full/tpips_8b/{op.TPIPS_FACTORS[a]}/emb.npy (held rows by keys.json)',
                'clay': f'{op.OUT}/clay/{SET}_{HELD}/{self.text_set}/{a}.npz (text set {self.text_set})',
                'crl': f'clip_l14 img store held rows + /data/SSD2/cosir_r11/20261029_retrieval/outputs/crl/{a}_text.npy (in memory)'}[row]

    def rows_for(self, row, a):
        key = (row, a)
        if row == 'turtle':
            return None
        if key not in self.cache:
            self.cache.clear()
            if row == 'tpips':
                x = om_data.tpips(SET, HELD, a)
            elif row == 'clay':
                om_cells.check_pin(self.picks, self.text_set, a)
                x = om_data.clay(SET, HELD, a, self.text_set)
            else:
                x = ns_data.crl(SET, HELD, a)
            self.cache[key] = x
        return self.cache[key]

    def parts(self, row, a, k):
        if row == 'turtle':
            return om_data.turtle_parts(SET, ENC, 'plain', k)          # int32 [3, n], starts 0 to 2
        return ns_metrics.partitions(self.rows_for(row, a), k, SET)

    def cell(self, row, a, k):
        t0 = time.time()
        p = np.asarray(self.parts(row, a, k))
        y = self.y[a]
        per = [ns_metrics.score_partitions('ami', y, p[i:i + 1]) for i in range(p.shape[0])]
        mean = ns_metrics.score_partitions('ami', y, p)
        return dict(id=f'{row}/{a}/ami/{k}', row=row, aspect=a, k=k, mean=mean, per_state=per,
                    n_labelled=int((y >= 0).sum()), source_inputs=self.inputs(row, a),
                    n_clusters_used=[int(len(np.unique(s))) for s in p], seconds=time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rows', default=','.join(ROWS))
    ap.add_argument('--only-existing', action='store_true')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    rows = a.rows.split(',')
    assert set(rows) <= set(ROWS), rows
    ref = references()
    g = Grid()
    out, t0 = [], time.time()
    for row in rows:
        for asp, ks in GRID.items():
            for k in ks:
                if a.only_existing and (row, asp, k) not in ref:
                    continue
                c = g.cell(row, asp, k)
                r = ref.get((row, asp, k))
                c['gate'] = r is not None
                if r:
                    c['reference_id'], c['reference'], c['diff'] = r[0], r[1], c['mean'] - r[1]
                out.append(c)
                print(f"{c['id']:28s} mean={c['mean']:.10f} " + (f"ref={r[1]:.10f} diff={c['diff']:+.2e} " if r else '')
                      + f"used={c['n_clusters_used']} {c['seconds']:.1f}s", flush=True)
    hdr = dict(protocol=('ns_metrics.partitions: MiniBatchKMeans n_init 3, batch 4096, random states 0,1,2, float32, thread limit '
                         'KMEANS_THREADS[artelingo]; all 12,281 held rows clustered, AMI on rows with y >= 0; mean over the 3 states '
                         '(TURTLE: saved fits, starts 0,1,2, no refit)'),
               code_commit=git_head(), dirty_note='git HEAD of the checkout; scored code files are unchanged tracked files',
               clay_text_set=g.text_set, held_split=HELD, total_seconds=time.time() - t0, only_existing=a.only_existing)
    Path(a.out).write_text(json.dumps(dict(header=hdr, cells=out), indent=1) + '\n')
    gated = [c for c in out if c['gate']]
    if gated:
        m = max(abs(c['diff']) for c in gated)
        print(f'GATE: {len(gated)} cells, max |diff| = {m:.3e}, {"PASS" if m <= 1e-6 and len(gated) == 24 else "see table"}')


if __name__ == '__main__':
    main()
