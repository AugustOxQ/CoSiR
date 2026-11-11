"""Cluster-count grid for two more Table 2 rows: VLM2Vec-V2 and BERTopic (r25's comparators), ArtELingo held.

Plumbing only: r25's loaders (`pb_score.load_vlm2vec`, `pb_score.load_bertopic`) and r17's `ns_metrics` are imported
unchanged. Same grid, same cell format as `kr_grid.py`. Mean AMI over the three states (VLM2Vec: k-means states 0 to 2;
BERTopic: UMAP seeds 0 to 2, one fit per K, scored against every aspect whose grid contains K).

    python kr_grid_r25.py --gate --out records/gate_r25.json      # 12 r25 cells + K=10 refit comparison
    python kr_grid_r25.py --grid --append records/cells.json      # full grid, appended after the gate passes
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
R25 = HERE.parent / '20261111_paper_baselines'
sys.path.insert(0, str(R25))
sys.path.insert(0, str(HERE))
import kr_grid as KG            # noqa: E402  (r17/r18 sys.path set-up, GRID, CLASS_COUNT)
import pb_paths as P            # noqa: E402
import pb_score as PS           # noqa: E402
import ns_data                  # noqa: E402
import ns_metrics               # noqa: E402

ROWS = ('vlm2vec', 'bertopic')
GRID, CLASS_COUNT = KG.GRID, KG.CLASS_COUNT
SET, HELD = KG.SET, KG.HELD
R25_CELLS = R25 / 'records' / 'cells.json'
R25_BT = P.DEFAULT_OUT / 'bertopic'
MINE = Path('/data/SSD2/cosir_paper/20261111_cluster_count_range/outputs')
MY_BT = MINE / 'bertopic'
REFIT = MINE / 'bertopic_refit_gate'
R25_KS = (9, 10, 20, 27)        # BERTopic fits that exist in r25's folder
TOL = 1e-6


def bt_root(k):
    return R25_BT if k in R25_KS else MY_BT


def references():
    """{(row, aspect, k): (id, comp)} of r25's 12 recorded cells."""
    ref = {}
    for c in json.loads(R25_CELLS.read_text()):
        row, a, _, k = c['id'].split('/')
        ref[(row, a, int(k))] = (c['id'], c['comp'])
    assert len(ref) == 12
    return ref


class Grid:
    def __init__(self):
        self.names = ns_data.keys(SET, HELD)
        self.y = {a: ns_data.labels(SET, HELD, a) for a in CLASS_COUNT}
        self.rows_cache, self.bt_cache = {}, {}

    def vlm_rows(self, a):
        if a not in self.rows_cache:
            self.rows_cache.clear()
            self.rows_cache[a] = PS.load_vlm2vec(P.VLM2VEC_STORE, a, self.names)
        return self.rows_cache[a]

    def bt(self, k):
        if k not in self.bt_cache:
            self.bt_cache[k] = PS.load_bertopic(bt_root(k), self.names, k)
        return self.bt_cache[k]

    def inputs(self, row, a, k):
        if row == 'vlm2vec':
            return f'{P.VLM2VEC_STORE}/{a}/emb.npy (+keys.json, meta.json)'
        return f'{bt_root(k)}/' + ('topics_seed{0,1,2}.npz' if k == 20 else f'topics_k{k}_seed{{0,1,2}}.npz')

    def cell(self, row, a, k):
        t0 = time.time()
        if row == 'vlm2vec':
            p = np.asarray(ns_metrics.partitions(self.vlm_rows(a)[0], k, SET))
        else:
            p = self.bt(k)[0]
        y = self.y[a]
        per = [ns_metrics.score_partitions('ami', y, p[i:i + 1]) for i in range(p.shape[0])]
        mean = ns_metrics.score_partitions('ami', y, p)
        return dict(id=f'{row}/{a}/ami/{k}', row=row, aspect=a, k=k, mean=mean, per_state=per,
                    n_labelled=int((y >= 0).sum()), source_inputs=self.inputs(row, a, k),
                    n_clusters_used=[int(len(np.unique(s))) for s in p], seconds=time.time() - t0)


def gate(g):
    ref = references()
    cells = []
    for (row, a, k), (rid, val) in sorted(ref.items()):
        c = g.cell(row, a, k)
        c.update(gate=True, reference_id=rid, reference=val, diff=c['mean'] - val)
        cells.append(c)
        print(f"{c['id']:26s} mean={c['mean']:.12f} ref={val:.12f} diff={c['diff']:+.2e} {c['seconds']:.1f}s", flush=True)
    cells_max = max(abs(c['diff']) for c in cells)
    # refit: K = 10, three seeds, new fit path vs r25's saved partitions
    from sklearn.metrics import adjusted_mutual_info_score as ami
    refit = []
    for s in P.UMAP_SEEDS:
        import pb_partitions
        new = np.load(pb_partitions.topic_path(REFIT, s, 10), allow_pickle=False)
        old = np.load(pb_partitions.topic_path(R25_BT, s, 10), allow_pickle=False)
        same_names = new['names'].tolist() == old['names'].tolist() == list(g.names)
        ident = bool(np.array_equal(new['topics'], old['topics']))
        nm = json.loads(pb_partitions.topic_path(REFIT, s, 10).with_suffix('.json').read_text())
        om = json.loads(pb_partitions.topic_path(R25_BT, s, 10).with_suffix('.json').read_text())
        refit.append(dict(seed=s, names_identical=same_names, labels_identical=ident,
                          n_differing=int((new['topics'] != old['topics']).sum()),
                          ami_between=float(ami(old['topics'], new['topics'])),
                          hdbscan_topics=(nm['topics_found_hdbscan'], om['topics_found_hdbscan']),
                          outliers_before=(nm['outlier_counts']['after_hdbscan'], om['outlier_counts']['after_hdbscan']),
                          versions_equal=nm['versions'] == om['versions'],
                          bertopic_source_sha_equal=nm['bertopic_source_sha256'] == om['bertopic_source_sha256'],
                          umap_args_equal=nm['umap_arguments'] == om['umap_arguments'],
                          hdbscan_args_equal=nm['hdbscan_arguments'] == om['hdbscan_arguments'],
                          refit_seconds=nm['seconds'], npz_sha256_new=nm['npz_sha256'], npz_sha256_r25=om['npz_sha256']))
        print('refit', refit[-1], flush=True)
    refit_ok = all(r['labels_identical'] for r in refit)
    ok = cells_max <= TOL and refit_ok
    print(f'GATE: 12 cells max |diff| = {cells_max:.3e}; refit identical = {refit_ok}; {"PASS" if ok else "FAIL"}')
    return dict(passed=bool(ok), tolerance=TOL, max_abs_diff=cells_max, cells=cells, refit_k10=refit,
                refit_dir=str(REFIT), r25_dir=str(R25_BT))


def grid(g):
    ref = references()
    out = []
    for row in ROWS:
        for a, ks in GRID.items():
            for k in ks:
                c = g.cell(row, a, k)
                r = ref.get((row, a, k))
                c['gate'] = r is not None
                if r:
                    c['reference_id'], c['reference'], c['diff'] = r[0], r[1], c['mean'] - r[1]
                out.append(c)
                print(f"{c['id']:26s} mean={c['mean']:.10f} " + (f"diff={c['diff']:+.2e} " if r else '')
                      + f"used={c['n_clusters_used']} {c['seconds']:.1f}s", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gate', action='store_true')
    ap.add_argument('--grid', action='store_true')
    ap.add_argument('--out')
    ap.add_argument('--append')
    a = ap.parse_args()
    g, t0 = Grid(), time.time()
    if a.gate:
        res = gate(g)
        res['total_seconds'] = time.time() - t0
        res['code_commit'] = KG.git_head()
        Path(a.out).write_text(json.dumps(res, indent=1) + '\n')
    if a.grid:
        cells = grid(g)
        gated = [c for c in cells if c['gate']]
        m = max(abs(c['diff']) for c in gated)
        print(f'GRID: {len(cells)} cells, {len(gated)} gate cells, max |diff| = {m:.3e}, {time.time() - t0:.0f}s')
        assert len(gated) == 12 and m <= TOL
        path = Path(a.append)
        doc = json.loads(path.read_text())
        before = json.dumps(doc['cells'], sort_keys=True)
        assert not {c['id'] for c in cells} & {c['id'] for c in doc['cells']}, 'cells already present'
        doc['cells'] = doc['cells'] + cells
        assert json.dumps(doc['cells'][:-len(cells)], sort_keys=True) == before
        doc['header_r25_rows'] = dict(
            rows=list(ROWS), total_seconds=time.time() - t0, code_commit=KG.git_head(),
            protocol=('VLM2Vec: pb_score.load_vlm2vec + ns_metrics.partitions (MiniBatchKMeans n_init 3, batch 4096, states 0,1,2) per aspect '
                      'per K; BERTopic: one fresh default fit per (UMAP seed 0,1,2, K) by pb_bertopic.run_seed (nr_topics K+1 with outliers, '
                      'outliers reassigned), scored against every aspect whose grid has K; mean AMI over the 3 states/seeds, rows with y >= 0'),
            bertopic_dirs=dict(r25=str(R25_BT), new=str(MY_BT)))
        tmp = path.with_name(path.name + '.tmp')
        tmp.write_text(json.dumps(doc, indent=1) + '\n')
        tmp.replace(path)


if __name__ == '__main__':
    main()
