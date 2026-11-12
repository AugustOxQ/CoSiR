"""r28 job A: the encoder pick on the ArtELingo validation split (6,152 paintings), handoff section 4, tier 0, item 1.

Scores our model (c*, arm `k11=40`) per encoder from r14's saved validation exports (r10's for Qwen):
`<root>/train/k11=40/R6-5/<aspect>__<graph>__s<seed>.npz`, image side, seeds 0 to 4. Six cells per encoder:
  AMI at the class counts (emotion 9, style 27, genre 10): the r17 / t5 path, unchanged: stored float16 rows cast to
      float32 (`ns_data.cstar(unit=False)`), `ns_metrics.partitions` (MiniBatchKMeans, n_init 3, batch 4,096, states 0
      to 2, no further normalisation, 8 threads) and `ns_clu.metric_table` (mean over the 3 states per seed).
  plain-cosine label mAP: `rt_metrics.retrieval(unit rows, labels, qe_k=0)` mean AP, the function behind r14's
      `bs_artret.validation_ranges` (`ns_data.cstar(unit=True)` rows), labelled rows only.
Cell value = mean over the five seeds. Pick = encoder with the highest mean of its six cell means (Qwen is an
information row: never part of the pick).

Gates: the AMI per-seed values equal r17's `records/stage1_ranges.json` (key `cstar/artelingo/<enc>/<aspect>/ami`);
the mAP per-seed values equal r14's `stage1/artret_ranges_<enc>.json` (not for Qwen). Reported, never silently used.

    python r28_valpick.py run [--encoders siglip1 siglip2 clip_l14 qwen] [--out DIR] [--records DIR]
    python r28_valpick.py merge [--out DIR] [--records DIR]      (rebuild valpick.json from the per-encoder files)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TEST = HERE.parent
sys.path.insert(0, str(TEST / '20261103_new_scores'))

PICK_ENCODERS = ('siglip1', 'siglip2', 'clip_l14')
INFO_ENCODERS = ('qwen',)
ASPECTS = ('emotion', 'style', 'genre')
SEEDS = (0, 1, 2, 3, 4)
CLASS_COUNT = dict(emotion=9, style=27, genre=10)
CELLS = tuple(f'{m}/{a}' for m in ('ami', 'map') for a in ASPECTS)
OUT = Path('/data/SSD2/cosir_r28/20261112_siglip_main/valpick')
RECORDS = HERE / 'records'
R17_RANGES = TEST / '20261103_new_scores/records/stage1_ranges.json'
R14_STAGE1 = Path('/data/SSD2/cosir_r14/20261102_backbones_sweep/outputs/stage1')
TOL = 1e-9


# ------------------------------------------------------------------------------------------------ pure parts
def cell_stats(per_seed):
    """Seed mean and min-max of one cell."""
    v = np.asarray(per_seed, dtype=np.float64)
    assert v.shape == (len(SEEDS),) and np.isfinite(v).all(), v
    return dict(per_seed=[float(x) for x in v], mean=float(v.mean()), min=float(v.min()), max=float(v.max()))


def encoder_score(cells):
    """Mean of the six cell means; `cells` {'ami/emotion': {'mean': ...}, ...} with exactly CELLS."""
    assert tuple(sorted(cells)) == tuple(sorted(CELLS)), sorted(cells)
    return float(np.mean([cells[c]['mean'] for c in CELLS]))


def pick(table, candidates=PICK_ENCODERS):
    """Highest encoder score among `candidates` (every candidate must be present; information rows are ignored).
    Returns dict(pick, scores, ranking, margin, tie); a tie within 1e-12 for the top is flagged and `pick` is None."""
    missing = [e for e in candidates if e not in table]
    assert not missing, f'missing encoders: {missing}'
    scores = {e: encoder_score(table[e]) for e in candidates}
    ranking = sorted(candidates, key=lambda e: -scores[e])
    top = scores[ranking[0]]
    tied = [e for e in ranking if top - scores[e] <= 1e-12]
    margin = float(top - scores[ranking[1]]) if len(ranking) > 1 else None
    return dict(pick=ranking[0] if len(tied) == 1 else None, tie=tied if len(tied) > 1 else None, scores=scores,
                ranking=ranking, margin=margin)


def join_check(sidecar_dev_sha, names_sha, n_rows, n_names, n_labels, where):
    """Boundary assertions for one export: the sidecar's validation-name hash equals the hash of the label-order
    names, and the row counts agree (C6)."""
    assert sidecar_dev_sha == names_sha, f'{where}: export names hash differs from the label order'
    assert n_rows == n_names == n_labels, f'{where}: rows {n_rows}, names {n_names}, labels {n_labels}'


def gate_diff(got, expected):
    d = np.abs(np.asarray(got, dtype=np.float64) - np.asarray(expected, dtype=np.float64))
    return float(d.max())


# ------------------------------------------------------------------------------------------------ scoring
def _imports():
    import ns_paths
    ns_paths.add_rounds()
    import ns_clu
    import ns_data
    import ns_metrics
    import rs_vh
    import rt_data
    import rt_metrics
    return ns_clu, ns_data, ns_metrics, rs_vh, rt_data, rt_metrics


def sidecar_dev_sha(enc, aspect, seed, ns_data, rt_data):
    root = ns_data.CSTAR_ROOT[enc]
    path = root / f'train/k11=40/R6-5/{aspect}__{rt_data.GRAPHS[aspect]}__s{seed}.json'
    side = json.loads(path.read_text())
    assert side['config']['rg']['arm'] == 'k11=40' and side['config']['seed'] == seed, path
    return side['config']['names_sha256']['dev'], path


def score_encoder(enc, seeds=SEEDS, log=print):
    ns_clu, ns_data, ns_metrics, rs_vh, rt_data, rt_metrics = _imports()
    names = ns_data.keys('artelingo', 'val')
    assert names == rt_data.names('val') and len(names) == 6152
    names_sha = rs_vh.names_sha(names)
    cells, secs = {}, {}
    for a in ASPECTS:
        y = ns_data.labels('artelingo', 'val', a)
        assert np.array_equal(y, np.asarray(rt_data.labels('val')[a], dtype=np.int64))
        k = CLASS_COUNT[a]
        assert ns_metrics.class_count(y) == k, (a, ns_metrics.class_count(y), k)
        parts, aps = [], []
        for s in seeds:
            t = time.monotonic()
            dev, path = sidecar_dev_sha(enc, a, s, ns_data, rt_data)
            xc = ns_data.cstar('artelingo', 'val', enc, a, s, unit=False)     # sidecar + shape + unit-norm checks inside
            join_check(dev, names_sha, len(xc), len(names), len(y), str(path))
            parts.append(ns_metrics.partitions(xc, k, 'artelingo'))
            xu = ns_data.cstar('artelingo', 'val', enc, a, s, unit=True)
            ap = rt_metrics.retrieval(xu, y, qe_k=0, workers=8)['ap']
            assert np.isfinite(ap).all()
            aps.append(float(ap.mean()))
            log(f'  {enc} {a} seed {s}: {time.monotonic() - t:.1f}s  map {aps[-1]:.5f}')
        (row,) = ns_clu.metric_table(f'cstar-{enc}', a, np.stack(parts), y, k, metrics=('ami',))
        cells[f'ami/{a}'] = cell_stats(row['per_seed'])
        cells[f'map/{a}'] = cell_stats(aps)
        cells[f'ami/{a}']['k'] = k
        cells[f'ami/{a}']['n_labelled'] = int((y >= 0).sum())
        cells[f'map/{a}']['n_labelled'] = int((y >= 0).sum())
        log(f'{enc} {a}: AMI {cells[f"ami/{a}"]["mean"]:.5f}  mAP {cells[f"map/{a}"]["mean"]:.5f}')
    return dict(encoder=enc, cells=cells, score=encoder_score(cells), names_sha256=names_sha, n_val=len(names),
                gate=gates(enc, cells))


def gates(enc, cells):
    out = {}
    r17 = json.loads(R17_RANGES.read_text())
    diffs = {}
    for a in ASPECTS:
        rec = r17[f'cstar/artelingo/{enc}/{a}/ami']
        assert rec['k'] == CLASS_COUNT[a]
        diffs[a] = gate_diff(cells[f'ami/{a}']['per_seed'], [rec['per_seed'][str(s)] for s in SEEDS])
    out['ami_vs_r17_stage1_ranges'] = dict(source=str(R17_RANGES), max_abs_diff=diffs,
                                           **{'pass': bool(max(diffs.values()) <= TOL)})
    f = R14_STAGE1 / f'artret_ranges_{enc}.json'
    if f.exists():
        rec = json.loads(f.read_text())
        diffs = {a: gate_diff(cells[f'map/{a}']['per_seed'], [rec['per_seed'][a][str(s)] for s in SEEDS])
                 for a in ASPECTS}
        out['map_vs_r14_artret_ranges'] = dict(source=str(f), max_abs_diff=diffs,
                                               **{'pass': bool(max(diffs.values()) <= TOL)})
    else:
        out['map_vs_r14_artret_ranges'] = dict(source=None, note='no r14 record for this encoder')
    return out


# ------------------------------------------------------------------------------------------------ files
def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1, allow_nan=False) + '\n')
    tmp.replace(path)


def merge(out=OUT, records=RECORDS):
    table = {}
    for e in PICK_ENCODERS + INFO_ENCODERS:
        f = Path(out) / f'valpick_{e}.json'
        if f.exists():
            table[e] = json.loads(f.read_text())
    doc = dict(kind='r28_valpick', split='artelingo validation', n_val=6152, arm='k11=40 (c*)', seeds=list(SEEDS),
               cells=list(CELLS), pick_candidates=list(PICK_ENCODERS), information_rows=list(INFO_ENCODERS),
               rule='highest mean of the six cell means (AMI 9/27/10, plain-cosine mAP), seed means',
               kmeans='ns_metrics.partitions: MiniBatchKMeans n_init 3, batch 4096, states 0-2, 8 threads, stored '
                      'float16 rows cast to float32, no renormalisation; AMI mean over 3 states per seed (r17 / t5 path)',
               map='rt_metrics.retrieval(unit rows, labels, qe_k=0) mean AP, labelled rows (r14 validation_ranges)',
               encoders={e: v for e, v in table.items()})
    if all(e in table for e in PICK_ENCODERS):
        doc['pick'] = pick({e: v['cells'] for e, v in table.items()})
        doc['complete'] = True
    else:
        doc['complete'] = False
    write_json(Path(out) / 'valpick.json', doc)
    if doc['complete']:
        write_json(Path(records) / 'valpick.json', doc)
    return doc


def table_md(doc):
    encs = list(doc['encoders'])
    lines = ['| cell | ' + ' | '.join(encs) + ' |', '|---|' + '---:|' * len(encs)]
    for c in CELLS:
        row = []
        for e in encs:
            s = doc['encoders'][e]['cells'][c]
            row.append(f'{s["mean"]:.4f} ({s["min"]:.4f}-{s["max"]:.4f})')
        lines.append(f'| {c} | ' + ' | '.join(row) + ' |')
    lines.append('| mean of six | ' + ' | '.join(f'{doc["encoders"][e]["score"]:.4f}' for e in encs) + ' |')
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('cmd', choices=('run', 'merge'))
    p.add_argument('--encoders', nargs='+', default=list(PICK_ENCODERS + INFO_ENCODERS),
                   choices=PICK_ENCODERS + INFO_ENCODERS)
    p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--records', type=Path, default=RECORDS)
    a = p.parse_args()
    if a.cmd == 'run':
        for e in a.encoders:
            t = time.monotonic()
            res = score_encoder(e)
            res['seconds'] = time.monotonic() - t
            write_json(a.out / f'valpick_{e}.json', res)
            print(f'{e}: score {res["score"]:.5f} ({res["seconds"]:.0f}s) gates '
                  f'{ {k: v.get("pass") for k, v in res["gate"].items()} }', flush=True)
    doc = merge(a.out, a.records)
    print(table_md(doc))
    print('pick:', json.dumps(doc.get('pick')))


if __name__ == '__main__':
    main()
