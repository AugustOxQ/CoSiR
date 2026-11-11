"""r25 claim-review check: recompute comparator AMI at the class count and 20 outside r17's scorer.

Independent path: sklearn's own `adjusted_mutual_info_score` on the labelled rows (labels from r17's `ns_data.labels`,
the same labels the gate checked); VLM2Vec partitions refitted here with sklearn MiniBatchKMeans at r10's settings
(n_init 3, batch 4,096, states 0 to 2), BERTopic partitions read from its npz files. Compares each mean with
OUT/score/cells.json `comp`. With `--stale`, also scores the first BERTopic run (fallback MiniLM) for the B1 size check.

    python pb_check.py [--stale]   -> OUT/score/check.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_mutual_info_score
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pb_paths as P  # noqa: E402
from pb_partitions import CLASS_COUNTS, TARGET_KS, aspect_ks, topic_path
from pb_io import write_json


def ami_mean(y, parts):
    m = y >= 0
    return float(np.mean([adjusted_mutual_info_score(y[m], p[m]) for p in parts]))


def bertopic_parts(folder, names, k=P.K):
    out = []
    for s in P.UMAP_SEEDS:
        path = topic_path(folder, s, k)
        with np.load(path, allow_pickle=False) as z:
            assert list(z['names']) == names, f'{path}: names differ'
            topics = z['topics']
            assert topics.shape == (len(names),) and np.issubdtype(topics.dtype, np.integer), path
            assert np.array_equal(np.unique(topics), np.arange(k)), f'{path}: expected {k} real topics'
            out.append(topics)
    return out


def vlm2vec_parts(rows, k):
    """Independent sklearn fit; deliberately does not call r17's partition helper."""
    with threadpool_limits(limits=8):
        return [MiniBatchKMeans(n_clusters=k, random_state=s, n_init=3, batch_size=4096)
                .fit(rows.astype(np.float32)).labels_ for s in (0, 1, 2)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--stale', action='store_true')
    a = ap.parse_args(argv)
    ns_paths = P.add_scorer_paths()
    import ns_data
    names = ns_data.keys(P.SET, P.SPLIT)
    record = json.loads((P.OUT / 'score' / 'cells.json').read_text())
    cells = {c['id']: c for c in (record['cells'] if isinstance(record, dict) else record)}
    rows, bt = [], {k: bertopic_parts(P.BERTOPIC_OUT, names, k) for k in TARGET_KS}
    stale = bertopic_parts(P.OUT / 'bertopic_stale_fallback_minilm', names) if a.stale else None
    for aspect in P.ASPECTS:
        y = ns_data.labels(P.SET, P.SPLIT, aspect)
        assert len(np.unique(y[y >= 0])) == ns_paths.CLASS_COUNT[(P.SET, aspect)] == CLASS_COUNTS[aspect], aspect
        x = np.load(P.VLM2VEC_STORE / aspect / 'emb.npy', allow_pickle=False)
        assert json.loads((P.VLM2VEC_STORE / aspect / 'keys.json').read_text()) == names
        for k in aspect_ks(aspect):
            vp = vlm2vec_parts(x, k)
            for comp, parts in (('vlm2vec', vp), ('bertopic', bt[k])):
                cid = f'{comp}/{aspect}/ami/{k}'
                mine, theirs = ami_mean(y, parts), float(cells[cid]['comp'])
                rows.append(dict(id=cid, recomputed=mine, scorer=theirs, gap=abs(mine - theirs)))
        if stale is not None:
            rows.append(dict(id=f'bertopic_stale/{aspect}/ami/20', recomputed=ami_mean(y, stale), scorer=None, gap=None))
    out = P.OUT / 'score' / 'check.json'
    write_json(out, dict(rows=rows, max_gap=max(r['gap'] for r in rows if r['gap'] is not None)))
    for r in rows:
        print(r)


if __name__ == '__main__':
    main()
