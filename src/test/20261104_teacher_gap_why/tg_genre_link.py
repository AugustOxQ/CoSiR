"""Does the teacher reach contentment through genre? A diagnostic: nothing here decides or changes a claim.

Held rows of the painting sets, emotion partitions at the class count, rows with a genre label only. For c* on each
encoder and the Qwen teacher it writes:
  - emotion ARI and ACC on these rows, and how much genre each emotion partition carries (AMI against genre);
  - the emotion ARI within each genre (the row-weighted mean takes genre out of the comparison);
  - contentment's matched cluster: precision, recall per genre, genre mix;
  - the reference: genre labels used directly as the partition, and P(contentment | genre).

Run through the queue: res-queue cpu --short --label r20:teacher-gap -- python tg_genre_link.py
Output: outputs/genre_link.json next to this file.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
R17 = HERE.parent / '20261103_new_scores'
sys.path.insert(0, str(R17))
import ns_paths as P  # noqa: E402

P.add_rounds()
import ns_clu_cells  # noqa: E402
import ns_data  # noqa: E402
import ns_metrics  # noqa: E402

CONTENTMENT = 3
GENRES = ('abstract', 'cityscape', 'genre painting', 'illustration', 'landscape', 'nude', 'portrait', 'religious',
          'sketch', 'still life')


def arm_stats(parts, y_e, y_g, k):
    keep = y_g >= 0
    flat = parts.reshape(-1, parts.shape[-1])
    ari, acc, ami_g, within, prec, size, rec_g, mix = [], [], [], [], [], [], [], []
    for p in flat:
        table = ns_metrics.contingency(np.where(keep, y_e, -1), p, shape=(9, k))
        ari.append(ns_metrics.from_contingency(table, 'ari'))
        acc.append(ns_metrics.from_contingency(table, 'acc'))
        ami_g.append(ns_metrics.from_contingency(ns_metrics.contingency(y_g, p, shape=(10, k)), 'ami'))
        vals, weights = [], []
        for g in range(10):
            sel = (y_g == g) & (y_e >= 0)
            sub = ns_metrics.contingency(y_e[sel], p[sel], shape=(9, k))
            vals.append(ns_metrics.from_contingency(sub, 'ari'))
            weights.append(sel.sum())
        within.append(float(np.average(vals, weights=weights)))
        r, c = linear_sum_assignment(table, maximize=True)
        j = c[list(r).index(CONTENTMENT)]
        in_cluster = keep & (y_e >= 0) & (p == j)
        size.append(float(in_cluster.sum() / (keep & (y_e >= 0)).sum()))
        prec.append(float((y_e[in_cluster] == CONTENTMENT).mean()))
        mix.append(np.bincount(y_g[in_cluster], minlength=10) / in_cluster.sum())
        rec_g.append([float((p[(y_e == CONTENTMENT) & (y_g == g)] == j).mean()) if ((y_e == CONTENTMENT) & (y_g == g)).any()
                      else 0.0 for g in range(10)])
    return dict(ari=float(np.mean(ari)), acc=float(np.mean(acc)), ami_genre=float(np.mean(ami_g)),
                within_genre_ari=float(np.mean(within)), precision=float(np.mean(prec)), size=float(np.mean(size)),
                recall_by_genre=np.mean(rec_g, axis=0).tolist(), genre_mix=np.mean(mix, axis=0).tolist())


def main():
    out = {}
    for set_ in P.PAINTING_SETS:
        split = P.HELD_SPLIT[set_]
        k = P.CLASS_COUNT[(set_, 'emotion')]
        y_e = np.asarray(ns_data.labels(set_, split, 'emotion'))
        y_g = np.asarray(ns_data.labels(set_, split, 'genre'))
        both = (y_e >= 0) & (y_g >= 0)
        direct = ns_metrics.contingency(y_e[both], y_g[both], shape=(9, 10))
        doc = dict(n_rows=int(both.sum()), genre_as_partition=dict(ari=ns_metrics.from_contingency(direct, 'ari'),
                                                                   acc=ns_metrics.from_contingency(direct, 'acc')),
                   contentment_given_genre=(direct[CONTENTMENT] / direct.sum(0)).tolist(),
                   contentment_rows_by_genre=direct[CONTENTMENT].tolist(), genre_size=direct.sum(0).tolist(),
                   genres=GENRES)
        teach = ns_clu_cells._comp_parts(set_, split, 'qwen', 'emotion', 'teacher', k)
        doc['teacher'] = arm_stats(teach, y_e, y_g, k)
        for enc in P.ENCODERS:
            cand = ns_clu_cells._cand_parts(set_, split, enc, 'emotion', k)
            doc[enc] = arm_stats(cand, y_e, y_g, k)
            print(set_, enc, 'done', flush=True)
        out[set_] = doc
    (HERE / 'outputs/genre_link.json').write_text(json.dumps(out, indent=1))
    print('tg_genre_link done', flush=True)


if __name__ == '__main__':
    main()
