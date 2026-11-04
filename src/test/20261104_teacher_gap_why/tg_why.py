"""Why do ARI and ACC miss the teacher where AMI and NMI win? A diagnostic: nothing here decides or changes a claim.

Reads r17's frozen scorer (imports only, no edit) and the saved embeddings. For c*, the Qwen teacher and the plain image
at the class count, it writes one JSON per cell with:
  - a replay of the four metrics against r17's verdicts.json (the check that this is the same measurement);
  - per-class parts of ARI, ACC and mutual information; the parts of one partition sum to its ARI, ACC and MI;
  - cluster balance and how much each class stays in one cluster;
  - ARI, ACC and NMI on a class-balanced table (every class weighs the same);
  - emotion only: ARI and ACC with one class masked after clustering, with r17's paired bootstrap.

Run through the queue: res-queue cpu --label r20:teacher-gap --est 20m -- python tg_why.py
Outputs: outputs/<set>__<split>__<enc>__<aspect>.json next to this file.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
R17 = HERE.parent / '20261103_new_scores'
sys.path.insert(0, str(R17))
import ns_paths as P  # noqa: E402

P.add_rounds()
import ns_clu  # noqa: E402
import ns_clu_cells  # noqa: E402
import ns_data  # noqa: E402
import ns_metrics  # noqa: E402

EMOTION = ('amusement', 'anger', 'awe', 'contentment', 'disgust', 'excitement', 'fear', 'sadness', 'something else')
OUT = HERE / 'outputs'
DRAWS = 2000


def tables(parts, y, k):
    """[m, classes, k] from partitions of any leading shape; unlabelled rows drop out in `contingency`."""
    flat = np.asarray(parts).reshape(-1, np.asarray(parts).shape[-1])
    ky = int(y.max()) + 1
    return np.stack([ns_metrics.contingency(y, p, shape=(ky, k)) for p in flat])


def pairs(x):
    x = np.asarray(x, dtype=np.float64)
    return x * (x - 1) / 2


def parts_of(T):
    """Per-class parts of one table. Sums: ari_i to ARI, acc_i to ACC, mi_i to the mutual information (nats)."""
    T = np.asarray(T, dtype=np.float64)
    n, a, b = T.sum(), T.sum(1), T.sum(0)
    sa, sb, total = pairs(a).sum(), pairs(b).sum(), pairs(n)
    denom = 0.5 * (sa + sb) - sa * sb / total
    ari_i = (pairs(T).sum(1) - pairs(a) * sb / total) / denom
    r, c = linear_sum_assignment(T, maximize=True)
    acc_i = np.zeros(len(a))
    acc_i[r] = T[r, c] / n
    p, pi, pj = T / n, a / n, b / n
    with np.errstate(divide='ignore', invalid='ignore'):
        cell = p * np.log(p / np.outer(pi, pj))
    mi_i = np.nan_to_num(cell).sum(1)
    q = b[b > 0] / n
    safe = np.where(a > 0, a, 1)
    return dict(ari_i=ari_i, acc_i=acc_i, mi_i=mi_i, top_share=T.max(1) / safe, recall=acc_i * n / safe,
                entropy=float(-(q * np.log(q)).sum() / np.log(T.shape[1])), largest=float(b.max() / n))


def arm_summary(tabs):
    """Mean over partitions of each part, with the whole-metric values and the spread across partitions."""
    each = [parts_of(t) for t in tabs]
    out = {key: np.mean([e[key] for e in each], axis=0).tolist() for key in each[0]}
    for metric in ns_metrics.METRICS:
        vals = [ns_metrics.from_contingency(t, metric) for t in tabs]
        out[metric] = float(np.mean(vals))
    return out


def balanced(tabs, scale=10000):
    """Class-balanced metrics: each class row is rescaled to the same mass (integers, as `from_contingency` needs)."""
    vals = {m: [] for m in ('ari', 'acc', 'nmi')}
    for t in tabs:
        a = t.sum(1, keepdims=True)
        row = np.rint(t[a[:, 0] > 0] * (scale / a[a[:, 0] > 0])).astype(np.int64)
        for m in vals:
            vals[m].append(ns_metrics.from_contingency(row, m))
    return {m: dict(mean=float(np.mean(v)), lo=float(np.min(v)), hi=float(np.max(v))) for m, v in vals.items()}


def recorded(cells, set_, enc, aspect, k):
    found = {}
    for c in cells:
        if (c['sc'], c['set'], c['encoder'], c['aspect'], c['k'], c['comparator']) == ('SC3', set_, enc, aspect, k, 'teacher'):
            found[c['metric']] = (c['cand'], c['comp'])
    return found


def run_cell(set_, split, enc, aspect, cells):
    k = P.CLASS_COUNT[(set_, aspect)]
    y = np.asarray(ns_data.labels(set_, split, aspect))
    cand = ns_clu_cells._cand_parts(set_, split, enc, aspect, k)
    teach = ns_clu_cells._comp_parts(set_, split, enc, aspect, 'teacher', k)
    plain = ns_clu_cells._comp_parts(set_, split, enc, aspect, 'plain', k)
    t_c, t_t, t_p = tables(cand, y, k), tables(teach, y, k), tables(plain, y, k)
    doc = dict(set=set_, split=split, encoder=enc, aspect=aspect, k=k,
               class_size=t_t[0].sum(1).tolist(),
               cstar=arm_summary(t_c), teacher=arm_summary(t_t), plain=arm_summary(t_p),
               balanced=dict(cstar=balanced(t_c), teacher=balanced(t_t), plain=balanced(t_p)))
    if split == P.HELD_SPLIT[set_]:
        rec = recorded(cells, set_, enc, aspect, k)
        doc['replay_max_diff'] = max(max(abs(doc['cstar'][m] - v[0]), abs(doc['teacher'][m] - v[1])) for m, v in rec.items())
    if aspect == 'emotion':
        doc['masked'] = {}
        for name in ('contentment', 'sadness', 'amusement', 'awe'):
            yy = y.copy()
            yy[y == EMOTION.index(name)] = -1
            res = ns_clu.paired_cell(cand, teach, yy, k, r=None, metrics=('ari', 'acc'), draws=DRAWS, rng_seed=0)
            doc['masked'][name] = {m: {f: res[m][f] for f in ('cand', 'comp', 'd', 'lo', 'hi')} for m in ('ari', 'acc')}
    return doc


def main():
    OUT.mkdir(exist_ok=True)
    cells = json.loads((R17 / 'records/verdicts.json').read_text())['cells']
    todo = [(s, P.HELD_SPLIT[s], e, a) for a in ('emotion', 'genre') for s in P.PAINTING_SETS for e in P.ENCODERS]
    todo += [(s, 'val', e, 'emotion') for s in P.PAINTING_SETS for e in P.ENCODERS]
    for set_, split, enc, aspect in todo:
        start = time.time()
        doc = run_cell(set_, split, enc, aspect, cells)
        (OUT / f'{set_}__{split}__{enc}__{aspect}.json').write_text(json.dumps(doc, indent=1))
        print(set_, split, enc, aspect, f'{time.time() - start:.0f}s', 'replay', doc.get('replay_max_diff'), flush=True)
    print('tg_why done', flush=True)


if __name__ == '__main__':
    main()
