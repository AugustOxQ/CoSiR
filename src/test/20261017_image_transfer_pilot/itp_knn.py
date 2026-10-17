"""Exploration (controller, after the pilot failed its bar): model-free neighbour transfer.

For each evaluation painting, average the targets of its k nearest training paintings in an image space
(cosine), then score the averaged vectors like the students. No labels enter the prediction.
"""
import json
import time

import numpy as np
from threadpoolctl import threadpool_limits

from itp_data import HERE, default_images, emotion_images, l2, load_frozen, split, targets
from apc_data import paint_first, paint_majority
from pav_score import ami, partitions, probe, summary


def neighbour_mean(query, pool, y, k):
    out = np.empty((len(query), y.shape[1]), dtype=np.float32)
    for a in range(0, len(query), 500):
        sim = query[a:a + 500] @ pool.T
        ix = np.argpartition(-sim, k - 1, axis=1)[:, :k]
        out[a:a + 500] = y[ix].mean(1)
    return l2(out)


def main():
    start = time.monotonic()
    s = load_frozen()
    subset, count = s['subset_pid'], len(s['first'])
    train = split(s)[0]
    labels = dict(emotion=paint_majority(s['emo'], s['pid'], count, 9)[subset],
                  genre=paint_first(s['gen'], s['pid'], count)[subset],
                  style=paint_first(s['sty'], s['pid'], count)[subset])
    inputs = dict(default=default_images(s)[0], emotion_focus=emotion_images(s))
    ys = dict(emotion_focus=targets(s, 'emotion_focus'), default=targets(s, 'default'))
    out = {}
    for inp, x in inputs.items():
        for tgt, y in ys.items():
            for k in (20, 100):
                name = f'{inp}__{tgt}__k{k}'
                pred = neighbour_mean(x[subset], x[train], y[train], k)
                ls = partitions(pred, 20)
                out[name] = dict(painting_ami={lab: summary([ami(v[v >= 0], p[v >= 0]) for p in ls])
                                               for lab, v in labels.items()},
                                 emotion_probe=probe(pred, labels['emotion'], subset)['accuracy'])
                a = out[name]['painting_ami']
                print(f"{name}: emotion {a['emotion']['mean']:.4f} genre {a['genre']['mean']:.4f} "
                      f"style {a['style']['mean']:.4f} probe {out[name]['emotion_probe']['mean']:.4f}", flush=True)
    (HERE / 'itp_knn_results.json').write_text(json.dumps(out, indent=2))
    print(f'Finished in {time.monotonic() - start:.1f}s', flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=8):
        main()
