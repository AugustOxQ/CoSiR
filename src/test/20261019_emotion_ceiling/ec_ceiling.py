"""CPU diagnostic: supervised linear ceiling of image-side AMI per aspect and image feature.

A logistic probe learns each aspect's painting label from image features on the 13,400 fit paintings; the 1,489 dev
paintings pick C. On the 5,000 evaluation paintings, its class probabilities are clustered at k = 20 (the image-side
AMI protocol) and its argmax is scored directly. Labels train the probe here: a diagnostic, never part of a method.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
for folder in ('20261017_image_transfer_pilot', '20261018_purified_buddy_pilot', '20261018_aspect_graph_sheet',
               '20261018_percept_core'):
    sys.path.insert(0, str(HERE.parent / folder))
from itp_data import atomic_json, load_frozen, split
from pbp_graphs import label_vectors
from pav_score import ami, partitions, summary
from apc_data import l2
from ags_trainimg import aspect_images
from pc_data import image_features

ASPECTS = ('emotion', 'style', 'genre')
PROMPT = dict(emotion='emotion_focus', style='style_word', genre='genre_word')
CS = (0.0003, 0.001, 0.003, 0.01)  # grid 1 (0.01 to 10) picked its lowest C every time
OUT = HERE / 'outputs' / 'ec_results.json'


def features(s, aspect, cache):
    if 'qwen_default' not in cache:
        cache['qwen_default'] = l2(np.asarray(image_features(s, 'qwen_default'), dtype=np.float32))
        cache['clip_image'] = l2(np.asarray(image_features(s, 'clip_image'), dtype=np.float32))
    return dict(qwen_default=cache['qwen_default'], clip_image=cache['clip_image'],
                qwen_prompted=aspect_images(s, PROMPT[aspect]))


def kmeans_ami(x, y):
    assert np.max(np.abs(np.linalg.norm(x, axis=1) - 1)) < .01
    return summary([ami(y[y >= 0], lab[y >= 0]) for lab in partitions(x, 20)])


def ceiling(x, y, fit, dev, ev):
    f, d = fit[y[fit] >= 0], dev[y[dev] >= 0]
    mu, sd = x[f].mean(0), x[f].std(0) + 1e-6
    z = (x - mu) / sd
    tried = {}
    for c in CS:
        clf = LogisticRegression(C=c, max_iter=1000).fit(z[f], y[f])
        tried[c] = (float((clf.predict(z[d]) == y[d]).mean()), clf)
    best = max(CS, key=lambda c: (tried[c][0], -c))
    clf = tried[best][1]
    prob, ye = clf.predict_proba(z[ev]), y[ev]
    keep = ye >= 0
    pred = clf.classes_[prob.argmax(1)]
    return dict(C=best, dev_accuracy={str(c): tried[c][0] for c in CS},
                eval_accuracy=float((pred[keep] == ye[keep]).mean()),
                majority_rate=float(np.bincount(ye[keep]).max() / keep.sum()),
                ami_argmax=float(ami(ye[keep], pred[keep])),
                ami_k20_prob=kmeans_ami(l2(prob.astype(np.float32)), ye),
                ami_k20_raw=kmeans_ami(x[ev], ye), n_fit=int(len(f)), n_dev=int(len(d)), n_eval=int(keep.sum()),
                n_classes=int(len(clf.classes_)))


def main():
    s = load_frozen()
    _, fit, dev = split(s)
    ev = s['subset_pid']
    labels = label_vectors(s, np.arange(len(s['first'])))
    result, cache = {}, {}
    with threadpool_limits(8):
        for aspect in ASPECTS:
            for name, x in features(s, aspect, cache).items():
                t = time.time()
                r = ceiling(x, labels[aspect], fit, dev, ev)
                result.setdefault(aspect, {})[name] = r
                print(f"{aspect:8s} {name:14s} C={r['C']:<5} acc={r['eval_accuracy']:.3f} "
                      f"(majority {r['majority_rate']:.3f}) AMI argmax={r['ami_argmax']:.3f} "
                      f"k20 probe={r['ami_k20_prob']['mean']:.3f} k20 raw={r['ami_k20_raw']['mean']:.3f} "
                      f"[{time.time() - t:.0f} s]", flush=True)
                atomic_json(OUT, result)


if __name__ == '__main__':
    main()
