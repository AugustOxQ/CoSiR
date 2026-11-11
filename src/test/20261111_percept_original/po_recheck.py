"""Independent recompute of r26's k40 test AMI from the exported test files (not po_score, not ns_metrics).

Labels: the claim test's ct_held.painting_labels (joined by painting name). k-means: scikit-learn MiniBatchKMeans,
n_init 3, batch 4096, states 0-2, on L2-normalised probabilities; AMI on rows with a label (y >= 0).
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_mutual_info_score
from threadpoolctl import threadpool_limits

sys.path.insert(0, '/project/CoSiR/src/test/20261020_claim_test')
import ct_held  # noqa: E402

OUT = Path('/data/SSD2/cosir_r26/20261111_percept_original/outputs')
K = dict(emotion=9, style=27, genre=10)
res = {r: {a: [] for a in K} for r in ('rclass', 'r20')}
names0 = None
for seed in range(5):
    z = np.load(OUT / 'test' / 'k40' / f'L1__s{seed}.npz')
    assert sorted(z.files) == ['names', 'prob'], z.files
    prob, names = z['prob'].astype(np.float32), z['names'].tolist()
    assert prob.shape == (12281, 40) and np.isfinite(prob).all() and prob.min() >= 0 and prob.max() <= 1
    assert len(set(names)) == 12281
    if names0 is None:
        names0 = names
        labels = ct_held.painting_labels(names)
        print('labelled:', {a: int((labels[a] >= 0).sum()) for a in K})
    assert names == names0, 'row order differs between seeds'
    held = np.load(OUT / 'held' / f'L1__k40__s{seed}.npz')['prob']
    assert np.array_equal(held, z['prob']), 'test export differs from the held output'
    x = prob / np.linalg.norm(prob, axis=1, keepdims=True)
    for reading in ('rclass', 'r20'):
        for a in K:
            k = K[a] if reading == 'rclass' else 20
            y = labels[a]
            m = y >= 0
            vals = []
            with threadpool_limits(limits=8):
                for st in (0, 1, 2):
                    c = MiniBatchKMeans(n_clusters=k, random_state=st, n_init=3, batch_size=4096).fit(x).labels_
                    vals.append(adjusted_mutual_info_score(y[m], c[m]))
            res[reading][a].append(float(np.mean(vals)))
    print(f'seed {seed}:', {r: {a: round(res[r][a][-1], 4) for a in K} for r in res}, flush=True)
mean = {r: {a: float(np.mean(v)) for a, v in res[r].items()} for r in res}
print('MEAN', json.dumps(mean))
