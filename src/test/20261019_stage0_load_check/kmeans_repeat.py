"""Repeat the September k-means start (K=60, n_init=10, seed 42) on the saved pre-DEC latents; compare runs bitwise."""
import hashlib
import json
import os
import sys
import time

import numpy as np
from sklearn.cluster import KMeans

tag = sys.argv[1]
repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 3
with np.load('/project/CoSiR/res/percept/percept_stage1_embedding_snapshot.npz', allow_pickle=True) as z:
    x = z['train_latent_pre']
    old = z['train_initial_label']
out = []
for r in range(repeats):
    t = time.time()
    km = KMeans(n_clusters=60, n_init=10, random_state=42).fit(x)
    c = np.ascontiguousarray(km.cluster_centers_)
    out.append(dict(tag=tag, repeat=r, load1=os.getloadavg()[0], seconds=round(time.time() - t, 1),
                    centers_sha=hashlib.sha256(c.tobytes()).hexdigest()[:16],
                    labels_sha=hashlib.sha256(km.labels_.astype(np.int64).tobytes()).hexdigest()[:16],
                    labels_equal_september=float(np.mean(km.labels_ == old)), inertia=float(km.inertia_)))
    np.save(f'/project/CoSiR/src/test/20261019_stage0_load_check/centers_{tag}_{r}.npy', c)
    print(json.dumps(out[-1]), flush=True)
