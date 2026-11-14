# Claim review: recompute genre mAP change without content correction, shared vs independent seed draws.
import json, glob, sys
import numpy as np
cache = '/project/CoSiR/src/test/20261112_siglip_only/rederive/r29rd_s2abl_cache'
src = {}
for f in glob.glob(cache + '/*.json'):
    d = json.load(open(f)); h = d['identity']['npz']
    s = [k for k, v in d['inputs'].items() if v == h]
    if s: src[s[0]] = f[:-5] + '.npz'
base = '/data/SSD2/cosir_r29/20261112_siglip_only/'
def load(prefix):
    aps, q = [], None
    for s in range(5):
        z = np.load(src[f'{base}{prefix}/genre__s{s}.npz'])
        aps.append(z['ap']); q = z['qids'] if q is None else q
        assert np.array_equal(q, z['qids'])
    return np.stack(aps), q
K, qk = load('cstar/siglip1/held/siglip1_prefix')
R, qr = load('ablation/siglip1/siglip1_prefix/held/content_correction')
assert np.array_equal(qk, qr)
print('n_queries', K.shape[1], 'per-seed mAP K*', K.mean(1).round(5), 'removal', R.mean(1).round(5))
print('change (removal - K*):', (R.mean() - K.mean()))
def interval(R, K, shared, draws=10000):
    ns, n = K.shape
    rng = np.random.default_rng(0); boot = np.empty(draws)
    for st in range(0, draws, 128):
        size = min(128, draws - st)
        w = np.empty((size, n)); sR = np.empty((size, ns), int); sK = np.empty((size, ns), int)
        for j in range(size):
            w[j] = np.bincount(rng.integers(0, n, size=n), minlength=n)
            sR[j] = rng.choice(ns, size=ns, replace=True)
            sK[j] = sR[j] if shared else rng.choice(ns, size=ns, replace=True)
        mR = w @ R.T / n; mK = w @ K.T / n
        boot[st:st+size] = np.take_along_axis(mR, sR, 1).mean(1) - np.take_along_axis(mK, sK, 1).mean(1)
    return np.percentile(boot, [2.5, 97.5])
print('shared seeds     ', interval(R, K, True))
print('independent seeds', interval(R, K, False))
