#!/root/miniconda3/envs/CoSiR/bin/python
"""Independent r29 SC5/SC6 derivation. No original round scoring imports.

Read only permitted specs, stores, exports and frozen records. `replay` is
hardwired to r14/r15 outputs, never stage-2 outputs. `held` requires an explicit
controller attestation and the digest of this script's complete validation file.
All caches and writes remain in rederive/ with this job's prefix.
"""
import os
import sys
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '8'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.special import gammaln
from sklearn.cluster import MiniBatchKMeans
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
TEST = Path('/project/CoSiR/src/test')
SPECS = Path('/project/CoSiR/docs/superpowers/specs')
R29 = Path('/data/SSD2/cosir_r29/20261112_siglip_only')
R14 = Path('/data/SSD2/cosir_r14/20261102_backbones_sweep/outputs')
R15 = Path('/data/SSD2/cosir_r15/20261102_mmart_test/outputs')
R17 = Path('/data/SSD2/cosir_r17/20261103_new_scores/outputs')
R12 = Path('/data/SSD2/cosir_r12/20261030_affection_confirmation/outputs')
R13 = Path('/data/SSD2/cosir_r13/20261031_photo_finetune/outputs')
F = Path('/data/SSD2/pre_extract')
MM_INDEX = Path('/data/SSD/mmart/mmart_index.parquet')
MM_NEAR = Path('/data/SSD/mmart/mmart_nearcopy.parquet')
AFF_ANN = Path('/data/SSD/affection/language/version_0/affection_preprocessed_public_version_0.csv')
ASPECTS = ('emotion', 'style', 'genre')
SEEDS = tuple(range(5))
COUNTS = dict(mmart=dict(emotion=9, style=20, genre=10), affection=dict(emotion=9), emoset=dict(emotion=8))
EMOTIONS = ('amusement', 'anger', 'awe', 'contentment', 'disgust', 'excitement', 'fear', 'sadness', 'something else')
VIEWS = dict(emotion='emotion_focus', style='style_word', genre='genre_word')
NOTES = [
    'Protocols: r29 §§3.2–3.5/6 and D17–D19/D22–D24/A1/A2; r15 §3.4 + mt_c13 for MMArt; r17 §3.4 + ns_c13 for class-count AMI and plain-side untold retrieval; r12 §3.4/6 and r14 §3.4 + bs_c13 for photos.',
    'MMArt and EmoSet evaluation order is ascending SHA256(key); Affection is sorted key. All joins require explicit unique keys. MMArt cut is exact copies, max_cos>=.95, then first remaining md5. EmoSet uses r14 frozen near-copy removal list; no new cut is selected.',
    'AMI: MiniBatchKMeans n_init=3, batch_size=4096, states 0,1,2; fit all rows before masking missing labels. MMArt 8 threads; photos 1 thread. Adapter exports are promoted float16->float32 without renormalization for clustering; store rows normalized once.',
    'AMI intervals: fixed partitions, 2000 draws, fresh default_rng(0) per cell; draw all rows then five seed positions. Mean over states then seeds; unlabelled rows masked after sampling. Retrieval intervals: 10000 draws, queries then five seeds, fresh default_rng(0) per cell.',
    'R0 seed-range measurements use the original normalized rows directly; application of a picked none map and cross-fit none still normalize in float64 then cast float32, exactly as the other W maps. A frozen-only probe confirmed this matters at the 1e-9 gate.',
    'Retrieval numeric convention (permitted rt_metrics/rt_readout docstrings): input float32 normalized rows; maps/normalization float64 then cast float32; cosine float64, stable descending sort with self removed; QE normalized mean of query and k neighbours. AP and all reductions float64.',
    'Read-outs: r11/r12 menu none,center,whiten1024,whiten256,whiten64 crossed k=0,1,3,5,10. Parity-half cross-fit fit before masking unlabelled rows; mean of two half mAPs then seed means. Covariance ddof=1, epsilon=1e-6*largest eigenvalue. Guard permits no aspect below R0 minus its range; within .001 score units ties choose R0, no expansion, none<center<whiten, smaller k, larger PCA size.',
    'MMArt plain-side candidate uses r17 SC7 untold concatenation: normalize each E/S/G head then divide joined vector by sqrt(3); Q1-side candidate uses told aspect heads (r15). Separate picks are made, with told aspect-head R0 ranges as the shared per-aspect units. This inherits the paper cells under D18; r29 table itself does not explicitly say untold.',
    'EmoSet D19 prose rounds r_E to .00695. Use full precision committed r14 stage2_emoset ranges.map.r, .0069460867449697705, as r28. This unit serves F1 EmoSet read-out selection and LEVEL; its own R0 range is descriptive only.',
    'Affection labels are plurality over non-ignored annotation rows, alphabetical emotion tie breaking; split consistency is checked. EmoSet class names come only from split-file label fields, never parsed from image paths.',
    'Comparator picks and values are frozen inputs; no comparator is selected on candidate validation or held data. Rebuilt mAP points must match the unrounded record within 1e-9, or only the affected cells are INCONCLUSIVE.',
    'Stage-2 held runs are controller-only, after picks commit: require --picks-committed and --val-sha256; complete validation on all five seeds and full menu only. Replay mode hardcodes old exports. No stage-2 held values were calculated during development.',
    'Open choices follow earlier rounds: numpy percentile linear interpolation, default_rng(0), stable ties, inclusive .001 tie tolerance with 1e-12 arithmetic guard. No multiplicity correction. Inconclusive outcomes count neither WIN nor LEVEL, with maximum-possible ladder reported under D22.',
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def keysha(keys):
    return hashlib.sha256('\n'.join(keys).encode()).hexdigest()


def join(have, want):
    have, want = list(have), list(want)
    assert len(set(have)) == len(have), 'duplicate input keys'
    assert len(set(want)) == len(want), 'duplicate requested keys'
    pos = {k: i for i, k in enumerate(have)}
    return np.asarray([pos[k] for k in want], np.int64)


def order(keys):
    keys = list(keys)
    join(keys, keys)
    return sorted(keys, key=lambda k: hashlib.sha256(k.encode()).hexdigest())


def unit(x, dtype=np.float32):
    x = np.asarray(x, dtype=dtype)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    assert np.isfinite(x).all() and (norms > 0).all()
    return x / norms


def write(path, obj):
    path = Path(path).resolve()
    assert path.parent == HERE and (path.name.startswith('r29rd_s2mp') or path.name in ('s2mp_val.json', 's2mp_held.json')), path
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


class IO:
    def __init__(self):
        self.inputs = {}
        self.replays = []
        self.started = time.monotonic()
        self.file(Path(__file__))
        for name in ('2026-11-12-siglip-only-adapter-design.md', '2026-11-02-mmart-test-design.md',
                     '2026-11-03-new-scores-design.md', '2026-11-02-backbones-sweep-design.md',
                     '2026-10-30-affection-confirmation-design.md', '2026-10-29-retrieval-design.md'):
            self.file(SPECS / name)
        for rel in ('20261103_new_scores/ns_c13.py', '20261102_mmart_test/mt_c13.py',
                    '20261102_backbones_sweep/bs_c13.py', '20261102_backbones_sweep/bs_c13_partb.py'):
            self.file(TEST / rel)

    def file(self, path):
        path = Path(path).resolve()
        # Defense against accidentally opening any forbidden r29 scorer artifacts.
        forbidden = TEST / '20261112_siglip_only'
        if path.is_relative_to(forbidden) and not path.is_relative_to(HERE):
            assert path.name in ('r29_model.py', 'r29_captions.py'), path
        if path.is_relative_to(R29):
            assert path.relative_to(R29).parts[0] in ('cstar', 'mmart', 'photo', 'exports'), path
        key = str(path)
        if key not in self.inputs:
            self.inputs[key] = sha(path)
        return self.inputs[key]

    def json(self, path):
        self.file(path)
        return json.loads(Path(path).read_text())

    def npz(self, path):
        self.file(path)
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    def cache(self, name, arrays, params, compute):
        # Source, input arrays and all numerical choices bind cache identity.
        h = hashlib.sha256((sha(__file__)+np.__version__+__import__('sklearn').__version__).encode())
        h.update(json.dumps(params, sort_keys=True).encode())
        for a in arrays:
            a = np.asarray(a)
            h.update(str((a.shape, a.dtype.str)).encode())
            h.update(np.ascontiguousarray(a).tobytes())
        p = HERE / f'r29rd_s2mp_cache_{name}_{h.hexdigest()}.npz'
        if p.exists():
            return self.npz(p)['value']
        t = time.monotonic()
        v = np.asarray(compute())
        assert np.isfinite(v).all(), name
        np.savez_compressed(p, value=v)
        self.file(p)
        print(f'{name}: {time.monotonic()-t:.2f}s', flush=True)
        return v

    def compare(self, family, path, key, value, recorded, tolerance=1e-9):
        diff = abs(float(value)-float(recorded)) if isinstance(recorded, (int, float)) else (0. if value == recorded else None)
        row = dict(cell_type=family, target_record=str(path), key=key, value=value, recorded_value=recorded,
                   absolute_difference=diff, tolerance=tolerance, passed=diff is not None and diff <= tolerance)
        self.replays.append(row)
        return row['passed']

    def envelope(self, **content):
        return dict(**content, replays=self.replays, notes=NOTES, inputs=dict(sorted(self.inputs.items())),
                    seconds=time.monotonic()-self.started, versions=dict(numpy=np.__version__, sklearn=__import__('sklearn').__version__))


class Data:
    def __init__(self, io):
        self.io = io
        self.frames = {}
        self.store_indices = {}

    def frame(self, dataset, split):
        tag = dataset, split
        if tag in self.frames:
            return self.frames[tag]
        if dataset == 'mmart':
            self.io.file(MM_INDEX)
            index = pd.read_parquet(MM_INDEX)
            assert index.key.is_unique
            rows = index[index.split == ('val' if split == 'val' else 'new')].set_index('key')
            keys = order(rows.index)
            if split != 'val':
                self.io.file(MM_NEAR)
                near = pd.read_parquet(MM_NEAR).set_index('key')
                assert near.index.is_unique
                seen, kept = set(), []
                for k in keys:
                    row = rows.loc[k]
                    if row.exact_dup_trainval or near.loc[k, 'max_cos'] >= .95 or row.md5 in seen:
                        continue
                    seen.add(row.md5)
                    kept.append(k)
                keys = kept
            rows = rows.loc[keys]
            labels = {}
            for a in ASPECTS:
                values = rows['genre_name' if a == 'genre' else a].fillna('').tolist()
                codes = {v: i for i, v in enumerate(sorted(set(values)-{''}))}
                labels[a] = np.asarray([codes.get(v, -1) for v in values], np.int64)
            names = rows.painting.tolist()
            assert len(keys) == (5769 if split == 'val' else 16384)
        elif dataset == 'affection':
            self.io.file(AFF_ANN)
            annotations = pd.read_csv(AFF_ANN, usecols=['dataset', 'image_name', 'emotion', 'split'])
            annotations = annotations[annotations.split != 'ignored'].copy()
            annotations['key'] = annotations.dataset + '/' + annotations.image_name
            assert annotations.groupby('key').split.nunique().max() == 1
            rows = annotations[annotations.split == split]
            counts = defaultdict(Counter)
            for k, e in zip(rows.key, rows.emotion):
                assert e in EMOTIONS
                counts[k][e] += 1
            keys = sorted(counts)
            y = [EMOTIONS.index(min(counts[k], key=lambda e: (-counts[k][e], e))) for k in keys]
            labels, names = {'emotion': np.asarray(y, np.int64)}, keys
            assert len(keys) == (4251 if split == 'val' else 8501)
            expected = [798,54,622,1387,189,620,203,270,108] if split == 'val' else [1651,81,1460,2626,351,1185,448,501,198]
            assert np.bincount(labels['emotion']).tolist() == expected
        else:
            path = Path('/data/SSD/emoset') / f'{split}.json'
            rows = self.io.json(path)
            classes = {e: i for i, e in enumerate(sorted(set(r[0] for r in rows)))}
            assert len(classes) == 8
            keyed = {r[1]: r[0] for r in rows}
            assert len(keyed) == len(rows)
            near = self.io.json(TEST / '20261102_backbones_sweep/records/nearcopy_emoset.json')
            removed = set(near['splits'][split]['removed_keys'])
            assert removed <= set(keyed)
            keys = order(set(keyed)-removed)
            names = keys
            labels = {'emotion': np.asarray([classes[keyed[k]] for k in keys], np.int64)}
            assert len(keys) == (5770 if split == 'val' else 17348)
        self.frames[tag] = dict(keys=keys, names=names, labels=labels)
        return self.frames[tag]

    def store(self, folder, keys, field='key'):
        folder = Path(folder)
        if folder not in self.store_indices:
            if (folder / 'keys.json').exists():
                have = self.io.json(folder / 'keys.json')
            elif (folder / 'ids.npz').exists():
                ids = self.io.npz(folder / 'ids.npz')
                have = ids['painting_names'].tolist()
            else:
                idx = self.io.json(folder / 'index.json')
                have = [r[field] for r in idx['images']]
            join(have, have)
            self.store_indices[folder] = have
        have = self.store_indices[folder]
        self.io.file(folder / 'img.npy')
        raw = np.load(folder / 'img.npy', mmap_mode='r', allow_pickle=False)
        return np.asarray(raw[join(have, keys)], np.float32)

    def vectors(self, dataset, split, arm, aspect='emotion', seed=0, frozen=False, retrieval=False, photo_encoder='siglip1'):
        fr = self.frame(dataset, split)
        keys = fr['keys']
        if arm in ('candidate', 'untold'):
            if arm == 'untold':
                heads = [unit(self.vectors(dataset, split, 'candidate', a, seed, frozen)) for a in ASPECTS]
                return (np.concatenate(heads, axis=1) / np.float32(np.sqrt(3))).astype(np.float32)
            if dataset == 'mmart':
                base = R15 if frozen else R29 / 'mmart/siglip1_prefix'
                path = base / 'exports' / ('val' if split == 'val' else 'new') / 'siglip1/cstar' / f'{aspect}__s{seed}.npz'
            else:
                base = R14 if frozen else R29
                path = base / 'exports' / dataset / f'photo_{photo_encoder}' / f'{split}__s{seed}.npz'
            side = self.io.json(path.with_suffix('.json'))
            assert side.get('sha256') == self.io.file(path), f'export digest mismatch: {path}'
            z = self.io.npz(path)
            have = z.get('keys', z.get('names'))
            if have is None:
                have = side.get('keys', side.get('names'))
            assert have is not None, f'no explicit row identities: {path}'
            have = list(have)
            if 'keys_sha256' in side:
                assert keysha(have) == side['keys_sha256']
            x = z.get('emb', z.get('img'))
            assert x is not None and x.dtype == np.float16 and x.shape == (len(have), 2048)
            assert np.isfinite(x).all()
            assert np.max(np.abs(np.linalg.norm(x.astype(np.float32),axis=1)-1)) < 1e-3, path
            if dataset == 'mmart':
                assert side['encoder'] == 'siglip1' and side['aspect'] == aspect and side['seed'] == seed
                assert side['keyset'] == ('val' if split == 'val' else 'new')
                if not frozen:
                    assert side.get('device') == 'cpu', 'MMArt export must be CPU, as r15'
            else:
                assert (side['dataset'], side['split'], side['seed']) == (dataset, split, seed)
                assert side['arm'] == f'photo_{photo_encoder}'
            x = x[join(have, keys)].astype(np.float32)
        else:
            view = 'siglip_so400m' if arm == 'p1' else 'qwen3vl_emb_2b' if arm == 'qwen_plain' else f'qwen3vl_emb_2b_aspect_prompts/{VIEWS[aspect]}'
            if dataset == 'mmart' and split == 'val':
                if arm == 'q1':
                    view += '_s2img'
                x = self.store(F / 'artelingo_full' / view, fr['names'], 'painting')
            else:
                root = {'mmart':'mmart', 'affection':'affection_local', 'emoset':'emoset'}[dataset]
                x = self.store(F / root / view, keys)
            x = unit(x)
            # r15 mt_data.features returns normalized store rows; retrieval caller normalizes again.
            if dataset == 'mmart' and retrieval:
                x = unit(x)
            return x
        return unit(x) if retrieval else x


def menu(width):
    ws = ['none', 'center'] + [f'whiten{d}' for d in sorted({min(width, d) for d in (64,256,1024)}, reverse=True)]
    return [f'{w}+k{k}' for w in ws for k in (0,1,3,5,10)]


def parse(readout):
    w, k = readout.split('+k')
    return w, int(k)


def mapping(fit, w):
    x = np.asarray(fit, np.float64)
    if w == 'none':
        return None, None
    mu = x.mean(0)
    if w == 'center':
        return mu, None
    centered = x-mu
    cov = centered.T @ centered / (len(x)-1)
    size = min(int(w[6:]), cov.shape[0])
    # Full symmetric solver preserves earlier round's eigensolver convention.
    vals, vecs = np.linalg.eigh(cov)
    vals, vecs = vals[::-1], vecs[:, ::-1]
    eps = 1e-6*max(float(vals[0]), 1e-12)
    return mu, vecs[:, :size] / np.sqrt(np.maximum(vals[:size], 0)+eps)


def mapped(x, fit):
    mu, transform = fit
    z = np.asarray(x, np.float64)
    if mu is not None:
        z = z-mu
    if transform is not None:
        z = z@transform
    return unit(z, np.float64).astype(np.float32)


def ranking(q, gallery, own, device='cpu'):
    if device == 'cpu':
        scores = np.asarray(q, np.float64) @ np.asarray(gallery, np.float64).T
    else:
        import torch
        with torch.inference_mode():
            g = gallery if torch.is_tensor(gallery) else torch.as_tensor(np.asarray(gallery,dtype=np.float64),device=device)
            scores = (torch.as_tensor(q, dtype=torch.float64, device=device) @ g.T).cpu().numpy()
    scores[np.arange(len(q)), own] = -np.inf
    ranked = np.argsort(-scores, axis=1, kind='stable')
    return ranked[ranked != np.asarray(own)[:, None]].reshape(len(q), -1)


def aps(x, y, ks=(0,), limit=0, batch=64, device='cpu'):
    valid = np.flatnonzero(y >= 0)
    z, yy = np.asarray(x)[valid], np.asarray(y)[valid]
    queries = np.arange(len(yy))
    if device=='cpu':
        rank_gallery=z.astype(np.float64)
    else:
        import torch
        rank_gallery=torch.as_tensor(z.astype(np.float64),device=device)
    if limit:
        queries = np.unique(np.linspace(0, len(yy)-1, min(limit,len(yy)), dtype=np.int64))
    result = {k: [] for k in ks}
    for start in range(0, len(queries), batch):
        idx = queries[start:start+batch]
        q = z[idx]
        ranks = ranking(q, rank_gallery, idx, device)
        for k in ks:
            rank = ranks
            if k:
                assert k < len(z)
                # Frozen metric forms the average in float64, normalizes and casts once.
                expanded = unit((q.astype(np.float64) + z[ranks[:,:k]].astype(np.float64).sum(1))/(k+1), np.float64).astype(np.float32)
                rank = ranking(expanded, rank_gallery, idx, device)
            relevant = yy[rank] == yy[idx, None]
            den = relevant.sum(1)
            values = ((relevant.cumsum(1, dtype=np.float64) / np.arange(1, relevant.shape[1]+1)) * relevant).sum(1)
            ap = np.divide(values, den, out=np.full(len(idx), np.nan), where=den > 0)
            result[k].append(ap)
    return {k: np.concatenate(v) for k,v in result.items()}, valid[queries]


def image_ap(val, x, y, readout, args):
    w, k = parse(readout)
    z = mapped(x, mapping(val, w))
    return aps(z, y, (k,), args.query_limit, args.batch_size, args.device)[0][k]


def crossfit(x, labels, args):
    readouts = menu(x.shape[1]) if not args.quick else ['none+k0', 'center+k0', 'center+k10']
    result = {a: {r: [] for r in readouts} for a in labels}
    for parity in (0,1):
        test = np.arange(len(x)) % 2 == parity
        fit, held = x[~test], x[test]
        spectral = None
        for w in dict.fromkeys(parse(r)[0] for r in readouts):
            if w.startswith('whiten'):
                if spectral is None:
                    spectral = mapping(fit, 'whiten'+str(min(x.shape[1],1024)))
                mp = spectral[0], spectral[1][:,:int(w[6:])]
            else:
                mp = mapping(fit, w)
            z = mapped(held, mp)
            ks = [parse(r)[1] for r in readouts if parse(r)[0] == w]
            for a, full_y in labels.items():
                got, _ = aps(z, full_y[test], ks, args.query_limit, args.batch_size, args.device)
                for k, ap in got.items():
                    result[a][f'{w}+k{k}'].append(float(np.nanmean(ap)))
    return {a: {r: float(np.mean(v)) for r,v in rs.items()} for a,rs in result.items()}


def select(rows, ranges):
    aspects = tuple(ranges)
    assert aspects and all(np.isfinite(ranges[a]) and ranges[a] > 0 for a in aspects)
    base = rows['none+k0']
    scores = {r: float(np.mean([(row[a]-base[a])/ranges[a] for a in aspects])) for r,row in rows.items()}
    guard = {r: all(row[a] >= base[a]-ranges[a]-1e-12 for a in aspects) for r,row in rows.items()}
    best = max(scores[r] for r in rows if guard[r])
    tied = [r for r in rows if guard[r] and scores[r] >= best-.001-1e-12]
    def priority(r):
        w,k = parse(r)
        return r!='none+k0', k>0, 0 if w=='none' else 1 if w=='center' else 2, k, -int(w[6:]) if w.startswith('whiten') else 0
    pick = min(tied, key=priority)
    return dict(pick=pick, table={r:dict(map=rows[r], score=scores[r], kept=guard[r]) for r in rows}, units=ranges)


def partitions(x, k, dataset):
    with threadpool_limits(8 if dataset == 'mmart' else 1):
        return np.stack([MiniBatchKMeans(n_clusters=k, n_init=3, batch_size=4096, random_state=s).fit(np.asarray(x,np.float32)).labels_
                         for s in (0,1,2)]).astype(np.int32)


@lru_cache(maxsize=32)
def logfacts(n):
    return gammaln(np.arange(n+1, dtype=np.float64)+1)


def ami_table(C):
    # Independent exact hypergeometric EMI; formula from permitted ns_c13.
    C = C[C.sum(1)>0][:,C.sum(0)>0]
    n = int(C.sum())
    if n == 0 or C.shape == (1,1):
        return 1.
    a,b = C.sum(1), C.sum(0)
    i,j = C.nonzero()
    v = C[i,j].astype(np.float64)
    mi = max(0., float(np.sum(v/n*(np.log(v)+np.log(n)-np.log(a[i])-np.log(b[j])))))
    ha,hb = -np.sum(a/n*np.log(a/n)), -np.sum(b/n*np.log(b/n))
    au,ac = np.unique(a, return_counts=True)
    bu,bc = np.unique(b, return_counts=True)
    g,emi = logfacts(n), 0.
    for ai, mult in zip(au,ac):
        ks = np.arange(1,min(int(ai),int(bu.max()))+1,dtype=np.int64)[None,:]
        bj = bu[:,None]
        valid = (ks<=bj)&(ks>=np.maximum(1,ai+bj-n))
        lp = g[ai]+g[n-ai]+g[bj]+g[n-bj]-g[n]-g[ks]-g[ai-ks]-g[np.maximum(bj-ks,0)]-g[np.maximum(n-ai-bj+ks,0)]
        terms = np.exp(np.where(valid,lp,-np.inf))*(ks/n)*(np.log(n)+np.log(ks)-np.log(ai)-np.log(bj))
        emi += mult*float(np.sum(terms*bc[:,None]))
    den = (ha+hb)/2-emi
    eps = np.finfo(np.float64).eps
    den = min(den,-eps) if den < 0 else max(den,eps)
    return float((mi-emi)/den)


def ami(parts, y, weights=None, seeds=None):
    p = np.asarray(parts)
    p = p[None] if p.ndim == 2 else p
    chosen = np.arange(len(p)) if seeds is None else seeds
    good = y>=0
    yy = y[good]
    w = np.ones(len(yy),np.int64) if weights is None else weights[good]
    ny = int(yy.max())+1
    out = {}
    for s in np.unique(chosen):
        values=[]
        for part in p[s]:
            pp = part[good]
            nk = int(pp.max())+1
            C = np.bincount(yy*nk+pp,weights=w,minlength=ny*nk).reshape(ny,nk).astype(np.int64)
            values.append(ami_table(C))
        out[s]=float(np.mean(values))
    return float(np.mean([out[s] for s in chosen]))


_BOOT=None

def _boot_init(cand, comp, y):
    global _BOOT
    _BOOT=cand,comp,y
    threadpool_limits(1)


def _ami_draw(task):
    cand,comp,y = _BOOT
    rows,seeds=task
    w=np.bincount(rows,minlength=len(y))
    return ami(cand,y,w,seeds)-ami(comp,y,w)


def ami_interval(cand, comp, y, draws=2000, workers=8):
    import multiprocessing as mp
    cv,bv=ami(cand,y),ami(comp,y)
    rng=np.random.default_rng(0)
    tasks=[(rng.integers(0,len(y),size=len(y)), rng.choice(5,size=5,replace=True)) for _ in range(draws)]
    if workers>1:
        with mp.get_context('spawn').Pool(workers, initializer=_boot_init, initargs=(cand,comp,y)) as pool:
            boot=np.asarray(pool.map(_ami_draw,tasks,chunksize=8))
    else:
        _boot_init(cand,comp,y)
        boot=np.asarray([_ami_draw(t) for t in tasks])
    lo,hi=np.percentile(boot,[2.5,97.5])
    return dict(value=cv,comparator_value=bv,d=cv-bv,lo=float(lo),hi=float(hi))


def ret_interval(cand, comp, draws=10000):
    cand,comp=np.asarray(cand,np.float64),np.asarray(comp,np.float64)
    assert cand.shape == (5,len(comp)) and np.isfinite(cand).all() and np.isfinite(comp).all()
    n=len(comp)
    rng=np.random.default_rng(0)
    boot=np.empty(draws,np.float64)
    for start in range(0,draws,128):
        size=min(128,draws-start)
        counts=np.empty((size,n),np.float64)
        selected=np.empty((size,5),np.int64)
        for b in range(size):
            counts[b]=np.bincount(rng.integers(0,n,size=n),minlength=n)
            selected[b]=rng.choice([0,1,2,3,4],size=5,replace=True)
        cm=counts@cand.T/n
        bm=counts@comp/n
        boot[start:start+size]=np.take_along_axis(cm,selected,1).mean(1)-bm
    lo,hi=np.percentile(boot,[2.5,97.5])
    cv,bv=float(cand.mean()),float(comp.mean())
    return dict(value=cv,comparator_value=bv,d=cv-bv,lo=float(lo),hi=float(hi))


def outcome(lo,r):
    return 'WIN' if lo>0 else 'LEVEL' if lo>-r else 'BELOW'


def ladder(cells, n, threshold):
    expected={c['id'] for c in cells}
    assert len(cells)==2*n and len(expected)==2*n
    p=sum(c['id'].endswith('/p1') and c['outcome']=='WIN' for c in cells)
    q=sum(c['id'].endswith('/q1') and c['outcome'] in ('WIN','LEVEL') for c in cells)
    up=sum(c['id'].endswith('/p1') and c['outcome']=='INCONCLUSIVE' for c in cells)
    uq=sum(c['id'].endswith('/q1') and c['outcome']=='INCONCLUSIVE' for c in cells)
    def lev(x,y):
        return 3 if x==n and y>=threshold else 2 if x==n else 1 if x>=threshold else 0
    actual,possible=lev(p,q),lev(p+up,q+uq)
    return dict(level=actual,wins_p1=p,wins_or_level_q1=q,inconclusive_p1=up,inconclusive_q1=uq,
                maximum_possible_level=possible,inconclusive_above=possible>actual)


class Work:
    def __init__(self, args):
        self.args=args
        self.io=IO()
        self.data=Data(self.io)
        self.memory={}

    def vectors(self,d,sp,arm,a='emotion',s=0,frozen=False,retrieval=False,encoder='siglip1'):
        return self.data.vectors(d,sp,arm,a,s,frozen,retrieval,encoder)

    def parts(self,d,sp,arm,a,s=0,frozen=False,k=None,encoder='siglip1'):
        k=k or COUNTS[d][a]
        tag=(d,sp,arm,a,s,frozen,k,encoder)
        if tag not in self.memory:
            x=self.vectors(d,sp,arm,a,s,frozen,encoder=encoder)
            self.memory[tag]=self.io.cache('parts',[x],dict(dataset=d,k=k,threads=8 if d=='mmart' else 1),lambda:partitions(x,k,d))
        return self.memory[tag]

    def arm_parts(self,d,sp,arm,a,frozen=False,k=None,encoder='siglip1'):
        if arm=='candidate':
            return np.stack([self.parts(d,sp,arm,a,s,frozen,k,encoder) for s in self.args.seeds])
        return self.parts(d,sp,arm,a,frozen=frozen,k=k,encoder=encoder)

    def ap(self,d,sp,arm,a,s=0,frozen=False,pick='none+k0',encoder='siglip1',raw_r0=False):
        x=self.vectors(d,sp,arm,a,s,frozen,True,encoder)
        y=self.data.frame(d,sp)['labels'][a]
        # Raw R0 must not be normalized an extra time by readout application.
        val=x if sp=='val' or parse(pick)[0]=='none' else self.vectors(d,'val',arm,a,s,frozen,True,encoder)
        return self.io.cache('ap',[val,x,y],dict(pick=pick,raw_r0=raw_r0,limit=self.args.query_limit,batch=self.args.batch_size,device=self.args.device),
                             lambda:aps(x,y,(0,),self.args.query_limit,self.args.batch_size,self.args.device)[0][0] if raw_r0 else image_ap(val,x,y,pick,self.args))

    def range(self,d,a,frozen=False,k=None,encoder='siglip1'):
        y=self.data.frame(d,'val')['labels'][a]
        ps=[];rs=[]
        for s in self.args.seeds:
            if self.args.metric in ('all','ami'):
                ps.append(ami(self.parts(d,'val','candidate',a,s,frozen,k,encoder),y))
            if self.args.metric in ('all','map'):
                rs.append(float(self.ap(d,'val','candidate',a,s,frozen,encoder=encoder,raw_r0=True).mean()))
        def record(values):
            return dict(per_seed={str(s):v for s,v in zip(self.args.seeds,values)},r=float(np.ptp(values)),
                        n=len(y),n_labelled=int((y>=0).sum()))
        return {m:record(v) for m,v in [('ami',ps),('map',rs)] if v}

    def pick(self,d,arm,ranges,frozen=False,encoder='siglip1'):
        labels=self.data.frame(d,'val')['labels']
        tables=[]
        for s in self.args.seeds if arm in ('candidate','untold') else (0,):
            per_aspect={}
            # Untold vector and plain rows are identical across aspects: share PCA fits.
            if arm in ('untold','p1'):
                x=self.vectors(d,'val',arm,'emotion',s,frozen,True,encoder)
                needed={a:labels[a] for a in ranges}
                ros=menu(x.shape[1]) if not self.args.quick else ['none+k0','center+k0','center+k10']
                values=self.io.cache('crossfit',[x,*needed.values()],dict(readouts=ros,limit=self.args.query_limit,device=self.args.device,batch=self.args.batch_size),
                    lambda:self._crossfit_array(x,needed,ros))
                per_aspect.update({a:dict(zip(ros,values[i].tolist())) for i,a in enumerate(needed)})
            else:
                for a in ranges:
                    x=self.vectors(d,'val',arm,a,s,frozen,True,encoder)
                    ros=menu(x.shape[1]) if not self.args.quick else ['none+k0','center+k0','center+k10']
                    values=self.io.cache('crossfit',[x,labels[a]],dict(readouts=ros,limit=self.args.query_limit,device=self.args.device,batch=self.args.batch_size),
                        lambda x=x,a=a,ros=ros:self._crossfit_array(x,{a:labels[a]},ros))
                    per_aspect[a]=dict(zip(ros,values[0].tolist()))
            tables.append({r:{a:per_aspect[a][r] for a in ranges} for r in ros})
        rows={r:{a:float(np.mean([t[r][a] for t in tables])) for a in ranges} for r in tables[0]}
        return select(rows,ranges)

    def _crossfit_array(self,x,labels,ros):
        result=crossfit(x,labels,self.args)
        return np.asarray([[result[a][r] for r in ros] for a in labels])

    def ns_cells(self):
        return self.io.json(TEST/'20261103_new_scores/records/verdicts.json')['cells']

    def ns_cell(self,sc,d,a,comp,metric='ami',encoder='siglip1'):
        found=[r for r in self.ns_cells() if (r['sc'],r['set'],r['encoder'],r['aspect'],r['metric'],r['comparator'])==(sc,d,encoder,a,metric,comp)]
        assert len(found)==1, (sc,d,a,comp,metric,encoder)
        return found[0]

    def r14_cell(self,sc,d,metric='ret',encoder='siglip1'):
        path=R14/'stage3'/f'cells_{d}.json'
        found=[r for r in self.io.json(path) if r['sc']==sc and r['encoder']==encoder and r['task']==metric]
        assert len(found)==1,(sc,d,metric,encoder)
        return path,found[0]

    def comparator(self,d,a,comp,metric):
        """Frozen pick and unrounded point target; never derives a new pick."""
        if metric=='ami':
            sc='SC1' if d=='mmart' and comp=='p1' else 'SC3' if d=='mmart' else 'SC4'
            encoder='qwen' if d=='emoset' else 'siglip1'
            if d=='emoset' and comp=='p1':
                path=TEST/'20261112_siglip_main/records/photos.json'
                return None,self.io.json(path)['cells']['c']['ami']['mean'],path,'cells.c.ami.mean'
            row=self.ns_cell(sc,d,a,'plain' if comp=='p1' else 'teacher',encoder=encoder)
            return None,row['comp'],TEST/'20261103_new_scores/records/verdicts.json',row['id']+'.comp'
        if d=='mmart':
            if comp=='p1':
                path=R15/'stage1/picks_siglip1.json'
                pick=self.io.json(path)['arms']['frozen']['pick']
                row=self.ns_cell('SC7',d,a,'plain','map')
                return pick,row['comp'],TEST/'20261103_new_scores/records/verdicts.json',row['id']+'.comp'
            pick=self.io.json(R15/'stage1/picks_teacher.json')['arms']['teacher']['pick']
            path=R15/'stage2/cells_ret.json'
            found=[r for r in self.io.json(path) if r['encoder']=='siglip1' and r['aspect']==a and r['sc']=='SC4']
            assert len(found)==1
            row=found[0]
            return pick,row['comparator_value'],path,f'SC4/siglip1/{a}.comparator_value'
        if d=='affection':
            if comp=='p1':
                path,row=self.r14_cell('SC5',d)
                return row['readout_comparator'],row['comparator_value'],path,'SC5/siglip1.comparator_value'
            path=R13/'stage3/verdict.json'
            row=self.io.json(path)['sc']['SC4']
            return row['readout_t'],row['point_t'],path,'sc.SC4.point_t'
        if comp=='p1':
            path=TEST/'20261112_siglip_main/records/photos.json'
            row=self.io.json(path)['cells']['c']
            return row['readout'],row['map']['mean'],path,'cells.c.map.mean'
        path,row=self.r14_cell('SC10',d,encoder='qwen')
        return row['readout_comparator'],row['comparator_value'],path,'SC10/qwen.comparator_value'

    def rebuild(self,d,a,comp,metric):
        pick,recorded,path,key=self.comparator(d,a,comp,metric)
        if metric=='ami':
            y=self.data.frame(d,'test')['labels'][a]
            values=self.arm_parts(d,'test',comp,a)
            value=ami(values,y)
        else:
            values=self.ap(d,'test',comp,a,pick=pick)
            value=float(values.mean())
        full=not self.args.query_limit
        passed=self.io.compare(d+'/'+metric+'/rebuild',path,key,value,recorded) if full else None
        return values,dict(pick=pick,value=value,recorded_value=recorded,passed=passed,
                           target_record=str(path),key=key,full_queries=full)

    def compare_stats(self,family,path,key,stats,target,full_interval=True):
        fields={'value':'cand','comparator_value':'comp','d':'d','lo':'lo','hi':'hi'} if 'cand' in target else {
            'value':'candidate_value','comparator_value':'comparator_value','d':'d','lo':'lo','hi':'hi'}
        for own,old in fields.items():
            if own in ('lo','hi') and not full_interval:
                continue
            self.io.compare(family,path,key+'.'+old,stats[own],target[old])
        if 'outcome' in target and full_interval:
            r=target.get('r_used',target.get('r'))
            self.io.compare(family,path,key+'.outcome',outcome(stats['lo'],r or 0.),target['outcome'])

    def validation(self):
        result={}
        for piece in pieces(self.args):
            try:
                if piece=='mmart':
                    ranges={a:self.range('mmart',a) for a in ASPECTS}
                    units={a:ranges[a]['map']['r'] for a in ASPECTS}
                    picks={side:self.pick('mmart',arm,units) for side,arm in [('p1','untold'),('q1','candidate')]}
                else:
                    ranges={d:self.range(d,'emotion') for d in ('affection','emoset')}
                    committed=self.io.json(TEST/'20261102_backbones_sweep/records/stage2_emoset.json')['ranges']['map']['r']
                    ranges['emoset']['map']['own_r0_range']=ranges['emoset']['map']['r']
                    ranges['emoset']['map']['r']=committed
                    ranges['emoset']['map']['r_source']='r14 committed r_E (D19), not F1 own range'
                    picks={d:self.pick(d,'candidate',{'emotion':ranges[d]['map']['r']}) for d in ('affection','emoset')}
                result[piece]=dict(picks=picks,ranges=ranges,complete=full_validation(self.args))
            except FileNotFoundError as e:
                result[piece]=dict(picks={},ranges={},complete=False,pending=str(e.filename or e))
                print(f'{piece}: pending input {e.filename}',flush=True)
        forward=self.forward_check() if self.args.forward_check else {'status':'not_requested', 'reason':'Use val --forward-check after CPU MMArt validation exports exist; optional photo check deferred.'}
        return self.io.envelope(**result,mode='validation',complete=all(v['complete'] for v in result.values()),forward_check=forward)

    def forward_check(self):
        import torch
        torch.set_num_threads(8)
        fr=self.data.frame('mmart','val')
        x=self.vectors('mmart','val','p1')[:500]
        results=[]
        for a in ASPECTS:
            path=R29/'cstar/siglip1/weights/siglip1_prefix'/f'{a}__s0.pt'
            try:
                self.io.file(path)
                target=self.vectors('mmart','val','candidate',a,0)[:500].astype(np.float16)
            except FileNotFoundError as e:
                results.append(dict(aspect=a,status='pending',reason=str(e)))
                continue
            state=torch.load(path,map_location='cpu',weights_only=False)
            state=state.get('state_dict',state.get('model',state))
            assert isinstance(state,dict)
            state={k.removeprefix('module.'):v for k,v in state.items()}
            # r29_model allowed architecture; fixed two-layer image tower.
            import torch.nn.functional as fn
            with torch.inference_mode():
                z=torch.as_tensor(x)
                z=fn.gelu(fn.linear(z,state['img.trunk.0.weight'],state['img.trunk.0.bias']))
                z=fn.linear(z,state['img.output.weight'],state['img.output.bias'])
                pred=fn.normalize(z,dim=1).numpy().astype(np.float16)
            equal=pred==target
            up=np.nextafter(target,np.float16(np.inf)).astype(np.float32)-target.astype(np.float32)
            down=target.astype(np.float32)-np.nextafter(target,np.float16(-np.inf)).astype(np.float32)
            delta=np.abs(pred.astype(np.float32)-target.astype(np.float32))
            passed=equal.mean()>=.995 and np.all(delta<=np.maximum(up,down))
            results.append(dict(aspect=a,n=500,keys_sha256=keysha(fr['keys'][:500]),equal_fraction=float(equal.mean()),max_abs_diff=float(delta.max()),passed=bool(passed),device='cpu'))
        return dict(mmart=results,photo={'status':'optional_not_implemented'})

    def replay(self):
        reports=[]
        for d in datasets(self.args):
            for group in (('validation','ami','map','rebuild') if self.args.replay_group=='all' else (self.args.replay_group,)):
                if group=='validation':
                    reports.extend(self.replay_validation(d))
                elif group=='ami' and self.args.metric in ('all','ami'):
                    reports.extend(self.replay_ami(d))
                elif group=='map' and self.args.metric in ('all','map'):
                    reports.extend(self.replay_map(d))
                elif group=='rebuild':
                    for a in selected_aspects(self.args,d):
                        for comp in self.args.comparators:
                            for m in (('ami','map') if self.args.metric=='all' else (self.args.metric,)):
                                _,report=self.rebuild(d,a,comp,m)
                                reports.append(dict(dataset=d,aspect=a,comparator=comp,metric=m,group=group,**report))
        summary={}
        for family in sorted({r['cell_type'] for r in self.io.replays}):
            rows=[r for r in self.io.replays if r['cell_type']==family]
            diffs=[r['absolute_difference'] for r in rows if r['absolute_difference'] is not None]
            summary[family]=dict(n=len(rows),passed=all(r['passed'] for r in rows),largest_absolute_difference=max(diffs,default=None))
        full=not self.args.query_limit and not self.args.quick and self.args.seeds==SEEDS
        return self.io.envelope(mode='frozen_replay',reports=reports,summary=summary,
            scope=dict(comparators=list(self.args.comparators),piece=self.args.piece,dataset=self.args.dataset,aspects=list(self.args.aspects),metric=self.args.metric,group=self.args.replay_group,
                       seeds=list(self.args.seeds),clu_draws=self.args.clu_draws,ret_draws=self.args.ret_draws,full_queries=not self.args.query_limit,
                       full_protocol=full and self.args.clu_draws==2000 and self.args.ret_draws==10000))

    def replay_ami(self,d):
        reports=[]
        for a in selected_aspects(self.args,d):
            y=self.data.frame(d,'test')['labels'][a]
            cand=self.arm_parts(d,'test','candidate',a,True)
            encoder='qwen' if d=='emoset' else 'siglip1'
            # r17 SC4 EmoSet only made a Qwen candidate cell; replay that candidate too.
            if d=='emoset':
                siglip_value=ami(cand,y)
                p=TEST/'20261112_siglip_main/records/photos.json'
                target=self.io.json(p)['cells']['a']['ami']
                if self.args.seeds==SEEDS:
                    self.io.compare('emoset/ami',p,'cells.a.ami.mean',siglip_value,target['mean'])
                for i,s in enumerate(self.args.seeds):
                    self.io.compare('emoset/ami',p,f'cells.a.ami.per_seed.{s}',ami(cand[i],y),target['per_seed'][s])
                cand=self.arm_parts(d,'test','candidate',a,True,encoder='qwen')
            for comp in self.args.comparators:
                other=self.arm_parts(d,'test','qwen_plain' if d=='emoset' and comp=='p1' else comp,a)
                stats=ami_interval(cand,other,y,self.args.clu_draws,self.args.workers) if len(cand)==5 else dict(value=ami(cand,y),comparator_value=ami(other,y))
                sc='SC1' if d=='mmart' and comp=='p1' else 'SC3' if d=='mmart' else 'SC4'
                target=self.ns_cell(sc,d,a,'plain' if comp=='p1' else 'teacher',encoder=encoder)
                path=TEST/'20261103_new_scores/records/verdicts.json'
                if len(cand)==5:
                    self.compare_stats(d+'/ami',path,target['id'],stats,target,self.args.clu_draws==2000)
                else:
                    self.io.compare(d+'/ami',path,target['id']+'.comp',stats['comparator_value'],target['comp'],1e-9)
                reports.append(dict(dataset=d,aspect=a,group='ami',comparator=comp,stats=stats))
            if d=='mmart' and self.args.seeds==SEEDS:
                path=TEST/'20261112_siglip_main/records/mmart.json'
                target=self.io.json(path)['siglip1']['cstar_ami']['values'][ASPECTS.index(a)]
                self.io.compare('mmart/ami',path,f'siglip1.cstar_ami.values.{ASPECTS.index(a)}',ami(cand,y),target)
        return reports

    def replay_map(self,d):
        reports=[]
        for a in selected_aspects(self.args,d):
            y=self.data.frame(d,'test')['labels'][a]
            if d=='mmart':
                pathsides=[('p1','untold', self.io.json(TEST/'20261103_new_scores/records/stage1_picks.json')['untold/mmart/siglip1']['pick']),
                           ('q1','candidate',self.io.json(R15/'stage1/picks_siglip1.json')['arms']['cstar']['pick'])]
            else:
                pick=(self.io.json(TEST/'20261102_backbones_sweep/records/stage1_siglip1.json')['affection']['photo_readout']['pick'] if d=='affection' else
                      self.io.json(TEST/'20261112_siglip_main/records/photos.json')['cells']['a']['readout'])
                pathsides=[(comp,'candidate',pick) for comp in self.args.comparators]
            for comp,arm,pick in pathsides:
                if comp not in self.args.comparators:
                    continue
                encoder='qwen' if d=='emoset' and comp=='q1' else 'siglip1'
                if encoder=='qwen':
                    pick=self.io.json(TEST/'20261102_backbones_sweep/records/stage2_emoset.json')['readout']['photo_qwen']['pick']
                cand=np.stack([self.ap(d,'test',arm,a,s,True,pick,encoder) for s in self.args.seeds])
                other,rebuild=self.rebuild(d,a,comp,'map')
                full=not self.args.query_limit and len(cand)==5
                stats=ret_interval(cand,other,self.args.ret_draws) if full else dict(value=float(cand.mean()),comparator_value=float(other.mean()))
                if full:
                    if d=='mmart' and comp=='p1':
                        target=self.ns_cell('SC7',d,a,'plain','map')
                        path=TEST/'20261103_new_scores/records/verdicts.json'
                        self.compare_stats(d+'/map',path,target['id'],stats,target,self.args.ret_draws==10000)
                    elif d=='affection':
                        path,target=self.r14_cell('SC5' if comp=='p1' else 'SC6',d)
                        self.compare_stats(d+'/map',path,f'{target["sc"]}/siglip1',stats,target,self.args.ret_draws==10000)
                    elif d=='emoset' and comp=='q1':
                        path,target=self.r14_cell('SC10',d,encoder='qwen')
                        self.compare_stats(d+'/map',path,'SC10/qwen',stats,target,self.args.ret_draws==10000)
                    if d=='mmart' and comp=='q1':
                        rp=R15/'stage2/cells_ret.json'
                        rt=[r for r in self.io.json(rp) if r['encoder']=='siglip1' and r['aspect']==a and r['sc']=='SC4']
                        assert len(rt)==1
                        self.compare_stats(d+'/map',rp,f'SC4/siglip1/{a}',stats,rt[0],self.args.ret_draws==10000)
                        path=TEST/'20261112_siglip_main/records/mmart.json'
                        target=self.io.json(path)['siglip1']['cstar_map']['values'][ASPECTS.index(a)]
                        self.io.compare(d+'/map',path,f'siglip1.cstar_map.values.{ASPECTS.index(a)}',stats['value'],target)
                    if d=='emoset' and comp=='p1':
                        path=TEST/'20261112_siglip_main/records/photos.json'
                        target=self.io.json(path)['cells']['a']['map']
                        self.io.compare(d+'/map',path,'cells.a.map.mean',stats['value'],target['mean'])
                        for i,s in enumerate(self.args.seeds):
                            self.io.compare(d+'/map',path,f'cells.a.map.per_seed.{s}',float(cand[i].mean()),target['per_seed'][s])
                if d=='emoset' and comp=='p1' and not self.args.query_limit and not full:
                    path=TEST/'20261112_siglip_main/records/photos.json'
                    target=self.io.json(path)['cells']['a']['map']
                    for i,s in enumerate(self.args.seeds):
                        self.io.compare(d+'/map/seed',path,f'cells.a.map.per_seed.{s}',float(cand[i].mean()),target['per_seed'][s])
                reports.append(dict(dataset=d,aspect=a,group='map',comparator=comp,pick=pick,stats=stats,rebuild=rebuild))
        return reports

    def replay_validation(self,d):
        reports=[]
        aspects=selected_aspects(self.args,d)
        ranges={a:self.range(d,a,True) for a in aspects}
        path=TEST/'20261103_new_scores/records/stage1_ranges.json'
        doc=self.io.json(path)
        for a in aspects:
            if 'ami' in ranges[a]:
                key=f'{"cstar" if d=="mmart" else "photo"}/{d}/siglip1/{a}/ami'
                # r17 did not train/check a SigLIP photo model on EmoSet.
                if key in doc:
                    for s,v in ranges[a]['ami']['per_seed'].items():
                        self.io.compare(d+'/validation/ami',path,key+f'.per_seed.{s}',v,doc[key]['per_seed'][s],1e-6)
                    self.io.compare(d+'/validation/ami',path,key+'.r',ranges[a]['ami']['r'],doc[key]['r'],1e-6)
            if 'map' in ranges[a] and not self.args.query_limit:
                if d=='mmart':
                    p=R15/'stage1/ret_ranges_siglip1.json'
                    t=self.io.json(p)
                    for s,v in ranges[a]['map']['per_seed'].items():
                        self.io.compare(d+'/validation/map',p,f'per_seed.{a}.{s}',v,t['per_seed'][a][s],1e-6)
                    self.io.compare(d+'/validation/map',p,f'ranges.{a}',ranges[a]['map']['r'],t['ranges'][a],1e-6)
                elif d=='affection':
                    p=TEST/'20261102_backbones_sweep/records/stage1_siglip1.json'
                    t=self.io.json(p)['affection']['ranges_doc']['map']
                    for s,v in ranges[a]['map']['per_seed'].items():
                        self.io.compare(d+'/validation/map',p,f'affection.ranges_doc.map.per_seed.{s}',v,t['per_seed'][int(s)],1e-6)
        if d=='mmart' and self.args.metric in ('all','ami'):
            rp=R15/'stage1/clu_ranges_siglip1.json'
            target=self.io.json(rp)
            for aspect in aspects:
                y=self.data.frame(d,'val')['labels'][aspect]
                vals=[]
                for seed in self.args.seeds:
                    value=ami(self.parts(d,'val','candidate',aspect,seed,True,20),y)
                    vals.append(value)
                    self.io.compare(d+'/validation/ami',rp,f'per_seed.{aspect}.{seed}',value,target['per_seed'][aspect][str(seed)],1e-6)
                if self.args.seeds==SEEDS:
                    self.io.compare(d+'/validation/ami',rp,f'ranges.{aspect}',float(np.ptp(vals)),target['ranges'][aspect],1e-6)
        if d=='emoset' and self.args.metric in ('all','ami'):
            qp=self.range(d,'emotion',True,encoder='qwen')['ami']
            qkey='photo/emoset/qwen/emotion/ami'
            for seed,value in qp['per_seed'].items():
                self.io.compare(d+'/validation/ami',path,qkey+'.per_seed.'+seed,value,doc[qkey]['per_seed'][seed],1e-6)
            if self.args.seeds==SEEDS:
                self.io.compare(d+'/validation/ami',path,qkey+'.r',qp['r'],doc[qkey]['r'],1e-6)
        if self.args.metric in ('all','map'):
            units={a:ranges[a]['map']['r'] for a in aspects}
            if d=='emoset':
                units={'emotion':self.io.json(TEST/'20261102_backbones_sweep/records/stage2_emoset.json')['ranges']['map']['r']}
            for arm in (('untold','candidate','p1','q1') if d=='mmart' else ('candidate',)):
                arm_units=units
                if d=='mmart' and arm=='q1':
                    frozen_qwen=self.io.json(R15/'stage1/ret_ranges_qwen.json')['ranges']
                    arm_units={a:frozen_qwen[a] for a in aspects}
                pick=self.pick(d,arm,arm_units,True)
                if d=='mmart' and arm=='untold':
                    p=TEST/'20261103_new_scores/records/stage1_picks.json'
                    t=self.io.json(p)['untold/mmart/siglip1']; key='untold/mmart/siglip1'
                elif d=='mmart' and arm=='q1':
                    p=R15/'stage1/picks_teacher.json';t=self.io.json(p)['arms']['teacher'];key='arms.teacher'
                elif d=='mmart':
                    label='frozen' if arm=='p1' else 'cstar'
                    p=R15/'stage1/picks_siglip1.json';t=self.io.json(p)['arms'][label];key='arms.'+label
                elif d=='affection':
                    p=TEST/'20261102_backbones_sweep/records/stage1_siglip1.json';t=self.io.json(p)['affection']['photo_readout'];key='affection.photo_readout'
                else:
                    p=TEST/'20261112_siglip_main/records/photos.json';t=self.io.json(p)['cells']['a'];key='cells.a'
                if len(aspects)==len(COUNTS[d]) and not self.args.quick and not self.args.query_limit and self.args.seeds==SEEDS:
                    self.io.compare(d+'/validation/pick',p,key+'.pick',pick['pick'],t.get('pick',t.get('readout')),1e-6)
                    table=t.get('table',t.get('scores'))
                    if d=='mmart' and arm=='untold':
                        table=self.io.json(R17/t['table_path'])['table']
                    for r,v in pick['table'].items():
                        for a,val in v['map'].items():
                            recorded=t['val_scores'][r] if d=='emoset' else table[r]['map'][a]
                            self.io.compare(d+'/validation/map',p,key+'.table.'+r+'.'+a,val,recorded,1e-6)
                reports.append(dict(dataset=d,group='validation',arm=arm,pick=pick,ranges=ranges))
        else:
            reports.append(dict(dataset=d,group='validation',ranges=ranges))
        return reports

    def held(self):
        a=self.args
        if not a.picks_committed or not a.val_sha256:
            raise ValueError('held requires controller --picks-committed and --val-sha256 after validation picks commit')
        if a.comparators!=('p1','q1') or not full_validation(a) or a.clu_draws!=2000 or a.ret_draws!=10000:
            raise ValueError('held requires full seeds, aspects, queries, menu, and protocol draw counts')
        val=self.io.json(a.val_json)
        assert self.io.file(a.val_json)==a.val_sha256, 'validation digest differs from committed picks'
        assert val['mode']=='validation'
        proof=[]
        for p in a.replay_json:
            proof.append(self.io.json(p))
        result={}
        for piece in pieces(a):
            n,threshold=(6,5) if piece=='mmart' else (4,3)
            cells=[]
            committed=val.get(piece,{})
            for d in (('mmart',) if piece=='mmart' else ('affection','emoset')):
                for aspect in COUNTS[d]:
                    for metric in ('ami','map'):
                        for comp in ('p1','q1'):
                            cid=f'mmart/{metric}/{aspect}/{comp}' if d=='mmart' else f'{d}/{metric}/{comp}'
                            row=dict(id=cid,outcome='INCONCLUSIVE',value=None,comparator_value=None,d=None,lo=None,hi=None,r=None,n=None)
                            reason=None
                            if not committed.get('complete'):
                                reason='missing or incomplete committed validation picks/ranges'
                            elif not replay_pass(proof,d,metric):
                                reason=f'missing or failed complete frozen replay for {d}/{metric}'
                            if reason:
                                cells.append(dict(**row,reason=reason));continue
                            try:
                                ranges=committed['ranges'][aspect] if d=='mmart' else committed['ranges'][d]
                                r=ranges[metric]['r']
                                if metric=='ami':
                                    y=self.data.frame(d,'test')['labels'][aspect]
                                    cand=self.arm_parts(d,'test','candidate',aspect)
                                    other,rebuilt=self.rebuild(d,aspect,comp,'ami')
                                    stats=ami_interval(cand,other,y,a.clu_draws,a.workers)
                                    nrows=len(y)
                                else:
                                    pick=committed['picks'][comp if d=='mmart' else d]['pick']
                                    arm='untold' if d=='mmart' and comp=='p1' else 'candidate'
                                    cand=np.stack([self.ap(d,'test',arm,aspect,s,pick=pick) for s in SEEDS])
                                    other,rebuilt=self.rebuild(d,aspect,comp,'map')
                                    stats=ret_interval(cand,other,a.ret_draws)
                                    nrows=len(other)
                                row.update(**stats,r=r,n=nrows,n_draws=a.clu_draws if metric=='ami' else a.ret_draws,
                                           comparator_rebuild=rebuilt, outcome=outcome(stats['lo'],r) if rebuilt['passed'] else 'INCONCLUSIVE')
                                if not rebuilt['passed']:
                                    row['reason']='comparator rebuilt point differs from frozen record by more than 1e-9'
                                cells.append(row)
                            except FileNotFoundError as e:
                                cells.append(dict(**row,reason=f'missing input: {e.filename}'))
            result[piece]=dict(cells=cells,**{'sc5' if piece=='mmart' else 'sc6':ladder(cells,n,threshold)})
        return self.io.envelope(mode='held',validation_sha256=a.val_sha256,picks_committed_attestation=True,**result)


def replay_pass(proofs,d,metric):
    """Fail closed unless complete old-data evidence covers scoring and rebuild.

    A separate AMI/validation replay and rebuild report can be supplied. Evidence
    files must be generated by exactly the present source, never a candidate run.
    """
    # Union complete dataset/group fragments; every compared field must pass.
    rows=[]
    for doc in proofs:
        if doc.get('mode')!='frozen_replay' or not doc.get('scope',{}).get('full_protocol'):
            continue
        if doc.get('inputs',{}).get(str(Path(__file__).resolve())) != sha(__file__):
            continue
        if doc['scope'].get('aspects') != list(ASPECTS):
            continue
        rows.extend(doc.get('replays',[]))
    def group(name,minimum):
        selected=[r for r in rows if r['cell_type']==name]
        distinct={(r['target_record'],r['key']) for r in selected}
        return len(distinct)>=minimum and all(r['passed'] for r in selected)
    core_counts={('mmart','ami'):36,('mmart','map'):30,
                 ('affection','ami'):12,('affection','map'):10,
                 ('emoset','ami'):18,('emoset','map'):11}
    core=group(d+'/'+metric,core_counts[d,metric])
    rebuilt=group(d+'/'+metric+'/rebuild',2*len(COUNTS[d]))
    validation=group(d+'/validation/'+metric,6*len(COUNTS[d]) if metric=='ami' else 25*len(COUNTS[d]))
    picked=metric=='ami' or group(d+'/validation/pick',4 if d=='mmart' else 1)
    return core and rebuilt and validation and picked


def pieces(a):
    return ('mmart','photo') if a.piece=='all' else (a.piece,)


def datasets(a):
    available=('mmart','affection','emoset') if a.piece=='all' else ('mmart',) if a.piece=='mmart' else ('affection','emoset')
    return available if a.dataset=='all' else tuple(d for d in available if d==a.dataset)


def selected_aspects(a,d):
    return tuple(x for x in a.aspects if x in COUNTS[d])


def full_validation(a):
    return a.seeds==SEEDS and a.aspects==ASPECTS and not a.query_limit and not a.quick and a.metric=='all'


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('val','held','replay'))
    p.add_argument('--piece',choices=('all','mmart','photo'),default='all')
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--out',type=Path)
    p.add_argument('--dataset',choices=('all','mmart','affection','emoset'),default='all',help='replay scope only')
    p.add_argument('--metric',choices=('all','ami','map'),default='all',help='replay scope; partial val never complete')
    p.add_argument('--replay-group',choices=('all','validation','ami','map','rebuild'),default='all')
    p.add_argument('--seeds',type=int,nargs='+',default=list(SEEDS))
    p.add_argument('--aspects',choices=ASPECTS,nargs='+',default=list(ASPECTS))
    p.add_argument('--comparators',choices=('p1','q1'),nargs='+',default=['p1','q1'],help='frozen replay scope only')
    p.add_argument('--query-limit',type=int,default=0,help='diagnostic queries only; full gallery retained')
    p.add_argument('--quick',action='store_true',help='diagnostic reduced menu; never committed picks')
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--clu-draws',type=int,default=2000)
    p.add_argument('--ret-draws',type=int,default=10000)
    p.add_argument('--workers',type=int,choices=range(1,9),default=8)
    p.add_argument('--forward-check',action='store_true')
    p.add_argument('--val-json',type=Path,default=HERE/'s2mp_val.json')
    p.add_argument('--picks-committed',action='store_true',help='controller attests picks committed before any piece held number')
    p.add_argument('--val-sha256',help='digest of the independently committed validation record')
    p.add_argument('--replay-json',type=Path,nargs='*',default=[])
    a=p.parse_args(argv)
    a.seeds=tuple(a.seeds);a.aspects=tuple(a.aspects);a.comparators=tuple(a.comparators)
    assert a.seeds and len(set(a.seeds))==len(a.seeds) and set(a.seeds)<=set(SEEDS)
    assert a.query_limit>=0 and a.batch_size>0 and a.clu_draws>0 and a.ret_draws>0
    if a.command=='held' and (not a.picks_committed or not a.val_sha256):
        p.error('held is controller-only after picks commit; supply --picks-committed --val-sha256 DIGEST')
    if a.command=='val' and (a.dataset!='all' or a.metric!='all'):
        p.error('validation selects full picks/ranges; use replay for metric/dataset subsets')
    if a.device=='cuda':
        import torch
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        assert torch.cuda.is_available()
    with threadpool_limits(8):
        work=Work(a)
        result={'val':work.validation,'held':work.held,'replay':work.replay}[a.command]()
    out=a.out or HERE/('s2mp_val.json' if a.command=='val' else 's2mp_held.json' if a.command=='held' else 'r29rd_s2mp_replay.json')
    # Piece-only runs safely merge only their completed piece into the requested common record.
    if a.command in ('val','held') and a.piece!='all' and out.exists():
        previous=work.io.json(out)
        result['inputs'][str(out.resolve())]=work.io.file(out)
        other='photo' if a.piece=='mmart' else 'mmart'
        if other in previous and previous.get('mode')==result['mode']:
            result[other]=previous[other]
            result['inputs']={**previous.get('inputs',{}),**result['inputs']}
            result['replays']=previous.get('replays',[])+result['replays']
            if a.command=='val':
                result['complete']=all(result.get(x,{}).get('complete',False) for x in ('mmart','photo'))
    write(out,result)
    print(json.dumps(dict(output=str(out),seconds=round(result['seconds'],2),summary=result.get('summary'),complete=result.get('complete')),allow_nan=False))


if __name__=='__main__':
    main()
