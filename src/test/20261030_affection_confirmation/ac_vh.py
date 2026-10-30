"""Independent C13 check, derived from r12 spec §§3.3–3.5/6 and inherited r11/r10 rules.

No round ac_* module is imported, and round result JSON is opened only by compare_final,
after all independent calculations. Frozen r10 label/split helpers are inputs, not statistics.

Readings where the text/interface leaves a detail open:
* The Affection one-aspect read-out inherits r11's guard (gain >= -r).
* CRL's 'within 0.001 r' means <= .001 in the mean normalized score units,
  including ArtELingo's three different r_a. Wording's > .001*r is strict.
* A zero range makes normalized selection undefined: stop; never invent an epsilon.
* Smoke takes the first N sorted keys and retains the same read-out rules/k20 (default N=2100).
  Draw counts are explicit flags, defaulting to 2000/10000 even in smoke. It may
  slice full exports or use smoke exports; a full run rejects smoke exports.
* Retrieval input/W/CRL arithmetic follows the frozen r11 log's numeric convention. Covariance
  uses n-1, no eigenvalue clipping; zero vector norms are floored at 1e-12.
  c* clustering instead casts stored float16 to float32 and checks the unit norm
  within 1e-3 without repairing it, as frozen r10's r20_partition does.
* Export keys_sha256 has no serialization in the index. Accept SHA256 of exactly
  the expected ordered list encoded as default JSON, compact JSON, or newline-joined
  UTF-8 (with/without a final newline); never accept a row count as an identity check.
* Labels are checked against §3.4 on full runs. Gate entries named label_counts_{split}
  and label_ties_{split} are compared when present; the index does not name them.
* ArtELingo fixed picks remain fixed; only CLIP CRL's R0 and fixed-pick cross-fit scores are remeasured.
  ArtELingo CRL arm IDs use the same crl_<version> names as Affection.
* Clustering SC rows use null readout_c/readout_t: r20 partitions have no retrieval
  read-out. Extra descriptive output fields are ignored, but picks and SC IDs are exact.

CLI: --stage {0b,1,2,3} --outputs DIR [--smoke [N]] [--views-root DIR] [--draws-clu N] [--draws-ret N]. In smoke the
committed records are read from <outputs>/records/ (never the round folder). Missing input: exit 2;
comparison mismatch: exit 1. Every compared leaf is saved to outputs/vh/<stage>.json.
Real runs can be expensive; this implementation does no GPU work.
"""
import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from functools import cached_property
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_mutual_info_score
from threadpoolctl import threadpool_limits

MAIN = Path('/project/CoSiR')
ROUND = Path(__file__).resolve().parent
AFF = Path('/data/SSD2/pre_extract/affection_local')
ANNOTATIONS = Path('/data/SSD/affection/language/version_0/affection_preprocessed_public_version_0.csv')
DEFAULT_OUTPUTS = Path('/data/SSD2/cosir_r12/20261030_affection_confirmation/outputs')
R11 = Path('/data/SSD2/cosir_r11/20261029_retrieval/outputs')
R10 = Path('/data/SSD2/cosir_r10/20261027_recipe_sweep/outputs/full')
ART = Path('/data/SSD2/pre_extract/artelingo_full')
EMOTIONS = ('amusement', 'anger', 'awe', 'contentment', 'disgust', 'excitement', 'fear',
            'sadness', 'something else')
ASPECTS = ('emotion', 'style', 'genre')
SEEDS = tuple(range(5))
VERSIONS = ('clip_painting', 'clip_photo', 'qwen_painting', 'qwen_photo')
AFF_ARMS = ('cstar', 'plain', 'teacher_artwork', 'teacher_photo', 'clip', 'percept') + tuple(
    'crl_' + v for v in VERSIONS)
ART_ARMS = ('cstar', 'crl_clip_painting', 'crl_qwen_painting')
R0 = ('none', 0, 0)
EXPECTED_COUNTS = {'val': [798, 54, 622, 1387, 189, 620, 203, 270, 108],
                   'test': [1651, 81, 1460, 2626, 351, 1185, 448, 501, 198]}
EXPECTED_TIES = {'val': 907, 'test': 1917}
SKIPPED = 'skipped in smoke'                          # the runner's marker for a full-split gate item in smoke
DEFAULT_INSTRUCTION = "Represent the user's input."  # frozen pav_prompts.PROMPTS['default']
TEXT_MODELS = {'clip': 'openai/clip-vit-large-patch14', 'qwen': 'Qwen/Qwen3-VL-Embedding-2B'}
VIEW_INSTRUCTIONS = {
    # r10's frozen pav_prompts.PROMPTS['emotion_focus']; r12 spec §3.2 photo text.
    'teacher_artwork': 'Represent the emotion that this artwork, or this description of an artwork, conveys. Focus only on the emotion.',
    'teacher_photo': 'Represent the emotion that this photo, or this description of a photo, conveys. Focus only on the emotion.',
    'clip': None,
}


def read_json(path):
    return json.loads(Path(path).read_text())


def unit(x, dtype=None):
    x = np.asarray(x, dtype=dtype)
    assert x.ndim == 2 and np.isfinite(x).all()
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def annotations(path):
    """Read evaluation labels only. Ignored captions contribute neither votes nor splits."""
    counts, splits = {}, {}
    with Path(path).open(newline='') as f:
        for row in csv.DictReader(f):
            sp = row['split']
            if sp == 'ignored':
                continue
            key = row['dataset'] + '/' + row['image_name']
            assert sp in ('train', 'val', 'test'), sp
            assert key not in splits or splits[key] == sp, f'multiple splits: {key}'
            splits[key] = sp
            if sp == 'train':
                continue
            assert row['emotion'] in EMOTIONS, row['emotion']
            counts.setdefault(key, np.zeros(9, dtype=np.int64))[EMOTIONS.index(row['emotion'])] += 1
    result = {}
    for sp in ('val', 'test'):
        keys = sorted(k for k in counts if splits[k] == sp)
        votes = np.stack([counts[k] for k in keys]) if keys else np.empty((0, 9), dtype=np.int64)
        result[sp] = dict(keys=keys, labels=votes.argmax(axis=1),
                          ties=(votes == votes.max(axis=1, keepdims=True)).sum(axis=1) > 1,
                          dist=votes / votes.sum(axis=1, keepdims=True))
    return result


def label_summary(records):
    return {sp: dict(counts=np.bincount(r['labels'], minlength=9).tolist(),
                     ties=int(r['ties'].sum()), n=len(r['keys'])) for sp, r in records.items()}


def key_hashes(keys):
    strings = (json.dumps(keys), json.dumps(keys, separators=(',', ':')),
               '\n'.join(keys), '\n'.join(keys) + '\n')
    return {hashlib.sha256(s.encode('utf-8')).hexdigest() for s in strings}


def joined_rows(x, keys, wanted):
    assert x.ndim == 2 and len(x) == len(keys) and len(set(keys)) == len(keys)
    ix = {k: i for i, k in enumerate(keys)}
    assert all(k in ix for k in wanted), 'missing key at array boundary'
    return np.asarray(x[[ix[k] for k in wanted]])


def crl_vectors(image, text):
    assert text.dtype == np.float32 and text.ndim == 2 and text.shape[1] == image.shape[1]
    assert np.isfinite(text).all() and np.max(np.abs(np.linalg.norm(text, axis=1) - 1)) < 1e-3
    return unit(image.astype(np.float64) @ text.astype(np.float64).T).astype(np.float32)


class Data:
    """Independent loaders. No result, pick, per-query or partition file from this round is an input."""
    def __init__(self, outputs, smoke=0, aff=AFF, annotation=ANNOTATIONS, views_root=None):
        self.outputs, self.smoke, self.aff = Path(outputs), smoke, Path(aff)
        # The three view folders (qwen3vl_emb_2b_aspect_prompts/*, clip_l14_valtest) are read under views_root;
        # a smoke run points it at <outputs>/smoke_views. The default store always stays under aff.
        self.views_root = Path(views_root) if views_root else self.aff
        self.frozen_crl_aspects = set()
        self.full = annotations(annotation)
        self.records = {sp: {k: v[:smoke] if smoke else v for k, v in r.items()}
                        for sp, r in self.full.items()}
        if not smoke:
            for sp, r in self.full.items():
                assert len(r['keys']) == sum(EXPECTED_COUNTS[sp]), (sp, len(r['keys']))

    def keys(self, dataset, split):
        return self.records[split]['keys'] if dataset == 'affection' else self.art_records[split]['keys']

    def labels(self, dataset, split, aspect):
        return (self.records[split]['labels'] if dataset == 'affection'
                else self.art_records[split]['labels'][aspect])

    def dist(self, dataset, split):
        return self.records[split]['dist'] if dataset == 'affection' else self.art_records[split]['dist']

    @cached_property
    def frozen_art_names(self):
        sys.path.insert(0, str(MAIN / 'src/test/20261027_recipe_sweep'))
        import ct2_data
        split = ct2_data.split2('full')
        val = [str(x) for x in split['all_painting_ids'][split['dev']]]
        held = read_json(MAIN / 'src/test/20261020_claim_test/outputs/held/held_paintings.json')['names']
        assert len(val) == 6152 and len(held) == 12281
        return dict(val=val, held=held)

    @cached_property
    def art_records(self):
        # Only frozen r10 domain-label and split functions are reused, as allowed by C13.
        sys.path.insert(0, str(MAIN / 'src/test/20261027_recipe_sweep'))
        import rs_vh
        raw = read_json(rs_vh.ANNOTATIONS)
        result = {}
        for sp, names in self.frozen_art_names.items():
            assert names == sorted(names) and len(set(names)) == len(names)
            if self.smoke:
                names = names[:self.smoke]
            votes = {n: np.zeros(9, dtype=np.int64) for n in names}
            for row in raw:
                if row['painting'] in votes:
                    votes[row['painting']][EMOTIONS.index(row['emotion'])] += 1
            v = np.stack([votes[n] for n in names])
            result[sp] = dict(keys=names, dist=v / v.sum(axis=1, keepdims=True), labels={
                'emotion': v.argmax(axis=1), 'style': rs_vh.style_labels(names, raw),
                'genre': rs_vh.genre_labels(names, raw, rs_vh.GENRE_CSVS)})
        return result

    def _view(self, folder, wanted, instruction=None):
        meta = read_json(folder / 'meta.json')
        assert meta['instruction'] == instruction, 'view instruction'
        assert meta['gate']['ok'] is True, 'view extraction gate'
        assert self.smoke or meta['smoke'] is False, 'full run cannot use a smoke view'
        keys = read_json(folder / 'keys.json')
        x = np.load(folder / 'img.npy', mmap_mode='r', allow_pickle=False)
        assert x.dtype == np.float16
        return joined_rows(x, keys, wanted)

    def _export(self, arm, split, seed, aspect):
        stem = (self.outputs / 'exports' / arm / split /
                (f'{aspect}__s{seed}' if arm == 'cstar' else f'P3__s{seed}'))
        meta = read_json(stem.with_suffix('.json'))
        # Export arrays carry only this key digest, so verify the entire list before smoke slicing.
        keys = self.full[split]['keys']
        assert meta['split'] == split and meta['seed'] == seed
        assert self.smoke or not meta['smoke'], 'full run cannot use a smoke export'
        if arm == 'cstar':
            assert meta['aspect'] == aspect
        n = len(self.records[split]['keys'])
        if meta['keys_sha256'] in key_hashes(keys):
            expected = len(keys)
        else:
            assert self.smoke and meta['keys_sha256'] in key_hashes(keys[:n]), 'export key digest mismatch'
            expected = n
        with np.load(stem.with_suffix('.npz'), allow_pickle=False) as z:
            x = z['img' if arm == 'cstar' else 'prob']
        assert x.shape == (expected, 2048 if arm == 'cstar' else 40)
        assert x.dtype == (np.float16 if arm == 'cstar' else np.float32)
        assert hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest() == meta['array_sha256']
        return joined_rows(x, keys[:expected], self.records[split]['keys'])

    def cluster_cstar(self, split, seed):
        """r10's clustering input: cast only; preserve stored float16 norm deviations."""
        x = self._export('cstar', split, seed, 'emotion').astype(np.float32)
        assert np.isfinite(x).all()
        assert np.max(np.abs(np.linalg.norm(x, axis=1) - 1)) < 1e-3, 'cstar rows not unit norm'
        return x

    def embed(self, dataset, arm, split, aspect='emotion', seed=None):
        assert (seed in SEEDS) if arm in ('cstar', 'percept') else seed is None
        if arm.startswith('crl_'):
            encoder, wording = arm[4:].split('_')
            values = read_json(R11 / 'crl/lists.json')['aspects'][aspect]['values']
            assert encoder in TEXT_MODELS and wording in ('painting', 'photo')
            if dataset == 'artelingo' and (encoder, wording) == ('clip', 'painting'):
                # r11's CLIP CRL arm (spec 3.2): r11's own texts and text vectors, so SC5–SC7 must not inherit a
                # text-vector error from this round. The round writes a local copy for emotion only (Affection's arm).
                self.frozen_crl_aspects.add(aspect)
                side = read_json(R11 / 'crl' / f'{aspect}_text.json')
                assert side['model'] == TEXT_MODELS[encoder], 'CRL text model'
                path = R11 / 'crl' / f'{aspect}_text.npy'
            else:
                side = read_json(self.outputs / 'crl' / f'{encoder}_{wording}_{aspect}.json')
                assert side['encoder'] == encoder, 'CRL encoder'
                assert side['model_id'] == TEXT_MODELS[encoder], 'CRL text model'
                instruction = DEFAULT_INSTRUCTION if encoder == 'qwen' else None
                assert side['instruction'] == instruction, 'CRL text instruction'
                path = self.outputs / 'crl' / f'{encoder}_{wording}_{aspect}.npy'
            assert side['texts'] == [f'a {wording} with the {aspect} of {v}' for v in values], 'CRL texts/order'
            image = self.embed(dataset, 'clip' if encoder == 'clip' else 'plain', split, aspect)
            text = np.load(path, allow_pickle=False)
            assert text.ndim == 2 and len(text) == len(values), 'CRL text/value count'
            return crl_vectors(image, text)
        if dataset == 'artelingo':
            return self._art_embed(arm, split, aspect, seed)
        assert dataset == 'affection' and split in ('val', 'test')
        wanted = self.keys(dataset, split)
        if arm in ('cstar', 'percept'):
            x = self._export(arm, split, seed, aspect)
        elif arm == 'plain':
            folder = self.aff / 'qwen3vl_emb_2b'
            images = read_json(folder / 'index.json')['images']
            keys = [r['source'] + '/' + r['image_name'] for r in images]
            x = np.load(folder / 'img.npy', mmap_mode='r', allow_pickle=False)
            assert x.dtype == np.float16 and x.shape[1] == 2048
            x = joined_rows(x, keys, wanted)
        else:
            folders = {'teacher_artwork': self.views_root / 'qwen3vl_emb_2b_aspect_prompts/emotion_focus',
                       'teacher_photo': self.views_root / 'qwen3vl_emb_2b_aspect_prompts/emotion_focus_photo',
                       'clip': self.views_root / 'clip_l14_valtest'}
            x = self._view(folders[arm], wanted, VIEW_INSTRUCTIONS[arm])
            assert x.shape[1] == (768 if arm == 'clip' else 2048)
        assert len(x) == len(wanted)
        return unit(x, np.float32)

    def _art_embed(self, arm, split, aspect, seed):
        wanted = self.keys('artelingo', split)
        if arm == 'cstar':
            files = {'emotion': 'emotion__Gres', 'style': 'style__Fres05', 'genre': 'genre__R0o'}
            path = (R10 / f'held/k11=40/{aspect}__s{seed}.npz' if split == 'held' else
                    R10 / f'train/k11=40/R6-5/{files[aspect]}__s{seed}.npz')
            with np.load(path, allow_pickle=False) as z:
                # Frozen r10 export row order is the frozen split list (r11 input contract).
                x = z['img']
            keys = self.frozen_art_names[split]
            assert x.shape == (len(keys), 2048) and x.dtype == np.float16
            x = joined_rows(x, keys, wanted)
        else:
            assert arm in ('plain', 'clip')
            folder = ART / ('clip_l14' if arm == 'clip' else 'qwen3vl_emb_2b')
            keys = [r['painting'] for r in read_json(folder / 'index.json')['images']]
            x = np.load(folder / 'img.npy', mmap_mode='r', allow_pickle=False)
            assert x.dtype == np.float16
            x = joined_rows(x, keys, wanted)
        return unit(x, np.float32)


def menu(width):
    ws = [('none', 0), ('center', 0)]
    ws += list(dict.fromkeys(('whiten', min(n, width)) for n in (64, 256, 1024)))
    return [(w, n, k) for w, n in ws for k in (0, 1, 3, 5, 10)]


def ro_name(ro):
    w, size, k = ro
    return f'{w}{size or ""}+k{k}'


def parse_ro(name):
    m = re.fullmatch(r'(none|center|whiten)(\d*)\+k(0|1|3|5|10)', name)
    assert m, f'invalid read-out: {name}'
    w, size, k = m.groups()
    assert (w == 'whiten') == bool(size)
    return w, int(size or 0), int(k)


def tie_key(ro):
    w, size, k = ro
    return ro != R0, k != 0, ('none', 'center', 'whiten').index(w), k, -size


def fit(x):
    x = x.astype(np.float64)
    mu = x.mean(axis=0)
    z = x - mu
    assert len(x) > 1
    lam, u = np.linalg.eigh(z.T @ z / (len(x) - 1))
    return mu, lam[::-1], u[:, ::-1]


def transform(x, ro, stats=None):
    w, size, _ = ro
    z = x.astype(np.float64)
    if w != 'none':
        mu, lam, u = stats
        z = z - mu
        if w == 'whiten':
            assert lam[0] > 0, 'constant fit set'
            z = (z @ u[:, :size]) / np.sqrt(lam[:size] + 1e-6 * lam[0])
    return unit(z).astype(np.float32)


def ranking(gallery, query, rows):
    sim = query @ gallery.T
    sim[np.arange(len(rows)), rows] = -np.inf
    order = np.argsort(-sim, axis=1, kind='stable')
    assert np.array_equal(order[:, -1], rows)
    return order[:, :-1]


def retrieval(x, labels, dist=None, ks=(0,)):
    """Full ranking AP, optional overlap nDCG. NaN marks a dropped singleton query.

    Unlabelled rows are excluded from both query and gallery; row IDs retain the
    original key order. Blocks bound memory, while the gallery always stays fixed.
    """
    rows = np.flatnonzero(labels >= 0)
    g = x[rows].astype(np.float64)
    lab = labels[rows]
    assert len(g) > max(ks) and len(g) > 1, 'gallery smaller than fixed k'
    result = {k: [] for k in ks}
    nd = []
    rank = np.arange(1, len(g), dtype=np.float64)
    discount = 1 / np.log2(rank + 1)
    for start in range(0, len(g), 64):
        qi = np.arange(start, min(start + 64, len(g)))
        order = ranking(g, g[qi], qi)
        for k in ks:
            o = order if k == 0 else ranking(g, unit(g[qi] + g[order[:, :k]].sum(axis=1)), qi)
            rel = lab[o] == lab[qi, None]
            nr = rel.sum(axis=1)
            num = (rel * np.cumsum(rel, axis=1) / rank).sum(axis=1)
            ap = np.divide(num, nr, out=np.full(len(qi), np.nan), where=nr > 0)
            result[k].append(ap)
        if dist is not None:
            p = dist[rows]
            gain = np.minimum(p[qi, None, :], p[None, :, :]).sum(axis=2)
            gain[np.arange(len(qi)), qi] = -np.inf
            ideal = -np.sort(-gain, axis=1)[:, :-1]
            denom = ideal @ discount
            assert (denom > 0).all()
            nd.append((np.take_along_axis(gain, order, axis=1) @ discount) / denom)
    result = {k: np.concatenate(v) for k, v in result.items()}
    return dict(ap=result, ndcg=np.concatenate(nd) if nd else None, rows=rows)


def partitions(x, k=20):
    assert x.dtype == np.float32 and np.isfinite(x).all()
    assert np.max(np.abs(np.linalg.norm(x, axis=1) - 1)) < 1e-3
    with threadpool_limits(limits=1):
        return np.stack([MiniBatchKMeans(n_clusters=k, n_init=3, batch_size=4096,
                                        random_state=s).fit(x).labels_ for s in range(3)]).astype(np.int32)


def ami(parts, labels):
    return float(np.mean([adjusted_mutual_info_score(labels, p) for p in parts]))


def paired_cluster(cand, comp, labels, draws=2000):
    """Fit partitions once outside. Each draw resamples images, then the shared seed pool."""
    c = cand if cand.ndim == 3 else cand[None]
    t = comp if comp.ndim == 3 else comp[None]
    assert c.shape[1:] == t.shape[1:] == (3, len(labels))
    assert len(c) in (1, 5) and len(t) in (1, 5)
    seeded = len(c) == 5 or len(t) == 5
    pc, pt = np.mean([ami(p, labels) for p in c]), np.mean([ami(p, labels) for p in t])
    rng, boot = np.random.default_rng(0), []
    for _ in range(draws):
        rows = rng.integers(0, len(labels), size=len(labels))
        seeds = rng.choice(SEEDS, size=5, replace=True) if seeded else [0]
        # Repeated seeds reuse their AMI; multiplicity is retained in the final mean.
        cv = {s: ami(c[s][:, rows], labels[rows]) for s in (set(seeds) if len(c) == 5 else [0])}
        tv = {s: ami(t[s][:, rows], labels[rows]) for s in (set(seeds) if len(t) == 5 else [0])}
        boot.append(np.mean([cv[s if len(c) == 5 else 0] - tv[s if len(t) == 5 else 0] for s in seeds]))
    return dict(point_c=float(pc), point_t=float(pt), d=float(pc - pt),
                interval=np.percentile(boot, [2.5, 97.5]).tolist())


def paired_retrieval(cand, comp, draws=10000):
    """Joint query/seed bootstrap, for AP or nDCG per-query arrays."""
    c, t = np.asarray(cand), np.asarray(comp)
    assert c.shape[-1] == t.shape[-1] and c.ndim in (1, 2) and t.ndim in (1, 2)
    assert (c.ndim != 2 or c.shape[0] == 5) and (t.ndim != 2 or t.shape[0] == 5)
    assert np.isfinite(c).all() and np.isfinite(t).all()
    diff = np.atleast_2d(c) - np.atleast_2d(t)
    rng, boot = np.random.default_rng(0), []
    for _ in range(draws):
        rows = rng.integers(0, diff.shape[1], size=diff.shape[1])
        if c.ndim == 2 or t.ndim == 2:
            seeds = rng.choice(SEEDS, size=5, replace=True)
            boot.append(diff[seeds][:, rows].mean())
        else:
            boot.append(diff[0, rows].mean())
    return dict(point_c=float(c.mean()), point_t=float(t.mean()), d=float(diff.mean()),
                interval=np.percentile(boot, [2.5, 97.5]).tolist())


def outcome(lower, r):
    assert np.isfinite(lower) and np.isfinite(r) and r >= 0
    return 'WIN' if lower > 0 else ('LEVEL' if lower > -r else 'BELOW')


def passes(sid, result):
    return result in ('WIN', 'LEVEL') if sid in ('SC3', 'SC4') else result == 'WIN'


def verdict(outcomes):
    p = {sid: passes(sid, outcomes[sid]) for sid in ('SC1', 'SC2', 'SC3', 'SC4')}
    if all(p.values()):
        return 'holds'
    if p['SC1'] and p['SC2'] and (p['SC3'] != p['SC4']):
        return 'qualified'
    return 'fails'


def repeats_line(sc3):
    return 'yes' if sc3 == 'WIN' else 'no'


def seed_range(values):
    v = np.asarray(values, dtype=np.float64)
    assert v.shape == (5,) and np.isfinite(v).all()
    return dict(per_seed=v.tolist(), r=float(v.max() - v.min()))


def select_readout(maps, ranges):
    """maps: {readout tuple: {aspect: cross-fit mAP}}; all quantities unrounded."""
    assert ranges and all(np.isfinite(r) and r > 0 for r in ranges.values()), 'zero/invalid selection range'
    base = maps[R0]
    table = {}
    for ro, values in maps.items():
        gains = {a: (values[a] - base[a]) / ranges[a] for a in ranges}
        table[ro_name(ro)] = dict(map=values, score=float(np.mean(list(gains.values()))))
    eligible = [ro for ro in maps if all(maps[ro][a] - base[a] >= -ranges[a] for a in ranges)]
    best = max(table[ro_name(ro)]['score'] for ro in eligible)
    pool = [ro for ro in eligible if best - table[ro_name(ro)]['score'] <= .001]
    return ro_name(min(pool, key=tie_key)), table


def teacher_pick(artwork, photo, r):
    assert r >= 0
    return 'photo' if photo - artwork > .001 * r else 'artwork'


def version_pick(scores):
    best = max(scores.values())
    return next(v for v in VERSIONS if v in scores and best - scores[v] <= .001)


class Run:
    def __init__(self, data, smoke=0, draws_clu=2000, draws_ret=10000):
        assert draws_clu > 0 and draws_ret > 0, 'bootstrap draws must be positive'
        self.data, self.smoke = data, smoke
        self.draws_clu, self.draws_ret = draws_clu, draws_ret
        self.cluster_cache, self.query_cache = {}, {}

    def parts(self, split, arm):
        key = split, arm
        if key not in self.cluster_cache:
            result = [partitions(self.data.cluster_cstar(split, s) if arm == 'cstar' else
                                 self.data.embed('affection', arm, split, seed=s))
                      for s in (SEEDS if arm in ('cstar', 'percept') else (None,))]
            self.cluster_cache[key] = np.stack(result) if arm in ('cstar', 'percept') else result[0]
        return self.cluster_cache[key]

    @cached_property
    def ranges(self):
        y = self.data.labels('affection', 'val', 'emotion')
        cv, av, nv = [], [], []
        for s, p in enumerate(self.parts('val', 'cstar')):
            cv.append(ami(p, y))
            x = self.data.embed('affection', 'cstar', 'val', seed=s)
            q = retrieval(x, y, self.data.dist('affection', 'val'))
            av.append(float(np.nanmean(q['ap'][0])))
            nv.append(float(q['ndcg'].mean()))
        return {metric: seed_range(v) for metric, v in (('clu', cv), ('map', av), ('ndcg', nv))}

    @cached_property
    def art_ranges(self):
        # Frozen r11 ranges are explicitly inherited inputs, never round r12 output.
        source = read_json(R11 / 'stage0a/ranges.json')
        return {a: source['map'][a]['r'] for a in ASPECTS}

    def crossfit(self, dataset, arm, aspects, ros=None):
        by = defaultdict(lambda: defaultdict(list))
        for aspect in aspects:
            for seed in (SEEDS if arm in ('cstar', 'percept') else (None,)):
                x = self.data.embed(dataset, arm, 'val', aspect, seed)
                y = self.data.labels(dataset, 'val', aspect)
                groups = defaultdict(list)
                for w, size, k in (menu(x.shape[1]) if ros is None else sorted(ros, key=tie_key)):
                    groups[w, size].append(k)
                for half in (0, 1):
                    score = np.arange(len(x)) % 2 == half
                    stats = fit(x[~score])
                    for (w, size), ks in groups.items():
                        z = transform(x[score], (w, size, 0), stats)
                        result = retrieval(z, y[score], ks=tuple(ks))
                        for k, ap in result['ap'].items():
                            assert np.isfinite(ap).any(), 'no cross-fit queries with relevant gallery rows'
                            by[(w, size, k)][aspect].append(float(np.nanmean(ap)))
        return {ro: {a: float(np.mean(v)) for a, v in values.items()} for ro, values in by.items()}

    @cached_property
    def selection(self):
        scores = dict(readout={}, teacher_wording={}, crl_version={})
        picks = {}
        for dataset, arms, aspects, ranges in (
                ('affection', AFF_ARMS, ('emotion',), {'emotion': self.ranges['map']['r']}),
                ('artelingo', ART_ARMS, ASPECTS, self.art_ranges)):
            maps, ro_picks = {}, {}
            scores['readout'][dataset] = {}
            for arm in arms:
                if dataset == 'artelingo' and arm == 'cstar':
                    ro_picks[arm] = 'center+k10'
                    continue
                ros = {R0, ('none', 0, 10)} if dataset == 'artelingo' and arm == 'crl_clip_painting' else None
                table = self.crossfit(dataset, arm, aspects, ros=ros)
                pick, summary = select_readout(table, ranges)
                if dataset == 'artelingo' and arm == 'crl_clip_painting':
                    pick = 'none+k10'
                maps[arm], ro_picks[arm] = table, pick
                scores['readout'][dataset][arm] = summary
            versions = VERSIONS if dataset == 'affection' else ('clip_painting', 'qwen_painting')
            vs = {v: float(np.mean([maps['crl_' + v][parse_ro(ro_picks['crl_' + v])][a] / ranges[a]
                                   for a in aspects])) for v in versions}
            scores['crl_version'][dataset] = vs
            picks[dataset] = dict(readout=ro_picks, crl_version=version_pick(vs))
            if dataset == 'affection':
                y = self.data.labels(dataset, 'val', 'emotion')
                clustering = {w: ami(self.parts('val', 'teacher_' + w), y) for w in ('artwork', 'photo')}
                ret = {w: maps['teacher_' + w][parse_ro(ro_picks['teacher_' + w])]['emotion']
                       for w in ('artwork', 'photo')}
                scores['teacher_wording'] = dict(clustering=clustering, retrieval=ret)
                picks[dataset]['teacher_wording'] = {
                    'clustering': teacher_pick(clustering['artwork'], clustering['photo'], self.ranges['clu']['r']),
                    'retrieval': teacher_pick(ret['artwork'], ret['photo'], self.ranges['map']['r'])}
        return dict(scores=scores, picks=picks)

    def perquery(self, dataset, arm, aspect, ro_name_):
        key = dataset, arm, aspect, ro_name_
        if key not in self.query_cache:
            split = 'test' if dataset == 'affection' else 'held'
            ro = parse_ro(ro_name_)
            values = []
            for seed in (SEEDS if arm in ('cstar', 'percept') else (None,)):
                val = self.data.embed(dataset, arm, 'val', aspect, seed)
                x = self.data.embed(dataset, arm, split, aspect, seed)
                stats = fit(val) if ro[0] != 'none' else None
                q = retrieval(transform(x, ro, stats), self.data.labels(dataset, split, aspect), ks=(ro[2],))
                values.append(q['ap'][ro[2]])
            values = np.stack(values)
            keep = np.isfinite(values[0])
            assert (np.isfinite(values) == keep[None]).all() and keep.any()
            self.query_cache[key] = values[:, keep] if arm in ('cstar', 'percept') else values[0, keep]
        return self.query_cache[key]

    def sc_row(self, sid, dataset, aspect, comp, metric):
        picks = self.selection['picks'][dataset]['readout']
        if metric == 'ami':
            stat = paired_cluster(self.parts('test', 'cstar'), self.parts('test', comp),
                                  self.data.labels(dataset, 'test', aspect), self.draws_clu)
            r, roc, rot = self.ranges['clu']['r'], None, None
        else:
            roc, rot = picks['cstar'], picks[comp]
            stat = paired_retrieval(self.perquery(dataset, 'cstar', aspect, roc),
                                    self.perquery(dataset, comp, aspect, rot), self.draws_ret)
            r = self.ranges['map']['r'] if dataset == 'affection' else self.art_ranges[aspect]
        result = outcome(stat['interval'][0], r)
        return dict(candidate='cstar', comparator=comp, dataset=dataset, aspect=aspect, metric=metric,
                    readout_c=roc, readout_t=rot, **stat, r=r, outcome=result, **{'pass': passes(sid, result)})

    def derive(self, stage):
        if stage == '0b':
            return self.ranges
        if stage == '1':
            return self.selection
        aff = self.selection['picks']['affection']
        if stage == '2':
            clu = 'teacher_' + aff['teacher_wording']['clustering']
            ret = 'teacher_' + aff['teacher_wording']['retrieval']
            sc = {sid: self.sc_row(sid, 'affection', 'emotion', comp, metric) for sid, comp, metric in (
                ('SC1', 'plain', 'ami'), ('SC2', 'plain', 'map'), ('SC3', clu, 'ami'), ('SC4', ret, 'map'))}
            return dict(sc=sc, verdict=verdict({k: v['outcome'] for k, v in sc.items()}),
                        repeats_artelingo_win=repeats_line(sc['SC3']['outcome']))
        art_comp = 'crl_' + self.selection['picks']['artelingo']['crl_version']
        sc = {f'SC{i}': self.sc_row(f'SC{i}', 'artelingo', a, art_comp, 'map')
              for i, a in enumerate(ASPECTS, 5)}
        sc['SC8'] = self.sc_row('SC8', 'affection', 'emotion', 'crl_' + aff['crl_version'], 'map')
        return dict(sc=sc, claims={k: v['pass'] for k, v in sc.items()})


class Check:
    def __init__(self):
        self.items = {}

    def compare(self, name, ours, theirs, strict=False):
        """Compare every expected leaf; extra descriptive output fields do not decide C13."""
        if isinstance(ours, dict):
            if not isinstance(theirs, dict):
                self.items[name] = dict(ours=ours, theirs=theirs, ok=False)
            else:
                if strict:
                    self.compare(f'{name}.keys', sorted(ours), sorted(theirs))
                for k, v in ours.items():
                    if k not in theirs:
                        self.items[f'{name}.{k}'] = dict(ours=v, theirs='<missing key>', ok=False)
                    else:
                        self.compare(f'{name}.{k}', v, theirs[k], strict=strict)
            return
        if isinstance(ours, (list, tuple)):
            if not isinstance(theirs, (list, tuple)) or len(ours) != len(theirs):
                self.items[name] = dict(ours=ours, theirs=theirs, ok=False)
            else:
                for i, (a, b) in enumerate(zip(ours, theirs)):
                    self.compare(f'{name}[{i}]', a, b)
            return
        if isinstance(ours, (float, np.floating)):
            ok = (isinstance(theirs, (int, float)) and not isinstance(theirs, bool)
                  and np.isfinite(ours) and np.isfinite(theirs) and abs(ours - theirs) <= 1e-6)
        else:
            ok = type(ours) is type(theirs) and ours == theirs
        self.items[name] = dict(ours=ours, theirs=theirs, ok=bool(ok))

    @property
    def ok(self):
        return all(v['ok'] for v in self.items.values())


def records_dir(outputs, smoke):
    """Where the committed records (stage0b_ranges.json, stage1_picks.json) are read: the round folder in a full
    run; in smoke the runner's scratch copies under <outputs>/records, so a smoke run never touches the round folder."""
    return Path(outputs) / 'records' if smoke else ROUND


def compare_final(run, stage, mine, outputs):
    """The ONLY round result reader. All independent calculations have already finished."""
    outputs = Path(outputs)
    check = Check()
    for aspect in sorted(getattr(run.data, 'frozen_crl_aspects', ())):
        if aspect != 'emotion' and not (outputs / 'crl' / f'clip_painting_{aspect}.npy').exists():
            continue                                  # the round copies r11's CLIP painting texts for emotion only
        frozen = np.load(R11 / 'crl' / f'{aspect}_text.npy', allow_pickle=False)
        local = np.load(outputs / 'crl' / f'clip_painting_{aspect}.npy', allow_pickle=False)
        equal = local.dtype == frozen.dtype and np.array_equal(local, frozen)
        check.compare(f'inputs.crl.clip_painting_{aspect}.equals_r11', True, bool(equal))
    summary = label_summary(run.data.full)
    if not run.smoke:
        check.compare('labels.spec', summary, {sp: dict(counts=EXPECTED_COUNTS[sp], ties=EXPECTED_TIES[sp],
                                                       n=sum(EXPECTED_COUNTS[sp])) for sp in ('val', 'test')})
    gates = read_json(outputs / 'stage0a/gates.json')
    check.compare('stage0a.ok', True, gates.get('ok'))
    items = gates['items']
    for sp in ('val', 'test'):
        for field, prefix in (('counts', 'label_counts'), ('ties', 'label_ties')):
            name = f'{prefix}_{sp}'
            if name in items:
                if run.smoke and items[name]['value'] == SKIPPED:    # full-split numbers: the runner skips them in smoke
                    check.items[f'labels.gates.{name}'] = dict(ours=summary[sp][field], theirs=SKIPPED, ok=True)
                    continue
                check.compare(f'labels.gates.{name}', summary[sp][field], items[name]['value'])
    if stage == '1':
        check.compare('scores', mine['scores'], read_json(outputs / 'stage1/scores.json'))
        check.compare('picks', mine['picks'], read_json(outputs / 'stage1/picks.json'), strict=True)
    else:
        target = 'ranges.json' if stage == '0b' else 'verdict.json'
        theirs = read_json(outputs / f'stage{stage}' / target)
        check.compare('result', mine, theirs)
        if stage in ('2', '3'):
            check.compare('sc.ids', sorted(mine['sc']), sorted(theirs.get('sc', {})))
    # Later stages also verify independently re-derived dependencies at the final comparison.
    if stage in ('2', '3'):
        check.compare('picks', run.selection['picks'], read_json(outputs / 'stage1/picks.json'), strict=True)
    if stage != '0b':
        check.compare('ranges', run.ranges, read_json(outputs / 'stage0b/ranges.json'))
    records = records_dir(outputs, run.smoke)
    check.compare('committed.ranges', mine if stage == '0b' else run.ranges,
                  read_json(records / 'stage0b_ranges.json'))
    if stage != '0b':
        check.compare('committed.picks', run.selection['picks'], read_json(records / 'stage1_picks.json'), strict=True)
    out = outputs / 'vh' / f'{stage}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(stage=stage, smoke=run.smoke, ok=check.ok, items=check.items,
                                   draws_clu=run.draws_clu, draws_ret=run.draws_ret,
                                   labels=summary, mine=mine), indent=2, allow_nan=False) + '\n')
    bad = sum(not v['ok'] for v in check.items.values())
    print(f'C13 stage {stage}: {len(check.items)} items, {bad} mismatches; {out}', flush=True)
    return 0 if check.ok else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--stage', required=True, choices=('0b', '1', '2', '3'))
    parser.add_argument('--outputs', type=Path, default=Path(os.environ.get('AC_OUTPUTS', DEFAULT_OUTPUTS)))
    parser.add_argument('--smoke', type=int, nargs='?', const=2100, default=0, metavar='N')
    parser.add_argument('--views-root', type=Path, default=None, metavar='DIR',
                        help='folder holding qwen3vl_emb_2b_aspect_prompts/ and clip_l14_valtest/ '
                             '(default: the real Affection folders); a smoke run uses <outputs>/smoke_views')
    parser.add_argument('--draws-clu', type=int, default=2000)
    parser.add_argument('--draws-ret', type=int, default=10000)
    args = parser.parse_args(argv)
    if args.smoke < 0:
        parser.error('--smoke must be positive (or omitted)')
    env_smoke = int(os.environ.get('AC_SMOKE') or 0)     # vh imports no ac_*: it reads the env itself
    if env_smoke and args.smoke != env_smoke:
        parser.error(f'AC_SMOKE={env_smoke} is set: pass --smoke {env_smoke} (got {args.smoke})')
    if args.draws_clu <= 0 or args.draws_ret <= 0:
        parser.error('bootstrap draws must be positive')
    try:
        extra = {'views_root': args.views_root} if args.views_root else {}
        run = Run(Data(args.outputs, args.smoke, **extra), args.smoke, args.draws_clu, args.draws_ret)
        mine = run.derive(args.stage)
        return compare_final(run, args.stage, mine, args.outputs)
    except (FileNotFoundError, KeyError, AssertionError, ValueError) as error:
        print(f'C13 stage {args.stage}: STOP: {type(error).__name__}: {error}', flush=True)
        return 2


if __name__ == '__main__':
    sys.exit(main())
