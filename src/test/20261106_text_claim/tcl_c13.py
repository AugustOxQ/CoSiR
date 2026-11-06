"""Independent C13 re-derivation of r23 §3.4 and §6.

Numerics taken from the run's path, logic derived from the rule; no numerical
diversity. Derive never opens r23 outputs. Only compare imports the round.
Run with PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8.
"""
import argparse
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROUND = Path('/project/CoSiR/src/test/20261106_text_claim')
DEFAULT = Path('/data/SSD2/cosir_r23/20261106_text_claim')
SMOKE_OUT = Path('/tmp/claude-0/-project-CoSiR/f61ddf0c-95e6-4a41-b251-ddcd6fdaaf07/scratchpad/c13_smoke')
FULL = Path('/data/SSD2/pre_extract/artelingo_full')
MM = Path('/data/SSD2/pre_extract/mmart')
R22 = Path('/data/SSD2/cosir_r22/20261105_text_retrieval')
WEIGHTS = Path('/data/SSD2/cosir_r10/20261027_recipe_sweep/outputs/full/weights/k11=40')
ASPECTS = ('emotion', 'style', 'genre')
ROWS = ('cstar', 'late', 'plain', 'teacher')
READOUTS = ('r0', 'center', 'center_k10')
DIRECTIONS = ('t2i', 'i2t', 't2t')
SEEDS = (0, 1, 2, 3, 4)
INSTRUCTIONS = dict(emotion='emotion_focus', style='style_word', genre='genre_word')
NOTE = "numerics taken from the run's path, logic derived from the rule; no numerical diversity"
PREFIXES = ('tcl_', 'tx_', 'rt_', 'mt_', 'mp_')


def read_json(path):
    return json.loads(Path(path).read_text())


def output_root(path):
    path = Path(path).resolve()
    assert path.is_relative_to(SMOKE_OUT) or path.is_relative_to(DEFAULT / 'c13'), path
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=True) + '\n')
    os.replace(tmp, path)


def atomic_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def complete(path, fields):
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as z:
            return all(k in z and z[k].size for k in fields)
    except (ValueError, OSError, EOFError):
        return False


def kind_of(group):
    return 'caption' if group == 'art' else 'narrative'


def configurations():
    for row in ROWS:
        for head in (('all',) if row == 'plain' else ASPECTS):
            for seed in (SEEDS if row in ('cstar', 'late') else ('x',)):
                yield row, head, seed


def score_path(out, name, row, head, seed, direction, readout):
    return out / 'scores' / name / kind_of(name.split('_')[0]) / row / f'{head}__s{seed}__{direction}__{readout}.npz'


def vector_path(out, category, name, row, head, seed):
    return out / category / name / kind_of(name.split('_')[0]) / row / f'{head}__s{seed}.npz'


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    assert np.isfinite(x).all() and (norm > 0).all()
    return x / norm


def gather(stored, requested):
    stored = list(stored)
    assert len(stored) == len(set(stored)), 'duplicate store IDs'
    lookup = {k: i for i, k in enumerate(stored)}
    return np.asarray([lookup[k] for k in requested], dtype=np.int64)


def store_rows(folder, side, requested, ids):
    folder = Path(folder)
    if ids == 'index':
        index = read_json(folder / 'index.json')
        stored = [r['sample_id' if side == 't' else 'painting'] for r in index['texts' if side == 't' else 'images']]
    elif ids == 'keys':
        stored = read_json(folder / 'keys.json')
    else:
        with np.load(folder / 'ids.npz', allow_pickle=False) as z:
            stored = z['sample_ids' if side == 't' else 'painting_names'].tolist()
    x = np.load(folder / ('txt.npy' if side == 't' else 'img.npy'), mmap_mode='r', allow_pickle=False)
    assert len(x) == len(stored)
    return unit(x[gather(stored, requested)])


class Inputs:
    """Only the frozen names/labels functions cross the older-round boundary."""
    def __init__(self, smoke=0, stores=None):
        assert smoke or not stores, '--stores requires --smoke'
        self.smoke, self.stores = smoke, Path(stores) if stores else None
        before = set(sys.modules)
        assert not any(n.startswith(PREFIXES) for n in before), 'derive loaded a round module before frozen readers'
        for folder in ('20261029_retrieval', '20261102_mmart_test'):
            sys.path.insert(0, '/project/CoSiR/src/test/' + folder)
        from rt_data import names, labels
        from mt_data import keys, labels as mm_labels
        self.art_names, self.art_labels = names, labels
        self.mm_keys, self.mm_labels = keys, mm_labels
        self.allowed = {n for n in set(sys.modules) - before if n.startswith(PREFIXES)}

    @functools.lru_cache(maxsize=1)
    def frame(self, name):
        group, split = name.split('_')
        if group == 'art':
            names = self.art_names(split)
            if self.smoke:
                names = names[:self.smoke]
            labels = {a: x[:len(names)] for a, x in self.art_labels(split).items()}
            lookup = {n: i for i, n in enumerate(names)}
            index = read_json(FULL / 'qwen3vl_emb_2b/index.json')
            texts = sorted((r for r in index['texts'] if r['painting'] in lookup), key=lambda r: r['sample_id'])
            ids = [r['sample_id'] for r in texts]
            assert len(ids) == len(set(ids))
            tp = np.asarray([lookup[r['painting']] for r in texts], dtype=np.int64)
            annotations = read_json('/data/PDD/artelingo/artelingo_train.json')
            codes = {v: i for i, v in enumerate(sorted({r['emotion'] for r in annotations}))}
            assert len(codes) == 9
            tl = {a: x[tp] for a, x in labels.items()}
            for r in texts:
                assert annotations[r['sample_id']]['painting'] == r['painting']
            tl['emotion'] = np.asarray([codes[annotations[k]['emotion']] for k in ids], dtype=np.int64)
            index_rows = None
        else:
            names = self.mm_keys(split)
            if self.smoke:
                names = names[:self.smoke]
            labels = {a: self.mm_labels(names, a) for a in ASPECTS}
            ids, tp, tl = names, np.arange(len(names), dtype=np.int64), labels
            index = pd.read_parquet('/data/SSD/mmart/mmart_index.parquet', columns=['key', 'painting', 'artelingo'])
            assert index.key.is_unique
            index_rows = index.set_index('key').loc[names]
        assert len(names) == len(set(names))
        if not self.smoke:
            sizes = {'art_val': (6152, 30872), 'art_held': (12281, 61744),
                     'mm_val': (5769, 5769), 'mm_test': (16384, 16384)}
            assert (len(names), len(ids)) == sizes[name], (name, len(names), len(ids), sizes[name])
        return dict(names=names, ids=ids, tp=tp, ip=np.arange(len(names), dtype=np.int64),
                    labels={a: {'t': tl[a], 'i': labels[a]} for a in ASPECTS}, index=index_rows)

    def new_store(self, root, suffix):
        return (self.stores if self.stores else root) / suffix

    @functools.lru_cache(maxsize=1)
    def plain(self, name):
        return self.load(name, None)

    def load(self, name, aspect):
        group, split = name.split('_')
        fr = self.frame(name)
        inst = INSTRUCTIONS.get(aspect)
        if group == 'art':
            if aspect is None:
                t = store_rows(FULL / 'qwen3vl_emb_2b', 't', fr['ids'], 'index')
                folder = FULL / 'qwen3vl_emb_2b'
                id_kind = 'index'
            else:
                if split == 'held':
                    tf = self.new_store(FULL, f'qwen3vl_emb_2b_aspect_prompts/{inst}_heldtxt')
                elif aspect == 'emotion':
                    tf = FULL / f'qwen3vl_emb_2b_aspect_prompts/{inst}_s2txt'
                else:
                    tf = R22 / f'text/artelingo_val/{inst}'
                assert (tf / 'done.json').is_file(), f'incomplete text store: {tf}'
                t = store_rows(tf, 't', fr['ids'], 'ids')
                folder = FULL / f'qwen3vl_emb_2b_aspect_prompts/{inst}_{"held" if split == "held" else "s2img"}'
                id_kind = 'ids'
            i = store_rows(folder, 'i', fr['names'], id_kind)
        else:
            if aspect is None and split == 'val':
                tf = R22 / 'text/mmart_val/e_narrative'
            else:
                tf = self.new_store(MM, f'qwen3vl_emb_2b_text/narrative/{inst or "default"}/{split}')
            assert (tf / 'done.json').is_file(), f'incomplete text store: {tf}'
            t = store_rows(tf, 't', fr['names'], 'keys')
            index = fr['index']
            i = np.empty_like(t)
            for own in (False, True):
                pos = np.flatnonzero((index.artelingo.to_numpy() != 'none') == own)
                if not len(pos):
                    continue
                if own:
                    folder = FULL / (f'qwen3vl_emb_2b_aspect_prompts/{inst}_s2img' if aspect else 'qwen3vl_emb_2b')
                    requested = index.painting.iloc[pos].tolist()
                    id_kind = 'ids' if aspect else 'index'
                else:
                    folder = MM / (f'qwen3vl_emb_2b_aspect_prompts/{inst}' if aspect else 'qwen3vl_emb_2b')
                    requested = [fr['names'][p] for p in pos]
                    id_kind = 'keys'
                i[pos] = store_rows(folder, 'i', requested, id_kind)
        return t, i

    def rows(self, name, row, head, seed, device='cuda'):
        if row == 'teacher':
            return self.load(name, head)
        plain = self.plain(name)
        if row == 'plain':
            return plain
        state = torch.load(WEIGHTS / f'{head}__s{seed}.pt', map_location=device, weights_only=True)
        result = []
        with torch.inference_mode():
            for x, side in zip(plain, ('cap', 'img')):
                x = unit(x)  # second numpy normalization, numeric path line 2
                chunks = []
                for start in range(0, len(x), 4096):
                    z = torch.as_tensor(x[start:start + 4096], device=device)
                    z = F.linear(z, state[f'{side}.trunk.0.weight'], state[f'{side}.trunk.0.bias'])
                    z = F.gelu(z)
                    z = F.linear(z, state[f'{side}.output.weight'], state[f'{side}.output.bias'])
                    chunks.append((z / z.norm(dim=1, keepdim=True)).cpu().numpy())
                result.append(np.concatenate(chunks))
        if row == 'late':
            result = [np.concatenate((math.sqrt(1 - 0.7) * c, math.sqrt(0.7) * p), axis=1).astype(np.float32)
                      for c, p in zip(result, plain)]
        return tuple(result)

    def assert_independent(self):
        bad = [n for n in sys.modules if n.startswith(PREFIXES) and n not in self.allowed]
        assert not bad, f'non-input modules loaded in derive: {bad}'


def fit_means(t, i):
    return dict(t=t.astype(np.float64).mean(0).astype(np.float32),
                i=i.astype(np.float64).mean(0).astype(np.float32), n_t=np.asarray(len(t)), n_i=np.asarray(len(i)))


def rank_order(scores, own=None):
    if own is not None:
        scores[torch.arange(len(scores), device=scores.device), own] = -torch.inf
    order = scores.argsort(dim=1, descending=True)
    # The numeric path requires ascending gallery position for exact ties.
    sorted_scores = scores.gather(1, order)
    tied = (sorted_scores[:, 1:] == sorted_scores[:, :-1]).any(1)
    if tied.any():
        order[tied] = scores[tied].argsort(dim=1, descending=True, stable=True)
    if own is not None:
        order = order[order != own[:, None]].reshape(len(scores), -1)
    return order


def expand(q, g, order):
    e = q + g[order[:, :10]].sum(1)
    return e / e.norm(dim=1, keepdim=True)


def metric_arrays(order, qpid, gpid, labels):
    """Rank all gallery rows first; unlabelled rows disappear only in AP."""
    device = order.device
    qpid = torch.as_tensor(qpid, device=device)
    gpid = torch.as_tensor(gpid, device=device)
    hit = gpid[order] == qpid[:, None]
    first = torch.where(hit.any(1), hit.to(torch.int8).argmax(1).float() + 1,
                        torch.full((len(order),), float('nan'), device=device))
    result = dict(q_pid=qpid.cpu().numpy(), first_rank=first.cpu().numpy())
    for aspect, (ql, gl) in labels.items():
        ql = torch.as_tensor(ql, device=device)
        gl = torch.as_tensor(gl, device=device)[order]
        relevant = (gl == ql[:, None]) & (ql[:, None] >= 0)
        denom = (gl >= 0).cumsum(1)
        nrel = relevant.sum(1)
        precision = relevant.cumsum(1).double() / denom.clamp_min(1)
        ap = (precision * relevant).sum(1) / nrel.clamp_min(1)
        ap[(ql < 0) | (nrel == 0)] = float('nan')
        result['ap__' + aspect] = ap.float().cpu().numpy()
    return result


def evaluate(t, i, frame, head, direction, readouts, means, device='cuda'):
    qs, gs = dict(t2i=('t', 'i'), i2t=('i', 't'), t2t=('t', 't'))[direction]
    vectors = dict(t=t, i=i)
    q = torch.as_tensor(vectors[qs], device=device)
    g = torch.as_tensor(vectors[gs], device=device)
    qp, gp = frame['tp' if qs == 't' else 'ip'], frame['tp' if gs == 't' else 'ip']
    aspects = ASPECTS if head == 'all' else (head,)
    labels = {a: (frame['labels'][a][qs], frame['labels'][a][gs]) for a in aspects}
    result = {r: [] for r in readouts}
    for mode in ('raw', 'centered'):
        menu = [r for r in readouts if (r == 'r0') == (mode == 'raw')]
        if not menu:
            continue
        if mode == 'centered':
            q = q - torch.as_tensor(means[qs], device=device)
            g = g - torch.as_tensor(means[gs], device=device)
            q = q / q.norm(dim=1, keepdim=True)
            g = g / g.norm(dim=1, keepdim=True)
        for a in range(0, len(q), 1024):
            b = min(a + 1024, len(q))
            own = torch.arange(a, b, device=device) if direction == 't2t' else None
            order = rank_order(q[a:b] @ g.T, own)
            lab = {k: (v[0][a:b], v[1]) for k, v in labels.items()}
            for r in menu:
                o = rank_order(expand(q[a:b], g, order) @ g.T, own) if r == 'center_k10' else order
                result[r].append(metric_arrays(o, qp[a:b], gp, lab))
    return {r: {k: np.concatenate([chunk[k] for chunk in chunks]) for k in chunks[0]} for r, chunks in result.items()}


def pick(values, menu=READOUTS):
    assert all(np.isfinite(values[k]) for k in menu), 'unscored validation pick'
    return max(menu, key=lambda k: values[k])


def pair_summary(rank):
    rank = np.asarray(rank, dtype=np.float64)
    return dict(hits=int(np.sum(rank <= 10)), mrr=float(np.nansum(1 / rank) / len(rank)))


def make_stage1(out, group):
    name, kind = group + '_val', kind_of(group)
    values, picks, ranges = {}, {}, {}
    for row in ROWS:
        values[row], picks[row] = {}, {}
        if row in ('cstar', 'late'):
            ranges[row] = {}
        for aspect in ASPECTS:
            values[row][aspect], picks[row][aspect] = {}, {}
            if row in ranges:
                ranges[row][aspect] = {}
            for d in DIRECTIONS:
                raw = {}
                for r in READOUTS:
                    raw[r] = []
                    for seed in (SEEDS if row in ('cstar', 'late') else ('x',)):
                        p = score_path(out, name, row, 'all' if row == 'plain' else aspect, seed, d, r)
                        with np.load(p) as z:
                            raw[r].append(float(np.nanmean(z['ap__' + aspect].astype(np.float64))))
                vals = {r: float(np.mean(raw[r])) for r in READOUTS}
                chosen = pick(vals)
                values[row][aspect][d] = vals
                picks[row][aspect][d] = chosen
                if row in ranges:
                    ranges[row][aspect][d] = max(raw[chosen]) - min(raw[chosen])
    pair = {}
    for d in DIRECTIONS[:2]:
        head_values = {}
        for head in ASPECTS:
            stats = {}
            for r in READOUTS:
                summary = []
                for s in SEEDS:
                    with np.load(score_path(out, name, 'late', head, s, d, r)) as z:
                        summary.append(pair_summary(z['first_rank']))
                stats[r] = dict(mrr=float(np.mean([x['mrr'] for x in summary])), hits=sum(x['hits'] for x in summary))
            r = pick({r: stats[r]['mrr'] for r in READOUTS})
            head_values[head] = dict(readout=r, hits=stats[r]['hits'])
        h = pick({h: head_values[h]['hits'] for h in ASPECTS}, ASPECTS)
        plain = {}
        for r in READOUTS:
            with np.load(score_path(out, name, 'plain', 'all', 'x', d, r)) as z:
                plain[r] = pair_summary(z['first_rank'])['mrr']
        pair[d] = dict(late=dict(head=h, readout=head_values[h]['readout']), plain=dict(readout=pick(plain)))
    return dict(group=group, picks={kind: picks}, values={kind: values}, ranges={kind: ranges}, pair_picks={kind: pair})


def needed(stage, row, head):
    kind = kind_of(stage['group'])
    plan = set()
    for aspect in (ASPECTS if head == 'all' else (head,)):
        for d in DIRECTIONS:
            plan.add((d, stage['picks'][kind][row][aspect][d]))
    if row in ('late', 'plain'):
        for d in DIRECTIONS[:2]:
            p = stage['pair_picks'][kind][d][row]
            if row == 'plain' or p['head'] == head:
                plan.add((d, p['readout']))
    return plan


def replicate(sum_c, sum_m, counts, I, S):
    weights = np.bincount(I, minlength=len(counts)).astype(np.float64)
    den = weights @ counts
    return float(np.mean((sum_c @ weights)[S]) / den - (sum_m @ weights) / den)


def painting_interval(candidate, comparator, pid, draws=10000):
    candidate = np.asarray(candidate, dtype=np.float64)
    comparator = np.asarray(comparator, dtype=np.float64)
    pid = np.asarray(pid)
    assert candidate.shape == (5, len(comparator)) and len(pid) == len(comparator)
    mask = np.isfinite(comparator)
    assert np.array_equal(np.isfinite(candidate), np.broadcast_to(mask, candidate.shape)), 'scored masks differ'
    _, inv = np.unique(pid[mask], return_inverse=True)
    m = int(inv.max() + 1) if len(inv) else 0
    assert m, 'no scored paintings'
    counts = np.bincount(inv, minlength=m).astype(np.float64)
    sums = np.stack([np.bincount(inv, weights=x[mask], minlength=m) for x in candidate])
    baseline = np.bincount(inv, weights=comparator[mask], minlength=m)
    rng = np.random.default_rng(0)
    values = np.empty(draws, dtype=np.float64)
    for j in range(draws):
        I = rng.integers(0, m, size=m)
        S = rng.choice([0, 1, 2, 3, 4], size=5, replace=True)
        values[j] = replicate(sums, baseline, counts, I, S)
    lower, upper = np.percentile(values, [2.5, 97.5])
    return dict(point=float(candidate[:, mask].mean() - comparator[mask].mean()), lower=float(lower), upper=float(upper),
                n_query=int(mask.sum()), n_units=m)


def outcome(lower, seed_range):
    return 'WIN' if lower > 0 else ('LEVEL' if lower > -seed_range else 'BELOW')


def verdict(P, T, Q=None):
    if P != 9 or Q is False:
        return 'fails'
    return 'holds' if T >= 5 else ('qualified' if T in (3, 4) else 'fails')


def cell_key(c):
    return tuple(c[k] for k in ('sc', 'cell', 'candidate', 'comparator', 'aspect', 'direction'))


def build_cells(out, groups):
    cells, verdicts = [], {}
    for group in groups:
        stage = read_json(out / f'stage1_{group}.json')
        kind = kind_of(group)
        name = 'art_held' if group == 'art' else 'mm_test'
        for row in ('cstar', 'late'):
            sc = {('art', 'cstar'): 'SC1', ('mm', 'cstar'): 'SC2', ('art', 'late'): 'SC3', ('mm', 'late'): 'SC4'}[group, row]
            own_cells = []
            for comp in ('plain', 'teacher'):
                for aspect in ASPECTS:
                    for d in DIRECTIONS:
                        cr = stage['picks'][kind][row][aspect][d]
                        br = stage['picks'][kind][comp][aspect][d]
                        candidates = []
                        for seed in SEEDS:
                            with np.load(score_path(out, name, row, aspect, seed, d, cr)) as z:
                                candidates.append(z['ap__' + aspect])
                                pid = z['q_pid']
                        with np.load(score_path(out, name, comp, 'all' if comp == 'plain' else aspect, 'x', d, br)) as z:
                            assert np.array_equal(pid, z['q_pid'])
                            interval = painting_interval(candidates, z['ap__' + aspect], pid)
                        sr = stage['ranges'][kind][row][aspect][d]
                        c = dict(sc=sc, group=group, kind=kind, candidate=row, comparator=comp, cell='label', aspect=aspect,
                                 direction=d, candidate_readout=cr, comparator_readout=br, head=aspect, seed_range=sr,
                                 outcome=outcome(interval['lower'], sr), claim_voided=False, **interval)
                        cells.append(c)
                        own_cells.append(c)
            P = sum(c['outcome'] == 'WIN' for c in own_cells if c['comparator'] == 'plain')
            T = sum(c['outcome'] == 'WIN' for c in own_cells if c['comparator'] == 'teacher')
            Q = None
            if row == 'late':
                above = []
                for d in DIRECTIONS[:2]:
                    p = stage['pair_picks'][kind][d]
                    candidates = []
                    for seed in SEEDS:
                        with np.load(score_path(out, name, row, p[row]['head'], seed, d, p[row]['readout'])) as z:
                            candidates.append((z['first_rank'] <= 10).astype(np.float64))
                            pid = z['q_pid']
                    with np.load(score_path(out, name, 'plain', 'all', 'x', d, p['plain']['readout'])) as z:
                        assert np.array_equal(pid, z['q_pid'])
                        interval = painting_interval(candidates, (z['first_rank'] <= 10).astype(np.float64), pid)
                    limit = interval['lower'] > -0.02
                    above.append(limit)
                    cells.append(dict(sc=sc, group=group, kind=kind, candidate=row, comparator='plain', cell='pair',
                                      aspect=None, direction=d, candidate_readout=p[row]['readout'], comparator_readout=p['plain']['readout'],
                                      head=p[row]['head'], seed_range=None, outcome=None, above_limit=limit, claim_voided=False, **interval))
                Q = all(above)
            verdicts[sc] = dict(P=P, T=T, Q=Q, verdict=verdict(P, T, Q))
    return cells, dict(verdicts=verdicts, voided=[])


def derive_sha256(source=None):
    """Identity of the UTF-8 source before class Audit (all derive code).

    Compare and the CLI dispatch follow that boundary and do not affect caches.
    Imports, constants, input readers and shared numeric helpers precede it.
    """
    if source is None:
        source = Path(__file__).read_text()
    derive_source = source.split('\nclass Audit:', 1)[0]
    return hashlib.sha256(derive_source.encode('utf-8')).hexdigest()


def derive(args):
    assert args.smoke or not args.stores, '--stores requires --smoke'
    out = output_root(args.out or (SMOKE_OUT if args.smoke else DEFAULT / 'c13'))
    manifest = dict(version=2, derive_sha256=derive_sha256(), smoke=args.smoke,
                    stores=str(Path(args.stores).resolve()) if args.stores else None)
    mp = out / 'inputs.json'
    if mp.exists():
        assert read_json(mp) == manifest, 'output belongs to different inputs or derive code; use a fresh --out'
    else:
        atomic_json(mp, manifest)
    inputs = Inputs(args.smoke, args.stores)
    groups = ('art', 'mm') if args.group == 'all' else (args.group,)
    torch.set_num_threads(8)
    with torch.inference_mode():
        for part in (('val', 'held') if args.part == 'all' else (args.part,)):
            for group in groups:
                name = group + '_val' if part == 'val' else ('art_held' if group == 'art' else 'mm_test')
                stage = None if part == 'val' else read_json(out / f'stage1_{group}.json')
                frame = inputs.frame(name)
                for row, head, seed in configurations():
                    rp = vector_path(out, 'rows', name, row, head, seed)
                    if not complete(rp, ('t', 'i')):
                        t, i = inputs.rows(name, row, head, seed)
                        atomic_npz(rp, t=t, i=i)
                    else:
                        with np.load(rp) as z:
                            t, i = z['t'], z['i']
                    mean_path = vector_path(out, 'means', group + '_val', row, head, seed)
                    if part == 'val' and not complete(mean_path, ('t', 'i', 'n_t', 'n_i')):
                        atomic_npz(mean_path, **fit_means(t, i))
                    with np.load(mean_path) as z:
                        means = {k: z[k] for k in z.files}
                    plan = set((d, r) for d in DIRECTIONS for r in READOUTS) if part == 'val' else needed(stage, row, head)
                    for d in DIRECTIONS:
                        menu = [r for r in READOUTS if (d, r) in plan and not complete(score_path(out, name, row, head, seed, d, r),
                                ('q_pid', 'first_rank') + tuple('ap__' + a for a in (ASPECTS if head == 'all' else (head,))))]
                        if menu:
                            scores = evaluate(t, i, frame, head, d, menu, means)
                            for r, arrays in scores.items():
                                atomic_npz(score_path(out, name, row, head, seed, d, r), **arrays)
                            print(f'C13 {name} {row} {head} s{seed} {d} {",".join(menu)}', flush=True)
                if part == 'val':
                    atomic_json(out / f'stage1_{group}.json', make_stage1(out, group))
        if args.part in ('held', 'all') and args.group == 'all':
            cells, outcomes = build_cells(out, groups)
            atomic_json(out / 'cells.json', cells)
            atomic_json(out / 'outcomes.json', outcomes)
    inputs.assert_independent()


class Audit:
    """Fail closed per item; a controller-authorized void records differences apart."""
    def __init__(self, c13_void=()):
        self.categories = {}
        self.failures = []
        self.failed_total = 0
        self.c13_void = frozenset(c13_void)
        self.void_differences = []

    def check(self, category, key, expected, actual, exact=False, scs=None):
        stat = self.categories.setdefault(category, dict(items=0, failed=0, void_differences=0,
                                                       largest_difference=0.0, not_bit_identical=0))
        stat['items'] += 1
        why, diff = None, 0.0
        try:
            if expected is MISSING or actual is MISSING:
                why = 'missing item'
            elif exact or expected is None or isinstance(expected, (str, bool)):
                if isinstance(expected, np.ndarray):
                    a, b = expected, np.asarray(actual)
                    ok = a.shape == b.shape and np.array_equal(a, b, equal_nan=a.dtype.kind in 'fc')
                else:
                    ok = expected == actual
                if not ok:
                    why = 'exact value differs'
            else:
                a, b = np.asarray(expected), np.asarray(actual)
                if a.shape != b.shape:
                    why = f'shape differs: {a.shape} vs {b.shape}'
                elif not (np.array_equal(np.isnan(a), np.isnan(b)) and np.array_equal(np.isposinf(a), np.isposinf(b))
                          and np.array_equal(np.isneginf(a), np.isneginf(b))):
                    why = 'NaN/nonfinite mask differs'
                else:
                    finite = np.isfinite(a)
                    diff = float(np.max(np.abs(a[finite].astype(np.float64) - b[finite].astype(np.float64)), initial=0))
                    identical = a.dtype == b.dtype and a.tobytes() == b.tobytes()
                    stat['not_bit_identical'] += int(not identical)
                    if diff > 1e-6:
                        why = f'difference {diff:.12g} exceeds 1e-6'
        except (TypeError, ValueError, OverflowError) as e:
            why = f'type fault: {e}'
        stat['largest_difference'] = max(stat['largest_difference'], diff)
        if why:
            readers = readers_of(category, key) if scs is None else frozenset(scs)
            item = dict(what=category, key=str(key), why=why)
            if readers and readers <= self.c13_void:
                stat['void_differences'] += 1
                self.void_differences.append(dict(item, scs=sorted(readers)))
            else:
                stat['failed'] += 1
                self.failed_total += 1
                if len(self.failures) < 200:
                    self.failures.append(item)
        return why is None

    def error(self, category, key, error, scs=None):
        self.check(category, f'{key}: {error}', True, MISSING, scs=scs)

    def tree(self, category, expected, actual, prefix=''):
        if isinstance(expected, dict):
            if not isinstance(actual, dict):
                self.check(category, prefix, expected, MISSING)
                return
            for k, v in expected.items():
                self.tree(category, v, actual.get(k, MISSING), prefix + '/' + str(k))
        else:
            self.check(category, prefix, expected, actual, exact=isinstance(expected, (str, bool, int)))

    def result(self):
        return {'pass': self.failed_total == 0, 'categories': self.categories,
                'failed_total': self.failed_total, 'failures': self.failures,
                'void_differences': self.void_differences, 'note': NOTE}


def row_scs(group, row):
    cstar, late = ('SC1', 'SC3') if group == 'art' else ('SC2', 'SC4')
    return frozenset((cstar,)) if row == 'cstar' else (frozenset((late,)) if row == 'late' else frozenset((cstar, late)))


def readers_of(category, key):
    """Dependencies of an audited row/mean/ranking/pick, or one verdict."""
    parts = str(key).strip('/').split('/')
    if category == 'verdicts' and parts[0] in ('SC1', 'SC2', 'SC3', 'SC4'):
        return frozenset((parts[0],))
    if category in ('rows', 'means', 'scores', 'q_pid', 'first_rank', 'ap') and len(parts) > 2:
        if parts[0].startswith(('art_', 'mm_')) and parts[2] in ROWS:
            return row_scs(parts[0].split('_')[0], parts[2])
    if category == 'stage1' and parts[0] in ('art', 'mm'):
        # picks/values/ranges: group/section/kind/row; pair_picks: group/section/kind/direction/row.
        index = 4 if len(parts) > 1 and parts[1] == 'pair_picks' else 3
        if len(parts) > index and parts[index] in ROWS:
            return row_scs(parts[0], parts[index])
    return frozenset()


def void_is_authorized(status, sc):
    if sc not in ('SC1', 'SC2', 'SC3', 'SC4'):
        return False
    group = 'art' if sc in ('SC1', 'SC3') else 'mm'
    if sc in status.get('c13_void', []):
        return True
    if status.get('gate_iii') is False:
        return True
    if any(status.get(k, {}).get(group) is False for k in ('gate_ii', 'gate_iv', 'stage1')):
        return True
    for key, passed in status.get('gate_i', {}).items():
        parts = key.split('/')
        if (len(parts) == 4 and parts[0].startswith(group + '_') and parts[1] == kind_of(group)
                and parts[3] in ('plain', 'teacher') and passed is False):
            return True
    return False


MISSING = object()


def compare_score(audit, key, expected, actual):
    for field in expected:
        audit.check('q_pid' if field == 'q_pid' else ('first_rank' if field == 'first_rank' else 'ap'),
                    key + '/' + field, expected[field], actual.get(field, MISSING), exact=field in ('q_pid', 'first_rank'))


def compare_cells(audit, expected, actual):
    want, got = {}, {}
    for target, entries, label in ((want, expected, 'derived'), (got, actual, 'round')):
        if not isinstance(entries, list):
            audit.error('cells', label, 'expected a list')
            continue
        for c in entries:
            try:
                k = cell_key(c)
                if k in target:
                    audit.error('cells', k, 'duplicate ' + label + ' cell', scs=(c['sc'],))
                target[k] = c
            except (KeyError, TypeError) as e:
                audit.error('cells', label, e, scs=(c['sc'],) if isinstance(c, dict) and c.get('sc') in ('SC1', 'SC2', 'SC3', 'SC4') else ())
    changed = set(want) ^ set(got)
    scopes = {k[0] for k in changed}
    audit.check('cell_set', 'exact cell keys', sorted(want, key=str), sorted(got, key=str), exact=True, scs=scopes)
    numeric = ('point', 'lower', 'upper', 'seed_range')
    for key, cell in want.items():
        other = got.get(key, {})
        for field, value in cell.items():
            audit.check('cells', str(key) + '/' + field, value, other.get(field, MISSING),
                        exact=field not in numeric, scs=(cell['sc'],))


def closest_to_bar(cells):
    closest = None
    for c in cells:
        bars = [-0.02] if c['cell'] == 'pair' else [0.0, -c['seed_range']]
        for bar in bars:
            entry = dict(key=list(cell_key(c)), lower=c['lower'], bar=bar, distance=abs(c['lower'] - bar))
            if closest is None or entry['distance'] < closest['distance']:
                closest = entry
    return closest


def retire_result(path):
    """Retire stale results without deletion, as required by this job's file rule."""
    if path.exists():
        import time
        archive = path.with_name(path.name + f'.previous.{time.time_ns()}')
        os.replace(path, archive)
        return str(archive)
    return None


def compare(args):
    out = output_root(args.c13_out or DEFAULT / 'c13')
    result_path = out / f'compare_{args.part}.json'
    retired = [retire_result(out / f'compare_{part}.json') for part in ('val', 'all')]
    retired = [p for p in retired if p]  # before imports or any input read
    audit = Audit()
    hashes, voided, closest = {}, [], None
    round_out, records = Path(args.round_out or DEFAULT), Path(args.records or ROUND / 'records')
    try:
        # These are the ONLY imports of round modules in this file.
        import tcl_paths as P
        from tcl_forward import rows as round_rows
        status = read_json(records / 'status.json')
        c13_void = status.get('c13_void', [])
        known_scs = {'SC1', 'SC2', 'SC3', 'SC4'}
        valid_voids = isinstance(c13_void, list) and all(isinstance(sc, str) and sc in known_scs for sc in c13_void)
        audit.check('status', 'c13_void valid', True, valid_voids, exact=True)
        audit.c13_void = frozenset(c13_void) if valid_voids else frozenset()
        manifest = read_json(out / 'inputs.json')
        audit.check('inputs', 'derive code identity', derive_sha256(), manifest.get('derive_sha256', MISSING), exact=True)
        audit.check('inputs', 'stores require smoke (manifest)', True, bool(manifest.get('smoke') or not manifest.get('stores')), exact=True)
        store_root = getattr(P, 'STORE_ROOT', None)
        audit.check('inputs', 'stores require smoke (round)', True, bool(P.SMOKE or not store_root), exact=True)
        audit.check('inputs', 'stores setting', manifest.get('stores', MISSING),
                    str(Path(store_root).resolve()) if store_root else None, exact=True)
        # Use cached independent rows; compare never re-derives using the round.
        for group in ('art', 'mm'):
            stage = read_json(out / f'stage1_{group}.json')
            try:
                audit.tree('stage1', stage, read_json(records / f'stage1_{group}.json'), group)
            except Exception as e:
                audit.error('stage1', group, repr(e), scs=row_scs(group, 'plain'))
            sets = (group + '_val',) if args.part == 'val' else (group + '_val', 'art_held' if group == 'art' else 'mm_test')
            for name in sets:
                val = name.endswith('_val')
                kind = kind_of(group)
                for row, head, seed in configurations():
                    key = f'{name}/{kind}/{row}/{head}/{seed}'
                    try:
                        with np.load(vector_path(out, 'rows', name, row, head, seed)) as z:
                            own = (z['t'], z['i'])
                        actual = round_rows(name, kind, row, head, seed, 'cuda')
                        audit.check('rows', key + '/sides', 2, len(actual), exact=True)
                        for side, a, b in zip(('t', 'i'), own, actual):
                            if isinstance(b, torch.Tensor):
                                b = b.detach().cpu().numpy()
                            audit.check('rows', key + '/' + side, a, b)
                    except Exception as e:
                        audit.error('rows', key, repr(e))
                    if val:
                        try:
                            with np.load(vector_path(out, 'means', name, row, head, seed)) as a, np.load(P.means_path(name, kind, row, head, seed, out=round_out)) as b:
                                for field in ('t', 'i', 'n_t', 'n_i'):
                                    audit.check('means', key + '/' + field, a[field], b[field] if field in b else MISSING, exact=field.startswith('n_'))
                        except Exception as e:
                            audit.error('means', key, repr(e))
                    plan = set((d, r) for d in DIRECTIONS for r in READOUTS) if val else needed(stage, row, head)
                    for d, r in sorted(plan):
                        sk = key + '/' + d + '/' + r
                        try:
                            with np.load(score_path(out, name, row, head, seed, d, r)) as a, np.load(P.score_path(name, kind, row, head, seed, d, r, out=round_out)) as b:
                                compare_score(audit, sk, {k: a[k] for k in a.files}, {k: b[k] for k in b.files})
                        except Exception as e:
                            audit.error('scores', sk, repr(e))
        if args.part == 'all':
            expected = read_json(out / 'cells.json')
            actual_cells_path = round_out / 'stage2/cells.json'
            actual_outcomes_path = round_out / 'stage2/outcomes.json'
            for path in (actual_cells_path, actual_outcomes_path):
                hashes[str(path.relative_to(round_out))] = hashlib.sha256(path.read_bytes()).hexdigest()
            actual_cells, actual_outcomes = read_json(actual_cells_path), read_json(actual_outcomes_path)
            expected_outcomes = read_json(out / 'outcomes.json')
            voided = actual_outcomes.get('voided', MISSING)
            audit.check('voided', 'voided list present', True, isinstance(voided, list), exact=True)
            if not isinstance(voided, list):
                voided = []
            declared = set()
            for entry in voided:
                valid = isinstance(entry, dict) and all(isinstance(entry.get(k), str) for k in ('what', 'key', 'why'))
                audit.check('voided', str(entry), True, valid, exact=True)
                if valid and entry['what'] == 'verdict':
                    sc = entry['key']
                    audit.check('voided', str(sc) + '/authorized', True, void_is_authorized(status, sc), exact=True)
                    if sc in expected_outcomes['verdicts']:
                        expected_outcomes['verdicts'][sc] = dict(P=None, T=None, Q=None, verdict='inconclusive')
                        declared.add(sc)
            for c in expected:
                c['claim_voided'] = c['sc'] in declared
            audit.check('cell_count', 'derived 76 cells', 76, len(expected), exact=True)
            compare_cells(audit, expected, actual_cells)
            audit.check('verdict_set', 'SC keys', sorted(expected_outcomes['verdicts']), sorted(actual_outcomes.get('verdicts', {})), exact=True)
            audit.tree('verdicts', expected_outcomes['verdicts'], actual_outcomes.get('verdicts', MISSING))
            closest = closest_to_bar(expected)
        audit.check('inputs', 'smoke setting', manifest['smoke'], P.SMOKE, exact=True)
    except Exception as e:
        audit.error('exception', 'compare', repr(e))
    head = subprocess.run(['git', '-C', str(ROUND), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    result = dict(audit.result(), sha256=hashes, commit=head.stdout.strip(), voided=voided,
                  closest_to_bar=closest, retired_result=retired)
    if head.returncode:
        audit.error('commit', 'HEAD', head.stderr.strip())
        result.update(audit.result())
    atomic_json(result_path, result)
    print(json.dumps({'pass': result['pass'], 'failed_total': result['failed_total'], 'categories': result['categories']}), flush=True)
    return 0 if result['pass'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    d = sub.add_parser('derive')
    d.add_argument('--part', choices=('val', 'held', 'all'), required=True)
    d.add_argument('--group', choices=('art', 'mm', 'all'), default='all')
    d.add_argument('--smoke', type=int, default=0)
    d.add_argument('--stores', type=Path)
    d.add_argument('--out', type=Path)
    c = sub.add_parser('compare')
    c.add_argument('--part', choices=('val', 'all'), required=True)
    c.add_argument('--round-out', type=Path)
    c.add_argument('--c13-out', type=Path)
    c.add_argument('--records', type=Path)
    args = parser.parse_args()
    if args.command == 'derive':
        assert args.smoke >= 0
        derive(args)
        return 0
    return compare(args)


if __name__ == '__main__':
    sys.exit(main())
