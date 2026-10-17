"""CPU scoring of frozen train-side views; checkpointed after every metric.

Use --reference-only for cached baselines. --stage caption runs the reproducibility
check first; --stage all (default) adds painting scores, probes and buddy graphs.
Missing Qwen keys are reported and skipped. Results accumulate by view; tables
are regenerated after every checkpoint. No labels enter feature construction.
"""
import argparse
import hashlib
import json
import time
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
from pav_sample import HERE, load_frozen
from pav_prompts import PROMPTS
from pav_extract import OUTPUT, identity, validate_key
from apc_data import l2, paint_first, paint_majority, paint_mean, STYLE_ROOT


def summary(xs):
    return dict(mean=float(np.mean(xs)), sd=float(np.std(xs)))


def partitions(x, k):
    return [MiniBatchKMeans(k, random_state=s, n_init=3, batch_size=4096).fit_predict(x)
            for s in (0, 1, 2)]


def probe(x, y, groups):
    mask = y >= 0
    x, y, groups = x[mask], y[mask], groups[mask]
    scores, baselines = [], []
    for fold, (tr, te) in enumerate(GroupKFold(5).split(x, y, groups), 1):
        mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-6
        clf = LogisticRegression(C=0.1, max_iter=300)
        clf.fit((x[tr] - mu) / sd, y[tr])
        scores.append(float((clf.predict((x[te] - mu) / sd) == y[te]).mean()))
        majority = np.bincount(y[tr]).argmax()
        baselines.append(float((y[te] == majority).mean()))
        print(f'    fold {fold}/5: accuracy={scores[-1]:.4f}, iterations={int(clf.n_iter_.max())}', flush=True)
    return dict(accuracy=summary(scores), majority_class_rate=float(np.bincount(y).max() / len(y)),
                training_majority_test_accuracy=summary(baselines), n=int(len(y)))


def mutual_pairs(x):
    """Exact cosine top-20, blocked to bound memory; remove self explicitly."""
    x = l2(x)
    n = len(x)
    adjacency = np.zeros((n, n), dtype=bool)
    for a in range(0, n, 256):
        b = min(a + 256, n)
        sim = x[a:b] @ x.T
        sim[np.arange(b - a), np.arange(a, b)] = -np.inf
        ix = np.argpartition(-sim, 19, axis=1)[:, :20]
        adjacency[np.arange(a, b)[:, None], ix] = True
    i, j = np.nonzero(np.triu(adjacency & adjacency.T, k=1))
    return set(zip(i.tolist(), j.tolist()))


def graph_stats(pairs, labels):
    ij = np.asarray(sorted(pairs), dtype=int).reshape(-1, 2)
    out = dict(n_pairs=len(pairs), labels={})
    for name, y in labels.items():
        counts = np.bincount(y[y >= 0])
        n = int(counts.sum())
        random_rate = float(np.sum(counts * (counts - 1)) / (n * (n - 1))) if n > 1 else None
        valid = np.all(y[ij] >= 0, axis=1)
        rate = float(np.mean(y[ij[valid, 0]] == y[ij[valid, 1]])) if valid.any() else None
        out['labels'][name] = dict(n_labelled_pairs=int(valid.sum()), same_label_rate=rate,
                                  random_same_label_rate=random_rate,
                                  lift=rate / random_rate if rate is not None and random_rate else None)
    return out


def references(s):
    from src.data.artelingo import load_artelingo
    from src.data.artelingo_splits import artelingo_splits
    d = load_artelingo()
    assert np.array_equal(d.sample_ids[s['rows']], s['sample_ids'])
    # Frozen train rows must still belong to the same split, independently of labels.
    sp = artelingo_splits(d)
    assert np.isin(s['rows'], np.concatenate([sp.scorer_train, sp.selection])).all()
    yield 'clip_text', l2(d.txt_features[s['rows']]), None
    yield 'clip_image', None, l2(d.img_features[s['rows'][s['first'][s['subset_pid']]]])
    for name, folder in [('csd_style', 'style_csd_vitl'), ('vgg_gram', 'style_vgg19_gram')]:
        r2i = np.load(STYLE_ROOT / folder / 'row_to_image.npy')
        emb = np.load(STYLE_ROOT / folder / 'embeddings.npy', mmap_mode='r')
        ii = r2i[s['rows']]
        assert np.array_equal(ii, ii[s['first']][s['pid']]), folder
        assert len(np.unique(ii[s['first']])) == len(s['first']), folder
        yield name, None, l2(np.asarray(emb[ii[s['first'][s['subset_pid']]]], dtype=np.float32))
    archive = HERE.parents[2] / 'archive/v2_conditioned/src/test'
    probs = np.full((len(d.sample_ids), 28), np.nan, dtype=np.float32)
    with np.load(archive / '20261018_affect_factor_learning/cache/affect_prepare.npz') as z:
        assert len(z['affect_probs']) == len(sp.scorer_train)
        probs[np.sort(sp.scorer_train)] = z['affect_probs']
    with np.load(archive / '20261123_idea3_goemotions/cache/r5_goemotions_selection.npz') as z:
        assert np.array_equal(z['rows'], np.sort(sp.selection))
        assert np.array_equal(z['sample_ids'], d.sample_ids[z['rows']])
        probs[z['rows']] = z['probs']
    x = probs[s['rows']]
    assert np.isfinite(x).all()
    yield 'goemotions', x, None  # raw probabilities, caption only (earlier check 4)


def markdown(r):
    lines = ['# Prompted aspect views', '', 'CPU scores; missing views are omitted. AMI and accuracy: mean ± population sd.', '',
             '## Caption emotion AMI', '', '| View | k=9 | k=20 | k=40 |', '|---|---:|---:|---:|']
    fmt = lambda v: f"{v['mean']:.5f} ± {v['sd']:.5f}" if v else '—'
    for name, v in r['views'].items():
        if 'caption_ami' in v:
            lines.append('| ' + name + ' | ' + ' | '.join(fmt(v['caption_ami'].get(str(k))) for k in (9, 20, 40)) + ' |')
    lines += ['', '## Shift: cosine to default', '', '| View | Image | Caption |', '|---|---:|---:|']
    for name, v in r['views'].items():
        if 'shift' in v:
            lines.append(f"| {name} | {v['shift'].get('image', '—')} | {v['shift'].get('caption', '—')} |")
    lines += ['', '## Painting view × label AMI (k=20)', '', '| View | Emotion | Genre | Style |', '|---|---:|---:|---:|']
    for name, v in r['views'].items():
        for side, vals in v.get('painting_ami', {}).items():
            lines.append(f'| {name}/{side} | ' + ' | '.join(fmt(vals[k]) for k in ('emotion', 'genre', 'style')) + ' |')
    lines += ['', '## Probes', '', '| View | Task | Accuracy | Majority rate |', '|---|---|---:|---:|']
    for name, v in r['views'].items():
        for task, val in v.get('probes', {}).items():
            lines.append(f"| {name} | {task} | {fmt(val['accuracy'])} | {val['majority_class_rate']:.5f} |")
    for side, p in r.get('pairwise', {}).items():
        lines += ['', f'## {side} partition AMI (seed 0)', '', '| View | ' + ' | '.join(p['names']) + ' |',
                  '|---|' + '---:|' * len(p['names'])]
        for i, name in enumerate(p['names']):
            lines.append('| ' + name + ' | ' + ' | '.join(f'{v:.4f}' for v in p['matrix'][i]) + ' |')
        lines += ['', '| View | Seed 0 vs 1 |', '|---|---:|']
        lines += [f'| {n} | {v:.4f} |' for n, v in p['seed0_vs_seed1'].items()]
    lines += ['', '## Buddy graphs (mutual cosine top-20)', '', '| Key | Graph | Pairs | Emotion lift | Genre lift | Style lift |', '|---|---|---:|---:|---:|---:|']
    for name, v in r['views'].items():
        for graph, val in v.get('buddies', {}).items():
            lifts = [val['labels'][k]['lift'] for k in ('emotion', 'genre', 'style')]
            lines.append(f"| {name} | {graph} | {val['n_pairs']} | " + ' | '.join('—' if x is None else f'{x:.4f}' for x in lifts) + ' |')
    return '\n'.join(lines) + '\n'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-only', action='store_true')
    p.add_argument('--keys', nargs='+', choices=list(PROMPTS))
    p.add_argument('--stage', choices=['all', 'caption'], default='all')
    p.add_argument('--no-references', action='store_true', help='score only selected Qwen keys')
    args = p.parse_args()
    s = load_frozen()
    result_file = HERE / 'results.json'
    r = json.loads(result_file.read_text()) if result_file.exists() else dict(
        schema=1, n_captions=99999, n_paintings=19889, n_subset=5000, views={}, pairwise={},
        conventions='MiniBatchKMeans n_init=3 batch=4096 seeds=0,1,2; LR C=.1 max_iter=300; GroupKFold(5); genre>=0')
    assert r['schema'] == 1
    fingerprint = hashlib.sha256(b''.join(np.ascontiguousarray(s[k]).tobytes() for k in
        ('sample_ids', 'pid', 'subset_pid', 'subset_image_indices', 'emo', 'gen', 'sty'))).hexdigest()
    assert r.get('sample_sha256', fingerprint) == fingerprint, 'results belong to a different sample'
    assert r.get('prompts', PROMPTS) == PROMPTS, 'results belong to different instructions'
    r['sample_sha256'] = fingerprint
    r['prompts'] = PROMPTS
    r['n_subset_captions'] = int(np.isin(s['pid'], s['subset_pid']).sum())
    def save():
        tmp = HERE / 'results.pending.json'
        tmp.write_text(json.dumps(r, indent=2, allow_nan=False))
        tmp.replace(result_file)
        (HERE / 'tables.md').write_text(markdown(r))
    subset = s['subset_pid']
    pcount = len(s['first'])
    labels = dict(emotion=paint_majority(s['emo'], s['pid'], pcount, 9)[subset],
                  genre=paint_first(s['gen'], s['pid'], pcount)[subset],
                  style=paint_first(s['sty'], s['pid'], pcount)[subset])
    cm = np.isin(s['pid'], subset)
    partitions_by_side = dict(image={}, mean_caption={})
    start = time.monotonic()
    def score(name, cap, image):
        print(f'[{time.monotonic()-start:.0f}s] {name}', flush=True)
        v = r['views'].setdefault(name, {})
        if cap is not None:
            out = v.setdefault('caption_ami', {})
            for k in (9, 20, 40):
                if str(k) not in out:
                    out[str(k)] = summary([ami(s['emo'], lab) for lab in partitions(cap, k)])
                    save()
                print(f"  caption k={k}: {out[str(k)]}", flush=True)
        if args.stage == 'caption':
            return
        views = {}
        if image is not None:
            views['image'] = image
        if cap is not None and name != 'goemotions':
            views['mean_caption'] = l2(paint_mean(cap, s['pid'], pcount)[subset])
        for side, x in views.items():
            stored = v.setdefault('partitions', {})
            if side not in stored:
                stored[side] = [lab.tolist() for lab in partitions(x, 20)]
                save()
            ls = [np.asarray(lab) for lab in stored[side]]
            partitions_by_side[side][name] = ls
            v.setdefault('painting_ami', {})[side] = {
                label: summary([ami(y[y >= 0], lab[y >= 0]) for lab in ls]) for label, y in labels.items()}
            save()
            for label, y in labels.items():
                task = f'{side}/{label}'
                if task not in v.setdefault('probes', {}):
                    v['probes'][task] = probe(x, y, subset)
                    save()
                print(f"  probe {task}: {v['probes'][task]['accuracy']}", flush=True)
        if cap is not None:
            task = 'caption/emotion'
            if task not in v.setdefault('probes', {}):
                v['probes'][task] = probe(cap[cm], s['emo'][cm], s['pid'][cm])
                save()
            print(f"  probe {task}: {v['probes'][task]['accuracy']}", flush=True)
        if 'image' in views and 'mean_caption' in views and 'buddies' not in v:
            a, b = mutual_pairs(views['image']), mutual_pairs(views['mean_caption'])
            v['buddies'] = {n: graph_stats(pairs, labels) for n, pairs in
                            [('image', a), ('mean_caption', b), ('strict', a & b), ('union', a | b)]}
            save()
    if not args.no_references:
        for name, cap, image in references(s):
            # CLIP's two reference sides also form one paired buddy graph.
            if name == 'clip_text':
                clip_cap = cap
            score(name, cap, image)
            if name == 'clip_image' and args.stage == 'all':
                v = r['views']['clip_image']
                if 'buddies' not in v:
                    a = mutual_pairs(image)
                    b = mutual_pairs(l2(paint_mean(clip_cap, s['pid'], pcount)[subset]))
                    v['buddies'] = {n: graph_stats(ps, labels) for n, ps in
                                    [('image', a), ('mean_caption', b), ('strict', a & b), ('union', a | b)]}
                    save()
    if not args.reference_only:
        expected = identity(s, 5000, 99999)
        default = None
        if all((OUTPUT / 'default' / f).exists() for f in ('img.npy', 'txt.npy', 'ids.npz', 'done.json')):
            default = validate_key(OUTPUT / 'default', expected, PROMPTS['default'])
        for key in args.keys or PROMPTS:
            path = OUTPUT / key
            if not all((path / f).exists() for f in ('img.npy', 'txt.npy', 'ids.npz', 'done.json')):
                print(f'{key}: missing/incomplete; skip', flush=True)
                continue
            image, cap = validate_key(path, expected, PROMPTS[key])
            v = r['views'].setdefault(key, {})
            if default is not None:
                shift = {}
                for side, x, d in [('image', image, default[0]), ('caption', cap, default[1])]:
                    total = 0.
                    for a in range(0, len(x), 4096):
                        total += float(np.sum(l2(x[a:a+4096].astype('float32')) * l2(d[a:a+4096].astype('float32')), dtype=np.float64))
                    shift[side] = total / len(x)
                v['shift'] = shift
                save()
            else:
                print(f'{key}: default missing; shift unavailable', flush=True)
            score(key, np.asarray(cap, dtype=np.float32), np.asarray(image, dtype=np.float32))
    if args.stage == 'all':
        for side, views in partitions_by_side.items():
            names = list(views)
            matrix = [[float(ami(views[a][0], views[b][0])) for b in names] for a in names]
            r['pairwise'][side] = dict(names=names, matrix=matrix,
                seed0_vs_seed1={n: float(ami(ls[0], ls[1])) for n, ls in views.items()})
    save()
    print(f'Finished in {time.monotonic()-start:.1f}s; {result_file}', flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=8):
        main()
