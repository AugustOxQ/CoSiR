"""Held ArtELingo ablation retrieval, plain cosine (R0 / none+k0).

Full bootstrap: unchanged ns_ret.paired_ret, shared training-seed draws.
Smoke: first 2,000 held names, emotion, two seeds; equivalent shared bootstrap
with two seeds because ns_ret.paired_ret explicitly requires five.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TEST = HERE.parent
sys.path.insert(0, str(TEST / '20261027_recipe_sweep'))
import rs_paths  # noqa: E402,F401 -- older rounds, as r10/conftest.py

sys.path.insert(0, str(TEST / '20261103_new_scores'))
import ns_ret  # noqa: E402
import rb_encoders  # noqa: E402
import rb_part2_report  # noqa: E402
import rt_data  # noqa: E402
import rt_metrics  # noqa: E402

EXPORTS = Path('/data/SSD2/cosir_r10/20261027_recipe_sweep/outputs/full/held')
REFERENCE = Path('/data/SSD2/cosir_r11/20261029_retrieval/outputs/stage1/verdict.json')
ASPECTS = ('emotion', 'style', 'genre')
REMOVALS = ('buddy', 'caption_endpoints', 'content_correction', 'direct',
            'instructed_graphs', 'relational', 'shared_trunk', 'sharp_row')
ARMS = ('k11=40',) + tuple('k11=40-no_' + r for r in REMOVALS)
SEEDS = tuple(range(5))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def export_path(arm, aspect, seed):
    return EXPORTS / arm / f'{aspect}__s{seed}.npz'


def check_sidecars(list_sha):
    """Check all 135 sidecars, including in smoke; bind positional rows to names."""
    for arm in ARMS:
        for aspect in ASPECTS:
            for seed in SEEDS:
                path = export_path(arm, aspect, seed)
                meta = json.loads(path.with_suffix('.json').read_text())
                assert path.is_file(), path
                assert meta['rows'] == 12281 and meta['list_sha256'] == list_sha, path
                assert (meta['arm'], meta['aspect'], meta['seed']) == (arm, aspect, seed), path


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    assert x.ndim == 2 and np.isfinite(x).all()
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    assert (norms > 0).all(), 'zero image row'
    return x / np.maximum(norms, 1e-12)


def load_export(arm, aspect, seed, n, hashes):
    path = export_path(arm, aspect, seed)
    hashes[str(path)] = sha256(path)
    hashes[str(path.with_suffix('.json'))] = sha256(path.with_suffix('.json'))
    with np.load(path, allow_pickle=False) as z:
        x = z['img']
    assert x.shape == (12281, 2048) and x.dtype == np.float16, path
    return unit(x[:n])


def score_rows(x, labels, names, workers=8):
    """Existing AP; convert its positional query indices into actual painting IDs."""
    names = np.asarray(names)
    assert names.ndim == 1 and len(names) == len(x) == len(labels)
    assert names.dtype.kind in 'US' and len(set(names.tolist())) == len(names)
    assert np.asarray(labels).dtype == np.int64
    res = rt_metrics.retrieval(x, labels, qe_k=0, workers=workers)
    assert np.isfinite(res['ap']).all()
    return dict(ap=res['ap'], query_ids=names[res['rows']], dropped=res['dropped'])


def same_queries(left, right):
    assert np.array_equal(left['query_ids'], right['query_ids']), 'query painting order differs'
    assert left['dropped'] == right['dropped'], 'dropped queries differ'


def paired(cand, comp, draws=10000, smoke=False):
    same_queries(cand, comp)
    a, b = cand['ap'], comp['ap']
    assert a.shape == b.shape and a.dtype == b.dtype == np.float64
    if not smoke:
        p = ns_ret.paired_ret(a, b, r=None, draws=draws, comp_seeds='shared', rng_seed=0)
        return {k: p[k] for k in ('d', 'lo', 'hi')}
    # Only the smoke adapter: same I-then-S RNG order as ns_ret._boot, two seeds.
    assert a.ndim == 2 and a.shape[0] == 2
    rng = np.random.default_rng(0)
    boot = np.empty(draws)
    n = a.shape[1]
    for draw in range(draws):
        rows = rng.integers(0, n, size=n)
        seeds = rng.choice([0, 1], size=2, replace=True)
        boot[draw] = a[np.ix_(seeds, rows)].mean() - b[np.ix_(seeds, rows)].mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(d=float(a.mean() - b.mean()), lo=float(lo), hi=float(hi))


def summary(res, seeded=True):
    ap = res['ap']
    per_seed = ap.mean(axis=1).tolist() if seeded else None
    return dict(map=float(np.mean(per_seed)) if seeded else float(ap.mean()),
                per_seed=per_seed, n_queries=len(res['query_ids']), dropped=res['dropped'])


def sanity(point, aspect):
    ref = json.loads(REFERENCE.read_text())['map']['cstar'][aspect]
    delta = point - ref['mean']
    return dict(source=str(REFERENCE), aspect=aspect, recorded=ref['mean'],
                computed=point, difference=delta, matches_1e_6=abs(delta) <= 1e-6)


def markdown(doc):
    m = doc['metadata']
    lines = [f"Read-out: R0 / none+k0, plain cosine on unit float32 rows; no map or query expansion. "
             f"Seeds: {m['seeds']}; shared query/seed bootstrap, {m['draws']} draws (95%). "
             f"Held list: 12,281 rows; scored rows: {m['n_rows']}. "
             f"Labelled full rows: emotion 12,281, style 12,281, genre 9,949. "
             f"Smoke: {m['smoke']}. Queries exclude self; singleton queries are dropped.", '']
    if m['smoke']:
        lines += ['Smoke uses two-seed shared resampling; full scoring uses ns_ret.paired_ret unchanged.', '']
    for aspect, result in doc['aspects'].items():
        lines += [f"### {aspect} ({result['arms']['k11=40']['n_queries']:,} queries)", '',
                  '| arm | mAP | Δ [95%] |', '|---|---:|---:|']
        for arm, row in {**result['arms'], **result['context']}.items():
            p = row.get('comparison')
            delta = f"{p['d']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}]" if p else '—'
            lines.append(f"| {arm} | {row['map']:.4f} | {delta} |")
        lines.append('')
    lines.append('Δ = removal minus c*. Context rows are unseeded.')
    return '\n'.join(lines) + '\n'


def run(out, smoke=False, workers=8):
    start = time.monotonic()
    out = Path(out).resolve()
    assert out.is_relative_to(HERE), f'output must be inside {HERE}'
    out.mkdir(parents=True, exist_ok=True)
    names, list_sha, labels = rb_part2_report.held_labels()
    names = np.asarray(names)
    assert len(names) == 12281 and names.tolist() == sorted(names.tolist())
    assert hashlib.sha256('\n'.join(names).encode()).hexdigest() == list_sha
    assert {a: int((labels[a] >= 0).sum()) for a in ASPECTS} == dict(emotion=12281, style=12281, genre=9949)
    check_sidecars(list_sha)
    n, seeds, draws = (2000, (0, 1), 500) if smoke else (12281, SEEDS, 10000)
    aspects, arms = (ASPECTS[:1], ARMS[:2]) if smoke else (ASPECTS, ARMS)
    hashes = {}
    doc = dict(metadata=dict(readout='R0 / none+k0', ap_path='rt_metrics.retrieval(qe_k=0)',
                             bootstrap_path='ns_ret.paired_ret' if not smoke else 'two-seed smoke adapter',
                             comp_seeds='shared', seed_pairing='same seed passed to train_cohort for every r10 arm',
                             seeds=list(seeds), draws=draws, rng_seed=0, n_rows=n,
                             list_sha256=list_sha, smoke=smoke, workers=workers),
               code_sha256=sha256(__file__), input_sha256=hashes, aspects={})
    doc['dependency_sha256'] = {str(Path(mod.__file__)): sha256(mod.__file__)
                                for mod in (rt_metrics, ns_ret, rb_part2_report, rb_encoders, rt_data)}
    # Plain loader joins by painting name. Teacher loader checks held IDs/hash before gathering.
    plain = rb_encoders.full_images('qwen3vl_emb_2b', names[:n])
    plain_root = rb_encoders.FULL_ROOT / 'qwen3vl_emb_2b'
    for filename in ('img.npy', 'index.json'):
        path = plain_root / filename
        hashes[str(path)] = sha256(path)
    for aspect in aspects:
        y = np.asarray(labels[aspect][:n], dtype=np.int64)
        results, base = {}, None
        for arm in arms:
            scored = [score_rows(load_export(arm, aspect, s, n, hashes), y, names[:n], workers) for s in seeds]
            for row in scored[1:]:
                same_queries(scored[0], row)
            res = dict(scored[0], ap=np.stack([r['ap'] for r in scored]))
            row = summary(res)
            if base is None:
                base = res
            else:
                row['comparison'] = paired(res, base, draws, smoke)
            results[arm] = row
            print(f"{aspect} {arm}: mAP={row['map']:.10f} n={row['n_queries']}", flush=True)
        teacher = rt_data.embed('instructed', 'held', aspect)[:n]
        teacher_root = rt_data.FULL_ROOT / f'qwen3vl_emb_2b_aspect_prompts/{rt_data.PROMPTS[aspect]}_held'
        for filename in ('img.npy', 'ids.npz', 'meta.json'):
            path = teacher_root / filename
            hashes[str(path)] = sha256(path)
        context = {}
        for key, x in (('plain image', plain), ('teacher', teacher)):
            res = score_rows(x, y, names[:n], workers)
            same_queries(base, res)
            context[key] = summary(res, seeded=False)
        doc['aspects'][aspect] = dict(arms=results, context=context, query_ids=base['query_ids'].tolist(),
                                      n_labelled=int((y >= 0).sum()))
        if not smoke:
            doc['aspects'][aspect]['sanity_check'] = sanity(results['k11=40']['map'], aspect)
    doc['seconds'] = time.monotonic() - start
    (out / 'ablation_ret.json').write_text(json.dumps(doc, indent=2, allow_nan=False) + '\n')
    (out / 'ablation_ret.md').write_text(markdown(doc))
    print(f"{doc['seconds']:.2f}s -> {out / 'ablation_ret.json'}", flush=True)
    return doc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--workers', type=int, default=8, choices=range(1, 9))
    args = parser.parse_args()
    run(args.out, args.smoke, args.workers)


if __name__ == '__main__':
    main()
