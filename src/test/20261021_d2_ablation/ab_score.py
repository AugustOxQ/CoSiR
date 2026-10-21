"""Scorer and attribution table for the r10 D2 ablation (spec 3.4, 3.5; D5, D7, D16, D18, D19). CPU only.

Reads every trained row in <outputs>/train/<row>/ (ms_train output files plus ab_row.json) and writes
<outputs>/scores/scores.json and <outputs>/scores/attribution.md.

Per row, aspect, side (img, cap) and model seed 0 to 2:
  k-means k = 20 on random states 0 to 2: AMI, NMI (v-measure), ARI, FMI via `ms_score.measures` (the model sheet's own
  scorer, cache under <outputs>/scores/cache); the row value is the mean over random states, then over seeds, with the
  population sd over seeds (`ms_score.aggregate`'s rule);
  Leiden reading: union 20-NN graph, resolution 1.0 (modularity), Leiden seeds 0 to 2, AMI with the own label;
  probe (D18): ab_metrics.probe on the image side of every seed and the caption side of seed 0.
Effect (D19): image-side k-means AMI of a row minus R0's, percentile 95% interval from paired draws. The draw list is
built once per aspect with `default_rng(0)` and is the same for every row: each draw resamples the labelled evaluation
paintings (one list shared by all rows) and, for every row in table order R0 to R12, the model seeds with replacement.

Reuse note: the claim test's `ct_score.draw_bootstrap`, `score_indices` and `paired_bootstrap` assert the arm names
K1, B1 to B5 and read one fixed arm against the best of the others, so they cannot take R0 to R12. `draw_list` and
`bootstrap_effects` below keep their scheme (default_rng(0); rows then seeds per arm in table order; percentile 2.5 and
97.5; fork pool with one BLAS thread per worker) and are generalized to named rows and a per-row reference.

R7's caption side is "not scored" (D16; `caption_side` in ab_row.json). A row with missing files is listed as missing and
the rest are scored (spec 3.5 stage 1). No verdict code: the table is descriptive (spec 9).
"""
import argparse
import hashlib
import inspect
import json
import multiprocessing
import sys
import time
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import adjusted_mutual_info_score as ami
from threadpoolctl import threadpool_limits

ROUND = Path(__file__).resolve().parent
MAIN_ROUND = Path('/project/CoSiR/src/test/20261021_d2_ablation')
OUTPUTS = MAIN_ROUND / 'outputs'
SAMPLE = Path('/project/CoSiR/src/test/20261017_prompted_aspect_views/sample.npz')
BACKFILL = Path('/project/CoSiR/src/test/20261019_model_sheet/outputs/backfill.json')
_FOLDERS = ('20261017_aspect_premise_checks', '20261017_prompted_aspect_views', '20261017_image_transfer_pilot',
            '20261018_purified_buddy_pilot', '20261018_aspect_graph_sheet', '20261018_percept_core',
            '20261019_model_sheet', '20261020_claim_test')
for _folder in _FOLDERS:
    sys.path.insert(0, str(ROUND.parent / _folder))
sys.path.insert(0, str(ROUND))
import ab_metrics
import ct_diag
import ms_score
import pav_score
from itp_data import atomic_json, fingerprint
from pbp_graphs import label_vectors, sha_file

for _module in (ab_metrics, ct_diag, ms_score, pav_score):  # frozen code must come from this checkout
    assert Path(_module.__file__).resolve().is_relative_to(ROUND.parent.resolve()), _module.__file__

ROW_ORDER = tuple(f'R{i}' for i in range(13))
ASPECTS = ('emotion', 'style', 'genre')
SEEDS = (0, 1, 2)
REFERENCE_ROW = 'R0'
SCHEMA = 1
CHANGE = dict(  # spec 3.2, "Change from D2"
    R0='none (D2)', R1='separate heads, no shared trunk', R2='no distillation (= T1)',
    R3='teacher = default Qwen image', R4='distillation only, no graph terms',
    R5='no buddy terms', R6='no own-caption terms', R7='image endpoints only (no caption head)',
    R8='plain instructed graphs (R0, R0o)', R9='one default-view graph', R10='random pairs, degrees kept',
    R11='R9 and R2: no instruction anywhere', R12='instructed inputs on both sides')
NOTES = ('kmeans.nmi is sklearn v_measure_score (equal to NMI with arithmetic mean). Row values: mean over k-means random states, '
         'then over the 3 model seeds; sd is the population sd over model seeds. Leiden: union 20-NN graph, modularity '
         '(resolution 1), Leiden seeds 0 to 2, same averaging. Probe: ab_metrics.probe on 1 BLAS thread per worker '
         '(balanced accuracy can differ from an 8-thread run in the 3rd decimal). Effects: row minus R0 on the image-side '
         'k-means AMI, percentile 95% interval over paired draws.')


# ------------------------------------------------------------------------------------------------ reading rows
def load_sample(path=SAMPLE):
    with np.load(path, allow_pickle=False) as z:
        s = {k: z[k] for k in z.files}
    assert (len(s['sample_ids']), len(s['first']), len(s['subset_pid'])) == (99999, 19889, 5000)
    assert np.array_equal(s['all_painting_ids'][s['subset_pid']], s['subset_painting_ids'])
    return s


def find_row(outputs, row, seeds=SEEDS, aspects=ASPECTS):
    """Locate a row's files without loading arrays. Returns (info, reasons); reasons non-empty means missing."""
    folder = Path(outputs) / 'train' / row
    reasons = []
    if not folder.is_dir():
        return None, [f'no folder train/{row}']
    meta_path = folder / 'ab_row.json'
    meta = None
    if not meta_path.exists():
        reasons.append('no ab_row.json')
    else:
        meta = json.loads(meta_path.read_text())
        if 'name' in meta:
            assert meta['name'] == row, f'{meta_path}: name {meta["name"]} is not {row}'
        assert meta.get('caption_side', 'scored') in ('scored', 'not scored'), meta_path
    files = {}
    for aspect in aspects:
        for seed in seeds:
            found = sorted(folder.glob(f'{aspect}__*__s{seed}.npz'))
            if len(found) != 1:
                reasons.append(f'{aspect} seed {seed}: {len(found)} npz files')
            elif not found[0].with_suffix('.json').exists():
                reasons.append(f'{found[0].name}: no json')
            else:
                files[(aspect, seed)] = found[0]
    if reasons:
        return None, reasons
    return dict(row=row, meta=meta, files=files, caption_scored=meta.get('caption_side', 'scored') == 'scored'), []


def read_meta(info, aspect, seed, sample_sha):
    """Cheap asserts on one output's json and file; returns the json."""
    path = info['files'][(aspect, seed)]
    meta = json.loads(path.with_suffix('.json').read_text())
    config = meta['config']
    assert config['aspect'] == aspect and config['seed'] == seed, path
    assert config['sample_sha256'] == sample_sha, f'{path}: sample sha256 differs'
    assert path.name == f'{aspect}__{config.get("graph_row") or "none"}__s{seed}.npz', path
    assert sha_file(path) == meta['output_sha256'], f'{path}: output checksum mismatch'
    assert 1 <= meta['best_epoch'] <= meta['epochs_run'], path
    return meta


def check_vectors(path):
    """Both sides of one npz, checked: shape (5000, 2048), finite, unit norm within .01, name set."""
    with np.load(path, allow_pickle=False) as z:
        assert set(z.files) == {'img', 'cap'}, path
        out = {}
        for side in ('img', 'cap'):
            assert z[side].shape == (5000, 2048), (path, side, z[side].shape)
            out[side] = ms_score.check_view(z[side])
    return out


# ------------------------------------------------------------------------------------------------ per-vector scoring
def _agg(values):
    values = [float(v) for v in values]
    return dict(mean=float(np.mean(values)), sd=float(np.std(values)), per_seed=values)


def _own_ami(y, partition_set):
    mask = y >= 0
    return [float(ami(y[mask], p[mask])) for p in partition_set]


def _code_identity():
    """Binds a cached vector score to the code that made it (the per-vector functions and what they call)."""
    mine = ''.join(inspect.getsource(f) for f in (check_vectors, _own_ami, _score_vector, score_task))
    parts = [Path(f).read_bytes() for f in (ab_metrics.__file__, ct_diag.__file__)]
    return hashlib.sha256(b''.join(parts) + mine.encode() + ms_score.SCORER_VERSION.encode()
                          + sklearn.__version__.encode()).hexdigest()


_W = {}


def _init_worker(labels, cache_dir):
    _W.update(labels=labels, cache=Path(cache_dir))
    threadpool_limits(limits=1)
    # ms_score.measures re-opens an 8 thread scope around k-means; the partitions do not depend on the thread count
    # (checked: identical on 1 and 8 threads), so keep one thread per worker.
    ms_score.threadpool_limits = lambda *a, **k: threadpool_limits(limits=1)


def _score_vector(x, y, aspect, cache_dir, probe, keep_partitions):
    km = ms_score.measures(x, {aspect: y}, cache_dir)[aspect]
    graph = ct_diag.union_knn(x, 20)
    leiden = _own_ami(y, ct_diag.leiden_partitions(graph))
    out = dict(kmeans={m: dict(v) for m, v in km.items()}, leiden_ami=leiden)
    if keep_partitions:  # deterministic: the same partitions `measures` scored
        parts = np.asarray(pav_score.partitions(x, 20))
        got = float(np.mean(_own_ami(y, parts)))
        assert abs(got - km['ami']['mean']) < 1e-9, (got, km['ami'])
        out['partitions'] = parts.tolist()
    if probe:
        r = ab_metrics.probe(x, y)
        out['probe'] = dict(balanced_accuracy=r['balanced_accuracy'], per_class_recall=r['per_class_recall'].tolist(),
                            classes=r['classes'].tolist(), majority_rate=r['majority_rate'], chance=r['chance'],
                            n=r['n'], convergence_warnings=r['convergence_warnings'])
    return out


def score_task(task):
    """One (row, aspect, seed): both sides of one npz. Cached on the file hash and the code."""
    row, aspect, seed, path, output_sha, caption_scored = task
    y = _W['labels'][aspect]
    identity = dict(npz_sha256=output_sha, aspect=aspect, seed=seed, caption_scored=caption_scored,
                    code=_code_identity(), label=ms_score.array_sha(y), schema=SCHEMA)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = _W['cache'] / f'task_{key}.json'
    if cache.exists():
        r = json.loads(cache.read_text())
        assert r['identity'] == identity
        return (row, aspect, seed), r['result']
    vectors = check_vectors(Path(path))
    result = dict(img=_score_vector(vectors['img'], y, aspect, _W['cache'], True, True))
    if caption_scored:
        result['cap'] = _score_vector(vectors['cap'], y, aspect, _W['cache'], seed == 0, False)
    atomic_json(cache, dict(identity=identity, result=result))
    return (row, aspect, seed), result


def summarize_row(per_seed, caption_scored):
    """Combine the three seeds' task results of one (row, aspect) into the row's JSON entry."""
    assert sorted(per_seed) == list(SEEDS)
    entry = {}
    for side in ('img', 'cap'):
        if side == 'cap' and not caption_scored:
            entry['cap'] = 'not scored'
            continue
        seeds = [per_seed[s][side] for s in SEEDS]
        side_entry = dict(kmeans={('nmi' if m == 'v' else m): _agg([r['kmeans'][m]['mean'] for r in seeds])
                                  for m in seeds[0]['kmeans']},
                          leiden_ami=_agg([np.mean(r['leiden_ami']) for r in seeds]),
                          leiden_ami_seeds=[r['leiden_ami'] for r in seeds])
        probed = {s: per_seed[s][side]['probe'] for s in SEEDS if 'probe' in per_seed[s][side]}
        side_entry['probe'] = dict(
            seeds=sorted(probed), balanced_accuracy=_agg([probed[s]['balanced_accuracy'] for s in sorted(probed)]),
            per_class_recall={str(s): probed[s]['per_class_recall'] for s in sorted(probed)},
            classes=probed[min(probed)]['classes'], majority_rate=probed[min(probed)]['majority_rate'],
            chance=probed[min(probed)]['chance'],
            convergence_warnings={str(s): probed[s]['convergence_warnings'] for s in sorted(probed)})
        entry[side] = side_entry
    return entry


# ------------------------------------------------------------------------------------------------ paired bootstrap
def draw_list(labelled, n_resamples=2000, rows=ROW_ORDER, seeds=SEEDS):
    """The whole RNG transcript for one aspect, before any AMI is scheduled; independent of which rows exist.

    `labelled` = indices of the labelled evaluation paintings. Each draw: painting indices resampled with replacement
    from `labelled`, then for each row in table order (all of R0 to R12, present or not) the model seeds resampled.
    """
    labelled = np.asarray(labelled)
    assert labelled.ndim == 1 and len(labelled) > 0 and n_resamples > 0
    rng = np.random.default_rng(0)
    draws = []
    for _ in range(n_resamples):
        idx = labelled[rng.integers(0, len(labelled), size=len(labelled))]
        chosen = {row: rng.choice(list(seeds), size=len(seeds), replace=True) for row in rows}
        draws.append((idx, chosen))
    return draws


def _draw_values(parts, y, draw):
    idx, chosen = draw
    yy = y[idx]
    out = {}
    for row, by_seed in parts.items():
        cache = {}
        values = []
        for seed in chosen[row]:
            seed = int(seed)
            if seed not in cache:
                cache[seed] = float(np.mean([ami(yy, p[idx]) for p in by_seed[seed]]))
            values.append(cache[seed])
        out[row] = float(np.mean(values))
    return out


_BP = {}


def _init_bootstrap(parts, y):
    _BP.update(parts=parts, y=y)
    threadpool_limits(limits=1)


def _bootstrap_one(draw):
    return _draw_values(_BP['parts'], _BP['y'], draw)


def bootstrap_effects(parts, y, draws, ref=REFERENCE_ROW, workers=8):
    """Effect of every row against `ref` on the image-side AMI, with a paired percentile interval.

    parts[row][seed] = (3, n) k-means partitions of the evaluation paintings; y = labels (negative = unlabelled).
    A row against itself: same seed draws on both sides, so the effect is exactly 0 in every draw.
    """
    assert 1 <= workers <= 8
    labelled = np.flatnonzero(y >= 0)
    full = {row: float(np.mean([np.mean(_own_ami(y, by_seed[s])) for s in SEEDS])) for row, by_seed in parts.items()}
    if workers == 1 or len(draws) < 2:
        values = [_draw_values(parts, y, d) for d in draws]
    else:
        with multiprocessing.get_context('fork').Pool(workers, initializer=_init_bootstrap, initargs=(parts, y)) as pool:
            values = pool.map(_bootstrap_one, draws, chunksize=max(1, len(draws) // (workers * 8)))
    result = {}
    for row in parts:
        if ref not in parts:
            result[row] = dict(point_ami=full[row], effect=None)
            continue
        diffs = np.array([v[row] - v[ref] for v in values])
        lo, hi = np.percentile(diffs, [2.5, 97.5]).tolist()
        result[row] = dict(point_ami=full[row], effect=full[row] - full[ref], lo=lo, hi=hi,
                           resamples=len(draws), draw_sd=float(diffs.std()))
    return result


# ------------------------------------------------------------------------------------------------ reference lines
def reference_lines():
    """Default Qwen image, instructed Qwen image, PercepT core, per aspect, with their sources (no training)."""
    md = (Path(ms_score.__file__).resolve().parent / 'ms_tables_final.md').read_text()
    bars, section = {}, None
    for line in md.splitlines():
        if line.startswith('## '):
            section = line[3:].strip()
        elif section in ASPECTS and line.startswith('| Bars |'):
            cells = [c.strip() for c in line.strip('|').split('|')]
            bars[(section, cells[1])] = cells[4]
    refs = {}
    backfill = json.loads(BACKFILL.read_text())['rows']['qwen_default']
    for a in ASPECTS:
        qd = backfill[a]['measures']['ami']
        if a != 'style':  # the sheet's image_encoder bar is Qwen default for emotion and genre, CLIP for style
            assert bars[(a, 'image_encoder')].startswith(f'{qd["mean"]:.6f}'), (a, bars[(a, 'image_encoder')], qd)
        refs[a] = {
            'default Qwen image': dict(text=f'{qd["mean"]:.6f} ± {qd["sd"]:.6f}', source=f'{BACKFILL} rows.qwen_default'),
            'instructed Qwen image': dict(text=bars[(a, 'target')],
                                          source="ms_tables_final.md, section %s, Bars target" % a),
            f'PercepT core ({ms_score.BEST[a]})': dict(text=bars[(a, 'percept')],
                                                       source="ms_tables_final.md, section %s, Bars percept" % a)}
    return refs


# ------------------------------------------------------------------------------------------------ report
def _pm(d, nd=6):
    return f'{d["mean"]:.{nd}f} ± {d["sd"]:.{nd}f}'


def _effect(e):
    if not e or e.get('effect') is None:
        return 'n/a'
    return f'{e["effect"]:+.6f} [{e["lo"]:+.6f}, {e["hi"]:+.6f}]'


def attribution_markdown(result, refs):
    out = ['# D2 ablation: attribution table', '',
           f'Scored {result["written"]}. Image-side k-means (k = 20) AMI with the own label, mean over random states 0 to 2, '
           f'then over model seeds 0 to 2 (± population sd over seeds). Effect = row minus R0, with a percentile 95% '
           f'interval from {result["resamples"]} paired draws over the labelled evaluation paintings and model seeds '
           '(the same draws for every row). Leiden: union 20-NN graph, resolution 1.0, 3 seeds, no interval. Probe: '
           'image-side balanced accuracy (5-fold logistic regression), mean over seeds. Best epochs per seed. '
           'Descriptive; nothing here decides anything.', '']
    if result['missing']:
        out += ['Missing rows: ' + '; '.join(f'{r} ({", ".join(v)})' for r, v in result['missing'].items()), '']
    for a in ASPECTS:
        block = result['aspects'][a]
        out += [f'## {a}', '', f'Labelled evaluation paintings: {block["n_labelled"]}.', '',
                '| Row | Change from D2 | Image AMI | Effect vs R0 [95% interval] | Caption AMI | Leiden AMI (img) | '
                'Probe bal. acc. (img) | Best epochs |', '|---|---|---:|---:|---:|---:|---:|---|']
        for row in ROW_ORDER:
            if row in result['missing']:
                out.append(f'| {row} | {CHANGE[row]} | missing | | | | | |')
                continue
            if row not in block['rows']:
                continue
            e = block['rows'][row]
            cap = e['cap'] if e['cap'] == 'not scored' else _pm(e['cap']['kmeans']['ami'])
            epochs = ', '.join(f's{s}:{ep}' for s, ep in e['best_epoch'].items())
            out.append(f'| {row} | {CHANGE[row]} | {_pm(e["img"]["kmeans"]["ami"])} | {_effect(e["effect"])} | {cap} | '
                       f'{_pm(e["img"]["leiden_ami"])} | {e["img"]["probe"]["balanced_accuracy"]["mean"]:.4f} | {epochs} |')
        out += ['']
        for name, ref in refs[a].items():
            out.append(f'- Reference, no training: {name}: image AMI {ref["text"]} (source: {ref["source"]}).')
        out += ['']
    return '\n'.join(out) + '\n'


# ------------------------------------------------------------------------------------------------ main
def run(outputs, rows=ROW_ORDER, workers=8, resamples=2000, aspects=ASPECTS):
    start = time.monotonic()
    outputs = Path(outputs)
    cache_dir = outputs / 'scores' / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = [r for r in ROW_ORDER if r in rows]
    s = load_sample()
    sample_sha = fingerprint(s)
    labels_all = label_vectors(s, s['subset_pid'])
    labels = {a: labels_all[a] for a in aspects}
    present, missing = {}, {}
    for row in rows:
        info, reasons = find_row(outputs, row, aspects=aspects)
        if reasons:
            missing[row] = reasons
        else:
            present[row] = info
    metas = {row: {(a, sd): read_meta(info, a, sd, sample_sha) for (a, sd) in info['files']} for row, info in present.items()}
    tasks = [(row, a, sd, str(info['files'][(a, sd)]), metas[row][(a, sd)]['output_sha256'], info['caption_scored'])
             for row, info in present.items() for a in aspects for sd in SEEDS]
    print(f'{len(present)} rows present, {len(missing)} missing; {len(tasks)} vector sets', flush=True)
    done = {}
    if workers == 1:
        _init_worker(labels, cache_dir)
        for t in tasks:
            k, r = score_task(t)
            done[k] = r
    else:
        with multiprocessing.get_context('fork').Pool(workers, initializer=_init_worker,
                                                      initargs=(labels, cache_dir)) as pool:
            for k, r in pool.imap_unordered(score_task, tasks):
                done[k] = r
                print(f'  scored {k} ({len(done)}/{len(tasks)}, {time.monotonic() - start:.0f}s)', flush=True)
    result = dict(schema=SCHEMA, written=time.strftime('%Y-%m-%d %H:%M'), outputs=str(outputs), sample_sha256=sample_sha,
                  resamples=resamples, rng='numpy.random.default_rng(0) per aspect', workers=workers, notes=NOTES,
                  rows_requested=rows, rows_scored=list(present), missing=missing, aspects={})
    for a in aspects:
        y = labels[a]
        labelled = np.flatnonzero(y >= 0)
        entries = {}
        for row, info in present.items():
            per_seed = {sd: done[(row, a, sd)] for sd in SEEDS}
            entry = summarize_row(per_seed, info['caption_scored'])
            entry['best_epoch'] = {str(sd): metas[row][(a, sd)]['best_epoch'] for sd in SEEDS}
            entry['epochs_run'] = {str(sd): metas[row][(a, sd)]['epochs_run'] for sd in SEEDS}
            entry['graph_row'] = {str(sd): metas[row][(a, sd)]['config'].get('graph_row') for sd in SEEDS}
            entry['caption_side'] = 'scored' if info['caption_scored'] else 'not scored'
            entries[row] = entry
        parts = {row: {sd: np.asarray(done[(row, a, sd)]['img']['partitions']) for sd in SEEDS} for row in present}
        t0 = time.monotonic()
        effects = bootstrap_effects(parts, y, draw_list(labelled, resamples), workers=workers) if parts else {}
        print(f'bootstrap {a}: {time.monotonic() - t0:.0f}s', flush=True)
        for row, e in effects.items():
            entries[row]['effect'] = dict(vs=REFERENCE_ROW, **e)
            assert abs(e['point_ami'] - entries[row]['img']['kmeans']['ami']['mean']) < 1e-9
        result['aspects'][a] = dict(n_labelled=int(len(labelled)), rows=entries)
    result['reference_lines'] = reference_lines()
    scores = outputs / 'scores'
    atomic_json(scores / 'scores.json', result)
    md = attribution_markdown(result, result['reference_lines'])
    pending = scores / 'attribution.pending.md'
    pending.write_text(md)
    pending.replace(scores / 'attribution.md')
    print(f'wrote {scores / "scores.json"} and attribution.md in {time.monotonic() - start:.0f}s', flush=True)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--outputs', type=Path, default=OUTPUTS)
    p.add_argument('--rows', nargs='+', default=list(ROW_ORDER), choices=ROW_ORDER)
    p.add_argument('--aspects', nargs='+', default=list(ASPECTS), choices=ASPECTS)
    p.add_argument('--workers', type=int, default=8, choices=range(1, 9))
    p.add_argument('--resamples', type=int, default=2000)
    args = p.parse_args(argv)
    run(args.outputs, args.rows, args.workers, args.resamples, tuple(args.aspects))


if __name__ == '__main__':
    main()
