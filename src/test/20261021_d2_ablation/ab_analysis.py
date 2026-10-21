"""Stage 2 analysis of the r10 D2 ablation (spec 3.6, D8). CPU only. They decide nothing.

--part views   the four Qwen views (default, emotion_focus, style_word, genre_word) on the 5,000 evaluation paintings:
               pairwise linear CKA and top-20 neighbour overlap (6 pairs, image and caption side), each view against each
               label (k-means k = 20 AMI, random states 0 to 2, and the probe), image against caption agreement per view,
               and the instruction offset (instructed minus default, image side): effective rank, principal angles,
               probe on the offset alone. Needs no training output.
--part rows    per trained row, aspect and seed (image side): CKA and neighbour overlap of the head output with the
               instructed image and with the default image, cluster size entropy, probe-minus-AMI gap (from ticket 04's
               scores.json), per-class recall and the class-to-cluster table (seed 0, random state 0).

Writes <outputs>/analysis/{views,rows}.json and .md (smoke: views_smoke / rows_smoke). Reads <outputs>/train/<row>/.
"""
import argparse
import hashlib
import itertools
import json
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
TEST = HERE.parent
for _folder in ('20261017_aspect_premise_checks', '20261017_prompted_aspect_views', '20261017_image_transfer_pilot',
                '20261018_purified_buddy_pilot', '20261018_aspect_graph_sheet', '20261019_model_sheet'):
    sys.path.insert(0, str(TEST / _folder))
sys.path.insert(0, str(HERE))

import pav_sample
# Artifacts are gitignored and exist only in the main checkout (ct_d2.SAMPLE does the same).
SAMPLE = Path('/project/CoSiR/src/test/20261017_prompted_aspect_views/sample.npz')
pav_sample.SAMPLE = SAMPLE
import ags_head
import itp_data
import ms_score
import pav_score
import ab_metrics as M

for _mod in (ags_head, itp_data, ms_score, pav_score, M):
    assert Path(_mod.__file__).resolve().is_relative_to(TEST), f'{_mod.__name__} resolved outside the worktree'

MAIN_OUTPUTS = Path('/project/CoSiR/src/test/20261021_d2_ablation/outputs')
ASPECTS = ('emotion', 'style', 'genre')
KEYS = dict(emotion='emotion_focus', style='style_word', genre='genre_word')
VIEWS = ('default', 'emotion_focus', 'style_word', 'genre_word')
PAIRS = tuple(itertools.combinations(VIEWS, 2))
SIDES = ('img', 'cap')
RANKS = (8, 16, 32)
K_NN = 20
K_MEANS = 20
SEEDS = (0, 1, 2)
KMEANS_THREADS = 8  # ticket 04's scorer runs k-means under 8 threads; keep the partitions comparable
SCORES_NEEDED = 'needs scores.json'


# ---------------------------------------------------------------------------------------------------- small helpers
def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, np.generic):
        return x.item()
    return x


def atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + '.pending' + path.suffix)
    tmp.write_text(text)
    tmp.replace(path)


def atomic_json(path, value):
    atomic_write(path, json.dumps(jsonable(value), indent=2, allow_nan=False) + '\n')


def sha_array(x):
    x = np.ascontiguousarray(x)
    h = hashlib.sha256(str((x.shape, x.dtype.str)).encode())
    h.update(x.tobytes())
    return h.hexdigest()


def check_array(x, n, unit=True, d=None):
    """Cheap assert at every read: shape (n, d), finite, and unit norm within .01 when `unit`."""
    x = np.asarray(x)
    assert x.ndim == 2 and x.shape[0] == n and (d is None or x.shape[1] == d), x.shape
    assert np.isfinite(x).all()
    if unit:
        assert np.max(np.abs(np.linalg.norm(x.astype(np.float32), axis=1) - 1)) < .01
    return x


def fmt(v, digits=3):
    return f'{v:.{digits}f}'


def ms(d, digits=3):
    return f"{d['mean']:.{digits}f} ± {d['sd']:.{digits}f}"


def pmap(fn, items, workers):
    """Map fn over items in forked workers (the data sit in module globals); workers = 1 runs inline."""
    items = list(items)
    if workers <= 1 or len(items) <= 1:
        return [fn(i) for i in items]
    with ProcessPoolExecutor(min(workers, len(items)), mp_context=multiprocessing.get_context('fork')) as ex:
        return list(ex.map(fn, items))


# ---------------------------------------------------------------------------------------------------- evaluation joins
def check_order(s, pid):
    """The evaluation rows must be s['subset_pid'] in that order (constitution C6)."""
    pid = np.asarray(pid)
    assert pid.shape == s['subset_pid'].shape and np.array_equal(pid, s['subset_pid']), \
        'evaluation rows are not s["subset_pid"] in order'


def eval_rows(full, s, pid=None):
    """Rows of a per-painting array (19,889, d) for the evaluation paintings, in s['subset_pid'] order."""
    pid = s['subset_pid'] if pid is None else pid
    check_order(s, pid)
    assert full.shape[0] == len(s['first']), full.shape
    return full[pid]


def positional_check(x_eval, reference):
    """Evaluation rows of a joined store equal the independently stored evaluation file row by row."""
    ref = np.asarray(reference, dtype=np.float32)
    ref = ref / np.linalg.norm(ref, axis=1, keepdims=True)
    assert x_eval.shape == ref.shape
    cos = np.sum(x_eval * ref, axis=1)
    assert cos.min() > .99 and cos.mean() > .999, (float(cos.mean()), float(cos.min()))
    return float(cos.mean())


def instruction_offset(instructed, default):
    """Instructed minus default, per painting, rows in the same painting order."""
    instructed, default = np.asarray(instructed), np.asarray(default)
    assert instructed.shape == default.shape and instructed.ndim == 2, (instructed.shape, default.shape)
    return instructed.astype(np.float64) - default.astype(np.float64)


def dev_rows(s, smoke=False):
    """Fit and early-stopping paintings (14,889, ascending), disjoint from the evaluation paintings (3.3)."""
    train_all = itp_data.split(s)[0]
    assert len(train_all) == 14889 and (np.diff(train_all) > 0).all()
    assert not np.intersect1d(train_all, s['subset_pid']).size
    return train_all[:1500] if smoke else train_all


# ---------------------------------------------------------------------------------------------------- context
def load_context(smoke=False, log=print):
    t = time.monotonic()
    s = ags_head.load_frozen()
    labels = ms_score.label_vectors(s, s['subset_pid'])
    for y in labels.values():
        assert y.shape == (5000,) and np.issubdtype(y.dtype, np.integer)
    inputs = {'default': ags_head.load_inputs(s, 'emotion', 'default')}
    for a in ASPECTS:
        inputs[KEYS[a]] = ags_head.load_inputs(s, a, 'prompted')
    for v in inputs.values():
        for side in SIDES:
            assert v[side].shape == (19889, 2048)
    ev = {v: {side: eval_rows(inputs[v][side], s) for side in SIDES} for v in VIEWS}
    # positional join check against the stores read independently of the join
    cos = {v: positional_check(ev[v]['img'], itp_data.prompted(s, v)[0]) for v in VIEWS}
    log(f'context loaded in {time.monotonic() - t:.1f}s; join cosine to stored evaluation images: '
        + ', '.join(f'{v} {c:.6f}' for v, c in cos.items()))
    return dict(s=s, labels=labels, inputs=inputs, eval=ev, join_cosine=cos)


# ---------------------------------------------------------------------------------------------------- partitions, AMI
def get_partitions(x, cache_dir=None):
    """Three k-means (k = 20, random states 0 to 2) partitions, pav_score.partitions as the scorer calls them."""
    x = np.asarray(x, dtype=np.float32)
    path = None
    if cache_dir is not None:
        path = Path(cache_dir) / f'parts_{sha_array(x)[:32]}_k{K_MEANS}.npy'
        if path.exists():
            p = np.load(path)
            assert p.shape == (3, len(x)) and p.min() >= 0 and p.max() < K_MEANS
            return p
    with threadpool_limits(KMEANS_THREADS):
        p = np.stack(pav_score.partitions(x, K_MEANS)).astype(np.int16)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + '.pending.npy')
        np.save(tmp, p)
        tmp.replace(path)
    return p


def ami_table(x, labels, cache_dir=None):
    """Per label: AMI of the three partitions (labelled paintings only), mean and population sd over random states."""
    from sklearn.metrics import adjusted_mutual_info_score
    parts = get_partitions(x, cache_dir)
    out = {}
    for name, y in labels.items():
        keep = y >= 0
        out[name] = pav_score.summary([adjusted_mutual_info_score(y[keep], p[keep]) for p in parts])
    return out


# ---------------------------------------------------------------------------------------------------- views part
def pairwise(vectors, seed=0, k=K_NN):
    """CKA and neighbour overlap for each of the 6 pairs, with chance: neighbour overlap k/(n-1); CKA of the
    second view with its rows shuffled (fixed seed), which is the finite-sample floor of the statistic."""
    n = len(next(iter(vectors.values())))
    perm = np.random.default_rng(seed).permutation(n)
    out = {}
    for a, b in PAIRS:
        out[f'{a}|{b}'] = dict(cka=M.linear_cka(vectors[a], vectors[b]),
                               knn_overlap=M.knn_overlap(vectors[a], vectors[b], k),
                               cka_shuffled=M.linear_cka(vectors[a], vectors[b][perm]),
                               knn_chance=M.knn_chance(n, k))
    return out


def probe_dict(r):
    return dict(balanced_accuracy=r['balanced_accuracy'], per_class_recall=r['per_class_recall'],
                classes=r['classes'], majority_rate=r['majority_rate'], chance=r['chance'], n=r['n'],
                n_classes=r['n_classes'], convergence_warnings=r['convergence_warnings'])


_G = {}  # fork-shared data for pmap tasks


def _probe_task(key):
    name, label = key
    with threadpool_limits(_G['threads']):
        return key, probe_dict(M.probe(_G['probe_x'][name], _G['labels'][label]))


def run_probes(arrays, labels, workers):
    """arrays: name -> (n, d); returns {name: {label: probe dict}}. One 5-fold probe per array and label."""
    _G.update(probe_x=arrays, labels=labels, threads=max(1, 8 // max(1, workers)) if workers > 1 else 8)
    tasks = [(name, label) for name in arrays for label in labels]
    out = {name: {} for name in arrays}
    for (name, label), r in pmap(_probe_task, tasks, workers):
        out[name][label] = r
    return out


def subspace_basis(x):
    """Right singular vectors (d, rank) and singular values of the column-centred matrix (the PCA subspaces)."""
    _, sv, vt = np.linalg.svd(M._centred(x), full_matrices=False)
    return vt.T, sv


def angles_from_bases(va, vb, r):
    """Same as M.principal_angles(a, b, r), from bases computed once."""
    cos = np.clip(np.linalg.svd(va[:, :r].T @ vb[:, :r], compute_uv=False), 0.0, 1.0)
    return cos, float((cos ** 2).mean())


def offset_geometry(offsets, default, ranks=RANKS):
    """Effective rank of each offset and of the default view; principal angles between the offsets' top-r subspaces.

    offsets: aspect -> (n, d) instructed minus default; default: (n, d). Subspaces and ranks over these rows."""
    d = default.shape[1]
    bases = {a: subspace_basis(o) for a, o in offsets.items()}
    out = dict(n=len(default), d=d,
               effective_rank={**{f'offset_{a}': M.effective_rank(o) for a, o in offsets.items()},
                               'default': M.effective_rank(default)},
               top_energy={a: {str(r): float((sv[:r] ** 2).sum() / (sv ** 2).sum()) for r in ranks}
                           for a, (_, sv) in bases.items()},
               angles={})
    for a, b in itertools.combinations(offsets, 2):
        out['angles'][f'{a}|{b}'] = {}
        for r in ranks:
            cos, msq = angles_from_bases(bases[a][0], bases[b][0], r)
            out['angles'][f'{a}|{b}'][str(r)] = dict(cosines=cos, mean_sq_cosine=msq, chance=M.subspace_chance(r, d))
    return out


def run_views(ctx, outputs, smoke=False, workers=4, log=print):
    s, labels, ev, inputs = ctx['s'], ctx['labels'], ctx['eval'], ctx['inputs']
    n = 500 if smoke else 5000
    sel = slice(0, n)
    labels = {k: v[sel] for k, v in labels.items()}
    X = {side: {v: check_array(ev[v][side][sel], n, d=2048) for v in VIEWS} for side in SIDES}
    cache = Path(outputs) / 'analysis' / 'cache'
    result = dict(part='views', smoke=smoke, n_eval=n, n_labelled={k: int((v >= 0).sum()) for k, v in labels.items()},
                  sample_sha256=itp_data.fingerprint(s), views=VIEWS, join_cosine=ctx['join_cosine'],
                  k_nn=K_NN, k_means=K_MEANS, timing={})

    def timed(name, fn):
        t = time.monotonic()
        r = fn()
        result['timing'][name] = round(time.monotonic() - t, 1)
        log(f'  views/{name}: {result["timing"][name]}s')
        return r

    # 1. pairwise agreement, both sides
    result['pairs'] = {side: timed(f'pairs_{side}', lambda side=side: pairwise(X[side])) for side in SIDES}
    # 2. image against caption, per view
    result['image_vs_caption'] = timed('image_vs_caption', lambda: {
        v: dict(cka=M.linear_cka(X['img'][v], X['cap'][v]), knn_overlap=M.knn_overlap(X['img'][v], X['cap'][v], K_NN),
                knn_chance=M.knn_chance(n, K_NN)) for v in VIEWS})
    # 3. each view against each label: AMI and probe, per side
    ami = timed('ami', lambda: {side: {v: ami_table(X[side][v], labels, cache) for v in VIEWS} for side in SIDES})
    arrays = {f'{side}|{v}': X[side][v] for side in SIDES for v in VIEWS}
    # offset on the 5,000 (or smoke) evaluation paintings, probed alone
    off_eval = {a: instruction_offset(X['img'][KEYS[a]], X['img']['default']) for a in ASPECTS}
    arrays.update({f'offset|{a}': o for a, o in off_eval.items()})
    probes = timed('probes', lambda: run_probes(arrays, labels, workers))
    result['table'] = {side: {v: {lab: dict(ami=ami[side][v][lab], probe=probes[f'{side}|{v}'][lab])
                                  for lab in labels} for v in VIEWS} for side in SIDES}
    # 4. the instruction offset: development estimate on the fit and early-stopping paintings, check on the evaluation
    rows = dev_rows(s, smoke)

    def dev_geometry():
        d_default = inputs['default']['img'][rows]
        offs = {a: instruction_offset(inputs[KEYS[a]]['img'][rows], d_default) for a in ASPECTS}
        return offset_geometry(offs, d_default.astype(np.float64))
    result['offset'] = dict(
        development=timed('offset_dev', dev_geometry),
        evaluation_check=timed('offset_eval', lambda: offset_geometry(
            off_eval, X['img']['default'].astype(np.float64))),
        probe_alone={a: {lab: dict(offset=probes[f'offset|{a}'][lab]['balanced_accuracy'],
                                   default=probes['img|default'][lab]['balanced_accuracy'],
                                   instructed=probes[f'img|{KEYS[a]}'][lab]['balanced_accuracy'],
                                   chance=probes[f'offset|{a}'][lab]['chance'])
                         for lab in labels} for a in ASPECTS})
    return result


def views_markdown(r):
    n = r['n_eval']
    L = ['# The four Qwen views' + (' (smoke)' if r['smoke'] else ''), '',
         f"{n} evaluation paintings; descriptive, nothing here decides. Neighbour overlap uses the top {r['k_nn']} cosine "
         f"neighbours (chance {r['pairs']['img'][f'{VIEWS[0]}|{VIEWS[1]}']['knn_chance']:.4f}). CKA floor = CKA with the "
         "second view's rows shuffled.", '']
    for side, title in (('img', 'Image side'), ('cap', 'Caption side (painting-level caption vector)')):
        L += [f'## {title}: pairwise agreement', '', '| Pair | CKA | CKA shuffled | Neighbour overlap |', '|---|---:|---:|---:|']
        for pair, v in r['pairs'][side].items():
            L.append(f"| {pair.replace('|', ' / ')} | {fmt(v['cka'])} | {fmt(v['cka_shuffled'])} | {fmt(v['knn_overlap'])} |")
        L.append('')
    L += ['## Image against caption, per view', '', '| View | CKA | Neighbour overlap |', '|---|---:|---:|']
    for v, x in r['image_vs_caption'].items():
        L.append(f"| {v} | {fmt(x['cka'])} | {fmt(x['knn_overlap'])} |")
    L.append('')
    for side, title in (('img', 'Image side'), ('cap', 'Caption side')):
        labs = list(r['table'][side][VIEWS[0]])
        L += [f'## {title}: view against label, k-means k = 20 AMI (mean ± sd over random states) and probe balanced accuracy',
              '', '| View | ' + ' | '.join(f'{lab} AMI | {lab} probe' for lab in labs) + ' |',
              '|---|' + '---:|---:|' * len(labs)]
        for v in VIEWS:
            cells = []
            for lab in labs:
                c = r['table'][side][v][lab]
                cells += [ms(c['ami']), fmt(c['probe']['balanced_accuracy'])]
            L.append(f'| {v} | ' + ' | '.join(cells) + ' |')
        chance = ', '.join(f"{lab} {fmt(r['table'][side][VIEWS[0]][lab]['probe']['chance'])}" for lab in labs)
        L += ['', f'Probe chance (1 / classes): {chance}.', '']
    for key, title in (('development', 'Instruction offset: fit and early-stopping paintings'),
                       ('evaluation_check', 'Instruction offset: evaluation paintings (check)')):
        g = r['offset'][key]
        L += [f"## {title} ({g['n']} paintings)", '', '| Matrix | Effective rank |', '|---|---:|']
        for name, v in g['effective_rank'].items():
            L.append(f'| {name} | {fmt(v, 1)} |')
        L += ['', 'Principal angles between the offsets\' top-r subspaces: mean squared cosine (chance r / d).', '',
              '| Pair | ' + ' | '.join(f'r = {x}' for x in RANKS) + ' |', '|---|' + '---:|' * len(RANKS)]
        for pair, byr in g['angles'].items():
            L.append(f"| {pair.replace('|', ' / ')} | " + ' | '.join(
                f"{fmt(byr[str(x)]['mean_sq_cosine'])} ({fmt(byr[str(x)]['chance'], 4)})" for x in RANKS) + ' |')
        L += ['', 'Share of the offset energy in its top-r directions:', '', '| Offset | ' + ' | '.join(
            f'r = {x}' for x in RANKS) + ' |', '|---|' + '---:|' * len(RANKS)]
        for a, byr in g['top_energy'].items():
            L.append(f'| {a} | ' + ' | '.join(fmt(byr[str(x)]) for x in RANKS) + ' |')
        L.append('')
    L += ['## Probe balanced accuracy on the offset alone (image side)', '',
          '| Offset | Label | Offset alone | Default view | Instructed view | Chance |', '|---|---|---:|---:|---:|---:|']
    for a, byl in r['offset']['probe_alone'].items():
        for lab, c in byl.items():
            L.append(f"| {a} | {lab} | {fmt(c['offset'])} | {fmt(c['default'])} | {fmt(c['instructed'])} | {fmt(c['chance'])} |")
    L += ['', 'Timing (seconds): ' + ', '.join(f'{k} {v}' for k, v in r['timing'].items()), '']
    return '\n'.join(L)


# ---------------------------------------------------------------------------------------------------- rows part
def read_row_meta(folder):
    """ab_row.json (ticket 02): the keys this script uses are asserted."""
    folder = Path(folder)
    meta = json.loads((folder / 'ab_row.json').read_text())
    assert isinstance(meta, dict) and meta['name'] == folder.name, (folder, meta.get('name'))
    sw = meta['switches']
    assert sw['teacher'] in ('instructed', 'default') and sw['endpoints'] in ('both', 'image'), sw
    return dict(name=meta['name'], switches=sw, store=meta.get('store'), caption_side=meta.get('caption_side'))


def read_scores(path):
    """Ticket 04's scores.json. Returns None when the file is absent. Expected layout (asserted where used):
    scores['rows'][row][aspect]['img'][str(seed)] = {'ami': {'mean': ..} or float,
    'probe': {'balanced_accuracy': .., 'per_class_recall': [..], 'classes': [..]}}."""
    path = Path(path)
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    assert isinstance(d, dict) and isinstance(d.get('rows'), dict), f'{path}: no "rows" mapping'
    return d


def score_entry(scores, row, aspect, seed):
    """(ami, probe dict) of one image-side cell, or None if the row, aspect or seed is not in scores.json."""
    try:
        cell = scores['rows'][row][aspect]['img'][str(seed)]
    except KeyError:
        return None
    ami = cell['ami']
    ami = ami['mean'] if isinstance(ami, dict) else ami
    probe = cell['probe']
    assert np.isfinite(ami) and 0 <= probe['balanced_accuracy'] <= 1, (row, aspect, seed)
    return float(ami), probe


def find_output(folder, aspect, seed):
    hits = sorted(Path(folder).glob(f'{aspect}__*__s{seed}.npz'))
    hits = [h for h in hits if '.pending' not in h.name]
    assert len(hits) <= 1, f'{folder}: several outputs for {aspect} seed {seed}: {hits}'
    return hits[0] if hits else None


def read_output(s, path, row, aspect, seed):
    """One exported head output (image side, float32, unit norm) with the row folder's own checks."""
    path = Path(path)
    info = json.loads(path.with_suffix('.json').read_text())
    c = info['config']
    assert c['aspect'] == aspect and c['seed'] == seed and c['sample_sha256'] == itp_data.fingerprint(s), path
    assert ms_score.sha_file(path) == info['output_sha256'], f'{path}: output checksum mismatch'
    with np.load(path, allow_pickle=False) as z:
        assert {'img', 'cap'} <= set(z.files)
        assert z['img'].dtype == np.float16
        x = check_array(z['img'].astype(np.float32), 5000, d=2048)
    return x, info


def class_to_cluster(y, partition, k=K_MEANS):
    """Counts (classes x k) of labelled paintings, plus per class the share in its largest cluster."""
    keep = y >= 0
    classes = np.unique(y[keep])
    counts = np.zeros((len(classes), k), dtype=np.int64)
    idx = np.searchsorted(classes, y[keep])
    np.add.at(counts, (idx, partition[keep].astype(np.int64)), 1)
    assert counts.sum() == keep.sum()
    return dict(classes=classes, counts=counts, top_cluster=counts.argmax(1),
                top_share=counts.max(1) / counts.sum(1))


def _row_task(task):
    row, aspect, seed, path = task
    G = _G
    x, info = read_output(G['s'], path, row, aspect, seed)
    ref_i, ref_d = G['instructed'][aspect], G['default']
    with threadpool_limits(G['threads']):
        out = dict(row=row, aspect=aspect, seed=seed, best_epoch=info['best_epoch'], epochs_run=info['epochs_run'],
                   cka_instructed=M.linear_cka(x, ref_i), knn_instructed=M.knn_overlap(x, ref_i, K_NN),
                   cka_default=M.linear_cka(x, ref_d), knn_default=M.knn_overlap(x, ref_d, K_NN))
    parts = get_partitions(x, G['cache'])
    out['entropy'] = [M.size_entropy(p, K_MEANS) for p in parts]
    out['entropy_mean'] = float(np.mean(out['entropy']))
    out['n_empty_clusters'] = [int(K_MEANS - len(np.unique(p))) for p in parts]
    if seed == 0:
        out['class_to_cluster'] = class_to_cluster(G['labels'][aspect], parts[0])
    return out


def run_rows(ctx, outputs, rows=None, workers=4, smoke=False, log=print):
    s, labels = ctx['s'], ctx['labels']
    outputs = Path(outputs)
    train = outputs / 'train'
    if rows:
        names = list(rows)
    else:
        names = sorted(p.name for p in train.iterdir() if (p / 'ab_row.json').exists()) if train.exists() else []
    scores = read_scores(outputs / 'scores' / 'scores.json')
    if scores is None:
        log(f'  {outputs / "scores" / "scores.json"} absent: probe-minus-AMI gap and per-class recall marked "{SCORES_NEEDED}"')
    meta, missing, tasks = {}, {}, []
    for row in names:
        folder = train / row
        if not (folder / 'ab_row.json').exists():
            missing[row] = ['no ab_row.json']
            continue
        meta[row] = read_row_meta(folder)
        for aspect in ASPECTS:
            for seed in SEEDS:
                p = find_output(folder, aspect, seed)
                if p is None:
                    missing.setdefault(row, []).append(f'{aspect}/s{seed}')
                else:
                    tasks.append((row, aspect, seed, p))
    _G.update(s=s, labels=labels, instructed={a: ctx['eval'][KEYS[a]]['img'] for a in ASPECTS},
              default=ctx['eval']['default']['img'], cache=outputs / 'analysis' / 'cache',
              threads=max(1, 8 // workers) if workers > 1 else 8)
    t = time.monotonic()
    done = pmap(_row_task, tasks, workers)
    log(f'  rows: {len(done)} cells in {time.monotonic() - t:.1f}s')
    result = dict(part='rows', smoke=smoke, sample_sha256=itp_data.fingerprint(s), k_nn=K_NN, k_means=K_MEANS,
                  scores_json='present' if scores is not None else SCORES_NEEDED, missing=missing, rows={})
    for row in meta:
        result['rows'][row] = dict(meta=meta[row], aspects={})
    for r in done:
        a = result['rows'][r['row']]['aspects'].setdefault(r['aspect'], dict(seeds={}))
        cell = {k: v for k, v in r.items() if k not in ('row', 'aspect', 'seed')}
        entry = score_entry(scores, r['row'], r['aspect'], r['seed']) if scores is not None else None
        if entry is None:
            cell['probe_balanced_accuracy'] = cell['ami'] = cell['probe_minus_ami'] = SCORES_NEEDED
        else:
            ami, probe = entry
            cell.update(ami=ami, probe_balanced_accuracy=probe['balanced_accuracy'],
                        probe_minus_ami=probe['balanced_accuracy'] - ami)
            if r['seed'] == 0:
                cell['probe_per_class_recall'] = dict(classes=probe['classes'], recall=probe['per_class_recall'])
        a['seeds'][str(r['seed'])] = cell
    scalar = ('cka_instructed', 'knn_instructed', 'cka_default', 'knn_default', 'entropy_mean', 'ami',
              'probe_balanced_accuracy', 'probe_minus_ami')
    for row in result['rows'].values():
        for a in row['aspects'].values():
            a['seeds'] = dict(sorted(a['seeds'].items()))
            a['n_seeds'] = len(a['seeds'])
            for k in scalar:
                v = [c[k] for c in a['seeds'].values()]
                a[k] = (dict(mean=float(np.mean(v)), sd=float(np.std(v))) if all(isinstance(x, float) for x in v)
                        else SCORES_NEEDED)
    return result


def rows_markdown(r):
    L = ['# Per-row diagnostics' + (' (smoke)' if r['smoke'] else ''), '',
         'Image side. Mean ± population sd over model seeds. CKA = linear CKA and overlap = top-20 neighbour overlap '
         'of the head output (5,000 evaluation paintings) with the instructed image of its aspect and with the default '
         'image. Entropy = cluster size entropy of the k-means k = 20 partitions over log 20 (random states 0 to 2). '
         'Gap = probe balanced accuracy minus AMI. A row\'s own teacher is in the "Teacher" column.', '']
    if r['missing']:
        L += ['Missing: ' + '; '.join(f"{k}: {', '.join(v)}" for k, v in r['missing'].items()), '']
    cols = (('cka_instructed', 'CKA instr.'), ('knn_instructed', 'Overlap instr.'), ('cka_default', 'CKA default'),
            ('knn_default', 'Overlap default'), ('entropy_mean', 'Entropy'), ('ami', 'AMI'),
            ('probe_balanced_accuracy', 'Probe'), ('probe_minus_ami', 'Gap'))
    for aspect in ASPECTS:
        L += [f'## {aspect}', '', '| Row | Teacher | Seeds | ' + ' | '.join(c[1] for c in cols) + ' |',
              '|---|---|---:|' + '---:|' * len(cols)]
        for name, row in r['rows'].items():
            a = row['aspects'].get(aspect)
            if a is None:
                L.append(f"| {name} | {row['meta']['switches']['teacher']} | 0 |" + ' missing |' * len(cols))
                continue
            L.append(f"| {name} | {row['meta']['switches']['teacher']} | {a['n_seeds']} | " + ' | '.join(
                ms(a[k]) if isinstance(a[k], dict) else a[k] for k, _ in cols) + ' |')
        L.append('')
    L += ['## Class to cluster, seed 0, random state 0', '',
          'Per class: the cluster holding most of its paintings and that cluster\'s share of the class; probe recall '
          '(seed 0) beside it when scores.json is present. Full count tables are in rows.json.', '']
    for aspect in ASPECTS:
        for name, row in r['rows'].items():
            c = row['aspects'].get(aspect, {}).get('seeds', {}).get('0')
            if not c or 'class_to_cluster' not in c:
                continue
            t = c['class_to_cluster']
            rec = c.get('probe_per_class_recall')
            if rec is not None:
                assert list(rec['classes']) == list(t['classes']), (name, aspect)
            L += [f'### {name} / {aspect}', '', '| Class | Paintings | Top cluster | Share in it | Probe recall |', '|---:|---:|---:|---:|---:|']
            for i, cl in enumerate(t['classes']):
                L.append(f"| {cl} | {int(np.sum(t['counts'][i]))} | {t['top_cluster'][i]} | {fmt(t['top_share'][i])} | "
                         f"{fmt(rec['recall'][i]) if rec else SCORES_NEEDED} |")
            L.append('')
    return '\n'.join(L)


# ---------------------------------------------------------------------------------------------------- CLI
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--outputs', type=Path, default=MAIN_OUTPUTS)
    p.add_argument('--part', nargs='+', choices=('rows', 'views'), default=['rows', 'views'])
    p.add_argument('--rows', nargs='+', help='row names under <outputs>/train (default: every folder with ab_row.json)')
    p.add_argument('--workers', type=int, default=4, help='worker processes (8 at most)')
    p.add_argument('--smoke', action='store_true', help='views on 500 paintings (dev part 1,500); rows on the smoke outputs')
    args = p.parse_args(argv)
    assert 1 <= args.workers <= 8
    out = args.outputs / 'analysis'
    suffix = '_smoke' if args.smoke else ''
    with threadpool_limits(8):
        ctx = load_context(args.smoke)
        if 'views' in args.part:
            t = time.monotonic()
            r = run_views(ctx, args.outputs, args.smoke, args.workers)
            r['timing']['total'] = round(time.monotonic() - t, 1)
            atomic_json(out / f'views{suffix}.json', r)
            atomic_write(out / f'views{suffix}.md', views_markdown(r))
            print(f'views done in {r["timing"]["total"]}s -> {out / f"views{suffix}.json"}', flush=True)
        if 'rows' in args.part:
            t = time.monotonic()
            r = run_rows(ctx, args.outputs, args.rows, args.workers, args.smoke)
            atomic_json(out / f'rows{suffix}.json', r)
            atomic_write(out / f'rows{suffix}.md', rows_markdown(r))
            print(f'rows done in {time.monotonic() - t:.1f}s -> {out / f"rows{suffix}.json"}', flush=True)


if __name__ == '__main__':
    main()
