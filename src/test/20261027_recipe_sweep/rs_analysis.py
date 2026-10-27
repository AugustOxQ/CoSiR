"""The analysis (spec §3.6, ticket 13): five descriptive readings of a model's held outputs. CPU. Decides nothing (rule §4.5).

  python rs_analysis.py --model G [--outputs DIR] [--smoke]

`--model G` reads 12-27's full-scale held outputs (`rs_paths.RG_FULL/held/GS-0.02/`); any other name reads the round's
(`full_root(outputs)/held/<name>/`). It runs only after the stage verdict exists: `<outputs>/stage0/verdict.json` for G,
`<outputs>/stage4/verdict.json` naming the model (field `candidate`) for the others. Only that existence and field are
read, never a number. `--smoke` takes the first 1,000 rows of H and writes under /tmp/claude-0/r10-rs-13/.

Writes `<outputs>/analysis/<NAME>/analysis.json` (atomic, input SHA-256s), `analysis.md` (the tables) and
`galleries/*.png`. The five readings (each on 5 seeds; galleries and cluster composition on seed 0, first k-means
partition), beside T (the instructed Qwen image of each aspect on H) and F (the frozen default Qwen image) where they apply:
  1 heads against label sets   AMI of each head's r20 partitions against each aspect's labels (3 by 3)
  2 per class                  best-matching cluster F1 per label class (model, T, model minus T); cluster composition;
                               gallery of the 16 paintings nearest each cluster centre (members of the cluster)
  3 agreement between heads    AMI between the three heads' partitions of the same paintings
  4 geometry                   effective rank, hubness (skew of 10-NN counts), alignment, uniformity
  5 neighbours vs the teacher  overlap of the 20 nearest neighbours under the head and under T, per class; gallery of
                               the lowest overlap (ties broken by a seeded random order)
Partitions are the frozen r20 ones where they exist (12-27's for G; T's and F's from 12-24); missing ones are computed
with `rb_score.fixed_partitions` and cached under `analysis/<NAME>/partitions/`.
Alignment needs the caption side of each head: the saved best states (weights/<name>/) are applied to the painting's
mean default Qwen caption vector (L2 normalised), as training did (`ct2_data.default_captions`, the Qwen store only: the
A/default store agrees with it to cosine > 0.999 by that function's own gate). The image side of the same replica is
checked against the saved held outputs.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import skew
from sklearn.metrics import adjusted_mutual_info_score as ami
from threadpoolctl import threadpool_limits

import rs_paths as paths
import rg_held
import rg_full
import rb_encoders
import rb_part2_report
import rb_score
import rb_student

SEEDS = (0, 1, 2, 3, 4)
K_NN, K_HUB = 20, 10
GALLERY = 16
THUMB = 72
SMOKE_ROWS = 1000
WIKIART = Path('/data/PDD/wikiart_proj/wikiart')                  # ARTELINGO_IMAGES of scripts/extract_features.py
QWEN = Path('/data/SSD2/pre_extract/artelingo_full/qwen3vl_emb_2b')
F_PARTITION = paths.RB_PART2 / 'partitions/full/F4/frozen__s0__r20.npz'    # the frozen default Qwen image's partitions
G_PARTITIONS = paths.RG_STAGE_C / 'partitions' / paths.G_ARM
PAIRS = (('emotion', 'style'), ('emotion', 'genre'), ('style', 'genre'))


def sha_file(path):
    return rb_score.sha_file(path)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.pending.json')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


# --- the run order guard and the inputs -------------------------------------------------------------------------

def guard(outputs, model, smoke):
    """G: stage0/verdict.json exists. Others: stage4/verdict.json exists and names the model. Nothing else is read."""
    if smoke:
        return
    if model == 'G':
        assert (Path(outputs) / 'stage0/verdict.json').is_file(), 'refused: stage0/verdict.json does not exist yet'
    else:
        path = Path(outputs) / 'stage4/verdict.json'
        assert path.is_file(), 'refused: stage4/verdict.json does not exist yet'
        verdict = json.loads(path.read_text())
        assert verdict.get('candidate') == model, f'refused: stage4/verdict.json does not name {model}'
        assert not verdict.get('v_failure'), f'refused: stage4/verdict.json has no outcome ({verdict["v_failure"]})'


def model_dirs(outputs, model):
    """(held folder, weights folder, arm name in the sidecars, folder of existing partitions or None)."""
    if model == 'G':
        return paths.RG_FULL / 'held' / paths.G_ARM, paths.RG_FULL / 'weights' / paths.G_ARM, paths.G_ARM, G_PARTITIONS
    root = paths.full_root(outputs)
    return root / 'held' / model, root / 'weights' / model, model, None


def class_names(labels):
    """Sorted-name order of the label codes (`encode_labels`; genre: the WikiArt list), checked against the code counts."""
    import ct_held
    from src.data.artelingo_splits import encode_labels
    from src.data.wikiart_genre import GENRE_NAMES
    records = ct_held._annotations()
    names = {'emotion': encode_labels([r['emotion'] for r in records])[1],
             'style': encode_labels([r['art_style'] for r in records])[1], 'genre': list(GENRE_NAMES)}
    for aspect, y in labels.items():
        assert int(y.max()) < len(names[aspect]), (aspect, int(y.max()), len(names[aspect]))
    return names


def painting_files(names):
    index = json.loads((QWEN / 'index.json').read_text())
    lookup = {row['painting']: row['image'] for row in index['images']}
    return [WIKIART / lookup[n] for n in names], index


def caption_vectors(names, index):
    """L2 of the mean Qwen caption vector of each painting in `names` (float32 [n, 2048]) and the caption rows used."""
    first = {n: i for i, n in enumerate(names)}
    rows, owner = [], []
    for r, t in enumerate(index['texts']):
        i = first.get(t['painting'])
        if i is not None:
            rows.append(r)
            owner.append(i)
    rows, owner = np.asarray(rows), np.asarray(owner)
    raw = np.load(QWEN / 'txt.npy', mmap_mode='r')
    assert raw.shape == (index['n_texts'], 2048) and raw.dtype == np.float16
    x = np.asarray(raw[rows], dtype=np.float32)
    assert np.isfinite(x).all() and (np.linalg.norm(x, axis=1) > 0).all()
    total = np.zeros((len(names), 2048), dtype=np.float64)
    np.add.at(total, owner, x)
    count = np.bincount(owner, minlength=len(names))
    assert (count > 0).all(), 'a held painting has no caption'
    mean = (total / count[:, None]).astype(np.float32)
    return mean / np.maximum(np.linalg.norm(mean, axis=1, keepdims=True), 1e-12), int(len(rows))


class Head:
    """The saved best state of one head as plain tensors: both sides are Linear(2048, h), GELU, Linear(h, 2048), L2."""

    def __init__(self, weights):
        state = torch.load(weights, map_location='cpu', weights_only=True)
        want = {f'{side}.{k}' for side in ('img', 'cap') for k in ('trunk.0.weight', 'trunk.0.bias', 'output.weight', 'output.bias')}
        assert set(state) == want, (weights, sorted(set(state) ^ want))
        self.state = {k: v.float() for k, v in state.items()}

    def __call__(self, side, x, batch=2048):
        s = self.state
        out = []
        with torch.no_grad():
            for i in range(0, len(x), batch):
                h = torch.nn.functional.gelu(torch.from_numpy(x[i:i + batch]) @ s[f'{side}.trunk.0.weight'].T + s[f'{side}.trunk.0.bias'])
                z = h @ s[f'{side}.output.weight'].T + s[f'{side}.output.bias']
                out.append(torch.nn.functional.normalize(z, dim=1).numpy())
        return np.concatenate(out)


# --- the statistics ---------------------------------------------------------------------------------------------

def knn(x, k=K_NN, chunk=1024):
    """Cosine nearest neighbours of unit rows, self excluded: int64 [n, k], nearest first."""
    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    out = np.empty((len(x), k), dtype=np.int64)
    for i in range(0, len(x), chunk):
        s = t[i:i + chunk] @ t.T
        s[torch.arange(len(s)), torch.arange(i, i + len(s))] = -np.inf
        out[i:i + chunk] = torch.topk(s, k, dim=1).indices.numpy()
    return out


def effective_rank(x, center):
    """exp of the entropy of the normalised singular values (Roy and Vetterli), from the Gram matrix's eigenvalues."""
    x = x.astype(np.float64)
    if center:
        x = x - x.mean(0)
    sv = np.sqrt(np.maximum(np.linalg.eigvalsh(x.T @ x), 0))
    p = sv[sv > 0] / sv.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def hubness(nn):
    """Skewness of how often each painting is among the 10 nearest neighbours of the others, and the largest count."""
    count = np.bincount(nn[:, :K_HUB].ravel(), minlength=len(nn))
    return float(skew(count)), int(count.max())


def uniformity(x, chunk=2048):
    """Wang and Isola: log of the mean of exp(-2 |x - y|^2) over all pairs of distinct rows."""
    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float64))
    total = 0.0
    for i in range(0, len(t), chunk):
        g = t[i:i + chunk] @ t.T
        total += float(torch.exp(-2 * (2 - 2 * g)).sum())
    n = len(t)
    return float(np.log((total - n) / (n * (n - 1))))                # the n self pairs each contribute exp(0)


def alignment(img, cap):
    """Wang and Isola: mean |image output - caption output|^2 of the same painting."""
    return float(np.mean(np.sum((img - cap) ** 2, axis=1)))


def geometry(x, cap=None):
    nn = knn(x)
    skew_, top = hubness(nn)
    row = dict(erank_centered=effective_rank(x, True), erank_raw=effective_rank(x, False), hubness_skew=skew_,
               hub_max_count=top, uniformity=uniformity(x))
    if cap is not None:
        row['alignment'] = alignment(x, cap)
    return row, nn


def best_f1(y, p, n_classes):
    """F1 of the best-matching cluster of each label class; y >= 0 (labelled rows only)."""
    k = int(p.max()) + 1
    table = np.bincount(y * k + p, minlength=n_classes * k).reshape(n_classes, k).astype(np.float64)
    precision = table / np.maximum(table.sum(0), 1)
    recall = table / np.maximum(table.sum(1, keepdims=True), 1)
    return (2 * precision * recall / np.maximum(precision + recall, 1e-12)).max(1)


def composition(y, p, n_classes):
    """Label shares per cluster (rows: clusters, labelled paintings only) and the cluster sizes (all paintings)."""
    k = int(p.max()) + 1
    mask = y >= 0
    table = np.bincount(p[mask] * n_classes + y[mask], minlength=k * n_classes).reshape(k, n_classes).astype(np.float64)
    return table / np.maximum(table.sum(1, keepdims=True), 1), np.bincount(p, minlength=k)


def mean_sd(values):
    values = np.asarray(values, dtype=np.float64)
    return dict(mean=float(values.mean(axis=0)), sd=float(values.std(axis=0, ddof=0)), values=[float(v) for v in values])


# --- galleries --------------------------------------------------------------------------------------------------

def thumb(file, cache):
    from PIL import Image
    if file not in cache:
        im = Image.open(file).convert('RGB')
        w, h = im.size
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((THUMB, THUMB))
        cache[file] = im
    return cache[file]


def sheet(rows, path, label_width=200):
    """rows: [(label, [files or None])]; one row of thumbnails per entry, a text label on the left."""
    from PIL import Image, ImageDraw
    width = max(len(files) for _, files in rows)
    canvas = Image.new('RGB', (label_width + width * THUMB, len(rows) * THUMB), 'white')
    draw = ImageDraw.Draw(canvas)
    cache = sheet.cache
    for r, (label, files) in enumerate(rows):
        draw.text((4, r * THUMB + 4), label, fill='black')
        for c, file in enumerate(files):
            if file is not None:
                canvas.paste(thumb(file, cache), (label_width + c * THUMB, r * THUMB))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


sheet.cache = {}


def cluster_gallery(x, part, y, aspect_names, files, path, tag):
    """Per cluster of one partition: the 16 member paintings nearest the cluster centre (mean output, renormalised)."""
    comp, sizes = composition(y, part, len(aspect_names))
    rows = []
    for j in range(len(sizes)):
        members = np.flatnonzero(part == j)
        if not len(members):
            continue
        centre = x[members].mean(0)
        centre /= np.linalg.norm(centre)
        near = members[np.argsort(-(x[members] @ centre), kind='stable')[:GALLERY]]
        top = int(comp[j].argmax())
        rows.append((f'{tag} c{j} n={len(members)}\n{aspect_names[top]} {comp[j, top]:.0%}', [files[i] for i in near]))
    sheet(rows, path)


def neighbour_gallery(nn_head, nn_t, overlap, files, path, tag, rng):
    """The 16 paintings of lowest overlap: the painting, its 4 nearest under the head, then its 4 nearest under T."""
    order = np.lexsort((rng.permutation(len(overlap)), overlap))[:GALLERY]
    rows = [(f'{tag} p{i} ov={overlap[i]:.2f}\nhead | T', [files[i]] + [files[j] for j in nn_head[i, :4]] + [None] + [files[j] for j in nn_t[i, :4]])
            for i in order]
    sheet(rows, path)


# --- the run ----------------------------------------------------------------------------------------------------

def read_partitions(x, aspect, seed, held_file, sha, folder, store, smoke):
    """r20 partitions of one head: the frozen file when `folder` has it, else computed (cached under `store`)."""
    if folder is not None and (Path(folder) / f'{aspect}__s{seed}__r20.npz').exists():
        file = Path(folder) / f'{aspect}__s{seed}__r20.npz'
        return rg_held.read_frozen_partitions(file, source_file=held_file), file
    assert not smoke, f'smoke needs the frozen partitions of {aspect} seed {seed}'
    labels = rb_score.fixed_partitions(x, 'K', store, aspect, seed, sha, sha_file(held_file))['r20']
    return labels, Path(store) / f'{aspect}__s{seed}__r20.npz'


def run(model, outputs, smoke=False):
    t0 = time.time()
    timings = {}

    def tick(name):
        timings[name] = round(time.time() - t0, 1)
        print(f'[{timings[name]:7.1f}s] {name}', flush=True)

    paths.check_imports()
    outputs = paths.check_outputs(outputs, smoke)
    guard(outputs, model, smoke)
    torch.set_num_threads(8)
    held_dir, weights_dir, arm, existing = model_dirs(outputs, model)
    out_dir = outputs / 'analysis' / model
    names, sha, labels = rb_part2_report.held_labels()
    assert sha == paths.HELD_SHA256 and len(names) == paths.N_HELD
    n = SMOKE_ROWS if smoke else paths.N_HELD
    aspect_names = class_names(labels)
    files_all, index = painting_files(names)
    cap_all, n_captions = caption_vectors(names[:n], index)
    y = {a: labels[a][:n] for a in paths.ASPECTS}
    files = files_all[:n]
    inputs = dict(held_sha256=sha, rows=n, smoke=bool(smoke), qwen_index_sha256=sha_file(QWEN / 'index.json'),
                  caption_vectors_sha256=rb_student.sha_array(cap_all), n_captions=n_captions)
    # the instructed images T, the frozen default image F, and the model's outputs, partitions and caption outputs
    default_full = rb_encoders.full_images(rg_full.ENCODER, names)
    inputs['default_images_sha256'] = rb_student.sha_array(default_full)
    default = default_full[:n]
    t_img, part = {}, {}
    for a in paths.ASPECTS:
        t = rb_part2_report.target_image(a, names, sha)
        inputs[f'T_{a}_sha256'] = rb_student.sha_array(t)
        t_img[a] = t[:n]
    part['T'] = {a: rg_held._frozen_r20('T', a, [0])[0]['r20'][:, :n] for a in paths.ASPECTS}
    part['F'] = rg_held.read_frozen_partitions(F_PARTITION)[:, :n]
    inputs['T_partitions_sha256'] = {a: sha_file(paths.RB_PART2 / f'partitions/full/T/{a}__s0__r20.npz') for a in paths.ASPECTS}
    inputs['F_partition_sha256'] = sha_file(F_PARTITION)
    out, cap, part['M'] = {}, {}, {}
    inputs['held_outputs_sha256'], inputs['partitions_sha256'], inputs['weights_sha256'] = {}, {}, {}
    for a in paths.ASPECTS:
        for s in SEEDS:
            file = held_dir / f'{a}__s{s}.npz'
            assert file.is_file(), f'missing held output {file}'
            x_full = rg_held.load_held_output(file, arm, s, sha, aspect=a)
            side = json.loads(file.with_suffix('.json').read_text())
            weights = weights_dir / f'{a}__s{s}.pt'
            assert sha_file(weights) == side['weights_sha256'], f'{weights}: differs from the one that made {file}'
            labels_all, pfile = read_partitions(x_full, a, s, file, sha, existing, out_dir / 'partitions', smoke)
            assert labels_all.shape == (3, paths.N_HELD)
            out[a, s], part['M'][a, s] = x_full[:n], labels_all[:, :n]
            head = Head(weights)
            check = np.abs(head('img', default[:512]) - out[a, s][:512]).max()
            assert check < 5e-3, f'{weights}: image-side replica differs from the saved output by {check}'
            cap[a, s] = head('cap', cap_all)
            inputs['held_outputs_sha256'][f'{a}__s{s}'] = sha_file(file)
            inputs['partitions_sha256'][f'{a}__s{s}'] = sha_file(pfile)
            inputs['weights_sha256'][f'{a}__s{s}'] = sha_file(weights)
    tick('inputs read, caption sides applied, image-side replica checked')

    result = dict(schema=1, model=model, arm=arm, smoke=bool(smoke), rows=n, seeds=list(SEEDS), inputs=inputs,
                  note='descriptive only; decides nothing (rule §4.5)')
    # 1 heads against label sets: A[head][label aspect]
    def ami_rows(p, a_label):
        m = y[a_label] >= 0
        return [float(ami(y[a_label][m], pj[m])) for pj in p]

    h1 = {'model': {}, 'T': {}, 'F': {}}
    for h in paths.ASPECTS:
        h1['model'][h] = {a: mean_sd([np.mean(ami_rows(part['M'][h, s], a)) for s in SEEDS]) for a in paths.ASPECTS}
        h1['T'][h] = {a: dict(mean=float(np.mean(ami_rows(part['T'][h], a)))) for a in paths.ASPECTS}
    h1['F'] = {a: dict(mean=float(np.mean(ami_rows(part['F'], a)))) for a in paths.ASPECTS}
    result['heads_vs_labels'] = h1
    tick('1 heads against label sets')

    # 2 per class (own aspect), composition and cluster galleries (seed 0, first partition)
    h2 = {}
    for a in paths.ASPECTS:
        m = y[a] >= 0
        nc = len(aspect_names[a])
        f_model = np.array([[best_f1(y[a][m], part['M'][a, s][j][m], nc) for j in range(3)] for s in SEEDS]).mean(1)   # [seed, class]
        f_t = np.mean([best_f1(y[a][m], part['T'][a][j][m], nc) for j in range(3)], axis=0)
        f_f = np.mean([best_f1(y[a][m], part['F'][j][m], nc) for j in range(3)], axis=0)
        comp, sizes = composition(y[a], part['M'][a, 0][0], nc)
        h2[a] = dict(classes=aspect_names[a], count=np.bincount(y[a][m], minlength=nc).tolist(),
                     f1_model=f_model.mean(0).tolist(), f1_model_sd=f_model.std(0).tolist(), f1_T=f_t.tolist(),
                     f1_default=f_f.tolist(), f1_model_minus_T=(f_model.mean(0) - f_t).tolist(),
                     composition_seed0=comp.tolist(), cluster_sizes_seed0=sizes.tolist())
    result['per_class'] = h2
    galleries = []
    for a in paths.ASPECTS:
        path = out_dir / 'galleries' / f'clusters_{a}_s0.png'
        cluster_gallery(out[a, 0], part['M'][a, 0][0], y[a], aspect_names[a], files, path, a)
        galleries.append(str(path))
    tick('2 per class and cluster galleries')

    # 3 agreement between heads
    result['agreement'] = {
        'model': {f'{a}-{b}': mean_sd([np.mean([ami(part['M'][a, s][j], part['M'][b, s][j]) for j in range(3)]) for s in SEEDS])
                  for a, b in PAIRS},
        'T': {f'{a}-{b}': dict(mean=float(np.mean([ami(part['T'][a][j], part['T'][b][j]) for j in range(3)]))) for a, b in PAIRS}}
    tick('3 agreement between heads')

    # 4 geometry and 5 neighbours (the 20-NN lists serve both)
    geo = {'model': {a: {} for a in paths.ASPECTS}, 'T': {}, 'F': None}
    nn = {}
    for a in paths.ASPECTS:
        rows = []
        for s in SEEDS:
            row, nn[a, s] = geometry(out[a, s], cap[a, s])
            rows.append(row)
        geo['model'][a] = {k: mean_sd([r[k] for r in rows]) for k in rows[0]}
        geo['T'][a], nn['T', a] = geometry(t_img[a])
    geo['F'], nn['F'] = geometry(default)
    result['geometry'] = geo
    tick('4 geometry')
    h5 = {}
    rng = np.random.default_rng(0)
    for a in paths.ASPECTS:
        overlap = lambda nh, nt: (nh[:, :, None] == nt[:, None, :]).any(2).sum(1) / K_NN
        per_seed = np.array([overlap(nn[a, s], nn['T', a]) for s in SEEDS])               # [seed, painting]
        f_over = overlap(nn['F'], nn['T', a])
        m = y[a] >= 0
        nc = len(aspect_names[a])
        cls = lambda v: [float(v[m & (y[a] == c)].mean()) if (m & (y[a] == c)).any() else None for c in range(nc)]
        h5[a] = dict(classes=aspect_names[a], model=mean_sd(per_seed.mean(1)), default=float(f_over.mean()),
                     per_class_model=cls(per_seed.mean(0)), per_class_default=cls(f_over),
                     model_labelled=float(per_seed.mean(0)[m].mean()), default_labelled=float(f_over[m].mean()))
        path = out_dir / 'galleries' / f'lowest_overlap_{a}_s0.png'
        neighbour_gallery(nn[a, 0], nn['T', a], per_seed[0], files, path, a, rng)
        galleries.append(str(path))
    result['neighbours'] = h5
    result['galleries'] = galleries
    tick('5 neighbours and galleries')
    timings['total_s'] = round(time.time() - t0, 1)
    result['timings_s'] = timings
    atomic_json(out_dir / 'analysis.json', result)
    (out_dir / 'analysis.md').write_text(markdown(result))
    tick('written')
    return result


# --- the tables -------------------------------------------------------------------------------------------------

def f(v, d=3):
    return 'n/a' if v is None else f'{v:.{d}f}'


def pm(d, k=3):
    return f'{d["mean"]:.{k}f} +- {d["sd"]:.{k}f}'


def markdown(r):
    A = paths.ASPECTS
    L = [f'# Analysis of {r["model"]} (spec §3.6; descriptive, decides nothing)', '',
         f'Held paintings: {r["rows"]}{" (SMOKE: first 1,000 rows)" if r["smoke"] else ""}; seeds {r["seeds"]}; mean +- sd over seeds. '
         'T: the instructed Qwen image of the aspect (seed 0 only, no sd); F: the frozen default Qwen image.', '']
    L += ['## 1 Heads against label sets (AMI of the r20 partitions, mean of the 3 partitions)', '',
          '| head | ' + ' | '.join(f'vs {a}' for a in A) + ' |', '|---|' + '---|' * len(A)]
    for h in A:
        L.append(f'| model {h} | ' + ' | '.join(pm(r['heads_vs_labels']['model'][h][a]) for a in A) + ' |')
    for h in A:
        L.append(f'| T {h} | ' + ' | '.join(f(r['heads_vs_labels']['T'][h][a]['mean']) for a in A) + ' |')
    L.append('| F (default) | ' + ' | '.join(f(r['heads_vs_labels']['F'][a]['mean']) for a in A) + ' |')
    L += ['', '## 2 Per class (best-matching cluster F1, own aspect; mean of 3 partitions, then of seeds)', '']
    for a in A:
        p = r['per_class'][a]
        L += [f'### {a}', '', '| class | n | model | T | model - T | F |', '|---|---|---|---|---|---|']
        for c, name in enumerate(p['classes']):
            L.append(f'| {name} | {p["count"][c]} | {f(p["f1_model"][c])} +- {f(p["f1_model_sd"][c])} | {f(p["f1_T"][c])} | '
                     f'{p["f1_model_minus_T"][c]:+.3f} | {f(p["f1_default"][c])} |')
        L += ['', f'Cluster composition (seed 0, first partition): top class and its share per cluster; the full shares are in analysis.json.', '',
              '| cluster | size | top class | share |', '|---|---|---|---|']
        for j, (sz, row) in enumerate(zip(p['cluster_sizes_seed0'], p['composition_seed0'])):
            top = int(np.argmax(row))
            L.append(f'| {j} | {sz} | {p["classes"][top]} | {row[top]:.2f} |')
        L.append('')
    L += ['Galleries (16 member paintings nearest each cluster centre, seed 0, first partition): '
          + ', '.join(g for g in r['galleries'] if 'clusters_' in g), '']
    L += ['## 3 Agreement between heads (AMI between the heads\' partitions of the same paintings)', '',
          '| pair | model | T |', '|---|---|---|']
    for k in r['agreement']['model']:
        L.append(f'| {k} | {pm(r["agreement"]["model"][k])} | {f(r["agreement"]["T"][k]["mean"])} |')
    L += ['', '## 4 Geometry', '', 'Effective rank: exp entropy of the singular values (centred; raw in brackets). Hubness: skew of the 10-NN counts '
          '(max count in brackets). Alignment: mean squared distance, image vs caption output of the same painting (lower is closer). '
          'Uniformity: log mean exp(-2 d^2) over pairs (lower is more spread).', '',
          '| output | eff. rank (raw) | hubness skew (max) | alignment | uniformity |', '|---|---|---|---|---|']
    g = r['geometry']
    for a in A:
        m = g['model'][a]
        L.append(f'| model {a} | {m["erank_centered"]["mean"]:.1f} ({m["erank_raw"]["mean"]:.1f}) | {m["hubness_skew"]["mean"]:.2f} '
                 f'({m["hub_max_count"]["mean"]:.0f}) | {pm(m["alignment"], 4)} | {pm(m["uniformity"])} |')
    for a in A:
        t = g['T'][a]
        L.append(f'| T {a} | {t["erank_centered"]:.1f} ({t["erank_raw"]:.1f}) | {t["hubness_skew"]:.2f} ({t["hub_max_count"]}) | n/a | {t["uniformity"]:.3f} |')
    t = g['F']
    L.append(f'| F (default) | {t["erank_centered"]:.1f} ({t["erank_raw"]:.1f}) | {t["hubness_skew"]:.2f} ({t["hub_max_count"]}) | n/a | {t["uniformity"]:.3f} |')
    L += ['', '## 5 Neighbours against the teacher (overlap of the 20 nearest neighbours under the head and under T)', '',
          '| aspect | model (all) | F vs T (all) | model (labelled) | F vs T (labelled) |', '|---|---|---|---|---|']
    for a in A:
        n = r['neighbours'][a]
        L.append(f'| {a} | {pm(n["model"], 4)} | {n["default"]:.4f} | {n["model_labelled"]:.4f} | {n["default_labelled"]:.4f} |')
    for a in A:
        n = r['neighbours'][a]
        L += ['', f'### {a}, per class', '', '| class | model | F vs T |', '|---|---|---|']
        L += [f'| {c} | {f(m, 4)} | {f(d, 4)} |' for c, m, d in zip(n['classes'], n['per_class_model'], n['per_class_default'])]
    L += ['', 'Galleries of the 16 lowest-overlap paintings (seed 0; per row the painting, its 4 nearest under the head, a gap, its 4 nearest under T): '
          + ', '.join(g for g in r['galleries'] if 'lowest_' in g), '',
          f'Timings (s, cumulative): {r["timings_s"]}', '']
    return '\n'.join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--model', required=True)
    ap.add_argument('--outputs', default=None)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args(argv)
    outputs = Path(args.outputs) if args.outputs else (paths.SMOKE_ROOT / 'r10-rs-13/smoke' if args.smoke else paths.OUTPUTS)
    with threadpool_limits(limits=8):
        run(args.model, outputs, args.smoke)


if __name__ == '__main__':
    main()
