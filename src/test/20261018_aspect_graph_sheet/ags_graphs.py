"""CPU aspect graph sheet; stage 0 always precedes label-free sweep construction."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
for folder in ('20261017_image_transfer_pilot', '20261017_prompted_aspect_views',
               '20261018_purified_buddy_pilot'):
    sys.path.insert(0, str(HERE.parent / folder))
from itp_data import atomic_json, default_images, fingerprint, load_frozen, split, targets
from itp_score import top20_neighbours
from pav_score import graph_stats, mutual_pairs, references
from pbp_graphs import label_vectors, pair_array, sha_file
from apc_data import l2, paint_mean
from ags_trainimg import aspect_images

BASE = HERE / 'graphs'
PILOT = Path('/project/CoSiR/src/test/20261018_purified_buddy_pilot/graphs')
ASPECTS = dict(emotion='emotion_focus', style='style_word', genre='genre_word')
ROWS = ('R0', 'R0o', 'R1', 'R1c', 'Prank5', 'Prank10', 'Psim50', 'Psim75', 'Psim90',
        'Fres025', 'Fres05', 'Fres1', 'Forth', 'Fside025', 'Fside05', 'Fside075',
        'Frank', 'Gdown025', 'Gdown05', 'Gagree', 'Gres')
METHOD = 'ags rows.md 21 rows; block256; sd200000 rng0 offdiagonal; rank quicksort; v1'


def family(row):
    return dict(R='reference', P='prune', F='fuse', G='guided')[row[0]]


def atomic_npz(path, **arrays):
    pending = path.with_name(path.stem + '.pending.npz')
    np.savez(pending, **arrays)
    pending.replace(path)


def atomic_markdown(path, value):
    pending = path.with_name(path.stem + '.pending.md')
    pending.write_text(value)
    pending.replace(path)


def reciprocal(nn):
    directed = {(a, int(b)) for a, row in enumerate(nn) for b in row}
    return {(a, b) for a, b in directed if a < b and (b, a) in directed}


def source_pairs(x):
    pairs = mutual_pairs(x)
    assert pairs == reciprocal(top20_neighbours(x)), 'pilot reciprocity mismatch'
    return pairs


def top(score, start, k=20):
    score = score.copy()
    score[np.arange(len(score)), start + np.arange(len(score))] = -np.inf
    return np.argpartition(-score, k - 1, axis=1)[:, :k]


def ranks(score, start):
    score = score.copy()
    score[np.arange(len(score)), start + np.arange(len(score))] = -np.inf
    order = np.argsort(-score, axis=1, kind='quicksort')
    out = np.empty(score.shape, dtype=np.int32)
    np.put_along_axis(out, order, np.arange(1, score.shape[1] + 1)[None, :], axis=1)
    return out


def stage0(s, parts, spaces):
    tick = time.monotonic()
    rebuilt, differences = {}, []
    pilot_manifest = json.loads((PILOT / 'manifest.json').read_text())
    if pilot_manifest['identity']['sample_sha256'] != fingerprint(s):
        differences.append('pilot sample fingerprint mismatch')
    for side, nodes in parts.items():
        g = {name: source_pairs(np.ascontiguousarray(x[nodes])) for name, x in spaces.items()}
        g.update(C=g['D'] | g['I'])
        g.update(P=g['E'] - g['C'], EC=g['E'] & g['C'])
        rebuilt[side] = g
        path = PILOT / f'{side}.npz'
        if sha_file(path) != pilot_manifest['archives'][side]:
            differences.append(f'{side}: pilot checksum mismatch')
        with np.load(path, allow_pickle=False) as z:
            if not np.array_equal(z['node_ids'], nodes):
                differences.append(f'{side}: node order mismatch')
            for name in ('D', 'I', 'C', 'E', 'P', 'EC'):
                actual = pair_array(g[name])
                if not np.array_equal(actual, z[name]):
                    old = set(map(tuple, z[name].tolist()))
                    differences.append(f'{side}.{name}: +{len(g[name] - old)} -{len(old - g[name])}')
        print(f'stage0 {side}: ' + ' '.join(f'{k}={len(v)}' for k, v in g.items()), flush=True)
    # The gate's labels are scoped here; sweep labels are read after all archives seal.
    labels = label_vectors(s, parts['fit'])
    pilot_stats = json.loads((PILOT / 'stats.json').read_text())
    lifts = {}
    for name, expected in zip(('E', 'P', 'EC'), (2.4204897561, 2.3849313416, 2.5528809893)):
        lift = graph_stats(rebuilt['fit'][name], labels)['labels']['emotion']['lift']
        lifts[name] = lift
        archived = pilot_stats['graphs'][name]['labels']['emotion']['lift']
        if abs(lift - expected) > 1e-9 or abs(lift - archived) > 1e-9:
            differences.append(f'{name}: emotion lift {lift} != {expected} / archive {archived}')
    line = 'STAGE0 ' + ('FAIL ' + '; '.join(differences) if differences else
        'PASS ' + ' / '.join(f'{lifts[k]:.10f}' for k in ('E', 'P', 'EC')))
    atomic_json(BASE / 'stage0.json', dict(pass_gate=not differences, differences=differences,
        lifts=lifts, line=line, sample_sha256=fingerprint(s), seconds=time.monotonic() - tick))
    print(line, flush=True)
    if differences:
        raise SystemExit(1)
    return rebuilt


def clip_content(s):
    # Same CLIP loading and joins as pbp_clip_content.py, including the frozen split check.
    name, txt, _ = next(references(s))
    assert name == 'clip_text' and len(txt) == len(s['pid']) == len(s['rows'])
    count = len(s['first'])
    assert np.array_equal(s['pid'][s['first']], np.arange(count))
    from src.data.artelingo import load_artelingo
    d = load_artelingo()
    assert np.array_equal(d.sample_ids[s['rows']], s['sample_ids'])
    img = l2(d.img_features[s['rows'][s['first']]])
    cap = l2(paint_mean(txt, s['pid'], count))
    assert img.shape[0] == cap.shape[0] == count
    assert np.isfinite(img).all() and np.isfinite(cap).all()
    return dict(Dc=cap, Ic=img)


def draw_pairs(n):
    rng = np.random.default_rng(0)
    i = rng.integers(n, size=200000)
    j = rng.integers(n - 1, size=200000)
    j += j >= i
    assert np.all(i != j)
    return i, j


def similarity_sd(x, pairs):
    i, j = pairs
    values = np.empty(len(i), dtype=np.float32)
    for a in range(0, len(i), 1024):
        b = a + 1024
        values[a:b] = np.sum(x[i[a:b]] * x[j[a:b]], axis=1)
    sd = float(np.std(values, dtype=np.float64))
    assert np.isfinite(sd) and sd > 0
    return sd


def pair_scores(x, pairs):
    ij = pair_array(pairs)
    scores = np.empty(len(ij), dtype=np.float32)
    for a in range(0, len(ij), 1024):
        batch = ij[a:a + 1024]
        scores[a:a + len(batch)] = np.sum(x[batch[:, 0]] * x[batch[:, 1]], axis=1)
    return scores


def validate_rows(arrays, side, n, content):
    sets = {}
    for row in ROWS:
        a, w = arrays[f'{row}__{side}'], arrays[f'{row}__{side}_w']
        assert a.dtype == np.int64 and a.ndim == 2 and a.shape[1] == 2
        assert w.dtype == np.float32 and w.shape == (len(a),)
        assert np.isfinite(w).all() and np.all((w > 0) & (w <= 1))
        assert np.all((a[:, 0] >= 0) & (a[:, 0] < a[:, 1]) & (a[:, 1] < n))
        codes = a[:, 0] * n + a[:, 1]
        assert np.all(np.diff(codes) > 0), (row, 'unsorted/duplicate')
        sets[row] = set(map(tuple, a.tolist()))
        if row not in ('Gdown025', 'Gdown05'):
            assert np.all(w == 1)
    r0, r1 = sets['R0'], sets['R1']
    assert r1 == r0 - content
    assert sets['Prank5'] <= r0 and sets['Prank10'] <= r0
    assert sets['Prank5'] >= sets['Prank10'] >= r1
    assert sets['Psim50'] <= sets['Psim75'] <= sets['Psim90'] <= r0
    assert sets['Gagree'] <= r0 and r1 <= sets['Gagree']
    for row, alpha in (('Gdown025', .25), ('Gdown05', .5)):
        assert sets[row] == r0
        a = arrays[f'{row}__{side}']
        expected = np.asarray([alpha if tuple(p) in content else 1 for p in a], dtype=np.float32)
        assert np.array_equal(arrays[f'{row}__{side}_w'], expected)
    return sets


def build_partition(side, nodes, spaces, rebuilt, active, outputs, metadata, seconds):
    n = len(nodes)
    x = {key: np.ascontiguousarray(l2(value[nodes])) for key, value in spaces.items() if key != 'E'}
    draws = draw_pairs(n)
    sd = {key: similarity_sd(value, draws) for key, value in x.items()}
    c = rebuilt[side]['C']
    cc = mutual_pairs(spaces['Dc'][nodes]) | mutual_pairs(spaces['Ic'][nodes])
    data = {}
    for aspect in active:
        tick = time.monotonic()
        akey = f'{aspect}.img' if aspect == 'style' else f'{aspect}.cap'
        okey = f'{aspect}.cap' if aspect == 'style' else f'{aspect}.img'
        r0 = rebuilt[side]['E'] if aspect == 'emotion' else source_pairs(spaces[akey][nodes])
        other = source_pairs(spaces[okey][nodes])
        content_key = 'I' if aspect == 'style' else 'D'
        orth = x[akey] - np.sum(x[akey] * x[content_key], axis=1)[:, None] * x[content_key]
        assert np.all(np.linalg.norm(orth, axis=1) > 1e-12)
        g = dict(R0=r0, R0o=other, R1=r0 - c, R1c=r0 - cc,
                 Forth=mutual_pairs(l2(orth)), Gdown025=r0, Gdown05=r0,
                 Gagree=(r0 - c) | (r0 & c & other))
        cs = (pair_scores(x['D'], r0) / sd['D'] + pair_scores(x['I'], r0) / sd['I']) / 2
        thresholds = {}
        for row, q in (('Psim50', .5), ('Psim75', .75), ('Psim90', .9)):
            threshold = float(np.quantile(cs, q))
            thresholds[row] = threshold
            g[row] = {tuple(p) for p, score in zip(pair_array(r0), cs) if score <= threshold}
        deg = np.bincount(pair_array(r0).ravel(), minlength=n)
        shared_deg = np.bincount(pair_array(r0 & c).ravel(), minlength=n)
        lam = np.zeros(n, dtype=np.float32)
        lam[deg > 0] = shared_deg[deg > 0] / deg[deg > 0]
        lam[deg == 0] = lam[deg > 0].mean()
        fused = ('Fres025', 'Fres05', 'Fres1', 'Fside025', 'Fside05', 'Fside075', 'Frank', 'Gres')
        nn = {row: np.empty((n, 20), dtype=np.int64) for row in fused}
        if side == 'fit' and aspect == active[0]:
            nn['zero'] = np.empty((n, 20), dtype=np.int64)
        data[aspect] = dict(g=g, nn=nn, lam=lam)
        metadata[aspect][side] = dict(sd=sd, psim_thresholds=thresholds,
            lambda_mean=float(lam.mean()), clip_content_pairs=len(cc))
        seconds[aspect] += time.monotonic() - tick
    content_nn = {r: {k: np.empty((n, r), dtype=np.int64) for k in ('D', 'I')} for r in (5, 10)}
    # Only block x n scores are held; content scores/ranks are shared by all aspects.
    for start in range(0, n, 256):
        end = min(start + 256, n)
        ds, ims = x['D'][start:end] @ x['D'].T, x['I'][start:end] @ x['I'].T
        content = (ds / sd['D'] + ims / sd['I']) / 2
        content_rank = ranks(content, start)
        for r in (5, 10):
            for key, sim in (('D', ds), ('I', ims)):
                content_nn[r][key][start:end] = top(sim, start, r)
        for aspect, d in data.items():
            tick = time.monotonic()
            capkey, imgkey = f'{aspect}.cap', f'{aspect}.img'
            cap = x[capkey][start:end] @ x[capkey].T
            img = x[imgkey][start:end] @ x[imgkey].T
            primary = img / sd[imgkey] if aspect == 'style' else cap / sd[capkey]
            scores = {row: primary - weight * content for row, weight in
                      (('Fres025', .25), ('Fres05', .5), ('Fres1', 1))}
            for row, weight in (('Fside025', .25), ('Fside05', .5), ('Fside075', .75)):
                scores[row] = weight * cap / sd[capkey] + (1 - weight) * img / sd[imgkey]
            scores['Frank'] = 1 / (60 + ranks(cap, start)) + 1 / (60 + ranks(img, start)) - 1 / (60 + content_rank)
            scores['Gres'] = primary - d['lam'][start:end, None] * content
            if 'zero' in d['nn']:
                scores['zero'] = primary - 0 * content
            for row, score in scores.items():
                d['nn'][row][start:end] = top(score, start)
            seconds[aspect] += time.monotonic() - tick
        if start % 2560 == 0:
            print(f'{side}: fused blocks {end}/{n}', flush=True)
    prune_content = {r: reciprocal(v['D']) | reciprocal(v['I']) for r, v in content_nn.items()}
    for aspect, d in data.items():
        tick = time.monotonic()
        g = d['g']
        for row, nn in d['nn'].items():
            pairs = reciprocal(nn)
            if row == 'zero':
                assert pairs == g['R0'], 'zero-residual fused path differs from mutual_pairs'
                print(f'{aspect}: fit zero-residual path equals R0 ({len(pairs)} pairs)', flush=True)
            else:
                g[row] = pairs
        for r in (5, 10):
            g[f'Prank{r}'] = g['R0'] - prune_content[r]
        for row in ROWS:
            a = pair_array(g[row])
            w = np.ones(len(a), dtype=np.float32)
            if row in ('Gdown025', 'Gdown05'):
                alpha = .25 if row == 'Gdown025' else .5
                w[:] = [alpha if tuple(p) in c else 1 for p in a]
            outputs[aspect][f'{row}__{side}'] = a
            outputs[aspect][f'{row}__{side}_w'] = w
        validate_rows(outputs[aspect], side, n, c)
        seconds[aspect] += time.monotonic() - tick


def load_aspect(aspect, parts, manifest, rebuilt):
    path = BASE / f'{aspect}.npz'
    assert sha_file(path) == manifest['archives'][aspect], f'{aspect}: checksum mismatch'
    with np.load(path, allow_pickle=False) as z:
        expected = {f'{side}_node_ids' for side in parts} | {
            f'{row}__{side}{suffix}' for row in ROWS for side in parts for suffix in ('', '_w')}
        assert set(z.files) == expected
        arrays = {key: z[key] for key in z.files}
    for side, nodes in parts.items():
        assert np.array_equal(arrays[f'{side}_node_ids'], nodes), 'node order mismatch'
        validate_rows(arrays, side, len(nodes), rebuilt[side]['C'])
        assert manifest['counts'][aspect][side] == {row: len(arrays[f'{row}__{side}']) for row in ROWS}
    return arrays


def weighted_stats(pairs, weights, labels):
    out = graph_stats(pairs, labels)
    ij = pair_array(pairs)
    assert len(weights) == len(ij)
    for name, y in labels.items():
        valid = np.all(y[ij] >= 0, axis=1)
        w = weights[valid].astype(np.float64)
        rate = float(np.sum(w * (y[ij[valid, 0]] == y[ij[valid, 1]])) / w.sum()) if len(w) else None
        stat = out['labels'][name]
        random = stat['random_same_label_rate']
        stat.update(labelled_weight_sum=float(w.sum()), same_label_rate=rate,
                    lift=rate / random if rate is not None and random else None)
    return out


def sheet_markdown(sheet):
    lines = ['# Aspect graph sheet', '', 'Fit labels only; development exploration. Overlaps are pair fractions.', '']
    fmt = lambda v: '—' if v is None else f'{v:.6f}'
    for aspect, rows in sheet['aspects'].items():
        lines += [f'## {aspect}', '',
            '| Row | Family | Fit pairs | Dev pairs | Degree 0 | Emotion lift | Style lift | Genre lift | Lift ratio | Overlap R0 | Overlap C | Jaccard R0 |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for row, g in rows.items():
            cells = [row, g['family'], str(g['n_pairs']), str(g['dev_pairs']), str(g['degree_zero'])]
            cells += [fmt(g['labels'][label]['lift']) for label in ('emotion', 'style', 'genre')]
            cells += [fmt(g[key]) for key in ('lift_ratio', 'overlap_R0', 'overlap_C', 'jaccard_R0')]
            lines.append('| ' + ' | '.join(cells) + ' |')
        lines += ['', '| Row | Weight sum | Mean degree | Dropped pairs | Dropped emotion lift | Dropped style lift | Dropped genre lift | Lost all pairs |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for row, g in rows.items():
            d = g['diagnostics']
            lifts = [d['dropped']['labels'][label]['lift'] for label in ('emotion', 'style', 'genre')]
            lines.append(f"| {row} | {g['weight_sum']:.2f} | {g['mean_degree']:.6f} | {d['dropped']['n_pairs']} | " +
                         ' | '.join(fmt(v) for v in lifts) + f" | {d['lost_all_pairs']} |")
        lines += ['', 'Selected: ' + ', '.join(sheet['selected'][aspect]) + '.', '']
    return '\n'.join(lines)


def score_sealed(s, parts, manifest, rebuilt):
    # Read every sealed archive before labels, including aspects from earlier invocations.
    archives = {a: load_aspect(a, parts, manifest, rebuilt) for a in ASPECTS if a in manifest['archives']}
    labels = label_vectors(s, parts['fit'])
    n = len(parts['fit'])
    sheet = dict(identity=manifest['identity'], archives=manifest['archives'], aspects={}, selected={},
                 labels_sha256=hashlib.sha256(b''.join(s[k].tobytes() for k in ('emo', 'gen', 'sty'))).hexdigest())
    for aspect, arrays in archives.items():
        r0 = set(map(tuple, arrays['R0__fit'].tolist()))
        r0deg = np.bincount(arrays['R0__fit'].ravel(), minlength=n)
        rows = {}
        for row in ROWS:
            a = arrays[f'{row}__fit']
            pairs = set(map(tuple, a.tolist()))
            w = arrays[f'{row}__fit_w']
            deg = np.bincount(a.ravel(), minlength=n)
            g = weighted_stats(pairs, w, labels)
            lifts = {name: val['lift'] for name, val in g['labels'].items()}
            assert all(v is not None and np.isfinite(v) for v in lifts.values())
            g.update(family=family(row), dev_pairs=len(arrays[f'{row}__dev']), weight_sum=float(w.sum(dtype=np.float64)),
                degree_zero=int((deg == 0).sum()), mean_degree=float(deg.mean()), target_lift=lifts[aspect],
                lift_ratio=lifts[aspect] / np.mean([v for k, v in lifts.items() if k != aspect]),
                overlap_R0=len(pairs & r0) / len(pairs), overlap_C=len(pairs & rebuilt['fit']['C']) / len(pairs),
                jaccard_R0=len(pairs & r0) / len(pairs | r0),
                diagnostics=dict(dropped=graph_stats(r0 - pairs, labels),
                                 lost_all_pairs=int(((r0deg > 0) & (deg == 0)).sum())))
            rows[row] = g
        eligible = [row for row in ROWS if rows[row]['dev_pairs'] >= 200 and rows[row]['n_pairs'] >= .2 * len(r0)]
        assert 'R0' in eligible and 'R1' in eligible
        selected = ['R0', 'R1']
        for fam in ('prune', 'fuse', 'guided'):
            candidates = [row for row in eligible if family(row) == fam]
            if candidates:
                winner = max(candidates, key=lambda row: rows[row]['lift_ratio'])
                if winner not in selected:
                    selected.append(winner)
        winner = max(eligible, key=lambda row: rows[row]['target_lift'])
        if winner not in selected:
            selected.append(winner)
        sheet['aspects'][aspect], sheet['selected'][aspect] = rows, selected
    atomic_json(BASE / 'sheet.json', sheet)
    atomic_markdown(BASE / 'sheet.md', sheet_markdown(sheet))
    atomic_json(BASE / 'selected.json', sheet['selected'])
    print('Selected: ' + json.dumps(sheet['selected']), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aspects', nargs='+', choices=list(ASPECTS), default=list(ASPECTS))
    parser.add_argument('--stage0-only', action='store_true')
    args = parser.parse_args()
    start = time.monotonic()
    s = load_frozen()
    _, fit, dev = split(s)
    parts = dict(fit=fit, dev=dev)
    BASE.mkdir(parents=True, exist_ok=True)
    spaces = dict(D=targets(s, 'default'), I=default_images(s)[0], E=targets(s, 'emotion_focus'))
    rebuilt = stage0(s, parts, spaces)
    if args.stage0_only:
        return
    identity = dict(method=METHOD, sample_sha256=fingerprint(s), fit_size=len(fit), dev_size=len(dev))
    marker = BASE / 'manifest.json'
    manifest = json.loads(marker.read_text()) if marker.exists() else dict(
        identity=identity, archives={}, counts={}, metadata={}, seconds={}, source_sha256={})
    assert manifest['identity'] == identity, 'graph identity mismatch'
    active = []
    for aspect in dict.fromkeys(args.aspects):
        if aspect in manifest['archives']:
            load_aspect(aspect, parts, manifest, rebuilt)
            print(f'{aspect}: sealed identity/checksum verified; skip', flush=True)
        else:
            assert not (BASE / f'{aspect}.npz').exists(), 'unsealed archive exists; inspect before reuse'
            active.append(aspect)
    if active:
        shared_start = time.monotonic()
        spaces.update(clip_content(s))
        for aspect in active:
            spaces[f'{aspect}.cap'] = spaces['E'] if aspect == 'emotion' else targets(s, ASPECTS[aspect])
            spaces[f'{aspect}.img'] = aspect_images(s, ASPECTS[aspect])
        outputs = {a: {f'{side}_node_ids': nodes for side, nodes in parts.items()} for a in active}
        metadata, seconds = {a: {} for a in active}, {a: 0. for a in active}
        for side, nodes in parts.items():
            build_partition(side, nodes, spaces, rebuilt, active, outputs, metadata, seconds)
        shared_seconds = time.monotonic() - shared_start - sum(seconds.values())
        for aspect in active:
            path = BASE / f'{aspect}.npz'
            atomic_npz(path, **outputs[aspect])
            manifest['archives'][aspect] = sha_file(path)
            manifest['counts'][aspect] = {side: {row: len(outputs[aspect][f'{row}__{side}']) for row in ROWS} for side in parts}
            manifest['metadata'][aspect] = metadata[aspect]
            manifest['seconds'][aspect] = dict(aspect=seconds[aspect], shared=shared_seconds)
            keys = ('D', 'I', 'Dc', 'Ic', f'{aspect}.cap', f'{aspect}.img')
            manifest['source_sha256'][aspect] = {key: hashlib.sha256(np.ascontiguousarray(spaces[key]).tobytes()).hexdigest() for key in keys}
            atomic_json(marker, manifest)
            print(f'{aspect}: sealed 21 rows; aspect={seconds[aspect]:.1f}s shared={shared_seconds:.1f}s', flush=True)
    score_sealed(s, parts, manifest, rebuilt)
    print(f'Graph sheet PASS: {time.monotonic() - start:.1f}s; {BASE}', flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=8):
        main()
