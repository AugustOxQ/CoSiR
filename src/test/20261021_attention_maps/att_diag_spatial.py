"""Spatial diagnostics for the frozen Qwen attention store; no extraction/cache shortcut.

Examples (use the project's CoSiR Python and PYTHONDONTWRITEBYTECODE=1):
  att_diag_spatial.py --parts structure,sheet --out /project/CoSiR/outputs/spatial
  att_diag_spatial.py --device cuda --parts geometry,recompute,sinks,objects
  att_diag_spatial.py --device cpu --parts geometry --geometry-limit 1 --out ...

The last command is an explicitly incomplete, one-image CPU smoke. C always scans
the entire store; --n-paintings controls D, while B always checks the first four.
No files are deleted. Reruns retain unrequested results and replace requested parts.
"""

import argparse
import itertools
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PAV = REPO / 'src/test/20261017_prompted_aspect_views'
for _path in (REPO, PAV, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
import att_core as C

STORE = Path('/data/SSD2/pre_extract/artelingo_full/qwen3vl_emb_2b_attention')
DEFAULT_OUT = Path('/tmp/claude-0/-project-CoSiR/e5b42f77-07da-4e87-9759-def96bd42c89/scratchpad/diag/out')
PARTS = ('geometry', 'recompute', 'structure', 'sinks', 'objects', 'sheet')


def stats(values):
    x = np.asarray(values, dtype=float).ravel()
    good = x[np.isfinite(x)]
    if not len(good):
        return dict(n=0, undefined=int(len(x)), median=None, p10=None, p90=None)
    p10, med, p90 = np.percentile(good, [10, 50, 90])
    return dict(n=int(len(good)), undefined=int(len(x) - len(good)),
                median=float(med), p10=float(p10), p90=float(p90))


def normalize(x):
    x = np.asarray(x, np.float64)
    total = x.sum(axis=-1, keepdims=True)
    if np.any(total <= 0) or not np.isfinite(x).all():
        raise ValueError('nonpositive or nonfinite attention mass')
    return x / total


def top_indices(x, n):
    # Stable ties make the exact top-k definition reproducible (no inflated tie sets).
    return np.argsort(-np.asarray(x), kind='stable')[:n]


def top_count(n, fraction):
    return max(1, math.ceil(n * fraction))


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b)


def border(grid):
    b = np.zeros(grid, bool)
    b[[0, -1], :] = True
    b[:, [0, -1]] = True
    return b.ravel()


def moran(x, grid):
    """Global Moran I, symmetric binary rook weights, no row standardisation.

    Each undirected edge occurs twice in numerator and S0, cancelling the two.
    Constant maps are undefined (NaN internally, null plus a count in summaries).
    """
    z = np.asarray(x, np.float64).reshape(grid)
    z = z - z.mean()
    den = np.square(z).sum()
    r, c = grid
    edges = (r - 1) * c + r * (c - 1)
    if den <= 0 or edges == 0:
        return float('nan')
    products = (z[:-1] * z[1:]).sum() + (z[:, :-1] * z[:, 1:]).sum()
    return float(z.size * products / (edges * den))


def plot_module():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


class Images:
    """Exactly the adapter/sample and image preparation used by Extractor and Ctx."""

    def __init__(self):
        import pav_sample
        from scripts.extract_features import adapter_artelingo_full
        from src.data.feature_extract import _qwen_prepare_image, QWEN_MAX_PIXELS
        pav_sample.SAMPLE = PAV / 'sample.npz'
        self.sample = pav_sample.load_frozen()
        self.spec = adapter_artelingo_full(sample_ids=self.sample['sample_ids'], expected=None)
        self.prepare = _qwen_prepare_image
        self.max_pixels = QWEN_MAX_PIXELS
        assert np.array_equal([self.spec.images[int(i)]['painting']
                               for i in self.sample['subset_image_indices']],
                              self.sample['subset_painting_ids'])

    def painting(self, pm):
        k = int(pm['k'])
        assert int(pm['image_index']) == int(self.sample['subset_image_indices'][k])
        assert str(pm['painting_id']) == str(self.sample['subset_painting_ids'][k])
        im = self.spec.load_image(int(pm['image_index']))
        prepared = self.prepare(im, self.max_pixels)
        im.close()
        r, c = map(int, pm['grid_hw'])
        assert prepared.size == (32 * c, 32 * r)
        return prepared


class PlainModel:
    def __init__(self, device, images):
        import torch
        from pav_prompts import PROMPTS
        from src.data.feature_extract import Qwen3VLEmb
        self.torch, self.prompts = torch, PROMPTS
        self.device, self.images = device, images
        self.dtype = torch.bfloat16 if device == 'cuda' else torch.float32
        self.enc = Qwen3VLEmb(device=device)
        if device == 'cpu':
            self.enc.m.float()
        self.enc.m.set_attn_implementation('eager')
        self.enc.m.eval()
        assert self.enc.max_pixels == images.max_pixels
        tok = self.enc.p.tokenizer
        assert tok.convert_tokens_to_ids('<|endoftext|>') == C.ENDOFTEXT

    def run(self, im, view, geometry=False, sinks=False):
        """One full batch-1 forward; only compact CPU measurements leave this call."""
        torch, enc = self.torch, self.enc
        enc.instruction = self.prompts[view]
        prepared = self.images.prepare(im, enc.max_pixels)
        o = enc.p(text=[enc._prompt([{'type': 'image'}])], images=[prepared],
                  do_resize=False, return_tensors='pt')
        ids = o['input_ids'][0].numpy()
        tt = C.token_types(ids, 'img')
        thw = o['image_grid_thw'][0].numpy()
        grid = (int(thw[1]) // 2, int(thw[2]) // 2)
        assert thw[0] == 1 and prepared.size == (grid[1] * 32, grid[0] * 32)
        assert int((tt == 2).sum()) == grid[0] * grid[1]
        assert ids[-1] == C.ENDOFTEXT
        assert int(o['attention_mask'][0].sum()) == len(ids)
        assert np.array_equal(o['mm_token_type_ids'][0].numpy() == 1, tt == 2)
        kw = {k: v.to(self.device, self.dtype) if v.is_floating_point() else v.to(self.device)
              for k, v in o.items()}
        with torch.inference_mode():
            out = enc.m(**kw, use_cache=False, output_attentions=True, output_hidden_states=True)
            assert len(out.attentions) == C.N_LAYERS
            assert len(out.hidden_states) == C.N_LAYERS + 1
            # Independent raw row: deliberately no att_core.head_mean_norm or cached prefix.
            rows = torch.stack([a[0, :, -1, :].float().mean(0) for a in out.attentions])
            result = dict(ids=ids, ttype=tt, grid=grid, layers=rows.cpu().numpy(),
                          query_index=len(ids) - 1, query_token=enc.p.tokenizer.decode([int(ids[-1])]))
            pos = torch.as_tensor(np.flatnonzero(tt == 2), device=self.device)
            if geometry:
                result['geometry_vectors'] = {
                    str(l): out.hidden_states[l][0, pos].float().cpu().numpy() for l in (0, 8)}
                # An independent positive calibration of the processor stream. Four
                # consecutive 16px patches form one merged 32px token. This is not
                # a substitute for recovering regions from the model's embeddings.
                result['pixel_vectors'] = o['pixel_values'].float().numpy().reshape(len(pos), -1)
            if sinks:
                _, start, end = C.BANDS[0]
                # hidden_states[0] is LLM input; layer l's output is [l+1].
                # The final entry includes the model's terminal RMSNorm.
                norm, maxabs = [], []
                for l in range(start, end):
                    h = out.hidden_states[l + 1][0].float()
                    norm.append(h.norm(dim=-1))
                    maxabs.append(h.abs().amax(dim=-1))
                result['norm'] = torch.stack(norm).mean(0).cpu().numpy()
                result['maxabs'] = torch.stack(maxabs).amax(0).cpu().numpy()
                result['mean_layer_maxabs'] = torch.stack(maxabs).mean(0).cpu().numpy()
                heads = torch.stack([a[0, :, -1, pos].float() for a in out.attentions[start:end]]).sum(0)
                result['heads'] = normalize(heads.cpu().numpy())
        prepared.close()
        return result


def kmeans(vectors, k, rng, restarts=12):
    """Lloyd k-means with k-means++ restarts; labels never consult region truth."""
    x = np.asarray(vectors, np.float64)
    x = x - x.mean(0)
    x /= max(float(np.sqrt(np.mean(x * x))), 1e-12)
    xx = np.square(x).sum(1)
    best = None
    for _ in range(restarts):
        centers = [x[int(rng.integers(len(x)))]]
        nearest = np.full(len(x), np.inf)
        for _ in range(1, k):
            nearest = np.minimum(nearest, np.square(x - centers[-1]).sum(1))
            idx = int(rng.choice(len(x), p=nearest / nearest.sum())) if nearest.sum() else 0
            centers.append(x[idx])
        centers = np.asarray(centers)
        previous = None
        for _ in range(100):
            distances = np.maximum(xx[:, None] + np.square(centers).sum(1) - 2 * x @ centers.T, 0)
            labels = distances.argmin(1)
            if previous is not None and np.array_equal(labels, previous):
                break
            previous = labels.copy()
            centers = np.asarray([x[labels == j].mean(0) if np.any(labels == j)
                                  else x[distances.min(1).argmax()] for j in range(k)])
        loss = float(np.square(x - centers[labels]).sum())
        if best is None or loss < best[0]:
            best = (loss, labels.copy())
    return best[1]


def matching_score(labels, truth, k):
    valid = truth >= 0
    return max(float(np.mean(np.asarray(p)[labels[valid]] == truth[valid]))
               for p in itertools.permutations(range(k)))


def cell_truth(pixel_labels):
    h, w = pixel_labels.shape
    blocks = pixel_labels.reshape(h // 32, 32, w // 32, 32)
    lo, hi = blocks.min(axis=(1, 3)), blocks.max(axis=(1, 3))
    return np.where(lo == hi, lo, -1)


def synthetic_regions():
    from PIL import Image
    configs = [('a_left_red', 640, 384), ('b_top_white', 384, 640),
               ('c_square', 640, 384), ('d_quadrants', 384, 640)]
    for name, w, h in configs:
        truth = np.zeros((h, w), np.int8)
        if name.startswith('a'):
            truth[:, 3 * w // 8:] = 1
            colors = [(255, 0, 0), (0, 0, 255)]
        elif name.startswith('b'):
            truth[h // 4:] = 1
            colors = [(255, 255, 255), (0, 0, 0)]
        elif name.startswith('c'):
            truth[224:320, 128:224] = 1
            colors = [(128, 128, 128), (255, 255, 255)]
        else:
            truth[:h // 2, w // 2:] = 1
            truth[h // 2:, :w // 2] = 2
            truth[h // 2:, w // 2:] = 3
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        yield name, Image.fromarray(np.asarray(colors, np.uint8)[truth]), cell_truth(truth), len(colors)


def geometry_base(name, image):
    """Paired base X and binary changed-cell truth, independent of model outputs."""
    from PIL import Image, ImageDraw
    if name.startswith('a'):
        base = Image.new('RGB', image.size, (0, 0, 255))
    elif name.startswith('b'):
        base = Image.new('RGB', image.size, (0, 0, 0))
    elif name.startswith('c'):
        base = Image.new('RGB', image.size, (128, 128, 128))
    else:
        base = image.copy()
        # Only top-right changes: red in X, original green in Y.
        ImageDraw.Draw(base).rectangle((image.width // 2, 0, image.width - 1,
                                        image.height // 2 - 1), fill=(255, 0, 0))
    changed = np.any(np.asarray(base) != np.asarray(image), axis=-1).astype(np.int8)
    return base, cell_truth(changed)


def two_means_split(values):
    """Exact minimum-SSE 1-D two-means (Otsu), with the high group = changed.

    Search only distinct adjacent values. Neither region truth nor token positions
    enter the split; all cells, including boundary cells, contribute to fitting it.
    """
    x = np.sort(np.asarray(values, np.float64).ravel())
    if len(x) < 2 or not np.isfinite(x).all() or x[0] == x[-1]:
        raise ValueError('differential distances need at least two distinct finite values')
    # Scaling avoids overflow and leaves the optimal split unchanged.
    scaled = (x - x[0]) / (x[-1] - x[0])
    left_n = np.arange(1, len(x), dtype=np.float64)
    right_n = len(x) - left_n
    sums = np.cumsum(scaled)
    mean_gap = sums[:-1] / left_n - (sums[-1] - sums[:-1]) / right_n
    separation = left_n * right_n * np.square(mean_gap)
    separation[x[:-1] == x[1:]] = -np.inf
    cut = int(separation.argmax())
    return float(x[cut] + (x[cut + 1] - x[cut]) / 2)


def geometry(model, args):
    plt = plot_module()
    fixtures = list(synthetic_regions())[:args.geometry_limit]
    fig, axes = plt.subplots(len(fixtures), 5, figsize=(15, 3 * len(fixtures)), squeeze=False)
    records = []
    for i, (name, image, truth, k) in enumerate(fixtures):
        p = model.run(image, 'default', geometry=True)
        base, changed_truth = geometry_base(name, image)
        reference = model.run(base, 'default', geometry=True)
        grid = p['grid']
        assert grid == truth.shape and image.size == (grid[1] * 32, grid[0] * 32)
        assert reference['grid'] == grid and np.array_equal(p['ids'], reference['ids'])
        assert np.array_equal(p['ttype'], reference['ttype'])
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(name)
        valid = changed_truth >= 0
        rec = dict(image=name, grid_hw=list(grid), valid_cells=int(valid.sum()),
                   changed_cells=int((changed_truth == 1).sum()), layers={})
        pixel_labels = kmeans(p['pixel_vectors'], k, np.random.default_rng(args.seed + i))
        rec['processor_calibration'] = {
            'true': matching_score(pixel_labels.reshape(grid), truth, k),
            'transpose': matching_score(pixel_labels.reshape(grid[1], grid[0]).T, truth, k),
            'random': matching_score(np.random.default_rng(args.seed + i).permutation(pixel_labels).reshape(grid), truth, k)}
        for li, layer in enumerate(('0', '8')):
            rng = np.random.default_rng(np.random.SeedSequence([args.seed, i, int(layer)]))
            delta = p['geometry_vectors'][layer].astype(np.float64) - reference['geometry_vectors'][layer]
            distances = np.linalg.norm(delta, axis=-1)
            threshold = two_means_split(distances)
            true = distances.reshape(grid)
            transposed = distances.reshape(grid[1], grid[0]).T
            shuffled = rng.permutation(distances).reshape(grid)
            scores = {order: float(np.mean((d[valid] > threshold) == changed_truth[valid]))
                      for order, d in [('true', true), ('transpose', transposed), ('random', shuffled)]}
            controls_required = not name.startswith('c')
            controls_pass = scores['true'] - scores['transpose'] >= 0.10
            scores.update(threshold=threshold, min_distance=float(distances.min()),
                          max_distance=float(distances.max()), predicted_changed_cells=int((distances > threshold).sum()))
            if layer == '0':
                scores['pass'] = scores['true'] >= 0.95 and (not controls_required or controls_pass)
            rec['layers'][layer] = scores
            for col, (d, order) in enumerate(((true, 'true'), (transposed, 'transpose')), start=1 + 2 * li):
                axes[i, col].imshow(d, vmin=0, vmax=distances.max(), cmap='viridis', interpolation='nearest')
                axes[i, col].set_title(f'||delta hidden[{layer}]|| {order}: {scores[order]:.3f}')
        records.append(rec)
        image.close()
        base.close()
        print(f'geometry {name}: ' + json.dumps(rec['layers']), flush=True)
    for ax in axes.ravel():
        ax.axis('off')
    fig.tight_layout()
    path = args.out / 'geometry.png'
    fig.savefig(path, dpi=140)
    plt.close(fig)
    complete = len(fixtures) == 4
    passed = all(r['layers']['0']['pass'] for r in records)
    return dict(status=('PASS' if complete else 'PARTIAL') if passed else 'FAIL',
                complete=complete, smoke_checks_pass=passed, records=records, figure=str(path),
                rule='Differential hidden[0]: true >= .95 for all layouts; a,b,d transpose >= .10 lower. '
                'Hidden[8] and random controls are reported, not gated.',
                interpretation='Per-cell d = L2(h_Y - h_X); an exact 1-D two-means split of d alone predicts '
                'the high-distance group as changed. Agreement excludes boundary-straddling cells. '
                'Position terms cancel between paired inputs; vision/LLM mixing can spread differences. '
                'Figure pairs share a colour scale within each layer. Processor calibration is unchanged.')


def viewer_check():
    query = urlencode(dict(k=0, mod='img', layer=C.BANDS[0][0]))
    url = 'http://localhost:8765/api/maps?' + query
    try:
        with urlopen('http://localhost:8765/', timeout=5) as response:
            response.read(128)
        with urlopen(url, timeout=10) as response:
            payload = json.load(response)
    except HTTPError as exc:
        return dict(status='FAIL', url=url, reason=str(exc))
    except (URLError, TimeoutError, OSError) as exc:
        return dict(status='SKIP', url=url, reason=str(exc))
    stored = C.load_painting(STORE, 0)
    grid = tuple(map(int, stored['painting']['grid_hw']))
    errors = {}
    for view in C.VIEWS:
        got = np.asarray(payload['views'][view]['image_map'], float)
        reference = C.band_map(stored[view]['img'][0], C.BANDS[0])[0].reshape(grid)
        if got.shape != reference.shape:
            return dict(status='FAIL', url=url, reason=f'{view}: shape mismatch')
        errors[view] = float(np.max(np.abs(got - reference)))
    return dict(status='PASS' if max(errors.values()) <= 1e-4 else 'FAIL',
                url=url, max_abs_errors=errors)


def sink_record(k, view, p, stored):
    from scipy.stats import spearmanr
    image = p['ttype'] == 2
    norms, maxabs = p['norm'][image], p['maxabs'][image]
    m = C.band_map(stored, C.BANDS[0])[0]
    b = border(p['grid'])
    norm5 = top_indices(norms, 5)
    att5 = top_indices(m, 5)
    fractions = {}
    for fraction in (0.01, 0.05):
        ix = top_indices(norms, top_count(len(m), fraction))
        fractions[str(fraction)] = dict(attention_mass=float(m[ix].sum()), token_share=len(ix) / len(m))
    head_results = [dict(head=i, moran=moran(h, p['grid']),
                         top_5pct_mass=float(h[top_indices(h, top_count(len(h), .05))].sum()))
                    for i, h in enumerate(p['heads'])]
    assert len(head_results) == 16
    return dict(k=k, view=view, grid_hw=list(p['grid']),
                norm_attention_spearman=float(spearmanr(norms, m).statistic),
                maxabs_attention_spearman=float(spearmanr(maxabs, m).statistic),
                highest_norm_mass=fractions, top5_intersection=len(set(norm5) & set(att5)),
                norm_top5_indices=norm5.tolist(), attention_top5_indices=att5.tolist(),
                norm_top5_border_share=float(b[norm5].mean()), norm_top1_on_border=bool(b[norm5[0]]),
                border_token_share=float(b.mean()),
                norms={name: dict(stats(p['norm'][mask]), max=float(p['norm'][mask].max()))
                       for name, mask in [('image', image), ('text_instruction', p['ttype'] == 1),
                                          ('template', p['ttype'] == 0)] if mask.any()},
                image_norm_max_over_text_median=float(norms.max() / np.median(p['norm'][p['ttype'] == 1])),
                image_norm_max_over_image_median=float(norms.max() / np.median(norms)),
                # Preserve per-token measurements to make sink claims independently inspectable.
                image_token_mean_l2=norms.tolist(), image_token_max_abs=maxabs.tolist(),
                image_token_mean_layer_max_abs=p['mean_layer_maxabs'][image].tolist(), heads=head_results)


def real_passes(model, images, args, want_recompute, want_sinks):
    l1, sink_rows, cross_views = [], [], []
    count = max(4 if want_recompute else 0, args.n_paintings if want_sinks else 0)
    for k in range(count):
        painting = C.load_painting(STORE, k)
        im = images.painting(painting['painting'])
        norm_sets = {}
        views = C.VIEWS if want_sinks and k < args.n_paintings else C.VIEWS[:2]
        for view in views:
            need_b = want_recompute and k < 4 and view in C.VIEWS[:2]
            need_d = want_sinks and k < args.n_paintings
            if not (need_b or need_d):
                continue
            p = model.run(im, view, sinks=need_d)
            stored = painting[view]['img'][0]
            assert tuple(map(int, painting['painting']['grid_hw'])) == p['grid']
            assert np.array_equal(p['ids'], stored['ids']), f'{k}/{view}: token ids changed'
            assert np.array_equal(p['ttype'], stored['ttype'])
            if need_b:
                for band in C.BANDS:
                    _, start, end = band
                    # Sum raw head means before selecting and normalizing image columns.
                    fresh = normalize(p['layers'][start:end].astype(np.float64).sum(0)[p['ttype'] == 2])
                    reference = C.band_map(stored, band)[0]
                    distance = float(np.abs(fresh - reference).sum())
                    l1.append(dict(k=k, view=view, band=band[0], l1=distance,
                                   query_index=p['query_index'], query_token=p['query_token'],
                                   query_token_id=int(p['ids'][-1]), stored_last_token_id=int(stored['ids'][-1])))
            if need_d:
                rec = sink_record(k, view, p, stored)
                sink_rows.append(rec)
                norm_sets[view] = rec['norm_top5_indices']
            print(f'plain pass k={k} view={view} B={need_b} D={need_d}', flush=True)
        if norm_sets:
            for a, b in itertools.combinations(C.VIEWS, 2):
                cross_views.append(dict(k=k, views=[a, b], top5_norm_jaccard=jaccard(norm_sets[a], norm_sets[b])))
        im.close()
    out = {}
    if want_recompute:
        viewer = viewer_check()
        passed = all(r['l1'] <= .15 for r in l1) and viewer['status'] != 'FAIL'
        out['recompute'] = dict(status='PASS' if passed else 'FAIL', records=l1,
                                max_l1=max(r['l1'] for r in l1), viewer=viewer,
                                rule='All 24 L1 distances <= .15; answering viewer within 1e-4; final token is endoftext.')
    if want_sinks:
        summaries = {}
        keys = ('norm_attention_spearman', 'maxabs_attention_spearman', 'top5_intersection',
                'norm_top5_border_share', 'norm_top1_on_border', 'border_token_share',
                'image_norm_max_over_text_median', 'image_norm_max_over_image_median')
        for view in C.VIEWS:
            rows = [r for r in sink_rows if r['view'] == view]
            summaries[view] = {key: stats([r[key] for r in rows]) for key in keys}
            summaries[view]['highest_norm_mass'] = {
                f: stats([r['highest_norm_mass'][f]['attention_mass'] for r in rows]) for f in ('0.01', '0.05')}
            summaries[view]['heads'] = {
                str(h): {key: stats([r['heads'][h][key] for r in rows])
                         for key in ('moran', 'top_5pct_mass')} for h in range(16)}
        out['sinks'] = dict(status='DESCRIPTIVE', n_paintings=args.n_paintings,
                            records=sink_rows, by_view=summaries, cross_view=cross_views,
                            cross_view_jaccard=stats([r['top5_norm_jaccard'] for r in cross_views]),
                            hidden_state_convention='Attention layer l uses hidden_states[l+1] output; '
                            'layer 27 output has terminal RMSNorm. Mean L2 across outputs 16..27; '
                            'max_abs is the maximum over their layers and channels. Text = instruction tokens.')
    return out


def structure_worker(task):
    k, seed = task
    painting = C.load_painting(STORE, k)
    grid = tuple(map(int, painting['painting']['grid_hw']))
    b = border(grid)
    rows = []
    for bi, band in enumerate(C.BANDS):
        maps = {v: C.band_map(painting[v]['img'][0], band)[0] for v in C.VIEWS}
        default5 = top_indices(maps['default'], 5)
        for vi, view in enumerate(C.VIEWS):
            m = maps[view]
            assert len(m) == grid[0] * grid[1]
            rng = np.random.default_rng(np.random.SeedSequence([seed, k, vi, bi]))
            shuffled = rng.permutation(m)
            row = dict(moran=moran(m, grid), shuffled_moran=moran(shuffled, grid),
                       top_1pct_mass=float(m[top_indices(m, top_count(len(m), .01))].sum()),
                       top_5pct_mass=float(m[top_indices(m, top_count(len(m), .05))].sum()),
                       border_mass=float(m[b].sum()), border_token_share=float(b.mean()),
                       border_enrichment=float(m[b].sum() / b.mean()),
                       top5_default_jaccard=jaccard(top_indices(m, 5), default5))
            rows.append((view, band[0], row))
    return rows


def structure(args):
    ks = sorted(int(p.stem[1:]) for p in STORE.glob('p[0-9][0-9][0-9][0-9].npz'))
    assert ks == list(range(5000)), f'expected all 5000 stored paintings, got {len(ks)}'
    workers = min(8, os.cpu_count() or 1)
    grouped = {v: {b[0]: {} for b in C.BANDS} for v in C.VIEWS}
    started = time.monotonic()
    # Spawn is safe even when another requested part has already initialised torch/CUDA.
    with mp.get_context('spawn').Pool(workers) as pool:
        for i, rows in enumerate(pool.imap(structure_worker, ((k, args.seed) for k in ks), chunksize=16), 1):
            for view, band, row in rows:
                group = grouped[view][band]
                for key, value in row.items():
                    group.setdefault(key, []).append(value)
            if i % 1000 == 0:
                print(f'structure {i}/5000 ({time.monotonic() - started:.1f}s)', flush=True)
    result = {v: {b: {metric: stats(vals) for metric, vals in metrics.items()}
                  for b, metrics in bands.items()} for v, bands in grouped.items()}
    return dict(status='DESCRIPTIVE', n_paintings=len(ks), workers=workers,
                seconds=time.monotonic() - started, by_view_band=result,
                definition='Binary symmetric rook Moran I; ceil(percent*N) top masses; exact top-5 Jaccard; '
                'one independently seeded shuffle per map. Constant-map I is undefined.')


def overlay(ax, image, m, rank=False):
    ax.imshow(image)
    r, c = m.shape
    if rank:
        from scipy.stats import rankdata
        values = (rankdata(m.ravel(), method='average') - 1).reshape(r, c) / max(m.size - 1, 1)
    else:
        values = m / max(float(m.max()), 1e-300)
    ax.imshow(values, cmap='jet', alpha=.5, interpolation='nearest', vmin=0, vmax=1,
              extent=(-.5, image.width - .5, image.height - .5, -.5))
    ax.axis('off')


def sheet(images, args):
    import io
    from PIL import Image
    plt = plot_module()
    figures = [plt.subplots(8, 5, figsize=(16, 22)) for _ in range(2)]
    ks = list(range(0, 5000, 625))
    for row, k in enumerate(ks):
        painting = C.load_painting(STORE, k)
        prepared = images.painting(painting['painting'])
        # Match Ctx.image_jpeg's quality=90, including the background shown in the browser.
        buf = io.BytesIO()
        prepared.save(buf, 'JPEG', quality=90)
        prepared.close()
        buf.seek(0)
        image = Image.open(buf).convert('RGB')
        grid = tuple(map(int, painting['painting']['grid_hw']))
        for rank, (_, axes) in enumerate(figures):
            axes[row, 0].imshow(image)
            axes[row, 0].axis('off')
            axes[row, 0].set_title(f'k={k}')
            for col, view in enumerate(C.VIEWS, 1):
                m = C.band_map(painting[view]['img'][0], C.BANDS[0])[0].reshape(grid)
                overlay(axes[row, col], image, m, rank=bool(rank))
                if row == 0:
                    axes[row, col].set_title(view)
        image.close()
    paths = []
    for (fig, _), name in zip(figures, ('sheet_values.png', 'sheet_ranks.png')):
        fig.suptitle('Layers 16–27; image only; ' + ('percentile ranks' if 'ranks' in name else 'value / map max'))
        fig.tight_layout(rect=(0, 0, 1, .98))
        path = args.out / name
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(str(path))
    return dict(status='DESCRIPTIVE', paintings=ks, figures=paths,
                rendering='Prepared image JPEG quality 90; jet alpha .5, nearest; ranks use average ties.')


def object_fixtures():
    from PIL import Image, ImageDraw
    w, h = 640, 448
    for kind in ('disc', 'face'):
        for i, (cx, cy) in enumerate(((160, 112), (480, 112), (160, 336), (480, 336))):
            image = Image.new('RGB', (w, h), (128, 128, 128))
            mask = Image.new('L', (w, h), 0)
            box = (cx - 48, cy - 48, cx + 47, cy + 47)
            draw = ImageDraw.Draw(image)
            draw.ellipse(box, fill=(255, 0, 0) if kind == 'disc' else (244, 188, 145))
            ImageDraw.Draw(mask).ellipse(box, fill=255)
            if kind == 'face':
                for ex in (cx - 17, cx + 17):
                    draw.ellipse((ex - 5, cy - 16, ex + 5, cy - 6), fill=(20, 20, 20))
                draw.arc((cx - 22, cy - 7, cx + 22, cy + 28), 15, 165, fill=(60, 20, 20), width=4)
            pixel_mask = np.asarray(mask) > 0
            center_cells = pixel_mask[16::32, 16::32]
            coverage = pixel_mask.reshape(h // 32, 32, w // 32, 32).mean(axis=(1, 3))
            yield f'{kind}_{i}', image, (cx, cy), center_cells, coverage
            mask.close()


def objects(model, args):
    plt = plot_module()
    fixtures = list(object_fixtures())
    fig, axes = plt.subplots(8, 5, figsize=(17, 20))
    records = []
    for row, (name, image, center, mask, coverage) in enumerate(fixtures):
        axes[row, 0].imshow(image)
        axes[row, 0].set_title(name)
        axes[row, 0].axis('off')
        for vi, view in enumerate(C.VIEWS[:2]):
            p = model.run(image, view)
            assert p['grid'] == mask.shape
            for bi, band in enumerate(C.BANDS[:2]):
                m = C.band_map(p, band)[0].reshape(p['grid'])
                rr, cc = np.unravel_index(m.argmax(), m.shape)
                share, mass = float(mask.mean()), float(m[mask].sum())
                records.append(dict(object=name, center_xy=list(center), view=view, band=band[0],
                                    object_mass=mass, object_token_share=share, enrichment=mass / share,
                                    object_hottest_rank=1 + int((m > m[mask].max()).sum()),
                                    n_cells=int(m.size),
                                    fractional_cell_mass=float((m * coverage).sum()),
                                    fractional_token_share=float(coverage.mean()),
                                    hottest_row_col=[int(rr), int(cc)],
                                    hottest_pixel_center_xy=[int(32 * cc + 16), int(32 * rr + 16)],
                                    hottest_inside_object=bool(mask[rr, cc])))
                ax = axes[row, 1 + 2 * vi + bi]
                overlay(ax, image, m)
                ax.plot(32 * cc + 16, 32 * rr + 16, 'wx', ms=7)
                ax.set_title(f'{view}\n{band[0]} mass={mass:.3f} / share={share:.3f}', fontsize=8)
        image.close()
        print(f'objects {name} done', flush=True)
    fig.tight_layout()
    path = args.out / 'objects.png'
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return dict(status='DESCRIPTIVE', records=records, figure=str(path),
                definition='Object cells have their 32x32 cell centre inside the disc silhouette; '
                'fractional metrics weight each cell by silhouette pixel coverage.')


def format_stats(s):
    if s['median'] is None:
        return 'undefined'
    return f"{s['median']:.4f} [{s['p10']:.4f}, {s['p90']:.4f}]"


def summary(results, out):
    lines = ['# Spatial attention diagnostics', '',
             'Medians [10th, 90th percentiles]. Descriptive results do not establish object grounding.',
             'A/B checks completed on CPU float32 are provisional for the controller’s CUDA bf16 run.', '']
    for part in PARTS:
        if part not in results:
            continue
        r = results[part]
        lines += [f'## {part}', '', f"{r['status']}", '']
        if r['status'] == 'ERROR':
            lines += [r['error'], '']
            continue
        if part == 'geometry':
            lines += [r['rule'], r.get('interpretation', 'Hidden-state clustering failure alone does not '
                      'distinguish an ordering error from nonseparable colour embeddings.'), '',
                      '| Image | Hidden index | True | Transpose | Random |',
                      '|---|---:|---:|---:|---:|']
            for row in r['records']:
                for layer, scores in row['layers'].items():
                    lines.append(f"| {row['image']} | {layer} | {scores['true']:.4f} | "
                                 f"{scores['transpose']:.4f} | {scores['random']:.4f} |")
                scores = row['processor_calibration']
                lines.append(f"| {row['image']} (processor calibration only) | pixels | {scores['true']:.4f} | "
                             f"{scores['transpose']:.4f} | {scores['random']:.4f} |")
            if not r['complete']:
                lines += ['', 'PARTIAL smoke only: no complete Part A verdict.']
        elif part == 'recompute':
            lines += [f"Max L1: {r['max_l1']:.5f}; viewer: {json.dumps(r['viewer'])}", '',
                      '| k | View | Band | L1 |', '|---:|---|---|---:|']
            for row in r['records']:
                lines.append(f"| {row['k']} | {row['view']} | {row['band']} | {row['l1']:.5f} |")
            lines += ['', 'Query row is the final unpadded token, verified `<|endoftext|>` (151643).']
        elif part == 'structure':
            lines += [f"{r['n_paintings']} paintings; {r['workers']} workers; {r['seconds']:.1f} seconds.",
                      r['definition'], '',
                      '| View | Band | Moran I | Shuffle I | Top 1% | Top 5% | Border mass | Border share | '
                      'Border enrichment | Top-5 Jaccard/default |', '|---|---|---|---|---|---|---|---|---|---|']
            keys = ('moran', 'shuffled_moran', 'top_1pct_mass', 'top_5pct_mass', 'border_mass',
                    'border_token_share', 'border_enrichment', 'top5_default_jaccard')
            for view, bands in r['by_view_band'].items():
                for band, s in bands.items():
                    lines.append(f'| {view} | {band} | ' + ' | '.join(format_stats(s[key]) for key in keys) + ' |')
        elif part == 'sinks':
            lines += [r['hidden_state_convention'], '',
                      f"Cross-view top-5 norm Jaccard: {format_stats(r['cross_view_jaccard'])}.", '',
                      '| View | Norm rho | Maxabs rho | Top 1% norm mass | Top 5% norm mass | Top-5 intersection | '
                      'Top norm border | Border share | Max image / text median | Max image / image median |',
                      '|---|---|---|---|---|---|---|---|---|---|']
            for view, s in r['by_view'].items():
                vals = [s['norm_attention_spearman'], s['maxabs_attention_spearman'],
                        s['highest_norm_mass']['0.01'], s['highest_norm_mass']['0.05'],
                        s['top5_intersection'], s['norm_top5_border_share'], s['border_token_share'],
                        s['image_norm_max_over_text_median'], s['image_norm_max_over_image_median']]
                lines.append(f'| {view} | ' + ' | '.join(map(format_stats, vals)) + ' |')
            lines += ['', 'Per-head cells: Moran I / top-5% mass, each median [p10, p90].', '',
                      '| Head | ' + ' | '.join(C.VIEWS) + ' |', '|---:|---|---|---|---|']
            for h in range(16):
                cells = []
                for view in C.VIEWS:
                    metrics = r['by_view'][view]['heads'][str(h)]
                    cells.append(f"{format_stats(metrics['moran'])} / {format_stats(metrics['top_5pct_mass'])}")
                lines.append(f'| {h} | ' + ' | '.join(cells) + ' |')
        elif part == 'objects':
            lines += [r['definition'], '', '| Object | View | Band | Object mass | Token share | Enrichment | Hottest r,c |',
                      '|---|---|---|---:|---:|---:|---|']
            for row in r['records']:
                lines.append(f"| {row['object']} | {row['view']} | {row['band']} | {row['object_mass']:.4f} | "
                             f"{row['object_token_share']:.4f} | {row['enrichment']:.3f} | {row['hottest_row_col']} |")
            lines += ['', 'Medians over the four positions; hottest object-cell rank among all cells '
                      'is 1 + the number of cells with strictly greater attention (1 = hottest).', '',
                      '| Object | View | Band | Median mass | Median token share | Median hottest object rank |',
                      '|---|---|---|---:|---:|---:|']
            groups = {}
            for row in r['records']:
                key = (row['object'].rsplit('_', 1)[0], row['view'], row['band'])
                groups.setdefault(key, []).append(row)
            for (kind, view, band), rows in groups.items():
                mass = float(np.median([row['object_mass'] for row in rows]))
                share = float(np.median([row['object_token_share'] for row in rows]))
                rank = (f"{np.median([row['object_hottest_rank'] for row in rows]):.1f}"
                        if all('object_hottest_rank' in row for row in rows) else 'unavailable; rerun objects')
                lines.append(f'| {kind} | {view} | {band} | {mass:.4f} | {share:.4f} | {rank} |')
        else:
            lines += [f"Paintings: {r['paintings']}", r['rendering']]
        paths = r.get('figures', [r['figure']] if 'figure' in r else [])
        lines += ['', *[f'Figure: [{Path(p).name}]({Path(p).name})' for p in paths], '']
    lines += ['Cleanup candidates (nothing deleted): generated PNGs, results.json, summary.md, run logs and '
              'the local Matplotlib cache under this output directory. No source or stored data is a removal candidate.', '']
    (out / 'summary.md').write_text('\n'.join(lines))


def json_safe(obj):
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (float, np.floating)) and not np.isfinite(obj):
        return None
    return obj


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    ap.add_argument('--n-paintings', type=int, default=16)
    ap.add_argument('--parts', default=','.join(PARTS))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--geometry-limit', type=int, choices=(1, 2, 3, 4), default=4,
                    help='CPU smoke only: fewer than 4 means PARTIAL, never a full geometry PASS')
    args = ap.parse_args()
    parts = {p.strip() for p in args.parts.split(',')}
    if not parts or parts - set(PARTS):
        ap.error(f'--parts must be a comma list from {PARTS}')
    if args.n_paintings < 1 or args.n_paintings > 5000 or args.seed < 0:
        ap.error('--n-paintings must be 1..5000 and --seed must be nonnegative')
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(args.out / '.mplconfig'))
    result_path = args.out / 'results.json'
    results = json.loads(result_path.read_text()) if result_path.exists() else {}
    runs = results.setdefault('runs', [])
    runs.append(dict(parts=sorted(parts), device=args.device, n_paintings=args.n_paintings,
                     seed=args.seed, geometry_limit=args.geometry_limit, started_utc=time.strftime('%FT%TZ', time.gmtime())))
    # Keep prior CPU results when the controller later runs only the GPU parts.
    images = model = None

    def save():
        result_path.write_text(json.dumps(json_safe(results), indent=2, allow_nan=False) + '\n')
        summary(results, args.out)

    def run_part(part, fn):
        print(f'start {part}', flush=True)
        try:
            results[part] = dict(fn(), device=args.device if part not in ('structure', 'sheet') else 'cpu', seed=args.seed)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            results[part] = dict(status='ERROR', error=f'{type(exc).__name__}: {exc}', device=args.device, seed=args.seed)
        save()
        print(f"{part}: {results[part]['status']}", flush=True)

    # CPU parts first: the pool never inherits model allocations.
    if 'structure' in parts:
        run_part('structure', lambda: structure(args))
    if parts - {'structure'}:
        images = Images()
    if 'sheet' in parts:
        run_part('sheet', lambda: sheet(images, args))
    if parts & {'geometry', 'recompute', 'sinks', 'objects'}:
        print(f'loading plain model on {args.device}', flush=True)
        model = PlainModel(args.device, images)
    if 'geometry' in parts:
        run_part('geometry', lambda: geometry(model, args))
    if parts & {'recompute', 'sinks'}:
        # Reuse each full forward when B and D ask for the same image/view.
        try:
            real = real_passes(model, images, args, 'recompute' in parts, 'sinks' in parts)
            for part, data in real.items():
                results[part] = dict(data, device=args.device, seed=args.seed)
                print(f"{part}: {data['status']}", flush=True)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            for part in parts & {'recompute', 'sinks'}:
                results[part] = dict(status='ERROR', error=f'{type(exc).__name__}: {exc}', device=args.device, seed=args.seed)
        save()
    if 'objects' in parts:
        run_part('objects', lambda: objects(model, args))
    save()
    print(f'wrote {result_path} and {args.out / "summary.md"}', flush=True)
    if any(results.get(p, {}).get('status') in ('FAIL', 'ERROR') for p in parts):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
