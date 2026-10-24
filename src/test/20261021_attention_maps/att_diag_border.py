"""Border check (option 1, small): are the border peaks of the image maps position or content, and do the view
differences need the border?

interior (CPU): drop the outer ring of image tokens, renormalise, and re-measure the L1 between each instruction's map
  and the default map, plus the part of the full L1 that sits on the ring. Raw stored maps on all 5,000 development
  paintings; raw and value-weighted maps on the 304 paintings of att_diag_value_weighted.
pad (GPU): each painting shrunk by one token per side (bicubic) inside a grey (128) ring of one token; same grid.
  If the peaks are position, the grey ring keeps them; if content, they move one ring in with the painting's edge.

Reading rules, fixed before the run (2026-10-24), layers 16-27, emotion view first:
- position if the padded grey ring holds the hottest token about as often as the original outer ring (within 0.10);
- content if that falls to the ring's token share or below and ring 1 (the painting's new edge) gains it instead;
- mixed otherwise;
- the view differences do not need the border if the interior-only L1 to the default keeps at least 80% of the
  full-map L1 for every instruction (medians).

  att_diag_border.py --part pad       (GPU)
  att_diag_border.py --part analyse   (CPU; writes outputs/border_check/)
No files are deleted.
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import att_diag_value_weighted as VW   # noqa: E402  (sets sys.path for the repo)
import att_diag_spatial as S           # noqa: E402
import att_core as C                   # noqa: E402

STORE = S.STORE
SCRATCH = Path('/tmp/claude-0/-project-CoSiR/e5b42f77-07da-4e87-9759-def96bd42c89/scratchpad/border_check')
VW_IMG = VW.SCRATCH / 'img'
OUT = HERE / 'outputs/border_check'
A, B = 16, 28                       # layers 16-27


def rings(grid):
    r, c = grid
    i, j = np.mgrid[:r, :c]
    return np.minimum(np.minimum(i, r - 1 - i), np.minimum(j, c - 1 - j)).ravel()


def norm(m):
    return m / m.sum()


def top_mass(m, frac=0.01):
    return VW.top_mass(m, frac)


def measures(maps, grid, rng):
    """maps: {view: image map (sums to 1)}. Per view: ring enrichment, ring of the hottest token, L1 parts."""
    rg = rings(grid)
    b0 = rg == 0
    res = {}
    for v, m in maps.items():
        res[v] = dict(enrich0=m[b0].sum() / b0.mean(), enrich1=m[rg == 1].sum() / (rg == 1).mean(),
                      hot0=float(rg[np.argmax(m)] == 0), hot1=float(rg[np.argmax(m)] == 1),
                      top1=top_mass(m), top1_interior=top_mass(norm(m[~b0])))
    for v in C.VIEWS[1:]:
        a, b = maps['default'], maps[v]
        d = np.abs(a - b)
        ai, bi = norm(a[~b0]), norm(b[~b0])
        res[v].update(l1=d.sum(), l1_ring0_part=d[b0].sum() / d.sum(), l1_interior=np.abs(ai - bi).sum(),
                      l1_interior_shuffled=np.abs(ai - rng.permutation(bi)).sum())
    return res, float(b0.mean()), float((rg == 1).mean())


def stored_maps(k):
    out = {}
    with np.load(C.painting_path(STORE, k)) as z:
        g = tuple(int(x) for x in z['grid_hw'])
        for v in C.VIEWS:
            off = z[f'{v}__img__off']
            a, b = int(off[0]), int(off[1])
            s = z[f'{v}__img__layers'][A:B, a:b].astype(np.float64).sum(0)
            out[v] = norm(s[z[f'{v}__img__ttype'][a:b] == 2])
    return k, g, out


def record_maps(path, kind):
    z = np.load(path)
    g = tuple(int(x) for x in z['grid'])
    return g, {v: norm(z[f'{v}__{kind}_bands'][0][z[f'{v}__tt'] == 2].astype(np.float64)) for v in C.VIEWS}


def padded(prepared):
    from PIL import Image
    W, H = prepared.size
    canvas = Image.new('RGB', (W, H), (128, 128, 128))
    canvas.paste(prepared.resize((W - 64, H - 64), Image.BICUBIC), (32, 32))
    return canvas


def pad(args):
    images = S.Images()
    model = VW.Model()
    out = SCRATCH / 'pad'
    out.mkdir(parents=True, exist_ok=True)
    ks = sorted(int(p.stem[1:]) for p in VW_IMG.glob('p*.npz'))
    ks = sorted(set(ks[:args.limit]) | set(VW.SHEET_KS)) if args.limit else ks
    t0 = time.time()
    for n, k in enumerate(ks):
        path = out / f'p{k:04d}.npz'
        if path.exists():
            continue
        pm = C.load_painting(STORE, k)['painting']
        grid = tuple(int(x) for x in pm['grid_hw'])
        prepared = images.painting(pm)
        im = padded(prepared)
        rec = {}
        for view in C.VIEWS:
            ids, tt, per = model.image_pass(im, view)
            assert int((tt == 2).sum()) == grid[0] * grid[1]
            for key, val in VW.band_summary(ids, tt, per).items():
                rec[f'{view}__{key}'] = val
        prepared.close()
        im.close()
        np.savez(path, grid=np.array(grid), **rec)
        if n % 50 == 0:
            print(f'pad {n + 1}/{len(ks)} k={k} {time.time() - t0:.0f}s', flush=True)
    print(f'pad done {len(ks)} paintings {time.time() - t0:.0f}s', flush=True)


def collect(items, rng):
    """items: iterable of (grid, maps). Returns {view: {measure: array}}, ring-0 and ring-1 token shares."""
    acc, sh0, sh1 = {}, [], []
    for grid, maps in items:
        res, s0, s1 = measures(maps, grid, rng)
        sh0.append(s0)
        sh1.append(s1)
        for v, d in res.items():
            for key, x in d.items():
                acc.setdefault(v, {}).setdefault(key, []).append(x)
    return {v: {key: np.array(x) for key, x in d.items()} for v, d in acc.items()}, np.array(sh0), np.array(sh1)


def summarise(acc, sh0, sh1):
    out = dict(n=int(len(sh0)), ring0_token_share=float(sh0.mean()), ring1_token_share=float(sh1.mean()))
    for v, d in acc.items():
        row = {}
        for key, x in d.items():
            row[key] = float(x.mean()) if key.startswith('hot') else S.stats(x)
        if 'l1' in d:
            row['l1_interior_over_full'] = S.stats(d['l1_interior'] / d['l1'])
        out[v] = row
    return out


def analyse(args):
    rng = np.random.default_rng(args.seed)
    R = {}
    with Pool(8) as p:
        rows = p.map(stored_maps, range(5000), chunksize=25)
    R['stored_raw_5000'] = summarise(*collect(((g, m) for _, g, m in rows), rng))
    padf = sorted((SCRATCH / 'pad').glob('p*.npz'))
    vw = [VW_IMG / p.name for p in padf]      # the same paintings, original input
    print(f'{len(padf)} padded paintings of {len(list(VW_IMG.glob("p*.npz")))} value-weighted ones', flush=True)
    for kind in ('raw', 'w'):
        R[f'original_{kind}_304'] = summarise(*collect((record_maps(p, kind) for p in vw), rng))
        R[f'padded_{kind}_304'] = summarise(*collect((record_maps(p, kind) for p in padf), rng))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'results.json').write_text(json.dumps(R, indent=1))
    write_md(R)
    sheet()


def f3(s):
    return f"{s['median']:.3f} [{s['p10']:.3f}, {s['p90']:.3f}]"


def write_md(R):
    L = ['# Border check', '', 'Layers 16-27, image only. Ring 0 = the outer ring of image tokens, ring 1 the next one. '
         'Enrichment = mass on the ring / the ring\'s token share (1 = no preference). Hottest = fraction of paintings '
         'whose hottest token is on that ring. Cells: median [10th, 90th percentile] over paintings.', '',
         f"## Grey-ring test ({R['original_raw_304']['n']} paintings): original against padded", '',
         f"Token share: ring 0 {R['original_raw_304']['ring0_token_share']:.3f}, ring 1 "
         f"{R['original_raw_304']['ring1_token_share']:.3f} (mean over paintings).", '',
         '| Map | View | Hottest on ring 0: original | padded (grey) | Hottest on ring 1: original | padded | '
         'Ring-0 enrichment: original | padded | Ring-1 enrichment: original | padded |',
         '|---|---|---|---|---|---|---|---|---|---|']
    for kind in ('raw', 'w'):
        o, p = R[f'original_{kind}_304'], R[f'padded_{kind}_304']
        for v in C.VIEWS:
            L.append(f"| {kind} | {v} | {o[v]['hot0']:.2f} | {p[v]['hot0']:.2f} | {o[v]['hot1']:.2f} | "
                     f"{p[v]['hot1']:.2f} | {f3(o[v]['enrich0'])} | {f3(p[v]['enrich0'])} | {f3(o[v]['enrich1'])} | "
                     f"{f3(p[v]['enrich1'])} |")
    L += ['', '## Interior-only maps: L1 to the default map', '',
          '| Maps | View | Full L1 | Interior L1 | Interior / full | Interior, shuffled | Part of full L1 on ring 0 | '
          'Top-1% mass | interior |', '|---|---|---|---|---|---|---|---|---|']
    for name, key in (('stored raw, 5,000', 'stored_raw_5000'), ('raw, 304', 'original_raw_304'),
                      ('weighted, 304', 'original_w_304'), ('padded raw, 304', 'padded_raw_304'),
                      ('padded weighted, 304', 'padded_w_304')):
        d = R[key]
        for v in C.VIEWS[1:]:
            L.append(f"| {name} | {v} | {f3(d[v]['l1'])} | {f3(d[v]['l1_interior'])} | "
                     f"{f3(d[v]['l1_interior_over_full'])} | {f3(d[v]['l1_interior_shuffled'])} | "
                     f"{f3(d[v]['l1_ring0_part'])} | {f3(d[v]['top1'])} | {f3(d[v]['top1_interior'])} |")
        L.append(f"| {name} | default | | | | | | {f3(d['default']['top1'])} | {f3(d['default']['top1_interior'])} |")
    (OUT / 'summary.md').write_text('\n'.join(L) + '\n')


def sheet():
    import io
    from PIL import Image
    plt = S.plot_module()
    images = S.Images()
    ks = VW.SHEET_KS
    fig, axes = plt.subplots(len(ks), 9, figsize=(27, 3.2 * len(ks)))
    for row, k in enumerate(ks):
        pm = C.load_painting(STORE, k)['painting']
        prepared = images.painting(pm)
        shown = {}
        for name, im in (('original', prepared), ('padded', padded(prepared))):
            buf = io.BytesIO()
            im.save(buf, 'JPEG', quality=90)
            im.close()
            buf.seek(0)
            shown[name] = Image.open(buf).convert('RGB')
        g, orig = record_maps(VW_IMG / f'p{k:04d}.npz', 'w')
        _, pad_ = record_maps(SCRATCH / 'pad' / f'p{k:04d}.npz', 'w')
        axes[row, 0].imshow(shown['padded'])
        axes[row, 0].axis('off')
        axes[row, 0].set_title(f'k={k}, padded input')
        for j, v in enumerate(C.VIEWS):
            for i, (name, m) in enumerate((('original', orig[v]), ('padded', pad_[v]))):
                ax = axes[row, 1 + 2 * j + i]
                S.overlay(ax, shown[name], m.reshape(g))
                ax.set_title(f'{v} {name}', fontsize=9)
    fig.suptitle('Layers 16-27, value-weighted maps: original input against the input shrunk inside a grey ring')
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(OUT / 'sheet_original_vs_padded.jpg', dpi=110)
    plt.close(fig)


def main():
    global SCRATCH, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', choices=('pad', 'analyse'), required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--limit', type=int, default=0, help='smoke: first N paintings plus the sheet paintings')
    ap.add_argument('--scratch', default=str(SCRATCH))
    ap.add_argument('--out', default=str(OUT))
    args = ap.parse_args()
    SCRATCH, OUT = Path(args.scratch), Path(args.out)
    pad(args) if args.part == 'pad' else analyse(args)


if __name__ == '__main__':
    main()
