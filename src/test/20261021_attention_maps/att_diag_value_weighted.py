"""Value-weighted attention: is the speckle, or any headline result, an effect of weight that heads park on the sink?

Raw map (as stored): the pooled token's head-mean attention a_hj, summed over a band, renormalised over the image.
Weighted map (Kobayashi et al. 2020): ||sum_h a_hj W_O^h v_hj||, the size of what token j adds to the pooled
token's attention output in one layer; summed over the band and renormalised the same way. A second variant,
sum_h a_hj ||W_O^h v_hj||, lets no head cancel another. Weight on a token with a small value vector counts little.

Reading rules, fixed before the run (2026-10-22):
- the first token is a no-op sink if its weighted share is far below its raw share (0.62 to 0.67);
- weighting makes the maps less speckled if the median Moran's I rises by at least 0.05 and in >= 75% of paintings;
- a headline result holds if its direction holds under weighting: default image share above each instructed one;
  instructed-vs-default L1 above the rounding floor (0.09) and below shuffled; the emotion view's top caption words
  still emotion words.

  att_diag_value_weighted.py --part gpu      (image passes on --n-img paintings, caption passes on all captions)
  att_diag_value_weighted.py --part analyse  (CPU; writes outputs/value_weighted/)
No files are deleted.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import att_diag_spatial as S   # noqa: E402  (sets sys.path for the repo)
import att_core as C           # noqa: E402

STORE = S.STORE
SCRATCH = Path('/tmp/claude-0/-project-CoSiR/e5b42f77-07da-4e87-9759-def96bd42c89/scratchpad/value_weighted')
OUT = HERE / 'outputs/value_weighted'
KINDS = ('raw', 'w', 'w2')            # raw head-mean attention; ||sum_h a f_h||; sum_h a ||f_h||
CATS = ('first', 'instruction', 'image', 'caption', 'other')   # first = token 0, <|im_start|>
SHEET_KS = (0, 1250, 2500, 3750)
MIN_OCC = 50                          # as att_analysis m5


def categories(tt):
    c = np.full(len(tt), 4, np.int8)
    c[tt == 1], c[tt == 2], c[tt == 3] = 1, 2, 3
    c[0] = 0
    return c


class Model:
    def __init__(self):
        import torch
        from pav_prompts import PROMPTS
        from src.data.feature_extract import Qwen3VLEmb
        self.torch, self.prompts = torch, PROMPTS
        self.enc = Qwen3VLEmb(device='cuda')
        self.enc.m.set_attn_implementation('eager')
        self.enc.m.eval()
        self.tok = self.enc.p.tokenizer
        lm = self.enc.m.language_model
        cfg = lm.config
        self.H, self.KV, self.D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        assert len(lm.layers) == C.N_LAYERS
        self.v = [None] * C.N_LAYERS
        for i, layer in enumerate(lm.layers):
            layer.self_attn.v_proj.register_forward_hook(lambda mod, inp, out, i=i: self.v.__setitem__(i, out))
        # o_proj input is the heads' outputs concatenated head-major (h * D + d), as eager_attention_forward builds it
        self.Wo = [layer.self_attn.o_proj.weight.detach().float().view(-1, self.H, self.D) for layer in lm.layers]

    def layer_terms(self, l, att, last):
        """Pooled-row terms of layer l. att (B, H, L, L); last (B,) query index. Returns four (B, L) tensors."""
        torch = self.torch
        B, L = att.shape[0], att.shape[-1]
        ar = torch.arange(B, device=att.device)
        a = att[ar, :, last, :].float()                                      # (B, H, L)
        # repeat_kv: head h reads key-value head h // (H / KV)
        v = self.v[l].float().view(B, L, self.KV, self.D).repeat_interleave(self.H // self.KV, dim=2)
        f = torch.einsum('blhd,ehd->blhe', v, self.Wo[l])                    # W_O^h v_j, (B, L, H, E)
        fn = f.norm(dim=-1)                                                  # (B, L, H)
        w = torch.einsum('bhl,blhe->ble', a, f).norm(dim=-1)
        w2 = (a.transpose(1, 2) * fn).sum(-1)
        return a.mean(1), w, w2, fn.mean(-1)

    def image_pass(self, prepared, view):
        torch, enc = self.torch, self.enc
        enc.instruction = self.prompts[view]
        o = enc.p(text=[enc._prompt([{'type': 'image'}])], images=[prepared], do_resize=False, return_tensors='pt')
        ids = o['input_ids'][0].numpy()
        tt = C.token_types(ids, 'img')
        assert ids[-1] == C.ENDOFTEXT and ids[0] == C.IM_START
        kw = {k: v.to('cuda', torch.bfloat16) if v.is_floating_point() else v.to('cuda') for k, v in o.items()}
        with torch.inference_mode():
            out = enc.m(**kw, use_cache=False, output_attentions=True, output_hidden_states=True)
            last = torch.tensor([len(ids) - 1], device='cuda')
            terms = [self.layer_terms(l, out.attentions[l], last) for l in range(C.N_LAYERS)]
            per = {name: torch.stack([t[i][0] for t in terms]).cpu().numpy() for i, name in
                   enumerate(('raw', 'w', 'w2', 'fm'))}                       # (28, L)
            # outputs of layers 0..26 (hidden_states[28] carries the final RMSNorm)
            hs = out.hidden_states[1:C.N_LAYERS]
            per['maxabs'] = torch.stack([h[0].float().abs().amax(-1) for h in hs]).cpu().numpy()   # (27, L)
            per['norm'] = torch.stack([h[0].float().norm(dim=-1) for h in hs]).cpu().numpy()
        return ids, tt, per

    def caption_batch(self, texts, view):
        torch, enc = self.torch, self.enc
        enc.instruction = self.prompts[view]
        t = self.tok([enc._prompt([{'type': 'text', 'text': x}]) for x in texts], return_tensors='pt',
                     padding=True, truncation=True, max_length=512)
        mask = t['attention_mask']
        lens = mask.sum(1)
        assert all(bool(mask[i, :int(n)].all()) for i, n in enumerate(lens)), 'expected right padding'
        with torch.inference_mode():
            out = enc.m(input_ids=t['input_ids'].to('cuda'), attention_mask=mask.to('cuda'), output_attentions=True)
            last = (lens - 1).to('cuda')
            terms = [self.layer_terms(l, out.attentions[l], last) for l in range(C.N_LAYERS)]
            per = {name: torch.stack([tm[i] for tm in terms], 1).cpu().numpy() for i, name in
                   enumerate(('raw', 'w', 'w2', 'fm'))}                       # (B, 28, Lmax)
        res = []
        for i, n in enumerate(lens.tolist()):
            ids = t['input_ids'][i, :n].numpy()
            res.append((ids, {k: v[i, :, :n] for k, v in per.items()}))
        return res


def band_summary(ids, tt, per):
    """Compact record of one pass: band sums of each kind, category sums per layer, first-token value size."""
    cat = categories(tt)
    rec = dict(ids=ids.astype(np.int32), tt=tt)
    for kind in KINDS:
        x = per[kind].astype(np.float64)
        rec[f'{kind}_bands'] = np.stack([x[a:b].sum(0) for _, a, b in C.BANDS]).astype(np.float32)   # (3, L)
        rec[f'{kind}_cats'] = np.stack([x[:, cat == c].sum(1) for c in range(len(CATS))], 1).astype(np.float32)
    img = tt == 2 if (tt == 2).any() else tt == 3
    rec['fm_first'] = per['fm'][:, 0].astype(np.float32)
    rec['fm_rest_median'] = np.median(per['fm'][:, img], 1).astype(np.float32)
    rec['fm_rest_name'] = 'image' if (tt == 2).any() else 'caption'
    if 'maxabs' in per:
        rec['maxabs_first'] = per['maxabs'][:, 0]
        rec['maxabs_median'] = np.median(per['maxabs'][:, 1:], 1)
        rec['norm_first'] = per['norm'][:, 0]
        rec['norm_median'] = np.median(per['norm'][:, 1:], 1)
    return rec


def caption_rows(images):
    s = images.sample
    order = np.argsort(s['pid'], kind='stable')
    starts = np.searchsorted(s['pid'][order], s['subset_pid'], 'left')
    ends = np.searchsorted(s['pid'][order], s['subset_pid'], 'right')
    rows = [order[a:b] for a, b in zip(starts, ends)]
    assert sum(len(r) for r in rows) == 25160
    return rows


def gpu(args):
    images = S.Images()
    model = Model()
    rng = np.random.default_rng(args.seed)
    ks = sorted(set(rng.choice(5000, args.n_img, replace=False).tolist()) | set(SHEET_KS))
    img_dir = SCRATCH / 'img'
    img_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for n, k in enumerate(ks):
        path = img_dir / f'p{k:04d}.npz'
        if path.exists():
            continue
        painting = C.load_painting(STORE, k)
        prepared = images.painting(painting['painting'])
        rec = {}
        for view in C.VIEWS:
            ids, tt, per = model.image_pass(prepared, view)
            stored = painting[view]['img'][0]
            assert np.array_equal(ids, stored['ids'])
            if n < 4:   # the recomputed raw map must match the stored one (bf16; earlier recompute max L1 0.068)
                mine = per['raw'][16:28][:, tt == 2].sum(0)
                l1 = float(np.abs(mine / mine.sum() - C.band_map(stored, C.BANDS[0])[0]).sum())
                assert l1 < 0.1, (k, view, l1)
            for key, val in band_summary(ids, tt, per).items():
                rec[f'{view}__{key}'] = val
        prepared.close()
        np.savez(path, grid=np.array(painting['painting']['grid_hw']), **rec)
        if n % 25 == 0:
            print(f'img {n + 1}/{len(ks)} k={k} {time.time() - t0:.0f}s', flush=True)
    print(f'img done {len(ks)} paintings {time.time() - t0:.0f}s', flush=True)
    if args.n_cap_paintings == 0:
        return
    rows = caption_rows(images)
    texts = [(k, images.spec.texts[int(r)]) for k in range(args.n_cap_paintings) for r in rows[k]]
    check = C.load_painting(STORE, 0)
    t0 = time.time()
    for view in C.VIEWS:
        path = SCRATCH / f'cap_{view}.npz'
        if path.exists():
            continue
        cnt = {kind: np.zeros(160000) for kind in KINDS}
        tot = {kind: np.zeros(160000) for kind in KINDS}
        cats, fm_first, fm_rest = {kind: [] for kind in KINDS}, [], []
        for b in range(0, len(texts), args.batch):
            chunk = texts[b:b + args.batch]
            for j, (ids, per) in enumerate(model.caption_batch([x for _, x in chunk], view)):
                tt = C.token_types(ids, 'txt')
                if b + j < len(check[view]['txt']) and chunk[j][0] == 0:
                    stored = check[view]['txt'][b + j]
                    assert np.array_equal(ids, stored['ids'])
                    mine = per['raw'][16:28][:, tt == 3].sum(0)
                    l1 = float(np.abs(mine / mine.sum() - C.band_map(stored, C.BANDS[0], 'caption')[0]).sum())
                    assert l1 < 0.1, (view, b + j, l1)
                rec = band_summary(ids, tt, per)
                cap = tt == 3
                for kind in KINDS:
                    m = rec[f'{kind}_bands'][0, cap].astype(np.float64)
                    m = m / m.sum()
                    np.add.at(cnt[kind], ids[cap], 1)
                    np.add.at(tot[kind], ids[cap], m)
                    cats[kind].append(rec[f'{kind}_cats'])
                fm_first.append(rec['fm_first'])
                fm_rest.append(rec['fm_rest_median'])
            if (b // args.batch) % 100 == 0:
                print(f'cap {view} {b}/{len(texts)} {time.time() - t0:.0f}s', flush=True)
        np.savez(path, count=cnt['raw'], **{f'sum_{k}': tot[k] for k in KINDS},
                 **{f'cats_{k}': np.stack(cats[k]) for k in KINDS},
                 fm_first=np.stack(fm_first), fm_rest=np.stack(fm_rest))
    print(f'cap done {len(texts)} captions x {len(C.VIEWS)} views {time.time() - t0:.0f}s', flush=True)


# ---------------------------------------------------------------- analysis (CPU)
def spearman(a, b):
    from scipy.stats import rankdata
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def top_mass(m, frac=0.05):
    n = max(1, int(np.ceil(len(m) * frac)))
    return float(np.sort(m)[-n:].sum())


def analyse(args):
    rng = np.random.default_rng(args.seed)
    files = sorted((SCRATCH / 'img').glob('p*.npz'))
    R = {}
    # per (band, view): lists of measures
    per = {}
    l1 = {}
    value_size, maxabs_ratio, norm_ratio = {v: [] for v in C.VIEWS}, {v: [] for v in C.VIEWS}, {v: [] for v in C.VIEWS}
    for f in files:
        z = np.load(f)
        grid = tuple(int(x) for x in z['grid'])
        brd = S.border(grid)
        maps = {}
        for v in C.VIEWS:
            tt = z[f'{v}__tt']
            img = tt == 2
            fr = z[f'{v}__fm_first'].astype(np.float64)
            rest = z[f'{v}__fm_rest_median'].astype(np.float64)
            value_size[v].append(fr / rest)                                       # (28,)
            maxabs_ratio[v].append(z[f'{v}__maxabs_first'] / z[f'{v}__maxabs_median'])
            norm_ratio[v].append(z[f'{v}__norm_first'] / z[f'{v}__norm_median'])
            for bi, (bn, a, b) in enumerate(C.BANDS):
                d = per.setdefault((bn, v), {})
                for kind in KINDS:
                    cats = z[f'{v}__{kind}_cats'][a:b].astype(np.float64).sum(0)
                    d.setdefault(f'share_first_{kind}', []).append(cats[0] / cats.sum())
                    d.setdefault(f'share_image_{kind}', []).append(cats[2] / cats.sum())
                    m = z[f'{v}__{kind}_bands'][bi][img].astype(np.float64)
                    m = m / m.sum()
                    maps[(bn, v, kind)] = m
                    d.setdefault(f'moran_{kind}', []).append(S.moran(m, grid))
                    d.setdefault(f'top5_{kind}', []).append(top_mass(m))
                    d.setdefault(f'border_hot_{kind}', []).append(float(brd[np.argmax(m)]))
                d.setdefault('moran_shuffled_w', []).append(S.moran(rng.permutation(maps[(bn, v, 'w')]), grid))
                d.setdefault('moran_gain_w', []).append(d['moran_w'][-1] - d['moran_raw'][-1])
                d.setdefault('spearman_raw_w', []).append(spearman(maps[(bn, v, 'raw')], maps[(bn, v, 'w')]))
        for bn, _, _ in C.BANDS:
            for v in C.VIEWS[1:]:
                for kind in KINDS:
                    a, b = maps[(bn, 'default', kind)], maps[(bn, v, kind)]
                    l1.setdefault((bn, v, kind), []).append(float(np.abs(a - b).sum()))
                b = maps[(bn, v, 'w')]
                l1.setdefault((bn, v, 'w_shuffled'), []).append(float(np.abs(maps[(bn, 'default', 'w')]
                                                                              - rng.permutation(b)).sum()))
    R['n_paintings_image'] = len(files)
    R['image'] = {f'{bn}|{v}': {k: S.stats(x) for k, x in d.items()} for (bn, v), d in per.items()}
    for (bn, v), d in per.items():
        R['image'][f'{bn}|{v}']['frac_moran_w_above_raw'] = float(np.mean(np.array(d['moran_gain_w']) > 0))
        R['image'][f'{bn}|{v}']['frac_moran_w2_above_raw'] = float(np.mean(np.array(d['moran_w2'])
                                                                           > np.array(d['moran_raw'])))
        for kind in KINDS:
            R['image'][f'{bn}|{v}'][f'border_hot_{kind}_mean'] = float(np.mean(d[f'border_hot_{kind}']))
    R['l1_vs_default'] = {f'{bn}|{v}|{kind}': S.stats(x) for (bn, v, kind), x in l1.items()}
    R['first_token'] = {v: dict(
        value_size_ratio_by_layer=np.median(np.stack(value_size[v]), 0).round(4).tolist(),
        maxabs_ratio_by_hidden_layer=np.median(np.stack(maxabs_ratio[v]), 0).round(2).tolist(),
        norm_ratio_by_hidden_layer=np.median(np.stack(norm_ratio[v]), 0).round(2).tolist()) for v in C.VIEWS}
    R['first_token_note'] = ('value_size_ratio: mean over heads of ||W_O^h v|| of token 0 / median over image '
                             'tokens, per attention layer 0..27; maxabs/norm ratio: token 0 / median over the other '
                             'tokens, hidden outputs of layers 0..26. Medians over paintings.')
    # captions
    caps = sorted(SCRATCH.glob('cap_*.npz'))
    if caps:
        from transformers import AutoTokenizer
        from src.data.feature_extract import QWEN_REPO
        tok = AutoTokenizer.from_pretrained(QWEN_REPO)
        Z = {v: np.load(SCRATCH / f'cap_{v}.npz') for v in C.VIEWS}
        R['captions'] = {'n_captions': int(len(Z['default']['cats_raw']))}
        mean = {(v, k): Z[v][f'sum_{k}'] / np.maximum(Z[v]['count'], 1) for v in C.VIEWS for k in KINDS}
        for v in C.VIEWS:
            cnt = Z[v]['count']
            ok = np.flatnonzero(cnt >= MIN_OCC)
            row = {'n_eligible_tokens': int(len(ok))}
            for kind in KINDS:
                top = ok[np.argsort(-mean[(v, kind)][ok])[:12]]
                row[f'top_{kind}'] = [(tok.decode([int(i)]).strip(), round(float(mean[(v, kind)][i]), 3)) for i in top]
                if v != 'default':
                    okr = ok[Z['default']['count'][ok] >= MIN_OCC]
                    rise = mean[(v, kind)][okr] - mean[('default', kind)][okr]
                    row[f'rise_{kind}'] = [(tok.decode([int(i)]).strip(), round(float(r), 3))
                                           for i, r in zip(okr[np.argsort(-rise)[:12]], np.sort(rise)[::-1][:12])]
            row['spearman_token_means_raw_w'] = spearman(mean[(v, 'raw')][ok], mean[(v, 'w')][ok])
            top20 = {kind: set(ok[np.argsort(-mean[(v, kind)][ok])[:20]].tolist()) for kind in KINDS}
            row['top20_overlap_raw_w'] = len(top20['raw'] & top20['w']) / 20
            for kind in KINDS:
                c = Z[v][f'cats_{kind}'][:, 16:28].astype(np.float64).sum(1)      # (n, 5)
                sh = c / c.sum(1, keepdims=True)
                row[f'share_first_{kind}'] = S.stats(sh[:, 0])
                row[f'share_caption_{kind}'] = S.stats(sh[:, 3])
            row['value_size_ratio_first_vs_caption_by_layer'] = np.median(
                Z[v]['fm_first'] / Z[v]['fm_rest'], 0).round(4).tolist()
            R['captions'][v] = row
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'results.json').write_text(json.dumps(R, indent=1, default=float))
    write_md(R)
    sheet()


def f3(s):
    return f"{s['median']:.3f} [{s['p10']:.3f}, {s['p90']:.3f}]"


def write_md(R):
    L = ['# Value-weighted attention against raw attention', '',
         f"Image only, {R['n_paintings_image']} development paintings (random, seed 0, plus the sheet paintings). "
         'Cells: median [10th, 90th percentile] over paintings. raw = head-mean attention (the stored maps); '
         'w = ||sum_h a_h W_O^h v_h|| (Kobayashi et al. 2020); w2 = sum_h a_h ||W_O^h v_h||.', '']
    for bn, _, _ in C.BANDS:
        L += [f'## {bn}', '',
              '| View | First-token share raw | w | Image share raw | w | Moran raw | w | w2 | shuffled w | '
              'Moran w > raw | Spearman raw-w | Top-5% mass raw | w | Border-hottest raw | w |',
              '|---|' + '---|' * 14]
        for v in C.VIEWS:
            d = R['image'][f'{bn}|{v}']
            L.append(f"| {v} | {f3(d['share_first_raw'])} | {f3(d['share_first_w'])} | {f3(d['share_image_raw'])} | "
                     f"{f3(d['share_image_w'])} | {f3(d['moran_raw'])} | {f3(d['moran_w'])} | {f3(d['moran_w2'])} | "
                     f"{f3(d['moran_shuffled_w'])} | {d['frac_moran_w_above_raw']:.2f} | {f3(d['spearman_raw_w'])} | "
                     f"{f3(d['top5_raw'])} | {f3(d['top5_w'])} | {d['border_hot_raw_mean']:.2f} | "
                     f"{d['border_hot_w_mean']:.2f} |")
        L += ['', 'L1 to the default map (rounding floor 0.09 in the main report):', '',
              '| View | raw | w | w2 | w, shuffled |', '|---|---|---|---|---|']
        for v in C.VIEWS[1:]:
            g = R['l1_vs_default']
            L.append(f"| {v} | {f3(g[f'{bn}|{v}|raw'])} | {f3(g[f'{bn}|{v}|w'])} | {f3(g[f'{bn}|{v}|w2'])} | "
                     f"{f3(g[f'{bn}|{v}|w_shuffled'])} |")
        L.append('')
    L += ['## First token (<|im_start|>)', '', R['first_token_note'], '']
    for v in C.VIEWS:
        ft = R['first_token'][v]
        vs, mx, nr = (ft[f'{x}_by_{y}'] for x, y in (('value_size_ratio', 'layer'), ('maxabs_ratio', 'hidden_layer'),
                                                     ('norm_ratio', 'hidden_layer')))
        L.append(f"- {v}: value size ratio, layers 0/4/8/16/20/24/27: "
                 f"{', '.join(f'{vs[i]:.3f}' for i in (0, 4, 8, 16, 20, 24, 27))}; median over layers 16-27 "
                 f"{np.median(vs[16:]):.3f}. Max-abs ratio, hidden 0/2/4/8/16/26: "
                 f"{', '.join(f'{mx[i]:.1f}' for i in (0, 2, 4, 8, 16, 26))}. Norm ratio, same layers: "
                 f"{', '.join(f'{nr[i]:.1f}' for i in (0, 2, 4, 8, 16, 26))}.")
    if 'captions' in R:
        c = R['captions']
        L += ['', f"## Captions only ({c['n_captions']} captions, layers 16-27, tokens with >= {MIN_OCC} occurrences)", '']
        for v in C.VIEWS:
            r = c[v]
            L.append(f"### {v}")
            L.append(f"- first-token share raw {f3(r['share_first_raw'])}, w {f3(r['share_first_w'])}; caption share "
                     f"raw {f3(r['share_caption_raw'])}, w {f3(r['share_caption_w'])}")
            L.append(f"- Spearman of token means raw vs w: {r['spearman_token_means_raw_w']:.3f}; top-20 overlap "
                     f"{r['top20_overlap_raw_w']:.2f}; eligible tokens {r['n_eligible_tokens']}")
            for kind in ('raw', 'w'):
                L.append(f"- top {kind}: " + ', '.join(f'{t} {x}' for t, x in r[f'top_{kind}']))
                if f'rise_{kind}' in r:
                    L.append(f"- rise over default, {kind}: " + ', '.join(f'{t} {x:+}' for t, x in r[f'rise_{kind}']))
            L.append('')
    (OUT / 'summary.md').write_text('\n'.join(L) + '\n')


def sheet():
    import io
    from PIL import Image
    plt = S.plot_module()
    images = S.Images()
    fig, axes = plt.subplots(len(SHEET_KS), 9, figsize=(27, 3.2 * len(SHEET_KS)))
    for row, k in enumerate(SHEET_KS):
        z = np.load(SCRATCH / 'img' / f'p{k:04d}.npz')
        grid = tuple(int(x) for x in z['grid'])
        painting = C.load_painting(STORE, k)
        prepared = images.painting(painting['painting'])
        buf = io.BytesIO()
        prepared.save(buf, 'JPEG', quality=90)
        prepared.close()
        buf.seek(0)
        image = Image.open(buf).convert('RGB')
        axes[row, 0].imshow(image)
        axes[row, 0].axis('off')
        axes[row, 0].set_title(f'k={k}')
        for j, v in enumerate(C.VIEWS):
            img = z[f'{v}__tt'] == 2
            for i, kind in enumerate(('raw', 'w')):
                m = z[f'{v}__{kind}_bands'][0][img].astype(np.float64)
                ax = axes[row, 1 + 2 * j + i]
                S.overlay(ax, image, (m / m.sum()).reshape(grid))
                ax.set_title(f"{v} {'raw' if kind == 'raw' else 'weighted'}", fontsize=9)
        image.close()
    fig.suptitle('Layers 16-27, image only: raw attention against value-weighted attention (value / map max)')
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(OUT / 'sheet_raw_vs_weighted.jpg', dpi=110)
    plt.close(fig)


def main():
    global SCRATCH, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', choices=('gpu', 'analyse'), required=True)
    ap.add_argument('--n-img', type=int, default=300)
    ap.add_argument('--n-cap-paintings', type=int, default=5000)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--scratch', default=str(SCRATCH))
    ap.add_argument('--out', default=str(OUT))
    args = ap.parse_args()
    SCRATCH, OUT = Path(args.scratch), Path(args.out)
    gpu(args) if args.part == 'gpu' else analyse(args)


if __name__ == '__main__':
    main()
