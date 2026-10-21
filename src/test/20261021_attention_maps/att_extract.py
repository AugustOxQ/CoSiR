"""Record Qwen3-VL-Embedding-2B attention maps (raw per-layer rows of the pooled token, its rollout as a record, raw cross maps).

Four instructions x (image only, caption only, image+caption) for the 5,000 ArtELingo paintings of the frozen
prompted-aspect-views sample. One file per painting (see att_core.py for the layout).

Shared-start shortcut: attention is causal and the ids up to <|vision_end|> are the same for the image-only input and
every image+caption input of one painting and view. The prefix runs once (use_cache, output_attentions); the image-only
suffix and all caption suffixes of the painting run as one batch over copies of the prefix cache. Rollout rows of the
suffix positions are built from the prefix rollout blocks (att_core.suffix_rollout); only the pooled row is kept.
Cross maps are raw head-mean attention of each caption-token row over the image columns, summed over each band of
att_core.BANDS (float16, (3, n_caption_tokens, n_image_tokens)).
Caption-only inputs are plain batched passes. Gate a (stored embeddings) and gate b (shortcut against plain passes)
raise on failure; the first G = max(32, check_full) paintings are held back until the gates pass.
"""
import argparse
import copy
import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
import numpy as np

REPO = Path('/project/CoSiR')
PAV = REPO / 'src/test/20261017_prompted_aspect_views'
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(PAV))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pav_sample  # noqa: E402
pav_sample.SAMPLE = PAV / 'sample.npz'
import att_core as C  # noqa: E402

DEFAULT_OUT = '/data/SSD2/pre_extract/artelingo_full/qwen3vl_emb_2b_attention/'
GATE_A_MEAN, GATE_A_MIN = 0.998, 0.99
GATE_B_COS_MIN = 0.995
# Raw maps in bf16: shortcut and plain pass both differ from a float32 reference by the same L1 (att_diag_raw_fp32.py,
# 2026-10-21: image 0.093 mean, caption-word rows 0.085), so the gap is rounding. Limits are about 1.5x the measured
# shortcut-vs-plain maxima (image 0.074, cross 0.102); a wrong cache or position gives L1 near 1 or more.
GATE_B_IMG_RAW, GATE_B_CROSS = 0.10, 0.20
STORED = Path('/data/SSD2/pre_extract/artelingo_full/qwen3vl_emb_2b_aspect_prompts')


def git_head():
    try:
        return subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    except Exception:
        return ''


class Extractor:
    def __init__(self, device, views):
        import torch
        from pav_prompts import PROMPTS
        from scripts.extract_features import adapter_artelingo_full
        from src.data.feature_extract import Qwen3VLEmb, _qwen_prepare_image
        self.torch, self.PROMPTS, self.prepare_image = torch, PROMPTS, _qwen_prepare_image
        self.views = list(views)
        self.s = pav_sample.load_frozen()
        s = self.s
        self.spec = adapter_artelingo_full(sample_ids=s['sample_ids'], expected=None)
        assert np.array_equal([m['sample_id'] for m in self.spec.text_meta], s['sample_ids'])
        assert np.array_equal([self.spec.images[int(i)]['painting'] for i in s['subset_image_indices']],
                              s['subset_painting_ids'])
        order = np.argsort(s['pid'], kind='stable')           # ascending row within each painting
        starts = np.searchsorted(s['pid'][order], s['subset_pid'], 'left')
        ends = np.searchsorted(s['pid'][order], s['subset_pid'], 'right')
        self.cap_rows = [order[a:b] for a, b in zip(starts, ends)]
        assert sum(len(r) for r in self.cap_rows) == 25160 and all(3 <= len(r) <= 12 for r in self.cap_rows)
        self.device = device
        self.enc = Qwen3VLEmb(device=device)
        self.enc.m.set_attn_implementation('eager')
        self.tok = self.enc.p.tokenizer
        ids = self.tok.convert_tokens_to_ids(['<|im_start|>', '<|im_end|>', '<|endoftext|>', '<|vision_start|>',
                                              '<|vision_end|>', '<|image_pad|>'])
        assert ids == [C.IM_START, C.IM_END, C.ENDOFTEXT, C.VISION_START, C.VISION_END, C.IMAGE_PAD], ids
        self.pad_id = self.tok.pad_token_id
        self.stored = {}
        self.dtype = torch.bfloat16     # att_diag_fp32 sets float32 to test the shortcut without bf16 noise

    # ------------------------------------------------------------ CPU side (worker thread)
    def prep(self, k, need_full):
        """Everything that needs no GPU for painting k: image, processor outputs, tokenised suffixes."""
        torch, s, enc, tok = self.torch, self.s, self.enc, self.tok
        rows = self.cap_rows[k]
        img_index = int(s['subset_image_indices'][k])
        im = self.spec.load_image(img_index)
        size = tuple(int(x) for x in im.size)
        pim = self.prepare_image(im, enc.max_pixels)
        try:
            im.close()
        except Exception:
            pass
        texts = [self.spec.texts[int(r)] for r in rows]
        d = dict(k=k, rows=rows, size=size, img_index=img_index, views={})
        for v in self.views:
            enc.instruction = self.PROMPTS[v]
            o = enc.p(text=[enc._prompt([{"type": "image"}])], images=[pim], do_resize=False, return_tensors='pt')
            ids_img = o['input_ids'][0]
            mm = o['mm_token_type_ids'][0]
            P = int((ids_img == C.VISION_END).nonzero()[0, 0]) + 1
            assert int((ids_img == C.VISION_END).sum()) == 1 and ids_img[P - 1] == C.VISION_END
            n_pad = int((ids_img == C.IMAGE_PAD).sum())
            ids_img = ids_img.numpy()
            joint_full, joint_proc = [], []
            for t in texts:
                sj = enc._prompt([{"type": "image"}, {"type": "text", "text": t}])
                assert sj.count('<|image_pad|>') == 1
                full = np.asarray(tok(sj.replace('<|image_pad|>', '<|image_pad|>' * n_pad))['input_ids'])
                # shared-start assertion: the ids up to and including <|vision_end|> are those of the image-only input
                assert np.array_equal(full[:P], ids_img[:P]), 'prefix ids differ between image-only and image+text'
                joint_full.append(full)
                if need_full:
                    o2 = enc.p(text=[sj], images=[pim], do_resize=False, return_tensors='pt')
                    assert np.array_equal(o2['input_ids'][0].numpy(), full), 'processor ids differ from tokenizer ids'
                    joint_proc.append(full)
            tt = tok([enc._prompt([{"type": "text", "text": t}]) for t in texts], return_tensors='pt',
                     padding=True, truncation=True, max_length=512)
            d['views'][v] = dict(ids_img=ids_img, mm=mm.numpy(), P=P, pix=o['pixel_values'], thw=o['image_grid_thw'],
                                 joint=joint_full, txt_ids=tt['input_ids'], txt_mask=tt['attention_mask'])
        return d

    # ------------------------------------------------------------ GPU side
    def pool(self, h):
        return self.torch.nn.functional.normalize(h.float(), dim=-1)

    def prefix_pass(self, vd):
        torch = self.torch
        P = vd['P']
        ids = torch.from_numpy(vd['ids_img'][:P]).long()[None].to(self.device)
        mm = torch.from_numpy(vd['mm'][:P]).long()[None].to(self.device)
        with torch.no_grad():
            out = self.enc.m(input_ids=ids, attention_mask=torch.ones_like(ids),
                             pixel_values=vd['pix'].to(self.device, self.dtype),
                             image_grid_thw=vd['thw'].to(self.device), mm_token_type_ids=mm,
                             use_cache=True, output_attentions=True)
            att = [C.head_mean_norm(a) for a in out.attentions]
            RP = C.rollout_full(att, keep_all=True)
            RP = [r[0] for r in RP]
        assert len(att) == C.N_LAYERS
        return out.past_key_values, RP

    def suffix_batch(self, vd, cache, RP, suffixes):
        """suffixes: list of int arrays (img suffix first, then caption suffixes). Returns per-item dicts (no ttype)."""
        torch, dev = self.torch, self.device
        P = vd['P']
        B = len(suffixes)
        lens = [len(x) for x in suffixes]
        Ls = max(lens)
        ids = torch.full((B, Ls), self.pad_id, dtype=torch.long)
        mask = torch.zeros(B, P + Ls, dtype=torch.long)
        mask[:, :P] = 1
        pos = torch.ones(3, B, Ls, dtype=torch.long)
        prefix = vd['ids_img'][:P]
        for i, sfx in enumerate(suffixes):
            ids[i, :lens[i]] = torch.from_numpy(np.asarray(sfx))
            mask[i, P:P + lens[i]] = 1
            full = torch.from_numpy(np.concatenate([prefix, sfx])).long()[None].to(dev)
            mmf = torch.zeros_like(full)
            mmf[0, :P] = torch.from_numpy(vd['mm'][:P]).long().to(dev)
            p3, _ = self.enc.m.get_rope_index(full, mmf, image_grid_thw=vd['thw'].to(dev),
                                              attention_mask=torch.ones_like(full))
            pos[:, i, :lens[i]] = p3[:, 0, P:].cpu()
        c = copy.deepcopy(cache)
        c.batch_repeat_interleave(B)
        with torch.no_grad():
            out = self.enc.m(input_ids=ids.to(dev), attention_mask=mask.to(dev), position_ids=pos.to(dev),
                             past_key_values=c, use_cache=True, output_attentions=True)
            att = [C.head_mean_norm(a) for a in out.attentions]       # (B, Ls, P+Ls)
            R = C.suffix_rollout(att, RP, P)
            last = torch.tensor(lens, device=dev) - 1
            ar = torch.arange(B, device=dev)
            emb = self.pool(out.last_hidden_state[ar, last])
            means = [a_raw.float().mean(1) for a_raw in out.attentions]    # raw head mean, (B, Ls, P+Ls)
            raw_last = torch.stack([m[ar, last] for m in means])
            roll_last = R[ar, last]                                    # (B, P+Ls)
            img_pos = torch.from_numpy(np.nonzero(prefix == C.IMAGE_PAD)[0]).to(dev)
            cross = self.band_cross(means, img_pos, 0, Ls) if B > 1 else None
        res = []
        for i, sfx in enumerate(suffixes):
            L = P + lens[i]
            p = dict(roll=roll_last[i, :L].cpu().numpy(),
                     layers=raw_last[:, i, :L].cpu().numpy().astype(np.float16),
                     emb=emb[i].cpu().numpy(), norm_rowsum_dev=float((raw_last[:, i, :L].sum(-1) - 1).abs().max()))
            if i > 0:
                p['cross_rows'] = cross[:, i, :lens[i]].cpu().numpy().astype(np.float16)  # (3, suffix rows, n_img)
            res.append(p)
        return res

    def band_cross(self, means, img_pos, r0, r1):
        """means: per-layer raw head-mean attention (B, Lq, Lk) tensors; rows r0:r1 over the image columns img_pos,
        summed over each band: (3, B, r1 - r0, n_img) float32."""
        sub = self.torch.stack([m[:, r0:r1][:, :, img_pos] for m in means])
        return self.torch.stack([sub[a:b].sum(0) for _, a, b in C.BANDS])

    def raw_head_mean(self, a, ar, last):
        """Raw (not renormalised) head-mean attention row of each item's last token: (B, P+Ls) float32."""
        return a.float().mean(1)[ar, last]

    def txt_pass(self, vd):
        torch, dev = self.torch, self.device
        ids, mask = vd['txt_ids'].to(dev), vd['txt_mask'].to(dev)
        with torch.no_grad():
            out = self.enc.m(input_ids=ids, attention_mask=mask, output_attentions=True)
            att = [C.head_mean_norm(a) for a in out.attentions]
            R = C.rollout_full(att)
            lens = vd['txt_mask'].sum(1)
            assert bool((vd['txt_mask'][torch.arange(len(lens)), lens - 1] == 1).all())
            last = (lens - 1).to(dev)
            ar = torch.arange(len(lens), device=dev)
            emb = self.pool(out.last_hidden_state[ar, last])
            roll_last = R[ar, last]
            raw_last = torch.stack([self.raw_head_mean(a, ar, last) for a in out.attentions])
        res = []
        for i in range(len(lens)):
            L = int(lens[i])
            res.append(dict(ids=vd['txt_ids'][i, :L].numpy().astype(np.int32), roll=roll_last[i, :L].cpu().numpy(),
                            layers=raw_last[:, i, :L].cpu().numpy().astype(np.float16), emb=emb[i].cpu().numpy()))
        return res

    def full_pass(self, ids, vd, want_rows):
        """Plain pass (batch 1, no cache) on a complete image-only or image+text input. Returns dict like suffix items,
        plus raw last-token rows per layer ('layers', 28 x L) and, when want_rows is a (start, end) range, the raw cross
        maps of those rows ('cross', (3, n_rows, n_img))."""
        torch, dev = self.torch, self.device
        ids_t = torch.from_numpy(np.asarray(ids)).long()[None].to(dev)
        mm = (ids_t == C.IMAGE_PAD).long()
        with torch.no_grad():
            out = self.enc.m(input_ids=ids_t, attention_mask=torch.ones_like(ids_t),
                             pixel_values=vd['pix'].to(dev, self.dtype), image_grid_thw=vd['thw'].to(dev),
                             mm_token_type_ids=mm, output_attentions=True)
            att = [C.head_mean_norm(a) for a in out.attentions]
            R = C.rollout_full(att)[0]
            means = [a.float().mean(1) for a in out.attentions]       # (1, L, L)
            p = dict(roll=R[-1].cpu().numpy(), emb=self.pool(out.last_hidden_state[0, -1]).cpu().numpy(),
                     layers=torch.stack([m[0, -1] for m in means]).cpu().numpy())
            if want_rows is not None:
                img_pos = (ids_t[0] == C.IMAGE_PAD).nonzero()[:, 0]
                p['cross'] = self.band_cross(means, img_pos, want_rows[0], want_rows[1])[:, 0].cpu().numpy()
        return p

    # ------------------------------------------------------------ one painting
    def process(self, d, stats, check_full):
        """Returns (painting dict, data {view: {modality: [pass]}}, gate-b records)."""
        torch, s, k = self.torch, self.s, d['k']
        rows = d['rows']
        data, gateb = {}, []
        grid = None
        for v in self.views:
            vd = d['views'][v]
            P = vd['P']
            thw = vd['thw'][0].tolist()
            g = (thw[1] // 2, thw[2] // 2)
            if grid is None:
                grid = g
            assert g == grid and g[0] * g[1] == int((vd['ids_img'] == C.IMAGE_PAD).sum())
            stats['t_wait'] += 0
            torch.cuda.synchronize(); t0 = time.time()
            cache, RP = self.prefix_pass(vd)
            torch.cuda.synchronize(); t1 = time.time()
            img_sfx = vd['ids_img'][P:]
            cap_sfx = [f[P:] for f in vd['joint']]
            res = self.suffix_batch(vd, cache, RP, [img_sfx] + cap_sfx)
            del cache
            torch.cuda.synchronize(); t2 = time.time()
            txt = self.txt_pass(vd)
            torch.cuda.synchronize(); t3 = time.time()
            stats['t_prefix'] += t1 - t0
            stats['t_suffix'] += t2 - t1
            stats['t_txt'] += t3 - t2
            stats['n_prefix'] += 1
            stats['n_img'] += 1
            stats['n_joint'] += len(cap_sfx)
            stats['n_txt'] += len(txt)
            # assemble passes
            ids_i = vd['ids_img']
            img = dict(ids=ids_i.astype(np.int32), ttype=C.token_types(ids_i, 'img'), roll=res[0]['roll'],
                       layers=res[0]['layers'], emb=res[0]['emb'])
            joint = []
            for i, f in enumerate(vd['joint']):
                r = res[i + 1]
                tt = C.token_types(f, 'joint')
                cap_idx = np.nonzero(tt == 3)[0]
                assert cap_idx[0] >= P
                cross = r['cross_rows'][:, cap_idx - P]
                joint.append(dict(ids=f.astype(np.int32), ttype=tt, roll=r['roll'], layers=r['layers'], emb=r['emb'],
                                  cross=cross))
            txtp = []
            for p in txt:
                p['ttype'] = C.token_types(p['ids'], 'txt')
                txtp.append(p)
            stats['rowsum_dev'] = max(stats['rowsum_dev'], max(r['norm_rowsum_dev'] for r in res))
            data[v] = dict(img=[img], txt=txtp, joint=joint)
            # gate c
            C.check_pass(img, 'img', grid)
            for p in joint:
                C.check_pass(p, 'joint', grid)
            for p in txtp:
                C.check_pass(p, 'txt')
            # gate b records
            if check_full:
                t4 = time.time()
                gateb.append(self.compare_full(vd, data[v], v))
                torch.cuda.synchronize()
                stats['t_full'] += time.time() - t4
        painting = dict(k=k, subset_pid=int(s['subset_pid'][k]), painting_id=str(s['subset_painting_ids'][k]),
                        image_index=d['img_index'], image_size=np.array(d['size']), grid_hw=np.array(grid),
                        cap_rows=rows, cap_sample_ids=s['sample_ids'][rows], cap_emo=s['emo'][rows],
                        sty=s['sty'][rows], gen=s['gen'][rows])
        return painting, data, gateb

    def compare_full(self, vd, dv, v):
        """Shortcut against plain full passes for img and every caption's joint input."""
        rec = dict(cos=[], roll_maxabs=[], img_l1=[], img_l1_raw=[], cross_l1=[], view=v,
                   img_l1_raw_band=[[] for _ in C.BANDS])
        img_pos_cache = None
        items = [('img', dv['img'][0], None)] + [('joint', p, True) for p in dv['joint']]
        for kind, p, is_joint in items:
            rng = None
            if is_joint:
                cap = np.nonzero(p['ttype'] == 3)[0]
                rng = (int(cap[0]), int(cap[-1]) + 1)
            f = self.full_pass(p['ids'], vd, rng)
            rec['cos'].append(float(np.dot(f['emb'], p['emb'])))
            rec['roll_maxabs'].append(float(np.abs(f['roll'] - p['roll']).max()))
            ip = p['ttype'] == 2
            rec['img_l1'].append(float(C.l1_norm_maps(f['roll'][ip], p['roll'][ip])))
            fp = dict(layers=f['layers'], ttype=p['ttype'])
            for bi, band in enumerate(C.BANDS):
                x = float(C.l1_norm_maps(C.band_map(fp, band)[0], C.band_map(p, band)[0]))
                rec['img_l1_raw'].append(x)
                rec['img_l1_raw_band'][bi].append(x)
            if is_joint:
                rec['cross_l1'].extend(C.l1_norm_maps(f['cross'], np.asarray(p['cross'], np.float32)).reshape(-1).tolist())
        return rec

    # ------------------------------------------------------------ stored-embedding gate (gate a)
    def stored_arrays(self, v):
        if v not in self.stored:
            self.stored[v] = (np.load(STORED / v / 'img.npy', mmap_mode='r'), np.load(STORED / v / 'txt.npy', mmap_mode='r'))
        return self.stored[v]


def summarise(x):
    x = np.asarray(x, np.float64)
    return dict(mean=float(x.mean()), max=float(x.max()), min=float(x.min()), n=int(len(x)))


def gate_a(ex, buffered):
    """Pooled embeddings of the held-back paintings against the stored per-view arrays."""
    res = {}
    for v in ex.views:
        img_s, txt_s = ex.stored_arrays(v)
        ci, ct = [], []
        for painting, data, _ in buffered:
            k = int(painting['k'])
            e = np.asarray(data[v]['img'][0]['emb'], np.float32)
            ci.append(float(np.dot(e, np.asarray(img_s[k], np.float32) / np.linalg.norm(np.asarray(img_s[k], np.float32)))))
            for r, p in zip(painting['cap_rows'], data[v]['txt']):
                st = np.asarray(txt_s[int(r)], np.float32)
                ct.append(float(np.dot(p['emb'], st / np.linalg.norm(st))))
        res[v] = dict(img=summarise(ci), txt=summarise(ct))
        # order check: each painting's stored row is the nearest one among the held-back paintings
        E = np.stack([d[v]['img'][0]['emb'] for _, d, _ in buffered])
        S = np.stack([np.asarray(img_s[int(p['k'])], np.float32) for p, _, _ in buffered])
        S /= np.linalg.norm(S, axis=1, keepdims=True)
        assert (np.argmax(E @ S.T, axis=1) == np.arange(len(E))).all(), f'gate a failed: image order {v}'
        res[v]['img']['nearest_row_is_own'] = True
        for m in ('img', 'txt'):
            # eager attention in bf16 gives mean cosine about 0.9987 to 0.9993 against the stored sdpa batch values
            # (att_diag_gate_a.py); a wrong row or instruction gives far lower values, so 0.998 / 0.99 is used
            assert res[v][m]['mean'] > GATE_A_MEAN and res[v][m]['min'] > GATE_A_MIN, f'gate a failed: {v} {m} {res[v][m]}'
    return res


def gate_b(records):
    out = dict(cos=[], roll_maxabs=[], img_l1=[], img_l1_raw=[], cross_l1=[])
    for r in records:
        for key in out:
            out[key].extend(r[key])
    res = {key: summarise(val) for key, val in out.items()}
    for bi, band in enumerate(C.BANDS):
        res[f'img_l1_raw {band[0]}'] = summarise([x for r in records for x in r['img_l1_raw_band'][bi]])
    print('gate b', json.dumps(res), flush=True)
    # bf16: batch shape / cache changes the rounding; eager against sdpa full passes shows the same size of gap (min 0.9965).
    # With the model in float32 the shortcut matches the plain pass to 1e-6 (att_diag_fp32.py), so the cosine floor is 0.995.
    assert res['cos']['min'] >= GATE_B_COS_MIN and res['cos']['mean'] >= 0.999, f"gate b failed: embedding cosine {res['cos']}"
    assert res['img_l1']['max'] <= 0.05, f"gate b failed: image-map L1 max {res['img_l1']['max']}"
    assert res['img_l1_raw']['max'] <= GATE_B_IMG_RAW, f"gate b failed: raw image-map L1 max {res['img_l1_raw']['max']}"
    assert res['cross_l1']['max'] <= GATE_B_CROSS, f"gate b failed: raw cross-map L1 max {res['cross_l1']['max']}"
    return res


class Prefetch:
    """One worker thread runs the CPU preparation (tokenizer and processor are not shared with the main thread)."""

    def __init__(self, ex, work, full_set, depth=4):
        self.q = queue.Queue(maxsize=depth)
        self.t = threading.Thread(target=self.run, args=(ex, work, full_set), daemon=True)
        self.t.start()

    def run(self, ex, work, full_set):
        try:
            for k in work:
                self.q.put(ex.prep(k, k in full_set))
        except BaseException as e:  # noqa: BLE001
            self.q.put(e)
            return
        self.q.put(None)

    def get(self):
        x = self.q.get()
        if isinstance(x, BaseException):
            raise x
        return x


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--views', nargs='+', default=list(C.VIEWS), choices=list(C.VIEWS))
    ap.add_argument('--check-full', type=int, default=0, help='gate b on the first N computed paintings')
    ap.add_argument('--no-shortcut', action='store_true',
                    help='fall back to plain full passes for img and joint (not implemented unless the gate fails)')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    if args.no_shortcut:
        raise SystemExit('--no-shortcut: plain full passes for joint are not implemented because the shortcut gate '
                         'passed; implement att_extract.Extractor.full_pass batching if the gate ever fails.')
    import torch
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    ex = Extractor(args.device, args.views)
    t_load = time.time() - t_start
    end = 5000 if args.limit is None else min(5000, args.start + args.limit)
    work = [k for k in range(args.start, end) if not C.is_complete(out, k, args.views)]
    n_skip = (end - args.start) - len(work)
    G = max(32, args.check_full)
    print(f'work {len(work)} paintings ({n_skip} complete, skipped); views {args.views}', flush=True)
    stats = dict(t_wait=0.0, t_prefix=0.0, t_suffix=0.0, t_txt=0.0, t_full=0.0, t_write=0.0, n_prefix=0, n_img=0,
                 n_joint=0, n_txt=0, rowsum_dev=0.0)
    pre = Prefetch(ex, work, set(work[:args.check_full]))
    buffered, gateb, gates, done = [], [], {}, 0
    t_loop = time.time()

    def flush_buffer():
        t = time.time()
        for painting, data, _ in buffered:
            C.write_painting(out, painting['k'], painting, data)
        stats['t_write'] += time.time() - t
        buffered.clear()

    for i in range(len(work)):
        t = time.time()
        d = pre.get()
        stats['t_wait'] += time.time() - t
        painting, data, gb = ex.process(d, stats, i < args.check_full)
        gateb.extend(gb)
        done += 1
        if gates.get('a') is None:
            buffered.append((painting, data, gb))
            if len(buffered) >= G or i == len(work) - 1:
                gates['a'] = gate_a(ex, buffered[:32])
                if args.check_full:
                    gates['b'] = gate_b(gateb)
                print('gates passed', flush=True)
                flush_buffer()
        else:
            t = time.time()
            C.write_painting(out, painting['k'], painting, data)
            stats['t_write'] += time.time() - t
        if done % 50 == 0:
            el = time.time() - t_loop
            print(f'[{done}/{len(work)}] k={painting["k"]} {el / done:.2f} s/painting, '
                  f'peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB', flush=True)
    el = time.time() - t_loop
    sfx = stats['t_prefix'] + stats['t_suffix']
    summary = dict(
        start=args.start, limit=args.limit, views=args.views, n_paintings=done, n_skipped_complete=n_skip,
        counts=dict(prefix=stats['n_prefix'], img=stats['n_img'], joint=stats['n_joint'], txt=stats['n_txt']),
        seconds=dict(load=t_load, loop=el, wait_for_cpu_prep=stats['t_wait'], prefix=stats['t_prefix'],
                     suffix_batch=stats['t_suffix'], txt=stats['t_txt'], gate_b_full_passes=stats['t_full'],
                     write=stats['t_write']),
        passes_per_second=dict(
            txt=stats['n_txt'] / max(stats['t_txt'], 1e-9),
            prefix=stats['n_prefix'] / max(stats['t_prefix'], 1e-9),
            suffix_batch_items=(stats['n_img'] + stats['n_joint']) / max(stats['t_suffix'], 1e-9),
            img_plus_joint_incl_prefix=(stats['n_img'] + stats['n_joint']) / max(sfx, 1e-9)),
        seconds_per_painting=el / max(done, 1),
        shortcut_used=dict(img=True, joint=True),
        gate_a=gates.get('a'), gate_b=gates.get('b'), check_full=args.check_full,
        max_abs_rowsum_dev_of_bf16_attention_rows=stats['rowsum_dev'],
        peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9, git_head=git_head())
    tag = f'run_{args.start}_{args.limit if args.limit is not None else "all"}.json'
    with open(out / tag, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'done {done} paintings in {el:.0f}s ({el / max(done, 1):.2f} s/painting); wrote {out / tag}', flush=True)


if __name__ == '__main__':
    main()
