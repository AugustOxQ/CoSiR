"""r26 ticket A: extract the two new feature stores (CLIP ViT-L/14 patch tokens; ModernBERT-GoEmotions captions).

    python po_extract.py {clip_patch,modernbert} [--outdir DIR] [--device cuda|cpu] [--smoke N]

Store formats are fixed in po_paths.py. Atomic writing follows src/test/20261018_percept_core/pc_roberta.py: exclusive
`writing.pending` marker, `*.pending.*` files renamed at the end, a non-empty folder or one with done.json is refused.
A failed parity gate leaves the pending files and the marker in place (store not finalised) and the script crashes.
"""
import argparse
import json
import os
import time
from pathlib import Path

# Set before transformers imports; extraction must never contact the hub.
os.environ['HF_HOME'] = '/data/SSD2/HF_home'
os.environ['HF_HUB_OFFLINE'] = '1'

import numpy as np
import torch

import po_paths as P

CLIP_BATCH = 128          # the global L/14 store used 128 images per batch (scripts/extract_features.py BATCH)
MB_BATCH = 256            # captions per batch, as pc_roberta.py
MB_MAX_LENGTH = 64
LOAD_WORKERS = 8
CLIP_PARITY_MEAN, CLIP_PARITY_MIN = 0.999, 0.99
MB_PARITY_ROWS, MB_PARITY_TOL = 64, 1e-4

CLIP_FORM = 'visual_projection(vision_model(pixel_values).last_hidden_state), all 257 tokens, CLS first; no post-layernorm, no normalisation'
CLIP_PREPROC = ('AutoProcessor.image_processor on PIL image.convert("RGB") lists, as ClipL14.encode_images in '
                'src/data/feature_extract.py (global store); model dtype float16 on cuda, float32 on cpu, no autocast')
MB_POOLING = 'attention-mask mean of last_hidden_state, including special tokens'


def snapshot(model, revision):
    path = P.HF_HOME / 'hub' / ('models--' + model.replace('/', '--')) / 'snapshots' / revision
    assert path.is_dir(), path
    return path


def open_store(outdir, smoke):
    """Refuse a finished or non-empty folder; take the exclusive writing marker. Returns the marker path."""
    outdir = Path(outdir)
    if smoke:
        assert outdir.resolve() not in (P.CLIP_L14_PATCH.resolve(), P.MODERNBERT.resolve()), \
            'smoke needs a separate --outdir'
    assert not (outdir / 'done.json').exists(), 'refusing to overwrite a finished store'
    outdir.mkdir(parents=True, exist_ok=True)
    assert not any(outdir.iterdir()), 'store folder must be new or empty'
    marker = outdir / 'writing.pending'
    with marker.open('x'):   # exclusive: two writers cannot start in the same empty folder
        pass
    return marker


def finalise(outdir, marker, files, done):
    """files: [(pending name, final name)] already written; done.json is renamed last."""
    (outdir / 'done.pending.json').write_text(json.dumps(done, indent=2, allow_nan=False) + '\n')
    for pending, final in files:
        (outdir / pending).replace(outdir / final)
    marker.unlink()
    (outdir / 'done.pending.json').replace(outdir / 'done.json')


def load_annotations():
    rows = json.load(open(P.ANNOTATIONS))
    assert len(rows) == P.N_CAPTIONS, len(rows)
    return rows


# ------------------------------------------------------------------------------------------------ clip_patch
class BatchImages(torch.utils.data.Dataset):
    """One item = one processed batch [B, 3, 224, 224] float32; the processor call is the global store's."""

    def __init__(self, paths, processor, batch):
        self.paths, self.processor, self.batch = paths, processor, batch

    def __len__(self):
        return (len(self.paths) + self.batch - 1) // self.batch

    def __getitem__(self, b):
        from PIL import Image
        images = []
        for path in self.paths[b * self.batch:(b + 1) * self.batch]:
            with Image.open(path) as im:
                images.append(im.convert('RGB'))
        return self.processor.image_processor(images, return_tensors='pt')['pixel_values']


def extract_clip_patch(outdir, device, smoke):
    from transformers import AutoModel, AutoProcessor
    outdir = Path(outdir)
    marker = open_store(outdir, smoke)
    started = time.perf_counter()

    rows = load_annotations()
    names = sorted({r['painting'] for r in rows})
    assert len(names) == P.N_PAINTINGS, len(names)
    index = json.load(open(P.CLIP_L14_GLOBAL / 'index.json'))
    images = index['images']
    assert len(images) == P.N_PAINTINGS and index['backbone'] == 'clip_l14' and index['dtype'] == 'float16'
    gname = [e['painting'] for e in images]
    assert len(set(gname)) == len(gname) and set(gname) == set(names), 'painting name sets differ'
    g_row = {n: i for i, n in enumerate(gname)}
    g_image = {e['painting']: e['image'] for e in images}
    del rows
    count = P.N_PAINTINGS
    if smoke:
        assert 0 < smoke <= count
        count = smoke
    names = names[:count]
    paths = [P.IMAGE_ROOT / g_image[n] for n in names]
    assert all(p.is_file() for p in paths[:1000]), 'image path check'

    img_global = np.load(P.CLIP_L14_GLOBAL / 'img.npy', mmap_mode='r')
    assert img_global.shape == (P.N_PAINTINGS, P.PATCH_WIDTH) and img_global.dtype == np.float16
    ref_rows = np.array([g_row[n] for n in names])

    snap = snapshot(P.CLIP_L14_MODEL, P.CLIP_L14_REVISION)
    dtype = torch.float16 if device.startswith('cuda') else torch.float32
    model = AutoModel.from_pretrained(str(snap), dtype=dtype, local_files_only=True).to(device).eval()
    processor = AutoProcessor.from_pretrained(str(snap), local_files_only=True)
    assert model.config.projection_dim == P.PATCH_WIDTH
    vision, proj = model.vision_model, model.visual_projection

    pending = outdir / 'tokens.pending.npy'
    tokens = np.lib.format.open_memmap(pending, mode='w+', dtype=np.float16,
                                       shape=(count, P.PATCH_TOKENS, P.PATCH_WIDTH))
    cosines = np.empty(count, dtype=np.float64)
    loader = torch.utils.data.DataLoader(
        BatchImages(paths, processor, CLIP_BATCH), batch_size=None, shuffle=False,
        num_workers=min(LOAD_WORKERS, (count + CLIP_BATCH - 1) // CLIP_BATCH),
        worker_init_fn=lambda _: torch.set_num_threads(1), prefetch_factor=2)
    encoding_started = time.perf_counter()
    done_rows, finite = 0, True
    with torch.inference_mode():
        for batch_no, pixels in enumerate(loader):
            a, b = done_rows, done_rows + len(pixels)
            assert pixels.shape == (b - a, 3, 224, 224) and b <= count
            hidden = vision(pixel_values=pixels.to(device, dtype)).last_hidden_state
            assert hidden.shape == (b - a, P.PATCH_TOKENS, 1024), hidden.shape
            out = proj(hidden)
            assert out.shape == (b - a, P.PATCH_TOKENS, P.PATCH_WIDTH)
            # global embedding as the global store made it: projected post-layernorm CLS, L2-normalised
            glob = torch.nn.functional.normalize(proj(vision.post_layernorm(hidden[:, 0])).float(), dim=-1).cpu().numpy()
            out16 = out.to(torch.float16).cpu().numpy()
            finite &= bool(np.isfinite(out16).all())
            assert finite, f'non-finite tokens in rows {a}:{b}'
            tokens[a:b] = out16
            ref = np.asarray(img_global[ref_rows[a:b]], dtype=np.float64)
            ref /= np.linalg.norm(ref, axis=1, keepdims=True)
            cosines[a:b] = (glob.astype(np.float64) * ref).sum(1)
            done_rows = b
            if (batch_no + 1) % 20 == 0 or b == count:
                elapsed = time.perf_counter() - encoding_started
                print(f'CLIP L/14 patch: {b:,}/{count:,} images, {elapsed / b:.3f} s/image, '
                      f'cosine so far mean {cosines[:b].mean():.6f} min {cosines[:b].min():.6f}', flush=True)
    assert done_rows == count
    tokens.flush()
    del tokens
    encoding_seconds = time.perf_counter() - encoding_started

    gate = dict(cosine_mean=float(cosines.mean()), cosine_min=float(cosines.min()), n_rows=count,
                tokens_finite=finite, thresholds=dict(mean_above=CLIP_PARITY_MEAN, min_above=CLIP_PARITY_MIN),
                reference=str(P.CLIP_L14_GLOBAL / 'img.npy'),
                global_form='normalize(visual_projection(post_layernorm(last_hidden_state[:, 0]))), joined by painting name')
    gate['ok'] = bool(finite and gate['cosine_mean'] > CLIP_PARITY_MEAN and gate['cosine_min'] > CLIP_PARITY_MIN)
    print('parity gate:', json.dumps(gate), flush=True)
    assert gate['ok'], f'parity gate failed; store not finalised (pending files kept in {outdir}): {gate}'

    np.savez(outdir / 'ids.pending.npz', painting_names=np.array(names))
    done = dict(model_id=P.CLIP_L14_MODEL, revision=P.CLIP_L14_REVISION, form=CLIP_FORM, preprocessing=CLIP_PREPROC,
                image_root=str(P.IMAGE_ROOT), count=count, shape=[count, P.PATCH_TOKENS, P.PATCH_WIDTH],
                dtype='float16', model_dtype=str(dtype), device=device, batch=CLIP_BATCH,
                attn_implementation=str(model.config.vision_config._attn_implementation),
                seconds=time.perf_counter() - started, seconds_per_image=encoding_seconds / count,
                parity_gate=gate, smoke=bool(smoke))
    finalise(outdir, marker, [('tokens.pending.npy', 'tokens.npy'), ('ids.pending.npz', 'ids.npz')], done)
    print(f'{encoding_seconds / count:.4f} s/image; wrote {outdir}', flush=True)
    return done


# ------------------------------------------------------------------------------------------------ modernbert
def mb_encode(captions, tokenizer, model, device):
    """pc_roberta.py's encode: padding=True, truncation, attention-mask mean of last_hidden_state."""
    assert captions and all(isinstance(t, str) and t.strip() for t in captions)
    assert not model.training
    with torch.no_grad():
        enc = tokenizer(captions, padding=True, truncation=True, max_length=MB_MAX_LENGTH,
                        return_tensors='pt').to(device)
        hidden = model(**enc).last_hidden_state
        mask = enc['attention_mask'].unsqueeze(-1)
        x = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        x = x.cpu().numpy().astype(np.float32, copy=False)
    assert x.shape == (len(captions), P.TEXT_WIDTH) and x.dtype == np.float32 and np.isfinite(x).all()
    return x


def extract_modernbert(outdir, device, smoke):
    from transformers import AutoModel, AutoTokenizer
    outdir = Path(outdir)
    marker = open_store(outdir, smoke)
    started = time.perf_counter()

    rows = load_annotations()
    texts = [r['caption'] for r in rows]
    painting_ids = np.array([r['painting'] for r in rows])
    del rows
    assert len(texts) == len(painting_ids) == P.N_CAPTIONS and np.issubdtype(painting_ids.dtype, np.str_)
    count = P.N_CAPTIONS
    if smoke:
        assert MB_PARITY_ROWS <= smoke <= count, 'smoke must cover the 64 parity captions'
        count = smoke
    texts, painting_ids = texts[:count], painting_ids[:count]
    sample_ids = np.arange(count, dtype=np.int64)    # sample_id = annotation row index

    snap = snapshot(P.MODERNBERT_MODEL, P.MODERNBERT_REVISION)
    tokenizer = AutoTokenizer.from_pretrained(str(snap), local_files_only=True)
    model = AutoModel.from_pretrained(str(snap), local_files_only=True).to(device).eval()   # fp32
    assert type(model).__name__ == 'ModernBertModel' and next(model.parameters()).dtype == torch.float32
    attn = str(model.config._attn_implementation)
    print(f'ModernBERT attention implementation: {attn}', flush=True)

    # captions the 64-token cut shortens (special tokens counted, as the model sees them)
    n_truncated = 0
    for s in range(0, count, 4096):
        lengths = tokenizer(texts[s:s + 4096], truncation=False, add_special_tokens=True)['input_ids']
        n_truncated += sum(len(i) > MB_MAX_LENGTH for i in lengths)
    print(f'captions longer than {MB_MAX_LENGTH} tokens: {n_truncated:,}/{count:,}', flush=True)

    pending = outdir / 'txt.pending.npy'
    x = np.lib.format.open_memmap(pending, mode='w+', dtype=np.float32, shape=(count, P.TEXT_WIDTH))
    encoding_started = time.perf_counter()
    for start in range(0, count, MB_BATCH):
        end = min(start + MB_BATCH, count)
        x[start:end] = mb_encode(texts[start:end], tokenizer, model, device)
        if (start // MB_BATCH + 1) % 20 == 0 or end == count:
            print(f'ModernBERT: {end:,}/{count:,} captions, '
                  f'{(time.perf_counter() - encoding_started) / end * 1000:.3f} ms/caption', flush=True)
    x.flush()
    encoding_seconds = time.perf_counter() - encoding_started

    # parity gate: first 64 captions batched (as stored) against each caption alone (batch 1, no padding)
    alone = np.concatenate([mb_encode([t], tokenizer, model, device) for t in texts[:MB_PARITY_ROWS]])
    diff = float(np.abs(np.asarray(x[:MB_PARITY_ROWS]) - alone).max())
    parity = dict(rows=MB_PARITY_ROWS, max_abs_diff=diff, tolerance=MB_PARITY_TOL, ok=bool(diff < MB_PARITY_TOL))
    print('parity gate:', json.dumps(parity), flush=True)
    assert parity['ok'], f'parity gate failed; store not finalised (pending files kept in {outdir}): {parity}'
    del x, model, tokenizer

    np.savez(outdir / 'ids.pending.npz', sample_ids=sample_ids, painting_ids=painting_ids)
    done = dict(model_id=P.MODERNBERT_MODEL, revision=P.MODERNBERT_REVISION, pooling=MB_POOLING,
                max_length=MB_MAX_LENGTH, n_truncated=n_truncated, batch=MB_BATCH, count=count,
                shape=[count, P.TEXT_WIDTH], dtype='float32', device=device, attn_implementation=attn,
                seconds=time.perf_counter() - started, seconds_per_caption=encoding_seconds / count,
                parity=parity, smoke=bool(smoke))
    finalise(outdir, marker, [('txt.pending.npy', 'txt.npy'), ('ids.pending.npz', 'ids.npz')], done)
    print(f'{encoding_seconds / count * 1000:.3f} ms/caption; wrote {outdir}', flush=True)
    return done


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('which', choices=['clip_patch', 'modernbert'])
    parser.add_argument('--outdir', type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--smoke', type=int, default=0, help='extract only the first N rows (needs --outdir)')
    args = parser.parse_args()
    default = P.CLIP_L14_PATCH if args.which == 'clip_patch' else P.MODERNBERT
    assert not args.smoke or args.outdir is not None, 'smoke needs a separate --outdir'
    outdir = args.outdir or default
    run = extract_clip_patch if args.which == 'clip_patch' else extract_modernbert
    run(outdir, args.device, args.smoke)


if __name__ == '__main__':
    main()
