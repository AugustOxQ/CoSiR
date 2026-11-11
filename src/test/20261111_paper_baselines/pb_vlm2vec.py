"""r25 VLM2Vec-V2.0 image embeddings under the three aspect instructions (run with /root/venvs/vlm2vec/bin/python).

For each aspect in pb_paths.ASPECTS writes pb_paths.VLM2VEC_STORE/<aspect>/{emb.npy, keys.json, meta.json}:
    emb.npy  float32 [N, d], unit L2 rows, rows in held_images.json `names` order (= r17's held key order)
    keys.json  the names; meta.json  model, revisions, versions, exact prompt, image settings, timing, hashes.

The embedding path is the VLM2Vec repo's own MMEB-V2 image-query path (eval.py + eval_collator + image_cls parser):
    model   MMEBModel.load(ModelArguments(model_name='Qwen/Qwen2-VL-2B-Instruct', checkpoint_path=<V2.0 LoRA>,
            lora=True, pooling='last', normalize=True, model_backbone='qwen2_vl')), bf16, merged LoRA
    prompt  process_input_text(instruction, 'qwen2_vl', add_image_token=True) -> '<|image_pad|> <instruction>', then
            the parser's `.replace(" \\n", "\\n") + "\\n"`   (text '<|image_pad|> <instruction>\\n')
    image   PIL RGB, resized inside Qwen2VLImageProcessor (DataArguments min 28*28*4, max 28*28*1280 pixels)
    batch   Qwen2_VL_process_fn (the repo's collate fn) -> model(qry=inputs)['qry_reps'] under bf16 autocast.
Deviations (numerics only): flash-attn is not installed, so the Qwen2-VL attention is sdpa (the repo asks for
flash_attention_2); and the vendored vision sdpa attention, which builds one dense [total_patches]^2 block mask over the
whole batch (out of memory at batch >= 4 with 1,280-token images), is replaced by the same attention run per image
(`_patch_vision_attention`), which is the same math with memory linear in the batch.

    pb_vlm2vec.py [--limit N] [--store DIR] [--force] [--batch-size B] [--aspects a b ..] [--check] [--demo]
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pb_paths as P  # noqa: E402  (imports nothing heavy)

# Import isolation: VLM2Vec's top-level package is `src`; put its checkout first before anything imports `src`.
sys.path.insert(0, str(P.VLM2VEC_CODE))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402
from PIL import Image  # noqa: E402

import src  # noqa: E402
assert Path(src.__file__).resolve().is_relative_to(P.VLM2VEC_CODE.resolve()), f'`src` resolved to {src.__file__}'
from src.arguments import ModelArguments, DataArguments  # noqa: E402
from src.model.model import MMEBModel  # noqa: E402
from src.model import model as model_mod  # noqa: E402
from src.model.processor import (load_processor, process_input_text, Qwen2_VL_process_fn, backbone2model,  # noqa: E402
                                 QWEN2_VL, VLM_IMAGE_TOKENS)
from src.utils.basic_utils import batch_to_device  # noqa: E402
assert Path(model_mod.__file__).resolve().is_relative_to(P.VLM2VEC_CODE.resolve())

BASE_MODEL = 'Qwen/Qwen2-VL-2B-Instruct'
BACKBONE = 'qwen2_vl'
DEFAULT_BATCH = 16
DTYPE = 'bfloat16'
ATTN = 'sdpa'


def names_sha(names):
    return hashlib.sha256('\n'.join(names).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'wb') as f:
        writer(f)
    tmp.replace(path)


def read_inputs(limit):
    d = json.loads((P.INPUTS / 'held_images.json').read_text())
    names, images = d['names'], d['images']
    assert names_sha(names) == d['names_sha256'], 'held_images.json: names_sha256 mismatch'
    assert len(names) == len(images) == P.N_HELD, len(names)
    if limit:
        names, images = names[:limit], images[:limit]
    return names, images, d['names_sha256']


def prompt_for(instruction):
    """The repo's MMEB-V2 image-query string (image_cls parser): image token, instruction, trailing newline."""
    t = process_input_text(instruction, BACKBONE, text=None, add_image_token=True)
    return t.replace(' \n', '\n') + '\n'


def snapshot_sha(local_dir):
    return Path(local_dir).resolve().name


def _patch_vision_attention():
    """Per-image attention in the vision tower: identical to the repo's block-diagonal mask, without the dense mask."""
    from src.model.vlm_backbone.qwen2_vl import modeling_qwen2_vl as mq
    F = torch.nn.functional

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb=None):
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        q = mq.apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        k = mq.apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)
        q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)      # [heads, seq, dim]
        bounds = cu_seqlens.tolist()
        outs = [F.scaled_dot_product_attention(q[:, a:b].unsqueeze(0), k[:, a:b].unsqueeze(0), v[:, a:b].unsqueeze(0),
                                               dropout_p=0.0).squeeze(0) for a, b in zip(bounds[:-1], bounds[1:])]
        attn_output = torch.cat(outs, dim=1).transpose(0, 1).reshape(seq_length, -1)
        return self.proj(attn_output)

    mq.VisionSdpaAttention.forward = forward


def load_model():
    _patch_vision_attention()
    from huggingface_hub import snapshot_download
    ckpt = snapshot_download(P.VLM2VEC_MODEL)   # LoRA adapter + tokenizer/processor files, default HF cache
    base = snapshot_download(BASE_MODEL)
    # flash-attn is not installed: the repo sets config._attn_implementation='flash_attention_2' before building the
    # backbone, so swap in sdpa at the from_pretrained call (the vendored Qwen2-VL modeling supports sdpa).
    orig_cls = backbone2model[QWEN2_VL]

    class _Sdpa(orig_cls):
        @classmethod
        def from_pretrained(cls, *a, **kw):
            cfg = kw['config']
            cfg._attn_implementation = ATTN
            if hasattr(cfg, 'vision_config'):
                cfg.vision_config._attn_implementation = ATTN
            return orig_cls.from_pretrained.__func__(orig_cls, *a, **kw)
    backbone2model[QWEN2_VL] = _Sdpa
    margs = ModelArguments(model_name=BASE_MODEL, checkpoint_path=ckpt, pooling='last', normalize=True,
                           model_backbone=BACKBONE, lora=True)
    dargs = DataArguments()
    processor = load_processor(margs, dargs)
    model = MMEBModel.load(margs, is_trainable=False)
    model = model.to('cuda:0', dtype=torch.bfloat16).eval()
    info = {'model': P.VLM2VEC_MODEL, 'revision': snapshot_sha(ckpt), 'base_model': BASE_MODEL,
            'base_revision': snapshot_sha(base), 'min_pixels': dargs.resize_min_pixels,
            'max_pixels': dargs.resize_max_pixels, 'image_resolution': dargs.image_resolution,
            'resize_use_processor': dargs.resize_use_processor, 'attn_implementation': ATTN,
            'vision_attention': 'sdpa per image (_patch_vision_attention)'}
    return model, processor, info


def code_commit():
    return subprocess.check_output(['git', '-C', str(P.VLM2VEC_CODE), 'rev-parse', 'HEAD'], text=True).strip()


def open_rgb(rel):
    im = Image.open(P.IMAGE_ROOT / rel)
    im = im.convert('RGB')
    return im


@torch.no_grad()
def embed(model, processor, paths, prompt, batch, prefetch=None):
    """[len(paths), d] float32 unit rows; images opened with PIL (RGB), batched through the repo's collate fn."""
    from concurrent.futures import ThreadPoolExecutor

    def prep(chunk):
        ims = [open_rgb(p) for p in chunk]
        return Qwen2_VL_process_fn({'text': [prompt] * len(ims), 'images': ims}, processor)

    chunks = [paths[i:i + batch] for i in range(0, len(paths), batch)]
    out = []
    with ThreadPoolExecutor(2) as ex:
        futs = [ex.submit(prep, c) for c in chunks[:2]]
        for i in range(len(chunks)):
            inputs = futs[i].result()
            if i + 2 < len(chunks):
                futs.append(ex.submit(prep, chunks[i + 2]))
            futs[i] = None
            inputs = batch_to_device(inputs, 'cuda:0')
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                r = model(qry=inputs)['qry_reps']
            out.append(torch.nn.functional.normalize(r.float(), dim=-1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def complete(d, n):
    try:
        m = json.loads((d / 'meta.json').read_text())
        return (d / 'emb.npy').exists() and (d / 'keys.json').exists() and m.get('n') == n
    except Exception:
        return False


def run_aspects(a, model, processor, info, names, images, nsha, load_seconds):
    store = Path(a.store)
    commit = code_commit()
    for asp in a.aspects:
        d = store / asp
        if complete(d, len(names)) and not a.force:
            print(f'{asp}: complete, skip', flush=True)
            continue
        prompt = prompt_for(P.INSTRUCTIONS[asp])
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        emb = embed(model, processor, images, prompt, a.batch_size)
        torch.cuda.synchronize()
        sec = time.time() - t0
        assert emb.shape[0] == len(names) and np.isfinite(emb).all()
        atomic_write(d / 'emb.npy', lambda f: np.save(f, emb))
        atomic_write(d / 'keys.json', lambda f: f.write(json.dumps(names).encode()))
        meta = {**info, 'aspect': asp, 'instruction': P.INSTRUCTIONS[asp], 'prompt': prompt,
                'vlm2vec_code_commit': commit, 'transformers': transformers.__version__, 'torch': torch.__version__,
                'dtype': DTYPE, 'autocast': DTYPE, 'batch_size': a.batch_size, 'pooling': 'last (last non-pad token)',
                'normalize': 'L2 in model (bf16) and again in float32 here', 'dim': int(emb.shape[1]),
                'seconds': round(sec, 2), 'ms_per_image': round(1000 * sec / len(names), 2),
                'peak_gpu_gb': round(torch.cuda.max_memory_allocated() / 2**30, 2), 'model_load_seconds': round(load_seconds, 2), 'n': len(names), 'emb_sha256': file_sha(d / 'emb.npy'),
                'names_sha256': nsha, 'limit': a.limit or None,
                'image_open': 'PIL Image.open(...).convert("RGB"), resized inside Qwen2VLImageProcessor'}
        atomic_write(d / 'meta.json', lambda f: f.write(json.dumps(meta, indent=1).encode()))
        nrm = np.linalg.norm(emb, axis=1)
        print(f'{asp}: {emb.shape} norm [{nrm.min():.6f},{nrm.max():.6f}] {meta["ms_per_image"]} ms/img '
              f'prompt={prompt!r}', flush=True)


def batch_check(a, model, processor, names, images):
    n = min(8, len(images))   # batch 1 vs batch n
    prompt = prompt_for(P.INSTRUCTIONS['emotion'])
    e1 = embed(model, processor, images[:n], prompt, 1)
    e8 = embed(model, processor, images[:n], prompt, n)
    cos = (e1 * e8).sum(1)
    res = {'n': n, 'prompt': prompt, 'cosine_batch1_vs_batch8': [round(float(c), 6) for c in cos],
           'min_cosine': float(cos.min()), 'pass_0.999': bool(cos.min() >= 0.999)}
    atomic_write(Path(a.store) / 'batch_check.json', lambda f: f.write(json.dumps(res, indent=1).encode()))
    print('batch check', res, flush=True)


def demo_check(model, processor):
    """README demo (experiments/examples/qwen2vl/demo_v2_image.py): expected 0.3281 / 0.2871 and, batched, 0.2974 / 0.2390."""
    ex = P.VLM2VEC_CODE / 'assets' / 'example.jpg'
    q = f'{VLM_IMAGE_TOKENS[QWEN2_VL]} Represent the given image with the following question: What is in the image'
    texts = ['A cat and a dog', 'A cat and a tiger']
    with torch.no_grad():
        qi = Qwen2_VL_process_fn({'text': [q, q], 'images': [Image.open(ex), Image.open(ex)]}, processor)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            qr = model(qry=batch_to_device(qi, 'cuda:0'))['qry_reps']
            ti = Qwen2_VL_process_fn({'text': texts, 'images': [None, None]}, processor)
            tr = model(tgt=batch_to_device(ti, 'cuda:0'))['tgt_reps']
        sim = model.compute_similarity(qr, tr).float().cpu().numpy()
        # single query, single targets (the first half of the demo)
        qi1 = Qwen2_VL_process_fn({'text': [q], 'images': [Image.open(ex)]}, processor)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            qr1 = model(qry=batch_to_device(qi1, 'cuda:0'))['qry_reps']
        single = model.compute_similarity(qr1, tr).float().cpu().numpy()
    print('demo single-query sims (README 0.3281, 0.2871):', np.round(single, 4).tolist(), flush=True)
    print('demo batched sims (README [[0.2974 0.2390],[0.2978 0.2390]]):', np.round(sim, 4).tolist(), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--store', default=str(P.VLM2VEC_STORE))
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--batch-size', type=int, default=DEFAULT_BATCH)
    ap.add_argument('--aspects', nargs='+', default=list(P.ASPECTS))
    ap.add_argument('--check', action='store_true', help='batch 1 vs batch 8 consistency on the first 8 images')
    ap.add_argument('--demo', action='store_true', help="reproduce the repo's README/demo similarities")
    a = ap.parse_args(argv)
    names, images, nsha = read_inputs(a.limit)
    todo = [x for x in a.aspects if not (complete(Path(a.store) / x, len(names)) and not a.force)]
    if not todo and not a.check and not a.demo:
        print('all aspects complete, nothing to do')
        return
    t0 = time.time()
    model, processor, info = load_model()
    load_seconds = time.time() - t0
    print(f'model loaded in {load_seconds:.1f}s; transformers {transformers.__version__}, torch {torch.__version__}',
          flush=True)
    if a.demo:
        demo_check(model, processor)
    if a.check:
        batch_check(a, model, processor, names, images)
    run_aspects(a, model, processor, info, names, images, nsha, load_seconds)


if __name__ == '__main__':
    main()
