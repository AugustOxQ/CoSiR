"""r28 job C (Table 8): Qwen3-VL-Embedding-2B text pass per caption, and the caption tower per caption.
Same setup as r19's probe (`src/test/20261103_past_picks/pp_probe.py`): batch 32, 3 warm-up batches (the first three,
not timed), wall time per batch with cuda synchronize before and after, 3 repeats, foreign GPU processes polled before and
after every repeat (a repeat with one is discarded and rerun). Model time only: captions are already strings and the
embeddings already in memory; no file reads or decoding inside the timed loop. The Qwen text pass is
`enc.encode_texts(batch, 32)` as the extractor calls it, so the tokenizer call is inside the timed region (r19's image pass
likewise included the processor). Caption sample: 2,000 rows drawn with numpy seed 0 from the 308,723 rows of the store.
Caption tower: the three seed-0 SigLIP 1 c* views' `cap` side (2048 -> out), fp32, on the stored Qwen caption embeddings;
host-to-device copy once, three forwards, three outputs copied back (as r19's heads pass)."""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, '/project/CoSiR/src/test/20261103_past_picks')
import pp_paths as PP  # noqa: E402  (must come first)
import pp_probe as PR  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

STORE = Path('/data/SSD2/pre_extract/artelingo_full/qwen3vl_emb_2b')
W = '/data/SSD2/cosir_r14/20261102_backbones_sweep/outputs/cstar/siglip1/weights/k11=40'
OUT = Path('/project/CoSiR/src/test/20261112_siglip_main/records/time_text.json')
N, BATCH, WARMUP, REPEATS, SEED = 2000, 32, 3, 3, 0
GENERIC = "Represent the user's input."


def summary(rec):
    s = np.array(rec['batch_seconds'])
    full = [i for i, n in enumerate(rec['batch_n']) if n == BATCH]
    ms = 1000 * s[full] / BATCH
    return {'ms_per_caption': rec['ms_per_image'], 'n': rec['n_images'], 'n_batches': len(s),
            'batch_ms_per_caption_mean_full_batches': float(ms.mean()), 'batch_ms_per_caption_std': float(ms.std(ddof=1)),
            'batch_ms_per_caption_min': float(ms.min()), 'batch_ms_per_caption_max': float(ms.max()),
            'attempt': rec['attempt'], 'cpu_load_before': rec['cpu_load_before'], 'cpu_load_after': rec['cpu_load_after'],
            'batch_seconds': rec['batch_seconds']}


def main():
    own = os.getpid()
    gpu_before = PR.poll_gpu()
    import pp_extract
    import rb_rows
    import rb_student
    assert torch.cuda.is_available()
    idx = json.loads((STORE / 'index.json').read_text())
    n_txt = idx['n_texts']
    txt = np.load(STORE / 'txt.npy', mmap_mode='r')
    assert txt.shape == (n_txt, 2048) and len(idx['texts']) == n_txt
    rows = np.sort(np.random.RandomState(SEED).choice(n_txt, N, replace=False))
    ann = json.load(open('/data/PDD/artelingo/artelingo_train.json'))
    texts = [ann[idx['texts'][int(i)]['sample_id']]['caption'] for i in rows]      # joined by the store's sample_ids
    assert all(isinstance(t, str) for t in texts)
    emb = np.ascontiguousarray(txt[rows], dtype=np.float32)

    enc, identity, _ = pp_extract.load_encoder('qwen')
    assert enc.instruction == GENERIC, enc.instruction
    sync, clock = torch.cuda.synchronize, time.perf_counter

    # sanity: the live pass reproduces the stored rows (cosine), on the first batch
    live = enc.encode_texts(texts[:BATCH], BATCH)
    cos = (live * emb[:BATCH]).sum(1) / np.linalg.norm(live, axis=1) / np.linalg.norm(emb[:BATCH], axis=1)
    print('live vs stored cosine min', cos.min(), flush=True)

    out = {'task': 'r28 Table 8 text-pass time', 'written': time.strftime('%F %H:%M'), 'gpu': torch.cuda.get_device_name(0),
           'torch': torch.__version__, 'batch': BATCH, 'warmup_batches': WARMUP, 'repeats': REPEATS, 'n_captions': N,
           'sample': f'np.random.RandomState({SEED}).choice({n_txt}, {N}, replace=False), sorted; store {STORE}',
           'instruction': enc.instruction, 'model': identity, 'live_vs_stored_cos_min': float(cos.min()),
           'gpu_processes_before_start': gpu_before}
    discards = []

    def repeats(name, encode, items):
        recs = []
        for k in range(REPEATS):
            rec, d = PR.run_repeat(name, k, lambda: PR.model_repeat(name, encode, items, k, BATCH, WARMUP, sync, clock),
                                   PR.poll_gpu, own, wait=30.0)
            discards.extend(d)
            recs.append(summary(rec))
            print(name, k, rec['ms_per_image'], flush=True)
        m = [r['ms_per_caption'] for r in recs]
        return {'repeats': recs, 'ms_per_caption_mean_of_repeats': float(np.mean(m)),
                'ms_per_caption_repeat_min': min(m), 'ms_per_caption_repeat_max': max(m)}

    out['qwen_text_pass'] = repeats('qwen_text', lambda b: enc.encode_texts(b, BATCH), texts)
    del enc
    torch.cuda.empty_cache()

    device = torch.device('cuda')
    caps = []
    for aspect in PP.ASPECTS:
        _, views = rb_student.build_model(rb_rows.recipe(rb_rows.REFERENCE)['base'], list(PP.ASPECTS), device, width=1152)
        v = views[aspect]
        v.load_state_dict(torch.load(f'{W}/{aspect}__s0.pt', map_location=device, weights_only=True))
        v.eval()
        caps.append(v['cap'])

    def enc_cap(xb):
        with torch.no_grad():
            t = torch.from_numpy(xb).to(device)
            outs = [m(t) for m in caps]
            [o.cpu().numpy() for o in outs]

    out['caption_tower_three_aspects'] = repeats('cap_tower', enc_cap, emb)
    out['caption_tower_params_each'] = [sum(p.numel() for p in m.parameters()) for m in caps]
    out['discarded_repeats'] = [{k: v for k, v in d.items() if k != 'record'} for d in discards]
    out['gpu_processes_after_end'] = PR.poll_gpu()
    out['foreign_during_timing'] = bool(discards)
    OUT.write_text(json.dumps(out, indent=1))
    print('written', OUT, flush=True)


if __name__ == '__main__':
    main()
