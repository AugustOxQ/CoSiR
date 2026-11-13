"""r29 Table 8 (ticket 09; spec §3.4, SC8, D8): text-pass time per caption, r28's protocol (r28_time.py: batch 32, three
warm-up batches, wall time per batch with cuda synchronize, three repeats, foreign GPU processes polled around every repeat,
tokenisation inside the timed loop, the same 2,000 captions) for (a) the SigLIP 1 text pass (`encode_texts` of
`load_encoder('siglip_so400m')`, on the texts in K*'s own caption form: with the prefix when K* is `siglip1_prefix`) and (b) the new caption tower (`view['cap']` of the three aspects' K* checkpoints, timed together as r28's three-aspect pass, 1152 -> 2048, fp32) on
the stored rows of those captions from K*'s store (K* is read from the committed `records/prefix_pick.json`). r28's Qwen3 value is read from r28's record. Descriptive; no verdict.

    python r29_time.py --weights-dir DIR [--seed S] [--out PATH] [--repeats N]
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r29_paths as P  # noqa: E402

P.use_older_rounds()
import r28_time as R28  # noqa: E402  (imports pp_paths first, as it requires; puts r19's pp_probe on sys.path)
import pp_probe as PR  # noqa: E402
import r29_captions as C  # noqa: E402
import r29_model as M  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
P.check_older_imports()  # every older-round module loaded so far is the main checkout's copy

R28_RECORD = P.R28_RECORDS / 'time_text.json'
OUT = P.RECORDS / 'time_text.json'
from r29_time_inputs import PICK, FORMS, read_kstar, caption_texts  # noqa: E402
SIGLIP_WIDTH, TOWER_WIDTH = 1152, 2048


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights-dir', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=str(OUT))
    ap.add_argument('--repeats', type=int, default=R28.REPEATS)
    a = ap.parse_args()
    kstar, kform = read_kstar()
    N, BATCH, WARMUP, SEED = R28.N, R28.BATCH, R28.WARMUP, R28.SEED
    own = os.getpid()
    gpu_before = PR.poll_gpu()
    assert torch.cuda.is_available()
    r28 = json.loads(R28_RECORD.read_text())
    assert (r28['batch'], r28['warmup_batches'], r28['n_captions']) == (BATCH, WARMUP, N)
    qwen_ms = r28['qwen_text_pass']['ms_per_caption_mean_of_repeats']
    assert abs(qwen_ms - 3.616) < 5e-4, qwen_ms

    # the same 2,000 captions as r28: its draw from the Qwen store, joined to the annotations by sample ID
    idx = json.loads((R28.STORE / 'index.json').read_text())
    n_txt = idx['n_texts']
    rows = np.sort(np.random.RandomState(SEED).choice(n_txt, N, replace=False))
    assert r28['sample'].startswith(f'np.random.RandomState({SEED}).choice({n_txt}, {N}, replace=False)')
    sample_ids = np.asarray([idx['texts'][int(i)]['sample_id'] for i in rows])
    ann = json.load(open('/data/PDD/artelingo/artelingo_train.json'))
    captions = [ann[s]['caption'] for s in sample_ids]
    assert all(isinstance(t, str) for t in captions) and len(captions) == N
    texts = caption_texts(captions, kform)                                        # K*'s own caption form
    ids_sha = hashlib.sha256(json.dumps([str(s) for s in sample_ids]).encode()).hexdigest()
    paintings = C.annotation_paintings()[[int(s) for s in sample_ids]]
    emb = C.caption_rows(kstar, [int(s) for s in sample_ids], paintings)      # K*'s rows (width 1152); ID and painting checked
    assert emb.shape == (N, SIGLIP_WIDTH) and emb.dtype == np.float32

    device = torch.device('cuda')
    towers, paths = [], []
    for aspect in P.ASPECTS:                       # as r28: the three aspects' cap sides, one seed
        w = Path(a.weights_dir) / f'{aspect}__s{a.seed}.pt'
        cfg, rb = M.read_sidecar(w)
        assert cfg['aspect'] == aspect and rb['cap_source'] == kstar, (w, cfg['aspect'], rb['cap_source'], kstar)
        view = M.load_view(w, device, expect_store_sha256=C.store_hash(kstar))
        assert view['cap'].trunk[0].weight.shape[1] == SIGLIP_WIDTH
        towers.append(view['cap'])
        paths.append(str(w))

    sys.path.insert(0, str(P.REPO))
    from src.data import feature_extract as FE
    enc = FE.load_encoder('siglip_so400m')
    live = enc.encode_texts(texts[:BATCH], BATCH)
    assert live.shape == (BATCH, SIGLIP_WIDTH), live.shape                       # width check: SigLIP 1
    cos = (live * emb[:BATCH]).sum(1) / np.linalg.norm(live, axis=1) / np.linalg.norm(emb[:BATCH], axis=1)
    print('live vs stored cosine min', cos.min(), flush=True)
    sync, clock = torch.cuda.synchronize, time.perf_counter

    out = {'task': 'r29 Table 8 text-pass time', 'written': time.strftime('%F %H:%M'),
           'kstar': kstar, 'kstar_form': kform, 'cap_source': kstar,
           'siglip1_text_input': f'{kform} captions' + (f' ({P.PREFIX!r} + caption)' if kform == 'prefixed' else ''), 'gpu': torch.cuda.get_device_name(0),
           'torch': torch.__version__, 'batch': BATCH, 'warmup_batches': WARMUP, 'repeats': a.repeats, 'n_captions': N,
           'sample': r28['sample'], 'caption_sample_ids_sha256': ids_sha, 'same_captions_as': str(R28_RECORD),
           'siglip1_live_vs_stored_cos_min': float(cos.min()), 'gpu_processes_before_start': gpu_before,
           'caption_tower_weights': paths,
           'caption_tower_input': f'stored {kstar} rows of the same captions, fp32',
           'qwen3_text_pass_r28': {'ms_per_caption_mean_of_repeats': qwen_ms, 'source': str(R28_RECORD),
                                   'written': r28['written'], 'model': r28['model']}}
    discards = []

    def repeats(name, encode, items):
        recs = []
        for k in range(a.repeats):
            rec, d = PR.run_repeat(name, k, lambda: PR.model_repeat(name, encode, items, k, BATCH, WARMUP, sync, clock),
                                   PR.poll_gpu, own, wait=30.0)
            discards.extend(d)
            recs.append(R28.summary(rec))
            print(name, k, rec['ms_per_image'], flush=True)
        m = [r['ms_per_caption'] for r in recs]
        return {'repeats': recs, 'ms_per_caption_mean_of_repeats': float(np.mean(m)),
                'ms_per_caption_repeat_min': min(m), 'ms_per_caption_repeat_max': max(m)}

    out['siglip1_text_pass'] = repeats('siglip1_text', lambda b: enc.encode_texts(b, BATCH), texts)
    del enc
    torch.cuda.empty_cache()

    widths = []

    def enc_cap(xb):       # as r28: host-to-device copy of the batch, forward, copy back
        with torch.no_grad():
            t = torch.from_numpy(xb).to(device)
            outs = [m(t).cpu().numpy() for m in towers]
        widths.extend(o.shape[1] for o in outs)

    out['caption_tower_three_aspects'] = repeats('cap_tower', enc_cap, emb)
    assert set(widths) == {TOWER_WIDTH}, set(widths)                              # width check: caption tower
    out['output_widths'] = {'siglip1_text_pass': SIGLIP_WIDTH, 'caption_tower': TOWER_WIDTH}
    out['caption_tower_params_each'] = [sum(p.numel() for p in m.parameters()) for m in towers]
    out['discarded_repeats'] = [{k: v for k, v in d.items() if k != 'record'} for d in discards]
    out['gpu_processes_after_end'] = PR.poll_gpu()
    out['foreign_during_timing'] = bool(discards)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print('written', a.out, flush=True)


if __name__ == '__main__':
    main()
