"""Extract frozen-caption RoBERTa features with September's token pooling."""
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
from transformers import AutoModel, AutoTokenizer

import pc_data

MODEL_NAME = 'SamLowe/roberta-base-go_emotions'
REVISION = 'd75048347613a25d77de8cf6412eaae9fa7b26be'
SNAPSHOT = Path(os.environ['HF_HOME']) / 'hub/models--SamLowe--roberta-base-go_emotions/snapshots' / REVISION
BATCH_SIZE = 256
MAX_LENGTH = 64


def load_encoder(device):
    assert SNAPSHOT.is_dir(), SNAPSHOT
    tokenizer = AutoTokenizer.from_pretrained(str(SNAPSHOT), local_files_only=True)
    model = AutoModel.from_pretrained(str(SNAPSHOT), local_files_only=True).to(device)
    model.eval()
    return tokenizer, model


def encode(captions, tokenizer, model, device):
    assert captions and all(isinstance(text, str) and text.strip() for text in captions)
    assert not model.training
    with torch.no_grad():
        encoded = tokenizer(captions, padding=True, truncation=True,
                            max_length=MAX_LENGTH, return_tensors='pt').to(device)
        outputs = model(**encoded).last_hidden_state
        mask = encoded['attention_mask'].unsqueeze(-1)
        x = ((outputs * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1))
        x = x.cpu().numpy().astype(np.float32, copy=False)
    return pc_data.check_array(x, (len(captions), 768))


def smoke_check(outdir, texts, device):
    """Give each caption its own node in the unchanged September function."""
    import pc_roberta_reference as reference

    reference.MODEL_NAME = str(SNAPSHOT)
    names = [str(i) for i in range(len(texts))]
    path = outdir / 'smoke_captions.pending.json'
    path.write_text(json.dumps([dict(painting=name, caption=text)
                                for name, text in zip(names, texts)]))
    expected = reference.extract_affect_embedding_nodes(str(path), names, device, print)
    stored = np.load(outdir / 'txt.pending.npy', mmap_mode='r')
    difference = float(np.max(np.abs(stored - expected)))
    assert difference < 1e-5, f'September max abs diff {difference}'
    print(f'September max abs diff: {difference:.8g}', flush=True)
    path.unlink()
    return difference


def extract(s, outdir, device='cuda', smoke=False, count=None):
    outdir = Path(outdir)
    assert not (outdir / 'done.json').exists(), 'refusing to overwrite a finished store'
    outdir.mkdir(parents=True, exist_ok=True)
    assert not any(outdir.iterdir()), 'store folder must be new or empty'
    # Exclusive marker also prevents two writers starting in the same empty folder.
    marker = outdir / 'writing.pending'
    with marker.open('x'):
        pass
    started = time.perf_counter()
    all_texts = pc_data.caption_texts(s)
    assert len(all_texts) == len(s['sample_ids']) == 99999
    assert s['sample_ids'].dtype == np.int64
    assert s['painting_ids'].shape == s['sample_ids'].shape
    assert np.issubdtype(s['painting_ids'].dtype, np.str_)
    if smoke:
        count = 512 if count is None else count
        assert 0 < count <= 512
    else:
        assert count is None, '--count is only for smoke'
        count = len(all_texts)
    texts = all_texts[:count]
    tokenizer, model = load_encoder(device)
    pending = outdir / 'txt.pending.npy'
    x = np.lib.format.open_memmap(pending, mode='w+', dtype=np.float32, shape=(count, 768))
    encoding_started = time.perf_counter()
    for start in range(0, count, BATCH_SIZE):
        end = min(start + BATCH_SIZE, count)
        x[start:end] = encode(texts[start:end], tokenizer, model, device)
        print(f'RoBERTa: {end:,}/{count:,} captions', flush=True)
    x.flush()
    del x, model, tokenizer
    encoding_seconds = time.perf_counter() - encoding_started
    difference = smoke_check(outdir, texts, device) if smoke else None
    np.savez(outdir / 'ids.pending.npz', sample_ids=s['sample_ids'][:count],
             painting_ids=s['painting_ids'][:count])
    done = dict(model_id=MODEL_NAME, snapshot_revision=REVISION,
                pooling='attention-mask mean of last_hidden_state, including special tokens',
                max_length=MAX_LENGTH, batch=BATCH_SIZE, count=count,
                seconds=time.perf_counter() - started, sample_sha256=pc_data.fingerprint(s),
                smoke=smoke, captions_per_second=count / encoding_seconds)
    if smoke:
        done['september_max_abs_diff'] = difference
    (outdir / 'done.pending.json').write_text(json.dumps(done, indent=2, allow_nan=False) + '\n')
    pending.replace(outdir / 'txt.npy')
    (outdir / 'ids.pending.npz').replace(outdir / 'ids.npz')
    marker.unlink()
    (outdir / 'done.pending.json').replace(outdir / 'done.json')
    print(f'{count / encoding_seconds:.2f} captions/second; wrote {outdir}', flush=True)
    return done


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', type=Path, default=pc_data.ROBERTA)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--count', type=int, help='smoke caption count (default 512; CPU check: 64)')
    args = parser.parse_args()
    assert not args.smoke or args.outdir != pc_data.ROBERTA, 'smoke needs a separate --outdir'
    extract(pc_data.load_frozen(), args.outdir, args.device, args.smoke, args.count)


if __name__ == '__main__':
    main()
