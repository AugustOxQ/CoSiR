"""Explicit encoder jobs; importing this module never loads an encoder."""
import argparse
import time
from pathlib import Path

import numpy as np

import tx_paths as paths
import tx_data as data
from tx_core import release, write_json
from pav_prompts import PROMPTS
from ct2_data import QWEN_REVISION


def load(name, device, instruction='default'):
    from src.data.feature_extract import load_encoder
    encoder = load_encoder(name, device, PROMPTS[instruction])
    if name == 'qwen3vl_emb_2b':
        assert encoder.m.config._commit_hash == QWEN_REVISION, 'Qwen revision differs'
    return encoder


def completed(path, instruction):
    done = Path(path) / 'done.json'
    if not done.exists():
        return False
    meta = data.read_json(done)
    assert meta['instruction'] == PROMPTS[instruction], f'instruction mismatch: {done}'
    if not meta['smoke']:
        assert (Path(path) / 'txt.npy').exists()
        return True
    return False


def encode_store(encoder, texts, path, instruction, smoke, batch, ids=None, keys=None):
    path = Path(path)
    if completed(path, instruction):
        print(f'skip complete: {path}', flush=True)
        return
    started = time.monotonic()
    texts = texts[:smoke] if smoke else texts
    encoder.instruction = PROMPTS[instruction]
    x = data.unit(encoder.encode_texts(list(texts), batch_size=batch))
    assert x.shape == (len(texts), 2048)
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / 'txt.npy', x.astype(np.float16))
    if ids is not None:
        np.savez_compressed(path / 'ids.npz', sample_ids=np.asarray(ids[:len(x)], dtype=np.int64))
    if keys is not None:
        write_json(path / 'keys.json', list(keys[:len(x)]))
    write_json(path / 'done.json', dict(instruction=PROMPTS[instruction],
               model_id='Qwen/Qwen3-VL-Embedding-2B', model_revision=QWEN_REVISION,
               count=len(x), shape=list(x.shape), dtype='float16', seconds=time.monotonic() - started,
               batch_size=batch, smoke=bool(smoke)))


def teacher(out, device, smoke):
    c = data.val_captions()
    keys = ('style_word', 'genre_word')
    if all(completed(out / 'text/artelingo_val' / key, key) for key in keys):
        return
    encoder = load('qwen3vl_emb_2b', device)
    gate = {}
    for key, name, aspect in [('emotion_focus', 'teacher', 'emotion'), ('default', 'plain', None)]:
        encoder.instruction = PROMPTS[key]
        got = data.unit(encoder.encode_texts(c['text'][:256].tolist(), batch_size=128))
        expected = data.text_rows(name, aspect, out)[:256]
        cosine = np.sum(got * expected, axis=1)
        gate[key] = dict(mean_cosine=float(cosine.mean()), min_cosine=float(cosine.min()), n=256)
        assert cosine.mean() > 0.999, (key, gate[key])
    write_json(out / 'text/artelingo_val/gate.json', gate)
    for key in keys:
        encode_store(encoder, c['text'], out / 'text/artelingo_val' / key, key, smoke, 128,
                     ids=c['sample_id'])
    del encoder
    release(device)


def mmart(out, device, smoke):
    index, texts = data.mmart_texts()
    encoder = load('qwen3vl_emb_2b', device)
    for col in data.MMART_COLUMNS:
        encode_store(encoder, texts[col], out / 'text/mmart_val' / col, 'default', smoke, 64,
                     keys=index['key'].tolist())
    del encoder
    release(device)


def semart(out, device, smoke):
    frame = data.semart_csv()
    encoder = load('qwen3vl_emb_2b', device)
    encode_store(encoder, frame['DESCRIPTION'].tolist(), out / 'text/semart_val_raw', 'default',
                 smoke, 64, keys=frame['IMAGE_FILE'].tolist())
    del encoder
    release(device)


def words(out, device, smoke):
    # The Qwen backbone is loaded once and shared across its four instructions.
    directory = out / 'text/words'
    directory.mkdir(parents=True, exist_ok=True)
    for name in ('qwen3vl_emb_2b', 'clip_l14', 'siglip2_so400m', 'siglip_so400m'):
        encoder = load(name, device)
        variants = ('plain', 'teacher') if name == 'qwen3vl_emb_2b' else (name,)
        for variant in variants:
            manifest = directory / f'{variant}.done.json'
            if smoke and manifest.exists() and not data.read_json(manifest)['smoke']:
                print(f'skip complete words: {variant}', flush=True)
                continue
            arrays, instructions = {}, {}
            for aspect in paths.ASPECTS:
                instruction = data.INSTRUCTIONS[aspect] if variant == 'teacher' else 'default'
                encoder.instruction = PROMPTS[instruction]
                instructions[aspect] = PROMPTS[instruction]
                terms = data.class_words(aspect)
                terms = terms[:smoke] if smoke else terms
                arrays[aspect + '__codes'] = np.asarray([r[0] for r in terms], dtype=np.int64)
                for form, column in [('bare', 1), ('template', 2)]:
                    arrays[f'{aspect}__{form}'] = data.unit(encoder.encode_texts(
                        [r[column] for r in terms], batch_size=128))
            np.savez_compressed(directory / f'{variant}.npz', **arrays)
            write_json(manifest, dict(encoder=name, instructions=instructions, smoke=bool(smoke),
                                     model_revision=QWEN_REVISION if name == 'qwen3vl_emb_2b' else None))
        del encoder
        release(device)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job', required=True, choices=('teacher', 'mmart', 'semart', 'words'))
    p.add_argument('--device', default='cuda')
    p.add_argument('--smoke', type=int, default=0)
    p.add_argument('--out', type=Path, default=paths.OUT)
    args = p.parse_args()
    assert args.smoke >= 0
    globals()[args.job](args.out, args.device, args.smoke)


if __name__ == '__main__':
    main()
