"""r23 text extraction and store checks (ticket 03; spec D9, D12, D17, gate (i), §8 truncation).

Jobs (each resumable by `done.json`; the stores follow tcl_data's layout, so writer and reader agree):
    art_teacher                         three held ArtELingo stores (emotion_focus, style_word, genre_word), batch 128
    mm_plain                            MMArt test, five kinds, default instruction, batch 64
    mm_teacher --kind K --split S       MMArt kind K, split val or test, three instructions, batch 64
    roberta                             held captions, RoBERTa go_emotions, max length 64, float32, 768-d
    gate_i                              spec §3.5 (i); writes OUT/stage0/gate_i.json (needs the GPU for parity)
TCL_STORES puts new stores under a scratch folder; TCL_SMOKE=N keeps the first N paintings (smoke stores are marked
`smoke: true`, and a smoke run refuses to write without TCL_STORES). Texts are always the frame's, in frame order.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

import tcl_paths as P
import tcl_data as D
import tx_data
from tx_core import release, write_json
from pav_prompts import PROMPTS
from ct2_data import QWEN_REVISION

QWEN_LIMIT, ROBERTA_LIMIT = 512, 64
PARITY_ROWS, PARITY_BAR = 200, 0.999
ROBERTA_ID = 'SamLowe/roberta-base-go_emotions'


def guard_smoke():
    assert not P.SMOKE or P.STORE_ROOT is not None, 'smoke needs TCL_STORES on a scratch folder'


def load_qwen(device):
    from src.data.feature_extract import load_encoder
    encoder = load_encoder('qwen3vl_emb_2b', device, PROMPTS['default'])
    assert encoder.m.config._commit_hash == QWEN_REVISION, 'Qwen revision differs'
    return encoder


def count_truncated(encoder, texts, instruction, batch=1024):
    """Texts whose tokenized prompt (the encoder's own template, with this instruction) is longer than 512 tokens, so
    `encode_texts` cuts them. The same tokenizer and prompt as `Qwen3VLEmb.encode_texts`, without the cut."""
    encoder.instruction = PROMPTS[instruction]
    n, longest = 0, 0
    for i in range(0, len(texts), batch):
        prompts = [encoder._prompt([{'type': 'text', 'text': t}]) for t in texts[i:i + batch]]
        lengths = [len(x) for x in encoder.p.tokenizer(prompts, truncation=False)['input_ids']]
        n += sum(x > QWEN_LIMIT for x in lengths)
        longest = max(longest, max(lengths))
    return n, longest


def completed(path, instruction):
    done = Path(path) / 'done.json'
    if not done.exists():
        return False
    meta = tx_data.read_json(done)
    assert meta['instruction'] == instruction, f'instruction mismatch: {done}'
    if meta['smoke']:
        return False
    assert (Path(path) / 'txt.npy').exists() and 'n_truncated' in meta
    return True


def encode_qwen_store(encoder, texts, path, instruction_name, batch, ids=None, keys=None):
    path = Path(path)
    instruction = PROMPTS[instruction_name]
    if completed(path, instruction):
        print(f'skip complete: {path}', flush=True)
        return
    started = time.monotonic()
    n_truncated, longest = count_truncated(encoder, texts, instruction_name)
    encoder.instruction = instruction
    x = tx_data.unit(encoder.encode_texts(list(texts), batch_size=batch))
    assert x.shape == (len(texts), 2048)
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / 'txt.npy', x.astype(np.float16))
    if ids is not None:
        np.savez_compressed(path / 'ids.npz', sample_ids=np.asarray(ids, dtype=np.int64))
    if keys is not None:
        write_json(path / 'keys.json', list(keys))
    write_json(path / 'done.json', dict(
        instruction=instruction, model_id='Qwen/Qwen3-VL-Embedding-2B', model_revision=QWEN_REVISION,
        count=len(x), shape=list(x.shape), dtype='float16', seconds=time.monotonic() - started, batch_size=batch,
        smoke=bool(P.SMOKE), n_truncated=int(n_truncated), longest_tokens=int(longest), token_limit=QWEN_LIMIT))
    print(f'wrote {path} n={len(x)} n_truncated={n_truncated}', flush=True)


# ------------------------------------------------------------------------------------------------ jobs
def art_teacher(device):
    f = D.frame('art_held', P.CAPTION)
    todo = [a for a in P.ASPECTS if not completed(D.art_teacher_path('held', a), PROMPTS[P.INSTRUCTIONS[a]])]
    if not todo:
        return
    encoder = load_qwen(device)
    for a in todo:
        encode_qwen_store(encoder, f['text'], D.art_teacher_path('held', a), P.INSTRUCTIONS[a], 128,
                          ids=f['sample_id'])
    del encoder
    release(device)


def mm_stores(set_name, kinds, instructions):
    """(kind, instruction, split, path) of the MMArt stores a job writes."""
    split = D.SPLIT[set_name]
    return [(k, i, split, D.mm_store_path(k, i, split)) for k in kinds for i in instructions]


def run_mm(set_name, kinds, instructions, device):
    stores = mm_stores(set_name, kinds, instructions)
    todo = [s for s in stores if not completed(s[3], PROMPTS[s[1]])]
    if not todo:
        return
    encoder = load_qwen(device)
    for kind, instruction, split, path in todo:
        f = D.frame(set_name, kind)
        encode_qwen_store(encoder, f['text'], path, instruction, 64, keys=f['names'])
    del encoder
    release(device)


def mm_plain(device):
    run_mm('mm_test', P.KINDS, ('default',), device)


def mm_teacher(kind, split, device):
    assert kind in P.KINDS and split in ('val', 'test')
    run_mm('mm_' + split, (kind,), tuple(P.INSTRUCTIONS.values()), device)


def roberta(device):
    sys.path.insert(0, str(P.MAIN / 'src/test/20261018_percept_core'))
    import pc_roberta
    path = D.store_base(P.ART_ROBERTA_STORE)
    if completed(path, None):
        print(f'skip complete: {path}', flush=True)
        return
    f = D.frame('art_held', P.CAPTION)
    texts = f['text']
    started = time.monotonic()
    tokenizer, model = pc_roberta.load_encoder(device)
    lengths = [len(x) for x in tokenizer(texts, truncation=False)['input_ids']]
    n_truncated = int(sum(x > ROBERTA_LIMIT for x in lengths))
    batch = pc_roberta.BATCH_SIZE
    x = np.concatenate([pc_roberta.encode(texts[i:i + batch], tokenizer, model, device)
                        for i in range(0, len(texts), batch)])
    assert x.shape == (len(texts), 768) and x.dtype == np.float32
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / 'txt.npy', x)
    np.savez_compressed(path / 'ids.npz', sample_ids=np.asarray(f['sample_id'], dtype=np.int64))
    write_json(path / 'done.json', dict(
        instruction=None, model_id=ROBERTA_ID, model_revision=pc_roberta.REVISION,
        pooling='attention-mask mean of last_hidden_state, including special tokens', max_length=pc_roberta.MAX_LENGTH,
        count=len(x), shape=list(x.shape), dtype='float32', seconds=time.monotonic() - started, batch_size=batch,
        smoke=bool(P.SMOKE), n_truncated=n_truncated, longest_tokens=int(max(lengths)), token_limit=ROBERTA_LIMIT))
    print(f'wrote {path} n={len(x)} n_truncated={n_truncated}', flush=True)
    del model
    release(device)


# ------------------------------------------------------------------------------------------------ gate (i)
def stores_to_check():
    """(key, set, kind, instruction name or None, row, path, id_kind). Key: set/kind/instruction/row."""
    rows = []
    for a in P.ASPECTS:
        rows.append(('art_held', P.CAPTION, P.INSTRUCTIONS[a], 'teacher', D.art_teacher_path('held', a), 'ids'))
    rows.append(('art_held', P.CAPTION, None, 'roberta', D.store_base(P.ART_ROBERTA_STORE), 'ids'))
    for k in P.KINDS:
        rows.append(('mm_test', k, 'default', 'plain', D.mm_store_path(k, 'default', 'test'), 'keys'))
    for s in ('mm_val', 'mm_test'):
        for k in P.KINDS:
            for i in P.INSTRUCTIONS.values():
                rows.append((s, k, i, 'teacher', D.mm_store_path(k, i, D.SPLIT[s]), 'keys'))
    return [(f'{s}/{k}/{i or "none"}/{r}', s, k, i, r, p, idk) for s, k, i, r, p, idk in rows]


def static_checks(s, k, i, r, path, id_kind):
    """Count, finite, unit norm, key join, manifest instruction, smoke flag, n_truncated. Returns (ok, why)."""
    path = Path(path)
    if not (path / 'done.json').exists():
        return False, 'store not written'
    done = tx_data.read_json(path / 'done.json')
    f = D.frame(s, k)
    ids = f['sample_id'].tolist() if id_kind == 'ids' else f['names']
    if bool(done['smoke']) != bool(P.SMOKE):
        return False, 'smoke flag differs from this run'
    want = None if i is None else PROMPTS[i]
    if done['instruction'] != want:
        return False, 'manifest instruction differs'
    if not isinstance(done.get('n_truncated'), int):
        return False, 'n_truncated absent'
    x = np.load(path / 'txt.npy', mmap_mode='r', allow_pickle=False)
    width, dtype = (768, np.float32) if r == 'roberta' else (2048, np.float16)
    if x.ndim != 2 or x.shape[1] != width or x.dtype != dtype:
        return False, f'shape or dtype {x.shape} {x.dtype}'
    if x.shape[0] != len(ids) or done['count'] != len(ids) or list(done['shape']) != list(x.shape):
        return False, f'count {x.shape[0]} vs {len(ids)}'
    if id_kind == 'ids':
        with np.load(path / 'ids.npz', allow_pickle=False) as z:
            stored = z['sample_ids'].tolist()
    else:
        stored = tx_data.read_json(path / 'keys.json')
    if stored != list(ids):
        return False, 'ids differ from the frame (C6 join)'
    y = np.asarray(x, dtype=np.float32)
    if not np.isfinite(y).all():
        return False, 'non-finite'
    norm = np.linalg.norm(y, axis=1)
    if r == 'roberta':      # mean-pooled, unnormalised by design; tcl_data normalises on read
        return ((norm > 0).all(), 'zero row' if not (norm > 0).all() else '')
    if np.abs(norm - 1).max() > 1e-3:
        return False, f'norm off by {np.abs(norm - 1).max():.2e}'
    return True, ''


def parity_reference(s, k, i, r, path):
    """(texts, reference unit rows) of the 200-row re-encode: held teacher vs the validation store of the same
    instruction (r22); plain mm_test path vs r22's mmart_val; MMArt teacher against its own stored rows."""
    n = PARITY_ROWS
    if s == 'art_held':
        f = D.frame('art_val', P.CAPTION)
        aspect = next(a for a, v in P.INSTRUCTIONS.items() if v == i)
        ref = D.read_store(D.art_teacher_path('val', aspect), f['sample_id'][:n].tolist(), 'ids', PROMPTS[i], 2048,
                           np.float16)
        return f['text'][:n], ref
    if r == 'plain':
        f = D.frame('mm_val', k)
        ref = D.read_store(D.mm_store_path(k, 'default', 'val'), f['names'][:n], 'keys', PROMPTS['default'], 2048,
                           np.float16)
        return f['text'][:n], ref
    f = D.frame(s, k)
    ref = D.read_store(path, f['names'][:n], 'keys', PROMPTS[i], 2048, np.float16)
    return f['text'][:n], ref


def gate_i(device):
    guard_smoke()
    report, encoder = {}, None
    for key, s, k, i, r, path, id_kind in stores_to_check():
        ok, why = static_checks(s, k, i, r, path, id_kind)
        entry = dict(static=bool(ok), why=why, path=str(path))
        if ok and r != 'roberta':
            if encoder is None:
                encoder = load_qwen(device)
            texts, ref = parity_reference(s, k, i, r, path)
            encoder.instruction = PROMPTS[i]
            got = tx_data.unit(encoder.encode_texts(list(texts), batch_size=64))
            cos = np.sum(got * ref, axis=1)
            entry.update(parity_min_cosine=float(cos.min()), parity_mean_cosine=float(cos.mean()), parity_n=len(cos),
                         parity=bool(cos.min() >= PARITY_BAR))
        else:
            entry['parity'] = None if ok else False       # RoBERTa: same code as the s2 store, no reference row set
        entry['pass'] = bool(ok and entry['parity'] is not False)
        report[key] = entry
    passed = {key: v['pass'] for key, v in report.items()}
    out = P.OUT / 'stage0'
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'gate_i.json', dict(passed=passed, overall=bool(all(passed.values())), n_stores=len(passed),
                                         smoke=bool(P.SMOKE), parity_bar=PARITY_BAR, detail=report))
    print(f'gate_i: {sum(passed.values())}/{len(passed)} stores pass; wrote {out / "gate_i.json"}', flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('job', choices=('art_teacher', 'mm_plain', 'mm_teacher', 'roberta', 'gate_i'))
    p.add_argument('--kind')
    p.add_argument('--split', choices=('val', 'test'))
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    guard_smoke()
    if a.job == 'mm_teacher':
        mm_teacher(a.kind, a.split, a.device)
    else:
        {'art_teacher': art_teacher, 'mm_plain': mm_plain, 'roberta': roberta, 'gate_i': gate_i}[a.job](a.device)


if __name__ == '__main__':
    main()
