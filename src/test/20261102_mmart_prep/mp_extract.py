"""MMArt extraction and CPU gates. GPU work is only performed by extract commands.

Hashes match r14: SHA256 of newline-joined keys and contiguous array bytes
(without the .npy header). Smoke runs require a noncanonical MP_STORE_ROOT.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mp_stores as S

N_NEW = 16782
N_PARITY = 200
COSINE_MIN = .999
NEAR_COPY_CUT = .85
BATCH = dict(qwen_default=32, emotion_focus=32, style_word=32, genre_word=32,
             siglip2=32, siglip1=32, clip_l14=128)
HELD_LIST = Path('/project/CoSiR/src/test/20261020_claim_test/outputs/held/held_paintings.json')
NEARCOPY = Path('/data/SSD/mmart/mmart_nearcopy.parquet')


def outputs():
    return Path(os.environ.get('MP_OUTPUTS') or HERE)


def write_report(name, value):
    path = outputs() / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def keys_sha(keys):
    return hashlib.sha256('\n'.join(keys).encode()).hexdigest()


def array_sha(x):
    h = hashlib.sha256()
    for start in range(0, len(x), 2048):
        h.update(np.ascontiguousarray(x[start:start + 2048]).tobytes())
    return h.hexdigest()


def pick(keys, n):
    if not n:
        return list(keys)
    assert 0 < n <= len(keys), 'invalid subset size'
    ids = np.linspace(0, len(keys) - 1, n).round().astype(int)
    assert len(np.unique(ids)) == n
    return [keys[i] for i in ids]


def target_keys(which='new', index=None):
    table = S.read_index(index)
    if which == 'new':
        keys = table.loc[table.artelingo == 'none', 'key'].tolist()
        assert len(keys) == N_NEW, f'expected {N_NEW} new keys, got {len(keys)}'
        return keys
    assert which == 'parity'
    return pick(table.loc[table.artelingo == 'held', 'key'].tolist(), N_PARITY)


def smoke_guard(smoke):
    assert smoke >= 0, 'smoke must be positive'
    if smoke:
        assert os.environ.get('MP_STORE_ROOT') and S.store_root().resolve() != S.DEFAULT_ROOT.resolve(), \
            'smoke requires MP_STORE_ROOT pointing to a scratch root'


def r14():
    sys.path.insert(0, str(HERE.parent / '20261102_backbones_sweep'))
    import bs_extract
    return bs_extract


def held_identity(view, art_root=None):
    sys.path.insert(0, str(HERE.parent / '20261017_prompted_aspect_views'))
    from pav_prompts import PROMPTS
    meta = json.loads((S.reference_folder(view, art_root) / 'meta.json').read_text())
    assert meta['instruction'] == PROMPTS[view], f'{view}: held instruction differs from pav_prompts'
    return meta


def assert_image_instruction(enc, instruction):
    """Probe the actual Qwen image prompt path without running its model.

    feature_extract.Qwen3VLEmb.encode_images calls _prompt([{'type':'image'}]);
    _prompt reads self.instruction into the system message. Capture that message.
    """
    from src.data.feature_extract import Qwen3VLEmb
    assert isinstance(enc, Qwen3VLEmb)
    import inspect
    assert 'self._prompt([{"type": "image"}])' in inspect.getsource(type(enc).encode_images)
    processor = enc.p
    class Capture:
        def apply_chat_template(self, conversation, **kwargs):
            return conversation
    try:
        enc.p = Capture()
        conversation = enc._prompt([{'type': 'image'}])
        assert conversation[0]['content'][0]['text'] == instruction
        assert conversation[1]['content'] == [{'type': 'image'}]
    finally:
        enc.p = processor


def load_encoder(view, device='cuda'):
    bs = r14()
    base = 'qwen_default' if view in ('style_word', 'genre_word') else view
    enc, identity, instruction = bs.load_encoder(base, device)
    assert BATCH[view] == (32 if view in ('style_word', 'genre_word') else bs.BATCH[view])
    if view in S.TEACHERS or view == 'qwen_default':
        meta = held_identity(view if view in S.TEACHERS else 'emotion_focus')
        assert identity['model_revision'] == meta['model_revision'], 'Qwen revision differs from held store'
        assert identity['model_id'] == meta['model_id']
        if view in S.TEACHERS:
            instruction = meta['instruction']
            enc.instruction = instruction
        assert_image_instruction(enc, instruction)
    return enc, identity, instruction


def check_target(out):
    out = Path(out)
    pending = out.with_name(out.name + '.pending')
    assert not pending.exists(), f'pending extraction exists; inspect before retrying: {pending}'
    assert not out.exists(), f'sealed or existing folder will not be overwritten: {out}'
    return pending


def finite_unit(x):
    for start in range(0, len(x), 2048):
        block = np.asarray(x[start:start + 2048], dtype=np.float32)
        if not np.isfinite(block).all() or not np.all(np.abs(np.linalg.norm(block, axis=1) - 1) <= 1e-3):
            return False
    return True


def seal(pending, out, view, which, keys, x, identity, instruction, smoke, seconds):
    assert x.dtype == np.float16 and x.shape == (len(keys), S.WIDTH[view])
    assert len(set(keys)) == len(keys) and finite_unit(x), 'extraction finite/unit gate failed'
    assert not out.exists(), f'existing folder: {out}'
    meta = dict(view=view, set=which, **identity, instruction=instruction, n=len(keys), d=x.shape[1],
                keys_sha256=keys_sha(keys), img_sha256=array_sha(x), seconds=seconds, smoke=bool(smoke))
    (pending / 'keys.json').write_text(json.dumps(keys))
    (pending / 'meta.json').write_text(json.dumps(meta, indent=2, allow_nan=False) + '\n')
    pending.rename(out)
    return meta


class Images:
    def __init__(self, paths):
        self.paths = paths

    def load_image(self, i):
        from extract_features import _open_rgb
        return _open_rgb(self.paths[i])


def extract(view, which='new', smoke=0, *, index=None, device='cuda'):
    smoke_guard(smoke)
    out = S.folder(view, which)
    pending = check_target(out)
    table = S.read_index(index)
    keys = pick(target_keys(which, table), smoke)
    paths = table.set_index('key').loc[keys].image_path.tolist()
    assert all(Path(p).is_file() for p in paths), 'missing WikiArt images'
    enc, identity, instruction = load_encoder(view, device)
    pending.mkdir(parents=True)
    x = np.lib.format.open_memmap(pending / 'img.npy', mode='w+', dtype=np.float16,
                                  shape=(len(keys), S.WIDTH[view]))
    t0, done = time.monotonic(), 0
    for start, value in r14().encode_rows(enc, Images(paths), np.arange(len(keys)), BATCH[view], S.WIDTH[view]):
        assert start == done
        x[start:start + len(value)] = value
        done += len(value)
        if (start // BATCH[view] + 1) % 50 == 0:
            print(f'{which}/{view}: {done}/{len(keys)} {time.monotonic() - t0:.0f}s', flush=True)
    assert done == len(keys)
    x.flush()
    seconds = time.monotonic() - t0
    meta = seal(pending, out, view, which, keys, x, identity, instruction, smoke, seconds)
    print(f'{which}/{view}: sealed {len(keys)} images, {seconds:.1f}s, {1000 * seconds / len(keys):.1f} ms/image', flush=True)
    return meta


def extract_all(smoke=0):
    smoke_guard(smoke)
    for which in ('new', 'parity'):
        for view in S.VIEWS:
            out = S.folder(view, which)
            assert not out.with_name(out.name + '.pending').exists(), f'pending extraction exists: {out}.pending'
            if (out / 'meta.json').is_file():
                meta = json.loads((out / 'meta.json').read_text())
                assert meta['smoke'] == bool(smoke), f'smoke/full store mismatch: {out}'
                assert json.loads((out / 'keys.json').read_text()) == pick(target_keys(which), smoke), \
                    f'existing keys differ: {out}'
                print(f'{which}/{view}: sealed already, skipped', flush=True)
                continue
            extract(view, which, smoke)
    return 0


def parity(*, index=None, art_root=None):
    table = S.read_index(index).set_index('key')
    reports = {}
    for view in S.VIEWS:
        try:
            keys, actual = S.read_store(S.folder(view, 'parity'))
            meta = json.loads((S.folder(view, 'parity') / 'meta.json').read_text())
            full = target_keys('parity', table.reset_index())
            assert keys == (pick(full, len(keys)) if meta['smoke'] else full), 'wrong parity keys'
            names = table.loc[keys].painting.tolist()
            expected = S.join(S.reference_folder(view, art_root), names, view,
                              teacher=view in S.TEACHERS, reference=True)
            a = np.asarray(actual, dtype=np.float32)
            assert a.shape == expected.shape and np.isfinite(a).all()
            cos = np.sum(a * expected, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(expected, axis=1))
            assert np.isfinite(cos).all(), 'nonfinite cosine'
            reports[view] = dict(n=len(keys), min_cosine=float(cos.min()), mean_cosine=float(cos.mean()),
                                 **{'pass': bool((cos >= COSINE_MIN).all())})
        except (AssertionError, OSError, ValueError, KeyError) as error:
            reports[view] = {'pass': False, 'error': str(error)}
        print(f'parity {view}: {reports[view]}', flush=True)
    write_report('mp_parity.json', reports)
    return 0 if all(r['pass'] for r in reports.values()) else 1


def nearest_rows(query, reference, query_chunk=256, reference_chunk=4096):
    """Bounded float32 cosine matmul, normalising only temporary comparison rows."""
    assert len(query) and len(reference)
    maxima, indices = np.empty(len(query), np.float32), np.empty(len(query), np.int64)
    for start in range(0, len(query), query_chunk):
        q = np.asarray(query[start:start + query_chunk], np.float32)
        assert finite_unit(q)
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
        best, ids = np.full(len(q), -np.inf, np.float32), np.zeros(len(q), np.int64)
        for offset in range(0, len(reference), reference_chunk):
            ref = np.asarray(reference[offset:offset + reference_chunk], np.float32)
            assert finite_unit(ref)
            ref = ref / np.linalg.norm(ref, axis=1, keepdims=True)
            scores = q @ ref.T
            local = scores.argmax(axis=1)
            values = scores[np.arange(len(q)), local]
            improve = values > best
            best[improve], ids[improve] = values[improve], offset + local[improve]
        maxima[start:start + len(q)], indices[start:start + len(q)] = best, ids
    return maxima, indices


def nearcopy(*, index=None, art_root=None, held_list=None, parquet_path=None):
    import pandas as pd
    keys, query = S.read_store(S.folder('qwen_default'))
    meta = json.loads((S.folder('qwen_default') / 'meta.json').read_text())
    full = target_keys('new', index)
    assert keys == (pick(full, len(keys)) if meta['smoke'] else full), 'wrong new keys'
    names, reference = S.read_store(S.reference_folder('qwen_default', art_root), reference=True)
    held = set(json.loads(Path(held_list or HELD_LIST).read_text())['names'])
    if art_root is None and held_list is None:
        assert len(held) == 12281 and len(names) == 61402 and held <= set(names)
    rows = [i for i, name in enumerate(names) if name not in held]
    trainval = [names[i] for i in rows]
    maxima, ids = nearest_rows(query, reference[rows])
    frame = pd.DataFrame(dict(key=keys, max_cos=maxima, nearest_painting=[trainval[i] for i in ids]))
    hits = frame[frame.max_cos >= NEAR_COPY_CUT].sort_values('max_cos', ascending=False, kind='stable')
    report = dict(n=len(keys), counts={f'{cut:.2f}': int((maxima >= cut).sum()) for cut in (.85, .90, .95, .99)},
                  matches=hits.rename(columns={'max_cos': 'cosine'}).to_dict('records'))
    # A scratch extraction must never write the canonical parquet.
    dest = Path(parquet_path) if parquet_path is not None else \
        (outputs() / 'mmart_nearcopy.parquet' if os.environ.get('MP_STORE_ROOT') or meta['smoke'] else NEARCOPY)
    dest.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(dest, index=False)
    write_report('mp_nearcopy.json', report)
    print(f'nearcopy: {report["counts"]}', flush=True)
    return 0


def check_store(view, which, index=None):
    out = S.folder(view, which)
    try:
        keys, x = S.read_store(out)
        meta = json.loads((out / 'meta.json').read_text())
        full = target_keys(which, index)
        expected = pick(full, len(keys)) if meta['smoke'] else full
        assert keys == expected, 'keys differ from expected set/order'
        assert x.dtype == np.float16 and x.shape == (len(keys), S.WIDTH[view]), 'dtype/shape mismatch'
        assert meta['n'] == len(keys) and meta['d'] == x.shape[1], 'metadata shape mismatch'
        assert meta['view'] == view and meta['set'] == which, 'metadata target mismatch'
        assert finite_unit(x), 'nonfinite or nonunit rows'
        assert meta['keys_sha256'] == keys_sha(keys), 'keys hash mismatch'
        assert meta['img_sha256'] == array_sha(x), 'image hash mismatch'
        assert not out.with_name(out.name + '.pending').exists(), 'pending folder also exists'
        return dict(view=view, set=which, n=len(keys), ok=True)
    except (AssertionError, OSError, ValueError, KeyError) as error:
        return dict(view=view, set=which, ok=False, error=str(error))


def check(*, index=None):
    results = [check_store(view, which, index) for which in ('new', 'parity') for view in S.VIEWS]
    ok = all(r['ok'] for r in results)
    write_report('mp_stores_check.json', dict(ok=ok, stores=results))
    print(f'check: {sum(r["ok"] for r in results)}/{len(results)} passed', flush=True)
    return 0 if ok else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    one = sub.add_parser('extract')
    one.add_argument('--view', choices=S.VIEWS, required=True)
    one.add_argument('--set', dest='which', choices=('new', 'parity'), default='new')
    one.add_argument('--smoke', type=int, default=0)
    all_parser = sub.add_parser('extract-all')
    all_parser.add_argument('--smoke', type=int, default=0)
    for command in ('parity', 'nearcopy', 'check'):
        sub.add_parser(command)
    args = parser.parse_args(argv)
    if args.command == 'extract':
        extract(args.view, args.which, args.smoke)
        return 0
    if args.command == 'extract-all':
        return extract_all(args.smoke)
    return {'parity': parity, 'nearcopy': nearcopy, 'check': check}[args.command]()


if __name__ == '__main__':
    sys.exit(main())
