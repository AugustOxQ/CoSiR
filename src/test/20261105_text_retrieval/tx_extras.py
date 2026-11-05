"""SemArt pairs, MMArt perspectives and validation class-word queries."""
import argparse
from pathlib import Path

import numpy as np

import tx_paths as paths
import tx_data as data
import tx_model
from tx_core import record_ranking, release, row_arrays, write_json
from pav_prompts import PROMPTS
import rt_data

SEMART = Path('/data/SSD2/pre_extract/semart')


def store_text(path, keys, smoke=False):
    """Read only requested text rows, and reject incomplete extraction manifests."""
    path = Path(path)
    meta = data.read_json(path / 'done.json')
    assert meta['instruction'] == PROMPTS['default']
    assert smoke or not meta['smoke']
    stored = data.read_json(path / 'keys.json')
    rows = data.gather(stored, list(keys))
    x = np.load(path / 'txt.npy', mmap_mode='r', allow_pickle=False)
    assert x.dtype == np.float16 and x.shape == (len(stored), 2048)
    assert meta['count'] == len(stored)
    return data.unit(x[rows])


def semart_rows(name, smoke=0):
    path = SEMART / data.STORES[name]
    index = data.read_json(path / 'index.json')
    image_rows = [i for i, r in enumerate(index['images']) if r['split'] == 'val']
    names = [index['images'][i]['image_file'] for i in image_rows]
    assert len(names) == len(set(names)) == 1069
    assert set(names) == set(data.semart_csv()['IMAGE_FILE'])
    if smoke:
        image_rows, names = image_rows[:smoke], names[:smoke]
    lookup = {r: i for i, r in enumerate(image_rows)}
    text_rows = [i for i, r in enumerate(index['texts']) if r['img'] in lookup]
    tp = np.asarray([lookup[index['texts'][i]['img']] for i in text_rows], dtype=np.int64)
    assert np.array_equal(np.unique(tp), np.arange(len(names)))
    image = np.load(path / 'img.npy', mmap_mode='r', allow_pickle=False)
    text = np.load(path / 'txt.npy', mmap_mode='r', allow_pickle=False)
    assert image.dtype == text.dtype == np.float16
    return data.unit(text[text_rows]), data.unit(image[image_rows]), tp, names


def semart(out, device, smoke):
    records, pairs, notes = [], [], []
    for row in ('plain', 'plain_raw', 'cstar', 'cstar_free', 'clip_l14', 'clip_b32',
                'siglip2_so400m', 'siglip_so400m'):
        store = 'plain' if row in ('plain_raw', 'cstar', 'cstar_free') else row
        if not (SEMART / data.STORES[store] / 'index.json').exists():
            note = f'skip {row}: SemArt store absent'
            notes.append(note)
            print(note, flush=True)
            continue
        text, image, tp, names = semart_rows(store, smoke)
        ip = np.arange(len(image), dtype=np.int64)
        if row == 'plain_raw':
            path = out / 'text/semart_val_raw'
            if smoke and not (path / 'done.json').exists():
                notes.append('skip plain_raw: raw text extraction absent')
                continue
            text, tp = store_text(path, names, bool(smoke)), ip
        for aspect in (paths.ASPECTS if row in ('cstar', 'cstar_free') else (None,)):
            if aspect is None:
                t, i = text, image
            else:
                image_side, caption_side = tx_model.load_sides(aspect, device=device)
                t = (caption_side if row == 'cstar' else image_side)(text)
                i = image_side(image)
                del image_side, caption_side
            for direction in ('t2i', 'i2t'):
                q, g, qp, gp = (t, i, tp, ip) if direction == 't2i' else (i, t, ip, tp)
                for readout in ('r0', 'center'):
                    record_ranking(out, row, aspect, direction, readout, q, g, qp, gp, {},
                                   device, records, pairs, save=False)
            del t, i
            release(device)
        del text, image
        release(device)
    write_json(out / 'results/semart.json', dict(smoke=bool(smoke), pairs=pairs, notes=notes))


def mmart_labels(index):
    labels, codebooks = {}, {}
    for a in paths.ASPECTS:
        values = ['' if v is None or str(v).strip() in ('', '-1', '-1.0', 'nan') else str(v).strip()
                  for v in index[a]]
        names = sorted(set(values) - {''})
        assert names
        codes = {s: i for i, s in enumerate(names)}
        labels[a] = np.asarray([codes.get(s, -1) for s in values], dtype=np.int64)
        codebooks[a] = names
    return labels, codebooks


def mmart(out, device, smoke):
    index, texts = data.mmart_texts()
    labels, codebooks = mmart_labels(index)
    # Match label words by emotion name, irrespective of MMArt's local code order.
    own_codes = {n: i for i, n in enumerate(data.label_names('emotion'))}
    emotion = [own_codes.get(str(v).strip(), -1) for v in index['emotion']]
    leak = np.asarray([data.emotion_named(t, y) for t, y in zip(texts['e_emotional'], emotion)])
    leak_share = float(leak.mean())
    n = min(smoke, len(index)) if smoke else len(index)
    index = index.iloc[:n]
    labels = {a: y[:n] for a, y in labels.items()}
    image_rows = data.gather(rt_data.names('val'), index['painting'].tolist())
    images = data.image_rows('plain')[image_rows]
    pid = np.arange(n, dtype=np.int64)
    records, pairs, notes = [], [], []
    for row in ('plain', 'cstar', 'cstar_free'):
        for aspect in (paths.ASPECTS if row != 'plain' else (None,)):
            if aspect is not None:
                image_side, caption_side = tx_model.load_sides(aspect, device=device)
                image = image_side(images)
            else:
                image = images
            for col in data.MMART_COLUMNS:
                path = out / 'text/mmart_val' / col
                if smoke and not (path / 'done.json').exists():
                    notes.append(f'skip {row}/{aspect}/{col}: text extraction absent')
                    continue
                text = store_text(path, index['key'].tolist(), bool(smoke))
                if aspect is not None:
                    text = (caption_side if row == 'cstar' else image_side)(text)
                entries = {a: dict(q=y, g=y) for a, y in labels.items()}
                for readout in ('r0', 'center'):
                    record_ranking(out, row, aspect, 't2i', readout, text, image, pid, pid,
                                   entries, device, records, pairs, save=False, extra=dict(column=col))
                del text
            if aspect is not None:
                del image_side, caption_side
            del image
            release(device)
    write_json(out / 'results/mmart.json', dict(smoke=bool(smoke), records=records, pairs=pairs,
               codebooks=codebooks, emotional_leak_share=leak_share, emotional_leak_n=len(leak), notes=notes))


def word_rows(out, row, aspect, form, device):
    source = 'plain' if row in ('cstar', 'cstar_free') else row
    path = out / 'text/words' / f'{source}.npz'
    manifest = data.read_json(path.with_suffix('.done.json'))
    instruction = data.INSTRUCTIONS[aspect] if row == 'teacher' else 'default'
    assert manifest['instructions'][aspect] == PROMPTS[instruction]
    with np.load(path, allow_pickle=False) as z:
        q, codes = z[f'{aspect}__{form}'], z[f'{aspect}__codes']
    assert q.dtype == np.float32 and codes.dtype == np.int64 and len(q) == len(codes)
    if row in ('cstar', 'cstar_free'):
        image_side, caption_side = tx_model.load_sides(aspect, device=device)
        q = (caption_side if row == 'cstar' else image_side)(q)
        del image_side, caption_side
    return data.unit(q), codes


def words(out, device, smoke):
    c = data.val_captions()
    n = smoke or paths.N_PAINTINGS
    keep = c['pid'] < n
    cl, pl = data.caption_labels(), data.painting_labels()
    records, pairs, notes = [], [], []
    for row, row_aspect, text, image in row_arrays(out, device, smoke):
        if row == 'roberta':
            continue
        for a in (paths.ASPECTS if row_aspect is None else (row_aspect,)):
            for form in ('bare', 'template'):
                source = 'plain' if row in ('cstar', 'cstar_free') else row
                if smoke and not (out / 'text/words' / f'{source}.npz').exists():
                    notes.append(f'skip {row}/{a}/{form}: word extraction absent')
                    continue
                q, codes = word_rows(out, row, a, form, device)
                for direction, gallery, labels in [('t2i', image, pl[a][:n]), ('t2t', text, cl[a][keep])]:
                    for readout in ('r0', 'center'):
                        result = record_ranking(out, row, a, direction, readout, q, gallery, None, None,
                                   {a: dict(q=codes, g=labels)}, device, records, pairs, save=False,
                                   extra=dict(form=form))
                        records[-1]['class_ap'] = {str(int(code)): float(ap)
                                                   for code, ap in zip(codes, result['ap'][a])}
                        records[-1]['class_codes'] = codes.tolist()
        del text, image
        release(device)
    write_json(out / 'results/words.json', dict(smoke=bool(smoke), records=records, notes=notes))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--part', choices=('p', 'q', 'r', 'all'), default='all')
    p.add_argument('--device', default='cuda')
    p.add_argument('--smoke', type=int, default=0)
    p.add_argument('--out', type=Path, default=paths.OUT)
    args = p.parse_args()
    assert args.smoke >= 0
    for part, fn in [('p', semart), ('q', mmart), ('r', words)]:
        if args.part in (part, 'all'):
            fn(args.out, args.device, args.smoke)


if __name__ == '__main__':
    main()
