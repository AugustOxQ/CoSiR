"""Validation-only joins, stored unit rows and fixed label-word checks."""
import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

import tx_paths as paths
import rt_data
import rs_vh
from apc_data import paint_mean
from pav_prompts import PROMPTS
from src.data.wikiart_genre import GENRE_NAMES

FULL = rt_data.FULL_ROOT
INSTRUCTIONS = dict(emotion='emotion_focus', style='style_word', genre='genre_word')
STORES = dict(plain='qwen3vl_emb_2b', clip_l14='clip_l14',
              siglip2_so400m='siglip2_so400m', siglip_so400m='siglip_so400m', clip_b32='clip_b32')
STYLE_PATTERNS = {
    'Abstract_Expressionism': r'abstract expressionis\w*', 'Action_painting': r'action painting',
    'Analytical_Cubism': r'cubis\w*', 'Cubism': r'cubis\w*', 'Synthetic_Cubism': r'cubis\w*',
    'Art_Nouveau_Modern': r'art nouveau', 'Baroque': r'baroque', 'Color_Field_Painting': r'colou?r field',
    'Contemporary_Realism': r'realism|realist', 'New_Realism': r'realism|realist', 'Realism': r'realism|realist',
    'Early_Renaissance': r'renaissance', 'High_Renaissance': r'renaissance', 'Northern_Renaissance': r'renaissance',
    'Expressionism': r'expressionis\w*', 'Fauvism': r'fauvis\w*', 'Impressionism': r'impressionis\w*',
    'Post_Impressionism': r'impressionis\w*', 'Mannerism_Late_Renaissance': r'manneris\w*|renaissance',
    'Minimalism': r'minimalis\w*', 'Naive_Art_Primitivism': r'primitiv\w*|naive art',
    'Pointillism': r'pointillis\w*', 'Pop_Art': r'pop art', 'Rococo': r'rococo',
    'Romanticism': r'romanticis\w*', 'Symbolism': r'symbolism|symbolist', 'Ukiyo_e': r'ukiyo',
}
GENRE_PATTERNS = {
    'portrait': r'portrait\w*', 'landscape': r'landscape\w*', 'genre_painting': r'genre',
    'religious_painting': r'religio\w*', 'abstract_painting': r'abstract\w*', 'cityscape': r'cityscape\w*',
    'sketch_and_study': r'sketch\w*|study|studies', 'still_life': r'still[- ]life',
    'illustration': r'illustrat\w*', 'nude_painting': r'nude\w*|naked',
}


def read_json(path):
    return json.loads(Path(path).read_text())


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    assert x.ndim == 2 and np.isfinite(x).all()
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    assert (norm > 0).all()
    return (x / norm).astype(np.float32)


def gather(stored, wanted):
    assert len(set(stored)) == len(stored), 'duplicate store IDs'
    lookup = {v: i for i, v in enumerate(stored)}
    assert all(v in lookup for v in wanted), 'missing joined IDs'
    rows = np.asarray([lookup[v] for v in wanted], dtype=np.int64)
    assert [stored[i] for i in rows] == list(wanted)
    return rows


@lru_cache(maxsize=1)
def val_captions():
    ns = rt_data.names('val')
    lookup = {name: i for i, name in enumerate(ns)}
    index = read_json(FULL / STORES['plain'] / 'index.json')
    selected = [(i, r) for i, r in enumerate(index['texts']) if r['painting'] in lookup]
    selected.sort(key=lambda x: x[1]['sample_id'])
    assert len(selected) == paths.N_CAPTIONS
    ids = np.asarray([r['sample_id'] for _, r in selected], dtype=np.int64)
    assert (np.diff(ids) > 0).all()
    annotations = read_json(rs_vh.ANNOTATIONS)
    rows = [annotations[int(sid)] for sid in ids]
    assert all(ann['painting'] == r['painting'] for ann, (_, r) in zip(rows, selected))
    emotion_names = sorted(rs_vh.EMOTIONS)
    emotion = np.asarray([emotion_names.index(r['emotion']) for r in rows], dtype=np.int64)
    pid = np.asarray([lookup[r['painting']] for _, r in selected], dtype=np.int64)
    counts = np.zeros((paths.N_PAINTINGS, len(emotion_names)), dtype=np.int64)
    np.add.at(counts, (pid, emotion), 1)
    assert (counts.sum(1) > 0).all()
    assert np.array_equal(counts.argmax(1), rt_data.labels('val')['emotion'])
    styles = sorted({r['art_style'] for r in rows})
    assert styles == sorted(STYLE_PATTERNS), 'style codebook differs'
    assert np.array_equal(np.asarray([styles.index(r['art_style']) for r in rows]),
                          rt_data.labels('val')['style'][pid])
    return dict(sample_id=ids, store_row=np.asarray([i for i, _ in selected], dtype=np.int64),
                pid=pid, text=np.asarray([r['caption'] for r in rows]), emotion=emotion)


def painting_labels():
    return rt_data.labels('val')


def caption_labels():
    c, p = val_captions(), painting_labels()
    return {a: c['emotion'].copy() if a == 'emotion' else p[a][c['pid']] for a in paths.ASPECTS}


def label_names(aspect):
    if aspect == 'emotion':
        return sorted(rs_vh.EMOTIONS)
    if aspect == 'style':
        val_captions()  # checks the sorted codebook against annotations
        return sorted(STYLE_PATTERNS)
    assert aspect == 'genre'
    assert set(GENRE_NAMES) == set(GENRE_PATTERNS)
    return list(GENRE_NAMES)


def text_path(name, aspect=None, out=None):
    out = paths.OUT if out is None else Path(out)
    if name in STORES:
        return FULL / STORES[name]
    if name == 'roberta':
        return FULL / 'roberta_go_emotions_s2'
    assert name == 'teacher' and aspect in paths.ASPECTS
    if aspect == 'emotion':
        return FULL / 'qwen3vl_emb_2b_aspect_prompts/emotion_focus_s2txt'
    return out / 'text/artelingo_val' / INSTRUCTIONS[aspect]


def text_rows(name, aspect=None, out=None, raw=False):
    c = val_captions()
    path = text_path(name, aspect, out)
    x = np.load(path / 'txt.npy', mmap_mode='r', allow_pickle=False)
    if name in STORES:
        index = read_json(path / 'index.json')
        rows = gather([r['sample_id'] for r in index['texts']], c['sample_id'].tolist())
        ns = rt_data.names('val')
        assert all(index['texts'][i]['painting'] == ns[p]
                   for i, p in zip(rows, c['pid']))
    else:
        done = read_json(path / 'done.json')
        assert not done['smoke'], f'smoke text store: {path}'
        if name == 'teacher':
            assert done['instruction'] == PROMPTS[INSTRUCTIONS[aspect]]
        with np.load(path / 'ids.npz', allow_pickle=False) as z:
            rows = gather(z['sample_ids'].tolist(), c['sample_id'].tolist())
    assert x.ndim == 2 and x.dtype == (np.float32 if name == 'roberta' else np.float16)
    width = dict(plain=2048, teacher=2048, clip_l14=768, siglip2_so400m=1152,
                 siglip_so400m=1152, roberta=768)[name]
    assert x.shape[1] == width
    values = np.asarray(x[rows], dtype=np.float32)
    assert values.shape == (paths.N_CAPTIONS, width) and np.isfinite(values).all()
    return values if raw else unit(values)


def image_rows(name, aspect='emotion'):
    if name != 'siglip_so400m':
        arm = dict(plain='plain', clip_l14='clip', siglip2_so400m='siglip2', teacher='instructed')[name]
        return rt_data.embed(arm, 'val', aspect)
    path = FULL / STORES[name]
    index = read_json(path / 'index.json')
    rows = gather([r['painting'] for r in index['images']], rt_data.names('val'))
    x = np.load(path / 'img.npy', mmap_mode='r', allow_pickle=False)
    assert x.dtype == np.float16 and x.shape[1] == 1152
    return unit(x[rows])


def caption_mean(raw, pid, n):
    assert raw.dtype == np.float32 and raw.ndim == 2 and pid.shape == (len(raw),)
    assert np.array_equal(np.unique(pid), np.arange(n))
    return unit(paint_mean(raw, pid, n))


def emotion_named(text, code):
    import ewm_masks
    own = label_names('emotion')[int(code)] if code >= 0 else None
    return any(ewm_masks.EMOTION_NAMES[ewm_masks.WORD_LABEL[m.group().lower()]] == own
               for m in ewm_masks.PATTERNS['label'].finditer(text))


@lru_cache(maxsize=3)
def names_label(aspect):
    c, labels = val_captions(), caption_labels()[aspect]
    if aspect == 'emotion':
        return np.asarray([emotion_named(t, y) for t, y in zip(c['text'], labels)], dtype=bool)
    table = STYLE_PATTERNS if aspect == 'style' else GENRE_PATTERNS
    names = label_names(aspect)
    assert all(names[int(y)] in table for y in np.unique(labels) if y >= 0)
    patterns = [re.compile(r'\b(?:' + table[name] + r')\b', re.IGNORECASE) for name in names]
    return np.asarray([y >= 0 and bool(patterns[int(y)].search(t))
                       for t, y in zip(c['text'], labels)], dtype=bool)


def class_words(aspect):
    names = label_names(aspect)
    template = dict(emotion='a painting that conveys {}', style='a painting in the style of {}',
                    genre='an artwork of the genre {}')[aspect]
    words = [(i, n.replace('_', ' ') if n != 'Ukiyo_e' else 'Ukiyo-e')
             for i, n in enumerate(names) if n != 'something else']
    assert len(words) == dict(emotion=8, style=27, genre=10)[aspect]
    return [(i, word, template.format(word)) for i, word in words]


MMART_COLUMNS = ('e_narrative', 'e_formal', 'e_emotional', 'e_historical', 'e_unified')


def mmart_index():
    import pandas as pd
    frame = pd.read_parquet('/data/SSD/mmart/mmart_index.parquet',
                            columns=['key', 'painting', 'emotion', 'style', 'genre'],
                            filters=[('split', '==', 'val')])
    assert len(frame) == 5769 and frame['key'].is_unique and frame['painting'].is_unique
    assert set(frame['painting']) <= set(rt_data.names('val'))
    return frame.reset_index(drop=True)


def mmart_texts():
    import pandas as pd
    index = mmart_index()
    frame = pd.read_parquet('/data/SSD/mmart/raw/data/data.parquet',
                            columns=['image_id', *MMART_COLUMNS],
                            filters=[('image_id', 'in', index['key'].tolist())])
    assert frame['image_id'].is_unique
    rows = gather(frame['image_id'].tolist(), index['key'].tolist())
    frame = frame.iloc[rows].reset_index(drop=True)
    assert frame['image_id'].tolist() == index['key'].tolist()
    assert not frame[list(MMART_COLUMNS)].isna().any().any()
    return index, {col: frame[col].tolist() for col in MMART_COLUMNS}


def semart_csv():
    import pandas as pd
    frame = pd.read_csv('/data/SSD/semart/SemArt/semart_val.csv', sep='\t', encoding='latin1',
                        usecols=['IMAGE_FILE', 'DESCRIPTION'])
    assert len(frame) == 1069 and frame['IMAGE_FILE'].is_unique
    assert not frame.isna().any().any()
    return frame
