"""MMArt (HF `Shuaimyself/MMArt`, the dataset called PolyArt in earlier notes): one index table, CPU only.

    python mp_index.py build      # writes INDEX (parquet) and SUMMARY (json); needs the full download
    python mp_index.py summary    # prints SUMMARY

One row per MMArt painting, in data.parquet order (74,234 rows). The row order is the key order for every later
feature store (C6: joins by key, never by range(n)). Columns:
  key               MMArt image_id = WikiArt relative path, e.g. `Romanticism/delacroix_x.jpg`
  painting          the file stem (ArtELingo's `painting` field has the same form)
  image_path        the local WikiArt file (/data/PDD/wikiart_proj/wikiart/<key>); all 74,234 exist
  bytes_match       the parquet image bytes equal the local file's bytes (md5)
  md5               md5 of the image bytes
  style             MMArt's 20-class WikiArt style
  emotion           MMArt's dominant_emotion (ArtEmis-v2 majority, 9 classes incl. "something else"; "" = none)
  genre             WikiArt genre id (ArtGAN CSVs, src.data.wikiart_genre; -1 = unknown), genre_name
  artelingo         'held' (in ArtELingo's 12,281 held paintings), 'trainval' (in ArtELingo's other paintings, seen by
                    c* in training or used for picks), 'none' (not in our ArtELingo store)
  split             ArtELingo's painting split for shared paintings ('train', 'val', 'held'); 'new' otherwise. The
                    MMArt protocol (user, 2026-11-02): the same split as ArtELingo; 'new' paintings are test, 'val'
                    is the validation set (scored with MMArt labels), 'train' is not used (c* trained on it)
  artelingo_path_same   the ArtELingo image path equals key (False where the same stem sits in another style folder)
  exact_dup_trainval    'none' row whose image bytes equal an ArtELingo trainval image (md5)
  artelingo_emotion, artelingo_style   ArtELingo's majority caption emotion and art style of the painting
Captions (e_narrative, e_formal, e_emotional, e_historical, e_unified) stay in data.parquet; join by key.
"""
import argparse
import collections
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from src.data.wikiart_genre import GENRE_NAMES, load_wikiart_genre  # noqa: E402

RAW = Path('/data/SSD/mmart/raw/data')
META = RAW / 'data.parquet'
WIKIART = Path('/data/PDD/wikiart_proj/wikiart')
ARTELINGO_ANN = Path('/data/PDD/artelingo/artelingo_train.json')
MAIN = Path('/project/CoSiR')   # gitignored frozen inputs of older rounds live only in the main checkout
HELD_LIST = MAIN / 'src/test/20261020_claim_test/outputs/held/held_paintings.json'
RETRIEVAL_ROUND = MAIN / 'src/test/20261029_retrieval'   # rt_data.names('val'): the 6,152 ArtELingo val paintings
INDEX = Path('/data/SSD/mmart/mmart_index.parquet')
SUMMARY = Path(__file__).resolve().parent / 'mp_index_summary.json'
N = 74234


def md5(b):
    return hashlib.md5(b).hexdigest()


def parquet_md5s():
    """{key: md5 of the image bytes} over the 22 image shards."""
    out = {}
    for shard in sorted(RAW.glob('train-*.parquet')):
        f = pq.ParquetFile(shard)
        for g in range(f.num_row_groups):
            t = f.read_row_group(g, columns=['image_id', 'image'])
            for k, img in zip(t.column('image_id').to_pylist(), t.column('image').to_pylist()):
                assert k not in out, f'duplicate key {k}'
                out[k] = md5(img['bytes'])
    return out


def file_md5s(paths, workers=8):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda p: md5(Path(p).read_bytes()), paths))


def artelingo_tables():
    """painting -> image path, majority emotion, art style (from artelingo_train.json)."""
    rows = json.load(open(ARTELINGO_ANN))
    path, style, emo = {}, {}, collections.defaultdict(collections.Counter)
    for r in rows:
        p = r['painting']
        assert path.setdefault(p, r['image']) == r['image']
        style.setdefault(p, r['art_style'])
        emo[p][r['emotion']] += 1
    major = {p: sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0] for p, c in emo.items()}
    return path, major, style


def build():
    import pandas as pd
    df = pq.read_table(META, columns=['image_id', 'title', 'artist', 'style', 'date', 'dominant_emotion',
                                      'artemis_coverage']).to_pandas()
    assert len(df) == N and df.image_id.is_unique
    keys = df.image_id.tolist()
    stems = [Path(k).stem for k in keys]
    assert len(set(stems)) == N, 'painting stems must be unique'
    paths = [str(WIKIART / k) for k in keys]
    assert all(Path(p).is_file() for p in paths)

    f_md5 = file_md5s(paths)
    if any(RAW.glob('train-*.parquet')):
        pq_md5 = parquet_md5s()
        assert set(pq_md5) == set(keys), 'image shards and data.parquet hold different keys'
        bytes_match = [pq_md5[k] == m for k, m in zip(keys, f_md5)]
    else:   # shards deleted 2026-11-02 (user) after the first build matched all 74,234: carry that result
        prev = pq.read_table(INDEX, columns=['key', 'md5', 'bytes_match']).to_pandas()
        assert prev.key.tolist() == keys and prev.md5.tolist() == f_md5, 'local files changed since the first build'
        bytes_match = prev.bytes_match.tolist()
    art_path, art_emo, art_style = artelingo_tables()
    held = set(json.load(open(HELD_LIST))['names'])
    assert len(held) == 12281 and held <= set(art_path)
    trainval_md5 = set(file_md5s([str(WIKIART / art_path[p]) for p in art_path if p not in held]))
    genre = load_wikiart_genre(np.asarray(stems, dtype=object))

    group = ['held' if s in held else 'trainval' if s in art_path else 'none' for s in stems]
    sys.path.insert(0, str(RETRIEVAL_ROUND))
    import rt_data
    val = set(rt_data.names('val'))
    assert len(val) == 6152 and val <= set(art_path) and not val & held
    split = ['val' if s in val else {'held': 'held', 'trainval': 'train', 'none': 'new'}[g]
             for s, g in zip(stems, group)]
    out = pd.DataFrame(dict(
        key=keys, painting=stems, image_path=paths,
        bytes_match=bytes_match, md5=f_md5,
        style=df['style'], emotion=df.dominant_emotion.fillna('').astype(str),
        genre=genre, genre_name=[GENRE_NAMES[g] if g >= 0 else '' for g in genre],
        artelingo=group, split=split, artelingo_path_same=[art_path.get(s) == k for s, k in zip(stems, keys)],
        exact_dup_trainval=[g == 'none' and m in trainval_md5 for g, m in zip(group, f_md5)],
        artelingo_emotion=[art_emo.get(s, '') for s in stems], artelingo_style=[art_style.get(s, '') for s in stems],
        title=df.title, artist=df.artist, date=df.date, artemis_coverage=df.artemis_coverage))
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(INDEX, index=False)
    SUMMARY.write_text(json.dumps(summarize(out), indent=1) + '\n')
    print(json.dumps(summarize(out), indent=1))


def summarize(t):
    ov = t[t.artelingo != 'none']
    dup_md5 = t.md5.duplicated(keep=False)
    return dict(
        n=len(t), bytes_match=int(t.bytes_match.sum()), index_sha256=hashlib.sha256(INDEX.read_bytes()).hexdigest(),
        artelingo=t.artelingo.value_counts().to_dict(),
        split=t.split.value_counts().to_dict(),
        artelingo_path_differs=int((~ov.artelingo_path_same).sum()),
        none_exact_dup_of_trainval=int(t.exact_dup_trainval.sum()),
        md5_duplicates_within_mmart=int(dup_md5.sum()),
        emotion=t.emotion.replace('', '(none)').value_counts().to_dict(),
        style=t['style'].value_counts().to_dict(),
        genre_labelled={g: int((t[t.artelingo == g].genre >= 0).sum()) for g in ('held', 'trainval', 'none')},
        genre=t.genre_name.replace('', '(none)').value_counts().to_dict(),
        emotion_agrees_artelingo_majority=dict(
            n=int((ov.emotion != '').sum()),
            same=int(((ov.emotion == ov.artelingo_emotion) & (ov.emotion != '')).sum())),
        style_agrees_artelingo=dict(
            n=len(ov), same=int((ov['style'] == ov.artelingo_style.str.lower()).sum())),
    )


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['build', 'summary'])
    a = ap.parse_args()
    if a.cmd == 'build':
        build()
    else:
        print(SUMMARY.read_text())
