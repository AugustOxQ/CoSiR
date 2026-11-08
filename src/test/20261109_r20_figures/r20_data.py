"""r20 data door: the two paintings sets (ArtELingo held, MMArt test), their labels, class names, artists, image paths,
c* outputs (seed 0) and the plain Qwen image. Everything is joined by painting name or key, never by position across
stores. Rows of every array returned here follow `Set.keys`.

Older rounds' code is imported from the main checkout (read only): r10's `rb_part2_report.held_labels`, r15's `mt_data`.
"""
import hashlib
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np

ROUND = Path(__file__).resolve().parent
MAIN = Path('/project/CoSiR')
MAIN_TEST = MAIN / 'src/test'
OUT = Path(os.environ.get('R20_OUTPUTS', ROUND / 'outputs'))

ASPECTS = ('emotion', 'style', 'genre')
SETS = ('artelingo', 'mmart')
SEED = 0
SMOKE_ROWS = 1500

R10_HELD = Path('/data/SSD2/cosir_r10/20261027_recipe_sweep/outputs/full/held/k11=40')
R15_CSTAR = Path('/data/SSD2/cosir_r15/20261102_mmart_test/outputs/exports/new/qwen/cstar')
ART_PLAIN = Path('/data/SSD2/pre_extract/artelingo_full/qwen3vl_emb_2b/img.npy')
MM_PLAIN = Path('/data/SSD2/pre_extract/mmart/qwen3vl_emb_2b/img.npy')
MM_PLAIN_KEYS = Path('/data/SSD2/pre_extract/mmart/qwen3vl_emb_2b/keys.json')
ART_IMAGES = Path('/data/PDD/wikiart_proj/wikiart')
ANNOTATIONS = Path('/data/PDD/artelingo/artelingo_train.json')


def setup_paths():
    """Put the older rounds on sys.path (idempotent)."""
    for p in (MAIN / 'src/test/20261102_mmart_test', MAIN / 'src/test/20261027_recipe_sweep', str(MAIN)):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import rs_paths  # noqa: F401  (puts r10's older rounds on sys.path)


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_list(items):
    return hashlib.sha256('\n'.join(items).encode('utf-8')).hexdigest()


def unit(x):
    """float32 rows divided by their norm (floor 1e-12)."""
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    assert np.isfinite(x).all() and (n > 0).all(), 'non-finite or zero row'
    return (x / np.maximum(n, 1e-12)).astype(np.float32)


def norm_slug(s):
    """Lowercase; drop dots, apostrophes and accents; runs of other non-alphanumerics -> '-'; strip edge dashes."""
    s = unicodedata.normalize('NFKD', str(s))
    s = ''.join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"[.'’‘`]", '', s)
    s = re.sub(r'[^a-z0-9]+', '-', s)
    return s.strip('-')


class Set:
    """One paintings set: keys (names), labels per aspect (-1 = none), class names, artist slug (as stored), image
    paths, and the loaders of c* and plain rows (unit float32, rows in `keys` order)."""

    def __init__(self, name, keys, labels, class_names, artists, image_paths, c_loader, plain_loader, sources):
        self.name, self.keys, self.labels, self.class_names = name, list(keys), labels, class_names
        self.artists, self.image_paths = list(artists), list(image_paths)
        self._c, self._plain, self.sources = c_loader, plain_loader, sources
        n = len(self.keys)
        assert len(set(self.keys)) == n and all(len(v) == n for v in labels.values())
        assert len(self.artists) == len(self.image_paths) == n

    def __len__(self):
        return len(self.keys)

    def cstar(self, aspect):
        x = unit(self._c(aspect))
        assert x.shape[0] == len(self)
        return x

    def plain(self):
        x = unit(self._plain())
        assert x.shape[0] == len(self)
        return x

    def subset(self, pos):
        """A new Set of the rows at positions `pos` (smoke runs); loaders are cut after loading."""
        pos = np.asarray(pos, dtype=np.int64)
        return Set(self.name, [self.keys[i] for i in pos], {a: v[pos] for a, v in self.labels.items()},
                   self.class_names, [self.artists[i] for i in pos], [self.image_paths[i] for i in pos],
                   lambda a: np.asarray(self._c(a))[pos], lambda: np.asarray(self._plain())[pos], self.sources)

    def smoke(self, n=SMOKE_ROWS):
        """n evenly spaced rows (names are sorted, so this spreads over artists)."""
        return self.subset(np.unique(np.linspace(0, len(self) - 1, n).astype(np.int64)))

    def keys_sha256(self):
        return sha256_list(self.keys)


# ------------------------------------------------------------------------------------------------ ArtELingo held
def _artelingo_class_names():
    """Emotion (9) and style (27) names: the sorted distinct annotation values, as `encode_labels` codes them."""
    setup_paths()
    from src.data.wikiart_genre import GENRE_NAMES
    records = json.loads(ANNOTATIONS.read_text())
    emotion = sorted({r['emotion'] for r in records if r['emotion'] not in ('', None)})
    style = sorted({r['art_style'] for r in records if r['art_style'] not in ('', None)})
    assert len(emotion) == 9 and len(style) == 27 and len(GENRE_NAMES) == 10
    return dict(emotion=emotion, style=style, genre=list(GENRE_NAMES))


def _index_wikiart():
    """stem -> sorted list of 'Folder/stem.jpg' (a stem can sit in two style folders)."""
    out = {}
    for d in sorted(os.listdir(ART_IMAGES)):
        if (ART_IMAGES / d).is_dir():
            for f in sorted(os.listdir(ART_IMAGES / d)):
                out.setdefault(os.path.splitext(f)[0], []).append(f'{d}/{f}')
    return out


def load_artelingo():
    setup_paths()
    import rb_part2_report
    import rb_encoders
    names, sha, labels = rb_part2_report.held_labels()
    assert len(names) == 12281 and names == sorted(names)
    class_names = _artelingo_class_names()
    for a in ASPECTS:
        assert labels[a].max() < len(class_names[a]), a
    stems = _index_wikiart()
    paths = []
    for i, n in enumerate(names):
        opts = stems[n]
        want = f'{class_names["style"][labels["style"][i]]}/{n}.jpg'
        paths.append(str(ART_IMAGES / (want if want in opts else opts[0])))
    artists = [n.split('_', 1)[0] for n in names]
    files = {a: R10_HELD / f'{a}__s{SEED}.npz' for a in ASPECTS}

    def c_loader(aspect):
        with np.load(files[aspect]) as z:
            return z['img']

    def plain_loader():
        return rb_encoders.full_images('qwen3vl_emb_2b', names)

    s = Set('artelingo', names, {a: np.asarray(labels[a]) for a in ASPECTS}, class_names, artists, paths,
            c_loader, plain_loader,
            dict(c_star={a: str(files[a]) for a in ASPECTS}, plain=str(ART_PLAIN), held_list_sha256=sha))
    return s


# ------------------------------------------------------------------------------------------------ MMArt test
def load_mmart():
    setup_paths()
    import mt_data
    import mt_paths
    keys = mt_data.keys('test')
    idx = mt_data._index().set_index('key').loc[keys]
    labels = {a: mt_data.labels(keys, a) for a in ASPECTS}
    class_names = {a: list(mt_paths.CLASSES[a]) for a in ASPECTS}
    files = {a: R15_CSTAR / f'{a}__s{SEED}.npz' for a in ASPECTS}

    def join(src_keys, x):
        pos = {k: i for i, k in enumerate(src_keys)}
        return np.asarray(x)[[pos[k] for k in keys]]

    def c_loader(aspect):
        with np.load(files[aspect], allow_pickle=False) as z:
            return join([str(k) for k in z['keys']], z['emb'])

    def plain_loader():
        src = json.loads(MM_PLAIN_KEYS.read_text())
        if isinstance(src, dict):
            src = src.get('keys', src)
        x = np.load(MM_PLAIN, mmap_mode='r')
        assert x.shape[0] == len(src), (x.shape, len(src))
        return join(src, np.asarray(x))

    return Set('mmart', keys, labels, class_names, idx.artist.tolist(), idx.image_path.tolist(), c_loader,
               plain_loader, dict(c_star={a: str(files[a]) for a in ASPECTS}, plain=str(MM_PLAIN)))


def load(name):
    return {'artelingo': load_artelingo, 'mmart': load_mmart}[name]()


def input_shas(s):
    """sha256 of every input file of a set's c* and plain rows (cached next to the outputs)."""
    cache = OUT / 'shas.json'
    book = json.loads(cache.read_text()) if cache.exists() else {}
    files = list(s.sources['c_star'].values()) + [s.sources['plain']]
    changed = False
    for f in files:
        st = os.stat(f)
        tag = f'{f}|{st.st_size}|{int(st.st_mtime)}'
        if tag not in book:
            book[tag] = sha256_file(f)
            changed = True
    if changed:
        OUT.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(book, indent=1))
    out = {}
    for f in files:
        st = os.stat(f)
        out[f] = book[f'{f}|{st.st_size}|{int(st.st_mtime)}']
    return out
