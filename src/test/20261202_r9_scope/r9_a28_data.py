"""ArtELingo-28 loader and pins (ticket 16; DECISION_RULE.md section 5 items 1 and 4, section 3.3 "Pins", section 6
item 1; constitution C6).

load_a28(backbone) puts a backbone's ArtELingo-28 store into the A28 row space: the rows of
results/read_set_a28.parquet in its order (position k = 0 to 74,568, one per read row, before the empty-output drops).
The store holds one image per read painting and one caption feature per KEPT row in `row_id` order
(index.json["texts"][j] = {row_id, painting, img}). The loader joins caption j to the k whose `row_id` it names, and
each row to its painting's image by painting name; both joins are asserted, never assumed. Dropped rows get NaN in
both feature arrays. Float32.

Fields of the result (n = rows of the read set): backbone, n, languages (sort order, as ticket 17's languages_of),
language (l = 1 to L per row), row_id, painting (names), paintings (sorted unique names), painting_index
(np.unique(painting, return_inverse=True)[1], the bootstrap code cl of the index "ArtELingo-28 episodes and C13"),
emotion / style / genre (int64, -1 unknown), held, dropped, kept (= ~dropped), img, txt (n, dim) float32.

pins: write_pins / check_pins, the pinned files of rule section 3.3 "Pins": per translation shard
`translations.parquet` and `coverage.json`; per backbone the store's `img.npy`, `txt.npy` and `index.json` (the join
key of C6); `value_sets_a28.json`, `language_flags_a28.json`, `read_set_a28.parquet`. Keys in the file:
translations/<shard>/<file>, store/<backbone>/<file>, results/<file>; each {"path": absolute, "sha256": hex}.
The runner (run_r9_a28_pins.py) writes the file; every read runner calls check_pins (rule section 6 item 1).
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r9_backbone as B  # noqa: E402  (before anything that imports src)
import r9_common as R  # noqa: E402
import r9_a28_episodes as AE  # noqa: E402  (the language order)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

STORE_ROOT = Path("/data/SSD2/pre_extract/artelingo28")
SHARD_GLOB = "nllb200*"
SHARD_FILES = ("translations.parquet", "coverage.json")
STORE_FILES = ("img.npy", "txt.npy", "index.json")
RESULT_FILES = ("value_sets_a28.json", "language_flags_a28.json", "read_set_a28.parquet")
TABLE_COLUMNS = AE.TABLE_COLUMNS
PINS_PATH = HERE / "pins_a28.json"
INDEX_KEYS = {"dataset", "backbone", "smoke", "dtype", "dim", "n_images", "n_texts", "images", "texts"}


def _require(cond, msg, exc=AssertionError):
    if not cond:
        raise exc(msg)


# ---------------------------------------------------------------- where the results live

def results_dir() -> Path:
    """This round's results folder: HERE/results when it holds the read set, otherwise the main checkout's (the
    stage-1 results are untracked and exist there; the same pattern as r6_common)."""
    mine = HERE / "results"
    if (mine / "read_set_a28.parquet").exists():
        return mine
    return R.MAIN / HERE.relative_to(R.CHECKOUT) / "results"


# ---------------------------------------------------------------- the loader

def load_read_set(path=None) -> pd.DataFrame:
    """results/read_set_a28.parquet with its columns and order asserted."""
    return pd.read_parquet(Path(path) if path else results_dir() / "read_set_a28.parquet")


def _check_table(tab):
    _require(list(tab.columns) == list(TABLE_COLUMNS), f"read set columns {list(tab.columns)}")
    _require(len(tab) > 0, "empty read set")
    row_id = tab["row_id"].to_numpy()
    _require((np.diff(row_id) > 0).all(), "read set: row_id is not strictly increasing")


def load_a28(backbone, store_root=STORE_ROOT, read_set=None) -> SimpleNamespace:
    """The backbone's ArtELingo-28 store in the A28 row space (see the module docstring).

    ``read_set``: a DataFrame, a path, or None (results/read_set_a28.parquet)."""
    _require(backbone in B.BACKBONES, f"unknown backbone: {backbone}")
    tab = read_set if isinstance(read_set, pd.DataFrame) else load_read_set(read_set)
    _check_table(tab)
    n = len(tab)
    row_id = tab["row_id"].to_numpy(dtype=np.int64)
    painting = tab["painting"].to_numpy().astype(str)
    dropped = tab["dropped"].to_numpy(dtype=bool)
    kept = ~dropped
    kept_pos = np.flatnonzero(kept)

    d = Path(store_root) / backbone
    index = json.loads((d / "index.json").read_text())
    _require(INDEX_KEYS <= set(index), f"{backbone}: index.json keys {sorted(index)}")
    dim = B.DIMENSIONS[backbone]
    _require((index["dataset"], index["backbone"], index["smoke"], index["dtype"], index["dim"]) ==
             ("artelingo28", backbone, False, "float16", dim),
             f"{backbone}: index.json header {[index[k] for k in ('dataset', 'backbone', 'smoke', 'dtype', 'dim')]}")
    img_store = np.load(d / "img.npy", allow_pickle=False)
    txt_store = np.load(d / "txt.npy", allow_pickle=False)
    _require(img_store.dtype == txt_store.dtype == np.float16, f"{backbone}: stores are not float16")
    images, texts = index["images"], index["texts"]
    _require(img_store.shape == (len(images), dim) == (index["n_images"], dim),
             f"{backbone}: img.npy {img_store.shape} for {len(images)} images, dim {dim}")
    _require(txt_store.shape == (len(texts), dim) == (index["n_texts"], dim),
             f"{backbone}: txt.npy {txt_store.shape} for {len(texts)} captions, dim {dim}")
    _require(len(texts) == len(kept_pos),
             f"{backbone}: {len(texts)} caption rows for {len(kept_pos)} kept rows of the read set")
    _require(np.isfinite(img_store).all() and np.isfinite(txt_store).all(), f"{backbone}: a stored value is not finite")

    # row_id join: caption j is the j-th kept row of the read set
    _require(all(set(t) == {"row_id", "painting", "img"} for t in texts), f"{backbone}: caption entry keys")
    store_row_id = np.asarray([t["row_id"] for t in texts], dtype=np.int64)
    _require(np.array_equal(store_row_id, row_id[kept_pos]),
             f"{backbone}: the store's row_id order differs from the kept rows of the read set")
    # painting join: the caption's painting is the read row's, and its image entry is that painting's image
    store_painting = np.asarray([t["painting"] for t in texts], dtype=str)
    _require(np.array_equal(store_painting, painting[kept_pos]),
             f"{backbone}: a caption's painting differs from its read row's painting")
    image_name = [im["painting"] for im in images]
    _require(len(set(image_name)) == len(image_name), f"{backbone}: two images for one painting")
    _require(set(image_name) == set(painting.tolist()),
             f"{backbone}: the store's paintings differ from the read set's (painting names)")
    point = np.asarray([t["img"] for t in texts], dtype=np.int64)
    _require(((0 <= point) & (point < len(images))).all(), f"{backbone}: a caption points outside the images")
    _require(np.array_equal(np.asarray(image_name, dtype=str)[point], store_painting),
             f"{backbone}: a caption points at another painting's image (painting join)")

    image_of = {p: i for i, p in enumerate(image_name)}
    row_image = np.asarray([image_of[p] for p in painting], dtype=np.int64)
    img = np.full((n, dim), np.nan, dtype=np.float32)
    txt = np.full((n, dim), np.nan, dtype=np.float32)
    img[kept_pos] = img_store[row_image[kept_pos]]
    txt[kept_pos] = txt_store

    languages = AE.languages_of(tab["language"].to_numpy())
    lang = tab["language"].to_numpy().astype(str)
    language = np.searchsorted(np.asarray(languages), lang).astype(np.int64) + 1
    _require((np.asarray(languages)[language - 1] == lang).all(), "language index")
    paintings, painting_index = np.unique(painting, return_inverse=True)
    out = SimpleNamespace(
        backbone=backbone, n=n, languages=languages, language=language, row_id=row_id, painting=painting,
        paintings=paintings, painting_index=painting_index.astype(np.int64),
        emotion=tab["emotion"].to_numpy(dtype=np.int64), style=tab["style"].to_numpy(dtype=np.int64),
        genre=tab["genre"].to_numpy(dtype=np.int64), held=tab["held"].to_numpy(dtype=bool),
        dropped=dropped, kept=kept, img=img, txt=txt)
    check_a28_space(out)
    return out


def check_a28_space(a):
    """Features finite on every kept row and NaN on every dropped row (every entry), counts equal to the table's."""
    n = a.n
    _require(a.img.shape == a.txt.shape and a.img.shape[0] == n and a.img.dtype == a.txt.dtype == np.float32,
             "feature arrays: shape or dtype")
    for name in ("row_id", "painting", "painting_index", "language", "emotion", "style", "genre", "held", "dropped",
                 "kept"):
        _require(len(getattr(a, name)) == n, f"{name} has {len(getattr(a, name))} rows, not {n}")
    _require((a.kept == ~a.dropped).all(), "kept is not ~dropped")
    for name in ("img", "txt"):
        x = getattr(a, name)
        finite = np.isfinite(x)
        _require(finite[a.kept].all(), f"{name}: a non-finite value on a kept row")
        _require(not finite[a.dropped].any(), f"{name}: a finite value on a dropped row")


# ---------------------------------------------------------------- pins (rule section 3.3 "Pins", section 6 item 1)

def _entry(path) -> dict:
    path = Path(path)
    _require(path.is_file(), f"pinned file is missing: {path}")
    return {"path": str(path), "sha256": R.sha256_file(path)}


def write_pins(out, store_root=STORE_ROOT, results=None, backbones=B.BACKBONES) -> dict:
    """Hash every pinned file and write ``out`` (refuses to overwrite; a missing file or store fails before any
    write). Returns the dict written."""
    out = Path(out)
    _require(not out.exists(), f"{out} exists: pins are written once")
    store_root = Path(store_root)
    results = Path(results) if results else results_dir()
    _require(set(backbones) <= set(B.BACKBONES) and len(set(backbones)) == len(backbones) > 0,
             f"backbones {backbones}")
    shards = sorted(p for p in store_root.glob(SHARD_GLOB) if p.is_dir())
    _require(shards, f"no translation shard {SHARD_GLOB} under {store_root}")
    files = {}
    for shard in shards:
        for f in SHARD_FILES:
            files[f"translations/{shard.name}/{f}"] = _entry(shard / f)
    for b in backbones:
        for f in STORE_FILES:
            files[f"store/{b}/{f}"] = _entry(store_root / b / f)
    for f in RESULT_FILES:
        files[f"results/{f}"] = _entry(results / f)
    pins = {"written": R.amsterdam_now(), "backbones": list(backbones), "files": files}
    out.write_text(json.dumps(pins, indent=1, allow_nan=False) + "\n")
    _require(json.loads(out.read_text()) == pins, "pins file did not read back")
    return pins


def check_pins(path=PINS_PATH, backbones=None) -> dict:
    """Rule section 6 item 1: every file the pins file lists exists and has its SHA-256 (SystemExit otherwise).

    The file must list both shard files of at least one shard, the three files of every backbone it names, and the
    three results files; ``backbones`` (default: the ones the file names) must all be covered.
    -> {key: sha256}."""
    pins = json.loads(Path(path).read_text())
    _require(isinstance(pins, dict) and isinstance(pins.get("files"), dict) and isinstance(pins.get("backbones"), list),
             f"{path}: not a pins file", SystemExit)
    files = pins["files"]
    named = list(pins["backbones"]) if backbones is None else list(backbones)
    uncovered = [b for b in named if b not in pins["backbones"]]
    _require(not uncovered, f"{path}: no pins for backbone {uncovered}", SystemExit)
    shards = {k.split("/")[1] for k in files if k.startswith("translations/")}
    need = {f"results/{f}" for f in RESULT_FILES}
    need |= {f"translations/{s}/{f}" for s in shards for f in SHARD_FILES}
    need |= {f"store/{b}/{f}" for b in pins["backbones"] for f in STORE_FILES}
    _require(shards, f"{path}: no translation shard is pinned", SystemExit)
    absent = sorted(need - set(files))
    _require(not absent, f"{path}: missing pins {absent}", SystemExit)
    got = {}
    for key, ent in files.items():
        sha, p = ent.get("sha256"), Path(ent.get("path", ""))
        _require(isinstance(sha, str) and len(sha) == 64 and all(c in "0123456789abcdef" for c in sha),
                 f"{key}: invalid SHA-256 pin", SystemExit)
        _require(p.is_file(), f"{key}: pinned file is missing: {p}", SystemExit)
        got[key] = R.sha256_file(p)
        _require(got[key] == sha, f"{key}: SHA-256 {got[key]} differs from {sha}", SystemExit)  # guard:pins_sha
    return got
