"""Tests of r9_a28_data (ticket 16): the ArtELingo-28 loader (rule section 5 item 4, constitution C6) and the pins
(rule section 3.3 "Pins", section 6 item 1) on a tiny synthetic store. No real file.

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python -m pytest -q -p no:cacheprovider \
        src/test/20261202_r9_scope/test_r9_a28_data.py
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r9_common as R  # noqa: E402,F401  (first: it puts MAIN's src in front and checks it)
import r9_backbone as B  # noqa: E402
import r9_a28_data as D  # noqa: E402

BACKBONE = "clip_b32"
DIM = B.DIMENSIONS[BACKBONE]

# 12 read rows, row_id is the CSV position (not 0..11), one row (position 5) dropped. Sort order of the languages:
# Hausa, Korean, arabic (upper case before lower case): l = 1, 2, 3.
ROWS = [  # (row_id, painting, language, emotion, style, genre, held, dropped)
    (10, "p_b", "Korean", 0, 1, 2, True, False),
    (11, "p_b", "Korean", 1, 1, 2, True, False),
    (12, "p_a", "Hausa", 2, 0, 1, False, False),
    (20, "p_a", "Hausa", -1, 0, 1, False, False),
    (21, "p_c", "arabic", 3, 2, 0, False, False),
    (22, "p_c", "arabic", 4, 2, 0, False, True),
    (30, "p_d", "Hausa", 5, 3, 3, True, False),
    (31, "p_d", "Korean", 6, 3, 3, True, False),
    (32, "p_a", "arabic", 7, 0, 1, False, False),
    (33, "p_b", "Hausa", 0, 1, 2, True, False),
    (40, "p_c", "Korean", 1, 2, 0, False, False),
    (41, "p_d", "arabic", 2, 3, 3, True, False),
]


def table(rows=ROWS):
    return pd.DataFrame(rows, columns=list(D.TABLE_COLUMNS))


def unit(rng, n):
    x = rng.normal(size=(n, DIM)).astype(np.float32)
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float16)


def write_store(root, tab, seed=0, backbone=BACKBONE, edit=None):
    """A store of the shared format under root/<backbone>: images sorted by painting name, captions = kept rows."""
    rng = np.random.default_rng(seed)
    names = sorted(set(tab.painting))
    img = unit(rng, len(names))
    kept = tab[~tab.dropped.to_numpy()]
    txt = unit(rng, len(kept))
    index = {"dataset": "artelingo28", "backbone": backbone, "smoke": False, "dtype": "float16", "dim": DIM,
             "n_images": len(names), "n_texts": len(kept),
             "images": [{"painting": p, "image": f"images/x/{p}.jpg"} for p in names],
             "texts": [{"row_id": int(r), "painting": p, "img": names.index(p)}
                       for r, p in zip(kept.row_id, kept.painting)]}
    if edit:
        index, img, txt = edit(copy.deepcopy(index), img.copy(), txt.copy())
    d = Path(root) / backbone
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "img.npy", img)
    np.save(d / "txt.npy", txt)
    (d / "index.json").write_text(json.dumps(index))
    (d / "progress.json").write_text(json.dumps({"n_img": len(img), "n_txt": len(txt)}))
    return d, img, txt


@pytest.fixture
def store(tmp_path):
    return write_store(tmp_path, table()), tmp_path


def load(tmp_path, tab=None):
    return D.load_a28(BACKBONE, store_root=tmp_path, read_set=table() if tab is None else tab)


# ---------------------------------------------------------------- the loader

def test_loader_places_features_by_row_id_and_painting(store):
    (d, img, txt), root = store
    a = load(root)
    n = len(ROWS)
    assert a.img.shape == a.txt.shape == (n, DIM) and a.img.dtype == a.txt.dtype == np.float32
    tab = table()
    drop = tab.dropped.to_numpy()
    assert (a.dropped == drop).all() and (a.kept == ~drop).all() and a.n == n
    assert np.isnan(a.img[drop]).all() and np.isnan(a.txt[drop]).all()
    assert np.isfinite(a.img[~drop]).all() and np.isfinite(a.txt[~drop]).all()
    names = sorted(set(tab.painting))
    kept_pos = np.flatnonzero(~drop)
    assert np.array_equal(a.txt[kept_pos], txt.astype(np.float32))               # by row_id order
    for k in kept_pos:                                                           # by painting
        assert np.array_equal(a.img[k], img[names.index(tab.painting[k])].astype(np.float32))
    # labels, row ids and painting index come straight from the table
    for col in ("emotion", "style", "genre"):
        assert np.array_equal(getattr(a, col), tab[col].to_numpy()) and getattr(a, col).dtype == np.int64
    assert np.array_equal(a.row_id, tab.row_id.to_numpy()) and (a.held == tab.held.to_numpy()).all()
    uniq, inv = np.unique(tab.painting.to_numpy().astype(str), return_inverse=True)
    assert np.array_equal(a.painting_index, inv) and list(a.paintings) == list(uniq)
    assert list(a.painting) == list(tab.painting)
    # language index l: 1-based position in the sort order, same function as ticket 17's
    assert a.languages == ["Hausa", "Korean", "arabic"]
    assert a.language.tolist() == [2, 2, 1, 1, 3, 3, 1, 2, 3, 1, 2, 3] and a.language.dtype == np.int64


def test_loader_without_dropped_rows(tmp_path):
    tab = table().assign(dropped=False)
    write_store(tmp_path, tab)
    a = load(tmp_path, tab)
    assert a.kept.all() and np.isfinite(a.img).all() and np.isfinite(a.txt).all()


def test_swapped_row_id_fails(tmp_path):
    def swap(index, img, txt):
        t = index["texts"]
        t[0]["row_id"], t[1]["row_id"] = t[1]["row_id"], t[0]["row_id"]
        return index, img, txt
    write_store(tmp_path, table(), edit=swap)
    with pytest.raises(AssertionError, match="row_id"):
        load(tmp_path)


def test_painting_mismatch_fails(tmp_path):
    def other(index, img, txt):
        index["texts"][3]["painting"] = "p_d"                   # the row's painting is p_a
        return index, img, txt
    write_store(tmp_path, table(), edit=other)
    with pytest.raises(AssertionError, match="painting"):
        load(tmp_path)


def test_caption_pointing_at_another_paintings_image_fails(tmp_path):
    def point(index, img, txt):
        index["texts"][3]["img"] = (index["texts"][3]["img"] + 1) % len(index["images"])
        return index, img, txt
    write_store(tmp_path, table(), edit=point)
    with pytest.raises(AssertionError, match="painting"):
        load(tmp_path)


def test_store_row_on_a_dropped_row_fails(tmp_path):
    """A caption feature for the dropped row (row_id 22) is a finite value on a dropped row."""
    def extra(index, img, txt):
        index["texts"].insert(5, {"row_id": 22, "painting": "p_c", "img": index["images"].index(
            {"painting": "p_c", "image": "images/x/p_c.jpg"})})
        index["n_texts"] += 1
        return index, img, np.insert(txt, 5, txt[0], axis=0)
    write_store(tmp_path, table(), edit=extra)
    with pytest.raises(AssertionError, match="row_id|rows"):
        load(tmp_path)


def test_check_space_rejects_finite_value_on_a_dropped_row(store):
    (d, img, txt), root = store
    a = load(root)
    D.check_a28_space(a)                                         # passes as loaded
    bad = copy.copy(a)
    bad.txt = a.txt.copy()
    bad.txt[5] = 0.5                                             # row 5 is dropped
    with pytest.raises(AssertionError, match="dropped"):
        D.check_a28_space(bad)
    bad = copy.copy(a)
    bad.img = a.img.copy()
    bad.img[5, 0] = 0.5                                          # one finite entry is enough
    with pytest.raises(AssertionError, match="dropped"):
        D.check_a28_space(bad)


def test_check_space_rejects_nonfinite_value_on_a_kept_row(store):
    (d, img, txt), root = store
    a = load(root)
    a.txt = a.txt.copy()
    a.txt[0, 3] = np.nan
    with pytest.raises(AssertionError, match="kept"):
        D.check_a28_space(a)


def test_nonfinite_stored_value_fails(tmp_path):
    def nan(index, img, txt):
        txt[2, 0] = np.nan
        return index, img, txt
    write_store(tmp_path, table(), edit=nan)
    with pytest.raises(AssertionError, match="finite"):
        load(tmp_path)


def test_wrong_count_dtype_backbone_dim_fail(tmp_path):
    for i, edit in enumerate((
            lambda ix, im, tx: (ix, im, tx[:-1]),                                   # caption count
            lambda ix, im, tx: (ix, im[:-1], tx),                                   # image count
            lambda ix, im, tx: (ix, im.astype(np.float32), tx),                     # dtype
            lambda ix, im, tx: ({**ix, "backbone": "qwen3vl_emb_2b"}, im, tx),      # backbone
            lambda ix, im, tx: ({**ix, "smoke": True}, im, tx),                     # smoke store
            lambda ix, im, tx: ({**ix, "dim": DIM + 1}, im, tx))):                  # dim
        root = tmp_path / str(i)
        write_store(root, table(), edit=edit)
        with pytest.raises(AssertionError):
            load(root)


def test_read_set_must_be_in_row_id_order_with_known_columns(store):
    (d, img, txt), root = store
    with pytest.raises(AssertionError, match="row_id"):
        load(root, table().iloc[::-1].reset_index(drop=True))
    with pytest.raises(AssertionError, match="columns"):
        load(root, table().drop(columns=["held"]))


def test_store_with_a_painting_not_in_the_read_set_fails(tmp_path):
    def extra(index, img, txt):
        index["images"].append({"painting": "p_z", "image": "images/x/p_z.jpg"})
        index["n_images"] += 1
        return index, np.vstack([img, img[:1]]), txt
    write_store(tmp_path, table(), edit=extra)
    with pytest.raises(AssertionError, match="painting"):
        load(tmp_path)


# ---------------------------------------------------------------- the pins

def make_tree(tmp_path, backbones=(BACKBONE,)):
    stores = tmp_path / "artelingo28"
    for i, b in enumerate(backbones):
        write_store(stores, table(), seed=i, backbone=b)
    for shard in ("nllb200_3.3b_en",):
        (stores / shard).mkdir(parents=True, exist_ok=True)
        (stores / shard / "translations.parquet").write_bytes(b"parquet " + shard.encode())
        (stores / shard / "coverage.json").write_text('{"x": 1}')
        (stores / shard / "meta.json").write_text('{"not": "pinned"}')
    res = tmp_path / "results"
    res.mkdir()
    for name in D.RESULT_FILES:
        (res / name).write_bytes(b"content of " + name.encode())
    return stores, res


def test_pins_cover_every_listed_file_and_check_passes(tmp_path):
    stores, res = make_tree(tmp_path, ("clip_b32", "qwen3vl_emb_2b"))
    out = tmp_path / "pins_a28.json"
    pins = D.write_pins(out, stores, res, backbones=("clip_b32", "qwen3vl_emb_2b"))
    on_disk = json.loads(out.read_text())
    assert on_disk == pins
    keys = set(pins["files"])
    want = {f"translations/nllb200_3.3b_en/{f}" for f in ("translations.parquet", "coverage.json")}
    want |= {f"store/{b}/{f}" for b in ("clip_b32", "qwen3vl_emb_2b") for f in ("img.npy", "txt.npy", "index.json")}
    want |= {f"results/{n}" for n in ("value_sets_a28.json", "language_flags_a28.json", "read_set_a28.parquet")}
    assert keys == want                                          # meta.json is not pinned
    assert all(len(v["sha256"]) == 64 for v in pins["files"].values())
    assert D.check_pins(out) == {k: v["sha256"] for k, v in pins["files"].items()}
    D.check_pins(out, backbones=("qwen3vl_emb_2b",))


def test_pins_refuse_a_missing_store(tmp_path):
    stores, res = make_tree(tmp_path, ("clip_b32",))
    with pytest.raises((AssertionError, FileNotFoundError)):
        D.write_pins(tmp_path / "pins_a28.json", stores, res, backbones=("clip_b32", "qwen3vl_emb_2b"))
    assert not (tmp_path / "pins_a28.json").exists()
    (stores / "clip_b32" / "txt.npy").unlink()
    with pytest.raises((AssertionError, FileNotFoundError)):
        D.write_pins(tmp_path / "pins_a28.json", stores, res, backbones=("clip_b32",))


def test_pins_refuse_a_missing_result_file_or_shard_file(tmp_path):
    stores, res = make_tree(tmp_path)
    (res / "value_sets_a28.json").unlink()
    with pytest.raises((AssertionError, FileNotFoundError)):
        D.write_pins(tmp_path / "pins_a28.json", stores, res, backbones=(BACKBONE,))
    (res / "value_sets_a28.json").write_text("{}")
    (stores / "nllb200_3.3b_en" / "coverage.json").unlink()
    with pytest.raises((AssertionError, FileNotFoundError)):
        D.write_pins(tmp_path / "pins_a28.json", stores, res, backbones=(BACKBONE,))


def test_pins_need_at_least_one_translation_shard(tmp_path):
    stores, res = make_tree(tmp_path)
    for f in (stores / "nllb200_3.3b_en").iterdir():
        f.unlink()
    (stores / "nllb200_3.3b_en").rmdir()
    with pytest.raises(AssertionError, match="shard"):
        D.write_pins(tmp_path / "pins_a28.json", stores, res, backbones=(BACKBONE,))


def test_check_pins_stops_on_a_changed_or_missing_file(tmp_path):
    stores, res = make_tree(tmp_path)
    out = tmp_path / "pins_a28.json"
    D.write_pins(out, stores, res, backbones=(BACKBONE,))
    for target in (stores / BACKBONE / "img.npy", stores / "nllb200_3.3b_en" / "coverage.json",
                   res / "read_set_a28.parquet"):
        old = target.read_bytes()
        target.write_bytes(old + b"x")
        with pytest.raises(SystemExit, match="SHA-256"):
            D.check_pins(out)
        target.write_bytes(old)
        D.check_pins(out)
    gone = stores / BACKBONE / "txt.npy"
    old = gone.read_bytes()
    gone.unlink()
    with pytest.raises(SystemExit, match="missing"):
        D.check_pins(out)
    gone.write_bytes(old)


def test_check_pins_requires_the_backbones_asked_for(tmp_path):
    stores, res = make_tree(tmp_path)
    out = tmp_path / "pins_a28.json"
    D.write_pins(out, stores, res, backbones=(BACKBONE,))
    with pytest.raises(SystemExit, match="qwen3vl_emb_2b"):
        D.check_pins(out, backbones=("qwen3vl_emb_2b",))


def test_check_pins_rejects_a_malformed_file(tmp_path):
    stores, res = make_tree(tmp_path)
    out = tmp_path / "pins_a28.json"
    pins = D.write_pins(out, stores, res, backbones=(BACKBONE,))
    for edit in (lambda p: p["files"].pop(f"store/{BACKBONE}/txt.npy"),          # a store file left out
                 lambda p: p["files"].pop("results/value_sets_a28.json"),
                 lambda p: p["files"].pop("translations/nllb200_3.3b_en/coverage.json"),
                 lambda p: p["files"][f"store/{BACKBONE}/img.npy"].update(sha256="zz")):
        bad = copy.deepcopy(pins)
        edit(bad)
        (tmp_path / "bad.json").write_text(json.dumps(bad))
        with pytest.raises(SystemExit):
            D.check_pins(tmp_path / "bad.json")


def test_write_pins_refuses_to_overwrite(tmp_path):
    stores, res = make_tree(tmp_path)
    out = tmp_path / "pins_a28.json"
    D.write_pins(out, stores, res, backbones=(BACKBONE,))
    with pytest.raises(AssertionError, match="exists"):
        D.write_pins(out, stores, res, backbones=(BACKBONE,))
