"""Ticket 01: r8_common (rule M1, M2, M10, section 9, section 10; constitution C6), on the real files.

Run from this folder (whole file well under 1 minute):
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  /root/miniconda3/envs/CoSiR/bin/python -m pytest -q -p no:cacheprovider --rootdir=. test_r8_common.py
The full section-10 hash pass (several GB) runs only with R8_FULL_INPUTS=1.
"""
import json
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r8_common as R  # noqa: E402

HEX = re.compile(r"\b[0-9a-f]{64}\b")
CHEAP = ("20261031_pseudo_partitions/results/partitions.npz",
         "20261117_reader_fix_csd/results/rb_halves.npz",
         "/data/SSD2/pre_extract/artelingo/style_csd_vitl/meta.json",
         "/data/SSD2/pre_extract/artelingo/style_csd_vitl/row_to_image.npy")


@pytest.fixture(scope="module")
def ctx():
    return R.ctx()


# ---------------------------------------------------------------- paths, rule, inputs

def test_paths_point_to_main_checkout():
    assert R.MAIN == Path("/project/CoSiR")
    assert R.RF == R.MAIN / "src/test/20261129_r8_grouping"
    assert R.RESULTS == R.RF / "results" and R.SMOKE == R.RESULTS / "smoke"
    assert R.RULE == R.RF / "DECISION_RULE.md" and R.RULE.is_file()


def test_assert_rule_returns_sha_and_refuses_a_changed_rule(monkeypatch):
    got = R.assert_rule()
    assert got == R.sha_file(R.RULE) == R.RULE_SHA256 and len(got) == 64
    monkeypatch.setattr(R, "RULE_SHA256", "0" * 64)
    with pytest.raises(SystemExit):
        R.assert_rule()


def _rule_rows():
    """Rule section 10 read row by row: {path as r8_common keys it: [sha256, ...]}. The rule writes `...` for the R7
    results folder, bare names for R7 code, a prose label for the HF files and ClustPy."""
    text = R.RULE.read_text()
    sec = text[text.index("## 10. Inputs"):text.index("## 11.")]
    rows = {}
    for line in sec.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if not line.startswith("|") or len(cells) < 2 or not HEX.search(cells[1]):
            continue
        hashes = HEX.findall(cells[1])
        toks = re.findall(r"`([^`]+)`", cells[0])
        label = cells[0]
        if label.startswith("`") and len(hashes) == len(toks):
            first_dir = os.path.dirname(toks[0]) if toks[0].startswith("/") else ""
            for t, h in zip(toks, hashes):
                if t.startswith("\u2026/"):
                    t = R.R7_NAME + "/" + t[2:]
                elif "/" not in t and first_dir:
                    t = first_dir + "/" + t
                rows[t] = [h]
        else:                                                    # HF / ClustPy rows, keyed by their own label
            rows[label] = hashes
    return rows


def test_inputs_table_is_rule_section_10_exactly():
    rows = _rule_rows()
    mine = {**{k: [v] for k, v in R.INPUTS.items()}, **{k: [v] for k, v in R.R7_CODE.items()}}
    labelled = {k: v for k, v in rows.items() if k.startswith("HF") or k.startswith("ClustPy")}
    plain = {k: v for k, v in rows.items() if k not in labelled}
    assert len(R.R7_CODE) == 10 and len(R.CLIP_WEIGHTS) == 2
    # every path-keyed row of the rule is one row of mine with the same hash, and the reverse
    hf = {k for k in mine if k.startswith(str(R._HF)) or "clustpy" in k}
    assert set(plain) == set(mine) - hf, (sorted(set(plain) - set(mine)), sorted(set(mine) - hf - set(plain)))
    for k, v in plain.items():
        assert mine[k] == v, k
    # the five labelled rows, each against its own path
    by_label = {"dinov2-large": "dinov2-large", "siglip2-so400m": "siglip2-so400m", "nrkmeans.py": "nrkmeans.py"}
    assert len(labelled) == 4
    for frag, keyfrag in by_label.items():
        (lab,) = [k for k in labelled if frag in k]
        (key,) = [k for k in hf if keyfrag in k]
        assert labelled[lab] == mine[key], lab
    (clip,) = [k for k in labelled if "clip-vit-base-patch32" in k]
    assert len(labelled[clip]) == 2
    for key, h in R.CLIP_WEIGHTS.items():                         # (bin, safetensors) in the rule's order
        assert h in labelled[clip]
        assert ("pytorch_model.bin" in key) == (labelled[clip].index(h) == 0), key
    assert [k for k in mine if k in R.CLIP_WEIGHTS] == []


def test_assert_inputs_subset_checks_and_caches(monkeypatch):
    got = R.assert_inputs(keys=CHEAP, r7=False)
    assert set(got) == set(CHEAP) and all(len(v) == 64 for v in got.values())
    calls = []
    real = R.sha_file
    monkeypatch.setattr(R, "sha_file", lambda p: calls.append(p) or real(p))
    assert R.assert_inputs(keys=CHEAP, r7=False) == got
    assert calls == [], "a verified file was hashed again"
    monkeypatch.setattr(R, "_CHECKED", {})                       # an empty cache hashes every file once
    assert R.assert_inputs(keys=CHEAP, r7=False) == got
    assert len(calls) == len(CHEAP)
    R.assert_inputs(keys=CHEAP, r7=False, fresh=True)           # fresh=True hashes again
    assert len(calls) == 2 * len(CHEAP)


def test_assert_inputs_stops_on_a_wrong_hash_or_a_missing_file(monkeypatch):
    key = CHEAP[2]
    monkeypatch.setattr(R, "_CHECKED", {})
    monkeypatch.setitem(R.INPUTS, key, "0" * 64)
    with pytest.raises(SystemExit):
        R.assert_inputs(keys=(key,), r7=False)
    monkeypatch.setitem(R.INPUTS, "nonexistent/file.npz", "0" * 64)
    with pytest.raises(SystemExit):
        R.assert_inputs(keys=("nonexistent/file.npz",), r7=False)
    with pytest.raises(KeyError):
        R.assert_inputs(keys=("not/in/the/table",), r7=False)


@pytest.mark.skipif(os.environ.get("R8_FULL_INPUTS") != "1", reason="hashes several GB; set R8_FULL_INPUTS=1")
def test_assert_inputs_full_table():
    got = R.assert_inputs(fresh=True)
    assert set(R.INPUTS) <= set(got) and set(R.R7_CODE) <= set(got)


def test_clip_weight_hashes_every_existing_file_and_the_record_helper(monkeypatch):
    existing = [k for k in R.CLIP_WEIGHTS if Path(k).is_file()]
    assert existing, "no CLIP weight file on this machine"
    calls = []
    real = R.sha_file
    monkeypatch.setattr(R, "sha_file", lambda p: calls.append(str(p)) or real(p))
    monkeypatch.setattr(R, "_CHECKED", {})
    got = R.assert_inputs(keys=(), r7=False, clip=True)
    assert sorted(calls) == sorted(existing), "every existing CLIP weight file is hashed, not the first one"
    assert got["clip_weight"] in existing and set(got["clip_weights"]) == set(existing)
    for k in existing:
        assert got["clip_weights"][k] == R.CLIP_WEIGHTS[k]
        rec = R.clip_weight_record(k)                                  # the path transformers resolved
        assert rec == {"path": k, "sha256": R.CLIP_WEIGHTS[k]}
        blob = Path(k).resolve()                                       # the blob behind the snapshot link
        assert blob.name == R.CLIP_WEIGHTS[k] and R.clip_weight_record(blob)["path"] == k
    with pytest.raises(SystemExit):
        R.clip_weight_record("/some/other/model.safetensors")
    # a file that exists but is not the rule's: stop
    monkeypatch.setitem(R.CLIP_WEIGHTS, existing[0], "0" * 64)
    monkeypatch.setattr(R, "_CHECKED", {})
    with pytest.raises(SystemExit):
        R.clip_weight_record(existing[0])


# ---------------------------------------------------------------- R7 module loader

def test_load_r7_returns_the_main_copy_once():
    m = R.load_r7("r7_heads")
    assert Path(m.__file__).resolve() == (R.R7D / "r7_heads.py").resolve()
    assert R.load_r7("r7_heads") is m and sys.modules["r7_heads"] is m
    assert R.load_r7("r7_common").RESULTS == R.R7D / "results"


def test_load_r7_refuses_unknown_names_and_changed_files(monkeypatch):
    with pytest.raises(KeyError):
        R.load_r7("r7_notamodule")
    with pytest.raises(KeyError):
        R.load_r7("run_step1")          # earlier rounds come through load_prior
    monkeypatch.setitem(R.R7_CODE, "r7_gate.py", "0" * 64)
    monkeypatch.delitem(sys.modules, "r7_gate", raising=False)
    monkeypatch.setattr(R, "_R7_LOADED", {})
    monkeypatch.setattr(R, "_R7_VERIFIED", {})      # independent of test order (the full-table test fills it)
    with pytest.raises(SystemExit):
        R.load_r7("r7_gate")


def test_load_r7_verifies_all_ten_modules_on_the_first_call(monkeypatch):
    # a module R7 code imports by bare name must be checked too: a changed r7_bundle.py stops a load of r7_heads
    monkeypatch.setitem(R.R7_CODE, "r7_bundle.py", "0" * 64)
    monkeypatch.setattr(R, "_R7_LOADED", {})
    monkeypatch.setattr(R, "_R7_VERIFIED", {})
    monkeypatch.delitem(sys.modules, "r7_heads", raising=False)
    with pytest.raises(SystemExit):
        R.load_r7("r7_heads")
    assert "r7_heads" not in sys.modules
    monkeypatch.undo()
    monkeypatch.setattr(R, "_R7_VERIFIED", {})
    R.load_r7("r7_heads")
    assert set(R._R7_VERIFIED) == set(R.R7_CODE)


def test_load_prior_gives_the_round_modules_through_r7_common():
    rs = R.load_prior("run_step1")
    assert Path(rs.__file__).name == "run_step1.py" and R.load_prior("run_step1") is rs


# ---------------------------------------------------------------- seeds, threads, time, writers

def test_seeds_threads_and_time(monkeypatch):
    assert R.DEV_SEED == 42 and R.SMOKE_SEEDS == (9001, 9002, 9003)
    assert re.fullmatch(r"\d{4}-\d\d-\d\d \d\d:\d\d", R.now_ams())
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.setenv("MKL_NUM_THREADS", "8")
    R.assert_threads()
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    with pytest.raises(AssertionError):
        R.assert_threads()


def test_write_once_helpers(tmp_path, monkeypatch):
    j = tmp_path / "a.json"
    rec = R.write_json_once(j, {"x": np.int64(3), "y": np.arange(2)})
    saved = json.loads(j.read_text())
    assert saved == rec and saved["x"] == 3 and saved["y"] == [0, 1]
    assert saved["rule_sha256"] == R.RULE_SHA256 and re.fullmatch(r"\d{4}-\d\d-\d\d \d\d:\d\d", saved["written"])
    with pytest.raises(SystemExit):
        R.write_json_once(j, {"x": 1})
    n = tmp_path / "sub" / "b.npz"
    R.write_npz_once(n, a=np.arange(3))
    assert np.array_equal(np.load(n)["a"], np.arange(3))
    with pytest.raises(SystemExit):
        R.write_npz_once(n, a=np.arange(3))
    p = tmp_path / "c.pkl"
    R.write_pickle_once(p, {"k": [1, 2]})
    assert pickle.loads(p.read_bytes()) == {"k": [1, 2]}
    with pytest.raises(SystemExit):
        R.write_pickle_once(p, 1)
    # the check-then-write race (up to 3 parallel readers): even if the exists-check passed, a file made in between
    # is never overwritten
    monkeypatch.setattr(R, "_prepare", lambda path: Path(path))
    for write, arg in ((R.write_json_once, {"x": 2}), (R.write_npz_once, None), (R.write_pickle_once, 2)):
        before = (j if write is R.write_json_once else n if write is R.write_npz_once else p).read_bytes()
        tgt = j if write is R.write_json_once else n if write is R.write_npz_once else p
        with pytest.raises(SystemExit):
            write(tgt, arg) if arg is not None else write(tgt, a=np.arange(9))
        assert tgt.read_bytes() == before
    monkeypatch.undo()
    # smoke outputs may be overwritten, nothing else may
    monkeypatch.setattr(R, "SMOKE", tmp_path / "smoke")
    s = R.SMOKE / "s.npz"
    R.write_npz_once(s, a=np.arange(2))
    R.write_npz_once(s, a=np.arange(5))
    assert len(np.load(s)["a"]) == 5
    f = R.fix_path(j)
    assert f.name == "a_fix1.json"


# ---------------------------------------------------------------- rows and paintings (M1)

def test_scorer_train_rows_and_painting_index(ctx):
    st, g = ctx.scorer_train, ctx.g_st
    assert st.dtype == np.int64 and st.shape == (183_694,) and (np.diff(st) > 0).all()
    assert g.dtype == np.int64 and g.shape == st.shape and ctx.n_paintings == 36_518
    assert np.array_equal(np.unique(g), np.arange(36_518))
    # independent derivation: the painting index is the ascending leakage-group code of the scorer-train rows
    assert np.array_equal(np.unique(ctx.split.groups[st], return_inverse=True)[1], g)
    # ... and it is round 1's paintings_sorted position (rb_halves.npz), not painting-id string order
    with np.load(R.path_of("20261117_reader_fix_csd/results/rb_halves.npz")) as z:
        assert np.array_equal(z["g_st"], g) and np.array_equal(z["scorer_train"], st)
        assert np.array_equal(z["paintings_sorted"], np.unique(ctx.split.groups[st]))


def test_painting_ids_and_first_rows(ctx):
    ids, first, g, st = ctx.painting_ids, ctx.first_rows, ctx.g_st, ctx.scorer_train
    assert ids.shape == first.shape == (36_518,) and ids.dtype.kind == "U" and first.dtype == np.int64
    assert len(set(ids.tolist())) == 36_518
    assert np.array_equal(g[first], np.arange(36_518))
    assert np.array_equal(np.unique(g, return_index=True)[1], first)        # the first occurrence, not any occurrence
    painting_of_row = np.asarray(ctx.data.paintings)[st]
    assert (painting_of_row == ids[g]).all()                                # every row carries its painting's id
    assert not np.array_equal(ids, np.sort(ids)), "p must not be painting-id string order (M1)"


def test_selection_rows(ctx):
    sel = ctx.selection
    assert sel.dtype == np.int64 and len(sel) == 32_413 and (np.diff(sel) > 0).all()
    assert sel[0] >= 0 and sel[-1] < len(ctx.split.groups)
    assert np.intersect1d(sel, ctx.scorer_train).size == 0
    assert np.intersect1d(sel, ctx.held).size == 0


def test_round_one_halves_per_painting(ctx):
    h, g = ctx.halves, ctx.g_st
    assert h.shape == (36_518,) and h.dtype == np.int64 and set(np.unique(h)) == {0, 1}
    assert np.bincount(h).tolist() == [18_259, 18_259]
    with np.load(R.path_of("20261117_reader_fix_csd/results/rb_halves.npz")) as z:
        assert (z["half_of_local_row"] == h[g]).all()                       # constant within painting, = round 1's
        sorted_ids = z["paintings_sorted"]
        assert np.array_equal(sorted_ids[h == 0], np.sort(z["paintings_half0"]))
        assert np.array_equal(sorted_ids[h == 1], np.sort(z["paintings_half1"]))
    d = ctx.halves_dict
    assert np.array_equal(d["scorer_train"], ctx.scorer_train) and np.array_equal(d["g_st"], g)


def test_content_grouping(ctx):
    c, g = ctx.content, ctx.g_st
    assert c.shape == (36_518,) and c.dtype == np.int64
    assert np.array_equal(np.unique(c), np.arange(64))
    sizes = np.bincount(c)
    assert sizes.min() >= 129 and (sizes.min(), sizes.max()) == (322, 992)
    with np.load(R.path_of("20261031_pseudo_partitions/results/partitions.npz")) as z:
        assert np.array_equal(z["image"], c[g])                             # constant within painting, = E2 per row
        assert np.array_equal(z["local_groups"], g)


# ---------------------------------------------------------------- CLIP vectors x (M1)

def test_unit_and_x_rows(ctx):
    rows = np.array([0, 5, 123_456, 308_722])
    x = R.x_rows(rows)
    assert x.dtype == np.float32 and x.shape == (4, 512)
    assert np.allclose(np.linalg.norm(x, axis=1), 1.0, atol=1e-6)
    raw = ctx.data.img_features[rows]
    assert np.abs(np.linalg.norm(raw, axis=1) - 1).max() > 1, "cached img_features are not unit: x must normalise"
    assert np.allclose(x, raw / np.linalg.norm(raw, axis=1, keepdims=True), atol=1e-6)
    assert np.array_equal(x, R.load_prior("run_checks").unit(raw))          # same recipe as the earlier rounds
    with pytest.raises(AssertionError):
        R.x_rows(np.array([308_723]))
    with pytest.raises(AssertionError):
        R.x_rows(np.array([-1]))


def test_x_paintings_is_the_first_scorer_train_row(ctx):
    xp = R.x_paintings()
    assert xp.shape == (36_518, 512) and xp.dtype == np.float32 and np.isfinite(xp).all()
    pick = np.array([0, 1, 777, 36_517])
    assert np.array_equal(xp[pick], R.x_rows(ctx.scorer_train[ctx.first_rows[pick]]))
    assert np.allclose(np.linalg.norm(xp, axis=1), 1.0, atol=1e-6)


# ---------------------------------------------------------------- images (M10, C6)

def test_row_to_image_is_checked_and_complete(ctx):
    r2i = R.row_to_image()
    assert r2i.dtype == np.int32 and r2i.shape == (308_723,) and r2i.min() == 0 and r2i.max() == 61_401
    meta = json.loads(Path("/data/SSD2/pre_extract/artelingo/style_csd_vitl/meta.json").read_text())
    assert R.load_r7("r7_groupings").sha_array(r2i) == meta["row_mapping"]["row_to_image_sha256"]
    assert R.row_to_image() is r2i


def test_nonheld_images(ctx):
    r2i, img = R.row_to_image(), R.nonheld_images()
    assert img.dtype == np.int32 and img.shape == (49_121,) and (np.diff(img) > 0).all()
    held = np.zeros(len(r2i), dtype=bool)
    held[ctx.held] = True
    assert np.array_equal(img, np.unique(r2i[~held]))
    assert ctx.held.dtype == np.int64 and len(ctx.held) == 61_744
    # equal to the 224 cache's image set (paintings.json), by image path
    meta = json.loads(Path("/data/SSD2/pre_extract/artelingo/style_csd_vitl/meta.json").read_text())
    paths = np.asarray(meta["images"]["image_paths"], dtype=object)
    cache = json.loads(Path("/data/SSD2/pre_extract/artelingo_clip224/paintings.json").read_text())["paintings"]
    assert set(paths[img].tolist()) == {v["image"] for v in cache.values()} and len(cache) == 49_121


def test_painting_and_selection_maps_into_nonheld_images(ctx):
    r2i, img = R.row_to_image(), R.nonheld_images()
    pi = R.painting_image()
    assert pi.dtype == np.int32 and pi.shape == (36_518,) and len(np.unique(pi)) == 36_518
    assert np.array_equal(pi, r2i[ctx.scorer_train[ctx.first_rows]])
    assert (r2i[ctx.scorer_train] == pi[ctx.g_st]).all()                    # one image per painting, all rows
    pos = R.painting_pos()
    assert pos.dtype == np.int64 and np.array_equal(img[pos], pi)
    si, sp = R.selection_image(), R.selection_pos()
    assert si.dtype == np.int32 and np.array_equal(si, r2i[ctx.selection])
    assert sp.dtype == np.int64 and np.array_equal(img[sp], si)


def test_x_images_is_x_of_the_first_nonheld_row(ctx):
    xi = R.x_images()
    assert xi.shape == (49_121, 512) and xi.dtype == np.float32 and np.isfinite(xi).all()
    assert np.allclose(np.linalg.norm(xi, axis=1), 1.0, atol=1e-6)
    r2i, img = R.row_to_image(), R.nonheld_images()
    held = np.zeros(len(r2i), dtype=bool)
    held[ctx.held] = True
    nonheld_rows = np.flatnonzero(~held)
    first = R.nonheld_first_rows()
    assert first.dtype == np.int64 and np.array_equal(r2i[first], img)
    for j in (0, 1, 1000, 49_120):                                          # brute-force first non-held row
        assert first[j] == nonheld_rows[np.argmax(r2i[nonheld_rows] == img[j])]
    assert np.array_equal(xi, R.x_rows(first))
    # a painting's two sources of x agree: its first scorer-train row and its image's first non-held row
    assert np.array_equal(xi[R.painting_pos()], R.x_paintings())


# ---------------------------------------------------------------- the 224 cache (M4 joins) and the CLIP feature dir

CACHE_JSON = Path("/data/SSD2/pre_extract/artelingo_clip224/paintings.json")
STYLE_META = Path("/data/SSD2/pre_extract/artelingo/style_csd_vitl/meta.json")


def test_cache224_rows_are_in_painting_id_order_so_the_join_is_needed(ctx):
    cache = json.loads(CACHE_JSON.read_text())
    assert cache["order"] == "painting id, sorted"
    keys = list(cache["paintings"])
    assert keys == sorted(keys) and [v["index"] for v in cache["paintings"].values()] == list(range(49_121))
    # a cache row is neither p (round 1's order) nor a position in nonheld_images()
    assert not np.array_equal(np.asarray(keys)[:5], ctx.painting_ids[:5])
    ci = R.cache224_index()
    assert not np.array_equal(ci, np.arange(49_121))


def test_cache224_index_joins_images_to_cache_rows(ctx):
    ci, nh = R.cache224_index(), R.nonheld_images()
    assert ci.dtype == np.int64 and ci.shape == (49_121,) and np.array_equal(np.sort(ci), np.arange(49_121))
    assert R.cache224_index() is ci
    cache = json.loads(CACHE_JSON.read_text())["paintings"]
    keys = list(cache)
    paths = json.loads(STYLE_META.read_text())["images"]["image_paths"]
    entries = list(cache.values())
    first = R.nonheld_first_rows()
    for j in list(range(0, 49_121, 997)) + [49_120]:                         # by image path ...
        assert entries[ci[j]]["image"] == paths[nh[j]], j
        assert keys[ci[j]] == str(ctx.data.paintings[first[j]]), j            # ... and by painting id
    # all of them, vectorised
    assert [entries[i]["image"] for i in ci.tolist()] == [paths[i] for i in nh.tolist()]
    assert [keys[i] for i in ci.tolist()] == [str(s) for s in np.asarray(ctx.data.paintings)[first]]


def test_painting_cache_row_gives_each_painting_its_cache_row(ctx):
    pcr = R.painting_cache_row()
    assert pcr.dtype == np.int64 and pcr.shape == (36_518,) and len(np.unique(pcr)) == 36_518
    keys = np.asarray(list(json.loads(CACHE_JSON.read_text())["paintings"]))
    assert (keys[pcr] == ctx.painting_ids).all()                              # through painting ids
    assert np.array_equal(pcr, R.cache224_index()[R.painting_pos()])          # and through images
    assert not np.array_equal(pcr, np.arange(36_518))
    assert R.painting_cache_row() is pcr


def test_cache224_join_asserts_the_cache_file_sha(monkeypatch):
    assert str(CACHE_JSON) in R.INPUTS
    monkeypatch.setattr(R, "_CHECKED", {})
    monkeypatch.setitem(R.INPUTS, str(CACHE_JSON), "0" * 64)
    monkeypatch.setattr(R, "_CACHE", {k: v for k, v in R._CACHE.items() if k != "cache224_index"})
    with pytest.raises(SystemExit):
        R.cache224_index()


def test_ctx_asserts_the_clip_feature_row_and_the_default_feature_dir(monkeypatch):
    key = "/data/SSD2/pre_extract/artelingo/features/metadata.json"
    assert key in R.INPUTS and key in R.assert_inputs(keys=(key,), r7=False)
    R.load_prior("run_step1")
    import importlib
    art = importlib.import_module("src.data.artelingo")
    assert Path(art.FEATURE_DIR) == Path("/data/SSD2/pre_extract/artelingo/features")
    # the build stops, before reading any feature, when the feature dir is redirected
    monkeypatch.setattr(art, "FEATURE_DIR", "/tmp/elsewhere")
    with pytest.raises(AssertionError):
        R._build_ctx()
    monkeypatch.undo()
    monkeypatch.setattr(R, "_CHECKED", {})
    monkeypatch.setitem(R.INPUTS, key, "0" * 64)                               # and when its SHA differs
    with pytest.raises(SystemExit):
        R._build_ctx()
