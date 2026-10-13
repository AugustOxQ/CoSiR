"""Tests of r9_recipe_groups (ticket 07; rule §3.2 item 1, §3.4, §11).

A small synthetic corpus: the made keys (image, caption, graph, bank) change with the features; the copied keys
(affect, local_groups) do not, and are byte-equal to CLIP's file. Written before the module.

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python -m pytest -q -p no:cacheprovider \
        src/test/20261202_r9_scope/test_r9_recipe_groups.py
"""
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r9_common as R  # noqa: E402  (first: MAIN and src)
import r9_recipe_groups as G  # noqa: E402

import numpy as np  # noqa: E402

P, PER, D, KK = 600, 4, 16, 6          # paintings, rows per painting, feature dim, clusters
N = P * PER
SMALL = dict(k=KK, min_group_rows=20)


def unit(x):
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def features(seed):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(KK, D))
    lab = rng.integers(0, KK, size=N)
    return unit(centres[lab] + 0.5 * rng.normal(size=(N, D)))


@pytest.fixture(scope="module")
def corpus():
    groups = np.repeat(np.arange(P), PER).astype(np.int64)
    scorer_train = np.arange(N, dtype=np.int64)
    rng = np.random.default_rng(7)
    frozen = {"affect": rng.integers(0, KK, size=N).astype(np.int64),
              "local_groups": np.unique(groups[scorer_train], return_inverse=True)[1].astype(np.int64)}
    return SimpleNs(groups=groups, scorer_train=scorer_train, frozen=frozen)


class SimpleNs:
    def __init__(self, **kw):
        self.__dict__.update(kw)


@pytest.fixture(scope="module")
def fits(corpus):
    out = {}
    for name, (si, st) in {"a": (1, 2), "b": (3, 4)}.items():
        img, txt = features(si), features(st)
        parts = G.fit_partitions(img, txt, corpus.groups, corpus.scorer_train, corpus.frozen, **SMALL)
        graph = G.fit_graph(img, txt, corpus.scorer_train)
        bank = G.fit_bank(parts, bank_size=60, min_paintings=5)
        out[name] = SimpleNs(parts=parts, graph=graph, bank=bank)
    return out


def test_keys_and_shapes(fits, corpus):
    p = fits["a"].parts
    assert tuple(p) == ("affect", "image", "caption", "local_groups")
    for key, arr in p.items():
        assert arr.dtype == np.int64 and arr.shape == (N,), key
    assert p["image"].min() == 0 and p["image"].max() == KK - 1
    assert p["caption"].min() == 0 and p["caption"].max() == KK - 1
    assert fits["a"].graph.shape == (N, N)
    bank = fits["a"].bank
    assert {"aspect_a", "aspect_b", "block_sizes", "block_pairs", "anchor", "candidates"} <= set(bank)
    assert list(bank["block_pairs"]) == ["affect__caption", "affect__image", "caption__image"]
    assert int(bank["block_sizes"].sum()) == len(bank["anchor"]) == 60


def test_made_keys_change_with_the_features(fits):
    a, b = fits["a"], fits["b"]
    assert not np.array_equal(a.parts["image"], b.parts["image"])
    assert not np.array_equal(a.parts["caption"], b.parts["caption"])
    assert (a.graph != b.graph).nnz > 0
    assert not np.array_equal(a.bank["anchor"], b.bank["anchor"])
    assert not np.array_equal(a.bank["candidates"], b.bank["candidates"])


def test_copied_keys_do_not(fits, corpus):
    for name in ("a", "b"):
        for key in ("affect", "local_groups"):
            assert fits[name].parts[key].tobytes() == corpus.frozen[key].tobytes(), (name, key)


def test_same_features_same_made_keys(corpus):
    img, txt = features(1), features(2)
    one = G.fit_partitions(img, txt, corpus.groups, corpus.scorer_train, corpus.frozen, **SMALL)
    two = G.fit_partitions(img, txt, corpus.groups, corpus.scorer_train, corpus.frozen, **SMALL)
    assert all(np.array_equal(one[k], two[k]) for k in one)


def test_image_uses_image_features_and_caption_uses_caption_features(corpus):
    img, txt = features(1), features(2)
    swapped = G.fit_partitions(txt, img, corpus.groups, corpus.scorer_train, corpus.frozen, **SMALL)
    plain = G.fit_partitions(img, txt, corpus.groups, corpus.scorer_train, corpus.frozen, **SMALL)
    assert np.array_equal(swapped["image"], plain["caption"]) and np.array_equal(swapped["caption"], plain["image"])


def test_misaligned_local_groups_refused(corpus):
    bad = {**corpus.frozen, "local_groups": corpus.frozen["local_groups"][::-1].copy()}
    with pytest.raises(AssertionError):
        G.fit_partitions(features(1), features(2), corpus.groups, corpus.scorer_train, bad, **SMALL)


def test_unlabelled_scorer_train_row_refused(corpus):
    with pytest.raises(AssertionError):     # a scorer-train row outside the fit rows would get label -1
        G.fit_partitions(features(1), features(2), corpus.groups, corpus.scorer_train[:-1], corpus.frozen, **SMALL)


# ---------------------------------------------------------------- CLIP's real file

def test_copied_keys_are_byte_equal_to_clips_file():
    from r9_backbone import parts
    frozen = G.load_frozen()
    with np.load(parts("clip_b32")["affect_km"], allow_pickle=False) as z:
        for key in ("affect", "local_groups"):
            assert frozen[key].tobytes() == z[key].tobytes() and frozen[key].dtype == z[key].dtype
    # every backbone's copy source is CLIP's file (the frozen role), never a recipe output
    assert parts("qwen3vl_emb_2b")["affect_km"] == parts("clip_b32")["affect_km"]


# ---------------------------------------------------------------- records, refusal

def test_record_and_refusal(tmp_path, fits, corpus):
    out = tmp_path / "o"
    shas = {"x": "0" * 64}
    arrays = {k: v for k, v in fits["a"].parts.items()}
    G.write_part(out, "partitions.npz", arrays, backbone="clip_b32", input_sha256=shas,
                 made=["image", "caption"], copied=["affect", "local_groups"])
    rec = json.loads((out / "partitions.npz.record.json").read_text())
    assert rec["keys_made"] == ["image", "caption"] and rec["keys_copied"] == ["affect", "local_groups"]
    assert rec["input_sha256"] == shas and rec["script_sha256"] == R.sha256_file(G.__file__)
    assert rec["backbone"] == "clip_b32"
    with np.load(out / "partitions.npz") as z:
        assert set(z.files) == set(arrays)
    with pytest.raises(FileExistsError):
        G.preflight(out, "clip_b32", stop_after="groups")
    with pytest.raises(FileExistsError):
        G.write_part(out, "partitions.npz", arrays, backbone="clip_b32", input_sha256=shas,
                     made=["image"], copied=[])


def test_preflight_names_all_steps(tmp_path):
    assert G.output_names("groups") == ["partitions.npz"]
    assert G.output_names("graph") == ["partitions.npz", "graph.npz"]
    assert G.output_names("bank") == ["partitions.npz", "graph.npz", "bank_AIC.npz"]
    assert G.output_names("heads") == ["partitions.npz", "graph.npz", "bank_AIC.npz",
                                       "n6_posteriors.npz", "step1_heads_style.npz"]
    (tmp_path / "bank_AIC.npz.record.json").write_text("{}")
    with pytest.raises(FileExistsError):
        G.preflight(tmp_path, "clip_b32", stop_after="bank")
    G.preflight(tmp_path, "clip_b32", stop_after="graph")      # bank files not in this run


# ---------------------------------------------------------------- the heads step wiring (fits stubbed)

def test_heads_step_fits_on_the_new_partitions_file(tmp_path, monkeypatch, corpus):
    import r9_heads as H
    out = tmp_path / "o"
    G.write_part(out, "partitions.npz", corpus.frozen, backbone="qwen3vl_emb_2b", input_sha256={},
                 made=[], copied=list(corpus.frozen))
    seen = {}
    monkeypatch.setattr(H, "fit_heads_r9", lambda data, groups, st, *, backbone, input_pins: (
        seen.update(pins=input_pins, backbone=backbone) or "HEADS"))
    monkeypatch.setattr(H, "recipe_input_shas", lambda backbone, upstream=None: (
        seen.update(upstream=upstream) or {"u": "1" * 64}))
    monkeypatch.setattr(H, "write_posteriors", lambda o, heads, data, selection, **kw: (
        seen.update(write=(o, heads, kw)) or {}))
    monkeypatch.setattr(H, "recipe_head_pins", lambda b: {H.E2_PARTITIONS_REL: ("clip", "x"), "other": ("p", "s")})
    split = SimpleNs(groups="g", scorer_train="st", selection="sel")
    G.fit_heads_step(out, "qwen3vl_emb_2b", "DATA", split)
    path, sha = seen["pins"][H.E2_PARTITIONS_REL]
    assert Path(path) == out / "partitions.npz" and sha == R.sha256_file(out / "partitions.npz")
    assert seen["pins"]["other"] == ("p", "s") and seen["backbone"] == "qwen3vl_emb_2b"
    o, heads, kw = seen["write"]
    assert Path(o) == out and heads == "HEADS" and kw["backbone"] == "qwen3vl_emb_2b"
    assert kw["input_sha256"] == {"u": "1" * 64}
