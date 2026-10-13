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
        graph = G.fit_graph(img, txt, corpus.scorer_train, allow_cpu=True)
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


def test_scorer_train_length_mismatch_refused(corpus):
    with pytest.raises(AssertionError):     # copied affect/local_groups have one row per scorer-train row
        G.fit_partitions(features(1), features(2), corpus.groups, corpus.scorer_train[:-1], corpus.frozen, **SMALL)


def test_unsorted_or_repeated_scorer_train_refused(corpus):
    for bad in (corpus.scorer_train[::-1], np.r_[corpus.scorer_train[:-1], corpus.scorer_train[0]]):
        with pytest.raises(AssertionError, match="sorted and unique"):
            G.fit_partitions(features(1), features(2), corpus.groups, bad, corpus.frozen, **SMALL)


def test_copied_affect_of_the_wrong_length_refused(corpus):
    bad = {**corpus.frozen, "affect": corpus.frozen["affect"][:-1]}
    with pytest.raises(AssertionError, match="do not match scorer_train"):
        G.fit_partitions(features(1), features(2), corpus.groups, corpus.scorer_train, bad, **SMALL)


class _Labels:
    """A ClipClusterSource stand-in whose image labels leave one scorer-train row unlabelled (-1) or lose a group."""
    def __init__(self, mode):
        self.mode = mode

    def __call__(self, img, txt, rows, **kw):
        real = _REAL_SOURCE(img, txt, rows, **kw)
        image = real.labels_by_view["image"].copy()
        if self.mode == "minus_one":
            image[rows[0]] = -1                         # every group keeps other rows, so only the -1 guard can fire
        else:
            image[image == KK - 1] = 0                  # k-1 groups left, no label below 0
        self.labels_by_view = {**real.labels_by_view, "image": image}
        return self


_REAL_SOURCE = G.ClipClusterSource


WHY = {"minus_one": "no label", "lost_group": "groups"}


@pytest.mark.parametrize("mode", ["minus_one", "lost_group"])
def test_unlabelled_scorer_train_row_or_lost_group_refused(corpus, monkeypatch, mode):
    monkeypatch.setattr(G, "ClipClusterSource", _Labels(mode))
    with pytest.raises(AssertionError, match=f"image.*{WHY[mode]}"):
        G.fit_partitions(features(1), features(2), corpus.groups, corpus.scorer_train, corpus.frozen, **SMALL)


@pytest.mark.parametrize("mode", ["minus_one", "lost_group"])
def test_bad_caption_labels_refused(corpus, monkeypatch, mode):
    real = G.kmeans_partition

    def bad(x, rows, k, seed):
        labels = real(x, rows, k=k, seed=seed).copy()
        if mode == "minus_one":
            labels[0] = -1
        else:
            labels[labels == k - 1] = 0
        return labels
    monkeypatch.setattr(G, "kmeans_partition", bad)
    with pytest.raises(AssertionError, match=f"caption.*{WHY[mode]}"):
        G.fit_partitions(features(1), features(2), corpus.groups, corpus.scorer_train, corpus.frozen, **SMALL)


# ---------------------------------------------------------------- scorer-train as a sorted proper subset

@pytest.fixture(scope="module")
def subset():
    groups = np.repeat(np.arange(P), PER).astype(np.int64)
    drop = np.r_[np.flatnonzero(groups % 5 == 0), 6, 13, 14]            # whole paintings and a few single rows
    st = np.setdiff1d(np.arange(N), drop).astype(np.int64)             # sorted, unique, a proper subset
    assert 0 < len(st) < N and (np.diff(st) > 0).all() and st[0] != 0
    rng = np.random.default_rng(11)
    frozen = {"affect": rng.integers(0, KK, size=len(st)).astype(np.int64),
              "local_groups": np.unique(groups[st], return_inverse=True)[1].astype(np.int64)}
    img, txt = features(1), features(2)
    return SimpleNs(groups=groups, st=st, frozen=frozen, img=img, txt=txt,
                    parts=G.fit_partitions(img, txt, groups, st, frozen, **SMALL))


def test_subset_labels_are_the_k_means_of_the_scorer_train_rows(subset):
    s = subset
    n = len(s.st)
    assert all(arr.shape == (n,) for arr in s.parts.values())
    ref = G.fit_partitions(s.img[s.st], s.txt[s.st], s.groups[s.st], np.arange(n), s.frozen, **SMALL)
    assert all(np.array_equal(s.parts[k], ref[k]) for k in s.parts)     # fitted on the rows, indexed by position
    first = G.fit_partitions(s.img[:n], s.txt[:n], s.groups[:n], np.arange(n), {
        "affect": s.frozen["affect"], "local_groups": np.unique(s.groups[:n], return_inverse=True)[1]}, **SMALL)
    assert not np.array_equal(s.parts["image"], first["image"])         # not the first n rows


def test_subset_graph_and_bank(subset):
    s = subset
    n = len(s.st)
    graph = G.fit_graph(s.img, s.txt, s.st, allow_cpu=True)
    assert graph.shape == (n, n)
    ref = G.fit_graph(s.img[s.st], s.txt[s.st], np.arange(n), allow_cpu=True)
    assert (graph != ref).nnz == 0
    bank = G.fit_bank(s.parts, bank_size=60, min_paintings=5)
    assert bank["anchor"].max() < n and bank["candidates"].max() < n


def test_graph_of_the_wrong_shape_refused(subset, monkeypatch):
    import scipy.sparse as sp
    monkeypatch.setattr(G, "build_content_graph", lambda img, txt, cfg: sp.eye(len(img) + 1, format="csr"))
    with pytest.raises(AssertionError, match="graph shape"):
        G.fit_graph(subset.img, subset.txt, subset.st, allow_cpu=True)


def test_bank_is_the_first_rows_of_the_seed_42_bank(subset):
    from dataclasses import fields
    from src.train.pseudo_partitions import build_episode_bank
    p = {k: subset.parts[k] for k in G.AIC}
    full = build_episode_bank(p, subset.parts["local_groups"], np.arange(len(subset.st)), n_per_pair=20, seed=42,
                              min_paintings=5)
    bank = G.fit_bank(subset.parts, bank_size=58, min_paintings=5)         # 3 blocks of 20, trimmed to 58
    assert list(bank["block_sizes"]) == [20, 20, 18] and len(bank["anchor"]) == 58
    for f in fields(full)[2:]:
        assert np.array_equal(bank[f.name], getattr(full, f.name)[:58]), f.name


def test_copied_keys_must_match_the_subset(subset):
    wrong = {**subset.frozen, "local_groups": np.arange(len(subset.st), dtype=np.int64)}
    with pytest.raises(AssertionError, match="local_groups"):
        G.fit_partitions(subset.img, subset.txt, subset.groups, subset.st, wrong, **SMALL)


# ---------------------------------------------------------------- constants equal the original scripts'

def _load_by_path(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_constants_equal_the_original_scripts():
    import inspect
    bp = G._load_build_partitions()
    rs = _load_by_path("r9t07_run_selection", R.TEST / "20261013_stage_d_selection/run_selection.py")
    assert (G.K, G.SEED, G.BANK_SIZE, G.N_SCORER) == (bp.K, bp.SEED, bp.BANK_SIZE, bp.N_ROWS) == (64, 42, 65_536, 183_694)
    assert (G.K, G.SEED) == (rs.N_CLUSTERS, rs.SEED)
    assert G.AIC == bp.SETS["AIC"]
    assert "min_paintings=30" in (R.TEST / "20261031_pseudo_partitions/build_partitions.py").read_text()
    assert G.MIN_PAINTINGS == R.MIN_PAINTINGS == 30
    fp, fb = inspect.signature(G.fit_partitions).parameters, inspect.signature(G.fit_bank).parameters
    assert (fp["k"].default, fp["seed"].default, fp["min_group_rows"].default) == (G.K, G.SEED, 200)
    assert (fb["bank_size"].default, fb["min_paintings"].default) == (G.BANK_SIZE, G.MIN_PAINTINGS)


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


# ---------------------------------------------------------------- the graph needs the GPU (finding 1)

def test_graph_refuses_the_cpu_unless_allowed(monkeypatch, subset):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="GPU"):
        G.fit_graph(subset.img, subset.txt, subset.st)
    with pytest.raises(RuntimeError, match="GPU"):
        G.graph_device_record()
    G.fit_graph(subset.img[:40], subset.txt[:40], np.arange(40), allow_cpu=True)


def test_graph_device_record(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i=0: "Fake GPU")
    assert G.graph_device_record() == {"device": "cuda", "device_name": "Fake GPU", "torch": torch.__version__,
                                       "cuda": torch.version.cuda, "use_half": True, "cpu_allowed": False}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert G.graph_device_record(allow_cpu=True) == {"device": "cpu", "device_name": "cpu", "torch": torch.__version__,
                                                     "cuda": torch.version.cuda, "use_half": False, "cpu_allowed": True}


# ---------------------------------------------------------------- code and version records (finding 3)

def test_code_record_hashes_the_modules_that_ran():
    import importlib
    import scipy
    import sklearn
    import torch
    rec = G.code_record()
    assert set(rec["modules"]) == {"src.model.graph", "src.conditional_buddy.buddy_graph",
                                   "src.train.condition_sources", "src.train.pseudo_partitions",
                                   "src.eval.aspect_episodes", "build_partitions", "r9_heads", "r9_data", "r9_common"}
    for name, entry in rec["modules"].items():
        mod = G._load_build_partitions() if name == "build_partitions" else importlib.import_module(name)
        assert entry["file"] == str(Path(mod.__file__).resolve()) and entry["sha256"] == R.sha256_file(mod.__file__), name
    assert rec["modules"]["build_partitions"]["file"].endswith("20261031_pseudo_partitions/build_partitions.py")
    assert rec["modules"]["src.model.graph"]["file"].endswith("src/model/graph.py")
    assert rec["versions"] == {"numpy": np.__version__, "scipy": scipy.__version__,
                               "sklearn": sklearn.__version__, "torch": torch.__version__}


# ---------------------------------------------------------------- bank keys (finding 2)

def test_bank_made_and_copied_keys(fits):
    made, copied = G.split_bank_keys(fits["a"].bank)
    assert made == ["anchor", "candidates", "pairs_a_img", "pairs_a_txt", "pairs_b_img", "pairs_b_txt"]
    assert copied == ["aspect_a", "aspect_b", "block_sizes", "block_pairs"]
    a, b = fits["a"].bank, fits["b"].bank
    assert all(not np.array_equal(a[k], b[k]) for k in made)                  # ticket 13: every made array differs
    assert all(np.array_equal(a[k], b[k]) for k in copied)                    # the constants do not
    assert set(a) == set(made) | set(copied)
    with pytest.raises(AssertionError):
        G.split_bank_keys({**a, "extra": np.zeros(1)})


# ---------------------------------------------------------------- records, refusal, read files

def test_write_part_records_code_and_the_keys_really_written(tmp_path, subset):
    out = tmp_path / "o"
    graph = G.fit_graph(subset.img[:40], subset.txt[:40], np.arange(40), allow_cpu=True)
    rec = G.write_part(out, "graph.npz", graph, backbone="qwen3vl_emb_2b", input_sha256={}, made=None, copied=[],
                       extra={"graph_build": {"device": "cpu"}})
    with np.load(out / "graph.npz") as z:
        assert rec["keys_made"] == list(z.files) and "graph" not in z.files
    on_disk = json.loads((out / "graph.npz.record.json").read_text())
    assert on_disk == rec and on_disk["code"] == G.code_record() and on_disk["graph_build"] == {"device": "cpu"}
    G.write_part(out, "p.npz", {"a": np.zeros(2)}, backbone="clip_b32", input_sha256={}, made=["a"], copied=[])
    assert json.loads((out / "p.npz.record.json").read_text())["code"] == G.code_record()


def test_refusal_names_the_files_to_remove(tmp_path):
    for name in ("partitions.npz", "partitions.npz.record.json", "graph.npz"):
        (tmp_path / name).write_text("x")
    with pytest.raises(FileExistsError) as err:
        G.preflight(tmp_path, "clip_b32", stop_after="bank")
    msg = str(err.value)
    assert "no resume" in msg and "remove" in msg
    for name in ("partitions.npz", "partitions.npz.record.json", "graph.npz"):
        assert str(tmp_path / name) in msg
    assert str(tmp_path / "bank_AIC.npz") not in msg                         # not there, nothing to remove
    other = tmp_path / "b"
    other.mkdir()
    (other / "build_record.json").write_text("{}")
    with pytest.raises(FileExistsError, match="overwrite .*build_record.json"):
        G.preflight(other, "clip_b32", stop_after="bank")                     # a rerun that would replace it
    G.preflight(other, "clip_b32", stop_after="graph")                         # not part of a run that stops earlier
    assert G.output_names("bank")[-1] == "bank_AIC.npz"


def test_read_shas_cover_the_new_store_clips_store_and_the_annotations(tmp_path, monkeypatch):
    ann = tmp_path / "ann.json"
    ann.write_text("[]")
    monkeypatch.setattr(G, "ANNOTATIONS_PATH", ann)
    monkeypatch.setattr(G, "store_shas", lambda b: {f"{b}/f": b[:3] * 22})
    shas = G.read_shas("qwen3vl_emb_2b")
    assert set(shas) == {"qwen3vl_emb_2b/f", "clip_b32/f", str(ann)} and shas[str(ann)] == R.sha256_file(ann)
    assert set(G.read_shas("clip_b32")) == {"clip_b32/f", str(ann)}


# ---------------------------------------------------------------- main with data and split stubbed (finding 4)

@pytest.fixture
def run(tmp_path, monkeypatch, subset):
    import functools
    import r9_heads as H
    s = subset
    calls = {"load_data": [], "heads": []}
    data = SimpleNs(img_features=s.img, txt_features=s.txt)
    split = SimpleNs(groups=s.groups, scorer_train=s.st)
    monkeypatch.setattr(R, "load_data", lambda b: calls["load_data"].append(b) or data)
    monkeypatch.setattr(R, "load_split", lambda d: split if d is data else pytest.fail("split from other data"))
    monkeypatch.setattr(R, "assert_input", lambda rel, backbone="clip_b32": hashlib_sha(rel))
    monkeypatch.setattr(H, "require_threads", lambda: None)
    monkeypatch.setattr(G, "read_shas", lambda b: {f"store/{b}": "5" * 64})
    monkeypatch.setattr(G, "load_frozen", lambda: dict(s.frozen))
    monkeypatch.setattr(G, "N_SCORER", len(s.st))
    monkeypatch.setattr(G, "fit_partitions", functools.partial(G.fit_partitions, **SMALL))
    monkeypatch.setattr(G, "fit_bank", functools.partial(G.fit_bank, bank_size=60, min_paintings=5))
    monkeypatch.setattr(G, "fit_heads_step", lambda *a: calls["heads"].append(a))
    return SimpleNs(out=tmp_path / "o", calls=calls, data=data, split=split, s=s)


def hashlib_sha(text):
    import hashlib
    return hashlib.sha256(str(text).encode()).hexdigest()


def run_main(x, *extra, allow_cpu=True):
    return G.main(["--backbone", "qwen3vl_emb_2b", "--out", str(x.out), *extra], allow_cpu=allow_cpu)


def test_main_all_steps(run, monkeypatch):
    import hashlib
    x, s = run, run.s
    graph_args = []
    real_graph = G.build_content_graph
    monkeypatch.setattr(G, "build_content_graph", lambda img, txt, cfg: graph_args.append((img, txt)) or real_graph(img, txt, cfg))
    assert run_main(x) == 0
    assert len(graph_args) == 1                               # image features first, caption features second, scorer-train rows
    assert np.array_equal(graph_args[0][0], s.img[s.st]) and np.array_equal(graph_args[0][1], s.txt[s.st])
    assert x.calls["load_data"] == ["qwen3vl_emb_2b"]
    assert x.calls["heads"] == [(x.out, "qwen3vl_emb_2b", x.data, x.split)]
    files = {p.name for p in x.out.iterdir()}
    assert files == {"partitions.npz", "graph.npz", "bank_AIC.npz", "build_record.json",
                     "partitions.npz.record.json", "graph.npz.record.json", "bank_AIC.npz.record.json"}
    with np.load(x.out / "partitions.npz") as z:
        parts = {k: z[k] for k in z.files}
    assert tuple(parts) == ("affect", "image", "caption", "local_groups")
    ref = G.fit_partitions(s.img, s.txt, s.groups, s.st, s.frozen)          # the stubbed features, not swapped
    assert all(np.array_equal(parts[k], ref[k]) for k in ref)
    swapped = G.fit_partitions(s.txt, s.img, s.groups, s.st, s.frozen)
    assert not np.array_equal(parts["image"], swapped["image"])
    rec = {n: json.loads((x.out / (n + ".record.json")).read_text()) for n in ("partitions.npz", "graph.npz", "bank_AIC.npz")}
    store, frozen_key = {"store/qwen3vl_emb_2b": "5" * 64}, str(G.parts("clip_b32")["affect_km"])
    prep, held = (str(R.INPUT_PATHS[r]) for r in (R.PREPARE_REL, R.HELD_CODES_REL))
    base = {**store, prep: hashlib_sha(R.PREPARE_REL), held: hashlib_sha(R.HELD_CODES_REL)}
    assert rec["partitions.npz"]["input_sha256"] == {**base, frozen_key: hashlib_sha(G.H.E2_PARTITIONS_REL)}
    assert rec["graph.npz"]["input_sha256"] == base
    mine = R.sha256_file(x.out / "partitions.npz")
    assert rec["bank_AIC.npz"]["input_sha256"] == {**store, str(x.out / "partitions.npz"): mine}
    assert (rec["partitions.npz"]["keys_made"], rec["partitions.npz"]["keys_copied"]) == (
        ["image", "caption"], ["affect", "local_groups"])
    with np.load(x.out / "graph.npz") as z:
        assert rec["graph.npz"]["keys_made"] == list(z.files) and rec["graph.npz"]["keys_copied"] == []
    assert rec["graph.npz"]["graph_build"] == G.graph_device_record(allow_cpu=True)
    assert rec["bank_AIC.npz"]["keys_made"] == ["anchor", "candidates", "pairs_a_img", "pairs_a_txt",
                                                "pairs_b_img", "pairs_b_txt"]
    assert rec["bank_AIC.npz"]["keys_copied"] == ["aspect_a", "aspect_b", "block_sizes", "block_pairs"]
    for name, r in rec.items():
        assert r["backbone"] == "qwen3vl_emb_2b" and r["code"] == G.code_record(), name
        assert r["script_sha256"] == R.sha256_file(G.__file__) 
        assert ("graph_build" in r) == (name == "graph.npz"), name
    # build_record.json: the three SHA-256s run_gonogo.train compares (sha_array of int64 local_groups, file SHAs)
    build = json.loads((x.out / "build_record.json").read_text())
    assert build["sha256"] == {
        "local_groups": hashlib.sha256(np.ascontiguousarray(parts["local_groups"].astype(np.int64)).tobytes()).hexdigest(),
        "graph.npz": R.sha256_file(x.out / "graph.npz"), "bank_AIC.npz": R.sha256_file(x.out / "bank_AIC.npz")}
    assert build["sha256"]["local_groups"] == G.sha_array(s.frozen["local_groups"])


def test_main_stop_after_groups_needs_no_gpu_and_no_build_record(run, monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    x = run
    assert run_main(x, "--stop-after", "groups", allow_cpu=False) == 0
    assert {p.name for p in x.out.iterdir()} == {"partitions.npz", "partitions.npz.record.json"}
    assert x.calls["heads"] == []


def test_main_graph_step_without_gpu_stops_before_any_work(run, monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    x = run
    with pytest.raises(RuntimeError, match="GPU"):
        run_main(x, "--stop-after", "graph", allow_cpu=False)
    assert x.calls["load_data"] == [] and not x.out.exists()


def test_main_refuses_a_wrong_scorer_train_size(run, monkeypatch):
    monkeypatch.setattr(G, "N_SCORER", len(run.s.st) + 1)
    with pytest.raises(AssertionError, match="scorer_train"):
        run_main(run, "--stop-after", "groups")
    assert not run.out.exists()


def test_main_rerun_after_a_crash_is_refused_with_the_files_named(run, monkeypatch):
    x = run
    monkeypatch.setattr(G, "fit_bank", lambda parts: (_ for _ in ()).throw(RuntimeError("crash")))
    with pytest.raises(RuntimeError, match="crash"):
        run_main(x)
    with pytest.raises(FileExistsError, match=str(x.out / "graph.npz")):
        run_main(x)


# ---------------------------------------------------------------- the heads step wiring (fits stubbed)

def _stub_heads(monkeypatch, seen):
    import r9_heads as H
    monkeypatch.setattr(H, "fit_heads_r9", lambda data, groups, st, *, backbone, input_pins: (
        seen.update(pins=input_pins, backbone=backbone) or "HEADS"))
    monkeypatch.setattr(H, "recipe_input_shas", lambda backbone, upstream=None: (
        seen.update(upstream=upstream) or {"u": "1" * 64}))
    monkeypatch.setattr(H, "write_posteriors", lambda o, heads, data, selection, **kw: (
        seen.update(write=(o, heads, kw)) or {}))


def test_heads_step_fits_on_the_new_partitions_file_of_a_non_default_out(tmp_path, monkeypatch, corpus):
    import r9_heads as H
    out = tmp_path / "o"
    default = R.RESULTS / "recipe" / "qwen3vl_emb_2b" / "partitions.npz"
    assert not default.exists()                       # the default location is empty: hashing it would crash
    G.write_part(out, "partitions.npz", corpus.frozen, backbone="qwen3vl_emb_2b", input_sha256={},
                 made=[], copied=list(corpus.frozen))
    seen = {}
    _stub_heads(monkeypatch, seen)
    split = SimpleNs(groups="g", scorer_train="st", selection="sel")
    G.fit_heads_step(out, "qwen3vl_emb_2b", "DATA", split)
    path, sha = seen["pins"][H.E2_PARTITIONS_REL]
    assert Path(path) == out / "partitions.npz" and sha == R.sha256_file(out / "partitions.npz")
    assert seen["pins"][H.PARTITION_L_REL][0] == R.INPUT_PATHS[H.PARTITION_L_REL]      # frozen pins still CLIP's
    assert seen["pins"][H.AFFECT_KM_REL][0] == R.INPUT_PATHS[H.E2_PARTITIONS_REL]
    assert seen["upstream"] == seen["pins"]
    o, heads, kw = seen["write"]
    assert Path(o) == out and heads == "HEADS" and kw["backbone"] == "qwen3vl_emb_2b"
    assert kw["input_sha256"] == {"u": "1" * 64}


def test_heads_step_refuses_a_partitions_pin_that_is_not_its_own_file(tmp_path, monkeypatch, corpus):
    import r9_heads as H
    out = tmp_path / "o"
    G.write_part(out, "partitions.npz", corpus.frozen, backbone="qwen3vl_emb_2b", input_sha256={},
                 made=[], copied=list(corpus.frozen))
    seen = {}
    _stub_heads(monkeypatch, seen)
    monkeypatch.setattr(H, "recipe_head_pins", lambda b, root=None: {H.E2_PARTITIONS_REL: (tmp_path / "other.npz", "x")})
    with pytest.raises(AssertionError, match="not this run's file"):
        G.fit_heads_step(out, "qwen3vl_emb_2b", "DATA", SimpleNs(groups="g", scorer_train="st", selection="sel"))
    assert "pins" not in seen


def test_recipe_head_pins_root_is_optional(tmp_path):
    import r9_heads as H
    mine = tmp_path / "partitions.npz"
    mine.write_bytes(b"not read, only hashed")
    pins = H.recipe_head_pins("qwen3vl_emb_2b", mine.parent)
    assert pins[H.E2_PARTITIONS_REL] == (mine, R.sha256_file(mine))
    assert pins[H.PARTITION_L_REL][0] == R.INPUT_PATHS[H.PARTITION_L_REL]
    assert H.recipe_head_pins.__defaults__ == (None,)                       # 05's callers keep the old default
