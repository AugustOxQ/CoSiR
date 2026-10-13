"""Tests of r9_recipe_reader (ticket 09; rule §3.2 item 1: halves, cross-fitted heads, A0 banks, half-readers, tau).

A small synthetic corpus: 600 paintings of 4 rows, three groupings of 6 groups. Written before the module. The
verdict-side facts under test: the pin check on every grouping source, the asserted size parameters, the cross-fit
(a row is scored by the head of the other half), the banks (rows inside their half, the rule's blocks), the readers
(one per half, C by 5-fold CV), tau (percentiles 0, 25, 50, 75 of exactly 24,576 margins). The slow test (R9_SLOW=1)
refits CLIP's `image` heads on the real data and compares them with the pinned `rb_heads_image.npz`.

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python -m pytest -q -p no:cacheprovider \
        src/test/20261202_r9_scope/test_r9_recipe_reader.py
"""
import ast
import importlib.util
import itertools
import json
import os
import pickle
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r9_common as R  # noqa: E402  (first: MAIN and src)
import r9_recipe_reader as RR  # noqa: E402

import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

P, PER, D, KK = 600, 4, 16, 6          # paintings, rows per painting, feature dim, groups per grouping
N = P * PER
NG = {"affect": KK, "image": KK, "caption": KK, "csd": 3, "rand": 3}
SMALL = {**RR.PARAMS, "n_st": N, "n_paintings": P, "half_paintings": P // 2, "head_rows": 800, "check_rows": 10_000,
         "copy_check_rows": 300, "bank_n": 20, "bank_size": 60, "min_paintings": 5, "validate_n": 30,
         "n_tau_values": 2 * 3 * 50}
_COUNT = itertools.count()


def unit(x):
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def features(seed, labels):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(KK, D))
    return unit(centres[labels] + 0.8 * rng.normal(size=(N, D)))


def make_halves(st, groups, seed=0):
    """rb_build.stage_halves' construction on the synthetic corpus."""
    paint = groups[st]
    paintings, g_st = np.unique(paint, return_inverse=True)
    order = np.random.default_rng(seed).permutation(len(paintings))
    half_of_code = np.zeros(len(paintings), dtype=np.int8)
    half_of_code[order[len(paintings) // 2:]] = 1
    half = half_of_code[g_st]
    local = {k: np.flatnonzero(half == k).astype(np.int64) for k in (0, 1)}
    return {"scorer_train": st, "painting_of_local_row": paint.astype(np.int64), "g_st": g_st.astype(np.int64),
            "half_of_local_row": half, "local_rows_half0": local[0], "local_rows_half1": local[1],
            "global_rows_half0": st[local[0]], "global_rows_half1": st[local[1]]}


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    rng = np.random.default_rng(3)
    groups = np.repeat(np.arange(P), PER).astype(np.int64)
    st = np.arange(N, dtype=np.int64)
    g_st = np.unique(groups[st], return_inverse=True)[1].astype(np.int64)
    # group labels follow the painting a little, so that the heads and the reader have something to learn
    lab = {g: rng.integers(0, KK, size=N).astype(np.int64) for g in ("affect", "image", "caption")}
    for g in lab:
        lab[g][:NG[g]] = np.arange(NG[g])                  # every group present
    for g in ("csd", "rand"):                              # constant within a painting, 3 groups
        codes = rng.integers(0, 3, size=P).astype(np.int64)
        codes[:3] = np.arange(3)
        lab[g] = codes[g_st]
    img = features(1, lab["image"])
    txt = features(2, lab["caption"])
    root = tmp_path_factory.mktemp("sources")
    files = {"partitions": root / "partitions.npz", "partition_L": root / "per_anchor_told_oracle.npz",
             "style": root / "step1_group_style.npz"}
    np.savez(files["partitions"], affect=lab["affect"], image=lab["image"], caption=lab["caption"], local_groups=g_st)
    np.savez(files["partition_L"], partition_L=lab["affect"], partition_R0=lab["affect"])
    np.savez(files["style"], scorer_train=st, g_st=g_st, style_csd=lab["csd"], style_rand=lab["rand"])
    sources = {"affect": (files["partition_L"], RR.sha256_file(files["partition_L"]), "partition_L"),
               "image": (files["partitions"], RR.sha256_file(files["partitions"]), "image"),
               "caption": (files["partitions"], RR.sha256_file(files["partitions"]), "caption"),
               "csd": (files["style"], RR.sha256_file(files["style"]), "style_csd"),
               "rand": (files["style"], RR.sha256_file(files["style"]), "style_rand")}
    halves = make_halves(st, groups)
    data = SimpleNamespace(img_features=img, txt_features=txt)
    split = SimpleNamespace(scorer_train=st, groups=groups, selection=np.arange(0, N, 7, dtype=np.int64))
    env = SimpleNamespace(backbone="qwen3vl_emb_2b", data=data, split=split, halves=halves, sources=sources,
                          params=SMALL, n_groups=NG, e2=(files["partitions"], RR.sha256_file(files["partitions"])),
                          input_sha256={}, lab=lab,
                          halves_sha256="0" * 64)
    return env


@pytest.fixture(scope="module", autouse=True)
def threads():
    assert os.environ.get("OMP_NUM_THREADS") == "8" and os.environ.get("MKL_NUM_THREADS") == "8"


# ---------------------------------------------------------------- parameters (rb_build's size constants, asserted)

def test_production_parameters_are_the_rules_numbers():
    p = RR.PARAMS
    assert (p["n_st"], p["n_paintings"], p["half_paintings"]) == (183_694, 36_518, 18_259)
    assert (p["head_rows"], p["check_rows"], p["copy_check_rows"]) == (60_000, 10_000, 3_000)
    assert (p["bank_n"], p["bank_size"], tuple(p["bank_seeds"]), p["min_paintings"]) == (
        16_384, 49_152, (11_700, 11_800), 30)
    assert p["bank_size"] == 3 * p["bank_n"]
    assert (p["reader_max_iter"], tuple(p["c_grid"]), p["n_folds"]) == (2000, (0.01, 0.1, 1.0, 10.0, 100.0), 5)
    assert (p["n_per_pair_seed42"], p["n_tau_values"], tuple(p["tau_percentiles"])) == (4096, 24_576, (0, 25, 50, 75))
    assert p["n_tau_values"] == 2 * 3 * p["n_per_pair_seed42"]
    RR.check_params(p)


@pytest.mark.parametrize("key,value", [("n_st", 183_693), ("n_paintings", 36_519), ("half_paintings", 18_258),
                                       ("head_rows", 59_999), ("check_rows", 9_999), ("bank_n", 16_383),
                                       ("bank_size", 49_151), ("bank_seeds", (11_700, 11_801)),
                                       ("min_paintings", 29), ("reader_max_iter", 1999),
                                       ("c_grid", (0.01, 0.1, 1.0, 10.0)), ("n_folds", 4),
                                       ("n_tau_values", 24_575), ("tau_percentiles", (0, 25, 50, 80)),
                                       ("n_per_pair_seed42", 4095), ("validate_n", 999)])
def test_a_changed_size_parameter_stops_the_run(key, value):
    with pytest.raises(AssertionError, match=key):
        RR.check_params({**RR.PARAMS, key: value})


def test_unknown_or_missing_parameter_stops_the_run():
    with pytest.raises(AssertionError):
        RR.check_params({**RR.PARAMS, "extra": 1})
    with pytest.raises(AssertionError):
        RR.check_params({k: v for k, v in RR.PARAMS.items() if k != "bank_n"})


# ---------------------------------------------------------------- the pin check on every grouping source

def test_clip_sources_are_round_1s_pins():
    src = RR.source_pins("clip_b32")
    assert set(src) == set(RR.GROUPINGS)
    for name, (path, sha, key) in src.items():
        assert (path, sha, key) == tuple(RR.rb.SOURCES[name]), name
    assert src["affect"][2] == "partition_L" and src["csd"][2] == "style_csd" and src["rand"][2] == "style_rand"


def write_part_record(part, backbone):
    """07's `write_part` record next to a partitions file (only the field 09 reads)."""
    Path(str(part) + ".record.json").write_text(json.dumps({"backbone": backbone}))


def test_a_new_backbones_sources_are_its_parts_and_clips_frozen_files(tmp_path):
    part = tmp_path / "partitions.npz"
    np.savez(part, image=np.arange(3), caption=np.arange(3))
    write_part_record(part, "qwen3vl_emb_2b")
    src = RR.source_pins("qwen3vl_emb_2b", tmp_path)
    for name in ("image", "caption"):
        assert src[name][0] == part and src[name][1] == RR.sha256_file(part) and src[name][2] == name
    for name in ("affect", "csd", "rand"):                      # partition_L and style stay CLIP's
        assert tuple(src[name]) == tuple(RR.rb.SOURCES[name]), name


def test_the_right_sources_load(corpus):
    lab = RR.load_groupings(list(RR.GROUPINGS), corpus.split.scorer_train, corpus.halves["g_st"], corpus.sources,
                            corpus.n_groups, e2=corpus.e2)
    for g, a in lab.items():
        assert np.array_equal(a, corpus.lab[g]), g


def swapped(corpus, tmp_path, name, how):
    """A copy of the sources with ``name``'s file replaced by a wrong part (the pin is still the right file's)."""
    path, sha, key = corpus.sources[name]
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    if how == "labels":                                         # same keys, one label changed
        arrays[key] = (arrays[key] + 1) % NG[name]
    elif how == "other_key":                                    # the other grouping's labels under this key
        arrays[key] = arrays["caption" if key == "image" else "image"] if key in ("image", "caption") else arrays[key][::-1]
    wrong = tmp_path / f"wrong_{next(_COUNT)}.npz"
    np.savez(wrong, **arrays)
    return {**corpus.sources, name: (wrong, sha, key)}


@pytest.mark.parametrize("name", RR.GROUPINGS)
@pytest.mark.parametrize("how", ["labels", "other_key"])
def test_a_source_swapped_for_a_wrong_part_fails_the_pin_check(corpus, tmp_path, name, how):
    bad = swapped(corpus, tmp_path, name, how)
    with pytest.raises(SystemExit, match="SHA-256"):
        RR.load_groupings([name], corpus.split.scorer_train, corpus.halves["g_st"], bad, corpus.n_groups,
                          e2=corpus.e2)


def test_the_pin_check_is_a_marked_guard_and_the_mutant_lets_the_wrong_part_through(corpus, tmp_path):
    mut = mutant(tmp_path, "r9_recipe_reader.py", "source_pin")
    bad = swapped(corpus, tmp_path, "image", "labels")
    out = mut.load_groupings(["image"], corpus.split.scorer_train, corpus.halves["g_st"], bad, corpus.n_groups,
                             e2=corpus.e2)
    assert not np.array_equal(out["image"], corpus.lab["image"])        # without the guard the wrong labels come in


def test_labels_not_aligned_with_the_halves_are_refused(corpus):
    path, sha, key = corpus.sources["image"]
    with pytest.raises(AssertionError, match="aligned"):
        RR.load_groupings(["image"], corpus.split.scorer_train, corpus.halves["g_st"][::-1].copy(), corpus.sources,
                          corpus.n_groups, e2=corpus.e2)


def test_a_grouping_with_a_missing_group_is_refused(corpus):
    with pytest.raises(AssertionError, match="non-empty groups"):
        RR.load_groupings(["image"], corpus.split.scorer_train, corpus.halves["g_st"], corpus.sources,
                          {**corpus.n_groups, "image": KK + 1}, e2=corpus.e2)


# ---------------------------------------------------------------- the halves

def test_good_halves_pass(corpus):
    RR.check_halves(corpus.halves, corpus.split, corpus.params)


@pytest.mark.parametrize("what", ["scorer_train", "painting", "half_rows", "sizes"])
def test_bad_halves_are_refused(corpus, what):
    h = dict(corpus.halves)
    if what == "scorer_train":
        h["scorer_train"] = h["scorer_train"][::-1].copy()
    elif what == "painting":
        h["painting_of_local_row"] = h["painting_of_local_row"][::-1].copy()
    elif what == "half_rows":                                   # a row listed in both halves
        h["local_rows_half1"] = np.sort(np.r_[h["local_rows_half1"], h["local_rows_half0"][:1]])
    elif what == "sizes":
        pass
    with pytest.raises(AssertionError):
        RR.check_halves(h, corpus.split, {**corpus.params, "half_paintings": P // 2 - 1} if what == "sizes"
                        else corpus.params)


# ---------------------------------------------------------------- cross-fitted heads

@pytest.fixture(scope="module")
def heads(corpus):
    out = {}
    for g in ("image", "affect"):
        lab = RR.rto_labels(corpus.lab[g], corpus.split.scorer_train, len(corpus.split.groups))
        arrays, prov = RR.fit_crossfit_heads(
            {"img": corpus.data.img_features, "txt": corpus.data.txt_features}, lab, corpus.halves, g,
            corpus.n_groups[g], n_rows=corpus.params["head_rows"], check_rows=corpus.params["check_rows"],
            backbone=corpus.backbone)
        out[g] = SimpleNamespace(arrays=arrays, prov=prov, lab=lab)
    return out


def test_heads_keys_shapes_and_filling(corpus, heads):
    a = heads["image"].arrays
    assert tuple(a) == ("img", "txt", "filled_by_half", "classes_half0", "classes_half1")
    for m in ("img", "txt"):
        assert a[m].shape == (N, KK) and a[m].dtype == np.float32 and np.isfinite(a[m]).all()
        assert np.allclose(a[m].sum(axis=1), 1.0, atol=1e-5)
    assert np.array_equal(a["filled_by_half"], 1 - corpus.halves["half_of_local_row"])
    assert a["filled_by_half"].dtype == np.int8


def test_a_row_is_scored_by_the_head_of_the_other_half(corpus, heads):
    """Independent refit: the head of half k (draw from half k only) gives the posteriors of half 1-k's rows."""
    F = {"img": corpus.data.img_features, "txt": corpus.data.txt_features}
    h = corpus.halves
    lab = heads["image"].lab
    for k in (0, 1):
        own = h[f"global_rows_half{k}"]
        draw = np.random.default_rng(0).choice(own, corpus.params["head_rows"], replace=False)
        for m in ("img", "txt"):
            clf = LogisticRegression(C=1.0, max_iter=300).fit(unit(F[m][draw]), lab[draw])
            want = clf.predict_proba(unit(F[m][h[f"global_rows_half{1 - k}"]])).astype(np.float32)
            got = heads["image"].arrays[m][h[f"local_rows_half{1 - k}"]]
            assert np.array_equal(got, want), (k, m)


def test_head_draw_stays_inside_its_half(corpus, heads):
    for k in (0, 1):
        assert heads["image"].prov[str(k)]["n_draw"] == corpus.params["head_rows"]
        assert heads["image"].prov[str(k)]["all_groups_in_draw"] is True


def test_a_group_missing_from_a_draw_stops_the_run(corpus):
    lab = RR.rto_labels(corpus.lab["image"].copy(), corpus.split.scorer_train, len(corpus.split.groups))
    lab[lab == 0] = 1                                           # group 0 has no row at all
    with pytest.raises(SystemExit, match="missing"):
        RR.fit_crossfit_heads({"img": corpus.data.img_features, "txt": corpus.data.txt_features}, lab,
                              corpus.halves, "image", KK, n_rows=corpus.params["head_rows"], check_rows=10_000,
                              backbone=corpus.backbone)


def test_the_recipe_copy_equals_the_original_head_fit(corpus):
    lab = RR.rto_labels(corpus.lab["image"], corpus.split.scorer_train, len(corpus.split.groups))
    ok = RR.check_head_copy(corpus.data, corpus.split, lab, n_rows=corpus.params["copy_check_rows"])
    assert ok is True


# ---------------------------------------------------------------- banks

@pytest.fixture(scope="module")
def banks(corpus):
    labels = {g: corpus.lab[g] for g in RR.A0}
    out = {}
    for j in (0, 1):
        out[j] = RR.fit_bank_half(labels, corpus.halves, j, params=corpus.params)
    return out


def test_bank_blocks_rows_and_sizes(corpus, banks):
    assert RR.rb.BLOCKS["A0"] == [("affect", "caption"), ("affect", "image"), ("caption", "image")]
    n = corpus.params["bank_n"]
    for j, b in banks.items():
        arrays = b.arrays
        assert list(arrays["block_pairs"]) == ["affect__caption", "affect__image", "caption__image"]
        assert int(arrays["block_size"]) == n and int(arrays["half"]) == j
        assert int(arrays["seed"]) == corpus.params["bank_seeds"][j]
        assert len(arrays["anchor"]) == 3 * n == corpus.params["bank_size"]
        rows = np.concatenate([arrays[f] .ravel() for f in ("anchor", "candidates", "pairs_a_img", "pairs_a_txt",
                                                          "pairs_b_img", "pairs_b_txt")])
        assert (corpus.halves["half_of_local_row"][rows] == j).all()      # every member row lies in its half
        assert b.prov["validated_per_block"]


def test_the_two_halves_use_different_seeds_and_different_episodes(banks):
    assert not np.array_equal(banks[0].arrays["anchor"], banks[1].arrays["anchor"])


def test_a_bank_of_the_wrong_size_is_refused(corpus):
    labels = {g: corpus.lab[g] for g in RR.A0}
    with pytest.raises(AssertionError, match="bank_size"):
        RR.fit_bank_half(labels, corpus.halves, 0, params={**corpus.params, "bank_size": 61})


def test_a_builder_row_outside_the_half_is_refused(corpus, monkeypatch):
    import src.train.pseudo_partitions as PP
    real = PP.build_episode_bank
    foreign = corpus.halves["local_rows_half1"][0]

    def leaky(*a, **kw):
        bank = real(*a, **kw)
        bank.anchor[0] = foreign                                 # half 0 is being built: a row of half 1
        return bank
    monkeypatch.setattr(PP, "build_episode_bank", leaky)
    labels = {g: corpus.lab[g] for g in RR.A0}
    with pytest.raises(AssertionError, match="outside half 0"):
        RR.fit_bank_half(labels, corpus.halves, 0, params=corpus.params)


def test_bank_follows_the_label_arrays_it_is_given(corpus):
    """The groupings are the inputs: the episodes of a changed image labelling change."""
    labels = {g: corpus.lab[g] for g in RR.A0}
    other = {**labels, "image": (labels["image"] + 1) % KK}
    a = RR.fit_bank_half(labels, corpus.halves, 0, params=corpus.params).arrays
    b = RR.fit_bank_half(other, corpus.halves, 0, params=corpus.params).arrays
    assert not np.array_equal(a["pairs_a_img"], b["pairs_a_img"])


# ---------------------------------------------------------------- readers

@pytest.fixture(scope="module")
def readers(corpus, heads, banks):
    # the heads of the other groupings are the same shape; reuse image's for caption (a stand-in for the reader test)
    post = {"affect": heads["affect"].arrays, "image": heads["image"].arrays, "caption": heads["image"].arrays}
    return RR.fit_readers(post, {j: banks[j].arrays for j in (0, 1)}, corpus.halves, params=corpus.params), post


def test_reader_pickle_format_is_round_1s(corpus, readers):
    (pkl, rec), _ = readers
    assert pkl["config"] == "A0" and tuple(pkl["groupings"]) == RR.A0 == ("affect", "image", "caption")
    assert pkl["feature_names"] == RR.rf.feature_names(RR.A0) and len(pkl["feature_names"]) == 18
    assert pkl["rule_sha256"] == RR.C.RULE_SHA and pkl["smoke"] is False
    assert len(pkl["halves"]) == 2
    for h in pkl["halves"]:
        assert set(h) == {"scaler", "model", "C"} and h["C"] in RR.PARAMS["c_grid"]
        assert np.array_equal(h["model"].classes_, np.arange(3))
        assert h["model"].coef_.shape == (3, 18)
    for j in ("0", "1"):
        assert rec["halves"][j]["chosen_C"] in RR.PARAMS["c_grid"]
        assert len(rec["halves"][j]["cv_table"]) == len(RR.PARAMS["c_grid"])
        assert rec["halves"][j]["n_examples"] == 2 * SMALL["bank_size"]


def test_each_half_reader_sees_only_its_halfs_bank(corpus, readers, banks):
    (pkl, rec), post = readers
    for j in (0, 1):
        ep = RR.bank_episodes(banks[j].arrays)
        F = RR.rf.both_conditions({g: {"img": post[g]["img"], "txt": post[g]["txt"]} for g in RR.A0}, RR.A0, ep)
        X, y, epi = RR.stack_conditions_for(F, banks[j].arrays)
        scaler = pkl["halves"][j]["scaler"]
        assert np.allclose(scaler.mean_, X.mean(axis=0))                 # scaler fitted on all of half j's bank
        assert rec["halves"][str(j)]["n_examples"] == len(y)


def test_a_bank_row_outside_its_half_is_refused(corpus, banks, heads):
    bad = {k: v.copy() for k, v in banks[0].arrays.items()}
    other_half_row = corpus.halves["local_rows_half1"][0]
    bad["anchor"][0] = other_half_row
    post = {g: heads["image"].arrays for g in RR.A0}
    with pytest.raises(AssertionError, match="outside half"):
        RR.fit_readers(post, {0: bad, 1: banks[1].arrays}, corpus.halves, params=corpus.params)


def test_heads_scored_by_the_same_half_are_refused(corpus, banks, heads):
    bad = {k: v.copy() for k, v in heads["image"].arrays.items()}
    bad["filled_by_half"] = corpus.halves["half_of_local_row"].astype(np.int8)       # rows scored by their own half
    post = {"affect": bad, "image": bad, "caption": bad}
    with pytest.raises(AssertionError, match="other half"):
        RR.fit_readers(post, {j: banks[j].arrays for j in (0, 1)}, corpus.halves, params=corpus.params)


# ---------------------------------------------------------------- tau

def _readers_for_tau(corpus, readers):
    (pkl, _), _ = readers
    return pkl


def synthetic_f42(n, seed=0):
    rng = np.random.default_rng(seed)
    return {c: rng.normal(size=(n, 18)) for c in ("a", "b")}


def test_margins_are_the_averaged_top_two_gap(corpus, readers):
    pkl = _readers_for_tau(corpus, readers)
    F = synthetic_f42(50)
    margins, probs = RR.reader_margins(pkl, F)
    for c in ("a", "b"):
        p = np.mean([h["model"].predict_proba(h["scaler"].transform(F[c])) for h in pkl["halves"]], axis=0)
        top = np.sort(p, axis=1)
        assert np.allclose(probs[c], p, atol=1e-15, rtol=0)
        assert np.array_equal(margins[c], top[:, -1] - top[:, -2])
        assert margins[c].shape == (50,) and (margins[c] >= 0).all()


def test_tau_is_the_four_percentiles_of_all_margins(corpus):
    rng = np.random.default_rng(1)
    m = {"a": rng.random(50), "b": rng.random(50)}
    taus, n = RR.thresholds(m, n_expected=100)
    both = np.concatenate([m["a"], m["b"]])
    assert n == 100 and len(taus) == 4 and all(type(t) is float for t in taus)
    assert taus == [float(np.percentile(both, q)) for q in (0, 25, 50, 75)]
    assert taus[0] == both.min() and taus == sorted(taus)


@pytest.mark.parametrize("n_expected", [99, 101, 24_576])
def test_the_number_of_margins_is_asserted(n_expected):
    m = {"a": np.linspace(0, 1, 50), "b": np.linspace(0, 1, 50)}
    with pytest.raises(AssertionError, match="24,576|margins"):
        RR.thresholds(m, n_expected=n_expected)


def test_non_finite_margins_are_refused():
    m = {"a": np.array([0.1, np.nan]), "b": np.array([0.2, 0.3])}
    with pytest.raises(AssertionError, match="finite"):
        RR.thresholds(m, n_expected=4)


def test_tau_json_roundtrips_at_full_precision(tmp_path):
    taus = [0.0, 0.123456789012345678, 1 / 3, 0.7]
    rec = RR.tau_record("qwen3vl_emb_2b", taus, 24_576, reader_sha256="ab" * 32)
    assert rec["taus"] == taus and rec["percentiles"] == [0, 25, 50, 75] and rec["n_values"] == 24_576
    assert json.loads(json.dumps(rec))["taus"] == taus


# ---------------------------------------------------------------- outputs, records, refusal

def test_output_names_per_stage():
    assert RR.output_names("heads") == [f"rb_heads_{g}.npz" for g in RR.GROUPINGS]
    assert RR.output_names("heads", ["image"]) == ["rb_heads_image.npz"]
    assert RR.output_names("bank") == ["rb_bank_A0_half0.npz", "rb_bank_A0_half1.npz"]
    assert RR.output_names("train") == ["rb_reader_A0.pkl", "rb_reader_A0.json"]
    assert RR.output_names("tau") == ["rc_tau.json"]
    assert RR.output_names("all") == (RR.output_names("heads") + RR.output_names("bank") + RR.output_names("train")
                                      + RR.output_names("tau"))
    # the names are the roles' file names of r9_backbone
    names = {p.name for p in RR.parts("qwen3vl_emb_2b").values()}
    assert set(RR.output_names("all")) <= names


def test_existing_output_or_record_is_refused_before_any_work(tmp_path):
    (tmp_path / "rb_bank_A0_half1.npz.record.json").write_text("{}")
    with pytest.raises(FileExistsError, match="rb_bank_A0_half1.npz.record.json"):
        RR.preflight(tmp_path, "qwen3vl_emb_2b", "all")
    RR.preflight(tmp_path, "qwen3vl_emb_2b", "heads")                  # the heads stage does not touch it
    (tmp_path / "rc_tau.json").write_text("{}")
    with pytest.raises(FileExistsError, match="rc_tau.json"):
        RR.preflight(tmp_path, "qwen3vl_emb_2b", "tau")


def test_unknown_stage_or_backbone_is_refused(tmp_path):
    with pytest.raises(AssertionError):
        RR.preflight(tmp_path, "qwen3vl_emb_2b", "halves")
    with pytest.raises(AssertionError):
        RR.preflight(tmp_path, "nope", "all")


def test_stages_write_files_with_records_and_never_overwrite(corpus, tmp_path):
    out = tmp_path / "out"
    RR.stage_heads(corpus, out, ["image"])
    RR.stage_bank(corpus, out)
    assert sorted(p.name for p in out.iterdir()) == sorted(
        ["rb_heads_image.npz", "rb_heads_image.npz.record.json", "rb_bank_A0_half0.npz",
         "rb_bank_A0_half0.npz.record.json", "rb_bank_A0_half1.npz", "rb_bank_A0_half1.npz.record.json"])
    rec = json.loads((out / "rb_heads_image.npz.record.json").read_text())
    assert rec["backbone"] == "qwen3vl_emb_2b" and rec["script_sha256"] == R.sha256_file(RR.__file__)
    assert rec["keys_made"] == ["img", "txt"]                       # the posteriors; the rest equals CLIP's (L8)
    assert rec["keys_copied"] == ["filled_by_half", "classes_half0", "classes_half1"]
    assert str(corpus.sources["image"][0]) in rec["input_sha256"]
    assert rec["input_sha256"][str(corpus.sources["image"][0])] == corpus.sources["image"][1]
    assert rec["halves_sha256"] == corpus.halves_sha256
    brec = json.loads((out / "rb_bank_A0_half0.npz.record.json").read_text())
    assert brec["keys_made"] == list(RR.rb.EPISODE_FIELDS)
    assert brec["keys_copied"] == ["block_pairs", "block_size", "seed", "half"]
    with pytest.raises(FileExistsError):
        RR.stage_heads(corpus, out, ["image"])
    with pytest.raises(FileExistsError):
        RR.stage_bank(corpus, out)


def test_the_chain_through_train_and_tau_on_the_synthetic_corpus(corpus, tmp_path):
    out = tmp_path / "chain"
    RR.stage_heads(corpus, out, list(RR.A0))
    RR.stage_bank(corpus, out)
    RR.stage_train(corpus, out)
    pkl = pickle.loads((out / "rb_reader_A0.pkl").read_bytes())
    rjson = json.loads((out / "rb_reader_A0.json").read_text())
    assert rjson["provenance"]["pkl_sha256"] == R.sha256_file(out / "rb_reader_A0.pkl")
    assert rjson["rule_sha256"] == RR.C.RULE_SHA and rjson["smoke"] is False
    assert pkl["config"] == "A0"
    # the pickle is what r9_bundle.load_readers' checks need
    import r9_bundle as B
    B.check_readers(pkl)
    F = synthetic_f42(SMALL["n_tau_values"] // 2)
    RR.stage_tau(corpus, out, F, tau_checks(corpus))
    tau = json.loads((out / "rc_tau.json").read_text())
    assert len(tau["taus"]) == 4 and tau["n_values"] == SMALL["n_tau_values"]
    assert json.loads((out / "rc_tau.json.record.json").read_text())["keys_made"] == ["taus"]
    with pytest.raises(FileExistsError):
        RR.stage_tau(corpus, out, F, tau_checks(corpus))


def test_train_needs_the_heads_of_this_halves_and_banks(corpus, tmp_path):
    out = tmp_path / "missing"
    RR.stage_heads(corpus, out, ["image"])
    RR.stage_bank(corpus, out)
    with pytest.raises(AssertionError, match="missing.*rb_heads_affect"):
        RR.stage_train(corpus, out)


def test_heads_of_other_halves_are_refused_by_train(corpus, tmp_path):
    out = tmp_path / "otherhalves"
    RR.stage_heads(corpus, out, list(RR.A0))
    RR.stage_bank(corpus, out)
    other = SimpleNamespace(**{**vars(corpus), "halves_sha256": "f" * 64})
    with pytest.raises(AssertionError, match="halves"):
        RR.stage_train(other, out)


# ---------------------------------------------------------------- fix round: review of ticket 09

def fake_parts(monkeypatch, backbone, **paths):
    real = RR.parts
    monkeypatch.setattr(RR, "parts", lambda b: {**real(b), **paths} if b == backbone else real(b))


def tau_checks(corpus, **over):
    """What `seed42_features` hands to `stage_tau`: the groupings the n6 heads were fit on, and the head inputs."""
    chk = {"head_groupings": {"partitions_path": str(corpus.sources["image"][0]),
                              "partitions_sha256": corpus.sources["image"][1],
                              "label_sha256": {g: RR.rb.sha_array(corpus.lab[g]) for g in RR.A0}},
           "head_input_sha256": {"/inputs/partitions.npz": corpus.sources["image"][1], "/inputs/prepare.npz": "ab" * 32},
           "upstream_posterior_sha256": {"/inputs/n6_posteriors.npz": "cd" * 32,
                                         "/inputs/step1_heads_style.npz": "ef" * 32}}
    chk.update(over)
    return chk


def trained(corpus, out):
    RR.stage_heads(corpus, out, list(RR.A0))
    RR.stage_bank(corpus, out)
    RR.stage_train(corpus, out)


# L1: the default source path of a new backbone

def test_a_new_backbone_without_upstream_reads_its_own_partitions(tmp_path, monkeypatch):
    part = tmp_path / "own" / "partitions.npz"
    part.parent.mkdir()
    np.savez(part, image=np.arange(3), caption=np.arange(3))
    write_part_record(part, "qwen3vl_emb_2b")
    fake_parts(monkeypatch, "qwen3vl_emb_2b", partitions=part)
    src = RR.source_pins("qwen3vl_emb_2b")
    for name in ("image", "caption"):
        assert src[name][0] == part and src[name][1] == RR.sha256_file(part) and src[name][2] == name
        assert src[name][0] != RR.rb.SOURCES[name][0]                  # not CLIP's partitions


# L4: --upstream belongs to the backbone

def test_an_upstream_made_for_another_backbone_is_refused(tmp_path):
    np.savez(tmp_path / "partitions.npz", image=np.arange(3), caption=np.arange(3))
    write_part_record(tmp_path / "partitions.npz", "siglip2_so400m")
    with pytest.raises(AssertionError, match="siglip2_so400m"):
        RR.source_pins("qwen3vl_emb_2b", tmp_path)


def test_an_upstream_without_a_record_is_refused(tmp_path):
    np.savez(tmp_path / "partitions.npz", image=np.arange(3), caption=np.arange(3))
    with pytest.raises(AssertionError, match="record"):
        RR.source_pins("qwen3vl_emb_2b", tmp_path)


def test_clips_replay_upstream_needs_clips_record_and_clips_pinned_files_need_none(tmp_path):
    np.savez(tmp_path / "partitions.npz", image=np.arange(3), caption=np.arange(3))
    write_part_record(tmp_path / "partitions.npz", "clip_b32")
    assert RR.source_pins("clip_b32", tmp_path)["image"][0] == tmp_path / "partitions.npz"
    write_part_record(tmp_path / "partitions.npz", "qwen3vl_emb_2b")
    with pytest.raises(AssertionError, match="clip_b32"):
        RR.source_pins("clip_b32", tmp_path)
    assert RR.source_pins("clip_b32")["image"][0] == RR.rb.SOURCES["image"][0]


# L6: the standard heads' accuracies are CLIP's

@pytest.mark.parametrize("backbone", ["clip_b32", "qwen3vl_emb_2b"])
def test_clips_standard_accuracies_are_written_for_clip_only(corpus, backbone):
    lab = RR.rto_labels(corpus.lab["image"], corpus.split.scorer_train, len(corpus.split.groups))
    _, prov = RR.fit_crossfit_heads({"img": corpus.data.img_features, "txt": corpus.data.txt_features}, lab,
                                    corpus.halves, "image", KK, n_rows=corpus.params["head_rows"],
                                    check_rows=corpus.params["check_rows"], backbone=backbone)
    for k in ("0", "1"):
        if backbone == "clip_b32":
            assert prov[k]["standard_heldout_accuracy"] == {"img": 92.7, "txt": 22.9}
            assert "standard_heldout_accuracy_unavailable" not in prov[k]
        else:
            assert prov[k]["standard_heldout_accuracy"] is None
            assert "CLIP" in prov[k]["standard_heldout_accuracy_unavailable"]


# M1: tau uses the groupings the readers were trained on

def test_tau_refuses_groupings_other_than_the_ones_the_readers_were_trained_on(corpus, tmp_path):
    out = tmp_path / "m1"
    trained(corpus, out)
    F = synthetic_f42(SMALL["n_tau_values"] // 2)
    other = tau_checks(corpus)
    other["head_groupings"]["label_sha256"]["image"] = RR.rb.sha_array((corpus.lab["image"] + 1) % KK)
    with pytest.raises(AssertionError, match="image.*labels"):
        RR.stage_tau(corpus, out, F, other)
    assert not (out / "rc_tau.json").exists()
    RR.stage_tau(corpus, out, F, tau_checks(corpus))                    # the same groupings: accepted


def test_tau_refuses_a_partitions_file_other_than_the_heads(corpus, tmp_path):
    out = tmp_path / "m1b"
    trained(corpus, out)
    F = synthetic_f42(SMALL["n_tau_values"] // 2)
    other = tau_checks(corpus)
    other["head_groupings"]["partitions_sha256"] = "0" * 64
    with pytest.raises(AssertionError, match="partitions"):
        RR.stage_tau(corpus, out, F, other)
    assert not (out / "rc_tau.json").exists()


# L-a: the heads files on disk are the ones `train` read

def redo_heads_with_other_image_labels(corpus, out, tmp_path):
    """Review of ticket 09, L-a: after `train`, the image and caption heads and their records are deleted and made
    again from another partitions file (a rerun of ticket 07; the image labels differ). Returns the tau checks that
    match the new heads."""
    for name in ("rb_heads_image.npz", "rb_heads_image.npz.record.json",
                 "rb_heads_caption.npz", "rb_heads_caption.npz.record.json"):    # both come from partitions.npz
        (out / name).unlink()
    new_labels = (corpus.lab["image"] + 1) % KK
    path = tmp_path / "partitions_other.npz"
    np.savez(path, affect=corpus.lab["affect"], image=new_labels, caption=corpus.lab["caption"],
             local_groups=corpus.halves["g_st"])
    sha = RR.sha256_file(path)
    env2 = SimpleNamespace(**{**vars(corpus), "sources": {**corpus.sources, "image": (path, sha, "image"),
                                                           "caption": (path, sha, "caption")}})
    RR.stage_heads(env2, out, ["image", "caption"])
    other = tau_checks(corpus)
    other["head_groupings"].update(partitions_path=str(path), partitions_sha256=sha)
    other["head_groupings"]["label_sha256"]["image"] = RR.rb.sha_array(new_labels)
    return other


def test_tau_refuses_heads_made_again_after_train(corpus, tmp_path):
    out = tmp_path / "la"
    trained(corpus, out)
    other = redo_heads_with_other_image_labels(corpus, out, tmp_path)
    with pytest.raises(AssertionError, match="rb_heads_image.npz is not the file the readers were trained on"):
        RR.stage_tau(corpus, out, synthetic_f42(SMALL["n_tau_values"] // 2), other)
    assert not (out / "rc_tau.json").exists()


def test_the_trained_heads_guard_is_the_only_stop_for_heads_made_again(corpus, tmp_path):
    out = tmp_path / "la_mut"
    trained(corpus, out)
    other = redo_heads_with_other_image_labels(corpus, out, tmp_path)
    mut = mutant(tmp_path, "r9_recipe_reader.py", "heads_are_the_trained_ones")
    inputs = mut.check_tau_groupings(corpus, out, other["head_groupings"])      # without the guard the new heads pass
    assert set(Path(k).name for k in inputs) == {f"rb_heads_{g}.npz" for g in RR.A0}


def test_tau_accepts_the_heads_the_readers_were_trained_on_after_the_folder_moved(corpus, tmp_path):
    out = tmp_path / "la_moved"
    trained(corpus, out)
    moved = tmp_path / "la_moved_elsewhere"
    out.rename(moved)                                                   # the pickle's record names the old folder
    inputs = RR.check_tau_groupings(corpus, moved, tau_checks(corpus)["head_groupings"])
    assert sorted(Path(k).name for k in inputs) == sorted(f"rb_heads_{g}.npz" for g in RR.A0)


# L-b: the groupings the tau stage compares (`head_groupings`) and the order in `seed42_features` (L-d)

def test_head_groupings_hash_each_grouping_at_the_scorer_train_rows():
    rng = np.random.default_rng(5)
    n = 50
    st = np.arange(3, 43, dtype=np.int64)
    labels = {g: np.where(np.isin(np.arange(n), st), rng.integers(0, KK, size=n), -1).astype(np.int64)
              for g in RR.A0}
    pins = {RR.H.E2_PARTITIONS_REL: (Path("/x/partitions.npz"), "ab" * 32)}
    got = RR.head_groupings(pins, labels, st)
    assert got["partitions_path"] == "/x/partitions.npz" and got["partitions_sha256"] == "ab" * 32
    assert list(got["label_sha256"]) == list(RR.A0)
    for g in RR.A0:
        assert got["label_sha256"][g] == RR.rb.sha_array(labels[g][st]), g
    assert len(set(got["label_sha256"].values())) == len(RR.A0)           # one hash per grouping, not one for all
    labels["caption"] = labels["caption"].copy()
    labels["caption"][st[0]] = (labels["caption"][st[0]] + 1) % KK
    again = RR.head_groupings(pins, labels, st)
    assert again["label_sha256"]["caption"] != got["label_sha256"]["caption"]
    assert again["label_sha256"]["image"] == got["label_sha256"]["image"]


class Stop(Exception):
    pass


def fake_seed42(monkeypatch, corpus, calls):
    """`seed42_features` with the heavy parts replaced: rows, pins, labels, and a refit that stops the run."""
    labels = {g: corpus.lab[g] for g in RR.A0}
    pins = {RR.H.E2_PARTITIONS_REL: (Path("/x/partitions.npz"), "ab" * 32)}
    monkeypatch.setattr(RR, "rows_from", lambda data, split: SimpleNamespace(data=data, split=split))
    monkeypatch.setattr(RR.H, "recipe_head_pins", lambda backbone, root=None: pins)
    monkeypatch.setattr(RR.H, "head_labels", lambda groups, st, backbone, input_pins=None: labels)

    def fit(*args, **kw):
        calls.append("fit")
        raise Stop("fit")
    monkeypatch.setattr(RR.H, "fit_heads_r9", fit)
    return pins, labels


def test_seed42_features_hands_the_groupings_to_the_precheck_before_the_refit(corpus, monkeypatch):
    calls = []
    pins, labels = fake_seed42(monkeypatch, corpus, calls)
    seen = []

    def precheck(groupings):
        calls.append("precheck")
        seen.append(groupings)
        raise AssertionError("stop here")
    with pytest.raises(AssertionError, match="stop here"):
        RR.seed42_features("qwen3vl_emb_2b", None, corpus.data, corpus.split, precheck=precheck)
    assert calls == ["precheck"]                                         # the refit never started
    assert seen == [RR.head_groupings(pins, labels, corpus.split.scorer_train)]
    calls.clear()
    with pytest.raises(Stop):                                            # without a precheck the refit starts
        RR.seed42_features("qwen3vl_emb_2b", None, corpus.data, corpus.split)
    assert calls == ["fit"]


def test_the_tau_stage_checks_the_groupings_before_the_refit(corpus, tmp_path, monkeypatch):
    got = {}
    monkeypatch.setattr(RR, "build_env", lambda backbone, upstream, stage: corpus)
    monkeypatch.setattr(RR, "preflight", lambda *a, **k: None)
    monkeypatch.setattr(RR.H, "require_threads", lambda: None)

    def fake(backbone, upstream, data, split, precheck=None):
        got["precheck"] = precheck
        return "F42", "checks"
    monkeypatch.setattr(RR, "seed42_features", fake)
    monkeypatch.setattr(RR, "stage_tau", lambda env, out, F, checks: got.update(stage=(F, checks)))
    out = tmp_path / "ld"
    trained(corpus, out)
    assert RR.main(["--backbone", "qwen3vl_emb_2b", "--out", str(out), "--stage", "tau"]) == 0
    assert got["stage"] == ("F42", "checks")
    got["precheck"](tau_checks(corpus)["head_groupings"])                # the right groupings pass
    wrong = tau_checks(corpus)["head_groupings"]
    wrong["label_sha256"]["caption"] = "0" * 64
    with pytest.raises(AssertionError, match="caption.*labels"):
        got["precheck"](wrong)


# L-c: `rows_from` equals `r9_common.load_rows`

def test_rows_from_equals_load_rows_on_what_tau_uses(corpus, monkeypatch):
    import src.data.artelingo_splits as AS
    n = N
    rng = np.random.default_rng(11)
    lab = {"emotion": rng.integers(0, 3, size=n), "style": rng.integers(0, 4, size=n),
           "genre": rng.integers(0, 2, size=n)}
    lab["genre"][::13] = -1                                              # some rows unlabelled
    monkeypatch.setattr(AS, "artelingo_aspect_labels", lambda data: {k: v.copy() for k, v in lab.items()})
    monkeypatch.setattr(R, "N_ROWS", n)
    monkeypatch.setattr(R, "VALUE_COUNTS", {"emotion": 3, "style": 4, "genre": 2})
    monkeypatch.setattr(R, "MIN_PAINTINGS", 5)
    monkeypatch.setattr(R, "load_data", lambda backbone="clip_b32": corpus.data)
    monkeypatch.setattr(R, "load_split", lambda data: corpus.split)
    want = R.load_rows("qwen3vl_emb_2b")
    got = RR.rows_from(corpus.data, corpus.split)
    assert got.data is want.data and got.split is want.split
    assert list(got.labels) == list(want.labels)
    for k in want.labels:
        assert np.array_equal(got.labels[k], want.labels[k]), k
    assert got.value_sets == want.value_sets
    assert all(len(v) for v in got.value_sets.values())                  # a real comparison, not two empty sets


# L3: the tau record lists the head inputs, with stable keys

def test_tau_record_lists_the_head_inputs_with_absolute_keys(corpus, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = Path("relative_out")                                          # --out given relative to the working folder
    trained(corpus, out)
    RR.stage_tau(corpus, out, synthetic_f42(SMALL["n_tau_values"] // 2), tau_checks(corpus))
    rec = json.loads((out / "rc_tau.json.record.json").read_text())
    shas = rec["input_sha256"]
    assert all(Path(k).is_absolute() for k in shas), [k for k in shas if not Path(k).is_absolute()]
    pkl = tmp_path / "relative_out" / "rb_reader_A0.pkl"
    assert shas[str(pkl)] == RR.sha256_file(pkl)
    for k, v in {**tau_checks(corpus)["head_input_sha256"], **tau_checks(corpus)["upstream_posterior_sha256"]}.items():
        assert shas[k] == v
    for g in RR.A0:                                                     # the heads files the groupings were checked on
        head = tmp_path / "relative_out" / f"rb_heads_{g}.npz"
        assert shas[str(head)] == RR.sha256_file(head)
    for rec_path in out.glob("*.record.json"):                          # every record of the run: no relative key
        assert all(Path(k).is_absolute() for k in json.loads(rec_path.read_text())["input_sha256"]), rec_path


def test_a_relative_upstream_becomes_an_absolute_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("up").mkdir()
    np.savez("up/partitions.npz", image=np.arange(3), caption=np.arange(3))
    write_part_record(Path("up/partitions.npz"), "qwen3vl_emb_2b")
    assert RR.source_pins("qwen3vl_emb_2b", Path("up"))["image"][0] == tmp_path / "up" / "partitions.npz"


# L2: guards without a test

def test_train_refuses_heads_and_banks_made_from_different_labels(corpus, tmp_path):
    out = tmp_path / "labels"
    RR.stage_heads(corpus, out, list(RR.A0))
    RR.stage_bank(corpus, out)
    rec_path = out / "rb_bank_A0_half1.npz.record.json"
    rec = json.loads(rec_path.read_text())
    rec["label_sha256"]["caption"] = "0" * 64
    rec_path.write_text(json.dumps(rec))
    with pytest.raises(AssertionError, match="rb_heads_caption and bank half 1.*different caption labels"):
        RR.stage_train(corpus, out)


def test_a_part_made_for_another_backbone_is_refused(corpus, tmp_path):
    out = tmp_path / "backbone"
    RR.stage_heads(corpus, out, list(RR.A0))
    RR.stage_bank(corpus, out)
    other = SimpleNamespace(**{**vars(corpus), "backbone": "siglip2_so400m"})
    with pytest.raises(AssertionError, match="made for qwen3vl_emb_2b, not siglip2_so400m"):
        RR.stage_train(other, out)


def fake_stored(tmp_path, monkeypatch, *, change=None, coef="same"):
    """A stand-in for the heads, `H.predict` and the upstream posterior files of a new backbone."""
    rng = np.random.default_rng(0)
    selection = np.arange(0, 70, 7)
    post = {h: {m: rng.random((70, 4)).astype(np.float32) for m in ("img", "txt")} for h in RR.H.HEADS}
    monkeypatch.setattr(RR.H, "predict", lambda heads, data, rows: post)
    monkeypatch.setattr(RR.H, "coef_sha256", lambda heads: {"affect": {"img": "aa"}})
    for rel, keys in RR.H.TARGETS.items():
        arrays = {key: post[h][m][selection].copy() for key, (h, m) in keys.items()}
        if change and rel == change[0]:
            arrays[change[1]][0, 0] += np.float32(1e-3)
        path = tmp_path / RR.parts("qwen3vl_emb_2b")["n6_posteriors" if rel == RR.H.N6_REL else "step1_heads_style"].name
        np.savez(path, selection=selection, **arrays)
        if rel == RR.H.N6_REL:
            Path(str(path) + ".record.json").write_text(json.dumps(
                {"coef_sha256": {"affect": {"img": "aa" if coef == "same" else "bb"}}}))
    return selection


def test_stored_posteriors_equal_the_refit_heads(tmp_path, monkeypatch):
    selection = fake_stored(tmp_path, monkeypatch)
    out = RR.check_stored_posteriors(object(), object(), selection, "qwen3vl_emb_2b", tmp_path)
    assert len(out["items"]) == 8 and all(v["equal"] for v in out["items"].values())


@pytest.mark.parametrize("rel,key", [(RR.H.N6_REL, "image__img"), (RR.H.CSD_HEADS_REL, "style_csd__txt")])
def test_a_changed_stored_posterior_is_refused(tmp_path, monkeypatch, rel, key):
    selection = fake_stored(tmp_path, monkeypatch, change=(rel, key))
    with pytest.raises(AssertionError, match=key):
        RR.check_stored_posteriors(object(), object(), selection, "qwen3vl_emb_2b", tmp_path)


def test_changed_n6_coefficients_are_refused_for_a_new_backbone(tmp_path, monkeypatch):
    selection = fake_stored(tmp_path, monkeypatch, coef="other")
    with pytest.raises(AssertionError, match="coefficients"):
        RR.check_stored_posteriors(object(), object(), selection, "qwen3vl_emb_2b", tmp_path)


# ---------------------------------------------------------------- bookkeeping: guards and the mutation helper

def mutant(tmp_path, module, guard):
    """Import a copy of ``module`` (file name in HERE) with its `# guard:<guard>` statements replaced by `pass`."""
    src = (HERE / module).read_text()
    lines = src.splitlines(keepends=True)
    hits = [n for n in ast.walk(ast.parse(src))
            if isinstance(n, (ast.Expr, ast.Assign, ast.If, ast.Assert)) and f"# guard:{guard}" in lines[n.lineno - 1]]
    assert hits, f"no statement of {module} carries # guard:{guard}"
    for n in hits:
        first = lines[n.lineno - 1]
        lines[n.lineno - 1] = first[:len(first) - len(first.lstrip())] + "pass\n"
        for i in range(n.lineno, n.end_lineno):
            lines[i] = "\n"
    path = tmp_path / f"{Path(module).stem}_mut{next(_COUNT)}.py"
    path.write_text("".join(lines))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    saved = list(sys.path)
    sys.modules[path.stem] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(path.stem, None)
        sys.path[:] = saved
    return mod


def test_every_marked_guard_has_a_mutation_test():
    names = set(re.findall(r"# guard:([a-z_0-9]+)", (HERE / "r9_recipe_reader.py").read_text()))
    covered = set(re.findall(r'mutant\(tmp_path, "r9_recipe_reader\.py", "([a-z_0-9]+)"\)', Path(__file__).read_text()))
    assert names and names == covered, (names - covered, covered - names)


def test_the_module_pins_the_frozen_halves_to_the_pins_file():
    pins = json.loads((HERE / "pins_clip.json").read_text())["parts"]
    assert RR.HALVES_SHA256 == pins["rb_halves"]["sha256"]
    for g in RR.GROUPINGS:
        assert RR.clip_pinned_sha(f"rb_heads_{g}") == pins[f"rb_heads_{g}"]["sha256"]


def test_recipe_roles_cover_the_files_written():
    roles = set(RR.parts("qwen3vl_emb_2b"))
    assert {f"rb_heads_{g}" for g in RR.GROUPINGS} | {"rb_bank_A0_half0", "rb_bank_A0_half1", "rb_reader_A0",
                                                      "rb_reader_A0_json", "rc_tau"} <= roles
    assert "rb_halves" in __import__("r9_backbone").FROZEN_ROLES


# ---------------------------------------------------------------- slow: CLIP's image heads against the pinned file

@pytest.mark.skipif(os.environ.get("R9_SLOW") != "1", reason="refits CLIP's heads on the real data (about 2 minutes)")
def test_clip_image_heads_equal_the_pinned_file(tmp_path):
    rc = RR.main(["--backbone", "clip_b32", "--out", str(tmp_path), "--stage", "heads", "--grouping", "image"])
    assert rc == 0
    got, want = np.load(tmp_path / "rb_heads_image.npz"), np.load(RR.parts("clip_b32")["rb_heads_image"])
    assert set(got.files) == set(want.files)
    for k in want.files:
        assert got[k].dtype == want[k].dtype and np.array_equal(got[k], want[k]), k
