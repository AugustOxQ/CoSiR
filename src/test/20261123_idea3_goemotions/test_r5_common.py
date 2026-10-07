"""Tests of r5_common.py and r5_guard.py (rule DECISION_RULE.md of this folder: §10 list A item 1 and D11's guard).
Synthetic only: no bundle is built, no GoEmotions model is run, no head is fitted, no seed 52 to 54 file is opened.
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python -m pytest -q -p no:cacheprovider test_r5_common.py
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r5_common as R5  # noqa: E402
import r5_guard as G  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_guard(monkeypatch):
    G._reset_for_tests()
    saved = dict(R5.RB3._HEADS)
    R5.RB3._HEADS.clear()                       # each test sets up the cache it needs
    monkeypatch.setattr(R5, "N_ROWS", 40)       # synthetic sizes for ge_from_file
    monkeypatch.setattr(R5, "N_SELECTION", 20)
    monkeypatch.setattr(G, "_GE_FPS", set())
    yield
    R5.RB3._HEADS.clear()
    R5.RB3._HEADS.update(saved)
    G._reset_for_tests()


# ---------------------------------------------------------------- r5_common: rule, inputs, modules

def test_rule_asserts_pass():
    R5.assert_rule()
    assert R5.RULE_SHA == "19e59fc7220c05b630f4773a94578aa3858d29853dbf7d438455e1ee973d735e"
    assert R5.R4C.RULE_SHA == "cf11a8739e963995d6674648845ac9b0957632583a364cb099c354810f8e911b"
    assert R5.R3.RULE_SHA == "2d311dbeac1a561fc2a719b730075c30819f4f7f3a592041cb64b65821dc5925"


def test_rule_tamper_stops(monkeypatch):
    monkeypatch.setattr(R5, "RULE_SHA", "0" * 64)
    with pytest.raises(SystemExit):
        R5.assert_rule()


def test_inputs_all_pass():
    out = R5.assert_inputs(list(R5.INPUTS))
    assert set(out) == set(R5.INPUTS) and len(R5.INPUTS) == 38


def test_inputs_delegate_to_earlier_rounds():
    out = R5.assert_inputs(["20261121_round3_affect_gate/r3_stats.py", "20261117_reader_fix_csd/results/rc_tau.json",
                            "20261116_grouping_step1_style/run_step1.py"])
    assert len(out) == 3
    with pytest.raises(KeyError):
        R5.assert_inputs(["not/an/input.py"])


def test_patched_input_hash_stops(monkeypatch):
    name = "src/data/affect.py"
    monkeypatch.setitem(R5.INPUTS, name, "0" * 64)
    monkeypatch.setattr(R5, "_CHECKED", {})
    with pytest.raises(SystemExit):
        R5.assert_inputs([name])
    # an earlier round's table is checked too
    monkeypatch.setitem(R5.R4C.INPUTS, "20261121_round3_affect_gate/r3_stats.py", "0" * 64)
    monkeypatch.setattr(R5.R4C, "_CHECKED", {})
    with pytest.raises(SystemExit):
        R5.assert_inputs(["20261121_round3_affect_gate/r3_stats.py"])


def test_modules_pass_and_cover_the_imports():
    R5.assert_modules()
    want = {"r3_common", "r3_bundle", "r3_fusion", "r3_stats", "r4_common", "r4_bundle", "r4_stats",
            "run_told_oracle", "run_checks", "run_n6", "run_gonogo", "rc_core", "common", "rb_build", "rb_eval",
            "rb_features", "r2_fusion"}
    assert want == {Path(k).stem for k in R5.MODULE_SHA}


def test_patched_module_hash_stops(monkeypatch):
    k = next(k for k in R5.MODULE_SHA if k.endswith("r3_bundle.py"))
    patched = dict(R5.MODULE_SHA)
    patched[k] = "0" * 64
    monkeypatch.setattr(R5, "MODULE_SHA", patched)
    with pytest.raises(SystemExit):
        R5.assert_modules()


def test_module_hashes_agree_with_earlier_tables():
    for rel, sha in R5.MODULE_SHA.items():
        sources = [t[rel] for t in (R5.R3.INPUTS, R5.R4C.INPUTS) if rel in t]
        if "src/test/" + rel in R5.INPUTS:
            sources.append(R5.INPUTS["src/test/" + rel])
        assert sources, rel
        assert all(s == sha for s in sources), rel


def test_imports_resolve_to_earlier_rounds():
    assert Path(R5.R4C.__file__).resolve().parent == R5.TEST / "20261122_round4_aff_vetoes"
    assert Path(R5.RB4.__file__).resolve().parent == R5.TEST / "20261122_round4_aff_vetoes"
    assert Path(R5.RS4.__file__).resolve().parent == R5.TEST / "20261122_round4_aff_vetoes"
    assert Path(R5.RF3.__file__).resolve().parent == R5.TEST / "20261121_round3_affect_gate"
    assert Path(R5.RTO.__file__).resolve() == R5.TEST / "20261111_community_told_oracle/run_told_oracle.py"
    assert Path(R5.RCHK.__file__).resolve() == R5.TEST / "20261108_new_method_quick_checks/run_checks.py"
    assert Path(R5.N6.__file__).resolve() == R5.TEST / "20261108_new_method_quick_checks/run_n6.py"
    assert Path(R5.rc_core.__file__).resolve() == R5.TEST / "20261117_reader_fix_csd/rc_core.py"


# ---------------------------------------------------------------- seed guard

def test_test_seeds_equal_round4s():
    assert R5.TEST_SEEDS == (52, 53, 54) and R5.R3.TEST_SEEDS == (52, 53, 54)
    assert tuple(R5.R4C.TEST_SEEDS) == R5.TEST_SEEDS == tuple(R5.R3.TEST_SEEDS)
    assert R5.SMOKE_SEEDS == (9001, 9002, 9003)
    assert R5.EARLIER_SEEDS == (42, 43, 45, 47, 48, 49, 50, 51)


def test_seed_guard():
    for s in (42, 52, 53, 54):
        R5.check_seed(s, False)
    for s in (9001, 9002, 9003):
        R5.check_seed(s, True)
    for s in (49, 55, 43):
        with pytest.raises(ValueError):
            R5.check_seed(s, False)
    for s in (9001, 9002, 9003):
        with pytest.raises(ValueError):
            R5.check_seed(s, False)
    with pytest.raises(ValueError):
        R5.check_seed(52, True)
    with pytest.raises(ValueError):
        R5.check_seed(42, True)


def test_seed_guard_refuses_when_round3_drifts(monkeypatch):
    monkeypatch.setattr(R5.R3, "TEST_SEEDS", (49, 50, 51))
    with pytest.raises(ValueError):
        R5.check_seed(52, False)


# ---------------------------------------------------------------- constants

def test_constants():
    assert R5.CANDIDATES == ("G-T", "G-TF") and R5.TIE_BAND_UNITS == 24
    assert (R5.GOEMO_BATCH, R5.GOEMO_MAXLEN, R5.REG_SAMPLE, R5.SPOT_SAMPLE) == (256, 64, (5, 2048), (6, 1024))
    assert (R5.GOEMO_TOL, R5.GE_MAX_ITER, R5.GE_FALLBACK_MAX_ITER) == (1e-4, 300, 3000)
    assert R5.GOEMO_FILE_SHA is None and R5.GE_POST_SHA is None
    assert R5.AUC_AFF == 0.7870951145887375 and R5.GROUP_LIFT == 2.7111312041209863
    assert R5.AFF_MINUS_BP1 == (0.33162434895833337, (0.048231414333532084, 0.6246158772581268))
    assert R5.AFF_EITHER_PER_GAIN == 0.5238718116415958
    assert R5.AFF_EITHER_PER_GAIN == pytest.approx(-R5.AFF_ITEM1["either"] / R5.AFF_ITEM1["gain_statistic"][0], rel=1e-14)
    assert R5.CLIP_HEAD_RECORD["heldout_accuracy"] == {"img": 9.81, "txt": 35.72}


def test_targets_agree_with_earlier_rounds():
    for k, v in R5.AFF_ITEM1.items():
        if k in R5.R3.AFF_BRAINSTORM:
            assert R5.R3.AFF_BRAINSTORM[k] == v, k
    assert R5.AFF_CELLS == R5.R3.AFF_CELLS
    told = json.loads((R5.TEST / "20261111_community_told_oracle/results/told_oracle.json").read_text())["arms"]["L"]
    assert told["head"] == {k: R5.CLIP_HEAD_RECORD[k] for k in told["head"]}
    assert told["pairs"]["groups"]["lift"]["lift"] == R5.GROUP_LIFT
    dev = json.loads((R5.TEST / "20261122_round4_aff_vetoes/results/dev_seed42.json").read_text())
    assert dev["beside_aff"]["AFF_minus_Bprime_A1"]["point"] == R5.AFF_MINUS_BP1[0]
    assert dev["beside_aff"]["AFF_minus_Bprime_A1"]["ci95"] == list(R5.AFF_MINUS_BP1[1])


# ---------------------------------------------------------------- helpers

def test_helpers(tmp_path, monkeypatch):
    monkeypatch.setattr(R5, "RESULTS", tmp_path / "results")
    d = R5.res_dir(False)
    assert d == tmp_path / "results" and R5.res_dir(True) == tmp_path / "results" / "smoke"
    p = d / "x.json"
    rec = R5.write_json_once(p, {"a": 1.5}, False)
    assert json.loads(p.read_text())["provenance"]["rule_sha256"] == R5.RULE_SHA and rec["a"] == 1.5
    with pytest.raises(SystemExit):
        R5.write_json_once(p, {"a": 2}, False)
    with pytest.raises(SystemExit):
        R5.refuse_existing([p], False)
    R5.refuse_existing([p], True)
    R5.write_json_once(tmp_path / "results" / "smoke" / "y.json", {"a": 1}, True)
    R5.write_json_once(tmp_path / "results" / "smoke" / "y.json", {"a": 2}, True)
    with pytest.raises(Exception):
        R5.write_json_once(tmp_path / "results" / "z.json", {"a": float("nan")}, False)
    import re
    assert re.fullmatch(r"\d{4}-\d\d-\d\d \d\d:\d\d", R5.now_ams())
    f = tmp_path / "f.bin"
    f.write_bytes(b"abc")
    assert R5.sha256_file(f) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


# ---------------------------------------------------------------- r5_guard

N, K = 40, 41


def _post(seed=0):
    rng = np.random.default_rng(seed)
    Q = np.full((N, K), np.nan, np.float32)
    Q[5:25] = rng.dirichlet(np.ones(K), 20).astype(np.float32)
    return Q


def _cache(Q, key=60000):
    R5.RB3._HEADS[key] = {"post": {"affect": {"img": _post(1), "txt": Q}}}


def _bundle(Q, cached=True):
    if cached:
        _cache(Q)
    return SimpleNamespace(post={"affect": {"img": _post(1), "txt": Q}})


def _ge_file(tmp_path, n_sel=20, rows=None, classes=None, sums=1.0, name="ge.npz", nan=False):
    rng = np.random.default_rng(3)
    ps = (rng.dirichlet(np.ones(K), n_sel) * sums).astype(np.float32)
    if nan:
        ps[3, 4] = np.nan
    rows = np.arange(5, 5 + n_sel, dtype=np.int64) if rows is None else rows
    classes = np.arange(K) if classes is None else classes
    p = tmp_path / name
    np.savez(p, post_sel=ps, rows=rows, classes=classes)
    return p, ps, rows


def test_clip_from_bundle_is_a_read_only_view():
    Q = _post()
    b = _bundle(Q)
    pl = G.clip_from_bundle(b)
    assert pl.kind == "clip" and np.shares_memory(pl.Q, Q) and not pl.Q.flags.writeable
    assert Q.flags.writeable   # the bundle's own array is untouched
    with pytest.raises(ValueError):
        pl.Q[5, 0] = 0.0
    assert G.require(pl, "x") is pl
    Q[6, 3] += 0.25    # the owner mutates after the fact: the fingerprint no longer matches
    with pytest.raises(G.GuardError):
        G.require(pl, "x")


def test_clip_from_bundle_needs_an_equal_cached_head():
    Q = _post()
    _cache(Q.copy())                                       # an equal array (by value) passes
    assert G.clip_from_bundle(_bundle(Q, cached=False)).kind == "clip"
    R5.RB3._HEADS.clear()
    _cache(_post(7))                                       # a different array refuses
    with pytest.raises(G.GuardError):
        G.clip_from_bundle(_bundle(Q, cached=False))
    R5.RB3._HEADS.clear()                                  # an empty cache refuses
    with pytest.raises(G.GuardError):
        G.clip_from_bundle(_bundle(Q, cached=False))


def test_placements_are_minted_only_by_the_two_constructors(tmp_path, monkeypatch):
    with pytest.raises(G.GuardError):
        G.Placement("clip", _post(), "0" * 64)
    p, ps, rows = _ge_file(tmp_path)
    monkeypatch.setattr(R5, "GE_POST_SHA", R5.sha256_file(p))
    ge = G.ge_from_file(p)
    with pytest.raises(G.GuardError):                      # forged clip carrying the GE array
        G.Placement("clip", ge.Q, G.fingerprint(ge.Q))
    with pytest.raises(G.GuardError):                      # forged ge skipping the SHA check
        G.Placement("ge", ge.Q, "0" * 64)
    with pytest.raises(G.GuardError):                      # the bundle's Q being the GE array is not a clip
        G.clip_from_bundle(_bundle(np.array(ge.Q)))
    G.release(_regression(tmp_path))
    assert G.require(ge, "x") is ge
    assert G.require(G.clip_from_bundle(_bundle(_post())), "x").kind == "clip"


def test_clip_from_bundle_refuses_a_wrong_array():
    with pytest.raises(G.GuardError):
        G.clip_from_bundle(_bundle(_post().astype(np.float64)))
    with pytest.raises(G.GuardError):
        G.clip_from_bundle(_bundle(_post()[:, :40]))


def test_require_refuses_ge_before_release_and_accepts_clip(tmp_path, monkeypatch):
    p, ps, rows = _ge_file(tmp_path)
    monkeypatch.setattr(R5, "GE_POST_SHA", R5.sha256_file(p))
    ge = G.ge_from_file(p)
    clip = G.clip_from_bundle(_bundle(_post()))
    assert G.require(clip, "x") is clip
    with pytest.raises(G.GuardError):
        G.require(ge, "stack_G")
    with pytest.raises(G.GuardError):
        G.require("not a placement", "x")
    with pytest.raises(G.GuardError):
        G.Placement("other", _post(), "0" * 64)
    reg = _regression(tmp_path)
    G.release(reg)
    assert G.require(ge, "stack_G") is ge


def test_ge_from_file(tmp_path, monkeypatch):
    p, ps, rows = _ge_file(tmp_path)
    monkeypatch.setattr(R5, "GE_POST_SHA", R5.sha256_file(p))
    ge = G.ge_from_file(p)
    assert ge.kind == "ge" and ge.sha256 == R5.GE_POST_SHA
    assert ge.Q.dtype == np.float32 and ge.Q.shape == (N, K) and not ge.Q.flags.writeable
    assert np.array_equal(ge.Q[rows], ps) and np.isnan(ge.Q[:5]).all() and np.isnan(ge.Q[25:]).all()


def test_ge_from_file_refuses_wrong_or_unset_sha(tmp_path, monkeypatch):
    p, _, _ = _ge_file(tmp_path)
    with pytest.raises(G.GuardError):          # constant not committed yet (None)
        G.ge_from_file(p)
    monkeypatch.setattr(R5, "GE_POST_SHA", "0" * 64)
    with pytest.raises(G.GuardError):
        G.ge_from_file(p)


def test_ge_from_file_refuses_malformed_files(tmp_path, monkeypatch):
    cases = {
        "unsorted": dict(rows=np.arange(5, 25, dtype=np.int64)[::-1].copy()),
        "classes": dict(classes=np.arange(K)[::-1].copy()),
        "sums": dict(sums=0.9),
        "short": dict(n_sel=19, rows=np.arange(5, 24, dtype=np.int64)),
        "high_row": dict(rows=np.arange(30, 50, dtype=np.int64)),
        "negative_row": dict(rows=np.arange(-3, 17, dtype=np.int64)),
        "nan": dict(nan=True),
    }
    for name, kw in cases.items():
        p, _, _ = _ge_file(tmp_path, name=name + ".npz", **kw)
        monkeypatch.setattr(R5, "GE_POST_SHA", R5.sha256_file(p))
        with pytest.raises(G.GuardError):
            G.ge_from_file(p)


def _regression(tmp_path, mutate=None, name="results"):
    items = {str(i): {"passed": True, "note": "x"} for i in (1, 2, 3, 4)}
    rec = {"rule_sha256": R5.RULE_SHA, "items": items, "all_passed": True, "written": "x"}
    if mutate:
        mutate(rec)
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    p = d / "regression_check.json"
    p.write_text(json.dumps(rec))
    return p


def test_release_accepts_only_a_full_record(tmp_path):
    assert G.is_released() is False
    G.release(_regression(tmp_path))
    assert G.is_released() is True


@pytest.mark.parametrize("mutate", [
    lambda r: r["items"]["3"].update(passed=False),
    lambda r: r["items"].pop("2"),
    lambda r: r.update(rule_sha256="0" * 64),
    lambda r: r.pop("rule_sha256"),
    lambda r: r.update(all_passed=False),
    lambda r: r.pop("all_passed"),
    lambda r: r["items"]["4"].update(passed="yes"),
    lambda r: r["items"].update({"5": {"passed": True}}) or r["items"].pop("1"),
])
def test_release_refuses_bad_records(tmp_path, mutate):
    with pytest.raises(G.GuardError):
        G.release(_regression(tmp_path, mutate))
    assert G.is_released() is False


def test_release_refuses_missing_garbage_and_wrong_place(tmp_path):
    with pytest.raises(G.GuardError):
        G.release(tmp_path / "results" / "regression_check.json")
    d = tmp_path / "results"
    d.mkdir()
    (d / "regression_check.json").write_text("{not json")
    with pytest.raises(G.GuardError):
        G.release(d / "regression_check.json")
    good = _regression(tmp_path, name="elsewhere")
    with pytest.raises(G.GuardError):                    # not a results/ folder
        G.release(good)
    assert G.is_released() is False


def test_require_carry(tmp_path):
    d = tmp_path / "results"
    d.mkdir()
    p = d / "carry.json"
    with pytest.raises(G.GuardError):
        G.require_carry(p)
    p.write_text(json.dumps({"rule_sha256": "0" * 64}))
    with pytest.raises(G.GuardError):
        G.require_carry(p)
    p.write_text(json.dumps({"carry": {}}))
    with pytest.raises(G.GuardError):
        G.require_carry(p)
    p.write_text(json.dumps({"rule_sha256": R5.RULE_SHA, "carry": {"carried": None}}))
    G.require_carry(p)


def test_reset_for_tests(tmp_path):
    G.release(_regression(tmp_path))
    assert G.is_released()
    G._reset_for_tests()
    assert not G.is_released()
