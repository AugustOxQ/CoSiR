"""Tests of run_r9_stage0a.py (ticket 06; rule 3.1, 10 row 1): the stage 0a gate on hand-built records and on round 6's
real reference files, the runner's exit codes and the wiring of its item list.

Real files read: round 6's regression_seed42.json and seed42_pass_counts.json (SHA-256 asserted), round 3's
seed42_arrays.npz and AB's per_anchor_seed42.npz (their key lists and shapes). No feature, label or episode.
The full run (data, heads, bundle) is not run here; see the module docstring of run_r9_stage0a.py.

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python -m pytest -q -p no:cacheprovider \
        src/test/20261202_r9_scope/test_r9_stage0a.py
"""
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r9_common as R  # noqa: E402  (first, before anything that imports src)
import run_r9_stage0a as RS  # noqa: E402

import numpy as np  # noqa: E402

REGRESSION_SHA = "ca168eb628f6a51e9afe779b0194088d05b293417bac04617a0d8a428d3984be"   # rule 3.1
COUNTS_SHA = "5618c59a3907af67bbcced0f29e0db8462b12ca4fb2e40a69e4be537de29d5e5"       # rule 3.1


@pytest.fixture(scope="module")
def stored():
    reg = json.loads(RS.REFERENCES["regression_seed42.json"][0].read_text())
    counts = json.loads(RS.REFERENCES["seed42_pass_counts.json"][0].read_text())
    return SimpleNamespace(reg=reg, counts=counts)


def small_items():
    """A hand-built item list: a scalar, a string, a list, a hash and two arrays, as the runner's helpers make them."""
    return {
        "scalar": RS.item(19.136555989583336, 19.136555989583336),
        "bar_comparator": RS.item("B_prime", "B_prime"),
        "ci": RS.item([0.5, 0.25, 0.75], [0.5, 0.25, 0.75]),
        "sha": RS.item("ab" * 32, "ab" * 32),
        "arr64": RS.array_item(np.arange(6, dtype=np.float64), np.arange(6, dtype=np.float64)),
        "gate": RS.array_item(np.ones((4, 3), np.float32), np.ones((4, 3), np.float32)),
    }


# ---------------------------------------------------------------- the references

def test_reference_files_are_the_rules():
    assert RS.REFERENCES["regression_seed42.json"][1] == REGRESSION_SHA
    assert RS.REFERENCES["seed42_pass_counts.json"][1] == COUNTS_SHA
    for name, (path, sha) in RS.REFERENCES.items():
        assert path.parent == R.R6D / "results", name
        assert R.sha256_file(path) == sha, name


def test_run_specific_fields_are_the_rules_list():
    assert RS.RUN_SPECIFIC == ("rule_sha256", "runner_sha256", "module_sha256", "time", "input_sha256")


def test_the_stored_item_list(stored):
    items = stored.reg["items"]
    assert stored.reg["passed"] is True and stored.reg["n_items"] == len(items) == 131 and stored.reg["n_failed"] == 0
    assert sum(k.startswith("seed42_arrays/") for k in items) == 39
    assert sum(k.startswith("per_anchor_seed42/") for k in items) == 57
    assert len(items) - 39 - 57 == 35
    assert RS.gate_references(stored.reg, stored.counts) == []


# ---------------------------------------------------------------- the item gate

def test_equal_items_pass():
    items = small_items()
    assert RS.compare_items(items, copy.deepcopy(items)) == []


def test_real_items_equal_themselves(stored):
    assert RS.compare_items(copy.deepcopy(stored.reg["items"]), stored.reg["items"]) == []


def test_a_differing_scalar_fails():
    want, got = small_items(), small_items()
    got["scalar"] = RS.item(19.136555989583336, 19.136555989583340)
    diff = RS.compare_items(got, want)
    assert diff and diff[0].startswith("scalar")
    got = small_items()
    got["bar_comparator"] = RS.item("B_prime", "counterpart")
    assert RS.compare_items(got, want)[0].startswith("bar_comparator")
    got = small_items()
    got["sha"] = RS.item("ab" * 32, "ac" + "ab" * 31)
    assert RS.compare_items(got, want)[0].startswith("sha")


def test_a_scalar_that_is_not_equal_fails_even_when_the_record_agrees():
    """An item whose own comparison failed (equal False) fails the gate even if the stored record shows the same."""
    want = small_items()
    want["scalar"] = RS.item(1.0, 2.0)
    assert not want["scalar"]["equal"]
    assert RS.compare_items(copy.deepcopy(want), want) == ["scalar: equal is not true"]


@pytest.mark.parametrize("kind", ["dtype", "shape", "value"])
def test_a_differing_array_fails(kind):
    want = small_items()
    got = small_items()
    ref = np.arange(6, dtype=np.float64)
    if kind == "dtype":
        got["arr64"] = RS.array_item(ref.astype(np.float32), ref)
    elif kind == "shape":
        got["arr64"] = RS.array_item(ref[:5], ref)
    else:
        v = ref.copy()
        v[3] = np.nextafter(v[3], 10.0)
        got["arr64"] = RS.array_item(v, ref)
    diff = RS.compare_items(got, want)
    assert diff and diff[0].startswith("arr64"), diff
    assert len(diff) == 1                                          # the other items still agree


def test_an_array_that_differs_from_its_own_target_fails_against_a_matching_record():
    """The stored item says n_diff 0; ours says 1: the item dict differs (equal, n_diff and max_abs_diff)."""
    want = small_items()
    got = small_items()
    ref = np.arange(6, dtype=np.float64)
    bad = ref.copy()
    bad[0] += 1.0
    got["arr64"] = RS.array_item(bad, ref)
    assert not got["arr64"]["equal"] and got["arr64"]["n_diff"] == 1
    assert RS.compare_items(got, want)


def test_missing_and_extra_items_fail():
    want = small_items()
    got = small_items()
    del got["gate"]
    assert RS.compare_items(got, want) == ["gate: missing"]
    got = small_items()
    got["new"] = RS.item(1, 1)
    assert RS.compare_items(got, want) == ["new: not in the reference"]


def test_item_order_and_key_names_are_part_of_the_gate():
    want = small_items()
    got = {k: want[k] for k in reversed(list(want))}
    assert RS.compare_items(got, want) == ["item order differs from the reference"]


def test_item_gate_on_real_items_catches_one_changed_array(stored):
    got = copy.deepcopy(stored.reg["items"])
    key = "seed42_arrays/aff_gate__a"
    got[key]["got"] = "array (4, 12288) float64"                  # a dtype change in the record
    assert RS.compare_items(got, stored.reg["items"]) == [f"{key}: got differs"]


# ---------------------------------------------------------------- the pass-count gate

def test_equal_counts_pass(stored):
    assert RS.compare_counts(copy.deepcopy(stored.counts), stored.counts) == []


MUST_EQUAL = ("mode", "seeds", "n_episodes", "n_clusters", "checks", "holm_order", "secondary", "episodes_sha256",
              "coef_sha256")


def test_the_fields_that_must_be_equal_are_the_rules_list(stored):
    assert set(stored.counts) - set(RS.RUN_SPECIFIC) == set(MUST_EQUAL)


@pytest.mark.parametrize("field", MUST_EQUAL)
def test_a_differing_pass_count_field_fails(stored, field):
    got = copy.deepcopy(stored.counts)
    v = got[field]
    if field == "checks":
        v["P3"]["point"] = np.nextafter(v["P3"]["point"], 100.0)
    elif field == "secondary":
        v["S1"]["ci95"][0] += 1e-9
    elif field == "holm_order":
        v[0], v[1] = v[1], v[0]
    elif field == "seeds":
        got[field] = [43]
    elif field in ("n_episodes", "n_clusters"):
        got[field] = v + 1
    elif field == "mode":
        got[field] = "smoke"
    elif field == "episodes_sha256":
        v["42"]["emotion__style"] = "0" * 64
    else:
        v["csd"]["txt"] = "0" * 64
    diff = RS.compare_counts(got, stored.counts)
    assert diff and diff[0].startswith(field), diff


def test_a_nested_count_inside_checks_is_named(stored):
    got = copy.deepcopy(stored.counts)
    got["checks"]["P5"]["n"] += 1
    assert RS.compare_counts(got, stored.counts) == ["checks differs"]


@pytest.mark.parametrize("field", RS.RUN_SPECIFIC)
def test_a_run_specific_field_is_ignored(stored, field):
    got = copy.deepcopy(stored.counts)
    got[field] = "other" if field not in ("module_sha256", "input_sha256") else {"x": "y"}
    assert RS.compare_counts(got, stored.counts) == []
    del got[field]                                                 # absent in ours: still ignored
    assert RS.compare_counts(got, stored.counts) == []


def test_extra_and_missing_count_fields_fail(stored):
    got = copy.deepcopy(stored.counts)
    got["extra"] = 1
    assert RS.compare_counts(got, stored.counts) == ["extra: not in the reference"]
    got = copy.deepcopy(stored.counts)
    del got["n_clusters"]
    assert RS.compare_counts(got, stored.counts) == ["n_clusters: missing"]


def test_counts_are_compared_after_a_json_round_trip(stored):
    """Our counts have int seed keys and tuples before they are written; the file has strings and lists."""
    got = copy.deepcopy(stored.counts)
    got["episodes_sha256"] = {int(s): v for s, v in got["episodes_sha256"].items()}
    got["seeds"] = tuple(got["seeds"])
    assert RS.compare_counts(got, stored.counts) == []


# ---------------------------------------------------------------- the runner's end: record and exit code

def refs_of(items, counts):
    return {"regression_seed42.json": {"items": items, "n_items": len(items), "passed": True, "n_failed": 0},
            "seed42_pass_counts.json": counts}


@pytest.fixture
def small_counts():
    return {"rule_sha256": "a", "runner_sha256": "b", "module_sha256": {}, "time": "t", "input_sha256": {},
            "mode": "regression", "seeds": [42], "n_episodes": 12, "n_clusters": 4, "checks": {"P1": {"n": 0}},
            "holm_order": ["P1"], "secondary": {}, "episodes_sha256": {"42": {"p": "q"}}, "coef_sha256": {}}


def test_finish_passes_and_writes_the_record(tmp_path, small_counts):
    items = small_items()
    code = RS.finish(copy.deepcopy(items), copy.deepcopy(small_counts), refs_of(items, small_counts), tmp_path,
                     meta={"runtime_s": 1.0})
    assert code == 0
    rec = json.loads((tmp_path / "stage0a_regression.json").read_text())
    assert rec["passed"] is True and rec["n_items"] == len(items) and rec["n_failed"] == 0
    assert rec["differences"] == {"items": [], "pass_counts": []}
    assert not (tmp_path / "stage0a_regression_failed.json").exists()
    assert json.loads((tmp_path / "stage0a_pass_counts.json").read_text())["mode"] == "regression"


def test_finish_exits_3_on_a_differing_item_and_keeps_a_failed_copy(tmp_path, small_counts):
    items = small_items()
    got = small_items()
    got["scalar"] = RS.item(1.0, 1.5)
    code = RS.finish(got, copy.deepcopy(small_counts), refs_of(items, small_counts), tmp_path, meta={})
    assert code == RS.EXIT_DIFF == 3
    rec = json.loads((tmp_path / "stage0a_regression.json").read_text())
    assert rec["passed"] is False and rec["differences"]["items"]
    assert json.loads((tmp_path / "stage0a_regression_failed.json").read_text())["passed"] is False


def test_finish_exits_3_on_a_differing_count_field(tmp_path, small_counts):
    items = small_items()
    got = copy.deepcopy(small_counts)
    got["n_clusters"] = 5
    code = RS.finish(copy.deepcopy(items), got, refs_of(items, small_counts), tmp_path, meta={})
    assert code == 3
    rec = json.loads((tmp_path / "stage0a_regression.json").read_text())
    assert rec["passed"] is False and rec["differences"]["pass_counts"] == ["n_clusters differs"]
    assert rec["differences"]["items"] == []


def test_a_failed_run_replaces_an_earlier_pass(tmp_path, small_counts):
    items = small_items()
    assert RS.finish(copy.deepcopy(items), copy.deepcopy(small_counts), refs_of(items, small_counts), tmp_path,
                     meta={}) == 0
    bad = small_items()
    bad["sha"] = RS.item("ab" * 32, "cd" * 32)
    assert RS.finish(bad, copy.deepcopy(small_counts), refs_of(items, small_counts), tmp_path, meta={}) == 3
    assert json.loads((tmp_path / "stage0a_regression.json").read_text())["passed"] is False


def test_the_reference_must_itself_be_a_pass(tmp_path, small_counts):
    items = small_items()
    refs = refs_of(items, small_counts)
    refs["regression_seed42.json"]["n_failed"] = 1
    with pytest.raises(AssertionError, match="reference"):
        RS.finish(copy.deepcopy(items), copy.deepcopy(small_counts), refs, tmp_path, meta={})


def test_the_reference_item_count_is_asserted(tmp_path, small_counts):
    items = small_items()
    refs = refs_of(items, small_counts)
    refs["regression_seed42.json"]["n_items"] = 131
    with pytest.raises(AssertionError, match="reference"):
        RS.finish(copy.deepcopy(items), copy.deepcopy(small_counts), refs, tmp_path, meta={})


# ---------------------------------------------------------------- loading the references

def test_load_references_checks_the_sha(tmp_path, monkeypatch):
    path = tmp_path / "regression_seed42.json"
    path.write_text("{}")
    monkeypatch.setitem(RS.REFERENCES, "regression_seed42.json", (path, REGRESSION_SHA))
    with pytest.raises(SystemExit, match="SHA-256"):
        RS.load_references()


def test_load_references_reads_the_real_files():
    refs = RS.load_references()
    assert set(refs) == {"regression_seed42.json", "seed42_pass_counts.json", "refit_check.json"}
    assert refs["regression_seed42.json"]["n_items"] == 131


# ---------------------------------------------------------------- the item list of the real run

def scored_like_the_run(rng, n=R.N_PER_PAIR * 3):
    """per_anchor-shaped results for every key regression_items reads (random values: the items will not be equal,
    the wiring is what is under test)."""
    from src.eval.aspect_metrics import METRICS
    names = RS.AB_SCORERS + ("B", "B0", "B1", "aff_fused", "aff_cf", "r1_fused", "r1_cf")
    scored = {k: {m: rng.random(n) * (1 if m == "r1" else 2) for m in METRICS} for k in names}
    for k in scored:
        scored[k]["r1"] = np.round(rng.random(n) * 4) / 4
    scored["gates"] = {w: {c: rng.integers(0, 2, (4, n)).astype(np.float32) for c in "ab"} for w in ("aff", "r1")}
    scored["reader"] = {"P": {c: rng.random((n, 3)) for c in "ab"}, "m": {c: rng.random(n) for c in "ab"},
                        "pick": {c: rng.integers(0, 3, n) for c in "ab"}}
    return scored


def test_regression_items_follow_the_stored_item_list(stored, monkeypatch):
    rng = np.random.default_rng(0)
    n = R.N_PER_PAIR * 3
    scored = scored_like_the_run(rng, n)
    bundle = SimpleNamespace(cl=rng.integers(0, 4602, n), pair_index=np.repeat(np.arange(3), R.N_PER_PAIR),
                             parity=np.arange(n) % 2, episodes_sha256=dict(zip(R.PAIR_NAMES, "xyz")),
                             coef_sha256={"c": 1})
    env = SimpleNamespace(head_check={"passed": True}, coef_sha256={"c": 1}, labels=None,
                          split=SimpleNamespace(groups=None, selection=None), index=None, value_sets=None)
    ident = {"items": {f"episodes_seed{s}__{p}": RS.item("h", "h") for s in (42, 9001, 9002, 9003)
                       for p in R.PAIR_NAMES}}
    monkeypatch.setattr(RS.E, "identity_check", lambda *a: ident)
    monkeypatch.setattr(RS.E, "identity_targets", lambda: {42: {"episodes_sha256": dict(zip(R.PAIR_NAMES, "xyz"))}})
    orders = {"refit": {"coef_sha256": {"c": 1}}, "picks": {"coef_sha256": {"c": 1}}}
    items, bar_v = RS.regression_items(env, bundle, scored, orders)
    assert list(items) == list(stored.reg["items"])               # the same 131 keys in the same order
    assert len(items) == 131
    for key, it in items.items():
        assert set(it) == set(stored.reg["items"][key]), key       # the same fields (arrays carry n_diff and so on)
    for key in ("seed42_arrays/aff_fused__r1", "per_anchor_seed42/cosine__r1"):
        assert items[key]["got"] == stored.reg["items"][key]["got"]    # dtype and shape of ours equal the record's
