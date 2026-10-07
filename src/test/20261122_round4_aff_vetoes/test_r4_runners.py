"""Unit tests of run_r4_seed42.py (rule DECISION_RULE.md §5, §6.1, §8 boundaries, §9, §10): the guard that refuses
candidate results while a regression item is pending, item 5's order and gate algebra, the §5 order test (M20: a failed
regression item leaves no candidate result in any output file or on stdout; a passing run writes
regression_check.json before dev_seed42.json), the boundary stop and its SHA-256-bound continuation, refusals to
overwrite non-smoke results, and the --dry behaviour (no decimal number on stdout, value files deleted).

Synthetic data only. Items 1 to 4 need seed-42 data and are stubs here (only `run_r4_seed42.py --dry` exercises them);
item 5, the development step, the carry and the sensitivity projection run for real on a synthetic world. Run from this
folder:
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python -m pytest -q test_r4_runners.py -p no:cacheprovider
"""
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r4_common as R4  # noqa: E402
import r4_fusion as RF4  # noqa: E402
import r4_stats as RS4  # noqa: E402
import run_r4_seed42 as S  # noqa: E402

RF3, R3 = R4.RF3, R4.R3
from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS, METRICS  # noqa: E402

# any decimal number, one-decimal numbers included (0.5, 19.1, .5), and scientific notation
LEAK = re.compile(r"\.\d|\d[eE][-+]?\d")
N = 48
REAL_OUT = ("regression_check.json", "seed42_arrays.npz", "dev_seed42.json", "boundary_seed42.json", "carry.json",
            "sensitivity.json")


# ---------------------------------------------------------------- a synthetic seed-42 world

def _pa(rng, n, gain0=True):
    return {"r1": rng.integers(0, 5, n) / 4.0, "gain": np.zeros(n) if gain0 else rng.integers(-4, 5, n) / 4.0,
            "other": rng.integers(0, 5, n) / 4.0, "swap": np.zeros(n), "strict": np.zeros(n)}


def make_world(n=N, seed=2, neutral=False):
    """The state items 1 to 4 leave behind (st), on synthetic data. neutral=True: a_v = 1 and pi_A1 = affect on every
    episode, so each candidate's gate is AFF's and every Delta_k is exactly 0 (a rule §8 boundary)."""
    rng = np.random.default_rng(seed)
    base = {d: rng.normal(size=(n, 13)).astype(np.float32) for d in DIRECTIONS}
    B = {c: {d: base[d].copy() for d in DIRECTIONS} for c in CONDITIONS}
    T = {c: {} for c in CONDITIONS}
    for c, col in (("a", 0), ("b", 1)):
        for d in DIRECTIONS:
            t = rng.normal(size=(n, 13)).astype(np.float32)
            t[:, col] += 1.5 * (rng.random(n) < 0.6)
            T[c][d] = t
    v = rng.random(n) * (0.01 if neutral else 0.028)                    # about 75% below v75 unless neutral
    F = {c: np.zeros((n, 18)) for c in CONDITIONS}
    F["a"][:, 6], F["a"][:, 7] = v, v + 0.005
    F["b"][:, 6], F["b"][:, 7] = v + 0.005, v
    b = SimpleNamespace(seed=42, smoke=False, n=n, B=B, parity=np.arange(n) % 2, cl=rng.integers(0, 12, n),
                        pair_index=np.arange(n) % 3, F=F, v=v, pB=_pa(rng, n), pBp=_pa(rng, n), pBp1=_pa(rng, n))
    m = {c: rng.random(n) for c in CONDITIONS}
    P = {c: rng.dirichlet(np.ones(3), n) for c in CONDITIONS}
    pick = {c: P[c].argmax(axis=1).astype(np.int64) for c in CONDITIONS}
    P1 = {c: rng.dirichlet(np.array([4.0, 1.0, 1.0, 1.0]), n) for c in CONDITIONS}     # affect-leaning picks
    pick1 = {c: (np.zeros(n, np.int64) if neutral else P1[c].argmax(axis=1).astype(np.int64)) for c in CONDITIONS}
    g_r1 = RF3.gates_r1(m, R3.TAUS)
    g_aff = RF3.gates_aff(m, pick, R3.TAUS)
    keep = RF4.abstain(v)
    g_img = RF4.gates_imgabst_r1(g_r1, keep)
    return {"b": b, "ext": {"cosine": _pa(rng, n, False), "rca": _pa(rng, n, False)},
            "rd": {"T": T, "m": m, "pick": pick, "P": P}, "g_r1": g_r1, "fam_r1": RF3.run_family(b, T, g_r1),
            "g_aff": g_aff, "fam_aff": RF3.run_family(b, T, g_aff), "keep": keep, "g_img": g_img,
            "fam_img": RF3.run_family(b, T, g_img), "ra1": {"P": P1, "pick": pick1},
            "pick_stored": {c: pick1[c].astype(np.int8) for c in CONDITIONS}}


def stub_items(world, fail_at=None, raise_at=None):
    """Items 1 to 4 as stubs (item 1 leaves the synthetic state; a failing item records one failed comparison or
    raises); item 5 is the real one."""
    def mk(k):
        def item(rec, st, guard):
            if k == 1:
                st.update(world)
            if raise_at == k:
                raise RuntimeError("stub crash")
            rec.add(k, f"stub_comparison_{k}", 1, 2 if fail_at == k else 1)
        item.__name__ = f"item{k}"
        return item
    return (mk(1), mk(2), mk(3), mk(4), S.item5)


class Spy:
    def __init__(self, fn):
        self.fn, self.calls = fn, []

    def __call__(self, *a, **k):
        self.calls.append((a, k))
        return self.fn(*a, **k)


@pytest.fixture
def env(monkeypatch, tmp_path):
    """Results go to tmp_path; the input SHA-256 checks are skipped (no seed-42 file is read); the candidate functions
    are spied on."""
    monkeypatch.setattr(R4, "RES", tmp_path / "results")
    monkeypatch.setattr(S, "check_inputs", lambda: ({}, R3.TAUS))
    spies = {}
    for mod, name in ((RF4, "run_candidate"), (RS4, "dev_record"), (RS4, "carry"), (RS4, "sensitivity_all")):
        spies[name] = Spy(getattr(mod, name))
        monkeypatch.setattr(mod, name, spies[name])
    order = []
    w0, z0 = R4.write_json_once, np.savez_compressed

    def w(p, rec, smoke):
        order.append(Path(p).name)
        return w0(p, rec, smoke)

    def z(p, **a):
        order.append(Path(p).name)
        return z0(p, **a)
    monkeypatch.setattr(R4, "write_json_once", w)
    monkeypatch.setattr(np, "savez_compressed", z)
    return SimpleNamespace(res=tmp_path / "results", spies=spies, order=order, mp=monkeypatch)


def _names(d):
    return sorted(p.name for p in Path(d).iterdir() if p.is_file()) if Path(d).exists() else []


def _floats(x):
    if isinstance(x, float):
        return [x]
    if isinstance(x, dict):
        return [f for v in x.values() for f in _floats(v)]
    if isinstance(x, list):
        return [f for v in x for f in _floats(v)]
    return []


# ---------------------------------------------------------------- the leak pattern itself

def test_leak_pattern_catches_any_decimal_number():
    for s in ("bar 0.5", "19.1", "x .5 y", "-1.25", "1e-12", "2E+3", "[0.4, 0.9]"):
        assert LEAK.search(s), s
    for s in ("CHECK item5 V2_equals_AFF_times_stored_pick_affect.tau0.a PASS", "ITEM 1 PASS (59 comparisons)",
              "dev_seed42_dry.json written", "seed42_arrays_dry.npz written (212 arrays)", "SEED42_DRY PASS [31s]"):
        assert not LEAK.search(s), s


# ---------------------------------------------------------------- the guard

def test_guard_refuses_every_candidate_output_before_release(tmp_path, capsys):
    g = S.Guard()
    p = tmp_path / "dev_seed42.json"
    with pytest.raises(S.GuardError):
        g.write_json(p, {"x": 1}, False)
    with pytest.raises(S.GuardError):
        g.savez(tmp_path / "seed42_arrays.npz", {"x": np.zeros(2)})
    with pytest.raises(S.GuardError):
        g.say("V4 fused 19")
    with pytest.raises(S.GuardError):
        g.call(lambda: 1)
    with pytest.raises(S.GuardError):
        g.require_gates()
    assert _names(tmp_path) == [] and capsys.readouterr().out == ""


def test_guard_items_must_pass_in_order_and_release_needs_a_passed_record(tmp_path):
    g, rec = S.Guard(), S.Recorder()
    rec.add(1, "x", 1, 1)
    with pytest.raises(S.GuardError):                       # item 2 before item 1
        g.item_passed(2, rec)
    g.item_passed(1, rec)
    with pytest.raises(S.GuardError):                       # no comparison recorded for item 2
        g.item_passed(2, rec)
    rec.add(2, "y", 1, 2)                                   # a failed comparison
    with pytest.raises(S.GuardError):
        g.item_passed(2, rec)
    g2, rec2 = S.Guard(), S.Recorder()
    for k in range(1, 5):
        rec2.add(k, "x", 1, 1)
        g2.item_passed(k, rec2)
    g2.require_gates()                                      # items 1 to 4 passed: the gates may be computed
    reg = tmp_path / "regression_check.json"
    with pytest.raises(S.GuardError):                       # item 5 pending
        g2.release(reg)
    rec2.add(5, "x", 1, 1)
    g2.item_passed(5, rec2)
    with pytest.raises(S.GuardError):                       # the record is not written yet
        g2.release(reg)
    reg.write_text(json.dumps({"passed": False, "items_passed": [1, 2, 3, 4, 5]}))
    with pytest.raises(S.GuardError):
        g2.release(reg)
    reg.write_text(json.dumps({"passed": True, "items_passed": [1, 2, 3, 4, 5]}))
    g2.release(reg)
    g2.write_json(tmp_path / "dev_seed42.json", {"x": 1}, False)
    assert (tmp_path / "dev_seed42.json").exists()


# ---------------------------------------------------------------- item 5

def _passed_guard(upto=4):
    g, rec = S.Guard(), S.Recorder()
    for k in range(1, upto + 1):
        rec.add(k, "x", 1, 1)
        g.item_passed(k, rec)
    return g


def test_item5_refuses_before_items_1_to_4_have_passed():
    w = make_world()
    with pytest.raises(S.GuardError):
        S.item5(S.Recorder(), w, S.Guard())
    with pytest.raises(S.GuardError):
        S.item5(S.Recorder(), w, _passed_guard(3))


def test_item5_gate_algebra_on_consistent_inputs():
    w = make_world()
    rec = S.Recorder()
    S.item5(rec, w, _passed_guard())
    rows = [r for r in rec.rows if r["item"] == 5]
    assert rows and all(r["pass"] for r in rows)
    names = {r["name"] for r in rows}
    for t in range(4):
        for c in CONDITIONS:
            for k in R4.CANDIDATES:
                assert f"{k}_closed_where_AFF_closed.tau{t}.{c}" in names
            for s in ("V24_equals_V4_times_V2", "V2_equals_AFF_times_stored_pick_affect",
                      "V4_equals_AFF_times_v_below_v75"):
                assert f"{s}.tau{t}.{c}" in names
    counts = w["open_tau0"]
    assert set(counts) == {"AFF", *R4.CANDIDATES}
    assert all(type(counts[k][c]) is int for k in counts for c in CONDITIONS)
    for k in R4.CANDIDATES:
        assert all(counts[k][c] <= counts["AFF"][c] for c in CONDITIONS)
        for t in range(4):
            assert all(w["G"][k][t][c].dtype == np.float32 for c in CONDITIONS)


def _open_index(w):
    return int(np.flatnonzero(np.asarray(w["g_aff"][0]["a"]) == 1)[0])


def test_item5_fails_when_the_stored_picks_disagree():
    w = make_world()
    i = _open_index(w)
    w["pick_stored"]["a"][i] = 1 if w["pick_stored"]["a"][i] == 0 else 0
    rec = S.Recorder()
    S.item5(rec, w, _passed_guard())
    bad = [r["name"] for r in rec.rows if not r["pass"]]
    assert "V2_equals_AFF_times_stored_pick_affect.tau0.a" in bad


def test_item5_fails_when_v_from_the_a0_features_disagrees():
    w = make_world()
    i = _open_index(w)
    w["b"].F["a"][i, 6:8] = 1.0 if w["b"].v[i] < R4.V75 else 0.0      # flips 1[v < v75] for this episode
    rec = S.Recorder()
    S.item5(rec, w, _passed_guard())
    bad = [r["name"] for r in rec.rows if not r["pass"]]
    assert "V4_equals_AFF_times_v_below_v75.tau0.a" in bad


# ---------------------------------------------------------------- §5 order test (M20)

@pytest.mark.parametrize("fail_at", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("dry", [False, True])
def test_a_failed_regression_item_leaves_no_candidate_result(env, capsys, fail_at, dry):
    w = make_world()
    if fail_at == 5:                                  # a real comparison of item 5 fails (a tampered stored pick)
        i = _open_index(w)
        w["pick_stored"]["a"][i] = 1 if w["pick_stored"]["a"][i] == 0 else 0
    env.mp.setattr(S, "ITEMS", stub_items(w, fail_at=fail_at if fail_at < 5 else None))
    assert S.run(dry) == 1
    out = capsys.readouterr().out
    sfx = "_dry" if dry else ""
    d = env.res / "smoke" if dry else env.res
    assert _names(d) == [f"regression_check{sfx}.json"]
    assert env.order == [f"regression_check{sfx}.json"]
    for name, spy in env.spies.items():
        assert spy.calls == [], name
    reg = json.loads((d / f"regression_check{sfx}.json").read_text())
    assert reg["passed"] is False and reg["stopped_at_item"] == fail_at
    assert reg["items_passed"] == list(range(1, fail_at))
    assert not {"candidates", "carry", "dev", "sensitivity"} & set(reg)
    assert f"ITEM {fail_at} FAIL" in out and "DEV" not in out and "carry" not in out.lower()
    assert not LEAK.search(out)
    for k in range(fail_at + 1, 6):
        assert f"item{k}" not in out


@pytest.mark.parametrize("raise_at", [1, 3])
def test_an_item_that_raises_is_a_failed_item(env, capsys, raise_at):
    env.mp.setattr(S, "ITEMS", stub_items(make_world(), raise_at=raise_at))
    assert S.run(False) == 1
    assert _names(env.res) == ["regression_check.json"]
    reg = json.loads((env.res / "regression_check.json").read_text())
    assert reg["stopped_at_item"] == raise_at and "RuntimeError" in json.dumps(reg["comparisons"])
    assert all(spy.calls == [] for spy in env.spies.values())


def test_a_passing_run_writes_the_regression_record_first(env, capsys):
    w = make_world()
    w5 = dict(w)                                          # item 5's gates and counts, computed beside the run
    S.item5(S.Recorder(), w5, _passed_guard())
    env.mp.setattr(S, "ITEMS", stub_items(w))
    assert S.run(False) == 0
    out = capsys.readouterr().out
    o = env.order
    assert o[0] == "regression_check.json"
    assert o.index("regression_check.json") < o.index("seed42_arrays.npz") < o.index("dev_seed42.json") \
        < o.index("carry.json")
    reg = json.loads((env.res / "regression_check.json").read_text())
    assert reg["passed"] is True and reg["items_passed"] == [1, 2, 3, 4, 5]
    assert reg["open_tau0_counts"] == w5["open_tau0"]
    lines = out.splitlines()
    first_dev = min(i for i, l in enumerate(lines) if l.startswith("DEV"))
    assert max(i for i, l in enumerate(lines) if l.startswith("CHECK item")) < first_dev
    assert min(i for i, l in enumerate(lines) if l.startswith("REGRESSION PASS")) < first_dev
    # each candidate's family is run once, with its own factors (D6), against AFF's gates
    calls = [(a[2], k.get("pick_a1") is None, k.get("keep") is None) for a, k in env.spies["run_candidate"].calls]
    assert calls == [("V4", True, False), ("V2", False, True), ("V24", False, False)]
    for a, _ in env.spies["run_candidate"].calls:
        assert a[3] is w["g_aff"]
    dev = json.loads((env.res / "dev_seed42.json").read_text())
    assert dev["rule_sha256"] == R4.RULE_SHA and dev["written_amsterdam"] and dev["provenance"]["smoke"] is False
    assert list(dev["candidates"]) == list(R4.CANDIDATES)
    for k in R4.CANDIDATES:
        r = dev["candidates"][k]
        assert type(r["delta_int"]) is int and set(r["d10"]) == {"c1", "c2", "c3", "clears"}
        assert r["boundaries"] == []                          # this world has no boundary (precondition)
    carry = json.loads((env.res / "carry.json").read_text())
    assert carry["delta_int"] == {k: dev["candidates"][k]["delta_int"] for k in R4.CANDIDATES}
    assert carry["dev_seed42_sha256"] == R4.sha_file(env.res / "dev_seed42.json")
    assert ("sensitivity.json" in _names(env.res)) == (carry["carry"]["carried"] is not None)
    with np.load(env.res / "seed42_arrays.npz") as z:
        assert set(z.files) == set(S.array_keys())
        for k in ("aff", "v4", "v2", "v24"):
            for c in CONDITIONS:
                assert z[f"{k}_gate__{c}"].shape == (4, N)
        np.testing.assert_array_equal(z["v4_gate__a"][0], w5["G"]["V4"][0]["a"])
        np.testing.assert_array_equal(z["aff_fused__r1"], w["fam_aff"]["fused"]["r1"])


def test_development_refuses_without_a_written_regression_record(env, capsys):
    """The guard is released by the record on disk, not by the items alone: a lost regression record stops the run
    before any candidate is computed."""
    env.mp.setattr(S, "ITEMS", stub_items(make_world()))
    w0 = R4.write_json_once

    def lose_reg(p, rec, smoke):
        if Path(p).name == "regression_check.json":
            return rec                                    # "written" but never reaches the disk
        return w0(p, rec, smoke)
    env.mp.setattr(R4, "write_json_once", lose_reg)
    with pytest.raises(S.GuardError):
        S.run(False)
    assert _names(env.res) == []
    assert all(spy.calls == [] for spy in env.spies.values())
    assert "DEV" not in capsys.readouterr().out


@pytest.mark.parametrize("carried, n_checks", [("V4", 8), ("V2", 9)])
def test_a_carried_candidate_gets_its_sensitivity_after_the_carry(env, capsys, carried, n_checks):
    env.mp.setattr(S, "ITEMS", stub_items(make_world()))
    real = env.spies["carry"].fn

    def forced(records):
        c = real(records)
        return {**c, "E": [carried], "M": records[carried]["delta_int"], "tied": [carried], "carried": carried}
    env.mp.setattr(RS4, "carry", forced)
    assert S.run(False) == 0
    o = env.order
    assert o.index("carry.json") < o.index("sensitivity.json")
    sens = json.loads((env.res / "sensitivity.json").read_text())
    assert sens["candidate"] == carried and list(sens["checks"]) == RS4._check_names(carried)
    assert len(sens["checks"]) == n_checks and list(sens["checks"])[-1] == "vs_AFF"
    assert sens["carry_sha256"] == R4.sha_file(env.res / "carry.json")
    assert sens["seed42_arrays_sha256"] == R4.sha_file(env.res / "seed42_arrays.npz")
    # the projection reads the saved arrays: recomputing from the npz gives the same numbers
    with np.load(env.res / "seed42_arrays.npz") as z:
        arr = {k: z[k] for k in z.files}
    again = env.spies["sensitivity_all"].fn(carried, S.seed42_view(arr, carried))
    assert json.loads(json.dumps(R4.C.jsonable(again))) == sens["checks"]
    assert "SENSITIVITY" in capsys.readouterr().out


# ---------------------------------------------------------------- §8 boundaries

def test_a_boundary_stops_before_the_carry_and_resumes_only_with_its_sha(env, capsys):
    env.mp.setattr(S, "ITEMS", stub_items(make_world(neutral=True)))
    assert S.run(False) == 3
    out = capsys.readouterr().out
    assert "BOUNDARY" in out
    assert _names(env.res) == sorted(["regression_check.json", "seed42_arrays.npz", "dev_seed42.json",
                                      "boundary_seed42.json"])
    bnd = json.loads((env.res / "boundary_seed42.json").read_text())
    assert any("Delta_k is exactly 0" in s for s in bnd["boundaries"])
    assert bnd["carry_by_the_stated_inequalities"]["carried"] is None
    sha = R4.sha_file(env.res / "boundary_seed42.json")

    def boom(*a, **k):
        raise AssertionError("the continuation must not rerun the items")
    env.mp.setattr(S, "ITEMS", (boom,) * 5)
    with pytest.raises(SystemExit, match="SHA-256"):
        S.run(False, boundary_reported="0" * 64)
    assert "carry.json" not in _names(env.res)
    assert S.run(False, boundary_reported=sha) == 0
    carry = json.loads((env.res / "carry.json").read_text())
    assert carry["carry"]["carried"] is None and carry["carry"]["E"] == []
    assert carry["boundary_reported"]["boundary_seed42_sha256"] == sha
    assert carry["boundary_reported"]["boundaries"] == bnd["boundaries"]
    assert "sensitivity.json" not in _names(env.res)
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        S.run(False, boundary_reported=sha)


def test_the_continuation_refuses_a_dev_file_changed_after_the_boundary(env, capsys):
    env.mp.setattr(S, "ITEMS", stub_items(make_world(neutral=True)))
    assert S.run(False) == 3
    sha = R4.sha_file(env.res / "boundary_seed42.json")
    p = env.res / "dev_seed42.json"
    p.write_text(p.read_text().replace('"delta_int": 0', '"delta_int": 30', 1))
    with pytest.raises(SystemExit, match="dev_seed42.json"):
        S.run(False, boundary_reported=sha)
    assert "carry.json" not in _names(env.res)


def test_boundary_flag_is_refused_in_dry_mode(env):
    with pytest.raises(SystemExit):
        S.main(["--dry", "--boundary-reported", "0" * 64])


# ---------------------------------------------------------------- refusals to overwrite non-smoke results

@pytest.mark.parametrize("name", REAL_OUT)
def test_refuses_to_overwrite_any_non_smoke_result(env, name):
    env.res.mkdir(parents=True)
    (env.res / name).write_text("earlier result")

    def boom(*a, **k):
        raise AssertionError("nothing may run when an output exists")
    env.mp.setattr(S, "ITEMS", (boom,) * 5)
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        S.run(False)
    assert _names(env.res) == [name] and (env.res / name).read_text() == "earlier result"


def test_dry_runs_may_overwrite_smoke_files(env, capsys):
    sm = env.res / "smoke"
    sm.mkdir(parents=True)
    for n in ("regression_check_dry.json", "dev_seed42_dry.json", "carry_dry.json"):
        (sm / n).write_text("stale")
    env.mp.setattr(S, "ITEMS", stub_items(make_world()))
    assert S.run(True) == 0
    assert not {"regression_check_dry.json", "dev_seed42_dry.json", "carry_dry.json"} & set(_names(sm))


# ---------------------------------------------------------------- --dry

@pytest.mark.parametrize("neutral", [False, True])
def test_dry_prints_no_decimal_number_and_deletes_its_value_files(env, capsys, neutral):
    env.mp.setattr(S, "ITEMS", stub_items(make_world(neutral=neutral)))
    assert S.run(True) == 0
    out = capsys.readouterr().out
    assert not LEAK.search(out), [l for l in out.splitlines() if LEAK.search(l)]
    for word in ("carried", "KILL", "BOUNDARY", "Delta", "clears", "DEV", "SENSITIVITY"):
        assert word not in out, word
    assert "SEED42_DRY PASS" in out
    for line in out.splitlines():
        assert re.fullmatch(r"(CHECK \S+ \S+ (PASS|FAIL)|ITEM \d (PASS|FAIL) \(\d+ comparisons\)|"
                            r"REGRESSION PASS: \d+ comparisons, items 1 to 5|\S+_dry\.(json|npz) (written|deleted).*|"
                            r"CHECK files\.\S+ (PASS|FAIL)|SEED42_DRY (PASS|FAIL) \[\d+s\])", line), line
    sm = env.res / "smoke"
    assert _names(sm) == ["seed42_dry.json"]
    assert not env.res.joinpath("regression_check.json").exists()
    summary = json.loads((sm / "seed42_dry.json").read_text())
    assert summary["passed"] is True and _floats(summary) == []
    assert set(summary["files_checked"]) == {"regression_check_dry.json", "seed42_arrays_dry.npz",
                                             "dev_seed42_dry.json", "carry_dry.json", "sensitivity_dry.json"}
    assert all(v is True for v in summary["files_checked"].values())


def test_dry_computes_the_sensitivity_of_every_candidate(env, capsys):
    env.mp.setattr(S, "ITEMS", stub_items(make_world()))
    assert S.run(True) == 0
    assert [a[0] for a, _ in env.spies["sensitivity_all"].calls] == list(R4.CANDIDATES)


# ---------------------------------------------------------------- inputs

def test_seed42_inputs_are_rule_inputs_and_cover_what_the_runner_reads():
    for n in S.SEED42_INPUTS:
        assert n in R4.INPUTS or n in R3.INPUTS, n
    for n in ("20261121_round3_affect_gate/r3_fusion.py", "20261121_round3_affect_gate/r3_stats.py",
              "20261118_reader_fix_round2/results/cand_R1_A1.npz", "20261118_reader_fix_round2/results/cand_R1_A1.json",
              "20261120_r1_levers_brainstorm/results/bs_04_readers.json",
              "20261117_reader_fix_csd/results/cand_Rc_Rb_expected_A0.npz"):
        assert n in S.SEED42_INPUTS, n
    for cell, text in S.CELL_TEXT.items():
        d = RF3.describe(cell)
        assert (d["tau_index"], d["lambda_u"], d["lambda_a"]) == text
    assert set(R4.IMGABST_TARGETS["cells"]["fused"]) | set(R4.IMGABST_TARGETS["cells"]["cf"]) <= set(S.CELL_TEXT)


def test_dry_deletes_its_value_files_when_the_development_step_raises(env, capsys):
    env.mp.setattr(S, "ITEMS", stub_items(make_world()))

    def crash(*a, **k):
        raise RuntimeError("stub crash in the development step")
    env.mp.setattr(RS4, "carry", crash)
    with pytest.raises(RuntimeError):
        S.run(True)
    assert _names(env.res / "smoke") == []
    out = capsys.readouterr().out
    assert "SEED42_DRY FAIL" in out and not LEAK.search(out)
