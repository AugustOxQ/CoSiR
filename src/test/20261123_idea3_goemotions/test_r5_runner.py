"""Tests of run_r5_seed42.py (rule DECISION_RULE.md of this folder: §5, D5, D7, D11, §6.1, §8 order, boundaries and
lapses 1 to 6, §9, §10 list A item 7). Run from this folder, output redirected:
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python -m pytest -q -p no:cacheprovider test_r5_runner.py \
        > results/smoke/test_r5_runner.log 2>&1

Synthetic data only (no seed-42 array, no GoEmotions model, no GE placement on real data). Items 1 and 4 need the real
seed-42 bundle and are stubs here (`run_r5_seed42.py --dry` runs them for real); items 2 and 3 run for real on
synthetic records; the development step, the carry, the boundary stop and its continuation, the diagnostics and the
sensitivity path run for real on test_r5_bundle's synthetic round-4 bundles (bundle seed 42, not smoke, the 24,576
margin count patched to the synthetic size) with a synthetic GE file in a temporary cache. Covered (list A item 7):
  - the guard refuses every function that can take Q_GE, each called directly with a 'ge' object before release, and
    refuses the diagnostics before results/carry.json (D11, M20, T3a-2);
  - items 1 to 4 run in the rule's order; regression_check.json is written and the guard released before any GE
    result; a failing item leaves no GE-placement result in any output or on stdout (M20);
  - the development step checks and stores the gates it used; a mutated tau' fires (T3a-3);
  - every entry and resume path runs the full input check first; a patched hash stops before any build (T3a-1, T1);
  - the --sensitivity refusals, the nine checks in §6.5's order, SE and x on a hand-computed example (rule check S1);
  - the boundary continuation with a carried candidate and with a kill (T3a-4); the CARRY and KILL lines (T3a-5);
  - no non-smoke overwrite; the dry run stops at the guard and its console holds no decimal number.
"""
import json
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r5_common as R5  # noqa: E402
import r5_guard as R5G  # noqa: E402
import r5_bundle as R5B  # noqa: E402
import r5_fusion as R5F  # noqa: E402
import r5_stats as R5S  # noqa: E402
import r5_diag as R5D  # noqa: E402
import run_r5_seed42 as S  # noqa: E402
from test_r5_bundle import make_ge, release_guard, synth_bundle  # noqa: E402

from src.eval.aspect_metrics import CONDITIONS, METRICS  # noqa: E402

R3, RB3, RB4, RF3, RS3, C = R5.R3, R5.RB3, R5.RB4, R5.RF3, R5.RS3, R5.C
TAUS = tuple(R3.TAUS)
# round 4's leak pattern: any decimal number, one-decimal numbers included (0.5, 19.1, .5), and scientific notation
LEAK = re.compile(r"\.\d|\d[eE][-+]?\d")
REAL_PINNED = {k: tuple(v) for k, v in S.PINNED_NAMES.items()}   # the runner's pin, before any test patches it
REAL_OUT = ("regression_check.json", "seed42_arrays.npz", "dev_seed42.json", "boundary_seed42.json", "carry.json",
            "diagnostics_seed42.json", "sensitivity.json")


# ---------------------------------------------------------------- isolation and the synthetic world

@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    """Every test: results, smoke and cache folders in tmp_path, the run log in tmp_path, the guard closed, _HEADS as
    found."""
    saved = dict(RB3._HEADS)
    R5G._reset_for_tests()
    monkeypatch.setattr(R5, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(R5, "SMOKE", tmp_path / "results" / "smoke")
    monkeypatch.setattr(R5, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(S, "RUN_LOG", tmp_path / "run_log.md")
    monkeypatch.setattr(S, "PINNED_NAMES", {1: ("stub_comparison_1",), 4: ("stub_comparison_4",)})   # stub items
    (tmp_path / "cache").mkdir()
    yield
    RB3._HEADS.clear()
    RB3._HEADS.update(saved)
    R5G._reset_for_tests()


def _pa(rng, n, gain0=True):
    return {"r1": rng.integers(0, 5, n) / 4.0, "gain": np.zeros(n) if gain0 else rng.integers(-4, 5, n) / 4.0,
            "other": rng.integers(0, 5, n) / 4.0, "swap": np.zeros(n), "strict": np.zeros(n)}


def world_fn(tmp_path, monkeypatch, seed=3):
    """A function returning the state items 1 to 4 leave behind (st) on a synthetic seed-42 bundle, deterministic in
    `seed` (the continuation's rebuild calls it again); the GE file lies in the patched cache folder."""
    n = 24
    monkeypatch.setattr(R5F, "N_MARGINS_42", 2 * n)
    monkeypatch.setattr(S, "N_MARGINS", 2 * n)
    b0 = synth_bundle(n=n, rows=50, seed=seed, bundle_seed=42, smoke=False)
    make_ge(R5.CACHE, monkeypatch, b0, seed=100 + seed, name="r5_ge_posterior.npz")   # once, as the placement step

    def make():
        b = synth_bundle(n=n, rows=50, seed=seed, bundle_seed=42, smoke=False)
        rng = np.random.default_rng(1000 + seed)
        sel = np.asarray(b.ctx.selection)
        labS = {a: rng.integers(0, 3, len(sel)).astype(np.int64) for a in ("emotion", "style", "genre")}
        gS = np.asarray(b.ctx.groups)[sel]
        ext = {"cosine": _pa(rng, n, False), "rca": _pa(rng, n, False)}
        return S.base_state(b, ext, labS, gS, TAUS)
    return make


def stub_items(make, fail_at=None, raise_at=None, skip=(), say=None, calls=None):
    """Items 1 to 4 as stubs: item 1 leaves the synthetic state; a failing item records one failed comparison or
    raises; `skip` items record a SKIP (dry run, records absent); `say` is printed by item 1."""
    def mk(k):
        def item(rec, st):
            if calls is not None:
                calls.append(k)
            if k == 1:
                st.update(make())
                if say:
                    print(say, flush=True)
            if raise_at == k:
                raise RuntimeError("stub crash")
            if k in skip:
                rec.skip(k, f"stub_record_{k}", "stub record absent")
                return
            rec.add(k, f"stub_comparison_{k}", 1, 2 if fail_at == k else 1)
        item.__name__ = f"item{k}"
        return item
    return tuple(mk(k) for k in (1, 2, 3, 4))


def no_inputs(monkeypatch, calls=None):
    def fake(dry=False):
        if calls is not None:
            calls.append(("check_inputs", dry))
        return {"stub": True, "taus": list(TAUS)}
    monkeypatch.setattr(S, "check_inputs", fake)


class Spy:
    def __init__(self, fn):
        self.fn, self.calls, self.returns = fn, [], []

    def __call__(self, *a, **k):
        self.calls.append((a, k))
        out = self.fn(*a, **k)
        self.returns.append(out)
        return out


def _names(d):
    d = Path(d)
    return sorted(p.name for p in d.iterdir() if p.is_file()) if d.exists() else []


def _patch_record(monkeypatch, overrides):
    """dev_record with fields overridden per candidate (the carry and the boundaries read only these fields)."""
    real = R5S.dev_record

    def fake(name, *a, **k):
        rec = real(name, *a, **k)
        rec.update(overrides.get(name, {}))
        return rec
    monkeypatch.setattr(R5S, "dev_record", fake)


CLEARS = {"bar_margin": {"point": 1.0, "ci95": [0.25, 1.75]}, "gain_statistic": {"point": 2.0, "ci95": [0.5, 3.5]}}
FAILS = {"bar_margin": {"point": 0.25, "ci95": [-0.5, 1.0]}, "gain_statistic": {"point": 2.0, "ci95": [0.5, 3.5]}}
CARRIED = {"G-T": {**CLEARS, "delta_int": 30, "boundaries": []},
           "G-TF": {**CLEARS, "delta_int": 7, "boundaries": []}}
KILLED = {"G-T": {**FAILS, "delta_int": 30, "boundaries": []}, "G-TF": {**CLEARS, "delta_int": -5, "boundaries": []}}
# a tie gap of exactly 24 (rule §8 boundary): G-T 30, G-TF 6 -> both tied, G-T carried
CARRIED_BOUNDARY = {"G-T": {**CLEARS, "delta_int": 30, "boundaries": []},
                    "G-TF": {**CLEARS, "delta_int": 6, "boundaries": []}}
KILLED_BOUNDARY = {"G-T": {**FAILS, "delta_int": 30, "boundaries": []},
                   "G-TF": {**CLEARS, "delta_int": 0, "boundaries": ["G-TF: Delta_k is exactly 0"]}}


# ---------------------------------------------------------------- the leak pattern

def test_leak_pattern_catches_any_decimal_number():
    for s in ("bar 0.5", "x .5 y", "5e-03", "19.1", "-1.25", "1e-12", "2E+3", "[0.4, 0.9]", "3.0"):
        assert LEAK.search(s), s
        assert S.LEAK.search(s), s
    for s in ("CHECK item1 r3.cos.a.i2t PASS", "ITEM 4 PASS (120 comparisons)", "SEED42_DRY PASS [138s]",
              "[02:19:30] [r3_bundle] seed 42: 12288 episodes (4096 per pair)", "stack (12288, 3, 13)",
              "CHECK item2 stub_record_2 SKIP (stub record absent)", "regression_check_dry.json written"):
        assert not LEAK.search(s), s
        assert not S.LEAK.search(s), s


# ---------------------------------------------------------------- order (rule §5, §8 step 5)

def test_items_are_the_rules_items_in_the_rules_order():
    assert [f.__name__ for f in S.ITEMS] == ["item1", "item2", "item3", "item4"]
    assert S.POST_RELEASE == ("positive_check_D5", "item5_development_numbers", "item6_development_bar",
                              "boundary_stop_or_items7_8", "measured_diagnostics")
    assert S.SENS_ORDER == ("r1_vs_cosine", "r1_vs_rca", "r1_vs_B", "r1_vs_Bprime_A0", "r1_vs_Bprime_G",
                            "r1_vs_counterpart", "gain_statistic", "gain_vs_rca", "r1_vs_AFF")


def test_order_items_then_record_then_release_then_development(tmp_path, monkeypatch, capsys):
    make = world_fn(tmp_path, monkeypatch)
    calls, events = [], []
    no_inputs(monkeypatch)
    monkeypatch.setattr(S, "ITEMS", stub_items(make, calls=calls))
    w0, z0, rel0, ge0 = R5.write_json_once, S._write_npz, R5G.release, R5G.ge_from_file

    def w(p, rec, smoke):
        events.append(Path(p).name)
        return w0(p, rec, smoke)

    def z(p, arrays):
        events.append(Path(p).name)
        return z0(p, arrays)

    def rel(p):
        events.append(("release", sorted(_names(R5.RESULTS))))
        return rel0(p)

    def ge(p):
        events.append(("ge_from_file", R5G.is_released()))
        return ge0(p)
    monkeypatch.setattr(R5, "write_json_once", w)
    monkeypatch.setattr(S, "_write_npz", z)
    monkeypatch.setattr(R5G, "release", rel)
    monkeypatch.setattr(R5G, "ge_from_file", ge)
    _patch_record(monkeypatch, CARRIED)
    assert S.run(False) == 0
    assert calls == [1, 2, 3, 4]
    assert events == ["regression_check.json", ("release", ["regression_check.json"]), ("ge_from_file", True),
                      "seed42_arrays.npz", "dev_seed42.json", "carry.json", "diagnostics_seed42.json"]
    reg = json.loads((R5.RESULTS / "regression_check.json").read_text())
    assert reg["all_passed"] is True and reg["rule_sha256"] == R5.RULE_SHA
    assert list(reg["items"]) == ["1", "2", "3", "4"] and all(v["passed"] is True for v in reg["items"].values())
    out = capsys.readouterr().out
    assert "CARRY G-T (pending the phase-1 agreement, rule §8)" in out.splitlines()
    assert out.index("regression_check.json written") < out.index("GUARD released") < out.index("DEV G-T")


@pytest.mark.parametrize("k", [1, 2, 3, 4])
@pytest.mark.parametrize("mode", ["fail", "raise"])
def test_a_failing_item_leaves_no_ge_result_anywhere(tmp_path, monkeypatch, capsys, k, mode):
    make = world_fn(tmp_path, monkeypatch)
    calls = []
    no_inputs(monkeypatch)
    monkeypatch.setattr(S, "ITEMS", stub_items(make, fail_at=k if mode == "fail" else None,
                                               raise_at=k if mode == "raise" else None, calls=calls))
    spies = {}
    for mod, name in ((R5G, "ge_from_file"), (R5B, "extend"), (R5F, "candidate"), (R5F, "run_candidate"),
                      (R5S, "dev_record"), (R5S, "carry"), (S, "develop_core"), (S, "development")):
        spies[name] = Spy(getattr(mod, name))
        monkeypatch.setattr(mod, name, spies[name])
    assert S.run(False) == 1
    assert calls == list(range(1, k + 1))                      # nothing after the failed item runs
    assert _names(R5.RESULTS) == ["regression_check.json"]
    reg = json.loads((R5.RESULTS / "regression_check.json").read_text())
    assert reg["all_passed"] is False and reg["stopped_at_item"] == k
    assert all(not s.calls for s in spies.values()), {n: len(s.calls) for n, s in spies.items()}
    assert not R5G.is_released()
    with pytest.raises(R5G.GuardError):
        R5G.release(R5.RESULTS / "regression_check.json")    # a failure record never releases
    out = capsys.readouterr().out
    for word in ("DEV", "CARRY", "KILL", "BOUNDARY", "positive", "GUARD released"):
        assert word not in out, word
    assert f"FAIL at item {k}" in out


def test_order_refuses_an_item_out_of_order_and_a_release_before_all_four(tmp_path):
    release_guard(tmp_path)                                    # a valid record exists: only Order can refuse
    R5G._reset_for_tests()
    o = S.Order()
    with pytest.raises(R5G.GuardError):
        o.mark(2, "pass")
    o.mark(1, "pass")
    o.mark(2, "pass")
    with pytest.raises(R5G.GuardError):
        o.release(R5.RESULTS / "regression_check.json")
    o.mark(3, "skip")
    o.mark(4, "pass")
    with pytest.raises(R5G.GuardError):                     # a skipped item never releases
        o.release(R5.RESULTS / "regression_check.json")
    assert not R5G.is_released()


# ---------------------------------------------------------------- the guard, called directly (D11, T3a-2)

def _ge_state(tmp_path, monkeypatch):
    """A synthetic state, the GE placement and a GE extension built after a synthetic release; then the guard is
    closed again, so every call below happens with the guard closed."""
    make = world_fn(tmp_path, monkeypatch)
    st = make()
    ge = R5G.ge_from_file(S.ge_npz())
    release_guard(tmp_path)
    ext = R5B.extend(st["b"], ge)
    cand = R5F.candidate("G-T", st["b"], ext)
    fam = R5F.run_candidate("G-T", st["b"], ext, cand)
    dv = {"ge": ge, "ext": ext, "cand": {"G-T": cand}, "fam": {"G-T": fam}}
    R5G._reset_for_tests()
    (R5.RESULTS / "regression_check.json").unlink()            # the synthetic release record is not an output
    return st, ge, ext, cand, fam, dv


def _direct_calls(st, ge, ext, cand, fam, dv, tmp_path):
    """Every function that can take Q_GE, called directly with the 'ge' object; the diagnostics get a valid carry.json
    so that only the release guard can refuse them."""
    b = st["b"]
    sel, pi, cl = st["sel"], np.asarray(b.pair_index), np.asarray(b.cl)
    r4 = tmp_path / "results" / "r4.npz"
    cp = R5.RESULTS / "carry.json"
    return {
        "r5_bundle.extend": lambda: R5B.extend(b, ge),
        "r5_bundle.positive_check": lambda: R5B.positive_check(b, ext, ge),
        "r5_bundle.same_as_bundle": lambda: R5B.same_as_bundle(b, ext),
        "r5_bundle.validate_ext": lambda: R5B.validate_ext(ext),
        "r5_bundle.check_pair": lambda: R5B.check_pair(b, ext),
        "r5_bundle.require_ext": lambda: R5B.require_ext(ext, "r5_bundle.require_ext"),
        "r5_bundle.exts_equal": lambda: R5B.exts_equal(ext, ext),
        "r5_bundle.save_ext": lambda: R5B.save_ext(ext, r4, {"r3": "x", "a1": "y"}),
        "r5_bundle.load_ext": lambda: R5B.load_ext(r4, {"r3": "x", "a1": "y"}, "z", ge, b),
        "r5_fusion.candidate": lambda: R5F.candidate("G-TF", b, ext),
        "r5_fusion.expected_d6": lambda: R5F.expected_d6("G-TF", b, ext),
        "r5_fusion.run_candidate": lambda: R5F.run_candidate("G-T", b, ext, cand),
        "auc_emotion": lambda: R5D.auc_emotion({c: cand["P"][c] for c in CONDITIONS}, pi, ge, cp),
        "auc_delta": lambda: R5D.auc_delta(ext.F, pi, ge, cp),
        "pair_lift": lambda: R5D.pair_lift(b.post["affect"]["img"][sel], ge.Q[sel], st["labS"], st["gS"], ge, sel,
                                           cp),
        "reassemble": lambda: R5D.reassemble(b, cand["T"], fam["gates"], fam, ge, cp),
        "sharper_term": lambda: R5D.sharper_term(b, cand["T"], fam["gates"], fam, cl, pi, TAUS, ge, cp),
        "run_r5_seed42.develop_core": lambda: S.develop_core(st, ge),
        "run_r5_seed42.diagnostics": lambda: S.diagnostics(st, dv, {}, ge, cp),
        "run_r5_seed42.write_ge_json": lambda: S.write_ge_json(R5.RESULTS / "dev_seed42.json", {"x": 1}, ge),
        "run_r5_seed42.savez_ge": lambda: S.savez_ge(R5.RESULTS / "seed42_arrays.npz", {"x": np.zeros(2)}, ge),
        "run_r5_seed42.say_ge": lambda: S.say_ge("a GE line", ge),
        "run_r5_seed42.records": lambda: S.records(st, dv),
        "run_r5_seed42.seed42_arrays": lambda: S.seed42_arrays(st, dv),
        "run_r5_seed42.verify_saved": lambda: S.verify_saved(st, dv, S.paths(False)),
    }


GUARDED = ("r5_bundle.extend", "r5_bundle.positive_check", "r5_bundle.same_as_bundle", "r5_bundle.validate_ext",
           "r5_bundle.check_pair", "r5_bundle.require_ext", "r5_bundle.exts_equal", "r5_bundle.save_ext",
           "r5_bundle.load_ext", "r5_fusion.candidate", "r5_fusion.expected_d6", "r5_fusion.run_candidate",
           "auc_emotion", "auc_delta", "pair_lift", "reassemble", "sharper_term", "run_r5_seed42.develop_core",
           "run_r5_seed42.diagnostics", "run_r5_seed42.write_ge_json", "run_r5_seed42.savez_ge",
           "run_r5_seed42.say_ge", "run_r5_seed42.records", "run_r5_seed42.seed42_arrays", "run_r5_seed42.verify_saved")
DOWNSTREAM = ((R5B, "extend"), (R5B, "positive_check"), (R5F, "candidate"), (R5F, "run_candidate"), (RF3, "reader"),
              (RF3, "run_family"), (R5D, "auc_emotion"), (R5D, "auc_delta"), (R5D, "pair_lift"), (R5D, "sharper_term"),
              (R5, "write_json_once"), (S, "_write_npz"), (R5S, "dev_record"))


def _guard_outcome(what, tmp_path, monkeypatch, capsys, bypass=False):
    """Calls `what` directly with the 'ge' object while the guard is closed. bypass=True deletes that function's own
    guard (r5_guard.require answers its own name only), which must change the outcome. -> {"refused_by_itself": a
    GuardError naming the function, "downstream": calls of the downstream spies, "printed", "outputs"}."""
    st, ge, ext, cand, fam, dv = _ge_state(tmp_path, monkeypatch)
    R5.RESULTS.mkdir(parents=True, exist_ok=True)
    (R5.RESULTS / "carry.json").write_text(json.dumps({"rule_sha256": R5.RULE_SHA, "carried": None}))
    calls = _direct_calls(st, ge, ext, cand, fam, dv, tmp_path)
    assert set(calls) == set(GUARDED)
    downstream = {}
    for mod, name in DOWNSTREAM:
        if not (what == name or what.endswith("." + name)):
            downstream[name] = Spy(getattr(mod, name))
            monkeypatch.setattr(mod, name, downstream[name])
    if bypass:
        real = R5G.require

        def require(pl, label, *a, **k):
            return pl if label.split(" ")[0] == what else real(pl, label, *a, **k)
        monkeypatch.setattr(R5G, "require", require)
    capsys.readouterr()
    refused = False
    try:
        calls[what]()
    except R5G.GuardError as e:
        refused = what.split(".")[-1] in str(e)
    except BaseException:                                      # noqa: BLE001  anything else is not a refusal
        refused = False
    return {"refused_by_itself": refused, "downstream": {n: len(s.calls) for n, s in downstream.items() if s.calls},
            "printed": "a GE line" in capsys.readouterr().out,
            "outputs": [x for x in _names(R5.RESULTS) if x != "carry.json"]}


@pytest.mark.parametrize("what", GUARDED)
def test_the_guard_refuses_every_ge_taking_function_called_directly(tmp_path, monkeypatch, capsys, what):
    """Each function refuses the 'ge' object itself (the GuardError names it) before anything downstream runs."""
    out = _guard_outcome(what, tmp_path, monkeypatch, capsys)
    assert out == {"refused_by_itself": True, "downstream": {}, "printed": False, "outputs": []}, out


@pytest.mark.parametrize("what", GUARDED)
def test_deleting_a_functions_own_guard_makes_the_guard_test_fail(tmp_path, monkeypatch, capsys, what):
    """The mutation check of the test above, committed: with that one function's guard deleted, the outcome the test
    above requires does not hold."""
    out = _guard_outcome(what, tmp_path, monkeypatch, capsys, bypass=True)
    assert out != {"refused_by_itself": True, "downstream": {}, "printed": False, "outputs": []}, out


def test_the_diagnostics_are_refused_before_carry_json(tmp_path, monkeypatch):
    st, ge, ext, cand, fam, dv = _ge_state(tmp_path, monkeypatch)
    release_guard(tmp_path)                                    # released, but no carry.json
    spies = {n: Spy(getattr(R5D, n)) for n in ("auc_emotion", "auc_delta", "pair_lift", "sharper_term")}
    for n, s in spies.items():
        monkeypatch.setattr(R5D, n, s)
    with pytest.raises(R5G.GuardError, match="diagnostics"):
        S.diagnostics(st, dv, {}, ge, R5.RESULTS / "carry.json")
    assert all(not s.calls for s in spies.values())
    with pytest.raises(R5G.GuardError):                        # a carry.json of another rule does not open it
        (R5.RESULTS / "carry.json").write_text(json.dumps({"rule_sha256": "0" * 64}))
        S.diagnostics(st, dv, {}, ge, R5.RESULTS / "carry.json")
    assert all(not s.calls for s in spies.values())


# ---------------------------------------------------------------- the development step (D6, D7, T3a-3)

def _released_world(tmp_path, monkeypatch):
    make = world_fn(tmp_path, monkeypatch)
    st = make()
    release_guard(tmp_path)
    return make, st


def test_the_development_step_stores_and_checks_the_gates_it_used(tmp_path, monkeypatch, capsys):
    make, st = _released_world(tmp_path, monkeypatch)
    _patch_record(monkeypatch, CARRIED)
    rec_spy, ext_spy = Spy(R5S.dev_record), Spy(R5B.extend)
    monkeypatch.setattr(R5S, "dev_record", rec_spy)
    monkeypatch.setattr(R5B, "extend", ext_spy)
    P = S.paths(False)
    assert S.development(P, st) == 0
    with np.load(P["arr"]) as z:
        arr = {k: z[k] for k in z.files}
    dev = json.loads(P["dev"].read_text())
    assert set(arr) == set(S.array_keys())
    tp = tuple(dev["tau_prime"])
    assert np.array_equal(arr["tau_prime"], np.asarray(tp)) and np.array_equal(arr["taus"], np.asarray(TAUS))
    for c in CONDITIONS:                                       # G-T's gates are AFF's (D6); G-TF's from tau'
        assert arr[f"gt_gate__{c}"].dtype == np.float32
        assert np.array_equal(arr[f"gt_gate__{c}"], np.stack([st["g_aff"][t][c] for t in range(4)]))
        want = np.stack([((arr[f"gtf_margin__{c}"] >= t) & (arr[f"gtf_pick__{c}"] == 0)).astype(np.float32)
                         for t in tp])
        assert np.array_equal(arr[f"gtf_gate__{c}"], want)
    allm = np.concatenate([arr["gtf_margin__a"], arr["gtf_margin__b"]])
    assert list(tp) == [float(x) for x in np.percentile(allm, [0, 25, 50, 75])]
    assert dev["gates_checked"]["G-T"]["runner"]["gates_equal_definition"] is True
    assert dev["gates_checked"]["G-TF"]["runner"]["taus_equal_definition"] is True
    assert all(v is True for k, v in dev["positive_check_D5"].items())
    assert dev["candidates"]["G-TF"]["cell_text"]["fused"]["0"]["tau"] in tp
    assert set(dev["open_tau0_counts"]) == {"AFF", "G-T", "G-TF"}
    assert all(type(v) is int for d in dev["open_tau0_counts"].values() for v in d.values())
    assert dev["open_tau0_counts"]["G-T"] == dev["open_tau0_counts"]["AFF"]
    for k in ("G-T", "G-TF"):
        assert dev["candidates"][k]["comparator_means"]["Bprime_G"] == 100 * float(np.mean(arr["BpG__r1"]))
        assert dev["candidates"][k]["comparator_means"]["Bprime_A0"] == 100 * float(np.mean(arr["Bp0__r1"]))
    assert dev["comparators"]["Bprime_G_mean_r1"] == 100 * float(np.mean(arr["BpG__r1"]))
    # D8 wiring: each record receives B'(A0) = the bundle's pBp and B'_G = the GE extension's pBp (by identity)
    exts = [r for r in ext_spy.returns if r.kind == "ge"]
    assert len(exts) == 1 and len(rec_spy.calls) == 2
    for a, k in rec_spy.calls:
        assert a[4] is st["b"].pBp and a[5] is exts[0].pBp and a[6] is st["b"].pBp1 and a[2] is st["fam_aff"]
    carry = json.loads(P["carry"].read_text())
    assert carry["dev_seed42_sha256"] == R5.sha256_file(P["dev"]) and carry["carried"] == "G-T"
    diag = json.loads(P["diag"].read_text())
    keys = [k for k in diag if k[:2] in ("a_", "b_", "c_", "d_")]
    assert keys[:4] == ["a_detection_auc", "b_delta_affect_auc", "c_pair_lift", "d_sharper_term"]   # the rule's order
    assert diag["carry_sha256"] == R5.sha256_file(P["carry"])
    # diagnostic (d) describes G-TF's chosen cells with tau' and AFF's and G-T's with tau (fix round 1: the
    # reviewer's mutant passing TAUS for G-TF survived every earlier test)
    assert all(t != u for t, u in zip(tp, TAUS))
    for who, taus in (("AFF", TAUS), ("G-T", TAUS), ("G-TF", tp)):
        cells = diag["d_sharper_term"][who]["cells"]
        for part in ("fused", "cf"):
            for half in ("0", "1"):
                cell = cells[part][half]
                assert cell["tau"] == taus[cell["tau_index"]], (who, part, half)


def test_a_mutated_tau_prime_in_the_shared_function_fires_the_runners_own_check(tmp_path, monkeypatch):
    """T3a-3: tau' perturbed where candidate() and D7's recomputation both read it, so only the runner's independent
    numpy.percentile recomputation can see it."""
    make, st = _released_world(tmp_path, monkeypatch)
    real = R5F.tau_prime

    def mutated(*a, **k):
        t = list(real(*a, **k))
        t[2] += 1e-9
        return tuple(t)
    monkeypatch.setattr(R5F, "tau_prime", mutated)
    with pytest.raises(AssertionError, match=r"rule D7 \(runner\)"):
        S.development(S.paths(False), st)
    assert "dev_seed42.json" not in _names(R5.RESULTS) and "seed42_arrays.npz" not in _names(R5.RESULTS)


def test_a_mutated_tau_prime_inside_the_candidate_fires_d7(tmp_path, monkeypatch):
    """T3a-3: the candidate's gates built from a mutated tau' (self-consistent), the family's D7 check fires."""
    make, st = _released_world(tmp_path, monkeypatch)
    real = R5F.candidate

    def mutated(name, *a, **k):
        cand = real(name, *a, **k)
        if name == "G-TF":
            t = (cand["taus"][0], cand["taus"][1] + 1e-9, cand["taus"][2], cand["taus"][3])
            cand = dict(cand, taus=t, gates=RF3.gates_aff(cand["m"], cand["pick"], t))
        return cand
    monkeypatch.setattr(R5F, "candidate", mutated)
    with pytest.raises(AssertionError, match="D7"):
        S.development(S.paths(False), st)
    assert "dev_seed42.json" not in _names(R5.RESULTS) and "seed42_arrays.npz" not in _names(R5.RESULTS)


def test_a_failed_positive_check_stops_before_any_development_number(tmp_path, monkeypatch, capsys):
    """D5's positive check: an extension that does not carry the GE placement stops the run (nothing written)."""
    make, st = _released_world(tmp_path, monkeypatch)
    real = R5B.positive_check

    def failing(*a, **k):
        return dict(real(*a, **k), affect_slice_differs_from_bundle_on_an_episode=False)
    monkeypatch.setattr(R5B, "positive_check", failing)
    spy = Spy(R5F.candidate)
    monkeypatch.setattr(R5F, "candidate", spy)
    with pytest.raises(AssertionError, match="D5"):
        S.development(S.paths(False), st)
    assert not spy.calls and _names(R5.RESULTS) == ["regression_check.json"]
    assert "positive_check PASS" not in capsys.readouterr().out


def test_g_t_gates_other_than_affs_fire_the_runners_check(tmp_path, monkeypatch):
    make, st = _released_world(tmp_path, monkeypatch)
    g = [{c: np.asarray(x[c]).copy() for c in CONDITIONS} for x in st["g_aff"]]
    g[1]["a"][0] = 1.0 - g[1]["a"][0]
    st["g_aff"] = g                                            # AFF's gates of item 1 differ from what G-T ran from
    with pytest.raises(AssertionError, match=r"rule D7 \(runner\)"):
        S.development(S.paths(False), st)


# ---------------------------------------------------------------- the CARRY and KILL lines (items 7, 8; T3a-5)

@pytest.mark.parametrize("over, line", [(CARRIED, "CARRY G-T (pending the phase-1 agreement, rule §8)"),
                                        (KILLED, "KILL (pending the phase-1 agreement, rule §8)")])
def test_the_carry_and_kill_lines(tmp_path, monkeypatch, capsys, over, line):
    make, st = _released_world(tmp_path, monkeypatch)
    _patch_record(monkeypatch, over)
    P = S.paths(False)
    assert S.development(P, st) == 0
    out = capsys.readouterr().out.splitlines()
    assert line in out
    assert sum(("CARRY" in x) or ("KILL" in x) for x in out) == 1
    carry = json.loads(P["carry"].read_text())
    assert carry["carried"] == ("G-T" if over is CARRIED else None) and carry["kill"] is (over is KILLED)
    assert "diagnostics_seed42.json" in _names(R5.RESULTS)


# ---------------------------------------------------------------- boundary stop and continuation (§8, T3a-4)

def _boundary_run(tmp_path, monkeypatch, capsys, over):
    make = world_fn(tmp_path, monkeypatch)
    no_inputs(monkeypatch)
    monkeypatch.setattr(S, "ITEMS", stub_items(make))
    _patch_record(monkeypatch, over)
    assert S.run(False) == 3
    assert _names(R5.RESULTS) == ["boundary_seed42.json", "dev_seed42.json", "regression_check.json",
                                  "seed42_arrays.npz"]
    out = capsys.readouterr().out
    assert "BOUNDARY" in out and "CARRY" not in out and "KILL" not in out
    R5G._reset_for_tests()
    monkeypatch.setattr(S, "rebuild_state", make)
    return make, R5.sha256_file(R5.RESULTS / "boundary_seed42.json")


@pytest.mark.parametrize("over, carried", [(CARRIED_BOUNDARY, "G-T"), (KILLED_BOUNDARY, None)])
def test_the_boundary_continuation_with_a_carried_candidate_and_with_a_kill(tmp_path, monkeypatch, capsys, over,
                                                                            carried):
    make, sha = _boundary_run(tmp_path, monkeypatch, capsys, over)
    calls = []
    no_inputs(monkeypatch, calls)
    with pytest.raises(SystemExit, match="differs"):
        S.continue_boundary("0" * 64)
    assert calls == [("check_inputs", False)]                  # the input check ran first, even on a refusal
    assert "carry.json" not in _names(R5.RESULTS)
    assert S.continue_boundary(sha) == 0
    out = capsys.readouterr().out.splitlines()
    want = (f"CARRY {carried} (pending the phase-1 agreement, rule §8)" if carried
            else "KILL (pending the phase-1 agreement, rule §8)")
    assert want in out
    carry = json.loads((R5.RESULTS / "carry.json").read_text())
    assert carry["carried"] == carried and carry["boundary_reported"]["boundary_seed42_sha256"] == sha
    assert carry["boundaries"]                                 # the reported boundary is kept in the record
    assert "diagnostics_seed42.json" in _names(R5.RESULTS)
    with pytest.raises(SystemExit, match="exist"):             # a second continuation never overwrites
        S.continue_boundary(sha)
    if carried:                                                # the carried path: sensitivity after the agreement
        line = f"| 09:00 | phase-1 agreement reached: carry.json {R5.sha256_file(R5.RESULTS / 'carry.json')} |"
        S.RUN_LOG.write_text(line + "\n")
        assert S.sensitivity() == 0
        sens = json.loads((R5.RESULTS / "sensitivity.json").read_text())
        assert sens["candidate"] == carried and list(sens["checks"]) == list(S.SENS_ORDER)
    else:
        S.RUN_LOG.write_text("| 09:00 | phase-1 agreement on the kill |\n")
        with pytest.raises(SystemExit, match="no carried candidate"):
            S.sensitivity()


def test_the_continuation_refuses_changed_files_and_a_missing_boundary(tmp_path, monkeypatch, capsys):
    make, sha = _boundary_run(tmp_path, monkeypatch, capsys, CARRIED_BOUNDARY)
    dev = R5.RESULTS / "dev_seed42.json"
    keep = dev.read_text()
    dev.write_text(keep.replace('"G-T"', '"G-T" ', 1))
    with pytest.raises(SystemExit, match="changed"):
        S.continue_boundary(sha)
    dev.write_text(keep)
    bp = R5.RESULTS / "boundary_seed42.json"
    bkeep = bp.read_text()
    bnd = json.loads(bkeep)
    bnd["carry_by_the_stated_inequalities"]["carried"] = "G-TF"     # a boundary record whose carry was altered
    bp.write_text(json.dumps(bnd))
    with pytest.raises(SystemExit, match="boundary record"):
        S.continue_boundary(R5.sha256_file(bp))
    bp.write_text(bkeep)
    bp.unlink()
    with pytest.raises(SystemExit, match="missing"):
        S.continue_boundary(sha)
    assert "carry.json" not in _names(R5.RESULTS)


def test_the_continuation_checks_the_rebuilt_state_against_the_saved_arrays(tmp_path, monkeypatch, capsys):
    make, sha = _boundary_run(tmp_path, monkeypatch, capsys, CARRIED_BOUNDARY)

    def other():
        st = make()
        st["fam_aff"] = dict(st["fam_aff"], fused={**st["fam_aff"]["fused"],
                                                   "r1": 1.0 - np.asarray(st["fam_aff"]["fused"]["r1"])})
        return st
    monkeypatch.setattr(S, "rebuild_state", other)
    with pytest.raises(SystemExit, match="saved"):
        S.continue_boundary(sha)
    assert "diagnostics_seed42.json" not in _names(R5.RESULTS)
    assert "carry.json" not in _names(R5.RESULTS)                # fix round 1: rebuilt and checked before the carry


# ---------------------------------------------------------------- every entry and resume path: inputs first (T3a-1)

@pytest.mark.parametrize("argv", [[], ["--dry"], ["--continue-boundary", "a" * 64], ["--diagnostics-only"],
                                  ["--sensitivity"]])
def test_every_entry_and_resume_path_runs_the_full_input_check_first(tmp_path, monkeypatch, argv):
    seen = []

    def stop(dry=False):
        seen.append(dry)
        raise SystemExit("inputs checked first")
    monkeypatch.setattr(S, "check_inputs", stop)
    spies = {}
    for mod, name in ((RB4, "build_bundle"), (S, "paths"), (R5, "refuse_existing"), (R5G, "require_carry"),
                      (R5G, "release"), (R5G, "ge_from_file"), (S, "no_decimal_console")):
        spies[name] = Spy(getattr(mod, name))
        monkeypatch.setattr(mod, name, spies[name])
    monkeypatch.setattr(S, "ITEMS", tuple(Spy(lambda rec, st: None) for _ in range(4)))
    monkeypatch.setattr(S, "rebuild_state", Spy(lambda: None))
    with pytest.raises(SystemExit, match="inputs checked first"):
        S.main(argv)
    assert seen == [argv == ["--dry"]]
    called = [n for n, s in spies.items() if s.calls and n != "no_decimal_console"]
    assert not called, called                                  # nothing ran before the input check
    assert not any(f.calls for f in S.ITEMS) and not S.rebuild_state.calls
    assert not _names(R5.RESULTS) and not _names(R5.SMOKE)


@pytest.mark.parametrize("patch", ["module", "d12_input", "round3_input", "rule"])
def test_a_patched_hash_stops_the_input_check_before_any_build(tmp_path, monkeypatch, patch):
    monkeypatch.setattr(R5, "_CHECKED", {})
    monkeypatch.setattr(R5.R4C, "_CHECKED", {})
    monkeypatch.setattr(R3, "_CHECKED", {})
    if patch == "module":
        monkeypatch.setitem(R5.MODULE_SHA, "20261121_round3_affect_gate/r3_bundle.py", "0" * 64)
    elif patch == "d12_input":
        monkeypatch.setitem(R5.INPUTS, "src/test/20261122_round4_aff_vetoes/results/seed42_arrays.npz", "0" * 64)
    elif patch == "round3_input":
        monkeypatch.setitem(R3.INPUTS, "20261117_reader_fix_csd/results/rc_tau.json", "0" * 64)
    else:
        monkeypatch.setattr(R5, "RULE_SHA", "0" * 64)
    spy = Spy(RB4.build_bundle)
    monkeypatch.setattr(RB4, "build_bundle", spy)
    items = tuple(Spy(lambda rec, st: None) for _ in range(4))      # never the real items, even if the check fails
    monkeypatch.setattr(S, "ITEMS", items)
    with pytest.raises(SystemExit) as e:
        S.main(["--dry"])
    assert e.value.code not in (0, 1, None)                    # stopped by the input check, not by a finished run
    assert not spy.calls and not any(f.calls for f in items)


def test_the_input_check_is_the_full_one(monkeypatch):
    monkeypatch.setattr(R5, "GOEMO_FILE_SHA", None)            # independent of the state of the real cache
    monkeypatch.setattr(R5, "GE_POST_SHA", None)
    seen = {}
    for mod, name in ((R5, "assert_rule"), (R5, "assert_modules"), (R5, "assert_inputs"), (R3, "assert_taus")):
        spy = Spy(getattr(mod, name))
        monkeypatch.setattr(mod, name, spy)
        seen[name] = spy
    info = S.check_inputs(True)
    assert all(s.calls for s in seen.values())
    names = list(seen["assert_inputs"].calls[0][0][0])
    assert names == list(S.SEED42_INPUTS)
    for n in ("src/test/20261122_round4_aff_vetoes/results/seed42_arrays.npz",
              "src/test/20261122_round4_aff_vetoes/results/dev_seed42.json", S.RC_NPZ, S.RC_TAU, S.TOLD_JSON,
              S.BS07_JSON, S.EXT42, "src/data/artelingo_splits.py", "src/data/wikiart_genre.py"):
        assert n in names, n
    assert tuple(info["taus"]) == TAUS
    monkeypatch.setattr(R5, "GOEMO_FILE_SHA", None)
    with pytest.raises(SystemExit, match="GOEMO_FILE_SHA"):
        S.check_inputs(False)                                  # the real run refuses without the GoEmotions file
    monkeypatch.setattr(R5, "GOEMO_FILE_SHA", "0" * 64)
    (R5.CACHE / "r5_goemotions_selection.npz").write_bytes(b"not the file")
    with pytest.raises(SystemExit, match="GoEmotions"):
        S.check_inputs(True)                                   # a set constant is checked in the dry run too
    for name, const in (("r5_goemotions_selection.npz", "GOEMO_FILE_SHA"), ("r5_ge_posterior.npz", "GE_POST_SHA")):
        (R5.CACHE / name).write_bytes(name.encode())
        monkeypatch.setattr(R5, const, R5.sha256_file(R5.CACHE / name))
    info = S.check_inputs(False)                               # both own files at their constants: the real run passes
    assert info["own_files"] == {"GOEMO_FILE_SHA": R5.GOEMO_FILE_SHA, "GE_POST_SHA": R5.GE_POST_SHA}


# ---------------------------------------------------------------- items 2 and 3, re-asserted from their records

def _ctx(n_sel=6, rows=20):
    sel = np.array([1, 3, 4, 8, 11, 15][:n_sel], dtype=np.int64)
    data = SimpleNamespace(sample_ids=np.arange(rows, dtype=np.int64) * 7 + 3)
    return SimpleNamespace(ctx=SimpleNamespace(selection=sel, data=data))


def _goemo_files(monkeypatch, b, **over):
    sel = np.asarray(b.ctx.selection)
    probs = np.random.default_rng(0).random((len(sel), 28)).astype(np.float32)
    npz = R5.CACHE / "r5_goemotions_selection.npz"
    np.savez(npz, probs=probs, rows=over.pop("rows", sel), sample_ids=over.pop("sample_ids",
                                                                                 b.ctx.data.sample_ids[sel]))
    sha = R5.sha256_file(npz)
    import hashlib
    rec = {"item2": {"passed": True, "max_abs": 4.0e-6, "mean_abs": 1e-8, "n_above_1e-5": 0, "sample_rng": 5,
                     "sample_size": 2048, "tol": 1e-4},
           "npz_sha256": sha, "probs_sha256": hashlib.sha256(probs.tobytes()).hexdigest(), "rule_sha256": R5.RULE_SHA,
           "batch_size": 256, "max_length": 64, "n_rows": len(sel), "n_captions": len(sel), "device": "cpu"}
    for k, v in over.items():
        if k.startswith("item2."):
            rec["item2"][k[6:]] = v
        else:
            rec[k] = v
    (R5.CACHE / "r5_goemotions_selection.json").write_text(json.dumps(rec))
    monkeypatch.setattr(R5, "GOEMO_FILE_SHA", sha)
    monkeypatch.setattr(R5, "N_SELECTION", len(sel))


def _run_item(fn, st):
    rec = S.Recorder()
    fn(rec, st)
    return rec


@pytest.mark.parametrize("over, failing", [
    ({}, None), ({"item2.passed": False}, "item2_passed"), ({"item2.tol": 1e-3}, "item2_tolerance"),
    ({"item2.sample_rng": 6}, "item2_sample_rng"), ({"item2.sample_size": 1024}, "item2_sample_size"),
    ({"item2.max_abs": 2e-4}, "item2_max_abs_within_tolerance"), ({"rule_sha256": "0" * 64}, "record_rule_sha256"),
    ({"npz_sha256": "0" * 64}, "record_npz_sha256_equals_file"), ({"batch_size": 64}, "batch_size"),
    ({"rows": np.array([1, 3, 4, 8, 11, 16])}, "rows_equal_ctx_selection"),
    ({"sample_ids": np.arange(6)}, "sample_ids_equal_ctx_sample_ids_at_selection"),
    ({"probs_sha256": "0" * 64}, "probs_sha256_equals_record")])
def test_item2_is_reasserted_from_the_goemotions_record(monkeypatch, over, failing):
    b = _ctx()
    _goemo_files(monkeypatch, b, **over)
    rec = _run_item(S.item2, {"b": b, "dry": False})
    bad = [r["name"] for r in rec.rows if r["status"] != "pass"]
    assert bad == ([] if failing is None else [failing])


def test_item2_fails_on_a_wrong_file_sha_a_failure_record_or_an_unset_constant(monkeypatch):
    b = _ctx()
    _goemo_files(monkeypatch, b)
    monkeypatch.setattr(R5, "GOEMO_FILE_SHA", "0" * 64)
    assert "goemotions_file_sha256_equals_r5_common" in [r["name"] for r in _run_item(S.item2, {"b": b, "dry": False}).rows
                                                         if r["status"] == "fail"]
    _goemo_files(monkeypatch, b)
    (R5.CACHE / "r5_goemotions_item2_failure.json").write_text("{}")
    assert [r["name"] for r in _run_item(S.item2, {"b": b, "dry": False}).rows
            if r["status"] == "fail"] == ["no_item2_failure_record"]
    monkeypatch.setattr(R5, "GOEMO_FILE_SHA", None)
    assert [r["name"] for r in _run_item(S.item2, {"b": b, "dry": False}).rows
            if r["status"] == "fail"] == ["goemotions_file_sha_constant_set"]


def _placement_files(monkeypatch, b, **over):
    sel = np.asarray(b.ctx.selection)
    post = np.random.default_rng(1).dirichlet(np.ones(41), size=len(sel)).astype(np.float32)
    gp = R5.CACHE / "r5_ge_posterior.npz"
    np.savez(gp, post_sel=post, rows=over.pop("rows", sel), classes=np.arange(41))
    sha = R5.sha256_file(gp)
    item3 = {"passed": True, "equals_fit_one_head": {"txt": True, "img": True},
             "classes_ok": {"txt": True, "img": True}, "heldout_accuracy": {"txt": 35.72, "img": 9.81},
             "n_iter": {"txt": 176, "img": 154}, "accuracy_equals_record_txt": True,
             "accuracy_equals_record_img": True, "draw_positions_ok": True,
             "head_roundtrip_equals_told_oracle_arm_L": True, "head_equals_constants": True}
    rec = {"rule_sha256": R5.RULE_SHA, "item3": item3, "ge_head": {"fallback_used": False, "n_iter": 120},
           "ge_posterior_sha256": sha}
    for k, v in over.items():
        if k.startswith("item3."):
            item3[k[6:]] = v
        else:
            rec[k] = v
    R5.RESULTS.mkdir(parents=True, exist_ok=True)
    (R5.RESULTS / "placement.json").write_text(json.dumps(rec))
    monkeypatch.setattr(R5, "GE_POST_SHA", sha)


@pytest.mark.parametrize("over, failing", [
    ({}, None), ({"item3.passed": False}, "item3_passed"),
    ({"item3.equals_fit_one_head": {"txt": True, "img": False}}, "item3_equals_fit_one_head_img"),
    ({"item3.heldout_accuracy": {"txt": 35.71, "img": 9.81}}, "item3_heldout_accuracy_txt"),
    ({"item3.draw_positions_ok": False}, "item3_draw_positions_ok"),
    ({"rule_sha256": "0" * 64}, "record_rule_sha256"),
    ({"ge_posterior_sha256": "0" * 64}, "record_ge_posterior_sha256_equals_file"),
    ({"ge_head": None}, "ge_head_record_present"),
    ({"rows": np.array([1, 3, 4, 8, 11, 16])}, "ge_rows_equal_ctx_selection")])
def test_item3_is_reasserted_from_the_placement_record(monkeypatch, over, failing):
    b = _ctx()
    _placement_files(monkeypatch, b, **over)
    rec = _run_item(S.item3, {"b": b, "dry": False})
    bad = [r["name"] for r in rec.rows if r["status"] != "pass"]
    assert bad == ([] if failing is None else [failing])


@pytest.mark.parametrize("fn", [S.item2, S.item3])
def test_items_2_and_3_skip_in_the_dry_run_and_fail_in_the_real_run_when_the_record_is_absent(fn):
    b = _ctx()
    dry = _run_item(fn, {"b": b, "dry": True})
    assert dry.rows and all(r["status"] == "skip" for r in dry.rows)
    assert dry.require(int(fn.__name__[-1]), allow_skip=True) == "skip"
    real = _run_item(fn, {"b": b, "dry": False})
    assert any(r["status"] == "fail" for r in real.rows)
    with pytest.raises(S.Stop):
        real.require(int(fn.__name__[-1]), allow_skip=False)
    with pytest.raises(S.Stop):                                # a skip never passes in the real run
        dry.require(int(fn.__name__[-1]), allow_skip=False)


# ---------------------------------------------------------------- --sensitivity (§6.1, rule check S1)

def test_sensitivity_reproduces_se_and_x_on_a_hand_example():
    """Three paintings, two episodes each, differences (pp) [100, 100], [0, 50], [0, 0]: grand mean 125/3;
    SSB = 2 (175/3)^2 + 2 (50/3)^2 + 2 (125/3)^2 = 97500/9, MSB = 48750/9; SSW = 1250, MSW = 1250/3; n0 = (6 - 12/6)/2 = 2;
    sigma_a2 = (48750/9 - 3750/9)/2 = 2500; SE^2 = (2500 (9*12 - 6*6) + (1250/3) 3*6) / 18^2 = 187500/324."""
    d = np.array([1.0, 1.0, 0.0, 0.5, 0.0, 0.0])
    cl = np.array([7, 7, 8, 8, 9, 9])
    r = RS3.sensitivity(d, cl)
    se = math.sqrt(187500 / 324)
    assert math.isclose(r["SE"], se, rel_tol=1e-12)
    assert math.isclose(r["x"], 2.80 * se, rel_tol=1e-12) and math.isclose(r["half_width"], 1.96 * se, rel_tol=1e-12)
    assert math.isclose(r["sigma_a2"], 2500, rel_tol=1e-12) and math.isclose(r["sigma_e2"], 1250 / 3, rel_tol=1e-12)


def _sens_files(carried="G-TF", log_line=True, carry_sha_in_line=True):
    """Synthetic regression record, seed42_arrays, dev and carry files, and the run log; the carried candidate's fused
    R@1 minus cosine's is the hand example's difference."""
    R5.RESULTS.mkdir(parents=True, exist_ok=True)
    (R5.RESULTS / "regression_check.json").write_text(json.dumps(
        {"rule_sha256": R5.RULE_SHA, "all_passed": True, "items": {str(i): {"passed": True} for i in range(1, 5)}}))
    n = 6
    rng = np.random.default_rng(5)
    arr = {"cl": np.array([7, 7, 8, 8, 9, 9])}
    for who in ("aff_fused", "aff_cf", "gt_fused", "gt_cf", "gtf_fused", "gtf_cf", "B", "Bp0", "BpG", "Bp1", "cosine",
                "rca"):
        for m in METRICS:
            arr[f"{who}__{m}"] = rng.integers(0, 5, n) / 4.0
    for who in ("aff_cf", "gt_cf", "gtf_cf", "B", "Bp0", "BpG", "Bp1"):
        arr[f"{who}__gain"] = np.zeros(n)
    k = {"G-T": "gt", "G-TF": "gtf"}[carried or "G-T"]
    arr[f"{k}_fused__r1"] = np.array([1.0, 1.0, 0.0, 0.5, 0.0, 0.0])
    arr["cosine__r1"] = np.zeros(n)
    np.savez_compressed(R5.RESULTS / "seed42_arrays.npz", **arr)
    (R5.RESULTS / "dev_seed42.json").write_text(json.dumps(
        {"rule_sha256": R5.RULE_SHA, "seed42_arrays_sha256": R5.sha256_file(R5.RESULTS / "seed42_arrays.npz")}))
    cy = {"rule_sha256": R5.RULE_SHA, "carried": carried, "E": [carried] if carried else [], "kill": carried is None,
          "dev_seed42_sha256": R5.sha256_file(R5.RESULTS / "dev_seed42.json"),
          "seed42_arrays_sha256": R5.sha256_file(R5.RESULTS / "seed42_arrays.npz"),
          "regression_check_sha256": R5.sha256_file(R5.RESULTS / "regression_check.json")}
    (R5.RESULTS / "carry.json").write_text(json.dumps(cy))
    sha = R5.sha256_file(R5.RESULTS / "carry.json")
    lines = ["| 10:00 | CARRY G-TF (pending the phase-1 agreement, rule §8) |"]
    if log_line:
        lines.append(f"| 11:00 | phase-1 agreement reached{': carry.json ' + sha if carry_sha_in_line else ''} |")
    S.RUN_LOG.write_text("\n".join(lines) + "\n")
    return arr


def _sens_env(tmp_path, monkeypatch):
    no_inputs(monkeypatch)
    make = world_fn(tmp_path, monkeypatch)
    make()                                                     # writes the synthetic GE file and its constant


@pytest.mark.parametrize("case", ["no_carry_json", "kill", "no_agreement_line", "only_the_pending_line",
                                  "pending_line_with_the_carry_sha", "agreement_without_the_carry_sha",
                                  "sensitivity_exists", "other_rule_carry", "regression_record_changed"])
def test_the_sensitivity_path_refusals(tmp_path, monkeypatch, case):
    _sens_env(tmp_path, monkeypatch)
    spy, rel = Spy(RS3.sensitivity), Spy(R5G.release)
    monkeypatch.setattr(RS3, "sensitivity", spy)
    monkeypatch.setattr(R5G, "release", rel)
    if case == "no_carry_json":
        _sens_files()
        (R5.RESULTS / "carry.json").unlink()
    elif case == "kill":
        _sens_files(carried=None)
    elif case == "no_agreement_line":
        _sens_files(log_line=False)
        S.RUN_LOG.unlink()
    elif case == "only_the_pending_line":
        _sens_files(log_line=False)
    elif case == "pending_line_with_the_carry_sha":
        _sens_files(log_line=False)
        sha = R5.sha256_file(R5.RESULTS / "carry.json")
        S.RUN_LOG.write_text(f"| 10:00 | CARRY G-TF (pending the phase-1 agreement, rule §8); carry.json {sha} |\n")
    elif case == "agreement_without_the_carry_sha":
        _sens_files(carry_sha_in_line=False)
    elif case == "regression_record_changed":
        _sens_files()
        reg = json.loads((R5.RESULTS / "regression_check.json").read_text())
        (R5.RESULTS / "regression_check.json").write_text(json.dumps(dict(reg, note="changed")))
    elif case == "sensitivity_exists":
        _sens_files()
        (R5.RESULTS / "sensitivity.json").write_text("{}")
    else:
        _sens_files()
        rec = json.loads((R5.RESULTS / "carry.json").read_text())
        rec["rule_sha256"] = "0" * 64
        (R5.RESULTS / "carry.json").write_text(json.dumps(rec))
        S.RUN_LOG.write_text(f"| 11:00 | phase-1 agreement reached: carry.json "
                             f"{R5.sha256_file(R5.RESULTS / 'carry.json')} |\n")
    with pytest.raises((SystemExit, R5G.GuardError)):
        S.sensitivity()
    assert not spy.calls and not rel.calls                     # refused before the guard is released
    if case != "sensitivity_exists":
        assert "sensitivity.json" not in _names(R5.RESULTS)


def test_the_sensitivity_path_nine_checks_in_order_with_the_hand_example(tmp_path, monkeypatch, capsys):
    _sens_env(tmp_path, monkeypatch)
    arr = _sens_files(carried="G-TF")
    assert S.sensitivity() == 0
    sens = json.loads((R5.RESULTS / "sensitivity.json").read_text())
    assert sens["candidate"] == "G-TF" and list(sens["checks"]) == list(S.SENS_ORDER)
    se = math.sqrt(187500 / 324)
    assert math.isclose(sens["checks"]["r1_vs_cosine"]["SE"], se, rel_tol=1e-12)
    assert math.isclose(sens["checks"]["r1_vs_cosine"]["x"], 2.80 * se, rel_tol=1e-12)
    d = S.sensitivity_diffs("G-TF", arr)
    f = lambda k: np.asarray(arr[k], np.float64)  # noqa: E731
    want = {"r1_vs_cosine": f("gtf_fused__r1") - f("cosine__r1"), "r1_vs_rca": f("gtf_fused__r1") - f("rca__r1"),
            "r1_vs_B": f("gtf_fused__r1") - f("B__r1"), "r1_vs_Bprime_A0": f("gtf_fused__r1") - f("Bp0__r1"),
            "r1_vs_Bprime_G": f("gtf_fused__r1") - f("BpG__r1"), "r1_vs_counterpart": f("gtf_fused__r1") - f("gtf_cf__r1"),
            "gain_statistic": f("gtf_fused__gain"), "gain_vs_rca": f("gtf_fused__gain") - f("rca__gain"),
            "r1_vs_AFF": f("gtf_fused__r1") - f("aff_fused__r1")}
    assert list(d) == list(want) and all(np.array_equal(d[k], want[k]) for k in want)
    assert "Bp1" not in json.dumps(sens["checks"])             # B'(A1) is in no check
    out = capsys.readouterr().out
    assert "SENSITIVITY" in out and sens["agreement_line"].startswith("| 11:00 | phase-1 agreement")


def test_the_sensitivity_path_refuses_changed_arrays(tmp_path, monkeypatch):
    _sens_env(tmp_path, monkeypatch)
    _sens_files()
    with np.load(R5.RESULTS / "seed42_arrays.npz") as z:
        arr = {k: z[k] for k in z.files}
    arr["cosine__r1"] = arr["cosine__r1"] + 0.25
    np.savez_compressed(R5.RESULTS / "seed42_arrays.npz", **arr)
    with pytest.raises(SystemExit, match="SHA-256"):
        S.sensitivity()


# ---------------------------------------------------------------- no non-smoke overwrite

@pytest.mark.parametrize("name", REAL_OUT)
def test_the_real_run_refuses_to_start_when_any_output_exists(tmp_path, monkeypatch, name):
    make = world_fn(tmp_path, monkeypatch)
    calls = []
    no_inputs(monkeypatch)
    monkeypatch.setattr(S, "ITEMS", stub_items(make, calls=calls))
    R5.RESULTS.mkdir(parents=True, exist_ok=True)
    (R5.RESULTS / name).write_text("kept")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        S.run(False)
    assert not calls and (R5.RESULTS / name).read_text() == "kept"


def test_the_ge_writers_never_overwrite(tmp_path, monkeypatch):
    st, ge, ext, cand, fam, dv = _ge_state(tmp_path, monkeypatch)
    release_guard(tmp_path)
    p = R5.RESULTS / "dev_seed42.json"
    p.write_text("kept")
    with pytest.raises(SystemExit):
        S.write_ge_json(p, {"x": 1}, ge)
    q = R5.RESULTS / "seed42_arrays.npz"
    q.write_text("kept")
    with pytest.raises(SystemExit):
        S.savez_ge(q, {"x": np.zeros(2)}, ge)
    assert p.read_text() == "kept" and q.read_text() == "kept"


# ---------------------------------------------------------------- the dry run (§8 lapse 5, §10)

def test_the_dry_run_stops_at_the_guard_and_prints_no_decimal(tmp_path, monkeypatch, capsys):
    make = world_fn(tmp_path, monkeypatch)
    no_inputs(monkeypatch)
    monkeypatch.setattr(S, "ITEMS", stub_items(make, skip=(2, 3)))
    spies = {}
    for mod, name in ((R5G, "ge_from_file"), (R5G, "release"), (S, "development"), (S, "develop_core"),
                      (R5F, "candidate"), (R5S, "dev_record")):
        spies[name] = Spy(getattr(mod, name))
        monkeypatch.setattr(mod, name, spies[name])
    (R5.RESULTS).mkdir(parents=True, exist_ok=True)
    assert S.run(True) == 0
    assert all(not s.calls for s in spies.values())
    assert not R5G.is_released()
    assert _names(R5.RESULTS) == []                            # nothing outside results/smoke/
    assert _names(R5.SMOKE) == ["seed42_dry.json"]             # the value file was checked and deleted
    summ = json.loads((R5.SMOKE / "seed42_dry.json").read_text())
    assert summ["passed"] is True and summ["items"] == {"1": "pass", "2": "skip", "3": "skip", "4": "pass"}
    out = capsys.readouterr().out
    assert not LEAK.search(out), [x for x in out.splitlines() if LEAK.search(x)]
    assert "CHECK item2 stub_record_2 SKIP" in out and "ITEM 2 SKIP" in out and "ITEM 2 PASS" not in out
    assert "GUARD" in out and "SEED42_DRY PASS" in out


def test_the_dry_run_withholds_a_line_with_a_decimal_and_fails(tmp_path, monkeypatch, capsys):
    make = world_fn(tmp_path, monkeypatch)
    no_inputs(monkeypatch)
    monkeypatch.setattr(S, "ITEMS", stub_items(make, skip=(2, 3), say="a value 0.5 and .5 and 5e-03 and 19.1"))
    assert S.run(True) == 1
    out = capsys.readouterr().out
    assert not LEAK.search(out), [x for x in out.splitlines() if LEAK.search(x)]
    assert "withheld" in out and "SEED42_DRY FAIL" in out


def test_the_dry_run_fails_when_its_regression_record_is_not_what_it_ran(tmp_path, monkeypatch, capsys):
    """dry_finish reads the record back: one that lost item 4 (or says it is not a dry run) fails the dry run."""
    make = world_fn(tmp_path, monkeypatch)
    no_inputs(monkeypatch)
    monkeypatch.setattr(S, "ITEMS", stub_items(make, skip=(2, 3)))
    w0 = R5.write_json_once

    def w(path, rec, smoke):
        if Path(path).name == "regression_check_dry.json":
            rec = dict(rec, items={k: v for k, v in rec["items"].items() if k != "4"})
        return w0(path, rec, smoke)
    monkeypatch.setattr(R5, "write_json_once", w)
    assert S.run(True) == 1
    summ = json.loads((R5.SMOKE / "seed42_dry.json").read_text())
    assert summ["passed"] is False and summ["checks"]["regression_record"] is False
    assert "regression_check_dry.json" in _names(R5.SMOKE)     # kept for tracing
    assert "SEED42_DRY FAIL" in capsys.readouterr().out


def test_a_failed_dry_run_keeps_its_regression_record_and_releases_nothing(tmp_path, monkeypatch, capsys):
    make = world_fn(tmp_path, monkeypatch)
    no_inputs(monkeypatch)
    monkeypatch.setattr(S, "ITEMS", stub_items(make, fail_at=4, skip=(2, 3)))
    assert S.run(True) == 1
    assert _names(R5.SMOKE) == ["regression_check_dry.json"] and _names(R5.RESULTS) == []
    assert not R5G.is_released()
    assert not LEAK.search(capsys.readouterr().out)


# ---------------------------------------------------------------- fix round 1: the exact phase-1 agreement record

@pytest.mark.parametrize("variant, opens", [
    ("exact", True), ("exact_with_date", True), ("not_reached", False), ("not_after", False), ("failed", False),
    ("disagree", False), ("pending", False), ("loose_wording", False), ("wrong_sha", False), ("longer_hex", False)])
def test_the_agreement_record_has_one_exact_format(variant, opens):
    sha = "ab" * 32
    line = {"exact": f"| 18:05 | phase-1 agreement reached: carry.json {sha} |",
            "exact_with_date": f"| 2026-10-08 09:00 | phase-1 agreement reached: carry.json {sha} |",
            "not_reached": f"| 18:05 | phase-1 agreement NOT reached: carry.json {sha} |",
            "not_after": f"| 18:05 | phase-1 agreement reached: carry.json {sha} | NOT confirmed |",
            "failed": f"| 18:05 | failed: phase-1 agreement reached: carry.json {sha} |",
            "disagree": f"| 18:05 | phase-1 agreement reached: carry.json {sha} (rederive disagrees) |",
            "pending": f"| 18:05 | phase-1 agreement reached: carry.json {sha} (pending the review) |",
            "loose_wording": f"| 18:05 | phase-1 agreement was reached, carry.json {sha} |",
            "wrong_sha": f"| 18:05 | phase-1 agreement reached: carry.json {'cd' * 32} |",
            "longer_hex": f"| 18:05 | phase-1 agreement reached: carry.json {sha}0 |"}[variant]
    S.RUN_LOG.write_text("| 17:00 | CARRY G-T (pending the phase-1 agreement, rule §8) |\n" + line + "\n")
    if opens:
        assert S.agreement_line(sha) == line
    else:
        with pytest.raises(SystemExit, match="phase-1 agreement reached: carry.json"):
            S.agreement_line(sha)


# ---------------------------------------------------------------- fix round 1: the pinned comparison names

def test_the_pin_holds_183_and_135_unique_names_without_a_decimal():
    assert set(REAL_PINNED) == {1, 4}
    assert len(REAL_PINNED[1]) == len(set(REAL_PINNED[1])) == 183
    assert len(REAL_PINNED[4]) == len(set(REAL_PINNED[4])) == 135
    for names in REAL_PINNED.values():
        assert S.PIN_ROW not in names and not any(LEAK.search(n) for n in names)
    for must in ("auc_emotion_AFF", "G-TF.taus_equal_D1_each_element", "G-T.taus_equal_D1_each_element",
                 "pair_lift.heads_equal_told_oracle_arms_L_pairs_heads", "G-TF.dev.delta_int_zero_against_AFF"):
        assert must in REAL_PINNED[4], must
    for must in ("AFF.tau0_open_count.a", "AFF_minus_Bprime_A1", "R1.tau_recomputed_equals_rc_tau",
                 "round4_arrays.aff_gate__b", "r3.redundancy.affect_least_redundant_both_directions",
                 "a1.Bprime_A1_mean_r1_equals_rule"):
        assert must in REAL_PINNED[1], must


def test_item1_records_exactly_the_pinned_names(tmp_path, monkeypatch, capsys):
    """Item 1 on a synthetic seed-42 bundle (round 3's and round 4's compare functions stubbed with their pinned check
    names): values fail, but the set of names is the pin, so deleting any comparison fails this test."""
    monkeypatch.setattr(S, "PINNED_NAMES", REAL_PINNED)
    st0 = world_fn(tmp_path, monkeypatch)()
    b = st0["b"]
    r3n = [n[3:] for n in REAL_PINNED[1] if n.startswith("r3.")]
    a1n = [n[3:] for n in REAL_PINNED[1] if n.startswith("a1.")]
    monkeypatch.setattr(RB4, "build_bundle", lambda seed, smoke: b)
    monkeypatch.setattr(C, "load_bundle", lambda smoke=False: SimpleNamespace())
    monkeypatch.setattr(RB3, "compare_with_round1",
                        lambda bb, r1: {"checks": {n: True for n in r3n}, "redundancy": R3.REDUNDANCY_42})
    monkeypatch.setattr(RB4, "compare_a1_with_round1", lambda bb, r1: {"checks": {n: True for n in a1n}})
    monkeypatch.setattr(RB3, "load_external", lambda bb: st0["ext"])
    rec = S.Recorder()
    S.item1(rec, {"taus": TAUS, "dry": False})
    names = [r["name"] for r in rec.rows]
    assert len(names) == len(set(names)) == 183
    assert S.pinned_names_check(rec, 1) == {"missing": [], "extra": []}
    assert not LEAK.search(capsys.readouterr().out)


def test_item4_records_exactly_the_pinned_names(tmp_path, monkeypatch, capsys):
    """Item 4 on a synthetic seed-42 bundle with the CLIP placement: deleting the AUC, the tau' == tau comparison or any
    other comparison fails this test."""
    monkeypatch.setattr(S, "PINNED_NAMES", REAL_PINNED)
    st = world_fn(tmp_path, monkeypatch)()
    monkeypatch.setattr(S, "selection_labels", lambda bb: (st["labS"], st["gS"]))
    rec = S.Recorder()
    S.item4(rec, st)
    names = [r["name"] for r in rec.rows]
    assert len(names) == len(set(names)) == 135
    assert S.pinned_names_check(rec, 4) == {"missing": [], "extra": []}
    assert not R5G.is_released()


@pytest.mark.parametrize("pin, which", [(("stub_comparison_1", "a_pinned_name"), "missing"), ((), "extra")])
def test_a_missing_or_extra_name_stops_the_real_run_before_release(tmp_path, monkeypatch, capsys, pin, which):
    make = world_fn(tmp_path, monkeypatch)
    calls = []
    no_inputs(monkeypatch)
    monkeypatch.setattr(S, "ITEMS", stub_items(make, calls=calls))
    monkeypatch.setattr(S, "PINNED_NAMES", {1: pin, 4: ("stub_comparison_4",)})
    rel = Spy(R5G.release)
    monkeypatch.setattr(R5G, "release", rel)
    assert S.run(False) == 1
    assert calls == [1] and not rel.calls and not R5G.is_released()
    reg = json.loads((R5.RESULTS / "regression_check.json").read_text())
    assert reg["stopped_at_item"] == 1 and reg["all_passed"] is False
    row = [r for r in reg["comparisons"] if r["name"] == S.PIN_ROW][0]
    assert row["pass"] is False and row["got"][which] == (["a_pinned_name"] if which == "missing"
                                                         else ["stub_comparison_1"])
    assert _names(R5.RESULTS) == ["regression_check.json"]


def test_a_missing_name_fails_the_dry_run(tmp_path, monkeypatch, capsys):
    make = world_fn(tmp_path, monkeypatch)
    no_inputs(monkeypatch)
    monkeypatch.setattr(S, "ITEMS", stub_items(make, skip=(2, 3)))
    monkeypatch.setattr(S, "PINNED_NAMES", {1: ("stub_comparison_1",), 4: ("stub_comparison_4", "a_pinned_name")})
    assert S.run(True) == 1
    summ = json.loads((R5.SMOKE / "seed42_dry.json").read_text())
    assert summ["passed"] is False and summ["checks"]["comparison_names_pinned"] is False
    out = capsys.readouterr().out
    assert "CHECK dry.comparison_names_pinned FAIL" in out and not LEAK.search(out)


# ---------------------------------------------------------------- fix round 1: --diagnostics-only and exit codes

def _crashed_after_carry(tmp_path, monkeypatch, capsys):
    """A real run (stubbed items) whose diagnostics crash after carry.json was written: exit 4, no diagnostics."""
    make = world_fn(tmp_path, monkeypatch)
    no_inputs(monkeypatch)
    monkeypatch.setattr(S, "ITEMS", stub_items(make))
    _patch_record(monkeypatch, CARRIED)
    real = S.diagnostics

    def crash(*a, **k):
        raise RuntimeError("diagnostics crashed")
    monkeypatch.setattr(S, "diagnostics", crash)
    with pytest.raises(SystemExit) as e:
        S.main([])
    assert e.value.code == 4
    assert "carry.json" in _names(R5.RESULTS) and "diagnostics_seed42.json" not in _names(R5.RESULTS)
    assert "EXIT 4" in capsys.readouterr().out
    monkeypatch.setattr(S, "diagnostics", real)
    monkeypatch.setattr(S, "rebuild_state", make)
    R5G._reset_for_tests()
    return make


def test_diagnostics_only_resumes_after_a_crash_after_the_carry(tmp_path, monkeypatch, capsys):
    make = _crashed_after_carry(tmp_path, monkeypatch, capsys)
    calls = []
    no_inputs(monkeypatch, calls)
    vs = Spy(S.verify_saved)
    monkeypatch.setattr(S, "verify_saved", vs)
    with pytest.raises(SystemExit) as e:
        S.main(["--diagnostics-only"])
    assert e.value.code == 0 and calls == [("check_inputs", False)] and len(vs.calls) == 1
    diag = json.loads((R5.RESULTS / "diagnostics_seed42.json").read_text())
    assert diag["carry_sha256"] == R5.sha256_file(R5.RESULTS / "carry.json")
    assert "d_sharper_term" in diag
    R5G._reset_for_tests()
    with pytest.raises(SystemExit, match="refusing to overwrite"):   # written once
        S.main(["--diagnostics-only"])


@pytest.mark.parametrize("case", ["regression_changed", "dev_changed", "arrays_changed", "carry_missing",
                                  "diagnostics_exist", "rebuild_differs"])
def test_diagnostics_only_refusals(tmp_path, monkeypatch, capsys, case):
    make = _crashed_after_carry(tmp_path, monkeypatch, capsys)
    rel = Spy(R5G.release)
    monkeypatch.setattr(R5G, "release", rel)
    res = R5.RESULTS
    if case == "regression_changed":
        reg = json.loads((res / "regression_check.json").read_text())
        (res / "regression_check.json").write_text(json.dumps(dict(reg, note="changed")))
        match = "regression_check.json"
    elif case == "dev_changed":
        dev = json.loads((res / "dev_seed42.json").read_text())
        (res / "dev_seed42.json").write_text(json.dumps(dict(dev, note="changed")))
        match = "dev_seed42.json"
    elif case == "arrays_changed":
        with np.load(res / "seed42_arrays.npz") as z:
            arr = {k: z[k] for k in z.files}
        arr["cosine__r1"] = arr["cosine__r1"] + 0.25
        np.savez_compressed(res / "seed42_arrays.npz", **arr)
        match = "SHA-256"
    elif case == "carry_missing":
        (res / "carry.json").unlink()
        match = "missing"
    elif case == "diagnostics_exist":
        (res / "diagnostics_seed42.json").write_text("{}")
        match = "refusing to overwrite"
    else:
        def other():
            st = make()
            st["fam_aff"] = dict(st["fam_aff"], fused={**st["fam_aff"]["fused"],
                                                       "r1": 1.0 - np.asarray(st["fam_aff"]["fused"]["r1"])})
            return st
        monkeypatch.setattr(S, "rebuild_state", other)
        match = "saved"
    with pytest.raises(SystemExit, match=match):
        S.diagnostics_only()
    if case != "diagnostics_exist":
        assert "diagnostics_seed42.json" not in _names(res)
    if case != "rebuild_differs":
        assert not rel.calls                                   # refused before the guard is released


@pytest.mark.parametrize("case, code", [("ok", 0), ("failed_item", 1), ("boundary", 3), ("after_release", 4)])
def test_exit_codes(tmp_path, monkeypatch, capsys, case, code):
    make = world_fn(tmp_path, monkeypatch)
    no_inputs(monkeypatch)
    monkeypatch.setattr(S, "ITEMS", stub_items(make, fail_at=2 if case == "failed_item" else None))
    _patch_record(monkeypatch, CARRIED_BOUNDARY if case == "boundary" else CARRIED)
    if case == "after_release":
        def boom(*a, **k):
            raise RuntimeError("crash after the release")
        monkeypatch.setattr(S, "develop_core", boom)
    with pytest.raises(SystemExit) as e:
        S.main([])
    assert e.value.code == code


def test_an_exception_before_the_release_keeps_its_own_type(tmp_path, monkeypatch):
    def boom(dry=False):
        raise RuntimeError("before the release")
    monkeypatch.setattr(S, "check_inputs", boom)
    with pytest.raises(RuntimeError, match="before the release"):
        S.main([])
