"""Tests of run_r6_held.py's regression mode and run_r6_sensitivity.py without real features (ticket 07; rule section 6
items 4, 5 and 7; contracts section 7). The real-data tests (seed 42, selection rows, the exit-3 mutations) are in
test_r6_regression_seed42.py.

Here: the targets against round 3's constants and dev_seed42.json; the module closure each runner ran; the order guard
(refit_check.json and picks_seed42.json missing, not passed, or written by other module bytes) on records in tmp_path;
the output wiring of a regression run with the heavy steps stubbed (a synthetic per-anchor dict of the real shapes:
12,288 episodes, quarter multiples, CF gain 0) and the exit codes; the sensitivity runner on that output, its sigma
parts equal to r3_stats.sensitivity's. Mutation tests load a copy of the runner from tmp_path with its
`# guard:<name>` statements replaced by `pass` (never in place). Real files read: the rule's inputs (baselines_seed42,
dev_seed42.json) and the r6 modules' bytes; no feature, no label, no held row.

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python -m pytest -q -p no:cacheprovider \
        src/test/20261125_artelingo_held_test/test_r6_regression.py
"""
import ast
import importlib.util
import itertools
import json
import re
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r6_common as R  # noqa: E402  (first, before anything that imports src)
import r6_picks as P  # noqa: E402
import r6_stats as ST  # noqa: E402
import run_r6_held as RH  # noqa: E402
import run_r6_sensitivity as RS6  # noqa: E402

import numpy as np  # noqa: E402

RS = R.RS
N_EP = 3 * R.N_PER_PAIR
_COUNT = itertools.count()
GUARDED = ("run_r6_held.py", "run_r6_sensitivity.py")
COEF = {h: {"img": "a" * 64, "txt": "b" * 64} for h in ("affect", "affect_km", "image", "caption", "csd")}
PICKS = {"B": {"0": [4.0, 16.0], "1": [0.5, 0.25]}, "B0": {"0": [0.0, 1.0], "1": [2.0, 0.0]},
         "B1": {"0": [16.0, 8.0], "1": [1.0, 4.0]}}
DECIMAL = re.compile(r"\d*\.\d+")


class Reached(Exception):
    """Raised by a stub: the run went past the point the test says it must not reach."""


def mutant(tmp_path, module, *guards):
    """Import a copy of ``module`` (file name in HERE) with its `# guard:<g>` statements replaced by `pass`."""
    src = (HERE / module).read_text()
    lines = src.splitlines(keepends=True)
    for guard in guards:
        hits = [n for n in ast.walk(ast.parse(src))
                if isinstance(n, (ast.Expr, ast.Assign)) and f"# guard:{guard}" in lines[n.end_lineno - 1]]
        assert hits, f"no statement of {module} carries # guard:{guard}"
        for n in hits:
            first = lines[n.lineno - 1]
            lines[n.lineno - 1] = first[:len(first) - len(first.lstrip())] + "pass\n"
            for i in range(n.lineno, n.end_lineno):
                lines[i] = "\n"
        src = "".join(lines)
        lines = src.splitlines(keepends=True)
    path = tmp_path / f"{Path(module).stem}_mut{next(_COUNT)}.py"
    path.write_text(src)
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


# ---------------------------------------------------------------- records in tmp_path

def write_records(results, shas=None, refit=None, picks=None):
    """refit_check.json and picks_seed42.json as their runners write them (passed, the current module SHAs)."""
    shas = dict(shas or R.r6_module_shas())
    results.mkdir(parents=True, exist_ok=True)
    rec = {"passed": True, "items": {}, "affect_identity": True, "coef_sha256": COEF, "module_sha256": dict(shas),
           "time": "2026-10-09 15:00:00", **(refit or {})}
    (results / RH.REFIT_NAME).write_text(json.dumps(rec))
    prec = {**PICKS, "mean_r1": dict(P.MEAN_R1_TARGETS), "targets": dict(P.MEAN_R1_TARGETS), "passed": True,
            "coef_sha256": COEF, "module_sha256": dict(shas), "time": "2026-10-09 15:00:00", **(picks or {})}
    (results / RH.PICKS_NAME).write_text(json.dumps(prec))
    return results


def key(name):
    return RH._key(name, HERE)


def edit(results, name, **changes):
    p = results / name
    rec = json.loads(p.read_text())
    rec.update(changes)
    p.write_text(json.dumps(rec))


def stale_shas(name):
    shas = R.r6_module_shas()
    shas[key(name)] = "0" * 64
    return shas


# ---------------------------------------------------------------- bookkeeping

def test_every_marked_guard_has_a_mutation_test():
    names = set()
    for module in GUARDED:
        names |= {(module, g) for g in re.findall(r"# guard:([a-z_0-9]+)", (HERE / module).read_text())}
    text = Path(__file__).read_text()
    covered = set()
    for module, args in re.findall(r'mutant\(tmp_path, "(run_r6_held\.py|run_r6_sensitivity\.py)", ([^)]*)\)', text):
        covered |= {(module, g) for g in re.findall(r'"([a-z_0-9]+)"', args)}
    assert names and names <= covered, sorted(names - covered)


def test_targets_are_the_rule_numbers_and_round3_constants():
    A, N = R.R3.AFF_BRAINSTORM, R.R3.RC_NUMBERS
    rule = (HERE / "DECISION_RULE.md").read_text()
    for k, v in RH.TARGETS.items():
        for x in (v if isinstance(v, list) else [v]):
            if isinstance(x, float):
                assert repr(x) in rule, (k, x)                     # every number is written in the rule's text
    assert RH.TARGETS["aff_bar_margin"] == list(A["bar"]) and RH.TARGETS["r1_margin_vs_counterpart"] == list(N["bar"])
    assert RH.TARGETS["aff_bar_comparator"] == "B_prime" and RH.TARGETS["r1_bar_comparator"] == "counterpart"
    assert RH.EXTRA_TARGETS["aff_quarter_hits"] == 9406
    assert RH.check_targets() == {RH.DEV42_REL: R.INPUT_SHA256[RH.DEV42_REL]}


def test_ran_modules_follow_imports():
    assert RH.ran_modules("run_r6_refit.py") == ("r6_common.py", "r6_heads.py", "run_r6_refit.py")
    picks = set(RH.ran_modules("run_r6_picks.py"))
    assert picks == {"run_r6_picks.py", "r6_common.py", "r6_bundle.py", "r6_context.py", "r6_episodes.py",
                     "r6_heads.py", "r6_picks.py", "r6_score.py"}            # r6_score: imported inside a function
    held = set(RH.ran_modules("run_r6_held.py"))
    assert picks - {"run_r6_picks.py"} <= held and {"r6_stats.py", "run_r6_held.py"} <= held
    assert not any(n.startswith(("r6_gpu", "r6_dts", "r6_ft")) for n in held)


# ---------------------------------------------------------------- the order guard

def test_order_guard_passes_with_current_records(tmp_path):
    res = write_records(tmp_path / "results")
    got = RH.order_guard(res, HERE)
    assert got["refit"]["passed"] and got["picks"]["passed"]
    assert set(got["sha256"]) == {RH.REFIT_NAME, RH.PICKS_NAME}


@pytest.fixture
def no_work(monkeypatch):
    """Stubs on everything run_regression does after the guard: a refusal must come before all of it."""
    def stop(name):
        def f(*a, **k):
            raise Reached(name)
        return f

    for name in ("check_targets", "setup", "seed_bundle", "score_bundle"):
        monkeypatch.setattr(RH, name, stop(name))
    for name in ("load_picks", "frozen_lambdas"):
        monkeypatch.setattr(P, name, stop(name))


def refused(res, capsys):
    code = RH.run_regression(results=res, here=HERE)
    out = capsys.readouterr().out
    return code == RH.EXIT_REFUSE and out.startswith("refused:") and not (res / RH.REGRESSION_NAME).exists()


def test_refuses_without_refit_check(tmp_path, no_work, capsys):
    res = write_records(tmp_path / "results")
    (res / RH.REFIT_NAME).unlink()
    assert refused(res, capsys)


def test_refuses_without_picks(tmp_path, no_work, capsys):
    res = write_records(tmp_path / "results")
    (res / RH.PICKS_NAME).unlink()
    assert refused(res, capsys)


def test_refuses_without_either_file(tmp_path, no_work, capsys):
    res = tmp_path / "results"
    res.mkdir()
    assert refused(res, capsys)


def test_refuses_a_refit_that_did_not_pass(tmp_path, no_work, capsys):
    res = write_records(tmp_path / "results", refit={"passed": False})
    assert refused(res, capsys)
    edit(res, RH.REFIT_NAME, passed="true")                     # only the JSON true counts
    assert refused(res, capsys)


def test_refuses_picks_that_did_not_pass(tmp_path, no_work, capsys):
    res = write_records(tmp_path / "results", picks={"passed": False})
    assert refused(res, capsys)


@pytest.mark.parametrize("name", ["r6_heads.py", "r6_common.py", "run_r6_refit.py"])
def test_refuses_a_refit_with_a_stale_module_sha(tmp_path, no_work, capsys, name):
    res = write_records(tmp_path / "results")
    edit(res, RH.REFIT_NAME, module_sha256=stale_shas(name))
    assert refused(res, capsys)


@pytest.mark.parametrize("name", ["r6_score.py", "r6_episodes.py", "r6_bundle.py", "run_r6_picks.py"])
def test_refuses_picks_with_a_stale_module_sha(tmp_path, no_work, capsys, name):
    res = write_records(tmp_path / "results")
    edit(res, RH.PICKS_NAME, module_sha256=stale_shas(name))
    assert refused(res, capsys)


def test_refuses_a_record_without_a_module_or_without_module_shas(tmp_path, no_work, capsys):
    res = write_records(tmp_path / "results")
    shas = R.r6_module_shas()
    del shas[key("r6_context.py")]
    edit(res, RH.PICKS_NAME, module_sha256=shas)
    assert refused(res, capsys)
    res = write_records(tmp_path / "results2")
    rec = json.loads((res / RH.REFIT_NAME).read_text())
    del rec["module_sha256"]
    (res / RH.REFIT_NAME).write_text(json.dumps(rec))
    assert refused(res, capsys)


def test_a_module_the_runner_did_not_run_does_not_matter(tmp_path):
    res = write_records(tmp_path / "results")
    edit(res, RH.REFIT_NAME, module_sha256=stale_shas("r6_score.py"))      # the refit never ran r6_score
    edit(res, RH.PICKS_NAME, module_sha256=stale_shas("r6_stats.py"))      # nor did the picks run r6_stats
    assert RH.order_guard(res, HERE)["refit"]["passed"]


def test_changed_module_bytes_make_the_records_stale(tmp_path):
    """A copy of the folder (checkout/src/test/F): records written, then one module's bytes change."""
    here = tmp_path / "checkout/src/test" / HERE.name
    here.mkdir(parents=True)
    for p in list(HERE.glob("r6_*.py")) + list(HERE.glob("run_r6_*.py")):
        shutil.copy2(p, here / p.name)
    res = write_records(tmp_path / "results", shas=R.r6_module_shas(here))
    assert RH.order_guard(res, here)
    with open(here / "r6_heads.py", "a") as f:
        f.write("\n# changed\n")
    with pytest.raises(RH.Refused, match="r6_heads.py"):
        RH.order_guard(res, here)
    res2 = write_records(tmp_path / "results2", shas=R.r6_module_shas(here))
    assert RH.order_guard(res2, here)
    with open(here / "r6_score.py", "a") as f:                  # run by the picks, not by the refit
        f.write("\n# changed\n")
    with pytest.raises(RH.Refused, match="picks_seed42.json was written by other bytes of .*r6_score.py"):
        RH.order_guard(res2, here)


# ---- guard mutations (on copies)

def test_guard_refit_missing(tmp_path):
    res = write_records(tmp_path / "results")
    (res / RH.REFIT_NAME).unlink()
    with pytest.raises(RH.Refused):
        RH.order_guard(res, HERE)
    with pytest.raises(FileNotFoundError):                      # the refusal is gone: the run goes on and crashes
        mutant(tmp_path, "run_r6_held.py", "refit_missing").order_guard(res, HERE)


def test_guard_refit_passed(tmp_path):
    res = write_records(tmp_path / "results", refit={"passed": False})
    with pytest.raises(RH.Refused):
        RH.order_guard(res, HERE)
    assert mutant(tmp_path, "run_r6_held.py", "refit_passed").order_guard(res, HERE)


def test_guard_refit_stale(tmp_path):
    res = write_records(tmp_path / "results")
    edit(res, RH.REFIT_NAME, module_sha256=stale_shas("r6_heads.py"))
    with pytest.raises(RH.Refused):
        RH.order_guard(res, HERE)
    assert mutant(tmp_path, "run_r6_held.py", "refit_stale").order_guard(res, HERE)


def test_guard_picks_missing(tmp_path):
    res = write_records(tmp_path / "results")
    (res / RH.PICKS_NAME).unlink()
    with pytest.raises(RH.Refused):
        RH.order_guard(res, HERE)
    with pytest.raises(FileNotFoundError):
        mutant(tmp_path, "run_r6_held.py", "picks_missing").order_guard(res, HERE)


def test_guard_picks_passed(tmp_path):
    res = write_records(tmp_path / "results", picks={"passed": False})
    with pytest.raises(RH.Refused):
        RH.order_guard(res, HERE)
    assert mutant(tmp_path, "run_r6_held.py", "picks_passed").order_guard(res, HERE)


def test_guard_picks_stale(tmp_path):
    res = write_records(tmp_path / "results")
    edit(res, RH.PICKS_NAME, module_sha256=stale_shas("r6_score.py"))
    with pytest.raises(RH.Refused):
        RH.order_guard(res, HERE)
    assert mutant(tmp_path, "run_r6_held.py", "picks_stale").order_guard(res, HERE)


# ---------------------------------------------------------------- a stubbed regression run (output wiring)

def synthetic_scored(rng_seed=3):
    """score_seed's per-anchor dicts for the keys the checks read, real shapes, quarter multiples, CF gain 0."""
    rng = np.random.default_rng(rng_seed)
    n = N_EP

    def pa(cf=False):
        r1 = rng.integers(0, 5, n) / 4.0
        gain = np.zeros(n) if cf else rng.integers(-4, 5, n) / 4.0
        return {"r1": r1, "gain": gain, "other": r1 - gain, "swap": np.zeros(n), "strict": np.zeros(n)}

    out = {k: pa(cf=k == "aff_cf") for k in ("aff_fused", "aff_cf", "cosine", "rca", "B", "B0", "B1", "r1_fused")}
    out["cl"] = rng.integers(0, 3000, n).astype(np.int64)
    out["pair_index"] = np.repeat(np.arange(3), R.N_PER_PAIR)
    return out


def stub_run(mod, monkeypatch, failing=(), seen=None):
    """Stubs for the heavy steps of ``mod.run_regression`` (setup, bundle, scoring, items)."""
    scored = synthetic_scored()
    env = SimpleNamespace(inputs={"x": "c" * 64}, coef_sha256=COEF, readers=None, head_check={"passed": True})
    bundle = SimpleNamespace(mode="selection", seed=R.DEV_SEED, smoke=False, n=N_EP, parity=np.arange(N_EP) % 2,
                             episodes_sha256={p: "d" * 64 for p in R.PAIR_NAMES}, fit_rows_sha256="e" * 64)
    seen = {} if seen is None else seen

    def score(env_, bundle_, picks, lambdas, *, include_pm):
        seen.update(include_pm=include_pm, picks=picks, lambdas=lambdas)
        return scored

    def items(env_, bundle_, scored_, orders):
        return {f"item{i}": {"equal": f"item{i}" not in failing, "got": 1, "want": 1} for i in range(4)}, None

    monkeypatch.setattr(mod, "setup", lambda: env)
    monkeypatch.setattr(mod, "seed_bundle", lambda *a, **k: bundle)
    monkeypatch.setattr(mod, "score_bundle", score)
    monkeypatch.setattr(mod, "regression_items", items)
    return scored


def test_stubbed_regression_writes_its_outputs(tmp_path, monkeypatch, capsys):
    res = write_records(tmp_path / "results")
    seen = {}
    scored = stub_run(RH, monkeypatch, seen=seen)
    assert RH.run_regression(results=res, here=HERE) == 0
    out = capsys.readouterr().out
    assert seen["include_pm"] is True                                       # the regression scores PM and r1_cf
    assert seen["picks"] == P.load_picks(res / RH.PICKS_NAME) and seen["lambdas"] == P.frozen_lambdas()
    assert "PASSED (4 items, 0 failed)" in out and not DECIMAL.search(out.replace(str(tmp_path), ""))
    rec = json.loads((res / RH.REGRESSION_NAME).read_text())
    assert rec["passed"] is True and rec["mode"] == "regression" and rec["seed"] == 42
    assert rec["module_sha256"] == R.r6_module_shas() and rec["runner_sha256"] == R.sha256_file(HERE / "run_r6_held.py")
    assert rec["coef_sha256"] == COEF and RH.DEV42_REL in rec["input_sha256"] and "time" in rec
    assert rec["order_inputs"] == {n: R.sha256_file(res / n) for n in (RH.REFIT_NAME, RH.PICKS_NAME)}
    npz, counts = res / RH.PER_EPISODE_NAME, res / RH.COUNTS_NAME
    assert rec["outputs"] == {RH.PER_EPISODE_NAME: R.sha256_file(npz), RH.COUNTS_NAME: R.sha256_file(counts)}
    with np.load(npz) as z:
        assert set(z.files) == {"cl", "pair_index", "meta"} | {f"diff__{c}" for c in ST.CHECKS + ST.SECONDARY}
        for c, (metric, comp, _) in ST.QUANTITIES.items():
            want = np.asarray(scored["aff_fused"][metric]) - np.asarray(scored[comp][metric])
            assert z[f"diff__{c}"].dtype == np.float64 and np.array_equal(z[f"diff__{c}"], want), c
        assert np.array_equal(z["cl"], scored["cl"]) and np.array_equal(z["pair_index"], scored["pair_index"])
        meta = json.loads(str(z["meta"]))
        assert meta["module_sha256"] == R.r6_module_shas() and meta["coef_sha256"] == COEF and meta["seed"] == 42
    c = json.loads(counts.read_text())
    assert c["mode"] == "regression" and c["seeds"] == [42] and c["n_episodes"] == N_EP
    assert set(c["checks"]) == set(ST.CHECKS) and set(c["secondary"]) == set(ST.SECONDARY)
    assert c["checks"]["P1"]["n"] == int(ST.bootstrap_draws(*ST.check_diffs([scored])["P1"])[1].sum())
    assert c["module_sha256"] == R.r6_module_shas() and c["coef_sha256"] == COEF and "input_sha256" in c


def test_stubbed_regression_with_a_differing_item_exits_3(tmp_path, monkeypatch, capsys):
    res = write_records(tmp_path / "results")
    stub_run(RH, monkeypatch, failing=("item2",))
    assert RH.run_regression(results=res, here=HERE) == RH.EXIT_DIFF == 3
    out = capsys.readouterr().out
    assert "item2: FAIL" in out and "item1: pass" in out and "FAILED (4 items, 1 failed)" in out
    rec = json.loads((res / "regression_seed42_failed.json").read_text())
    assert rec["passed"] is False and rec["n_failed"] == 1 and "outputs" not in rec
    for name in (RH.REGRESSION_NAME, RH.PER_EPISODE_NAME, RH.COUNTS_NAME):
        assert not (res / name).exists(), name


def test_guard_exit_code(tmp_path, monkeypatch, capsys):
    res = write_records(tmp_path / "results")
    mod = mutant(tmp_path, "run_r6_held.py", "exit_code")
    stub_run(mod, monkeypatch, failing=("item2",))
    assert mod.run_regression(results=res, here=HERE) == 0                  # the difference no longer stops the run


def test_main_has_only_the_regression_mode():
    with pytest.raises(SystemExit):
        RH.main(["--mode", "held"])
    with pytest.raises(SystemExit):
        RH.main([])


# ---------------------------------------------------------------- the sensitivity runner

@pytest.fixture
def regressed(tmp_path, monkeypatch, capsys):
    """A results folder holding a stubbed regression's outputs (real shapes)."""
    res = write_records(tmp_path / "results")
    scored = stub_run(RH, monkeypatch)
    assert RH.run_regression(results=res, here=HERE) == 0
    capsys.readouterr()
    return SimpleNamespace(res=res, scored=scored)


def test_sensitivity_equals_r3_stats_parts(regressed, capsys):
    assert RS6.run(regressed.res, HERE) == 0
    out = capsys.readouterr().out
    assert not DECIMAL.search(out.replace(str(regressed.res), ""))
    rec = json.loads((regressed.res / RS6.OUT_NAME).read_text())
    diffs = ST.check_diffs([regressed.scored])
    assert set(ST.CHECKS + ST.SECONDARY) <= set(rec)
    for c in ST.CHECKS + ST.SECONDARY:
        d, cl = diffs[c]
        r3 = RS.sensitivity(d, cl)
        assert rec[c] == {"quantity": ST.QUANTITIES[c][2], "sigma_a2": r3["sigma_a2"], "sigma_eps2": r3["sigma_e2"]}, c
    assert rec["N"] == N_EP and rec["n_paintings"] == len(np.unique(regressed.scored["cl"])) and rec["seed"] == 42
    assert rec["per_episode_sha256"] == R.sha256_file(regressed.res / RH.PER_EPISODE_NAME)
    assert rec["module_sha256"] == R.r6_module_shas() and "time" in rec and rec["input_sha256"]
    assert rec["coef_sha256"] == COEF


def test_sensitivity_refuses_without_a_passed_current_regression(regressed, capsys):
    res = regressed.res
    reg = res / RH.REGRESSION_NAME
    keep = reg.read_text()
    reg.unlink()
    assert RS6.run(res, HERE) == RH.EXIT_REFUSE
    reg.write_text(keep)
    edit(res, RH.REGRESSION_NAME, passed=False)
    assert RS6.run(res, HERE) == RH.EXIT_REFUSE
    reg.write_text(keep)
    edit(res, RH.REGRESSION_NAME, module_sha256=stale_shas("r6_stats.py"))
    assert RS6.run(res, HERE) == RH.EXIT_REFUSE
    reg.write_text(keep)
    with np.load(res / RH.PER_EPISODE_NAME) as z:
        arrays = {k: z[k] for k in z.files}
    arrays["diff__P1"] = arrays["diff__P1"].copy()
    arrays["diff__P1"][0] += 0.25
    np.savez(res / RH.PER_EPISODE_NAME, **arrays)                          # not the file the regression wrote
    assert RS6.run(res, HERE) == RH.EXIT_REFUSE
    assert not (res / RS6.OUT_NAME).exists()
    assert all(line.startswith("refused:") for line in capsys.readouterr().out.splitlines())


def test_guard_regression_missing(regressed, tmp_path):
    (regressed.res / RH.REGRESSION_NAME).unlink()
    with pytest.raises(RH.Refused):
        RS6.guard(regressed.res, HERE)
    with pytest.raises(FileNotFoundError):
        mutant(tmp_path, "run_r6_sensitivity.py", "regression_missing").guard(regressed.res, HERE)


def test_guard_regression_passed(regressed, tmp_path):
    edit(regressed.res, RH.REGRESSION_NAME, passed=False)
    with pytest.raises(RH.Refused):
        RS6.guard(regressed.res, HERE)
    assert mutant(tmp_path, "run_r6_sensitivity.py", "regression_passed").guard(regressed.res, HERE)


def test_guard_regression_stale(regressed, tmp_path):
    edit(regressed.res, RH.REGRESSION_NAME, module_sha256=stale_shas("r6_score.py"))
    with pytest.raises(RH.Refused):
        RS6.guard(regressed.res, HERE)
    assert mutant(tmp_path, "run_r6_sensitivity.py", "regression_stale").guard(regressed.res, HERE)


def test_guard_per_episode_sha(regressed, tmp_path):
    edit(regressed.res, RH.REGRESSION_NAME, outputs={RH.PER_EPISODE_NAME: "0" * 64})
    with pytest.raises(RH.Refused):
        RS6.guard(regressed.res, HERE)
    assert mutant(tmp_path, "run_r6_sensitivity.py", "per_episode_sha").guard(regressed.res, HERE)


def test_real_sensitivity_file_if_present():
    """The real seed-42 files (only when this checkout holds them): sigma parts equal r3_stats.sensitivity's."""
    npz, out = R.RESULTS / RH.PER_EPISODE_NAME, R.RESULTS / RS6.OUT_NAME
    if not (npz.is_file() and out.is_file()):
        pytest.skip("no real seed-42 sensitivity file in this checkout")
    rec = json.loads(out.read_text())
    assert rec["per_episode_sha256"] == R.sha256_file(npz)
    with np.load(npz) as z:
        for c in ST.CHECKS + ST.SECONDARY:
            r3 = RS.sensitivity(z[f"diff__{c}"], z["cl"])
            assert (rec[c]["sigma_a2"], rec[c]["sigma_eps2"]) == (r3["sigma_a2"], r3["sigma_e2"]), c
