"""Tests of the descriptive pass, core (run_r6_descriptive.py, r6_descriptive.py; ticket 13; rule section 10 item 5,
section 8 item 3), on synthetic inputs. The real-data seed-42 stand-in is test_r6_descriptive_seed42.py.

Two module fixtures build a complete results folder in tmp_path, the way the chain writes it: synthetic episodes
saved with r6_episodes.save_episodes; synthetic bundles (real dtypes; 13 candidates, 4 + 4 pairs, the ten RCA and PM
terms, reader features, stack) scored with score_seed(..., include_pm=False) as the held pass does; held_arrays.npz
(r6_descriptive.arrays_from_scored, contracts section 7); held_pass.json (r6_stats.pass_record); a synthetic
sensitivity file; the phase-2 agreement; held_verdict.json written by run_r6_apply_rule.apply itself.
  - smoke: seeds 9001 to 9003, 64 per pair, selection bundles (smoke mode);
  - held: seeds 52 to 54, 4,096 per pair (12,288 episodes per seed, 36,864 pooled; the real shapes), held bundles.
No held row and no real feature is read; real files read: the A0 half-readers, rc_tau.json, the rule. Guards are
removed on copies in tmp_path (`# guard:<name>` statement replaced by `pass`), never in place.

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python -m pytest -q -p no:cacheprovider \
        src/test/20261125_artelingo_held_test/test_r6_descriptive.py
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
import r6_bundle as B  # noqa: E402
import r6_descriptive as D  # noqa: E402
import r6_episodes as E  # noqa: E402
import r6_score as S  # noqa: E402
import r6_stats as ST  # noqa: E402
import run_r6_apply_rule as RA  # noqa: E402
import run_r6_descriptive as RD  # noqa: E402
import run_r6_held as RH  # noqa: E402

import numpy as np  # noqa: E402

from src.eval.aspect_episodes import AspectEpisodes, concat_episodes, episodes_sha256  # noqa: E402
from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS, METRICS, per_anchor  # noqa: E402
from src.eval.aspect_scorers import fused_scores  # noqa: E402

C = R.C
K = 13
N_GROUPS = 4000
DECIMAL = re.compile(r"\d*\.\d+")
_COUNT = itertools.count()
GUARDED = ("run_r6_descriptive.py", "r6_descriptive.py")
NESTED_PICKS = {"B": {0: [4.0, 16.0], 1: [0.5, 0.25]}, "B0": {0: [0.0, 1.0], 1: [2.0, 0.0]},
                "B1": {0: [16.0, 8.0], 1: [1.0, 4.0]}}


def lambdas_all():
    out = {"rca": {0: 0.5, 1: 2.0}}
    for i, name in enumerate(B.PM_NAMES):
        out[name] = {0: [0.0, 0.25, 1.0, float("inf")][i % 4], 1: [4.0, 16.0, 0.5, 32.0][i % 4]}
    return out


class Reached(Exception):
    """Raised by a stub placed after a guard: the scenario went through the guard."""


# ---------------------------------------------------------------- helpers

def mutant(tmp_path, module, *guards):
    """Import a copy of ``module`` (file name in HERE) with its `# guard:<g>` statements replaced by `pass`."""
    src = (HERE / module).read_text()
    lines = src.splitlines(keepends=True)
    for guard in guards:
        hits = [n for n in ast.walk(ast.parse(src))
                if isinstance(n, (ast.Expr, ast.Assign, ast.Raise)) and f"# guard:{guard}" in lines[n.end_lineno - 1]]
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


def free_term(rng, n):
    x = {d: rng.standard_normal((n, K), dtype=np.float32) for d in DIRECTIONS}
    return {c: {d: x[d].copy() for d in DIRECTIONS} for c in CONDITIONS}


def cond_term(rng, n):
    return {c: {d: rng.standard_normal((n, K), dtype=np.float32) for d in DIRECTIONS} for c in CONDITIONS}


def synth_episodes(seed, n_per_pair, rng):
    """build_seed's namespace of random member rows (int64 global row ids)."""
    def rows(shape):
        return rng.integers(0, R.N_ROWS, shape).astype(np.int64)
    per_pair, sha = [], {}
    for (a, b, _), name in zip(R.PAIRS, R.PAIR_NAMES):
        n = n_per_pair
        ep = AspectEpisodes(a, b, rows((n,)), rows((n, K)), rows((n, 4)), rows((n, 4)), rows((n, 4)), rows((n, 4)))
        per_pair.append(ep)
        sha[name] = episodes_sha256(ep)
    n = len(R.PAIRS) * n_per_pair
    return SimpleNamespace(seed=int(seed), per_pair=per_pair, pooled=concat_episodes(per_pair), sha=sha, n=n,
                           pair_index=np.repeat(np.arange(3, dtype=np.int64), n_per_pair),
                           parity=np.arange(n, dtype=np.int64) % 2)


def synth_bundle(eps, groups, mode, rng):
    n = eps.n
    Fa, Fb = rng.standard_normal((n, 18)), rng.standard_normal((n, 18))
    for j in range(3):
        Fb[:, 6 * j + 2] = -Fa[:, 6 * j + 2]
    return SimpleNamespace(
        n=n, mode=mode, seed=eps.seed, smoke=mode == "selection" and eps.seed in R.SMOKE_SEEDS,
        cl=groups[eps.pooled.anchor], parity=np.arange(n) % 2, pair_index=eps.pair_index.copy(),
        anchor=eps.pooled.anchor.copy(), cos=free_term(rng, n), t_n1u=free_term(rng, n), t6u_B=free_term(rng, n),
        t6u_B0=free_term(rng, n), t6u_B1=free_term(rng, n),
        stack={d: rng.random((n, 3, K), dtype=np.float32) for d in DIRECTIONS}, F={"a": Fa, "b": Fb},
        pm_terms={name: cond_term(rng, n) for name in B.PM_TERM_NAMES}, pooled=eps.pooled,
        episodes_sha256=dict(eps.sha))


def sha(path) -> str:
    return R.sha256_file(path)


def write_json(path, rec):
    Path(path).write_text(json.dumps(rec, indent=1))


COEF = {h: {"img": f"{i}a" * 32, "txt": f"{i}b" * 32}
        for i, h in enumerate(("affect", "affect_km", "image", "caption", "csd"))}


def synth_ctx(eps, b, rng):
    """A RowContext's fields that ticket 14's DTS hook reads (masked img and txt shrunk to a placeholder)."""
    return SimpleNamespace(seed=eps.seed, n=eps.n, pooled=eps.pooled, img=rng.standard_normal((8, 4)),
                           txt=rng.standard_normal((8, 4)), cos=b.cos, parity=b.parity, pair_index=b.pair_index)


def started_record(shas, coef=COEF, attempts=None):
    """held_started.json as ticket 08 writes it: this rule, attempts with flags, coef_sha256 and episode hashes."""
    att = attempts or [{"attempt": 1, "time": "2026-10-10 09:00:00", "mode": "held",
                        "flags": {"after_crash": False, "fix": None, "reserve": False, "smoke": False},
                        "coef_sha256": coef, "episodes_sha256": {str(s): h for s, h in shas.items()}}]
    return {"rule_sha256": R.RULE_SHA256, "attempts": att}


def build_dir(out, smoke, readers, rng_seed):
    """A complete results folder of one mode (module docstring), the stage records (refit_check.json) in a sibling
    folder. -> namespace for RD.run."""
    out.mkdir(parents=True, exist_ok=True)
    records = out.parent / "records"
    records.mkdir(exist_ok=True)
    write_json(records / RD.REFIT_NAME, {"passed": True, "coef_sha256": COEF})
    md = RD.mode_of(smoke)
    rng = np.random.default_rng(rng_seed)
    groups = rng.integers(0, N_GROUPS, R.N_ROWS).astype(np.int64)
    bundles, ctx, scored, shas = {}, {}, [], {}
    for s in md.seeds:
        eps = synth_episodes(s, md.n_per_pair, rng)
        E.save_episodes(out / f"held_episodes_seed{s}.npz", eps)
        b = synth_bundle(eps, groups, "selection" if smoke else "held", rng)
        bundles[s], ctx[s] = b, synth_ctx(eps, b, rng)
        scored.append(S.score_seed(b, NESTED_PICKS, lambdas_all(), readers, include_pm=False))
        shas[s] = dict(eps.sha)
    if not smoke:
        write_json(out / "held_started.json", started_record(shas))
        # ticket 14: real mode requires every external job listed; here none was run
        write_json(out / RD.XT.SOURCES_NAME, {k: {"missing": "not run in this test"} for k in RD.XT.SOURCE_KEYS})
    np.savez(out / "held_arrays.npz", **D.arrays_from_scored(scored, md.seeds))
    extra = {"episodes_sha256": shas, "runner_sha256": "0" * 64, "module_sha256": {"run_r6_held.py": "1" * 64}}
    write_json(out / "held_pass.json", ST.pass_record(scored, md.name, md.seeds, extra))
    sens = {c: {"quantity": ST.QUANTITIES[c][2], "SE": 0.1, "x" if c in ST.CHECKS else "x2": 0.3, "x95": 0.28}
            for c in ST.CHECKS + ST.SECONDARY}
    write_json(out / "sensitivity_held.json", sens)
    write_json(out / "rederive_agreement.json",
               {"phase": 2, "smoke": smoke, "all_agree": True, "held_pass_sha256": sha(out / "held_pass.json"),
                "pass_file": "held_pass.json", "n_quantities": 9, "disagreements": [], "time": R.amsterdam_now()})
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(RA, "SMOKE" if smoke else "RESULTS", out)
        RA.apply(smoke=smoke)
    env = SimpleNamespace(readers=readers, split=SimpleNamespace(groups=groups), head_check={"passed": True},
                          inputs={}, coef_sha256=json.loads(json.dumps(COEF)))
    return SimpleNamespace(out=out, records=records, smoke=smoke, env=env, bundles=bundles, ctx=ctx, groups=groups,
                           md=md, scored=scored, shas=shas)


def copy_dir(fx, tmp_path, name="res"):
    dst = tmp_path / name
    shutil.copytree(fx.out, dst)
    return dst


def copy_records(fx, tmp_path, name="records_copy"):
    dst = tmp_path / name
    shutil.copytree(fx.records, dst)
    return dst


def run(fx, out, capsys=None, module=RD, **kw):
    args = dict(smoke=fx.smoke, out=out, env=fx.env, bundle_fn=lambda s: (fx.ctx[s], fx.bundles[s]),
                picks=NESTED_PICKS, lambdas=lambdas_all(), records=fx.records)
    args.update(kw)
    code = module.run(**args)
    text = ""
    if capsys is not None:
        cap = capsys.readouterr()
        text = cap.out + cap.err
    return code, text


def edit_json(path, fn):
    rec = json.loads(Path(path).read_text())
    fn(rec)
    write_json(path, rec)


def rebind(out, verdict="held_verdict.json"):
    """After an edit of the pass or agreement file: the agreement names the pass file's SHA-256 again, and the
    verdict both files' SHA-256s (so that only the edited content differs)."""
    edit_json(out / "rederive_agreement.json", lambda a: a.update(held_pass_sha256=sha(out / "held_pass.json")))
    edit_json(out / verdict, lambda v: v.update(held_pass_sha256=sha(out / "held_pass.json"),
                                                agreement_sha256=sha(out / "rederive_agreement.json")))


@pytest.fixture(scope="module")
def readers():
    return B.load_readers()


@pytest.fixture(scope="module")
def smoke_fx(tmp_path_factory, readers):
    return build_dir(tmp_path_factory.mktemp("smoke") / "smoke", True, readers, 11)


@pytest.fixture(scope="module")
def held_fx(tmp_path_factory, readers):
    return build_dir(tmp_path_factory.mktemp("held") / "results", False, readers, 12)


@pytest.fixture(scope="module")
def held_record(held_fx, tmp_path_factory):
    out = tmp_path_factory.mktemp("held_run") / "results"
    shutil.copytree(held_fx.out, out)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(S, "VERDICT", out / "held_verdict.json")    # score_seed's own guard: PM on held after the verdict
        code, _ = run(held_fx, out)
    assert code == 0
    return json.loads((out / "descriptive.json").read_text())


# ---------------------------------------------------------------- bookkeeping

GUARD_TESTS = {
    "verdict_missing": "test_refuses_without_a_verdict",
    "verdict_rule": "test_refuses_a_verdict_of_another_rule",
    "verdict_mode": "test_real_mode_refuses_a_smoke_verdict",
    "verdict_value": "test_refuses_a_verdict_without_go_or_no_go",
    "file_names": "test_refuses_invalid_file_names",
    "agreement_missing": "test_refuses_without_the_agreement_file",
    "agreement_sha": "test_refuses_an_agreement_whose_sha_does_not_match",
    "agreement_all_agree": "test_refuses_an_agreement_without_all_agree",
    "agreement_smoke": "test_refuses_an_agreement_of_the_other_mode",
    "agreement_pass_file": "test_refuses_an_agreement_naming_another_pass",
    "pass_missing": "test_refuses_without_the_pass_file",
    "pass_sha": "test_refuses_a_pass_file_whose_sha_does_not_match",
    "pass_mode": "test_refuses_a_pass_of_another_mode_or_seeds",
    "output_exists": "test_refuses_when_the_output_exists",
    "episodes_pass": "test_episode_file_hashes_must_be_the_pass_files",
    "head_check": "test_head_check_must_have_passed",
    "refit_missing": "test_refuses_without_refit_check",
    "refit_passed": "test_refuses_a_refit_check_that_did_not_pass",
    "started_missing": "test_refuses_without_the_started_file",
    "started_attempt": "test_refuses_without_the_producing_attempt",
    "coef_refit": "test_refuses_coefficients_unlike_refit_check",
    "coef_started": "test_refuses_coefficients_unlike_the_started_attempt",
    "arrays_keys": "test_load_core_layout",
    "pass_counts": "test_check_pass_counts",
    "arrays_reproduce_pass": "test_arrays_must_reproduce_the_pass",
    "bundle_episodes": "test_bundle_episodes_must_be_the_files",
    "cl_pair_index": "test_cl_must_be_groups_of_the_anchor",
    "core_reproduced": "test_rebuilt_bundle_must_reproduce_the_arrays",
    "two_way_weight": "test_two_way_resample_means_by_hand",
    "two_way_anchor_stream": "test_two_way_anchor_stream_is_cluster_bootstraps",
}


def test_every_marked_guard_has_a_mutation_test():
    found = set()
    for module in GUARDED:
        found |= set(re.findall(r"# guard:([a-z0-9_]+)", (HERE / module).read_text()))
    assert found == set(GUARD_TESTS), (sorted(found - set(GUARD_TESTS)), sorted(set(GUARD_TESTS) - found))
    names = set(globals())
    assert all(t in names for t in GUARD_TESTS.values()), [t for t in GUARD_TESTS.values() if t not in names]


# ---------------------------------------------------------------- the smoke run, end to end

TOP_KEYS = {"what", "mode", "seeds", "n_per_pair", "n_episodes", "n_clusters", "verdict", "consistency", "scorers",
            "labels", "rows", "checks_by_scope", "bar_margin", "aff_minus_b1", "r1_checks", "two_way_bootstrap",
            "item_reuse", "gate_open_shares", "pick_accuracy", "redundancy_D7", "frozen", "external", "rule_sha256",
            "module_sha256", "runner_sha256", "input_sha256", "time", "runtime_s", "coef_sha256"}


def test_smoke_run_end_to_end_prints_no_decimal(smoke_fx, tmp_path, capsys):
    out = copy_dir(smoke_fx, tmp_path)
    code, text = run(smoke_fx, out, capsys)
    assert code == 0, text
    assert DECIMAL.search(text) is None, text
    assert "written:" in text and str(out / "descriptive.json") in text
    rec = json.loads((out / "descriptive.json").read_text())
    assert set(rec) == TOP_KEYS
    assert rec["mode"] == "smoke" and rec["seeds"] == list(R.SMOKE_SEEDS) and rec["n_episodes"] == 9 * R.N_SMOKE
    assert rec["scorers"] == list(S.ALL_SCORERS)
    # ticket 14: without external_sources.json every external row is reported "missing", with no number
    assert list(rec["external"]) == list(RD.XT.NAMES)
    for name, e in rec["external"].items():
        assert e["seeds"] == [] and "external_sources.json" in e["missing"] and name not in rec["rows"], name
    for name in S.ALL_SCORERS:
        r = rec["rows"][name]
        assert set(r["per_seed"]) == {str(s) for s in R.SMOKE_SEEDS} and set(r["per_pair"]) == set(R.PAIR_NAMES)
        assert ("aff_minus" in r["pooled"]) is (name != "aff_fused")
        assert set(r["pooled"]) >= {"r1", "gain", "swap", "n_episodes"}
    assert set(rec["two_way_bootstrap"]["quantities"]) == set(ST.CHECKS + ST.SECONDARY)
    assert rec["consistency"]["arrays_reproduce_pass"] == {c: True for c in ST.CHECKS + ST.SECONDARY}
    v = json.loads((out / "held_verdict.json").read_text())
    assert rec["verdict"]["sha256"] == sha(out / "held_verdict.json") and rec["verdict"]["verdict"] == v["verdict"]
    assert rec["verdict"]["agreement_sha256"] == sha(out / "rederive_agreement.json")
    assert set(rec["redundancy_D7"]) == {str(s) for s in R.SMOKE_SEEDS}
    assert set(rec["gate_open_shares"]) == {"AFF", "R1"}
    assert rec["module_sha256"] == R.r6_module_shas()
    assert rec["coef_sha256"]["heads"] == COEF and rec["coef_sha256"]["held_started"] is None
    assert rec["coef_sha256"]["refit_check"]["sha256"] == sha(smoke_fx.records / RD.REFIT_NAME)
    dev = rec["item_reuse"]["development_seed42"]
    assert dev["split"].startswith("selection rows, seed 42") and dev["anchors"]["slots"] == 3 * R.N_PER_PAIR
    assert dev["candidates"]["slots"] == 13 * 3 * R.N_PER_PAIR and dev["members"]["slots"] == 30 * 3 * R.N_PER_PAIR
    assert rec["input_sha256"][D.DEV_EPISODES_REL] == R.INPUT_SHA256[D.DEV_EPISODES_REL]


def test_smoke_rows_are_the_scored_arrays(smoke_fx, tmp_path):
    """The rows are point_ci of the per-anchor arrays: AFF's from held_arrays.npz, PM and r1_cf from the rescoring;
    the per-scope checks equal the pass file's points and intervals."""
    out = copy_dir(smoke_fx, tmp_path)
    assert run(smoke_fx, out)[0] == 0
    rec = json.loads((out / "descriptive.json").read_text())
    seeds = list(R.SMOKE_SEEDS)
    cl = np.concatenate([np.asarray(s["cl"]) for s in smoke_fx.scored])
    aff = np.concatenate([s["aff_fused"]["r1"] for s in smoke_fx.scored])
    assert rec["rows"]["aff_fused"]["pooled"]["r1"] == C.point_ci(aff, cl)
    full = [S.score_seed(smoke_fx.bundles[s], NESTED_PICKS, lambdas_all(), smoke_fx.env.readers, include_pm=True)
            for s in seeds]
    for name in ("wang", "r1_cf"):
        x = np.concatenate([f[name]["gain"] for f in full])
        assert rec["rows"][name]["pooled"]["gain"] == C.point_ci(x, cl)
        assert rec["rows"][name]["pooled"]["aff_minus"]["r1"] == C.point_ci(
            aff - np.concatenate([f[name]["r1"] for f in full]), cl)
    pas = json.loads((out / "held_pass.json").read_text())
    for c in ST.CHECKS + ST.SECONDARY:
        want = (pas["checks"] if c in ST.CHECKS else pas["secondary"])[c]
        got = rec["checks_by_scope"]["pooled"][c]
        assert (got["point"], got["ci95"]) == (want["point"], want["ci95"]), c


# ---------------------------------------------------------------- the held run (real shapes)

def test_held_run_real_shapes(held_fx, held_record):
    rec = held_record
    assert rec["mode"] == "held" and rec["seeds"] == list(R.HELD_SEEDS) and rec["n_per_pair"] == R.N_PER_PAIR
    assert rec["n_episodes"] == 3 * 3 * R.N_PER_PAIR == 36864
    assert rec["two_way_bootstrap"]["n_episodes"] == 36864
    pas = json.loads((held_fx.out / "held_pass.json").read_text())
    for c in ST.CHECKS + ST.SECONDARY:
        want = (pas["checks"] if c in ST.CHECKS else pas["secondary"])[c]
        got = rec["checks_by_scope"]["pooled"][c]
        assert (got["point"], got["ci95"]) == (want["point"], want["ci95"]), c
        tw = rec["two_way_bootstrap"]["quantities"][c]
        assert (tw["point"], tw["ci95_anchor"]) == (want["point"], want["ci95"]), c
    ir = rec["item_reuse"]["pooled"]
    assert ir["candidates"]["slots"] == 36864 * 13 and ir["members"]["slots"] == 36864 * 30
    assert ir["anchors"]["slots"] == 36864
    for who in ("AFF", "R1"):
        g = rec["gate_open_shares"][who]["pooled"]
        assert set(g) == {f"tau_{t}" for t in range(4)} and g["tau_0"]["n_episodes"] == 36864
    assert rec["pick_accuracy"]["pooled"]["pick_accuracy"]["chance"] == 100 / 3
    assert set(rec["r1_checks"]["pooled"]["checks"]) == set(R.RS.GO_CHECKS)
    hs = rec["coef_sha256"]["held_started"]
    assert rec["coef_sha256"]["heads"] == COEF and hs["attempt"] == 1 and hs["file"] == "held_started.json"
    assert hs["sha256"] == sha(held_fx.out / "held_started.json")


def test_held_pm_needs_the_real_verdict_file(held_fx, tmp_path, capsys):
    """score_seed's own guard: the nine PM scorers are scored on held episodes only once results/held_verdict.json
    exists (here it does not exist at S.VERDICT): the pass stops (exit 5), nothing written."""
    out = copy_dir(held_fx, tmp_path)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(S, "VERDICT", tmp_path / "nowhere" / "held_verdict.json")
        code, text = run(held_fx, out, capsys)
    assert code == RD.EXIT_CONTRADICTION and "held_verdict.json exists" in text
    assert not (out / "descriptive.json").exists()


# ---------------------------------------------------------------- refusals (exit 4), each with its guard removed

def _reach(monkeypatch, mod):
    def stub(*a, **k):
        raise Reached("after_verdict reached")
    monkeypatch.setattr(mod, "after_verdict", stub)


def _refused(code, text, words):
    return code == RD.EXIT_REFUSE and "REFUSED" in text and words in text


def test_refuses_without_a_verdict(smoke_fx, tmp_path, capsys):
    out = copy_dir(smoke_fx, tmp_path)
    (out / "held_verdict.json").unlink()
    code, text = run(smoke_fx, out, capsys)
    assert _refused(code, text, "held_verdict.json is missing")
    assert not (out / "descriptive.json").exists()

    def boom(seed):
        raise Reached("a bundle was built before the refusal")
    assert run(smoke_fx, out, bundle_fn=boom, env=None, picks=None, lambdas=None)[0] == RD.EXIT_REFUSE   # nothing
    # loaded: no setup, no picks, no bundle
    mut = mutant(tmp_path, "run_r6_descriptive.py", "verdict_missing")
    with pytest.raises(FileNotFoundError):                   # without the guard the run cannot refuse cleanly
        run(smoke_fx, out, module=mut)


def test_refuses_a_verdict_of_another_rule(smoke_fx, tmp_path, capsys, monkeypatch):
    out = copy_dir(smoke_fx, tmp_path)
    edit_json(out / "held_verdict.json", lambda v: v.update(rule_sha256="0" * 64))
    assert _refused(*run(smoke_fx, out, capsys), "another rule")
    mut = mutant(tmp_path, "run_r6_descriptive.py", "verdict_rule")
    _reach(monkeypatch, mut)
    with pytest.raises(Reached):
        run(smoke_fx, out, module=mut)


def test_real_mode_refuses_a_smoke_verdict(smoke_fx, held_fx, tmp_path, capsys, monkeypatch):
    """A smoke verdict (mode "smoke") in a real results folder: refused; and a real verdict in smoke mode."""
    out = copy_dir(held_fx, tmp_path)
    shutil.copy2(smoke_fx.out / "held_verdict.json", out / "held_verdict.json")
    assert _refused(*run(held_fx, out, capsys), "'smoke' verdict")

    def boom(seed):
        raise Reached("a held bundle was built before the refusal")
    assert run(held_fx, out, bundle_fn=boom)[0] == RD.EXIT_REFUSE                # no held bundle, no PM score
    # the mode field alone: everything else of the real folder consistent
    out2 = copy_dir(held_fx, tmp_path, "res2")
    edit_json(out2 / "held_verdict.json", lambda v: v.update(mode="smoke"))
    assert _refused(*run(held_fx, out2, capsys), "'smoke' verdict")
    mut = mutant(tmp_path, "run_r6_descriptive.py", "verdict_mode")
    _reach(monkeypatch, mut)
    with pytest.raises(Reached):
        run(held_fx, out2, module=mut)
    # and the reverse: a held verdict in smoke mode
    out3 = copy_dir(smoke_fx, tmp_path, "res3")
    shutil.copy2(held_fx.out / "held_verdict.json", out3 / "held_verdict.json")
    assert _refused(*run(smoke_fx, out3, capsys), "'held' verdict")


def test_refuses_a_verdict_without_go_or_no_go(smoke_fx, tmp_path, capsys, monkeypatch):
    out = copy_dir(smoke_fx, tmp_path)
    edit_json(out / "held_verdict.json", lambda v: v.update(verdict="MAYBE"))
    assert _refused(*run(smoke_fx, out, capsys), "no GO or NO-GO")
    mut = mutant(tmp_path, "run_r6_descriptive.py", "verdict_value")
    _reach(monkeypatch, mut)
    with pytest.raises(Reached):
        run(smoke_fx, out, module=mut)
    # a reserve verdict read without --reserve
    out2 = copy_dir(smoke_fx, tmp_path, "res2")
    edit_json(out2 / "held_verdict.json", lambda v: v.update(reserve=True))
    assert _refused(*run(smoke_fx, out2, capsys), "no GO or NO-GO verdict of this kind")


def test_refuses_invalid_file_names(smoke_fx, tmp_path, capsys, monkeypatch):
    out = copy_dir(smoke_fx, tmp_path)
    shutil.copy2(out / "rederive_agreement.json", out / "rederive_agreement_fix1.json")
    edit_json(out / "held_verdict.json", lambda v: v.update(agreement_file="rederive_agreement_fix1.json"))
    assert _refused(*run(smoke_fx, out, capsys), "no valid agreement and pass file pair")
    mut = mutant(tmp_path, "run_r6_descriptive.py", "file_names")
    _reach(monkeypatch, mut)
    with pytest.raises(Reached):
        run(smoke_fx, out, module=mut)
    for bad in ("../held_pass.json", "held_pass_fix2.json", None):
        out2 = copy_dir(smoke_fx, tmp_path, f"res_{next(_COUNT)}")
        edit_json(out2 / "held_verdict.json", lambda v: v.update(pass_file=bad))
        assert run(smoke_fx, out2)[0] == RD.EXIT_REFUSE


def test_refuses_without_the_agreement_file(smoke_fx, tmp_path, capsys):
    out = copy_dir(smoke_fx, tmp_path)
    (out / "rederive_agreement.json").unlink()
    assert _refused(*run(smoke_fx, out, capsys), "rederive_agreement.json is missing")
    mut = mutant(tmp_path, "run_r6_descriptive.py", "agreement_missing")
    with pytest.raises(FileNotFoundError):
        run(smoke_fx, out, module=mut)


def test_refuses_an_agreement_whose_sha_does_not_match(smoke_fx, tmp_path, capsys, monkeypatch):
    out = copy_dir(smoke_fx, tmp_path)
    edit_json(out / "rederive_agreement.json", lambda a: a.update(time="2026-10-09 23:59:59"))   # content changed
    assert _refused(*run(smoke_fx, out, capsys), "not the verdict's agreement_sha256")
    mut = mutant(tmp_path, "run_r6_descriptive.py", "agreement_sha")
    _reach(monkeypatch, mut)
    with pytest.raises(Reached):
        run(smoke_fx, out, module=mut)


def test_refuses_an_agreement_without_all_agree(smoke_fx, tmp_path, capsys, monkeypatch):
    out = copy_dir(smoke_fx, tmp_path)
    edit_json(out / "rederive_agreement.json", lambda a: a.update(all_agree=False))
    rebind(out)                                           # the verdict names the edited file's SHA-256
    assert _refused(*run(smoke_fx, out, capsys), "phase-2 agreement")
    mut = mutant(tmp_path, "run_r6_descriptive.py", "agreement_all_agree")
    _reach(monkeypatch, mut)
    with pytest.raises(Reached):
        run(smoke_fx, out, module=mut)
    for bad in ({"phase": 1}, {"disagreements": ["P3"]}):
        out2 = copy_dir(smoke_fx, tmp_path, f"res_{next(_COUNT)}")
        edit_json(out2 / "rederive_agreement.json", lambda a: a.update(bad))
        rebind(out2)
        assert run(smoke_fx, out2)[0] == RD.EXIT_REFUSE


def test_refuses_an_agreement_of_the_other_mode(smoke_fx, tmp_path, capsys, monkeypatch):
    out = copy_dir(smoke_fx, tmp_path)
    edit_json(out / "rederive_agreement.json", lambda a: a.update(smoke=False))
    rebind(out)
    assert _refused(*run(smoke_fx, out, capsys), "real agreement record is refused in smoke mode")
    mut = mutant(tmp_path, "run_r6_descriptive.py", "agreement_smoke")
    _reach(monkeypatch, mut)
    with pytest.raises(Reached):
        run(smoke_fx, out, module=mut)


def test_refuses_an_agreement_naming_another_pass(smoke_fx, tmp_path, capsys, monkeypatch):
    out = copy_dir(smoke_fx, tmp_path)
    edit_json(out / "rederive_agreement.json", lambda a: a.update(pass_file="held_pass_fix1.json"))
    rebind(out)
    assert _refused(*run(smoke_fx, out, capsys), "names another pass file")
    mut = mutant(tmp_path, "run_r6_descriptive.py", "agreement_pass_file")
    _reach(monkeypatch, mut)
    with pytest.raises(Reached):
        run(smoke_fx, out, module=mut)


def test_refuses_without_the_pass_file(smoke_fx, tmp_path, capsys):
    out = copy_dir(smoke_fx, tmp_path)
    (out / "held_pass.json").unlink()
    assert _refused(*run(smoke_fx, out, capsys), "held_pass.json, named by the verdict, is missing")
    mut = mutant(tmp_path, "run_r6_descriptive.py", "pass_missing")
    with pytest.raises(FileNotFoundError):
        run(smoke_fx, out, module=mut)


def test_refuses_a_pass_file_whose_sha_does_not_match(smoke_fx, tmp_path, capsys, monkeypatch):
    out = copy_dir(smoke_fx, tmp_path)
    edit_json(out / "held_pass.json", lambda p: p.update(time="2026-10-09 23:59:59"))
    assert _refused(*run(smoke_fx, out, capsys), "not the one the verdict and the agreement checked")
    mut = mutant(tmp_path, "run_r6_descriptive.py", "pass_sha")
    _reach(monkeypatch, mut)
    with pytest.raises(Reached):
        run(smoke_fx, out, module=mut)


def test_refuses_a_pass_of_another_mode_or_seeds(smoke_fx, tmp_path, capsys, monkeypatch):
    out = copy_dir(smoke_fx, tmp_path)
    edit_json(out / "held_pass.json", lambda p: p.update(seeds=[9001, 9002, 9004]))
    rebind(out)
    assert _refused(*run(smoke_fx, out, capsys), "is not a smoke pass of this rule")
    mut = mutant(tmp_path, "run_r6_descriptive.py", "pass_mode")
    _reach(monkeypatch, mut)
    with pytest.raises(Reached):
        run(smoke_fx, out, module=mut)
    out2 = copy_dir(smoke_fx, tmp_path, "res2")
    edit_json(out2 / "held_pass.json", lambda p: p.update(mode="held"))
    rebind(out2)
    assert run(smoke_fx, out2)[0] == RD.EXIT_REFUSE


def test_refuses_when_the_output_exists(smoke_fx, tmp_path, capsys, monkeypatch):
    out = copy_dir(smoke_fx, tmp_path)
    (out / "descriptive.json").write_text("{}")
    assert _refused(*run(smoke_fx, out, capsys), "descriptive.json exists")
    assert (out / "descriptive.json").read_text() == "{}"
    mut = mutant(tmp_path, "run_r6_descriptive.py", "output_exists")
    _reach(monkeypatch, mut)
    with pytest.raises(Reached):
        run(smoke_fx, out, module=mut)
    # without the guard, write_once still never replaces the file
    mut2 = mutant(tmp_path, "run_r6_descriptive.py", "output_exists")
    assert run(smoke_fx, out, module=mut2)[0] == RD.EXIT_REFUSE and (out / "descriptive.json").read_text() == "{}"


def test_other_refusals(smoke_fx, tmp_path):
    out = copy_dir(smoke_fx, tmp_path)
    assert run(smoke_fx, out, reserve=True)[0] == RD.EXIT_REFUSE              # --smoke with --reserve
    assert RD.run(smoke=False, subdir="fix1") == RD.EXIT_REFUSE                # --smoke-subdir without --smoke
    assert RD.run(smoke=True, subdir="../x") == RD.EXIT_REFUSE
    assert RD.main(["--smoke", "--smoke-subdir", "no_such_folder_r6_test"]) == RD.EXIT_REFUSE   # no verdict there


# ---------------------------------------------------------------- contradictions (exit 5)

def test_episode_file_hashes_must_be_the_pass_files(smoke_fx, tmp_path, capsys):
    out = copy_dir(smoke_fx, tmp_path)
    edit_json(out / "held_pass.json",
              lambda p: p["episodes_sha256"]["9001"].update(emotion__style="f" * 64))
    rebind(out)
    code, text = run(smoke_fx, out, capsys)
    assert code == RD.EXIT_CONTRADICTION and "per-pair hashes differ" in text
    assert not (out / "descriptive.json").exists()
    mut = mutant(tmp_path, "run_r6_descriptive.py", "episodes_pass")
    assert run(smoke_fx, out, module=mut)[0] == 0                            # goes through
    # a swapped episode file is caught as well (its own seed and hashes)
    out2 = copy_dir(smoke_fx, tmp_path, "res2")
    shutil.copy2(out2 / "held_episodes_seed9002.npz", out2 / "held_episodes_seed9001.npz")
    assert run(smoke_fx, out2)[0] == RD.EXIT_CONTRADICTION


def test_head_check_must_have_passed(smoke_fx, tmp_path, capsys):
    out = copy_dir(smoke_fx, tmp_path)
    env = SimpleNamespace(**{**vars(smoke_fx.env), "head_check": {"passed": False}})
    code, text = run(smoke_fx, out, capsys, env=env)
    assert code == RD.EXIT_CONTRADICTION and "refit heads" in text
    mut = mutant(tmp_path, "run_r6_descriptive.py", "head_check")
    assert run(smoke_fx, out, module=mut, env=env)[0] == 0


def test_arrays_must_reproduce_the_pass(smoke_fx, tmp_path, capsys, monkeypatch):
    out = copy_dir(smoke_fx, tmp_path)
    with np.load(out / "held_arrays.npz") as z:
        a = {k: z[k] for k in z.files}
    i = int(np.flatnonzero(a["aff_fused__r1"] == 0)[0])
    a["aff_fused__r1"][i] = 0.5                                              # one episode moved by 0.5 R@1
    np.savez(out / "held_arrays.npz", **a)
    code, text = run(smoke_fx, out, capsys)
    assert code == RD.EXIT_CONTRADICTION and "does not reproduce the pass file" in text
    mut = mutant(tmp_path, "r6_descriptive.py", "arrays_reproduce_pass")
    core = D.load_core(out / "held_arrays.npz", list(R.SMOKE_SEEDS), R.N_SMOKE)
    prec = json.loads((out / "held_pass.json").read_text())
    with pytest.raises(AssertionError, match="does not reproduce"):
        D.check_pass(core, prec)
    assert set(mut.check_pass(core, prec)) == set(ST.CHECKS + ST.SECONDARY)   # goes through
    # in the run, the rebuilt bundle then still catches the edited array
    monkeypatch.setattr(RD, "D", mut)
    code, text = run(smoke_fx, out, capsys)
    assert code == RD.EXIT_CONTRADICTION and "does not reproduce held_arrays.npz" in text
    out2 = copy_dir(smoke_fx, tmp_path, "res2")
    (out2 / "held_arrays.npz").unlink()
    assert run(smoke_fx, out2)[0] == RD.EXIT_CONTRADICTION


def test_check_pass_counts(smoke_fx, tmp_path):
    core = D.load_core(smoke_fx.out / "held_arrays.npz", list(R.SMOKE_SEEDS), R.N_SMOKE)
    prec = json.loads((smoke_fx.out / "held_pass.json").read_text())
    assert set(D.check_pass(core, prec)) == set(ST.CHECKS + ST.SECONDARY)
    bad = {**prec, "n_clusters": prec["n_clusters"] + 1}
    with pytest.raises(AssertionError, match="cluster count"):
        D.check_pass(core, bad)
    mut = mutant(tmp_path, "r6_descriptive.py", "pass_counts")
    assert set(mut.check_pass(core, bad)) == set(ST.CHECKS + ST.SECONDARY)


def test_load_core_layout(smoke_fx, tmp_path):
    seeds = list(R.SMOKE_SEEDS)
    with np.load(smoke_fx.out / "held_arrays.npz") as z:
        a = {k: z[k] for k in z.files}
    assert set(a) == set(D.array_keys())
    core = D.load_core(smoke_fx.out / "held_arrays.npz", seeds, R.N_SMOKE)
    for i, s in enumerate(seeds):
        assert np.array_equal(core[s]["cl"], smoke_fx.scored[i]["cl"])
        for name in D.CORE:
            for m in METRICS:
                assert np.array_equal(core[s]["pa"][name][m], smoke_fx.scored[i][name][m])
    # contracts section 7 (amendment 16:22): seed_index holds positions, not seeds; blocks consecutive; the keys
    # exactly the layout (a PM or r1_cf key, a metric before the verdict, is refused); int64 indices
    np.savez(tmp_path / "b.npz", **{**a, "seed_index": np.repeat(np.asarray(seeds, dtype=np.int64), 3 * R.N_SMOKE)})
    with pytest.raises(AssertionError, match="positions"):
        D.load_core(tmp_path / "b.npz", seeds, R.N_SMOKE)
    np.savez(tmp_path / "d.npz", **{**a, "seed_index": np.tile(np.arange(3, dtype=np.int64), 3 * R.N_SMOKE)})
    with pytest.raises(AssertionError, match="positions"):
        D.load_core(tmp_path / "d.npz", seeds, R.N_SMOKE)
    np.savez(tmp_path / "e.npz", **{**a, "cl": a["cl"].astype(np.int32)})
    with pytest.raises(AssertionError, match="int64"):
        D.load_core(tmp_path / "e.npz", seeds, R.N_SMOKE)
    np.savez(tmp_path / "c.npz", **{k: v for k, v in a.items() if k != "r1_fused__swap"})
    with pytest.raises(AssertionError, match="keys missing"):
        D.load_core(tmp_path / "c.npz", seeds, R.N_SMOKE)
    np.savez(tmp_path / "f.npz", **a, wang__r1=a["cosine__r1"])
    with pytest.raises(AssertionError, match="not in the layout"):
        D.load_core(tmp_path / "f.npz", seeds, R.N_SMOKE)
    mut = mutant(tmp_path, "r6_descriptive.py", "arrays_keys")
    assert set(mut.load_core(tmp_path / "f.npz", seeds, R.N_SMOKE)) == set(seeds)       # goes through
    with pytest.raises(KeyError):                         # and a missing key crashes unexplained, not refused
        mut.load_core(tmp_path / "c.npz", seeds, R.N_SMOKE)


# ---------------------------------------------------------------- seed_inputs guards (direct calls)

def _seed_args(fx, seed=9001):
    core = D.load_core(fx.out / "held_arrays.npz", list(R.SMOKE_SEEDS), R.N_SMOKE)
    eps = E.load_episodes(fx.out / f"held_episodes_seed{seed}.npz")
    return dict(bundle=fx.bundles[seed], picks=NESTED_PICKS, lambdas=lambdas_all(), readers=fx.env.readers,
                core=core[seed], episodes=eps, groups=fx.groups)


def test_seed_inputs_contents(smoke_fx):
    d = D.seed_inputs(**_seed_args(smoke_fx))
    n = 3 * R.N_SMOKE
    assert set(d.pa) == set(S.ALL_SCORERS) and d.cand_rows.shape == (n, K) and d.member_rows.shape == (n, 30)
    assert np.array_equal(d.member_rows[:, 0], d.anchor_rows) and np.array_equal(d.member_rows[:, 1:14], d.cand_rows)
    assert np.array_equal(d.cand_cl, smoke_fx.groups[d.cand_rows])
    assert len(d.gates["aff"]) == 4 and d.gates["aff"][0]["a"].shape == (n,)
    assert set(d.redundancy) == {"affect", "image", "caption"} and isinstance(d.affect_least_redundant, bool)
    # D7 on the frozen B: round 3's function on B assembled with B's frozen picks
    b = smoke_fx.bundles[9001]
    fb = S.frozen_nested(b.cos, b.t_n1u, b.t6u_B, NESTED_PICKS["B"], b.parity)
    assert d.redundancy == R._R3B.redundancy(SimpleNamespace(B=fb, stack=b.stack))


def test_bundle_episodes_must_be_the_files(smoke_fx, tmp_path):
    args = _seed_args(smoke_fx)
    bad = SimpleNamespace(**vars(args["bundle"]))
    bad.episodes_sha256 = {**bad.episodes_sha256, "style__genre": "e" * 64}
    with pytest.raises(AssertionError, match="episodes differ from the episode file"):
        D.seed_inputs(**{**args, "bundle": bad})
    mut = mutant(tmp_path, "r6_descriptive.py", "bundle_episodes")
    assert mut.seed_inputs(**{**args, "bundle": bad}).seed == 9001          # goes through


def test_cl_must_be_groups_of_the_anchor(smoke_fx, tmp_path):
    args = _seed_args(smoke_fx)
    core = {**args["core"], "cl": args["core"]["cl"] + 1}
    with pytest.raises(AssertionError, match="cl or pair_index differ"):
        D.seed_inputs(**{**args, "core": core})
    mut = mutant(tmp_path, "r6_descriptive.py", "cl_pair_index")
    assert mut.seed_inputs(**{**args, "core": core}).seed == 9001


def test_rebuilt_bundle_must_reproduce_the_arrays(smoke_fx, tmp_path, capsys):
    """A bundle whose cosine differs from the held pass's (as a bundle rebuilt from other features would): exit 5."""
    args = _seed_args(smoke_fx)
    bad = SimpleNamespace(**vars(args["bundle"]))
    rng = np.random.default_rng(99)
    bad.cos = free_term(rng, bad.n)
    with pytest.raises(AssertionError, match="does not reproduce held_arrays.npz"):
        D.seed_inputs(**{**args, "bundle": bad})
    mut = mutant(tmp_path, "r6_descriptive.py", "core_reproduced")
    assert mut.seed_inputs(**{**args, "bundle": bad}).seed == 9001
    out = copy_dir(smoke_fx, tmp_path)
    code, text = run(smoke_fx, out, capsys,
                     bundle_fn=lambda s: (smoke_fx.ctx[s], bad if s == 9001 else smoke_fx.bundles[s]))
    assert code == RD.EXIT_CONTRADICTION and "seed 9001" in text and not (out / "descriptive.json").exists()


# ---------------------------------------------------------------- the two-way bootstrap and item reuse, by hand

def test_two_way_resample_means_by_hand(tmp_path):
    """3 episodes, anchor clusters (0, 0, 1), two candidate columns with clusters (0, 1), (1, 2), (2, 0), values
    (1, 0.5, -0.25).
      W = (2, 0), V = (1, 2, 0): mean V per episode (1.5, 1, 0.5); w = (3, 2, 0); mean = (3 + 1) / 5 = 0.8.
      W = (1, 1), V = (0, 0, 3): mean V (0, 1.5, 1.5); w = (0, 1.5, 1.5); mean = (0.75 - 0.375) / 3 = 0.125.
      W = (0, 2), V = (0, 3, 0): mean V (1.5, 1.5, 0); w = (0, 0, 0): no weight, refused."""
    x = np.array([1.0, 0.5, -0.25])
    a = np.array([0, 0, 1])
    c = np.array([[0, 1], [1, 2], [2, 0]])
    got = D.resample_means(x, a, c, [[2, 0], [1, 1]], [[1, 2, 0], [0, 0, 3]])
    assert got.shape == (2, 1) and got[0, 0] == 0.8 and got[1, 0] == 0.125
    two = D.resample_means(np.stack([x, 2 * x], axis=1), a, c, [[2, 0]], [[1, 2, 0]])
    assert two.tolist() == [[0.8, 1.6]]
    with pytest.raises(AssertionError, match="total weight 0"):
        D.resample_means(x, a, c, [[0, 2]], [[0, 3, 0]])
    mut = mutant(tmp_path, "r6_descriptive.py", "two_way_weight")
    with np.errstate(invalid="ignore", divide="ignore"):
        assert np.isnan(mut.resample_means(x, a, c, [[0, 2]], [[0, 3, 0]])[0, 0])      # goes through


def test_two_way_draws_are_the_two_streams():
    """Resample r uses W_r = bincount of cluster_bootstrap's r-th draw row (default_rng(42), chunks of 250) and V_r =
    bincount of the r-th row of default_rng([42, 1]) over the candidate clusters."""
    rng = np.random.default_rng(5)
    n, ka, kc = 600, 40, 70
    x = rng.integers(-4, 5, n) / 4.0
    a_cl = rng.integers(0, ka, n) * 7 + 3                   # cluster labels need not be 0..k-1
    c_cl = rng.integers(0, kc, (n, K)) * 11 + 1
    rec = D.two_way_bootstrap(x, a_cl, c_cl, ["q"], return_draws=True)
    _, a_idx = np.unique(a_cl, return_inverse=True)
    _, c_flat = np.unique(c_cl.reshape(-1), return_inverse=True)
    c_idx = c_flat.reshape(c_cl.shape)
    da = np.random.default_rng(42).integers(0, ka, size=(250, ka))
    dc = np.random.default_rng([42, 1]).integers(0, kc, size=(250, kc))
    for r in (0, 1, 249):
        W, V = np.bincount(da[r], minlength=ka), np.bincount(dc[r], minlength=kc)
        assert rec["draws"]["two_way"][r, 0] == D.resample_means(x, a_idx, c_idx, [W], [V])[0, 0]
        assert rec["draws"]["two_way_targets"][r, 0] == D.resample_means(x, a_idx, c_idx[:, :2], [W], [V])[0, 0]
        # by hand: w = W[anchor] x mean of V over the episode's candidates (or its two targets)
        for cols, key in ((slice(None), "two_way"), (slice(0, 2), "two_way_targets")):
            w = W[a_idx] * V[c_idx[:, cols]].mean(axis=1)
            assert np.isclose(rec["draws"][key][r, 0], (w * x).sum() / w.sum(), rtol=1e-13, atol=0)
        assert np.isclose(rec["draws"]["anchor"][r, 0], (np.bincount(a_idx, weights=x)[da[r]].sum()
                                                         / np.bincount(a_idx).astype(float)[da[r]].sum()),
                          rtol=1e-13, atol=0)
    assert rec["n_anchor_clusters"] == ka and rec["n_candidate_clusters"] == kc
    assert rec["n_target_clusters"] == len(np.unique(c_cl[:, :2]))


def test_two_way_anchor_stream_is_cluster_bootstraps(tmp_path, monkeypatch):
    rng = np.random.default_rng(6)
    n = 900
    x = np.stack([rng.integers(-4, 5, n) / 4.0, rng.integers(0, 3, n) / 2.0], axis=1)
    a_cl, c_cl = rng.integers(0, 150, n), rng.integers(0, 400, (n, K))
    rec = D.two_way_bootstrap(x, a_cl, c_cl, ["p", "q"])
    for j, nm in enumerate(("p", "q")):
        ref = C.point_ci(x[:, j], a_cl)
        assert rec["quantities"][nm]["ci95_anchor"] == ref["ci95"] and rec["quantities"][nm]["point"] == ref["point"]
    # a cluster_bootstrap that drew otherwise: the guard fires; without it the run goes through
    def other(values, clusters, n_boot=5000, seed=42, chunk=250):
        return {"point": float(np.mean(values)), "ci95": [-1.0, 1.0], "n_clusters": 2}
    monkeypatch.setattr(D, "cluster_bootstrap", other)
    with pytest.raises(AssertionError, match="anchor-only interval"):
        D.two_way_bootstrap(x[:, 0], a_cl, c_cl, ["p"])
    mut = mutant(tmp_path, "r6_descriptive.py", "two_way_anchor_stream")
    monkeypatch.setattr(mut, "cluster_bootstrap", other)
    assert set(mut.two_way_bootstrap(x[:, 0], a_cl, c_cl, ["p"])["quantities"]) == {"p"}


def test_two_way_reduces_to_anchor_only_without_candidate_variation():
    """Every episode with the same 13 candidate paintings: mean V is one constant per resample, which cancels; the
    two-way interval is the anchor-only one (up to summation order). The targets' side is set to all 13 columns
    here (two shared targets alone would be drawn zero times together in about one resample in nine)."""
    rng = np.random.default_rng(7)
    n = 1200
    x = rng.integers(-4, 5, n) / 4.0
    a_cl = rng.integers(0, 200, n)
    c_cl = np.tile(np.arange(K) * 3, (n, 1))
    q = D.two_way_bootstrap(x, a_cl, c_cl, ["q"], target_cols=tuple(range(K)))["quantities"]["q"]
    for side in ("two_way", "two_way_targets"):
        assert np.allclose(q[f"ci95_{side}"], q["ci95_anchor"], rtol=0, atol=1e-9)
    assert abs(q["half_width_ratio"] - 1) < 1e-9 and abs(q["half_width_ratio_targets"] - 1) < 1e-9


def test_two_way_widens_with_shared_candidate_effects():
    """Outcomes driven by the target candidate's painting (column 0), shared by about 200 episodes each: both two-way
    intervals are wider than the anchor-only one, the targets' side the most; outcomes independent of the candidates:
    the all-candidates side about as wide (the targets' side is wider from weight noise alone: the pigeonhole
    bootstrap is conservative)."""
    rng = np.random.default_rng(8)
    n, kc = 6000, 30
    a_cl = rng.integers(0, 3000, n)
    c_cl = rng.integers(0, kc, (n, K))
    effect = rng.choice([-1.0, 1.0], kc)
    x_dep = np.clip(np.rint(4 * (effect[c_cl[:, 0]] * 0.75 + rng.normal(0, 0.1, n))) / 4, -1, 1)
    x_ind = rng.integers(-4, 5, n) / 4.0
    q = D.two_way_bootstrap(np.stack([x_dep, x_ind], axis=1), a_cl, c_cl, ["dep", "ind"])["quantities"]
    assert q["dep"]["half_width_ratio_targets"] > q["dep"]["half_width_ratio"] > 1.2
    assert 0.9 < q["ind"]["half_width_ratio"] < 1.1


def test_item_reuse_by_hand():
    """Candidates rows (10, 11), (11, 12), (12, 10): 6 slots, 3 distinct rows (reuse 50%), paintings of 10, 11, 12 =
    0, 1, 1: 2 distinct (reuse 100 x (1 - 2/6)); anchors 10, 13: 2 slots, 2 rows, paintings 0, 0 (reuse 50%)."""
    groups = np.zeros(20, dtype=np.int64)
    groups[[11, 12]] = 1
    cand = np.array([[10, 11], [11, 12], [12, 10]])
    r = D.item_reuse(np.array([10, 13]), cand, np.concatenate([cand, cand], axis=1), groups)
    assert r["candidates"] == {"slots": 6, "distinct_rows": 3, "distinct_paintings": 2, "row_reuse_pct": 50.0,
                               "painting_reuse_pct": 100.0 * (1.0 - 2 / 6), "mean_uses_per_row": 2.0,
                               "mean_uses_per_painting": 3.0}
    assert r["anchors"]["row_reuse_pct"] == 0.0 and r["anchors"]["painting_reuse_pct"] == 50.0
    assert r["members"]["slots"] == 12 and r["members"]["distinct_rows"] == 3


# ---------------------------------------------------------------- ticket 14's hook

def test_external_rows_hook(smoke_fx):
    per_seed = [D.seed_inputs(**_seed_args(smoke_fx, s)) for s in R.SMOKE_SEEDS]
    dts = {s: per_seed[i].pa["wang"] for i, s in enumerate(R.SMOKE_SEEDS)}
    mllm = {9001: per_seed[0].pa["tip"]}
    ext = {"DTS": {"pa": dts, "label": "describe-then-score", "info": {"parsing_failures": {"9001": {"a": 3}}}},
           "MLLM": {"pa": mllm, "label": "in-context reranker", "info": {"seeds_read": [9001]}},
           "FT-LoRA": {"pa": {}, "label": "FT-LoRA", "info": {"missing": "the job was not run"}}}
    rec = D.describe(per_seed, external=ext)
    assert rec["rows"]["DTS"] == rec["rows"]["wang"]                   # the same per-anchor dicts give the same rows
    assert set(rec["rows"]["MLLM"]["per_seed"]) == {"9001"}
    assert rec["rows"]["MLLM"]["pooled"] == rec["rows"]["tip"]["per_seed"]["9001"]
    assert "FT-LoRA" not in rec["rows"] and rec["external"]["FT-LoRA"] == {"seeds": [], "missing": "the job was not run"}
    assert rec["external"]["DTS"]["parsing_failures"] == {"9001": {"a": 3}} and rec["labels"]["MLLM"] == \
        "in-context reranker"
    assert rec["scorers"] == list(S.ALL_SCORERS) + ["DTS", "MLLM"]
    assert set(rec["checks_by_scope"]["pooled"]) == set(ST.CHECKS + ST.SECONDARY)
    with pytest.raises(AssertionError, match="collides"):
        D.describe(per_seed, external={"wang": {"pa": dts}})


# ---------------------------------------------------------------- head coefficients (rule section 5 item 5)

def _boom(seed):
    raise Reached("a bundle was built")


def test_refuses_without_refit_check(smoke_fx, tmp_path, capsys):
    out, rec = copy_dir(smoke_fx, tmp_path), copy_records(smoke_fx, tmp_path)
    (rec / RD.REFIT_NAME).unlink()
    assert _refused(*run(smoke_fx, out, capsys, records=rec, bundle_fn=_boom), "refit_check.json is missing")
    mut = mutant(tmp_path, "run_r6_descriptive.py", "refit_missing")
    with pytest.raises(FileNotFoundError):                   # without the guard: an unexplained crash
        run(smoke_fx, out, module=mut, records=rec)


def test_refuses_a_refit_check_that_did_not_pass(smoke_fx, tmp_path, capsys):
    out, rec = copy_dir(smoke_fx, tmp_path), copy_records(smoke_fx, tmp_path)
    edit_json(rec / RD.REFIT_NAME, lambda r: r.update(passed=False))
    assert _refused(*run(smoke_fx, out, capsys, records=rec, bundle_fn=_boom), "did not pass")
    mut = mutant(tmp_path, "run_r6_descriptive.py", "refit_passed")
    assert run(smoke_fx, out, module=mut, records=rec)[0] == 0                 # goes through


def test_refuses_coefficients_unlike_refit_check(smoke_fx, tmp_path, capsys):
    """Smoke mode checks refit_check.json only; a heads' coefficient SHA-256 that differs: refused before any bundle."""
    out, rec = copy_dir(smoke_fx, tmp_path), copy_records(smoke_fx, tmp_path)
    edit_json(rec / RD.REFIT_NAME, lambda r: r["coef_sha256"]["csd"].update(txt="f" * 64))
    assert _refused(*run(smoke_fx, out, capsys, records=rec, bundle_fn=_boom), "differ from refit_check.json's")
    assert not (out / "descriptive.json").exists()
    mut = mutant(tmp_path, "run_r6_descriptive.py", "coef_refit")
    assert run(smoke_fx, out, module=mut, records=rec)[0] == 0                 # goes through


def test_refuses_without_the_started_file(held_fx, tmp_path, capsys):
    out = copy_dir(held_fx, tmp_path)
    (out / "held_started.json").unlink()
    assert _refused(*run(held_fx, out, capsys, bundle_fn=_boom), "held_started.json is missing")
    mut = mutant(tmp_path, "run_r6_descriptive.py", "started_missing")
    with pytest.raises(FileNotFoundError):
        run(held_fx, out, module=mut, bundle_fn=_boom)


def test_refuses_without_the_producing_attempt(held_fx, tmp_path, capsys):
    """No attempt of this rule whose fix flag matches the pass file: refused (here: another rule's started file; and a
    started file holding only a --fix 1 attempt for held_pass.json)."""
    out = copy_dir(held_fx, tmp_path)
    edit_json(out / "held_started.json", lambda r: r.update(rule_sha256="0" * 64))
    assert _refused(*run(held_fx, out, capsys, bundle_fn=_boom), "holds no attempt of this rule")
    mut = mutant(tmp_path, "run_r6_descriptive.py", "started_attempt")
    with pytest.raises(Reached):                             # goes through, to the first bundle
        run(held_fx, out, module=mut, bundle_fn=_boom)
    out2 = copy_dir(held_fx, tmp_path, "res2")
    edit_json(out2 / "held_started.json", lambda r: r["attempts"][0]["flags"].update(fix=1))
    assert _refused(*run(held_fx, out2, capsys, bundle_fn=_boom), "holds no attempt of this rule")


def test_refuses_coefficients_unlike_the_started_attempt(held_fx, tmp_path, capsys):
    out = copy_dir(held_fx, tmp_path)
    edit_json(out / "held_started.json", lambda r: r["attempts"][0]["coef_sha256"]["affect"].update(img="e" * 64))
    assert _refused(*run(held_fx, out, capsys, bundle_fn=_boom), "differ from those of attempt 1 in held_started.json")
    mut = mutant(tmp_path, "run_r6_descriptive.py", "coef_started")
    with pytest.raises(Reached):
        run(held_fx, out, module=mut, bundle_fn=_boom)


def test_producing_attempt_is_chosen_by_the_fix_flag(held_fx, tmp_path):
    """held_pass.json: the last attempt without a fix flag (an --after-crash rerun after a crash); held_pass_fix1.json:
    the last --fix 1 attempt; held_pass_reserve.json: the last attempt of held_started_reserve.json."""
    other = {h: {"img": "c" * 64, "txt": "d" * 64} for h in COEF}

    def att(k, fix, coef, crash=False):
        return {"attempt": k, "flags": {"after_crash": crash, "fix": fix, "reserve": False, "smoke": False},
                "coef_sha256": coef, "episodes_sha256": {}}
    env = held_fx.env
    for attempts, suffix, ok in (([att(1, None, other), att(2, None, COEF, True)], "", True),
                                 ([att(1, None, COEF), att(2, 1, other)], "", True),
                                 ([att(1, None, COEF), att(2, 1, other)], "_fix1", False),
                                 ([att(1, None, other), att(2, 1, COEF)], "_fix1", True)):
        out = tmp_path / f"o{next(_COUNT)}"
        out.mkdir()
        write_json(out / "held_started.json", started_record({}, attempts=attempts))
        vr = SimpleNamespace(suffix=suffix, reserve=False, pass_path=out / f"held_pass{suffix}.json")
        coef = RD.coef_records(out, False, vr, held_fx.records)
        if ok:
            assert RD.coef_guard(env, coef)["held_started"]["attempt"] == coef.attempt["attempt"]
        else:
            with pytest.raises(RD.Refused):
                RD.coef_guard(env, coef)
    out = tmp_path / "reserve"
    out.mkdir()
    write_json(out / "held_started_reserve.json", started_record({}, attempts=[att(1, None, COEF)]))
    vr = SimpleNamespace(suffix="_reserve", reserve=True, pass_path=out / "held_pass_reserve.json")
    assert RD.coef_guard(env, RD.coef_records(out, False, vr, held_fx.records))["held_started"]["file"] == \
        "held_started_reserve.json"


# ---------------------------------------------------------------- ticket 14's per-seed hook, with the RowContext

def test_per_seed_hook_receives_the_context_and_adds_rows(smoke_fx, tmp_path, monkeypatch):
    """A DTS-like stub: it reads the seed's RowContext (masked img and txt, cos, pooled, parity, pair_index) and
    returns a fused per-anchor dict; an MLLM-like stub answers on seed 9001 only; a missing job gives info only. Each
    becomes an external row of descriptive.json (the runner's loop drops ctx and bundle after the hook)."""
    seen, made = [], {}

    def hook(env, seed, ctx, bundle, episodes, out, smoke):
        assert all(hasattr(ctx, f) for f in ("img", "txt", "cos", "pooled", "parity", "pair_index"))
        assert ctx.seed == seed == episodes.seed and np.array_equal(ctx.pooled.anchor, episodes.pooled.anchor)
        assert env is smoke_fx.env and smoke is True and Path(out).is_dir()
        seen.append(seed)
        pa = per_anchor(fused_scores(ctx.cos, bundle.pm_terms["wang"], 1.0))
        made[seed] = pa
        rows = {"DTS": {"pa": pa, "label": "describe-then-score (stub)", "info": {"parsing_failures": {"a": 1}}},
                "FT-LoRA": {"pa": None, "label": "FT-LoRA", "info": {"missing": "job not run"}}}
        if seed == 9001:
            rows["MLLM"] = {"pa": per_anchor(ctx.cos), "label": "reranker (stub)", "info": {}}
        return rows
    monkeypatch.setattr(RD, "external_seed_rows", hook)
    out = copy_dir(smoke_fx, tmp_path)
    assert run(smoke_fx, out)[0] == 0
    assert seen == list(R.SMOKE_SEEDS)
    rec = json.loads((out / "descriptive.json").read_text())
    cl = np.concatenate([np.asarray(s["cl"]) for s in smoke_fx.scored])
    aff = np.concatenate([s["aff_fused"]["r1"] for s in smoke_fx.scored])
    dts = np.concatenate([made[s]["r1"] for s in R.SMOKE_SEEDS])
    assert rec["rows"]["DTS"]["pooled"]["r1"] == C.point_ci(dts, cl)
    assert rec["rows"]["DTS"]["pooled"]["aff_minus"]["r1"] == C.point_ci(aff - dts, cl)
    assert rec["external"]["DTS"] == {"seeds": list(R.SMOKE_SEEDS),
                                      "per_seed": {str(s): {"parsing_failures": {"a": 1}} for s in R.SMOKE_SEEDS}}
    assert set(rec["rows"]["MLLM"]["per_seed"]) == {"9001"} and rec["external"]["MLLM"]["seeds"] == [9001]
    assert "FT-LoRA" not in rec["rows"] and rec["external"]["FT-LoRA"]["seeds"] == []
    assert rec["external"]["FT-LoRA"]["per_seed"]["9001"] == {"missing": "job not run"}
    assert rec["labels"]["DTS"] == "describe-then-score (stub)" and rec["scorers"][-2:] == ["DTS", "MLLM"]


def test_seed_ctx_bundle_is_run_r6_helds_seed_bundle():
    """The descriptive pass builds each bundle with run_r6_held.seed_bundle's own two calls (same RowContext and
    build_bundle_r6 arguments, same modules), only keeping the context and passing no episode recorder."""
    def calls(path, name):
        fn = next(n for n in ast.walk(ast.parse(Path(path).read_text()))
                  if isinstance(n, ast.FunctionDef) and n.name == name)
        return {n.func.attr: n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr in ("RowContext", "build_bundle_r6")}
    mine, theirs = calls(RD.__file__, "seed_ctx_bundle"), calls(RH.__file__, "seed_bundle")
    assert set(mine) == set(theirs) == {"RowContext", "build_bundle_r6"}
    for k in mine:
        assert ast.dump(mine[k].func) == ast.dump(theirs[k].func)
        assert [ast.dump(a) for a in mine[k].args] == [ast.dump(a) for a in theirs[k].args], k
    assert [kw.arg for kw in theirs["RowContext"].keywords] == ["on_episodes"] and not mine["RowContext"].keywords
    assert not mine["build_bundle_r6"].keywords and not theirs["build_bundle_r6"].keywords
    assert RD.X is RH.X and RD.B is RH.B
    src = Path(RD.__file__).read_text()
    assert not re.search(r"RH\.(run_|main\b|start_attempt|write_started|EpisodeRecorder)", src)   # never held mode


# ---------------------------------------------------------------- smoke messages without decimal numbers

def test_smoke_stop_messages_print_no_decimal(smoke_fx, tmp_path, capsys, monkeypatch):
    out = copy_dir(smoke_fx, tmp_path)
    with np.load(out / "held_arrays.npz") as z:
        a = {k: z[k] for k in z.files}
    a["aff_fused__r1"][0] = 0.3                              # not a quarter: r6_stats says "multiple of 0.25"
    np.savez(out / "held_arrays.npz", **a)
    code, text = run(smoke_fx, out, capsys)
    assert code == RD.EXIT_CONTRADICTION and "multiple of #" in text and DECIMAL.search(text) is None, text

    def bad(*a, **k):
        raise AssertionError("a planted check failed at 18.3 against .5")
    out2 = copy_dir(smoke_fx, tmp_path, "res2")
    monkeypatch.setattr(RD.D, "describe", bad)
    code, text = run(smoke_fx, out2, capsys)
    assert code == RD.EXIT_CONTRADICTION and "at # against #" in text and DECIMAL.search(text) is None, text
    assert RD.numberless("x 0.25 y 12 z .5") == "x # y 12 z #"


# ---------------------------------------------------------------- the development figure of item reuse

def test_development_reuse_is_seed42s_selection_episodes():
    ep = D.development_episodes()
    assert len(ep.anchor) == 3 * R.N_PER_PAIR and D.member_rows(ep).shape == (3 * R.N_PER_PAIR, 30)
    groups = np.arange(R.N_ROWS, dtype=np.int64) // 5
    rec = RD.development_reuse(groups)
    want = D.item_reuse(ep.anchor, ep.candidates, D.member_rows(ep), groups)
    assert {k: v for k, v in rec.items() if k != "split"} == want
