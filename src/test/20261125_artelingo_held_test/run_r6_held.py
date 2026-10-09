"""The round-6 held runner (DECISION_RULE.md of this folder; contracts section 7). One file, one mode per run:

  --mode regression   stage 1b, second half (rule section 6 items 4 and 5, C12; ticket 07): the held runner's own
                      scoring code in selection mode on seed 42 (4,096 episodes per pair) with the frozen picks of
                      rule section 5 item 4 reproduces every target of rule section 6 item 4 exactly
  (ticket 08 adds --mode held and --mode smoke to this file, on the same shared functions)

Layout: the order guard and the module-SHA bookkeeping, then the shared setup and scoring functions (rows, heads,
PM fits and readers, one seed's bundle, scoring), then one function per mode, then main.

Regression mode, in this order:
  1. Order guard (mandatory, round 4's lesson T3a-2; no flag skips it): results/refit_check.json exists with
     "passed": true and results/picks_seed42.json exists with "passed": true, and each records, for every r6 module
     its runner ran (the runner and the r6_*.py it imports, followed through imports), the module's current SHA-256.
     Otherwise it refuses (exit 4) before any input or data is loaded.
  2. Inputs (rule section 6 item 1) and the rule's targets against round 3's constants and dev_seed42.json.
  3. Shared setup: data, split (rule section 5 item 1), labels, development value sets, the five heads refit and
     checked bit for bit on selection rows, the PM fits, the A0 half-readers.
  4. The seed-42 selection bundle (RowContext + build_bundle_r6) and score_seed(..., include_pm=True) with B, B0, B1
     from picks_seed42.json and RCA and the nine PM lambdas from frozen_lambdas() (the cells are r6_score's).
  5. Items (each {"equal", "got", "want"}, compared exactly):
       heads        the refit posteriors on selection rows equal the stored ones; the coefficient SHA-256s equal
                    refit_check.json's and picks_seed42.json's
       episodes     ticket 02's identity check (seeds 42, 9001, 9002, 9003) and the bundle's seed-42 hashes
       numbers      AFF fused R@1, CF R@1, bar comparator, bar margin, gain statistic, AFF - R1 fused, R1's margin
                    against its counterpart (comparator asserted "counterpart"), its gain statistic, AFF - B1
                    (rule section 6 item 4); beside them R1 fused and counterpart R@1 (r3_common.RC_NUMBERS), AFF's
                    9,406 quarter-hits, B, B0, B1 mean R@1 (rule section 6 item 3)
       arrays       every aff_fused__*, aff_cf__*, r1_fused__*, r1_cf__*, aff_gate__*, r1_gate__* of round 3's
                    seed42_arrays.npz, plus its cl, pair_index, parity, bar_v, cells and the reader's pick, margin
                    and P; every cosine__*, rca__* and nine-PM key of per_anchor_seed42.npz, plus its anchor_group
                    and pair_index (shape, dtype and values exact)
  6. Passed: results/seed42_per_episode.npz (cl, pair_index, diff__P1 .. diff__S2 in R@1 fractions),
     results/seed42_pass_counts.json (r6_stats.pass_record on seed 42 alone, mode "regression": the phase-1 target of
     the re-derivation, rule section 8 item 4) and results/regression_seed42.json. Exit 0.
     Any item differs: results/regression_seed42.json with "passed": false (it replaces an earlier pass) and a
     copy, results/regression_seed42_failed.json; no per-episode or counts file is written, and those of an earlier
     pass are named by no passed record (the sensitivity runner refuses). The difference is traced and the user
     decides. Exit 3.
Every output records module_sha256 (r6_module_shas()), input_sha256, the coefficient SHA-256s and the time.
run_r6_sensitivity.py turns seed42_per_episode.npz into sensitivity_seed42.json (rule section 6 item 5).

Selection rows only: no held row is loaded beyond the split's index arrays. CPU only, 8 threads (asserted). Prints
pass or FAIL per item and the file paths, never a metric:

    cd /project/CoSiR && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python src/test/20261125_artelingo_held_test/run_r6_held.py --mode regression \
        > src/test/20261125_artelingo_held_test/results/run_r6_held_regression.log 2>&1

Guards carry a `# guard:<name>` marker; test_r6_regression.py deletes each on a copy and shows that its scenario then
goes through.
"""
import argparse
import ast
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r6_common as R  # noqa: E402  (first, before anything that imports src)
import r6_bundle as B  # noqa: E402
import r6_context as X  # noqa: E402
import r6_episodes as E  # noqa: E402
import r6_heads as H  # noqa: E402
import r6_picks as P  # noqa: E402
import r6_score as S  # noqa: E402
import r6_stats as ST  # noqa: E402

import numpy as np  # noqa: E402

from src.eval.aspect_episodes import PaintingValueIndex  # noqa: E402
from src.eval.aspect_metrics import CONDITIONS, METRICS  # noqa: E402

C, R3 = R.C, R.R3

EXIT_DIFF = 3
EXIT_REFUSE = 4
MODES = ("regression",)
REFIT_NAME = "refit_check.json"
PICKS_NAME = P.PICKS_NAME
# the order guard's files and the runner that wrote each
ORDER_FILES = {REFIT_NAME: "run_r6_refit.py", PICKS_NAME: "run_r6_picks.py"}
REGRESSION_NAME = "regression_seed42.json"
COUNTS_NAME = "seed42_pass_counts.json"
PER_EPISODE_NAME = "seed42_per_episode.npz"
R3_ARRAYS_REL = "20261121_round3_affect_gate/results/seed42_arrays.npz"
PER_ANCHOR42_REL = "20261030_aspect_baselines/results/per_anchor_seed42.npz"
DEV42_REL = "20261122_round4_aff_vetoes/results/dev_seed42.json"
WHO = ("aff", "r1")
PARTS = ("fused", "cf")
AB_SCORERS = ("cosine", "rca") + B.PM_NAMES          # the keys of per_anchor_seed42.npz the regression compares

# rule section 6 item 4 (numbers: R@1 points; [point, lower, upper] of the 95% interval)
TARGETS = {
    "aff_fused_r1": 19.136555989583336,
    "aff_cf_r1": 18.39599609375,
    "aff_bar_comparator": "B_prime",
    "aff_bar_margin": [0.6998697916666667, 0.4598852740816973, 0.9371680126852968],
    "aff_gain_statistic": [3.110758463541667, 2.780005709854805, 3.4559584315470384],
    "aff_minus_r1_fused": [0.21769205729166666, 0.06425880757348419, 0.3709597330984391],
    "r1_bar_comparator": "counterpart",
    "r1_margin_vs_counterpart": [0.4435221354166667, 0.21646171563312194, 0.6735669710776852],
    "r1_gain_statistic": [2.667236328125, 2.325087836946873, 3.012361650695922],
    "aff_minus_b1": [0.33162434895833337, 0.048231414333532084, 0.6246158772581268],
}
# beside them: round 3's R1 numbers (r3_common.RC_NUMBERS), AFF's quarter-hits, rule section 6 item 3's means
EXTRA_TARGETS = {
    "r1_fused_r1": 18.918863932291664,
    "r1_cf_r1": 18.475341796875,
    "aff_quarter_hits": R.AFF_HITS_SEED42,
    **{f"{name}_mean_r1": P.MEAN_R1_TARGETS[name] for name in P.NESTED},
}

_A, _N = R3.AFF_BRAINSTORM, R3.RC_NUMBERS
if not (TARGETS["aff_fused_r1"] == _A["fused_r1"] and TARGETS["aff_cf_r1"] == _A["cf_r1"]
        and TARGETS["aff_bar_comparator"] == _A["comparator"] and TARGETS["aff_bar_margin"] == list(_A["bar"])
        and TARGETS["aff_gain_statistic"] == list(_A["gain_statistic"])
        and TARGETS["aff_minus_r1_fused"] == list(_A["aff_minus_r1_fused"])
        and TARGETS["r1_bar_comparator"] == _N["comparator"]
        and TARGETS["r1_margin_vs_counterpart"] == list(_N["bar"])
        and TARGETS["r1_gain_statistic"] == list(_N["gain_statistic"])
        and EXTRA_TARGETS["r1_fused_r1"] == _N["fused_r1"] and EXTRA_TARGETS["r1_cf_r1"] == _N["cf_r1"]
        and R.N_RANKINGS_SEED42 == 4 * len(R.PAIRS) * R.N_PER_PAIR):
    raise ImportError("run_r6_held: the regression targets differ from round 3's recorded constants")


class Refused(Exception):
    """The run refuses to start (exit 4): a precondition of the rule is not met."""


def _require(cond, msg, exc=AssertionError):
    if not cond:
        raise exc(msg)


def _refuse_unless(cond, msg):
    if not cond:
        raise Refused(msg)


def log(msg):
    R.run_checks.log(f"[run_r6_held] {msg}")


# ---------------------------------------------------------------- module SHAs and the order guard

def _key(path, here) -> str:
    """r6_module_shas()'s key of a file of ``here``: its path relative to the checkout that holds ``here``."""
    here = Path(here).resolve()
    return (here / path).relative_to(here.parents[2]).as_posix()


def r6_imports(path) -> set:
    """Names of the r6 modules (r6_*, run_r6_*) a file imports anywhere (top level or inside a function)."""
    names = set()
    for node in ast.walk(ast.parse(Path(path).read_text())):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return {n for n in names if n.startswith(("r6_", "run_r6_"))}


def ran_modules(runner, here=None) -> tuple:
    """File names of ``runner`` and of every r6 module of ``here`` it imports, followed through imports."""
    here = Path(here or HERE)
    todo, seen = [Path(runner).name], set()
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        path = here / name
        _require(path.is_file(), f"{path} does not exist")
        seen.add(name)
        todo += [f"{m}.py" for m in sorted(r6_imports(path)) if (here / f"{m}.py").is_file()]
    return tuple(sorted(seen))


def stale_modules(record, runner, here=None) -> list:
    """Keys of the modules ``runner`` ran whose SHA-256 in ``record["module_sha256"]`` is missing or differs from the
    file's current SHA-256."""
    here = Path(here or HERE)
    recorded = record.get("module_sha256") if isinstance(record, dict) else None
    recorded = recorded if isinstance(recorded, dict) else {}
    out = []
    for name in ran_modules(runner, here):
        key = _key(name, here)
        if recorded.get(key) != R.sha256_file(here / name):
            out.append(key)
    return out


def _read_json(path):
    return json.loads(Path(path).read_text())


def order_guard(results=None, here=None) -> dict:
    """Round 4's lesson T3a-2, mandatory: the stage-1a refit check and the seed-42 picks exist, passed, and were run
    by the current bytes of every r6 module their runners ran. Raises Refused otherwise; -> {"refit": record,
    "picks": record, "sha256": {file name: SHA-256}}."""
    results, here = Path(results or R.RESULTS), Path(here or HERE)
    refit, picks = results / REFIT_NAME, results / PICKS_NAME
    _refuse_unless(refit.is_file(), f"{refit} does not exist: run run_r6_refit.py first (rule section 6 item "
                                    f"2)")  # guard:refit_missing
    rec = _read_json(refit)
    _refuse_unless(rec.get("passed") is True, f"{refit} did not pass (rule section 6 item 2)")  # guard:refit_passed
    stale = stale_modules(rec, ORDER_FILES[REFIT_NAME], here)
    _refuse_unless(not stale, f"{refit} was written by other bytes of {stale}: rerun run_r6_refit.py (rule section 6 "
                              f"item 7)")  # guard:refit_stale
    _refuse_unless(picks.is_file(), f"{picks} does not exist: run run_r6_picks.py first (rule section 6 item "
                                    f"3)")  # guard:picks_missing
    prec = _read_json(picks)
    _refuse_unless(prec.get("passed") is True, f"{picks} did not pass (rule section 6 item 3)")  # guard:picks_passed
    stale = stale_modules(prec, ORDER_FILES[PICKS_NAME], here)
    _refuse_unless(not stale, f"{picks} was written by other bytes of {stale}: rerun run_r6_picks.py (rule section 6 "
                              f"item 7)")  # guard:picks_stale
    return {"refit": rec, "picks": prec, "sha256": {p.name: R.sha256_file(p) for p in (refit, picks)}}


def runner_sha256() -> str:
    return R.sha256_file(Path(__file__).resolve())


# ---------------------------------------------------------------- shared setup

def load_rows() -> SimpleNamespace:
    """ArtELingo, the split (rule section 5 item 1, every assertion), the aspect labels, the development value sets
    (rule section 5 item 2) and the painting-value index over all rows."""
    from src.data.artelingo import load_artelingo
    from src.data.artelingo_splits import artelingo_aspect_labels
    data = load_artelingo()
    split = R.load_split(data)
    labels = artelingo_aspect_labels(data)
    value_sets = R.development_value_sets(labels, split.groups, split.selection)
    index = PaintingValueIndex(labels, split.groups)
    return SimpleNamespace(data=data, split=split, labels=labels, value_sets=value_sets, index=index)


def fit_heads_checked(data, split) -> SimpleNamespace:
    """The five heads of rule section 6 item 2 refit on scorer-train rows; their selection-row posteriors checked bit
    for bit against the stored arrays (check_selection) and their coefficient SHA-256s (rule section 5 item 5)."""
    heads = H.fit_heads_r6(data, split.groups, split.scorer_train)
    return SimpleNamespace(heads=heads, check=H.check_selection(heads, data, split.selection),
                           coef_sha256=H.coef_sha256(heads))


def setup() -> SimpleNamespace:
    """Everything a seed's bundle needs, fitted once: inputs asserted, rows, heads (checked), PM fits, readers."""
    H.require_threads()
    t0 = time.time()
    inputs = R.assert_inputs()
    rows = load_rows()
    log(f"data and split [{time.time() - t0:.0f}s]")
    hd = fit_heads_checked(rows.data, rows.split)
    log(f"heads [{time.time() - t0:.0f}s]")
    pm = B.fit_pm(rows.data, rows.split.scorer_train)
    readers = B.load_readers()
    log(f"setup done [{time.time() - t0:.0f}s]")
    return SimpleNamespace(inputs=inputs, **vars(rows), heads=hd.heads, head_check=hd.check,
                           coef_sha256=hd.coef_sha256, pm=pm, readers=readers)


def seed_bundle(env, mode, seed, n_per_pair, on_episodes=None):
    """One seed's bundle: RowContext (episodes first, then cosine and posteriors) and build_bundle_r6."""
    ctx = X.RowContext(mode, seed, env.data, env.split, env.labels, env.heads, env.value_sets, n_per_pair,
                       on_episodes=on_episodes)
    return B.build_bundle_r6(ctx, env.readers, env.pm)


def score_bundle(env, bundle, picks, lambdas, *, include_pm) -> dict:
    """r6_score.score_seed with the frozen picks (contracts section 5): include_pm=True for the regression and the
    descriptive pass, False for the held pass before the verdict (rule section 8 item 3)."""
    return S.score_seed(bundle, picks, lambdas, env.readers, include_pm=include_pm)


# ---------------------------------------------------------------- regression items

def _norm(x):
    if isinstance(x, (list, tuple)):
        return [_norm(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _norm(v) for k, v in x.items()}
    if isinstance(x, np.generic):
        return x.item()
    return x


def item(want, got) -> dict:
    """An exact comparison of plain values (floats compared with ==)."""
    return {"equal": bool(_norm(want) == _norm(got)), "got": _norm(got), "want": _norm(want)}


def array_item(got, want) -> dict:
    """An exact comparison of arrays: shape, dtype and every value."""
    g, w = np.asarray(got), np.asarray(want)
    out = {"equal": bool(g.shape == w.shape and g.dtype == w.dtype and np.array_equal(g, w)),
           "got": f"array {g.shape} {g.dtype}", "want": f"array {w.shape} {w.dtype}"}
    if g.shape == w.shape and g.size:
        diff = g.astype(np.float64) != w.astype(np.float64)
        out["n_diff"] = int(diff.sum())
        out["max_abs_diff"] = float(np.max(np.abs(g.astype(np.float64) - w.astype(np.float64))))
    return out


def _ci3(r) -> list:
    return [r["point"], *r["ci95"]]


def check_targets() -> dict:
    """AFF - B1's target equals round 4's dev_seed42.json (SHA-256 asserted); -> {DEV42_REL: sha}."""
    sha = R.assert_input(DEV42_REL)
    rec = _read_json(R.INPUT_PATHS[DEV42_REL])["beside_aff"]
    _require(TARGETS["aff_minus_b1"] == _ci3(rec["AFF_minus_Bprime_A1"])
             and [EXTRA_TARGETS[f"{n}_mean_r1"] for n in P.NESTED]
             == [rec["B_mean_r1"], rec["Bprime_A0_mean_r1"], rec["Bprime_A1_mean_r1"]],
             "run_r6_held's targets differ from dev_seed42.json")
    return {DEV42_REL: sha}


def round3_arrays(scored, bundle, bar_v) -> dict:
    """The keys of round 3's seed42_arrays.npz this run reproduces, built as round 3's run_r3_seed42._arrays built
    them (same casts): cl, pair_index, parity, per-anchor metrics, bar_v, gates, cells, reader pick, margin, P."""
    arr = {"cl": np.asarray(bundle.cl), "pair_index": np.asarray(bundle.pair_index),
           "parity": np.asarray(bundle.parity)}
    for who in WHO:
        for part in PARTS:
            for m in METRICS:
                arr[f"{who}_{part}__{m}"] = np.asarray(scored[f"{who}_{part}"][m], np.float64)
        arr[f"{who}_bar_v"] = np.asarray(bar_v[who], np.float64)
        for c in CONDITIONS:
            arr[f"{who}_gate__{c}"] = scored["gates"][who][c]
        for part in PARTS:
            arr[f"{who}_{part}_cells"] = np.array(S.CELLS[who][part], np.int64)
    rd = scored["reader"]
    for c in CONDITIONS:
        arr[f"pick__{c}"] = np.asarray(rd["pick"][c], np.int64)
        arr[f"margin__{c}"] = np.asarray(rd["m"][c], np.float64)
        arr[f"P__{c}"] = np.asarray(rd["P"][c], np.float64)
    return arr


def number_items(scored, cl, pair_index) -> tuple:
    """-> (items, bar_v): rule section 6 item 4's numbers (C.bar_info, C.diff3, C.point_ci, as round 3's items 2 and
    3 and round 4's dev_seed42.json computed them) and the extra numbers beside them."""
    aff, aff_cf, r1, r1_cf = (scored[k] for k in ("aff_fused", "aff_cf", "r1_fused", "r1_cf"))
    pB, pB0, pB1 = scored["B"], scored["B0"], scored["B1"]
    f64 = (lambda x: np.asarray(x, np.float64))
    bar_v, items = {}, {}
    bar_v["aff"], bar = C.bar_info(aff, aff_cf, pB0, pB, cl, pair_index)
    bar_v["r1"], bar_r1 = C.bar_info(r1, r1_cf, pB0, pB, cl, pair_index)
    got = {
        "aff_fused_r1": 100 * float(np.mean(aff["r1"])),
        "aff_cf_r1": 100 * float(np.mean(aff_cf["r1"])),
        "aff_bar_comparator": bar["comparator"],
        "aff_bar_margin": _ci3(bar["r1"]),
        "aff_gain_statistic": _ci3(C.diff3(aff, aff_cf, cl)["gain"]),
        "aff_minus_r1_fused": _ci3(C.point_ci(f64(aff["r1"]) - f64(r1["r1"]), cl)),
        "r1_bar_comparator": bar_r1["comparator"],
        "r1_margin_vs_counterpart": _ci3(bar_r1["r1"]),
        "r1_gain_statistic": _ci3(C.diff3(r1, r1_cf, cl)["gain"]),
        "aff_minus_b1": _ci3(C.point_ci(f64(aff["r1"]) - f64(pB1["r1"]), cl)),
        "r1_fused_r1": 100 * float(np.mean(r1["r1"])),
        "r1_cf_r1": 100 * float(np.mean(r1_cf["r1"])),
        "aff_quarter_hits": quarter_hits(aff["r1"]),
        **{f"{name}_mean_r1": 100 * float(np.mean(scored[name]["r1"])) for name in P.NESTED},
    }
    want = {**TARGETS, **EXTRA_TARGETS}
    _require(tuple(got) == tuple(want), "the regression's numbers are out of the targets' order")
    for k in want:
        items[k] = item(want[k], got[k])
    return items, bar_v


def quarter_hits(r1) -> int:
    """Sum of round 2's as_int4 (int8 rint(4 x), multiple of 0.25 asserted) in int64 (hazard 5)."""
    return int(np.sum(R.F2.as_int4(np.asarray(r1, np.float64), "AFF r1"), dtype=np.int64))


def regression_items(env, bundle, scored, orders) -> tuple:
    """-> (items, bar_v): every comparison of the regression, in a fixed order (heads, episodes, numbers, round 3's
    arrays, AB's per-anchor arrays)."""
    items = {}
    items["heads_selection_bits_equal_stored"] = item(True, bool(env.head_check["passed"]))
    items["coef_sha256_equal_refit_check"] = item(orders["refit"].get("coef_sha256"), env.coef_sha256)
    items["coef_sha256_equal_picks_seed42"] = item(orders["picks"].get("coef_sha256"), env.coef_sha256)
    items["bundle_coef_sha256_equal_heads"] = item(env.coef_sha256, bundle.coef_sha256)

    ident = E.identity_check(env.labels, env.split.groups, env.split.selection, env.index, env.value_sets)
    items.update(ident["items"])                                       # episodes_seed<s>__<pair>
    want42 = E.identity_targets()[R.DEV_SEED]["episodes_sha256"]
    for p in R.PAIR_NAMES:
        items[f"bundle_episodes_seed42__{p}"] = item(want42[p], bundle.episodes_sha256[p])
    log("episode identity checked")

    cl, pair_index = np.asarray(bundle.cl), np.asarray(bundle.pair_index)
    nums, bar_v = number_items(scored, cl, pair_index)
    items.update(nums)
    log("numbers compared")

    R.assert_input(R3_ARRAYS_REL)
    mine = round3_arrays(scored, bundle, bar_v)
    with np.load(R.INPUT_PATHS[R3_ARRAYS_REL]) as z:
        _require(set(mine) <= set(z.files), f"seed42_arrays.npz lacks {sorted(set(mine) - set(z.files))}")
        for k in mine:
            items[f"seed42_arrays/{k}"] = array_item(mine[k], z[k])
    R.assert_input(PER_ANCHOR42_REL)
    with np.load(R.INPUT_PATHS[PER_ANCHOR42_REL]) as z:
        items["per_anchor_seed42/anchor_group"] = array_item(cl, z["anchor_group"])
        items["per_anchor_seed42/pair_index"] = array_item(pair_index, z["pair_index"])
        for s in AB_SCORERS:
            for m in METRICS:
                items[f"per_anchor_seed42/{s}__{m}"] = array_item(np.asarray(scored[s][m]), z[f"{s}__{m}"])
    return items, bar_v


# ---------------------------------------------------------------- seed-42 outputs

def per_episode_arrays(scored) -> dict:
    """{cl, pair_index, diff__P1 .. diff__S2}: the per-episode differences of rule section 3 and 4's checks on seed
    42, in R@1 fractions (r6_stats.check_diffs, the same differences pass_record bootstraps)."""
    diffs = ST.check_diffs([scored])
    out = {"cl": np.asarray(scored["cl"]), "pair_index": np.asarray(scored["pair_index"])}
    for c in ST.CHECKS + ST.SECONDARY:
        d, cl = diffs[c]
        _require(np.array_equal(cl, out["cl"]), f"{c}: clusters differ from the seed's cl")
        out[f"diff__{c}"] = np.asarray(d, np.float64)
    return out


def _meta(env, extra=None) -> dict:
    return {"rule_sha256": R.RULE_SHA256, "runner_sha256": runner_sha256(), "module_sha256": R.r6_module_shas(),
            "input_sha256": dict(env.inputs), "coef_sha256": env.coef_sha256, "time": R.amsterdam_now(),
            **(extra or {})}


def failed_path(out) -> Path:
    out = Path(out)
    return out.with_name(f"{out.stem}_failed{out.suffix}")


def write_json(path, rec):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=1))


# ---------------------------------------------------------------- modes

def run_regression(results=None, here=None, env=None, bundle=None) -> int:
    """Regression mode (module docstring). ``env`` and ``bundle`` (setup() and the seed-42 selection bundle) may be
    given by a test; the order guard runs first in every case. -> exit code (0 passed, 3 an item differs, 4
    refused)."""
    t0 = time.time()
    results = Path(results or R.RESULTS)
    try:
        orders = order_guard(results, here)
    except Refused as e:
        print(f"refused: {e}")
        return EXIT_REFUSE
    picks = P.load_picks(results / PICKS_NAME)                         # passed, means equal the rule's targets
    lambdas = P.frozen_lambdas()
    target_inputs = check_targets()
    env = env or setup()
    bundle = bundle or seed_bundle(env, "selection", R.DEV_SEED, R.N_PER_PAIR)
    P.check_seed42(bundle)
    scored = score_bundle(env, bundle, picks, lambdas, include_pm=True)
    log("scored")
    items, _ = regression_items(env, bundle, scored, orders)
    passed = all(it["equal"] for it in items.values())
    rec = {"passed": passed, "n_items": len(items), "n_failed": sum(not it["equal"] for it in items.values()),
           "items": items, "mode": "regression", "seed": R.DEV_SEED, "n_per_pair": R.N_PER_PAIR,
           "episodes_sha256": dict(bundle.episodes_sha256),
           "picks": {name: {str(h): list(picks[name][h]) for h in (0, 1)} for name in P.NESTED},
           "lambdas": {name: {str(h): ("inf" if np.isinf(v) else v) for h, v in lam.items()}
                       for name, lam in lambdas.items()},
           "cells": {who: {part: list(S.CELLS[who][part]) for part in PARTS} for who in WHO},
           "order_inputs": orders["sha256"], "fit_rows_sha256": bundle.fit_rows_sha256}
    rec.update(_meta(env, {"input_sha256": {**env.inputs, **target_inputs}}))
    code = 0
    if not passed:
        code = EXIT_DIFF  # guard:exit_code
    if code == 0:
        arrays = per_episode_arrays(scored)
        meta = _meta(env, {"seed": R.DEV_SEED, "units": "R@1 fractions per episode",
                           "checks": {c: ST.QUANTITIES[c][2] for c in ST.CHECKS + ST.SECONDARY}})
        npz = results / PER_EPISODE_NAME
        npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(npz, **arrays, meta=np.array(json.dumps(meta)))
        extra = {"episodes_sha256": {R.DEV_SEED: dict(bundle.episodes_sha256)}, "runner_sha256": runner_sha256(),
                 "module_sha256": R.r6_module_shas()}
        counts = ST.pass_record([scored], "regression", [R.DEV_SEED], extra)
        counts.update({"input_sha256": rec["input_sha256"], "coef_sha256": env.coef_sha256})
        write_json(results / COUNTS_NAME, counts)
        rec["outputs"] = {p.name: R.sha256_file(p) for p in (npz, results / COUNTS_NAME)}
    rec["runtime_s"] = round(time.time() - t0, 1)
    path = results / REGRESSION_NAME
    write_json(path, rec)                       # passed or not: an earlier pass never survives a failed run
    if code:
        write_json(failed_path(path), rec)      # and a kept copy of the failed record
    for name, it in items.items():
        print(f"{name}: {'pass' if it['equal'] else 'FAIL'}")
    print(f"regression seed 42: {'PASSED' if passed else 'FAILED'} ({rec['n_items']} items, {rec['n_failed']} "
          f"failed); record {path}")
    if code == 0:
        print(f"written {results / PER_EPISODE_NAME} and {results / COUNTS_NAME}")
    else:
        print(f"stage-1b stop (rule section 6 item 4): exit {code}")
    return code


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mode", required=True, choices=MODES)
    args = ap.parse_args(argv)
    t0 = time.time()
    code = {"regression": run_regression}[args.mode]()
    print(f"runtime {time.time() - t0:.0f} s")
    return code


if __name__ == "__main__":
    sys.exit(main())
