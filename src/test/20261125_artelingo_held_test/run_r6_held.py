"""The round-6 held runner (DECISION_RULE.md of this folder; contracts section 7). One file, one mode per run:

  --mode regression   stage 1b, second half (rule section 6 items 4 and 5, C12; ticket 07): the held runner's own
                      scoring code in selection mode on seed 42 (4,096 episodes per pair) with the frozen picks of
                      rule section 5 item 4 reproduces every target of rule section 6 item 4 exactly
  --mode held [--after-crash | --fix 1 | --reserve]
                      the read (rule section 5 item 5, section 8 items 1 to 3, section 9; ticket 08): held seeds 52,
                      53, 54, 4,096 episodes per pair; writes held_pass.json (or _fix1, _reserve), never a verdict
  --mode smoke [--smoke-subdir NAME]
                      the same read code (steps 3 to 8 below, the same functions) in selection mode on the smoke
                      seeds 9001 to 9003, 64 episodes per pair, under results/smoke/ (or results/smoke/NAME/ for a
                      repeated smoke); no ledger, no refusal on the real read's files; prints no metric

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

Held mode, in this order (rule section 8 item 1 to the letter; nothing is loaded before step 3, and nothing is written
before step 5):
  1. Refusal (exit 4, one line): held_verdict.json exists; or held_pass.json exists and --fix 1 is not given; or
     held_started.json exists and neither --after-crash nor --fix 1 is given; or --after-crash is given and
     held_started.json already records two attempts; or ledger row H5 of docs/superpowers/held_ledger.md (under the
     main checkout) is missing, its latest script SHA-256 (the last 64-hex string of its script cell) is not this
     file's, or its report cell is not exactly "(pending)". --fix 1 also refuses if held_pass_fix1.json exists or
     smoke_record_fix1.json is missing. --reserve reads row H5-R and the _reserve names (it refuses if
     held_started_reserve.json or held_verdict_reserve.json exists). Agent defaults, stricter than the rule: a
     --after-crash without a recorded attempt, a --fix 1 without held_pass.json, a second --fix 1 attempt, a --fix 1
     whose row H5 does not hold the nine episode SHA-256s of the first read, and a --reserve without
     held_verdict.json (or without the nine episode SHA-256s in held_started.json) refuse too.
  2. r6_module_shas() (this runner, every r6_ module, the DTS settings and the r6 scripts) equals the module_sha256 of
     the latest smoke record (results/smoke_record.json; smoke_record_fix1.json for --fix 1 or when it exists;
     smoke_record_reserve.json for --reserve), contracts section 8 amendment 12:20.
  3. Inputs, rerun on every attempt: assert_inputs(), recorded_hashes() (the non-smoke episode hashes, read now and
     handed to assert_distinct later), and the seed-42 records: refit_check.json and picks_seed42.json passed and
     current (the order guard), regression_seed42.json passed and current, sensitivity_seed42.json current (its
     regression_sha256 the current regression file's) with every check's sigma parts (agent default: rule section 6
     item 7 asks these items to have run on the current bytes).
  4. setup(): the heads refit on scorer-train rows, their selection posteriors rechecked bit for bit; refuses unless
     the check passed and every coefficient SHA-256 equals refit_check.json's (rule section 5 item 5).
  5. First write: the attempt (time, SHA-256s, flags, coefficient SHA-256s) appended to held_started.json and its
     copy in this folder. Only now are held features, held posteriors and held episodes touched.
  6. Per seed (52, 53, 54 in order), from the context's on_episodes, before cosine or any posterior: the per-pair
     episode SHA-256s appended to this attempt at once; on --after-crash, --fix 1 or --reserve compared with the
     hashes recorded before (refusal on a difference); assert_distinct over every held seed so far against the
     recorded hashes; held_episodes_seed<s>.npz saved (or, when an earlier attempt saved it, compared).
  7. assert_held_eligible; sensitivity_held.json (rule section 8 item 2: M_p from the pooled held anchors, sigma from
     sensitivity_seed42.json) before any held score. A --fix 1 pass reuses (compares) it.
  8. score_seed(..., include_pm=False) per seed: only AFF fused, CF, COS, RCA, B, B0, B1, R1 fused (rule section 8
     item 3, asserted); held_arrays.npz (or _fix1, _reserve); r6_stats.pass_record pooled over the three seeds ->
     held_pass.json (or _fix1, _reserve). No verdict, no per-seed or per-pair number. Prints "held pass written" and
     the file's SHA-256.
A results file of the read is never overwritten (rule section 9): a file an earlier attempt wrote (episodes,
sensitivity, arrays) is compared instead, and a difference stops the run. Smoke mode overwrites its own files and
refuses only if its folder holds a smoke verdict (a repeated smoke goes to --smoke-subdir).

    cd /project/CoSiR && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python src/test/20261125_artelingo_held_test/run_r6_held.py --mode smoke \
        > src/test/20261125_artelingo_held_test/results/run_r6_held_smoke.log 2>&1

Guards carry a `# guard:<name>` marker; test_r6_regression.py and test_r6_held.py delete each on a copy and show that
its scenario then goes through.
"""
import argparse
import ast
import io
import json
import math
import os
import re
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
MODES = ("regression", "held", "smoke")
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


# ---------------------------------------------------------------- the read: names, ledger, refusal (held mode, step 1)

LEDGER = R.MAIN / "docs/superpowers/held_ledger.md"
SENS42_NAME = "sensitivity_seed42.json"                 # run_r6_sensitivity.OUT_NAME (asserted in the tests)
SMOKE_RECORDS = {"held": "smoke_record.json", "fix1": "smoke_record_fix1.json",
                 "reserve": "smoke_record_reserve.json"}
LEDGER_ROWS = {"held": "H5", "fix1": "H5", "reserve": "H5-R"}
LEDGER_CELLS = ("#", "Date", "Dataset and split", "Purpose", "Episode / data SHA-256", "Script SHA-256", "Report")
EPISODE_CELL, SCRIPT_CELL, REPORT_CELL = 4, 5, 6
PENDING = "(pending)"
MAX_ATTEMPTS = 2                                        # the read and one --after-crash rerun (rule section 8 item 1)
HEX64 = re.compile(r"(?<![0-9a-fA-F])[0-9a-f]{64}(?![0-9a-fA-F])")
SUBDIR_RE = re.compile(r"[A-Za-z0-9_]+")                # run_r6_apply_rule's --smoke-subdir names
SMOKE_COPY_DIR = "folder"                               # the smoke's stand-in for this folder's committed copy
READ_SCORERS = S.CORE_SCORERS                           # rule section 8 item 3: AFF, CF, COS, RCA, B, B0, B1, R1 fused
READ_KEYS = S.CORE_SCORERS + S.EXTRA_KEYS
SIGMA_KEYS = ("sigma_a2", "sigma_eps2")


def read_names(kind) -> SimpleNamespace:
    """File names and ledger row of one kind of read: "held" (the read and its --after-crash rerun), "fix1"
    (--fix 1: its own pass and arrays, the read's started, episode and sensitivity files), "reserve" (--reserve: the
    _reserve names, the read's episode files) and "smoke"."""
    sfx = "_reserve" if kind == "reserve" else ""
    pfx = {"fix1": "_fix1", "reserve": "_reserve"}.get(kind, "")
    return SimpleNamespace(kind=kind, started=f"held_started{sfx}.json", verdict=f"held_verdict{sfx}.json",
                           pass_=f"held_pass{pfx}.json", sensitivity=f"sensitivity_held{sfx}.json",
                           arrays=f"held_arrays{pfx}.npz", smoke_record=SMOKE_RECORDS.get(kind),
                           row=LEDGER_ROWS.get(kind))


def episodes_name(seed) -> str:
    return f"held_episodes_seed{int(seed)}.npz"


def ledger_row(path, row_id) -> dict:
    """The held ledger's table row whose first cell is ``row_id`` (contracts section 7): {"row", "cells", "episodes"
    (the 64-hex strings of the episode cell), "scripts" (those of the script cell, in order), "report"}. Refused when
    the ledger or the row is missing, the row appears twice, or it does not have the table's seven cells."""
    path = Path(path)
    _refuse_unless(path.is_file(), f"the held ledger {path} does not exist (row {row_id} is missing)")
    rows = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if s.startswith("|") and s.endswith("|") and len(s) > 1:
            cells = [c.strip() for c in s[1:-1].split("|")]
            if cells[0] == row_id:
                rows.append(cells)
    _refuse_unless(rows, f"ledger row {row_id} is missing in {path} (rule section 8 item 1)")  # guard:ledger_row
    _refuse_unless(len(rows) <= 1, f"ledger row {row_id} appears {len(rows)} times in {path}")
    cells = rows[0]
    _refuse_unless(len(cells) == len(LEDGER_CELLS), f"ledger row {row_id} has {len(cells)} cells, not the table's "
                                                    f"{len(LEDGER_CELLS)}")
    return {"row": row_id, "cells": cells, "episodes": HEX64.findall(cells[EPISODE_CELL]),
            "scripts": HEX64.findall(cells[SCRIPT_CELL]), "report": cells[REPORT_CELL]}


def ledger_guard(path, row_id, runner_sha) -> dict:
    """Rule section 8 item 1: the row exists, its latest script SHA-256 is this runner's, its report cell is exactly
    "(pending)". Refused otherwise; -> ledger_row's dict."""
    row = ledger_row(path, row_id)
    _refuse_unless(row["scripts"] and row["scripts"][-1] == runner_sha,
                   f"ledger row {row_id}: its latest script SHA-256 is not this runner's "
                   f"{runner_sha}")  # guard:ledger_sha
    _refuse_unless(row["report"] == PENDING, f"ledger row {row_id}: its report cell is {row['report']!r}, not "
                                             f"{PENDING!r}")  # guard:ledger_report
    return row


def read_attempts(path) -> list:
    """The attempts of a started file (this rule's), each with its flags and its per-seed episode hashes."""
    path = Path(path)
    try:
        rec = _read_json(path)
    except (OSError, ValueError) as e:
        raise Refused(f"{path.name} cannot be read ({type(e).__name__}): the user decides")
    rec = rec if isinstance(rec, dict) else {}
    att = rec.get("attempts")
    _refuse_unless(rec.get("rule_sha256") == R.RULE_SHA256 and isinstance(att, list) and att
                   and all(isinstance(a, dict) and isinstance(a.get("flags"), dict)
                           and isinstance(a.get("episodes_sha256"), dict) for a in att),
                   f"{path.name} is not a started file of this rule: the user decides")
    return att


def attempt_hashes(attempts) -> dict:
    """{seed: {pair: sha}} recorded over the attempts (each seed once per attempt); two attempts that disagree on a
    seed refuse (an --after-crash rerun asserted them equal)."""
    out = {}
    for a in attempts:
        for s, h in a["episodes_sha256"].items():
            h = {p: str(h[p]) for p in R.PAIR_NAMES} if isinstance(h, dict) and set(h) == set(R.PAIR_NAMES) else None
            _refuse_unless(h is not None, f"attempt {a.get('attempt')}: seed {s} lacks a pair's episode hash")
            _refuse_unless(out.setdefault(int(s), h) == h, f"the attempts disagree on seed {s}'s episode hashes")
    return out


def _complete(hashes) -> bool:
    return set(hashes) == set(R.HELD_SEEDS)


def refuse_or_go(results, ledger, kind, after_crash=False) -> SimpleNamespace:
    """Step 1 (rule section 8 item 1 and section 9): reads only the results folder's file names, the started file
    and the ledger. Raises Refused; -> SimpleNamespace(names, attempts (earlier attempts of this started file), want
    ({seed: {pair: sha}} this read must reproduce), ledger (the row))."""
    n = read_names(kind)
    res = Path(results)
    _refuse_unless(not (res / n.verdict).exists(), f"{n.verdict} exists: the read is over (rule section 8 item "
                                                   f"1)")  # guard:verdict_exists
    _refuse_unless(not (res / n.pass_).exists(),
                   f"{n.pass_} exists" + ("" if kind == "fix1" else " and --fix 1 is not given") + " (rule section 8 "
                   "item 1)")  # guard:pass_exists
    started = res / n.started
    if kind == "reserve":
        _refuse_unless(not started.exists(), f"{n.started} exists: one reserve read only (rule section "
                                             f"9)")  # guard:reserve_started
        _refuse_unless((res / "held_verdict.json").is_file(),
                       "--reserve corrects a read whose held_verdict.json exists, and it does not (rule section "
                       "9)")  # guard:reserve_needs_verdict
    elif kind == "fix1":
        _refuse_unless((res / "held_pass.json").is_file(), "--fix 1 corrects held_pass.json, which does not exist "
                                                           "(rule section 8 item 1)")  # guard:fix_needs_pass
        _refuse_unless((res / n.smoke_record).is_file(), f"{n.smoke_record} is missing: the corrected runner needs "
                                                         f"its own smoke (rule section 8 item 1)")  # guard:fix_smoke
        _refuse_unless(started.is_file(), f"--fix 1: {n.started} does not exist")
    elif started.exists():
        _refuse_unless(after_crash, f"{n.started} exists: a read has started; rerun only with --after-crash (or "
                                    f"--fix 1) (rule section 8 item 1)")  # guard:started_exists
    else:
        _refuse_unless(not after_crash, f"--after-crash: {n.started} does not exist, no attempt to "
                                        f"rerun")  # guard:after_crash_needs_start
    attempts = read_attempts(started) if started.exists() else []
    if after_crash:
        _refuse_unless(len(attempts) < MAX_ATTEMPTS, f"{n.started} already records {len(attempts)} attempts: the "
                                                     f"user decides (rule section 8 item 1)")  # guard:two_attempts
    if kind == "fix1":
        _refuse_unless(not any(a["flags"].get("fix") for a in attempts),
                       f"{n.started} already records a --fix 1 attempt: anything further goes to the user (rule "
                       f"section 8 item 1)")  # guard:fix_once
    row = ledger_guard(ledger, n.row, runner_sha256())
    if kind == "held":
        want = attempt_hashes(attempts)
    elif kind == "fix1":
        want = attempt_hashes(attempts)
        _refuse_unless(_complete(want) and set(row["episodes"]) == {h for hs in want.values() for h in hs.values()},
                       f"--fix 1: ledger row {n.row}'s episode cell does not hold exactly the nine episode SHA-256s "
                       f"of {n.started} (rule section 8 item 1)")  # guard:fix_h5_episodes
    else:
        first = res / read_names("held").started
        want = attempt_hashes(read_attempts(first)) if first.is_file() else {}
        _refuse_unless(_complete(want), f"--reserve: {first.name} does not record the nine held episode SHA-256s "
                                        f"(the same held episodes, rule section 9)")  # guard:reserve_hashes
    return SimpleNamespace(names=n, attempts=attempts, want=want, ledger=row)


# ---------------------------------------------------------------- steps 2 to 4: smoke record, inputs, heads

def latest_smoke_record(results, kind) -> Path:
    """smoke_record_fix1.json for --fix 1, smoke_record_reserve.json for --reserve; for the read and its rerun
    smoke_record.json, or smoke_record_fix1.json when it exists (contracts section 8 amendment 12:20)."""
    res = Path(results)
    if kind == "held" and (res / SMOKE_RECORDS["fix1"]).is_file():
        return res / SMOKE_RECORDS["fix1"]
    return res / SMOKE_RECORDS[kind]


def smoke_guard(results, kind, here=None) -> dict:
    """Step 2 (rule section 8 item 1, section 6 item 7): r6_module_shas() equals the latest smoke record's
    module_sha256, key for key. Refused otherwise; -> {"name", "sha256"} of the record."""
    path = latest_smoke_record(results, kind)
    _refuse_unless(path.is_file(), f"{path.name} does not exist: the smoke (rule section 6 item 7) comes "
                                   f"first")  # guard:smoke_missing
    rec = _read_json(path)
    _refuse_unless(isinstance(rec, dict) and rec.get("passed", True) is True,
                   f"{path.name} records a smoke that did not pass")  # guard:smoke_passed
    mods = rec.get("module_sha256")
    mods = mods if isinstance(mods, dict) else {}
    cur = R.r6_module_shas(here)
    diff = sorted(k for k in set(mods) | set(cur) if mods.get(k) != cur.get(k))
    _refuse_unless(mods and not diff, f"{path.name}: the SHA-256s of {diff or 'every module'} differ from the "
                                      f"smoke's: a new smoke is needed (rule section 6 item 7)")  # guard:smoke_stale
    return {"name": path.name, "sha256": R.sha256_file(path)}


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def seed42_guard(results, here=None) -> dict:
    """Step 3's seed-42 records (rule section 6 items 2 to 5 and 7; agent default beyond the order guard): the order
    guard (refit_check.json and picks_seed42.json passed and current), regression_seed42.json passed and current,
    sensitivity_seed42.json current (its regression_sha256 the regression file's, its modules current) with finite,
    non-negative sigma parts for P1 to P7, S1, S2. Raises Refused; -> order_guard's dict plus "sigma" {check:
    {"sigma_a2", "sigma_eps2"}} and the four files' SHA-256s in "sha256"."""
    results = Path(results)
    out = order_guard(results, here)
    reg_path, sens_path = results / REGRESSION_NAME, results / SENS42_NAME
    _refuse_unless(reg_path.is_file(), f"{reg_path.name} does not exist: run --mode regression first (rule section 6 "
                                       f"item 4)")  # guard:regression_missing
    reg = _read_json(reg_path)
    _refuse_unless(reg.get("passed") is True, f"{reg_path.name} did not pass (rule section 6 item "
                                              f"4)")  # guard:regression_passed
    stale = stale_modules(reg, "run_r6_held.py", here)
    _refuse_unless(not stale, f"{reg_path.name} was written by other bytes of {stale}: rerun the regression (rule "
                              f"section 6 item 7)")  # guard:regression_stale
    _refuse_unless(sens_path.is_file(), f"{sens_path.name} does not exist: run run_r6_sensitivity.py first (rule "
                                        f"section 6 item 5)")  # guard:sens42_missing
    sens = _read_json(sens_path)
    reg_sha = R.sha256_file(reg_path)
    stale = stale_modules(sens, "run_r6_sensitivity.py", here)
    _refuse_unless(sens.get("regression_sha256") == reg_sha and not stale,
                   f"{sens_path.name} was not written from the current {reg_path.name} by the current bytes "
                   f"({stale}): rerun run_r6_sensitivity.py (rule section 6 item 7)")  # guard:sens42_current
    sigma = {c: {k: (sens.get(c) or {}).get(k) for k in SIGMA_KEYS} for c in ST.CHECKS + ST.SECONDARY}
    bad = [c for c, v in sigma.items() if not all(_is_num(x) and x >= 0 for x in v.values())]
    _refuse_unless(not bad, f"{sens_path.name}: no finite non-negative sigma parts for {bad}")  # guard:sens42_sigma
    out["sigma"] = sigma
    out["sha256"] = {**out["sha256"], reg_path.name: reg_sha, sens_path.name: R.sha256_file(sens_path)}
    return out


def head_guard(env, refit) -> dict:
    """Step 4 (rule section 5 item 5): the refit heads' selection posteriors equal the stored ones bit for bit
    (check_selection passed) and every head's coefficient SHA-256 equals refit_check.json's. setup() returns both
    without asserting them; this refuses (before held_started.json). -> the coefficient SHA-256s."""
    _refuse_unless(isinstance(env.head_check, dict) and env.head_check.get("passed") is True,
                   "the refit heads' selection posteriors differ from the stored ones: no read (rule section 5 item "
                   "5, section 6 item 2)")  # guard:head_check
    want = refit.get("coef_sha256")
    _refuse_unless(isinstance(want, dict) and want and _norm(env.coef_sha256) == _norm(want),
                   "the heads' coefficient SHA-256s differ from refit_check.json's (rule section 5 item "
                   "5)")  # guard:coef_sha
    return _norm(env.coef_sha256)


# ---------------------------------------------------------------- writes of the read

def _write_bytes(path, data: bytes):
    """Atomic: a .partial file beside ``path``, then os.replace, so a crash never leaves a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _json_bytes(rec) -> bytes:
    return json.dumps(rec, indent=1).encode()


def _npz_bytes(arrays) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def _plain(x):
    """The value as JSON reads it back (keys strings, tuples lists)."""
    return json.loads(json.dumps(_norm(x)))


def write_started(path, copy_to, rec):
    """The started file and its copy (this folder's held_started.json, committed by the run chat), same bytes."""
    data = _json_bytes(rec)
    _write_bytes(path, data)
    if copy_to is not None:
        _write_bytes(copy_to, data)


def start_attempt(path, copy_to, attempts, flags, record, overwrite=False) -> int:
    """Step 5, the read's first write: the earlier attempts (re-read, asserted unchanged since step 1) plus this one
    (Amsterdam time, flags, SHA-256s, coefficient SHA-256s, an empty per-seed episode record). -> its number. The
    smoke (``overwrite``) starts a new file whatever an earlier smoke left."""
    path = Path(path)
    if not overwrite:
        now = read_attempts(path) if path.exists() else []
        _require(_plain(now) == _plain(attempts), f"{path.name} changed since the refusal check")
    k = len(attempts) + 1
    att = {"attempt": k, "time": R.amsterdam_now(), "flags": dict(flags), **record, "episodes_sha256": {}}
    write_started(path, copy_to, {"rule_sha256": R.RULE_SHA256, "attempts": [*attempts, att]})
    return k


def append_hashes(path, copy_to, attempt, seed, hashes):
    """One seed's per-pair episode SHA-256s appended to this attempt in the started file (and its copy)."""
    rec = _read_json(path)
    att = rec["attempts"][-1]
    _require(att["attempt"] == attempt and str(seed) not in att["episodes_sha256"],
             f"{Path(path).name}: attempt {attempt} is not the last, or seed {seed} is recorded twice")
    att["episodes_sha256"][str(seed)] = dict(hashes)
    write_started(path, copy_to, rec)


def write_new(path, data: bytes, overwrite=False):
    """A results file of the read: never overwritten (rule section 9), except by the smoke."""
    _require(overwrite or not Path(path).exists(), f"{path} exists: a results file is never overwritten (rule section "
                                                   f"9)")  # guard:never_overwrite
    _write_bytes(path, data)


def keep_or_write_json(path, rec, same, overwrite=False) -> str:
    """Write ``rec``; if an earlier attempt wrote the file (and this is not the smoke), its ``same`` keys must equal
    this attempt's instead (rule section 9: never overwritten). -> "written" or "kept"."""
    path = Path(path)
    if path.exists() and not overwrite:
        old = _read_json(path)
        diff = [k for k in same if old.get(k) != _plain(rec[k])]
        _require(not diff, f"{path.name}, written by an earlier attempt, differs from this attempt's in {diff}: the "
                           f"user decides")  # guard:kept_json
        return "kept"
    _write_bytes(path, _json_bytes(rec))
    return "written"


def keep_or_write_npz(path, arrays, overwrite=False) -> str:
    """As keep_or_write_json for an npz: an earlier attempt's file must hold the same keys and equal arrays (shape,
    dtype, values)."""
    path = Path(path)
    if path.exists() and not overwrite:
        with np.load(path) as z:
            same = set(z.files) == set(arrays) and all(array_item(arrays[k], z[k])["equal"] for k in arrays)
        _require(same, f"{path.name}, written by an earlier attempt, differs from this attempt's arrays: the user "
                       f"decides")  # guard:kept_npz
        return "kept"
    _write_bytes(path, _npz_bytes(arrays))
    return "written"


# ---------------------------------------------------------------- step 6: the read's bundles and episode record

def read_seed_bundle(env, mode, seed, n_per_pair, on_episodes=None):
    """seed_bundle with the row scope of the read checked: every A0 posterior of the bundle is finite on exactly the
    split's rows of ``mode`` (held, or selection) and NaN elsewhere, so r6_heads.predict ran in its own call on
    exactly that row set, never on a union (ticket 03's review)."""
    rows = np.asarray(env.split.held if mode == "held" else env.split.selection)
    in_rows = np.zeros(R.N_ROWS, dtype=bool)
    in_rows[rows] = True
    bundle = seed_bundle(env, mode, seed, n_per_pair, on_episodes=on_episodes)
    _require(bundle.mode == mode and int(bundle.seed) == int(seed), f"a bundle of {bundle.mode} seed {bundle.seed}, "
                                                                    f"not {mode} seed {seed}")
    for h, sides in bundle.post.items():
        for side, p in sides.items():
            _require(np.array_equal(np.isfinite(np.asarray(p)).all(axis=1), in_rows),
                     f"posteriors {h}/{side} are not finite on exactly split.{mode} (rule section 5 item "
                     f"5)")  # guard:post_rows
    return bundle


def held_seed_bundle(env, seed, on_episodes=None):
    """One held seed's bundle (rule section 5 item 5): RowContext in held mode on exactly split.held, 4,096 episodes
    per pair, then build_bundle_r6, the posteriors' row scope checked. ``env`` is setup()'s namespace. The read calls
    it with the episode recorder; ticket 13's descriptive pass calls it after the verdict and scores the bundle with
    score_bundle(..., include_pm=True)."""
    _require(int(seed) in R.HELD_SEEDS, f"seed {seed} is not a held seed {R.HELD_SEEDS}")
    return read_seed_bundle(env, "held", int(seed), R.N_PER_PAIR, on_episodes)


class EpisodeRecorder:
    """RowContext's on_episodes for one read (rule section 8 item 1; ticket 08 step 6). Per seed, right after its
    episodes exist and before cosine or any posterior, in this order: the per-pair SHA-256s appended to this attempt
    in the started file; compared with the hashes recorded before (Refused on a difference); assert_distinct over
    every seed of the read so far against the non-smoke hashes read in step 3; the episode file saved, or compared
    when an earlier attempt saved it."""

    def __init__(self, started, copy_to, attempt, want, recorded, out, overwrite=False):
        self.started, self.copy_to, self.attempt = Path(started), copy_to, attempt
        self.want, self.recorded, self.out, self.overwrite = dict(want), recorded, Path(out), overwrite
        self.hashes, self.expected = {}, None

    def expect(self, seed):
        self.expected = int(seed)

    def __call__(self, eps):
        s = int(eps.seed)
        _require(s == self.expected and s not in self.hashes, f"episodes of seed {s}, expected seed {self.expected}")
        h = {p: str(eps.sha[p]) for p in R.PAIR_NAMES}
        append_hashes(self.started, self.copy_to, self.attempt, s, h)
        if s in self.want:
            _refuse_unless(h == self.want[s], f"seed {s}: the episode SHA-256s differ from those recorded before "
                                              f"(rule section 8 item 1): the user decides")  # guard:rerun_hashes
        self.hashes[s] = h
        E.assert_distinct(dict(self.hashes), recorded=self.recorded)
        path = self.out / episodes_name(s)
        if path.exists() and not self.overwrite:
            got = E.load_episodes(path)
            _require(got.seed == s and got.sha == h, f"{path.name}, saved by an earlier attempt, holds other "
                                                     f"episodes: the user decides")  # guard:episode_file
        else:
            E.save_episodes(path, eps)


# ---------------------------------------------------------------- steps 7 and 8: sensitivity, scores, the pass

def sensitivity_held(bundles, seeds, sigma, n_per_pair) -> dict:
    """Rule section 8 item 2 (contracts section 7 amendment 05:45): per check SE, x (P) or x2 (S) and x95 from
    sigma (sensitivity_seed42.json) and M_p, the pooled episode count of anchor painting p over the seeds (the
    bundles' cl), N the pooled episode count (36,864 on held seeds). R@1 points."""
    cl = np.concatenate([np.asarray(b.cl) for b in bundles])
    n = len(seeds) * len(R.PAIRS) * int(n_per_pair)
    _require(len(bundles) == len(seeds) and cl.shape == (n,), f"{cl.shape} pooled anchors, not ({n},)")
    _, m_p = np.unique(cl, return_counts=True)
    rec = {}
    for c in ST.CHECKS + ST.SECONDARY:
        sig = {k: float(sigma[c][k]) for k in SIGMA_KEYS}
        rec[c] = {"quantity": ST.QUANTITIES[c][2], **sig,
                  **ST.detectable(sig, m_p, n, "P" if c in ST.CHECKS else "S")}
    rec.update({"N": n, "n_paintings": int(m_p.size), "seeds": [int(s) for s in seeds]})
    return rec


def read_arrays(scored) -> dict:
    """held_arrays.npz (contracts section 7): <scorer>__<metric> for the eight scorers of rule section 8 item 3, the
    seeds concatenated in the read's order, with cl, pair_index and seed_index (the seed's position in the read)."""
    out = {f"{name}__{m}": np.concatenate([np.asarray(sc[name][m]) for sc in scored])
           for name in READ_SCORERS for m in METRICS}
    out["cl"] = np.concatenate([np.asarray(sc["cl"]) for sc in scored])
    out["pair_index"] = np.concatenate([np.asarray(sc["pair_index"]) for sc in scored])
    out["seed_index"] = np.concatenate([np.full(len(sc["cl"]), i, dtype=np.int64) for i, sc in enumerate(scored)])
    return out


def run_read(spec, results, here=None, env=None) -> int:
    """Steps 3 to 8 of the module docstring, shared by held mode and smoke mode (``spec``: mode, names, ctx_mode,
    seeds, n_per_pair, out, copy_to, overwrite, flags, attempts, want, record). Raises Refused (exit 4)."""
    t0 = time.time()
    results, out, n = Path(results), Path(spec.out), spec.names
    # step 3: inputs, on every attempt
    inputs = R.assert_inputs()
    recorded = E.recorded_hashes()
    s42 = seed42_guard(results, here)
    picks = P.load_picks(results / PICKS_NAME)
    lambdas = P.frozen_lambdas()
    log("inputs and seed-42 records checked")
    # step 4: heads refit and checked (rule section 5 item 5)
    env = env or setup()
    coef = head_guard(env, s42["refit"])
    log("heads checked")
    # step 5: the first write
    started = out / n.started
    record = {"mode": spec.mode, "runner_sha256": runner_sha256(), "module_sha256": R.r6_module_shas(here),
              "input_sha256": dict(inputs), "seed42_records_sha256": s42["sha256"], "coef_sha256": coef,
              **spec.record}
    k = start_attempt(started, spec.copy_to, spec.attempts, spec.flags, record, spec.overwrite)
    log(f"attempt {k} recorded in {started}")
    # step 6: per seed, episodes first (recorded, compared, distinct, saved), then the bundle
    rec = EpisodeRecorder(started, spec.copy_to, k, spec.want, recorded, out, spec.overwrite)
    bundles = []
    for s in spec.seeds:
        rec.expect(s)
        if spec.ctx_mode == "held":
            b = held_seed_bundle(env, s, on_episodes=rec)
        else:
            b = read_seed_bundle(env, spec.ctx_mode, s, spec.n_per_pair, on_episodes=rec)
        _require(s in rec.hashes and dict(b.episodes_sha256) == rec.hashes[s],
                 f"seed {s}: the bundle's episodes were not reported to the recorder")  # guard:episodes_reported
        bundles.append(b)
        log(f"seed {s}: bundle built")
    # step 7: eligibility, then the sensitivity before any score
    rows = env.split.held if spec.ctx_mode == "held" else env.split.selection
    R.assert_held_eligible(env.labels, env.split.groups, rows, env.value_sets)
    sens = sensitivity_held(bundles, spec.seeds, s42["sigma"], spec.n_per_pair)
    same = list(sens)
    sens.update({"episodes_sha256": {str(s): rec.hashes[s] for s in spec.seeds},
                 "sigma_source": {"file": SENS42_NAME, "sha256": s42["sha256"][SENS42_NAME]},
                 "module_sha256": R.r6_module_shas(here), "time": R.amsterdam_now()})
    keep_or_write_json(out / n.sensitivity, sens, same + ["episodes_sha256"], spec.overwrite)
    log(f"{n.sensitivity} done")
    # step 8: the scores of rule section 8 item 3 only, the arrays, the pooled pass record
    scored = []
    for b in bundles:
        sc = score_bundle(env, b, picks, lambdas, include_pm=False)
        _require(tuple(sc) == READ_KEYS, f"scores of {tuple(sc)}: only {READ_KEYS} before the verdict (rule section 8 "
                                         f"item 3)")  # guard:allowed_keys
        scored.append(sc)
    log("scored")
    arrays = read_arrays(scored)
    keep_or_write_npz(out / n.arrays, arrays, spec.overwrite)
    extra = {"episodes_sha256": {s: rec.hashes[s] for s in spec.seeds}, "runner_sha256": runner_sha256(),
             "module_sha256": R.r6_module_shas(here)}
    pr = ST.pass_record(scored, spec.mode, list(spec.seeds), extra)
    pr.update({"kind": n.kind, "attempt": k, "input_sha256": dict(inputs), "coef_sha256": coef,
               "seed42_records_sha256": s42["sha256"],
               "outputs": {p.name: R.sha256_file(p) for p in (out / n.arrays, out / n.sensitivity)},
               "runtime_s": round(time.time() - t0)})
    path = out / n.pass_
    write_new(path, _json_bytes(pr), spec.overwrite)
    print(f"{'smoke ' if spec.mode == 'smoke' else ''}held pass written {R.sha256_file(path)}")
    return 0


# ---------------------------------------------------------------- modes: held and smoke

def run_held(after_crash=False, fix=None, reserve=False, results=None, ledger=None, here=None, folder=None,
             env=None) -> int:
    """Held mode (module docstring). ``results``, ``ledger``, ``here`` (the folder whose module SHA-256s count),
    ``folder`` (where the started file's copy goes) and ``env`` (setup()'s namespace) may be given by a test.
    -> 0, or 4 refused; a failed assertion raises."""
    results = Path(results or R.RESULTS)
    try:
        _refuse_unless(fix in (None, 1) and sum((bool(after_crash), fix is not None, bool(reserve))) <= 1,
                       "at most one of --after-crash, --fix 1, --reserve")
        kind = "reserve" if reserve else ("fix1" if fix else "held")
        go = refuse_or_go(results, ledger or LEDGER, kind, after_crash)
        smoke = smoke_guard(results, kind, here)
        spec = SimpleNamespace(
            mode="held", names=go.names, ctx_mode="held", seeds=R.HELD_SEEDS, n_per_pair=R.N_PER_PAIR, out=results,
            copy_to=Path(folder or HERE) / go.names.started, overwrite=False,
            flags={"after_crash": bool(after_crash), "fix": fix, "reserve": bool(reserve), "smoke": False},
            attempts=go.attempts, want=go.want,
            record={"smoke_record": smoke, "ledger": {"row": go.ledger["row"],
                                                      "script_sha256": go.ledger["scripts"][-1]}})
        return run_read(spec, results, here, env)
    except Refused as e:
        print(f"refused: {e}")
        return EXIT_REFUSE


def run_smoke(results=None, subdir=None, here=None, env=None) -> int:
    """Smoke mode (rule section 6 item 7): steps 3 to 8 of the read with the same functions, selection rows, seeds
    9001 to 9003, 64 episodes per pair, under results/smoke/ (or results/smoke/<subdir>/); no ledger, no refusal on
    the real read's files; its files are overwritten by the next smoke in the same folder, which refuses once the
    folder holds a smoke verdict. Prints no metric. -> 0, or 4 refused."""
    results = Path(results or R.RESULTS)
    try:
        _refuse_unless(subdir is None or SUBDIR_RE.fullmatch(str(subdir)),
                       "--smoke-subdir must be a plain folder name (letters, digits, underscore)")
        out = results / "smoke" / (subdir or "")
        _refuse_unless(not (out / "held_verdict.json").exists(),
                       f"{out / 'held_verdict.json'} exists: a repeated smoke goes to its own "
                       f"--smoke-subdir")  # guard:smoke_verdict
        names = read_names("smoke")
        spec = SimpleNamespace(
            mode="smoke", names=names, ctx_mode="selection", seeds=R.SMOKE_SEEDS, n_per_pair=R.N_SMOKE, out=out,
            copy_to=out / SMOKE_COPY_DIR / names.started, overwrite=True,
            flags={"after_crash": False, "fix": None, "reserve": False, "smoke": True},
            attempts=[], want={}, record={"smoke_subdir": subdir})
        return run_read(spec, results, here, env)
    except Refused as e:
        print(f"refused: {e}")
        return EXIT_REFUSE


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mode", required=True, choices=MODES)
    once = ap.add_mutually_exclusive_group()
    once.add_argument("--after-crash", action="store_true", help="held: the one rerun after a crash (rule 8.1)")
    once.add_argument("--fix", type=int, choices=(1,), help="held: the one corrected pass (rule 8.1)")
    once.add_argument("--reserve", action="store_true", help="held: the reserve read, ledger row H5-R (rule 9)")
    ap.add_argument("--smoke-subdir", default=None, help="smoke: results/smoke/<name>/ for a repeated smoke")
    args = ap.parse_args(argv)
    if args.mode != "held" and (args.after_crash or args.fix is not None or args.reserve):
        ap.error("--after-crash, --fix and --reserve belong to --mode held")
    if args.mode != "smoke" and args.smoke_subdir is not None:
        ap.error("--smoke-subdir belongs to --mode smoke")
    t0 = time.time()
    if args.mode == "regression":
        code = run_regression()
    elif args.mode == "held":
        code = run_held(after_crash=args.after_crash, fix=args.fix, reserve=args.reserve)
    else:
        code = run_smoke(subdir=args.smoke_subdir)
    print(f"runtime {time.time() - t0:.0f} s")
    return code


if __name__ == "__main__":
    sys.exit(main())
