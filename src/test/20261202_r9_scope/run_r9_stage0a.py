"""The stage 0a regression runner of round 9 (DECISION_RULE.md of this folder: section 3.1, section 10 row 1; ticket 06).

CLIP ViT-B/32 only. Round 9's copied modules (r9_*.py) score seed 42 (selection rows, 4,096 episodes per pair) with the
frozen picks of round 6 and must reproduce round 6's two reference files exactly: `regression_seed42.json` (131
items: 35 numbers and hashes, 39 arrays of round 3's seed42_arrays.npz, 57 arrays of AB's per_anchor_seed42.npz) and
`seed42_pass_counts.json` (P1 to P7 counts, Holm order, S1 and S2, episode and coefficient hashes). The gate is
exact: a differing scalar, array (dtype, shape or any value) or count stops the run (exit 3).

Run-specific fields are ignored (rule 3.1): rule_sha256, runner_sha256, module_sha256, time, input_sha256.

Steps (the order guard of round 6 is replaced by the reference SHA-256s and the pins):
  1. load_references: the three round-6 files by SHA-256 (refit_check.json supplies the coefficient hashes, as it did
     for round 6's order guard), each reference a pass.
  2. pins: pins_clip.json equals what build_pins finds now (every part, every input); the cells part equals
     r9_score.CELLS; the picks of picks_seed42.json (mean R@1 at the rule's targets) and the lambdas of
     baselines_seed42.json; the targets equal dev_seed42.json.  --check-only stops here.
  3. setup (rows, heads refit and checked against the stored posteriors, PM fits, readers), the seed-42 bundle,
     score_seed(include_pm=True), the 131 items, r9_stats.pass_record, finish().
Writes results/stage0a_regression.json (passed or not; a failed run replaces an earlier pass and also writes
stage0a_regression_failed.json), and, on a pass, results/stage0a_pass_counts.json. Exit 0 pass, 3 a difference.

    cd /project/CoSiR && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python src/test/20261202_r9_scope/run_r9_stage0a.py \
        > src/test/20261202_r9_scope/results/stage0a_run.log 2>&1
"""
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r9_common as R  # noqa: E402  (first, before anything that imports src)
import r9_bundle as B  # noqa: E402
import r9_context as X  # noqa: E402
import r9_episodes as E  # noqa: E402
import r9_heads as H  # noqa: E402
import r9_picks as P  # noqa: E402
import r9_pins as PINS  # noqa: E402
import r9_score as S  # noqa: E402
import r9_stats as ST  # noqa: E402
from r9_backbone import parts as backbone_parts  # noqa: E402

import numpy as np  # noqa: E402

from src.eval.aspect_metrics import CONDITIONS, METRICS  # noqa: E402

C, R3 = R.C, R.R3

BACKBONE = "clip_b32"
EXIT_DIFF = 3
RUN_SPECIFIC = ("rule_sha256", "runner_sha256", "module_sha256", "time", "input_sha256")   # rule 3.1
REFERENCES = {                                           # name: (path, SHA-256); the SHA-256s are rule 3.1's
    "regression_seed42.json": (R.R6D / "results/regression_seed42.json",
                               "ca168eb628f6a51e9afe779b0194088d05b293417bac04617a0d8a428d3984be"),
    "seed42_pass_counts.json": (R.R6D / "results/seed42_pass_counts.json",
                                "5618c59a3907af67bbcced0f29e0db8462b12ca4fb2e40a69e4be537de29d5e5"),
    "refit_check.json": (R.R6D / "results/refit_check.json",
                         "aebc840fa26266754370e01ae690d120d483cc86a4090b6130feb39be679736d"),
}
REGRESSION_NAME, COUNTS_NAME, REFIT_NAME = REFERENCES
OUT_NAME = "stage0a_regression.json"
OUT_COUNTS_NAME = "stage0a_pass_counts.json"
N_ITEMS = 131
R3_ARRAYS_REL = "20261121_round3_affect_gate/results/seed42_arrays.npz"
PER_ANCHOR42_REL = "20261030_aspect_baselines/results/per_anchor_seed42.npz"
DEV42_REL = "20261122_round4_aff_vetoes/results/dev_seed42.json"
WHO = ("aff", "r1")
PARTS = ("fused", "cf")
AB_SCORERS = ("cosine", "rca") + B.PM_NAMES              # the keys of per_anchor_seed42.npz the regression compares

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
    raise ImportError("run_r9_stage0a: the regression targets differ from round 3's recorded constants")


def _require(cond, msg, exc=AssertionError):
    if not cond:
        raise exc(msg)


def log(msg):
    print(f"[run_r9_stage0a] {msg}", flush=True)


def runner_sha256() -> str:
    return R.sha256_file(Path(__file__).resolve())


# ---------------------------------------------------------------- the references and the gate

def load_references() -> dict:
    """{name: parsed json} of the three round-6 reference files; a SHA-256 that differs from the pin stops the run."""
    out = {}
    for name, (path, sha) in REFERENCES.items():
        got = R.sha256_file(path)
        _require(got == sha, f"reference {name}: SHA-256 {got} differs from the pinned {sha}", SystemExit)
        out[name] = json.loads(Path(path).read_text())
    return out


def gate_references(reg, counts) -> list:
    """Problems that make the reference files unfit as a gate (empty list: fit)."""
    problems = []
    items = reg.get("items")
    if reg.get("passed") is not True or reg.get("n_failed") != 0:
        problems.append("regression record is not a pass")
    if not isinstance(items, dict) or reg.get("n_items") != len(items):
        problems.append("regression record: n_items differs from its item list")
    elif not all(it.get("equal") is True for it in items.values()):
        problems.append("regression record: an item is not equal")
    if counts.get("mode") != "regression":
        problems.append("pass counts: mode is not regression")
    return problems


def _plain(x):
    """The JSON form of a record (int keys to strings, tuples to lists, numpy scalars to Python)."""
    return json.loads(json.dumps(_norm(x)))


def compare_items(got, want) -> list:
    """Differences between our items and the reference's: missing, extra, not equal, any field differing, order."""
    got, want = _plain(got), _plain(want)
    diff = [f"{k}: missing" for k in want if k not in got]
    diff += [f"{k}: not in the reference" for k in got if k not in want]
    for k in want:
        if k not in got:
            continue
        if got[k].get("equal") is not True:
            diff.append(f"{k}: equal is not true")
        elif got[k] != want[k]:
            fields = sorted(f for f in set(got[k]) | set(want[k]) if got[k].get(f) != want[k].get(f))
            diff.append(f"{k}: {', '.join(fields)} differs")
    if not diff and list(got) != list(want):
        diff.append("item order differs from the reference")
    return diff


def compare_counts(got, want) -> list:
    """Differences between our pass counts and the reference's, without the run-specific fields."""
    got, want = _plain(got), _plain(want)
    diff = []
    for k in want:
        if k in RUN_SPECIFIC:
            continue
        if k not in got:
            diff.append(f"{k}: missing")
        elif got[k] != want[k]:
            diff.append(f"{k} differs")
    diff += [f"{k}: not in the reference" for k in got if k not in want and k not in RUN_SPECIFIC]
    return diff


def write_json(path, rec):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_norm(rec), indent=1))


def failed_path(out) -> Path:
    out = Path(out)
    return out.with_name(f"{out.stem}_failed{out.suffix}")


def finish(items, counts, refs, results, meta) -> int:
    """The end of the run: the reference must be a pass; our items and counts compared with it; the record written
    (a failed run replaces an earlier pass, with a kept copy); the pass counts written on a pass. -> 0 or EXIT_DIFF."""
    reg, ref_counts = refs[REGRESSION_NAME], refs[COUNTS_NAME]
    problems = gate_references(reg, ref_counts)
    _require(not problems, f"reference files are not a pass: {problems}")
    diff = {"items": compare_items(items, reg["items"]), "pass_counts": compare_counts(counts, ref_counts)}
    passed = not diff["items"] and not diff["pass_counts"]
    results = Path(results)
    rec = {"passed": passed, "n_items": len(items), "n_failed": sum(not it["equal"] for it in items.values()),
           "differences": diff, "items": items, "mode": "stage0a_regression",
           "references": {name: sha for name, (_, sha) in REFERENCES.items()}, **meta}
    if not passed:
        rec["pass_counts"] = counts
    path = results / OUT_NAME
    write_json(path, rec)
    if passed:
        write_json(results / OUT_COUNTS_NAME, counts)
    else:
        write_json(failed_path(path), rec)
    for name, it in items.items():
        print(f"{name}: {'pass' if it['equal'] else 'FAIL'}")
    for line in diff["items"] + diff["pass_counts"]:
        print(f"difference: {line}")
    print(f"stage 0a regression: {'PASSED' if passed else 'FAILED'} ({rec['n_items']} items, {rec['n_failed']} "
          f"failed, {len(diff['items'])} item and {len(diff['pass_counts'])} count differences); record {path}")
    return 0 if passed else EXIT_DIFF


# ---------------------------------------------------------------- shared setup

def setup() -> SimpleNamespace:
    """Everything a seed's bundle needs, fitted once: inputs asserted, rows, heads (checked), PM fits, readers."""
    H.require_threads()
    t0 = time.time()
    inputs = R.assert_inputs(BACKBONE)
    rows = R.load_rows(BACKBONE)
    log(f"data and split [{time.time() - t0:.0f}s]")
    heads = H.fit_heads_r9(rows.data, rows.split.groups, rows.split.scorer_train, backbone=BACKBONE)
    check = H.check_selection(heads, rows.data, rows.split.selection, BACKBONE)
    log(f"heads [{time.time() - t0:.0f}s]")
    pm = B.fit_pm(rows.data, rows.split.scorer_train, BACKBONE)
    readers = B.load_readers(BACKBONE)
    log(f"setup done [{time.time() - t0:.0f}s]")
    return SimpleNamespace(inputs=inputs, **vars(rows), heads=heads, head_check=check,
                           coef_sha256=H.coef_sha256(heads), pm=pm, readers=readers)


def seed_bundle(env, mode, seed, n_per_pair, on_episodes=None):
    """One seed's bundle: RowContext (episodes first, then cosine and posteriors) and build_bundle_r6."""
    ctx = X.RowContext(mode, seed, env.data, env.split, env.labels, env.heads, env.value_sets, n_per_pair,
                       on_episodes=on_episodes)
    return B.build_bundle_r6(ctx, env.readers, env.pm, BACKBONE)


# ---------------------------------------------------------------- regression items (copied from run_r6_held.py)

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
    rec = json.loads(R.INPUT_PATHS[DEV42_REL].read_text())["beside_aff"]
    _require(TARGETS["aff_minus_b1"] == _ci3(rec["AFF_minus_Bprime_A1"])
             and [EXTRA_TARGETS[f"{n}_mean_r1"] for n in P.NESTED]
             == [rec["B_mean_r1"], rec["Bprime_A0_mean_r1"], rec["Bprime_A1_mean_r1"]],
             "run_r9_stage0a's targets differ from dev_seed42.json")
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


def quarter_hits(r1) -> int:
    """Sum of round 2's as_int4 (int8 rint(4 x), multiple of 0.25 asserted) in int64 (hazard 5)."""
    return int(np.sum(R.F2.as_int4(np.asarray(r1, np.float64), "AFF r1"), dtype=np.int64))


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


def regression_items(env, bundle, scored, orders) -> tuple:
    """-> (items, bar_v): every comparison of the regression, in a fixed order (heads, episodes, numbers, round 3's
    arrays, AB's per-anchor arrays). ``orders`` = {"refit": refit_check.json, "picks": picks_seed42.json}, parsed:
    the coefficient hashes round 6 recorded (round 6's order guard read them from these two files)."""
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


# ---------------------------------------------------------------- the run

def check_pins() -> dict:
    """pins_clip.json equals what build_pins finds now; the cells part equals r9_score.CELLS and the clip record."""
    pins_file = R.pins_path(BACKBONE)
    _require(pins_file.is_file(), f"{pins_file} does not exist: run r9_pins.py --backbone {BACKBONE}", SystemExit)
    pins = json.loads(pins_file.read_text())
    _require(pins == PINS.build_pins(BACKBONE), f"{pins_file.name} differs from the files it names",
             SystemExit)  # guard:pins
    cells = P.read_cells(backbone_parts(BACKBONE)["cells_seed42"])
    _require(cells == PINS.clip_cells(), "cells_seed42_clip.json differs from CLIP's constants")
    _require([cells["aff"], cells["cf"], cells["r1_fused"], cells["r1_counterpart"]]
             == [list(S.CELLS["aff"]["fused"]), list(S.CELLS["aff"]["cf"]), list(S.CELLS["r1"]["fused"]),
                 list(S.CELLS["r1"]["cf"])], "r9_score.CELLS differ from the cells part")  # guard:cells
    return pins


def _meta(env, extra=None) -> dict:
    return {"rule_sha256": R.RULE_SHA256, "runner_sha256": runner_sha256(), "module_sha256": R.r9_module_shas(),
            "input_sha256": dict(env.inputs), "coef_sha256": env.coef_sha256, "time": R.amsterdam_now(),
            **(extra or {})}


def run(results=None, check_only=False) -> int:
    t0 = time.time()
    results = Path(results or R.RESULTS)
    refs = load_references()
    _require(not gate_references(refs[REGRESSION_NAME], refs[COUNTS_NAME]), "reference files are not a pass")
    _require(refs[REGRESSION_NAME]["n_items"] == N_ITEMS, f"the reference holds {N_ITEMS} items")
    check_pins()
    parts = backbone_parts(BACKBONE)
    picks = P.load_picks(parts["picks_seed42"], BACKBONE)              # passed, means equal the rule's targets
    lambdas = P.frozen_lambdas(BACKBONE)
    target_inputs = check_targets()
    orders = {"refit": refs[REFIT_NAME], "picks": json.loads(Path(parts["picks_seed42"]).read_text())}
    log("references, pins, picks and targets checked")
    if check_only:
        return 0
    env = setup()
    bundle = seed_bundle(env, "selection", R.DEV_SEED, R.N_PER_PAIR)
    P.check_seed42(bundle)
    scored = S.score_seed(bundle, picks, lambdas, env.readers, include_pm=True)
    log("scored")
    items, _ = regression_items(env, bundle, scored, orders)
    _require(len(items) == N_ITEMS, f"{len(items)} items, not {N_ITEMS}")
    extra = {"episodes_sha256": {R.DEV_SEED: dict(bundle.episodes_sha256)}, "runner_sha256": runner_sha256(),
             "module_sha256": R.r9_module_shas()}
    counts = ST.pass_record([scored], "regression", [R.DEV_SEED], extra)
    counts.update({"input_sha256": {**env.inputs, **target_inputs}, "coef_sha256": env.coef_sha256})
    meta = _meta(env, {"seed": R.DEV_SEED, "n_per_pair": R.N_PER_PAIR, "backbone": BACKBONE,
                       "episodes_sha256": dict(bundle.episodes_sha256), "fit_rows_sha256": bundle.fit_rows_sha256,
                       "input_sha256": counts["input_sha256"], "runtime_s": round(time.time() - t0, 1)})
    return finish(items, counts, refs, results, meta)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check-only", action="store_true", help="check references, pins, picks and targets; no data")
    a = ap.parse_args(argv)
    return run(check_only=a.check_only)


if __name__ == "__main__":
    sys.exit(main())
