"""Round 4 on seed 42 (DECISION_RULE.md of this folder: §5, §6.1, §8 boundaries, §9, §10): the regression checks
(§5 items 1 to 5, in order), then the development step (items 6 to 9) and, if a candidate is carried, the sensitivity
projection (§6.1). The main session runs it for real; implementers run only --dry.

    cd src/test/20261122_round4_aff_vetoes && \
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python run_r4_seed42.py [--dry | --boundary-reported <sha256>]

Items 1 to 5 (every comparison in results/regression_check.json with expected, obtained and pass; the first failed
item stops the run, exit 1, and nothing further is computed or written):
  1. bundle: r4_bundle.build_bundle(42, False); round 3's compare_with_round1 (D7 included) and the A1 comparisons
     (compare_a1_with_round1) against round 1's common.load_bundle(), loaded once; B'(A1)'s mean R@1; v75 and its keep
     count (check_v75);
  2. R1 = round-1 R-c (every comparison of round 3's item 2) and AFF = round 3's item 3 targets: AFF's gates through
     gates_candidate("V24", ..., pick_a1 = affect everywhere, keep = 1), equal to round 3's gates_aff exactly, and
     AFF's family run from them (run_candidate); R1's and AFF's numbers also through this round's dev_record;
  3. IMGABST_q75 = R1's gates times a_v (gates_imgabst_r1, V4's factor function) through the 224 cells, both
     cross-fits and R1's comparators (dev_record) against r4_common.IMGABST_TARGETS (themselves checked against
     bs_04_readers.json), at full precision;
  4. the A1 reader (reader_a1) equals cand_R1_A1.npz's probs__{a,b} and pick__{a,b} exactly;
  5. gate algebra at every tau index and condition: each candidate's gate is closed wherever AFF's is; V24 = V4 * V2;
     V2 = AFF * 1[stored pick__c = 0]; V4 = AFF * 1[v < 0.021043562795966864] with v from round 3's A0 features; the
     tau_0 open counts (gate statistics) as integers.
A Guard refuses every candidate computation, candidate file and candidate print until items 1 to 5 have passed in
order and regression_check.json says so (rule §5; M20); item 5's gates need items 1 to 4. Then:
  6-7. each candidate's family (run_candidate with its own gates; AFF's family is item 2's) and dev_record ->
       results/seed42_arrays.npz (per-anchor arrays, gates, cells and sigma* of R1, IMGABST, AFF, V4, V2 and V24, the
       comparators B, B'(A0), B'(A1), cosine and RCA, the readers' picks) and results/dev_seed42.json (rule SHA-256,
       Amsterdam time);
  8-9. carry -> results/carry.json (a kill when nothing is carried: no test seed is built);
  6.1. if a candidate is carried: sensitivity_all on the saved arrays -> results/sensitivity.json.
Boundaries (rule §8): a D10 clause within 1e-12 of its threshold, a Delta_k of exactly 0 or a tie gap of exactly 24
stops the run after dev_seed42.json (exit 3) with results/boundary_seed42.json, before the carry is recorded. After the
user was told, `--boundary-reported <SHA-256 of boundary_seed42.json>` writes carry.json (and sensitivity.json) from the
saved files, which must be unchanged.

--dry: the same computations on seed 42, written to results/smoke/ (*_dry.*). It prints only PASS/FAIL lines, counts and
file names; it never stops at a boundary and computes the sensitivity of every candidate, so that nothing on the console
depends on a candidate result; it checks the files it wrote, deletes them and keeps only results/smoke/seed42_dry.json
(booleans and counts). A dry run that fails at items 1 to 5 keeps regression_check_dry.json (no candidate result) for
tracing; one that raises in the development step deletes its value files. Non-smoke outputs are never overwritten.
"""
import argparse
import gc
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r4_common as R4  # noqa: E402  (imports round 3's modules by path and sets R3.TEST_SEEDS)
import r4_bundle as R4B  # noqa: E402
import r4_fusion as R4F  # noqa: E402
import r4_stats as R4S  # noqa: E402

R3, RB3, RF3, RS3, C = R4.R3, R4.RB3, R4.RF3, R4.RS3, R4.C
from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS, METRICS  # noqa: E402

RC_NPZ = "20261117_reader_fix_csd/results/cand_Rc_Rb_expected_A0.npz"
RC_JSON = "20261117_reader_fix_csd/results/cand_Rc_Rb_expected_A0.json"
RC_TAU = "20261117_reader_fix_csd/results/rc_tau.json"
BS04 = "20261120_r1_levers_brainstorm/results/bs_04_readers.json"
BS05 = "20261120_r1_levers_brainstorm/results/bs_05_aff.json"
EXT42 = "20261030_aspect_baselines/results/per_anchor_seed42.npz"
R1A1_NPZ = "20261118_reader_fix_round2/results/cand_R1_A1.npz"
R1A1_JSON = "20261118_reader_fix_round2/results/cand_R1_A1.json"
R3_FILES = [f"20261121_round3_affect_gate/{f}"
            for f in ("DECISION_RULE.md", "r3_common.py", "r3_bundle.py", "r3_fusion.py", "r3_stats.py")]
SEED42_INPUTS = R3_FILES + [RC_NPZ, RC_JSON, RC_TAU, BS04, BS05, EXT42, R1A1_NPZ, R1A1_JSON]

# the rule's text for the named cells (round 3's §5 items 2 and 3, this rule's §5 item 3): cell -> (tau index,
# lambda_u, lambda_a)
CELL_TEXT = {116: (2, 0, 2), 119: (2, 0, 16), 58: (1, 0, 0.5), 123: (2, 0.5, 1), 39: (0, 4, 16), 149: (2, 4, 4),
             10: (0, 0.5, 0.5), 117: (2, 0, 4), 67: (1, 0.5, 1)}
V75_TEXT = 0.021043562795966864                  # rule D5 and §5 item 5, as the rule writes it
LABEL = {"B_prime": "Bprime_A0", "counterpart": "counterpart", "B": "B"}   # round 1's bar labels -> r4_stats's
N_MARGINS = 24_576
USES_V = {"V4": True, "V2": False, "V24": True}   # D6: the abstention factor a_v
WHO = ("r1", "imgabst", "aff", "v4", "v2", "v24")
COMPARATORS = ("B", "Bp0", "Bp1", "cosine", "rca")
OUTPUTS = {"reg": "regression_check.json", "arr": "seed42_arrays.npz", "dev": "dev_seed42.json",
           "bnd": "boundary_seed42.json", "carry": "carry.json", "sens": "sensitivity.json"}
DRY_SUMMARY = "seed42_dry.json"
DEV_KEYS = ("fused_r1", "cf_r1", "cells", "sigma", "bar_comparator", "bar_margin", "margin_vs_counterpart",
            "gain_statistic", "either_change", "d10", "per_pair_bar_margin", "cell_text", "delta_int", "delta",
            "boundaries")


class Stop(Exception):
    """A regression item failed: nothing further is computed or written."""


class GuardError(RuntimeError):
    """A candidate result was requested while a regression item is pending (rule §5)."""


# ---------------------------------------------------------------- the guard (rule §5, M20)

class Guard:
    """No candidate result (rule §5: a score, per-anchor array, chosen cell, R@1, margin, gain, either rate, bar margin
    or Delta_k of V4, V2 or V24) is computed, written or printed before items 1 to 5 have passed. Items are marked
    passed one at a time, in order, and only when every comparison the Recorder holds for the item passed; item 5's
    gates (gate statistics) need items 1 to 4; release() needs all five and a written regression record that says so.
    Every candidate computation, file and console line of this runner goes through call(), write_json(), savez() or
    say()."""

    def __init__(self):
        self.passed, self.released = [], False

    def item_passed(self, k, rec):
        if k != len(self.passed) + 1:
            raise GuardError(f"item {k} out of order (passed so far: {self.passed})")
        rows = [r for r in rec.rows if r["item"] == k]
        if not rows or not all(r["pass"] for r in rows):
            raise GuardError(f"item {k} has not passed")
        self.passed.append(k)

    def require_gates(self):
        if self.passed[:4] != [1, 2, 3, 4]:
            raise GuardError("the candidates' gates are refused before items 1 to 4 have passed (rule §5 item 5)")

    @staticmethod
    def _record_passed(reg_path):
        p = Path(reg_path)
        if not p.exists():
            raise GuardError(f"{p.name} is not written")
        rec = json.loads(p.read_text())
        if rec.get("passed") is not True or rec.get("items_passed") != [1, 2, 3, 4, 5]:
            raise GuardError(f"{p.name} does not record items 1 to 5 as passed")

    def release(self, reg_path):
        if self.passed != [1, 2, 3, 4, 5]:
            raise GuardError(f"items 1 to 5 have not all passed ({self.passed})")
        self._record_passed(reg_path)
        self.released = True

    @classmethod
    def from_record(cls, reg_path):
        """The continuation after a reported boundary: released only by a regression record of a passed run."""
        g = cls()
        cls._record_passed(reg_path)
        g.passed, g.released = [1, 2, 3, 4, 5], True
        return g

    def require(self, what):
        if not self.released:
            raise GuardError(f"{what}: a candidate result is refused while a regression item is pending (rule §5)")

    def call(self, fn, *a, **k):
        self.require(getattr(fn, "__name__", "a computation"))
        return fn(*a, **k)

    def write_json(self, path, rec, smoke):
        self.require(Path(path).name)
        return R4.write_json_once(path, rec, smoke)

    def savez(self, path, arrays):
        self.require(Path(path).name)
        if not smoke_path(path):
            R4.refuse_existing([path], False)
        np.savez_compressed(path, **arrays)

    def say(self, msg):
        self.require("console")
        print(msg, flush=True)


def smoke_path(path):
    return Path(path).parent.name == "smoke"


# ---------------------------------------------------------------- the recorder (prints only PASS/FAIL)

def _js(x):
    if isinstance(x, np.ndarray):
        return f"array {x.shape} {x.dtype}"
    return C.jsonable(x)


def _norm(x):
    """Comparable plain-Python form (tuples -> lists, numpy scalars -> Python)."""
    if isinstance(x, (list, tuple)):
        return [_norm(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _norm(v) for k, v in x.items()}
    if isinstance(x, np.generic):
        return x.item()
    return x


class Recorder:
    """Every comparison of §5 items 1 to 5 with its expected and obtained value; prints only PASS/FAIL."""

    def __init__(self):
        self.rows = []

    def add(self, item, name, expected, got, ok=None):
        if ok is None:
            ok = _norm(expected) == _norm(got)
        self.rows.append({"item": item, "name": name, "expected": _js(expected), "got": _js(got), "pass": bool(ok)})
        print(f"CHECK item{item} {name} {'PASS' if ok else 'FAIL'}", flush=True)

    def arr(self, item, name, got, expected):
        """Exact array equality (shape and value); the JSON gets shapes, dtypes and the largest absolute difference."""
        g, e = np.asarray(got), np.asarray(expected)
        ok = bool(g.shape == e.shape and np.array_equal(g, e))
        row = {"item": item, "name": name, "expected": _js(e), "got": _js(g), "pass": ok}
        if g.shape == e.shape and g.size:
            row["max_abs_diff"] = float(np.max(np.abs(g.astype(np.float64) - e.astype(np.float64))))
        self.rows.append(row)
        print(f"CHECK item{item} {name} {'PASS' if ok else 'FAIL'}", flush=True)

    def error(self, item, exc, verbose):
        """An exception inside an item is a failed comparison of that item (type, message and traceback recorded)."""
        self.rows.append({"item": item, "name": "exception", "expected": "no exception",
                          "got": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "pass": False})
        print(f"CHECK item{item} exception FAIL ({type(exc).__name__})", flush=True)
        if verbose:
            traceback.print_exc()

    def require(self, item):
        rows = [r for r in self.rows if r["item"] == item]
        bad = [r["name"] for r in rows if not r["pass"]]
        ok = bool(rows) and not bad
        print(f"ITEM {item} {'PASS' if ok else 'FAIL'} ({len(rows)} comparisons)", flush=True)
        if not ok:
            raise Stop(item)


def _cell_text(cell):
    d = RF3.describe(int(cell))
    return (d["tau_index"], d["lambda_u"], d["lambda_a"])


def _ci3(r):
    return [r["point"], *r["ci95"]]


def _cells(fam, key):
    return [int(fam[key][0]), int(fam[key][1])]


def _sig(fam):
    return [float(fam["sigma"][0]), float(fam["sigma"][1])]


# ---------------------------------------------------------------- items 1 to 5

def item1(rec, st, guard):
    """§5 item 1: round 3's bundle comparisons (D7 included), the A1 comparisons, B'(A1)'s mean, v75."""
    b = R4B.build_bundle(42, False)
    with RB3._quiet():                         # round 1 prints seed-42 margins; discarded (rule §10)
        r1 = C.load_bundle(smoke=False)
    for part, fn in (("r3", RB3.compare_with_round1), ("a1", R4B.compare_a1_with_round1)):
        try:
            res = fn(b, r1)
        except RB3.BundleMismatch as e:
            res = e.result
        for k, v in res["checks"].items():
            rec.add(1, f"{part}.{k}", True, v is True)
        if part == "r3":
            red = res["redundancy"]
    del r1
    gc.collect()
    for h in R3.A0:
        for d in DIRECTIONS:
            rec.add(1, f"D7.{h}.{d}", R3.REDUNDANCY_42[h][d], red[h][d])
    rec.add(1, "D7.affect_least_redundant_both_directions", True, RB3.affect_least_redundant(red))
    rec.add(1, "Bprime_A1_mean_r1", R4.BPA1_MEAN_42, 100 * float(np.mean(b.pBp1["r1"])))
    try:
        res = R4B.check_v75(b)
    except RB3.BundleMismatch as e:
        res = e.result
    for k, v in res["checks"].items():
        rec.add(1, f"v75.{k}", True, v is True)
    v = np.asarray(b.v)
    rec.add(1, "v75_value", R4.V75, float(np.percentile(v, 75)))
    rec.add(1, "v75_keep_count", R4.V75_KEEP_42, int(np.count_nonzero(v < R4.V75)))
    ext = RB3.load_external(b)                 # per_anchor_seed42.npz (SHA-256 and alignment asserted)
    rec.add(1, "external_cosine_rca_loaded", True, set(ext) == {"cosine", "rca"})
    st.update(b=b, red=red, ext=ext)


def item2(rec, st, guard):
    """§5 item 2: R1 = round-1 R-c and AFF = round 3's targets (round 3's items 2 and 3), AFF through the candidate
    gate function with both factors 1 and run_candidate."""
    b, taus = st["b"], st["taus"]
    cl, pi = np.asarray(b.cl), np.asarray(b.pair_index)
    # ---- R1 (round 3's item 2)
    with np.load(R4.input_path(RC_NPZ)) as z:
        stored = {k: z[k] for k in z.files}
    sj = json.loads(R4.input_path(RC_JSON).read_text())
    rd = RF3.reader(b, readers=b.readers)
    for c in CONDITIONS:
        for d in DIRECTIONS:
            rec.arr(2, f"R1.T__{c}__{d}", rd["T"][c][d], stored[f"T__{c}__{d}"])
        rec.arr(2, f"R1.margin__{c}", rd["m"][c], stored[f"margin__{c}"])
        rec.arr(2, f"R1.pick__{c}", np.asarray(rd["pick"][c], np.int64), stored[f"pick__{c}"].astype(np.int64))
    m_all = np.concatenate([np.asarray(rd["m"]["a"], np.float64), np.asarray(rd["m"]["b"], np.float64)])
    rec.add(2, "R1.n_margins", N_MARGINS, int(m_all.size))
    rec.add(2, "R1.tau_recomputed_equals_rc_tau", list(taus),
            [float(x) for x in np.percentile(m_all, [0, 25, 50, 75])])
    rec.add(2, "R1.stored_extra_taus_equal_rc_tau", list(taus), [float(x) for x in stored["extra__taus"]])
    g_r1 = RF3.gates_r1(rd["m"], taus)
    for c in CONDITIONS:
        rec.arr(2, f"R1.gates__{c}", np.stack([g_r1[t][c] for t in range(len(taus))]), stored[f"extra__gate_{c}"])
    fam = RF3.run_family(b, rd["T"], g_r1)
    rc = R4.RC_CELLS
    rec.add(2, "R1.fused_cells", list(rc["fused"]), _cells(fam, "fpick"))
    rec.add(2, "R1.cf_cells", list(rc["cf"]), _cells(fam, "cpick"))
    rec.add(2, "R1.sigma_star", list(rc["sigma"]), _sig(fam))
    for cell in (*rc["fused"], *rc["cf"]):
        rec.add(2, f"R1.cell_{cell}_is_the_rule_text", list(CELL_TEXT[cell]), list(_cell_text(cell)))
    for m in METRICS:
        rec.arr(2, f"R1.fused__{m}", fam["fused"][m], stored[f"fused__{m}"])
        rec.arr(2, f"R1.cf__{m}", fam["cf"][m], stored[f"cf__{m}"])
    rec.add(2, "R1.cf_gain_exactly_0", True, bool((np.asarray(fam["cf"]["gain"]) == 0).all()))
    bar_v_r1, bar = C.bar_info(fam["fused"], fam["cf"], b.pBp, b.pB, cl, pi)
    rec.arr(2, "R1.bar_v", bar_v_r1, stored["bar_v"])
    gs = C.diff3(fam["fused"], fam["cf"], cl)["gain"]
    N = R4.RC_NUMBERS
    rec.add(2, "R1.comparator", N["comparator"], bar["comparator"])
    rec.add(2, "R1.fused_r1", N["fused_r1"], 100 * float(np.mean(fam["fused"]["r1"])))
    rec.add(2, "R1.cf_r1", N["cf_r1"], 100 * float(np.mean(fam["cf"]["r1"])))
    rec.add(2, "R1.bar_margin", list(N["bar"]), _ci3(bar["r1"]))
    rec.add(2, "R1.gain_statistic", list(N["gain_statistic"]), _ci3(gs))
    rec.add(2, "R1.bar_margin_equals_stored_json", _ci3(sj["bar"]["r1"]), _ci3(bar["r1"]))
    rec.add(2, "R1.gain_statistic_equals_stored_json", _ci3(sj["gain_statistic"]), _ci3(gs))
    rec.add(2, "R1.comparator_equals_stored_json", sj["bar"]["comparator"], bar["comparator"])
    dr1 = R4S.dev_record("R1", fam, None, b.pB, b.pBp, b.pBp1, cl, pi)     # item 6's function on R1
    rec.add(2, "R1.dev_record.fused_r1", N["fused_r1"], dr1["fused_r1"])
    rec.add(2, "R1.dev_record.cf_r1", N["cf_r1"], dr1["cf_r1"])
    rec.add(2, "R1.dev_record.bar_comparator", LABEL[N["comparator"]], dr1["bar_comparator"])
    rec.add(2, "R1.dev_record.bar_margin", list(N["bar"]), _ci3(dr1["bar_margin"]))
    rec.add(2, "R1.dev_record.gain_statistic", list(N["gain_statistic"]), _ci3(dr1["gain_statistic"]))
    # ---- AFF (round 3's item 3), through this round's candidate gate function and run_candidate
    A = R4.AFF_BRAINSTORM
    bs4 = json.loads(R4.input_path(BS04).read_text())["results"]["AFF"]
    bs5 = json.loads(R4.input_path(BS05).read_text())["A0"]["AFF_minus_R1"]
    rec.add(2, "AFF.rule_constants_equal_bs_04_readers.json",
            [A["fused_r1"], A["cf_r1"], A["comparator"], list(A["bar"]), list(A["margin"]), list(A["gain_statistic"]),
             A["either"], [A["per_pair_bar"][p] for p in C.POOLED_ORDER]],
            [bs4["fused_r1"], bs4["cf_r1"], bs4["comparator"], _ci3(bs4["bar"]), _ci3(bs4["margin"]),
             _ci3(bs4["gain"]), bs4["either"]["point"], [bs4["per_pair_bar"][p] for p in C.POOLED_ORDER]])
    rec.add(2, "AFF.rule_constants_equal_bs_05_aff.json",
            [list(A["aff_minus_r1_fused"]), list(A["aff_minus_r1_bar"])],
            [_ci3(bs5["fused_r1"]), _ci3(bs5["bar_margin"])])
    g_aff = RF3.gates_aff(rd["m"], rd["pick"], taus)                       # round 3's D6
    unit_pick = {c: np.full(int(b.n), R4F.AFFECT, np.int64) for c in CONDITIONS}
    unit_keep = np.ones(int(b.n), np.float32)
    g_unit = R4F.gates_candidate("V24", g_aff, pick_a1=unit_pick, keep=unit_keep)
    for t in range(len(taus)):
        for c in CONDITIONS:
            rec.arr(2, f"AFF.unit_factor_gate_equals_gates_aff.tau{t}.{c}", g_unit[t][c], g_aff[t][c])
    rec.add(2, "AFF.unit_factor_gates_float32", True,
            all(np.asarray(g_unit[t][c]).dtype == np.float32 for t in range(len(taus)) for c in CONDITIONS))
    fa = R4F.run_candidate(b, rd["T"], "V24", g_aff, pick_a1=unit_pick, keep=unit_keep)   # AFF's family
    ac = R4.AFF_CELLS
    rec.add(2, "AFF.fused_cells", list(ac["fused"]), _cells(fa, "fpick"))
    rec.add(2, "AFF.cf_cells", list(ac["cf"]), _cells(fa, "cpick"))
    rec.add(2, "AFF.sigma_star", list(ac["sigma"]), _sig(fa))
    for cell in (*ac["fused"], *ac["cf"]):
        rec.add(2, f"AFF.cell_{cell}_is_the_rule_text", list(CELL_TEXT[cell]), list(_cell_text(cell)))
    rec.add(2, "AFF.cf_gain_exactly_0", True, bool((np.asarray(fa["cf"]["gain"]) == 0).all()))
    bar_v, bar = C.bar_info(fa["fused"], fa["cf"], b.pBp, b.pB, cl, pi)
    d3 = C.diff3(fa["fused"], fa["cf"], cl)
    rec.add(2, "AFF.fused_r1", A["fused_r1"], 100 * float(np.mean(fa["fused"]["r1"])))
    rec.add(2, "AFF.cf_r1", A["cf_r1"], 100 * float(np.mean(fa["cf"]["r1"])))
    rec.add(2, "AFF.comparator", A["comparator"], bar["comparator"])
    rec.add(2, "AFF.bar_margin", list(A["bar"]), _ci3(bar["r1"]))
    rec.add(2, "AFF.margin_vs_counterpart", list(A["margin"]), _ci3(d3["r1"]))
    rec.add(2, "AFF.gain_statistic", list(A["gain_statistic"]), _ci3(d3["gain"]))
    rec.add(2, "AFF.either_vs_counterpart", A["either"], d3["either"]["point"])
    for p in C.POOLED_ORDER:
        rec.add(2, f"AFF.per_pair_bar_margin.{p}", A["per_pair_bar"][p], bar["per_pair_r1"][p]["point"])
    rec.add(2, "AFF.aff_minus_r1_fused_r1", list(A["aff_minus_r1_fused"]),
            _ci3(C.point_ci(np.asarray(fa["fused"]["r1"], np.float64) - np.asarray(fam["fused"]["r1"], np.float64),
                            cl)))
    rec.add(2, "AFF.aff_minus_r1_bar_margin", list(A["aff_minus_r1_bar"]), _ci3(C.point_ci(bar_v - bar_v_r1, cl)))
    for c in CONDITIONS:
        rec.add(2, f"AFF.tau0_open_count.{c}", A["open_tau0_counts"][c], RF3.open_count(g_aff[0], c))
    dra = R4S.dev_record("AFF", fa, None, b.pB, b.pBp, b.pBp1, cl, pi)     # item 6's function on AFF
    rec.add(2, "AFF.dev_record.fused_r1", A["fused_r1"], dra["fused_r1"])
    rec.add(2, "AFF.dev_record.cf_r1", A["cf_r1"], dra["cf_r1"])
    rec.add(2, "AFF.dev_record.bar_comparator", LABEL[A["comparator"]], dra["bar_comparator"])
    rec.add(2, "AFF.dev_record.bar_margin", list(A["bar"]), _ci3(dra["bar_margin"]))
    rec.add(2, "AFF.dev_record.margin_vs_counterpart", list(A["margin"]), _ci3(dra["margin_vs_counterpart"]))
    rec.add(2, "AFF.dev_record.gain_statistic", list(A["gain_statistic"]), _ci3(dra["gain_statistic"]))
    rec.add(2, "AFF.dev_record.either_change", A["either"], dra["either_change"])
    for p in C.POOLED_ORDER:
        rec.add(2, f"AFF.dev_record.per_pair_bar_margin.{p}", A["per_pair_bar"][p],
                dra["per_pair_bar_margin"][p]["point"])
    rec.add(2, "AFF.dev_record.d10_clears", True, dra["d10"]["clears"])
    st.update(rd=rd, g_r1=g_r1, fam_r1=fam, g_aff=g_aff, fam_aff=fa)


def item3(rec, st, guard):
    """§5 item 3: the abstention path (R1's gates times a_v) equals the brainstorm's IMGABST_q75 at full precision."""
    b, rd = st["b"], st["rd"]
    cl, pi = np.asarray(b.cl), np.asarray(b.pair_index)
    A = R4.IMGABST_TARGETS
    bs = json.loads(R4.input_path(BS04).read_text())["results"]["IMGABST_q75"]
    rec.add(3, "rule_constants_equal_bs_04_readers.json",
            [A["fused_r1"], A["cf_r1"], A["comparator"], list(A["bar_margin"]), list(A["gain_statistic"]), A["either"],
             [A["per_pair_bar"][p] for p in C.POOLED_ORDER], [list(CELL_TEXT[x]) for x in A["cells"]["fused"]],
             [list(CELL_TEXT[x]) for x in A["cells"]["cf"]]],
            [bs["fused_r1"], bs["cf_r1"], bs["comparator"], _ci3(bs["bar"]), _ci3(bs["gain"]), bs["either"]["point"],
             [bs["per_pair_bar"][p] for p in C.POOLED_ORDER], [list(x[:3]) for x in bs["fused_cells"]],
             [list(x[:3]) for x in bs["cf_cells"]]])
    v = np.asarray(b.v)
    rec.add(3, "v75_recomputed", R4.V75, float(np.percentile(v, 75)))
    keep = R4F.abstain(v)
    rec.add(3, "a_v_keep_count", R4.V75_KEEP_42, int(np.count_nonzero(keep)))
    rec.add(3, "v_b_equals_v_a", True,
            bool(np.array_equal(np.minimum(np.asarray(b.F["b"])[:, 6], np.asarray(b.F["b"])[:, 7]), v)))
    g_img = R4F.gates_imgabst_r1(st["g_r1"], keep)
    fam = RF3.run_family(b, rd["T"], g_img)
    dr = R4S.dev_record("IMGABST", fam, None, b.pB, b.pBp, b.pBp1, cl, pi)
    rec.add(3, "fused_r1", A["fused_r1"], dr["fused_r1"])
    rec.add(3, "cf_r1", A["cf_r1"], dr["cf_r1"])
    rec.add(3, "bar_comparator", A["comparator"], dr["bar_comparator"])
    rec.add(3, "bar_margin", list(A["bar_margin"]), _ci3(dr["bar_margin"]))
    rec.add(3, "gain_statistic", list(A["gain_statistic"]), _ci3(dr["gain_statistic"]))
    rec.add(3, "either_vs_counterpart", A["either"], dr["either_change"])
    for p in C.POOLED_ORDER:
        rec.add(3, f"per_pair_bar_margin.{p}", A["per_pair_bar"][p], dr["per_pair_bar_margin"][p]["point"])
    rec.add(3, "fused_cells", list(A["cells"]["fused"]), _cells(fam, "fpick"))
    rec.add(3, "cf_cells", list(A["cells"]["cf"]), _cells(fam, "cpick"))
    rec.add(3, "sigma_star", list(A["cells"]["sigma"]), _sig(fam))
    for cell in (*A["cells"]["fused"], *A["cells"]["cf"]):
        rec.add(3, f"cell_{cell}_is_the_rule_text", list(CELL_TEXT[cell]), list(_cell_text(cell)))
    rec.add(3, "cf_gain_exactly_0", True, bool((np.asarray(fam["cf"]["gain"]) == 0).all()))
    st.update(keep=keep, g_img=g_img, fam_img=fam)


def item4(rec, st, guard):
    """§5 item 4: the A1 reader equals round 2's stored R1/A1 reader (cand_R1_A1.npz) exactly."""
    b = st["b"]
    R4.assert_inputs([R1A1_NPZ, R1A1_JSON])
    with np.load(R4.input_path(R1A1_NPZ)) as z:
        stored = {k: np.asarray(z[k]) for k in ("probs__a", "probs__b", "pick__a", "pick__b")}
    ra1 = R4F.reader_a1(b)
    for c in CONDITIONS:
        P, pk = np.asarray(ra1["P"][c]), np.asarray(ra1["pick"][c])
        rec.arr(4, f"P_A1__{c}_equals_probs__{c}", P, stored[f"probs__{c}"])
        rec.add(4, f"P_A1__{c}_dtype_equals_stored", str(stored[f"probs__{c}"].dtype), str(P.dtype))
        rec.add(4, f"stored_pick__{c}_is_int8", "int8", str(stored[f"pick__{c}"].dtype))
        rec.arr(4, f"pi_A1__{c}_equals_pick__{c}", pk.astype(np.int64), stored[f"pick__{c}"].astype(np.int64))
        top = np.sort(P.astype(np.float64), axis=1)
        rec.add(4, f"no_exact_argmax_tie__{c}", 0, int(np.count_nonzero(top[:, -1] == top[:, -2])))
    st.update(ra1=ra1, pick_stored={c: stored[f"pick__{c}"] for c in CONDITIONS})


def item5(rec, st, guard):
    """§5 item 5: gate algebra on seed 42 at every tau index and condition, with an independent source of pi_A1 (the
    stored picks) and of v (round 3's A0 features against the rule's literal v75); tau_0 open counts as integers."""
    guard.require_gates()
    b, g_aff = st["b"], st["g_aff"]
    keep = R4F.abstain(b.v)
    pick = st["ra1"]["pick"]
    G = {"V4": R4F.gates_candidate("V4", g_aff, keep=keep),
         "V2": R4F.gates_candidate("V2", g_aff, pick_a1=pick),
         "V24": R4F.gates_candidate("V24", g_aff, pick_a1=pick, keep=keep)}
    Fa = np.asarray(b.F["a"], np.float64)
    below = (np.minimum(Fa[:, 6], Fa[:, 7]) < V75_TEXT).astype(np.float32)
    rec.add(5, "v75_rule_text_equals_r4_common", V75_TEXT, R4.V75)
    for t in range(len(g_aff)):
        for c in CONDITIONS:
            a = np.asarray(g_aff[t][c])
            for k in R4.CANDIDATES:
                rec.add(5, f"{k}_closed_where_AFF_closed.tau{t}.{c}", True,
                        bool(np.all(np.asarray(G[k][t][c])[a == 0] == 0)))
            rec.arr(5, f"V24_equals_V4_times_V2.tau{t}.{c}", G["V24"][t][c], G["V4"][t][c] * G["V2"][t][c])
            rec.arr(5, f"V2_equals_AFF_times_stored_pick_affect.tau{t}.{c}", G["V2"][t][c],
                    a * (np.asarray(st["pick_stored"][c]) == 0))
            rec.arr(5, f"V4_equals_AFF_times_v_below_v75.tau{t}.{c}", G["V4"][t][c], a * below)
    rec.add(5, "candidate_gates_float32_0_1", True,
            all(np.asarray(g[c]).dtype == np.float32 and bool(np.all((np.asarray(g[c]) == 0) | (np.asarray(g[c]) == 1)))
                for k in R4.CANDIDATES for g in G[k] for c in CONDITIONS))
    counts = {"AFF": {c: RF3.open_count(g_aff[0], c) for c in CONDITIONS}}
    counts.update({k: {c: RF3.open_count(G[k][0], c) for c in CONDITIONS} for k in R4.CANDIDATES})
    rec.add(5, "tau0_open_counts_are_integers", True, all(type(counts[k][c]) is int for k in counts for c in CONDITIONS))
    st.update(G=G, keep=keep, open_tau0=counts)


ITEMS = (item1, item2, item3, item4, item5)


# ---------------------------------------------------------------- items 6 to 9 and §6.1

def array_keys():
    keys = ["cl", "pair_index", "parity", "taus", "v", "keep"]
    for who in WHO:
        keys += [f"{who}_{part}__{m}" for part in ("fused", "cf") for m in METRICS]
        keys += [f"{who}_gate__{c}" for c in CONDITIONS]
        keys += [f"{who}_fused_cells", f"{who}_cf_cells", f"{who}_sigma"]
    keys += [f"{k}__{m}" for k in COMPARATORS for m in METRICS]
    for c in CONDITIONS:
        keys += [f"pick__{c}", f"margin__{c}", f"P__{c}", f"P_A1__{c}", f"pick_A1__{c}"]
    return tuple(keys)


def develop(st, guard):
    """Items 6 and 7: each candidate's family with its own gates (D6, D7) and its development record against AFF's
    family of item 2 (D8 to D10). -> ({name: family}, {name: dev_record}, AFF's dev_record)."""
    b, T, g_aff = st["b"], st["rd"]["T"], st["g_aff"]
    cl, pi = np.asarray(b.cl), np.asarray(b.pair_index)
    fam, recs = {}, {}
    for k in R4.CANDIDATES:
        fam[k] = guard.call(R4F.run_candidate, b, T, k, g_aff,
                            pick_a1=st["ra1"]["pick"] if R4.READS_CSD[k] else None,
                            keep=st["keep"] if USES_V[k] else None)
        recs[k] = guard.call(R4S.dev_record, k, fam[k], st["fam_aff"], b.pB, b.pBp, b.pBp1, cl, pi)
    aff = guard.call(R4S.dev_record, "AFF", st["fam_aff"], None, b.pB, b.pBp, b.pBp1, cl, pi)
    return fam, recs, aff


def arrays(st, fam):
    b, ext, rd, ra1 = st["b"], st["ext"], st["rd"], st["ra1"]
    fams = {"r1": st["fam_r1"], "imgabst": st["fam_img"], "aff": st["fam_aff"],
            **{k.lower(): fam[k] for k in R4.CANDIDATES}}
    gates = {"r1": st["g_r1"], "imgabst": st["g_img"], "aff": st["g_aff"],
             **{k.lower(): st["G"][k] for k in R4.CANDIDATES}}
    arr = {"cl": np.asarray(b.cl), "pair_index": np.asarray(b.pair_index), "parity": np.asarray(b.parity),
           "taus": np.asarray(st["taus"], np.float64), "v": np.asarray(b.v, np.float64),
           "keep": np.asarray(st["keep"], np.float32)}
    for who in WHO:
        f = fams[who]
        for part in ("fused", "cf"):
            for m in METRICS:
                arr[f"{who}_{part}__{m}"] = np.asarray(f[part][m], np.float64)
        for c in CONDITIONS:
            arr[f"{who}_gate__{c}"] = np.stack([np.asarray(g[c]) for g in gates[who]])
        arr[f"{who}_fused_cells"] = np.array(_cells(f, "fpick"), np.int64)
        arr[f"{who}_cf_cells"] = np.array(_cells(f, "cpick"), np.int64)
        arr[f"{who}_sigma"] = np.array(_sig(f), np.float64)
    for name, pa in (("B", b.pB), ("Bp0", b.pBp), ("Bp1", b.pBp1), ("cosine", ext["cosine"]), ("rca", ext["rca"])):
        for m in METRICS:
            arr[f"{name}__{m}"] = np.asarray(pa[m], np.float64)
    for c in CONDITIONS:
        arr[f"pick__{c}"] = np.asarray(rd["pick"][c], np.int64)
        arr[f"margin__{c}"] = np.asarray(rd["m"][c], np.float64)
        arr[f"P__{c}"] = np.asarray(rd["P"][c], np.float64)
        arr[f"P_A1__{c}"] = np.asarray(ra1["P"][c], np.float64)
        arr[f"pick_A1__{c}"] = np.asarray(ra1["pick"][c], np.int64)
    if set(arr) != set(array_keys()):
        raise AssertionError("seed42_arrays: keys differ from array_keys()")
    return arr


def seed42_view(arr, name):
    """One seed's arrays for r4_stats.sensitivity_all (§6.1), read from seed42_arrays (a dict or a loaded npz)."""
    def pa(who):
        return {m: np.asarray(arr[f"{who}__{m}"], np.float64) for m in METRICS}
    k = name.lower()
    return {"cl": np.asarray(arr["cl"]), "fused": pa(f"{k}_fused"), "cf": pa(f"{k}_cf"), "aff_fused": pa("aff_fused"),
            "B": pa("B"), "Bp0": pa("Bp0"), "Bp1": pa("Bp1") if R4.READS_CSD[name] else None,
            "cosine": pa("cosine"), "rca": pa("rca")}


def _fmt_ci(r):
    return f"{r['point']:+.4f} [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]"


def dev_table(guard, recs, aff, extra):
    guard.say("DEV seed 42 (rule §5 items 6 and 7; pp): fused R@1 | counterpart R@1 | bar comparator | bar margin "
              "[95%] | gain statistic [95%] | D10 c1 c2 c3 | Delta_int | Delta vs AFF [95%]")
    for k in R4.CANDIDATES:
        r = recs[k]
        d = r["d10"]
        guard.say(f"DEV {k:4s} {r['fused_r1']:.4f} | {r['cf_r1']:.4f} | {r['bar_comparator']} | "
                  f"{_fmt_ci(r['bar_margin'])} | {_fmt_ci(r['gain_statistic'])} | "
                  f"{int(d['c1'])} {int(d['c2'])} {int(d['c3'])} | {r['delta_int']} | {_fmt_ci(r['delta'])}")
    guard.say(f"DEV AFF  {aff['fused_r1']:.4f} | {aff['cf_r1']:.4f} | {aff['bar_comparator']} | "
              f"{_fmt_ci(aff['bar_margin'])} | {_fmt_ci(aff['gain_statistic'])} | (reference)")
    guard.say(f"DEV B'(A1) mean R@1 {extra['Bprime_A1_mean_r1']:.4f}; AFF minus B'(A1) "
              f"{_fmt_ci(extra['AFF_minus_Bprime_A1'])}")


def finish(P, dry, guard, recs, cy, arr, ack=None):
    """Items 8 and 9 recorded (carry.json), then §6.1 for the carried candidate (dry: for every candidate)."""
    guard.write_json(P["carry"], {
        "what": "rule §5 items 8 and 9 on seed 42: E = candidates that clear D10 with Delta_k > 0 (integers); M = the "
                "largest Delta_k in E; tied = members of E with M - Delta_k <= 24; carried = the first tied in the "
                "order V4, V2, V24; none carried is a kill" + ("; dry run, not a result" if dry else ""),
        "dry_run": bool(dry), "order": list(R4.CANDIDATES), "tie_band_units": R4.TIE_BAND_UNITS,
        "carry": {k: cy[k] for k in ("E", "M", "tied", "carried", "boundaries")},
        "kill": cy["carried"] is None,
        "delta_int": {k: recs[k]["delta_int"] for k in R4.CANDIDATES},
        "d10": {k: recs[k]["d10"] for k in R4.CANDIDATES},
        "boundary_reported": ack,
        "regression_check_sha256": R4.sha_file(P["reg"]), "seed42_arrays_sha256": R4.sha_file(P["arr"]),
        "dev_seed42_sha256": R4.sha_file(P["dev"])}, dry)
    print(f"{P['carry'].name} written", flush=True)
    if not dry:
        if cy["carried"] is None:
            guard.say("KILL (rule §5 item 9): no candidate cleared the development bar with Delta_k > 0; no test seed is "
                      "built; AFF stays the current best; the seed-42 results go to the user")
        else:
            guard.say(f"CARRY (rule §5 item 8): E {cy['E']}, M {cy['M']}, tied {cy['tied']}, carried {cy['carried']}")
    names = list(R4.CANDIDATES) if dry else ([cy["carried"]] if cy["carried"] is not None else [])
    if not names:
        return 0
    sens = {k: guard.call(R4S.sensitivity_all, k, seed42_view(arr, k)) for k in names}
    rec = {"what": "rule §6.1: projected pooled SE over three seeds from the carried candidate's seed-42 per-episode "
                   "differences (round 3's one-way decomposition by anchor painting); half_width = 1.96 SE, detectable "
                   "margin x = 2.80 SE, and the seed-42 bootstrap half-width beside them; percentage points; used only "
                   "to read a failed check (§6.7)" + ("; dry run: every candidate, not a result" if dry else ""),
           "dry_run": bool(dry), "units": "percentage points", "z": RS3.Z_HALF, "k_detect": RS3.K_DETECT,
           "regression_check_sha256": R4.sha_file(P["reg"]), "seed42_arrays_sha256": R4.sha_file(P["arr"]),
           "carry_sha256": R4.sha_file(P["carry"])}
    if dry:
        rec["candidates"] = sens
    else:
        rec["candidate"] = names[0]
        rec["checks"] = sens[names[0]]
    guard.write_json(P["sens"], rec, dry)
    if dry:
        print(f"{P['sens'].name} written (dry: every candidate)", flush=True)
    else:
        guard.say(f"SENSITIVITY (rule §6.1; {names[0]}; pp): SE, 1.96 SE, x = 2.80 SE, seed-42 bootstrap half-width")
        for k, v in sens[names[0]].items():
            guard.say(f"  {k:16s} {v['SE']:.4f} {v['half_width']:.4f} {v['x']:.4f} {v['seed42_half_width']:.4f}")
        print(f"{P['sens'].name} written", flush=True)
    return 0


def development(P, dry, st, guard):
    """Items 6 to 9 and §6.1 after the regression record was written and the guard released. -> exit status."""
    b = st["b"]
    cl = np.asarray(b.cl)
    fam, recs, aff = develop(st, guard)
    arr = arrays(st, fam)
    guard.savez(P["arr"], arr)
    print(f"{P['arr'].name} written ({len(arr)} arrays)", flush=True)
    aff_r1 = np.asarray(st["fam_aff"]["fused"]["r1"], np.float64)
    extra = {"Bprime_A1_mean_r1": 100 * float(np.mean(b.pBp1["r1"])),
             "Bprime_A0_mean_r1": 100 * float(np.mean(b.pBp["r1"])), "B_mean_r1": 100 * float(np.mean(b.pB["r1"])),
             "AFF_fused_r1": aff["fused_r1"],
             "AFF_minus_Bprime_A1": C.point_ci(aff_r1 - np.asarray(b.pBp1["r1"], np.float64), cl)}
    guard.write_json(P["dev"], {
        "what": "rule §5 items 6 and 7 on seed 42: each candidate's family (D7) with its own gates (D6), its "
                "comparators and bar comparator (D8), Delta_k against AFF (D9) and the development bar (D10); AFF's "
                "own record and B'(A1) beside it" + ("; dry run, not a result" if dry else ""),
        "rule_sha256": R4.RULE_SHA, "written_amsterdam": R4.now_ams(), "dry_run": bool(dry),
        "order": list(R4.CANDIDATES), "candidates": recs, "aff": aff, "beside_aff": extra,
        "open_tau0_counts": st["open_tau0"],
        "regression_check_sha256": R4.sha_file(P["reg"]), "seed42_arrays_sha256": R4.sha_file(P["arr"])}, dry)
    print(f"{P['dev'].name} written", flush=True)
    if not dry:
        dev_table(guard, recs, aff, extra)
    cy = guard.call(R4S.carry, recs)
    bnd = list(cy["boundaries"])
    if bnd and not dry:                        # rule §8: reported to the user before the carry is recorded
        guard.write_json(P["bnd"], {
            "what": "rule §8 boundary on seed 42: reported to the user before the carry is recorded; the stated "
                    "inequalities still decide; resume with --boundary-reported <SHA-256 of this file>",
            "boundaries": bnd, "carry_by_the_stated_inequalities": cy,
            "regression_check_sha256": R4.sha_file(P["reg"]), "seed42_arrays_sha256": R4.sha_file(P["arr"]),
            "dev_seed42_sha256": R4.sha_file(P["dev"])}, dry)
        guard.say("BOUNDARY (rule §8): the step that depends on these values is not recorded; tell the user, then run "
                  f"--boundary-reported {R4.sha_file(P['bnd'])}")
        for s in bnd:
            guard.say(f"BOUNDARY   {s}")
        print(f"{P['bnd'].name} written", flush=True)
        return 3
    return finish(P, dry, guard, recs, cy, arr)


# ---------------------------------------------------------------- the run

def paths(dry):
    out = R4.res_dir(dry)
    sfx = "_dry" if dry else ""
    return {k: out / f"{Path(v).stem}{sfx}{Path(v).suffix}" for k, v in OUTPUTS.items()}


def check_inputs():
    """D11 and round 3's D15: every input this runner reads, by SHA-256; tau as the rule states it."""
    return R4.assert_inputs(SEED42_INPUTS), R3.assert_taus()


def dry_finish(P, rc, n_comparisons, t0):
    """--dry: check the files written (keys and structure only), delete them, keep a record of booleans and counts."""
    ok = {}

    def check(key, fn):
        try:
            ok[P[key].name] = bool(fn())
        except Exception:                      # a missing or malformed file is a failed check
            ok[P[key].name] = False

    def j(key):
        return json.loads(P[key].read_text())

    def arr_ok():
        with np.load(P["arr"]) as z:
            return set(z.files) == set(array_keys())
    check("reg", lambda: j("reg")["passed"] is True and j("reg")["items_passed"] == [1, 2, 3, 4, 5])
    check("arr", arr_ok)
    check("dev", lambda: list(j("dev")["candidates"]) == list(R4.CANDIDATES) and j("dev")["rule_sha256"] == R4.RULE_SHA
          and all(set(DEV_KEYS) <= set(r) and type(r["delta_int"]) is int for r in j("dev")["candidates"].values()))
    check("carry", lambda: {"E", "M", "tied", "carried", "boundaries"} <= set(j("carry")["carry"])
          and j("carry")["dev_seed42_sha256"] == R4.sha_file(P["dev"]))
    check("sens", lambda: list(j("sens")["candidates"]) == list(R4.CANDIDATES)
          and all(list(j("sens")["candidates"][k]) == R4S._check_names(k) for k in R4.CANDIDATES))
    for name, v in ok.items():
        print(f"CHECK files.{name} {'PASS' if v else 'FAIL'}", flush=True)
    deleted = []
    for p in P.values():
        if p.exists():
            p.unlink()
            deleted.append(p.name)
            print(f"{p.name} deleted", flush=True)
    passed = rc == 0 and all(ok.values())
    out = P["reg"].parent / DRY_SUMMARY
    R4.write_json_once(out, {"what": "run_r4_seed42.py --dry: a dry run on seed 42, not a result; booleans and counts "
                                     "only (its value files were checked and deleted)",
                             "passed": passed, "exit_status_before_file_checks": int(rc),
                             "n_comparisons": int(n_comparisons), "files_checked": ok, "deleted": deleted}, True)
    print(f"{out.name} written", flush=True)
    print(f"SEED42_DRY {'PASS' if passed else 'FAIL'} [{int(time.time() - t0)}s]", flush=True)
    return 0 if passed else 1


def resume(sha):
    """After a reported boundary (rule §8): the carry (and the sensitivity) from the saved, unchanged files."""
    P = paths(False)
    for k in ("reg", "arr", "dev", "bnd"):
        if not P[k].exists():
            raise SystemExit(f"{P[k].name} is missing: nothing to resume")
    if R4.sha_file(P["bnd"]) != sha:
        raise SystemExit(f"--boundary-reported: the SHA-256 given differs from {P['bnd'].name}'s; refusing")
    R4.refuse_existing([P["carry"], P["sens"]], False)
    bnd = json.loads(P["bnd"].read_text())
    for k, key in (("reg", "regression_check_sha256"), ("arr", "seed42_arrays_sha256"), ("dev", "dev_seed42_sha256")):
        if R4.sha_file(P[k]) != bnd[key]:
            raise SystemExit(f"{P[k].name} changed after the boundary was recorded (SHA-256); refusing")
    guard = Guard.from_record(P["reg"])
    recs = json.loads(P["dev"].read_text())["candidates"]
    cy = guard.call(R4S.carry, recs)
    if C.jsonable(cy) != bnd["carry_by_the_stated_inequalities"]:
        raise SystemExit("the carry recomputed from dev_seed42.json differs from the boundary record; refusing")
    with np.load(P["arr"]) as z:
        arr = {k: z[k] for k in z.files}
    ack = {"boundary_seed42_sha256": sha, "boundaries": bnd["boundaries"], "acknowledged_amsterdam": R4.now_ams()}
    return finish(P, False, guard, recs, cy, arr, ack)


def run(dry, boundary_reported=None) -> int:
    R4.assert_rule()
    if boundary_reported is not None:
        if dry:
            raise SystemExit("--boundary-reported is for the real run only")
        return resume(boundary_reported)
    P = paths(dry)
    if dry:                                    # smoke files may be overwritten; stale ones are removed first
        for p in (*P.values(), P["reg"].parent / DRY_SUMMARY):
            p.unlink(missing_ok=True)
    else:
        R4.refuse_existing(P.values(), dry)
    inputs, taus = check_inputs()
    rec, st, guard, t0 = Recorder(), {"taus": tuple(taus)}, Guard(), time.time()
    base = {"what": "rule §5 items 1 to 5 on seed 42 (regression checks)" + ("; dry run, not a result" if dry else ""),
            "dry_run": bool(dry), "inputs_sha256": inputs}
    try:
        for k, fn in enumerate(ITEMS, 1):
            try:
                fn(rec, st, guard)
            except Exception as e:             # noqa: BLE001  an exception is a failed comparison of the item
                rec.error(k, e, verbose=not dry)
            rec.require(k)
            guard.item_passed(k, rec)
    except Stop as e:
        R4.write_json_once(P["reg"], {**base, "passed": False, "stopped_at_item": int(e.args[0]),
                                      "items_passed": list(guard.passed),
                                      "failed": [r["name"] for r in rec.rows if not r["pass"]],
                                      "n_comparisons": len(rec.rows), "comparisons": rec.rows,
                                      "runtime_s": int(time.time() - t0)}, dry)
        print(f"SEED42{'_DRY' if dry else ''} FAIL at item {e.args[0]}: stop and report to the user (rule §5, §9); "
              f"{P['reg'].name} written, nothing further", flush=True)
        return 1
    cells = {name: {"fused": _cells(st[f], "fpick"), "counterpart": _cells(st[f], "cpick"), "sigma": _sig(st[f])}
             for name, f in (("R1", "fam_r1"), ("AFF", "fam_aff"), ("IMGABST_q75", "fam_img"))}
    R4.write_json_once(P["reg"], {
        **base, "passed": True, "stopped_at_item": None, "items_passed": list(guard.passed), "failed": [],
        "n_comparisons": len(rec.rows), "comparisons": rec.rows, "redundancy_D7": st.get("red"), "cells": cells,
        "open_tau0_counts": st["open_tau0"], "runtime_s": int(time.time() - t0)}, dry)
    print(f"REGRESSION PASS: {len(rec.rows)} comparisons, items 1 to 5", flush=True)
    print(f"{P['reg'].name} written", flush=True)
    guard.release(P["reg"])
    try:
        rc = development(P, dry, st, guard)
    except BaseException:
        if dry:                                # a dry run never leaves value files behind
            for p in P.values():
                p.unlink(missing_ok=True)
            print("SEED42_DRY FAIL (exception in the development step; value files deleted)", flush=True)
        raise
    if dry:
        return dry_finish(P, rc, len(rec.rows), t0)
    print(f"SEED42 {'PASS' if rc == 0 else 'STOPPED (boundary)'} [{int(time.time() - t0)}s]", flush=True)
    return rc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry", action="store_true", help="write to results/smoke/ only; print only PASS/FAIL")
    ap.add_argument("--boundary-reported", metavar="SHA256", default=None,
                    help="after a rule §8 boundary was reported to the user: the SHA-256 of boundary_seed42.json")
    args = ap.parse_args(argv)
    if args.dry and args.boundary_reported is not None:
        ap.error("--boundary-reported is for the real run only")
    sys.exit(run(args.dry, args.boundary_reported))


if __name__ == "__main__":
    main()
