"""Round 5 on seed 42 (DECISION_RULE.md of this folder: §5 items 1 to 8 and the measured diagnostics, D5's positive
check, D6, D7, D8 to D11, §6.1, §8 order, boundaries and round 4's lapses 1 to 6, §9, §10 list A item 7). The main
session runs it for real, after GOEMO_FILE_SHA and GE_POST_SHA are committed in r5_common.py; implementers run only
--dry.

    cd src/test/20261123_idea3_goemotions && \\
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \\
    /root/miniconda3/envs/CoSiR/bin/python run_r5_seed42.py [--dry | --continue-boundary <sha256> | --sensitivity]

Every entry and resume path runs check_inputs() first (§8 T1, T3a-1): this rule's, round 4's and round 3's rule
SHA-256, every imported module file of rounds 1 to 4 (r5_common.assert_modules), every input this runner reads (D12 and
the tables of rounds 3 and 4), tau as the rule states it, and this round's GoEmotions file and GE posterior file against
r5_common's constants (the real paths refuse while a constant is None).

The real run, in the rule's order (§5, §8 step 5); every comparison of items 1 to 4 goes to regression_check.json with
expected, obtained and pass, the console shows CHECK/ITEM lines with PASS, FAIL or SKIP only, and the first failed item
stops the run (exit 1) with nothing further computed or written (§5, §9):
  1. item 1: r4_bundle.build_bundle(42, False); round 3's compare_with_round1 (D7 included) and round 4's
     compare_a1_with_round1 against round 1's common.load_bundle() (loaded once, its console discarded); B'(A1)'s mean;
     per_anchor_seed42.npz (round 3's §4 item 2); R1 = round-1 R-c and AFF = round 3's targets through round 3's reader,
     gates_r1, gates_aff and run_family; R1's and AFF's arrays equal round 4's seed42_arrays.npz; AFF minus B'(A1);
  2. item 2: re-asserted from cache/r5_goemotions_selection.json and its npz (file SHA-256 = GOEMO_FILE_SHA, passed,
     sample 2,048 from rng 5, tolerance 1e-4, rows = ctx.selection, sample_ids = ctx.data.sample_ids[selection]);
  3. item 3: re-asserted from results/placement.json and cache/r5_ge_posterior.npz (SHA-256 = GE_POST_SHA);
  4. item 4: the CLIP placement (r5_guard.clip_from_bundle) through this round's code: extension, G-T's and G-TF's
     reader outputs, tau' = tau, gates, families, development records (bar comparator B'_Q, D10 true, Delta_k = 0), the
     pair-lift code against told_oracle.json and the AUC code against 0.7870951145887375;
  then results/regression_check.json, and the guard released from it (r5_guard.release; D11). Only then is the GE
  placement minted (r5_guard.ge_from_file) and the steps of POST_RELEASE run:
  5. D5's positive check (booleans); item 5: stack_G, F_G, B'_G, G-T and G-TF (candidate, run_candidate with D7's
     check, and this runner's own check of the gates against D6 with an independent numpy.percentile tau'), their
     development records (D8 to D10, Delta_k) -> results/seed42_arrays.npz, then results/dev_seed42.json (rule SHA-256,
     Amsterdam time, tau'); item 6: D10 per candidate (in each record); the development table on the console;
  6. the boundary stop (§8: a D10 clause within 1e-12 of its threshold, a Delta_k of exactly 0 or a tie gap of exactly
     24): results/boundary_seed42.json, exit 3; otherwise items 7 and 8: results/carry.json and the console line
     "CARRY <name> (pending the phase-1 agreement, rule §8)" or "KILL (pending the phase-1 agreement, rule §8)";
  7. the measured diagnostics (a) to (d), refused before carry.json -> results/diagnostics_seed42.json.

--continue-boundary <SHA-256 of boundary_seed42.json>: after the boundary was reported to the user. The input check,
the given SHA-256 and the recorded SHA-256s of regression_check.json, seed42_arrays.npz and dev_seed42.json; the guard
released from regression_check.json; the carry recomputed from dev_seed42.json must equal the boundary record's;
carry.json and the console line; then the seed-42 state is rebuilt, checked against seed42_arrays.npz and tau', and the
diagnostics are written.

--sensitivity (§6.1): refused unless results/carry.json (this rule's SHA-256) names a carried candidate and the run log
20261123_idea3_goemotions_log.md holds the phase-1 agreement record: a line with "phase-1 agreement" and the SHA-256 of
results/carry.json, without "pending" (the CARRY console line says "pending the phase-1 agreement" and never counts).
The nine checks of §6.5 in its order (SENS_ORDER) through round 3's r3_stats.sensitivity on the saved seed-42 arrays
(SHA-256s chained from carry.json) -> results/sensitivity.json and the console.

--dry: items 1 to 4 on seed 42 with the CLIP placement only; items 2 and 3 SKIP while their records do not exist; the
regression record goes to results/smoke/regression_check_dry.json and the run stops where the guard would be released
(§8 lapse 5): no GE placement is minted and no GE-placement result is computed. The console prints PASS, FAIL or SKIP,
counts and file names only; every line holding a decimal number is withheld and fails the dry run. A passing dry run
deletes its regression record after checking it and keeps results/smoke/seed42_dry.json (booleans and counts).
Non-smoke outputs are never overwritten.
"""
import argparse
import contextlib
import gc
import hashlib
import io
import json
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r5_common as R5  # noqa: E402  (imports rounds 1 to 4 by path; sets R3.TEST_SEEDS)
import r5_guard as R5G  # noqa: E402
import r5_bundle as R5B  # noqa: E402
import r5_fusion as R5F  # noqa: E402
import r5_stats as R5S  # noqa: E402
import r5_diag as R5D  # noqa: E402

R3, RB3, RF3, RS3, RB4, RTO, C = R5.R3, R5.RB3, R5.RF3, R5.RS3, R5.RB4, R5.RTO, R5.C
from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS, METRICS  # noqa: E402

A0 = R3.A0
TAUS = tuple(R3.TAUS)
N_TAU = len(TAUS)
N_MARGINS = 24_576                                   # D6: tau' from G-TF's 24,576 seed-42 margins
CANDIDATES = R5.CANDIDATES
WHO = {"G-T": "gt", "G-TF": "gtf"}
RUN_LOG = HERE / "20261123_idea3_goemotions_log.md"

# inputs this runner reads (rounds 3 and 4's tables: relative to src/test/; this rule's D12: relative to the repo root)
RC_NPZ = "20261117_reader_fix_csd/results/cand_Rc_Rb_expected_A0.npz"
RC_JSON = "20261117_reader_fix_csd/results/cand_Rc_Rb_expected_A0.json"
RC_TAU = "20261117_reader_fix_csd/results/rc_tau.json"
BS04 = "20261120_r1_levers_brainstorm/results/bs_04_readers.json"
BS05 = "20261120_r1_levers_brainstorm/results/bs_05_aff.json"
EXT42 = "20261030_aspect_baselines/results/per_anchor_seed42.npz"
TOLD_JSON = "20261111_community_told_oracle/results/told_oracle.json"
BS07_JSON = "20261120_r1_levers_brainstorm/results/bs_07_detector.json"
BS07_PY = "20261120_r1_levers_brainstorm/bs_07_detector.py"
R4_ARR = "src/test/20261122_round4_aff_vetoes/results/seed42_arrays.npz"
R4_DEV = "src/test/20261122_round4_aff_vetoes/results/dev_seed42.json"
BS09_JSON = "src/test/20261120_r1_levers_brainstorm/results/bs_09_direction.json"
SEED42_INPUTS = [RC_NPZ, RC_JSON, RC_TAU, BS04, BS05, EXT42, TOLD_JSON, BS07_JSON, BS07_PY,
                 R4_ARR, R4_DEV, BS09_JSON,
                 "src/test/20261122_round4_aff_vetoes/run_r4_seed42.py",
                 "src/test/20261122_round4_aff_vetoes/r4_fusion.py",
                 "src/test/20261120_r1_levers_brainstorm/bs_09_direction.py",
                 "src/test/20261118_reader_fix_round2/results/cand_R1_A0.npz",
                 "src/data/artelingo_splits.py", "src/data/wikiart_genre.py"]

# the rule's text for the named cells (round 3's §5 items 2 and 3): cell -> (tau index, lambda_u, lambda_a)
CELL_TEXT = {116: (2, 0, 2), 119: (2, 0, 16), 58: (1, 0, 0.5), 123: (2, 0.5, 1), 39: (0, 4, 16), 149: (2, 4, 4),
             10: (0, 0.5, 0.5)}
OUTPUTS = {"reg": "regression_check.json", "arr": "seed42_arrays.npz", "dev": "dev_seed42.json",
           "bnd": "boundary_seed42.json", "carry": "carry.json", "diag": "diagnostics_seed42.json",
           "sens": "sensitivity.json"}
DRY_SUMMARY = "seed42_dry.json"
POST_RELEASE = ("positive_check_D5", "item5_development_numbers", "item6_development_bar",
                "boundary_stop_or_items7_8", "measured_diagnostics")
SENS_ORDER = ("r1_vs_cosine", "r1_vs_rca", "r1_vs_B", "r1_vs_Bprime_A0", "r1_vs_Bprime_G", "r1_vs_counterpart",
              "gain_statistic", "gain_vs_rca", "r1_vs_AFF")          # rule §6.5's nine checks, its order
# any decimal number, one-decimal numbers (0.5, 19.1), ".5" and scientific notation ("5e-03") included (round 4's)
LEAK = re.compile(r"\.\d|\d[eE][-+]?\d")


class Stop(Exception):
    """A regression item failed: nothing further is computed or written (rule §5, §9)."""


# ---------------------------------------------------------------- files

def paths(dry):
    out = R5.res_dir(dry)
    sfx = "_dry" if dry else ""
    return {k: out / f"{Path(v).stem}{sfx}{Path(v).suffix}" for k, v in OUTPUTS.items()}


def goemo_npz():
    return Path(R5.CACHE) / "r5_goemotions_selection.npz"


def goemo_json():
    return Path(R5.CACHE) / "r5_goemotions_selection.json"


def goemo_failure():
    return Path(R5.CACHE) / "r5_goemotions_item2_failure.json"


def ge_npz():
    return Path(R5.CACHE) / "r5_ge_posterior.npz"


def placement_json():
    return Path(R5.RESULTS) / "placement.json"


def check_inputs(dry=False) -> dict:
    """The full input check, first on every entry and resume path: rules, imported modules, this runner's inputs,
    tau, and this round's two own files by r5_common's constants (refused while None, except in the dry run)."""
    R5.assert_rule()
    mods = R5.assert_modules()
    shas = R5.assert_inputs(SEED42_INPUTS)
    taus = R3.assert_taus()
    own = {}
    for const, path, what in (("GOEMO_FILE_SHA", goemo_npz(), "GoEmotions file"),
                              ("GE_POST_SHA", ge_npz(), "GE posterior file")):
        want = getattr(R5, const)
        if want is None:
            if not dry:
                raise SystemExit(f"r5_common.{const} is not set: the {what} is not yet an input; refusing (rule D2, D4)")
            own[const] = None
            continue
        if not Path(path).is_file():
            raise SystemExit(f"the {what} {Path(path).name} is missing although r5_common.{const} is set")
        if R5.sha256_file(path) != want:
            raise SystemExit(f"the {what} {Path(path).name}: SHA-256 differs from r5_common.{const}")
        own[const] = want
    return {"modules": mods, "inputs": shas, "taus": list(taus), "own_files": own}


def _write_npz(path, arrays):
    path = Path(path)
    tmp = path.with_name(path.stem + ".partial.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


# ---------------------------------------------------------------- GE outputs pass the guard (D11)

def write_ge_json(path, rec, ge):
    """A results JSON that holds GE-placement results: the guard first (refused before release), never overwritten."""
    R5G.require(ge, "run_r5_seed42.write_ge_json")
    return R5.write_json_once(path, rec, False)


def savez_ge(path, arrays, ge):
    R5G.require(ge, "run_r5_seed42.savez_ge")
    R5.refuse_existing([path], False)
    _write_npz(path, arrays)


def say_ge(msg, ge):
    """A console line that may hold a GE-placement result: the guard first."""
    R5G.require(ge, "run_r5_seed42.say_ge")
    print(msg, flush=True)


# ---------------------------------------------------------------- the recorder (prints PASS, FAIL or SKIP only)

def _plain(x):
    """JSON-safe comparison value: arrays as shape and dtype, tuples as lists, non-finite floats as their repr."""
    if isinstance(x, np.ndarray):
        return f"array {x.shape} {x.dtype}"
    x = C.jsonable(x)
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_plain(v) for v in x]
    if isinstance(x, float) and not np.isfinite(x):
        return repr(x)
    return x


def _norm(x):
    if isinstance(x, (list, tuple)):
        return [_norm(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _norm(v) for k, v in x.items()}
    if isinstance(x, np.generic):
        return x.item()
    return x


class Recorder:
    """Every comparison of items 1 to 4 with its expected and obtained value; the console gets PASS, FAIL or SKIP."""

    def __init__(self):
        self.rows = []

    def _row(self, item, name, status, **kw):
        self.rows.append({"item": item, "name": name, "status": status, "pass": status == "pass", **kw})
        print(f"CHECK item{item} {name} {status.upper()}{kw.get('note', '')}", flush=True)

    def add(self, item, name, expected, got, ok=None):
        if ok is None:
            ok = _norm(expected) == _norm(got)
        self._row(item, name, "pass" if ok else "fail", expected=_plain(expected), got=_plain(got))

    def arr(self, item, name, got, expected, dtype=False):
        """Exact array equality (shape and values; with dtype=True also the dtype)."""
        g, e = np.asarray(got), np.asarray(expected)
        ok = bool(g.shape == e.shape and np.array_equal(g, e) and (not dtype or g.dtype == e.dtype))
        extra = {}
        if g.shape == e.shape and g.size and np.issubdtype(g.dtype, np.number) and np.issubdtype(e.dtype, np.number):
            dif = np.abs(g.astype(np.float64) - e.astype(np.float64))
            extra["max_abs_diff"] = float(np.nanmax(dif)) if np.isfinite(dif).any() else "nan"
        self._row(item, name, "pass" if ok else "fail", expected=_plain(e), got=_plain(g), **extra)

    def skip(self, item, name, reason):
        self._row(item, name, "skip", expected="a record", got=reason, note=f" ({reason})")

    def error(self, item, exc, verbose):
        """An exception inside an item is a failed comparison of that item (type, message, traceback recorded)."""
        self.rows.append({"item": item, "name": "exception", "status": "fail", "pass": False,
                          "expected": "no exception", "got": f"{type(exc).__name__}: {exc}",
                          "traceback": traceback.format_exc()})
        print(f"CHECK item{item} exception FAIL ({type(exc).__name__})", flush=True)
        if verbose:
            traceback.print_exc()

    def require(self, item, allow_skip=False) -> str:
        """'pass' or 'skip' (dry run only, every row of the item skipped); otherwise Stop."""
        rows = [r for r in self.rows if r["item"] == item]
        st = {r["status"] for r in rows}
        if rows and st == {"pass"}:
            status = "pass"
        elif rows and st == {"skip"} and allow_skip:
            status = "skip"
        else:
            status = "fail"
        print(f"ITEM {item} {status.upper()} ({len(rows)} comparisons)", flush=True)
        if status == "fail":
            raise Stop(item)
        return status


class Order:
    """Rule §5 and §8: items 1 to 4 are marked one at a time, in order; the guard is released only after all four
    passed in this process and results/regression_check.json records them as passed (r5_guard.release)."""

    def __init__(self):
        self.done = []

    def mark(self, k, status):
        if k != len(self.done) + 1:
            raise R5G.GuardError(f"item {k} out of order (done so far: {self.done})")
        if status not in ("pass", "skip"):
            raise R5G.GuardError(f"item {k}: status {status!r}")
        self.done.append((k, status))

    def all_passed(self):
        return self.done == [(1, "pass"), (2, "pass"), (3, "pass"), (4, "pass")]

    def release(self, reg_path):
        if not self.all_passed():
            raise R5G.GuardError(f"the guard is released only after items 1 to 4 passed in order ({self.done})")
        R5G.release(reg_path)


def _ci3(r):
    return [r["point"], *r["ci95"]]


def _cells(fam, key):
    return [int(fam[key][0]), int(fam[key][1])]


def _sig(fam):
    return [float(fam["sigma"][0]), float(fam["sigma"][1])]


def _cell_text(cell):
    d = RF3.describe(int(cell))
    return (d["tau_index"], d["lambda_u"], d["lambda_a"])


def _f64(x):
    return np.asarray(x, dtype=np.float64)


# ---------------------------------------------------------------- shared state: AFF on the CLIP placement

def aff_core(b, taus) -> dict:
    """AFF exactly as round 3 tested it (D1): the frozen A0 half-readers on F with the bundle's stack, AFF's gates at
    tau, round 3's run_family. -> {"rd", "g_aff", "fam_aff"}."""
    rd = RF3.reader(b, readers=b.readers)
    g_aff = RF3.gates_aff(rd["m"], rd["pick"], taus)
    return {"rd": rd, "g_aff": g_aff, "fam_aff": RF3.run_family(b, rd["T"], g_aff)}


def aff_reference(st) -> dict:
    """AFF's seed-42 numbers (round 3's §5 item 3 measures), the reference block of dev_seed42.json."""
    b, fa = st["b"], st["fam_aff"]
    cl, pi = np.asarray(b.cl), np.asarray(b.pair_index)
    _, bar = C.bar_info(fa["fused"], fa["cf"], b.pBp, b.pB, cl, pi)
    d3 = C.diff3(fa["fused"], fa["cf"], cl)
    return {"fused_r1": 100 * float(np.mean(fa["fused"]["r1"])), "cf_r1": 100 * float(np.mean(fa["cf"]["r1"])),
            "bar_comparator": bar["comparator"], "bar_margin": bar["r1"], "margin_vs_counterpart": d3["r1"],
            "gain_statistic": d3["gain"], "either_change": d3["either"]["point"],
            "per_pair_bar_margin": bar["per_pair_r1"], "cells": {"fused": _cells(fa, "fpick"),
                                                                 "cf": _cells(fa, "cpick")},
            "sigma": _sig(fa),
            "open_tau0_counts": {c: RF3.open_count(st["g_aff"][0], c) for c in CONDITIONS},
            "minus_Bprime_A1": C.point_ci(_f64(fa["fused"]["r1"]) - _f64(b.pBp1["r1"]), cl)}


def selection_labels(b):
    """labS (emotion, style and genre codes on the selection rows) and gS (painting groups), as run_told_oracle."""
    from src.data.artelingo_splits import artelingo_aspect_labels
    sel = np.asarray(b.ctx.selection)
    lab = artelingo_aspect_labels(b.ctx.data)
    return {a: lab[a][sel] for a in ("emotion", "style", "genre")}, np.asarray(b.ctx.groups)[sel]


def base_state(b, ext, labS, gS, taus=TAUS) -> dict:
    """The state the post-release steps read: bundle, external baselines, AFF (aff_core), the CLIP placement, the
    selection labels and AFF's reference numbers."""
    st = {"b": b, "taus": tuple(taus), "ext": ext, "labS": labS, "gS": gS, "sel": np.asarray(b.ctx.selection)}
    st.update(aff_core(b, taus))
    st["clip"] = R5G.clip_from_bundle(b)
    st["aff_ref"] = aff_reference(st)
    return st


def rebuild_state() -> dict:
    """The continuation's seed-42 state, rebuilt in a new process (no regression comparison; the guard is released
    from the recorded regression check)."""
    b = RB4.build_bundle(42, False)
    labS, gS = selection_labels(b)
    return base_state(b, RB3.load_external(b), labS, gS)


# ---------------------------------------------------------------- §5 items 1 to 4

def item1(rec, st):
    """Item 1: bundle (round 3's and round 4's comparisons against round 1), R1, AFF and B'(A1)."""
    taus = st["taus"]
    b = RB4.build_bundle(42, False)
    with RB3._quiet():                                   # round 1 prints seed-42 margins; discarded (rule §10)
        r1 = C.load_bundle(smoke=False)
    red = None
    for part, fn in (("r3", RB3.compare_with_round1), ("a1", RB4.compare_a1_with_round1)):
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
    for h in A0:
        for d in DIRECTIONS:
            rec.add(1, f"D7.{h}.{d}", R3.REDUNDANCY_42[h][d], red[h][d])
    rec.add(1, "D7.affect_least_redundant_both_directions", True, RB3.affect_least_redundant(red))
    rec.add(1, "Bprime_A1_mean_r1", R5.BPRIME_A1_MEAN_42, 100 * float(np.mean(b.pBp1["r1"])))
    ext = RB3.load_external(b)                           # per_anchor_seed42.npz, SHA-256 and alignment asserted
    rec.add(1, "external_cosine_rca_loaded", True, set(ext) == {"cosine", "rca"})
    st.update(b=b, ext=ext, red=red)
    cl, pi = np.asarray(b.cl), np.asarray(b.pair_index)

    # ---- R1 = round-1 R-c (round 3's §5 item 2)
    with np.load(R3.input_path(RC_NPZ)) as z:
        stored = {k: z[k] for k in z.files}
    sj = json.loads(R3.input_path(RC_JSON).read_text())
    core = aff_core(b, taus)
    rd = core["rd"]
    for c in CONDITIONS:
        for d in DIRECTIONS:
            rec.arr(1, f"R1.T__{c}__{d}", rd["T"][c][d], stored[f"T__{c}__{d}"])
        rec.arr(1, f"R1.margin__{c}", rd["m"][c], stored[f"margin__{c}"])
        rec.arr(1, f"R1.pick__{c}", np.asarray(rd["pick"][c], np.int64), stored[f"pick__{c}"].astype(np.int64))
    m_all = np.concatenate([_f64(rd["m"]["a"]), _f64(rd["m"]["b"])])
    rec.add(1, "R1.n_margins", N_MARGINS, int(m_all.size))
    rec.add(1, "R1.tau_recomputed_equals_rc_tau", list(taus), [float(x) for x in np.percentile(m_all, [0, 25, 50, 75])])
    rec.add(1, "R1.stored_extra_taus_equal_rc_tau", list(taus), [float(x) for x in stored["extra__taus"]])
    g_r1 = RF3.gates_r1(rd["m"], taus)
    for c in CONDITIONS:
        rec.arr(1, f"R1.gates__{c}", np.stack([g_r1[t][c] for t in range(N_TAU)]), stored[f"extra__gate_{c}"])
    fam_r1 = RF3.run_family(b, rd["T"], g_r1)
    rc = R3.RC_CELLS
    rec.add(1, "R1.fused_cells", list(rc["fused"]), _cells(fam_r1, "fpick"))
    rec.add(1, "R1.cf_cells", list(rc["cf"]), _cells(fam_r1, "cpick"))
    rec.add(1, "R1.sigma_star", list(rc["sigma"]), _sig(fam_r1))
    for cell in (*rc["fused"], *rc["cf"]):
        rec.add(1, f"R1.cell_{cell}_is_the_rule_text", list(CELL_TEXT[cell]), list(_cell_text(cell)))
    for m in METRICS:
        rec.arr(1, f"R1.fused__{m}", fam_r1["fused"][m], stored[f"fused__{m}"])
        rec.arr(1, f"R1.cf__{m}", fam_r1["cf"][m], stored[f"cf__{m}"])
    rec.add(1, "R1.cf_gain_exactly_0", True, bool((np.asarray(fam_r1["cf"]["gain"]) == 0).all()))
    bar_v_r1, bar_r1 = C.bar_info(fam_r1["fused"], fam_r1["cf"], b.pBp, b.pB, cl, pi)
    rec.arr(1, "R1.bar_v", bar_v_r1, stored["bar_v"])
    gs_r1 = C.diff3(fam_r1["fused"], fam_r1["cf"], cl)["gain"]
    N = R3.RC_NUMBERS
    rec.add(1, "R1.comparator", N["comparator"], bar_r1["comparator"])
    rec.add(1, "R1.fused_r1", N["fused_r1"], 100 * float(np.mean(fam_r1["fused"]["r1"])))
    rec.add(1, "R1.cf_r1", N["cf_r1"], 100 * float(np.mean(fam_r1["cf"]["r1"])))
    rec.add(1, "R1.bar_margin", list(N["bar"]), _ci3(bar_r1["r1"]))
    rec.add(1, "R1.gain_statistic", list(N["gain_statistic"]), _ci3(gs_r1))
    rec.add(1, "R1.bar_margin_equals_stored_json", _ci3(sj["bar"]["r1"]), _ci3(bar_r1["r1"]))
    rec.add(1, "R1.gain_statistic_equals_stored_json", _ci3(sj["gain_statistic"]), _ci3(gs_r1))

    # ---- AFF (round 3's §5 item 3, round 4's §5 item 2): round 3's reader, gates_aff and run_family
    A = R5.AFF_ITEM1
    rec.add(1, "AFF.r5_constants_equal_round3s", R3.AFF_BRAINSTORM, A)
    rec.add(1, "AFF.r5_cells_equal_round3s", R3.AFF_CELLS, R5.AFF_CELLS)
    bs4 = json.loads(R3.input_path(BS04).read_text())["results"]["AFF"]
    bs5 = json.loads(R3.input_path(BS05).read_text())["A0"]["AFF_minus_R1"]
    rec.add(1, "AFF.rule_constants_equal_bs_04_readers.json",
            [A["fused_r1"], A["cf_r1"], A["comparator"], list(A["bar"]), list(A["margin"]), list(A["gain_statistic"]),
             A["either"], [A["per_pair_bar"][p] for p in C.POOLED_ORDER]],
            [bs4["fused_r1"], bs4["cf_r1"], bs4["comparator"], _ci3(bs4["bar"]), _ci3(bs4["margin"]),
             _ci3(bs4["gain"]), bs4["either"]["point"], [bs4["per_pair_bar"][p] for p in C.POOLED_ORDER]])
    rec.add(1, "AFF.rule_constants_equal_bs_05_aff.json", [list(A["aff_minus_r1_fused"]), list(A["aff_minus_r1_bar"])],
            [_ci3(bs5["fused_r1"]), _ci3(bs5["bar_margin"])])
    g_aff, fa = core["g_aff"], core["fam_aff"]
    ac = R5.AFF_CELLS
    rec.add(1, "AFF.fused_cells", list(ac["fused"]), _cells(fa, "fpick"))
    rec.add(1, "AFF.cf_cells", list(ac["cf"]), _cells(fa, "cpick"))
    rec.add(1, "AFF.sigma_star", list(ac["sigma"]), _sig(fa))
    for cell in (*ac["fused"], *ac["cf"]):
        rec.add(1, f"AFF.cell_{cell}_is_the_rule_text", list(CELL_TEXT[cell]), list(_cell_text(cell)))
    rec.add(1, "AFF.cf_gain_exactly_0", True, bool((np.asarray(fa["cf"]["gain"]) == 0).all()))
    bar_v, bar = C.bar_info(fa["fused"], fa["cf"], b.pBp, b.pB, cl, pi)
    d3 = C.diff3(fa["fused"], fa["cf"], cl)
    rec.add(1, "AFF.fused_r1", A["fused_r1"], 100 * float(np.mean(fa["fused"]["r1"])))
    rec.add(1, "AFF.cf_r1", A["cf_r1"], 100 * float(np.mean(fa["cf"]["r1"])))
    rec.add(1, "AFF.comparator", A["comparator"], bar["comparator"])
    rec.add(1, "AFF.bar_margin", list(A["bar"]), _ci3(bar["r1"]))
    rec.add(1, "AFF.margin_vs_counterpart", list(A["margin"]), _ci3(d3["r1"]))
    rec.add(1, "AFF.gain_statistic", list(A["gain_statistic"]), _ci3(d3["gain"]))
    rec.add(1, "AFF.either_vs_counterpart", A["either"], d3["either"]["point"])
    for p in C.POOLED_ORDER:
        rec.add(1, f"AFF.per_pair_bar_margin.{p}", A["per_pair_bar"][p], bar["per_pair_r1"][p]["point"])
    rec.add(1, "AFF.aff_minus_r1_fused_r1", list(A["aff_minus_r1_fused"]),
            _ci3(C.point_ci(_f64(fa["fused"]["r1"]) - _f64(fam_r1["fused"]["r1"]), cl)))
    rec.add(1, "AFF.aff_minus_r1_bar_margin", list(A["aff_minus_r1_bar"]), _ci3(C.point_ci(bar_v - bar_v_r1, cl)))
    for c in CONDITIONS:
        rec.add(1, f"AFF.tau0_open_count.{c}", A["open_tau0_counts"][c], RF3.open_count(g_aff[0], c))

    # ---- R1's and AFF's arrays equal round 4's seed42_arrays.npz (keys r1_*, aff_*)
    with np.load(R5.input_path(R4_ARR)) as z:
        r4 = {k: z[k] for k in z.files if k.startswith(("r1_", "aff_"))}
    for who, fam, g in (("r1", fam_r1, g_r1), ("aff", fa, g_aff)):
        for part in ("fused", "cf"):
            for m in METRICS:
                rec.arr(1, f"round4_arrays.{who}_{part}__{m}", _f64(fam[part][m]), r4[f"{who}_{part}__{m}"])
        for c in CONDITIONS:
            rec.arr(1, f"round4_arrays.{who}_gate__{c}", np.stack([g[t][c] for t in range(N_TAU)]),
                    r4[f"{who}_gate__{c}"], dtype=True)
        rec.arr(1, f"round4_arrays.{who}_fused_cells", np.array(_cells(fam, "fpick"), np.int64), r4[f"{who}_fused_cells"])
        rec.arr(1, f"round4_arrays.{who}_cf_cells", np.array(_cells(fam, "cpick"), np.int64), r4[f"{who}_cf_cells"])
        rec.arr(1, f"round4_arrays.{who}_sigma", np.array(_sig(fam), np.float64), r4[f"{who}_sigma"])

    # ---- AFF minus B'(A1) (round 4's dev_seed42.json, beside_aff)
    r4dev = json.loads(R5.input_path(R4_DEV).read_text())["beside_aff"]["AFF_minus_Bprime_A1"]
    want = [R5.AFF_MINUS_BP1[0], *R5.AFF_MINUS_BP1[1]]
    rec.add(1, "AFF_minus_Bprime_A1_constant_equals_round4_dev_seed42", want, _ci3(r4dev))
    rec.add(1, "AFF_minus_Bprime_A1", want, _ci3(C.point_ci(_f64(fa["fused"]["r1"]) - _f64(b.pBp1["r1"]), cl)))
    st.update(core)
    st.update(g_r1=g_r1, fam_r1=fam_r1, sel=np.asarray(b.ctx.selection))
    st["aff_ref"] = aff_reference(st)


def _absent(paths_):
    return [Path(p).name for p in paths_ if not Path(p).is_file()]


def item2(rec, st):
    """Item 2, re-asserted from the GoEmotions step's record and file (rule §5: file SHA-256 equal to r5_common's
    constant, passed, the stated sample, tolerance and rows)."""
    absent = _absent([goemo_json(), goemo_npz()])
    if absent and st.get("dry"):
        rec.skip(2, "goemotions_record", "record absent; dry run")
        return
    rec.add(2, "goemotions_record_present", [], absent)
    if absent:
        return
    rec.add(2, "goemotions_file_sha_constant_set", True, R5.GOEMO_FILE_SHA is not None)
    if R5.GOEMO_FILE_SHA is None:
        return
    data = goemo_npz().read_bytes()                      # one read: the bytes hashed are the bytes loaded
    sha = hashlib.sha256(data).hexdigest()
    rec.add(2, "goemotions_file_sha256_equals_r5_common", R5.GOEMO_FILE_SHA, sha)
    r = json.loads(goemo_json().read_text())
    i2 = r.get("item2") if isinstance(r.get("item2"), dict) else {}
    rec.add(2, "record_rule_sha256", R5.RULE_SHA, r.get("rule_sha256"))
    rec.add(2, "record_npz_sha256_equals_file", sha, r.get("npz_sha256"))
    rec.add(2, "no_item2_failure_record", False, goemo_failure().exists())
    rec.add(2, "item2_passed", True, i2.get("passed") is True)
    rec.add(2, "item2_sample_rng", R5.REG_SAMPLE[0], i2.get("sample_rng"))
    rec.add(2, "item2_sample_size", R5.REG_SAMPLE[1], i2.get("sample_size"))
    rec.add(2, "item2_tolerance", R5.GOEMO_TOL, i2.get("tol"))
    mx = i2.get("max_abs")
    rec.add(2, "item2_max_abs_within_tolerance", True,
            isinstance(mx, float) and bool(np.isfinite(mx)) and mx <= R5.GOEMO_TOL)
    rec.add(2, "batch_size", R5.GOEMO_BATCH, r.get("batch_size"))
    rec.add(2, "max_length", R5.GOEMO_MAXLEN, r.get("max_length"))
    rec.add(2, "n_rows", R5.N_SELECTION, r.get("n_rows"))
    rec.add(2, "n_captions", R5.N_SELECTION, r.get("n_captions"))
    with np.load(io.BytesIO(data)) as z:
        probs, rows, sids = z["probs"], z["rows"], z["sample_ids"]
    ctx = st["b"].ctx
    sel = np.asarray(ctx.selection)
    rec.add(2, "probs_float32_n_by_28", ["float32", [R5.N_SELECTION, R5.N_GOEMO]], [str(probs.dtype), list(probs.shape)])
    rec.add(2, "probs_finite_in_0_1", True, bool(np.isfinite(probs).all() and probs.min() >= 0 and probs.max() <= 1))
    rec.add(2, "probs_sha256_equals_record", r.get("probs_sha256"), hashlib.sha256(probs.tobytes()).hexdigest())
    rec.arr(2, "rows_equal_ctx_selection", rows, sel.astype(np.int64), dtype=True)
    rec.arr(2, "sample_ids_equal_ctx_sample_ids_at_selection", sids,
            np.asarray(ctx.data.sample_ids)[sel].astype(np.int64), dtype=True)


def item3(rec, st):
    """Item 3, re-asserted from the placement step's record (results/placement.json) and the GE posterior file
    (SHA-256 equal to r5_common.GE_POST_SHA)."""
    absent = _absent([placement_json(), ge_npz()])
    if absent and st.get("dry"):
        rec.skip(3, "placement_record", "record absent; dry run")
        return
    rec.add(3, "placement_record_present", [], absent)
    if absent:
        return
    rec.add(3, "ge_posterior_sha_constant_set", True, R5.GE_POST_SHA is not None)
    if R5.GE_POST_SHA is None:
        return
    data = ge_npz().read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    rec.add(3, "ge_posterior_sha256_equals_r5_common", R5.GE_POST_SHA, sha)
    r = json.loads(placement_json().read_text())
    i3 = r.get("item3") if isinstance(r.get("item3"), dict) else {}
    rec.add(3, "record_rule_sha256", R5.RULE_SHA, r.get("rule_sha256"))
    rec.add(3, "record_ge_posterior_sha256_equals_file", sha, r.get("ge_posterior_sha256"))
    rec.add(3, "no_placement_failure_record", False, (Path(R5.RESULTS) / "placement_failure.json").exists())
    rec.add(3, "item3_passed", True, i3.get("passed") is True)
    for m in ("txt", "img"):
        rec.add(3, f"item3_equals_fit_one_head_{m}", True, (i3.get("equals_fit_one_head") or {}).get(m) is True)
        rec.add(3, f"item3_classes_ok_{m}", True, (i3.get("classes_ok") or {}).get(m) is True)
        rec.add(3, f"item3_accuracy_equals_record_{m}", True, i3.get(f"accuracy_equals_record_{m}") is True)
        acc = (i3.get("heldout_accuracy") or {}).get(m)
        rec.add(3, f"item3_heldout_accuracy_{m}", R5.CLIP_HEAD_RECORD["heldout_accuracy"][m],
                round(acc, 2) if isinstance(acc, float) else acc)
    for k in ("draw_positions_ok", "head_roundtrip_equals_told_oracle_arm_L", "head_equals_constants"):
        rec.add(3, f"item3_{k}", True, i3.get(k) is True)
    rec.add(3, "ge_head_record_present", True, isinstance(r.get("ge_head"), dict))
    with np.load(io.BytesIO(data)) as z:
        rows, classes = z["rows"], z["classes"]
    rec.arr(3, "ge_rows_equal_ctx_selection", rows, np.asarray(st["b"].ctx.selection).astype(np.int64), dtype=True)
    rec.arr(3, "ge_classes_are_0_to_40", classes, np.arange(R5.N_CLASSES))


def _same(x, y):
    x, y = np.asarray(x), np.asarray(y)
    return bool(x.shape == y.shape and x.dtype == y.dtype and np.array_equal(x, y))


def item4(rec, st):
    """Item 4: the CLIP placement through this round's code reproduces the bundle's fields and AFF exactly."""
    b, rd, fa, g_aff = st["b"], st["rd"], st["fam_aff"], st["g_aff"]
    cl, pi, sel = np.asarray(b.cl), np.asarray(b.pair_index), np.asarray(b.ctx.selection)
    clip = R5G.clip_from_bundle(b)
    rec.add(4, "clip_placement_kind", "clip", clip.kind)
    ext = R5B.extend(b, clip)
    for k, v in R5B.same_as_bundle(b, ext).items():
        rec.add(4, f"clip_ext.{k}", True, v is True)
    with np.load(R5.input_path(R4_ARR)) as z:
        r4 = {k: z[k] for k in z.files if k.startswith("aff_")}
    A = R5.AFF_ITEM1
    for name in CANDIDATES:
        cand = R5F.candidate(name, b, ext)               # G-TF on seed 42: tau' from its margins (here AFF's)
        for c in CONDITIONS:
            rec.add(4, f"{name}.P.{c}_equals_AFF", True, _same(cand["P"][c], rd["P"][c]))
            rec.add(4, f"{name}.m.{c}_equals_AFF", True, _same(cand["m"][c], rd["m"][c]))
            rec.add(4, f"{name}.pick.{c}_equals_AFF", True, _same(cand["pick"][c], rd["pick"][c]))
            for d in DIRECTIONS:
                rec.add(4, f"{name}.T.{c}.{d}_equals_AFF", True, _same(cand["T"][c][d], rd["T"][c][d]))
        rec.add(4, f"{name}.taus_equal_D1_each_element", [True] * N_TAU,
                [bool(x == y) for x, y in zip(cand["taus"], TAUS)] if len(cand["taus"]) == N_TAU else [])
        for t in range(N_TAU):
            for c in CONDITIONS:
                rec.add(4, f"{name}.gate.tau{t}.{c}_equals_AFF", True, _same(cand["gates"][t][c], g_aff[t][c]))
        fam = R5F.run_candidate(name, b, ext, cand)
        rec.add(4, f"{name}.fused_cells", list(R5.AFF_CELLS["fused"]), _cells(fam, "fpick"))
        rec.add(4, f"{name}.cf_cells", list(R5.AFF_CELLS["cf"]), _cells(fam, "cpick"))
        rec.add(4, f"{name}.sigma_star", list(R5.AFF_CELLS["sigma"]), _sig(fam))
        for part in ("fused", "cf"):
            for m in METRICS:
                rec.arr(4, f"{name}.{part}__{m}_equals_item1", fam[part][m], fa[part][m])
                rec.arr(4, f"{name}.{part}__{m}_equals_round4_aff", _f64(fam[part][m]), r4[f"aff_{part}__{m}"])
        dr = R5S.dev_record(name, fam, fa, b.pB, b.pBp, ext.pBp, b.pBp1, cl, pi, taus=cand["taus"])
        rec.add(4, f"{name}.dev.fused_r1", A["fused_r1"], dr["fused_r1"])
        rec.add(4, f"{name}.dev.cf_r1", A["cf_r1"], dr["cf_r1"])
        rec.add(4, f"{name}.dev.bar_comparator_is_Bprime_Q", "Bprime_G", dr["bar_comparator"])
        rec.add(4, f"{name}.dev.Bprime_Q_mean_equals_Bprime_A0_mean", True,
                dr["comparator_means"]["Bprime_G"] == dr["comparator_means"]["Bprime_A0"])
        rec.add(4, f"{name}.dev.Bprime_Q_first_in_D8_tie_order", "Bprime_G",
                R5S.comparators(b.pB, b.pBp, ext.pBp, fam["cf"])[0][0])
        rec.add(4, f"{name}.dev.bar_margin", list(A["bar"]), _ci3(dr["bar_margin"]))
        rec.add(4, f"{name}.dev.margin_vs_counterpart", list(A["margin"]), _ci3(dr["margin_vs_counterpart"]))
        rec.add(4, f"{name}.dev.gain_statistic", list(A["gain_statistic"]), _ci3(dr["gain_statistic"]))
        rec.add(4, f"{name}.dev.either_change", A["either"], dr["either_change"])
        for p in C.POOLED_ORDER:
            rec.add(4, f"{name}.dev.per_pair_bar_margin.{p}", A["per_pair_bar"][p],
                    dr["per_pair_bar_margin"][p]["point"])
        rec.add(4, f"{name}.dev.d10_all_three_clauses", {"c1": True, "c2": True, "c3": True, "clears": True},
                dr["d10"])
        rec.add(4, f"{name}.dev.delta_int_zero_against_AFF", 0, dr["delta_int"])
        rec.add(4, f"{name}.dev.beside_AFF_minus_Bprime_A1", [R5.AFF_MINUS_BP1[0], *R5.AFF_MINUS_BP1[1]],
                _ci3(dr["beside_Bprime_A1"]["candidate_minus"]))
        rec.add(4, f"{name}.dev.Bprime_Q_minus_Bprime_A0_zero", 0.0, dr["Bprime_G_minus_Bprime_A0"]["point"])
    # pair lift: told_oracle.json arms.L.pairs.heads through this round's code
    labS, gS = selection_labels(b)
    Pi = np.asarray(b.post["affect"]["img"])[sel].astype(np.float64)
    Pt = clip.Q[sel].astype(np.float64)
    lift = R5D.pair_lift(Pi, Pt, labS, gS, clip, sel)
    stored = json.loads(R3.input_path(TOLD_JSON).read_text())["arms"]["L"]["pairs"]["heads"]
    rec.add(4, "pair_lift.heads_equal_told_oracle_arms_L_pairs_heads", stored, RTO.roundtrip(lift["full"]))
    rec.add(4, "pair_lift.ratios_equal_rule", R5.PAIR_RATIOS,
            {k: lift[k] for k in ("ratio_same_over_diff", "emotionxstyle", "emotionxgenre")})
    # the AUC code of diagnostic (a) on AFF's P^c(affect)
    bs07 = json.loads(R5.R4C.input_path(BS07_JSON).read_text())["results"]["R1aff"]["auc_emotion"]
    rec.add(4, "auc_constant_equals_bs_07_detector", bs07, R5.AUC_AFF)
    rec.add(4, "auc_emotion_AFF", R5.AUC_AFF, R5D.auc_emotion({c: rd["P"][c] for c in CONDITIONS}, pi, clip))
    st.update(clip=clip, labS=labS, gS=gS)
    del ext
    gc.collect()


ITEMS = (item1, item2, item3, item4)


# ---------------------------------------------------------------- after the release: item 5 to the carry

def runner_gate_check(name, st, cand, fam) -> dict:
    """D7 in this runner, independent of r5_fusion: tau recomputed from the definition (G-T: AFF's tau; G-TF:
    numpy.percentile, linear, at 0, 25, 50, 75 over the 24,576 margins, condition a first), the gates the family ran
    from equal 1[m >= tau_t] * 1[pick = affect] at every tau index and condition (float32), and for G-T the reader
    outputs and the gates are AFF's of item 1. Raises AssertionError naming the parts that differ."""
    if name == "G-T":
        want_taus, m, pick = TAUS, st["rd"]["m"], st["rd"]["pick"]
    else:
        m, pick = cand["m"], cand["pick"]
        allm = np.concatenate([_f64(m["a"]), _f64(m["b"])])
        if allm.size != N_MARGINS:
            raise AssertionError(f"rule D7 (runner): tau' needs the {N_MARGINS} seed-42 margins, got {allm.size}")
        want_taus = tuple(float(x) for x in np.percentile(allm, [0, 25, 50, 75]))
    gates = fam["gates"]
    out = {"taus_equal_definition": (len(cand["taus"]) == N_TAU and len(fam["taus"]) == N_TAU
                                     and all(x == y for x, y in zip(cand["taus"], want_taus))
                                     and all(x == y for x, y in zip(fam["taus"], want_taus))),
           "gates_equal_definition": len(gates) == N_TAU and all(
               _same(gates[t][c], ((_f64(m[c]) >= want_taus[t]) & (np.asarray(pick[c]) == 0)).astype(np.float32))
               for t in range(N_TAU) for c in CONDITIONS),
           "gates_equal_candidates": len(gates) == N_TAU and all(_same(gates[t][c], cand["gates"][t][c])
                                                                 for t in range(N_TAU) for c in CONDITIONS)}
    if name == "G-T":
        out["reader_outputs_are_affs"] = all(_same(cand[k][c], st["rd"][k][c]) for k in ("P", "m", "pick")
                                             for c in CONDITIONS)
        out["gates_are_affs_of_item1"] = all(_same(gates[t][c], st["g_aff"][t][c]) for t in range(N_TAU)
                                             for c in CONDITIONS)
    bad = [k for k, v in out.items() if v is not True]
    if bad:
        raise AssertionError(f"rule D7 (runner): {name}'s {bad} differ from D6's definition (the gates a family runs "
                             f"from must be D6's at every tau index and condition)")
    return out


def develop_core(st, ge) -> dict:
    """D5's extension and positive check with Q_GE, then each candidate (D6) and its family (D7: r5_fusion's check
    and this runner's own). -> {"ge", "ext", "pc", "cand", "fam", "gate_checks"}. Nothing is written or printed."""
    R5G.require(ge, "run_r5_seed42.develop_core")
    b = st["b"]
    ext = R5B.extend(b, ge)
    pc = R5B.positive_check(b, ext, ge)
    if not (pc.get("all_pass") is True and all(v is True for v in pc.values())):
        raise AssertionError(f"rule D5: the positive check failed ({[k for k, v in pc.items() if v is not True]})")
    cand, fam, checks = {}, {}, {}
    for name in CANDIDATES:
        cand[name] = R5F.candidate(name, b, ext)         # G-TF: tau' computed on seed 42 from its own margins
        fam[name] = R5F.run_candidate(name, b, ext, cand[name])
        checks[name] = {"runner": runner_gate_check(name, st, cand[name], fam[name]),
                        "family_D7": fam[name].get("gate_check")}
    return {"ge": ge, "ext": ext, "pc": pc, "cand": cand, "fam": fam, "gate_checks": checks}


def records(st, dv):
    """Item 5's records (D8 to D10, Delta_k against AFF) and the descriptive comparator lines."""
    R5G.require(dv["ge"], "run_r5_seed42.records")
    b, ext = st["b"], dv["ext"]
    cl, pi = np.asarray(b.cl), np.asarray(b.pair_index)
    recs = {name: R5S.dev_record(name, dv["fam"][name], st["fam_aff"], b.pB, b.pBp, ext.pBp, b.pBp1, cl, pi,
                                 taus=dv["cand"][name]["taus"]) for name in CANDIDATES}
    opens = {"AFF": {c: RF3.open_count(st["g_aff"][0], c) for c in CONDITIONS}}
    opens.update({name: {c: RF3.open_count(dv["fam"][name]["gates"][0], c) for c in CONDITIONS}
                  for name in CANDIDATES})
    extra = {"comparators": {"B_mean_r1": 100 * float(np.mean(b.pB["r1"])),
                             "Bprime_A0_mean_r1": 100 * float(np.mean(b.pBp["r1"])),
                             "Bprime_G_mean_r1": 100 * float(np.mean(ext.pBp["r1"])),
                             "Bprime_G_minus_Bprime_A0": C.point_ci(_f64(ext.pBp["r1"]) - _f64(b.pBp["r1"]), cl)},
             "beside": {"Bprime_A1_mean_r1": 100 * float(np.mean(b.pBp1["r1"])),
                        "AFF_minus_Bprime_A1": st["aff_ref"]["minus_Bprime_A1"]},
             "open_tau0_counts": opens}
    return recs, extra


def array_keys():
    keys = ["cl", "pair_index", "parity", "taus", "tau_prime"]
    for who in ("aff", "gt", "gtf"):
        keys += [f"{who}_{part}__{m}" for part in ("fused", "cf") for m in METRICS]
        keys += [f"{who}_gate__{c}" for c in CONDITIONS]
        keys += [f"{who}_fused_cells", f"{who}_cf_cells", f"{who}_sigma"]
    keys += [f"{k}__{m}" for k in ("B", "Bp0", "BpG", "Bp1", "cosine", "rca") for m in METRICS]
    for who in ("aff", "gtf"):
        keys += [f"{who}_{x}__{c}" for x in ("P", "margin", "pick") for c in CONDITIONS]
    return tuple(keys)


def seed42_arrays(st, dv) -> dict:
    """seed42_arrays.npz (§4 item 8, §5 item 5): per scorer the per-anchor arrays, the gates the families ran from,
    the chosen cells and sigma*; tau and tau'; B, B'(A0), B'_G, B'(A1), cosine and RCA per anchor; the readers' P,
    margins and picks for AFF and G-TF."""
    R5G.require(dv["ge"], "run_r5_seed42.seed42_arrays")
    b, ext, fam = st["b"], dv["ext"], dv["fam"]
    arr = {"cl": np.asarray(b.cl), "pair_index": np.asarray(b.pair_index), "parity": np.asarray(b.parity),
           "taus": np.asarray(TAUS, np.float64), "tau_prime": np.asarray(dv["cand"]["G-TF"]["taus"], np.float64)}
    for who, f, g in (("aff", st["fam_aff"], st["g_aff"]), ("gt", fam["G-T"], fam["G-T"]["gates"]),
                      ("gtf", fam["G-TF"], fam["G-TF"]["gates"])):
        for part in ("fused", "cf"):
            for m in METRICS:
                arr[f"{who}_{part}__{m}"] = _f64(f[part][m])
        for c in CONDITIONS:
            arr[f"{who}_gate__{c}"] = np.stack([np.asarray(g[t][c]) for t in range(N_TAU)])
        arr[f"{who}_fused_cells"] = np.array(_cells(f, "fpick"), np.int64)
        arr[f"{who}_cf_cells"] = np.array(_cells(f, "cpick"), np.int64)
        arr[f"{who}_sigma"] = np.array(_sig(f), np.float64)
    for name, pa in (("B", b.pB), ("Bp0", b.pBp), ("BpG", ext.pBp), ("Bp1", b.pBp1), ("cosine", st["ext"]["cosine"]),
                     ("rca", st["ext"]["rca"])):
        for m in METRICS:
            arr[f"{name}__{m}"] = _f64(pa[m])
    for who, r in (("aff", st["rd"]), ("gtf", dv["cand"]["G-TF"])):
        for c in CONDITIONS:
            arr[f"{who}_P__{c}"] = _f64(r["P"][c])
            arr[f"{who}_margin__{c}"] = _f64(r["m"][c])
            arr[f"{who}_pick__{c}"] = np.asarray(r["pick"][c], np.int64)
    if set(arr) != set(array_keys()):
        raise AssertionError("seed42_arrays: keys differ from array_keys()")
    return arr


def _fmt_ci(r):
    return f"{r['point']:+.4f} [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]"


def dev_table(ge, recs, st, extra, tau_prime):
    say_ge("DEV seed 42 (rule §5 items 5 and 6; pp): fused R@1 | counterpart R@1 | bar comparator | bar margin [95%] | "
           "gain statistic [95%] | D10 c1 c2 c3 | Delta_int | Delta vs AFF [95%] | tau_0 open a/b", ge)
    op = extra["open_tau0_counts"]
    for k in CANDIDATES:
        r, d = recs[k], recs[k]["d10"]
        say_ge(f"DEV {k:4s} {r['fused_r1']:.4f} | {r['cf_r1']:.4f} | {r['bar_comparator']} | "
               f"{_fmt_ci(r['bar_margin'])} | {_fmt_ci(r['gain_statistic'])} | {int(d['c1'])} {int(d['c2'])} "
               f"{int(d['c3'])} | {r['delta_int']} | {_fmt_ci(r['delta'])} | {op[k]['a']}/{op[k]['b']}", ge)
    a = st["aff_ref"]
    say_ge(f"DEV AFF  {a['fused_r1']:.4f} | {a['cf_r1']:.4f} | {a['bar_comparator']} | {_fmt_ci(a['bar_margin'])} | "
           f"{_fmt_ci(a['gain_statistic'])} | (reference) | {op['AFF']['a']}/{op['AFF']['b']}", ge)
    cp, bs = extra["comparators"], extra["beside"]
    say_ge(f"DEV B'_G mean R@1 {cp['Bprime_G_mean_r1']:.4f}; B'(A0) {cp['Bprime_A0_mean_r1']:.4f}; B "
           f"{cp['B_mean_r1']:.4f}; B'_G minus B'(A0) {_fmt_ci(cp['Bprime_G_minus_Bprime_A0'])}", ge)
    say_ge(f"DEV beside: B'(A1) mean R@1 {bs['Bprime_A1_mean_r1']:.4f}; AFF minus B'(A1) "
           f"{_fmt_ci(bs['AFF_minus_Bprime_A1'])}; "
           + "; ".join(f"{k} minus B'(A1) {_fmt_ci(recs[k]['beside_Bprime_A1']['candidate_minus'])}"
                       for k in CANDIDATES), ge)
    say_ge(f"DEV tau' (G-TF) {list(tau_prime)}", ge)


def finish_carry(P, cy, ge, ack=None):
    """Items 7 and 8 recorded: results/carry.json (this rule's SHA-256 at the top level, as r5_guard.require_carry
    reads it), then the console line, which says the carry or kill is pending the phase-1 agreement (§8 lapse 6)."""
    rec = {**cy, "what": "rule §5 items 7 and 8 on seed 42: E = candidates that clear D10 with Delta_k > 0 "
                         "(integers); M = the largest Delta_k in E; tied = members of E with M - Delta_k <= 24; "
                         "carried = the first tied in the order G-T, G-TF; E empty is a kill; pending the phase-1 "
                         "agreement (rule §8)",
           "order": list(CANDIDATES), "tie_band_units": R5.TIE_BAND_UNITS, "boundary_reported": ack,
           "regression_check_sha256": R5.sha256_file(P["reg"]), "seed42_arrays_sha256": R5.sha256_file(P["arr"]),
           "written_amsterdam": R5.now_ams()}
    write_ge_json(P["carry"], rec, ge)
    print(f"{P['carry'].name} written", flush=True)
    say_ge(R5S.console_line(cy), ge)


def _strip(x):
    """The sharper-term output without its episode arrays ("diff", for the paired AFF comparison only)."""
    return {k: v for k, v in x.items() if k != "diff"}


def diagnostics(st, dv, recs, ge, carry_path) -> dict:
    """The measured diagnostics (§5; descriptive, decide nothing), in the rule's order (a) to (d), refused before
    results/carry.json exists (D11)."""
    R5G.require(ge, "run_r5_seed42.diagnostics")
    try:
        R5G.require_carry(carry_path)
    except R5G.GuardError as e:
        raise R5G.GuardError(f"run_r5_seed42.diagnostics: refused before results/carry.json ({e})") from None
    b, ext, cand, fam, clip = st["b"], dv["ext"], dv["cand"], dv["fam"], st["clip"]
    cl, pi, sel = np.asarray(b.cl), np.asarray(b.pair_index), st["sel"]
    out = {}
    # (a) detection AUC of G-TF's P'^c(affect), beside AFF's (R1's)
    out["a_detection_auc"] = {
        "G-TF": R5D.auc_emotion({c: cand["G-TF"]["P"][c] for c in CONDITIONS}, pi, ge, carry_path),
        "AFF_recomputed": R5D.auc_emotion({c: st["rd"]["P"][c] for c in CONDITIONS}, pi, clip, carry_path),
        "AFF_rule": R5.AUC_AFF}
    # (b) feature-level AUC of Delta_affect (column 2): F_G's against F's
    out["b_delta_affect_auc"] = {"F_G": R5D.auc_delta(ext.F, pi, ge, carry_path),
                                 "F": R5D.auc_delta(b.F, pi, clip, carry_path)}
    # (c) emotion pair lift through the placement, against the CLIP values of item 4, beside the group lift
    Pi = np.asarray(b.post["affect"]["img"])[sel].astype(np.float64)
    lift = R5D.pair_lift(Pi, ge.Q[sel].astype(np.float64), st["labS"], st["gS"], ge, sel, carry_path)
    out["c_pair_lift"] = {"GE": lift, "CLIP_item4": R5.PAIR_RATIOS, "group_lift": R5.GROUP_LIFT}
    # (d) the sharper term
    aff_d = R5D.sharper_term(b, st["rd"]["T"], st["g_aff"], st["fam_aff"], cl, pi, TAUS, clip, carry_path)
    d = {"AFF": _strip(aff_d)}
    for name in CANDIDATES:
        d[name] = _strip(R5D.sharper_term(b, cand[name]["T"], fam[name]["gates"], fam[name], cl, pi,
                                          cand[name]["taus"], ge, carry_path, aff=aff_d))
    out["d_sharper_term"] = d
    bs09 = json.loads(R5.input_path(BS09_JSON).read_text())
    out["d_orientation_bs_09_direction"] = {k: bs09[k] for k in ("AFF", "R1")}
    a = st["aff_ref"]
    out["d_either_cost_per_unit_gain"] = {          # None where the gain statistic's point is exactly 0
        **{name: (R5D.either_cost_per_gain(recs[name]) if recs[name]["gain_statistic"]["point"] != 0 else None)
           for name in CANDIDATES},
        "AFF_recomputed": (-a["either_change"] / a["gain_statistic"]["point"]
                           if a["gain_statistic"]["point"] != 0 else None),
        "AFF_rule": R5.AFF_EITHER_PER_GAIN}
    out.update({"what": "rule §5 measured diagnostics (a) to (d) on seed 42: descriptive, decide nothing",
                "rule_sha256": R5.RULE_SHA, "written_amsterdam": R5.now_ams(),
                "carry_sha256": R5.sha256_file(carry_path)})
    return out


def development(P, st) -> int:
    """POST_RELEASE, after the regression record was written and the guard released. -> exit status (0, or 3 at a
    boundary)."""
    ge = R5G.ge_from_file(ge_npz())                      # minted only now (D11: SHA-256 = GE_POST_SHA)
    dv = develop_core(st, ge)
    print("CHECK D5.positive_check PASS", flush=True)
    recs, extra = records(st, dv)
    arr = seed42_arrays(st, dv)
    savez_ge(P["arr"], arr, ge)
    print(f"{P['arr'].name} written ({len(arr)} arrays)", flush=True)
    tau_prime = list(dv["cand"]["G-TF"]["taus"])
    write_ge_json(P["dev"], {
        "what": "rule §5 items 5 and 6 on seed 42: G-T and G-TF with the GE placement (D6), their families (D7), "
                "comparators and bar comparator (D8), Delta_k against AFF (D9) and the development bar (D10); AFF's "
                "reference numbers, B'_G and B'(A1) beside",
        "rule_sha256": R5.RULE_SHA, "written_amsterdam": R5.now_ams(), "order": list(CANDIDATES),
        "candidates": recs, "tau_prime": tau_prime, "taus_AFF": list(TAUS), **extra,
        "aff": st["aff_ref"], "gates_checked": dv["gate_checks"], "positive_check_D5": dv["pc"],
        "regression_check_sha256": R5.sha256_file(P["reg"]), "seed42_arrays_sha256": R5.sha256_file(P["arr"]),
        "goemotions_file_sha256": R5.GOEMO_FILE_SHA, "ge_posterior_sha256": ge.sha256}, ge)
    print(f"{P['dev'].name} written", flush=True)
    dev_table(ge, recs, st, extra, tau_prime)
    cy = R5S.carry(recs, R5.sha256_file(P["dev"]))
    if cy["boundaries"]:                                 # rule §8: reported before the carry is recorded
        write_ge_json(P["bnd"], {
            "what": "rule §8 boundary on seed 42: reported to the user before the carry is recorded; the stated "
                    "inequalities still decide; resume with --continue-boundary <SHA-256 of this file>",
            "rule_sha256": R5.RULE_SHA, "boundaries": cy["boundaries"], "carry_by_the_stated_inequalities": cy,
            "regression_check_sha256": R5.sha256_file(P["reg"]), "seed42_arrays_sha256": R5.sha256_file(P["arr"]),
            "dev_seed42_sha256": R5.sha256_file(P["dev"])}, ge)
        for s in cy["boundaries"]:
            say_ge(f"BOUNDARY   {s}", ge)
        say_ge("BOUNDARY (rule §8): the carry is not recorded; tell the user, then run --continue-boundary "
               f"{R5.sha256_file(P['bnd'])}", ge)
        return 3
    finish_carry(P, cy, ge)
    diag = diagnostics(st, dv, recs, ge, P["carry"])
    write_ge_json(P["diag"], diag, ge)
    print(f"{P['diag'].name} written", flush=True)
    return 0


# ---------------------------------------------------------------- --continue-boundary (§8 boundaries, T3a-4)

def verify_saved(st, dv, P):
    """The rebuilt state reproduces seed42_arrays.npz exactly (every key, value, shape and dtype), tau' included."""
    R5G.require(dv["ge"], "run_r5_seed42.verify_saved")
    now = seed42_arrays(st, dv)
    with np.load(P["arr"]) as z:
        saved = {k: z[k] for k in z.files}
    bad = sorted(k for k in set(now) | set(saved) if k not in now or k not in saved or not _same(now[k], saved[k]))
    if bad:
        raise SystemExit(f"the rebuilt seed-42 state differs from the saved seed42_arrays.npz in {bad}; refusing")


def continue_boundary(sha) -> int:
    check_inputs(False)                                  # the full input check first (T3a-1)
    P = paths(False)
    missing = [P[k].name for k in ("reg", "arr", "dev", "bnd") if not P[k].exists()]
    if missing:
        raise SystemExit(f"--continue-boundary: {missing} missing; nothing to resume")
    R5.refuse_existing([P["carry"], P["diag"], P["sens"]], False)
    if R5.sha256_file(P["bnd"]) != sha:
        raise SystemExit(f"--continue-boundary: the SHA-256 given differs from {P['bnd'].name}'s; refusing")
    bnd = json.loads(P["bnd"].read_text())
    for k, key in (("reg", "regression_check_sha256"), ("arr", "seed42_arrays_sha256"), ("dev", "dev_seed42_sha256")):
        if R5.sha256_file(P[k]) != bnd.get(key):
            raise SystemExit(f"{P[k].name} changed after the boundary was recorded (SHA-256); refusing")
    R5G.release(P["reg"])                                # only a record of items 1 to 4 passed releases
    ge = R5G.ge_from_file(ge_npz())
    dev = json.loads(P["dev"].read_text())
    recs = dev["candidates"]
    cy = R5S.carry(recs, R5.sha256_file(P["dev"]))
    if C.jsonable(cy) != bnd["carry_by_the_stated_inequalities"]:
        raise SystemExit("the carry recomputed from dev_seed42.json differs from the boundary record's; refusing")
    ack = {"boundary_seed42_sha256": sha, "boundaries": bnd["boundaries"], "acknowledged_amsterdam": R5.now_ams()}
    finish_carry(P, cy, ge, ack)
    st = rebuild_state()
    dv = develop_core(st, ge)
    verify_saved(st, dv, P)
    if list(dv["cand"]["G-TF"]["taus"]) != dev["tau_prime"]:
        raise SystemExit("the rebuilt tau' differs from dev_seed42.json's; refusing")
    diag = diagnostics(st, dv, recs, ge, P["carry"])
    write_ge_json(P["diag"], diag, ge)
    print(f"{P['diag'].name} written", flush=True)
    return 0


# ---------------------------------------------------------------- --sensitivity (§6.1)

def agreement_line(carry_sha) -> str:
    """The phase-1 agreement record of this round's run log: a line holding 'phase-1 agreement' and the SHA-256 of
    results/carry.json, without 'pending' (the CARRY/KILL console line never counts). The last such line."""
    text = Path(RUN_LOG).read_text() if Path(RUN_LOG).is_file() else ""
    hits = [ln for ln in text.splitlines()
            if "phase-1 agreement" in ln.lower() and "pending" not in ln.lower() and carry_sha in ln]
    if not hits:
        raise SystemExit("--sensitivity: the run log holds no phase-1 agreement record for this carry (a line with "
                         f"'phase-1 agreement' and carry.json's SHA-256 {carry_sha}, not 'pending'); refused "
                         "(rule §6.1, §8)")
    return hits[-1]


def sensitivity_diffs(carried, arr) -> dict:
    """§6.1's per-episode differences of the carried candidate, in §6.5's order: its fused R@1 minus cosine's, RCA's,
    B's, B'(A0)'s, B'_G's and its counterpart's; its fused gain minus the counterpart's (exactly 0, so 'minus 0'); its
    fused gain minus RCA's; its fused R@1 minus AFF's fused R@1."""
    k = WHO[carried]

    def pa(who):
        return {m: _f64(arr[f"{who}__{m}"]) for m in ("r1", "gain")}
    fu, cf, aff = pa(f"{k}_fused"), pa(f"{k}_cf"), pa("aff_fused")
    if not np.all(cf["gain"] == 0):
        raise AssertionError("the counterpart's condition gain is not exactly 0")
    out = {"r1_vs_cosine": fu["r1"] - pa("cosine")["r1"], "r1_vs_rca": fu["r1"] - pa("rca")["r1"],
           "r1_vs_B": fu["r1"] - pa("B")["r1"], "r1_vs_Bprime_A0": fu["r1"] - pa("Bp0")["r1"],
           "r1_vs_Bprime_G": fu["r1"] - pa("BpG")["r1"], "r1_vs_counterpart": fu["r1"] - cf["r1"],
           "gain_statistic": fu["gain"] - cf["gain"], "gain_vs_rca": fu["gain"] - pa("rca")["gain"],
           "r1_vs_AFF": fu["r1"] - aff["r1"]}
    if tuple(out) != SENS_ORDER:
        raise AssertionError("sensitivity checks out of rule §6.5's order")
    return out


def sensitivity_checks(carried, arr) -> dict:
    d = sensitivity_diffs(carried, arr)
    cl = np.asarray(arr["cl"])
    return {k: RS3.sensitivity(d[k], cl) for k in SENS_ORDER}


def sensitivity() -> int:
    check_inputs(False)                                  # the full input check first (T3a-1)
    P = paths(False)
    cy = R5G.require_carry(P["carry"])                   # this rule's SHA-256, else GuardError
    carried = cy.get("carried")
    if carried not in CANDIDATES:
        raise SystemExit("--sensitivity: results/carry.json names no carried candidate (a kill); refused (rule §6.1)")
    carry_sha = R5.sha256_file(P["carry"])
    line = agreement_line(carry_sha)
    R5.refuse_existing([P["sens"]], False)
    R5G.release(P["reg"])
    ge = R5G.ge_from_file(ge_npz())
    R5G.require(ge, "run_r5_seed42.sensitivity")
    dev_sha = R5.sha256_file(P["dev"])
    if dev_sha != cy.get("dev_seed42_sha256"):
        raise SystemExit("dev_seed42.json: SHA-256 differs from carry.json's; refusing")
    dev = json.loads(P["dev"].read_text())
    arr_sha = R5.sha256_file(P["arr"])
    if arr_sha != dev.get("seed42_arrays_sha256") or arr_sha != cy.get("seed42_arrays_sha256"):
        raise SystemExit("seed42_arrays.npz: SHA-256 differs from dev_seed42.json's or carry.json's; refusing")
    with np.load(P["arr"]) as z:
        arr = {k: z[k] for k in z.files}
    sens = sensitivity_checks(carried, arr)
    write_ge_json(P["sens"], {
        "what": "rule §6.1: projected pooled SE over three seeds from the carried candidate's seed-42 per-episode "
                "differences (round 3's r3_stats.sensitivity, one-way decomposition by anchor painting); half_width = "
                "1.96 SE, detectable margin x = 2.80 SE, and the seed-42 bootstrap half-width beside them; percentage "
                "points; used only to read a failed check (§6.7)",
        "rule_sha256": R5.RULE_SHA, "written_amsterdam": R5.now_ams(), "candidate": carried,
        "order": list(SENS_ORDER), "units": "percentage points", "z": RS3.Z_HALF, "k_detect": RS3.K_DETECT,
        "checks": sens, "agreement_line": line, "carry_sha256": carry_sha, "dev_seed42_sha256": dev_sha,
        "seed42_arrays_sha256": arr_sha}, ge)
    say_ge(f"SENSITIVITY (rule §6.1; {carried}; pp): SE, 1.96 SE, x = 2.80 SE, seed-42 bootstrap half-width", ge)
    for k, v in sens.items():
        say_ge(f"  {k:18s} {v['SE']:.4f} {v['half_width']:.4f} {v['x']:.4f} {v['seed42_half_width']:.4f}", ge)
    print(f"{P['sens'].name} written", flush=True)
    return 0


# ---------------------------------------------------------------- the dry run's console (§8 lapse 5, §10)

class _Withhold(io.TextIOBase):
    """A console stream that withholds every line holding a decimal number (LEAK) and counts it."""

    def __init__(self, target, counter):
        self.target, self.counter, self.buf = target, counter, ""

    def writable(self):
        return True

    def write(self, s):
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            self._emit(line)
        return len(s)

    def _emit(self, line):
        if LEAK.search(line):
            self.counter[0] += 1
            self.target.write("[a line holding a decimal number was withheld: dry run]\n")
        else:
            self.target.write(line + "\n")

    def flush(self):
        self.target.flush()

    def finish(self):
        if self.buf:
            self._emit(self.buf)
            self.buf = ""
        self.target.flush()

    def isatty(self):
        return False

    @property
    def encoding(self):
        return getattr(self.target, "encoding", "utf-8")


@contextlib.contextmanager
def no_decimal_console():
    counter = [0]
    out, err = _Withhold(sys.stdout, counter), _Withhold(sys.stderr, counter)
    old = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = out, err
    try:
        yield counter
    finally:
        out.finish()
        err.finish()
        sys.stdout, sys.stderr = old


def _non_smoke_listing():
    root = Path(R5.RESULTS)
    if not root.exists():
        return []
    return sorted((str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns) for p in root.rglob("*")
                  if p.is_file() and "smoke" not in p.relative_to(root).parts)


def dry_finish(P, rec, order, t0, before, counter) -> int:
    """--dry after the regression record: the record checked (keys and statuses only), the guard closed, no non-smoke
    file written, no decimal on the console; the record deleted on a pass; seed42_dry.json kept."""
    items = {str(k): s for k, s in order.done}
    ok = {}
    try:
        reg = json.loads(P["reg"].read_text())
        ok["regression_record"] = bool(
            reg.get("rule_sha256") == R5.RULE_SHA and reg.get("dry_run") is True and list(reg.get("items", {}))
            == ["1", "2", "3", "4"] and reg["items"]["1"]["passed"] is True and reg["items"]["4"]["passed"] is True
            and all(reg["items"][k]["passed"] is True or reg["items"][k].get("skipped") is True for k in ("2", "3")))
    except Exception:                                    # noqa: BLE001  a missing or malformed record is a failure
        ok["regression_record"] = False
    ok["items_1_and_4_passed"] = items.get("1") == "pass" and items.get("4") == "pass"
    ok["guard_closed"] = not R5G.is_released()
    ok["no_non_smoke_output"] = _non_smoke_listing() == before
    ok["console_holds_no_decimal_number"] = counter[0] == 0
    for k, v in ok.items():
        print(f"CHECK dry.{k} {'PASS' if v else 'FAIL'}", flush=True)
    passed = all(ok.values())
    if passed:
        P["reg"].unlink()
        print(f"{P['reg'].name} deleted (checked)", flush=True)
    out = P["reg"].parent / DRY_SUMMARY
    R5.write_json_once(out, {"what": "run_r5_seed42.py --dry: items 1 to 4 on seed 42 with the CLIP placement, "
                                     "stopped where the guard would be released; booleans and counts only",
                             "passed": passed, "items": items, "n_comparisons": len(rec.rows),
                             "n_skipped": sum(r["status"] == "skip" for r in rec.rows), "checks": ok,
                             "lines_withheld": int(counter[0])}, True)
    print(f"{out.name} written", flush=True)
    skipped = [k for k, s in order.done if s == "skip"]
    note = f" (items {' and '.join(map(str, skipped))} SKIP: records absent)" if skipped else ""
    print(f"SEED42_DRY {'PASS' if passed else 'FAIL'}{note} [{int(time.time() - t0)}s]", flush=True)
    return 0 if passed else 1


# ---------------------------------------------------------------- the run

def run(dry: bool) -> int:
    if dry:
        with no_decimal_console() as counter:
            return _run(True, counter)
    return _run(False, None)


def _run(dry, counter) -> int:
    t0 = time.time()
    info = check_inputs(dry)                             # the full input check first (T3a-1)
    P = paths(dry)
    if dry:
        for p in (P["reg"], P["reg"].parent / DRY_SUMMARY):
            p.unlink(missing_ok=True)
        before = _non_smoke_listing()
    else:
        R5.refuse_existing(P.values(), False)            # non-smoke outputs are never overwritten
    print("INPUTS PASS" + ("" if all(info.get("own_files", {}).values()) or not dry
                           else " (GoEmotions and GE files not yet set: dry run)"), flush=True)
    rec, order = Recorder(), Order()
    st = {"dry": bool(dry), "taus": tuple(info["taus"])}
    base = {"what": "rule §5 items 1 to 4 on seed 42 (regression checks)" + ("; dry run, not a result" if dry else ""),
            "rule_sha256": R5.RULE_SHA, "dry_run": bool(dry), "inputs_sha256": info.get("inputs"),
            "modules_sha256": info.get("modules"), "own_files_sha256": info.get("own_files"),
            "written_amsterdam": R5.now_ams()}

    def items_rec():
        out = {}
        for k, s in order.done:
            n = sum(r["item"] == k for r in rec.rows)
            out[str(k)] = {"passed": True, "n_comparisons": n} if s == "pass" else \
                {"passed": None, "skipped": True, "n_comparisons": n}
        return out
    try:
        for k, fn in enumerate(ITEMS, 1):
            try:
                fn(rec, st)
            except Exception as e:                       # noqa: BLE001  an exception is a failed comparison
                rec.error(k, e, verbose=not dry)
            order.mark(k, rec.require(k, allow_skip=dry))
    except Stop as e:
        k = int(e.args[0])
        R5.write_json_once(P["reg"], {**base, "items": {**items_rec(), str(k): {"passed": False}},
                                      "all_passed": False, "stopped_at_item": k,
                                      "failed": [r["name"] for r in rec.rows if r["status"] == "fail"],
                                      "n_comparisons": len(rec.rows), "comparisons": rec.rows,
                                      "runtime_s": int(time.time() - t0)}, dry)
        print(f"SEED42{'_DRY' if dry else ''} FAIL at item {k}: stop and report to the user (rule §5, §9); "
              f"{P['reg'].name} written, nothing further", flush=True)
        return 1
    R5.write_json_once(P["reg"], {**base, "items": items_rec(), "all_passed": order.all_passed(),
                                  "stopped_at_item": None, "failed": [], "n_comparisons": len(rec.rows),
                                  "comparisons": rec.rows, "redundancy_D7": st.get("red"),
                                  "runtime_s": int(time.time() - t0)}, dry)
    print(f"REGRESSION {'PASS' if order.all_passed() else 'PASS WITH SKIPS'}: {len(rec.rows)} comparisons, items 1 "
          f"to 4", flush=True)
    print(f"{P['reg'].name} written", flush=True)
    if dry:
        print("GUARD the dry run stops where the guard would be released (rule §8 lapse 5): no GE placement is minted "
              "and no GE-placement result is computed", flush=True)
        return dry_finish(P, rec, order, t0, before, counter)
    order.release(P["reg"])
    print("GUARD released from regression_check.json (items 1 to 4 passed in order)", flush=True)
    rc = development(P, st)
    print(f"SEED42 {'DONE' if rc == 0 else 'STOPPED (boundary)'} [{int(time.time() - t0)}s]", flush=True)
    return rc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--dry", action="store_true", help="items 1 to 4 with the CLIP placement; stops at the guard")
    g.add_argument("--continue-boundary", metavar="SHA256", default=None,
                   help="after a rule §8 boundary was reported to the user: the SHA-256 of boundary_seed42.json")
    g.add_argument("--sensitivity", action="store_true", help="rule §6.1, after the carry and the phase-1 agreement")
    args = ap.parse_args(argv)
    if args.continue_boundary is not None:
        sys.exit(continue_boundary(args.continue_boundary))
    if args.sensitivity:
        sys.exit(sensitivity())
    sys.exit(run(args.dry))


if __name__ == "__main__":
    main()
