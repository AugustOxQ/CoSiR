"""EXPLORATORY (decides nothing; seed 42 development episodes only). PLAN.md in this folder (SHA-256 asserted) fixes
three label-light checks on the existing groupings; nothing is re-clustered:

  a  same-painting ceiling of the affect grouping: p_same, p_diff, R = p_same / p_diff (exact pair counts on
     scorer-train rows), leave-one-out painting-majority accuracy A_loo, majority share, against 20 size-keeping random
     relabellings; read for R0 (E2 affect k-means 64) and L (Leiden default, 41 groups) against the image-head
     held-out accuracy A_head
  b  placeability (P_ami: AMI of argmax image head vs argmax caption head on selection rows; P_lift: same-row over
     different-painting agreement) for the 9 Leiden cells and 8 k-means controls of the sweep (affect heads refit with
     run_told_oracle.fit_one_head, held-out accuracies asserted equal to the stored ones), R0 (stored posteriors) and
     image/caption (references); 8-of-9 rule against the matched k-means
  c  sibling-aware agreement p_img^T S p_txt (applied by replacing every caption posterior q with S q) on the
     told-oracle arm L setting: arms I (identity; must reproduce told-oracle arm L exactly), F1-all (primary), F1-affect,
     F3-all, F2-all (descriptive); label-free AUC gate; told and reader margins on B, paired differences against I

Reuses run_told_oracle (fit_one_head, evaluate_arm, pairs2/dpairs/mix, margin_arrays) and run_sweep (setup) by import;
modifies nothing outside this folder. CPU only.

Run (from /project/CoSiR):
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python src/test/20261115_grouping_step0_checks/run_checks.py {a,b,c} [--smoke]
    -> results/step0{a,b,c}.json, .txt (+ results/step0c_per_anchor.npz); --smoke -> results/smoke/ (overwritable).
    Non-smoke outputs are never overwritten.
"""
import argparse
import json
import math
import sys
import time
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse import csr_matrix
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_mutual_info_score, roc_auc_score

HERE = Path(__file__).resolve().parent
TOLD = HERE.parent / "20261111_community_told_oracle"
SWEEP = HERE.parent / "20261112_community_sweep"
sys.path.insert(0, str(TOLD))
import run_told_oracle as rto  # noqa: E402  (imports diagnose_counterparts, diagnose_fixes, run_checks, run_n6, repo root)
sys.path.insert(0, str(SWEEP))
import run_sweep as rsw  # noqa: E402

df, rc, rg, n6, n6c, dc = rto.df, rto.rc, rto.rg, rto.n6, rto.n6c, rto.dc

from src.data.artelingo import load_artelingo  # noqa: E402
from src.data.artelingo_splits import artelingo_splits  # noqa: E402
from src.eval.aspect_metrics import METRICS, per_anchor  # noqa: E402
from src.eval.aspect_quick_checks import centered_term, crossfit_condition_free  # noqa: E402

PLAN = HERE / "PLAN.md"
PLAN_SHA = "b26d98c5f40e8ff95d79f8031e288f7444a009f6e3e0e992c7e51a2e457e7fac"
SWEEP_RES = SWEEP / "results"
CELLS_DIR = SWEEP_RES / "cells"
SWEEP_JSON = SWEEP_RES / "sweep.json"
TOLD_JSON, TOLD_NPZ = rsw.STORED_JSON, rsw.STORED_NPZ
STORED_N6 = rto.STORED_N6
SWEEP_SCRIPT_SHA = "13314f6f17edc4fe8665e229608300f6ff38a50025ce321823260f3fab0681b1"   # cells/*.json script_sha256

LEIDEN = tuple(rsw.leiden_name(k, r) for k, r in rsw.GRID)
MATCHED = {"leiden_k10_r0.25": 15, "leiden_k10_r1.0": 44, "leiden_k10_r4.0": 118,          # PLAN.md 0b
           "leiden_k20_r0.25": 14, "leiden_k20_r1.0": 41, "leiden_k20_r4.0": 95,
           "leiden_k40_r0.25": 15, "leiden_k40_r1.0": 31, "leiden_k40_r4.0": 88}
KMEANS = tuple(rsw.kmeans_name(n) for n in sorted(set(MATCHED.values())))
CELLS = LEIDEN + KMEANS                                                                  # 9 + 8 = 17
SMOKE_CELLS = ("leiden_k20_r1.0", "kmeans_n41")
A_HEAD_PLAN = {"R0": 13.47, "L": 9.81}
ROOM, NO_ROOM = 1.5, 1.2
N_PERM, SMOKE_N_PERM = 20, 2
ADOPT_AT_LEAST = 8
GATE_PAIRS, GATE_SEED, GATE_TOL = 200_000, 0, 0.005
ARMS = ("I", "F1-all", "F1-affect", "F3-all", "F2-all")
NAMES = ("told", "reader")
DIFF_METRICS = ("r1", "gain", "either")
L_STATED = {"told": (1.64, 1.37, 1.92), "reader": (0.35, 0.15, 0.57)}                     # PLAN.md 0c arm I


# ---------------------------------------------------------------- common

def out_paths(stage, smoke):
    d = HERE / "results" / ("smoke" if smoke else "")
    p = {"json": d / f"step0{stage}.json", "txt": d / f"step0{stage}.txt"}
    if stage == "c":
        p["npz"] = d / "step0c_per_anchor.npz"
    return d, p


def start(stage, smoke):
    if rg.sha_file(PLAN) != PLAN_SHA:
        raise SystemExit("PLAN.md differs from the dispatched version (SHA-256)")
    d, paths = out_paths(stage, smoke)
    if not smoke and any(p.exists() for p in paths.values()):
        raise SystemExit(f"results of stage {stage} exist in {d}; refusing to overwrite")
    d.mkdir(parents=True, exist_ok=True)
    return paths


def header(stage, smoke, inputs):
    return {"exploratory": "decides nothing; seed 42 development episodes only; PLAN.md applied as written",
            "stage": stage, "smoke": bool(smoke), "plan_sha256": PLAN_SHA,
            "provenance": {"script_sha256": rg.sha_file(Path(__file__)),
                           "run_told_oracle_sha256": rg.sha_file(Path(rto.__file__)),
                           "run_sweep_sha256": rg.sha_file(Path(rsw.__file__)),
                           "inputs_sha256": {rc.rel(p): rg.sha_file(p) for p in inputs},
                           "started": now_ams()}}


def write(paths, result, text_fn, npz=None):
    result["provenance"]["finished"] = now_ams()
    result = rto.roundtrip(result)
    rg.assert_finite_tree(result)
    paths["json"].write_text(json.dumps(result, indent=1, ensure_ascii=False))
    if npz is not None:
        np.savez_compressed(paths["npz"], **npz)
    txt = text_fn(result)
    paths["txt"].write_text(txt + "\n")
    print(txt)


def load_cell(name):
    """A sweep cell's stored local partition (scorer-train rows), checked against its JSON's partition_sha256."""
    rec = json.loads((CELLS_DIR / f"{name}.json").read_text())
    if rec["script_sha256"] != SWEEP_SCRIPT_SHA or rec["plan_sha256"] != rsw.PLAN_SHA:
        raise AssertionError(f"{name}: cell written by another run_sweep.py or PLAN.md")
    local = np.asarray(np.load(CELLS_DIR / f"{name}.npz")["partition"], dtype=np.int64)
    if rg.sha_array(local) != rec["partition_sha256"]:
        raise AssertionError(f"{name}: partition SHA-256 differs from its JSON")
    return local, rec


def e2_partitions(groups, scorer_train):
    """E2's local partitions (affect, image, caption), with run_n6's SHA and painting-alignment checks."""
    n6.partition_labels(groups, scorer_train)              # asserts SHA and local_groups == np.unique(...)[1]
    z = np.load(n6.PARTITIONS)
    return {h: np.asarray(z[h], dtype=np.int64) for h in ("affect", "image", "caption")}


def now_ams():
    return datetime.now(ZoneInfo("Europe/Amsterdam")).isoformat(timespec="seconds")


def summary(values):
    v = np.asarray(values, dtype=np.float64)
    return {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max())}


# ---------------------------------------------------------------- stage a: same-painting ceiling

def pair_counts(lab, g):
    """Exact unordered pair counts by unique codes (run_told_oracle.pairs2 / dpairs)."""
    n = len(g)
    within, within_same = rto.pairs2([g], n), rto.pairs2([g, lab], n)
    between, between_same = rto.dpairs([], g), rto.dpairs([lab], g)
    p_same, p_diff = within_same / within, between_same / between
    return {"within_pairs": within, "within_same_group": within_same, "between_pairs": between,
            "between_same_group": between_same, "p_same": p_same, "p_diff": p_diff, "R": p_same / p_diff}


def loo_scores(lab, g):
    """Per-row leave-one-out painting-majority score (rows of paintings with at least 2 rows; NaN elsewhere). For a row
    in group g of painting p the other rows' counts are n_pg - 1 for g and n_pk for k != g; m = their maximum; the row
    scores 1/|{k: count_k = m}| if n_pg - 1 = m, else 0."""
    lab, g = np.asarray(lab, np.int64), np.asarray(g, np.int64)
    n = len(lab)
    _, cell, cnt = np.unique(rto.mix([g, lab]), return_inverse=True, return_counts=True)
    first = np.zeros(len(cnt), dtype=np.int64)
    first[cell] = np.arange(n)
    cell_p = g[first]                                       # painting of each (painting, group) cell
    order = np.lexsort((-cnt, cell_p))                      # per painting, cells by count descending
    sp, sc = cell_p[order], cnt[order]
    P = int(g.max()) + 1
    starts = np.searchsorted(sp, np.arange(P))
    n_cells = np.bincount(cell_p, minlength=P)
    max1 = np.zeros(P, np.int64)
    max2 = np.zeros(P, np.int64)
    has = n_cells > 0
    max1[has] = sc[starts[has]]
    two = n_cells >= 2
    max2[two] = sc[starts[two] + 1]
    vmax = int(cnt.max()) + 1                               # (painting, count value) -> number of cells
    vkeys, vnum = np.unique(cell_p * vmax + cnt, return_counts=True)

    def n_at(p, v):
        key = p * vmax + v
        pos = np.clip(np.searchsorted(vkeys, key), 0, len(vkeys) - 1)
        return np.where(vkeys[pos] == key, vnum[pos], 0)

    p_row, c_row = g, cnt[cell]
    n_max1 = n_at(np.arange(P), max1)
    other_max = np.where((c_row < max1[p_row]) | (n_max1[p_row] >= 2), max1[p_row], max2[p_row])
    m = np.maximum(c_row - 1, other_max)
    hit = c_row - 1 == m
    ties = hit.astype(np.int64) + n_at(p_row, m) - (c_row == m).astype(np.int64)
    score = np.where(hit, 1.0 / np.maximum(ties, 1), 0.0)
    used = np.bincount(g, minlength=P)[g] >= 2
    return np.where(used, score, np.nan)


def loo_bruteforce(lab, g):
    out = np.full(len(lab), np.nan)
    for p in np.unique(g):
        rows = np.flatnonzero(g == p)
        if len(rows) < 2:
            continue
        for r in rows:
            cnt = Counter(lab[rows[rows != r]].tolist())
            m = max(cnt.values())
            tied = [k for k, v in cnt.items() if v == m]
            out[r] = 1.0 / len(tied) if lab[r] in tied else 0.0
    return out


def pairs_bruteforce(lab, g):
    i, j = np.triu_indices(len(lab), k=1)
    same_p, same_c = g[i] == g[j], lab[i] == lab[j]
    return (same_p & same_c).sum() / same_p.sum(), (~same_p & same_c).sum() / (~same_p).sum()


def ceiling_measures(lab, g):
    pc = pair_counts(lab, g)
    s = loo_scores(lab, g)
    used = np.isfinite(s)
    sizes = np.bincount(lab)
    return {**pc, "A_loo_pct": 100 * float(s[used].mean()), "rows_used": int(used.sum()),
            "n_groups": int((sizes > 0).sum()), "majority_share_pct": 100 * float(sizes.max() / sizes.sum()),
            "majority_share_rows_used_pct": 100 * float(np.bincount(lab[used]).max() / used.sum())}


def self_test_a(groupings, g):
    """Vectorised A_loo and pair counts against brute force: all rows of the first 600 paintings (R0, L, one
    permutation), and a synthetic labelling with many ties."""
    sub = g < 600
    gs = np.unique(g[sub], return_inverse=True)[1]
    labs = {"R0": groupings["R0"][sub], "L": groupings["L"][sub],
            "R0_permuted": np.random.default_rng(123).permutation(groupings["R0"])[sub]}
    rng = np.random.default_rng(7)
    syn_g = np.repeat(np.arange(400), rng.integers(1, 9, size=400))
    labs["synthetic_ties"] = rng.integers(0, 3, size=len(syn_g))
    out, ok = {}, True
    for name, lab in labs.items():
        gg = syn_g if name == "synthetic_ties" else gs
        a, b = loo_scores(lab, gg), loo_bruteforce(lab, gg)
        loo_ok = bool(np.array_equal(np.isnan(a), np.isnan(b)) and np.array_equal(a[~np.isnan(a)], b[~np.isnan(b)]))
        pc = pair_counts(lab, gg)
        bs, bd = pairs_bruteforce(lab, gg)
        pairs_ok = bool(np.isclose(pc["p_same"], bs, rtol=1e-12, atol=0) and np.isclose(pc["p_diff"], bd, rtol=1e-12, atol=0))
        out[name] = {"rows": int(len(lab)), "loo_equal": loo_ok, "pairs_equal": pairs_ok}
        ok &= loo_ok and pairs_ok
    return bool(ok), out


def stage_a(smoke):
    paths = start("a", smoke)
    t0 = time.time()
    cells = SMOKE_CELLS if smoke else CELLS
    n_perm = SMOKE_N_PERM if smoke else N_PERM
    inputs = [n6.PARTITIONS, TOLD_NPZ, TOLD_JSON, STORED_N6] + [CELLS_DIR / f"{c}.{e}" for c in cells for e in ("npz", "json")]
    result = header("a", smoke, inputs)
    checks = {}

    data = load_artelingo()
    sp = artelingo_splits(data)
    groups, scorer_train = np.asarray(sp.groups), np.asarray(sp.scorer_train)
    g = np.unique(groups[scorer_train], return_inverse=True)[1].astype(np.int64)
    e2 = e2_partitions(groups, scorer_train)
    checks["E2_partitions_sha_and_painting_alignment"] = True
    z = np.load(TOLD_NPZ)
    groupings = {"R0": e2["affect"], "L": np.asarray(z["partition_L"], np.int64), "image": e2["image"],
                 "caption": e2["caption"]}
    if not np.array_equal(np.asarray(z["partition_R0"], np.int64), e2["affect"]):
        raise AssertionError("told oracle's partition_R0 differs from E2's affect partition")
    checks["told_oracle_partition_R0_equals_E2_affect"] = True
    for c in cells:
        groupings[c], _ = load_cell(c)
    checks["cell_partitions_match_their_sha256"] = True
    if "leiden_k20_r1.0" in groupings:
        checks["partition_L_equals_sweep_leiden_k20_r1.0"] = bool(np.array_equal(groupings["L"], groupings["leiden_k20_r1.0"]))
        if not checks["partition_L_equals_sweep_leiden_k20_r1.0"]:
            raise SystemExit("partition_L differs from the sweep's leiden_k20_r1.0 partition: stopping (PLAN.md)")
    for nm, lab in groupings.items():
        if len(lab) != len(scorer_train) or lab.min() < 0:
            raise AssertionError(f"{nm}: not a full scorer-train labelling")
    a_head = {"R0": json.loads(STORED_N6.read_text())["heads"]["affect"]["heldout_accuracy"]["img"],
              "L": json.loads(TOLD_JSON.read_text())["arms"]["L"]["head"]["heldout_accuracy"]["img"]}
    checks["A_head_equals_PLAN"] = all(round(a_head[k], 2) == A_HEAD_PLAN[k] for k in A_HEAD_PLAN)
    if not checks["A_head_equals_PLAN"]:
        raise AssertionError(f"stored A_head {a_head} differs from PLAN.md {A_HEAD_PLAN}")
    checks["self_test_vs_bruteforce"], self_test = self_test_a(groupings, g)
    if not checks["self_test_vs_bruteforce"]:
        raise AssertionError(f"vectorised A_loo or pair counts differ from brute force: {self_test}")
    rc.log(f"stage a: {len(groupings)} groupings, {len(g)} rows, {int(g.max()) + 1} paintings; self-test passed")

    out = {}
    for nm, lab in groupings.items():
        t = time.time()
        meas = ceiling_measures(lab, g)
        per_seed = []
        for s in range(n_perm):
            perm = np.random.default_rng(s).permutation(lab)
            pm = ceiling_measures(perm, g)
            per_seed.append({k: pm[k] for k in ("p_same", "p_diff", "R", "A_loo_pct")})
        meas["control"] = {"seeds": list(range(n_perm)), "per_seed": per_seed,
                           **{k: summary([x[k] for x in per_seed]) for k in ("p_same", "p_diff", "R", "A_loo_pct")}}
        out[nm] = meas
        rc.log(f"{nm}: p_same {meas['p_same']:.4f} p_diff {meas['p_diff']:.4f} R {meas['R']:.3f} A_loo "
               f"{meas['A_loo_pct']:.2f}% (control R {meas['control']['R']['mean']:.3f}, A_loo "
               f"{meas['control']['A_loo_pct']['mean']:.2f}%) [{time.time() - t:.1f}s]")

    reading = {}
    for arm in ("R0", "L"):
        m = out[arm]
        ratio = m["A_loo_pct"] / a_head[arm]
        room = bool(ratio >= ROOM and m["R"] >= ROOM)
        no_room = bool(ratio < NO_ROOM or m["R"] < NO_ROOM)
        reading[arm] = {"A_loo_pct": m["A_loo_pct"], "A_head_pct": a_head[arm], "A_loo_over_A_head": ratio,
                        "R": m["R"], "label": "Room" if room else "No room" if no_room else "Limited room",
                        "rule": f"Room: A_loo >= {ROOM} x A_head and R >= {ROOM}; No room: A_loo < {NO_ROOM} x A_head "
                                f"or R < {NO_ROOM}; Limited room otherwise"}
    result.update(n_rows=int(len(g)), n_paintings=int(g.max()) + 1, checks=checks, self_test=self_test,
                  A_head=a_head, groupings=out, reading=reading, runtime_s=round(time.time() - t0, 1))
    write(paths, result, text_a)


def text_a(r):
    L = [f"EXPLORATORY step 0a, same-painting ceiling (PLAN.md sha {r['plan_sha256'][:12]}{', SMOKE' if r['smoke'] else ''}); "
         f"scorer-train rows {r['n_rows']}, paintings {r['n_paintings']}; pairs counted exactly; control = "
         f"{len(r['groupings']['R0']['control']['seeds'])} size-keeping random relabellings (mean [min, max]).",
         "Checks: " + "; ".join(f"{k} {v}" for k, v in r["checks"].items()), "",
         f"  {'grouping':18s} {'grp':>4s} {'p_same':>7s} {'p_diff':>7s} {'R':>7s} {'A_loo%':>7s} {'rows':>7s} {'maj%':>6s} | "
         f"control p_same | p_diff | R | A_loo%"]
    for nm, m in r["groupings"].items():
        c = m["control"]
        L.append(f"  {nm:18s} {m['n_groups']:4d} {m['p_same']:7.4f} {m['p_diff']:7.4f} {m['R']:7.3f} {m['A_loo_pct']:7.2f} "
                 f"{m['rows_used']:7d} {m['majority_share_pct']:6.2f} | "
                 f"{c['p_same']['mean']:.4f} [{c['p_same']['min']:.4f}, {c['p_same']['max']:.4f}] | "
                 f"{c['p_diff']['mean']:.4f} [{c['p_diff']['min']:.4f}, {c['p_diff']['max']:.4f}] | "
                 f"{c['R']['mean']:.3f} [{c['R']['min']:.3f}, {c['R']['max']:.3f}] | "
                 f"{c['A_loo_pct']['mean']:.2f} [{c['A_loo_pct']['min']:.2f}, {c['A_loo_pct']['max']:.2f}]")
    L += ["", "Reading (PLAN.md, applied literally)"]
    for arm, x in r["reading"].items():
        L.append(f"  {arm}: {x['label']} | A_loo {x['A_loo_pct']:.2f}% vs A_head {x['A_head_pct']:.2f}% "
                 f"(ratio {x['A_loo_over_A_head']:.3f}); R {x['R']:.3f}")
    L.append(f"runtime {r['runtime_s']:.0f}s")
    return "\n".join(L)


# ---------------------------------------------------------------- stage b: placeability

def placeability(Pi, Pt, g):
    """P_ami = AMI(argmax image head, argmax caption head); P_lift = mean same-row p_i . q_i over the mean over ordered
    pairs from different paintings, exact from per-painting sums."""
    Pi, Pt = np.asarray(Pi, np.float64), np.asarray(Pt, np.float64)
    ai, at = Pi.argmax(axis=1), Pt.argmax(axis=1)
    same = float((Pi * Pt).sum(axis=1).mean())
    inv = np.unique(g, return_inverse=True)[1]
    k = int(inv.max()) + 1
    si = np.stack([np.bincount(inv, weights=Pi[:, c], minlength=k) for c in range(Pi.shape[1])], 1)
    st = np.stack([np.bincount(inv, weights=Pt[:, c], minlength=k) for c in range(Pt.shape[1])], 1)
    n_p = np.bincount(inv, minlength=k).astype(np.float64)
    N = float(len(g))
    diff = float((Pi.sum(axis=0) @ Pt.sum(axis=0) - (si * st).sum()) / (N * N - (n_p ** 2).sum()))
    return {"P_ami": float(adjusted_mutual_info_score(ai, at)), "P_lift": same / diff, "same_row_mean": same,
            "diff_painting_mean": diff, "argmax_agreement_pct": 100 * float((ai == at).mean()), "n_rows": int(len(g))}


def lift_bruteforce(Pi, Pt, g, n=2000, seed=0):
    rows = np.sort(np.random.default_rng(seed).choice(len(g), n, replace=False))
    a, b, gg = np.asarray(Pi, np.float64)[rows], np.asarray(Pt, np.float64)[rows], g[rows]
    M = a @ b.T
    mask = gg[:, None] != gg[None, :]
    brute = float(np.diag(M).mean() / M[mask].mean())
    return brute, placeability(a, b, gg)["P_lift"]


def stage_b(smoke):
    paths = start("b", smoke)
    t0 = time.time()
    cells = SMOKE_CELLS if smoke else CELLS
    rows = n6.SMOKE_HEAD_ROWS if smoke else n6.HEAD_ROWS
    inputs = [df.POSTERIORS, n6.PARTITIONS, SWEEP_JSON, STORED_N6] + [CELLS_DIR / f"{c}.{e}" for c in cells
                                                                      for e in ("npz", "json")]
    result = header("b", smoke, inputs)
    checks = {}
    ctx = rg.EvalContext(df.SEED, False)
    scorer_train = np.asarray(artelingo_splits(ctx.data).scorer_train)
    e2_partitions(ctx.groups, scorer_train)
    checks["E2_partitions_sha_and_painting_alignment"] = True
    if rg.sha_file(df.POSTERIORS) != n6c.POSTERIORS_SHA:
        raise AssertionError("stored posteriors: SHA-256 differs from ADDENDUM_3_N6C.md")
    checks["stored_posteriors_sha256"] = True
    post_stored = n6.load_posteriors(df.POSTERIORS, ctx)
    sel = ctx.selection
    g_sel = ctx.groups[sel]
    sweep = json.loads(SWEEP_JSON.read_text())
    if sweep["script_sha256"] != SWEEP_SCRIPT_SHA:
        raise AssertionError("sweep.json was merged from cells of another run_sweep.py")
    brute, formula = lift_bruteforce(post_stored["affect"]["img"][sel], post_stored["affect"]["txt"][sel], g_sel)
    checks["P_lift_formula_vs_bruteforce_2000_rows"] = {"bruteforce": brute, "formula": formula,
                                                        "equal_rtol_1e-9": bool(np.isclose(brute, formula, rtol=1e-9))}
    if not checks["P_lift_formula_vs_bruteforce_2000_rows"]["equal_rtol_1e-9"]:
        raise AssertionError("P_lift formula differs from brute force")

    out, reproduced = {}, {}
    for nm in cells:
        t = time.time()
        local, rec = load_cell(nm)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            post, prov = rto.fit_one_head(ctx, rto.global_labels(local, scorer_train, len(ctx.groups)), scorer_train, rows)
        prov["convergence_warnings"] = int(sum(issubclass(w.category, ConvergenceWarning) for w in caught))
        same = bool(prov["heldout_accuracy"] == rec["head"]["heldout_accuracy"]
                    and prov["draw_rows_sha256"] == rec["head"]["draw_rows_sha256"]
                    and prov["n_classes"] == rec["head"]["n_classes"])
        reproduced[nm] = same
        if not smoke and not same:
            raise SystemExit(f"{nm}: refit held-out accuracy {prov['heldout_accuracy']} differs from stored "
                             f"{rec['head']['heldout_accuracy']}: stopping")
        kind = "leiden" if nm in LEIDEN else "kmeans"
        told = (sweep["leiden"][nm] if kind == "leiden" else sweep["kmeans"][nm.split("_n")[1]])
        out[nm] = {"kind": kind, "n_groups": rec["groups"]["n_groups"], "head": prov,
                   "stored_heldout_accuracy": rec["head"]["heldout_accuracy"], "head_reproduces_stored": same,
                   **placeability(post["img"][sel], post["txt"][sel], g_sel),
                   "told_margin_r1_sweep": told["eval"]["told"]["fusedT_vs_fusedTcf"]["r1"]}
        rc.log(f"{nm}: {out[nm]['n_groups']} groups, held-out {prov['heldout_accuracy']['img']:.2f}/"
               f"{prov['heldout_accuracy']['txt']:.2f} (stored equal {same}), P_ami {out[nm]['P_ami']:.4f}, "
               f"P_lift {out[nm]['P_lift']:.4f} [{time.time() - t:.0f}s]")
    checks["refit_heldout_equals_stored"] = "skipped (smoke: 3,000-row heads)" if smoke else all(reproduced.values())

    stored_heads = json.loads(STORED_N6.read_text())["heads"]
    refs = {h: {"source": "stored N6 posteriors", "n_groups": 64,
                "heldout_accuracy": stored_heads[h]["heldout_accuracy"],
                **placeability(post_stored[h]["img"][sel], post_stored[h]["txt"][sel], g_sel)}
            for h in ("affect", "image", "caption")}

    pairs, n_ami, n_lift = {}, 0, 0
    for nm, n in MATCHED.items():
        km = rsw.kmeans_name(n)
        if nm not in out or km not in out:
            continue
        sw = sweep["paired_leiden_minus_kmeans"][nm]
        if sw["control_count"] != n or not sw["exact"]:
            raise AssertionError(f"{nm}: sweep.json's matched control is not k-means {n}")
        a, b = out[nm], out[km]
        pairs[nm] = {"kmeans": km, "P_ami_leiden": a["P_ami"], "P_ami_kmeans": b["P_ami"],
                     "ami_leiden_higher": bool(a["P_ami"] > b["P_ami"]), "P_lift_leiden": a["P_lift"],
                     "P_lift_kmeans": b["P_lift"], "lift_leiden_higher": bool(a["P_lift"] > b["P_lift"]),
                     "told_paired_leiden_minus_kmeans_sweep": sw["told"]["r1"]}
        n_ami += pairs[nm]["ami_leiden_higher"]
        n_lift += pairs[nm]["lift_leiden_higher"]
    names = [nm for nm in CELLS if nm in out]
    told = [out[nm]["told_margin_r1_sweep"]["point"] for nm in names]
    corr = {}
    if len(names) >= 3:
        for key in ("P_ami", "P_lift"):
            s = spearmanr([out[nm][key] for nm in names], told)
            corr[key] = {"rho": float(s.statistic), "p_value": float(s.pvalue), "n_cells": len(names)}
    reading = {"pairs_compared": len(pairs), "P_ami_leiden_higher": int(n_ami), "P_lift_leiden_higher": int(n_lift),
               "rule": f"adopt placeability if P_ami(Leiden) > P_ami(matched k-means) in at least {ADOPT_AT_LEAST} of 9",
               "label": ("adopted" if n_ami >= ADOPT_AT_LEAST else "not adopted") if len(pairs) == 9
               else f"not applicable (smoke: {len(pairs)} pair)",
               "P_ami_range_leiden": summary([out[nm]["P_ami"] for nm in LEIDEN if nm in out]),
               "P_ami_range_kmeans": summary([out[nm]["P_ami"] for nm in KMEANS if nm in out])}
    result.update(checks=checks, cells=out, references=refs, pairs=pairs, spearman_vs_told_margin=corr,
                  reading=reading, runtime_s=round(time.time() - t0, 1))
    write(paths, result, text_b)


def text_b(r):
    L = [f"EXPLORATORY step 0b, placeability (PLAN.md sha {r['plan_sha256'][:12]}{', SMOKE' if r['smoke'] else ''}); "
         "selection rows; P_ami = AMI(argmax image head, argmax caption head); P_lift = same-row agreement over "
         "different-painting pairs. Told margins read once from sweep.json (disclosed).",
         "Checks: " + "; ".join(f"{k} {v}" for k, v in r["checks"].items()), "",
         f"  {'cell':18s} {'grp':>4s} {'acc img/txt (stored)':>26s} {'P_ami':>7s} {'P_lift':>7s} {'argmax=':>7s} | told margin (sweep)"]
    for nm, x in r["cells"].items():
        h, s = x["head"]["heldout_accuracy"], x["stored_heldout_accuracy"]
        L.append(f"  {nm:18s} {x['n_groups']:4d} {h['img']:5.2f}/{h['txt']:5.2f} ({s['img']:5.2f}/{s['txt']:5.2f}) "
                 f"{x['P_ami']:7.4f} {x['P_lift']:7.4f} {x['argmax_agreement_pct']:6.2f}% | {rto.c(x['told_margin_r1_sweep'])}")
    L.append("  references (stored N6 posteriors, 64 groups each):")
    for h, x in r["references"].items():
        L.append(f"  {h + (' (R0)' if h == 'affect' else ''):18s} {x['n_groups']:4d} "
                 f"{x['heldout_accuracy']['img']:5.2f}/{x['heldout_accuracy']['txt']:5.2f} {'':15s}"
                 f"{x['P_ami']:7.4f} {x['P_lift']:7.4f} {x['argmax_agreement_pct']:6.2f}%")
    L += ["", "Matched pairs: P_ami Leiden vs k-means (higher?) | P_lift Leiden vs k-means (higher?) | told Leiden - k-means (sweep)"]
    for nm, p in r["pairs"].items():
        L.append(f"  {nm:18s} vs {p['kmeans']:12s} {p['P_ami_leiden']:.4f} vs {p['P_ami_kmeans']:.4f} ({p['ami_leiden_higher']}) | "
                 f"{p['P_lift_leiden']:.4f} vs {p['P_lift_kmeans']:.4f} ({p['lift_leiden_higher']}) | "
                 f"{rto.c(p['told_paired_leiden_minus_kmeans_sweep'])}")
    rd = r["reading"]
    L += ["", f"Reading (PLAN.md): P_ami higher for Leiden in {rd['P_ami_leiden_higher']} of {rd['pairs_compared']} pairs "
              f"-> {rd['label']}; P_lift (descriptive) {rd['P_lift_leiden_higher']} of {rd['pairs_compared']}"]
    for k, v in r["spearman_vs_told_margin"].items():
        L.append(f"  Spearman({k}, told margin) over {v['n_cells']} cells: rho {v['rho']:+.3f} (p {v['p_value']:.3f}; descriptive)")
    L.append(f"runtime {r['runtime_s']:.0f}s")
    return "\n".join(L)


# ---------------------------------------------------------------- stage c: sibling-aware agreement

def fit_one_head_with_check(ctx, lab, scorer_train, n_rows):
    """run_told_oracle.fit_one_head (same draw, same 10,000 check rows, same LogisticRegression(C=1, max_iter=300) on
    unit-normalised CLIP features) that also returns both heads' posteriors on the check rows (float64) and the class
    order."""
    draw = np.random.default_rng(rc.PROBE_SEED).choice(scorer_train, n_rows, replace=False)
    rest = np.setdiff1d(scorer_train, draw)
    check = np.random.default_rng(1).choice(rest, min(n6.CHECK_ROWS, len(rest)), replace=False)
    feats = {"img": ctx.data.img_features, "txt": ctx.data.txt_features}
    sel = ctx.selection
    clfs = {m: LogisticRegression(C=1.0, max_iter=300).fit(rc.unit(F[draw]), lab[draw]) for m, F in feats.items()}
    if not np.array_equal(clfs["img"].classes_, clfs["txt"].classes_):
        raise AssertionError("image and caption heads have different classes")
    post, check_post = {}, {}
    for m, F in feats.items():
        full = np.full((len(ctx.groups), len(clfs[m].classes_)), np.nan, dtype=np.float32)
        full[sel] = clfs[m].predict_proba(rc.unit(F[sel]))
        if not (np.isfinite(full[sel]).all() and np.isnan(full[~ctx.in_sel]).all()):
            raise AssertionError(f"{m}: posteriors must be finite on selection rows and NaN elsewhere")
        post[m] = full
        check_post[m] = clfs[m].predict_proba(rc.unit(F[check])).astype(np.float64)
    counts = np.bincount(lab[check])
    prov = {"n_classes": int(len(clfs["img"].classes_)), "draw_rows_sha256": rg.sha_array(np.sort(draw)),
            "heldout_accuracy": {m: 100 * float(clfs[m].score(rc.unit(F[check]), lab[check])) for m, F in feats.items()},
            "check_majority_share": 100 * float(counts.max() / counts.sum()), "uniform": 100.0 / len(clfs["img"].classes_),
            "check_rows_sha256": rg.sha_array(np.sort(check))}
    return post, prov, check_post, np.asarray(clfs["img"].classes_)


def centred_centroids(lab, X):
    """c_g = mu_g - mu_bar: group means of X (float64) minus the mean of X over all rows."""
    n, K = len(lab), int(lab.max()) + 1
    onehot = csr_matrix((np.ones(n), (lab, np.arange(n))), shape=(K, n))
    sizes = np.bincount(lab, minlength=K).astype(np.float64)
    mu = np.asarray(onehot @ X) / sizes[:, None]
    return mu - X.mean(axis=0)


def s_f1(cent):
    u = cent / np.linalg.norm(cent, axis=1, keepdims=True)
    S = np.maximum(0.0, u @ u.T)
    S = 0.5 * (S + S.T)
    np.fill_diagonal(S, 1.0)
    return S


def s_f3(cent):
    K = len(cent)
    Z = linkage(cent, method="average", metric="cosine")
    cuts = [K, math.ceil(K / 2), math.ceil(K / 4), math.ceil(K / 8)]
    labs = [fcluster(Z, t=t, criterion="maxclust") for t in cuts]
    S = np.mean([(lb[:, None] == lb[None, :]).astype(np.float64) for lb in labs], axis=0)
    return S, {"cuts_requested": cuts, "clusters_obtained": [int(len(np.unique(lb))) for lb in labs]}


def s_f2(ci, ct):
    C = ci.T @ ct
    Cs = 0.5 * (C + C.T)
    d = np.diag(Cs)
    S = np.clip(Cs / np.sqrt(np.outer(d, d)), 0.0, 1.0)
    S = 0.5 * (S + S.T)
    np.fill_diagonal(S, 1.0)
    return S


def s_stats(S):
    off = S[~np.eye(len(S), dtype=bool)]
    return {"K": int(len(S)), "offdiag_mean": float(off.mean()), "offdiag_max": float(off.max()),
            "offdiag_share_positive": float((off > 0).mean()), "row_sum_mean": float(S.sum(axis=1).mean()),
            "symmetric": bool(np.array_equal(S, S.T)), "diag_one": bool(np.all(np.diag(S) == 1.0)),
            "min_eigenvalue": float(np.linalg.eigvalsh(S).min())}


def apply_s(post_h, S, ctx):
    """Replace the caption posterior q of every selection row with S q (S symmetric: q @ S), in float64, stored as
    float32 like every other posterior; rows outside selection stay NaN. S None = identity (arrays untouched)."""
    if S is None:
        return post_h
    if not np.array_equal(S, S.T) or S.shape != (post_h["txt"].shape[1],) * 2:
        raise AssertionError("S must be symmetric and match the head's class count")
    sel = ctx.selection
    txt = np.full_like(post_h["txt"], np.nan)
    txt[sel] = (post_h["txt"][sel].astype(np.float64) @ S).astype(np.float32)
    if not (np.isfinite(txt[sel]).all() and np.isnan(txt[~ctx.in_sel]).all()):
        raise AssertionError("transformed posteriors must be finite on selection rows and NaN elsewhere")
    return {"img": post_h["img"], "txt": txt}


def gate_pairs(g_sel, n_pairs, seed):
    """Ordered pairs of selection rows from different paintings: (i, j) uniform with replacement, same-painting pairs
    rejected, the first n_pairs accepted (batches of n_pairs draws)."""
    rng = np.random.default_rng(seed)
    n, I, J, got = len(g_sel), [], [], 0
    while got < n_pairs:
        i, j = rng.integers(0, n, size=n_pairs), rng.integers(0, n, size=n_pairs)
        keep = g_sel[i] != g_sel[j]
        I.append(i[keep])
        J.append(j[keep])
        got += int(keep.sum())
    return np.concatenate(I)[:n_pairs], np.concatenate(J)[:n_pairs]


def agreement_auc(Pi, Q, gi, gj):
    """AUC of same-row agreement p_i . q_i (all selection rows) against random different-painting pairs p_i . q_j
    (ties count half, sklearn roc_auc_score); and mean same-row over mean random-pair agreement."""
    Pi, Q = np.asarray(Pi, np.float64), np.asarray(Q, np.float64)
    same = (Pi * Q).sum(axis=1)
    rand = (Pi[gi] * Q[gj]).sum(axis=1)
    y = np.r_[np.ones(len(same)), np.zeros(len(rand))]
    return {"auc": float(roc_auc_score(y, np.r_[same, rand])), "same_row_mean": float(same.mean()),
            "random_pair_mean": float(rand.mean()), "ratio": float(same.mean() / rand.mean())}, same, rand


def stage_c(smoke):
    paths = start("c", smoke)
    t0 = time.time()
    rows = n6.SMOKE_HEAD_ROWS if smoke else n6.HEAD_ROWS
    inputs = [TOLD_JSON, TOLD_NPZ, df.POSTERIORS, df.GATE, n6.PARTITIONS, rto.AFFECT_NPZ, rto.AFFECT_JSON, STORED_N6,
              rto.STORED_CF, rto.STORED_FIXES, CELLS_DIR / "leiden_k20_r1.0.npz", CELLS_DIR / "leiden_k20_r1.0.json"]
    result = header("c", smoke, inputs)

    S = rsw.setup()                     # B = C2, R0 and stored arms L/K reproduce exactly (run_sweep.setup asserts)
    ctx, cl, sel = S.ctx, S.cl, S.ctx.selection
    checks = dict(S.checks)
    stored_L = S.stored["arms"]["L"]
    local_cell, _ = load_cell("leiden_k20_r1.0")
    checks["partition_L_equals_sweep_leiden_k20_r1.0"] = bool(np.array_equal(S.partition_L, local_cell))
    if not checks["partition_L_equals_sweep_leiden_k20_r1.0"]:
        raise SystemExit("partition_L differs from the sweep's leiden_k20_r1.0 partition: stopping (PLAN.md)")

    rc.log("A3 condition-free term for B' (information only)")
    inp, _, _ = rc.model_inputs(ctx, "A3", S.scorer_train, False)
    t_n1u = centered_term(inp, ctx.pooled, uniform=True)
    del inp

    # ---------------------------------------------------------- heads
    e2 = e2_partitions(ctx.groups, S.scorer_train)
    checks["E2_partitions_sha_and_painting_alignment"] = True
    local = {"affect": S.partition_L, "image": e2["image"], "caption": e2["caption"]}
    lab = {h: rto.global_labels(local[h], S.scorer_train, len(ctx.groups)) for h in local}
    rc.log(f"affect L heads with fit_one_head ({rows} rows)")
    post_L, prov_L = rto.fit_one_head(ctx, lab["affect"], S.scorer_train, rows)
    if smoke:
        checks["fit_one_head_L_equals_told_oracle_head"] = "skipped (smoke: 3,000-row heads)"
    else:
        checks["fit_one_head_L_equals_told_oracle_head"] = bool(rto.roundtrip(prov_L) == stored_L["head"])
        if not checks["fit_one_head_L_equals_told_oracle_head"]:
            raise SystemExit(f"arm L's refit head {prov_L} differs from told_oracle.json {stored_L['head']}: stopping")
    stored_heads = json.loads(STORED_N6.read_text())["heads"]
    heads, check_post, classes = {}, {}, {}
    for h in ("affect", "image", "caption"):
        rc.log(f"{h}: heads with check-row posteriors ({rows} rows)")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            post_v, prov_v, check_post[h], classes[h] = fit_one_head_with_check(ctx, lab[h], S.scorer_train, rows)
        prov_v["convergence_warnings"] = int(sum(issubclass(w.category, ConvergenceWarning) for w in caught))
        heads[h] = prov_v
        if h == "affect":
            ok = bool(all(np.array_equal(post_v[m], post_L[m], equal_nan=True) for m in ("img", "txt"))
                      and prov_v["heldout_accuracy"] == prov_L["heldout_accuracy"])
            checks["variant_affect_selection_posteriors_equal_fit_one_head"] = ok
            if not ok:
                raise AssertionError("fit_one_head_with_check differs from fit_one_head on L's labels")
        elif smoke:
            checks[f"variant_{h}_selection_posteriors_equal_stored_N6"] = "skipped (smoke: 3,000-row heads)"
        else:
            ok = bool(all(np.array_equal(post_v[m], S.post_stored[h][m], equal_nan=True) for m in ("img", "txt"))
                      and prov_v["heldout_accuracy"] == stored_heads[h]["heldout_accuracy"])
            checks[f"variant_{h}_selection_posteriors_equal_stored_N6"] = ok
            if not ok:
                raise AssertionError(f"{h}: the refit heads differ from the stored N6 posteriors")
        if not smoke and not np.array_equal(classes[h], np.arange(int(local[h].max()) + 1)):
            raise AssertionError(f"{h}: head classes are not 0..K-1")
        del post_v

    # ---------------------------------------------------------- S matrices (scorer-train rows)
    X_img = np.asarray(ctx.data.img_features[S.scorer_train], np.float64)
    X_img /= np.linalg.norm(X_img, axis=1, keepdims=True)
    X_txt = np.asarray(ctx.data.txt_features[S.scorer_train], np.float64)
    X_txt /= np.linalg.norm(X_txt, axis=1, keepdims=True)
    sources = {"affect": np.asarray(S.probs, np.float64), "image": X_img, "caption": X_txt}
    Smat, s_info = {"F1": {}, "F3": {}, "F2": {}}, {}
    for h in ("affect", "image", "caption"):
        cent = centred_centroids(local[h], sources[h])
        Smat["F1"][h] = s_f1(cent)
        Smat["F3"][h], f3_cuts = s_f3(cent)
        f2 = s_f2(check_post[h]["img"], check_post[h]["txt"])
        idx = classes[h]                                     # smoke: a class can be missing from the draw
        Smat["F1"][h], Smat["F3"][h] = Smat["F1"][h][np.ix_(idx, idx)], Smat["F3"][h][np.ix_(idx, idx)]
        Smat["F2"][h] = f2
        s_info[h] = {"source": {"affect": "28 GoEmotions probabilities (raw)", "image": "unit-normalised CLIP image features",
                                "caption": "unit-normalised CLIP text features"}[h],
                     "n_groups": int(local[h].max()) + 1, "F1": s_stats(Smat["F1"][h]),
                     "F3": {**s_stats(Smat["F3"][h]), **f3_cuts}, "F2": s_stats(Smat["F2"][h])}
        rc.log(f"S {h}: F1 offdiag mean {s_info[h]['F1']['offdiag_mean']:.3f}, F3 {s_info[h]['F3']['offdiag_mean']:.3f} "
               f"(clusters {f3_cuts['clusters_obtained']}), F2 {s_info[h]['F2']['offdiag_mean']:.3f}")
    del X_img, X_txt, sources
    arm_S = {"I": {}, "F1-all": dict(Smat["F1"]), "F1-affect": {"affect": Smat["F1"]["affect"]},
             "F3-all": dict(Smat["F3"]), "F2-all": dict(Smat["F2"])}

    base = {"affect": post_L, "image": S.post_stored["image"], "caption": S.post_stored["caption"]}
    eye_ok = all(np.array_equal(apply_s(base[h], np.eye(base[h]["txt"].shape[1]), ctx)["txt"], base[h]["txt"],
                                equal_nan=True) for h in base)
    checks["transform_with_identity_is_bit_identical"] = bool(eye_ok)
    if not eye_ok:
        raise AssertionError("apply_s with the identity changes the posteriors")

    # ---------------------------------------------------------- label-free gate (selection rows)
    g_sel = ctx.groups[sel]
    gi, gj = gate_pairs(g_sel, GATE_PAIRS, GATE_SEED)
    gate_I, same_I = {}, {}
    for h in base:
        gate_I[h], same_I[h], rand = agreement_auc(base[h]["img"][sel], base[h]["txt"][sel], gi, gj)
        if h == "affect":
            u = mannwhitneyu(same_I[h], rand, alternative="two-sided").statistic / (len(same_I[h]) * len(rand))
            checks["auc_equals_mann_whitney_u"] = bool(np.isclose(u, gate_I[h]["auc"], rtol=1e-9, atol=0))

    # ---------------------------------------------------------- arms
    arms, npz, arr_I = {}, {}, None
    for arm in ARMS:
        t = time.time()
        rc.log(f"arm {arm}: S on {sorted(arm_S[arm]) or 'nothing (identity)'}")
        post = {h: apply_s(base[h], arm_S[arm].get(h), ctx) for h in base}
        ev, arr, t6u = rto.evaluate_arm(post, ctx, S.B, S.pB, S.part_idx)
        Bp, picks_Bp = crossfit_condition_free(ctx.cos, t_n1u, t6u, ctx.parity)
        pBp = per_anchor(Bp)
        entry = {"S_on": sorted(arm_S[arm]), "eval": ev,
                 "B_prime": {"describe": df.describe(pBp, cl), "vs_B": df.diff(pBp, S.pB, cl), "picks": picks_Bp}}
        gate = {}
        for h, Sh in arm_S[arm].items():
            g_s, _, _ = agreement_auc(post[h]["img"][sel], post[h]["txt"][sel], gi, gj)
            gate[h] = {"S": g_s, "I": gate_I[h], "auc_change": g_s["auc"] - gate_I[h]["auc"],
                       "passes": bool(g_s["auc"] >= gate_I[h]["auc"] - GATE_TOL)}
        entry["gate"] = {"groupings": gate, "passes": bool(all(x["passes"] for x in gate.values())) if gate else None}
        if arm == "I":
            arr_I = arr
            if smoke:
                checks["arm_I_reproduces_told_oracle_L"] = "skipped (smoke: 3,000-row heads)"
            else:
                ok = {k: rto.roundtrip(ev[k]) == stored_L["eval"][k] for k in ("told", "reader", "pick")}
                ok["arrays"] = all(np.array_equal(np.asarray(arr[nm][j][m]), S.arr_L[nm][j][m])
                                   for nm in NAMES for j in (0, 1) for m in ("r1", "gain", "other"))
                ok["B_prime"] = rto.roundtrip(entry["B_prime"]) == stored_L["B_prime"]
                for nm, (pt, lo, hi) in L_STATED.items():
                    x = ev[nm]["fusedT_vs_fusedTcf"]["r1"]
                    ok[f"{nm}_is_{pt:+.2f}_[{lo:.2f},{hi:.2f}]"] = (round(x["point"], 2), round(x["ci95"][0], 2),
                                                                    round(x["ci95"][1], 2)) == (pt, lo, hi)
                checks["arm_I_reproduces_told_oracle_L"] = ok
                if not all(ok.values()):
                    print(json.dumps(ok, indent=1))
                    raise SystemExit("arm I does not reproduce told-oracle arm L exactly: stopping (PLAN.md)")
                rc.log("arm I reproduces told-oracle arm L exactly (told, reader, pick, arrays, B')")
        else:
            entry["vs_I"] = {nm: {m: rc.point_ci(rto.margin_arrays(arr, nm, m) - rto.margin_arrays(arr_I, nm, m), cl)
                                  for m in DIFF_METRICS} for nm in NAMES}
            entry["vs_I_per_pair_r1"] = {
                nm: {p: rc.point_ci((rto.margin_arrays(arr, nm, "r1") - rto.margin_arrays(arr_I, nm, "r1"))[ctx.pair_index == i],
                                    cl[ctx.pair_index == i]) for i, p in enumerate(rg.POOLED_ORDER)} for nm in NAMES}
        for nm in NAMES:
            for j, part in enumerate(("fused", "cf")):
                for m in METRICS:
                    npz[f"{arm}__{nm}__{part}__{m}"] = np.asarray(arr[nm][j][m])
        for m in METRICS:
            npz[f"{arm}__Bprime__{m}"] = np.asarray(pBp[m])
        entry["runtime_s"] = round(time.time() - t, 1)
        arms[arm] = entry
        e = ev
        rc.log(f"arm {arm}: told {rto.c(e['told']['fusedT_vs_fusedTcf']['r1'])} reader "
               f"{rto.c(e['reader']['fusedT_vs_fusedTcf']['r1'])} pick {e['pick']['correct_share']['point']:.1f} gate "
               f"{entry['gate']['passes']} [{entry['runtime_s']:.0f}s]")
        del post

    # ---------------------------------------------------------- reading (PLAN.md, primary arm F1-all, literally)
    def read(arm):
        a = arms[arm]
        t, r = a["vs_I"]["told"]["r1"], a["vs_I"]["reader"]["r1"]
        gate_ok = a["gate"]["passes"]
        adopt = bool(gate_ok and ((t["ci95"][0] > 0 and r["point"] >= 0) or (r["ci95"][0] > 0 and t["point"] >= 0)))
        reject = bool((not gate_ok) or (t["ci95"][0] <= 0 and r["ci95"][0] <= 0))
        return {"label": "Adopt S" if adopt else "Do not adopt" if reject else "Mixed", "gate_passes": gate_ok,
                "told_minus_I": t, "reader_minus_I": r,
                "rule": "Adopt: gate passes and one paired difference (R@1) has lower bound > 0 with the other's point "
                        ">= 0; Do not adopt: gate fails or both lower bounds <= 0; Mixed otherwise"}
    reading = {"F1-all": read("F1-all"),
               "secondary_same_rule_descriptive": {a: read(a)["label"] for a in ("F1-affect", "F3-all", "F2-all")}}

    npz.update(anchor_group=cl, pair_index=ctx.pair_index, gate_pairs_i=gi, gate_pairs_j=gj,
               partition_L=S.partition_L, **{f"B__{m}": np.asarray(S.pB[m]) for m in METRICS},
               **{f"S__{f}__{h}": Smat[f][h] for f in Smat for h in Smat[f]})
    result.update(n_episodes=int(ctx.n), n_clusters=int(len(np.unique(cl))), checks=checks, B=df.describe(S.pB, cl),
                  heads={"affect_fit_one_head": prov_L, **{f"{h}_with_check": v for h, v in heads.items()}},
                  S=s_info, gate={"n_random_pairs": GATE_PAIRS, "seed": GATE_SEED, "tolerance": GATE_TOL,
                                  "n_same_row": int(len(sel)), "identity": gate_I},
                  arms=arms, reading=reading, runtime_s=round(time.time() - t0, 1))
    write(paths, result, text_c, npz)


def text_c(r):
    c = rto.c
    L = [f"EXPLORATORY step 0c, sibling-aware agreement (PLAN.md sha {r['plan_sha256'][:12]}{', SMOKE' if r['smoke'] else ''}); "
         f"seed 42 dev episodes (n={r['n_episodes']}, {r['n_clusters']} paintings); 95% CIs resample paintings (5,000). "
         "Margin = fused T minus fused T_cf on B (pp).",
         "Checks: " + "; ".join(f"{k} {v}" for k, v in r["checks"].items()),
         f"B (= stored C2): R@1 {c(r['B']['r1'])}", "",
         "1. S per grouping: off-diagonal mean | share > 0 | max | row-sum mean | min eigenvalue"]
    for h, x in r["S"].items():
        for f in ("F1", "F3", "F2"):
            s = x[f]
            extra = f" | F3 cuts {s['cuts_requested']} -> clusters {s['clusters_obtained']}" if f == "F3" else ""
            L.append(f"  {h:8s} {f} (K={s['K']}): {s['offdiag_mean']:.4f} | {s['offdiag_share_positive']:.3f} | "
                     f"{s['offdiag_max']:.3f} | {s['row_sum_mean']:.2f} | {s['min_eigenvalue']:+.3f}{extra}")
    L += ["", f"2. Label-free gate: AUC same-row ({r['gate']['n_same_row']}) vs {r['gate']['n_random_pairs']} random "
              f"different-painting pairs; passes if AUC_S >= AUC_I - {r['gate']['tolerance']}; ratio of means beside it",
          "  identity: " + "; ".join(f"{h} AUC {v['auc']:.4f} ratio {v['ratio']:.3f}" for h, v in r["gate"]["identity"].items())]
    for a, x in r["arms"].items():
        if a == "I":
            continue
        L.append(f"  {a:9s} passes {x['gate']['passes']}: " + "; ".join(
            f"{h} AUC {v['S']['auc']:.4f} ({v['auc_change']:+.4f}) ratio {v['S']['ratio']:.3f}" for h, v in x["gate"]["groupings"].items()))
    L += ["", "3. Told and reader margins on B: R@1 | gain | either; pick accuracy; B' (information only)"]
    for a, x in r["arms"].items():
        e = x["eval"]
        for nm in NAMES:
            m = e[nm]["fusedT_vs_fusedTcf"]
            L.append(f"  {a:9s} {nm:6s} {c(m['r1'])} | {c(m['gain'])} | {c(m['either'])} | per pair R@1 " + " / ".join(
                f"{df.PAIR_SHORT[p]} {v['margin']['r1']['point']:+.2f}" for p, v in e[nm]["per_pair"].items()))
        L.append(f"  {a:9s} pick {e['pick']['correct_share']['point']:.1f} [{e['pick']['correct_share']['ci95'][0]:.1f}, "
                 f"{e['pick']['correct_share']['ci95'][1]:.1f}]; B' R@1 {c(x['B_prime']['describe']['r1'])}, B' - B "
                 f"{c(x['B_prime']['vs_B']['r1'])}")
    L += ["", "4. Paired differences against arm I (per anchor): told R@1 | gain | either ; reader R@1 | gain | either"]
    for a, x in r["arms"].items():
        if a == "I":
            continue
        v = x["vs_I"]
        L.append(f"  {a:9s} told {c(v['told']['r1'])} | {c(v['told']['gain'])} | {c(v['told']['either'])} ; reader "
                 f"{c(v['reader']['r1'])} | {c(v['reader']['gain'])} | {c(v['reader']['either'])}")
    rd = r["reading"]["F1-all"]
    L += ["", f"5. Reading (PLAN.md, primary arm F1-all): {rd['label']} | gate {rd['gate_passes']}; told - I "
              f"{c(rd['told_minus_I'])}; reader - I {c(rd['reader_minus_I'])}",
          "   secondary arms under the same rule (descriptive, not a PLAN reading): " + "; ".join(
              f"{a} {lab}" for a, lab in r["reading"]["secondary_same_rule_descriptive"].items()),
          f"runtime {r['runtime_s']:.0f}s"]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("a", "b", "c"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    {"a": stage_a, "b": stage_b, "c": stage_c}[args.stage](args.smoke)


if __name__ == "__main__":
    main()
