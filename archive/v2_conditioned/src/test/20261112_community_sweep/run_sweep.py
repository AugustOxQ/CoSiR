"""EXPLORATORY (decides nothing; seed 42 development episodes only). PLAN.md in this folder (SHA-256 asserted) fixes the
grid, the measurements and the pick rule:

  Leiden cells  the kNN union graph of src.model.communities.detect_communities (unweighted, simplified; k in
                {10, 20, 40}) on the 28 GoEmotions probabilities per scorer-train row, Leiden with
                RBConfigurationVertexPartition (resolution in {0.25, 1.0, 4.0}, seed 42); communities under 200 rows
                merged as in arm L (run_told_oracle.merge_small)
  k-means       MiniBatchKMeans with E2's settings at every distinct merged group count of the grid

Only the affect partition changes; image and caption posteriors, the told mapping and B (= stored C2) stay as in R0.
Each cell is measured exactly as arms L and K of ../20261111_community_told_oracle/run_told_oracle.py, whose functions
are imported (that folder is not modified): group sizes, affect x emotion pair statistics on the groups and through the
refit heads, held-out affect head accuracy, the told term and N6's reader fused on B against their matched
counterparts (overall and per aspect pair), pick accuracy, and paired per-anchor differences against R0 and arm L.

Subcommands (from /project/CoSiR; prefix CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PYTHONDONTWRITEBYTECODE=1 and use /root/miniconda3/envs/CoSiR/bin/python):
  run_sweep.py run --cells 10:0.25 10:1.0 [...] [--kmeans auto 41 ...] [--out DIR]
      Builds the context, B and R0 once; R0 must reproduce the stored numbers exactly before any cell runs. Then the
      Leiden cells (sorted by k, so each graph is built once per process) and then the k-means controls, writing
      DIR/cells/<cell>.npz and DIR/cells/<cell>.json per cell as it goes (the JSON is written last and marks the cell
      done). Leiden cells refuse to overwrite. A k-means count is skipped when its JSON exists or another live process
      holds its claim file; 'auto' = the merged group counts of this process's own Leiden cells.
  run_sweep.py merge [--out DIR] [--allow-incomplete]
      Collects DIR/cells/ into DIR/sweep.json, DIR/sweep.txt, DIR/per_anchor_sweep.npz (refuses to overwrite).
      Applies PLAN.md's pick rule only when all nine Leiden cells are present, unless --allow-incomplete.
  run_sweep.py probe --cells 40:4.0
      Graph + Leiden + merge only (no heads, no fusion); prints counts, timings and peak RSS; writes nothing.
DIR defaults to src/test/20261112_community_sweep/results.
"""
import argparse
import json
import os
import resource
import sys
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import igraph as ig
import leidenalg
import numpy as np
import psutil
from sklearn.cluster import MiniBatchKMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).resolve().parent
TOLD = HERE.parent / "20261111_community_told_oracle"
sys.path.insert(0, str(TOLD))
import run_told_oracle as rto  # noqa: E402  (imports diagnose_counterparts, diagnose_fixes, run_checks, run_n6, repo root)

df, rc, rg, n6, n6c, dc = rto.df, rto.rc, rto.rg, rto.n6, rto.n6c, rto.dc

from src.data.artelingo_splits import artelingo_aspect_labels, artelingo_splits  # noqa: E402
from src.eval.aspect_metrics import CONDITIONS, METRICS, per_anchor  # noqa: E402
from src.eval.aspect_quick_checks import centered_term, crossfit_condition_free  # noqa: E402

PLAN = HERE / "PLAN.md"
PLAN_SHA = "b477b287fd0dbce4c99ceae251a5d8a502c870f1f50c18483792f72138bdfac1"
STORED_JSON = TOLD / "results" / "told_oracle.json"
STORED_NPZ = TOLD / "results" / "per_anchor_told_oracle.npz"
DEFAULT_OUT = HERE / "results"
KS = (10, 20, 40)                                   # PLAN.md: graph k
RESOLUTIONS = (0.25, 1.0, 4.0)                      # PLAN.md: RBConfiguration resolution
GRID = tuple((k, r) for k in KS for r in RESOLUTIONS)
LEIDEN_SEED = 42
MIN_ROWS = rto.MIN_ROWS                             # 200, as arm L
PICK_WITHIN = 0.05                                  # PLAN.md pick rule (pp of reader margin R@1)
NAMES = ("told", "reader")
DIFF_METRICS = ("r1", "gain", "either")
R0_STATED = {"told": (1.14, 0.90, 1.41), "reader": (0.14, -0.04, 0.32)}   # dispatch brief / previous PLAN.md
REF_KEYS = ("fusedT_vs_B", "fusedTcf_vs_B", "fusedT_vs_fusedTcf", "T_picks", "cf_picks")


def leiden_name(k, res):
    return f"leiden_k{k}_r{res}"


def kmeans_name(n):
    return f"kmeans_n{n}"


def parse_cell(s):
    try:
        k, r = s.split(":")
        k, r = int(k), float(r)
    except ValueError:
        raise SystemExit(f"{s!r}: a cell is written k:resolution, e.g. 20:1.0")
    if (k, r) not in GRID:
        raise SystemExit(f"{s}: not a PLAN.md grid cell (k in {KS}, resolution in {RESOLUTIONS})")
    return k, r


def max_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2      # Linux: kB


def atomic_write(path, write):
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        write(f)
    os.replace(tmp, path)


# ---------------------------------------------------------------- graph and Leiden

def knn_union_graph(X, k):
    """The graph of src.model.communities.detect_communities, line for line: each row picks its k nearest other rows
    (sklearn NearestNeighbors defaults, Euclidean), an undirected unweighted edge per choice, simplify (multiple edges
    and loops). Only the partition call differs (RBConfiguration with a resolution instead of Modularity)."""
    X = np.asarray(X)
    n = len(X)
    neighbors = NearestNeighbors(n_neighbors=min(k, n - 1))
    neighbors.fit(X)
    indices = neighbors.kneighbors(return_distance=False)
    edges = [(i, int(j)) for i, row in enumerate(indices) for j in row]
    graph = ig.Graph(n=n, edges=edges, directed=False)
    graph.simplify(multiple=True, loops=True)
    return graph


def leiden(graph, res):
    p = leidenalg.find_partition(graph, leidenalg.RBConfigurationVertexPartition, resolution_parameter=res,
                                 seed=LEIDEN_SEED)
    labels = np.unique(np.asarray(p.membership, dtype=np.int64), return_inverse=True)[1].astype(np.int64)
    return labels, {"quality": float(p.quality()), "modularity": float(p.modularity)}


class Graphs:
    """One kNN graph per k, kept until a different k is asked for (cells are run sorted by k)."""

    def __init__(self, X):
        self.X, self.k, self.graph, self.build_s = X, None, None, None

    def get(self, k):
        if self.k == k:
            return self.graph, self.build_s, True
        self.graph = None
        t = time.time()
        self.graph = knn_union_graph(self.X, k)
        self.k, self.build_s = k, time.time() - t
        return self.graph, self.build_s, False


def leiden_partition(graphs, probs, k, res):
    graph, graph_s, reused = graphs.get(k)
    t = time.time()
    raw, quality = leiden(graph, res)
    leiden_s = time.time() - t
    t = time.time()
    merged, moves = rto.merge_small(raw, probs, MIN_ROWS)
    merge_s = time.time() - t
    info = {"settings": {"graph": "detect_communities kNN union graph (unweighted, simplified, Euclidean)", "k": k,
                         "partition": "leidenalg.RBConfigurationVertexPartition", "resolution_parameter": res,
                         "seed": LEIDEN_SEED, "merge_under_rows": MIN_ROWS},
            "graph": {"n_vertices": graph.vcount(), "n_edges": graph.ecount(), "build_s": graph_s,
                      "built_for_an_earlier_cell": reused},
            "n_raw": int(raw.max()) + 1, "raw": rto.group_stats(raw), "moves": moves,
            "rows_merged": int(sum(m["rows"] for m in moves)), "leiden_s": leiden_s, "merge_s": merge_s, **quality}
    return raw, merged, info


# ---------------------------------------------------------------- context, B, R0 and L (built once per process)

def load_probs():
    rec = json.loads(rto.AFFECT_JSON.read_text())
    if rg.sha_file(rto.AFFECT_NPZ) != rec["affect_npz_sha256"]:
        raise AssertionError("affect_prepare.npz: SHA-256 differs from affect_prepare.json")
    z = np.load(rto.AFFECT_NPZ)
    probs, affect_local = z["affect_probs"], z["affect_local"].astype(np.int64)
    if rg.sha_array(probs) != rec["affect_probs_sha256"] or list(probs.shape) != list(rec["affect_shape"]):
        raise AssertionError("affect_probs: SHA-256 or shape differs from affect_prepare.json")
    return probs, affect_local, rec


def arrays_from_npz(z, prefix, metrics):
    return {name: ({m: np.asarray(z[f"{prefix}__{name}__fused__{m}"]) for m in metrics},
                   {m: np.asarray(z[f"{prefix}__{name}__cf__{m}"]) for m in metrics}) for name in NAMES}


def setup():
    """run_told_oracle.main's context, B and R0 (stored posteriors), asserted to reproduce the stored numbers exactly;
    arm L's and arm K's stored per-anchor arrays, asserted to reproduce told_oracle.json."""
    t0 = time.time()
    if rg.sha_file(PLAN) != PLAN_SHA:
        raise SystemExit("PLAN.md differs from the dispatched version (SHA-256)")
    stored = json.loads(STORED_JSON.read_text())
    if rg.sha_file(Path(rto.__file__)) != stored["provenance"]["script_sha256"]:
        raise SystemExit("run_told_oracle.py differs from the version that wrote told_oracle.json")
    checks = {}

    ctx = rg.EvalContext(df.SEED, False)
    ep, cl = ctx.pooled, ctx.anchor_group
    scorer_train = np.asarray(artelingo_splits(ctx.data).scorer_train)
    inp, _, _ = rc.model_inputs(ctx, "A3", scorer_train, False)
    t_n1u = centered_term(inp, ep, uniform=True)
    if rg.sha_file(df.POSTERIORS) != n6c.POSTERIORS_SHA:
        raise AssertionError("stored posteriors: SHA-256 differs from ADDENDUM_3_N6C.md")
    post_stored = n6.load_posteriors(df.POSTERIORS, ctx)
    _, _, t6u_stored, _, _ = n6.n6_terms(post_stored, ep)
    B, _ = crossfit_condition_free(ctx.cos, t_n1u, t6u_stored, ctx.parity)
    pB = per_anchor(B)
    gate = np.load(df.GATE)
    if not (np.array_equal(gate["anchor_group"], cl) and np.array_equal(gate["pair_index"], ctx.pair_index)
            and all(np.array_equal(pB[m], gate[f"matched__{m}"]) for m in METRICS)):
        raise SystemExit("per_anchor(B) differs from the stored C2 (matched__*) arrays: stopping")
    checks["B_equals_stored_C2"] = True
    del inp, t_n1u, t6u_stored
    rc.log(f"B reproduces C2 exactly ({ctx.n} episodes)")

    aspect_of = {"a": np.array([rg.PAIRS[i][0] for i in ctx.pair_index]),
                 "b": np.array([rg.PAIRS[i][1] for i in ctx.pair_index])}
    part_idx = {cnd: np.array([df.PARTS.index(df.TOLD_PART[a]) for a in aspect_of[cnd]]) for cnd in CONDITIONS}

    lab_all = artelingo_aspect_labels(ctx.data)
    lab_st = {a: lab_all[a][scorer_train] for a in ("emotion", "style", "genre")}
    lab_sel = {a: lab_all[a][ctx.selection] for a in ("emotion", "style", "genre")}
    g_st = np.unique(ctx.groups[scorer_train], return_inverse=True)[1]
    g_sel = ctx.groups[ctx.selection]
    probs, affect_local, affect_rec = load_probs()
    if probs.shape != (len(scorer_train), 28):
        raise AssertionError("affect_probs rows differ from scorer_train")
    e2_affect = np.load(n6.PARTITIONS)["affect"]
    if rg.sha_file(n6.PARTITIONS) != n6.PARTITIONS_SHA or not np.array_equal(e2_affect, affect_local):
        raise AssertionError("affect_local differs from E2's affect partition")
    n6.partition_labels(ctx.groups, scorer_train)
    checks["affect_local_equals_E2_affect"] = True
    km_settings = dict(affect_rec["kmeans_settings"])
    if not np.array_equal(MiniBatchKMeans(**km_settings).fit_predict(probs).astype(np.int64), affect_local):
        raise AssertionError("E2's k-means settings no longer reproduce E2's affect partition")
    checks["E2_kmeans_settings_reproduce_E2_affect"] = True

    # ---------------------------------------------------------- R0 (stored posteriors): must reproduce exactly
    rc.log("R0 (stored posteriors) on B")
    ev_r0, arr_r0, _ = rto.evaluate_arm(post_stored, ctx, B, pB, part_idx)
    stored_cf = json.loads(rto.STORED_CF.read_text())["matched"]
    stored_fix = json.loads(rto.STORED_FIXES.read_text())["reader"]
    z = np.load(STORED_NPZ)
    checks["R0_told_equals_diagnose_counterparts"] = all(
        rto.roundtrip(ev_r0["told"][k]) == stored_cf["T6oracle"][k] for k in REF_KEYS)
    checks["R0_reader_equals_diagnose_counterparts"] = all(
        rto.roundtrip(ev_r0["reader"][k]) == stored_cf["T6"][k] for k in REF_KEYS)
    checks["R0_pick_equals_diagnose_fixes"] = bool(
        rto.roundtrip(ev_r0["pick"]["correct_share"]) == stored_fix["correct_share"]
        and ev_r0["pick"]["both_correct_share"] == stored_fix["both_correct_share"])
    checks["R0_eval_equals_told_oracle_json"] = rto.roundtrip(ev_r0) == stored["arms"]["R0"]["eval"]
    ref_r0 = arrays_from_npz(z, "R0", ("r1", "gain", "other"))
    checks["R0_arrays_equal_stored_npz"] = all(
        np.array_equal(np.asarray(arr_r0[name][j][m]), ref_r0[name][j][m])
        for name in NAMES for j in (0, 1) for m in ("r1", "gain", "other"))
    stated = {}
    for name, (pt, lo, hi) in R0_STATED.items():
        x = ev_r0[name]["fusedT_vs_fusedTcf"]["r1"]
        stated[name] = (round(x["point"], 2), round(x["ci95"][0], 2), round(x["ci95"][1], 2))
        checks[f"R0_{name}_margin_is_{pt:+.2f}_[{lo:+.2f},{hi:+.2f}]"] = stated[name] == (pt, lo, hi)
    if not all(v is True for v in checks.values()):
        print(json.dumps({"checks": checks, "stated": stated}, indent=1))
        raise SystemExit("R0 does not reproduce the stored told/reader margins: stopping (no cell is run)")
    rc.log(f"R0 reproduces the stored numbers exactly: told margin {stated['told']}, reader margin {stated['reader']}")

    # ---------------------------------------------------------- stored arm L and K arrays (paired references)
    if not (np.array_equal(z["anchor_group"], cl) and np.array_equal(z["pair_index"], ctx.pair_index)):
        raise AssertionError("per_anchor_told_oracle.npz: anchor_group or pair_index differ from this context")
    refs = {}
    for arm in ("L", "K"):
        arr = arrays_from_npz(z, arm, ("r1", "gain", "other"))
        ok = all(rto.roundtrip(df.diff(arr[name][0], arr[name][1], cl))
                 == stored["arms"][arm]["eval"][name]["fusedT_vs_fusedTcf"] for name in NAMES)
        ok &= all(rto.roundtrip(rc.point_ci(rto.margin_arrays(arr, name, m) - rto.margin_arrays(arr_r0, name, m), cl))
                  == stored["arms"][arm]["vs_R0"][name][m] for name in NAMES for m in DIFF_METRICS)
        checks[f"stored_{arm}_arrays_reproduce_told_oracle_json"] = bool(ok)
        if not ok:
            raise AssertionError(f"stored arm {arm} arrays do not reproduce told_oracle.json")
        refs[arm] = arr
    rc.log(f"setup done in {time.time() - t0:.0f}s; max RSS {max_rss_gb():.2f} GB")
    return SimpleNamespace(
        ctx=ctx, cl=cl, B=B, pB=pB, part_idx=part_idx, scorer_train=scorer_train, lab_st=lab_st, lab_sel=lab_sel,
        g_st=g_st, g_sel=g_sel, probs=probs, affect_local=affect_local, km_settings=km_settings,
        post_stored=post_stored, ev_r0=ev_r0, arr_r0=arr_r0, arr_L=refs["L"], arr_K=refs["K"],
        partition_L=np.asarray(z["partition_L"], dtype=np.int64),
        partition_L_raw=np.asarray(z["partition_L_raw"], dtype=np.int64),
        partition_K=np.asarray(z["partition_K"], dtype=np.int64), stored=stored, checks=checks,
        setup_s=time.time() - t0)


# ---------------------------------------------------------------- one cell

def measure(S, local):
    """Arms L/K of run_told_oracle.main for one affect partition (scorer-train local labels)."""
    ctx, sel, out, times = S.ctx, S.ctx.selection, {}, {}
    t = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        post_aff, prov = rto.fit_one_head(ctx, rto.global_labels(local, S.scorer_train, len(ctx.groups)),
                                          S.scorer_train, n6.HEAD_ROWS)
    times["heads_s"] = time.time() - t
    prov["convergence_warnings"] = int(sum(issubclass(w.category, ConvergenceWarning) for w in caught))
    t = time.time()
    out["pairs"] = {"groups": rto.pair_stats_groups(local, S.lab_st, S.g_st),
                    "heads": rto.pair_stats_heads(post_aff["img"][sel].astype(np.float64),
                                                  post_aff["txt"][sel].astype(np.float64), S.lab_sel, S.g_sel)}
    times["pairs_s"] = time.time() - t
    t = time.time()
    post = {"affect": post_aff, "image": S.post_stored["image"], "caption": S.post_stored["caption"]}
    ev, arr, _ = rto.evaluate_arm(post, ctx, S.B, S.pB, S.part_idx)
    times["eval_s"] = time.time() - t
    out["head"], out["eval"] = prov, ev
    for ref, ref_arr in (("R0", S.arr_r0), ("L", S.arr_L)):
        out[f"vs_{ref}"] = {name: {m: rc.point_ci(rto.margin_arrays(arr, name, m) - rto.margin_arrays(ref_arr, name, m),
                                                  S.cl) for m in DIFF_METRICS} for name in NAMES}
    out["partition_vs"] = {"equal_partition_L": bool(np.array_equal(local, S.partition_L)),
                           "ari_vs_L": float(adjusted_rand_score(S.partition_L, local)),
                           "ari_vs_R0": float(adjusted_rand_score(S.affect_local, local))}
    return out, arr, times


def cell_arrays(arr, S, local, raw=None):
    a = {"anchor_group": S.cl, "pair_index": S.ctx.pair_index, "partition": local}
    if raw is not None:
        a["partition_raw"] = raw
    for name in NAMES:
        for j, part in enumerate(("fused", "cf")):
            for m in METRICS:
                a[f"{name}__{part}__{m}"] = np.asarray(arr[name][j][m])
    return a


def finish_cell(S, out_dir, name, record, arrays, t0):
    record["runtime"]["cell_total_s"] = time.time() - t0
    record["max_rss_gb_so_far"] = max_rss_gb()
    record = rto.roundtrip(record)
    rg.assert_finite_tree(record)
    atomic_write(out_dir / f"{name}.npz", lambda f: np.savez_compressed(f, **arrays))
    atomic_write(out_dir / f"{name}.json",
                 lambda f: f.write(json.dumps(record, indent=1, ensure_ascii=False).encode()))
    e = record["eval"]
    rc.log(f"{name}: {record['groups']['n_groups']} groups | told margin "
           f"{rto.c(e['told']['fusedT_vs_fusedTcf']['r1'])} | reader margin "
           f"{rto.c(e['reader']['fusedT_vs_fusedTcf']['r1'])} | pick {e['pick']['correct_share']['point']:.1f} | "
           f"held-out {record['head']['heldout_accuracy']['img']:.1f}/{record['head']['heldout_accuracy']['txt']:.1f} | "
           f"{record['runtime']['cell_total_s']:.0f}s | max RSS {record['max_rss_gb_so_far']:.2f} GB")


def header(kind, name):
    return {"exploratory": "decides nothing; seed 42 development episodes only; PLAN.md applied as written",
            "plan_sha256": PLAN_SHA, "script_sha256": rg.sha_file(Path(__file__)), "cell": name, "kind": kind,
            "pid": os.getpid(), "host_cpu_count": os.cpu_count(),
            "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES")}}


def run_leiden_cell(S, graphs, out_dir, k, res):
    t0 = time.time()
    name = leiden_name(k, res)
    rc.log(f"{name}: graph + Leiden")
    raw, local, info = leiden_partition(graphs, S.probs, k, res)
    rc.log(f"{name}: {info['n_raw']} raw communities -> {int(local.max()) + 1} after merging (graph "
           f"{info['graph']['build_s']:.0f}s{' reused' if info['graph']['built_for_an_earlier_cell'] else ''}, "
           f"Leiden {info['leiden_s']:.0f}s)")
    meas, arr, times = measure(S, local)
    meas["partition_vs"]["raw_equal_partition_L_raw"] = bool(np.array_equal(raw, S.partition_L_raw))
    record = {**header("leiden", name), "leiden": info, "groups": rto.group_stats(local),
              "partition_sha256": rg.sha_array(local), **meas,
              "runtime": {"graph_build_s": info["graph"]["build_s"],
                          "graph_built_in_this_cell": not info["graph"]["built_for_an_earlier_cell"],
                          "leiden_s": info["leiden_s"], "merge_s": info["merge_s"], **times}}
    finish_cell(S, out_dir, name, record, cell_arrays(arr, S, local, raw), t0)
    return int(local.max()) + 1


def claim(path):
    """Exclusive claim file holding the owner's pid; a claim whose pid is gone is taken over."""
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                pid = int(path.read_text().strip())
            except (OSError, ValueError):
                pid = None
            if pid is not None and pid != os.getpid() and psutil.pid_exists(pid):
                return False
            path.unlink(missing_ok=True)
            continue
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    return False


def run_kmeans_cell(S, out_dir, n):
    name = kmeans_name(n)
    if (out_dir / f"{name}.json").exists():
        rc.log(f"{name}: result exists, skipped")
        return
    claim_path = out_dir / f"{name}.claim"
    if not claim(claim_path):
        rc.log(f"{name}: claimed by another live process ({claim_path.read_text().strip()}), skipped")
        return
    try:
        if (out_dir / f"{name}.json").exists():                 # finished between the check above and the claim
            rc.log(f"{name}: result exists, skipped")
            return
        t0 = time.time()
        settings = {**S.km_settings, "n_clusters": int(n)}
        t = time.time()
        lab = MiniBatchKMeans(**settings).fit_predict(S.probs).astype(np.int64)
        kmeans_s = time.time() - t
        used, local = np.unique(lab, return_inverse=True)
        local = local.astype(np.int64)
        info = {"settings": settings, "runtime_s": kmeans_s, "n_empty_clusters": int(n - len(used)),
                "merged": False, "equals_stored_partition_K_41": bool(np.array_equal(local, S.partition_K)),
                "equals_R0_partition_64": bool(np.array_equal(local, S.affect_local))}
        rc.log(f"{name}: MiniBatchKMeans {settings} ({kmeans_s:.1f}s); {len(used)} non-empty clusters")
        meas, arr, times = measure(S, local)
        if info["equals_stored_partition_K_41"]:
            info["arrays_equal_stored_K"] = all(np.array_equal(np.asarray(arr[nm][j][m]), S.arr_K[nm][j][m])
                                                for nm in NAMES for j in (0, 1) for m in ("r1", "gain", "other"))
        record = {**header("kmeans", name), "kmeans": info, "groups": rto.group_stats(local),
                  "partition_sha256": rg.sha_array(local), **meas, "runtime": {"kmeans_s": kmeans_s, **times}}
        finish_cell(S, out_dir, name, record, cell_arrays(arr, S, local), t0)
    finally:
        claim_path.unlink(missing_ok=True)


def cmd_run(args):
    out_dir = Path(args.out) / "cells"
    cells = sorted({parse_cell(s) for s in (args.cells or [])})
    auto, counts = False, []
    for tok in args.kmeans or []:
        if tok == "auto":
            auto = True
        elif tok.isdigit() and int(tok) >= 2:
            counts.append(int(tok))
        else:
            raise SystemExit(f"--kmeans {tok!r}: a group count (>= 2) or 'auto'")
    if not cells and not counts:
        raise SystemExit("nothing to run: give --cells and/or --kmeans counts")
    if auto and not cells:
        raise SystemExit("--kmeans auto needs --cells (it uses their merged group counts)")
    taken = [leiden_name(k, r) for k, r in cells
             if (out_dir / f"{leiden_name(k, r)}.json").exists() or (out_dir / f"{leiden_name(k, r)}.npz").exists()]
    if taken:
        raise SystemExit(f"results exist for {taken} in {out_dir}; refusing to overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rc.log(f"pid {os.getpid()}: Leiden cells {[leiden_name(k, r) for k, r in cells]}; k-means "
           f"{'auto + ' if auto else ''}{counts}; out {out_dir}")
    S = setup()
    graphs = Graphs(S.probs)
    for k, r in cells:
        n = run_leiden_cell(S, graphs, out_dir, k, r)
        if auto:
            counts.append(n)
    graphs.graph = None
    for n in dict.fromkeys(counts):
        run_kmeans_cell(S, out_dir, n)
    rc.log(f"pid {os.getpid()} done in {time.time() - t0:.0f}s (setup {S.setup_s:.0f}s); max RSS {max_rss_gb():.2f} GB")


def cmd_probe(args):
    probs, _, _ = load_probs()
    graphs = Graphs(probs)
    for k, r in sorted({parse_cell(s) for s in args.cells}):
        raw, local, info = leiden_partition(graphs, probs, k, r)
        print(json.dumps({"cell": leiden_name(k, r), "n_raw": info["n_raw"], "n_merged": int(local.max()) + 1,
                          "rows_merged": info["rows_merged"], "n_edges": info["graph"]["n_edges"],
                          "graph_build_s": round(info["graph"]["build_s"], 1),
                          "graph_reused": info["graph"]["built_for_an_earlier_cell"],
                          "leiden_s": round(info["leiden_s"], 1), "merge_s": round(info["merge_s"], 2),
                          "max_rss_gb": round(max_rss_gb(), 2)}), flush=True)


# ---------------------------------------------------------------- merge

def load_cell_arrays(path):
    z = np.load(path)
    arr = {name: tuple({m: np.asarray(z[f"{name}__{part}__{m}"]) for m in METRICS} for part in ("fused", "cf"))
           for name in NAMES}
    extra = {k: np.asarray(z[k]) for k in z.files if k.startswith("partition") or k in ("anchor_group", "pair_index")}
    return arr, extra


def margin(rec, name, m="r1"):
    return rec["eval"][name]["fusedT_vs_fusedTcf"][m]


def nearest_control(n, controls):
    """Exact count if present, else the nearest count (ties to the smaller count), flagged."""
    if n in controls:
        return n, True
    if not controls:
        return None, False
    return min(controls, key=lambda c: (abs(c - n), c)), False


def apply_pick_rule(leiden):
    """PLAN.md: largest reader margin (R@1 point, full precision); every cell within 0.05 of that largest margin is a
    candidate, and the candidate with the fewest merged groups is carried (equal counts: the larger reader margin)."""
    reader = {nm: margin(rec, "reader")["point"] for nm, rec in leiden.items()}
    best = max(reader.values())
    top = max(reader, key=reader.get)
    cands = sorted((nm for nm in reader if reader[nm] >= best - PICK_WITHIN),
                   key=lambda nm: (leiden[nm]["groups"]["n_groups"], -reader[nm]))
    return {"rule": "largest label-free reader margin over its counterpart (R@1 point, full precision); cells within "
                    f"{PICK_WITHIN} pp of the largest are candidates and the one with the fewest merged groups is "
                    "carried (equal counts: larger reader margin). The told margin does not pick.",
            "largest_reader_margin_cell": top, "largest_reader_margin": best,
            "candidates_within_0.05": {nm: {"reader_margin": reader[nm], "n_groups": leiden[nm]["groups"]["n_groups"]}
                                       for nm in cands},
            "pick": cands[0]}


def fmt_ci(x, w=6):
    return rto.c(x, w)


def fmt_pct(x):
    return f"{x['point']:.1f} [{x['ci95'][0]:.1f}, {x['ci95'][1]:.1f}]"


def merge_text(r):
    L = [f"EXPLORATORY Leiden affect-community sweep (PLAN.md sha {PLAN_SHA[:12]}); decides nothing; seed 42 dev "
         f"episodes only (n={r['n_episodes']}, {r['n_clusters']} paintings); 95% CIs resample paintings (5,000). "
         "Margin = fused T minus fused T_cf on B (R@1, pp).",
         f"Cells present: {len(r['leiden'])}/9 Leiden, k-means counts {sorted(r['kmeans_counts'])}; missing Leiden "
         f"{r['missing_leiden'] or 'none'}; missing k-means counts {r['missing_kmeans_counts'] or 'none'}",
         f"References (told_oracle.json): R0 k-means 64 told {fmt_ci(r['reference']['R0']['told'])} reader "
         f"{fmt_ci(r['reference']['R0']['reader'])}; L (detect_communities defaults, 41 groups) told "
         f"{fmt_ci(r['reference']['L']['told'])} reader {fmt_ci(r['reference']['L']['reader'])}; K (k-means 41) told "
         f"{fmt_ci(r['reference']['K']['told'])} reader {fmt_ci(r['reference']['K']['reader'])}", ""]
    cols = "".join(f"{'res ' + str(res):>26s}" for res in RESOLUTIONS)
    for title, fn in (("1. Merged group count (raw Leiden count)",
                       lambda rec: f"{rec['groups']['n_groups']} ({rec['leiden']['n_raw']})"),
                      ("2. Told margin R@1", lambda rec: fmt_ci(margin(rec, "told"))),
                      ("3. Reader margin R@1", lambda rec: fmt_ci(margin(rec, "reader"))),
                      ("4. Reader pick accuracy (%)", lambda rec: fmt_pct(rec["eval"]["pick"]["correct_share"]))):
        L += [title, f"  {'':6s}{cols}"]
        for k in KS:
            row = "".join(f"{(fn(r['leiden'][leiden_name(k, res)]) if leiden_name(k, res) in r['leiden'] else '-'):>26s}"
                          for res in RESOLUTIONS)
            L.append(f"  k={k:<4d}{row}")
        L.append("")
    L += ["5. Leiden cells in detail",
          f"  {'cell':18s} {'grp':>4s} {'min':>5s} {'max':>6s} {'effN':>6s} {'lift g/h':>11s} {'acc img/txt':>11s} "
          f"| per pair told e×s / e×g / s×g | per pair reader e×s / e×g / s×g | conv | time"]
    for nm, rec in r["leiden"].items():
        g, p = rec["groups"], rec["pairs"]
        pt = " / ".join(f"{v['margin']['r1']['point']:+.2f}" for v in rec["eval"]["told"]["per_pair"].values())
        pr = " / ".join(f"{v['margin']['r1']['point']:+.2f}" for v in rec["eval"]["reader"]["per_pair"].values())
        L.append(f"  {nm:18s} {g['n_groups']:4d} {g['min']:5d} {g['max']:6d} {g['effective_n']:6.1f} "
                 f"{p['groups']['lift']['lift']:5.2f}/{p['heads']['by_aspect']['ratio_same_over_diff']:5.3f} "
                 f"{rec['head']['heldout_accuracy']['img']:5.1f}/{rec['head']['heldout_accuracy']['txt']:5.1f} "
                 f"| {pt} | {pr} | {rec['head']['convergence_warnings']} | {rec['runtime']['cell_total_s']:.0f}s")
    L += ["", "6. Paired differences per cell (R@1): told vs R0 | told vs L | reader vs R0 | reader vs L"]
    for nm, rec in r["leiden"].items():
        L.append(f"  {nm:18s} {fmt_ci(rec['vs_R0']['told']['r1'])} | {fmt_ci(rec['vs_L']['told']['r1'])} | "
                 f"{fmt_ci(rec['vs_R0']['reader']['r1'])} | {fmt_ci(rec['vs_L']['reader']['r1'])}")
    L += ["", "7. k-means controls (E2's settings) by count: groups | told margin | reader margin | pick | acc img/txt"]
    for n in sorted(r["kmeans"]):
        rec = r["kmeans"][n]
        L.append(f"  n={n:<4d} {rec['groups']['n_groups']:4d} | {fmt_ci(margin(rec, 'told'))} | "
                 f"{fmt_ci(margin(rec, 'reader'))} | {rec['eval']['pick']['correct_share']['point']:.1f} | "
                 f"{rec['head']['heldout_accuracy']['img']:.1f}/{rec['head']['heldout_accuracy']['txt']:.1f}"
                 + (" (equals stored K)" if rec["kmeans"].get("arrays_equal_stored_K") else ""))
    L += ["", "8. Paired Leiden minus k-means at the matched count (R@1): told | reader  [control count; exact?]"]
    for nm, pr in r["paired_leiden_minus_kmeans"].items():
        if pr is None:
            L.append(f"  {nm:18s} no k-means control available")
            continue
        flag = "exact" if pr["exact"] else f"NEAREST (Leiden count {pr['leiden_count']} has no control)"
        L.append(f"  {nm:18s} {fmt_ci(pr['told']['r1'])} | {fmt_ci(pr['reader']['r1'])}  [n={pr['control_count']}; {flag}]")
    c20 = r["leiden"].get(leiden_name(20, 1.0))
    if c20:
        pv = c20["partition_vs"]
        L += ["", f"9. k=20, resolution 1.0 (RBConfiguration) against stored arm L (Modularity): partition equal "
                  f"{pv['equal_partition_L']} (raw equal {pv['raw_equal_partition_L_raw']}), ARI {pv['ari_vs_L']:.6f}; "
                  f"told {fmt_ci(margin(c20, 'told'))} vs L {fmt_ci(r['reference']['L']['told'])}; reader "
                  f"{fmt_ci(margin(c20, 'reader'))} vs L {fmt_ci(r['reference']['L']['reader'])}"]
    pk = r["pick"]
    L += ["", "10. Pick (PLAN.md rule, applied literally)"]
    if pk.get("withheld"):
        L.append(f"  WITHHELD: {pk['withheld']}")
    if "pick" in pk:
        rec = r["leiden"][pk["pick"]]
        L.append(f"  candidates within {PICK_WITHIN} of the largest reader margin ({pk['largest_reader_margin_cell']}, "
                 f"{pk['largest_reader_margin']:+.3f}): "
                 + "; ".join(f"{nm} {v['reader_margin']:+.3f} ({v['n_groups']} groups)"
                             for nm, v in pk["candidates_within_0.05"].items()))
        L.append(f"  {'PICK (present cells only)' if pk.get('withheld') else 'PICK'} {pk['pick']}: "
                 f"{rec['groups']['n_groups']} groups | reader margin "
                 f"{fmt_ci(margin(rec, 'reader'))} | told margin {fmt_ci(margin(rec, 'told'))} | pick accuracy "
                 f"{rec['eval']['pick']['correct_share']['point']:.1f}")
        L.append(f"  arm L     : 41 groups | reader margin {fmt_ci(r['reference']['L']['reader'])} | told margin "
                 f"{fmt_ci(r['reference']['L']['told'])} | pick accuracy {r['reference']['L']['pick']:.1f}")
        L.append(f"  pick minus L, paired: reader {fmt_ci(rec['vs_L']['reader']['r1'])} | told "
                 f"{fmt_ci(rec['vs_L']['told']['r1'])}")
    return "\n".join(L)


def cmd_merge(args):
    out = Path(args.out)
    cells_dir = out / "cells"
    paths = (out / "sweep.json", out / "sweep.txt", out / "per_anchor_sweep.npz")
    if any(p.exists() for p in paths):
        raise SystemExit(f"merge outputs exist in {out}; refusing to overwrite")
    if rg.sha_file(PLAN) != PLAN_SHA:
        raise SystemExit("PLAN.md differs from the dispatched version (SHA-256)")
    stored = json.loads(STORED_JSON.read_text())
    recs = {p.stem: json.loads(p.read_text()) for p in sorted(cells_dir.glob("*.json"))}
    if not recs:
        raise SystemExit(f"no cell results in {cells_dir}")
    shas = {rec["script_sha256"] for rec in recs.values()}
    if len(shas) != 1 or {rec["plan_sha256"] for rec in recs.values()} != {PLAN_SHA}:
        raise SystemExit(f"cells were written by different script versions {shas} or another PLAN.md")
    leiden = {leiden_name(k, r): recs[leiden_name(k, r)] for k, r in GRID if leiden_name(k, r) in recs}
    kmeans = {rec["kmeans"]["settings"]["n_clusters"]: rec for nm, rec in recs.items() if rec["kind"] == "kmeans"}
    missing_leiden = [leiden_name(k, r) for k, r in GRID if leiden_name(k, r) not in recs]
    leiden_counts = sorted({rec["groups"]["n_groups"] for rec in leiden.values()})
    missing_km = [n for n in leiden_counts if n not in kmeans]

    arrays, npz_out, cl, pair_index = {}, {}, None, None
    for nm in list(leiden) + [kmeans_name(n) for n in kmeans]:
        arr, extra = load_cell_arrays(cells_dir / f"{nm}.npz")
        if cl is None:
            cl, pair_index = extra["anchor_group"], extra["pair_index"]
        elif not (np.array_equal(cl, extra["anchor_group"]) and np.array_equal(pair_index, extra["pair_index"])):
            raise SystemExit(f"{nm}: anchor_group or pair_index differ between cells")
        arrays[nm] = arr
        for name in NAMES:
            for j, part in enumerate(("fused", "cf")):
                for m in METRICS:
                    npz_out[f"{nm}__{name}__{part}__{m}"] = arr[name][j][m]
        for k, v in extra.items():
            if k.startswith("partition"):
                npz_out[f"{nm}__{k}"] = v
    npz_out.update(anchor_group=cl, pair_index=pair_index)

    paired = {}
    for nm, rec in leiden.items():
        n = rec["groups"]["n_groups"]
        ctrl, exact = nearest_control(n, kmeans)
        if ctrl is None:
            paired[nm] = None
            continue
        a, b = arrays[nm], arrays[kmeans_name(ctrl)]
        paired[nm] = {"leiden_count": n, "control_count": ctrl, "exact": exact,
                      **{name: {m: rc.point_ci(rto.margin_arrays(a, name, m) - rto.margin_arrays(b, name, m), cl)
                                for m in DIFF_METRICS} for name in NAMES}}

    if missing_leiden and not args.allow_incomplete:
        raise SystemExit(f"grid incomplete (missing {missing_leiden}); rerun those cells, or pass --allow-incomplete "
                         "to merge without a pick")
    pick = apply_pick_rule(leiden) if leiden else {}
    if missing_leiden:
        pick["withheld"] = (f"grid incomplete ({len(leiden)}/9 cells; missing {missing_leiden}): the rule below is "
                            "over the present cells only and is not the PLAN.md pick")

    ref = {arm: {"told": stored["arms"][arm]["eval"]["told"]["fusedT_vs_fusedTcf"]["r1"],
                 "reader": stored["arms"][arm]["eval"]["reader"]["fusedT_vs_fusedTcf"]["r1"],
                 "pick": stored["arms"][arm]["eval"]["pick"]["correct_share"]["point"],
                 "n_groups": stored["arms"][arm]["groups"]["n_groups"]} for arm in ("R0", "L", "K")}
    result = {"exploratory": "decides nothing; seed 42 development episodes only; PLAN.md applied as written",
              "plan_sha256": PLAN_SHA, "script_sha256": shas.pop(), "merge_script_sha256": rg.sha_file(Path(__file__)),
              "n_episodes": int(len(cl)), "n_clusters": int(len(np.unique(cl))),
              "reference": ref, "leiden": leiden, "kmeans": {str(n): rec for n, rec in sorted(kmeans.items())},
              "kmeans_counts": sorted(kmeans), "missing_leiden": missing_leiden, "missing_kmeans_counts": missing_km,
              "tables": {"n_groups": {f"k{k}": {str(r): leiden[leiden_name(k, r)]["groups"]["n_groups"]
                                                for r in RESOLUTIONS if leiden_name(k, r) in leiden} for k in KS},
                         "told_margin_r1": {f"k{k}": {str(r): margin(leiden[leiden_name(k, r)], "told")
                                                      for r in RESOLUTIONS if leiden_name(k, r) in leiden} for k in KS},
                         "reader_margin_r1": {f"k{k}": {str(r): margin(leiden[leiden_name(k, r)], "reader")
                                                        for r in RESOLUTIONS if leiden_name(k, r) in leiden}
                                              for k in KS}},
              "paired_leiden_minus_kmeans": paired, "pick": pick}
    result = rto.roundtrip(result)
    rg.assert_finite_tree(result)
    result_txt = merge_text({**result, "kmeans": kmeans})
    paths[0].write_text(json.dumps(result, indent=1, ensure_ascii=False))
    np.savez_compressed(paths[2], **npz_out)
    paths[1].write_text(result_txt + "\n")
    print(result_txt)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--cells", nargs="*", default=[], help="Leiden cells k:resolution, e.g. 20:1.0")
    p.add_argument("--kmeans", nargs="*", default=[], help="k-means group counts and/or 'auto'")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p = sub.add_parser("merge")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--allow-incomplete", action="store_true")
    p = sub.add_parser("probe")
    p.add_argument("--cells", nargs="+", required=True)
    args = ap.parse_args()
    {"run": cmd_run, "merge": cmd_merge, "probe": cmd_probe}[args.cmd](args)


if __name__ == "__main__":
    main()
