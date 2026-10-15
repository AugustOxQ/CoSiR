"""EXPLORATORY (decides nothing; seed 42 development episodes only). PLAN.md in this folder (SHA-256 asserted) fixes
step 1 of the grouping redesign: a style grouping beside affect, image and caption. This script is the CPU side of
PLAN.md (sections 2 to 6 and the evaluation side of section 7):

  group     Painting-level Leiden groupings: one node per scorer-train painting (36,518), the kNN union graph of
            src.model.communities.detect_communities (k 20, Euclidean on the unit vectors; run_sweep.knn_union_graph),
            Leiden RBConfigurationVertexPartition, resolution 1.0, seed 42 (run_sweep.leiden); rows inherit their
            painting's community; communities under 200 rows merged into the nearest-centroid community
            (run_told_oracle.merge_small, centroids in the source space). style_csd (CSD), style_gram (VGG-Gram) and
            image_leiden (CLIP image) use it; caption_leiden runs detect_communities defaults at row level on unit
            CLIP caption features plus the same merge; style_rand permutes style_csd's painting labels across
            paintings (seed 0). Leiden seeds 43 and 44 are kept for the stability diagnostic.
  heads     run_told_oracle.fit_one_head (CLIP ViT-B/32, the 60,000-row draw, LogisticRegression(C=1, max_iter=300))
            for every new grouping; for style_csd and style_gram also the image head on the grouping's own source
            features (same draw, same classifier; the caption head stays on CLIP caption features).
  eval      The arms of PLAN.md section 4 with a generalised run_told_oracle.evaluate_arm (any groupings, any told
            mapping), proven equal to the original on A0: told term and N6's hard reader fused on B (= stored C2)
            against their matched counterparts, B', the bar margin, per aspect pair, pick accuracy, paired differences
            against A0 and AR, readings R1 to R3 (section 6).
  describe  Label-free diagnostics of the set's new groupings (section 5); with --set style, afterwards the one
            disclosed label description of every new grouping (requires both eval outputs).

The style features arrive later than the rest, so every stage runs per set:
  --set descriptive   image_leiden, caption_leiden; arms A0, A3
  --set style         style_csd, style_gram, style_rand; arms A0 (recomputed as the paired reference), AR, A1, A1s,
                      A2, A2s
--smoke: the cached DINOv2-small CLS features (src/test/20260929_cross_encoder_stage1/features/dinov2_img.npy) stand in
for both CSD and Gram, heads on 3,000 rows, outputs in results/smoke/ (overwritable). Smoke numbers are not results.

Reuses by import, modifies nothing outside this folder: run_told_oracle (fit_one_head, evaluate_arm, merge_small,
group_stats, per_pair_margin, margin_arrays, pairs2/dpairs), run_sweep (knn_union_graph, leiden, setup) and the step-0
run_checks (e2_partitions, placeability, load_cell, now_ams; loaded under another module name). CPU only.

Run (from /project/CoSiR):
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python src/test/20261116_grouping_step1_style/run_step1.py \
        {group,heads,eval,describe} --set {descriptive,style} [--smoke]
    -> results/step1_<stage>_<set>.json, .txt (+ .npz for group, heads, eval). Non-smoke outputs are never overwritten.
"""
import argparse
import importlib.util
import json
import os
import sys
import time
import warnings
from pathlib import Path

import leidenalg
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_mutual_info_score

HERE = Path(__file__).resolve().parent
TOLD = HERE.parent / "20261111_community_told_oracle"
SWEEP = HERE.parent / "20261112_community_sweep"
STEP0 = HERE.parent / "20261115_grouping_step0_checks"
sys.path.insert(0, str(TOLD))
import run_told_oracle as rto  # noqa: E402  (imports diagnose_counterparts, diagnose_fixes, the old run_checks, run_n6)
sys.path.insert(0, str(SWEEP))
import run_sweep as rsw  # noqa: E402

# The quick-checks module is also called run_checks (run_told_oracle loaded it above); load step 0's under another name.
_spec = importlib.util.spec_from_file_location("step0_run_checks", STEP0 / "run_checks.py")
rk = importlib.util.module_from_spec(_spec)
sys.modules["step0_run_checks"] = rk
_spec.loader.exec_module(rk)

df, rc, rg, n6, n6c, dc = rto.df, rto.rc, rto.rg, rto.n6, rto.n6c, rto.dc

from src.data.artelingo import ANNOTATIONS_PATH, load_artelingo  # noqa: E402
from src.data.artelingo_splits import artelingo_aspect_labels, artelingo_splits  # noqa: E402
from src.eval.aspect_metrics import CONDITIONS, METRICS, per_anchor  # noqa: E402
from src.eval.aspect_nested import crossfit_nested  # noqa: E402
from src.eval.aspect_quick_checks import (centered_term, crossfit_condition_free, inferred_scores,  # noqa: E402
                                          uniform_probe_scores)
from src.model.communities import detect_communities  # noqa: E402

ROOT = rc.ROOT
PLAN = HERE / "PLAN.md"
PLAN_SHA = "b00a4a7aa796957addc232dd92e1b1579306f8b312748a36ac1479331f8b0520"
RES = HERE / "results"
STANDIN = ROOT / "src/test/20260929_cross_encoder_stage1/features/dinov2_img.npy"
STANDIN_ROWS = 308_723
STYLE_DIR = {"style_csd": Path("/data/SSD2/pre_extract/artelingo/style_csd_vitl"),
             "style_gram": Path("/data/SSD2/pre_extract/artelingo/style_vgg19_gram")}
GRAPH_K, RESOLUTION, LEIDEN_SEED, STAB_SEEDS = 20, 1.0, rsw.LEIDEN_SEED, (42, 43, 44)
MIN_ROWS = rto.MIN_ROWS                                    # 200
RAND_SEED = 0
SETS = {"descriptive": ("image_leiden", "caption_leiden"), "style": ("style_csd", "style_gram", "style_rand")}
SOURCE_HEADS = ("style_csd", "style_gram")
BASE = ("affect", "image", "caption")
if tuple(df.PARTS) != BASE or df.TOLD_PART != {"emotion": "affect", "style": "image", "genre": "image"}:
    raise AssertionError("diagnose_fixes.PARTS / TOLD_PART are not the told oracle's")


def _arm(extra, style_to, secondary=None, parts=None):
    return {"parts": tuple(parts) if parts else BASE + ((extra,) if extra else ()),
            "told": {"emotion": "affect", "style": style_to, "genre": "image"}, "secondary": secondary}


ARMS = {"A0": _arm(None, "image"),
        "A1": _arm("style_csd", "style_csd"), "A1s": _arm("style_csd", "style_csd", "style_csd"),
        "A2": _arm("style_gram", "style_gram"), "A2s": _arm("style_gram", "style_gram", "style_gram"),
        "AR": _arm("style_rand", "style_rand"),
        "A3": {"parts": ("affect", "image_leiden", "caption_leiden"),
               "told": {"emotion": "affect", "style": "image_leiden", "genre": "image_leiden"}, "secondary": None}}
ARMS_OF_SET = {"descriptive": ("A0", "A3"), "style": ("A0", "AR", "A1", "A1s", "A2", "A2s")}
STYLE_ARMS = ("A1", "A1s", "A2", "A2s")
HAND_MATCHED = ("A1", "A1s")
NAMES = ("told", "reader")
DIFF_METRICS = ("r1", "gain", "either")
L_STATED = {"told": (1.64, 1.37, 1.92), "reader": (0.35, 0.15, 0.57)}   # dispatch: told-oracle arm L
PICK_STATED = 54.7
A0_BAR_STATED = (0.31, 18.75, 18.44)                                     # PLAN.md R2: bar, reader R@1, B'
BAR_TARGET = 0.5
SXG = "style__genre"


# ---------------------------------------------------------------- common

def out_paths(stage, set_, smoke):
    d = RES / "smoke" if smoke else RES
    stem = f"step1_{stage}_{set_}"
    p = {"json": d / f"{stem}.json", "txt": d / f"{stem}.txt"}
    if stage in ("group", "heads", "eval"):
        p["npz"] = d / f"{stem}.npz"
    return d, p


def start(stage, set_, smoke):
    if rg.sha_file(PLAN) != PLAN_SHA:
        raise SystemExit("PLAN.md differs from the dispatched version (SHA-256)")
    d, paths = out_paths(stage, set_, smoke)
    if not smoke and any(p.exists() for p in paths.values()):
        raise SystemExit(f"results of stage {stage} --set {set_} exist in {d}; refusing to overwrite")
    d.mkdir(parents=True, exist_ok=True)
    return paths


def header(stage, set_, smoke, inputs):
    return {"exploratory": "decides nothing; seed 42 development episodes only; PLAN.md applied as written",
            "stage": stage, "set": set_, "smoke": bool(smoke), "plan_sha256": PLAN_SHA,
            "provenance": {"script_sha256": rg.sha_file(Path(__file__)),
                           "run_told_oracle_sha256": rg.sha_file(Path(rto.__file__)),
                           "run_sweep_sha256": rg.sha_file(Path(rsw.__file__)),
                           "step0_run_checks_sha256": rg.sha_file(Path(rk.__file__)),
                           "inputs_sha256": {rel(p): rg.sha_file(p) for p in inputs},
                           "argv": sys.argv, "pid": os.getpid(), "host_cpu_count": os.cpu_count(),
                           "threads": {k: os.environ.get(k) for k in
                                       ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES")},
                           "started": rk.now_ams()}}


def rel(p):
    p = Path(p)
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def write(paths, result, text_fn, npz=None):
    """npz first (its SHA-256 goes into the JSON), then the JSON (marks the stage done), then the text."""
    if npz is not None:
        np.savez_compressed(paths["npz"], **npz)
        result["provenance"]["npz_sha256"] = rg.sha_file(paths["npz"])
    result["provenance"]["finished"] = rk.now_ams()
    result = rto.roundtrip(result)
    rg.assert_finite_tree(result)
    paths["json"].write_text(json.dumps(result, indent=1, ensure_ascii=False))
    txt = text_fn(result)
    paths["txt"].write_text(txt + "\n")
    print(txt)


def load_stage(stage, set_, smoke):
    _, p = out_paths(stage, set_, smoke)
    if not p["json"].exists():
        raise SystemExit(f"{rel(p['json'])} is missing: run stage {stage} --set {set_}{' --smoke' if smoke else ''} first")
    rec = json.loads(p["json"].read_text())
    if rec["plan_sha256"] != PLAN_SHA or rec["smoke"] != bool(smoke) or rec["set"] != set_:
        raise AssertionError(f"{rel(p['json'])}: written under another PLAN.md, smoke flag or set")
    z = None
    if "npz" in p:
        if rg.sha_file(p["npz"]) != rec["provenance"]["npz_sha256"]:
            raise AssertionError(f"{rel(p['npz'])}: SHA-256 differs from its JSON")
        z = np.load(p["npz"])
    return rec, z, [p["json"]] + ([p["npz"]] if "npz" in p else [])


def base_data():
    data = load_artelingo()
    sp = artelingo_splits(data)
    scorer_train, groups = np.asarray(sp.scorer_train), np.asarray(sp.groups)
    g_st = np.unique(groups[scorer_train], return_inverse=True)[1].astype(np.int64)
    return data, sp, scorer_train, groups, g_st


def first_rows(g):
    """Index of the first row (in the given order) of every painting code 0..P-1."""
    P = int(g.max()) + 1
    first = np.zeros(P, dtype=np.int64)
    first[g[::-1]] = np.arange(len(g))[::-1]
    return first


def ci_txt(x, w=6):
    return rto.c(x, w)


# ---------------------------------------------------------------- style features (section 1 outputs or stand-in)

def annotation_image_paths(data):
    """The annotation 'image' field per load_artelingo() row (row i is annotations[sample_ids[i]])."""
    ann = json.loads(Path(ANNOTATIONS_PATH).read_text())
    sids = np.asarray(data.sample_ids, dtype=np.int64)
    paths = np.asarray([ann[int(i)]["image"] for i in sids], dtype=object)
    paintings = np.asarray([ann[int(i)]["painting"] for i in sids], dtype=object)
    if not np.array_equal(paintings, np.asarray(data.paintings, dtype=object)):
        raise AssertionError("annotation join differs from load_artelingo()'s painting field")
    return ann, sids, paths


STYLE_KIND = {"style_csd": ("CSD style embedding", 768), "style_gram": ("VGG-19 Gram descriptor", 640)}
N_IMAGES = 61_402


def real_style_features(name, ann, sids, paths, data):
    """Features of the GPU extraction (PLAN.md section 1), read through meta.json: embeddings.npy holds one row per
    distinct image (meta images.image_paths order) and row_to_image.npy maps load_artelingo() rows to images
    (meta: features_in_load_artelingo_order = embeddings[row_to_image]). The alignment is re-asserted here."""
    d = STYLE_DIR[name]
    meta_p, emb_p, r2i_p, r2ia_p = (d / "meta.json", d / "embeddings.npy", d / "row_to_image.npy",
                                    d / "row_to_image_annotation_order.npy")
    meta = json.loads(meta_p.read_text())
    kind, dim = STYLE_KIND[name]
    if meta["plan"]["sha256"] != PLAN_SHA or meta["mode"] != "full" or not meta["kind"].startswith(kind):
        raise AssertionError(f"{name}: meta.json is not the full {kind} pass under this PLAN.md")
    if meta["images"]["annotations_sha256"] != rg.sha_file(Path(ANNOTATIONS_PATH)):
        raise AssertionError(f"{name}: features were extracted against another annotation file")
    img_paths = np.asarray(meta["images"]["image_paths"], dtype=object)
    emb = np.load(emb_p)
    r2i_raw, r2ia_raw = np.load(r2i_p), np.load(r2ia_p)
    r2i, r2ia = np.asarray(r2i_raw, dtype=np.int64), np.asarray(r2ia_raw, dtype=np.int64)
    rm = meta["row_mapping"]
    checks = {       # meta.json hashes the arrays' bytes (extract_style_features.sha256_array), not the .npy files
        "row_to_image_array_sha256_equals_meta": rg.sha_array(r2i_raw) == rm["row_to_image_sha256"],
        "row_to_image_annotation_order_array_sha256_equals_meta":
            rg.sha_array(r2ia_raw) == rm["row_to_image_annotation_order_sha256"],
        "n_images_61402": bool(len(img_paths) == N_IMAGES == meta["images"]["n_images_total"]
                               == meta["images"]["n_images_in_this_file"] == len(set(img_paths.tolist()))),
        "embeddings_shape": bool(emb.shape == (N_IMAGES, dim) and emb.dtype == np.float32 and np.isfinite(emb).all()),
        "annotation_order_map_matches_annotation_image_field": bool(
            len(r2ia) == len(ann) and np.array_equal(img_paths[r2ia], np.asarray([a["image"] for a in ann], dtype=object))),
        "row_to_image_equals_annotation_map_at_sample_ids": bool(np.array_equal(r2i, r2ia[sids])),
        "load_artelingo_rows_map_to_their_annotation_image": bool(np.array_equal(img_paths[r2i], paths)),
        "every_image_used": bool(len(np.unique(r2i)) == N_IMAGES),
    }
    # rows of one painting map to one image (and one image to one painting)
    _, p_inv = np.unique(np.asarray(data.paintings, dtype=object), return_inverse=True)
    pi = np.unique(np.stack([p_inv, r2i], 1), axis=0)
    checks["painting_to_image_one_to_one"] = bool(len(pi) == len(np.unique(pi[:, 0])) == len(np.unique(pi[:, 1])))
    if not all(checks.values()):
        raise AssertionError(f"{name}: alignment checks failed: {checks}")
    X = np.ascontiguousarray(emb[r2i], dtype=np.float32)
    prov = {"source": f"{meta['kind']} ({rel(d)}; embeddings[row_to_image], meta.json row_mapping)",
            "files_sha256": {rel(p): rg.sha_file(p) for p in (meta_p, emb_p, r2i_p, r2ia_p)},
            "extraction_script_sha256": meta["script"]["sha256"], "model": meta["model"],
            "alignment_checks": checks}
    return X, prov


def style_features(name, smoke, data):
    """(n_rows, D) float32 features in load_artelingo() row order, the alignment checks and the provenance."""
    ann, sids, paths = annotation_image_paths(data)
    if smoke:
        arr = np.load(STANDIN, mmap_mode="r")
        if arr.shape[0] != STANDIN_ROWS or len(ann) != STANDIN_ROWS:
            raise AssertionError("stand-in features are not row-aligned with the annotation file")
        X = np.ascontiguousarray(arr[sids], dtype=np.float32)
        prov = {"source": "SMOKE STAND-IN: DINOv2-small CLS, row-aligned with artelingo_train.json (README of "
                          "src/test/20260929_cross_encoder_stage1); row i of load_artelingo() = annotation sample_ids[i]",
                "file": rel(STANDIN), "sha256": rg.sha_file(STANDIN), "shape": list(arr.shape)}
    else:
        X, prov = real_style_features(name, ann, sids, paths, data)
    if X.shape[0] != len(sids) or not np.isfinite(X).all():
        raise AssertionError(f"{name}: features must be finite, one row per load_artelingo() row")
    # rows that share an image path must carry the same vector (one vector per image, mapped by path)
    _, inv = np.unique(paths, return_inverse=True)
    first = first_rows(inv)
    same = bool(np.array_equal(X, X[first[inv]]))
    if not same:
        raise AssertionError(f"{name}: rows of one image path carry different vectors")
    norms = np.linalg.norm(X.astype(np.float64), axis=1)
    prov["checks"] = {"rows": int(len(X)), "dim": int(X.shape[1]), "distinct_image_paths": int(len(first)),
                      "rows_sharing_a_path_identical": same, "row_norm_min": float(norms.min()),
                      "row_norm_max": float(norms.max())}
    return X, prov


# ---------------------------------------------------------------- stage group

def leiden_seeded(graph, kind, seed):
    """leidenalg.find_partition with RBConfiguration (resolution 1.0) or Modularity at a given seed, labels compacted."""
    if kind == "rb":
        p = leidenalg.find_partition(graph, leidenalg.RBConfigurationVertexPartition, resolution_parameter=RESOLUTION,
                                     seed=seed)
    else:
        p = leidenalg.find_partition(graph, leidenalg.ModularityVertexPartition, seed=seed)
    labels = np.unique(np.asarray(p.membership, dtype=np.int64), return_inverse=True)[1].astype(np.int64)
    return labels, {"quality": float(p.quality()), "modularity": float(p.modularity)}


def merged_info(raw, merged, moves):
    return {"raw": rto.group_stats(raw), "n_raw": int(raw.max()) + 1, "moves": moves,
            "rows_merged": int(sum(m["rows"] for m in moves)), "groups": rto.group_stats(merged),
            "partition_sha256": rg.sha_array(merged)}


def painting_grouping(name, X_st, g_st, source):
    """PLAN.md section 2, painting-level recipe. X_st: source features of the scorer-train rows."""
    t0 = time.time()
    first = first_rows(g_st)
    Xp_raw = np.asarray(X_st[first], dtype=np.float32)
    dev = np.abs(X_st - Xp_raw[g_st]).max(axis=1)
    differ = np.zeros(len(first), dtype=bool)
    np.logical_or.at(differ, g_st, dev > 0)
    Xp = rc.unit(Xp_raw)                                    # unit vectors, Euclidean kNN
    t = time.time()
    graph = rsw.knn_union_graph(Xp, GRAPH_K)
    graph_s = time.time() - t
    t = time.time()
    raw_p, quality = rsw.leiden(graph, RESOLUTION)          # RBConfiguration, resolution 1.0, seed 42
    leiden_s = time.time() - t
    seeds = {s: leiden_seeded(graph, "rb", s)[0] for s in STAB_SEEDS}
    wrapper_ok = bool(np.array_equal(seeds[LEIDEN_SEED], raw_p))
    if not wrapper_ok:
        raise AssertionError(f"{name}: the seeded Leiden wrapper differs from run_sweep.leiden at seed 42")
    X_rows = Xp[g_st]                                       # rows inherit their painting's vector (source space)
    merged = {}
    for s in STAB_SEEDS:
        merged[s] = rto.merge_small(seeds[s][g_st], X_rows, MIN_ROWS)
    local, moves = merged[LEIDEN_SEED]
    const = bool(np.array_equal(local, local[first][g_st]))
    if not const:
        raise AssertionError(f"{name}: merged labels are not constant within a painting")
    info = {"source": source, "recipe": "painting-level",
            "settings": {"nodes": "one per scorer-train painting, vector of its first scorer-train row, unit-normalised",
                         "graph": "run_sweep.knn_union_graph (detect_communities' kNN union graph)", "k": GRAPH_K,
                         "partition": "leidenalg.RBConfigurationVertexPartition", "resolution": RESOLUTION,
                         "seed": LEIDEN_SEED, "merge_under_rows": MIN_ROWS,
                         "merge_centroids": "row-weighted means of the unit painting vectors"},
            "painting_vectors": {"paintings": int(len(first)), "paintings_with_differing_rows": int(differ.sum()),
                                 "max_abs_deviation": float(dev.max()),
                                 "norm_before_unit_min": float(np.linalg.norm(Xp_raw, axis=1).min()),
                                 "norm_before_unit_max": float(np.linalg.norm(Xp_raw, axis=1).max())},
            "graph": {"n_vertices": graph.vcount(), "n_edges": graph.ecount(), "build_s": graph_s},
            "leiden": {**quality, "n_painting_communities": int(raw_p.max()) + 1, "leiden_s": leiden_s,
                       "seeded_wrapper_equals_run_sweep_leiden": wrapper_ok},
            **merged_info(raw_p[g_st], local, moves),
            "stability_seeds": {str(s): {"n_raw": int(seeds[s].max()) + 1, "n_groups": int(merged[s][0].max()) + 1,
                                         "partition_sha256": rg.sha_array(merged[s][0])} for s in STAB_SEEDS},
            "runtime_s": time.time() - t0}
    arrays = {name: local, f"{name}__raw": raw_p[g_st], f"{name}__painting_raw": raw_p,
              **{f"{name}__seed{s}": merged[s][0] for s in STAB_SEEDS if s != LEIDEN_SEED}}
    rc.log(f"{name}: {info['leiden']['n_painting_communities']} painting communities -> {info['groups']['n_groups']} "
           f"groups after merging ({info['rows_merged']} rows moved); graph {graph_s:.0f}s, Leiden {leiden_s:.1f}s")
    return info, arrays


def caption_grouping(X_st):
    """PLAN.md section 2: detect_communities defaults at row level on unit CLIP caption features, plus the merge."""
    t0 = time.time()
    X = rc.unit(X_st)
    t = time.time()
    raw = detect_communities(X)                             # k 20, ModularityVertexPartition, seed 42
    dc_s = time.time() - t
    t = time.time()
    graph = rsw.knn_union_graph(X, GRAPH_K)                 # the same graph, kept for seeds 43 and 44
    graph_s = time.time() - t
    seeds, quality = {}, None
    for s in STAB_SEEDS:
        seeds[s], q = leiden_seeded(graph, "modularity", s)
        if s == LEIDEN_SEED:
            quality = q
    wrapper_ok = bool(np.array_equal(seeds[LEIDEN_SEED], raw))
    if not wrapper_ok:
        raise AssertionError("caption_leiden: graph + Modularity at seed 42 differs from detect_communities")
    merged = {s: rto.merge_small(seeds[s], X, MIN_ROWS) for s in STAB_SEEDS}
    local, moves = merged[LEIDEN_SEED]
    info = {"source": "CLIP ViT-B/32 caption features of the scorer-train rows, unit-normalised", "recipe": "row-level",
            "settings": {"call": "src.model.communities.detect_communities(X) defaults: k 20 kNN union graph "
                                 "(Euclidean), ModularityVertexPartition, seed 42", "merge_under_rows": MIN_ROWS,
                         "merge_centroids": "row means of the unit caption features"},
            "graph": {"n_vertices": graph.vcount(), "n_edges": graph.ecount(), "build_s": graph_s},
            "leiden": {**quality, "detect_communities_s": dc_s,
                       "graph_plus_modularity_seed42_equals_detect_communities": wrapper_ok},
            **merged_info(raw, local, moves),
            "stability_seeds": {str(s): {"n_raw": int(seeds[s].max()) + 1, "n_groups": int(merged[s][0].max()) + 1,
                                         "partition_sha256": rg.sha_array(merged[s][0])} for s in STAB_SEEDS},
            "runtime_s": time.time() - t0}
    arrays = {"caption_leiden": local, "caption_leiden__raw": raw,
              **{f"caption_leiden__seed{s}": merged[s][0] for s in STAB_SEEDS if s != LEIDEN_SEED}}
    rc.log(f"caption_leiden: {info['n_raw']} communities -> {info['groups']['n_groups']} after merging "
           f"({info['rows_merged']} rows moved); detect_communities {dc_s:.0f}s, graph again {graph_s:.0f}s")
    return info, arrays


def random_slot(csd_local, g_st):
    """style_csd's painting labels permuted across paintings (seed 0); painting counts per group kept exactly."""
    first = first_rows(g_st)
    lab_p = csd_local[first]
    if not np.array_equal(lab_p[g_st], csd_local):
        raise AssertionError("style_csd labels are not constant within a painting")
    perm = np.random.default_rng(RAND_SEED).permutation(lab_p)
    local = perm[g_st].astype(np.int64)
    kept = bool(np.array_equal(np.bincount(perm), np.bincount(lab_p)))
    info = {"source": "style_csd's merged painting labels, permuted across paintings",
            "settings": {"permutation": f"np.random.default_rng({RAND_SEED}).permutation over the 36,518 painting "
                                        "labels", "merge": "none (sizes kept in paintings)"},
            "painting_counts_kept": kept, "groups": rto.group_stats(local), "partition_sha256": rg.sha_array(local),
            "paintings_moved_share": float((perm != lab_p).mean())}
    if not kept:
        raise AssertionError("style_rand: painting counts per group changed")
    return info, {"style_rand": local, "style_rand__painting": perm}


def stage_group(set_, smoke):
    paths = start("group", set_, smoke)
    t0 = time.time()
    result = header("group", set_, smoke, [])
    data, sp, scorer_train, groups, g_st = base_data()
    out, npz, features = {}, {"scorer_train": scorer_train, "g_st": g_st}, {}
    if set_ == "descriptive":
        out["image_leiden"], arr = painting_grouping("image_leiden", data.img_features[scorer_train], g_st,
                                                     "CLIP ViT-B/32 image features (the cached CoSiR features)")
        npz.update(arr)
        out["caption_leiden"], arr = caption_grouping(data.txt_features[scorer_train])
        npz.update(arr)
    else:
        for name in SOURCE_HEADS:
            X, prov = style_features(name, smoke, data)
            features[name] = prov
            out[name], arr = painting_grouping(name, X[scorer_train], g_st, prov["source"])
            npz.update(arr)
            del X
        out["style_rand"], arr = random_slot(npz["style_csd"], g_st)
        npz.update(arr)
    result.update(n_rows=int(len(scorer_train)), n_paintings=int(g_st.max()) + 1, features=features, groupings=out,
                  runtime_s=round(time.time() - t0, 1))
    write(paths, result, text_group, npz)


def text_group(r):
    L = [f"EXPLORATORY step 1, groupings (PLAN.md sha {r['plan_sha256'][:12]}; set {r['set']}"
         f"{'; SMOKE (stand-in features)' if r['smoke'] else ''}); scorer-train rows {r['n_rows']}, paintings "
         f"{r['n_paintings']}", "",
         f"  {'grouping':15s} {'recipe':14s} {'raw':>5s} {'merged':>6s} {'rows moved':>10s} {'min':>6s} {'median':>7s} "
         f"{'max':>6s} {'effN':>6s} {'largest%':>8s} | seeds 43/44 groups | runtime"]
    for nm, g in r["groupings"].items():
        s = g["groups"]
        raw = g.get("n_raw", "-")
        st = g.get("stability_seeds")
        st_txt = "/".join(str(st[k]["n_groups"]) for k in ("43", "44")) if st else "n/a"
        L.append(f"  {nm:15s} {g.get('recipe', 'permutation'):14s} {raw!s:>5s} {s['n_groups']:6d} "
                 f"{g.get('rows_merged', 0):10d} {s['min']:6d} {s['median']:7.0f} {s['max']:6d} {s['effective_n']:6.1f} "
                 f"{100 * s['largest_share']:8.2f} | {st_txt:>17s} | {g.get('runtime_s', 0):.0f}s")
    for nm, g in r["groupings"].items():
        if "painting_vectors" in g:
            pv, gr = g["painting_vectors"], g["graph"]
            L.append(f"  {nm}: graph {gr['n_vertices']} nodes / {gr['n_edges']} edges; {g['leiden']['n_painting_communities']} "
                     f"painting communities; paintings whose rows differ in the source vector {pv['paintings_with_differing_rows']} "
                     f"(max |dev| {pv['max_abs_deviation']:.2e}); norm before unit [{pv['norm_before_unit_min']:.4f}, "
                     f"{pv['norm_before_unit_max']:.4f}]")
        elif nm == "caption_leiden":
            gr = g["graph"]
            L.append(f"  {nm}: graph {gr['n_vertices']} nodes / {gr['n_edges']} edges; graph + Modularity seed 42 "
                     f"equals detect_communities: {g['leiden']['graph_plus_modularity_seed42_equals_detect_communities']}")
        else:
            L.append(f"  {nm}: painting counts kept {g['painting_counts_kept']}; paintings moved "
                     f"{100 * g['paintings_moved_share']:.1f}%")
    for nm, f in r["features"].items():
        L.append(f"  features {nm}: {f['source']} | checks {f['checks']}")
    L.append(f"runtime {r['runtime_s']:.0f}s")
    return "\n".join(L)


# ---------------------------------------------------------------- stage heads

def fit_image_head(ctx, lab, scorer_train, n_rows, F):
    """run_told_oracle.fit_one_head's image branch with the image features replaced by F (rows aligned with ctx.data):
    the same draw (rng PROBE_SEED), the same 10,000 check rows (rng 1), the same LogisticRegression(C=1, max_iter=300)
    on unit-normalised features. Returns the posteriors (selection rows; NaN elsewhere), provenance and classes."""
    draw = np.random.default_rng(rc.PROBE_SEED).choice(scorer_train, n_rows, replace=False)
    rest = np.setdiff1d(scorer_train, draw)
    check = np.random.default_rng(1).choice(rest, min(n6.CHECK_ROWS, len(rest)), replace=False)
    sel = ctx.selection
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        clf = LogisticRegression(C=1.0, max_iter=300).fit(rc.unit(F[draw]), lab[draw])
    full = np.full((len(ctx.groups), len(clf.classes_)), np.nan, dtype=np.float32)
    full[sel] = clf.predict_proba(rc.unit(F[sel]))
    if not (np.isfinite(full[sel]).all() and np.isnan(full[~ctx.in_sel]).all()):
        raise AssertionError("posteriors must be finite on selection rows and NaN elsewhere")
    counts = np.bincount(lab[check])
    prov = {"n_classes": int(len(clf.classes_)), "draw_rows_sha256": rg.sha_array(np.sort(draw)),
            "heldout_accuracy": {"img": 100 * float(clf.score(rc.unit(F[check]), lab[check]))},
            "check_majority_share": 100 * float(counts.max() / counts.sum()), "uniform": 100.0 / len(clf.classes_),
            "convergence_warnings": int(sum(issubclass(w.category, ConvergenceWarning) for w in caught))}
    return full, prov, np.asarray(clf.classes_)


def stage_heads(set_, smoke):
    paths = start("heads", set_, smoke)
    t0 = time.time()
    grec, gz, ginputs = load_stage("group", set_, smoke)
    result = header("heads", set_, smoke, ginputs)
    rows = n6.SMOKE_HEAD_ROWS if smoke else n6.HEAD_ROWS
    ctx = rg.EvalContext(df.SEED, False)
    scorer_train = np.asarray(artelingo_splits(ctx.data).scorer_train)
    if not np.array_equal(scorer_train, gz["scorer_train"]):
        raise AssertionError("scorer_train differs from the group stage's")
    sel = ctx.selection
    checks, heads, npz = {}, {}, {"selection": sel}
    post_primary = {}
    for name in SETS[set_]:
        t = time.time()
        local = np.asarray(gz[name], dtype=np.int64)
        if rg.sha_array(local) != grec["groupings"][name]["partition_sha256"]:
            raise AssertionError(f"{name}: partition SHA-256 differs from the group JSON")
        lab = rto.global_labels(local, scorer_train, len(ctx.groups))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            post, prov = rto.fit_one_head(ctx, lab, scorer_train, rows)
        prov["convergence_warnings"] = int(sum(issubclass(w.category, ConvergenceWarning) for w in caught))
        prov["n_groups"] = int(local.max()) + 1
        prov["all_groups_in_draw"] = bool(prov["n_classes"] == prov["n_groups"])
        if not smoke and not prov["all_groups_in_draw"]:
            raise AssertionError(f"{name}: a group is missing from the 60,000-row draw")
        npz[f"{name}__img"], npz[f"{name}__txt"] = post["img"][sel], post["txt"][sel]
        post_primary[name] = post
        entry = {"clip": prov}
        rc.log(f"{name}: CLIP heads held-out {prov['heldout_accuracy']['img']:.2f}/{prov['heldout_accuracy']['txt']:.2f} "
               f"(majority {prov['check_majority_share']:.2f}) [{time.time() - t:.0f}s]")
        if name in SOURCE_HEADS:
            X, fprov = style_features(name, smoke, ctx.data)
            if name == SOURCE_HEADS[0]:
                # the copy with CLIP image features must equal fit_one_head's image head
                p_chk, prov_chk, _ = fit_image_head(ctx, lab, scorer_train, rows, ctx.data.img_features)
                ok = bool(np.array_equal(p_chk, post["img"], equal_nan=True)
                          and prov_chk["heldout_accuracy"]["img"] == prov["heldout_accuracy"]["img"]
                          and prov_chk["draw_rows_sha256"] == prov["draw_rows_sha256"])
                checks["fit_image_head_with_CLIP_equals_fit_one_head_image_head"] = ok
                if not ok:
                    raise AssertionError("the source-feature image head copy differs from fit_one_head on CLIP features")
                del p_chk
            p_src, prov_src, classes = fit_image_head(ctx, lab, scorer_train, rows, X)
            if not (prov_src["n_classes"] == prov["n_classes"] and prov_src["draw_rows_sha256"] == prov["draw_rows_sha256"]
                    and np.array_equal(classes, np.unique(lab[np.random.default_rng(rc.PROBE_SEED).choice(
                        scorer_train, rows, replace=False)]))):
                raise AssertionError(f"{name}: source image head differs in classes or draw from the CLIP heads")
            npz[f"{name}__img_src"] = p_src[sel]
            entry["source_image"] = {**prov_src, "features": fprov,
                                     "caption_head": "the CLIP caption head above (unchanged)"}
            rc.log(f"{name}: source image head held-out {prov_src['heldout_accuracy']['img']:.2f} [{time.time() - t:.0f}s]")
            del X, p_src
        heads[name] = entry
    checks["selection_rows"] = int(len(sel))
    result.update(head_rows=int(rows), checks=checks, heads=heads, runtime_s=round(time.time() - t0, 1))
    write(paths, result, text_heads, npz)


def text_heads(r):
    L = [f"EXPLORATORY step 1, heads (PLAN.md sha {r['plan_sha256'][:12]}; set {r['set']}"
         f"{'; SMOKE (stand-in features, 3,000-row heads)' if r['smoke'] else ''}); draw {r['head_rows']} rows; "
         "held-out on 10,000 scorer-train check rows (%)",
         "Checks: " + "; ".join(f"{k} {v}" for k, v in r["checks"].items()), "",
         f"  {'grouping':15s} {'groups':>6s} {'classes':>7s} {'img':>6s} {'txt':>6s} {'majority':>8s} {'1/k':>6s} "
         f"{'conv':>4s} | source image head"]
    for nm, e in r["heads"].items():
        h = e["clip"]
        src = e.get("source_image")
        src_txt = (f"{src['heldout_accuracy']['img']:.2f} (classes {src['n_classes']}, conv {src['convergence_warnings']})"
                   if src else "-")
        L.append(f"  {nm:15s} {h['n_groups']:6d} {h['n_classes']:7d} {h['heldout_accuracy']['img']:6.2f} "
                 f"{h['heldout_accuracy']['txt']:6.2f} {h['check_majority_share']:8.2f} {h['uniform']:6.2f} "
                 f"{h['convergence_warnings']:4d} | {src_txt}")
    L.append(f"runtime {r['runtime_s']:.0f}s")
    return "\n".join(L)


# ---------------------------------------------------------------- stage eval

def evaluate_general(post, parts, told, ctx, B, pB):
    """run_told_oracle.evaluate_arm for any ordered tuple of groupings and any told mapping (aspect -> grouping).
    The same calls as the original: T_6 = inferred_scores(post, ep, 'hard', parts) and T_6u =
    uniform_probe_scores(post, ep, parts) (what run_n6.n6_terms returns for parts = PARTS); the told term picks
    diagnose_fixes.stacked(post, ep, parts) at the told grouping's index. Also returns T_6u and the reader's picks."""
    ep, cl = ctx.pooled, ctx.anchor_group
    parts = tuple(parts)
    aspect_of = {"a": np.array([rg.PAIRS[i][0] for i in ctx.pair_index]),
                 "b": np.array([rg.PAIRS[i][1] for i in ctx.pair_index])}
    part_idx = {cnd: np.array([parts.index(told[a]) for a in aspect_of[cnd]]) for cnd in CONDITIONS}
    t6, info = inferred_scores(post, ep, "hard", parts)
    t6u = uniform_probe_scores(post, ep, parts)
    st6 = df.stacked(post, ep, parts)
    terms = {"told": df.as_f32({c: df.pick(st6, part_idx[c]) for c in CONDITIONS}), "reader": df.as_f32(t6)}
    out, arrays = {}, {}
    for name, s in terms.items():
        rc.assert_finite_scores(s, name)
        nested, ctrl, picks = crossfit_nested(B, B, s, ctx.parity)
        cf, cf_picks = crossfit_condition_free(B, B, dc.cf_version(s), ctx.parity)
        pn, pc = per_anchor(nested), per_anchor(cf)
        out[name] = {"fusedT_vs_B": df.diff(pn, pB, cl), "fusedTcf_vs_B": df.diff(pc, pB, cl),
                     "fusedT_vs_fusedTcf": df.diff(pn, pc, cl), "per_pair": rto.per_pair_margin(pn, pc, pB, ctx),
                     "T_picks": picks, "cf_picks": cf_picks,
                     "control_ranks_as_B": df.same_metrics(per_anchor(ctrl), pB)}
        arrays[name] = (pn, pc)
        rc.log(f"  {name}: margin R@1 {out[name]['fusedT_vs_fusedTcf']['r1']['point']:.2f}")
    picked = {c: info[c]["weights"].argmax(axis=1) for c in CONDITIONS}
    correct = {c: picked[c] == part_idx[c] for c in CONDITIONS}
    both = correct["a"] & correct["b"]
    out["pick"] = {
        "correct_share": rc.point_ci(0.5 * (correct["a"].astype(float) + correct["b"].astype(float)), cl),
        "per_pair_condition": {name: {c: 100 * float(correct[c][ctx.pair_index == i].mean()) for c in CONDITIONS}
                               for i, name in enumerate(rg.POOLED_ORDER)},
        "both_correct_share": 100 * float(both.mean()),
        "picked_partition": {name: {c: {h: 100 * float(np.mean(picked[c][ctx.pair_index == i] == j))
                                        for j, h in enumerate(parts)} for c in CONDITIONS}
                             for i, name in enumerate(rg.POOLED_ORDER)}}
    return out, arrays, t6u, picked, part_idx


def bar_margin(arrays, pBp, ctx):
    """Bar margin: fused reader minus whichever of B' and the reader's counterpart has the larger mean R@1 (paired,
    per anchor). Per aspect pair with the same comparator."""
    cl = ctx.anchor_group
    pn, pc = arrays["reader"]
    m_bp, m_cf = float(np.mean(pBp["r1"])), float(np.mean(pc["r1"]))
    comp_name = "B_prime" if m_bp >= m_cf else "counterpart"
    comp = pBp if comp_name == "B_prime" else pc
    v = np.asarray(pn["r1"], dtype=np.float64) - np.asarray(comp["r1"], dtype=np.float64)
    info = {"comparator": comp_name, "reader_r1": 100 * float(np.mean(pn["r1"])), "B_prime_r1": 100 * m_bp,
            "counterpart_r1": 100 * m_cf, "r1": rc.point_ci(v, cl),
            "gain_vs_comparator": rc.point_ci(np.asarray(pn["gain"], np.float64) - np.asarray(comp["gain"], np.float64), cl),
            "per_pair_r1": {p: rc.point_ci(v[ctx.pair_index == i], cl[ctx.pair_index == i])
                            for i, p in enumerate(rg.POOLED_ORDER)}}
    return v, info


def paired(arr, ref, bar_v, ref_bar, ctx):
    cl, pi = ctx.anchor_group, ctx.pair_index
    out = {nm: {m: rc.point_ci(rto.margin_arrays(arr, nm, m) - rto.margin_arrays(ref, nm, m), cl) for m in DIFF_METRICS}
           for nm in NAMES}
    out["per_pair_r1"] = {nm: {p: rc.point_ci((rto.margin_arrays(arr, nm, "r1") - rto.margin_arrays(ref, nm, "r1"))[pi == i],
                                              cl[pi == i]) for i, p in enumerate(rg.POOLED_ORDER)} for nm in NAMES}
    out["bar_r1"] = rc.point_ci(bar_v - ref_bar, cl)
    return out


def r2(entry):
    bar, gain = entry["bar"]["r1"], entry["eval"]["reader"]["fusedT_vs_fusedTcf"]["gain"]
    ok = bool(bar["point"] >= BAR_TARGET and bar["ci95"][0] > 0 and gain["ci95"][0] > 0)
    return {"bar_margin": bar, "comparator": entry["bar"]["comparator"], "reader_gain_margin": gain,
            "bar_at_least_0.5": bool(bar["point"] >= BAR_TARGET), "bar_lower_above_0": bool(bar["ci95"][0] > 0),
            "gain_lower_above_0": bool(gain["ci95"][0] > 0), "meets": ok}


def readings(arms, set_):
    out = {"R2_reference_A0": r2(arms["A0"])}
    if set_ == "descriptive":
        out["R2_descriptive_A3"] = {**r2(arms["A3"]), "note": "A3 is descriptive; PLAN.md's readings are not about it"}
        return out
    out["R1"] = {a: {"told_sxg_minus_A0": arms[a]["vs_A0"]["per_pair_r1"]["told"][SXG],
                     "meets": bool(arms[a]["vs_A0"]["per_pair_r1"]["told"][SXG]["ci95"][0] > 0)} for a in STYLE_ARMS}
    out["R2"] = {a: r2(arms[a]) for a in ("AR",) + STYLE_ARMS}
    out["R3"] = {a: {"reader_margin_minus_AR": arms[a]["vs_AR"]["reader"]["r1"],
                     "meets": bool(arms[a]["vs_AR"]["reader"]["r1"]["ci95"][0] > 0)} for a in STYLE_ARMS}
    hand = [a for a in HAND_MATCHED if out["R2"][a]["meets"] and out["R3"][a]["meets"]]
    any_r2 = [a for a in STYLE_ARMS if out["R2"][a]["meets"]]
    if hand:
        prop = max(hand, key=lambda a: out["R2"][a]["bar_margin"]["point"])
        default = (f"{prop} meets R2 and R3 (hand-matched; larger bar margin among {hand}): proposed for a "
                   "pre-registered fresh-seed test (49 to 51) with its own decision rule; A2 and A2s reported beside it "
                   "as the generalisation result. The user decides.")
    elif not any_r2:
        prop = None
        default = ("no arm (A1, A1s, A2, A2s) meets R2: no test is built; 9 October decides between continuing "
                   "(design L, step 3) and changing course. The user decides.")
    else:
        prop = None
        default = (f"{any_r2} meet R2 but no hand-matched arm meets both R2 and R3: PLAN.md's default does not cover "
                   "this case; the user decides.")
    out["default_for_9_october"] = {"hand_matched_meeting_R2_and_R3": hand, "arms_meeting_R2": any_r2,
                                    "proposed": prop, "text": default,
                                    "rule": "if a hand-matched arm (A1, A1s) meets R2 and R3, the one with the larger "
                                            "bar margin is proposed; if no arm meets R2, no test is built"}
    return out


def full_post(z, key, ctx):
    a = np.asarray(z[key], dtype=np.float32)
    full = np.full((len(ctx.groups), a.shape[1]), np.nan, dtype=np.float32)
    full[ctx.selection] = a
    return full


def stage_eval(set_, smoke):
    paths = start("eval", set_, smoke)
    t0 = time.time()
    hrec, hz, hinputs = load_stage("heads", set_, smoke)
    grec, _, ginputs = load_stage("group", set_, smoke)
    rows = n6.SMOKE_HEAD_ROWS if smoke else n6.HEAD_ROWS
    inputs = hinputs + ginputs + [rsw.STORED_JSON, rsw.STORED_NPZ, df.POSTERIORS, df.GATE, n6.PARTITIONS,
                                  rto.AFFECT_NPZ, rto.AFFECT_JSON, rto.STORED_CF, rto.STORED_FIXES]
    result = header("eval", set_, smoke, inputs)

    S = rsw.setup()                 # B = C2; R0 and stored arms L/K reproduce exactly (asserted inside)
    ctx, cl = S.ctx, S.cl
    checks = dict(S.checks)
    stored_L = S.stored["arms"]["L"]
    local_cell, _ = rk.load_cell("leiden_k20_r1.0")
    checks["partition_L_equals_sweep_leiden_k20_r1.0"] = bool(np.array_equal(S.partition_L, local_cell))
    if not checks["partition_L_equals_sweep_leiden_k20_r1.0"]:
        raise SystemExit("partition_L differs from the sweep's leiden_k20_r1.0 partition: stopping")
    if not np.array_equal(hz["selection"], ctx.selection):
        raise AssertionError("heads were computed on another selection row set")

    rc.log("A3-model condition-free term T_N1u for B'")
    inp, _, _ = rc.model_inputs(ctx, "A3", S.scorer_train, False)
    t_n1u = centered_term(inp, ctx.pooled, uniform=True)
    del inp

    rc.log(f"affect L heads with fit_one_head ({rows} rows)")
    post_L, prov_L = rto.fit_one_head(ctx, rto.global_labels(S.partition_L, S.scorer_train, len(ctx.groups)),
                                      S.scorer_train, rows)
    if smoke:
        checks["fit_one_head_L_equals_told_oracle_head"] = "skipped (smoke: 3,000-row heads)"
    else:
        checks["fit_one_head_L_equals_told_oracle_head"] = bool(rto.roundtrip(prov_L) == stored_L["head"])
        if not checks["fit_one_head_L_equals_told_oracle_head"]:
            raise SystemExit(f"arm L's refit head {prov_L} differs from told_oracle.json {stored_L['head']}: stopping")
    post = {"affect": post_L, "image": S.post_stored["image"], "caption": S.post_stored["caption"]}
    for name in SETS[set_]:
        post[name] = {"img": full_post(hz, f"{name}__img", ctx), "txt": full_post(hz, f"{name}__txt", ctx)}
    secondary = {name: {"img": full_post(hz, f"{name}__img_src", ctx), "txt": post[name]["txt"]}
                 for name in SOURCE_HEADS if set_ == "style"}

    # ---------------------------------------------------------- the generalisation equals the original on A0
    rc.log("A0 with the original run_told_oracle.evaluate_arm (reference for the generalised version)")
    post_A0 = {h: post[h] for h in BASE}
    ev_o, arr_o, t6u_o = rto.evaluate_arm(post_A0, ctx, S.B, S.pB, S.part_idx)

    arms, npz, refs = {}, {}, {}
    for arm in ARMS_OF_SET[set_]:
        t = time.time()
        spec = ARMS[arm]
        rc.log(f"arm {arm}: groupings {spec['parts']}, told {spec['told']}"
               + (f", {spec['secondary']} image head on its source features" if spec["secondary"] else ""))
        p = {h: (secondary[h] if h == spec["secondary"] else post[h]) for h in spec["parts"]}
        ev, arr, t6u, picked, part_idx = evaluate_general(p, spec["parts"], spec["told"], ctx, S.B, S.pB)
        Bp, picks_Bp = crossfit_condition_free(ctx.cos, t_n1u, t6u, ctx.parity)
        pBp = per_anchor(Bp)
        bar_v, bar_info = bar_margin(arr, pBp, ctx)
        entry = {"groupings": list(spec["parts"]), "told_mapping": spec["told"],
                 "secondary_image_head": spec["secondary"], "eval": ev,
                 "fused_r1": {nm: {"fused": df.describe(arr[nm][0], cl), "counterpart": df.describe(arr[nm][1], cl)}
                              for nm in NAMES},
                 "B_prime": {"describe": df.describe(pBp, cl), "vs_B": df.diff(pBp, S.pB, cl), "picks": picks_Bp},
                 "bar": bar_info}
        if arm == "A0":
            ok = {f"{k}_equals_original": rto.roundtrip(ev[k]) == rto.roundtrip(ev_o[k]) for k in ("told", "reader", "pick")}
            ok["arrays_equal_original"] = all(np.array_equal(np.asarray(arr[nm][j][m]), np.asarray(arr_o[nm][j][m]))
                                              for nm in NAMES for j in (0, 1) for m in METRICS)
            ok["T6u_equal_original"] = all(np.array_equal(t6u[c][d], t6u_o[c][d]) for c in t6u for d in t6u[c])
            ok["part_idx_equal_original"] = all(np.array_equal(part_idx[c], S.part_idx[c]) for c in CONDITIONS)
            checks["generalised_evaluate_arm_equals_original_on_A0"] = ok
            if not all(ok.values()):
                print(json.dumps(ok, indent=1))
                raise SystemExit("the generalised evaluate_arm differs from the original on A0: stopping")
            if smoke:
                checks["A0_reproduces_told_oracle_L"] = "skipped (smoke: 3,000-row heads)"
            else:
                ok = {k: rto.roundtrip(ev[k]) == stored_L["eval"][k] for k in ("told", "reader", "pick")}
                ok["arrays"] = all(np.array_equal(np.asarray(arr[nm][j][m]), S.arr_L[nm][j][m])
                                   for nm in NAMES for j in (0, 1) for m in ("r1", "gain", "other"))
                ok["B_prime"] = rto.roundtrip(entry["B_prime"]) == stored_L["B_prime"]
                for nm, (pt, lo, hi) in L_STATED.items():
                    x = ev[nm]["fusedT_vs_fusedTcf"]["r1"]
                    ok[f"{nm}_is_{pt:+.2f}_[{lo:.2f},{hi:.2f}]"] = (round(x["point"], 2), round(x["ci95"][0], 2),
                                                                    round(x["ci95"][1], 2)) == (pt, lo, hi)
                ok[f"pick_is_{PICK_STATED}"] = round(ev["pick"]["correct_share"]["point"], 1) == PICK_STATED
                b = bar_info
                ok[f"bar_is_{A0_BAR_STATED[0]:+.2f}_({A0_BAR_STATED[1]}-{A0_BAR_STATED[2]})"] = (
                    b["comparator"] == "B_prime" and round(b["r1"]["point"], 2) == A0_BAR_STATED[0]
                    and round(b["reader_r1"], 2) == A0_BAR_STATED[1] and round(b["B_prime_r1"], 2) == A0_BAR_STATED[2])
                checks["A0_reproduces_told_oracle_L"] = ok
                if not all(ok.values()):
                    print(json.dumps(ok, indent=1))
                    raise SystemExit("A0 does not reproduce told-oracle arm L exactly: stopping (PLAN.md)")
                rc.log("A0 reproduces told-oracle arm L exactly (told, reader, pick, arrays, B', stated numbers, bar)")
            del ev_o, arr_o, t6u_o
        refs[arm] = (arr, bar_v)
        if arm != "A0":
            entry["vs_A0"] = paired(arr, refs["A0"][0], bar_v, refs["A0"][1], ctx)
        if arm in STYLE_ARMS:
            entry["vs_AR"] = paired(arr, refs["AR"][0], bar_v, refs["AR"][1], ctx)
        for nm in NAMES:
            for j, part in enumerate(("fused", "cf")):
                for m in METRICS:
                    npz[f"{arm}__{nm}__{part}__{m}"] = np.asarray(arr[nm][j][m])
        for m in METRICS:
            npz[f"{arm}__Bprime__{m}"] = np.asarray(pBp[m])
        for c in CONDITIONS:
            npz[f"{arm}__reader_pick__{c}"] = picked[c].astype(np.int8)
            npz[f"{arm}__told_index__{c}"] = part_idx[c].astype(np.int8)
        entry["runtime_s"] = round(time.time() - t, 1)
        arms[arm] = entry
        rc.log(f"arm {arm}: told {ci_txt(ev['told']['fusedT_vs_fusedTcf']['r1'])} reader "
               f"{ci_txt(ev['reader']['fusedT_vs_fusedTcf']['r1'])} bar {ci_txt(bar_info['r1'])} "
               f"({bar_info['comparator']}) pick {ev['pick']['correct_share']['point']:.1f} [{entry['runtime_s']:.0f}s]")
        del p

    npz.update(anchor_group=cl, pair_index=ctx.pair_index, partition_L=S.partition_L,
               **{f"B__{m}": np.asarray(S.pB[m]) for m in METRICS})
    result.update(n_episodes=int(ctx.n), n_clusters=int(len(np.unique(cl))), head_rows=int(rows), checks=checks,
                  B=df.describe(S.pB, cl), affect_L_head=prov_L, arms=arms, reading=readings(arms, set_),
                  runtime_s=round(time.time() - t0, 1))
    write(paths, result, text_eval, npz)


def text_eval(r):
    c = ci_txt
    L = [f"EXPLORATORY step 1, arms (PLAN.md sha {r['plan_sha256'][:12]}; set {r['set']}"
         f"{'; SMOKE (stand-in features, 3,000-row heads)' if r['smoke'] else ''}); seed 42 dev episodes "
         f"(n={r['n_episodes']}, {r['n_clusters']} paintings); 95% CIs resample paintings (5,000). Margin = fused T "
         "minus its counterpart fused T_cf, both on B (pp).",
         "Checks: " + "; ".join(f"{k} {v}" for k, v in r["checks"].items()),
         f"B (= stored C2): R@1 {c(r['B']['r1'])}", "",
         "1. Margins: R@1 | gain | either (fused T minus fused T_cf), fused T minus B (R@1); pick accuracy"]
    for a, x in r["arms"].items():
        e = x["eval"]
        L.append(f"  {a:4s} groupings {x['groupings']}; told {x['told_mapping']}"
                 + (f"; {x['secondary_image_head']} image head on source features" if x["secondary_image_head"] else ""))
        for nm in NAMES:
            m = e[nm]["fusedT_vs_fusedTcf"]
            L.append(f"       {nm:6s} {c(m['r1'])} | {c(m['gain'])} | {c(m['either'])} | vs B {c(e[nm]['fusedT_vs_B']['r1'])}"
                     f" (T_cf vs B {c(e[nm]['fusedTcf_vs_B']['r1'])}) | R@1 fused {x['fused_r1'][nm]['fused']['r1']['point']:.2f}"
                     f" / T_cf {x['fused_r1'][nm]['counterpart']['r1']['point']:.2f}")
        L.append(f"       pick {e['pick']['correct_share']['point']:.1f} [{e['pick']['correct_share']['ci95'][0]:.1f}, "
                 f"{e['pick']['correct_share']['ci95'][1]:.1f}], both correct {e['pick']['both_correct_share']:.1f}%")
    L += ["", "2. Per aspect pair, margin R@1 (told | reader) and pick accuracy a/b under the arm's told mapping"]
    for a, x in r["arms"].items():
        e = x["eval"]
        L.append(f"  {a:4s} " + " ; ".join(
            f"{df.PAIR_SHORT[p]} {e['told']['per_pair'][p]['margin']['r1']['point']:+.2f} | "
            f"{e['reader']['per_pair'][p]['margin']['r1']['point']:+.2f} | "
            f"{e['pick']['per_pair_condition'][p]['a']:.1f}/{e['pick']['per_pair_condition'][p]['b']:.1f}"
            for p in rg.POOLED_ORDER))
    L += ["", "3. Bar margin: fused reader R@1 minus max(B', counterpart) (paired); B' = B rebuilt with the arm's "
              "averaged-heads term"]
    for a, x in r["arms"].items():
        b = x["bar"]
        L.append(f"  {a:4s} bar {c(b['r1'])} = reader {b['reader_r1']:.2f} - {b['comparator']} "
                 f"(B' {b['B_prime_r1']:.2f}, counterpart {b['counterpart_r1']:.2f}); B' - B "
                 f"{c(x['B_prime']['vs_B']['r1'])}; per pair " + " / ".join(
                     f"{df.PAIR_SHORT[p]} {v['point']:+.2f}" for p, v in b["per_pair_r1"].items()))
    L += ["", "4. Paired differences (per anchor): told R@1 | reader R@1 | bar R@1 | told s×g R@1"]
    for a, x in r["arms"].items():
        for ref in ("A0", "AR"):
            v = x.get(f"vs_{ref}")
            if v:
                L.append(f"  {a:4s} - {ref}: {c(v['told']['r1'])} | {c(v['reader']['r1'])} | {c(v['bar_r1'])} | "
                         f"{c(v['per_pair_r1']['told'][SXG])}")
    rd = r["reading"]
    L += ["", "5. Readings (PLAN.md section 6, applied literally)"]
    a0 = rd["R2_reference_A0"]
    L.append(f"  A0 reference: bar {c(a0['bar_margin'])} ({a0['comparator']}), reader gain margin "
             f"{c(a0['reader_gain_margin'])}; R2 met: {a0['meets']}")
    if "R2_descriptive_A3" in rd:
        x = rd["R2_descriptive_A3"]
        L.append(f"  A3 (descriptive): bar {c(x['bar_margin'])} ({x['comparator']}), reader gain margin "
                 f"{c(x['reader_gain_margin'])}; R2 met: {x['meets']}")
    for key in ("R1", "R2", "R3"):
        for a, x in rd.get(key, {}).items():
            if key == "R1":
                L.append(f"  R1 {a:4s}: told s×g minus A0 {c(x['told_sxg_minus_A0'])} -> {x['meets']}")
            elif key == "R2":
                L.append(f"  R2 {a:4s}: bar {c(x['bar_margin'])} (>= 0.5 {x['bar_at_least_0.5']}, lower > 0 "
                         f"{x['bar_lower_above_0']}); reader gain margin {c(x['reader_gain_margin'])} (lower > 0 "
                         f"{x['gain_lower_above_0']}) -> {x['meets']}")
            else:
                L.append(f"  R3 {a:4s}: reader margin minus AR {c(x['reader_margin_minus_AR'])} -> {x['meets']}")
    if "default_for_9_october" in rd:
        L.append(f"  Default for 9 October: {rd['default_for_9_october']['text']}")
    L.append(f"runtime {r['runtime_s']:.0f}s")
    return "\n".join(L)


# ---------------------------------------------------------------- stage describe

def contrast(cl, la, lb, g):
    """Profile check 3's episode contrast for aspects A, B (scorer-train rows labelled for both; different-painting
    pairs): s_AB = P(same group | same A, different B), s_BA = P(same group | same B, different A)."""
    m = (la >= 0) & (lb >= 0)
    c, xa, xb, gg = cl[m], la[m], lb[m], g[m]
    dp = rto.dpairs
    nA = dp([xa], gg) - dp([xa, xb], gg)
    nB = dp([xb], gg) - dp([xa, xb], gg)
    sAB = (dp([xa, c], gg) - dp([xa, xb, c], gg)) / nA
    sBA = (dp([xb, c], gg) - dp([xa, xb, c], gg)) / nB
    return {"s_AB": sAB, "s_BA": sBA, "diff": sAB - sBA, "ratio": sAB / sBA, "base": dp([c], gg) / dp([], gg)}


def ami(a, b, mask=None):
    if mask is not None:
        a, b = a[mask], b[mask]
    return float(adjusted_mutual_info_score(a, b))


def stage_describe(set_, smoke):
    paths = start("describe", set_, smoke)
    t0 = time.time()
    grec, gz, ginputs = load_stage("group", set_, smoke)
    hrec, hz, hinputs = load_stage("heads", set_, smoke)
    result = header("describe", set_, smoke, ginputs + hinputs + [n6.PARTITIONS, rsw.STORED_NPZ, df.POSTERIORS])
    data, sp, scorer_train, groups, g_st = base_data()
    if not np.array_equal(scorer_train, gz["scorer_train"]):
        raise AssertionError("scorer_train differs from the group stage's")
    checks = {}
    e2 = rk.e2_partitions(groups, scorer_train)
    checks["E2_partitions_sha_and_painting_alignment"] = True
    part_L = np.asarray(np.load(rsw.STORED_NPZ)["partition_L"], dtype=np.int64)
    checks["partition_L_equals_sweep_leiden_k20_r1.0"] = bool(np.array_equal(part_L, rk.load_cell("leiden_k20_r1.0")[0]))
    if not checks["partition_L_equals_sweep_leiden_k20_r1.0"]:
        raise AssertionError("partition_L differs from the sweep's leiden_k20_r1.0")
    refs = {"E2_image": e2["image"], "E2_caption": e2["caption"], "affect_L": part_L}
    sel = np.asarray(sp.selection)
    if not np.array_equal(hz["selection"], sel):
        raise AssertionError("heads were computed on another selection row set")
    g_sel = groups[sel]

    # ---------------------------------------------------------- label-free diagnostics (section 5)
    out = {}
    for name in SETS[set_]:
        local = np.asarray(gz[name], dtype=np.int64)
        gr = grec["groupings"][name]
        if rg.sha_array(local) != gr["partition_sha256"]:
            raise AssertionError(f"{name}: partition SHA-256 differs from the group JSON")
        x = {"groups": gr["groups"], "ami_with": {k: ami(local, v) for k, v in refs.items()}}
        if name == "style_rand":
            x["stability"] = "not applicable (a permutation of style_csd's labels)"
            x["ami_with"]["style_csd"] = ami(local, np.asarray(gz["style_csd"], np.int64))
        else:
            parts_s = {LEIDEN_SEED: local, **{s: np.asarray(gz[f"{name}__seed{s}"], np.int64)
                                              for s in STAB_SEEDS if s != LEIDEN_SEED}}
            pairs = [(a, b) for i, a in enumerate(STAB_SEEDS) for b in STAB_SEEDS[i + 1:]]
            pw = {f"{a}-{b}": ami(parts_s[a], parts_s[b]) for a, b in pairs}
            x["stability"] = {"pairwise_ami": pw, "mean_pairwise_ami": float(np.mean(list(pw.values()))),
                              "n_groups_per_seed": {str(s): int(v.max()) + 1 for s, v in parts_s.items()}}
        h = hrec["heads"][name]["clip"]
        x["head"] = {"heldout_accuracy": h["heldout_accuracy"], "check_majority_share": h["check_majority_share"],
                     "uniform": h["uniform"], "n_classes": h["n_classes"]}
        x["placeability"] = rk.placeability(hz[f"{name}__img"], hz[f"{name}__txt"], g_sel)
        if name in SOURCE_HEADS:
            hs = hrec["heads"][name]["source_image"]
            x["source_image_head"] = {"heldout_accuracy_img": hs["heldout_accuracy"]["img"],
                                      "check_majority_share": hs["check_majority_share"],
                                      "placeability": rk.placeability(hz[f"{name}__img_src"], hz[f"{name}__txt"], g_sel)}
        out[name] = x
        rc.log(f"{name}: {x['groups']['n_groups']} groups; AMI " + ", ".join(f"{k} {v:.4f}" for k, v in x["ami_with"].items())
               + f"; P_ami {x['placeability']['P_ami']:.4f}")
    zpost = np.load(df.POSTERIORS)
    if not np.array_equal(zpost["selection"], sel):
        raise AssertionError("stored posteriors were computed on another selection row set")
    references = {h: {"source": "stored N6 posteriors (E2 k-means 64)",
                      **rk.placeability(zpost[f"{h}__img"], zpost[f"{h}__txt"], g_sel)} for h in ("image", "caption")}
    references["ami_E2_image_vs_E2_caption"] = ami(e2["image"], e2["caption"])

    # ---------------------------------------------------------- the one disclosed label description (after eval)
    label = None
    if set_ == "style":
        for s in ("descriptive", "style"):
            if not out_paths("eval", s, smoke)[1]["json"].exists():
                raise SystemExit(f"the label description comes after the arms: eval --set {s} has not run")
        drec, dz, dinputs = load_stage("group", "descriptive", smoke)
        for p in dinputs + [out_paths("eval", s, smoke)[1]["json"] for s in ("descriptive", "style")] + [rto.PROFILE]:
            result["provenance"]["inputs_sha256"][rel(p)] = rg.sha_file(p)
        lab_all = artelingo_aspect_labels(data)
        lab = {a: np.asarray(lab_all[a][scorer_train], dtype=np.int64) for a in ("emotion", "style", "genre")}
        prof = json.loads(rto.PROFILE.read_text())
        ref_c = prof["check3"]["contrast"]["image|stylexgenre"]
        mine = contrast(e2["image"], lab["style"], lab["genre"], g_st)
        ok_prof = all(np.isclose(mine[k], ref_c[k], rtol=1e-9, atol=0) for k in ("s_AB", "s_BA", "ratio"))
        mine_es = contrast(part_L, lab["emotion"], lab["style"], g_st)
        ref_es = rto.pair_stats_groups(part_L, lab, g_st)["contrast"]["emotionxstyle"]
        ok_rto = all(np.isclose(mine_es[k], ref_es[k], rtol=1e-12, atol=0) for k in ("s_AB", "s_BA", "ratio"))
        checks["contrast_copy_reproduces_profile_image_style_x_genre"] = bool(ok_prof)
        checks["contrast_copy_equals_pair_stats_groups_L_emotion_x_style"] = bool(ok_rto)
        if not (ok_prof and ok_rto):
            raise AssertionError("the style x genre contrast copy does not reproduce check 3")
        news = {"image_leiden": dz["image_leiden"], "caption_leiden": dz["caption_leiden"],
                **{n: gz[n] for n in SETS["style"]}}
        genre_ok = lab["genre"] >= 0
        label = {"rows": int(len(scorer_train)), "rows_with_genre": int(genre_ok.sum()),
                 "rows_with_style": int((lab["style"] >= 0).sum()), "groupings": {}, "references": {}}
        for kind, coll in (("groupings", news), ("references", refs)):
            for nm, v in coll.items():
                v = np.asarray(v, dtype=np.int64)
                label[kind][nm] = {"ami_style": ami(v, lab["style"], lab["style"] >= 0),
                                   "ami_genre": ami(v, lab["genre"], genre_ok),
                                   "style_x_genre": contrast(v, lab["style"], lab["genre"], g_st)}
        rc.log("label description: " + "; ".join(
            f"{nm} AMI style {x['ami_style']:.4f} genre {x['ami_genre']:.4f} ratio {x['style_x_genre']['ratio']:.3f}"
            for nm, x in label["groupings"].items()))

    result.update(checks=checks, groupings=out, references=references, label_description=label,
                  runtime_s=round(time.time() - t0, 1))
    write(paths, result, text_describe)


def text_describe(r):
    L = [f"EXPLORATORY step 1, diagnostics of the new groupings (PLAN.md sha {r['plan_sha256'][:12]}; set {r['set']}"
         f"{'; SMOKE' if r['smoke'] else ''}); label free unless marked; AMI sklearn defaults on scorer-train rows",
         "Checks: " + "; ".join(f"{k} {v}" for k, v in r["checks"].items()), "",
         f"  {'grouping':15s} {'grp':>4s} {'min':>6s} {'max':>6s} {'effN':>6s} | AMI E2img E2cap affL | seed AMI "
         f"(42-43/42-44/43-44; mean) | held-out img/txt (majority) | P_ami (P_lift)"]
    for nm, x in r["groupings"].items():
        g, a, h, p = x["groups"], x["ami_with"], x["head"], x["placeability"]
        st = x["stability"]
        st_txt = (" / ".join(f"{v:.3f}" for v in st["pairwise_ami"].values()) + f"; {st['mean_pairwise_ami']:.3f}"
                  if isinstance(st, dict) else "n/a")
        L.append(f"  {nm:15s} {g['n_groups']:4d} {g['min']:6d} {g['max']:6d} {g['effective_n']:6.1f} | "
                 f"{a['E2_image']:.4f} {a['E2_caption']:.4f} {a['affect_L']:.4f} | {st_txt} | "
                 f"{h['heldout_accuracy']['img']:.2f}/{h['heldout_accuracy']['txt']:.2f} ({h['check_majority_share']:.2f}) | "
                 f"{p['P_ami']:.4f} ({p['P_lift']:.3f})")
        if "source_image_head" in x:
            s = x["source_image_head"]
            L.append(f"  {'':15s} source-feature image head: held-out {s['heldout_accuracy_img']:.2f} (majority "
                     f"{s['check_majority_share']:.2f}); P_ami {s['placeability']['P_ami']:.4f} "
                     f"(P_lift {s['placeability']['P_lift']:.3f})")
        if "style_csd" in a:
            L.append(f"  {'':15s} AMI with style_csd {a['style_csd']:.4f}")
    rf = r["references"]
    L.append("  references (stored N6 posteriors): " + "; ".join(
        f"{h} P_ami {rf[h]['P_ami']:.4f} (P_lift {rf[h]['P_lift']:.3f})" for h in ("image", "caption"))
        + f"; AMI(E2 image, E2 caption) {rf['ami_E2_image_vs_E2_caption']:.4f}")
    lab = r.get("label_description")
    if lab:
        L += ["", f"Label description (disclosed, computed after the arms, chooses nothing): AMI with style (all "
                  f"{lab['rows_with_style']} rows) and genre ({lab['rows_with_genre']} labelled rows); style x genre "
                  "contrast s_AB = P(same group | same style, diff genre), s_BA = P(same group | same genre, diff style)"]
        for kind in ("groupings", "references"):
            for nm, x in lab[kind].items():
                cg = x["style_x_genre"]
                L.append(f"  {nm:15s} AMI style {x['ami_style']:.4f} genre {x['ami_genre']:.4f} | s_AB {cg['s_AB']:.4f} "
                         f"s_BA {cg['s_BA']:.4f} ratio {cg['ratio']:.3f}{' (reference)' if kind == 'references' else ''}")
    L.append(f"runtime {r['runtime_s']:.0f}s")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("group", "heads", "eval", "describe"))
    ap.add_argument("--set", dest="set_", required=True, choices=tuple(SETS))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    {"group": stage_group, "heads": stage_heads, "eval": stage_eval, "describe": stage_describe}[args.stage](
        args.set_, args.smoke)


if __name__ == "__main__":
    main()
