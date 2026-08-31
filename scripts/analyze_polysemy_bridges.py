"""
Experiment 12 (spec docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md,
Experiment 12 subsection): cross-modal polysemy bridge-node diagnostic.

Does the existing buddy-graph construction already implicitly reflect cross-modal
polysemy -- a bridge node A connected to B via an image-only mutual-kNN edge and to C via
a text-only one -- and does the buddy-init spectral embedding place B and C closer
together than a degree-matched random baseline? If so, is that pull graded by real
shared-neighbor structure (legitimate signal), or flat/arbitrary ("false transitivity",
the risk Experiment 10's own diagnostic flagged but never measured)? Separately: does a
per-node polysemy label predict anything about per-sample retrieval rank / condition
drift (reusing Experiment 11.2's per-sample outputs)?

No new training, no new graph-construction mechanism -- reuses classify_edges/
bridge_node_stats (src/conditional_buddy/buddy_graph.py) and an already-completed
buddy-init template.

Usage
-----
  python scripts/analyze_polysemy_bridges.py --selftest
  python scripts/analyze_polysemy_bridges.py \\
      --storage-dir /data/SSD2/pre_extract/redcaps_150k/features \\
      --template-dir res/CoSiR_condition_freeze_ablation/redcaps_150k/template_embeddings \\
      --n-bridge-sample 5000

Requires: numpy, scipy (all already deps).
"""
import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import rankdata

import os
import sys

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
_ROOT = os.path.abspath(os.path.join(_SCRIPTS_DIR, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from analyze_condition_retrieval_correlation import spearman_correlate


def degree_deciles(E: csr_matrix, n_buckets: int = 10) -> np.ndarray:
    """Per-node degree-decile bucket id (0 = lowest degree, n_buckets-1 = highest),
    used to sample a degree-matched baseline node for the false-transitivity check."""
    degree = np.diff(E.indptr)
    ranks = rankdata(degree, method="average") / len(degree)  # in (0, 1]
    buckets = np.minimum((ranks * n_buckets).astype(np.int64), n_buckets - 1)
    return buckets


def sample_baselines(
    pairs: np.ndarray,
    E: csr_matrix,
    buckets: np.ndarray,
    rng: np.random.Generator,
    max_tries: int = 50,
) -> np.ndarray:
    """For each (A, B, C) row, sample a degree-bucket-matched C' that is NOT a direct
    E-neighbor of B and not equal to B or C -- the "is B pulled toward C specifically, or
    just toward any similarly-connected node" baseline. -1 where no candidate was found
    within max_tries (excluded from downstream stats by the caller)."""
    node_ids_by_bucket = {k: np.where(buckets == k)[0] for k in np.unique(buckets)}
    out = np.full(len(pairs), -1, dtype=np.int64)
    for row in range(len(pairs)):
        a, b, c = pairs[row]
        candidates = node_ids_by_bucket[buckets[c]]
        b_neighbors = set(E.indices[E.indptr[b]:E.indptr[b + 1]].tolist())
        for _ in range(max_tries):
            c_prime = int(rng.choice(candidates))
            if c_prime != b and c_prime != c and c_prime not in b_neighbors:
                out[row] = c_prime
                break
    return out


def shared_neighbor_jaccard(E: csr_matrix, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Jaccard overlap of each (b[i], c[i]) pair's neighbor sets in E -- the LINE/GraRep-
    style second-order-proximity score used to test whether any embedded pull is graded
    by real shared-neighbor structure. Loop-based: fine at the capped sample size
    (~thousands of pairs) this experiment uses."""
    out = np.empty(len(b), dtype=np.float64)
    for idx in range(len(b)):
        nb = set(E.indices[E.indptr[b[idx]]:E.indptr[b[idx] + 1]].tolist())
        nc = set(E.indices[E.indptr[c[idx]]:E.indptr[c[idx] + 1]].tolist())
        union = len(nb | nc)
        out[idx] = len(nb & nc) / union if union > 0 else 0.0
    return out


def embedded_l2_distance(emb: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Euclidean distance between rows i and j of the buddy-init embedding -- matches
    this project's existing condition_drift L2 convention."""
    return np.linalg.norm(emb[i] - emb[j], axis=1)


def paired_pull_summary(dist_bc: np.ndarray, dist_bc_baseline: np.ndarray) -> dict:
    """Paired difference (baseline - bridge_pair): positive means the bridge-derived
    (B, C) pair sits CLOSER together in the embedding than its degree-matched baseline
    pair, i.e. a 'pull'. Same mean/std/sem/z convention as this project's other
    paired-delta analysis scripts."""
    pull = dist_bc_baseline - dist_bc
    n = len(pull)
    mean = float(pull.mean())
    std = float(pull.std(ddof=1)) if n > 1 else float("nan")
    sem = std / np.sqrt(n) if n > 1 and std == std else float("nan")
    z = mean / sem if sem == sem and sem > 0 else float("nan")
    return {
        "n": n, "mean": mean, "std": std, "sem": sem, "z": z,
        "frac_pulled_closer": float((pull > 0).mean()),
    }


def build_typed_adjacency(typed: dict, N: int) -> Tuple[csr_matrix, csr_matrix]:
    """Symmetric binary adjacency for just the img-only and just the txt-only edges of
    the union graph, so per-node neighbor lists can be sliced by modality-provenance
    type without re-scanning classify_edges's flat edge list each time."""
    keys = typed["keys"]
    i = (keys // N).astype(np.int64)
    j = (keys % N).astype(np.int64)

    def _sym(mask):
        ii, jj = i[mask], j[mask]
        rows = np.concatenate([ii, jj])
        cols = np.concatenate([jj, ii])
        data = np.ones(len(rows), dtype=np.float32)
        return csr_matrix((data, (rows, cols)), shape=(N, N))

    return _sym(typed["img_only"]), _sym(typed["txt_only"])


def extract_bridge_pairs(
    bridge_stats: dict,
    E_img_only: csr_matrix,
    E_txt_only: csr_matrix,
    n_sample: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """For up to n_sample bridge nodes, pick one random img-only neighbor (B) and one
    random txt-only neighbor (C). Returns int64 array of shape (M, 3): columns [A, B, C].
    B != C always -- a node's img-only and txt-only neighbor sets are disjoint by
    construction (classify_edges assigns each edge to exactly one bucket)."""
    bridge_ids = np.where(bridge_stats["is_bridge"])[0]
    if len(bridge_ids) > n_sample:
        bridge_ids = rng.choice(bridge_ids, size=n_sample, replace=False)

    triples = []
    for a in bridge_ids:
        b_candidates = E_img_only.indices[E_img_only.indptr[a]:E_img_only.indptr[a + 1]]
        c_candidates = E_txt_only.indices[E_txt_only.indptr[a]:E_txt_only.indptr[a + 1]]
        if len(b_candidates) == 0 or len(c_candidates) == 0:
            continue
        b = int(rng.choice(b_candidates))
        c = int(rng.choice(c_candidates))
        triples.append((int(a), b, c))
    if not triples:
        return np.zeros((0, 3), dtype=np.int64)
    return np.array(triples, dtype=np.int64)


def extract_hub_pairs(
    typed: dict,
    bridge_stats: dict,
    N: int,
    n_sample_per_group: int,
    rng: np.random.Generator,
) -> dict:
    """Experiment 14: sample up to n_sample_per_group closed-triangle pairs and up to
    n_sample_per_group open hub pairs INDEPENDENTLY (see buddy_graph.hub_neighbor_pairs
    for the closed/open definition), so a sparse closed-triangle population isn't
    crowded out by open pairs if both groups exist in very different numbers.

    Returns {"pairs": int64 (M, 3) array of columns [hub, c, d], "is_closed": bool (M,)}."""
    from src.conditional_buddy.buddy_graph import hub_neighbor_pairs

    raw = hub_neighbor_pairs(typed, bridge_stats, N)
    if len(raw["hub"]) == 0:
        return {"pairs": np.zeros((0, 3), dtype=np.int64), "is_closed": np.zeros(0, dtype=bool)}

    pairs_all = np.stack([raw["hub"], raw["c"], raw["d"]], axis=1)
    is_closed_all = raw["is_closed"]

    kept_pairs, kept_closed = [], []
    for want_closed in (True, False):
        idx = np.where(is_closed_all == want_closed)[0]
        if len(idx) > n_sample_per_group:
            idx = rng.choice(idx, size=n_sample_per_group, replace=False)
        kept_pairs.append(pairs_all[idx])
        kept_closed.append(np.full(len(idx), want_closed, dtype=bool))

    return {
        "pairs": np.concatenate(kept_pairs, axis=0),
        "is_closed": np.concatenate(kept_closed, axis=0),
    }


def label_nodes(bridge_stats: dict) -> np.ndarray:
    """Per-node polysemy label from classify_edges/bridge_node_stats's own per-node
    degree counts: 'bridge' (has both an img-only AND a txt-only edge -- the "A" node in
    Experiment 12's A/B/C example), 'img_only_only' / 'txt_only_only' (has edges of only
    one such type), or 'neither' (only 'both'/'repair' edges, or no edges of these types
    at all)."""
    deg_img = bridge_stats["deg_img_only"]
    deg_txt = bridge_stats["deg_txt_only"]
    is_bridge = bridge_stats["is_bridge"]
    labels = np.full(len(deg_img), "neither", dtype="<U16")
    labels[(deg_img > 0) & ~is_bridge] = "img_only_only"
    labels[(deg_txt > 0) & ~is_bridge] = "txt_only_only"
    labels[is_bridge] = "bridge"
    return labels


def closed_triangle_membership(hub_pairs: dict, N: int) -> Tuple[np.ndarray, np.ndarray]:
    """Per-node boolean flags derived from hub_neighbor_pairs' RAW (unsampled) output:
    in_closed_triangle is True for any node appearing as the c/d endpoint of at least
    one closed pair; in_open_hub_pair is True for any node appearing as the c/d endpoint
    of at least one open pair. NOT mutually exclusive with each other, or with
    label_nodes' categories -- a node with 3+ txt_only neighbors can be in both groups."""
    in_closed = np.zeros(N, dtype=bool)
    in_open = np.zeros(N, dtype=bool)
    closed_mask = hub_pairs["is_closed"]
    for endpoints in (hub_pairs["c"], hub_pairs["d"]):
        in_closed[endpoints[closed_mask]] = True
        in_open[endpoints[~closed_mask]] = True
    return in_closed, in_open


def _selftest():
    # label_nodes: 4 nodes -- one bridge, one img-only-only, one txt-only-only, one bare.
    bridge_stats = {
        "deg_img_only": np.array([2, 1, 0, 0]),
        "deg_txt_only": np.array([1, 0, 3, 0]),
        "is_bridge": np.array([True, False, False, False]),
    }
    labels = label_nodes(bridge_stats)
    assert list(labels) == ["bridge", "img_only_only", "txt_only_only", "neither"], labels

    # build_typed_adjacency + extract_bridge_pairs, on the same synthetic graph as
    # buddy_graph's own bridge-node test: node 1 is a bridge (img-only to 0, txt-only to 4).
    from scipy.sparse import csr_matrix as _csr
    from src.conditional_buddy.buddy_graph import bridge_node_stats, classify_edges, hub_neighbor_pairs

    def _sym(n, edges):
        rows, cols = [], []
        for i, j in edges:
            rows += [i, j]; cols += [j, i]
        return _csr((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n))

    n = 5
    A_img = _sym(n, [(0, 1), (2, 3)])
    A_txt = _sym(n, [(1, 4), (2, 3)])
    E = _sym(n, [(0, 1), (1, 4), (2, 3)])
    typed = classify_edges(A_img, A_txt, E, n)
    bstats = bridge_node_stats(typed, n)

    E_img_only, E_txt_only = build_typed_adjacency(typed, n)
    assert E_img_only[0, 1] == 1 and E_img_only[1, 0] == 1
    assert E_txt_only[1, 4] == 1 and E_txt_only[4, 1] == 1
    assert E_img_only[2, 3] == 0 and E_txt_only[2, 3] == 0  # (2,3) is a "both" edge

    rng = np.random.default_rng(0)
    pairs = extract_bridge_pairs(bstats, E_img_only, E_txt_only, n_sample=10, rng=rng)
    assert pairs.shape == (1, 3), pairs.shape
    a, b, c = pairs[0]
    assert a == 1 and b == 0 and c == 4, pairs

    # degree_deciles: 10 nodes with distinct degrees -> deciles 0..9 in order.
    n10 = 10
    edges10 = [(0, k) for k in range(1, 10)]  # node 0 has degree 9; nodes 1-9 have degree 1
    E10 = _sym(n10, edges10)
    buckets = degree_deciles(E10, n_buckets=10)
    assert buckets[0] == 9, buckets  # highest degree -> top decile
    assert buckets[1] < buckets[0], buckets

    # sample_baselines: node 5 (degree 1, bucket low) should never return node 0 (a direct
    # neighbor of b=1) or nodes with a very different degree.
    pairs10 = np.array([[0, 1, 5]])  # a=0, b=1 (neighbor of 0), c=5
    rng10 = np.random.default_rng(1)
    baselines = sample_baselines(pairs10, E10, buckets, rng10)
    assert baselines.shape == (1,)
    assert baselines[0] not in (1, 5, -1), baselines  # not b, not c, and a candidate WAS found

    # shared_neighbor_jaccard: b and c share exactly node 0 as a neighbor (of 9 total).
    jac = shared_neighbor_jaccard(E10, np.array([1]), np.array([2]))
    assert abs(jac[0] - (1 / 1)) < 1e-9, jac  # N(1)={0}, N(2)={0} -> intersection=union=1

    # embedded_l2_distance: known Euclidean distances.
    emb = np.array([[0.0, 0.0], [3.0, 4.0], [0.0, 0.0]])
    d = embedded_l2_distance(emb, np.array([0, 1]), np.array([1, 2]))
    assert np.allclose(d, [5.0, 5.0]), d

    # paired_pull_summary: baseline consistently 1.0 farther than the bridge pair ->
    # mean pull exactly 1.0, all wins.
    dist_bc = np.array([1.0, 1.0, 1.0])
    dist_baseline = np.array([2.0, 2.0, 2.0])
    summary = paired_pull_summary(dist_bc, dist_baseline)
    assert summary["n"] == 3
    assert abs(summary["mean"] - 1.0) < 1e-9, summary
    assert summary["frac_pulled_closer"] == 1.0, summary

    # correlate_polysemy_with_retrieval: 4 nodes, sample_ids in FeatureManager order;
    # dump covers only 3 of them (id 103 missing -> must be excluded, not crash).
    import tempfile
    labels4 = np.array(["bridge", "neither", "img_only_only", "neither"])
    sample_ids4 = [100, 101, 102, 103]
    with tempfile.TemporaryDirectory() as tmp:
        npz_path = os.path.join(tmp, "dump.npz")
        np.savez(
            npz_path,
            sample_ids=np.array([100, 101, 102], dtype=np.int64),
            delta_rank=np.array([10, 0, -5]),
            delta_rank_swap=np.array([8, 0, -3]),
            condition_drift=np.array([0.5, 0.1, 0.2]),
            embedding_shift=np.array([0.05, 0.01, 0.02]),
        )
        result = correlate_polysemy_with_retrieval(labels4, sample_ids4, npz_path)
    assert result["n_joined"] == 3, result
    assert result["bridge"]["n"] == 1 and result["bridge"]["median_abs_delta_rank"] == 10.0, result
    assert result["neither"]["n"] == 1  # only sample 101 (id 103 was excluded, not in the dump)
    assert "corr_is_polysemic_vs_abs_delta_rank" in result, result
    assert result["bridge"]["median_delta_rank"] == 10.0, result  # only one bridge row (id 100, delta_rank=10)
    assert "corr_is_polysemic_vs_delta_rank" in result, result

    # pool_cross_references: two runs' cross-reference results -> pooled mean/std/sem/z
    # of each run's own rho, matching the project's standard multi-seed convention.
    r1 = {"corr_is_polysemic_vs_abs_delta_rank": {"rho": 0.10, "p": 0.01},
          "corr_is_polysemic_vs_delta_rank": {"rho": 0.02, "p": 0.5}, "n_joined": 100}
    r2 = {"corr_is_polysemic_vs_abs_delta_rank": {"rho": 0.20, "p": 0.01},
          "corr_is_polysemic_vs_delta_rank": {"rho": -0.02, "p": 0.5}, "n_joined": 90}
    pooled = pool_cross_references([r1, r2], tags=["trained/seed1", "trained/seed2"])
    assert pooled["n_runs"] == 2
    assert set(pooled["per_run"].keys()) == {"trained/seed1", "trained/seed2"}
    abs_stats = pooled["pooled"]["corr_is_polysemic_vs_abs_delta_rank"]
    assert abs(abs_stats["mean"] - 0.15) < 1e-9, abs_stats
    assert abs_stats["n"] == 2

    # save_raw_arrays: always saves pair arrays, and includes retrieval arrays only
    # when the optional cross-reference was requested.
    with tempfile.TemporaryDirectory() as tmp:
        raw_path = os.path.join(tmp, "raw.npz")
        save_raw_arrays(
            raw_path,
            a_idx=np.array([1, 2]), b_idx=np.array([3, 4]), c_idx=np.array([5, 6]),
            dist_bc=np.array([0.1, 0.2]), dist_bc_baseline=np.array([0.3, 0.4]),
            jaccard=np.array([0.0, 0.5]),
            retrieval_raw={
                "label_kept": np.array(["bridge", "neither"]),
                "delta_rank": np.array([2.0, -3.0]),
            },
        )
        raw = np.load(raw_path)
        assert set(raw.files) == {
            "a_idx", "b_idx", "c_idx", "dist_bc", "dist_bc_baseline", "jaccard",
            "label_kept", "delta_rank",
        }, raw.files
        assert raw["label_kept"].tolist() == ["bridge", "neither"]

    # extract_hub_pairs + closed_triangle_membership: reuse the same synthetic
    # closed/open hub structure as buddy_graph's own hub_neighbor_pairs test.
    n8 = 8
    A_img8 = _sym(n8, [(0, 1), (2, 5)])
    A_txt8 = _sym(n8, [(1, 2), (1, 5), (6, 3), (6, 4), (7, 0)])
    E8 = _sym(n8, [(0, 1), (2, 5), (1, 2), (1, 5), (6, 3), (6, 4), (7, 0)])
    typed8 = classify_edges(A_img8, A_txt8, E8, n8)
    bstats8 = bridge_node_stats(typed8, n8)
    raw_hub_pairs = hub_neighbor_pairs(typed8, bstats8, n8)
    assert len(raw_hub_pairs["hub"]) == 2, raw_hub_pairs  # one closed pair, one open pair

    rng8 = np.random.default_rng(2)
    sampled = extract_hub_pairs(typed8, bstats8, n8, n_sample_per_group=10, rng=rng8)
    assert sampled["pairs"].shape == (2, 3), sampled["pairs"].shape
    assert set(sampled["is_closed"].tolist()) == {True, False}, sampled["is_closed"]

    in_closed, in_open = closed_triangle_membership(raw_hub_pairs, n8)
    assert in_closed[2] and in_closed[5], in_closed  # the closed triangle's C/D
    assert not in_closed[3] and not in_closed[4], in_closed  # the open pair's C/D
    assert in_open[3] and in_open[4], in_open
    assert not in_open[2] and not in_open[5], in_open
    print("PASS extract_hub_pairs + closed_triangle_membership")

    # correlate_polysemy_with_retrieval + pool_cross_references: extra_flags param
    # (Experiment 14's is_hub/in_closed_triangle/in_open_hub_pair cross-references).
    extra_flags4 = {"is_hub": np.array([True, False, True, False])}
    with tempfile.TemporaryDirectory() as tmp:
        npz_path4 = os.path.join(tmp, "dump4.npz")
        np.savez(
            npz_path4,
            sample_ids=np.array([100, 101, 102], dtype=np.int64),
            delta_rank=np.array([10, 0, -5]),
            delta_rank_swap=np.array([8, 0, -3]),
            condition_drift=np.array([0.5, 0.1, 0.2]),
            embedding_shift=np.array([0.05, 0.01, 0.02]),
        )
        result_flagged = correlate_polysemy_with_retrieval(
            labels4, sample_ids4, npz_path4, extra_flags=extra_flags4
        )
    assert "corr_is_hub_vs_delta_rank" in result_flagged, result_flagged
    assert "corr_is_hub_vs_abs_delta_rank" in result_flagged, result_flagged
    assert result_flagged["is_hub_true"]["n"] == 2, result_flagged  # sample ids 100, 102

    r1f = {**r1, "corr_is_hub_vs_abs_delta_rank": {"rho": 0.30, "p": 0.01},
           "corr_is_hub_vs_delta_rank": {"rho": 0.05, "p": 0.5}}
    r2f = {**r2, "corr_is_hub_vs_abs_delta_rank": {"rho": 0.50, "p": 0.01},
           "corr_is_hub_vs_delta_rank": {"rho": -0.05, "p": 0.5}}
    pooled_flagged = pool_cross_references(
        [r1f, r2f], tags=["trained/seed1", "trained/seed2"], extra_flag_names=["is_hub"]
    )
    assert "corr_is_hub_vs_abs_delta_rank" in pooled_flagged["pooled"], pooled_flagged
    assert abs(pooled_flagged["pooled"]["corr_is_hub_vs_abs_delta_rank"]["mean"] - 0.40) < 1e-9, pooled_flagged
    print("PASS correlate_polysemy_with_retrieval/pool_cross_references extra_flags")
    print("SELFTEST OK")


def correlate_polysemy_with_retrieval(
    labels: np.ndarray, sample_ids: List[int], npz_path: str, return_raw: bool = False,
    extra_flags: dict = None,
) -> dict:
    """Join the per-node polysemy label (row-aligned to sample_ids, this script's own
    FeatureManager-order labeling) against Experiment 11.2's per-sample retrieval-rank/
    drift dump (Task 2's .npz, keyed by actual sample id -- a different population, the
    training rows of one specific trained run, so only the intersection is used). Reports
    per-label median |delta_rank|/condition_drift/embedding_shift, plus Spearman
    correlations between "is this sample polysemic at all" and each retrieval-side metric.
    """
    data = np.load(npz_path)
    dump_ids = data["sample_ids"].tolist()
    id_to_row = {sid: row for row, sid in enumerate(sample_ids)}
    keep = [i for i, sid in enumerate(dump_ids) if sid in id_to_row]
    rows = np.array([id_to_row[dump_ids[i]] for i in keep], dtype=np.int64)

    label_kept = labels[rows]
    delta_rank = data["delta_rank"][keep].astype(float)
    condition_drift = data["condition_drift"][keep].astype(float)
    embedding_shift = data["embedding_shift"][keep].astype(float)
    is_polysemic = (label_kept != "neither").astype(float)

    result: dict = {"n_joined": len(keep)}
    for lbl in ("neither", "img_only_only", "txt_only_only", "bridge"):
        mask = label_kept == lbl
        if mask.sum() == 0:
            continue
        result[lbl] = {
            "n": int(mask.sum()),
            "median_abs_delta_rank": float(np.median(np.abs(delta_rank[mask]))),
            "median_delta_rank": float(np.median(delta_rank[mask])),
            "median_condition_drift": float(np.median(condition_drift[mask])),
            "median_embedding_shift": float(np.median(embedding_shift[mask])),
        }
    result["corr_is_polysemic_vs_abs_delta_rank"] = spearman_correlate(is_polysemic, np.abs(delta_rank))
    result["corr_is_polysemic_vs_delta_rank"] = spearman_correlate(is_polysemic, delta_rank)
    result["corr_is_polysemic_vs_condition_drift"] = spearman_correlate(is_polysemic, condition_drift)
    result["corr_is_polysemic_vs_embedding_shift"] = spearman_correlate(is_polysemic, embedding_shift)
    if extra_flags:
        for flag_name, flag_arr in extra_flags.items():
            flag_kept = flag_arr[rows].astype(float)
            result[f"corr_{flag_name}_vs_abs_delta_rank"] = spearman_correlate(flag_kept, np.abs(delta_rank))
            result[f"corr_{flag_name}_vs_delta_rank"] = spearman_correlate(flag_kept, delta_rank)
            mask_true = flag_kept > 0
            if mask_true.sum() > 0:
                result[f"{flag_name}_true"] = {
                    "n": int(mask_true.sum()),
                    "median_abs_delta_rank": float(np.median(np.abs(delta_rank[mask_true]))),
                    "median_delta_rank": float(np.median(delta_rank[mask_true])),
                }
    if return_raw:
        return result, {"label_kept": label_kept, "delta_rank": delta_rank}
    return result


def pool_cross_references(results: List[dict], tags: List[str], extra_flag_names: List[str] = None) -> dict:
    """Pool multiple already-computed correlate_polysemy_with_retrieval() results (one
    per run/seed) into a per-run table plus mean/std/sem/z of each run's own rho, across
    runs -- this project's standard multi-seed synthesis convention (see summarize() in
    scripts/analyze_condition_freeze_ablation.py). extra_flag_names pools the same
    corr_{name}_vs_{abs_}delta_rank keys correlate_polysemy_with_retrieval's extra_flags
    param produces (Experiment 14), on top of the always-present is_polysemic keys. Does
    not re-touch any per-sample data; purely aggregates already-computed per-run dicts."""
    assert len(results) == len(tags), (len(results), len(tags))
    per_run = {tag: r for tag, r in zip(tags, results)}
    corr_keys = ["corr_is_polysemic_vs_abs_delta_rank", "corr_is_polysemic_vs_delta_rank"]
    for flag in (extra_flag_names or []):
        corr_keys += [f"corr_{flag}_vs_abs_delta_rank", f"corr_{flag}_vs_delta_rank"]
    pooled = {}
    for corr_key in corr_keys:
        rhos = np.array([r[corr_key]["rho"] for r in results], dtype=float)
        n = len(rhos)
        mean = float(rhos.mean())
        std = float(rhos.std(ddof=1)) if n > 1 else float("nan")
        sem = std / np.sqrt(n) if n > 1 and std == std else float("nan")
        z = mean / sem if sem == sem and sem > 0 else float("nan")
        pooled[corr_key] = {"n": n, "mean": mean, "std": std, "sem": sem, "z": z}
    return {"n_runs": len(results), "per_run": per_run, "pooled": pooled}


def save_raw_arrays(
    path: str,
    a_idx: np.ndarray,
    b_idx: np.ndarray,
    c_idx: np.ndarray,
    dist_bc: np.ndarray,
    dist_bc_baseline: np.ndarray,
    jaccard: np.ndarray,
    retrieval_raw: dict = None,
) -> None:
    """Write the optional raw arrays needed to reproduce Experiment 12 figures."""
    payload = {
        "a_idx": a_idx,
        "b_idx": b_idx,
        "c_idx": c_idx,
        "dist_bc": dist_bc,
        "dist_bc_baseline": dist_bc_baseline,
        "jaccard": jaccard,
    }
    if retrieval_raw is not None:
        payload.update(retrieval_raw)
    np.savez(path, **payload)


def _load_features(storage_dir: str):
    from src.utils import FeatureManager

    fm = FeatureManager(storage_dir)
    data = fm.load_all_to_ram(["img_features", "txt_features"])
    img = data["img_features"].numpy().astype(np.float32)
    txt = data["txt_features"].numpy().astype(np.float32)
    sample_ids = [int(s) for s in data["sample_ids"].tolist()]
    return img, txt, sample_ids


def run(
    storage_dir: str,
    template_dir: str,
    K: int = 30,
    alpha: float = 0.5,
    n_bridge_sample: int = 5000,
    seed: int = 0,
    device: str = "cuda",
    per_sample_npz=None,
    save_raw: str = None,
) -> dict:
    """End-to-end Experiment 12 pass: rebuild the buddy graph from cached features,
    classify its edges, sample bridge-node (A, B, C) triples, measure whether the
    ALREADY-SAVED buddy-init embedding (template_dir) pulls B and C together vs. a
    degree-matched baseline, check whether that pull is graded by shared-neighbor
    structure, and (if per_sample_npz is given) cross-reference the per-node polysemy
    label against Experiment 11.2's per-sample retrieval-rank/drift dump."""
    from src.conditional_buddy.buddy_graph import bridge_node_stats, classify_edges
    from src.conditional_buddy.compute_buddies import _l2_normalize, build_buddy_graphs

    img, txt, sample_ids = _load_features(storage_dir)
    img_n, txt_n = _l2_normalize(img), _l2_normalize(txt)
    A_img, A_txt, E = build_buddy_graphs(img_n, txt_n, K=K, alpha=alpha, device=device)

    template_ids = np.load(Path(template_dir) / "sample_ids.npy").tolist()
    assert template_ids == sample_ids, (
        "template_dir's sample_ids.npy must match the freshly-loaded feature store's "
        "sample order exactly (CLAUDE.md's sample-id-consistency rule) -- do not proceed "
        "past this assertion if it fires; it means the wrong template/feature-store pair "
        "was passed"
    )
    emb = np.load(Path(template_dir) / "embeddings.npy")

    N = len(sample_ids)
    typed = classify_edges(A_img, A_txt, E, N)
    bstats = bridge_node_stats(typed, N)
    labels = label_nodes(bstats)
    E_img_only, E_txt_only = build_typed_adjacency(typed, N)

    rng = np.random.default_rng(seed)
    pairs = extract_bridge_pairs(bstats, E_img_only, E_txt_only, n_bridge_sample, rng)
    buckets = degree_deciles(E)
    baselines = sample_baselines(pairs, E, buckets, rng)
    valid = baselines >= 0
    pairs, baselines = pairs[valid], baselines[valid]

    a_idx, b_idx, c_idx = pairs[:, 0], pairs[:, 1], pairs[:, 2]
    dist_bc = embedded_l2_distance(emb, b_idx, c_idx)
    dist_bc_baseline = embedded_l2_distance(emb, b_idx, baselines)
    jaccard = shared_neighbor_jaccard(E, b_idx, c_idx)
    pull_summary = paired_pull_summary(dist_bc, dist_bc_baseline)
    grading_corr = spearman_correlate(jaccard, dist_bc_baseline - dist_bc)

    result = {
        "n_bridge_nodes": bstats["n_bridge_nodes"],
        "frac_bridge_nodes": bstats["frac_bridge_nodes"],
        "n_pairs_sampled": int(len(pairs)),
        "pull_summary": pull_summary,
        "grading_corr_jaccard_vs_pull": grading_corr,
        "label_counts": {lbl: int((labels == lbl).sum())
                         for lbl in ("neither", "img_only_only", "txt_only_only", "bridge")},
    }
    retrieval_raw = None
    if per_sample_npz is not None:
        npz_list = [per_sample_npz] if isinstance(per_sample_npz, str) else list(per_sample_npz)
        if len(npz_list) == 1:
            if save_raw is not None:
                result["retrieval_correlation"], retrieval_raw = correlate_polysemy_with_retrieval(
                    labels, sample_ids, npz_list[0], return_raw=True
                )
            else:
                result["retrieval_correlation"] = correlate_polysemy_with_retrieval(
                    labels, sample_ids, npz_list[0]
                )
        else:
            per_run_results = [correlate_polysemy_with_retrieval(labels, sample_ids, p) for p in npz_list]
            tags = [Path(p).parent.parent.name for p in npz_list]
            result["retrieval_correlation"] = pool_cross_references(per_run_results, tags)
    if save_raw is not None:
        save_raw_arrays(
            save_raw, a_idx, b_idx, c_idx, dist_bc, dist_bc_baseline, jaccard, retrieval_raw
        )
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--storage-dir", default="/data/SSD2/pre_extract/redcaps_150k/features")
    ap.add_argument("--template-dir",
                    default="res/CoSiR_condition_freeze_ablation/redcaps_150k/template_embeddings")
    ap.add_argument("--K", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--n-bridge-sample", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--per-sample-npz", default=None, nargs="+",
                    help="one or more Task 2 --dump-per-sample .npz paths for the retrieval-rank/"
                         "drift cross-reference (optional; multiple paths are pooled across runs)")
    ap.add_argument("--out", default=None, help="write the JSON result here (optional)")
    ap.add_argument("--save-raw", default=None,
                    help="write sampled-pair raw arrays (and retrieval arrays when available) to .npz")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return

    result = run(
        storage_dir=args.storage_dir, template_dir=args.template_dir, K=args.K,
        alpha=args.alpha, n_bridge_sample=args.n_bridge_sample, seed=args.seed,
        device=args.device, per_sample_npz=args.per_sample_npz, save_raw=args.save_raw,
    )

    print(f"\n{'='*78}\nExperiment 12 - cross-modal polysemy bridge-node diagnostic\n{'='*78}")
    print(f"  bridge nodes: {result['n_bridge_nodes']:,} ({100*result['frac_bridge_nodes']:.1f}% of nodes)")
    print(f"  label counts: {result['label_counts']}")
    print(f"  sampled bridge pairs: {result['n_pairs_sampled']:,}")
    ps = result["pull_summary"]
    sig = f"  mean/SEM={ps['z']:+.1f}{' *' if ps['z'] == ps['z'] and abs(ps['z']) >= 2 else ''}" if ps["n"] > 1 else ""
    print(f"  pull (baseline_dist - bc_dist): mean={ps['mean']:+.4f} (n={ps['n']}, "
          f"frac_pulled_closer={ps['frac_pulled_closer']:.3f}){sig}")
    gc = result["grading_corr_jaccard_vs_pull"]
    print(f"  grading check: corr(shared_neighbor_jaccard, pull) rho={gc['rho']:+.3f} p={gc['p']:.3e}")
    if "retrieval_correlation" in result:
        rc = result["retrieval_correlation"]
        if "n_runs" in rc:
            print(f"  retrieval cross-reference, pooled across {rc['n_runs']} run(s): "
                  f"{sorted(rc['per_run'].keys())}")
            for corr_key, human in (
                ("corr_is_polysemic_vs_abs_delta_rank", "|delta_rank|"),
                ("corr_is_polysemic_vs_delta_rank", "delta_rank"),
            ):
                p = rc["pooled"][corr_key]
                sig = f"  mean/SEM={p['z']:+.1f}{' *' if p['z'] == p['z'] and abs(p['z']) >= 2 else ''}" if p["n"] > 1 else ""
                print(f"    corr(is_polysemic, {human}) across runs: mean rho={p['mean']:+.3f} (n={p['n']}){sig}")
        else:
            print(f"  retrieval cross-reference (n_joined={rc['n_joined']}):")
            for lbl in ("neither", "img_only_only", "txt_only_only", "bridge"):
                if lbl in rc:
                    print(f"    {lbl}: n={rc[lbl]['n']} median|delta_rank|={rc[lbl]['median_abs_delta_rank']:.1f} "
                          f"median_delta_rank={rc[lbl]['median_delta_rank']:+.1f} "
                          f"median_drift={rc[lbl]['median_condition_drift']:.4f}")
            c1 = rc["corr_is_polysemic_vs_abs_delta_rank"]
            print(f"    corr(is_polysemic, |delta_rank|): rho={c1['rho']:+.3f} p={c1['p']:.3e}")
            c2 = rc["corr_is_polysemic_vs_delta_rank"]
            print(f"    corr(is_polysemic, delta_rank):   rho={c2['rho']:+.3f} p={c2['p']:.3e}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Wrote {args.out}")
    if args.save_raw:
        print(f"  Wrote {args.save_raw}")


if __name__ == "__main__":
    main()
