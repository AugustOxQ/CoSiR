#!/usr/bin/env python3
"""Experiment 16.1 Stage-A RedCaps buddy-graph diagnostic sweep."""

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
REDCAPS_BUDDY_DIR = ROOT / "src" / "test" / "20260623_redcaps_buddy"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REDCAPS_BUDDY_DIR))

from redcaps_buddy import Data, subreddit_lift, subreddit_of
from src.conditional_buddy.buddy_graph import (
    CUVS_MIN_N,
    classify_edges,
    ensure_min_degree,
    mutual_knn,
    union_graph,
)
from src.conditional_buddy.compute_buddies import _l2_normalize
from src.utils import FeatureManager


K_GRID = [10, 20, 30, 50, 75, 100]
SCALES = {
    "150k": {
        "n_samples": 150_000,
        "feature_store": "/data/SSD2/pre_extract/redcaps_150k/features",
        "annotation_path": "/data/PDD/redcaps/redcaps_plus/redcaps_150k.json",
        "dataset_variant": "redcaps_150k",
    },
    "300k": {
        "n_samples": 300_000,
        "feature_store": "/data/SSD2/pre_extract/redcaps_300k_diverse/features",
        "annotation_path": "/data/PDD/redcaps/redcaps_plus/redcaps_300k_diverse.json",
        "dataset_variant": "redcaps_300k_diverse",
    },
    "500k": {
        "n_samples": 500_000,
        "feature_store": "/data/SSD2/pre_extract/redcaps_500k_diverse/features",
        "annotation_path": "/data/PDD/redcaps/redcaps_plus/redcaps_500k_diverse.json",
        "dataset_variant": "redcaps_500k_diverse",
    },
}


def load_scale(scale_label: str) -> Data:
    cfg = SCALES[scale_label]
    loaded = FeatureManager(cfg["feature_store"]).load_all_to_ram(
        ["img_features", "txt_features"]
    )
    img = _l2_normalize(loaded["img_features"].numpy())
    txt = _l2_normalize(loaded["txt_features"].numpy())
    sample_ids = [int(sample_id) for sample_id in loaded["sample_ids"].tolist()]
    with open(cfg["annotation_path"]) as handle:
        metadata = json.load(handle)
    records = [metadata[sample_id] for sample_id in sample_ids]
    subreddit_names = [subreddit_of(record) for record in records]
    unique_names = sorted(set(subreddit_names))
    name_to_id = {name: index for index, name in enumerate(unique_names)}
    sub_id = np.asarray([name_to_id[name] for name in subreddit_names], dtype=np.int64)
    data = Data(img, txt, sample_ids, sub_id, unique_names, records)
    if data.n != cfg["n_samples"]:
        raise ValueError(f"{scale_label} expected {cfg['n_samples']:,} rows, got {data.n:,}")
    return data


def strict_stats(A_img, A_txt: object, n_samples: int) -> tuple[object, dict]:
    strict = A_img.multiply(A_txt).tocsr()
    strict.data[:] = 1.0
    degrees = np.diff(strict.indptr)
    return strict, {
        "strict_edge_count": int(strict.nnz // 2),
        "strict_avg_degree": float(strict.nnz / n_samples),
        "strict_zero_degree_fraction": float(np.mean(degrees == 0)),
        "strict_median_degree": float(np.median(degrees)),
        "strict_degree_p90": float(np.percentile(degrees, 90)),
    }


def union_stats(A_img, A_txt, img: np.ndarray, txt: np.ndarray, device: str) -> tuple[object, dict]:
    n_samples = img.shape[0]
    union, repair_info = ensure_min_degree(union_graph(A_img, A_txt), img, txt, device)
    typed = classify_edges(A_img, A_txt, union, n_samples)
    counts = {name: int(np.sum(typed[name])) for name in ("img_only", "txt_only", "both", "repair")}
    edge_count = len(typed["keys"])
    fractions = {name: counts[name] / edge_count for name in counts}
    if not np.isclose(sum(fractions.values()), 1.0, atol=1e-6):
        raise AssertionError("union edge-type fractions do not sum to 1")
    return typed, {
        "union_edge_count": int(edge_count),
        "union_avg_degree": float(union.nnz / n_samples),
        "union_img_only_edge_count": counts["img_only"],
        "union_txt_only_edge_count": counts["txt_only"],
        "union_both_edge_count": counts["both"],
        "union_repair_edge_count": counts["repair"],
        "union_img_only_fraction": float(fractions["img_only"]),
        "union_txt_only_fraction": float(fractions["txt_only"]),
        "union_both_fraction": float(fractions["both"]),
        "union_repair_fraction": float(fractions["repair"]),
        "repair_isolated_node_count": int(repair_info["num_isolated"]),
    }


def quality_stats(data: Data, typed: dict, n_samples: int) -> dict:
    keys = typed["keys"]
    edges = np.stack([keys // n_samples, keys % n_samples], axis=1).astype(np.int64, copy=False)
    lift = subreddit_lift(data, edges, top_k=None)
    return {
        "union_subreddit_obs_same_fraction": float(lift["obs_same_frac"]),
        "union_subreddit_expected_same_fraction": float(lift["exp_same_frac"]),
        "union_subreddit_lift": float(lift["overall_lift"]),
        "union_subreddit_n_qualifying": int(len(lift["top_enriched"])),
    }


def run_cell(scale_label: str, data: Data, k: int, device: str, batch_size: int) -> dict:
    cfg = SCALES[scale_label]
    A_img = mutual_knn(data.img, k, device, batch_size=batch_size)
    A_txt = mutual_knn(data.txt, k, device, batch_size=batch_size)
    strict, strict_result = strict_stats(A_img, A_txt, data.n)
    typed, union_result = union_stats(A_img, A_txt, data.img, data.txt, device)
    if union_result["union_both_edge_count"] != strict_result["strict_edge_count"]:
        raise AssertionError("union both-edge count must equal strict edge count")
    return {
        "experiment_id": "16.1",
        "stage": "A",
        "dataset_variant": cfg["dataset_variant"],
        "n_samples": data.n,
        "scale_label": scale_label,
        "k": k,
        "knn_backend": "cuvs" if data.n >= CUVS_MIN_N else "torch",
        **strict_result,
        **union_result,
        **quality_stats(data, typed, data.n),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _monotone_degrees(degrees: np.ndarray) -> tuple[np.ndarray, int]:
    violations = int(np.sum(np.diff(degrees) < 0))
    if violations == 0:
        return degrees, violations
    try:
        from sklearn.isotonic import IsotonicRegression

        fitted = IsotonicRegression(increasing=True, out_of_bounds="clip").fit_transform(
            np.arange(degrees.size), degrees
        )
    except ImportError:
        fitted = np.maximum.accumulate(degrees)
    return np.asarray(fitted, dtype=np.float64), violations


def derive_k_prediction(rows: list[dict], target_n: int, reference_scale: str = "150k", reference_k: int = 30) -> dict:
    reference = next(
        row for row in rows if row["scale_label"] == reference_scale and row["k"] == reference_k
    )
    target_rows = sorted((row for row in rows if row["n_samples"] == target_n), key=lambda row: row["k"])
    ks = np.asarray([row["k"] for row in target_rows], dtype=np.float64)
    degrees = np.asarray([row["strict_avg_degree"] for row in target_rows], dtype=np.float64)
    fitted, violations = _monotone_degrees(degrees)
    target_degree = float(reference["strict_avg_degree"])
    low, high = float(fitted[0]), float(fitted[-1])
    if target_degree < low:
        k_continuous, status = float(ks[0]), "out_of_grid_low_clamped"
    elif target_degree > high:
        k_continuous, status = float(ks[-1]), "out_of_grid_high_clamped"
    else:
        from scipy.interpolate import PchipInterpolator

        curve = PchipInterpolator(ks, fitted)
        left, right = float(ks[0]), float(ks[-1])
        for _ in range(64):
            middle = (left + right) / 2
            if float(curve(middle)) >= target_degree:
                right = middle
            else:
                left = middle
        k_continuous, status = right, "interpolated"
    return {
        "k_continuous": float(k_continuous),
        "k_status": status,
        "target_degree": target_degree,
        "attainable_degree_min": low,
        "attainable_degree_max": high,
        "monotonicity_violations": violations,
    }


def select_shortlist(rows: list[dict], target_n: int) -> dict:
    prediction = derive_k_prediction(rows, target_n)
    k_continuous = prediction["k_continuous"]
    anchor_k = 30
    integer_k = int(np.clip(round(k_continuous), min(K_GRID), max(K_GRID)))
    if abs(integer_k - anchor_k) <= 1:
        integer_k = anchor_k
    selected = {anchor_k, integer_k}
    opposite = [
        k for k in K_GRID if k not in selected and (k > k_continuous if anchor_k < k_continuous else k < k_continuous)
    ]
    candidates = opposite or [k for k in K_GRID if k not in selected]
    bracket_k = min(candidates, key=lambda k: abs(np.log(k / k_continuous)))
    selected.add(bracket_k)
    return {
        "selected_k": sorted(selected),
        "k_continuous": k_continuous,
        "k_train_prediction": integer_k,
        "k_status": prediction["k_status"],
        "bracket_k": bracket_k,
    }


def write_rows(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "stage_a_cells.json", "w") as handle:
        json.dump(rows, handle, indent=2)
        handle.write("\n")
    with open(output_dir / "stage_a_cells.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict], shortlists: dict[str, dict]) -> None:
    print("scale  K  strict_deg  union_deg  both_frac  lift")
    for row in rows:
        print(
            f"{row['scale_label']:>5} {row['k']:>3} {row['strict_avg_degree']:>11.3f} "
            f"{row['union_avg_degree']:>10.3f} {row['union_both_fraction']:>10.3f} "
            f"{row['union_subreddit_lift']:>5.3f}"
        )
    for scale, shortlist in shortlists.items():
        print(
            f"{scale}: K*={shortlist['k_continuous']:.3f} ({shortlist['k_status']}), "
            f"shortlist={shortlist['selected_k']}"
        )


def selftest() -> None:
    rows = []
    for n_samples, scale, multiplier in ((150_000, "150k", 1.0), (300_000, "300k", 0.7), (500_000, "500k", 0.5)):
        for k in K_GRID:
            rows.append({"n_samples": n_samples, "scale_label": scale, "k": k, "strict_avg_degree": multiplier * k})
    prediction = derive_k_prediction(rows, 300_000)
    assert prediction["k_status"] == "interpolated"
    assert np.isclose(prediction["k_continuous"], 30 / 0.7, atol=1e-6)
    shortlist = select_shortlist(rows, 300_000)
    assert 30 in shortlist["selected_k"]
    assert shortlist["bracket_k"] in K_GRID
    print("selftest: pass")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for scale_label in SCALES:
        print(f"Loading {scale_label}...")
        data = load_scale(scale_label)
        for k in K_GRID:
            print(f"Running {scale_label}, K={k}...")
            rows.append(run_cell(scale_label, data, k, device, args.batch_size))
    if len(rows) != len(SCALES) * len(K_GRID):
        raise AssertionError(f"expected 18 cells, got {len(rows)}")
    output_dir = Path(__file__).resolve().parent
    write_rows(rows, output_dir)
    shortlists = {scale: select_shortlist(rows, SCALES[scale]["n_samples"]) for scale in ("300k", "500k")}
    print_summary(rows, shortlists)


if __name__ == "__main__":
    main()
