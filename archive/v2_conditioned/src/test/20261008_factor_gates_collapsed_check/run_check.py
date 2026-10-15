"""Sanity-check the factor-geometry gates on the known-collapsed seed-42 codes.

The cached codes (Task 9 mechanism run) were fit on the row-split train rows.
Fit rows here are those train rows and eval rows are the held rows. Community
labels reproduce the graph -> Stage 1 -> communities steps of
``prepare_item_disjoint_codes`` on the train rows only (no factor training).
Run from the repository root:

    /root/miniconda3/envs/CoSiR/bin/python src/test/20261008_factor_gates_collapsed_check/run_check.py
"""

import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from time import perf_counter

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TASK9 = ROOT / "src/test/20261007_naive_rule_mechanism_analysis"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TASK9))

from run_mechanism import EXPECTED_SAMPLES, load_real_features, split_items  # noqa: E402
from src.eval.factor_gates import FactorGateThresholds, evaluate_factor_gates  # noqa: E402
from src.model.communities import community_stats, detect_communities  # noqa: E402
from src.model.graph import GraphConfig, build_content_graph  # noqa: E402
from src.train.stage1 import Stage1Config, train_stage1  # noqa: E402


def train_row_community_labels(img_features: np.ndarray, txt_features: np.ndarray,
                               train: np.ndarray) -> tuple[np.ndarray, int]:
    """Graph -> Stage 1 -> communities on train rows, as prepare_item_disjoint_codes does."""
    train_img, train_txt = img_features[train], txt_features[train]
    graph = build_content_graph(train_img, train_txt, GraphConfig())
    print(f"Train-only graph: {graph.nnz // 2:,} edges", flush=True)
    with redirect_stdout(io.StringIO()):
        _, embeddings = train_stage1(train_img, train_txt, graph, Stage1Config())
    if not np.isfinite(embeddings).all():
        raise ValueError("Stage 1 produced non-finite embeddings")
    labels = np.asarray(detect_communities(embeddings), dtype=np.int64)
    count = community_stats(labels)["num_communities"]
    print(f"Train-only communities: {count}", flush=True)
    return labels, count


def main() -> None:
    started = perf_counter()
    img_codes = np.load(TASK9 / "cache/factor42_img.npy")
    txt_codes = np.load(TASK9 / "cache/factor42_txt.npy")
    img_features, txt_features = load_real_features()
    if not (len(img_codes) == len(txt_codes) == len(img_features) == len(txt_features) == EXPECTED_SAMPLES):
        raise ValueError("Cached codes and features must cover all rows")
    train, held = split_items(EXPECTED_SAMPLES)
    fit_items = json.loads((TASK9 / "cache/factor42_meta.json").read_text())["factor_fit_items"]
    if fit_items != len(train):
        raise ValueError(f"Codes were fit on {fit_items} rows, split has {len(train)} train rows")
    labels, n_communities = train_row_community_labels(img_features, txt_features, train)

    thresholds = FactorGateThresholds()
    report = evaluate_factor_gates(
        fit_img_codes=img_codes[train], fit_txt_codes=txt_codes[train],
        fit_img_features=img_features[train], fit_txt_features=txt_features[train],
        eval_img_codes=img_codes[held], eval_txt_codes=txt_codes[held],
        eval_img_features=img_features[held], eval_txt_features=txt_features[held],
        community_img_codes=img_codes[train], community_txt_codes=txt_codes[train],
        community_labels=labels, thresholds=thresholds,
    )
    runtime = perf_counter() - started
    print("== Values")
    for key, value in report.values.items():
        print(f"{key}: {value}")
    print("== Gates")
    for key, ok in report.passed.items():
        print(f"{key}: {'PASS' if ok else 'FAIL'}")
    print(f"all_passed: {report.all_passed}; runtime {runtime:.1f}s")

    (HERE / "gates_collapsed.json").write_text(json.dumps({
        "values": report.values, "passed": report.passed, "all_passed": report.all_passed,
        "thresholds": vars(thresholds), "train_rows": len(train), "eval_rows": len(held),
        "train_communities": n_communities, "runtime_seconds": runtime,
    }, indent=2))


if __name__ == "__main__":
    main()
