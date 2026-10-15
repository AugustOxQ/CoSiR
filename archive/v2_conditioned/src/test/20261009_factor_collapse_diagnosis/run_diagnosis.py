"""One-variable-at-a-time diagnosis of the factor-space collapse (existing code only).

Seven pre-registered variants (D0-D6) of the EXISTING ``train_factors`` recipe on
the painting-grouped split. Nothing under ``src/`` is modified. Run from the
repository root with the CoSiR environment:

    /root/miniconda3/envs/CoSiR/bin/python src/test/20261009_factor_collapse_diagnosis/run_diagnosis.py

Setup shared by all variants: content graph, Stage 1 and communities on the
``train`` rows only; factor training on ``train`` rows; ``val`` rows encoded by
the returned model in eval mode (8,192-row batches). Gates: fit = train,
eval = val, community codes / labels = train. Per-variant results are cached
as JSON so an interrupted run resumes without repeating a finished variant.
"""

import io
import json
import re
import sys
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy.sparse as sp
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from src.data.artelingo import load_artelingo  # noqa: E402
from src.data.splits import grouped_split, leakage_groups, split_leakage  # noqa: E402
from src.eval.factor_gates import FactorGateThresholds, evaluate_factor_gates  # noqa: E402
from src.model.communities import community_stats, detect_communities  # noqa: E402
from src.model.graph import GraphConfig, build_content_graph  # noqa: E402
from src.train.stage1 import Stage1Config, train_stage1  # noqa: E402
from src.train.train_factors import FactorTrainingConfig, train_factors  # noqa: E402

SEED = 42
EXPECTED_SPLIT = (216_107, 30_872, 61_744)
CACHE = HERE / "cache"
RESULTS = HERE / "results"
ZERO_AUX = dict(lambda_paired=0.0, lambda_graph=0.0, lambda_sparsity=0.0,
                lambda_anti_split=0.0, lambda_usage_balance=0.0)
D0 = dict(lambda_usage_balance=0.1)
# id -> (overrides on FactorTrainingConfig, center encoder input?)
VARIANTS = {
    "D0": (dict(D0), False),
    "D1": (dict(ZERO_AUX), False),
    "D2": ({**ZERO_AUX, "lambda_sparsity": 0.01}, False),
    "D3": ({**D0, "lambda_paired": 0.0}, False),
    "D4": ({**D0, "lambda_usage_balance": 0.0}, False),
    "D5": ({**D0, "lambda_graph": 0.0}, False),
    "D6": (dict(D0), True),
}


def pc1_share(codes: np.ndarray) -> float:
    """Share of centered-covariance variance on the first principal component."""
    energy = np.linalg.svd(np.asarray(codes, np.float64) - np.mean(codes, axis=0, dtype=np.float64),
                           compute_uv=False) ** 2
    return float(energy[0] / energy.sum()) if energy.sum() > 0 else 0.0


def encode_rows(model, img: np.ndarray, txt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Encode rows with the frozen model in eval mode, 8,192-row batches."""
    device = next(model.parameters()).device
    model.eval()
    img_out, txt_out = [], []
    with torch.no_grad():
        for start in range(0, len(img), 8192):
            img_out.append(model.encode_image(torch.as_tensor(
                img[start:start + 8192], dtype=torch.float32, device=device)).cpu().numpy())
            txt_out.append(model.encode_text(torch.as_tensor(
                txt[start:start + 8192], dtype=torch.float32, device=device)).cpu().numpy())
    return np.concatenate(img_out), np.concatenate(txt_out)


def prepare_train_side(train_img, train_txt):
    """Graph -> Stage 1 -> communities on the train rows; cached for resumption."""
    CACHE.mkdir(exist_ok=True)
    graph_path, labels_path, meta_path = CACHE / "graph.npz", CACHE / "labels.npy", CACHE / "setup.json"
    if graph_path.exists() and labels_path.exists() and meta_path.exists():
        print("Loading cached train-row graph and communities", flush=True)
        return sp.load_npz(graph_path).tocsr(), np.load(labels_path), json.loads(meta_path.read_text())
    started = perf_counter()
    graph = build_content_graph(train_img, train_txt, GraphConfig())
    edges = int(graph.nnz // 2)
    print(f"Train-only graph: {edges:,} edges", flush=True)
    with redirect_stdout(io.StringIO()):
        _, embeddings = train_stage1(train_img, train_txt, graph, Stage1Config())
    if not np.isfinite(embeddings).all():
        raise ValueError("Stage 1 produced non-finite embeddings")
    labels = np.asarray(detect_communities(embeddings), dtype=np.int64)
    stats = community_stats(labels)
    print(f"Train-only communities: {stats['num_communities']}", flush=True)
    meta = {"edges": edges, "communities": int(stats["num_communities"]),
            "setup_seconds": perf_counter() - started}
    sp.save_npz(graph_path, graph)
    np.save(labels_path, labels)
    meta_path.write_text(json.dumps(meta, indent=2))
    return graph, labels, meta


def run_variant(name, overrides, center, data, split, graph, labels, thresholds) -> dict:
    train, val = split.train, split.val
    train_img, train_txt = data.img_features[train], data.txt_features[train]
    val_img, val_txt = data.img_features[val], data.txt_features[val]
    config = FactorTrainingConfig(**overrides)
    if config.seed != SEED or config.num_factors != 32 or config.epochs != 2000:
        raise AssertionError("Variants must keep seed 42, 32 factors, 2,000 epochs")
    started = perf_counter()
    if center:
        # Center ONLY the encoder input, by train-row means; gates get the originals.
        img_mean, txt_mean = train_img.mean(axis=0), train_txt.mean(axis=0)
        fit_img_in, fit_txt_in = train_img - img_mean, train_txt - txt_mean
        val_img_in, val_txt_in = val_img - img_mean, val_txt - txt_mean
    else:
        fit_img_in, fit_txt_in, val_img_in, val_txt_in = train_img, train_txt, val_img, val_txt
    log = io.StringIO()
    with redirect_stdout(log):
        model, train_img_codes, train_txt_codes = train_factors(fit_img_in, fit_txt_in, graph, config)
    losses = [float(m) for m in re.findall(r"loss=([-+0-9.eE]+|nan|inf)", log.getvalue())]
    val_img_codes, val_txt_codes = encode_rows(model, val_img_in, val_txt_in)
    for array in (train_img_codes, train_txt_codes, val_img_codes, val_txt_codes):
        if not np.isfinite(array).all():
            raise ValueError(f"{name}: non-finite codes")
    report = evaluate_factor_gates(
        fit_img_codes=train_img_codes, fit_txt_codes=train_txt_codes,
        fit_img_features=train_img, fit_txt_features=train_txt,
        eval_img_codes=val_img_codes, eval_txt_codes=val_txt_codes,
        eval_img_features=val_img, eval_txt_features=val_txt,
        community_img_codes=train_img_codes, community_txt_codes=train_txt_codes,
        community_labels=labels, thresholds=thresholds,
    )
    result = {
        "variant": name, "overrides": overrides, "centered_input": center,
        "config": asdict(config),
        "values": report.values, "passed": report.passed, "all_passed": report.all_passed,
        "pc1_share_img": pc1_share(val_img_codes), "pc1_share_txt": pc1_share(val_txt_codes),
        "mean_code_img": float(val_img_codes.mean()), "mean_code_txt": float(val_txt_codes.mean()),
        "loss_first": losses[0], "loss_last": losses[-1],
        "loss_mean_last100": float(np.mean(losses[-100:])), "n_loss_lines": len(losses),
        "train_rows": len(train), "eval_rows": len(val),
        "runtime_seconds": perf_counter() - started,
    }
    return result


def print_result(result: dict) -> None:
    v = result["values"]
    print(f"[{result['variant']}] PR img/txt {v['participation_ratio_img']:.4f}/{v['participation_ratio_txt']:.4f}"
          f"  max|r| {v['correlation']['max_abs']:.5f} ({v['correlation']['pairs_at_or_above']}/"
          f"{v['correlation']['pairs_total']})  PC1 {result['pc1_share_img']:.4f}/{result['pc1_share_txt']:.4f}"
          f"  active {v['active_fraction_img']:.4f}/{v['active_fraction_txt']:.4f}"
          f"  readout img {v['readout_img']:.4f} (pca10 {v['pca10_img']:.4f}) txt {v['readout_txt']:.4f}"
          f" (pca10 {v['pca10_txt']:.4f})  top2 {v['top2_mass_share']:.4f}"
          f"  span {v['community']['spanning_fraction']:.3f}  ret {v['retrieval_ratio']:.4f}"
          f"  all_passed {result['all_passed']}  {result['runtime_seconds']:.0f}s", flush=True)
    print("     gates: " + ", ".join(f"{k}={'PASS' if ok else 'FAIL'}" for k, ok in result["passed"].items()), flush=True)


def main() -> None:
    started = perf_counter()
    RESULTS.mkdir(exist_ok=True)
    print("Loading ArtELingo...", flush=True)
    data = load_artelingo()
    groups = leakage_groups(data.paintings, data.img_features)
    split = grouped_split(groups, fractions=(0.7, 0.1, 0.2), seed=SEED)
    sizes = (len(split.train), len(split.val), len(split.held))
    print(f"Split train/val/held = {sizes}", flush=True)
    if sizes != EXPECTED_SPLIT:
        raise AssertionError(f"Unexpected split sizes {sizes}, expected {EXPECTED_SPLIT}")
    leak = split_leakage(split, data.paintings, data.img_features)
    if any(leak.values()):
        raise AssertionError(f"Split leakage: {leak}")
    graph, labels, setup_meta = prepare_train_side(
        data.img_features[split.train], data.txt_features[split.train])
    thresholds = FactorGateThresholds()
    (RESULTS / "setup.json").write_text(json.dumps({
        **setup_meta, "split_sizes": sizes, "leakage": leak, "thresholds": asdict(thresholds)}, indent=2))
    for name, (overrides, center) in VARIANTS.items():
        path = RESULTS / f"{name}.json"
        if path.exists():
            print(f"[{name}] cached result found, not re-running", flush=True)
            print_result(json.loads(path.read_text()))
            continue
        print(f"[{name}] running overrides={overrides} centered={center}", flush=True)
        result = run_variant(name, overrides, center, data, split, graph, labels, thresholds)
        path.write_text(json.dumps(result, indent=2))
        print_result(result)
    print(f"Total runtime {perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
