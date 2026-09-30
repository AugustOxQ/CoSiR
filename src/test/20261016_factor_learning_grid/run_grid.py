"""CoSiR v2 Candidate A factor-learning 2x2 grid (spec docs/superpowers/specs/2026-09-30-cosir-v2-candidate-a-factor-learning-design.md).

Four factor models on scorer-train rows (spec §4): C0 (row agreement, no condition; the matched control),
A (painting agreement), S (CLIP image k-means condition episodes), AS (both). Run from the repository root:

    python src/test/20261016_factor_learning_grid/run_grid.py --prepare
    python src/test/20261016_factor_learning_grid/run_grid.py --smoke            # timing: local GPU or DAS6?
    python src/test/20261016_factor_learning_grid/run_grid.py --run C0 --seed 42
    python src/test/20261016_factor_learning_grid/run_grid.py --evaluate         # Task 4
    python src/test/20261016_factor_learning_grid/run_grid.py --tables

Row scope: training, the graph and the partitions use scorer-train rows only (local indices 0..n-1); evaluation
reads selection rows only; val and held rows are never read.
"""

import argparse
import dataclasses
import hashlib
import importlib.util
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from scipy.sparse import load_npz, save_npz

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

_FINAL_PATH = ROOT / "src/test/20261014_stage_d_final/run_final.py"
_spec = importlib.util.spec_from_file_location("run_final", _FINAL_PATH)
fin = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fin)
sel = fin.sel

from src.data.artelingo import load_artelingo  # noqa: E402
from src.eval.factor_gates import AMENDED_2026_09_29_THRESHOLDS, evaluate_factor_gates  # noqa: E402
from src.model.graph import GraphConfig, build_content_graph  # noqa: E402
from src.train.condition_sources import CommunitySource  # noqa: E402
from src.train.train_factors import (R3_CONFIG, encode_rows, load_factor_checkpoint,  # noqa: E402
                                     save_factor_checkpoint, train_factors)

SEED = 42
FULL_STEPS = 2000
CELLS = {"C0": {"agreement_level": "pair", "lambda_condition": 0.0},
         "A": {"agreement_level": "painting", "lambda_condition": 0.0},
         "S": {"agreement_level": "pair", "lambda_condition": 1.0},
         "AS": {"agreement_level": "painting", "lambda_condition": 1.0}}
R0_PATH = ROOT / "src/test/20261011_factor_repair_grid/checkpoints/R0_seed42.pt"
R0_SHA256 = "4229dfe55f735bc7e9849c8d7af623b5872a9de940f616969ef477fb00a253a7"
CACHE, CKPT, RESULTS = HERE / "cache", HERE / "checkpoints", HERE / "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SMOKE_STEPS = (10, 60)                      # two lengths; per-step time = slope (removes warm-up cost)
HEAVY_RUN_SECONDS = 45 * 60                 # stop-point thresholds (plan Task 3 Step 5)
HEAVY_TOTAL_SECONDS = 3 * 3600
HEAVY_PEAK_GIB = 20.0
log = sel.log


def cell_config(cell: str, seed: int, steps: int = FULL_STEPS):
    return dataclasses.replace(R3_CONFIG, painting_batches=True, seed=seed, epochs=steps, **CELLS[cell])


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gate_report(img_fit, txt_fit, img_eval, txt_eval, data, cache, community_local, reference):
    st, sl = cache["scorer_train"], cache["selection"]
    return evaluate_factor_gates(
        fit_img_codes=img_fit, fit_txt_codes=txt_fit,
        fit_img_features=data.img_features[st], fit_txt_features=data.txt_features[st],
        eval_img_codes=img_eval, eval_txt_codes=txt_eval,
        eval_img_features=data.img_features[sl], eval_txt_features=data.txt_features[sl],
        community_img_codes=img_fit, community_txt_codes=txt_fit, community_labels=community_local,
        thresholds=AMENDED_2026_09_29_THRESHOLDS, readout_reference=reference)


def prepare() -> None:
    started = perf_counter()
    for folder in (CACHE, CKPT, RESULTS):
        folder.mkdir(exist_ok=True)
    cache, _ = sel.load_prepared()
    data = load_artelingo()
    st, sl = cache["scorer_train"], cache["selection"]
    if np.intersect1d(cache["groups"][st], cache["groups"][sl]).size:
        raise AssertionError("a painting spans scorer-train and selection")
    _, local_groups = np.unique(cache["groups"][st], return_inverse=True)
    clip_image_local, community_local = cache["clip_image"][st], cache["community"][st]
    if (clip_image_local < 0).any() or (community_local < 0).any():
        raise AssertionError("cached partitions must label every scorer-train row")
    t0 = perf_counter()
    graph = build_content_graph(data.img_features[st], data.txt_features[st], GraphConfig())
    save_npz(CACHE / "graph.npz", graph)
    graph_seconds = perf_counter() - t0
    if sha256_file(R0_PATH) != R0_SHA256:
        raise AssertionError("R0 checkpoint SHA-256 mismatch")
    r0, _ = load_factor_checkpoint(R0_PATH, device=DEVICE)
    fit = encode_rows(r0, data.img_features, data.txt_features, rows=st)
    ev = encode_rows(r0, data.img_features, data.txt_features, rows=sl)
    r0_gates = gate_report(*fit, *ev, data, cache, community_local, reference=None)
    reference = [float(r0_gates.values["readout_img"]), float(r0_gates.values["readout_txt"])]
    np.savez(CACHE / "grid_prepare.npz", local_groups=local_groups.astype(np.int64),
             clip_image_local=clip_image_local.astype(np.int64), community_local=community_local.astype(np.int64))
    meta = {"scorer_train_rows": int(len(st)), "selection_rows": int(len(sl)),
            "paintings": int(local_groups.max() + 1), "graph_edges": int(graph.nnz // 2),
            "graph_seconds": graph_seconds, "r0_sha256": R0_SHA256, "r0_readout_reference": reference,
            "clip_image_groups": int(len(np.unique(clip_image_local))), "seconds": perf_counter() - started}
    (CACHE / "grid_prepare.json").write_text(json.dumps(meta, indent=2))
    log(f"Prepared: {meta}")


def load_grid():
    cache, _ = sel.load_prepared()
    prep = dict(np.load(CACHE / "grid_prepare.npz"))
    meta = json.loads((CACHE / "grid_prepare.json").read_text())
    return cache, prep, meta, load_npz(CACHE / "graph.npz").tocsr()


def run_cell(cell: str, seed: int, steps: int = FULL_STEPS, tag: str = "") -> dict:
    cache, prep, _, graph = load_grid()
    data = load_artelingo()
    st = cache["scorer_train"]
    config = cell_config(cell, seed, steps)
    source = (CommunitySource(prep["clip_image_local"], np.arange(len(st)))
              if config.lambda_condition > 0 else None)
    history: dict = {}
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = perf_counter()
    out = io.StringIO()
    with redirect_stdout(out):                                  # train_factors prints one line per step
        model, img_codes, txt_codes = train_factors(
            data.img_features[st], data.txt_features[st], graph, config, device=DEVICE,
            group_ids=prep["local_groups"], condition_source=source, history=history)
    seconds = perf_counter() - t0
    if not (np.isfinite(img_codes).all() and np.isfinite(txt_codes).all()):
        raise AssertionError(f"{cell} seed {seed}: non-finite codes")
    peak = torch.cuda.max_memory_allocated() / 2**30 if DEVICE == "cuda" else 0.0
    name = f"{cell}_seed{seed}{tag}"
    save_factor_checkpoint(model, config, CKPT / f"{name}.pt")
    record = {"cell": cell, "seed": seed, "steps": steps, "seconds": seconds, "peak_gpu_gib": peak,
              "history": history, "last_print": out.getvalue().strip().splitlines()[-1],
              "config": dataclasses.asdict(config)}
    (RESULTS / f"history_{name}.json").write_text(json.dumps(record, indent=2))
    log(f"{name}: {steps} steps in {seconds:.1f} s, peak GPU {peak:.2f} GiB")
    return record


def smoke() -> dict:
    per_cell = {}
    for cell in CELLS:
        short, long = (run_cell(cell, SEED, steps, tag=f"_smoke{steps}") for steps in SMOKE_STEPS)
        per_step = (long["seconds"] - short["seconds"]) / (SMOKE_STEPS[1] - SMOKE_STEPS[0])
        fixed = max(short["seconds"] - per_step * SMOKE_STEPS[0], 0.0)
        per_cell[cell] = {"seconds_per_step": per_step, "fixed_seconds": fixed,
                          "projected_full_seconds": fixed + per_step * FULL_STEPS,
                          "peak_gpu_gib": max(short["peak_gpu_gib"], long["peak_gpu_gib"])}
    slowest = max(v["projected_full_seconds"] for v in per_cell.values())
    grid = sum(v["projected_full_seconds"] for v in per_cell.values())
    replication = 2 * (slowest + per_cell["C0"]["projected_full_seconds"])   # picked cell unknown: slowest
    total = grid + replication
    peak = max(v["peak_gpu_gib"] for v in per_cell.values())
    heavy = slowest > HEAVY_RUN_SECONDS or total > HEAVY_TOTAL_SECONDS or peak > HEAVY_PEAK_GIB
    result = {"per_cell": per_cell, "projected_grid_seconds": grid, "projected_replication_seconds": replication,
              "projected_total_training_seconds_sequential": total, "peak_gpu_gib": peak,
              "thresholds": {"run_seconds": HEAVY_RUN_SECONDS, "total_seconds": HEAVY_TOTAL_SECONDS,
                             "peak_gib": HEAVY_PEAK_GIB},
              "decision": "ask_user_for_das6_node" if heavy else "run_locally", "device": DEVICE}
    (RESULTS / "smoke_timing.json").write_text(json.dumps(result, indent=2))
    log(f"Smoke timing: {json.dumps(result, indent=2)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run", choices=sorted(CELLS))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--tables", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.smoke:
        smoke()
    elif args.run:
        run_cell(args.run, args.seed)
    elif args.evaluate or args.tables:
        raise NotImplementedError("Task 4")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
