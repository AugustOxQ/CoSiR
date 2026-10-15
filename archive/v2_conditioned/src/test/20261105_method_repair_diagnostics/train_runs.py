"""Training runner of the method-repair diagnostics (PREREGISTRATION.md §4): L3, L5, LT (label bank) and MK3 (matched-k bank).

  flock -n -o -E 75 /tmp/gpu0.lock python src/test/20261105_method_repair_diagnostics/train_runs.py --train LT --seed 42 [--smoke]

Row scope: scorer-train rows only. Same refusals, graph checks, history fields and failure handling as E3's
run_gonogo.train. --smoke trains 50 steps on the smoke banks and E2's smoke graph into checkpoints/smoke/ and results/smoke/.
LAB runs (L3, L5, LT) append their checkpoint SHA-256 to results/label_checkpoints.json (smoke: results/smoke/).
"""
import argparse
import dataclasses
import fcntl
import json
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from scipy.sparse import load_npz

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LABEL_RUNS, ROOT, RUNS, folders, load_bank, local_rows, rg, run_config  # noqa: E402

from src.data.artelingo import load_artelingo  # noqa: E402
from src.data.artelingo_splits import artelingo_splits  # noqa: E402
from src.train.train_factors import save_factor_checkpoint, train_factors  # noqa: E402

SEEDS = (42, 43)


def register_checkpoint(reg: Path, name: str, sha: str) -> None:
    """Add {name: sha} to the registry under an exclusive lock (parallel runs), writing atomically."""
    with open(str(reg) + ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        shas = json.loads(reg.read_text()) if reg.exists() else {}
        shas[name] = sha
        tmp = reg.with_name(reg.name + f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(shas, indent=1, sort_keys=True))
        os.replace(tmp, reg)


def train(run: str, seed: int, smoke: bool) -> None:
    f = folders(smoke)
    steps = rg.SMOKE_STEPS if smoke else rg.FULL_STEPS
    name = f"{run}_seed{seed}"
    ckpt, hist = f["ckpt"] / f"{name}.pt", f["res"] / f"history_{name}.json"
    if not smoke:
        for path in (ckpt, hist, f["res"] / f"failed_{name}.json"):
            if path.exists():
                raise FileExistsError(f"{path} exists: it may be evidence behind the diagnostics. Refusing to overwrite.")
    e2 = rg.E2 / ("smoke" if smoke else "")
    record_path = e2 / "build_record.json"
    if not record_path.exists():
        raise FileNotFoundError(f"{record_path} missing")
    f["ckpt"].mkdir(parents=True, exist_ok=True)
    f["res"].mkdir(parents=True, exist_ok=True)

    t_load = perf_counter()
    data = load_artelingo()
    sp = artelingo_splits(data)
    st, local_groups = local_rows(data, sp)
    n = len(st)
    record = json.loads(record_path.read_text())
    if rg.sha_array(local_groups) != record["sha256"]["local_groups"]:
        raise AssertionError("local_groups SHA-256 differs from E2's build_record.json")
    graph_path = e2 / "graph.npz"
    graph_sha = rg.sha_file(graph_path)
    if graph_sha != record["sha256"]["graph.npz"]:
        raise AssertionError("graph.npz SHA-256 differs from E2's build_record.json")
    graph = load_npz(graph_path).tocsr()
    if graph.shape != (n, n):
        raise AssertionError(f"graph shape {graph.shape} != ({n}, {n})")
    bank_name = RUNS[run][0]
    bank, bank_sha = load_bank(bank_name, smoke)
    rows = bank.rows()
    if rows.min() < 0 or rows.max() >= n:
        raise AssertionError(f"bank {bank_name} rows leave 0..{n - 1}: [{rows.min()}, {rows.max()}]")
    rg.check_base_config(seed, steps)
    config = run_config(run, seed, steps)
    load_s = perf_counter() - t_load
    rg.log(f"{name}: loaded {n} scorer-train rows, graph {graph.shape}, bank {bank_name} ({len(bank.anchor)} episodes) "
           f"in {load_s:.1f}s; config {dataclasses.asdict(config)}")

    history: dict = {}
    if rg.DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = perf_counter()
    model, img_codes, txt_codes = train_factors(
        data.img_features[st], data.txt_features[st], graph, config, device=rg.DEVICE, group_ids=local_groups,
        aspect_bank=bank, history=history, log_every=50)
    train_s = perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30 if rg.DEVICE == "cuda" else 0.0
    out = {"run": run, "seed": seed, "smoke": smoke, "steps": steps, "bank": bank_name,
           "bank_sha256": bank_sha, "graph_sha256": graph_sha, "build_record_sha256": rg.sha_file(record_path),
           "config": dataclasses.asdict(config), "load_s": load_s, "train_s": train_s,
           "wall_s": perf_counter() - t_load, "peak_gpu_gib": peak, "device": rg.DEVICE,
           "script_sha256": rg.sha_file(Path(__file__)), "history": history}
    if not (np.isfinite(img_codes).all() and np.isfinite(txt_codes).all()):
        (f["res"] / f"failed_{name}.json").write_text(json.dumps({**out, "failure": "non-finite codes"}, indent=1))
        raise SystemExit(f"{name}: non-finite codes; recorded as failed")
    save_factor_checkpoint(model, config, ckpt)
    out.update(checkpoint=str(ckpt.relative_to(ROOT)), checkpoint_sha256=rg.sha_file(ckpt),
               code_stats_scorer_train=rg.code_stats(img_codes, txt_codes))
    hist.write_text(json.dumps(out, indent=1))
    if run in LABEL_RUNS:
        register_checkpoint(f["res"] / "label_checkpoints.json", name, out["checkpoint_sha256"])
    rg.log(f"{name}: {steps} steps in {train_s:.1f}s (load {load_s:.1f}s), peak GPU {peak:.2f} GiB, "
           f"codes {out['code_stats_scorer_train']} -> {ckpt.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True, choices=sorted(RUNS), metavar="RUN", help=f"one of {sorted(RUNS)}")
    ap.add_argument("--seed", type=int, default=42, choices=SEEDS)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 8)))
    train(args.train, args.seed, args.smoke)
    rg.log("done")


if __name__ == "__main__":
    main()
