"""CoSiR v2 Candidate A affect-signal factor learning, cells E and SE (spec docs/superpowers/specs/2026-09-30-cosir-v2-candidate-a-affect-factor-learning-design.md).

Factor encoders trained like the 2x2's S cell (R3 config, pair agreement, condition episodes) but with the condition
source built from GoEmotions affect clusters of the captions: E (affect partition alone) and SE (affect partition and
the CLIP image partition, each drawn with probability 1/2). C0 (the 2x2's matched control) and S (reference row) are
existing checkpoints of the previous grid. Run from the repository root:

    python src/test/20261018_affect_factor_learning/run_affect.py --prepare      # affect extraction, k-means, diagnostics
    python src/test/20261018_affect_factor_learning/run_affect.py --smoke        # timing: local GPU or DAS6?
    python src/test/20261018_affect_factor_learning/run_affect.py --run E --seed 42    # refuses an existing
                                                                                       # checkpoint (--overwrite)
    python src/test/20261018_affect_factor_learning/run_affect.py --evaluate     # Task 3 (not yet implemented)
    python src/test/20261018_affect_factor_learning/run_affect.py --replicate    # Task 3 (not yet implemented)
    python src/test/20261018_affect_factor_learning/run_affect.py --tables       # Task 3 (not yet implemented)

Row scope (hard rule): only scorer-train captions are ever passed to the affect model. Selection, val and held
captions never are. The diagnostic probe uses a painting-grouped 80/20 split INSIDE scorer-train.
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
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

_GRID_PATH = ROOT / "src/test/20261016_factor_learning_grid/run_grid.py"
_gspec = importlib.util.spec_from_file_location("run_grid", _GRID_PATH)
grid = importlib.util.module_from_spec(_gspec)
_gspec.loader.exec_module(grid)

from src.data.affect import goemotions_probabilities, load_goemotions  # noqa: E402
from src.data.artelingo import ANNOTATIONS_PATH, join_captions, load_artelingo  # noqa: E402
from src.data.splits import grouped_subsplit  # noqa: E402
from src.train.condition_sources import CommunitySource, MultiPartitionSource  # noqa: E402
from src.train.train_factors import load_factor_checkpoint, save_factor_checkpoint, train_factors  # noqa: E402

SEED = 42
CELLS = ("E", "SE")
AFFECT_K = 64
KMEANS_SETTINGS = {"n_clusters": AFFECT_K, "random_state": SEED, "n_init": 3, "batch_size": 4096}
DIAG_HELDOUT_FRACTION = 0.2                       # painting-grouped split INSIDE scorer-train, diagnostic only
C0_REF = grid.CKPT / "C0_seed42.pt"               # the 2x2's matched control
S_REF = grid.CKPT / "S_seed42.pt"                 # the 2x2's style cell, reference row only
CACHE, CKPT, RESULTS = HERE / "cache", HERE / "checkpoints", HERE / "results"
GRID_SMOKE_JSON = _GRID_PATH.parent / "results" / "smoke_timing.json"
EXPECTED_AFFECT_SHAPE = (183_694, 28)
MIN_GROUP_ROWS = 200
log = grid.log


# ----------------------------------------------------------------------------- prepare

def _ami(labels_a, labels_b) -> float:
    return float(adjusted_mutual_info_score(labels_a, labels_b))


def _probe_accuracy(x, y, first_idx, second_idx) -> float:
    scaler = StandardScaler().fit(x[first_idx])
    clf = LogisticRegression(C=1.0, max_iter=1000).fit(scaler.transform(x[first_idx]), y[first_idx])
    return float((clf.predict(scaler.transform(x[second_idx])) == y[second_idx]).mean())


def prepare() -> None:
    started = perf_counter()
    for folder in (CACHE, CKPT, RESULTS):
        folder.mkdir(exist_ok=True)
    cache, prep, meta, _ = grid.load_grid()
    data = load_artelingo()
    st = cache["scorer_train"]

    # Captions: ONLY scorer-train rows are joined and passed to the affect model (row-scope rule).
    annotations = json.loads(Path(ANNOTATIONS_PATH).read_text())
    captions = join_captions(data.sample_ids[st], annotations)
    if len(captions) != len(st):
        raise AssertionError("captions must be one per scorer-train row")
    for held_out in ("selection", "val", "held"):
        if held_out in cache and np.intersect1d(cache[held_out], st).size:
            raise AssertionError(f"{held_out} rows overlap scorer-train: row scope violated")
    del annotations

    t0 = perf_counter()
    loaded = load_goemotions(device=grid.DEVICE)
    affect = goemotions_probabilities(captions, loaded=loaded, batch_size=256, max_length=64)
    extract_seconds = perf_counter() - t0
    if affect.shape != EXPECTED_AFFECT_SHAPE:
        raise AssertionError(f"affect shape {affect.shape}, expected {EXPECTED_AFFECT_SHAPE}")
    if not (np.isfinite(affect).all() and affect.min() >= 0.0 and affect.max() <= 1.0):
        raise AssertionError("affect probabilities must be finite and in [0, 1]")
    log(f"Affect extracted: {affect.shape} in {extract_seconds:.1f} s (device {grid.DEVICE})")
    del loaded
    if grid.DEVICE == "cuda":
        torch.cuda.empty_cache()

    # Partition: k-means on the raw vectors.
    t0 = perf_counter()
    affect_local = MiniBatchKMeans(**KMEANS_SETTINGS).fit_predict(affect).astype(np.int64)
    kmeans_seconds = perf_counter() - t0
    sizes = np.bincount(affect_local, minlength=AFFECT_K)
    n_valid = int((sizes >= MIN_GROUP_ROWS).sum())
    log(f"k-means: {int((sizes > 0).sum())} non-empty groups, {n_valid} with >= {MIN_GROUP_ROWS} rows; "
        f"sizes min/median/max = {sizes.min()}/{int(np.median(sizes))}/{sizes.max()} ({kmeans_seconds:.1f} s)")

    # References: hashes and config asserts.
    for path, cell in ((C0_REF, "C0"), (S_REF, "S")):
        _, stored = load_factor_checkpoint(path, device="cpu")
        if dataclasses.asdict(stored) != dataclasses.asdict(grid.cell_config(cell, 42)):
            raise AssertionError(f"{path.name}: stored config differs from grid.cell_config({cell!r}, 42)")
    c0_sha, s_sha = grid.sha256_file(C0_REF), grid.sha256_file(S_REF)
    log(f"Reference checkpoints config-asserted: C0 {c0_sha[:12]}, S {s_sha[:12]}")

    # Diagnostics (measured only).
    emotion, style = data.emotions[st], data.art_styles[st]
    partitions = {"affect_k64": affect_local, "clip_image": prep["clip_image_local"],
                  "clip_caption": cache["clip_caption"][st]}
    ami = {name: {"emotion": _ami(labels, emotion), "art_style": _ami(labels, style)}
           for name, labels in partitions.items()}
    # local indices into st: painting-grouped 80/20 split inside scorer-train
    first, second = grouped_subsplit(cache["groups"], st, DIAG_HELDOUT_FRACTION, seed=42)
    if not (np.isin(first, st).all() and np.isin(second, st).all()):
        raise AssertionError("probe split must stay inside scorer-train")
    if np.intersect1d(cache["groups"][first], cache["groups"][second]).size:
        raise AssertionError("a painting spans the probe split")
    pos = np.full(len(cache["groups"]), -1, dtype=np.int64)
    pos[st] = np.arange(len(st))
    first_l, second_l = pos[first], pos[second]
    _, emo_ids = np.unique(emotion, return_inverse=True)
    majority = float(np.bincount(emo_ids[second_l]).max() / len(second_l))
    probe = {"affect28_to_emotion": _probe_accuracy(affect, emo_ids, first_l, second_l),
             "clip_caption_to_emotion": _probe_accuracy(data.txt_features[st], emo_ids, first_l, second_l),
             "majority_class": majority, "train_rows": int(len(first)), "test_rows": int(len(second))}
    log(f"AMI (scorer-train rows): {json.dumps(ami, indent=2)}")
    log(f"Probe accuracies on the held-out 20% of scorer-train paintings: {json.dumps(probe, indent=2)}")

    np.savez(CACHE / "affect_prepare.npz", affect_probs=affect.astype(np.float32), affect_local=affect_local)
    record = {"affect_npz_sha256": grid.sha256_file(CACHE / "affect_prepare.npz"),
              "affect_probs_sha256": hashlib.sha256(np.ascontiguousarray(affect).tobytes()).hexdigest(),
              "c0_ref_sha256": c0_sha, "s_ref_sha256": s_sha,
              "affect_shape": list(affect.shape), "kmeans_settings": KMEANS_SETTINGS,
              "group_sizes": sizes.tolist(), "groups_nonempty": int((sizes > 0).sum()),
              "groups_ge_min_rows": n_valid, "min_group_rows": MIN_GROUP_ROWS,
              "ami": ami, "probe": probe,
              "timings": {"extract_seconds": extract_seconds, "kmeans_seconds": kmeans_seconds,
                          "total_seconds": perf_counter() - started}, "device": grid.DEVICE}
    (CACHE / "affect_prepare.json").write_text(json.dumps(record, indent=2))
    log(f"Prepared in {perf_counter() - started:.1f} s")


# ----------------------------------------------------------------------------- cells

def affect_cell_config(cell: str, seed: int, steps: int = grid.FULL_STEPS):
    return grid.cell_config("C0" if cell == "C0" else "S", seed, steps)


def affect_source(cell: str, prep_grid: dict, affect_local: np.ndarray):
    n = len(affect_local)
    if cell == "E":
        return CommunitySource(affect_local, np.arange(n))
    if cell == "SE":
        return MultiPartitionSource({"affect": affect_local, "image": prep_grid["clip_image_local"]}, np.arange(n))
    if cell == "C0":
        return None
    raise ValueError(cell)


def run_affect_cell(cell: str, seed: int, steps: int = grid.FULL_STEPS, tag: str = "",
                    overwrite: bool = False) -> dict:
    name = f"{cell}_seed{seed}{tag}"
    if not tag and not overwrite and (CKPT / f"{name}.pt").exists():   # full runs only; smoke tags may rerun
        raise FileExistsError(f"{CKPT / f'{name}.pt'} exists: it may be evidence behind a report. Refusing to "
                              f"overwrite it; pass --overwrite to retrain and replace it.")
    for folder in (CKPT, RESULTS):
        folder.mkdir(exist_ok=True)
    cache, prep, _, graph = grid.load_grid()
    data = load_artelingo()
    st = cache["scorer_train"]
    affect_local = np.load(CACHE / "affect_prepare.npz")["affect_local"]
    if len(affect_local) != len(st):
        raise AssertionError("affect labels must cover scorer-train rows")
    config = affect_cell_config(cell, seed, steps)
    source = affect_source(cell, prep, affect_local)
    history: dict = {}
    if grid.DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = perf_counter()
    out = io.StringIO()
    with redirect_stdout(out):                                  # train_factors prints one line per step
        model, img_codes, txt_codes = train_factors(
            data.img_features[st], data.txt_features[st], graph, config, device=grid.DEVICE,
            group_ids=prep["local_groups"], condition_source=source, history=history)
    seconds = perf_counter() - t0
    if not (np.isfinite(img_codes).all() and np.isfinite(txt_codes).all()):
        raise AssertionError(f"{cell} seed {seed}: non-finite codes")
    peak = torch.cuda.max_memory_allocated() / 2**30 if grid.DEVICE == "cuda" else 0.0
    save_factor_checkpoint(model, config, CKPT / f"{name}.pt")
    views = sorted({key[0] for key in source.valid_keys}) if source is not None else []
    record = {"cell": cell, "source": cell, "source_views": views, "seed": seed, "steps": steps,
              "seconds": seconds, "peak_gpu_gib": peak, "history": history,
              "last_print": out.getvalue().strip().splitlines()[-1],
              "config": dataclasses.asdict(config)}
    (RESULTS / f"history_{name}.json").write_text(json.dumps(record, indent=2))
    log(f"{name}: {steps} steps in {seconds:.1f} s, peak GPU {peak:.2f} GiB")
    return record


# ----------------------------------------------------------------------------- smoke

def smoke() -> dict:
    per_cell = {}
    for cell in CELLS:
        short, long = (run_affect_cell(cell, SEED, steps, tag=f"_smoke{steps}") for steps in grid.SMOKE_STEPS)
        per_step = (long["seconds"] - short["seconds"]) / (grid.SMOKE_STEPS[1] - grid.SMOKE_STEPS[0])
        fixed = max(short["seconds"] - per_step * grid.SMOKE_STEPS[0], 0.0)
        per_cell[cell] = {"seconds_per_step": per_step, "fixed_seconds": fixed,
                          "projected_full_seconds": fixed + per_step * grid.FULL_STEPS,
                          "peak_gpu_gib": max(short["peak_gpu_gib"], long["peak_gpu_gib"])}
    c0 = json.loads(GRID_SMOKE_JSON.read_text())["per_cell"]["C0"]
    c0_full = c0["fixed_seconds"] + c0["seconds_per_step"] * grid.FULL_STEPS
    slowest = max(v["projected_full_seconds"] for v in per_cell.values())
    selection = sum(v["projected_full_seconds"] for v in per_cell.values())
    replication = 2 * (slowest + c0_full)                      # picked cell unknown: the slower cell
    total = selection + replication
    peak = max(max(v["peak_gpu_gib"] for v in per_cell.values()), c0["peak_gpu_gib"])
    heavy = slowest > grid.HEAVY_RUN_SECONDS or total > grid.HEAVY_TOTAL_SECONDS or peak > grid.HEAVY_PEAK_GIB
    result = {"per_cell": per_cell,
              "c0_from_grid_smoke": {"seconds_per_step": c0["seconds_per_step"], "projected_full_seconds": c0_full,
                                     "peak_gpu_gib": c0["peak_gpu_gib"]},
              "projected_selection_seconds": selection, "projected_replication_seconds": replication,
              "projected_total_training_seconds_sequential": total, "peak_gpu_gib": peak,
              "thresholds": {"run_seconds": grid.HEAVY_RUN_SECONDS, "total_seconds": grid.HEAVY_TOTAL_SECONDS,
                             "peak_gib": grid.HEAVY_PEAK_GIB},
              "decision": "ask_user_for_das6_node" if heavy else "run_locally", "device": grid.DEVICE}
    (RESULTS / "smoke_timing.json").write_text(json.dumps(result, indent=2))
    log(f"Smoke timing: {json.dumps(result, indent=2)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run", choices=CELLS + ("C0",))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--replicate", action="store_true")
    parser.add_argument("--tables", action="store_true")
    args = parser.parse_args()
    if args.evaluate or args.replicate or args.tables:
        raise NotImplementedError("Task 3")
    if args.prepare:
        prepare()
    if args.smoke:
        smoke()
    if args.run:
        run_affect_cell(args.run, args.seed, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
