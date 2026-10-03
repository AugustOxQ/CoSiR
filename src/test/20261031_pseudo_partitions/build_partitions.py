"""E2: pseudo-partitions over scorer-train rows and the pseudo-aspect training episode banks AIC / AI / IC.

Run from the repo root:  python src/test/20261031_pseudo_partitions/build_partitions.py [--smoke]
Works on scorer-train LOCAL rows 0..n-1. No evaluation label touches anything saved for training: the AMI table is a
report-only diagnostic and is computed after the banks are written.
"""
import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import fields
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.sparse import load_npz
from sklearn.metrics import adjusted_mutual_info_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from src.data.artelingo import load_artelingo  # noqa: E402
from src.data.artelingo_splits import artelingo_aspect_labels, artelingo_splits  # noqa: E402
from src.eval.aspect_episodes import (AspectEpisodes, PaintingValueIndex, concat_episodes,  # noqa: E402
                                      validate_aspect_episodes)
from src.train.pseudo_partitions import build_episode_bank, kmeans_partition  # noqa: E402

STAGE_D = ROOT / "src/test/20261013_stage_d_selection/cache/prepare.npz"
GRID = ROOT / "src/test/20261016_factor_learning_grid/cache"
AFFECT = ROOT / "src/test/20261018_affect_factor_learning/cache/affect_prepare.npz"
N_ROWS, BANK_SIZE, SEED, K = 183_694, 65_536, 42, 64
VALIDATE_N = 1000
SETS = {"AIC": ("affect", "image", "caption"), "AI": ("affect", "image"), "IC": ("image", "caption")}
T0 = perf_counter()


def log(*a):
    print(f"[{perf_counter() - T0:7.1f}s]", *a, flush=True)


def sha(arr) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def block_slices(sizes):
    edges = np.concatenate([[0], np.cumsum(sizes)])
    return [slice(int(edges[i]), int(edges[i + 1])) for i in range(len(sizes))]


def subset(ep: AspectEpisodes, name_a, name_b, idx) -> AspectEpisodes:
    return AspectEpisodes(name_a, name_b, *(getattr(ep, f.name)[idx] for f in fields(ep)[2:]))


def validate_set(bank, sizes, pairs, names, labels, groups, index, rng):
    sample = np.sort(rng.choice(len(bank.anchor), size=min(VALIDATE_N, len(bank.anchor)), replace=False))
    checked = {}
    for sl, (a, b) in zip(block_slices(sizes), pairs):
        idx = sample[(sample >= sl.start) & (sample < sl.stop)]
        rest = [n for n in names if n not in (a, b)]
        third = rest[0] if len(rest) == 1 else None
        validate_aspect_episodes(subset(bank, a, b, idx), labels, groups, index, third=third)
        checked[f"{a}__{b}"] = int(len(idx))
    return checked


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="256 episodes per pair, writes to results/smoke/")
    args = ap.parse_args()
    out = HERE / "results" / ("smoke" if args.smoke else "")
    out.mkdir(parents=True, exist_ok=True)
    rec = {"smoke": args.smoke, "timings": {}, "sha256": {}}

    # ---- alignment: the caches must be in the row order of artelingo_splits().scorer_train
    t0 = perf_counter()
    data = load_artelingo()
    splits = artelingo_splits(data)
    st = np.asarray(splits.scorer_train)
    n = len(st)
    assert n == N_ROWS, n
    stage = np.load(STAGE_D)
    assert np.array_equal(stage["scorer_train"], st), "stage (d) cache scorer_train order differs from artelingo_splits"
    assert np.array_equal(stage["groups"], splits.groups), "stage (d) cache groups differ"
    grid = np.load(GRID / "grid_prepare.npz")
    _, groups_expected = np.unique(splits.groups[st], return_inverse=True)
    assert np.array_equal(grid["local_groups"], groups_expected), "local_groups misaligned with scorer_train order"
    assert np.array_equal(grid["clip_image_local"], stage["clip_image"][st]), "clip_image_local misaligned"
    affect_local = np.load(AFFECT)["affect_local"]
    image, local_groups = grid["clip_image_local"].astype(np.int64), grid["local_groups"].astype(np.int64)
    affect_local = affect_local.astype(np.int64)
    for name, arr in (("affect", affect_local), ("image", image), ("local_groups", local_groups)):
        assert arr.shape == (n,), f"{name} length {arr.shape}"
    assert (affect_local >= 0).all() and (image >= 0).all()
    rec["alignment"] = ("scorer_train order == stage (d) cache order == artelingo_splits order; groups equal; "
                        "local_groups == np.unique(groups[st], return_inverse); clip_image_local == "
                        "cache clip_image[st]; affect_local checked by length and by AMI(affect, emotion) vs the "
                        "affect_prepare.json record (below)")
    log(f"alignment verified (n={n}) in {perf_counter() - t0:.1f}s")

    # ---- graph
    graph_src, graph_dst = GRID / "graph.npz", out / "graph.npz"
    g = load_npz(graph_src)
    assert g.shape == (n, n), g.shape
    shutil.copyfile(graph_src, graph_dst)
    rec["sha256"]["graph.npz"] = sha_file(graph_dst)
    assert rec["sha256"]["graph.npz"] == sha_file(graph_src)
    rec["graph"] = {"shape": list(g.shape), "edges_directed": int(g.nnz), "copied": True}
    del g

    # ---- partitions
    t0 = perf_counter()
    caption = kmeans_partition(data.txt_features[st], np.arange(n), k=K, seed=SEED)
    rec["timings"]["caption_kmeans_s"] = perf_counter() - t0
    parts = {"affect": affect_local, "image": image, "caption": caption}
    for name, arr in parts.items():
        rec["sha256"][name] = sha(arr)
    rec["sha256"]["local_groups"] = sha(local_groups)
    rec["partition_sizes"] = {k: {"clusters": int(len(np.unique(v))), "min": int(np.bincount(v).min()),
                                  "median": int(np.median(np.bincount(v))), "max": int(np.bincount(v).max())}
                              for k, v in parts.items()}
    np.savez(out / "partitions.npz", affect=affect_local, image=image, caption=caption, local_groups=local_groups)
    log(f"partitions: {rec['partition_sizes']}")

    # ---- banks
    rows = np.arange(n)
    index = PaintingValueIndex(parts, local_groups)
    rng = np.random.default_rng(SEED)
    rec["banks"] = {}
    for name, names in SETS.items():
        t0 = perf_counter()
        pairs = [(a, b) for i, a in enumerate(sorted(names)) for b in sorted(names)[i + 1:]]
        sub = {k: parts[k] for k in names}
        if len(pairs) == 1:
            per_pair, total = (256 if args.smoke else BANK_SIZE), None
        else:
            per_pair = 256 if args.smoke else -(-BANK_SIZE // len(pairs))
            total = len(pairs) * per_pair if args.smoke else BANK_SIZE
        bank = build_episode_bank(sub, local_groups, rows, n_per_pair=per_pair, seed=SEED, min_paintings=30)
        sizes = [per_pair] * len(pairs)
        if total is not None and len(bank.anchor) > total:
            keep = np.arange(total)             # first BANK_SIZE rows of the concatenation
            bank = AspectEpisodes(bank.aspect_a, bank.aspect_b, *(getattr(bank, f.name)[keep] for f in fields(bank)[2:]))
            sizes[-1] -= sum(sizes) - total
        assert len(bank.anchor) == sum(sizes) == (total or per_pair)
        build_s = perf_counter() - t0
        t0 = perf_counter()
        checked = validate_set(bank, sizes, pairs, sorted(names), parts, local_groups, index, rng)
        validate_s = perf_counter() - t0
        assert np.isin(bank.rows(), rows).all()
        arrays = {f.name: getattr(bank, f.name) for f in fields(bank)[2:]}
        np.savez(out / f"bank_{name}.npz", aspect_a=np.array(bank.aspect_a), aspect_b=np.array(bank.aspect_b),
                 block_sizes=np.array(sizes), block_pairs=np.array([f"{a}__{b}" for a, b in pairs]), **arrays)
        rec["sha256"][f"bank_{name}.npz"] = sha_file(out / f"bank_{name}.npz")
        rec["banks"][name] = {"partitions": list(names), "episodes": int(len(bank.anchor)), "block_sizes": sizes,
                              "block_pairs": [f"{a}__{b}" for a, b in pairs], "validated_per_block": checked,
                              "validation": "passed", "build_s": build_s, "validate_s": validate_s,
                              "s_per_episode": build_s / len(bank.anchor)}
        log(f"bank {name}: {rec['banks'][name]}")
    rec["timings"]["banks_s"] = sum(b["build_s"] for b in rec["banks"].values())

    # ---- diagnostic AMI (report only; training labels are read nowhere above)
    t0 = perf_counter()
    labels = artelingo_aspect_labels(data)
    ami = {}
    for pname, p in parts.items():
        ami[pname] = {}
        for lname, lab in labels.items():
            y = lab[st]
            ok = y >= 0
            ami[pname][lname] = float(adjusted_mutual_info_score(y[ok], p[ok]))
            ami[pname][f"{lname}_rows"] = int(ok.sum())
    rec["ami"] = ami
    rec["timings"]["ami_s"] = perf_counter() - t0
    ref = json.loads((ROOT / "src/test/20261018_affect_factor_learning/cache/affect_prepare.json").read_text())["ami"]
    rec["ami_vs_record"] = {"affect_emotion": [ami["affect"]["emotion"], ref["affect_k64"]["emotion"]],
                            "image_style": [ami["image"]["style"], ref["clip_image"]["art_style"]]}
    assert ami["affect"]["emotion"] > 0.1 and ami["image"]["style"] > 0.2, "affect/image partitions look misaligned"
    log(f"AMI: {json.dumps(ami)}")
    log(f"AMI vs recorded: {rec['ami_vs_record']}")

    rec["timings"]["total_s"] = perf_counter() - T0
    (out / "build_record.json").write_text(json.dumps(rec, indent=2))
    log(f"done -> {out}")


if __name__ == "__main__":
    main()
