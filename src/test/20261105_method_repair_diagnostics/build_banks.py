"""Method-repair diagnostics: the LAB bank (evaluation labels of scorer-train rows; H3 diagnostic only) and the MK
bank (label-free k-means at matched granularity). Rules: PREREGISTRATION.md §3. Run from the repo root:

  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /root/miniconda3/envs/CoSiR/bin/python \
      src/test/20261105_method_repair_diagnostics/build_banks.py [--smoke]
"""
import argparse
import sys
import json
from pathlib import Path
from dataclasses import fields
from itertools import combinations
from time import perf_counter

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_mutual_info_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import AFFECT_CACHE, BANK_SEEDS, BANK_SIZE, LAB_PARTS, MK_K, folders, local_rows, rg
from src.data.artelingo import load_artelingo
from src.data.artelingo_splits import artelingo_aspect_labels, artelingo_splits
from src.eval.aspect_episodes import AspectEpisodes, PaintingValueIndex, eligible_values, validate_aspect_episodes
from src.train.pseudo_partitions import build_episode_bank, kmeans_partition

VALIDATE_N = 1000


def block_slices(sizes):
    edges = np.concatenate([[0], np.cumsum(sizes)])
    return [slice(int(edges[i]), int(edges[i + 1])) for i in range(len(sizes))]


def subset(ep, name_a, name_b, idx):
    return AspectEpisodes(name_a, name_b, *(getattr(ep, f.name)[idx] for f in fields(ep)[2:]))


def build(name, parts, groups, n, smoke, rng):
    names = sorted(parts)
    pairs = list(combinations(names, 2))
    per_pair = 256 if smoke else -(-BANK_SIZE // len(pairs))
    total = len(pairs) * per_pair if smoke else BANK_SIZE
    t0 = perf_counter()
    bank = build_episode_bank(parts, groups, np.arange(n), n_per_pair=per_pair, seed=BANK_SEEDS[name],
                              min_paintings=30)
    sizes = [per_pair] * len(pairs)
    if len(bank.anchor) > total:                     # first BANK_SIZE rows of the concatenation, as in E2
        keep = np.arange(total)
        bank = AspectEpisodes(bank.aspect_a, bank.aspect_b, *(getattr(bank, f.name)[keep] for f in fields(bank)[2:]))
        sizes[-1] -= sum(sizes) - total
    assert len(bank.anchor) == sum(sizes) == total
    assert bank.rows().min() >= 0 and bank.rows().max() < n
    build_s = perf_counter() - t0
    index = PaintingValueIndex(parts, groups)
    sample = np.sort(rng.choice(len(bank.anchor), size=min(VALIDATE_N, len(bank.anchor)), replace=False))
    checked = {}
    for sl, (a, b) in zip(block_slices(sizes), pairs):
        idx = sample[(sample >= sl.start) & (sample < sl.stop)]
        third = [x for x in names if x not in (a, b)][0]
        validate_aspect_episodes(subset(bank, a, b, idx), parts, groups, index, third=third)
        checked[f"{a}__{b}"] = int(len(idx))
    return bank, sizes, pairs, checked, build_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="256 episodes per pair, writes to results/smoke/")
    args = ap.parse_args()
    out = folders(args.smoke)["res"]
    out.mkdir(parents=True, exist_ok=True)
    for f in ("bank_LAB.npz", "bank_MK.npz", "build_record.json"):
        if not args.smoke and (out / f).exists():
            raise FileExistsError(f"{out / f} exists; refusing to overwrite pre-registered inputs")
    data = load_artelingo()
    splits = artelingo_splits(data)
    st, groups = local_rows(data, splits)
    n = len(st)
    labels = artelingo_aspect_labels(data)
    lab = {k: labels[k][st].astype(np.int64) for k in LAB_PARTS}
    probs = np.load(AFFECT_CACHE)["affect_probs"]
    assert probs.shape == (n, 28), probs.shape
    mk = {
        "affect8": MiniBatchKMeans(n_clusters=MK_K["affect8"], random_state=42, n_init=3, batch_size=4096)
        .fit_predict(probs).astype(np.int64),
        "caption10": kmeans_partition(data.txt_features[st], np.arange(n), k=MK_K["caption10"], seed=42),
        "image23": kmeans_partition(data.img_features[st], np.arange(n), k=MK_K["image23"], seed=42),
    }
    rec = {"smoke": args.smoke, "n_rows": n, "sha256": {}, "banks": {}, "eligible_values": {}, "mk_ami": {}}
    rng = np.random.default_rng(42)
    for name, parts in (("LAB", lab), ("MK", mk)):
        rec["eligible_values"][name] = {k: len(eligible_values(v, groups, np.arange(n)[v >= 0], 30))
                                        for k, v in parts.items()}
        bank, sizes, pairs, checked, build_s = build(name, parts, groups, n, args.smoke, rng)
        arrays = {f.name: getattr(bank, f.name) for f in fields(bank)[2:]}
        np.savez(out / f"bank_{name}.npz", aspect_a=np.array(bank.aspect_a), aspect_b=np.array(bank.aspect_b),
                 block_sizes=np.array(sizes), block_pairs=np.array([f"{a}__{b}" for a, b in pairs]), **arrays)
        np.savez(out / f"partitions_{name}.npz", **parts, local_groups=groups)
        for f in (f"bank_{name}.npz", f"partitions_{name}.npz"):
            rec["sha256"][f] = rg.sha_file(out / f)
        rec["banks"][name] = {"partitions": sorted(parts), "episodes": int(len(bank.anchor)), "block_sizes": sizes,
                              "block_pairs": [f"{a}__{b}" for a, b in pairs], "validated_per_block": checked,
                              "validation": "passed", "build_s": build_s, "seed": BANK_SEEDS[name]}
        print(f"bank {name}: {rec['banks'][name]}", flush=True)
    for p, v in mk.items():                          # descriptive diagnostic; labels used only here
        rec["mk_ami"][p] = {a: float(adjusted_mutual_info_score(lab[a][lab[a] >= 0], v[lab[a] >= 0]))
                            for a in LAB_PARTS}
    (out / "build_record.json").write_text(json.dumps(rec, indent=1))
    print("build_record:", json.dumps(rec["mk_ami"]), flush=True)


if __name__ == "__main__":
    main()
