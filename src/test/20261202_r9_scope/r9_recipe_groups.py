"""Recipe I (ticket 07; rule §3.2 item 1, §3.4, §11): the grouping layer re-fitted from one backbone's features.

    python r9_recipe_groups.py --backbone B [--out DIR] [--stop-after {groups,graph,bank,heads}]

Order and the original code each step calls (nothing of the earlier rounds is edited or copied; the three pieces that
sit inline in a script's `main` are re-run here in the same order, with the same arguments, in this file):
1. `groups`: `partitions.npz` (CLIP's keys `affect, image, caption, local_groups`).
   made:   `image` = `ClipClusterSource(img, txt, scorer_train, n_clusters=64, seed=42)` as `run_selection.py --prepare`
           calls it (its "image" view on scorer-train rows = `clip_image_local` of `run_grid.py` prepare);
           `caption` = `kmeans_partition(txt[scorer_train], arange(n), k=64, seed=42)` as `build_partitions.py` calls it.
   copied: `affect` (affect-km, built from GoEmotions text probabilities) and `local_groups` (one code per painting of
           the frozen split), both from CLIP's `partitions.npz`, never recomputed.
2. `graph`: `graph.npz`, `build_content_graph(img[st], txt[st], GraphConfig())` as `run_grid.py` prepare calls it.
   The other arrays of `grid_prepare.npz` are `local_groups` (copied, above), `clip_image_local` (= `image`, above) and
   `community_local` (the stage-1 community of CLIP features; no role reads it, so it is neither made nor copied).
   `build_content_graph` runs on the GPU when torch sees one (fp16 matmul), else exact fp32 on the CPU; CLIP's stored
   graph was made on the GPU in 5 s. The CPU graph is NOT bit-equal to it (see the ticket 07 log entry).
3. `bank`: `bank_AIC.npz`, the AIC branch of `build_partitions.py`'s bank loop on the new partitions
   (`build_episode_bank`, 65,536 episodes, first rows kept, validated as there).
4. `heads`: `r9_heads` (ticket 05) on the new `partitions.npz`: `n6_posteriors.npz`, `step1_heads_style.npz`
   (CSD labels, the backbone's features).
Frozen and never touched: the split and leakage groups, `partition_L`, `told_oracle.json`, `step1_group_style.npz`.

Every output file gets `<file>.record.json` (SHA-256 of the feature-store files and upstream parts, of this script,
the keys made and the keys copied). No existing output or record is overwritten: all names are checked before any work.
The CLIP replay of stage 0b writes to `results/replay/clip_b32/`.
"""
import argparse
import importlib.util
import json
import sys
import time
from dataclasses import fields
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r9_common as R  # noqa: E402  (first: MAIN and src)
import r9_heads as H  # noqa: E402
from r9_backbone import BACKBONES, STORE_PATHS, parts  # noqa: E402

import numpy as np  # noqa: E402
from scipy.sparse import issparse, save_npz  # noqa: E402
from src.eval.aspect_episodes import AspectEpisodes, PaintingValueIndex  # noqa: E402
from src.model.graph import GraphConfig, build_content_graph  # noqa: E402
from src.train.condition_sources import ClipClusterSource  # noqa: E402
from src.train.pseudo_partitions import build_episode_bank, kmeans_partition  # noqa: E402

K, SEED = 64, 42                          # run_selection.N_CLUSTERS / SEED; build_partitions.K / SEED
N_SCORER = 183_694                        # build_partitions.N_ROWS
BANK_SIZE, MIN_PAINTINGS = 65_536, 30     # build_partitions.BANK_SIZE; build_episode_bank(min_paintings=30)
AIC = ("affect", "image", "caption")      # build_partitions.SETS["AIC"]
STEPS = ("groups", "graph", "bank", "heads")
FILES = {"groups": ["partitions.npz"], "graph": ["graph.npz"], "bank": ["bank_AIC.npz"],
         "heads": ["n6_posteriors.npz", "step1_heads_style.npz"]}
T0 = time.monotonic()


def log(*a):
    print(f"[{time.monotonic() - T0:7.1f}s]", *a, flush=True)


def _load_build_partitions():
    """build_partitions.py's own helpers (validate_set), imported by path; its main() is not run."""
    path = R.TEST / "20261031_pseudo_partitions/build_partitions.py"
    spec = importlib.util.spec_from_file_location("r9_build_partitions_orig", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- the fits (features in, arrays out)

def load_frozen() -> dict:
    """affect-km and local_groups from CLIP's partitions.npz (SHA-256 pinned), the copy source for every backbone."""
    R.assert_input(H.E2_PARTITIONS_REL, "clip_b32")
    with np.load(parts("clip_b32")["affect_km"], allow_pickle=False) as z:
        assert {"affect", "local_groups"} <= set(z.files), "CLIP partitions.npz columns"
        return {key: np.array(z[key]) for key in ("affect", "local_groups")}


def fit_partitions(img, txt, groups, scorer_train, frozen, *, k=K, seed=SEED, min_group_rows=200) -> dict:
    """{affect, image, caption, local_groups} on scorer-train rows (local index = position in scorer_train)."""
    st = np.asarray(scorer_train, dtype=np.int64)
    assert (np.diff(st) > 0).all(), "scorer_train must be sorted and unique"
    n = len(st)
    local_groups = np.asarray(frozen["local_groups"], dtype=np.int64)
    affect = np.asarray(frozen["affect"], dtype=np.int64)
    assert np.array_equal(local_groups, np.unique(np.asarray(groups)[st], return_inverse=True)[1]), \
        "local_groups misaligned with scorer_train"
    assert affect.shape == local_groups.shape == (n,), "copied affect/local_groups do not match scorer_train"
    source = ClipClusterSource(img, txt, st, n_clusters=k, seed=seed, min_group_rows=min_group_rows)
    image = source.labels_by_view["image"][st].astype(np.int64)
    caption = kmeans_partition(np.asarray(txt)[st], np.arange(n), k=k, seed=seed)
    for name, labels in (("image", image), ("caption", caption)):
        assert labels.shape == (n,) and labels.dtype == np.int64, f"{name} labels shape or dtype"
        assert labels.min() >= 0 and len(np.unique(labels)) == k, f"{name}: not {k} groups on every row"
    return {"affect": affect, "image": image, "caption": caption, "local_groups": local_groups}


def fit_graph(img, txt, scorer_train):
    st = np.asarray(scorer_train, dtype=np.int64)
    graph = build_content_graph(np.asarray(img)[st], np.asarray(txt)[st], GraphConfig())
    assert graph.shape == (len(st), len(st)), f"graph shape {graph.shape}"
    return graph


def fit_bank(partitions, *, bank_size=BANK_SIZE, min_paintings=MIN_PAINTINGS) -> dict:
    """bank_AIC.npz arrays: build_partitions.main's loop body for SETS["AIC"] (its first set, so its fresh rng)."""
    bp = _load_build_partitions()
    local_groups = np.asarray(partitions["local_groups"], dtype=np.int64)
    n = len(local_groups)
    parts_ = {k: np.asarray(partitions[k], dtype=np.int64) for k in AIC}
    names = AIC
    rows = np.arange(n)
    index = PaintingValueIndex(parts_, local_groups)
    rng = np.random.default_rng(SEED)
    pairs = [(a, b) for i, a in enumerate(sorted(names)) for b in sorted(names)[i + 1:]]
    sub = {k: parts_[k] for k in names}
    per_pair = -(-bank_size // len(pairs))
    total = bank_size
    bank = build_episode_bank(sub, local_groups, rows, n_per_pair=per_pair, seed=SEED, min_paintings=min_paintings)
    sizes = [per_pair] * len(pairs)
    if len(bank.anchor) > total:
        keep = np.arange(total)                 # first bank_size rows of the concatenation
        bank = AspectEpisodes(bank.aspect_a, bank.aspect_b, *(getattr(bank, f.name)[keep] for f in fields(bank)[2:]))
        sizes[-1] -= sum(sizes) - total
    assert len(bank.anchor) == sum(sizes) == total
    bp.validate_set(bank, sizes, pairs, sorted(names), parts_, local_groups, index, rng)
    assert np.isin(bank.rows(), rows).all()
    arrays = {f.name: getattr(bank, f.name) for f in fields(bank)[2:]}
    return {"aspect_a": np.array(bank.aspect_a), "aspect_b": np.array(bank.aspect_b),
            "block_sizes": np.array(sizes), "block_pairs": np.array([f"{a}__{b}" for a, b in pairs]), **arrays}


# ---------------------------------------------------------------- the heads step (the one place that calls r9_heads)

def fit_heads_step(out, backbone, data, split) -> dict:
    """Fit the five heads on the partitions file just written to ``out`` and write both posterior files there."""
    out = Path(out)
    pins = dict(H.recipe_head_pins(backbone))
    mine = out / "partitions.npz"
    pins[H.E2_PARTITIONS_REL] = (mine, R.sha256_file(mine))      # not CLIP's file: the one this run made
    heads = H.fit_heads_r9(data, split.groups, split.scorer_train, backbone=backbone, input_pins=pins)
    shas = H.recipe_input_shas(backbone, pins)
    return H.write_posteriors(out, heads, data, split.selection, backbone=backbone, input_sha256=shas,
                              script_path=__file__)


# ---------------------------------------------------------------- files, records, refusal

def output_names(stop_after) -> list:
    assert stop_after in STEPS, f"--stop-after must be one of {STEPS}"
    return [name for step in STEPS[:STEPS.index(stop_after) + 1] for name in FILES[step]]


def preflight(out, backbone, stop_after):
    assert backbone in BACKBONES, f"unknown backbone: {backbone}"
    for name in output_names(stop_after):
        for target in (Path(out) / name, Path(out) / (name + ".record.json")):
            if target.exists():
                raise FileExistsError(f"refusing to overwrite {target}")


def store_shas(backbone) -> dict:
    root = STORE_PATHS[backbone]
    files = (sorted(root.rglob("*.npy")) + [root / "metadata.json"] if backbone == "clip_b32"
             else [root / name for name in ("img.npy", "txt.npy", "index.json")])
    assert files and all(p.is_file() for p in files), f"{root}: missing feature-store files"
    return {str(p): R.sha256_file(p) for p in files}


def write_part(out, name, arrays, *, backbone, input_sha256, made, copied):
    """One npz (or sparse matrix) plus its record. Exclusive creation: an existing file is never replaced."""
    out = Path(out)
    path, rec_path = out / name, out / (name + ".record.json")
    for target in (path, rec_path):
        if target.exists():
            raise FileExistsError(f"refusing to overwrite {target}")
    if not issparse(arrays):
        assert set(arrays) == set(made) | set(copied), f"{name}: keys made+copied differ from keys written"
    out.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as file:
        if issparse(arrays):
            save_npz(file, arrays)
        else:
            np.savez(file, **arrays)
    rec = {"backbone": backbone, "input_sha256": dict(input_sha256), "script": str(Path(__file__)),
           "script_sha256": R.sha256_file(__file__), "keys_made": list(made), "keys_copied": list(copied)}
    with rec_path.open("x") as file:
        json.dump(rec, file, indent=2)
        file.write("\n")
    return rec


# ---------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", choices=BACKBONES, required=True)
    ap.add_argument("--out", type=Path, default=None, help="default results/recipe/<backbone>/")
    ap.add_argument("--stop-after", choices=STEPS, default="heads",
                    help="run the steps up to this one (a partial run is a subset check; default: all)")
    args = ap.parse_args(argv)
    out = args.out or R.RESULTS / "recipe" / args.backbone
    H.require_threads()
    preflight(out, args.backbone, args.stop_after)
    last = STEPS.index(args.stop_after)

    data = R.load_data(args.backbone)
    split = R.load_split(data)
    st = split.scorer_train
    assert len(st) == N_SCORER, f"scorer_train has {len(st)} rows, not {N_SCORER}"
    log(f"{args.backbone}: data and split loaded")
    base = {**store_shas(args.backbone), str(R.INPUT_PATHS[R.PREPARE_REL]): R.assert_input(R.PREPARE_REL),
            str(R.INPUT_PATHS[R.HELD_CODES_REL]): R.assert_input(R.HELD_CODES_REL)}

    frozen = load_frozen()
    frozen_sha = {str(parts("clip_b32")["affect_km"]): R.assert_input(H.E2_PARTITIONS_REL, "clip_b32")}
    new_partitions = fit_partitions(data.img_features, data.txt_features, split.groups, st, frozen)
    log("image k-means (ClipClusterSource) and caption k-means (kmeans_partition) done")
    write_part(out, "partitions.npz", new_partitions, backbone=args.backbone, input_sha256={**base, **frozen_sha},
               made=["image", "caption"], copied=["affect", "local_groups"])
    if last >= 1:
        graph = fit_graph(data.img_features, data.txt_features, st)
        write_part(out, "graph.npz", graph, backbone=args.backbone, input_sha256=base,
                   made=["graph"], copied=[])
        log(f"graph done ({graph.nnz // 2:,} edges)")
    if last >= 2:
        bank = fit_bank(new_partitions)
        mine = out / "partitions.npz"
        write_part(out, "bank_AIC.npz", bank, backbone=args.backbone,
                   input_sha256={str(mine): R.sha256_file(mine)}, made=list(bank), copied=[])
        log(f"bank AIC done ({len(bank['anchor']):,} episodes)")
    if last >= 3:
        fit_heads_step(out, args.backbone, data, split)
        log("heads done")
    log(f"done -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
