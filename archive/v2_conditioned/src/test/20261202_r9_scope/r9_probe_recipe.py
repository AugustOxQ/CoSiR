"""Probe recipe, part 1 (ticket 27): affect grouping and the recipe stages on a CUB or SemArt fit split. Exploration:
no verdict, no bar, no unit tests. Library for `run_r9_probe.py`.

Row space: every caption row is one row (CUB: 10 per image; SemArt: 1 per painting), image features repeated per
caption row, as ArtELingo's rows are. `groups` = the unit (image / painting) of the row (episode and head-draw
distinctness, as ArtELingo's painting groups). Leakage between fit and selection rows: CUB by species, SemArt by painting.

Split: the fit rows are cut by `grouped_subsplit(leak, fit_rows, 0.15, seed=42)` into scorer-train (heads, groupings,
A3, banks) and selection (picks, tau), as artelingo_splits does; the scoring rows are plan.json's scoring split.

Each recipe function below is the recipe's own (r9_recipe_groups, r9_recipe_reader, run_gonogo, rb_build) called on the
corpus's arrays; the ones adapted here are marked ADAPTED with the reason. Nothing in an existing r9 file is edited.
"""
import json
import pickle
import time
from pathlib import Path
from types import SimpleNamespace

import r9_common as R  # first: MAIN and src on the path
import numpy as np
import sklearn
import torch
from scipy.sparse import load_npz, save_npz
from sklearn.cluster import MiniBatchKMeans

import r9_heads as H
import r9_recipe_groups as G
import r9_recipe_reader as RR
from r9_probe_data import HERE, load_probe_data
from src.data.affect import goemotions_probabilities, load_goemotions
from src.data.splits import grouped_subsplit
from src.eval.aspect_episodes import AspectEpisodes
from src.model.communities import detect_communities

A0, rb, rf = RR.A0, RR.rb, RR.rf
SEED = 42
SELECTION_FRACTION = {"semart": 0.15, "cub": 0.30}    # CUB: 15% (about 1,000 images, 240 with all three labels) leaves
# under 5 eligible values per aspect for the episode builder; 30% gives 6 per aspect (see r9_probe_score.CONFIG)
AFFECT_KM = {"n_clusters": 64, "random_state": SEED, "n_init": 3, "batch_size": 4096}   # run_affect.KMEANS_SETTINGS
T0 = time.monotonic()


def log(*a):
    print(f"[{time.monotonic() - T0:7.1f}s]", *a, flush=True)


def out_dir(dataset, limit=None):
    d = HERE / "results/probes" / (dataset if not limit else f"{dataset}_limit{limit}")
    d.mkdir(parents=True, exist_ok=True)
    return d


def sha_array(a):
    return rb.sha_array(np.asarray(a))


# ---------------------------------------------------------------- rows and split

def load_rows(dataset, limit=None):
    """The corpus in row space with its split. ``limit`` (smoke only): that many random fit units and limit // 4
    scoring units, everything else as in the full run."""
    c = load_probe_data(dataset)
    fit_u, score_u = c.fit_units, c.scoring_units
    if limit:
        rng = np.random.default_rng(0)
        fit_u = np.sort(rng.choice(fit_u, min(limit, len(fit_u)), replace=False))
        score_u = np.sort(rng.choice(score_u, min(max(limit // 4, 40), len(score_u)), replace=False))
    keep = np.flatnonzero(np.isin(c.row_to_unit, np.r_[fit_u, score_u]))        # sorted caption rows
    unit = c.row_to_unit[keep]
    assert len(c.captions) == len(c.txt_features) == len(c.row_to_unit)
    data = SimpleNamespace(img_features=np.ascontiguousarray(c.img_features[unit]),
                           txt_features=np.ascontiguousarray(c.txt_features[keep]))
    assert data.img_features.dtype == data.txt_features.dtype == np.float32
    species = c.groups[unit]
    leak = species if dataset == "cub" else unit
    fit_rows = np.flatnonzero(np.isin(unit, fit_u))
    scoring_rows = np.flatnonzero(np.isin(unit, score_u))
    assert len(fit_rows) + len(scoring_rows) == len(keep) and not np.intersect1d(fit_rows, scoring_rows).size
    if dataset == "cub":     # species never shared between fit and scoring
        assert not np.intersect1d(species[fit_rows], species[scoring_rows]).size
    scorer_train, selection = grouped_subsplit(leak, fit_rows, SELECTION_FRACTION[dataset], seed=SEED)
    assert not np.intersect1d(leak[scorer_train], leak[selection]).size
    return SimpleNamespace(
        name=dataset, plan=c.plan, data=data, captions=[c.captions[i] for i in keep], groups=unit, leak=leak,
        labels={a: np.asarray(v)[unit] for a, v in c.labels.items()}, fit_rows=fit_rows, scoring_rows=scoring_rows,
        scorer_train=scorer_train, selection=selection, n_rows=len(keep), limit=limit,
        counts={"rows": len(keep), "fit_rows": len(fit_rows), "scorer_train_rows": len(scorer_train),
                "selection_rows": len(selection), "scoring_rows": len(scoring_rows),
                "scorer_train_units": int(len(np.unique(unit[scorer_train]))),
                "selection_units": int(len(np.unique(unit[selection]))),
                "scoring_units": int(len(np.unique(unit[scoring_rows])))})


def check_split(rows, out):
    """The split is recomputed by every stage; the first stage writes it, the later ones compare."""
    path = out / "split.npz"
    arrs = {k: getattr(rows, k) for k in ("scorer_train", "selection", "scoring_rows", "fit_rows")}
    if not path.exists():
        np.savez(path, **arrs)
        return
    with np.load(path) as z:
        assert all(np.array_equal(z[k], v) for k, v in arrs.items()), f"{path} differs from the recomputed split"


def write_json(path, obj):
    assert not Path(path).exists(), f"refusing to overwrite {path}"
    Path(path).write_text(json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")


def fail(out, stage, why):
    """A step that cannot run on this corpus: the reason goes to disk, the run stops with exit code 3."""
    p = out / f"{stage}_failed.json"
    p.write_text(json.dumps({"stage": stage, "reason": why}, indent=2) + "\n")
    log(f"{stage} cannot run on this corpus: {why}")
    return 3


# ---------------------------------------------------------------- stage affect (GPU for GoEmotions)

def stage_affect(rows, out):
    """Code map 1.3 and 1.5 on scorer-train captions (the fit split; selection and scoring captions never go in):
    GoEmotions probabilities; Leiden `detect_communities` + `merge_small` (min 200 rows, run_told_oracle's MIN_ROWS) =
    partition L; MiniBatchKMeans k=64 (run_affect's settings) = affect-km."""
    assert not (out / "affect.npz").exists(), "affect.npz exists"
    st = rows.scorer_train
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t = time.time()
    cache = out / "affect_probs.npz"
    if cache.exists():       # a rerun after a failure downstream of the slow GPU step reuses the probabilities
        probs = np.load(cache)["affect_probs"]
        log("GoEmotions probabilities read from affect_probs.npz")
    else:
        probs = goemotions_probabilities([rows.captions[i] for i in st], loaded=load_goemotions(device=device),
                                         batch_size=256, max_length=64)
        np.savez(cache, affect_probs=probs.astype(np.float32))
    assert probs.shape == (len(st), 28) and np.isfinite(probs).all() and probs.min() >= 0 and probs.max() <= 1
    log(f"GoEmotions {probs.shape} on {device} [{time.time() - t:.0f}s]")
    raw = detect_communities(probs)
    min_rows = R.rto.MIN_ROWS if not rows.limit else 20     # smoke subsets have too few rows for 200
    part_l, moves = R.rto.merge_small(raw, probs, min_rows)
    km = MiniBatchKMeans(**AFFECT_KM).fit_predict(probs).astype(np.int64)
    sizes = np.bincount(km, minlength=64)
    np.savez(out / "affect.npz", partition_L_raw=raw, partition_L=part_l, affect_km=km)
    rec = {"device": device, "n_rows": int(len(st)), "raw_communities": int(raw.max() + 1),
           "partition_L_groups": int(part_l.max() + 1), "merge_min_rows": int(min_rows), "merges": moves,
           "partition_L_sizes": np.bincount(part_l).tolist(), "affect_km_nonempty": int((sizes > 0).sum()),
           "affect_km_sizes_min_median_max": [int(sizes.min()), int(np.median(sizes)), int(sizes.max())],
           "probs_sha256": sha_array(probs), "counts": rows.counts}
    write_json(out / "affect_record.json", rec)
    log(f"partition L: {rec['raw_communities']} raw -> {rec['partition_L_groups']} groups; affect-km sizes "
        f"{rec['affect_km_sizes_min_median_max']}")
    return 0


# ---------------------------------------------------------------- stage groups (CPU)

def load_affect(out, n):
    with np.load(out / "affect.npz") as z:
        a = {k: np.asarray(z[k], dtype=np.int64) for k in ("partition_L", "affect_km")}
    assert all(len(v) == n for v in a.values()), "affect.npz is not one label per scorer-train row"
    return a


MIN_GROUP_UNITS = 30      # = the bank's min_paintings: a group under this size cannot give a bank value anyway


def merge_tiny_groups(parts, rows, out):
    """ADAPTED (not in the recipe): k-means with k=64 on these corpora leaves groups of 1 to 11 rows (SemArt: caption and
    affect-km), which the cross-fitted head draw then misses (the recipe stops: reader_heads). Groups under 30 rows
    are merged into the nearest group by centroid (`merge_small`, the same helper as partition L), in the space the
    grouping was made in (affect-km: GoEmotions probabilities; image, caption: the backbone's features), and codes
    re-compacted. Sizes before and after go to the record."""
    st = rows.scorer_train
    space = {"affect": np.load(out / "affect_probs.npz")["affect_probs"], "image": rows.data.img_features[st],
             "caption": rows.data.txt_features[st]}
    merged, parts = {}, dict(parts)
    for g in G.AIC:
        sizes = np.bincount(parts[g])
        merged[g] = {"sizes_before_min": int(sizes.min()), "n_under_min": int((sizes < MIN_GROUP_UNITS).sum())}
        if merged[g]["n_under_min"]:
            parts[g], moves = R.rto.merge_small(parts[g], space[g], MIN_GROUP_UNITS)
            merged[g]["moves"] = moves
            assert np.bincount(parts[g]).min() >= MIN_GROUP_UNITS
    return parts, merged


def stage_groups(rows, out, min_group_rows=200):
    min_group_rows = min_group_rows if not rows.limit else 5      # smoke subsets
    """partitions.npz (affect = affect-km, image, caption, local_groups, + partition_L) and bank_AIC.npz.
    ADAPTED: ArtELingo copies `affect` and `local_groups` from CLIP's files; here the first is this corpus's affect-km
    and the second is computed from the split (the identity r9_recipe_groups asserts). fit_partitions and fit_bank are the
    recipe's functions unchanged."""
    for f in ("partitions.npz", "bank_AIC.npz"):
        assert not (out / f).exists(), f"{f} exists"
    st = rows.scorer_train
    aff = load_affect(out, len(st))
    local_groups = np.unique(rows.groups[st], return_inverse=True)[1].astype(np.int64)
    t = time.time()
    try:
        parts = G.fit_partitions(rows.data.img_features, rows.data.txt_features, rows.groups, st,
                                 {"affect": aff["affect_km"], "local_groups": local_groups},
                                 min_group_rows=min_group_rows)
    except AssertionError as e:
        return fail(out, "groups", f"fit_partitions: {e} (min_group_rows={min_group_rows})")
    log(f"image and caption k-means [{time.time() - t:.0f}s]")
    parts, merged = merge_tiny_groups(parts, rows, out)
    np.savez(out / "partitions.npz", **parts, partition_L=aff["partition_L"])
    t = time.time()
    try:
        bank = G.fit_bank(parts)
    except (RuntimeError, AssertionError, ValueError) as e:
        return fail(out, "groups_bank", f"fit_bank: {type(e).__name__}: {e}")
    np.savez(out / "bank_AIC.npz", **bank)
    write_json(out / "groups_record.json", {
        "min_group_rows": min_group_rows, "tiny_group_merges": merged,
        "n_groups": {k: int(parts[k].max() + 1) for k in G.AIC}, "n_local_rows": int(len(st)), "n_units": int(local_groups.max() + 1),
        "bank_episodes": int(len(bank["anchor"])), "bank_seconds": round(time.time() - t, 1),
        "label_sha256": {k: sha_array(v) for k, v in parts.items()}, "counts": rows.counts})
    log(f"bank {len(bank['anchor'])} episodes [{time.time() - t:.0f}s]")
    return 0


# ---------------------------------------------------------------- stage graph (GPU)

def stage_graph(rows, out):
    """graph.npz: the recipe's `fit_graph` (build_content_graph, fp16 on CUDA). Refuses without a GPU as the recipe does."""
    assert not (out / "graph.npz").exists(), "graph.npz exists"
    t = time.time()
    graph = G.fit_graph(rows.data.img_features, rows.data.txt_features, rows.scorer_train)
    save_npz(out / "graph.npz", graph)
    write_json(out / "graph_record.json", {"shape": list(graph.shape), "nnz": int(graph.nnz),
                                           "device": G.graph_device_record(), "seconds": round(time.time() - t, 1)})
    log(f"graph {graph.shape} nnz {graph.nnz} [{time.time() - t:.0f}s]")
    return 0


# ---------------------------------------------------------------- stage a3 (GPU)

def stage_a3(rows, out):
    """A3 seed 42, 2000 steps: `run_gonogo.train`'s call (train_factors on scorer-train rows, graph, local_groups, bank AIC),
    with rg.run_config("A3", 42, 2000). ADAPTED as r9_recipe_a3 does: run_gonogo.train is wired to CLIP/ArtELingo files.
    No identity gate (CLIP-only); the config comes from run_config as the recipe's does."""
    from src.train.train_factors import save_factor_checkpoint, train_factors
    ckpt = out / "A3_seed42.pt"
    assert not ckpt.exists(), "A3_seed42.pt exists"
    assert torch.cuda.is_available(), "A3 trains on the GPU (gpu0 lock)"
    rg, st = R.rg, rows.scorer_train
    config = rg.run_config("A3", SEED, 2000)
    rg.check_base_config(SEED, 2000)
    graph = load_npz(out / "graph.npz").tocsr()
    assert graph.shape == (len(st), len(st))
    with np.load(out / "partitions.npz") as z:
        local_groups = z["local_groups"].astype(np.int64)
    with np.load(out / "bank_AIC.npz") as z:
        bank = AspectEpisodes(str(z["aspect_a"]), str(z["aspect_b"]), *(z[f].astype(np.int64) for f in rg.FIELDS))
    assert 0 <= bank.rows().min() and bank.rows().max() < len(st)
    history = {}
    t = time.time()
    model, ic, tc = train_factors(rows.data.img_features[st], rows.data.txt_features[st], graph, config,
                                  device="cuda", group_ids=local_groups, aspect_bank=bank, history=history,
                                  log_every=50)
    assert np.isfinite(ic).all() and np.isfinite(tc).all()
    save_factor_checkpoint(model, config, ckpt)
    write_json(out / "a3_record.json", {"seconds": round(time.time() - t, 1), "steps": 2000,
                                        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
                                        "checkpoint_sha256": R.sha256_file(ckpt), "first_loss": history["loss"][0],
                                        "last_loss": history["loss"][-1],
                                        "code_stats": rg.code_stats(ic, tc)})
    log(f"A3 trained [{time.time() - t:.0f}s]")
    return 0


# ---------------------------------------------------------------- stage reader (CPU)

def make_halves(rows):
    """rb_build.stage_halves for this corpus: unit halves of scorer-train (permutation with HALF_SEED, first half of the
    units = half 0). ADAPTED: ArtELingo's halves are CLIP's frozen file for 36,518 paintings."""
    st = rows.scorer_train
    unit = rows.groups[st]
    units, g_st = np.unique(unit, return_inverse=True)
    order = np.random.default_rng(rb.HALF_SEED).permutation(len(units))
    half_of_code = np.zeros(len(units), dtype=np.int8)
    half_of_code[order[len(units) // 2:]] = 1
    half_of_row = half_of_code[g_st]
    local = {k: np.flatnonzero(half_of_row == k).astype(np.int64) for k in (0, 1)}
    assert len(local[0]) + len(local[1]) == len(st)
    return {"scorer_train": st, "painting_of_local_row": unit, "g_st": g_st, "half_of_local_row": half_of_row,
            "local_rows_half0": local[0], "local_rows_half1": local[1],
            "global_rows_half0": st[local[0]], "global_rows_half1": st[local[1]]}


def reader_params(halves):
    """ADAPTED sizes of RR.PARAMS (ArtELingo's 183,694 rows, 36,518 paintings): head draw 80% of a half's rows (at most
    60,000), check rows at most 10,000, copy-check dropped; bank 16,384 per block as the rule; the rest as RR.PARAMS."""
    half_rows = min(len(halves["local_rows_half0"]), len(halves["local_rows_half1"]))
    head_rows = min(60_000, int(0.8 * half_rows))
    p = dict(RR.PARAMS)
    p.update(n_st=len(halves["scorer_train"]), n_paintings=int(halves["g_st"].max() + 1),
             half_paintings=int(halves["g_st"].max() + 1) // 2, head_rows=head_rows,
             check_rows=min(10_000, half_rows - head_rows), bank_size=3 * p["bank_n"])
    return p


def stage_reader(rows, out):
    """Heads (cross-fitted over the halves) for the A0 groupings, the two practice banks, the two half-readers.
    Calls RR.fit_crossfit_heads, RR.fit_bank_half and RR.fit_readers unchanged with the corpus's halves and sizes.
    Writes rb_heads_<g>.npz, rb_bank_A0_half<j>.npz, rb_reader_A0.pkl, reader_record.json."""
    names = [f"rb_heads_{g}.npz" for g in A0] + [f"rb_bank_A0_half{j}.npz" for j in (0, 1)] + ["rb_reader_A0.pkl"]
    assert not any((out / n).exists() for n in names), "reader outputs exist"
    st = rows.scorer_train
    halves = make_halves(rows)
    params = reader_params(halves)
    aff = load_affect(out, len(st))
    with np.load(out / "partitions.npz") as z:
        local = {"affect": aff["partition_L"], "image": z["image"].astype(np.int64),
                 "caption": z["caption"].astype(np.int64)}
    n_groups = {g: int(local[g].max() + 1) for g in A0}
    assert all(np.bincount(local[g]).min() > 0 for g in A0), "a grouping has an empty code"
    F = {"img": rows.data.img_features, "txt": rows.data.txt_features}
    post, rec = {}, {"params": params, "n_groups": n_groups, "counts": rows.counts, "heads": {}, "banks": {}}
    for g in A0:
        t = time.time()
        lab = RR.rto_labels(local[g], st, rows.n_rows)
        try:
            arrays, prov = RR.fit_crossfit_heads(F, lab, halves, g, n_groups[g], n_rows=params["head_rows"],
                                                 check_rows=params["check_rows"], backbone="probe")
        except SystemExit as e:
            return fail(out, "reader_heads", f"{g}: {e}")
        np.savez(out / f"rb_heads_{g}.npz", **arrays)
        post[g] = arrays
        rec["heads"][g] = prov
        log(f"{g} cross-fitted heads [{time.time() - t:.0f}s]")
    banks = {}
    for j in (0, 1):
        t = time.time()
        try:
            bank = RR.fit_bank_half(local, halves, j, params=params)
        except SystemExit as e:
            return fail(out, "reader_bank", f"half {j}: {e}")
        np.savez(out / f"rb_bank_A0_half{j}.npz", **bank.arrays)
        banks[j] = bank.arrays
        rec["banks"][str(j)] = bank.prov
        log(f"bank half {j} [{time.time() - t:.0f}s]")
    pk, rrec = RR.fit_readers(post, banks, halves, params=params)
    with open(out / "rb_reader_A0.pkl", "wb") as f:
        pickle.dump(pk, f)
    rec["reader"] = rrec
    rec["reader_pkl_sha256"] = R.sha256_file(out / "rb_reader_A0.pkl")
    write_json(out / "reader_record.json", rec)
    return 0
