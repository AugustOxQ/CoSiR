"""CoSiR v2 Candidate A affect-signal factor learning, cells E and SE (spec docs/superpowers/specs/2026-09-30-cosir-v2-candidate-a-affect-factor-learning-design.md).

Factor encoders trained like the 2x2's S cell (R3 config, pair agreement, condition episodes) but with the condition
source built from GoEmotions affect clusters of the captions: E (affect partition alone) and SE (affect partition and
the CLIP image partition, each drawn with probability 1/2). C0 (the 2x2's matched control) and S (reference row) are
existing checkpoints of the previous grid. Run from the repository root:

    python src/test/20261018_affect_factor_learning/run_affect.py --prepare      # affect extraction, k-means, diagnostics
    python src/test/20261018_affect_factor_learning/run_affect.py --smoke        # timing: local GPU or DAS6?
    python src/test/20261018_affect_factor_learning/run_affect.py --run E --seed 42    # refuses an existing
                                                                                       # checkpoint (--overwrite)
    python src/test/20261018_affect_factor_learning/run_affect.py --evaluate     # spec §6: gates, D_emo / D_style,
                                                                                 # the rule, reported extras
    python src/test/20261018_affect_factor_learning/run_affect.py --tables       # reprint from the JSON
    python src/test/20261018_affect_factor_learning/run_affect.py --replicate    # Task 4 (not yet implemented)

Row scope (hard rule): only scorer-train captions are ever passed to the affect model. Selection, val and held
captions never are. The diagnostic probe uses a painting-grouped 80/20 split INSIDE scorer-train.

--evaluate (spec §6): original R3 (stage (d)'s cached codes), C0 and S (the 2x2's seed-42 checkpoints) and this
folder's E and SE on 4,096 selection label episodes per label (the first 2,048 are stage (d)'s: prefix SHA-256
asserted). Factors encode scorer-train and selection rows only; evaluation codes and CLIP features are NaN outside
selection rows (asserted). Per model: the nine gates (eight binding, sparsity reported), the naive rule on the beta
grid, the label oracle at beta 0.3 and 0 and its null at 0. Then D_emo / D_style vs C0 and the pre-registered rule
(``apply_affect_rule``), plus reported extras that gate nothing: guard power, the balance-matched-beta comparison,
per-modality code probes and the within-painting caption-residual emotion probe (fit on scorer-train rows), code
AMI, the per-target breakdown, ties, and the condition-loss / tau histories. Writes results/selection_results.json
and results/selection_ranks.npz.
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
from joblib import Parallel, delayed
from scipy.stats import norm
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
from src.eval.condition_eval import paired_bootstrap  # noqa: E402
from src.eval.label_episodes import (LabelEpisodes, label_episode_weights,  # noqa: E402
                                     label_episodes_sha256, standard_label_episodes)
from src.model.conditioning import conditional_score  # noqa: E402
from src.train.condition_sources import CommunitySource, MultiPartitionSource  # noqa: E402
from src.train.train_factors import (encode_rows, load_factor_checkpoint, save_factor_checkpoint,  # noqa: E402
                                     train_factors)

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


# ----------------------------------------------------------------------------- Task 3: selection rule (spec §6)

BINDING_GATES = ("participation_ratio", "redundancy", "readout", "dead", "modality_private",
                 "usage_concentration", "community_spanning", "pair_retrieval")      # sparsity: reported only
STYLE_MARGIN, TIE_POINTS = -1.5, 0.5
CHANGES = {"E": 1, "SE": 2}


def apply_affect_rule(gates_passed: dict, d_emo: dict, d_style: dict) -> dict:
    """Affect spec §6. Eligible = the 8 binding gates pass (sparsity is not counted); qualifies = eligible and
    D_emo lower bound > 0 and D_style lower bound > -1.5; pick = highest D_emo, cells within 0.5 tie, a tie goes to
    E (fewer changes), then to the higher D_emo. ``gates_passed`` maps model -> {gate: bool}; ``d_emo`` / ``d_style``
    map cell -> {"point": R@1 points, "ci95": [lo, hi]}."""
    binding_ok = {m: all(bool(g[name]) for name in BINDING_GATES) for m, g in gates_passed.items()}
    if not binding_ok["C0"]:
        return {"stop": "C0 fails a binding gate: the setup is broken", "picked": None, "qualifying": [],
                "tie_band": [], "binding_ok": binding_ok}
    qualifying = [c for c in CELLS if binding_ok[c] and d_emo[c]["ci95"][0] > 0
                  and d_style[c]["ci95"][0] > STYLE_MARGIN]
    if not qualifying:
        return {"stop": "no cell qualifies", "picked": None, "qualifying": [], "tie_band": [],
                "binding_ok": binding_ok}
    best = max(d_emo[c]["point"] for c in qualifying)
    tied = [c for c in qualifying if d_emo[c]["point"] >= best - TIE_POINTS]
    picked = min(tied, key=lambda c: (CHANGES[c], -d_emo[c]["point"]))
    return {"stop": None, "picked": picked, "qualifying": qualifying, "tie_band": tied, "binding_ok": binding_ok}


def _check_affect_rule() -> None:
    names = BINDING_GATES + ("sparsity",)
    ok = {m: {n: True for n in names} for m in ("C0", "E", "SE")}
    blk = lambda p, lo: {"point": p, "ci95": [lo, p + 1.0]}                        # noqa: E731
    style_fine = {c: blk(0.0, -0.8) for c in CELLS}
    bad_c0 = {**ok, "C0": {**ok["C0"], "readout": False}}
    assert apply_affect_rule(bad_c0, {c: blk(2, 1) for c in CELLS}, style_fine)["stop"].startswith("C0")
    sparse_only = {**ok, "E": {**ok["E"], "sparsity": False}}                     # sparsity is not binding
    assert apply_affect_rule(sparse_only, {"E": blk(1.0, 0.2), "SE": blk(0.1, -0.5)}, style_fine)["picked"] == "E"
    assert apply_affect_rule(ok, {c: blk(0.5, -0.1) for c in CELLS}, style_fine)["stop"] == "no cell qualifies"
    assert apply_affect_rule(ok, {"E": blk(1.0, 0.2), "SE": blk(1.4, 0.5)}, style_fine)["picked"] == "E"    # tie band
    assert apply_affect_rule(ok, {"E": blk(1.0, 0.2), "SE": blk(1.8, 0.9)}, style_fine)["picked"] == "SE"
    assert apply_affect_rule(ok, {"E": blk(1.0, 0.2), "SE": blk(1.8, 0.9)},
                             {**style_fine, "SE": blk(-1.0, -2.0)})["picked"] == "E"                     # style guard


# ----------------------------------------------------------------------------- Task 3: selection evaluation

sel, probe = grid.sel, grid.probe
MODELS = ("R3", "C0", "S", "E", "SE")         # R3 = stage (d)'s cached codes; S = the 2x2's style cell (reference)
VS_C0 = ("S", "E", "SE")                      # compared with C0; only E and SE are candidates (CELLS)
LABELS, SCOPES, DIRECTIONS = grid.LABELS, grid.SCOPES, grid.DIRECTIONS
BETAS, BETA_FIXED, key = grid.BETAS, grid.BETA_FIXED, grid.key
ORACLE_STEPS, ORACLE_BETAS, NULL_BETA = grid.ORACLE_STEPS, grid.ORACLE_BETAS, grid.NULL_BETA
GATE_NAMES = grid.GATE_NAMES
N_EPISODES, N_PREFIX = 4096, 2048             # the first 2,048 per label are stage (d)'s selection episodes
HALF_WIDTH_Z = 1.959964
SPEC_STYLE_SE = 0.42                          # spec §6's pre-run style SE (from the 2x2's 0.57 at 2,048)
REFERENCE_SHA_KEYS = {"C0": "c0_ref_sha256", "S": "s_ref_sha256"}
SELECTION_JSON, SELECTION_NPZ = RESULTS / "selection_results.json", RESULTS / "selection_ranks.npz"
_POSTHOC_PATH = _GRID_PATH.parent / "run_posthoc.py"


def _load_posthoc():
    """run_posthoc.py (the 2x2's post-hoc helpers), imported on demand; its 2x2-specific functions are not used."""
    spec = importlib.util.spec_from_file_location("run_posthoc", _POSTHOC_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def selection_episodes(data, cache) -> tuple[dict, dict, dict]:
    """4,096 selection label episodes per label (prefix SHA-256 = stage (d)'s, rows asserted) and null targets."""
    in_sel = grid._row_mask(len(cache["groups"]), cache["selection"])
    episodes, meta = {}, {}
    for label in LABELS:
        eps = standard_label_episodes(data, cache["groups"], cache["selection"], label, N_EPISODES, seed=SEED)
        prefix = LabelEpisodes(**{f.name: getattr(eps, f.name)[:N_PREFIX] for f in dataclasses.fields(eps)})
        prefix_sha = label_episodes_sha256(prefix)
        if prefix_sha != grid.fin.SELECTION_SHA256[label]:
            raise AssertionError(f"{label}: the first {N_PREFIX} episodes differ from stage (d)'s ({prefix_sha})")
        if not in_sel[np.column_stack([eps.anchor, eps.positive, eps.supports, eps.contrasts, eps.distractors])].all():
            raise AssertionError(f"{label} episodes use rows outside the selection set")
        if eps.distractors.shape[1] + 1 != 13:
            raise AssertionError("expected 13 candidates per episode")
        episodes[label] = eps
        meta[label] = {"n": int(len(eps.anchor)), "targets": int(len(np.unique(eps.labels))),
                       "sha256": label_episodes_sha256(eps), "prefix_2048_sha256": prefix_sha,
                       "prefix_equals_stage_d_sha256": True, "all_rows_in_selection": True}
    if LABELS != ("emotion", "art_style"):
        raise AssertionError("null targets are drawn in (emotion, art_style) order")
    rng = np.random.default_rng(SEED)
    nulls = {label: rng.integers(1, 13, len(episodes[label].anchor)) for label in LABELS}
    log(f"Episodes (prefix SHA-256 asserted, rows in selection asserted): {meta}")
    return episodes, nulls, meta


def model_path(name: str, seed: int = SEED) -> Path:
    if seed != SEED:                                   # replication seeds are trained in this folder
        return CKPT / f"{name}_seed{seed}.pt"
    return {"C0": C0_REF, "S": S_REF}.get(name, CKPT / f"{name}_seed{SEED}.pt")


def model_codes(name: str, data, cache, prep_record: dict, seed: int = SEED) -> tuple[np.ndarray, np.ndarray, dict]:
    """Full-length (n_rows, 32) codes: finite on scorer-train and selection rows, NaN elsewhere."""
    rows = np.concatenate([cache["scorer_train"], cache["selection"]])
    if name == "R3":
        return (sel.masked(cache["img_codes"], rows), sel.masked(cache["txt_codes"], rows),
                {"source": "stage (d) cache img_codes / txt_codes (original R3, trained on all train rows)"})
    path = model_path(name, seed)
    sha = grid.sha256_file(path)
    if seed == SEED and name in REFERENCE_SHA_KEYS and sha != prep_record[REFERENCE_SHA_KEYS[name]]:
        raise AssertionError(f"{path.name}: SHA-256 differs from the one recorded at prepare")
    model, config = load_factor_checkpoint(path, device=grid.DEVICE)
    expected = grid.cell_config("C0" if name == "C0" else "S", seed)
    if dataclasses.asdict(config) != dataclasses.asdict(expected):
        raise AssertionError(f"{path.name}: stored config differs from grid.cell_config({'C0' if name == 'C0' else 'S'!r}, {seed})")
    ic, tc = encode_rows(model, data.img_features, data.txt_features, rows=rows)
    out = []
    for codes in (ic, tc):
        full = np.full((len(cache["groups"]), codes.shape[1]), np.nan, dtype=np.float32)
        full[rows] = codes
        out.append(full)
    info = {"checkpoint": str(path.relative_to(ROOT)), "sha256": sha,
            "config": f"grid.cell_config('{'C0' if name == 'C0' else 'S'}', {seed})"}
    return out[0], out[1], info


def training_record(name: str, seed: int = SEED) -> dict:
    folder = RESULTS if name in CELLS or seed != SEED else grid.RESULTS
    record = json.loads((folder / f"history_{name}_seed{seed}.json").read_text())
    if record["steps"] != grid.FULL_STEPS:
        raise AssertionError(f"{name}: history is not a {grid.FULL_STEPS}-step run")
    h = record["history"]
    out = {"seconds": record["seconds"], "peak_gpu_gib": record["peak_gpu_gib"], "history": h,
           "source_views": record.get("source_views"), "final_loss": h["loss"][-1],
           "final_agreement": h["agreement"][-1],
           "final_condition_loss": h["condition_loss"][-1] if "condition_loss" in h else None,
           "final_tau": h["tau"][-1] if "tau" in h else None}
    if "condition_loss" in h:
        cl = np.asarray(h["condition_loss"])
        out["condition_summary"] = {"steps_logged": int(len(cl)), "step_1": float(cl[0]),
                                    "mean_steps_50_500": float(cl[1:11].mean()),
                                    "mean_last_10_logged": float(cl[-10:].mean()),
                                    "last_10_steps": h["step"][-10:], "final_single_batch": float(cl[-1]),
                                    "tau_step_1": float(h["tau"][0]), "tau_final": float(h["tau"][-1]),
                                    "all_finite": bool(np.isfinite(cl).all() and np.isfinite(h["tau"]).all())}
    return out


def tie_breakdown(img, txt, ic, tc, episodes: dict, weights: dict, beta: float, stored: dict) -> dict:
    """Naive R@1 as scored (a tie is a miss) and with random tie-breaking; episodes with the positive tied at top."""
    random_break, tied_top = {}, {}
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32)                    # noqa: E731
    for label in LABELS:
        eps = episodes[label]
        cands = np.concatenate([eps.positive[:, None], eps.distractors], axis=1)
        random_break[label] = {}
        for d, qf, cf, qc, cc in (("i2t", img, txt, ic, tc), ("t2i", txt, img, tc, ic)):
            s = conditional_score(t(qf[eps.anchor]), t(cf[cands]), t(qc[eps.anchor]), t(cc[cands]), weights[label],
                                  beta)
            pos, others = s[:, :1], s[:, 1:]
            greater, equal = (others > pos).sum(dim=1).numpy(), (others == pos).sum(dim=1).numpy()
            if not np.array_equal(1.0 + greater + 0.5 * equal, np.asarray(stored[label][d])):
                raise AssertionError(f"{label} {d} beta {beta}: recomputed ranks differ from the scored ranks")
            random_break[label][d] = np.where(greater == 0, 1.0 / (1.0 + equal), 0.0)
            tied_top[f"{label}|{d}"] = int(((greater == 0) & (equal > 0)).sum())
    expected = {}
    for scope in SCOPES:
        per = random_break if scope == "pooled" else {scope: random_break[scope]}
        vals = {d: 100 * float(np.concatenate([per[lab][d] for lab in per]).mean()) for d in DIRECTIONS}
        expected[scope] = {**vals, "mean": 0.5 * (vals["i2t"] + vals["t2i"])}
    return {"tie_aware": probe.r1_points(stored), "random_tie_break": expected, "positive_tied_at_top": tied_top}


def guard_power(d_style: dict, d_emo: dict) -> dict:
    """Normal approximation from the measured D_style CIs: SE = mean over E, SE of (point - lower bound) / 1.96."""
    cells = {c: {"point": d_style[c]["point"], "ci95": d_style[c]["ci95"],
                 "half_width": 0.5 * (d_style[c]["ci95"][1] - d_style[c]["ci95"][0]),
                 "lower_distance": d_style[c]["point"] - d_style[c]["ci95"][0]} for c in CELLS}
    lower = float(np.mean([v["lower_distance"] for v in cells.values()]))
    se = lower / HALF_WIDTH_Z

    def pass_probability(se_value: float) -> dict:
        threshold = STYLE_MARGIN + HALF_WIDTH_Z * se_value          # point estimate needed for lower bound > -1.5
        return {"pass_threshold_point": threshold,
                "by_true_effect": {f"{mu:+.1f}": float(1 - norm.cdf((threshold - mu) / se_value))
                                   for mu in (0.5, 0.0, -0.5, -1.0, -1.5)}}

    emo_lower = [d_emo[c]["point"] - d_emo[c]["ci95"][0] for c in CELLS]
    return {"cells": cells, "mean_lower_distance": lower, "se_measured": se, "measured": pass_probability(se),
            "spec_planned": {"se": SPEC_STYLE_SE, **pass_probability(SPEC_STYLE_SE)},
            "d_emo_se_measured": float(np.mean(emo_lower) / HALF_WIDTH_Z)}


def balance_matched(spread: dict, r1: dict, weights: dict, img, txt, codes_sel: dict, episodes: dict,
                    naive: dict) -> dict:
    """E / SE vs C0 with the balance of the two score terms matched (the 2x2 post-hoc's method, in memory)."""
    out = {}
    for x in CELLS:
        fx, fc = spread[x]["factor_spread"], spread["C0"]["factor_spread"]
        beta_c0, beta_x = BETA_FIXED * fc / fx, BETA_FIXED * fx / fc
        c0_m, _ = probe.fixed_weight_ranks(img, txt, *codes_sel["C0"], episodes, weights["C0"], beta_c0)
        x_m, _ = probe.fixed_weight_ranks(img, txt, *codes_sel[x], episodes, weights[x], beta_x)

        def interp(model: str, beta: float, scope: str) -> float:
            return float(np.interp(beta, BETAS, [r1[key("naive", model, b)][scope]["mean"] for b in BETAS]))

        interpolated = {scope: {f"{x}@0.3 - C0@0.3": interp(x, 0.3, scope) - interp("C0", 0.3, scope),
                                f"{x}@0.3 - C0@matched": interp(x, 0.3, scope) - interp("C0", beta_c0, scope),
                                f"{x}@matched - C0@0.3": interp(x, beta_x, scope) - interp("C0", 0.3, scope)}
                        for scope in SCOPES}
        rescored = {f"{x}@0.3 - C0@matched": probe.r1_diff(naive[x], c0_m),
                    f"{x}@matched - C0@0.3": probe.r1_diff(x_m, naive["C0"])}
        out[x] = {"factor_to_clip": {x: spread[x]["factor_to_clip"], "C0": spread["C0"]["factor_to_clip"]},
                  "clip_reliance_ratio_C0_over_x": spread["C0"]["factor_to_clip"] / spread[x]["factor_to_clip"],
                  "beta_C0_matched": beta_c0, "beta_x_matched": beta_x,
                  "matched_beta_outside_grid": bool(not (0 <= beta_c0 <= 1 and 0 <= beta_x <= 1)),
                  "interpolated": interpolated, "rescored": rescored,
                  "C0@matched": probe.r1_points(c0_m), "x@matched": probe.r1_points(x_m)}
    return out


def per_target(ranks: dict, episodes: dict) -> dict:
    """Per target label: X - C0 (X in E, SE) in R@1, naive at beta 0.3 and 0, oracle at 0 (paired bootstrap)."""
    out = {}
    for x in CELLS:
        out[x] = {}
        for label in LABELS:
            targets = episodes[label].labels
            out[x][label] = {}
            for col, scorer, beta in (("naive@0.3", "naive", 0.3), ("naive@0", "naive", 0.0),
                                      ("oracle@0", "oracle", 0.0)):
                hx = {d: probe.hits(ranks[key(scorer, x, beta)][label][d]) for d in DIRECTIONS}
                hc = {d: probe.hits(ranks[key(scorer, "C0", beta)][label][d]) for d in DIRECTIONS}
                rows = {}
                for target in np.unique(targets):
                    mask = targets == target
                    diff = {d: hx[d][mask] - hc[d][mask] for d in DIRECTIONS}
                    rows[str(target)] = {
                        "n": int(mask.sum()),
                        "C0": 100 * float(0.5 * (hc["i2t"][mask] + hc["t2i"][mask]).mean()),
                        x: 100 * float(0.5 * (hx["i2t"][mask] + hx["t2i"][mask]).mean()),
                        "diff": probe.pct_block(paired_bootstrap(0.5 * (diff["i2t"] + diff["t2i"]))),
                        **{f"diff_{d}": 100 * float(diff[d].mean()) for d in DIRECTIONS}}
                total = sum(v["n"] * v["diff"]["point"] for v in rows.values())
                for v in rows.values():
                    v["share_of_total"] = (v["n"] * v["diff"]["point"] / total) if total else float("nan")
                out[x][label][col] = rows
    return out


def probe_analyses(ph, codes: dict, txt_clip: np.ndarray, cache: dict, labels: dict) -> tuple[dict, dict]:
    """run_posthoc.probe_analyses over this folder's models (same probes, helpers and bootstrap)."""
    st, sl, groups = cache["scorer_train"], cache["selection"], cache["groups"]
    y = {lab: (ph.label_rows(labels, lab, st), ph.label_rows(labels, lab, sl)) for lab in LABELS}
    jobs, meta = [], []
    for m in MODELS:
        side = {"img": codes[m][0], "txt": codes[m][1]}
        for modality, lab in ph.PROBE_TASKS:
            jobs.append((side[modality][st].astype(np.float64), y[lab][0], side[modality][sl].astype(np.float64)))
            meta.append(("code", m, modality, lab))
    residual_keep = {}
    for m in (*MODELS, "clip512"):
        x = txt_clip if m == "clip512" else codes[m][1]
        r_fit, k_fit = ph.within_painting_residual(x, st, groups)
        r_sel, k_sel = ph.within_painting_residual(x, sl, groups)
        xf = x[st].astype(np.float64)
        share = float((r_fit[k_fit] ** 2).sum() / ((xf[k_fit] - xf[k_fit].mean(axis=0)) ** 2).sum())
        residual_keep[m] = (k_fit, k_sel, share)
        jobs.append((r_fit[k_fit], y["emotion"][0][k_fit], r_sel[k_sel]))
        meta.append(("residual", m, "txt", "emotion"))
    log(f"Probes: {len(jobs)} logistic fits on {ph.PROBE_JOBS} workers")
    t0 = perf_counter()
    fitted = Parallel(n_jobs=ph.PROBE_JOBS, backend="loky")(delayed(ph._fit_probe)(*j) for j in jobs)
    log(f"Probes done in {perf_counter() - t0:.1f} s")
    code_probes, residual_probes, correct = {}, {}, {}
    sel_groups = groups[sl]
    for (kind, m, modality, lab), out in zip(meta, fitted):
        if kind == "code":
            truth = y[lab][1]
            ok = out["pred"] == truth
            top, base = ph.majority(y[lab][0], truth)
            correct[(kind, m, modality, lab)] = (ok, sel_groups)
            code_probes.setdefault(m, {})[f"{modality}->{lab}"] = {
                "accuracy": 100 * float(ok.mean()), "majority": 100 * base, "majority_class": top,
                "n_fit": int(len(st)), "n_eval": int(len(sl)), "n_iter": out["n_iter"],
                "converged": out["converged"], "seconds": out["seconds"]}
        else:
            k_fit, k_sel, share = residual_keep[m]
            truth = y["emotion"][1][k_sel]
            ok = out["pred"] == truth
            top, base = ph.majority(y["emotion"][0][k_fit], truth)
            correct[(kind, m, modality, lab)] = (ok, sel_groups[k_sel])
            residual_probes[m] = {"accuracy": 100 * float(ok.mean()), "majority": 100 * base, "majority_class": top,
                                  "n_fit": int(k_fit.sum()), "n_eval": int(k_sel.sum()),
                                  "within_painting_variance_share": share, "n_iter": out["n_iter"],
                                  "converged": out["converged"], "seconds": out["seconds"]}
    for m in MODELS:
        if m == "C0":
            continue
        for modality, lab in ph.PROBE_TASKS:
            a, ga = correct[("code", m, modality, lab)]
            b, _ = correct[("code", "C0", modality, lab)]
            code_probes[m][f"{modality}->{lab}"]["minus_C0"] = ph.cluster_bootstrap_diff(a, b, ga)
        a, ga = correct[("residual", m, "txt", "emotion")]
        b, _ = correct[("residual", "C0", "txt", "emotion")]
        residual_probes[m]["minus_C0"] = ph.cluster_bootstrap_diff(a, b, ga)
    return code_probes, residual_probes


def ami_analysis(ph, codes: dict, cache: dict, labels: dict, affect_local: np.ndarray) -> dict:
    """AMI of each model's argmax pair-code factor with the CLIP image clusters and the affect clusters
    (scorer-train rows: both partitions label scorer-train rows only) and with emotion / art style."""
    st, sl = cache["scorer_train"], cache["selection"]
    clip_image = cache["clip_image"][st]
    if (clip_image < 0).any() or len(affect_local) != len(st):
        raise AssertionError("the CLIP image and affect partitions must label every scorer-train row")
    out = {}
    for m in MODELS:
        ic, tc = codes[m]
        pair = 0.5 * (ic.astype(np.float64) + tc.astype(np.float64))
        arg = np.where(pair.max(axis=1) > 0, pair.argmax(axis=1), -1)          # all-zero pair code: own group
        res = {"clip_image@scorer_train": _ami(clip_image, arg[st]),
               "affect_clusters@scorer_train": _ami(affect_local, arg[st])}
        for part, rows in (("selection", sl), ("scorer_train", st)):
            for lab in LABELS:
                res[f"{lab}@{part}"] = _ami(ph.label_rows(labels, lab, rows), arg[rows])
        res["factors_used@selection"] = int(len(np.unique(arg[sl][arg[sl] >= 0])))
        res["all_zero_share@selection"] = float(np.mean(arg[sl] < 0))
        out[m] = res
        log(f"AMI {m}: " + ", ".join(f"{k} {v:.4f}" for k, v in res.items() if isinstance(v, float)))
    out["reference"] = {
        "clip_image": {lab: _ami(ph.label_rows(labels, lab, st), clip_image) for lab in LABELS},
        "affect_clusters": {lab: _ami(ph.label_rows(labels, lab, st), affect_local) for lab in LABELS},
        "affect_vs_clip_image": _ami(affect_local, clip_image)}
    return out


def reproduce_grid(ranks: dict) -> dict:
    """The first 2,048 episodes of R3, C0 and S must rank as in the 2x2's stored selection ranks."""
    stored = np.load(grid.SELECTION_NPZ)
    keys = [key("naive", m, b) for m in ("R3", "C0", "S") for b in BETAS] + [key("clip_only", "-", BETA_FIXED)]
    out = {}
    for k in keys:
        prefix = k.replace("|", "__")
        share = min(float(np.mean(np.asarray(ranks[k][label][d])[:N_PREFIX] == stored[f"{prefix}__{label}__{d}"]))
                    for label in LABELS for d in DIRECTIONS)
        out[k] = share
        if share < grid.IDENTICAL_SHARE_MIN:
            raise AssertionError(f"{k}: first {N_PREFIX} episodes rank differently from the 2x2 ({share:.4f})")
    log(f"First {N_PREFIX} episodes vs the 2x2's stored ranks (identical share): {out}")
    return out


def evaluate() -> dict:
    _check_affect_rule()
    started, timings = perf_counter(), {}
    t0 = perf_counter()
    data = load_artelingo()
    cache, prep, meta, _ = grid.load_grid()
    prep_record = json.loads((CACHE / "affect_prepare.json").read_text())
    affect_local = np.load(CACHE / "affect_prepare.npz")["affect_local"]
    st, sl = cache["scorer_train"], cache["selection"]
    n_rows = len(cache["groups"])
    if np.intersect1d(cache["groups"][st], cache["groups"][sl]).size:
        raise AssertionError("a painting spans scorer-train and selection")
    community_local = cache["community"][st]
    if not np.array_equal(community_local, prep["community_local"]):
        raise AssertionError("community labels differ from the grid's prepared ones")
    episodes, nulls, eps_meta = selection_episodes(data, cache)
    img, txt = (sel.masked(x, sl) for x in (data.img_features, data.txt_features))
    allowed, in_sel = grid._row_mask(n_rows, np.concatenate([st, sl])), grid._row_mask(n_rows, sl)
    for name, arr in (("img eval features", img), ("txt eval features", txt)):
        probe.assert_row_scope(name, arr, in_sel)
    timings["load_and_episodes"] = perf_counter() - t0
    reference = meta["r0_readout_reference"]
    ranks, ties, gates, weights_info, spread, scale, model_info = {}, {}, {}, {}, {}, {}, {}
    codes_full, codes_sel, weights, tie_info = {}, {}, {}, {}
    for name in MODELS:
        t0 = perf_counter()
        ic_full, tc_full, model_info[name] = model_codes(name, data, cache, prep_record)
        for side, arr in (("img", ic_full), ("txt", tc_full)):
            probe.assert_row_scope(f"{name} {side} codes", arr, allowed)
        g = grid.gate_report(ic_full[st], tc_full[st], ic_full[sl], tc_full[sl], data, cache, community_local,
                             reference=reference)
        passed = {n: bool(v) for n, v in g.passed.items()}
        gates[name] = {"values": grid._jsonable(g.values), "passed": passed, "all_passed": bool(g.all_passed),
                       "n_passed": int(sum(passed.values())),
                       "binding_passed": bool(all(passed[n] for n in BINDING_GATES)),
                       "n_binding_passed": int(sum(passed[n] for n in BINDING_GATES))}
        scale[name] = {"mean_rms_scorer_train": probe.mean_rms(ic_full, tc_full, st),
                       "mean_rms_selection": probe.mean_rms(ic_full, tc_full, sl)}
        ic, tc = sel.masked(ic_full, sl), sel.masked(tc_full, sl)
        for side, arr in (("img", ic), ("txt", tc)):
            probe.assert_row_scope(f"{name} {side} eval codes", arr, in_sel)
        codes_full[name], codes_sel[name] = (ic_full, tc_full), (ic, tc)
        weights[name] = {label: label_episode_weights(ic, tc, episodes[label]) for label in LABELS}
        weights_info[name] = grid.weight_summary(weights[name])
        spread[name] = grid.term_spread(img, txt, ic, tc, episodes, weights[name])
        for beta in BETAS:
            ranks[key("naive", name, beta)], ties[key("naive", name, beta)] = probe.fixed_weight_ranks(
                img, txt, ic, tc, episodes, weights[name], beta)
        tie_info[name] = {f"{b:g}": tie_breakdown(img, txt, ic, tc, episodes, weights[name], b,
                                                   ranks[key("naive", name, b)]) for b in (0.0, BETA_FIXED)}
        for beta in ORACLE_BETAS:
            ranks[key("oracle", name, beta)] = probe.oracle_ranks(img, txt, ic, tc, episodes, beta, ORACLE_STEPS,
                                                                  grid.DEVICE)
        ranks[key("oracle_null", name, NULL_BETA)] = probe.oracle_ranks(img, txt, ic, tc, episodes, NULL_BETA,
                                                                        ORACLE_STEPS, grid.DEVICE, nulls)
        if name == "R3":                                # CLIP-only: zero weights, beta 0.3 (codes irrelevant)
            zero = {label: torch.zeros(len(episodes[label].anchor), ic.shape[1]) for label in LABELS}
            ranks[key("clip_only", "-", BETA_FIXED)], _ = probe.fixed_weight_ranks(img, txt, ic, tc, episodes, zero,
                                                                                   BETA_FIXED)
        timings[f"model:{name}"] = perf_counter() - t0
        r = {s: probe.r1_points(ranks[key(s, name, b)])["pooled"]["mean"]
             for s, b in (("naive", BETA_FIXED), ("oracle", 0.0))}
        log(f"{name}: gates {gates[name]['n_passed']}/9, binding {gates[name]['n_binding_passed']}/8; pooled R@1 "
            f"naive@0.3 {r['naive']:.2f}, oracle@0 {r['oracle']:.2f} ({timings[f'model:{name}']:.1f} s)")
    reproduction = reproduce_grid(ranks)

    t0 = perf_counter()
    naive = {m: ranks[key("naive", m, BETA_FIXED)] for m in MODELS}
    vs_c0 = {c: probe.r1_diff(naive[c], naive["C0"]) for c in VS_C0}
    d_emo = {c: vs_c0[c]["emotion"]["mean"] for c in VS_C0}
    d_style = {c: vs_c0[c]["art_style"]["mean"] for c in VS_C0}
    d_pooled = {c: vs_c0[c]["pooled"]["mean"] for c in VS_C0}
    vs_r3 = {m: probe.r1_diff(naive[m], naive["R3"]) for m in MODELS if m != "R3"}
    vs_c0_beta = {c: {f"{b:g}": probe.r1_diff(ranks[key("naive", c, b)], ranks[key("naive", "C0", b)])
                      for b in BETAS} for c in VS_C0}
    oracle_vs_naive = {variant: {m: probe.r1_diff(ranks[key("oracle", m, ob)], ranks[key("naive", m, nb)])
                                 for m in MODELS}
                       for variant, ob, nb in (("same_beta_0.3", 0.3, 0.3), ("same_beta_0", 0.0, 0.0),
                                               ("mixed_oracle0_minus_naive0.3", 0.0, 0.3))}
    oracle_vs_c0 = {c: {f"{b:g}": probe.r1_diff(ranks[key("oracle", c, b)], ranks[key("oracle", "C0", b)])
                        for b in ORACLE_BETAS} for c in VS_C0}
    oracle_vs_r3 = {m: probe.r1_diff(ranks[key("oracle", m, 0.0)], ranks[key("oracle", "R3", 0.0)])
                    for m in MODELS if m != "R3"}
    headline = {m: {"naive": probe.r1_with_ci(naive[m]),
                    **{f"oracle_{b:g}": probe.r1_with_ci(ranks[key("oracle", m, b)]) for b in ORACLE_BETAS},
                    f"oracle_null_{NULL_BETA:g}": probe.r1_with_ci(ranks[key("oracle_null", m, NULL_BETA)])}
                for m in MODELS}
    headline["clip_only"] = probe.r1_with_ci(ranks[key("clip_only", "-", BETA_FIXED)])
    r1 = {k: probe.r1_points(v) for k, v in ranks.items()}
    timings["comparisons"] = perf_counter() - t0

    rule = apply_affect_rule({m: gates[m]["passed"] for m in ("C0", *CELLS)}, d_emo, d_style)
    training = {m: training_record(m) for m in ("C0", "S", *CELLS)}

    t0 = perf_counter()
    extras = {"guard_power": guard_power(d_style, d_emo),
              "balance_matched": balance_matched(spread, r1, weights, img, txt, codes_sel, episodes, naive),
              "per_target": per_target(ranks, episodes), "ties": tie_info}
    timings["extras_stored"] = perf_counter() - t0
    t0 = perf_counter()
    ph = _load_posthoc()
    labels = {"emotion": np.where(allowed, np.asarray(data.emotions), ""),
              "art_style": np.where(allowed, np.asarray(data.art_styles), "")}
    txt_clip = sel.masked(data.txt_features, np.concatenate([st, sl]))
    probe.assert_row_scope("txt features (probes)", txt_clip, allowed)
    extras["ami"] = ami_analysis(ph, codes_full, cache, labels, affect_local)
    extras["code_probes"], extras["residual_probes"] = probe_analyses(ph, codes_full, txt_clip, cache, labels)
    timings["extras_probes_ami"] = perf_counter() - t0

    results = {
        "label": "affect selection evaluation (spec §6) on selection rows; seed-42 models; pre-registered rule",
        "settings": {"models": list(MODELS), "candidates": list(CELLS), "betas": list(BETAS),
                     "beta_fixed": BETA_FIXED, "n_episodes_per_label": N_EPISODES, "prefix_asserted": N_PREFIX,
                     "oracle": {**probe.ORACLE_SETTINGS, "steps": ORACLE_STEPS, "betas": list(ORACLE_BETAS),
                                "null_beta": NULL_BETA},
                     "bootstrap": {"n_boot": 5000, "seed": 42, "unit": "episode (paired)"},
                     "rule": {"binding_gates": list(BINDING_GATES), "style_margin": STYLE_MARGIN,
                              "tie_points": TIE_POINTS, "changes": CHANGES},
                     "device": grid.DEVICE, "gate_thresholds": "AMENDED_2026_09_29_THRESHOLDS",
                     "readout_reference": reference,
                     "probes": {**ph.PROBE_SETTINGS, "cluster_bootstrap": ph.CLUSTER_BOOT,
                                "fit": "scorer-train rows", "score": "selection rows"},
                     "rows": "gates fit on scorer-train rows, evaluated on selection rows; episodes, CLIP features and "
                             "evaluation codes NaN outside selection rows; val and held never read"},
        "episodes": eps_meta, "models": model_info, "gates": gates, "code_scale": scale,
        "naive_weights": weights_info, "term_spread_beta0.3": spread, "r1": r1, "naive_tied_episodes": ties,
        "headline_beta0.3": headline, "chance_r1": probe.CHANCE_R1,
        "d_emo": d_emo, "d_style": d_style, "d_pooled": d_pooled, "vs_c0": vs_c0, "vs_c0_beta_grid": vs_c0_beta,
        "vs_r3": vs_r3, "oracle_minus_own_naive": oracle_vs_naive, "oracle_minus_c0_oracle": oracle_vs_c0,
        "oracle_minus_r3_oracle": oracle_vs_r3, "rule": rule, "training": training, "extras": extras,
        "affect_diagnostics": {"ami": prep_record["ami"], "probe": prep_record["probe"],
                               "group_sizes": prep_record["group_sizes"]},
        "grid_reproduction_identical_share": reproduction}
    timings["total"] = perf_counter() - started
    results["timings_seconds"] = timings
    SELECTION_JSON.write_text(json.dumps(grid._jsonable(results), indent=2))
    np.savez(SELECTION_NPZ, **{f"{k.replace('|', '__')}__{label}__{dd}": np.asarray(v[label][dd])
                               for k, v in ranks.items() for label in LABELS for dd in DIRECTIONS})
    verdict = (f"STOP: {rule['stop']}" if rule["stop"] else f"PICKED {rule['picked']} (qualifying "
               f"{rule['qualifying']}, tie band {rule['tie_band']})")
    log("D_emo / D_style vs C0 (mean of directions, R@1 points): "
        + "; ".join(f"{c} {_ci(d_emo[c])} / {_ci(d_style[c])}" for c in VS_C0))
    log(f"Binding gates passed: {rule['binding_ok']}")
    log(f"Rule outcome: {verdict}")
    log(f"Evaluation in {timings['total']:.1f} s -> {SELECTION_JSON}")
    return results


# ----------------------------------------------------------------------------- Task 4: replication

REPLICATION_SEEDS = (43, 44)
REPLICATION_JSON = RESULTS / "replication.json"


def replicate() -> dict:
    """Seeds 43 and 44 of the picked cell and C0, on evaluate()'s selection episodes and code path (same rows,
    masking, gates with the same fit/eval rows and readout reference, naive at beta 0.3, paired bootstrap).
    Reported only; the verdict rests on seed 42. Seed 42 is recomputed through this path as a consistency check."""
    started = perf_counter()
    picked = json.loads(SELECTION_JSON.read_text())["rule"]["picked"]
    if picked not in CELLS:
        raise AssertionError(f"selection did not pick a cell: {picked}")
    data = load_artelingo()
    cache, prep, meta, _ = grid.load_grid()
    prep_record = json.loads((CACHE / "affect_prepare.json").read_text())
    st, sl = cache["scorer_train"], cache["selection"]
    n_rows = len(cache["groups"])
    community_local = cache["community"][st]
    if not np.array_equal(community_local, prep["community_local"]):
        raise AssertionError("community labels differ from the grid's prepared ones")
    episodes, _, eps_meta = selection_episodes(data, cache)
    img, txt = (sel.masked(x, sl) for x in (data.img_features, data.txt_features))
    allowed, in_sel = grid._row_mask(n_rows, np.concatenate([st, sl])), grid._row_mask(n_rows, sl)
    for name, arr in (("img eval features", img), ("txt eval features", txt)):
        probe.assert_row_scope(name, arr, in_sel)
    reference = meta["r0_readout_reference"]

    def one(name: str, seed: int) -> dict:
        ic_full, tc_full, info = model_codes(name, data, cache, prep_record, seed)
        for side, arr in (("img", ic_full), ("txt", tc_full)):
            probe.assert_row_scope(f"{name} seed {seed} {side} codes", arr, allowed)
        g = grid.gate_report(ic_full[st], tc_full[st], ic_full[sl], tc_full[sl], data, cache, community_local,
                             reference=reference)
        passed = {n: bool(v) for n, v in g.passed.items()}
        ic, tc = sel.masked(ic_full, sl), sel.masked(tc_full, sl)
        weights = {label: label_episode_weights(ic, tc, episodes[label]) for label in LABELS}
        ranks, _ = probe.fixed_weight_ranks(img, txt, ic, tc, episodes, weights, BETA_FIXED)
        return {"ranks": ranks, "model": info, "training": training_record(name, seed),
                "gates": {"values": grid._jsonable(g.values), "passed": passed,
                          "binding_passed": bool(all(passed[n] for n in BINDING_GATES)),
                          "n_binding_passed": int(sum(passed[n] for n in BINDING_GATES)),
                          "sparsity_passed": passed["sparsity"], "n_passed": int(sum(passed.values()))},
                "naive_r1": probe.r1_with_ci(ranks)}

    per_seed = {}
    for seed in (SEED, *REPLICATION_SEEDS):
        t0 = perf_counter()
        models = {name: one(name, seed) for name in (picked, "C0")}
        diff = probe.r1_diff(models[picked]["ranks"], models["C0"]["ranks"])
        per_seed[str(seed)] = {
            "d_emo": diff["emotion"]["mean"], "d_style": diff["art_style"]["mean"], "d_pooled": diff["pooled"]["mean"],
            "vs_c0": diff,
            "binding_ok": {m: models[m]["gates"]["binding_passed"] for m in models},
            "sparsity_passed": {m: models[m]["gates"]["sparsity_passed"] for m in models},
            "models": {m: {k: v for k, v in models[m].items() if k != "ranks"} for m in models}}
        log(f"seed {seed} {picked} - C0: D_emo {_ci(diff['emotion']['mean'])}, D_style {_ci(diff['art_style']['mean'])}, "
            f"pooled {_ci(diff['pooled']['mean'])}; binding {per_seed[str(seed)]['binding_ok']}, sparsity "
            f"{per_seed[str(seed)]['sparsity_passed']} ({perf_counter() - t0:.1f} s)")
    stored = json.loads(SELECTION_JSON.read_text())
    check = {"d_emo_point_diff": per_seed[str(SEED)]["d_emo"]["point"] - stored["d_emo"][picked]["point"],
             "d_style_point_diff": per_seed[str(SEED)]["d_style"]["point"] - stored["d_style"][picked]["point"]}
    if max(abs(v) for v in check.values()) > 1e-6:
        raise AssertionError(f"seed-42 recomputation differs from selection_results.json: {check}")
    results = {"label": f"replication (seeds {list(REPLICATION_SEEDS)}) of the picked cell {picked} vs C0 on the "
                        f"selection episodes; REPORTED ONLY, the verdict rests on seed 42 (seed 42 recomputed here "
                        f"as a consistency check)",
              "picked": picked, "episodes": eps_meta,
              "settings": {"n_episodes_per_label": N_EPISODES, "beta_fixed": BETA_FIXED,
                           "bootstrap": {"n_boot": 5000, "seed": 42, "unit": "episode (paired)"},
                           "readout_reference": reference, "rows": "as evaluate()"},
              "seed42_consistency_vs_selection_json": check, "per_seed": per_seed,
              "seconds": perf_counter() - started}
    REPLICATION_JSON.write_text(json.dumps(grid._jsonable(results), indent=2))
    log(f"Replication in {results['seconds']:.1f} s -> {REPLICATION_JSON}")
    return results


def replication_tables(path: Path = REPLICATION_JSON) -> None:
    res = json.loads(path.read_text())
    picked = res["picked"]
    out = [f"### Replication: {picked} - C0 per seed (naive, beta 0.3; paired R@1 points, 95% CI; reported only)\n",
           "| Seed | D_emo | D_style | pooled | binding gates " + f"{picked} / C0 | sparsity {picked} / C0 |",
           "|---|---:|---:|---:|---|---|"]
    for seed, v in res["per_seed"].items():
        b, sp = v["binding_ok"], v["sparsity_passed"]
        out.append(f"| {seed}{' (selection)' if int(seed) == SEED else ''} | {_ci(v['d_emo'])} | {_ci(v['d_style'])} | "
                   f"{_ci(v['d_pooled'])} | {v['models'][picked]['gates']['n_binding_passed']}/8 "
                   f"({'pass' if b[picked] else 'FAIL'}) / {v['models']['C0']['gates']['n_binding_passed']}/8 "
                   f"({'pass' if b['C0'] else 'FAIL'}) | "
                   + " / ".join(f"{grid._gate_value('sparsity', v['models'][m]['gates']['values'])} "
                                f"{'pass' if sp[m] else 'FAIL'}" for m in (picked, "C0")) + " |")
    out.append("\n| Seed | model | naive R@1 emotion (mean) | naive R@1 art style (mean) | pooled | final condition loss | "
               "final tau | wall-clock (min) |")
    out.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for seed, v in res["per_seed"].items():
        for m, d in v["models"].items():
            r, t = d["naive_r1"], d["training"]
            out.append(f"| {seed} | {m} | {_r1ci(r['emotion']['mean'])} | {_r1ci(r['art_style']['mean'])} | "
                       f"{_r1ci(r['pooled']['mean'])} | "
                       f"{'n/a' if t['final_condition_loss'] is None else format(t['final_condition_loss'], '.3f')} | "
                       f"{'n/a' if t['final_tau'] is None else format(t['final_tau'], '.4f')} | {t['seconds'] / 60:.1f} |")
    out.append(f"\nSeed-42 recomputation vs selection_results.json: {res['seed42_consistency_vs_selection_json']}")
    print("\n".join(out))


# ----------------------------------------------------------------------------- Task 3: tables

def _ci(block: dict) -> str:
    lo, hi = block["ci95"]
    return f"{block['point']:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def _r1ci(block: dict) -> str:
    lo, hi = block["ci95"]
    return f"{block['point']:.2f} [{lo:.2f}, {hi:.2f}]"


def tables(path: Path = SELECTION_JSON) -> None:
    res = json.loads(path.read_text())
    ex = res["extras"]
    out = [f"All numbers: {res['label']}. Episodes: "
           + "; ".join(f"{lab} n={e['n']} targets={e['targets']} sha {e['sha256'][:12]} (prefix {e['prefix_2048_sha256'][:12]} "
                       f"= stage (d))" for lab, e in res["episodes"].items()) + "\n"]

    out.append("### Training runs (seed 42, 2,000 steps, scorer-train rows)\n")
    out.append("| Model | wall-clock (min) | peak GPU (GiB) | final loss | final agreement | condition loss: step 1 / "
               "mean 50-500 / mean last 10 / final | tau: step 1 / final | views |")
    out.append("|---|---:|---:|---:|---:|---|---|---|")
    for m, t in res["training"].items():
        c = t.get("condition_summary")
        cl = "n/a" if c is None else (f"{c['step_1']:.3f} / {c['mean_steps_50_500']:.3f} / "
                                      f"{c['mean_last_10_logged']:.3f} / {c['final_single_batch']:.3f}")
        tau = "n/a" if c is None else f"{c['tau_step_1']:.4f} / {c['tau_final']:.4f}"
        out.append(f"| {m} | {t['seconds'] / 60:.1f} | {t['peak_gpu_gib']:.2f} | {t['final_loss']:.4f} | "
                   f"{t['final_agreement']:.4f} | {cl} | {tau} | {t['source_views'] or ''} |")
    out.append("")

    out.append("### Gates (fit scorer-train, eval selection; amended 2026-09-29 thresholds; img / txt where two)\n")
    out.append("| Gate | " + " | ".join(MODELS) + " |")
    out.append("|---|" + "---|" * len(MODELS))
    for gname in GATE_NAMES:
        tag = "" if gname in BINDING_GATES else " (reported only)"
        cells = [f"{grid._gate_value(gname, res['gates'][m]['values'])} "
                 f"{'pass' if res['gates'][m]['passed'][gname] else 'FAIL'}" for m in MODELS]
        out.append(f"| {gname}{tag} | " + " | ".join(cells) + " |")
    out.append("| **binding 8** | " + " | ".join(f"{res['gates'][m]['n_binding_passed']}/8" for m in MODELS) + " |")
    out.append("| all 9 | " + " | ".join(f"{res['gates'][m]['n_passed']}/9" for m in MODELS) + " |")
    ref = res["settings"]["readout_reference"]
    out.append(f"\nReadout reference (R0 on the same rows): img {ref[0]:.4f}, txt {ref[1]:.4f}.\n")

    out.append("### Criterion: D_emo, D_style and pooled vs C0 (naive, beta 0.3; paired R@1 points, 95% CI)\n")
    out.append("| Cell | D_emo (mean) | D_emo i2t | D_emo t2i | D_style (mean) | D_style i2t | D_style t2i | pooled mean |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for c in VS_C0:
        v = res["vs_c0"][c]
        out.append(f"| {c}{' (reference)' if c == 'S' else ''} | {_ci(res['d_emo'][c])} | {_ci(v['emotion']['i2t'])} | "
                   f"{_ci(v['emotion']['t2i'])} | {_ci(res['d_style'][c])} | {_ci(v['art_style']['i2t'])} | "
                   f"{_ci(v['art_style']['t2i'])} | {_ci(res['d_pooled'][c])} |")
    out.append("")

    out.append("### Naive R@1 (%) at beta 0.3, 95% bootstrap CI\n")
    out.append("| Model | " + " | ".join(f"{s} {d}" for s in SCOPES for d in (*DIRECTIONS, "mean")) + " |")
    out.append("|---|" + "---:|" * (3 * len(SCOPES)))
    for m in (*MODELS, "clip_only"):
        h = res["headline_beta0.3"][m] if m == "clip_only" else res["headline_beta0.3"][m]["naive"]
        out.append(f"| {m} | " + " | ".join(_r1ci(h[s][d]) for s in SCOPES for d in (*DIRECTIONS, "mean")) + " |")
    out.append(f"\nChance {res['chance_r1']:.2f}.\n")

    out.append("### Context: naive(model, 0.3) - naive(original R3, 0.3) (paired R@1 points)\n")
    out.append("| Model | pooled mean | emotion mean | art style mean |")
    out.append("|---|---:|---:|---:|")
    for m, v in res["vs_r3"].items():
        out.append(f"| {m} | {_ci(v['pooled']['mean'])} | {_ci(v['emotion']['mean'])} | {_ci(v['art_style']['mean'])} |")
    out.append("")

    out.append("### Naive R@1 (%) over the beta grid (mean of directions): pooled / emotion / style\n")
    out.append("| Model | " + " | ".join(f"beta {b:g}" for b in BETAS) + " |")
    out.append("|---|" + "---|" * len(BETAS))
    for m in MODELS:
        cells = []
        for b in BETAS:
            r = res["r1"][key("naive", m, b)]
            cells.append(f"{r['pooled']['mean']:.2f} / {r['emotion']['mean']:.2f} / {r['art_style']['mean']:.2f}")
        out.append(f"| {m} | " + " | ".join(cells) + " |")
    out.append("")
    out.append("### Emotion naive R@1 (%) per direction on the beta grid (i2t / t2i)\n")
    out.append("| Model | " + " | ".join(f"beta {b:g}" for b in BETAS) + " | oracle 0 |")
    out.append("|---|" + "---|" * (len(BETAS) + 1))
    for m in MODELS:
        cells = [f"{res['r1'][key('naive', m, b)]['emotion']['i2t']:.2f} / "
                 f"{res['r1'][key('naive', m, b)]['emotion']['t2i']:.2f}" for b in BETAS]
        o = res["r1"][key("oracle", m, 0.0)]["emotion"]
        out.append(f"| {m} | " + " | ".join(cells) + f" | {o['i2t']:.2f} / {o['t2i']:.2f} |")
    out.append("")
    out.append("### Cell - C0 at the same beta (naive; paired R@1 points, 95% CI; context, not gating)\n")
    out.append("| Cell | beta | pooled mean | emotion mean | emotion i2t | emotion t2i | art style mean |")
    out.append("|---|---:|---:|---:|---:|---:|---:|")
    for c, per_beta in res["vs_c0_beta_grid"].items():
        for b, v in per_beta.items():
            out.append(f"| {c} | {b} | {_ci(v['pooled']['mean'])} | {_ci(v['emotion']['mean'])} | "
                       f"{_ci(v['emotion']['i2t'])} | {_ci(v['emotion']['t2i'])} | {_ci(v['art_style']['mean'])} |")
    out.append("")
    out.append("### Code scale and score-term spread at beta 0.3\n")
    out.append("| Model | mean RMS (scorer-train) | mean RMS (selection) | spread cos | spread factor | "
               "factor / (0.3 cos) | all-zero naive weights | mean non-zero factors |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for m in MODELS:
        s, t, w = res["code_scale"][m], res["term_spread_beta0.3"][m], res["naive_weights"][m]
        out.append(f"| {m} | {s['mean_rms_scorer_train']:.4f} | {s['mean_rms_selection']:.4f} | {t['cos_spread']:.4f} | "
                   f"{t['factor_spread']:.4f} | {t['factor_to_clip']:.2f} | {100 * w['all_zero_share']:.2f}% | "
                   f"{w['mean_nonzero_factors']:.1f} |")
    out.append("")
    out.append("### Balance-matched comparison vs C0 (naive; R@1 points)\n")
    out.append("| Cell | ratio x / C0 | matched beta C0 / x | comparison | method | pooled | emotion | art style |")
    out.append("|---|---|---|---|---|---:|---:|---:|")
    for x, bm in ex["balance_matched"].items():
        head = (f"{bm['factor_to_clip'][x]:.2f} / {bm['factor_to_clip']['C0']:.2f} | "
                f"{bm['beta_C0_matched']:.4f} / {bm['beta_x_matched']:.4f}")
        for variant, v in bm["interpolated"]["pooled"].items():
            i = bm["interpolated"]
            out.append(f"| {x} | {head} | {variant} | interpolated | {v:+.2f} | {i['emotion'][variant]:+.2f} | "
                       f"{i['art_style'][variant]:+.2f} |")
        for variant, v in bm["rescored"].items():
            out.append(f"| {x} | {head} | {variant} | re-scored | {_ci(v['pooled']['mean'])} | "
                       f"{_ci(v['emotion']['mean'])} | {_ci(v['art_style']['mean'])} |")
    out.append("")

    out.append("### Label oracle (%, mean of directions, 95% CI) and its null\n")
    out.append("| Model | naive 0.3 pooled | oracle 0.3 pooled | oracle 0 pooled | oracle 0 emotion | oracle 0 style | "
               "null 0 pooled | oracle - own naive, same 0.3 (pooled) | same 0 (pooled) | same 0 (emotion) | "
               "oracle 0 - R3 oracle 0 |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for m in MODELS:
        h, o = res["headline_beta0.3"][m], res["oracle_minus_own_naive"]
        diff_r3 = _ci(res["oracle_minus_r3_oracle"][m]["pooled"]["mean"]) if m != "R3" else "(reference)"
        out.append(f"| {m} | {_r1ci(h['naive']['pooled']['mean'])} | {_r1ci(h['oracle_0.3']['pooled']['mean'])} | "
                   f"{_r1ci(h['oracle_0']['pooled']['mean'])} | {_r1ci(h['oracle_0']['emotion']['mean'])} | "
                   f"{_r1ci(h['oracle_0']['art_style']['mean'])} | {_r1ci(h['oracle_null_0']['pooled']['mean'])} | "
                   f"{_ci(o['same_beta_0.3'][m]['pooled']['mean'])} | {_ci(o['same_beta_0'][m]['pooled']['mean'])} | "
                   f"{_ci(o['same_beta_0'][m]['emotion']['mean'])} | {diff_r3} |")
    out.append("\n| Cell - C0, oracle | beta | pooled | emotion | emotion i2t | emotion t2i | art style |")
    out.append("|---|---:|---:|---:|---:|---:|---:|")
    for c, per_beta in res["oracle_minus_c0_oracle"].items():
        for b, v in per_beta.items():
            out.append(f"| {c} | {b} | {_ci(v['pooled']['mean'])} | {_ci(v['emotion']['mean'])} | "
                       f"{_ci(v['emotion']['i2t'])} | {_ci(v['emotion']['t2i'])} | {_ci(v['art_style']['mean'])} |")
    out.append("")

    gp = ex["guard_power"]
    out.append("### Style guard power (normal approximation from the measured D_style CIs)\n")
    out.append("| Cell | D_style | half-width | point - lower bound |")
    out.append("|---|---:|---:|---:|")
    for c, v in gp["cells"].items():
        out.append(f"| {c} | {v['point']:+.2f} [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}] | {v['half_width']:.2f} | "
                   f"{v['lower_distance']:.2f} |")
    for name, block, se in (("measured", gp["measured"], gp["se_measured"]),
                            ("spec (pre-run)", gp["spec_planned"], gp["spec_planned"]["se"])):
        out.append(f"\n{name}: SE {se:.3f}; passes when the point estimate exceeds {block['pass_threshold_point']:+.3f}. "
                   "P(pass | true style effect): " + ", ".join(f"{k}: {100 * v:.1f}%"
                                                              for k, v in block["by_true_effect"].items()))
    out.append(f"\nD_emo SE (measured, mean of E and SE): {gp['d_emo_se_measured']:.3f}\n")

    tie = ex["ties"]
    out.append("### Ties: naive R@1 (%) as scored (a tie is a miss) vs random tie-breaking, mean of directions\n")
    out.append("| Model | beta | pooled scored / random | emotion scored / random | style scored / random | "
               "positive tied at top (emo i2t/t2i, style i2t/t2i) | episodes with any tie (emo i2t/t2i, style i2t/t2i) |")
    out.append("|---|---:|---|---|---|---|---|")
    for m in MODELS:
        for b, t in tie[m].items():
            tt, anyt = t["positive_tied_at_top"], res["naive_tied_episodes"][key("naive", m, float(b))]
            out.append(f"| {m} | {b} | " + " | ".join(
                f"{t['tie_aware'][s]['mean']:.2f} / {t['random_tie_break'][s]['mean']:.2f}" for s in SCOPES)
                + f" | {tt['emotion|i2t']} / {tt['emotion|t2i']}, {tt['art_style|i2t']} / {tt['art_style|t2i']} | "
                f"{anyt['emotion']['i2t']} / {anyt['emotion']['t2i']}, {anyt['art_style']['i2t']} / "
                f"{anyt['art_style']['t2i']} |")
    out.append("")

    for x, per_label in ex["per_target"].items():
        out.append(f"### Per-target {x} - C0 (mean of directions; paired bootstrap per target)\n")
        for label, pt in per_label.items():
            cols = list(pt)
            out.append(f"**{label}**\n")
            out.append("| Target | n | " + " | ".join(f"{c}: C0 / {x} / {x} - C0 [CI] / i2t, t2i" for c in cols) + " |")
            out.append("|---|---:|" + "---|" * len(cols))
            for t in sorted(pt[cols[0]], key=lambda t: pt["naive@0.3"][t]["diff"]["point"]):
                cells = [f"{pt[c][t]['C0']:.1f} / {pt[c][t][x]:.1f} / {_ci(pt[c][t]['diff'])} / "
                         f"{pt[c][t]['diff_i2t']:+.1f}, {pt[c][t]['diff_t2i']:+.1f}" for c in cols]
                out.append(f"| {t} | {pt[cols[0]][t]['n']} | " + " | ".join(cells) + " |")
            out.append("")

    a = ex["ami"]
    out.append("### AMI of the argmax pair-code factor\n")
    out.append("| Model | CLIP image clusters (st) | affect clusters (st) | art style (sel) | emotion (sel) | "
               "art style (st) | emotion (st) | factors used | all-zero rows |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for m in MODELS:
        v = a[m]
        out.append(f"| {m} | {v['clip_image@scorer_train']:.4f} | {v['affect_clusters@scorer_train']:.4f} | "
                   f"{v['art_style@selection']:.4f} | {v['emotion@selection']:.4f} | {v['art_style@scorer_train']:.4f} | "
                   f"{v['emotion@scorer_train']:.4f} | {v['factors_used@selection']} | "
                   f"{100 * v['all_zero_share@selection']:.2f}% |")
    ref = a["reference"]
    out.append(f"\nReference (scorer-train): CLIP image clusters vs style {ref['clip_image']['art_style']:.4f} / emotion "
               f"{ref['clip_image']['emotion']:.4f}; affect clusters vs style {ref['affect_clusters']['art_style']:.4f} / "
               f"emotion {ref['affect_clusters']['emotion']:.4f}; affect vs CLIP image clusters "
               f"{ref['affect_vs_clip_image']:.4f}.\n")

    out.append("### Linear probes on 32-d codes: top-1 accuracy (%) on selection rows (minus C0, painting bootstrap)\n")
    cp = ex["code_probes"]
    tasks = list(cp["C0"])
    out.append("| Model | " + " | ".join(tasks) + " |")
    out.append("|---|" + "---:|" * len(tasks))
    for m in MODELS:
        out.append(f"| {m} | " + " | ".join(f"{cp[m][t]['accuracy']:.2f}" + (
            f" ({_ci(cp[m][t]['minus_C0'])})" if "minus_C0" in cp[m][t] else "") for t in tasks) + " |")
    out.append("| majority | " + " | ".join(f"{cp['C0'][t]['majority']:.2f}" for t in tasks) + " |")
    out.append(f"\nAll probes converged: {all(cp[m][t]['converged'] for m in MODELS for t in tasks)}.\n")
    rp = ex["residual_probes"]
    out.append("### Within-painting caption-residual emotion probe (%)\n")
    out.append("| Code | accuracy | minus C0 | majority | within-painting variance share | fit / eval rows | converged |")
    out.append("|---|---:|---:|---:|---:|---|---|")
    for m in (*MODELS, "clip512"):
        v = rp[m]
        diff = _ci(v["minus_C0"]) if "minus_C0" in v else ""
        out.append(f"| {m} | {v['accuracy']:.2f} | {diff} | {v['majority']:.2f} | "
                   f"{v['within_painting_variance_share']:.3f} | {v['n_fit']:,} / {v['n_eval']:,} | {v['converged']} |")
    out.append("")

    diag = res["affect_diagnostics"]
    out.append("### Affect diagnostics (prepare, scorer-train rows; context)\n")
    out.append("| Partition | AMI emotion | AMI art style |")
    out.append("|---|---:|---:|")
    for name, v in diag["ami"].items():
        out.append(f"| {name} | {v['emotion']:.3f} | {v['art_style']:.3f} |")
    pr = diag["probe"]
    out.append(f"\nProbe -> emotion (20% of scorer-train paintings): affect-28 {100 * pr['affect28_to_emotion']:.1f}%, "
               f"CLIP caption {100 * pr['clip_caption_to_emotion']:.1f}%, majority {100 * pr['majority_class']:.1f}%.\n")

    rule = res["rule"]
    out.append("### Rule outcome (spec §6)\n")
    out.append(f"- binding gates passed: {rule['binding_ok']}")
    for c in CELLS:
        out.append(f"- {c}: binding {rule['binding_ok'][c]}; D_emo lower bound {res['d_emo'][c]['ci95'][0]:+.2f} "
                   f"(> 0: {res['d_emo'][c]['ci95'][0] > 0}); D_style lower bound {res['d_style'][c]['ci95'][0]:+.2f} "
                   f"(> {STYLE_MARGIN}: {res['d_style'][c]['ci95'][0] > STYLE_MARGIN})")
    out.append(f"- qualifying: {rule['qualifying']}; tie band: {rule['tie_band']}")
    out.append(f"- **{'STOP: ' + rule['stop'] if rule['stop'] else 'picked: ' + rule['picked']}**")
    out.append(f"\nFirst 2,048 episodes vs the 2x2's stored ranks (identical share): "
               f"{res['grid_reproduction_identical_share']}")
    out.append("Timings (s): " + ", ".join(f"{k} {v:.1f}" for k, v in res["timings_seconds"].items()))
    print("\n".join(out))


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
    if args.replicate:
        replicate()
        replication_tables()
    if args.prepare:
        prepare()
    if args.smoke:
        smoke()
    if args.run:
        run_affect_cell(args.run, args.seed, overwrite=args.overwrite)
    if args.evaluate:
        evaluate()
        tables()
    elif args.tables:
        tables()
        if REPLICATION_JSON.exists():
            replication_tables()


if __name__ == "__main__":
    main()
