"""CoSiR v2 Candidate A factor-learning 2x2 grid (spec docs/superpowers/specs/2026-09-30-cosir-v2-candidate-a-factor-learning-design.md).

Four factor models on scorer-train rows (spec §4): C0 (row agreement, no condition; the matched control),
A (painting agreement), S (CLIP image k-means condition episodes), AS (both). Run from the repository root:

    python src/test/20261016_factor_learning_grid/run_grid.py --prepare
    python src/test/20261016_factor_learning_grid/run_grid.py --smoke            # timing: local GPU or DAS6?
    python src/test/20261016_factor_learning_grid/run_grid.py --run C0 --seed 42
    python src/test/20261016_factor_learning_grid/run_grid.py --evaluate         # Task 4: gates, metrics, rule
    python src/test/20261016_factor_learning_grid/run_grid.py --tables           # reprint from the JSON

Row scope: training, the graph and the partitions use scorer-train rows only (local indices 0..n-1); evaluation
reads selection rows only; val and held rows are never read.

--evaluate (spec §6): the four seed-42 checkpoints plus original R3 (stage (d)'s cached codes) on stage (d)'s
selection label episodes (SHA-256 asserted): the nine gates, the naive rule on the beta grid, the label oracle
at beta 0.3 and 0 with its null at 0, CLIP-only, D and D_emotion vs C0, and the pre-registered rule
(``apply_rule``). Writes results/selection_results.json and results/selection_ranks.npz.
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
_PROBE_PATH = ROOT / "src/test/20261015_factor_headroom_probe/run_probe.py"
_pspec = importlib.util.spec_from_file_location("run_probe", _PROBE_PATH)
probe = importlib.util.module_from_spec(_pspec)
_pspec.loader.exec_module(probe)

from src.data.artelingo import load_artelingo  # noqa: E402
from src.eval.factor_gates import AMENDED_2026_09_29_THRESHOLDS, evaluate_factor_gates  # noqa: E402
from src.eval.label_episodes import (label_episode_weights, label_episodes_sha256,  # noqa: E402
                                     standard_label_episodes)
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


# ----------------------------------------------------------------------------- Task 4: selection rule

CHANGES = {"A": 1, "S": 1, "AS": 2}
TIE_POINTS, GUARD_POINTS = 0.5, -1.0


def apply_rule(gates_ok: dict, d: dict, d_emotion: dict) -> dict:
    """Spec §6. Eligible = all gates pass; qualifies = eligible and D lower bound > 0 and D_emotion lower bound
    > -1.0; pick = highest D, cells within 0.5 points tie, a tie goes to fewer changes (A or S before AS),
    then to higher D. ``d`` / ``d_emotion`` map cell -> {"point": R@1 points, "ci95": [lo, hi]}."""
    if not gates_ok["C0"]:
        return {"stop": "C0 fails the gates: the setup is broken", "picked": None, "qualifying": [], "tie_band": []}
    qualifying = [c for c in ("A", "S", "AS")
                  if gates_ok[c] and d[c]["ci95"][0] > 0 and d_emotion[c]["ci95"][0] > GUARD_POINTS]
    if not qualifying:
        return {"stop": "no cell qualifies", "picked": None, "qualifying": [], "tie_band": []}
    best = max(d[c]["point"] for c in qualifying)
    tied = [c for c in qualifying if d[c]["point"] >= best - TIE_POINTS]
    picked = min(tied, key=lambda c: (CHANGES[c], -d[c]["point"]))
    return {"stop": None, "picked": picked, "qualifying": qualifying, "tie_band": tied}


def _check_rule() -> None:
    ok = {"C0": True, "A": True, "S": True, "AS": True}
    blk = lambda p, lo: {"point": p, "ci95": [lo, p + 1]}          # noqa: E731
    fine = {c: blk(0.0, -0.5) for c in CHANGES}
    assert apply_rule({**ok, "C0": False}, fine, fine)["stop"].startswith("C0")
    assert apply_rule(ok, {c: blk(1.0, -0.1) for c in CHANGES}, fine)["stop"] == "no cell qualifies"
    d = {"A": blk(2.0, 0.5), "S": blk(1.2, 0.1), "AS": blk(2.4, 0.9)}
    assert apply_rule(ok, d, fine)["picked"] == "A"                  # AS within 0.5 of best -> fewer changes
    d = {"A": blk(1.0, 0.2), "S": blk(1.3, 0.3), "AS": blk(2.4, 0.9)}
    assert apply_rule(ok, d, fine)["picked"] == "AS"                 # A and S fall outside the tie band
    assert apply_rule(ok, d, {**fine, "AS": blk(-2.0, -3.0)})["picked"] == "S"   # emotion guard drops AS


# ----------------------------------------------------------------------------- Task 4: selection evaluation

MODELS = ("R3", "C0", "A", "S", "AS")                        # R3 = original R3 codes from stage (d)'s cache
GRID_CELLS = ("C0", "A", "S", "AS")
LABELS, SCOPES, DIRECTIONS = probe.LABELS, probe.SCOPES, probe.DIRECTIONS
BETAS, BETA_FIXED, key = probe.BETAS, 0.3, probe.key
ORACLE_STEPS, ORACLE_BETAS, NULL_BETA = 200, (0.3, 0.0), 0.0
N_EPISODES = 2048
IDENTICAL_SHARE_MIN = 0.999                                  # R3 rows vs the probe's stored ranks
PROBE_RANKS_NPZ = probe.RESULTS / "probe_ranks.npz"
SELECTION_JSON, SELECTION_NPZ = RESULTS / "selection_results.json", RESULTS / "selection_ranks.npz"
GATE_NAMES = ("participation_ratio", "redundancy", "readout", "sparsity", "dead", "modality_private",
              "usage_concentration", "community_spanning", "pair_retrieval")


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _row_mask(n_rows: int, rows: np.ndarray) -> np.ndarray:
    mask = np.zeros(n_rows, dtype=bool)
    mask[rows] = True
    return mask


def model_codes(name: str, data, cache) -> tuple[np.ndarray, np.ndarray]:
    """Full-length (n_rows, 32) codes: finite on scorer-train and selection rows, NaN elsewhere."""
    rows = np.concatenate([cache["scorer_train"], cache["selection"]])
    if name == "R3":
        return (sel.masked(cache["img_codes"], rows), sel.masked(cache["txt_codes"], rows))
    path = CKPT / f"{name}_seed{SEED}.pt"
    model, config = load_factor_checkpoint(path, device=DEVICE)
    if dataclasses.asdict(config) != dataclasses.asdict(cell_config(name, SEED)):
        raise AssertionError(f"{path.name}: stored config differs from the pre-registered cell config")
    ic, tc = encode_rows(model, data.img_features, data.txt_features, rows=rows)
    out = []
    for codes in (ic, tc):
        full = np.full((len(cache["groups"]), codes.shape[1]), np.nan, dtype=np.float32)
        full[rows] = codes
        out.append(full)
    return tuple(out)


def selection_episodes(data, cache) -> tuple[dict, dict, dict]:
    """Stage (d)'s selection label episodes (SHA-256 asserted) and the probe's null targets."""
    in_sel = _row_mask(len(cache["groups"]), cache["selection"])
    episodes, meta = {}, {}
    for label in LABELS:
        eps = standard_label_episodes(data, cache["groups"], cache["selection"], label, N_EPISODES, seed=SEED)
        sha = label_episodes_sha256(eps)
        if sha != fin.SELECTION_SHA256[label]:
            raise AssertionError(f"{label} selection episodes differ from stage (d)'s: {sha}")
        if not in_sel[np.column_stack([eps.anchor, eps.positive, eps.supports, eps.contrasts, eps.distractors])].all():
            raise AssertionError(f"{label} episodes use rows outside the selection set")
        episodes[label] = eps
        meta[label] = {"n": int(len(eps.anchor)), "targets": int(len(np.unique(eps.labels))), "sha256": sha,
                       "equals_stage_d_sha256": True, "all_rows_in_selection": True}
    if LABELS != ("emotion", "art_style"):
        raise AssertionError("null targets are drawn in (emotion, art_style) order")
    rng = np.random.default_rng(SEED)                   # exactly stage (d)'s / the probe's null targets
    nulls = {}
    for label in LABELS:
        if episodes[label].distractors.shape[1] + 1 != 13:
            raise AssertionError("expected 13 candidates per episode")
        nulls[label] = rng.integers(1, 13, len(episodes[label].anchor))
    log(f"Episodes (SHA-256 asserted, rows in selection asserted): {meta}")
    return episodes, nulls, meta


def weight_summary(weights: dict) -> dict:
    """Share of episodes whose naive weights are all zero (CLIP-only fallback) and mean non-zero factor count."""
    w = torch.cat([weights[label] for label in LABELS]).numpy()
    nonzero = (w > 0).sum(axis=1)
    return {"all_zero_share": float(np.mean(nonzero == 0)), "mean_nonzero_factors": float(nonzero.mean())}


def term_spread(img, txt, ic, tc, episodes: dict, weights: dict) -> dict:
    """Mean over episodes (both labels, both directions) of the across-candidate std of each score term.

    ``factor_to_clip`` = spread(factor term) / (0.3 * spread(cos)): how strongly the factor term outweighs
    the CLIP term at beta 0.3 for this model's code scale (larger = beta 0.3 acts like a smaller beta).
    """
    cos_sd, fac_sd = [], []
    for label in LABELS:
        eps = episodes[label]
        cands = np.concatenate([eps.positive[:, None], eps.distractors], axis=1)
        w = weights[label].numpy().astype(np.float64)
        for qf, cf, qc, cc in ((img, txt, ic, tc), (txt, img, tc, ic)):
            q, c = probe.l2_rows(qf[eps.anchor].astype(np.float64)), cf[cands].astype(np.float64)
            c = c / np.maximum(np.linalg.norm(c, axis=2, keepdims=True), 1e-12)
            cos_sd.append(np.einsum("nd,nkd->nk", q, c).std(axis=1))
            fac_sd.append(np.einsum("nf,nkf->nk", w * qc[eps.anchor], cc[cands].astype(np.float64)).std(axis=1))
    cos_sd, fac_sd = float(np.concatenate(cos_sd).mean()), float(np.concatenate(fac_sd).mean())
    return {"cos_spread": cos_sd, "factor_spread": fac_sd, "factor_to_clip": fac_sd / (BETA_FIXED * cos_sd)}


def history_record(cell: str) -> dict:
    record = json.loads((RESULTS / f"history_{cell}_seed{SEED}.json").read_text())
    if record["steps"] != FULL_STEPS:
        raise AssertionError(f"{cell}: history is not a {FULL_STEPS}-step run")
    h = record["history"]
    return {"seconds": record["seconds"], "peak_gpu_gib": record["peak_gpu_gib"], "history": h,
            "final_loss": h["loss"][-1], "final_agreement": h["agreement"][-1],
            "final_condition_loss": h["condition_loss"][-1] if "condition_loss" in h else None,
            "final_tau": h["tau"][-1] if "tau" in h else None}


def reproduce_probe(ranks: dict) -> dict:
    """R3 rows must reproduce the headroom probe's stored ranks (naive at every beta and CLIP-only asserted)."""
    stored = np.load(PROBE_RANKS_NPZ)
    out = {}
    for k in [key("naive", "R3", b) for b in BETAS] + [key("clip_only", "-", BETA_FIXED)] + \
             [key("oracle", "R3", b) for b in ORACLE_BETAS] + [key("oracle_null", "R3", NULL_BETA)]:
        prefix = k.replace("|", "__")
        share = min(float(np.mean(np.asarray(ranks[k][label][d]) == stored[f"{prefix}__{label}__{d}"]))
                    for label in LABELS for d in DIRECTIONS)
        out[k] = share
        if not k.startswith("oracle") and share < IDENTICAL_SHARE_MIN:
            raise AssertionError(f"{k}: R3 ranks differ from the headroom probe's (identical share {share:.4f})")
    log(f"R3 vs the probe's stored ranks (identical share): {out}")
    return out


def evaluate() -> dict:
    _check_rule()
    started, timings = perf_counter(), {}
    t0 = perf_counter()
    data = load_artelingo()
    cache, prep, meta, _ = load_grid()
    st, sl = cache["scorer_train"], cache["selection"]
    n_rows = len(cache["groups"])
    if np.intersect1d(cache["groups"][st], cache["groups"][sl]).size:
        raise AssertionError("a painting spans scorer-train and selection")
    episodes, nulls, eps_meta = selection_episodes(data, cache)
    img, txt = (sel.masked(x, sl) for x in (data.img_features, data.txt_features))
    allowed, in_sel = _row_mask(n_rows, np.concatenate([st, sl])), _row_mask(n_rows, sl)
    timings["load_and_episodes"] = perf_counter() - t0
    reference = meta["r0_readout_reference"]
    ranks, ties, gates, weights_info, spread, scale = {}, {}, {}, {}, {}, {}
    for name in MODELS:
        t0 = perf_counter()
        ic_full, tc_full = model_codes(name, data, cache)
        for side, arr in (("img", ic_full), ("txt", tc_full)):
            probe.assert_row_scope(f"{name} {side} codes", arr, allowed)
        g = gate_report(ic_full[st], tc_full[st], ic_full[sl], tc_full[sl], data, cache, prep["community_local"],
                        reference=reference)
        gates[name] = {"values": _jsonable(g.values), "passed": dict(g.passed), "all_passed": bool(g.all_passed),
                       "n_passed": int(sum(g.passed.values()))}
        scale[name] = {"mean_rms_scorer_train": probe.mean_rms(ic_full, tc_full, st),
                       "mean_rms_selection": probe.mean_rms(ic_full, tc_full, sl)}
        ic, tc = sel.masked(ic_full, sl), sel.masked(tc_full, sl)
        for side, arr in (("img", ic), ("txt", tc)):
            probe.assert_row_scope(f"{name} {side} eval codes", arr, in_sel)
        del ic_full, tc_full
        weights = {label: label_episode_weights(ic, tc, episodes[label]) for label in LABELS}
        weights_info[name] = weight_summary(weights)
        spread[name] = term_spread(img, txt, ic, tc, episodes, weights)
        for beta in BETAS:
            ranks[key("naive", name, beta)], ties[key("naive", name, beta)] = probe.fixed_weight_ranks(
                img, txt, ic, tc, episodes, weights, beta)
        for beta in ORACLE_BETAS:
            ranks[key("oracle", name, beta)] = probe.oracle_ranks(img, txt, ic, tc, episodes, beta, ORACLE_STEPS,
                                                                  DEVICE)
        ranks[key("oracle_null", name, NULL_BETA)] = probe.oracle_ranks(img, txt, ic, tc, episodes, NULL_BETA,
                                                                        ORACLE_STEPS, DEVICE, nulls)
        if name == "R3":                                # CLIP-only: zero weights, beta 0.3 (codes irrelevant)
            zero = {label: torch.zeros(len(episodes[label].anchor), ic.shape[1]) for label in LABELS}
            ranks[key("clip_only", "-", BETA_FIXED)], _ = probe.fixed_weight_ranks(img, txt, ic, tc, episodes, zero,
                                                                                   BETA_FIXED)
        timings[f"model:{name}"] = perf_counter() - t0
        r = {s: probe.r1_points(ranks[key(s, name, b)])["pooled"]["mean"]
             for s, b in (("naive", BETA_FIXED), ("oracle", 0.0))}
        log(f"{name}: gates {gates[name]['n_passed']}/9 (all passed: {gates[name]['all_passed']}); pooled R@1 "
            f"naive@0.3 {r['naive']:.2f}, oracle@0 {r['oracle']:.2f} ({timings[f'model:{name}']:.1f} s)")
    reproduction = reproduce_probe(ranks)

    t0 = perf_counter()
    naive = {m: ranks[key("naive", m, BETA_FIXED)] for m in MODELS}
    vs_c0 = {c: probe.r1_diff(naive[c], naive["C0"]) for c in ("A", "S", "AS")}
    d = {c: vs_c0[c]["pooled"]["mean"] for c in vs_c0}
    d_emotion = {c: vs_c0[c]["emotion"]["mean"] for c in vs_c0}
    vs_r3 = {m: probe.r1_diff(naive[m], naive["R3"]) for m in MODELS if m != "R3"}
    oracle_vs_naive = {m: probe.r1_diff(ranks[key("oracle", m, 0.0)], naive[m]) for m in MODELS}
    oracle_vs_r3_oracle = {m: probe.r1_diff(ranks[key("oracle", m, 0.0)], ranks[key("oracle", "R3", 0.0)])
                           for m in MODELS if m != "R3"}
    oracle_vs_c0_oracle = {c: probe.r1_diff(ranks[key("oracle", c, 0.0)], ranks[key("oracle", "C0", 0.0)])
                           for c in ("A", "S", "AS")}
    vs_c0_beta = {c: {f"{b:g}": probe.r1_diff(ranks[key("naive", c, b)], ranks[key("naive", "C0", b)])
                      for b in BETAS} for c in ("A", "S", "AS")}      # context: is a gain a code-scale effect?
    headline = {m: {"naive": probe.r1_with_ci(naive[m]),
                    **{f"oracle_{b:g}": probe.r1_with_ci(ranks[key("oracle", m, b)]) for b in ORACLE_BETAS},
                    f"oracle_null_{NULL_BETA:g}": probe.r1_with_ci(ranks[key("oracle_null", m, NULL_BETA)])}
                for m in MODELS}
    headline["clip_only"] = probe.r1_with_ci(ranks[key("clip_only", "-", BETA_FIXED)])
    timings["comparisons"] = perf_counter() - t0
    gates_ok = {m: gates[m]["all_passed"] for m in GRID_CELLS}
    rule = apply_rule(gates_ok, d, d_emotion)
    rule["gates_ok"] = gates_ok
    training = {c: history_record(c) for c in GRID_CELLS}
    results = {
        "label": "selection evaluation (spec §6) on selection rows; seed-42 models; pre-registered rule",
        "settings": {"models": list(MODELS), "betas": list(BETAS), "beta_fixed": BETA_FIXED,
                     "oracle": {**probe.ORACLE_SETTINGS, "steps": ORACLE_STEPS, "betas": list(ORACLE_BETAS),
                                "null_beta": NULL_BETA},
                     "bootstrap": {"n_boot": 5000, "seed": 42, "unit": "episode (paired)"},
                     "rule": {"tie_points": TIE_POINTS, "guard_points": GUARD_POINTS, "changes": CHANGES},
                     "device": DEVICE, "gate_thresholds": "AMENDED_2026_09_29_THRESHOLDS",
                     "readout_reference": reference,
                     "rows": "gates fit on scorer-train rows, evaluated on selection rows; episodes, CLIP features and "
                             "codes NaN outside selection rows; val and held never read"},
        "episodes": eps_meta, "gates": gates, "code_scale": scale, "naive_weights": weights_info,
        "term_spread_beta0.3": spread,
        "r1": {k: probe.r1_points(v) for k, v in ranks.items()}, "naive_tied_episodes": ties,
        "headline_beta0.3": headline, "chance_r1": probe.CHANCE_R1,
        "d": d, "d_emotion": d_emotion, "vs_c0": vs_c0, "vs_c0_beta_grid": vs_c0_beta, "vs_r3": vs_r3,
        "oracle_minus_own_naive": oracle_vs_naive, "oracle_minus_r3_oracle": oracle_vs_r3_oracle,
        "oracle_minus_c0_oracle": oracle_vs_c0_oracle, "rule": rule, "training": training,
        "probe_reproduction_identical_share": reproduction}
    timings["total"] = perf_counter() - started
    results["timings_seconds"] = timings
    SELECTION_JSON.write_text(json.dumps(_jsonable(results), indent=2))
    np.savez(SELECTION_NPZ, **{f"{k.replace('|', '__')}__{label}__{dd}": np.asarray(v[label][dd])
                               for k, v in ranks.items() for label in LABELS for dd in DIRECTIONS})
    verdict = (f"STOP: {rule['stop']}" if rule["stop"] else f"PICKED {rule['picked']} (qualifying "
               f"{rule['qualifying']}, tie band {rule['tie_band']})")
    log("D / D_emotion vs C0 (pooled / emotion, mean of directions, R@1 points): "
        + "; ".join(f"{c} {d[c]['point']:+.2f} [{d[c]['ci95'][0]:+.2f}, {d[c]['ci95'][1]:+.2f}] / "
                    f"{d_emotion[c]['point']:+.2f} [{d_emotion[c]['ci95'][0]:+.2f}, {d_emotion[c]['ci95'][1]:+.2f}]"
                    for c in d))
    log(f"Gates all passed: {gates_ok}")
    log(f"Rule outcome: {verdict}")
    log(f"Evaluation in {timings['total']:.1f} s -> {SELECTION_JSON}")
    return results


# ----------------------------------------------------------------------------- Task 4: tables

def _ci(block: dict) -> str:
    lo, hi = block["ci95"]
    return f"{block['point']:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def _r1ci(block: dict) -> str:
    lo, hi = block["ci95"]
    return f"{block['point']:.2f} [{lo:.2f}, {hi:.2f}]"


def _gate_value(name: str, v: dict) -> str:
    return {"participation_ratio": lambda: f"{v['participation_ratio_img']:.1f} / {v['participation_ratio_txt']:.1f}",
            "redundancy": lambda: f"{v['correlation']['max_abs']:.3f}",
            "readout": lambda: f"{v['readout_img']:.4f} / {v['readout_txt']:.4f}",
            "sparsity": lambda: f"{v['active_fraction_img']:.3f} / {v['active_fraction_txt']:.3f}",
            "dead": lambda: f"{len(v['dead_indices'])}",
            "modality_private": lambda: f"{len(v['private_indices'])}",
            "usage_concentration": lambda: f"{v['top2_mass_share']:.3f}",
            "community_spanning": lambda: f"{v['community']['spanning_fraction']:.3f}",
            "pair_retrieval": lambda: f"{v['retrieval_ratio']:.3f}"}[name]()


def tables(path: Path = SELECTION_JSON) -> None:
    res = json.loads(path.read_text())
    out = [f"All numbers: {res['label']}. Episodes: "
           + "; ".join(f"{lab} n={e['n']} targets={e['targets']} sha {e['sha256'][:12]}" for lab, e in
                       res["episodes"].items()) + "\n"]

    out.append("### Training runs (seed 42, 2,000 steps, scorer-train rows)\n")
    out.append("| Cell | wall-clock (min) | peak GPU (GiB) | final loss | final agreement | final condition loss | "
               "final tau |")
    out.append("|---|---:|---:|---:|---:|---:|---:|")
    for c, t in res["training"].items():
        cl = "n/a" if t["final_condition_loss"] is None else f"{t['final_condition_loss']:.4f}"
        tau = "n/a" if t["final_tau"] is None else f"{t['final_tau']:.4f}"
        out.append(f"| {c} | {t['seconds'] / 60:.1f} | {t['peak_gpu_gib']:.2f} | {t['final_loss']:.4f} | "
                   f"{t['final_agreement']:.4f} | {cl} | {tau} |")
    out.append("")

    out.append("### Gates (fit scorer-train, eval selection; amended 2026-09-29 thresholds; img / txt where two)\n")
    out.append("| Gate | " + " | ".join(MODELS) + " |")
    out.append("|---|" + "---|" * len(MODELS))
    for gname in GATE_NAMES:
        cells = [f"{_gate_value(gname, res['gates'][m]['values'])} {'pass' if res['gates'][m]['passed'][gname] else 'FAIL'}"
                 for m in MODELS]
        out.append(f"| {gname} | " + " | ".join(cells) + " |")
    out.append("| **all 9** | " + " | ".join(f"{res['gates'][m]['n_passed']}/9" for m in MODELS) + " |")
    ref = res["settings"]["readout_reference"]
    out.append(f"\nReadout reference (R0 on the same rows): img {ref[0]:.4f}, txt {ref[1]:.4f}.\n")

    out.append("### Naive R@1 (%) at beta 0.3, 95% bootstrap CI\n")
    out.append("| Model | " + " | ".join(f"{s} {d}" for s in SCOPES for d in (*DIRECTIONS, "mean")) + " |")
    out.append("|---|" + "---:|" * (3 * len(SCOPES)))
    for m in (*MODELS, "clip_only"):
        h = res["headline_beta0.3"][m] if m == "clip_only" else res["headline_beta0.3"][m]["naive"]
        out.append(f"| {m} | " + " | ".join(_r1ci(h[s][d]) for s in SCOPES for d in (*DIRECTIONS, "mean")) + " |")
    out.append(f"\nChance {res['chance_r1']:.2f}.\n")

    out.append("### D and D_emotion vs C0 (naive, beta 0.3; paired R@1 points, 95% CI)\n")
    out.append("| Cell | D (pooled mean) | D i2t | D t2i | D_emotion (mean) | art style mean |")
    out.append("|---|---:|---:|---:|---:|---:|")
    for c in ("A", "S", "AS"):
        v = res["vs_c0"][c]
        out.append(f"| {c} | {_ci(res['d'][c])} | {_ci(v['pooled']['i2t'])} | {_ci(v['pooled']['t2i'])} | "
                   f"{_ci(res['d_emotion'][c])} | {_ci(v['art_style']['mean'])} |")
    out.append("")

    out.append("### Context: naive(model, 0.3) - naive(original R3, 0.3) (paired R@1 points)\n")
    out.append("| Model | pooled mean | emotion mean | art style mean |")
    out.append("|---|---:|---:|---:|")
    for m, v in res["vs_r3"].items():
        out.append(f"| {m} | {_ci(v['pooled']['mean'])} | {_ci(v['emotion']['mean'])} | {_ci(v['art_style']['mean'])} |")
    out.append("")

    out.append("### Naive R@1 (%) over the beta grid (mean of directions)\n")
    out.append("| Model | " + " | ".join(f"beta {b:g} pooled / emo / style" for b in BETAS) + " |")
    out.append("|---|" + "---|" * len(BETAS))
    for m in MODELS:
        cells = []
        for b in BETAS:
            r = res["r1"][key("naive", m, b)]
            cells.append(f"{r['pooled']['mean']:.2f} / {r['emotion']['mean']:.2f} / {r['art_style']['mean']:.2f}")
        out.append(f"| {m} | " + " | ".join(cells) + " |")
    out.append("")
    out.append("### Cell - C0 at the same beta (naive; paired R@1 points, 95% CI; context, not gating)\n")
    out.append("| Cell | beta | pooled mean | emotion mean | art style mean |")
    out.append("|---|---:|---:|---:|---:|")
    for c, per_beta in res["vs_c0_beta_grid"].items():
        for b, v in per_beta.items():
            out.append(f"| {c} | {b} | {_ci(v['pooled']['mean'])} | {_ci(v['emotion']['mean'])} | "
                       f"{_ci(v['art_style']['mean'])} |")
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

    out.append("### Label oracle (%, mean of directions, 95% CI) and its null\n")
    out.append("| Model | naive 0.3 pooled | oracle 0.3 pooled | oracle 0 pooled | oracle 0 emotion | oracle 0 style | "
               "null 0 pooled | oracle 0 - own naive 0.3 | oracle 0 - R3 oracle 0 |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for m in MODELS:
        h = res["headline_beta0.3"][m]
        diff_r3 = _ci(res["oracle_minus_r3_oracle"][m]["pooled"]["mean"]) if m != "R3" else "(reference)"
        out.append(f"| {m} | {_r1ci(h['naive']['pooled']['mean'])} | {_r1ci(h['oracle_0.3']['pooled']['mean'])} | "
                   f"{_r1ci(h['oracle_0']['pooled']['mean'])} | {_r1ci(h['oracle_0']['emotion']['mean'])} | "
                   f"{_r1ci(h['oracle_0']['art_style']['mean'])} | {_r1ci(h['oracle_null_0']['pooled']['mean'])} | "
                   f"{_ci(res['oracle_minus_own_naive'][m]['pooled']['mean'])} | {diff_r3} |")
    out.append("")

    rule = res["rule"]
    out.append("### Rule outcome (spec §6)\n")
    out.append(f"- gates all passed: {rule['gates_ok']}")
    out.append(f"- qualifying: {rule['qualifying']}; tie band: {rule['tie_band']}")
    out.append(f"- **{'STOP: ' + rule['stop'] if rule['stop'] else 'picked: ' + rule['picked']}**")
    out.append(f"\nR3 vs probe stored ranks (identical share): {res['probe_reproduction_identical_share']}")
    out.append("Timings (s): " + ", ".join(f"{k} {v:.1f}" for k, v in res["timings_seconds"].items()))
    print("\n".join(out))


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
    elif args.evaluate:
        evaluate()
        tables()
    elif args.tables:
        tables()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
