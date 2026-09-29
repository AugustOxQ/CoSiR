"""Factor-repair grid (R0-R8): pre-registered gates, goal-based selection, seed replication.

Plan Task 6 (docs/superpowers/plans/2026-10-08-cosir-v2-candidate-a-factor-repair.md). Run from
the repository root with the CoSiR environment:

    /root/miniconda3/envs/CoSiR/bin/python src/test/20261011_factor_repair_grid/run_grid.py

1. Painting-grouped split (seed 42). Content graph -> Stage 1 -> communities on the train rows,
   rebuilt here (Task 3's cache is not keyed on split/seed, so it is not loaded).
2. Validation label episodes built ONCE from val rows (2,048 emotion, 2,048 art style, seed 42)
   and shared by every run.
3. The nine pre-registered runs R0-R8 on train rows; val rows encoded in eval mode (8,192-row
   batches); gates on val; condition lift on both label-episode sets.
4. Pre-registered selection: all gates pass (necessary) -> highest selection score -> scores
   within 1.0 R@1 point of the best are tied -> lower mean readout -> earlier run.
5. Seed replication (43, 44) of the selected recipe: gates, scores, Hungarian factor alignment.
6. Gates on held rows for the selected seed-42 model and R0 (the ONLY use of held rows),
   checkpoints, and a reload check that re-encodes 1,000 held rows bit-identically.

``--smoke`` exercises every code path with 3 epochs and 256 episodes, writes only under
``cache/smoke/`` and substitutes val rows for held rows in step 6 (held rows are never encoded
in smoke mode). Its numbers are discarded. ``--tables`` reprints the markdown tables from
``results/summary.json`` without training anything.

``--amended`` is the continuation after the user's AFTER-THE-FACT gate amendment (plan commit d490b96):
readout must be no worse than R0 on the same rows, and at most 50% of factors may be active. It re-gates the
stored grid (no retraining) and applies the unchanged selection rule. Then it retrains R0 and the selected
recipe at seed 42 (deterministic; checked against the stored metrics), replicates with seeds 43/44, and runs
the held check with checkpoints. ``--amended --regate-only`` does only the re-gating;
``--amended --tables`` reprints its tables.
"""

import argparse
import io
import json
import re
import sys
from collections import Counter
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from src.data.artelingo import ANNOTATIONS_PATH, load_artelingo  # noqa: E402
from src.data.splits import grouped_split, leakage_groups, split_leakage  # noqa: E402
from src.eval.factor_gates import FactorGateThresholds, evaluate_factor_gates  # noqa: E402
from src.eval.label_episodes import build_label_episodes, condition_lift  # noqa: E402
from src.model.communities import community_stats, detect_communities  # noqa: E402
from src.model.graph import GraphConfig, build_content_graph  # noqa: E402
from src.train.stage1 import Stage1Config, train_stage1  # noqa: E402
from src.train.train_factors import (  # noqa: E402
    FactorTrainingConfig,
    load_factor_checkpoint,
    save_factor_checkpoint,
    train_factors,
)

SEED = 42
REPLICATION_SEEDS = (43, 44)
EXPECTED_SPLIT = (216_107, 30_872, 61_744)
EPOCHS = 2000
N_EPISODES = 2048
TIE_POINTS = 0.01          # 1.0 R@1 point: scores this close to the best passing score are tied
SANITY_POINTS = 0.02       # 2 R@1 points: gate sanity-check margin
TIME_LIMIT_SECONDS = 45 * 60
RELOAD_ROWS = 1000

BASE = dict(lambda_usage_balance=0.1)                 # plus seed, 32 factors, 2,000 epochs
_R3 = dict(agreement="infonce", lambda_decorrelation=1.0)
_R4 = dict(agreement="infonce", activation="topk", topk=8, lambda_sparsity=0.0)
GRID = {                                              # exactly the pre-registered grid, in table order
    "R0": {},
    "R1": dict(agreement="infonce"),
    "R2": dict(lambda_decorrelation=1.0),
    "R3": dict(_R3),
    "R4": dict(_R4),
    "R5": {**_R4, "lambda_decorrelation": 1.0},
    "R6": {**_R3, "center_inputs": True},
    "R7": {**_R3, "lambda_sparsity": 0.1},
    "R8": dict(lambda_paired=0.0),
}
GATES = ("participation_ratio", "redundancy", "readout", "sparsity", "dead", "modality_private",
         "usage_concentration", "community_spanning", "pair_retrieval")
LABEL_SETS = ("emotion", "art_style")


# ----------------------------------------------------------------------------- helpers

def make_config(overrides: dict, seed: int, epochs: int) -> FactorTrainingConfig:
    config = FactorTrainingConfig(**{**BASE, **overrides, "seed": seed, "epochs": epochs})
    if config.num_factors != 32:
        raise AssertionError("Every run keeps 32 factors")
    return config


def pc1_share(codes: np.ndarray) -> float:
    energy = np.linalg.svd(np.asarray(codes, np.float64) - np.mean(codes, axis=0, dtype=np.float64),
                           compute_uv=False) ** 2
    return float(energy[0] / energy.sum()) if energy.sum() > 0 else 0.0


def encode_rows(model, img: np.ndarray, txt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Frozen model, eval mode, no_grad, 8,192-row batches (same as Task 3)."""
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


def all_finite(*arrays) -> bool:
    return all(bool(np.isfinite(a).all()) for a in arrays)


def load_art_styles(sample_ids: np.ndarray, paintings: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Ruling 12: positional join annotations[int(sample_id)]["art_style"]; one style per painting."""
    with Path(ANNOTATIONS_PATH).open() as file:
        annotations = json.load(file)
    styles = np.asarray([annotations[int(i)]["art_style"] for i in sample_ids])
    if any(not s for s in styles):
        raise AssertionError("Empty art_style")
    for key_name, keys in (("painting", paintings), ("leakage group", groups)):
        _, inverse = np.unique(keys, return_inverse=True)
        _, style_codes = np.unique(styles, return_inverse=True)
        pairs = np.unique(np.stack([inverse, style_codes], axis=1), axis=0)
        if len(pairs) != inverse.max() + 1:
            raise AssertionError(f"Some {key_name} maps to more than one art_style")
    return styles


def build_train_side(train_img: np.ndarray, train_txt: np.ndarray):
    """Graph -> Stage 1 -> communities on the train rows (Task 3's pattern, rebuilt here)."""
    started = perf_counter()
    graph = build_content_graph(train_img, train_txt, GraphConfig())
    with redirect_stdout(io.StringIO()):
        _, embeddings = train_stage1(train_img, train_txt, graph, Stage1Config())
    if not np.isfinite(embeddings).all():
        raise ValueError("Stage 1 produced non-finite embeddings")
    labels = np.asarray(detect_communities(embeddings), dtype=np.int64)
    stats = community_stats(labels)
    meta = {"edges": int(graph.nnz // 2), "communities": int(stats["num_communities"]),
            "setup_seconds": perf_counter() - started}
    return graph, labels, meta


def build_val_episodes(labels_by_set: dict, groups: np.ndarray, val: np.ndarray, n_episodes: int):
    """Ruling 13: paintings = leakage group ids (full array), clean negatives for both label types,
    emotion excludes the catch-all as a target. Built once, shared by every run.

    The same call on val-only arrays must give the same episodes (mapped back through `val`): that
    proves no non-val row influences the episodes (groups are split-disjoint)."""
    episodes, meta = {}, {}
    for name in LABEL_SETS:
        exclude = ("something else",) if name == "emotion" else ()
        kwargs = dict(n_episodes=n_episodes, seed=SEED, exclude_target_labels=exclude,
                      exclude_target_paintings_from_negatives=True)
        full = build_label_episodes(labels_by_set[name], groups, val, **kwargs)
        local = build_label_episodes(labels_by_set[name][val], groups[val], np.arange(len(val)), **kwargs)
        for field in ("anchor", "positive", "supports", "contrasts", "distractors"):
            if not np.array_equal(getattr(full, field), val[getattr(local, field)]):
                raise AssertionError(f"{name} episodes depend on non-val rows ({field})")
        if not np.array_equal(full.labels, local.labels):
            raise AssertionError(f"{name} episode labels depend on non-val rows")
        used = np.concatenate([full.anchor[:, None], full.positive[:, None], full.supports,
                               full.contrasts, full.distractors], axis=1)
        if not np.isin(used, val).all():
            raise AssertionError(f"{name} episodes use non-val rows")
        per_episode_groups = groups[used]
        if any(len(set(row)) != used.shape[1] for row in per_episode_groups.tolist()):
            raise AssertionError(f"{name} episodes repeat a painting group")
        counts = Counter(full.labels.tolist())
        episodes[name] = full
        meta[name] = {"n_episodes": len(full.anchor), "eligible_labels": len(counts),
                      "label_counts": dict(sorted(counts.items())),
                      "excluded_target_labels": list(exclude), "rows_per_episode": used.shape[1]}
    return episodes, meta


def lift_summary(lift: dict) -> dict:
    out = {"lift": lift["lift"], "lift_mean": lift["lift_mean"]}
    for variant in ("naive", "uniform"):
        out[variant] = {d: {k: lift[variant][d][k] for k in ("recall1", "recall3", "tied_episodes")}
                        for d in ("i2t", "t2i")}
    return out


def factor_alignment(reference: np.ndarray, other: np.ndarray) -> dict:
    """Hungarian matching on -|r| between the factors of two val pair-code matrices."""
    a = np.asarray(reference, np.float64) - np.mean(reference, axis=0, dtype=np.float64)
    b = np.asarray(other, np.float64) - np.mean(other, axis=0, dtype=np.float64)
    sa, sb = np.linalg.norm(a, axis=0), np.linalg.norm(b, axis=0)
    denom = np.outer(sa, sb)
    corr = np.divide(a.T @ b, denom, out=np.zeros((a.shape[1], b.shape[1])), where=denom > 0)
    rows, cols = linear_sum_assignment(-np.abs(corr))
    matched = np.abs(corr[rows, cols])
    return {"mean": float(matched.mean()), "median": float(np.median(matched)), "min": float(matched.min()),
            "n_matched_above_0.9": int((matched >= 0.9).sum()), "n_matched_below_0.5": int((matched < 0.5).sum()),
            "matched_abs_r": matched.tolist(), "assignment": cols.tolist(),
            "constant_factors_reference": np.flatnonzero(sa == 0).tolist(),
            "constant_factors_other": np.flatnonzero(sb == 0).tolist()}


# ----------------------------------------------------------------------------- one run

class Context:
    """Everything shared by the runs (data, split, train-side graph/communities, episodes)."""

    def __init__(self, data, split, groups, graph, community_labels, episodes, thresholds, epochs):
        self.data, self.split, self.groups = data, split, groups
        self.graph, self.community_labels = graph, community_labels
        self.episodes, self.thresholds, self.epochs = episodes, thresholds, epochs
        self.train_img = data.img_features[split.train]
        self.train_txt = data.txt_features[split.train]
        self.val_img = data.img_features[split.val]
        self.val_txt = data.txt_features[split.val]
        self.train_groups = groups[split.train]
        self.n_rows = len(data.img_features)


def gate_report(ctx: Context, train_img_codes, train_txt_codes, eval_img_codes, eval_txt_codes,
                eval_img_features, eval_txt_features) -> dict:
    report = evaluate_factor_gates(
        fit_img_codes=train_img_codes, fit_txt_codes=train_txt_codes,
        fit_img_features=ctx.train_img, fit_txt_features=ctx.train_txt,
        eval_img_codes=eval_img_codes, eval_txt_codes=eval_txt_codes,
        eval_img_features=eval_img_features, eval_txt_features=eval_txt_features,
        community_img_codes=train_img_codes, community_txt_codes=train_txt_codes,
        community_labels=ctx.community_labels, thresholds=ctx.thresholds,
    )
    return {"values": report.values, "passed": report.passed, "all_passed": report.all_passed,
            "gates_passed": int(sum(report.passed.values())),
            "pc1_share_img": pc1_share(eval_img_codes), "pc1_share_txt": pc1_share(eval_txt_codes)}


def run_config(run_id: str, overrides: dict, seed: int, ctx: Context):
    """Train one configuration on train rows; gates + condition lift on val rows."""
    config = make_config(overrides, seed, ctx.epochs)
    started = perf_counter()
    log = io.StringIO()
    with redirect_stdout(log):
        # ORIGINAL features always: a center_inputs model stores the train means itself.
        model, train_img_codes, train_txt_codes = train_factors(
            ctx.train_img, ctx.train_txt, ctx.graph, config, group_ids=ctx.train_groups)
    train_seconds = perf_counter() - started
    losses = [float(m) for m in re.findall(r"loss=(nan|-inf|inf|[-+0-9.eE]+)", log.getvalue())]
    val_img_codes, val_txt_codes = encode_rows(model, ctx.val_img, ctx.val_txt)
    finite = all_finite(train_img_codes, train_txt_codes, val_img_codes, val_txt_codes)
    result = {"run": run_id, "seed": seed, "overrides": overrides, "config": asdict(config),
              "finite": finite, "loss_first": losses[0], "loss_last": losses[-1],
              "loss_mean_last100": float(np.mean(losses[-100:])), "loss_all_finite": bool(np.isfinite(losses).all()),
              "n_loss_lines": len(losses), "train_seconds": train_seconds,
              "mean_code_img": float(np.mean(val_img_codes)), "mean_code_txt": float(np.mean(val_txt_codes))}
    if finite:
        result.update(gate_report(ctx, train_img_codes, train_txt_codes, val_img_codes, val_txt_codes,
                                  ctx.val_img, ctx.val_txt))
        result["mean_readout"] = 0.5 * (result["values"]["readout_img"] + result["values"]["readout_txt"])
    else:                                  # reported as a failed run, never silently dropped
        result.update({"values": None, "passed": {g: False for g in GATES}, "all_passed": False,
                       "gates_passed": 0, "mean_readout": float("nan"),
                       "failure": "non-finite codes"})
    # Condition lift needs global row indexing: NaN everywhere except val rows (a guard, not a value).
    full_img = np.full((ctx.n_rows, config.num_factors), np.nan, dtype=np.float32)
    full_txt = np.full((ctx.n_rows, config.num_factors), np.nan, dtype=np.float32)
    full_img[ctx.split.val], full_txt[ctx.split.val] = val_img_codes, val_txt_codes
    lifts = {}
    for name in LABEL_SETS:
        lift = condition_lift(ctx.data.img_features, ctx.data.txt_features, full_img, full_txt,
                              ctx.episodes[name])
        for key in ("i2t", "t2i"):
            if not -1.0 <= lift["lift"][key] <= 1.0:
                raise AssertionError(f"{run_id} seed {seed}: lift outside [-1, 1]: {lift['lift']}")
        lifts[name] = lift_summary(lift)
    result["lifts"] = lifts
    result["selection_score"] = 0.5 * (lifts["emotion"]["lift_mean"] + lifts["art_style"]["lift_mean"])
    result["run_seconds"] = perf_counter() - started
    val_pair_codes = 0.5 * (val_img_codes.astype(np.float64) + val_txt_codes.astype(np.float64))
    return result, model, config, (train_img_codes, train_txt_codes), val_pair_codes


def print_run(result: dict) -> None:
    head = f"[{result['run']} seed {result['seed']}]"
    if result["values"] is None:
        print(f"{head} NON-FINITE codes -> failed run; score {result['selection_score']:.4f}", flush=True)
        return
    v = result["values"]
    print(f"{head} PR {v['participation_ratio_img']:.3f}/{v['participation_ratio_txt']:.3f}"
          f"  max|r| {v['correlation']['max_abs']:.4f}  active {v['active_fraction_img']:.3f}/"
          f"{v['active_fraction_txt']:.3f}  readout {v['readout_img']:.4f}({v['pca10_img']:.4f})/"
          f"{v['readout_txt']:.4f}({v['pca10_txt']:.4f})  dead {len(v['dead_indices'])}"
          f"  private {len(v['private_indices'])}  top2 {v['top2_mass_share']:.3f}"
          f"  span {v['community']['spanning_fraction']:.3f}  ret {v['retrieval_ratio']:.4f}"
          f"  gates {result['gates_passed']}/9  lift emo {result['lifts']['emotion']['lift_mean']:+.4f}"
          f" art {result['lifts']['art_style']['lift_mean']:+.4f}  score {result['selection_score']:+.4f}"
          f"  {result['run_seconds']:.0f}s", flush=True)
    failed = [g for g, ok in result["passed"].items() if not ok]
    print(f"{head} failed gates: {failed or 'none'}", flush=True)


# ----------------------------------------------------------------------------- selection

def select(results: list[dict], smoke: bool) -> dict:
    """Pre-registered rule (Step 4). Never changed by the sanity check."""
    order = [r["run"] for r in results]
    passing = [r for r in results if r["all_passed"]]
    out = {"passing_runs": [r["run"] for r in passing],
           "binding_gates": {r["run"]: [g for g, ok in r["passed"].items() if not ok] for r in results},
           "scores": {r["run"]: r["selection_score"] for r in results}}
    if not passing:
        out.update(selected=None, stop="no run passes all gates on val", sanity_flags=[])
        if smoke:                                       # smoke only: exercise the downstream code path
            finite = [r for r in results if r["finite"]]
            out["smoke_forced_selection"] = max(finite, key=lambda r: r["selection_score"])["run"]
        return out
    best = max(r["selection_score"] for r in passing)
    tied = [r for r in passing if best - r["selection_score"] <= TIE_POINTS + 1e-12]
    chosen = min(tied, key=lambda r: (r["mean_readout"], order.index(r["run"])))
    out.update(best_passing_score=best, tied_within_1_point=[r["run"] for r in tied],
               tie_break_mean_readout={r["run"]: r["mean_readout"] for r in tied},
               selected=chosen["run"], stop=None)
    out["sanity_flags"] = [
        {"run": r["run"], "score": r["selection_score"], "margin_over_best_passing": r["selection_score"] - best,
         "failed_gates": [g for g, ok in r["passed"].items() if not ok]}
        for r in results if not r["all_passed"] and r["selection_score"] > best + SANITY_POINTS]
    return out


# ----------------------------------------------------------------------------- held check

def held_check(label: str, model, config, train_codes, ctx: Context, eval_rows: np.ndarray,
               checkpoint_path: Path) -> dict:
    img, txt = ctx.data.img_features[eval_rows], ctx.data.txt_features[eval_rows]
    img_codes, txt_codes = encode_rows(model, img, txt)
    if not all_finite(img_codes, txt_codes):           # reported as failed, never dropped
        return {"model": label, "finite": False, "values": None, "passed": {g: False for g in GATES},
                "all_passed": False, "gates_passed": 0, "checkpoint": None}
    result = {"model": label, "finite": True, "eval_rows": len(eval_rows),
              **gate_report(ctx, train_codes[0], train_codes[1], img_codes, txt_codes, img, txt)}
    result["mean_readout"] = 0.5 * (result["values"]["readout_img"] + result["values"]["readout_txt"])

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    save_factor_checkpoint(model, config, checkpoint_path)
    device = str(next(model.parameters()).device)
    reloaded, reloaded_config = load_factor_checkpoint(checkpoint_path, device=device)
    rows = np.sort(np.random.default_rng(SEED).choice(len(eval_rows), RELOAD_ROWS, replace=False))
    session = encode_rows(model, img[rows], txt[rows])
    again = encode_rows(reloaded, img[rows], txt[rows])
    result["checkpoint"] = {
        "path": str(checkpoint_path.relative_to(ROOT)), "reload_device": device, "rows": RELOAD_ROWS,
        "config_roundtrip_equal": asdict(reloaded_config) == asdict(config),
        "bit_identical": bool(np.array_equal(session[0], again[0]) and np.array_equal(session[1], again[1])),
        "max_abs_diff": float(max(np.abs(session[0] - again[0]).max(), np.abs(session[1] - again[1]).max())),
        # informational: same rows inside the full 8,192-row-batch encoding
        "max_abs_diff_vs_full_batch_encoding": float(max(np.abs(session[0] - img_codes[rows]).max(),
                                                         np.abs(session[1] - txt_codes[rows]).max())),
    }
    return result


# ----------------------------------------------------------------------------- tables

def _fmt_gate(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def print_tables(summary: dict) -> None:
    runs = summary["grid"]
    print("\n### Gate values (val)")
    print("| run | PR img | PR txt | max abs r (pairs>=.9) | PC1 img/txt | readout img (PCA-10) | readout txt (PCA-10)"
          " | active img | active txt | dead | private | top-2 | spanning | code R@10 | ratio |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        if r["values"] is None:
            print(f"| {r['run']} | non-finite | | | | | | | | | | | | | |")
            continue
        v = r["values"]
        print(f"| {r['run']} | {v['participation_ratio_img']:.3f} | {v['participation_ratio_txt']:.3f}"
              f" | {v['correlation']['max_abs']:.4f} ({v['correlation']['pairs_at_or_above']}/{v['correlation']['pairs_total']})"
              f" | {r['pc1_share_img']:.3f}/{r['pc1_share_txt']:.3f}"
              f" | {v['readout_img']:.4f} ({v['pca10_img']:.4f}) | {v['readout_txt']:.4f} ({v['pca10_txt']:.4f})"
              f" | {v['active_fraction_img']:.4f} | {v['active_fraction_txt']:.4f} | {len(v['dead_indices'])}"
              f" | {len(v['private_indices'])} | {v['top2_mass_share']:.4f} | {v['community']['spanning_fraction']:.3f}"
              f" | {v['code_retrieval_recall']:.4f} | {v['retrieval_ratio']:.4f} |")
    print("\n### Gate pass/fail (val) and selection score")
    print("| run | " + " | ".join(GATES) + " | passed | emotion lift | art-style lift | selection score | mean readout |")
    print("|---|" + "---|" * (len(GATES) + 5))
    for r in runs:
        print(f"| {r['run']} | " + " | ".join(_fmt_gate(r["passed"][g]) for g in GATES)
              + f" | {r['gates_passed']}/9 | {r['lifts']['emotion']['lift_mean']:+.4f}"
              f" | {r['lifts']['art_style']['lift_mean']:+.4f} | {r['selection_score']:+.4f}"
              f" | {r['mean_readout']:.4f} |")
    print("\n### Condition lift detail (val, beta=0): naive / uniform R@1 per direction")
    print("| run | emo i2t naive/unif | emo t2i naive/unif | art i2t naive/unif | art t2i naive/unif |")
    print("|---|---|---|---|---|")
    for r in runs:
        cells = []
        for name in LABEL_SETS:
            lift = r["lifts"][name]
            for d in ("i2t", "t2i"):
                cells.append(f"{lift['naive'][d]['recall1']:.4f} / {lift['uniform'][d]['recall1']:.4f}")
        print(f"| {r['run']} | " + " | ".join(cells) + " |")
    print("\n### Loss / runtime")
    print("| run | loss first | loss last | mean last 100 | mean code img/txt | train s |")
    print("|---|---|---|---|---|---|")
    for r in runs:
        print(f"| {r['run']} | {r['loss_first']:.5f} | {r['loss_last']:.5f} | {r['loss_mean_last100']:.5f}"
              f" | {r['mean_code_img']:.4f}/{r['mean_code_txt']:.4f} | {r['train_seconds']:.0f} |")
    sel = summary.get("selection", {})
    print("\n### Selection")
    print(json.dumps({k: v for k, v in sel.items() if k != "binding_gates"}, indent=1))
    rep = summary.get("replication")
    if rep:
        print("\n### Replication (val)")
        print("| seed | gates | failed | PR img/txt | max abs r | active img/txt | readout img/txt | ratio"
              " | emotion lift | art lift | score | Hungarian mean/median/min |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in rep["runs"]:
            v = r["values"]
            align = rep["alignment"].get(str(r["seed"]))
            align_cell = "reference" if align is None else f"{align['mean']:.3f} / {align['median']:.3f} / {align['min']:.3f}"
            if v is None:
                print(f"| {r['seed']} | 0/9 | non-finite | | | | | | | | {r['selection_score']:+.4f} | {align_cell} |")
                continue
            failed = [g for g, ok in r["passed"].items() if not ok]
            print(f"| {r['seed']} | {r['gates_passed']}/9 | {', '.join(failed) or 'none'}"
                  f" | {v['participation_ratio_img']:.3f}/{v['participation_ratio_txt']:.3f}"
                  f" | {v['correlation']['max_abs']:.4f} | {v['active_fraction_img']:.4f}/{v['active_fraction_txt']:.4f}"
                  f" | {v['readout_img']:.4f}/{v['readout_txt']:.4f} | {v['retrieval_ratio']:.4f}"
                  f" | {r['lifts']['emotion']['lift_mean']:+.4f} | {r['lifts']['art_style']['lift_mean']:+.4f}"
                  f" | {r['selection_score']:+.4f} | {align_cell} |")
        print(json.dumps({k: v for k, v in rep.items() if k not in ("runs", "alignment")}, indent=1))
    held = summary.get("held")
    if held:
        print("\n### Held check (gates on held rows)")
        print("| model | " + " | ".join(GATES) + " | passed | PR img/txt | max abs r | active img/txt"
              " | readout img (PCA-10) | readout txt (PCA-10) | ratio | reload bit-identical |")
        print("|---|" + "---|" * (len(GATES) + 8))
        for h in held["models"]:
            v = h["values"]
            if v is None:
                print(f"| {h['model']} | non-finite codes on held |")
                continue
            print(f"| {h['model']} | " + " | ".join(_fmt_gate(h["passed"][g]) for g in GATES)
                  + f" | {h['gates_passed']}/9 | {v['participation_ratio_img']:.3f}/{v['participation_ratio_txt']:.3f}"
                  f" | {v['correlation']['max_abs']:.4f} | {v['active_fraction_img']:.4f}/{v['active_fraction_txt']:.4f}"
                  f" | {v['readout_img']:.4f} ({v['pca10_img']:.4f}) | {v['readout_txt']:.4f} ({v['pca10_txt']:.4f})"
                  f" | {v['retrieval_ratio']:.4f} | {h['checkpoint']['bit_identical']} |")


# ----------------------------------------------------------------------------- main

# ----------------------------------------------------------------------------- amended gates

# User decision 2026-09-29, AFTER the Step 4.4 stop (plan amendment, commit d490b96). This is an
# after-the-fact threshold change. Only two gates change; everything else is the pre-registered rule.
#   readout:  pass iff readout <= R0's readout in each modality, on the same rows
#             (val: R0 seed-42 val readout; held: R0's held readout)
#   sparsity: max active fraction <= 0.50 (was 0.375)
AMENDED_MAX_ACTIVE = 0.5
ORIGINAL_THRESHOLDS = FactorGateThresholds()
AMENDED_THRESHOLDS = FactorGateThresholds(max_active_fraction=AMENDED_MAX_ACTIVE)
READOUT_KEYS = ("readout_img", "readout_txt")
REPRO_KEYS = ("participation_ratio_img", "participation_ratio_txt", "active_fraction_img", "active_fraction_txt",
              "readout_img", "readout_txt", "pca10_img", "pca10_txt", "top2_mass_share",
              "code_retrieval_recall", "clip_retrieval_recall", "retrieval_ratio", "dead_indices", "private_indices")


def max_active(values: dict) -> float:
    return max(values["active_fraction_img"], values["active_fraction_txt"])


def both_flag_sets(values: dict, returned: dict, reference: dict, thresholds_used: FactorGateThresholds):
    """Original (pre-registered) and amended flags from one set of gate values.

    Every flag except readout and sparsity is evaluate_factor_gates' own. The flags it returned are
    checked against a re-derivation from the values under the thresholds it was called with."""
    rank = ORIGINAL_THRESHOLDS.readout_pca_rank
    pca_readout = bool(values["readout_img"] <= values[f"pca{rank}_img"]
                       and values["readout_txt"] <= values[f"pca{rank}_txt"])
    if returned["readout"] != pca_readout:
        raise AssertionError("Returned readout flag disagrees with the PCA-10 re-derivation")
    if returned["sparsity"] != (max_active(values) <= thresholds_used.max_active_fraction):
        raise AssertionError("Returned sparsity flag disagrees with the thresholds it was computed under")
    original = {**returned, "readout": pca_readout,
                "sparsity": bool(max_active(values) <= ORIGINAL_THRESHOLDS.max_active_fraction)}
    amended = {**returned,
               "readout": bool(all(values[k] <= reference[k] for k in READOUT_KEYS)),
               "sparsity": bool(max_active(values) <= AMENDED_MAX_ACTIVE)}
    return original, amended


def regate(result: dict, reference: dict, thresholds_used: FactorGateThresholds) -> dict:
    """Copy of a result whose `passed` / `all_passed` are the AMENDED flags; originals kept alongside."""
    out = dict(result)
    if result.get("values") is None:                        # non-finite run: failed under both rules
        flags = {g: False for g in GATES}
        original, amended = flags, dict(flags)
    else:
        original, amended = both_flag_sets(result["values"], result["passed"], reference, thresholds_used)
    out.update(original_passed=original, original_all_passed=all(original.values()),
               original_gates_passed=int(sum(original.values())),
               passed=amended, all_passed=all(amended.values()), gates_passed=int(sum(amended.values())),
               readout_reference=reference)
    return out


def compare_to_stored(result: dict, stored: dict) -> dict:
    """Exact comparison of a deterministic retrain against the stored grid result."""
    pairs = {f"values.{k}": (result["values"][k], stored["values"][k]) for k in REPRO_KEYS}
    pairs["values.correlation.max_abs"] = (result["values"]["correlation"]["max_abs"],
                                           stored["values"]["correlation"]["max_abs"])
    pairs["values.community.spanning_fraction"] = (result["values"]["community"]["spanning_fraction"],
                                                   stored["values"]["community"]["spanning_fraction"])
    for key in ("selection_score", "pc1_share_img", "pc1_share_txt", "loss_first", "loss_last", "mean_code_img"):
        pairs[key] = (result[key], stored[key])
    for name in LABEL_SETS:
        pairs[f"lifts.{name}.lift_mean"] = (result["lifts"][name]["lift_mean"], stored["lifts"][name]["lift_mean"])
    mismatches = {k: {"retrained": a, "stored": b} for k, (a, b) in pairs.items() if a != b}
    return {"identical": not mismatches, "n_compared": len(pairs), "mismatches": mismatches}


def print_amended_tables(amended: dict) -> None:
    print("\n### Re-gated grid (val): original -> amended flags")
    print("| run | readout orig -> amended | sparsity orig -> amended | other failing gates | original passed"
          " | amended passed | selection score | mean readout |")
    print("|---|---|---|---|---|---|---|---|")
    for r in amended["regated"]:
        other = [g for g in GATES if g not in ("readout", "sparsity") and not r["passed"][g]]
        print(f"| {r['run']} | {_fmt_gate(r['original_passed']['readout'])} -> {_fmt_gate(r['passed']['readout'])}"
              f" | {_fmt_gate(r['original_passed']['sparsity'])} -> {_fmt_gate(r['passed']['sparsity'])}"
              f" | {', '.join(other) or 'none'} | {r['original_gates_passed']}/9 | {r['gates_passed']}/9"
              f" | {r['selection_score']:+.4f} | {r['mean_readout']:.4f} |")
    print("\n### Amended flags, all gates (val)")
    print("| run | " + " | ".join(GATES) + " | passed |")
    print("|---|" + "---|" * (len(GATES) + 1))
    for r in amended["regated"]:
        print(f"| {r['run']} | " + " | ".join(_fmt_gate(r["passed"][g]) for g in GATES) + f" | {r['gates_passed']}/9 |")
    print("\n### Amended selection")
    print(json.dumps({k: v for k, v in amended["selection"].items() if k != "binding_gates"}, indent=1))
    for label, repro in amended.get("reproduction", {}).items():
        print(f"Deterministic retrain {label}: identical={repro['identical']} over {repro['n_compared']} values"
              f" {repro['mismatches'] or ''}")
    rep = amended.get("replication")
    if rep:
        print("\n### Replication (val, amended gates; readout reference = R0 seed-42 val)")
        print("| seed | amended gates | failed (amended) | original gates | PR img/txt | max abs r | active img/txt"
              " | readout img/txt | ratio | emotion lift | art lift | score | Hungarian mean/median/min |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in rep["runs"]:
            v = r["values"]
            align = r.get("alignment_to_seed42")
            align_cell = "reference" if align is None else f"{align['mean']:.4f} / {align['median']:.4f} / {align['min']:.4f}"
            failed = [g for g in GATES if not r["passed"][g]]
            print(f"| {r['seed']} | {r['gates_passed']}/9 | {', '.join(failed) or 'none'} | {r['original_gates_passed']}/9"
                  f" | {v['participation_ratio_img']:.3f}/{v['participation_ratio_txt']:.3f}"
                  f" | {v['correlation']['max_abs']:.4f} | {v['active_fraction_img']:.4f}/{v['active_fraction_txt']:.4f}"
                  f" | {v['readout_img']:.4f}/{v['readout_txt']:.4f} | {v['retrieval_ratio']:.4f}"
                  f" | {r['lifts']['emotion']['lift_mean']:+.4f} | {r['lifts']['art_style']['lift_mean']:+.4f}"
                  f" | {r['selection_score']:+.4f} | {align_cell} |")
        print(json.dumps({k: v for k, v in rep.items() if k != "runs"}, indent=1))
    held = amended.get("held")
    if held:
        print(f"\n### Held check ({held['eval_part']} rows; readout reference = R0's held readout)")
        print("| model | " + " | ".join(GATES) + " | amended passed | original passed | PR img/txt | max abs r"
              " | active img/txt | readout img (PCA-10) | readout txt (PCA-10) | ratio | reload bit-identical |")
        print("|---|" + "---|" * (len(GATES) + 9))
        for h in held["models"]:
            v = h["values"]
            print(f"| {h['model']} | " + " | ".join(_fmt_gate(h["passed"][g]) for g in GATES)
                  + f" | {h['gates_passed']}/9 | {h['original_gates_passed']}/9"
                  f" | {v['participation_ratio_img']:.3f}/{v['participation_ratio_txt']:.3f}"
                  f" | {v['correlation']['max_abs']:.4f} | {v['active_fraction_img']:.4f}/{v['active_fraction_txt']:.4f}"
                  f" | {v['readout_img']:.4f} ({v['pca10_img']:.4f}) | {v['readout_txt']:.4f} ({v['pca10_txt']:.4f})"
                  f" | {v['retrieval_ratio']:.4f} | {h['checkpoint']['bit_identical']} |")
            print(f"    checkpoint: {h['checkpoint']}")


def run_amended(args, results_dir: Path, checkpoint_dir: Path, epochs: int, n_episodes: int) -> None:
    """Continuation after the user's amendment: re-gate stored runs, select, replicate, held check."""
    started = perf_counter()
    amended_path = results_dir / "amended_summary.json"
    grid_summary = json.loads((results_dir / "summary.json").read_text())
    stored = grid_summary["grid"]
    if [r["run"] for r in stored] != list(GRID):
        raise AssertionError("Stored grid does not match the pre-registered R0-R8")
    r0_val = next(r for r in stored if r["run"] == "R0")
    val_reference = {k: r0_val["values"][k] for k in READOUT_KEYS}      # exact stored values

    # Step A: re-gate the nine stored runs (no retraining); pre-registered selection on amended flags.
    regated = [regate(r, val_reference, ORIGINAL_THRESHOLDS) for r in stored]
    for r, s in zip(regated, stored):
        if r["original_passed"] != s["passed"]:
            raise AssertionError(f"{r['run']}: re-derived original flags differ from the stored ones")
    selection = select(regated, args.smoke)
    amended = {"mode": grid_summary["mode"], "amendment": {
                   "decision": "user, 2026-09-29, after the Step 4.4 stop; plan commit d490b96; after the fact",
                   "readout": "readout <= R0 readout per modality on the same rows",
                   "max_active_fraction": AMENDED_MAX_ACTIVE, "original_max_active_fraction":
                   ORIGINAL_THRESHOLDS.max_active_fraction},
               "readout_reference_val": val_reference,
               "regated": [{k: r[k] for k in ("run", "passed", "all_passed", "gates_passed", "original_passed",
                                              "original_all_passed", "original_gates_passed", "selection_score",
                                              "mean_readout")} for r in regated],
               "selection": selection}

    def write() -> None:
        amended["runtime_seconds"] = perf_counter() - started
        amended_path.write_text(json.dumps(amended, indent=1))

    write()
    selected = selection["selected"] or selection.get("smoke_forced_selection")
    print(f"Amended selection: {json.dumps({k: v for k, v in selection.items() if k != 'binding_gates'})}", flush=True)
    if selected is None or args.regate_only:
        print_amended_tables(amended)
        if selected is None:
            print("STOP: no run passes all amended gates on val.", flush=True)
        return

    ctx, meta = setup_context(epochs, n_episodes, AMENDED_THRESHOLDS)
    if meta["episodes"] != grid_summary["episodes"] or list(meta["split_sizes"]) != list(grid_summary["split_sizes"]):
        raise AssertionError("Val episodes or split differ from the stored grid run")
    amended.update(torch=meta["torch"], device=meta["device"], setup=meta["setup"])

    # Deterministic seed-42 retrain of R0 and the selected recipe (models for the held check,
    # seed-42 val pair codes for the Hungarian alignment); must reproduce the stored metrics exactly.
    models, configs, train_codes, pair_codes = {}, {}, {}, {}
    amended["reproduction"] = {}
    for run_id in dict.fromkeys(("R0", selected)):
        print(f"[{run_id} seed {SEED}] deterministic retrain", flush=True)
        result, model, config, codes, pairs = run_config(run_id, GRID[run_id], SEED, ctx)
        stored_run = next(r for r in stored if r["run"] == run_id)
        amended["reproduction"][f"{run_id}_seed{SEED}"] = compare_to_stored(result, stored_run)
        models[run_id], configs[run_id], train_codes[run_id], pair_codes[run_id] = model, config, codes, pairs
        print(f"[{run_id} seed {SEED}] reproduction {amended['reproduction'][f'{run_id}_seed{SEED}']}", flush=True)
        write()
    np.save(results_dir / f"{selected}_seed{SEED}_val_pair_codes.npy", pair_codes[selected])

    # Step 5: replication with seeds 43 and 44 under the amended gates (reference: R0 seed-42 val).
    selected_regated = next(r for r in regated if r["run"] == selected)
    replication = {"recipe": selected, "overrides": GRID[selected], "runs": [{**selected_regated, "seed": SEED}]}
    for seed in REPLICATION_SEEDS:
        print(f"[{selected} seed {seed}] replication", flush=True)
        result, _, _, _, pairs = run_config(selected, GRID[selected], seed, ctx)
        np.save(results_dir / f"{selected}_seed{seed}_val_pair_codes.npy", pairs)
        result = regate(result, val_reference, AMENDED_THRESHOLDS)
        result["alignment_to_seed42"] = factor_alignment(pair_codes[selected], pairs) if result["finite"] else None
        (results_dir / f"{selected}_seed{seed}.json").write_text(json.dumps(result, indent=1))
        replication["runs"].append(result)
        print_run(result)
    rep_runs = replication["runs"][1:]
    r0_score = r0_val["selection_score"]
    replication.update(all_seeds_pass=all(r["all_passed"] for r in rep_runs),
                       seeds_at_or_below_R0=[r["seed"] for r in rep_runs if r["selection_score"] <= r0_score],
                       scores={str(r["seed"]): r["selection_score"] for r in replication["runs"]})
    amended["replication"] = replication
    write()
    if not replication["all_seeds_pass"] and not args.smoke:
        amended["stop"] = "a replication seed fails an amended gate on val: recipe not selected (Step 5)"
        write()
        print("STOP: a replication seed fails an amended gate on val. Held rows untouched.", flush=True)
        print_amended_tables(amended)
        return

    # Step 6: held check, once — the first time held rows are encoded. R0 first: its held readout
    # is the readout reference for the selected model. Smoke mode substitutes val rows.
    eval_rows = ctx.split.val if args.smoke else ctx.split.held
    held = {"eval_part": "val (smoke stand-in)" if args.smoke else "held", "models": []}
    r0_held = held_check("R0", models["R0"], configs["R0"], train_codes["R0"], ctx, eval_rows,
                         checkpoint_dir / "R0_seed42.pt")
    held_reference = {k: r0_held["values"][k] for k in READOUT_KEYS}
    held["readout_reference"] = held_reference
    held["models"].append({**regate(r0_held, held_reference, AMENDED_THRESHOLDS),
                           "note": "R0 is the readout reference itself (readout flag trivially PASS)"})
    selected_held = held_check("selected " + selected, models[selected], configs[selected], train_codes[selected],
                               ctx, eval_rows, checkpoint_dir / "selected_seed42.pt")
    held["models"].insert(0, regate(selected_held, held_reference, AMENDED_THRESHOLDS))
    amended["held"] = held
    amended["selected_config"] = asdict(configs[selected])
    write()
    print(f"Selected recipe config: {json.dumps(amended['selected_config'])}", flush=True)
    print_amended_tables(amended)
    print(f"Total runtime {perf_counter() - started:.1f}s", flush=True)


def setup_context(epochs: int, n_episodes: int, thresholds: FactorGateThresholds):
    """Data, painting-grouped split, art styles, train-side graph/communities, shared val episodes."""
    data = load_artelingo()
    groups = leakage_groups(data.paintings, data.img_features)
    split = grouped_split(groups, fractions=(0.7, 0.1, 0.2), seed=SEED)
    sizes = (len(split.train), len(split.val), len(split.held))
    if sizes != EXPECTED_SPLIT:
        raise AssertionError(f"Unexpected split sizes {sizes}")
    leak = split_leakage(split, data.paintings, data.img_features)
    if any(leak.values()):
        raise AssertionError(f"Split leakage: {leak}")
    print(f"Split train/val/held = {sizes}, leakage zero", flush=True)
    styles = load_art_styles(data.sample_ids, data.paintings, groups)
    print(f"art_style joined: {len(np.unique(styles))} styles, one per painting and per leakage group", flush=True)
    graph, community_labels, setup = build_train_side(data.img_features[split.train], data.txt_features[split.train])
    print(f"Train-side setup: {setup['edges']:,} edges, {setup['communities']} communities, "
          f"{setup['setup_seconds']:.1f}s", flush=True)
    episodes, episode_meta = build_val_episodes({"emotion": data.emotions, "art_style": styles},
                                                groups, split.val, n_episodes)
    for name, meta in episode_meta.items():
        print(f"{name} episodes: {meta['n_episodes']} over {meta['eligible_labels']} target labels "
              f"{meta['label_counts']}", flush=True)
    ctx = Context(data, split, groups, graph, community_labels, episodes, thresholds, epochs)
    meta = {"torch": torch.__version__,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "split_sizes": sizes, "leakage": leak, "setup": setup, "episodes": episode_meta}
    return ctx, meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--tables", action="store_true")
    parser.add_argument("--amended", action="store_true",
                        help="continuation after the user's after-the-fact gate amendment (Steps A, 5, 6)")
    parser.add_argument("--regate-only", action="store_true", help="with --amended: Step A only, no training")
    args = parser.parse_args()
    out_dir = HERE / "cache" / "smoke" if args.smoke else HERE
    results_dir, checkpoint_dir = out_dir / "results", out_dir / "checkpoints"
    summary_path = results_dir / "summary.json"
    if args.tables:
        if args.amended:
            print_amended_tables(json.loads((results_dir / "amended_summary.json").read_text()))
        else:
            print_tables(json.loads(summary_path.read_text()))
        return
    epochs, n_episodes = (3, 256) if args.smoke else (EPOCHS, N_EPISODES)
    if args.amended:
        run_amended(args, results_dir, checkpoint_dir, epochs, n_episodes)
        return
    results_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()

    def elapsed_guard(stage: str) -> None:
        if perf_counter() - started > TIME_LIMIT_SECONDS:
            raise SystemExit(f"Stopped after {stage}: runtime over {TIME_LIMIT_SECONDS // 60} minutes")

    def write_summary() -> None:
        summary["runtime_seconds"] = perf_counter() - started
        summary_path.write_text(json.dumps(summary, indent=1))

    print(f"Mode: {'SMOKE (numbers discarded)' if args.smoke else 'REAL'}; epochs {epochs}; "
          f"episodes {n_episodes} per label type", flush=True)
    thresholds = FactorGateThresholds()
    ctx, meta = setup_context(epochs, n_episodes, thresholds)
    split = ctx.split
    summary = {"mode": "smoke" if args.smoke else "real", **meta, "thresholds": asdict(thresholds),
               "base": BASE, "grid_overrides": GRID, "grid": []}

    # Steps 1-3: the grid (seed 42), gates + condition lift on val.
    models, train_codes, pair_codes, configs = {}, {}, {}, {}
    for run_id, overrides in GRID.items():
        print(f"[{run_id}] overrides={overrides}", flush=True)
        result, model, config, codes, pairs = run_config(run_id, overrides, SEED, ctx)
        summary["grid"].append(result)
        models[run_id], train_codes[run_id], pair_codes[run_id], configs[run_id] = model, codes, pairs, config
        (results_dir / f"{run_id}_seed{SEED}.json").write_text(json.dumps(result, indent=1))
        write_summary()
        print_run(result)
        elapsed_guard(run_id)

    # Step 4: pre-registered selection.
    selection = select(summary["grid"], args.smoke)
    summary["selection"] = selection
    write_summary()
    print(f"Selection: {json.dumps({k: v for k, v in selection.items() if k != 'binding_gates'})}", flush=True)
    selected = selection["selected"] or selection.get("smoke_forced_selection")
    if selected is None:
        print("STOP: no run passes all gates on val (Step 4.4). Held rows untouched.", flush=True)
        print_tables(summary)
        return

    # Step 5: seed replication of the selected recipe.
    replication = {"recipe": selected, "overrides": GRID[selected], "runs": [], "alignment": {}}
    reference = next(r for r in summary["grid"] if r["run"] == selected)
    replication["runs"].append({**reference, "seed": SEED})
    np.save(results_dir / f"{selected}_seed{SEED}_val_pair_codes.npy", pair_codes[selected])
    for seed in REPLICATION_SEEDS:
        print(f"[{selected} seed {seed}] replication", flush=True)
        result, _, _, _, pairs = run_config(selected, GRID[selected], seed, ctx)
        np.save(results_dir / f"{selected}_seed{seed}_val_pair_codes.npy", pairs)
        (results_dir / f"{selected}_seed{seed}.json").write_text(json.dumps(result, indent=1))
        result["alignment_to_seed42"] = factor_alignment(pair_codes[selected], pairs) if result["finite"] else None
        replication["runs"].append(result)
        if result["finite"]:
            replication["alignment"][str(seed)] = result["alignment_to_seed42"]
        print_run(result)
        elapsed_guard(f"replication seed {seed}")
    r0_score = next(r for r in summary["grid"] if r["run"] == "R0")["selection_score"]
    rep_runs = replication["runs"][1:]
    replication["all_seeds_pass"] = all(r["all_passed"] for r in rep_runs)
    replication["seeds_at_or_below_R0"] = [r["seed"] for r in rep_runs if r["selection_score"] <= r0_score]
    replication["scores"] = {str(r["seed"]): r["selection_score"] for r in replication["runs"]}
    summary["replication"] = replication
    write_summary()
    if not replication["all_seeds_pass"] and not args.smoke:
        summary["selection"]["stop"] = "a replication seed fails a gate on val: recipe not selected (Step 5)"
        write_summary()
        print("STOP: a replication seed fails a gate on val; the recipe is NOT selected. Held rows untouched.",
              flush=True)
        print_tables(summary)
        return

    # Step 6: final held check, once (the only use of held rows). Smoke mode substitutes val rows.
    eval_rows = split.val if args.smoke else split.held
    held = {"eval_part": "val (smoke stand-in)" if args.smoke else "held", "models": []}
    for label, run_id, name in (("selected " + selected, selected, "selected_seed42.pt"),
                                ("R0", "R0", "R0_seed42.pt")):
        print(f"[held] {label}", flush=True)
        check = held_check(label, models[run_id], configs[run_id], train_codes[run_id], ctx, eval_rows,
                           checkpoint_dir / name)
        held["models"].append(check)
        print(f"[held] {label}: gates {check.get('gates_passed')}/9 all_passed {check['all_passed']} "
              f"reload {check.get('checkpoint')}", flush=True)
    summary["held"] = held
    summary["selected_config"] = asdict(configs[selected])
    write_summary()
    print(f"Selected recipe config: {json.dumps(asdict(configs[selected]))}", flush=True)
    print_tables(summary)
    print(f"Total runtime {perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
