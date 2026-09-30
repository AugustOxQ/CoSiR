"""One head-to-head trial: a system's Stage 1 -> topics at a matched count ->
both Stage-1 AMI yardsticks and the shared Stage-2 mapper on one held-out
subset, over several seeds. `objective` (mean primary macro AUC over seeds,
-1.0 on any K miss) is what the W&B sweeps maximize.

`resolve_h2h_config` turns a flat wandb.config dict into an `H2HConfig`:
top-level keys (system, k_target, leiden_graph, merge_small_threshold,
leiden_resolution), `buddy_<field>` for every `BuddyStage1Config` field,
`percept_<field>` for every `PerceptStage1Config` field, and the
`Stage2Config` field names unprefixed; values are coerced to the dataclass
field types (wandb may send numbers and bools as strings).
"""
import math
import numbers
import time
import typing
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from scripts.buddy_percept_sweep.clustering import merge_small_communities
from scripts.buddy_percept_sweep.h2h_buddy import (
    HARNESS_HEADS, BuddyStage1Config, fit_buddy_stage1, pilot_constants,
)
from scripts.buddy_percept_sweep.h2h_eval import eval_labels, seed_all, stage1_metrics, stage2_metrics
from scripts.buddy_percept_sweep.h2h_percept import PerceptStage1Config, fit_percept_stage1
from scripts.buddy_percept_sweep.h2h_store import H2HStore
from scripts.buddy_percept_sweep.h2h_topics import (
    GRAPH_KINDS, build_topic_graph, leiden_on_graph, target_k_partition,
)
from scripts.buddy_percept_sweep.h2h_types import H2HSplit

K_TOLERANCE = 2            # spec R3: buddy accepts K within +-2 of k_target
OBJECTIVE_FAIL = -1.0
SYSTEMS = ("buddy", "percept")
BUDDY_IMPLS = ("pilot", "harness")
MLP_HEADS = ("linear", "one_hidden")


@dataclass
class Stage2Config:
    mapper_lr: float = 1e-2
    mapper_epochs: int = 400
    num_queries: int = 1
    mlp_head: str = "linear"
    weight_decay_stage2: float = 0.0
    class_balanced_loss: bool = False
    target_cutoff: Union[str, float] = "single_label"
    train_target_k: int = 20


@dataclass
class H2HConfig:
    system: str                     # "buddy" | "percept"
    k_target: int                   # 16 | 40 ; 0 = fixed-resolution reference mode (buddy only)
    buddy: BuddyStage1Config
    percept: PerceptStage1Config
    stage2: Stage2Config
    leiden_graph: str = "mknn"      # buddy: "mknn" | "pilot_repaired"
    merge_small_threshold: float = 0.0
    leiden_resolution: float = 1.0  # used only when k_target == 0


# ---------------------------------------------------------------- config resolution

_TOP_LEVEL = {"system": str, "k_target": int, "leiden_graph": str,
              "merge_small_threshold": float, "leiden_resolution": float}


def _is_bool(value) -> bool:
    return isinstance(value, (bool, np.bool_))


def _to_bool(key: str, value) -> bool:
    if _is_bool(value):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    raise ValueError(f"{key}: expected a bool or 'true'/'false', got {value!r}")


def _to_float(key: str, value) -> float:
    if isinstance(value, str):
        try:
            result = float(value.strip())
        except ValueError:
            raise ValueError(f"{key}: expected a number, got {value!r}") from None
    elif isinstance(value, numbers.Real) and not _is_bool(value):
        result = float(value)
    else:
        raise ValueError(f"{key}: expected a number, got {value!r}")
    if not math.isfinite(result):
        raise ValueError(f"{key}: expected a finite number, got {value!r}")
    return result


def _to_int(key: str, value) -> int:
    if isinstance(value, numbers.Integral) and not _is_bool(value):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            pass
    as_float = _to_float(key, value)      # "16.0" / 16.0 -> 16; 16.5 -> error
    if not as_float.is_integer():
        raise ValueError(f"{key}: expected an integer, got {value!r}")
    return int(as_float)


def _coerce(key: str, value, typ):
    if typ is bool:
        return _to_bool(key, value)
    if typ is int:
        return _to_int(key, value)
    if typ is float:
        return _to_float(key, value)
    if typ is str:
        if not isinstance(value, str):
            raise ValueError(f"{key}: expected a string, got {value!r}")
        return value
    if typ == Union[str, float]:
        # A numeric string or number becomes a float; any other string stays
        # a string (validated per field, e.g. target_cutoff == "single_label").
        if isinstance(value, str):
            try:
                float(value.strip())
            except ValueError:
                return value
        return _to_float(key, value)
    raise TypeError(f"{key}: unsupported config field type {typ!r}")


def _validate_topics(cfg: H2HConfig) -> None:
    if cfg.system not in SYSTEMS:
        raise ValueError(f"system must be one of {SYSTEMS}, got {cfg.system!r}")
    if _is_bool(cfg.k_target) or not isinstance(cfg.k_target, numbers.Integral) or cfg.k_target < 0:
        raise ValueError(f"k_target must be an integer >= 0, got {cfg.k_target!r}")
    if cfg.k_target == 0 and cfg.system != "buddy":
        raise ValueError("k_target == 0 (fixed-resolution reference mode) is buddy only")
    if cfg.leiden_graph not in GRAPH_KINDS:
        raise ValueError(f"leiden_graph must be one of {GRAPH_KINDS}, got {cfg.leiden_graph!r}")
    if not 0.0 <= cfg.merge_small_threshold < 1.0:
        raise ValueError(f"merge_small_threshold must be in [0, 1), got {cfg.merge_small_threshold!r}")
    if not cfg.leiden_resolution > 0.0:
        raise ValueError(f"leiden_resolution must be > 0, got {cfg.leiden_resolution!r}")


def _validate_buddy(buddy: BuddyStage1Config) -> None:
    if buddy.impl not in BUDDY_IMPLS:
        raise ValueError(f"buddy impl must be one of {BUDDY_IMPLS}, got {buddy.impl!r}")
    if buddy.impl == "pilot":
        pilot_constants(buddy)              # raises for heads the pilot port does not have
    elif buddy.heads not in HARNESS_HEADS:
        raise ValueError(f"harness impl heads must be one of {HARNESS_HEADS}, got {buddy.heads!r}")


def _validate_stage2(s2: Stage2Config) -> None:
    if s2.mlp_head not in MLP_HEADS:
        raise ValueError(f"mlp_head must be one of {MLP_HEADS}, got {s2.mlp_head!r}")
    cutoff = s2.target_cutoff
    if cutoff != "single_label" and (isinstance(cutoff, str) or _is_bool(cutoff)
                                     or not 0.0 <= float(cutoff) <= 1.0):
        raise ValueError(f"target_cutoff must be 'single_label' or a number in [0, 1], got {cutoff!r}")
    for name in ("mapper_epochs", "num_queries", "train_target_k"):
        if getattr(s2, name) < 1:
            raise ValueError(f"{name} must be >= 1, got {getattr(s2, name)!r}")


def validate_h2h_config(cfg: H2HConfig) -> None:
    """Domain checks shared by resolve_h2h_config and run_h2h_trial."""
    _validate_topics(cfg)
    _validate_buddy(cfg.buddy)
    _validate_stage2(cfg.stage2)


def resolve_h2h_config(raw: dict) -> H2HConfig:
    groups = {"buddy": (BuddyStage1Config, "buddy_"), "percept": (PerceptStage1Config, "percept_")}
    hints = {name: typing.get_type_hints(cls) for name, (cls, _) in groups.items()}
    stage2_hints = typing.get_type_hints(Stage2Config)
    top, values = {}, {"buddy": {}, "percept": {}, "stage2": {}}
    for key, value in raw.items():
        if key.startswith("_"):
            continue
        if key in _TOP_LEVEL:
            top[key] = _coerce(key, value, _TOP_LEVEL[key])
            continue
        for name, (_, prefix) in groups.items():
            if key.startswith(prefix) and key[len(prefix):] in hints[name]:
                field_name = key[len(prefix):]
                values[name][field_name] = _coerce(key, value, hints[name][field_name])
                break
        else:
            if key not in stage2_hints:
                raise ValueError(f"unknown config key {key!r}")
            values["stage2"][key] = _coerce(key, value, stage2_hints[key])
    for required in ("system", "k_target"):
        if required not in top:
            raise ValueError(f"missing required config key {required!r}")
    cfg = H2HConfig(**top, buddy=BuddyStage1Config(**values["buddy"]),
                    percept=PerceptStage1Config(**values["percept"]), stage2=Stage2Config(**values["stage2"]))
    validate_h2h_config(cfg)
    return cfg


# ---------------------------------------------------------------- trial

def _plain(value):
    """numpy scalars -> Python scalars, so rows are JSON / wandb friendly."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _buddy_topics(cfg: H2HConfig, train_embedding: np.ndarray, pilot, seed: int, device: str,
                  topic_graph_device: Optional[str] = None):
    """(labels, resolution, k_miss, k_raw, bisect_steps) on the buddy train
    embedding. k_raw is K before merging at the chosen resolution; bisect_steps
    the Leiden runs used (0 in fixed-resolution mode)."""
    graph = build_topic_graph(train_embedding, cfg.leiden_graph, pilot, topic_graph_device or device)
    if cfg.k_target == 0:
        raw = leiden_on_graph(graph, cfg.leiden_resolution, seed)
        labels = merge_small_communities(train_embedding, raw, cfg.merge_small_threshold)[0]
        return labels, cfg.leiden_resolution, False, int(len(np.unique(raw))), 0
    labels, info = target_k_partition(train_embedding, graph, cfg.k_target, K_TOLERANCE,
                                      cfg.merge_small_threshold, seed)
    return labels, info["resolution"], not info["hit"], int(info["k_raw"]), int(info["steps"])


def _run_seed(cfg: H2HConfig, store: H2HStore, subset_idx: np.ndarray, monitor_idx: np.ndarray, seed: int,
              pilot, percept_mods, device: str, topic_graph_device: Optional[str] = None) -> dict:
    """One seed. Every step is seeded from `seed` (global RNGs are reseeded
    before Stage 1 and again before topic formation), so a rerun with the same
    config, seed and inputs gives the same row. stage1_seconds covers Stage 1,
    topics and the Stage-1 metrics; stage2_seconds the mapper (0 when skipped)."""
    seed_all(seed)
    start = time.monotonic()
    if cfg.system == "buddy":
        stage1 = fit_buddy_stage1(cfg.buddy, store, seed, monitor_idx, pilot, device)
        seed_all(seed)
        labels, resolution, k_miss, k_raw, bisect_steps = _buddy_topics(cfg, stage1.train_embedding, pilot,
                                                                         seed, device, topic_graph_device)
        native_subset = None
    else:
        stage1 = fit_percept_stage1(cfg.percept, store, seed, cfg.k_target, percept_mods, device)
        seed_all(seed)
        labels, resolution, k_miss = stage1.train_labels, math.nan, False
        k_raw, bisect_steps = None, 0
        native_subset = np.asarray(stage1.heldout_native)[subset_idx]
    labels = np.asarray(labels, dtype=np.int64)
    n_topics = int(len(np.unique(labels)))

    transfer = eval_labels(stage1.train_embedding, labels, stage1.heldout_embedding, subset_idx)
    metrics1 = stage1_metrics(stage1.train_embedding, labels, stage1.heldout_embedding, subset_idx,
                              store.heldout_emotion, store.heldout_genre, seed, pilot, device,
                              native_subset=native_subset, transfer_labels=transfer)
    stage1_seconds = time.monotonic() - start

    label_sets = {"primary": transfer}
    if native_subset is not None:
        label_sets["native"] = native_subset
    start = time.monotonic()
    if k_miss:
        # Spec R3: a K miss fails the trial; Stage 2 is not trained.
        metrics2 = {f"{kind}_{name}": math.nan for name in label_sets for kind in ("auc", "skipped")}
        stage2_seconds = 0.0
    else:
        metrics2 = stage2_metrics(cfg.stage2, store, stage1.train_embedding, labels, subset_idx, label_sets,
                                  seed, device)
        stage2_seconds = time.monotonic() - start

    row = {"seed": seed, "n_topics": n_topics, "k_miss": bool(k_miss), "resolution": float(resolution),
           "k_raw": k_raw, "bisect_steps": bisect_steps, **metrics1, **metrics2, "stage1_seconds": stage1_seconds, "stage2_seconds": stage2_seconds}
    return {key: _plain(value) for key, value in row.items()}


def _objective(rows: list) -> float:
    """Mean primary AUC over seeds; -1.0 if any seed missed K. A non-finite
    AUC (every topic skipped) also fails, so the sweep never sees NaN."""
    aucs = [row["auc_primary"] for row in rows]
    if any(row["k_miss"] for row in rows) or not all(math.isfinite(auc) for auc in aucs):
        return OBJECTIVE_FAIL
    return float(np.mean(aucs))


def _mean_over_seeds(rows: list) -> dict:
    """Plain mean of every numeric row key except `seed` (bools as 0/1; NaN
    propagates, so a K-miss seed makes the Stage-2 means NaN)."""
    out = {}
    for key in rows[0]:
        values = [row.get(key) for row in rows]
        if key == "seed" or not all(isinstance(v, (bool, int, float)) for v in values):
            continue
        out[key] = float(np.mean([float(v) for v in values]))
    return out


def run_h2h_trial(cfg: H2HConfig, store: H2HStore, split: H2HSplit, subset: str, seeds: tuple,
                  pilot, percept_mods, device: str, monitor: str = "val",
                  topic_graph_device: Optional[str] = None) -> dict:
    """Runs every seed on held-out subset `subset` ("val" | "test"). The buddy
    pilot's plateau monitor watches `split.val_idx` (monitor="val") or all
    held-out rows (monitor="all"). `topic_graph_device` (default None = `device`)
    forces the device of the buddy topic graph (e.g. "cpu" for §6i fidelity).
    The first seed whose row has `k_miss` ends the loop (the objective is
    already -1.0); `aborted_after_k_miss` is True when that skipped seeds.
    Returns {"per_seed": [row, ...], "objective": float, "mean": {...}, "split_digest": str,
    "aborted_after_k_miss": bool}."""
    validate_h2h_config(cfg)
    if subset == "val":
        subset_idx = split.val_idx
    elif subset == "test":
        subset_idx = split.test_idx
    else:
        raise ValueError(f"subset must be 'val' or 'test', got {subset!r}")
    if monitor == "val":
        monitor_idx = split.val_idx
    elif monitor == "all":
        monitor_idx = np.arange(len(store.heldout_paintings))
    else:
        raise ValueError(f"monitor must be 'val' or 'all', got {monitor!r}")
    if len(seeds) == 0:
        raise ValueError("run_h2h_trial needs at least one seed")
    subset_idx = np.asarray(subset_idx, dtype=np.int64)
    monitor_idx = np.asarray(monitor_idx, dtype=np.int64)
    rows = []
    for seed in seeds:
        rows.append(_run_seed(cfg, store, subset_idx, monitor_idx, int(seed), pilot, percept_mods, device,
                              topic_graph_device))
        if rows[-1]["k_miss"]:
            break
    return {"per_seed": rows, "objective": _objective(rows), "mean": _mean_over_seeds(rows),
            "split_digest": split.digest, "aborted_after_k_miss": len(rows) < len(seeds)}
