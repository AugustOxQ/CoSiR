"""Select, stress, test and summarize the matched-topic-count head-to-head.

Subcommands (line-oriented output; arguments never need shell quoting):

  select     query one finished W&B sweep, freeze the top-N (objective > -1)
             to a JSON file
  run        re-run chosen ranks of a frozen finalists file at several seeds on
             the val or test subset; prints H2H_SEED / H2H_RESULT lines
  reference  same for the two fixed reference configs (m8x7ifx4, percept_6g)
  summarize  parse H2H_RESULT lines from job logs into a markdown table

`run` / `reference` call `run_h2h_trial` once per seed, so a K miss on one seed
never hides the others; the summary excludes missed seeds from the AUC stats
and reports how many there were. Heavy imports (torch, real data) and wandb
stay out of module scope.
"""
import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DEFAULT_FINALISTS = "src/test/20260928_buddy_percept_sweep/finalists.json"
REFERENCES = ("m8x7ifx4", "percept_6g")
_BUDDY_STAGE1_KEYS = ("heads", "num_heads", "d_shared", "lr", "noise_std", "lambda_affect", "batch_size",
                      "weight_decay", "teacher_graph_K", "content_pca_dim")
_STAGE2_KEYS = ("mapper_lr", "mapper_epochs", "num_queries", "mlp_head", "target_cutoff", "class_balanced_loss",
                "weight_decay_stage2")


# ---------------------------------------------------------------- pure functions

def accepted_config_keys() -> set:
    """Every key `resolve_h2h_config` accepts."""
    import typing
    from scripts.buddy_percept_sweep.h2h_buddy import BuddyStage1Config
    from scripts.buddy_percept_sweep.h2h_percept import PerceptStage1Config
    from scripts.buddy_percept_sweep.h2h_trial import _TOP_LEVEL, Stage2Config
    keys = set(_TOP_LEVEL) | set(typing.get_type_hints(Stage2Config))
    keys |= {f"buddy_{k}" for k in typing.get_type_hints(BuddyStage1Config)}
    keys |= {f"percept_{k}" for k in typing.get_type_hints(PerceptStage1Config)}
    return keys


def config_for_trial(config: dict) -> dict:
    """A W&B run config minus `_` keys and minus keys the trial config does not take."""
    accepted = accepted_config_keys()
    return {k: v for k, v in config.items() if not k.startswith("_") and k in accepted}


def select_top(runs: list, n: int) -> list:
    kept = [r for r in runs if r.get("objective") is not None and r["objective"] > -1.0]
    kept.sort(key=lambda r: r["objective"], reverse=True)  # stable
    return [{"rank": rank, "id": r["id"], "objective": r["objective"], "config": config_for_trial(r["config"])}
            for rank, r in enumerate(kept[:n], start=1)]


def _finite(values) -> np.ndarray:
    return np.array([v for v in values if v is not None and math.isfinite(v)], dtype=float)


def _mean(values) -> float:
    arr = _finite(values)
    return float(arr.mean()) if len(arr) else math.nan


def summarize_rows(per_seed: list) -> dict:
    """AUC / AMI stats over the seeds that hit K (missed seeds are counted)."""
    ok = [r for r in per_seed if not r["k_miss"]]
    aucs = _finite(r["auc_primary"] for r in ok)
    summary = {
        "n_seeds": len(per_seed),
        "k_miss_count": len(per_seed) - len(ok),
        "auc_mean": float(aucs.mean()) if len(aucs) else math.nan,
        "auc_std": float(aucs.std()) if len(aucs) else math.nan,
        "auc_min": float(aucs.min()) if len(aucs) else math.nan,
        "auc_max": float(aucs.max()) if len(aucs) else math.nan,
        "n_topics": [r["n_topics"] for r in per_seed],
    }
    for name in ("ind_emo", "ind_genre", "transfer_emo", "transfer_genre"):
        summary[f"{name}_mean"] = _mean(r.get(name) for r in ok)
    if any("auc_native" in r for r in per_seed):
        summary["auc_native_mean"] = _mean(r.get("auc_native") for r in ok)
    for name in ("native_emo", "native_genre"):
        if any(name in r for r in per_seed):
            summary[f"{name}_mean"] = _mean(r.get(name) for r in ok)
    return summary


def _jsonable(value):
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.generic,)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def dumps(payload: dict) -> str:
    return json.dumps(_jsonable(payload), separators=(",", ":"), allow_nan=False)


def parse_lines(paths: list, prefix: str) -> list:
    """JSON payloads of every `prefix`-line, deduplicated on (tag, rank, run_id)
    (plus subset when present); the last one wins."""
    by_key: dict = {}
    for path in paths:
        with open(path) as handle:
            for line in handle:
                if line.startswith(prefix):
                    item = json.loads(line[len(prefix):])
                    by_key[(item.get("tag"), item.get("rank"), item.get("run_id"), item.get("subset"))] = item
    return list(by_key.values())


def _fmt(value, spec=".4f") -> str:
    return "n/a" if value is None or (isinstance(value, float) and not math.isfinite(value)) else format(value, spec)


def render_markdown(summaries: list) -> str:
    groups: dict = {}
    for s in summaries:
        groups.setdefault((s.get("tag", ""), s.get("subset", "")), []).append(s)
    header = ["rank", "run id", "sweep objective", "AUC mean ± std", "AUC min", "AUC max", "ind emo", "ind genre",
              "transfer emo", "transfer genre", "AUC native", "topics per seed", "K misses"]
    out = []
    for (tag, subset), items in groups.items():
        items = sorted(items, key=lambda s: -(s["auc_mean"] if s.get("auc_mean") is not None else -math.inf))
        label = f"{tag}, {subset}" if subset else tag
        out += [f"### {label}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        for s in items:
            out.append("| " + " | ".join([
                str(s.get("rank")), str(s.get("run_id")), _fmt(s.get("sweep_objective")),
                f"{_fmt(s.get('auc_mean'))} ± {_fmt(s.get('auc_std'))}", _fmt(s.get("auc_min")),
                _fmt(s.get("auc_max")), _fmt(s.get("ind_emo_mean")), _fmt(s.get("ind_genre_mean")),
                _fmt(s.get("transfer_emo_mean")), _fmt(s.get("transfer_genre_mean")),
                _fmt(s.get("auc_native_mean")), "/".join(str(t) for t in s.get("n_topics", [])),
                str(s.get("k_miss_count", 0)),
            ]) + " |")
        best = items[0]
        if best.get("auc_mean") is None:
            out += ["", f"Winner ({label}): no valid config", ""]
        else:
            out += ["", f"Winner ({label}): rank {best['rank']} ({best['run_id']})", ""]
    return "\n".join(out)


def reference_config(which: str, finalists_path) -> dict:
    """Flat H2H config for a fixed reference."""
    if which == "percept_6g":
        return {"system": "percept", "k_target": 40, "mapper_lr": 1e-2, "mapper_epochs": 400,
                "target_cutoff": "single_label", "num_queries": 1, "mlp_head": "linear"}
    if which != "m8x7ifx4":
        raise ValueError(f"unknown reference {which!r}; expected one of {REFERENCES}")
    finalists = json.loads(Path(finalists_path).read_text())
    src = next((f for f in finalists if f.get("id") == "m8x7ifx4"), None)
    if src is None:
        raise ValueError(f"run m8x7ifx4 not in {finalists_path}")
    c = src["config"]
    flat = {"system": "buddy", "k_target": 0, "buddy_impl": "harness", "leiden_graph": "mknn",
            "leiden_resolution": c["leiden_resolution"], "merge_small_threshold": c["merge_small_threshold"],
            "train_target_k": c["transfer_k"]}
    flat.update({f"buddy_{k}": c[k] for k in _BUDDY_STAGE1_KEYS})
    flat.update({k: c[k] for k in _STAGE2_KEYS})
    return flat


def parse_int_list(text: str, flag: str) -> list:
    try:
        return [int(part) for part in text.split(",")]
    except ValueError:
        sys.exit(f"error: {flag} must be comma-separated integers, got {text!r}")


# ---------------------------------------------------------------- running

@dataclass
class RunContext:
    store: object
    split: object
    pilot: object
    percept_mods: Optional[object]
    device: str


def load_context(need_percept: bool) -> RunContext:
    import torch
    from scripts.buddy_percept_sweep.h2h_split import make_split
    from scripts.buddy_percept_sweep.h2h_store import load_or_build_store
    from scripts.buddy_percept_sweep.pilot_metrics import load_pilot_modules
    store = load_or_build_store()
    mods = None
    if need_percept:
        from scripts.buddy_percept_sweep.h2h_percept import load_percept_modules
        mods = load_percept_modules()
    return RunContext(store, make_split(store.heldout_emotion, store.heldout_genre), load_pilot_modules(), mods,
                      "cuda" if torch.cuda.is_available() else "cpu")


def _scalar_row(row: dict) -> dict:
    return {k: v for k, v in row.items() if v is None or isinstance(v, (bool, int, float, str, np.generic))}


def run_config(raw: dict, ctx: RunContext, tag: str, rank: int, run_id: str, subset: str, seeds: list,
               monitor: str, sweep_objective=None, topic_graph_device: Optional[str] = None) -> dict:
    from scripts.buddy_percept_sweep import h2h_trial
    cfg = h2h_trial.resolve_h2h_config(config_for_trial(raw))
    ident = {"tag": tag, "rank": rank, "run_id": run_id, "subset": subset}
    rows = []
    for seed in seeds:  # one call per seed: a K miss must not abort the others
        result = h2h_trial.run_h2h_trial(cfg, ctx.store, ctx.split, subset, (int(seed),), ctx.pilot,
                                         ctx.percept_mods, ctx.device, monitor=monitor,
                                         topic_graph_device=topic_graph_device)
        row = result["per_seed"][0]
        rows.append(row)
        print("H2H_SEED " + dumps({**ident, **_scalar_row(row)}), flush=True)
    summary = {**ident, "sweep_objective": sweep_objective, **summarize_rows(rows)}
    print("H2H_RESULT " + dumps(summary), flush=True)
    return summary


# ---------------------------------------------------------------- CLI

def cmd_select(args) -> None:
    import wandb  # only this path needs wandb
    parts = args.sweep.split("/")
    if len(parts) != 3:
        sys.exit(f"error: --sweep must be entity/project/sweep_id, got {args.sweep!r}")
    api = wandb.Api(timeout=120)
    runs = api.runs("/".join(parts[:2]), filters={"sweep": parts[2], "summary_metrics.objective": {"$gt": -1}},
                    order="-summary_metrics.objective", per_page=200)
    rows = [{"id": r.id, "objective": r.summary_metrics.get("objective"), "config": dict(r.config)} for r in runs]
    top = select_top(rows, args.top)
    Path(args.out).write_text(json.dumps(top, indent=2) + "\n")
    print(f"{len(rows)} finished runs with objective > -1 -> {len(top)} written to {args.out}")
    for f in top:
        print(f"{f['rank']:>4}  {f['id']:<10}  {f['objective']:.4f}")


def cmd_run(args) -> None:
    finalists = json.loads(Path(args.finalists).read_text())
    by_rank = {f["rank"]: f for f in finalists}
    ranks = parse_int_list(args.ranks, "--ranks")
    missing = [r for r in ranks if r not in by_rank]
    if missing:
        sys.exit(f"error: rank(s) {missing} not in {args.finalists} (available: {sorted(by_rank)})")
    from scripts.buddy_percept_sweep.h2h_types import STRESS_SEEDS, TEST_SEEDS
    seeds = parse_int_list(args.seeds, "--seeds") if args.seeds else list(
        STRESS_SEEDS if args.subset == "val" else TEST_SEEDS)
    cfgs = {r: config_for_trial(by_rank[r]["config"]) for r in ranks}
    ctx = load_context(any(c.get("system") == "percept" for c in cfgs.values()))
    for r in ranks:
        run_config(cfgs[r], ctx, args.tag, r, by_rank[r]["id"], args.subset, seeds, args.monitor,
                   sweep_objective=by_rank[r].get("objective"))


def cmd_reference(args) -> None:
    from scripts.buddy_percept_sweep.h2h_types import TEST_SEEDS
    raw = reference_config(args.which, args.finalists)
    seeds = parse_int_list(args.seeds, "--seeds") if args.seeds else list(TEST_SEEDS)
    ctx = load_context(raw["system"] == "percept")
    # m8x7ifx4's §6i topic graph was built on CPU; keep that for fidelity.
    run_config(raw, ctx, args.tag or f"ref_{args.which}", 0, args.which, args.subset, seeds, args.monitor,
               topic_graph_device="cpu" if args.which == "m8x7ifx4" else None)


def cmd_summarize(args) -> None:
    results = parse_lines(args.inputs, "H2H_RESULT ")
    if not results:
        sys.exit(f"error: no H2H_RESULT lines found in: {', '.join(args.inputs)}")
    markdown = render_markdown(results)
    print(markdown)
    if args.out_md:
        Path(args.out_md).write_text(markdown + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("select", help="freeze a sweep's top-N to JSON (needs wandb)")
    p.add_argument("--sweep", required=True, help="entity/project/sweep_id")
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_select)

    p = sub.add_parser("run", help="re-run finalist ranks at several seeds")
    p.add_argument("--finalists", required=True)
    p.add_argument("--ranks", required=True, help="comma-separated 1-based ranks, e.g. 1,2")
    p.add_argument("--subset", required=True, choices=("val", "test"))
    p.add_argument("--seeds", default=None, help="default: stress seeds for val, test seeds for test")
    p.add_argument("--tag", required=True)
    p.add_argument("--monitor", default="val", choices=("val", "all"))
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("reference", help="run a fixed reference config")
    p.add_argument("--which", required=True, choices=REFERENCES)
    p.add_argument("--subset", default="test", choices=("val", "test"))
    p.add_argument("--seeds", default=None, help="default: test seeds")
    p.add_argument("--tag", default=None)
    p.add_argument("--monitor", default="val", choices=("val", "all"))
    p.add_argument("--finalists", default=DEFAULT_FINALISTS)
    p.set_defaults(func=cmd_reference)

    p = sub.add_parser("summarize", help="markdown table from job logs")
    p.add_argument("--in", dest="inputs", nargs="+", required=True, metavar="LOG")
    p.add_argument("--out-md", default=None)
    p.set_defaults(func=cmd_summarize)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
