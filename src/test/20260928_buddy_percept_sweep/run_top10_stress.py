"""Post-sweep finalist stress test for the buddy-percept W&B sweep.

The sweep (polysemic/CoSiR-buddy-percept-sweep/40i43gt5) ran every trial at
seed 42 only. This tool re-runs the best configs at several seeds to find a
robust winner. Three subcommands:

  select     query the stopped sweep once, filter + rank finalists, freeze the
             result to a JSON file (so every parallel cluster job reads the
             same leaderboard)
  stress     re-run chosen finalist ranks at several seeds (needs the real
             data + a GPU; meant to be launched on DAS6 via
             scripts/run_buddy_percept_top10_stress.sh)
  summarize  parse STRESS_RESULT lines out of stress job logs and print the
             final markdown comparison table + winner

Finalist filter: only runs that passed the gate (objective > -1), had topic
merging on (merge_small_threshold > 0), and ended with at most 45 topics.
Gate-passing runs with more than 100 topics all had merging off; their graphs
fragmented into hundreds of tiny topics, which makes macro AUC noisy and not
comparable to the rest.
"""
import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

GATE_BARS = (0.1236, 0.1954)  # emotion, genre -- informational only
DEFAULT_SEEDS = (42, 7, 123, 2024)


def select_finalists(runs: list[dict], n: int = 10, max_topics: int = 45) -> list[dict]:
    kept = [
        r for r in runs
        if r["objective"] is not None and r["objective"] > -1.0
        and r["n_topics"] is not None and r["n_topics"] <= max_topics
        and float(r["config"]["merge_small_threshold"]) > 0
    ]
    kept.sort(key=lambda r: r["objective"], reverse=True)  # stable, also with reverse=True
    return [{**r, "rank": rank} for rank, r in enumerate(kept[:n], start=1)]


def summarize_seed_results(per_seed: list[dict]) -> dict:
    aucs = np.array([r["auc"] for r in per_seed], dtype=float)
    objectives = np.array([r["objective"] for r in per_seed], dtype=float)
    return {
        "n_seeds": len(per_seed),
        "gate_pass_count": int((objectives > -1.0).sum()),
        "auc_mean": float(aucs.mean()),
        "auc_std": float(aucs.std()),
        "auc_min": float(aucs.min()),
        "auc_max": float(aucs.max()),
        "emo_mean": float(np.mean([r["emo"] for r in per_seed])),
        "genre_mean": float(np.mean([r["genre"] for r in per_seed])),
        "objective_mean": float(objectives.mean()),
        "n_topics": [r["n_topics"] for r in per_seed],
        "per_seed": per_seed,
    }


def rank_finalists(summaries: list[dict]) -> list[dict]:
    return sorted(summaries, key=lambda s: (-s["gate_pass_count"], -s["auc_mean"]))


def parse_stress_results(paths: list[str]) -> list[dict]:
    prefix = "STRESS_RESULT "
    by_key: dict[tuple, dict] = {}
    for path in paths:
        with open(path) as handle:
            for line in handle:
                if line.startswith(prefix):
                    result = json.loads(line[len(prefix):])
                    by_key[(result["rank"], result["run_id"])] = result  # last one wins
    return list(by_key.values())


def format_summary_markdown(ranked: list[dict]) -> str:
    header = [
        "stress rank", "sweep rank", "run id", "sweep objective",
        "seed-42 re-run objective", "gate passes", "AUC mean ± std", "AUC min",
        "AUC max", "emotion AMI mean", "genre AMI mean", "topics per seed",
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for stress_rank, s in enumerate(ranked, start=1):
        seed42 = next((r["objective"] for r in s["per_seed"] if r["seed"] == 42), None)
        cells = [
            str(stress_rank), str(s["rank"]), s["run_id"], f"{s['sweep_objective']:.4f}",
            "n/a" if seed42 is None else f"{seed42:.4f}",
            f"{s['gate_pass_count']}/{s['n_seeds']}",
            f"{s['auc_mean']:.4f} ± {s['auc_std']:.4f}",
            f"{s['auc_min']:.4f}", f"{s['auc_max']:.4f}",
            f"{s['emo_mean']:.4f}", f"{s['genre_mean']:.4f}",
            "/".join(str(t) for t in s["n_topics"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    winner = ranked[0]
    lines += ["", f"Winner: rank {winner['rank']} ({winner['run_id']})"]
    return "\n".join(lines) + "\n"


def _dumps(payload: dict) -> str:
    return json.dumps(payload, separators=(",", ":"))


def _parse_int_list(text: str, flag: str) -> list[int]:
    try:
        values = [int(part) for part in text.split(",")]
    except ValueError:
        sys.exit(f"error: {flag} must be comma-separated integers, got {text!r}")
    return values


def _run_to_row(run) -> dict:
    summary = run.summary_metrics
    return {
        "id": run.id,
        "objective": summary.get("objective"),
        "auc": summary.get("stage2_macro_auc"),
        "emo": summary.get("stage1_emotion_ami"),
        "genre": summary.get("stage1_genre_ami"),
        "n_topics": summary.get("n_topics_after_merge"),
        "config": {k: v for k, v in run.config.items() if not k.startswith("_")},
    }


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def cmd_select(args: argparse.Namespace) -> None:
    import wandb  # only this path needs wandb

    parts = args.sweep.split("/")
    if len(parts) != 3:
        sys.exit(f"error: --sweep must be entity/project/sweep_id, got {args.sweep!r}")
    project_path, sweep_id = "/".join(parts[:2]), parts[2]

    api = wandb.Api(timeout=120)
    runs = api.runs(
        project_path,
        filters={"sweep": sweep_id, "summary_metrics.objective": {"$gt": -1}},
        order="-summary_metrics.objective",
        per_page=200,
    )
    rows = [_run_to_row(run) for run in runs]
    finalists = select_finalists(rows, n=args.n, max_topics=args.max_topics)

    out_path = Path(args.out)
    out_path.write_text(json.dumps(finalists, indent=2) + "\n")

    print(f"{len(rows)} gate-passing runs -> {len(finalists)} finalists written to {out_path}")
    print(f"{'rank':>4}  {'id':<10}  {'objective':>9}  {'n_topics':>8}  {'emo':>7}  {'genre':>7}")
    for f in finalists:
        print(f"{f['rank']:>4}  {f['id']:<10}  {_fmt(f['objective']):>9}  {f['n_topics']:>8}  "
              f"{_fmt(f['emo']):>7}  {_fmt(f['genre']):>7}")


def cmd_stress(args: argparse.Namespace) -> None:
    finalists = json.loads(Path(args.finalists).read_text())
    by_rank = {f["rank"]: f for f in finalists}
    ranks = _parse_int_list(args.ranks, "--ranks")
    seeds = _parse_int_list(args.seeds, "--seeds")
    missing = [rank for rank in ranks if rank not in by_rank]
    if missing:
        sys.exit(f"error: rank(s) {missing} not in {args.finalists} (available: {sorted(by_rank)})")

    # Heavy imports (torch, sklearn, real data) stay out of module scope so
    # `--help`, `select`, `summarize` and the unit tests remain lightweight.
    from scripts.buddy_percept_sweep.cache import FixedInputCache
    from scripts.buddy_percept_sweep.config import resolve_trial_config
    from scripts.buddy_percept_sweep.pipeline import run_trial
    from scripts.buddy_percept_sweep.real_data import load_real_raw_inputs

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    cache = FixedInputCache()  # one per process: finalists share the raw-data load
    for rank in ranks:
        finalist = by_rank[rank]
        config = resolve_trial_config(finalist["config"])
        fixed_inputs = cache.get(content_pca_dim=config.content_pca_dim,
                                 raw_loader=load_real_raw_inputs)
        per_seed = []
        for seed in seeds:
            result = run_trial(dataclasses.replace(config, seed=seed), fixed_inputs)
            row = {
                "seed": seed,
                "objective": result.objective,
                "auc": result.stage2_macro_auc,
                "emo": result.emotion_ami,
                "genre": result.genre_ami,
                "n_topics": result.n_topics_after_merge,
                "stage1_seconds": result.stage1_seconds,
                "stage2_seconds": result.stage2_seconds,
            }
            per_seed.append(row)
            print("STRESS_SEED " + _dumps({"rank": rank, "run_id": finalist["id"], **row}), flush=True)

        summary = {
            **summarize_seed_results(per_seed),
            "rank": rank,
            "run_id": finalist["id"],
            "sweep_objective": finalist["objective"],
            "sweep_n_topics": finalist["n_topics"],
        }
        print("STRESS_RESULT " + _dumps(summary), flush=True)
        if out_dir is not None:
            path = out_dir / f"rank{rank:02d}_{finalist['id']}.json"
            path.write_text(json.dumps(summary, indent=2) + "\n")


def cmd_summarize(args: argparse.Namespace) -> None:
    results = parse_stress_results(args.inputs)
    if not results:
        sys.exit(f"error: no STRESS_RESULT lines found in: {', '.join(args.inputs)}")
    markdown = format_summary_markdown(rank_finalists(results))
    print(markdown, end="")
    if args.out_md:
        Path(args.out_md).write_text(markdown)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_select = sub.add_parser("select", help="query W&B and freeze the finalist list to JSON")
    p_select.add_argument("--sweep", required=True, help="entity/project/sweep_id")
    p_select.add_argument("--n", type=int, default=10)
    p_select.add_argument("--max-topics", type=int, default=45)
    p_select.add_argument("--out", required=True)
    p_select.set_defaults(func=cmd_select)

    p_stress = sub.add_parser("stress", help="re-run finalist ranks at several seeds")
    p_stress.add_argument("--finalists", required=True, help="JSON written by `select`")
    p_stress.add_argument("--ranks", required=True, help="comma-separated 1-based ranks, e.g. 1,10")
    p_stress.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    p_stress.add_argument("--out-dir", default=None)
    p_stress.set_defaults(func=cmd_stress)

    p_sum = sub.add_parser("summarize", help="rank finalists from stress job logs")
    p_sum.add_argument("--in", dest="inputs", nargs="+", required=True, metavar="LOG")
    p_sum.add_argument("--out-md", default=None)
    p_sum.set_defaults(func=cmd_summarize)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
