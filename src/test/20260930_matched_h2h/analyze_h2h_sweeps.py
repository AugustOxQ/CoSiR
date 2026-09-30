"""Controller analysis of the four matched head-to-head sweeps (read-only W&B).

    dump         every run of every cell -> sweep_runs.json (cell, id, state,
                 objective, scalar summary, config)
    pareto       per cell, the Pareto front of val AUC (objective) vs val
                 independent emotion AMI, from sweep_runs.json
    constrained  per cell, finalists (h2h_select JSON format) among finished
                 runs whose val independent emotion AMI >= the floor

The floor for the secondary, emotion-constrained selection is fixed by a
symmetric rule on val data only, before any test run: the largest value F
such that every cell has at least MIN_PER_CELL finished runs with
ind_emo >= F, i.e. min over cells of each cell's MIN_PER_CELL-th largest
ind_emo. (Controller ruling, SDD ledger 2026-09-30.)

Usage:
    python analyze_h2h_sweeps.py dump
    python analyze_h2h_sweeps.py pareto
    python analyze_h2h_sweeps.py constrained --top 5
"""
import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

ENTITY_PROJECT = "polysemic/CoSiR-h2h"
CELLS = {"buddy_k16": "l0hg1hb7", "buddy_k40": "0o1hc9gm",
         "percept_k16": "dzzmonjy", "percept_k40": "7un6c1ak"}
DUMP_PATH = HERE / "sweep_runs.json"
MIN_PER_CELL = 10


def _scalar(value):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return None if isinstance(value, float) and not math.isfinite(value) else value
    if isinstance(value, str):
        return value
    return None


def cmd_dump(_args):
    import wandb
    api = wandb.Api(timeout=120)
    rows = []
    for cell, sweep_id in CELLS.items():
        sweep = api.sweep(f"{ENTITY_PROJECT}/{sweep_id}")
        for run in sweep.runs:
            summary = {k: _scalar(v) for k, v in dict(run.summary).items()
                       if not k.startswith("_") and _scalar(v) is not None}
            rows.append({"cell": cell, "id": run.id, "state": run.state,
                         "created_at": run.created_at,
                         "objective": summary.get("objective"),
                         "summary": summary,
                         "config": {k: v for k, v in dict(run.config).items() if not k.startswith("_")}})
    DUMP_PATH.write_text(json.dumps(rows, indent=1) + "\n")
    counts = {cell: sum(1 for r in rows if r["cell"] == cell and r["state"] == "finished") for cell in CELLS}
    print(f"dumped {len(rows)} runs -> {DUMP_PATH}; finished per cell: {counts}")


def _finished_valid(rows, cell):
    return [r for r in rows if r["cell"] == cell and r["state"] == "finished"
            and r["objective"] is not None and r["objective"] > -1.0
            and r["summary"].get("ind_emo") is not None]


def pareto_front(points):
    """points: [(auc, emo, id)]; maximise both. Returns the non-dominated
    points sorted by emo descending."""
    front, best_auc = [], -math.inf
    for auc, emo, rid in sorted(points, key=lambda p: (-p[1], -p[0])):
        if auc > best_auc:
            front.append((auc, emo, rid))
            best_auc = auc
    return front


def emotion_floor(rows, min_per_cell=MIN_PER_CELL):
    per_cell = []
    for cell in CELLS:
        emos = sorted((r["summary"]["ind_emo"] for r in _finished_valid(rows, cell)), reverse=True)
        if len(emos) < min_per_cell:
            raise SystemExit(f"cell {cell} has only {len(emos)} valid runs (< {min_per_cell})")
        per_cell.append(emos[min_per_cell - 1])
    return min(per_cell)


def cmd_pareto(_args):
    rows = json.loads(DUMP_PATH.read_text())
    for cell in CELLS:
        valid = _finished_valid(rows, cell)
        front = pareto_front([(r["objective"], r["summary"]["ind_emo"], r["id"]) for r in valid])
        print(f"== {cell}: {len(valid)} valid runs; Pareto front (val AUC, val independent emotion AMI):")
        for auc, emo, rid in front:
            print(f"   {rid}  auc={auc:.4f}  ind_emo={emo:.4f}")


def cmd_constrained(args):
    from scripts.buddy_percept_sweep.h2h_select import config_for_trial
    rows = json.loads(DUMP_PATH.read_text())
    floor = emotion_floor(rows)
    print(f"emotion floor (min over cells of each cell's {MIN_PER_CELL}th-largest val ind_emo) = {floor:.4f}")
    for cell in CELLS:
        kept = [r for r in _finished_valid(rows, cell) if r["summary"]["ind_emo"] >= floor]
        kept.sort(key=lambda r: r["objective"], reverse=True)
        top = [{"rank": rank, "id": r["id"], "objective": r["objective"],
                "ind_emo": r["summary"]["ind_emo"], "config": config_for_trial(r["config"])}
               for rank, r in enumerate(kept[:args.top], start=1)]
        out = HERE / f"finalists_constrained_{cell}.json"
        out.write_text(json.dumps(top, indent=2) + "\n")
        print(f"{cell}: {len(kept)} runs above floor -> {len(top)} written to {out.name}")
        for f in top:
            print(f"   {f['rank']}  {f['id']}  auc={f['objective']:.4f}  ind_emo={f['ind_emo']:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("dump")
    sub.add_parser("pareto")
    p_con = sub.add_parser("constrained")
    p_con.add_argument("--top", type=int, default=5)
    args = parser.parse_args()
    {"dump": cmd_dump, "pareto": cmd_pareto, "constrained": cmd_constrained}[args.cmd](args)


if __name__ == "__main__":
    main()
