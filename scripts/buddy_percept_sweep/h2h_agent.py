"""In-process W&B agent for the matched-topic-count head-to-head sweeps.

    python scripts/buddy_percept_sweep/h2h_agent.py --sweep entity/project/sweep_id [--sweep ...] [--count N]
    python scripts/buddy_percept_sweep/h2h_agent.py --sweeps id_a,id_b,id_c,id_d [--count N]

Runs `wandb.agent(..., function=_trial, count=1)` in this process, so the
fixed-input store, the split and the pilot / PercepT modules load once per
process. With several sweeps the agent goes round-robin (one trial per sweep
per pass), so every cell progresses together and no GPU idles when one cell
reaches its `run_cap`; it stops after a full pass that ran no trial, or after
`--count` trials in total. A failing trial logs objective=-1.0 and the error
and never kills the agent.
"""
import argparse
import gc
import numbers
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.buddy_percept_sweep.h2h_trial import OBJECTIVE_FAIL, resolve_h2h_config, run_h2h_trial
from scripts.buddy_percept_sweep.h2h_types import SEARCH_SEEDS

TRIALS_PER_CELL = 300     # `run_cap` in all four scripts/h2h_sweeps/*.yaml; keep them in step


@dataclass
class AgentState:
    store: object
    split: object
    pilot: object
    percept_mods: Optional[object]
    device: str


_STATE: Optional[AgentState] = None
_TRIALS_RUN = 0            # incremented by every _trial call; run_agent reads it to see whether a pass did work


def _get_state(system: str) -> AgentState:
    """Loads the store, split and pilot modules once; PercepT modules lazily,
    on the first trial of a PercepT sweep."""
    global _STATE
    if _STATE is None:
        import torch
        from scripts.buddy_percept_sweep.h2h_split import make_split
        from scripts.buddy_percept_sweep.h2h_store import load_or_build_store
        from scripts.buddy_percept_sweep.pilot_metrics import load_pilot_modules
        store = load_or_build_store()
        split = make_split(store.heldout_emotion, store.heldout_genre)
        _STATE = AgentState(store, split, load_pilot_modules(), None,
                            "cuda" if torch.cuda.is_available() else "cpu")
    if system == "percept" and _STATE.percept_mods is None:
        from scripts.buddy_percept_sweep.h2h_percept import load_percept_modules
        _STATE.percept_mods = load_percept_modules()
    return _STATE


def parse_sweep(text: str) -> tuple:
    parts = text.split("/")
    if len(parts) != 3 or not all(parts):
        raise ValueError(f"--sweep must be entity/project/sweep_id, got {text!r}")
    return tuple(parts)


def parse_sweep_list(items) -> list:
    """Every `--sweep` / `--sweeps` value, each possibly comma-separated, as
    [(entity, project, sweep_id), ...] in the order given."""
    sweeps = [parse_sweep(part.strip()) for item in items for part in item.split(",") if part.strip()]
    if not sweeps:
        raise ValueError("no sweep given; pass --sweep entity/project/sweep_id (repeatable) or --sweeps a,b,c")
    return sweeps


def _scalar(value):
    """A wandb-safe scalar (bool -> int, numpy -> Python), or None to skip."""
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, numbers.Real):
        return value
    return None


def _flatten(result: dict) -> dict:
    out = {"objective": float(result["objective"]), "split_digest": str(result["split_digest"])}
    for key, value in result["mean"].items():
        if (v := _scalar(value)) is not None:
            out[key] = v
    for i, row in enumerate(result["per_seed"]):
        for key, value in row.items():
            if (v := _scalar(value)) is not None:
                out[f"s{i}_{key}"] = v
    return out


def _free_cuda() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            gc.collect()                    # drop dead tensors first so empty_cache can release their blocks
            torch.cuda.empty_cache()
    except Exception:
        pass


def _trial() -> None:
    global _TRIALS_RUN
    _TRIALS_RUN += 1
    import wandb
    run = wandb.init()
    try:
        cfg = resolve_h2h_config(dict(wandb.config))
        state = _get_state(cfg.system)
        result = run_h2h_trial(cfg, state.store, state.split, "val", SEARCH_SEEDS, state.pilot,
                               state.percept_mods, state.device)
        wandb.log(_flatten(result))
    except Exception as exc:               # never kill the agent
        wandb.log({"objective": OBJECTIVE_FAIL, "error": repr(exc)})
    finally:
        try:
            run.finish()
        finally:
            _free_cuda()


def run_agent(sweeps: list, count: Optional[int]) -> int:
    """Round-robin over `sweeps` [(entity, project, sweep_id), ...], one trial
    per sweep per pass. Stops when a full pass ran no trial (every sweep is
    finished) or when `count` trials have run in total. Returns the trial count."""
    import wandb
    start = _TRIALS_RUN
    while True:
        pass_start = _TRIALS_RUN
        for entity, project, sweep_id in sweeps:
            if count is not None and _TRIALS_RUN - start >= count:
                return _TRIALS_RUN - start
            wandb.agent(sweep_id, function=_trial, entity=entity, project=project, count=1)
        if _TRIALS_RUN == pass_start:
            return _TRIALS_RUN - start


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", action="append", default=[],
                        help="entity/project/sweep_id; repeatable, each value may be comma-separated")
    parser.add_argument("--sweeps", action="append", default=[], help="comma-separated entity/project/sweep_id list")
    parser.add_argument("--count", type=int, default=None, help="max trials in total (default: until all sweeps finish)")
    args = parser.parse_args(argv)
    try:
        sweeps = parse_sweep_list(args.sweep + args.sweeps)
    except ValueError as exc:
        parser.error(str(exc))
    run_agent(sweeps, args.count)


if __name__ == "__main__":
    main()
