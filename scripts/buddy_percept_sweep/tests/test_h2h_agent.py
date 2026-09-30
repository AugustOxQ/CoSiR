import itertools
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts.buddy_percept_sweep import h2h_agent
from scripts.buddy_percept_sweep.h2h_trial import resolve_h2h_config

SWEEP_DIR = Path(__file__).resolve().parents[2] / "h2h_sweeps"
CELLS = [("buddy", 16), ("buddy", 40), ("percept", 16), ("percept", 40)]
STAGE2 = {"mapper_lr", "mapper_epochs", "num_queries", "mlp_head", "weight_decay_stage2",
          "class_balanced_loss", "target_cutoff", "train_target_k"}


def _load(system, k):
    return yaml.safe_load((SWEEP_DIR / f"{system}_k{k}.yaml").read_text())


def _first(spec):
    if "value" in spec:
        return spec["value"]
    if "values" in spec:
        return spec["values"][0]
    return spec["min"]


@pytest.mark.parametrize("system,k", CELLS)
def test_yaml_structure(system, k):
    y = _load(system, k)
    assert y["method"] == "bayes"
    assert y["metric"] == {"name": "objective", "goal": "maximize"}
    assert y["run_cap"] == h2h_agent.TRIALS_PER_CELL
    assert y["parameters"]["system"] == {"value": system}
    assert y["parameters"]["k_target"] == {"value": k}


@pytest.mark.parametrize("system,k", CELLS)
def test_yaml_params_resolve(system, k):
    params = _load(system, k)["parameters"]
    cfg = resolve_h2h_config({name: _first(spec) for name, spec in params.items()})
    assert cfg.system == system and cfg.k_target == k
    for name, spec in params.items():          # every listed value resolves too
        for value in spec.get("values", []):
            base = {n: _first(s) for n, s in params.items()}
            base[name] = value
            resolve_h2h_config(base)


def test_stage2_block_identical():
    blocks = [{n: s for n, s in _load(*c)["parameters"].items() if n in STAGE2} for c in CELLS]
    assert set(blocks[0]) == STAGE2
    assert all(b == blocks[0] for b in blocks)


def test_system_specific_params():
    for k in (16, 40):
        assert not any(n.startswith("percept_") for n in _load("buddy", k)["parameters"])
        assert not any(n.startswith("buddy_") for n in _load("percept", k)["parameters"])
    assert _load("buddy", 16)["parameters"]["merge_small_threshold"]["values"] == [0.005, 0.01, 0.02]


def test_buddy_heads_combinations_resolve():
    p = _load("buddy", 16)["parameters"]
    for impl, heads, nh in itertools.product(p["buddy_impl"]["values"], p["buddy_heads"]["values"],
                                             p["buddy_num_heads"]["values"]):
        resolve_h2h_config({"system": "buddy", "k_target": 16, "buddy_impl": impl,
                            "buddy_heads": heads, "buddy_num_heads": nh})


def test_parse_sweep():
    assert h2h_agent.parse_sweep("polysemic/CoSiR-h2h/abc123") == ("polysemic", "CoSiR-h2h", "abc123")
    with pytest.raises(ValueError):
        h2h_agent.parse_sweep("abc123")


class FakeRun:
    def __init__(self):
        self.finished = False

    def finish(self):
        self.finished = True


def _fake_wandb(monkeypatch, config):
    logs, run = [], FakeRun()
    mod = types.SimpleNamespace(init=lambda: run, config=config, log=lambda d: logs.append(d))
    monkeypatch.setitem(sys.modules, "wandb", mod)
    return logs, run


def test_trial_exception_logs_failure(monkeypatch):
    logs, run = _fake_wandb(monkeypatch, {"system": "buddy", "k_target": 16, "bogus": 1})
    h2h_agent._trial()
    assert len(logs) == 1 and logs[0]["objective"] == -1.0 and "bogus" in logs[0]["error"]
    assert run.finished


def test_trial_logs_scalars_only(monkeypatch):
    logs, run = _fake_wandb(monkeypatch, {"system": "buddy", "k_target": 16})
    result = {"per_seed": [{"seed": 1, "k_miss": True, "auc_primary": math.nan, "x": np.float32(0.5),
                            "arr": np.zeros(3), "name": "a"},
                           {"seed": 2, "k_miss": False, "auc_primary": 0.7}],
              "objective": 0.6, "mean": {"auc_primary": 0.7}, "split_digest": "dig"}
    monkeypatch.setattr(h2h_agent, "_get_state", lambda system: h2h_agent.AgentState(
        store=None, split=None, pilot=None, percept_mods=None, device="cpu"))
    monkeypatch.setattr(h2h_agent, "run_h2h_trial", lambda *a, **k: result)
    h2h_agent._trial()
    (row,) = logs
    assert row["objective"] == 0.6 and row["split_digest"] == "dig" and row["auc_primary"] == 0.7
    assert row["s0_k_miss"] == 1 and row["s1_k_miss"] == 0 and row["s0_x"] == 0.5
    assert math.isnan(row["s0_auc_primary"])
    assert not any(k in row for k in ("s0_arr", "s0_name"))
    assert all(isinstance(v, (int, float, str)) for v in row.values())
    assert run.finished
