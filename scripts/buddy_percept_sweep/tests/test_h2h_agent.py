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
    assert _load("buddy", 40)["parameters"]["merge_small_threshold"]["values"] == [0.002, 0.005, 0.01]


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


# ---------------------------------------------------------------- multi-sweep round robin

def test_parse_sweep_list_accepts_repeats_and_commas():
    assert h2h_agent.parse_sweep_list(["a/p/1,a/p/2", "a/p/3"]) == [("a", "p", "1"), ("a", "p", "2"), ("a", "p", "3")]
    with pytest.raises(ValueError):
        h2h_agent.parse_sweep_list(["a/p/1,bad"])
    with pytest.raises(ValueError):
        h2h_agent.parse_sweep_list([])


def test_main_requires_a_sweep_and_merges_both_flags(monkeypatch):
    seen = []
    monkeypatch.setattr(h2h_agent, "run_agent", lambda sweeps, count: seen.append((sweeps, count)))
    h2h_agent.main(["--sweep", "a/p/1", "--sweeps", "a/p/2,a/p/3", "--count", "5"])
    assert seen == [([("a", "p", "1"), ("a", "p", "2"), ("a", "p", "3")], 5)]
    with pytest.raises(SystemExit):
        h2h_agent.main([])


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr(h2h_agent.time, "sleep", lambda s: sleeps.append(s))
    return sleeps


def _round_robin_setup(monkeypatch, budgets):
    """Fake wandb whose agent runs `function` once per call while the sweep's budget remains."""
    logs, agent_calls = [], []
    run = FakeRun()

    def agent(sweep_id, function=None, entity=None, project=None, count=None):
        agent_calls.append((entity, project, sweep_id, count))
        if budgets[sweep_id] > 0:
            budgets[sweep_id] -= 1
            function()

    mod = types.SimpleNamespace(init=lambda: run, config={"system": "buddy", "k_target": 16},
                                log=lambda d: logs.append(d), agent=agent)
    monkeypatch.setitem(sys.modules, "wandb", mod)
    monkeypatch.setattr(h2h_agent, "_get_state", lambda system: h2h_agent.AgentState(
        store=None, split=None, pilot=None, percept_mods=None, device="cpu"))
    monkeypatch.setattr(h2h_agent, "run_h2h_trial", lambda *a, **k: {
        "per_seed": [], "objective": 0.5, "mean": {}, "split_digest": "d"})
    return logs, agent_calls


SWEEPS = [("e", "p", "a"), ("e", "p", "b"), ("e", "p", "c")]


def test_round_robin_visits_sweeps_in_order_and_stops_after_a_zero_trial_pass(monkeypatch):
    budgets = {"a": 2, "b": 0, "c": 1}
    logs, calls = _round_robin_setup(monkeypatch, budgets)
    h2h_agent.run_agent(SWEEPS, None)
    # passes 1-2 run trials (2, then 1); passes 3-5 are empty -> stop
    assert [c[2] for c in calls] == list("abc") * 5
    assert all(c[:2] == ("e", "p") and c[3] == 1 for c in calls)
    assert len(logs) == 3 and budgets == {"a": 0, "b": 0, "c": 0}


def test_round_robin_never_exceeds_count(monkeypatch):
    budgets = {"a": 10, "b": 10, "c": 10}
    logs, calls = _round_robin_setup(monkeypatch, budgets)
    h2h_agent.run_agent(SWEEPS, 4)
    assert len(logs) == 4 and [c[2] for c in calls] == list("abca")
    assert budgets == {"a": 8, "b": 9, "c": 9}


def test_round_robin_counts_failed_trials(monkeypatch):
    budgets = {"a": 1, "b": 1, "c": 0}
    logs, calls = _round_robin_setup(monkeypatch, budgets)
    monkeypatch.setattr(h2h_agent, "run_h2h_trial", lambda *a, **k: 1 / 0)
    h2h_agent.run_agent(SWEEPS, None)
    assert len(logs) == 2 and all(l["objective"] == -1.0 for l in logs)


def test_free_cuda_collects_garbage_before_emptying_the_cache(monkeypatch):
    order = []
    monkeypatch.setattr(h2h_agent.gc, "collect", lambda *a: order.append("gc"))
    cuda = types.SimpleNamespace(is_available=lambda: True, empty_cache=lambda: order.append("empty_cache"))
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda))
    h2h_agent._free_cuda()
    assert order == ["gc", "empty_cache"]


def test_three_empty_passes_stop_with_two_sleeps(monkeypatch, _no_sleep):
    logs, calls = _round_robin_setup(monkeypatch, {"a": 0, "b": 0, "c": 0})
    assert h2h_agent.run_agent(SWEEPS, None) == 0
    assert len(calls) == 9 and _no_sleep == [h2h_agent.EMPTY_PASS_SLEEP_S] * 2   # sleeps between passes 1-2, 2-3


def test_empty_pass_then_resume_continues(monkeypatch, _no_sleep):
    budgets = {"a": 2, "b": 0, "c": 0}
    logs, calls = _round_robin_setup(monkeypatch, budgets)
    real_agent = sys.modules["wandb"].agent

    def flaky(sweep_id, **kw):
        if 3 <= len(calls) < 6:
            calls.append((kw["entity"], kw["project"], sweep_id, kw["count"]))
            return
        return real_agent(sweep_id, **kw)

    sys.modules["wandb"].agent = flaky
    assert h2h_agent.run_agent(SWEEPS, None) == 2
    assert len(logs) == 2 and len(_no_sleep) >= 3      # slept after the empty pass, then resumed


def test_wandb_agent_exception_does_not_crash_loop(monkeypatch, capsys):
    budgets = {"a": 1, "b": 1, "c": 0}
    logs, calls = _round_robin_setup(monkeypatch, budgets)
    real_agent = sys.modules["wandb"].agent
    boom = {"left": 1}

    def agent(sweep_id, **kw):
        if boom["left"]:
            boom["left"] -= 1
            raise RuntimeError("api down")
        return real_agent(sweep_id, **kw)

    sys.modules["wandb"].agent = agent
    assert h2h_agent.run_agent(SWEEPS, None) == 2
    assert "api down" in capsys.readouterr().err and len(logs) == 2
