import dataclasses
import json
import math
import typing
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.buddy_percept_sweep import h2h_trial
from scripts.buddy_percept_sweep.h2h_buddy import BuddyStage1Config
from scripts.buddy_percept_sweep.h2h_percept import PerceptStage1Config
from scripts.buddy_percept_sweep.h2h_store import H2HStore
from scripts.buddy_percept_sweep.h2h_trial import H2HConfig, Stage2Config, resolve_h2h_config, run_h2h_trial
from scripts.buddy_percept_sweep.h2h_types import H2HSplit, Stage1Output


def test_resolve_routes_prefixed_keys():
    cfg = resolve_h2h_config({"system": "buddy", "k_target": 16, "buddy_lr": 3e-4, "buddy_impl": "harness",
                              "percept_lambda_balance": 10.0, "mapper_lr": 3e-3, "target_cutoff": "0.3",
                              "_wandb": {}})
    assert cfg.buddy.lr == 3e-4 and cfg.buddy.impl == "harness"
    assert cfg.percept.lambda_balance == 10.0
    assert cfg.stage2.mapper_lr == 3e-3 and cfg.stage2.target_cutoff == 0.3


def test_resolve_rejects_unknown_key():
    with pytest.raises(ValueError):
        resolve_h2h_config({"system": "buddy", "k_target": 16, "bogus": 1})


def test_fixed_resolution_mode_is_buddy_only():
    with pytest.raises(ValueError):
        resolve_h2h_config({"system": "percept", "k_target": 0})


# ---------------------------------------------------------------- resolve_h2h_config

_VALID_STR = {"impl": "harness", "heads": "attn4", "mlp_head": "one_hidden"}


def _alternative(name, typ, default):
    """A non-default value of the field's type, and how wandb might send it."""
    if typ is bool:
        return (not default), ("false" if default else "true")
    if typ is int:
        return default + 3, str(default + 3)
    if typ is float:
        value = default * 2 + 0.5
        return value, repr(value)
    if typ is str:
        return _VALID_STR[name], _VALID_STR[name]
    if name == "target_cutoff":
        return 0.15, "0.15"
    raise AssertionError(f"no test value for {name}: {typ}")


@pytest.mark.parametrize("cls,prefix,attr", [(BuddyStage1Config, "buddy_", "buddy"),
                                             (PerceptStage1Config, "percept_", "percept"),
                                             (Stage2Config, "", "stage2")])
def test_resolve_routes_and_coerces_every_field_sent_as_string(cls, prefix, attr):
    hints = typing.get_type_hints(cls)
    raw, expected = {"system": "buddy", "k_target": "16"}, {}
    for field in dataclasses.fields(cls):
        value, sent = _alternative(field.name, hints[field.name], field.default)
        raw[prefix + field.name] = sent
        expected[field.name] = value
    cfg = resolve_h2h_config(raw)
    sub = getattr(cfg, attr)
    for name, value in expected.items():
        got = getattr(sub, name)
        assert got == value and type(got) is type(value), (name, got, value)
    assert cfg.k_target == 16 and type(cfg.k_target) is int


def test_resolve_missing_keys_take_dataclass_defaults():
    cfg = resolve_h2h_config({"system": "percept", "k_target": 40})
    assert cfg == H2HConfig(system="percept", k_target=40, buddy=BuddyStage1Config(),
                            percept=PerceptStage1Config(), stage2=Stage2Config())
    assert (cfg.leiden_graph, cfg.merge_small_threshold, cfg.leiden_resolution) == ("mknn", 0.0, 1.0)


@pytest.mark.parametrize("raw", [{"k_target": 16}, {"system": "buddy"}])
def test_resolve_requires_system_and_k_target(raw):
    with pytest.raises(ValueError):
        resolve_h2h_config(raw)


def test_resolve_top_level_keys_and_zero_k_for_buddy():
    cfg = resolve_h2h_config({"system": "buddy", "k_target": 0, "leiden_graph": "pilot_repaired",
                              "merge_small_threshold": "0.02", "leiden_resolution": 2})
    assert cfg.k_target == 0 and cfg.leiden_graph == "pilot_repaired"
    assert cfg.merge_small_threshold == 0.02 and cfg.leiden_resolution == 2.0
    assert type(cfg.leiden_resolution) is float


@pytest.mark.parametrize("sent,value", [(True, True), (False, False), ("true", True), ("false", False),
                                        ("True", True), ("FALSE", False), (np.bool_(True), True)])
def test_resolve_bool_accepts_bools_and_true_false_strings(sent, value):
    cfg = resolve_h2h_config({"system": "buddy", "k_target": 16, "class_balanced_loss": sent})
    assert cfg.stage2.class_balanced_loss is value


@pytest.mark.parametrize("sent", ["yes", "1", "0", "", 1, 0, 0.0, None])
def test_resolve_bool_parsing_is_strict(sent):
    with pytest.raises(ValueError):
        resolve_h2h_config({"system": "buddy", "k_target": 16, "class_balanced_loss": sent})


@pytest.mark.parametrize("sent,value", [("40", 40), (40.0, 40), ("16.0", 16), (np.int64(16), 16)])
def test_resolve_int_coercion(sent, value):
    cfg = resolve_h2h_config({"system": "buddy", "k_target": sent})
    assert cfg.k_target == value and type(cfg.k_target) is int


@pytest.mark.parametrize("sent", [16.5, "16.5", True, "abc", None])
def test_resolve_int_rejects_non_integers(sent):
    with pytest.raises(ValueError):
        resolve_h2h_config({"system": "buddy", "k_target": sent})


@pytest.mark.parametrize("sent", ["nan", float("inf"), True, "fast", None])
def test_resolve_float_rejects_bad_values(sent):
    with pytest.raises(ValueError):
        resolve_h2h_config({"system": "buddy", "k_target": 16, "buddy_lr": sent})


@pytest.mark.parametrize("sent,value", [("single_label", "single_label"), (0.15, 0.15), ("0.3", 0.3),
                                        (0, 0.0), ("1", 1.0)])
def test_resolve_target_cutoff(sent, value):
    cfg = resolve_h2h_config({"system": "buddy", "k_target": 16, "target_cutoff": sent})
    assert cfg.stage2.target_cutoff == value and type(cfg.stage2.target_cutoff) is type(value)


@pytest.mark.parametrize("sent", ["bogus", "single-label", 1.5, -0.1, "nan", True])
def test_resolve_target_cutoff_rejects_bad_values(sent):
    with pytest.raises(ValueError):
        resolve_h2h_config({"system": "buddy", "k_target": 16, "target_cutoff": sent})


@pytest.mark.parametrize("extra", [{"system": "both"}, {"k_target": -1}, {"leiden_graph": "knn"},
                                   {"buddy_impl": "nope"}, {"buddy_impl": 1}, {"mlp_head": "mlp"},
                                   {"buddy_impl": "pilot", "buddy_heads": "attn"},
                                   {"merge_small_threshold": 1.0}, {"leiden_resolution": 0.0},
                                   {"mapper_epochs": 0}, {"num_queries": 0}, {"train_target_k": 0}])
def test_resolve_rejects_bad_values(extra):
    raw = {"system": "buddy", "k_target": 16}
    raw.update(extra)
    with pytest.raises(ValueError):
        resolve_h2h_config(raw)


# ---------------------------------------------------------------- run_h2h_trial helpers

def _store(n_train=120, n_heldout=60, d_patch=8, seed=0) -> H2HStore:
    rng = np.random.default_rng(seed)
    empty = np.zeros(0)
    return H2HStore(
        train_paintings=np.array([f"t{i}" for i in range(n_train)], dtype=object),
        heldout_paintings=np.array([f"h{i}" for i in range(n_heldout)], dtype=object),
        train_img=empty, train_txt=empty, heldout_img=empty, heldout_txt=empty,
        train_content_raw=empty, heldout_content_raw=empty, train_affect28=empty, heldout_affect28=empty,
        train_percept_h=empty, heldout_percept_h=empty,
        train_emotion=empty, heldout_emotion=rng.choice(np.array(["awe", "fear", "joy"], dtype=object), n_heldout),
        train_genre=empty, heldout_genre=rng.choice(np.array(["x", "y", ""], dtype=object), n_heldout),
        train_patches=torch.as_tensor(rng.normal(size=(n_train, 50, d_patch)), dtype=torch.float32),
        heldout_patches=torch.as_tensor(rng.normal(size=(n_heldout, 50, d_patch)), dtype=torch.float32),
    )


def _split(n_heldout=60) -> H2HSplit:
    idx = np.arange(n_heldout)
    return H2HSplit(val_idx=idx[idx % 2 == 0].astype(np.int64), test_idx=idx[idx % 2 == 1].astype(np.int64),
                    digest="d1g3st")


def _clustered(seed, n_train, n_heldout, n_topics=6, dim=8):
    rng = np.random.default_rng(seed)
    centers = 5.0 * rng.normal(size=(n_topics, dim))
    train_topic = rng.permutation(np.arange(n_train) % n_topics)
    heldout_topic = rng.permutation(np.arange(n_heldout) % n_topics)
    train = (centers[train_topic] + 0.3 * rng.normal(size=(n_train, dim))).astype(np.float32)
    heldout = (centers[heldout_topic] + 0.3 * rng.normal(size=(n_heldout, dim))).astype(np.float32)
    return train, heldout, train_topic, heldout_topic


def _fake_buddy(calls):
    def fit_buddy_stage1(cfg, store, seed, monitor_idx, pilot, device):
        calls.append({"cfg": cfg, "seed": seed, "monitor_idx": np.asarray(monitor_idx).copy(), "device": device})
        train, heldout, _, _ = _clustered(seed, len(store.train_paintings), len(store.heldout_paintings))
        return Stage1Output(train, heldout, None, None, info={"seconds": 0.0})
    return fit_buddy_stage1


def _fake_percept(calls, n_topics=6):
    def fit_percept_stage1(cfg, store, seed, k_target, mods, device):
        calls.append({"cfg": cfg, "seed": seed, "k_target": k_target, "mods": mods, "device": device})
        train, heldout, train_topic, heldout_topic = _clustered(
            seed, len(store.train_paintings), len(store.heldout_paintings), n_topics=n_topics)
        return Stage1Output(train, heldout, train_topic, heldout_topic,
                            info={"n_train_topics": n_topics, "seconds": 0.0})
    return fit_percept_stage1


def _fake_stage1_metrics(calls):
    def stage1_metrics(train_embedding, train_labels, heldout_embedding, subset_idx, heldout_emotion,
                       heldout_genre, seed, pilot, device, native_subset=None, transfer_labels=None):
        calls.append({"subset_idx": np.asarray(subset_idx).copy(), "seed": seed, "native_subset": native_subset,
                      "n_labels": len(np.unique(train_labels)), "transfer_labels": transfer_labels})
        out = {"transfer_emo": 0.2, "transfer_genre": 0.3, "ind_emo": 0.1, "ind_genre": 0.25, "ind_k": 9}
        if native_subset is not None:
            out.update(native_emo=0.15, native_genre=0.35)
        return out
    return stage1_metrics


def _fake_stage2_metrics(calls, auc_by_seed=None):
    def stage2_metrics(s2cfg, store, train_embedding, train_labels, subset_idx, eval_label_sets, seed, device):
        calls.append({"seed": seed, "subset_idx": np.asarray(subset_idx).copy(),
                      "width": int(train_labels.max()) + 1, "n_distinct": len(np.unique(train_labels)),
                      "sets": {k: np.asarray(v).copy() for k, v in eval_label_sets.items()}})
        auc = (auc_by_seed or {}).get(seed, 0.7)
        out = {}
        for name in eval_label_sets:
            out[f"auc_{name}"] = auc
            out[f"skipped_{name}"] = 0
        return out
    return stage2_metrics


def _patch_fakes(monkeypatch, buddy_calls=None, percept_calls=None, s1_calls=None, s2_calls=None,
                 auc_by_seed=None, percept_topics=6):
    monkeypatch.setattr(h2h_trial, "fit_buddy_stage1", _fake_buddy([] if buddy_calls is None else buddy_calls))
    monkeypatch.setattr(h2h_trial, "fit_percept_stage1",
                        _fake_percept([] if percept_calls is None else percept_calls, percept_topics))
    monkeypatch.setattr(h2h_trial, "stage1_metrics", _fake_stage1_metrics([] if s1_calls is None else s1_calls))
    monkeypatch.setattr(h2h_trial, "stage2_metrics",
                        _fake_stage2_metrics([] if s2_calls is None else s2_calls, auc_by_seed))


def _fake_target_k(calls, miss_seeds=()):
    def target_k_partition(embedding, graph, k_target, tolerance, merge_threshold, seed, **kwargs):
        calls.append({"seed": seed, "k_target": k_target, "tolerance": tolerance, "merge": merge_threshold,
                      "graph": graph, **kwargs})
        labels = np.arange(len(embedding)) % k_target
        hit = seed not in miss_seeds
        return labels, {"resolution": 1.5, "k_raw": k_target, "k_after_merge": k_target,
                        "steps": 3 if hit else 12, "hit": hit}
    return target_k_partition


# ---------------------------------------------------------------- run_h2h_trial

def test_k_miss_on_one_seed_gives_objective_minus_one_and_keeps_both_rows(monkeypatch):
    """Review Focus 1 (a miss on the last seed: every seed ran, nothing left to abort)."""
    s2_calls, tk_calls = [], []
    _patch_fakes(monkeypatch, s2_calls=s2_calls)
    monkeypatch.setattr(h2h_trial, "build_topic_graph", lambda emb, kind, pilot, device, **kw: ("graph", kind))
    monkeypatch.setattr(h2h_trial, "target_k_partition", _fake_target_k(tk_calls, miss_seeds=(2,)))
    cfg = resolve_h2h_config({"system": "buddy", "k_target": 16, "merge_small_threshold": 0.01})
    out = run_h2h_trial(cfg, _store(), _split(), "val", (1, 2), pilot=None, percept_mods=None, device="cpu")
    rows = out["per_seed"]
    assert out["objective"] == -1.0
    assert [r["seed"] for r in rows] == [1, 2]
    assert [r["k_miss"] for r in rows] == [False, True]
    assert rows[0]["auc_primary"] == 0.7 and math.isnan(rows[1]["auc_primary"])
    assert [c["seed"] for c in s2_calls] == [1]          # Stage 2 skipped on the missed seed
    assert rows[1]["n_topics"] == 16 and rows[1]["stage2_seconds"] == 0.0
    assert all((c["tolerance"], c["k_target"], c["merge"], c["graph"]) == (2, 16, 0.01, ("graph", "mknn"))
               for c in tk_calls)
    assert out["split_digest"] == "d1g3st"
    assert out["aborted_after_k_miss"] is False


def test_k_miss_on_first_seed_aborts_the_remaining_seeds(monkeypatch):
    buddy_calls, s2_calls = [], []
    _patch_fakes(monkeypatch, buddy_calls=buddy_calls, s2_calls=s2_calls)
    monkeypatch.setattr(h2h_trial, "build_topic_graph", lambda emb, kind, pilot, device, **kw: "graph")
    monkeypatch.setattr(h2h_trial, "target_k_partition", _fake_target_k([], miss_seeds=(1,)))
    cfg = resolve_h2h_config({"system": "buddy", "k_target": 16})
    out = run_h2h_trial(cfg, _store(), _split(), "val", (1, 2), pilot=None, percept_mods=None, device="cpu")
    assert out["objective"] == -1.0
    assert [r["seed"] for r in out["per_seed"]] == [1] and out["per_seed"][0]["k_miss"] is True
    assert [c["seed"] for c in buddy_calls] == [1]          # Stage 1 ran once
    assert s2_calls == [] and out["aborted_after_k_miss"] is True


def test_buddy_rows_carry_bisection_diagnostics(monkeypatch):
    _patch_fakes(monkeypatch)
    monkeypatch.setattr(h2h_trial, "build_topic_graph", lambda *a, **k: "graph")
    monkeypatch.setattr(h2h_trial, "target_k_partition", _fake_target_k([]))
    cfg = resolve_h2h_config({"system": "buddy", "k_target": 16})
    row, = run_h2h_trial(cfg, _store(), _split(), "val", (1,), None, None, "cpu")["per_seed"]
    assert row["k_raw"] == 16 and row["bisect_steps"] == 3


def test_percept_rows_have_no_bisection_diagnostics(monkeypatch):
    _patch_fakes(monkeypatch)
    cfg = resolve_h2h_config({"system": "percept", "k_target": 16})
    row, = run_h2h_trial(cfg, _store(), _split(), "val", (1,), None, "mods", "cpu")["per_seed"]
    assert row["k_raw"] is None and row["bisect_steps"] == 0


def test_percept_n_topics_is_distinct_train_labels_not_k_target(monkeypatch):
    """Review Focus 4: DEC leaves only 38 populated topics for k_target 40."""
    percept_calls, s1_calls, s2_calls = [], [], []
    _patch_fakes(monkeypatch, percept_calls=percept_calls, s1_calls=s1_calls, s2_calls=s2_calls,
                 percept_topics=38)
    cfg = resolve_h2h_config({"system": "percept", "k_target": 40})
    split, store = _split(), _store()
    out = run_h2h_trial(cfg, store, split, "val", (5,), pilot=None, percept_mods="mods", device="cpu")
    row, = out["per_seed"]
    assert row["n_topics"] == 38 and row["k_miss"] is False and math.isnan(row["resolution"])
    assert percept_calls[0]["k_target"] == 40 and percept_calls[0]["seed"] == 5 and percept_calls[0]["mods"] == "mods"
    assert s2_calls[0]["width"] == 38 and s2_calls[0]["n_distinct"] == 38
    assert set(s2_calls[0]["sets"]) == {"primary", "native"}
    _, _, _, heldout_native = _clustered(5, 120, 60, n_topics=38)
    assert np.array_equal(s2_calls[0]["sets"]["native"], heldout_native[split.val_idx])
    assert np.array_equal(s1_calls[0]["native_subset"], heldout_native[split.val_idx])
    assert np.array_equal(s2_calls[0]["sets"]["primary"], s1_calls[0]["transfer_labels"])
    for key in ("auc_native", "skipped_native", "native_emo", "native_genre"):
        assert key in row
    assert out["objective"] == 0.7


def test_fixed_resolution_mode_uses_config_resolution_and_never_bisects(monkeypatch):
    """Review Focus 5."""
    leiden_calls, merge_calls = [], []
    _patch_fakes(monkeypatch)
    monkeypatch.setattr(h2h_trial, "build_topic_graph", lambda emb, kind, pilot, device, **kw: "graph")
    monkeypatch.setattr(h2h_trial, "target_k_partition",
                        lambda *a, **k: pytest.fail("target_k_partition called in k_target=0 mode"))

    def fake_leiden(graph, resolution, seed):
        leiden_calls.append((graph, resolution, seed))
        return np.arange(120) % 7

    real_merge = h2h_trial.merge_small_communities

    def spy_merge(embedding, labels, min_fraction):
        merge_calls.append(min_fraction)
        return real_merge(embedding, labels, min_fraction)

    monkeypatch.setattr(h2h_trial, "leiden_on_graph", fake_leiden)
    monkeypatch.setattr(h2h_trial, "merge_small_communities", spy_merge)
    cfg = resolve_h2h_config({"system": "buddy", "k_target": 0, "leiden_resolution": 0.37,
                              "merge_small_threshold": 0.005})
    out = run_h2h_trial(cfg, _store(), _split(), "val", (3, 4), pilot=None, percept_mods=None, device="cpu")
    assert leiden_calls == [("graph", 0.37, 3), ("graph", 0.37, 4)]
    assert merge_calls == [0.005, 0.005]
    assert [r["resolution"] for r in out["per_seed"]] == [0.37, 0.37]
    assert [r["k_miss"] for r in out["per_seed"]] == [False, False]
    assert [r["n_topics"] for r in out["per_seed"]] == [7, 7]
    assert [(r["k_raw"], r["bisect_steps"]) for r in out["per_seed"]] == [(7, 0), (7, 0)]
    assert out["objective"] == pytest.approx(0.7)


def test_run_routes_subset_and_monitor_indices(monkeypatch):
    buddy_calls, s1_calls, s2_calls = [], [], []
    _patch_fakes(monkeypatch, buddy_calls=buddy_calls, s1_calls=s1_calls, s2_calls=s2_calls)
    monkeypatch.setattr(h2h_trial, "build_topic_graph", lambda *a, **k: "graph")
    monkeypatch.setattr(h2h_trial, "target_k_partition", _fake_target_k([]))
    cfg = resolve_h2h_config({"system": "buddy", "k_target": 16, "buddy_lr": 0.002})
    split, store = _split(), _store()
    run_h2h_trial(cfg, store, split, "test", (9,), pilot="pilot", percept_mods=None, device="cpu")
    run_h2h_trial(cfg, store, split, "val", (9,), pilot="pilot", percept_mods=None, device="cpu", monitor="all")
    assert np.array_equal(buddy_calls[0]["monitor_idx"], split.val_idx)
    assert np.array_equal(buddy_calls[1]["monitor_idx"], np.arange(60))
    assert buddy_calls[0]["cfg"] is cfg.buddy and buddy_calls[0]["cfg"].lr == 0.002
    assert np.array_equal(s1_calls[0]["subset_idx"], split.test_idx)
    assert np.array_equal(s2_calls[0]["subset_idx"], split.test_idx)
    assert np.array_equal(s1_calls[1]["subset_idx"], split.val_idx)
    assert len(s2_calls[0]["sets"]["primary"]) == len(split.test_idx)


@pytest.mark.parametrize("subset,monitor", [("train", "val"), ("val", "test"), ("all", "val")])
def test_run_rejects_bad_subset_or_monitor(monkeypatch, subset, monitor):
    _patch_fakes(monkeypatch)
    cfg = resolve_h2h_config({"system": "percept", "k_target": 6})
    with pytest.raises(ValueError):
        run_h2h_trial(cfg, _store(), _split(), subset, (1,), None, None, "cpu", monitor=monitor)


def test_run_rejects_percept_fixed_resolution_mode(monkeypatch):
    _patch_fakes(monkeypatch)
    cfg = dataclasses.replace(resolve_h2h_config({"system": "percept", "k_target": 6}), k_target=0)
    with pytest.raises(ValueError):
        run_h2h_trial(cfg, _store(), _split(), "val", (1,), None, None, "cpu")


def test_objective_is_mean_primary_auc_and_row_keys(monkeypatch):
    _patch_fakes(monkeypatch, auc_by_seed={1: 0.7, 2: 0.8})
    monkeypatch.setattr(h2h_trial, "build_topic_graph", lambda *a, **k: "graph")
    monkeypatch.setattr(h2h_trial, "target_k_partition", _fake_target_k([]))
    cfg = resolve_h2h_config({"system": "buddy", "k_target": 16})
    out = run_h2h_trial(cfg, _store(), _split(), "val", (1, 2), None, None, "cpu")
    assert out["objective"] == pytest.approx(0.75)
    assert out["mean"]["auc_primary"] == pytest.approx(0.75)
    assert out["mean"]["k_miss"] == 0.0 and "seed" not in out["mean"]
    assert set(out) == {"per_seed", "objective", "mean", "split_digest", "aborted_after_k_miss"}
    assert set(out["per_seed"][0]) == {
        "seed", "n_topics", "k_miss", "resolution", "k_raw", "bisect_steps", "auc_primary", "skipped_primary", "transfer_emo",
        "transfer_genre", "ind_emo", "ind_genre", "ind_k", "stage1_seconds", "stage2_seconds"}


def test_non_finite_primary_auc_fails_the_objective(monkeypatch):
    _patch_fakes(monkeypatch, auc_by_seed={1: 0.7, 2: float("nan")})
    cfg = resolve_h2h_config({"system": "percept", "k_target": 6})
    out = run_h2h_trial(cfg, _store(), _split(), "val", (1, 2), None, None, "cpu")
    assert out["objective"] == -1.0


# ---------------------------------------------------------------- determinism (real topics + metrics)

def _mknn_pilot():
    from src.conditional_buddy.buddy_graph import mutual_knn

    def build_single_modality_graph(name, nodes, pipeline, affect_pilot, device, expected_nodes):
        return mutual_knn(np.asarray(nodes, dtype=np.float32), K=5, device="cpu")

    return SimpleNamespace(pipeline="train", heldout_pipeline="heldout", affect_pilot=None,
                           single_modality=SimpleNamespace(build_single_modality_graph=build_single_modality_graph))


def _without_timing(rows):
    return [json.dumps({k: v for k, v in row.items() if not k.endswith("_seconds")}, sort_keys=True)
            for row in rows]


@pytest.mark.parametrize("system", ["buddy", "percept"])
def test_same_config_and_seed_give_identical_rows(monkeypatch, system):
    """Only Stage 1 is faked (seed-dependent random output); topics, both
    AMI yardsticks and the Stage 2 mapper run for real on CPU."""
    monkeypatch.setattr(h2h_trial, "fit_buddy_stage1", _fake_buddy([]))
    monkeypatch.setattr(h2h_trial, "fit_percept_stage1", _fake_percept([]))
    cfg = resolve_h2h_config({"system": system, "k_target": 6, "mapper_epochs": 5, "target_cutoff": "0.3",
                              "train_target_k": 10})
    store, split, pilot = _store(), _split(), _mknn_pilot()
    first = run_h2h_trial(cfg, store, split, "val", (1, 2), pilot, None, "cpu")
    torch.rand(10); np.random.rand(10)
    second = run_h2h_trial(cfg, store, split, "val", (1, 2), pilot, None, "cpu")
    assert _without_timing(first["per_seed"]) == _without_timing(second["per_seed"])
    assert first["objective"] == second["objective"]
    assert all(not r["k_miss"] and math.isfinite(r["auc_primary"]) for r in first["per_seed"])
    a, b = _without_timing(first["per_seed"])
    assert a.replace('"seed": 1', "") != b.replace('"seed": 2', "")   # the fake really is seed-dependent
    json.dumps(first)                                                  # rows are JSON-serializable
