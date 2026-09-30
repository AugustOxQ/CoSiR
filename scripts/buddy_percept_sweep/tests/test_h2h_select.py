import json
import math
from pathlib import Path

import pytest

from scripts.buddy_percept_sweep import h2h_select as sel
from scripts.buddy_percept_sweep.h2h_trial import resolve_h2h_config

REPO = Path(__file__).resolve().parents[3]
WANDB_CONFIG = {
    "_wandb": {"runtime": 3}, "system": "buddy", "k_target": 16, "leiden_graph": "mknn",
    "merge_small_threshold": 0.02, "leiden_resolution": 1.0, "buddy_impl": "harness",
    "buddy_heads": "attn1", "buddy_lr": 0.0005, "percept_max_dec_epochs": 300, "mapper_lr": 0.01,
    "mapper_epochs": 400, "num_queries": 2, "mlp_head": "linear", "target_cutoff": "single_label",
    "class_balanced_loss": False, "weight_decay_stage2": 0.0, "train_target_k": 20,
    "some_sweep_only_key": 7, "run_cap": 300,
}


def _run(i, obj, config=None):
    return {"id": f"r{i}", "objective": obj, "config": config or {"system": "buddy", "k_target": 16}}


def test_config_for_trial_roundtrips_wandb_config():
    clean = sel.config_for_trial(WANDB_CONFIG)
    assert "_wandb" not in clean and "some_sweep_only_key" not in clean and "run_cap" not in clean
    assert clean["buddy_lr"] == 0.0005 and clean["percept_max_dec_epochs"] == 300
    cfg = resolve_h2h_config(clean)
    assert cfg.system == "buddy" and cfg.stage2.num_queries == 2


def test_select_top_filters_sorts_and_strips():
    runs = [_run(1, 0.6), _run(2, -1.0), _run(3, None), _run(4, 0.8), _run(5, 0.7, WANDB_CONFIG), _run(6, 0.6)]
    top = sel.select_top(runs, 3)
    assert [(r["rank"], r["id"]) for r in top] == [(1, "r4"), (2, "r5"), (3, "r1")]
    assert "_wandb" not in top[1]["config"]
    assert set(top[0]) == {"rank", "id", "objective", "config"}


def _row(seed, auc, miss=False, **kw):
    row = {"seed": seed, "n_topics": 16, "k_miss": miss, "auc_primary": auc, "ind_emo": 0.2, "ind_genre": 0.3,
           "transfer_emo": 0.1, "transfer_genre": 0.15}
    row.update(kw)
    return row


def test_summarize_rows_excludes_k_miss_but_counts():
    s = sel.summarize_rows([_row(1, 0.6), _row(2, math.nan, miss=True), _row(3, 0.8)])
    assert s["auc_mean"] == pytest.approx(0.7)
    assert s["auc_min"] == 0.6 and s["auc_max"] == 0.8
    assert s["auc_std"] == pytest.approx(0.1)
    assert s["k_miss_count"] == 1 and s["n_seeds"] == 3
    assert s["n_topics"] == [16, 16, 16]
    assert s["ind_emo_mean"] == pytest.approx(0.2)
    assert "auc_native_mean" not in s


def test_summarize_rows_all_missed_and_native():
    s = sel.summarize_rows([_row(1, math.nan, miss=True)])
    assert math.isnan(s["auc_mean"]) and s["k_miss_count"] == 1
    s = sel.summarize_rows([_row(1, 0.6, auc_native=0.5), _row(2, 0.7, auc_native=math.nan)])
    assert s["auc_native_mean"] == pytest.approx(0.5)


def test_jsonable_turns_nan_into_null():
    assert json.loads(sel.dumps({"a": math.nan, "b": 1.5, "c": None})) == {"a": None, "b": 1.5, "c": None}
    assert "NaN" not in sel.dumps({"a": float("nan")})


def test_parse_lines_dedups_last_wins(tmp_path):
    a, b = tmp_path / "a.log", tmp_path / "b.log"
    a.write_text('noise\nH2H_RESULT {"tag":"t","rank":1,"run_id":"x","auc_mean":0.1}\n'
                 'H2H_SEED {"tag":"t","rank":1,"run_id":"x"}\n')
    b.write_text('H2H_RESULT {"tag":"t","rank":1,"run_id":"x","auc_mean":0.2}\n'
                 'H2H_RESULT {"tag":"t","rank":2,"run_id":"y","auc_mean":0.3}\n')
    out = sel.parse_lines([str(a), str(b)], "H2H_RESULT ")
    assert {(r["rank"], r["auc_mean"]) for r in out} == {(1, 0.2), (2, 0.3)}


def test_render_markdown_groups_sorts_and_names_winner():
    summaries = [
        {"tag": "A", "subset": "val", "rank": 1, "run_id": "x", "sweep_objective": 0.7, "auc_mean": 0.6,
         "auc_std": 0.01, "auc_min": 0.59, "auc_max": 0.61, "n_topics": [16, 16], "k_miss_count": 0},
        {"tag": "A", "subset": "val", "rank": 2, "run_id": "y", "sweep_objective": 0.71, "auc_mean": 0.65,
         "auc_std": 0.02, "auc_min": 0.6, "auc_max": 0.7, "n_topics": [16, 17], "k_miss_count": 1},
        {"tag": "B", "subset": "test", "rank": 0, "run_id": "z", "auc_mean": None, "k_miss_count": 2,
         "n_topics": [40]},
    ]
    md = sel.render_markdown(summaries)
    assert md.index("| 2 | y") < md.index("| 1 | x")
    assert "Winner (A, val): rank 2 (y)" in md
    assert "no valid config" in md
    assert "NaN" not in md


def test_reference_configs_resolve():
    fin = REPO / "src/test/20260928_buddy_percept_sweep/finalists.json"
    m = sel.reference_config("m8x7ifx4", fin)
    cfg = resolve_h2h_config(m)
    assert cfg.system == "buddy" and cfg.k_target == 0 and cfg.buddy.impl == "harness"
    assert cfg.buddy.heads == "attn1" and cfg.buddy.num_heads == 4 and cfg.buddy.teacher_graph_K == 15
    assert cfg.leiden_graph == "mknn" and cfg.merge_small_threshold == 0.02
    assert cfg.leiden_resolution == pytest.approx(0.8355571774303963)
    assert cfg.stage2.train_target_k == 40 and cfg.stage2.target_cutoff == 0.15
    assert cfg.stage2.num_queries == 8 and cfg.stage2.mlp_head == "one_hidden"
    p = resolve_h2h_config(sel.reference_config("percept_6g", fin))
    assert p.system == "percept" and p.k_target == 40 and p.stage2.target_cutoff == "single_label"
    assert p.stage2.mapper_lr == 1e-2 and p.stage2.train_target_k == 20
    with pytest.raises(ValueError):
        sel.reference_config("nope", fin)


def test_parse_int_list_and_parser():
    assert sel.parse_int_list("1,2", "--x") == [1, 2]
    args = sel.build_parser().parse_args(
        ["run", "--finalists", "f.json", "--ranks", "1,2", "--subset", "val", "--tag", "c"])
    assert args.subset == "val" and args.seeds is None


def test_run_config_calls_trial_per_seed_and_emits_lines(monkeypatch, capsys):
    calls = []

    def fake_trial(cfg, store, split, subset, seeds, pilot, mods, device, monitor="val", topic_graph_device=None):
        calls.append((seeds, subset, monitor, topic_graph_device))
        miss = seeds == (2,)
        return {"per_seed": [_row(seeds[0], math.nan if miss else 0.6, miss=miss, k_raw=None)]}

    import scripts.buddy_percept_sweep.h2h_trial as ht
    monkeypatch.setattr(ht, "run_h2h_trial", fake_trial)
    ctx = sel.RunContext(store="s", split="sp", pilot="p", percept_mods=None, device="cpu")
    summary = sel.run_config(WANDB_CONFIG, ctx, tag="t", rank=3, run_id="abc", subset="val", seeds=[1, 2, 3],
                             monitor="val", sweep_objective=0.7, topic_graph_device="cpu")
    assert [c[0] for c in calls] == [(1,), (2,), (3,)]      # a K miss never aborts later seeds
    assert all(c[3] == "cpu" for c in calls)
    assert summary["k_miss_count"] == 1 and summary["auc_mean"] == pytest.approx(0.6)
    lines = capsys.readouterr().out.strip().splitlines()
    seeds = [json.loads(l[len("H2H_SEED "):]) for l in lines if l.startswith("H2H_SEED ")]
    res = [json.loads(l[len("H2H_RESULT "):]) for l in lines if l.startswith("H2H_RESULT ")]
    assert len(seeds) == 3 and len(res) == 1
    assert seeds[1]["auc_primary"] is None and seeds[0]["tag"] == "t" and seeds[0]["subset"] == "val"
    assert res[0]["sweep_objective"] == 0.7 and res[0]["rank"] == 3 and res[0]["run_id"] == "abc"


def test_select_top_drops_non_finished_runs():
    runs = [{**_run(1, 0.9), "state": "running"}, {**_run(2, 0.8), "state": "crashed"},
            {**_run(3, 0.7), "state": "finished"}]
    assert [r["id"] for r in sel.select_top(runs, 5)] == ["r3"]
