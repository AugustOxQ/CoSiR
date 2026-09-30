"""Unit tests for the pure functions in run_top10_stress.py (finalist
selection, per-seed summarising, ranking, log parsing). No wandb, GPU, or
real data is touched.
"""
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_top10_stress as m


def _run(run_id, objective, n_topics, merge):
    return {
        "id": run_id, "objective": objective, "n_topics": n_topics,
        "auc": objective, "emo": 0.2, "genre": 0.3,
        "config": {"merge_small_threshold": merge, "heads": "attn1"},
    }


def _mixed_runs():
    return [
        _run("gatefail", -1.0, 20, 0.01),        # gate-failed
        _run("huge455", 0.9488, 455, 0.0),       # highest objective, 455 topics, merge off
        _run("nomerge30", 0.93, 30, 0.0),        # few topics but merge off
        _run("over45", 0.92, 50, 0.01),          # merge on but over the topic cap
        _run("ok_a", 0.90, 40, 0.01),
        _run("ok_b", 0.91, 45, 0.05),            # exactly at the cap: kept
        _run("ok_c", 0.85, 12, 0.02),
        _run("noobj", None, 12, 0.02),           # objective missing
        _run("notopics", 0.95, None, 0.02),      # n_topics missing
    ]


def test_select_finalists_filters_sorts_and_ranks():
    runs = _mixed_runs()
    snapshot = copy.deepcopy(runs)

    out = m.select_finalists(runs)

    assert [r["id"] for r in out] == ["ok_b", "ok_a", "ok_c"]
    assert [r["rank"] for r in out] == [1, 2, 3]
    assert runs == snapshot  # input not mutated
    assert all("rank" not in r for r in runs)
    assert out[0] is not runs[5]  # new dicts, not the input rows


def test_select_finalists_returns_fewer_than_n_when_few_qualify():
    out = m.select_finalists(_mixed_runs(), n=10)
    assert len(out) == 3


def test_select_finalists_caps_at_n():
    runs = [_run(f"r{i}", 0.5 + i / 100, 20, 0.01) for i in range(15)]
    out = m.select_finalists(runs, n=10)
    assert len(out) == 10
    assert [r["id"] for r in out][:3] == ["r14", "r13", "r12"]
    assert [r["rank"] for r in out] == list(range(1, 11))


def test_select_finalists_respects_max_topics_override():
    out = m.select_finalists(_mixed_runs(), n=10, max_topics=40)
    assert [r["id"] for r in out] == ["ok_a", "ok_c"]


def test_select_finalists_is_stable_on_ties():
    runs = [_run("first", 0.9, 20, 0.01), _run("second", 0.9, 20, 0.01)]
    out = m.select_finalists(runs)
    assert [r["id"] for r in out] == ["first", "second"]


def test_select_finalists_treats_string_merge_threshold_numerically():
    runs = [_run("strzero", 0.9, 20, "0.0"), _run("strpos", 0.8, 20, "0.01")]
    out = m.select_finalists(runs)
    assert [r["id"] for r in out] == ["strpos"]


def _seed_row(seed, objective, auc, emo=0.2, genre=0.3, n_topics=30):
    return {"seed": seed, "objective": objective, "auc": auc,
            "emo": emo, "genre": genre, "n_topics": n_topics}


def test_summarize_seed_results_with_one_gate_failure():
    per_seed = [
        _seed_row(42, 0.90, 0.90, emo=0.20, genre=0.30, n_topics=30),
        _seed_row(7, 0.80, 0.80, emo=0.10, genre=0.40, n_topics=31),
        _seed_row(123, -1.0, 0.70, emo=0.30, genre=0.20, n_topics=32),
        _seed_row(2024, 0.86, 0.86, emo=0.40, genre=0.10, n_topics=33),
    ]
    aucs = [0.90, 0.80, 0.70, 0.86]

    s = m.summarize_seed_results(per_seed)

    assert s["n_seeds"] == 4
    assert s["gate_pass_count"] == 3
    assert s["auc_mean"] == pytest.approx(np.mean(aucs))
    assert s["auc_std"] == pytest.approx(np.std(aucs, ddof=0))
    assert s["auc_min"] == pytest.approx(0.70)
    assert s["auc_max"] == pytest.approx(0.90)
    assert s["emo_mean"] == pytest.approx(0.25)
    assert s["genre_mean"] == pytest.approx(0.25)
    assert s["objective_mean"] == pytest.approx((0.90 + 0.80 - 1.0 + 0.86) / 4)
    assert s["n_topics"] == [30, 31, 32, 33]
    assert s["per_seed"] == per_seed


def test_summarize_seed_results_is_json_serialisable():
    s = m.summarize_seed_results([_seed_row(42, 0.9, 0.9), _seed_row(7, 0.8, 0.8)])
    json.dumps(s, separators=(",", ":"))


def _summary(rank, run_id, gate_pass_count, auc_mean):
    return {"rank": rank, "run_id": run_id,
            "gate_pass_count": gate_pass_count, "auc_mean": auc_mean}


def test_rank_finalists_gate_pass_dominates_auc():
    summaries = [
        _summary(1, "high_auc_flaky", 2, 0.95),
        _summary(2, "solid", 4, 0.90),
        _summary(3, "mid", 3, 0.99),
    ]
    snapshot = copy.deepcopy(summaries)

    out = m.rank_finalists(summaries)

    assert [s["run_id"] for s in out] == ["solid", "mid", "high_auc_flaky"]
    assert summaries == snapshot  # input order and content untouched


def test_rank_finalists_breaks_gate_ties_by_auc_mean():
    summaries = [
        _summary(1, "lower", 4, 0.90),
        _summary(2, "higher", 4, 0.93),
        _summary(3, "worse", 3, 0.99),
    ]
    out = m.rank_finalists(summaries)
    assert [s["run_id"] for s in out] == ["higher", "lower", "worse"]


def test_parse_stress_results_ignores_noise_and_dedups_last_wins(tmp_path):
    log_a = tmp_path / "a.log"
    log_b = tmp_path / "b.log"
    first_r1 = {"rank": 1, "run_id": "aaa", "auc_mean": 0.80}
    r10 = {"rank": 10, "run_id": "jjj", "auc_mean": 0.70}
    second_r1 = {"rank": 1, "run_id": "aaa", "auc_mean": 0.85}
    seed_line = 'STRESS_SEED {"rank":1,"run_id":"aaa","seed":42}'
    log_a.write_text(
        "loading data...\n"
        + seed_line + "\n"
        + "STRESS_RESULT " + json.dumps(first_r1, separators=(",", ":")) + "\n"
        + "Traceback noise STRESS_RESULT not at line start\n"
        + "STRESS_RESULT " + json.dumps(r10, separators=(",", ":")) + "\n"
    )
    log_b.write_text(
        "\n"
        "STRESS_RESULT " + json.dumps(second_r1, separators=(",", ":")) + "\n"
        "done\n"
    )

    out = m.parse_stress_results([str(log_a), str(log_b)])

    by_key = {(r["rank"], r["run_id"]): r for r in out}
    assert len(out) == 2
    assert by_key[(1, "aaa")]["auc_mean"] == 0.85  # last occurrence wins
    assert by_key[(10, "jjj")]["auc_mean"] == 0.70


def test_parse_stress_results_empty_when_no_result_lines(tmp_path):
    log = tmp_path / "empty.log"
    log.write_text("just noise\nSTRESS_SEED {}\n")
    assert m.parse_stress_results([str(log)]) == []


def test_format_summary_markdown_lists_winner_and_columns():
    def result(rank, run_id, passes, aucs):
        per_seed = [
            _seed_row(s, a if i < passes else -1.0, a, n_topics=20 + i)
            for i, (s, a) in enumerate(zip((42, 7, 123, 2024), aucs))
        ]
        summary = m.summarize_seed_results(per_seed)
        summary.update(rank=rank, run_id=run_id, sweep_objective=0.9, sweep_n_topics=20)
        return summary

    ranked = m.rank_finalists([
        result(1, "flaky", 2, [0.9, 0.9, 0.9, 0.9]),
        result(2, "steady", 4, [0.8, 0.8, 0.8, 0.8]),
    ])

    md = m.format_summary_markdown(ranked)

    assert md.rstrip().splitlines()[-1] == "Winner: rank 2 (steady)"
    assert "| 1 | 2 | steady |" in md      # stress rank 1 is sweep rank 2
    assert "4/4" in md and "2/4" in md
    assert "20/21/22/23" in md
    assert "AUC mean" in md and "topics per seed" in md.lower()


def test_run_to_row_maps_summary_keys_and_drops_underscore_config():
    fake_run = SimpleNamespace(
        id="abc123",
        summary_metrics={
            "objective": 0.91, "stage2_macro_auc": 0.91, "stage1_emotion_ami": 0.2,
            "stage1_genre_ami": 0.3, "n_topics_after_merge": 33, "_runtime": 99,
        },
        config={"heads": "attn1", "merge_small_threshold": 0.01, "_wandb": {"x": 1}},
    )

    row = m._run_to_row(fake_run)

    assert row == {
        "id": "abc123", "objective": 0.91, "auc": 0.91, "emo": 0.2, "genre": 0.3,
        "n_topics": 33, "config": {"heads": "attn1", "merge_small_threshold": 0.01},
    }


def test_stress_errors_clearly_on_missing_rank_before_loading_anything(tmp_path):
    finalists = tmp_path / "finalists.json"
    finalists.write_text(json.dumps(m.select_finalists([_run("a", 0.9, 20, 0.01)])))
    args = m.build_parser().parse_args(
        ["stress", "--finalists", str(finalists), "--ranks", "1,3"])

    with pytest.raises(SystemExit) as excinfo:
        args.func(args)

    assert "[3]" in str(excinfo.value)
    assert "available: [1]" in str(excinfo.value)


def test_apply_overrides_parses_json_and_falls_back_to_string():
    config = {"transfer_k": 40, "class_balanced_loss": False, "heads": "attn1", "noise_std": 0.1}
    snapshot = copy.deepcopy(config)

    out = m.apply_overrides(
        config, ["transfer_k=20", "class_balanced_loss=true", "heads=mlp128", "noise_std=0.25"])

    assert out == {"transfer_k": 20, "class_balanced_loss": True, "heads": "mlp128", "noise_std": 0.25}
    assert config == snapshot  # input not mutated


def test_apply_overrides_unknown_key_errors_clearly():
    with pytest.raises(SystemExit) as excinfo:
        m.apply_overrides({"transfer_k": 40}, ["transferk=20"])
    assert "transferk" in str(excinfo.value)


def test_apply_overrides_rejects_missing_equals():
    with pytest.raises(SystemExit):
        m.apply_overrides({"transfer_k": 40}, ["transfer_k"])


def test_summarize_seed_results_adds_independent_keys_only_when_every_row_has_them():
    rows = [
        {**_seed_row(42, 0.9, 0.9), "ind_emo": 0.13, "ind_genre": 0.20, "ind_k": 30},
        {**_seed_row(7, 0.9, 0.9), "ind_emo": 0.12, "ind_genre": 0.25, "ind_k": 31},
        {**_seed_row(123, 0.9, 0.9), "ind_emo": 0.14, "ind_genre": 0.19, "ind_k": 29},
    ]
    s = m.summarize_seed_results(rows)
    assert s["ind_emo_mean"] == pytest.approx((0.13 + 0.12 + 0.14) / 3)
    assert s["ind_genre_mean"] == pytest.approx((0.20 + 0.25 + 0.19) / 3)
    assert s["ind_gate_pass_count"] == 1  # only the first row clears both bars

    partial = m.summarize_seed_results(rows[:2] + [_seed_row(123, 0.9, 0.9)])
    assert not any(key.startswith("ind_") for key in partial)
    plain = m.summarize_seed_results([_seed_row(42, 0.9, 0.9)])
    assert not any(key.startswith("ind_") for key in plain)


def test_format_summary_markdown_shows_tag_and_independent_columns_when_present():
    per_seed = [{**_seed_row(s, 0.9, 0.9), "ind_emo": 0.13, "ind_genre": 0.21, "ind_k": 30}
                for s in (42, 7)]
    summary = m.summarize_seed_results(per_seed)
    summary.update(rank=1, run_id="abc", sweep_objective=0.9, sweep_n_topics=20, tag="k20")

    md = m.format_summary_markdown([summary])

    assert "| tag |" in md and "| k20 |" in md
    assert "ind emo mean" in md and "ind genre mean" in md and "ind gate passes" in md
    assert "| 2/2 |" in md  # ind gate passes

    plain = m.summarize_seed_results([_seed_row(42, 0.9, 0.9)])
    plain.update(rank=1, run_id="abc", sweep_objective=0.9, sweep_n_topics=20)
    plain_md = m.format_summary_markdown([plain])
    assert "tag" not in plain_md and "ind emo" not in plain_md


def test_stress_parser_accepts_new_flags():
    args = m.build_parser().parse_args(
        ["stress", "--finalists", "f.json", "--ranks", "1", "--set", "transfer_k=20",
         "--set", "seed=3", "--tag", "k20", "--independent-ami"])
    assert args.set == ["transfer_k=20", "seed=3"]
    assert args.tag == "k20" and args.independent_ami is True
    defaults = m.build_parser().parse_args(["stress", "--finalists", "f.json", "--ranks", "1"])
    assert defaults.set == [] and defaults.tag == "" and defaults.independent_ami is False


def test_stress_errors_on_unknown_set_key_before_loading_anything(tmp_path):
    finalists = tmp_path / "finalists.json"
    finalists.write_text(json.dumps(m.select_finalists([_run("a", 0.9, 20, 0.01)])))
    args = m.build_parser().parse_args(
        ["stress", "--finalists", str(finalists), "--ranks", "1", "--set", "nope=1"])

    with pytest.raises(SystemExit) as excinfo:
        args.func(args)

    assert "nope" in str(excinfo.value)
