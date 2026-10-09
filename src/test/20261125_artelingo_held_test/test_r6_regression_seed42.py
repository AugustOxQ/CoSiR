"""Real-data tests of run_r6_held.py's regression mode on seed 42, selection rows (ticket 07; rule section 6 items 4
and 5). Fast tests (order guard, wiring, sensitivity, guard mutations) are in test_r6_regression.py.

One module fixture runs the runner's own setup (inputs, rows, heads refit and checked, PM fits, readers) and builds the
seed-42 selection bundle with its seed_bundle; the refit and picks records are written into tmp_path by the real
stage-1a and stage-1b functions (run_r6_refit.record, run_r6_picks.run). Then:
  - the unmutated regression passes every item (exit 0) and the sensitivity runner reproduces r3_stats.sensitivity's
    parts on its per-episode file;
  - on a copy of picks_seed42.json with one comparator's picks swapped between tune halves, the regression fails with
    exit code 3, on the items that depend on that comparator only;
  - with one frozen cell changed (AFF fused, tune half 1: 119 -> 118, in memory), it fails with exit code 3 on AFF's
    items while the comparators' items still pass.
Nothing of a held row is read; F/results is never written. Prints no metric.

About 6 to 10 minutes on CPU:

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python -m pytest -q -p no:cacheprovider \
        src/test/20261125_artelingo_held_test/test_r6_regression_seed42.py
"""
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r6_common as R  # noqa: E402  (first, before anything that imports src)
import r6_score as S  # noqa: E402
import r6_stats as ST  # noqa: E402
import run_r6_held as RH  # noqa: E402
import run_r6_picks as RP  # noqa: E402
import run_r6_refit as RR  # noqa: E402
import run_r6_sensitivity as RS6  # noqa: E402

import numpy as np  # noqa: E402


@pytest.fixture(scope="module")
def real(tmp_path_factory):
    env = RH.setup()
    bundle = RH.seed_bundle(env, "selection", R.DEV_SEED, R.N_PER_PAIR)
    rec_dir = tmp_path_factory.mktemp("records")
    RR.report(RR.record(env.head_check, env.heads, env.inputs), rec_dir / RH.REFIT_NAME)
    assert RP.run(bundle, rec_dir / RH.PICKS_NAME) == 0
    return SimpleNamespace(env=env, bundle=bundle, rec_dir=rec_dir)


def results_copy(real, tmp_path):
    res = tmp_path / "results"
    res.mkdir()
    for name in (RH.REFIT_NAME, RH.PICKS_NAME):
        shutil.copy2(real.rec_dir / name, res / name)
    return res


def run(real, res, capsys):
    code = RH.run_regression(results=res, here=HERE, env=real.env, bundle=real.bundle)
    out = capsys.readouterr().out
    rec_path = res / (RH.REGRESSION_NAME if code == 0 else "regression_seed42_failed.json")
    return code, out, json.loads(rec_path.read_text())


def failed(rec) -> set:
    return {k for k, it in rec["items"].items() if not it["equal"]}


def test_regression_passes_every_item(real, tmp_path, capsys):
    res = results_copy(real, tmp_path)
    code, out, rec = run(real, res, capsys)
    assert code == 0 and rec["passed"] is True and failed(rec) == set(), sorted(failed(rec))
    names = set(rec["items"])
    for k in RH.TARGETS:
        assert k in names, k
    for who in RH.WHO:
        for part in RH.PARTS:
            assert all(f"seed42_arrays/{who}_{part}__{m}" in names for m in ("r1", "gain", "other", "swap", "strict"))
        assert {f"seed42_arrays/{who}_gate__a", f"seed42_arrays/{who}_gate__b"} <= names
    for s in RH.AB_SCORERS:
        assert f"per_anchor_seed42/{s}__r1" in names and f"per_anchor_seed42/{s}__gain" in names
    for seed in (42, 9001, 9002, 9003):
        assert all(f"episodes_seed{seed}__{p}" in names for p in R.PAIR_NAMES)
    assert f"PASSED ({len(names)} items, 0 failed)" in out
    counts = json.loads((res / RH.COUNTS_NAME).read_text())
    assert counts["mode"] == "regression" and counts["seeds"] == [42] and counts["n_episodes"] == 3 * R.N_PER_PAIR
    assert counts["episodes_sha256"] == {"42": rec["episodes_sha256"]}
    # the sensitivity input on the same output: r3_stats.sensitivity's parts
    assert RS6.run(res, HERE) == 0
    sens = json.loads((res / RS6.OUT_NAME).read_text())
    with np.load(res / RH.PER_EPISODE_NAME) as z:
        assert np.array_equal(z["cl"], real.bundle.cl) and np.array_equal(z["pair_index"], real.bundle.pair_index)
        for c in ST.CHECKS + ST.SECONDARY:
            r3 = R.RS.sensitivity(z[f"diff__{c}"], z["cl"])
            assert (sens[c]["sigma_a2"], sens[c]["sigma_eps2"]) == (r3["sigma_a2"], r3["sigma_e2"]), c


def test_swapped_picks_on_a_copy_fail_with_exit_3(real, tmp_path, capsys):
    """B0's picks swapped between tune halves (B if B0's halves agree): B0 moves, and with it the bar margins (their
    comparator, B'(A0), and D10's comparator choice) and the P4 diff; AFF's own arrays do not."""
    res = results_copy(real, tmp_path)
    rec = json.loads((res / RH.PICKS_NAME).read_text())
    name = "B0" if rec["B0"]["0"] != rec["B0"]["1"] else "B"
    assert rec[name]["0"] != rec[name]["1"], "both comparators have equal picks in the two halves"
    rec[name] = {"0": rec[name]["1"], "1": rec[name]["0"]}
    (res / RH.PICKS_NAME).write_text(json.dumps(rec))           # the copy, not F/results
    code, out, frec = run(real, res, capsys)
    assert code == RH.EXIT_DIFF == 3 and frec["passed"] is False
    bad = failed(frec)
    assert f"{name}_mean_r1" in bad and "aff_bar_margin" in bad, sorted(bad)
    assert not (res / RH.REGRESSION_NAME).exists() and not (res / RH.COUNTS_NAME).exists()
    for k in ("seed42_arrays/aff_fused__r1", "seed42_arrays/aff_cf__r1", "aff_fused_r1", "episodes_seed42__"
              "emotion__style", "per_anchor_seed42/rca__r1", "B1_mean_r1"):
        assert k not in bad, k
    assert "stage-1b stop" in out


def test_one_changed_cell_fails_with_exit_3(real, tmp_path, capsys, monkeypatch):
    res = results_copy(real, tmp_path)
    monkeypatch.setitem(S.CELLS["aff"], "fused", (39, 118))     # tune half 1: cell 119 (tau2, 0, 16) -> 118 (tau2, 0, 8)
    code, out, frec = run(real, res, capsys)
    assert code == 3 and frec["passed"] is False
    bad = failed(frec)
    assert {"aff_fused_r1", "seed42_arrays/aff_fused__r1", "seed42_arrays/aff_fused_cells"} <= bad, sorted(bad)
    for k in ("aff_cf_r1", "seed42_arrays/aff_cf__r1", "seed42_arrays/r1_fused__r1", "r1_margin_vs_counterpart",
              "B0_mean_r1", "per_anchor_seed42/cosine__r1"):
        assert k not in bad, k
    assert not (res / RH.REGRESSION_NAME).exists()
