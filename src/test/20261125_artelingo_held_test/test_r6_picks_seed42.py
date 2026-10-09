"""Real-data tests of r6_picks, r6_score and run_r6_picks on seed 42, selection rows (ticket 06; rule section 5 item
4, section 6 items 3 and 4, section 11). Synthetic tests and guard mutations are in test_r6_score.py.

The bundle is built by run_r6_picks.build_bundle_seed42 (the runner's own path: refit heads, the selection-mode
RowContext, fit_pm, build_bundle_r6). Targets: rule section 6 item 3's three mean R@1 values; round 3's
results/seed42_arrays.npz (AFF fused, CF, R1 fused, R1 counterpart, the gates and the reader's picks, margins and
probabilities); AB's per_anchor_seed42.npz (cosine, RCA and the nine PM scorers at the frozen lambdas). Selection rows
only; nothing of a held row is read. Prints no metric (assertion messages on failure aside).

About 8 to 10 minutes on CPU (the bundle):

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python -m pytest -q -p no:cacheprovider \
        src/test/20261125_artelingo_held_test/test_r6_picks_seed42.py
"""
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r6_common as R  # noqa: E402  (first, before anything that imports src)
import r6_bundle as B  # noqa: E402
import r6_picks as P  # noqa: E402
import r6_score as S  # noqa: E402
import run_r6_picks as RP  # noqa: E402

import numpy as np  # noqa: E402

from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS, METRICS, per_anchor  # noqa: E402

R3_ARRAYS_REL = "20261121_round3_affect_gate/results/seed42_arrays.npz"
PER_ANCHOR42_REL = "20261030_aspect_baselines/results/per_anchor_seed42.npz"
TARGETS = {"B": 18.341064453125, "B0": 18.436686197916664, "B1": 18.804931640625}   # rule section 6 item 3


def bits(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def scores_bits(s, t) -> bool:
    return all(bits(s[c][d], t[c][d]) for c in CONDITIONS for d in DIRECTIONS)


@pytest.fixture(scope="module")
def real():
    bundle = RP.build_bundle_seed42()
    picks, scores = P.crossfit_seed42(bundle)
    return SimpleNamespace(bundle=bundle, picks=picks, scores=scores)


@pytest.fixture(scope="module")
def scored(real):
    return S.score_seed(real.bundle, {n: real.picks[n] for n in P.NESTED}, P.frozen_lambdas(), real.bundle.readers,
                        include_pm=True)


def test_mean_r1_hits_the_rule_targets(real):
    assert P.MEAN_R1_TARGETS == TARGETS
    for name, want in TARGETS.items():
        got = real.picks["mean_r1"][name]
        assert type(got) is float and got == want, name


def test_frozen_equals_crossfit_element_for_element(real):
    """The frozen path with seed 42's own picks reproduces crossfit_condition_free's scores bit for bit."""
    b = real.bundle
    for name in P.NESTED:
        frozen = S.frozen_nested(b.cos, b.t_n1u, getattr(b, P.TERM_OF[name]), real.picks[name], b.parity)
        assert scores_bits(frozen, real.scores[name]), name
    assert P.frozen_equals_crossfit(b, real.picks, real.scores) == {n: True for n in P.NESTED}


def test_run_writes_picks_seed42(real, tmp_path, capsys):
    out = tmp_path / "picks_seed42.json"
    assert RP.run(real.bundle, out) == 0
    rec = json.loads(out.read_text())
    assert rec["passed"] is True
    for name in P.NESTED:
        assert rec[name] == {"0": real.picks[name][0], "1": real.picks[name][1]}
        assert rec["mean_r1"][name] == TARGETS[name]
    assert all(all(v.values()) for v in rec["checks"].values())
    assert set(rec["checks"]) == {"mean_r1_equal_target", "frozen_equals_crossfit", "lambda_convention"}
    assert rec["module_sha256"] == R.r6_module_shas()
    assert rec["episodes_sha256"] == real.bundle.episodes_sha256 and rec["coef_sha256"] == real.bundle.coef_sha256
    assert P.load_picks(out) == {name: real.picks[name] for name in P.NESTED}
    printed = capsys.readouterr().out
    for v in TARGETS.values():
        assert repr(v) not in printed and f"{v:.2f}" not in printed


def test_run_exits_3_on_a_one_ulp_difference(real, tmp_path, monkeypatch):
    monkeypatch.setitem(P.MEAN_R1_TARGETS, "B0", math.nextafter(TARGETS["B0"], 100.0))
    out = tmp_path / "picks_seed42.json"
    assert RP.run(real.bundle, out) == 3 and not out.exists()
    rec = json.loads((tmp_path / "picks_seed42_failed.json").read_text())
    assert rec["checks"]["mean_r1_equal_target"] == {"B": True, "B0": False, "B1": True}


def test_lambda_key_convention(real):
    """Rerunning crossfit_lambda on seed 42 gives baselines_seed42.json's lambda_picks key for key ("0" = tuned on
    half 0); the halves swapped (probe 0.25/0.5, diag 0/0.25) disagree."""
    lams = P.frozen_lambdas()
    assert P.assert_lambda_convention(real.bundle, lams) == {n: True for n in P.LAMBDA_SCORERS}
    swapped = {name: {0: p[1], 1: p[0]} for name, p in lams.items()}
    got = P.lambda_convention(real.bundle, swapped)
    assert got["probe"] is False and got["diag"] is False and got["rca"] is True
    with pytest.raises(AssertionError, match="probe"):
        P.assert_lambda_convention(real.bundle, swapped)


def test_score_seed_reproduces_round3_arrays(real, scored):
    """AFF fused, CF, R1 fused and R1 counterpart per-anchor arrays, the gates and the reader's outputs equal round 3's
    seed42_arrays.npz exactly (the frozen cells on z of the frozen B)."""
    R.assert_input(R3_ARRAYS_REL)
    bad = []
    with np.load(R.INPUT_PATHS[R3_ARRAYS_REL]) as z:
        assert np.array_equal(z["cl"], scored["cl"]) and np.array_equal(z["pair_index"], scored["pair_index"])
        assert np.array_equal(z["parity"], real.bundle.parity)
        for who in ("aff", "r1"):
            for part in ("fused", "cf"):
                bad += [f"{who}_{part}__{m}" for m in METRICS
                        if not bits(scored[f"{who}_{part}"][m], z[f"{who}_{part}__{m}"])]
            bad += [f"{who}_gate__{c}" for c in CONDITIONS if not bits(scored["gates"][who][c], z[f"{who}_gate__{c}"])]
            assert tuple(z[f"{who}_fused_cells"]) == S.CELLS[who]["fused"] and tuple(z[f"{who}_cf_cells"]) == \
                S.CELLS[who]["cf"]
        for c in CONDITIONS:
            bad += [f"{k}__{c}" for k, mine in (("pick", "pick"), ("margin", "m"), ("P", "P"))
                    if not bits(scored["reader"][mine][c], z[f"{k}__{c}"])]
    assert not bad, bad


def test_score_seed_reproduces_per_anchor_seed42(scored):
    """Cosine, RCA and the nine PM scorers at the frozen lambdas equal per_anchor_seed42.npz exactly."""
    R.assert_input(PER_ANCHOR42_REL)
    bad = []
    with np.load(R.INPUT_PATHS[PER_ANCHOR42_REL]) as z:
        assert np.array_equal(z["anchor_group"], scored["cl"])
        for name in ("cosine", "rca") + B.PM_NAMES:
            bad += [f"{name}__{m}" for m in METRICS if not np.array_equal(scored[name][m], z[f"{name}__{m}"])]
    assert not bad, bad


def test_score_seed_nested_means_and_no_pm(real, scored):
    for name, want in TARGETS.items():
        assert 100 * float(np.mean(scored[name]["r1"])) == want, name
        assert all(bits(scored[name][m], per_anchor(real.scores[name])[m]) for m in METRICS), name
    out = S.score_seed(real.bundle, {n: real.picks[n] for n in P.NESTED}, P.frozen_lambdas(), real.bundle.readers,
                       include_pm=False)
    assert tuple(out) == S.CORE_SCORERS + S.EXTRA_KEYS
    for k in S.CORE_SCORERS:
        assert all(bits(out[k][m], scored[k][m]) for m in METRICS), k
