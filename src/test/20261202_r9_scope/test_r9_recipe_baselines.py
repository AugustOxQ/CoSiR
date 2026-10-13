"""Tests of r9_recipe_baselines (ticket 10; rule §2, §3.2 items 1 and 3, §11; constitution C1). Written before the module.

Synthetic bundles of the real shapes (12,288 episodes, 13 candidates) for the verdict code (lambda picks, the pick
records, the exit codes, the refusals); the fit-row hash assertion on CLIP's real scorer-train index array with
synthetic features; the light bundle against `r9_bundle.build_bundle_r6` on the synthetic row context of
test_r9_context.py. The real-data checks (CLIP's pinned lambdas, picks and means) are the CLIP run of the script.

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python -m pytest -q -p no:cacheprovider \
        src/test/20261202_r9_scope/test_r9_recipe_baselines.py
"""
import json
import math
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r9_common as R  # noqa: E402  (first, before anything that imports src)
import r9_bundle as B  # noqa: E402
import r9_picks as P  # noqa: E402
import r9_recipe_baselines as RB  # noqa: E402
from test_r9_context import held, pm_syn, readers, scores_bits, syn  # noqa: E402,F401  (fixtures)
from test_r9_score import make_bundle  # noqa: E402

import numpy as np  # noqa: E402

from src.eval.aspect_scorers import EvalInputs  # noqa: E402

CLIP_BASE = json.loads(Path(R.input_pins("clip_b32")[B.BASELINES42_REL][0]).read_text())
CLIP_FIT_SHA = CLIP_BASE["fit_rows_sha256"]
CLIP_EPISODES = CLIP_BASE["episodes_sha256"]
OTHER = "siglip2_so400m"


# ---------------------------------------------------------------- helpers

def seed42_bundle(rng_seed=7):
    """A synthetic seed-42 selection bundle that carries CLIP's episode and fit-row hashes."""
    b = make_bundle(seed=R.DEV_SEED, mode="selection", rng_seed=rng_seed)
    b.episodes_sha256 = dict(CLIP_EPISODES)
    b.fit_rows_sha256 = CLIP_FIT_SHA
    b.coef_sha256 = {"affect": {"img": "a" * 64, "txt": "b" * 64}}
    return b


@pytest.fixture(scope="module")
def sel42():
    return seed42_bundle()


@pytest.fixture(scope="module")
def lams(sel42):
    return RB.crossfit_lambdas(sel42)


@pytest.fixture(scope="module")
def crossfit(sel42):
    return P.crossfit_seed42(sel42)


FAKE_INPUTS = {"/x/store/img.npy": "c" * 64}


# ---------------------------------------------------------------- the fit rows (rule §11; ticket 10's unit test)

def test_changed_fit_row_draw_fails_the_hash_assert(syn, monkeypatch):
    st = R._read_npz_keys(R.INPUT_PATHS[R.PREPARE_REL], ("scorer_train",))["scorer_train"]
    pm = B.fit_pm(syn.data, st, "clip_b32")                     # CLIP's draw: the SHA holds
    assert pm.fit_rows_sha256 == CLIP_FIT_SHA
    monkeypatch.setattr(B, "fit_pca_basis", lambda *a, **k: pytest.fail("fitted on a draw that failed the hash"))
    with pytest.raises(AssertionError, match="unsorted draw"):
        B.fit_pm(syn.data, st[1:], "clip_b32")                  # another scorer-train array: another draw
    monkeypatch.setattr(B, "FIT_SEED", 1)                               # another seed: another draw
    with pytest.raises(AssertionError, match="unsorted draw"):
        B.fit_pm(syn.data, st, OTHER)


def test_baselines_record_refuses_a_changed_fit_row_hash_or_episodes(sel42, lams):
    RB.baselines_record(sel42, lams)                                    # the matching bundle passes
    bad = seed42_bundle()
    bad.fit_rows_sha256 = "0" * 64
    with pytest.raises(AssertionError, match="fit_rows_sha256"):
        RB.baselines_record(bad, lams)
    bad = seed42_bundle()
    bad.episodes_sha256 = {**CLIP_EPISODES, "style__genre": "1" * 64}
    with pytest.raises(AssertionError, match="episodes_sha256"):
        RB.baselines_record(bad, lams)


# ---------------------------------------------------------------- lambda picks and the baselines record

def test_crossfit_lambdas_are_crossfit_lambda_per_scorer(sel42, lams):
    assert tuple(lams) == P.LAMBDA_SCORERS and len(lams) == 10
    from src.eval.aspect_scorers import crossfit_lambda
    for name, lam in lams.items():
        assert lam == P.as_lambda_pick(crossfit_lambda(sel42.cos, sel42.pm_terms[name], sel42.parity)[1], name)


def test_baselines_record_keys_and_the_chain_reads_it(sel42, lams):
    rec = RB.baselines_record(sel42, lams)
    for key in ("fit_rows", "fit_rows_sha256", "lambda_picks", "episodes_sha256"):
        assert key in rec, key
    assert rec["fit_rows"] == 60000 and rec["fit_rows_sha256"] == CLIP_FIT_SHA
    assert rec["episodes_sha256"] == CLIP_EPISODES and rec["pair_order"] == list(R.PAIR_NAMES)
    assert tuple(rec["lambda_picks"]) == P.LAMBDA_SCORERS
    assert all(rec["scorers"][n]["lambda_picks"] == rec["lambda_picks"][n] for n in P.LAMBDA_SCORERS)
    again = json.loads(json.dumps(rec))                                 # through the file format
    assert P.parse_lambda_picks(again) == lams                          # the reader of the chain (r9_picks)


def test_an_infinite_lambda_is_written_as_the_string(sel42):
    lams = {n: {0: 0.5, 1: float("inf")} for n in P.LAMBDA_SCORERS}
    rec = json.loads(json.dumps(RB.baselines_record(sel42, lams)))
    assert rec["lambda_picks"]["rca"] == {"0": 0.5, "1": "inf"}
    got = P.parse_lambda_picks(rec)
    assert math.isinf(got["wang"][1]) and got["wang"][0] == 0.5


# ---------------------------------------------------------------- the picks record and the exit codes

def test_clip_run_fails_without_the_pinned_numbers(sel42, tmp_path):
    """Synthetic scores cannot equal CLIP's means or lambdas: exit 3, only the failed record, no baselines."""
    code = RB.finish(sel42, tmp_path, "clip_b32", FAKE_INPUTS)
    assert code == RB.EXIT_DIFF
    assert sorted(p.name for p in tmp_path.iterdir()) == ["picks_seed42_failed.json"]
    rec = json.loads((tmp_path / "picks_seed42_failed.json").read_text())
    assert rec["passed"] is False
    assert not any(rec["checks"]["mean_r1_equal_target"].values())
    assert "lambda_equals_pinned" in rec["checks"]


def test_clip_run_passes_when_the_pins_are_the_computed_values(sel42, lams, crossfit, tmp_path, monkeypatch):
    """The CLIP success branch, with the pinned values replaced by the synthetic ones."""
    monkeypatch.setattr(P, "MEAN_R1_TARGETS", dict(crossfit[0]["mean_r1"]))
    monkeypatch.setattr(P, "frozen_lambdas", lambda backbone="clip_b32": lams)
    assert RB.finish(sel42, tmp_path, "clip_b32", FAKE_INPUTS) == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "baselines_seed42.json", "baselines_seed42.json.record.json",
        "picks_seed42.json", "picks_seed42.json.record.json"]
    assert P.load_picks(tmp_path / "picks_seed42.json", "clip_b32") == {n: crossfit[0][n] for n in P.NESTED}


def test_a_lambda_differing_from_the_pin_fails_clip(sel42, lams, crossfit, tmp_path, monkeypatch):
    monkeypatch.setattr(P, "MEAN_R1_TARGETS", dict(crossfit[0]["mean_r1"]))
    other = {n: dict(v) for n, v in lams.items()}
    other["kissme"][0] = 32.0 if other["kissme"][0] != 32.0 else 0.0
    monkeypatch.setattr(P, "frozen_lambdas", lambda backbone="clip_b32": other)
    assert RB.finish(sel42, tmp_path, "clip_b32", FAKE_INPUTS) == RB.EXIT_DIFF
    rec = json.loads((tmp_path / "picks_seed42_failed.json").read_text())
    assert rec["checks"]["lambda_equals_pinned"] == {n: n != "kissme" for n in P.LAMBDA_SCORERS}
    assert not (tmp_path / "baselines_seed42.json").exists() and not (tmp_path / "picks_seed42.json").exists()


def test_a_new_backbone_passes_without_clip_targets(sel42, lams, crossfit, tmp_path):
    assert RB.finish(sel42, tmp_path, OTHER, FAKE_INPUTS) == 0
    rec = json.loads((tmp_path / "picks_seed42.json").read_text())
    assert rec["passed"] is True and "mean_r1_equal_target" not in rec["checks"]
    assert rec["targets"] is None
    got = P.load_picks(tmp_path / "picks_seed42.json", OTHER)           # what the chain reads for this backbone
    assert got == {n: crossfit[0][n] for n in P.NESTED}
    assert P.parse_lambda_picks(json.loads((tmp_path / "baselines_seed42.json").read_text())) == lams
    assert rec["checks"]["frozen_equals_crossfit"] == {n: True for n in P.NESTED}
    assert rec["checks"]["lambda_convention"] == {n: True for n in P.LAMBDA_SCORERS}
    for k in ("episodes_sha256", "coef_sha256", "fit_rows_sha256", "input_sha256", "lambda_picks"):
        assert k in rec, k
    assert rec["mean_r1"] == crossfit[0]["mean_r1"]


def test_records_name_inputs_script_and_keys(sel42, tmp_path):
    RB.finish(sel42, tmp_path, OTHER, FAKE_INPUTS)
    for name, made in (("baselines_seed42.json", ["scorers", "lambda_picks", "convention"]),
                       ("picks_seed42.json", ["B", "B0", "B1", "mean_r1"])):
        rec = json.loads((tmp_path / (name + ".record.json")).read_text())
        assert rec["backbone"] == OTHER and rec["input_sha256"] == FAKE_INPUTS
        assert rec["script_sha256"] == R.sha256_file(RB.__file__)
        assert set(made) <= set(rec["keys_made"])
        written = set(json.loads((tmp_path / name).read_text()))
        assert written >= set(rec["keys_made"]) and written >= set(rec["keys_copied"])
        assert "r9_picks" in rec["code"]["modules"]


def test_existing_outputs_are_never_overwritten(sel42, tmp_path):
    assert RB.finish(sel42, tmp_path, OTHER, FAKE_INPUTS) == 0
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        RB.preflight(tmp_path, OTHER)
    with pytest.raises(FileExistsError):
        RB.finish(sel42, tmp_path, OTHER, FAKE_INPUTS)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_a_non_seed42_bundle_is_refused(tmp_path):
    held_like = make_bundle(seed=52, mode="held")
    held_like.episodes_sha256, held_like.fit_rows_sha256 = dict(CLIP_EPISODES), CLIP_FIT_SHA
    with pytest.raises(AssertionError):
        RB.finish(held_like, tmp_path, OTHER, FAKE_INPUTS)
    assert not list(tmp_path.iterdir())


# ---------------------------------------------------------------- the light bundle equals build_bundle_r6's terms

def test_light_bundle_equals_the_chain_bundle(held, pm_syn, readers):
    full = B.build_bundle_r6(held, readers, pm_syn)
    a3 = B.model_inputs(held)
    light = RB.build_baseline_bundle(held, pm_syn, a3)
    assert light.n == full.n and np.array_equal(light.parity, full.parity)
    for k in ("cos", "t_n1u", "t6u_B", "t6u_B0", "t6u_B1"):
        assert scores_bits(getattr(light, k), getattr(full, k)), k
    assert tuple(light.pm_terms) == tuple(full.pm_terms) == B.PM_TERM_NAMES
    for k in full.pm_terms:
        assert scores_bits(light.pm_terms[k], full.pm_terms[k]), k
    assert light.episodes_sha256 == full.episodes_sha256 and light.coef_sha256 == full.coef_sha256
    assert (light.mode, light.seed, light.smoke) == (full.mode, full.seed, full.smoke)
    assert light.fit_rows_sha256 == "synthetic"
    for absent in ("stack", "F", "readers"):
        assert not hasattr(light, absent), absent                      # ticket 09's readers are not needed


def test_light_bundle_refuses_codes_that_are_not_the_contexts(held, pm_syn):
    other = EvalInputs(held.img, held.txt)                              # no A3 codes
    with pytest.raises(AssertionError):
        RB.build_baseline_bundle(held, pm_syn, other)
