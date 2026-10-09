"""Tests of r6_picks, r6_score and run_r6_picks on synthetic bundles of the real shapes (ticket 06; rule section 5
item 4, section 6 item 3, section 8 item 3, section 11). The real-data tests (selection rows, seed 42) are in
test_r6_picks_seed42.py.

Synthetic bundles: 12,288 episodes (3 x 4,096), 13 candidates, float32 score terms (cosine, T_N1u and the three T_6u
condition-free), stack (n, 3, 13) float32, reader features (n, 18) float64, the ten RCA and PM terms; held seed 52 and
selection seed 42. Real files read: round 1's A0 half-readers, rc_tau.json (through round 3's assert_taus) and
baselines_seed42.json (frozen lambdas); no held feature, label or episode. Mutation tests load a copy of the module
from tmp_path with its `# guard:<name>` statements replaced by `pass` and show that the guard's scenario then goes
through (never in place).

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python -m pytest -q -p no:cacheprovider \
        src/test/20261125_artelingo_held_test/test_r6_score.py
"""
import ast
import importlib.util
import itertools
import json
import math
import re
import sys
from collections.abc import Mapping
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
import torch  # noqa: E402

from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS, METRICS, per_anchor  # noqa: E402
from src.eval.aspect_nested import _combine, _zdict, nested_scores  # noqa: E402
from src.eval.aspect_quick_checks import crossfit_condition_free  # noqa: E402
from src.eval.aspect_scorers import crossfit_lambda, fused_scores  # noqa: E402

RF, RC, F2, R3 = R.RF, R.RC, R.F2, R.R3
N_EP = 3 * R.N_PER_PAIR
K = 13
_COUNT = itertools.count()
GUARDED = ("r6_picks.py", "r6_score.py", "run_r6_picks.py")


class Reached(Exception):
    """Raised by a stub placed after a guard: the scenario went through the guard."""


# ---------------------------------------------------------------- helpers

def mutant(tmp_path, module, *guards):
    """Import a copy of ``module`` (file name in HERE) with its `# guard:<g>` statements replaced by `pass`, for each
    g in ``guards``."""
    src = (HERE / module).read_text()
    lines = src.splitlines(keepends=True)
    for guard in guards:
        hits = [n for n in ast.walk(ast.parse(src))
                if isinstance(n, (ast.Expr, ast.Assign)) and f"# guard:{guard}" in lines[n.end_lineno - 1]]
        assert hits, f"no statement of {module} carries # guard:{guard}"
        for n in hits:
            first = lines[n.lineno - 1]
            lines[n.lineno - 1] = first[:len(first) - len(first.lstrip())] + "pass\n"
            for i in range(n.lineno, n.end_lineno):
                lines[i] = "\n"
        src = "".join(lines)
        lines = src.splitlines(keepends=True)
    path = tmp_path / f"{Path(module).stem}_mut{next(_COUNT)}.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    saved = list(sys.path)
    sys.modules[path.stem] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(path.stem, None)
        sys.path[:] = saved
    return mod


def bits(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def scores_bits(s, t) -> bool:
    return all(bits(s[c][d], t[c][d]) for c in CONDITIONS for d in DIRECTIONS)


def pa_bits(x, y) -> bool:
    return all(bits(x[m], y[m]) for m in METRICS)


def rows_of(s, mask):
    return {c: {d: np.asarray(s[c][d])[mask] for d in DIRECTIONS} for c in CONDITIONS}


def free_term(rng):
    """A condition-free float32 term (equal under both conditions)."""
    x = {d: rng.standard_normal((N_EP, K), dtype=np.float32) for d in DIRECTIONS}
    return {c: {d: x[d].copy() for d in DIRECTIONS} for c in CONDITIONS}


def cond_term(rng):
    return {c: {d: rng.standard_normal((N_EP, K), dtype=np.float32) for d in DIRECTIONS} for c in CONDITIONS}


class PMPoison(Mapping):
    """pm_terms whose nine PM entries raise when read (rca reads normally)."""

    def __init__(self, terms):
        self._t = dict(terms)

    def __getitem__(self, k):
        if k in B.PM_NAMES:
            raise Reached(f"PM term {k} was read")
        return self._t[k]

    def __iter__(self):
        return iter(self._t)

    def __len__(self):
        return len(self._t)


def make_bundle(seed=52, mode="held", rng_seed=6):
    rng = np.random.default_rng(rng_seed)
    n = N_EP
    Fa = rng.standard_normal((n, 18))
    Fb = rng.standard_normal((n, 18))
    for j in range(3):
        Fb[:, 6 * j + 2] = -Fa[:, 6 * j + 2]
    stack = {d: rng.random((n, 3, K), dtype=np.float32) for d in DIRECTIONS}
    return SimpleNamespace(
        n=n, mode=mode, seed=seed, smoke=False, cl=rng.integers(0, 3000, n).astype(np.int64),
        parity=np.arange(n) % 2, pair_index=np.repeat(np.arange(3), R.N_PER_PAIR),
        anchor=rng.integers(0, R.N_ROWS, n).astype(np.int64),
        cos=free_term(rng), t_n1u=free_term(rng), t6u_B=free_term(rng), t6u_B0=free_term(rng),
        t6u_B1=free_term(rng), stack=stack, F={"a": Fa, "b": Fb},
        pm_terms={name: cond_term(rng) for name in B.PM_TERM_NAMES})


NESTED_PICKS = {"B": {0: [4.0, 16.0], 1: [0.5, 0.25]}, "B0": {0: [0.0, 1.0], 1: [2.0, 0.0]},
                "B1": {0: [16.0, 8.0], 1: [1.0, 4.0]}}


def lambdas_all():
    out = {"rca": {0: 0.5, 1: 2.0}}
    for i, name in enumerate(B.PM_NAMES):
        out[name] = {0: [0.0, 0.25, 1.0, float("inf")][i % 4], 1: [4.0, 16.0, 0.5, 32.0][i % 4]}
    return out


@pytest.fixture(scope="module")
def readers():
    return B.load_readers()


@pytest.fixture(scope="module")
def held():
    return make_bundle()


@pytest.fixture(scope="module")
def sel42():
    return make_bundle(seed=R.DEV_SEED, mode="selection", rng_seed=7)


@pytest.fixture(scope="module")
def reader_gates(held, readers):
    rd = RF.reader(held, readers)
    taus = R3.assert_taus()
    return SimpleNamespace(rd=rd, g_r1=RF.gates_r1(rd["m"], taus), g_aff=RF.gates_aff(rd["m"], rd["pick"], taus))


@pytest.fixture(scope="module")
def scored_held(held, readers):
    return S.score_seed(held, NESTED_PICKS, lambdas_all(), readers, include_pm=False)


# ---------------------------------------------------------------- bookkeeping

def test_every_marked_guard_has_a_mutation_test():
    names = set()
    for module in GUARDED:
        names |= {(module, g) for g in re.findall(r"# guard:([a-z_0-9]+)", (HERE / module).read_text())}
    text = Path(__file__).read_text()
    covered = set()
    for module, args in re.findall(r'mutant\(tmp_path, "((?:r6_picks|r6_score|run_r6_picks)\.py)", ([^)]*)\)', text):
        covered |= {(module, g) for g in re.findall(r'"([a-z_0-9]+)"', args)}
    assert names and names <= covered, sorted(names - covered)


def test_cells_are_the_rule_table():
    """Rule section 5 item 4: AFF (39, 119), CF (149, 10), R1 fused (116, 119), R1 counterpart (58, 123); each cell's
    (tau index, lambda_u, lambda_a) as the rule's table, k_top 13; equal to round 3's constants."""
    assert S.CELLS == {"aff": {"fused": (39, 119), "cf": (149, 10)}, "r1": {"fused": (116, 119), "cf": (58, 123)}}
    assert S.CELLS["aff"]["fused"] == R3.AFF_CELLS["fused"] and S.CELLS["aff"]["cf"] == R3.AFF_CELLS["cf"]
    assert S.CELLS["r1"]["fused"] == R3.RC_CELLS["fused"] and S.CELLS["r1"]["cf"] == R3.RC_CELLS["cf"]
    text = {39: (0, 4, 16), 119: (2, 0, 16), 149: (2, 4, 4), 10: (0, 0.5, 0.5), 116: (2, 0, 2), 58: (1, 0, 0.5),
            123: (2, 0.5, 1)}
    for cell, (t, u, a) in text.items():
        assert F2.cell_values(cell) == (13, t, u, a), cell


def test_scorer_names():
    assert S.PM_SCORERS == B.PM_NAMES
    assert S.CORE_SCORERS == ("cosine", "rca", "B", "B0", "B1", "aff_fused", "aff_cf", "r1_fused", "r1_cf")
    assert S.ALL_SCORERS == ("cosine", "rca") + B.PM_NAMES + ("B", "B0", "B1", "aff_fused", "aff_cf", "r1_fused",
                                                               "r1_cf")
    assert P.NESTED == ("B", "B0", "B1") and P.TERM_OF == {"B": "t6u_B", "B0": "t6u_B0", "B1": "t6u_B1"}
    assert P.MEAN_R1_TARGETS == {"B": 18.341064453125, "B0": 18.436686197916664, "B1": 18.804931640625}
    assert P.LAMBDA_SCORERS == ("rca",) + B.PM_NAMES


# ---------------------------------------------------------------- the parity mapping (tune half h -> parity 1 - h)

def test_nested_parity_mapping(held):
    """Different picks per half: the pick of tune half 0 scored exactly the parity-1 episodes, the pick of tune
    half 1 the parity-0 episodes."""
    b = held
    pick = {0: [0.0, 0.0], 1: [16.0, 16.0]}
    got = S.frozen_nested(b.cos, b.t_n1u, b.t6u_B, pick, b.parity)
    s0 = nested_scores(b.cos, b.t_n1u, b.t6u_B, 0.0, 0.0)
    s1 = nested_scores(b.cos, b.t_n1u, b.t6u_B, 16.0, 16.0)
    odd, even = b.parity == 1, b.parity == 0
    assert scores_bits(rows_of(got, odd), rows_of(s0, odd))
    assert scores_bits(rows_of(got, even), rows_of(s1, even))
    assert not np.array_equal(s0["a"]["i2t"][even], s1["a"]["i2t"][even])     # the two picks differ on every half
    assert not np.array_equal(s0["a"]["i2t"][odd], s1["a"]["i2t"][odd])
    assert all(got[c][d].dtype == np.float32 for c in CONDITIONS for d in DIRECTIONS)


def test_nested_frozen_equals_crossfit_with_its_own_picks(sel42):
    b = sel42
    s, p = crossfit_condition_free(b.cos, b.t_n1u, b.t6u_B0, b.parity)
    assert scores_bits(S.frozen_nested(b.cos, b.t_n1u, b.t6u_B0, p, b.parity), s)


def test_fused_parity_mapping(held):
    b = held
    term = b.pm_terms["rca"]
    got = S.frozen_fused(b.cos, term, {0: 0.0, 1: float("inf")}, b.parity)
    odd, even = b.parity == 1, b.parity == 0
    assert scores_bits(rows_of(got, odd), rows_of(fused_scores(b.cos, term, 0.0), odd))         # z(cos)
    assert scores_bits(rows_of(got, even), rows_of(fused_scores(b.cos, term, float("inf")), even))  # z(term)
    assert not np.array_equal(got["a"]["i2t"][odd], fused_scores(b.cos, term, float("inf"))["a"]["i2t"][odd])


def test_fused_frozen_equals_crossfit_lambda_with_its_own_picks(sel42):
    b = sel42
    s, p = crossfit_lambda(b.cos, b.pm_terms["probe"], b.parity)
    assert scores_bits(S.frozen_fused(b.cos, b.pm_terms["probe"], p, b.parity), s)


def test_cells_parity_mapping(held, reader_gates):
    """Fused cell 0 (tau_0, 0, 0: z(B) alone) for tune half 0 and cell 223 (tau_3, 16, 16) for tune half 1."""
    b = held
    Bs = b.t6u_B                                   # any condition-free float32 base
    T, g = reader_gates.rd["T"], reader_gates.g_aff
    fused, _ = S.frozen_cells(Bs, b.parity, T, g, (0, 223), (149, 10))
    zB, zT = _zdict(Bs), _zdict(T)
    gated3 = RC.gated_terms(zT, g[3])
    odd, even = b.parity == 1, b.parity == 0
    for c in CONDITIONS:
        for d in DIRECTIONS:
            assert np.array_equal(fused[c][d][odd], zB[c][d].numpy()[odd].astype(np.float64))
            want = _combine(zB, zB, gated3, 16.0, 16.0)
            assert np.array_equal(fused[c][d][even], want[c][d][even].astype(np.float64))


def test_frozen_cells_equal_round3_score_frozen(held, reader_gates):
    """The r6 copy gives round 3's score_frozen bit for bit (AFF and R1 cells, on a frozen B)."""
    b = held
    Bs = S.frozen_nested(b.cos, b.t_n1u, b.t6u_B, NESTED_PICKS["B"], b.parity)
    T = reader_gates.rd["T"]
    for who, g in (("aff", reader_gates.g_aff), ("r1", reader_gates.g_r1)):
        fused, cf = S.frozen_cells(Bs, b.parity, T, g, S.CELLS[who]["fused"], S.CELLS[who]["cf"])
        ref = RF.score_frozen(SimpleNamespace(B=Bs, parity=b.parity), T, g, S.CELLS[who]["fused"], S.CELLS[who]["cf"])
        assert pa_bits(per_anchor(fused), ref["fused"]) and pa_bits(per_anchor(cf), ref["cf"]), who
        assert (per_anchor(cf)["gain"] == 0).all()


# ---------------------------------------------------------------- CF stays condition-free (hazard 10)

def test_cf_term_that_differs_between_conditions_raises(held, reader_gates, tmp_path, monkeypatch):
    """AFF's gated term where CF expects G_cf (the smoke's wiring mutation, rule section 6 item 7): the per-cell
    condition-free assertion fires; with it deleted, the assembly goes through."""
    b = held
    Bs = S.frozen_nested(b.cos, b.t_n1u, b.t6u_B, NESTED_PICKS["B"], b.parity)
    T, g = reader_gates.rd["T"], reader_gates.g_aff
    monkeypatch.setattr(S, "cf_terms", lambda gated: gated)
    with pytest.raises(ValueError, match="condition-free"):
        S.frozen_cells(Bs, b.parity, T, g, S.CELLS["aff"]["fused"], S.CELLS["aff"]["cf"])
    mut = mutant(tmp_path, "r6_score.py", "cf_cell")
    monkeypatch.setattr(mut, "cf_terms", lambda gated: gated)
    _, cf = mut.frozen_cells(Bs, b.parity, T, g, S.CELLS["aff"]["fused"], S.CELLS["aff"]["cf"])
    assert not (per_anchor(cf)["gain"] == 0).all()


def test_cf_gain_guard(held, readers, tmp_path, monkeypatch):
    """score_seed asserts gain == 0 on every episode of CF and R1's counterpart: with the per-cell guard deleted and
    the gated term wired in, the gain guard fires; with both deleted, the scoring goes through."""
    mut = mutant(tmp_path, "r6_score.py", "cf_cell")
    monkeypatch.setattr(mut, "cf_terms", lambda gated: gated)
    with pytest.raises(AssertionError, match="gain"):
        mut.score_seed(held, NESTED_PICKS, lambdas_all(), readers, include_pm=False)
    mut2 = mutant(tmp_path, "r6_score.py", "cf_cell", "cf_gain")
    monkeypatch.setattr(mut2, "cf_terms", lambda gated: gated)
    out = mut2.score_seed(held, NESTED_PICKS, lambdas_all(), readers, include_pm=False)
    assert not (out["aff_cf"]["gain"] == 0).all()


# ---------------------------------------------------------------- score_seed

def test_score_seed_keys_and_shapes(held, scored_held):
    out = scored_held
    assert tuple(out) == S.CORE_SCORERS + S.EXTRA_KEYS
    assert S.EXTRA_KEYS == ("gates", "reader", "cl", "pair_index")
    for k in S.CORE_SCORERS:
        assert tuple(out[k]) == METRICS and all(out[k][m].shape == (N_EP,) and out[k][m].dtype == np.float64
                                                for m in METRICS), k
    for who in ("aff", "r1"):
        assert tuple(out["gates"][who]) == CONDITIONS
        for c in CONDITIONS:
            x = out["gates"][who][c]
            assert x.shape == (4, N_EP) and x.dtype == np.float32 and set(np.unique(x)) <= {0.0, 1.0}
    rd = out["reader"]
    assert tuple(rd) == ("P", "m", "pick")
    for c in CONDITIONS:
        assert rd["P"][c].shape == (N_EP, 3) and rd["P"][c].dtype == np.float64
        assert rd["m"][c].shape == (N_EP,) and rd["pick"][c].dtype == np.int64
    assert np.array_equal(out["cl"], held.cl) and np.array_equal(out["pair_index"], held.pair_index)
    for k in ("cosine", "B", "B0", "B1", "aff_cf", "r1_cf"):
        assert (out[k]["gain"] == 0).all(), k


def test_score_seed_assembly(held, readers, reader_gates, scored_held):
    """Each scorer as the rule's assembly: COS = per_anchor(cos); B, B0, B1 nested with their own picks; RCA fused at
    its lambdas; AFF, CF, R1 the frozen cells on z(B) of the frozen B (hazard 1), not of B0 or cosine."""
    b, out = held, scored_held
    assert pa_bits(out["cosine"], per_anchor(b.cos))
    frozen = {name: S.frozen_nested(b.cos, b.t_n1u, getattr(b, P.TERM_OF[name]), NESTED_PICKS[name], b.parity)
              for name in P.NESTED}
    for name in P.NESTED:
        assert pa_bits(out[name], per_anchor(frozen[name])), name
    assert pa_bits(out["rca"], per_anchor(S.frozen_fused(b.cos, b.pm_terms["rca"], lambdas_all()["rca"], b.parity)))
    T = reader_gates.rd["T"]
    for who, g in (("aff", reader_gates.g_aff), ("r1", reader_gates.g_r1)):
        fused, cf = S.frozen_cells(frozen["B"], b.parity, T, g, S.CELLS[who]["fused"], S.CELLS[who]["cf"])
        assert pa_bits(out[f"{who}_fused"], per_anchor(fused)) and pa_bits(out[f"{who}_cf"], per_anchor(cf)), who
        other, _ = S.frozen_cells(frozen["B0"], b.parity, T, g, S.CELLS[who]["fused"], S.CELLS[who]["cf"])
        assert not pa_bits(out[f"{who}_fused"], per_anchor(other))
        for c in CONDITIONS:
            assert bits(out["gates"][who][c], np.stack([g[t][c] for t in range(4)]))
    for k in ("P", "m", "pick"):
        for c in CONDITIONS:
            assert bits(out["reader"][k][c], reader_gates.rd[k][c])


def test_score_seed_picks_take_string_or_int_halves(held, readers, scored_held):
    as_json = {name: {str(h): v for h, v in p.items()} for name, p in NESTED_PICKS.items()}
    lams = {name: {str(h): ("inf" if math.isinf(v) else v) for h, v in p.items()} for name, p in lambdas_all().items()}
    out = S.score_seed(held, as_json, lams, readers, include_pm=False)
    for k in S.CORE_SCORERS:
        assert pa_bits(out[k], scored_held[k]), k


# ---------------------------------------------------------------- PM before the verdict (rule section 8 item 3)

def test_include_pm_has_no_default(held, readers):
    with pytest.raises(TypeError):
        S.score_seed(held, NESTED_PICKS, lambdas_all(), readers)


def test_include_pm_must_be_a_bool(held, readers, tmp_path):
    with pytest.raises(AssertionError, match="include_pm"):
        S.score_seed(held, NESTED_PICKS, lambdas_all(), readers, include_pm=None)
    mut = mutant(tmp_path, "r6_score.py", "include_pm_bool")
    out = mut.score_seed(held, NESTED_PICKS, {"rca": lambdas_all()["rca"]}, readers, include_pm=None)
    assert not set(B.PM_NAMES) & set(out)


def test_held_without_pm_scores_no_pm_key(held, readers):
    """The held pass before the verdict: no PM key is scored, no PM term or lambda is read."""
    b = SimpleNamespace(**{**vars(held), "pm_terms": PMPoison(held.pm_terms)})
    out = S.score_seed(b, NESTED_PICKS, {"rca": lambdas_all()["rca"]}, readers, include_pm=False)
    assert tuple(out) == S.CORE_SCORERS + S.EXTRA_KEYS
    assert not set(B.PM_NAMES) & set(out)
    with pytest.raises(Reached):                       # the poison is live: a PM read would have raised
        b.pm_terms["probe"]


def test_held_pm_refused_before_the_verdict(held, readers, tmp_path, monkeypatch):
    monkeypatch.setattr(S, "VERDICT", tmp_path / "held_verdict.json")
    with pytest.raises(AssertionError, match="held_verdict.json"):
        S.score_seed(held, NESTED_PICKS, lambdas_all(), readers, include_pm=True)
    mut = mutant(tmp_path, "r6_score.py", "pm_before_verdict")
    monkeypatch.setattr(mut, "VERDICT", tmp_path / "held_verdict.json")
    out = mut.score_seed(held, NESTED_PICKS, lambdas_all(), readers, include_pm=True)
    assert set(B.PM_NAMES) <= set(out)


def test_held_pm_after_the_verdict_and_selection_pm(held, sel42, readers, tmp_path, monkeypatch):
    verdict = tmp_path / "held_verdict.json"
    verdict.write_text("{}")
    monkeypatch.setattr(S, "VERDICT", verdict)
    out = S.score_seed(held, NESTED_PICKS, lambdas_all(), readers, include_pm=True)
    assert tuple(out) == S.ALL_SCORERS + S.EXTRA_KEYS
    for name in B.PM_NAMES:
        want = per_anchor(S.frozen_fused(held.cos, held.pm_terms[name], lambdas_all()[name], held.parity))
        assert pa_bits(out[name], want), name
    monkeypatch.setattr(S, "VERDICT", tmp_path / "absent.json")
    out = S.score_seed(sel42, NESTED_PICKS, lambdas_all(), readers, include_pm=True)   # selection: no verdict needed
    assert tuple(out) == S.ALL_SCORERS + S.EXTRA_KEYS


# ---------------------------------------------------------------- score_seed's other guards

def test_admitted_guard(held, readers, tmp_path):
    b = SimpleNamespace(**{**vars(held), "seed": 49})              # round 3's test seed: not a round-6 seed
    with pytest.raises(AssertionError, match="admitted"):
        S.score_seed(b, NESTED_PICKS, lambdas_all(), readers, include_pm=False)
    mut = mutant(tmp_path, "r6_score.py", "admitted")
    assert "aff_fused" in mut.score_seed(b, NESTED_PICKS, lambdas_all(), readers, include_pm=False)


def test_parity_guard(held, readers, tmp_path):
    b = SimpleNamespace(**{**vars(held), "parity": 1 - held.parity})
    with pytest.raises(AssertionError, match="parity"):
        S.score_seed(b, NESTED_PICKS, lambdas_all(), readers, include_pm=False)
    mut = mutant(tmp_path, "r6_score.py", "parity")
    assert "aff_fused" in mut.score_seed(b, NESTED_PICKS, lambdas_all(), readers, include_pm=False)


# ---------------------------------------------------------------- pick conventions

def test_as_nested_pick():
    assert P.as_nested_pick({"0": [4, 16], "1": [0, 0.5]}, "B") == {0: [4.0, 16.0], 1: [0.0, 0.5]}
    assert P.as_nested_pick({0: (0.5, 0.25), 1: [16.0, 16.0]}, "B") == {0: [0.5, 0.25], 1: [16.0, 16.0]}
    for bad in ({0: [4, 16]}, {0: [4, 16], 1: [0, 0.5], 2: [0, 0]}, {"0": [4, 16], 0: [4, 16], 1: [0, 0]},
                {0: [4, 16, 1], 1: [0, 0]}):
        with pytest.raises(AssertionError):
            P.as_nested_pick(bad, "B")


def test_nested_cell_guard(tmp_path):
    with pytest.raises(AssertionError, match="nested cell"):
        P.as_nested_pick({0: [3.0, 16.0], 1: [0.0, 0.0]}, "B")          # 3 is not in NESTED_U
    mut = mutant(tmp_path, "r6_picks.py", "nested_cell")
    assert mut.as_nested_pick({0: [3.0, 16.0], 1: [0.0, 0.0]}, "B") == {0: [3.0, 16.0], 1: [0.0, 0.0]}


def test_parse_lambda_picks_reads_inf():
    rec = {"scorers": {name: {"lambda_picks": {"0": 0.25, "1": 16.0}} for name in P.LAMBDA_SCORERS}}
    rec["scorers"]["rca"]["lambda_picks"] = {"0": "inf", "1": 0.5}
    rec["scorers"]["wang"]["lambda_picks"] = {"0": 32.0, "1": "inf"}
    got = P.parse_lambda_picks(rec)
    assert tuple(got) == P.LAMBDA_SCORERS
    assert got["rca"] == {0: float("inf"), 1: 0.5} and got["wang"] == {0: 32.0, 1: float("inf")}
    assert got["probe"] == {0: 0.25, 1: 16.0}
    for name in P.LAMBDA_SCORERS:
        assert set(got[name]) == {0, 1} and all(type(v) is float for v in got[name].values())
    for broken in ({"0": 0.25}, {"0": 0.25, "1": 0.5, "2": 0.5}, {0: 0.25, 1: 0.5}, {"0": "Infinity", "1": 0.5}):
        r = json.loads(json.dumps(rec))
        r["scorers"]["probe"]["lambda_picks"] = broken
        with pytest.raises(AssertionError):
            P.parse_lambda_picks(r)


def test_lambda_grid_guard(tmp_path):
    rec = {"scorers": {name: {"lambda_picks": {"0": 0.25, "1": 0.5}} for name in P.LAMBDA_SCORERS}}
    rec["scorers"]["xing"]["lambda_picks"] = {"0": 3.0, "1": 0.5}              # not a lambda of the grid
    with pytest.raises(AssertionError, match="grid"):
        P.parse_lambda_picks(rec)
    mut = mutant(tmp_path, "r6_picks.py", "lambda_grid")
    assert mut.parse_lambda_picks(rec)["xing"] == {0: 3.0, 1: 0.5}


def test_frozen_lambdas_from_baselines_seed42():
    got = P.frozen_lambdas()
    want = {"rca": (0.5, 0.5), "diag": (0.0, 0.25), "diag_relu": (0.0, 0.25), "bilinear": (0.0, 0.25),
            "kissme": (0.0, 0.25), "xing": (0.25, 0.25), "wang": (1.0, 1.0), "probe": (0.25, 0.5),
            "tip": (0.0, 0.25), "value_prototype": (0.0, 0.0)}
    assert tuple(got) == P.LAMBDA_SCORERS
    assert {k: (v[0], v[1]) for k, v in got.items()} == want


def test_lambda_rerun_guard(sel42, tmp_path):
    """assert_lambda_convention reruns crossfit_lambda on the seed-42 bundle and compares half by half; a disagreeing
    stored pick raises, and with the guard deleted it goes through."""
    b = sel42
    lams = {name: crossfit_lambda(b.cos, b.pm_terms[name], b.parity)[1] for name in P.LAMBDA_SCORERS}
    lams = {name: {h: float(v) for h, v in p.items()} for name, p in lams.items()}
    assert P.assert_lambda_convention(b, lams) == {name: True for name in P.LAMBDA_SCORERS}
    bad = {name: dict(p) for name, p in lams.items()}
    bad["kissme"][1] = 64.0 if bad["kissme"][1] != 64.0 else 0.0
    with pytest.raises(AssertionError, match="kissme"):
        P.assert_lambda_convention(b, bad)
    mut = mutant(tmp_path, "r6_picks.py", "lambda_rerun")
    mut.assert_lambda_convention(b, bad)


def test_seed42_guard(held, sel42, tmp_path):
    for fn in (P.crossfit_seed42, lambda b: P.assert_lambda_convention(b, {})):
        with pytest.raises(AssertionError, match="seed-42"):
            fn(held)
    mut = mutant(tmp_path, "r6_picks.py", "seed42")
    picks, _ = mut.crossfit_seed42(held)
    assert set(picks) == {"B", "B0", "B1", "mean_r1"}


# ---------------------------------------------------------------- picks_seed42 and its file

def test_crossfit_seed42_and_frozen_equal(sel42):
    """picks_seed42 keeps crossfit_condition_free's picks per tune half and its mean R@1 (x100, exact); the frozen
    assembly with those picks equals the cross-fit element for element."""
    b = sel42
    picks, scores = P.crossfit_seed42(b)
    assert P.picks_seed42(b) == picks
    assert tuple(picks) == ("B", "B0", "B1", "mean_r1")
    for name in P.NESTED:
        s, p = crossfit_condition_free(b.cos, b.t_n1u, getattr(b, P.TERM_OF[name]), b.parity)
        assert picks[name] == {0: p[0], 1: p[1]} and scores_bits(scores[name], s)
        assert picks["mean_r1"][name] == 100 * float(np.mean(per_anchor(s)["r1"]))
    assert P.frozen_equals_crossfit(b, picks, scores) == {name: True for name in P.NESTED}
    wrong = {**picks, "B": {0: picks["B"][1], 1: picks["B"][0]}}
    if wrong["B"] != picks["B"]:
        assert P.frozen_equals_crossfit(b, wrong, scores)["B"] is False


def _targets_of(picks):
    return dict(picks["mean_r1"])


def test_picks_record_and_load(sel42, tmp_path, monkeypatch):
    picks, _ = P.crossfit_seed42(sel42)
    monkeypatch.setattr(P, "MEAN_R1_TARGETS", _targets_of(picks))
    rec = P.picks_record(picks, {"frozen_equals_crossfit": {n: True for n in P.NESTED}})
    assert rec["passed"] is True
    for name in P.NESTED:
        assert rec[name] == {"0": picks[name][0], "1": picks[name][1]}
    assert rec["mean_r1"] == picks["mean_r1"] and rec["module_sha256"] == R.r6_module_shas()
    assert re.fullmatch(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d", rec["time"])
    path = tmp_path / "picks_seed42.json"
    path.write_text(json.dumps(rec, indent=1))
    loaded = P.load_picks(path)
    assert loaded == {name: picks[name] for name in P.NESTED}


def test_picks_target_guard(sel42, tmp_path, monkeypatch):
    """load_picks refuses a record whose mean R@1 is not the rule's target exactly (one ulp off), or not passed."""
    picks, _ = P.crossfit_seed42(sel42)
    targets = _targets_of(picks)
    monkeypatch.setattr(P, "MEAN_R1_TARGETS", targets)
    rec = P.picks_record(picks, {})
    off = json.loads(json.dumps(rec))
    off["mean_r1"]["B1"] = math.nextafter(off["mean_r1"]["B1"], 0.0)
    path = tmp_path / "off.json"
    path.write_text(json.dumps(off))
    with pytest.raises(AssertionError, match="target"):
        P.load_picks(path)
    failed = json.loads(json.dumps(rec))
    failed["passed"] = False
    (tmp_path / "failed.json").write_text(json.dumps(failed))
    with pytest.raises(AssertionError, match="target"):
        P.load_picks(tmp_path / "failed.json")
    mut = mutant(tmp_path, "r6_picks.py", "picks_target")
    monkeypatch.setattr(mut, "MEAN_R1_TARGETS", targets)
    assert mut.load_picks(path)["B"] == picks["B"]


# ---------------------------------------------------------------- the runner

def test_run_writes_the_record_or_exits_3(sel42, tmp_path, monkeypatch, capsys):
    b = sel42
    picks, _ = P.crossfit_seed42(b)
    lams = {name: {h: float(v) for h, v in crossfit_lambda(b.cos, b.pm_terms[name], b.parity)[1].items()}
            for name in P.LAMBDA_SCORERS}
    monkeypatch.setattr(P, "frozen_lambdas", lambda: lams)
    out = tmp_path / "picks_seed42.json"
    # the real targets cannot hold on synthetic data: exit 3, the canonical file not written, the failed record kept
    assert RP.run(b, out) == RP.EXIT_DIFF == 3
    assert not out.exists()
    failed = json.loads((tmp_path / "picks_seed42_failed.json").read_text())
    assert failed["passed"] is False and failed["checks"]["mean_r1_equal_target"] == {n: False for n in P.NESTED}
    # with the synthetic means as targets: exit 0 and the record
    monkeypatch.setattr(P, "MEAN_R1_TARGETS", _targets_of(picks))
    assert RP.run(b, out) == 0
    rec = json.loads(out.read_text())
    assert rec["passed"] is True and rec["checks"]["lambda_convention"] == {n: True for n in P.LAMBDA_SCORERS}
    assert rec["checks"]["frozen_equals_crossfit"] == {n: True for n in P.NESTED}
    assert P.load_picks(out) == {name: picks[name] for name in P.NESTED}
    printed = capsys.readouterr().out
    for name in P.NESTED:                                          # no metric printed
        assert repr(picks["mean_r1"][name]) not in printed and f"{picks['mean_r1'][name]:.2f}" not in printed
    # a lambda disagreement is a difference too
    bad = {name: dict(p) for name, p in lams.items()}
    bad["rca"][0] = 64.0 if bad["rca"][0] != 64.0 else 0.0
    monkeypatch.setattr(P, "frozen_lambdas", lambda: bad)
    assert RP.run(b, tmp_path / "again.json") == 3 and not (tmp_path / "again.json").exists()


def test_exit_code_guard(sel42, tmp_path, monkeypatch):
    b = sel42
    lams = {name: {h: float(v) for h, v in crossfit_lambda(b.cos, b.pm_terms[name], b.parity)[1].items()}
            for name in P.LAMBDA_SCORERS}
    monkeypatch.setattr(P, "frozen_lambdas", lambda: lams)
    mut = mutant(tmp_path, "run_r6_picks.py", "exit_code")
    assert mut.run(b, tmp_path / "picks_seed42.json") == 0             # the difference goes through unnoticed
