"""Tests of the shared scoring changes of ticket 20a (rule sections 4.1, 4.2, 5, 7.2, 7.4): cells_from_record, taus_of,
score_seed(cells=, taus=, secondary=, verdict=) and r9_stats' secondary=False. Synthetic bundles of the real shapes
(test_r9_score.make_bundle); no held row. Guards (`# guard:<name>` in r9_score.py) each get a mutation test
(copy of test_r9_score.mutant). The "default arguments" tests compare with the modules as committed in e54b8b2 (before
20a), loaded from git.

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python -m pytest -q -p no:cacheprovider \
        src/test/20261202_r9_scope/test_r9_held_shared.py
"""
import ast
import importlib.util
import itertools
import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r9_common as R  # noqa: E402  (first, before anything that imports src)
import r9_backbone as BB  # noqa: E402
import r9_bundle as B  # noqa: E402
import r9_context as X  # noqa: E402
import r9_picks as P  # noqa: E402
import r9_score as S  # noqa: E402
import r9_stats as ST  # noqa: E402
import test_r9_score as T  # noqa: E402  (synthetic bundle, picks, lambdas, bit comparison)
import test_r9_stats as TS  # noqa: E402  (synthetic per-seed score dicts)

import numpy as np  # noqa: E402

from src.eval.aspect_metrics import CONDITIONS, METRICS, per_anchor  # noqa: E402

RF, R3, F2 = R.RF, R.R3, R.F2
BASE = "e54b8b2"                                  # the commit before ticket 20a
_COUNT = itertools.count()
A28_SEED = 55


# ---------------------------------------------------------------- helpers

def mutant(tmp_path, module, *guards):
    """Import a copy of ``module`` (file name in HERE) with its `# guard:<g>` statements replaced by `pass`, for each
    g in ``guards`` (copy of test_r9_score.mutant)."""
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
    return _import(tmp_path, f"{Path(module).stem}_mut{next(_COUNT)}", src)


def _import(tmp_path, name, src):
    path = tmp_path / f"{name}.py"
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


def old_module(tmp_path, module):
    """``module`` as committed in BASE (before ticket 20a), imported from tmp_path."""
    rel = (HERE / module).relative_to(subprocess.check_output(
        ["git", "-C", str(HERE), "rev-parse", "--show-toplevel"], text=True).strip())
    src = subprocess.check_output(["git", "-C", str(HERE), "show", f"{BASE}:{rel}"], text=True)
    return _import(tmp_path, f"{Path(module).stem}_{BASE}_{next(_COUNT)}", src)


def same(x, y) -> bool:
    """Bit-equal nested dicts of arrays."""
    if isinstance(x, dict):
        return isinstance(y, dict) and list(x) == list(y) and all(same(x[k], y[k]) for k in x)
    return T.bits(x, y)


@pytest.fixture(scope="module")
def readers():
    return B.load_readers()


@pytest.fixture(scope="module")
def held():
    return T.make_bundle()


@pytest.fixture(scope="module")
def sel42():
    return T.make_bundle(seed=R.DEV_SEED, mode="selection", rng_seed=7)


@pytest.fixture
def a28(held):
    """A stand-in a28 bundle (the synthetic held bundle relabelled; ("a28", 55) is in X.ADMITTED since ticket 18)."""
    return SimpleNamespace(**{**vars(held), "mode": "a28", "seed": A28_SEED})


@pytest.fixture
def v6(tmp_path):
    """An existing H6 verdict file: the held scoring that is allowed after the verdict."""
    path = tmp_path / "held_verdict_H6.json"
    path.write_text("{}")
    return path


def run(bundle, readers, **kw):
    kw.setdefault("include_pm", False)
    return S.score_seed(bundle, T.NESTED_PICKS, T.lambdas_all(), readers, **kw)


# ---------------------------------------------------------------- bookkeeping

def test_every_guard_of_r9_score_has_a_mutation_test_in_this_file_or_in_test_r9_score():
    names = {g for g in re.findall(r"# guard:([a-z_0-9]+)", (HERE / "r9_score.py").read_text())}
    new = {"secondary_bool", "secondary_no_pm", "read_verdict_path", "read_verdict_name", "secondary_before_verdict"}
    assert new <= names and "a28_verdict_path" not in names
    text = Path(__file__).read_text()
    covered = set()
    for args in re.findall(r'mutant\(tmp_path, "r9_score\.py", ([^)]*)\)', text):
        covered |= set(re.findall(r'"([a-z_0-9]+)"', args))
    assert new | {"pm_before_verdict"} <= covered, sorted((new | {"pm_before_verdict"}) - covered)


# ---------------------------------------------------------------- cells_from_record

def test_cells_from_record_on_the_real_clip_file_gives_the_pinned_cells():
    rec = P.read_cells(R.HERE / "cells_seed42_clip.json")
    got = S.cells_from_record(rec)
    assert got == S.CELLS
    assert all(type(c) is int for who in got.values() for pair in who.values() for c in pair)
    assert got == {"aff": {"fused": (39, 119), "cf": (149, 10)}, "r1": {"fused": (116, 119), "cf": (58, 123)}}


def test_cells_from_record_refuses_bad_records():
    rec = json.loads((R.HERE / "cells_seed42_clip.json").read_text())
    for key, bad in (("aff", [39]), ("cf", [149, 224]), ("r1_fused", [-1, 3]), ("r1_counterpart", [58.0, 123]),
                     ("aff", "39,119"), ("cf", [True, 10])):
        with pytest.raises(AssertionError, match=key):
            S.cells_from_record({**rec, key: bad})
    with pytest.raises(AssertionError, match="keys"):
        S.cells_from_record({k: v for k, v in rec.items() if k != "r1_counterpart"})


# ---------------------------------------------------------------- taus_of

def test_taus_of_clip_is_round_threes_taus():
    assert S.taus_of("clip_b32") == R3.assert_taus() == tuple(R3.TAUS)


def _new_backbone_pins(tmp_path, monkeypatch, taus, wrong_path=False):
    rc = tmp_path / "rc_tau.json"
    rc.write_text(json.dumps({"taus": taus}))
    clip = BB.parts("clip_b32")
    resolved = {**clip, "rc_tau": rc}
    monkeypatch.setattr(BB, "parts", lambda b: clip if wrong_path else resolved)
    manifest = tmp_path / "pins_new.json"
    manifest.write_text(json.dumps({"input_sha256": {S.RC_TAU_REL: R.sha256_file(rc)}}))
    monkeypatch.setattr(R, "pins_path", lambda b: manifest)
    # input_paths replaces CLIP's path by the backbone's part, so the patched parts decides what is pinned
    monkeypatch.setattr(R, "input_paths", lambda b: {**{r: R.TEST / r for r in R.INPUT_SHA256}, S.RC_TAU_REL: rc})
    return rc


def test_taus_of_new_backbone_reads_its_pinned_rc_tau(tmp_path, monkeypatch):
    taus = [0.01, 0.2, 0.45, 0.8]
    _new_backbone_pins(tmp_path, monkeypatch, taus)
    got = S.taus_of("qwen3vl_emb_2b")
    assert got == tuple(taus) and got != tuple(R3.TAUS)


def test_taus_of_new_backbone_refuses_a_changed_file_or_bad_taus(tmp_path, monkeypatch):
    rc = _new_backbone_pins(tmp_path, monkeypatch, [0.01, 0.2, 0.45, 0.8])
    rc.write_text(json.dumps({"taus": [0.01, 0.2, 0.45, 0.9]}))               # edited after the pin
    with pytest.raises(SystemExit, match="SHA-256"):
        S.taus_of("qwen3vl_emb_2b")
    for bad in ([0.01, 0.2, 0.45], [0.3, 0.2, 0.45, 0.8], [0.1, 0.2, 0.3, 1], [0.1, 0.2, 0.3, float("nan")],
                [0.1, 0.2, 0.3, float("inf")]):
        _new_backbone_pins(tmp_path, monkeypatch, bad)
        with pytest.raises(AssertionError, match="taus"):
            S.taus_of("qwen3vl_emb_2b")


def test_taus_of_hashes_and_parses_the_same_bytes(tmp_path, monkeypatch):
    """L2: one read of the file; its bytes are hashed and the same bytes parsed (no second read_text or hash pass)."""
    rc = _new_backbone_pins(tmp_path, monkeypatch, [0.01, 0.2, 0.45, 0.8])
    reads = []
    real_bytes, real_text, real_sha = Path.read_bytes, Path.read_text, R.sha256_file

    def counting(self):
        reads.append(self)
        return real_bytes(self)

    def text(self, *a, **k):
        assert self != rc, "second read of the file"
        return real_text(self, *a, **k)

    def sha(path):
        assert Path(path) != rc, "second read of the file"
        return real_sha(path)

    monkeypatch.setattr(Path, "read_bytes", counting)
    monkeypatch.setattr(Path, "read_text", text)
    monkeypatch.setattr(R, "sha256_file", sha)
    assert S.taus_of("qwen3vl_emb_2b") == (0.01, 0.2, 0.45, 0.8)
    assert [p for p in reads if p == rc] == [rc]


def test_taus_of_refuses_a_path_that_is_not_the_recipe_part(tmp_path, monkeypatch):
    _new_backbone_pins(tmp_path, monkeypatch, [0.01, 0.2, 0.45, 0.8], wrong_path=True)
    with pytest.raises(AssertionError, match="recipe part"):
        S.taus_of("qwen3vl_emb_2b")


# ---------------------------------------------------------------- score_seed: cells and taus

def test_score_seed_uses_the_given_cells_and_taus(held, readers, v6):
    cells = S.cells_from_record({"aff": [119, 39], "cf": [10, 149], "r1_fused": [119, 116],
                                 "r1_counterpart": [123, 58]})
    taus = (0.01, 0.2, 0.45, 0.8)
    default = run(held, readers, verdict=v6)
    out = run(held, readers, cells=cells, taus=taus, verdict=v6)
    rd = RF.reader(held, readers)
    gates = RF.gates_aff(rd["m"], rd["pick"], taus)
    want_gates = {c: np.stack([gates[t][c] for t in range(4)]) for c in CONDITIONS}
    assert same(out["gates"]["aff"], want_gates) and not same(default["gates"]["aff"], want_gates)
    frozen_b = S.frozen_nested(held.cos, held.t_n1u, held.t6u_B, T.NESTED_PICKS["B"], held.parity)
    fused, cf = S.frozen_cells(frozen_b, held.parity, rd["T"], gates, cells["aff"]["fused"], cells["aff"]["cf"])
    assert T.pa_bits(out["aff_fused"], per_anchor(fused)) and T.pa_bits(out["aff_cf"], per_anchor(cf))
    g1 = RF.gates_r1(rd["m"], taus)
    r1, _ = S.frozen_cells(frozen_b, held.parity, rd["T"], g1, cells["r1"]["fused"], None)
    assert T.pa_bits(out["r1_fused"], per_anchor(r1))
    assert not T.pa_bits(out["aff_fused"], default["aff_fused"])
    assert T.pa_bits(out["cosine"], default["cosine"]) and T.pa_bits(out["B"], default["B"])


def test_score_seed_refuses_bad_taus_and_incomplete_cells(held, readers, v6):
    for bad in ((0.1, 0.2, 0.3), (0.3, 0.2, 0.1, 0.0), [1, 2, 3, 4], (0.1, 0.2, 0.3, float("inf"))):
        with pytest.raises(AssertionError, match="taus"):
            run(held, readers, taus=bad, verdict=v6)
    with pytest.raises(AssertionError, match="cells"):
        run(held, readers, cells={"aff": {"fused": (39, 119), "cf": (149, 10)}}, verdict=v6)
    out = run(held, readers, cells={"aff": S.CELLS["aff"]}, secondary=False, verdict=v6)   # aff alone is enough
    assert tuple(out) == S.PRIMARY_SCORERS + S.EXTRA_KEYS


# ---------------------------------------------------------------- score_seed: secondary=False

def test_secondary_false_scores_only_what_p1_to_p7_read(held, readers, v6):
    poisoned = SimpleNamespace(**{**vars(held), "pm_terms": T.PMPoison(held.pm_terms)})
    del poisoned.t6u_B1                                   # B1's term must not be read
    picks = {k: v for k, v in T.NESTED_PICKS.items() if k != "B1"}   # nor its pick
    out = S.score_seed(poisoned, picks, {"rca": T.lambdas_all()["rca"]}, readers, include_pm=False, secondary=False,
                       verdict=v6)
    assert tuple(out) == S.PRIMARY_SCORERS + S.EXTRA_KEYS
    assert S.PRIMARY_SCORERS == ("cosine", "rca", "B", "B0", "aff_fused", "aff_cf")
    assert tuple(out["gates"]) == ("aff",)
    full = run(held, readers, verdict=v6)
    for k in S.PRIMARY_SCORERS:
        assert T.pa_bits(out[k], full[k]), k
    assert same(out["gates"]["aff"], full["gates"]["aff"]) and same(out["reader"], full["reader"])
    assert (np.asarray(out["aff_cf"]["gain"]) == 0).all() and (np.asarray(out["B"]["gain"]) == 0).all()


def test_secondary_false_refuses_include_pm(held, sel42, readers, tmp_path):
    for b in (sel42, held):
        with pytest.raises(AssertionError, match="include_pm must be False"):
            run(b, readers, include_pm=True, secondary=False)
    mut = mutant(tmp_path, "r9_score.py", "secondary_no_pm")
    with pytest.raises(KeyError):                         # the guard gone: the run goes on and breaks later
        mut.score_seed(sel42, T.NESTED_PICKS, T.lambdas_all(), readers, include_pm=True, secondary=False)


def test_secondary_must_be_a_bool(held, readers, tmp_path, v6):
    for bad in (None, 0, "no"):
        with pytest.raises(AssertionError, match="secondary must be"):
            run(held, readers, secondary=bad)
    mut = mutant(tmp_path, "r9_score.py", "secondary_bool")
    out = mut.score_seed(held, T.NESTED_PICKS, T.lambdas_all(), readers, include_pm=False, secondary=None, verdict=v6)
    assert tuple(out) == S.PRIMARY_SCORERS + S.EXTRA_KEYS


# ---------------------------------------------------------------- score_seed: the verdict file

def test_held_without_a_verdict_path_is_refused_whatever_the_other_arguments(held, readers, tmp_path):
    """Both read modes name their read's verdict file (rule 7.2, 7.4). Round 6's one-file default is gone: a held run
    that copies round 6's call (secondary left True) can no longer reach B1, R1 or PM before a verdict."""
    for kw in ({}, {"include_pm": True}, {"secondary": False}, {"verdict": None}):
        with pytest.raises(AssertionError, match="needs the read's held_verdict_<Hn>.json path as verdict="):
            run(held, readers, **kw)
    mut = mutant(tmp_path, "r9_score.py", "read_verdict_path")
    out = mut.score_seed(held, T.NESTED_PICKS, T.lambdas_all(), readers, include_pm=False, secondary=False)
    assert tuple(out) == S.PRIMARY_SCORERS + S.EXTRA_KEYS             # the guard gone: it goes through


def test_held_with_a_verdict_path_scores_no_pm_b1_or_r1_before_it(held, readers, tmp_path):
    v = tmp_path / "held_verdict_H6.json"
    with pytest.raises(AssertionError, match="PM metric on held episodes before held_verdict_H6.json"):
        run(held, readers, include_pm=True, verdict=v)
    with pytest.raises(AssertionError, match="no B1 or R1 metric on held episodes before held_verdict_H6.json"):
        run(held, readers, verdict=v)
    out = run(held, readers, secondary=False, verdict=v)             # the pre-verdict pass: P1 to P7 only
    assert tuple(out) == S.PRIMARY_SCORERS + S.EXTRA_KEYS
    v.write_text("{}")
    assert tuple(run(held, readers, verdict=v)) == S.CORE_SCORERS + S.EXTRA_KEYS
    assert tuple(run(held, readers, include_pm=True, verdict=v)) == S.ALL_SCORERS + S.EXTRA_KEYS


def test_held_verdict_file_of_another_read_does_not_count(held, readers, tmp_path):
    (tmp_path / "held_verdict_H7.json").write_text("{}")             # H7's file exists, H6's does not
    with pytest.raises(AssertionError, match="held_verdict_H6.json"):
        run(held, readers, include_pm=True, verdict=tmp_path / "held_verdict_H6.json")


def test_pm_before_verdict_guard_held_and_a28(held, a28, readers, tmp_path):
    mut = mutant(tmp_path, "r9_score.py", "pm_before_verdict")
    for b, name in ((held, "held_verdict_H6.json"), (a28, "held_verdict_H8.json")):
        v = tmp_path / name
        with pytest.raises(AssertionError, match="PM metric"):
            run(b, readers, include_pm=True, verdict=v)
        # the PM guard alone gone: the B1/R1 guard still refuses a28 and a held run (depth)
        with pytest.raises(AssertionError, match="B1 or R1"):
            mut.score_seed(b, T.NESTED_PICKS, T.lambdas_all(), readers, include_pm=True, verdict=v)
    both = mutant(tmp_path, "r9_score.py", "pm_before_verdict", "secondary_before_verdict")
    for b, name in ((held, "held_verdict_H6.json"), (a28, "held_verdict_H8.json")):
        out = both.score_seed(b, T.NESTED_PICKS, T.lambdas_all(), readers, include_pm=True, verdict=tmp_path / name)
        assert set(B.PM_NAMES) <= set(out)


def test_a28_needs_the_verdict_path(a28, readers, tmp_path):
    with pytest.raises(AssertionError, match="verdict= "):
        run(a28, readers, secondary=False)
    with pytest.raises(AssertionError, match="verdict= "):
        run(a28, readers, secondary=False, verdict=None)
    out = run(a28, readers, secondary=False, verdict=tmp_path / "held_verdict_H8.json")   # absent file: fine pre-verdict
    assert tuple(out) == S.PRIMARY_SCORERS + S.EXTRA_KEYS
    mut = mutant(tmp_path, "r9_score.py", "read_verdict_path")
    out = mut.score_seed(a28, T.NESTED_PICKS, T.lambdas_all(), readers, include_pm=False, secondary=False)
    assert tuple(out) == S.PRIMARY_SCORERS + S.EXTRA_KEYS


def test_verdict_names_a_verdict_file_of_this_reads_family(held, a28, readers, tmp_path):
    """L1: a cheap name check (held: H6, H7; a28: H8 to H10; an optional _reserve). Which read, and that it is this
    one, is ticket 20b's job."""
    good = {"held": ("held_verdict_H6.json", "held_verdict_H7_reserve.json"),
            "a28": ("held_verdict_H8.json", "held_verdict_H10.json", "held_verdict_H9_reserve.json")}
    bad = {"held": ("held_verdict_H8.json", "held_verdict.json", "held_verdict_H6.json.bak", "verdict_H6.json",
                    "held_verdict_H5.json"),
           "a28": ("held_verdict_H6.json", "held_verdict_H11.json", "held_verdict.json", "xheld_verdict_H8.json")}
    mut = mutant(tmp_path, "r9_score.py", "read_verdict_name")
    for b in (held, a28):
        for name in good[b.mode]:
            assert "gates" in run(b, readers, secondary=False, verdict=tmp_path / name)
        for name in bad[b.mode]:
            with pytest.raises(AssertionError, match="verdict file"):
                run(b, readers, secondary=False, verdict=tmp_path / name)
            assert "gates" in mut.score_seed(b, T.NESTED_PICKS, T.lambdas_all(), readers, include_pm=False,
                                             secondary=False, verdict=tmp_path / name)
    run(held, readers, secondary=False, verdict=str(tmp_path / "held_verdict_H6.json"))     # a str path is fine


def test_a28_refuses_b1_and_r1_before_the_verdict_and_scores_them_after(a28, readers, tmp_path):
    v = tmp_path / "held_verdict_H8.json"
    with pytest.raises(AssertionError, match="no B1 or R1 metric on a28 episodes before held_verdict_H8.json"):
        run(a28, readers, verdict=v)
    mut = mutant(tmp_path, "r9_score.py", "secondary_before_verdict")
    out = mut.score_seed(a28, T.NESTED_PICKS, T.lambdas_all(), readers, include_pm=False, verdict=v)
    assert tuple(out) == S.CORE_SCORERS + S.EXTRA_KEYS
    v.write_text("{}")
    assert tuple(run(a28, readers, verdict=v)) == S.CORE_SCORERS + S.EXTRA_KEYS
    assert tuple(run(a28, readers, include_pm=True, verdict=v)) == S.ALL_SCORERS + S.EXTRA_KEYS


def test_held_with_a_verdict_path_secondary_guard_mutant(held, readers, tmp_path):
    v = tmp_path / "held_verdict_H7.json"
    with pytest.raises(AssertionError, match="B1 or R1"):
        run(held, readers, verdict=v)
    mut = mutant(tmp_path, "r9_score.py", "secondary_before_verdict")
    assert tuple(mut.score_seed(held, T.NESTED_PICKS, T.lambdas_all(), readers, include_pm=False, verdict=v)) \
        == S.CORE_SCORERS + S.EXTRA_KEYS


def test_a28_bundle_is_checked_against_the_admitted_table(held, readers, tmp_path):
    b = SimpleNamespace(**{**vars(held), "mode": "a28", "seed": 52})            # 52 is a held seed, not an a28 seed
    assert ("a28", 52) not in X.ADMITTED and ("a28", A28_SEED) in X.ADMITTED
    with pytest.raises(AssertionError, match="admitted"):
        run(b, readers, secondary=False, verdict=tmp_path / "held_verdict_H8.json")


# ---------------------------------------------------------------- default arguments equal today's

def test_default_arguments_give_the_results_of_before_20a(held, sel42, readers, tmp_path, v6):
    """Selection (stage 0a) is untouched. A held bundle no longer runs on defaults (see the refusal test above); with
    its verdict file named and present, the numbers are those of before 20a."""
    old = old_module(tmp_path, "r9_score.py")
    for bundle, pm, kw in ((held, False, {"verdict": v6}), (sel42, True, {})):
        new = S.score_seed(bundle, T.NESTED_PICKS, T.lambdas_all(), readers, include_pm=pm, **kw)
        was = old.score_seed(bundle, T.NESTED_PICKS, T.lambdas_all(), readers, include_pm=pm)
        explicit = S.score_seed(bundle, T.NESTED_PICKS, T.lambdas_all(), readers, include_pm=pm, cells=S.CELLS,
                                taus=R3.assert_taus(), secondary=True, **{"verdict": None, **kw})
        assert list(new) == list(was) == list(explicit)
        assert same(was, new) and same(new, explicit)
    with pytest.raises(AssertionError, match="verdict="):
        S.score_seed(held, T.NESTED_PICKS, T.lambdas_all(), readers, include_pm=False)       # was allowed before 20a


# ---------------------------------------------------------------- r9_stats

def _smoke_scores(drop=()):
    rng = np.random.default_rng(9001)
    n = 3 * R.N_SMOKE
    cl = rng.integers(0, 40, 3 * n).astype(np.int64)
    per_seed = [TS._seed_scores(rng, cl[i * n:(i + 1) * n]) for i in range(3)]
    for s in per_seed:
        for k in drop:
            del s[k]
    return per_seed


def test_check_diffs_secondary_false_has_no_s1_s2_and_needs_no_b1_or_r1():
    full = _smoke_scores()
    short = _smoke_scores(drop=("B1", "r1_fused"))
    both = ST.check_diffs(full)
    only = ST.check_diffs(short, secondary=False)
    assert tuple(both) == ST.CHECKS + ST.SECONDARY and tuple(only) == ST.CHECKS
    for c in ST.CHECKS:
        assert np.array_equal(both[c][0], only[c][0]) and np.array_equal(both[c][1], only[c][1])
    assert tuple(ST.check_diffs(full, secondary=False)) == ST.CHECKS
    with pytest.raises(KeyError):
        ST.check_diffs(short)                              # secondary=True still needs B1 and R1 fused
    for bad in (None, 0, "no"):
        with pytest.raises(ValueError, match="secondary"):
            ST.check_diffs(full, secondary=bad)


def test_pass_record_secondary_false_drops_the_secondary_key_and_changes_nothing_else():
    per_seed = _smoke_scores()
    seeds = list(R.SMOKE_SEEDS)
    extra = TS._extra(seeds)
    rec = ST.pass_record(per_seed, "smoke", seeds, extra)
    rec_f = ST.pass_record(_smoke_scores(drop=("B1", "r1_fused")), "smoke", seeds, extra, secondary=False)
    assert "secondary" in rec and set(rec["secondary"]) == set(ST.SECONDARY)
    assert "secondary" not in rec_f
    drop = ("secondary", "time")
    assert {k: v for k, v in rec.items() if k not in drop} == {k: v for k, v in rec_f.items() if k not in drop}
    assert list(rec_f) == [k for k in rec if k != "secondary"]


def test_pass_record_default_equals_before_20a(tmp_path):
    old = old_module(tmp_path, "r9_stats.py")
    per_seed = _smoke_scores()
    seeds = list(R.SMOKE_SEEDS)
    extra = TS._extra(seeds)
    new = ST.pass_record(per_seed, "smoke", seeds, extra)
    was = old.pass_record(per_seed, "smoke", seeds, extra)
    assert list(new) == list(was)
    assert {k: v for k, v in new.items() if k != "time"} == {k: v for k, v in was.items() if k != "time"}
    d_new, d_old = ST.check_diffs(per_seed), old.check_diffs(per_seed)
    assert list(d_new) == list(d_old) and all(np.array_equal(d_new[c][0], d_old[c][0]) for c in d_new)
