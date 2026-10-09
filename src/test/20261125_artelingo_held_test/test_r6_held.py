"""Tests of run_r6_held.py's held and smoke modes without features (ticket 08; rule section 5 item 5, section 8 items
1 to 3, section 9; contracts section 7). The end-to-end held run on synthetic data of the real shapes (the real
context, bundle and scoring code) is in test_r6_held_synthetic.py, the smoke on real selection rows in
test_r6_held_smoke.py.

Here the heavy steps are stubs that record what happens, in order: setup (an env with a fake held row set of the real
size), the seed bundle (fake episodes reported through on_episodes, then fake posteriors finite on exactly the rows),
the scores (per-anchor dicts of the real shapes: 12,288 episodes per held seed, quarter multiples, CF gain 0), the
episode file, assert_inputs, recorded_hashes and assert_held_eligible. Everything else is the runner's own code:
refusal, ledger, smoke record, seed-42 records, head guard, started file, episode recorder (assert_distinct real),
sensitivity, arrays, pass_record (real) and the writes. Records, ledger and results live in tmp_path with the real
file names. Mutation tests load a copy of the runner from tmp_path with one `# guard:<name>` statement replaced by
`pass` (never in place). Real files read: the rule's inputs (baselines_seed42.json for the lambdas) and the r6
modules' bytes; no feature, no label, no held row.

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python -m pytest -q -p no:cacheprovider \
        src/test/20261125_artelingo_held_test/test_r6_held.py
"""
import ast
import hashlib
import importlib.util
import itertools
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r6_common as R  # noqa: E402  (first, before anything that imports src)
import r6_episodes as E  # noqa: E402
import r6_picks as P  # noqa: E402
import r6_score as S  # noqa: E402
import r6_stats as ST  # noqa: E402
import run_r6_held as RH  # noqa: E402
import run_r6_sensitivity as RS6  # noqa: E402

import numpy as np  # noqa: E402

from src.eval.aspect_metrics import METRICS  # noqa: E402

_COUNT = itertools.count()
DECIMAL = re.compile(r"\d*\.\d+")
COEF = {h: {"img": "a" * 64, "txt": "b" * 64} for h in ("affect", "affect_km", "image", "caption", "csd")}
PICKS = {"B": {"0": [4.0, 16.0], "1": [0.5, 0.25]}, "B0": {"0": [0.0, 1.0], "1": [2.0, 0.0]},
         "B1": {"0": [16.0, 8.0], "1": [1.0, 4.0]}}
HELD = np.arange(160_000, 160_000 + 61_744, dtype=np.int64)            # a fake held row set of the real size
SELECTION = np.arange(10, 10 + 32_413, dtype=np.int64)
SEEDS = R.HELD_SEEDS
LEDGER_HEAD = ["# Held-out ledger", "",
               "| # | Date | Dataset and split | Purpose | Episode / data SHA-256 | Script SHA-256 | Report |",
               "|---|---|---|---|---|---|---|",
               "| H4 | 2026-10-02 | CUB standard test split | backbone check | n/a | see report | [backbone](x.md) |"]
SIGMA = {c: {"quantity": ST.QUANTITIES[c][2], "sigma_a2": 4.0 + i, "sigma_eps2": 1500.0 + 10 * i}
         for i, c in enumerate(ST.CHECKS + ST.SECONDARY)}
# Stand-in development value sets of the rule's sizes (8, 23, 10) and stand-in names; install() puts value_names on
# r6_common, so the runner's recomputed value_sets.json (r6_common.value_sets_record) is this one.
VALUE_SETS = {"emotion": list(range(8)), "style": list(range(23)), "genre": list(range(10))}
VALUE_NAMES = {x: [f"{x} value {i}" for i in range(30)] for x in R.ASPECTS}


def value_names(data):
    return VALUE_NAMES


def value_sets_json(value_sets=None) -> dict:
    """The content of value_sets.json for the stand-in names."""
    vs = value_sets or VALUE_SETS
    return {x: [{"code": int(c), "name": VALUE_NAMES[x][int(c)]} for c in vs[x]] for x in R.ASPECTS}


class Reached(Exception):
    """Raised by a stub: the run went past the point the test says it must stop at."""


def mutant(tmp_path, module, *guards):
    """Import a copy of ``module`` (file name in HERE) with its `# guard:<g>` statements replaced by `pass`."""
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


def hx(*parts) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()


def seed_hashes(seed, salt="") -> dict:
    return {p: hx(seed, p, salt) for p in R.PAIR_NAMES}


# ---------------------------------------------------------------- records, ledger, started files in tmp_path

def write_records(res, shas=None, coef=COEF, smoke_names=("smoke_record.json",), value_sets=None, data=None):
    """The seed-42 records (refit, picks, value sets, regression, sensitivity, the DTS stop) and the smoke record(s),
    as their runners write them: passed, with the current module SHA-256s. value_sets.json holds the stand-in value
    sets and names, or, with ``data`` (a real setup()'s), r6_common.write_value_sets of ``value_sets`` and ``data``."""
    shas = dict(shas or R.r6_module_shas())
    res.mkdir(parents=True, exist_ok=True)
    meta = {"module_sha256": shas, "time": "2026-10-10 09:00:00"}
    (res / RH.REFIT_NAME).write_text(json.dumps({"passed": True, "items": {}, "affect_identity": True,
                                                 "coef_sha256": coef, **meta}))
    (res / RH.PICKS_NAME).write_text(json.dumps({**PICKS, "mean_r1": dict(P.MEAN_R1_TARGETS),
                                                 "targets": dict(P.MEAN_R1_TARGETS), "passed": True,
                                                 "coef_sha256": coef, **meta}))
    vs_path = res / RH.VALUE_SETS_NAME
    if data is None:
        vs_path.write_text(json.dumps(value_sets_json(value_sets), indent=1))
    else:
        R.write_value_sets(vs_path, value_sets, data)
    (res / RH.REGRESSION_NAME).write_text(json.dumps({"passed": True, "items": {}, "coef_sha256": coef,
                                                      "value_sets_sha256": R.sha256_file(vs_path), **meta}))
    (res / RH.SENS42_NAME).write_text(json.dumps({**SIGMA, "seed": 42, "N": 12288, "n_paintings": 3000,
                                                  "regression_sha256": R.sha256_file(res / RH.REGRESSION_NAME),
                                                  **meta}))
    (res / RH.DTS_CHOSEN_NAME).write_text(json.dumps({"stage": "chosen", "seed": 42, **meta}))
    (res / RH.DTS_STOP_NAME).write_text(json.dumps({"stage": "stop", "seed": 42, "built": True, "stop": False,
                                                    "budget": {"clock_start": R.DTS_CLOCK_START},
                                                    "input_sha256": {RH.DTS_CHOSEN_NAME: R.sha256_file(
                                                        res / RH.DTS_CHOSEN_NAME)}, **meta}))
    for name in smoke_names:
        (res / name).write_text(json.dumps({"passed": True, "module_sha256": shas, "time": "2026-10-10 09:30:00"}))
    return res


def ledger_line(row="H5", scripts=None, report="(pending)", episodes="(pending)") -> str:
    scripts = ", ".join(scripts if scripts is not None else [R.sha256_file(HERE / "run_r6_held.py")])
    return (f"| {row} | 2026-10-10 | ArtELingo held rows, aspect episodes, seeds 52 to 54, 4,096 per pair | AFF held "
            f"read, round 6 (rule c394b60) | {episodes} | {scripts} | {report} |")


def write_ledger(path, *lines):
    Path(path).write_text("\n".join(LEDGER_HEAD + list(lines)) + "\n")
    return Path(path)


def edit(path, **changes):
    rec = json.loads(Path(path).read_text())
    rec.update(changes)
    Path(path).write_text(json.dumps(rec))


def started_file(path, *attempts, copy=True):
    """A started file as start_attempt writes it, in results/ and (``copy``) the same bytes in the committed copy's
    folder beside it (tmp/folder, the case's ``folder``); each attempt given as (flags, {seed: hashes})."""
    att = [{"attempt": i + 1, "time": "2026-10-10 10:00:00", "flags": {"after_crash": False, "fix": None,
                                                                     "reserve": False, "smoke": False, **fl},
            "episodes_sha256": {str(s): h for s, h in hs.items()}} for i, (fl, hs) in enumerate(attempts)]
    path = Path(path)
    path.write_text(json.dumps({"rule_sha256": R.RULE_SHA256, "attempts": att}))
    if copy:
        (path.parent.parent / "folder").mkdir(exist_ok=True)
        (path.parent.parent / "folder" / path.name).write_bytes(path.read_bytes())
    return path


def nine(salt="") -> dict:
    return {s: seed_hashes(s, salt) for s in SEEDS}


def all_hashes(salt="") -> str:
    return ", ".join(h for s in SEEDS for h in seed_hashes(s, salt).values())


@pytest.fixture
def case(tmp_path):
    """A results folder with current records, a ledger with row H5 for this runner, a folder for the copy."""
    res = write_records(tmp_path / "results")
    led = write_ledger(tmp_path / "held_ledger.md", ledger_line())
    return SimpleNamespace(res=res, ledger=led, folder=tmp_path / "folder", tmp=tmp_path)


# ---------------------------------------------------------------- stubs of the heavy steps

def fake_scored(seed, n, extra_key=None) -> dict:
    """score_seed(..., include_pm=False)'s dict for one seed: the eight scorers, gates, reader, cl, pair_index."""
    rng = np.random.default_rng(seed)
    zero = np.zeros(n)

    def pa(cf=False):
        r1 = rng.integers(0, 5, n) / 4.0
        gain = zero if cf else rng.integers(-4, 5, n) / 4.0
        return {"r1": r1, "gain": gain, "other": r1 - gain, "swap": zero, "strict": zero}

    out = {k: pa(cf=k in RH.S.CONDITION_FREE) for k in S.CORE_SCORERS}
    out.update({"gates": {}, "reader": {}, "cl": rng.integers(0, 3000, n).astype(np.int64),
                "pair_index": np.repeat(np.arange(3, dtype=np.int64), n // 3)})
    if extra_key:
        out[extra_key] = pa()
    return out


def fake_post(rows, extra_rows=()):
    p = np.full((R.N_ROWS, 2), np.nan, dtype=np.float32)
    p[rows] = 0.5
    p[list(extra_rows)] = 0.5
    return {"affect": {"img": p, "txt": p}, "image": {"img": p, "txt": p}}


def make_env(coef=COEF, head_ok=True):
    split = SimpleNamespace(held=HELD, selection=SELECTION, groups=np.arange(R.N_ROWS, dtype=np.int64) // 4)
    return SimpleNamespace(inputs={"x": "c" * 64}, data=None, split=split, labels=None,
                           value_sets={x: list(v) for x, v in VALUE_SETS.items()},
                           heads=None, head_check={"passed": head_ok}, coef_sha256=coef, pm=None, readers=None)


def install(monkeypatch, mod=RH, stop_at=(), salt="", salts=None, recorded=None, coef=COEF, head_ok=True,
            skip_report=False, extra_key=None, post_extra=()):
    """Stubs on ``mod`` (and on the shared r6 modules) for the heavy steps; -> namespace with ``events`` (in call
    order), ``scored`` per seed and ``include_pm`` seen. A stub whose name is in ``stop_at`` raises Reached."""
    ns = SimpleNamespace(events=[], scored={}, include_pm=[], env=make_env(coef, head_ok))
    ev = ns.events

    def at(name, *a):
        ev.append((name, *a))
        if name in stop_at or (name, *a) in stop_at:
            raise Reached(name)

    def assert_inputs():
        at("inputs")
        return {"x": "c" * 64}

    def recorded_hashes(*a, **k):
        at("recorded")
        return dict(recorded or {})

    def setup():
        at("setup")
        return ns.env

    guard = mod.head_guard

    def head_guard(env, refit):
        at("head_guard")
        return guard(env, refit)

    def seed_bundle(env, mode, seed, n_per_pair, on_episodes=None):
        at("context", seed)                                          # the first array of the seed's rows
        rows = env.split.held if mode == "held" else env.split.selection
        sha = seed_hashes(seed, (salts or {}).get(seed, salt))
        eps = SimpleNamespace(seed=seed, sha=sha)
        at("episodes", seed)
        if on_episodes is not None and not skip_report:
            on_episodes(eps)
        at("posteriors", seed)
        n = 3 * n_per_pair
        ns.scored[seed] = fake_scored(seed, n, extra_key)
        return SimpleNamespace(mode=mode, seed=seed, n=n, cl=ns.scored[seed]["cl"], episodes_sha256=dict(sha),
                               post=fake_post(rows, post_extra))

    def save_episodes(path, eps):
        at("write", Path(path).name)
        np.savez(path, seed=np.int64(eps.seed), **{f"sha__{p}": np.array(eps.sha[p]) for p in R.PAIR_NAMES})
        return Path(path)

    def load_episodes(path):
        with np.load(path) as z:
            return SimpleNamespace(seed=int(z["seed"]), sha={p: str(z[f"sha__{p}"]) for p in R.PAIR_NAMES})

    def eligible(labels, groups, held, value_sets):
        at("eligible", len(held))

    def score_bundle(env, bundle, picks, lambdas, *, include_pm):
        at("score", bundle.seed)
        ns.include_pm.append(include_pm)
        return ns.scored[bundle.seed]

    real_load_picks, real_pass, real_write = P.load_picks, ST.pass_record, mod._write_bytes

    def load_picks(path=None):
        at("picks")
        return real_load_picks(path)

    def pass_record(*a, **k):
        at("pass")
        return real_pass(*a, **k)

    def write_bytes(path, data):
        at("write", Path(path).name)
        return real_write(path, data)

    monkeypatch.setattr(R, "assert_inputs", assert_inputs)
    monkeypatch.setattr(R, "value_names", value_names)               # the stand-in names of value_sets.json
    monkeypatch.setattr(E, "recorded_hashes", recorded_hashes)
    monkeypatch.setattr(E, "save_episodes", save_episodes)
    monkeypatch.setattr(E, "load_episodes", load_episodes)
    monkeypatch.setattr(R, "assert_held_eligible", eligible)
    monkeypatch.setattr(P, "load_picks", load_picks)
    monkeypatch.setattr(ST, "pass_record", pass_record)
    for name, f in (("setup", setup), ("head_guard", head_guard), ("seed_bundle", seed_bundle),
                    ("score_bundle", score_bundle), ("_write_bytes", write_bytes)):
        monkeypatch.setattr(mod, name, f)
    return ns


def held(c, mod=RH, **flags):
    return mod.run_held(results=c.res, ledger=c.ledger, here=HERE, folder=c.folder, **flags)


def refused(code, capsys, *, says=None):
    """Exit 4 and one refusal line, printed last (after step 3 timed progress lines may precede it)."""
    out = capsys.readouterr().out
    lines = out.strip().splitlines()
    ok = (code == RH.EXIT_REFUSE and lines and lines[-1].startswith("refused:")
          and sum(ln.startswith("refused:") for ln in lines) == 1
          and all(" [run_r6_held] " in ln for ln in lines[:-1]))
    return bool(ok) and (says is None or says in lines[-1])


def nothing_started(c):
    return not any(p.name.startswith("held_started") for p in list(c.res.iterdir()) + list(c.tmp.glob("folder/*")))


# ---------------------------------------------------------------- bookkeeping

def test_names_and_constants():
    assert RH.SENS42_NAME == RS6.OUT_NAME
    assert RH.MODES == ("regression", "held", "smoke")
    assert RH.READ_SCORERS == ("cosine", "rca", "B", "B0", "B1", "aff_fused", "aff_cf", "r1_fused")
    assert not set(RH.READ_KEYS) & set(S.PM_SCORERS + ("r1_cf",))
    n = {k: RH.read_names(k) for k in ("held", "fix1", "reserve", "smoke")}
    assert (n["held"].started, n["held"].pass_, n["held"].arrays, n["held"].sensitivity, n["held"].row) == (
        "held_started.json", "held_pass.json", "held_arrays.npz", "sensitivity_held.json", "H5")
    assert (n["fix1"].started, n["fix1"].pass_, n["fix1"].arrays, n["fix1"].sensitivity, n["fix1"].smoke_record,
            n["fix1"].row) == ("held_started.json", "held_pass_fix1.json", "held_arrays_fix1.npz",
                               "sensitivity_held.json", "smoke_record_fix1.json", "H5")
    assert (n["reserve"].started, n["reserve"].verdict, n["reserve"].pass_, n["reserve"].sensitivity,
            n["reserve"].row) == ("held_started_reserve.json", "held_verdict_reserve.json",
                                  "held_pass_reserve.json", "sensitivity_held_reserve.json", "H5-R")
    assert RH.LEDGER == R.MAIN / "docs/superpowers/held_ledger.md"


def test_ledger_header_matches_the_real_ledger():
    """The tmp ledgers use the real table's header (read only)."""
    head = [ln for ln in RH.LEDGER.read_text().splitlines() if ln.startswith("| # |")]
    assert head == [LEDGER_HEAD[2]]
    assert tuple(c.strip() for c in head[0].strip("|").split("|")) == RH.LEDGER_CELLS


def test_ledger_row_parsing(tmp_path):
    a, b = "1" * 64, "2" * 64
    led = write_ledger(tmp_path / "l.md", ledger_line(scripts=[a, b]), ledger_line("H5-R", scripts=[a]))
    row = RH.ledger_row(led, "H5")
    assert row["scripts"] == [a, b] and row["report"] == "(pending)" and row["episodes"] == []
    assert RH.ledger_row(led, "H5-R")["scripts"] == [a]
    assert RH.ledger_guard(led, "H5", b)["row"] == "H5"
    with pytest.raises(RH.Refused, match="latest script"):
        RH.ledger_guard(led, "H5", a)                          # a is not the latest
    long = write_ledger(tmp_path / "l2.md", ledger_line(scripts=[a + "0"]))
    assert RH.ledger_row(long, "H5")["scripts"] == []          # a 65-hex string is no SHA-256
    with pytest.raises(RH.Refused, match="appears 2 times"):
        RH.ledger_row(write_ledger(tmp_path / "l3.md", ledger_line(), ledger_line()), "H5")
    with pytest.raises(RH.Refused, match="cells"):
        RH.ledger_row(write_ledger(tmp_path / "l4.md", "| H5 | 2026-10-10 | x | (pending) |"), "H5")
    with pytest.raises(RH.Refused, match="does not exist"):
        RH.ledger_row(tmp_path / "missing.md", "H5")


# ---------------------------------------------------------------- refusal, step 1 (one test per clause)

@pytest.fixture
def no_work(monkeypatch):
    """A refusal must come before step 3: the first input check raises Reached."""
    return install(monkeypatch, stop_at={"inputs"})


def test_refuses_when_the_verdict_exists(case, no_work, capsys):
    (case.res / "held_verdict.json").write_text("{}")
    assert refused(held(case), capsys, says="held_verdict.json exists") and nothing_started(case)
    assert refused(held(case, after_crash=True), capsys, says="held_verdict.json exists")
    assert refused(held(case, fix=1), capsys, says="held_verdict.json exists")
    assert no_work.events == []


def test_refuses_when_the_pass_exists_without_fix(case, no_work, capsys):
    (case.res / "held_pass.json").write_text("{}")
    assert refused(held(case), capsys, says="held_pass.json exists and --fix 1 is not given") and nothing_started(case)
    started_file(case.res / "held_started.json", ({}, nine()))
    assert refused(held(case, after_crash=True), capsys, says="held_pass.json exists")


def test_refuses_when_a_read_has_started_without_a_flag(case, no_work, capsys):
    started_file(case.res / "held_started.json", ({}, {52: seed_hashes(52)}))
    assert refused(held(case), capsys, says="held_started.json exists")
    assert no_work.events == []


def test_after_crash_refuses_a_third_attempt(case, no_work, capsys):
    started_file(case.res / "held_started.json", ({}, {}), ({"after_crash": True}, {}))
    assert refused(held(case, after_crash=True), capsys, says="already records 2 attempts")
    started_file(case.res / "held_started.json", ({}, {}))
    with pytest.raises(Reached):                              # one attempt recorded: the rerun goes on to step 3
        held(case, after_crash=True)


def test_refuses_without_ledger_row_h5(case, no_work, capsys):
    write_ledger(case.ledger)                                 # the table without H5
    assert refused(held(case), capsys, says="ledger row H5 is missing")
    write_ledger(case.ledger, ledger_line("H5-R"))            # the reserve row is not H5
    assert refused(held(case), capsys, says="ledger row H5 is missing")
    write_ledger(case.ledger, ledger_line(), ledger_line())
    assert refused(held(case), capsys, says="appears 2 times")
    case.ledger.unlink()
    assert refused(held(case), capsys, says="does not exist") and nothing_started(case)


def test_refuses_a_ledger_sha_other_than_this_runner(case, no_work, capsys):
    me = R.sha256_file(HERE / "run_r6_held.py")
    write_ledger(case.ledger, ledger_line(scripts=["3" * 64]))
    assert refused(held(case), capsys, says="latest script SHA-256 is not this runner's")
    write_ledger(case.ledger, ledger_line(scripts=[me, "3" * 64]))      # this runner's, but not the latest
    assert refused(held(case), capsys, says="latest script SHA-256 is not this runner's")
    write_ledger(case.ledger, ledger_line(scripts=[]))
    assert refused(held(case), capsys, says="latest script SHA-256 is not this runner's")
    write_ledger(case.ledger, ledger_line(scripts=["3" * 64, me]))      # the latest is this runner's
    with pytest.raises(Reached):
        held(case)


@pytest.mark.parametrize("report", ["pending", "(pending) x", "", "[verdict](x.md)", "(Pending)"])
def test_refuses_a_report_cell_other_than_pending(case, no_work, capsys, report):
    write_ledger(case.ledger, ledger_line(report=report))
    assert refused(held(case), capsys, says="report cell") and nothing_started(case)


def test_fix_refuses_when_the_fix_pass_exists(case, no_work, capsys):
    fix_ready(case)
    (case.res / "held_pass_fix1.json").write_text("{}")
    assert refused(held(case, fix=1), capsys, says="held_pass_fix1.json exists")


def test_fix_refuses_without_its_smoke_record(case, no_work, capsys):
    fix_ready(case)
    (case.res / "smoke_record_fix1.json").unlink()
    assert refused(held(case, fix=1), capsys, says="smoke_record_fix1.json is missing")


def test_reserve_uses_row_h5r_and_the_reserve_names(case, no_work, capsys):
    reserve_ready(case)
    with pytest.raises(Reached):
        held(case, reserve=True)                              # every precondition met: it goes on to step 3
    write_ledger(case.ledger, ledger_line())                  # H5 alone does not do for the reserve
    assert refused(held(case, reserve=True), capsys, says="ledger row H5-R is missing")
    write_ledger(case.ledger, ledger_line(), ledger_line("H5-R", scripts=["3" * 64]))
    assert refused(held(case, reserve=True), capsys, says="ledger row H5-R: its latest script")
    write_ledger(case.ledger, ledger_line(), ledger_line("H5-R", report="done"))
    assert refused(held(case, reserve=True), capsys, says="ledger row H5-R: its report cell")


def test_reserve_refuses_when_its_started_or_verdict_file_exists(case, no_work, capsys):
    reserve_ready(case)
    started_file(case.res / "held_started_reserve.json", ({"reserve": True}, {}))
    assert refused(held(case, reserve=True), capsys, says="held_started_reserve.json exists")
    (case.res / "held_started_reserve.json").rename(case.tmp / "kept_started_reserve.json")
    (case.res / "held_verdict_reserve.json").write_text("{}")
    assert refused(held(case, reserve=True), capsys, says="held_verdict_reserve.json exists")
    (case.res / "held_verdict_reserve.json").rename(case.tmp / "kept_verdict_reserve.json")
    (case.res / "held_pass_reserve.json").write_text("{}")
    assert refused(held(case, reserve=True), capsys, says="held_pass_reserve.json exists")


def fix_ready(case):
    """Everything a --fix 1 needs: the read's pass and started file (nine hashes), row H5 with the nine hashes, the
    fix smoke record."""
    (case.res / "held_pass.json").write_text("{}")
    started_file(case.res / "held_started.json", ({}, nine()))
    write_ledger(case.ledger, ledger_line(episodes=all_hashes()))
    write_records(case.res, smoke_names=("smoke_record.json", "smoke_record_fix1.json"))


def reserve_ready(case):
    """Everything a --reserve needs: the read's verdict and started file (nine hashes), rows H5 and H5-R, the reserve
    smoke record."""
    (case.res / "held_pass.json").write_text("{}")
    (case.res / "held_verdict.json").write_text("{}")
    started_file(case.res / "held_started.json", ({}, nine()))
    write_ledger(case.ledger, ledger_line(report="abc"), ledger_line("H5-R"))
    write_records(case.res, smoke_names=("smoke_record.json", "smoke_record_reserve.json"))


# ---- agent defaults (stricter than the rule)

def test_after_crash_without_a_started_file_refuses(case, no_work, capsys):
    assert refused(held(case, after_crash=True), capsys, says="does not exist, no attempt to rerun")


def test_fix_needs_the_pass_and_runs_once(case, no_work, capsys):
    fix_ready(case)
    with pytest.raises(Reached):
        held(case, fix=1)
    (case.res / "held_pass.json").rename(case.tmp / "kept_pass.json")
    assert refused(held(case, fix=1), capsys, says="--fix 1 corrects held_pass.json")
    (case.tmp / "kept_pass.json").rename(case.res / "held_pass.json")
    started_file(case.res / "held_started.json", ({}, nine()), ({"fix": 1}, nine()))
    assert refused(held(case, fix=1), capsys, says="already records a --fix 1 attempt")


def test_fix_needs_the_nine_hashes_in_row_h5(case, no_work, capsys):
    fix_ready(case)
    write_ledger(case.ledger, ledger_line(episodes="(pending)"))
    assert refused(held(case, fix=1), capsys, says="episode cell")
    write_ledger(case.ledger, ledger_line(episodes=all_hashes(salt="other")))
    assert refused(held(case, fix=1), capsys, says="episode cell")
    short = ", ".join(h[:8] + "…" for s in SEEDS for h in seed_hashes(s).values())
    write_ledger(case.ledger, ledger_line(episodes=short))
    assert refused(held(case, fix=1), capsys, says="episode cell")
    started_file(case.res / "held_started.json", ({}, {52: seed_hashes(52)}))
    write_ledger(case.ledger, ledger_line(episodes=all_hashes()))
    assert refused(held(case, fix=1), capsys, says="episode cell")


def test_reserve_needs_the_verdict_and_the_nine_hashes(case, no_work, capsys):
    reserve_ready(case)
    (case.res / "held_verdict.json").rename(case.tmp / "kept_verdict.json")
    assert refused(held(case, reserve=True), capsys, says="held_verdict.json exists, and it does not")
    (case.tmp / "kept_verdict.json").rename(case.res / "held_verdict.json")
    started_file(case.res / "held_started.json", ({}, {52: seed_hashes(52), 53: seed_hashes(53)}))
    assert refused(held(case, reserve=True), capsys, says="nine held episode SHA-256s")


def test_flags_are_exclusive(case, no_work, capsys):
    assert refused(held(case, after_crash=True, fix=1), capsys, says="at most one")
    assert refused(held(case, fix=2), capsys, says="at most one")
    for argv in (["--mode", "held", "--fix", "2"], ["--mode", "held", "--after-crash", "--reserve"],
                 ["--mode", "smoke", "--after-crash"], ["--mode", "regression", "--fix", "1"],
                 ["--mode", "held", "--smoke-subdir", "x"], ["--mode", "verdict"], []):
        with pytest.raises(SystemExit):
            RH.main(argv)


def test_malformed_started_file_refuses(case, no_work, capsys):
    case.folder.mkdir()
    for p in (case.res, case.folder):
        (p / "held_started.json").write_text("{not json")
    assert refused(held(case, after_crash=True), capsys, says="cannot be read")
    for p in (case.res, case.folder):
        (p / "held_started.json").write_text(json.dumps({"rule_sha256": "0" * 64, "attempts": []}))
    assert refused(held(case, after_crash=True), capsys, says="not a started file of this rule")


# ---------------------------------------------------------------- step 2: the smoke record

def test_refuses_a_stale_module_sha_against_the_smoke_record(case, no_work, capsys):
    for name in ("run_r6_held.py", "r6_score.py", "r6_context.py", "r6_gpu_inputs.py", "dts_settings.json"):
        shas = R.r6_module_shas()
        key = RH._key(name, HERE)
        assert key in shas, key
        shas[key] = "0" * 64
        edit(case.res / "smoke_record.json", module_sha256=shas)
        assert refused(held(case), capsys, says=f"the SHA-256s of ['{key}'] differ from the smoke's"), name
    shas = R.r6_module_shas()
    edit(case.res / "smoke_record.json", module_sha256={**shas, "src/test/x/r6_new.py": "1" * 64})
    assert refused(held(case), capsys, says="r6_new.py")                    # a module the smoke ran that is gone
    del shas[RH._key("r6_stats.py", HERE)]
    edit(case.res / "smoke_record.json", module_sha256=shas)
    assert refused(held(case), capsys, says="r6_stats.py")                  # a module the smoke did not run
    edit(case.res / "smoke_record.json", module_sha256=R.r6_module_shas())
    with pytest.raises(Reached):
        held(case)
    assert nothing_started(case)


def test_smoke_record_missing_or_failed_refuses(case, no_work, capsys):
    edit(case.res / "smoke_record.json", passed=False)
    assert refused(held(case), capsys, says="did not pass")
    (case.res / "smoke_record.json").unlink()
    assert refused(held(case), capsys, says="smoke_record.json does not exist")


def test_the_latest_smoke_record_counts(case, no_work, capsys):
    """For --fix 1 the fix record; for the read, the fix record when it exists (contracts amendment 12:20)."""
    fix_ready(case)
    stale = {**R.r6_module_shas(), RH._key("r6_bundle.py", HERE): "0" * 64}
    edit(case.res / "smoke_record_fix1.json", module_sha256=stale)
    assert refused(held(case, fix=1), capsys, says="smoke_record_fix1.json: the SHA-256s")
    assert RH.latest_smoke_record(case.res, "held").name == "smoke_record_fix1.json"
    (case.res / "smoke_record_fix1.json").rename(case.tmp / "kept.json")
    assert RH.latest_smoke_record(case.res, "held").name == "smoke_record.json"
    assert RH.latest_smoke_record(case.res, "reserve").name == "smoke_record_reserve.json"


# ---------------------------------------------------------------- step 3: the seed-42 records

@pytest.fixture
def at_picks(monkeypatch):
    return install(monkeypatch, stop_at={"picks"})


def test_seed42_records_must_be_passed_and_current(case, at_picks, capsys):
    with pytest.raises(Reached):
        held(case)
    assert [e[0] for e in at_picks.events] == ["inputs", "recorded", "picks"]
    reg, sens = case.res / RH.REGRESSION_NAME, case.res / RH.SENS42_NAME
    keep_reg, keep_sens = reg.read_text(), sens.read_text()
    edit_regression(case.res, passed=False)
    assert refused(held(case), capsys, says="regression_seed42.json did not pass")
    edit_regression(case.res, passed=True, module_sha256=stale("r6_score.py"))
    assert refused(held(case), capsys, says="rerun the regression")
    reg.write_text(keep_reg)
    sens.write_text(keep_sens)
    with pytest.raises(Reached):
        held(case)
    edit(reg, time="later")                                   # the regression rerun: the sensitivity is not current
    assert refused(held(case), capsys, says="rerun run_r6_sensitivity.py")
    reg.write_text(keep_reg)
    edit(sens, regression_sha256="0" * 64)
    assert refused(held(case), capsys, says="rerun run_r6_sensitivity.py")
    sens.write_text(keep_sens)
    edit(sens, module_sha256={**R.r6_module_shas(), RH._key("r6_stats.py", HERE): "0" * 64})
    assert refused(held(case), capsys, says="rerun run_r6_sensitivity.py")
    sens.write_text(keep_sens)
    edit(sens, P3={"sigma_a2": -1.0, "sigma_eps2": 2.0})
    assert refused(held(case), capsys, says="['P3']")
    edit(sens, P3={"sigma_a2": 1.0, "sigma_eps2": float("nan")})
    assert refused(held(case), capsys, says="['P3']")
    sens.write_text(keep_sens)
    edit(case.res / RH.REFIT_NAME, passed=False)
    assert refused(held(case), capsys, says="refit_check.json did not pass")
    assert nothing_started(case)


def test_held_mode_refuses_without_a_passed_current_dts_stop(case, at_picks, capsys):
    """Rule section 9 (final review, 2026-10-09): every held read refuses (exit 4, nothing started) unless
    dts_stop.json is seed 42's stop record, built, with no stop, written by the current DTS bytes."""
    stop = case.res / RH.DTS_STOP_NAME
    keep = stop.read_text()
    with pytest.raises(Reached):
        held(case)
    stop.unlink()
    assert refused(held(case), capsys, says="dts_stop.json does not exist")
    for change, says in (({"stop": True}, "the user decides"), ({"built": False}, "the user decides"),
                         ({"seed": 9001}, "not seed 42's stop record"), ({"stage": "chosen"}, "not seed 42's stop"),
                         ({"module_sha256": stale("r6_dts.py")}, "rerun the DTS stages"),
                         ({"module_sha256": stale("dts_settings.json")}, "rerun the DTS stages"),
                         ({"input_sha256": {}}, "not evaluated on the current bytes of ['dts_seed42.json']"),
                         ({"budget": {"clock_start": "2026-10-10 12:29"}}, "not the first DTS commit's"),
                         ({"budget": None}, "the clock start None"),
                         ({"input_sha256": {RH.DTS_CHOSEN_NAME: "0" * 64}}, "rerun the stop"),
                         ({"input_sha256": {RH.DTS_CHOSEN_NAME: R.sha256_file(case.res / RH.DTS_CHOSEN_NAME),
                                            "dts_tune.json": "1" * 64}}, "bytes of ['dts_tune.json']")):
        stop.write_text(keep)
        edit(stop, **change)
        assert refused(held(case), capsys, says=says), change
    assert RH.dts_stop_problem(case.res, HERE)[1] is False                  # a stale record is a rerun
    stop.write_text(keep)
    edit(stop, stop=True)
    assert RH.dts_stop_problem(case.res, HERE)[1] is True                   # a stop goes to the user, not a rerun
    assert nothing_started(case)


def test_dts_stop_refuses_every_flag_but_not_the_smoke(case, monkeypatch, capsys):
    install(monkeypatch, stop_at={"picks"})
    started_file(case.res / "held_started.json", ({}, {}))                 # --after-crash: a read has started
    (case.res / RH.DTS_STOP_NAME).unlink()
    assert refused(held(case, after_crash=True), capsys, says="dts_stop.json does not exist")
    fix_ready(case)
    (case.res / RH.DTS_STOP_NAME).unlink()
    assert refused(held(case, fix=1), capsys, says="dts_stop.json does not exist")
    reserve_ready(case)
    (case.res / RH.DTS_STOP_NAME).unlink()
    assert refused(held(case, reserve=True), capsys, says="dts_stop.json does not exist")
    # the smoke does not read it (the smoke chain runs its own seed's stop)
    with pytest.raises(Reached):
        RH.run_smoke(results=case.res, here=HERE)


def test_the_dts_stop_sha_is_kept_in_the_attempt_and_the_pass(case, monkeypatch, capsys):
    install(monkeypatch)
    assert held(case) == 0
    want = {"file": RH.DTS_STOP_NAME, "sha256": R.sha256_file(case.res / RH.DTS_STOP_NAME)}
    att = json.loads((case.res / "held_started.json").read_text())["attempts"][0]
    assert att["dts_stop"] == want
    assert json.loads((case.res / "held_pass.json").read_text())["dts_stop"] == want
    assert RH.DTS_STOP_NAME == "dts_stop.json"                               # run_r6_dts.STOP


def test_guard_dts_stop(case, monkeypatch, capsys):
    seed42_mut(case, monkeypatch, capsys, mutant(case.tmp, "run_r6_held.py", "dts_stop"),
               lambda res: (res / RH.DTS_STOP_NAME).unlink(), Reached)


def test_seed42_records_missing_refuse(case, at_picks, capsys):
    (case.res / RH.SENS42_NAME).unlink()
    assert refused(held(case), capsys, says="run run_r6_sensitivity.py first")
    (case.res / RH.REGRESSION_NAME).unlink()
    assert refused(held(case), capsys, says="run --mode regression first")
    (case.res / RH.PICKS_NAME).unlink()
    assert refused(held(case), capsys, says="run run_r6_picks.py first")


# ---------------------------------------------------------------- value_sets.json (rule section 5 item 2; final review B1)

READ_KINDS = ("first", "after_crash", "fix", "reserve", "smoke")


def ready(case, kind) -> dict:
    """The results folder of a first read, an --after-crash rerun, a --fix 1 pass, a --reserve read or a smoke;
    -> run_held's flags."""
    if kind == "after_crash":
        started_file(case.res / "held_started.json", ({}, {}))
    elif kind == "fix":
        fix_ready(case)
    elif kind == "reserve":
        reserve_ready(case)
    return {"after_crash": {"after_crash": True}, "fix": {"fix": 1}, "reserve": {"reserve": True}}.get(kind, {})


def run_kind(case, kind, flags) -> int:
    return RH.run_smoke(results=case.res, here=HERE) if kind == "smoke" else held(case, **flags)


def snapshot(c) -> dict:
    return {p: p.read_bytes() for p in c.tmp.rglob("*") if p.is_file()}


def vs_change(res, text):
    """value_sets.json rewritten as ``text``, and the regression record naming its new bytes (so that only the
    content is wrong)."""
    path = res / RH.VALUE_SETS_NAME
    path.write_text(text)
    edit_regression(res, value_sets_sha256=R.sha256_file(path))


def changed_value_sets(change) -> str:
    rec = value_sets_json()
    if change == "one value fewer":
        rec["style"] = rec["style"][:-1]
    elif change == "a name changed":
        rec["emotion"][0] = {**rec["emotion"][0], "name": "another name"}
    else:
        return "{"
    return json.dumps(rec, indent=1)


def test_value_sets_record_is_the_rule_file_format():
    """r6_common.value_sets_record: codes with their names (the stand-in names here; the real names in
    test_r6_common's real-data test of write_value_sets)."""
    names = {x: [f"n{i}" for i in range(30)] for x in R.ASPECTS}
    orig = R.value_names
    try:
        R.value_names = lambda data: names
        rec = R.value_sets_record(VALUE_SETS, None)
    finally:
        R.value_names = orig
    assert {x: len(v) for x, v in rec.items()} == R.VALUE_COUNTS
    assert rec["style"][22] == {"code": 22, "name": "n22"}


@pytest.mark.parametrize("kind", READ_KINDS)
def test_value_sets_file_missing_or_not_the_regressions_refuses_before_any_load(case, monkeypatch, capsys, kind):
    ns = install(monkeypatch, stop_at={"picks"})
    flags = ready(case, kind)
    path = case.res / RH.VALUE_SETS_NAME
    keep = path.read_bytes()
    path.unlink()
    before = snapshot(case)
    assert refused(run_kind(case, kind, flags), capsys, says="value_sets.json does not exist: run --mode regression")
    assert snapshot(case) == before and "setup" not in [e[0] for e in ns.events]
    path.write_bytes(keep)                                                 # the regression's file again
    with pytest.raises(Reached):
        run_kind(case, kind, flags)
    for sha in ("0" * 64, None):                    # another file's SHA-256; a regression record of before the fix
        edit_regression(case.res, value_sets_sha256=sha)
        before = snapshot(case)
        assert refused(run_kind(case, kind, flags), capsys,
                       says="value_sets.json is not the file regression_seed42.json records")
        assert snapshot(case) == before


@pytest.mark.parametrize("kind", READ_KINDS)
@pytest.mark.parametrize("change", ("one value fewer", "a name changed", "not JSON", "other recomputed sets"))
def test_value_sets_other_than_the_recomputed_refuse_before_the_first_write(case, monkeypatch, capsys, kind, change):
    ns = install(monkeypatch)
    flags = ready(case, kind)
    if change == "other recomputed sets":
        ns.env.value_sets["genre"] = list(range(1, 10))                    # setup() recomputes other sets
    else:
        vs_change(case.res, changed_value_sets(change))
    before = snapshot(case)
    assert refused(run_kind(case, kind, flags), capsys,
                   says="value_sets.json does not hold the development value sets recomputed now")
    assert snapshot(case) == before and ns.events[-1] == ("head_guard",)   # after setup, before the first write


def test_value_sets_changed_after_step_3_refuses(case, monkeypatch, capsys):
    install(monkeypatch)
    stub = RH.head_guard

    def head_guard(env, refit):                     # the file is rewritten during setup(): same content, other bytes
        (case.res / RH.VALUE_SETS_NAME).write_text(json.dumps(value_sets_json()))
        return stub(env, refit)

    monkeypatch.setattr(RH, "head_guard", head_guard)
    assert refused(held(case), capsys, says="or changed since step 3") and nothing_started(case)


def test_the_value_sets_sha_is_kept_in_the_attempt_and_the_pass(case, monkeypatch, capsys):
    install(monkeypatch)
    want = {"file": RH.VALUE_SETS_NAME, "sha256": R.sha256_file(case.res / RH.VALUE_SETS_NAME)}
    assert held(case) == 0
    assert json.loads((case.res / "held_started.json").read_text())["attempts"][0]["value_sets"] == want
    assert json.loads((case.res / "held_pass.json").read_text())["value_sets"] == want
    assert RH.run_smoke(results=case.res, here=HERE) == 0
    smoke = case.res / "smoke"
    assert json.loads((smoke / "held_started.json").read_text())["attempts"][0]["value_sets"] == want
    assert json.loads((smoke / "held_pass.json").read_text())["value_sets"] == want
    assert RH.VALUE_SETS_NAME == "value_sets.json"                         # rule section 5 item 2's name


def test_guard_value_sets_file(case, monkeypatch, capsys):
    edit_regression(case.res, value_sets_sha256="0" * 64)
    install(monkeypatch, stop_at={"picks"})
    assert refused(held(case), capsys, says="is not the file regression_seed42.json records")
    mod = mut_case(case, mutant(case.tmp, "run_r6_held.py", "value_sets_file"))
    install(monkeypatch, mod, stop_at={"picks"})
    with pytest.raises(Reached):                                           # the mutant goes past step 3
        held(case, mod)


def test_guard_value_sets_equal(case, monkeypatch, capsys):
    vs_change(case.res, changed_value_sets("one value fewer"))
    install(monkeypatch, stop_at={"context"})
    assert refused(held(case), capsys, says="does not hold the development value sets") and nothing_started(case)
    mod = mut_case(case, mutant(case.tmp, "run_r6_held.py", "value_sets_equal"))
    install(monkeypatch, mod, stop_at={"context"})
    with pytest.raises(Reached):                                           # the read starts on the wrong value sets
        held(case, mod)
    assert (case.res / "held_started.json").is_file()


# ---------------------------------------------------------------- step 4: the head checks (rule section 5 item 5)

def test_refuses_unless_the_head_check_passed(case, monkeypatch, capsys):
    ns = install(monkeypatch, head_ok=False)
    assert refused(held(case), capsys, says="selection posteriors differ") and nothing_started(case)
    assert [e[0] for e in ns.events] == ["inputs", "recorded", "picks", "setup", "head_guard"]


def test_refuses_unless_the_coefficient_shas_equal_refit_check(case, monkeypatch, capsys):
    other = {**COEF, "csd": {"img": "a" * 64, "txt": "c" * 64}}
    ns = install(monkeypatch, coef=other)
    assert refused(held(case), capsys, says="coefficient SHA-256s differ") and nothing_started(case)
    assert ns.events[-1] == ("head_guard",)
    ns = install(monkeypatch, coef={})
    assert refused(held(case), capsys, says="coefficient SHA-256s differ")
    edit(case.res / RH.REFIT_NAME, coef_sha256={})
    ns = install(monkeypatch, coef={})
    assert refused(held(case), capsys, says="coefficient SHA-256s differ")       # an empty record never matches


# ---------------------------------------------------------------- a stubbed read, end to end

def test_a_stubbed_read_writes_every_file_and_no_verdict(case, monkeypatch, capsys):
    ns = install(monkeypatch)
    assert held(case) == 0
    out = capsys.readouterr().out
    assert ns.include_pm == [False, False, False]                          # rule section 8 item 3
    names = sorted(p.name for p in case.res.iterdir())
    assert names == sorted([RH.REFIT_NAME, RH.PICKS_NAME, RH.REGRESSION_NAME, RH.SENS42_NAME, RH.DTS_STOP_NAME,
                            RH.DTS_CHOSEN_NAME, RH.VALUE_SETS_NAME, "smoke_record.json",
                            "held_started.json", "held_episodes_seed52.npz", "held_episodes_seed53.npz",
                            "held_episodes_seed54.npz", "sensitivity_held.json", "held_arrays.npz",
                            "held_pass.json"])
    assert not any("verdict" in n or n.endswith(".partial") for n in names)
    pass_sha = R.sha256_file(case.res / "held_pass.json")
    lines = out.strip().splitlines()
    assert lines[-1] == f"held pass written {pass_sha}"
    assert all(" [run_r6_held] " in ln for ln in lines[:-1])                 # the rest: timed progress lines
    assert not DECIMAL.search(out.replace(str(case.tmp), ""))

    st = json.loads((case.res / "held_started.json").read_text())
    assert (case.folder / "held_started.json").read_bytes() == (case.res / "held_started.json").read_bytes()
    assert st["rule_sha256"] == R.RULE_SHA256 and len(st["attempts"]) == 1
    a = st["attempts"][0]
    assert a["attempt"] == 1 and a["flags"] == {"after_crash": False, "fix": None, "reserve": False, "smoke": False}
    assert a["episodes_sha256"] == {str(s): seed_hashes(s) for s in SEEDS}
    assert a["coef_sha256"] == COEF and a["runner_sha256"] == R.sha256_file(HERE / "run_r6_held.py")
    assert a["module_sha256"] == R.r6_module_shas() and a["input_sha256"] == {"x": "c" * 64}
    assert a["smoke_record"] == {"name": "smoke_record.json", "sha256": R.sha256_file(case.res / "smoke_record.json")}
    assert a["ledger"] == {"row": "H5", "script_sha256": R.sha256_file(HERE / "run_r6_held.py")}
    assert set(a["seed42_records_sha256"]) == {RH.REFIT_NAME, RH.PICKS_NAME, RH.REGRESSION_NAME, RH.SENS42_NAME}
    assert re.fullmatch(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d", a["time"])

    pr = json.loads((case.res / "held_pass.json").read_text())
    scored = [ns.scored[s] for s in SEEDS]
    assert pr["mode"] == "held" and pr["seeds"] == list(SEEDS) and pr["n_episodes"] == 3 * 3 * R.N_PER_PAIR
    assert set(pr["checks"]) == set(ST.CHECKS) and set(pr["secondary"]) == set(ST.SECONDARY)
    assert pr["episodes_sha256"] == {str(s): seed_hashes(s) for s in SEEDS} and pr["kind"] == "held"
    want = ST.check_diffs(scored)["P2"]
    assert pr["checks"]["P2"]["n"] == int(ST.bootstrap_draws(*want)[1].sum())
    assert pr["outputs"] == {n: R.sha256_file(case.res / n) for n in ("held_arrays.npz", "sensitivity_held.json")}
    assert "verdict" not in pr and not any("verdict" in k for k in pr)

    with np.load(case.res / "held_arrays.npz") as z:
        assert set(z.files) == ({f"{s}__{m}" for s in RH.READ_SCORERS for m in METRICS}
                                | {"cl", "pair_index", "seed_index"})
        assert np.array_equal(z["aff_fused__r1"], np.concatenate([x["aff_fused"]["r1"] for x in scored]))
        assert np.array_equal(z["seed_index"], np.repeat(np.arange(3), 3 * R.N_PER_PAIR))
        assert np.array_equal(z["cl"], np.concatenate([x["cl"] for x in scored]))
    for s in SEEDS:
        with np.load(case.res / f"held_episodes_seed{s}.npz") as z:
            assert int(z["seed"]) == s and str(z["sha__style__genre"]) == seed_hashes(s)["style__genre"]


def test_sensitivity_held_is_the_rule_formula(case, monkeypatch, capsys):
    ns = install(monkeypatch)
    assert held(case) == 0
    rec = json.loads((case.res / "sensitivity_held.json").read_text())
    cl = np.concatenate([ns.scored[s]["cl"] for s in SEEDS])
    _, m = np.unique(cl, return_counts=True)
    n = 36_864
    assert rec["N"] == n and rec["n_paintings"] == len(m) and rec["seeds"] == list(SEEDS)
    for c in ST.CHECKS + ST.SECONDARY:
        a2, e2 = SIGMA[c]["sigma_a2"], SIGMA[c]["sigma_eps2"]
        se = float(np.sqrt((a2 * float(np.sum(m.astype(np.int64) ** 2)) + e2 * n) / float(n) ** 2))
        keys = {"quantity", "sigma_a2", "sigma_eps2", "SE", "x95", "x" if c in ST.CHECKS else "x2"}
        assert set(rec[c]) == keys, c
        assert rec[c]["SE"] == se and rec[c]["x95"] == 2.80 * se, c
        assert rec[c]["x" if c in ST.CHECKS else "x2"] == (3.532 if c in ST.CHECKS else 3.083) * se, c
        assert rec[c]["quantity"] == ST.QUANTITIES[c][2]
    assert rec["sigma_source"] == {"file": RH.SENS42_NAME, "sha256": R.sha256_file(case.res / RH.SENS42_NAME)}
    assert rec["episodes_sha256"] == {str(s): seed_hashes(s) for s in SEEDS}
    assert rec["module_sha256"] == R.r6_module_shas() and "time" in rec
    # M_p pools a painting's anchors over the three seeds
    b = [SimpleNamespace(cl=np.array([7] * 192)), SimpleNamespace(cl=np.array([7] * 96 + [8] * 96)),
         SimpleNamespace(cl=np.array([9] * 192))]
    got = RH.sensitivity_held(b, R.SMOKE_SEEDS, SIGMA, R.N_SMOKE)
    m = np.array([288, 96, 192])
    se = float(np.sqrt((SIGMA["P1"]["sigma_a2"] * float(np.sum(m ** 2)) + SIGMA["P1"]["sigma_eps2"] * 576) / 576 ** 2))
    assert got["P1"]["SE"] == se and got["N"] == 576 and got["n_paintings"] == 3


def test_order_of_the_read(case, monkeypatch, capsys):
    """Rule section 8 items 1 to 3 with every step recorded: nothing written before the heads are checked; the
    started file first, before any array of a held row; each seed's hashes appended before its posteriors and
    before any score; the sensitivity before the scores; the pass last."""
    ns = install(monkeypatch)
    assert held(case) == 0
    ev = ns.events
    pos = {e: i for i, e in reversed(list(enumerate(ev)))}               # first position of each event
    writes = [(i, e[1]) for i, e in enumerate(ev) if e[0] == "write"]
    first_write = writes[0][0]
    assert ev[first_write] == ("write", "held_started.json") and writes[1] == (first_write + 1, "held_started.json")
    for e in [("inputs",), ("recorded",), ("picks",), ("setup",), ("head_guard",)]:
        assert pos[e] < first_write, e
    assert pos[("inputs",)] < pos[("recorded",)] < pos[("setup",)] < pos[("head_guard",)]
    assert first_write < min(i for i, e in enumerate(ev) if e[0] in ("context", "episodes", "posteriors", "score"))
    first_score = min(i for i, e in enumerate(ev) if e[0] == "score")
    for s in SEEDS:
        i_eps, i_post = pos[("episodes", s)], pos[("posteriors", s)]
        between = [w for i, w in writes if i_eps < i < i_post]
        assert between == ["held_started.json", "held_started.json", f"held_episodes_seed{s}.npz"], (s, between)
        assert i_post < first_score
    assert pos[("eligible", len(HELD))] < pos[("write", "sensitivity_held.json")] < first_score
    assert [e[1] for e in ev if e[0] == "score"] == list(SEEDS)
    last_score = max(i for i, e in enumerate(ev) if e[0] == "score")
    assert last_score < pos[("write", "held_arrays.npz")] < pos[("pass",)] < pos[("write", "held_pass.json")]
    assert writes[-1][1] == "held_pass.json"
    assert [w for _, w in writes].count("held_started.json") == 2 * (1 + len(SEEDS))   # results and copy, 4 times


# ---------------------------------------------------------------- --after-crash

def test_after_crash_reproduces_the_hashes_and_keeps_earlier_files(case, monkeypatch, capsys):
    install(monkeypatch, stop_at={("posteriors", 53)})
    with pytest.raises(Reached):                                           # a crash during seed 53
        held(case)
    a1 = json.loads((case.res / "held_started.json").read_text())["attempts"]
    assert len(a1) == 1 and set(a1[0]["episodes_sha256"]) == {"52", "53"}
    kept = {s: (case.res / f"held_episodes_seed{s}.npz").read_bytes() for s in (52, 53)}
    capsys.readouterr()
    assert refused(held(case), capsys, says="held_started.json exists")
    ns = install(monkeypatch)
    assert held(case, after_crash=True) == 0
    att = json.loads((case.res / "held_started.json").read_text())["attempts"]
    assert len(att) == 2 and att[1]["flags"]["after_crash"] is True and att[0] == a1[0]
    assert att[1]["episodes_sha256"] == {str(s): seed_hashes(s) for s in SEEDS}
    assert all((case.res / f"held_episodes_seed{s}.npz").read_bytes() == kept[s] for s in (52, 53))  # never rewritten
    assert ("write", "held_episodes_seed52.npz") not in ns.events and ("write", "held_episodes_seed54.npz") in ns.events
    assert (case.res / "held_pass.json").is_file()
    capsys.readouterr()
    assert refused(held(case, after_crash=True), capsys, says="held_pass.json exists")


def test_after_crash_with_other_hashes_refuses_and_a_third_attempt_refuses(case, monkeypatch, capsys):
    install(monkeypatch, stop_at={("posteriors", 52)})
    with pytest.raises(Reached):
        held(case)
    install(monkeypatch, salt="other")
    assert refused(held(case, after_crash=True), capsys, says="seed 52: the episode SHA-256s differ")
    att = json.loads((case.res / "held_started.json").read_text())["attempts"]
    assert len(att) == 2 and att[1]["episodes_sha256"] == {"52": seed_hashes(52, "other")}    # recorded first
    assert not (case.res / "sensitivity_held.json").exists() and not (case.res / "held_pass.json").exists()
    install(monkeypatch)
    assert refused(held(case, after_crash=True), capsys, says="already records 2 attempts")
    assert len(json.loads((case.res / "held_started.json").read_text())["attempts"]) == 2


def test_after_crash_compares_files_an_earlier_attempt_wrote(case, monkeypatch, capsys):
    started_file(case.res / "held_started.json", ({}, {}))
    (case.res / "sensitivity_held.json").write_text(json.dumps({"N": 1}))
    install(monkeypatch)
    with pytest.raises(AssertionError, match="sensitivity_held.json, written by an earlier attempt"):
        held(case, after_crash=True)


def test_a_repeat_across_seeds_is_caught(case, monkeypatch, capsys):
    """assert_distinct is handed every held seed so far: a hash of seed 53 equal to one of seed 52 stops the run."""
    install(monkeypatch, salts={53: "x"})
    monkeypatch.setattr(RH, "seed_bundle", _repeat(RH.seed_bundle))
    with pytest.raises(AssertionError, match="repeats the episode hash of seed 52"):
        held(case)
    att = json.loads((case.res / "held_started.json").read_text())["attempts"][0]
    assert set(att["episodes_sha256"]) == {"52", "53"} and not (case.res / "held_episodes_seed53.npz").exists()


def _repeat(inner):
    def seed_bundle(env, mode, seed, n_per_pair, on_episodes=None):
        def report(eps):
            if seed == 53:
                eps.sha = {**eps.sha, "style__genre": seed_hashes(52)["emotion__genre"]}
            on_episodes(eps)
        return inner(env, mode, seed, n_per_pair, on_episodes=report)
    return seed_bundle


def test_a_recorded_hash_is_caught(case, monkeypatch, capsys):
    install(monkeypatch, recorded={49: seed_hashes(53)})
    with pytest.raises(AssertionError, match="equal recorded ones"):
        held(case)


# ---------------------------------------------------------------- --fix 1 and --reserve, end to end (stubbed)

def test_fix_writes_its_own_pass_and_reuses_the_read(case, monkeypatch, capsys):
    install(monkeypatch)
    assert held(case) == 0
    first = {n: (case.res / n).read_bytes() for n in ("held_pass.json", "held_arrays.npz", "sensitivity_held.json",
                                                      "held_episodes_seed52.npz")}
    write_ledger(case.ledger, ledger_line(episodes=all_hashes()))
    write_records(case.res, smoke_names=("smoke_record.json", "smoke_record_fix1.json"))
    capsys.readouterr()
    ns = install(monkeypatch)
    assert held(case, fix=1) == 0
    assert capsys.readouterr().out.strip().splitlines()[-1].startswith("held pass written ")
    for n, b in first.items():
        assert (case.res / n).read_bytes() == b, n                          # kept, never rewritten
    assert {("write", n) for n in first} & set(ns.events) == set()
    pr = json.loads((case.res / "held_pass_fix1.json").read_text())
    assert pr["kind"] == "fix1" and pr["attempt"] == 2 and (case.res / "held_arrays_fix1.npz").is_file()
    att = json.loads((case.res / "held_started.json").read_text())["attempts"]
    assert att[1]["flags"]["fix"] == 1 and att[1]["smoke_record"]["name"] == "smoke_record_fix1.json"
    assert not (case.res / "sensitivity_held_fix1.json").exists()
    assert refused(held(case, fix=1), capsys, says="held_pass_fix1.json exists")


def test_fix_with_other_episodes_refuses(case, monkeypatch, capsys):
    install(monkeypatch)
    assert held(case) == 0
    write_ledger(case.ledger, ledger_line(episodes=all_hashes()))
    write_records(case.res, smoke_names=("smoke_record.json", "smoke_record_fix1.json"))
    capsys.readouterr()
    install(monkeypatch, salts={54: "other"})
    assert refused(held(case, fix=1), capsys, says="seed 54: the episode SHA-256s differ")
    assert not (case.res / "held_pass_fix1.json").exists()


def test_reserve_writes_the_reserve_files(case, monkeypatch, capsys):
    install(monkeypatch)
    assert held(case) == 0
    (case.res / "held_verdict.json").write_text("{}")                      # as the apply step would
    write_ledger(case.ledger, ledger_line(report="sha"), ledger_line("H5-R"))
    write_records(case.res, smoke_names=("smoke_record.json", "smoke_record_reserve.json"))
    before = {p.name: p.read_bytes() for p in case.res.iterdir()}
    capsys.readouterr()
    assert held(case, reserve=True) == 0
    new = sorted(p.name for p in case.res.iterdir() if p.name not in before)
    assert new == sorted(["held_started_reserve.json", "sensitivity_held_reserve.json", "held_arrays_reserve.npz",
                          "held_pass_reserve.json"])
    assert all((case.res / n).read_bytes() == b for n, b in before.items())  # the originals kept
    assert (case.folder / "held_started_reserve.json").read_bytes() == (
        case.res / "held_started_reserve.json").read_bytes()
    assert json.loads((case.res / "held_pass_reserve.json").read_text())["kind"] == "reserve"
    capsys.readouterr()
    assert refused(held(case, reserve=True), capsys, says="held_pass_reserve.json exists")


# ---------------------------------------------------------------- the bundle's row scope; held_seed_bundle

def test_posteriors_on_a_union_are_caught(case, monkeypatch, capsys):
    install(monkeypatch, post_extra=(int(SELECTION[0]),))
    with pytest.raises(AssertionError, match="not finite on exactly split.held"):
        held(case)


def test_held_seed_bundle_is_the_held_read_bundle(monkeypatch):
    seen = []
    monkeypatch.setattr(RH, "read_seed_bundle", lambda *a: seen.append(a) or "bundle")
    env, cb = object(), object()
    assert RH.held_seed_bundle(env, 53, on_episodes=cb) == "bundle" and seen == [(env, "held", 53, R.N_PER_PAIR, cb)]
    for s in (42, 9001, 55):
        with pytest.raises(AssertionError, match="not a held seed"):
            RH.held_seed_bundle(env, s)


def test_only_the_allowed_scores_before_the_verdict(case, monkeypatch, capsys):
    install(monkeypatch, extra_key="diag")
    with pytest.raises(AssertionError, match="only .* before the verdict"):
        held(case)
    assert not (case.res / "held_arrays.npz").exists()


# ---------------------------------------------------------------- smoke mode (stubbed)

def test_smoke_runs_the_same_steps_under_results_smoke(case, monkeypatch, capsys):
    for n in ("held_started.json", "held_pass.json", "held_verdict.json"):   # the real read's files do not matter
        (case.res / n).write_text("{}")
    ns = install(monkeypatch)
    assert RH.run_smoke(results=case.res, here=HERE) == 0
    out = capsys.readouterr().out
    smoke = case.res / "smoke"
    assert out.strip().splitlines()[-1] == f"smoke held pass written {R.sha256_file(smoke / 'held_pass.json')}"
    assert not DECIMAL.search(out.replace(str(case.tmp), ""))
    assert sorted(p.name for p in smoke.iterdir()) == sorted(
        ["held_started.json", "folder", "sensitivity_held.json", "held_arrays.npz", "held_pass.json"]
        + [f"held_episodes_seed{s}.npz" for s in R.SMOKE_SEEDS])
    pr = json.loads((smoke / "held_pass.json").read_text())
    assert pr["mode"] == "smoke" and pr["seeds"] == list(R.SMOKE_SEEDS) and pr["n_episodes"] == 9 * R.N_SMOKE
    att = json.loads((smoke / "held_started.json").read_text())["attempts"]
    assert len(att) == 1 and att[0]["flags"]["smoke"] is True and "ledger" not in att[0]
    assert (smoke / "folder/held_started.json").read_bytes() == (smoke / "held_started.json").read_bytes()
    assert ("eligible", len(SELECTION)) in ns.events and ns.include_pm == [False] * 3
    assert json.loads((smoke / "sensitivity_held.json").read_text())["N"] == 9 * R.N_SMOKE
    assert [e[0] for e in ns.events[:5]] == ["inputs", "recorded", "picks", "setup", "head_guard"]
    # a second smoke in the same folder overwrites its files
    install(monkeypatch, salt="again")
    assert RH.run_smoke(results=case.res, here=HERE) == 0
    att = json.loads((smoke / "held_started.json").read_text())["attempts"]
    assert len(att) == 1 and att[0]["episodes_sha256"]["9001"] == seed_hashes(9001, "again")


def test_smoke_refuses_after_a_smoke_verdict_and_takes_a_subdir(case, monkeypatch, capsys):
    install(monkeypatch)
    (case.res / "smoke").mkdir()
    (case.res / "smoke/held_verdict.json").write_text("{}")
    assert refused(RH.run_smoke(results=case.res, here=HERE), capsys, says="--smoke-subdir")
    assert RH.run_smoke(results=case.res, subdir="fix1", here=HERE) == 0
    assert (case.res / "smoke/fix1/held_pass.json").is_file()
    capsys.readouterr()
    for bad in ("../x", "a/b", "", "fix 1"):
        assert refused(RH.run_smoke(results=case.res, subdir=bad, here=HERE), capsys, says="plain folder name")


def test_smoke_needs_no_ledger_and_no_smoke_record(case, monkeypatch, capsys):
    (case.res / "smoke_record.json").unlink()
    case.ledger.unlink()
    install(monkeypatch)
    assert RH.run_smoke(results=case.res, here=HERE) == 0


def test_smoke_checks_the_heads_and_records_like_the_read(case, monkeypatch, capsys):
    install(monkeypatch, head_ok=False)
    assert refused(RH.run_smoke(results=case.res, here=HERE), capsys, says="selection posteriors differ")
    edit(case.res / RH.REGRESSION_NAME, passed=False)
    install(monkeypatch)
    assert refused(RH.run_smoke(results=case.res, here=HERE), capsys, says="did not pass")
    assert not (case.res / "smoke").exists()


# ---------------------------------------------------------------- guard mutations (on copies)

def mut_case(case, mod):
    """The ledger names the mutant's own SHA-256 (the runner refuses any other script)."""
    write_ledger(case.ledger, ledger_line(scripts=[mod.runner_sha256()]))
    return mod


def test_guard_verdict_exists(case, monkeypatch, capsys):
    (case.res / "held_verdict.json").write_text("{}")
    install(monkeypatch, stop_at={"inputs"})
    assert refused(held(case), capsys)
    mod = mut_case(case, mutant(case.tmp, "run_r6_held.py", "verdict_exists"))
    install(monkeypatch, mod, stop_at={"inputs"})
    with pytest.raises(Reached):
        held(case, mod)


def test_guard_pass_exists(case, monkeypatch, capsys):
    (case.res / "held_pass.json").write_text("{}")
    mod = mut_case(case, mutant(case.tmp, "run_r6_held.py", "pass_exists"))
    install(monkeypatch, mod, stop_at={"inputs"})
    with pytest.raises(Reached):
        held(case, mod)


def test_guard_started_exists(case, monkeypatch, capsys):
    started_file(case.res / "held_started.json", ({}, {}), copy=False)
    mod = mut_case(case, mutant(case.tmp, "run_r6_held.py", "started_exists"))
    install(monkeypatch, mod, stop_at={"inputs"})
    with pytest.raises(Reached):
        held(case, mod)


def test_guard_after_crash_needs_start(case, monkeypatch, capsys):
    mod = mut_case(case, mutant(case.tmp, "run_r6_held.py", "after_crash_needs_start"))
    install(monkeypatch, mod, stop_at={"inputs"})
    assert refused(held(case, mod, after_crash=True), capsys, says="is missing or differs")    # the copy check's


def test_guard_two_attempts(case, monkeypatch, capsys):
    started_file(case.res / "held_started.json", ({}, {}), ({"after_crash": True}, {}))
    mod = mut_case(case, mutant(case.tmp, "run_r6_held.py", "two_attempts"))
    install(monkeypatch, mod, stop_at={"inputs"})
    with pytest.raises(Reached):
        held(case, mod, after_crash=True)


def test_guard_ledger_row(case, monkeypatch, capsys):
    mod = mutant(case.tmp, "run_r6_held.py", "ledger_row")
    write_ledger(case.ledger)
    install(monkeypatch, mod, stop_at={"inputs"})
    with pytest.raises(IndexError):                        # the refusal is gone: the run goes on and crashes
        held(case, mod)


def test_guard_ledger_sha(case, monkeypatch, capsys):
    mod = mutant(case.tmp, "run_r6_held.py", "ledger_sha")
    write_ledger(case.ledger, ledger_line(scripts=["3" * 64]))
    install(monkeypatch, mod, stop_at={"inputs"})
    with pytest.raises(Reached):
        held(case, mod)


def test_guard_ledger_report(case, monkeypatch, capsys):
    mod = mutant(case.tmp, "run_r6_held.py", "ledger_report")
    write_ledger(case.ledger, ledger_line(scripts=[mod.runner_sha256()], report="done"))
    install(monkeypatch, mod, stop_at={"inputs"})
    with pytest.raises(Reached):
        held(case, mod)


def test_guard_fix_needs_pass(case, monkeypatch, capsys):
    fix_ready(case)
    (case.res / "held_pass.json").rename(case.tmp / "kept.json")
    mod = mutant(case.tmp, "run_r6_held.py", "fix_needs_pass")
    write_ledger(case.ledger, ledger_line(scripts=[mod.runner_sha256()], episodes=all_hashes()))
    install(monkeypatch, mod, stop_at={"inputs"})
    with pytest.raises(Reached):
        held(case, mod, fix=1)


def test_guard_fix_smoke(case, monkeypatch, capsys):
    fix_ready(case)
    (case.res / "smoke_record_fix1.json").unlink()
    mod = mutant(case.tmp, "run_r6_held.py", "fix_smoke")
    write_ledger(case.ledger, ledger_line(scripts=[mod.runner_sha256()], episodes=all_hashes()))
    install(monkeypatch, mod, stop_at={"inputs"})
    assert refused(held(case, mod, fix=1), capsys, says="smoke_record_fix1.json does not exist")   # step 2's refusal


def test_guard_fix_once(case, monkeypatch, capsys):
    fix_ready(case)
    started_file(case.res / "held_started.json", ({}, nine()), ({"fix": 1}, nine()))
    mod = mutant(case.tmp, "run_r6_held.py", "fix_once")
    write_ledger(case.ledger, ledger_line(scripts=[mod.runner_sha256()], episodes=all_hashes()))
    install(monkeypatch, mod, stop_at={"inputs"})
    with pytest.raises(Reached):
        held(case, mod, fix=1)


def test_guard_fix_h5_episodes(case, monkeypatch, capsys):
    fix_ready(case)
    mod = mutant(case.tmp, "run_r6_held.py", "fix_h5_episodes")
    write_ledger(case.ledger, ledger_line(scripts=[mod.runner_sha256()]))
    install(monkeypatch, mod, stop_at={"inputs"})
    with pytest.raises(Reached):
        held(case, mod, fix=1)


def reserve_mut(case, mod):
    reserve_ready(case)
    write_ledger(case.ledger, ledger_line(report="abc"), ledger_line("H5-R", scripts=[mod.runner_sha256()]))
    return mod


def test_guard_reserve_started(case, monkeypatch, capsys):
    mod = reserve_mut(case, mutant(case.tmp, "run_r6_held.py", "reserve_started"))
    started_file(case.res / "held_started_reserve.json", ({"reserve": True}, {}), copy=False)
    install(monkeypatch, mod, stop_at={"inputs"})
    with pytest.raises(Reached):
        held(case, mod, reserve=True)


def test_guard_reserve_needs_verdict(case, monkeypatch, capsys):
    mod = reserve_mut(case, mutant(case.tmp, "run_r6_held.py", "reserve_needs_verdict"))
    (case.res / "held_verdict.json").rename(case.tmp / "kept.json")
    install(monkeypatch, mod, stop_at={"inputs"})
    with pytest.raises(Reached):
        held(case, mod, reserve=True)


def test_guard_reserve_hashes(case, monkeypatch, capsys):
    mod = reserve_mut(case, mutant(case.tmp, "run_r6_held.py", "reserve_hashes"))
    started_file(case.res / "held_started.json", ({}, {52: seed_hashes(52)}))
    install(monkeypatch, mod, stop_at={"inputs"})
    with pytest.raises(Reached):
        held(case, mod, reserve=True)


def test_guard_smoke_missing(case, monkeypatch, capsys):
    (case.res / "smoke_record.json").unlink()
    mod = mut_case(case, mutant(case.tmp, "run_r6_held.py", "smoke_missing"))
    install(monkeypatch, mod, stop_at={"inputs"})
    with pytest.raises(FileNotFoundError):
        held(case, mod)


def test_guard_smoke_passed(case, monkeypatch, capsys):
    edit(case.res / "smoke_record.json", passed=False)
    mod = mut_case(case, mutant(case.tmp, "run_r6_held.py", "smoke_passed"))
    install(monkeypatch, mod, stop_at={"inputs"})
    with pytest.raises(Reached):
        held(case, mod)


def test_guard_smoke_stale(case, monkeypatch, capsys):
    edit(case.res / "smoke_record.json",
         module_sha256={**R.r6_module_shas(), RH._key("r6_context.py", HERE): "0" * 64})
    mod = mut_case(case, mutant(case.tmp, "run_r6_held.py", "smoke_stale"))
    install(monkeypatch, mod, stop_at={"inputs"})
    with pytest.raises(Reached):
        held(case, mod)


def seed42_mut(case, monkeypatch, capsys, mod, change, exc):
    """The record changed as ``change`` says: the runner refuses at step 3; the mutant goes past the guard."""
    change(case.res)
    install(monkeypatch, stop_at={"picks"})
    assert refused(held(case), capsys)
    mut_case(case, mod)
    install(monkeypatch, mod, stop_at={"picks"})
    with pytest.raises(exc):
        held(case, mod)


def stale(name):
    return {**R.r6_module_shas(), RH._key(name, HERE): "0" * 64}


def test_guard_regression_missing(case, monkeypatch, capsys):
    seed42_mut(case, monkeypatch, capsys, mutant(case.tmp, "run_r6_held.py", "regression_missing"),
               lambda res: (res / RH.REGRESSION_NAME).unlink(), FileNotFoundError)


def edit_regression(res, **changes):
    """Change the regression record and keep the sensitivity record naming it (only the change is wrong)."""
    edit(res / RH.REGRESSION_NAME, **changes)
    edit(res / RH.SENS42_NAME, regression_sha256=R.sha256_file(res / RH.REGRESSION_NAME))


def test_guard_regression_passed(case, monkeypatch, capsys):
    seed42_mut(case, monkeypatch, capsys, mutant(case.tmp, "run_r6_held.py", "regression_passed"),
               lambda res: edit_regression(res, passed=False), Reached)


def test_guard_regression_stale(case, monkeypatch, capsys):
    seed42_mut(case, monkeypatch, capsys, mutant(case.tmp, "run_r6_held.py", "regression_stale"),
               lambda res: edit_regression(res, module_sha256=stale("r6_bundle.py")), Reached)


def test_guard_sens42_missing(case, monkeypatch, capsys):
    seed42_mut(case, monkeypatch, capsys, mutant(case.tmp, "run_r6_held.py", "sens42_missing"),
               lambda res: (res / RH.SENS42_NAME).unlink(), FileNotFoundError)


def test_guard_sens42_current(case, monkeypatch, capsys):
    seed42_mut(case, monkeypatch, capsys, mutant(case.tmp, "run_r6_held.py", "sens42_current"),
               lambda res: edit(res / RH.SENS42_NAME, regression_sha256="0" * 64), Reached)


def test_guard_sens42_sigma(case, monkeypatch, capsys):
    seed42_mut(case, monkeypatch, capsys, mutant(case.tmp, "run_r6_held.py", "sens42_sigma"),
               lambda res: edit(res / RH.SENS42_NAME, S2={"sigma_a2": -1.0, "sigma_eps2": 1.0}), Reached)


def test_guard_head_check(case, monkeypatch, capsys):
    mod = mut_case(case, mutant(case.tmp, "run_r6_held.py", "head_check"))
    install(monkeypatch, mod, head_ok=False, stop_at={"context"})
    with pytest.raises(Reached):                           # the read starts despite the failed check
        held(case, mod)
    assert (case.res / "held_started.json").is_file()


def test_guard_coef_sha(case, monkeypatch, capsys):
    mod = mut_case(case, mutant(case.tmp, "run_r6_held.py", "coef_sha"))
    install(monkeypatch, mod, coef={**COEF, "affect": {"img": "f" * 64, "txt": "b" * 64}}, stop_at={"context"})
    with pytest.raises(Reached):
        held(case, mod)
    assert (case.res / "held_started.json").is_file()


def test_guard_post_rows(case, monkeypatch, capsys):
    mod = mut_case(case, mutant(case.tmp, "run_r6_held.py", "post_rows"))
    install(monkeypatch, mod, post_extra=(int(SELECTION[0]),), stop_at={"eligible"})
    with pytest.raises(Reached):
        held(case, mod)


def test_guard_rerun_hashes(case, monkeypatch, capsys):
    started_file(case.res / "held_started.json", ({}, {52: seed_hashes(52)}))      # no episode file was saved
    mod = mut_case(case, mutant(case.tmp, "run_r6_held.py", "rerun_hashes"))
    install(monkeypatch, mod, salt="other", stop_at={"eligible"})
    with pytest.raises(Reached):
        held(case, mod, after_crash=True)


def test_guard_episode_file(case, monkeypatch, capsys):
    started_file(case.res / "held_started.json", ({}, {}))
    np.savez(case.res / "held_episodes_seed52.npz", seed=np.int64(52),
             **{f"sha__{p}": np.array(h) for p, h in seed_hashes(52, "other").items()})
    install(monkeypatch, stop_at={"eligible"})
    with pytest.raises(AssertionError, match="holds other episodes"):
        held(case, after_crash=True)
    started_file(case.res / "held_started.json", ({}, {}))
    mod = mut_case(case, mutant(case.tmp, "run_r6_held.py", "episode_file"))
    install(monkeypatch, mod, stop_at={"eligible"})
    with pytest.raises(Reached):
        held(case, mod, after_crash=True)


def test_guard_episodes_reported(case, monkeypatch, capsys):
    install(monkeypatch, skip_report=True)
    with pytest.raises(AssertionError, match="not reported to the recorder"):
        held(case)
    mod = mutant(case.tmp, "run_r6_held.py", "episodes_reported")
    write_ledger(case.ledger, ledger_line(scripts=[mod.runner_sha256()]))
    (case.res / "held_started.json").rename(case.tmp / "kept.json")
    (case.folder / "held_started.json").rename(case.tmp / "kept_copy.json")         # a clean first read again
    install(monkeypatch, mod, skip_report=True, stop_at={"eligible"})
    with pytest.raises(Reached):
        held(case, mod)


def test_guard_kept_json(case, monkeypatch, capsys):
    started_file(case.res / "held_started.json", ({}, {}))
    (case.res / "sensitivity_held.json").write_text(json.dumps({"N": 1}))
    mod = mut_case(case, mutant(case.tmp, "run_r6_held.py", "kept_json"))
    install(monkeypatch, mod, stop_at={"score"})
    with pytest.raises(Reached):
        held(case, mod, after_crash=True)
    assert json.loads((case.res / "sensitivity_held.json").read_text()) == {"N": 1}


def test_guard_kept_npz(case, monkeypatch, capsys):
    started_file(case.res / "held_started.json", ({}, {}))
    np.savez(case.res / "held_arrays.npz", cl=np.zeros(3))
    install(monkeypatch, stop_at={"pass"})
    with pytest.raises(AssertionError, match="held_arrays.npz, written by an earlier attempt"):
        held(case, after_crash=True)
    started_file(case.res / "held_started.json", ({}, {}))
    mod = mut_case(case, mutant(case.tmp, "run_r6_held.py", "kept_npz"))
    install(monkeypatch, mod, stop_at={"pass"})
    with pytest.raises(Reached):
        held(case, mod, after_crash=True)


def test_guard_allowed_keys(case, monkeypatch, capsys):
    mod = mut_case(case, mutant(case.tmp, "run_r6_held.py", "allowed_keys"))
    install(monkeypatch, mod, extra_key="diag", stop_at={"pass"})
    with pytest.raises(Reached):
        held(case, mod)


def test_guard_never_overwrite(tmp_path):
    p = tmp_path / "held_pass.json"
    p.write_text("first")
    with pytest.raises(AssertionError, match="never overwritten"):
        RH.write_new(p, b"second")
    RH.write_new(p, b"smoke", overwrite=True)
    assert p.read_text() == "smoke"
    p.write_text("first")
    mutant(tmp_path, "run_r6_held.py", "never_overwrite").write_new(p, b"second")
    assert p.read_text() == "second"


def test_guard_smoke_verdict(case, monkeypatch, capsys):
    (case.res / "smoke").mkdir()
    (case.res / "smoke/held_verdict.json").write_text("{}")
    mod = mutant(case.tmp, "run_r6_held.py", "smoke_verdict")
    install(monkeypatch, mod, stop_at={"inputs"})
    with pytest.raises(Reached):
        mod.run_smoke(results=case.res, here=HERE)


# ---------------------------------------------------------------- contracts section 7, amendment 16:22 (ticket 13)

def test_arrays_layout_and_names(case, monkeypatch, capsys):
    """held_arrays*.npz: named after the pass file; <scorer>__<metric> float64 (n,) for exactly CORE_SCORERS and
    score_seed's metrics; cl, pair_index, seed_index int64; seed_index the seed's position; seeds in order."""
    names = {k: RH.read_names(k) for k in ("held", "fix1", "reserve", "smoke")}
    assert {k: (n.pass_, n.arrays) for k, n in names.items()} == {
        "held": ("held_pass.json", "held_arrays.npz"), "fix1": ("held_pass_fix1.json", "held_arrays_fix1.npz"),
        "reserve": ("held_pass_reserve.json", "held_arrays_reserve.npz"),
        "smoke": ("held_pass.json", "held_arrays.npz")}
    ns = install(monkeypatch)
    assert held(case) == 0
    n = 3 * R.N_PER_PAIR
    with np.load(case.res / "held_arrays.npz") as z:
        assert set(z.files) == {f"{s}__{m}" for s in S.CORE_SCORERS for m in METRICS} | {
            "cl", "pair_index", "seed_index"}
        assert not any(k.split("__")[0] in S.PM_SCORERS + ("r1_cf",) for k in z.files)
        for k in z.files:
            want = np.int64 if k in ("cl", "pair_index", "seed_index") else np.float64
            assert z[k].dtype == want and z[k].shape == (3 * n,), k
        assert np.array_equal(z["seed_index"], np.repeat(np.arange(3, dtype=np.int64), n))       # not 52, 53, 54
        for i, s in enumerate(SEEDS):
            for m in METRICS:
                assert np.array_equal(z[f"B1__{m}"][i * n:(i + 1) * n], ns.scored[s]["B1"][m])
            assert np.array_equal(z["pair_index"][i * n:(i + 1) * n], ns.scored[s]["pair_index"])
    sc = [fake_scored(s, 3 * R.N_SMOKE) for s in R.SMOKE_SEEDS]
    sc[1]["cl"] = sc[1]["cl"].astype(np.int32)
    got = RH.read_arrays(sc)
    assert got["cl"].dtype == np.int64 and np.array_equal(got["cl"], np.concatenate([x["cl"] for x in sc]))
    sc[2]["aff_fused"]["r1"] = sc[2]["aff_fused"]["r1"].astype(np.float32)
    with pytest.raises(AssertionError, match="float64"):
        RH.read_arrays(sc)
    sc = [fake_scored(s, 3 * R.N_SMOKE) for s in R.SMOKE_SEEDS]
    del sc[0]["rca"]["strict"]
    with pytest.raises(AssertionError, match="metrics"):
        RH.read_arrays(sc)


def test_seed_bundle_keeps_its_signature_and_writes_nothing(tmp_path, monkeypatch):
    """Ticket 13 builds held bundles after the verdict with seed_bundle(env, "held", seed, 4096): it only builds the
    context and the bundle; the started-file recording lives in the read's on_episodes callback."""
    import builtins
    import inspect
    import io
    import os
    sig = inspect.signature(RH.seed_bundle)
    assert list(sig.parameters) == ["env", "mode", "seed", "n_per_pair", "on_episodes"]
    assert sig.parameters["on_episodes"].default is None
    calls = [n.func for n in ast.walk(ast.parse(inspect.getsource(RH.seed_bundle))) if isinstance(n, ast.Call)]
    assert sorted(ast.unparse(f) for f in calls) == ["B.build_bundle_r6", "X.RowContext"]
    writes, seen = [], []
    real_open, real_io_open, real_replace = builtins.open, io.open, os.replace

    def opener(real):
        def f(file, mode="r", *a, **k):
            if any(c in mode for c in "wax+"):
                writes.append(str(file))
            return real(file, mode, *a, **k)
        return f

    monkeypatch.setattr(builtins, "open", opener(real_open))
    monkeypatch.setattr(io, "open", opener(real_io_open))
    monkeypatch.setattr(os, "replace", lambda *a, **k: writes.append(a) or real_replace(*a, **k))
    monkeypatch.setattr(RH.X, "RowContext", lambda *a, **k: seen.append(("ctx", a[:3], k)) or "ctx")
    monkeypatch.setattr(RH.B, "build_bundle_r6", lambda ctx, readers, pm: seen.append(("bundle", ctx)) or "bundle")
    env = SimpleNamespace(data="d", split="s", labels="l", heads="h", value_sets="v", readers="r", pm="p")
    monkeypatch.chdir(tmp_path)
    assert RH.seed_bundle(env, "held", 52, R.N_PER_PAIR) == "bundle"
    assert seen == [("ctx", ("held", 52, "d"), {"on_episodes": None}), ("bundle", "ctx")] and writes == []
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------- fix round (controller, 2026-10-09): the committed
# copy, the ledger's episode cell, the crash smoke record, "passed" required

def fresh_results(case, name):
    """Another results folder with current records only (the read's results moved or copied away)."""
    case.res = write_records(case.tmp / name)
    return case.res


def test_a_moved_results_folder_cannot_start_the_read_again(case, monkeypatch, capsys):
    install(monkeypatch)
    assert held(case) == 0
    copy = (case.folder / "held_started.json").read_bytes()
    fresh_results(case, "results_moved")                    # no held_started.json, no pass, no verdict
    install(monkeypatch, stop_at={"inputs"})
    capsys.readouterr()
    assert refused(held(case), capsys, says="exists: a read has started (its committed copy)")
    assert (case.folder / "held_started.json").read_bytes() == copy and nothing_started_in(case.res)
    write_ledger(case.ledger, ledger_line(episodes=all_hashes()))           # the run chat filled the episode cell
    (case.folder / "held_started.json").rename(case.tmp / "kept_copy.json")
    assert refused(held(case), capsys, says="its episode cell is")
    for cell in ("pending", "-", "", "(pending), x"):
        write_ledger(case.ledger, ledger_line(episodes=cell))
        assert refused(held(case), capsys, says="its episode cell is"), cell
    write_ledger(case.ledger, ledger_line())
    with pytest.raises(Reached):                            # every first-read check met again
        held(case)


def nothing_started_in(res):
    return not any(p.name.startswith("held_started") for p in res.iterdir())


def test_reserve_checks_its_own_copy_and_row_h5r(case, monkeypatch, capsys):
    install(monkeypatch, stop_at={"inputs"})
    reserve_ready(case)
    (case.folder / "held_started_reserve.json").write_text("{}")
    assert refused(held(case, reserve=True), capsys, says="held_started_reserve.json exists: a read has started")
    (case.folder / "held_started_reserve.json").rename(case.tmp / "kept.json")
    write_ledger(case.ledger, ledger_line(report="abc"), ledger_line("H5-R", episodes=all_hashes()))
    assert refused(held(case, reserve=True), capsys, says="ledger row H5-R: its episode cell")
    write_ledger(case.ledger, ledger_line(report="abc", episodes=all_hashes()), ledger_line("H5-R"))
    with pytest.raises(Reached):                            # H5's filled cell does not matter to the reserve
        held(case, reserve=True)


def test_rerun_needs_the_same_bytes_in_the_committed_copy(case, monkeypatch, capsys):
    install(monkeypatch, stop_at={"inputs"})
    started_file(case.res / "held_started.json", ({}, {52: seed_hashes(52)}), copy=False)
    assert refused(held(case, after_crash=True), capsys, says="is missing or differs")
    (case.tmp / "other").mkdir()
    other = started_file(case.tmp / "other/held_started.json", ({}, {52: seed_hashes(52, "x")}), copy=False)
    case.folder.mkdir(exist_ok=True)
    (case.folder / "held_started.json").write_bytes(other.read_bytes())
    assert refused(held(case, after_crash=True), capsys, says="is missing or differs")
    (case.folder / "held_started.json").write_bytes((case.res / "held_started.json").read_bytes() + b"\n")
    assert refused(held(case, after_crash=True), capsys, says="is missing or differs")     # bytes, not content
    (case.folder / "held_started.json").write_bytes((case.res / "held_started.json").read_bytes())
    with pytest.raises(Reached):
        held(case, after_crash=True)
    fix_ready(case)
    (case.folder / "held_started.json").rename(case.tmp / "kept.json")
    assert refused(held(case, fix=1), capsys, says="is missing or differs")
    (case.tmp / "kept.json").rename(case.folder / "held_started.json")
    with pytest.raises(Reached):
        held(case, fix=1)


def test_guard_copy_exists(case, monkeypatch, capsys):
    case.folder.mkdir()
    (case.folder / "held_started.json").write_text("{}")
    install(monkeypatch, stop_at={"inputs"})
    assert refused(held(case), capsys, says="its committed copy")
    mod = mut_case(case, mutant(case.tmp, "run_r6_held.py", "copy_exists"))
    install(monkeypatch, mod, stop_at={"inputs"})
    with pytest.raises(Reached):
        held(case, mod)


def test_guard_episodes_pending(case, monkeypatch, capsys):
    install(monkeypatch, stop_at={"inputs"})
    write_ledger(case.ledger, ledger_line(episodes=all_hashes()))
    assert refused(held(case), capsys, says="its episode cell")
    mod = mutant(case.tmp, "run_r6_held.py", "episodes_pending")
    write_ledger(case.ledger, ledger_line(scripts=[mod.runner_sha256()], episodes=all_hashes()))
    install(monkeypatch, mod, stop_at={"inputs"})
    with pytest.raises(Reached):
        held(case, mod)


def test_guard_copy_matches(case, monkeypatch, capsys):
    started_file(case.res / "held_started.json", ({}, {}), copy=False)
    install(monkeypatch, stop_at={"inputs"})
    assert refused(held(case, after_crash=True), capsys, says="is missing or differs")
    mod = mut_case(case, mutant(case.tmp, "run_r6_held.py", "copy_matches"))
    install(monkeypatch, mod, stop_at={"inputs"})
    with pytest.raises(Reached):
        held(case, mod, after_crash=True)


def test_after_crash_reads_the_crash_smoke_record(case, monkeypatch, capsys):
    """Code corrected after a crash gets its own smoke: --after-crash checks smoke_record_crash1.json when it exists,
    else smoke_record.json; the first read never reads it."""
    install(monkeypatch, stop_at={"inputs"})
    started_file(case.res / "held_started.json", ({}, {}))
    stale = {**R.r6_module_shas(), RH._key("r6_score.py", HERE): "0" * 64}
    edit(case.res / "smoke_record.json", module_sha256=stale)
    assert RH.latest_smoke_record(case.res, "held", after_crash=True).name == "smoke_record.json"
    assert refused(held(case, after_crash=True), capsys, says="smoke_record.json: the SHA-256s")
    write_records(case.res, smoke_names=("smoke_record_crash1.json",))   # the corrected code's smoke, current
    edit(case.res / "smoke_record.json", module_sha256=stale)
    assert RH.latest_smoke_record(case.res, "held", after_crash=True).name == "smoke_record_crash1.json"
    with pytest.raises(Reached):
        held(case, after_crash=True)
    edit(case.res / "smoke_record_crash1.json", module_sha256=stale)
    edit(case.res / "smoke_record.json", module_sha256=R.r6_module_shas())
    assert refused(held(case, after_crash=True), capsys, says="smoke_record_crash1.json: the SHA-256s")
    assert RH.latest_smoke_record(case.res, "held").name == "smoke_record_crash1.json"    # the latest, for every read
    assert RH.latest_smoke_record(case.res, "fix1").name == "smoke_record_fix1.json"


def test_crash_then_corrected_code_then_after_crash(case, monkeypatch, capsys):
    """The legitimate walk: a crash, the corrected runner's SHA appended last in H5's script cell, its smoke record
    smoke_record_crash1.json, then --after-crash completes and records that smoke."""
    install(monkeypatch, stop_at={("posteriors", 53)})
    with pytest.raises(Reached):
        held(case)
    me = R.sha256_file(HERE / "run_r6_held.py")
    write_ledger(case.ledger, ledger_line(scripts=["3" * 64, me]))
    write_records(case.res, smoke_names=("smoke_record_crash1.json",))
    install(monkeypatch)
    capsys.readouterr()
    assert held(case, after_crash=True) == 0
    att = json.loads((case.res / "held_started.json").read_text())["attempts"]
    assert att[1]["smoke_record"]["name"] == "smoke_record_crash1.json" and att[1]["flags"]["after_crash"]
    assert (case.folder / "held_started.json").read_bytes() == (case.res / "held_started.json").read_bytes()


def test_smoke_record_needs_passed_true(case, no_work, capsys):
    rec = json.loads((case.res / "smoke_record.json").read_text())
    del rec["passed"]
    (case.res / "smoke_record.json").write_text(json.dumps(rec))
    assert refused(held(case), capsys, says="did not pass")
    for v in ("true", 1, None):
        edit(case.res / "smoke_record.json", passed=v)
        assert refused(held(case), capsys, says="did not pass"), v
    edit(case.res / "smoke_record.json", passed=True)
    with pytest.raises(Reached):
        held(case)
