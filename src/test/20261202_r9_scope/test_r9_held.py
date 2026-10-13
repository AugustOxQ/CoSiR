"""C5 guard scenarios, one-statement mutations, and the held read's ordering contract.

All writable ledgers and read artefacts are in tmp_path. The real H5 ledger is read only.
"""
import ast
import importlib.util
import itertools
import json
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_r9_held as U

COUNT = itertools.count()


def mutant(tmp_path, guard=None, replacement=None):
    src = (HERE / "run_r9_held.py").read_text()
    if guard is not None:
        lines = src.splitlines(keepends=True)
        hits = [n for n in ast.walk(ast.parse(src)) if isinstance(n, (ast.Expr, ast.Assign))
                and f"# guard:{guard}" in lines[n.end_lineno - 1]]
        assert len(hits) == 1
        n = hits[0]
        first = lines[n.lineno - 1]
        lines[n.lineno - 1] = first[:len(first) - len(first.lstrip())] + "pass\n"
        for i in range(n.lineno, n.end_lineno):
            lines[i] = "\n"
        src = "".join(lines)
    else:
        old, new = replacement
        assert old in src
        src = src.replace(old, new)
    path = tmp_path / f"held_mut{next(COUNT)}.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    saved = list(sys.path)
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path[:] = saved
    return mod


def put(path, rec):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec))


def hashes(seed):
    return {p: f"{seed * 10 + i:064x}" for i, p in enumerate(U.R.PAIR_NAMES)}


def ledger(c, **change):
    cells = {"row": "H6", "backbone": U.LEDGER_NAMES[c.backbone], "episodes": "(pending)",
             "scripts": f"run_r9_held.py {'a' * 64}; {c.pins.name} {U.R.sha256_file(c.pins)}",
             "report": "(pending)", **change}
    c.ledger.write_text("| " + " | ".join([cells['row'], '2026-10-13', 'ArtELingo held',
                          cells['backbone'], 'AFF', cells['episodes'], cells['scripts'], cells['report']]) + " |\n")


def start(c, mod, count=1, fix=False, complete=True, copy=True):
    attempts = [{"attempt": i + 1, "flags": {"fix": 1 if fix else None},
                 "episodes_sha256": {str(s): hashes(s) for s in mod.R.HELD_SEEDS} if complete else {}}
                for i in range(count)]
    rec = {"rule_sha256": mod.R.RULE_SHA256, "attempts": attempts}
    put(c.res / mod.read_names("H6", "held").started, rec)
    if copy:
        put(c.folder / mod.read_names("H6", "held").started, rec)
    return rec


@pytest.fixture
def case(tmp_path, monkeypatch):
    c = NS(res=tmp_path / "results", folder=tmp_path / "folder", ledger=tmp_path / "ledger.md",
           pins=tmp_path / "pins_qwen3vl_emb_2b.json", backbone="qwen3vl_emb_2b")
    c.res.mkdir()
    c.folder.mkdir()
    put(c.pins, {"pinned": True})
    return c


def configure(mod, c, monkeypatch):
    # These scenarios isolate held guards; ticket 22 tests the new launch gate.
    import run_r9_preread
    monkeypatch.setattr(run_r9_preread, 'preread_guard', lambda *a: {})
    monkeypatch.setattr(mod, "runner_sha256", lambda: "a" * 64)
    monkeypatch.setattr(mod.R, "pins_path", lambda b: c.pins)
    monkeypatch.setattr(mod, "amsterdam_today", lambda: date(2026, 10, 18))
    ledger(c)


# guard -> (exception, regex). scenario() supplies the individual bad boundary.
REFUSALS = {
    "ledger_row": "row H6 is missing", "ledger_duplicate": "appears twice", "ledger_cells": "eight cells",
    "ledger_backbone": "another backbone",
    "ledger_runner_sha": "runner SHA", "ledger_pins_sha": "pins SHA", "ledger_report": "report cell",
    "ledger_not_run": "not run", "episodes_pending": "episode cell", "read_backbone": "requires",
    "time_box": "Amsterdam date", "verdict_exists": "verdict.*exists", "pass_exists": "pass.*exists",
    "started_exists": "read has started", "after_crash_needs_start": "no attempt",
    "copy_exists": "committed copy", "copy_matches": "missing or differs", "two_attempts": "2 attempts",
    "fix_needs_pass": "corrects.*pass", "fix_smoke": "own smoke", "fix_once": "fix 1 attempt",
    "fix_episodes": "nine episode", "reserve_started": "one reserve", "reserve_needs_verdict": "corrects a read",
    "reserve_hashes": "nine held", "smoke_missing": "smoke.*missing", "smoke_passed": "did not pass",
    "smoke_stale": "new smoke", "pins": "differs", "sigma": "sigma parts", "switched_off": "switched off",
    "value_sets_file": "value_sets.json", "value_sets_equal": "recomputed",
    "head_check": "selection posteriors", "coef_sha": "coefficient", "rerun_hashes": "recorded before",
    "h5_hashes": "H5", "smoke_verdict": "repeated smoke",
}
ASSERTIONS = {
    "post_rows": "finite on exactly", "episode_file": "other episodes", "episodes_reported": "not reported",
    "kept_json": "earlier attempt", "kept_npz": "earlier attempt", "allowed_keys": "only",
    "never_overwrite": "never overwritten", "started_first": "first write", "sensitivity_first": "before scores",
}
GUARDS = {**{g: (U.Refused, r) for g, r in REFUSALS.items()},
          **{g: (AssertionError, r) for g, r in ASSERTIONS.items()}}


def scenario(g, mod, c, mp):
    configure(mod, c, mp)
    if g == "ledger_row":
        ledger(c, row="H7")
        return mod.ledger_row(c.ledger, "H6")
    if g in {"ledger_duplicate", "ledger_cells"}:
        row = c.ledger.read_text()
        c.ledger.write_text(row + row if g == "ledger_duplicate" else row.rstrip() + " extra |\n")
        return mod.ledger_row(c.ledger, "H6")
    if g.startswith("ledger_"):
        change = {"ledger_backbone": {"backbone": "CLIP ViT-B/32"},
                  "ledger_runner_sha": {"scripts": f"run_r9_held.py {'b'*64}; {c.pins.name} {mod.R.sha256_file(c.pins)}"},
                  "ledger_pins_sha": {"scripts": f"run_r9_held.py {'a'*64}; {c.pins.name} {'b'*64}"},
                  "ledger_report": {"report": "GO"}, "ledger_not_run": {"episodes": "not run: switched off"}}[g]
        ledger(c, **change)
        return mod.ledger_guard(c.ledger, "H6", c.backbone, "a"*64, mod.R.sha256_file(c.pins))
    if g in {"read_backbone", "switched_off"}:
        return mod.read_guard("H6", "clip_b32" if g == "read_backbone" else c.backbone,
                              {"switched_off": g == "switched_off"}, smoke=g == "switched_off")
    if g in {"time_box", "verdict_exists", "pass_exists", "started_exists", "after_crash_needs_start",
             "copy_exists", "copy_matches", "two_attempts", "fix_needs_pass", "fix_smoke", "fix_once",
             "fix_episodes", "reserve_started", "reserve_needs_verdict", "reserve_hashes", "episodes_pending"}:
        variant, crash = "held", False
        n = mod.read_names("H6", variant)
        if g == "time_box":
            mp.setattr(mod, "amsterdam_today", lambda: date(2026, 10, 19))
        if g in {"verdict_exists", "pass_exists"}:
            (c.res / (n.verdict if g == "verdict_exists" else n.pass_)).touch()
        if g in {"started_exists", "copy_matches", "two_attempts"}:
            start(c, mod, count=2 if g == "two_attempts" else 1, copy=g != "started_exists")
            crash = g != "started_exists"
        if g == "after_crash_needs_start":
            crash = True
        if g == "copy_exists":
            (c.folder / n.started).touch()
        if g == "copy_matches":
            (c.folder / n.started).write_text("different")
        if g.startswith("fix_"):
            variant = "fix1"
            rec = start(c, mod, fix=g == "fix_once", complete=g != "fix_episodes")
            ledger(c, episodes="; ".join(h for a in rec['attempts'] for hs in a['episodes_sha256'].values()
                                         for h in hs.values()) or "(pending)")
            if g != "fix_needs_pass":
                (c.res / n.pass_).touch()
            if g != "fix_smoke":
                (c.res / "smoke_record_fix1.json").touch()
        if g.startswith("reserve_"):
            variant = "reserve"
            ledger(c, row="H6-R")
            if g != "reserve_hashes":
                start(c, mod)
            if g != "reserve_needs_verdict":
                (c.res / n.verdict).touch()
            if g == "reserve_started":
                put(c.res / mod.read_names("H6", variant).started,
                    {"rule_sha256": mod.R.RULE_SHA256, "attempts": [{"flags": {}, "episodes_sha256": {}}]})
        if g == "episodes_pending":
            ledger(c, episodes="already read")
        return mod.refuse_or_go(c.res, c.ledger, "H6", c.backbone, variant, crash, c.folder)
    if g in {"smoke_missing", "smoke_passed", "smoke_stale"}:
        path = c.res / "smoke_record.json"
        mp.setattr(mod.R, "r9_module_shas", lambda here=None: {"runner": "a"*64})
        rec = {"passed": g != "smoke_passed", "module_sha256": {"runner": "b"*64 if g == "smoke_stale" else "a"*64}}
        if g != "smoke_missing":
            put(path, rec)
        else:
            mp.setattr(mod, "_read_json", lambda path: rec)
            mp.setattr(mod.R, "sha256_file", lambda path: "a"*64)
        return mod.smoke_guard(c.res, "held")
    if g == "pins":
        mp.setattr(mod.PINS, "build_pins", lambda b: {"pinned": False})
        return mod.check_pins(c.backbone)
    if g == "sigma":
        path = c.res / f"sensitivity_seed42_{c.backbone}.json"
        put(path, {k: {"sigma_a2": -1, "sigma_eps2": 0} for k in mod.ST.CHECKS})
        return mod.sigma_guard(c.backbone, c.res)
    if g in {"value_sets_file", "value_sets_equal"}:
        path = c.res / "value_sets.json"
        put(path, {"wrong": True})
        mp.setattr(mod, "VALUE_SETS_PATH", path)
        if g == "value_sets_file":
            return mod.value_sets_file_guard()
        checked = {"file": str(path), "sha256": mod.R.sha256_file(path)}
        mp.setattr(mod.R, "value_sets_record", lambda v, d: {"right": True})
        return mod.value_sets_guard(NS(value_sets={}, data=None), checked)
    if g in {"head_check", "coef_sha"}:
        return mod.head_guard(NS(head_check={"passed": g != "head_check"}, coef_sha256={"head": "a"}),
                              {"coef_sha256": {"head": "b" if g == "coef_sha" else "a"}})
    if g in {"never_overwrite", "kept_json", "kept_npz"}:
        path = c.res / ("old.npz" if g == "kept_npz" else "old.json")
        if g == "kept_npz":
            np.savez(path, a=np.array([1]))
            return mod.keep_or_write_npz(path, {"a": np.array([2])})
        put(path, {"a": 1})
        if g == "kept_json":
            return mod.keep_or_write_json(path, {"a": 2}, ["a"])
        return mod.write_new(path, b"new")
    if g == "started_first":
        return mod.started_guard(c.res / "missing.json", c.folder / "missing.json")
    if g == "sensitivity_first":
        return mod.sensitivity_guard(c.res / "missing.json")
    if g in {"rerun_hashes", "h5_hashes", "episode_file"}:
        n = mod.read_names("H6", "held")
        start(c, mod, complete=False)
        want = {52: hashes(53)} if g == "rerun_hashes" else {}
        h5 = {52: hashes(53)} if g == "h5_hashes" else {52: hashes(52)}
        rec = mod.EpisodeRecorder(c.res / n.started, c.folder / n.started, 1, want, {}, c.res,
                                 read="H6", h5=h5)
        rec.expect(52)
        eps = NS(seed=52, sha=hashes(52))
        mp.setattr(mod.E, "save_episodes", lambda path, eps: None)
        if g == "episode_file":
            (c.res / mod.episodes_name("H6", 52)).touch()
            mp.setattr(mod.E, "load_episodes", lambda path: NS(seed=52, sha=hashes(53)))
        return rec(eps)
    if g == "post_rows":
        mp.setattr(mod.R, "N_ROWS", 4)
        b = NS(mode="held", seed=52, post={"A0": {"img": np.ones((4, 2))}})
        mp.setattr(mod, "seed_bundle", lambda *a, **kw: b)
        return mod.read_seed_bundle(NS(split=NS(held=np.array([0, 1]))), "held", 52, 1, c.backbone)
    if g == "smoke_verdict":
        out = c.res / "smoke"
        put(out / mod.read_names("H6", "smoke").verdict, {})
        mp.setattr(mod, "run_read", lambda *a, **kw: 0)
        # run_smoke translates Refused to exit 4; expose that refusal to the table.
        code = mod.run_smoke("clip_b32", "H6", results=c.res)
        if code == 4:
            raise mod.Refused("repeated smoke")
        return code
    if g in {"episodes_reported", "allowed_keys"}:
        spec, env, events, scores, passes = stub_read(mod, c, mp)
        if g == "episodes_reported":
            mp.setattr(mod, "read_seed_bundle", lambda env, mode, seed, *a, **kw:
                       NS(seed=seed, cl=np.array([0, 1, 0]), episodes_sha256={}))
            # The assertion after the callback is the boundary under test.
            mp.setattr(mod, "EpisodeRecorder", lambda *a, **kw: NS(hashes={s: hashes(s) for s in spec.seeds}, expect=lambda s: None))
        else:
            base = mod.S.score_seed
            mp.setattr(mod.S, "score_seed", lambda *a, **kw: {**base(*a, **kw), "B1": {}})
        return mod.run_read(spec, c.res, env=env)
    raise AssertionError(f"no scenario for {g}")


@pytest.mark.parametrize("guard", GUARDS)
def test_c5_guard(guard, case, monkeypatch):
    exc, regex = GUARDS[guard]
    with pytest.raises(exc, match=regex):
        scenario(guard, U, case, monkeypatch)


@pytest.mark.parametrize("guard", GUARDS)
def test_c5_mutation(guard, case, monkeypatch, tmp_path):
    mod = mutant(tmp_path, guard)
    scenario(guard, mod, case, monkeypatch)


def test_guard_table_covers_every_marker():
    markers = re.findall(r"# guard:(\w+)", (HERE / "run_r9_held.py").read_text())
    assert len(markers) == len(set(markers))
    from test_r9_held_a28 import GUARDS as A28_GUARDS
    from test_r9_preread import HELD_GUARDS as PREREAD_GUARDS
    assert set(markers) == set(GUARDS) | set(A28_GUARDS) | PREREAD_GUARDS


@pytest.mark.parametrize("read", ["H6", "H7"])
@pytest.mark.parametrize("variant", ["held", "fix1", "reserve", "smoke"])
def test_read_names_match_apply_rule(read, variant):
    n = U.read_names(read, variant)
    suffix = "_reserve" if variant == "reserve" else ""
    pass_suffix = "_fix1" if variant == "fix1" else suffix
    assert n.verdict == f"held_verdict_{read}{suffix}.json"
    assert n.started == f"held_started_{read}{suffix}.json"
    assert n.sensitivity == f"sensitivity_{read}{suffix}.json"
    assert n.pass_ == f"held_pass_{read}{pass_suffix}.json"
    assert n.arrays == f"held_arrays_{read}{pass_suffix}.npz"
    assert n.row == read + ("-R" if variant == "reserve" else "")
    assert U.episodes_name(read, 52) == f"held_episodes_{read}_seed52.npz"


def test_h5_hash_parser_on_real_ledger():
    got = U.h5_hashes(U.LEDGER)
    rec = json.loads((U.R.R6D / "results/held_started.json").read_text())
    assert got == {int(s): hs for s, hs in rec['attempts'][0]['episodes_sha256'].items()}
    assert set(got) == {52, 53, 54}
    assert len({h for hs in got.values() for h in hs.values()}) == 9


def stub_read(mod, c, mp, read="H6", variant="held", crash_at=None):
    """Stub expensive inputs and scoring, keeping the real read orchestration and atomic files."""
    events, scores, passes = [], [], []
    n = mod.read_names(read, variant)
    mp.setattr(mod.R, "pins_path", lambda b: c.pins)
    spec = NS(backbone=mod.READ_BACKBONES[read], read=read, mode="held", names=n, ctx_mode="held", seeds=(52, 53, 54),
              n_per_pair=1, out=c.res, copy_to=c.folder / n.started, overwrite=False,
              flags={"after_crash": False, "fix": None, "reserve": False, "smoke": False},
              attempts=[], want={}, record={}, h5={s: hashes(s) for s in (52, 53, 54)})
    env = NS(split=NS(held=np.array([0, 1]), selection=np.array([2]), groups=np.arange(4)),
             labels={}, value_sets={}, data=None, readers=object())
    def input_event(name, value):
        def call(*a, **kw):
            events.append(name)
            return value
        return call
    mp.setattr(mod.R, "assert_inputs", input_event("inputs", {}))
    mp.setattr(mod, "check_pins", input_event("pins", {}))
    mp.setattr(mod.E, "recorded_hashes", lambda: {})
    mp.setattr(mod.P, "load_picks", input_event("picks", {}))
    mp.setattr(mod.P, "frozen_lambdas", lambda b: {})
    mp.setattr(mod.P, "read_cells", input_event("cells", {"switched_off": False}))
    mp.setattr(mod.S, "cells_from_record", lambda rec: {"aff": {"fused": (1, 2), "cf": (3, 4)}})
    mp.setattr(mod.S, "taus_of", lambda b: (0., 1., 2., 3.))
    sigma = {k: {"sigma_a2": 1., "sigma_eps2": 1.} for k in mod.ST.CHECKS}
    mp.setattr(mod, "sigma_guard", lambda *a: {"sigma": sigma, "file": "sensitivity_seed42.json", "sha256": "a"*64})
    mp.setattr(mod, "value_sets_file_guard", lambda: {})
    mp.setattr(mod, "value_sets_guard", lambda *a: {})
    mp.setattr(mod, "head_guard", lambda *a: {})
    mp.setattr(mod, "refit_reference", lambda b: {})
    mp.setattr(mod.R, "r9_module_shas", lambda here=None: {"runner": "a"*64})
    mp.setattr(mod.R, "assert_held_eligible", lambda *a: events.append("eligible"))
    mp.setattr(mod, "git_provenance", lambda here=None: {})
    def bundle(env, mode, seed, n_per_pair, backbone, on_episodes=None):
        assert (c.res / n.started).is_file()
        assert spec.copy_to.read_bytes() == (c.res / n.started).read_bytes()
        events.append(f"episodes:{seed}")
        eps = NS(seed=seed, sha=hashes(seed))
        on_episodes(eps)
        if seed == crash_at:
            raise RuntimeError("synthetic crash")
        return NS(seed=seed, mode=mode, episodes_sha256=hashes(seed), cl=np.array([0, 1, 0]))
    mp.setattr(mod, "read_seed_bundle", bundle)
    def save_ep(path, eps):
        put(path, {"seed": eps.seed, "sha": eps.sha})
    mp.setattr(mod.E, "save_episodes", save_ep)
    mp.setattr(mod.E, "load_episodes", lambda path: NS(**json.loads(path.read_text())))
    def score(b, *a, **kw):
        scores.append(kw)
        assert (c.res / n.sensitivity).is_file()
        assert not (c.res / n.verdict).exists()
        events.append(f"score:{b.seed}")
        # Saving these arrays must never reduce them to a rate or gate share.
        return {**{name: {m: np.zeros(3, dtype=np.float64) for m in mod.METRICS} for name in mod.READ_SCORERS},
                "gates": {"aff": {"a": np.zeros((4, 3), dtype=bool), "b": np.ones((4, 3), dtype=bool)}},
                "reader": {"P": {c: np.zeros((3, 3)) for c in mod.CONDITIONS},
                           "m": {c: np.ones(3) for c in mod.CONDITIONS},
                           "pick": {c: np.zeros(3, dtype=np.int64) for c in mod.CONDITIONS}},
                "cl": b.cl, "pair_index": np.arange(3)}
    mp.setattr(mod.S, "score_seed", score)
    def passed(scored, mode, seeds, extra, **kw):
        passes.append(kw)
        events.append("pass")
        return {"mode": mode, "seeds": seeds, **extra}
    mp.setattr(mod.ST, "pass_record", passed)
    return spec, env, events, scores, passes


def assert_call_arguments(c, spec, scores, passes):
    assert len(scores) == 3 and len(passes) == 1
    for kw in scores:
        assert kw == {"include_pm": False, "secondary": False,
                      "cells": {"aff": {"fused": (1, 2), "cf": (3, 4)}}, "taus": (0., 1., 2., 3.),
                      "verdict": c.res / spec.names.verdict}
    assert passes == [{"secondary": False}]


@pytest.mark.parametrize("read", ["H6", "H7"])
@pytest.mark.parametrize("variant", ["held", "fix1", "reserve"])
def test_run_read_order_and_every_score_and_pass_argument(case, monkeypatch, capsys, read, variant):
    spec, env, events, scores, passes = stub_read(U, case, monkeypatch, read, variant)
    assert U.run_read(spec, case.res, env=env) == 0
    assert events[:4] == ["inputs", "pins", "picks", "cells"]
    assert events[4:] == ["episodes:52", "episodes:53", "episodes:54", "eligible", "score:52", "score:53", "score:54", "pass"]
    assert_call_arguments(case, spec, scores, passes)
    assert capsys.readouterr().out == (f"held pass written {spec.names.pass_} "
                                       f"{U.R.sha256_file(case.res / spec.names.pass_)}\n")
    assert not (case.res / spec.names.verdict).exists()
    assert (case.res / spec.names.pass_).is_file()
    with np.load(case.res / spec.names.arrays) as z:
        assert z['aff_gate__a'].shape == (4, 9)
        assert z['pick__a'].shape == (9,)
        assert z['margin__b'].shape == (9,)
        assert z['P__b'].shape == (9, 3)
        assert not any("share" in k or "rate" in k or k.startswith(("B1__", "r1_fused__")) for k in z.files)


@pytest.mark.parametrize("replacement", [
    ("include_pm=False, secondary=False,", "include_pm=True, secondary=False,"),
    ("include_pm=False, secondary=False,", "include_pm=False, secondary=True,"),
    ("cells=cells, taus=taus,", "cells=None, taus=taus,"),
    ("cells=cells, taus=taus,", "cells=cells, taus=None,"),
    ("verdict=out / n.verdict", "verdict=out / 'held_verdict_H6.json'"),
    ("extra, secondary=False)", "extra, secondary=True)"),
])
def test_call_argument_wiring_mutations_are_detected(case, monkeypatch, tmp_path, replacement):
    mod = mutant(tmp_path, replacement=replacement)
    # H7 reserve distinguishes the read ID and the reserve verdict from H6 and the base verdict.
    spec, env, events, scores, passes = stub_read(mod, case, monkeypatch, "H7", "reserve")
    assert mod.run_read(spec, case.res, env=env) == 0
    with pytest.raises(AssertionError):
        assert_call_arguments(case, spec, scores, passes)


def test_after_crash_rerun_reproduces_hashes(case, monkeypatch):
    configure(U, case, monkeypatch)
    monkeypatch.setattr(U.R, "N_PER_PAIR", 1)  # small bundles, with the production --after-crash orchestration
    spec, env, *_ = stub_read(U, case, monkeypatch, crash_at=53)
    with pytest.raises(RuntimeError, match="synthetic crash"):
        U.run_read(spec, case.res, env=env)
    before = U.attempt_hashes(U.read_attempts(case.res / spec.names.started))
    spec, env, *_ = stub_read(U, case, monkeypatch)
    monkeypatch.setattr(U, "smoke_guard", lambda *a: {})
    monkeypatch.setattr(U, "h5_hashes", lambda path: spec.h5)
    assert U.run_held(case.backbone, "H6", after_crash=True, results=case.res,
                      ledger=case.ledger, folder=case.folder, env=env) == 0
    attempts = U.read_attempts(case.res / spec.names.started)
    assert len(attempts) == 2
    assert attempts[1]['flags']['after_crash'] is True
    assert {s: attempts[1]['episodes_sha256'][str(s)] for s in before} == before
    assert U.attempt_hashes(attempts) == {s: hashes(s) for s in spec.seeds}
    assert (case.res / spec.names.started).read_bytes() == spec.copy_to.read_bytes()


BAD_INPUTS = {
    'ledger_runner': 'runner SHA', 'ledger_pins': 'pins SHA', 'ledger_backbone': 'another backbone',
    'ledger_report': 'report cell', 'h5': 'ledger H5', 'smoke': 'smoke.*missing',
    'read_backbone': 'requires', 'switched_off': 'switched off', 'sigma': 'sigma parts',
    'value_sets_file': 'frozen R6 value_sets', 'value_sets_equal': 'recomputed',
    'head': 'selection posteriors',
}


def append_h5(c, mod, wrong=False):
    records = {s: hashes(s) for s in mod.R.HELD_SEEDS}
    if wrong:
        records[52][mod.R.PAIR_NAMES[0]] = hashes(99)[mod.R.PAIR_NAMES[0]]
    cell = '; '.join(f"seed {s}: " + ', '.join(f"{p.replace('__', ' × ')} {h}" for p, h in hs.items())
                     for s, hs in records.items())
    with c.ledger.open('a') as f:
        f.write(f'| H5 | 2026-10-10 | ArtELingo held | CLIP ViT-B/32 | AFF | {cell} | old runner | GO |\n')


def held_bad_input(mod, c, mp, bad):
    """Exercise real guards at their production call sites, without loading data or fitting heads."""
    guards = {n: getattr(mod, n) for n in ('sigma_guard', 'value_sets_file_guard', 'value_sets_guard', 'head_guard')}
    clock = mod.amsterdam_today
    configure(mod, c, mp)
    spec, env, events, *_ = stub_read(mod, c, mp)
    for name, guard in guards.items():
        mp.setattr(mod, name, guard)
    mp.setattr(mod.R, 'N_PER_PAIR', 1)
    sigma = {k: {'sigma_a2': 1., 'sigma_eps2': 1.} for k in mod.ST.CHECKS}
    if bad == 'sigma':
        sigma['P1']['sigma_a2'] = -1.
    put(c.res / f'sensitivity_seed42_{c.backbone}.json', sigma)
    values = {'development': [1, 2]}
    vs_path = c.res / 'value_sets.json'
    put(vs_path, values)
    mp.setattr(mod, 'VALUE_SETS_PATH', vs_path)
    mp.setattr(mod, 'VALUE_SETS_SHA', mod.R.sha256_file(vs_path))
    mp.setattr(mod.R, 'value_sets_record', lambda *a: values if bad != 'value_sets_equal' else {})
    if bad == 'value_sets_file':
        # Equivalent parsed data with different bytes: isolate the file-pin call from the recomputation check.
        vs_path.write_text(json.dumps(values, indent=2))
    env.head_check = {'passed': bad != 'head'}
    env.coef_sha256 = {'head': 'a'}
    mp.setattr(mod, 'refit_reference', lambda b: {'coef_sha256': {'head': 'a'}})
    if bad.startswith('ledger_'):
        changes = {
            'ledger_runner': {'scripts': f"run_r9_held.py {'b'*64}; {c.pins.name} {mod.R.sha256_file(c.pins)}"},
            'ledger_pins': {'scripts': f"run_r9_held.py {'a'*64}; {c.pins.name} {'b'*64}"},
            'ledger_backbone': {'backbone': 'CLIP ViT-B/32'},
            'ledger_report': {'report': '(pending) extra'},
        }
        ledger(c, **changes[bad])
    append_h5(c, mod, wrong=bad == 'h5')
    if bad != 'smoke':
        put(c.res / 'smoke_record.json', {'passed': True, 'module_sha256': mod.R.r9_module_shas()})
    if bad == 'read_backbone':
        # Ledger and inputs name SigLIP correctly; H6 still requires Qwen.
        c.backbone = 'siglip2_so400m'
        ledger(c)
        append_h5(c, mod)
        put(c.res / f'sensitivity_seed42_{c.backbone}.json', sigma)
    if bad == 'switched_off':
        mp.setattr(mod.P, 'read_cells', lambda path: {'switched_off': True})
    if bad == 'time_zone':
        mp.setattr(mod, 'amsterdam_today', clock)  # configure normally patches this; test the real clock here
        mp.setattr(mod, 'datetime', BoundaryDatetime)
    code = mod.run_held(c.backbone, 'H6', results=c.res, ledger=c.ledger, folder=c.folder, env=env)
    assert code == 4, f'{bad}: invalid input reached a completed read'
    assert not (c.res / spec.names.pass_).exists()
    if bad not in ('h5', 'time_zone'):
        assert not (c.res / spec.names.started).exists()
    return code


@pytest.mark.parametrize('bad', BAD_INPUTS)
def test_run_held_bad_input_reaches_its_guard(case, monkeypatch, capsys, bad):
    held_bad_input(U, case, monkeypatch, bad)
    assert re.search(BAD_INPUTS[bad], capsys.readouterr().out)


CALL_SITE_MUTATIONS = [
    ('ledger_runner', ('ledger_guard(ledger, n.row, backbone, runner_sha256())', 'ledger_row(ledger, n.row)')),
    ('h5', ('h5=h5_hashes(ledger or LEDGER)', 'h5=None')),
    ('h5', ('read=spec.read, h5=spec.h5', 'read=spec.read, h5=None')),
    ('smoke', ('smoke = smoke_guard(results, variant, here)', 'smoke = {}')),
    ('read_backbone', ("    read_guard(spec.read, backbone, cell_record, smoke=spec.mode == 'smoke')", '    pass')),
    ('sigma', ('sigma = sigma_guard(backbone, results)',
               "sigma = {'sigma': {c: {'sigma_a2': 1., 'sigma_eps2': 1.} for c in ST.CHECKS}, 'file': 'stub', 'sha256': 'a'*64}")),
    ('value_sets_file', ('vs_file = value_sets_file_guard()',
                        "vs_file = {'file': str(VALUE_SETS_PATH), 'sha256': R.sha256_file(VALUE_SETS_PATH)}")),
    ('value_sets_equal', ('values = value_sets_guard(env, vs_file)', 'values = {}')),
    ('head', ('coef = head_guard(env, refit_reference(backbone))', 'coef = {}')),
]


@pytest.mark.parametrize('bad,replacement', CALL_SITE_MUTATIONS)
def test_guard_call_site_deletion_is_detected(case, monkeypatch, tmp_path, bad, replacement):
    mod = mutant(tmp_path, replacement=replacement)
    # When the call is deleted its bad input completes a read: the production regression assertion fails.
    with pytest.raises(AssertionError, match='invalid input reached a completed read'):
        held_bad_input(mod, case, monkeypatch, bad)


def recorder_collision(mod, c, mp):
    configure(mod, c, mp)
    start(c, mod, complete=False)
    n = mod.read_names('H6')
    rec = mod.EpisodeRecorder(c.res / n.started, c.folder / n.started, 1, {}, {42: hashes(52)},
                             c.res, read='H6', h5={52: hashes(52)})
    rec.expect(52)
    mp.setattr(mod.E, 'save_episodes', lambda *a: None)
    try:
        rec(NS(seed=52, sha=hashes(52)))
    except AssertionError as e:
        assert 'episode hashes equal recorded ones' in str(e)
        return
    pytest.fail('recorded episode hash was accepted')


def test_recorder_refuses_an_already_recorded_hash(case, monkeypatch):
    recorder_collision(U, case, monkeypatch)


def test_recorder_distinctness_call_deletion_is_detected(case, monkeypatch, tmp_path):
    mod = mutant(tmp_path, replacement=('        E.assert_distinct(dict(self.hashes), recorded=self.recorded)',
                                        '        pass'))
    with pytest.raises(pytest.fail.Exception, match='recorded episode hash was accepted'):
        recorder_collision(mod, case, monkeypatch)


def labelled_ledger(mod, c, mp, label, correct_last):
    configure(mod, c, mp)
    runner, pins = 'a' * 64, mod.R.sha256_file(c.pins)
    entries = [f'run_r9_held.py {runner}', f'{c.pins.name} {pins}']
    expected = runner if label == 'runner' else pins
    name = 'run_r9_held.py' if label == 'runner' else c.pins.name
    ordered = ['b' * 64, expected] if correct_last else [expected, 'b' * 64]
    entries[label == 'pins'] = '; '.join(f'{name} {sha}' for sha in ordered)
    ledger(c, scripts='; '.join(entries))
    return mod.ledger_guard(c.ledger, 'H6', c.backbone, runner, pins)


@pytest.mark.parametrize('label', ['runner', 'pins'])
@pytest.mark.parametrize('correct_last', [True, False])
def test_ledger_uses_last_entry_for_each_label(case, monkeypatch, label, correct_last):
    if correct_last:
        assert labelled_ledger(U, case, monkeypatch, label, True)['row'] == 'H6'
    else:
        with pytest.raises(U.Refused, match=f'{label} SHA'):
            labelled_ledger(U, case, monkeypatch, label, False)


@pytest.mark.parametrize('label', ['runner', 'pins'])
def test_first_label_entry_mutation_is_detected(case, monkeypatch, tmp_path, label):
    mod = mutant(tmp_path, replacement=(f'{label}[-1]', f'{label}[0]'))
    with pytest.raises(mod.Refused, match=f'{label} SHA'):
        labelled_ledger(mod, case, monkeypatch, label, True)


class BoundaryDatetime(datetime):
    """A UTC date still on the deadline when Amsterdam has passed it, without patching amsterdam_today."""
    @classmethod
    def now(cls, tz=None):
        instant = datetime(2026, 10, 18, 22, 30, tzinfo=timezone.utc)
        return instant.astimezone(tz) if tz is not None else instant.replace(tzinfo=None)

    @classmethod
    def utcnow(cls):
        return datetime(2026, 10, 18, 22, 30)


def test_amsterdam_clock_refuses_first_read_at_utc_boundary(case, monkeypatch, capsys):
    held_bad_input(U, case, monkeypatch, 'time_zone')
    assert U.amsterdam_today() == date(2026, 10, 19)
    assert 'Amsterdam date 2026-10-19' in capsys.readouterr().out


def test_utc_clock_mutation_is_detected(case, monkeypatch, tmp_path):
    mod = mutant(tmp_path, replacement=('datetime.now(ZoneInfo("Europe/Amsterdam")).date()',
                                        'datetime.utcnow().date()'))
    with pytest.raises(AssertionError, match='invalid input reached a completed read'):
        held_bad_input(mod, case, monkeypatch, 'time_zone')


@pytest.mark.parametrize('report', ['(pending) extra', '(pending?)', '(pend', '(pending)\u00a0x'])
def test_report_cell_requires_exact_pending(case, monkeypatch, report):
    configure(U, case, monkeypatch)
    ledger(case, report=report)
    with pytest.raises(U.Refused, match='report cell'):
        U.ledger_guard(case.ledger, 'H6', case.backbone, 'a'*64, U.R.sha256_file(case.pins))


def prepare_variant(mod, c, mp, variant, today=date(2026, 10, 18)):
    configure(mod, c, mp)
    mp.setattr(mod, 'amsterdam_today', lambda: today)
    rec = start(c, mod)
    base = mod.read_names('H6')
    if variant == 'reserve':
        ledger(c, row='H6-R')
        put(c.res / base.verdict, {})
    elif variant == 'fix1':
        put(c.res / base.pass_, {})
        put(c.res / 'smoke_record_fix1.json', {})
        ledger(c, episodes='; '.join(h for hs in rec['attempts'][0]['episodes_sha256'].values() for h in hs.values()))
    return rec


def fix_missing_hash(mod, c, mp):
    rec = prepare_variant(mod, c, mp, 'fix1')
    values = [h for hs in rec['attempts'][0]['episodes_sha256'].values() for h in hs.values()]
    ledger(c, episodes='; '.join(values[:-1]))
    return mod.refuse_or_go(c.res, c.ledger, 'H6', c.backbone, 'fix1', folder=c.folder)


def test_fix_requires_exactly_all_nine_ledger_hashes(case, monkeypatch):
    with pytest.raises(U.Refused, match='nine episode hashes'):
        fix_missing_hash(U, case, monkeypatch)


def test_fix_hash_subset_mutation_is_detected(case, monkeypatch, tmp_path):
    mod = mutant(tmp_path, replacement=("set(row['episodes']) ==", "set(row['episodes']) <="))
    assert fix_missing_hash(mod, case, monkeypatch).names.kind == 'fix1'


def bad_reserve_first(mod, c, mp, bad):
    prepare_variant(mod, c, mp, 'reserve')
    if bad == 'copy':
        put(c.folder / mod.read_names('H6', 'reserve').started, {})
    else:
        ledger(c, row='H6-R', episodes='already read')
    return mod.refuse_or_go(c.res, c.ledger, 'H6', c.backbone, 'reserve', folder=c.folder)


@pytest.mark.parametrize('bad,regex', [('copy', 'committed copy'), ('episodes', 'episode cell')])
def test_reserve_enforces_first_read_checks(case, monkeypatch, bad, regex):
    with pytest.raises(U.Refused, match=regex):
        bad_reserve_first(U, case, monkeypatch, bad)


@pytest.mark.parametrize('bad', ['copy', 'episodes'])
def test_reserve_not_first_mutation_is_detected(case, monkeypatch, tmp_path, bad):
    mod = mutant(tmp_path, replacement=("first = variant == \"reserve\" or (variant == \"held\" and not after_crash)",
                                        'first = variant == "held" and not after_crash'))
    assert bad_reserve_first(mod, case, monkeypatch, bad).names.kind == 'reserve'


@pytest.mark.parametrize('variant', ['reserve', 'fix1', 'after_crash'])
@pytest.mark.parametrize('today', [date(2026, 10, 18), date(2026, 10, 19)])
def test_deadline_boxes_reserve_but_continues_fix_and_crash(case, monkeypatch, variant, today):
    kind = 'held' if variant == 'after_crash' else variant
    prepare_variant(U, case, monkeypatch, kind, today)
    kwargs = dict(folder=case.folder, after_crash=variant == 'after_crash')
    if variant == 'reserve' and today.day == 19:
        with pytest.raises(U.Refused, match='Amsterdam date'):
            U.refuse_or_go(case.res, case.ledger, 'H6', case.backbone, kind, **kwargs)
    else:
        assert U.refuse_or_go(case.res, case.ledger, 'H6', case.backbone, kind, **kwargs).names.kind == kind


def shared_sensitivity_case(mod, case, monkeypatch):
    spec, env, *_ = stub_read(mod, case, monkeypatch)
    sigma = {c: {'sigma_a2': float(i + 1), 'sigma_eps2': float(i + 2)} for i, c in enumerate(mod.ST.CHECKS)}
    source = {'file': 'synthetic_sigma.json', 'sha256': 'b' * 64}
    monkeypatch.setattr(mod, 'sigma_guard', lambda *a: {'sigma': sigma, **source})
    calls = []
    shared = mod.ST.sensitivity_record
    def sensitivity(sig, M_p, N, design):
        calls.append((sig, M_p.copy(), N, design))
        return shared(sig, M_p, N, design)
    monkeypatch.setattr(mod.ST, 'sensitivity_record', sensitivity)
    assert mod.run_read(spec, case.res, env=env) == 0
    assert len(calls) == 1, 'run_read must use the shared sensitivity writer'
    sig, M_p, N, design = calls[0]
    assert sig == sigma and np.array_equal(M_p, np.array([6, 3])) and N == 9
    got = json.loads((case.res / spec.names.sensitivity).read_text())
    # The removed R6 writer took these same synthetic pooled painting counts and sigma parts.
    # Independent reference expression fixes its exact arithmetic and both multipliers.
    for c in mod.ST.CHECKS:
        SE = float(np.sqrt((sigma[c]['sigma_a2'] * float(np.sum(M_p ** 2)) + sigma[c]['sigma_eps2'] * N) / float(N) ** 2))
        old = {'SE': SE, 'x': 3.532 * SE, 'x95': 2.80 * SE}
        assert {k: got[c][k] for k in old} == old
        assert got[c]['quantity'] == mod.ST.QUANTITIES[c][2]
        assert {k: got[c][k] for k in mod.SIGMA_KEYS} == sigma[c]
    assert got['N'] == N and got['n_paintings'] == 2 and got['seeds'] == list(spec.seeds)
    assert got['episodes_sha256'] == {str(s): hashes(s) for s in spec.seeds}
    assert got['sigma_source'] == source and got['module_sha256'] == mod.R.r9_module_shas()
    assert isinstance(got['time'], str)
    assert design == 'seven'


def test_run_read_uses_shared_sensitivity_and_preserves_old_values(case, monkeypatch):
    shared_sensitivity_case(U, case, monkeypatch)


def test_wrong_sensitivity_family_mutation_is_detected(case, monkeypatch, tmp_path):
    mod = mutant(tmp_path, replacement=("ST.sensitivity_record(sigma['sigma'], m_p, n_episodes, 'seven')",
                                        "ST.sensitivity_record(sigma['sigma'], m_p, n_episodes, 'two_level')"))
    with pytest.raises(AssertionError):
        shared_sensitivity_case(mod, case, monkeypatch)


def test_loose_report_cell_mutation_is_detected(case, monkeypatch, tmp_path):
    mod = mutant(tmp_path, replacement=("row['report'] == PENDING", "row['report'].startswith('(pend')"))
    with pytest.raises(AssertionError, match='invalid input reached a completed read'):
        held_bad_input(mod, case, monkeypatch, 'ledger_report')


def pass_write_collision(mod, c, mp):
    spec, env, *_ = stub_read(mod, c, mp)
    path = c.res / spec.names.pass_
    before = b'existing pass: never overwrite'
    passed = mod.ST.pass_record
    def competing_write(*a, **kw):
        path.write_bytes(before)  # file appears after the preflight, just before the final write
        return passed(*a, **kw)
    mp.setattr(mod.ST, 'pass_record', competing_write)
    try:
        mod.run_read(spec, c.res, env=env)
    except AssertionError as e:
        assert 'never overwritten' in str(e)
        assert path.read_bytes() == before
        return
    pytest.fail('existing pass was overwritten')


def test_run_read_pass_write_never_overwrites(case, monkeypatch):
    pass_write_collision(U, case, monkeypatch)


def test_pass_write_guard_call_deletion_is_detected(case, monkeypatch, tmp_path):
    mod = mutant(tmp_path, replacement=('write_new(path, _json_bytes(pr), spec.overwrite)',
                                        '_write_bytes(path, _json_bytes(pr))'))
    with pytest.raises(pytest.fail.Exception, match='existing pass was overwritten'):
        pass_write_collision(mod, case, monkeypatch)


def test_reserve_deadline_scope_mutation_is_detected(case, monkeypatch, tmp_path):
    mod = mutant(tmp_path, replacement=('    if first:\n        today = amsterdam_today()',
                                        '    if variant == "held" and not after_crash:\n        today = amsterdam_today()'))
    prepare_variant(mod, case, monkeypatch, 'reserve', date(2026, 10, 19))
    assert mod.refuse_or_go(case.res, case.ledger, 'H6', case.backbone, 'reserve', folder=case.folder).names.kind == 'reserve'


def short_ledger(mod, c, mp):
    configure(mod, c, mp)
    cells = c.ledger.read_text().strip().strip('|').split('|')
    c.ledger.write_text('|' + '|'.join(cells[:-1]) + '|\n')
    return mod.ledger_row(c.ledger, 'H6')


def test_seven_cell_ledger_is_a_clean_refusal(case, monkeypatch):
    with pytest.raises(U.Refused, match='eight cells'):
        short_ledger(U, case, monkeypatch)


def test_seven_cell_ledger_mutant_is_detected(case, monkeypatch, tmp_path):
    mod = mutant(tmp_path, replacement=('len(cells) == len(LEDGER_CELLS)', 'len(cells) <= len(LEDGER_CELLS)'))
    with pytest.raises(IndexError):
        short_ledger(mod, case, monkeypatch)


def h5_collision(mod, c, mp):
    configure(mod, c, mp)
    append_h5(c, mod)
    text = c.ledger.read_text().replace(hashes(53)[mod.R.PAIR_NAMES[0]], hashes(52)[mod.R.PAIR_NAMES[0]])
    c.ledger.write_text(text)
    return mod.h5_hashes(c.ledger)


def test_h5_parser_calls_distinctness(case, monkeypatch):
    with pytest.raises(AssertionError, match='repeats the episode hash'):
        h5_collision(U, case, monkeypatch)


def test_h5_distinctness_call_mutant(case, monkeypatch, tmp_path):
    mod = mutant(tmp_path, replacement=('    E.assert_distinct(out, recorded={})', '    pass'))
    assert len(h5_collision(mod, case, monkeypatch)) == 3


@pytest.mark.parametrize('name', ['started_guard', 'sensitivity_guard'])
def test_read_calls_order_guard(case, monkeypatch, name):
    order_guard_site(U, case, monkeypatch, name)


def order_guard_site(mod, c, mp, name):
    spec, env, *_ = stub_read(mod, c, mp)
    calls = []
    real = getattr(mod, name)
    def spy(*a):
        calls.append(name)
        return real(*a)
    mp.setattr(mod, name, spy)
    assert mod.run_read(spec, c.res, env=env) == 0
    assert calls == [name], f'{name} call missing'


@pytest.mark.parametrize('name,line', [
    ('started_guard', '    started_guard(started, spec.copy_to)'),
    ('sensitivity_guard', '    sensitivity_guard(out / n.sensitivity)'),
])
def test_order_guard_call_mutant(case, monkeypatch, tmp_path, name, line):
    mod = mutant(tmp_path, replacement=(line, '    pass'))
    with pytest.raises(AssertionError, match='call missing'):
        order_guard_site(mod, case, monkeypatch, name)


def changed_sensitivity_rerun(mod, c, mp):
    spec, env, *_ = stub_read(mod, c, mp)
    score = mod.S.score_seed
    def crash(*a, **kw):
        raise RuntimeError('after sensitivity')
    mp.setattr(mod.S, 'score_seed', crash)
    with pytest.raises(RuntimeError, match='after sensitivity'):
        mod.run_read(spec, c.res, env=env)
    path = c.res / spec.names.sensitivity
    rec = json.loads(path.read_text())
    rec['P4']['SE'] += 1.
    put(path, rec)
    spec.attempts = mod.read_attempts(c.res / spec.names.started)
    spec.want = mod.attempt_hashes(spec.attempts)
    mp.setattr(mod.S, 'score_seed', score)
    return mod.run_read(spec, c.res, env=env)


def test_changed_sensitivity_stops_after_crash_rerun(case, monkeypatch):
    with pytest.raises(AssertionError, match='differs.*P4.*user decides'):
        changed_sensitivity_rerun(U, case, monkeypatch)


def test_sensitivity_rerun_comparison_mutant(case, monkeypatch, tmp_path):
    mod = mutant(tmp_path, replacement=("same + ['episodes_sha256', 'sigma_source']", "['N']"))
    assert changed_sensitivity_rerun(mod, case, monkeypatch) == 0
