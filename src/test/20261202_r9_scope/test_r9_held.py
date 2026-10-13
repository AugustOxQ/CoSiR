"""C5 guard scenarios, one-statement mutations, and the held read's ordering contract.

All writable ledgers and read artefacts are in tmp_path. The real H5 ledger is read only.
"""
import ast
import importlib.util
import itertools
import json
import re
import sys
from datetime import date
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


def start(c, mod, count=1, fix=False, complete=True):
    attempts = [{"attempt": i + 1, "flags": {"fix": 1 if fix else None},
                 "episodes_sha256": {str(s): hashes(s) for s in mod.R.HELD_SEEDS} if complete else {}}
                for i in range(count)]
    rec = {"rule_sha256": mod.R.RULE_SHA256, "attempts": attempts}
    put(c.res / mod.read_names("H6", "held").started, rec)
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
    monkeypatch.setattr(mod, "runner_sha256", lambda: "a" * 64)
    monkeypatch.setattr(mod.R, "pins_path", lambda b: c.pins)
    monkeypatch.setattr(mod, "amsterdam_today", lambda: date(2026, 10, 18))
    ledger(c)


# guard -> (exception, regex). scenario() supplies the individual bad boundary.
REFUSALS = {
    "ledger_row": "row H6 is missing", "ledger_backbone": "another backbone",
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
            start(c, mod, count=2 if g == "two_attempts" else 1)
            crash = g != "started_exists"
            if g == "started_exists":
                (c.folder / n.started).unlink()  # isolate the results-file refusal from the copy refusal
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
            mp.setattr(mod, "sensitivity_held", lambda *a: {"P1": {}})
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
    assert set(markers) == set(GUARDS)


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
