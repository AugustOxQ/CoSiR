"""Final review A2, item 3: run_r9_held's real mode end to end on synthetic scores, with the real r9_stats.

Run: pytest -q final_review/fr_a2_e2e.py (from the round folder). Everything is written under tmp_path; the
ledgers are test_r9_held's tmp_path ledgers. score_seed is replaced by a synthetic per-anchor scorer (no held row,
no feature, no head); pass_record, sensitivity_record, bootstrap_draws and point_ci are the real ones.
Then the pass file is recomputed from the saved held_arrays npz by my own reassembly and compared.
Writes fr_a2_e2e.json beside this file.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import test_r9_held as TH  # noqa: E402
from test_r9_held import case, configure, stub_read, put, hashes  # noqa: E402,F401
import r9_stats as ST  # noqa: E402

NPP = 128
OUT = {}


def synth(mod, mp, spec, rng_seed=0):
    """Synthetic bundles (cl of 3·NPP paintings) and scores; the real ST functions."""
    mp.setattr(mod.R, "N_PER_PAIR", NPP)
    mp.setattr(ST.R6C, "N_PER_PAIR", NPP)
    spec.n_per_pair = NPP
    rng = np.random.default_rng(rng_seed)
    cls = {s: rng.integers(0, 60, 3 * NPP).astype(np.int64) for s in spec.seeds}

    def bundle(env, mode, seed, n_per_pair, backbone, on_episodes=None):
        on_episodes(NS(seed=seed, sha=hashes(seed)))
        return NS(seed=seed, mode=mode, episodes_sha256=hashes(seed), cl=cls[seed])
    mp.setattr(mod, "read_seed_bundle", bundle)

    def score(b, *a, **kw):
        assert kw["include_pm"] is False and kw["secondary"] is False
        n = len(b.cl)
        r = np.random.default_rng(1000 + b.seed)
        out = {}
        for name in mod.READ_SCORERS:
            r1 = r.integers(0, 2, n).astype(np.float64)
            gain = (r.integers(-1, 2, n).astype(np.float64) if name != ST.CF else np.zeros(n))
            out[name] = {m: (r1 if m == "r1" else gain if m == "gain" else r.random(n)) for m in mod.METRICS}
        out["gates"] = {"aff": {c: r.random((4, n)) < .5 for c in mod.CONDITIONS}}
        out["reader"] = {"P": {c: r.random((n, 3)) for c in mod.CONDITIONS},
                         "m": {c: r.random(n) for c in mod.CONDITIONS},
                         "pick": {c: r.integers(0, 3, n) for c in mod.CONDITIONS}}
        out["cl"], out["pair_index"] = b.cl, np.repeat(np.arange(3), NPP)
        return out
    mp.setattr(mod.S, "score_seed", score)
    return cls


def recompute(mod, res, names, seeds):
    """My reassembly of per-seed score dicts from the npz, then r9_stats.pass_record on them."""
    rec = json.loads((res / names.pass_).read_text())
    with np.load(res / names.arrays) as z:
        arr = {k: z[k] for k in z.files}
    si = arr["seed_index"]
    per_seed = []
    for i, _ in enumerate(seeds):
        m = si == i
        d = {name: {met: arr[f"{name}__{met}"][m] for met in mod.METRICS} for name in mod.READ_SCORERS}
        d["cl"], d["pair_index"] = arr["cl"][m], arr["pair_index"][m]
        per_seed.append(d)
    extra = {"episodes_sha256": {s: hashes(s) for s in seeds}, "runner_sha256": "a" * 64,
             "module_sha256": {"runner": "a" * 64}}
    return rec, per_seed, extra


def compare(rec, mine):
    keys = ("checks", "N", "n_paintings", "design", "seeds", "mode", "families", "level1", "level2")
    diff = [k for k in keys if (k in rec or k in mine) and json.dumps(rec.get(k), sort_keys=True, default=float)
            != json.dumps(json.loads(json.dumps(mine.get(k), default=float)), sort_keys=True, default=float)]
    return diff


def test_h6_real_mode_pass_equals_r9_stats_on_saved_arrays(case, monkeypatch):
    mod = TH.U
    real_pass = ST.pass_record
    spec, env, events, scores, passes = stub_read(mod, case, monkeypatch, "H6", "held")
    mp = monkeypatch
    synth(mod, mp, spec)
    mp.setattr(ST, "pass_record", real_pass)          # undo stub_read's pass_record stub
    assert mod.ST is ST
    ST2 = ST
    assert mod.run_read(spec, case.res, env=env) == 0
    rec, per_seed, extra = recompute(mod, case.res, spec.names, spec.seeds)
    mine = ST2.pass_record(per_seed, "held", list(spec.seeds), extra, secondary=False)
    diff = compare(rec, mine)
    sens = json.loads((case.res / spec.names.sensitivity).read_text())
    _, m_p = np.unique(np.concatenate([d["cl"] for d in per_seed]), return_counts=True)
    OUT["H6"] = {"differing_keys": diff, "checks": sorted(rec["checks"]),
                 "n": {c: rec["checks"][c]["n"] for c in rec["checks"]},
                 "sens_N": sens["N"], "sens_n_paintings": sens["n_paintings"], "my_n_paintings": int(m_p.size),
                 "pass_has_secondary": "secondary" in rec}
    assert diff == [] and sens["n_paintings"] == m_p.size and "secondary" not in rec
    # guards around a finished read: a second run, the order guard, the verdict-exists refusal
    (case.res / spec.names.verdict).write_text("{}")
    OUT["H6_after"] = {}
    for label, kw in (("rerun_without_flag", {}), ("after_crash_with_verdict", {"after_crash": True})):
        code = mod.run_held(case.backbone, "H6", results=case.res, ledger=case.ledger, folder=case.folder, env=env,
                            **kw)
        OUT["H6_after"][label] = code
        assert code == mod.EXIT_REFUSE




def test_h8_real_mode_pass_equals_r9_stats_on_saved_arrays(case, monkeypatch):
    import test_r9_held_a28 as TA
    mod = TH.U
    real = {k: getattr(ST, k) for k in ("bootstrap_draws",)}
    real_pc = ST.C.point_ci
    spec, env, events, scores = TA.stub_a28(mod, case, monkeypatch)[:4]
    monkeypatch.setattr(ST, "bootstrap_draws", real["bootstrap_draws"])
    monkeypatch.setattr(ST.C, "point_ci", real_pc)

    def score(b, *a, **kw):
        assert kw["include_pm"] is False and kw["secondary"] is False
        n = len(b.cl)
        r = np.random.default_rng(2000 + b.seed)
        out = {}
        for name in mod.READ_SCORERS:
            r1 = r.integers(0, 2, n).astype(np.float64)
            gain = (r.integers(-1, 2, n).astype(np.float64) if name != ST.CF else np.zeros(n))
            out[name] = {m: (r1 if m == "r1" else gain if m == "gain" else r.random(n)) for m in mod.METRICS}
        out["gates"] = {"aff": {c: r.random((4, n)) < .5 for c in mod.CONDITIONS}}
        out["reader"] = {"P": {c: r.random((n, 3)) for c in mod.CONDITIONS},
                         "m": {c: r.random(n) for c in mod.CONDITIONS},
                         "pick": {c: r.integers(0, 3, n) for c in mod.CONDITIONS}}
        out["cl"], out["pair_index"] = np.asarray(b.cl), np.arange(n) // max(1, n // 3)
        return out
    monkeypatch.setattr(mod.S, "score_seed", score)
    assert mod.run_read(spec, case.res, env=env) == 0
    rec, per_seed, extra = recompute(mod, case.res, spec.names, spec.seeds)
    mine = ST.pass_record_a28(per_seed, "a28", list(spec.seeds), extra, set_aside=rec["set_aside"],
                              flagged_languages=rec["flagged_languages"], near_duplicates=rec["near_duplicates"])
    diff = compare(rec, mine)
    OUT["H8"] = {"differing_keys": diff, "n_anchors": int(sum(len(d["cl"]) for d in per_seed)),
                 "levels": sorted(k for k in rec if "level" in k), "design": rec.get("design"),
                 "mode": rec.get("mode")}
    assert diff == []


def teardown_module(module):
    (HERE / "fr_a2_e2e.json").write_text(json.dumps(OUT, indent=2, default=str) + "\n")
