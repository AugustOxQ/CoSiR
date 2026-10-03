"""H1 nested-score pilot (PREREGISTRATION.md §5 and §6): seed-42 selection episodes only, CPU.

Real run:  python score_pilot.py          -> results/pilot_seed42.json (refuses to overwrite it)
Smoke run: python score_pilot.py --smoke  -> results/smoke/ (A1's smoke checkpoint stands in for every E3 run)
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from common import ROOT, folders, rg  # noqa: E402

from src.eval.aspect_metrics import METRICS, compare, per_anchor  # noqa: E402
from src.eval.aspect_nested import (ceiling_threshold, control_scores, control_sums, crossfit_nested,  # noqa: E402
                                    margin_reading, nested_cells, nested_scores, predicted_power, se_from_ci)
from src.eval.aspect_scorers import EvalInputs, agreement_term  # noqa: E402

MODELS = ("A3", "A1", "A2", "A4", "A5", "A6", "C0", "SE")
SEED = 42


def model_codes(ctx, name, smoke):
    if name in ("C0", "SE"):
        path = rg.E1 / f"codes_{name}.npz"
        z = np.load(path)
        return ctx.masked(z["img"]), ctx.masked(z["txt"]), {"codes": str(path.relative_to(ROOT)),
                                                            "sha256": rg.sha_file(path)}
    ckpt = rg.checkpoint_path(name, SEED, smoke)          # smoke: A1's smoke checkpoint stands in
    sha = rg.sha_file(ckpt)
    if name == "A3" and not smoke:
        pick = json.loads((rg.HERE / "results" / "picked.json").read_text())
        assert pick["run"] == "A3" and sha == pick["checkpoint_sha256"], "A3 checkpoint != E3's pick"
    ic, tc = ctx.encode(ckpt)
    return ic, tc, {"checkpoint": str(ckpt.relative_to(ROOT)), "sha256": sha}


def zeros_like(pa):
    return {m: np.zeros_like(np.asarray(pa[m], dtype=np.float64)) for m in pa}


def either_point(arr, cl):
    """Pooled mean (pp) and clustered 95% interval of one per-anchor array."""
    v = compare({"e": arr}, {"e": np.zeros_like(arr)}, cl, "e")
    return {"point": v["point"], "ci95": v["ci95"]}


def paired(pa, pb, cl):
    return {m: compare(pa, pb, cl, m) for m in ("r1", "gain")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    smoke = ap.parse_args().smoke
    res = folders(smoke)["res"]
    out_json = res / "pilot_seed42.json"
    if out_json.exists() and not smoke:
        raise SystemExit(f"{out_json} exists; refusing to overwrite a real pilot result")
    res.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    ctx = rg.EvalContext(SEED, smoke)
    cl = ctx.anchor_group
    t9 = np.load((rg.E1 / "smoke" if smoke else rg.E1) / "per_anchor_seed42.npz")
    missing = [f"rca__{m}" for m in METRICS if f"rca__{m}" not in t9.files]
    assert not missing, f"RCA per-anchor keys missing: {missing}"
    rca = {m: t9[f"rca__{m}"] for m in METRICS}
    cos_pa = per_anchor(ctx.cos)
    assert all(np.array_equal(cos_pa[m], t9[f"cosine__{m}"]) for m in METRICS), "episodes misaligned with E1"

    result = {"provenance": {"episodes_sha256": ctx.shas, "n_episodes": int(ctx.n),
                             "n_clusters": int(len(np.unique(cl))), "smoke": smoke,
                             "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    arrays, pas = {}, {}
    for name in MODELS:
        ic, tc, prov = model_codes(ctx, name, smoke)
        inp = EvalInputs(ctx.img, ctx.txt, ic, tc)
        tu = agreement_term(inp, ctx.pooled, uniform=True)
        ta = agreement_term(inp, ctx.pooled)
        nested, control, picks = crossfit_nested(ctx.cos, tu, ta, ctx.parity)
        pn, pc = per_anchor(nested), per_anchor(control)
        pas[name] = pn
        for kind, pa in (("nested", pn), ("control", pc)):
            for m in METRICS:
                arrays[f"{name}__{kind}__{m}"] = np.asarray(pa[m])
        entry = {"provenance": prov, "nested": ctx.summary(pn), "control": ctx.summary(pc),
                 "either": {"nested": either_point(pn["r1"] + pn["other"], cl),
                            "control": either_point(pc["r1"] + pc["other"], cl),
                            "cosine": either_point(cos_pa["r1"] + cos_pa["other"], cl)},
                 "picks": picks, "vs_control": paired(pn, pc, cl), "vs_cosine": paired(pn, cos_pa, cl),
                 "vs_rca": paired(pn, rca, cl)}
        if name == "A3":
            vr, vg = entry["vs_control"]["r1"], entry["vs_control"]["gain"]
            m_r, se_r = vr["point"], se_from_ci(vr["ci95"])
            m_g, se_g = vg["point"], se_from_ci(vg["ci95"])
            power = {"r1": predicted_power(m_r, se_r), "gain": predicted_power(m_g, se_g)}
            power["joint_if_independent"] = power["r1"] * power["gain"]
            entry.update(m_R=m_r, SE_R=se_r, m_g=m_g, SE_g=se_g, reading=margin_reading(m_r, se_r, m_g, se_g),
                         g_star=ceiling_threshold(se_r, se_g), predicted_power=power)
            prof = {"cells": {}, "control_r1": {}}
            for u, a in nested_cells():
                pa = per_anchor(nested_scores(ctx.cos, tu, ta, u, a))
                prof["cells"][f"{u:g},{a:g}"] = {"r1": 100 * float(pa["r1"].mean()),
                                                 "gain": 100 * float(pa["gain"].mean()),
                                                 "either": 100 * float((pa["r1"] + pa["other"]).mean())}
            for s in control_sums():
                prof["control_r1"][f"{s:g}"] = 100 * float(per_anchor(control_scores(ctx.cos, tu, s))["r1"].mean())
            assert len(prof["cells"]) == 56 and len(prof["control_r1"]) == 30
            result["profile_A3"] = prof
        result[name] = entry
        print(f"[{name}] done at {time.time() - t0:.0f}s", flush=True)

    result["A3_minus_C0_nested"] = paired(pas["A3"], pas["C0"], cl)
    rg.assert_finite_tree(result)
    out_json.write_text(json.dumps(result, indent=1))
    np.savez_compressed(res / "per_anchor_pilot_seed42.npz", **arrays)

    lines = [f"H1 pilot, seed {SEED}, n={ctx.n} episodes{' (SMOKE)' if smoke else ''}",
             f"{'model':5s} {'nested R@1':>10s} {'control R@1':>11s} {'m_R':>7s} {'nested gain':>11s} {'m_g':>7s}"]
    for name in MODELS:
        e = result[name]
        lines.append(f"{name:5s} {e['nested']['r1']['point']:10.2f} {e['control']['r1']['point']:11.2f} "
                     f"{e['vs_control']['r1']['point']:7.2f} {e['nested']['gain']['point']:11.2f} "
                     f"{e['vs_control']['gain']['point']:7.2f}")
    a = result["A3"]
    lines.append(f"A3 reading: {a['reading']} (m_R={a['m_R']:.3f} SE_R={a['SE_R']:.3f}, m_g={a['m_g']:.3f} "
                 f"SE_g={a['SE_g']:.3f}); g*={a['g_star']:.3f}; predicted power r1={a['predicted_power']['r1']:.3f} "
                 f"gain={a['predicted_power']['gain']:.3f} joint={a['predicted_power']['joint_if_independent']:.3f}")
    text = "\n".join(lines)
    (res / "pilot_seed42.txt").write_text(text + "\n")
    print(text)
    print(f"runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
